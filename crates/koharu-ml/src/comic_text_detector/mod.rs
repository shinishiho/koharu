mod onnx;
mod postprocess;

use std::cmp;

use anyhow::{Context, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::object_detection::{Bbox, non_maximum_suppression};
use image::{DynamicImage, GenericImageView, GrayImage};
use koharu_runtime::RuntimeManager;
use tracing::instrument;

use crate::types::TextRegion;

pub use postprocess::{
    ComicTextDetection, Quad, crop_text_block_bbox, expanded_text_block_crop_bounds,
    extract_text_block_regions, refine_segmentation_mask,
};

const CONFIDENCE_THRESHOLD: f32 = 0.4;
const NMS_THRESHOLD: f32 = 0.35;
const DBNET_BINARIZE_K: f64 = 50.0;
const BINARY_THRESHOLD: u8 = 60;
const DILATION_RADIUS: u32 = 3;
const HOLE_CLOSE_RADIUS: u32 = 10;
const BBOX_DILATION: f32 = 1.0;

pub struct ComicTextDetector {
    model: onnx::OnnxDetector,
}

impl ComicTextDetector {
    /// One graph covers detection and segmentation both, so there is no
    /// segmentation-only variant to load.
    pub async fn load(runtime: &RuntimeManager, cpu: bool) -> anyhow::Result<Self> {
        Ok(Self {
            model: onnx::OnnxDetector::load(runtime, cpu).await?,
        })
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference(&self, image: &DynamicImage) -> anyhow::Result<ComicTextDetection> {
        let original_dimensions = image.dimensions();
        let (image_tensor, resized_dimensions) = self.preprocess(image)?;
        let (predictions, mask, shrink_threshold) = self.forward(&image_tensor)?;

        let bboxes = postprocess_yolo(&predictions, original_dimensions, resized_dimensions)?;
        let shrink_map = tensor_channel_to_gray_resized(
            &shrink_threshold.i((0, 0))?,
            original_dimensions.0,
            original_dimensions.1,
        )?;
        let threshold_map = tensor_channel_to_gray_resized(
            &shrink_threshold.i((0, 1))?,
            original_dimensions.0,
            original_dimensions.1,
        )?;
        let mask = postprocess_mask(
            &mask,
            &shrink_threshold,
            original_dimensions,
            resized_dimensions,
        )?;

        Ok(ComicTextDetection {
            shrink_map,
            threshold_map,
            line_polygons: Vec::new(),
            text_blocks: bboxes_to_text_blocks(bboxes),
            mask,
        })
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference_segmentation(&self, image: &DynamicImage) -> anyhow::Result<GrayImage> {
        let original_dimensions = image.dimensions();
        let (image_tensor, resized_dimensions) = self.preprocess(image)?;
        let mask = self.forward_mask(&image_tensor)?;
        postprocess_unet_mask(&mask, original_dimensions, resized_dimensions)
    }

    /// Letterbox into the graph's fixed 1024² input.
    fn preprocess(&self, image: &DynamicImage) -> anyhow::Result<(Tensor, (u32, u32))> {
        preprocess(image, &Device::Cpu, DType::F32, onnx::INPUT_SIZE)
    }

    #[instrument(level = "debug", skip_all)]
    fn forward(&self, image: &Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        self.model.forward(image)
    }

    /// One graph, so the mask comes out of the same pass as everything else —
    /// the other two outputs are simply dropped.
    #[instrument(level = "debug", skip_all)]
    fn forward_mask(&self, image: &Tensor) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(image)?.1)
    }
}

#[instrument(level = "debug", skip_all)]
fn preprocess(
    image: &DynamicImage,
    device: &Device,
    dtype: DType,
    image_size: u32,
) -> anyhow::Result<(Tensor, (u32, u32))> {
    let (orig_w, orig_h) = image.dimensions();
    let (width, height) = if orig_w >= orig_h {
        (image_size, image_size * orig_h / orig_w)
    } else {
        (image_size * orig_w / orig_h, image_size)
    };
    let (w, h) = (width as usize, height as usize);
    let tensor = (Tensor::from_vec(
        image.to_rgb8().into_raw(),
        (1, orig_h as usize, orig_w as usize, 3),
        device,
    )?
    .permute((0, 3, 1, 2))?
    .to_dtype(DType::F32)?
    .interpolate2d(h, w)?
    .pad_with_zeros(2, 0, image_size as usize - h)?
    .pad_with_zeros(3, 0, image_size as usize - w)?
        * (1. / 255.))?;

    Ok((tensor.to_dtype(dtype)?, (width, height)))
}

#[instrument(level = "debug", skip(predictions))]
fn postprocess_yolo(
    predictions: &Tensor,
    original_dimensions: (u32, u32),
    resized_dimensions: (u32, u32),
) -> anyhow::Result<Vec<Bbox<usize>>> {
    let predictions = predictions.squeeze(0)?.to_dtype(DType::F32)?;
    let (_, num_outputs) = predictions.dims2()?;
    if num_outputs < 6 {
        bail!("invalid prediction shape: expected at least 6 outputs, got {num_outputs}");
    }

    let num_classes = num_outputs - 5;
    let (orig_w, orig_h) = original_dimensions;
    let (resized_w, resized_h) = resized_dimensions;
    let w_ratio = orig_w as f32 / resized_w as f32;
    let h_ratio = orig_h as f32 / resized_h as f32;

    let mut bboxes: Vec<Vec<Bbox<usize>>> = (0..num_classes).map(|_| Vec::new()).collect();
    let predictions: Vec<Vec<f32>> = predictions.to_vec2()?;
    for pred in predictions {
        let (class_index, confidence) = {
            let (cls_idx, cls_score) = pred[5..]
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, 0.0));
            (cls_idx, pred[4] * cls_score)
        };
        if confidence < CONFIDENCE_THRESHOLD {
            continue;
        }

        let xmin = ((pred[0] - pred[2] / 2.) * w_ratio - BBOX_DILATION).clamp(0., orig_w as f32);
        let xmax = ((pred[0] + pred[2] / 2.) * w_ratio + BBOX_DILATION).clamp(0., orig_w as f32);
        let ymin = ((pred[1] - pred[3] / 2.) * h_ratio - BBOX_DILATION).clamp(0., orig_h as f32);
        let ymax = ((pred[1] + pred[3] / 2.) * h_ratio + BBOX_DILATION).clamp(0., orig_h as f32);

        bboxes[class_index].push(Bbox {
            xmin,
            xmax,
            ymin,
            ymax,
            confidence,
            data: class_index,
        });
    }

    non_maximum_suppression(&mut bboxes, NMS_THRESHOLD);
    Ok(bboxes.into_iter().flatten().collect())
}

#[instrument(level = "debug", skip(mask, shrink_thresh))]
fn postprocess_mask(
    mask: &Tensor,
    shrink_thresh: &Tensor,
    original_dimensions: (u32, u32),
    resized_dimensions: (u32, u32),
) -> anyhow::Result<GrayImage> {
    let shrink_and_thresh = shrink_thresh.squeeze(0)?;
    let shrink = shrink_and_thresh.i(0)?.to_dtype(DType::F32)?;
    let thresh = shrink_and_thresh.i(1)?.to_dtype(DType::F32)?;
    let unet_mask = mask.squeeze(0)?.to_dtype(DType::F32)?;

    let (_, h_db, w_db) = shrink_and_thresh.dims3()?;
    let (_, h_unet, w_unet) = unet_mask.dims3()?;
    let h = cmp::min(h_db, h_unet);
    let w = cmp::min(w_db, w_unet);

    let shrink = shrink.narrow(0, 0, h)?.narrow(1, 0, w)?;
    let thresh = thresh.narrow(0, 0, h)?.narrow(1, 0, w)?;
    let unet_mask = unet_mask.narrow(1, 0, h)?.narrow(2, 0, w)?.squeeze(0)?;

    let prob = candle_nn::ops::sigmoid(&((&shrink - &thresh)? * DBNET_BINARIZE_K)?)?;
    let fused = prob.maximum(&unet_mask)?;

    let (mask_h, mask_w) = fused.dims2()?;
    let valid_h = mask_h.min(resized_dimensions.1 as usize);
    let valid_w = mask_w.min(resized_dimensions.0 as usize);

    let fused = fused
        .narrow(0, 0, valid_h)?
        .narrow(1, 0, valid_w)?
        .unsqueeze(0)?
        .unsqueeze(0)?;

    let resized = fused.interpolate2d(
        original_dimensions.1 as usize,
        original_dimensions.0 as usize,
    )?;
    let threshold = BINARY_THRESHOLD as f32 / 255.0;
    let binary = resized.ge(threshold)?.to_dtype(DType::F32)?;

    let closed = morph_close(&binary, HOLE_CLOSE_RADIUS as usize)?;
    let dilated = dilate_tensor(&closed, DILATION_RADIUS as usize)?;
    let mask = dilated.squeeze(0)?.squeeze(0)?;

    let mask = (mask * 255.)?.to_dtype(DType::U8)?;
    let data: Vec<u8> = mask.flatten_all()?.to_vec1()?;
    GrayImage::from_raw(original_dimensions.0, original_dimensions.1, data)
        .context("failed to build mask image")
}

fn postprocess_unet_mask(
    mask: &Tensor,
    original_dimensions: (u32, u32),
    resized_dimensions: (u32, u32),
) -> anyhow::Result<GrayImage> {
    let unet_mask = mask.i((0, 0))?;
    let (mask_h, mask_w) = unet_mask.dims2()?;
    let valid_h = mask_h.min(resized_dimensions.1 as usize);
    let valid_w = mask_w.min(resized_dimensions.0 as usize);
    let unet_mask = unet_mask.narrow(0, 0, valid_h)?.narrow(1, 0, valid_w)?;
    tensor_channel_to_gray_resized(&unet_mask, original_dimensions.0, original_dimensions.1)
}

fn tensor_channel_to_gray_resized(
    tensor: &Tensor,
    width: u32,
    height: u32,
) -> anyhow::Result<GrayImage> {
    let (th, tw) = tensor.dims2()?;
    let values: Vec<f32> = tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let pixels: Vec<u8> = values
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let small = GrayImage::from_raw(tw as u32, th as u32, pixels)
        .context("failed to create gray image from tensor")?;
    if tw as u32 == width && th as u32 == height {
        return Ok(small);
    }
    Ok(image::imageops::resize(
        &small,
        width,
        height,
        image::imageops::FilterType::Nearest,
    ))
}

fn bboxes_to_text_blocks(mut bboxes: Vec<Bbox<usize>>) -> Vec<TextRegion> {
    bboxes.sort_unstable_by(|a, b| {
        let ay = a.ymin + (a.ymax - a.ymin) * 0.5;
        let by = b.ymin + (b.ymax - b.ymin) * 0.5;
        ay.partial_cmp(&by).unwrap_or(std::cmp::Ordering::Equal)
    });

    bboxes
        .into_iter()
        .map(|bbox| TextRegion {
            x: bbox.xmin,
            y: bbox.ymin,
            width: bbox.xmax - bbox.xmin,
            height: bbox.ymax - bbox.ymin,
            confidence: bbox.confidence,
            ..Default::default()
        })
        .collect()
}

fn dilate_tensor(mask: &Tensor, radius: usize) -> anyhow::Result<Tensor> {
    let kernel = 2 * radius + 1;
    let padded = mask
        .pad_with_zeros(2, radius, radius)?
        .pad_with_zeros(3, radius, radius)?;
    Ok(padded.max_pool2d_with_stride((kernel, kernel), (1, 1))?)
}

fn erode_tensor(mask: &Tensor, radius: usize) -> anyhow::Result<Tensor> {
    let inverted = (1.0 - mask)?;
    let dilated = dilate_tensor(&inverted, radius)?;
    Ok((1.0 - dilated)?)
}

fn morph_close(mask: &Tensor, radius: usize) -> anyhow::Result<Tensor> {
    let dilated = dilate_tensor(mask, radius)?;
    erode_tensor(&dilated, radius)
}

/// One graph covers detection and segmentation both, so both entry points
/// fetch the same file.
pub async fn prefetch(runtime: &RuntimeManager) -> anyhow::Result<()> {
    onnx::prefetch(runtime).await
}

pub async fn prefetch_segmentation(runtime: &RuntimeManager) -> anyhow::Result<()> {
    prefetch(runtime).await
}
