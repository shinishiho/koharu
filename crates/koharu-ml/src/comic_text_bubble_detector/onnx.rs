//! ONNX Runtime backend for the RT-DETRv2 comic text/bubble detector.
//!
//! `detector.onnx` is the upstream deploy export: it takes the resized image
//! plus the original slice size and returns already-decoded `labels`/`boxes`/
//! `scores` in original pixel coordinates, so the Rust side has no decode step.

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use koharu_runtime::RuntimeManager;
use ort::value::Tensor;

use super::{ComicTextBubbleRegion, HF_REPO, RTDetrImageProcessorConfig, RTDetrV2Config};
use crate::onnx::{OnnxSession, blank_image_input, ort_err};

const ONNX_FILENAME: &str = "detector.onnx";

koharu_runtime::declare_hf_model_package!(
    id: "model:comic-text-bubble-detector:onnx",
    repo: "ogkalu/comic-text-and-bubble-detector",
    file: "detector.onnx",
    bootstrap: false,
    order: 123,
);

pub(super) struct OnnxDetector {
    session: OnnxSession,
}

impl OnnxDetector {
    pub(super) async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        let path = runtime
            .downloads()
            .huggingface_model(HF_REPO, ONNX_FILENAME)
            .await?;
        Self::load_from_path(&path, cpu)
    }

    pub(super) fn load_from_path(path: &Path, cpu: bool) -> Result<Self> {
        let session = OnnxSession::open(path, cpu, |session| {
            let images = blank_image_input(640, 640)?;
            let sizes = Tensor::from_array((vec![1i64, 2], vec![640i64, 640])).map_err(ort_err)?;
            session
                .run(ort::inputs!["images" => images, "orig_target_sizes" => sizes])
                .map_err(ort_err)?;
            Ok(())
        })?;

        Ok(Self { session })
    }

    pub(super) fn detect(
        &self,
        image: &DynamicImage,
        threshold: f32,
        config: &RTDetrV2Config,
        preprocessor: &RTDetrImageProcessorConfig,
    ) -> Result<Vec<ComicTextBubbleRegion>> {
        let (original_width, original_height) = image.dimensions();
        let target_h = preprocessor.size.height;
        let target_w = preprocessor.size.width;
        let resized = image.resize_exact(target_w as u32, target_h as u32, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // HWC u8 -> CHW f32 in [0, 1]. `do_normalize` is false for this model.
        let pixel_count = target_w * target_h;
        let mut pixels = vec![0f32; pixel_count * 3];
        for (index, pixel) in rgb.pixels().enumerate() {
            pixels[index] = pixel.0[0] as f32 * preprocessor.rescale_factor;
            pixels[pixel_count + index] = pixel.0[1] as f32 * preprocessor.rescale_factor;
            pixels[2 * pixel_count + index] = pixel.0[2] as f32 * preprocessor.rescale_factor;
        }

        let images = Tensor::from_array((vec![1, 3, target_h as i64, target_w as i64], pixels))
            .map_err(ort_err)?;
        // Upstream postprocessing multiplies the normalized boxes by
        // `[w, h, w, h]`, so this is (width, height) — not (height, width).
        let sizes = Tensor::from_array((
            vec![1, 2],
            vec![original_width as i64, original_height as i64],
        ))
        .map_err(ort_err)?;

        self.session.run(
            ort::inputs![
                "images" => images,
                "orig_target_sizes" => sizes,
            ],
            |outputs| {
                let (_, labels) = outputs["labels"]
                    .try_extract_tensor::<i64>()
                    .map_err(ort_err)?;
                let (boxes_shape, boxes) = outputs["boxes"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)?;
                let (_, scores) = outputs["scores"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)?;
                if boxes.len() != labels.len() * 4 || scores.len() != labels.len() {
                    bail!(
                        "unexpected ONNX output shapes: labels={} boxes={:?} scores={}",
                        labels.len(),
                        boxes_shape,
                        scores.len()
                    );
                }

                let num_labels = config.num_labels();
                let mut detections = Vec::new();
                for (index, (&label, &score)) in labels.iter().zip(scores.iter()).enumerate() {
                    if score < threshold {
                        continue;
                    }
                    let label_id = usize::try_from(label)
                        .with_context(|| format!("negative ONNX label id {label}"))?;
                    if label_id >= num_labels {
                        bail!("ONNX label id {label_id} exceeds configured {num_labels} labels");
                    }
                    let bbox = &boxes[index * 4..index * 4 + 4];
                    detections.push(ComicTextBubbleRegion {
                        label_id,
                        label: config.label(label_id),
                        score,
                        bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
                    });
                }

                Ok(detections)
            },
        )
    }
}
