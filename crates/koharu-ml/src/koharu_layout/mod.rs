//! koharu-layout: RF-DETR-Seg 2XL @ 1152, instance segmentation over
//! `text` / `onomatopoeia` / `bubble` / `panel`.
//!
//! ONNX only — there is no candle port. The graph is a plain deploy export:
//! one NCHW image in, three tensors out (`dets`, `labels`, `masks`), all
//! decoding done here.
//!
//! Preprocessing, class names and per-class thresholds come from the
//! `onnx_config.json` that ships beside the graph, so re-exporting the model
//! with different normalization cannot silently desync the Rust side.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgb, Rgb32FImage, imageops::FilterType};
use koharu_runtime::RuntimeManager;
use ort::value::{Shape, Tensor};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::onnx::{OnnxSession, blank_image_input, ort_err};

const HF_REPO: &str = "ShiniShiho/koharu-layout-rfdetr-seg-2xl-1152-onnx";
const ONNX_FILENAME: &str = "rfdetr-seg-2xlarge.onnx";
const CONFIG_FILENAME: &str = "onnx_config.json";

koharu_runtime::declare_hf_model_package!(
    id: "model:koharu-layout:onnx",
    repo: "ShiniShiho/koharu-layout-rfdetr-seg-2xl-1152-onnx",
    file: "rfdetr-seg-2xlarge.onnx",
    bootstrap: false,
    order: 124,
);

// ---------------------------------------------------------------------------
// Export config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct LayoutOnnxConfig {
    pub input: LayoutInputConfig,
    pub classes: BTreeMap<usize, String>,
    pub recommended_thresholds: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayoutInputConfig {
    pub name: String,
    /// `[batch, channels, height, width]`.
    pub shape: [usize; 4],
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl LayoutOnnxConfig {
    fn width(&self) -> usize {
        self.input.shape[3]
    }

    fn height(&self) -> usize {
        self.input.shape[2]
    }

    /// Per-class score floor. Classes absent from `recommended_thresholds`
    /// fall back to the caller's threshold.
    fn threshold(&self, label: &str, fallback: f32) -> f32 {
        self.recommended_thresholds
            .get(label)
            .copied()
            .unwrap_or(fallback)
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// One detected layout instance. `bbox` is xyxy in original page pixels;
/// `mask` is a page-sized binary mask (0 or 255) when masks were requested.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayoutRegion {
    pub label_id: usize,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
    #[serde(skip)]
    pub mask: Option<GrayImage>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayoutDetection {
    pub image_width: u32,
    pub image_height: u32,
    pub regions: Vec<LayoutRegion>,
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

pub struct KoharuLayoutDetector {
    session: OnnxSession,
    config: LayoutOnnxConfig,
}

impl KoharuLayoutDetector {
    pub async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        let downloads = runtime.downloads();
        let config_path = downloads
            .huggingface_model(HF_REPO, CONFIG_FILENAME)
            .await?;
        let config: LayoutOnnxConfig =
            serde_json::from_slice(&std::fs::read(&config_path)?).with_context(|| {
                format!("failed to parse {}", config_path.display())
            })?;
        let model_path = downloads.huggingface_model(HF_REPO, ONNX_FILENAME).await?;

        let input_name = config.input.name.clone();
        let (width, height) = (config.width(), config.height());
        let session = OnnxSession::open(&model_path, cpu, move |session| {
            let image = blank_image_input(width, height)?;
            session
                .run(ort::inputs![input_name.as_str() => image])
                .map_err(ort_err)?;
            Ok(())
        })?;

        Ok(Self { session, config })
    }

    pub fn config(&self) -> &LayoutOnnxConfig {
        &self.config
    }

    /// Detect with the export's recommended per-class thresholds.
    pub fn inference(&self, image: &DynamicImage, masks: bool) -> Result<LayoutDetection> {
        self.inference_with_threshold(image, 0.0, masks)
    }

    /// `fallback_threshold` applies only to classes the export does not list a
    /// recommended threshold for.
    #[instrument(level = "debug", skip_all)]
    pub fn inference_with_threshold(
        &self,
        image: &DynamicImage,
        fallback_threshold: f32,
        masks: bool,
    ) -> Result<LayoutDetection> {
        let (page_width, page_height) = image.dimensions();
        let input = self.preprocess(image)?;

        let regions = self.session.run(
            ort::inputs![self.config.input.name.as_str() => input],
            |outputs| {
                let (dets_shape, dets) = outputs["dets"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)?;
                let (labels_shape, labels) = outputs["labels"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)?;

                let queries = *dets_shape
                    .get(1)
                    .context("dets output missing query dimension")?
                    as usize;
                let channels = *labels_shape
                    .get(2)
                    .context("labels output missing class dimension")?
                    as usize;
                if dets.len() != queries * 4 {
                    bail!("unexpected dets shape {dets_shape:?}");
                }
                if labels.len() != queries * channels {
                    bail!("unexpected labels shape {labels_shape:?}");
                }
                // The last logit channel is the no-object slot; the export's
                // own notes say to ignore it.
                let num_classes = channels
                    .checked_sub(1)
                    .context("labels output has no class channels")?;
                if num_classes > self.config.classes.len() {
                    bail!(
                        "graph has {num_classes} classes but the config names {}",
                        self.config.classes.len()
                    );
                }

                let mask_maps = if masks {
                    Some(outputs["masks"].try_extract_tensor::<f32>().map_err(ort_err)?)
                } else {
                    None
                };

                let mut regions = Vec::new();
                for query in 0..queries {
                    // Sigmoid focal-loss head: classes are scored independently,
                    // so one query can legitimately clear two class thresholds.
                    // Both are kept — cross-class nesting is resolved by the
                    // fusion stage, not here.
                    for class in 0..num_classes {
                        let label = self
                            .config
                            .classes
                            .get(&class)
                            .with_context(|| format!("config names no class {class}"))?;
                        let score = sigmoid(labels[query * channels + class]);
                        if score < self.config.threshold(label, fallback_threshold) {
                            continue;
                        }

                        let bbox = scale_box(
                            &dets[query * 4..query * 4 + 4],
                            page_width as f32,
                            page_height as f32,
                        );
                        let mask = match mask_maps {
                            Some((shape, data)) => Some(decode_mask(
                                shape, data, query, page_width, page_height,
                            )?),
                            None => None,
                        };
                        regions.push(LayoutRegion {
                            label_id: class,
                            label: label.clone(),
                            score,
                            bbox,
                            mask,
                        });
                    }
                }
                regions.sort_by(|a, b| b.score.total_cmp(&a.score));
                Ok(regions)
            },
        )?;

        tracing::debug!(
            width = page_width,
            height = page_height,
            regions = regions.len(),
            "koharu layout detection"
        );

        Ok(LayoutDetection {
            image_width: page_width,
            image_height: page_height,
            regions,
        })
    }

    /// Resize to the graph's fixed square input, then scale and normalize into
    /// CHW. The export takes a plain resize, not a letterbox, so normalized
    /// output coordinates map straight back onto the original page.
    fn preprocess(&self, image: &DynamicImage) -> Result<Tensor<f32>> {
        let (width, height) = (self.config.width(), self.config.height());
        let LayoutInputConfig {
            scale, mean, std, ..
        } = self.config.input;

        // Scale to float *before* resizing, matching the reference transform's
        // ToTensor -> resize -> normalize order. Resizing in u8 instead
        // requantizes every interpolated pixel and shifts borderline scores by
        // enough to flip detections near the low SFX threshold.
        let source = image.to_rgb8();
        let scaled = Rgb32FImage::from_fn(source.width(), source.height(), |x, y| {
            let pixel = source.get_pixel(x, y).0;
            Rgb([
                pixel[0] as f32 * scale,
                pixel[1] as f32 * scale,
                pixel[2] as f32 * scale,
            ])
        });
        let resized = image::imageops::resize(
            &scaled,
            width as u32,
            height as u32,
            FilterType::Triangle,
        );

        let pixel_count = width * height;
        let mut pixels = vec![0f32; pixel_count * 3];
        for (index, pixel) in resized.pixels().enumerate() {
            for channel in 0..3 {
                pixels[channel * pixel_count + index] =
                    (pixel.0[channel] - mean[channel]) / std[channel];
            }
        }

        Tensor::from_array((vec![1i64, 3, height as i64, width as i64], pixels)).map_err(ort_err)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Normalized `cx, cy, w, h` -> pixel `x1, y1, x2, y2`, clamped to the page.
fn scale_box(raw: &[f32], width: f32, height: f32) -> [f32; 4] {
    let (cx, cy, w, h) = (raw[0], raw[1], raw[2], raw[3]);
    [
        ((cx - w / 2.0) * width).clamp(0.0, width),
        ((cy - h / 2.0) * height).clamp(0.0, height),
        ((cx + w / 2.0) * width).clamp(0.0, width),
        ((cy + h / 2.0) * height).clamp(0.0, height),
    ]
}

/// One query's mask logits -> page-sized binary mask.
///
/// The logits are mapped through sigmoid to u8 before resizing so that the
/// `logit >= 0` test becomes `>= 128`; interpolating probabilities rather than
/// logits moves the boundary by a sub-pixel amount and keeps the resize in
/// `image`'s integer path.
fn decode_mask(
    shape: &Shape,
    data: &[f32],
    query: usize,
    page_width: u32,
    page_height: u32,
) -> Result<GrayImage> {
    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    let (mask_h, mask_w) = match dims.as_slice() {
        [_, _, h, w] => (*h, *w),
        _ => bail!("unexpected masks shape {shape:?}"),
    };
    let stride = mask_h * mask_w;
    let offset = query * stride;
    let logits = data
        .get(offset..offset + stride)
        .context("masks output shorter than its declared shape")?;

    let small = GrayImage::from_fn(mask_w as u32, mask_h as u32, |x, y| {
        Luma([(sigmoid(logits[y as usize * mask_w + x as usize]) * 255.0).round() as u8])
    });
    let resized = image::imageops::resize(&small, page_width, page_height, FilterType::Triangle);
    Ok(GrayImage::from_fn(page_width, page_height, |x, y| {
        Luma([if resized.get_pixel(x, y).0[0] >= 128 { 255 } else { 0 }])
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_box_converts_center_form_and_clamps() {
        // Centered half-size box on a 100x200 page.
        assert_eq!(scale_box(&[0.5, 0.5, 0.5, 0.5], 100.0, 200.0), [25.0, 50.0, 75.0, 150.0]);
        // Box running off the top-left is clipped, not negative.
        assert_eq!(scale_box(&[0.1, 0.1, 0.5, 0.5], 100.0, 200.0), [0.0, 0.0, 35.0, 70.0]);
    }

    #[test]
    fn threshold_prefers_recommended_over_fallback() {
        let config: LayoutOnnxConfig = serde_json::from_str(
            r#"{
                "input": {"name": "input", "shape": [1, 3, 8, 8], "scale": 1.0,
                          "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
                "classes": {"0": "text", "1": "bubble"},
                "recommended_thresholds": {"text": 0.25}
            }"#,
        )
        .unwrap();
        assert_eq!(config.threshold("text", 0.9), 0.25);
        assert_eq!(config.threshold("bubble", 0.9), 0.9);
        assert_eq!((config.width(), config.height()), (8, 8));
    }

    #[test]
    fn mask_decode_thresholds_at_logit_zero() {
        // 2x2 logits: only the first cell is positive.
        let shape = Shape::new([1i64, 1, 2, 2]);
        let data = [4.0f32, -4.0, -4.0, -4.0];
        let mask = decode_mask(&shape, &data, 0, 2, 2).unwrap();
        assert_eq!(mask.get_pixel(0, 0).0[0], 255);
        assert_eq!(mask.get_pixel(1, 1).0[0], 0);
    }
}
