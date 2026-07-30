mod onnx;

use std::{path::Path, time::Instant};

use anyhow::{Result, bail};
use candle_core::{IndexOp, Tensor};
use candle_transformers::object_detection::{Bbox, non_maximum_suppression};
use image::{
    DynamicImage, Rgb, RgbImage,
    imageops::{self, FilterType},
};
use koharu_runtime::RuntimeManager;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::types::TextRegion;
const INPUT_SIZE: u32 = 640;
const NUM_CLASSES: usize = 1;
const DEFAULT_VARIANT: AnimeTextYoloVariant = AnimeTextYoloVariant::N;
/// Fallback only. The ONNX export ships its own per-variant threshold; see
/// [`AnimeTextDetector::recommended_confidence`].
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.25;
const DEFAULT_NMS_THRESHOLD: f32 = 0.45;
const LETTERBOX_COLOR: u8 = 114;
const DETECTOR_NAME: &str = "anime-text-yolo";
const CLASS_NAMES: [&str; NUM_CLASSES] = ["text_block"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnimeTextYoloVariant {
    N,
    S,
    M,
    L,
    X,
}

impl AnimeTextYoloVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::N => "n",
            Self::S => "s",
            Self::M => "m",
            Self::L => "l",
            Self::X => "x",
        }
    }
}

impl std::fmt::Display for AnimeTextYoloVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct AnimeTextDetector {
    model: onnx::OnnxDetector,
    variant: AnimeTextYoloVariant,
}

/// Where the letterboxed image sits inside the square input, so detections can
/// be mapped back to source pixels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Letterbox {
    original_width: u32,
    original_height: u32,
    pad_x: u32,
    pad_y: u32,
    scale: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimeTextDetection {
    pub image_width: u32,
    pub image_height: u32,
    pub variant: AnimeTextYoloVariant,
    pub regions: Vec<AnimeTextRegion>,
    pub text_blocks: Vec<TextRegion>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimeTextRegion {
    pub label_id: usize,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
}

impl AnimeTextDetector {
    pub async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        Self::load_variant(runtime, DEFAULT_VARIANT, cpu).await
    }

    /// The export carries no NMS (`nms: False`), so decode and suppression
    /// live in this module; only the forward pass is ONNX's.
    pub async fn load_variant(
        runtime: &RuntimeManager,
        variant: AnimeTextYoloVariant,
        cpu: bool,
    ) -> Result<Self> {
        Ok(Self {
            model: onnx::OnnxDetector::load(runtime, variant, cpu).await?,
            variant,
        })
    }

    pub fn load_from_path(
        path: impl AsRef<Path>,
        variant: AnimeTextYoloVariant,
        cpu: bool,
    ) -> Result<Self> {
        Ok(Self {
            model: onnx::OnnxDetector::load_from_path(path.as_ref(), cpu)?,
            variant,
        })
    }

    pub fn variant(&self) -> AnimeTextYoloVariant {
        self.variant
    }

    /// The confidence threshold upstream measured as F1-optimal for these
    /// weights. deepghs pins one next to each export; a model loaded straight
    /// from a path has none, since the file sits beside it in the repo.
    pub fn recommended_confidence(&self) -> Option<f32> {
        self.model.recommended_confidence()
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference(&self, image: &DynamicImage) -> Result<AnimeTextDetection> {
        self.inference_with_thresholds(
            image,
            self.recommended_confidence()
                .unwrap_or(DEFAULT_CONFIDENCE_THRESHOLD),
            DEFAULT_NMS_THRESHOLD,
        )
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference_with_thresholds(
        &self,
        image: &DynamicImage,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<AnimeTextDetection> {
        let started = Instant::now();
        let letterboxed = letterbox(image);
        let predictions = self.model.forward(&letterboxed)?;
        let regions = postprocess(
            &predictions,
            &letterboxed.letterbox,
            confidence_threshold,
            nms_threshold,
        )?;
        let text_blocks = regions_to_text_blocks(&regions);

        tracing::info!(
            width = image.width(),
            height = image.height(),
            variant = %self.variant,
            detections = regions.len(),
            total_ms = started.elapsed().as_millis(),
            "anime text YOLO timings"
        );

        Ok(AnimeTextDetection {
            image_width: letterboxed.letterbox.original_width,
            image_height: letterboxed.letterbox.original_height,
            variant: self.variant,
            regions,
            text_blocks,
        })
    }
}

/// The letterboxed square input plus the geometry needed to undo it.
pub(crate) struct Letterboxed {
    image: RgbImage,
    letterbox: Letterbox,
}

fn letterbox(image: &DynamicImage) -> Letterboxed {
    let rgb = image.to_rgb8();
    let (original_width, original_height) = rgb.dimensions();
    let scale = f32::min(
        INPUT_SIZE as f32 / original_width.max(1) as f32,
        INPUT_SIZE as f32 / original_height.max(1) as f32,
    );
    let resized_width = ((original_width as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
    let resized_height = ((original_height as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
    let pad_x = (INPUT_SIZE - resized_width) / 2;
    let pad_y = (INPUT_SIZE - resized_height) / 2;

    let resized = if resized_width == original_width && resized_height == original_height {
        rgb
    } else {
        imageops::resize(&rgb, resized_width, resized_height, FilterType::Triangle)
    };

    let mut canvas = RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, Rgb([LETTERBOX_COLOR; 3]));
    imageops::overlay(&mut canvas, &resized, i64::from(pad_x), i64::from(pad_y));

    Letterboxed {
        image: canvas,
        letterbox: Letterbox {
            original_width,
            original_height,
            pad_x,
            pad_y,
            scale,
        },
    }
}

pub async fn prefetch(runtime: &RuntimeManager) -> Result<()> {
    prefetch_variant(runtime, DEFAULT_VARIANT).await
}

pub async fn prefetch_variant(
    runtime: &RuntimeManager,
    variant: AnimeTextYoloVariant,
) -> Result<()> {
    onnx::prefetch(runtime, variant).await
}

/// Decode a `(4 + NUM_CLASSES, anchors)` prediction plane into source-pixel
/// regions.
fn postprocess(
    pred: &Tensor,
    letterbox: &Letterbox,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<AnimeTextRegion>> {
    let (channels, anchors) = pred.dims2()?;
    let expected_channels = 4 + NUM_CLASSES;
    if channels != expected_channels {
        bail!(
            "unexpected anime text YOLO prediction channels {channels}, expected {expected_channels}"
        );
    }

    let mut grouped: Vec<Vec<Bbox<usize>>> = (0..NUM_CLASSES).map(|_| Vec::new()).collect();
    for anchor_idx in 0..anchors {
        let values = pred.i((.., anchor_idx))?.to_vec1::<f32>()?;
        let class_scores = &values[4..4 + NUM_CLASSES];
        let Some((label_id, &score)) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
        else {
            continue;
        };
        if score < confidence_threshold {
            continue;
        }

        let bbox = map_bbox_to_original(
            [
                values[0] - values[2] * 0.5,
                values[1] - values[3] * 0.5,
                values[0] + values[2] * 0.5,
                values[1] + values[3] * 0.5,
            ],
            letterbox,
        );
        if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            continue;
        }

        grouped[label_id].push(Bbox {
            xmin: bbox[0],
            ymin: bbox[1],
            xmax: bbox[2],
            ymax: bbox[3],
            confidence: score,
            data: label_id,
        });
    }

    non_maximum_suppression(&mut grouped, nms_threshold);

    let mut regions = Vec::new();
    for (label_id, bboxes) in grouped.into_iter().enumerate() {
        let label = CLASS_NAMES
            .get(label_id)
            .copied()
            .unwrap_or("text_block")
            .to_string();
        for bbox in bboxes {
            regions.push(AnimeTextRegion {
                label_id,
                label: label.clone(),
                score: bbox.confidence,
                bbox: [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
            });
        }
    }
    regions.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(regions)
}

fn map_bbox_to_original(bbox: [f32; 4], letterbox: &Letterbox) -> [f32; 4] {
    let width = letterbox.original_width as f32;
    let height = letterbox.original_height as f32;
    let pad_x = letterbox.pad_x as f32;
    let pad_y = letterbox.pad_y as f32;
    [
        ((bbox[0] - pad_x) / letterbox.scale).clamp(0.0, width),
        ((bbox[1] - pad_y) / letterbox.scale).clamp(0.0, height),
        ((bbox[2] - pad_x) / letterbox.scale).clamp(0.0, width),
        ((bbox[3] - pad_y) / letterbox.scale).clamp(0.0, height),
    ]
}

fn regions_to_text_blocks(regions: &[AnimeTextRegion]) -> Vec<TextRegion> {
    regions
        .iter()
        .filter_map(|region| {
            let width = (region.bbox[2] - region.bbox[0]).max(0.0);
            let height = (region.bbox[3] - region.bbox[1]).max(0.0);
            if width <= 1.0 || height <= 1.0 {
                return None;
            }
            Some(TextRegion {
                x: region.bbox[0],
                y: region.bbox[1],
                width,
                height,
                confidence: region.score,
                detector: Some(DETECTOR_NAME.to_string()),
                ..Default::default()
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Letterbox, map_bbox_to_original};

    #[test]
    fn map_bbox_to_original_removes_letterbox_padding() {
        let letterbox = Letterbox {
            original_width: 1000,
            original_height: 500,
            pad_x: 0,
            pad_y: 160,
            scale: 0.64,
        };

        let bbox = map_bbox_to_original([100.0, 200.0, 540.0, 440.0], &letterbox);
        assert!((bbox[0] - 156.25).abs() < 1e-3);
        assert!((bbox[1] - 62.5).abs() < 1e-3);
        assert!((bbox[2] - 843.75).abs() < 1e-3);
        assert!((bbox[3] - 437.5).abs() < 1e-3);
    }
}
