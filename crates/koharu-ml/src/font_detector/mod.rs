use std::{fs, path::PathBuf, time::Instant};

use crate::{device, loading};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use koharu_runtime::RuntimeManager;
use rayon::prelude::*;

mod models;
#[cfg(feature = "onnx")]
mod onnx;

pub use models::ModelKind;

pub(super) const FONT_COUNT: usize = 6_150;
const REGRESSION_START: usize = FONT_COUNT + 2;
pub(super) const REGRESSION_DIM: usize = 10;

const HF_REPO: &str = "fffonion/yuzumarker-font-detection";

koharu_runtime::declare_hf_model_package!(
    id: "model:font-detector:weights",
    repo: "fffonion/yuzumarker-font-detection",
    file: "yuzumarker-font-detection.safetensors",
    bootstrap: false,
    order: 140,
);
koharu_runtime::declare_hf_model_package!(
    id: "model:font-detector:labels",
    repo: "fffonion/yuzumarker-font-detection",
    file: "font-labels-ex.json",
    bootstrap: false,
    order: 141,
);

pub use crate::types::{FontPrediction, NamedFontPrediction, TextDirection, TopFont};

pub struct FontDetector {
    backend: Backend,
    labels: FontLabels,
}

enum Backend {
    Candle {
        model: models::Model,
        device: Device,
        dtype: DType,
    },
    #[cfg(feature = "onnx")]
    Onnx(onnx::OnnxFontDetector),
}

impl FontDetector {
    pub async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        Self::load_with_kind(runtime, cpu, ModelKind::default()).await
    }

    pub async fn load_with_kind(
        runtime: &RuntimeManager,
        cpu: bool,
        kind: ModelKind,
    ) -> Result<Self> {
        let device = device(cpu)?;
        let dtype = loading::model_dtype(&device);
        let downloads = runtime.downloads();
        let weights_path = downloads
            .huggingface_model(HF_REPO, "yuzumarker-font-detection.safetensors")
            .await?;
        let model = loading::load_mmaped_safetensors_path_with_dtype(
            &weights_path,
            &device,
            dtype,
            move |vb| models::Model::load(vb.pp("model._orig_mod.model"), kind),
        )?;
        let labels = FontLabels::load(runtime).await?;

        Ok(Self {
            backend: Backend::Candle {
                model,
                device,
                dtype,
            },
            labels,
        })
    }

    /// Same checkpoint, run through ONNX Runtime on ogkalu's export instead of
    /// the hand-ported candle backbone.
    ///
    /// Labels still come from the safetensors repo — the ONNX repo ships only
    /// the graph.
    #[cfg(feature = "onnx")]
    pub async fn load_onnx(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        Ok(Self {
            backend: Backend::Onnx(onnx::OnnxFontDetector::load(runtime, cpu).await?),
            labels: FontLabels::load(runtime).await?,
        })
    }

    fn input_size(&self) -> usize {
        match &self.backend {
            Backend::Candle { model, .. } => model.input_size(),
            #[cfg(feature = "onnx")]
            Backend::Onnx(_) => onnx::INPUT_SIZE,
        }
    }

    /// One row per image: font logits, two direction logits, then the
    /// regression block already squashed into [0, 1].
    ///
    /// The ONNX graph sigmoids the regression block itself, so the candle path
    /// does the same here rather than leaving it to the decoder — one
    /// convention, no per-backend branch downstream.
    fn forward(&self, batch: &Tensor) -> Result<Vec<Vec<f32>>> {
        match &self.backend {
            Backend::Candle {
                model,
                device,
                dtype,
            } => {
                let batch = batch.to_device(device)?.to_dtype(*dtype)?;
                let mut rows = model
                    .forward(&batch, false)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec2::<f32>()?;
                for row in &mut rows {
                    for value in &mut row[REGRESSION_START..REGRESSION_START + REGRESSION_DIM] {
                        *value = sigmoid_scalar(*value);
                    }
                }
                Ok(rows)
            }
            #[cfg(feature = "onnx")]
            Backend::Onnx(detector) => detector.forward(batch),
        }
    }

    pub fn inference(&self, images: &[DynamicImage], top_k: usize) -> Result<Vec<FontPrediction>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let started = Instant::now();
        let input_size = self.input_size();
        let original_sizes = images
            .iter()
            .map(|image| image.dimensions().0)
            .collect::<Vec<_>>();
        let preprocess_started = Instant::now();
        let processed = images
            .par_iter()
            .map(|image| preprocess_image(image, input_size))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let batch = Tensor::stack(&processed, 0)?;
        let preprocess_elapsed = preprocess_started.elapsed();

        let forward_started = Instant::now();
        let rows = self.forward(&batch)?;
        let forward_elapsed = forward_started.elapsed();

        let postprocess_started = Instant::now();
        let mut predictions = Vec::with_capacity(images.len());
        for (row, width) in rows.into_iter().zip(original_sizes) {
            let ranked: Vec<TopFont> = top_k_softmax(&row[..FONT_COUNT], top_k.min(FONT_COUNT))
                .into_iter()
                .map(|(index, score)| TopFont { index, score })
                .collect();

            let named_fonts = ranked
                .iter()
                .filter_map(|tf| {
                    self.labels
                        .entry(tf.index)
                        .map(|label| NamedFontPrediction {
                            index: tf.index,
                            name: label.name.clone(),
                            language: label.language.clone(),
                            probability: tf.score,
                            serif: label.serif,
                        })
                })
                .collect();

            let direction = if row[FONT_COUNT + 1] > row[FONT_COUNT] {
                TextDirection::Vertical
            } else {
                TextDirection::Horizontal
            };

            let regression = &row[REGRESSION_START..REGRESSION_START + REGRESSION_DIM];
            let clamp01 = |v: f32| v.clamp(0.0, 1.0);
            let text_color = [
                (clamp01(regression[0]) * 255.0).round() as u8,
                (clamp01(regression[1]) * 255.0).round() as u8,
                (clamp01(regression[2]) * 255.0).round() as u8,
            ];
            let font_size_px = clamp01(regression[3]) * width as f32;
            let stroke_width_px = clamp01(regression[4]) * width as f32;
            let stroke_color = [
                (clamp01(regression[5]) * 255.0).round() as u8,
                (clamp01(regression[6]) * 255.0).round() as u8,
                (clamp01(regression[7]) * 255.0).round() as u8,
            ];
            let line_spacing_px = clamp01(regression[8]) * width as f32;
            let line_height = if font_size_px > 0.0 {
                1.0 + line_spacing_px / font_size_px
            } else {
                1.2
            };
            let angle_deg = (regression[9] - 0.5) * 180.0;

            predictions.push(FontPrediction {
                top_fonts: ranked,
                named_fonts,
                direction,
                text_color,
                stroke_color,
                font_size_px,
                stroke_width_px,
                line_height,
                angle_deg,
            });
        }

        tracing::info!(
            images = images.len(),
            input_size,
            preprocess_ms = preprocess_elapsed.as_millis(),
            forward_ms = forward_elapsed.as_millis(),
            postprocess_ms = postprocess_started.elapsed().as_millis(),
            total_ms = started.elapsed().as_millis(),
            "font detector timings"
        );

        Ok(predictions)
    }
}

#[derive(Debug, Clone)]
pub struct FontLabel {
    pub name: String,
    pub language: Option<String>,
    pub serif: bool,
}

#[derive(Debug, Clone)]
pub struct FontLabels {
    labels: Vec<FontLabel>,
}

impl FontLabels {
    pub async fn load(runtime: &RuntimeManager) -> Result<Self> {
        let path = runtime
            .downloads()
            .huggingface_model(HF_REPO, "font-labels-ex.json")
            .await?;
        Self::from_path(&path)
    }

    pub fn from_path(path: &PathBuf) -> Result<Self> {
        let data = fs::read_to_string(path)
            .with_context(|| format!("Failed to read labels file {}", path.display()))?;
        let entries: Vec<FontLabelEntry> = serde_json::from_str(&data)
            .with_context(|| format!("Failed to parse labels file {}", path.display()))?;
        let mut labels = Vec::with_capacity(entries.len());
        for entry in entries {
            labels.push(FontLabel {
                name: entry.path,
                language: entry.language,
                serif: entry.serif,
            });
        }
        Ok(Self { labels })
    }

    pub fn entry(&self, idx: usize) -> Option<&FontLabel> {
        self.labels.get(idx)
    }

    pub fn name(&self, idx: usize) -> Option<&str> {
        self.entry(idx).map(|label| label.name.as_str())
    }

    pub fn language(&self, idx: usize) -> Option<&str> {
        self.entry(idx).and_then(|label| label.language.as_deref())
    }
}

#[derive(serde::Deserialize)]
struct FontLabelEntry {
    path: String,
    language: Option<String>,
    serif: bool,
}

fn preprocess_image(image: &DynamicImage, target: usize) -> Result<Tensor> {
    let resized = image.resize_exact(target as u32, target as u32, FilterType::CatmullRom);
    let data = resized.to_rgb8().into_raw();
    let tensor = Tensor::from_vec(
        data,
        (target, target, 3),
        &Device::Cpu,
    )?
    .to_dtype(DType::F32)?
    .permute((2, 0, 1))? // (3, H, W)
    * (1.0 / 255.0);
    tensor.map_err(Into::into)
}

fn top_k_softmax(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let top_k = top_k.min(logits.len());
    if top_k == 0 {
        return Vec::new();
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom = logits
        .iter()
        .map(|&logit| ((logit - max_logit) as f64).exp())
        .sum::<f64>()
        .max(f64::MIN_POSITIVE);

    let mut best = Vec::with_capacity(top_k);
    for (index, &logit) in logits.iter().enumerate() {
        insert_ranked(&mut best, (index, logit), top_k);
    }

    best.into_iter()
        .map(|(index, logit)| (index, (((logit - max_logit) as f64).exp() / denom) as f32))
        .collect()
}

fn insert_ranked(best: &mut Vec<(usize, f32)>, candidate: (usize, f32), limit: usize) {
    let position = best
        .iter()
        .position(|(_, value)| candidate.1 > *value)
        .unwrap_or(best.len());
    if position < limit {
        best.insert(position, candidate);
        if best.len() > limit {
            best.pop();
        }
    } else if best.len() < limit {
        best.push(candidate);
    }
}

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
