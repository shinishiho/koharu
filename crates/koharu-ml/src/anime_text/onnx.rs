//! ONNX Runtime backend for the AnimeText YOLO12 text-block detector.
//!
//! `model.onnx` is deepghs' own Ultralytics export of the same weights koharu
//! already ships as safetensors — verified box-identical (IoU 0.9999, score
//! delta 4e-4) against the candle path. It carries no NMS, so this module owns
//! only the forward pass; decode, suppression and letterbox-unmapping stay in
//! the parent module.
//!
//! The repo is gated. `Downloads` explains how to authenticate when the fetch
//! comes back 401, so nothing extra is needed here.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use koharu_runtime::RuntimeManager;
use ort::value::Tensor as OrtTensor;
use serde::Deserialize;

use super::{AnimeTextYoloVariant, INPUT_SIZE, Letterboxed};
use crate::onnx::{OnnxSession, blank_image_input, ort_err};

/// deepghs' export, one directory per variant. Gated (`gated: auto`): users
/// accept the terms once, then koharu needs a token. Kept separate from
/// `mayocream/anime-text-yolo` (ungated, safetensors) so the candle path keeps
/// working with no account at all.
const HF_REPO: &str = "deepghs/AnimeText_yolo";

koharu_runtime::declare_hf_model_package!(
    id: "model:anime-text-yolo:onnx-yolo12n",
    repo: "deepghs/AnimeText_yolo",
    file: "yolo12n_animetext/model.onnx",
    bootstrap: false,
    order: 125,
);
koharu_runtime::declare_hf_model_package!(
    id: "model:anime-text-yolo:onnx-yolo12s",
    repo: "deepghs/AnimeText_yolo",
    file: "yolo12s_animetext/model.onnx",
    bootstrap: false,
    order: 126,
);
koharu_runtime::declare_hf_model_package!(
    id: "model:anime-text-yolo:onnx-yolo12m",
    repo: "deepghs/AnimeText_yolo",
    file: "yolo12m_animetext/model.onnx",
    bootstrap: false,
    order: 127,
);
koharu_runtime::declare_hf_model_package!(
    id: "model:anime-text-yolo:onnx-yolo12l",
    repo: "deepghs/AnimeText_yolo",
    file: "yolo12l_animetext/model.onnx",
    bootstrap: false,
    order: 128,
);
koharu_runtime::declare_hf_model_package!(
    id: "model:anime-text-yolo:onnx-yolo12x",
    repo: "deepghs/AnimeText_yolo",
    file: "yolo12x_animetext/model.onnx",
    bootstrap: false,
    order: 129,
);

/// deepghs' F1-optimal confidence, pinned per export next to the weights.
#[derive(Deserialize)]
struct ThresholdFile {
    threshold: f32,
}

#[derive(Debug)]
pub(super) struct OnnxDetector {
    session: OnnxSession,
    recommended_confidence: Option<f32>,
}

impl OnnxDetector {
    pub(super) async fn load(
        runtime: &RuntimeManager,
        variant: AnimeTextYoloVariant,
        cpu: bool,
    ) -> Result<Self> {
        let directory = variant_directory(variant);
        let path = runtime
            .downloads()
            .huggingface_model(HF_REPO, &format!("{directory}/model.onnx"))
            .await?;
        let mut detector = Self::load_from_path(&path, cpu)?;
        detector.recommended_confidence = read_threshold(runtime, &directory).await;
        Ok(detector)
    }

    pub(super) fn load_from_path(path: &Path, cpu: bool) -> Result<Self> {
        let session = OnnxSession::open(path, cpu, |session| {
            let images = blank_image_input(INPUT_SIZE as usize, INPUT_SIZE as usize)?;
            session
                .run(ort::inputs!["images" => images])
                .map_err(ort_err)?;
            Ok(())
        })?;

        Ok(Self {
            session,
            recommended_confidence: None,
        })
    }

    pub(super) fn recommended_confidence(&self) -> Option<f32> {
        self.recommended_confidence
    }

    /// One forward pass, returning the `(4 + classes, anchors)` prediction
    /// plane the parent module decodes.
    pub(super) fn forward(&self, letterboxed: &Letterboxed) -> Result<Tensor> {
        let side = INPUT_SIZE as usize;
        let pixels = letterboxed.image.as_raw();
        let plane = side * side;
        let mut input = vec![0f32; plane * 3];
        for (index, chunk) in pixels.chunks_exact(3).enumerate() {
            for channel in 0..3 {
                input[channel * plane + index] = chunk[channel] as f32 / 255.0;
            }
        }
        let images = OrtTensor::from_array((vec![1i64, 3, side as i64, side as i64], input))
            .map_err(ort_err)?;

        self.session
            .run(ort::inputs!["images" => images], |outputs| {
                let (shape, data) = outputs["output0"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)
                    .context("failed to read AnimeText ONNX output")?;
                let dims = shape.iter().map(|d| *d as usize).collect::<Vec<_>>();
                let [batch, channels, anchors] = dims[..] else {
                    bail!("unexpected AnimeText ONNX output rank {dims:?}, expected 3 dims");
                };
                if batch != 1 {
                    bail!("unexpected AnimeText ONNX batch {batch}, expected 1");
                }
                Ok(Tensor::from_slice(data, (channels, anchors), &Device::Cpu)?)
            })
    }
}

fn variant_directory(variant: AnimeTextYoloVariant) -> String {
    format!("yolo12{}_animetext", variant.as_str())
}

/// Read the export's recommended confidence threshold.
///
/// Each variant peaks at a different confidence (0.251 for n, 0.425 for x), so
/// one hardcoded default is wrong for four of the five. Reading upstream's own
/// number also means a retrain can't silently desync koharu's default.
///
/// Not fatal when it fails: the model itself already downloaded, so access is
/// proven and a failure here means the file is gone or malformed. The caller
/// falls back to [`super::DEFAULT_CONFIDENCE_THRESHOLD`].
async fn read_threshold(runtime: &RuntimeManager, directory: &str) -> Option<f32> {
    let filename = format!("{directory}/threshold.json");
    let read = async {
        let path = runtime
            .downloads()
            .huggingface_model(HF_REPO, &filename)
            .await?;
        let bytes =
            std::fs::read(&path).with_context(|| format!("failed to read `{}`", path.display()))?;
        Ok::<f32, anyhow::Error>(serde_json::from_slice::<ThresholdFile>(&bytes)?.threshold)
    };

    match read.await {
        Ok(threshold) if (0.0..1.0).contains(&threshold) => {
            tracing::debug!(threshold, %filename, "using upstream confidence threshold");
            Some(threshold)
        }
        Ok(threshold) => {
            tracing::warn!(threshold, %filename, "confidence threshold out of range, ignoring");
            None
        }
        Err(error) => {
            tracing::warn!(%filename, error = %format!("{error:#}"), "no upstream confidence threshold, using koharu's default");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_directory_matches_the_repo_layout() {
        assert_eq!(
            variant_directory(AnimeTextYoloVariant::X),
            "yolo12x_animetext"
        );
        assert_eq!(
            variant_directory(AnimeTextYoloVariant::N),
            "yolo12n_animetext"
        );
    }

    #[test]
    fn threshold_file_parses_upstream_shape() {
        let parsed: ThresholdFile = serde_json::from_str(r#"{"threshold": 0.425}"#).unwrap();
        assert!((parsed.threshold - 0.425).abs() < 1e-6);
    }
}
