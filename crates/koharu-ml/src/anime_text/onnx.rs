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

#[derive(Debug)]
pub(super) struct OnnxDetector {
    session: OnnxSession,
}

impl OnnxDetector {
    pub(super) async fn load(
        runtime: &RuntimeManager,
        variant: AnimeTextYoloVariant,
        cpu: bool,
    ) -> Result<Self> {
        let filename = onnx_filename(variant);
        let path = runtime
            .downloads()
            .huggingface_model(HF_REPO, &filename)
            .await?;
        Self::load_from_path(&path, cpu)
    }

    pub(super) fn load_from_path(path: &Path, cpu: bool) -> Result<Self> {
        let session = OnnxSession::open(path, cpu, |session| {
            let images = blank_image_input(INPUT_SIZE as usize, INPUT_SIZE as usize)?;
            session
                .run(ort::inputs!["images" => images])
                .map_err(ort_err)?;
            Ok(())
        })?;

        Ok(Self { session })
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

fn onnx_filename(variant: AnimeTextYoloVariant) -> String {
    format!("yolo12{}_animetext/model.onnx", variant.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_filename_matches_the_repo_layout() {
        assert_eq!(
            onnx_filename(AnimeTextYoloVariant::X),
            "yolo12x_animetext/model.onnx"
        );
        assert_eq!(
            onnx_filename(AnimeTextYoloVariant::N),
            "yolo12n_animetext/model.onnx"
        );
    }
}
