//! ONNX Runtime backend for the YuzuMarker font detector.
//!
//! ogkalu's export carries the same checkpoint koharu ships as safetensors —
//! `model.model.fc.weight` is bit-identical to `fffonion`'s — but runs
//! upstream's own graph. The hand-ported candle backbone drifts: on a constant
//! grey 512² input, where preprocessing cannot differ, the two disagree on
//! text colour by 4/255 and on rotation by 0.3°. Same weights, so this path is
//! the faithful one.
//!
//! The graph sigmoids its last ten outputs — the regression block — before
//! returning them, so those arrive in [0, 1] and the parent module must not
//! sigmoid them a second time.

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use koharu_runtime::RuntimeManager;
use ort::value::Tensor as OrtTensor;

use super::{FONT_COUNT, REGRESSION_DIM};
use crate::onnx::{OnnxSession, blank_image_input, ort_err};

pub(super) const INPUT_SIZE: usize = 512;
const HF_REPO: &str = "ogkalu/yuzumarker-font-detection-onnx";
const MODEL_FILE: &str = "font-detector.onnx";
/// Font logits, two direction logits, then the regression block.
const OUTPUT_WIDTH: usize = FONT_COUNT + 2 + REGRESSION_DIM;

koharu_runtime::declare_hf_model_package!(
    id: "model:font-detector:onnx",
    repo: "ogkalu/yuzumarker-font-detection-onnx",
    file: "font-detector.onnx",
    bootstrap: false,
    order: 142,
);

#[derive(Debug)]
pub(super) struct OnnxFontDetector {
    session: OnnxSession,
}

impl OnnxFontDetector {
    pub(super) async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        let path = runtime
            .downloads()
            .huggingface_model(HF_REPO, MODEL_FILE)
            .await?;
        let session = OnnxSession::open(&path, cpu, |session| {
            let input = blank_image_input(INPUT_SIZE, INPUT_SIZE)?;
            session
                .run(ort::inputs!["input" => input])
                .map_err(ort_err)?;
            Ok(())
        })?;

        Ok(Self { session })
    }

    /// One forward pass over a `(batch, 3, 512, 512)` f32 batch, returning one
    /// row of [`OUTPUT_WIDTH`] values per image.
    ///
    /// Takes the candle tensor the parent module already built so both backends
    /// share one preprocessing implementation — a resize kernel difference here
    /// would show up as a different font.
    pub(super) fn forward(&self, batch: &Tensor) -> Result<Vec<Vec<f32>>> {
        let (count, _, height, width) = batch.dims4()?;
        let values = batch.flatten_all()?.to_vec1::<f32>()?;
        let input =
            OrtTensor::from_array((vec![count as i64, 3, height as i64, width as i64], values))
                .map_err(ort_err)?;

        self.session.run(ort::inputs!["input" => input], |outputs| {
            let (shape, data) = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)
                .context("failed to read font detector ONNX output")?;
            let dims = shape.iter().map(|d| *d as usize).collect::<Vec<_>>();
            let [rows, columns] = dims[..] else {
                bail!("unexpected font detector ONNX output rank {dims:?}, expected 2 dims");
            };
            if columns != OUTPUT_WIDTH {
                bail!(
                    "unexpected font detector ONNX output width {columns}, expected {OUTPUT_WIDTH}"
                );
            }
            if rows != count {
                bail!("font detector ONNX returned {rows} rows for {count} images");
            }
            Ok(data.chunks_exact(columns).map(<[f32]>::to_vec).collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::OUTPUT_WIDTH;

    #[test]
    fn output_width_matches_the_export() {
        // The graph declares `output` as [batch, 6162]; koharu's slice offsets
        // are derived from FONT_COUNT, so a mismatch means one of them moved.
        assert_eq!(OUTPUT_WIDTH, 6162);
    }
}
