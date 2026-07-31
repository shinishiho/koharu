//! ONNX Runtime backend for comic-text-detector.
//!
//! dmMaze's export puts all three heads koharu ports by hand — YOLOv5 boxes,
//! the U-Net mask, DBNet's shrink/threshold pair — in one graph, and its output
//! convention is the one the postprocessing expects: boxes as `cxcywh` in input
//! pixels with objectness and class scores, and all three maps post-sigmoid in
//! [0, 1]. So this module owns only the forward pass; decode, fusion and
//! morphology stay in the parent module.
//!
//! One behavioural difference: the graph's input is fixed at 1024², so the CPU
//! graph is fixed at 1024², so CPU pays the full size too.

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use koharu_runtime::RuntimeManager;
use ort::{session::SessionOutputs, value::Tensor as OrtTensor};

use crate::onnx::{OnnxSession, blank_image_input, ort_err};

const HF_REPO: &str = "mayocream/comic-text-detector-onnx";
const MODEL_FILE: &str = "comic-text-detector.onnx";
/// The graph declares `images` as `[1, 3, 1024, 1024]` — not dynamic.
pub(super) const INPUT_SIZE: u32 = 1024;

koharu_runtime::declare_hf_model_package!(
    id: "model:comic-text-detector:onnx",
    repo: "mayocream/comic-text-detector-onnx",
    file: "comic-text-detector.onnx",
    bootstrap: false,
    order: 113,
);

/// Download the graph without building a session.
pub(super) async fn prefetch(runtime: &RuntimeManager) -> Result<()> {
    runtime
        .downloads()
        .huggingface_model(HF_REPO, MODEL_FILE)
        .await?;
    Ok(())
}

#[derive(Debug)]
pub(super) struct OnnxDetector {
    session: OnnxSession,
}

impl OnnxDetector {
    pub(super) async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        let path = runtime
            .downloads()
            .huggingface_model(HF_REPO, MODEL_FILE)
            .await?;
        let session = OnnxSession::open(&path, cpu, |session| {
            let images = blank_image_input(INPUT_SIZE as usize, INPUT_SIZE as usize)?;
            session
                .run(ort::inputs!["images" => images])
                .map_err(ort_err)?;
            Ok(())
        })?;

        Ok(Self { session })
    }

    /// One pass over a `[1, 3, 1024, 1024]` f32 image, returning
    /// `(boxes, mask, shrink_threshold)` shaped exactly as the postprocessing
    /// produce them: `[1, anchors, 5 + classes]`, `[1, 1, h, w]`, `[1, 2, h, w]`.
    pub(super) fn forward(&self, image: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, channels, height, width) = image.dims4()?;
        let values = image.flatten_all()?.to_vec1::<f32>()?;
        let images = OrtTensor::from_array((
            vec![batch as i64, channels as i64, height as i64, width as i64],
            values,
        ))
        .map_err(ort_err)?;

        self.session
            .run(ort::inputs!["images" => images], |outputs| {
                let boxes = extract(outputs, "blk")?;
                let mask = extract(outputs, "seg")?;
                let shrink_threshold = extract(outputs, "det")?;

                if boxes.rank() != 3 || boxes.dim(2)? < 6 {
                    bail!(
                        "unexpected comic-text-detector ONNX `blk` shape {:?}, expected [1, anchors, 5 + classes]",
                        boxes.shape()
                    );
                }
                if mask.rank() != 4 || shrink_threshold.rank() != 4 {
                    bail!(
                        "unexpected comic-text-detector ONNX map ranks: `seg` {:?}, `det` {:?}",
                        mask.shape(),
                        shrink_threshold.shape()
                    );
                }

                Ok((boxes, mask, shrink_threshold))
            })
    }
}

/// Copy one f32 output into a candle tensor, keeping its shape.
fn extract(outputs: &SessionOutputs<'_>, name: &str) -> Result<Tensor> {
    let value = outputs
        .get(name)
        .with_context(|| format!("comic-text-detector ONNX output `{name}` missing"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(ort_err)
        .with_context(|| format!("failed to read comic-text-detector ONNX output `{name}`"))?;
    let dims = shape.iter().map(|d| *d as usize).collect::<Vec<_>>();
    Ok(Tensor::from_slice(data, dims, &Device::Cpu)?)
}
