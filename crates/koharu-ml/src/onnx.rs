//! Shared ONNX Runtime session plumbing.
//!
//! Every ONNX-backed model needs the same three things: an `ort::Error` that
//! can cross into `anyhow`, a session built with the platform's accelerated
//! execution provider, and a guarantee that the provider can actually execute
//! the graph. This module owns all three; models supply only their own warmup
//! inputs and their own pre/post-processing.

use std::{path::Path, sync::Mutex};

use anyhow::{Context, Result};
use ort::{
    session::{Session, SessionInputs, SessionOutputs, builder::GraphOptimizationLevel},
    value::Tensor,
};

/// `ort::Error` is neither `Send` nor `Sync` (it can carry a panic payload, and
/// builder errors hand back the builder), so it cannot cross into `anyhow`
/// directly.
pub fn ort_err<T>(err: ort::Error<T>) -> anyhow::Error {
    anyhow::anyhow!("onnxruntime: {err}")
}

/// A loaded ONNX graph, ready to run.
pub struct OnnxSession {
    // ponytail: `Session::run` needs `&mut self`, so inference is serialized.
    // Per-thread sessions only if a caller ever wants concurrent pages.
    session: Mutex<Session>,
}

impl OnnxSession {
    /// Load `path`, using the platform's accelerated execution provider unless
    /// `cpu` is set.
    ///
    /// `warmup` runs one forward pass with dummy inputs of the graph's expected
    /// shapes. Accelerated providers can claim nodes they cannot actually
    /// execute — CoreML swallows RT-DETR's deformable-attention `GridSample`
    /// subgraph and then fails at compute time — so an unusable provider is
    /// detected here and demoted to CPU at load instead of mid-pipeline.
    pub fn open(
        path: &Path,
        cpu: bool,
        warmup: impl Fn(&mut Session) -> Result<()>,
    ) -> Result<Self> {
        let session = if cpu {
            commit(path, false)?
        } else {
            match commit(path, true).and_then(|mut session| {
                warmup(&mut session)?;
                Ok(session)
            }) {
                Ok(session) => session,
                Err(err) => {
                    tracing::warn!(
                        "ONNX accelerated execution provider unusable for {}, falling back to CPU: {err:#}",
                        path.display()
                    );
                    commit(path, false)?
                }
            }
        };

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Run one forward pass. `extract` reads the outputs while the session lock
    /// is still held — `SessionOutputs` borrows the session, so it cannot
    /// outlive the guard.
    pub fn run<'i, 'v: 'i, const N: usize, R>(
        &self,
        inputs: impl Into<SessionInputs<'i, 'v, N>>,
        extract: impl FnOnce(&SessionOutputs<'_>) -> Result<R>,
    ) -> Result<R> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX session mutex poisoned"))?;
        let outputs = session.run(inputs).map_err(ort_err)?;
        extract(&outputs)
    }
}

fn commit(path: &Path, accelerated: bool) -> Result<Session> {
    #[allow(unused_mut)]
    let mut builder = Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?;

    if accelerated {
        #[cfg(target_os = "macos")]
        {
            builder = builder
                .with_execution_providers([ort::ep::coreml::CoreML::default().build()])
                .map_err(ort_err)?;
        }
        #[cfg(any(feature = "cuda", feature = "onnx-cuda"))]
        {
            builder = builder
                .with_execution_providers([ort::ep::cuda::CUDA::default().build()])
                .map_err(ort_err)?;
        }
    }

    builder
        .commit_from_file(path)
        .map_err(ort_err)
        .with_context(|| format!("failed to load {}", path.display()))
}

/// A zero-filled `[1, 3, height, width]` f32 image tensor, for warmup passes.
pub fn blank_image_input(width: usize, height: usize) -> Result<Tensor<f32>> {
    Tensor::from_array((
        vec![1i64, 3, height as i64, width as i64],
        vec![0f32; 3 * width * height],
    ))
    .map_err(ort_err)
}
