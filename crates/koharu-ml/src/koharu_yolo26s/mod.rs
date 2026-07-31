//! koharu-yolo26s: YOLO26-s instance segmentation over `frame` /
//! `dialogue_text` / `balloon` / `onomatopoeia_text` — the structural layout
//! role the fusion stage is built around.
//!
//! ONNX only. There is no candle port and no plan for one, so this module owns
//! preprocessing, the forward pass and decode outright — no `Backend` enum.
//!
//! The export is faithful: against the PyTorch checkpoint on byte-identical
//! input it matched 144/144 instances at confidence 0.25 and 166/166 at 0.10,
//! with box and mask IoU 1.000 and a worst confidence delta of 6e-05. So the
//! numbers below are all properties of the model or of the deployment, never of
//! the conversion.
//!
//! Two deployment choices here are measured, not guessed: the head emits some
//! instances twice even though it needs no NMS (see [`dedup`]), and the padding
//! convention changes results (see [`letterbox`]).

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use image::{
    DynamicImage, GrayImage, Luma, Rgb, RgbImage,
    imageops::{self, FilterType},
};
use koharu_runtime::RuntimeManager;
use ort::value::Tensor as OrtTensor;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::onnx::{OnnxSession, ort_err};

const HF_REPO: &str = "ShiniShiho/koharu-yolo26s-onnx";
const MODEL_FILE: &str = "koharu-yolo26s.onnx";
/// Long edge of the letterboxed input, as exported (`imgsz=1280`).
const INPUT_SIZE: u32 = 1280;
/// Shapes are dynamic, but only down to the backbone's stride.
const STRIDE: u32 = 32;
const LETTERBOX_COLOR: u8 = 114;
/// Mask prototypes, and so mask coefficients per instance.
const MASK_COEFFICIENTS: usize = 32;
/// `x1 y1 x2 y2 confidence class_id`, then the coefficients.
const ROW_WIDTH: usize = 6 + MASK_COEFFICIENTS;
/// The export's own recommended confidence, from its `config.json`.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.25;
/// Same-class boxes at or above this IoU are one instance emitted twice.
const DEDUP_IOU: f32 = 0.7;
/// A mask probability of 0.5, in the u8 space the resize runs in.
const MASK_THRESHOLD: u8 = 128;

koharu_runtime::declare_hf_model_package!(
    id: "model:koharu-yolo26s:onnx",
    repo: "ShiniShiho/koharu-yolo26s-onnx",
    file: "koharu-yolo26s.onnx",
    bootstrap: false,
    order: 124,
);

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------

/// The four classes, in the export's own id order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Yolo26sClass {
    /// Panel border.
    Frame,
    DialogueText,
    Balloon,
    /// Sound effects. Fusion treats these as proposals: the rectangle covers
    /// art as often as glyphs.
    OnomatopoeiaText,
}

impl Yolo26sClass {
    fn from_id(id: usize) -> Option<Self> {
        Some(match id {
            0 => Self::Frame,
            1 => Self::DialogueText,
            2 => Self::Balloon,
            3 => Self::OnomatopoeiaText,
            _ => return None,
        })
    }

    pub fn id(self) -> usize {
        self as usize
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Frame => "frame",
            Self::DialogueText => "dialogue_text",
            Self::Balloon => "balloon",
            Self::OnomatopoeiaText => "onomatopoeia_text",
        }
    }

    /// Every class, for callers that want masks on all of them.
    pub const ALL: [Self; 4] = [
        Self::Frame,
        Self::DialogueText,
        Self::Balloon,
        Self::OnomatopoeiaText,
    ];
}

impl std::fmt::Display for Yolo26sClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// One instance. `bbox` is xyxy in original page pixels; `mask` is a page-sized
/// binary mask (0 or 255) when masks were requested.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Yolo26sRegion {
    pub class: Yolo26sClass,
    pub score: f32,
    pub bbox: [f32; 4],
    #[serde(skip)]
    pub mask: Option<GrayImage>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Yolo26sDetection {
    pub image_width: u32,
    pub image_height: u32,
    pub regions: Vec<Yolo26sRegion>,
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct KoharuYolo26sDetector {
    session: OnnxSession,
}

impl KoharuYolo26sDetector {
    pub async fn load(runtime: &RuntimeManager) -> Result<Self> {
        let path = runtime
            .downloads()
            .huggingface_model(HF_REPO, MODEL_FILE)
            .await?;
        Self::load_from_path(&path)
    }

    /// Loads on the CPU execution provider, deliberately, with no accelerated
    /// option.
    ///
    /// CoreML agrees on geometry (box IoU at worst 0.995) but shifts
    /// confidences by up to 0.141 and invented two instances across 130, all of
    /// them near the threshold. Fusion counts detector votes, so a detection
    /// that appears or vanishes with the host's execution provider changes what
    /// gets erased — reproducibility is worth more here than the latency.
    pub fn load_from_path(path: &Path) -> Result<Self> {
        // No warmup pass: the CPU provider executes every node, so there is no
        // provider claim to probe.
        let session = OnnxSession::open(path, true, |_| Ok(()))?;
        Ok(Self { session })
    }

    /// Detect at the export's recommended confidence, decoding instance masks
    /// for the classes in `masks` (`&[]` for none).
    pub fn inference(
        &self,
        image: &DynamicImage,
        masks: &[Yolo26sClass],
    ) -> Result<Yolo26sDetection> {
        self.inference_with_threshold(image, DEFAULT_CONFIDENCE_THRESHOLD, masks)
    }

    /// `masks` selects *which classes* get their instance mask decoded, because
    /// decoding is per-instance page-sized resampling and callers usually read
    /// one or two classes' masks. The matmul that produces the probabilities is
    /// shared and cheap; `mask_to_page` is what costs, so it runs only for the
    /// instances that were asked for.
    #[instrument(level = "debug", skip_all)]
    pub fn inference_with_threshold(
        &self,
        image: &DynamicImage,
        confidence_threshold: f32,
        masks: &[Yolo26sClass],
    ) -> Result<Yolo26sDetection> {
        let letterboxed = letterbox(image);
        let geometry = letterboxed.letterbox;
        let input = input_tensor(&letterboxed)?;

        let (instances, prototypes) =
            self.session
                .run(ort::inputs!["images" => input], |outputs| {
                    let instances = read_instances(outputs, &geometry, confidence_threshold)?;
                    let wanted =
                        !masks.is_empty() && instances.iter().any(|i| masks.contains(&i.class));
                    let prototypes = if wanted {
                        Some(read_prototypes(outputs)?)
                    } else {
                        None
                    };
                    Ok((instances, prototypes))
                })?;

        let instances = dedup(instances);
        let mut decoded: Vec<Option<GrayImage>> = vec![None; instances.len()];
        if let Some(prototypes) = &prototypes {
            let selected: Vec<(usize, &Instance)> = instances
                .iter()
                .enumerate()
                .filter(|(_, instance)| masks.contains(&instance.class))
                .collect();
            let wanted: Vec<&Instance> = selected.iter().map(|(_, i)| *i).collect();
            for ((at, _), mask) in selected
                .iter()
                .zip(decode_masks(&wanted, prototypes, &geometry)?)
            {
                decoded[*at] = Some(mask);
            }
        }

        let regions = instances
            .into_iter()
            .zip(decoded.drain(..))
            .map(|(instance, mask)| Yolo26sRegion {
                class: instance.class,
                score: instance.score,
                bbox: instance.bbox,
                mask,
            })
            .collect::<Vec<_>>();

        tracing::debug!(
            width = geometry.original_width,
            height = geometry.original_height,
            input = format_args!("{}x{}", geometry.input_width, geometry.input_height),
            regions = regions.len(),
            "koharu-yolo26s detection"
        );

        Ok(Yolo26sDetection {
            image_width: geometry.original_width,
            image_height: geometry.original_height,
            regions,
        })
    }
}

pub async fn prefetch(runtime: &RuntimeManager) -> Result<()> {
    runtime
        .downloads()
        .huggingface_model(HF_REPO, MODEL_FILE)
        .await
        .with_context(|| format!("failed to download {MODEL_FILE} from {HF_REPO}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Where the resized page sits inside the padded input, so instances can be
/// mapped back to page pixels.
#[derive(Debug, Clone, Copy)]
struct Letterbox {
    original_width: u32,
    original_height: u32,
    resized_width: u32,
    resized_height: u32,
    input_width: u32,
    input_height: u32,
    pad_x: u32,
    pad_y: u32,
    scale: f32,
}

struct Letterboxed {
    image: RgbImage,
    letterbox: Letterbox,
}

/// Scale the long edge to 1280 and pad to the next stride multiple, centred —
/// Ultralytics' own `predict` convention.
///
/// Not interchangeable with padding out to a square 1280x1280. Same graph, same
/// pages: the two agree on 109 instances but each finds some the other misses
/// (2 square-only, 5 rect-only), box IoU falls to 0.900, and confidences move
/// by 0.064 on average and 0.634 at worst. Rectangular is also ~24% faster (445
/// ms against 582 ms per page) — there is simply less grey to convolve.
fn letterbox(image: &DynamicImage) -> Letterboxed {
    let rgb = image.to_rgb8();
    let (original_width, original_height) = rgb.dimensions();
    let scale = INPUT_SIZE as f32 / original_width.max(original_height).max(1) as f32;
    let resized_width = ((original_width as f32 * scale).round() as u32).max(1);
    let resized_height = ((original_height as f32 * scale).round() as u32).max(1);
    let input_width = resized_width.div_ceil(STRIDE) * STRIDE;
    let input_height = resized_height.div_ceil(STRIDE) * STRIDE;
    let pad_x = (input_width - resized_width) / 2;
    let pad_y = (input_height - resized_height) / 2;

    let resized = if resized_width == original_width && resized_height == original_height {
        rgb
    } else {
        imageops::resize(&rgb, resized_width, resized_height, FilterType::Triangle)
    };

    let mut canvas = RgbImage::from_pixel(input_width, input_height, Rgb([LETTERBOX_COLOR; 3]));
    imageops::overlay(&mut canvas, &resized, i64::from(pad_x), i64::from(pad_y));

    Letterboxed {
        image: canvas,
        letterbox: Letterbox {
            original_width,
            original_height,
            resized_width,
            resized_height,
            input_width,
            input_height,
            pad_x,
            pad_y,
            scale,
        },
    }
}

fn input_tensor(letterboxed: &Letterboxed) -> Result<OrtTensor<f32>> {
    let (width, height) = (
        letterboxed.letterbox.input_width as usize,
        letterboxed.letterbox.input_height as usize,
    );
    let plane = width * height;
    let mut values = vec![0f32; plane * 3];
    for (index, pixel) in letterboxed.image.as_raw().chunks_exact(3).enumerate() {
        for channel in 0..3 {
            values[channel * plane + index] = pixel[channel] as f32 / 255.0;
        }
    }

    OrtTensor::from_array((vec![1i64, 3, height as i64, width as i64], values)).map_err(ort_err)
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// A surviving detection, before its mask is decoded.
struct Instance {
    class: Yolo26sClass,
    score: f32,
    bbox: [f32; 4],
    coefficients: Vec<f32>,
}

/// The shared mask basis, at a quarter of the input resolution.
struct Prototypes {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

fn read_instances(
    outputs: &ort::session::SessionOutputs<'_>,
    letterbox: &Letterbox,
    confidence_threshold: f32,
) -> Result<Vec<Instance>> {
    let (shape, data) = outputs
        .get("output0")
        .context("koharu-yolo26s ONNX output `output0` missing")?
        .try_extract_tensor::<f32>()
        .map_err(ort_err)?;
    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    let [batch, _, row_width] = dims[..] else {
        bail!("unexpected koharu-yolo26s `output0` shape {dims:?}, expected [1, instances, 38]");
    };
    if batch != 1 || row_width != ROW_WIDTH {
        bail!("unexpected koharu-yolo26s `output0` shape {dims:?}, expected [1, instances, 38]");
    }

    let mut instances = Vec::new();
    for row in data.chunks_exact(row_width) {
        let score = row[4];
        if score < confidence_threshold {
            continue;
        }
        let class = Yolo26sClass::from_id(row[5] as usize)
            .with_context(|| format!("koharu-yolo26s returned unknown class id {}", row[5]))?;
        let bbox = map_box(&row[..4], letterbox);
        if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            continue;
        }
        instances.push(Instance {
            class,
            score,
            bbox,
            coefficients: row[6..].to_vec(),
        });
    }
    // Rows arrive confidence-sorted, but `dedup` depends on that, so make it
    // true here rather than trust it.
    instances.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(instances)
}

fn read_prototypes(outputs: &ort::session::SessionOutputs<'_>) -> Result<Prototypes> {
    let (shape, data) = outputs
        .get("output1")
        .context("koharu-yolo26s ONNX output `output1` missing")?
        .try_extract_tensor::<f32>()
        .map_err(ort_err)?;
    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    let [batch, channels, height, width] = dims[..] else {
        bail!("unexpected koharu-yolo26s `output1` shape {dims:?}, expected [1, 32, h, w]");
    };
    if batch != 1 || channels != MASK_COEFFICIENTS {
        bail!("unexpected koharu-yolo26s `output1` shape {dims:?}, expected [1, 32, h, w]");
    }

    Ok(Prototypes {
        width,
        height,
        values: data.to_vec(),
    })
}

/// Input-space `x1 y1 x2 y2` to page pixels.
fn map_box(raw: &[f32], letterbox: &Letterbox) -> [f32; 4] {
    let width = letterbox.original_width as f32;
    let height = letterbox.original_height as f32;
    let pad_x = letterbox.pad_x as f32;
    let pad_y = letterbox.pad_y as f32;
    [
        ((raw[0] - pad_x) / letterbox.scale).clamp(0.0, width),
        ((raw[1] - pad_y) / letterbox.scale).clamp(0.0, height),
        ((raw[2] - pad_x) / letterbox.scale).clamp(0.0, width),
        ((raw[3] - pad_y) / letterbox.scale).clamp(0.0, height),
    ]
}

/// Drop repeated instances of a single object.
///
/// The head is end-to-end and runs no NMS, which is right for overlap *across*
/// classes — a balloon and the text inside it are two real objects, and
/// suppressing either is how SFX and captions get lost. But it does emit the
/// same instance twice: 3 of 144 instances at confidence 0.25, 12 of 166 at
/// 0.10, over twelve pages. The PyTorch checkpoint duplicates identically, so
/// this is the head's behaviour and not the export's.
///
/// Expects descending score, so the survivor of a pair is the stronger one.
fn dedup(instances: Vec<Instance>) -> Vec<Instance> {
    // ponytail: O(n²) over a page's tens of instances. Spatial index only if a
    // page ever holds thousands.
    let mut kept: Vec<Instance> = Vec::with_capacity(instances.len());
    for candidate in instances {
        let duplicate = kept
            .iter()
            .any(|k| k.class == candidate.class && iou(k.bbox, candidate.bbox) >= DEDUP_IOU);
        if !duplicate {
            kept.push(candidate);
        }
    }
    kept
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = width * height;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Every requested instance mask in one matmul: `sigmoid(coefficients · prototypes)`.
fn decode_masks(
    instances: &[&Instance],
    prototypes: &Prototypes,
    letterbox: &Letterbox,
) -> Result<Vec<GrayImage>> {
    if instances.is_empty() {
        return Ok(Vec::new());
    }
    let coefficients: Vec<f32> = instances
        .iter()
        .flat_map(|instance| instance.coefficients.iter().copied())
        .collect();
    let coefficients = Tensor::from_vec(
        coefficients,
        (instances.len(), MASK_COEFFICIENTS),
        &Device::Cpu,
    )?;
    let basis = Tensor::from_slice(
        &prototypes.values,
        (MASK_COEFFICIENTS, prototypes.width * prototypes.height),
        &Device::Cpu,
    )?;
    let probabilities = candle_nn::ops::sigmoid(&coefficients.matmul(&basis)?)?.to_vec2::<f32>()?;

    probabilities
        .iter()
        .zip(instances)
        .map(|(row, instance)| mask_to_page(row, prototypes, letterbox, instance.bbox))
        .collect()
}

/// One instance's mask probabilities to a page-sized binary mask.
///
/// Upsampled to the full padded input before the padding is cropped off,
/// because the prototypes are quarter-resolution: `pad / 4` is fractional, and
/// cropping there would shift the whole mask by up to two page pixels.
/// Probabilities are interpolated rather than logits, which keeps the 0.5
/// decision a `>= 128` test inside `image`'s integer resize path.
///
/// The basis is shared across instances, so an uncropped mask carries
/// probability mass from elsewhere on the page; the box is the crop.
fn mask_to_page(
    probabilities: &[f32],
    prototypes: &Prototypes,
    letterbox: &Letterbox,
    bbox: [f32; 4],
) -> Result<GrayImage> {
    let small = GrayImage::from_fn(prototypes.width as u32, prototypes.height as u32, |x, y| {
        let value = probabilities[y as usize * prototypes.width + x as usize];
        Luma([(value.clamp(0.0, 1.0) * 255.0).round() as u8])
    });

    // ponytail: upsamples the whole plane once per instance. Crop to the box in
    // prototype space first if a page ever carries hundreds.
    let upscaled = imageops::resize(
        &small,
        letterbox.input_width,
        letterbox.input_height,
        FilterType::Triangle,
    );
    let content = imageops::crop_imm(
        &upscaled,
        letterbox.pad_x,
        letterbox.pad_y,
        letterbox.resized_width,
        letterbox.resized_height,
    )
    .to_image();
    let page = imageops::resize(
        &content,
        letterbox.original_width,
        letterbox.original_height,
        FilterType::Triangle,
    );

    Ok(GrayImage::from_fn(
        letterbox.original_width,
        letterbox.original_height,
        |x, y| {
            let inside = (x as f32) >= bbox[0].floor()
                && (x as f32) < bbox[2].ceil()
                && (y as f32) >= bbox[1].floor()
                && (y as f32) < bbox[3].ceil();
            Luma([if inside && page.get_pixel(x, y).0[0] >= MASK_THRESHOLD {
                255
            } else {
                0
            }])
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance(class: Yolo26sClass, score: f32, bbox: [f32; 4]) -> Instance {
        Instance {
            class,
            score,
            bbox,
            coefficients: vec![0.0; MASK_COEFFICIENTS],
        }
    }

    #[test]
    fn letterbox_pads_the_short_edge_to_a_stride_multiple() {
        let page = DynamicImage::new_rgb8(850, 1200);
        let letterbox = letterbox(&page).letterbox;

        assert_eq!(letterbox.resized_height, INPUT_SIZE, "long edge hits imgsz");
        assert_eq!(letterbox.input_height, INPUT_SIZE);
        assert_eq!(letterbox.resized_width, 907);
        // 907 rounds up to 928, split evenly either side.
        assert_eq!(letterbox.input_width, 928);
        assert_eq!(letterbox.pad_x, 10);
        assert_eq!(letterbox.pad_y, 0);
        assert_eq!(letterbox.input_width % STRIDE, 0);
        assert_eq!(letterbox.input_height % STRIDE, 0);
    }

    #[test]
    fn map_box_undoes_the_letterbox() {
        let letterbox = letterbox(&DynamicImage::new_rgb8(850, 1200)).letterbox;
        // The padded input's own corners map to the page's corners, clamped.
        let full = map_box(
            &[
                letterbox.pad_x as f32,
                letterbox.pad_y as f32,
                (letterbox.pad_x + letterbox.resized_width) as f32,
                (letterbox.pad_y + letterbox.resized_height) as f32,
            ],
            &letterbox,
        );
        assert!(full[0].abs() < 1e-3 && full[1].abs() < 1e-3);
        assert!((full[2] - 850.0).abs() < 1.0);
        assert!((full[3] - 1200.0).abs() < 1.0);
    }

    #[test]
    fn dedup_drops_the_weaker_twin_and_keeps_nested_classes() {
        let balloon = [100.0, 100.0, 300.0, 260.0];
        let kept = dedup(vec![
            instance(Yolo26sClass::Balloon, 0.91, balloon),
            // Same balloon again, 0.88 IoU — the head's duplicate.
            instance(Yolo26sClass::Balloon, 0.62, [104.0, 104.0, 296.0, 256.0]),
            // The text inside it: heavy overlap, different class, real object.
            instance(
                Yolo26sClass::DialogueText,
                0.80,
                [110.0, 110.0, 290.0, 250.0],
            ),
        ]);

        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].class, Yolo26sClass::Balloon);
        assert_eq!(kept[0].bbox, balloon, "the stronger twin survives");
        assert_eq!(kept[1].class, Yolo26sClass::DialogueText);
    }

    #[test]
    fn mask_decode_thresholds_at_half_and_crops_to_the_box() {
        // A page that needs no resize or padding, so the mask geometry is the
        // only thing under test.
        let letterbox = Letterbox {
            original_width: 32,
            original_height: 32,
            resized_width: 32,
            resized_height: 32,
            input_width: 32,
            input_height: 32,
            pad_x: 0,
            pad_y: 0,
            scale: 1.0,
        };
        let prototypes = Prototypes {
            width: 8,
            height: 8,
            values: Vec::new(),
        };
        // Certain in the top-left quadrant and the bottom-right one.
        let mut probabilities = vec![0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                let quadrant = (x < 4 && y < 4) || (x >= 4 && y >= 4);
                probabilities[y * 8 + x] = if quadrant { 1.0 } else { 0.0 };
            }
        }

        let mask = mask_to_page(
            &probabilities,
            &prototypes,
            &letterbox,
            [0.0, 0.0, 16.0, 16.0],
        )
        .unwrap();
        assert_eq!(mask.get_pixel(4, 4).0[0], 255, "inside the box");
        assert_eq!(
            mask.get_pixel(24, 24).0[0],
            0,
            "another instance's mass, cropped away by the box"
        );
        assert_eq!(mask.get_pixel(24, 4).0[0], 0, "low probability");
    }

    #[test]
    fn class_ids_match_the_export() {
        for (id, expected) in [
            (0, Yolo26sClass::Frame),
            (1, Yolo26sClass::DialogueText),
            (2, Yolo26sClass::Balloon),
            (3, Yolo26sClass::OnomatopoeiaText),
        ] {
            assert_eq!(Yolo26sClass::from_id(id), Some(expected));
            assert_eq!(expected.id(), id);
        }
        assert_eq!(Yolo26sClass::from_id(4), None);
        assert_eq!(Yolo26sClass::OnomatopoeiaText.as_str(), "onomatopoeia_text");
    }
}
