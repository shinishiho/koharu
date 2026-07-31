//! Fused detector: four models propose, `pipeline::fusion` decides.
//!
//! The layout model contributes structure (balloons, dialogue, SFX) and its
//! instance masks; RT-DETR a second opinion on balloons and text; AnimeText and
//! comic-text-detector broad text boxes, the latter also the pixel mask that
//! settles whether a rectangle actually holds glyphs.
//!
//! Only `accepted` regions become text nodes. Inpainting expands its mask
//! around every text node it finds (`lama.rs` → `expand_mask_for_inpainting`),
//! so a proposal kept "for review" would be erased exactly like a confirmed
//! one. Until the scene can carry a proposal that nothing acts on, fusion's
//! unaccepted half is counted in a log line and dropped.

use anyhow::Result;
use async_trait::async_trait;
use image::GrayImage;
use koharu_core::{Op, TextData, TextDirection};
use koharu_ml::anime_text::{AnimeTextDetector, AnimeTextYoloVariant};
use koharu_ml::comic_text_bubble_detector::ComicTextBubbleDetector;
use koharu_ml::comic_text_detector::ComicTextDetector;
use koharu_ml::koharu_yolo26s::{
    DEFAULT_CONFIDENCE_THRESHOLD as LAYOUT_CONFIDENCE, KoharuYolo26sDetector, Yolo26sClass,
    Yolo26sRegion,
};
use koharu_runtime::RuntimeManager;

use crate::pipeline::artifacts::Artifact;
use crate::pipeline::engine::{Engine, EngineCtx, EngineInfo};
use crate::pipeline::engines::support::{
    clear_text_nodes_ops, load_source_image, new_text_node, page_node_count,
    sort_manga_reading_order,
};
use crate::pipeline::fusion::{Candidate, Class, Detector, FusedRegion, FusionConfig, fuse};

const DETECTOR_NAME: &str = "koharu-fusion";
/// Taller than this much of its own width reads as vertical Japanese.
const VERTICAL_ASPECT: f32 = 1.15;
/// A mask pixel at or above this counts as text.
const MASK_ON: u8 = 128;

pub struct Model {
    layout: KoharuYolo26sDetector,
    rtdetr: ComicTextBubbleDetector,
    ctd: ComicTextDetector,
    /// `deepghs/AnimeText_yolo` is gated, so this is the one voter that can be
    /// absent. Fusion counts distinct detectors and never assumes four: with
    /// three, every threshold it has is still reachable. A missing token costs
    /// recall on text outside balloons, not correctness.
    anime: Option<AnimeTextDetector>,
    config: FusionConfig,
}

#[async_trait]
impl Engine for Model {
    async fn run(&self, ctx: EngineCtx<'_>) -> Result<Vec<Op>> {
        let image = load_source_image(ctx.scene, ctx.page, ctx.blobs)?;

        // Onomatopoeia masks only. Fusion's SFX rule asks how much of the box
        // the mask actually traced, which nothing but the decoded mask can
        // answer; every other class either has no mask clause or resolves it
        // from the detector alone, and decoding costs ~24 ms an instance.
        let layout = self
            .layout
            .inference(&image, &[Yolo26sClass::OnomatopoeiaText])?;
        let rtdetr = self.rtdetr.inference(&image)?;
        let ctd = self.ctd.inference(&image)?;

        let mut candidates = layout_candidates(&layout.regions);
        candidates.extend(rtdetr.detections.iter().map(|region| Candidate {
            bbox: region.bbox,
            detector: Detector::RtDetr,
            class: if region.is_bubble() {
                Class::Bubble
            } else {
                Class::Text
            },
            score: region.score,
            has_mask: false,
            mask_fill: None,
        }));
        candidates.extend(ctd.text_blocks.iter().map(|block| Candidate {
            bbox: [
                block.x,
                block.y,
                block.x + block.width,
                block.y + block.height,
            ],
            detector: Detector::ComicTextDetector,
            class: Class::Text,
            score: block.confidence,
            has_mask: false,
            mask_fill: None,
        }));
        if let Some(anime) = &self.anime {
            let detection = anime.inference(&image)?;
            candidates.extend(detection.regions.iter().map(|region| Candidate {
                bbox: region.bbox,
                detector: Detector::AnimeText,
                class: Class::Text,
                score: region.score,
                has_mask: false,
                mask_fill: None,
            }));
        }

        let mask = ctd.mask;
        let fusion = fuse(&candidates, &self.config, |bbox| mask_coverage(&mask, bbox));

        // Per-region evidence, at debug level: these are the measurements
        // `mask_coverage` and `sfx_mask_fill` are tuned from, and both move
        // whenever the models behind them change.
        if tracing::enabled!(tracing::Level::DEBUG) {
            for region in &fusion.regions {
                tracing::debug!(
                    coverage = mask_coverage(&mask, region.bbox),
                    fill = region.mask_fill,
                    votes = region.detectors.len(),
                    role = ?region.role,
                    accepted = region.accepted,
                    "fusion region"
                );
            }
        }

        let accepted = fusion.regions.iter().filter(|r| r.accepted).count();
        let voters: std::collections::HashSet<Detector> =
            candidates.iter().map(|c| c.detector).collect();
        tracing::info!(
            candidates = candidates.len(),
            bubbles = fusion.bubbles.len(),
            regions = fusion.regions.len(),
            accepted,
            dropped = fusion.regions.len() - accepted,
            // What actually proposed something, not how many models loaded.
            voters = voters.len(),
            "fusion"
        );

        let mut ops = clear_text_nodes_ops(ctx.scene, ctx.page);
        let removed = ops.len();
        let insertion_start = page_node_count(ctx.scene, ctx.page).saturating_sub(removed);

        let mut blocks: Vec<([f32; 4], TextData)> = fusion
            .regions
            .iter()
            .filter(|region| region.accepted)
            .map(text_block)
            .collect();
        sort_manga_reading_order(&mut blocks, ctx.options.reading_order.unwrap_or_default());
        ops.reserve(blocks.len());
        for (at, (bbox, text)) in (insertion_start..).zip(blocks) {
            let node = new_text_node(bbox, text);
            ops.push(Op::AddNode {
                page: ctx.page,
                node,
                at,
            });
        }
        Ok(ops)
    }
}

inventory::submit! {
    EngineInfo {
        id: "koharu-fusion",
        name: "Koharu Fusion (4 detectors)",
        needs: &[],
        produces: &[Artifact::TextBoxes],
        load: |runtime, cpu| Box::pin(async move {
            Ok(Box::new(load(runtime, cpu).await?) as Box<dyn Engine>)
        }),
    }
}

async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Model> {
    Ok(Model {
        // The layout model pins CPU itself: its confidences move with the
        // execution provider, and fusion counts detections.
        layout: KoharuYolo26sDetector::load(runtime).await?,
        rtdetr: ComicTextBubbleDetector::load(runtime, cpu).await?,
        ctd: ComicTextDetector::load(runtime, cpu).await?,
        anime: match AnimeTextDetector::load_variant(runtime, AnimeTextYoloVariant::N, cpu).await {
            Ok(model) => Some(model),
            Err(error) => {
                tracing::warn!(
                    error = %format!("{error:#}"),
                    "AnimeText unavailable, fusion runs with three voters; add a HuggingFace token in Settings to enable it"
                );
                None
            }
        },
        config: FusionConfig::default(),
    })
}

/// Layout classes onto fusion's. `Frame` is panel geometry, not a text
/// proposal, so it never becomes a candidate.
fn layout_candidates(regions: &[Yolo26sRegion]) -> Vec<Candidate> {
    regions
        .iter()
        .filter(|region| region.score >= LAYOUT_CONFIDENCE)
        .filter_map(|region| {
            let class = match region.class {
                Yolo26sClass::Balloon => Class::Bubble,
                Yolo26sClass::DialogueText => Class::Text,
                Yolo26sClass::OnomatopoeiaText => Class::Sfx,
                Yolo26sClass::Frame => return None,
            };
            Some(Candidate {
                bbox: region.bbox,
                detector: Detector::Layout,
                class,
                score: region.score,
                // Every yolo26s instance carries mask coefficients, decoded or
                // not — this says the proposal came from the model that
                // segments, which is what tells it apart from RT-DETR's.
                has_mask: true,
                // Decoded only for the classes asked for at inference; the box
                // is in page pixels and so is the mask.
                mask_fill: region
                    .mask
                    .as_ref()
                    .map(|mask| mask_coverage(mask, region.bbox)),
            })
        })
        .collect()
}

/// Fraction of `bbox` the text mask covers. Boxes are clamped to the mask,
/// which is page-sized, so an off-page box scores on the part that exists.
fn mask_coverage(mask: &GrayImage, bbox: [f32; 4]) -> f32 {
    let (width, height) = mask.dimensions();
    let x1 = bbox[0].max(0.0).min(width as f32) as u32;
    let y1 = bbox[1].max(0.0).min(height as f32) as u32;
    let x2 = bbox[2].max(0.0).min(width as f32) as u32;
    let y2 = bbox[3].max(0.0).min(height as f32) as u32;
    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let mut on = 0u32;
    for y in y1..y2 {
        for x in x1..x2 {
            if mask.get_pixel(x, y).0[0] >= MASK_ON {
                on += 1;
            }
        }
    }
    on as f32 / ((x2 - x1) * (y2 - y1)) as f32
}

fn text_block(region: &FusedRegion) -> ([f32; 4], TextData) {
    let width = (region.bbox[2] - region.bbox[0]).max(1.0);
    let height = (region.bbox[3] - region.bbox[1]).max(1.0);
    let direction = if height >= width * VERTICAL_ASPECT {
        TextDirection::Vertical
    } else {
        TextDirection::Horizontal
    };
    let text = TextData {
        confidence: region.score,
        source_direction: Some(direction),
        source_lang: Some("unknown".to_string()),
        rotation_deg: Some(0.0),
        detected_font_size_px: Some(width.min(height)),
        // Which models voted, so a bad box can be traced back to who proposed
        // it: "koharu-fusion(layout+ctd)".
        detector: Some(format!(
            "{DETECTOR_NAME}({})",
            region
                .detectors
                .iter()
                .map(detector_tag)
                .collect::<Vec<_>>()
                .join("+")
        )),
        ..Default::default()
    };
    (region.bbox, text)
}

fn detector_tag(detector: &Detector) -> &'static str {
    match detector {
        Detector::Layout => "layout",
        Detector::RtDetr => "rtdetr",
        Detector::AnimeText => "animetext",
        Detector::ComicTextDetector => "ctd",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(class: Yolo26sClass, bbox: [f32; 4], score: f32) -> Yolo26sRegion {
        Yolo26sRegion {
            class,
            score,
            bbox,
            mask: None,
        }
    }

    /// A page-sized mask lighting `lit` inside `bbox`, for mask-fill tests.
    fn mask_with(width: u32, height: u32, lit: [u32; 4]) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            let inside = x >= lit[0] && x < lit[2] && y >= lit[1] && y < lit[3];
            image::Luma([if inside { 255 } else { 0 }])
        })
    }

    #[test]
    fn layout_classes_map_onto_fusion_classes() {
        let candidates = layout_candidates(&[
            region(Yolo26sClass::Frame, [0.0, 0.0, 400.0, 600.0], 0.95),
            region(Yolo26sClass::Balloon, [10.0, 10.0, 90.0, 120.0], 0.9),
            region(Yolo26sClass::DialogueText, [20.0, 20.0, 60.0, 110.0], 0.8),
            region(
                Yolo26sClass::OnomatopoeiaText,
                [200.0, 40.0, 320.0, 90.0],
                0.6,
            ),
            region(
                Yolo26sClass::DialogueText,
                [0.0, 0.0, 10.0, 10.0],
                LAYOUT_CONFIDENCE - 0.01,
            ),
        ]);

        let classes: Vec<Class> = candidates.iter().map(|c| c.class).collect();
        assert_eq!(classes, vec![Class::Bubble, Class::Text, Class::Sfx]);
        assert!(candidates.iter().all(|c| c.has_mask));
        // Undecoded masks leave no fill to measure, which is what keeps the SFX
        // rule from resting on a value nobody computed.
        assert!(candidates.iter().all(|c| c.mask_fill.is_none()));
    }

    #[test]
    fn sfx_mask_fill_measures_the_decoded_mask_not_its_existence() {
        // Two SFX boxes the model scored identically: one whose mask traced a
        // fifth of the box, one whose mask lit nothing. Before masks were
        // decoded both were `has_mask: true` and indistinguishable.
        let bbox = [0.0, 0.0, 100.0, 100.0];
        let traced = Yolo26sRegion {
            mask: Some(mask_with(100, 100, [0, 0, 100, 20])),
            ..region(Yolo26sClass::OnomatopoeiaText, bbox, 0.6)
        };
        let empty = Yolo26sRegion {
            mask: Some(mask_with(100, 100, [0, 0, 0, 0])),
            ..region(Yolo26sClass::OnomatopoeiaText, bbox, 0.6)
        };

        let candidates = layout_candidates(&[traced, empty]);
        assert_eq!(candidates[0].mask_fill, Some(0.2));
        assert_eq!(candidates[1].mask_fill, Some(0.0));
        assert!(candidates.iter().all(|c| c.has_mask), "same on both");

        // Same glyph evidence, opposite outcomes — the whole point of the fill.
        let config = FusionConfig::default();
        let accepted = |candidate: &Candidate| {
            fuse(std::slice::from_ref(candidate), &config, |_| 1.0).regions[0].accepted
        };
        assert!(accepted(&candidates[0]));
        assert!(!accepted(&candidates[1]));
    }

    #[test]
    fn coverage_counts_only_lit_pixels_inside_the_box() {
        let mut mask = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..5 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }

        assert_eq!(mask_coverage(&mask, [0.0, 0.0, 10.0, 10.0]), 0.5);
        assert_eq!(mask_coverage(&mask, [0.0, 0.0, 5.0, 10.0]), 1.0);
        assert_eq!(mask_coverage(&mask, [5.0, 0.0, 10.0, 10.0]), 0.0);
    }

    #[test]
    fn coverage_clamps_boxes_that_run_off_the_page() {
        let mut mask = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }

        assert_eq!(mask_coverage(&mask, [-50.0, -50.0, 50.0, 50.0]), 1.0);
        assert_eq!(mask_coverage(&mask, [20.0, 20.0, 30.0, 30.0]), 0.0);
    }

    #[test]
    fn text_block_records_every_voter() {
        let (bbox, text) = text_block(&FusedRegion {
            bbox: [10.0, 10.0, 40.0, 90.0],
            role: crate::pipeline::fusion::Role::Outside,
            score: 0.7,
            detectors: vec![Detector::Layout, Detector::ComicTextDetector],
            accepted: true,
            bubble: None,
            mask_fill: None,
        });

        assert_eq!(bbox, [10.0, 10.0, 40.0, 90.0]);
        assert_eq!(text.detector.as_deref(), Some("koharu-fusion(layout+ctd)"));
        assert_eq!(text.source_direction, Some(TextDirection::Vertical));
    }
}
