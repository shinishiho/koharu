//! koharu's YOLO26-seg layout detector. Emits one `AddNode { Text }` per
//! region that holds glyphs — dialogue text and SFX — and drops the structural
//! classes (frame, balloon), which describe where text sits rather than being
//! text themselves.

use anyhow::Result;
use async_trait::async_trait;
use koharu_core::{Op, TextData, TextDirection};
use koharu_ml::koharu_yolo26s::{
    DEFAULT_CONFIDENCE_THRESHOLD, KoharuYolo26sDetector, Yolo26sClass, Yolo26sRegion,
};

use crate::pipeline::artifacts::Artifact;
use crate::pipeline::engine::{Engine, EngineCtx, EngineInfo};
use crate::pipeline::engines::support::{
    clear_text_nodes_ops, load_source_image, new_text_node, page_node_count,
    sort_manga_reading_order,
};

const DETECTOR_NAME: &str = "koharu-yolo26s";
/// Taller than this much of its own width reads as vertical Japanese.
const VERTICAL_ASPECT: f32 = 1.15;

pub struct Model(KoharuYolo26sDetector);

#[async_trait]
impl Engine for Model {
    async fn run(&self, ctx: EngineCtx<'_>) -> Result<Vec<Op>> {
        let image = load_source_image(ctx.scene, ctx.page, ctx.blobs)?;
        // No masks: the pipeline wants boxes, and decoding masks costs ~24 ms
        // per instance.
        let detection = self.0.inference(&image, false)?;

        let mut ops = clear_text_nodes_ops(ctx.scene, ctx.page);
        let removed = ops.len();
        let insertion_start = page_node_count(ctx.scene, ctx.page).saturating_sub(removed);

        let mut blocks = text_blocks(&detection.regions);
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
        id: "koharu-yolo26s",
        name: "Koharu Layout (YOLO26-seg)",
        needs: &[],
        produces: &[Artifact::TextBoxes],
        load: |runtime, _cpu| Box::pin(async move {
            // The detector pins CPU itself: its confidences move with the
            // execution provider, and fusion counts detections.
            let m = KoharuYolo26sDetector::load(runtime).await?;
            Ok(Box::new(Model(m)) as Box<dyn Engine>)
        }),
    }
}

fn text_blocks(regions: &[Yolo26sRegion]) -> Vec<([f32; 4], TextData)> {
    regions
        .iter()
        .filter(|r| {
            matches!(
                r.class,
                Yolo26sClass::DialogueText | Yolo26sClass::OnomatopoeiaText
            )
        })
        .filter(|r| r.score >= DEFAULT_CONFIDENCE_THRESHOLD)
        .map(|r| {
            let width = (r.bbox[2] - r.bbox[0]).max(1.0);
            let height = (r.bbox[3] - r.bbox[1]).max(1.0);
            let direction = if height >= width * VERTICAL_ASPECT {
                TextDirection::Vertical
            } else {
                TextDirection::Horizontal
            };
            let text = TextData {
                confidence: r.score,
                source_direction: Some(direction),
                source_lang: Some("unknown".to_string()),
                rotation_deg: Some(0.0),
                detected_font_size_px: Some(width.min(height)),
                detector: Some(DETECTOR_NAME.to_string()),
                ..Default::default()
            };
            (r.bbox, text)
        })
        .collect()
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

    #[test]
    fn keeps_glyph_classes_and_drops_structure() {
        let blocks = text_blocks(&[
            region(Yolo26sClass::Frame, [0.0, 0.0, 400.0, 600.0], 0.95),
            region(Yolo26sClass::Balloon, [10.0, 10.0, 90.0, 120.0], 0.9),
            region(Yolo26sClass::DialogueText, [20.0, 20.0, 60.0, 110.0], 0.8),
            region(
                Yolo26sClass::OnomatopoeiaText,
                [200.0, 40.0, 320.0, 90.0],
                0.6,
            ),
        ]);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].1.source_direction, Some(TextDirection::Vertical));
        assert_eq!(
            blocks[1].1.source_direction,
            Some(TextDirection::Horizontal)
        );
    }

    #[test]
    fn drops_regions_below_the_threshold() {
        let blocks = text_blocks(&[region(
            Yolo26sClass::DialogueText,
            [0.0, 0.0, 10.0, 10.0],
            DEFAULT_CONFIDENCE_THRESHOLD - 0.01,
        )]);
        assert!(blocks.is_empty());
    }
}
