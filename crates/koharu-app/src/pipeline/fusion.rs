//! Multi-detector fusion geometry.
//!
//! Detectors disagree. The layout model and RT-DETR both propose balloons;
//! AnimeText, RT-DETR and comic-text-detector all propose text. Union of
//! everything maximizes recall but erasing a false positive is destructive, so
//! fusion separates the two questions: *is there something here* (keep it, for
//! OCR and review) and *is the evidence strong enough to inpaint without
//! asking* (`accepted`).
//!
//! Pure geometry — no models, no scene types. Callers map detector output in
//! and `FusedRegion`s out. The pixel mask arrives as a coverage closure so this
//! module stays independent of how the mask is stored or scaled.

/// Which model proposed a candidate. Fusion counts *distinct* detectors, so
/// two boxes from the same model are one vote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Detector {
    /// Structural layout model: text / onomatopoeia / bubble / panel, with
    /// instance masks — `koharu_ml::koharu_yolo26s`. The model exists; no
    /// engine feeds it in here yet.
    Layout,
    /// ogkalu RT-DETR-v2: bubble / text_bubble / text_free.
    RtDetr,
    /// AnimeText YOLO12x: broad text boxes.
    AnimeText,
    /// comic-text-detector: text boxes plus the authoritative pixel mask.
    ComicTextDetector,
}

/// What a candidate claims to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Bubble,
    Text,
    /// Onomatopoeia / sound effects. Proposal-only: SFX boxes cover art as
    /// often as glyphs, so they need pixel or second-detector confirmation.
    Sfx,
}

/// One detector's proposal.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Pixel `[x1, y1, x2, y2]` in source-image space.
    pub bbox: [f32; 4],
    pub detector: Detector,
    pub class: Class,
    pub score: f32,
    /// The detector also produced an instance mask for this box. Only the
    /// layout model does; a masked balloon is self-confirming evidence.
    pub has_mask: bool,
}

/// How a fused text region relates to the page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Text substantially inside an accepted balloon.
    Dialogue,
    /// Text outside every balloon — titles, captions, credits, narration.
    Outside,
    /// Confirmed sound effect outside every balloon.
    Sfx,
}

#[derive(Debug, Clone)]
pub struct FusedBubble {
    pub bbox: [f32; 4],
    pub score: f32,
    pub detectors: Vec<Detector>,
    /// Two detectors agreed, or the layout model produced a strong mask.
    /// Unaccepted balloons still shape routing but never authorize an erase.
    pub accepted: bool,
}

#[derive(Debug, Clone)]
pub struct FusedRegion {
    pub bbox: [f32; 4],
    pub role: Role,
    pub score: f32,
    pub detectors: Vec<Detector>,
    /// Safe to inpaint without confirmation.
    pub accepted: bool,
    /// Index into `Fusion::bubbles` for `Role::Dialogue`.
    pub bubble: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct Fusion {
    pub bubbles: Vec<FusedBubble>,
    pub regions: Vec<FusedRegion>,
}

/// Fusion thresholds. Defaults are the eyeballed values from the model-audit
/// pass; nothing here is fit to a labelled set yet.
#[derive(Debug, Clone, Copy)]
pub struct FusionConfig {
    /// Two boxes of the same class are the same object at or above this IoU.
    pub match_iou: f32,
    /// …or when either box has this fraction of *its own* area inside the
    /// other. Catches a tight box nested in a loose one, where IoU is low.
    pub match_ioa: f32,
    /// Fraction of a text box inside a balloon for it to count as dialogue.
    pub dialogue_ioa: f32,
    /// Fraction of a box covered by the pixel text mask to count as pixel
    /// confirmation.
    pub mask_coverage: f32,
    /// Layout-model score at which its own instance mask alone accepts a
    /// balloon with no second detector.
    pub strong_mask_score: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            match_iou: 0.30,
            match_ioa: 0.50,
            dialogue_ioa: 0.60,
            mask_coverage: 0.10,
            strong_mask_score: 0.50,
        }
    }
}

/// Fuse detector proposals into balloons and text regions.
///
/// `mask_coverage` returns the fraction of a pixel box covered by the
/// authoritative text mask (0.0 when no mask is available). It is the only
/// pixel evidence fusion consults.
pub fn fuse(
    candidates: &[Candidate],
    config: &FusionConfig,
    mask_coverage: impl Fn([f32; 4]) -> f32,
) -> Fusion {
    let bubbles: Vec<FusedBubble> = cluster(candidates, Class::Bubble, config)
        .into_iter()
        .map(|c| FusedBubble {
            accepted: c.detectors.len() >= 2
                || c.masked_score
                    .is_some_and(|score| score >= config.strong_mask_score),
            bbox: c.bbox,
            score: c.score,
            detectors: c.detectors,
        })
        .collect();

    // Text boxes from anything but the layout model confirm its SFX proposals —
    // same rule as the pixel mask, a second opinion that the rectangle holds
    // glyphs. The layout model is excluded from its own confirmation: a
    // detector agreeing with itself is one vote, not two.
    let text_boxes: Vec<[f32; 4]> = candidates
        .iter()
        .filter(|c| c.class == Class::Text && c.detector != Detector::Layout)
        .map(|c| c.bbox)
        .collect();

    let mut regions = Vec::new();
    for class in [Class::Text, Class::Sfx] {
        for cluster in cluster(candidates, class, config) {
            let inside = bubbles
                .iter()
                .enumerate()
                .filter(|(_, b)| ioa(cluster.bbox, b.bbox) >= config.dialogue_ioa)
                .max_by(|(_, a), (_, b)| a.accepted.cmp(&b.accepted));

            let pixels = mask_coverage(cluster.bbox) >= config.mask_coverage;
            let (role, bubble, accepted) = match inside {
                // Inside an accepted balloon the balloon *is* the evidence,
                // for SFX-classified boxes too — a balloon's contents get
                // erased either way.
                Some((index, bubble)) if bubble.accepted => (Role::Dialogue, Some(index), true),
                // Inside a balloon nobody confirmed: keep for OCR, don't erase.
                Some((index, _)) => (Role::Dialogue, Some(index), false),
                // An SFX rectangle covers art as often as glyphs, so it takes
                // two independent things to erase one: the layout model's own
                // instance mask, which traces the strokes rather than the box,
                // and glyph evidence from somewhere else — the pixel text mask
                // or another detector's text box. Either alone stays a
                // proposal. The mask without a second opinion is only as good
                // as the class call that produced it, and glyph evidence
                // without a mask says text is *somewhere* in a rectangle that
                // may be mostly artwork.
                None if class == Class::Sfx => {
                    let glyphs = pixels
                        || text_boxes
                            .iter()
                            .any(|b| overlaps(cluster.bbox, *b, config));
                    let accepted = cluster.masked_score.is_some() && glyphs;
                    (Role::Sfx, None, accepted)
                }
                None => {
                    let accepted = cluster.detectors.len() >= 2 || pixels;
                    (Role::Outside, None, accepted)
                }
            };

            regions.push(FusedRegion {
                bbox: cluster.bbox,
                role,
                score: cluster.score,
                detectors: cluster.detectors,
                accepted,
                bubble,
            });
        }
    }

    Fusion { bubbles, regions }
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

struct Cluster {
    bbox: [f32; 4],
    score: f32,
    detectors: Vec<Detector>,
    /// Best score among candidates that carried an instance mask.
    masked_score: Option<f32>,
}

/// Group same-class candidates that describe one object. Cross-class nesting
/// is deliberately left alone: a text box inside a balloon box is two real
/// objects, and collapsing them is what loses SFX and captions.
fn cluster(candidates: &[Candidate], class: Class, config: &FusionConfig) -> Vec<Cluster> {
    let mut sorted: Vec<&Candidate> = candidates.iter().filter(|c| c.class == class).collect();
    sorted.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut clusters: Vec<Cluster> = Vec::new();
    for candidate in sorted {
        // ponytail: O(n²) against cluster representatives. Page-scale n is
        // tens of boxes; spatial index only if a page ever has thousands.
        let existing = clusters
            .iter_mut()
            .find(|c| overlaps(c.bbox, candidate.bbox, config));
        match existing {
            Some(cluster) => {
                if !cluster.detectors.contains(&candidate.detector) {
                    cluster.detectors.push(candidate.detector);
                }
                if candidate.has_mask {
                    cluster.masked_score = Some(
                        cluster
                            .masked_score
                            .map_or(candidate.score, |s| s.max(candidate.score)),
                    );
                }
            }
            // Highest-scoring box wins the geometry. Unioning boxes across
            // detectors drifts outward every merge and over-erases.
            None => clusters.push(Cluster {
                bbox: candidate.bbox,
                score: candidate.score,
                detectors: vec![candidate.detector],
                masked_score: candidate.has_mask.then_some(candidate.score),
            }),
        }
    }
    clusters
}

// ---------------------------------------------------------------------------
// Box geometry
// ---------------------------------------------------------------------------

fn area(b: [f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn intersection(a: [f32; 4], b: [f32; 4]) -> f32 {
    let w = a[2].min(b[2]) - a[0].max(b[0]);
    let h = a[3].min(b[3]) - a[1].max(b[1]);
    w.max(0.0) * h.max(0.0)
}

/// Intersection over union.
pub fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let inter = intersection(a, b);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Directional intersection over area: how much of `a` sits inside `b`.
pub fn ioa(a: [f32; 4], b: [f32; 4]) -> f32 {
    let a_area = area(a);
    if a_area <= 0.0 {
        0.0
    } else {
        intersection(a, b) / a_area
    }
}

/// Same-object test: IoU, or either box mostly swallowed by the other.
fn overlaps(a: [f32; 4], b: [f32; 4], config: &FusionConfig) -> bool {
    iou(a, b) >= config.match_iou || ioa(a, b) >= config.match_ioa || ioa(b, a) >= config.match_ioa
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(bbox: [f32; 4], detector: Detector, class: Class, score: f32) -> Candidate {
        Candidate {
            bbox,
            detector,
            class,
            score,
            has_mask: false,
        }
    }

    fn no_mask(_: [f32; 4]) -> f32 {
        0.0
    }

    #[test]
    fn nested_boxes_match_by_ioa_not_iou() {
        let config = FusionConfig::default();
        let outer = [0.0, 0.0, 100.0, 100.0];
        let inner = [10.0, 10.0, 40.0, 40.0];
        // 900 / 10000 area ratio — far below the IoU gate.
        assert!(iou(outer, inner) < config.match_iou);
        assert!(ioa(inner, outer) >= config.match_ioa);
        assert!(overlaps(outer, inner, &config));
    }

    #[test]
    fn two_detectors_accept_a_bubble_one_does_not() {
        let config = FusionConfig::default();
        let agreed = fuse(
            &[
                candidate(
                    [0.0, 0.0, 100.0, 100.0],
                    Detector::Layout,
                    Class::Bubble,
                    0.9,
                ),
                candidate(
                    [4.0, 4.0, 104.0, 104.0],
                    Detector::RtDetr,
                    Class::Bubble,
                    0.7,
                ),
            ],
            &config,
            no_mask,
        );
        assert_eq!(agreed.bubbles.len(), 1, "same balloon, one cluster");
        assert_eq!(agreed.bubbles[0].detectors.len(), 2);
        assert!(agreed.bubbles[0].accepted);
        // Geometry comes from the higher-scoring box, not a union.
        assert_eq!(agreed.bubbles[0].bbox, [0.0, 0.0, 100.0, 100.0]);

        let singleton = fuse(
            &[candidate(
                [0.0, 0.0, 100.0, 100.0],
                Detector::RtDetr,
                Class::Bubble,
                0.9,
            )],
            &config,
            no_mask,
        );
        assert!(!singleton.bubbles[0].accepted, "one vote is review-only");
    }

    #[test]
    fn strong_layout_mask_accepts_a_bubble_alone() {
        let mut only = candidate(
            [0.0, 0.0, 100.0, 100.0],
            Detector::Layout,
            Class::Bubble,
            0.8,
        );
        only.has_mask = true;
        let fusion = fuse(&[only.clone()], &FusionConfig::default(), no_mask);
        assert!(fusion.bubbles[0].accepted);

        only.score = 0.3; // below strong_mask_score
        let weak = fuse(&[only], &FusionConfig::default(), no_mask);
        assert!(!weak.bubbles[0].accepted);
    }

    #[test]
    fn text_in_an_accepted_bubble_is_dialogue() {
        let fusion = fuse(
            &[
                candidate(
                    [0.0, 0.0, 100.0, 100.0],
                    Detector::Layout,
                    Class::Bubble,
                    0.9,
                ),
                candidate(
                    [0.0, 0.0, 100.0, 100.0],
                    Detector::RtDetr,
                    Class::Bubble,
                    0.8,
                ),
                candidate(
                    [20.0, 20.0, 60.0, 60.0],
                    Detector::AnimeText,
                    Class::Text,
                    0.5,
                ),
            ],
            &FusionConfig::default(),
            no_mask,
        );
        let region = &fusion.regions[0];
        assert_eq!(region.role, Role::Dialogue);
        assert_eq!(region.bubble, Some(0));
        // One text detector, but the accepted balloon carries the evidence.
        assert!(region.accepted);
    }

    #[test]
    fn text_in_an_unconfirmed_bubble_stays_for_review() {
        let fusion = fuse(
            &[
                candidate(
                    [0.0, 0.0, 100.0, 100.0],
                    Detector::RtDetr,
                    Class::Bubble,
                    0.9,
                ),
                candidate(
                    [20.0, 20.0, 60.0, 60.0],
                    Detector::AnimeText,
                    Class::Text,
                    0.5,
                ),
            ],
            &FusionConfig::default(),
            no_mask,
        );
        assert_eq!(fusion.regions[0].role, Role::Dialogue);
        assert!(!fusion.regions[0].accepted);
    }

    #[test]
    fn outside_text_needs_two_votes_or_pixels() {
        let config = FusionConfig::default();
        let lone = [500.0, 500.0, 600.0, 540.0];

        let review = fuse(
            &[candidate(lone, Detector::AnimeText, Class::Text, 0.5)],
            &config,
            no_mask,
        );
        assert_eq!(review.regions[0].role, Role::Outside);
        assert!(!review.regions[0].accepted);

        let by_pixels = fuse(
            &[candidate(lone, Detector::AnimeText, Class::Text, 0.5)],
            &config,
            |_| 0.4,
        );
        assert!(by_pixels.regions[0].accepted);

        let by_votes = fuse(
            &[
                candidate(lone, Detector::AnimeText, Class::Text, 0.5),
                candidate(lone, Detector::ComicTextDetector, Class::Text, 0.4),
            ],
            &config,
            no_mask,
        );
        assert_eq!(by_votes.regions.len(), 1);
        assert!(by_votes.regions[0].accepted);
    }

    #[test]
    fn sfx_needs_a_mask_and_glyph_confirmation() {
        let config = FusionConfig::default();
        let sfx = [300.0, 300.0, 460.0, 380.0];
        let masked = || {
            let mut c = candidate(sfx, Detector::Layout, Class::Sfx, 0.45);
            c.has_mask = true;
            c
        };
        let sfx_region = |fusion: Fusion| {
            fusion
                .regions
                .into_iter()
                .find(|r| r.role == Role::Sfx)
                .expect("SFX region")
        };

        let bare = fuse(
            &[candidate(sfx, Detector::Layout, Class::Sfx, 0.45)],
            &config,
            no_mask,
        );
        assert_eq!(bare.regions[0].role, Role::Sfx);
        assert!(
            !bare.regions[0].accepted,
            "raw SFX rectangles are proposals"
        );

        // An instance mask with nothing to corroborate it is still one opinion.
        assert!(!sfx_region(fuse(&[masked()], &config, no_mask)).accepted);
        // Glyph evidence over a rectangle nobody segmented, likewise.
        assert!(
            !sfx_region(fuse(
                &[candidate(sfx, Detector::Layout, Class::Sfx, 0.45)],
                &config,
                |_| 0.3
            ))
            .accepted
        );

        // Mask plus the pixel text mask.
        assert!(sfx_region(fuse(&[masked()], &config, |_| 0.3)).accepted);

        // Mask plus another detector's text box over the same rectangle.
        for detector in [Detector::AnimeText, Detector::ComicTextDetector] {
            let confirmed = sfx_region(fuse(
                &[
                    masked(),
                    candidate([310.0, 310.0, 450.0, 370.0], detector, Class::Text, 0.5),
                ],
                &config,
                no_mask,
            ));
            assert!(confirmed.accepted, "{detector:?} should confirm");
        }

        // …but the layout model cannot confirm its own SFX box by also calling
        // it text.
        let self_confirmed = sfx_region(fuse(
            &[
                masked(),
                candidate(
                    [310.0, 310.0, 450.0, 370.0],
                    Detector::Layout,
                    Class::Text,
                    0.5,
                ),
            ],
            &config,
            no_mask,
        ));
        assert!(!self_confirmed.accepted);
    }
}
