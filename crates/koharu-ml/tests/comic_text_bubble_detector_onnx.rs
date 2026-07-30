#![cfg(feature = "onnx")]

//! ONNX-vs-candle spike check for the RT-DETRv2 comic text/bubble detector.
//! Asserts the two backends agree on text blocks, and prints both wall times.

use std::path::Path;
use std::time::Instant;

use koharu_ml::comic_text_bubble_detector::ComicTextBubbleDetector;

mod support;

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area = |r: &[f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[tokio::test]
#[ignore = "requires model downloads (safetensors + 168MB onnx); spike benchmark"]
async fn onnx_matches_candle() -> anyhow::Result<()> {
    // Surfaces the execution-provider fallback warning under `--nocapture`.
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    let runtime = support::cpu_runtime();
    let image = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/1.jpg"))?;

    let candle = ComicTextBubbleDetector::load(&runtime, true).await?;
    let started = Instant::now();
    let candle_out = candle.inference(&image)?;
    let candle_ms = started.elapsed().as_millis();

    let onnx = ComicTextBubbleDetector::load_onnx(&runtime, true).await?;
    // First run includes graph warmup, so time the second.
    let _ = onnx.inference(&image)?;
    let started = Instant::now();
    let onnx_out = onnx.inference(&image)?;
    let onnx_ms = started.elapsed().as_millis();

    println!(
        "candle: {} regions / {} blocks in {candle_ms}ms | onnx: {} regions / {} blocks in {onnx_ms}ms",
        candle_out.detections.len(),
        candle_out.text_blocks.len(),
        onnx_out.detections.len(),
        onnx_out.text_blocks.len(),
    );

    assert!(
        !onnx_out.detections.is_empty() && !onnx_out.text_blocks.is_empty(),
        "ONNX backend produced nothing"
    );

    // Every candle text block should have a closely matching ONNX block.
    let candle_boxes: Vec<[f32; 4]> = candle_out
        .text_blocks
        .iter()
        .map(|b| [b.x, b.y, b.x + b.width, b.y + b.height])
        .collect();
    let onnx_boxes: Vec<[f32; 4]> = onnx_out
        .text_blocks
        .iter()
        .map(|b| [b.x, b.y, b.x + b.width, b.y + b.height])
        .collect();

    let mut worst = 1.0f32;
    for candle_box in &candle_boxes {
        let best = onnx_boxes
            .iter()
            .map(|onnx_box| iou(candle_box, onnx_box))
            .fold(0.0f32, f32::max);
        worst = worst.min(best);
    }
    println!("worst per-block IoU: {worst:.3}");
    assert!(
        worst > 0.8,
        "backends disagree: worst candle->onnx text block IoU {worst:.3}"
    );

    // Accelerated load must never hard-fail: CoreML cannot execute this graph,
    // so the loader has to notice and fall back to CPU.
    let accelerated = ComicTextBubbleDetector::load_onnx(&runtime, false).await?;
    let started = Instant::now();
    let accelerated_out = accelerated.inference(&image)?;
    println!(
        "onnx accelerated: {} blocks in {}ms",
        accelerated_out.text_blocks.len(),
        started.elapsed().as_millis()
    );
    assert_eq!(
        accelerated_out.text_blocks.len(),
        onnx_out.text_blocks.len(),
        "accelerated ONNX load produced a different block count than the CPU session"
    );

    Ok(())
}
