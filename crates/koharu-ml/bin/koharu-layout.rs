use anyhow::{Result, anyhow};
use clap::Parser;
use image::{Rgba, RgbaImage};
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use koharu_ml::koharu_layout::{KoharuLayoutDetector, LayoutRegion};
use koharu_runtime::{ComputePolicy, RuntimeManager, default_app_data_root};
use tokio::runtime::Builder;

#[path = "common.rs"]
mod common;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: String,

    /// Score floor for classes the export lists no recommended threshold for.
    #[arg(long, default_value_t = 0.3)]
    threshold: f32,

    #[arg(long, value_name = "FILE")]
    output: Option<String>,

    /// Page with per-class boxes and translucent instance masks drawn on top.
    #[arg(long, value_name = "FILE")]
    annotated_output: Option<String>,

    /// Skip mask decoding (boxes only) — the mask head is the expensive part.
    #[arg(long, default_value_t = false)]
    no_masks: bool,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

/// One colour per class id, in the export's class order.
fn color_for_label(label_id: usize) -> Rgba<u8> {
    match label_id {
        0 => Rgba([0, 160, 255, 255]),  // text
        1 => Rgba([255, 64, 200, 255]), // onomatopoeia
        2 => Rgba([0, 255, 0, 255]),    // bubble
        3 => Rgba([255, 176, 0, 255]),  // panel
        _ => Rgba([255, 255, 255, 255]),
    }
}

fn stroke_radius(width: u32, height: u32) -> i32 {
    ((width.max(height) as f32 / 1800.0).round() as i32).clamp(1, 8)
}

fn draw_thick_rect(image: &mut RgbaImage, rect: Rect, color: Rgba<u8>, radius: i32) {
    for offset in -radius..=radius {
        draw_hollow_rect_mut(
            image,
            Rect::at(rect.left() + offset, rect.top() + offset)
                .of_size(rect.width(), rect.height()),
            color,
        );
    }
}

fn blend_mask(image: &mut RgbaImage, region: &LayoutRegion, color: Rgba<u8>) {
    let Some(mask) = region.mask.as_ref() else {
        return;
    };
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if x >= mask.width() || y >= mask.height() || mask.get_pixel(x, y).0[0] == 0 {
            continue;
        }
        for channel in 0..3 {
            // 40% of the class colour over the page.
            pixel.0[channel] =
                ((pixel.0[channel] as u32 * 3 + color.0[channel] as u32 * 2) / 5).min(255) as u8;
        }
    }
}

fn main() -> Result<()> {
    common::init_tracing();

    std::thread::Builder::new()
        .name("koharu-layout".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let runtime = Builder::new_current_thread().enable_all().build()?;
            runtime.block_on(async_main())
        })?
        .join()
        .map_err(|_| anyhow!("koharu-layout thread panicked"))?
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();
    let runtime = RuntimeManager::new(
        default_app_data_root(),
        if cli.cpu {
            ComputePolicy::CpuOnly
        } else {
            ComputePolicy::PreferGpu
        },
    )?;
    runtime.prepare().await?;

    let model = KoharuLayoutDetector::load(&runtime, cli.cpu).await?;
    let bytes = std::fs::read(&cli.input)?;
    let format = image::guess_format(&bytes)?;
    let image = image::load_from_memory_with_format(&bytes, format)?;

    let want_masks = !cli.no_masks;
    let started = std::time::Instant::now();
    let detection = model.inference_with_threshold(&image, cli.threshold, want_masks)?;
    let elapsed_ms = started.elapsed().as_millis();

    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for region in &detection.regions {
        *counts.entry(region.label.as_str()).or_default() += 1;
    }
    tracing::info!(
        regions = detection.regions.len(),
        total_ms = elapsed_ms,
        counts = ?counts,
        "koharu layout"
    );

    if let Some(path) = &cli.annotated_output {
        let mut annotated = image.to_rgba8();
        let radius = stroke_radius(annotated.width(), annotated.height());
        for region in &detection.regions {
            blend_mask(&mut annotated, region, color_for_label(region.label_id));
        }
        for region in &detection.regions {
            let [x1, y1, x2, y2] = region.bbox;
            draw_thick_rect(
                &mut annotated,
                Rect::at(x1 as i32, y1 as i32)
                    .of_size((x2 - x1).max(1.0) as u32, (y2 - y1).max(1.0) as u32),
                color_for_label(region.label_id),
                radius,
            );
        }
        image::DynamicImage::ImageRgba8(annotated).save(path)?;
        println!("legend: text=blue onomatopoeia=pink bubble=green panel=orange");
    }

    let json = serde_json::to_string_pretty(&detection)?;
    if let Some(output) = cli.output {
        std::fs::write(output, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}
