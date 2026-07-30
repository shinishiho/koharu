use anyhow::{Result, anyhow};
use clap::Parser;
use image::{Rgba, RgbaImage};
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use koharu_ml::koharu_yolo26s::{
    DEFAULT_CONFIDENCE_THRESHOLD, KoharuYolo26sDetector, Yolo26sClass, Yolo26sRegion,
};
use koharu_runtime::{ComputePolicy, RuntimeManager, default_app_data_root};
use tokio::runtime::Builder;

#[path = "common.rs"]
mod common;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: String,

    #[arg(long, default_value_t = DEFAULT_CONFIDENCE_THRESHOLD)]
    confidence_threshold: f32,

    /// Detection JSON. Printed to stdout when omitted.
    #[arg(long, value_name = "FILE")]
    output: Option<String>,

    /// Page with boxes and translucent instance masks drawn on top.
    #[arg(long, value_name = "FILE")]
    annotated_output: Option<String>,

    /// Skip mask decoding, boxes only.
    #[arg(long, default_value_t = false)]
    no_masks: bool,

    /// Load a local `.onnx` instead of fetching the published export.
    #[arg(long, value_name = "FILE")]
    model: Option<String>,
}

fn color_for(class: Yolo26sClass) -> Rgba<u8> {
    match class {
        Yolo26sClass::Frame => Rgba([255, 176, 0, 255]),
        Yolo26sClass::DialogueText => Rgba([0, 160, 255, 255]),
        Yolo26sClass::Balloon => Rgba([0, 255, 0, 255]),
        Yolo26sClass::OnomatopoeiaText => Rgba([255, 64, 200, 255]),
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

fn blend_mask(image: &mut RgbaImage, region: &Yolo26sRegion, color: Rgba<u8>) {
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
        .name("koharu-yolo26s".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let runtime = Builder::new_current_thread().enable_all().build()?;
            runtime.block_on(async_main())
        })?
        .join()
        .map_err(|_| anyhow!("koharu-yolo26s thread panicked"))?
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();

    // The detector pins the CPU execution provider, so there is no GPU to ask
    // the runtime for.
    let runtime = RuntimeManager::new(default_app_data_root(), ComputePolicy::CpuOnly)?;
    runtime.prepare().await?;

    let model = match &cli.model {
        Some(path) => KoharuYolo26sDetector::load_from_path(std::path::Path::new(path))?,
        None => KoharuYolo26sDetector::load(&runtime).await?,
    };

    let bytes = std::fs::read(&cli.input)?;
    let format = image::guess_format(&bytes)?;
    let image = image::load_from_memory_with_format(&bytes, format)?;

    let started = std::time::Instant::now();
    let detection =
        model.inference_with_threshold(&image, cli.confidence_threshold, !cli.no_masks)?;
    let elapsed_ms = started.elapsed().as_millis();

    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for region in &detection.regions {
        *counts.entry(region.class.as_str()).or_default() += 1;
    }
    tracing::info!(
        regions = detection.regions.len(),
        total_ms = elapsed_ms,
        counts = ?counts,
        "koharu-yolo26s"
    );

    if let Some(path) = &cli.annotated_output {
        let mut annotated = image.to_rgba8();
        let radius = stroke_radius(annotated.width(), annotated.height());
        for region in &detection.regions {
            blend_mask(&mut annotated, region, color_for(region.class));
        }
        for region in &detection.regions {
            let [x1, y1, x2, y2] = region.bbox;
            draw_thick_rect(
                &mut annotated,
                Rect::at(x1 as i32, y1 as i32)
                    .of_size((x2 - x1).max(1.0) as u32, (y2 - y1).max(1.0) as u32),
                color_for(region.class),
                radius,
            );
        }
        image::DynamicImage::ImageRgba8(annotated).save(path)?;
        println!("legend: frame=orange dialogue_text=blue balloon=green onomatopoeia_text=pink");
    }

    let json = serde_json::to_string_pretty(&detection)?;
    match cli.output {
        Some(path) => std::fs::write(path, json)?,
        None => println!("{json}"),
    }
    Ok(())
}
