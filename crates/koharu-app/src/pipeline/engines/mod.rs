//! Concrete engine implementations.
//!
//! Each engine lives in its own file and registers itself via
//! `inventory::submit! { EngineInfo { … } }`. The registry picks them up
//! automatically at link time.

pub mod aot;
pub mod bubble_segmentation;
pub mod ctd_segment;
pub mod flux2_klein;
pub mod koharu_yolo26s;
pub mod lama;
pub mod llm_translate;
pub mod manga_ocr;
pub mod mit48px_ocr;
pub mod paddle_ocr;
pub mod renderer;
pub mod support;
pub mod yuzumarker_font;
