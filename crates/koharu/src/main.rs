#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use koharu::app;
use koharu::panic;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    panic::install();
    app::run().await
}
