use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use obj_viewer::app;

/// Simple OBJ mesh viewer with orbit camera
#[derive(Parser, Debug)]
#[command(name = "obj_viewer")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the OBJ file to view
    #[arg(required = true)]
    obj_path: PathBuf,

    /// Window width in pixels
    #[arg(short = 'W', long, default_value = "1280")]
    width: u32,

    /// Window height in pixels
    #[arg(short = 'H', long, default_value = "720")]
    height: u32,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args = Args::parse();

    // Validate OBJ file exists
    if !args.obj_path.exists() {
        anyhow::bail!("OBJ file not found: {:?}", args.obj_path);
    }

    if !args.obj_path.is_file() {
        anyhow::bail!("Path is not a file: {:?}", args.obj_path);
    }

    log::info!("Starting OBJ Viewer");
    log::info!("  File: {:?}", args.obj_path);
    log::info!("  Window: {}x{}", args.width, args.height);

    // Run application
    app::run(args.obj_path, args.width, args.height)
}
