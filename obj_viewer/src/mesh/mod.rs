pub mod geometry;
pub mod loader;

pub use geometry::ProcessedMesh;
pub use loader::{parse_obj_str, RawMesh};

#[cfg(not(target_arch = "wasm32"))]
pub use loader::load_obj;
