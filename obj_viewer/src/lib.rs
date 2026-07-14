pub mod camera;
pub mod input;
pub mod mesh;
pub mod metadata;
pub mod renderer;

#[cfg(not(target_arch = "wasm32"))]
pub mod app;

#[cfg(target_arch = "wasm32")]
pub mod web;
