# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A dual-target 3D OBJ mesh viewer built with Rust, wgpu, and winit. The application compiles to both native desktop binary and WebAssembly for browser deployment. Features an orbit camera system with mouse controls, Blinn-Phong shading, and IFC metadata integration for BIM workflows. The web version includes a Discovery mode for browsing objects by IFC type.

## Build and Run Commands

### Native Desktop Application
```bash
# Build and run (requires an OBJ file path)
cargo run --release -- path/to/model.obj

# With custom window size
cargo run --release -- path/to/model.obj --width 1920 --height 1080

# Build only
cargo build --release

# Run tests
cargo test
```

### WebAssembly Build
```bash
# Build for web (requires wasm-pack)
./build-web.sh
# or manually:
wasm-pack build --target web --out-dir web/pkg

# Serve locally
python3 serve.py
# Then visit http://localhost:8080
```

The `serve.py` script is required for local testing as it provides proper MIME types for WASM and sets necessary CORS headers for SharedArrayBuffer.

## Architecture

### Dual-Target Compilation Strategy

The codebase uses Rust's conditional compilation to maintain separate entry points and platform-specific code:

- **Native (`cfg(not(target_arch = "wasm32"))`)**: Uses `main.rs` with synchronous file loading via `pollster::block_on` and CLI argument parsing with `clap`
- **WASM (`cfg(target_arch = "wasm32")`)**: Uses `lib.rs` + `web.rs` with async initialization, JavaScript interop via `wasm-bindgen`, and browser file loading

Key architectural pattern: Core modules (`renderer`, `camera`, `mesh`, `input`) are platform-agnostic, with only the application shell differing between targets.

### Module Structure

**`app.rs`** (native only)
Entry point implementing winit's `ApplicationHandler`. Manages the event loop, window lifecycle, and synchronous mesh loading from filesystem.

**`web.rs`** (WASM only)
Browser entry point exposing `init_viewer()` and `ViewerHandle` to JavaScript. Handles async renderer initialization and file loading from `FileReader` API.

**`renderer/`**
- `state.rs`: Core `Renderer` struct managing wgpu device, surface, pipeline, and buffers. Async `new()` for both platforms, sync `render()` and `resize()` methods.
- `vertex.rs`: `Vertex` struct with `[repr(C)]` for GPU layout, includes position and normal attributes.

**`camera/`**
- `orbit.rs`: `OrbitCamera` with spherical coordinates (azimuth, elevation, distance). Methods: `fit_to_mesh()`, `reset()`, `rotate()`, `zoom()`. Computes view matrix from spherical position.
- `uniform.rs`: `CameraUniform` matching WGSL struct layout, contains view-projection matrix, camera position, and light position.

**`mesh/`**
- `loader.rs`: OBJ parser (`parse_obj_str()`) for v/vn/f lines. Native uses `load_obj()` with `std::fs`, WASM skips filesystem code.
- `geometry.rs`: `ProcessedMesh` computes vertex normals (averaged face normals), bounding sphere center, and radius for camera fitting.

**`input.rs`**
Stateful input handler tracking mouse button states and previous cursor position. Converts winit events to camera transformations (rotation on drag, zoom on scroll). Returns `KeyAction` enum for R (reset) and Esc (quit).

**`metadata.rs`**
IFC metadata management for BIM workflows. Parses JSON arrays containing IFC object metadata (GlobalId, IfcType, mesh_filename). Provides efficient indexing by filename and IFC type for Discovery mode. Uses serde for JSON deserialization in WASM target.

**`shaders/shader.wgsl`**
WGSL shader implementing Blinn-Phong lighting with ambient (15%), diffuse (Lambertian), and specular components. Material properties hardcoded in fragment shader.

### WebGPU Rendering Pipeline

The renderer follows this initialization sequence:
1. Request wgpu adapter and device (async)
2. Configure surface with BGRA8Unorm format and Fifo present mode
3. Compile WGSL shader and create render pipeline with depth testing
4. Create camera uniform buffer (bind group 0, binding 0)
5. Upload mesh vertex/index buffers

Render loop:
1. Update camera uniform buffer with current view-projection matrix
2. Acquire surface texture
3. Begin render pass with depth buffer clear
4. Draw indexed mesh
5. Submit command buffer and present

### Mesh Processing

OBJ files are parsed to extract:
- Vertices (v x y z)
- Normals (vn x y z) - if not present, computed per-face then averaged per-vertex
- Faces (f v//vn v//vn v//vn) - triangulated if >3 vertices

Bounding sphere calculation uses centroid as center and max vertex distance as radius for camera auto-framing.

### IFC Metadata and Discovery Mode (WASM only)

The web viewer supports loading IFC metadata JSON files to associate mesh files with BIM metadata. This enables Discovery mode for browsing objects by type.

**Metadata JSON Format:**
```json
[
  {
    "relative_source_path": "path/to/source.ifc",
    "GlobalId": "3AqJtUVg1CMfnOYWdS2QwN",
    "IfcType": "IfcBeam",
    "mesh_filename": "3AqJtUVg1CMfnOYWdS2QwN.obj"
  }
]
```

**Metadata Management:**
- `MetadataManager` maintains two HashMaps: filename → metadata and IFC type → list of metadata
- Provides O(1) lookup by filename for current object metadata display
- Provides efficient querying of all objects of a specific IFC type for Discovery mode

**Discovery Mode Workflow:**
1. User loads metadata.json via "Load Metadata" button
2. Discovery tab becomes enabled
3. User switches to Discovery tab and clicks "Load Mesh Folder"
4. User selects a folder containing OBJ files (folder is indexed, files are NOT loaded yet)
5. User selects an IFC type from dropdown (shows counts, e.g., "IfcBeam (25)")
6. List displays all objects of that type with availability indicators:
   - ✓ (blue) = mesh file found in selected folder
   - ✗ (red) = mesh file not in selected folder (grayed out)
7. User can navigate through available objects using:
   - Arrow buttons (automatically skips unavailable meshes)
   - Keyboard left/right arrows
   - Clicking on an available object in the list
8. When object is selected, the specific mesh file is loaded on-demand from the folder

**Current Object Metadata Display:**
When metadata is loaded and current mesh filename matches an entry, the status bar displays IFC type badge and GlobalId.

**Technical Implementation:**
- Mesh folder selection uses HTML5 File API with `webkitdirectory` attribute for folder selection
- Only File object references are stored in a JavaScript Map (filename → File reference)
- Individual mesh files are read on-demand when user selects an object to view
- Navigation automatically skips unavailable meshes by checking Map keys before loading
- Availability indicators (✓/✗) update instantly when folder is selected
- No server-side storage required - entire workflow runs client-side
- Efficient memory usage: only one mesh loaded at a time

## Dependencies

**Core Graphics**:
- `wgpu`: Cross-platform WebGPU implementation
- `winit`: Window and event handling
- `glam`: SIMD math library for vectors/matrices (with bytemuck integration)

**Platform-Specific**:
- Native: `pollster` (block on async), `clap` (CLI), `env_logger`
- WASM: `wasm-bindgen`, `web-sys`, `js-sys`, `serde`, `serde_json`, `serde-wasm-bindgen`, `console_log`, `console_error_panic_hook`

## Controls

**Mouse**:
- Left-click drag: Orbit camera (azimuth/elevation)
- Scroll wheel: Zoom in/out (distance)

**Keyboard**:
- `R`: Reset camera to fit mesh
- `Esc`: Quit (native only)
- `Left Arrow`: Previous object in Discovery mode (WASM only)
- `Right Arrow`: Next object in Discovery mode (WASM only)

## Development Notes

- Renderer uses depth testing (DepthStencilState with Depth32Float) for correct triangle ordering
- Camera light position follows camera for consistent illumination
- WASM build requires `wasm-pack`, installable via `cargo install wasm-pack`
- WebGPU requires modern browsers: Chrome/Edge 113+, Firefox 126+
- The project has no test files currently - add tests to `src/` modules or `tests/` directory

**Web UI Features:**
- Tab-based interface: Viewer (default) and Discovery modes
- Drag-and-drop support for OBJ files in Viewer mode
- Metadata JSON upload for IFC integration
- Folder selection for mesh path indexing (webkitdirectory)
- Real-time IFC type and GlobalId display in status bar
- Discovery panel with:
  - Mesh folder selection button (indexes available files)
  - IFC type selector with object counts
  - Navigation controls (prev/next with auto-skip unavailable)
  - List view showing availability status (✓/✗ indicators)
  - Click-to-load functionality for available meshes
- On-demand file loading: meshes load only when selected (efficient memory usage)
