use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::OrbitCamera;
use crate::input::{InputState, KeyAction};
use crate::mesh::{load_obj, ProcessedMesh};
use crate::renderer::Renderer;

/// Application state
pub struct App {
    /// Window handle (initialized on resume)
    window: Option<Arc<Window>>,
    /// wgpu renderer (initialized after window)
    renderer: Option<Renderer>,
    /// Orbit camera
    camera: OrbitCamera,
    /// Input state tracker
    input_state: InputState,
    /// Loaded mesh data
    mesh: Option<ProcessedMesh>,
    /// Path to OBJ file
    obj_path: PathBuf,
    /// Initial window size
    initial_size: PhysicalSize<u32>,
}

impl App {
    pub fn new(obj_path: PathBuf, width: u32, height: u32) -> Self {
        Self {
            window: None,
            renderer: None,
            camera: OrbitCamera::default(),
            input_state: InputState::new(),
            mesh: None,
            obj_path,
            initial_size: PhysicalSize::new(width, height),
        }
    }

    /// Initialize renderer with loaded mesh
    fn initialize(&mut self, window: Arc<Window>) {
        // Load OBJ file
        log::info!("Loading OBJ: {:?}", self.obj_path);
        let raw_mesh = match load_obj(&self.obj_path) {
            Ok(mesh) => mesh,
            Err(e) => {
                log::error!("Failed to load OBJ: {}", e);
                return;
            }
        };

        // Process mesh (compute normals, bounds)
        let processed_mesh = ProcessedMesh::from_raw(&raw_mesh);

        // Fit camera to mesh
        self.camera.fit_to_mesh(processed_mesh.center, processed_mesh.radius);

        // Create renderer
        let renderer = pollster::block_on(Renderer::new(window.clone(), &processed_mesh));
        match renderer {
            Ok(r) => {
                self.renderer = Some(r);
                log::info!("Renderer initialized successfully");
            }
            Err(e) => {
                log::error!("Failed to create renderer: {}", e);
            }
        }

        self.mesh = Some(processed_mesh);
        self.window = Some(window);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // Already initialized
        }

        // Create window
        let window_attrs = WindowAttributes::default()
            .with_title("OBJ Viewer")
            .with_inner_size(self.initial_size);

        match event_loop.create_window(window_attrs) {
            Ok(window) => {
                let window = Arc::new(window);
                self.initialize(window);
            }
            Err(e) => {
                log::error!("Failed to create window: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, exiting...");
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(physical_size);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    match renderer.render(&self.camera) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let size = PhysicalSize::new(renderer.config.width, renderer.config.height);
                            renderer.resize(size);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of memory!");
                            event_loop.exit();
                        }
                        Err(e) => {
                            log::error!("Render error: {:?}", e);
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                self.input_state.handle_mouse_button(button, state);
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.input_state
                    .handle_cursor_moved((position.x, position.y), &mut self.camera);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                self.input_state.handle_scroll(delta, &mut self.camera);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(action) = self.input_state.handle_keyboard(&event) {
                    match action {
                        KeyAction::Quit => {
                            log::info!("Quit requested");
                            event_loop.exit();
                        }
                        KeyAction::ResetCamera => {
                            if let Some(mesh) = &self.mesh {
                                self.camera.reset(mesh.center, mesh.radius);
                                log::info!("Camera reset");
                            }
                        }
                    }
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        _event: DeviceEvent,
    ) {
        // Could handle raw mouse motion here for smoother camera control
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Request continuous redraw
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

/// Run the application
pub fn run(obj_path: PathBuf, width: u32, height: u32) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App::new(obj_path, width, height);
    event_loop.run_app(&mut app)?;
    Ok(())
}
