use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::OrbitCamera;
use crate::input::{InputState, KeyAction};
use crate::mesh::{parse_obj_str, ProcessedMesh};
use crate::metadata::{MetadataManager, ObjectMetadata};
use crate::renderer::Renderer;

/// WASM entry point - called when module loads
#[wasm_bindgen(start)]
pub fn wasm_main() {
    // Initialize panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize console logging
    console_log::init_with_level(log::Level::Info).expect("Failed to init logger");

    log::info!("OBJ Viewer WASM module loaded");
}

/// Shared application state for web
struct WebAppState {
    window: Arc<Window>,
    renderer: Option<Renderer>,
    camera: OrbitCamera,
    input_state: InputState,
    mesh: Option<ProcessedMesh>,
    metadata: Option<MetadataManager>,
    current_filename: Option<String>,
}

/// Web application handler
pub struct WebApp {
    state: Rc<RefCell<Option<WebAppState>>>,
    canvas_id: String,
}

impl WebApp {
    pub fn new(canvas_id: String) -> Self {
        Self {
            state: Rc::new(RefCell::new(None)),
            canvas_id,
        }
    }
}

impl ApplicationHandler for WebApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.borrow().is_some() {
            return; // Already initialized
        }

        // Get canvas element
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document
            .get_element_by_id(&self.canvas_id)
            .expect("Canvas not found")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Element is not a canvas");

        let window_attrs = WindowAttributes::default().with_canvas(Some(canvas));

        match event_loop.create_window(window_attrs) {
            Ok(window) => {
                let window = Arc::new(window);
                *self.state.borrow_mut() = Some(WebAppState {
                    window,
                    renderer: None,
                    camera: OrbitCamera::default(),
                    input_state: InputState::new(),
                    mesh: None,
                    metadata: None,
                    current_filename: None,
                });
                log::info!("Window created successfully");
            }
            Err(e) => {
                log::error!("Failed to create window: {}", e);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let mut state_ref = self.state.borrow_mut();
        let state = match state_ref.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut state.renderer {
                    renderer.resize(physical_size);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut state.renderer {
                    match renderer.render(&state.camera) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let size =
                                PhysicalSize::new(renderer.config.width, renderer.config.height);
                            renderer.resize(size);
                        }
                        Err(e) => {
                            log::error!("Render error: {:?}", e);
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                state.input_state.handle_mouse_button(button, btn_state);
            }

            WindowEvent::CursorMoved { position, .. } => {
                state
                    .input_state
                    .handle_cursor_moved((position.x, position.y), &mut state.camera);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                state.input_state.handle_scroll(delta, &mut state.camera);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(action) = state.input_state.handle_keyboard(&event) {
                    match action {
                        KeyAction::Quit => {
                            // Can't quit in web, just ignore
                        }
                        KeyAction::ResetCamera => {
                            if let Some(mesh) = &state.mesh {
                                state.camera.reset(mesh.center, mesh.radius);
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
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.borrow().as_ref() {
            state.window.request_redraw();
        }
    }
}

/// Handle to the running viewer, exposed to JavaScript
#[wasm_bindgen]
pub struct ViewerHandle {
    state: Rc<RefCell<Option<WebAppState>>>,
}

/// Metadata information for JavaScript
#[wasm_bindgen]
#[derive(Clone, serde::Serialize)]
pub struct MetadataInfo {
    global_id: String,
    ifc_type: String,
    mesh_filename: String,
}

#[wasm_bindgen]
impl MetadataInfo {
    #[wasm_bindgen(getter)]
    pub fn global_id(&self) -> String {
        self.global_id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ifc_type(&self) -> String {
        self.ifc_type.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn mesh_filename(&self) -> String {
        self.mesh_filename.clone()
    }
}

impl MetadataInfo {
    fn from_metadata(meta: &ObjectMetadata) -> Self {
        Self {
            global_id: meta.global_id.clone(),
            ifc_type: meta.ifc_type.clone().unwrap_or_default(),
            mesh_filename: meta.mesh_filename.clone(),
        }
    }
}

#[wasm_bindgen]
impl ViewerHandle {
    /// Load an OBJ file from string content with optional filename for metadata tracking
    pub fn load_obj(&self, content: &str, filename: Option<String>) -> Result<(), JsValue> {
        let mut state_ref = self.state.borrow_mut();
        let state = state_ref
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Viewer not initialized"))?;

        // Parse OBJ
        let raw_mesh = parse_obj_str(content)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse OBJ: {}", e)))?;

        let processed_mesh = ProcessedMesh::from_raw(&raw_mesh);

        // Fit camera to mesh
        state.camera.fit_to_mesh(processed_mesh.center, processed_mesh.radius);

        // Update or create renderer
        if let Some(renderer) = &mut state.renderer {
            renderer.update_mesh(&processed_mesh);
        } else {
            // First mesh load - create renderer
            let window = state.window.clone();
            let mesh = processed_mesh.clone();

            // We need to initialize renderer asynchronously
            let state_clone = self.state.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match Renderer::new(window, &mesh).await {
                    Ok(renderer) => {
                        if let Some(s) = state_clone.borrow_mut().as_mut() {
                            s.renderer = Some(renderer);
                            s.mesh = Some(mesh);
                            log::info!("Renderer initialized");
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to create renderer: {}", e);
                    }
                }
            });
            return Ok(());
        }

        state.mesh = Some(processed_mesh);
        state.current_filename = filename;
        log::info!("Mesh loaded successfully");
        Ok(())
    }

    /// Load metadata from JSON string
    pub fn load_metadata(&self, json: &str) -> Result<(), JsValue> {
        let manager = MetadataManager::from_json(json)
            .map_err(|e| JsValue::from_str(&e))?;

        let mut state_ref = self.state.borrow_mut();
        if let Some(state) = state_ref.as_mut() {
            log::info!("Loaded metadata for {} objects", manager.total_count());
            state.metadata = Some(manager);
        }

        Ok(())
    }

    /// Get metadata for the currently loaded object
    pub fn get_current_metadata(&self) -> Option<MetadataInfo> {
        let state_ref = self.state.borrow();
        let state = state_ref.as_ref()?;

        let filename = state.current_filename.as_ref()?;
        let manager = state.metadata.as_ref()?;

        manager
            .get_by_filename(filename)
            .map(MetadataInfo::from_metadata)
    }

    /// Get all available IFC types as JSON array
    pub fn get_ifc_types(&self) -> JsValue {
        let state_ref = self.state.borrow();
        if let Some(state) = state_ref.as_ref() {
            if let Some(manager) = &state.metadata {
                let types = manager.get_ifc_types();
                return serde_wasm_bindgen::to_value(&types).unwrap_or(JsValue::NULL);
            }
        }
        JsValue::NULL
    }

    /// Get count of objects for a specific IFC type
    pub fn get_type_count(&self, ifc_type: &str) -> usize {
        let state_ref = self.state.borrow();
        if let Some(state) = state_ref.as_ref() {
            if let Some(manager) = &state.metadata {
                return manager.count_by_type(ifc_type);
            }
        }
        0
    }

    /// Get metadata for all objects of a specific type as JSON array
    pub fn get_objects_by_type(&self, ifc_type: &str) -> JsValue {
        let state_ref = self.state.borrow();
        if let Some(state) = state_ref.as_ref() {
            if let Some(manager) = &state.metadata {
                let objects: Vec<MetadataInfo> = manager
                    .get_by_type(ifc_type)
                    .into_iter()
                    .map(MetadataInfo::from_metadata)
                    .collect();
                return serde_wasm_bindgen::to_value(&objects).unwrap_or(JsValue::NULL);
            }
        }
        JsValue::NULL
    }

    /// Reset camera to fit the current mesh
    pub fn reset_camera(&self) {
        if let Some(state) = self.state.borrow_mut().as_mut() {
            if let Some(mesh) = &state.mesh {
                state.camera.reset(mesh.center, mesh.radius);
            }
        }
    }
}

/// Initialize the viewer on a canvas element
#[wasm_bindgen]
pub async fn init_viewer(canvas_id: &str) -> Result<ViewerHandle, JsValue> {
    let app = WebApp::new(canvas_id.to_string());
    let state = app.state.clone();

    let event_loop = EventLoop::new().map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use spawn to run the event loop (required for web)
    use winit::platform::web::EventLoopExtWebSys;
    event_loop.spawn_app(app);

    log::info!("Viewer initialized on canvas: {}", canvas_id);

    Ok(ViewerHandle { state })
}
