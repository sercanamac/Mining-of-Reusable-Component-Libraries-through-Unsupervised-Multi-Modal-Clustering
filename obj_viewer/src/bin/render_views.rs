use std::path::PathBuf;

use anyhow::Result;
use bytemuck;
use clap::Parser;
use glam::Vec3;
use wgpu::util::DeviceExt;

use obj_viewer::camera::{CameraUniform, OrbitCamera};
use obj_viewer::mesh::{load_obj, ProcessedMesh};
use obj_viewer::renderer::Vertex;

/// Render an OBJ mesh from multiple viewpoints and save as PNG images
#[derive(Parser, Debug)]
#[command(name = "render_views")]
struct Args {
    /// Path to the OBJ file
    #[arg(required = true)]
    obj_path: PathBuf,

    /// Output directory for rendered PNGs
    #[arg(short, long, default_value = "renders")]
    output: PathBuf,

    /// Image width
    #[arg(long, default_value = "800")]
    width: u32,

    /// Image height
    #[arg(long, default_value = "600")]
    height: u32,

    /// Uniform material color as R,G,B floats (0.0-1.0), e.g. "0.8,0.2,0.2"
    /// Overrides vertex colors entirely.
    #[arg(long)]
    color: Option<String>,

    /// Path to per-vertex color file (binary: n_vertices × 3 × f32 RGB).
    /// Takes precedence over default vertex colors but not --color.
    #[arg(long)]
    vertex_colors: Option<PathBuf>,

    /// Context OBJ files (surrounding objects, rendered semi-transparent)
    #[arg(long)]
    context: Vec<PathBuf>,

    /// Color for the target mesh as R,G,B (default: "0.2,0.5,0.9")
    #[arg(long, default_value = "0.2,0.5,0.9")]
    target_color: String,

    /// Color for context meshes as R,G,B (default: "0.5,0.5,0.5")
    #[arg(long, default_value = "0.5,0.5,0.5")]
    context_color: String,

    /// Opacity for context meshes (0.0-1.0, default 0.3)
    #[arg(long, default_value = "0.3")]
    context_alpha: f32,

    /// Camera zoom: multiplier on target bounding radius for camera distance.
    /// Larger = more zoomed out. Default 5.0 shows target + nearby context.
    #[arg(long, default_value = "5.0")]
    zoom: f32,
}

struct Viewpoint {
    name: &'static str,
    yaw: f32,
    pitch: f32,
}

fn parse_color(s: &str) -> Result<[f32; 3]> {
    let parts: Vec<f32> = s.split(',').map(|p| p.trim().parse()).collect::<std::result::Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        anyhow::bail!("Expected R,G,B but got {} components", parts.len());
    }
    Ok([parts[0], parts[1], parts[2]])
}

/// Mesh buffers for a single draw call
struct MeshBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    if !args.obj_path.exists() {
        anyhow::bail!("OBJ file not found: {:?}", args.obj_path);
    }

    std::fs::create_dir_all(&args.output)?;

    // Load target mesh
    let raw_mesh = load_obj(&args.obj_path)?;
    let mut target_mesh = ProcessedMesh::from_raw(&raw_mesh);
    let target_center = target_mesh.center;
    let target_radius = target_mesh.radius;

    // Apply per-vertex colors if provided
    let use_vert_colors = if let Some(ref vc_path) = args.vertex_colors {
        let data = std::fs::read(vc_path)?;
        let n_floats = data.len() / 4;
        let n_verts = n_floats / 3;
        if n_verts != target_mesh.vertices.len() {
            anyhow::bail!(
                "Vertex color file has {} vertices but mesh has {}",
                n_verts,
                target_mesh.vertices.len()
            );
        }
        let colors: &[f32] = bytemuck::cast_slice(&data);
        for (i, vert) in target_mesh.vertices.iter_mut().enumerate() {
            vert.color = [colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]];
        }
        log::info!("Applied per-vertex colors from {:?}", vc_path);
        true
    } else {
        false
    };

    // Build context mesh (merged) if context files provided
    let context_mesh = if !args.context.is_empty() {
        let mut ctx_vertices: Vec<Vertex> = Vec::new();
        let mut ctx_indices: Vec<u32> = Vec::new();

        for ctx_path in &args.context {
            if !ctx_path.exists() {
                log::warn!("Context OBJ not found, skipping: {:?}", ctx_path);
                continue;
            }
            let ctx_raw = load_obj(ctx_path)?;
            let ctx_mesh = ProcessedMesh::from_raw(&ctx_raw);

            let base_idx = ctx_vertices.len() as u32;
            let n = ctx_mesh.vertices.len();
            ctx_vertices.extend(ctx_mesh.vertices);
            for idx in &ctx_mesh.indices {
                ctx_indices.push(base_idx + idx);
            }
            log::info!("Added context mesh: {:?} ({} verts)", ctx_path, n);
        }

        log::info!("Context total: {} verts, {} tris", ctx_vertices.len(), ctx_indices.len() / 3);
        Some(ProcessedMesh {
            vertices: ctx_vertices,
            indices: ctx_indices,
            center: Vec3::ZERO, // unused
            radius: 0.0,       // unused
        })
    } else {
        None
    };

    // Camera: center on target, zoom based on target size
    let view_radius = if context_mesh.is_some() {
        target_radius * args.zoom
    } else {
        target_radius
    };

    let viewpoints = vec![
        Viewpoint { name: "front",       yaw: 0.0,                          pitch: 0.0 },
        Viewpoint { name: "right",       yaw: std::f32::consts::FRAC_PI_2,  pitch: 0.0 },
        Viewpoint { name: "back",        yaw: std::f32::consts::PI,         pitch: 0.0 },
        Viewpoint { name: "left",        yaw: -std::f32::consts::FRAC_PI_2, pitch: 0.0 },
        Viewpoint { name: "top",         yaw: 0.0,                          pitch: 1.4 },
        Viewpoint { name: "bottom",      yaw: 0.0,                          pitch: -1.4 },
        Viewpoint { name: "front_right", yaw: 0.7,                          pitch: 0.5 },
        Viewpoint { name: "back_left",   yaw: std::f32::consts::PI + 0.7,   pitch: 0.5 },
        Viewpoint { name: "iso",         yaw: 2.3,                          pitch: 0.6 },
    ];

    pollster::block_on(render_all(
        &args,
        &target_mesh,
        context_mesh.as_ref(),
        target_center,
        view_radius,
        &viewpoints,
        use_vert_colors,
    ))?;

    Ok(())
}

async fn render_all(
    args: &Args,
    target_mesh: &ProcessedMesh,
    context_mesh: Option<&ProcessedMesh>,
    camera_center: Vec3,
    camera_radius: f32,
    viewpoints: &[Viewpoint],
    use_vertex_colors: bool,
) -> Result<()> {
    let width = args.width;
    let height = args.height;
    let has_context = context_mesh.is_some();

    // Create headless wgpu device
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;

    log::info!("Using adapter: {:?}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to request device: {}", e))?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    // Textures
    let render_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render Target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let render_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Readback buffer
    let bytes_per_row = (4 * width + 255) & !255;
    let buffer_size = (bytes_per_row * height) as u64;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Target mesh buffers
    let target_bufs = MeshBuffers {
        vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Target VB"),
            contents: bytemuck::cast_slice(&target_mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Target IB"),
            contents: bytemuck::cast_slice(&target_mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        }),
        num_indices: target_mesh.indices.len() as u32,
    };

    // Context mesh buffers (if any)
    let context_bufs = context_mesh.map(|cm| MeshBuffers {
        vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Context VB"),
            contents: bytemuck::cast_slice(&cm.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Context IB"),
            contents: bytemuck::cast_slice(&cm.indices),
            usage: wgpu::BufferUsages::INDEX,
        }),
        num_indices: cm.indices.len() as u32,
    });

    // Camera uniform buffers — need two for scene mode (context + target in same pass)
    let camera_uniform = CameraUniform::new();
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Uniform"),
        contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let camera_buffer2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Uniform 2"),
        contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Camera BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Camera BG"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });
    let bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Camera BG 2"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer2.as_entire_binding(),
        }],
    });

    // Shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shader.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Opaque pipeline (for target / single mesh)
    let opaque_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Opaque Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // Transparent pipeline (for context — alpha blend, no depth write)
    let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Transparent Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // No culling for transparent — see interior
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false, // Don't write depth for transparent
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // Camera setup
    let mut camera = OrbitCamera::new(Vec3::ZERO, 5.0);
    camera.fit_to_mesh(camera_center, camera_radius);
    let aspect = width as f32 / height as f32;

    // Parse colors
    let uniform_color: Option<[f32; 3]> = args.color.as_ref().map(|c| {
        let parts: Vec<f32> = c.split(',').map(|s| s.trim().parse().unwrap()).collect();
        [parts[0], parts[1], parts[2]]
    });
    let target_col = parse_color(&args.target_color)?;
    let context_col = parse_color(&args.context_color)?;

    for vp in viewpoints {
        camera.yaw = vp.yaw;
        camera.pitch = vp.pitch;

        // Write uniform buffers BEFORE the render pass
        if has_context {
            // Buffer 1: context (transparent)
            let mut cam_ctx = CameraUniform::new();
            cam_ctx.update(&camera, aspect);
            cam_ctx.set_material_color_alpha(
                context_col[0], context_col[1], context_col[2],
                args.context_alpha,
            );
            queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[cam_ctx]));

            // Buffer 2: target (opaque)
            let mut cam_tgt = CameraUniform::new();
            cam_tgt.update(&camera, aspect);
            cam_tgt.set_material_color(target_col[0], target_col[1], target_col[2]);
            queue.write_buffer(&camera_buffer2, 0, bytemuck::cast_slice(&[cam_tgt]));
        } else {
            let mut cam_uni = CameraUniform::new();
            cam_uni.update(&camera, aspect);
            if let Some([r, g, b]) = uniform_color {
                cam_uni.set_material_color(r, g, b);
            } else if use_vertex_colors {
                cam_uni.use_vertex_colors();
            }
            queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[cam_uni]));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1, g: 0.1, b: 0.15, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if has_context {
                // Pass 1: context (transparent)
                if let Some(ref ctx) = context_bufs {
                    pass.set_pipeline(&transparent_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.set_vertex_buffer(0, ctx.vertex_buffer.slice(..));
                    pass.set_index_buffer(ctx.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..ctx.num_indices, 0, 0..1);
                }

                // Pass 2: target (opaque, on top)
                pass.set_pipeline(&opaque_pipeline);
                pass.set_bind_group(0, &bind_group2, &[]);
                pass.set_vertex_buffer(0, target_bufs.vertex_buffer.slice(..));
                pass.set_index_buffer(target_bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..target_bufs.num_indices, 0, 0..1);
            } else {
                pass.set_pipeline(&opaque_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_vertex_buffer(0, target_bufs.vertex_buffer.slice(..));
                pass.set_index_buffer(target_bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..target_bufs.num_indices, 0, 0..1);
            }
        }

        // Copy texture to readback buffer
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Read back pixels
        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv()??;

        let data = buffer_slice.get_mapped_range();
        let row_bytes = (4 * width) as usize;
        let mut pixels = Vec::with_capacity(row_bytes * height as usize);
        for y in 0..height as usize {
            let start = y * bytes_per_row as usize;
            pixels.extend_from_slice(&data[start..start + row_bytes]);
        }
        drop(data);
        readback_buffer.unmap();

        let img = image::RgbaImage::from_raw(width, height, pixels).expect("Failed to create image");
        let out_path = args.output.join(format!("{}.png", vp.name));
        img.save(&out_path)?;
        log::info!("Saved: {}", out_path.display());
    }

    println!("{}", args.output.display());
    Ok(())
}
