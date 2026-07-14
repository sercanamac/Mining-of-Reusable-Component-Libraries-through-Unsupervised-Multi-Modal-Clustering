// Camera and lighting uniform buffer
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
    light_position: vec4<f32>,
    material_color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Vertex input from buffer
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

// Data passed from vertex to fragment shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) vertex_color: vec3<f32>,
}

// Vertex shader
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.world_position = in.position;
    out.world_normal = in.normal;
    out.vertex_color = in.color;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);

    return out;
}

// Fragment shader with Blinn-Phong lighting
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Material properties
    let ambient_strength = 0.15;
    let specular_color = vec3<f32>(1.0, 1.0, 1.0);
    let shininess = 32.0;

    // Pick diffuse color and alpha:
    // a > 0.0 → use uniform material_color.rgb with material_color.a as opacity
    // a == 0.0 → use per-vertex color, fully opaque
    var diffuse_color: vec3<f32>;
    var alpha: f32 = 1.0;
    if camera.material_color.a > 0.0 {
        diffuse_color = camera.material_color.rgb;
        alpha = camera.material_color.a;
    } else {
        diffuse_color = in.vertex_color;
    }

    // Normalize interpolated normal
    let normal = normalize(in.world_normal);

    // Light direction (from surface to light)
    let light_dir = normalize(camera.light_position.xyz - in.world_position);

    // View direction (from surface to camera)
    let view_dir = normalize(camera.view_position.xyz - in.world_position);

    // Half vector for Blinn-Phong
    let half_dir = normalize(light_dir + view_dir);

    // Ambient component
    let ambient = ambient_strength * diffuse_color;

    // Diffuse component (Lambertian)
    let diff = max(dot(normal, light_dir), 0.0);
    let diffuse = diff * diffuse_color;

    // Specular component (Blinn-Phong)
    let spec = pow(max(dot(normal, half_dir), 0.0), shininess);
    let specular = spec * specular_color * 0.5;

    // Combine all lighting components
    let result = ambient + diffuse + specular;

    return vec4<f32>(result, alpha);
}
