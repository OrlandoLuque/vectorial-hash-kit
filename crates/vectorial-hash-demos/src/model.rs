//! Minimal glTF (`.glb`) loader for the `siege` demo — turns a Quaternius model
//! into a flat triangle mesh (`position`, `normal`, baked vertex `colour`) ready
//! for the instanced model pipeline in [`crate::instanced3d`].
//!
//! What it does and doesn't do:
//! - Walks the scene graph and **bakes each node's world transform** into the
//!   vertices, so multi-part models land correctly assembled.
//! - Reads the per-primitive **base-colour**: samples the base-colour *texture*
//!   at each vertex UV when present (Quaternius palette atlases), else uses the
//!   material's `base_color_factor`. The colour is stored per vertex; lighting is
//!   applied later in the shader from the normal (so this is unlit colour).
//! - **Normalises** the model: centred on XZ, feet at `y = 0`, scaled to unit
//!   height — the caller then scales/positions it per unit.
//! - Ignores skinning/animation (we render the rest pose).

use macroquad::prelude::{Mat4, Vec3, Vec4};

/// One model vertex for the instanced pipeline (`repr(C)` to match the vertex
/// attributes `in_pos` Float3, `in_normal` Float3, `in_color` Float4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModelVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// CPU-side loaded model: a single merged triangle mesh.
pub struct ModelCpu {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u16>,
    /// XZ footprint half-extent in normalised units (height = 1) — `0.5 ×
    /// max(width, depth)`. Multiply by the model's world height to get the
    /// world-space body radius (what the model actually occupies on the ground).
    pub footprint: f32,
}

/// Parse a `.glb` byte slice into a normalised [`ModelCpu`]. Panics on a parse
/// error (the models are committed assets — a failure is a build-time mistake).
pub fn load_glb(bytes: &[u8]) -> ModelCpu {
    let (doc, buffers, images) = gltf::import_slice(bytes).expect("glb parse");
    let mut verts: Vec<ModelVertex> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();

    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            visit(&node, Mat4::IDENTITY, &buffers, &images, &mut verts, &mut indices);
        }
    }
    let footprint = normalise(&mut verts);
    ModelCpu { vertices: verts, indices, footprint }
}

fn visit(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    verts: &mut Vec<ModelVertex>,
    indices: &mut Vec<u16>,
) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    let normal_mat = Mat4::from_mat3(macroquad::prelude::Mat3::from_mat4(world)); // rotation+scale, no translation

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let positions: Vec<[f32; 3]> = match reader.read_positions() { Some(p) => p.collect(), None => continue };
            let normals: Vec<[f32; 3]> = reader.read_normals().map(|n| n.collect()).unwrap_or_default();
            let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0).map(|t| t.into_f32().collect()).unwrap_or_default();

            // Base colour: texture (sampled per vertex) × factor, else just factor.
            let pbr = prim.material().pbr_metallic_roughness();
            let factor = pbr.base_color_factor();
            let tex = pbr.base_color_texture().and_then(|info| images.get(info.texture().source().index()));

            let base = verts.len() as u16;
            for (i, pos) in positions.iter().enumerate() {
                let p = world.transform_point3(Vec3::from_array(*pos));
                let n = normals.get(i).map(|n| (normal_mat * Vec4::from((Vec3::from_array(*n), 0.0))).truncate().normalize_or_zero()).unwrap_or(Vec3::Y);
                let col = match (tex, uvs.get(i)) {
                    (Some(img), Some(uv)) => mul4(sample(img, *uv), factor),
                    _ => factor,
                };
                verts.push(ModelVertex { pos: p.to_array(), normal: n.to_array(), color: col });
            }
            match reader.read_indices() {
                Some(it) => { for idx in it.into_u32() { indices.push(base + idx as u16); } }
                None => { for k in 0..positions.len() as u16 { indices.push(base + k); } }
            }
        }
    }
    for child in node.children() { visit(&child, world, buffers, images, verts, indices); }
}

/// Nearest-neighbour sample of a glTF base-colour image at `uv`, returned as
/// linear RGBA in 0..1. Handles the RGB and RGBA pixel formats Quaternius uses.
fn sample(img: &gltf::image::Data, uv: [f32; 2]) -> [f32; 4] {
    use gltf::image::Format;
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 { return [1.0; 4]; }
    let px = ((uv[0].rem_euclid(1.0)) * w as f32) as usize % w;
    let py = ((uv[1].rem_euclid(1.0)) * h as f32) as usize % h;
    let ch = match img.format { Format::R8G8B8 => 3, Format::R8G8B8A8 => 4, _ => 4 };
    let o = (py * w + px) * ch;
    let g = |k: usize| img.pixels.get(o + k).copied().unwrap_or(255) as f32 / 255.0;
    match img.format {
        Format::R8G8B8 => [g(0), g(1), g(2), 1.0],
        _ => [g(0), g(1), g(2), if ch == 4 { g(3) } else { 1.0 }],
    }
}

fn mul4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] { [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]] }

/// Centre on XZ, drop feet to `y = 0`, scale to unit height. Returns the XZ
/// footprint half-extent (`0.5 × max(width, depth)`) in the normalised space.
fn normalise(verts: &mut [ModelVertex]) -> f32 {
    if verts.is_empty() { return 0.5; }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in verts.iter() {
        for k in 0..3 { lo[k] = lo[k].min(v.pos[k]); hi[k] = hi[k].max(v.pos[k]); }
    }
    let cx = (lo[0] + hi[0]) * 0.5;
    let cz = (lo[2] + hi[2]) * 0.5;
    let height = (hi[1] - lo[1]).max(1e-4);
    let s = 1.0 / height;
    for v in verts.iter_mut() {
        v.pos[0] = (v.pos[0] - cx) * s;
        v.pos[1] = (v.pos[1] - lo[1]) * s;
        v.pos[2] = (v.pos[2] - cz) * s;
    }
    0.5 * ((hi[0] - lo[0]).max(hi[2] - lo[2])) * s
}

#[cfg(test)]
mod tests {
    /// Parse a real committed model end to end: non-empty, indices in range,
    /// normalised (feet at y≈0, unit height). Guards the loader against regressions.
    #[test]
    fn loads_dragon() {
        let m = super::load_glb(include_bytes!("../assets/siege/models/dragon.glb"));
        assert!(!m.vertices.is_empty(), "no vertices");
        assert!(!m.indices.is_empty(), "no indices");
        let n = m.vertices.len() as u16;
        assert!(m.indices.iter().all(|&i| i < n), "index out of range");
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in &m.vertices { lo = lo.min(v.pos[1]); hi = hi.max(v.pos[1]); }
        assert!(lo.abs() < 1e-3, "feet not at y=0 (lo={lo})");
        assert!((hi - 1.0).abs() < 1e-3, "height not normalised (hi={hi})");
    }
}
