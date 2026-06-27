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
//! - [`load_glb_animated`] additionally **bakes a skeletal animation** into N
//!   static frames (CPU skinning done once at load, never per frame) — the demo
//!   renders the right frame per unit, so the army animates without runtime
//!   skinning. Static `load_glb` (rest pose) is kept for props.

use macroquad::prelude::{Mat4, Quat, Vec3, Vec4};

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

// ------------------------------------------------------- skeletal animation

/// Load a `.glb` and **bake** a skeletal animation clip into `n_frames` static
/// frames — CPU skinning done once here, never per frame. `prefs` is a
/// priority-ordered list of substrings to choose the clip by name (e.g.
/// `["walk","run"]` for movement, `["attack","sword"]` for an attack). Falls back
/// to the first clip; if the model has no skin / no animation, returns a single
/// static frame. All frames share frame-0's normalisation so the model doesn't
/// pulse; `indices` and `footprint` are identical across frames.
pub fn load_glb_clip(bytes: &[u8], n_frames: usize, prefs: &[&str]) -> Vec<ModelCpu> {
    let (doc, buffers, images) = gltf::import_slice(bytes).expect("glb parse");
    let anim = pick_animation(&doc, prefs);
    if doc.skins().next().is_none() || anim.is_none() || n_frames < 2 {
        return vec![load_glb(bytes)]; // static fallback (props, or un-rigged)
    }
    let anim = anim.unwrap();
    let dur = anim_duration(&anim, &buffers).max(1e-3);
    let (parent, order) = build_hierarchy(&doc);

    let mut frames_v: Vec<Vec<ModelVertex>> = Vec::with_capacity(n_frames);
    let mut indices: Vec<u16> = Vec::new();
    for fi in 0..n_frames {
        let t = dur * fi as f32 / n_frames as f32; // one full loop over the clip
        let global = pose_globals(&doc, &buffers, &anim, t, &parent, &order);
        let (verts, idx) = skin_frame(&doc, &buffers, &images, &global);
        if fi == 0 { indices = idx; }
        frames_v.push(verts);
    }
    if frames_v[0].is_empty() { return vec![load_glb(bytes)]; }
    // One normalisation (from frame 0) applied to every frame.
    let (cx, cz, lo_y, s, footprint) = norm_params(&frames_v[0]);
    frames_v.into_iter().map(|mut verts| {
        for v in verts.iter_mut() {
            v.pos[0] = (v.pos[0] - cx) * s;
            v.pos[1] = (v.pos[1] - lo_y) * s;
            v.pos[2] = (v.pos[2] - cz) * s;
        }
        ModelCpu { vertices: verts, indices: indices.clone(), footprint }
    }).collect()
}

/// Choose the clip whose name best matches `prefs` (priority-ordered substrings,
/// case-insensitive), else the first animation.
fn pick_animation<'a>(doc: &'a gltf::Document, prefs: &[&str]) -> Option<gltf::Animation<'a>> {
    let rank = |n: &str| { let l = n.to_lowercase(); prefs.iter().position(|p| l.contains(p)).unwrap_or(prefs.len()) };
    let mut best: Option<(usize, gltf::Animation)> = None;
    for a in doc.animations() {
        let r = a.name().map(rank).unwrap_or(prefs.len() + 1);
        if best.as_ref().is_none_or(|(br, _)| r < *br) { best = Some((r, a)); }
    }
    best.map(|(_, a)| a)
}

fn anim_duration(anim: &gltf::Animation, buffers: &[gltf::buffer::Data]) -> f32 {
    let mut d = 0.0f32;
    for ch in anim.channels() {
        let reader = ch.reader(|b| buffers.get(b.index()).map(|x| &x.0[..]));
        if let Some(inp) = reader.read_inputs() { for t in inp { d = d.max(t); } }
    }
    d
}

/// Parent index + a parent-before-child visit order for the node forest.
fn build_hierarchy(doc: &gltf::Document) -> (Vec<Option<usize>>, Vec<usize>) {
    let n = doc.nodes().count();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut parent: Vec<Option<usize>> = vec![None; n];
    for node in doc.nodes() {
        for c in node.children() { children[node.index()].push(c.index()); parent[c.index()] = Some(node.index()); }
    }
    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<usize> = (0..n).filter(|&i| parent[i].is_none()).collect();
    while let Some(i) = stack.pop() {
        order.push(i);
        for &c in &children[i] { stack.push(c); }
    }
    (parent, order)
}

/// Global transform of every node at animation time `t`.
fn pose_globals(doc: &gltf::Document, buffers: &[gltf::buffer::Data], anim: &gltf::Animation, t: f32, parent: &[Option<usize>], order: &[usize]) -> Vec<Mat4> {
    use gltf::animation::util::ReadOutputs;
    let n = doc.nodes().count();
    let (mut tr, mut ro, mut sc) = (vec![Vec3::ZERO; n], vec![Quat::IDENTITY; n], vec![Vec3::ONE; n]);
    for node in doc.nodes() {
        let (t3, q4, s3) = node.transform().decomposed();
        let i = node.index();
        tr[i] = Vec3::from_array(t3); ro[i] = Quat::from_array(q4); sc[i] = Vec3::from_array(s3);
    }
    for ch in anim.channels() {
        let target = ch.target().node().index();
        let reader = ch.reader(|b| buffers.get(b.index()).map(|x| &x.0[..]));
        let times: Vec<f32> = match reader.read_inputs() { Some(it) => it.collect(), None => continue };
        match reader.read_outputs() {
            Some(ReadOutputs::Translations(it)) => { let v: Vec<[f32; 3]> = it.collect(); tr[target] = sample_v3(&times, &v, t); }
            Some(ReadOutputs::Rotations(it)) => { let v: Vec<[f32; 4]> = it.into_f32().collect(); ro[target] = sample_quat(&times, &v, t); }
            Some(ReadOutputs::Scales(it)) => { let v: Vec<[f32; 3]> = it.collect(); sc[target] = sample_v3(&times, &v, t); }
            _ => {}
        }
    }
    let local: Vec<Mat4> = (0..n).map(|i| Mat4::from_scale_rotation_translation(sc[i], ro[i], tr[i])).collect();
    let mut global = vec![Mat4::IDENTITY; n];
    for &i in order {
        global[i] = match parent[i] { Some(p) => global[p] * local[i], None => local[i] };
    }
    global
}

fn keyframe(times: &[f32], t: f32) -> (usize, usize, f32) {
    if times.len() < 2 { return (0, 0, 0.0); }
    let last = times.len() - 1;
    if t <= times[0] { return (0, 0, 0.0); }
    if t >= times[last] { return (last, last, 0.0); }
    let mut i = 0;
    while i + 1 < times.len() && times[i + 1] < t { i += 1; }
    let (a, b) = (times[i], times[i + 1]);
    (i, i + 1, if b > a { (t - a) / (b - a) } else { 0.0 })
}
fn sample_v3(times: &[f32], vals: &[[f32; 3]], t: f32) -> Vec3 {
    if vals.is_empty() { return Vec3::ZERO; }
    let (i, j, f) = keyframe(times, t);
    Vec3::from_array(vals[i]).lerp(Vec3::from_array(vals[j]), f)
}
fn sample_quat(times: &[f32], vals: &[[f32; 4]], t: f32) -> Quat {
    if vals.is_empty() { return Quat::IDENTITY; }
    let (i, j, f) = keyframe(times, t);
    Quat::from_array(vals[i]).slerp(Quat::from_array(vals[j]), f)
}

/// CPU-skin every skinned primitive at the given pose into a flat mesh.
fn skin_frame(doc: &gltf::Document, buffers: &[gltf::buffer::Data], images: &[gltf::image::Data], global: &[Mat4]) -> (Vec<ModelVertex>, Vec<u16>) {
    let (mut verts, mut indices) = (Vec::new(), Vec::new());
    for node in doc.nodes() {
        let (mesh, skin) = match (node.mesh(), node.skin()) { (Some(m), Some(s)) => (m, s), _ => continue };
        let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
        let inv: Vec<Mat4> = skin.reader(|b| buffers.get(b.index()).map(|x| &x.0[..]))
            .read_inverse_bind_matrices()
            .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_else(|| vec![Mat4::IDENTITY; joints.len()]);
        let jmat: Vec<Mat4> = joints.iter().enumerate().map(|(k, &nj)| global[nj] * inv[k]).collect();
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| buffers.get(b.index()).map(|x| &x.0[..]));
            let positions: Vec<[f32; 3]> = match reader.read_positions() { Some(p) => p.collect(), None => continue };
            let normals: Vec<[f32; 3]> = reader.read_normals().map(|n| n.collect()).unwrap_or_default();
            let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0).map(|t| t.into_f32().collect()).unwrap_or_default();
            let jn: Vec<[u16; 4]> = reader.read_joints(0).map(|j| j.into_u16().collect()).unwrap_or_default();
            let wt: Vec<[f32; 4]> = reader.read_weights(0).map(|w| w.into_f32().collect()).unwrap_or_default();
            let pbr = prim.material().pbr_metallic_roughness();
            let factor = pbr.base_color_factor();
            let tex = pbr.base_color_texture().and_then(|i| images.get(i.texture().source().index()));
            let base = verts.len() as u16;
            for (vi, pos) in positions.iter().enumerate() {
                let p = Vec3::from_array(*pos);
                let n = normals.get(vi).copied().map(Vec3::from_array).unwrap_or(Vec3::Y);
                let (mut sp, mut sn, mut wsum) = (Vec3::ZERO, Vec3::ZERO, 0.0f32);
                if let (Some(j), Some(w)) = (jn.get(vi), wt.get(vi)) {
                    for k in 0..4 {
                        if w[k] <= 0.0 { continue; }
                        let m = jmat[j[k] as usize];
                        sp += w[k] * m.transform_point3(p);
                        sn += w[k] * m.transform_vector3(n);
                        wsum += w[k];
                    }
                }
                let (fp, fnm) = if wsum > 1e-4 { (sp / wsum, sn.normalize_or_zero()) } else { (p, n) };
                let col = match (tex, uvs.get(vi)) { (Some(img), Some(uv)) => mul4(sample(img, *uv), factor), _ => factor };
                verts.push(ModelVertex { pos: fp.to_array(), normal: fnm.to_array(), color: col });
            }
            match reader.read_indices() {
                Some(it) => { for idx in it.into_u32() { indices.push(base + idx as u16); } }
                None => { for k in 0..positions.len() as u16 { indices.push(base + k); } }
            }
        }
    }
    (verts, indices)
}

/// (centre x, centre z, min y, scale, footprint) to normalise a frame to unit
/// height with feet at y=0, centred on XZ.
fn norm_params(verts: &[ModelVertex]) -> (f32, f32, f32, f32, f32) {
    if verts.is_empty() { return (0.0, 0.0, 0.0, 1.0, 0.5); }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in verts { for k in 0..3 { lo[k] = lo[k].min(v.pos[k]); hi[k] = hi[k].max(v.pos[k]); } }
    let s = 1.0 / (hi[1] - lo[1]).max(1e-4);
    let footprint = 0.5 * ((hi[0] - lo[0]).max(hi[2] - lo[2])) * s;
    ((lo[0] + hi[0]) * 0.5, (lo[2] + hi[2]) * 0.5, lo[1], s, footprint)
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

    /// Baked skeletal animation: multiple frames, all normalised + finite, and
    /// the frames actually differ (the skin is moving, not a frozen T-pose).
    #[test]
    fn bakes_animation() {
        let frames = super::load_glb_clip(include_bytes!("../assets/siege/models/skeleton_a.glb"), 8, &["walk"]);
        assert!(frames.len() > 1, "expected baked frames, got {}", frames.len());
        for f in &frames {
            assert!(!f.vertices.is_empty() && !f.indices.is_empty());
            // Sane bounds: finite, roughly unit-scaled (only frame 0 is exactly
            // [0,1]; other poses legitimately dip/stretch a little). Catches an
            // exploded skin (huge bbox) or a collapsed one.
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for v in &f.vertices {
                assert!(v.pos.iter().all(|c| c.is_finite()), "non-finite vertex");
                lo = lo.min(v.pos[1]); hi = hi.max(v.pos[1]);
            }
            assert!((-0.5..0.4).contains(&lo) && (0.6..1.6).contains(&hi), "frame bbox off (lo={lo}, hi={hi})");
        }
        let mid = frames.len() / 2;
        let d: f32 = frames[0].vertices.iter().zip(&frames[mid].vertices)
            .map(|(a, b)| (a.pos[0] - b.pos[0]).abs() + (a.pos[1] - b.pos[1]).abs()).sum();
        assert!(d > 1e-3, "frames identical — skinning produced no motion");
    }
}
