//! GPU-instanced 3D point renderer for the critters demo — the "option B"
//! path: true hardware instancing via raw miniquad (the layer beneath
//! macroquad). One base mesh (a low-poly sphere, or a camera-facing quad for
//! billboards) is uploaded **once**; every frame only a small per-instance
//! buffer (centre + radius + colour, 32 bytes/critter) is updated, and all
//! critters draw in a **single instanced draw call**. This scales to hundreds
//! of thousands of critters, where the immediate-mode `draw_sphere` path
//! (geometry rebuilt and re-uploaded every frame) collapses.
//!
//! Integration: macroquad batches its own draws; we call `get_internal_gl()`,
//! `flush()` that batch, then issue raw miniquad draws inside our own
//! `begin_default_pass(PassAction::Nothing)` (preserving the colour+depth
//! macroquad already wrote, so the instanced spheres depth-test correctly
//! against the world box). See `bin/critters3d.rs` for the call site.

use macroquad::miniquad::{
    BlendFactor, BlendState, BlendValue, Bindings, BufferId, BufferLayout, BufferSource, BufferType,
    BufferUsage, Comparison, CullFace, Equation, Pipeline, PipelineParams, PrimitiveType,
    RenderingBackend, ShaderMeta, ShaderSource, UniformBlockLayout, UniformDesc, UniformType,
    UniformsSource, VertexAttribute, VertexFormat, VertexStep,
};
use macroquad::prelude::{Mat4, Vec3};

/// Per-instance data, `repr(C)` to match the instance vertex attributes
/// `in_inst` (Float4: centre.xyz + radius) and `in_color` (Float4: RGBA).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Instance {
    pub pos_radius: [f32; 4],
    pub color: [f32; 4],
}

impl Instance {
    pub fn new(center: Vec3, radius: f32, color: [f32; 4]) -> Self {
        Instance { pos_radius: [center.x, center.y, center.z, radius], color }
    }
}

/// Which base geometry to instance.
#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Spheres,
    /// Round billboards (a `discard` masks the quad to a disc). Prettiest, but
    /// the discard disables early-Z so they go fill-bound under overdraw.
    Billboards,
    /// Square billboards — no `discard`, so early-Z stays on and the fragment
    /// stage is minimal. Fastest at huge counts (the point of billboards).
    BillboardsSquare,
}

// Uniform blocks — `repr(C)` field order/offsets must match the `ShaderMeta`
// uniform lists below (miniquad packs uniforms by declared order, not std140).
#[repr(C)]
struct SphereUniforms {
    mvp: [f32; 16],
    light_dir: [f32; 3],
}
#[repr(C)]
struct BillboardUniforms {
    mvp: [f32; 16],
    cam_right: [f32; 3],
    cam_up: [f32; 3],
}
#[repr(C)]
struct EffectUniforms {
    mvp: [f32; 16],
}

/// Which base wireframe to instance for the combat effects.
#[derive(Clone, Copy, PartialEq)]
pub enum EffectMesh {
    Sphere, // attack blasts + death bursts
    Drop,   // flamer teardrops
}

/// Per-effect instance: a full model transform (columns of a `Mat4`, so the
/// canonical wireframe is positioned/rotated/scaled per effect) plus a colour.
/// `repr(C)`; the four columns map to attributes `in_m0..in_m3` and `in_color`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EffectInstance {
    pub m0: [f32; 4],
    pub m1: [f32; 4],
    pub m2: [f32; 4],
    pub m3: [f32; 4],
    pub color: [f32; 4],
}
impl EffectInstance {
    pub fn new(model: Mat4, color: [f32; 4]) -> Self {
        let c = model.to_cols_array_2d();
        EffectInstance { m0: c[0], m1: c[1], m2: c[2], m3: c[3], color }
    }
}

pub struct InstancedRenderer {
    sphere_pipe: Pipeline,
    sphere_vbuf: BufferId,
    sphere_ibuf: BufferId,
    sphere_nidx: i32,
    quad_pipe: Pipeline,
    square_pipe: Pipeline,
    quad_vbuf: BufferId,
    quad_ibuf: BufferId,
    quad_nidx: i32,
    inst_buf: BufferId,
    inst_cap: usize,
    // Combat effects: a line-list pipeline + two base wireframes, instanced.
    effect_pipe: Pipeline,
    sphere_wire_vbuf: BufferId,
    sphere_wire_ibuf: BufferId,
    sphere_wire_nidx: i32,
    drop_wire_vbuf: BufferId,
    drop_wire_ibuf: BufferId,
    drop_wire_nidx: i32,
    effect_inst_buf: BufferId,
    effect_inst_cap: usize,
}

impl InstancedRenderer {
    pub fn new(ctx: &mut dyn RenderingBackend) -> Self {
        // --- base meshes ---
        let (sv, si) = unit_sphere(8, 12);
        let sphere_vbuf =
            ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&sv));
        let sphere_ibuf =
            ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&si));
        let sphere_nidx = si.len() as i32;

        // unit quad in local [-1,1] space (corners), expanded to face camera.
        let qv: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        let qi: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let quad_vbuf =
            ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&qv));
        let quad_ibuf =
            ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&qi));
        let quad_nidx = qi.len() as i32;

        // --- per-frame instance buffer (grown on demand) ---
        let inst_cap = 4096usize;
        let inst_buf = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Stream,
            BufferSource::empty::<Instance>(inst_cap),
        );

        // --- pipelines: one PerVertex base buffer + one PerInstance buffer ---
        let layouts = [
            BufferLayout::default(),
            BufferLayout { step_func: VertexStep::PerInstance, ..Default::default() },
        ];
        let params = PipelineParams {
            depth_test: Comparison::LessOrEqual,
            depth_write: true,
            cull_face: CullFace::Nothing,
            ..Default::default()
        };

        let sphere_shader = ctx
            .new_shader(
                ShaderSource::Glsl { vertex: SPHERE_VS, fragment: SPHERE_FS },
                sphere_meta(),
            )
            .expect("sphere instanced shader compiles");
        let sphere_pipe = ctx.new_pipeline(
            &layouts,
            &[
                VertexAttribute::with_buffer("in_pos", VertexFormat::Float3, 0),
                VertexAttribute::with_buffer("in_inst", VertexFormat::Float4, 1),
                VertexAttribute::with_buffer("in_color", VertexFormat::Float4, 1),
            ],
            sphere_shader,
            params,
        );

        let quad_shader = ctx
            .new_shader(
                ShaderSource::Glsl { vertex: QUAD_VS, fragment: QUAD_FS },
                billboard_meta(),
            )
            .expect("billboard instanced shader compiles");
        let quad_attrs = [
            VertexAttribute::with_buffer("in_local", VertexFormat::Float2, 0),
            VertexAttribute::with_buffer("in_inst", VertexFormat::Float4, 1),
            VertexAttribute::with_buffer("in_color", VertexFormat::Float4, 1),
        ];
        let quad_pipe = ctx.new_pipeline(&layouts, &quad_attrs, quad_shader, params);

        // Square billboards: same vertex shader, a fragment shader with no
        // `discard`, so the GPU keeps early-Z (a separate program is required —
        // a *conditional* discard would still disable early-Z).
        let square_shader = ctx
            .new_shader(
                ShaderSource::Glsl { vertex: QUAD_VS, fragment: QUAD_FS_SQUARE },
                billboard_meta(),
            )
            .expect("square billboard instanced shader compiles");
        let square_pipe = ctx.new_pipeline(&layouts, &quad_attrs, square_shader, params);

        // --- combat effects: line-list wireframes, instanced, alpha-blended ---
        let (swv, swi) = wire_to_buffers(unit_sphere_wire());
        let sphere_wire_vbuf = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&swv));
        let sphere_wire_ibuf = ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&swi));
        let sphere_wire_nidx = swi.len() as i32;
        let (dwv, dwi) = wire_to_buffers(unit_drop_wire());
        let drop_wire_vbuf = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&dwv));
        let drop_wire_ibuf = ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&dwi));
        let drop_wire_nidx = dwi.len() as i32;
        let effect_inst_cap = 1024usize;
        let effect_inst_buf = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Stream, BufferSource::empty::<EffectInstance>(effect_inst_cap));

        let effect_shader = ctx
            .new_shader(ShaderSource::Glsl { vertex: EFFECT_VS, fragment: EFFECT_FS }, effect_meta())
            .expect("effect instanced shader compiles");
        let effect_pipe = ctx.new_pipeline(
            &layouts,
            &[
                VertexAttribute::with_buffer("in_pos", VertexFormat::Float3, 0),
                VertexAttribute::with_buffer("in_m0", VertexFormat::Float4, 1),
                VertexAttribute::with_buffer("in_m1", VertexFormat::Float4, 1),
                VertexAttribute::with_buffer("in_m2", VertexFormat::Float4, 1),
                VertexAttribute::with_buffer("in_m3", VertexFormat::Float4, 1),
                VertexAttribute::with_buffer("in_color", VertexFormat::Float4, 1),
            ],
            effect_shader,
            PipelineParams {
                primitive_type: PrimitiveType::Lines,
                depth_test: Comparison::LessOrEqual,
                depth_write: false, // effects don't occlude each other
                cull_face: CullFace::Nothing,
                color_blend: Some(BlendState::new(Equation::Add, BlendFactor::Value(BlendValue::SourceAlpha), BlendFactor::OneMinusValue(BlendValue::SourceAlpha))),
                ..Default::default()
            },
        );

        Self {
            sphere_pipe,
            sphere_vbuf,
            sphere_ibuf,
            sphere_nidx,
            quad_pipe,
            square_pipe,
            quad_vbuf,
            quad_ibuf,
            quad_nidx,
            inst_buf,
            inst_cap,
            effect_pipe,
            sphere_wire_vbuf,
            sphere_wire_ibuf,
            sphere_wire_nidx,
            drop_wire_vbuf,
            drop_wire_ibuf,
            drop_wire_nidx,
            effect_inst_buf,
            effect_inst_cap,
        }
    }

    /// Triangles in one instanced sphere (the base mesh). Multiply by the
    /// instance count for the total submitted in sphere mode.
    pub fn sphere_triangles(&self) -> i32 { self.sphere_nidx / 3 }

    /// Draw all `instances` in one instanced call. Must be invoked inside an
    /// active render pass (the caller wraps it in `begin_default_pass` /
    /// `end_render_pass`). `mvp` is the camera's `proj*view`; `cam_right` /
    /// `cam_up` are the camera basis (only used for billboards).
    pub fn draw(
        &mut self,
        ctx: &mut dyn RenderingBackend,
        mode: Mode,
        instances: &[Instance],
        mvp: Mat4,
        cam_right: Vec3,
        cam_up: Vec3,
    ) {
        if instances.is_empty() {
            return;
        }
        if instances.len() > self.inst_cap {
            ctx.delete_buffer(self.inst_buf);
            self.inst_cap = (instances.len() * 2).next_power_of_two();
            self.inst_buf = ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Stream,
                BufferSource::empty::<Instance>(self.inst_cap),
            );
        }
        ctx.buffer_update(self.inst_buf, BufferSource::slice(instances));

        let n = instances.len() as i32;
        let mvp = mvp.to_cols_array();
        match mode {
            Mode::Spheres => {
                ctx.apply_pipeline(&self.sphere_pipe);
                ctx.apply_bindings(&Bindings {
                    vertex_buffers: vec![self.sphere_vbuf, self.inst_buf],
                    index_buffer: self.sphere_ibuf,
                    images: vec![],
                });
                let u = SphereUniforms { mvp, light_dir: [0.45, 0.8, 0.35] };
                ctx.apply_uniforms(UniformsSource::table(&u));
                ctx.draw(0, self.sphere_nidx, n);
            }
            Mode::Billboards | Mode::BillboardsSquare => {
                let pipe = if mode == Mode::BillboardsSquare { &self.square_pipe } else { &self.quad_pipe };
                ctx.apply_pipeline(pipe);
                ctx.apply_bindings(&Bindings {
                    vertex_buffers: vec![self.quad_vbuf, self.inst_buf],
                    index_buffer: self.quad_ibuf,
                    images: vec![],
                });
                let u = BillboardUniforms {
                    mvp,
                    cam_right: cam_right.to_array(),
                    cam_up: cam_up.to_array(),
                };
                ctx.apply_uniforms(UniformsSource::table(&u));
                ctx.draw(0, self.quad_nidx, n);
            }
        }
    }

    /// Draw combat effects (attack/burst spheres, or flamer drops) instanced as
    /// wireframes through our own line pipeline — so they bypass macroquad's
    /// shared immediate-line batch. Each `EffectInstance` carries a model
    /// transform that places/orients/scales the canonical base wireframe.
    pub fn draw_effects(&mut self, ctx: &mut dyn RenderingBackend, mesh: EffectMesh, instances: &[EffectInstance], mvp: Mat4) {
        if instances.is_empty() {
            return;
        }
        if instances.len() > self.effect_inst_cap {
            ctx.delete_buffer(self.effect_inst_buf);
            self.effect_inst_cap = (instances.len() * 2).next_power_of_two();
            self.effect_inst_buf = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Stream, BufferSource::empty::<EffectInstance>(self.effect_inst_cap));
        }
        ctx.buffer_update(self.effect_inst_buf, BufferSource::slice(instances));
        let (vbuf, ibuf, nidx) = match mesh {
            EffectMesh::Sphere => (self.sphere_wire_vbuf, self.sphere_wire_ibuf, self.sphere_wire_nidx),
            EffectMesh::Drop => (self.drop_wire_vbuf, self.drop_wire_ibuf, self.drop_wire_nidx),
        };
        ctx.apply_pipeline(&self.effect_pipe);
        ctx.apply_bindings(&Bindings {
            vertex_buffers: vec![vbuf, self.effect_inst_buf],
            index_buffer: ibuf,
            images: vec![],
        });
        let u = EffectUniforms { mvp: mvp.to_cols_array() };
        ctx.apply_uniforms(UniformsSource::table(&u));
        ctx.draw(0, nidx, instances.len() as i32);
    }
}

/// Low-poly UV sphere centred at the origin, unit radius. Returns flat
/// `[x,y,z, ...]` positions and triangle indices. Vertices double as normals
/// (unit sphere), used for cheap diffuse shading in the vertex shader.
fn unit_sphere(rings: u16, sectors: u16) -> (Vec<f32>, Vec<u16>) {
    use std::f32::consts::PI;
    let mut verts = Vec::with_capacity(((rings + 1) * (sectors + 1) * 3) as usize);
    let mut idx = Vec::new();
    for r in 0..=rings {
        let phi = PI * r as f32 / rings as f32;
        let (sp, cp) = phi.sin_cos();
        for s in 0..=sectors {
            let theta = 2.0 * PI * s as f32 / sectors as f32;
            let (st, ct) = theta.sin_cos();
            verts.push(sp * ct);
            verts.push(cp);
            verts.push(sp * st);
        }
    }
    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + stride;
            idx.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    (verts, idx)
}

/// Convert flat wireframe positions (segment endpoints, in pairs) into a
/// `(vertices, indices)` pair for a Lines draw — indices are just `0..n`.
fn wire_to_buffers(verts: Vec<f32>) -> (Vec<f32>, Vec<u16>) {
    let nverts = (verts.len() / 3) as u16;
    let indices: Vec<u16> = (0..nverts).collect();
    (verts, indices)
}

/// Unit sphere wireframe (latitude rings + meridians) as line segments.
fn unit_sphere_wire() -> Vec<f32> {
    use std::f32::consts::PI;
    let (rings, seg, merid) = (4usize, 16usize, 8usize);
    let mut v = Vec::new();
    for ri in 1..rings {
        let phi = PI * ri as f32 / rings as f32;
        let (sp, cp) = phi.sin_cos();
        for s in 0..seg {
            let a0 = 2.0 * PI * s as f32 / seg as f32;
            let a1 = 2.0 * PI * (s + 1) as f32 / seg as f32;
            v.extend_from_slice(&[sp * a0.cos(), cp, sp * a0.sin(), sp * a1.cos(), cp, sp * a1.sin()]);
        }
    }
    for mi in 0..merid {
        let theta = 2.0 * PI * mi as f32 / merid as f32;
        let (st, ct) = theta.sin_cos();
        let rs = rings * 2;
        for ri in 0..rs {
            let p0 = PI * ri as f32 / rs as f32;
            let p1 = PI * (ri + 1) as f32 / rs as f32;
            v.extend_from_slice(&[p0.sin() * ct, p0.cos(), p0.sin() * st, p1.sin() * ct, p1.cos(), p1.sin() * st]);
        }
    }
    v
}

/// Canonical flamer/teardrop wireframe: apex at the origin, axis +Y, length 3,
/// radius 0.85 (matching `flamer_dims(1.0)` so a uniform scale maps it to any
/// attack size without distorting the rounded cap). Cone edges + far ring +
/// hemisphere cap meridians, as line segments.
fn unit_drop_wire() -> Vec<f32> {
    use std::f32::consts::{FRAC_PI_2, TAU};
    let (length, radius) = (3.0f32, 0.85f32);
    let (n, m) = (12usize, 3usize);
    let mut v = Vec::new();
    let mut prev = (radius, length, 0.0f32); // ring point at angle 0
    for k in 1..=n {
        let a = TAU * k as f32 / n as f32;
        let (rx, rz) = (a.cos() * radius, a.sin() * radius);
        let ring = (rx, length, rz);
        v.extend_from_slice(&[prev.0, prev.1, prev.2, ring.0, ring.1, ring.2]); // ring segment
        v.extend_from_slice(&[0.0, 0.0, 0.0, ring.0, ring.1, ring.2]); // cone edge from apex
        let mut pa = ring;
        for s in 1..=m {
            let beta = FRAC_PI_2 * s as f32 / m as f32;
            let arc = (a.cos() * radius * beta.cos(), length + radius * beta.sin(), a.sin() * radius * beta.cos());
            v.extend_from_slice(&[pa.0, pa.1, pa.2, arc.0, arc.1, arc.2]);
            pa = arc;
        }
        prev = ring;
    }
    v
}

fn sphere_meta() -> ShaderMeta {
    ShaderMeta {
        images: vec![],
        uniforms: UniformBlockLayout {
            uniforms: vec![
                UniformDesc::new("mvp", UniformType::Mat4),
                UniformDesc::new("light_dir", UniformType::Float3),
            ],
        },
    }
}

fn effect_meta() -> ShaderMeta {
    ShaderMeta {
        images: vec![],
        uniforms: UniformBlockLayout { uniforms: vec![UniformDesc::new("mvp", UniformType::Mat4)] },
    }
}

fn billboard_meta() -> ShaderMeta {
    ShaderMeta {
        images: vec![],
        uniforms: UniformBlockLayout {
            uniforms: vec![
                UniformDesc::new("mvp", UniformType::Mat4),
                UniformDesc::new("cam_right", UniformType::Float3),
                UniformDesc::new("cam_up", UniformType::Float3),
            ],
        },
    }
}

// GLSL ES 1.00 (the version macroquad's own shaders use — portable to WebGL1
// and accepted by desktop GL). Instancing divisors are set by miniquad from
// the `PerInstance` BufferLayout, so the per-instance attributes are ordinary
// `attribute`s here.
const SPHERE_VS: &str = r#"#version 100
attribute vec3 in_pos;
attribute vec4 in_inst;
attribute vec4 in_color;
uniform mat4 mvp;
uniform vec3 light_dir;
varying lowp vec4 v_color;
void main() {
    vec3 world = in_inst.xyz + in_inst.w * in_pos;
    gl_Position = mvp * vec4(world, 1.0);
    vec3 n = normalize(in_pos);
    float diff = max(dot(n, normalize(light_dir)), 0.0);
    float sh = 0.35 + 0.65 * diff;
    v_color = vec4(in_color.rgb * sh, in_color.a);
}
"#;

const SPHERE_FS: &str = r#"#version 100
precision mediump float;
varying lowp vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"#;

const QUAD_VS: &str = r#"#version 100
attribute vec2 in_local;
attribute vec4 in_inst;
attribute vec4 in_color;
uniform mat4 mvp;
uniform vec3 cam_right;
uniform vec3 cam_up;
varying lowp vec4 v_color;
varying lowp vec2 v_local;
void main() {
    vec3 world = in_inst.xyz + in_inst.w * (in_local.x * cam_right + in_local.y * cam_up);
    gl_Position = mvp * vec4(world, 1.0);
    v_color = in_color;
    v_local = in_local;
}
"#;

// Round, faux-lit billboard: discard outside the unit disc, brighten toward
// the centre so a flat quad reads as a little ball. Parabolic falloff
// (1 - k·d²) instead of a sqrt — same look, no per-fragment sqrt.
const QUAD_FS: &str = r#"#version 100
precision mediump float;
varying lowp vec4 v_color;
varying lowp vec2 v_local;
void main() {
    float d2 = dot(v_local, v_local);
    if (d2 > 1.0) { discard; }
    float sh = 1.0 - 0.55 * d2;
    gl_FragColor = vec4(v_color.rgb * sh, v_color.a);
}
"#;

// Square billboard: NO discard (keeps early-Z) and no sqrt — the cheapest
// fragment stage. A faint corner darkening still gives a little shading.
const QUAD_FS_SQUARE: &str = r#"#version 100
precision mediump float;
varying lowp vec4 v_color;
varying lowp vec2 v_local;
void main() {
    float sh = 1.0 - 0.25 * dot(v_local, v_local);
    gl_FragColor = vec4(v_color.rgb * sh, v_color.a);
}
"#;

// Combat-effect wireframes: a per-instance model matrix (its four columns come
// in as in_m0..in_m3) places/orients/scales the canonical line mesh.
const EFFECT_VS: &str = r#"#version 100
attribute vec3 in_pos;
attribute vec4 in_m0;
attribute vec4 in_m1;
attribute vec4 in_m2;
attribute vec4 in_m3;
attribute vec4 in_color;
uniform mat4 mvp;
varying lowp vec4 v_color;
void main() {
    mat4 model = mat4(in_m0, in_m1, in_m2, in_m3);
    gl_Position = mvp * model * vec4(in_pos, 1.0);
    v_color = in_color;
}
"#;

const EFFECT_FS: &str = r#"#version 100
precision mediump float;
varying lowp vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"#;
