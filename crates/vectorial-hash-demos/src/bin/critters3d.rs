//! Visual 3D critters — the 3D analogue of the 2D `critters` demo, drawn
//! with macroquad's 3D pipeline. Critters drift inside a cube indexed by
//! `Tree3`; an observer at the centre runs a sphere "vision" cull every
//! frame and the culled critters light up and draw a sight-line. The same
//! index workload runs headless via `critters3d_headless`.
//!
//! Run: `cargo run -p vectorial-hash-demos --bin critters3d --release`
//!
//! Two simulations, switchable live with `T`: *observe* (a vision-cull
//! showcase — one sphere from the centre lights the critters it sees) and
//! *combat* (the red critters become predators that attack nearby prey; each
//! attack is an index cull, so it is the "many queries per frame" workload).
//!
//! Most controls are also clickable in the top-right on-screen panel. vsync is
//! off, so the fps counter shows the real ceiling, not the monitor refresh.
//!
//! Controls:
//! - drag left mouse: orbit the camera; scroll: zoom
//! - `+` / `-`: add / remove 200 critters (hold to repeat); panel has a slider
//! - `T`: toggle observe / combat
//! - `R`: toggle attack desync (combat) — on = min+random cooldowns, off =
//!   all predators in lockstep (a synchronized saturation spike to stress-test)
//! - `O`: toggle separation (no two critters share a space; one cull/critter)
//! - `[` / `]`: shrink / grow the radius (vision in observe, attack in combat)
//! - `M`: cycle the index structure — binary-3D `Tree3` / `Octree3` (8-way) /
//!   projection (one 2D `Tree` on xy + z-reject, the author's variant)
//! - `G`: cycle the render path — GPU-instanced spheres / round billboards /
//!   square billboards / NO RENDER (CPU only, to read the CPU's fps ceiling)
//! - `C`: cycle cull repetitions (1/50/200/1000) — averages the per-cull time
//! - `B`: toggle the leaf-box wireframes (extruded columns in projection mode)
//! - `Space`: pause, `Esc`: quit
//!
//! Env: CRITTERS3D_MAX_FRAMES=N exits after N frames (smoke testing);
//! CRITTERS3D_RENDER=instanced|billboards|square|none initial render path;
//! CRITTERS3D_COMBAT=1 starts in combat; CRITTERS3D_SEP=1 starts with separation.

use std::time::Instant;

use macroquad::camera::Camera;
use macroquad::miniquad::PassAction;
use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets};
use macroquad::window::get_internal_gl;

use vectorial_hash::{
    Aabb, CellState, Octree3, Point, Point3, Positioned, Positioned3, Rect, Shape, Shape3, Sphere3,
    Tree, Tree3,
};
use vectorial_hash_demos::instanced3d::{Instance, InstancedRenderer, Mode as RenderGeom};

const WORLD: f32 = 200.0;
const MARGIN: f32 = 4.0;
const ITEM_LIMIT: usize = 16;

// On-screen control panel (top-right) size.
const UI_W: f32 = 290.0;
const UI_H: f32 = 360.0;

// Combat-mode tuning.
const COOLDOWN: f32 = 0.7; // minimum seconds between a predator's attacks
const COOLDOWN_JITTER: f32 = 0.8; // + up to this much random, so attacks desync
const ATTACK_LIFE: f32 = 0.35; // attack-sphere fade time
const BURST_LIFE: f32 = 0.30; // kill-marker fade time
const FLASH_LIFE: f32 = 0.25; // freshly respawned critter flash time

// Separation ("no two critters in the same space") tuning.
const SEP_R: f32 = 5.0; // neighbours within this radius push apart
const SEP_STRENGTH: f32 = 0.25; // push scale per frame
const SEP_MAX: f32 = 2.0; // clamp per-frame displacement (avoids jitter blowups)

/// Which simulation the demo runs. `Observe` is the vision-cull showcase
/// (one sphere from the centre); `Combat` makes the red critters predators
/// that attack nearby prey — each attack is an index cull, so this is the
/// "many queries per frame" workload where the index cost actually shows.
#[derive(Clone, Copy, PartialEq)]
enum SimMode { Observe, Combat }

/// The two attack shapes a predator can throw — a round sphere or a tall
/// "drop" (vertical ellipsoid). Each gets its own colour so they read apart.
#[derive(Clone, Copy, PartialEq)]
enum AttackKind { Sphere, Drop }
impl AttackKind {
    /// Half of the predators throw drops (by id parity) so both appear.
    fn for_predator(id: u32) -> Self { if id % 2 == 0 { AttackKind::Sphere } else { AttackKind::Drop } }
}

/// Ellipsoid radii for a drop of base radius `r` — narrower in x/z, taller in y.
fn drop_radii(r: f32) -> (f32, f32, f32) { (r * 0.7, r * 1.5, r * 0.7) }

/// A vertical-ellipsoid attack volume (the "drop"). Reduces to a unit sphere
/// under per-axis scaling, so the sphere-vs-AABB classify carries straight over.
struct DropShape { cx: f64, cy: f64, cz: f64, rx: f64, ry: f64, rz: f64 }
impl Shape3 for DropShape {
    fn bounding_box(&self) -> Aabb {
        Aabb::new(self.cx - self.rx, self.cy - self.ry, self.cz - self.rz, 2.0 * self.rx, 2.0 * self.ry, 2.0 * self.rz)
    }
    fn contains_point(&self, p: Point3) -> bool {
        let (dx, dy, dz) = ((p.x - self.cx) / self.rx, (p.y - self.cy) / self.ry, (p.z - self.cz) / self.rz);
        dx * dx + dy * dy + dz * dz <= 1.0
    }
    fn classify_aabb(&self, b: &Aabb) -> CellState {
        let nx = (self.cx.clamp(b.x, b.x_max()) - self.cx) / self.rx;
        let ny = (self.cy.clamp(b.y, b.y_max()) - self.cy) / self.ry;
        let nz = (self.cz.clamp(b.z, b.z_max()) - self.cz) / self.rz;
        if nx * nx + ny * ny + nz * nz > 1.0 { return CellState::Out; }
        let fx = (if (self.cx - b.x).abs() > (self.cx - b.x_max()).abs() { b.x } else { b.x_max() } - self.cx) / self.rx;
        let fy = (if (self.cy - b.y).abs() > (self.cy - b.y_max()).abs() { b.y } else { b.y_max() } - self.cy) / self.ry;
        let fz = (if (self.cz - b.z).abs() > (self.cz - b.z_max()).abs() { b.z } else { b.z_max() } - self.cz) / self.rz;
        if fx * fx + fy * fy + fz * fz <= 1.0 { CellState::In } else { CellState::Maybe }
    }
}

/// A live attack volume (predator emits one; prey inside are culled & killed).
struct Attack { center: Vec3, radius: f32, age: f32, kind: AttackKind }

/// Hold-to-repeat for a key (OS-style): fires once on the press edge, then —
/// after `DELAY` held — repeats every `RATE` while the key stays down. Used
/// for the `+`/`-` population keys so you can hold to ramp the count.
struct KeyRepeat { held: f32, acc: f32 }
impl KeyRepeat {
    fn new() -> Self { KeyRepeat { held: 0.0, acc: 0.0 } }
    fn fires(&mut self, pressed: bool, down: bool, dt: f32) -> bool {
        const DELAY: f32 = 0.5;
        const RATE: f32 = 0.05;
        if !down {
            self.held = 0.0;
            self.acc = 0.0;
            return pressed;
        }
        self.held += dt;
        let mut fire = pressed;
        if self.held > DELAY {
            self.acc += dt;
            if self.acc >= RATE { self.acc -= RATE; fire = true; }
        }
        fire
    }
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.max(1)) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.unit() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct C3 { id: u32, p: Point3 }
impl Positioned3 for C3 { fn position(&self) -> Point3 { self.p } }

/// A critter projected onto the xy plane (the author's projection variant),
/// carrying its z so the narrowphase can reject by depth and test exactly.
#[derive(Clone, Copy)]
struct P2 { id: u32, p: Point, z: f64 }
impl Positioned for P2 { fn position(&self) -> Point { self.p } }

/// The sphere's shadow on the xy plane — the 2D broadphase for projection mode.
struct Disc { cx: f64, cy: f64, r: f64 }
impl Shape for Disc {
    fn bounding_box(&self) -> Rect { Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r) }
    fn contains_point(&self, p: Point) -> bool {
        let (dx, dy) = (p.x - self.cx, p.y - self.cy);
        dx * dx + dy * dy <= self.r * self.r
    }
}

/// How the critters are drawn — both are the GPU-instanced path (one draw
/// call, see [`vectorial_hash_demos::instanced3d`]). The old per-critter
/// immediate `draw_sphere` mode was dropped once instancing was confirmed
/// working; it pinned the demo to ~10 fps at a few thousand critters.
#[derive(Clone, Copy, PartialEq)]
enum RenderMode { InstancedSpheres, Billboards, SquareBillboards, None }
impl RenderMode {
    fn next(self) -> Self {
        match self {
            RenderMode::InstancedSpheres => RenderMode::Billboards,
            RenderMode::Billboards => RenderMode::SquareBillboards,
            RenderMode::SquareBillboards => RenderMode::None,
            RenderMode::None => RenderMode::InstancedSpheres,
        }
    }
    fn label(self) -> &'static str {
        match self {
            RenderMode::InstancedSpheres => "instanced spheres (GPU)",
            RenderMode::Billboards => "instanced billboards, round (GPU)",
            RenderMode::SquareBillboards => "instanced billboards, square/fast (GPU)",
            RenderMode::None => "NO RENDER (CPU only)",
        }
    }
    /// `None` when the critters aren't drawn (isolates CPU); else the geometry.
    fn geom(self) -> Option<RenderGeom> {
        match self {
            RenderMode::InstancedSpheres => Some(RenderGeom::Spheres),
            RenderMode::Billboards => Some(RenderGeom::Billboards),
            RenderMode::SquareBillboards => Some(RenderGeom::BillboardsSquare),
            RenderMode::None => Option::None,
        }
    }
    fn from_env() -> Self {
        match std::env::var("CRITTERS3D_RENDER").ok().as_deref() {
            Some("billboards") => RenderMode::Billboards,
            Some("square") => RenderMode::SquareBillboards,
            Some("none") => RenderMode::None,
            _ => RenderMode::InstancedSpheres,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Structure { Binary3, Octree, Projection }
impl Structure {
    fn next(self) -> Self {
        match self { Structure::Binary3 => Structure::Octree, Structure::Octree => Structure::Projection, Structure::Projection => Structure::Binary3 }
    }
    fn label(self) -> &'static str {
        match self { Structure::Binary3 => "Tree3 (binary-3D)", Structure::Octree => "Octree3 (8-way)", Structure::Projection => "projection (1×2D + z-reject)" }
    }
}

struct Critter {
    pos: Vec3,
    vel: Vec3,
    kind: u8,      // 0,1,2 — colour; in combat, kind 1 = predator
    cooldown: f32, // combat: time until this predator's next attack
    flash: f32,    // combat: remaining flash time after respawn
}

fn world_aabb() -> Aabb { Aabb::new(0.0, 0.0, 0.0, WORLD as f64, WORLD as f64, WORLD as f64) }

/// A 3D AABB → (centre, size) pair for `draw_cube_wires`.
fn aabb_box(b: &Aabb) -> (Vec3, Vec3) {
    (
        vec3((b.x + b.w * 0.5) as f32, (b.y + b.h * 0.5) as f32, (b.z + b.d * 0.5) as f32),
        vec3(b.w as f32, b.h as f32, b.d as f32),
    )
}

fn kind_color(k: u8, lit: bool) -> Color {
    if lit { return Color::new(1.0, 0.95, 0.3, 1.0); }
    match k {
        0 => Color::new(0.40, 0.75, 1.00, 1.0),
        1 => Color::new(1.00, 0.45, 0.45, 1.0),
        _ => Color::new(0.55, 1.00, 0.60, 1.0),
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "vectorial-hash critters 3D".to_owned(),
        window_width: 1600,
        window_height: 1000,
        // vsync off: the FPS counter shows the real ceiling (CPU and/or GPU)
        // instead of being pinned to the monitor refresh — needed to see what
        // "NO RENDER (CPU only)" actually reaches.
        platform: macroquad::miniquad::conf::Platform { swap_interval: Some(0), ..Default::default() },
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut rng = Rng::new(42);
    let mut critters: Vec<Critter> = Vec::new();
    let spawn = |rng: &mut Rng, kind: u8| -> Critter {
        let p = vec3(
            rng.range(MARGIN, WORLD - MARGIN),
            rng.range(MARGIN, WORLD - MARGIN),
            rng.range(MARGIN, WORLD - MARGIN),
        );
        let speed = rng.range(20.0, 55.0);
        // random unit direction
        let a = rng.range(0.0, std::f32::consts::TAU);
        let z = rng.range(-1.0, 1.0);
        let s = (1.0 - z * z).max(0.0).sqrt();
        let v = vec3(s * a.cos(), s * a.sin(), z) * speed;
        let cooldown = if kind == 1 { rng.range(0.0, COOLDOWN + COOLDOWN_JITTER) } else { 0.0 };
        Critter { pos: p, vel: v, kind, cooldown, flash: 0.0 }
    };
    for i in 0..2400 {
        critters.push(spawn(&mut rng, (i % 3) as u8));
    }

    // Persistent binary-3D index. Instead of rebuilding it every frame, we
    // keep it across frames and `update` each critter's position in place
    // (ascend-to-LCA) — most critters stay in their leaf, so this is far
    // cheaper than a full rebuild. `tree_pos[i]` is where critter i currently
    // sits in the tree (the `old` position `update` needs to find it).
    let pt3 = |c: &Critter| Point3::new(c.pos.x as f64, c.pos.y as f64, c.pos.z as f64);
    let mut tree = Tree3::<C3>::new(world_aabb(), ITEM_LIMIT);
    let mut tree_pos: Vec<Point3> = Vec::with_capacity(critters.len());
    for (i, c) in critters.iter().enumerate() {
        let p = pt3(c);
        tree.insert(C3 { id: i as u32, p });
        tree_pos.push(p);
    }

    // Camera orbit state.
    let mut yaw: f32 = 0.7;
    let mut pitch: f32 = 0.5;
    let mut dist: f32 = WORLD * 2.2;
    let mut last_mouse = mouse_position();

    let mut vision_r: f32 = 40.0;
    let mut paused = false;
    let mut show_boxes = false;
    let mut structure = Structure::Binary3;
    let mut render_mode = RenderMode::from_env();
    let cull_rep_steps = [1usize, 50, 200, 1000];
    let mut cull_rep_idx = 0usize;

    // Combat-mode state.
    let mut sim_mode = if std::env::var("CRITTERS3D_COMBAT").is_ok() { SimMode::Combat } else { SimMode::Observe };
    let mut random_attacks = true; // desync cooldowns; off = synchronized saturation
    let mut prev_random = random_attacks; // to reseed cooldowns when this flips
    let mut separation = std::env::var("CRITTERS3D_SEP").is_ok(); // push critters apart (heavy)
    let mut fps_cap = false; // V: cap to the monitor refresh so the GPU isn't pegged
    let mut pop_f = critters.len() as f32; // target population (driven by +/- and the slider)
    let mut attack_r: f32 = 22.0;
    let mut attacks: Vec<Attack> = Vec::new();
    let mut bursts: Vec<(Vec3, f32)> = Vec::new(); // kill markers (pos, age)
    let mut kills: u64 = 0;
    // Latched stats of the last attack wave (so a synchronized burst stays
    // readable between waves, when most frames have zero attacks).
    let mut last_wave_attacks = 0usize;
    let mut last_wave_us = 0.0f64;

    // The instanced renderer owns GPU resources (shaders, base meshes); build
    // it once from the raw miniquad context. Shader compilation happens here,
    // so a GLSL error surfaces at startup (and in the headless smoke test).
    let mut renderer = {
        let gl = unsafe { get_internal_gl() };
        InstancedRenderer::new(gl.quad_context)
    };
    let mut frame: u64 = 0;
    let max_frames: Option<u64> = std::env::var("CRITTERS3D_MAX_FRAMES").ok().and_then(|s| s.parse().ok());

    // Hold-to-repeat state for the +/- population and [ ] radius keys.
    let mut add_rep = KeyRepeat::new();
    let mut sub_rep = KeyRepeat::new();
    let mut dec_rep = KeyRepeat::new();
    let mut inc_rep = KeyRepeat::new();
    // Averaged FPS (raw get_fps() is the instantaneous 1/frame_time and jitters);
    // accumulate real frame time and refresh the shown value once per second.
    let mut fps_accum_t = 0.0f32;
    let mut fps_accum_n = 0u32;
    let mut fps_display = 0.0f32;
    // Rolling average of the per-cull time (a single frame's measurement is
    // noisy — this smooths it into a real average without rounding to ~1 µs).
    let mut cull_us_avg = 0.0f64;
    // Rolling CPU-phase timings (ms total, µs for sim/prep) for the bound check.
    let mut cpu_ms_avg = 0.0f64;
    let mut sim_us_avg = 0.0f64;
    let mut prep_us_avg = 0.0f64;

    loop {
        let dt = (get_frame_time()).min(1.0 / 30.0);

        // Averaged FPS over ~1 s (uses the real, unclamped frame time).
        fps_accum_t += get_frame_time();
        fps_accum_n += 1;
        if fps_accum_t >= 1.0 {
            fps_display = fps_accum_n as f32 / fps_accum_t;
            fps_accum_t = 0.0;
            fps_accum_n = 0;
        }
        // CPU wall-clock for this frame's work (everything up to the vsync
        // wait) — to tell whether we're CPU-bound vs GPU-bound / vsync-capped.
        let frame_t0 = Instant::now();
        let sim_us;
        let prep_us;

        // Control panel rect (top-right); the camera ignores the mouse over it.
        let (ui_x, ui_y) = (screen_width() - UI_W - 8.0, 8.0);
        let mp_now = mouse_position();
        let over_ui = mp_now.0 >= ui_x && mp_now.0 <= ui_x + UI_W && mp_now.1 >= ui_y && mp_now.1 <= ui_y + UI_H;

        // --- input ---
        if is_key_pressed(KeyCode::Escape) { break; }
        if is_key_pressed(KeyCode::Space) { paused = !paused; }
        if is_key_pressed(KeyCode::B) { show_boxes = !show_boxes; }
        if is_key_pressed(KeyCode::M) { structure = structure.next(); }
        if is_key_pressed(KeyCode::G) { render_mode = render_mode.next(); }
        if is_key_pressed(KeyCode::T) { sim_mode = match sim_mode { SimMode::Observe => SimMode::Combat, SimMode::Combat => SimMode::Observe }; }
        if is_key_pressed(KeyCode::R) { random_attacks = !random_attacks; }
        if is_key_pressed(KeyCode::C) { cull_rep_idx = (cull_rep_idx + 1) % cull_rep_steps.len(); }
        if is_key_pressed(KeyCode::O) { separation = !separation; }
        if is_key_pressed(KeyCode::V) { fps_cap = !fps_cap; }
        // +/- ramp the population: one step per press, or hold to auto-repeat.
        let add = add_rep.fires(
            is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd),
            is_key_down(KeyCode::Equal) || is_key_down(KeyCode::KpAdd),
            dt,
        );
        // +/- snap to the next/previous multiple of 200 (so a mouse-dragged
        // slider value like 3455 lands back on a round number).
        if add { pop_f = (((pop_f / 200.0).floor() + 1.0) * 200.0).min(80000.0); }
        let sub = sub_rep.fires(
            is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract),
            is_key_down(KeyCode::Minus) || is_key_down(KeyCode::KpSubtract),
            dt,
        );
        if sub { pop_f = (((pop_f / 200.0).ceil() - 1.0) * 200.0).max(0.0); }
        // [ / ] step the radius (vision in observe, attack in combat) in
        // multiples of 5 (snapping off-grid slider values), hold to repeat.
        const R_STEP: f32 = 5.0;
        let dec = dec_rep.fires(is_key_pressed(KeyCode::LeftBracket), is_key_down(KeyCode::LeftBracket), dt);
        let inc = inc_rep.fires(is_key_pressed(KeyCode::RightBracket), is_key_down(KeyCode::RightBracket), dt);
        if dec {
            match sim_mode {
                SimMode::Observe => vision_r = (((vision_r / R_STEP).ceil() - 1.0) * R_STEP).max(R_STEP),
                SimMode::Combat => attack_r = (((attack_r / R_STEP).ceil() - 1.0) * R_STEP).max(R_STEP),
            }
        }
        if inc {
            match sim_mode {
                SimMode::Observe => vision_r = (((vision_r / R_STEP).floor() + 1.0) * R_STEP).min(WORLD),
                SimMode::Combat => attack_r = (((attack_r / R_STEP).floor() + 1.0) * R_STEP).min(WORLD * 0.5),
            }
        }
        let mp = mouse_position();
        if is_mouse_button_down(MouseButton::Left) && !over_ui {
            yaw += (mp.0 - last_mouse.0) * 0.01;
            pitch = (pitch + (mp.1 - last_mouse.1) * 0.01).clamp(-1.4, 1.4);
        }
        last_mouse = mp;
        let scroll = mouse_wheel().1;
        if scroll != 0.0 && !over_ui { dist = (dist * (1.0 - scroll.signum() * 0.1)).clamp(WORLD * 0.6, WORLD * 6.0); }

        // Reconcile population (and the index) to the target — drives both the
        // +/- keys and the panel slider through one path.
        let target = pop_f.round().max(0.0) as usize;
        while critters.len() < target {
            let id = critters.len() as u32;
            let c = spawn(&mut rng, (id % 3) as u8);
            let p = pt3(&c);
            tree.insert(C3 { id, p });
            tree_pos.push(p);
            critters.push(c);
        }
        while critters.len() > target {
            critters.pop();
            let i = critters.len();
            tree.remove(tree_pos[i], |c| c.id == i as u32);
            tree_pos.pop();
        }

        // --- simulate ---
        let t_sim = Instant::now();
        if !paused {
            for c in critters.iter_mut() {
                let mut np = c.pos + c.vel * dt;
                for axis in 0..3 {
                    if np[axis] < MARGIN { np[axis] = MARGIN; c.vel[axis] = -c.vel[axis]; }
                    if np[axis] > WORLD - MARGIN { np[axis] = WORLD - MARGIN; c.vel[axis] = -c.vel[axis]; }
                }
                c.pos = np;
                // Cooldown only ticks in combat — otherwise time spent in
                // observe would drain every predator to 0 and they'd all fire
                // together the instant you switch to combat.
                if sim_mode == SimMode::Combat { c.cooldown -= dt; }
                c.flash -= dt;
            }
            // Separation: no two critters share the same space. For each one,
            // cull its neighbourhood from the (last-frame) index and push away
            // from anyone closer than SEP_R. One cull per critter -> heavy at
            // high counts, hence the toggle. Pushes are gathered then applied.
            if separation {
                let mut push = vec![Vec3::ZERO; critters.len()];
                for i in 0..critters.len() {
                    let p = critters[i].pos;
                    let s = Sphere3::new(p.x as f64, p.y as f64, p.z as f64, SEP_R as f64);
                    for h in tree.cull(&s) {
                        let j = h.id as usize;
                        if j == i { continue; }
                        let d = p - critters[j].pos;
                        let dist = d.length();
                        if dist > 1e-4 && dist < SEP_R {
                            push[i] += d / dist * (SEP_R - dist);
                        }
                    }
                }
                for i in 0..critters.len() {
                    let mut mv = push[i] * SEP_STRENGTH;
                    let l = mv.length();
                    if l > SEP_MAX { mv = mv / l * SEP_MAX; }
                    let mut np = critters[i].pos + mv;
                    for axis in 0..3 { np[axis] = np[axis].clamp(MARGIN, WORLD - MARGIN); }
                    critters[i].pos = np;
                }
            }
            // age and retire combat effects
            for a in attacks.iter_mut() { a.age += dt; }
            attacks.retain(|a| a.age < ATTACK_LIFE);
            for b in bursts.iter_mut() { b.1 += dt; }
            bursts.retain(|b| b.1 < BURST_LIFE);
        }
        sim_us = t_sim.elapsed().as_secs_f64() * 1e6;

        // Sync the persistent index to the critters' new positions (move each
        // in place via ascend-to-LCA). This replaces the per-frame full
        // rebuild — `sync_us` is what the Binary3 / combat paths report as
        // "build". Combat respawns (which jump a critter) are picked up by the
        // next frame's sync.
        let t_sync = Instant::now();
        if !paused {
            for i in 0..critters.len() {
                let np = pt3(&critters[i]);
                tree.update(tree_pos[i], |c| c.id == i as u32, |c| c.p = np);
                tree_pos[i] = np;
            }
        }
        let sync_us = t_sync.elapsed().as_secs_f64() * 1e6;

        // --- (re)build the chosen index and run a vision cull from the centre ---
        let observer = vec3(WORLD * 0.5, WORLD * 0.5, WORLD * 0.5);
        let (ox, oy, oz, r) = (observer.x as f64, observer.y as f64, observer.z as f64, vision_r as f64);
        let cull_reps = cull_rep_steps[cull_rep_idx];
        // Optional self-check: the persistent tree's vision cull must match a
        // brute-force scan (catches any update/insert/remove bookkeeping bug).
        if std::env::var("CRITTERS3D_VERIFY").is_ok() {
            let s = Sphere3::new(ox, oy, oz, r);
            let mut got: Vec<u32> = tree.cull(&s).iter().map(|c| c.id).collect();
            let mut want: Vec<u32> = (0..critters.len() as u32).filter(|&i| {
                let c = &critters[i as usize];
                let (dx, dy, dz) = (c.pos.x as f64 - ox, c.pos.y as f64 - oy, c.pos.z as f64 - oz);
                dx * dx + dy * dy + dz * dz <= r * r
            }).collect();
            got.sort_unstable();
            want.sort_unstable();
            if got != want {
                eprintln!("VERIFY mismatch frame {frame}: tree {} vs brute {} (pop {})", got.len(), want.len(), critters.len());
            }
        }
        let mut lit = vec![false; critters.len()];
        let mut boxes: Vec<(Vec3, Vec3)> = Vec::new();
        let stat_line: String;
        let cand_n: Option<usize>; // broadphase candidates (projection only)
        let r2 = r * r;
        // Build time, and the cull averaged over `cull_reps` (a single sphere
        // cull is microseconds — repeating it gives a stable, readable number
        // that isolates the structure from the render cost).
        let t_build_us: f64;
        let t_cull_us: f64;
        let mut frame_attacks = 0usize; // combat: predators that attacked this frame

        if sim_mode == SimMode::Observe {
        match structure {
            Structure::Binary3 => {
                // Persistent tree, already synced above — just cull it.
                t_build_us = sync_us;
                let tc = Instant::now();
                for rep in 0..cull_reps {
                    let sphere = Sphere3::new(ox + rep as f64 * 0.01, oy, oz, r);
                    let hits = tree.cull(&sphere);
                    if rep == 0 { for c in hits { lit[c.id as usize] = true; } }
                }
                t_cull_us = tc.elapsed().as_secs_f64() * 1e6 / cull_reps as f64;
                if show_boxes { tree.visit_leaves(|l| boxes.push(aabb_box(&l.bbox))); }
                cand_n = None;
                stat_line = format!("Tree3 persistent/update: {:>6} leaves, {:>6} arena (item_limit {})", tree.leaf_count(), tree.node_count(), ITEM_LIMIT);
            }
            Structure::Octree => {
                // Comparison structure — rebuilt each frame (no persistent update).
                let tb = Instant::now();
                let mut t = Octree3::<C3>::new(world_aabb(), ITEM_LIMIT);
                for (i, c) in critters.iter().enumerate() {
                    t.insert(C3 { id: i as u32, p: pt3(c) });
                }
                t_build_us = tb.elapsed().as_secs_f64() * 1e6;
                let tc = Instant::now();
                for rep in 0..cull_reps {
                    let sphere = Sphere3::new(ox + rep as f64 * 0.01, oy, oz, r);
                    let hits = t.cull(&sphere);
                    if rep == 0 { for c in hits { lit[c.id as usize] = true; } }
                }
                t_cull_us = tc.elapsed().as_secs_f64() * 1e6 / cull_reps as f64;
                if show_boxes { t.visit_leaves(|l| boxes.push(aabb_box(&l.bbox))); }
                cand_n = None;
                stat_line = format!("Octree3 rebuilt/frame: {:>6} leaves, {:>6} arena (item_limit {})", t.leaf_count(), t.node_count(), ITEM_LIMIT);
            }
            Structure::Projection => {
                // Author's variant: index the xy projection in a 2D Tree, cull
                // the sphere's shadow (a disc), then z-reject + exact 3D test.
                let tb = Instant::now();
                let mut t = Tree::<P2>::new(Rect::new(0.0, 0.0, WORLD as f64, WORLD as f64), ITEM_LIMIT);
                for (i, c) in critters.iter().enumerate() {
                    t.insert(P2 { id: i as u32, p: Point::new(c.pos.x as f64, c.pos.y as f64), z: c.pos.z as f64 });
                }
                t_build_us = tb.elapsed().as_secs_f64() * 1e6;
                // The query is broadphase (disc cull) + narrowphase (z-reject +
                // exact 3D) — time the whole thing, that's the variant's cost.
                let tc = Instant::now();
                let mut nc = 0;
                for rep in 0..cull_reps {
                    let disc = Disc { cx: ox + rep as f64 * 0.01, cy: oy, r };
                    let cand = t.cull(&disc);
                    if rep == 0 { nc = cand.len(); }
                    for p2 in &cand {
                        let (dx, dy, dz) = (p2.p.x - ox, p2.p.y - oy, p2.z - oz);
                        let inside = dx * dx + dy * dy + dz * dz <= r2;
                        if rep == 0 && inside { lit[p2.id as usize] = true; }
                    }
                }
                t_cull_us = tc.elapsed().as_secs_f64() * 1e6 / cull_reps as f64;
                // boxes: the 2D leaf rects, extruded through the full z depth.
                if show_boxes {
                    t.visit_leaves(|_, l| {
                        let b = l.bbox;
                        let c = vec3((b.x + b.width * 0.5) as f32, (b.y + b.height * 0.5) as f32, WORLD * 0.5);
                        boxes.push((c, vec3(b.width as f32, b.height as f32, WORLD)));
                    });
                }
                cand_n = Some(nc);
                stat_line = format!("{}: {:>6} leaves, {:>6} arena (item_limit {})", structure.label(), t.leaf_count(), t.node_count(), ITEM_LIMIT);
            }
        }
        } else {
            // === COMBAT: predators (kind 1) attack nearby prey; each attack is
            // an index cull against the persistent tree (synced above), so this
            // is the "many queries per frame" workload — no rebuild.
            t_build_us = sync_us;
            let mut killed = vec![false; critters.len()];
            let mut cull_us = 0.0f64;
            let t_wave = Instant::now();
            if !paused {
                for i in 0..critters.len() {
                    if critters[i].kind != 1 || critters[i].cooldown > 0.0 { continue; }
                    let center = critters[i].pos;
                    critters[i].cooldown = COOLDOWN + if random_attacks { rng.range(0.0, COOLDOWN_JITTER) } else { 0.0 };
                    let akind = AttackKind::for_predator(i as u32);
                    attacks.push(Attack { center, radius: attack_r, age: 0.0, kind: akind });
                    frame_attacks += 1;
                    let (cx, cy, cz) = (center.x as f64, center.y as f64, center.z as f64);
                    let tc = Instant::now();
                    let hit_ids: Vec<usize> = match akind {
                        AttackKind::Sphere => {
                            let s = Sphere3::new(cx, cy, cz, attack_r as f64);
                            tree.cull(&s).iter().map(|h| h.id as usize).collect()
                        }
                        AttackKind::Drop => {
                            let (rx, ry, rz) = drop_radii(attack_r);
                            let s = DropShape { cx, cy, cz, rx: rx as f64, ry: ry as f64, rz: rz as f64 };
                            tree.cull(&s).iter().map(|h| h.id as usize).collect()
                        }
                    };
                    cull_us += tc.elapsed().as_secs_f64() * 1e6;
                    for j in hit_ids {
                        if j != i && !killed[j] && critters[j].kind != 1 { killed[j] = true; }
                    }
                }
                // apply kills: drop a burst marker, respawn as prey, flash, count
                for j in 0..critters.len() {
                    if killed[j] {
                        bursts.push((critters[j].pos, 0.0));
                        let k = if rng.unit() < 0.5 { 0u8 } else { 2u8 };
                        critters[j] = spawn(&mut rng, k);
                        critters[j].flash = FLASH_LIFE;
                        kills += 1;
                    }
                }
            }
            // Latch the last wave that actually fired so a synchronized burst
            // (a spike every COOLDOWN seconds) stays on screen long enough to read.
            if frame_attacks > 0 {
                last_wave_attacks = frame_attacks;
                last_wave_us = t_wave.elapsed().as_secs_f64() * 1e6;
            }
            t_cull_us = if frame_attacks > 0 { cull_us / frame_attacks as f64 } else { 0.0 };
            cand_n = None;
            let predators = critters.iter().filter(|c| c.kind == 1).count();
            stat_line = format!("combat: {:>5} predators | last wave: {:>4} attacks resolved in {:>7.0} us", predators, last_wave_attacks, last_wave_us);
        }
        let seen_n = lit.iter().filter(|&&b| b).count();
        // Feed the rolling cull-time average (skip frames with no measurement,
        // e.g. combat frames where no predator was off cooldown).
        if t_cull_us > 0.0 {
            cull_us_avg = if cull_us_avg <= 0.0 { t_cull_us } else { cull_us_avg * 0.9 + t_cull_us * 0.1 };
        }

        // --- render 3D ---
        clear_background(Color::new(0.05, 0.06, 0.09, 1.0));
        let eye = observer + vec3(
            dist * pitch.cos() * yaw.cos(),
            dist * pitch.sin(),
            dist * pitch.cos() * yaw.sin(),
        );
        let cam = Camera3D { position: eye, up: vec3(0.0, 1.0, 0.0), target: observer, ..Default::default() };
        set_camera(&cam);
        // Camera basis for billboards, and the proj*view matrix the instanced
        // shader needs (the same one macroquad uses for this camera).
        let mvp = cam.matrix();
        let fwd = (observer - eye).normalize();
        let cam_right = fwd.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let cam_up = cam_right.cross(fwd).normalize();

        // world box
        draw_cube_wires(observer, vec3(WORLD, WORLD, WORLD), Color::new(0.3, 0.35, 0.45, 1.0));

        // optional leaf-box wireframes (collected during the cull above)
        if show_boxes {
            for (c, sz) in &boxes {
                draw_cube_wires(*c, *sz, Color::new(0.18, 0.22, 0.3, 1.0));
            }
        }

        // vision sphere + observer marker — only meaningful in observe mode
        if sim_mode == SimMode::Observe {
            draw_sphere_wires(observer, vision_r, None, Color::new(1.0, 0.9, 0.3, 0.5));
            draw_sphere(observer, 3.0, None, WHITE);
        }

        // critters — GPU-instanced (spheres/billboards), or skipped entirely in
        // NO RENDER mode (so the frame is pure CPU: sim + index + cull).
        let t_prep = Instant::now();
        if let Some(geom) = render_mode.geom() {
            // Per-critter colour & radius. Observe lights the seen critters;
            // combat reds the predators and flashes fresh respawns.
            let mut cols: Vec<Color> = Vec::with_capacity(critters.len());
            let mut rads: Vec<f32> = Vec::with_capacity(critters.len());
            for (i, c) in critters.iter().enumerate() {
                let (col, rad) = match sim_mode {
                    SimMode::Observe => (kind_color(c.kind, lit[i]), if lit[i] { 2.2 } else { 1.5 }),
                    SimMode::Combat => {
                        if c.flash > 0.0 {
                            (Color::new(1.0, 1.0, 1.0, 1.0), 2.6)
                        } else if c.kind == 1 {
                            (Color::new(1.0, 0.35, 0.30, 1.0), 2.3)
                        } else {
                            (kind_color(c.kind, false), 1.4)
                        }
                    }
                };
                cols.push(col);
                rads.push(rad);
            }
            let instances: Vec<Instance> = critters.iter().enumerate().map(|(i, c)| {
                Instance::new(c.pos, rads[i], [cols[i].r, cols[i].g, cols[i].b, cols[i].a])
            }).collect();
            prep_us = t_prep.elapsed().as_secs_f64() * 1e6;
            // Flush macroquad's batch (renders the world box / wires drawn
            // above), then issue the instanced draw into the same default pass
            // with depth preserved.
            let mut gl = unsafe { get_internal_gl() };
            gl.flush();
            let ctx = gl.quad_context;
            ctx.begin_default_pass(PassAction::Nothing);
            renderer.draw(ctx, geom, &instances, mvp, cam_right, cam_up);
            ctx.end_render_pass();
        } else {
            prep_us = 0.0;
        }

        if sim_mode == SimMode::Observe {
            // sight-lines for the seen critters
            for (i, c) in critters.iter().enumerate() {
                if lit[i] { draw_line_3d(c.pos, observer, Color::new(1.0, 0.85, 0.3, 0.25)); }
            }
        } else {
            // combat effects: attack volumes (expand + fade) and kill bursts.
            // Sphere attacks are orange, drop attacks cyan + teardrop-shaped.
            for a in &attacks {
                let f = (a.age / ATTACK_LIFE).clamp(0.0, 1.0);
                let grow = 0.7 + 0.5 * f;
                let alpha = (1.0 - f) * 0.6;
                match a.kind {
                    AttackKind::Sphere => {
                        draw_sphere_wires(a.center, a.radius * grow, None, Color::new(1.0, 0.7 - 0.5 * f, 0.2, alpha));
                    }
                    AttackKind::Drop => {
                        let (rx, ry, _rz) = drop_radii(a.radius);
                        let col = Color::new(0.25, 0.85, 1.0, alpha); // cyan
                        let bulb = a.center + vec3(0.0, -ry * 0.25 * grow, 0.0);
                        draw_sphere_wires(bulb, rx * grow, None, col);
                        let tip = a.center + vec3(0.0, ry * grow, 0.0);
                        for k in 0..6 {
                            let ang = std::f32::consts::TAU * k as f32 / 6.0;
                            let edge = bulb + vec3(ang.cos() * rx * grow, ry * 0.2 * grow, ang.sin() * rx * grow);
                            draw_line_3d(edge, tip, col);
                        }
                    }
                }
            }
            for (p, age) in &bursts {
                let f = (age / BURST_LIFE).clamp(0.0, 1.0);
                draw_sphere_wires(*p, 2.0 + 9.0 * f, None, Color::new(1.0, 1.0, 1.0, (1.0 - f) * 0.8));
            }
        }

        // (CPU timing is taken at the very end of the frame — see below — so it
        // includes the HUD/UI; the HUD shows the previous frame's rolling value.)

        // --- HUD (2D overlay) ---
        set_default_camera();
        let hud = |y: f32, s: String| draw_text(&s, 12.0, y, 20.0, Color::new(0.85, 0.9, 1.0, 1.0));
        // Numbers are right-aligned in fixed-width fields so they don't shift
        // the text sideways as they change (which made it jitter illegibly).
        let mode_str = if sim_mode == SimMode::Combat { "COMBAT " } else { "observe" };
        hud(24.0, format!("critters 3D  |  pop {:>6}  |  mode {}  |  fps {:>6.0}{}", critters.len(), mode_str, fps_display, if fps_cap { "  [capped 165]" } else { "" }));
        hud(46.0, stat_line);
        let info_str = match sim_mode {
            SimMode::Observe => match cand_n {
                Some(nc) => format!("vision r={:>3.0}  ->  {:>6} candidates -> {:>6} seen", vision_r, nc, seen_n),
                None => format!("vision r={:>3.0}  ->  {:>6} seen", vision_r, seen_n),
            },
            SimMode::Combat => format!("attack r={:>3.0}  |  {:>7} kills  |  {} [R]", attack_r, kills, if random_attacks { "desynced" } else { "SYNCED (saturation)" }),
        };
        hud(68.0, format!("{}{}", info_str, if paused { "   [PAUSED]" } else { "" }));
        let cull_note = match sim_mode {
            SimMode::Observe => format!("cull {:>8.3} us (rolling avg, x{:>4} reps)", cull_us_avg, cull_reps),
            SimMode::Combat => format!("cull {:>8.3} us (rolling avg, per attack)", cull_us_avg),
        };
        hud(90.0, format!("index: build {:>6.0} us | {}", t_build_us, cull_note));
        let render_note = match render_mode {
            RenderMode::InstancedSpheres => {
                let tris = renderer.sphere_triangles() as i64 * critters.len() as i64;
                format!("{}  |  {:>3} tris/sphere -> {:>5.2}M tris", render_mode.label(), renderer.sphere_triangles(), tris as f64 / 1e6)
            }
            RenderMode::None => render_mode.label().to_string(),
            _ => format!("{}  |  2 tris/critter", render_mode.label()),
        };
        hud(112.0, format!("render: {}  <- G switches", render_note));
        // With vsync off, the frame time ~= max(CPU, GPU). If the CPU fills
        // ~all of it, we're CPU-bound; if the frame is longer than the CPU work,
        // the GPU is the limit. "CPU ceiling" = the fps the CPU alone sustains.
        let frame_ms = if fps_display > 1.0 { 1000.0 / fps_display as f64 } else { cpu_ms_avg };
        let cpu_ceiling = if cpu_ms_avg > 0.0 { 1000.0 / cpu_ms_avg } else { 0.0 };
        let bound = if frame_ms > 0.0 && cpu_ms_avg >= 0.85 * frame_ms { "CPU-BOUND" } else { "GPU-bound" };
        hud(134.0, format!("cpu ~{:.2} ms (sim {:.0}+build {:.0}+prep {:.0} us) -> CPU ceiling ~{:.0} fps -> {}", cpu_ms_avg, sim_us_avg, t_build_us, prep_us_avg, cpu_ceiling, bound));
        hud(screen_height() - 18.0, "drag/zoom | +/-: pop | [ ]: radius | T: combat | R: sync | O: separation | V: fps cap | M: structure | G: render | C: reps | B: boxes | Space | Esc".to_string());

        // --- on-screen mouse controls (top-right panel; keys still work too) ---
        widgets::Window::new(hash!(), vec2(ui_x, ui_y), vec2(UI_W, UI_H))
            .label("controls (click / drag)")
            .titlebar(true)
            .movable(false)
            .ui(&mut root_ui(), |ui| {
                if ui.button(None, format!("mode: {} [T]", if sim_mode == SimMode::Combat { "COMBAT" } else { "observe" })) {
                    sim_mode = match sim_mode { SimMode::Observe => SimMode::Combat, SimMode::Combat => SimMode::Observe };
                }
                if ui.button(None, format!("structure: {} [M]", structure.label())) { structure = structure.next(); }
                if ui.button(None, format!("render: {} [G]", render_mode.label())) { render_mode = render_mode.next(); }
                ui.separator();
                if ui.button(None, format!("attacks: {} [R]", if random_attacks { "desynced" } else { "SYNCED" })) { random_attacks = !random_attacks; }
                if ui.button(None, format!("separation: {} [O]", if separation { "ON" } else { "off" })) { separation = !separation; }
                if ui.button(None, format!("fps cap: {} [V]", if fps_cap { "165" } else { "OFF" })) { fps_cap = !fps_cap; }
                if ui.button(None, format!("leaf boxes: {} [B]", if show_boxes { "ON" } else { "off" })) { show_boxes = !show_boxes; }
                if ui.button(None, format!("cull reps: {} [C]", cull_reps)) { cull_rep_idx = (cull_rep_idx + 1) % cull_rep_steps.len(); }
                ui.separator();
                ui.slider(hash!(), "population", 0f32..80000f32, &mut pop_f);
                match sim_mode {
                    SimMode::Observe => ui.slider(hash!(), "vision r", 6f32..WORLD, &mut vision_r),
                    SimMode::Combat => ui.slider(hash!(), "attack r", 4f32..(WORLD * 0.5), &mut attack_r),
                }
            });
        pop_f = pop_f.round().clamp(0.0, 80000.0);

        // Reseed predator cooldowns when the desync toggle flips (key or panel),
        // so the change is visible immediately (spread out vs lockstep).
        if random_attacks != prev_random {
            for c in critters.iter_mut() {
                if c.kind == 1 {
                    c.cooldown = if random_attacks { rng.range(0.0, COOLDOWN + COOLDOWN_JITTER) } else { COOLDOWN };
                }
            }
            prev_random = random_attacks;
        }

        // End-of-frame CPU wall-clock (includes the HUD + UI panel), rolling
        // averaged. Shown next frame by the HUD above.
        let ema = |avg: f64, x: f64| if avg <= 0.0 { x } else { avg * 0.9 + x * 0.1 };
        cpu_ms_avg = ema(cpu_ms_avg, frame_t0.elapsed().as_secs_f64() * 1000.0);
        sim_us_avg = ema(sim_us_avg, sim_us);
        prep_us_avg = ema(prep_us_avg, prep_us);

        // Optional fps cap (V): with vsync off the GPU would otherwise render
        // thousands of fps the monitor can't show. Sleep the bulk of the
        // remaining frame budget, then spin the last ~2 ms for a precise cap.
        if fps_cap {
            let target = std::time::Duration::from_secs_f64(1.0 / 165.0);
            loop {
                let spent = frame_t0.elapsed();
                if spent >= target { break; }
                let remain = target - spent;
                if remain > std::time::Duration::from_millis(2) {
                    std::thread::sleep(remain - std::time::Duration::from_millis(1));
                }
            }
        }

        frame += 1;
        if let Some(m) = max_frames { if frame >= m { break; } }
        next_frame().await;
    }
}
