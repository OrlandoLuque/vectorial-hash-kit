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
//! *combat* (every critter attacks by its kind — Drifter: random-direction
//! flamer drop; Hunter: drop aimed at the nearest critter; Pulsar:
//! omnidirectional sphere blast — and anyone caught dies & respawns; each
//! attack is an index cull, the "many queries per frame" workload).
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
//! - panel **threads** slider (combat, native, multicore only): sizes the rayon
//!   pool the attack culls fan out over — the many-queries-per-frame crossover
//!   lever, like the siege demo. wasm runs the wave serially.
//!
//! Env: CRITTERS3D_MAX_FRAMES=N exits after N frames (smoke testing) and, when
//! set, prints a one-line STRESS summary (mean fps / frame ms / cpu ms / bound);
//! CRITTERS3D_POP=N initial population, CRITTERS3D_WORLD=N initial world size
//! (both for the instancing stress sweep);
//! CRITTERS3D_RENDER=instanced|billboards|square|none initial render path;
//! CRITTERS3D_STRUCTURE=binary|octree|projection initial index structure;
//! CRITTERS3D_COMBAT=1 starts in combat; CRITTERS3D_SEP=1 starts with separation.

use std::collections::VecDeque;
use vectorial_hash_demos::time::Instant; // wasm-compatible drop-in (native + browser)

use macroquad::camera::Camera;
use macroquad::miniquad::PassAction;
use macroquad::prelude::*;
use macroquad::window::get_internal_gl;
// The combat-wave culls fan out over a live-sized rayon pool (native only; wasm
// has no threads and runs the same wave serially).
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use vectorial_hash::{
    Aabb, MortonGrid3, Octree3, Point, Point3, Positioned, Positioned3, Rect, Shape, Shape3, Sphere3,
    Tree, Tree3,
};
use vectorial_hash_demos::instanced3d::{
    EffectInstance, EffectMesh, Instance, InstancedRenderer, Mode as RenderGeom,
};

const MARGIN: f32 = 4.0;
const ITEM_LIMIT: usize = 16;
// World size is a runtime value now (a stepped pow-2 slider) — see `world` in main.
const SIZE_STEPS: [f32; 5] = [64.0, 128.0, 256.0, 512.0, 1024.0];
const HIST_MAX: usize = 90; // frames kept in the replay history ring

// On-screen control panel (top-right) size.
const UI_W: f32 = 300.0;
const UI_H: f32 = 400.0;

// Combat-mode tuning.
// Per-kind attack cooldown ranges (seconds) — see `kind_cooldown`.
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

/// The two attack shapes a predator can throw — a round sphere (a blast,
/// centred on the predator) or a "drop": a flamethrower **cone** that starts at
/// the predator's edge and fans out along its facing (any 3D direction), like a
/// Warhammer 40k flamer template. Each gets its own colour so they read apart.
#[derive(Clone, Copy, PartialEq)]
enum AttackKind { Sphere, Drop }

/// Flamer-cone dimensions for base radius `r`: (tip offset from the predator
/// centre so it doesn't hit itself, cone length, cone radius at the far end).
fn flamer_dims(r: f32) -> (f32, f32, f32) { (3.0, r * 3.0, r * 0.85) }

/// A flamethrower-cone attack volume (the "drop"): apex `tip`, unit axis `dir`,
/// reaching `length`, widening linearly to `max_r` at the far end. A point is
/// inside if its axial distance is in `[0, length]` and its perpendicular
/// distance is within the cone radius at that depth.
struct DropShape { tip: Point3, dir: [f64; 3], length: f64, max_r: f64 }
impl Shape3 for DropShape {
    fn bounding_box(&self) -> Aabb {
        // Conservative: the tip→far-centre segment, expanded by max_r each way.
        let far = [self.tip.x + self.dir[0] * self.length, self.tip.y + self.dir[1] * self.length, self.tip.z + self.dir[2] * self.length];
        let lo = |a: f64, b: f64| a.min(b) - self.max_r;
        let hi = |a: f64, b: f64| a.max(b) + self.max_r;
        let (x0, y0, z0) = (lo(self.tip.x, far[0]), lo(self.tip.y, far[1]), lo(self.tip.z, far[2]));
        Aabb::new(x0, y0, z0, hi(self.tip.x, far[0]) - x0, hi(self.tip.y, far[1]) - y0, hi(self.tip.z, far[2]) - z0)
    }
    fn contains_point(&self, p: Point3) -> bool {
        let v = [p.x - self.tip.x, p.y - self.tip.y, p.z - self.tip.z];
        let t = v[0] * self.dir[0] + v[1] * self.dir[1] + v[2] * self.dir[2]; // axial distance
        if t < 0.0 { return false; } // behind the apex
        let perp2 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - t * t; // |v|^2 - (v·dir)^2
        if t <= self.length {
            let r_at = (t / self.length) * self.max_r; // cone widens with depth
            perp2 <= r_at * r_at
        } else {
            // rounded cap: a hemisphere of radius max_r at the far centre, so the
            // drop ends like a teardrop instead of a flat disc.
            let dt = t - self.length;
            dt * dt + perp2 <= self.max_r * self.max_r
        }
    }
}

/// A homogeneous attack volume so a whole combat wave — Pulsar spheres +
/// Hunter/Drifter drops mixed — culls in one batch (needed to fan the culls
/// across a thread pool: `cull` wants a single `S: Shape3` per call).
enum AttackVolume { Sphere(Sphere3), Drop(DropShape) }
impl Shape3 for AttackVolume {
    fn bounding_box(&self) -> Aabb {
        match self { AttackVolume::Sphere(s) => s.bounding_box(), AttackVolume::Drop(d) => d.bounding_box() }
    }
    fn contains_point(&self, p: Point3) -> bool {
        match self { AttackVolume::Sphere(s) => s.contains_point(p), AttackVolume::Drop(d) => d.contains_point(p) }
    }
}

/// A live attack volume (predator emits one; prey inside are culled & killed).
/// `dir` is the flamer axis for drops (zero for spheres).
struct Attack { center: Vec3, radius: f32, age: f32, kind: AttackKind, dir: Vec3 }

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
enum Structure { Binary3, Octree, Morton, Projection }
impl Structure {
    fn next(self) -> Self {
        match self {
            Structure::Binary3 => Structure::Octree,
            Structure::Octree => Structure::Morton,
            Structure::Morton => Structure::Projection,
            Structure::Projection => Structure::Binary3,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Structure::Binary3 => "Tree3 (binary-3D)",
            Structure::Octree => "Octree3 (8-way)",
            Structure::Morton => "MortonGrid3 (Z-order)",
            Structure::Projection => "projection (1×2D + z-reject)",
        }
    }
    fn from_env() -> Self {
        match std::env::var("CRITTERS3D_STRUCTURE").ok().as_deref() {
            Some("octree") => Structure::Octree,
            Some("morton") => Structure::Morton,
            Some("projection") => Structure::Projection,
            _ => Structure::Binary3,
        }
    }
}

struct Critter {
    pos: Vec3,
    vel: Vec3,
    kind: u8,       // 0=Drifter, 1=Hunter, 2=Pulsar (colour + combat behaviour)
    cooldown: f32,  // combat: time until this critter's next attack
    flash: f32,     // combat: remaining flash time after respawn
    target: Vec3,   // combat (Hunter): chased prey position; ZERO = none
}

fn world_aabb(world: f32) -> Aabb { Aabb::new(0.0, 0.0, 0.0, world as f64, world as f64, world as f64) }


/// A tiny immediate-mode control panel drawn in screen space: stacked rows of
/// buttons and `[-] bar [+]` sliders with manual mouse hit-testing, plus a help
/// box at the bottom that describes whatever row the mouse is hovering.
/// Which media-player buttons fired this frame (step buttons also report `held`
/// so the caller can drive hold-to-repeat).
#[derive(Default)]
struct PlayerActions {
    rec: bool,
    play: bool,
    back_click: bool,
    back_held: bool,
    fwd_click: bool,
    fwd_held: bool,
}

struct Panel {
    x: f32,
    w: f32,
    cur: f32,    // y of the next row
    row: f32,    // row height
    mx: f32,
    my: f32,
    pressed: bool,  // left mouse pressed this frame (buttons / +/-)
    down: bool,     // left mouse held (slider drag)
    rpressed: bool, // right mouse pressed (start keyboard edit on a slider)
    help: Option<&'static str>,
}
impl Panel {
    fn new(x: f32, y: f32, w: f32, mx: f32, my: f32, pressed: bool, down: bool, rpressed: bool) -> Self {
        draw_rectangle(x, y, w, UI_H, Color::new(0.06, 0.07, 0.10, 0.92));
        draw_rectangle(x, y, w, 19.0, Color::new(0.12, 0.14, 0.20, 0.95));
        draw_text("controls", x + 8.0, y + 14.0, 15.0, Color::new(0.8, 0.9, 1.0, 1.0));
        Panel { x, w, cur: y + 22.0, row: 21.0, mx, my, pressed, down, rpressed, help: None }
    }
    fn over(&self, rx: f32, ry: f32, rw: f32, rh: f32) -> bool {
        self.mx >= rx && self.mx <= rx + rw && self.my >= ry && self.my <= ry + rh
    }
    fn cell(rx: f32, ry: f32, rw: f32, rh: f32, hov: bool, text: &str, tx: f32) {
        draw_rectangle(rx, ry, rw, rh, if hov { Color::new(0.30, 0.36, 0.5, 0.95) } else { Color::new(0.18, 0.21, 0.30, 0.9) });
        draw_text(text, rx + tx, ry + rh * 0.72, 14.0, Color::new(0.92, 0.95, 1.0, 1.0));
    }
    fn button(&mut self, label: &str, help: &'static str) -> bool {
        let (rx, ry, rw, rh) = (self.x + 6.0, self.cur, self.w - 12.0, self.row - 5.0);
        let hov = self.over(rx, ry, rw, rh);
        Self::cell(rx, ry, rw, rh, hov, label, 8.0);
        if hov { self.help = Some(help); }
        self.cur += self.row;
        hov && self.pressed
    }
    /// `[-] label: value bar [+]` — `step` snaps to multiples, drag the bar to
    /// set, **right-click to type the value** (`id` identifies this slider; the
    /// main loop captures the keystrokes into `edit_buf` and applies on Enter).
    fn slider(&mut self, label: &str, min: f32, max: f32, step: f32, val: &mut f32, id: u8, editing: &mut Option<u8>, edit_buf: &mut String, drag: &mut Option<u8>, icon: Option<Color>, help: &'static str) {
        let (ry, rh, bw) = (self.cur, self.row - 4.0, 20.0);
        let (mx0, px0) = (self.x + 6.0, self.x + self.w - 6.0 - bw);
        let bx = mx0 + bw + 4.0;
        let bar_w = px0 - bx - 4.0;
        let (hov_m, hov_p, hov_b) = (
            self.over(mx0, ry, bw, rh),
            self.over(px0, ry, bw, rh),
            self.over(bx, ry, bar_w, rh),
        );
        let is_editing = *editing == Some(id);
        Self::cell(mx0, ry, bw, rh, hov_m && !is_editing, "-", 8.0);
        Self::cell(px0, ry, bw, rh, hov_p && !is_editing, "+", 7.0);
        draw_rectangle(bx, ry, bar_w, rh, Color::new(0.14, 0.16, 0.22, 0.9));
        // Optional kind icon (a coloured dot) + left text offset for it.
        let tx = if let Some(c) = icon {
            draw_circle(bx + 9.0, ry + rh * 0.5, 5.0, c);
            18.0
        } else {
            6.0
        };
        if is_editing {
            draw_text(&format!("{}: {}_", label, edit_buf), bx + tx, ry + rh * 0.72, 13.0, Color::new(1.0, 0.9, 0.4, 1.0));
        } else {
            if hov_m && self.pressed { *val = (((*val / step).ceil() - 1.0) * step).max(min); }
            if hov_p && self.pressed { *val = (((*val / step).floor() + 1.0) * step).min(max); }
            // A press on the bar captures the drag for THIS slider; while held,
            // only the captured slider follows the mouse — so dragging off onto
            // a neighbouring slider no longer hijacks it.
            if hov_b && self.pressed { *drag = Some(id); }
            if *drag == Some(id) && self.down { *val = (min + ((self.mx - bx) / bar_w) * (max - min)).clamp(min, max); }
            if hov_b && self.rpressed { *editing = Some(id); *edit_buf = format!("{:.0}", *val); } // start keyboard edit
            let frac = ((*val - min) / (max - min)).clamp(0.0, 1.0);
            draw_rectangle(bx, ry, bar_w * frac, rh, Color::new(0.28, 0.5, 0.7, 0.9));
            draw_text(&format!("{}: {:.0}", label, *val), bx + tx, ry + rh * 0.72, 13.0, Color::new(0.92, 0.95, 1.0, 1.0));
        }
        if hov_m || hov_p || hov_b { self.help = Some(help); }
        self.cur += self.row;
    }
    /// Like `slider`, but the value steps through a discrete list (e.g. powers
    /// of 2) — `[-]`/`[+]` move one entry, the bar snaps to the nearest.
    fn stepper(&mut self, label: &str, values: &[f32], cur: &mut f32, id: u8, drag: &mut Option<u8>, help: &'static str) {
        let (ry, rh, bw) = (self.cur, self.row - 4.0, 20.0);
        let (mx0, px0) = (self.x + 6.0, self.x + self.w - 6.0 - bw);
        let bx = mx0 + bw + 4.0;
        let bar_w = px0 - bx - 4.0;
        let mut idx = 0;
        let mut best = f32::INFINITY;
        for (i, &v) in values.iter().enumerate() {
            let d = (v - *cur).abs();
            if d < best { best = d; idx = i; }
        }
        let (hov_m, hov_p, hov_b) = (
            self.over(mx0, ry, bw, rh),
            self.over(px0, ry, bw, rh),
            self.over(bx, ry, bar_w, rh),
        );
        Self::cell(mx0, ry, bw, rh, hov_m, "-", 8.0);
        Self::cell(px0, ry, bw, rh, hov_p, "+", 7.0);
        if hov_m && self.pressed && idx > 0 { idx -= 1; }
        if hov_p && self.pressed && idx + 1 < values.len() { idx += 1; }
        if hov_b && self.pressed { *drag = Some(id); }
        if *drag == Some(id) && self.down {
            let frac = ((self.mx - bx) / bar_w).clamp(0.0, 1.0);
            idx = (frac * (values.len() - 1) as f32).round() as usize;
        }
        *cur = values[idx.min(values.len() - 1)];
        draw_rectangle(bx, ry, bar_w, rh, Color::new(0.14, 0.16, 0.22, 0.9));
        let frac = idx as f32 / (values.len() - 1).max(1) as f32;
        draw_rectangle(bx, ry, bar_w * frac, rh, Color::new(0.28, 0.5, 0.7, 0.9));
        draw_text(&format!("{}: {:.0}", label, *cur), bx + 6.0, ry + rh * 0.72, 13.0, Color::new(0.92, 0.95, 1.0, 1.0));
        if hov_m || hov_p || hov_b { self.help = Some(help); }
        self.cur += self.row;
    }
    /// A media-player row: REC ● | step-back ◀ | play/pause | step-forward ▶.
    /// Icons drawn with primitives; returns each button's click + (for the step
    /// buttons) held state so the caller can add hold-to-repeat.
    fn player(&mut self, rec_on: bool, live_paused: bool) -> PlayerActions {
        let (ry, rh) = (self.cur, self.row - 5.0);
        let gap = 4.0;
        let bw = (self.w - 12.0 - gap * 3.0) / 4.0;
        let icon = Color::new(0.92, 0.95, 1.0, 1.0);
        let mut out = PlayerActions::default();
        for slot in 0..4 {
            let rx = self.x + 6.0 + (bw + gap) * slot as f32;
            let hov = self.over(rx, ry, bw, rh);
            draw_rectangle(rx, ry, bw, rh, if hov { Color::new(0.30, 0.36, 0.5, 0.95) } else { Color::new(0.18, 0.21, 0.30, 0.9) });
            let (cx, cy) = (rx + bw * 0.5, ry + rh * 0.5);
            let help = match slot {
                0 => { // REC
                    draw_circle(cx, cy, 6.0, if rec_on { RED } else { Color::new(0.5, 0.25, 0.25, 1.0) });
                    "REC [K]: record frames into the history\nring (so you can step back)."
                }
                1 => { // step back
                    draw_rectangle(cx - 9.0, cy - 7.0, 3.0, 14.0, icon);
                    draw_triangle(vec2(cx - 4.0, cy), vec2(cx + 6.0, cy - 7.0), vec2(cx + 6.0, cy + 7.0), icon);
                    "Step back [Left] -- a frame back through the\nrecorded history (hold to rewind)."
                }
                2 => { // play / pause
                    if live_paused {
                        draw_triangle(vec2(cx - 5.0, cy - 7.0), vec2(cx - 5.0, cy + 7.0), vec2(cx + 7.0, cy), icon);
                    } else {
                        draw_rectangle(cx - 6.0, cy - 7.0, 4.0, 14.0, icon);
                        draw_rectangle(cx + 2.0, cy - 7.0, 4.0, 14.0, icon);
                    }
                    "Play / pause [Space]. From the history it\njumps back to the live view."
                }
                _ => { // step forward
                    draw_triangle(vec2(cx - 6.0, cy - 7.0), vec2(cx - 6.0, cy + 7.0), vec2(cx + 4.0, cy), icon);
                    draw_rectangle(cx + 6.0, cy - 7.0, 3.0, 14.0, icon);
                    "Step forward [Right] -- a frame on; at the\nnewest it generates a new one (hold to play)."
                }
            };
            if hov { self.help = Some(help); }
            let clicked = hov && self.pressed;
            let held = hov && self.down;
            match slot {
                0 => out.rec = clicked,
                1 => { out.back_click = clicked; out.back_held = held; }
                2 => out.play = clicked,
                _ => { out.fwd_click = clicked; out.fwd_held = held; }
            }
        }
        self.cur += self.row;
        out
    }
    fn separator(&mut self) {
        draw_rectangle(self.x + 6.0, self.cur + 2.0, self.w - 12.0, 1.0, Color::new(0.3, 0.34, 0.42, 0.8));
        self.cur += 8.0;
    }
    /// Draw the help box filling the rest of the panel with the hovered help.
    fn finish(&self, panel_y: f32) {
        let hy = self.cur + 4.0;
        let hh = panel_y + UI_H - hy - 4.0;
        if hh < 12.0 { return; }
        draw_rectangle(self.x + 4.0, hy, self.w - 8.0, hh, Color::new(0.08, 0.09, 0.13, 0.95));
        match self.help {
            Some(h) => {
                for (i, line) in h.split('\n').enumerate() {
                    draw_text(line, self.x + 10.0, hy + 16.0 + i as f32 * 14.0, 13.0, Color::new(0.82, 0.87, 0.97, 1.0));
                }
            }
            None => {
                draw_text("hover a control for help", self.x + 10.0, hy + 16.0, 13.0, Color::new(0.5, 0.55, 0.65, 1.0));
            }
        }
    }
}

/// Push a sample into a rolling history (newest at the end), capped at `GRAPH_N`.
const GRAPH_N: usize = 240;
fn push_hist(v: &mut Vec<f32>, x: f32) {
    v.push(x);
    if v.len() > GRAPH_N { v.remove(0); }
}

/// Draw a time-series line graph (screen space): auto-scaled to the data's max,
/// newest on the right, with a label + current/peak values.
fn draw_graph(x: f32, y: f32, w: f32, h: f32, label: &str, data: &[f32], color: Color) {
    draw_rectangle(x, y, w, h, Color::new(0.06, 0.07, 0.10, 0.92));
    let maxv = data.iter().cloned().fold(1e-6, f32::max);
    if data.len() >= 2 {
        let plot_h = h - 16.0;
        let n = data.len();
        for i in 1..n {
            let x0 = x + (i - 1) as f32 / (n - 1) as f32 * w;
            let x1 = x + i as f32 / (n - 1) as f32 * w;
            let y0 = y + h - (data[i - 1] / maxv) * plot_h;
            let y1 = y + h - (data[i] / maxv) * plot_h;
            draw_line(x0, y0, x1, y1, 1.5, color);
        }
        draw_text(&format!("{}: {:.0}  (peak {:.0})", label, data[n - 1], maxv), x + 5.0, y + 12.0, 13.0, color);
    } else {
        draw_text(label, x + 5.0, y + 12.0, 13.0, color);
    }
}

/// One frame's drawable state — recorded into the history ring so a past frame
/// can be redrawn without re-simulating. Holds exactly the render inputs:
/// the critter instances, the effect instances, and the index's leaf boxes.
#[derive(Clone, Default)]
struct Frame {
    instances: Vec<Instance>,
    sphere_fx: Vec<EffectInstance>,
    drop_fx: Vec<EffectInstance>,
    boxes: Vec<(Vec3, Vec3)>,
}

/// Draw a `Frame`'s world: the cube, the leaf boxes (if shown), the instanced
/// critters, and the instanced effects. Shared by the live path and replay.
fn draw_world_visuals(
    renderer: &mut InstancedRenderer,
    geom: Option<RenderGeom>,
    frame: &Frame,
    mvp: Mat4,
    cam_right: Vec3,
    cam_up: Vec3,
    show_boxes: bool,
    world: f32,
    observer: Vec3,
) {
    draw_cube_wires(observer, vec3(world, world, world), Color::new(0.3, 0.35, 0.45, 1.0));
    if show_boxes {
        for (c, sz) in &frame.boxes {
            draw_cube_wires(*c, *sz, Color::new(0.18, 0.22, 0.3, 1.0));
        }
    }
    if let Some(geom) = geom {
        let mut gl = unsafe { get_internal_gl() };
        gl.flush();
        let ctx = gl.quad_context;
        ctx.begin_default_pass(PassAction::Nothing);
        renderer.draw(ctx, geom, &frame.instances, mvp, cam_right, cam_up);
        ctx.end_render_pass();
    }
    if !frame.sphere_fx.is_empty() || !frame.drop_fx.is_empty() {
        let mut gl = unsafe { get_internal_gl() };
        gl.flush();
        let ctx = gl.quad_context;
        ctx.begin_default_pass(PassAction::Nothing);
        renderer.draw_effects(ctx, EffectMesh::Sphere, &frame.sphere_fx, mvp);
        renderer.draw_effects(ctx, EffectMesh::Drop, &frame.drop_fx, mvp);
        ctx.end_render_pass();
    }
}

/// A 3D AABB → (centre, size) pair for `draw_cube_wires`.
fn aabb_box(b: &Aabb) -> (Vec3, Vec3) {
    (
        vec3((b.x + b.w * 0.5) as f32, (b.y + b.h * 0.5) as f32, (b.z + b.d * 0.5) as f32),
        vec3(b.w as f32, b.h as f32, b.d as f32),
    )
}

fn pt3(c: &Critter) -> Point3 { Point3::new(c.pos.x as f64, c.pos.y as f64, c.pos.z as f64) }

/// The single **active** index — the structure the `M` toggle selects now
/// resolves *everything* (observe vision, combat attacks, persistence), and the
/// others don't exist. Rebuilt on a structure / world / population change;
/// otherwise relocated in place each frame: the binary tree via the **stable
/// `ItemRef`** (O(1), no predicate scan), the octree via `update`, the
/// pointer-free Morton grid and the 2D-projection tree by a flat rebuild (their
/// "update" is a re-bucket). All exact.
enum Index {
    Binary { tree: Tree3<C3>, refs: Vec<vectorial_hash::ItemRef> },
    Octree { tree: Octree3<C3>, refs: Vec<vectorial_hash::ItemRef> },
    Morton { grid: MortonGrid3<C3> },
    Projection { tree: Tree<P2>, refs: Vec<vectorial_hash::ItemRef> },
}

impl Index {
    fn build(s: Structure, world: f32, cell: f32, critters: &[Critter]) -> Index {
        let wa = world_aabb(world);
        match s {
            Structure::Binary3 => {
                let mut tree = Tree3::<C3>::new(wa, ITEM_LIMIT);
                let refs = critters.iter().enumerate()
                    .map(|(i, c)| tree.insert_ref(C3 { id: i as u32, p: pt3(c) }).unwrap())
                    .collect();
                Index::Binary { tree, refs }
            }
            Structure::Octree => {
                let mut tree = Octree3::<C3>::new(wa, ITEM_LIMIT);
                let refs = critters.iter().enumerate()
                    .map(|(i, c)| tree.insert_ref(C3 { id: i as u32, p: pt3(c) }).unwrap())
                    .collect();
                Index::Octree { tree, refs }
            }
            Structure::Morton => {
                let levels = MortonGrid3::<C3>::levels_for_cell_size(wa, (cell as f64).max(4.0));
                let mut grid = MortonGrid3::<C3>::new(wa, levels);
                for (i, c) in critters.iter().enumerate() { grid.insert(C3 { id: i as u32, p: pt3(c) }); }
                Index::Morton { grid }
            }
            Structure::Projection => {
                let mut tree = Tree::<P2>::new(Rect::new(0.0, 0.0, world as f64, world as f64), ITEM_LIMIT);
                let refs = critters.iter().enumerate()
                    .map(|(i, c)| tree.insert_ref(P2 { id: i as u32, p: Point::new(c.pos.x as f64, c.pos.y as f64), z: c.pos.z as f64 }).unwrap())
                    .collect();
                Index::Projection { tree, refs }
            }
        }
    }

    fn structure(&self) -> Structure {
        match self {
            Index::Binary { .. } => Structure::Binary3,
            Index::Octree { .. } => Structure::Octree,
            Index::Morton { .. } => Structure::Morton,
            Index::Projection { .. } => Structure::Projection,
        }
    }

    /// Relocate every critter in place (persistence). Binary/octree update;
    /// Morton/projection re-bucket (a fresh build — their cheap "update").
    fn sync(&mut self, world: f32, cell: f32, critters: &[Critter]) {
        match self {
            Index::Binary { tree, refs } => {
                for (i, c) in critters.iter().enumerate() { let np = pt3(c); tree.update_ref(refs[i], |x| x.p = np); }
            }
            Index::Octree { tree, refs } => {
                for (i, c) in critters.iter().enumerate() { let np = pt3(c); tree.update_ref(refs[i], |x| x.p = np); }
            }
            Index::Morton { .. } => *self = Index::build(Structure::Morton, world, cell, critters),
            Index::Projection { tree, refs } => {
                for (i, c) in critters.iter().enumerate() {
                    tree.update_ref(refs[i], |p| { p.p = Point::new(c.pos.x as f64, c.pos.y as f64); p.z = c.pos.z as f64; });
                }
            }
        }
    }

    /// Cull any 3D query shape → the hit ids. The projection variant turns the
    /// shape's xy-shadow into a disc broadphase, then exact-filters in 3D.
    fn cull<S: Shape3>(&self, shape: &S) -> Vec<u32> {
        match self {
            Index::Binary { tree, .. } => tree.cull(shape).iter().map(|c| c.id).collect(),
            Index::Octree { tree, .. } => tree.cull(shape).iter().map(|c| c.id).collect(),
            Index::Morton { grid } => grid.cull(shape).iter().map(|c| c.id).collect(),
            Index::Projection { tree, .. } => {
                let bb = shape.bounding_box();
                let (cx, cy) = (bb.x + bb.w * 0.5, bb.y + bb.h * 0.5);
                let r = (bb.w * bb.w + bb.h * bb.h).sqrt() * 0.5; // disc covering the xy bbox
                tree.cull(&Disc { cx, cy, r }).iter()
                    .filter(|p2| shape.contains_point(Point3::new(p2.p.x, p2.p.y, p2.z)))
                    .map(|p2| p2.id).collect()
            }
        }
    }

    fn visit_boxes(&self, world: f32, boxes: &mut Vec<(Vec3, Vec3)>) {
        match self {
            Index::Binary { tree, .. } => tree.visit_leaves(|l| boxes.push(aabb_box(&l.bbox))),
            Index::Octree { tree, .. } => tree.visit_leaves(|l| boxes.push(aabb_box(&l.bbox))),
            Index::Morton { grid } => grid.visit_cells(|b, _| boxes.push(aabb_box(b))),
            Index::Projection { tree, .. } => tree.visit_leaves(|_, l| {
                let b = l.bbox;
                let c = vec3((b.x + b.width * 0.5) as f32, (b.y + b.height * 0.5) as f32, world * 0.5);
                boxes.push((c, vec3(b.width as f32, b.height as f32, world)));
            }),
        }
    }

    fn stat(&self) -> String {
        match self {
            Index::Binary { tree, .. } => format!("Tree3 (binary, ItemRef): {:>6} leaves, {:>6} arena", tree.leaf_count(), tree.node_count()),
            Index::Octree { tree, .. } => format!("Octree3 (8-way, ItemRef): {:>6} leaves, {:>6} arena", tree.leaf_count(), tree.node_count()),
            Index::Morton { grid } => format!("MortonGrid3 (Z-order, rebuilt): {:>6} cells, {:.2} items/cell", grid.cell_count(), grid.item_count() as f64 / grid.cell_count().max(1) as f64),
            Index::Projection { tree, .. } => format!("projection (2D xy ItemRef + z-reject): {:>6} leaves, {:>6} arena", tree.leaf_count(), tree.node_count()),
        }
    }
}

fn kind_color(k: u8, lit: bool) -> Color {
    // The observe "seen" highlight is white (not yellow) so it doesn't clash
    // with the GOLD Pulsars. Kind colours match the 2D demo.
    if lit { return WHITE; }
    match k {
        0 => SKYBLUE, // Drifter
        1 => RED,     // Hunter
        _ => GOLD,    // Pulsar
    }
}

/// Per-kind combat behaviour, mirroring the 2D demo's three kinds:
/// 0 = Drifter (random-direction drop), 1 = Hunter (drop aimed at the nearest
/// other critter), 2 = Pulsar (omnidirectional sphere blast). Cooldown ranges
/// match the 2D values; all kinds attack and any of them can be killed.
fn kind_cooldown(kind: u8) -> (f32, f32) {
    match kind {
        1 => (1.8, 3.5), // Hunter — aims, so the fastest
        2 => (3.0, 5.0), // Pulsar — omnidirectional
        _ => (2.5, 4.5), // Drifter — random drop
    }
}

/// A uniformly random unit vector on the sphere.
fn rand_dir(rng: &mut Rng) -> Vec3 {
    let a = rng.range(0.0, std::f32::consts::TAU);
    let z = rng.range(-1.0, 1.0);
    let s = (1.0 - z * z).max(0.0).sqrt();
    vec3(s * a.cos(), s * a.sin(), z)
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
    let spawn = |rng: &mut Rng, kind: u8, world: f32| -> Critter {
        let p = vec3(
            rng.range(MARGIN, world - MARGIN),
            rng.range(MARGIN, world - MARGIN),
            rng.range(MARGIN, world - MARGIN),
        );
        let speed = rng.range(20.0, 55.0);
        // random unit direction
        let a = rng.range(0.0, std::f32::consts::TAU);
        let z = rng.range(-1.0, 1.0);
        let s = (1.0 - z * z).max(0.0).sqrt();
        let v = vec3(s * a.cos(), s * a.sin(), z) * speed;
        let (_cmin, cmax) = kind_cooldown(kind);
        let cooldown = rng.range(0.0, cmax); // spread initial cooldowns so they desync
        Critter { pos: p, vel: v, kind, cooldown, flash: 0.0, target: Vec3::ZERO }
    };
    // World size (a stepped pow-2 slider mutates it; the tree is rebuilt on
    // change). `CRITTERS3D_WORLD` sets the initial size (handy with large pops
    // so the index stays shallow during the stress sweep).
    let mut world: f32 = std::env::var("CRITTERS3D_WORLD").ok().and_then(|s| s.parse().ok()).unwrap_or(256.0);
    let mut prev_world = world;
    // Initial population (total, split across the 3 kinds). `CRITTERS3D_POP`
    // overrides it for the instancing stress sweep.
    let init_pop: usize = std::env::var("CRITTERS3D_POP").ok().and_then(|s| s.parse().ok()).unwrap_or(2400);
    // Per-kind population ceiling (sliders + reconcile). 30k each by default;
    // raised to fit CRITTERS3D_POP so the stress sweep isn't clamped back down.
    let pop_cap: f32 = (init_pop as f32 / 3.0).max(30000.0);
    for i in 0..init_pop {
        critters.push(spawn(&mut rng, (i % 3) as u8, world));
    }

    // The single active index is built below, after the structure/vision/mode
    // state it depends on (see `index`/`idx_world`/`idx_pop`).

    // Camera orbit state.
    let mut yaw: f32 = 0.7;
    let mut pitch: f32 = 0.5;
    let mut dist: f32 = world * 2.2;
    let mut last_mouse = mouse_position();

    let mut vision_r: f32 = 40.0;
    // Start paused (sim frozen) when CRITTERS3D_FREEZE is set: render-only, so
    // the stress sweep measures pure GPU throughput with the per-frame index
    // update removed (the CPU bottleneck that otherwise dominates).
    let mut paused = std::env::var("CRITTERS3D_FREEZE").is_ok();
    let mut show_boxes = false;
    // Visual-only mode: hide all 2D overlay UI (HUD text, control panel, graphs)
    // leaving just the rendered scene. Toggled by [U] or the web "visual only"
    // button. Keyboard shortcuts still work while hidden.
    let mut ui_hidden = false;
    let mut structure = Structure::from_env();
    let mut render_mode = RenderMode::from_env();
    let cull_rep_steps = [1usize, 50, 200, 1000];
    let mut cull_rep_idx = 0usize;

    // Combat-mode state.
    let mut sim_mode = if std::env::var("CRITTERS3D_COMBAT").is_ok() { SimMode::Combat } else { SimMode::Observe };
    let mut random_attacks = true; // desync cooldowns; off = synchronized saturation
    let mut prev_random = random_attacks; // to reseed cooldowns when this flips
    let mut separation = std::env::var("CRITTERS3D_SEP").is_ok(); // push critters apart (heavy)
    let mut fps_cap = false; // V: cap to the monitor refresh so the GPU isn't pegged
    // Target population per kind (Drifter / Hunter / Pulsar), driven by +/- and
    // the per-kind panel sliders. The reconcile matches the total and rebalances
    // kinds to these counts.
    let mut pop_kind: [f32; 3] = { let n = (critters.len() / 3) as f32; [n, n, n] };
    let mut attack_r: f32 = 22.0;

    // Live thread-count control (native): the combat wave is the "many queries
    // per frame" workload (one attack cull per firing critter), so its culls
    // fan out over a rayon pool sized by the panel slider — the same crossover
    // lever the siege demo exposes. wasm has no threads → the wave runs serially.
    #[cfg(not(target_arch = "wasm32"))]
    let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    #[cfg(not(target_arch = "wasm32"))]
    let mut cur_threads = max_threads;
    #[cfg(not(target_arch = "wasm32"))]
    let mut thread_pool = rayon::ThreadPoolBuilder::new().num_threads(cur_threads).build().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let mut n_threads_f = max_threads as f32; // f32 backing for the panel slider

    // The single ACTIVE index: the structure `M` selects resolves everything
    // (observe vision, combat attacks, persistence). Rebuilt when the structure,
    // world size, or population changes; relocated in place otherwise. The
    // binary tree uses the stable `ItemRef` (O(1) update_ref).
    let idx_cell = |sim_mode: SimMode, vision_r: f32, attack_r: f32| if sim_mode == SimMode::Combat { attack_r } else { vision_r };
    let mut index = Index::build(structure, world, idx_cell(sim_mode, vision_r, attack_r), &critters);
    let mut idx_world = world;
    let mut idx_pop = critters.len();

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
    let mut back_rep = KeyRepeat::new(); // history step back (◀ / Left)
    let mut fwd_rep = KeyRepeat::new(); // history step forward (▶ / Right)
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
    // Rolling histories for the time-series graphs below the panel.
    let mut g_fps: Vec<f32> = Vec::new();
    let mut g_cpu: Vec<f32> = Vec::new();
    let mut g_cull: Vec<f32> = Vec::new();
    // Frame history / replay: ring of drawable frames, REC toggle, scrub offset
    // (0 = live, N = N frames back), and a one-shot manual step request.
    let mut rec = false;
    let mut hist: VecDeque<Frame> = VecDeque::new();
    let mut scrub: usize = 0;
    let mut cur_frame: Frame; // (re)built every frame before it's drawn
    let mut step_request = false;
    // Keyboard editing of a slider value: which slider (0=pop, 1=radius) + buffer.
    let mut editing: Option<u8> = None;
    let mut edit_buf = String::new();
    // Which slider/stepper currently owns the mouse drag (so dragging off one
    // onto another doesn't hijack the second). Cleared when the button is up.
    let mut slider_drag: Option<u8> = None;

    // Stress-sweep accumulators (after a short warmup): mean real fps + CPU ms
    // over the run, printed at exit when CRITTERS3D_MAX_FRAMES is set.
    let mut stress_secs = 0.0f64;
    let mut stress_cpu_ms = 0.0f64;
    let mut stress_n = 0u64;

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
        // Typing a value into a slider (started with a right-click in the panel)
        // grabs the keyboard; the shortcuts below are suspended until Enter/Esc.
        if editing.is_some() {
            while let Some(ch) = get_char_pressed() {
                if ch.is_ascii_digit() || ch == '.' { edit_buf.push(ch); }
            }
            if is_key_pressed(KeyCode::Backspace) { edit_buf.pop(); }
            if is_key_pressed(KeyCode::Escape) { editing = None; }
            if is_key_pressed(KeyCode::Enter) || is_key_pressed(KeyCode::KpEnter) {
                if let Ok(v) = edit_buf.parse::<f32>() {
                    match editing {
                        Some(k @ 0..=2) => pop_kind[k as usize] = v.clamp(0.0, pop_cap),
                        Some(3) => match sim_mode {
                            SimMode::Observe => vision_r = v.clamp(6.0, world),
                            SimMode::Combat => attack_r = v.clamp(4.0, world * 0.5),
                        },
                        _ => {}
                    }
                }
                editing = None;
            }
        }
        if editing.is_none() {
        if is_key_pressed(KeyCode::Escape) { break; }
        if is_key_pressed(KeyCode::B) { show_boxes = !show_boxes; }
        if is_key_pressed(KeyCode::U) { ui_hidden = !ui_hidden; }
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
        if add { for k in 0..3 { pop_kind[k] = (((pop_kind[k] / 100.0).floor() + 1.0) * 100.0).min(pop_cap); } }
        let sub = sub_rep.fires(
            is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract),
            is_key_down(KeyCode::Minus) || is_key_down(KeyCode::KpSubtract),
            dt,
        );
        if sub { for k in 0..3 { pop_kind[k] = (((pop_kind[k] / 100.0).ceil() - 1.0) * 100.0).max(0.0); } }
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
                SimMode::Observe => vision_r = (((vision_r / R_STEP).floor() + 1.0) * R_STEP).min(world),
                SimMode::Combat => attack_r = (((attack_r / R_STEP).floor() + 1.0) * R_STEP).min(world * 0.5),
            }
        }
        } // end keyboard shortcuts (suspended while editing a value)
        let mp = mouse_position();
        if is_mouse_button_down(MouseButton::Left) && !over_ui {
            yaw += (mp.0 - last_mouse.0) * 0.01;
            pitch = (pitch + (mp.1 - last_mouse.1) * 0.01).clamp(-1.4, 1.4);
        }
        last_mouse = mp;
        let scroll = mouse_wheel().1;
        if scroll != 0.0 && !over_ui { dist = (dist * (1.0 - scroll.signum() * 0.1)).clamp(world * 0.6, world * 6.0); }

        // Does the sim advance this frame? Not while scrubbing the history, and
        // when live only if running (or a single manual step was requested).
        let step_now = step_request;
        step_request = false;
        let advance = scrub == 0 && (!paused || step_now);

        // Reconcile population to the per-kind targets — only while the sim
        // advances (frozen during pause / replay). Match the TOTAL by
        // spawning/removing at the end (keeps id == index), then rebalance kinds
        // by reassigning surplus kinds to deficient ones (no index churn).
        let want = [
            pop_kind[0].round().max(0.0) as usize,
            pop_kind[1].round().max(0.0) as usize,
            pop_kind[2].round().max(0.0) as usize,
        ];
        let total: usize = want.iter().sum();
        if advance {
            // Spawn/remove only touch `critters`; the index rebuilds below when
            // it notices the count changed (so reconcile is structure-agnostic).
            while critters.len() < total {
                let id = critters.len() as u32;
                critters.push(spawn(&mut rng, (id % 3) as u8, world));
            }
            while critters.len() > total {
                critters.pop();
            }
            let mut have = [0usize; 3];
            for c in &critters { have[c.kind as usize] += 1; }
            for c in critters.iter_mut() {
                let k = c.kind as usize;
                if have[k] > want[k] {
                    if let Some(u) = (0..3).find(|&u| have[u] < want[u]) {
                        have[k] -= 1;
                        have[u] += 1;
                        c.kind = u as u8;
                    }
                }
            }
        }

        // --- simulate ---
        let t_sim = Instant::now();
        if advance {
            // Combat steers each kind like the 2D demo: Hunters chase the
            // nearest critter in vision, Pulsars circle, Drifters wander. First
            // find the Hunters' targets (a vision cull each, from last frame's
            // index), then steer + move.
            let hunter_vision = world * 0.3;
            let mut targets: Vec<Option<Vec3>> = vec![None; critters.len()];
            if sim_mode == SimMode::Combat {
                for i in 0..critters.len() {
                    if critters[i].kind != 1 { continue; }
                    let p = critters[i].pos;
                    let s = Sphere3::new(p.x as f64, p.y as f64, p.z as f64, hunter_vision as f64);
                    let mut best_d2 = f32::INFINITY;
                    for jid in index.cull(&s) {
                        let j = jid as usize;
                        // Same stale-index guard as separation below: a pop shrink
                        // this frame can leave ids past the new end until the rebuild.
                        if j == i || j >= critters.len() { continue; }
                        let d2 = (critters[j].pos - p).length_squared();
                        if d2 < best_d2 { best_d2 = d2; targets[i] = Some(critters[j].pos); }
                    }
                }
            }
            for i in 0..critters.len() {
                if sim_mode == SimMode::Combat {
                    let speed = critters[i].vel.length().max(15.0);
                    match critters[i].kind {
                        1 => {
                            // Hunter: remember the target (for accurate aiming)
                            // and steer toward it — but only when it's not right
                            // on top of us, or the velocity would shrink to zero.
                            critters[i].target = targets[i].unwrap_or(Vec3::ZERO);
                            if let Some(t) = targets[i] {
                                let to = t - critters[i].pos;
                                if to.length() > 1.0 {
                                    let desired = to.normalize() * speed;
                                    critters[i].vel = critters[i].vel.lerp(desired, (3.0 * dt).min(1.0));
                                }
                            }
                        }
                        2 => {
                            let v = critters[i].vel;
                            let a = 1.6 * dt;
                            critters[i].vel = vec3(v.x * a.cos() - v.z * a.sin(), v.y, v.x * a.sin() + v.z * a.cos());
                        }
                        _ => {
                            let w = rand_dir(&mut rng) * (speed * 0.5 * dt);
                            critters[i].vel = (critters[i].vel + w).normalize_or_zero() * speed;
                        }
                    }
                    // Never let a critter freeze (e.g. a hunter that reached its
                    // prey, or one left stationary when switching modes).
                    let s = critters[i].vel.length();
                    if s < 8.0 {
                        critters[i].vel = if s > 0.01 { critters[i].vel / s * 12.0 } else { rand_dir(&mut rng) * 12.0 };
                    }
                }
                let c = &mut critters[i];
                let mut np = c.pos + c.vel * dt;
                for axis in 0..3 {
                    if np[axis] < MARGIN { np[axis] = MARGIN; c.vel[axis] = -c.vel[axis]; }
                    if np[axis] > world - MARGIN { np[axis] = world - MARGIN; c.vel[axis] = -c.vel[axis]; }
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
                    for jid in index.cull(&s) {
                        let j = jid as usize;
                        // A population shrink pops critters (from the end; id == index)
                        // BEFORE the index rebuilds further down, so this last-frame
                        // index can still return ids past the new end — skip those.
                        if j == i || j >= critters.len() { continue; }
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
                    for axis in 0..3 { np[axis] = np[axis].clamp(MARGIN, world - MARGIN); }
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

        // Maintain the active index. Rebuild it when the structure / world /
        // population changed (so the chosen structure resolves everything);
        // otherwise relocate every critter in place (binary via O(1) ItemRef,
        // octree via update, Morton/projection re-bucket). `sync_us` is what the
        // HUD reports as "build".
        let cell = idx_cell(sim_mode, vision_r, attack_r);
        let t_sync = Instant::now();
        if index.structure() != structure || (idx_world - world).abs() > 0.5 || idx_pop != critters.len() {
            index = Index::build(structure, world, cell, &critters);
            idx_world = world;
            idx_pop = critters.len();
        } else if advance {
            index.sync(world, cell, &critters);
        }
        let sync_us = t_sync.elapsed().as_secs_f64() * 1e6;

        // --- (re)build the chosen index and run a vision cull from the centre ---
        let observer = vec3(world * 0.5, world * 0.5, world * 0.5);
        let (ox, oy, oz, r) = (observer.x as f64, observer.y as f64, observer.z as f64, vision_r as f64);
        let cull_reps = cull_rep_steps[cull_rep_idx];
        // Optional self-check: the active index's vision cull must equal a
        // brute-force scan (catches any update/rebuild bookkeeping bug).
        if std::env::var("CRITTERS3D_VERIFY").is_ok() {
            let s = Sphere3::new(ox, oy, oz, r);
            let mut got = index.cull(&s);
            let mut want: Vec<u32> = (0..critters.len() as u32).filter(|&i| {
                let c = &critters[i as usize];
                let (dx, dy, dz) = (c.pos.x as f64 - ox, c.pos.y as f64 - oy, c.pos.z as f64 - oz);
                dx * dx + dy * dy + dz * dz <= r * r
            }).collect();
            got.sort_unstable();
            want.sort_unstable();
            if got != want {
                eprintln!("VERIFY mismatch frame {frame}: {} {} vs brute {} (pop {})", structure.label(), got.len(), want.len(), critters.len());
            }
        }
        let mut lit = vec![false; critters.len()];
        let mut boxes: Vec<(Vec3, Vec3)> = Vec::new();
        let stat_line: String;
        let cand_n: Option<usize>; // (unused now; kept for the HUD line)
        let t_build_us: f64;
        let t_cull_us: f64;
        let mut frame_attacks = 0usize; // combat: predators that attacked this frame

        // Resize the combat cull pool if the thread slider moved (native only).
        #[cfg(not(target_arch = "wasm32"))]
        {
            let want = (n_threads_f.round() as usize).clamp(1, max_threads);
            if want != cur_threads {
                thread_pool = rayon::ThreadPoolBuilder::new().num_threads(want).build().unwrap();
                cur_threads = want;
            }
        }

        if sim_mode == SimMode::Observe {
            // The active index (whatever structure M selected) resolves the
            // vision cull. Maintenance cost was paid above as `sync_us`.
            t_build_us = sync_us;
            let tc = Instant::now();
            for rep in 0..cull_reps {
                let sphere = Sphere3::new(ox + rep as f64 * 0.01, oy, oz, r);
                let hits = index.cull(&sphere);
                if rep == 0 { for id in hits { lit[id as usize] = true; } }
            }
            t_cull_us = tc.elapsed().as_secs_f64() * 1e6 / cull_reps as f64;
            if rec || show_boxes { index.visit_boxes(world, &mut boxes); }
            cand_n = None;
            stat_line = index.stat();
        } else {
            // === COMBAT: predators (kind 1) attack nearby prey; each attack is
            // an index cull against the persistent tree (synced above), so this
            // is the "many queries per frame" workload — no rebuild.
            t_build_us = sync_us;
            let mut killed = vec![false; critters.len()];
            let mut cull_us = 0.0f64;
            let t_wave = Instant::now();
            if advance {
                // 1) Decide pass (serial — mutates cooldowns, draws from `rng`):
                //    who fires this frame, and each firer's attack volume + the
                //    on-screen effect marker.
                let mut plan: Vec<(usize, AttackVolume)> = Vec::new();
                for i in 0..critters.len() {
                    if critters[i].cooldown > 0.0 { continue; } // every kind attacks
                    let kind = critters[i].kind;
                    let center = critters[i].pos;
                    let (cmin, cmax) = kind_cooldown(kind);
                    critters[i].cooldown = if random_attacks { rng.range(cmin, cmax) } else { cmin };
                    let (cx, cy, cz) = (center.x as f64, center.y as f64, center.z as f64);
                    frame_attacks += 1;
                    let (akind, adir, vol): (AttackKind, Vec3, AttackVolume) = match kind {
                        2 => {
                            // Pulsar: omnidirectional sphere blast (the 2D circle).
                            (AttackKind::Sphere, Vec3::ZERO, AttackVolume::Sphere(Sphere3::new(cx, cy, cz, attack_r as f64)))
                        }
                        _ => {
                            // Hunter (1): flamer drop along its heading — it has
                            // already steered toward its target this frame, so
                            // forward = at the prey. Drifter (0): random.
                            let (off, length, maxr) = flamer_dims(attack_r);
                            let dir = if kind == 1 {
                                // Aim straight at the stored target (accurate even
                                // when adjacent, unlike aiming along the velocity).
                                let d = critters[i].target - center;
                                if critters[i].target != Vec3::ZERO && d.length() > 1e-3 {
                                    d.normalize()
                                } else {
                                    let v = critters[i].vel;
                                    if v.length() > 0.1 { v.normalize() } else { rand_dir(&mut rng) }
                                }
                            } else {
                                rand_dir(&mut rng)
                            };
                            let tip = center + dir * off;
                            (AttackKind::Drop, dir, AttackVolume::Drop(DropShape {
                                tip: Point3::new(tip.x as f64, tip.y as f64, tip.z as f64),
                                dir: [dir.x as f64, dir.y as f64, dir.z as f64],
                                length: length as f64,
                                max_r: maxr as f64,
                            }))
                        }
                    };
                    attacks.push(Attack { center, radius: attack_r, age: 0.0, kind: akind, dir: adir });
                    plan.push((i, vol));
                }
                // 2) Cull pass — the "many queries per frame" workload. Each attack
                //    is a read-only cull against the shared index, so fanning them
                //    over the (live-sized) rayon pool yields the SAME hit sets as
                //    the serial path — only faster. wasm has no threads → serial.
                let tc = Instant::now();
                #[cfg(not(target_arch = "wasm32"))]
                let results: Vec<(usize, Vec<u32>)> = thread_pool.install(|| plan.par_iter().map(|(i, vol)| (*i, index.cull(vol))).collect());
                #[cfg(target_arch = "wasm32")]
                let results: Vec<(usize, Vec<u32>)> = plan.iter().map(|(i, vol)| (*i, index.cull(vol))).collect();
                cull_us += tc.elapsed().as_secs_f64() * 1e6;
                // 3) Resolve kills (serial): anyone inside an attack but the caster.
                for (i, hits) in results {
                    for id in hits { let j = id as usize; if j != i && !killed[j] { killed[j] = true; } }
                }
                // apply kills: burst marker, respawn keeping the kind, flash, count
                for j in 0..critters.len() {
                    if killed[j] {
                        bursts.push((critters[j].pos, 0.0));
                        let k = critters[j].kind;
                        critters[j] = spawn(&mut rng, k, world);
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
            // record/show the active index's cells (combat uses the same index)
            if rec || show_boxes { index.visit_boxes(world, &mut boxes); }
            cand_n = None;
            let drifters = critters.iter().filter(|c| c.kind == 0).count();
            let hunters = critters.iter().filter(|c| c.kind == 1).count();
            let pulsars = critters.iter().filter(|c| c.kind == 2).count();
            stat_line = format!("combat: Drift {:>5} / Hunt {:>5} / Puls {:>5} | last wave {:>4} in {:>7.0} us", drifters, hunters, pulsars, last_wave_attacks, last_wave_us);
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

        // --- build this frame's drawable state (critters + effects + boxes) ---
        let t_prep = Instant::now();
        let mut cur = Frame { boxes, ..Default::default() };
        if render_mode.geom().is_some() {
            for (i, c) in critters.iter().enumerate() {
                let (col, rad) = match sim_mode {
                    SimMode::Observe => (kind_color(c.kind, lit[i]), if lit[i] { 2.2 } else { 1.5 }),
                    SimMode::Combat => {
                        if c.flash > 0.0 { (Color::new(1.0, 1.0, 1.0, 1.0), 2.2) } else { (kind_color(c.kind, false), 1.8) }
                    }
                };
                cur.instances.push(Instance::new(c.pos, rad, [col.r, col.g, col.b, col.a]));
            }
        }
        if render_mode != RenderMode::None && sim_mode == SimMode::Combat {
            for a in &attacks {
                let f = (a.age / ATTACK_LIFE).clamp(0.0, 1.0);
                let alpha = (1.0 - f) * 0.6;
                match a.kind {
                    AttackKind::Sphere => {
                        let grow = 0.7 + 0.5 * f;
                        let model = Mat4::from_scale_rotation_translation(Vec3::splat(a.radius * grow), Quat::IDENTITY, a.center);
                        cur.sphere_fx.push(EffectInstance::new(model, [1.0, 0.7 - 0.5 * f, 0.2, alpha]));
                    }
                    AttackKind::Drop => {
                        let (off, _, _) = flamer_dims(a.radius);
                        let tip = a.center + a.dir * off;
                        let model = Mat4::from_scale_rotation_translation(Vec3::splat(a.radius), Quat::from_rotation_arc(Vec3::Y, a.dir), tip);
                        cur.drop_fx.push(EffectInstance::new(model, [0.25, 0.85, 1.0, alpha]));
                    }
                }
            }
            for (p, age) in &bursts {
                let f = (age / BURST_LIFE).clamp(0.0, 1.0);
                let model = Mat4::from_scale_rotation_translation(Vec3::splat(2.0 + 9.0 * f), Quat::IDENTITY, *p);
                cur.sphere_fx.push(EffectInstance::new(model, [1.0, 1.0, 1.0, (1.0 - f) * 0.8]));
            }
        }
        prep_us = t_prep.elapsed().as_secs_f64() * 1e6;

        // Record into the history ring when the sim actually advanced with REC on.
        if rec && advance {
            hist.push_back(cur.clone());
            while hist.len() > HIST_MAX { hist.pop_front(); }
        }
        cur_frame = cur;

        // Draw the live frame, or a recorded one while scrubbing (`scrub` frames
        // back). Same draw path for both, so replay looks identical to live.
        let view_frame: &Frame = if scrub > 0 && !hist.is_empty() {
            &hist[hist.len().saturating_sub(scrub).min(hist.len() - 1)]
        } else {
            &cur_frame
        };
        draw_world_visuals(&mut renderer, render_mode.geom(), view_frame, mvp, cam_right, cam_up, show_boxes, world, observer);

        // observe-only live extras (vision sphere + sight-lines) — not recorded.
        if scrub == 0 && sim_mode == SimMode::Observe && render_mode != RenderMode::None {
            draw_sphere_wires(observer, vision_r, None, Color::new(1.0, 0.9, 0.3, 0.5));
            draw_sphere(observer, 3.0, None, WHITE);
            for (i, c) in critters.iter().enumerate() {
                if lit[i] { draw_line_3d(c.pos, observer, Color::new(1.0, 0.85, 0.3, 0.25)); }
            }
        }

        // (CPU timing is taken at the very end of the frame — see below — so it
        // includes the HUD/UI; the HUD shows the previous frame's rolling value.)

        // --- HUD (2D overlay) ---
        set_default_camera();
        let hud = |y: f32, s: String| draw_text(&s, 12.0, y, 20.0, Color::new(0.85, 0.9, 1.0, 1.0));
        // Visual-only mode ([U] / web button) hides every 2D overlay: HUD, panel,
        // graphs. Keyboard shortcuts keep working so you can still drive it blind.
        if !ui_hidden {
        // Numbers are right-aligned in fixed-width fields so they don't shift
        // the text sideways as they change (which made it jitter illegibly).
        let mode_str = if sim_mode == SimMode::Combat { "COMBAT " } else { "observe" };
        let play_str = if scrub > 0 { format!("  [REPLAY -{}/{}]", scrub, hist.len()) }
            else if paused { "  [PAUSED]".to_string() }
            else if rec { format!("  [REC {} frames]", hist.len()) }
            else { String::new() };
        hud(24.0, format!("critters 3D  |  pop {:>6}  |  mode {}  |  fps {:>6.0}{}{}", critters.len(), mode_str, fps_display, if fps_cap { " cap" } else { "" }, play_str));
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
            SimMode::Combat => {
                #[cfg(not(target_arch = "wasm32"))]
                let thr = format!(", {} thread{}", cur_threads, if cur_threads == 1 { "" } else { "s" });
                #[cfg(target_arch = "wasm32")]
                let thr = "";
                format!("cull {:>8.3} us (rolling avg, per attack{})", cull_us_avg, thr)
            }
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
        hud(134.0, format!("cpu ~{:>6.2} ms (sim {:>6.0}+build {:>6.0}+prep {:>6.0} us) -> CPU ceiling ~{:>6.0} fps -> {}", cpu_ms_avg, sim_us_avg, t_build_us, prep_us_avg, cpu_ceiling, bound));
        hud(screen_height() - 18.0, "drag/zoom | +/-: pop | [ ]: radius | T R O V M G C B | Space: play/pause | <- ->: step | K: rec | U: hide UI | Esc".to_string());
        }

        // --- on-screen mouse controls (top-right panel; keys still work too) ---
        // Drag capture ends when the button is released.
        if !is_mouse_button_down(MouseButton::Left) { slider_drag = None; }
        let mut pa = PlayerActions::default();
        if !ui_hidden {
        let mut panel = Panel::new(ui_x, ui_y, UI_W, mp_now.0, mp_now.1,
            is_mouse_button_pressed(MouseButton::Left), is_mouse_button_down(MouseButton::Left),
            is_mouse_button_pressed(MouseButton::Right));
        if panel.button(&format!("mode: {} [T]", if sim_mode == SimMode::Combat { "COMBAT" } else { "observe" }),
            "Observe: one vision sphere from the centre\nlights the critters it sees.\nCombat: red critters become predators\nthat attack nearby prey (each attack is\nan index cull).") {
            sim_mode = match sim_mode { SimMode::Observe => SimMode::Combat, SimMode::Combat => SimMode::Observe };
        }
        if panel.button(&format!("structure: {} [M]", structure.label()),
            "Index used for the culls:\nTree3 (binary, persistent + update),\nOctree3 (8-way, persistent + update),\nMortonGrid3 (Z-order hash, rebuilt),\nprojection (one 2D tree on xy + z-reject).") { structure = structure.next(); }
        if panel.button(&format!("render: {} [G]", render_mode.label()),
            "How critters are drawn:\ninstanced spheres / round billboards /\nsquare billboards (fastest) /\nNO RENDER (CPU only, to read the ceiling).") { render_mode = render_mode.next(); }
        panel.separator();
        if panel.button(&format!("attacks: {} [R]", if random_attacks { "desynced" } else { "SYNCED" }),
            "Combat attack timing:\ndesynced = min + random cooldowns;\nSYNCED = all predators fire together,\na saturation spike to stress the index.") { random_attacks = !random_attacks; }
        if panel.button(&format!("separation: {} [O]", if separation { "ON" } else { "off" }),
            "Push critters apart so none overlap.\nOne neighbour cull per critter -> heavy\nat high counts (a good index workload).") { separation = !separation; }
        if panel.button(&format!("fps cap: {} [V]", if fps_cap { "165" } else { "OFF" }),
            "vsync is off so the fps counter shows the\nreal ceiling. Cap to ~165 to stop the GPU\nrendering frames the monitor can't show.") { fps_cap = !fps_cap; }
        if panel.button(&format!("leaf boxes: {} [B]", if show_boxes { "ON" } else { "off" }),
            "Draw the index's leaf cells (the boxes the\nculling descends through).") { show_boxes = !show_boxes; }
        if panel.button(&format!("cull reps: {} [C]", cull_reps),
            "Repeat the timed cull N times for a stable\nper-cull microsecond reading (one cull is\ntoo fast to time on its own).") { cull_rep_idx = (cull_rep_idx + 1) % cull_rep_steps.len(); }
        panel.separator();
        panel.slider("Drifter", 0.0, pop_cap, 100.0, &mut pop_kind[0], 0, &mut editing, &mut edit_buf, &mut slider_drag, Some(kind_color(0, false)),
            "How many Drifters (random-direction drop).\n-/+, drag, or right-click to type.");
        panel.slider("Hunter", 0.0, pop_cap, 100.0, &mut pop_kind[1], 1, &mut editing, &mut edit_buf, &mut slider_drag, Some(kind_color(1, false)),
            "How many Hunters (chase + aimed drop).");
        panel.slider("Pulsar", 0.0, pop_cap, 100.0, &mut pop_kind[2], 2, &mut editing, &mut edit_buf, &mut slider_drag, Some(kind_color(2, false)),
            "How many Pulsars (spin + sphere blast).");
        match sim_mode {
            SimMode::Observe => panel.slider("vision r", 6.0, world, 5.0, &mut vision_r, 3, &mut editing, &mut edit_buf, &mut slider_drag, None,
                "Radius of the centre vision sphere.\nRight-click to type a number."),
            SimMode::Combat => panel.slider("attack r", 4.0, world * 0.5, 5.0, &mut attack_r, 3, &mut editing, &mut edit_buf, &mut slider_drag, None,
                "Base size of the attacks.\nRight-click to type a number."),
        }
        // Combat's attack culls fan out over a rayon pool; the slider sizes it
        // live (native only; a 1-core box has nothing to tune). Watch cull-us / fps.
        #[cfg(not(target_arch = "wasm32"))]
        if sim_mode == SimMode::Combat && max_threads > 1 {
            panel.slider("threads", 1.0, max_threads as f32, 1.0, &mut n_threads_f, 4, &mut editing, &mut edit_buf, &mut slider_drag, None,
                "Rayon threads for the combat attack culls\n(the many-queries-per-frame workload).\nDrag and watch cull us / fps change.");
        }
        panel.stepper("world size", &SIZE_STEPS, &mut world, 4, &mut slider_drag,
            "Side of the cube the action lives in.\nStepped powers of 2 -- changing it rebuilds\nthe index and re-bounds the critters.");
        panel.separator();
        // Media-player controls: REC | step back | play/pause | step forward.
        pa = panel.player(rec, paused || scrub > 0);
        panel.finish(ui_y);
        }

        // Apply playback actions (panel buttons + keys), with hold-to-repeat on
        // the step controls. Play/pause from the history jumps back to live.
        if is_key_pressed(KeyCode::K) || pa.rec { rec = !rec; }
        if is_key_pressed(KeyCode::Space) || pa.play {
            if paused || scrub > 0 { paused = false; scrub = 0; } else { paused = true; }
        }
        let back = back_rep.fires(
            is_key_pressed(KeyCode::Left) || pa.back_click,
            is_key_down(KeyCode::Left) || pa.back_held,
            dt,
        );
        if back { scrub = (scrub + 1).min(hist.len().saturating_sub(1)); }
        let fwd = fwd_rep.fires(
            is_key_pressed(KeyCode::Right) || pa.fwd_click,
            is_key_down(KeyCode::Right) || pa.fwd_held,
            dt,
        );
        if fwd { if scrub > 0 { scrub -= 1; } else { step_request = true; } }

        // Time-series graphs below the control panel.
        push_hist(&mut g_fps, fps_display);
        push_hist(&mut g_cpu, cpu_ms_avg as f32);
        push_hist(&mut g_cull, cull_us_avg as f32);
        if !ui_hidden {
        let gh = 64.0;
        let mut gy = ui_y + UI_H + 8.0;
        draw_graph(ui_x, gy, UI_W, gh, "fps", &g_fps, Color::new(0.45, 0.9, 0.55, 1.0));
        gy += gh + 4.0;
        draw_graph(ui_x, gy, UI_W, gh, "cpu ms", &g_cpu, Color::new(1.0, 0.82, 0.35, 1.0));
        gy += gh + 4.0;
        draw_graph(ui_x, gy, UI_W, gh, "cull us", &g_cull, Color::new(0.4, 0.82, 1.0, 1.0));
        }
        for k in 0..3 { pop_kind[k] = pop_kind[k].round().clamp(0.0, pop_cap); }

        // World resized (stepper): re-bound the critters into the new cube. The
        // index notices `idx_world != world` next frame and rebuilds itself.
        if (world - prev_world).abs() > 0.5 {
            dist = (dist * world / prev_world).clamp(world * 0.6, world * 6.0);
            for c in critters.iter_mut() {
                for axis in 0..3 { c.pos[axis] = c.pos[axis].clamp(MARGIN, world - MARGIN); }
            }
            prev_world = world;
        }

        // Reseed cooldowns when the desync toggle flips (key or panel), so the
        // change is visible immediately (spread out vs each kind in lockstep).
        if random_attacks != prev_random {
            for c in critters.iter_mut() {
                let (cmin, cmax) = kind_cooldown(c.kind);
                c.cooldown = if random_attacks { rng.range(0.0, cmax) } else { cmin };
            }
            prev_random = random_attacks;
        }

        // End-of-frame CPU wall-clock (includes the HUD + UI panel), rolling
        // averaged. Shown next frame by the HUD above.
        let ema = |avg: f64, x: f64| if avg <= 0.0 { x } else { avg * 0.9 + x * 0.1 };
        let frame_cpu_ms = frame_t0.elapsed().as_secs_f64() * 1000.0;
        cpu_ms_avg = ema(cpu_ms_avg, frame_cpu_ms);
        sim_us_avg = ema(sim_us_avg, sim_us);
        prep_us_avg = ema(prep_us_avg, prep_us);

        // Stress sweep: accumulate true frame time + CPU ms after a warmup.
        if frame >= 30 {
            stress_secs += get_frame_time() as f64;
            stress_cpu_ms += frame_cpu_ms;
            stress_n += 1;
        }

        // Optional fps cap (V): with vsync off the GPU would otherwise render
        // thousands of fps the monitor can't show. Sleep the bulk of the
        // remaining frame budget, then spin the last ~2 ms for a precise cap.
        // On the web the browser caps the frame rate (requestAnimationFrame),
        // and blocking the main thread isn't allowed — so this spin/sleep cap is
        // native-only.
        #[cfg(not(target_arch = "wasm32"))]
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
        if let Some(m) = max_frames {
            if frame >= m {
                if stress_n > 0 {
                    let mean_fps = stress_n as f64 / stress_secs.max(1e-9);
                    let mean_frame_ms = 1000.0 / mean_fps.max(1e-9);
                    let mean_cpu_ms = stress_cpu_ms / stress_n as f64;
                    let cpu_ceiling = if mean_cpu_ms > 0.0 { 1000.0 / mean_cpu_ms } else { 0.0 };
                    let bound = if mean_cpu_ms >= 0.85 * mean_frame_ms { "CPU-BOUND" } else { "GPU-BOUND" };
                    println!(
                        "STRESS pop={:>7} render={:<16} structure={:<22} mode={} | mean_fps={:>7.1} frame_ms={:>6.2} cpu_ms={:>6.2} cpu_ceiling_fps={:>7.0} -> {}",
                        critters.len(), render_mode.label(), structure.label(),
                        if sim_mode == SimMode::Combat { "combat" } else { "observe" },
                        mean_fps, mean_frame_ms, mean_cpu_ms, cpu_ceiling, bound,
                    );
                }
                break;
            }
        }
        if let Some(p) = std::env::var_os("SHOT") { if frame >= 120 { let _ = get_screen_data().export_png(&p.to_string_lossy()); std::process::exit(0); } }
        next_frame().await;
    }
}
