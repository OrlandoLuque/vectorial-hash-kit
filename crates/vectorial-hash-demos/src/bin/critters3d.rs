//! Visual 3D critters — the 3D analogue of the 2D `critters` demo, drawn
//! with macroquad's 3D pipeline. Critters drift inside a cube indexed by
//! `Tree3`; an observer at the centre runs a sphere "vision" cull every
//! frame and the culled critters light up and draw a sight-line. The same
//! index workload runs headless via `critters3d_headless`.
//!
//! Run: `cargo run -p vectorial-hash-demos --bin critters3d --release`
//!
//! Controls:
//! - drag left mouse: orbit the camera; scroll: zoom
//! - `+` / `-`: add / remove 200 critters
//! - `[` / `]`: shrink / grow the vision radius
//! - `M`: cycle the index structure — binary-3D `Tree3` / `Octree3` (8-way) /
//!   projection (one 2D `Tree` on xy + z-reject, the author's variant)
//! - `B`: toggle the leaf-box wireframes (extruded columns in projection mode)
//! - `Space`: pause, `Esc`: quit
//!
//! Env: CRITTERS3D_MAX_FRAMES=N exits after N frames (smoke testing).

use macroquad::prelude::*;

use vectorial_hash::{
    Aabb, Octree3, Point, Point3, Positioned, Positioned3, Rect, Shape, Sphere3, Tree, Tree3,
};

const WORLD: f32 = 200.0;
const MARGIN: f32 = 4.0;
const ITEM_LIMIT: usize = 16;

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
    kind: u8, // 0,1,2 — colour only
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
    Conf { window_title: "vectorial-hash critters 3D".to_owned(), window_width: 1100, window_height: 800, ..Default::default() }
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
        Critter { pos: p, vel: v, kind }
    };
    for i in 0..2400 {
        critters.push(spawn(&mut rng, (i % 3) as u8));
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
    let mut frame: u64 = 0;
    let max_frames: Option<u64> = std::env::var("CRITTERS3D_MAX_FRAMES").ok().and_then(|s| s.parse().ok());

    loop {
        let dt = (get_frame_time()).min(1.0 / 30.0);

        // --- input ---
        if is_key_pressed(KeyCode::Escape) { break; }
        if is_key_pressed(KeyCode::Space) { paused = !paused; }
        if is_key_pressed(KeyCode::B) { show_boxes = !show_boxes; }
        if is_key_pressed(KeyCode::M) { structure = structure.next(); }
        if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) {
            for i in 0..200 { critters.push(spawn(&mut rng, (i % 3) as u8)); }
        }
        if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract) {
            for _ in 0..200 { critters.pop(); }
        }
        if is_key_down(KeyCode::LeftBracket) { vision_r = (vision_r - 60.0 * dt).max(6.0); }
        if is_key_down(KeyCode::RightBracket) { vision_r = (vision_r + 60.0 * dt).min(WORLD); }
        let mp = mouse_position();
        if is_mouse_button_down(MouseButton::Left) {
            yaw += (mp.0 - last_mouse.0) * 0.01;
            pitch = (pitch + (mp.1 - last_mouse.1) * 0.01).clamp(-1.4, 1.4);
        }
        last_mouse = mp;
        let scroll = mouse_wheel().1;
        if scroll != 0.0 { dist = (dist * (1.0 - scroll.signum() * 0.1)).clamp(WORLD * 0.6, WORLD * 6.0); }

        // --- simulate ---
        if !paused {
            for c in critters.iter_mut() {
                let mut np = c.pos + c.vel * dt;
                for axis in 0..3 {
                    if np[axis] < MARGIN { np[axis] = MARGIN; c.vel[axis] = -c.vel[axis]; }
                    if np[axis] > WORLD - MARGIN { np[axis] = WORLD - MARGIN; c.vel[axis] = -c.vel[axis]; }
                }
                c.pos = np;
            }
        }

        // --- (re)build the chosen index and run a vision cull from the centre ---
        let observer = vec3(WORLD * 0.5, WORLD * 0.5, WORLD * 0.5);
        let (ox, oy, oz, r) = (observer.x as f64, observer.y as f64, observer.z as f64, vision_r as f64);
        let mut lit = vec![false; critters.len()];
        let mut boxes: Vec<(Vec3, Vec3)> = Vec::new();
        let stat_line: String;
        let cand_n: Option<usize>; // broadphase candidates (projection only)

        match structure {
            Structure::Binary3 | Structure::Octree => {
                // Both go through the exact Shape3 sphere cull; pick the type.
                let octree = matches!(structure, Structure::Octree);
                let sphere = Sphere3::new(ox, oy, oz, r);
                let (leaves, arena) = if octree {
                    let mut t = Octree3::<C3>::new(world_aabb(), ITEM_LIMIT);
                    for (i, c) in critters.iter().enumerate() {
                        t.insert(C3 { id: i as u32, p: Point3::new(c.pos.x as f64, c.pos.y as f64, c.pos.z as f64) });
                    }
                    for c in t.cull(&sphere) { lit[c.id as usize] = true; }
                    if show_boxes { t.visit_leaves(|l| boxes.push(aabb_box(&l.bbox))); }
                    (t.leaf_count(), t.node_count())
                } else {
                    let mut t = Tree3::<C3>::new(world_aabb(), ITEM_LIMIT);
                    for (i, c) in critters.iter().enumerate() {
                        t.insert(C3 { id: i as u32, p: Point3::new(c.pos.x as f64, c.pos.y as f64, c.pos.z as f64) });
                    }
                    for c in t.cull(&sphere) { lit[c.id as usize] = true; }
                    if show_boxes { t.visit_leaves(|l| boxes.push(aabb_box(&l.bbox))); }
                    (t.leaf_count(), t.node_count())
                };
                cand_n = None;
                stat_line = format!("{}: {} leaves, {} arena nodes (item_limit {})", structure.label(), leaves, arena, ITEM_LIMIT);
            }
            Structure::Projection => {
                // Author's variant: index the xy projection in a 2D Tree, cull
                // the sphere's shadow (a disc), then z-reject + exact 3D test.
                let mut t = Tree::<P2>::new(Rect::new(0.0, 0.0, WORLD as f64, WORLD as f64), ITEM_LIMIT);
                for (i, c) in critters.iter().enumerate() {
                    t.insert(P2 { id: i as u32, p: Point::new(c.pos.x as f64, c.pos.y as f64), z: c.pos.z as f64 });
                }
                let disc = Disc { cx: ox, cy: oy, r };
                let cand = t.cull(&disc);
                let r2 = r * r;
                let mut nc = 0;
                for p2 in &cand {
                    nc += 1;
                    let (dx, dy, dz) = (p2.p.x - ox, p2.p.y - oy, p2.z - oz);
                    if dx * dx + dy * dy + dz * dz <= r2 { lit[p2.id as usize] = true; }
                }
                // boxes: the 2D leaf rects, extruded through the full z depth.
                if show_boxes {
                    t.visit_leaves(|_, l| {
                        let b = l.bbox;
                        let c = vec3((b.x + b.width * 0.5) as f32, (b.y + b.height * 0.5) as f32, WORLD * 0.5);
                        boxes.push((c, vec3(b.width as f32, b.height as f32, WORLD)));
                    });
                }
                cand_n = Some(nc);
                stat_line = format!("{}: {} leaves, {} arena nodes (item_limit {})", structure.label(), t.leaf_count(), t.node_count(), ITEM_LIMIT);
            }
        }
        let seen_n = lit.iter().filter(|&&b| b).count();

        // --- render 3D ---
        clear_background(Color::new(0.05, 0.06, 0.09, 1.0));
        let eye = observer + vec3(
            dist * pitch.cos() * yaw.cos(),
            dist * pitch.sin(),
            dist * pitch.cos() * yaw.sin(),
        );
        set_camera(&Camera3D { position: eye, up: vec3(0.0, 1.0, 0.0), target: observer, ..Default::default() });

        // world box
        draw_cube_wires(observer, vec3(WORLD, WORLD, WORLD), Color::new(0.3, 0.35, 0.45, 1.0));

        // optional leaf-box wireframes (collected during the cull above)
        if show_boxes {
            for (c, sz) in &boxes {
                draw_cube_wires(*c, *sz, Color::new(0.18, 0.22, 0.3, 1.0));
            }
        }

        // vision sphere (translucent wire via a smaller solid + wire cube proxy)
        draw_sphere_wires(observer, vision_r, None, Color::new(1.0, 0.9, 0.3, 0.5));

        // critters
        for (i, c) in critters.iter().enumerate() {
            let col = kind_color(c.kind, lit[i]);
            draw_sphere(c.pos, if lit[i] { 2.2 } else { 1.5 }, None, col);
            if lit[i] {
                draw_line_3d(c.pos, observer, Color::new(1.0, 0.85, 0.3, 0.25));
            }
        }
        // observer marker
        draw_sphere(observer, 3.0, None, WHITE);

        // --- HUD (2D overlay) ---
        set_default_camera();
        let hud = |y: f32, s: String| draw_text(&s, 12.0, y, 20.0, Color::new(0.85, 0.9, 1.0, 1.0));
        hud(24.0, format!("critters 3D  |  pop {}  |  fps {}", critters.len(), get_fps()));
        hud(46.0, stat_line);
        let seen_str = match cand_n {
            Some(nc) => format!("vision r={:.0}  ->  {} candidates -> {} seen", vision_r, nc, seen_n),
            None => format!("vision r={:.0}  ->  {} seen", vision_r, seen_n),
        };
        hud(68.0, format!("{}{}", seen_str, if paused { "   [PAUSED]" } else { "" }));
        hud(screen_height() - 18.0, "drag: orbit | scroll: zoom | +/-: pop | [ ]: vision | M: structure | B: boxes | Space: pause | Esc".to_string());

        frame += 1;
        if let Some(m) = max_frames { if frame >= m { break; } }
        next_frame().await;
    }
}
