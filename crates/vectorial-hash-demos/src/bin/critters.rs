//! Visual critters demo.
//!
//! A 2D map indexed by a `vectorial_hash::Tree`. Critters of three kinds move
//! with distinct behaviours and attack using **precomputed template areas**
//! (a template set generated at startup and served from a `TemplateIndex`).
//! Dead critters respawn after a delay. Each live tree region is drawn in its
//! own colour, so splits and merge-ups are visible in real time.
//!
//! Run: `cargo run -p vectorial-hash-demos --bin critters --release`
//!
//! Controls:
//! - `1`/`2`/`3` select the spawn brush (drifter / hunter / pulsar)
//! - left click spawns a brush critter at the cursor, right click removes the
//!   critter under the cursor (no respawn)
//! - `+`/`-` spawn / remove five critters at random
//! - `R` cycles region rendering (fill+lines / lines / off)
//! - `[` / `]` halve / double simulation speed, `Space` pauses, `Esc` quits
//! - the "tuning (live)" panel adjusts, while running: the tree's split and
//!   merge thresholds (tree is rebuilt on change), per-kind population
//!   targets, respawn delay, simulation speed and fire rate
//!
//! Env: CRITTERS_MAX_FRAMES=N exits after N frames (smoke testing).

use std::collections::{HashMap, HashSet};

use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets};

use vectorial_hash::{
    CellState, Point as VPoint, Positioned, Rect as VRect, Shape, TemplateGrid, Tree,
};
use vectorial_hash_templates::adapter::{matrix_to_template_grid, TemplateIndex, TemplateKey};
use vectorial_hash_templates::polygon::{create_circle, create_drop, rotated_copy, scaled_copy, Polygon};
use vectorial_hash_templates::templates::{angle_to_radians, get_template_grid_fast, TemplateStore};

const MAP_W: f64 = 800.0;
const MAP_H: f64 = 800.0;
const PANEL_W: f32 = 320.0;
const MARGIN: f64 = 4.0;

/// Template-grid cell size (px) used for the attack templates.
const G: i64 = 20;
/// Sub-cell anchor quantization: 4 offsets per axis, 5 px apart.
const OFF_STEPS: i32 = 4;
const OFF_STEP: f64 = G as f64 / OFF_STEPS as f64;
const ANGLE_STEP_DEG: f64 = 15.0;

const DROP_ID: u32 = 0;
const CIRCLE_ID: u32 = 1;
const DROP_SCALE: f64 = 110.0;
const CIRCLE_SCALE: f64 = 48.0;

const ITEM_LIMIT: usize = 3;
const RESPAWN_DELAY: f64 = 2.5;
const EFFECT_TTL: f64 = 0.45;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    Drifter,
    Hunter,
    Pulsar,
}

impl Kind {
    fn color(self) -> Color {
        match self {
            Kind::Drifter => SKYBLUE,
            Kind::Hunter => RED,
            Kind::Pulsar => GOLD,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Kind::Drifter => "drifter",
            Kind::Hunter => "hunter",
            Kind::Pulsar => "pulsar",
        }
    }
    fn speed(self) -> f64 {
        match self {
            Kind::Drifter => 60.0,
            Kind::Hunter => 85.0,
            Kind::Pulsar => 70.0,
        }
    }
    fn cooldown(self) -> (f32, f32) {
        match self {
            Kind::Drifter => (2.5, 4.5),
            Kind::Hunter => (1.8, 3.5),
            Kind::Pulsar => (3.0, 5.0),
        }
    }
}

#[derive(Clone)]
struct Critter {
    id: u32,
    kind: Kind,
    pos: VPoint,
    heading: f64,
}

impl Positioned for Critter {
    fn position(&self) -> VPoint {
        self.pos
    }
}

/// The "occasion" template set: every (shape, angle, sub-cell offset) combo
/// decoded once at startup into the hash-keyed TemplateIndex, plus the base
/// rotated polygons used for the per-point fallback at attack time.
struct Arsenal {
    index: TemplateIndex,
    base_polys: HashMap<(u32, i64), Polygon>,
    combos: usize,
    unique: u32,
}

fn build_arsenal() -> Arsenal {
    let store = TemplateStore::new();
    let mut index = TemplateIndex::new();
    let mut base_polys = HashMap::new();
    let mut combos = 0usize;

    let shapes: [(u32, f64, Polygon, Vec<f64>); 2] = [
        (
            DROP_ID,
            DROP_SCALE,
            scaled_copy(&create_drop(0.2, 0.8), DROP_SCALE, DROP_SCALE),
            (0..(360.0 / ANGLE_STEP_DEG) as i64)
                .map(|i| i as f64 * ANGLE_STEP_DEG)
                .collect(),
        ),
        (
            CIRCLE_ID,
            CIRCLE_SCALE,
            scaled_copy(&create_circle(1.0), CIRCLE_SCALE, CIRCLE_SCALE),
            vec![0.0],
        ),
    ];

    for (shape_id, scale, base, angles) in shapes {
        for angle in angles {
            let rotated = rotated_copy(&base, angle_to_radians(angle));
            base_polys.insert((shape_id, angle as i64), rotated.clone());
            for ox in 0..OFF_STEPS {
                for oy in 0..OFF_STEPS {
                    let mut moved = rotated.clone();
                    moved.move_by(ox as f64 * OFF_STEP, oy as f64 * OFF_STEP);
                    let gxr = [
                        (moved.x_min / G as f64).floor() as i64,
                        (moved.x_max / G as f64).ceil() as i64,
                    ];
                    let gyr = [
                        (moved.y_min / G as f64).floor() as i64,
                        (moved.y_max / G as f64).ceil() as i64,
                    ];
                    let matrix =
                        get_template_grid_fast(gxr[0], gyr[0], gxr[1], gyr[1], G, G, &moved);
                    store.store_dedup(
                        &matrix,
                        &format!("s{}-a{}-o{},{}", shape_id, angle, ox, oy),
                    );
                    let anchor =
                        VPoint::new(gxr[0] as f64 * G as f64, gyr[0] as f64 * G as f64);
                    let grid = matrix_to_template_grid(&matrix, anchor, G as f64, G as f64);
                    index.insert(
                        TemplateKey::new(shape_id, scale, angle, G, ox, oy),
                        grid,
                    );
                    combos += 1;
                }
            }
        }
    }

    Arsenal { index, base_polys, combos, unique: store.template_count() }
}

/// Attack area: precomputed template re-anchored at the (quantized) attack
/// position + the matching polygon for yellow-cell per-point checks.
struct AttackShape {
    poly: Polygon,
    bbox: VRect,
    grid: TemplateGrid,
}

impl Shape for AttackShape {
    fn bounding_box(&self) -> VRect {
        self.bbox
    }
    fn contains_point(&self, p: VPoint) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
    fn template_grid(&self) -> Option<&TemplateGrid> {
        Some(&self.grid)
    }
}

/// Build an attack at `pos`. For the drop, `aim` is the direction to fire
/// along (the drop's body extends along +y at angle 0). For the circle blast,
/// pass `None`.
fn make_attack(arsenal: &Arsenal, shape_id: u32, pos: VPoint, aim: Option<(f64, f64)>) -> Option<AttackShape> {
    let angle_deg = match aim {
        Some((dx, dy)) => {
            let a = (-dx).atan2(dy).to_degrees();
            ((a / ANGLE_STEP_DEG).round() * ANGLE_STEP_DEG).rem_euclid(360.0)
        }
        None => 0.0,
    };

    let bx = (pos.x / G as f64).floor();
    let by = (pos.y / G as f64).floor();
    let ox = (((pos.x - bx * G as f64) / OFF_STEP).floor() as i32).clamp(0, OFF_STEPS - 1);
    let oy = (((pos.y - by * G as f64) / OFF_STEP).floor() as i32).clamp(0, OFF_STEPS - 1);

    let scale = if shape_id == DROP_ID { DROP_SCALE } else { CIRCLE_SCALE };
    let key = TemplateKey::new(shape_id, scale, angle_deg, G, ox, oy);
    let grid = arsenal.index.get(&key)?.translated(bx * G as f64, by * G as f64);

    let mut poly = arsenal.base_polys.get(&(shape_id, angle_deg as i64))?.clone();
    poly.move_by(
        bx * G as f64 + ox as f64 * OFF_STEP,
        by * G as f64 + oy as f64 * OFF_STEP,
    );
    let bbox = VRect::new(
        poly.x_min,
        poly.y_min,
        poly.x_max - poly.x_min,
        poly.y_max - poly.y_min,
    );
    Some(AttackShape { poly, bbox, grid })
}

struct Effect {
    grid: TemplateGrid,
    origin: VPoint,
    color: Color,
    until: f64,
    /// Sampled outline of the actual attack polygon (arcs flattened).
    outline: Vec<(f32, f32)>,
}

/// Flatten a polygon (lines + arcs) into a point list for rendering.
fn sample_polygon(poly: &Polygon) -> Vec<(f32, f32)> {
    let n = poly.vertices.len();
    let mut pts = Vec::new();
    for i in 0..n {
        let v = &poly.vertices[i];
        let w = &poly.vertices[(i + 1) % n];
        pts.push((v.x as f32, v.y as f32));
        if v.seg.d != 0 {
            let (xc, yc) = (v.seg.xc, v.seg.yc);
            let r = ((v.x - xc).powi(2) + (v.y - yc).powi(2)).sqrt();
            let a0 = (v.y - yc).atan2(v.x - xc);
            let a1 = (w.y - yc).atan2(w.x - xc);
            let mut sweep = a1 - a0;
            if v.seg.d == 1 && sweep <= 0.0 {
                sweep += std::f64::consts::TAU;
            }
            if v.seg.d == -1 && sweep >= 0.0 {
                sweep -= std::f64::consts::TAU;
            }
            let steps = (sweep.abs() / 0.2).ceil().max(2.0) as usize;
            for s in 1..steps {
                let a = a0 + sweep * s as f64 / steps as f64;
                pts.push(((xc + r * a.cos()) as f32, (yc + r * a.sin()) as f32));
            }
        }
    }
    pts
}

/// Expanding ring marking a spawn, a removal, or a death.
struct Ring {
    pos: VPoint,
    color: Color,
    start: f64,
    until: f64,
}

fn spawn_at(tree: &mut Tree<Critter>, next_id: &mut u32, kind: Kind, pos: VPoint) -> VPoint {
    let id = *next_id;
    *next_id += 1;
    let heading = rand::gen_range(0.0_f32, std::f32::consts::TAU) as f64;
    tree.insert(Critter { id, kind, pos, heading });
    pos
}

fn spawn(tree: &mut Tree<Critter>, next_id: &mut u32, kind: Kind) -> VPoint {
    let pos = VPoint::new(
        rand::gen_range(20.0_f32, (MAP_W - 20.0) as f32) as f64,
        rand::gen_range(20.0_f32, (MAP_H - 20.0) as f32) as f64,
    );
    spawn_at(tree, next_id, kind, pos)
}

/// One movement step. Returns (new position, new heading).
fn step_critter(
    kind: Kind,
    pos: VPoint,
    heading: f64,
    dt: f64,
    snap: &[(u32, Kind, VPoint, f64)],
    self_id: u32,
) -> (VPoint, f64) {
    let mut h = heading;
    match kind {
        // Smooth random walk.
        Kind::Drifter => h += rand::gen_range(-2.5_f32, 2.5) as f64 * dt,
        // Constant turn rate: circles around the map.
        Kind::Pulsar => h += 1.6 * dt,
        // Steer toward the nearest critter of another kind.
        Kind::Hunter => {
            let target = snap
                .iter()
                .filter(|(id, k, _, _)| *id != self_id && *k != Kind::Hunter)
                .min_by(|a, b| {
                    let da = (a.2.x - pos.x).powi(2) + (a.2.y - pos.y).powi(2);
                    let db = (b.2.x - pos.x).powi(2) + (b.2.y - pos.y).powi(2);
                    da.partial_cmp(&db).unwrap()
                });
            match target {
                Some(&(_, _, tpos, _)) => {
                    let desired = (tpos.y - pos.y).atan2(tpos.x - pos.x);
                    let diff = (desired - h + std::f64::consts::PI)
                        .rem_euclid(std::f64::consts::TAU)
                        - std::f64::consts::PI;
                    h += diff.clamp(-3.0 * dt, 3.0 * dt);
                }
                None => h += rand::gen_range(-2.0_f32, 2.0) as f64 * dt,
            }
        }
    }

    let speed = kind.speed();
    let mut nx = pos.x + h.cos() * speed * dt;
    let mut ny = pos.y + h.sin() * speed * dt;
    if nx < MARGIN || nx > MAP_W - MARGIN {
        h = std::f64::consts::PI - h;
        nx = nx.clamp(MARGIN, MAP_W - MARGIN);
    }
    if ny < MARGIN || ny > MAP_H - MARGIN {
        h = -h;
        ny = ny.clamp(MARGIN, MAP_H - MARGIN);
    }
    (VPoint::new(nx, ny), h.rem_euclid(std::f64::consts::TAU))
}

/// Stable distinct colour per NodeId (golden-ratio hue walk).
fn region_color(id: u32) -> Color {
    let hue = (id as f64 * 0.618_033_988_75).fract() as f32;
    hue_to_rgb(hue)
}

fn hue_to_rgb(hue: f32) -> Color {
    let h = hue * 6.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as i32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    Color::new(0.25 + 0.75 * r, 0.25 + 0.75 * g, 0.25 + 0.75 * b, 1.0)
}

fn window_conf() -> Conf {
    Conf {
        window_title: "vectorial-hash critters".to_owned(),
        window_width: MAP_W as i32 + PANEL_W as i32,
        window_height: MAP_H as i32,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand(42);

    let arsenal = build_arsenal();
    let world = VRect::new(0.0, 0.0, MAP_W, MAP_H);
    let mut tree: Tree<Critter> = Tree::with_limits(world, ITEM_LIMIT, ITEM_LIMIT);
    let mut next_id = 0u32;
    let mut cooldowns: HashMap<u32, f64> = HashMap::new();

    // Live-tunable settings (sliders in the side panel).
    let mut split_f: f32 = ITEM_LIMIT as f32; // leaf splits above this
    let mut merge_f: f32 = ITEM_LIMIT as f32; // siblings merge at/below this
    let mut drifters_f: f32 = 16.0;
    let mut hunters_f: f32 = 12.0;
    let mut pulsars_f: f32 = 14.0;
    let mut respawn_f: f32 = RESPAWN_DELAY as f32;
    let mut speed_f: f32 = 1.0;
    let mut fire_f: f32 = 1.0;

    for _ in 0..16 {
        spawn(&mut tree, &mut next_id, Kind::Drifter);
    }
    for _ in 0..12 {
        spawn(&mut tree, &mut next_id, Kind::Hunter);
    }
    for _ in 0..14 {
        spawn(&mut tree, &mut next_id, Kind::Pulsar);
    }

    let mut respawns: Vec<(f64, Kind)> = Vec::new();
    let mut effects: Vec<Effect> = Vec::new();
    let mut rings: Vec<Ring> = Vec::new();
    let mut kills: u64 = 0;
    let mut kills_by: HashMap<Kind, u64> = HashMap::new();
    let mut last_paint: f64 = 0.0;
    let mut paused = false;
    let mut brush = Kind::Hunter;
    let mut region_mode: u8 = 0; // 0 = fill + lines, 1 = lines, 2 = off

    let max_frames: u64 = std::env::var("CRITTERS_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut frame: u64 = 0;

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::Key1) {
            brush = Kind::Drifter;
        }
        if is_key_pressed(KeyCode::Key2) {
            brush = Kind::Hunter;
        }
        if is_key_pressed(KeyCode::Key3) {
            brush = Kind::Pulsar;
        }
        if is_key_pressed(KeyCode::R) {
            region_mode = (region_mode + 1) % 3;
        }
        if is_key_pressed(KeyCode::LeftBracket) {
            speed_f = (speed_f * 0.5).max(0.25);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            speed_f = (speed_f * 2.0).min(4.0);
        }

        let now = get_time();
        let dt = (get_frame_time() as f64).min(0.05) * speed_f as f64;

        // --- interactive add / remove ---
        let (mx, my) = mouse_position();
        let mouse_in_map = (mx as f64) < MAP_W && mx >= 0.0 && my >= 0.0 && (my as f64) < MAP_H;
        // Hold to paint a stream of critters.
        if is_mouse_button_down(MouseButton::Left)
            && mouse_in_map
            && tree.item_count() < 400
            && now - last_paint > 0.12
        {
            last_paint = now;
            let pos = spawn_at(
                &mut tree,
                &mut next_id,
                brush,
                VPoint::new(mx as f64, my as f64),
            );
            rings.push(Ring { pos, color: brush.color(), start: now, until: now + 0.4 });
            match brush {
                Kind::Drifter => drifters_f += 1.0,
                Kind::Hunter => hunters_f += 1.0,
                Kind::Pulsar => pulsars_f += 1.0,
            }
        }
        if is_mouse_button_pressed(MouseButton::Right) && mouse_in_map {
            // Remove the critter closest to the cursor (within 30 px), no respawn.
            let mut best: Option<(f64, u32, VPoint, Kind)> = None;
            tree.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    let d = (c.pos.x - mx as f64).powi(2) + (c.pos.y - my as f64).powi(2);
                    if d < 30.0 * 30.0 && best.is_none_or(|(bd, ..)| d < bd) {
                        best = Some((d, c.id, c.pos, c.kind));
                    }
                }
            });
            if let Some((_, id, pos, kind)) = best {
                tree.remove(pos, |c| c.id == id);
                cooldowns.remove(&id);
                rings.push(Ring { pos, color: kind.color(), start: now, until: now + 0.4 });
                match kind {
                    Kind::Drifter => drifters_f = (drifters_f - 1.0).max(0.0),
                    Kind::Hunter => hunters_f = (hunters_f - 1.0).max(0.0),
                    Kind::Pulsar => pulsars_f = (pulsars_f - 1.0).max(0.0),
                }
            }
        }
        if is_key_pressed(KeyCode::Equal) && tree.item_count() < 400 {
            for _ in 0..5 {
                let pos = spawn(&mut tree, &mut next_id, brush);
                rings.push(Ring { pos, color: brush.color(), start: now, until: now + 0.4 });
                match brush {
                    Kind::Drifter => drifters_f += 1.0,
                    Kind::Hunter => hunters_f += 1.0,
                    Kind::Pulsar => pulsars_f += 1.0,
                }
            }
        }
        if is_key_pressed(KeyCode::Minus) {
            // Remove up to five random critters, no respawn.
            let mut all: Vec<(u32, VPoint, Kind)> = Vec::new();
            tree.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    all.push((c.id, c.pos, c.kind));
                }
            });
            for _ in 0..5 {
                if all.is_empty() {
                    break;
                }
                let i = rand::gen_range(0, all.len());
                let (id, pos, kind) = all.swap_remove(i);
                tree.remove(pos, |c| c.id == id);
                cooldowns.remove(&id);
                rings.push(Ring { pos, color: kind.color(), start: now, until: now + 0.4 });
                match kind {
                    Kind::Drifter => drifters_f = (drifters_f - 1.0).max(0.0),
                    Kind::Hunter => hunters_f = (hunters_f - 1.0).max(0.0),
                    Kind::Pulsar => pulsars_f = (pulsars_f - 1.0).max(0.0),
                }
            }
        }

        if !paused {
            // Snapshot all live critters (id, kind, pos, heading).
            let mut snap: Vec<(u32, Kind, VPoint, f64)> = Vec::new();
            tree.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    snap.push((c.id, c.kind, c.pos, c.heading));
                }
            });

            // Movement: each update relocates the item across leaves when it
            // walks out of its cell (split/merge happens live here).
            for &(id, kind, pos, heading) in &snap {
                let (np, nh) = step_critter(kind, pos, heading, dt, &snap, id);
                tree.update(pos, |c| c.id == id, |c| {
                    c.pos = np;
                    c.heading = nh;
                });
            }

            // Post-move snapshot for firing.
            let mut snap2: Vec<(u32, Kind, VPoint, f64)> = Vec::new();
            tree.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    snap2.push((c.id, c.kind, c.pos, c.heading));
                }
            });

            let mut killed: Vec<(u32, VPoint, Kind)> = Vec::new();
            let mut killed_ids: HashSet<u32> = HashSet::new();

            for &(id, kind, pos, _) in &snap2 {
                if killed_ids.contains(&id) {
                    continue; // died earlier this frame
                }
                let (cd_min, cd_max) = kind.cooldown();
                let cd = cooldowns
                    .entry(id)
                    .or_insert_with(|| rand::gen_range(cd_min, cd_max) as f64);
                *cd -= dt;
                if *cd > 0.0 {
                    continue;
                }
                *cd = rand::gen_range(cd_min, cd_max) as f64 / fire_f as f64;

                let attack = match kind {
                    // Aimed: drop pointed at the nearest non-hunter.
                    Kind::Hunter => snap2
                        .iter()
                        .filter(|(tid, k, _, _)| {
                            *tid != id && *k != Kind::Hunter && !killed_ids.contains(tid)
                        })
                        .min_by(|a, b| {
                            let da = (a.2.x - pos.x).powi(2) + (a.2.y - pos.y).powi(2);
                            let db = (b.2.x - pos.x).powi(2) + (b.2.y - pos.y).powi(2);
                            da.partial_cmp(&db).unwrap()
                        })
                        .and_then(|&(_, _, tpos, _)| {
                            make_attack(
                                &arsenal,
                                DROP_ID,
                                pos,
                                Some((tpos.x - pos.x, tpos.y - pos.y)),
                            )
                        }),
                    // Random direction drop.
                    Kind::Drifter => {
                        let a = rand::gen_range(0.0_f32, std::f32::consts::TAU) as f64;
                        make_attack(&arsenal, DROP_ID, pos, Some((a.cos(), a.sin())))
                    }
                    // Circular blast centred on itself.
                    Kind::Pulsar => make_attack(&arsenal, CIRCLE_ID, pos, None),
                };

                if let Some(atk) = attack {
                    for victim in tree.cull(&atk) {
                        if victim.id != id && !killed_ids.contains(&victim.id) {
                            killed_ids.insert(victim.id);
                            killed.push((victim.id, victim.pos, kind));
                        }
                    }
                    effects.push(Effect {
                        outline: sample_polygon(&atk.poly),
                        grid: atk.grid,
                        origin: pos,
                        color: kind.color(),
                        until: now + EFFECT_TTL,
                    });
                }
            }

            for (vid, vpos, attacker) in killed {
                if let Some(c) = tree.remove(vpos, |c| c.id == vid) {
                    cooldowns.remove(&vid);
                    respawns.push((now + respawn_f as f64, c.kind));
                    kills += 1;
                    *kills_by.entry(attacker).or_default() += 1;
                    rings.push(Ring {
                        pos: vpos,
                        color: c.kind.color(),
                        start: now,
                        until: now + 0.6,
                    });
                }
            }

            let mut i = 0;
            while i < respawns.len() {
                if respawns[i].0 <= now {
                    let (_, kind) = respawns.swap_remove(i);
                    let pos = spawn(&mut tree, &mut next_id, kind);
                    rings.push(Ring { pos, color: WHITE, start: now, until: now + 0.5 });
                } else {
                    i += 1;
                }
            }

            effects.retain(|e| e.until > now);
        }
        rings.retain(|r| r.until > now);

        // ---------- live tuning panel ----------
        widgets::Window::new(
            hash!(),
            vec2(MAP_W as f32 + 10.0, 446.0),
            vec2(PANEL_W - 20.0, 280.0),
        )
        .label("tuning (live)")
        .titlebar(true)
        .movable(false)
        .ui(&mut *root_ui(), |ui| {
            ui.slider(hash!(), "split >", 1f32..12f32, &mut split_f);
            ui.slider(hash!(), "merge <=", 1f32..12f32, &mut merge_f);
            ui.separator();
            ui.slider(hash!(), "drifters", 0f32..120f32, &mut drifters_f);
            ui.slider(hash!(), "hunters", 0f32..120f32, &mut hunters_f);
            ui.slider(hash!(), "pulsars", 0f32..120f32, &mut pulsars_f);
            ui.separator();
            ui.slider(hash!(), "respawn s", 0.5f32..10f32, &mut respawn_f);
            ui.slider(hash!(), "speed x", 0.25f32..4f32, &mut speed_f);
            ui.slider(hash!(), "fire x", 0.25f32..3f32, &mut fire_f);
        });

        // Apply split/merge thresholds; rebuild the tree when they change so
        // the whole structure obeys the new rules immediately.
        split_f = split_f.clamp(1.0, 12.0);
        merge_f = merge_f.clamp(1.0, split_f);
        let want_split = split_f.round() as usize;
        let want_merge = (merge_f.round() as usize).min(want_split);
        if want_split != tree.item_limit || want_merge != tree.merge_limit {
            let mut items: Vec<Critter> = Vec::new();
            tree.visit_leaves(|_, leaf| items.extend(leaf.items.iter().cloned()));
            let mut rebuilt = Tree::with_limits(world, want_split, want_merge);
            for c in items {
                rebuilt.insert(c);
            }
            tree = rebuilt;
        }

        // Steer populations toward the slider targets (counting queued
        // respawns so deaths don't double-spawn).
        let mut alive: HashMap<Kind, Vec<(u32, VPoint)>> = HashMap::new();
        tree.visit_leaves(|_, leaf| {
            for c in &leaf.items {
                alive.entry(c.kind).or_default().push((c.id, c.pos));
            }
        });
        for (kind, target_f) in [
            (Kind::Drifter, drifters_f),
            (Kind::Hunter, hunters_f),
            (Kind::Pulsar, pulsars_f),
        ] {
            let target = target_f.round() as i64;
            let queued = respawns.iter().filter(|(_, k)| *k == kind).count() as i64;
            let live = alive.get(&kind).map_or(0, |v| v.len()) as i64;
            let mut diff = target - live - queued;
            while diff > 0 {
                let pos = spawn(&mut tree, &mut next_id, kind);
                rings.push(Ring { pos, color: kind.color(), start: now, until: now + 0.4 });
                diff -= 1;
            }
            if diff < 0 {
                let mut to_remove = -diff;
                // Cancel queued respawns first (invisible), then remove live ones.
                let mut i = 0;
                while i < respawns.len() && to_remove > 0 {
                    if respawns[i].1 == kind {
                        respawns.swap_remove(i);
                        to_remove -= 1;
                    } else {
                        i += 1;
                    }
                }
                if let Some(list) = alive.get_mut(&kind) {
                    while to_remove > 0 && !list.is_empty() {
                        let idx = rand::gen_range(0, list.len());
                        let (id, pos) = list.swap_remove(idx);
                        if tree.remove(pos, |c| c.id == id).is_some() {
                            cooldowns.remove(&id);
                            rings.push(Ring {
                                pos,
                                color: kind.color(),
                                start: now,
                                until: now + 0.4,
                            });
                        }
                        to_remove -= 1;
                    }
                }
            }
        }

        // ---------- draw ----------
        clear_background(Color::new(0.07, 0.07, 0.10, 1.0));

        // Tree regions: one colour per live leaf.
        let mut leaves = 0usize;
        let mut items = 0usize;
        tree.visit_leaves(|id, leaf| {
            leaves += 1;
            items += leaf.items.len();
            let b = leaf.bbox;
            let c = region_color(id.0);
            if region_mode == 0 {
                draw_rectangle(
                    b.x as f32,
                    b.y as f32,
                    b.width as f32,
                    b.height as f32,
                    Color::new(c.r, c.g, c.b, 0.28),
                );
            }
            if region_mode <= 1 {
                draw_rectangle_lines(
                    b.x as f32,
                    b.y as f32,
                    b.width as f32,
                    b.height as f32,
                    1.0,
                    Color::new(1.0, 1.0, 1.0, 0.18),
                );
            }
        });

        // Attack effects: the template cells themselves (green = In, dimmer =
        // Maybe), fading out.
        for e in &effects {
            let fade = ((e.until - now) / EFFECT_TTL) as f32;
            for row in 0..e.grid.rows {
                for col in 0..e.grid.cols {
                    let alpha = match e.grid.cell(col, row) {
                        CellState::In => 0.5 * fade,
                        CellState::Maybe => 0.22 * fade,
                        CellState::Out => continue,
                    };
                    draw_rectangle(
                        (e.grid.origin_x + col as f64 * e.grid.cell_w) as f32,
                        (e.grid.origin_y + row as f64 * e.grid.cell_h) as f32,
                        e.grid.cell_w as f32,
                        e.grid.cell_h as f32,
                        Color::new(e.color.r, e.color.g, e.color.b, alpha),
                    );
                }
            }
            // The real attack polygon on top of its template cells.
            let oc = Color::new(e.color.r, e.color.g, e.color.b, (0.9 * fade).min(1.0));
            for i in 0..e.outline.len() {
                let (x1, y1) = e.outline[i];
                let (x2, y2) = e.outline[(i + 1) % e.outline.len()];
                draw_line(x1, y1, x2, y2, 1.5, oc);
            }
            draw_circle(e.origin.x as f32, e.origin.y as f32, 3.0, e.color);
        }

        // Critters (single snapshot reused for hunter sightlines + counts).
        let mut draw_snap: Vec<(u32, Kind, VPoint, f64)> = Vec::new();
        tree.visit_leaves(|_, leaf| {
            for c in &leaf.items {
                draw_snap.push((c.id, c.kind, c.pos, c.heading));
            }
        });

        // Hunter sightlines to their current prey.
        for &(id, kind, pos, _) in &draw_snap {
            if kind != Kind::Hunter {
                continue;
            }
            let target = draw_snap
                .iter()
                .filter(|(tid, k, _, _)| *tid != id && *k != Kind::Hunter)
                .min_by(|a, b| {
                    let da = (a.2.x - pos.x).powi(2) + (a.2.y - pos.y).powi(2);
                    let db = (b.2.x - pos.x).powi(2) + (b.2.y - pos.y).powi(2);
                    da.partial_cmp(&db).unwrap()
                });
            if let Some(&(_, _, tpos, _)) = target {
                draw_line(
                    pos.x as f32,
                    pos.y as f32,
                    tpos.x as f32,
                    tpos.y as f32,
                    1.0,
                    Color::new(1.0, 0.3, 0.3, 0.18),
                );
            }
        }

        for &(_, kind, pos, heading) in &draw_snap {
            let (x, y) = (pos.x as f32, pos.y as f32);
            draw_circle(x, y, 5.0, kind.color());
            draw_line(
                x,
                y,
                x + (heading.cos() * 9.0) as f32,
                y + (heading.sin() * 9.0) as f32,
                1.5,
                WHITE,
            );
        }

        // Spawn / death rings.
        for r in &rings {
            let t = ((now - r.start) / (r.until - r.start).max(1e-6)) as f32;
            draw_circle_lines(
                r.pos.x as f32,
                r.pos.y as f32,
                4.0 + 16.0 * t,
                2.0,
                Color::new(r.color.r, r.color.g, r.color.b, (1.0 - t).max(0.0)),
            );
        }

        // Map border + side panel.
        draw_rectangle_lines(0.0, 0.0, MAP_W as f32, MAP_H as f32, 2.0, GRAY);
        let px = MAP_W as f32 + 14.0;
        draw_rectangle(MAP_W as f32, 0.0, PANEL_W, MAP_H as f32, Color::new(0.04, 0.04, 0.06, 1.0));
        let mut ty = 28.0;
        let line = |text: &str, size: f32, color: Color, ty: &mut f32| {
            draw_text(text, px, *ty, size, color);
            *ty += size * 1.15;
        };
        line("vectorial-hash critters", 26.0, WHITE, &mut ty);
        ty += 8.0;
        line(
            &format!("template set: {} combos", arsenal.combos),
            18.0,
            LIGHTGRAY,
            &mut ty,
        );
        line(
            &format!("  -> {} unique after 8-sym dedup", arsenal.unique),
            18.0,
            LIGHTGRAY,
            &mut ty,
        );
        line(
            &format!("  index entries: {}", arsenal.index.len()),
            18.0,
            LIGHTGRAY,
            &mut ty,
        );
        ty += 10.0;
        let mut counts: HashMap<Kind, usize> = HashMap::new();
        for &(_, kind, _, _) in &draw_snap {
            *counts.entry(kind).or_default() += 1;
        }
        for kind in [Kind::Drifter, Kind::Hunter, Kind::Pulsar] {
            draw_circle(px + 6.0, ty - 5.0, 5.0, kind.color());
            let what = match kind {
                Kind::Drifter => "random walk, random drop shot",
                Kind::Hunter => "chases prey, aimed drop shot",
                Kind::Pulsar => "circles, radial blast",
            };
            line(
                &format!("    {} x{}: {}", kind.name(), counts.get(&kind).copied().unwrap_or(0), what),
                16.0,
                WHITE,
                &mut ty,
            );
        }
        ty += 10.0;
        line(&format!("alive: {}   kills: {}", items, kills), 18.0, WHITE, &mut ty);
        line(
            &format!(
                "  by: drifter {} / hunter {} / pulsar {}",
                kills_by.get(&Kind::Drifter).copied().unwrap_or(0),
                kills_by.get(&Kind::Hunter).copied().unwrap_or(0),
                kills_by.get(&Kind::Pulsar).copied().unwrap_or(0),
            ),
            16.0,
            LIGHTGRAY,
            &mut ty,
        );
        line(
            &format!("respawning: {}", respawns.len()),
            18.0,
            WHITE,
            &mut ty,
        );
        ty += 10.0;
        line(
            &format!(
                "tree leaves: {}   (split >{}, merge <={})",
                leaves, tree.item_limit, tree.merge_limit,
            ),
            18.0,
            SKYBLUE,
            &mut ty,
        );
        line(
            &format!("arena nodes: {} (incl. orphans)", tree.node_count()),
            18.0,
            SKYBLUE,
            &mut ty,
        );
        ty += 10.0;
        draw_circle(px + 6.0, ty - 5.0, 5.0, brush.color());
        line(
            &format!("    brush: {}  (keys 1/2/3)", brush.name()),
            18.0,
            WHITE,
            &mut ty,
        );
        line(
            &format!("speed: {:.2}x   regions: {}", speed_f, match region_mode {
                0 => "fill+lines",
                1 => "lines",
                _ => "off",
            }),
            18.0,
            WHITE,
            &mut ty,
        );
        ty += 10.0;
        line(&format!("fps: {}", get_fps()), 18.0, GREEN, &mut ty);
        if paused {
            line("PAUSED", 22.0, ORANGE, &mut ty);
        }
        let help = [
            "[LMB] spawn brush   [RMB] remove",
            "[+/-] spawn/remove 5   [R] regions",
            "[ / ] speed   [Space] pause   [Esc] quit",
        ];
        for (i, h) in help.iter().enumerate() {
            draw_text(
                h,
                px,
                MAP_H as f32 - 16.0 - 20.0 * (help.len() - 1 - i) as f32,
                17.0,
                GRAY,
            );
        }

        frame += 1;
        if max_frames > 0 && frame >= max_frames {
            break;
        }
        next_frame().await
    }
}
