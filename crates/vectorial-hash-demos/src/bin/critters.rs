//! Visual critters demo — precise template application, dual structures.
//!
//! A 2D map indexed by the binary-split [`Tree`], the reference
//! [`QuadTree`], or **both at once** (key `M` cycles). In dual mode every
//! operation — insert, remove, update, attack cull, vision cull — runs on
//! both structures with identical inputs, their cull results are checked
//! for agreement live, and the per-operation timings are plotted side by
//! side in the graphs column.
//!
//! Critters attack using precomputed template areas served from a
//! hierarchical [`TemplateBank`] (shape → dims → cell width → cell height →
//! angle → offset x → offset y), generated at startup. The attack figure is
//! never moved to fit a grid: it is applied at its real (integer) origin
//! and the bank serves, per tree-cell size, the template whose generation
//! offset matches — template cells align 1:1 with the map's cells. Leaf
//! items resolve with a 1×1 raster (exact geometry only on boundary
//! pixels).
//!
//! Run: `cargo run -p vectorial-hash-demos --bin critters --release`
//!
//! Controls:
//! - `M` cycles the spatial structure: binary tree / quadtree / both
//! - `1`/`2`/`3` select the spawn brush (drifter / hunter / pulsar)
//! - left click (hold to paint) spawns at the cursor, right click removes
//! - `+`/`-` spawn / remove five critters at random
//! - `R` cycles region rendering (fill+lines / lines / off)
//! - `[` / `]` halve / double simulation speed, `Space` pauses, `Esc` quits
//! - the "tuning (live)" panel adjusts split/merge thresholds (structures
//!   rebuilt on change — both at once in dual mode), per-kind population
//!   targets (up to 1200 each), respawn delay, speed and fire rate
//!
//! The whole simulation runs on ONE thread (only the startup bank
//! generation parallelizes); the graphs show where that thread's time goes.
//!
//! Env: CRITTERS_MAX_FRAMES=N exits after N frames (smoke testing).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets};

use vectorial_hash::{
    CellState, PlacedTemplate, Point as VPoint, Positioned, QuadTree, Rect as VRect, Shape,
    TemplateGrid, Tree,
};
use vectorial_hash_templates::bank::{FigureKey, TemplateBank};
use vectorial_hash_templates::polygon::{create_circle, create_drop, rotated_copy, scaled_copy, Polygon};
use vectorial_hash_templates::templates::angle_to_radians;

/// World size (map units). Power of two so binary splits produce integer
/// cell sizes aligned with the global virtual grids.
const MAP_W: f64 = 1024.0;
const MAP_H: f64 = 1024.0;
/// Render scale: world 1024 drawn at 768 px.
const S: f32 = 0.75;
const PANEL_W: f32 = 320.0;
/// Right-hand column with live performance graphs.
const GRAPH_W: f32 = 310.0;
const MARGIN: f64 = 4.0;
/// Hunters acquire prey with a cull of this vision radius.
const VISION_RADIUS: f64 = 280.0;
/// Hard cap on the total population.
const MAX_CRITTERS: usize = 4000;

const ANGLE_STEP_DEG: f64 = 15.0;
const DROP_ID: u32 = 0;
const CIRCLE_ID: u32 = 1;
const DROP_SCALE: f64 = 110.0;
const CIRCLE_RADIUS: f64 = 48.0;

/// Cell sizes (w, h) with full template sets. Tree cells of other sizes fall
/// back to bbox recursion (internal nodes) or the 1×1 raster (leaf items).
const TEMPLATE_SIZES: [(u32, u32); 7] =
    [(8, 8), (16, 16), (32, 32), (8, 16), (16, 8), (16, 32), (32, 16)];

const ITEM_LIMIT: usize = 3;
const RESPAWN_DELAY: f64 = 2.5;
const EFFECT_TTL: f64 = 0.45;

const COL_BIN: Color = ORANGE;
const COL_QUAD: Color = SKYBLUE;

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
            Kind::Drifter => 75.0,
            Kind::Hunter => 105.0,
            Kind::Pulsar => 88.0,
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

// ---------------------------------------------------------------- structures

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Binary,
    Quad,
    Both,
}

impl Mode {
    fn next(self) -> Mode {
        match self {
            Mode::Binary => Mode::Quad,
            Mode::Quad => Mode::Both,
            Mode::Both => Mode::Binary,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Mode::Binary => "binary tree",
            Mode::Quad => "quadtree",
            Mode::Both => "both (compare)",
        }
    }
}

/// Per-frame accumulated timings for one structure, in microseconds.
#[derive(Default, Clone, Copy)]
struct OpStats {
    mv: f64,
    atk: f64,
    atk_n: u32,
    vis: f64,
    vis_n: u32,
    rm: f64,
}

/// Owns whichever structures the current mode requires and applies every
/// operation to all of them with identical inputs, timing each separately.
struct Sims {
    tree: Option<Tree<Critter>>,
    quad: Option<QuadTree<Critter>>,
    t: OpStats,
    q: OpStats,
    mismatches: u64,
}

impl Sims {
    fn new(mode: Mode, world: VRect, split: usize, merge: usize) -> Self {
        let mut s = Sims { tree: None, quad: None, t: OpStats::default(), q: OpStats::default(), mismatches: 0 };
        s.apply_mode(mode, world, split, merge, &[]);
        s
    }

    fn mode(&self) -> Mode {
        match (&self.tree, &self.quad) {
            (Some(_), Some(_)) => Mode::Both,
            (Some(_), None) => Mode::Binary,
            _ => Mode::Quad,
        }
    }

    /// Rebuild for `mode` (and limits) from the given items.
    fn apply_mode(&mut self, mode: Mode, world: VRect, split: usize, merge: usize, items: &[Critter]) {
        self.tree = matches!(mode, Mode::Binary | Mode::Both).then(|| {
            let mut t = Tree::with_limits(world, split, merge);
            for c in items {
                t.insert(c.clone());
            }
            t
        });
        self.quad = matches!(mode, Mode::Quad | Mode::Both).then(|| {
            let mut q = QuadTree::with_limits(world, split, merge);
            for c in items {
                q.insert(c.clone());
            }
            q
        });
    }

    fn begin_frame(&mut self) {
        self.t = OpStats::default();
        self.q = OpStats::default();
    }

    fn snapshot(&self) -> Vec<(u32, Kind, VPoint, f64)> {
        let mut out = Vec::new();
        if let Some(t) = &self.tree {
            t.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    out.push((c.id, c.kind, c.pos, c.heading));
                }
            });
        } else if let Some(q) = &self.quad {
            q.visit_leaves(|_, leaf| {
                for c in &leaf.items {
                    out.push((c.id, c.kind, c.pos, c.heading));
                }
            });
        }
        out
    }

    fn item_count(&self) -> usize {
        if let Some(t) = &self.tree {
            t.item_count()
        } else if let Some(q) = &self.quad {
            q.item_count()
        } else {
            0
        }
    }

    fn insert(&mut self, c: &Critter) {
        if let Some(t) = &mut self.tree {
            let s = Instant::now();
            t.insert(c.clone());
            self.t.rm += s.elapsed().as_secs_f64() * 1e6;
        }
        if let Some(q) = &mut self.quad {
            let s = Instant::now();
            q.insert(c.clone());
            self.q.rm += s.elapsed().as_secs_f64() * 1e6;
        }
    }

    fn remove(&mut self, pos: VPoint, id: u32) -> Option<Critter> {
        let mut out = None;
        if let Some(t) = &mut self.tree {
            let s = Instant::now();
            out = t.remove(pos, |c| c.id == id);
            self.t.rm += s.elapsed().as_secs_f64() * 1e6;
        }
        if let Some(q) = &mut self.quad {
            let s = Instant::now();
            let r = q.remove(pos, |c| c.id == id);
            self.q.rm += s.elapsed().as_secs_f64() * 1e6;
            if out.is_none() {
                out = r;
            }
        }
        out
    }

    fn update_critter(&mut self, pos: VPoint, id: u32, np: VPoint, nh: f64) {
        if let Some(t) = &mut self.tree {
            let s = Instant::now();
            t.update(pos, |c| c.id == id, |c| {
                c.pos = np;
                c.heading = nh;
            });
            self.t.mv += s.elapsed().as_secs_f64() * 1e6;
        }
        if let Some(q) = &mut self.quad {
            let s = Instant::now();
            q.update(pos, |c| c.id == id, |c| {
                c.pos = np;
                c.heading = nh;
            });
            self.q.mv += s.elapsed().as_secs_f64() * 1e6;
        }
    }

    /// Attack cull: returns (id, pos) victims from the primary structure;
    /// in dual mode also runs on the other and counts disagreements.
    fn cull_attack<Sh: Shape>(&mut self, shape: &Sh) -> Vec<(u32, VPoint)> {
        let mut tree_ids: Option<Vec<(u32, VPoint)>> = None;
        let mut quad_ids: Option<Vec<(u32, VPoint)>> = None;
        if let Some(t) = &self.tree {
            let s = Instant::now();
            let hits = t.cull(shape);
            self.t.atk += s.elapsed().as_secs_f64() * 1e6;
            self.t.atk_n += 1;
            tree_ids = Some(hits.iter().map(|c| (c.id, c.pos)).collect());
        }
        if let Some(q) = &self.quad {
            let s = Instant::now();
            let hits = q.cull(shape);
            self.q.atk += s.elapsed().as_secs_f64() * 1e6;
            self.q.atk_n += 1;
            quad_ids = Some(hits.iter().map(|c| (c.id, c.pos)).collect());
        }
        if let (Some(a), Some(b)) = (&tree_ids, &quad_ids) {
            let sa: HashSet<u32> = a.iter().map(|(id, _)| *id).collect();
            let sb: HashSet<u32> = b.iter().map(|(id, _)| *id).collect();
            if sa != sb {
                self.mismatches += 1;
            }
        }
        tree_ids.or(quad_ids).unwrap_or_default()
    }

    /// Vision cull: nearest non-hunter within range, from the primary.
    fn vision_prey(&mut self, pos: VPoint, self_id: u32) -> Option<VPoint> {
        let vision = VisionCircle { center: pos, r: VISION_RADIUS };
        let nearest = |hits: Vec<&Critter>| {
            hits.into_iter()
                .filter(|c| c.id != self_id && c.kind != Kind::Hunter)
                .min_by(|a, b| {
                    let da = (a.pos.x - pos.x).powi(2) + (a.pos.y - pos.y).powi(2);
                    let db = (b.pos.x - pos.x).powi(2) + (b.pos.y - pos.y).powi(2);
                    da.partial_cmp(&db).unwrap()
                })
                .map(|c| (c.id, c.pos))
        };
        let mut from_tree = None;
        let mut from_quad = None;
        if let Some(t) = &self.tree {
            let s = Instant::now();
            from_tree = nearest(t.cull(&vision));
            self.t.vis += s.elapsed().as_secs_f64() * 1e6;
            self.t.vis_n += 1;
        }
        if let Some(q) = &self.quad {
            let s = Instant::now();
            from_quad = nearest(q.cull(&vision));
            self.q.vis += s.elapsed().as_secs_f64() * 1e6;
            self.q.vis_n += 1;
        }
        if let (Some((ia, _)), Some((ib, _))) = (from_tree, from_quad) {
            if ia != ib {
                self.mismatches += 1;
            }
        }
        from_tree.or(from_quad).map(|(_, p)| p)
    }
}

/// Hunter vision: a plain circle shape.
struct VisionCircle {
    center: VPoint,
    r: f64,
}

impl Shape for VisionCircle {
    fn bounding_box(&self) -> VRect {
        VRect::new(
            self.center.x - self.r,
            self.center.y - self.r,
            self.r * 2.0,
            self.r * 2.0,
        )
    }
    fn contains_point(&self, p: VPoint) -> bool {
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;
        dx * dx + dy * dy <= self.r * self.r
    }
}

// ------------------------------------------------------------------ arsenal

/// The startup-generated template bank plus per-figure metadata.
struct Arsenal {
    bank: TemplateBank,
    figures: HashMap<u32, FigureKey>,
    base_polys: HashMap<(u32, i64), Polygon>,
    gen_seconds: f64,
}

fn build_arsenal() -> Arsenal {
    let start = Instant::now();
    let mut bank = TemplateBank::new();
    let mut figures = HashMap::new();
    let mut base_polys = HashMap::new();

    let drop_angles: Vec<f64> = (0..(360.0 / ANGLE_STEP_DEG) as i64)
        .map(|i| i as f64 * ANGLE_STEP_DEG)
        .collect();

    let shapes: [(u32, Vec<f64>, Polygon, Vec<f64>); 2] = [
        (
            DROP_ID,
            vec![0.2 * DROP_SCALE, 0.8 * DROP_SCALE],
            scaled_copy(&create_drop(0.2, 0.8), DROP_SCALE, DROP_SCALE),
            drop_angles,
        ),
        (
            CIRCLE_ID,
            vec![CIRCLE_RADIUS],
            scaled_copy(&create_circle(1.0), CIRCLE_RADIUS, CIRCLE_RADIUS),
            vec![0.0],
        ),
    ];

    for (shape_id, dims, base, angles) in shapes {
        let figure = FigureKey::new(shape_id, &dims);
        for &(cw, ch) in &TEMPLATE_SIZES {
            bank.generate_size(&figure, &base, &angles, cw, ch);
        }
        bank.generate_size(&figure, &base, &angles, 1, 1);
        for &angle in &angles {
            base_polys.insert((shape_id, angle as i64), rotated_copy(&base, angle_to_radians(angle)));
        }
        figures.insert(shape_id, figure);
    }

    Arsenal {
        bank,
        figures,
        base_polys,
        gen_seconds: start.elapsed().as_secs_f64(),
    }
}

/// Attack area applied at its real integer origin.
struct AttackShape<'a> {
    bank: &'a TemplateBank,
    figure: FigureKey,
    angle_deg: f64,
    origin: (i64, i64),
    poly: Polygon,
    bbox: VRect,
    raster: Option<PlacedTemplate>,
}

impl Shape for AttackShape<'_> {
    fn bounding_box(&self) -> VRect {
        self.bbox
    }
    fn contains_point(&self, p: VPoint) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
        if cell_w.fract() != 0.0 || cell_h.fract() != 0.0 {
            return None;
        }
        self.bank.placed_for(
            &self.figure,
            cell_w as u32,
            cell_h as u32,
            self.angle_deg,
            self.origin,
        )
    }
    fn point_template(&self) -> Option<&PlacedTemplate> {
        self.raster.as_ref()
    }
}

fn make_attack<'a>(
    arsenal: &'a Arsenal,
    shape_id: u32,
    pos: VPoint,
    aim: Option<(f64, f64)>,
) -> Option<AttackShape<'a>> {
    let angle_deg = match aim {
        Some((dx, dy)) => {
            let a = (-dx).atan2(dy).to_degrees();
            ((a / ANGLE_STEP_DEG).round() * ANGLE_STEP_DEG).rem_euclid(360.0)
        }
        None => 0.0,
    };
    let origin = (pos.x.round() as i64, pos.y.round() as i64);

    let figure = arsenal.figures.get(&shape_id)?.clone();
    let mut poly = arsenal.base_polys.get(&(shape_id, angle_deg as i64))?.clone();
    poly.move_by(origin.0 as f64, origin.1 as f64);
    let bbox = VRect::new(
        poly.x_min,
        poly.y_min,
        poly.x_max - poly.x_min,
        poly.y_max - poly.y_min,
    );
    let raster = arsenal.bank.placed_raster(&figure, angle_deg, origin);

    Some(AttackShape { bank: &arsenal.bank, figure, angle_deg, origin, poly, bbox, raster })
}

// ------------------------------------------------------------------ visuals

struct Effect {
    grid: Option<TemplateGrid>,
    outline: Vec<(f32, f32)>,
    origin: VPoint,
    color: Color,
    until: f64,
}

struct Ring {
    pos: VPoint,
    color: Color,
    start: f64,
    until: f64,
}

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

fn make_critter(next_id: &mut u32, kind: Kind, pos: VPoint) -> Critter {
    let id = *next_id;
    *next_id += 1;
    let heading = rand::gen_range(0.0_f32, std::f32::consts::TAU) as f64;
    Critter { id, kind, pos, heading }
}

fn random_pos() -> VPoint {
    VPoint::new(
        rand::gen_range(20.0_f32, (MAP_W - 20.0) as f32) as f64,
        rand::gen_range(20.0_f32, (MAP_H - 20.0) as f32) as f64,
    )
}

/// One movement step. Returns (new position, new heading).
fn step_critter(kind: Kind, pos: VPoint, heading: f64, dt: f64, target: Option<VPoint>) -> (VPoint, f64) {
    let mut h = heading;
    match kind {
        Kind::Drifter => h += rand::gen_range(-2.5_f32, 2.5) as f64 * dt,
        Kind::Pulsar => h += 1.6 * dt,
        Kind::Hunter => match target {
            Some(tpos) => {
                let desired = (tpos.y - pos.y).atan2(tpos.x - pos.x);
                let diff = (desired - h + std::f64::consts::PI)
                    .rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI;
                h += diff.clamp(-3.0 * dt, 3.0 * dt);
            }
            None => h += rand::gen_range(-2.0_f32, 2.0) as f64 * dt,
        },
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

/// Stable distinct colour per node id (golden-ratio hue walk).
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

/// Fixed-capacity rolling series for the live graphs.
struct Series {
    data: Vec<f32>,
    cap: usize,
}

impl Series {
    fn new(cap: usize) -> Self {
        Self { data: Vec::with_capacity(cap), cap }
    }
    fn push(&mut self, v: f32) {
        if self.data.len() == self.cap {
            self.data.remove(0);
        }
        self.data.push(v);
    }
    fn last(&self) -> f32 {
        self.data.last().copied().unwrap_or(0.0)
    }
}

/// Autoscaled graph with up to two series sharing one scale.
fn draw_graph(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    label: &str,
    unit: &str,
    series: &[(&Series, Color, &str)],
) {
    draw_rectangle(x, y, w, h, Color::new(0.02, 0.02, 0.045, 1.0));
    draw_rectangle_lines(x, y, w, h, 1.0, Color::new(1.0, 1.0, 1.0, 0.15));
    let max = series
        .iter()
        .flat_map(|(s, _, _)| s.data.iter().cloned())
        .fold(0.0_f32, f32::max)
        .max(1e-6)
        * 1.1;
    for (s, color, _) in series {
        if s.data.len() < 2 {
            continue;
        }
        let plot_h = h - 26.0;
        let step = w / (s.cap.max(2) - 1) as f32;
        for i in 1..s.data.len() {
            let x1 = x + (i - 1) as f32 * step;
            let x2 = x + i as f32 * step;
            let y1 = y + h - 4.0 - (s.data[i - 1] / max) * plot_h;
            let y2 = y + h - 4.0 - (s.data[i] / max) * plot_h;
            draw_line(x1, y1, x2, y2, 1.5, *color);
        }
    }
    let mut header = format!("{label}:");
    for (s, _, name) in series {
        header.push_str(&format!(" {name} {:.2}", s.last()));
    }
    header.push_str(unit);
    draw_text(&header, x + 6.0, y + 16.0, 15.0, WHITE);
}

fn window_conf() -> Conf {
    Conf {
        window_title: "vectorial-hash critters".to_owned(),
        window_width: (MAP_W as f32 * S) as i32 + PANEL_W as i32 + GRAPH_W as i32,
        window_height: (MAP_H as f32 * S) as i32,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    rand::srand(42);

    clear_background(Color::new(0.07, 0.07, 0.10, 1.0));
    draw_text("generating template bank...", 40.0, 60.0, 28.0, WHITE);
    next_frame().await;

    let arsenal = build_arsenal();
    let mem = arsenal.bank.memory_usage();
    println!(
        "template bank: {} combos -> {} unique grids in {:.2}s",
        arsenal.bank.entry_count(),
        arsenal.bank.unique_count(),
        arsenal.gen_seconds,
    );
    println!(
        "bank memory: {:.2} MB total (grids {:.2} MB, index {:.2} MB, dedup keys {:.2} MB)",
        mem.total() as f64 / 1e6,
        mem.grids_bytes as f64 / 1e6,
        mem.index_bytes as f64 / 1e6,
        mem.dedup_keys_bytes as f64 / 1e6,
    );

    let world = VRect::new(0.0, 0.0, MAP_W, MAP_H);
    let mut sims = Sims::new(Mode::Binary, world, ITEM_LIMIT, ITEM_LIMIT);
    let mut next_id = 0u32;
    let mut cooldowns: HashMap<u32, f64> = HashMap::new();

    // Live-tunable settings.
    let mut split_f: f32 = ITEM_LIMIT as f32;
    let mut merge_f: f32 = ITEM_LIMIT as f32;
    let mut drifters_f: f32 = 16.0;
    let mut hunters_f: f32 = 12.0;
    let mut pulsars_f: f32 = 14.0;
    let mut respawn_f: f32 = RESPAWN_DELAY as f32;
    let mut speed_f: f32 = 1.0;
    let mut fire_f: f32 = 1.0;

    for _ in 0..16 {
        let c = make_critter(&mut next_id, Kind::Drifter, random_pos());
        sims.insert(&c);
    }
    for _ in 0..12 {
        let c = make_critter(&mut next_id, Kind::Hunter, random_pos());
        sims.insert(&c);
    }
    for _ in 0..14 {
        let c = make_critter(&mut next_id, Kind::Pulsar, random_pos());
        sims.insert(&c);
    }

    let mut respawns: Vec<(f64, Kind)> = Vec::new();
    let mut effects: Vec<Effect> = Vec::new();
    let mut rings: Vec<Ring> = Vec::new();
    let mut kills: u64 = 0;
    let mut kills_by: HashMap<Kind, u64> = HashMap::new();
    let mut last_paint: f64 = 0.0;
    let mut paused = false;
    let mut brush = Kind::Hunter;
    let mut region_mode: u8 = 0;

    let max_frames: u64 = std::env::var("CRITTERS_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut frame: u64 = 0;

    // Graph series: frame (global) + per-structure pairs.
    let cap = 240;
    let mut g_frame = Series::new(cap);
    let (mut g_atk_t, mut g_atk_q) = (Series::new(cap), Series::new(cap));
    let (mut g_vis_t, mut g_vis_q) = (Series::new(cap), Series::new(cap));
    let (mut g_mv_t, mut g_mv_q) = (Series::new(cap), Series::new(cap));
    let (mut g_rm_t, mut g_rm_q) = (Series::new(cap), Series::new(cap));
    let mut sightlines: Vec<(VPoint, VPoint)> = Vec::new();

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::M) {
            let items: Vec<Critter> = {
                let snap = sims.snapshot();
                snap.into_iter()
                    .map(|(id, kind, pos, heading)| Critter { id, kind, pos, heading })
                    .collect()
            };
            let new_mode = sims.mode().next();
            sims.apply_mode(
                new_mode,
                world,
                split_f.round() as usize,
                (merge_f.round() as usize).min(split_f.round() as usize),
                &items,
            );
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
        sims.begin_frame();

        // --- interactive add / remove ---
        let (mx, my) = mouse_position();
        let (wx, wy) = ((mx / S) as f64, (my / S) as f64);
        let mouse_in_map = wx >= 0.0 && wx < MAP_W && wy >= 0.0 && wy < MAP_H && mx < MAP_W as f32 * S;
        if is_mouse_button_down(MouseButton::Left)
            && mouse_in_map
            && sims.item_count() < MAX_CRITTERS
            && now - last_paint > 0.12
        {
            last_paint = now;
            let c = make_critter(&mut next_id, brush, VPoint::new(wx, wy));
            rings.push(Ring { pos: c.pos, color: brush.color(), start: now, until: now + 0.4 });
            sims.insert(&c);
            match brush {
                Kind::Drifter => drifters_f += 1.0,
                Kind::Hunter => hunters_f += 1.0,
                Kind::Pulsar => pulsars_f += 1.0,
            }
        }
        if is_mouse_button_pressed(MouseButton::Right) && mouse_in_map {
            let snap = sims.snapshot();
            let best = snap
                .iter()
                .map(|&(id, kind, pos, _)| {
                    let d = (pos.x - wx).powi(2) + (pos.y - wy).powi(2);
                    (d, id, pos, kind)
                })
                .filter(|(d, ..)| *d < 30.0 * 30.0)
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            if let Some((_, id, pos, kind)) = best {
                sims.remove(pos, id);
                cooldowns.remove(&id);
                rings.push(Ring { pos, color: kind.color(), start: now, until: now + 0.4 });
                match kind {
                    Kind::Drifter => drifters_f = (drifters_f - 1.0).max(0.0),
                    Kind::Hunter => hunters_f = (hunters_f - 1.0).max(0.0),
                    Kind::Pulsar => pulsars_f = (pulsars_f - 1.0).max(0.0),
                }
            }
        }
        if is_key_pressed(KeyCode::Equal) && sims.item_count() < MAX_CRITTERS {
            for _ in 0..5 {
                let c = make_critter(&mut next_id, brush, random_pos());
                rings.push(Ring { pos: c.pos, color: brush.color(), start: now, until: now + 0.4 });
                sims.insert(&c);
                match brush {
                    Kind::Drifter => drifters_f += 1.0,
                    Kind::Hunter => hunters_f += 1.0,
                    Kind::Pulsar => pulsars_f += 1.0,
                }
            }
        }
        if is_key_pressed(KeyCode::Minus) {
            let mut all = sims.snapshot();
            for _ in 0..5 {
                if all.is_empty() {
                    break;
                }
                let i = rand::gen_range(0, all.len());
                let (id, kind, pos, _) = all.swap_remove(i);
                sims.remove(pos, id);
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
            let snap = sims.snapshot();

            // Movement; hunters acquire prey with a vision cull (on every
            // structure present — timings recorded per structure).
            sightlines.clear();
            for &(id, kind, pos, heading) in &snap {
                let target = if kind == Kind::Hunter {
                    let prey = sims.vision_prey(pos, id);
                    if let Some(t) = prey {
                        sightlines.push((pos, t));
                    }
                    prey
                } else {
                    None
                };
                let (np, nh) = step_critter(kind, pos, heading, dt, target);
                sims.update_critter(pos, id, np, nh);
            }

            let snap2 = sims.snapshot();
            let mut killed: Vec<(u32, VPoint, Kind, Kind)> = Vec::new();
            let mut killed_ids: HashSet<u32> = HashSet::new();
            let kind_of: HashMap<u32, Kind> = snap2.iter().map(|&(id, k, _, _)| (id, k)).collect();

            for &(id, kind, pos, _) in &snap2 {
                if killed_ids.contains(&id) {
                    continue;
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
                    Kind::Hunter => sims.vision_prey(pos, id).and_then(|tpos| {
                        make_attack(&arsenal, DROP_ID, pos, Some((tpos.x - pos.x, tpos.y - pos.y)))
                    }),
                    Kind::Drifter => {
                        let a = rand::gen_range(0.0_f32, std::f32::consts::TAU) as f64;
                        make_attack(&arsenal, DROP_ID, pos, Some((a.cos(), a.sin())))
                    }
                    Kind::Pulsar => make_attack(&arsenal, CIRCLE_ID, pos, None),
                };

                if let Some(atk) = attack {
                    for (vid, vpos) in sims.cull_attack(&atk) {
                        if vid != id && !killed_ids.contains(&vid) {
                            killed_ids.insert(vid);
                            let vkind = kind_of.get(&vid).copied().unwrap_or(Kind::Drifter);
                            killed.push((vid, vpos, vkind, kind));
                        }
                    }
                    effects.push(Effect {
                        grid: arsenal.bank.template_for(&atk.figure, 16, 16, atk.angle_deg, atk.origin),
                        outline: sample_polygon(&atk.poly),
                        origin: pos,
                        color: kind.color(),
                        until: now + EFFECT_TTL,
                    });
                }
            }

            for (vid, vpos, vkind, attacker) in killed {
                if sims.remove(vpos, vid).is_some() {
                    cooldowns.remove(&vid);
                    respawns.push((now + respawn_f as f64, vkind));
                    kills += 1;
                    *kills_by.entry(attacker).or_default() += 1;
                    rings.push(Ring { pos: vpos, color: vkind.color(), start: now, until: now + 0.6 });
                }
            }

            let mut i = 0;
            while i < respawns.len() {
                if respawns[i].0 <= now {
                    let (_, kind) = respawns.swap_remove(i);
                    let c = make_critter(&mut next_id, kind, random_pos());
                    rings.push(Ring { pos: c.pos, color: WHITE, start: now, until: now + 0.5 });
                    sims.insert(&c);
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
            vec2(MAP_W as f32 * S + 10.0, 446.0),
            vec2(PANEL_W - 20.0, 280.0),
        )
        .label("tuning (live)")
        .titlebar(true)
        .movable(false)
        .ui(&mut *root_ui(), |ui| {
            ui.slider(hash!(), "split >", 1f32..12f32, &mut split_f);
            ui.slider(hash!(), "merge <=", 1f32..12f32, &mut merge_f);
            ui.separator();
            ui.slider(hash!(), "drifters", 0f32..1200f32, &mut drifters_f);
            ui.slider(hash!(), "hunters", 0f32..1200f32, &mut hunters_f);
            ui.slider(hash!(), "pulsars", 0f32..1200f32, &mut pulsars_f);
            ui.separator();
            ui.slider(hash!(), "respawn s", 0.5f32..10f32, &mut respawn_f);
            ui.slider(hash!(), "speed x", 0.25f32..4f32, &mut speed_f);
            ui.slider(hash!(), "fire x", 0.25f32..3f32, &mut fire_f);
        });

        // Integer-valued sliders snap to whole numbers.
        split_f = split_f.round().clamp(1.0, 12.0);
        merge_f = merge_f.round().clamp(1.0, split_f);
        drifters_f = drifters_f.round();
        hunters_f = hunters_f.round();
        pulsars_f = pulsars_f.round();

        // Apply split/merge thresholds; rebuild on change (both structures
        // in dual mode — the sliders affect them identically).
        let want_split = split_f as usize;
        let want_merge = (merge_f as usize).min(want_split);
        let current_limits = sims
            .tree
            .as_ref()
            .map(|t| (t.item_limit, t.merge_limit))
            .or_else(|| sims.quad.as_ref().map(|q| (q.item_limit, q.merge_limit)));
        if current_limits != Some((want_split, want_merge)) {
            let items: Vec<Critter> = sims
                .snapshot()
                .into_iter()
                .map(|(id, kind, pos, heading)| Critter { id, kind, pos, heading })
                .collect();
            let mode = sims.mode();
            sims.apply_mode(mode, world, want_split, want_merge, &items);
        }

        // Steer populations toward the slider targets.
        {
            let snap = sims.snapshot();
            let mut alive: HashMap<Kind, Vec<(u32, VPoint)>> = HashMap::new();
            for &(id, kind, pos, _) in &snap {
                alive.entry(kind).or_default().push((id, pos));
            }
            for (kind, target_f) in [
                (Kind::Drifter, drifters_f),
                (Kind::Hunter, hunters_f),
                (Kind::Pulsar, pulsars_f),
            ] {
                let target = target_f as i64;
                let queued = respawns.iter().filter(|(_, k)| *k == kind).count() as i64;
                let live = alive.get(&kind).map_or(0, |v| v.len()) as i64;
                let mut diff = target - live - queued;
                while diff > 0 && sims.item_count() < MAX_CRITTERS {
                    let c = make_critter(&mut next_id, kind, random_pos());
                    rings.push(Ring { pos: c.pos, color: kind.color(), start: now, until: now + 0.4 });
                    sims.insert(&c);
                    diff -= 1;
                }
                if diff < 0 {
                    let mut to_remove = -diff;
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
                            if sims.remove(pos, id).is_some() {
                                cooldowns.remove(&id);
                                rings.push(Ring { pos, color: kind.color(), start: now, until: now + 0.4 });
                            }
                            to_remove -= 1;
                        }
                    }
                }
            }
        }

        // Push graph samples.
        g_frame.push(get_frame_time() * 1000.0);
        let avg = |total: f64, n: u32| if n > 0 { (total / n as f64) as f32 } else { 0.0 };
        g_atk_t.push(avg(sims.t.atk, sims.t.atk_n));
        g_atk_q.push(avg(sims.q.atk, sims.q.atk_n));
        g_vis_t.push(avg(sims.t.vis, sims.t.vis_n));
        g_vis_q.push(avg(sims.q.vis, sims.q.vis_n));
        g_mv_t.push((sims.t.mv / 1000.0) as f32);
        g_mv_q.push((sims.q.mv / 1000.0) as f32);
        g_rm_t.push(sims.t.rm as f32);
        g_rm_q.push(sims.q.rm as f32);

        // ---------- draw (world coords × S) ----------
        clear_background(Color::new(0.07, 0.07, 0.10, 1.0));

        let mode = sims.mode();
        let mut tree_leaves = 0usize;
        let mut quad_leaves = 0usize;
        let mut items = 0usize;

        // Regions: fills from the primary structure; in dual mode the
        // quadtree's subdivision is overlaid as contrasting outlines.
        if let Some(t) = &sims.tree {
            t.visit_leaves(|id, leaf| {
                tree_leaves += 1;
                items += leaf.items.len();
                let b = leaf.bbox;
                let c = region_color(id.0);
                if region_mode == 0 {
                    draw_rectangle(
                        b.x as f32 * S,
                        b.y as f32 * S,
                        b.width as f32 * S,
                        b.height as f32 * S,
                        Color::new(c.r, c.g, c.b, 0.28),
                    );
                }
                if region_mode <= 1 {
                    draw_rectangle_lines(
                        b.x as f32 * S,
                        b.y as f32 * S,
                        b.width as f32 * S,
                        b.height as f32 * S,
                        1.0,
                        Color::new(1.0, 1.0, 1.0, 0.18),
                    );
                }
            });
        }
        if let Some(q) = &sims.quad {
            let quad_only = sims.tree.is_none();
            q.visit_leaves(|id, leaf| {
                quad_leaves += 1;
                if quad_only {
                    items += leaf.items.len();
                }
                let b = leaf.bbox;
                if quad_only {
                    let c = region_color(id.0.wrapping_add(101));
                    if region_mode == 0 {
                        draw_rectangle(
                            b.x as f32 * S,
                            b.y as f32 * S,
                            b.width as f32 * S,
                            b.height as f32 * S,
                            Color::new(c.r, c.g, c.b, 0.28),
                        );
                    }
                    if region_mode <= 1 {
                        draw_rectangle_lines(
                            b.x as f32 * S,
                            b.y as f32 * S,
                            b.width as f32 * S,
                            b.height as f32 * S,
                            1.0,
                            Color::new(1.0, 1.0, 1.0, 0.18),
                        );
                    }
                } else if region_mode <= 1 {
                    // Dual mode: quadtree as overlay outlines.
                    draw_rectangle_lines(
                        b.x as f32 * S,
                        b.y as f32 * S,
                        b.width as f32 * S,
                        b.height as f32 * S,
                        1.5,
                        Color::new(COL_QUAD.r, COL_QUAD.g, COL_QUAD.b, 0.55),
                    );
                }
            });
        }

        // Attack effects.
        for e in &effects {
            let fade = ((e.until - now) / EFFECT_TTL) as f32;
            if let Some(grid) = &e.grid {
                for row in 0..grid.rows {
                    for col in 0..grid.cols {
                        let alpha = match grid.cell(col, row) {
                            CellState::In => 0.5 * fade,
                            CellState::Maybe => 0.22 * fade,
                            CellState::Out => continue,
                        };
                        draw_rectangle(
                            (grid.origin_x + col as f64 * grid.cell_w) as f32 * S,
                            (grid.origin_y + row as f64 * grid.cell_h) as f32 * S,
                            grid.cell_w as f32 * S,
                            grid.cell_h as f32 * S,
                            Color::new(e.color.r, e.color.g, e.color.b, alpha),
                        );
                    }
                }
            }
            let oc = Color::new(e.color.r, e.color.g, e.color.b, (0.9 * fade).min(1.0));
            for i in 0..e.outline.len() {
                let (x1, y1) = e.outline[i];
                let (x2, y2) = e.outline[(i + 1) % e.outline.len()];
                draw_line(x1 * S, y1 * S, x2 * S, y2 * S, 1.5, oc);
            }
            draw_circle(e.origin.x as f32 * S, e.origin.y as f32 * S, 3.0, e.color);
        }

        // Sightlines + critters.
        for &(from, to) in &sightlines {
            draw_line(
                from.x as f32 * S,
                from.y as f32 * S,
                to.x as f32 * S,
                to.y as f32 * S,
                1.0,
                Color::new(1.0, 0.3, 0.3, 0.18),
            );
        }
        let draw_snap = sims.snapshot();
        for &(_, kind, pos, heading) in &draw_snap {
            let (x, y) = (pos.x as f32 * S, pos.y as f32 * S);
            draw_circle(x, y, 5.0, kind.color());
            draw_line(
                x,
                y,
                x + (heading.cos() * 9.0) as f32 * S,
                y + (heading.sin() * 9.0) as f32 * S,
                1.5,
                WHITE,
            );
        }
        for r in &rings {
            let t = ((now - r.start) / (r.until - r.start).max(1e-6)) as f32;
            draw_circle_lines(
                r.pos.x as f32 * S,
                r.pos.y as f32 * S,
                (4.0 + 16.0 * t) * S,
                2.0,
                Color::new(r.color.r, r.color.g, r.color.b, (1.0 - t).max(0.0)),
            );
        }

        // ---------- side panel ----------
        let map_px = MAP_W as f32 * S;
        let map_py = MAP_H as f32 * S;
        draw_rectangle_lines(0.0, 0.0, map_px, map_py, 2.0, GRAY);
        let px = map_px + 14.0;
        draw_rectangle(map_px, 0.0, PANEL_W, map_py, Color::new(0.04, 0.04, 0.06, 1.0));
        let mut ty = 28.0;
        let line = |text: &str, size: f32, color: Color, ty: &mut f32| {
            draw_text(text, px, *ty, size, color);
            *ty += size * 1.15;
        };
        line("vectorial-hash critters", 26.0, WHITE, &mut ty);
        ty += 6.0;
        line(
            &format!("[M] structure: {}", mode.name()),
            19.0,
            match mode {
                Mode::Binary => COL_BIN,
                Mode::Quad => COL_QUAD,
                Mode::Both => YELLOW,
            },
            &mut ty,
        );
        if mode == Mode::Both {
            line(
                &format!(
                    "  culls agree: {}",
                    if sims.mismatches == 0 {
                        "yes".to_string()
                    } else {
                        format!("{} mismatches!", sims.mismatches)
                    },
                ),
                16.0,
                if sims.mismatches == 0 { GREEN } else { RED },
                &mut ty,
            );
        }
        ty += 6.0;
        line(
            &format!("bank: {} combos -> {} unique", arsenal.bank.entry_count(), arsenal.bank.unique_count()),
            17.0,
            LIGHTGRAY,
            &mut ty,
        );
        ty += 8.0;
        let mut counts: HashMap<Kind, usize> = HashMap::new();
        for &(_, kind, _, _) in &draw_snap {
            *counts.entry(kind).or_default() += 1;
        }
        for kind in [Kind::Drifter, Kind::Hunter, Kind::Pulsar] {
            draw_circle(px + 6.0, ty - 5.0, 5.0, kind.color());
            line(
                &format!("    {} x{}", kind.name(), counts.get(&kind).copied().unwrap_or(0)),
                16.0,
                WHITE,
                &mut ty,
            );
        }
        ty += 8.0;
        line(&format!("alive: {}   kills: {}", items, kills), 18.0, WHITE, &mut ty);
        line(
            &format!(
                "  by: drf {} / hun {} / pul {}",
                kills_by.get(&Kind::Drifter).copied().unwrap_or(0),
                kills_by.get(&Kind::Hunter).copied().unwrap_or(0),
                kills_by.get(&Kind::Pulsar).copied().unwrap_or(0),
            ),
            15.0,
            LIGHTGRAY,
            &mut ty,
        );
        line(&format!("respawning: {}", respawns.len()), 17.0, WHITE, &mut ty);
        ty += 8.0;
        if let Some(t) = &sims.tree {
            line(
                &format!("binary: {} leaves, {} nodes", tree_leaves, t.node_count()),
                16.0,
                COL_BIN,
                &mut ty,
            );
        }
        if let Some(q) = &sims.quad {
            line(
                &format!("quad:   {} leaves, {} nodes", quad_leaves, q.node_count()),
                16.0,
                COL_QUAD,
                &mut ty,
            );
        }
        line(
            &format!("limits: split >{want_split}, merge <={want_merge}"),
            16.0,
            LIGHTGRAY,
            &mut ty,
        );
        ty += 8.0;
        draw_circle(px + 6.0, ty - 5.0, 5.0, brush.color());
        line(&format!("    brush: {} (1/2/3)", brush.name()), 17.0, WHITE, &mut ty);
        line(
            &format!("speed {:.2}x   regions: {}", speed_f, match region_mode {
                0 => "fill+lines",
                1 => "lines",
                _ => "off",
            }),
            16.0,
            WHITE,
            &mut ty,
        );
        line(&format!("fps: {}", get_fps()), 17.0, GREEN, &mut ty);
        if paused {
            line("PAUSED", 22.0, ORANGE, &mut ty);
        }

        // ---------- graphs column ----------
        let gx = map_px + PANEL_W;
        draw_rectangle(gx, 0.0, GRAPH_W, map_py, Color::new(0.05, 0.05, 0.08, 1.0));
        let gpad = 10.0;
        let gw = GRAPH_W - 2.0 * gpad;
        let gh = 118.0;
        let mut gy = 14.0;
        draw_text("performance (live, 1 thread)", gx + gpad, gy + 6.0, 19.0, WHITE);
        gy += 16.0;
        draw_graph(gx + gpad, gy, gw, gh, "frame", " ms", &[(&g_frame, GREEN, "")]);
        gy += gh + 8.0;
        draw_graph(
            gx + gpad, gy, gw, gh, "attack cull", " us",
            &[(&g_atk_t, COL_BIN, "bin"), (&g_atk_q, COL_QUAD, "quad")],
        );
        gy += gh + 8.0;
        draw_graph(
            gx + gpad, gy, gw, gh, "vision cull", " us",
            &[(&g_vis_t, COL_BIN, "bin"), (&g_vis_q, COL_QUAD, "quad")],
        );
        gy += gh + 8.0;
        draw_graph(
            gx + gpad, gy, gw, gh, "move+update", " ms",
            &[(&g_mv_t, COL_BIN, "bin"), (&g_mv_q, COL_QUAD, "quad")],
        );
        gy += gh + 8.0;
        draw_graph(
            gx + gpad, gy, gw, gh, "insert+remove", " us",
            &[(&g_rm_t, COL_BIN, "bin"), (&g_rm_q, COL_QUAD, "quad")],
        );

        let help = [
            "[M] structure   [1/2/3] brush",
            "[LMB] paint   [RMB] remove   [+/-] x5",
            "[R] regions   [ / ] speed   [Space] pause",
        ];
        for (i, h) in help.iter().enumerate() {
            draw_text(
                h,
                gx + gpad,
                map_py - 12.0 - 18.0 * (help.len() - 1 - i) as f32,
                15.0,
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
