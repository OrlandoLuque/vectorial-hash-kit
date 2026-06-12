//! Critters simulation core — deterministic, macroquad-free.
//!
//! A 2D world indexed by the binary-split [`Tree`], the reference
//! [`QuadTree`], or both at once (every operation applied to both with
//! identical inputs, cull results compared). Critters move with per-kind
//! behaviours and attack with precomputed template areas served by a
//! [`TemplateBank`]. All randomness flows through one seeded xorshift, so a
//! given (seed, mode, params) replays identically — which is what makes the
//! headless statistics comparable across structures.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use vectorial_hash::{
    PlacedTemplate, Point as VPoint, Positioned, QuadTree, Rect as VRect, Shape, Tree,
};
use vectorial_hash_templates::bank::{FigureKey, TemplateBank};
use vectorial_hash_templates::polygon::{
    create_circle, create_drop, rotated_copy, scaled_copy, Polygon,
};
use vectorial_hash_templates::templates::angle_to_radians;

pub const MAP_W: f64 = 1024.0;
pub const MAP_H: f64 = 1024.0;
pub const MARGIN: f64 = 4.0;
pub const VISION_RADIUS: f64 = 280.0;
pub const MAX_CRITTERS: usize = 4000;

pub const ANGLE_STEP_DEG: f64 = 15.0;
pub const DROP_ID: u32 = 0;
pub const CIRCLE_ID: u32 = 1;
pub const DROP_SCALE: f64 = 110.0;
pub const CIRCLE_RADIUS: f64 = 48.0;

/// Cell sizes (w, h) with full template sets.
pub const TEMPLATE_SIZES: [(u32, u32); 7] =
    [(8, 8), (16, 16), (32, 32), (8, 16), (16, 8), (16, 32), (32, 16)];

pub const ITEM_LIMIT: usize = 3;
pub const RESPAWN_DELAY: f64 = 2.5;

// ------------------------------------------------------------------ random

/// xorshift64* — all simulation randomness flows through this.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ----------------------------------------------------------------- critters

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    Drifter,
    Hunter,
    Pulsar,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Drifter, Kind::Hunter, Kind::Pulsar];

    pub fn idx(self) -> usize {
        match self {
            Kind::Drifter => 0,
            Kind::Hunter => 1,
            Kind::Pulsar => 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Drifter => "drifter",
            Kind::Hunter => "hunter",
            Kind::Pulsar => "pulsar",
        }
    }
    pub fn speed(self) -> f64 {
        match self {
            Kind::Drifter => 75.0,
            Kind::Hunter => 105.0,
            Kind::Pulsar => 88.0,
        }
    }
    pub fn cooldown(self) -> (f64, f64) {
        match self {
            Kind::Drifter => (2.5, 4.5),
            Kind::Hunter => (1.8, 3.5),
            Kind::Pulsar => (3.0, 5.0),
        }
    }
}

#[derive(Clone)]
pub struct Critter {
    pub id: u32,
    pub kind: Kind,
    pub pos: VPoint,
    pub heading: f64,
}

impl Positioned for Critter {
    fn position(&self) -> VPoint {
        self.pos
    }
}

// --------------------------------------------------------------- structures

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Binary,
    Quad,
    Both,
}

impl Mode {
    pub fn next(self) -> Mode {
        match self {
            Mode::Binary => Mode::Quad,
            Mode::Quad => Mode::Both,
            Mode::Both => Mode::Binary,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Mode::Binary => "binary tree",
            Mode::Quad => "quadtree",
            Mode::Both => "both (compare)",
        }
    }
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "binary" | "bin" | "tree" => Some(Mode::Binary),
            "quad" | "quadtree" => Some(Mode::Quad),
            "both" => Some(Mode::Both),
            _ => None,
        }
    }
}

/// Per-frame accumulated timings for one structure, in microseconds.
#[derive(Default, Clone, Copy)]
pub struct OpStats {
    pub mv: f64,
    pub atk: f64,
    pub atk_n: u32,
    pub vis: f64,
    pub vis_n: u32,
    pub rm: f64,
}

/// Owns whichever structures the current mode requires and applies every
/// operation to all of them with identical inputs, timing each separately.
pub struct Sims {
    pub tree: Option<Tree<Critter>>,
    pub quad: Option<QuadTree<Critter>>,
    pub t: OpStats,
    pub q: OpStats,
    pub mismatches: u64,
}

impl Sims {
    pub fn new(mode: Mode, world: VRect, split: usize, merge: usize) -> Self {
        let mut s = Sims {
            tree: None,
            quad: None,
            t: OpStats::default(),
            q: OpStats::default(),
            mismatches: 0,
        };
        s.apply_mode(mode, world, split, merge, &[]);
        s
    }

    pub fn mode(&self) -> Mode {
        match (&self.tree, &self.quad) {
            (Some(_), Some(_)) => Mode::Both,
            (Some(_), None) => Mode::Binary,
            _ => Mode::Quad,
        }
    }

    pub fn apply_mode(&mut self, mode: Mode, world: VRect, split: usize, merge: usize, items: &[Critter]) {
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

    pub fn begin_frame(&mut self) {
        self.t = OpStats::default();
        self.q = OpStats::default();
    }

    pub fn snapshot(&self) -> Vec<(u32, Kind, VPoint, f64)> {
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

    pub fn item_count(&self) -> usize {
        if let Some(t) = &self.tree {
            t.item_count()
        } else if let Some(q) = &self.quad {
            q.item_count()
        } else {
            0
        }
    }

    pub fn insert(&mut self, c: &Critter) {
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

    pub fn remove(&mut self, pos: VPoint, id: u32) -> Option<Critter> {
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

    pub fn update_critter(&mut self, pos: VPoint, id: u32, np: VPoint, nh: f64) {
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

    pub fn cull_attack<Sh: Shape>(&mut self, shape: &Sh) -> Vec<(u32, VPoint)> {
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

    pub fn vision_prey(&mut self, pos: VPoint, self_id: u32) -> Option<VPoint> {
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
pub struct VisionCircle {
    pub center: VPoint,
    pub r: f64,
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
pub struct Arsenal {
    pub bank: TemplateBank,
    pub figures: HashMap<u32, FigureKey>,
    pub base_polys: HashMap<(u32, i64), Polygon>,
    pub gen_seconds: f64,
}

pub fn build_arsenal() -> Arsenal {
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
            base_polys.insert(
                (shape_id, angle as i64),
                rotated_copy(&base, angle_to_radians(angle)),
            );
        }
        figures.insert(shape_id, figure);
    }

    Arsenal { bank, figures, base_polys, gen_seconds: start.elapsed().as_secs_f64() }
}

/// Attack area applied at its real integer origin.
pub struct AttackShape<'a> {
    pub bank: &'a TemplateBank,
    pub figure: FigureKey,
    pub angle_deg: f64,
    pub origin: (i64, i64),
    pub poly: Polygon,
    pub bbox: VRect,
    pub raster: Option<PlacedTemplate>,
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

pub fn make_attack<'a>(
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

// --------------------------------------------------------------------- sim

/// Events the renderer may care about; headless runs ignore them.
pub enum SimEvent {
    /// An attack fired: kind of the shooter, world origin, figure + angle
    /// (enough to rebuild the visual template/outline from the arsenal).
    Attack { kind: Kind, origin: VPoint, shape_id: u32, angle_deg: f64, origin_int: (i64, i64) },
    /// A critter died at `pos`.
    Kill { pos: VPoint, kind: Kind },
    /// A critter spawned at `pos` (`respawn` distinguishes respawns).
    Spawn { pos: VPoint, kind: Kind, respawn: bool },
    /// A critter was removed (population control / manual).
    Removed { pos: VPoint, kind: Kind },
}

/// Tunable parameters, applied every step (the graphical sliders map 1:1).
#[derive(Clone, Copy)]
pub struct SimParams {
    pub targets: [usize; 3],
    pub respawn_delay: f64,
    pub fire_rate: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams { targets: [16, 12, 14], respawn_delay: RESPAWN_DELAY, fire_rate: 1.0 }
    }
}

/// The full simulation: structures + behaviour + population management.
pub struct Sim {
    pub sims: Sims,
    pub rng: Rng,
    pub time: f64,
    pub kills: u64,
    pub kills_by: HashMap<Kind, u64>,
    pub events: Vec<SimEvent>,
    pub sightlines: Vec<(VPoint, VPoint)>,
    cooldowns: HashMap<u32, f64>,
    respawns: Vec<(f64, Kind)>,
    next_id: u32,
}

impl Sim {
    pub fn world() -> VRect {
        VRect::new(0.0, 0.0, MAP_W, MAP_H)
    }

    pub fn new(mode: Mode, split: usize, merge: usize, seed: u64) -> Self {
        Sim {
            sims: Sims::new(mode, Self::world(), split, merge),
            rng: Rng::new(seed),
            time: 0.0,
            kills: 0,
            kills_by: HashMap::new(),
            events: Vec::new(),
            sightlines: Vec::new(),
            cooldowns: HashMap::new(),
            respawns: Vec::new(),
            next_id: 0,
        }
    }

    pub fn respawn_queue_len(&self) -> usize {
        self.respawns.len()
    }

    pub fn respawns_of(&self, kind: Kind) -> usize {
        self.respawns.iter().filter(|(_, k)| *k == kind).count()
    }

    pub fn set_mode(&mut self, mode: Mode, split: usize, merge: usize) {
        let items = self.items();
        self.sims.apply_mode(mode, Self::world(), split, merge, &items);
    }

    pub fn set_limits(&mut self, split: usize, merge: usize) {
        let mode = self.sims.mode();
        let items = self.items();
        self.sims.apply_mode(mode, Self::world(), split, merge, &items);
    }

    fn items(&self) -> Vec<Critter> {
        self.sims
            .snapshot()
            .into_iter()
            .map(|(id, kind, pos, heading)| Critter { id, kind, pos, heading })
            .collect()
    }

    pub fn random_pos(&mut self) -> VPoint {
        VPoint::new(
            self.rng.range(20.0, MAP_W - 20.0),
            self.rng.range(20.0, MAP_H - 20.0),
        )
    }

    pub fn spawn_at(&mut self, kind: Kind, pos: VPoint, respawn: bool) {
        let id = self.next_id;
        self.next_id += 1;
        let heading = self.rng.range(0.0, std::f64::consts::TAU);
        let c = Critter { id, kind, pos, heading };
        self.sims.insert(&c);
        self.events.push(SimEvent::Spawn { pos, kind, respawn });
    }

    pub fn remove_by_id(&mut self, pos: VPoint, id: u32, kind: Kind) -> bool {
        let removed = self.sims.remove(pos, id).is_some();
        if removed {
            self.cooldowns.remove(&id);
            self.events.push(SimEvent::Removed { pos, kind });
        }
        removed
    }

    /// One simulation step of `dt` seconds: movement (with vision culls),
    /// firing (attack culls), deaths, respawns, population steering.
    pub fn step(&mut self, dt: f64, arsenal: &Arsenal, params: &SimParams) {
        self.time += dt;
        let now = self.time;

        // Movement.
        let snap = self.sims.snapshot();
        self.sightlines.clear();
        for &(id, kind, pos, heading) in &snap {
            let target = if kind == Kind::Hunter {
                let prey = self.sims.vision_prey(pos, id);
                if let Some(t) = prey {
                    self.sightlines.push((pos, t));
                }
                prey
            } else {
                None
            };
            let (np, nh) = self.step_critter(kind, pos, heading, dt, target);
            self.sims.update_critter(pos, id, np, nh);
        }

        // Firing.
        let snap2 = self.sims.snapshot();
        let kind_of: HashMap<u32, Kind> = snap2.iter().map(|&(id, k, _, _)| (id, k)).collect();
        let mut killed: Vec<(u32, VPoint, Kind, Kind)> = Vec::new();
        let mut killed_ids: HashSet<u32> = HashSet::new();
        for &(id, kind, pos, _) in &snap2 {
            if killed_ids.contains(&id) {
                continue;
            }
            let (cd_min, cd_max) = kind.cooldown();
            let cd = match self.cooldowns.get_mut(&id) {
                Some(v) => v,
                None => {
                    let init = self.rng.range(cd_min, cd_max);
                    self.cooldowns.entry(id).or_insert(init)
                }
            };
            *cd -= dt;
            if *cd > 0.0 {
                continue;
            }
            let reset = self.rng.range(cd_min, cd_max) / params.fire_rate;
            *self.cooldowns.get_mut(&id).unwrap() = reset;

            let attack = match kind {
                Kind::Hunter => self.sims.vision_prey(pos, id).and_then(|tpos| {
                    make_attack(arsenal, DROP_ID, pos, Some((tpos.x - pos.x, tpos.y - pos.y)))
                }),
                Kind::Drifter => {
                    let a = self.rng.range(0.0, std::f64::consts::TAU);
                    make_attack(arsenal, DROP_ID, pos, Some((a.cos(), a.sin())))
                }
                Kind::Pulsar => make_attack(arsenal, CIRCLE_ID, pos, None),
            };
            if let Some(atk) = attack {
                for (vid, vpos) in self.sims.cull_attack(&atk) {
                    if vid != id && !killed_ids.contains(&vid) {
                        killed_ids.insert(vid);
                        let vkind = kind_of.get(&vid).copied().unwrap_or(Kind::Drifter);
                        killed.push((vid, vpos, vkind, kind));
                    }
                }
                self.events.push(SimEvent::Attack {
                    kind,
                    origin: pos,
                    shape_id: if kind == Kind::Pulsar { CIRCLE_ID } else { DROP_ID },
                    angle_deg: atk.angle_deg,
                    origin_int: atk.origin,
                });
            }
        }
        for (vid, vpos, vkind, attacker) in killed {
            if self.sims.remove(vpos, vid).is_some() {
                self.cooldowns.remove(&vid);
                self.respawns.push((now + params.respawn_delay, vkind));
                self.kills += 1;
                *self.kills_by.entry(attacker).or_default() += 1;
                self.events.push(SimEvent::Kill { pos: vpos, kind: vkind });
            }
        }

        // Respawns due.
        let mut i = 0;
        while i < self.respawns.len() {
            if self.respawns[i].0 <= now {
                let (_, kind) = self.respawns.swap_remove(i);
                let pos = self.random_pos();
                self.spawn_at(kind, pos, true);
            } else {
                i += 1;
            }
        }

        // Population steering toward targets.
        let snap3 = self.sims.snapshot();
        let mut alive: HashMap<Kind, Vec<(u32, VPoint)>> = HashMap::new();
        for &(id, kind, pos, _) in &snap3 {
            alive.entry(kind).or_default().push((id, pos));
        }
        for kind in Kind::ALL {
            let target = params.targets[kind.idx()] as i64;
            let queued = self.respawns_of(kind) as i64;
            let live = alive.get(&kind).map_or(0, |v| v.len()) as i64;
            let mut diff = target - live - queued;
            while diff > 0 && self.sims.item_count() < MAX_CRITTERS {
                let pos = self.random_pos();
                self.spawn_at(kind, pos, false);
                diff -= 1;
            }
            if diff < 0 {
                let mut to_remove = -diff;
                let mut i = 0;
                while i < self.respawns.len() && to_remove > 0 {
                    if self.respawns[i].1 == kind {
                        self.respawns.swap_remove(i);
                        to_remove -= 1;
                    } else {
                        i += 1;
                    }
                }
                if let Some(list) = alive.get_mut(&kind) {
                    while to_remove > 0 && !list.is_empty() {
                        let idx = self.rng.below(list.len());
                        let (id, pos) = list.swap_remove(idx);
                        self.remove_by_id(pos, id, kind);
                        to_remove -= 1;
                    }
                }
            }
        }
    }

    fn step_critter(
        &mut self,
        kind: Kind,
        pos: VPoint,
        heading: f64,
        dt: f64,
        target: Option<VPoint>,
    ) -> (VPoint, f64) {
        let mut h = heading;
        match kind {
            Kind::Drifter => h += self.rng.range(-2.5, 2.5) * dt,
            Kind::Pulsar => h += 1.6 * dt,
            Kind::Hunter => match target {
                Some(tpos) => {
                    let desired = (tpos.y - pos.y).atan2(tpos.x - pos.x);
                    let diff = (desired - h + std::f64::consts::PI)
                        .rem_euclid(std::f64::consts::TAU)
                        - std::f64::consts::PI;
                    h += diff.clamp(-3.0 * dt, 3.0 * dt);
                }
                None => h += self.rng.range(-2.0, 2.0) * dt,
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
}
