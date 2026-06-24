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
    IntegerTree, IPoint, IPositioned, IRect, IUpdateStrategy, PlacedTemplate, Point as VPoint,
    Positioned, QuadTree, Rect as VRect, Shape, Tree, UpdateStrategy, WalkNeighbors,
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
pub const MAX_CRITTERS: usize = 40000;

pub const ANGLE_STEP_DEG: f64 = 15.0;
pub const DROP_ID: u32 = 0;
pub const CIRCLE_ID: u32 = 1;
pub const DROP_SCALE: f64 = 110.0;
pub const CIRCLE_RADIUS: f64 = 48.0;

/// Cell sizes (w, h) with full template sets.
pub const TEMPLATE_SIZES: [(u32, u32); 9] =
    [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128),
     (8, 16), (16, 8), (16, 32), (32, 16)];

/// Default split / merge threshold for both trees. 100 is the empirical
/// throughput optimum on this workload (1024² world, 10k pop, drop +
/// circle figures) — see `docs/UPDATE_STRATEGIES.md`'s extended sweep.
/// The optimum is workload-dependent: small figures or sparser
/// populations push it lower; richer template sets push it higher.
pub const ITEM_LIMIT: usize = 100;
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
    /// Integer-rounded copy of `pos`, maintained whenever `pos` is set.
    /// Cached so the `IPositioned` impl reads a field instead of recomputing
    /// `pos.round() as i32` on every IntegerTree internal lookup (locate,
    /// divide, cull). The cache made integer-mode cull go from ~6× slower
    /// than the float tree (recomputing each call) to comparable.
    pub ipos: IPoint,
    pub heading: f64,
}

impl Critter {
    pub fn new(id: u32, kind: Kind, pos: VPoint, heading: f64) -> Self {
        Critter { id, kind, pos, ipos: to_ipoint(pos), heading }
    }
    /// Set both float and integer position together. Use this everywhere
    /// `pos` changes — failing to refresh `ipos` makes the integer tree
    /// silently drift out of sync.
    pub fn set_pos(&mut self, p: VPoint) {
        self.pos = p;
        self.ipos = to_ipoint(p);
    }
}

impl Positioned for Critter {
    fn position(&self) -> VPoint {
        self.pos
    }
}

/// Integer-side view: cached, just reads `self.ipos` — no rounding at the
/// hot path.
impl IPositioned for Critter {
    fn position(&self) -> IPoint {
        self.ipos
    }
}

#[inline]
fn to_ipoint(p: VPoint) -> IPoint {
    IPoint::new(p.x.round() as i32, p.y.round() as i32)
}

// --------------------------------------------------------------- structures

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Binary,
    Quad,
    Both,
    /// Bit-shift / integer-coord binary-split tree. Standalone — does not
    /// participate in `Both` comparisons because position rounding
    /// (f64 → i32) creates expected sub-pixel divergences at boundaries.
    IBinary,
}

impl Mode {
    pub fn next(self) -> Mode {
        match self {
            Mode::Binary => Mode::Quad,
            Mode::Quad => Mode::Both,
            Mode::Both => Mode::IBinary,
            Mode::IBinary => Mode::Binary,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Mode::Binary => "binary tree",
            Mode::Quad => "quadtree",
            Mode::Both => "both (compare)",
            Mode::IBinary => "integer binary tree",
        }
    }
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "binary" | "bin" | "tree" => Some(Mode::Binary),
            "quad" | "quadtree" => Some(Mode::Quad),
            "both" => Some(Mode::Both),
            "int_binary" | "ibinary" | "itree" => Some(Mode::IBinary),
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
    pub itree: Option<IntegerTree<Critter>>,
    pub t: OpStats,
    pub q: OpStats,
    pub it: OpStats,
    pub mismatches: u64,
    /// Strategy used by [`Sims::update_critter`]. Headless bench flips this
    /// per run; the interactive demo just uses the default.
    pub update_strategy: UpdateStrategy,
    /// Strategy used by [`Sims::vision_prey`] to find prey. `Descent` is
    /// `Tree::cull`; `Walk*` are `Tree::cull_walk` variants. `vision_prey`
    /// is the dominant cull (3000+ per frame), so this is where the
    /// `neighbors` feature can in principle pay back its bookkeeping cost
    /// — this switch is what the cull_walk break-even measures.
    pub cull_strategy: CullStrategy,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CullStrategy {
    /// `Tree::cull` (the default): template-driven hierarchical descent.
    Descent,
    /// `Tree::cull_walk` with [`WalkNeighbors::Samet`]: walk by reading
    /// parent pointers, no extra storage.
    WalkSamet,
    /// `Tree::cull_walk` with [`WalkNeighbors::Probe`]: walk by probing
    /// from the root, no extra storage.
    WalkProbe,
    /// `Tree::cull_walk` with [`WalkNeighbors::Ropes`]: walk through the
    /// pre-stored rope lists. Only meaningful with the `neighbors`
    /// feature on. Without the feature this falls back to `WalkSamet`.
    WalkRopes,
}

impl Default for CullStrategy {
    fn default() -> Self { CullStrategy::Descent }
}

impl Sims {
    pub fn new(mode: Mode, world: VRect, split: usize, merge: usize) -> Self {
        let mut s = Sims {
            tree: None,
            quad: None,
            itree: None,
            t: OpStats::default(),
            q: OpStats::default(),
            it: OpStats::default(),
            mismatches: 0,
            update_strategy: UpdateStrategy::default(),
            cull_strategy: CullStrategy::default(),
        };
        s.apply_mode(mode, world, split, merge, &[]);
        s
    }

    pub fn mode(&self) -> Mode {
        match (&self.tree, &self.quad, &self.itree) {
            (Some(_), Some(_), _) => Mode::Both,
            (Some(_), None, _) => Mode::Binary,
            (None, None, Some(_)) => Mode::IBinary,
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
        self.itree = matches!(mode, Mode::IBinary).then(|| {
            // World must be square pow-2 for IntegerTree. `MAP_W`/`MAP_H`
            // are 1024 so this holds for the demo's default world.
            let iworld = IRect::new(
                world.x.round() as i32,
                world.y.round() as i32,
                world.width.round() as i32,
                world.height.round() as i32,
            );
            let mut t = IntegerTree::with_limits(iworld, split, merge);
            for c in items {
                t.insert(c.clone());
            }
            t
        });
    }

    pub fn begin_frame(&mut self) {
        self.t = OpStats::default();
        self.q = OpStats::default();
        self.it = OpStats::default();
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
        } else if let Some(t) = &self.itree {
            t.visit_leaves(|_, leaf| {
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
        } else if let Some(t) = &self.itree {
            t.item_count()
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
        if let Some(t) = &mut self.itree {
            let s = Instant::now();
            t.insert(c.clone());
            self.it.rm += s.elapsed().as_secs_f64() * 1e6;
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
        if let Some(t) = &mut self.itree {
            let s = Instant::now();
            let r = t.remove(to_ipoint(pos), |c| c.id == id);
            self.it.rm += s.elapsed().as_secs_f64() * 1e6;
            if out.is_none() {
                out = r;
            }
        }
        out
    }

    pub fn update_critter(&mut self, pos: VPoint, id: u32, np: VPoint, nh: f64) {
        let strategy = self.update_strategy;
        if let Some(t) = &mut self.tree {
            let s = Instant::now();
            t.update_with(strategy, pos, |c| c.id == id, |c| {
                c.set_pos(np);
                c.heading = nh;
            });
            self.t.mv += s.elapsed().as_secs_f64() * 1e6;
        }
        if let Some(q) = &mut self.quad {
            let s = Instant::now();
            q.update_with(strategy, pos, |c| c.id == id, |c| {
                c.set_pos(np);
                c.heading = nh;
            });
            self.q.mv += s.elapsed().as_secs_f64() * 1e6;
        }
        if let Some(t) = &mut self.itree {
            let istrategy = match strategy {
                UpdateStrategy::Legacy => IUpdateStrategy::Legacy,
                _ => IUpdateStrategy::Lca,
            };
            let s = Instant::now();
            t.update_with(istrategy, to_ipoint(pos), |c| c.id == id, |c| {
                c.set_pos(np);
                c.heading = nh;
            });
            self.it.mv += s.elapsed().as_secs_f64() * 1e6;
        }
    }

    pub fn cull_attack<Sh: Shape>(&mut self, shape: &Sh) -> Vec<(u32, VPoint)> {
        let mut tree_ids: Option<Vec<(u32, VPoint)>> = None;
        let mut quad_ids: Option<Vec<(u32, VPoint)>> = None;
        let mut itree_ids: Option<Vec<(u32, VPoint)>> = None;
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
        if let Some(t) = &self.itree {
            let s = Instant::now();
            let hits = t.cull(shape);
            self.it.atk += s.elapsed().as_secs_f64() * 1e6;
            self.it.atk_n += 1;
            itree_ids = Some(hits.iter().map(|c| (c.id, c.pos)).collect());
        }
        if let (Some(a), Some(b)) = (&tree_ids, &quad_ids) {
            let sa: HashSet<u32> = a.iter().map(|(id, _)| *id).collect();
            let sb: HashSet<u32> = b.iter().map(|(id, _)| *id).collect();
            if sa != sb {
                self.mismatches += 1;
            }
        }
        tree_ids.or(quad_ids).or(itree_ids).unwrap_or_default()
    }

    pub fn vision_prey(&mut self, pos: VPoint, self_id: u32) -> Option<VPoint> {
        self.vision_prey_dilated(pos, self_id, 0.0)
    }

    /// Vision that also accounts for the prey's body radius: prey whose
    /// centre is within `VISION_RADIUS + agent_radius` are visible (the
    /// vision circle dilated by the agent radius — a circle dilates to a
    /// bigger circle, so no inflated template is needed).
    pub fn vision_prey_dilated(&mut self, pos: VPoint, self_id: u32, agent_radius: f64) -> Option<VPoint> {
        let vision = VisionCircle { center: pos, r: VISION_RADIUS + agent_radius };
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
        let mut from_itree = None;
        if let Some(t) = &self.tree {
            let s = Instant::now();
            let hits = match self.cull_strategy {
                CullStrategy::Descent   => t.cull(&vision),
                CullStrategy::WalkSamet => t.cull_walk(&vision, vision.center, WalkNeighbors::Samet),
                CullStrategy::WalkProbe => t.cull_walk(&vision, vision.center, WalkNeighbors::Probe),
                #[cfg(feature = "neighbors")]
                CullStrategy::WalkRopes => t.cull_walk(&vision, vision.center, WalkNeighbors::Ropes),
                #[cfg(not(feature = "neighbors"))]
                CullStrategy::WalkRopes => t.cull_walk(&vision, vision.center, WalkNeighbors::Samet),
            };
            from_tree = nearest(hits);
            self.t.vis += s.elapsed().as_secs_f64() * 1e6;
            self.t.vis_n += 1;
        }
        if let Some(q) = &self.quad {
            let s = Instant::now();
            from_quad = nearest(q.cull(&vision));
            self.q.vis += s.elapsed().as_secs_f64() * 1e6;
            self.q.vis_n += 1;
        }
        if let Some(t) = &self.itree {
            let s = Instant::now();
            from_itree = nearest(t.cull(&vision));
            self.it.vis += s.elapsed().as_secs_f64() * 1e6;
            self.it.vis_n += 1;
        }
        if let (Some((ia, _)), Some((ib, _))) = (from_tree, from_quad) {
            if ia != ib {
                self.mismatches += 1;
            }
        }
        from_tree.or(from_quad).or(from_itree).map(|(_, p)| p)
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
    build_arsenal_scaled(1.0)
}

/// Build an arsenal with both figures (drop and circle) uniformly scaled by
/// `figure_scale`. The bank caches templates keyed by figure dimensions, so
/// scaled figures create separate entries — no contamination across scales.
pub fn build_arsenal_scaled(figure_scale: f64) -> Arsenal {
    let start = Instant::now();
    let mut bank = TemplateBank::new();
    let mut figures = HashMap::new();
    let mut base_polys = HashMap::new();

    let drop_angles: Vec<f64> = (0..(360.0 / ANGLE_STEP_DEG) as i64)
        .map(|i| i as f64 * ANGLE_STEP_DEG)
        .collect();

    let drop_scale = DROP_SCALE * figure_scale;
    let circle_radius = CIRCLE_RADIUS * figure_scale;

    let shapes: [(u32, Vec<f64>, Polygon, Vec<f64>); 2] = [
        (
            DROP_ID,
            vec![0.2 * drop_scale, 0.8 * drop_scale],
            scaled_copy(&create_drop(0.2, 0.8), drop_scale, drop_scale),
            drop_angles,
        ),
        (
            CIRCLE_ID,
            vec![circle_radius],
            scaled_copy(&create_circle(1.0), circle_radius, circle_radius),
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
///
/// When `agent_radius > 0` the shape behaves as the figure **dilated by the
/// agent's body radius** (the Minkowski "index dilation" device): a critter
/// whose *centre* lies within `agent_radius` of the real figure is a hit.
/// The bounding box grows by `agent_radius`, the per-point test becomes
/// `poly.within_dilation(r, p)`, and the precomputed (un-inflated) templates
/// and 1×1 raster are skipped so the cull falls back to bbox + per-point —
/// correct, just without the green short-circuit. (Precomputing *inflated*
/// template sets per radius — `bank` already supports it, see
/// `tests/dilation.rs::precached_dilated_templates_select_by_radius_at_runtime`
/// — is the optimisation that restores the short-circuit.)
pub struct AttackShape<'a> {
    pub bank: &'a TemplateBank,
    pub figure: FigureKey,
    pub angle_deg: f64,
    pub origin: (i64, i64),
    pub poly: Polygon,
    pub bbox: VRect,
    pub raster: Option<PlacedTemplate>,
    pub agent_radius: f64,
}

impl Shape for AttackShape<'_> {
    fn bounding_box(&self) -> VRect {
        if self.agent_radius > 0.0 {
            let r = self.agent_radius;
            VRect::new(self.bbox.x - r, self.bbox.y - r, self.bbox.width + 2.0 * r, self.bbox.height + 2.0 * r)
        } else {
            self.bbox
        }
    }
    fn contains_point(&self, p: VPoint) -> bool {
        if self.agent_radius > 0.0 {
            self.poly.within_dilation(self.agent_radius, p.x, p.y)
        } else {
            self.poly.is_inside(p.x, p.y)
        }
    }
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
        if self.agent_radius > 0.0 || cell_w.fract() != 0.0 || cell_h.fract() != 0.0 {
            return None; // dilated → no matching un-inflated template; bbox fallback
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
        if self.agent_radius > 0.0 { None } else { self.raster.as_ref() }
    }
}

pub fn make_attack<'a>(
    arsenal: &'a Arsenal,
    shape_id: u32,
    pos: VPoint,
    aim: Option<(f64, f64)>,
) -> Option<AttackShape<'a>> {
    make_attack_dilated(arsenal, shape_id, pos, aim, 0.0)
}

pub fn make_attack_dilated<'a>(
    arsenal: &'a Arsenal,
    shape_id: u32,
    pos: VPoint,
    aim: Option<(f64, f64)>,
    agent_radius: f64,
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

    Some(AttackShape { bank: &arsenal.bank, figure, angle_deg, origin, poly, bbox, raster, agent_radius })
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
    /// Skip the entire firing / kill / respawn phase. Used by the headless
    /// bench to isolate `update` cost from cull / insert / remove churn.
    pub no_attack: bool,
    /// Agent body radius — when > 0, attack and vision culls hit critters
    /// whose centre is within this distance of the figure (Minkowski index
    /// dilation). 0 = point agents (the classic behaviour).
    pub agent_radius: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            targets: [16, 12, 14],
            respawn_delay: RESPAWN_DELAY,
            fire_rate: 1.0,
            no_attack: false,
            agent_radius: 0.0,
        }
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
    /// Side of the square world. Runtime (stepper) rather than the `MAP_W`
    /// const so the demo can resize the world live; defaults to `MAP_W`. Kept
    /// square and (for the `IntegerTree` mode) a power of two — see
    /// [`WORLD_STEPS`].
    world_size: f64,
}

/// Selectable world sizes for the live stepper. All powers of two (square),
/// so every mode — including the `IntegerTree`, which requires a square pow-2
/// world — stays valid. 1024 is the historical default and the middle step.
pub const WORLD_STEPS: [f64; 5] = [256.0, 512.0, 1024.0, 2048.0, 4096.0];

impl Sim {
    /// The default world rect (`MAP_W` square). Prefer [`Sim::world_rect`] for
    /// the live size; this stays for callers that just want the default.
    pub fn world() -> VRect {
        VRect::new(0.0, 0.0, MAP_W, MAP_H)
    }

    /// The current world rect (square, side [`Sim::world_size`]).
    pub fn world_rect(&self) -> VRect {
        VRect::new(0.0, 0.0, self.world_size, self.world_size)
    }

    /// The current world side length.
    pub fn world_size(&self) -> f64 {
        self.world_size
    }

    /// Resize the (square) world: clamp every critter into the new bounds and
    /// rebuild the index for the new rect. Rare and user-driven, so a full
    /// rebuild is fine (mirrors the 3D demo's world stepper). `split`/`merge`
    /// are the current thresholds the index should keep.
    pub fn set_world_size(&mut self, size: f64, split: usize, merge: usize) {
        let mode = self.sims.mode();
        let items: Vec<Critter> = self
            .sims
            .snapshot()
            .into_iter()
            .map(|(id, kind, pos, heading)| {
                let p = VPoint::new(
                    pos.x.clamp(MARGIN, size - MARGIN),
                    pos.y.clamp(MARGIN, size - MARGIN),
                );
                Critter::new(id, kind, p, heading)
            })
            .collect();
        self.world_size = size;
        self.sims.apply_mode(mode, self.world_rect(), split, merge, &items);
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
            world_size: MAP_W,
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
        self.sims.apply_mode(mode, self.world_rect(), split, merge, &items);
    }

    pub fn set_limits(&mut self, split: usize, merge: usize) {
        let mode = self.sims.mode();
        let items = self.items();
        self.sims.apply_mode(mode, self.world_rect(), split, merge, &items);
    }

    fn items(&self) -> Vec<Critter> {
        self.sims
            .snapshot()
            .into_iter()
            .map(|(id, kind, pos, heading)| Critter::new(id, kind, pos, heading))
            .collect()
    }

    pub fn random_pos(&mut self) -> VPoint {
        let w = self.world_size;
        VPoint::new(
            self.rng.range(20.0, w - 20.0),
            self.rng.range(20.0, w - 20.0),
        )
    }

    pub fn spawn_at(&mut self, kind: Kind, pos: VPoint, respawn: bool) {
        let id = self.next_id;
        self.next_id += 1;
        let heading = self.rng.range(0.0, std::f64::consts::TAU);
        let c = Critter::new(id, kind, pos, heading);
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
                let prey = self.sims.vision_prey_dilated(pos, id, params.agent_radius);
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

        // Firing. Skipped entirely when `no_attack` is set — used by the
        // headless bench to isolate `update` cost from cull / insert /
        // remove churn.
        if params.no_attack {
            // Still run population steering to fill to target initially.
            self.steer_population(params);
            return;
        }
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

            let ar = params.agent_radius;
            let attack = match kind {
                Kind::Hunter => self.sims.vision_prey_dilated(pos, id, ar).and_then(|tpos| {
                    make_attack_dilated(arsenal, DROP_ID, pos, Some((tpos.x - pos.x, tpos.y - pos.y)), ar)
                }),
                Kind::Drifter => {
                    let a = self.rng.range(0.0, std::f64::consts::TAU);
                    make_attack_dilated(arsenal, DROP_ID, pos, Some((a.cos(), a.sin())), ar)
                }
                Kind::Pulsar => make_attack_dilated(arsenal, CIRCLE_ID, pos, None, ar),
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

        self.steer_population(params);
    }

    /// Spawn until each kind reaches its target, removing surplus (from the
    /// respawn queue first, then from alive) if over.
    fn steer_population(&mut self, params: &SimParams) {
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
        let w = self.world_size;
        let mut nx = pos.x + h.cos() * speed * dt;
        let mut ny = pos.y + h.sin() * speed * dt;
        if nx < MARGIN || nx > w - MARGIN {
            h = std::f64::consts::PI - h;
            nx = nx.clamp(MARGIN, w - MARGIN);
        }
        if ny < MARGIN || ny > w - MARGIN {
            h = -h;
            ny = ny.clamp(MARGIN, w - MARGIN);
        }
        (VPoint::new(nx, ny), h.rem_euclid(std::f64::consts::TAU))
    }
}
