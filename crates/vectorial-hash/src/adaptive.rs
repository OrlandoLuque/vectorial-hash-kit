//! `AdaptiveIndex` — one index that **changes structure underneath you**.
//!
//! The kit's measurements keep landing on the same shape of answer: no single structure
//! wins, the winner moves with population, churn and query load, and the margins at the
//! boundaries are small enough that another CPU could flip them. The
//! [`advisor`](crate::advisor) has been *observing* that since it was written, but nothing
//! ever acted on its recommendation. This does.
//!
//! It owns the items, keeps whichever backend currently fits, and migrates when the
//! workload has genuinely changed:
//!
//! | backend | chosen when | why |
//! | --- | --- | --- |
//! | brute scan | few items | below ~500-1000 a contiguous scan beats any descent |
//! | [`Tree3`] + `ItemRef` | items move, moderate churn | O(1) relocation; wins maintain in 15 of 16 sweep configs |
//! | [`MortonGrid3`] rebuilt | items move a LOT | when relocation dominates, a rebuild is cheaper than fixing the tree (measured on the fluid demo, where keeping the index loses the frame by 16%) |
//! | [`KdTree3`] | nothing has moved for a while | build-once, best query on skewed data |
//!
//! **Hysteresis is the whole difficulty.** A naive "pick the best for the current numbers"
//! flaps: at the boundary it rebuilds every frame and loses to *both* candidates. So a
//! switch needs the new choice to hold for [`Thresholds::hold_ticks`] consecutive ticks,
//! the boundaries are widened by [`Thresholds::margin`] in the direction you are moving,
//! and there is a hard [`Thresholds::cooldown`] after every migration. Rebuilding is not
//! free and the policy is written to respect that.
//!
//! **Thresholds are calibratable.** The defaults are this repo's measurements on one
//! machine; a different cache hierarchy moves them. [`Thresholds::from_env`] reads a file
//! written by the `calibrate` example (`VH_CALIBRATION=path`), so a program can ship the
//! defaults and let whoever runs it do better.

use crate::advisor::SpatialProfile;
use crate::kdtree3::KdTree3;
use crate::morton3::MortonGrid3;
use crate::tree3::{Aabb, Crossing, ItemRef, Point3, Positioned3, Shape3, Tree3};

/// Which concrete structure is holding the items right now.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    /// A contiguous scan: no index at all.
    Brute,
    /// [`Tree3`] maintained through the `ItemRef` handle layer.
    KeepTree,
    /// [`MortonGrid3`], refilled when the contents change.
    Grid,
    /// [`KdTree3`], built once because nothing is moving.
    Static,
}

/// The numbers the policy switches on. Defaults are measured (see the module docs); a
/// different machine can and should re-measure them.
#[derive(Copy, Clone, Debug)]
pub struct Thresholds {
    /// Below this many items, a linear scan wins. Measured crossover ~500-1000 for a
    /// single AoI query; `advisor::BRUTE_FORCE_MAX` is the shared default.
    pub brute_max: usize,
    /// Fraction of moves that cross a leaf. Kept because it describes the workload, but
    /// it is NOT what the policy switches on — see `rebuild_query_ratio`.
    pub high_churn: f64,
    /// **Queries per item per tick** above which rebuilding a grid beats keeping the tree.
    ///
    /// This replaced churn as the deciding variable after the calibration swept both:
    /// churn never flipped the winner at any level (it only moved keep's margin from 116x
    /// down to 6.4x), while query load flipped it every time. A rebuild pays a big fixed
    /// cost and then answers from a perfectly fitted structure, so the more you ask, the
    /// sooner it pays — which is exactly why the fluid demo (one neighbour query PER
    /// PARTICLE) is the one workload in the repo where keeping the index loses the frame.
    pub rebuild_query_ratio: f64,
    /// Consecutive tick with no movement at all before the workload counts as static.
    pub static_ticks: u32,
    /// Boundaries are widened by this fraction in the direction of travel, so a workload
    /// sitting exactly on one does not oscillate.
    pub margin: f64,
    /// A candidate must win this many consecutive ticks before a migration happens.
    pub hold_ticks: u32,
    /// Minimum ticks between migrations, whatever the numbers say.
    pub cooldown: u32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            brute_max: crate::advisor::BRUTE_FORCE_MAX,
            high_churn: crate::advisor::HIGH_RELOCATION,
            // 20k items, rebuild started winning between 256 and 4096 culls a frame.
            rebuild_query_ratio: 0.1,
            static_ticks: crate::advisor::STATIC_TICKS,
            margin: 0.25,
            hold_ticks: 30,
            cooldown: 120,
        }
    }
}

impl Thresholds {
    /// Load from the file named by `VH_CALIBRATION`, falling back to the defaults for
    /// anything absent or unparseable. Format is one `key = value` per line, `#` comments
    /// — what the `calibrate` example writes after measuring the local machine.
    pub fn from_env() -> Self {
        match std::env::var("VH_CALIBRATION").ok().and_then(|p| std::fs::read_to_string(p).ok()) {
            Some(text) => Self::parse(&text),
            None => Self::default(),
        }
    }

    /// Parse the calibration format. Unknown keys are ignored, so a newer file stays
    /// readable by an older binary.
    pub fn parse(text: &str) -> Self {
        let mut t = Thresholds::default();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((k, v)) = line.split_once('=') else { continue };
            let (k, v) = (k.trim(), v.trim());
            match k {
                "brute_max" => if let Ok(x) = v.parse() { t.brute_max = x },
                "high_churn" => if let Ok(x) = v.parse() { t.high_churn = x },
                "rebuild_query_ratio" => if let Ok(x) = v.parse() { t.rebuild_query_ratio = x },
                "static_ticks" => if let Ok(x) = v.parse() { t.static_ticks = x },
                "margin" => if let Ok(x) = v.parse() { t.margin = x },
                "hold_ticks" => if let Ok(x) = v.parse() { t.hold_ticks = x },
                "cooldown" => if let Ok(x) = v.parse() { t.cooldown = x },
                _ => {}
            }
        }
        t
    }

    /// Serialise, for the `calibrate` example to write.
    pub fn to_text(&self) -> String {
        format!(
            "# vectorial-hash adaptive-index calibration\n\
             brute_max = {}\nhigh_churn = {}\nstatic_ticks = {}\nmargin = {}\n\
             hold_ticks = {}\ncooldown = {}\n",
            self.brute_max, self.high_churn, self.static_ticks, self.margin, self.hold_ticks, self.cooldown)
    }
}

enum Held<T: Positioned3> {
    Brute,
    Keep(Box<Tree3<T>>, Vec<ItemRef>),
    Grid(Box<MortonGrid3<T>>),
    Static(Box<KdTree3<T>>),
}

/// A stable handle into an [`AdaptiveIndex`]. Survives migrations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Slot(pub u32);

/// An index that picks its own structure. See the module docs.
pub struct AdaptiveIndex<T: Positioned3 + Clone> {
    /// The source of truth. Every backend is a view onto this, rebuilt on migration.
    items: Vec<T>,
    world: Aabb,
    leaf: usize,
    held: Held<T>,
    profile: SpatialProfile,
    th: Thresholds,
    /// The candidate that has been winning, and for how long.
    pending: Option<(Backend, u32)>,
    cooling: u32,
    dirty: bool,
    switches: u32,
    /// Counts for the next `tick`, accumulated by the mutating calls.
    moves: u64,
    relocations: u64,
    queries: u64,
    /// Smoothed queries per item per tick: the variable the backend choice turns on.
    q_per_item: f64,
}

impl<T: Positioned3 + Clone> AdaptiveIndex<T> {
    /// `world` bounds the grid backend; `leaf` is the tree leaf capacity.
    pub fn new(world: Aabb, leaf: usize) -> Self {
        Self::with_thresholds(world, leaf, Thresholds::from_env())
    }

    pub fn with_thresholds(world: Aabb, leaf: usize, th: Thresholds) -> Self {
        AdaptiveIndex {
            items: Vec::new(), world, leaf: leaf.max(1), held: Held::Brute,
            profile: SpatialProfile::default(), th, pending: None, cooling: 0,
            dirty: false, switches: 0, moves: 0, relocations: 0, queries: 0, q_per_item: 0.0,
        }
    }

    /// Which structure is currently in use.
    pub fn backend(&self) -> Backend {
        match self.held { Held::Brute => Backend::Brute, Held::Keep(..) => Backend::KeepTree, Held::Grid(_) => Backend::Grid, Held::Static(_) => Backend::Static }
    }
    /// How many migrations have happened — a flapping policy shows up here.
    pub fn switch_count(&self) -> u32 { self.switches }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
    pub fn thresholds(&self) -> &Thresholds { &self.th }
    pub fn profile(&self) -> &SpatialProfile { &self.profile }
    /// Smoothed queries per item per tick — what the keep-vs-rebuild choice turns on.
    pub fn queries_per_item(&self) -> f64 { self.q_per_item }

    pub fn insert(&mut self, item: T) -> Slot {
        let slot = Slot(self.items.len() as u32);
        match &mut self.held {
            Held::Keep(t, refs) => { if let Some(r) = t.insert_ref(item.clone()) { refs.push(r); } else { self.dirty = true; refs.push(ItemRef(u32::MAX)); } }
            Held::Grid(g) => { if !g.insert(item.clone()) { self.dirty = true; } }
            Held::Brute => {}
            Held::Static(_) => self.dirty = true, // a build-once backend cannot take one more
        }
        self.items.push(item);
        slot
    }

    /// Move (or otherwise mutate) one item. The keep-index backend takes the O(1) path;
    /// the rebuild-based ones just note that they are stale.
    pub fn update<F: FnOnce(&mut T)>(&mut self, s: Slot, f: F) {
        let Some(it) = self.items.get_mut(s.0 as usize) else { return };
        f(it);
        let item = it.clone();
        self.moves += 1;
        match &mut self.held {
            Held::Keep(t, refs) => {
                if let Some(r) = refs.get(s.0 as usize).copied().filter(|r| r.0 != u32::MAX) {
                    // `_tracked` so the policy learns the REAL relocation rate: how often a
                    // move actually crosses a leaf, which is the number that decides
                    // whether keeping the index still beats rebuilding it.
                    match t.update_ref_tracked(r, |c| *c = item) {
                        Crossing::Stayed(_) => {}
                        Crossing::Moved { .. } => self.relocations += 1,
                        _ => self.dirty = true,
                    }
                } else { self.dirty = true; }
            }
            // Any other backend is stale now — and it cannot tell us whether that move
            // crossed a leaf, so it must NOT claim one. Reporting every move as a
            // relocation here made the policy jump to the grid on its very first frame,
            // before it had ever seen a real crossing. Churn is learned on the tree.
            _ => self.dirty = true,
        }
    }

    /// Everything inside `shape`. Rebuilds a stale backend first, so a caller never sees
    /// a partially-updated index.
    pub fn cull<S: Shape3>(&mut self, shape: &S) -> Vec<&T> {
        self.queries += 1;
        self.refresh();
        match &self.held {
            Held::Brute => self.items.iter().filter(|it| shape.contains_point(it.position())).collect(),
            Held::Keep(t, _) => t.cull(shape),
            Held::Grid(g) => g.cull(shape),
            Held::Static(k) => k.cull(shape),
        }
    }

    /// k nearest to `q`, as `(distance, &item)` sorted by distance.
    pub fn knn(&mut self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        self.queries += 1;
        self.refresh();
        match &self.held {
            Held::Brute => {
                let mut v: Vec<(f64, &T)> = self.items.iter().map(|it| {
                    let p = it.position();
                    let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z);
                    ((dx * dx + dy * dy + dz * dz).sqrt(), it)
                }).collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                v.truncate(k);
                v
            }
            Held::Keep(t, _) => t.knn(q, k),
            Held::Grid(g) => g.knn(q, k),
            Held::Static(kd) => kd.knn(q, k),
        }
    }

    /// Close the frame: feed the observed rates to the advisor and migrate if the
    /// workload has genuinely changed. Call once per tick.
    pub fn tick(&mut self) -> Backend {
        let qpi = if self.items.is_empty() { 0.0 } else { self.queries as f64 / self.items.len() as f64 };
        self.q_per_item += 0.1 * (qpi - self.q_per_item); // same EMA weight the advisor uses
        self.profile.observe(self.items.len(), self.moves, self.relocations, self.queries);
        self.moves = 0;
        self.relocations = 0;
        self.queries = 0;
        self.cooling = self.cooling.saturating_sub(1);

        let want = self.desired();
        if want == self.backend() { self.pending = None; return self.backend(); }
        let held = match self.pending {
            Some((b, n)) if b == want => n + 1,
            _ => 1,
        };
        self.pending = Some((want, held));
        if held >= self.th.hold_ticks && self.cooling == 0 {
            self.migrate(want);
            self.pending = None;
            self.cooling = self.th.cooldown;
        }
        self.backend()
    }

    /// The backend the current numbers argue for, with every boundary widened in the
    /// direction of travel so a workload sitting on one does not oscillate.
    ///
    /// The policy reads the profile's *observations* rather than calling its `recommend`:
    /// the advisor carries its own compiled-in constants, and the whole point of
    /// [`Thresholds`] is that a calibration file can override them. Delegating would have
    /// silently ignored the calibration — it did, until a test caught it.
    fn desired(&self) -> Backend {
        let n = self.items.len() as f64;
        let m = self.th.margin;
        let cur = self.backend();
        // Leaving brute costs a build, so demand a clearly bigger population than the one
        // that would have kept us there; entering it demands a clearly smaller one.
        let brute_edge = self.th.brute_max as f64 * if cur == Backend::Brute { 1.0 + m } else { 1.0 - m };
        if n <= brute_edge { return Backend::Brute; }
        if self.profile.still_ticks() >= self.th.static_ticks { return Backend::Static; }
        // Query INTENSITY decides, not churn — measured, see `rebuild_query_ratio`. Same
        // widening: once on the grid, stay until the query load drops well under.
        let edge = self.th.rebuild_query_ratio * if cur == Backend::Grid { 1.0 - m } else { 1.0 + m };
        if self.q_per_item > edge { return Backend::Grid; }
        Backend::KeepTree
    }

    fn migrate(&mut self, to: Backend) {
        self.switches += 1;
        self.held = Self::build(to, &self.items, self.world, self.leaf);
        self.dirty = false;
    }

    /// Rebuild a backend from the item list. This is the cost hysteresis exists to avoid
    /// paying twice.
    fn build(to: Backend, items: &[T], world: Aabb, leaf: usize) -> Held<T> {
        match to {
            Backend::Brute => Held::Brute,
            Backend::KeepTree => {
                let mut t = Tree3::new(world, leaf);
                let mut refs = Vec::with_capacity(items.len());
                for it in items { refs.push(t.insert_ref(it.clone()).unwrap_or(ItemRef(u32::MAX))); }
                Held::Keep(Box::new(t), refs)
            }
            Backend::Grid => {
                let levels = MortonGrid3::<T>::levels_for_cell_size(world, (world.w.max(world.h).max(world.d)) / 64.0);
                let mut g = MortonGrid3::new(world, levels);
                for it in items { g.insert(it.clone()); }
                Held::Grid(Box::new(g))
            }
            Backend::Static => Held::Static(Box::new(KdTree3::from_items(leaf, items.to_vec()))),
        }
    }

    /// Bring a stale rebuild-based backend up to date. The keep-index tree is never stale
    /// (that is the point of it), and brute force reads the items directly.
    fn refresh(&mut self) {
        if !self.dirty { return; }
        let b = self.backend();
        if b == Backend::Brute { self.dirty = false; return; }
        self.held = Self::build(b, &self.items, self.world, self.leaf);
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sphere3;

    #[derive(Clone, Copy, Debug)]
    struct P { p: Point3 }
    impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

    fn world() -> Aabb { Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0) }
    fn pt(i: usize) -> Point3 {
        let f = |k: u64| ((i as u64 * k) % 251) as f64;
        Point3::new(f(37), f(53), f(97))
    }
    /// Genuinely independent positions. `pt(t * a + k * b)` looks random but shifts every
    /// item by the SAME amount each tick — a rigid translation, where only ~27% of moves
    /// cross a leaf. That is not "wild movement", and using it made a churn test conclude
    /// the policy was broken when the workload simply was not churny.
    fn scatter(t: usize, k: usize) -> Point3 {
        let mut x = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let mut next = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x >> 40) % 251) as f64 };
        Point3::new(next(), next(), next())
    }

    /// Move one item in the index AND in the brute-force reference. Keeping only one of
    /// them updated is how the first version of these tests "found" a bug in the index.
    fn mv(ix: &mut AdaptiveIndex<P>, all: &mut [P], slot: usize, to: Point3) {
        ix.update(Slot(slot as u32), |c| c.p = to);
        all[slot].p = to;
    }

    /// Whatever backend it happens to be holding, the answers must match brute force —
    /// including immediately after a migration.
    fn assert_matches_brute(ix: &mut AdaptiveIndex<P>, all: &[P]) {
        for (cx, cy, cz, r) in [(50.0, 50.0, 50.0, 40.0), (200.0, 30.0, 120.0, 60.0), (0.0, 0.0, 0.0, 500.0)] {
            let s = Sphere3::new(cx, cy, cz, r);
            let mut want: Vec<(u64, u64, u64)> = all.iter()
                .filter(|q| { let (dx, dy, dz) = (q.p.x - cx, q.p.y - cy, q.p.z - cz); dx * dx + dy * dy + dz * dz <= r * r })
                .map(|q| (q.p.x.to_bits(), q.p.y.to_bits(), q.p.z.to_bits())).collect();
            let mut got: Vec<(u64, u64, u64)> = ix.cull(&s).iter()
                .map(|q| (q.p.x.to_bits(), q.p.y.to_bits(), q.p.z.to_bits())).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "backend {:?} disagreed with brute force", ix.backend());
        }
    }

    #[test]
    fn starts_brute_and_grows_into_a_tree() {
        let th = Thresholds { brute_max: 100, hold_ticks: 2, cooldown: 0, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
        let mut all = Vec::new();
        for i in 0..80 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        for _ in 0..10 { ix.tick(); }
        assert_eq!(ix.backend(), Backend::Brute, "80 items should not be indexed");
        assert_matches_brute(&mut ix, &all);

        // Past the widened boundary it must move to the keep-index tree, and still agree.
        for i in 80..400 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        for t in 0..20 {
            let k = t % 100;
            let p = all[k].p;
            mv(&mut ix, &mut all, k, Point3::new(p.x + 0.03, p.y, p.z));
            ix.tick();
        }
        assert_eq!(ix.backend(), Backend::KeepTree, "400 items should be indexed");
        assert_matches_brute(&mut ix, &all);
    }

    /// A QUERY-HEAVY workload — roughly one cull per item per tick, the shape SPH has —
    /// is where a rebuilt grid beats keeping the tree. Churn alone never flips it (the
    /// calibration swept both: churn moved keep's margin from 116x to 6.4x but never past
    /// 1.0), so this test drives the variable that actually decides.
    #[test]
    fn query_heavy_workload_switches_to_the_rebuilt_grid() {
        let th = Thresholds { brute_max: 10, hold_ticks: 2, cooldown: 0, rebuild_query_ratio: 0.1, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
        let mut all = Vec::new();
        for i in 0..300 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        for t in 0..80 {
            for k in 0..300 { mv(&mut ix, &mut all, k, scatter(t, k)); }
            for k in 0..120 { let _ = ix.cull(&Sphere3::new(k as f64, 40.0, 40.0, 10.0)); } // 0.4 queries/item
            ix.tick();
        }
        assert!(ix.queries_per_item() > 0.3, "workload is not query-heavy: {:.3}", ix.queries_per_item());
        assert_eq!(ix.backend(), Backend::Grid, "one query per few items should rebuild");
        assert_matches_brute(&mut ix, &all);
    }

    #[test]
    fn settling_switches_to_the_build_once_backend() {
        let th = Thresholds { brute_max: 10, static_ticks: 5, hold_ticks: 2, cooldown: 0, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
        let mut all = Vec::new();
        for i in 0..300 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        // A moving phase, then everything settles.
        // gentle movement: stays inside its leaf, so the keep-index tree is right
        for t in 0..20 {
            let k = t % 300;
            let p = all[k].p;
            mv(&mut ix, &mut all, k, Point3::new(p.x + 0.05, p.y, p.z));
            ix.tick();
        }
        assert_eq!(ix.backend(), Backend::KeepTree);
        for _ in 0..40 { let _ = ix.cull(&Sphere3::new(10.0, 10.0, 10.0, 20.0)); ix.tick(); }
        assert_eq!(ix.backend(), Backend::Static, "a settled workload should build once");
        assert_matches_brute(&mut ix, &all);
    }

    /// The property that makes hysteresis worth having: sitting exactly on a boundary must
    /// not migrate every tick. Without the margin and the hold, this flaps.
    #[test]
    fn does_not_flap_on_the_boundary() {
        let th = Thresholds { brute_max: 200, hold_ticks: 30, cooldown: 120, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
        for i in 0..200 { ix.insert(P { p: pt(i) }); }
        // Hover at exactly the threshold for a long time, moving items every tick.
        for t in 0..500 {
            ix.update(Slot((t % 200) as u32), |c| c.p = pt(t + 1));
            let _ = ix.cull(&Sphere3::new(50.0, 50.0, 50.0, 30.0));
            ix.tick();
        }
        assert!(ix.switch_count() <= 1, "flapped {} times on the boundary", ix.switch_count());
    }

    #[test]
    fn calibration_round_trips_and_ignores_unknown_keys() {
        let th = Thresholds { brute_max: 777, high_churn: 0.42, hold_ticks: 9, ..Default::default() };
        let parsed = Thresholds::parse(&(th.to_text() + "future_key = 12\n# a comment\n"));
        assert_eq!(parsed.brute_max, 777);
        assert!((parsed.high_churn - 0.42).abs() < 1e-9);
        assert_eq!(parsed.hold_ticks, 9);
    }
}
