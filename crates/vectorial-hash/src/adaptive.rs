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
//! | [`MortonGrid3`] | items move a LOT | a flat grid answers dense queries with fewer descents; it now **keeps in place** too (`MortonGrid3::update`), so it is no longer paying a full rebuild per mutation |
//! | [`KdTree3`] | nothing has moved for a while | build-once, best query on skewed data |
//!
//! **Hysteresis is the whole difficulty.** A naive "pick the best for the current numbers"
//! flaps: at the boundary it rebuilds every frame and loses to *both* candidates. So a
//! switch needs the new choice to hold for [`Thresholds::hold_ticks`] consecutive ticks,
//! the boundaries are widened by [`Thresholds::margin`] in the direction you are moving,
//! and there is a hard [`Thresholds::cooldown`] after every migration. Rebuilding is not
//! free and the policy is written to respect that.
//!
//! **Handles are slots, and they survive removal.** `insert` hands back a [`Slot`] that
//! stays valid through every migration *and* through other items being removed: the item
//! list is a slot table with a free list, not a `Vec` that gets swap-removed. Removing item
//! 7 must not silently repoint whoever held the handle to the last item, so it does not —
//! slot 7 becomes a hole and the next `insert` gets it back.
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
    ///
    /// **This number is now stale in a knowable direction and has NOT been re-derived.** It
    /// was calibrated when the grid backend could only rebuild, so it prices a full refill
    /// against every mutation. The grid keeps in place now, so the cost it is guarding
    /// against is much smaller and the true crossover must be *lower* — the grid should be
    /// reachable at less query load than 0.1 says. Re-deriving it needs `examples/calibrate`
    /// to grow a third arm (keep-tree vs rebuild-grid vs keep-grid); until then the default
    /// is conservative rather than wrong: it reaches for the grid later than it should, never
    /// earlier.
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
    Grid(Box<MortonGrid3<Tagged<T>>>),
    Static(Box<KdTree3<T>>),
}

/// An item in the grid backend, carrying the [`Slot`] it belongs to.
///
/// `MortonGrid3::update` finds an item by scanning its old cell for one matching a predicate,
/// which needs a way to tell two items apart. The tree backend gets that from `ItemRef`; a
/// uniform grid has no handles, and matching on position alone would move the wrong item
/// whenever two share a position exactly — rare, and exactly the case a settled fluid produces.
/// One `u32` per item buys the identity, and the caller never sees it.
#[derive(Clone)]
struct Tagged<T> { slot: u32, item: T }
impl<T: Positioned3> Positioned3 for Tagged<T> {
    fn position(&self) -> Point3 { self.item.position() }
}

/// A stable handle into an [`AdaptiveIndex`]. Survives migrations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Slot(pub u32);

/// An index that picks its own structure. See the module docs.
pub struct AdaptiveIndex<T: Positioned3 + Clone> {
    /// The source of truth. Every backend is a view onto this, rebuilt on migration.
    ///
    /// A **slot table**, not a list: `Slot` is the index into it, and the type promises that
    /// handle survives. A swap-remove would keep it dense at the cost of silently moving
    /// whichever item happened to be last — a different caller's handle. So a removed slot
    /// becomes a hole, goes on `free`, and is handed back out by the next `insert`.
    items: Vec<Option<T>>,
    /// Retired slots, newest first. Reusing them is what keeps the table from growing
    /// without bound under an insert/remove churn that never changes the live count.
    free: Vec<u32>,
    /// Live items. Not `items.len()`, which counts holes — and the policy turns on this one.
    live: usize,
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
            items: Vec::new(), free: Vec::new(), live: 0, world, leaf: leaf.max(1), held: Held::Brute,
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
    pub fn len(&self) -> usize { self.live }
    pub fn is_empty(&self) -> bool { self.live == 0 }
    /// Slots allocated, holes included. `len()` is what you almost always want; this one is
    /// here so a caller can see the table is being recycled rather than growing.
    pub fn slots(&self) -> usize { self.items.len() }
    /// The item behind a handle, or `None` if that slot is a hole.
    pub fn get(&self, s: Slot) -> Option<&T> { self.items.get(s.0 as usize).and_then(|o| o.as_ref()) }
    pub fn thresholds(&self) -> &Thresholds { &self.th }
    pub fn profile(&self) -> &SpatialProfile { &self.profile }
    /// Smoothed queries per item per tick — what the keep-vs-rebuild choice turns on.
    pub fn queries_per_item(&self) -> f64 { self.q_per_item }

    /// Add an item, reusing a retired slot if there is one.
    pub fn insert(&mut self, item: T) -> Slot {
        let slot = match self.free.pop() {
            Some(i) => i,
            None => { self.items.push(None); (self.items.len() - 1) as u32 }
        };
        let mut stale = false;
        match &mut self.held {
            Held::Keep(t, refs) => {
                let r = match t.insert_ref(item.clone()) { Some(r) => r, None => { stale = true; ItemRef(u32::MAX) } };
                // `refs` is indexed BY SLOT, holes included, so a recycled slot writes in
                // place rather than pushing and shifting everyone after it.
                if refs.len() <= slot as usize { refs.resize(slot as usize + 1, ItemRef(u32::MAX)); }
                refs[slot as usize] = r;
            }
            Held::Grid(g) => { if !g.insert(Tagged { slot, item: item.clone() }) { stale = true; } }
            Held::Brute => {}
            Held::Static(_) => stale = true, // a build-once backend cannot take one more
        }
        self.dirty |= stale;
        self.items[slot as usize] = Some(item);
        self.live += 1;
        Slot(slot)
    }

    /// Remove the item behind `s` and return it, or `None` if the slot is already empty.
    /// The slot is retired and a later `insert` may hand it back out; every OTHER handle
    /// keeps pointing at the same item, which is the property this whole slot table exists
    /// for. Only the keep-index tree can drop one item in place (`remove_ref`) — the grid
    /// and the build-once k-d tree have no removal at all, so they are marked stale and
    /// rebuilt on the next query, exactly as they are for a move.
    pub fn remove(&mut self, s: Slot) -> Option<T> {
        let taken = self.items.get_mut(s.0 as usize)?.take()?;
        self.live -= 1;
        self.free.push(s.0);
        match &mut self.held {
            Held::Keep(t, refs) => match refs.get(s.0 as usize).copied().filter(|r| r.0 != u32::MAX) {
                Some(r) => { t.remove_ref(r); refs[s.0 as usize] = ItemRef(u32::MAX); }
                None => self.dirty = true,
            },
            Held::Grid(g) => { g.remove(taken.position(), |c| c.slot == s.0); }
            Held::Brute => {}
            _ => self.dirty = true,
        }
        Some(taken)
    }

    /// Move (or otherwise mutate) one item. The keep-index backend takes the O(1) path;
    /// the rebuild-based ones just note that they are stale.
    pub fn update<F: FnOnce(&mut T)>(&mut self, s: Slot, f: F) {
        let Some(it) = self.items.get_mut(s.0 as usize).and_then(|o| o.as_mut()) else { return };
        // Captured BEFORE the mutation: the grid finds an item by where it used to be.
        let was = it.position();
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
            // The grid keeps in place too, since `MortonGrid3::update` exists: it does not
            // need a handle, only where the item was. A `Missing` means the grid and the item
            // list disagreed, which is recoverable by rebuilding rather than by pretending.
            Held::Grid(g) => {
                let slot = s.0;
                if g.update(was, |c| c.slot == slot, |c| c.item = item).is_missing() { self.dirty = true; }
            }
            // Whatever is left cannot be maintained (a build-once k-d tree) or does not need
            // to be (the brute scan reads `items` directly). It also cannot tell us whether
            // that move crossed a leaf, so it must NOT claim a relocation: reporting every
            // move as one made the policy jump to the grid on its very first frame, before it
            // had ever seen a real crossing. Churn is learned on the tree.
            _ => self.dirty = true,
        }
    }

    /// Everything inside `shape`. Rebuilds a stale backend first, so a caller never sees
    /// a partially-updated index.
    pub fn cull<S: Shape3>(&mut self, shape: &S) -> Vec<&T> {
        self.queries += 1;
        self.refresh();
        match &self.held {
            Held::Brute => self.items.iter().flatten().filter(|it| shape.contains_point(it.position())).collect(),
            Held::Keep(t, _) => t.cull(shape),
            Held::Grid(g) => g.cull(shape).into_iter().map(|t| &t.item).collect(),
            Held::Static(k) => k.cull(shape),
        }
    }

    /// k nearest to `q`, as `(distance, &item)` sorted by distance.
    pub fn knn(&mut self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        self.queries += 1;
        self.refresh();
        match &self.held {
            Held::Brute => {
                let mut v: Vec<(f64, &T)> = self.items.iter().flatten().map(|it| {
                    let p = it.position();
                    let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z);
                    ((dx * dx + dy * dy + dz * dz).sqrt(), it)
                }).collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                v.truncate(k);
                v
            }
            Held::Keep(t, _) => t.knn(q, k),
            Held::Grid(g) => g.knn(q, k).into_iter().map(|(d, t)| (d, &t.item)).collect(),
            Held::Static(kd) => kd.knn(q, k),
        }
    }

    /// Close the frame: feed the observed rates to the advisor and migrate if the
    /// workload has genuinely changed. Call once per tick.
    pub fn tick(&mut self) -> Backend {
        let qpi = if self.live == 0 { 0.0 } else { self.queries as f64 / self.live as f64 };
        self.q_per_item += 0.1 * (qpi - self.q_per_item); // same EMA weight the advisor uses
        self.profile.observe(self.live, self.moves, self.relocations, self.queries);
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
        let n = self.live as f64;
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
    fn build(to: Backend, items: &[Option<T>], world: Aabb, leaf: usize) -> Held<T> {
        match to {
            Backend::Brute => Held::Brute,
            Backend::KeepTree => {
                let mut t = Tree3::new(world, leaf);
                // One entry per SLOT, holes carried through as dead refs, so `refs[slot]`
                // survives a rebuild. Skipping holes here would shift every handle past the
                // first removal — the exact bug the slot table is built to prevent.
                let refs = items.iter().map(|o| match o {
                    Some(it) => t.insert_ref(it.clone()).unwrap_or(ItemRef(u32::MAX)),
                    None => ItemRef(u32::MAX),
                }).collect();
                Held::Keep(Box::new(t), refs)
            }
            Backend::Grid => {
                let levels = MortonGrid3::<Tagged<T>>::levels_for_cell_size(world, (world.w.max(world.h).max(world.d)) / 64.0);
                let mut g = MortonGrid3::new(world, levels);
                for (slot, it) in items.iter().enumerate() {
                    if let Some(v) = it { g.insert(Tagged { slot: slot as u32, item: v.clone() }); }
                }
                Held::Grid(Box::new(g))
            }
            Backend::Static => Held::Static(Box::new(KdTree3::from_items(leaf, items.iter().flatten().cloned().collect()))),
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
    trait Bits3 { fn to_bits3(&self) -> (u64, u64, u64); }
    impl Bits3 for Point3 { fn to_bits3(&self) -> (u64, u64, u64) { (self.x.to_bits(), self.y.to_bits(), self.z.to_bits()) } }
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

    /// Drive the index onto a named backend and assert it got there, so a test about
    /// removal on the grid cannot quietly pass while sitting on brute force.
    fn force(ix: &mut AdaptiveIndex<P>, all: &mut [Option<P>], want: Backend) {
        for t in 0..80 {
            if ix.backend() == want { return; }
            // SOMETHING has to move, or `still_ticks` routes every destination to Static —
            // which is how the first version of this helper "reached" the grid. The
            // reference moves with it, of course.
            if want != Backend::Static {
                if let Some(i) = all.iter().position(|o| o.is_some()) {
                    let to = scatter(t, i);
                    ix.update(Slot(i as u32), |c| c.p = to);
                    all[i] = Some(P { p: to });
                }
            }
            // Query intensity is what buys the grid; one cull per item per tick is well
            // over any sane rebuild_query_ratio.
            if want == Backend::Grid { for k in 0..ix.len() { let p = pt(k); let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 8.0)); } }
            ix.tick();
        }
        assert_eq!(ix.backend(), want, "could not reach {want:?}");
    }

    /// Remove on every backend, against brute force, with the reference kept in step. The
    /// grid and the k-d tree cannot remove at all, so this is really asking whether the
    /// stale-then-rebuild path notices — and it is the whole reason `remove` exists rather
    /// than callers being told to rebuild.
    #[test]
    fn remove_agrees_with_brute_force_on_every_backend() {
        for want in [Backend::Brute, Backend::KeepTree, Backend::Grid, Backend::Static] {
            let th = Thresholds { brute_max: if want == Backend::Brute { 400 } else { 40 },
                static_ticks: 6, hold_ticks: 1, cooldown: 0, ..Default::default() };
            let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
            let mut slots = Vec::new();
            let mut all: Vec<Option<P>> = Vec::new();
            for i in 0..200 { let it = P { p: pt(i) }; slots.push(ix.insert(it)); all.push(Some(it)); }
            force(&mut ix, &mut all, want);

            // Every third item, back to front, so the removals interleave with survivors.
            for i in (0..200).rev().step_by(3) {
                let got = ix.remove(slots[i]).expect("slot was live");
                assert_eq!(got.p.to_bits3(), all[i].unwrap().p.to_bits3(), "wrong item returned on {want:?}");
                all[i] = None;
                assert_eq!(ix.len(), all.iter().flatten().count(), "len drifted on {want:?}");
            }
            let live: Vec<P> = all.iter().flatten().copied().collect();
            assert_matches_brute(&mut ix, &live);
            // 199 is the first one the loop above took; asking again must find a hole.
            assert!(ix.remove(slots[199]).is_none(), "removing a hole twice must be None on {want:?}");
        }
    }

    /// The property the slot table exists for. A swap-remove would keep `items` dense and
    /// silently repoint whichever handle belonged to the last item — so this checks the
    /// SURVIVORS by handle, which is the thing that would break, not the answer set.
    #[test]
    fn removal_does_not_disturb_other_handles_and_recycles_the_slot() {
        let th = Thresholds { brute_max: 20, hold_ticks: 1, cooldown: 0, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
        let slots: Vec<Slot> = (0..60).map(|i| ix.insert(P { p: pt(i) })).collect();
        let mut all: Vec<Option<P>> = (0..60).map(|i| Some(P { p: pt(i) })).collect();
        force(&mut ix, &mut all, Backend::KeepTree);
        ix.remove(slots[7]);
        ix.remove(slots[31]);
        assert_eq!(ix.len(), 58);

        // Every survivor still answers as itself: move it via its handle and find it there.
        for (i, s) in slots.iter().enumerate() {
            if i == 7 || i == 31 { continue; }
            let to = Point3::new(200.0 + (i % 7) as f64, 200.0, 200.0);
            ix.update(*s, |c| c.p = to);
            let hit = ix.cull(&Sphere3::new(to.x, to.y, to.z, 0.25));
            assert!(hit.iter().any(|q| q.p.to_bits3() == to.to_bits3()), "handle {i} lost its item");
        }
        // And the retired slots come back rather than the table growing for ever.
        let before = ix.slots();
        let a = ix.insert(P { p: pt(500) });
        let b = ix.insert(P { p: pt(501) });
        assert_eq!(ix.slots(), before, "recycled inserts should not allocate new slots");
        assert!([a, b].contains(&slots[7]) && [a, b].contains(&slots[31]), "the freed slots were not reused");
        assert_eq!(ix.len(), 60);
    }

    /// Insert/remove churn that never changes the live count must not grow the table, and
    /// must not confuse the policy: `len()` counts items, not slots.
    #[test]
    fn churn_at_a_constant_population_neither_grows_nor_misleads() {
        let th = Thresholds { brute_max: 20, hold_ticks: 1, cooldown: 0, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
        let mut slots: Vec<Slot> = (0..80).map(|i| ix.insert(P { p: pt(i) })).collect();
        let mut seed: Vec<Option<P>> = (0..80).map(|i| Some(P { p: pt(i) })).collect();
        force(&mut ix, &mut seed, Backend::KeepTree);
        let table = ix.slots();
        for t in 0..300 {
            let i = t % slots.len();
            ix.remove(slots[i]);
            slots[i] = ix.insert(P { p: scatter(t, i) });
            if t % 4 == 0 { ix.tick(); }
        }
        assert_eq!(ix.len(), 80);
        assert_eq!(ix.slots(), table, "the slot table grew under constant-population churn");
        let all: Vec<P> = slots.iter().filter_map(|s| ix.get(*s).copied()).collect();
        assert_eq!(all.len(), 80);
        assert_matches_brute(&mut ix, &all);
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
