//! `AdaptiveIndex2` — the 2D twin of [`AdaptiveIndex`](crate::adaptive::AdaptiveIndex).
//!
//! Same policy, same [`Thresholds`], same [`Slot`] handles and the same slot table with a
//! free list. What changes is the cast, and the 2D measurements say it should:
//!
//! | backend | 3D holds | 2D holds | why the difference |
//! | --- | --- | --- | --- |
//! | keep | [`Tree3`](crate::Tree3) | [`Tree`](crate::Tree) | the binary tree wins maintain in both |
//! | grid | [`MortonGrid3`](crate::MortonGrid3) | [`MortonGrid`](crate::MortonGrid) | rebuild under heavy query load |
//! | static | [`KdTree3`](crate::KdTree3) | [`KdTree2`](crate::KdTree2) | build-once, best on skewed data |
//!
//! Worth knowing before reaching for this: in 2D the margins are **much thinner**. The kept
//! binary tree leads its rivals by 1.6-7.6x on 3D maintain but sits 4-10% behind a
//! `QuadTree` in 2D, and at 50k moving points the Morton rebuild takes both columns
//! outright (`examples/decision2d`). An adaptive index earns its keep where the winner
//! actually changes with the workload — which is still true here, just by less.
//!
//! `Backend`, `Slot` and `Thresholds` are dimension-agnostic and are re-exported from
//! [`crate::adaptive`] rather than duplicated: one calibration file configures both.

use crate::adaptive::{Backend, Slot, Thresholds};
use crate::advisor::SpatialProfile;
use crate::kdtree2::KdTree2;
use crate::morton::MortonGrid;
use crate::tree::{Crossing2, Tree};
use crate::{ItemRef, Point, Positioned, Rect, Shape};

enum Held2<T: Positioned> {
    Brute,
    Keep(Box<Tree<T>>, Vec<ItemRef>),
    Grid(Box<MortonGrid<T>>),
    Static(Box<KdTree2<T>>),
}

/// An index that picks its own structure. See the module docs.
pub struct AdaptiveIndex2<T: Positioned + Clone> {
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
    world: Rect,
    leaf: usize,
    held: Held2<T>,
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
    /// Smoothed moves per item per tick — see the 3D twin.
    m_per_item: f64,
    /// The live count the grid backend's cell size was derived from.
    ///
    /// A grid's levels are chosen when it is built, and with the keep path it is never
    /// rebuilt — so a grid built for 60 items was still serving 20 000 of them at 2 500 per
    /// cell, three times slower than it needed to be. Nothing marked it dirty because nothing
    /// was wrong with its CONTENTS; what had gone stale was its geometry.
    grid_for: usize,
}

impl<T: Positioned + Clone> AdaptiveIndex2<T> {
    /// `world` bounds the grid backend; `leaf` is the tree leaf capacity.
    pub fn new(world: Rect, leaf: usize) -> Self {
        Self::with_thresholds(world, leaf, Thresholds::from_env())
    }

    pub fn with_thresholds(world: Rect, leaf: usize, th: Thresholds) -> Self {
        AdaptiveIndex2 {
            items: Vec::new(), free: Vec::new(), live: 0, world, leaf: leaf.max(1), held: Held2::Brute,
            profile: SpatialProfile::default(), th, pending: None, cooling: 0,
            dirty: false, switches: 0, moves: 0, relocations: 0, queries: 0, q_per_item: 0.0,
            m_per_item: 0.0, grid_for: 0,
        }
    }

    /// Which structure is currently in use.
    pub fn backend(&self) -> Backend {
        match self.held { Held2::Brute => Backend::Brute, Held2::Keep(..) => Backend::KeepTree, Held2::Grid(_) => Backend::Grid, Held2::Static(_) => Backend::Static }
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
            Held2::Keep(t, refs) => {
                let r = match t.insert_ref(item.clone()) { Some(r) => r, None => { stale = true; ItemRef(u32::MAX) } };
                // `refs` is indexed BY SLOT, holes included, so a recycled slot writes in
                // place rather than pushing and shifting everyone after it.
                if refs.len() <= slot as usize { refs.resize(slot as usize + 1, ItemRef(u32::MAX)); }
                refs[slot as usize] = r;
            }
            Held2::Grid(g) => { if !g.insert(item.clone()) { stale = true; } }
            Held2::Brute => {}
            Held2::Static(_) => stale = true, // a build-once backend cannot take one more
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
            Held2::Keep(t, refs) => match refs.get(s.0 as usize).copied().filter(|r| r.0 != u32::MAX) {
                Some(r) => { t.remove_ref(r); refs[s.0 as usize] = ItemRef(u32::MAX); }
                None => self.dirty = true,
            },
            Held2::Brute => {}
            _ => self.dirty = true,
        }
        Some(taken)
    }

    /// Move (or otherwise mutate) one item. The keep-index backend takes the O(1) path;
    /// the rebuild-based ones just note that they are stale.
    pub fn update<F: FnOnce(&mut T)>(&mut self, s: Slot, f: F) {
        let Some(it) = self.items.get_mut(s.0 as usize).and_then(|o| o.as_mut()) else { return };
        f(it);
        let item = it.clone();
        self.moves += 1;
        match &mut self.held {
            Held2::Keep(t, refs) => {
                if let Some(r) = refs.get(s.0 as usize).copied().filter(|r| r.0 != u32::MAX) {
                    // `_tracked` so the policy learns the REAL relocation rate: how often a
                    // move actually crosses a leaf, which is the number that decides
                    // whether keeping the index still beats rebuilding it.
                    match t.update_ref_tracked(r, |c| *c = item) {
                        Crossing2::Stayed(_) => {}
                        Crossing2::Moved { .. } => self.relocations += 1,
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
    pub fn cull<S: Shape>(&mut self, shape: &S) -> Vec<&T> {
        self.queries += 1;
        self.refresh();
        match &self.held {
            Held2::Brute => self.items.iter().flatten().filter(|it| shape.contains_point(it.position())).collect(),
            Held2::Keep(t, _) => t.cull(shape),
            Held2::Grid(g) => g.cull(shape),
            Held2::Static(k) => k.cull(shape),
        }
    }

    /// k nearest to `q`, as `(distance, &item)` sorted by distance.
    pub fn knn(&mut self, q: Point, k: usize) -> Vec<(f64, &T)> {
        self.queries += 1;
        self.refresh();
        match &self.held {
            Held2::Brute => {
                let mut v: Vec<(f64, &T)> = self.items.iter().flatten().map(|it| {
                    let p = it.position();
                    let (dx, dy) = (p.x - q.x, p.y - q.y);
                    ((dx * dx + dy * dy).sqrt(), it)
                }).collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                v.truncate(k);
                v
            }
            Held2::Keep(t, _) => t.knn(q, k),
            Held2::Grid(g) => g.knn(q, k),
            Held2::Static(kd) => kd.knn(q, k),
        }
    }

    /// Close the frame: feed the observed rates to the advisor and migrate if the
    /// workload has genuinely changed. Call once per tick.
    pub fn tick(&mut self) -> Backend {
        let qpi = if self.live == 0 { 0.0 } else { self.queries as f64 / self.live as f64 };
        self.q_per_item += 0.1 * (qpi - self.q_per_item); // same EMA weight the advisor uses
        let mpi = if self.live == 0 { 0.0 } else { self.moves as f64 / self.live as f64 };
        self.m_per_item += 0.1 * (mpi - self.m_per_item);
        self.profile.observe(self.live, self.moves, self.relocations, self.queries);
        self.moves = 0;
        self.relocations = 0;
        self.queries = 0;
        self.cooling = self.cooling.saturating_sub(1);

        // A grid whose population has drifted far from what its cells were sized for is
        // stale in geometry rather than in contents, which nothing else would notice. The
        // rebuild goes through `build`, which re-derives the levels.
        if self.backend() == Backend::Grid && self.grid_for > 0 {
            let (a, b) = (self.live.max(1) as f64, self.grid_for as f64);
            if a / b > 4.0 || b / a > 4.0 { self.dirty = true; self.grid_for = self.live; }
        }
        let (want, decisive) = self.desired_with_confidence();
        if want == self.backend() { self.pending = None; return self.backend(); }
        let held = match self.pending {
            Some((b, n)) if b == want => n + 1,
            _ => 1,
        };
        self.pending = Some((want, held));
        if decisive || (held >= self.th.hold_ticks && self.cooling == 0) {
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
    /// See the 3D twin — the hysteresis is skipped when the numbers are decisive.
    fn desired_with_confidence(&self) -> (Backend, bool) {
        let want = self.desired();
        if want == self.backend() { return (want, false); }
        let f = self.th.decisive_factor;
        let n = self.live as f64;
        let decisive = match want {
            Backend::Brute => n * f <= self.th.brute_max as f64 || self.q_per_item * n * f <= self.m_per_item * self.th.scan_budget,
            Backend::Grid => self.q_per_item >= self.th.rebuild_query_ratio * f,
            Backend::KeepTree => self.q_per_item * n >= self.m_per_item * self.th.scan_budget * f && n >= self.th.brute_max as f64 * f,
            Backend::Static => self.profile.still_ticks() as f64 >= self.th.static_ticks as f64 * f,
        };
        (want, decisive)
    }

    fn desired(&self) -> Backend {
        let n = self.live as f64;
        let m = self.th.margin;
        let cur = self.backend();
        // Leaving brute costs a build, so demand a clearly bigger population than the one
        // that would have kept us there; entering it demands a clearly smaller one.
        let brute_edge = self.th.brute_max as f64 * if cur == Backend::Brute { 1.0 + m } else { 1.0 - m };
        if n <= brute_edge { return Backend::Brute; }
        // ...and a scan also wins when nobody is asking much — see the 3D twin's `scan_budget`.
        let scan_edge = self.m_per_item * self.th.scan_budget * if cur == Backend::Brute { 1.0 + m } else { 1.0 - m };
        if self.q_per_item * n < scan_edge { return Backend::Brute; }
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
        self.grid_for = self.live; // whatever the grid's cells were just sized for
        self.dirty = false;
    }

    /// Rebuild a backend from the item list. This is the cost hysteresis exists to avoid
    /// paying twice.
    fn build(to: Backend, items: &[Option<T>], world: Rect, leaf: usize) -> Held2<T> {
        match to {
            Backend::Brute => Held2::Brute,
            Backend::KeepTree => {
                let mut t = Tree::new(world, leaf);
                // One entry per SLOT, holes carried through as dead refs, so `refs[slot]`
                // survives a rebuild. Skipping holes here would shift every handle past the
                // first removal — the exact bug the slot table is built to prevent.
                let refs = items.iter().map(|o| match o {
                    Some(it) => t.insert_ref(it.clone()).unwrap_or(ItemRef(u32::MAX)),
                    None => ItemRef(u32::MAX),
                }).collect();
                Held2::Keep(Box::new(t), refs)
            }
            Backend::Grid => {
                // Cells sized by occupancy, not by a fixed fraction of the world — see the 3D
                // twin, where `world_max / 64` meant 0.08 items per cell at 20k population.
                let live = items.iter().flatten().count().max(1);
                let per_axis = (live as f64 / crate::adaptive::GRID_TARGET_PER_CELL).sqrt().max(1.0);
                let levels = (per_axis.log2().round().max(1.0) as u32).min(12);
                let mut g = MortonGrid::new(world, levels);
                for it in items.iter().flatten() { g.insert(it.clone()); }
                Held2::Grid(Box::new(g))
            }
            Backend::Static => Held2::Static(Box::new(KdTree2::from_items(leaf, items.iter().flatten().cloned().collect()))),
        }
    }

    /// Bring a stale rebuild-based backend up to date. The keep-index tree is never stale
    /// (that is the point of it), and brute force reads the items directly.
    fn refresh(&mut self) {
        if !self.dirty { return; }
        let b = self.backend();
        if b == Backend::Brute { self.dirty = false; return; }
        self.held = Self::build(b, &self.items, self.world, self.leaf);
        self.grid_for = self.live;
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Circle;

    #[derive(Clone, Copy, Debug)]
    struct P { p: Point }
    impl Positioned for P { fn position(&self) -> Point { self.p } }

    fn world() -> Rect { Rect::new(0.0, 0.0, 256.0, 256.0) }
    trait Bits2 { fn to_bits2(&self) -> (u64, u64); }
    impl Bits2 for Point { fn to_bits2(&self) -> (u64, u64) { (self.x.to_bits(), self.y.to_bits()) } }
    fn pt(i: usize) -> Point {
        let f = |k: u64| ((i as u64 * k) % 251) as f64;
        Point::new(f(37), f(53))
    }
    /// Genuinely independent positions. `pt(t * a + k * b)` looks random but shifts every
    /// item by the SAME amount each tick — a rigid translation, where only ~27% of moves
    /// cross a leaf. That is not "wild movement", and using it made a churn test conclude
    /// the policy was broken when the workload simply was not churny.
    fn scatter(t: usize, k: usize) -> Point {
        let mut x = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let mut next = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x >> 40) % 251) as f64 };
        Point::new(next(), next())
    }

    /// Move one item in the index AND in the brute-force reference. Keeping only one of
    /// them updated is how the first version of these tests "found" a bug in the index.
    fn mv(ix: &mut AdaptiveIndex2<P>, all: &mut [P], slot: usize, to: Point) {
        ix.update(Slot(slot as u32), |c| c.p = to);
        all[slot].p = to;
    }

    /// Whatever backend it happens to be holding, the answers must match brute force —
    /// including immediately after a migration.
    fn assert_matches_brute(ix: &mut AdaptiveIndex2<P>, all: &[P]) {
        for (cx, cy, r) in [(50.0, 50.0, 40.0), (200.0, 30.0, 60.0), (0.0, 0.0, 500.0)] {
            let s = Circle::new(Point::new(cx, cy), r);
            let mut want: Vec<(u64, u64)> = all.iter()
                .filter(|q| { let (dx, dy) = (q.p.x - cx, q.p.y - cy); dx * dx + dy * dy <= r * r })
                .map(|q| (q.p.x.to_bits(), q.p.y.to_bits())).collect();
            let mut got: Vec<(u64, u64)> = ix.cull(&s).iter()
                .map(|q| (q.p.x.to_bits(), q.p.y.to_bits())).collect();
            want.sort(); got.sort();
            assert_eq!(want, got, "backend {:?} disagreed with brute force", ix.backend());
        }
    }

    /// Drive the index onto a named backend and assert it got there, so a test about
    /// removal on the grid cannot quietly pass while sitting on brute force.
    fn force(ix: &mut AdaptiveIndex2<P>, all: &mut [Option<P>], want: Backend) {
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
            // The tree needs query load too, not just population — see the 3D twin's
            // `scan_budget`: a crowd nobody queries is a crowd a linear scan should serve.
            match want {
                Backend::Grid => for k in 0..ix.len() { let p = pt(k); let _ = ix.cull(&Circle::new(Point::new(p.x, p.y), 8.0)); },
                Backend::KeepTree => for k in 0..2 { let p = pt(k); let _ = ix.cull(&Circle::new(Point::new(p.x, p.y), 8.0)); },
                _ => {}
            }
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
            let mut ix = AdaptiveIndex2::with_thresholds(world(), 8, th);
            let mut slots = Vec::new();
            let mut all: Vec<Option<P>> = Vec::new();
            for i in 0..200 { let it = P { p: pt(i) }; slots.push(ix.insert(it)); all.push(Some(it)); }
            force(&mut ix, &mut all, want);

            // Every third item, back to front, so the removals interleave with survivors.
            for i in (0..200).rev().step_by(3) {
                let got = ix.remove(slots[i]).expect("slot was live");
                assert_eq!(got.p.to_bits2(), all[i].unwrap().p.to_bits2(), "wrong item returned on {want:?}");
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
        let mut ix = AdaptiveIndex2::with_thresholds(world(), 8, th);
        let slots: Vec<Slot> = (0..60).map(|i| ix.insert(P { p: pt(i) })).collect();
        let mut all: Vec<Option<P>> = (0..60).map(|i| Some(P { p: pt(i) })).collect();
        force(&mut ix, &mut all, Backend::KeepTree);
        ix.remove(slots[7]);
        ix.remove(slots[31]);
        assert_eq!(ix.len(), 58);

        // Every survivor still answers as itself: move it via its handle and find it there.
        for (i, s) in slots.iter().enumerate() {
            if i == 7 || i == 31 { continue; }
            let to = Point::new(200.0 + (i % 7) as f64, 200.0);
            ix.update(*s, |c| c.p = to);
            let hit = ix.cull(&Circle::new(Point::new(to.x, to.y), 0.25));
            assert!(hit.iter().any(|q| q.p.to_bits2() == to.to_bits2()), "handle {i} lost its item");
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
        let mut ix = AdaptiveIndex2::with_thresholds(world(), 8, th);
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
        let mut ix = AdaptiveIndex2::with_thresholds(world(), 8, th);
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
            mv(&mut ix, &mut all, k, Point::new(p.x + 0.03, p.y));
            let _ = ix.cull(&Circle::new(p, 12.0)); // queried, or a scan is the right answer
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
        let mut ix = AdaptiveIndex2::with_thresholds(world(), 8, th);
        let mut all = Vec::new();
        for i in 0..300 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        for t in 0..80 {
            for k in 0..300 { mv(&mut ix, &mut all, k, scatter(t, k)); }
            for k in 0..120 { let _ = ix.cull(&Circle::new(Point::new(k as f64, 40.0), 10.0)); } // 0.4 queries/item
            ix.tick();
        }
        assert!(ix.queries_per_item() > 0.3, "workload is not query-heavy: {:.3}", ix.queries_per_item());
        assert_eq!(ix.backend(), Backend::Grid, "one query per few items should rebuild");
        assert_matches_brute(&mut ix, &all);
    }

    #[test]
    fn settling_switches_to_the_build_once_backend() {
        let th = Thresholds { brute_max: 10, static_ticks: 5, hold_ticks: 2, cooldown: 0, ..Default::default() };
        let mut ix = AdaptiveIndex2::with_thresholds(world(), 8, th);
        let mut all = Vec::new();
        for i in 0..300 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        // A moving phase, then everything settles.
        // gentle movement: stays inside its leaf, so the keep-index tree is right
        for t in 0..20 {
            let k = t % 300;
            let p = all[k].p;
            mv(&mut ix, &mut all, k, Point::new(p.x + 0.05, p.y));
            let _ = ix.cull(&Circle::new(p, 12.0)); // queried, or a scan is the right answer
            ix.tick();
        }
        assert_eq!(ix.backend(), Backend::KeepTree);
        for _ in 0..40 { let _ = ix.cull(&Circle::new(Point::new(10.0, 10.0), 20.0)); ix.tick(); }
        assert_eq!(ix.backend(), Backend::Static, "a settled workload should build once");
        assert_matches_brute(&mut ix, &all);
    }

    /// The property that makes hysteresis worth having: sitting exactly on a boundary must
    /// not migrate every tick. Without the margin and the hold, this flaps.
    #[test]
    fn does_not_flap_on_the_boundary() {
        let th = Thresholds { brute_max: 200, hold_ticks: 30, cooldown: 120, ..Default::default() };
        let mut ix = AdaptiveIndex2::with_thresholds(world(), 8, th);
        for i in 0..200 { ix.insert(P { p: pt(i) }); }
        // Hover at exactly the threshold for a long time, moving items every tick.
        for t in 0..500 {
            ix.update(Slot((t % 200) as u32), |c| c.p = pt(t + 1));
            let _ = ix.cull(&Circle::new(Point::new(50.0, 50.0), 30.0));
            ix.tick();
        }
        assert!(ix.switch_count() <= 1, "flapped {} times on the boundary", ix.switch_count());
    }

    #[test]
    fn the_two_dimensions_make_the_same_decisions() {
        // `Thresholds` is shared, so re-testing its parser here would prove nothing. The
        // failure mode a TWIN file actually has is drift: someone tunes the policy in one
        // dimension and the other quietly keeps the old behaviour. So run one script of
        // work through both indexes and compare the sequence of backends they choose.
        //
        // The workload is identical by construction — the 3D run is the 2D run at z = 0 —
        // and both see the same population, the same movement and the same query count per
        // tick, which is everything `desired()` reads. Relocation counts CAN differ (the
        // leaves are not the same shape), and that is fine: churn was deliberately dropped
        // as the deciding variable. If this test ever fails, either the policies diverged
        // or churn crept back in.
        let th = Thresholds { brute_max: 40, static_ticks: 6, hold_ticks: 2, cooldown: 0, ..Default::default() };
        let mut a = AdaptiveIndex2::with_thresholds(world(), 8, th);
        let mut b = crate::AdaptiveIndex::with_thresholds(crate::Aabb::new(0.0, 0.0, 0.0, 256.0, 256.0, 256.0), 8, th);
        #[derive(Clone, Copy)]
        struct Q { p: crate::Point3 }
        impl crate::Positioned3 for Q { fn position(&self) -> crate::Point3 { self.p } }

        for i in 0..150 { let p = pt(i); a.insert(P { p }); b.insert(Q { p: crate::Point3::new(p.x, p.y, 0.0) }); }
        let (mut sa, mut sb) = (Vec::new(), Vec::new());
        for t in 0..120 {
            // Three regimes in one script: quiet, query-heavy (buys the grid), then frozen
            // (buys the build-once backend). Each transition is a decision worth comparing.
            if t < 80 {
                let p = scatter(t, t % 150);
                a.update(Slot((t % 150) as u32), |c| c.p = p);
                b.update(Slot((t % 150) as u32), |c| c.p = crate::Point3::new(p.x, p.y, 0.0));
            }
            if (30..80).contains(&t) {
                for k in 0..150 {
                    let p = pt(k);
                    let _ = a.cull(&Circle::new(p, 8.0));
                    let _ = b.cull(&crate::Sphere3::new(p.x, p.y, 0.0, 8.0));
                }
            }
            sa.push(a.tick());
            sb.push(b.tick());
        }
        assert_eq!(sa, sb, "the 2D and 3D policies diverged on an identical workload");
        // And non-vacuous: a script that never migrates would pass this trivially.
        assert!(sa.windows(2).filter(|w| w[0] != w[1]).count() >= 2,
            "the script did not exercise enough migrations to compare: {sa:?}");
    }
}
