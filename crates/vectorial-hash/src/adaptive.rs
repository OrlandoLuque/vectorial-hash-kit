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
//! | brute scan | few items, **or few queries** | a scan costs per query, an index per move — see [`Thresholds::scan_budget`] |
//! | [`Tree3`] + `ItemRef` | items move, queries are moderate | O(1) relocation; wins maintain in 15 of 16 sweep configs |
//! | [`MortonGrid3`] | queries per item are high | a flat grid answers dense queries with fewer descents, and **keeps in place** now, so it no longer pays a rebuild per mutation |
//! | [`KdTree3`] | nothing has moved for a while | build-once, best query on skewed data |
//!
//! ## What it is actually worth — measured, and not flattering everywhere
//!
//! Two benchmarks, deliberately asking different questions.
//!
//! **`examples/adaptive_vs_pinned`** runs a script whose character changes four times — small
//! and quiet, growing past the scan edge with everything moving, a query storm at one cull per
//! item, then frozen — through the adaptive index and through each backend *pinned*. A
//! stationary workload would just reward whichever fixed structure suited it and prove nothing.
//! Latest: **0.70× the best pinned choice**. It converts the catastrophic guess (a pinned brute
//! scan: ~22 000 ms) into ~1 200 ms without being told anything, and gives up ~30 % against the
//! choice you would have had to know in advance.
//!
//! **`fluid_wgpu`**, a real demo with a stationary workload, is the other end: **parity with
//! the best fixed choice** (347-360 fps against 352), found by itself, and 8-15 % ahead of the
//! other two. Its maintenance drops to 0.00 ms because it picks the grid and *keeps* it where
//! the fixed option refills it.
//!
//! So: on a workload that does not change, the policy costs nothing and picks correctly. On one
//! that changes, migrating and noticing are both real costs and it currently loses to the best
//! fixed choice. The honest pitch is **insurance, not optimisation** — it is worth having when
//! you cannot know the workload in advance, and worth *not* having when you can.
//!
//! Where the remaining loss goes, measured rather than guessed: frames spent on the previous
//! backend before any detector could have decided, plus the migration's own rebuild. Making the
//! detector instant ([`Thresholds::detector_alpha`] = 1.0) recovers ~10 % of one act and loses
//! it again on another.
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
    /// **Re-derived 2026-07-31, and the obvious inference was wrong.** When the grid backend
    /// gained an in-place `update`, the reasoning written here was: the grid got cheaper to
    /// maintain, so the crossover must move *down*. It does not. The threshold is read off at
    /// **maximum churn**, which is precisely the one regime where keeping a grid is worthless
    /// — at churn 1.0 every item re-buckets and a bulk refill wins. The two-armed and
    /// three-armed models therefore agree exactly where the number is taken, and the shipped
    /// 0.1 was simply too low: the three-arm calibration measures **0.205** here, so the old
    /// default reached for the grid about twice as early as the measurement supports. It was
    /// aggressive, not conservative. Corrected to 0.2.
    ///
    /// **And one scalar no longer describes the boundary.** With the grid rebuilding, the
    /// frontier was roughly vertical: enough queries and the refill pays for itself whatever
    /// the churn. With the grid keeping, it is diagonal — measured, 20k items:
    ///
    /// | churn ↓ / culls → | 16 | 256 | 4096 | 5000 |
    /// | --- | --- | --- | --- | --- |
    /// | 0.0 | grid 1.71× | grid 1.92× | grid 1.57× | grid 1.51× |
    /// | 0.2 | keep 4.50× | keep 1.55× | grid 1.35× | grid 1.29× |
    /// | 0.6 | keep 7.56× | keep 2.90× | grid 1.13× | grid 1.13× |
    /// | 1.0 | keep 5.92× | keep 2.71× | *rebuild* 1.11× | *rebuild* 1.13× |
    ///
    /// The grid strategy wins 11 of those 24 cells and a pure rebuild wins 2 — the corner at
    /// full churn and heavy query load, which this index cannot reach because its grid backend
    /// always keeps. That corner is a known, bounded loss of at most ~1.13×, and closing it
    /// would mean a fifth backend whose only difference is a rebuild.
    pub rebuild_query_ratio: f64,
    /// **How many point-tests a linear scan may cost before an index is worth maintaining**,
    /// expressed as a multiple of the per-item maintenance cost. This is the variable
    /// `brute_max` alone was missing.
    ///
    /// A scan's cost is per QUERY: `queries x items` point tests a frame. An index's cost is
    /// per MOVE: it maintains whatever moved, whether you query it once or a thousand times.
    /// So "is a linear scan good enough" cannot be answered by population alone, and answering
    /// it that way was measurably wrong — `examples/adaptive_vs_pinned`, act 2: 20 000 items,
    /// everything moving, but only 8 culls a frame, and a pinned brute scan beat the adaptive
    /// index **6.7x** (6.9 ms against 46.3) because the index was maintaining 20 000 items to
    /// serve 8 queries.
    ///
    /// The comparison, per item per frame: a scan costs `q_per_item x items` and maintenance
    /// costs `churn`, so the scan wins while `q_per_item x items < churn x scan_budget`. The
    /// default is measured on this machine as roughly the ratio between one tree update and
    /// one distance test.
    pub scan_budget: f64,
    /// Consecutive tick with no movement at all before the workload counts as static.
    pub static_ticks: u32,
    /// Boundaries are widened by this fraction in the direction of travel, so a workload
    /// sitting exactly on one does not oscillate.
    pub margin: f64,
    /// A candidate must win this many consecutive ticks before a migration happens.
    pub hold_ticks: u32,
    /// Minimum ticks between migrations, whatever the numbers say.
    pub cooldown: u32,
    /// **How fast the policy notices the workload changing.** The EMA weight applied to
    /// queries-per-item and moves-per-item each tick.
    ///
    /// Smoothing exists so that one busy frame does not trigger a migration, and that is still
    /// wanted. At 0.1 the detector needs ~10 ticks to register a change and ~16 to clear the
    /// decisive threshold, so during a 60-frame query storm the index spends its first quarter
    /// on the wrong backend.
    ///
    /// **Swept, and the obvious cure is not one.** Two runs each on
    /// `examples/adaptive_vs_pinned`:
    ///
    /// | alpha | act 3 (storm) | act 4 (frozen) | total | backends chosen |
    /// | ---: | ---: | ---: | ---: | --- |
    /// | 0.1 | 1096-1097 | 82-90 | 1185-1194 | Brute, KeepTree, Grid |
    /// | 1.0 | 973-1042 | 128-134 | 1108-1185 | Brute, Grid, KeepTree |
    ///
    /// Reacting instantly buys ~10 % on the storm and gives most of it back on the frozen act,
    /// where it never settles on the build-once backend at all. The net is inside the
    /// run-to-run noise. Left at 0.1, and calibratable for anyone whose phases are longer than
    /// this script's.
    ///
    /// It also corrects an over-claim: with the lag removed entirely, the storm is still ~1.4x
    /// behind a pinned grid, so the detector's delay was never most of that gap. What remains
    /// is the frames spent on the previous backend before any detector could have decided, plus
    /// the migration's own rebuild.
    pub detector_alpha: f64,
    /// **How far past a boundary counts as decisive enough to skip the hysteresis.**
    ///
    /// `hold_ticks` and `cooldown` exist to stop the policy flapping at a boundary — and at a
    /// boundary, being on the wrong side costs almost nothing, because the two candidates are
    /// nearly equal there. That reasoning stops applying the moment the workload is nowhere
    /// near the boundary, and the cost of waiting stops being small with it.
    ///
    /// Measured (`examples/adaptive_vs_pinned`): with a flat 120-tick cooldown the index
    /// entered the brute scan during a quiet act and was still on it through the next act's
    /// query storm — one cull per item, five times `rebuild_query_ratio` — for **14 834 ms
    /// against the best fixed choice's 1 000**. It migrated once in 240 frames. A rule meant
    /// to prevent a cheap oscillation had bought a 14x catastrophe.
    ///
    /// So a candidate that is past its boundary by more than this factor migrates at once,
    /// ignoring both the hold and the cooldown. Set it to `f64::MAX` for the old behaviour.
    pub decisive_factor: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            brute_max: crate::advisor::BRUTE_FORCE_MAX,
            high_churn: crate::advisor::HIGH_RELOCATION,
            // Re-measured with the three-arm calibration (2026-07-31): 0.205 on this
            // machine. Was 0.1, which switched to the grid roughly twice as early as the
            // measurement supports.
            rebuild_query_ratio: 0.2,
            scan_budget: 60.0,
            static_ticks: crate::advisor::STATIC_TICKS,
            margin: 0.25,
            hold_ticks: 30,
            cooldown: 120,
            decisive_factor: 4.0,
            detector_alpha: 0.1,
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
                "scan_budget" => if let Ok(x) = v.parse() { t.scan_budget = x },
                "static_ticks" => if let Ok(x) = v.parse() { t.static_ticks = x },
                "margin" => if let Ok(x) = v.parse() { t.margin = x },
                "hold_ticks" => if let Ok(x) = v.parse() { t.hold_ticks = x },
                "cooldown" => if let Ok(x) = v.parse() { t.cooldown = x },
                "decisive_factor" => if let Ok(x) = v.parse() { t.decisive_factor = x },
                "detector_alpha" => if let Ok(x) = v.parse() { t.detector_alpha = x },
                _ => {}
            }
        }
        t
    }

    /// Serialise, for the `calibrate` example to write.
    ///
    /// Every field — and there is a test that checks every field rather than a sample.
    /// `rebuild_query_ratio` was missing here for as long as it has existed, so `calibrate`
    /// measured it, printed it to the terminal, and then dropped it on the way to disk:
    /// anyone pointing `VH_CALIBRATION` at the resulting file got the compiled-in default for
    /// the one number the tool exists to produce.
    pub fn to_text(&self) -> String {
        format!(
            "# vectorial-hash adaptive-index calibration\n\
             brute_max = {}\nhigh_churn = {}\nrebuild_query_ratio = {}\nscan_budget = {}\n\
             static_ticks = {}\nmargin = {}\nhold_ticks = {}\ncooldown = {}\n\
             decisive_factor = {}\ndetector_alpha = {}\n",
            self.brute_max, self.high_churn, self.rebuild_query_ratio, self.scan_budget,
            self.static_ticks, self.margin, self.hold_ticks, self.cooldown, self.decisive_factor,
            self.detector_alpha)
    }
}

/// Items a grid cell should hold, and the reason there is a number here at all.
///
/// A uniform grid's cost is decided by occupancy: too many per cell and every query scans a
/// crowd, too few and it spends its time crossing empty cells looking for anything at all.
/// Measured across four workloads (docs/THREE_D.md), the sweet spot is "about the k you ask
/// for", and the failure mode on the thin side is far worse than on the fat side — 23x against
/// 1.2x. So this sits deliberately on the low-but-not-tiny side of k.
pub(crate) const GRID_TARGET_PER_CELL: f64 = 8.0;

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
    /// Smoothed MOVES per item per tick. Deliberately not `relocation_rate`, which counts
    /// leaf crossings and is therefore only observable while the tree backend is loaded — a
    /// policy rule that reads it can never fire on any other backend, which is exactly how the
    /// first version of the scan rule failed: on the brute scan it saw churn 0, computed a
    /// scan budget of 0, and could never conclude that the scan was still fine.
    m_per_item: f64,
    /// The live count the grid backend's cell size was derived from.
    ///
    /// A grid's levels are chosen when it is built, and with the keep path it is never
    /// rebuilt — so a grid built for 60 items was still serving 20 000 of them at 2 500 per
    /// cell, three times slower than it needed to be. Nothing marked it dirty because nothing
    /// was wrong with its CONTENTS; what had gone stale was its geometry.
    grid_for: usize,
    /// Smoothed extent of the query volumes this index is actually being asked for.
    ///
    /// A uniform grid's cell size wants to be about the size of a query — that is the classic
    /// SPH bucket rule, and it is why the fluid demo's hand-built grid asks for
    /// `levels_for_cell_size(rect, kernel_radius)`. Sizing by occupancy alone instead produced
    /// cells 3.4x the kernel radius there, which made maintenance cheap and every query sweep a
    /// far larger area: measured, 1.66-1.91 ms against the hand-sized grid's 1.43-1.56.
    ///
    /// The index could not size its cells as well as its caller because it never looked at what
    /// it was being asked. It does now: `cull` reports the bounding box of every shape, and the
    /// grid is built for the typical one. Zero until a query has been seen, in which case the
    /// occupancy rule stands in.
    q_extent: f64,
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
            m_per_item: 0.0, grid_for: 0, q_extent: 0.0,
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
    /// Smoothed moves per item per tick — what the scan-vs-index choice turns on.
    pub fn moves_per_item(&self) -> f64 { self.m_per_item }

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
        // What is this caller actually asking for? The grid's cells want to be about this big.
        let b = shape.bounding_box();
        let e = b.w.max(b.h).max(b.d);
        self.q_extent = if self.q_extent == 0.0 { e } else { self.q_extent + 0.1 * (e - self.q_extent) };
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
        let a = self.th.detector_alpha;
        self.q_per_item += a * (qpi - self.q_per_item); // same EMA weight the advisor uses
        let mpi = if self.live == 0 { 0.0 } else { self.moves as f64 / self.live as f64 };
        self.m_per_item += a * (mpi - self.m_per_item);
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
    /// The backend the numbers argue for, and whether they argue for it **decisively** —
    /// far enough past the boundary that the hysteresis has nothing to protect against. See
    /// [`Thresholds::decisive_factor`].
    fn desired_with_confidence(&self) -> (Backend, bool) {
        let want = self.desired();
        if want == self.backend() { return (want, false); }
        let f = self.th.decisive_factor;
        let n = self.live as f64;
        let decisive = match want {
            // Population, or query load, an order of magnitude clear of the edge.
            Backend::Brute => n * f <= self.th.brute_max as f64 || self.q_per_item * n * f <= self.m_per_item * self.th.scan_budget,
            Backend::Grid => self.q_per_item >= self.th.rebuild_query_ratio * f,
            // Leaving the scan because the queries have arrived, or nothing has moved for many
            // times longer than the rule asks.
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
        // ...and a scan also wins when nobody is asking much, however many items there are:
        // it costs per query while an index costs per move. See `scan_budget`.
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
        let order = self.warm_order();
        self.held = Self::build_ordered(to, &self.items, self.world, self.leaf, self.q_extent, &order);
        self.grid_for = self.live; // whatever the grid's cells were just sized for
        self.dirty = false;
    }

    /// Rebuild a backend from the item list. This is the cost hysteresis exists to avoid
    /// paying twice.
    /// [`Self::build`], but visiting the live slots in `order` when one is supplied (see
    /// [`Self::warm_order`]). An empty `order` means arrival order.
    ///
    /// `order` is a performance hint and nothing else: it must not change what the built
    /// structure answers, and `examples/migration_warm_start` asserts exactly that against a
    /// non-vacuous probe cull. It is also allowed to be incomplete or stale — any slot it does
    /// not mention is still inserted afterwards, so a wrong hint costs speed, never contents.
    fn build_ordered(to: Backend, items: &[Option<T>], world: Aabb, leaf: usize, q_extent: f64, order: &[u32]) -> Held<T> {
        // Visit `order` first, then anything it missed. `seen` is only allocated when a hint
        // was actually given, so the cold path is unchanged.
        let mut seen = vec![false; if order.is_empty() { 0 } else { items.len() }];
        let slots: Vec<usize> = if order.is_empty() { Vec::new() } else {
            let mut v = Vec::with_capacity(items.len());
            for &s in order { let s = s as usize; if s < items.len() && !seen[s] { seen[s] = true; v.push(s); } }
            for (s, hit) in seen.iter().enumerate() { if !hit { v.push(s); } }
            v
        };
        let visit: Box<dyn Iterator<Item = usize>> = if slots.is_empty() { Box::new(0..items.len()) } else { Box::new(slots.into_iter()) };
        Self::build_visiting(to, items, world, leaf, q_extent, visit)
    }

    fn build_visiting(to: Backend, items: &[Option<T>], world: Aabb, leaf: usize, q_extent: f64, visit: Box<dyn Iterator<Item = usize> + '_>) -> Held<T> {
        match to {
            Backend::Brute => { let _ = visit; Held::Brute }
            Backend::KeepTree => {
                let mut t = Tree3::new(world, leaf);
                // One entry per SLOT, holes carried through as dead refs, so `refs[slot]`
                // survives a rebuild. Skipping holes here would shift every handle past the
                // first removal — the exact bug the slot table is built to prevent. Note this
                // is why the visit ORDER can differ from the slot order without breaking
                // anything: `refs` is written BY slot, not appended in visit order.
                let mut refs = vec![ItemRef(u32::MAX); items.len()];
                for slot in visit {
                    if let Some(it) = &items[slot] {
                        refs[slot] = t.insert_ref(it.clone()).unwrap_or(ItemRef(u32::MAX));
                    }
                }
                Held::Keep(Box::new(t), refs)
            }
            Backend::Grid => {
                // Cells sized by OCCUPANCY, not by a fixed fraction of the world. This used to
                // ask for `world_max / 64`, i.e. 64 cells per axis whatever the population —
                // 262 144 cells for 20 000 items, or 0.08 items per cell. That is precisely
                // the pathology measured in docs/THREE_D.md ("size the cell to hold roughly k
                // points", where 0.08/cell ran 23x slower than a coarser grid), and it made
                // the grid backend lose to everything in `examples/adaptive_vs_pinned` — which
                // in turn made the policy look wrong when the geometry was.
                let live = items.iter().flatten().count().max(1);
                // A cell about the size of a typical query, when one has been seen; the
                // occupancy rule only as a fallback for a grid built before any query.
                let levels = if q_extent > 0.0 {
                    MortonGrid3::<Tagged<T>>::levels_for_cell_size(world, q_extent)
                } else {
                    let per_axis = (live as f64 / GRID_TARGET_PER_CELL).cbrt().max(1.0);
                    (per_axis.log2().round().max(1.0) as u32).min(10)
                };
                let mut g = MortonGrid3::new(world, levels);
                for slot in visit {
                    if let Some(v) = &items[slot] { g.insert(Tagged { slot: slot as u32, item: v.clone() }); }
                }
                Held::Grid(Box::new(g))
            }
            Backend::Static => Held::Static(Box::new(KdTree3::from_items(leaf,
                visit.filter_map(|slot| items[slot].clone()).collect()))),
        }
    }

    /// Bring a stale rebuild-based backend up to date. The keep-index tree is never stale
    /// (that is the point of it), and brute force reads the items directly.
    fn refresh(&mut self) {
        if !self.dirty { return; }
        let b = self.backend();
        if b == Backend::Brute { self.dirty = false; return; }
        let order = self.warm_order();
        self.held = Self::build_ordered(b, &self.items, self.world, self.leaf, self.q_extent, &order);
        self.grid_for = self.live;
        self.dirty = false;
    }

    /// **Warm-start migration**: the slot order the *outgoing* backend already had these points
    /// in, so its successor is built from a spatially coherent sequence instead of insertion
    /// order.
    ///
    /// A backend about to be discarded is not a blank slate — it spent its whole life sorting
    /// exactly these points in space. Throwing that away and rebuilding from slot order (which
    /// is arrival order, i.e. spatially arbitrary) discards work that was already paid for.
    /// Measured on 50 000 points, `examples/migration_warm_start`: building from Z-order rather
    /// than arrival order is **1.42x on `KdTree3`, 1.81x on `Tree3` inserts, 1.26x on
    /// `bulk_load`**, 1.07x on another grid.
    ///
    /// Only the grid can supply it for free — it stores `Tagged { slot, .. }`, so the slots come
    /// back with the points, and ordering costs one sort of the *cell* keys (far fewer than
    /// items). The keep-tree cannot: its items are bare `T` and the slot lives in a
    /// slot-to-handle table that does not invert. Deriving an order by sorting all N points
    /// would work for any backend, but then the migration pays for the sort it is trying to
    /// avoid, so this deliberately returns nothing rather than guessing that the trade is
    /// positive. Empty means "no opinion, use arrival order".
    fn warm_order(&self) -> Vec<u32> {
        match &self.held {
            Held::Grid(g) => g.iter_z_order().map(|t| t.slot).collect(),
            _ => Vec::new(),
        }
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
            // Query intensity is what buys an index at all. The grid needs one cull per item
            // per tick to clear `rebuild_query_ratio`; the TREE needs only a couple, but it
            // does need some — a population above `brute_max` with nobody querying is a
            // population a linear scan should keep serving, which is what `scan_budget` says
            // and what these tests used to assert the opposite of.
            match want {
                Backend::Grid => for k in 0..ix.len() { let p = pt(k); let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 8.0)); },
                Backend::KeepTree => for k in 0..2 { let p = pt(k); let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 8.0)); },
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

        // Past the widened boundary AND with someone actually querying, it must move to the
        // keep-index tree. The queries are not decoration: population alone does not make an
        // index worth maintaining, because a scan costs per query while an index costs per
        // move (see `Thresholds::scan_budget` — measured at 6.7x for 20 000 items served by 8
        // culls a frame). This test asserted the population-only rule until that was measured.
        for i in 80..400 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        for t in 0..20 {
            let k = t % 100;
            let p = all[k].p;
            mv(&mut ix, &mut all, k, Point3::new(p.x + 0.03, p.y, p.z));
            let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 12.0));
            ix.tick();
        }
        assert_eq!(ix.backend(), Backend::KeepTree, "400 queried items should be indexed");
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
            let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 12.0)); // queried, or a scan is right
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
        // EVERY field, each set to something no default produces. The first version checked
        // three of them, picked by hand — and `rebuild_query_ratio`, the one number the
        // calibrate tool exists to measure, was missing from `to_text` for as long as it had
        // existed. A round-trip test that samples fields tests the fields it samples.
        // An exhaustive literal on purpose: adding a field to Thresholds must BREAK THIS
        // TEST rather than silently leave the new field untested. It has already done its job
        // once — `scan_budget` and `decisive_factor` failed to compile here the moment they
        // were added, which is how they came to be covered at all.
        let th = Thresholds {
            brute_max: 777, high_churn: 0.42, rebuild_query_ratio: 0.37, scan_budget: 42.0,
            static_ticks: 13, margin: 0.11, hold_ticks: 9, cooldown: 5, decisive_factor: 9.0,
            detector_alpha: 0.33,
        };
        let parsed = Thresholds::parse(&(th.to_text() + "future_key = 12\n# a comment\n"));
        assert_eq!(parsed.brute_max, th.brute_max);
        assert!((parsed.high_churn - th.high_churn).abs() < 1e-9);
        assert!((parsed.rebuild_query_ratio - th.rebuild_query_ratio).abs() < 1e-9,
            "rebuild_query_ratio did not survive the round trip: {} != {}",
            parsed.rebuild_query_ratio, th.rebuild_query_ratio);
        assert!((parsed.scan_budget - th.scan_budget).abs() < 1e-9);
        assert!((parsed.decisive_factor - th.decisive_factor).abs() < 1e-9);
        assert!((parsed.detector_alpha - th.detector_alpha).abs() < 1e-9);
        assert_eq!(parsed.static_ticks, th.static_ticks);
        assert!((parsed.margin - th.margin).abs() < 1e-9);
        assert_eq!(parsed.hold_ticks, th.hold_ticks);
        assert_eq!(parsed.cooldown, th.cooldown);
        // None of those may equal the default, or every assertion above passes vacuously.
        let d = Thresholds::default();
        assert!(th.brute_max != d.brute_max && th.static_ticks != d.static_ticks
            && th.hold_ticks != d.hold_ticks && th.cooldown != d.cooldown
            && (th.rebuild_query_ratio - d.rebuild_query_ratio).abs() > 1e-9
            && (th.scan_budget - d.scan_budget).abs() > 1e-9
            && (th.decisive_factor - d.decisive_factor).abs() > 1e-9
            && (th.detector_alpha - d.detector_alpha).abs() > 1e-9,
            "the test values must differ from the defaults or this proves nothing");
    }

    /// A warm-start order hint is allowed to reorder the *build*; it is not allowed to change
    /// a single answer, and it must survive a hint that is wrong.
    ///
    /// The interesting cases are the broken hints, because a hint arrives from a backend that
    /// is about to be thrown away and may be stale by then: one that omits slots (the omitted
    /// items must still be there), one that repeats them (no duplicates), and one naming slots
    /// that do not exist (no panic). All four builds must answer identically to the cold one,
    /// and `Slot` handles must still address the right items afterwards.
    #[test]
    fn a_warm_start_order_changes_nothing_it_is_allowed_to_change() {
        let items: Vec<Option<P>> = (0..400).map(|i| Some(P { p: pt(i) })).collect();
        let (w, leaf) = (world(), 8);
        let probe = Sphere3::new(120.0, 120.0, 120.0, 70.0);

        let full: Vec<u32> = (0..400).rev().map(|i| i as u32).collect();     // every slot, reversed
        let partial: Vec<u32> = (0..400).step_by(3).map(|i| i as u32).collect(); // a third of them
        let repeated: Vec<u32> = (0..400).map(|i| (i % 50) as u32).collect();  // heavy duplicates
        let bogus: Vec<u32> = (0..400).map(|i| (i + 9000) as u32).collect();   // all out of range

        for backend in [Backend::KeepTree, Backend::Grid, Backend::Static, Backend::Brute] {
            let cold = AdaptiveIndex::build_ordered(backend, &items, w, leaf, 0.0, &[]);
            let mut want: Vec<(u64, u64, u64)> = held_cull(&cold, &probe, &items);
            want.sort_unstable();
            assert!(!want.is_empty(), "the probe must hit something or this proves nothing");

            for (name, order) in [("full", &full), ("partial", &partial), ("repeated", &repeated), ("bogus", &bogus)] {
                let warm = AdaptiveIndex::build_ordered(backend, &items, w, leaf, 0.0, order);
                let mut got = held_cull(&warm, &probe, &items);
                got.sort_unstable();
                assert_eq!(got, want, "{backend:?} answered differently with the {name} order hint");
            }
        }
    }

    /// Cull whatever a `Held` is holding, as raw coordinates so the four backends are
    /// comparable without depending on item identity.
    fn held_cull(h: &Held<P>, s: &Sphere3, items: &[Option<P>]) -> Vec<(u64, u64, u64)> {
        match h {
            Held::Brute => items.iter().flatten().filter(|it| s.contains_point(it.position())).map(|it| it.p.to_bits3()).collect(),
            Held::Keep(t, _) => t.cull(s).iter().map(|it| it.p.to_bits3()).collect(),
            Held::Grid(g) => g.cull(s).iter().map(|t| t.item.p.to_bits3()).collect(),
            Held::Static(k) => k.cull(s).iter().map(|it| it.p.to_bits3()).collect(),
        }
    }
}
