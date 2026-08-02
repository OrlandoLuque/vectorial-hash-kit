//! Workload profiling + a structure advisor — the kit measuring *itself* so a
//! caller can pick (or switch) the right structure per region/layer instead of
//! guessing. The inputs are cheap counts you already have: item count, moves
//! per tick, **relocations** per tick (the `Crossing::Moved` signal from
//! [`crate::Tree3::update_ref_tracked`]), and queries per tick.
//!
//! The recommendation crossovers are grounded in this repo's benchmarks — see
//! the reasoning on each constant — but they are **heuristics**: expose them and
//! let callers tune per workload. The point is a *self-tuning* index: a dense,
//! fast-moving region and a sparse, near-static one want different structures,
//! and only the local rates know which.

/// What structure best fits the observed workload.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StructureHint {
    /// Few items — a contiguous linear scan (SIMD-friendly, cache-perfect) beats
    /// any index; building/descending a tree is pure overhead at this N.
    BruteForce,
    /// Enough items that an index pays, and moves stay mostly *inside* their leaf
    /// → the adaptive keep-index tree (`update_ref`, O(1) relocation) is ideal.
    KeepIndexTree,
    /// High churn: items routinely cross leaves (fast movers, or cells too fine).
    /// A looser/coarser structure keeps moves in-cell (fewer relocations); or, if
    /// relocation is truly dominant, a per-tick rebuilt uniform grid can win.
    CoarserOrRebuild,
    /// Nothing has moved for a while: the maintain surface is worth nothing, and the
    /// whole cost is build + query. Use a **build-once** structure — `KdTree3` /
    /// `LinearOctree3` / `LinearQuadTree` / a Morton grid — which drop the handle and
    /// remove machinery the dynamic trees carry.
    ///
    /// *Which* build-once structure depends on how the points are distributed, and this
    /// profiler counts rates, not skew: uniform and dense → a Morton grid (cheapest
    /// build and cull); clustered/skewed and query-heavy → `KdTree3` (median split, so
    /// depth stays ~log₂(n/leaf) however the points clump). See `docs/CHOOSING.md`.
    StaticBuildOnce,
}

/// Crossover where a linear scan stops beating an index — **workload-dependent**,
/// so this is a conservative default. Measured (`threshold_bench`, query-only,
/// pre-built index): brute wins up to ~1000 points and the tree takes over
/// around ~1500 (the tree's descent + result-alloc is ~100 ns fixed; a scan is
/// ~1 ns/point). When you also pay the index BUILD (few queries per rebuild, as
/// in the `formations` regiment level) the crossover drops to the low hundreds.
/// So: query-heavy → raise toward ~1000; build-heavy/query-light → lower. 512 is
/// a middle default; tune per workload (the profiler's `query_per_move` helps).
///
/// **Note what "query-only, pre-built index" leaves out**, because the same flaw was found in
/// `examples/calibrate` on 2026-08-03 and fixed there: charging the index nothing for existing
/// is the reading most favourable to it. `examples/brute_edge` sweeps population *against query
/// load* with maintenance included, and the winner changes along a row — a scan wins **7× at
/// 2 048 items** at one cull per frame and loses at 128 when every item is queried. This
/// constant is a recommendation ("is an index worth having?") and remains a middle default for
/// that. It is deliberately **not** what [`crate::Thresholds::brute_max`] uses: that one is a
/// veto ("must I refuse to index?"), fires before any load-aware rule, and is therefore set from
/// the case least favourable to a scan — 64.
pub const BRUTE_FORCE_MAX: usize = 512;

/// Relocation rate above which the keep-index's per-move leaf-exit cost starts to
/// dominate → prefer a coarser/looser structure or a rebuild. From
/// `churn_relocation_bench`: at realistic small moves only ~3–16% relocate (keep
/// wins easily); a loose/coarse structure only pays once a meaningful fraction of
/// moves cross a leaf. Heuristic.
pub const HIGH_RELOCATION: f64 = 0.30;

/// Consecutive ticks with **zero moves** after which the workload counts as static and
/// the advisor stops recommending a structure you can maintain. ~1 second at 30 Hz: long
/// enough not to fire on a pause frame, short enough to notice a level that has settled.
pub const STATIC_TICKS: u32 = 30;

/// Rolling telemetry of a moving-point workload. Feed it per-tick counts; it
/// keeps exponential moving averages of the rates that pick the structure.
#[derive(Clone, Debug)]
pub struct SpatialProfile {
    items: usize,
    reloc_rate: f64, // EMA of relocations / moves  (0..1)
    query_move: f64, // EMA of queries / move
    alpha: f64,
    warmed: bool,
    still: u32,      // consecutive ticks with no movement at all
}

impl Default for SpatialProfile {
    fn default() -> Self { SpatialProfile { items: 0, reloc_rate: 0.0, query_move: 0.0, alpha: 0.1, warmed: false, still: 0 } }
}

impl SpatialProfile {
    /// `alpha` in (0,1] is the EMA weight of the newest tick (0.1 ≈ average over
    /// the last ~10 ticks). Use [`SpatialProfile::default`] for 0.1.
    pub fn with_alpha(alpha: f64) -> Self { SpatialProfile { alpha: alpha.clamp(1e-3, 1.0), ..Default::default() } }

    /// Record one tick: current `items`, and the counts of `moves`,
    /// `relocations` (moves that changed leaf — `Crossing::Moved`), and
    /// `queries` since the last call.
    pub fn observe(&mut self, items: usize, moves: u64, relocations: u64, queries: u64) {
        self.items = items;
        self.still = if moves == 0 { self.still.saturating_add(1) } else { 0 };
        let (rr, qm) = if moves > 0 {
            (relocations as f64 / moves as f64, queries as f64 / moves as f64)
        } else {
            (0.0, queries as f64) // no movement this tick
        };
        if self.warmed {
            self.reloc_rate += self.alpha * (rr - self.reloc_rate);
            self.query_move += self.alpha * (qm - self.query_move);
        } else {
            self.reloc_rate = rr; self.query_move = qm; self.warmed = true;
        }
    }

    pub fn items(&self) -> usize { self.items }
    /// Smoothed fraction of moves that cross a leaf (0..1).
    pub fn relocation_rate(&self) -> f64 { self.reloc_rate }
    /// Smoothed queries per move (read-heavy vs write-heavy).
    pub fn query_per_move(&self) -> f64 { self.query_move }
    /// Consecutive ticks observed with no movement at all.
    pub fn still_ticks(&self) -> u32 { self.still }

    /// The recommended structure for the observed workload.
    pub fn recommend(&self) -> StructureHint {
        if self.items < BRUTE_FORCE_MAX { return StructureHint::BruteForce; }
        // Stillness is checked before churn: a settled workload's LAST measured
        // relocation rate says nothing about a structure that no longer relocates.
        if self.still >= STATIC_TICKS { return StructureHint::StaticBuildOnce; }
        if self.reloc_rate > HIGH_RELOCATION { return StructureHint::CoarserOrRebuild; }
        StructureHint::KeepIndexTree
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advisor_picks_by_scale_and_churn() {
        // few items → brute force regardless of churn
        let mut p = SpatialProfile::default();
        for _ in 0..20 { p.observe(50, 50, 40, 10); }
        assert_eq!(p.recommend(), StructureHint::BruteForce);

        // many items, moves stay in leaf (low relocation) → keep-index tree
        let mut p = SpatialProfile::default();
        for _ in 0..20 { p.observe(100_000, 100_000, 3_500, 5_000); } // 3.5% reloc
        assert!(p.relocation_rate() < 0.1);
        assert_eq!(p.recommend(), StructureHint::KeepIndexTree);

        // many items, fast movers (high relocation) → coarser/rebuild
        let mut p = SpatialProfile::default();
        for _ in 0..20 { p.observe(100_000, 100_000, 60_000, 5_000); } // 60% reloc
        assert!(p.relocation_rate() > HIGH_RELOCATION);
        assert_eq!(p.recommend(), StructureHint::CoarserOrRebuild);

        // no movement this tick doesn't NaN the rate
        let mut p = SpatialProfile::default();
        p.observe(100_000, 0, 0, 1_000);
        assert!(p.relocation_rate().is_finite());
    }

    #[test]
    fn settled_workload_switches_to_a_build_once_structure() {
        let mut p = SpatialProfile::default();
        // a lively phase that would otherwise recommend a rebuild (60% relocation)
        for _ in 0..20 { p.observe(100_000, 100_000, 60_000, 5_000); }
        assert_eq!(p.recommend(), StructureHint::CoarserOrRebuild);
        // …then everything settles: queries keep coming, nothing moves
        for _ in 0..STATIC_TICKS { p.observe(100_000, 0, 0, 5_000); }
        assert_eq!(p.recommend(), StructureHint::StaticBuildOnce, "still for {} ticks", p.still_ticks());
        // one moving tick and it is a live workload again — but NOT instantly "high churn":
        // the still period decayed the relocation EMA, so the rate has to be re-earned.
        p.observe(100_000, 100_000, 60_000, 5_000);
        assert_eq!(p.still_ticks(), 0);
        assert_eq!(p.recommend(), StructureHint::KeepIndexTree, "one tick should not re-declare churn");
        for _ in 0..20 { p.observe(100_000, 100_000, 60_000, 5_000); }
        assert_eq!(p.recommend(), StructureHint::CoarserOrRebuild);
        // a short pause must NOT flip it (a paused frame is not a static level)
        let mut q = SpatialProfile::default();
        for _ in 0..20 { q.observe(100_000, 100_000, 3_500, 5_000); }
        for _ in 0..(STATIC_TICKS - 1) { q.observe(100_000, 0, 0, 5_000); }
        assert_eq!(q.recommend(), StructureHint::KeepIndexTree);
        // and a small static set is still brute force, not a build-once index
        let mut r = SpatialProfile::default();
        for _ in 0..(STATIC_TICKS + 5) { r.observe(50, 0, 0, 100); }
        assert_eq!(r.recommend(), StructureHint::BruteForce);
    }

    #[test]
    fn ema_smooths_a_transient_spike() {
        let mut p = SpatialProfile::with_alpha(0.1);
        for _ in 0..30 { p.observe(100_000, 100_000, 5_000, 0); } // steady 5%
        let base = p.relocation_rate();
        p.observe(100_000, 100_000, 90_000, 0); // one spike tick
        // a single 90% spike must not flip a 5% steady state past the threshold
        assert!(p.relocation_rate() < HIGH_RELOCATION, "one spike overwhelmed the EMA: {} (base {base})", p.relocation_rate());
    }
}
