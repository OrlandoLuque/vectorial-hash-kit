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
}

/// Crossover where a linear scan stops beating an index. From the `formations`
/// demo + micro-benchmarks: brute force wins into the low hundreds (contiguous +
/// SIMD), the tree pulls ahead above this. Heuristic — tune per element size.
pub const BRUTE_FORCE_MAX: usize = 256;

/// Relocation rate above which the keep-index's per-move leaf-exit cost starts to
/// dominate → prefer a coarser/looser structure or a rebuild. From
/// `churn_relocation_bench`: at realistic small moves only ~3–16% relocate (keep
/// wins easily); a loose/coarse structure only pays once a meaningful fraction of
/// moves cross a leaf. Heuristic.
pub const HIGH_RELOCATION: f64 = 0.30;

/// Rolling telemetry of a moving-point workload. Feed it per-tick counts; it
/// keeps exponential moving averages of the rates that pick the structure.
#[derive(Clone, Debug)]
pub struct SpatialProfile {
    items: usize,
    reloc_rate: f64, // EMA of relocations / moves  (0..1)
    query_move: f64, // EMA of queries / move
    alpha: f64,
    warmed: bool,
}

impl Default for SpatialProfile {
    fn default() -> Self { SpatialProfile { items: 0, reloc_rate: 0.0, query_move: 0.0, alpha: 0.1, warmed: false } }
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

    /// The recommended structure for the observed workload.
    pub fn recommend(&self) -> StructureHint {
        if self.items < BRUTE_FORCE_MAX { return StructureHint::BruteForce; }
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
    fn ema_smooths_a_transient_spike() {
        let mut p = SpatialProfile::with_alpha(0.1);
        for _ in 0..30 { p.observe(100_000, 100_000, 5_000, 0); } // steady 5%
        let base = p.relocation_rate();
        p.observe(100_000, 100_000, 90_000, 0); // one spike tick
        // a single 90% spike must not flip a 5% steady state past the threshold
        assert!(p.relocation_rate() < HIGH_RELOCATION, "one spike overwhelmed the EMA: {} (base {base})", p.relocation_rate());
    }
}
