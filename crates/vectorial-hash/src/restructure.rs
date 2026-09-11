//! Restructuring counters — how often a structure **changes its own shape** under load.
//!
//! The kit already counts the two kinds of work a *query* does: `work_counters` counts point
//! tests and box classifications, `grid-stats` counts cell lookups. Neither can see the third
//! kind, because it is not query work at all: an adaptive structure **splits** a leaf that has
//! grown past its item limit and **merges** children that between them have shrunk back under
//! it. That happens on the maintenance path, it is triggered by items *moving*, and it is the
//! one axis on which the kit's structures genuinely differ in kind rather than in constant:
//!
//! | | splits | merges |
//! | --- | --- | --- |
//! | `Tree`, `QuadTree`, `IntegerTree`, `Tree3`, `Octree3` | yes | yes |
//! | `LinearOctree3`, `LinearQuadTree` | yes | yes (since `try_merge_up`) |
//! | `MortonGrid`, `MortonGrid3` | **never** | **never** |
//! | `KdTree2`, `KdTree3` | build-once; cannot maintain at all | — |
//!
//! A fixed-resolution key has no shape to change: an item that moves either stays in its cell
//! or is re-keyed into another one, and the structure is bit-for-bit the structure it would
//! have been had the item started there. That is a property worth being able to *measure*
//! rather than assert, which is what this module is for — a grid reading exactly zero is the
//! evidence, and a tree's non-zero count is the price.
//!
//! **Global and per-thread, not per-structure** — the same contract as [`crate::morton3`]'s
//! cell counter, and the same caveat: it is a tuning diagnostic, not instrumentation to build
//! on. Genuinely zero-cost when the feature is off: neither the counters nor the increments
//! are compiled.
//!
//! ```ignore
//! let _ = vectorial_hash::restructure::reset();
//! // ... a frame of maintenance ...
//! let (splits, merges) = vectorial_hash::restructure::counts();
//! ```

#[cfg(feature = "struct-stats")]
mod cells {
    use std::cell::Cell;
    thread_local! {
        pub static SPLITS: Cell<u64> = const { Cell::new(0) };
        pub static MERGES: Cell<u64> = const { Cell::new(0) };
    }
}

/// `(splits, merges)` on this thread since the last [`reset`].
#[cfg(feature = "struct-stats")]
pub fn counts() -> (u64, u64) { (cells::SPLITS.with(|c| c.get()), cells::MERGES.with(|c| c.get())) }

/// Zero both counters and return what they held.
#[cfg(feature = "struct-stats")]
pub fn reset() -> (u64, u64) { (cells::SPLITS.with(|c| c.replace(0)), cells::MERGES.with(|c| c.replace(0))) }

/// One leaf subdivided into children.
#[cfg(feature = "struct-stats")]
#[inline]
pub(crate) fn count_split() { cells::SPLITS.with(|c| c.set(c.get() + 1)); }

/// One parent collapsed back into a leaf. Counted **per level**: `try_merge_up` walks upward
/// for as long as merging keeps being possible, and each level it collapses is a separate
/// piece of work, so a single call can legitimately report several.
#[cfg(feature = "struct-stats")]
#[inline]
pub(crate) fn count_merge() { cells::MERGES.with(|c| c.set(c.get() + 1)); }

#[cfg(not(feature = "struct-stats"))]
#[inline(always)]
pub(crate) fn count_split() {}

#[cfg(not(feature = "struct-stats"))]
#[inline(always)]
pub(crate) fn count_merge() {}
