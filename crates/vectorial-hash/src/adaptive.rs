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
//! ## The safety property: a query answers the same whoever is holding the data
//!
//! Switching structure under a running caller is only safe if the caller cannot tell. That is
//! more than "the same set": [`cull`](AdaptiveIndex::cull) returns it in a **canonical order**
//! (by [`Slot`], which is stable across migrations), and [`knn`](AdaptiveIndex::knn) returns a
//! canonical *set* too.
//!
//! Order is not cosmetic, and the list of consumers it silently changes is longer than it looks:
//!
//! | consumer | what changes with iteration order |
//! | --- | --- |
//! | k-NN, or "up to N in radius" | truncation at a tie → a different SET |
//! | "first valid target" | **which** entity gets picked |
//! | summing floats over neighbours | FP addition is not associative → different bits |
//! | early exit with a side effect | which one fired the effect |
//! | anything emitting an ordered log | a different stream from the same state |
//!
//! **k-NN needs more than a sort**, which is the part worth knowing. Each backend truncates to
//! `k` *inside* its own search, so when two items tie at the k-th distance the sets differ
//! before any sort could see them — the property test confirms it by failing with
//! *"KeepTree returned a different k-NN SET"* the moment the canonical path is removed. So
//! `knn` re-derives instead: take the k-th distance from the backend, cull that radius (a set
//! everyone agrees on), then order by `(distance, slot)`. One extra cull, and the answer stops
//! depending on which structure happens to be live.
//!
//! Both have an explicit opt-out — [`cull_unordered`](AdaptiveIndex::cull_unordered) and
//! [`knn_unordered`](AdaptiveIndex::knn_unordered) — for consumers *proven* order-insensitive.
//! Opt-out rather than opt-in on purpose: nothing at a call site announces that the loop below
//! truncates or accumulates, so the safe thing has to be the default and the unsafe thing has
//! to be something you typed.
//!
//! The cost is one `u32` per item per backend. The grid already paid it; the tree and the k-d
//! tree now do too, and a comment in `warm_order` that argued against exactly that — 4 bytes
//! forever to save work happening twice in a thousand frames — has been re-decided, because the
//! benefit is no longer a migration optimisation but whether a query has a defined answer.
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
//! **`horde_wgpu`'s `M` mode**, the third and the most favourable — and the one whose workload
//! the layer was actually designed for. One seeded battle, 30 000 zombies, 900 frames, three
//! indexes (`vectorial-hash-demos/examples/horde_index_modes`):
//!
//! | index | ms/step | |
//! | --- | ---: | --- |
//! | `Tree3` pinned | 1.611 | the best fixed choice |
//! | `MortonGrid3` pinned | 6.087 | the *wrong* fixed choice, 3.8x worse |
//! | adaptive | 2.220 | **1.38x the best** (range 1.34-1.61), found without being told |
//!
//! Three switches and 29 near-misses over the battle. That is the shape of the case for the
//! layer, stated fairly: it did not beat the best fixed choice, it cost ~38 % more than one —
//! and it cost 2.7x *less* than the fixed choice a reasonable person might have made instead.
//!
//! **The grid rule reads two quantities, because one was not enough.** `rebuild_query_ratio` was
//! fitted against a table swept at radius 36 — one grid cell wide, i.e. a value taken from the
//! grid arm itself — where the grid wins all 42 cells. At radius 8 the same table flips to the
//! tree in 34 of them. The frontier is a **surface in three axes** (churn x query load x query
//! extent) and the policy was reading two.
//!
//! Extent turned out to be a proxy for the thing that decides. `examples/extent_axis` sweeps two
//! densities 4x apart, which is what makes it a test: if radius decides they flip in the same
//! column, and if expected *points per query* decides they flip at the same points and therefore
//! at different radii. Measured: radius **24** and **16**, at **8.63** and **10.23** points per
//! query. So [`Thresholds::grid_min_hits`] thresholds `density x query volume`, and the mechanism
//! it encodes is that a grid pays a hash lookup per cell whether the cells hold anything or not,
//! while a tree prunes empty space for free. This is why the horde — whose commonest cull has
//! radius 3 — is right to prefer the tree while the two-axis rule said grid.
//!
//! **It shipped OFF for a day, and the reason it is on now is a fix to the INPUT, not the
//! constant.** Checked against three real workloads by running one binary with the veto on and
//! off through `VH_CALIBRATION`: inert on `adaptive_vs_pinned` and the horde (`rebuild_query_
//! ratio` refuses the grid first in both), and on **`fluid_wgpu` — the SPH demo, precisely the
//! shape this rule was built for** — it fired and picked the arm that demo's own counterbalanced
//! bake-off ranks last. The estimate said its queries would find under one neighbour; an SPH
//! kernel is designed to hold tens. So `expected_hits` was rewritten to report what culls
//! **returned** rather than to predict it from the declared world volume, which took the slab
//! error from 7.9x to 0.2 % and made the fluid pick the same backend with the veto on and off.
//!
//! Worth noting how close this came to reading as a success: the horde's ratio moved 1.38x ->
//! 1.34x in the same window, while `MortonGrid3` — an arm this change cannot touch — moved 6.087
//! -> 4.643 ms/step, so the machine had simply gone quiet. **A number that improves alongside
//! your change is not evidence your change did it; check the arm you did not touch.**
//!
//! **That row first read 1.03x, and the correction is the more useful part.** The example ran
//! each arm once, to completion, in a fixed order; five runs of it gave 1.03, 1.36, 1.47, 1.67,
//! 1.63, and 1.03 is simply the luckiest draw. A 900-frame battle is long enough for the machine
//! to drift underneath the three arms. Rotating the arm order per round and quoting the median
//! of the *per-round* ratios is `MEASURING.md` § 7 applied at macro scale — the same fix
//! `common::compare2` makes at micro scale, in an example written by the person who wrote § 7.
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
//! ## Migration, and why the bench cannot yet price the warm start
//!
//! Switching backend is a **migration**, and it rebuilds the successor from `items` in slot
//! order — which is arrival order, spatially arbitrary. The backend being discarded is not
//! arbitrary, though: it spent its life sorting exactly these points. Handing its order over
//! is a *warm start*, worth **1.42× on a `KdTree3` build, 1.81× on `Tree3` inserts** measured
//! directly (`examples/migration_warm_start`).
//!
//! In this index it is on by default ([`Thresholds::warm_start`]) — and on the bench it does
//! **nothing at all**, for a reason worth stating plainly: only the grid can supply that order
//! for free (it stores `Tagged { slot, .. }`), and `adaptive_vs_pinned`'s script migrates
//! `Brute → KeepTree → Grid`. Neither of those *leaves* a grid, so
//! [`AdaptiveIndex::warm_starts`] reports **0 of 2** and the two arms run identical code.
//!
//! That distinction is the point. A feature that cannot fire looks exactly like a feature that
//! does not help, and the paired totals prove it: the same "per migration" figure read −14.5 ms
//! on one run and +70.3 ms on the next. Both are noise. The bench reports the warm-start count
//! next to the timing so nobody reads either number as a verdict, and the real blocker is that
//! a keep-tree cannot supply an order — its items are bare `T` and the slot→handle table does
//! not invert.
//!
//! ## Why the linear trees are not backends
//!
//! `LinearOctree3` and `LinearQuadTree` gained `update`/`remove`/`try_merge_up` at the same time
//! the grids did, which raised the obvious question: should the policy be able to reach for one?
//! The answer is **no**, and it is not a close call — they are dominated on *both* axes by
//! backends already here. From the 3D decision map at its default config (20 000 points, 512³,
//! 16 culls/frame, 100 measured frames):
//!
//! | structure | maintain µs/frame | cull µs |
//! | --- | ---: | ---: |
//! | `Tree3` + `ItemRef` (the `KeepTree` backend) | **616** | 3.37 |
//! | `MortonGrid3` (the `Grid` backend) | 1 351 | **1.77** |
//! | `KdTree3` rebuilt (the `Static` backend) | 3 072 | 2.72 |
//! | `LinearOctree3` rebuilt | 1 785 | 4.52 |
//! | `LinearOctree3` kept | 5 158 | 4.32 |
//!
//! Rebuilt, it maintains 2.9× slower than the keep-tree and culls 2.6× slower than the grid.
//! Kept, it is worse still — keeping an *adaptive* structure in place also drifts its shape, so
//! it needs `try_merge_up` running just to stay as good as a rebuild would have been. There is
//! no cell of the workload space where it would be chosen, so adding it would cost a fifth
//! `Held` variant, a fifth migration path and a fifth set of tests to never fire.
//!
//! Its real niche is the **build** (~2.2× `Octree3`), and the build-once slot is already taken
//! by `KdTree3`, which builds slower but queries better — the right trade when the whole point
//! of that backend is that nothing is moving. Closed as evidence, not as an opinion.
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
use crate::tree3::{Aabb, Crossing, ItemRef, Point3, Positioned3, Shape3, Sphere3, Tree3};

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
    /// **An unconditional floor**: at or below this many items the policy returns
    /// [`Backend::Brute`] whatever else is true, before the load-aware rule
    /// ([`Thresholds::scan_budget`]) is consulted at all.
    ///
    /// # 512 and 182 were both wrong, and so was the question
    ///
    /// This shipped at 512 (borrowed from `advisor::BRUTE_FORCE_MAX`) while `calibrate`
    /// measured 182, and the disagreement stood for as long as both existed. It was never
    /// resolvable as posed: "below how many items does a scan win?" has no single answer,
    /// because a scan costs per **query** and an index costs per **move**.
    /// `examples/brute_edge` sweeps both axes (500³ world, 300 frames, a quarter moving each
    /// frame, every arm through this index with pinned thresholds so only the backend differs):
    ///
    /// | pop | q=1 | q=n/16 | q=n/4 | q=n |
    /// | ---: | --- | --- | --- | --- |
    /// | 64 | scan 1.96× | scan 1.41× | scan 1.10× | scan 1.06× |
    /// | 128 | scan 2.07× | scan 1.12× | **keep 1.10×** | **grid 1.28×** |
    /// | 182 | scan 2.47× | scan 1.08× | keep 1.08× | keep 1.25× |
    /// | 512 | scan 3.80× | keep 1.45× | keep 1.80× | grid 1.97× |
    /// | 2048 | **scan 7.00×** | keep 2.82× | grid 4.60× | grid 5.33× |
    ///
    /// Read along a row and the winner changes with load alone. So the two candidate numbers
    /// were answers to different questions, and both were too high for *this* one: a floor that
    /// fires before the load rule must be set by the case least favourable to a scan, because
    /// that is where it overrides a correct choice. At the heaviest load an index first wins at
    /// **128**, so the floor belongs below it — **64**, the largest population measured where a
    /// scan wins at *every* load.
    ///
    /// `calibrate` agrees independently, once its own probe was fixed: its ladder reads scan up
    /// to 182 and index from 256, against `brute_edge`'s 128 at the heaviest load (the two use
    /// different query radii, so a bracket of 128-256 is the honest reading). Both put the
    /// crossover far below the old 512.
    ///
    /// Lowering it is safe precisely because it is only a backstop: above it `scan_budget`
    /// decides, and it can see the query load. The scan's real reach is enormous and load-shaped
    /// — it still wins 7× at 2 048 items when nobody is querying — and capturing that is
    /// `scan_budget`'s job, not this one's.
    ///
    /// Deliberately no longer tied to `advisor::BRUTE_FORCE_MAX`. That number answers "is an
    /// index worth having at all?", which is a recommendation; this one answers "must I refuse
    /// to index?", which is a veto. They are different questions and were only ever sharing a
    /// constant by accident.
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
    /// The other half of the grid rule: how much a query must be expected to FIND before a grid
    /// is worth having. Below this, take the tree.
    ///
    /// `rebuild_query_ratio` alone reads query *load* and is blind to query *extent*, and the
    /// table it was fitted against was swept at one radius — one cell width, a value derived from
    /// the grid arm itself (`docs/MEASURING.md` § 8i). Adding extent as an axis showed the grid
    /// losing every cell at radius 8 and winning every cell at radius 36, at identical churn and
    /// query load.
    ///
    /// **Extent is not the predictor, though; it is a proxy.** `examples/extent_axis` sweeps two
    /// densities 4x apart, whose predictions disagree: if radius decides, they flip in the same
    /// column; if expected *points per query* decides, they flip at the same points/query and
    /// therefore at different radii. Measured: radius **24** and **16**, at **8.63** and **10.23**
    /// points per query — 1.5x apart in radius, 1.19x apart in points. The mechanism is that a
    /// grid pays its hash lookups whether or not the cells hold anything, while a tree prunes
    /// empty space for free, so the quantity to threshold is `density x query volume`.
    ///
    /// **Default 9.0, between the two measured crossovers**, and enabled — but it shipped
    /// disabled for a day, and that day is the useful part of this doc comment.
    ///
    /// The rule was right and its *input* was not. `expected_hits` derived density from the
    /// **declared world volume**, which is accurate to 0.8 % on uniform data and **7.9x low** on
    /// a slab — the horde's carpet across a cube, a fluid in its container. Turned on, it sent
    /// `fluid_wgpu` (the SPH demo, exactly the shape this rule was built for) from the grid to
    /// the keep-tree, the arm that demo's own counterbalanced bake-off ranks **last**. An SPH
    /// kernel is *designed* so a query holds tens of neighbours; the estimate said under one.
    ///
    /// The fix was not a smaller constant. `expected_hits` now reports what culls **returned**,
    /// which is the quantity the rule wanted all along and which the structure computes anyway;
    /// geometry survives only as a fallback for the first few queries. On the slab test it went
    /// from 7.9x out to **0.2 %**, and `fluid_wgpu` then chose the grid with the veto on and off
    /// alike. That equality was the acceptance test for turning it on.
    ///
    /// Set to 0.0 to disable — the two tests that need to reach a grid regardless of whether a
    /// grid is *wise* for their data do exactly that, and say so.
    pub grid_min_hits: f64,
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
    /// Build a migrated-to backend from the order its predecessor already had (see
    /// `AdaptiveIndex::warm_order`). On by default.
    ///
    /// The switch exists so the two can be raced **paired inside one process** rather than
    /// compared across runs. `examples/adaptive_vs_pinned` spreads 0.53-0.75x against the best
    /// pin from run to run, which is far wider than anything a cheaper migration contributes —
    /// an unpaired before/after cannot see the effect at all, and would happily report noise
    /// as either a win or a regression.
    pub warm_start: bool,
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
            brute_max: 64,
            high_churn: crate::advisor::HIGH_RELOCATION,
            // Re-measured with the three-arm calibration (2026-07-31): 0.205 on this
            // machine. Was 0.1, which switched to the grid roughly twice as early as the
            // measurement supports.
            rebuild_query_ratio: 0.2,
            // Measured crossover, `examples/extent_axis`: 8.63 and 10.23 expected points per
            // query at two densities 4x apart. Shipped disabled for one day while the input was
            // an estimate; enabled once it became an observation (#154).
            grid_min_hits: 9.0,
            scan_budget: 60.0,
            static_ticks: crate::advisor::STATIC_TICKS,
            margin: 0.25,
            hold_ticks: 30,
            cooldown: 120,
            decisive_factor: 4.0,
            detector_alpha: 0.1,
            warm_start: true,
        }
    }
}

impl Thresholds {
    /// Load from the file named by `VH_CALIBRATION`, falling back to the defaults for
    /// anything absent or unparseable. Format is one `key = value` per line, `#` comments
    /// — what the `calibrate` example writes after measuring the local machine.
    pub fn from_env() -> Self {
        match std::env::var("VH_CALIBRATION").ok().and_then(|p| std::fs::read_to_string(p).ok()) {
            Some(text) => {
                // A calibration is a set of measured crossovers, worth exactly as much as the
                // machine that measured them. Warn rather than refuse: a foreign calibration is
                // still a better guess than compiled-in defaults, and someone deliberately
                // shipping one file to a fleet should not be blocked by it.
                if let crate::machine::Provenance::OtherMachine(m) = crate::machine::verdict(&text) {
                    eprintln!("vectorial-hash: VH_CALIBRATION was measured on {m}, running on {}: these thresholds are a guess here — re-run `cargo run --example calibrate`.", crate::machine::machine_id());
                }
                Self::parse(&text)
            }
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
                "grid_min_hits" => if let Ok(x) = v.parse() { t.grid_min_hits = x },
                "scan_budget" => if let Ok(x) = v.parse() { t.scan_budget = x },
                "static_ticks" => if let Ok(x) = v.parse() { t.static_ticks = x },
                "margin" => if let Ok(x) = v.parse() { t.margin = x },
                "hold_ticks" => if let Ok(x) = v.parse() { t.hold_ticks = x },
                "cooldown" => if let Ok(x) = v.parse() { t.cooldown = x },
                "decisive_factor" => if let Ok(x) = v.parse() { t.decisive_factor = x },
                "detector_alpha" => if let Ok(x) = v.parse() { t.detector_alpha = x },
                "warm_start" => if let Ok(x) = v.parse() { t.warm_start = x },
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
            "# vectorial-hash adaptive-index calibration\n{}\
             brute_max = {}\nhigh_churn = {}\nrebuild_query_ratio = {}\nscan_budget = {}\ngrid_min_hits = {}\n\
             static_ticks = {}\nmargin = {}\nhold_ticks = {}\ncooldown = {}\n\
             decisive_factor = {}\ndetector_alpha = {}\nwarm_start = {}\n",
            crate::machine::machine_line(),
            self.brute_max, self.high_churn, self.rebuild_query_ratio, self.scan_budget, self.grid_min_hits,
            self.static_ticks, self.margin, self.hold_ticks, self.cooldown, self.decisive_factor,
            self.detector_alpha, self.warm_start)
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
    Keep(Box<Tree3<Tagged<T>>>, Vec<ItemRef>),
    Grid(Box<MortonGrid3<Tagged<T>>>),
    Static(Box<KdTree3<Tagged<T>>>),
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

/// What the switcher has actually been doing — the counters a caller needs to tell a policy
/// that is working from one that is thrashing.
///
/// `switch_count` alone cannot distinguish "migrated twice, correctly" from "oscillated between
/// two backends all night", and the second is a *misconfiguration*, not a performance problem.
/// So: which pair it moved between, how long it sat in each, and — the one that actually
/// diagnoses a badly-placed threshold — how often the policy WANTED to move and the hysteresis
/// stopped it. A high near-miss count with few switches means the band is sitting right on top
/// of the workload.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SwitchStats {
    /// `[from][to]` in [`Backend`] order: Brute, KeepTree, Grid, Static.
    pub pairs: [[u32; 4]; 4],
    /// Ticks spent holding each backend, same order.
    pub ticks_in: [u64; 4],
    /// Ticks where the policy preferred a different backend but hysteresis (hold, cooldown or
    /// margin) kept it where it was. Not a failure — it is what the band is FOR — but the rate
    /// is the signal.
    pub near_misses: u64,
}

impl SwitchStats {
    /// Total migrations, i.e. the old `switch_count`.
    pub fn switches(&self) -> u32 { self.pairs.iter().flatten().sum() }
    /// The busiest pair and its count, if anything has moved at all. A rate alarm reads this.
    pub fn hottest_pair(&self) -> Option<(Backend, Backend, u32)> {
        let order = [Backend::Brute, Backend::KeepTree, Backend::Grid, Backend::Static];
        let mut best: Option<(Backend, Backend, u32)> = None;
        for (i, row) in self.pairs.iter().enumerate() {
            for (j, &n) in row.iter().enumerate() {
                if n > 0 && best.is_none_or(|(_, _, b)| n > b) { best = Some((order[i], order[j], n)); }
            }
        }
        best
    }
}

pub(crate) fn backend_ix(b: Backend) -> usize {
    match b { Backend::Brute => 0, Backend::KeepTree => 1, Backend::Grid => 2, Backend::Static => 3 }
}

/// What a caller knows about what is coming, so a bulk moment does not thrash the switcher.
///
/// The policy learns from what it has already seen, which means a bulk load walks it up through
/// every backend on the way to the right one: a hundred items look like a scan, ten thousand
/// like a tree, and it migrates at each boundary — paying a rebuild for a state that lasted a
/// few hundred microseconds. A caller loading a level, or receiving a batch, usually knows the
/// answer before the first insert. This is how it says so.
///
/// Every field is optional because a partial hint is still worth having: `expected_count` alone
/// skips most of the thrash.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hints {
    /// How many items are about to arrive (total, not delta).
    pub expected_count: Option<usize>,
    /// Roughly how they will be spread. Decides grid cells and whether a median split pays.
    pub distribution: Option<Distribution>,
    /// Expected moves per item per tick once loaded. 0.0 means "static once built".
    pub churn: Option<f64>,
    /// Expected queries per item per tick.
    pub queries_per_item: Option<f64>,
    /// Typical query size in world units — the grid sizes its cells to this.
    pub query_extent: Option<f64>,
}

/// The shape of the incoming data, as much as the caller knows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distribution {
    /// Spread evenly through the world.
    Uniform,
    /// Concentrated in blobs, with empty space between them.
    Clustered,
    /// Strung along a path or a surface — thin in at least one axis.
    Linear,
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
    /// Migrations that actually received a spatial order from the backend they replaced.
    warm_starts: u32,
    /// Counts for the next `tick`, accumulated by the mutating calls.
    moves: u64,
    relocations: u64,
    queries: u64,
    /// Smoothed queries per item per tick: the variable the backend choice turns on.
    q_per_item: f64,
    stats: SwitchStats,
    frozen: bool,
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
    /// EMA of how many items culls actually RETURN, and how many samples it has seen.
    ///
    /// The rule that reads this wants to know whether a query finds enough to be worth a grid's
    /// lookups. Deriving that from density was an estimate with a known failure mode; counting
    /// what the culls returned is the quantity itself. See [`Self::expected_hits`].
    /// Interior mutability, because the count is only known once the result vector exists — and
    /// that vector borrows the index, so `&mut self` is unavailable at exactly the moment there
    /// is something to record.
    ///
    /// **Atomics rather than `Cell`, and that was not a free choice.** `Cell` is not `Sync`, and
    /// the demos crate holds an `AdaptiveIndex` inside a type it requires to be `Sync`; the build
    /// broke immediately, which is the right outcome for a change that silently narrows a public
    /// type's auto-traits. The EMA is kept as fixed point (1/1024ths of a hit) so no `f64` has to
    /// live in an atomic — hits are counts, and a thousandth of one is precision to spare.
    ///
    /// `Relaxed` throughout: these are statistics feeding a heuristic, and no other field is
    /// ordered against them.
    hits_ema_q10: std::sync::atomic::AtomicU64,
    hit_samples: std::sync::atomic::AtomicU64,
}

impl<T: Positioned3 + Clone> AdaptiveIndex<T> {
    /// `world` bounds the grid backend; `leaf` is the tree leaf capacity.
    pub fn new(world: Aabb, leaf: usize) -> Self {
        Self::with_thresholds(world, leaf, Thresholds::from_env())
    }

    pub fn with_thresholds(world: Aabb, leaf: usize, th: Thresholds) -> Self {
        AdaptiveIndex {
            items: Vec::new(), free: Vec::new(), live: 0, world, leaf: leaf.max(1), held: Held::Brute, warm_starts: 0,
            profile: SpatialProfile::default(), th, pending: None, cooling: 0,
            dirty: false, switches: 0, moves: 0, relocations: 0, queries: 0, q_per_item: 0.0, hits_ema_q10: std::sync::atomic::AtomicU64::new(0), hit_samples: std::sync::atomic::AtomicU64::new(0), stats: SwitchStats::default(), frozen: false,
            m_per_item: 0.0, grid_for: 0, q_extent: 0.0,
        }
    }

    /// Which structure is currently in use.
    pub fn backend(&self) -> Backend {
        match self.held { Held::Brute => Backend::Brute, Held::Keep(..) => Backend::KeepTree, Held::Grid(_) => Backend::Grid, Held::Static(_) => Backend::Static }
    }
    /// How many migrations have happened — a flapping policy shows up here.
    pub fn switch_count(&self) -> u32 { self.switches }
    /// Per-pair switch counts, time in each backend and near-misses. See [`SwitchStats`].
    pub fn stats(&self) -> &SwitchStats { &self.stats }
    /// Is the backend choice currently pinned? See [`freeze`](Self::freeze).
    pub fn is_frozen(&self) -> bool { self.frozen }

    /// Tell the index what is about to arrive, and let it pick the right backend NOW rather
    /// than discover it one migration at a time.
    ///
    /// Without this, a bulk load walks the policy up through every backend on the way to the
    /// right one, paying a rebuild at each boundary for a state that lasts microseconds. With
    /// it, the destination is chosen from the hints and the load arrives into the structure it
    /// was always going to end up in.
    ///
    /// The hints seed the same detector state the policy would otherwise have had to measure,
    /// so they decay naturally: if the caller says 10 000 and 30 turn up, the EMAs pull the
    /// estimate back toward reality within `detector_alpha`'s horizon and the policy corrects
    /// itself. Being wrong costs a migration — the same one it would have paid anyway.
    ///
    /// **Pair it with [`freeze`](Self::freeze) for a bulk load.** On its own it is not enough
    /// and can be actively worse: the population still climbs from zero, so the policy sees a
    /// handful of items, decides a scan is right, and migrates *away* from the destination you
    /// just chose — then back again once the count catches up. Measured on a 4 000-item load,
    /// `prepare` alone took **4** migrations where the cold load took 2; `prepare` + `freeze`
    /// takes **0**. The sequence is:
    ///
    /// ```ignore
    /// ix.prepare(hints);   // pick the destination now
    /// ix.freeze();         // and stop the climb from arguing with it
    /// for item in batch { ix.insert(item); }
    /// ix.thaw();           // back to normal, decisions resume on real data
    /// ```
    pub fn prepare(&mut self, h: Hints) {
        if let Some(n) = h.expected_count { self.items.reserve(n.saturating_sub(self.items.len())); }
        // Seed the detector rather than fake it: these are the very quantities `tick` smooths,
        // so a hint is just an observation the index has not made yet.
        if let Some(q) = h.queries_per_item { self.q_per_item = q; }
        if let Some(c) = h.churn { self.m_per_item = c; }
        if let Some(e) = h.query_extent { self.q_extent = e; }
        // `live` is what the population rules read, and the items are not here yet, so pick the
        // destination against the promised count and migrate to it before they arrive.
        let promised = h.expected_count.unwrap_or(self.live);
        let want = self.desired_at(promised as f64);
        if want != self.backend() { self.migrate(want); }
        // The grid's cells are sized at build time; a clustered load wants them smaller than a
        // uniform one at the same count. Only a nudge — `build` still measures occupancy.
        if let (Some(Distribution::Clustered), Backend::Grid) = (h.distribution, self.backend()) {
            self.dirty = true;
        }
    }

    /// Bring the index up to date and make it safe to query behind a shared reference.
    ///
    /// [`cull`](Self::cull) takes `&mut self` because it does two things: it answers, and it
    /// *observes* — counting the query and feeding the grid's cell-size estimate. That is fine
    /// for one caller in a loop and impossible for several, or for a parallel fan-out, or for a
    /// program that hands `&index` to a dozen systems in one frame.
    ///
    /// So the two halves separate: `settle` once per frame where you have `&mut`, then
    /// [`cull_ref`](Self::cull_ref) as many times as you like behind `&`, then
    /// [`note_queries`](Self::note_queries) to give the policy back what it would have counted.
    ///
    /// ```ignore
    /// ix.settle();                                   // &mut, once
    /// let hits: Vec<_> = systems.par_iter().map(|s| ix.cull_ref(&s.volume)).collect();
    /// ix.note_queries(systems.len() as u32, radius); // &mut, once
    /// ix.tick();
    /// ```
    pub fn settle(&mut self) { self.refresh(); }

    /// Everything inside `shape`, canonically ordered, from a settled index.
    ///
    /// The read-only half of [`cull`](Self::cull). It cannot rebuild a stale backend, so it is
    /// the caller's job to have called [`settle`](Self::settle) after the last mutation — and in
    /// debug builds it says so rather than quietly answering from a stale structure, which is
    /// the failure that would otherwise show up as an item that moved three frames ago.
    pub fn cull_ref<S: Shape3>(&self, shape: &S) -> Vec<&T> {
        debug_assert!(!self.dirty, "cull_ref on a stale index — call settle() after mutating");
        let mut v = self.cull_tagged(shape);
        v.sort_unstable_by_key(|e| e.0);
        self.note_hits(v.len());
        v.into_iter().map(|e| e.1).collect()
    }

    /// Backend order, from a settled index — see [`cull_unordered`](Self::cull_unordered) for
    /// when that is safe.
    pub fn cull_ref_unordered<S: Shape3>(&self, shape: &S) -> Vec<&T> {
        debug_assert!(!self.dirty, "cull_ref on a stale index — call settle() after mutating");
        let v = self.cull_tagged(shape);
        self.note_hits(v.len());
        v.into_iter().map(|e| e.1).collect()
    }

    /// Tell the policy about queries made through [`cull_ref`](Self::cull_ref), which could not
    /// count themselves. `extent` is the typical query size in world units — the grid sizes its
    /// cells to it. Without this the index looks unqueried and will happily migrate to a
    /// structure chosen for a workload that is not happening.
    pub fn note_queries(&mut self, n: u32, extent: f64) {
        self.queries += n as u64;
        if extent > 0.0 {
            self.q_extent = if self.q_extent == 0.0 { extent } else { self.q_extent + 0.1 * (extent - self.q_extent) };
        }
    }

    /// Switch backend **now**, whatever the policy thinks.
    ///
    /// The items come across; every [`Slot`] still addresses the same item afterwards, which is
    /// the property the slot table exists for. Warm-started where the outgoing backend can
    /// supply a spatial order.
    ///
    /// This is here because the built-in policy is a *default*, not a requirement. A caller who
    /// already knows their workload — a level loader, a phase change the program can see coming,
    /// or a cost model of their own with better inputs than this one has — should be able to say
    /// so directly instead of arranging for the detector to guess it. Pair it with
    /// [`freeze`](Self::freeze) if the choice should stick:
    ///
    /// ```ignore
    /// ix.migrate_to(Backend::Grid);   // I know: this phase is query-heavy
    /// ix.freeze();                    // and I do not want to be second-guessed
    /// ```
    ///
    /// Counted in [`stats`](Self::stats) like any other switch — a manual migration is still
    /// work, and a program doing it every frame should be able to see that it is.
    pub fn migrate_to(&mut self, to: Backend) {
        if to != self.backend() { self.migrate(to); }
    }

    /// What the built-in policy would choose right now, without acting on it.
    ///
    /// The other half of [`migrate_to`](Self::migrate_to): a caller running their own decision
    /// can still ask this one for a second opinion, or blend it, or log the disagreement. It
    /// reads only the smoothed counters, so it is cheap and side-effect free.
    pub fn recommended(&self) -> Backend { self.desired() }

    /// The observed rates the policy decides from: `(items, queries per item, moves per item)`,
    /// both smoothed by [`Thresholds::detector_alpha`].
    ///
    /// Exposed so an external policy does not have to duplicate the measurement to disagree
    /// with the conclusion. This is the whole input to [`recommended`](Self::recommended).
    pub fn observed(&self) -> (usize, f64, f64) { (self.live, self.q_per_item, self.m_per_item) }

    /// Pin the current backend: no migration until [`thaw`](Self::thaw).
    ///
    /// For windows where a switch would be wrong even if the policy is right — a bulk load
    /// whose midpoint looks like a different workload than its end, or a section where a
    /// rebuild's latency is unacceptable. The detector keeps observing throughout, so the
    /// decision waiting on the other side of `thaw` is made on real data, not stale data.
    pub fn freeze(&mut self) { self.frozen = true; }

    /// Release [`freeze`](Self::freeze). The next [`tick`](Self::tick) may migrate immediately:
    /// the hysteresis counters kept running, so a workload that changed during the freeze does
    /// not have to re-earn its hold time.
    pub fn thaw(&mut self) { self.frozen = false; }
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
                let r = match t.insert_ref(Tagged { slot, item: item.clone() }) { Some(r) => r, None => { stale = true; ItemRef(u32::MAX) } };
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
                    match t.update_ref_tracked(r, |c| c.item = item) {
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
        self.note_cull(shape);
        let mut v = self.cull_tagged(shape);
        // Canonical order: by slot. Slot is stable across migrations by construction, so this
        // is the one ordering every backend can agree on — and unlike sorting by position it
        // is still total when two items sit on the same point, which is exactly the case that
        // breaks a truncating consumer.
        v.sort_unstable_by_key(|e| e.0);
        self.note_hits(v.len());
        v.into_iter().map(|e| e.1).collect()
    }

    /// Everything inside `shape`, in whatever order the live backend happened to produce.
    ///
    /// **Explicit opt-out of [`cull`](Self::cull)'s canonical order**, for callers whose
    /// consumer is *proven* order-insensitive: membership tests, counts, sums of a commutative
    /// integer quantity. It saves one sort of the result.
    ///
    /// It is opt-out rather than opt-in on purpose. Order-dependence is invisible at the call
    /// site — nothing about `for it in index.cull(&s)` says whether the loop truncates, breaks
    /// early, or accumulates floats — so the safe thing has to be what you get by default, and
    /// the unsafe thing has to be something you typed. Opt-in-to-safety would rot.
    ///
    /// Things that are NOT order-insensitive, and have all been shipped as bugs somewhere:
    /// taking the first N of the result, `break` on the first match when the loop has a side
    /// effect, and summing floats (FP addition is not associative, so a different order is a
    /// different number, bit for bit).
    pub fn cull_unordered<S: Shape3>(&mut self, shape: &S) -> Vec<&T> {
        self.note_cull(shape);
        let v = self.cull_tagged(shape);
        self.note_hits(v.len());
        v.into_iter().map(|e| e.1).collect()
    }

    /// Fold one cull's result count into the running mean. Same EMA weight as the other rates,
    /// so they lag together rather than one leading the others across a threshold.
    fn note_hits(&self, n: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let h = ((n as u64) << 10) as i64;
        let prev = self.hits_ema_q10.load(Relaxed) as i64;
        // Same 0.1 weight as the other rates, in integer arithmetic so it can live in an atomic.
        let next = if self.hit_samples.load(Relaxed) == 0 { h } else { prev + (h - prev) / 10 };
        self.hits_ema_q10.store(next.max(0) as u64, Relaxed);
        self.hit_samples.fetch_add(1, Relaxed);
    }

    /// The EMA, back in hits.
    fn hits_mean(&self) -> f64 {
        self.hits_ema_q10.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1024.0
    }

    /// Counters and cell-size feedback for a cull, then bring a stale backend up to date.
    fn note_cull<S: Shape3>(&mut self, shape: &S) {
        self.queries += 1;
        // What is this caller actually asking for? The grid's cells want to be about this big.
        let b = shape.bounding_box();
        let e = b.w.max(b.h).max(b.d);
        self.q_extent = if self.q_extent == 0.0 { e } else { self.q_extent + 0.1 * (e - self.q_extent) };
        self.refresh();
    }

    /// The raw hit set as `(slot, &item)`, in backend order. The slot is what makes the four
    /// backends comparable at all: brute yields slot order, the tree yields its traversal, the
    /// grid yields cell order and the k-d tree yields node order.
    fn cull_tagged<S: Shape3>(&self, shape: &S) -> Vec<(u32, &T)> {
        match &self.held {
            Held::Brute => self.items.iter().enumerate()
                .filter_map(|(i, o)| o.as_ref().map(|it| (i as u32, it)))
                .filter(|(_, it)| shape.contains_point(it.position())).collect(),
            Held::Keep(t, _) => t.cull(shape).into_iter().map(|g| (g.slot, &g.item)).collect(),
            Held::Grid(g) => g.cull(shape).into_iter().map(|g| (g.slot, &g.item)).collect(),
            Held::Static(k) => k.cull(shape).into_iter().map(|g| (g.slot, &g.item)).collect(),
        }
    }

    /// The k nearest to `q`, as `(distance, &item)` — canonically, so the RESULT SET does not
    /// depend on which backend is live.
    ///
    /// Sorting a backend's answer is not enough here, and that is the subtle part. When two
    /// items tie at the k-th distance, each backend truncates to a different one *inside* its
    /// own search, so the sets differ before this ever sees them. So the canonical path asks
    /// the backend for a radius and then re-derives the set: take the k-th distance, cull a
    /// sphere of that radius (a set every backend agrees on), and order by `(distance, slot)`.
    ///
    /// That costs one extra cull. [`knn_unordered`](Self::knn_unordered) is the cheap backend
    /// path for callers who do not care which of two equidistant items they get.
    pub fn knn(&mut self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        if k == 0 { return Vec::new(); }
        let probe = self.knn_unordered(q, k);
        let Some(&(far, _)) = probe.last() else { return Vec::new() };
        // A hair of slack so the k-th item itself cannot fall outside its own radius through
        // float rounding. Anything the slack pulls in sorts after it and is truncated away.
        let r = far * (1.0 + 1e-12) + f64::EPSILON;
        let mut v: Vec<(f64, u32, &T)> = self.cull_tagged(&Sphere3::new(q.x, q.y, q.z, r)).into_iter()
            .map(|(slot, it)| { let p = it.position();
                let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z);
                ((dx * dx + dy * dy + dz * dz).sqrt(), slot, it) })
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        v.truncate(k);
        v.into_iter().map(|(d, _, it)| (d, it)).collect()
    }

    /// k nearest, straight from the live backend: sorted by distance, but ties broken however
    /// that backend's search happened to break them. See [`knn`](Self::knn) for why that makes
    /// the SET and not merely the order backend-dependent.
    pub fn knn_unordered(&mut self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        self.queries += 1;
        self.refresh();
        self.knn_backend(q, k)
    }

    /// The canonical k-NN from a settled index — the `&`-only twin of [`knn`](Self::knn).
    /// Same re-derivation, so the set does not depend on the live backend.
    pub fn knn_ref(&self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        debug_assert!(!self.dirty, "knn_ref on a stale index — call settle() after mutating");
        if k == 0 { return Vec::new(); }
        let Some(&(far, _)) = self.knn_backend(q, k).last() else { return Vec::new() };
        let r = far * (1.0 + 1e-12) + f64::EPSILON;
        let mut v: Vec<(f64, u32, &T)> = self.cull_tagged(&Sphere3::new(q.x, q.y, q.z, r)).into_iter()
            .map(|(slot, it)| { let p = it.position();
                let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z);
                ((dx * dx + dy * dy + dz * dz).sqrt(), slot, it) })
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        v.truncate(k);
        v.into_iter().map(|(d, _, it)| (d, it)).collect()
    }

    /// Whatever the live backend's own k-NN returns, from a settled index. Shared by the `&mut`
    /// and `&` paths so the two cannot answer differently.
    fn knn_backend(&self, q: Point3, k: usize) -> Vec<(f64, &T)> {
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
            Held::Keep(t, _) => t.knn(q, k).into_iter().map(|(d, g)| (d, &g.item)).collect(),
            Held::Grid(g) => g.knn(q, k).into_iter().map(|(d, g)| (d, &g.item)).collect(),
            Held::Static(kd) => kd.knn(q, k).into_iter().map(|(d, g)| (d, &g.item)).collect(),
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
        self.stats.ticks_in[backend_ix(self.backend())] += 1;
        let (want, decisive) = self.desired_with_confidence();
        if want == self.backend() { self.pending = None; return self.backend(); }
        let held = match self.pending {
            Some((b, n)) if b == want => n + 1,
            _ => 1,
        };
        self.pending = Some((want, held));
        if !self.frozen && (decisive || (held >= self.th.hold_ticks && self.cooling == 0)) {
            self.migrate(want);
            self.pending = None;
            self.cooling = self.th.cooldown;
        } else {
            // Wanted to move and did not. Counting these is what separates "the policy is
            // settled" from "the threshold is sitting on top of the workload and the only
            // reason it looks calm is the hysteresis".
            self.stats.near_misses += 1;
        }
        // A freeze suppresses the migration, not the observation: `pending` and the EMAs keep
        // running above, so `thaw` lands on a decision made from the whole window.
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

    fn desired(&self) -> Backend { self.desired_at(self.live as f64) }

    /// The policy, parameterised on population so [`prepare`](Self::prepare) can ask what it
    /// would want at a size it has not reached yet. One policy, asked two ways — duplicating it
    /// for the hint path would guarantee the two drifted.
    fn desired_at(&self, n: f64) -> Backend {
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
        // ...and query EXTENT vetoes it: a grid is only worth its lookups when the queries find
        // something. See `grid_min_hits` for the measurement, and for why the quantity is expected
        // hits rather than the radius that stands in for it.
        let hits_edge = self.th.grid_min_hits * if cur == Backend::Grid { 1.0 - m } else { 1.0 + m };
        if self.q_per_item > edge && self.expected_hits(n) >= hits_edge { return Backend::Grid; }
        Backend::KeepTree
    }

    /// How many points a typical observed cull is expected to return, from the density implied by
    /// `n` in the declared world and the EMA of observed query extents.
    ///
    /// Returns infinity when no cull has been seen yet (or the world is degenerate): an unknown
    /// extent must not veto anything, so the rule falls back to exactly its pre-extent behaviour
    /// rather than to a guess. A k-NN-only caller therefore never trips it.
    pub fn expected_hits(&self, n: f64) -> f64 {
        // Once enough culls have run, the honest answer is what they RETURNED. No geometry, no
        // world volume, no distribution assumption -- the quantity the rule wants, measured.
        //
        // This replaced a density estimate that was accurate to 0.8 % on uniform data and 7.9x
        // low on a slab, because density came from the DECLARED world and a carpet occupies a
        // sliver of its box. Estimating the input to a threshold, when the input is something the
        // structure already computes on every call, was the mistake.
        if self.hit_samples.load(std::sync::atomic::Ordering::Relaxed) >= Self::HIT_WARMUP {
            // Scaled for `prepare`, which asks about a population not yet reached: hold the
            // spatial distribution fixed and hits move with density, i.e. with n.
            let live = (self.live as f64).max(1.0);
            return self.hits_mean() * (n / live);
        }
        // Before that: the geometric fallback, so a fresh index still answers. Infinity when even
        // that is unavailable, because an unknown input must veto nothing.
        let w = &self.world;
        let vol = w.w * w.h * w.d;
        if vol <= 0.0 || self.q_extent <= 0.0 { return f64::INFINITY; }
        let r = self.q_extent * 0.5;
        (n / vol) * (4.0 / 3.0) * std::f64::consts::PI * r * r * r
    }

    /// Culls needed before [`expected_hits`](Self::expected_hits) trusts observation over
    /// geometry. Small on purpose: the EMA is already smoothing, and the geometric fallback is
    /// the thing being escaped from.
    const HIT_WARMUP: u64 = 8;

    /// The mean number of items recent culls actually returned, and how many were sampled.
    /// `(0.0, 0)` before the first cull.
    pub fn observed_hits(&self) -> (f64, u64) {
        (self.hits_mean(), self.hit_samples.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The EMA of observed cull extents (the largest side of the query's bounding box), 0 until
    /// the first cull. Exposed for the same reason [`observed`](Self::observed) is: a caller
    /// driving the switch themselves needs the policy's whole input, and this was the input the
    /// policy itself was not reading.
    pub fn query_extent(&self) -> f64 { self.q_extent }

    /// How many times the policy has changed backend. A migration is the one event a warm
    /// start affects, so any claim about it divides by this rather than by frames.
    pub fn migrations(&self) -> u32 { self.switches }

    /// How many of those migrations actually got a warm start — i.e. how often the backend
    /// being abandoned could hand over a spatial order.
    ///
    /// This is reported rather than assumed because the first paired measurement of the warm
    /// start read **zero effect**, and the reason was not that it does not work: the script
    /// migrated `Brute -> KeepTree -> Grid`, and neither of those *leaves* a grid, so the code
    /// never ran. A feature that cannot fire looks exactly like a feature that does not help.
    pub fn warm_starts(&self) -> u32 { self.warm_starts }

    fn migrate(&mut self, to: Backend) {
        self.stats.pairs[backend_ix(self.backend())][backend_ix(to)] += 1;
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
                        refs[slot] = t.insert_ref(Tagged { slot: slot as u32, item: it.clone() }).unwrap_or(ItemRef(u32::MAX));
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
                visit.filter_map(|slot| items[slot].clone().map(|item| Tagged { slot: slot as u32, item })).collect()))),
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
    fn warm_order(&mut self) -> Vec<u32> {
        if !self.th.warm_start { return Vec::new(); }
        let order: Vec<u32> = match &self.held {
            Held::Grid(g) => g.iter_z_order().map(|t| t.slot).collect(),
            // A tree can supply one too, and cheaply now. This used to invert the `refs`
            // table (slot -> handle) to recover a handle -> slot map, because the tree stored
            // bare items; the comment here argued that storing the slot beside every item
            // would cost 4 bytes forever to save work that happens twice in a thousand frames.
            // That trade-off was re-decided for a different reason: canonical query order needs
            // the slot on EVERY result, not just at migration, so it is stored now — and the
            // inversion this used to do is simply gone.
            Held::Keep(t, _) => t.handles_dfs().iter().filter_map(|h| t.get_ref(*h).map(|g| g.slot)).collect(),
            _ => Vec::new(),
        };
        if !order.is_empty() { self.warm_starts += 1; }
        order
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
        // Workload alone can no longer reach the grid at this population: the extent veto
        // (`grid_min_hits`) is right that 300 items with radius-8 culls do not justify one, and
        // no query shape at 256^3 would. But `remove` on the grid still has to be tested, and
        // driving there explicitly is exactly what `migrate_to` exists for — the policy is a
        // default, not a requirement. `freeze` then holds it there for the duration, which is
        // the documented use of the pair. The assertion below is unchanged and still fails if
        // the index is sitting on some other backend.
        if ix.backend() != want { ix.migrate_to(want); ix.freeze(); }
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

    /// ★ Do the policy's INPUTS mean what they claim? Answered by COUNTING, not by timing.
    ///
    /// Every threshold in [`Thresholds`] is compared against an estimate, and nothing checked
    /// that the estimates estimate the right thing. That gap shipped a defect: `grid_min_hits`
    /// was given a threshold measured to a tenth (8.63 and 10.23 points per query) and then fed
    /// `expected_hits`, which told it the SPH fluid's queries would find under one neighbour when
    /// an SPH kernel is designed to hold tens. The constant was right and the input was wrong.
    ///
    /// A count is the right instrument here for two reasons: it is exact, so there is no noise to
    /// hide behind, and it is machine-independent, so this runs in CI and on a laptop and means
    /// the same thing on both.
    #[test]
    fn expected_hits_predicts_the_real_hit_count_on_uniform_data() {
        let (n, r) = (4000usize, 24.0);
        let mut ix: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        let mut rng = 0x51ED_2701u64;
        let mut next = |lo: f64, hi: f64| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                                            lo + (rng >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo) };
        for _ in 0..n { ix.insert(P { p: Point3::new(next(0.0, 256.0), next(0.0, 256.0), next(0.0, 256.0)) }); }

        // Query centres are kept in the INTERIOR. A sphere against a wall loses part of its
        // volume to the outside and would make the estimator look wrong for a reason that has
        // nothing to do with the estimator.
        let mut hits = 0usize;
        const Q: usize = 300;
        for _ in 0..Q {
            let c = Point3::new(next(r, 256.0 - r), next(r, 256.0 - r), next(r, 256.0 - r));
            hits += ix.cull(&Sphere3::new(c.x, c.y, c.z, r)).len();
        }
        let actual = hits as f64 / Q as f64;
        let predicted = ix.expected_hits(ix.len() as f64);
        let err = if predicted > actual { predicted / actual } else { actual / predicted };
        assert!(err < 1.10, "expected_hits predicted {predicted:.2} but the culls returned \
                             {actual:.2} on uniform data — {err:.2}x out");
        // Non-vacuity: a query that finds nothing would satisfy any ratio assertion by accident.
        assert!(actual > 5.0, "the workload must actually hit something: {actual:.2}");
    }

    /// ★★ The same check on SLAB-shaped data — the shape that broke the first estimator.
    ///
    /// Points filling a thin layer of a large box (the horde carpet: 30k units across
    /// 1800x72x1800 declared cubic; a fluid in its container) read far sparser than they are when
    /// density is taken from the *declared world volume*. That version predicted 13.81 where the
    /// culls returned 108.53 — 7.9x low, in the direction that vetoes a grid that would have been
    /// right, and the reason `Thresholds::grid_min_hits` shipped disabled.
    ///
    /// This test was written asserting that failure, with a message telling whoever fixed it to
    /// INVERT rather than delete. #154 did, by observing what culls return instead of predicting
    /// it from geometry, and the assertion below is the inverted one: 108.31 predicted against
    /// 108.53 returned, 0.2 %. Same 1.10 band as the uniform twin, because the point of the fix
    /// is that the shape of the data stopped mattering.
    #[test]
    fn expected_hits_now_tracks_slab_data_too_which_was_154() {
        let (n, r, thickness) = (4000usize, 24.0, 8.0);
        let mut ix: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        let mut rng = 0x1234_ABCDu64;
        let mut next = |lo: f64, hi: f64| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                                           lo + (rng >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo) };
        for _ in 0..n { ix.insert(P { p: Point3::new(next(0.0, 256.0), next(0.0, thickness), next(0.0, 256.0)) }); }

        let mut hits = 0usize;
        const Q: usize = 300;
        for _ in 0..Q {
            let c = Point3::new(next(r, 256.0 - r), next(0.0, thickness), next(r, 256.0 - r));
            hits += ix.cull(&Sphere3::new(c.x, c.y, c.z, r)).len();
        }
        let actual = hits as f64 / Q as f64;
        let predicted = ix.expected_hits(ix.len() as f64);
        assert!(actual > 5.0, "the workload must actually hit something: {actual:.2}");
        let err = if predicted > actual { predicted / actual } else { actual / predicted };
        assert!(err < 1.10,
                "expected_hits predicted {predicted:.2} but the culls returned {actual:.2} on                  slab data, {err:.2}x out. The geometric estimator read 7.9x here; a regression                  toward that means the observed-hits path is no longer being reached.");
        // Non-vacuity: it must be reading OBSERVATION, not geometry that happens to agree.
        let (ema, samples) = ix.observed_hits();
        assert!(samples >= 8 && (ema - actual).abs() < actual * 0.2,
                "the observed-hits path was not exercised: ema {ema:.2}, samples {samples}");
    }


    /// ★ The policy's DECISIONS, pinned. A ratchet, in the spirit of `tests/work_counts.rs`.
    ///
    /// Everything else about the policy is asserted as a band — a rate within 0.1, a ratio under
    /// 1.10 — because the quantities are estimates. The *choices* are not estimates: for a fixed
    /// script the sequence of backends is a deterministic function of the thresholds, and it is
    /// the thing a caller actually experiences. Nothing was pinning it, so this week two default
    /// changes altered real decisions and the suite stayed green; the damage showed up only when
    /// a demo was run by hand.
    ///
    /// If this fails, the policy changed its mind about something. That is often correct — read
    /// the new sequence, decide whether it is an improvement, and bless it deliberately. What it
    /// must not be is a surprise.
    #[test]
    fn the_policy_makes_the_same_decisions_on_a_fixed_script() {
        // Four acts whose character differs, so more than one rule gets a turn: a small quiet
        // population (a scan should serve it), growth with churn and light queries, a query storm
        // wide enough to find things, then stillness.
        let th = Thresholds { brute_max: 40, hold_ticks: 2, cooldown: 0, static_ticks: 8, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
        let mut all: Vec<P> = Vec::new();
        let mut seen: Vec<Backend> = vec![ix.backend()];
        let note = |ix: &AdaptiveIndex<P>, seen: &mut Vec<Backend>| {
            if *seen.last().unwrap() != ix.backend() { seen.push(ix.backend()); }
        };

        for i in 0..30 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        for _ in 0..10 { let _ = ix.cull(&Sphere3::new(60.0, 60.0, 60.0, 20.0)); ix.tick(); note(&ix, &mut seen); }

        for i in 30..900 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
        for t in 0..30 {
            for k in 0..300 { mv(&mut ix, &mut all, k, scatter(t, k)); }
            for k in 0..30 { let p = pt(k); let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 20.0)); }
            ix.tick(); note(&ix, &mut seen);
        }

        for t in 30..70 {
            for k in 0..300 { mv(&mut ix, &mut all, k, scatter(t, k)); }
            for k in 0..900 { let p = pt(k); let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 40.0)); }
            ix.tick(); note(&ix, &mut seen);
        }

        for _ in 0..40 { let _ = ix.cull(&Sphere3::new(60.0, 60.0, 60.0, 40.0)); ix.tick(); note(&ix, &mut seen); }

        // Blessed sequence. Bless a new one on purpose, never to make a red test green.
        let expected = [Backend::Brute, Backend::KeepTree, Backend::Grid, Backend::Static];
        assert_eq!(seen.as_slice(), &expected[..],
                   "the policy changed its decisions on this script. If that is an improvement,                     update the blessed sequence and say why in the commit; if it is not, the                     threshold you just touched is the cause.");
        // Non-vacuity: a script that never migrates would satisfy any single-element sequence.
        assert!(seen.len() >= 3, "the script must exercise several backends, saw {seen:?}");
        assert_matches_brute(&mut ix, &all);
    }

    /// The two rate estimates, against the events that produced them. Cheap, exact, and the
    /// kind of thing that is obviously true until the day it is not.
    #[test]
    fn the_rate_estimates_track_the_events_they_count() {
        let th = Thresholds { brute_max: 10, hold_ticks: 2, cooldown: 0, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
        let mut all = Vec::new();
        for i in 0..500 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }

        // A fixed script: 250 culls and 125 moves per tick against 500 items, i.e. 0.5 and 0.25.
        for t in 0..120 {
            for k in 0..125 { mv(&mut ix, &mut all, k, scatter(t, k)); }
            for k in 0..250 { let p = pt(k); let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 12.0)); }
            ix.tick();
        }
        let (n, q, m) = ix.observed();
        assert_eq!(n, 500);
        // The EMA is deliberately lagging, so the band is loose — but a rate that drifted by an
        // order of magnitude, or counted culls as moves, would not survive it.
        assert!((q - 0.5).abs() < 0.1, "queries/item read {q:.3}, script says 0.5");
        assert!((m - 0.25).abs() < 0.1, "moves/item read {m:.3}, script says 0.25");
        // ...and they must not be the same number, which would mean one is reading the other.
        assert!((q - m).abs() > 0.15, "q {q:.3} and m {m:.3} are suspiciously equal");
    }

    /// ★ The extent veto, in BOTH directions from one workload.
    ///
    /// The two arms differ in exactly one thing — the radius of the culls — and everything else
    /// (population, world, churn, query COUNT, thresholds) is identical. That is what makes it a
    /// test of `grid_min_hits` rather than a demonstration: if the veto did nothing, both arms
    /// would land on the grid, and if it vetoed unconditionally, neither would.
    ///
    /// The numbers come from `examples/extent_axis`, not from tuning until it passed: 2000 items
    /// in a 256^3 world is a density of 1.19e-4, so radius 6 expects **0.11** points a query and
    /// radius 40 expects **31.9**, either side of the measured 8.63-10.23 crossover.
    #[test]
    fn query_extent_vetoes_the_grid_and_wide_queries_restore_it() {
        let run = |r: f64| {
            // Opts IN: the shipped default is 0.0 (off). This tests the mechanism, which is
            // correct; whether it should be ON by default is a separate question the fluid
            // counterexample currently answers no.
            let th = Thresholds { brute_max: 10, hold_ticks: 2, cooldown: 0, rebuild_query_ratio: 0.1, grid_min_hits: 9.0, ..Default::default() };
            let mut ix = AdaptiveIndex::with_thresholds(world(), 8, th);
            let mut all = Vec::new();
            for i in 0..2000 { let p = P { p: pt(i) }; all.push(p); ix.insert(p); }
            for t in 0..80 {
                for k in 0..2000 { mv(&mut ix, &mut all, k, scatter(t, k)); }
                for k in 0..800 { let _ = ix.cull(&Sphere3::new(k as f64 % 250.0, 40.0, 40.0, r)); }
                ix.tick();
            }
            (ix.backend(), ix.expected_hits(ix.len() as f64), ix.queries_per_item())
        };
        let (narrow, hits_n, q_n) = run(6.0);
        let (wide, hits_w, q_w) = run(40.0);

        // Both arms must clear the OTHER half of the rule, or the narrow arm would be choosing
        // the tree for a reason that has nothing to do with extent and the test would pass for
        // the wrong reason.
        assert!(q_n > 0.3 && q_w > 0.3, "both arms must be query-heavy: {q_n:.3} and {q_w:.3}");
        assert!(hits_n < 1.0, "narrow arm should expect almost nothing: {hits_n:.3}");
        assert!(hits_w > 20.0, "wide arm should expect plenty: {hits_w:.3}");

        assert_eq!(narrow, Backend::KeepTree, "narrow queries find nothing; a grid is not worth its lookups");
        assert_eq!(wide, Backend::Grid, "wide queries find plenty; the grid should win");
    }

    /// A QUERY-HEAVY workload — roughly one cull per item per tick, the shape SPH has —
    /// is where a rebuilt grid beats keeping the tree. Churn alone never flips it (the
    /// calibration swept both: churn moved keep's margin from 116x to 6.4x but never past
    /// 1.0), so this test drives the variable that actually decides.
    #[test]
    fn query_heavy_workload_switches_to_the_rebuilt_grid() {
        // `grid_min_hits: 0.0` disables the extent veto explicitly — redundant against today's
        // default, and deliberately kept so this test does not start failing the day that
        // default flips. It is the POINT of the test, not a workaround. 300 items in a 256^3 world with radius-10 culls expect 0.075 hits a
        // query: no query shape justifies a grid at that population, so with the veto live this
        // workload can never reach one. This test asserts the OTHER half of the grid rule — that
        // query load, not churn, is what moves it — and needs the veto out of the way to do it.
        // The veto itself is covered by `query_extent_vetoes_the_grid_and_wide_queries_restore_it`.
        let th = Thresholds { brute_max: 10, hold_ticks: 2, cooldown: 0, rebuild_query_ratio: 0.1, grid_min_hits: 0.0, ..Default::default() };
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
            grid_min_hits: 3.5,
            warm_start: true,
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
        assert!((parsed.grid_min_hits - th.grid_min_hits).abs() < 1e-9);
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
            && (th.detector_alpha - d.detector_alpha).abs() > 1e-9
            && (th.grid_min_hits - d.grid_min_hits).abs() > 1e-9,
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

    /// The `&`-only read path must answer exactly what the `&mut` one does, and the policy must
    /// still learn about queries it could not count itself.
    #[test]
    fn settle_then_read_behind_a_shared_reference() {
        let mut ix: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        for i in 0..600 { ix.insert(P { p: pt(i) }); }
        let probe = Sphere3::new(120.0, 120.0, 120.0, 70.0);

        ix.settle();
        // Several readers at once — the thing `cull(&mut self)` makes impossible.
        let a = ix.cull_ref(&probe);
        let b = ix.cull_ref(&probe);
        assert_eq!(a.len(), b.len());
        assert!(a.len() > 20, "the probe must hit plenty or this proves nothing");
        let shared: Vec<(u64, u64, u64)> = a.iter().map(|p| p.p.to_bits3()).collect();
        let by_ref_b: Vec<(u64, u64, u64)> = b.iter().map(|p| p.p.to_bits3()).collect();
        assert_eq!(shared, by_ref_b, "two shared reads must agree");
        drop((a, b));
        let owned: Vec<(u64, u64, u64)> = ix.cull(&probe).iter().map(|p| p.p.to_bits3()).collect();
        assert_eq!(shared, owned, "the &-path must answer exactly what the &mut path does");

        // A mutation makes it stale, and settle is what clears that. (The debug_assert in
        // cull_ref is the loud version of this; here we just check the flag is honest.)
        ix.update(Slot(0), |p| p.p.x = (p.p.x + 40.0) % 250.0);
        ix.settle();
        assert_eq!(ix.cull_ref(&probe).len(), ix.cull(&probe).len());

        // And the policy must SEE queries it did not count: without note_queries an index
        // queried only through cull_ref looks idle, and migrates for a workload that is not
        // happening.
        let mut quiet: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        let mut loud: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        for i in 0..200 { quiet.insert(P { p: pt(i) }); loud.insert(P { p: pt(i) }); }
        // Query load well over `rebuild_query_ratio`, and keep everything moving so the
        // build-once arm is not the answer: this is the workload the grid exists for, and the
        // only difference between the two indexes is whether anyone told them it is happening.
        for t in 0..80 {
            for i in 0..200u32 {
                let d = ((t + i) % 7) as f64;
                quiet.update(Slot(i), |p| p.p.x = (p.p.x + d) % 250.0);
                loud.update(Slot(i), |p| p.p.x = (p.p.x + d) % 250.0);
            }
            quiet.settle(); loud.settle();
            for _ in 0..120 { let _ = quiet.cull_ref(&probe); let _ = loud.cull_ref(&probe); }
            loud.note_queries(120, 140.0);
            quiet.tick(); loud.tick();
        }
        assert_eq!(quiet.observed().1, 0.0, "an unreported reader leaves the policy blind");
        assert!(loud.observed().1 > 0.0, "note_queries must reach the detector");
        assert_ne!(quiet.backend(), loud.backend(),
            "and being blind must actually change the decision, or this reports nothing");
    }

    /// A caller driving the switch by hand must get the same guarantees the policy does: the
    /// items survive, the slots still address them, and the answers do not change.
    #[test]
    fn a_hand_driven_migration_keeps_every_slot_and_every_answer() {
        let mut ix: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        let slots: Vec<Slot> = (0..500).map(|i| ix.insert(P { p: pt(i) })).collect();
        let probe = Sphere3::new(120.0, 120.0, 120.0, 70.0);

        let want: Vec<(u64, u64, u64)> = ix.cull(&probe).iter().map(|p| p.p.to_bits3()).collect();
        assert!(want.len() > 20, "the probe must hit plenty or this proves nothing");
        let by_slot: Vec<(u64, u64, u64)> = slots.iter().map(|s| ix.get(*s).unwrap().p.to_bits3()).collect();

        // Round trip every backend, by hand, in an order the policy would never choose.
        for to in [Backend::Static, Backend::Brute, Backend::Grid, Backend::KeepTree, Backend::Grid] {
            ix.migrate_to(to);
            assert_eq!(ix.backend(), to, "migrate_to must actually arrive");
            assert_eq!(ix.len(), 500, "no item may be lost in transit");
            let got: Vec<(u64, u64, u64)> = ix.cull(&probe).iter().map(|p| p.p.to_bits3()).collect();
            assert_eq!(got, want, "{to:?}: a hand-driven switch changed the answer");
            let now: Vec<(u64, u64, u64)> = slots.iter().map(|s| ix.get(*s).unwrap().p.to_bits3()).collect();
            assert_eq!(now, by_slot, "{to:?}: a slot stopped pointing at its item");
        }
        // Migrating to where we already are is not work, and must not be counted as a switch.
        let before = ix.switch_count();
        ix.migrate_to(ix.backend());
        assert_eq!(ix.switch_count(), before, "a switch to the current backend is not a switch");
        assert_eq!(ix.stats().switches(), before);

        // `recommended` reports without acting, and `observed` is its whole input.
        let held = ix.backend();
        let _ = ix.recommended();
        assert_eq!(ix.backend(), held, "recommended() must not migrate");
        let (n, q, m) = ix.observed();
        assert_eq!(n, 500);
        assert!(q.is_finite() && m.is_finite());
    }

    /// `prepare` must save the walk up through the backends, and `freeze` must actually stop a
    /// migration the policy would otherwise make.
    #[test]
    fn prepare_skips_the_thrash_and_freeze_holds_the_line() {
        let load = |ix: &mut AdaptiveIndex<P>| {
            let probe = Sphere3::new(120.0, 120.0, 120.0, 70.0);
            for i in 0..4000 {
                ix.insert(P { p: pt(i) });
                // A query every few inserts, so the policy has something to react to on the
                // way up — which is exactly what makes a cold load migrate repeatedly.
                if i % 25 == 0 { for _ in 0..8 { ix.cull(&probe); } ix.tick(); }
            }
        };

        let mut cold: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        load(&mut cold);

        let hints = Hints { expected_count: Some(4000), queries_per_item: Some(8.0 / 4000.0),
                            churn: Some(0.0), query_extent: Some(140.0),
                            distribution: Some(Distribution::Uniform) };

        // prepare ALONE is not enough, and the number is the reason the docs say so: the
        // population still climbs from zero, so the policy migrates away from the destination
        // and back. Asserted, not hand-waved, so nobody "simplifies" the freeze away.
        let mut hinted: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        hinted.prepare(hints);
        load(&mut hinted);

        // prepare + freeze is the documented sequence.
        let mut warm: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        warm.prepare(hints);
        // The one migration `prepare` makes is its whole job, and it is paid here — on an
        // EMPTY index, where a rebuild costs nothing. That is the difference between one
        // migration and one migration that matters.
        let at_prepare = warm.switch_count();
        assert_eq!(at_prepare, 1, "prepare should move once, to the destination");
        assert!(warm.is_empty(), "and it should do it before the items arrive");
        warm.freeze();
        load(&mut warm);
        warm.thaw();

        assert!(cold.switch_count() > 0, "the cold load must actually thrash or this proves nothing");
        assert!(hinted.switch_count() >= cold.switch_count(),
            "if a bare hint stopped costing extra, the docs' warning is stale (cold {}, hinted {})",
            cold.switch_count(), hinted.switch_count());
        assert_eq!(warm.switch_count(), at_prepare,
            "prepare + freeze should load straight in: not one migration after the empty one");

        // And the freeze: pin a backend the policy is actively trying to leave.
        let mut ix: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        for i in 0..500 { ix.insert(P { p: pt(i) }); }
        let probe = Sphere3::new(120.0, 120.0, 120.0, 70.0);
        for _ in 0..40 { for _ in 0..8 { ix.cull(&probe); } ix.tick(); }
        let pinned = ix.backend();
        ix.freeze();
        assert!(ix.is_frozen());
        let before = ix.switch_count();
        // Invert the workload: heavy movement, no queries — the policy wants to leave.
        for _ in 0..200 {
            for i in 0..500 { ix.update(Slot(i), |p| p.p.x = (p.p.x + 3.0) % 250.0); }
            ix.tick();
        }
        assert_eq!(ix.backend(), pinned, "a frozen index must not migrate");
        assert_eq!(ix.switch_count(), before, "a frozen index must not migrate");
        assert!(ix.stats().near_misses > 0, "it must have WANTED to move, or the freeze proved nothing");

        // Thawing lets the decision land, on data gathered during the freeze.
        ix.thaw();
        for _ in 0..40 { ix.tick(); }
        assert_ne!(ix.backend(), pinned, "after thaw the pending decision should take effect");
    }

    /// The counters must separate a settled policy from one that is only calm because the
    /// hysteresis is holding it down. Both look identical through `switch_count`.
    #[test]
    fn switch_stats_tell_a_settled_policy_from_a_suppressed_one() {
        let mut ix: AdaptiveIndex<P> = AdaptiveIndex::new(world(), 8);
        for i in 0..400 { ix.insert(P { p: pt(i) }); }
        let probe = Sphere3::new(120.0, 120.0, 120.0, 70.0);
        // Query hard so the policy wants the grid, and move nothing, so it also wants Static.
        for _ in 0..60 { for _ in 0..8 { ix.cull(&probe); } ix.tick(); }

        let st = ix.stats().clone();
        assert_eq!(st.switches(), ix.switch_count(), "the two counters must agree");
        assert!(st.ticks_in.iter().sum::<u64>() >= 60, "every tick must be attributed to a backend");
        assert!(st.switches() > 0, "this script must migrate or the test proves nothing");
        let (from, to, n) = st.hottest_pair().expect("something moved");
        assert_ne!(from, to, "a switch to itself is not a switch");
        assert!(n > 0);
        // The pair matrix and the flat count must be the same event seen twice.
        assert_eq!(st.pairs.iter().flatten().sum::<u32>(), st.switches());
    }

    /// ★ R1 — the four backends must answer with the SAME SEQUENCE, not merely the same set.
    ///
    /// This is the property that makes it safe to swap a structure under a running caller. It is
    /// tested with deliberate ties, because ties are where it actually breaks: identical
    /// positions (so no position-based ordering can separate them) and, for k-NN, more items at
    /// exactly the k-th distance than there is room for — so which ones come back depends on how
    /// the backend's own search happened to truncate.
    #[test]
    fn every_backend_answers_in_the_same_canonical_order() {
        #[derive(Clone, Copy, Debug)]
        struct Q { id: u32, p: Point3 }
        impl Positioned3 for Q { fn position(&self) -> Point3 { self.p } }

        let (w, leaf) = (world(), 8);
        let mut ix: AdaptiveIndex<Q> = AdaptiveIndex::new(w, leaf);
        let mut id = 0u32;
        // A scatter, plus COINCIDENT points: five items sharing one position cannot be ordered
        // by where they are, only by which slot they occupy.
        for i in 0..300u32 {
            let f = i as f64;
            ix.insert(Q { id, p: Point3::new(20.0 + (f * 7.0) % 200.0, 20.0 + (f * 13.0) % 200.0, 20.0 + (f * 29.0) % 200.0) });
            id += 1;
        }
        for _ in 0..5 { ix.insert(Q { id, p: Point3::new(128.0, 128.0, 128.0) }); id += 1; }
        // Eight items at exactly the same distance from the k-NN probe, so a k of 4 must choose.
        let c = Point3::new(90.0, 90.0, 90.0);
        for (dx, dy, dz) in [(10.0, 0.0, 0.0), (-10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, -10.0, 0.0),
                             (0.0, 0.0, 10.0), (0.0, 0.0, -10.0), (6.0, 8.0, 0.0), (-6.0, -8.0, 0.0)] {
            ix.insert(Q { id, p: Point3::new(c.x + dx, c.y + dy, c.z + dz) });
            id += 1;
        }

        let probe = Sphere3::new(120.0, 120.0, 120.0, 60.0);
        let items = ix.items.clone();
        let (mut canon_cull, mut canon_knn, mut raw_orders) = (None, None, Vec::new());

        for backend in [Backend::Brute, Backend::KeepTree, Backend::Grid, Backend::Static] {
            ix.held = AdaptiveIndex::build_ordered(backend, &items, w, leaf, 0.0, &[]);
            ix.dirty = false;

            let cull: Vec<u32> = ix.cull(&probe).iter().map(|q| q.id).collect();
            assert!(cull.len() > 20, "{backend:?}: the probe must hit plenty or this proves nothing");
            match &canon_cull {
                None => canon_cull = Some(cull),
                Some(want) => assert_eq!(&cull, want, "{backend:?} culled a different SEQUENCE"),
            }

            let knn: Vec<(u64, u32)> = ix.knn(c, 4).iter().map(|(d, q)| (d.to_bits(), q.id)).collect();
            assert_eq!(knn.len(), 4);
            match &canon_knn {
                None => canon_knn = Some(knn),
                Some(want) => assert_eq!(&knn, want, "{backend:?} returned a different k-NN SET (the tie broke elsewhere)"),
            }

            raw_orders.push(ix.cull_unordered(&probe).iter().map(|q| q.id).collect::<Vec<_>>());
        }

        // Non-vacuity: if every backend happened to emit in the same raw order anyway, the sort
        // above is doing nothing and this test would pass with the canonicalisation deleted.
        let mut sets: Vec<Vec<u32>> = raw_orders.iter().map(|v| { let mut c = v.clone(); c.sort_unstable(); c }).collect();
        sets.dedup();
        assert_eq!(sets.len(), 1, "the backends disagreed on the SET, which is a different bug");
        assert!(raw_orders.iter().any(|o| o != &raw_orders[0]),
            "no backend emitted a different raw order, so this test cannot see canonicalisation working");
    }

    /// Cull whatever a `Held` is holding, as raw coordinates so the four backends are
    /// comparable without depending on item identity.
    fn held_cull(h: &Held<P>, s: &Sphere3, items: &[Option<P>]) -> Vec<(u64, u64, u64)> {
        match h {
            Held::Brute => items.iter().flatten().filter(|it| s.contains_point(it.position())).map(|it| it.p.to_bits3()).collect(),
            Held::Keep(t, _) => t.cull(s).iter().map(|it| it.item.p.to_bits3()).collect(),
            Held::Grid(g) => g.cull(s).iter().map(|t| t.item.p.to_bits3()).collect(),
            Held::Static(k) => k.cull(s).iter().map(|it| it.item.p.to_bits3()).collect(),
        }
    }
}
