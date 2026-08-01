# Stealth — a view cone that is an actual frustum cull

`stealth_wgpu` is a small sneak-past-the-guards game whose whole detection loop is kit
calls. Every other demo in this repo culls with **spheres**; this one exists for the two
query verbs none of them showcase.

```bash
cargo run -p vectorial-hash-demos --bin stealth_wgpu --release
```
Controls: `W A S D` move (camera-relative) · drag to orbit · wheel to zoom ·
`[` `]` crowd size · `L` sight lines · `P` pause · `R` restart.
Env: `STEALTH_CIVS`, `STEALTH_MAX_FRAMES` (headless smoke).

## How a guard sees

1. **The view cone is a frustum** — `Polyhedron3::from_corners` over a small near quad
   and a large far quad, i.e. six inward half-spaces. Culling the agent index with it
   answers *"who is inside my cone"* in **one query**, with no per-agent distance-and-
   angle maths. The wireframe you see drawn is those exact 8 corners, so what's on
   screen is the query volume.
2. **The sight line is broadphased with a capsule** — a `Segment3` from the guard's eye
   to the candidate collects the crates *near* that line.
3. **…then resolved exactly** — `Polyhedron3::segment_hit(eye, target)` on those few
   crates. `Some(t)` with `t < 1` means the crate blocks the line before the target.
   Green rays are clear sight, red rays are blocked.

That is the broadphase-then-exact shape the GPU visibility bench measures, on the CPU,
driving a game. The crates are `Polyhedron3`s built from their own 8 corners — the
constructor recovers an axis-aligned box's six faces, so no special case is needed.

## Does the index even pay? (measured, live)

Every frame the *same* cones are also resolved by a **linear scan** of every agent
against the same six half-spaces, the two answers are compared, and both costs go on the
HUD. Per frame, 9 guards, 90 crates — **mean over 600 stepped frames, median of 3 passes**:

| crowd | index cull | linear scan | winner |
| ---: | ---: | ---: | :--- |
| 40 | 3.1 µs | **1.1 µs** | scan, 2.7× |
| 160 | 7.6 µs | **3.7 µs** | scan, 2.1× |
| 640 | 22.8 µs | **16.8 µs** | scan, 1.4× |
| 2 560 | **92.0 µs** | 224.4 µs | index, 2.4× |
| 10 240 | **226.8 µs** | 934.8 µs | index, 4.1× |
| 40 000 | **514.4 µs** | 3 431.7 µs | index, **6.7×** |

Reproduce with `cargo run -p bench-runner --release -- --group demos --only stealth
--repeat 3`. The demo reports **means over the run**, not the last frame's reading: an
early version printed one frame, and in a batch of three passes one of them landed on a
frame that had not stepped and reported a clean, plausible-looking **zero**.

### What that table was quietly not charging for

Those columns are **cull only**, and until 2026-08-01 the demo rebuilt its agent index from
scratch every frame, outside every timer. So the index raced a linear scan that has no
maintenance at all, while its own maintenance was free by omission — the comparison measured
half of one side. The index is **kept** now (`update_ref`, O(1) while an agent stays in its
leaf), and both halves are on the clock. `$STEALTH_REBUILD=1` restores the old path so the
difference is reproducible rather than asserted:

| crowd | kept: maintain + cull = total | rebuilt: maintain + cull = total | scan | scan ÷ total, kept |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 1.4 + 9.8 = **11.2 µs** | 25.5 + 9.1 = 34.6 µs | 4.5 µs | 0.40× (scan wins) |
| 2 000 | 15.2 + 78.9 = **94.1 µs** | 339.7 + 80.0 = 419.7 µs | 151.4 µs | **1.61×** |
| 20 000 | 296.8 + 351.0 = **647.8 µs** | 4 903.0 + 354.0 = 5 257.0 µs | 1 760.2 µs | **2.72×** |

Keeping the index is **18–22× cheaper than rebuilding it** at every size. The load-bearing
consequence is the verdict, not the ratio: charged for a per-frame rebuild the index **never
wins at any size measured** (0.14×, 0.40×, 0.35× — every one below 1.0), and a demo whose
whole point is "does the index pay here?" would have been answering *no* for the wrong reason.
Charged for a keep, the crossover on honest total cost sits at **~1 100 agents**, barely above
the cull-only ~1 000, because keeping costs so little. Maintenance is still 46 % of the index's
total at 20 000 — visible, but no longer decisive.

**The crossover is around a thousand agents.** Below it the tree is honestly slower: a
`contains_point` against 6 planes is a handful of multiply-adds, and at 40 agents the
traversal costs more than just looking at all of them. Above it the index pulls away and
keeps pulling — 6.7× by 40 000. If your game has 30 guards and 200 NPCs, a loop is the
right answer, and this demo says so on screen rather than pretending otherwise.

Note what dominates at scale: **exact LoS** (7.6 ms at 40 000) swamps both, because it runs
per *candidate*. Cheapening the broadphase matters far less than keeping the candidate
set small — which is the argument for the cone being a tight frustum in the first place.

## The bug the comparison caught

The two answers disagreed on ~77% of frames — the scan finding ~12 more agents than the
index. The kit was innocent (`examples/frustum_check.rs`, 400 random frustums against
4 000 points: **0 disagreements**). The demo was at fault twice over:

- The wandering civilians reflected off the arena walls by flipping velocity **whenever
  they were past the line**, including when already heading back in. That chatters, and
  with a variable `dt` a wanderer can walk right through.
- Once outside, they were **outside the index's world box**, so `bulk_load` dropped them
  — correctly. The linear scan still counted them.

So the disagreement was real but neither side was wrong: *they were answering questions
about different sets*. Worth remembering whenever you compare an index against a scan —
**an index only knows what it holds**, and "is it in the box?" is part of the contract.
Fixed by reflecting only when actually outbound and clamping into the arena; agreement
is now exact over 6 000-frame runs at 600 and 5 000 agents.
