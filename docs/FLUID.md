# Fluid (SPH) — the archetypal index workload, made interactive

`fluid_wgpu` is a 2D **position-based fluid** you stir with the mouse or a finger.
It exists because smoothed-particle hydrodynamics is the workload a spatial index is
*for*: every particle needs the ones inside its kernel radius `H`, **every step**, and
a fluid is about as clustered as data gets. So the demo is also a head-to-head — `M`
cycles the neighbour index live and the HUD splits the frame into **maintain** (build
or relocate) vs **query** (the neighbour culls) vs **physics**.

```bash
cargo run -p vectorial-hash-demos --bin fluid_wgpu --release
```
Controls: **hold left mouse / drag a finger** to stir · `M` index · `[` `]` particles ·
`P` pause · `R` reset · `G` flip gravity.
Env: `FLUID_N` (particles), `FLUID_INDEX=morton|keep|linear`, `FLUID_MAX_FRAMES` (headless smoke).

## The measured head-to-head

2 200 particles, 420 frames, one PBF step per frame (3 constraint iterations) — **per
frame**, median of 3 passes:

| index | maintain | query | physics | fps |
| --- | ---: | ---: | ---: | ---: |
| `MortonGrid` (rebuilt each step) | 0.107 ms | **1.541 ms** | 1.082 ms | **337** |
| `Tree` + `ItemRef` (kept, relocated in place) | **0.031 ms** | 1.884 ms | 1.095 ms | 306 |
| `LinearQuadTree` (rebuilt each step) | 0.121 ms | 1.620 ms | 1.077 ms | 327 |

Numbers are the **median of repeated passes** taken by `bench-runner`, which waits for
the machine to be idle before each one (`cargo run -p bench-runner --release -- --group demos --repeat 3`).

Two things worth reading off that table:

- **The keep-index tree's maintain is 3.5–3.9× cheaper than either rebuild** — the
  `ItemRef` thesis, on a workload where *every* particle moves every step. It gives more
  than that back in query (+22%): a kept tree drifts from the ideal partition as the
  fluid sloshes, while a rebuild is always perfectly fitted. On this workload the trade
  is a losing one, and the demo says so.
- **The rebuilt structures are within 5% of each other on query**, with the flat
  `MortonGrid` marginally ahead of the adaptive `LinearQuadTree`. An earlier run of this
  table had the linear quadtree ahead; repeated passes on an idle machine do not support
  that — at this particle count the fluid's density is not skewed enough for adaptivity
  to pay for itself. (`LinearQuadTree`'s measured win is on the *static skewed* sets in
  `linear_quadtree_bench`, not here.)

The query dominates the frame in every mode, which is the honest headline: in an SPH
sim the *neighbour search is the simulation cost*, and the physics is comparatively
cheap arithmetic.

**This is the kit's first measured counterexample to "keep the index".** The siege demo
relocates 20 000 units per frame and keeping the index wins 1.05× (1 thread) → 1.50×
(16). Here every particle also relocates every step — and keeping loses the frame by 16%.
The two differ in how far an item moves relative to its leaf, and in how query-heavy the
frame is: SPH runs a neighbour query *per particle*, so partition drift is paid 2 200
times a step while the relocation saving is paid once. That was measured rather than assumed, and it settled which rule the advisor should
carry. Instrumented with `update_ref_tracked`, the fluid reports:

| | measured here | threshold | verdict |
| --- | ---: | ---: | --- |
| relocation rate (moves that leave their leaf) | **13.6%** | `HIGH_RELOCATION` 30% | would have said **keep** — wrong |
| queries per item per step | **1.00** | `rebuild_query_ratio` 0.10 | says **rebuild** — right, by 10× |

So the churn rule would have got the repo's only genuine counterexample backwards, and the
query-intensity rule gets it right. That is why `adaptive::Thresholds` switches on
`rebuild_query_ratio` and keeps `high_churn` only as a description of the workload.

## `C` — race every index on the state you are looking at

The HUD can tell you what the adaptive index *chose*. It cannot tell you whether that was right,
because it never runs the alternatives. `C` does: it clones the fluid, races all five choices from
the current state, and prints a verdict that stays on the HUD.

```
bake-off on the live state | 2200 particles | 120 frames per arm, min of 2 passes
  MORTONGRID REBUILD                  2865.4 us/frame
  MORTONGRID KEEP                     2878.0 us/frame
  TREE KEEP-INDEX                     3342.8 us/frame
  LINEARQUADTREE REBUILD              2987.6 us/frame
  ADAPTIVEINDEX2 (picks its own)      2712.1 us/frame
  -> best fixed: MORTONGRID REBUILD at 2865.4 us | adaptive: 2712.1 us (it is holding: grid)
  -> adaptive 1.06x the best fixed
```

`$FLUID_BAKEOFF=1` runs the same thing headless (after settling the fluid for 200 frames — racing
five indexes on frame zero measures a lattice no fluid ever looks like again).

Two details that are not decoration. The arms run **forward and then in reverse**, each keeping
its minimum: that gives every arm the same mean position in the sequence, so a machine that
drifts mid-bake-off cannot favour whoever went first — the counterbalancing argument in
[`MEASURING.md`](MEASURING.md) § 8e, applied to five arms instead of two. And it is a *minimum*
rather than a mean because this machine's noise is episodic and can only ever add time, so the
smallest reading is the closest to the truth.

## Running it without a screen

`$FLUID_HEADLESS=<frames>` runs the simulation with no window, no GPU and no wgpu adapter, then
prints the same per-phase split the HUD draws. The five-way index race was previously observable
only by a human watching bars move, which meant it could not run in CI, on a machine without a
display, or across a sweep of populations — a comparison this repo leans on quite hard for
something nobody could reproduce automatically.

```bash
FLUID_HEADLESS=400 FLUID_INDEX=adaptive cargo run -p vectorial-hash-demos --bin fluid_wgpu --release
```

2 200 particles, 400 frames after a warm-up, means over the run (`#M` lines are emitted for
`bench-runner`):

| index | maintain µs | query µs | physics µs | frame µs | sim fps |
| --- | ---: | ---: | ---: | ---: | ---: |
| MortonGrid rebuild | 117.9 | 1 525.5 | 1 098.3 | 2 741.6 | 365 |
| MortonGrid keep | 130.2 | 1 554.1 | 1 098.6 | 2 782.8 | 359 |
| Tree + `ItemRef` keep | **45.9** | 1 876.7 | 1 107.3 | 3 029.9 | 330 |
| LinearQuadTree rebuild | 151.4 | 1 616.0 | 1 111.1 | 2 878.6 | 347 |
| **AdaptiveIndex2** (picks the grid) | 135.9 | **1 360.5** | 1 104.9 | **2 601.2** | **384** |

It shares `Fluid::step` with the interactive path rather than reimplementing the loop — a
headless mode that reimplements the simulation measures the reimplementation. The warm-up is not
decoration either: the first frame pays for every bucket the index has never allocated, which is
a build cost wearing a maintain cost's clothes.

**The first version of this printed 3 420 ms per frame** and did not complain. `Fluid::step`
returns microseconds; the labels said milliseconds. Nothing in the code could have caught it —
only reading the number and knowing the demo runs at ~350 fps.

## Physics: PBF, not the textbook EOS

The demo solves **density constraints** (Macklin & Müller 2013) rather than the
classic Müller-2003 equation of state. The EOS formulation needs a sub-millisecond
time step to stay stable and its constants (mass ↔ rest density ↔ kernel radius) must
be mutually consistent or the density estimate silently collapses; PBF is stable at
`dt = 1/60` with a handful of iterations, which is what an interactive toy needs.

Rest density is **derived, not guessed**: it's the density the initial lattice
actually has. Two failures the headless smoke test caught (it reports simulation
health — density vs rest, top speed, particles in tank, finiteness — because the
window can't be inspected from a terminal):

1. **The relaxation ε is unit-dependent.** The paper's `600` lives in its units
   (ρ₀ = 1000, h = 0.1). Here the kernel scale puts `Σ|∇C|²` around `1e-2`, so a
   literal `300` swamped the denominator, `λ` went to zero, the solver quietly stopped
   solving — and the fluid compressed to **13× rest density**.
2. **Position-based solvers inject energy** through `v = (q − p)/dt`: a large
   constraint correction becomes a large velocity. Density held at 1.0× rest for
   ~20 frames while the top speed quietly climbed to 3× free-fall, and the fluid then
   tore itself apart. Fixed with an over-relaxation factor (< 1) and a hard per-
   iteration cap on how far one correction may move a particle.

Wall handling jitters each particle's stand-off: clamping everything to exactly `EPS`
welds a one-particle-wide column to the wall, which the density solver then drives
along it.

Health now: **~1.10× rest density** (peak 1.42×), top speed consistent with free fall
in the tank, stable over 420 frames.

## Why it's on the web too

The whole point is that you can poke it. The published build takes touch drags as well
as the mouse, and the mobile control overlay exposes `M` / `P` / `R` / `G` / `[` `]` as
buttons — see the [demo index](https://orlandoluque.github.io/vectorial-hash-kit/).
