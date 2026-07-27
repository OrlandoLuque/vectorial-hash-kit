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
