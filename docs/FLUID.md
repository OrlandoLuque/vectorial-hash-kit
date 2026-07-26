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

2 200 particles, 420 frames, release build, one PBF step per frame (3 constraint
iterations), RTX-class box — **per frame**:

| index | maintain | query | physics | fps |
| --- | ---: | ---: | ---: | ---: |
| `MortonGrid` (rebuilt each step) | 0.21 ms | 1.90 ms | 1.82 ms | 254 |
| `Tree` + `ItemRef` (kept, relocated in place) | **0.06 ms** | 2.22 ms | 1.76 ms | 253 |
| `LinearQuadTree` (rebuilt each step) | 0.21 ms | **1.84 ms** | 1.40 ms | **269** |

Two things worth reading off that table:

- **The keep-index tree's maintain is ~3.5× cheaper than either rebuild** — the
  `ItemRef` thesis, on a workload where *every* particle moves every step. It gives
  part of it back in query (a kept tree drifts from the ideal partition as the fluid
  sloshes, while a rebuild is always perfectly fitted).
- **The adaptive `LinearQuadTree` wins the query.** This is its measured niche —
  skewed data you rebuild often — showing up on a real workload rather than a
  synthetic bench: the fluid is dense where the water is and empty everywhere else,
  which is exactly what a uniform grid can't exploit.

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
