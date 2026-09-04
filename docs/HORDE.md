# Horde — the sleeping index (They Are Billions-style assault)

`horde_wgpu` is the second flagship 3D demo: a fortified colony (wall ring,
towers, houses, Command Center) versus **tens of thousands of zombies — most of
them dormant** — with escalating waves, noise-wake cascades, breaches, repairs
and the infection chain. It exists to make two points concrete:

1. **Every mechanic is a spatial query** (the siege thesis, new verbs), and
2. **a mostly-sleeping population costs ~nothing to keep indexed** — the
   keep-index (`update_ref`) headline taken to its extreme.

```bash
cargo run -p vectorial-hash-demos --bin horde_wgpu --release --features parallel
```

Live (WebGPU): <https://orlandoluque.github.io/vectorial-hash-kit/horde_wgpu.html>

- drag: orbit · scroll: zoom · `P`/button: pause · **`N`/WAVE button: summon
  the next wave** (announce at 5 s; press again to land it NOW) · **`A`/ALL
  button: wake every sleeper at once** (the "what does 100k active cost"
  stress button) · **`G`: map scenario** (OPEN → PASS → RIVER → FOREST →
  PATCHES) · **`M`: index structure** (TREE3 ↔ MORTON, live) · **`O`: flow
  goal** (the CC ↔ every building — the multi-source field) · **`L`: night
  assault** (torches on the ring) · `[` `]` + slider: population
  (2 000 – 100 000) · `T`: tower targeting (nearest ↔ highest-threat) · `K`:
  frustum cull · `F`: free-fly (WASD/QE) · thread slider (native).
- Contact matters: a marching column **tramples sleepers awake** as it rolls
  over them (walking is silent — aggro spreads via combat noise and touch).
- Rendering: **impostor billboards** ("photos" of each model: 8 yaws × 3
  elevation bands × walk/idle/death frames, captured at startup) draw
  everything beyond ~170 wu; the camera bubble gets real GPU-skinned models
  (sleepers in it play their real Idle clip); corpses play the Death clip once
  from the moment they fall. The sleeping carpet is a static buffer (zero
  upload between changes, throttled rebuilds).
- Design + research (the TAB noise model with numbers, scenarios, defender AI):
  [HORDE_DESIGN.md](HORDE_DESIGN.md). Sim: `src/horde_sim.rs` (graphics-free,
  brute-force-gated tests); renderer: `src/bin/horde_wgpu.rs`.

## Every mechanic = a query

| Mechanic | Library primitive |
| --- | --- |
| Noise wakes sleepers | each noise event = **one sphere cull**; per-class hearing radius (4×watch) + threshold (1000/alertness, the TAB rule) |
| The infection cascade | a populated house falls → runners spawn + one 50·pop noise **detonation** (wakes a region) |
| Tower "nearest" | **k-NN(1)** per shot — the *static consultant* case |
| Tower "highest threat" | range **cull + score-max** (chubby > harpy > venom…) |
| Zombie melee target | k-NN(4) on the **static structure index** for the nearest LIVE piece; venom spits from 36 wu (outranges walls); harpies fly over (3D k-NN skips the wall under them) |
| Swarm routing | flow field to the CC where **walls are cost, not blockers** (breaches reroute the flood; a rebuilt wall re-deters gradually) |
| Commander's threat map | one **counting cull** per wall sector (1 Hz) + wave-warning anticipation |
| Noise discipline | silent rangers only, until the committed-push culls trip weapons-free |
| Crew/porter safety | repair jobs gated by a **safety cull**; a breach **recalls** works units in radius |
| Static base | built once with **`bulk_load`** (`bulk_load_par` with `--features parallel`) |
| The dormant horde | **keep-index**: sleepers never move → skipped by the moved-only sync |
| The horde's minimum paths (`G` scenarios) | blocked cells (ridge/water/woods) **never relax** in the flow-field Dijkstra — the flood funnels through the pass gaps / causeways / forest trails on its own |
| The multi-goal flood (`O`) | seed 0 at **every live building** and flood once — a **multi-source Dijkstra**: the field holds distance-to-nearest-goal, so N goals cost the same one flood as 1, and it re-routes to the survivors as buildings fall (measured: 31 goals ≈ 1 200 µs, *identical* to the single-CC rebuild) |
| The defenders' minimum paths | **A\*** over the same passability grid: one search per ranger-squad sortie (out the gate, along the trails) + a trail home on recall |
| The index itself (`M`) | live **Tree3 (keep, `update_ref`) ↔ MortonGrid3 (clear+reinsert per frame)** behind one query enum — the CHOOSING.md trade-off, watchable in the fps counter |

## The headline (measured, `examples/horde_bench`, 16 threads)

```
      pop |  all dormant        cascades woken        mass assault
   20 000 |  0.20 ms (5044 fps)  3.5 ms (290 fps)     5.0 ms (199 fps)
   50 000 |  0.29 ms (3498 fps)  10.8 ms (93 fps)     25.2 ms (40 fps)
  100 000 |  0.42 ms (2369 fps)  23.1 ms (43 fps)     100 ms (10 fps, ~110k ACTIVE)
```

**100 000 zombies indexed and asleep cost 0.42 ms/step.** Cost scales with the
*active front*, not the indexed population — sleepers are skipped by the
moved-only `sync_index` and early-outed in `decide`; only wake culls ever touch
them. (The all-active 100k row is the honest ceiling: the scale-pass levers —
decision buckets at 4–8 Hz, activation bubbles — are the queued next round.)

**End-to-end rendered FPS** (same machine: RTX 4080 SUPER, 1600×1000, whole map
in view, no vsync — `HORDE_NOVSYNC=1 HORDE_MAX_FRAMES=N` prints the average;
`HORDE_NOLOD=1` = the pre-LOD path for the A/B):

```
   pop      everything skinned (old)     LOD render pass (shipped)
  20 000        ~56 fps                     ~1136 fps    (20×)
  50 000        ~21 fps                      ~816 fps    (38×)
 100 000        ~11 fps                      ~475 fps    (42×)
```

The old path was exactly the **vertex-bound wall** the crowd research predicted
(~1.5–3k skinned verts × every zombie + a full instance-buffer re-upload each
frame). The shipped render scale pass removes it three ways:

1. **The sleeping carpet is a static instance buffer of ~72-tri slump proxies**,
   rebuilt only when the sim's `dormant_epoch` moves (a wake/re-sleep/death) —
   between changes the 100k sleepers cost zero CPU and zero upload. Visual
   bonus: sleepers are slumped shapes, so a wake wave literally **stands up**
   out of the carpet into the full animated model.
2. **Distance LOD for active zombies** (`LOD_DIST` = 620 wu): near = full
   GPU-skinned glb, far = a standing proxy. Zoomed out the horde is proxies
   (they're a few pixels anyway); zoom in and the front line becomes real
   models. Defenders (~50) are always skinned.
3. **Corpses are append-only**: only new bodies upload each frame.

The CPU sim was never the limit here (0.2–5 ms). Remaining refinement, queued:
textured billboard impostors to replace the far proxies (nicer silhouettes),
and the CPU-side decision buckets for the all-active 100k case.

## The loop

Waves land every ~70 s with a **direction warning + countdown** (banner), then
路 escalate (runners → venoms/chubbies → harpies); wave 8 is the **final**: all
four edges at once plus every dormant zombie left on the map. Between waves the
Commander schedules repairs (crews only work while **porters** haul stock — the
ant lines are the repair pacing) and the towers' own noise quietly pulls
stragglers in. The Command Center falling = defeat; clearing the final wave =
survival; either way the run resets with a fresh map a few seconds later.

## Where the frame budget lives now (measured)

Drawing is solved (impostors: 100k dormant ≈ 740+ fps) and sleeping is solved
(keep-index: 0.42 ms). The one remaining cost is **thinking while awake**:
~1 µs/zombie/frame of queries (separation cull + flow + targeting +
`update_ref`). CPU-only: ~35k active = 93 fps, ~110k active = 10 fps — and the
rendered FPS matches that curve exactly (the GPU idles). Three feedback loops
that used to ignite the whole map in seconds (per-frame carpet rebuilds,
contact-wake percolation, walking noise) are fixed; a wave now recruits
linearly along its path and via wall-combat noise.

### Decision buckets (measured, `$HORDE_DECIDE_N`)

A zombie away from the walls doesn't need a fresh plan every frame — it coasts on
its **cached velocity** between decisions while movement/heard/swings still run at
full rate (and near-wall combat is never bucketed). Staggering the re-decide by id
turns the decide pass into an every-`N`-frames cost. Sweeping `N` on the 100k mass
assault (~53k active, 16 threads):

```
  N   rate     ms/step   fps    Δ vs 15 Hz
  4   15 Hz     5.41     185     — (old default)
  8   7.5 Hz    4.44     225     +22 %
 15   4 Hz      4.09     244     +32 %
```

**Default is now `N=8` (7.5 Hz)** — it banks most of the win (+22 %) with steering
still visually coherent; 8→15 buys only +8 % more for coarser paths (the decide
pass is already a shrinking slice — movement, the separation cull, `apply` and the
index sync all run every frame). `$HORDE_DECIDE_N` overrides it live.

## Scenarios (`G`) — the same sim, five worlds

**OPEN** is the classic ring-in-a-field. **PASS** raises a rock ridge at
r≈330 with three gap passes; **RIVER** carves a meandering channel with two
causeway crossings; **FOREST** is impassable woods with a carved base
clearing, four winding gate trails, nest clearings and connectors — the
walkable network is *visible* as gaps in the tree canopy. **PATCHES** is the
**They Are Billions** structure (researched): mostly-open ground strewn with
blob patches of forest / rock / water at theme-weighted densities (three
independent value-noise masks), where the walkable network is the *residual*
space between blobs — gorges, small plains, pockets and chokepoints emerge
from how patches touch, nothing is hand-traced. A connectivity pass (flood
from the CC, then carve a corridor to each big unreached pocket) gives TAB's
playability guarantee; small pockets stay as dead-end recovecos on purpose.
(`HORDE_SCENARIO=PATCHES` boots straight into it.)

Impassable terrain turns both routing systems into real minimum-path solvers
(table above); ground units axis-slide along blocked edges, harpies just fly
over, wave landings resample until they hit passable, *reachable* ground.
All of it brute-force-gated: flow reaches the CC from 8 bearings per
scenario, no ground unit ever stands in a blocked cell, PATCHES stays >90%
connected and in a sane open-fraction band across 10 seeds, Morton culls ==
Tree culls == brute force.

**Forests are line-of-sight cover** (`los_clear`, a new query verb — a segment
sampled against the canopy grid). In FOREST and PATCHES a tower will not fire on a
zombie whose sight line crosses dense canopy: the tree network becomes safe
approach, so the horde filters *through* the woods and only takes tower fire once
it steps into the open (the TAB rule). Both tower targeting modes (nearest-`knn`
and threat-`cull`) pick the best zombie with a *clear* line instead of the best
zombie outright. Open scenarios short-circuit (no canopy → every line clear).
Brute-force-gated (`forest_canopy_is_line_of_sight_cover`): a line through canopy
is blocked and matches a dense reference sample, the woods give *partial* cover
(some lines blocked, not all), and OPEN/PASS/RIVER are always clear.

## Wake the whole horde (`A`) — 100k active, measured

The `A` key / ALL button (`wake_all`) rouses **every** sleeper into the march
at once — the worst case the demo can produce. Rendered end-to-end with all
100 000 zombies awake and marching (RTX 4080 SUPER, 1600×1000, no vsync):
**~79 fps** (`HORDE_POP=100000 HORDE_WAKE_ALL=1 HORDE_NOVSYNC=1
HORDE_MAX_FRAMES=600`). That's the honest ceiling with the entire indexed
population active — every zombie thinking, moving and re-syncing every frame,
plus the impostor draw for all of them. Normal play never wakes everything at
once (only the final wave does), so this is the stress bound, not the
steady state.

## The awake-front cap + stagger — 100k playable (measured, 2026-07-24)

`wake_all` is the *stress* bound; the *steady state* is governed by a **soft cap
on the simultaneously-active front**. The dormant carpet feeds the front up to
`active_cap`; the rest stay a reserve that rises only as the front is thinned, and
new wakes are also **rate-limited** (`wake_rate`/s) so the front *ramps* toward the
cap over ~45 s of pressure instead of saturating in one frame. Both the ambient
noise/contact cascade and the final-wave surge are metered through it (the surge
drains the whole reserve through the bounded front — a playable grind, not a single
dogpile). `HORDE_ACTIVE_CAP` / `HORDE_WAKE_RATE` override. On screen it's legible: a
**yellow tick on the dormant|active bar** marks where the front tops out (when the
green|red boundary reaches it, the active set is at the cap and the rest of the horde
is a dormant reserve — the green `SLP` count), and the `ACT n/cap` readout shows the
front against its ceiling.

**Why it's needed** (headless `examples/horde_balance`, seed 7, Patches): without
the cap the wake cascade scales ~linearly with population while the garrison
scales only ~pop^0.72, so **100k died at wave 1** (an instant dogpile). With the
cap the front is bounded and the fight is a real siege:

| pop | before (uncapped) | after (cap 2600 + stagger) | front (awake) |
|----:|-------------------|----------------------------|--------------:|
| 20k | fight, DEFEAT ~w5 | HELD, wave-8 surge scare (CC→71%) | ~500–2700 |
| 50k | overrun fast      | **HELD ≥ wave 8**, CC 100% | ~700–2700 |
| 100k| **DEFEAT wave 1** | **HELD ≥ wave 8**, CC 100% | ~900–2600 |

Measured lesson: there's an **absolute front ceiling (~2600)** above which the
defence collapses no matter how big the garrison — breaches open faster than crews
repair, so a 5k front routs even 244 fighters while a 2.6k front is held by 126.
So the cap is **flat**, not pop-scaled: a 100k horde plays like the tuned 20k
fight with a far deeper reserve behind the *same* sustained front — and because the
active set is bounded, the per-frame decide cost is too, so **100k renders fast**
(the front, not the population, sets the cost). `active_cap_bounds_the_awake_front`
brute-force-gates the invariant.

### The garrison: gentle scaling, anchored at 20k (measured, 2026-07-24)

Because the front is now flat-capped at every population, the garrison that makes
the tuned 20k fight (**~127 fighters** holding a 2.6k front with a wave-8 scare)
is the *right* garrison at every population. So the fighter count no longer scales
`pop^0.72` (the steep curve the *uncapped* cascade once needed — it over-garrisoned
big hordes into a trivial hold, the inversion where **more pop was easier**). It
now scales gently, anchored at 20k: `2.4·(pop/20k)^0.10` (`HORDE_FIGHTER_EXP`).
Higher populations get *harder* through **reserve depth** — a far longer surge
grinding the same line down — not a bigger instant garrison:

| pop | fighters (was → now) | outcome (seed 7, to wave 12) |
|----:|----------------------|------------------------------|
| 20k | 126 → **127** | HELD, CC 100%, walls 117→85 |
| 50k | 244 → **140** | HELD, **CC 95%**, walls →88 (a real fight) |
| 100k| 401 → **149** | HELD, **CC 90%**, walls 117→70 (hard siege) |

Now the CC damage and wall attrition *rise* with population (100/95/90 %) instead
of inverting — a genuine escalating fight at every size, survivable but costly
(consistent across seeds 3/7/42). Balance is measured, not guessed; `horde_balance`
is the harness.

## The multi-goal flow field (`O`) — the user's idea, measured

The baseline flow field floods from **one** goal (the CC). A colony has many
buildings, though, and TAB zombies want to level all of them. The user's idea:
*put a 0 on every goal cell and propagate outward (1, 2, 3…), with the
relaxation "if I'm a 2 and a neighbour is 4+, I become a 3" — then agents just
walk downhill.* That is exactly a **multi-source BFS/Dijkstra integration
field** (the relaxation IS edge relaxation), and the horde already ran the
single-goal version. The `O` toggle turns on the multi-source variant: seed 0
at **every live building**, one flood; the field then holds the distance to
each zombie's *nearest* building and re-routes to the survivors as buildings
fall — verified by a brute-force test (12 ring bearings all descend to a LIVE
building, before and after half the colony is demolished).

The headline the user predicted — **N goals cost the same as 1** — measured
(`examples/horde_bench`, 150-cell field, 31 building goals):

```
   map      single-CC goal      multi-building (31 goals)
  OPEN      1225 µs             1200 µs     (slightly FASTER)
  PATCHES    748 µs              743 µs
```

It's one Dijkstra either way; more seeds just start the wavefront wider, so it
terminates a touch sooner. Moving targets (the ~50 defenders) are deliberately
*not* in the field — reflooding per frame would be the "locura" the user
flagged; zombies find them with a cheap local brute-scan k-NN instead. (The
named incremental option for moving goals is **D\* Lite**; overkill at 50.)

## The molón round (`L`)

Night assaults: the sun drops to a cold moon and up to 64 **torches** light
the ring (towers, gates, every 7th wall piece, the CC — a fallen tower's
light dies with it), flickering per-pixel in the terrain/skin/billboard
shaders. Fresh corpses spill **blood pools** and a **kill ring** (ground
decals, one instanced draw); breaches, wave landings and the run ending
kick a **trauma camera** (rotational shake ∝ trauma², the GDC rule).

## The `M` toggle's grid: cubic cells, and what the toggle actually shows (2026-07-31)

`M` swaps the zombie index between `Tree3` + `ItemRef` and `MortonGrid3`. The grid was built on
the play area — 1800 × 72 × 1800 — and since `MortonGrid3` derives one cell side from `levels`
for all three axes, that produced cells of **56.25 × 2.25 × 56.25**: slabs. Declaring the index
world as a **cube** instead costs nothing (the cell store is a sparse hash, so the layers above
the play area are never stored and never traversed) and it frees `levels`, which was pinned at
5 because finer slabs sliced the 72-unit axis into ribbons.

Measured on the real population and the real query mix (`examples/horde_grid_shape`, 50k units,
against the shipped slab grid at levels 5):

| query | slab L5 | cube L5 | slab L6 | **cube L6** | cube L7 | cube L8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cull r=3 — separation, once per awake unit | 1.00× | 1.06× | 1.26× | **1.72×** | 2.17× | 2.19× |
| cull r=55 — guard | 1.00× | 1.69× | 0.51× | **1.84×** | 1.14× | 0.27× |
| cull r=84 — tower ring | 1.00× | 2.28× | 0.38× | **1.91×** | 0.79× | 0.14× |
| cull r=110 — sector | 1.00× | 2.61× | 0.34× | **1.69×** | 0.57× | 0.09× |
| k-NN k=8 — tower aim | 1.00× | 1.53× | 1.01× | **1.96×** | 2.48× | 2.29× |
| k-NN k=48 — commander | 1.00× | 1.21× | 1.09× | **1.63×** | 1.74× | 1.34× |

Small queries want fine cells, big ones want coarse cells, so most columns win somewhere and
lose somewhere — `cube L6` is the only one that never drops below **1.63×**, and that is what
ships. The prediction going in was that the radius-3 separation cull would *lose* under cubic
cells, since one cubic cell holds the whole vertical column where a slab holds a sliver of it.
It does test more points (316 vs 290 at L5) and it is still faster: the lookups saved outweigh
the extra distance tests. L7 and L8 win that query outright and then fall off a cliff on the
rings, down to 0.09×, which is the occupancy rule biting from the other end.

### And then the grid stopped rebuilding

The paragraph that stood here said the toggle was not a speed option, because `M` mode
**rebuilt all 50 000 units every frame** while the tree kept them and skipped the sleepers.
That was true of the code and false of grids: `MortonGrid3::update` now moves an item in place
given where it was, so the grid gets the same keep path the tree has had all along — a `zlast`
table of one `Point3` per unit standing in for `ItemRef`, and dormant zombies skipped entirely
because they never set `moved`.

| 50k units, ms/step | before | after |
| --- | ---: | ---: |
| `Tree3` + `ItemRef` (default) | 0.65 | 0.47 |
| `MortonGrid3` (`M`) | 7.31 | **2.77** |

**2.6× end to end**, on top of the 1.15× the cubic cells bought. The grid is still ~6× the
tree rather than ~11×, and what remains is now honestly the *queries* — maintenance is no
longer where the toggle loses. Verified the way the siege keep-index was: a test steps 150
frames in Morton mode and asserts the maintained grid holds and **answers** exactly what a grid
rebuilt from the live units would, deaths included. Its first run failed on its own
non-vacuity guard — nothing had died in 2.5 simulated seconds, so the removal path was never
reached — which is why it now kills 25 units outright partway through.

The keep/rebuild crossover for a grid sits near **70 % of the population moving per frame**
(`examples/grid_keep_bench`); the horde is nowhere near it, with almost everything asleep.

## The wall ring: towers first, then fit the walls to the arc

The ring used to be cut into equal 8-wu slots, every sixth one made a tower, and the *renderer*
slid tower-adjacent walls sideways to hide the resulting seam. That is a patch over a wrong
layout — a tower's slot had nothing to do with a tower's footprint (it is ~5.4 wu, not 8), so
pieces overlapped each other and the towers. The layout is now the user's method, and it lives
in `horde_sim::build_base` because positions drive the flow field, breaches and collision, not
just the picture:

```
D    = arc between tower centres       X    = tower centre → first wall's near edge
span = D - 2X                          run  = (span - gate_w) / 2      (a gated arc has two)
I    = ceil(run / wall_w)              step = wall_w - (I·wall_w - run) / I
```

`RING_TOWERS` towers are placed first; each gets its arc, and the walls are *fitted* into what
is left, with the rounding absorbed by overlapping them all by the same sliver rather than
leaving a gap at one end. With `RING_TOWERS` a multiple of 4 the four gates land exactly on the
cardinals, because a gate sits at the centre of an arc. `TOWER_WALL_X = 0.25` reproduces the
1/4 · 3/4 rule symmetrically: each neighbouring wall laps a quarter of the tower from its own
side. Solid widths live in `ring_footprint()` next to the radius, so they cannot drift from the
model scales in `horde_wgpu::building_tweak`.

**Checked as an inequality, not by eye.** `the_wall_ring_closes_with_no_gaps_and_no_pile_ups`
converts every piece to an arc position and half-width and asserts no hole, an overlap that
exists but is smaller than one wall, and the expected tower and gate counts. Both failure modes
have shipped before (walls skipped entirely; then jammed together), and neither was caught
reliably by looking at a screenshot. Verified to fail when the fit's `ceil` becomes `floor`.

## `M`: the third index mode is one that picks itself

`M` used to toggle Tree3 ↔ Morton. It now cycles a third: **ADAPTIVE**, an `AdaptiveIndex` that
changes structure underneath the sim as the battle does. The title names the structure currently
live and the switch count — `ADAPTIVE(Grid, 3 sw)` — because a label that only ever said
"ADAPTIVE" would hide the one thing worth watching.

This demo is the workload that layer exists for, and the reason it is worth wiring here rather
than anywhere else: the horde swings from a dormant carpet that never moves and is barely queried
to a 50k assault where everything relocates and every awake unit culls its neighbourhood. That is
the churn × query-load plane traversed live, in one run, without anyone scripting it.

Two things it needed from the library, both of which were missing and are general:

- **`settle()` + `cull_ref()`.** Every system here queries behind `&`, and `cull` takes `&mut`
  because it counts the query as it answers it. Splitting the observing from the answering is
  what lets a dozen systems read one index in a frame.
- **`note_queries`.** The reads through `cull_ref` cannot count themselves, and a policy that
  cannot see the queries is worse than one with none: it concludes the workload is idle and
  migrates for something that is not happening. The sim reports `active_n` as the count, which is
  an estimate — one cull per awake unit is the dominant query — and is labelled as one where it
  is passed.

`set_zmode(Adaptive)` uses the documented bulk sequence: `prepare` the expected population,
`freeze` while the units are loaded in, `thaw` after. Without the freeze the population climbing
from zero makes the policy migrate away from the destination and back, which is measurably worse
than not hinting at all.

**Measured** (`examples/horde_index_modes`, one seeded battle, 30 000 zombies, 900 frames — read
the ranking, the absolute milliseconds are whatever machine ran it):

| index | ms/step | |
| --- | ---: | --- |
| `Tree3` pinned | 1.611 | the best fixed choice here |
| `MortonGrid3` pinned | 6.087 | the wrong fixed choice, **3.8x worse** |
| adaptive | 2.220 | **1.38x the best** (range 1.34–1.61), found without being told |

Three switches, 29 near-misses. It did not beat the best fixed choice — it cost ~38 % more than
one, and 2.7x *less* than the other one. That is the argument for the layer stated fairly: it is
worth having when you cannot know which of those two you would have picked.

*(This first read 1.03×, from an example that ran each arm once in a fixed order. Five runs of
that version gave 1.03–1.67; the published figure was the best draw. It now rotates the arm order
per round and quotes the median with its range.)*

*(And the honest figure is **~1.4×, 1.34–1.47 across sessions**, not any single session's median.
Rotating the arms fixed drift *within* a run — a later session read a range of 1.46–1.47, tight —
but it cannot fix the difference between one evening's machine and another's, where the same three
arms read `Tree3` 1.485/1.611/1.831 and `MortonGrid3` 4.643/6.087/8.233 ms/step. The switch
statistics, being counts, were identical in all three: 3 switches, 29 near-misses,
`Brute→KeepTree ×1`. **Quote the counts as facts and the ratio as a range.**)*

**A contradiction, and its resolution: the missing axis was QUERY EXTENT.** `grid_tree_frontier`
maps the (churn × query-load) plane and said the grid should win *by 1.9×* at this very point —
the horde reports `0.0599 q/item, 0.0564 mv/item`, landing squarely in that cell — while the horde
measures the grid **3.8× worse**. Two of our own measurements disagreeing by ~7× at the same
coordinates meant one of them was answering a different question.

Both were right about their own slice. That plane was measured at **one radius, 36**, which is one
grid cell wide by construction — the extent most flattering to a grid. Re-running it at
`radius=8` flips almost the whole table to the tree; `pick_a_structure` at the horde's exact
`churn=0.056 queries=0.06` reads the **keep-tree** ahead at radius 1–4, the **grid** ahead from 12
to 90, and a **brute scan** ahead at 170. Three different winners at one point of the plane.

That is the horde: its highest-frequency cull is the radius-**3** separation query, which is the
regime where the tree wins, and its grid must serve radii from 3 to 300 with a single cell size
while every arm above was re-sized to its own radius. Clustering was checked and is *not* the
cause — the grid still beats the tree on clustered data at radius 36 and 90.

Re-run either side: `--example grid_tree_frontier -- radius=8`, or
`--example pick_a_structure -- n=30000 churn=0.056 queries=0.06 radius=3`.

### What `G` and the population slider actually cost (#166)

`adaptive_lab` hung for seconds because it rebuilt inside a frame, so the horde was checked for the
same shape — `set_population` and `set_scenario` both call `Horde::with_scenario`. Measured
(`--example horde_rebuild_cost`, min of 3):

| pop | OPEN | FOREST | PATCHES | frames @ 60 Hz |
| ---: | ---: | ---: | ---: | ---: |
| 2 000 | 20 ms | 17 | 56 | 3 |
| 30 000 | 68 | 48 | 93 | 6 |
| 100 000 | 77 | 60 | 99 | 6 |

**Worst case ~103 ms — a hitch, not a hang**, so the suspicion was wrong and nothing needs
spreading. Two things worth keeping from it anyway. The cost is **nearly flat in population** (77 ms
at 100 k against 20 ms at 2 k on the same scenario) and rises with scenario complexity instead:
what is expensive is generating the world — the passability grid and PATCHES' connectivity flood —
not placing zombies in it. And the measurement covers `Horde::with_scenario` only; a real `G` press
also rebuilds the terrain mesh and re-uploads GPU buffers, so the number is a floor for the stall,
not the whole of it.

**Counted, not timed** (`--example horde_query_counts --features sim-counters,grid-stats`): over
one 900-frame battle the grid asks for **6.51×** the points the tree does, and looks up **140 M
cells**. Two hypotheses were tested against that and both are refuted:

| suspect | verdict |
| --- | --- |
| k-NN (towers k=8, commander k=48) | **innocent** — 1.6–1.9× the tree's points, 18–30 cells |
| index maintenance (a grid `update` scans a cell) | **innocent** — with waves off and **peak awake = 0** the ratio *grows* to 8.18× |

What is left is the standing defence sweeping the map with **large rings that find nothing**. Per
query, at the front:

| call site | tree pts | grid pts | grid cells | hits |
| --- | ---: | ---: | ---: | ---: |
| separation (r 3) | 15 | 15 | 2 | 0 |
| guard ring (r 55) | 159 | 315 | 60 | 207 |
| tower ring (r 84) | 203 | 354 | 196 | 548 |
| pack ring (r 110) | 152 | 291 | 360 | 774 |
| **wave ring (r 300)** | **325** | 1 001 | **6 072** | 4 341 |
| k-NN k=8 | 59 | 95 | 18 | 8 |
| k-NN k=48 | 129 | 239 | 30 | 48 |

The horde's *most frequent* query, separation at radius 3, is a dead heat — 15 points each. The
cost is in the rings, and it is cell lookups rather than point tests, which no `position()`
counter alone could have shown.

So the horde **confirms the mechanism** behind `Thresholds::grid_min_hits` (a grid pays for cells
whether or not they hold anything) while also being the data shape whose density that rule
estimates wrongly — a carpet is a slab in a cube. Right mechanism, wrong input: #154.

*A note on the demo itself, found by a non-vacuity guard in that example:* running the same seed
under `M`=tree and `M`=grid does **not** produce the same battle. They diverged by three units at
radius 55 after 400 frames, because the two indexes return hits in different orders and the sim
picks targets from that order. Harmless for the demo, and the reason the per-query table builds
its grid *from* the tree's contents rather than from a second battle. It is also, precisely, the
R1 canonical-ordering problem `AdaptiveIndex` solves and the raw structures do not.

**The policy learned it — and it changed nothing here, which is worth saying.**
`Thresholds::grid_min_hits` now vetoes the grid when a typical cull is not expected to find much;
`examples/extent_axis` puts the crossover at ~9 expected points per query, measured at two
densities 4× apart so that radius and points-per-query made *different* predictions and only one
could survive. The horde clears that veto by a wide margin — and its adaptive arm behaves
identically before and after (**3 switches, 29 near-misses, `Brute→KeepTree ×1`**, unchanged),
because `q/item = 0.0599` is already under `rebuild_query_ratio = 0.2` and the grid was never
reached for the veto to block. The horde is the *evidence* for the surface's shape, not a case
the fix repairs.

Read the ms column across the two runs with that in mind: `Tree3` 1.611 → 1.485 and `MortonGrid3`
6.087 → 4.643 ms/step between them. Nothing in the fix touches the Morton arm, so a 24 % move
there is the machine getting quieter, and the ratio's apparent improvement (1.38× → 1.34×, ranges
1.34–1.61 and 1.17–1.45) is not attributable to the change. **A number that improves alongside a
change you made is not evidence the change made it** — check whether the arm you did not touch
moved too.

**Tested, and not only for equivalence.** `adaptive_mode_matches_the_others_and_actually_switches`
checks the answers against the tree AND against brute force after a real battle — but it also
asserts the policy moved or at least wanted to. A demo where the switcher silently sat on one
backend all night would pass an equivalence test while demonstrating nothing.

## Looking at it without eyes (`$HORDE_SHOT`)

The renderer can take its own screenshot, so a question like *"does the wall actually meet the
tower?"* stops being a thing to queue for the next time a human is at the machine.

```bash
HORDE_SHOT=out.png HORDE_SHOT_AFTER=90 horde_wgpu          # sim 90 frames, then shoot and exit
HORDE_CAM_YAW=0.0 HORDE_CAM_PITCH=1.40 HORDE_CAM_DIST=150  # frame it — nobody is dragging
```

The shot frame renders into an offscreen `COPY_SRC` texture rather than the swapchain, so it
does not depend on the surface format allowing copies, and works the same whether or not the
window is visible. Readback goes through a buffer whose rows are padded to
`COPY_BYTES_PER_ROW_ALIGNMENT` — skip that and the image shears — and BGRA is swizzled to RGBA
on the way out, because the Windows swapchain is BGRA and PNG is not; skip *that* and you get a
believable frame with an orange sky. The encoder (`src/png.rs`) writes uncompressed stored
deflate blocks, so a 1600x1000 frame is about 6 MB and the whole thing stays readable.

The machinery lives in `vectorial_hash_demos::shot`, not in this binary, so the other seven wgpu
demos can adopt it in three lines (`Shot::from_env("SIEGE")`, `Target::begin`, `target.finish`)
rather than each re-deriving the row padding and the BGRA swizzle. `siege_wgpu` and
`formations_wgpu` are the next candidates — both already have their own offscreen path for a
headless bench, so wiring them is a merge rather than an addition.

**A screenshot is not a golden-image test here.** Two runs of the same binary with the same seed
and the same frame count produce different bytes: the animation phase is driven by wall clock and
the HUD prints an FPS. So it verifies by being *looked at*, not by being hashed — which was worth
finding out before writing a comparison that would have failed on its first green run.

**Its first use also shows the limit of the tool, and it is worth writing down.** A top-down
shot was read as confirming the wall ring's geometry — closed, towers at the corners and
mid-spans, walls attaching off-centre. It was not the ring. At that camera distance the frame
was filled by the Command Center's castle model (scale 40) and the ring, radius 150, was off
frame entirely; the "corners and mid-spans" were the castle's own towers. Framing a shot is
part of the measurement, and a picture of the wrong thing is as confident-looking as a picture
of the right one. The ring's geometry is now checked by the test above instead.

What still needs a human is `building_tweak`'s per-model scale, which is taste rather than
correctness — and that is the useful split.

## Still queued

The Quaternius Ultimate Fantasy RTS wall/tower models to replace the
procedural boxes (manual download — the page is JS-driven), researching
TAB's actual map-generation algorithm for a TAB-like path-network scenario
option, and the user's own horde min-path idea (the flow field above is the
baseline to compare against). See [HORDE_DESIGN.md](HORDE_DESIGN.md) +
[BACKLOG.md](BACKLOG.md).
