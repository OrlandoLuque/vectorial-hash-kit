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
dogpile). `HORDE_ACTIVE_CAP` / `HORDE_WAKE_RATE` override.

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

## Still queued

The Quaternius Ultimate Fantasy RTS wall/tower models to replace the
procedural boxes (manual download — the page is JS-driven), researching
TAB's actual map-generation algorithm for a TAB-like path-network scenario
option, and the user's own horde min-path idea (the flow field above is the
baseline to compare against). See [HORDE_DESIGN.md](HORDE_DESIGN.md) +
[BACKLOG.md](BACKLOG.md).
