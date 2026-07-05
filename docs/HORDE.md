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
  the next wave** (announce at 5 s; press again to land it NOW) · **`G`: map
  scenario** (OPEN → PASS → RIVER → FOREST) · **`M`: index structure**
  (TREE3 ↔ MORTON, live) · **`L`: night assault** (torches on the ring) ·
  `[` `]` + slider: population (2 000 – 100 000) · `T`: tower targeting
  (nearest ↔ highest-threat) · `K`: frustum cull · `F`: free-fly (WASD/QE) ·
  thread slider (native).
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

## Scenarios (`G`) — the same sim, four worlds

**OPEN** is the classic ring-in-a-field. **PASS** raises a rock ridge at
r≈330 with three gap passes; **RIVER** carves a meandering channel with two
causeway crossings; **FOREST** is impassable woods with a carved base
clearing, four winding gate trails, nest clearings and connectors — the
walkable network is *visible* as gaps in the tree canopy. Impassable terrain
turns both routing systems into real minimum-path solvers (table above);
ground units axis-slide along blocked edges, harpies just fly over, wave
landings resample until they hit passable, *reachable* ground. All of it
brute-force-gated: flow reaches the CC from 8 bearings per scenario, no
ground unit ever stands in a blocked cell, Morton culls == Tree culls ==
brute force.

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
