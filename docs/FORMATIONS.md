# Formations — two-level spatial reasoning (Total War-style auto-battle)

`formations_wgpu` is the third flagship 3D demo: two armies (pirates vs
undead, ~60 regiments, thousands of soldiers) deploy in lines and fight an
**automatic battle** to the rout — melee, flanking, cavalry charges, archer
volleys, morale and chain-routs. It exists to make one architectural point:

> **Reason at two levels and pick the right tool for each.** The regiment
> level (~60 centroids) maneuvers with *honest brute force* — at that N an
> index loses ([CHOOSING.md](CHOOSING.md)). The soldier level (thousands)
> is the library workout: every combat mechanic is a query on ONE shared
> `Tree3<ISoldier>` (both factions, filtered per query — the siege pattern).

```bash
cargo run -p vectorial-hash-demos --bin formations_wgpu --release --features parallel
```

Live (WebGPU): <https://orlandoluque.github.io/vectorial-hash-kit/formations_wgpu.html>

- drag: orbit · scroll: zoom · `P`/button: pause · `[` `]` + slider: army
  size (200 – 4 000/side) · `K`: frustum cull · `F`: free-fly (WASD/QE) ·
  thread slider (native). Sim: `src/formations_sim.rs` (graphics-free,
  11 brute-force-gated tests); renderer: `src/bin/formations_wgpu.rs`;
  research + tuning tables: [FORMATIONS_DESIGN.md](FORMATIONS_DESIGN.md).

## Every mechanic = a query

| Mechanic | Library primitive |
| --- | --- |
| Melee pairing | **k-NN** per engaged soldier on the shared index (enemies filtered by item faction) |
| Flank / rear bonus | **sector classify** of the hit vector vs the victim regiment's facing (+5 flank / +7 rear, the community-mined RTW/M2TW values) |
| Cavalry charge | the corridor is a thick **`raycast`** attacker→target — a friendly in the lane refuses the charge; the bonus decays over ~13 s |
| Archer volleys | each arrow lands as **k-NN(1)** at the impact point with **NO friend/foe check** — that IS the friendly-fire model (dev-confirmed for TW: Warhammer) |
| The general's aura | a **sphere cull** (r ≈ 50) — literally the aura case |
| Router panic | chain-rout / enemy-rout morale pulses over a **radius cull** (r = 80) |
| Formation slots | **no query at all** — rigid slots on a rotating frame (`slot_pos`), the cheap contrast to boids: wheeling's outer-file sweep falls out of the frame rotation |
| The regiment brain | ~60 centroids, **brute force** — the honest "index loses at small N" case |
| The soldier index | **keep-index** (`update_ref` movers, unmoved free) — the measured house rule |

## Numbers from the design research (FORMATIONS_DESIGN.md)

Close spacing 1.2 m (3.0 wu), 8 ranks deep; P(kill) = 1.9% × 1.2^cf with the
combat factor clamped ±20; charge decays over 13 s and needs a 30–85 wu
run-up; morale thresholds at 10/50/80% casualties → −2/−8/−12, routers get a
speed boost and don't fight back (pursuers take free kills), a third rout =
shattered (never returns). A seed fully determines the battle — the decide
pass is rng-free and parallel (rayon on native, serial on wasm), all rolls
live in the serial passes; there's a bit-identical replay test.

## Rendering

GPU skeletal skinning through the siege/horde machinery: one model per
(faction, kind) × (move clip, attack clip) — soldiers in an Engage(d)
regiment swing, routers run away facing backwards; cavalry and the generals
ride **horses** (mount + lifted rider, two instances). Regiment **banners**
(pole + flag boxes) read state at a glance: faction colour scaled by
strength, a white rag = routing. Arrows are line segments arcing with the
sim's own ballistics; the fallen stay as flattened dark bodies. ~7 000
soldiers ≈ 100 fps end-to-end (RTX 4080 SUPER, 1600×1000, no vsync).

## Still queued

KayKit/Quaternius banner + weapon props, terrain features that matter
tactically (a hill's charge bonus, forest cover), and a formations-vs-horde
crossover scenario. See [BACKLOG.md](BACKLOG.md).
