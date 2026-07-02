# FORMATIONS — design doc (Total War-style army battle, automatic)

Research notes + build plan. Status: **researched 2026-07-02, not started**.
Self-playing battle between two armies of **regiments** — the showcase is
**two-level spatial reasoning** (regiment maneuver vs soldier combat) and
**directional queries** (flank/rear wedges → morale). Shared graphics-free sim +
both renderers, native + wasm. Companion: [HORDE_DESIGN.md](HORDE_DESIGN.md).
Lives in `vectorial-hash-demos`.

## Concept

Two armies deploy in historical lines: infantry centre, archers behind, cavalry
wings, a general with an aura. Regiments (~60–160 soldiers each, ~30–60
regiments/side) maneuver as blocks — soldiers hold **formation slots**; combat
resolves at the contact line soldier-vs-soldier; **charges** carry momentum;
**morale** (casualties, flanking, general, chain routs) breaks regiments into
routs, pursued by cavalry. Battle plays out to a victor, then resets with a new
seed/army composition. Readability is the point: coherent blocks + banners make
10k soldiers legible — and every mechanic underneath is an index query.

## How Total War actually does it (researched, with sources)

RTW/M2TW-era numbers (best documented, community-mined from moddable files);
Warhammer for the physics-era mechanics. Marked [M] where single-source.

- **Formation grid**: a unit is a rank/file grid of slots; per-unit spacing in
  metres — close ~**1.2×1.2 m**, loose exactly double (M2TW
  `export_descr_unit.txt`: `formation 1.2, 1.2, 2.4, 2.4, 8, square`), soldier
  collision radius ~**0.4 m**, default 8 ranks. Sizes 40–240 by unit-size
  multiplier. Soldiers don't path independently: slot targets recompute from
  (unit centre, facing, grid offset) and each man steers to his slot — wheeling's
  outer-file sweep falls out of rigid slots on a rotating frame for free.
- **Army layout is declarative**: official CA "Group Formations" docs — an army
  is a tree of blocks (anchor + relative-offset blocks, `InterEntitySpacing`,
  Line|Column arrangement, per-class priority lists). Our battle-plan generator
  can be exactly this.
- **Melee**: each soldier acquires the **nearest eligible enemy** and fights
  1-v-1 (animation-gated, ~1 swing/s). Combat factor = attack + bonuses −
  (defense skill + armour + shield), clamped ±20; **P(kill) ≈ 1.9% × 1.2^factor**.
  **Flank attack +5, rear +7**; shield only counts vs front/left; being
  mid-animation = can't defend (why surrounds are lethal). Back ranks refill
  slots (spear wall: first 2 ranks fight). [M for exact values]
- **Charges**: arm after a clear straight run-up (~30–35 m); bonus adds to
  attack+damage then **decays linearly over ~13 s** (hence cavalry cycle-charging).
  Impact physics = mass × speed knockdown; **bracing** (standing, facing the
  charge) multiplies effective mass; **spears braced + frontal = charge bonus
  nullified — but side/rear charges bypass bracing entirely**: flanking beats
  spear walls by geometry, not stats.
- **Morale** (RTW/M2TW community table [M]): base + modifiers — general nearby
  **+1/command star within ~50 m**, both flanks protected +4, outnumbering 3:1
  +4, enemy routing nearby up to +8; casualties 10/50/80% → −2/−8/−12, losing
  melee up to −8 (−14 vs cavalry), **friendly routers nearby up to −12** (chain
  routs), flanked/rear strong negative, general dead army-wide negative. States:
  steady → wavering → **routing** (uncontrollable, flees, may rally) →
  **shattered** (never returns). Routers don't fight back — pursuers get free
  kills; routers get a speed boost so only cavalry catches them.
- **Missiles**: every projectile physically simulated (dev-confirmed for WH2);
  no friend/foe check — a miss lands where ballistics says (that IS the
  friendly-fire model). Bows arc over friendlies; crossbows flat (need
  enfilade). Ammo per soldier (~20–30). Doctrine: archers at the line's edges
  firing diagonally = maximize enemies / minimize friends in the arc.
- **Engine**: "battle logic generates the 'future' while display renders the
  'now'" (CA lead engine programmer) — decoupled async sim, heavy multicore;
  per-soldier slot steering under unit-level pathing; sprite impostors at
  distance. Bannerlord's 2048-agent cap is the cautionary contrast: full-fidelity
  per-agent everything doesn't scale — LOD the sim.

## Every mechanic = a query (the roster table)

| Mechanic | Library primitive |
| --- | --- |
| Contact pairing | **k-NN(1–3)** among enemies, per engaged soldier (siege pattern) |
| Flank/rear detection (+5/+7, shield facing) | **sector classify vs facing** — cull contacts, dot-product into front/left/right/rear; or a **`Polyhedron3` wedge** behind the unit. Polyhedron3's first non-frustum starring role |
| Charge corridor ("enemies in my run-up?") | **`Polyhedron3` prism** along the charge vector, or **`Tree3::raycast`** (capsule) attacker→target — clear = charge arms |
| General's aura (+1/star, 50 m) | **sphere cull** — literally the aura case |
| Chain routs (±morale by radius) | sphere cull over units with `state == routing`, both signs |
| Outnumbered / casualties counts | **counting culls** around the melee |
| Archer arc targeting | wedge cull counting enemies-vs-friends in the firing arc; landing = point query / `VoxelRaster` cell |
| Pursuit / run-down | k-NN restricted to routing enemies; contact radius = free kill |
| Regiment maneuver level | ~60 centroids → **honest brute force** (CHOOSING.md: at that N the index loses — say so in the docs; the soldier level is the index showcase) |
| Formation slots | **no query** — kinematic grid targets (the cheap contrast to boids) |
| Per-frame index | keep (`sync_index` pattern) at the soldier level |

## Making it molón

1. **Banners** — one standard per regiment (KayKit banner models, faction tint
   in-shader, chevrons by experience): the block IS the readable atom. State
   signalling: banner flashes when wavering, drops/white when routing (the TW
   visual language, high-water mark Shogun 2 / M2TW).
2. **Formation motion tells**: wheeling sweeps, charge posture (speed-up +
   lances), the knockdown ripple on cavalry impact, rout scatter (loose chaos vs
   tight blocks reads instantly at distance).
3. **S — anim phase/tint jitter** per soldier (same as horde item 1–2).
4. **S — trauma camera shake** on cavalry impacts; **slow-mo** on a charge
   connecting or an army breaking.
5. **M — blood splat ring buffer → terrain RT** (256 entries) under melees.
6. **M — night/dusk variant + torches** (clustered forward in wgpu) — the most
   "trailer" visual for battle lines.
7. **Corpse persistence** (frozen-pose static buffer) — battlefields accumulate
   history; where the lines stood is written on the ground.
8. **Dust** behind cavalry from the density grid.
9. Kill-cam moment: pick the k-NN pair on a decisive kill, zoom the free camera.

## Assets (all CC0, verified 2026-07-02)

- **Soldiers: KayKit Adventurers** (Knight sword+shield, Ranger bow, crossbow
  loadouts; weapons are separate attachable objects) **+ KayKit Character
  Animations** — **161 humanoid clips** (1H/2H/block melee, bow/crossbow aim +
  reload, hit/death/spawn) on the shared KayKit rig, glTF, CC0.
  <https://kaylousberg.itch.io/kaykit-adventurers> ·
  <https://kaylousberg.itch.io/kaykit-character-animations>
- **Spear/pike gap**: no free rigged spearman exists in these lines — attach a
  CC0 spear mesh (poly.pizza) to the KayKit knight and drive it with the 2H
  clips.
- **Banners: KayKit Dungeon Remastered**; pole-mounted field standards & siege
  engines: **Kenney Castle Kit** (CC0 glTF, static — animate procedurally like
  our cannon). <https://kenney.nl/assets/castle-kit>
- We already have: knights/horses (cavalry), cannon (artillery), castle.
- Same caveat: verify clip naming vs `load_glb_clip` on import; KayKit rig ≠
  Quaternius rig (don't mix clips across families).

## Build plan (phases, each committable)

1. **Sim skeleton** (`formations_sim.rs`): regiment struct (grid, spacing,
   facing, state machine), slot steering, army deployment generator (the CA
   block-tree model), two-level decide→apply (regiment brains serial ~60/side;
   soldier pass parallel), keep-index at soldier level.
2. **Melee + charges**: contact pairing k-NN, combat-factor roll (P(kill) =
   1.9%×1.2^cf, flank/rear sector bonuses), charge arm (raycast corridor) +
   13 s decay + mass knockdown + bracing/spear rule.
3. **Morale + routs**: the modifier table, aura culls, chain-rout culls,
   wavering/rout/shatter states, pursuit free-kills, rally.
4. **Missiles**: volley scheduler, ballistic arcs (existing projectile system),
   no-friend-foe landing (point query), ammo, arc-occupancy targeting.
5. **Renderer macroquad**: blocks + banners + state flags, HUD (regiment count,
   morale states, sliders); then **wgpu parity** + night/torches + blood RT.
6. **Web + release + docs** (FORMATIONS.md roster table, PARALLEL.md two-level
   numbers, decision map).

Scale target: 5–10k soldiers/side native (TW-realistic and render-comfortable at
our measured 20k), ~2–4k wasm.

## Order vs horde

User: "las dos". Suggested: **horde first** (closer to the existing chassis,
capitalises the keep-index result immediately), formations second (bigger new
layer: slots/morale/two-level). Shared prerequisite: extract the generic sim
scaffolding (Rng, Fx, projectiles, terrain access, sync_index pattern) from
`siege_sim` into a shared module both new sims use.
