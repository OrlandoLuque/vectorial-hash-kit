# HORDE — design doc (They Are Billions-style zombie assault, automatic)

Research notes + build plan for the next flagship demo. Status: **researched
2026-07-02, not started**. Like siege: self-playing (no player), every mechanic a
spatial query, shared graphics-free sim + macroquad and wgpu renderers, native +
wasm. Companion: [FORMATIONS_DESIGN.md](FORMATIONS_DESIGN.md). Decisions taken
with the user: build **after/alongside formations** ("las dos"), **switchable
index structures** (Tree3 / MortonGrid3 / 2D-projection, like critters3d's `M`),
lives in `vectorial-hash-demos`.

## Concept

A fortified colony (walls, gates, towers, houses, Command Center) at the centre;
**tens of thousands of zombies** — most dormant, scattered across the map — plus
periodic **waves** from the edges with a direction warning and countdown. Towers
auto-fire (noise!), noise wakes sleepers in expanding cascades, walls get gnawed,
breaches flood, one zombie reaching a populated house triggers the **infection
cascade**. Waves escalate to a final all-edges assault that also wakes every
dormant zombie left. Colony falls or survives N waves → reset with a bigger map
seed. The performance headline: **~100k indexed agents where ~90% sleep — the
keep-index (`update_ref`) maintenance costs ~zero for sleepers; you pay only the
active front.** (Direct sequel to the 2026-07-02 rebuild-vs-keep result.)

## How They Are Billions actually does it (researched, with sources)

Community reverse-engineering; the noise model is the load-bearing mechanic.

- **Dormant zombies don't wander.** They stand still until woken by noise or
  sight, then walk to the *noise tile* and investigate. Wave zombies path to the
  Command Center. ([Steam threads](https://steamcommunity.com/app/644930/discussions/0/1620600279665778638/))
- **Noise = per-tile decaying scalar ("activity").** Events add activity to a tile
  (+ less to neighbours); it **halves every second**. Values: ranger shot 1,
  soldier 3, ballista 5, sniper 10, shocking tower 20, **Thanatos rocket 500**,
  **each colonist infected 50**. ([Fandom: Noise](https://they-are-billions.fandom.com/wiki/Noise))
- **Wake rule** (player-mined, 2 independent threads): a zombie hears activity
  within **4 × its watch range**; wakes when accumulated activity ≥ **1000 /
  alertness** (walkers alertness 2, runners 3, venoms 4, harpies 8 — harpies wake
  at 125, walkers at 500). Map-wide multiplier per difficulty (×0.5–×1.5). The
  chain reaction is **mediated by the noise field**: kills make noise → more come
  → more noise. ([thread](https://steamcommunity.com/app/644930/discussions/0/1697174779845532460/))
- **Waves**: ~10 per run, warning "swarm from the North" + countdown (8 h; final
  24 h) — but they path to the CC by **weighted shortest path where walls are
  high cost, not impassable**, so the swarm bends to the weakest side. The final
  wave spawns from **all edges** and picks up all remaining dormants.
  ([Fandom: Swarms](https://they-are-billions.fandom.com/wiki/Swarms))
- **Bestiary** (HP / dmg / speed / watch / noise-made): Walker 35/6/0.4/5/1 ·
  Runner 45/9/1.75/6/2 · **Harpy** 120/30/**5**/9/10 (jumps walls) · **Venom**
  120/30 AoE/1.75/8/10 (**range 4.5** — outranges walls) · **Chubby** 500/40
  (armor 15%, shields the pack) · **Giant** 4–20k HP/160 AoE (deaf, minimap dot).
  ([theyarebillionswiki.com/infected](https://theyarebillionswiki.com/infected))
- **Structures**: wood wall 400 HP, stone 1000, gates 600/1500, wood tower 800,
  stone 2000, CC 5000. Zombies attack the **nearest thing in reach** once
  engaged (they'll dumbly pound a wall next to a hole — pathing chose the hole at
  plan time, melee target is nearest). ([buildings](https://theyarebillionswiki.com/buildings))
- **Towers**: Great Ballista range 9 dmg 150 noise 5 · Shocking Tower ring r6
  dmg 60 noise 20 · Executor range ~12 dmg 100 AoE 0.6 noise 20. Target mode is
  **configurable: "nearest" or "highest threat"**. Noise-per-kill is the real
  defense currency (quiet defense doesn't grow the assault).
- **The infection cascade** (the signature): a zombie destroying a populated
  building instantly converts every colonist inside into runners **and emits 50
  noise each** — a 20-person house = 1000 activity = wakes a huge radius →
  exponential collapse. Most visually legible failure state in the game.
- **Engine**: custom C#, per-agent AI, 20–30k on map; "AI, pathfinding and game
  logic perfectly scalable across 12 cores; GPU drawing is the serial part" —
  literally our decide→apply story. ([GamesBeat](https://gamesbeat.com/the-making-of-early-access-hit-they-are-billions/))

## Every mechanic = a query (the demo's roster table)

| Mechanic | Library primitive |
| --- | --- |
| Noise wakes sleepers | noise events → **sphere/Circle cull** radius `4×watch×mod` over the dormant set; per-class radius + threshold `1000/alertness` |
| Infection burst | building death → one **large-radius cull** that visibly wakes a region (50·N activity) |
| Tower "nearest" | **k-NN(1)** within range, per tower per tick — the *static consultant* case (inverse of siege) |
| Tower "highest threat" | radius **cull + score-max** filter |
| Shocking ring | disc cull AoE |
| Venom standoff | asymmetric radii duel (their 4.5 > wall reach) |
| Zombie melee target | nearest structure/unit in reach = k-NN vs the **static index** |
| Harpy | skips the wall cost layer entirely (flies the path graph) |
| Swarm routing | coarse **flow field** to the CC, walls = high cost (grid technique, feeds the index story: breach = local field rebuild) |
| Horde separation | boids separation (`SepTables`, exists) |
| Static world | **`bulk_load_par` once** — the from-scratch static build case |
| Dormant horde | **keep-index**: sleepers never relocate → `update_ref` O(1) no-ops; only the active front pays |
| Senses at scale | `cull_many_par` batch reads |
| Structure A/B | `M`-style toggle: **Tree3 / MortonGrid3 / 2D projection** — dense uniform horde is Morton's home turf (CHOOSING.md) |

## Scenarios & unit logic (approved by the user, 2026-07-02)

**Design centre:** the waves are the *performance* headline and the climax; the
**noise/wake system is the mechanical star** — a feedback loop made entirely of
visible spatial queries (tower fires → noise → wake cull → walkers investigate →
attack noise → more wake), with the emergent irony that **the better you defend,
the bigger the horde you attract**. Waves without it would just be "siege with
more bodies"; noise is what gives units logic.

Minimal state machines:

| Unit | States / logic | Queries |
| --- | --- | --- |
| **Zombie** | `Dormant → Investigating (walk to noise tile) → Chasing (sight) → Attacking`. Classes: walker / runner / harpy (jumps walls) / venom (range 4.5 outranges walls) / brute (pack shield) | wake = sphere cull (per-class radius+threshold); melee target = k-NN vs static index |
| **Tower** | `Scanning → Firing (emits noise) → Reloading`; **nearest** vs **highest-threat** modes, toggleable live | k-NN(1) / cull+score-max from a fixed point |
| **Mobile defender** | Ranger **silent** (noise 1, kites) · soldier (noise 3) · sniper (noise 10, kills specials); patrol the wall, **fall back** when local density exceeds a threshold | density cull around self |
| **Repair crew** | after each wave, go to the most-damaged wall if no zombies near | safety cull + min-HP scan |
| **Colonist** | populates buildings; idle ones **haul materials** (storehouse → repair crew — the ant lines that pace all repair); **flees to the centre** when a breach is near; building death → infection burst | threat cull; haul job = k-NN nearest stocked storehouse/crew |

### Defender AI — three layers (TAB's brain is the player; ours is an AI)

**Layer 1 — the Commander** (the "player", ~1 Hz, cheap):
- **Sector threat map**: wall split into ~12–16 sectors; counting cull of active
  zombies near each + **anticipation from the wave warning** (pre-position
  before the wave lands — what makes it look smart). Honest note: at ~16
  sectors this layer is brute force; the index shines at the individual layer.
- **Assignment with hysteresis**: defenders ∝ threat; quiet sectors donate
  troops to hot ones (visible troop movements = free molón).
- **Noise discipline** (one rule, emergent restraint): no wave committed →
  only rangers engage (silent clearing); wall-adjacent density > threshold →
  **weapons free** (snipers/cannons — the horde is already coming).
- **Fall-back order**: outer-ring segment destroyed + density inside → abandon
  sector, man the second ring (TAB's concentric meta).
- Between waves: schedules the **expeditions** (scenario 4) — nearest dormant
  nest, ranger squad, abort on wake.

**Layer 2 — individual FSM**: `Post → Engage → FallBack → Rally`, flavoured by
class — ranger **kites** (distance band [min,max] via k-NN, fires retreating,
noise 1); soldier holds (higher fall-back threshold); sniper targets
**specials first** (class-filtered cull: venom > harpy > brute), not nearest.
Fall-back trigger: density cull around self, or low HP.

**Layer 3 — repair & rebuild** (the minimal economy — yes to both):
- **Limited crews (N)** + job queue: damaged segments sorted by
  `damage × sector-quietness`; claim only if the **safety cull** is clear (no
  active zombie within R); travel, repair at X HP/s, **flee** on threat
  (re-check with hysteresis).
- **Rebuild destroyed segments**: same crews, slower — using the researched TAB
  detail that **construction-in-progress already has partial HP** (progress =
  HP), so a half-rebuilt wall already slows the flood and its flow-field cost
  rises *gradually* as it grows.
- **Pacing without an economy — hauled, not abstract (confirmed)**: repair
  throughput is capped by **colonists physically carrying materials** from the
  storehouse to the crews — a crew only works while stocked, so the pacing
  counter *is* the ant line you see trotting across the base between waves.
  Colonist FSM gains a `Hauling` state (idle colonists claim delivery jobs:
  nearest stocked storehouse → crew, k-NN); haul routes near a hot sector are
  dangerous — a caught porter is one more infection burst, so the Commander's
  safety cull also gates hauling. Waves still leave scars (repair is bounded by
  porter round-trips), and the whole economy stays: jobs + walking, no numbers.

Scenarios (same sim; only map, base layout and wave script change — rotate
between runs or select):

1. **Classic survival** (default): central base, dormant field, ~8 escalating
   waves with direction warning + countdown, final all-edges wave that also
   wakes everything left. Base falls or survives → reset, new seed.
2. **The mountain pass**: base in a valley chokepoint, waves funnel through 2
   entrances — flow field + density at their most spectacular.
3. **The river crossing**: reuse siege's rivers + bridges; bridges are
   chokepoints, venoms fire from the far bank.
4. **The expedition** (second act between waves): a squad ventures out to
   **clear dormant nests** — the noise budget seen from the other side: the
   ranger clears silently; one sniper shot wakes the whole nest and the squad
   runs home with the horde on its heels.

## Making it molón (from the crowd-tech research)

Ranked shopping list (effort S/M/L → payoff), full sources in the research
summary below:

1. **S — Anim phase/speed/scale jitter per instance** (hash the id). The single
   biggest wow-per-line: without it a horde reads as a tiling texture.
2. **S — Tint jitter** (HSV ±15% over the existing color mix).
3. **S/M — Corpses that stay**: on death `remove_ref` + push the final transform
   with a frozen death-pose frame into a **static instance buffer** (zero
   per-frame cost, one extra draw). Cap ~30k, demote oldest to ground decals. An
   aftermath field of corpses is the cheapest "massive" signal there is.
4. **M — Sleeping + bucketed brains**: dormants out of the rayon loop entirely
   (event wake via the index); active zombies decide at 4–8 Hz (`id % N`
   buckets), move every frame. The only way to 100k on CPU.
5. **M — Single-goal flow field**: 1–2 m cells, ~50×50-cell tiles, Dijkstra/
   Eikonal wavefront + line-of-sight pass; ~0.1 µs/cell (a 256² field ≈ 8 ms
   single-thread → time-slice or rayon). Replaces per-agent pathfinding for
   "everyone converges on the base"; walls = high cost so breaches redirect the
   flood. (SupCom2's Game AI Pro ch. 23 is the recipe; Planet Coaster ran 10k
   guests on per-goal fields.)
6. **M — Mesh LOD tiers**: the wall is vertex count, not instance count (100k ×
   1.5k verts is dead; 100k × ~150 avg is fine). Switch at ~20/60/100 char-radii;
   decimate the Quaternius models to ~400/~120 tri offline. Beyond last LOD: an
   8-view billboard atlas (~192 KB/type) — also the only macroquad path to 100k.
7. **M — Blood/scorch splat ring buffer → terrain render target** (256 entries,
   Let Them Come recipe) — shares machinery with our crater remesh.
8. **S/M — Dust billboards** from sim density (cells with high count × speed).
9. **M — Night assaults + torches**: wgpu = clustered forward (hundreds of point
   lights); macroquad = baked low-res terrain lightmap + ~4 nearest lights.
   Waves at night = maximum dread.
10. **S — Trauma camera shake** (trauma², Perlin on yaw/pitch/roll) on breaches +
    **slow-mo** on the CC falling.

Wave staging for drama: direction warning banner + countdown (HUD), the wall of
dots pouring from the announced edge then **bending toward the weakest wall**,
mixed speeds making the front boil (walkers 0.4 / runners 1.75 / harpies 5).

## Assets (rule for every demo: reuse what's downloaded first; new packs welcome)

**Already local** (`crates/vectorial-hash-demos/assets/siege/models/`, 16 glb,
rigged where animated): zombie, skeleton_a, skeleton_sword, slime, bat, mako,
tentacle (the horde + specials core is basically covered), anne, sharky,
pirate_captain, henry, witch (defenders: ranger/soldier/sniper reskins), dragon,
cannon (tower armament), castle, horse. **The demo can start with zero
downloads** — the horde (zombie+skeletons+slime as chubby, bat as harpy-class
flyer) and the defenders are already here.

**To download when the phase needs them (all CC0, verified 2026-07-02):**
- **Walls/gates/towers/houses: Quaternius Ultimate Fantasy RTS** — native glTF,
  same author/palette as our set. Verified pieces: Stone/Wooden Wall, Wall
  Towers, Watch/Archery Towers, Fortress Gate, Castle Gate, Barracks,
  houses/huts, Town Center. <https://quaternius.com/packs/ultimatefantasyrts.html>
- **Brutes/runners variety: Quaternius Ultimate Monsters** — rigged+animated
  glTF (Orc/Yeti/Demon = chubby-class, Ninja = runner).
  <https://quaternius.com/packs/ultimatemonsters.html>
- **Extra undead skins: KayKit Skeletons** (free tier CC0, rigged glTF).
  <https://kaylousberg.itch.io/kaykit-skeletons>
- **Barricades/spikes/torches/banners: KayKit Dungeon Remastered** (CC0).
  <https://kaylousberg.itch.io/kaykit-dungeon-remastered>
- Integration caveat: check animation clip names against `load_glb_clip`'s
  walk/attack/idle expectations per pack; KayKit rig ≠ Quaternius rig.

## Build plan (phases, each committable)

1. **Sim skeleton** (`horde_sim.rs`): world gen (reuse heightfield, flatter map),
   static base layout (walls ring + towers + houses + CC), `bulk_load_par` static
   index; zombie spawn (dormant field + stats table above); decide→apply with
   keep-index sync; **noise field grid** (decay ×0.5/s) + wake culls.
2. **Waves + combat**: wave scheduler (direction, countdown, escalation,
   composition), tower targeting (k-NN nearest / threat toggle), wall HP +
   breach, infection cascade, flow-field routing with wall costs.
3. **Renderer macroquad**: instanced/phase-group zombies + static buildings,
   corpse buffer, HUD (wave countdown, populations, structure toggle `M`,
   pop/thread sliders).
4. **Scale pass**: sleeping buckets, LOD tiers, anim/tint jitter, dust; measure
   the keep-vs-active cost curve (the headline chart for PARALLEL.md).
5. **wgpu parity + night**: GPU-skinned actives, clustered torch lighting, blood
   RT, camera trauma.
6. **Web + release**: wasm build (smaller pop), Pages page, binaries release,
   HORDE.md (the roster-table doc), decision-map entries.

Population targets: native 50–100k total (10–20k active), wasm ~10–20k total.
