# Siege — asset inventory

Everything currently in `assets/siege/` (as downloaded `.glb` kits). All models
are **Quaternius**, **CC0** except the **Witch** (CC-BY 3.0). See `CREDITS.md`.
All `.glb` are self-contained (textures embedded) → easy to load/embed.

## Kits (counts)

| Kit (zip) | models | role |
| --- | --- | --- |
| Pirate kit | 71 | **the core**: pirate + undead characters, sea creatures, cannon, ships, docks, palms, rocks, houses, weapons, props |
| Ultimate Fantasy RTS | 107 | structures: castles/fortresses/walls/towers/gates, houses, farms, markets, temples, mountains, pines, rocks |
| Stylized Nature MegaKit | 68 | nature: trees/pines/dead trees, bushes, grass, flowers, mushrooms, pebbles, **rock paths** (→ bridges) |
| Medieval Village Pack | 39 | village: blacksmith, inn, stable, barracks, bell tower, mill, well, bonfire, cart, fences, props |
| Ultimate Platformer Pack | 21 | **projectiles/hazards**: Spiky Ball, Bomb, Cannon, Saw Blade, Spike Trap, Small Bridge, Tower |
| Animated Fish Bundle #1 | 52 | sea ambience: 50+ fish + boat + docks |
| Animated Fish Bundle #2 | 7 | sea: fish, dolphin, shark, **whale**, manta ray |
| Animated Dinosaur Bundle | 6 | T-Rex, Velociraptor, Triceratops, Stegosaurus, Apatosaurus, Parasaurolophus |
| Animated Enemies | 5 | Snake, Wasp, Rat, Spider, Frog |
| Farm Animal Pack | 7 | Horse (→ cavalry), Cow, Sheep, Pig, Llama, Zebra, Pug |
| Monsters list | 4 | Slime, Skeleton, Dragon, Bat |
| loose | — | `Witch.glb` (CC-BY), `Zombie.glb`, `Tentacle.glb` |

## Proposed unit roster (model → library query)

| Role (query) | 🏴‍☠️ Pirates (living) | 🧟 Undead |
| --- | --- | --- |
| Soldier — melee k-NN | **Anne** (axe) / **Mako** | **Skeleton** (axe) |
| Knight/leader — k-NN, fast (+Horse = cavalry) | **Pirate Captain** | **Skeleton** (sword) |
| Archer — raycast first-hit | **Sharky** (rifle/pistol) | **Bat** (flying harasser) |
| Artillery — raycast pierce / sphere-cull AoE | **Cannon** | **Cannon** |
| Mage — chain k-NN | **Witch** ⚠️CC-BY | **Slime** (acid lob) |
| Healer/bard — friendly k-NN | **Henry** (lute/banjo) | **Zombie** (tank, no heal) |
| Flying hero — sphere-cull AoE | **Dragon** | **Dragon** (shared) |
| Boss (optional) | **T-Rex** | **Whale** (sea boss) |
| Water hazard (neutral) | **Shark**, **Tentacle** roam water, hit anyone near | |
| Ambient (non-combat) | fish (×52), Cow/Sheep/Pig…, Bird | |

## Catalog by use

### Characters / creatures (animated)
- **Pirates:** Pirate Captain, Anne, Henry, Mako, Sharky
- **Undead:** Skeleton (×2: `Skeleton.glb`, `Skeleton-yq5ATpujSt.glb`), Zombie, Slime, Bat, Skeleton (monsters kit)
- **Caster:** Witch ⚠️CC-BY
- **Dragon** (Pirate-adjacent + monsters kit)
- **Sea:** Shark, Tentacle, + Fish Bundle #2 (dolphin, whale, manta ray, shark) + Fish Bundle #1 (52 fish)
- **Beasts (enemies):** Snake, Wasp, Rat, Spider, Frog
- **Dinos:** T-Rex, Velociraptor, Triceratops, Stegosaurus, Apatosaurus, Parasaurolophus
- **Farm:** Horse, Cow, Sheep, Pig, Llama, Zebra, Pug

### Structures (static) — bases & buildings
- **Castles/defence (RTS):** Castle, Castle Fortress, Castle Gate, Fortress, Wooden Fortress (+Gate), Stone Wall (+Towers), Wooden Wall, Wall Towers, Watch Tower, Small Watch Tower, Stone Tower, Tower House, Wooden Encampment, Wooden Monument
- **Town (RTS):** Town Center, Barracks, Market Stalls, Business Building, Mine, Windmill, Temple (×4), Wooden Temple, Farm (×3), Small Farm, Crops, Storage (House/Hut/Shed), House/Houses/Hut/Shack (many), Wooden house tower
- **Village pack:** Blacksmith, Fantasy Inn, Fantasy Stable, Fantasy Barracks, Fantasy House (×3), Bell Tower, Mill, Fantasy Sawmill, Gazebo
- **Ports (Pirate + RTS):** Ship, Small Ship, Dock, Dock Broken, Port, Shipping Port, Docks

### Nature (static) — forests, obstacles, paths→bridges
- **Trees:** Tree (×5 variants), Pine (×5), Twisted Tree (×5), Dead Tree (×4), Trees/Pine Trees/Trees cut (RTS), **Palm Tree (×3)** (Pirate)
- **Ground cover:** Bush, Bush with Flowers, Fern, Grass (×3), Tall Grass, Clover, Flowers (many), Plant (×3), Mushroom (×2)
- **Rocks/obstacles:** Rock (Pirate ×6, RTS, Nature Medium ×3), Rocks (Pirate ×5, RTS), Pebble Round (×5), Pebble Square (×6), Mountain(s) (RTS), Gold rocks, Logs
- **Paths → bridges/roads:** Rock Path Round (Small/Thin/Wide), Rock Path Square (Small/Thin/Wide), **Small Bridge** (Platformer), Path Straight (Village)

### Props / projectiles / items
- **Projectiles:** **Spiky Ball**, **Bomb** (Platformer + Pirate) → real flying cannon shots (travel time)
- **Hazards/traps:** Saw Blade, Spikes, Hazard Spike Trap, Cylinder Hazard
- **Cannon** (Pirate + Platformer)
- **Decor:** Barrel, Crate/Cube Crate, Chest (Closed/Gold), Coins, Gems (×3), Gold (Bag/ore/rocks), Anchor, Bucket (+of Fish), Prop Bottle (×2), Skull (×2), Bird, Bonfire, Cart, Bench, Well, Cauldron, Hay, Bags, Market Stand, Fence, Stairs, Doors, Windows, Wheat, Wood, Paper, Large Bone, Post, Smoke, Cloud, Fruit

### Weapons (attachable to characters, or as pickups)
Axe, Axe Rifle, Cutlass, Dagger, Sword(s), Pistol, Rifle, Shotgun, Lute (→ Henry the bard)

## Notes / cleanup
- **`Fish.zip` is `Fish.fbx`** (FBX, not GLB) → can't load as-is. We already have
  plenty of sea creatures; skip it, or re-download the fish-man as `.glb`.
- **`quitar o repes/`** holds loose `.glb` (Agile knight, Anne, Dragon, Henry,
  Shark, Skeleton ×2) that are **duplicates already inside the Pirate Kit** (or
  dropped, like the Agile knight) → can be deleted.
- Files with a `-XXXXXXXX` suffix (e.g. `House-2kytqGs4rH.glb`,
  `Twisted Tree-8oraKn9m0x.glb`) are **style variants** of the same model.
- Next step: extract the ~15-20 chosen models into a clean layout and wire the
  glTF loader (static first). Nothing is committed yet — these are raw downloads.
