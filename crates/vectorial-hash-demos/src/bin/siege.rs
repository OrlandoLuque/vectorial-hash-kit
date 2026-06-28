//! `siege` — a medieval-battlefield showcase for vectorial-hash in 3D.
//!
//! Two castles at opposite corners of a procedurally-generated battlefield
//! (rolling hills + a central volcano) sally their armies; the two factions
//! advance, clash in the middle, and respawn from their keep when they fall —
//! a continuous battle that never empties. The point is not the graphics (rough
//! by design — billboarded instanced spheres) but the **workload**: every unit,
//! every frame, runs read-only spatial queries on a single shared `Tree3`:
//!
//! - **targeting** — `knn` to find the nearest *enemy* (filter the k-NN result
//!   by faction);
//! - **area attacks** — the dragon's fire-breath is a `Sphere3` `cull` (one
//!   query, every enemy caught takes damage);
//! - (next layers: archer line-of-fire via `raycast`, smoke that blocks it,
//!   boids formations, and the parallel per-unit AI pass.)
//!
//! Combat uses the **parallel-safe split**: a *decide* pass reads the index and
//! writes each unit's intent into *its own* fields only (no cross-unit writes,
//! so it parallelises with `par_iter_mut`); a serial *apply* pass then resolves
//! damage, deaths and respawns. See `docs/PARALLEL.md` § "Per-unit AI".
//!
//! Run: `cargo run -p vectorial-hash-demos --bin siege --release`
//!  - drag left mouse: orbit the camera; scroll: zoom
//!  - `[` / `]`: smaller / larger armies (rebuild)
//!  - `P`: pause / resume the simulation

use macroquad::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use vectorial_hash::{Aabb, Point3, Positioned3, Sphere3, Tree3};
use vectorial_hash_demos::instanced3d::{EffectInstance, InstancedRenderer, ModelGpu};
use vectorial_hash_demos::model::{load_glb, load_glb_clip};

// ---------------------------------------------------------------- world config

/// Per-run map seed — offsets the terrain noise so each run is a different map.
/// Set once at startup (from the wall clock, or `$SIEGE_SEED`); 0 in tests.
static MAP_SEED: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
fn map_seed() -> f64 { *MAP_SEED.get_or_init(|| 0.0) }

const WORLD: f64 = 800.0; // battlefield is WORLD × WORLD in the ground plane
const SKY: f64 = 260.0; // index height — heights reach ~150, the dragon flies
const PER_FACTION: usize = 500; // units each side spawns with (tunable live)
const ATK_ANIM_LEN: f32 = 0.45; // attack-clip / lunge play window (seconds)
const LAVA_DPS: f64 = 45.0; // damage per second to a ground unit standing in lava
const WATER_LEVEL: f64 = 6.0; // terrain below this is water (units wade slowly)
const ANIM_FRAMES: usize = 12; // baked movement-clip frames per model (smoothness)
const ATTACK_FRAMES: usize = 6; // baked attack-clip frames (one-shot, fewer = fewer draws)
const ANIM_GROUPS: usize = 5; // units share a frame within a phase group (caps draw calls)
// Clip name preferences (priority-ordered substrings) — movement, attack, idle.
const MOVE_PREFS: &[&str] = &["walk", "run", "flying", "fly", "move"];
const ATTACK_PREFS: &[&str] = &["attack", "sword", "slash", "cast", "shoot", "punch", "bite", "kick"];
const IDLE_PREFS: &[&str] = &["idle"];

/// Per-(faction,kind) baked clips: movement (or idle for riders) + attack.
struct UnitAnim { mov: Vec<ModelGpu>, atk: Vec<ModelGpu> }

/// Which frame of the one-shot attack clip to show (plays once over the attack).
fn attack_frame(atk_anim: f32, nf: usize) -> usize {
    if nf <= 1 { return 0; }
    let prog = (1.0 - atk_anim / ATK_ANIM_LEN).clamp(0.0, 0.999);
    (prog * nf as f32) as usize
}

// ------------------------------------------------------------------------- rng

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

// ----------------------------------------------------------- procedural terrain

/// Hash an integer lattice point to [0,1) — the value-noise source.
fn hash2(x: i32, z: i32) -> f64 {
    let mut h = (x as u32).wrapping_mul(0x1659_5e3d).wrapping_add((z as u32).wrapping_mul(0x27d4_eb2f));
    h ^= h >> 15; h = h.wrapping_mul(0x85eb_ca6b); h ^= h >> 13;
    (h & 0x00ff_ffff) as f64 / 0x00ff_ffff as f64
}

/// Smoothstep-interpolated value noise in [0,1).
fn vnoise(x: f64, z: f64) -> f64 {
    let (xi, zi) = (x.floor() as i32, z.floor() as i32);
    let (fx, fz) = (x - xi as f64, z - zi as f64);
    let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let a = hash2(xi, zi); let b = hash2(xi + 1, zi);
    let c = hash2(xi, zi + 1); let d = hash2(xi + 1, zi + 1);
    let ab = a + (b - a) * sx; let cd = c + (d - c) * sx;
    ab + (cd - ab) * sz
}

/// The river's centre-line x at a given z (it meanders along Z, seeded).
fn river_center_x(z: f64) -> f64 {
    let o = map_seed();
    WORLD * 0.32 + (z * 0.011 + o).sin() * 95.0 + (z * 0.004 + o).cos() * 45.0
}

/// Z positions of the bridges across the river (away from the volcano band).
const BRIDGE_Z: [f64; 4] = [120.0, 260.0, 540.0, 680.0];
const BRIDGE_HALF_W: f64 = 48.0; // spans the river channel
const BRIDGE_HALF_D: f64 = 12.0; // deck depth along Z

/// Is (x,z) on a bridge deck? (so the unit crosses dry instead of wading.)
fn on_bridge(x: f64, z: f64) -> bool {
    BRIDGE_Z.iter().any(|&bz| (z - bz).abs() < BRIDGE_HALF_D && (x - river_center_x(bz)).abs() < BRIDGE_HALF_W)
}

/// How strongly a point sits in the river channel: 1 at the centre line, 0 past
/// the banks, faded to 0 near the volcano so the river doesn't carve the cone.
fn river_factor(x: f64, z: f64) -> f64 {
    let d = (x - river_center_x(z)).abs();
    let w = 44.0;
    if d >= w { return 0.0; }
    let t = 1.0 - d / w; // 0 at bank, 1 at centre
    let dc = ((x - WORLD * 0.5).powi(2) + (z - WORLD * 0.5).powi(2)).sqrt();
    let vol_mask = (dc / 220.0).clamp(0.0, 1.0); // 0 at volcano, 1 far away
    t * t * vol_mask
}

/// Terrain height at a world (x,z): two octaves of hills, a central volcano cone,
/// and a carved river channel. Deterministic — called per terrain tile + unit step.
fn terrain_height(x: f64, z: f64) -> f64 {
    let s = 1.0 / 150.0;
    let o = map_seed(); // per-run noise offset → a different map each run
    let mut h = vnoise(x * s + o, z * s + o * 0.7) * 45.0 + vnoise(x * s * 2.7 + o, z * s * 2.7 + o) * 16.0;
    // Volcano: a cone rising near the centre, with a crater dip at the very top.
    let (cx, cz) = (WORLD * 0.5, WORLD * 0.5);
    let d = ((x - cx) * (x - cx) + (z - cz) * (z - cz)).sqrt();
    if d < 170.0 {
        let cone = (170.0 - d) * 0.62;
        h += cone;
        if d < 28.0 { h -= (28.0 - d) * 1.2; } // crater
    }
    // River: pull the height down toward (and below) the water line in the channel.
    let r = river_factor(x, z);
    if r > 0.0 { h = h * (1.0 - r) - r * 5.0; }
    h
}

/// Surface colour at a point, plus an `emissive` flag (true = lava, drawn
/// full-bright, ignoring terrain shading). Elevation ramp water → sand → grass
/// → rock → snow, with a glowing crater pool and a lava flow down one flank.
fn terrain_surface(x: f64, z: f64, h: f64) -> (Color, bool) {
    let (cx, cz) = (WORLD * 0.5, WORLD * 0.5);
    let (px, pz) = (x - cx, z - cz);
    let d = (px * px + pz * pz).sqrt();
    if d < 30.0 { return (Color::new(1.0, 0.46, 0.10, 1.0), true); } // crater pool
    // Lava river: a narrow azimuth wedge flowing down one slope of the cone
    // (a different flank each run).
    let flow_dir = -2.1 + (map_seed() * 0.9).sin() * 1.8;
    let mut dang = (pz.atan2(px) - flow_dir).abs();
    if dang > std::f64::consts::PI { dang = std::f64::consts::TAU - dang; }
    if d < 165.0 && dang < 0.15 { return (Color::new(0.96, 0.34, 0.07, 1.0), true); }
    if d < 150.0 && h > 70.0 { return (Color::new(0.30, 0.11, 0.08, 1.0), false); } // scorched rock
    let c = if h < 6.0 { Color::new(0.12, 0.28, 0.45, 1.0) } // water
        else if h < 10.0 { Color::new(0.62, 0.56, 0.34, 1.0) } // sand
        else if h < 60.0 { Color::new(0.20, 0.42, 0.18, 1.0) } // grass
        else if h < 95.0 { Color::new(0.38, 0.34, 0.30, 1.0) } // rock
        else { Color::new(0.82, 0.84, 0.88, 1.0) }; // snow
    (c, false)
}

// ----------------------------------------------------------------- unit model

#[derive(Clone, Copy, PartialEq)]
enum Faction { Red, Blue }
impl Faction {
    fn other(self) -> Faction { match self { Faction::Red => Faction::Blue, Faction::Blue => Faction::Red } }
    fn castle(self) -> (f64, f64) { match self { Faction::Red => (90.0, 90.0), Faction::Blue => (WORLD - 90.0, WORLD - 90.0) } }
    fn index(self) -> usize { match self { Faction::Red => 0, Faction::Blue => 1 } }
}

/// The `.glb` model for a (faction, kind): **Red = pirates**, **Blue = undead**.
/// Quaternius CC0 (Witch is CC-BY) — see assets/siege/CREDITS.md. Some models
/// are shared across factions (dragon, cannon) and told apart by the tint.
fn model_for(f: Faction, k: Kind) -> &'static [u8] {
    use Faction::{Blue, Red};
    match (f, k) {
        // Pirates (Red)
        (Red, Kind::Soldier) => include_bytes!("../../assets/siege/models/anne.glb"),
        (Red, Kind::Archer) => include_bytes!("../../assets/siege/models/sharky.glb"),
        (Red, Kind::Knight) => include_bytes!("../../assets/siege/models/pirate_captain.glb"),
        (Red, Kind::Mage) => include_bytes!("../../assets/siege/models/witch.glb"),
        (Red, Kind::Healer) => include_bytes!("../../assets/siege/models/henry.glb"),
        // Undead (Blue)
        (Blue, Kind::Soldier) => include_bytes!("../../assets/siege/models/zombie.glb"),
        (Blue, Kind::Archer) => include_bytes!("../../assets/siege/models/skeleton_a.glb"),
        (Blue, Kind::Knight) => include_bytes!("../../assets/siege/models/skeleton_sword.glb"),
        (Blue, Kind::Mage) => include_bytes!("../../assets/siege/models/slime.glb"),
        (Blue, Kind::Healer) => include_bytes!("../../assets/siege/models/bat.glb"),
        // Shared
        (_, Kind::Dragon) => include_bytes!("../../assets/siege/models/dragon.glb"),
        (_, Kind::Catapult) | (_, Kind::Ballista) => include_bytes!("../../assets/siege/models/cannon.glb"),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind { Soldier, Archer, Knight, Dragon, Catapult, Mage, Ballista, Healer }
impl Kind {
    fn speed(self) -> f64 { match self { Kind::Soldier => 26.0, Kind::Archer => 22.0, Kind::Knight => 52.0, Kind::Dragon => 60.0, Kind::Catapult => 10.0, Kind::Mage => 20.0, Kind::Ballista => 13.0, Kind::Healer => 24.0 } }
    fn max_hp(self) -> f64 { match self { Kind::Soldier => 100.0, Kind::Archer => 60.0, Kind::Knight => 180.0, Kind::Dragon => 1400.0, Kind::Catapult => 160.0, Kind::Mage => 70.0, Kind::Ballista => 140.0, Kind::Healer => 90.0 } }
    /// Engagement range — for the siege engines the firing range, for the healer
    /// the heal range.
    fn reach(self) -> f64 { match self { Kind::Soldier => 9.0, Kind::Archer => 150.0, Kind::Knight => 12.0, Kind::Dragon => 60.0, Kind::Catapult => 260.0, Kind::Mage => 120.0, Kind::Ballista => 240.0, Kind::Healer => 70.0 } }
    /// Damage per strike — for the healer, the (positive) amount healed.
    fn dmg(self) -> f64 { match self { Kind::Soldier => 14.0, Kind::Archer => 18.0, Kind::Knight => 30.0, Kind::Dragon => 22.0, Kind::Catapult => 30.0, Kind::Mage => 16.0, Kind::Ballista => 26.0, Kind::Healer => 24.0 } }
    fn cooldown(self) -> f64 { match self { Kind::Soldier => 0.8, Kind::Archer => 1.1, Kind::Knight => 1.0, Kind::Dragon => 0.5, Kind::Catapult => 2.4, Kind::Mage => 1.3, Kind::Ballista => 1.7, Kind::Healer => 0.9 } }
    fn radius(self) -> f32 { match self { Kind::Soldier => 3.0, Kind::Archer => 3.0, Kind::Knight => 4.2, Kind::Dragon => 11.0, Kind::Catapult => 5.5, Kind::Mage => 3.4, Kind::Ballista => 5.0, Kind::Healer => 3.2 } }
    /// Ground units sit on the terrain; the dragon flies at a fixed altitude
    /// (low enough to menace the ground — it engages by horizontal distance).
    fn altitude(self) -> f64 { match self { Kind::Dragon => 46.0, _ => 0.0 } }

    /// All eight kinds, in `index()` order — the render groups units by this.
    const ALL: [Kind; 8] = [Kind::Soldier, Kind::Archer, Kind::Knight, Kind::Dragon, Kind::Catapult, Kind::Mage, Kind::Ballista, Kind::Healer];
    fn index(self) -> usize { match self { Kind::Soldier => 0, Kind::Archer => 1, Kind::Knight => 2, Kind::Dragon => 3, Kind::Catapult => 4, Kind::Mage => 5, Kind::Ballista => 6, Kind::Healer => 7 } }
    /// Per-model orientation + size corrections — some Quaternius models face a
/// different axis or read over/undersized. Returns (yaw offset in radians, scale
/// multiplier).
    fn model_tweak(self, f: Faction) -> (f32, f32) {
        match (f, self) {
            (Faction::Blue, Kind::Mage) => (-std::f32::consts::FRAC_PI_2, 0.55), // slime: faces +X, reads big
            _ => (0.0, 1.0),
        }
    }

    /// Visual model height in world units (the model is normalised to height 1).
    /// Per-kind because some models read bigger than their collision sphere.
    fn model_height(self) -> f32 {
        match self {
            Kind::Catapult | Kind::Ballista => self.radius() * 1.7, // chunky cannon model
            Kind::Knight => self.radius() * 2.3, // horse + rider; keep it trim
            Kind::Dragon => self.radius() * 2.2,
            _ => self.radius() * 2.6,
        }
    }
}

struct Unit {
    faction: Faction,
    kind: Kind,
    p: Point3,
    hp: f64,
    cooldown: f64,
    respawn_at: f64, // sim-time at which a dead unit returns (f64::INFINITY = alive)
    // Intent written by the *decide* pass (reads only this unit); consumed by the
    // serial *apply* pass. Keeping writes unit-local is what makes decide parallel.
    vel: (f64, f64, f64),
    attacks: Vec<(u32, f64)>, // (target unit id, damage) — many for AoE (dragon)
    emit: Option<Point3>, // strike point that should spawn a smoke puff this frame
    fire: Option<Point3>, // catapult: target point to lob a projectile at this frame
    fx: Vec<Fx>, // visible effects this unit produced this frame
    face: f32, // heading (radians about Y) for orienting the model
    phase: f32, // per-unit animation phase offset (so they don't bob in sync)
    atk_anim: f32, // attack-lunge countdown (seconds), set when the unit strikes
}
impl Unit {
    fn alive(&self) -> bool { self.hp > 0.0 }
}

/// The lightweight item actually stored in the index: id + faction + position.
/// Decoupled from `Unit` so the decide pass can hold `&Tree3<IUnit>` immutably
/// while it mutates the `units` slice through `par_iter_mut`.
#[derive(Clone, Copy)]
struct IUnit { id: u32, faction: Faction, p: Point3, health: f32 }
impl Positioned3 for IUnit { fn position(&self) -> Point3 { self.p } }

// ----------------------------------------------------------------- smoke (LoS)

const SMOKE_R: f64 = 24.0; // puff radius — also the raycast corridor half-width
const SMOKE_LIFE: f64 = 3.5; // seconds before a puff dissipates
const SMOKE_CAP: usize = 240; // hard cap on live puffs

/// A smoke cloud — a dynamic line-of-sight blocker. Catapult and dragon strikes
/// spawn one; it lives in its own `Tree3` so an archer/ballista shot can
/// `raycast` it: a puff between the shooter and the target blocks the shot.
#[derive(Clone, Copy)]
struct Puff { p: Point3, born: f64 }
impl Positioned3 for Puff { fn position(&self) -> Point3 { self.p } }

// ------------------------------------------------------------- visual effects

/// A transient combat effect — the *visible* part of an attack (the queries
/// resolve instantly, so without these the fight is invisible). Spawned with the
/// same parallel-safe pattern as smoke: `decide` pushes into the unit's own `fx`
/// list, the serial `apply` stamps a birth time and moves them to the global
/// pool, the render fades them by age.
#[derive(Clone, Copy)]
enum FxKind { Arrow, Bolt, Lightning, Ring, Spark }
#[derive(Clone, Copy)]
struct Fx { kind: FxKind, a: Vec3, b: Vec3, born: f64 }
impl Fx {
    fn new(kind: FxKind, a: Vec3, b: Vec3) -> Fx { Fx { kind, a, b, born: 0.0 } }
    fn life(kind: FxKind) -> f64 { match kind { FxKind::Arrow | FxKind::Bolt => 0.14, FxKind::Lightning => 0.10, FxKind::Ring => 0.45, FxKind::Spark => 0.30 } }
}
const FX_CAP: usize = 4000; // hard cap on live effects

/// `Point3` → macroquad `Vec3`.
fn v3(p: Point3) -> Vec3 { vec3(p.x as f32, p.y as f32, p.z as f32) }

// ------------------------------------------------------------- projectiles

const PROJ_GRAVITY: f64 = 220.0; // ballistic arc gravity (world units / s²)
const ARTY_STANDOFF: f64 = 95.0; // artillery keeps at least this far from the enemy

#[derive(Clone, Copy, PartialEq)]
enum ProjKind { Cannon, LavaRock }

/// A ballistic projectile (cannonball / lava bomb) — arcs under gravity and on
/// landing does a `Sphere3` AoE. Travel time makes the shot visible.
struct Projectile { p: Point3, v: (f64, f64, f64), kind: ProjKind, faction: Faction, dmg: f64, r: f64 }

/// Launch velocity for a ballistic arc from `a` to `t` over flight time `tf`.
fn arc_velocity(a: Point3, t: Point3, tf: f64) -> (f64, f64, f64) {
    ((t.x - a.x) / tf, (t.y - a.y) / tf + 0.5 * PROJ_GRAVITY * tf, (t.z - a.z) / tf)
}

// ----------------------------------------------------------------- spawning

fn spawn_unit(rng: &mut Rng, faction: Faction) -> Unit {
    // Roster mix: mostly foot soldiers, then archers/knights, a few siege engines
    // and mages, a rare dragon.
    let roll = rng.unit();
    let kind = if roll < 0.44 { Kind::Soldier }
        else if roll < 0.64 { Kind::Archer }
        else if roll < 0.76 { Kind::Knight }
        else if roll < 0.86 { Kind::Mage }
        else if roll < 0.94 { Kind::Healer }
        else if roll < 0.965 { Kind::Ballista } // siege engines are now rare
        else if roll < 0.985 { Kind::Catapult }
        else { Kind::Dragon };
    let mut u = Unit {
        faction, kind,
        p: Point3::new(0.0, 0.0, 0.0),
        hp: kind.max_hp(),
        cooldown: 0.0,
        respawn_at: f64::INFINITY,
        vel: (0.0, 0.0, 0.0),
        attacks: Vec::new(),
        emit: None,
        fire: None,
        fx: Vec::new(),
        face: 0.0,
        phase: rng.range(0.0, std::f64::consts::TAU) as f32,
        atk_anim: 0.0,
    };
    place_at_castle(rng, &mut u);
    u
}

/// Drop a unit at a random point just outside its castle, on the terrain.
fn place_at_castle(rng: &mut Rng, u: &mut Unit) {
    let (cx, cz) = u.faction.castle();
    let x = (cx + rng.range(-60.0, 60.0)).clamp(2.0, WORLD - 2.0);
    let z = (cz + rng.range(-60.0, 60.0)).clamp(2.0, WORLD - 2.0);
    let y = terrain_height(x, z) + u.kind.altitude() + u.kind.radius() as f64;
    u.p = Point3::new(x, y, z);
    u.hp = u.kind.max_hp();
    u.respawn_at = f64::INFINITY;
}

fn spawn_army(rng: &mut Rng, per_faction: usize) -> Vec<Unit> {
    let mut units = Vec::with_capacity(per_faction * 2);
    for _ in 0..per_faction { units.push(spawn_unit(rng, Faction::Red)); }
    for _ in 0..per_faction { units.push(spawn_unit(rng, Faction::Blue)); }
    units
}

// ----------------------------------------------------------------- AI: decide

/// One unit's per-frame brain — *read-only* on the shared index, writes only
/// into `u`'s own `vel`/`attacks`. This is the body that fans out over rayon
/// (`par_iter_mut`): each unit reads the shared index and mutates only itself.
///
/// Three library queries, one per concern: **k-NN** finds the nearest enemy
/// (targeting) *and* the nearby friends (boids); the dragon's AoE is a sphere
/// **`cull`**; the archer's line-of-fire is a thick **`raycast`**.
fn decide(u: &mut Unit, id: u32, index: &Tree3<IUnit>, smoke: &Tree3<Puff>, body_radius: &[[f64; 8]; 2]) {
    u.vel = (0.0, 0.0, 0.0);
    u.attacks.clear();
    u.emit = None;
    u.fire = None;
    u.fx.clear();
    if !u.alive() { return; }

    // One k-NN pass yields both the nearest enemy (targeting) and the nearby
    // friends used for flocking (separation + cohesion). k=16 reliably spans
    // both once the lines meet.
    let mut target: Option<(Point3, u32, f64)> = None; // nearest enemy (pos, id, dist)
    let mut heal: Option<(Point3, u32, f32, f64)> = None; // most-wounded friend (pos, id, health, dist)
    let (mut sep_x, mut sep_z) = (0.0, 0.0); // separation push (from ANY neighbour)
    let (mut coh_x, mut coh_z, mut friends) = (0.0, 0.0, 0u32); // cohesion centroid
    let sep_dist = body_radius[u.faction.index()][u.kind.index()] * 2.0; // two bodies of this size shan't overlap
    for (d, it) in index.knn(u.p, 16) {
        if it.id == id { continue; }
        // Separation from ANY neighbour (friend or foe) inside personal space,
        // weighted by closeness (0 at the edge, strong on contact) so no two
        // bodies overlap. The same one k-NN drives targeting, cohesion AND this.
        if d < sep_dist {
            let (dd, w) = (d.max(1e-3), 1.0 - d / sep_dist);
            sep_x += (u.p.x - it.p.x) / dd * w;
            sep_z += (u.p.z - it.p.z) / dd * w;
        }
        if it.faction != u.faction {
            if target.is_none() { target = Some((it.p, it.id, d)); }
        } else {
            coh_x += it.p.x; coh_z += it.p.z; friends += 1;
            if it.health < 0.97 && heal.is_none_or(|(_, _, h, _)| it.health < h) { heal = Some((it.p, it.id, it.health, d)); }
        }
    }

    // The healer peels off to its most-wounded comrade; with nobody hurt it
    // advances WITH the army (toward the nearest enemy / the enemy keep) rather
    // than drifting to the friend centroid — which made healers clump and jitter
    // instead of moving. Everyone else just seeks the nearest enemy.
    let advance = match target {
        Some((tp, _, d)) => (tp.x, tp.y, tp.z, d),
        None => { let (cx, cz) = u.faction.other().castle(); (cx, u.p.y, cz, f64::INFINITY) }
    };
    let (tx, ty, tz, tdist) = match (u.kind, heal) {
        (Kind::Healer, Some((p, _, _, d))) => (p.x, p.y, p.z, d),
        _ => advance,
    };

    // Velocity = seek (scaled by approach — zero once in reach) + separation
    // (ALWAYS applied, even while stopped and fighting, so bodies never overlap)
    // + gentle cohesion for ground-melee formations, capped so separation can't
    // fling a unit away.
    let seek = (tx - u.p.x, tz - u.p.z);
    let slen = (seek.0 * seek.0 + seek.1 * seek.1).sqrt().max(1e-6);
    let speed = u.kind.speed();
    // The dragon engages by HORIZONTAL distance (`slen`) — it flies far above the
    // ground, so a 3D check would never let it reach ground troops.
    let engage = if u.kind == Kind::Dragon { slen } else { tdist };
    // Artillery kites to keep a standoff (bombard from range, not melee); everyone
    // else closes until in reach.
    let approach = if matches!(u.kind, Kind::Catapult | Kind::Ballista) {
        if engage < ARTY_STANDOFF { -speed } else if engage > u.kind.reach() * 0.9 { speed } else { 0.0 }
    } else if engage < u.kind.reach() * 0.8 { 0.0 } else { speed };
    let (mut vx, mut vz) = (seek.0 / slen * approach, seek.1 / slen * approach);
    // Separation always on, for every kind (dragons included — they were piling
    // up): the personal space now comes from each model's real footprint.
    vx += sep_x * speed * 0.7;
    vz += sep_z * speed * 0.7;
    if matches!(u.kind, Kind::Soldier | Kind::Knight) && friends > 0 {
        let (cx, cz) = (coh_x / friends as f64 - u.p.x, coh_z / friends as f64 - u.p.z);
        let cl = (cx * cx + cz * cz).sqrt().max(1e-6);
        vx += cx / cl * speed * 0.12; vz += cz / cl * speed * 0.12; // cohesion
    }
    let vl = (vx * vx + vz * vz).sqrt();
    let cap = speed * 1.5;
    if vl > cap { let s = cap / vl; vx *= s; vz *= s; }
    u.vel = (vx, 0.0, vz); // the dragon's altitude is pinned in the apply pass

    // Attacking: only when in reach and off cooldown.
    if u.cooldown > 0.0 || engage > u.kind.reach() { return; }
    match u.kind {
        // Dragon fire-breath: an area cull — every enemy in the blast takes a hit,
        // and the scorched ground belches a smoke cloud (a new LoS blocker).
        Kind::Dragon => {
            let blast = Sphere3::new(tx, ty, tz, u.kind.reach() * 0.5);
            for it in index.cull(&blast) {
                if it.faction != u.faction { u.attacks.push((it.id, u.kind.dmg())); }
            }
            u.emit = Some(Point3::new(tx, ty, tz));
            let c = Point3::new(tx, ty, tz);
            u.fx.push(Fx::new(FxKind::Ring, v3(c), v3(c)));
            u.fx.push(Fx::new(FxKind::Bolt, v3(u.p), v3(c))); // breath stream from the dragon
        }
        // Catapult: lob a real boulder — record the target so the apply pass
        // launches an arcing projectile that does the AoE on impact (visible
        // travel time, instead of an instant hidden blast).
        Kind::Catapult => { u.fire = Some(Point3::new(tx, terrain_height(tx, tz), tz)); }
        // Ballista: a piercing bolt — an all-hits `raycast` that does NOT stop at
        // the first unit; every enemy on the line is skewered. (Contrast the
        // archer, who stops at the first hit.) Smoke blocks the line.
        Kind::Ballista => {
            let dir = Point3::new(tx - u.p.x, ty - u.p.y, tz - u.p.z);
            let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt().max(1e-6);
            let ndir = Point3::new(dir.x / len, dir.y / len, dir.z / len);
            if smoke.raycast_dda_first(u.p, ndir, len, SMOKE_R).is_some() { return; } // blocked
            for (_, it) in index.raycast(u.p, ndir, len + 30.0, 3.5) {
                if it.id != id && it.faction != u.faction { u.attacks.push((it.id, u.kind.dmg())); }
            }
            u.fx.push(Fx::new(FxKind::Bolt, v3(u.p), v3(Point3::new(tx, ty, tz))));
        }
        // Mage chain-lightning: a `knn` from the strike point arcs to the nearest
        // enemies — up to 4 links, each taking the bolt.
        Kind::Mage => {
            let mut links = 0;
            let mut from = u.p; // the arc hops shooter → enemy → enemy …
            for (_, it) in index.knn(Point3::new(tx, ty, tz), 10) {
                if it.faction == u.faction || it.id == id { continue; }
                u.attacks.push((it.id, u.kind.dmg()));
                u.fx.push(Fx::new(FxKind::Lightning, v3(from), v3(it.p)));
                from = it.p;
                links += 1;
                if links >= 4 { break; }
            }
        }
        // Archer line-of-fire: a thick raycast at the target. The *first* unit
        // struck takes the arrow — a friend in the way blocks the shot (real
        // line-of-sight, and a `raycast` showcase). The ray starts at the
        // archer, so skip the self-hit at t≈0.
        Kind::Archer => {
            let dir = Point3::new(tx - u.p.x, ty - u.p.y, tz - u.p.z);
            let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt().max(1e-6);
            let ndir = Point3::new(dir.x / len, dir.y / len, dir.z / len);
            if smoke.raycast_dda_first(u.p, ndir, len, SMOKE_R).is_some() { return; } // smoke blocks LoS
            for (_, it) in index.raycast(u.p, ndir, len + 4.0, 3.0) {
                if it.id == id { continue; }
                if it.faction != u.faction { u.attacks.push((it.id, u.kind.dmg())); }
                u.fx.push(Fx::new(FxKind::Arrow, v3(u.p), v3(it.p))); // arrow to whatever it hit
                break; // the first thing hit stops the arrow
            }
        }
        // Healer: mend the most-wounded nearby comrade — a friendly `knn`, the
        // heal applied as *negative* damage (capped at full HP in the apply pass).
        Kind::Healer => {
            if let Some((p, hid, _, _)) = heal {
                u.attacks.push((hid, -u.kind.dmg()));
                u.fx.push(Fx::new(FxKind::Spark, v3(p), v3(p)));
            }
        }
        // Soldier / knight: single melee strike on the k-NN target.
        _ => { if let Some((_, tid, _)) = target { u.attacks.push((tid, u.kind.dmg())); } }
    }
}

// ----------------------------------------------------------------- AI: apply

/// Serial resolution of one frame's intents: move units, apply accumulated
/// damage, kill the fallen, respawn the dead, turn smoke emissions into puffs and
/// collect this frame's visual effects (aging out the old ones). Reads every
/// unit's `vel`/`attacks`/`emit`/`fx` (written by `decide`) and is the only place
/// cross-unit writes happen.
fn apply(units: &mut [Unit], smoke: &mut Vec<Puff>, effects: &mut Vec<Fx>, projectiles: &mut Vec<Projectile>, rng: &mut Rng, dt: f64, now: f64) {
    // 1) movement + cooldown tick (each unit, independent).
    for u in units.iter_mut() {
        if !u.alive() { continue; }
        u.cooldown = (u.cooldown - dt).max(0.0);
        // Wading: ground units slow right down in the river/water (a soft obstacle).
        let wade = if u.kind.altitude() == 0.0 && terrain_height(u.p.x, u.p.z) < WATER_LEVEL && !on_bridge(u.p.x, u.p.z) { 0.4 } else { 1.0 };
        let nx = (u.p.x + u.vel.0 * dt * wade).clamp(2.0, WORLD - 2.0);
        let nz = (u.p.z + u.vel.2 * dt * wade).clamp(2.0, WORLD - 2.0);
        let ground = terrain_height(nx, nz) + u.kind.radius() as f64;
        let ny = if u.kind == Kind::Dragon { (terrain_height(nx, nz) + u.kind.altitude()).max(ground) } else { ground };
        u.p = Point3::new(nx, ny, nz);
        // Face the direction of travel (for orienting the model); keep the last
        // heading while stopped.
        if u.vel.0 * u.vel.0 + u.vel.2 * u.vel.2 > 1.0 { u.face = (u.vel.0 as f32).atan2(u.vel.2 as f32); }
        // Reload after a shot (an AoE that caught nobody still fired) and kick off
        // the attack-lunge animation; otherwise let the lunge decay.
        let fired = !u.attacks.is_empty() || u.emit.is_some() || u.fire.is_some();
        if fired { u.cooldown = u.kind.cooldown(); }
        u.atk_anim = if fired { ATK_ANIM_LEN } else { (u.atk_anim - dt as f32).max(0.0) };
        // Lava burns: a ground unit standing on emissive terrain takes damage.
        if u.kind.altitude() == 0.0 && terrain_surface(nx, nz, terrain_height(nx, nz)).1 {
            u.hp -= LAVA_DPS * dt;
            if u.hp <= 0.0 { u.respawn_at = now + 4.0; }
        }
    }

    // 2) damage resolution — gather first (immutable borrow), then apply.
    let mut dmg = vec![0.0f64; units.len()];
    for u in units.iter() {
        for &(tid, d) in &u.attacks {
            if let Some(slot) = dmg.get_mut(tid as usize) { *slot += d; }
        }
    }
    for (u, d) in units.iter_mut().zip(dmg) {
        // d may be negative (a healer's mend); cap healing at full HP. Dead units
        // are out of the index, so they can't be targeted or healed back.
        if d != 0.0 && u.alive() {
            u.hp = (u.hp - d).min(u.kind.max_hp());
            if u.hp <= 0.0 { u.respawn_at = now + 4.0; } // schedule a respawn
        }
    }

    // 3) respawns — dead units return from their keep after the delay.
    for u in units.iter_mut() {
        if !u.alive() && u.respawn_at.is_finite() && now >= u.respawn_at {
            place_at_castle(rng, u);
        }
    }

    // 4) smoke — spawn a puff per emission (under the cap), age out the rest.
    for u in units.iter() {
        if let Some(p) = u.emit {
            if smoke.len() < SMOKE_CAP { smoke.push(Puff { p, born: now }); }
        }
    }
    smoke.retain(|s| now - s.born < SMOKE_LIFE);

    // 5) visual effects — stamp birth time, collect, age out (cap to bound work).
    for u in units.iter() {
        for f in &u.fx {
            if effects.len() < FX_CAP { effects.push(Fx { born: now, ..*f }); }
        }
    }
    effects.retain(|f| now - f.born < Fx::life(f.kind));

    // 6) projectiles — launch lobbed shots, fly them, resolve impacts.
    for u in units.iter() {
        if let Some(t) = u.fire {
            let d = ((t.x - u.p.x).powi(2) + (t.z - u.p.z).powi(2)).sqrt();
            let tf = (d / 120.0).clamp(0.8, 2.5); // flight time → arc height
            projectiles.push(Projectile { p: u.p, v: arc_velocity(u.p, t, tf), kind: ProjKind::Cannon, faction: u.faction, dmg: Kind::Catapult.dmg() * 1.4, r: 30.0 });
        }
    }
    let mut impacts: Vec<(Point3, Faction, f64, f64, ProjKind)> = Vec::new();
    projectiles.retain_mut(|pr| {
        pr.v.1 -= PROJ_GRAVITY * dt;
        pr.p = Point3::new(pr.p.x + pr.v.0 * dt, pr.p.y + pr.v.1 * dt, pr.p.z + pr.v.2 * dt);
        let ground = terrain_height(pr.p.x, pr.p.z);
        let out = pr.p.x < 0.0 || pr.p.x > WORLD || pr.p.z < 0.0 || pr.p.z > WORLD;
        if pr.p.y <= ground || out {
            if !out { impacts.push((Point3::new(pr.p.x, ground, pr.p.z), pr.faction, pr.dmg, pr.r, pr.kind)); }
            false
        } else { true }
    });
    for (ip, fac, dmg, r, kind) in impacts {
        for u in units.iter_mut() {
            // Cannonballs hit the firer's enemies; lava bombs scorch everyone.
            let hits = u.alive() && u.kind.altitude() == 0.0 && (kind == ProjKind::LavaRock || u.faction != fac);
            if hits {
                let d2 = (u.p.x - ip.x).powi(2) + (u.p.z - ip.z).powi(2);
                if d2 < r * r { u.hp -= dmg; if u.hp <= 0.0 { u.respawn_at = now + 4.0; } }
            }
        }
        if smoke.len() < SMOKE_CAP { smoke.push(Puff { p: Point3::new(ip.x, ip.y + 8.0, ip.z), born: now }); }
        effects.push(Fx { kind: FxKind::Ring, a: v3(ip), b: v3(ip), born: now });
    }
}

// ----------------------------------------------------------------- rendering

/// Per-faction team colour + blend amount (alpha). The shader `mix()`es the
/// model's own colours toward this, so even dark models (knights, dragons) read
/// clearly as Red or Blue.
fn faction_tint(f: Faction) -> [f32; 4] {
    match f { Faction::Red => [0.90, 0.20, 0.14, 0.22], Faction::Blue => [0.30, 0.45, 1.0, 0.22] }
}

/// Which baked animation frame this unit shows now — the clip loops at a fixed
/// rate, the units split into `ANIM_GROUPS` phase groups. Quantising the phase to
/// groups means at most `ANIM_GROUPS` distinct frames are on screen per model at
/// once, so the draw-call count is bounded regardless of the army size (while the
/// clip stays smooth over time via the frame count). Static models (`nf`≤1) → 0.
fn anim_frame(u: &Unit, now: f64, nf: usize) -> usize {
    if nf <= 1 { return 0; }
    let group = ((u.phase * std::f32::consts::FRAC_1_PI * 0.5) * ANIM_GROUPS as f32) as usize % ANIM_GROUPS;
    let off = group as f32 / ANIM_GROUPS as f32; // this group's phase in the loop
    (((now as f32 * 1.6 + off) * nf as f32) as usize) % nf
}

/// Procedural "animation" offset for a unit's model this frame (cheap, scales to
/// the whole army — no skeletal skinning): a walk-bounce while moving (idle
/// breathe when stopped) plus a forward lunge during an attack. `h` is the
/// model height; returns a world-space translation to add to the unit's base.
fn anim_offset(u: &Unit, now: f64, h: f32) -> Vec3 {
    let moving = u.vel.0 * u.vel.0 + u.vel.2 * u.vel.2 > 1.0;
    let t = now as f32 * 9.0 + u.phase;
    // Small now that the skeleton itself animates — just a touch of extra life.
    let bob = if moving { t.sin().abs() * 0.03 * h } else { (now as f32 * 1.6 + u.phase).sin() * 0.015 * h };
    let prog = if u.atk_anim > 0.0 { 1.0 - u.atk_anim / ATK_ANIM_LEN } else { 0.0 };
    let lunge = (prog * std::f32::consts::PI).sin() * h * 0.16;
    vec3(0.0, bob, 0.0) + vec3(u.face.sin(), 0.0, u.face.cos()) * lunge
}

/// Build the terrain once as smooth triangle `Mesh` **chunks**. Each vertex sits
/// at its true height (so the surface is continuous, not stepped like the old
/// per-tile cubes), and **lambert shading is baked into the vertex colour** — the
/// slope normal dotted with a sun direction — because macroquad's `draw_mesh` is
/// unlit. That shading is what makes the relief readable when the terrain is
/// otherwise one flat green. Lava (crater + flow) is emissive: drawn full-bright.
///
/// Chunked because macroquad clamps any single drawcall at 10 000 verts / 5 000
/// indices; a 6×6 grid of 25-cell chunks (676 verts / 3 750 indices each) stays
/// under both caps.
fn build_terrain_chunks() -> Vec<Mesh> {
    const RES: usize = 150;
    const CHUNK: usize = 25; // cells/side: (CHUNK+1)²=676 verts, CHUNK²·6=3 750 indices
    let step = WORLD / RES as f64;
    let light = vec3(-0.45, 0.84, -0.30).normalize();
    let nchunks = RES / CHUNK;
    let mut meshes = Vec::with_capacity(nchunks * nchunks);
    for cz in 0..nchunks {
        for cx in 0..nchunks {
            let (ix0, iz0) = (cx * CHUNK, cz * CHUNK);
            let mut vertices = Vec::with_capacity((CHUNK + 1) * (CHUNK + 1));
            for jz in 0..=CHUNK {
                for jx in 0..=CHUNK {
                    let (x, z) = ((ix0 + jx) as f64 * step, (iz0 + jz) as f64 * step);
                    let h = terrain_height(x, z);
                    // Heightfield normal via central differences.
                    let hx = terrain_height(x + step, z) - terrain_height(x - step, z);
                    let hz = terrain_height(x, z + step) - terrain_height(x, z - step);
                    let n = vec3((-hx / (2.0 * step)) as f32, 1.0, (-hz / (2.0 * step)) as f32).normalize();
                    let (base, emissive) = terrain_surface(x, z, h);
                    let col = if emissive {
                        base
                    } else {
                        let b = 0.32 + 0.68 * n.dot(light).max(0.0); // ambient + diffuse
                        Color::new(base.r * b, base.g * b, base.b * b, 1.0)
                    };
                    vertices.push(Vertex::new(x as f32, h as f32, z as f32, 0.0, 0.0, col));
                }
            }
            let w = (CHUNK + 1) as u16;
            let mut indices: Vec<u16> = Vec::with_capacity(CHUNK * CHUNK * 6);
            for jz in 0..CHUNK as u16 {
                for jx in 0..CHUNK as u16 {
                    let a = jz * w + jx;
                    indices.extend_from_slice(&[a, a + w, a + 1, a + 1, a + w, a + w + 1]);
                }
            }
            meshes.push(Mesh { vertices, indices, texture: None });
        }
    }
    meshes
}

/// Voxel (blocky) terrain. Each grid cell is a flat-topped prism at its true
/// height, with vertical cliff walls dropping to any lower neighbour — the
/// stepped "voxel" look, watertight, units sit on the tops. Baked per-corner
/// ambient occlusion (darker where taller neighbours crowd a corner) with the
/// quad-flip fix from MAP_DESIGN.md, plus the elevation colour ramp. Chunked so
/// each mesh stays under macroquad's per-draw-call index cap. The cell heights
/// are a grid → alterable (crater carving) is a future step (BACKLOG).
fn build_voxel_chunks() -> Vec<Mesh> {
    const VRES: usize = 80;  // cells/side: cell ≈ 10 world units → visibly blocky
    const CHUNK: usize = 10; // cells/side per mesh: ≤100 cells · ≤30 idx = <5000
    let step = WORLD / VRES as f64;
    let light = vec3(-0.45, 0.84, -0.30).normalize();
    // Precompute the cell-centre heights once for O(1) neighbour AO lookups.
    let heights: Vec<f64> = (0..VRES * VRES).map(|k| {
        let (i, j) = (k % VRES, k / VRES);
        terrain_height((i as f64 + 0.5) * step, (j as f64 + 0.5) * step)
    }).collect();
    let hc = |i: i32, j: i32| -> f64 {
        if i < 0 || j < 0 || i >= VRES as i32 || j >= VRES as i32 { -1000.0 } else { heights[j as usize * VRES + i as usize] }
    };
    let nchunks = VRES.div_ceil(CHUNK);
    let mut meshes = Vec::with_capacity(nchunks * nchunks);
    for cz in 0..nchunks {
        for cx in 0..nchunks {
            let (i0, i1) = (cx * CHUNK, ((cx + 1) * CHUNK).min(VRES));
            let (j0, j1) = (cz * CHUNK, ((cz + 1) * CHUNK).min(VRES));
            let mut vertices: Vec<Vertex> = Vec::new();
            let mut indices: Vec<u16> = Vec::new();
            for j in j0..j1 {
                for i in i0..i1 {
                    let (ii, jj) = (i as i32, j as i32);
                    let h = hc(ii, jj);
                    let (x0, z0) = (i as f64 * step, j as f64 * step);
                    let (x1, z1) = (x0 + step, z0 + step);
                    let (base, emissive) = terrain_surface(x0 + step * 0.5, z0 + step * 0.5, h);
                    let col = |b: f32| Color::new((base.r * b).min(1.0), (base.g * b).min(1.0), (base.b * b).min(1.0), 1.0);
                    // Per-corner AO: a corner darkens for each taller neighbour cell
                    // touching it (edge neighbours + the diagonal if an edge is up).
                    let ao = |dx: i32, dz: i32| -> f32 {
                        if emissive { return 1.0; }
                        let up = h + step * 0.5;
                        let (s1, s2) = (hc(ii + dx, jj) > up, hc(ii, jj + dz) > up);
                        let sc = hc(ii + dx, jj + dz) > up;
                        let n = s1 as i32 + s2 as i32 + (sc && (s1 || s2)) as i32;
                        1.0 - 0.16 * n as f32
                    };
                    let topb = if emissive { 1.0 } else { 0.32 + 0.68 * light.y.max(0.0) as f32 };
                    let corners = [(x0, z0, ao(-1, -1)), (x1, z0, ao(1, -1)), (x1, z1, ao(1, 1)), (x0, z1, ao(-1, 1))];
                    let bi = vertices.len() as u16;
                    for (vx, vz, a) in corners { vertices.push(Vertex::new(vx as f32, h as f32, vz as f32, 0.0, 0.0, col(topb * a))); }
                    // Quad-flip: split along the diagonal joining the brighter pair
                    // so AO interpolates symmetrically (no dark-corner triangle leak).
                    if corners[0].2 + corners[2].2 >= corners[1].2 + corners[3].2 {
                        indices.extend_from_slice(&[bi, bi + 1, bi + 2, bi, bi + 2, bi + 3]);
                    } else {
                        indices.extend_from_slice(&[bi + 1, bi + 2, bi + 3, bi + 1, bi + 3, bi]);
                    }
                    // Cliff walls: one quad per lower neighbour, top at h down to it.
                    let sideb = if emissive { 1.0 } else { 0.32 + 0.68 * 0.42 };
                    for (dx, dz, e0, e1) in [(-1i32, 0i32, (x0, z1), (x0, z0)), (1, 0, (x1, z0), (x1, z1)), (0, -1, (x0, z0), (x1, z0)), (0, 1, (x1, z1), (x0, z1))] {
                        let hn = hc(ii + dx, jj + dz);
                        if hn < h - 0.01 {
                            let bottom = hn.max(-25.0) as f32;
                            let (top, bot) = (col(sideb), col(sideb * 0.65));
                            let si = vertices.len() as u16;
                            vertices.push(Vertex::new(e0.0 as f32, h as f32, e0.1 as f32, 0.0, 0.0, top));
                            vertices.push(Vertex::new(e1.0 as f32, h as f32, e1.1 as f32, 0.0, 0.0, top));
                            vertices.push(Vertex::new(e1.0 as f32, bottom, e1.1 as f32, 0.0, 0.0, bot));
                            vertices.push(Vertex::new(e0.0 as f32, bottom, e0.1 as f32, 0.0, 0.0, bot));
                            indices.extend_from_slice(&[si, si + 1, si + 2, si, si + 2, si + 3]);
                        }
                    }
                }
            }
            if !vertices.is_empty() { meshes.push(Mesh { vertices, indices, texture: None }); }
        }
    }
    meshes
}

/// Wooden bridge decks across the river (a plank + two rails per crossing).
fn draw_bridges() {
    let deck = Color::new(0.36, 0.24, 0.13, 1.0);
    let rail = Color::new(0.28, 0.18, 0.10, 1.0);
    for &bz in &BRIDGE_Z {
        let bx = river_center_x(bz);
        let y = 9.0f32;
        draw_cube(vec3(bx as f32, y, bz as f32), vec3((BRIDGE_HALF_W * 2.0) as f32, 2.5, (BRIDGE_HALF_D * 2.0) as f32), None, deck);
        for s in [-1.0, 1.0] {
            draw_cube(vec3(bx as f32, y + 3.0, (bz + s * BRIDGE_HALF_D) as f32), vec3((BRIDGE_HALF_W * 2.0) as f32, 4.0, 1.5), None, rail);
        }
    }
}

/// Draw the transient combat effects (the visible part of each attack): arrow /
/// bolt / lightning streaks, a healer's spark, and an expanding AoE ring. Immediate
/// 3D lines, faded by age.
fn draw_effects(effects: &[Fx], now: f64) {
    for f in effects {
        let age = ((now - f.born) / Fx::life(f.kind)).clamp(0.0, 1.0) as f32;
        match f.kind {
            FxKind::Arrow => draw_line_3d(f.a, f.b, Color::new(0.96, 0.90, 0.45, 1.0)),
            FxKind::Bolt => draw_line_3d(f.a, f.b, Color::new(1.0, 0.58, 0.16, 1.0)),
            FxKind::Lightning => draw_line_3d(f.a, f.b, Color::new(0.62, 0.86, 1.0, 1.0)),
            FxKind::Spark => draw_line_3d(f.a, f.a + vec3(0.0, 7.0, 0.0), Color::new(0.40, 1.0, 0.55, 1.0)),
            FxKind::Ring => {
                let r = 8.0 + 26.0 * age; // expanding shockwave
                let col = Color::new(1.0, 0.45, 0.12, 1.0 - age);
                let n = 22;
                let mut prev = f.a + vec3(r, 1.5, 0.0);
                for i in 1..=n {
                    let t = i as f32 / n as f32 * std::f32::consts::TAU;
                    let p = f.a + vec3(r * t.cos(), 1.5, r * t.sin());
                    draw_line_3d(prev, p, col);
                    prev = p;
                }
            }
        }
    }
}

/// A minimal screen-space slider over an integer range `min..=max`. Draggable
/// handle; updates `*value` and sets `*dragging` so the caller can suppress
/// camera-orbit while it's held.
#[allow(clippy::too_many_arguments)]
fn int_slider(x: f32, y: f32, w: f32, label: &str, value: &mut usize, min: usize, max: usize, dragging: &mut bool) {
    draw_rectangle(x, y - 3.0, w, 6.0, Color::new(0.30, 0.30, 0.36, 1.0));
    let span = (max - min).max(1) as f32;
    let hx = x + ((*value).saturating_sub(min) as f32 / span) * w;
    draw_circle(hx, y, 9.0, WHITE);
    let (mx, my) = mouse_position();
    let over = mx >= x - 12.0 && mx <= x + w + 12.0 && (my - y).abs() < 16.0;
    if is_mouse_button_pressed(MouseButton::Left) && over { *dragging = true; }
    if !is_mouse_button_down(MouseButton::Left) { *dragging = false; }
    if *dragging {
        let nt = ((mx - x) / w).clamp(0.0, 1.0);
        *value = min + (nt * span).round() as usize;
    }
    draw_text(format!("{label}: {value}"), x, y - 14.0, 20.0, WHITE);
}

fn window_conf() -> Conf {
    Conf {
        window_title: "vectorial-hash — siege".to_owned(),
        window_width: 1600,
        window_height: 1000,
        platform: macroquad::miniquad::conf::Platform { swap_interval: Some(0), ..Default::default() },
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Per-run seed: $SIEGE_SEED (reproducible) or the wall clock (varies). Drives
    // both the map (terrain noise offset) and the army composition.
    let seed = std::env::var("SIEGE_SEED").ok().and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| (macroquad::miniquad::date::now() * 1000.0) as u64);
    let _ = MAP_SEED.set((seed % 100_000) as f64 * 0.01);
    let mut rng = Rng::new(seed | 1);
    let mut per_faction = PER_FACTION; // live army size per side (population slider)
    let mut cur_pop = per_faction;
    let mut pop_drag = false;
    let mut units = spawn_army(&mut rng, per_faction);
    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD);
    let mut index = Tree3::<IUnit>::new(world, 8);
    // Smoke lives in its own index so archer/ballista shots can raycast it.
    let mut smoke: Vec<Puff> = Vec::new();
    let mut effects: Vec<Fx> = Vec::new(); // transient combat visuals
    let mut projectiles: Vec<Projectile> = Vec::new(); // arcing cannonballs / lava bombs
    let mut smoke_index = Tree3::<Puff>::new(world, 8);

    // Camera orbit state — looking down on the battlefield centre.
    let mut yaw: f32 = 0.9;
    let mut pitch: f32 = 0.75;
    let mut dist: f32 = 900.0;
    let observer = vec3((WORLD * 0.5) as f32, 60.0, (WORLD * 0.5) as f32);
    let mut last_mouse = mouse_position();

    let mut renderer = {
        let gl = unsafe { get_internal_gl() };
        InstancedRenderer::new(gl.quad_context)
    };

    // Load + upload a glTF model per (faction, kind) — pirates vs undead. While
    // loading, derive each one's world-space body radius from its own XZ footprint
    // × render height, so the space a unit occupies (separation) matches what's
    // drawn. Indexed [faction][kind].
    let mut body_radius = [[4.0f64; 8]; 2];
    let mut frames_mov = [[1usize; 8]; 2]; // movement-clip frame count per (f, k)
    let mut frames_atk = [[1usize; 8]; 2]; // attack-clip frame count per (f, k)
    let models: Vec<((Faction, Kind), UnitAnim)> = {
        let gl = unsafe { get_internal_gl() };
        let mut v = Vec::with_capacity(16);
        for &f in &[Faction::Red, Faction::Blue] {
            for &k in &Kind::ALL {
                // Riders sit on the horse → their "movement" clip is the idle (no
                // running legs); everyone else walks/runs/flies.
                let mov_prefs = if k == Kind::Knight { IDLE_PREFS } else { MOVE_PREFS };
                let mov_cpu = load_glb_clip(model_for(f, k), ANIM_FRAMES, mov_prefs);
                let atk_cpu = load_glb_clip(model_for(f, k), ATTACK_FRAMES, ATTACK_PREFS);
                let (_, sc) = k.model_tweak(f);
                body_radius[f.index()][k.index()] = (mov_cpu[0].footprint * k.model_height() * sc) as f64;
                frames_mov[f.index()][k.index()] = mov_cpu.len();
                frames_atk[f.index()][k.index()] = atk_cpu.len();
                let mov = mov_cpu.iter().map(|m| renderer.upload_model(gl.quad_context, &m.vertices, &m.indices)).collect();
                let atk = atk_cpu.iter().map(|m| renderer.upload_model(gl.quad_context, &m.vertices, &m.indices)).collect();
                v.push(((f, k), UnitAnim { mov, atk }));
            }
        }
        v
    };
    // The knight is cavalry: the rider model is raised onto this (shared) horse,
    // which is bigger than the rider — so the knight's footprint is the horse's.
    let horse: Vec<ModelGpu> = {
        let gl = unsafe { get_internal_gl() };
        let cpu = load_glb_clip(include_bytes!("../../assets/siege/models/horse.glb"), ANIM_FRAMES, MOVE_PREFS);
        let hr = (cpu[0].footprint * Kind::Knight.model_height()) as f64;
        body_radius[0][Kind::Knight.index()] = hr;
        body_radius[1][Kind::Knight.index()] = hr;
        cpu.iter().map(|m| renderer.upload_model(gl.quad_context, &m.vertices, &m.indices)).collect()
    };
    // Castle model (static) for the two faction keeps, replacing the cube blocks.
    let castle = {
        let gl = unsafe { get_internal_gl() };
        let m = load_glb(include_bytes!("../../assets/siege/models/castle.glb"));
        renderer.upload_model(gl.quad_context, &m.vertices, &m.indices)
    };

    // Static terrain, built once. Voxel (blocky) by default; flip to the smooth
    // heightfield with $SIEGE_SMOOTH=1.
    let terrain_chunks = if std::env::var("SIEGE_SMOOTH").is_ok() { build_terrain_chunks() } else { build_voxel_chunks() };
    let mut now = 0.0f64; // simulation clock
    let mut paused = false;
    let mut volcano_smoke_t = 0.0f64; // next crater-puff time
    let mut volcano_erupt_t = 7.0f64; // next eruption time

    // Live thread-count control (native). The decide pass runs inside a rayon
    // pool sized by the slider; dragging the slider blocks camera-orbit.
    #[cfg(not(target_arch = "wasm32"))]
    let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    #[cfg(not(target_arch = "wasm32"))]
    let (mut n_threads, mut cur_threads) = (max_threads, max_threads);
    #[cfg(not(target_arch = "wasm32"))]
    let mut pool = rayon::ThreadPoolBuilder::new().num_threads(cur_threads).build().unwrap();
    // Mutated only by the (native-only) thread slider; immutable on wasm.
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut slider_drag = false;

    // Headless smoke hook: run N frames then exit (CI / startup-panic check).
    let max_frames: Option<u64> = std::env::var("SIEGE_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    let mut frame_no: u64 = 0;

    loop {
        // ----- input: orbit / zoom / controls -----
        let mp = mouse_position();
        if is_mouse_button_down(MouseButton::Left) && !slider_drag && !pop_drag {
            yaw += (mp.0 - last_mouse.0) * 0.01;
            pitch = (pitch + (mp.1 - last_mouse.1) * 0.01).clamp(0.05, 1.50);
        }
        last_mouse = mp;
        let wheel = mouse_wheel().1;
        if wheel != 0.0 { dist = (dist - wheel * 0.5).clamp(200.0, 1600.0); }
        if is_key_pressed(KeyCode::P) { paused = !paused; }
        // Rebuild the army when the population slider changed (or on `[`/`]`).
        if is_key_pressed(KeyCode::RightBracket) { per_faction = (per_faction + 100).min(2000); }
        if is_key_pressed(KeyCode::LeftBracket) { per_faction = per_faction.saturating_sub(100).max(20); }
        if per_faction != cur_pop {
            units = spawn_army(&mut rng, per_faction);
            cur_pop = per_faction;
        }

        let dt = (get_frame_time() as f64).min(0.05); // clamp huge hitches

        // ----- simulation step -----
        if !paused {
            now += dt;

            // Rebuild the index from this frame's live positions. The build is
            // serial and cheap; the queries (decide) are the parallel part.
            index.clear();
            for (i, u) in units.iter().enumerate() {
                if u.alive() { index.insert(IUnit { id: i as u32, faction: u.faction, p: u.p, health: (u.hp / u.kind.max_hp()) as f32 }); }
            }
            // Rebuild the smoke index from last frame's live puffs.
            smoke_index.clear();
            for s in &smoke { smoke_index.insert(*s); }

            // Decide (read-only on both indices) then apply (serial resolution).
            // The decide pass fans out over the rayon pool (native) — each unit
            // mutates only itself while reading the shared indices. wasm: serial.
            #[cfg(not(target_arch = "wasm32"))]
            {
                if cur_threads != n_threads {
                    pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
                    cur_threads = n_threads;
                }
                let (idx, smk, br) = (&index, &smoke_index, &body_radius);
                pool.install(|| units.par_iter_mut().enumerate().for_each(|(i, u)| decide(u, i as u32, idx, smk, br)));
            }
            #[cfg(target_arch = "wasm32")]
            for i in 0..units.len() { decide(&mut units[i], i as u32, &index, &smoke_index, &body_radius); }
            apply(&mut units, &mut smoke, &mut effects, &mut projectiles, &mut rng, dt, now);

            // Volcano: a constant smoke plume from the crater, plus an occasional
            // eruption — a lava spray (orange streaks) + a smoke burst + a ring.
            let (vcx, vcz) = (WORLD * 0.5, WORLD * 0.5);
            let vcy = terrain_height(vcx, vcz) + 6.0;
            volcano_smoke_t -= dt;
            if volcano_smoke_t <= 0.0 {
                volcano_smoke_t = 0.35;
                if smoke.len() < SMOKE_CAP {
                    smoke.push(Puff { p: Point3::new(vcx + rng.range(-12.0, 12.0), vcy, vcz + rng.range(-12.0, 12.0)), born: now });
                }
            }
            volcano_erupt_t -= dt;
            if volcano_erupt_t <= 0.0 {
                volcano_erupt_t = rng.range(9.0, 16.0);
                let c = vec3(vcx as f32, vcy as f32, vcz as f32);
                effects.push(Fx { kind: FxKind::Ring, a: c, b: c, born: now });
                for _ in 0..16 {
                    let ang = rng.range(0.0, std::f64::consts::TAU) as f32;
                    let (up, out) = (rng.range(30.0, 65.0) as f32, rng.range(6.0, 48.0) as f32);
                    let tip = c + vec3(ang.cos() * out, up, ang.sin() * out);
                    effects.push(Fx { kind: FxKind::Bolt, a: c, b: tip, born: now }); // lava streak
                }
                for _ in 0..7 {
                    if smoke.len() < SMOKE_CAP {
                        smoke.push(Puff { p: Point3::new(vcx + rng.range(-22.0, 22.0), vcy + rng.range(0.0, 16.0), vcz + rng.range(-22.0, 22.0)), born: now });
                    }
                }
                // Spit real arcing lava bombs that land out on the slopes + scorch
                // whoever's there (uses the projectile system).
                let crater = Point3::new(vcx, vcy, vcz);
                for _ in 0..6 {
                    let ang = rng.range(0.0, std::f64::consts::TAU);
                    let dist = rng.range(80.0, 230.0);
                    let land = Point3::new((vcx + ang.cos() * dist).clamp(4.0, WORLD - 4.0), 0.0, (vcz + ang.sin() * dist).clamp(4.0, WORLD - 4.0));
                    let land = Point3::new(land.x, terrain_height(land.x, land.z), land.z);
                    let tf = rng.range(1.6, 2.6);
                    projectiles.push(Projectile { p: crater, v: arc_velocity(crater, land, tf), kind: ProjKind::LavaRock, faction: Faction::Red, dmg: 60.0, r: 26.0 });
                }
            }
        }

        // ----- render 3D -----
        clear_background(Color::new(0.55, 0.68, 0.85, 1.0)); // sky
        let eye = observer + vec3(
            dist * pitch.cos() * yaw.cos(),
            dist * pitch.sin(),
            dist * pitch.cos() * yaw.sin(),
        );
        let cam = Camera3D { position: eye, up: vec3(0.0, 1.0, 0.0), target: observer, ..Default::default() };
        set_camera(&cam);
        let mvp = cam.matrix();

        for m in &terrain_chunks { draw_mesh(m); }
        draw_bridges(); // castles are drawn as instanced models in the gl block below
        draw_effects(&effects, now);

        // Units → one instanced draw per (faction, kind), using its glTF model.
        // Group live units into [faction][kind] buckets of model matrices
        // (place · face · scale, plus the procedural animation offset). Knights
        // are cavalry: a horse at the feet + the rider raised onto its back.
        // [faction][kind][frame] instance buckets — units split by which baked
        // animation frame they show this instant, so each frame instances together.
        // Separate movement and attack clips: a unit that's mid-attack shows the
        // attack clip, else the movement (or idle, for riders) clip. Buckets are
        // [faction][kind][frame] per clip.
        let mut mov_b: [[Vec<Vec<EffectInstance>>; 8]; 2] =
            std::array::from_fn(|fi| std::array::from_fn(|ki| vec![Vec::new(); frames_mov[fi][ki]]));
        let mut atk_b: [[Vec<Vec<EffectInstance>>; 8]; 2] =
            std::array::from_fn(|fi| std::array::from_fn(|ki| vec![Vec::new(); frames_atk[fi][ki]]));
        let mut horses: Vec<Vec<EffectInstance>> = vec![Vec::new(); horse.len()];
        let (mut red, mut blue) = (0usize, 0usize);
        for u in units.iter() {
            if !u.alive() { continue; }
            match u.faction { Faction::Red => red += 1, Faction::Blue => blue += 1 }
            let feet_y = (u.p.y - u.kind.radius() as f64) as f32; // drop sphere centre to ground
            let (yaw_off, sc) = u.kind.model_tweak(u.faction);
            let h = u.kind.model_height() * sc;
            let yaw = u.face + yaw_off;
            let base = vec3(u.p.x as f32, feet_y, u.p.z as f32) + anim_offset(u, now, h);
            let tint = faction_tint(u.faction);
            let (fi, ki) = (u.faction.index(), u.kind.index());
            let inst = if u.kind == Kind::Knight {
                let hh = h; // horse height (= knight model height)
                let horse_m = Mat4::from_translation(base) * Mat4::from_rotation_y(u.face) * Mat4::from_scale(Vec3::splat(hh));
                horses[anim_frame(u, now, horse.len())].push(EffectInstance::new(horse_m, tint));
                let rider = Mat4::from_translation(base + vec3(0.0, hh * 0.5, 0.0)) * Mat4::from_rotation_y(yaw) * Mat4::from_scale(Vec3::splat(hh * 0.72));
                EffectInstance::new(rider, tint)
            } else {
                let m = Mat4::from_translation(base) * Mat4::from_rotation_y(yaw) * Mat4::from_scale(Vec3::splat(h));
                EffectInstance::new(m, tint)
            };
            if u.atk_anim > 0.0 {
                atk_b[fi][ki][attack_frame(u.atk_anim, frames_atk[fi][ki])].push(inst);
            } else {
                mov_b[fi][ki][anim_frame(u, now, frames_mov[fi][ki])].push(inst);
            }
        }
        {
            let gl = unsafe { get_internal_gl() };
            let light = vec3(-0.45, 0.84, -0.30).normalize();
            for ((f, k), anim) in &models {
                for (fr, gpu) in anim.mov.iter().enumerate() { renderer.draw_models(gl.quad_context, gpu, &mov_b[f.index()][k.index()][fr], mvp, light); }
                for (fr, gpu) in anim.atk.iter().enumerate() { renderer.draw_models(gl.quad_context, gpu, &atk_b[f.index()][k.index()][fr], mvp, light); }
            }
            for (fr, gpu) in horse.iter().enumerate() {
                renderer.draw_models(gl.quad_context, gpu, &horses[fr], mvp, light); // cavalry mounts
            }
            // Castles — one model per faction keep, facing the map centre.
            let mut castle_inst = Vec::with_capacity(2);
            for f in [Faction::Red, Faction::Blue] {
                let (cx, cz) = f.castle();
                let yaw = (WORLD * 0.5 - cx).atan2(WORLD * 0.5 - cz) as f32;
                let m = Mat4::from_translation(vec3(cx as f32, terrain_height(cx, cz) as f32, cz as f32)) * Mat4::from_rotation_y(yaw) * Mat4::from_scale(Vec3::splat(62.0));
                castle_inst.push(EffectInstance::new(m, faction_tint(f)));
            }
            renderer.draw_models(gl.quad_context, &castle, &castle_inst, mvp, light);
        }
        // Projectiles: small spheres arcing through the air (cannonballs / lava).
        for pr in &projectiles {
            let (col, rad) = match pr.kind {
                ProjKind::Cannon => (Color::new(0.16, 0.15, 0.17, 1.0), 3.6),
                ProjKind::LavaRock => (Color::new(1.0, 0.45, 0.10, 1.0), 4.2),
            };
            draw_sphere(v3(pr.p), rad, None, col);
        }
        // Smoke: each cloud is a few translucent billows that rise, spread and
        // fade as they age — so it reads as smoke yet you still see the fight
        // through it. Deterministic offsets (from the spawn point) keep each puff
        // stable frame to frame.
        for s in &smoke {
            let age = ((now - s.born) / SMOKE_LIFE).clamp(0.0, 1.0) as f32;
            let base_r = SMOKE_R as f32 * (0.42 + 0.55 * age);
            let centre = vec3(s.p.x as f32, s.p.y as f32 + age * 22.0, s.p.z as f32); // rises
            let seed = (s.p.x * 0.13 + s.p.z * 0.71) as f32;
            for k in 0..2 {
                let a = seed + k as f32 * 2.39996; // ~golden-angle spread
                let off = vec3(a.sin(), (a * 1.7).sin() * 0.35 + 0.25, a.cos()) * base_r * 0.55;
                let rr = base_r * (0.5 + 0.22 * (a * 2.1).cos().abs());
                let alpha = (0.12 * (1.0 - age) + 0.02) * if k == 0 { 1.2 } else { 0.85 };
                draw_sphere(centre + off, rr, None, Color::new(0.80, 0.80, 0.85, alpha));
            }
        }

        // ----- HUD -----
        set_default_camera();
        draw_text("vectorial-hash — SIEGE", 16.0, 28.0, 30.0, WHITE);
        draw_text(format!("fps {}", get_fps()), 16.0, 54.0, 22.0, LIGHTGRAY);
        draw_text(format!("Red {red}"), 16.0, 80.0, 24.0, Color::new(0.95, 0.4, 0.35, 1.0));
        draw_text(format!("Blue {blue}"), 16.0, 104.0, 24.0, Color::new(0.45, 0.6, 1.0, 1.0));
        // Live population slider (per faction) — the spatial-index stress lever.
        int_slider(20.0, 150.0, 220.0, "army/side", &mut per_faction, 20, 2000, &mut pop_drag);
        // Live thread-count slider drives the parallel AI pass (native only).
        #[cfg(not(target_arch = "wasm32"))]
        int_slider(20.0, 192.0, 220.0, "threads", &mut n_threads, 1, max_threads, &mut slider_drag);
        draw_text(
            "drag: orbit  scroll: zoom  P: pause  [ ]: \u{00b1}pop",
            16.0, screen_height() - 18.0, 20.0, LIGHTGRAY,
        );
        if paused { draw_text("PAUSED", screen_width() * 0.5 - 50.0, 40.0, 36.0, YELLOW); }

        next_frame().await;
        frame_no += 1;
        if let Some(m) = max_frames { if frame_no >= m { break; } }
    }
}
