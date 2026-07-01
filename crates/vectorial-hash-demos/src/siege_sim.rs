//! Shared simulation core for the `siege` battle — used by **both** the macroquad
//! binary (`bin/siege.rs`) and the wgpu binary (`bin/siege_wgpu.rs`) so the two
//! renderers stay in lockstep instead of drifting. Everything here is graphics-
//! free: it depends only on `vectorial_hash` (the index) and plain data.
//!
//! The split mirrors the per-unit-AI pattern: [`decide`] is read-only on the
//! shared index and writes only a unit's own intent (parallel-safe), [`apply`]
//! resolves the cross-unit writes serially. Each renderer owns its own GPU code
//! and calls these; the few render-data helpers that both need (model bytes,
//! tint, animation-frame selection) live here too, returning plain arrays.

use std::f64::consts::TAU;

use vectorial_hash::{Point3, Positioned3, Sphere3, Tree3};

// ---------------------------------------------------------------- world config

/// Per-run map seed — offsets the terrain noise so each run is a different map.
/// Set once at startup (from the wall clock, or `$SIEGE_SEED`); 0 in tests.
static MAP_SEED: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
pub fn map_seed() -> f64 { *MAP_SEED.get_or_init(|| 0.0) }
/// Set the per-run map seed (idempotent — only the first call takes effect).
pub fn set_map_seed(s: f64) { let _ = MAP_SEED.set(s); }

/// Live toggle for the boids **separation** (collision avoidance). On by default;
/// the `C` key flips it so you can A/B "no two units share a space" vs. letting
/// them overlap — the visual *and* the perf (units clump into the same cells,
/// which deepens the index and slows the k-NN). See [`decide`].
static SEPARATION_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub fn set_separation(on: bool) { SEPARATION_ON.store(on, std::sync::atomic::Ordering::Relaxed); }
pub fn separation_on() -> bool { SEPARATION_ON.load(std::sync::atomic::Ordering::Relaxed) }

/// "Classic Reynolds" steering: integrate the steering as **acceleration** so
/// turns have inertia (smoother, more organic) instead of snapping the velocity
/// each frame. Off by default — `$SIEGE_REYNOLDS=1` turns it on. Returns the blend
/// factor for `vel → steer` per frame (1.0 = the default kinematic snap).
static REYNOLDS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn reynolds_blend() -> f64 {
    if *REYNOLDS.get_or_init(|| std::env::var("SIEGE_REYNOLDS").is_ok()) { 0.18 } else { 1.0 }
}

pub const WORLD: f64 = 800.0; // battlefield is WORLD × WORLD in the ground plane
pub const SKY: f64 = 260.0; // index height — heights reach ~150, the dragon flies
pub const PER_FACTION: usize = 500; // units each side spawns with (tunable live)
pub const ATK_ANIM_LEN: f32 = 0.45; // attack-clip / lunge play window (seconds)
pub const LAVA_DPS: f64 = 45.0; // damage per second to a ground unit standing in lava
pub const WATER_LEVEL: f64 = 6.0; // terrain below this is water (units wade slowly)
pub const ANIM_FRAMES: usize = 12; // baked movement-clip frames per model (smoothness)
pub const ATTACK_FRAMES: usize = 6; // baked attack-clip frames (one-shot, fewer = fewer draws)
pub const ANIM_GROUPS: usize = 5; // units share a frame within a phase group (caps draw calls)
// Clip name preferences (priority-ordered substrings) — movement, attack, idle.
pub const MOVE_PREFS: &[&str] = &["walk", "run", "flying", "fly", "move"];
pub const ATTACK_PREFS: &[&str] = &["attack", "sword", "slash", "cast", "shoot", "punch", "bite", "kick"];
pub const IDLE_PREFS: &[&str] = &["idle"];

/// Which frame of the one-shot attack clip to show (plays once over the attack).
pub fn attack_frame(atk_anim: f32, nf: usize) -> usize {
    if nf <= 1 { return 0; }
    let prog = (1.0 - atk_anim / ATK_ANIM_LEN).clamp(0.0, 0.999);
    (prog * nf as f32) as usize
}

/// Which baked movement frame this unit shows now — the clip loops at a fixed
/// rate, the units split into `ANIM_GROUPS` phase groups so at most that many
/// distinct frames are on screen per model at once (bounding the draw-call count
/// regardless of army size). Static models (`nf`≤1) → 0.
pub fn anim_frame(u: &Unit, now: f64, nf: usize) -> usize {
    if nf <= 1 { return 0; }
    let group = ((u.phase * std::f32::consts::FRAC_1_PI * 0.5) * ANIM_GROUPS as f32) as usize % ANIM_GROUPS;
    let off = group as f32 / ANIM_GROUPS as f32; // this group's phase in the loop
    (((now as f32 * 1.6 + off) * nf as f32) as usize) % nf
}

// ------------------------------------------------------------------------- rng

pub struct Rng(u64);
impl Rng {
    pub fn new(s: u64) -> Self { Rng(s | 1) }
    #[allow(clippy::should_implement_trait)] // xorshift step, not an iterator
    pub fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    pub fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

// --------------------------------------------------------------------- terrain

fn hash2(x: i32, z: i32) -> f64 {
    let mut h = (x as u32).wrapping_mul(0x1659_5e3d).wrapping_add((z as u32).wrapping_mul(0x27d4_eb2f));
    h ^= h >> 15; h = h.wrapping_mul(0x85eb_ca6b); h ^= h >> 13;
    (h & 0x00ff_ffff) as f64 / 0x00ff_ffff as f64
}

/// Smoothstep-interpolated value noise in [0,1).
pub fn vnoise(x: f64, z: f64) -> f64 {
    let (xi, zi) = (x.floor() as i32, z.floor() as i32);
    let (fx, fz) = (x - xi as f64, z - zi as f64);
    let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let a = hash2(xi, zi); let b = hash2(xi + 1, zi);
    let c = hash2(xi, zi + 1); let d = hash2(xi + 1, zi + 1);
    let ab = a + (b - a) * sx; let cd = c + (d - c) * sx;
    ab + (cd - ab) * sz
}

/// The river's centre-line x at a given z (it meanders along Z, seeded).
pub fn river_center_x(z: f64) -> f64 {
    let o = map_seed();
    WORLD * 0.32 + (z * 0.011 + o).sin() * 95.0 + (z * 0.004 + o).cos() * 45.0
}

/// Z positions of the bridges across the river (away from the volcano band).
pub const BRIDGE_Z: [f64; 4] = [120.0, 260.0, 540.0, 680.0];
pub const BRIDGE_HALF_W: f64 = 48.0; // spans the river channel
pub const BRIDGE_HALF_D: f64 = 12.0; // deck depth along Z

/// Is (x,z) on a bridge deck? (so the unit crosses dry instead of wading.)
pub fn on_bridge(x: f64, z: f64) -> bool {
    BRIDGE_Z.iter().any(|&bz| (z - bz).abs() < BRIDGE_HALF_D && (x - river_center_x(bz)).abs() < BRIDGE_HALF_W)
}

/// How strongly a point sits in the river channel: 1 at the centre line, 0 past
/// the banks, faded to 0 near the volcano so the river doesn't carve the cone.
pub fn river_factor(x: f64, z: f64) -> f64 {
    let d = (x - river_center_x(z)).abs();
    let w = 44.0;
    if d >= w { return 0.0; }
    let t = 1.0 - d / w; // 0 at bank, 1 at centre
    let dc = ((x - WORLD * 0.5).powi(2) + (z - WORLD * 0.5).powi(2)).sqrt();
    let vol_mask = (dc / 220.0).clamp(0.0, 1.0); // 0 at volcano, 1 far away
    t * t * vol_mask
}

// ------------------------------------------------------------------- forests

/// Forest cover at (x, z): 0 = open ground, up to 1 = deep wood. A handful of
/// seeded copses, away from the river (nothing grows in the channel). Woods slow
/// ground movement and give ranged **cover** (Total War style) — see [`apply`]
/// (the wade/forest slowdown) and [`decide`] (halved ranged damage vs a target
/// in the trees). Flyers (the dragon) pass over untouched.
/// The seeded copse centres + radii, computed once (they depend only on the map
/// seed). Avoids recomputing 12 trig calls on every `forest_factor` — which is
/// hit per ground unit per frame (~20k×) in [`apply`].
fn forest_copses() -> &'static [(f64, f64, f64)] {
    static COPSES: std::sync::OnceLock<Vec<(f64, f64, f64)>> = std::sync::OnceLock::new();
    COPSES.get_or_init(|| {
        let o = map_seed();
        (0..6).map(|i| {
            let a = i as f64;
            let cx = WORLD * (0.15 + 0.70 * ((a * 1.7 + o).sin() * 0.5 + 0.5));
            let cz = WORLD * (0.12 + 0.76 * ((a * 2.3 + o * 1.4).cos() * 0.5 + 0.5));
            let r = WORLD * (0.07 + 0.05 * ((a * 0.9 + o).sin() * 0.5 + 0.5));
            (cx, cz, r)
        }).collect()
    })
}

pub fn forest_factor(x: f64, z: f64) -> f64 {
    // Cheap copse distance test first — the common case (unit not in any wood) is
    // a handful of subtracts, no trig, no sqrt. Only pay the river check (trig)
    // when actually inside a copse.
    let mut f = 0.0_f64;
    for &(cx, cz, r) in forest_copses() {
        let d2 = (x - cx).powi(2) + (z - cz).powi(2);
        if d2 < r * r { let t = 1.0 - d2.sqrt() / r; f = f.max(t * t * (3.0 - 2.0 * t)); }
    }
    if f > 0.0 && river_factor(x, z) > 0.15 { return 0.0; } // nothing grows in the channel
    f
}

/// Deep enough forest to count as cover / slow a unit down.
pub fn in_forest(x: f64, z: f64) -> bool { forest_factor(x, z) > 0.35 }

/// Deterministic canopy positions `(x, z)` for drawing the woods — a jittered
/// grid kept where the forest is dense. The renderer looks up the terrain height.
pub fn forest_trees() -> Vec<(f64, f64)> {
    let o = map_seed();
    let step = 17.0;
    let mut trees = Vec::new();
    let mut gz = 20.0;
    while gz < WORLD - 20.0 {
        let mut gx = 20.0;
        while gx < WORLD - 20.0 {
            let jx = gx + (gx * 0.21 + gz * 0.13 + o).sin() * step * 0.5;
            let jz = gz + (gx * 0.17 - gz * 0.29 + o).cos() * step * 0.5;
            if forest_factor(jx, jz) > 0.45 { trees.push((jx, jz)); }
            gx += step;
        }
        gz += step;
    }
    trees
}

/// Terrain height at a world (x,z): two octaves of hills, a central volcano cone,
/// and a carved river channel. Deterministic — called per terrain tile + unit step.
pub fn terrain_height(x: f64, z: f64) -> f64 {
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

/// Surface colour (linear RGB) at a point, plus an `emissive` flag (true = lava,
/// drawn full-bright). Elevation ramp water → sand → grass → rock → snow, with a
/// glowing crater pool and a lava flow down one flank. Graphics-free: returns a
/// plain `[f32; 3]` each renderer wraps into its own colour type.
pub fn terrain_surface(x: f64, z: f64, h: f64) -> ([f32; 3], bool) {
    let (cx, cz) = (WORLD * 0.5, WORLD * 0.5);
    let (px, pz) = (x - cx, z - cz);
    let d = (px * px + pz * pz).sqrt();
    if d < 30.0 { return ([1.0, 0.46, 0.10], true); } // crater pool
    // Lava river: a narrow azimuth wedge flowing down one slope (a different
    // flank each run).
    let flow_dir = -2.1 + (map_seed() * 0.9).sin() * 1.8;
    let mut dang = (pz.atan2(px) - flow_dir).abs();
    if dang > std::f64::consts::PI { dang = TAU - dang; }
    if d < 165.0 && dang < 0.15 { return ([0.96, 0.34, 0.07], true); }
    if d < 150.0 && h > 70.0 { return ([0.30, 0.11, 0.08], false); } // scorched rock
    (elevation_color(h), false)
}

/// Smooth elevation colour ramp: flat-ish biome bands (water / sand / grass /
/// rock / snow) with a **soft smoothstep transition** at each boundary, so a
/// slope shows a continuous gradient instead of hard "contour" steps — and the
/// sand line doesn't break into patches where the height wobbles along the river.
fn elevation_color(h: f64) -> [f32; 3] {
    const WATER: [f32; 3] = [0.12, 0.28, 0.45];
    const SAND: [f32; 3] = [0.62, 0.56, 0.34];
    const GRASS: [f32; 3] = [0.20, 0.42, 0.18];
    const ROCK: [f32; 3] = [0.38, 0.34, 0.30];
    const SNOW: [f32; 3] = [0.82, 0.84, 0.88];
    let mix = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    // smoothstep(e0,e1,h) — 0 below e0, 1 above e1, smooth in between.
    let s = |e0: f64, e1: f64| { let t = ((h - e0) / (e1 - e0)).clamp(0.0, 1.0) as f32; t * t * (3.0 - 2.0 * t) };
    // Chained: each band blends fully in (t→1) before the next transition starts,
    // so flat bands stay flat and only the boundaries (non-overlapping windows) blur.
    let mut c = WATER;
    c = mix(c, SAND, s(5.0, 8.0));
    c = mix(c, GRASS, s(9.0, 14.0));
    c = mix(c, ROCK, s(54.0, 66.0));
    c = mix(c, SNOW, s(88.0, 100.0));
    c
}

// ----------------------------------------------------------------- unit model

#[derive(Clone, Copy, PartialEq)]
pub enum Faction { Red, Blue }
impl Faction {
    pub fn other(self) -> Faction { match self { Faction::Red => Faction::Blue, Faction::Blue => Faction::Red } }
    pub fn castle(self) -> (f64, f64) { match self { Faction::Red => (90.0, 90.0), Faction::Blue => (WORLD - 90.0, WORLD - 90.0) } }
    pub fn index(self) -> usize { match self { Faction::Red => 0, Faction::Blue => 1 } }
    pub const ALL: [Faction; 2] = [Faction::Red, Faction::Blue];
}

/// The `.glb` model bytes for a (faction, kind): **Red = pirates**, **Blue =
/// undead**. Quaternius CC0 (Witch is CC-BY) — see assets/siege/CREDITS.md. Some
/// models are shared across factions (dragon, cannon) and told apart by the tint.
pub fn model_for(f: Faction, k: Kind) -> &'static [u8] {
    use Faction::{Blue, Red};
    match (f, k) {
        // Pirates (Red)
        (Red, Kind::Soldier) => include_bytes!("../assets/siege/models/anne.glb"),
        (Red, Kind::Archer) => include_bytes!("../assets/siege/models/sharky.glb"),
        (Red, Kind::Knight) => include_bytes!("../assets/siege/models/pirate_captain.glb"),
        (Red, Kind::Mage) => include_bytes!("../assets/siege/models/witch.glb"),
        (Red, Kind::Healer) => include_bytes!("../assets/siege/models/henry.glb"),
        // Undead (Blue)
        (Blue, Kind::Soldier) => include_bytes!("../assets/siege/models/zombie.glb"),
        (Blue, Kind::Archer) => include_bytes!("../assets/siege/models/skeleton_a.glb"),
        (Blue, Kind::Knight) => include_bytes!("../assets/siege/models/skeleton_sword.glb"),
        (Blue, Kind::Mage) => include_bytes!("../assets/siege/models/slime.glb"),
        (Blue, Kind::Healer) => include_bytes!("../assets/siege/models/bat.glb"),
        // Shared
        (_, Kind::Dragon) => include_bytes!("../assets/siege/models/dragon.glb"),
        (_, Kind::Catapult) | (_, Kind::Ballista) => include_bytes!("../assets/siege/models/cannon.glb"),
    }
}

/// The basename of the `.glb` for a (faction, kind). The **web** build fetches
/// models by name from `models/<name>` (so they aren't baked into the wasm —
/// keeps it ~1 MB instead of ~10). Must stay in lockstep with [`model_for`].
pub fn model_file(f: Faction, k: Kind) -> &'static str {
    use Faction::{Blue, Red};
    match (f, k) {
        (Red, Kind::Soldier) => "anne.glb",
        (Red, Kind::Archer) => "sharky.glb",
        (Red, Kind::Knight) => "pirate_captain.glb",
        (Red, Kind::Mage) => "witch.glb",
        (Red, Kind::Healer) => "henry.glb",
        (Blue, Kind::Soldier) => "zombie.glb",
        (Blue, Kind::Archer) => "skeleton_a.glb",
        (Blue, Kind::Knight) => "skeleton_sword.glb",
        (Blue, Kind::Mage) => "slime.glb",
        (Blue, Kind::Healer) => "bat.glb",
        (_, Kind::Dragon) => "dragon.glb",
        (_, Kind::Catapult) | (_, Kind::Ballista) => "cannon.glb",
    }
}

/// Every distinct `.glb` the siege demo needs — unit models (via [`model_file`])
/// plus the two props loaded directly (`horse`, `castle`). The web build fetches
/// these at startup; the build script copies exactly this set to `docs/models/`.
pub const SIEGE_MODEL_FILES: &[&str] = &[
    "anne.glb", "sharky.glb", "pirate_captain.glb", "witch.glb", "henry.glb",
    "zombie.glb", "skeleton_a.glb", "skeleton_sword.glb", "slime.glb", "bat.glb",
    "dragon.glb", "cannon.glb", "horse.glb", "castle.glb",
];

#[derive(Clone, Copy, PartialEq)]
pub enum Kind { Soldier, Archer, Knight, Dragon, Catapult, Mage, Ballista, Healer }
impl Kind {
    pub fn speed(self) -> f64 { match self { Kind::Soldier => 26.0, Kind::Archer => 22.0, Kind::Knight => 52.0, Kind::Dragon => 60.0, Kind::Catapult => 10.0, Kind::Mage => 20.0, Kind::Ballista => 13.0, Kind::Healer => 24.0 } }
    pub fn max_hp(self) -> f64 { match self { Kind::Soldier => 100.0, Kind::Archer => 60.0, Kind::Knight => 180.0, Kind::Dragon => 1400.0, Kind::Catapult => 160.0, Kind::Mage => 70.0, Kind::Ballista => 140.0, Kind::Healer => 90.0 } }
    /// Engagement range — firing range for the siege engines, heal range for the healer.
    pub fn reach(self) -> f64 { match self { Kind::Soldier => 9.0, Kind::Archer => 150.0, Kind::Knight => 12.0, Kind::Dragon => 60.0, Kind::Catapult => 260.0, Kind::Mage => 120.0, Kind::Ballista => 240.0, Kind::Healer => 70.0 } }
    /// Damage per strike — for the healer, the (positive) amount healed.
    pub fn dmg(self) -> f64 { match self { Kind::Soldier => 14.0, Kind::Archer => 18.0, Kind::Knight => 30.0, Kind::Dragon => 22.0, Kind::Catapult => 30.0, Kind::Mage => 16.0, Kind::Ballista => 26.0, Kind::Healer => 24.0 } }
    pub fn cooldown(self) -> f64 { match self { Kind::Soldier => 0.8, Kind::Archer => 1.1, Kind::Knight => 1.0, Kind::Dragon => 0.5, Kind::Catapult => 2.4, Kind::Mage => 1.3, Kind::Ballista => 1.7, Kind::Healer => 0.9 } }
    pub fn radius(self) -> f32 { match self { Kind::Soldier => 3.0, Kind::Archer => 3.0, Kind::Knight => 4.2, Kind::Dragon => 11.0, Kind::Catapult => 5.5, Kind::Mage => 3.4, Kind::Ballista => 5.0, Kind::Healer => 3.2 } }
    /// Ground units sit on the terrain; the dragon flies at a fixed altitude
    /// (low enough to menace the ground — it engages by horizontal distance).
    pub fn altitude(self) -> f64 { match self { Kind::Dragon => 46.0, _ => 0.0 } }

    /// All eight kinds, in `index()` order — the render groups units by this.
    pub const ALL: [Kind; 8] = [Kind::Soldier, Kind::Archer, Kind::Knight, Kind::Dragon, Kind::Catapult, Kind::Mage, Kind::Ballista, Kind::Healer];
    pub fn index(self) -> usize { match self { Kind::Soldier => 0, Kind::Archer => 1, Kind::Knight => 2, Kind::Dragon => 3, Kind::Catapult => 4, Kind::Mage => 5, Kind::Ballista => 6, Kind::Healer => 7 } }

    /// Per-model orientation + size corrections — some Quaternius models face a
    /// different axis or read over/undersized. Returns (yaw offset rad, scale mul).
    pub fn model_tweak(self, f: Faction) -> (f32, f32) {
        match (f, self) {
            (Faction::Blue, Kind::Mage) => (-std::f32::consts::FRAC_PI_2, 0.55), // slime: faces +X, reads big
            _ => (0.0, 1.0),
        }
    }

    /// Visual model height in world units (the model is normalised to height 1).
    /// Per-kind because some models read bigger than their collision sphere.
    pub fn model_height(self) -> f32 {
        match self {
            Kind::Catapult | Kind::Ballista => self.radius() * 1.7, // chunky cannon model
            Kind::Knight => self.radius() * 2.3, // horse + rider; keep it trim
            Kind::Dragon => self.radius() * 2.2,
            _ => self.radius() * 2.6,
        }
    }

    /// True for the siege engines that bombard from a standoff (kite back) rather
    /// than charging into melee.
    pub fn is_artillery(self) -> bool { matches!(self, Kind::Catapult | Kind::Ballista) }
}

/// Per-faction team colour + blend amount (alpha). The renderer `mix()`es the
/// model's own colours toward this, so even dark models (knights, dragons) read
/// clearly as Red or Blue.
pub fn faction_tint(f: Faction) -> [f32; 4] {
    match f { Faction::Red => [0.90, 0.20, 0.14, 0.22], Faction::Blue => [0.30, 0.45, 1.0, 0.22] }
}

pub struct Unit {
    pub faction: Faction,
    pub kind: Kind,
    pub p: Point3,
    pub hp: f64,
    pub cooldown: f64,
    pub respawn_at: f64, // sim-time at which a dead unit returns (f64::INFINITY = alive)
    // Intent written by the *decide* pass (reads only this unit); consumed by the
    // serial *apply* pass. Keeping writes unit-local is what makes decide parallel.
    pub vel: (f64, f64, f64), // actual velocity (apply integrates it; persists for inertia)
    pub steer: (f64, f64, f64), // desired velocity from decide (the steering target)
    pub attacks: Vec<(u32, f64)>, // (target unit id, damage) — many for AoE (dragon)
    pub emit: Option<Point3>, // strike point that should spawn a smoke puff this frame
    pub fire: Option<Point3>, // catapult: target point to lob a projectile at this frame
    pub aim: Option<f32>, // heading to the current enemy (so artillery aims forward while kiting)
    pub fx: Vec<Fx>, // visible effects this unit produced this frame
    pub face: f32, // heading (radians about Y) for orienting the model
    pub phase: f32, // per-unit animation phase offset (so they don't bob in sync)
    pub atk_anim: f32, // attack-lunge countdown (seconds), set when the unit strikes
}
impl Unit {
    pub fn alive(&self) -> bool { self.hp > 0.0 }
}

/// The lightweight item actually stored in the index: id + faction + position.
/// Decoupled from `Unit` so the decide pass can hold `&Tree3<IUnit>` immutably
/// while it mutates the `units` slice.
#[derive(Clone, Copy)]
pub struct IUnit { pub id: u32, pub faction: Faction, pub p: Point3, pub health: f32, pub face: f32 }
impl Positioned3 for IUnit { fn position(&self) -> Point3 { self.p } }

// ----------------------------------------------------------------- smoke (LoS)

pub const SMOKE_R: f64 = 24.0; // puff radius — also the raycast corridor half-width
pub const SMOKE_LIFE: f64 = 3.5; // seconds before a puff dissipates
pub const SMOKE_CAP: usize = 240; // hard cap on live puffs

/// A smoke cloud — a dynamic line-of-sight blocker. Catapult and dragon strikes
/// spawn one; it lives in its own `Tree3` so an archer/ballista shot can
/// `raycast` it: a puff between the shooter and the target blocks the shot.
#[derive(Clone, Copy)]
pub struct Puff { pub p: Point3, pub born: f64 }
impl Positioned3 for Puff { fn position(&self) -> Point3 { self.p } }

// ------------------------------------------------------------- visual effects

/// A transient combat effect — the *visible* part of an attack (the queries
/// resolve instantly, so without these the fight is invisible). Endpoints are
/// plain `[f32; 3]` (graphics-free); each renderer draws them its own way.
#[derive(Clone, Copy)]
pub enum FxKind { Arrow, Bolt, Lightning, Ring, Spark }
#[derive(Clone, Copy)]
pub struct Fx { pub kind: FxKind, pub a: [f32; 3], pub b: [f32; 3], pub born: f64 }
impl Fx {
    pub fn new(kind: FxKind, a: [f32; 3], b: [f32; 3]) -> Fx { Fx { kind, a, b, born: 0.0 } }
    pub fn life(kind: FxKind) -> f64 { match kind { FxKind::Arrow | FxKind::Bolt => 0.14, FxKind::Lightning => 0.10, FxKind::Ring => 0.45, FxKind::Spark => 0.30 } }
}
pub const FX_CAP: usize = 4000; // hard cap on live effects

/// `Point3` → `[f32; 3]` (for Fx endpoints).
pub fn fa3(p: Point3) -> [f32; 3] { [p.x as f32, p.y as f32, p.z as f32] }

// ------------------------------------------------------------- projectiles

pub const PROJ_GRAVITY: f64 = 220.0; // ballistic arc gravity (world units / s²)
pub const ARTY_STANDOFF: f64 = 95.0; // artillery keeps at least this far from the enemy

#[derive(Clone, Copy, PartialEq)]
pub enum ProjKind { Cannon, LavaRock }

/// A ballistic projectile (cannonball / lava bomb) — arcs under gravity and on
/// landing does a `Sphere3` AoE. Travel time makes the shot visible.
pub struct Projectile { pub p: Point3, pub v: (f64, f64, f64), pub kind: ProjKind, pub faction: Faction, pub dmg: f64, pub r: f64 }

/// Launch velocity for a ballistic arc from `a` to `t` over flight time `tf`.
pub fn arc_velocity(a: Point3, t: Point3, tf: f64) -> (f64, f64, f64) {
    ((t.x - a.x) / tf, (t.y - a.y) / tf + 0.5 * PROJ_GRAVITY * tf, (t.z - a.z) / tf)
}

// ----------------------------------------------------------------- craters

/// Alterable-terrain state: the impact craters carved into the battlefield.
/// **Shared** between the simulation (so units sink into them) and the renderers
/// (so both the voxel and smooth meshes deform identically) — one source of truth.
/// Each crater is a smooth bowl `(x, z, radius)`; `depth_at` sums the overlap.
#[derive(Default)]
pub struct Craters {
    list: Vec<(f32, f32, f32)>,
    pub dirty: bool, // a crater landed since the last remesh
}
/// Max craters kept (oldest dropped) — bounds both the depth sum and the remesh.
pub const CRATER_CAP: usize = 64;

impl Craters {
    pub fn new() -> Self { Self::default() }
    /// Carve a crater at `(x, z)` of the given AoE radius; caps the list.
    pub fn carve(&mut self, x: f64, z: f64, r: f64) {
        self.list.push((x as f32, z as f32, r as f32));
        if self.list.len() > CRATER_CAP { let drop = self.list.len() - CRATER_CAP; self.list.drain(0..drop); }
        self.dirty = true;
    }
    /// How far the ground has been lowered at `(x, z)` — a cone bowl per crater,
    /// ~0.45·r deep at the centre, summed. The single source the mesh + the unit
    /// ground both read, so they always agree.
    pub fn depth_at(&self, x: f64, z: f64) -> f64 {
        let mut d = 0.0;
        for &(cx, cz, cr) in &self.list {
            let dist = (((x - cx as f64).powi(2) + (z - cz as f64).powi(2)).sqrt()) as f32;
            if dist < cr {
                // Smoothstep bowl: zero slope at the rim *and* the centre, so the
                // crater is round (no cone tip / hard rim corner) even on the
                // smooth mesh. ~0.45·r deep at the middle.
                let t = 1.0 - dist / cr;
                d += (cr * 0.45 * t * t * (3.0 - 2.0 * t)) as f64;
            }
        }
        d
    }
    pub fn is_empty(&self) -> bool { self.list.is_empty() }
}

/// Ground surface height at `(x, z)` accounting for craters — the value both the
/// unit feet and the terrain mesh should use.
pub fn ground_height(x: f64, z: f64, craters: &Craters) -> f64 {
    terrain_height(x, z) - craters.depth_at(x, z)
}

// ----------------------------------------------------------------- spawning

/// Draw a unit kind from the roster distribution: mostly foot soldiers, then
/// archers/knights, a few mages/healers, and *rare* siege engines + dragons
/// (cannons and dragons are a third of what they were; the freed slots went to
/// foot soldiers, per user request).
pub fn roll_kind(rng: &mut Rng) -> Kind {
    let roll = rng.unit();
    if roll < 0.48 { Kind::Soldier }
    else if roll < 0.68 { Kind::Archer }
    else if roll < 0.80 { Kind::Knight }
    else if roll < 0.90 { Kind::Mage }
    else if roll < 0.98 { Kind::Healer }
    else if roll < 0.98833 { Kind::Ballista } // 2.5% -> 0.83%
    else if roll < 0.995 { Kind::Catapult }   // 2.0% -> 0.67%
    else { Kind::Dragon }                      // 1.5% -> 0.50%
}

pub fn spawn_unit(rng: &mut Rng, faction: Faction) -> Unit {
    let kind = roll_kind(rng);
    spawn_unit_of(rng, faction, kind)
}

/// Build a unit of a *given* kind (position still randomised by `place_at_castle`).
pub fn spawn_unit_of(rng: &mut Rng, faction: Faction, kind: Kind) -> Unit {
    let mut u = Unit {
        faction, kind,
        p: Point3::new(0.0, 0.0, 0.0),
        hp: kind.max_hp(),
        cooldown: 0.0,
        respawn_at: f64::INFINITY,
        vel: (0.0, 0.0, 0.0),
        steer: (0.0, 0.0, 0.0),
        attacks: Vec::new(),
        emit: None,
        fire: None,
        aim: None,
        fx: Vec::new(),
        face: 0.0,
        phase: rng.range(0.0, TAU) as f32,
        atk_anim: 0.0,
    };
    place_at_castle(rng, &mut u);
    u
}

/// Drop a unit at a random point just outside its castle, on the terrain. The
/// spread is wide and fully random (no per-kind spawn points), so kinds mix into
/// a broad front instead of marching as rigid lanes.
pub fn place_at_castle(rng: &mut Rng, u: &mut Unit) {
    let (cx, cz) = u.faction.castle();
    let x = (cx + rng.range(-95.0, 95.0)).clamp(2.0, WORLD - 2.0);
    let z = (cz + rng.range(-95.0, 95.0)).clamp(2.0, WORLD - 2.0);
    let y = terrain_height(x, z) + u.kind.altitude() + u.kind.radius() as f64;
    u.p = Point3::new(x, y, z);
    u.hp = u.kind.max_hp();
    u.respawn_at = f64::INFINITY;
    u.vel = (0.0, 0.0, 0.0); // start still (matters under inertia)
    u.steer = (0.0, 0.0, 0.0);
}

pub fn spawn_army(rng: &mut Rng, per_faction: usize) -> Vec<Unit> {
    // Mirror-identical armies: roll ONE roster of kinds and give both factions the
    // same count of every type, so neither side gets luckier on rare units
    // (dragons/cannons). Positions + model/tint still differ by faction; respawns
    // keep a unit's kind, so the balance holds over the battle.
    let roster: Vec<Kind> = (0..per_faction).map(|_| roll_kind(rng)).collect();
    let mut units = Vec::with_capacity(per_faction * 2);
    for &k in &roster { units.push(spawn_unit_of(rng, Faction::Red, k)); }
    for &k in &roster { units.push(spawn_unit_of(rng, Faction::Blue, k)); }
    units
}

// ----------------------------------------------------------------- AI: decide

/// One unit's per-frame brain — *read-only* on the shared index, writes only into
/// `u`'s own intent fields. This is the body that fans out over rayon
/// (`par_iter_mut`): each unit reads the shared index and mutates only itself.
///
/// Three library queries, one per concern: **k-NN** finds the nearest enemy
/// (targeting) *and* the nearby friends (boids); the dragon's AoE is a sphere
/// **`cull`**; the archer/ballista line-of-fire is a thick **`raycast`**.
pub fn decide(u: &mut Unit, id: u32, index: &Tree3<IUnit>, smoke: &Tree3<Puff>, sep: &SepTables) {
    u.steer = (0.0, 0.0, 0.0); // desired velocity (apply blends `vel` toward it)
    u.attacks.clear();
    u.emit = None;
    u.fire = None;
    u.aim = None;
    u.fx.clear();
    if !u.alive() { return; }

    // One k-NN pass yields both the nearest enemy (targeting) and the nearby
    // friends used for flocking (separation + cohesion).
    let mut target: Option<(Point3, u32, f64)> = None; // nearest enemy (pos, id, dist)
    let mut heal: Option<(Point3, u32, f32, f64)> = None; // most-wounded friend (pos, id, health, dist)
    let (mut sep_x, mut sep_z) = (0.0, 0.0); // separation push (from ANY neighbour)
    let (mut coh_x, mut coh_z, mut friends) = (0.0, 0.0, 0u32); // cohesion centroid
    let (mut ali_x, mut ali_z) = (0.0f64, 0.0f64); // alignment: sum of friends' heading vectors
    let (fi, ki) = (u.faction.index(), u.kind.index());
    let sep_dist = sep.sep_dist(fi, ki); // two bodies of this size shan't overlap
    let sep_on = separation_on(); // live collision-avoidance toggle (C key)
    for (d, it) in index.knn(u.p, 16) {
        if it.id == id { continue; }
        // Separation acts only *within* an altitude layer: a flying dragon must
        // not be shoved sideways by the ground troops directly under it (and vice
        // versa), or it drifts along with — and clumps above — its own army
        // instead of advancing. Same layer (|Δy| small) → normal body separation.
        if sep_on && d < sep_dist && (it.p.y - u.p.y).abs() < 24.0 {
            // Precomputed force table (offset → away·weight) — load + add, no sqrt/div.
            let (fx, fz) = sep.force(fi, ki, it.p.x - u.p.x, it.p.z - u.p.z);
            sep_x += fx; sep_z += fz;
        }
        if it.faction != u.faction {
            if target.is_none() { target = Some((it.p, it.id, d)); }
        } else {
            coh_x += it.p.x; coh_z += it.p.z; friends += 1;
            ali_x += (it.face.sin()) as f64; ali_z += (it.face.cos()) as f64; // heading as a unit vector
            if it.health < 0.97 && heal.is_none_or(|(_, _, h, _)| it.health < h) { heal = Some((it.p, it.id, it.health, d)); }
        }
    }

    // The healer peels off to its most-wounded comrade; with nobody hurt it
    // advances WITH the army (toward the nearest enemy / the enemy keep).
    let advance = match target {
        Some((tp, _, d)) => (tp.x, tp.y, tp.z, d),
        None => { let (cx, cz) = u.faction.other().castle(); (cx, u.p.y, cz, f64::INFINITY) }
    };
    let (tx, ty, tz, tdist) = match (u.kind, heal) {
        (Kind::Healer, Some((p, _, _, d))) => (p.x, p.y, p.z, d),
        _ => advance,
    };

    // Velocity = seek (scaled by approach — zero once in reach) + separation
    // (ALWAYS applied) + gentle cohesion for ground-melee formations.
    let seek = (tx - u.p.x, tz - u.p.z);
    let slen = (seek.0 * seek.0 + seek.1 * seek.1).sqrt().max(1e-6);
    let speed = u.kind.speed();
    // The dragon engages by HORIZONTAL distance (it flies far above the ground).
    let engage = if u.kind == Kind::Dragon { slen } else { tdist };
    // Artillery kites to keep a standoff; everyone else closes until in reach.
    let approach = if u.kind.is_artillery() {
        if engage < ARTY_STANDOFF { -speed } else if engage > u.kind.reach() * 0.9 { speed } else { 0.0 }
    } else if engage < u.kind.reach() * 0.8 { 0.0 } else { speed };
    let (mut vx, mut vz) = (seek.0 / slen * approach, seek.1 / slen * approach);
    // Dragons: when they'd otherwise hover over the target (approach == 0), circle
    // it instead of piling up. The tangential sign is split by id so a group
    // spreads into a ring (CW/CCW) rather than stalling in a clump.
    if u.kind == Kind::Dragon && approach == 0.0 {
        let dir = if id & 1 == 0 { 1.0 } else { -1.0 };
        vx += -seek.1 / slen * speed * 0.85 * dir;
        vz += seek.0 / slen * speed * 0.85 * dir;
    }
    vx += sep_x * speed * 0.7;
    vz += sep_z * speed * 0.7;
    if matches!(u.kind, Kind::Soldier | Kind::Knight) && friends > 0 {
        let (cx, cz) = (coh_x / friends as f64 - u.p.x, coh_z / friends as f64 - u.p.z);
        let cl = (cx * cx + cz * cz).sqrt().max(1e-6);
        vx += cx / cl * speed * 0.12; vz += cz / cl * speed * 0.12; // cohesion
        // Alignment: steer toward the friends' average heading, so a band advances
        // as a coherent line instead of a milling crowd (the third boids rule).
        let al = (ali_x * ali_x + ali_z * ali_z).sqrt();
        if al > 1e-3 { vx += ali_x / al * speed * 0.10; vz += ali_z / al * speed * 0.10; }
    }
    let vl = (vx * vx + vz * vz).sqrt();
    let cap = speed * 1.5;
    if vl > cap { let s = cap / vl; vx *= s; vz *= s; }
    u.steer = (vx, 0.0, vz); // desired velocity; apply integrates it (with inertia if Reynolds)

    // Aim: face the nearest enemy while one is in k-NN range — so artillery that
    // kites *backward* still fires forward, and melee orient toward the fight
    // even when separation shoves them sideways. (Travelling units with no enemy
    // in range fall back to facing their velocity, in `apply`.)
    if let Some((tp, _, _)) = target {
        u.aim = Some(((tp.x - u.p.x) as f32).atan2((tp.z - u.p.z) as f32));
    }

    // Attacking: only when in reach and off cooldown.
    if u.cooldown > 0.0 || engage > u.kind.reach() { return; }
    // Ranged cover: a target standing in the trees takes half damage from arrows
    // and bolts (Total War style). Melee ignores it; the dragon strikes from above.
    let cover = |p: Point3| -> f64 { if in_forest(p.x, p.z) { 0.5 } else { 1.0 } };
    match u.kind {
        // Dragon fire-breath: an area cull — every enemy in the blast takes a hit,
        // and the scorched ground belches a smoke cloud (a new LoS blocker).
        Kind::Dragon => {
            let blast = Sphere3::new(tx, ty, tz, u.kind.reach() * 0.5);
            for it in index.cull(&blast) {
                if it.faction != u.faction { u.attacks.push((it.id, u.kind.dmg())); }
            }
            u.emit = Some(Point3::new(tx, ty, tz));
            let c = fa3(Point3::new(tx, ty, tz));
            u.fx.push(Fx::new(FxKind::Ring, c, c));
            u.fx.push(Fx::new(FxKind::Bolt, fa3(u.p), c)); // breath stream from the dragon
        }
        // Catapult: lob a real boulder — record the target so the apply pass
        // launches an arcing projectile that does the AoE on impact.
        Kind::Catapult => { u.fire = Some(Point3::new(tx, terrain_height(tx, tz), tz)); }
        // Ballista: a piercing bolt — an all-hits `raycast` that does NOT stop at
        // the first unit; every enemy on the line is skewered. Smoke blocks it.
        Kind::Ballista => {
            let dir = Point3::new(tx - u.p.x, ty - u.p.y, tz - u.p.z);
            let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt().max(1e-6);
            let ndir = Point3::new(dir.x / len, dir.y / len, dir.z / len);
            if smoke.raycast_dda_first(u.p, ndir, len, SMOKE_R).is_some() { return; } // blocked
            for (_, it) in index.raycast(u.p, ndir, len + 30.0, 3.5) {
                if it.id != id && it.faction != u.faction { u.attacks.push((it.id, u.kind.dmg() * cover(it.p))); }
            }
            u.fx.push(Fx::new(FxKind::Bolt, fa3(u.p), fa3(Point3::new(tx, ty, tz))));
        }
        // Mage chain-lightning: a `knn` from the strike point arcs to the nearest
        // enemies — up to 4 links, each taking the bolt.
        Kind::Mage => {
            let mut links = 0;
            let mut from = u.p; // the arc hops shooter → enemy → enemy …
            for (_, it) in index.knn(Point3::new(tx, ty, tz), 10) {
                if it.faction == u.faction || it.id == id { continue; }
                u.attacks.push((it.id, u.kind.dmg()));
                u.fx.push(Fx::new(FxKind::Lightning, fa3(from), fa3(it.p)));
                from = it.p;
                links += 1;
                if links >= 4 { break; }
            }
        }
        // Archer line-of-fire: a thick raycast at the target. The *first* unit
        // struck takes the arrow — a friend in the way blocks the shot.
        Kind::Archer => {
            let dir = Point3::new(tx - u.p.x, ty - u.p.y, tz - u.p.z);
            let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt().max(1e-6);
            let ndir = Point3::new(dir.x / len, dir.y / len, dir.z / len);
            if smoke.raycast_dda_first(u.p, ndir, len, SMOKE_R).is_some() { return; } // smoke blocks LoS
            for (_, it) in index.raycast(u.p, ndir, len + 4.0, 3.0) {
                if it.id == id { continue; }
                if it.faction != u.faction { u.attacks.push((it.id, u.kind.dmg() * cover(it.p))); }
                u.fx.push(Fx::new(FxKind::Arrow, fa3(u.p), fa3(it.p))); // arrow to whatever it hit
                break; // the first thing hit stops the arrow
            }
        }
        // Healer: mend the most-wounded nearby comrade — a friendly `knn`, the
        // heal applied as *negative* damage (capped at full HP in the apply pass).
        Kind::Healer => {
            if let Some((p, hid, _, _)) = heal {
                u.attacks.push((hid, -u.kind.dmg()));
                u.fx.push(Fx::new(FxKind::Spark, fa3(p), fa3(p)));
            }
        }
        // Soldier / knight: single melee strike on the k-NN target.
        _ => { if let Some((_, tid, _)) = target { u.attacks.push((tid, u.kind.dmg())); } }
    }
}

// ----------------------------------------------------------------- AI: apply

/// Serial resolution of one frame's intents: move units, apply accumulated
/// damage, kill the fallen, respawn the dead, turn smoke emissions into puffs and
/// collect this frame's visual effects (aging out the old ones). The only place
/// cross-unit writes happen.
/// Returns this frame's ground impacts (point + AoE radius) — cannon / lava bombs
/// — so a renderer can carve craters there (the wgpu/macroquad alterable terrain).
#[allow(clippy::too_many_arguments)] // a frame's worth of mutable sim state
pub fn apply(units: &mut [Unit], smoke: &mut Vec<Puff>, effects: &mut Vec<Fx>, projectiles: &mut Vec<Projectile>, craters: &Craters, rng: &mut Rng, dt: f64, now: f64) -> Vec<(Point3, f64)> {
    // 1) movement + cooldown tick (each unit, independent).
    for u in units.iter_mut() {
        if !u.alive() { continue; }
        u.cooldown = (u.cooldown - dt).max(0.0);
        // Integrate the steering into the actual velocity: a snap by default
        // (blend = 1 → vel == steer, the kinematic model), or with **inertia** when
        // `$SIEGE_REYNOLDS=1` (blend < 1 → turns ease in, more organic).
        let k = reynolds_blend();
        u.vel.0 += (u.steer.0 - u.vel.0) * k;
        u.vel.1 = u.steer.1;
        u.vel.2 += (u.steer.2 - u.vel.2) * k;
        // Wading: ground units slow right down in the river/water (a soft obstacle);
        // forest undergrowth slows them too (flyers pass over both). The strongest
        // slowdown wins.
        let wade: f64 = if u.kind.altitude() == 0.0 && terrain_height(u.p.x, u.p.z) < WATER_LEVEL && !on_bridge(u.p.x, u.p.z) { 0.4 } else { 1.0 };
        let forest: f64 = if u.kind.altitude() == 0.0 { 1.0 - 0.5 * forest_factor(u.p.x, u.p.z) } else { 1.0 };
        let slow = wade.min(forest);
        let nx = (u.p.x + u.vel.0 * dt * slow).clamp(2.0, WORLD - 2.0);
        let nz = (u.p.z + u.vel.2 * dt * slow).clamp(2.0, WORLD - 2.0);
        let th = terrain_height(nx, nz); // computed once, reused for ground + lava
        let surf = th - craters.depth_at(nx, nz); // base terrain minus any crater
        let ground = surf + u.kind.radius() as f64;
        let ny = if u.kind == Kind::Dragon { (surf + u.kind.altitude()).max(ground) } else { ground };
        u.p = Point3::new(nx, ny, nz);
        // Face the enemy while engaged (the `aim` heading); else face the direction
        // of travel; keep the last heading while idle.
        if let Some(a) = u.aim { u.face = a; }
        else if u.vel.0 * u.vel.0 + u.vel.2 * u.vel.2 > 1.0 { u.face = (u.vel.0 as f32).atan2(u.vel.2 as f32); }
        // Reload after a shot and kick off the attack-lunge animation.
        let fired = !u.attacks.is_empty() || u.emit.is_some() || u.fire.is_some();
        if fired { u.cooldown = u.kind.cooldown(); }
        u.atk_anim = if fired { ATK_ANIM_LEN } else { (u.atk_anim - dt as f32).max(0.0) };
        // Lava burns: a ground unit standing on emissive terrain takes damage.
        if u.kind.altitude() == 0.0 && terrain_surface(nx, nz, th).1 {
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
        if d != 0.0 && u.alive() {
            u.hp = (u.hp - d).min(u.kind.max_hp());
            if u.hp <= 0.0 { u.respawn_at = now + 4.0; }
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
    let mut craters: Vec<(Point3, f64)> = Vec::new();
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
        effects.push(Fx { kind: FxKind::Ring, a: fa3(ip), b: fa3(ip), born: now });
        craters.push((ip, r));
    }
    craters
}

// ----------------------------------------------------------------- volcano

/// The central volcano's running state (next plume puff / next eruption time).
pub struct Volcano { pub smoke_t: f64, pub erupt_t: f64 }
impl Default for Volcano { fn default() -> Self { Volcano { smoke_t: 0.0, erupt_t: 7.0 } } }
impl Volcano { pub fn new() -> Self { Self::default() } }

/// Step the volcano: a constant crater plume, plus an occasional eruption — a
/// lava spray (orange streaks) + a smoke burst + a ring + real arcing lava bombs
/// (the projectile system) that land on the slopes and scorch whoever's there.
pub fn volcano_step(v: &mut Volcano, smoke: &mut Vec<Puff>, effects: &mut Vec<Fx>, projectiles: &mut Vec<Projectile>, rng: &mut Rng, dt: f64, now: f64) {
    let (vcx, vcz) = (WORLD * 0.5, WORLD * 0.5);
    let vcy = terrain_height(vcx, vcz) + 6.0;
    v.smoke_t -= dt;
    if v.smoke_t <= 0.0 {
        v.smoke_t = 0.35;
        if smoke.len() < SMOKE_CAP {
            smoke.push(Puff { p: Point3::new(vcx + rng.range(-12.0, 12.0), vcy, vcz + rng.range(-12.0, 12.0)), born: now });
        }
    }
    v.erupt_t -= dt;
    if v.erupt_t <= 0.0 {
        v.erupt_t = rng.range(9.0, 16.0);
        let c = [vcx as f32, vcy as f32, vcz as f32];
        effects.push(Fx { kind: FxKind::Ring, a: c, b: c, born: now });
        for _ in 0..16 {
            let ang = rng.range(0.0, TAU) as f32;
            let (up, out) = (rng.range(30.0, 65.0) as f32, rng.range(6.0, 48.0) as f32);
            let tip = [c[0] + ang.cos() * out, c[1] + up, c[2] + ang.sin() * out];
            effects.push(Fx { kind: FxKind::Bolt, a: c, b: tip, born: now }); // lava streak
        }
        for _ in 0..7 {
            if smoke.len() < SMOKE_CAP {
                smoke.push(Puff { p: Point3::new(vcx + rng.range(-22.0, 22.0), vcy + rng.range(0.0, 16.0), vcz + rng.range(-22.0, 22.0)), born: now });
            }
        }
        // Spit real arcing lava bombs that land out on the slopes.
        let crater = Point3::new(vcx, vcy, vcz);
        for _ in 0..6 {
            let ang = rng.range(0.0, TAU);
            let dist = rng.range(80.0, 230.0);
            let land = Point3::new((vcx + ang.cos() * dist).clamp(4.0, WORLD - 4.0), 0.0, (vcz + ang.sin() * dist).clamp(4.0, WORLD - 4.0));
            let land = Point3::new(land.x, terrain_height(land.x, land.z), land.z);
            let tf = rng.range(1.6, 2.6);
            projectiles.push(Projectile { p: crater, v: arc_velocity(crater, land, tf), kind: ProjKind::LavaRock, faction: Faction::Red, dmg: 60.0, r: 26.0 });
        }
    }
}

/// Per-(faction,kind) body radius used for separation, defaulting to the
/// collision sphere — a renderer with real model footprints overrides entries.
pub fn default_body_radius() -> [[f64; 8]; 2] {
    let mut br = [[4.0f64; 8]; 2];
    for row in &mut br { for k in Kind::ALL { row[k.index()] = (k.radius() * 2.0) as f64; } }
    br
}

/// **Precomputed separation-force table** — one grid per (faction, kind), indexed
/// by the integer relative offset `(Δx, Δz) → (Fx, Fz)`, so each neighbour's
/// separation push is a `load + add` instead of a `sqrt` + three divides. It's the
/// project's template idea (trade precompute for runtime maths) applied to the
/// boids steering: the force is a pure function of the offset, so it's tabulable.
///
/// **Measured outcome (kept honest):** on a modern CPU the table is actually a
/// touch *slower* than just doing the maths — a few divisions are cheap and
/// pipeline well, whereas the lookup costs a memory load and the big-unit grids
/// (a dragon's sep_dist ≈ 48 → a ~75 KB grid) miss the cache. The "memory wall":
/// compute got cheap, latency didn't. So the **default is the live maths**; set
/// `$SIEGE_BOID_TABLE=1` to use the table (for the demonstration / the benchmark).
/// The quantisation error is negligible either way (a soft steering nudge, capped
/// at the unit vector × weight). See `bench_boid_force_table`.
pub struct SepTables { grids: [[SepGrid; 8]; 2], live: bool }

struct SepGrid {
    r: f64,               // separation distance (= body_radius · 2)
    half: i32,            // grid covers [-half, half] in each axis
    size: usize,          // 2·half + 1
    force: Vec<[f32; 2]>, // size·size of (Fx, Fz)
}
impl SepGrid {
    fn build(r: f64) -> Self {
        let half = r.ceil().max(1.0) as i32;
        let size = (2 * half + 1) as usize;
        let mut force = vec![[0.0f32; 2]; size * size];
        for j in 0..size {
            for i in 0..size {
                let (dx, dz) = ((i as i32 - half) as f64, (j as i32 - half) as f64); // neighbour offset
                let d = (dx * dx + dz * dz).sqrt();
                if d > 1e-3 && d < r {
                    let w = 1.0 - d / r; // away from the neighbour, linear taper
                    force[j * size + i] = [(-dx / d * w) as f32, (-dz / d * w) as f32];
                }
            }
        }
        SepGrid { r, half, size, force }
    }
}
impl SepTables {
    pub fn new(body_radius: &[[f64; 8]; 2]) -> Self {
        // Default = live maths (measured faster); $SIEGE_BOID_TABLE=1 = the table.
        Self::new_with(body_radius, std::env::var("SIEGE_BOID_TABLE").is_err())
    }
    /// Force the table (`live=false`) or the exact-maths (`live=true`) path —
    /// for the A/B micro-benchmark, regardless of the env flag.
    pub fn new_with(body_radius: &[[f64; 8]; 2], live: bool) -> Self {
        let grids = std::array::from_fn(|fi| std::array::from_fn(|ki| SepGrid::build(body_radius[fi][ki] * 2.0)));
        SepTables { grids, live }
    }
    #[inline] pub fn sep_dist(&self, fi: usize, ki: usize) -> f64 { self.grids[fi][ki].r }
    /// Separation force from a neighbour at relative offset `(dx, dz)`.
    #[inline] pub fn force(&self, fi: usize, ki: usize, dx: f64, dz: f64) -> (f64, f64) {
        let g = &self.grids[fi][ki];
        if self.live { // exact maths (A/B comparison path)
            let d = (dx * dx + dz * dz).sqrt().max(1e-3);
            let w = 1.0 - d / g.r;
            return (-dx / d * w, -dz / d * w);
        }
        let i = (dx.round() as i32 + g.half).clamp(0, g.size as i32 - 1) as usize;
        let j = (dz.round() as i32 + g.half).clamp(0, g.size as i32 - 1) as usize;
        let f = g.force[j * g.size + i];
        (f[0] as f64, f[1] as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorial_hash::Aabb;

    #[test]
    fn terrain_is_deterministic_and_finite() {
        // Same input → same output (deterministic), and finite everywhere.
        for &(x, z) in &[(123.0, 456.0), (0.0, 0.0), (WORLD, WORLD), (WORLD * 0.5, WORLD * 0.5)] {
            assert_eq!(terrain_height(x, z), terrain_height(x, z));
            assert!(terrain_height(x, z).is_finite());
        }
        // The volcano cone raises the mid-slope above the same noise sampled far
        // outside the cone (compare points where the cone term dominates the noise).
        let mid_cone = terrain_height(WORLD * 0.5 + 40.0, WORLD * 0.5); // d=40, big cone term
        let outside = terrain_height(WORLD * 0.5 + 360.0, WORLD * 0.5); // d=360, no cone
        assert!(mid_cone > outside, "the volcano cone should stand above the plain");
    }

    #[test]
    fn army_spawns_balanced_and_alive() {
        let mut rng = Rng::new(7);
        let army = spawn_army(&mut rng, 50);
        assert_eq!(army.len(), 100);
        assert_eq!(army.iter().filter(|u| u.faction == Faction::Red).count(), 50);
        assert!(army.iter().all(|u| u.alive()));
    }

    #[test]
    fn a_few_steps_do_not_panic_and_conserve_ids() {
        let mut rng = Rng::new(11);
        let mut units = spawn_army(&mut rng, 60);
        let mut index = Tree3::<IUnit>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD), 8);
        let smoke_bounds = Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD);
        let mut smoke: Vec<Puff> = Vec::new();
        let mut effects: Vec<Fx> = Vec::new();
        let mut projectiles: Vec<Projectile> = Vec::new();
        let mut volcano = Volcano::new();
        let mut craters = Craters::new();
        let sep = SepTables::new(&default_body_radius());
        let mut now = 0.0;
        for _ in 0..30 {
            now += 0.05;
            index.clear();
            for (i, u) in units.iter().enumerate() {
                if u.alive() { index.insert(IUnit { id: i as u32, faction: u.faction, p: u.p, health: (u.hp / u.kind.max_hp()) as f32, face: u.face }); }
            }
            let mut smoke_index = Tree3::<Puff>::new(smoke_bounds, 8);
            for s in &smoke { smoke_index.insert(*s); }
            for i in 0..units.len() { decide(&mut units[i], i as u32, &index, &smoke_index, &sep); }
            let impacts = apply(&mut units, &mut smoke, &mut effects, &mut projectiles, &craters, &mut rng, 0.05, now);
            for (ip, r) in impacts { craters.carve(ip.x, ip.z, r); }
            volcano_step(&mut volcano, &mut smoke, &mut effects, &mut projectiles, &mut rng, 0.05, now);
        }
        // Craters carved by any impacts must lower the ground where they landed.
        if !craters.is_empty() { assert!(ground_height(WORLD * 0.5, WORLD * 0.5, &craters) <= terrain_height(WORLD * 0.5, WORLD * 0.5)); }
        // The battle ran without panic and units are still all accounted for.
        assert_eq!(units.len(), 120);
    }

    /// Micro-benchmark of the separation-force **table vs exact maths**, across the
    /// 4 combos (index yes/no × table yes/no). Illustrates the project's two axes:
    /// the **index** cuts the *number* of force evals (N² → k·N), the **table**
    /// cuts the *cost* of each. Run with:
    ///   cargo test -p vectorial-hash-demos --release -- --ignored --nocapture bench_boid
    #[test]
    #[ignore]
    fn bench_boid_force_table() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = Rng::new(99);
        let units = spawn_army(&mut rng, 1000); // 2000 units
        let br = default_body_radius();
        let tbl = SepTables::new_with(&br, false); // precomputed table
        let live = SepTables::new_with(&br, true); // exact sqrt/div
        let mut index = Tree3::<IUnit>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD), 8);
        for (i, u) in units.iter().enumerate() { index.insert(IUnit { id: i as u32, faction: u.faction, p: u.p, health: 1.0, face: u.face }); }

        // (A) With the k-NN index — the realistic decide separation path.
        let sep_indexed = |s: &SepTables| {
            let mut acc = (0.0, 0.0);
            for u in &units {
                let (fi, ki) = (u.faction.index(), u.kind.index());
                let sd = s.sep_dist(fi, ki);
                for (d, it) in index.knn(u.p, 16) {
                    if d < sd { let (fx, fz) = s.force(fi, ki, it.p.x - u.p.x, it.p.z - u.p.z); acc.0 += fx; acc.1 += fz; }
                }
            }
            acc
        };
        // (B) Brute force, no index — O(N²), where the table really pays. Smaller N.
        let n2 = 400.min(units.len());
        let sep_brute = |s: &SepTables| {
            let mut acc = (0.0, 0.0);
            for i in 0..n2 {
                let (u, fi, ki) = (&units[i], units[i].faction.index(), units[i].kind.index());
                let sd = s.sep_dist(fi, ki);
                for (j, v) in units.iter().enumerate().take(n2) {
                    if i == j { continue; }
                    let (dx, dz) = (v.p.x - u.p.x, v.p.z - u.p.z);
                    if dx * dx + dz * dz < sd * sd { let (fx, fz) = s.force(fi, ki, dx, dz); acc.0 += fx; acc.1 += fz; }
                }
            }
            acc
        };
        let time = |reps: u32, f: &dyn Fn(&SepTables) -> (f64, f64), s: &SepTables| {
            for _ in 0..2 { black_box(f(s)); } // warm up
            let t = Instant::now();
            for _ in 0..reps { black_box(f(s)); }
            t.elapsed() / reps
        };
        println!("\n== boids separation: table vs exact maths ==");
        println!("index + table  : {:?}", time(20, &sep_indexed, &tbl));
        println!("index + maths  : {:?}", time(20, &sep_indexed, &live));
        println!("no-index+table : {:?} (N={n2})", time(20, &sep_brute, &tbl));
        println!("no-index+maths : {:?} (N={n2})", time(20, &sep_brute, &live));
        println!("(index cuts the COUNT of evals; table cuts the COST of each)\n");
    }
}
