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
use vectorial_hash_demos::instanced3d::{Instance, InstancedRenderer, Mode};

// ---------------------------------------------------------------- world config

const WORLD: f64 = 800.0; // battlefield is WORLD × WORLD in the ground plane
const SKY: f64 = 260.0; // index height — heights reach ~150, the dragon flies
const PER_FACTION: usize = 500; // units each side spawns with (tunable live)

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

/// Terrain height at a world (x,z): two octaves of hills plus a central volcano
/// cone. Deterministic and cheap — called per terrain tile and per unit step.
fn terrain_height(x: f64, z: f64) -> f64 {
    let s = 1.0 / 150.0;
    let mut h = vnoise(x * s, z * s) * 45.0 + vnoise(x * s * 2.7, z * s * 2.7) * 16.0;
    // Volcano: a cone rising near the centre, with a crater dip at the very top.
    let (cx, cz) = (WORLD * 0.5, WORLD * 0.5);
    let d = ((x - cx) * (x - cx) + (z - cz) * (z - cz)).sqrt();
    if d < 170.0 {
        let cone = (170.0 - d) * 0.62;
        h += cone;
        if d < 28.0 { h -= (28.0 - d) * 1.2; } // crater
    }
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
    // Lava river: a narrow azimuth wedge flowing down one slope of the cone.
    let mut dang = (pz.atan2(px) - (-2.1)).abs();
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
    /// Ground units sit on the terrain; the dragon flies at a fixed altitude.
    fn altitude(self) -> f64 { match self { Kind::Dragon => 95.0, _ => 0.0 } }
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
    fx: Vec<Fx>, // visible effects this unit produced this frame
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

// ----------------------------------------------------------------- spawning

fn spawn_unit(rng: &mut Rng, faction: Faction) -> Unit {
    // Roster mix: mostly foot soldiers, then archers/knights, a few siege engines
    // and mages, a rare dragon.
    let roll = rng.unit();
    let kind = if roll < 0.40 { Kind::Soldier }
        else if roll < 0.58 { Kind::Archer }
        else if roll < 0.70 { Kind::Knight }
        else if roll < 0.80 { Kind::Mage }
        else if roll < 0.88 { Kind::Healer }
        else if roll < 0.93 { Kind::Ballista }
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
        fx: Vec::new(),
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

fn spawn_army(rng: &mut Rng) -> Vec<Unit> {
    let mut units = Vec::with_capacity(PER_FACTION * 2);
    for _ in 0..PER_FACTION { units.push(spawn_unit(rng, Faction::Red)); }
    for _ in 0..PER_FACTION { units.push(spawn_unit(rng, Faction::Blue)); }
    units
}

// ----------------------------------------------------------------- AI: decide

const SEP_RADIUS: f64 = 11.0; // boids: friends closer than this push apart

/// One unit's per-frame brain — *read-only* on the shared index, writes only
/// into `u`'s own `vel`/`attacks`. This is the body that fans out over rayon
/// (`par_iter_mut`): each unit reads the shared index and mutates only itself.
///
/// Three library queries, one per concern: **k-NN** finds the nearest enemy
/// (targeting) *and* the nearby friends (boids); the dragon's AoE is a sphere
/// **`cull`**; the archer's line-of-fire is a thick **`raycast`**.
fn decide(u: &mut Unit, id: u32, index: &Tree3<IUnit>, smoke: &Tree3<Puff>) {
    u.vel = (0.0, 0.0, 0.0);
    u.attacks.clear();
    u.emit = None;
    u.fx.clear();
    if !u.alive() { return; }

    // One k-NN pass yields both the nearest enemy (targeting) and the nearby
    // friends used for flocking (separation + cohesion). k=16 reliably spans
    // both once the lines meet.
    let mut target: Option<(Point3, u32, f64)> = None; // nearest enemy (pos, id, dist)
    let mut heal: Option<(Point3, u32, f32, f64)> = None; // most-wounded friend (pos, id, health, dist)
    let (mut sep_x, mut sep_z) = (0.0, 0.0); // separation: away from close friends
    let (mut coh_x, mut coh_z, mut friends) = (0.0, 0.0, 0u32); // cohesion centroid
    for (d, it) in index.knn(u.p, 16) {
        if it.id == id { continue; }
        if it.faction != u.faction {
            if target.is_none() { target = Some((it.p, it.id, d)); }
        } else {
            if d < SEP_RADIUS { let dd = d.max(1e-3); sep_x += (u.p.x - it.p.x) / dd; sep_z += (u.p.z - it.p.z) / dd; }
            coh_x += it.p.x; coh_z += it.p.z; friends += 1;
            if it.health < 0.97 && heal.is_none_or(|(_, _, h, _)| it.health < h) { heal = Some((it.p, it.id, it.health, d)); }
        }
    }

    // The healer seeks its most-wounded comrade (or, with none, drifts with the
    // band); everyone else seeks the nearest enemy (or marches on the keep).
    let (tx, ty, tz, tdist) = if u.kind == Kind::Healer {
        match heal {
            Some((p, _, _, d)) => (p.x, p.y, p.z, d),
            None if friends > 0 => (coh_x / friends as f64, u.p.y, coh_z / friends as f64, f64::INFINITY),
            None => { let (cx, cz) = u.faction.other().castle(); (cx, u.p.y, cz, f64::INFINITY) }
        }
    } else {
        match target {
            Some((tp, _, d)) => (tp.x, tp.y, tp.z, d),
            None => { let (cx, cz) = u.faction.other().castle(); (cx, u.p.y, cz, f64::INFINITY) }
        }
    };

    // Steering in direction space: seek the target, then (for ground melee that
    // fight shoulder-to-shoulder) add boids separation + cohesion so they hold a
    // loose formation instead of stacking. Normalise, then scale by speed.
    let seek = (tx - u.p.x, tz - u.p.z);
    let slen = (seek.0 * seek.0 + seek.1 * seek.1).sqrt().max(1e-6);
    let (mut dx, mut dz) = (seek.0 / slen, seek.1 / slen);
    if matches!(u.kind, Kind::Soldier | Kind::Knight) && friends > 0 {
        dx += sep_x * 0.5; dz += sep_z * 0.5; // push out of crowding
        let (cx, cz) = (coh_x / friends as f64 - u.p.x, coh_z / friends as f64 - u.p.z);
        let cl = (cx * cx + cz * cz).sqrt().max(1e-6);
        dx += cx / cl * 0.25; dz += cz / cl * 0.25; // drift toward the band's centre
    }
    let dlen = (dx * dx + dz * dz).sqrt().max(1e-6);
    let dy = if u.kind == Kind::Dragon { (ty - u.p.y).clamp(-1.0, 1.0) } else { 0.0 };
    let speed = u.kind.speed();
    let approach = if tdist < u.kind.reach() * 0.8 { 0.0 } else { speed };
    u.vel = (dx / dlen * approach, dy * approach, dz / dlen * approach);

    // Attacking: only when in reach and off cooldown.
    if u.cooldown > 0.0 || tdist > u.kind.reach() { return; }
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
        // Catapult: a lobbed boulder — a wide `Sphere3` AoE cull at the target
        // spot (ground siege analogue of the dragon), kicking up smoke on impact.
        Kind::Catapult => {
            let blast = Sphere3::new(tx, ty, tz, 26.0);
            for it in index.cull(&blast) {
                if it.faction != u.faction { u.attacks.push((it.id, u.kind.dmg())); }
            }
            u.emit = Some(Point3::new(tx, ty, tz));
            u.fx.push(Fx::new(FxKind::Ring, v3(Point3::new(tx, ty, tz)), v3(Point3::new(tx, ty, tz))));
        }
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
fn apply(units: &mut [Unit], smoke: &mut Vec<Puff>, effects: &mut Vec<Fx>, rng: &mut Rng, dt: f64, now: f64) {
    // 1) movement + cooldown tick (each unit, independent).
    for u in units.iter_mut() {
        if !u.alive() { continue; }
        u.cooldown = (u.cooldown - dt).max(0.0);
        let nx = (u.p.x + u.vel.0 * dt).clamp(2.0, WORLD - 2.0);
        let nz = (u.p.z + u.vel.2 * dt).clamp(2.0, WORLD - 2.0);
        let ground = terrain_height(nx, nz) + u.kind.radius() as f64;
        let ny = if u.kind == Kind::Dragon { (terrain_height(nx, nz) + u.kind.altitude()).max(ground) } else { ground };
        u.p = Point3::new(nx, ny, nz);
        // Reload after a shot (an AoE that caught nobody still fired).
        if !u.attacks.is_empty() || u.emit.is_some() { u.cooldown = u.kind.cooldown(); }
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
}

// ----------------------------------------------------------------- rendering

fn faction_color(f: Faction, k: Kind) -> [f32; 4] {
    let base = match f { Faction::Red => [0.85, 0.22, 0.18], Faction::Blue => [0.22, 0.40, 0.90] };
    match k {
        Kind::Dragon => [base[0] * 0.6 + 0.3, base[1] * 0.6 + 0.1, base[2] * 0.6, 1.0],
        Kind::Knight => [base[0] * 0.8 + 0.15, base[1] * 0.8 + 0.15, base[2] * 0.8 + 0.15, 1.0],
        Kind::Archer => [base[0] * 0.7, base[1] * 0.7 + 0.2, base[2] * 0.7, 1.0],
        Kind::Catapult => [base[0] * 0.5 + 0.2, base[1] * 0.5 + 0.12, base[2] * 0.4, 1.0],
        Kind::Ballista => [base[0] * 0.55 + 0.18, base[1] * 0.5 + 0.18, base[2] * 0.45, 1.0],
        Kind::Mage => [base[0] * 0.5 + 0.25, base[1] * 0.5 + 0.25, base[2] * 0.5 + 0.45, 1.0],
        Kind::Healer => [0.85, 0.92, 0.80, 1.0], // pale, faction-neutral white-green
        Kind::Soldier => [base[0], base[1], base[2], 1.0],
    }
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

fn draw_castles() {
    for f in [Faction::Red, Faction::Blue] {
        let (cx, cz) = f.castle();
        let h = terrain_height(cx, cz);
        let col = match f { Faction::Red => Color::new(0.6, 0.18, 0.16, 1.0), Faction::Blue => Color::new(0.16, 0.28, 0.62, 1.0) };
        // Keep + four corner towers.
        draw_cube(vec3(cx as f32, (h + 22.0) as f32, cz as f32), vec3(46.0, 44.0, 46.0), None, col);
        for (ox, oz) in [(-26.0, -26.0), (26.0, -26.0), (-26.0, 26.0), (26.0, 26.0)] {
            draw_cube(vec3((cx + ox) as f32, (h + 30.0) as f32, (cz + oz) as f32), vec3(14.0, 60.0, 14.0), None, col);
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

/// A minimal screen-space slider for the live thread count (native only — wasm
/// has no threads). Draggable handle; updates `*value` in `1..=max` and sets
/// `*dragging` so the caller can suppress camera-orbit while the slider is held.
#[cfg(not(target_arch = "wasm32"))]
fn thread_slider(x: f32, y: f32, w: f32, value: &mut usize, max: usize, dragging: &mut bool) {
    draw_rectangle(x, y - 3.0, w, 6.0, Color::new(0.30, 0.30, 0.36, 1.0));
    let t = if max > 1 { (*value - 1) as f32 / (max - 1) as f32 } else { 0.0 };
    let hx = x + t * w;
    draw_circle(hx, y, 9.0, WHITE);
    let (mx, my) = mouse_position();
    let over = mx >= x - 12.0 && mx <= x + w + 12.0 && (my - y).abs() < 16.0;
    if is_mouse_button_pressed(MouseButton::Left) && over { *dragging = true; }
    if !is_mouse_button_down(MouseButton::Left) { *dragging = false; }
    if *dragging && max > 1 {
        let nt = ((mx - x) / w).clamp(0.0, 1.0);
        *value = (1.0 + nt * (max - 1) as f32).round() as usize;
    }
    draw_text(format!("threads: {value} / {max}"), x, y - 14.0, 20.0, WHITE);
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
    let mut rng = Rng::new(0x5_1E6E);
    let mut units = spawn_army(&mut rng);
    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD);
    let mut index = Tree3::<IUnit>::new(world, 8);
    // Smoke lives in its own index so archer/ballista shots can raycast it.
    let mut smoke: Vec<Puff> = Vec::new();
    let mut effects: Vec<Fx> = Vec::new(); // transient combat visuals
    let mut smoke_index = Tree3::<Puff>::new(world, 8);
    let mut smoke_instances: Vec<Instance> = Vec::new();

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

    let terrain_chunks = build_terrain_chunks(); // static — built once, drawn each frame
    let mut now = 0.0f64; // simulation clock
    let mut paused = false;
    let mut instances: Vec<Instance> = Vec::with_capacity(units.len());

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
        if is_mouse_button_down(MouseButton::Left) && !slider_drag {
            yaw += (mp.0 - last_mouse.0) * 0.01;
            pitch = (pitch + (mp.1 - last_mouse.1) * 0.01).clamp(0.05, 1.50);
        }
        last_mouse = mp;
        let wheel = mouse_wheel().1;
        if wheel != 0.0 { dist = (dist - wheel * 0.5).clamp(200.0, 1600.0); }
        if is_key_pressed(KeyCode::P) { paused = !paused; }
        if is_key_pressed(KeyCode::RightBracket) || is_key_pressed(KeyCode::LeftBracket) {
            // (live army resize hook — rebuild from the current seed; placeholder
            // until a population slider lands)
            units = spawn_army(&mut rng);
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
                let (idx, smk) = (&index, &smoke_index);
                pool.install(|| units.par_iter_mut().enumerate().for_each(|(i, u)| decide(u, i as u32, idx, smk)));
            }
            #[cfg(target_arch = "wasm32")]
            for i in 0..units.len() { decide(&mut units[i], i as u32, &index, &smoke_index); }
            apply(&mut units, &mut smoke, &mut effects, &mut rng, dt, now);
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
        let fwd = (observer - eye).normalize();
        let cam_right = fwd.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let cam_up = cam_right.cross(fwd).normalize();

        for m in &terrain_chunks { draw_mesh(m); }
        draw_castles();
        draw_effects(&effects, now);

        // Units as one instanced draw call.
        instances.clear();
        let (mut red, mut blue) = (0usize, 0usize);
        for u in units.iter() {
            if !u.alive() { continue; }
            match u.faction { Faction::Red => red += 1, Faction::Blue => blue += 1 }
            instances.push(Instance::new(
                vec3(u.p.x as f32, u.p.y as f32, u.p.z as f32),
                u.kind.radius(),
                faction_color(u.faction, u.kind),
            ));
        }
        // Smoke as a second instanced batch — grey puffs that grow as they age.
        smoke_instances.clear();
        for s in &smoke {
            let age = ((now - s.born) / SMOKE_LIFE).clamp(0.0, 1.0) as f32;
            let r = SMOKE_R as f32 * (0.55 + 0.45 * age);
            let g = 0.78 - 0.15 * age; // darkens slightly as it thins
            smoke_instances.push(Instance::new(vec3(s.p.x as f32, s.p.y as f32, s.p.z as f32), r, [g, g, g + 0.02, 0.6]));
        }
        {
            let gl = unsafe { get_internal_gl() };
            renderer.draw(gl.quad_context, Mode::Spheres, &instances, mvp, cam_right, cam_up);
            renderer.draw(gl.quad_context, Mode::Spheres, &smoke_instances, mvp, cam_right, cam_up);
        }

        // ----- HUD -----
        set_default_camera();
        draw_text("vectorial-hash — SIEGE", 16.0, 28.0, 30.0, WHITE);
        draw_text(format!("fps {}", get_fps()), 16.0, 54.0, 22.0, LIGHTGRAY);
        draw_text(format!("Red {red}"), 16.0, 80.0, 24.0, Color::new(0.95, 0.4, 0.35, 1.0));
        draw_text(format!("Blue {blue}"), 16.0, 104.0, 24.0, Color::new(0.45, 0.6, 1.0, 1.0));
        // Live thread-count slider drives the parallel AI pass (native only).
        #[cfg(not(target_arch = "wasm32"))]
        thread_slider(20.0, 150.0, 220.0, &mut n_threads, max_threads, &mut slider_drag);
        draw_text(
            "drag: orbit  scroll: zoom  P: pause  [ ]: rebuild",
            16.0, screen_height() - 18.0, 20.0, LIGHTGRAY,
        );
        if paused { draw_text("PAUSED", screen_width() * 0.5 - 50.0, 40.0, 36.0, YELLOW); }

        next_frame().await;
        frame_no += 1;
        if let Some(m) = max_frames { if frame_no >= m { break; } }
    }
}
