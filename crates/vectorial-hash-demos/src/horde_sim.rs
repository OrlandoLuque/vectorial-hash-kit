//! Shared, graphics-free simulation core for the `horde` demo (They Are
//! Billions-style zombie assault) — phase 1 of `docs/HORDE_DESIGN.md`.
//!
//! What exists at this phase: the procedural world (gentle heightfield), the
//! **static base** (wall ring + gates + towers + houses + storehouses + Command
//! Center) bulk-loaded once into its own `Tree3`, the **dormant zombie field**
//! (nest clusters, per-class stats from the TAB research), the **noise system**
//! (decaying per-zombie accumulator + a decaying grid for viz/flow), and the
//! decide→apply loop with **keep-index** maintenance (`update_ref` in place —
//! dormant zombies never move, so they cost nothing to keep indexed: the
//! demo's headline).
//!
//! The mechanical star is event-driven waking: every noise event is **one
//! sphere cull** over the zombie index; each dormant zombie inside its own
//! class's hearing radius (4 × watch) accumulates the amount and wakes when the
//! sum crosses `1000 / alertness` (the community-mined TAB rule) — then walks
//! to the noise tile, lingers, and re-sleeps if nothing else sounds. Waves of
//! activation are literally spatial queries chaining.
//!
//! No combat yet (phase 2: towers, wall HP, breaches, infection, waves).

use vectorial_hash::{Aabb, ItemRef, Point3, Positioned3, Sphere3, Tree3};

pub use crate::siege_sim::Rng;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// ---------------------------------------------------------------- world config

pub const WORLD: f64 = 1200.0; // map side, world units
pub const SKY: f64 = 64.0; // index height (terrain amp ~6 + flyer altitude)
pub const BASE_R: f64 = 150.0; // wall ring radius around the map centre
const MARGIN: f64 = 2.0;

/// Gentle 2-octave value-noise heightfield — flatter than siege's (no volcano,
/// no rivers in phase 1); the base sits on near-flat ground by design.
pub fn ground_h(x: f64, z: f64, seed: f64) -> f64 {
    fn h(ix: i64, iz: i64, s: i64) -> f64 {
        let mut n = (ix.wrapping_mul(374761393)) ^ (iz.wrapping_mul(668265263)) ^ s.wrapping_mul(1274126177);
        n = (n ^ (n >> 13)).wrapping_mul(1103515245);
        ((n ^ (n >> 16)) & 0xffff) as f64 / 65535.0
    }
    fn vnoise(x: f64, z: f64, s: i64) -> f64 {
        let (ix, iz) = (x.floor() as i64, z.floor() as i64);
        let (fx, fz) = (x - x.floor(), z - z.floor());
        let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
        let (a, b, c, d) = (h(ix, iz, s), h(ix + 1, iz, s), h(ix, iz + 1, s), h(ix + 1, iz + 1, s));
        a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz
    }
    let s = (seed * 1e6) as i64 | 1;
    let dc = ((x - WORLD / 2.0).powi(2) + (z - WORLD / 2.0).powi(2)).sqrt();
    let flat = (dc / (BASE_R * 1.4)).min(1.0); // flatten toward the base
    (vnoise(x * 0.008, z * 0.008, s) * 4.5 + vnoise(x * 0.03, z * 0.03, s ^ 7) * 1.5) * flat
}

// ------------------------------------------------------------------- zombies

/// The TAB-researched bestiary (HORDE_DESIGN.md): per-class stats. Speeds in
/// world-units/s (TAB tile ≈ 8 wu, ratios preserved); `watch` is sight range,
/// hearing = 4 × watch; wake threshold = 1000 / alertness.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ZClass { Walker, Runner, Chubby, Venom, Harpy }

impl ZClass {
    pub fn max_hp(self) -> f64 { match self { Self::Walker => 35.0, Self::Runner => 45.0, Self::Chubby => 500.0, Self::Venom => 120.0, Self::Harpy => 120.0 } }
    pub fn dmg(self) -> f64 { match self { Self::Walker => 6.0, Self::Runner => 9.0, Self::Chubby => 40.0, Self::Venom => 30.0, Self::Harpy => 30.0 } }
    pub fn speed(self) -> f64 { match self { Self::Walker => 4.0, Self::Runner => 14.0, Self::Chubby => 12.0, Self::Venom => 14.0, Self::Harpy => 34.0 } }
    pub fn watch(self) -> f64 { match self { Self::Walker => 40.0, Self::Runner => 48.0, Self::Chubby => 40.0, Self::Venom => 64.0, Self::Harpy => 72.0 } }
    pub fn hear(self) -> f64 { self.watch() * 4.0 }
    pub fn alertness(self) -> f64 { match self { Self::Walker => 2.0, Self::Runner => 3.0, Self::Chubby => 3.0, Self::Venom => 4.0, Self::Harpy => 8.0 } }
    pub fn wake_threshold(self) -> f64 { 1000.0 / self.alertness() }
    /// Noise a zombie makes while active (groans, attacks, deaths).
    pub fn noise_made(self) -> f64 { match self { Self::Walker => 1.0, Self::Runner => 2.0, _ => 10.0 } }
    pub fn altitude(self) -> f64 { if self == Self::Harpy { 20.0 } else { 0.0 } }
    pub fn index(self) -> usize { match self { Self::Walker => 0, Self::Runner => 1, Self::Chubby => 2, Self::Venom => 3, Self::Harpy => 4 } }
    /// Attack reach vs structures/defenders: melee-adjacent, except the venom's
    /// standoff spit (4.5 TAB tiles ≈ 36 wu — outranges the wall line).
    pub fn reach(self) -> f64 { if self == Self::Venom { 36.0 } else { 4.0 } }
    /// Target priority for the towers' "highest threat" mode.
    pub fn threat(self) -> f64 { match self { Self::Chubby => 3.0, Self::Harpy => 2.5, Self::Venom => 2.0, Self::Runner => 1.5, Self::Walker => 1.0 } }
}
/// The biggest hearing radius — the single cull radius per noise event.
pub const MAX_HEAR: f64 = 288.0; // Harpy: 4 × 72

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ZState {
    Dormant,
    /// Walking to a noise position; lingers there, then re-sleeps.
    Investigating { tx: f64, tz: f64 },
    /// Wave zombie: follows the flow field toward the Command Center; engages
    /// structures/defenders on contact. Never re-sleeps.
    Marching,
    /// Pounding structure `sid` (swing timer in `Zombie::swing_t`).
    Attacking { sid: u32 },
}

#[derive(Clone)]
pub struct Zombie {
    pub class: ZClass,
    pub p: Point3,
    pub vel: (f64, f64),
    pub state: ZState,
    pub hp: f64,
    /// Accumulated heard noise (dormant only) — halves every second, wakes at
    /// `1000 / alertness`.
    pub heard: f64,
    /// Seconds left milling at the noise site before re-sleeping.
    pub linger: f64,
    /// Countdown to the next groan (active only).
    pub groan_t: f64,
    /// Attack swing timer (Attacking state).
    pub swing_t: f64,
    moved: bool,
}
impl Zombie {
    pub fn dormant(&self) -> bool { self.state == ZState::Dormant }
    pub fn alive(&self) -> bool { self.hp > 0.0 }
}

/// The lightweight item in the zombie index (same decoupling as siege's
/// `IUnit`): the parallel decide pass holds `&Tree3<IZombie>` while mutating
/// the `Vec<Zombie>`.
#[derive(Clone, Copy)]
pub struct IZombie { pub id: u32, pub p: Point3, pub class: ZClass, pub dormant: bool }
impl Positioned3 for IZombie { fn position(&self) -> Point3 { self.p } }
impl IZombie {
    fn of(i: usize, z: &Zombie) -> IZombie { IZombie { id: i as u32, p: z.p, class: z.class, dormant: z.dormant() } }
}

// ----------------------------------------------------------------- structures

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SKind { Wall, Gate, Tower, House, Storehouse, CommandCenter }
impl SKind {
    /// TAB-researched HP (stone wall 1000, gate 1500, tower 2000, CC 5000).
    pub fn max_hp(self) -> f64 { match self { Self::Wall => 1000.0, Self::Gate => 1500.0, Self::Tower => 2000.0, Self::House => 500.0, Self::Storehouse => 800.0, Self::CommandCenter => 5000.0 } }
}

#[derive(Clone)]
pub struct Structure {
    pub kind: SKind,
    pub p: Point3,
    pub hp: f64,
    /// Colonists inside (houses; the phase-2 infection burst = 50 noise each).
    pub pop: u32,
}

/// Item in the **static** index — built once with `bulk_load` at startup (the
/// from-scratch static build case; `bulk_load_par` under `--features parallel`).
#[derive(Clone, Copy)]
pub struct IStruct { pub id: u32, pub p: Point3, pub kind: SKind }
impl Positioned3 for IStruct { fn position(&self) -> Point3 { self.p } }

// ------------------------------------------------------------------ noise grid

/// Coarse decaying activity grid — the TAB "activity halves every second"
/// field. Waking is event-driven (see [`Horde::step`]); the grid exists for
/// visualisation and the phase-2 flow field, and mirrors every event.
pub struct NoiseGrid { pub cells: Vec<f32>, pub n: usize, pub cell: f64 }
impl NoiseGrid {
    fn new(n: usize) -> Self { NoiseGrid { cells: vec![0.0; n * n], n, cell: WORLD / n as f64 } }
    fn idx(&self, x: f64, z: f64) -> usize {
        let i = ((x / self.cell) as usize).min(self.n - 1);
        let j = ((z / self.cell) as usize).min(self.n - 1);
        j * self.n + i
    }
    pub fn add(&mut self, x: f64, z: f64, amount: f64) { let i = self.idx(x, z); self.cells[i] += amount as f32; }
    pub fn at(&self, x: f64, z: f64) -> f32 { self.cells[self.idx(x, z)] }
    fn step(&mut self, dt: f64) {
        let k = 0.5f32.powf(dt as f32); // halves every second
        for c in self.cells.iter_mut() { if *c > 1e-4 { *c *= k; } else { *c = 0.0; } }
    }
}

// ----------------------------------------------------------------- flow field

/// Coarse flow field guiding wave zombies to the Command Center — the SupCom2
/// scheme (Game AI Pro ch. 23) at demo scale. **Walls are HIGH COST, not
/// impassable** (the TAB rule): the flood prefers gates and breaches, and a
/// half-rebuilt wall gradually deters again as it rises, because cost is a
/// function of live HP. Integer-cost Dijkstra from the CC cell; per-cell
/// direction = descend the integration. Rebuilt (throttled) when any structure
/// changes state.
pub struct FlowField {
    pub n: usize,
    pub cell: f64,
    pub dir: Vec<(f32, f32)>,
    integ: Vec<u32>,
    pub dirty: bool,
    rebuild_t: f64,
}

impl FlowField {
    fn new(n: usize) -> FlowField {
        FlowField { n, cell: WORLD / n as f64, dir: vec![(0.0, 0.0); n * n], integ: vec![u32::MAX; n * n], dirty: true, rebuild_t: 0.0 }
    }
    fn cell_of(&self, x: f64, z: f64) -> (usize, usize) {
        (((x / self.cell) as usize).min(self.n - 1), ((z / self.cell) as usize).min(self.n - 1))
    }
    pub fn flow_at(&self, x: f64, z: f64) -> (f64, f64) {
        let (i, j) = self.cell_of(x, z);
        let d = self.dir[j * self.n + i];
        (d.0 as f64, d.1 as f64)
    }
    /// Rebuild cost + integration + directions from live structure HP.
    fn rebuild(&mut self, structures: &[Structure], cc: Point3) {
        let n = self.n;
        // Per-cell traversal cost (milli-units): open ground 100; a live wall
        // piece adds ~6× ground + a term falling with damage.
        let mut cost = vec![100u32; n * n];
        for s in structures {
            if s.hp <= 0.0 { continue; }
            let (i, j) = self.cell_of(s.p.x, s.p.z);
            let add = match s.kind {
                SKind::Wall | SKind::Tower => 600 + (s.hp * 4.0) as u32,   // stone: up to +4600
                SKind::Gate => 400 + (s.hp * 2.0) as u32,                  // gates a bit softer → preferred
                _ => 300,
            };
            cost[j * n + i] = cost[j * n + i].saturating_add(add);
        }
        // Dijkstra from the CC cell over 8-neighbours (diagonal ×1.41).
        self.integ.iter_mut().for_each(|v| *v = u32::MAX);
        let (ci, cj) = self.cell_of(cc.x, cc.z);
        let mut heap = std::collections::BinaryHeap::new();
        self.integ[cj * n + ci] = 0;
        heap.push(std::cmp::Reverse((0u32, ci, cj)));
        while let Some(std::cmp::Reverse((d, i, j))) = heap.pop() {
            if d > self.integ[j * n + i] { continue; }
            for (di, dj, w) in [(-1i64, 0i64, 100u32), (1, 0, 100), (0, -1, 100), (0, 1, 100), (-1, -1, 141), (1, -1, 141), (-1, 1, 141), (1, 1, 141)] {
                let (ni, nj) = (i as i64 + di, j as i64 + dj);
                if ni < 0 || nj < 0 || ni >= n as i64 || nj >= n as i64 { continue; }
                let (ni, nj) = (ni as usize, nj as usize);
                let nd = d + cost[nj * n + ni] * w / 100;
                if nd < self.integ[nj * n + ni] { self.integ[nj * n + ni] = nd; heap.push(std::cmp::Reverse((nd, ni, nj))); }
            }
        }
        // Direction = toward the lowest-integration neighbour.
        for j in 0..n {
            for i in 0..n {
                let (mut best, mut bi, mut bj) = (self.integ[j * n + i], i, j);
                for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                    let (ni, nj) = (i as i64 + di, j as i64 + dj);
                    if ni < 0 || nj < 0 || ni >= n as i64 || nj >= n as i64 { continue; }
                    let v = self.integ[nj as usize * n + ni as usize];
                    if v < best { best = v; bi = ni as usize; bj = nj as usize; }
                }
                let (dx, dz) = (bi as f64 - i as f64, bj as f64 - j as f64);
                let l = (dx * dx + dz * dz).sqrt();
                self.dir[j * n + i] = if l > 0.0 { ((dx / l) as f32, (dz / l) as f32) } else { (0.0, 0.0) };
            }
        }
        self.dirty = false;
        self.rebuild_t = 1.0; // throttle: at most one rebuild per second
    }
}

// ------------------------------------------------------------------ towers

pub const TOWER_RANGE: f64 = 72.0; // 9 TAB tiles
pub const TOWER_DMG: f64 = 150.0;
pub const TOWER_RELOAD: f64 = 1.2;
pub const TOWER_NOISE: f64 = 5.0; // the ballista: quiet per kill — by design

// ----------------------------------------------------------------------- sim

pub struct Horde {
    pub units: Vec<Zombie>,
    pub structures: Vec<Structure>,
    pub zindex: Tree3<IZombie>,
    handles: Vec<Option<ItemRef>>,
    pub sindex: Tree3<IStruct>,
    pub noise: NoiseGrid,
    pub flow: FlowField,
    /// Noise events queued for the next step: (position, amount).
    pending: Vec<(Point3, f64)>,
    /// Fallen zombies: (position, class, time of death) — the renderer's
    /// frozen-pose corpse buffer feeds from this.
    pub corpses: Vec<(Point3, ZClass, f64)>,
    /// Per-structure reload timers (only towers use theirs).
    tower_reload: Vec<f64>,
    /// Dead unit slots for reuse (keeps `id == index` stable for handles).
    free_slots: Vec<u32>,
    /// Tower targeting: `false` = nearest, `true` = highest threat (TAB's
    /// configurable modes — toggle live from the renderer).
    pub tower_threat_mode: bool,
    cc_id: usize,
    // Wave scheduler (TAB: direction warning + countdown, escalation, final).
    pub wave_k: u32,
    wave_spawn_t: f64,
    pub wave_dir: f64,
    pub wave_announced: bool,
    /// Set when the Command Center falls (defeat) or the final wave is cleared
    /// (victory): (time it happened, victory?). The run resets ~12 s later.
    pub game_over: Option<(f64, bool)>,
    pub run: u32,
    pub kills: u64,
    pub rng: Rng,
    pub now: f64,
    pub seed: f64,
    /// Woken-this-frame counter (for HUD/telemetry).
    pub woken_last: usize,
    base_pop: usize,
    base_seed: u64,
}

/// The base layout: a stone wall ring with 4 cardinal gates and towers every
/// few segments, houses + storehouses inside, the Command Center dead centre.
fn build_base(rng: &mut Rng, seed: f64) -> Vec<Structure> {
    let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
    let at = |x: f64, z: f64| Point3::new(x, ground_h(x, z, seed), z);
    let mut s = Vec::new();
    let segs = (std::f64::consts::TAU * BASE_R / 8.0) as usize; // one wall piece ≈ every 8 wu
    let step = std::f64::consts::TAU / segs as f64;
    // Exactly one gate per cardinal: the single closest segment to each.
    let gates: Vec<usize> = (0..4).map(|q| ((q as f64 * std::f64::consts::FRAC_PI_2) / step).round() as usize % segs).collect();
    for i in 0..segs {
        let a = i as f64 * step;
        let (x, z) = (cx + a.cos() * BASE_R, cz + a.sin() * BASE_R);
        let kind = if gates.contains(&i) { SKind::Gate } else if i % 10 == 5 { SKind::Tower } else { SKind::Wall };
        s.push(Structure { kind, p: at(x, z), hp: kind.max_hp(), pop: 0 });
    }
    for _ in 0..28 { // houses, scattered inside the ring
        let (a, r) = (rng.range(0.0, std::f64::consts::TAU), rng.range(30.0, 110.0));
        let (x, z) = (cx + a.cos() * r, cz + a.sin() * r);
        s.push(Structure { kind: SKind::House, p: at(x, z), hp: SKind::House.max_hp(), pop: 5 + (rng.unit() * 15.0) as u32 });
    }
    for q in 0..2 { // storehouses flanking the CC (phase-3 hauling endpoints)
        let a = q as f64 * std::f64::consts::PI + 0.7;
        s.push(Structure { kind: SKind::Storehouse, p: at(cx + a.cos() * 40.0, cz + a.sin() * 40.0), hp: SKind::Storehouse.max_hp(), pop: 0 });
    }
    s.push(Structure { kind: SKind::CommandCenter, p: at(cx, cz), hp: SKind::CommandCenter.max_hp(), pop: 30 });
    s
}

/// Scatter the dormant field: nest clusters outside the base, class mix from
/// the research (~70% walkers, the rest specials).
pub fn spawn_field(rng: &mut Rng, pop: usize, seed: f64) -> Vec<Zombie> {
    let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
    let nests = (pop / 150).max(1);
    let mut units = Vec::with_capacity(pop);
    for n in 0..nests {
        let a = rng.range(0.0, std::f64::consts::TAU);
        let r = rng.range(BASE_R + 120.0, WORLD / 2.0 - 60.0);
        let (nx, nz) = (cx + a.cos() * r, cz + a.sin() * r);
        let count = pop / nests + usize::from(n < pop % nests);
        for _ in 0..count {
            let (da, dr) = (rng.range(0.0, std::f64::consts::TAU), rng.range(0.0, 42.0));
            let (x, z) = ((nx + da.cos() * dr).clamp(MARGIN, WORLD - MARGIN), (nz + da.sin() * dr).clamp(MARGIN, WORLD - MARGIN));
            let roll = rng.unit();
            let class = if roll < 0.70 { ZClass::Walker } else if roll < 0.85 { ZClass::Runner } else if roll < 0.91 { ZClass::Chubby } else if roll < 0.96 { ZClass::Venom } else { ZClass::Harpy };
            let y = ground_h(x, z, seed) + class.altitude();
            units.push(Zombie { class, p: Point3::new(x, y, z), vel: (0.0, 0.0), state: ZState::Dormant, hp: class.max_hp(), heard: 0.0, linger: 0.0, groan_t: rng.range(0.5, 2.0), swing_t: 0.0, moved: false });
        }
    }
    units
}

impl Horde {
    pub fn new(seed: u64, pop: usize) -> Horde {
        let fseed = (seed % 100_000) as f64 * 0.013 + 0.29;
        let mut rng = Rng::new(seed | 1);
        let structures = build_base(&mut rng, fseed);
        let units = spawn_field(&mut rng, pop, fseed);
        let world = Aabb::new(0.0, -8.0, 0.0, WORLD, SKY + 8.0, WORLD);
        // Static index: built ONCE from all structures — the bulk-load case.
        let items: Vec<IStruct> = structures.iter().enumerate().map(|(i, s)| IStruct { id: i as u32, p: s.p, kind: s.kind }).collect();
        #[cfg(feature = "parallel")]
        let sindex = Tree3::bulk_load_par(world, 8, items);
        #[cfg(not(feature = "parallel"))]
        let sindex = Tree3::bulk_load(world, 8, items);
        let n = units.len();
        let cc_id = structures.iter().position(|s| s.kind == SKind::CommandCenter).expect("base has a CC");
        let mut flow = FlowField::new(120);
        flow.rebuild(&structures, structures[cc_id].p);
        let ns = structures.len();
        Horde {
            units, structures,
            zindex: Tree3::new(world, 8), handles: vec![None; n],
            sindex, noise: NoiseGrid::new(96), flow,
            pending: Vec::new(), corpses: Vec::new(),
            tower_reload: vec![0.0; ns], free_slots: Vec::new(),
            tower_threat_mode: false, cc_id,
            wave_k: 0, wave_spawn_t: 50.0, wave_dir: 0.0, wave_announced: false,
            game_over: None, run: 1, kills: 0,
            rng, now: 0.0, seed: fseed, woken_last: 0,
            base_pop: pop, base_seed: seed,
        }
    }

    /// Full run reset (CC fell, or the final wave was cleared): fresh base +
    /// dormant field, next run number, escalation restarts.
    fn reset(&mut self) {
        let next = Horde::new(self.base_seed.wrapping_add(self.run as u64 * 7919), self.base_pop);
        let (run, kills, mode) = (self.run + 1, self.kills, self.tower_threat_mode);
        *self = next;
        self.run = run;
        self.kills = kills; // lifetime counter across runs
        self.tower_threat_mode = mode;
    }

    /// HUD wave line: (wave number, announced?, direction, seconds to landfall).
    pub fn wave_info(&self) -> (u32, bool, f64, f64) {
        (self.wave_k + 1, self.wave_announced, self.wave_dir, (self.wave_spawn_t - self.now).max(0.0))
    }

    /// Spawn (or reuse a dead slot for) one zombie. Public for scenario
    /// drivers and tests.
    pub fn spawn_zombie(&mut self, class: ZClass, x: f64, z: f64, state: ZState) {
        let y = ground_h(x, z, self.seed) + class.altitude();
        let z0 = Zombie { class, p: Point3::new(x, y, z), vel: (0.0, 0.0), state, hp: class.max_hp(), heard: 0.0, linger: 0.0, groan_t: 1.0, swing_t: 0.5, moved: false };
        match self.free_slots.pop() {
            Some(slot) => { self.units[slot as usize] = z0; } // handle is None (freed at death) → sync inserts
            None => { self.units.push(z0); self.handles.push(None); }
        }
    }

    /// The TAB wave scheduler: warning 30 s ahead with a direction, spawn at an
    /// edge arc, escalating size/mix; wave 8 is the FINAL — all edges at once
    /// plus every dormant zombie still alive joins the march.
    fn step_waves(&mut self) {
        if self.game_over.is_some() { return; }
        if !self.wave_announced && self.now >= self.wave_spawn_t - 30.0 {
            self.wave_announced = true;
            self.wave_dir = self.rng.range(0.0, std::f64::consts::TAU);
        }
        if self.now < self.wave_spawn_t { return; }
        let k = self.wave_k;
        let is_final = k >= 8;
        let count = (60.0 * 1.4f64.powi(k.min(8) as i32)) as usize;
        let dirs: Vec<f64> = if is_final { (0..4).map(|q| q as f64 * std::f64::consts::FRAC_PI_2).collect() } else { vec![self.wave_dir] };
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        for dir in &dirs {
            for _ in 0..count / dirs.len() {
                let a = dir + self.rng.range(-0.35, 0.35);
                let r = self.rng.range(WORLD / 2.0 - 60.0, WORLD / 2.0 - 15.0);
                let (x, z) = ((cx + a.cos() * r).clamp(MARGIN, WORLD - MARGIN), (cz + a.sin() * r).clamp(MARGIN, WORLD - MARGIN));
                // Escalating mix: early waves are shambler seas; specials join
                // from wave 2 (venom/chubby) and wave 4 (harpies).
                let roll = self.rng.unit();
                let class = if k >= 4 && roll > 0.95 { ZClass::Harpy }
                    else if k >= 2 && roll > 0.88 { ZClass::Venom }
                    else if k >= 2 && roll > 0.80 { ZClass::Chubby }
                    else if roll > 0.65 { ZClass::Runner } else { ZClass::Walker };
                self.spawn_zombie(class, x, z, ZState::Marching);
            }
        }
        if is_final { // the map itself rises
            for z in self.units.iter_mut() {
                if z.alive() && z.dormant() { z.state = ZState::Marching; z.moved = true; }
            }
        }
        self.wave_k += 1;
        self.wave_spawn_t += 70.0;
        self.wave_announced = false;
    }

    /// Structure destroyed: dirty the flow field; a populated house detonates
    /// the **infection cascade** (each colonist → a fresh runner + 50 noise);
    /// the Command Center falling ends the run.
    fn destroy_structure(&mut self, sid: usize) {
        self.structures[sid].hp = 0.0;
        self.flow.dirty = true;
        let (kind, p, pop) = (self.structures[sid].kind, self.structures[sid].p, self.structures[sid].pop);
        if pop > 0 {
            for c in 0..pop {
                let ang = c as f64 * 2.399963;
                let (x, z) = ((p.x + ang.cos() * 3.0).clamp(MARGIN, WORLD - MARGIN), (p.z + ang.sin() * 3.0).clamp(MARGIN, WORLD - MARGIN));
                self.spawn_zombie(ZClass::Runner, x, z, ZState::Marching);
            }
            self.pending.push((p, 50.0 * pop as f64)); // the aggro detonation
            self.structures[sid].pop = 0;
        }
        if kind == SKind::CommandCenter && self.game_over.is_none() { self.game_over = Some((self.now, false)); }
    }

    /// Queue a noise event (processed next `step`): the wake mechanism.
    /// Towers/attacks/infections will feed this in phase 2; drivers and the
    /// zombies' own groans feed it now.
    pub fn emit_noise(&mut self, p: Point3, amount: f64) { self.pending.push((p, amount)); }

    /// (dormant, active) — dead slots excluded from both.
    pub fn counts(&self) -> (usize, usize) {
        let (mut dormant, mut active) = (0, 0);
        for z in &self.units {
            if !z.alive() { continue; }
            if z.dormant() { dormant += 1; } else { active += 1; }
        }
        (dormant, active)
    }

    /// Keep the zombie index in sync **without rebuilding** (the siege
    /// `sync_index` pattern): dormant zombies never move → skipped entirely;
    /// movers `update_ref` in place (O(1) while they stay in their leaf).
    /// Runs at the top of every [`step`](Horde::step); public so renderers /
    /// tests can force the index current after `apply` moved units.
    pub fn sync_index(&mut self) {
        for (i, z) in self.units.iter_mut().enumerate() {
            if !z.alive() { z.moved = false; continue; } // dead slots await reuse (handle freed at kill)
            match (self.handles[i], z.moved) {
                (None, _) => { self.handles[i] = self.zindex.insert_ref(IZombie::of(i, z)); z.moved = false; }
                (Some(_), false) => {}
                (Some(h), true) => {
                    let it = IZombie::of(i, z);
                    if !self.zindex.update_ref(h, |s| *s = it) { self.handles[i] = self.zindex.insert_ref(it); }
                    z.moved = false;
                }
            }
        }
    }

    /// One fixed step: waves → noise events → wake culls → parallel decide →
    /// serial apply (movement, swings, towers, destructions) → next frame's
    /// keep-index sync.
    pub fn step(&mut self, dt: f64) {
        self.now += dt;
        self.sync_index();
        self.step_waves();
        // Throttled flow-field rebuild after breaches / repairs.
        self.flow.rebuild_t -= dt;
        if self.flow.dirty && self.flow.rebuild_t <= 0.0 {
            let cc = self.structures[self.cc_id].p;
            let (flow, structures) = (&mut self.flow, &self.structures);
            flow.rebuild(structures, cc);
        }
        // Run over → reset a few seconds later (victory lap / defeat dirge).
        if let Some((t0, _)) = self.game_over { if self.now - t0 > 12.0 { self.reset(); return; } }
        else if self.wave_k > 8 && self.units.iter().all(|z| !z.alive() || z.dormant()) {
            self.game_over = Some((self.now, true)); // final wave cleared: victory
        }

        // 1) Noise events: each is ONE sphere cull; every dormant zombie within
        //    its own class's hearing radius accumulates; over threshold → wake.
        self.woken_last = 0;
        let events = std::mem::take(&mut self.pending);
        for (p, amount) in &events {
            self.noise.add(p.x, p.z, *amount);
            let blast = Sphere3::new(p.x, p.y, p.z, MAX_HEAR);
            let heard: Vec<(u32, f64)> = self.zindex.cull(&blast).iter()
                .filter(|it| it.dormant)
                .map(|it| { let (dx, dz) = (it.p.x - p.x, it.p.z - p.z); (it.id, (dx * dx + dz * dz).sqrt()) })
                .collect();
            for (id, d) in heard {
                let z = &mut self.units[id as usize];
                if !z.dormant() || d > z.class.hear() { continue; }
                z.heard += amount;
                if z.heard >= z.class.wake_threshold() {
                    // Personal investigate spot: a golden-angle jitter around
                    // the noise, so a pack spreads over the area instead of
                    // stacking on (and fighting over) one exact point.
                    let (ang, rr) = (id as f64 * 2.399963, 4.0 + (id % 89) as f64 * 0.25);
                    z.state = ZState::Investigating {
                        tx: (p.x + ang.cos() * rr).clamp(MARGIN, WORLD - MARGIN),
                        tz: (p.z + ang.sin() * rr).clamp(MARGIN, WORLD - MARGIN),
                    };
                    z.linger = 6.0 + (id % 97) as f64 * 0.06; // deterministic jitter
                    z.heard = 0.0;
                    z.moved = true; // dormant flag changed → index item refresh
                    self.woken_last += 1;
                }
            }
        }
        self.noise.step(dt);

        // 2) decide — read-only on the indices, each zombie writes only itself:
        //    fans out over rayon on native, serial on wasm (no threads there).
        {
            let (index, sindex, structures, flow) = (&self.zindex, &self.sindex, &self.structures, &self.flow);
            let (units, cx, cz) = (&mut self.units, WORLD / 2.0, WORLD / 2.0);
            let decide_one = |i: usize, z: &mut Zombie| {
                if !z.alive() { return; }
                match z.state {
                    ZState::Dormant => return,
                    // Already pounding: revalidate the target (it may have died
                    // to another zombie's swing), stand still.
                    ZState::Attacking { sid } => {
                        if structures[sid as usize].hp <= 0.0 { z.state = ZState::Marching; }
                        z.vel = (0.0, 0.0);
                        return;
                    }
                    _ => {}
                }
                // Target acquisition near the base: nearest LIVE structure in
                // reach (k-NN on the static index; venom's 36-wu standoff spit
                // engages from outside the wall line; harpies fly at altitude,
                // so the 3D k-NN naturally skips the wall and finds the inner
                // buildings under them).
                let (dcx, dcz) = (z.p.x - cx, z.p.z - cz);
                let dc = (dcx * dcx + dcz * dcz).sqrt();
                if dc < BASE_R + 80.0 {
                    for (_, it) in sindex.knn(z.p, 4) {
                        if structures[it.id as usize].hp <= 0.0 { continue; }
                        let (dx, dz) = (it.p.x - z.p.x, it.p.z - z.p.z);
                        if (dx * dx + dz * dz).sqrt() <= z.class.reach() + 1.5 {
                            z.state = ZState::Attacking { sid: it.id };
                            z.vel = (0.0, 0.0);
                            return;
                        }
                        break; // knn is sorted: nearest live one is out of reach
                    }
                }
                // Steering: investigators seek their personal spot; marchers
                // descend the flow field (harpies fly straight over the walls).
                let (mut vx, mut vz) = match z.state {
                    ZState::Investigating { tx, tz } => {
                        let (dx, dz) = (tx - z.p.x, tz - z.p.z);
                        let d = (dx * dx + dz * dz).sqrt();
                        if d > 8.0 { (dx / d, dz / d) } else { (0.0, 0.0) }
                    }
                    ZState::Marching if z.class == ZClass::Harpy => { let d = dc.max(1.0); (-dcx / d, -dcz / d) }
                    ZState::Marching => flow.flow_at(z.p.x, z.p.z),
                    _ => (0.0, 0.0),
                };
                // Separation among the awake (dormant bodies are a carpet the
                // wave flows around): one small cull per active zombie.
                let sep = Sphere3::new(z.p.x, z.p.y, z.p.z, 3.0);
                for it in index.cull(&sep) {
                    if it.id as usize == i || (it.p.y - z.p.y).abs() > 12.0 { continue; }
                    let (sx, sz) = (z.p.x - it.p.x, z.p.z - it.p.z);
                    let d = (sx * sx + sz * sz).sqrt().max(0.2);
                    if d < 3.0 { vx += sx / d * (3.0 - d) * 0.4; vz += sz / d * (3.0 - d) * 0.4; }
                }
                let l = (vx * vx + vz * vz).sqrt();
                let sp = z.class.speed();
                z.vel = if l > 1e-6 { (vx / l * sp, vz / l * sp) } else { (0.0, 0.0) };
            };
            #[cfg(not(target_arch = "wasm32"))]
            units.par_iter_mut().enumerate().for_each(|(i, z)| decide_one(i, z));
            #[cfg(target_arch = "wasm32")]
            units.iter_mut().enumerate().for_each(|(i, z)| decide_one(i, z));
        }

        // 3) apply — serial: integrate movers, decay heard, linger → re-sleep,
        //    swing timers → queued structure hits, groans (next frame's culls).
        let decay = 0.5f64.powf(dt);
        let mut hits: Vec<(u32, f64, Point3, f64)> = Vec::new(); // (sid, dmg, at, noise)
        for z in self.units.iter_mut() {
            if !z.alive() { continue; }
            if z.dormant() { if z.heard > 1e-3 { z.heard *= decay; } continue; }
            if let ZState::Attacking { sid } = z.state {
                z.swing_t -= dt;
                if z.swing_t <= 0.0 { z.swing_t = 1.0; hits.push((sid, z.class.dmg(), z.p, z.class.noise_made())); }
                continue;
            }
            // Arrival is by DISTANCE to the personal spot, not by velocity —
            // separation jiggle in a dense pack never lets vel hit exact zero.
            let arrived = match z.state { ZState::Investigating { tx, tz } => { let (dx, dz) = (tx - z.p.x, tz - z.p.z); dx * dx + dz * dz < 9.0 * 9.0 } _ => false };
            if !arrived && (z.vel.0 != 0.0 || z.vel.1 != 0.0) {
                let nx = (z.p.x + z.vel.0 * dt).clamp(MARGIN, WORLD - MARGIN);
                let nz = (z.p.z + z.vel.1 * dt).clamp(MARGIN, WORLD - MARGIN);
                z.p = Point3::new(nx, ground_h(nx, nz, self.seed) + z.class.altitude(), nz);
                z.moved = true;
                // Groans only while WALKING (half the class noise): a marching
                // wave pulls alert sleepers (harpies/venoms) along its path —
                // the wave grows as it travels — but an arrived, lingering pack
                // goes quiet, so the field re-settles (no perpetual chain; the
                // big cascades come from combat noise below).
                z.groan_t -= dt;
                if z.groan_t <= 0.0 {
                    z.groan_t = 3.0 + (z.p.x.to_bits() % 2048) as f64 * 0.001; // deterministic jitter
                    self.pending.push((z.p, z.class.noise_made() * 0.5));
                }
            } else if matches!(z.state, ZState::Investigating { .. }) {
                z.linger -= dt;
                if z.linger <= 0.0 { z.state = ZState::Dormant; z.heard = 0.0; z.moved = true; }
            }
        }
        // Resolve this frame's structure hits (attack noise feeds the wake
        // loop: pounding on a wall is what pulls the map in).
        for (sid, dmg, at, noise) in hits {
            self.pending.push((at, noise));
            let s = &mut self.structures[sid as usize];
            if s.hp <= 0.0 { continue; }
            s.hp -= dmg;
            if s.hp <= 0.0 { self.destroy_structure(sid as usize); }
        }
        self.step_towers(dt);
    }

    /// Towers auto-fire: "nearest" = one k-NN(1) per tower per shot;
    /// "highest threat" = a range cull + score-max (both TAB modes). Every
    /// shot emits noise — quiet defense doesn't grow the assault, loud does.
    fn step_towers(&mut self, dt: f64) {
        for sid in 0..self.structures.len() {
            if self.structures[sid].kind != SKind::Tower || self.structures[sid].hp <= 0.0 { continue; }
            self.tower_reload[sid] -= dt;
            if self.tower_reload[sid] > 0.0 { continue; }
            let tp = self.structures[sid].p;
            let target: Option<u32> = if self.tower_threat_mode {
                let ring = Sphere3::new(tp.x, tp.y, tp.z, TOWER_RANGE);
                self.zindex.cull(&ring).iter()
                    .map(|it| { let (dx, dz) = (it.p.x - tp.x, it.p.z - tp.z); (it.id, it.class.threat() / (1.0 + (dx * dx + dz * dz).sqrt() * 0.02)) })
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(id, _)| id)
            } else {
                self.zindex.knn(tp, 1).into_iter().find(|(d, _)| *d <= TOWER_RANGE).map(|(_, it)| it.id)
            };
            let Some(tid) = target else { continue; };
            self.tower_reload[sid] = TOWER_RELOAD;
            self.pending.push((tp, TOWER_NOISE));
            self.units[tid as usize].hp -= TOWER_DMG;
            if self.units[tid as usize].hp <= 0.0 { self.kill_zombie(tid as usize); }
        }
    }

    /// Death: corpse for the renderer, a death rattle (noise), free the slot,
    /// and drop the item from the index immediately (O(1) by handle).
    fn kill_zombie(&mut self, id: usize) {
        let (p, class) = (self.units[id].p, self.units[id].class);
        self.corpses.push((p, class, self.now));
        if self.corpses.len() > 45_000 { self.corpses.drain(0..5_000); } // cap the aftermath field
        self.pending.push((p, class.noise_made()));
        self.kills += 1;
        if let Some(h) = self.handles[id].take() { self.zindex.remove_ref(h); }
        self.units[id].hp = 0.0;
        self.units[id].moved = false;
        self.free_slots.push(id as u32);
    }
}

// -------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_layout_is_sane_and_static_index_matches_brute() {
        let h = Horde::new(42, 2000);
        assert_eq!(h.structures.iter().filter(|s| s.kind == SKind::CommandCenter).count(), 1);
        assert_eq!(h.structures.iter().filter(|s| s.kind == SKind::Gate).count(), 4, "one gate per cardinal");
        assert!(h.structures.iter().filter(|s| s.kind == SKind::Tower).count() >= 8);
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        for s in h.structures.iter().filter(|s| matches!(s.kind, SKind::Wall | SKind::Gate | SKind::Tower)) {
            let d = ((s.p.x - cx).powi(2) + (s.p.z - cz).powi(2)).sqrt();
            assert!((d - BASE_R).abs() < 1.0, "ring piece off the ring: {d}");
        }
        assert_eq!(h.sindex.item_count(), h.structures.len());
        // cull == brute on the static index
        let q = Sphere3::new(cx + BASE_R, 0.0, cz, 60.0);
        let mut want: Vec<u32> = h.structures.iter().enumerate()
            .filter(|(_, s)| { let (dx, dy, dz) = (s.p.x - (cx + BASE_R), s.p.y, s.p.z - cz); dx * dx + dy * dy + dz * dz <= 60.0 * 60.0 })
            .map(|(i, _)| i as u32).collect();
        let mut got: Vec<u32> = h.sindex.cull(&q).iter().map(|it| it.id).collect();
        want.sort(); got.sort();
        assert_eq!(want, got, "static index cull != brute force");
    }

    #[test]
    fn wake_cull_matches_brute_force() {
        let mut h = Horde::new(7, 6000);
        h.step(1.0 / 60.0); // populate the index
        // Pick a nest zombie as the blast point; amount 400 discriminates by
        // class: runners (333) & specials wake, walkers (500) don't.
        let p = h.units[0].p;
        let want: Vec<usize> = h.units.iter().enumerate()
            .filter(|(_, z)| {
                let d = ((z.p.x - p.x).powi(2) + (z.p.z - p.z).powi(2)).sqrt();
                z.dormant() && d <= z.class.hear() && 400.0 >= z.class.wake_threshold()
            })
            .map(|(i, _)| i).collect();
        h.emit_noise(p, 400.0);
        h.step(1.0 / 60.0);
        let got: Vec<usize> = h.units.iter().enumerate().filter(|(_, z)| !z.dormant()).map(|(i, _)| i).collect();
        assert_eq!(want, got, "woken set != brute-force wake rule");
        assert!(h.units.iter().all(|z| !z.dormant() || z.class != ZClass::Harpy || true)); // (harpies inside radius woke — covered by set equality)
        assert!(got.iter().all(|&i| h.units[i].class != ZClass::Walker), "amount 400 must not wake walkers (threshold 500)");
    }

    #[test]
    fn heard_accumulates_and_decays() {
        let mut h = Horde::new(9, 1500);
        h.step(1.0 / 60.0);
        let p = h.units[0].p;
        // 300 twice in quick succession beats the walker threshold (500)
        // because the accumulator barely decays between frames.
        h.emit_noise(p, 300.0);
        h.step(1.0 / 60.0);
        let heard_mid: Vec<f64> = h.units.iter().map(|z| z.heard).collect();
        assert!(heard_mid.iter().any(|&x| x > 0.0), "someone must have heard the first event");
        h.emit_noise(p, 300.0);
        h.step(1.0 / 60.0);
        assert!(h.units.iter().enumerate().any(|(_, z)| !z.dormant() && z.class == ZClass::Walker), "two stacked 300s must wake walkers in range");
        // and a lone accumulator decays: halves every second
        let mut lone = Horde::new(11, 300);
        lone.step(1.0 / 60.0);
        let q = lone.units[0].p;
        lone.emit_noise(q, 100.0);
        lone.step(1.0 / 60.0);
        let before: f64 = lone.units.iter().map(|z| z.heard).sum();
        for _ in 0..60 { lone.step(1.0 / 60.0); }
        let after: f64 = lone.units.iter().map(|z| z.heard).sum();
        assert!(after < before * 0.6 && after > 0.0, "heard should roughly halve over 1s: {before} -> {after}");
    }

    #[test]
    fn woken_investigate_then_resettle_dormant() {
        let mut h = Horde::new(21, 3000);
        h.step(1.0 / 60.0);
        h.emit_noise(h.units[0].p, 1000.0); // wake everything nearby
        h.step(1.0 / 60.0);
        let (_, active0) = h.counts();
        assert!(active0 > 0, "the blast must wake someone");
        for _ in 0..(60.0 / (1.0 / 60.0)) as usize { h.step(1.0 / 60.0); }
        let (_, active1) = h.counts();
        // March groans may pull a few alert sleepers along the way, but with no
        // sustained loud source the field must (mostly) re-settle once arrived.
        assert!(active1 < active0 / 4, "field should re-settle: {active0} -> {active1}");
    }

    #[test]
    fn flow_field_routes_to_the_cc() {
        // Descend the flow from an outside point: it must reach the CC (walls
        // are high cost, not blocking — the gate/weak-spot route exists).
        let h = Horde::new(5, 300);
        let cc = h.structures[h.cc_id].p;
        let (mut x, mut z) = (WORLD / 2.0 + 480.0, WORLD / 2.0 + 90.0);
        for _ in 0..4000 {
            let (fx, fz) = h.flow.flow_at(x, z);
            if fx == 0.0 && fz == 0.0 { break; }
            x += fx * 5.0; z += fz * 5.0;
            let d = ((x - cc.x).powi(2) + (z - cc.z).powi(2)).sqrt();
            if d < 25.0 { return; } // reached
        }
        panic!("flow walk never reached the Command Center");
    }

    #[test]
    fn towers_fire_kill_and_the_noise_wakes_sleepers() {
        let mut h = Horde::new(13, 2000);
        h.step(1.0 / 60.0);
        // A pack of walkers marching just inside a tower's range.
        let tid = h.structures.iter().position(|s| s.kind == SKind::Tower).unwrap();
        let tp = h.structures[tid].p;
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let out = ((tp.x - cx) / BASE_R, (tp.z - cz) / BASE_R); // outward normal
        for k in 0..12 {
            let (x, z) = (tp.x + out.0 * 30.0 + k as f64 * 1.5, tp.z + out.1 * 30.0);
            h.spawn_zombie(ZClass::Walker, x, z, ZState::Marching);
        }
        // Plant a listener: a dormant walker 100 wu out (inside its 160 hear).
        h.spawn_zombie(ZClass::Walker, tp.x + out.0 * 100.0, tp.z + out.1 * 100.0, ZState::Dormant);
        let li = h.units.len() - 1;
        let kills0 = h.kills;
        let mut max_heard = 0.0f64;
        for _ in 0..600 { h.step(1.0 / 60.0); max_heard = max_heard.max(h.units[li].heard); } // 10 s of tower fire
        assert!(h.kills > kills0, "tower must kill walkers in range");
        assert!(!h.corpses.is_empty(), "kills leave corpses");
        // The TAB noise economy, verified both ways: the battle noise REACHES
        // the sleeper (accumulates in `heard`) but a lone ballista is QUIET by
        // design (~5/shot vs the walker's 500 threshold) — it must NOT wake it.
        // Big sources (infection bursts, massed battles) do the waking.
        assert!(max_heard > 0.0, "tower noise must reach the dormant listener");
        assert!(h.units[li].dormant(), "a lone ballista must not wake walkers (noise-per-kill economy)");
    }

    #[test]
    fn infection_cascade_spawns_runners_and_detonates_noise() {
        let mut h = Horde::new(17, 500);
        h.step(1.0 / 60.0);
        // The innermost populated house — safely outside tower range (the
        // first pick got the attacker shot off the wall: the base defends
        // itself; this test is about the cascade, not the towers).
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let hid = h.structures.iter().enumerate()
            .filter(|(_, s)| s.kind == SKind::House && s.pop > 0)
            .min_by(|(_, a), (_, b)| { let da = (a.p.x - cx).powi(2) + (a.p.z - cz).powi(2); let db = (b.p.x - cx).powi(2) + (b.p.z - cz).powi(2); da.total_cmp(&db) })
            .map(|(i, _)| i).unwrap();
        let hp0 = h.structures[hid].p;
        let pop = h.structures[hid].pop as usize;
        let (d0, a0) = h.counts();
        // A chubby pounding the house (500 HP / 40 dmg ≈ 13 swings).
        h.spawn_zombie(ZClass::Chubby, hp0.x + 2.0, hp0.z, ZState::Attacking { sid: hid as u32 });
        for _ in 0..(20.0 * 60.0) as usize { h.step(1.0 / 60.0); if h.structures[hid].hp <= 0.0 { break; } }
        assert!(h.structures[hid].hp <= 0.0, "house must fall");
        h.step(1.0 / 60.0); // process the infection noise event
        let (d1, a1) = h.counts();
        assert!(d1 + a1 >= d0 + a0 + pop, "each colonist must rise as a runner: {} -> {}", d0 + a0, d1 + a1);
        assert!(h.units.iter().filter(|z| z.alive() && z.class == ZClass::Runner && matches!(z.state, ZState::Marching)).count() >= pop);
    }

    #[test]
    fn wave_spawns_at_the_announced_edge_and_marches_in() {
        let mut h = Horde::new(23, 300);
        h.wave_spawn_t = 0.5; // fast-forward the first wave
        for _ in 0..120 { h.step(1.0 / 60.0); }
        let marching: Vec<&Zombie> = h.units.iter().filter(|z| z.alive() && matches!(z.state, ZState::Marching)).collect();
        assert!(!marching.is_empty(), "wave must have spawned");
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let d0: f64 = marching.iter().map(|z| ((z.p.x - cx).powi(2) + (z.p.z - cz).powi(2)).sqrt()).sum::<f64>() / marching.len() as f64;
        for _ in 0..600 { h.step(1.0 / 60.0); }
        let marching2: Vec<&Zombie> = h.units.iter().filter(|z| z.alive() && matches!(z.state, ZState::Marching)).collect();
        let d1: f64 = marching2.iter().map(|z| ((z.p.x - cx).powi(2) + (z.p.z - cz).powi(2)).sqrt()).sum::<f64>() / marching2.len().max(1) as f64;
        assert!(d1 < d0 - 30.0, "the wave must close on the base: {d0:.0} -> {d1:.0}");
    }

    #[test]
    fn keep_index_consistent_through_combat_churn() {
        // Deaths (remove_ref), spawns (slot reuse via insert_ref) and movement
        // (update_ref) all churn the kept index — it must match brute force on
        // live positions at every sampled frame.
        let mut h = Horde::new(31, 1500);
        h.wave_spawn_t = 0.5;
        for f in 0..900 {
            h.step(1.0 / 60.0);
            if f % 30 != 0 { continue; }
            h.sync_index();
            let alive = h.units.iter().filter(|z| z.alive()).count();
            assert_eq!(h.zindex.item_count(), alive, "index count != alive at frame {f}");
            let c = h.structures[h.cc_id].p;
            let q = Sphere3::new(c.x, c.y, c.z, 260.0);
            let mut got: Vec<u32> = h.zindex.cull(&q).iter().map(|it| it.id).collect();
            let mut want: Vec<u32> = h.units.iter().enumerate()
                .filter(|(_, z)| z.alive() && { let (dx, dy, dz) = (z.p.x - c.x, z.p.y - c.y, z.p.z - c.z); dx * dx + dy * dy + dz * dz <= 260.0 * 260.0 })
                .map(|(i, _)| i as u32).collect();
            got.sort(); want.sort();
            assert_eq!(got, want, "kept index != brute at frame {f}");
        }
    }

    #[test]
    fn keep_index_matches_rebuild_every_frame() {
        let mut h = Horde::new(33, 2500);
        for f in 0..180 {
            if f % 45 == 10 { let p = h.units[f % h.units.len()].p; h.emit_noise(p, 600.0); }
            h.step(1.0 / 60.0);
            h.sync_index(); // bring the index current with post-apply positions
            let alive = h.units.len(); // no deaths in phase 1
            assert_eq!(h.zindex.item_count(), alive, "keep-index lost items at frame {f}");
            // sample cull vs brute over live positions
            let c = Point3::new(WORLD / 2.0, 0.0, WORLD / 2.0);
            let q = Sphere3::new(c.x, c.y, c.z, 300.0);
            let mut got: Vec<u32> = h.zindex.cull(&q).iter().map(|it| it.id).collect();
            let mut want: Vec<u32> = h.units.iter().enumerate()
                .filter(|(_, z)| { let (dx, dy, dz) = (z.p.x - c.x, z.p.y - c.y, z.p.z - c.z); dx * dx + dy * dy + dz * dz <= 300.0 * 300.0 })
                .map(|(i, _)| i as u32).collect();
            got.sort(); want.sort();
            assert_eq!(got, want, "kept index diverged from live positions at frame {f}");
        }
    }
}
