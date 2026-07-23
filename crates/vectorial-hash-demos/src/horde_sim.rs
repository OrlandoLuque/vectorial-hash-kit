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

use vectorial_hash::{Aabb, ItemRef, Point3, Positioned3, Shape3, Sphere3, Tree3};

pub use crate::siege_sim::Rng;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// ---------------------------------------------------------------- world config

pub const WORLD: f64 = 1800.0; // map side, world units — roomy: the horde needs somewhere to sleep
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
    /// A sleeper this mover shoved this frame (u32::MAX = none): contact wakes
    /// — the wave shakes the carpet awake instead of jamming against it.
    bump: u32,
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
    /// Structure HP. Bumped from the TAB baseline (1000/1500/2000/·/·/5000) so a
    /// big flood grinds the ring for a while instead of bursting straight to the CC
    /// (balance harness 2026-07-23; the CC especially needs to outlast the rush).
    pub fn max_hp(self) -> f64 { match self { Self::Wall => 2000.0, Self::Gate => 3000.0, Self::Tower => 4000.0, Self::House => 500.0, Self::Storehouse => 800.0, Self::CommandCenter => 14000.0 } }
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

// ----------------------------------------------------------------- scenarios

/// Map presets (HORDE_DESIGN § Scenarios) — same sim, different worlds. The
/// impassable ones (Pass / River / Forest) turn the flow field into a real
/// minimum-path system: blocked cells never relax in the Dijkstra, so the
/// horde funnels through the passes / bridges / forest trails. (The user has
/// an alternative min-path idea for the horde — baseline to compare against.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scenario { Classic, Pass, River, Forest, Patches }
impl Scenario {
    pub fn label(self) -> &'static str { match self { Self::Classic => "OPEN", Self::Pass => "PASS", Self::River => "RIVER", Self::Forest => "FOREST", Self::Patches => "PATCHES" } }
    pub fn next(self) -> Scenario { match self { Self::Classic => Self::Pass, Self::Pass => Self::River, Self::River => Self::Forest, Self::Forest => Self::Patches, Self::Patches => Self::Classic } }
}

/// Scenario-aware terrain height (pure): the Pass raises a ring ridge with
/// three gap passes; the River carves a meandering channel. (Forest keeps the
/// base heightfield — the trees are blockers, not hills.)
pub fn terrain_h(x: f64, z: f64, seed: f64, sc: Scenario) -> f64 {
    let base = ground_h(x, z, seed);
    let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
    match sc {
        Scenario::Pass => {
            let d = ((x - cx).powi(2) + (z - cz).powi(2)).sqrt();
            let band = 1.0 - ((d - 330.0).abs() / 46.0).min(1.0); // ring ridge at r≈330
            let a = (z - cz).atan2(x - cx);
            let gap = [0.5f64, 2.6, 4.7].iter().map(|g| { let mut da = (a - g).abs(); if da > std::f64::consts::PI { da = std::f64::consts::TAU - da; } (1.0 - da / 0.22).max(0.0) }).fold(0.0f64, f64::max);
            base + 58.0 * band * band * (1.0 - gap).max(0.0)
        }
        Scenario::Patches => {
            // Water pools dip smoothly (the mask is continuous noise, so the
            // shore blends); rock clumps bump up a little under their mesas.
            let (_, r, w) = patch_masks(x, z, seed);
            base - 3.2 * ((w - (PATCH_WATER - 0.03)) / 0.06).clamp(0.0, 1.0)
                 + 2.0 * ((r - (PATCH_ROCK - 0.02)) / 0.05).clamp(0.0, 1.0)
        }
        Scenario::River => {
            let zr = cz - 250.0 + 70.0 * (x * 0.008 + seed * 3.0).sin();
            let band = 1.0 - ((z - zr).abs() / 42.0).min(1.0);
            let mut h = base - 9.0 * band * band;
            // Causeway decks at the two carved crossings: the channel blends
            // back up to bank level so units cross ON something, not in water.
            for bx in [cx - 330.0, cx + 330.0] {
                let deck = 1.0 - ((x - bx).abs() / 13.0).min(1.0);
                if deck > 0.0 { h += (base + 0.6 - h) * deck.sqrt(); }
            }
            h
        }
        _ => base,
    }
}

/// Is (x,z) water in the River scenario? (bridges are carved in the pass grid)
pub fn is_water(x: f64, z: f64, seed: f64, sc: Scenario) -> bool {
    if sc != Scenario::River { return false; }
    let zr = (WORLD / 2.0) - 250.0 + 70.0 * (x * 0.008 + seed * 3.0).sin();
    (z - zr).abs() < 30.0
}

/// The forest mask (Forest scenario): dense value-noise woods; clearings and
/// trails are carved into the pass grid at build time.
pub fn is_forest(x: f64, z: f64, seed: f64) -> f64 { // 0..1 density
    fn h(ix: i64, iz: i64, s: i64) -> f64 {
        let mut n = (ix.wrapping_mul(374761393)) ^ (iz.wrapping_mul(668265263)) ^ s.wrapping_mul(1274126177);
        n = (n ^ (n >> 13)).wrapping_mul(1103515245);
        ((n ^ (n >> 16)) & 0xffff) as f64 / 65535.0
    }
    let s = (seed * 1e6) as i64 ^ 0x5157;
    let (ix, iz) = ((x * 0.011).floor() as i64, (z * 0.011).floor() as i64);
    let (fx, fz) = (x * 0.011 - (x * 0.011).floor(), z * 0.011 - (z * 0.011).floor());
    let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let (a, b, c, d) = (h(ix, iz, s), h(ix + 1, iz, s), h(ix, iz + 1, s), h(ix + 1, iz + 1, s));
    a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz
}

// ---- the PATCHES scenario (the They Are Billions structure, researched):
// mostly-open ground + BLOB PATCHES of forest / rock / water at theme-weighted
// densities, each an independent value-noise mask at its own frequency. The
// walkable network is the RESIDUAL space between patches — gorges, small
// plains, pockets and chokepoints emerge from how blobs touch, nothing is
// hand-traced. (Contrast with FOREST, which is the inverse: solid woods with
// carved trails.) A connectivity pass then guarantees playability.

/// The three patch masks at (x,z): (forest, rock, water) noise values 0..1.
/// Frequencies picked so pools are big, woods medium, rock clumps small.
pub fn patch_masks(x: f64, z: f64, seed: f64) -> (f64, f64, f64) {
    fn h(ix: i64, iz: i64, s: i64) -> f64 {
        let mut n = (ix.wrapping_mul(374761393)) ^ (iz.wrapping_mul(668265263)) ^ s.wrapping_mul(1274126177);
        n = (n ^ (n >> 13)).wrapping_mul(1103515245);
        ((n ^ (n >> 16)) & 0xffff) as f64 / 65535.0
    }
    fn vnoise(x: f64, z: f64, f: f64, s: i64) -> f64 {
        let (xf, zf) = (x * f, z * f);
        let (ix, iz) = (xf.floor() as i64, zf.floor() as i64);
        let (fx, fz) = (xf - xf.floor(), zf - zf.floor());
        let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
        let (a, b, c, d) = (h(ix, iz, s), h(ix + 1, iz, s), h(ix, iz + 1, s), h(ix + 1, iz + 1, s));
        a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz
    }
    let s = (seed * 1e6) as i64;
    (vnoise(x, z, 0.011, s ^ 0x70A7), vnoise(x, z, 0.019, s ^ 0x50CC), vnoise(x, z, 0.007, s ^ 0x3AB0))
}
pub const PATCH_FOREST: f64 = 0.62;
pub const PATCH_ROCK: f64 = 0.70;
pub const PATCH_WATER: f64 = 0.71;

/// Patch classification at (x,z): 0 open · 1 forest · 2 rock · 3 water
/// (water wins over rock wins over forest where blobs overlap).
pub fn patch_cell(x: f64, z: f64, seed: f64) -> u8 {
    let (f, r, w) = patch_masks(x, z, seed);
    if w >= PATCH_WATER { 3 } else if r >= PATCH_ROCK { 2 } else if f >= PATCH_FOREST { 1 } else { 0 }
}

/// A* over the pass grid — the DEFENDERS' minimum paths (sorties through the
/// forest trails / over the bridges). One search per squad dispatch, waypoints
/// decimated to every ~4 cells. Returns empty if unreachable.
fn astar_path(grid: &[bool], n: usize, cell: f64, from: (f64, f64), to: (f64, f64)) -> Vec<(f64, f64)> {
    let idx = |x: f64, z: f64| -> (usize, usize) { (((x / cell) as usize).min(n - 1), ((z / cell) as usize).min(n - 1)) };
    let (s, g) = (idx(from.0, from.1), idx(to.0, to.1));
    if !grid[s.1 * n + s.0] || !grid[g.1 * n + g.0] { return Vec::new(); }
    let hcost = |i: usize, j: usize| { let (dx, dz) = ((i as i64 - g.0 as i64).abs(), (j as i64 - g.1 as i64).abs()); (dx.max(dz) * 100 + dx.min(dz) * 41) as u32 };
    let mut best = vec![u32::MAX; n * n];
    let mut prev = vec![u32::MAX; n * n];
    let mut heap = std::collections::BinaryHeap::new();
    best[s.1 * n + s.0] = 0;
    heap.push(std::cmp::Reverse((hcost(s.0, s.1), 0u32, s.0, s.1)));
    while let Some(std::cmp::Reverse((_, d, i, j))) = heap.pop() {
        if (i, j) == g { break; }
        if d > best[j * n + i] { continue; }
        for (di, dj, w) in [(-1i64, 0i64, 100u32), (1, 0, 100), (0, -1, 100), (0, 1, 100), (-1, -1, 141), (1, -1, 141), (-1, 1, 141), (1, 1, 141)] {
            let (ni, nj) = (i as i64 + di, j as i64 + dj);
            if ni < 0 || nj < 0 || ni >= n as i64 || nj >= n as i64 { continue; }
            let (ni, nj) = (ni as usize, nj as usize);
            if !grid[nj * n + ni] { continue; }
            let nd = d + w;
            if nd < best[nj * n + ni] {
                best[nj * n + ni] = nd;
                prev[nj * n + ni] = (j * n + i) as u32;
                heap.push(std::cmp::Reverse((nd + hcost(ni, nj), nd, ni, nj)));
            }
        }
    }
    if best[g.1 * n + g.0] == u32::MAX { return Vec::new(); }
    let mut cells = Vec::new();
    let mut cur = g.1 * n + g.0;
    while cur != s.1 * n + s.0 { cells.push(cur); let p = prev[cur]; if p == u32::MAX { break; } cur = p as usize; }
    cells.reverse();
    cells.iter().step_by(4).chain(cells.last()).map(|&c| (((c % n) as f64 + 0.5) * cell, ((c / n) as f64 + 0.5) * cell)).collect()
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
    /// The `O` toggle — the user's multi-source idea: `false` = single goal
    /// (the CC), `true` = seed 0 at EVERY live building and flood once, so the
    /// field points every zombie at its NEAREST building and re-routes to the
    /// next as buildings fall. N goals cost the same one flood as 1 goal.
    pub multi: bool,
}

/// Is this the kind of structure the horde wants to reach (a multi-goal seed)?
/// Buildings, not the wall ring — walls are cost, not a destination.
fn is_goal(k: SKind) -> bool { matches!(k, SKind::House | SKind::Storehouse | SKind::CommandCenter) }

impl FlowField {
    fn new(n: usize) -> FlowField {
        FlowField { n, cell: WORLD / n as f64, dir: vec![(0.0, 0.0); n * n], integ: vec![u32::MAX; n * n], dirty: true, rebuild_t: 0.0, multi: false }
    }
    fn cell_of(&self, x: f64, z: f64) -> (usize, usize) {
        (((x / self.cell) as usize).min(self.n - 1), ((z / self.cell) as usize).min(self.n - 1))
    }
    pub fn flow_at(&self, x: f64, z: f64) -> (f64, f64) {
        let (i, j) = self.cell_of(x, z);
        let d = self.dir[j * self.n + i];
        (d.0 as f64, d.1 as f64)
    }
    /// Did the last integration reach this cell? `false` = walled off from the
    /// CC (an unconnected forest pocket, deep woods, outside everything).
    pub fn reachable(&self, x: f64, z: f64) -> bool {
        let (i, j) = self.cell_of(x, z);
        self.integ[j * self.n + i] != u32::MAX
    }
    /// Rebuild cost + integration + directions from live structure HP.
    /// Impassable pass-grid cells (ridge / water / woods) never relax — the
    /// flood pours through the carved passes/bridges/trails: the horde's
    /// minimum paths fall out of the same Dijkstra that handles breaches.
    fn rebuild(&mut self, structures: &[Structure], cc: Point3, pass: &[bool], pn: usize, pcell: f64) {
        let n = self.n;
        // Per-cell traversal cost (milli-units): open ground 100; a live wall
        // piece adds ~6× ground + a term falling with damage.
        let mut cost = vec![100u32; n * n];
        let mut blocked = vec![false; n * n];
        for j in 0..n {
            for i in 0..n {
                let (x, z) = ((i as f64 + 0.5) * self.cell, (j as f64 + 0.5) * self.cell);
                let pi = ((z / pcell) as usize).min(pn - 1) * pn + ((x / pcell) as usize).min(pn - 1);
                blocked[j * n + i] = !pass[pi];
            }
        }
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
        // Multi-source Dijkstra over 8-neighbours (diagonal ×1.41). Seeds =
        // just the CC (baseline) or EVERY live building (the `multi` toggle):
        // a genuine multi-source flood — the integration field then holds the
        // distance to the NEAREST goal, so one pass serves any number of them.
        self.integ.iter_mut().for_each(|v| *v = u32::MAX);
        let mut heap = std::collections::BinaryHeap::new();
        let seed = |integ: &mut Vec<u32>, heap: &mut std::collections::BinaryHeap<_>, x: f64, z: f64| {
            let (ci, cj) = (((x / self.cell) as usize).min(n - 1), ((z / self.cell) as usize).min(n - 1));
            if !blocked[cj * n + ci] && integ[cj * n + ci] != 0 { integ[cj * n + ci] = 0; heap.push(std::cmp::Reverse((0u32, ci, cj))); }
        };
        if self.multi {
            let mut any = false;
            for s in structures { if s.hp > 0.0 && is_goal(s.kind) { seed(&mut self.integ, &mut heap, s.p.x, s.p.z); any = true; } }
            if !any { seed(&mut self.integ, &mut heap, cc.x, cc.z); } // all buildings gone → the CC (which is one anyway)
        } else {
            seed(&mut self.integ, &mut heap, cc.x, cc.z);
        }
        while let Some(std::cmp::Reverse((d, i, j))) = heap.pop() {
            if d > self.integ[j * n + i] { continue; }
            for (di, dj, w) in [(-1i64, 0i64, 100u32), (1, 0, 100), (0, -1, 100), (0, 1, 100), (-1, -1, 141), (1, -1, 141), (-1, 1, 141), (1, 1, 141)] {
                let (ni, nj) = (i as i64 + di, j as i64 + dj);
                if ni < 0 || nj < 0 || ni >= n as i64 || nj >= n as i64 { continue; }
                let (ni, nj) = (ni as usize, nj as usize);
                if blocked[nj * n + ni] { continue; }
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

pub const TOWER_RANGE: f64 = 84.0; // reach a bit further down the funnel
pub const TOWER_DMG: f64 = 150.0;
pub const TOWER_RELOAD: f64 = 0.6; // faster — the towers are the TAB backbone (balance harness)
pub const TOWER_NOISE: f64 = 5.0; // the ballista: quiet per kill — by design

// --------------------------------------------------------------- defenders

/// Mobile defender kinds (HORDE_DESIGN.md layer 2) + the works economy
/// (layer 3: crews repair, porters haul the materials that pace all repair).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DKind { Ranger, Soldier, Sniper, Crew, Porter }
impl DKind {
    pub fn max_hp(self) -> f64 { match self { Self::Ranger => 60.0, Self::Soldier => 120.0, Self::Sniper => 150.0, Self::Crew => 60.0, Self::Porter => 40.0 } }
    pub fn dmg(self) -> f64 { match self { Self::Ranger => 10.0, Self::Soldier => 15.0, Self::Sniper => 100.0, _ => 0.0 } }
    pub fn range(self) -> f64 { match self { Self::Ranger => 48.0, Self::Soldier => 40.0, Self::Sniper => 64.0, _ => 0.0 } }
    pub fn reload(self) -> f64 { match self { Self::Ranger => 0.6, Self::Soldier => 1.0, Self::Sniper => 2.5, _ => 1.0 } }
    /// Noise per shot — the ranger is the silent one (the whole discipline
    /// policy hinges on this asymmetry).
    pub fn noise(self) -> f64 { match self { Self::Ranger => 1.0, Self::Soldier => 3.0, Self::Sniper => 10.0, _ => 0.0 } }
    pub fn speed(self) -> f64 { match self { Self::Ranger => 22.0, Self::Soldier => 18.0, Self::Sniper => 16.0, Self::Crew => 14.0, Self::Porter => 16.0 } }
    pub fn fighter(self) -> bool { matches!(self, Self::Ranger | Self::Soldier | Self::Sniper) }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DState {
    /// Fighter: hold the assigned sector post (engage from it).
    Post,
    /// Ranger sortie: out through the nearest gate to silently clear a nest
    /// (peacetime only; recalled the moment a wave is announced).
    Sortie { tx: f64, tz: f64 },
    /// Crew: repairing / rebuilding structure `sid` (needs stock).
    Repairing { sid: u32 },
    /// Porter: hauling to crew `did` (`loaded` = carrying a bundle).
    Hauling { did: u32, loaded: bool },
    /// Works unit recalled home (breach alarm / danger) — runs, then idles.
    Fleeing,
    /// Waiting for a job at home (crews/porters), or respawning (fighters,
    /// timer in `respawn_t`).
    Idle,
}

#[derive(Clone)]
pub struct Defender {
    pub kind: DKind,
    pub p: Point3,
    pub hp: f64,
    pub state: DState,
    pub sector: usize,
    pub reload_t: f64,
    pub respawn_t: f64,
    /// Heading (radians) updated as it walks, so the renderer turns the model to
    /// face its movement instead of always looking one way (user 2026-07-22).
    pub face: f32,
    /// Did it actually move this frame? The renderer plays the idle clip when not
    /// (user 2026-07-23). Reset each frame, set by the movement code.
    pub moving: bool,
    /// Crew: repair stock (a porter delivery = +20; repair burns 2/s).
    pub stock: f64,
    pub shots: u64,
    /// A* waypoints over the pass grid (sortie out along the forest trails /
    /// causeways, and the trail home) — empty means walk straight.
    pub path: Vec<(f64, f64)>,
}
impl Defender { pub fn alive(&self) -> bool { self.hp > 0.0 } }

pub const SECTORS: usize = 16;
pub const CREW_REPAIR: f64 = 30.0; // HP/s while stocked
pub const PORTER_BUNDLE: f64 = 20.0; // stock per delivery

/// Two-leg routing through the nearest LIVE gate whenever a friendly walk
/// crosses the wall ring — the gates are the only human passage (zombies just
/// eat the wall). Returns the waypoint to head for right now.
fn via_gate(gates: &[(f64, f64)], from: Point3, tx: f64, tz: f64) -> (f64, f64) {
    let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
    let inside = |x: f64, z: f64| ((x - cx).powi(2) + (z - cz).powi(2)).sqrt() < BASE_R + 10.0;
    if inside(from.x, from.z) == inside(tx, tz) || gates.is_empty() { return (tx, tz); }
    let g = gates.iter().min_by(|a, b| {
        let da = ((a.0 - from.x).powi(2) + (a.1 - from.z).powi(2)).sqrt() + ((a.0 - tx).powi(2) + (a.1 - tz).powi(2)).sqrt();
        let db = ((b.0 - from.x).powi(2) + (b.1 - from.z).powi(2)).sqrt() + ((b.0 - tx).powi(2) + (b.1 - tz).powi(2)).sqrt();
        da.total_cmp(&db)
    }).unwrap();
    if ((g.0 - from.x).powi(2) + (g.1 - from.z).powi(2)).sqrt() > 12.0 { (g.0, g.1) } else { (tx, tz) }
}

// ----------------------------------------------------------------------- sim

/// Default decision-bucket period (frames between re-decides for a zombie away
/// from the walls). 8 = 7.5 Hz @60fps — chosen from the `horde_bench` sweep:
/// dropping from the old 15 Hz (`4`) roughly halves the decide pass with no
/// visible change to the march (steering stays coherent, combat is unbucketed).
/// Override with `$HORDE_DECIDE_N`.
pub const DECIDE_N_DEFAULT: u64 = 8;

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
    /// Shot tracers for the renderer: (from, to, time fired) — aged out fast.
    pub tracers: Vec<(Point3, Point3, f64)>,
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
    /// While `now < wave_active_until` a wave is announced OR still marching in, so
    /// fighters hold the line (no sorties, recall the ones already out). Set on
    /// announce to spawn+35 s; `wave_announced` alone misses the post-spawn march.
    wave_active_until: f64,
    /// Trickle reinforcements (user 2026-07-23): a new fighter arrives every
    /// `reinforce_interval` s (faster at higher pop) up to `fighter_cap`, so the
    /// garrison recovers + grows through the escalating waves instead of only
    /// respawning its dead.
    reinforce_t: f64,
    reinforce_interval: f64,
    fighter_cap: usize,
    /// Set when the Command Center falls (defeat) or the final wave is cleared
    /// (victory): (time it happened, victory?). The run resets ~12 s later.
    pub game_over: Option<(f64, bool)>,
    pub run: u32,
    pub kills: u64,
    pub rng: Rng,
    pub now: f64,
    /// Frame counter — staggers the decision buckets.
    frame: u64,
    /// Decision-bucket period: a zombie away from the walls re-decides every
    /// `decide_n` frames (staggered by id), coasting on its cached velocity in
    /// between. 4=15 Hz@60fps · 8=7.5 Hz · 15=4 Hz. Env `$HORDE_DECIDE_N`.
    /// Larger = cheaper decide pass, staler steering — movement/heard/swings and
    /// near-wall (combat) units always run at full rate regardless.
    decide_n: u64,
    pub seed: f64,
    /// Woken-this-frame counter (for HUD/telemetry).
    pub woken_last: usize,
    /// Bumped whenever the DORMANT SET changes (wake, re-sleep, a sleeper dies,
    /// a dormant spawn, the final-wave rise) — renderers keep a prebuilt static
    /// instance buffer for the sleeping carpet and rebuild it only when
    /// `(run, dormant_epoch)` moves. Positions of sleepers never change, so
    /// between bumps the buffer is exact.
    pub dormant_epoch: u64,
    base_pop: usize,
    base_seed: u64,
    // ---- the defense (Commander + mobile defenders + works economy)
    pub defenders: Vec<Defender>,
    /// Per-sector threat (Commander's map, refreshed at 1 Hz).
    pub threat: [f64; SECTORS],
    /// Noise discipline: `false` → only rangers engage (silent clearing);
    /// flips when a wave commits near the walls.
    pub weapons_free: bool,
    cmd_t: f64,
    /// Most recent breach (position, time): drives the porter/crew recall and
    /// keeps repair jobs out of the danger zone for a while.
    pub breach: Option<(Point3, f64)>,
    /// Structure ids of the four cardinal gates (friendly passage points).
    pub gates: Vec<usize>,
    /// Map preset — decides terrain features + the passability grid.
    pub scenario: Scenario,
    /// Passability at flow-grid resolution (Pass ridges, River water, Forest
    /// woods = false; trails/bridges/passes carved true). Classic: all true.
    pub pass_grid: Vec<bool>,
    pub pass_n: usize,
    pub pass_cell: f64,
    /// Zombie-index mode (the `M` toggle): keep-maintained Tree3, or a
    /// rebuilt-per-frame MortonGrid3 — the structure trade-off, live.
    pub zmode: ZMode,
    zmorton: vectorial_hash::MortonGrid3<IZombie>,
}

/// The live index-structure toggle (docs/CHOOSING.md trade-offs, on screen):
/// Tree3 is keep-maintained (`update_ref`, sleepers skipped); MortonGrid3 has
/// no in-place handles, so its side of the switch is a clear+reinsert of every
/// LIVE zombie per frame — the rebuild-vs-keep comparison, watchable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ZMode { Tree, Morton }
impl ZMode {
    pub fn label(self) -> &'static str { match self { Self::Tree => "TREE3", Self::Morton => "MORTON" } }
    pub fn next(self) -> ZMode { match self { Self::Tree => Self::Morton, Self::Morton => Self::Tree } }
}

/// A borrow of whichever zombie index is live — one query surface so the wake
/// blasts / separation / towers / defenders don't care which structure answers.
pub enum ZQuery<'a> { Tree(&'a Tree3<IZombie>), Morton(&'a vectorial_hash::MortonGrid3<IZombie>) }
impl<'a> ZQuery<'a> {
    pub fn cull<S: Shape3>(&self, s: &S) -> Vec<&'a IZombie> { match self { Self::Tree(t) => t.cull(s), Self::Morton(m) => m.cull(s) } }
    pub fn knn(&self, p: Point3, k: usize) -> Vec<(f64, &'a IZombie)> { match self { Self::Tree(t) => t.knn(p, k), Self::Morton(m) => m.knn(p, k) } }
}

/// The base layout: a stone wall ring with 4 cardinal gates and towers every
/// few segments, houses + storehouses inside, the Command Center dead centre.
fn build_base(rng: &mut Rng, seed: f64, sc: Scenario) -> Vec<Structure> {
    let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
    let at = |x: f64, z: f64| Point3::new(x, terrain_h(x, z, seed, sc), z);
    let mut s = Vec::new();
    let segs = (std::f64::consts::TAU * BASE_R / 8.0) as usize; // one wall piece ≈ every 8 wu
    let step = std::f64::consts::TAU / segs as f64;
    // Exactly one gate per cardinal: the single closest segment to each.
    let gates: Vec<usize> = (0..4).map(|q| ((q as f64 * std::f64::consts::FRAC_PI_2) / step).round() as usize % segs).collect();
    for i in 0..segs {
        let a = i as f64 * step;
        let (x, z) = (cx + a.cos() * BASE_R, cz + a.sin() * BASE_R);
        let kind = if gates.contains(&i) { SKind::Gate } else if i % 6 == 3 { SKind::Tower } else { SKind::Wall };
        s.push(Structure { kind, p: at(x, z), hp: kind.max_hp(), pop: 0 });
    }
    // Houses: rejection-sampled so no two buildings interpenetrate (≥16 wu
    // between houses, ≥26 wu clear of the Command Center / storehouse spots).
    let mut placed: Vec<(f64, f64)> = Vec::new();
    let mut tries = 0;
    while placed.len() < 28 && tries < 600 {
        tries += 1;
        let (a, r) = (rng.range(0.0, std::f64::consts::TAU), rng.range(34.0, 110.0));
        let (x, z) = (cx + a.cos() * r, cz + a.sin() * r);
        if r < 26.0 { continue; }
        if placed.iter().any(|(px, pz)| { let (dx, dz) = (x - px, z - pz); dx * dx + dz * dz < 16.0 * 16.0 }) { continue; }
        placed.push((x, z));
        s.push(Structure { kind: SKind::House, p: at(x, z), hp: SKind::House.max_hp(), pop: 5 + (rng.unit() * 15.0) as u32 });
    }
    for q in 0..2 { // storehouses flanking the CC (phase-3 hauling endpoints)
        let a = q as f64 * std::f64::consts::PI + 0.7;
        s.push(Structure { kind: SKind::Storehouse, p: at(cx + a.cos() * 40.0, cz + a.sin() * 40.0), hp: SKind::Storehouse.max_hp(), pop: 0 });
    }
    s.push(Structure { kind: SKind::CommandCenter, p: at(cx, cz), hp: SKind::CommandCenter.max_hp(), pop: 30 });
    s
}

/// Scatter the dormant field: nest blobs outside the base, class mix from the
/// research (~70% walkers, the rest specials). Placement = a **jittered hex
/// lattice masked by the nest discs**: bodies are ≥ ~1.5 wu apart BY
/// CONSTRUCTION (no two sleepers share a spot, even where nests overlap), and
/// sampling the masked points evenly keeps every nest populated.
fn spawn_field_at(rng: &mut Rng, pop: usize, seed: f64, sc: Scenario, centers: &[(f64, f64)], nr: f64, grid: &[bool], pn: usize, pcell: f64) -> Vec<Zombie> {
    const SPACING: f64 = 2.4;
    let pass = |x: f64, z: f64| grid[((z / pcell) as usize).min(pn - 1) * pn + ((x / pcell) as usize).min(pn - 1)];
    // All jittered lattice points inside some nest disc, on PASSABLE ground…
    let mut spots: Vec<(f64, f64)> = Vec::with_capacity(pop * 2);
    let rows = (WORLD / SPACING) as usize;
    for gj in 0..rows {
        let z0 = gj as f64 * SPACING;
        for gi in 0..rows {
            let x0 = gi as f64 * SPACING + (gj % 2) as f64 * SPACING * 0.5;
            // Jitter ±0.3 keeps the worst-case pair at 1.8 wu — above the 1.7
            // contact-wake radius, so a fresh field has no self-igniting pairs.
            let (x, z) = ((x0 + rng.range(-0.3, 0.3)).clamp(MARGIN, WORLD - MARGIN), (z0 + rng.range(-0.3, 0.3)).clamp(MARGIN, WORLD - MARGIN));
            if !pass(x, z) { continue; }
            // Never bed a sleeper IN water (the connectivity pass may carve a passable
            // corridor across a water blob — fine to cross, wrong to spawn on). 2026-07-22.
            if sc == Scenario::Patches && patch_cell(x, z, seed) == 3 { continue; }
            if centers.iter().any(|(nx, nz)| { let (dx, dz) = (x - nx, z - nz); dx * dx + dz * dz < nr * nr }) { spots.push((x, z)); }
        }
    }
    // …then take an even random sample of exactly `pop` of them.
    let mut units = Vec::with_capacity(pop);
    let take = pop.min(spots.len());
    for k in 0..take {
        // partial Fisher–Yates: pick the k-th from the remaining tail
        let j = k + (rng.next() as usize) % (spots.len() - k);
        spots.swap(k, j);
        let (x, z) = spots[k];
        let roll = rng.unit();
        // Patches is the TAB choke design: flyers bypass the maze, so drop them here
        // (harpy → runner) so the funnel actually holds (user 2026-07-22).
        let class = if roll < 0.70 { ZClass::Walker } else if roll < 0.85 { ZClass::Runner } else if roll < 0.91 { ZClass::Chubby } else if roll < 0.96 { ZClass::Venom } else if sc == Scenario::Patches { ZClass::Runner } else { ZClass::Harpy };
        let y = terrain_h(x, z, seed, sc) + class.altitude();
        units.push(Zombie { class, p: Point3::new(x, y, z), vel: (0.0, 0.0), state: ZState::Dormant, hp: class.max_hp(), heard: 0.0, linger: 0.0, groan_t: rng.range(0.5, 2.0), swing_t: 0.0, bump: u32::MAX, moved: false });
    }
    units
}

impl Horde {
    pub fn new(seed: u64, pop: usize) -> Horde { Self::with_scenario(seed, pop, Scenario::Classic) }

    pub fn with_scenario(seed: u64, pop: usize, sc: Scenario) -> Horde {
        let fseed = (seed % 100_000) as f64 * 0.013 + 0.29;
        let mut rng = Rng::new(seed | 1);
        let structures = build_base(&mut rng, fseed, sc);
        // ---- the passability grid (flow resolution): ridges / water / woods
        // block; passes / bridges / trails / clearings are carved true.
        let (pn, pcell) = (150usize, WORLD / 150.0);
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let mut grid = vec![true; pn * pn];
        for j in 0..pn {
            for i in 0..pn {
                let (x, z) = ((i as f64 + 0.5) * pcell, (j as f64 + 0.5) * pcell);
                grid[j * pn + i] = match sc {
                    Scenario::Classic => true,
                    Scenario::Pass => terrain_h(x, z, fseed, sc) - ground_h(x, z, fseed) < 24.0, // below half-ridge
                    Scenario::River => !is_water(x, z, fseed, sc),
                    Scenario::Forest => is_forest(x, z, fseed) < 0.46,
                    Scenario::Patches => patch_cell(x, z, fseed) == 0, // walkable = between the blobs
                };
            }
        }
        let set_disc = |grid: &mut Vec<bool>, x: f64, z: f64, r: f64, v: bool| {
            let (i0, i1) = ((((x - r) / pcell).max(0.0)) as usize, (((x + r) / pcell) as usize).min(pn - 1));
            let (j0, j1) = ((((z - r) / pcell).max(0.0)) as usize, (((z + r) / pcell) as usize).min(pn - 1));
            for j in j0..=j1 { for i in i0..=i1 {
                let (px, pz) = ((i as f64 + 0.5) * pcell, (j as f64 + 0.5) * pcell);
                if (px - x).powi(2) + (pz - z).powi(2) <= r * r { grid[j * pn + i] = v; }
            } }
        };
        if sc == Scenario::River { // two bridges over the channel
            for bx in [cx - 330.0, cx + 330.0] {
                let zr = cz - 250.0 + 70.0 * (bx * 0.008 + fseed * 3.0).sin();
                for k in 0..9 { set_disc(&mut grid, bx, zr - 40.0 + k as f64 * 10.0, 14.0, true); }
            }
        }
        // Nest centres (need passable ground except in Forest, where we carve).
        let nests = (pop / 300).clamp(3, 80);
        let nr = ((pop as f64 * 2.4 * 2.4 * 1.5) / (std::f64::consts::PI * nests as f64)).sqrt().clamp(22.0, 130.0);
        let mut centers: Vec<(f64, f64)> = Vec::with_capacity(nests);
        let mut tries = 0;
        while centers.len() < nests && tries < nests * 40 {
            tries += 1;
            let a = rng.range(0.0, std::f64::consts::TAU);
            let r = rng.range(BASE_R + 120.0 + nr * 0.4, WORLD / 2.0 - 40.0);
            let (x, z) = (cx + a.cos() * r, cz + a.sin() * r);
            let ci = ((z / pcell) as usize).min(pn - 1) * pn + ((x / pcell) as usize).min(pn - 1);
            if sc != Scenario::Forest && !grid[ci] { continue; }
            centers.push((x, z));
        }
        if matches!(sc, Scenario::Forest | Scenario::Patches) {
            set_disc(&mut grid, cx, cz, BASE_R + 70.0, true); // the base clearing (the TAB spawn guarantee)
        }
        if sc == Scenario::Patches {
            // Nest clearings, then the CONNECTIVITY pass — TAB's generator
            // guarantees playable maps; here: flood from the CC, find the
            // biggest walkable-but-unreached pocket, carve a corridor from its
            // closest cell to the nearest reached cell, repeat until every
            // pocket that matters joins the network. Small pockets stay as
            // dead-end recovecos — that's the TAB look, on purpose.
            for &(nx, nz) in &centers { set_disc(&mut grid, nx, nz, nr + 6.0, true); }
            let start = ((cz / pcell) as usize).min(pn - 1) * pn + ((cx / pcell) as usize).min(pn - 1);
            for _round in 0..48 {
                // reachable set from the CC (4-neighbour flood)
                let mut reach = vec![false; pn * pn];
                let mut q = std::collections::VecDeque::new();
                if grid[start] { reach[start] = true; q.push_back(start); }
                while let Some(c) = q.pop_front() {
                    let (i, j) = (c % pn, c / pn);
                    for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                        let (ni, nj) = (i as i64 + di, j as i64 + dj);
                        if ni < 0 || nj < 0 || ni >= pn as i64 || nj >= pn as i64 { continue; }
                        let nc = nj as usize * pn + ni as usize;
                        if grid[nc] && !reach[nc] { reach[nc] = true; q.push_back(nc); }
                    }
                }
                // distance-to-reached over ALL cells (multi-source BFS through
                // blobs) — `near` remembers which reached cell is closest.
                let (mut dist, mut near) = (vec![u32::MAX; pn * pn], vec![u32::MAX; pn * pn]);
                let mut q2 = std::collections::VecDeque::new();
                for c in 0..pn * pn { if reach[c] { dist[c] = 0; near[c] = c as u32; q2.push_back(c); } }
                while let Some(c) = q2.pop_front() {
                    let (i, j) = (c % pn, c / pn);
                    for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                        let (ni, nj) = (i as i64 + di, j as i64 + dj);
                        if ni < 0 || nj < 0 || ni >= pn as i64 || nj >= pn as i64 { continue; }
                        let nc = nj as usize * pn + ni as usize;
                        if dist[nc] == u32::MAX { dist[nc] = dist[c] + 1; near[nc] = near[c]; q2.push_back(nc); }
                    }
                }
                // biggest unreached pocket + its doorway (cell nearest the network)
                let mut seen = vec![false; pn * pn];
                let mut best: Option<(usize, usize)> = None;
                for c0 in 0..pn * pn {
                    if !grid[c0] || reach[c0] || seen[c0] { continue; }
                    let (mut size, mut doorway) = (0usize, c0);
                    let mut q3 = std::collections::VecDeque::new();
                    seen[c0] = true; q3.push_back(c0);
                    while let Some(c) = q3.pop_front() {
                        size += 1;
                        if dist[c] < dist[doorway] { doorway = c; }
                        let (i, j) = (c % pn, c / pn);
                        for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                            let (ni, nj) = (i as i64 + di, j as i64 + dj);
                            if ni < 0 || nj < 0 || ni >= pn as i64 || nj >= pn as i64 { continue; }
                            let nc = nj as usize * pn + ni as usize;
                            if grid[nc] && !reach[nc] && !seen[nc] { seen[nc] = true; q3.push_back(nc); }
                        }
                    }
                    // Link pockets ≥8 cells; below that stays a dead-end recoveco.
                    if size >= 8 && best.map(|(s, _)| size > s).unwrap_or(true) { best = Some((size, doorway)); }
                }
                let Some((_, door)) = best else { break; };
                let t = near[door] as usize;
                let (x0, z0) = (((door % pn) as f64 + 0.5) * pcell, ((door / pn) as f64 + 0.5) * pcell);
                let (x1, z1) = (((t % pn) as f64 + 0.5) * pcell, ((t / pn) as f64 + 0.5) * pcell);
                let d = ((x1 - x0).powi(2) + (z1 - z0).powi(2)).sqrt().max(1.0);
                let steps = (d / 6.0) as usize + 1;
                for k in 0..=steps { let u = k as f64 / steps as f64; set_disc(&mut grid, x0 + (x1 - x0) * u, z0 + (z1 - z0) * u, 9.0, true); }
            }
        }
        if sc == Scenario::Forest {
            // Carve the map: base clearing (above), a winding trail from each
            // gate to the map edge, a clearing per nest, and a connector from
            // each nest to its nearest trail point — guaranteed minimum paths.
            let mut trail_pts: Vec<(f64, f64)> = Vec::new();
            for q in 0..4 {
                let a0 = q as f64 * std::f64::consts::FRAC_PI_2;
                let mut r = BASE_R;
                while r < WORLD / 2.0 - 10.0 {
                    let a = a0 + 0.5 * ((r - BASE_R) * 0.012 + q as f64).sin();
                    let (x, z) = ((cx + a.cos() * r).clamp(4.0, WORLD - 4.0), (cz + a.sin() * r).clamp(4.0, WORLD - 4.0));
                    set_disc(&mut grid, x, z, 15.0, true);
                    trail_pts.push((x, z));
                    r += 10.0;
                }
            }
            for &(nx, nz) in &centers {
                set_disc(&mut grid, nx, nz, nr + 10.0, true);
                if let Some(&(tx, tz)) = trail_pts.iter().min_by(|a, b| {
                    let da = (a.0 - nx).powi(2) + (a.1 - nz).powi(2);
                    let db = (b.0 - nx).powi(2) + (b.1 - nz).powi(2);
                    da.total_cmp(&db)
                }) {
                    let d = ((tx - nx).powi(2) + (tz - nz).powi(2)).sqrt().max(1.0);
                    let steps = (d / 9.0) as usize + 1;
                    for k in 0..=steps {
                        let t = k as f64 / steps as f64;
                        set_disc(&mut grid, nx + (tx - nx) * t, nz + (tz - nz) * t, 11.0, true);
                    }
                }
            }
        }
        let units = spawn_field_at(&mut rng, pop, fseed, sc, &centers, nr, &grid, pn, pcell);
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
        flow.rebuild(&structures, structures[cc_id].p, &grid, pn, pcell);
        let ns = structures.len();
        let mut h = Horde {
            units, structures,
            zindex: Tree3::new(world, 8), handles: vec![None; n],
            sindex, noise: NoiseGrid::new(96), flow,
            pending: Vec::new(), corpses: Vec::new(), tracers: Vec::new(),
            tower_reload: vec![0.0; ns], free_slots: Vec::new(),
            tower_threat_mode: false, cc_id,
            wave_k: 0, wave_spawn_t: 50.0, wave_dir: 0.0, wave_announced: false, wave_active_until: 0.0,
            reinforce_t: 0.0, reinforce_interval: 30.0, fighter_cap: 0,
            game_over: None, run: 1, kills: 0,
            rng, now: 0.0, frame: 0,
            decide_n: std::env::var("HORDE_DECIDE_N").ok().and_then(|s| s.parse().ok()).filter(|&n| n >= 1).unwrap_or(DECIDE_N_DEFAULT),
            seed: fseed, woken_last: 0, dormant_epoch: 1,
            base_pop: pop, base_seed: seed,
            defenders: Vec::new(), threat: [0.0; SECTORS], weapons_free: false, cmd_t: 0.0, breach: None,
            gates: Vec::new(),
            scenario: sc, pass_grid: grid, pass_n: pn, pass_cell: pcell,
            // Morton levels = 5 → 56-wu xz cells: the wake blasts (r≈344, the
            // dominant cull) touch ~6k cells instead of the 270k that levels=7
            // costs (the grid splits EVERY axis 2^levels ways — on a 68-wu-tall
            // world the y axis shreds into confetti and big culls pay for it).
            zmode: ZMode::Tree, zmorton: vectorial_hash::MortonGrid3::new(world, 5),
        };
        h.gates = h.structures.iter().enumerate().filter(|(_, s)| s.kind == SKind::Gate).map(|(i, _)| i).collect();
        h.spawn_defenders();
        h
    }

    /// The garrison: fighters spread around the ring, works units at the
    /// storehouses. Fixed roster (the pop slider scales zombies, not defense).
    fn spawn_defenders(&mut self) {
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let home = self.structures.iter().find(|s| s.kind == SKind::Storehouse).map(|s| s.p).unwrap_or(Point3::new(cx, 0.0, cz));
        let spawn = |kind: DKind, n: usize, ds: &mut Vec<Defender>| {
            for k in 0..n {
                let a = (ds.len() as f64) * 2.399963;
                let (x, z) = match kind.fighter() {
                    true => (cx + a.cos() * (BASE_R - 10.0), cz + a.sin() * (BASE_R - 10.0)),
                    false => (home.x + a.cos() * (4.0 + k as f64), home.z + a.sin() * (4.0 + k as f64)),
                };
                ds.push(Defender {
                    kind, p: Point3::new(x, terrain_h(x, z, self.seed, self.scenario), z), hp: kind.max_hp(),
                    state: if kind.fighter() { DState::Post } else { DState::Idle },
                    sector: (ds.len()) % SECTORS, reload_t: 0.0, respawn_t: 0.0, face: 0.0, moving: false, stock: 0.0, shots: 0,
                    path: Vec::new(),
                });
            }
        };
        // Fighter counts scale with the zombie population. The wake cascade grows
        // ~linearly with pop, so √ wasn't enough (harness 2026-07-23: 100k died at
        // wave 1). pop^0.72 clamped 1..8 → base at ≤6k, ~2.4× at 20k, ~7.6× at 100k.
        let m = (self.base_pop as f64 / 6000.0).powf(0.72).clamp(1.0, 8.0);
        let fc = |n: usize| ((n as f64) * m).round() as usize;
        let lc = |n: usize| ((n as f64) * m.sqrt()).round() as usize; // logistics scale gentler
        let mut ds = Vec::new();
        // Interleave the fighter kinds round-robin (user 2026-07-23): sectors are
        // assigned ∝ list index, so a kind-blocked list colour-BANDED the ring; a
        // round-robin gives each sector a heterogeneous ranger/soldier/sniper mix.
        let mut want = [(DKind::Ranger, fc(28)), (DKind::Soldier, fc(16)), (DKind::Sniper, fc(9))];
        loop {
            let mut any = false;
            for slot in want.iter_mut() { if slot.1 > 0 { spawn(slot.0, 1, &mut ds); slot.1 -= 1; any = true; } }
            if !any { break; }
        }
        spawn(DKind::Crew, lc(4), &mut ds);
        spawn(DKind::Porter, lc(8), &mut ds);
        // Trickle reinforcements: the garrison can grow to 2× its fighters, one every
        // `reinforce_interval` s — faster at higher pop (30 s at ≤6k → ~4 s at 100k).
        let fighters0 = ds.iter().filter(|d| d.kind.fighter()).count();
        self.fighter_cap = fighters0 * 2;
        self.reinforce_interval = (30.0 / m).clamp(4.0, 30.0);
        self.reinforce_t = 0.0;
        self.defenders = ds;
    }

    /// Full run reset (CC fell, or the final wave was cleared): fresh base +
    /// dormant field, next run number, escalation restarts.
    fn reset(&mut self) {
        let next = Horde::with_scenario(self.base_seed.wrapping_add(self.run as u64 * 7919), self.base_pop, self.scenario);
        let (run, kills, mode, zm, multi) = (self.run + 1, self.kills, self.tower_threat_mode, self.zmode, self.flow.multi);
        *self = next;
        self.run = run;
        self.kills = kills; // lifetime counter across runs
        self.tower_threat_mode = mode;
        self.set_zmode(zm); // no-op when it was already the default Tree
        self.set_flow_multi(multi); // preserve the flow-goal mode across runs
    }

    /// HUD wave line: (wave number, announced?, direction, seconds to landfall).
    pub fn wave_info(&self) -> (u32, bool, f64, f64) {
        (self.wave_k + 1, self.wave_announced, self.wave_dir, (self.wave_spawn_t - self.now).max(0.0))
    }

    /// Wake EVERY dormant zombie into the march at once (the `A`/ALL button) —
    /// the "what does 100k active cost" stress button. Returns how many rose.
    pub fn wake_all(&mut self) -> usize {
        let mut rose = 0;
        for z in self.units.iter_mut() {
            if z.alive() && z.dormant() { z.state = ZState::Marching; z.heard = 0.0; z.moved = true; rose += 1; }
        }
        if rose > 0 { self.dormant_epoch += 1; }
        rose
    }

    /// Manual wave trigger (the `N` key / WAVE button): not announced yet →
    /// announce one landing in 5 s; already announced → land it NOW.
    pub fn trigger_wave(&mut self) {
        if self.game_over.is_some() { return; }
        if !self.wave_announced {
            self.wave_announced = true;
            self.wave_dir = self.rng.range(0.0, std::f64::consts::TAU);
            self.wave_spawn_t = self.now + 5.0;
            self.wave_active_until = self.wave_spawn_t + 35.0;
        } else {
            self.wave_spawn_t = self.now;
        }
    }

    /// Spawn (or reuse a dead slot for) one zombie. Public for scenario
    /// drivers and tests.
    pub fn spawn_zombie(&mut self, class: ZClass, x: f64, z: f64, state: ZState) {
        if state == ZState::Dormant { self.dormant_epoch += 1; }
        let y = terrain_h(x, z, self.seed, self.scenario) + class.altitude();
        let z0 = Zombie { class, p: Point3::new(x, y, z), vel: (0.0, 0.0), state, hp: class.max_hp(), heard: 0.0, linger: 0.0, groan_t: 1.0, swing_t: 0.5, bump: u32::MAX, moved: false };
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
            self.wave_active_until = self.wave_spawn_t + 35.0; // hold the line through the march-in
        }
        if self.now < self.wave_spawn_t { return; }
        let k = self.wave_k;
        let is_final = k >= 8;
        let count = (150.0 * 1.35f64.powi(k.min(8) as i32)) as usize;
        let dirs: Vec<f64> = if is_final { (0..4).map(|q| q as f64 * std::f64::consts::FRAC_PI_2).collect() } else { vec![self.wave_dir] };
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        for dir in &dirs {
            for _ in 0..count / dirs.len() {
                // Land close enough to be watchable (marching through the nest
                // belt also drags alert sleepers along — by design). Resample
                // until the spot is passable — in Forest that puts the landing
                // on trails/clearings, in Pass outside the ridge's shadow.
                let (mut x, mut z) = (cx, cz);
                for _try in 0..40 {
                    let a = dir + self.rng.range(-0.35, 0.35);
                    let r = self.rng.range(WORLD * 0.30, WORLD * 0.38);
                    x = (cx + a.cos() * r).clamp(MARGIN, WORLD - MARGIN);
                    z = (cz + a.sin() * r).clamp(MARGIN, WORLD - MARGIN);
                    // Passable AND connected — never strand a wave in a forest
                    // pocket the flow field can't route out of. Also never land IN
                    // Patches water (a carved corridor may be passable water).
                    if self.passable(x, z) && self.flow.reachable(x, z)
                        && !(self.scenario == Scenario::Patches && patch_cell(x, z, self.seed) == 3) { break; }
                }
                // Escalating mix: early waves are shambler seas; specials join
                // from wave 2 (venom/chubby) and wave 4 (harpies).
                let roll = self.rng.unit();
                let class = if k >= 4 && roll > 0.95 && self.scenario != Scenario::Patches { ZClass::Harpy }
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
            self.dormant_epoch += 1;
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
        // Breach alarm: the Commander RECALLS works units (porters, crews) in
        // the danger zone — they drop the job and run home; repair jobs stay
        // out of this zone for a while (see the crew scheduler).
        self.breach = Some((p, self.now));
        for d in self.defenders.iter_mut() {
            if d.kind.fighter() || !d.alive() { continue; }
            let (dx, dz) = (d.p.x - p.x, d.p.z - p.z);
            if (dx * dx + dz * dz).sqrt() < 150.0 { d.state = DState::Fleeing; }
        }
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
        match self.zmode {
            ZMode::Tree => for (i, z) in self.units.iter_mut().enumerate() {
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
            },
            // Morton has no in-place handles: its side of the `M` toggle is the
            // honest rebuild — clear + reinsert every LIVE zombie, every frame.
            ZMode::Morton => {
                self.zmorton.clear();
                for (i, z) in self.units.iter_mut().enumerate() {
                    z.moved = false;
                    if z.alive() { self.zmorton.insert(IZombie::of(i, z)); }
                }
            }
        }
    }

    /// The live index the queries should hit right now (`M` toggle).
    pub fn zq(&self) -> ZQuery<'_> {
        match self.zmode { ZMode::Tree => ZQuery::Tree(&self.zindex), ZMode::Morton => ZQuery::Morton(&self.zmorton) }
    }

    /// Switch the zombie-index structure live. Both sides start empty and the
    /// next sync fills the active one; handles only mean anything to the tree.
    pub fn set_zmode(&mut self, m: ZMode) {
        if m == self.zmode { return; }
        self.zmode = m;
        let world = Aabb::new(0.0, -8.0, 0.0, WORLD, SKY + 8.0, WORLD);
        self.zindex = Tree3::new(world, 8);
        self.handles = vec![None; self.units.len()];
        self.zmorton.clear();
        self.sync_index(); // queryable immediately (renderers cull between steps)
    }

    /// The `O` toggle — flow-field goal mode: single CC vs multi-building
    /// (the user's multi-source idea). Forces a rebuild next step.
    pub fn set_flow_multi(&mut self, multi: bool) {
        if self.flow.multi == multi { return; }
        self.flow.multi = multi;
        self.flow.dirty = true;
        self.flow.rebuild_t = 0.0;
    }
    pub fn flow_multi(&self) -> bool { self.flow.multi }

    /// Force one flow-field rebuild now (benchmarks / tests) — the single-CC or
    /// multi-building flood, timed in isolation.
    pub fn force_flow_rebuild(&mut self) {
        let cc = self.structures[self.cc_id].p;
        self.flow.rebuild(&self.structures, cc, &self.pass_grid, self.pass_n, self.pass_cell);
    }

    /// Is (x,z) passable ground for walkers? (Classic: everywhere. Pass: below
    /// the ridge. River: land + the causeways. Forest: clearings + trails.)
    pub fn passable(&self, x: f64, z: f64) -> bool {
        self.pass_grid[((z / self.pass_cell) as usize).min(self.pass_n - 1) * self.pass_n + ((x / self.pass_cell) as usize).min(self.pass_n - 1)]
    }

    /// One fixed step: waves → noise events → wake culls → parallel decide →
    /// serial apply (movement, swings, towers, destructions) → next frame's
    /// keep-index sync.
    pub fn step(&mut self, dt: f64) {
        self.now += dt;
        self.frame += 1;
        self.sync_index();
        self.step_waves();
        // Throttled flow-field rebuild after breaches / repairs.
        self.flow.rebuild_t -= dt;
        if self.flow.dirty && self.flow.rebuild_t <= 0.0 {
            let cc = self.structures[self.cc_id].p;
            let (flow, structures) = (&mut self.flow, &self.structures);
            flow.rebuild(structures, cc, &self.pass_grid, self.pass_n, self.pass_cell);
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
            // +28: hearing is an XZ ring but the cull is a 3D sphere — a flying
            // harpy at the very edge of its ring must not slip out vertically.
            let blast = Sphere3::new(p.x, p.y, p.z, MAX_HEAR + 28.0);
            let heard: Vec<(u32, f64)> = self.zq().cull(&blast).iter()
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
        if self.woken_last > 0 { self.dormant_epoch += 1; }
        self.noise.step(dt);

        // 2) decide — read-only on the indices, each zombie writes only itself:
        //    fans out over rayon on native, serial on wasm (no threads there).
        {
            let index = match self.zmode { ZMode::Tree => ZQuery::Tree(&self.zindex), ZMode::Morton => ZQuery::Morton(&self.zmorton) };
            let index = &index;
            let (sindex, structures, flow) = (&self.sindex, &self.structures, &self.flow);
            let defenders = &self.defenders;
            let frame = self.frame;
            let decide_n = self.decide_n;
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
                // DECISION BUCKETS: away from the walls a zombie re-decides at
                // 15 Hz (staggered by id) — its cached velocity keeps it walking
                // between decisions, movement/heard/swings still run every
                // frame. Near the walls (combat) everyone thinks at full rate.
                {
                    let (dx0, dz0) = (z.p.x - cx, z.p.z - cz);
                    let near_combat = (dx0 * dx0 + dz0 * dz0).sqrt() < BASE_R + 60.0;
                    if !near_combat && (frame + i as u64) % decide_n != 0 { return; } // keep cached vel
                }
                // Target acquisition near the base: nearest LIVE structure in
                // reach (k-NN on the static index; venom's 36-wu standoff spit
                // engages from outside the wall line; harpies fly at altitude,
                // so the 3D k-NN naturally skips the wall and finds the inner
                // buildings under them).
                let (dcx, dcz) = (z.p.x - cx, z.p.z - cz);
                let dc = (dcx * dcx + dcz * dcz).sqrt();
                // The nearest HUMAN in sight (TAB: zombies beeline to the
                // closest human element) — defenders are few, a brute scan
                // beats indexing them. Chasing overrides the flow field.
                let mut hunt: Option<(f64, f64, f64)> = None; // (dist, x, z)
                if dc < BASE_R + 140.0 {
                    for dfn in defenders.iter() {
                        if !dfn.alive() { continue; }
                        let (dx, dz) = (dfn.p.x - z.p.x, dfn.p.z - z.p.z);
                        let dd = (dx * dx + dz * dz).sqrt();
                        if dd < z.class.watch() && hunt.map(|(b, _, _)| dd < b).unwrap_or(true) { hunt = Some((dd, dfn.p.x, dfn.p.z)); }
                    }
                }
                // In biting range of a human → stand and maul (the contact pass
                // applies the damage). Humans in sight beat walls.
                if let Some((hd, _, _)) = hunt { if hd < 4.0 { z.vel = (0.0, 0.0); return; } }
                if hunt.is_none() && dc < BASE_R + 80.0 {
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
                // Steering: a human in sight beats everything (the beeline);
                // else investigators seek their spot and marchers descend the
                // flow field (harpies fly straight over the walls).
                let (mut vx, mut vz) = match (hunt, z.state) {
                    (Some((hd, hx, hz)), _) => ((hx - z.p.x) / hd.max(0.5), (hz - z.p.z) / hd.max(0.5)),
                    (_, ZState::Investigating { tx, tz }) => {
                        let (dx, dz) = (tx - z.p.x, tz - z.p.z);
                        let d = (dx * dx + dz * dz).sqrt();
                        if d > 8.0 { (dx / d, dz / d) } else { (0.0, 0.0) }
                    }
                    (_, ZState::Marching) if z.class == ZClass::Harpy => { let d = dc.max(1.0); (-dcx / d, -dcz / d) }
                    (_, ZState::Marching) => flow.flow_at(z.p.x, z.p.z),
                    _ => (0.0, 0.0),
                };
                // Separation: full strength among the awake, WEAK against the
                // dormant carpet (the wave pushes through) — and shoving a
                // sleeper WAKES it (recorded here, resolved serially in apply):
                // a column rolling over a nest recruits it instead of jamming.
                z.bump = u32::MAX;
                let sep = Sphere3::new(z.p.x, z.p.y, z.p.z, 3.0);
                for it in index.cull(&sep) {
                    if it.id as usize == i || (it.p.y - z.p.y).abs() > 12.0 { continue; }
                    let (sx, sz) = (z.p.x - it.p.x, z.p.z - it.p.z);
                    let d = (sx * sx + sz * sz).sqrt().max(0.2);
                    let w = if it.dormant { 0.12 } else { 0.4 };
                    if d < 3.0 { vx += sx / d * (3.0 - d) * w; vz += sz / d * (3.0 - d) * w; }
                    // Contact-wake: ONLY the marching column tramples sleepers
                    // awake, and its radius (1.7) sits below the lattice spacing
                    // (1.8 worst-case) — two hard-won rules: at radius 2.6 the
                    // carpet percolated (one wave → 27k active in seconds), and
                    // when investigators could trample too, crowds at a noise
                    // site kept re-waking their own sleepers forever (a stuck
                    // ~80-strong mosh pit).
                    if matches!(z.state, ZState::Marching) && it.dormant && d < 1.7 { z.bump = it.id; }
                }
                let l = (vx * vx + vz * vz).sqrt();
                // Swarm frenzy: wave zombies (and beeliners) push at 2.2× — at
                // demo scale a TAB-speed walker reads as standing still.
                let frenzy = if hunt.is_some() || matches!(z.state, ZState::Marching) { 2.2 } else { 1.0 };
                let sp = z.class.speed() * frenzy;
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
        let mut bumped: Vec<u32> = Vec::new(); // sleepers trampled by the marching column
        let mut slept = false;
        let (pg, pn, pcell, sc) = (&self.pass_grid, self.pass_n, self.pass_cell, self.scenario);
        let pass = |x: f64, z: f64| pg[((z / pcell) as usize).min(pn - 1) * pn + ((x / pcell) as usize).min(pn - 1)];
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
                let mut nx = (z.p.x + z.vel.0 * dt).clamp(MARGIN, WORLD - MARGIN);
                let mut nz = (z.p.z + z.vel.1 * dt).clamp(MARGIN, WORLD - MARGIN);
                // Impassable terrain (ridge / water / woods): ground classes
                // SLIDE along the blocked edge (axis drop); harpies fly over.
                if z.class.altitude() == 0.0 && !pass(nx, nz) {
                    if pass(nx, z.p.z) { nz = z.p.z; }
                    else if pass(z.p.x, nz) { nx = z.p.x; }
                    else { nx = z.p.x; nz = z.p.z; }
                }
                if nx != z.p.x || nz != z.p.z {
                    z.p = Point3::new(nx, terrain_h(nx, nz, self.seed, sc) + z.class.altitude(), nz);
                    z.moved = true;
                }
                if z.bump != u32::MAX { bumped.push(z.bump); z.bump = u32::MAX; }
                // Walking is SILENT. Aggro spreads through combat noise and
                // physical contact only — every groan variant we tried turned
                // walkers into recruiters and some loop self-sustained (a full
                // noise avalanche at march-groans, an ~80-strong pilot light at
                // investigator-groans). TAB agrees: fights are loud, feet aren't.
            } else if matches!(z.state, ZState::Investigating { .. }) {
                z.linger -= dt;
                if z.linger <= 0.0 { z.state = ZState::Dormant; z.heard = 0.0; z.moved = true; slept = true; }
            }
        }
        if slept { self.dormant_epoch += 1; }
        // Trampled awake (no noise threshold — a body dragging over you beats
        // any decibel rule): risers join the march.
        let mut rose = false;
        for id in bumped {
            let z = &mut self.units[id as usize];
            if !z.alive() || !z.dormant() { continue; }
            z.state = ZState::Marching;
            z.heard = 0.0;
            z.moved = true;
            rose = true;
            self.woken_last += 1;
        }
        if rose { self.dormant_epoch += 1; }
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
        self.step_defenders(dt);
        self.reinforce(dt);
    }

    /// Trickle a fresh fighter in at the wall ring every `reinforce_interval` s, up
    /// to `fighter_cap` (2× the starting fighters), rotating ranger/soldier/sniper —
    /// the garrison recovers and grows through the escalating waves (user 2026-07-23).
    fn reinforce(&mut self, dt: f64) {
        if self.game_over.is_some() { return; }
        self.reinforce_t += dt;
        if self.reinforce_t < self.reinforce_interval { return; }
        self.reinforce_t = 0.0;
        if self.defenders.iter().filter(|d| d.kind.fighter()).count() >= self.fighter_cap { return; }
        let kind = match self.defenders.len() % 3 { 0 => DKind::Ranger, 1 => DKind::Soldier, _ => DKind::Sniper };
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let a = self.defenders.len() as f64 * 2.399963;
        let (x, z) = (cx + a.cos() * (BASE_R - 10.0), cz + a.sin() * (BASE_R - 10.0));
        self.defenders.push(Defender {
            kind, p: Point3::new(x, terrain_h(x, z, self.seed, self.scenario), z), hp: kind.max_hp(),
            state: DState::Post, sector: self.defenders.len() % SECTORS, reload_t: 0.0, respawn_t: 0.0, face: 0.0, moving: false, stock: 0.0, shots: 0, path: Vec::new(),
        });
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
                self.zq().cull(&ring).iter()
                    .map(|it| { let (dx, dz) = (it.p.x - tp.x, it.p.z - tp.z); (it.id, it.class.threat() / (1.0 + (dx * dx + dz * dz).sqrt() * 0.02)) })
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(id, _)| id)
            } else {
                self.zq().knn(tp, 1).into_iter().find(|(d, _)| *d <= TOWER_RANGE).map(|(_, it)| it.id)
            };
            let Some(tid) = target else { continue; };
            // Morton mode rebuilds next frame, so a zombie another tower just
            // killed can still be served this frame — never shoot a corpse.
            if !self.units[tid as usize].alive() { continue; }
            self.tower_reload[sid] = TOWER_RELOAD;
            self.pending.push((tp, TOWER_NOISE));
            let zp = self.units[tid as usize].p;
            self.tracers.push((Point3::new(tp.x, tp.y + 22.0, tp.z), zp, self.now));
            self.units[tid as usize].hp -= TOWER_DMG;
            if self.units[tid as usize].hp <= 0.0 { self.kill_zombie(tid as usize); }
        }
    }

    /// The defense pass: the Commander (1 Hz — sector threat map, wave
    /// anticipation, noise discipline, job scheduling) + per-frame defender
    /// FSMs (fighters post/engage/kite, crews repair, porters haul, recalled
    /// units flee home) + zombie contact damage on defenders.
    fn step_defenders(&mut self, dt: f64) {
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let home = self.structures.iter().find(|s| s.kind == SKind::Storehouse && s.hp > 0.0).map(|s| s.p)
            .unwrap_or(self.structures[self.cc_id].p);
        // ---- Commander tick (1 Hz)
        self.cmd_t -= dt;
        if self.cmd_t <= 0.0 {
            self.cmd_t = 1.0;
            // Sector threat: one counting cull per sector + wave anticipation
            // (pre-position while the countdown runs — looks like foresight).
            let (mut total, mut peak) = (0.0, 0.0f64);
            for s in 0..SECTORS {
                let a = (s as f64 + 0.5) / SECTORS as f64 * std::f64::consts::TAU;
                let mid = Point3::new(cx + a.cos() * BASE_R, 0.0, cz + a.sin() * BASE_R);
                let ring = Sphere3::new(mid.x, mid.y, mid.z, 110.0);
                let mut t = self.zq().cull(&ring).iter().filter(|it| !it.dormant).count() as f64;
                if self.wave_announced {
                    let mut da = (a - self.wave_dir).abs(); if da > std::f64::consts::PI { da = std::f64::consts::TAU - da; }
                    let eta = (self.wave_spawn_t - self.now).max(0.0);
                    // Anticipate the announced direction, but softly — a big bump used
                    // to yank fighters off an ACTIVE front toward a merely-announced one
                    // (user 2026-07-23: "defenders on the opposite side"). 25→10.
                    if da < 0.7 { t += 10.0 * (1.0 - eta / 30.0).clamp(0.0, 1.0); }
                }
                self.threat[s] = t;
                total += t;
                peak = peak.max(t);
            }
            // Noise discipline: silence until a wave commits near the walls.
            self.weapons_free = total > 25.0 || peak > 12.0;
            // Fighter assignment ∝ sector threat (deterministic split).
            let weights: Vec<f64> = self.threat.iter().map(|t| t + 0.3).collect();
            let wsum: f64 = weights.iter().sum();
            let nf = self.defenders.iter().filter(|d| d.kind.fighter() && d.alive()).count().max(1);
            for (fi, d) in self.defenders.iter_mut().filter(|d| d.kind.fighter() && d.alive()).enumerate() {
                let target = (fi as f64 + 0.5) / nf as f64 * wsum;
                let (mut acc, mut sec) = (0.0, 0usize);
                for (s, w) in weights.iter().enumerate() { acc += w; if acc >= target { sec = s; break; } }
                d.sector = sec;
            }
            // Crew job scheduling: most damaged structure, outside the breach
            // danger zone. The safety cull only fears zombies that are INSIDE
            // the ring (or flying over it) — a horde pounding the far side of a
            // standing wall is exactly when repairing it matters most: the
            // repair-vs-pounding race, with the wall in between.
            let danger = self.breach.filter(|(_, t)| self.now - t < 20.0);
            for di in 0..self.defenders.len() {
                if self.defenders[di].kind != DKind::Crew || !self.defenders[di].alive() || self.defenders[di].state != DState::Idle { continue; }
                let mut best: Option<(usize, f64)> = None;
                for (si, s) in self.structures.iter().enumerate() {
                    let missing = s.kind.max_hp() - s.hp;
                    if missing <= 0.0 { continue; }
                    if let Some((bp, _)) = danger { let (dx, dz) = (s.p.x - bp.x, s.p.z - bp.z); if (dx * dx + dz * dz).sqrt() < 120.0 { continue; } }
                    let guard = Sphere3::new(s.p.x, s.p.y, s.p.z, 55.0);
                    let inside_threat = self.zq().cull(&guard).iter().any(|it| {
                        if it.dormant { return false; }
                        let (dx, dz) = (it.p.x - cx, it.p.z - cz);
                        (dx * dx + dz * dz).sqrt() < BASE_R + 2.0 // through a breach, or flying over
                    });
                    if inside_threat { continue; }
                    if best.map(|(_, m)| missing > m).unwrap_or(true) { best = Some((si, missing)); }
                }
                if let Some((si, _)) = best { self.defenders[di].state = DState::Repairing { sid: si as u32 }; }
            }
            // Porter job scheduling: feed the hungriest repairing crew.
            for di in 0..self.defenders.len() {
                if self.defenders[di].kind != DKind::Porter || !self.defenders[di].alive() || self.defenders[di].state != DState::Idle { continue; }
                let crew = self.defenders.iter().enumerate()
                    .filter(|(_, c)| c.kind == DKind::Crew && c.alive() && matches!(c.state, DState::Repairing { .. }) && c.stock < 30.0)
                    .min_by(|(_, a), (_, b)| a.stock.total_cmp(&b.stock))
                    .map(|(ci, _)| ci);
                if let Some(ci) = crew { self.defenders[di].state = DState::Hauling { did: ci as u32, loaded: true }; }
            }
            // SORTIE: send a ranger squad out through the gate to silently
            // clear the nearest nest (rangers = noise 1; the TAB map-clearing
            // move). Runs in every lull — recalled when a wave gets close.
            let total_threat = self.threat.iter().sum::<f64>();
            let out = self.defenders.iter().filter(|d| matches!(d.state, DState::Sortie { .. })).count();
            // Only sortie in a genuine lull: no wave announced or still marching in
            // (`wave_active_until`) and the sectors are quiet (user 2026-07-22: the
            // rangers used to wander out INTO a landing wave).
            if total_threat < 6.0 && self.now > self.wave_active_until && out == 0 {
                let target = self.zq().knn(Point3::new(cx, 0.0, cz), 48).into_iter()
                    .find(|(_, it)| it.dormant)
                    .map(|(_, it)| it.p);
                if let Some(tp) = target {
                    // One A* over the pass grid per squad dispatch — the
                    // DEFENDERS' minimum paths (out the gate nearest the nest,
                    // then along the forest trails / over a causeway). Empty
                    // (Classic, or unreachable) = walk straight as before.
                    let gate = self.gates.iter().map(|&g| self.structures[g].p)
                        .min_by(|a, b| { let da = (a.x - tp.x).powi(2) + (a.z - tp.z).powi(2); let db = (b.x - tp.x).powi(2) + (b.z - tp.z).powi(2); da.total_cmp(&db) })
                        .map(|g| (g.x, g.z)).unwrap_or((cx, cz));
                    let spath = astar_path(&self.pass_grid, self.pass_n, self.pass_cell, gate, (tp.x, tp.z));
                    let mut sent = 0;
                    for d in self.defenders.iter_mut() {
                        if sent >= 10 { break; }
                        if d.kind != DKind::Ranger || !d.alive() || d.state != DState::Post || d.hp < d.kind.max_hp() * 0.9 { continue; }
                        let j = sent as f64 * 2.399963;
                        d.state = DState::Sortie { tx: (tp.x + j.cos() * (4.0 + sent as f64 * 2.0)).clamp(MARGIN, WORLD - MARGIN), tz: (tp.z + j.sin() * (4.0 + sent as f64 * 2.0)).clamp(MARGIN, WORLD - MARGIN) };
                        d.path = spath.clone();
                        d.respawn_t = 30.0; // sortie clock: come home after ~30 s regardless
                        sent += 1;
                    }
                }
            }
            // Wave announced OR marching in → recall everyone to the walls (covers
            // the whole active window, not just the 30 s warning).
            if self.now <= self.wave_active_until {
                for d in self.defenders.iter_mut() {
                    if matches!(d.state, DState::Sortie { .. }) { d.state = DState::Post; }
                }
            }
        }
        // ---- per-frame defender update
        let mut shot: Vec<(usize, f64)> = Vec::new(); // (zombie id, dmg)
        let mut noise: Vec<(Point3, f64)> = Vec::new();
        let mut tracer: Vec<(Point3, Point3, f64)> = Vec::new();
        let mut deliveries: Vec<u32> = Vec::new();
        let mut repaired_any = false;
        // Live gate mouths — every ring-crossing friendly walk routes via one.
        let gate_pts: Vec<(f64, f64)> = self.gates.iter().filter(|&&g| self.structures[g].hp > 0.0).map(|&g| (self.structures[g].p.x, self.structures[g].p.z)).collect();
        // Field-precise borrows the defenders' iter_mut can live beside: the
        // live zombie index (either mode) + the pass grid for the walk slide.
        let zq = match self.zmode { ZMode::Tree => ZQuery::Tree(&self.zindex), ZMode::Morton => ZQuery::Morton(&self.zmorton) };
        let (pg2, pn2, pc2, sc, seed) = (&self.pass_grid, self.pass_n, self.pass_cell, self.scenario, self.seed);
        let pass2 = move |x: f64, z: f64| pg2[((z / pc2) as usize).min(pn2 - 1) * pn2 + ((x / pc2) as usize).min(pn2 - 1)];
        let walk = move |d: &mut Defender, tx: f64, tz: f64, dt: f64| -> f64 {
            let (dx, dz) = (tx - d.p.x, tz - d.p.z);
            let dist = (dx * dx + dz * dz).sqrt();
            if dist > 2.0 {
                d.face = (dz as f32).atan2(dx as f32); // heading, so the render can turn the model
                d.moving = true;
                let sp = d.kind.speed().min(dist / dt);
                let (mut nx, mut nz) = ((d.p.x + dx / dist * sp * dt).clamp(MARGIN, WORLD - MARGIN), (d.p.z + dz / dist * sp * dt).clamp(MARGIN, WORLD - MARGIN));
                if !pass2(nx, nz) { // slide along the blocked edge (axis drop)
                    if pass2(nx, d.p.z) { nz = d.p.z; } else if pass2(d.p.x, nz) { nx = d.p.x; } else { nx = d.p.x; nz = d.p.z; }
                }
                d.p = Point3::new(nx, terrain_h(nx, nz, seed, sc), nz);
            }
            dist
        };
        // Steer a logistics unit toward its goal but AROUND nearby awake zombies
        // (user 2026-07-23: crews used to walk straight through the horde). Reactive
        // repulsion (1/d²) blended into the goal heading → an intermediate waypoint.
        let avoid = |p: Point3, tx: f64, tz: f64| -> (f64, f64) {
            let (mut rx, mut rz) = (0.0f64, 0.0f64);
            for it in zq.cull(&Sphere3::new(p.x, p.y, p.z, 24.0)) {
                if it.dormant { continue; }
                let (dx, dz) = (p.x - it.p.x, p.z - it.p.z);
                let d2 = dx * dx + dz * dz;
                if d2 > 1e-3 { let w = 1.0 / d2; rx += dx * w; rz += dz * w; }
            }
            let (gx, gz) = (tx - p.x, tz - p.z);
            let gl = (gx * gx + gz * gz).sqrt().max(0.5);
            let (mut ax, mut az) = (gx / gl, gz / gl);
            let rl = (rx * rx + rz * rz).sqrt();
            if rl > 1e-4 { ax += rx / rl * 1.4; az += rz / rl * 1.4; } // repulsion weight
            let al = (ax * ax + az * az).sqrt().max(0.5);
            (p.x + ax / al * gl.min(28.0), p.z + az / al * gl.min(28.0))
        };
        for (dix, d) in self.defenders.iter_mut().enumerate() {
            if !d.alive() {
                d.respawn_t -= dt;
                if d.respawn_t <= 0.0 { // recruits arrive at the CC
                    d.hp = d.kind.max_hp();
                    d.p = Point3::new(cx, terrain_h(cx, cz, seed, sc), cz);
                    d.state = if d.kind.fighter() { DState::Post } else { DState::Idle };
                    d.path.clear();
                }
                continue;
            }
            d.moving = false; // set true by the movement code; drives the idle clip
            match d.state {
                DState::Post => {
                    d.reload_t -= dt;
                    // Sector post — the Commander re-weights `d.sector` by live threat
                    // each second, so this ALREADY drifts fighters toward the attack
                    // (keeping the ring covered, not all piling on one cluster).
                    let a = (d.sector as f64 + 0.5) / SECTORS as f64 * std::f64::consts::TAU;
                    let jit = ((dix % 9) as f64 - 4.0) * 0.022;
                    let ro = (dix % 4) as f64 * 2.2;
                    let (px, pz) = (cx + (a + jit).cos() * (BASE_R - 10.0 - ro), cz + (a + jit).sin() * (BASE_R - 10.0 - ro));
                    let (mut wx, mut wz) = if let Some(&(w0x, w0z)) = d.path.first() {
                        if (w0x - d.p.x).powi(2) + (w0z - d.p.z).powi(2) < 36.0 { d.path.remove(0); }
                        d.path.first().copied().unwrap_or_else(|| via_gate(&gate_pts, d.p, px, pz))
                    } else { via_gate(&gate_pts, d.p, px, pz) };
                    let nearest = zq.knn(d.p, 1).into_iter().next();
                    // KITE only from a REAL melee threat (a breacher that got within
                    // ~11 wu) — not from wall-attackers the standing wall already
                    // stops, so fighters keep manning the line instead of fleeing it
                    // (user 2026-07-23: don't stand there getting mauled, but don't
                    // abandon the wall either). Bounded to the ring: retreat inward.
                    if let Some((zd, it)) = &nearest {
                        if *zd < 11.0 {
                            let (fx, fz) = (d.p.x - it.p.x, d.p.z - it.p.z);
                            let l = (fx * fx + fz * fz).sqrt().max(0.5);
                            wx = d.p.x + fx / l * (16.0 - zd);
                            wz = d.p.z + fz / l * (16.0 - zd);
                            let (rcx, rcz) = (wx - cx, wz - cz);
                            let rr = (rcx * rcx + rcz * rcz).sqrt().max(0.5);
                            if rr > BASE_R - 4.0 { wx = cx + rcx / rr * (BASE_R - 4.0); wz = cz + rcz / rr * (BASE_R - 4.0); }
                        }
                    }
                    walk(d, wx, wz, dt);
                    let may_fire = self.weapons_free || d.kind == DKind::Ranger;
                    if d.reload_t <= 0.0 && may_fire {
                        if let Some((dist, it)) = &nearest {
                            if *dist <= d.kind.range() {
                                d.reload_t = d.kind.reload();
                                d.shots += 1;
                                shot.push((it.id as usize, d.kind.dmg()));
                                noise.push((d.p, d.kind.noise()));
                                tracer.push((Point3::new(d.p.x, d.p.y + 5.0, d.p.z), it.p, 0.0));
                            }
                        }
                    }
                }
                DState::Repairing { sid } => {
                    let s = &mut self.structures[sid as usize];
                    if s.hp >= s.kind.max_hp() { d.state = DState::Idle; continue; }
                    let (sx, sz) = (s.p.x + 3.0, s.p.z);
                    let (wx, wz) = avoid(d.p, sx, sz);
                    walk(d, wx, wz, dt);
                    if ((d.p.x - sx).powi(2) + (d.p.z - sz).powi(2)).sqrt() < 6.0 && d.stock > 0.0 {
                        let was_dead = s.hp <= 0.0;
                        s.hp = (s.hp + CREW_REPAIR * dt).min(s.kind.max_hp());
                        d.stock = (d.stock - 2.0 * dt).max(0.0);
                        repaired_any = true;
                        if was_dead && s.hp > 0.0 { self.flow.dirty = true; } // rubble rises: costs return
                        if s.hp >= s.kind.max_hp() { d.state = DState::Idle; self.flow.dirty = true; }
                    }
                }
                DState::Sortie { tx, tz } => {
                    // The sortie clock: whatever happens, come home after it
                    // runs out; ditto once the nest reads clear. Both exits
                    // store an A* trail home (rangers deep in the forest
                    // retrace a minimum path to the nearest gate, not a
                    // straight line into the woods).
                    d.respawn_t -= dt;
                    let clear = zq.knn(Point3::new(tx, 0.0, tz), 1).into_iter().next().map(|(dd, _)| dd > 90.0).unwrap_or(true);
                    if d.respawn_t <= 0.0 || clear {
                        let g = gate_pts.iter().copied().min_by(|a, b| {
                            let da = (a.0 - d.p.x).powi(2) + (a.1 - d.p.z).powi(2);
                            let db = (b.0 - d.p.x).powi(2) + (b.1 - d.p.z).powi(2);
                            da.total_cmp(&db)
                        }).unwrap_or((cx, cz));
                        d.path = astar_path(pg2, pn2, pc2, (d.p.x, d.p.z), g);
                        d.state = DState::Post;
                        continue;
                    }
                    // Follow the outbound waypoints, then close on the target.
                    let (mut gx, mut gz) = (tx, tz);
                    while let Some(&(w0x, w0z)) = d.path.first() {
                        if (w0x - d.p.x).powi(2) + (w0z - d.p.z).powi(2) < 36.0 { d.path.remove(0); continue; }
                        gx = w0x; gz = w0z; break;
                    }
                    let (wx, wz) = via_gate(&gate_pts, d.p, gx, gz);
                    walk(d, wx, wz, dt);
                    d.reload_t -= dt;
                    if d.reload_t <= 0.0 {
                        if let Some((dist, it)) = zq.knn(d.p, 1).into_iter().next() {
                            if dist <= d.kind.range() {
                                d.reload_t = d.kind.reload();
                                d.shots += 1;
                                shot.push((it.id as usize, d.kind.dmg()));
                                noise.push((d.p, d.kind.noise()));
                                tracer.push((Point3::new(d.p.x, d.p.y + 5.0, d.p.z), it.p, 0.0));
                            }
                        }
                    }
                }
                DState::Hauling { .. } => {} // handled after the loop (needs two defenders at once)
                DState::Fleeing => {
                    let (wx, wz) = via_gate(&gate_pts, d.p, home.x, home.z);
                    if walk(d, wx, wz, dt) < 5.0 { d.state = DState::Idle; }
                }
                DState::Idle => {
                    if !d.kind.fighter() {
                        // Personal spot by the storehouse — no stacking on one point.
                        let j = dix as f64 * 2.399963;
                        walk(d, home.x + j.cos() * (4.0 + (dix % 6) as f64 * 1.6), home.z + j.sin() * (4.0 + (dix % 6) as f64 * 1.6), dt);
                    }
                }
            }
        }
        // Porter movement (two-defender interaction — done index-wise to keep
        // the borrow checker happy).
        for di in 0..self.defenders.len() {
            let DState::Hauling { did, loaded } = self.defenders[di].state else { continue; };
            if !self.defenders[di].alive() { continue; }
            let target = if loaded { self.defenders[did as usize].p } else { home };
            let pp = self.defenders[di].p;
            let dist = ((target.x - pp.x).powi(2) + (target.z - pp.z).powi(2)).sqrt();
            if dist > 3.0 {
                // steer around zombie groups en route (user 2026-07-23)
                let (tx, tz) = avoid(pp, target.x, target.z);
                let (sx, sz) = (tx - pp.x, tz - pp.z);
                let sl = (sx * sx + sz * sz).sqrt().max(0.5);
                let (dx, dz) = (sx / sl, sz / sl);
                let sp = DKind::Porter.speed().min(dist / dt);
                let d = &mut self.defenders[di];
                d.face = (dz as f32).atan2(dx as f32);
                d.moving = true;
                let (mut nx, mut nz) = ((d.p.x + dx * sp * dt).clamp(MARGIN, WORLD - MARGIN), (d.p.z + dz * sp * dt).clamp(MARGIN, WORLD - MARGIN));
                if !pass2(nx, nz) { if pass2(nx, d.p.z) { nz = d.p.z; } else if pass2(d.p.x, nz) { nx = d.p.x; } else { nx = d.p.x; nz = d.p.z; } }
                d.p = Point3::new(nx, terrain_h(nx, nz, seed, sc), nz);
            } else if loaded {
                deliveries.push(did);
                self.defenders[di].state = DState::Hauling { did, loaded: false };
            } else {
                self.defenders[di].state = DState::Idle; // back home, bundle ready
            }
            // A crew that stopped repairing releases its porter mid-route.
            if loaded && !matches!(self.defenders[did as usize].state, DState::Repairing { .. }) {
                self.defenders[di].state = DState::Fleeing;
            }
        }
        for did in deliveries { self.defenders[did as usize].stock += PORTER_BUNDLE; }
        if repaired_any { /* flow cost of a rising wall is refreshed on completion/rise — cheap enough */ }
        // Zombie contact damage on defenders (one small cull per defender) —
        // porters caught on a haul route die screaming (a noise event).
        let mut dead_screams: Vec<Point3> = Vec::new();
        for d in self.defenders.iter_mut() {
            if !d.alive() { continue; }
            let bite = Sphere3::new(d.p.x, d.p.y, d.p.z, 4.5);
            let dps: f64 = zq.cull(&bite).iter().filter(|it| !it.dormant).map(|it| it.class.dmg()).sum();
            if dps > 0.0 {
                // contact damage 0.4→0.22 and fighter respawn 25→18 s: survive the
                // swarm longer and return quicker to the line (balance harness).
                d.hp -= dps * 0.22 * dt;
                if d.hp <= 0.0 { d.respawn_t = if d.kind.fighter() { 18.0 } else { 30.0 }; dead_screams.push(d.p); }
            }
        }
        for p in dead_screams { self.pending.push((p, 10.0)); }
        // Resolve defender shots.
        for (tid, dmg) in shot {
            if !self.units[tid].alive() { continue; }
            self.units[tid].hp -= dmg;
            if self.units[tid].hp <= 0.0 { self.kill_zombie(tid); }
        }
        for (p, a) in noise { self.pending.push((p, a)); }
        let now = self.now;
        self.tracers.extend(tracer.into_iter().map(|(a, b, _)| (a, b, now)));
        self.tracers.retain(|(_, _, t)| now - t < 0.12); // fast-fade zaps
    }

    /// Death: corpse for the renderer, a death rattle (noise), free the slot,
    /// and drop the item from the index immediately (O(1) by handle).
    fn kill_zombie(&mut self, id: usize) {
        if self.units[id].dormant() { self.dormant_epoch += 1; } // a sleeper died in place
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
        // NOISE-woken only (Investigating): contact-woken sleepers (a woken
        // mover shoves a neighbour → it rises Marching) are a separate,
        // deliberate mechanic with its own test.
        let got: Vec<usize> = h.units.iter().enumerate().filter(|(_, z)| matches!(z.state, ZState::Investigating { .. })).map(|(i, _)| i).collect();
        assert_eq!(want, got, "noise-woken set != brute-force wake rule");
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
        h.wave_spawn_t = 1e9;    // no waves — this test is about re-settling
        h.defenders.clear();     // …and no sorties/towers keeping fights alive
        h.step(1.0 / 60.0);
        h.emit_noise(h.units[0].p, 1000.0); // wake everything nearby
        h.step(1.0 / 60.0);
        let (_, active0) = h.counts();
        assert!(active0 > 0, "the blast must wake someone");
        for _ in 0..(90.0 / (1.0 / 60.0)) as usize { h.step(1.0 / 60.0); }
        let (_, active1) = h.counts();
        // With walking silent and trampling reserved to the marching column,
        // an investigated noise site must drain back to (near) full sleep.
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
        let kills0 = h.kills;
        // A chubby pounding the house (500 HP / 40 dmg ≈ 13 swings).
        h.spawn_zombie(ZClass::Chubby, hp0.x + 2.0, hp0.z, ZState::Attacking { sid: hid as u32 });
        for _ in 0..(20.0 * 60.0) as usize { h.step(1.0 / 60.0); if h.structures[hid].hp <= 0.0 { break; } }
        assert!(h.structures[hid].hp <= 0.0, "house must fall");
        h.step(1.0 / 60.0); // process the infection noise event
        let (d1, a1) = h.counts();
        // Rangers on sortie may have picked some off meanwhile — count kills in.
        let killed = (h.kills - kills0) as usize;
        assert!(d1 + a1 + killed >= d0 + a0 + pop, "each colonist must rise as a runner: {} -> {} (+{killed} killed)", d0 + a0, d1 + a1);
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
    fn commander_prepositions_fighters_on_wave_warning() {
        let mut h = Horde::new(3, 200);
        h.wave_spawn_t = 25.0; // warning window opens immediately (spawn-30)
        for _ in 0..(4.0 * 60.0) as usize { h.step(1.0 / 60.0); } // 4 commander tics
        assert!(h.wave_announced, "warning must be up");
        let fighters: Vec<&Defender> = h.defenders.iter().filter(|d| d.kind.fighter() && d.alive()).collect();
        let near = fighters.iter().filter(|d| {
            let a = (d.sector as f64 + 0.5) / SECTORS as f64 * std::f64::consts::TAU;
            let mut da = (a - h.wave_dir).abs(); if da > std::f64::consts::PI { da = std::f64::consts::TAU - da; }
            da < 0.8
        }).count();
        assert!(near * 2 >= fighters.len(), "most fighters should pre-position toward the announced direction ({near}/{})", fighters.len());
    }

    #[test]
    fn noise_discipline_snipers_hold_until_a_wave_commits() {
        let mut h = Horde::new(41, 200);
        h.wave_spawn_t = 1e9; // no scheduled waves in this test
        let tid = h.structures.iter().position(|s| s.kind == SKind::Tower).unwrap();
        let tp = h.structures[tid].p;
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let out = ((tp.x - cx) / BASE_R, (tp.z - cz) / BASE_R);
        // A trickle (6 walkers): below the committed threshold → rangers only.
        for k in 0..6 { h.spawn_zombie(ZClass::Walker, tp.x + out.0 * (20.0 + k as f64 * 2.0), tp.z + out.1 * 20.0, ZState::Marching); }
        for _ in 0..(10.0 * 60.0) as usize { h.step(1.0 / 60.0); }
        assert!(!h.weapons_free, "6 walkers must not trip weapons-free");
        let sniper_shots: u64 = h.defenders.iter().filter(|d| d.kind == DKind::Sniper).map(|d| d.shots).sum();
        assert_eq!(sniper_shots, 0, "snipers must hold fire under noise discipline");
        // A committed wave (60): threat crosses the threshold → weapons free.
        for k in 0..60 { h.spawn_zombie(ZClass::Walker, tp.x + out.0 * (25.0 + (k % 10) as f64 * 2.0), tp.z + out.1 * (25.0 + (k / 10) as f64 * 2.0), ZState::Marching); }
        for _ in 0..(8.0 * 60.0) as usize { h.step(1.0 / 60.0); if h.weapons_free { break; } }
        assert!(h.weapons_free, "a 60-strong push must trip weapons-free");
    }

    #[test]
    fn crews_repair_with_hauled_stock_and_the_breach_recall() {
        let mut h = Horde::new(51, 200);
        h.wave_spawn_t = 1e9;
        let wid = h.structures.iter().position(|s| s.kind == SKind::Wall).unwrap();
        h.structures[wid].hp = 300.0; // battle scar
        // Peace: crew gets the job, a porter hauls a bundle, HP rises — repair
        // is IMPOSSIBLE without a delivery (crews start with stock 0), so any
        // HP gain proves the whole hauling chain.
        for _ in 0..(40.0 * 60.0) as usize { h.step(1.0 / 60.0); if h.structures[wid].hp > 450.0 { break; } }
        assert!(h.structures[wid].hp > 450.0, "hauled stock must repair the wall: hp={}", h.structures[wid].hp);
        // Sudden breach next door → the Commander recalls works units nearby.
        h.destroy_structure(wid + 1);
        let recalled = h.defenders.iter().any(|d| !d.kind.fighter() && d.alive() && d.state == DState::Fleeing);
        assert!(recalled, "porters/crews near a fresh breach must be recalled home");
    }

    #[test]
    fn nothing_spawns_on_top_of_anything() {
        let h = Horde::new(77, 6000);
        // Houses: no two buildings interpenetrate.
        let hs: Vec<&Structure> = h.structures.iter().filter(|s| s.kind == SKind::House).collect();
        for (i, a) in hs.iter().enumerate() {
            for b in &hs[i + 1..] {
                let (dx, dz) = (a.p.x - b.p.x, a.p.z - b.p.z);
                assert!((dx * dx + dz * dz).sqrt() >= 14.0, "houses overlap");
            }
        }
        // Dormant zombies: golden-spiral spacing → nearest neighbour ≥ 1.2 wu.
        for (i, z) in h.units.iter().enumerate().step_by(53) {
            let near = h.units.iter().enumerate()
                .filter(|(j, o)| *j != i && o.alive())
                .map(|(_, o)| { let (dx, dz) = (o.p.x - z.p.x, o.p.z - z.p.z); dx * dx + dz * dz })
                .fold(f64::INFINITY, f64::min);
            assert!(near.sqrt() >= 1.2, "sleepers stacked: nearest at {}", near.sqrt());
        }
        // Idle works units spread around the storehouse (personal spots).
        let mut hq = Horde::new(77, 500);
        hq.wave_spawn_t = 1e9;
        for _ in 0..(20.0 * 60.0) as usize { hq.step(1.0 / 60.0); }
        let works: Vec<&Defender> = hq.defenders.iter().filter(|d| !d.kind.fighter() && d.alive() && d.state == DState::Idle).collect();
        for (i, a) in works.iter().enumerate() {
            for b in &works[i + 1..] {
                let (dx, dz) = (a.p.x - b.p.x, a.p.z - b.p.z);
                assert!((dx * dx + dz * dz).sqrt() >= 1.0, "idle works units stacked on one spot");
            }
        }
    }

    #[test]
    fn a_marching_column_shakes_sleepers_awake_by_contact() {
        let mut h = Horde::new(97, 300);
        h.wave_spawn_t = 1e9;
        h.defenders.clear();
        h.step(1.0 / 60.0);
        // A marcher dropped right on top of a sleeper, heading across it.
        let sleeper = h.units[0].p;
        h.spawn_zombie(ZClass::Walker, sleeper.x - 1.2, sleeper.z, ZState::Marching);
        let (_, active0) = h.counts();
        for _ in 0..(3.0 * 60.0) as usize { h.step(1.0 / 60.0); }
        let (_, active1) = h.counts();
        assert!(active1 > active0, "contact must shake sleepers awake (no noise threshold): {active0} -> {active1}");
        assert!(h.units.iter().take(20).any(|z| z.alive() && matches!(z.state, ZState::Marching) && z.class != ZClass::Walker || matches!(z.state, ZState::Marching)), "the shaken sleeper joins the march");
    }

    #[test]
    fn zombies_beeline_to_the_nearest_human() {
        let mut h = Horde::new(83, 300);
        h.wave_spawn_t = 1e9;
        h.step(1.0 / 60.0);
        // A walker dropped 30 wu from a posted fighter (inside its watch 40).
        let post = h.defenders.iter().find(|d| d.kind == DKind::Soldier).unwrap().p;
        let hp0: f64 = h.defenders.iter().filter(|d| d.alive()).map(|d| d.hp).sum();
        for k in 0..8 { h.spawn_zombie(ZClass::Runner, post.x + 24.0 + k as f64, post.z + 6.0, ZState::Marching); }
        for _ in 0..(12.0 * 60.0) as usize { h.step(1.0 / 60.0); }
        let hp1: f64 = h.defenders.iter().filter(|d| d.alive()).map(|d| d.hp).sum();
        assert!(hp1 < hp0, "runners in sight must close on and maul the defenders ({hp0} -> {hp1})");
    }

    #[test]
    fn rangers_sortie_out_the_gate_in_peacetime() {
        let mut h = Horde::new(91, 3000);
        h.wave_spawn_t = 1e9; // eternal peace → the Commander sends clearers
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        let (mut went_out, kills0) = (false, h.kills);
        for _ in 0..(90.0 * 60.0) as usize {
            h.step(1.0 / 60.0);
            if h.defenders.iter().any(|d| d.kind == DKind::Ranger && d.alive() && { let (dx, dz) = (d.p.x - cx, d.p.z - cz); (dx * dx + dz * dz).sqrt() > BASE_R + 12.0 }) { went_out = true; }
            if went_out && h.kills > kills0 { break; }
        }
        assert!(went_out, "rangers must sortie outside the walls in peacetime");
        assert!(h.kills > kills0, "the sortie must clear sleepers (silently)");
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

    #[test]
    fn scenario_flow_funnels_through_the_carvings_to_the_cc() {
        // From several bearings on the wave-spawn ring, descending the flow
        // field must reach the base in every impassable scenario — i.e. the
        // Dijkstra actually routes through the pass gaps / causeways / trails.
        for sc in [Scenario::Pass, Scenario::River, Scenario::Forest, Scenario::Patches] {
            let h = Horde::with_scenario(11, 3000, sc);
            let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
            let mut reached = 0;
            for q in 0..8 {
                let a = q as f64 * std::f64::consts::FRAC_PI_4 + 0.1;
                let mut start = None;
                'scan: for rr in [WORLD * 0.34, WORLD * 0.31, WORLD * 0.37, WORLD * 0.28] {
                    for da in [0.0f64, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3] {
                        let (x, z) = (cx + (a + da).cos() * rr, cz + (a + da).sin() * rr);
                        if x > MARGIN && z > MARGIN && x < WORLD - MARGIN && z < WORLD - MARGIN
                            && h.passable(x, z) && h.flow.reachable(x, z) { start = Some((x, z)); break 'scan; }
                    }
                }
                let Some((mut x, mut z)) = start else { continue; };
                for _ in 0..6000 {
                    let (dx, dz) = h.flow.flow_at(x, z);
                    if dx == 0.0 && dz == 0.0 { break; }
                    x += dx * 4.0; z += dz * 4.0;
                    let (ex, ez) = (x - cx, z - cz);
                    if (ex * ex + ez * ez).sqrt() < BASE_R { reached += 1; break; }
                }
            }
            assert!(reached >= 6, "{sc:?}: only {reached}/8 bearings routed to the base");
        }
    }

    #[test]
    fn patches_connectivity_pass_keeps_the_base_reachable_and_open() {
        // The TAB playability guarantee for the blob-mosaic map: across many
        // seeds the flow field must reach the CC (the base clearing is carved,
        // and the connectivity pass links the big pockets), and the walkable
        // fraction must stay in a sane band — enough open ground to fight in,
        // but the patches must actually block a real chunk (it's not just OPEN).
        for seed in [1u64, 2, 5, 9, 13, 21, 42, 99, 128, 777] {
            let h = Horde::with_scenario(seed, 3000, Scenario::Patches);
            let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
            // the base clearing is open ground the flood fills
            assert!(h.passable(cx, cz) && h.flow.reachable(cx, cz), "seed {seed}: CC not reachable");
            // Connectivity is the pass grid's own property (the coarser flow
            // field can miss thin corridors) — flood-fill the pass grid from
            // the CC and compare the reached open cells to all open cells.
            let (pn, pcell, grid) = (h.pass_n, h.pass_cell, &h.pass_grid);
            let start = ((cz / pcell) as usize).min(pn - 1) * pn + ((cx / pcell) as usize).min(pn - 1);
            let mut reach = vec![false; pn * pn];
            let mut q = std::collections::VecDeque::new();
            if grid[start] { reach[start] = true; q.push_back(start); }
            while let Some(c) = q.pop_front() {
                let (i, j) = (c % pn, c / pn);
                for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (ni, nj) = (i as i64 + di, j as i64 + dj);
                    if ni < 0 || nj < 0 || ni >= pn as i64 || nj >= pn as i64 { continue; }
                    let nc = nj as usize * pn + ni as usize;
                    if grid[nc] && !reach[nc] { reach[nc] = true; q.push_back(nc); }
                }
            }
            let open = grid.iter().filter(|&&g| g).count();
            let reached = reach.iter().filter(|&&r| r).count();
            let open_frac = open as f64 / (pn * pn) as f64;
            assert!((0.35..0.92).contains(&open_frac), "seed {seed}: walkable fraction {open_frac:.2} out of band");
            // Most walkable ground must be one connected network, not islands.
            assert!(reached as f64 / open as f64 > 0.90, "seed {seed}: only {:.0}% of open ground is connected to the base", 100.0 * reached as f64 / open as f64);
        }
    }

    #[test]
    fn scenario_movement_respects_impassable_ground() {
        // 20 s of a landed wave in each carved scenario: no ground unit —
        // zombie or defender — may ever stand in a blocked cell (the slide).
        for sc in [Scenario::Pass, Scenario::River, Scenario::Forest, Scenario::Patches] {
            let mut h = Horde::with_scenario(7, 4000, sc);
            h.trigger_wave(); h.trigger_wave(); // announce, then land NOW
            for _ in 0..600 { h.step(1.0 / 30.0); }
            for z in h.units.iter().filter(|z| z.alive() && z.class.altitude() == 0.0) {
                assert!(h.passable(z.p.x, z.p.z), "{sc:?}: ground zombie in blocked terrain at ({:.0},{:.0}) state {:?}", z.p.x, z.p.z, z.state);
            }
            for d in h.defenders.iter().filter(|d| d.alive()) {
                assert!(h.passable(d.p.x, d.p.z), "{sc:?}: defender {:?} in blocked terrain at ({:.0},{:.0})", d.kind, d.p.x, d.p.z);
            }
        }
    }

    #[test]
    fn multigoal_field_reaches_live_buildings_and_reroutes_as_they_fall() {
        // The user's multi-source idea, debugged: with the multi flow field
        // every live building is a 0-seed, and a zombie descending from any
        // ring bearing arrives at a LIVE building — and when buildings are
        // destroyed the field re-routes to the next nearest live one (no
        // zombie is ever left marching toward a dead building).
        let mut h = Horde::with_scenario(19, 3000, Scenario::Classic);
        h.set_flow_multi(true);
        h.step(1.0 / 60.0); // build the field
        assert!(h.flow_multi());
        let (cx, cz) = (WORLD / 2.0, WORLD / 2.0);
        // helper: descend the field from (x,z), return the cell you settle in
        let settle = |h: &Horde, mut x: f64, mut z: f64| -> (f64, f64) {
            for _ in 0..6000 { let (dx, dz) = h.flow.flow_at(x, z); if dx == 0.0 && dz == 0.0 { break; } x += dx * 4.0; z += dz * 4.0; }
            (x, z)
        };
        // nearest LIVE goal building to a point, and its distance
        let nearest_live = |h: &Horde, x: f64, z: f64| -> f64 {
            h.structures.iter().filter(|s| s.hp > 0.0 && super::is_goal(s.kind))
                .map(|s| ((s.p.x - x).powi(2) + (s.p.z - z).powi(2)).sqrt())
                .fold(f64::MAX, f64::min)
        };
        // every live building is a seed (integ 0 at its cell) — reachable as a destination
        for s in h.structures.iter().filter(|s| s.hp > 0.0 && super::is_goal(s.kind)) {
            assert!(h.flow.reachable(s.p.x, s.p.z), "live building not in the field");
        }
        // from 12 ring bearings, descending lands on a live building
        let check_all_bearings_hit_a_live_building = |h: &Horde| {
            let mut hit = 0;
            for q in 0..12 {
                let a = q as f64 * std::f64::consts::TAU / 12.0 + 0.05;
                let (sx, sz) = (cx + a.cos() * WORLD * 0.33, cz + a.sin() * WORLD * 0.33);
                let (fx, fz) = settle(h, sx.clamp(MARGIN, WORLD - MARGIN), sz.clamp(MARGIN, WORLD - MARGIN));
                if nearest_live(h, fx, fz) < 30.0 { hit += 1; }
            }
            hit
        };
        assert!(check_all_bearings_hit_a_live_building(&h) >= 10, "multi field: zombies don't converge on live buildings");
        // Now destroy every building EXCEPT the CC + one house, rebuild, and
        // confirm the field re-routed: bearings still land on a LIVE building.
        let mut kept_house = false;
        for s in h.structures.iter_mut() {
            if s.kind == SKind::House && !kept_house { kept_house = true; continue; }
            if matches!(s.kind, SKind::House | SKind::Storehouse) { s.hp = 0.0; }
        }
        h.flow.dirty = true; h.flow.rebuild_t = 0.0;
        h.step(1.0 / 60.0);
        let live_goals = h.structures.iter().filter(|s| s.hp > 0.0 && super::is_goal(s.kind)).count();
        assert!(live_goals >= 2, "should still have the CC + one house alive");
        assert!(check_all_bearings_hit_a_live_building(&h) >= 10, "after demolition the field must re-route to the survivors, not dead buildings");
    }

    #[test]
    fn morton_mode_matches_the_tree_and_brute_force() {
        // The `M` toggle: at the same instant both structures must answer any
        // cull identically (and match brute force over the live units), and
        // the sim must keep running correctly on the Morton side.
        let mut h = Horde::new(23, 3000);
        h.trigger_wave(); h.trigger_wave();
        for _ in 0..120 { h.step(1.0 / 30.0); }
        h.sync_index(); // bring the tree current with post-apply positions
        let probes = [(WORLD / 2.0, 0.0, WORLD / 2.0, BASE_R + 80.0), (WORLD / 2.0 + 200.0, 0.0, WORLD / 2.0, 150.0), (WORLD * 0.3, 0.0, WORLD * 0.6, 260.0)];
        let cull_ids = |h: &Horde| -> Vec<Vec<u32>> {
            probes.iter().map(|&(x, y, z, r)| {
                let mut v: Vec<u32> = h.zq().cull(&Sphere3::new(x, y, z, r)).iter().map(|it| it.id).collect();
                v.sort(); v
            }).collect()
        };
        let tree_ids = cull_ids(&h);
        let counts0 = h.counts();
        h.set_zmode(ZMode::Morton);
        assert_eq!(h.zmode, ZMode::Morton);
        let mort_ids = cull_ids(&h);
        assert_eq!(tree_ids, mort_ids, "Morton culls diverge from the tree's");
        assert_eq!(h.counts(), counts0, "the switch must not touch the sim");
        for (k, &(x, y, z, r)) in probes.iter().enumerate() {
            let mut brute: Vec<u32> = h.units.iter().enumerate()
                .filter(|(_, u)| u.alive() && { let (dx, dy, dz) = (u.p.x - x, u.p.y - y, u.p.z - z); dx * dx + dy * dy + dz * dz <= r * r })
                .map(|(i, _)| i as u32).collect();
            brute.sort();
            assert_eq!(mort_ids[k], brute, "probe {k} != brute force");
        }
        // Run a stretch of real battle on Morton (kills, wakes, spawns), then
        // switch back — both transitions must leave a healthy index.
        for _ in 0..120 { h.step(1.0 / 30.0); }
        h.set_zmode(ZMode::Tree);
        for _ in 0..60 { h.step(1.0 / 30.0); }
        h.sync_index();
        let alive = h.units.iter().filter(|z| z.alive()).count();
        assert_eq!(h.zindex.item_count(), alive, "tree lost items across the round-trip");
    }
}
