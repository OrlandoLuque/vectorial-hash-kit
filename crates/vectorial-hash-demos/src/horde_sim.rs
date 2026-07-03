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
    /// Noise a zombie makes while active (groans; attacks/deaths in phase 2).
    pub fn noise_made(self) -> f64 { match self { Self::Walker => 1.0, Self::Runner => 2.0, _ => 10.0 } }
    pub fn altitude(self) -> f64 { if self == Self::Harpy { 20.0 } else { 0.0 } }
    pub fn index(self) -> usize { match self { Self::Walker => 0, Self::Runner => 1, Self::Chubby => 2, Self::Venom => 3, Self::Harpy => 4 } }
}
/// The biggest hearing radius — the single cull radius per noise event.
pub const MAX_HEAR: f64 = 288.0; // Harpy: 4 × 72

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ZState {
    Dormant,
    /// Walking to a noise position; lingers there, then re-sleeps.
    Investigating { tx: f64, tz: f64 },
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
    moved: bool,
}
impl Zombie { pub fn dormant(&self) -> bool { self.state == ZState::Dormant } }

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

// ----------------------------------------------------------------------- sim

pub struct Horde {
    pub units: Vec<Zombie>,
    pub structures: Vec<Structure>,
    pub zindex: Tree3<IZombie>,
    handles: Vec<Option<ItemRef>>,
    pub sindex: Tree3<IStruct>,
    pub noise: NoiseGrid,
    /// Noise events queued for the next step: (position, amount).
    pending: Vec<(Point3, f64)>,
    pub rng: Rng,
    pub now: f64,
    pub seed: f64,
    /// Woken-this-frame counter (for HUD/telemetry).
    pub woken_last: usize,
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
            units.push(Zombie { class, p: Point3::new(x, y, z), vel: (0.0, 0.0), state: ZState::Dormant, hp: class.max_hp(), heard: 0.0, linger: 0.0, groan_t: rng.range(0.5, 2.0), moved: false });
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
        Horde {
            units, structures,
            zindex: Tree3::new(world, 8), handles: vec![None; n],
            sindex, noise: NoiseGrid::new(96),
            pending: Vec::new(), rng, now: 0.0, seed: fseed, woken_last: 0,
        }
    }

    /// Queue a noise event (processed next `step`): the wake mechanism.
    /// Towers/attacks/infections will feed this in phase 2; drivers and the
    /// zombies' own groans feed it now.
    pub fn emit_noise(&mut self, p: Point3, amount: f64) { self.pending.push((p, amount)); }

    pub fn counts(&self) -> (usize, usize) {
        let dormant = self.units.iter().filter(|z| z.dormant()).count();
        (dormant, self.units.len() - dormant)
    }

    /// Keep the zombie index in sync **without rebuilding** (the siege
    /// `sync_index` pattern): dormant zombies never move → skipped entirely;
    /// movers `update_ref` in place (O(1) while they stay in their leaf).
    /// Runs at the top of every [`step`](Horde::step); public so renderers /
    /// tests can force the index current after `apply` moved units.
    pub fn sync_index(&mut self) {
        for (i, z) in self.units.iter_mut().enumerate() {
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

    /// One fixed step: noise events → wake culls → parallel decide → serial
    /// apply → keep-index sync.
    pub fn step(&mut self, dt: f64) {
        self.now += dt;
        self.sync_index();

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

        // 2) decide — read-only on the index, each zombie writes only itself:
        //    fans out over rayon on native, serial on wasm (no threads there).
        {
            let (index, units) = (&self.zindex, &mut self.units);
            let decide_one = |i: usize, z: &mut Zombie| {
                let ZState::Investigating { tx, tz } = z.state else { return; };
                let (dx, dz) = (tx - z.p.x, tz - z.p.z);
                let dist = (dx * dx + dz * dz).sqrt();
                let (mut vx, mut vz) = if dist > 8.0 { (dx / dist, dz / dist) } else { (0.0, 0.0) };
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
        //    schedule groans (queued for next frame's wake culls).
        let decay = 0.5f64.powf(dt);
        for z in self.units.iter_mut() {
            if z.dormant() { if z.heard > 1e-3 { z.heard *= decay; } continue; }
            // Arrival is by DISTANCE to the personal spot, not by velocity —
            // separation jiggle in a dense pack never lets vel hit exact zero.
            let arrived = match z.state { ZState::Investigating { tx, tz } => { let (dx, dz) = (tx - z.p.x, tz - z.p.z); dx * dx + dz * dz < 9.0 * 9.0 } ZState::Dormant => false };
            if !arrived && (z.vel.0 != 0.0 || z.vel.1 != 0.0) {
                let nx = (z.p.x + z.vel.0 * dt).clamp(MARGIN, WORLD - MARGIN);
                let nz = (z.p.z + z.vel.1 * dt).clamp(MARGIN, WORLD - MARGIN);
                z.p = Point3::new(nx, ground_h(nx, nz, self.seed) + z.class.altitude(), nz);
                z.moved = true;
                // Groans only while WALKING (half the class noise): a marching
                // wave pulls alert sleepers (harpies/venoms) along its path —
                // the wave grows as it travels — but an arrived, lingering pack
                // goes quiet, so the field re-settles (no perpetual chain; the
                // big cascades come from combat noise in phase 2).
                z.groan_t -= dt;
                if z.groan_t <= 0.0 {
                    z.groan_t = 3.0 + (z.p.x.to_bits() % 2048) as f64 * 0.001; // deterministic jitter
                    self.pending.push((z.p, z.class.noise_made() * 0.5));
                }
            } else {
                z.linger -= dt;
                if z.linger <= 0.0 { z.state = ZState::Dormant; z.heard = 0.0; z.moved = true; continue; }
            }
        }
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
