//! Shared, graphics-free simulation core for the `formations` demo — a Total
//! War-style **automatic army battle** (`docs/FORMATIONS_DESIGN.md`).
//!
//! The showcase is **two-level spatial reasoning**: the regiment level (~60
//! centroids/battle) maneuvers with *honest brute force* — at that N the index
//! loses (`docs/CHOOSING.md`) — while the soldier level (thousands) is the
//! index workout. Every combat mechanic underneath is a library query:
//! contact pairing = **k-NN** per engaged soldier; the flank/rear bonus =
//! **sector classify** vs the victim's regiment facing; the charge corridor =
//! a thick **`raycast`** attacker→target; the general's aura = a **sphere
//! cull** (literally the aura case); arrow landings = **k-NN(1)** at the
//! impact point with **no friend/foe check** (that IS the friendly-fire
//! model, dev-confirmed for TW: Warhammer).
//!
//! Numbers come from the RTW/M2TW-era community-mined tables in the design
//! doc: close spacing 1.2 m (×2.5 wu/m → 3.0 wu), 8 ranks deep, P(kill) =
//! 1.9% × 1.2^cf with cf clamped ±20, flank +5 / rear +7, charge bonus
//! decaying over ~13 s, morale casualties thresholds 10/50/80% → −2/−8/−12,
//! routers get a speed boost and don't fight back (pursuers get free kills),
//! shattered = never returns.
//!
//! House pattern (siege/horde): decide→apply split — `decide` fans out over
//! rayon on native (serial on wasm), reads the shared index and writes only
//! the soldier's own fields, **no rng**; all randomness (kill rolls, volley
//! scatter, cooldown jitter) lives in the serial passes, so a seed fully
//! determines the battle. The soldier index is **kept, not rebuilt**
//! (`sync_index`, the repo's measured house rule).

use vectorial_hash::{Aabb, ItemRef, Point3, Positioned3, Sphere3, Tree3};

pub use crate::siege_sim::{Faction, Rng};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// ---------------------------------------------------------------- world config

pub const WORLD: f64 = 1400.0; // map side, world units — a battlefield, not a continent
pub const SKY: f64 = 24.0; // index height (gentle terrain + arrow apex margin)
const MARGIN: f64 = 2.0;

/// Gentle 2-octave value-noise heightfield (the horde pattern, own seed) —
/// battle lines want near-flat ground, so the amplitude stays small.
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
    vnoise(x * 0.006, z * 0.006, s) * 3.2 + vnoise(x * 0.025, z * 0.025, s ^ 5) * 1.1
}

/// Default soldiers **per side** — override with `$FORM_POP`.
pub fn default_pop() -> usize {
    std::env::var("FORM_POP").ok().and_then(|s| s.parse().ok()).unwrap_or(4000)
}

// ------------------------------------------------------------------ the roster

/// Regiment kinds. Stats are the design doc's RTW-flavoured table: the combat
/// factor is `attack + charge + flank − defense`, so a ±2 swing ≈ ×1.44 kill
/// rate — small numbers carry real weight.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RKind { Sword, Spear, Archer, Cavalry, General }

impl RKind {
    pub fn attack(self) -> f64 { match self { Self::Sword => 9.0, Self::Spear => 7.0, Self::Archer => 5.0, Self::Cavalry => 10.0, Self::General => 12.0 } }
    pub fn defense(self) -> f64 { match self { Self::Sword => 9.0, Self::Spear => 11.0, Self::Archer => 5.0, Self::Cavalry => 8.0, Self::General => 12.0 } }
    /// Charge bonus at full momentum — decays linearly over [`CHARGE_SECS`].
    pub fn charge(self) -> f64 { match self { Self::Cavalry | Self::General => 8.0, Self::Archer => 1.0, _ => 2.0 } }
    pub fn base_morale(self) -> f64 { match self { Self::Archer => 8.0, Self::Cavalry => 11.0, Self::General => 14.0, _ => 10.0 } }
    pub fn speed(self) -> f64 { match self { Self::Cavalry => 26.0, Self::General => 24.0, Self::Spear => 10.0, _ => 11.0 } }
    /// Melee reach — the spear's covers two ranks (the design's "first 2 ranks fight").
    pub fn reach(self) -> f64 { match self { Self::Spear => 4.2, Self::Cavalry | Self::General => 3.4, Self::Archer => 2.2, Self::Sword => 2.6 } }
    /// Formation slot spacing (close order 1.2 m ≈ 3.0 wu; horses need more).
    pub fn spacing(self) -> f64 { match self { Self::Cavalry | Self::General => 4.6, Self::Archer => 3.4, _ => 3.0 } }
    /// Rank depth (M2TW default 8 for foot; cavalry deploys wide and shallow).
    pub fn ranks(self) -> u32 { match self { Self::Cavalry => 3, Self::General => 2, _ => 8 } }
    /// Wheel rate, rad/s — the outer-file sweep falls out of rigid slots on a
    /// rotating frame, so this is the only turning knob a regiment has.
    pub fn wheel(self) -> f64 { match self { Self::Cavalry | Self::General => 1.4, _ => 0.7 } }
    pub fn melee_kind(self) -> bool { !matches!(self, Self::Archer) }
    pub fn index(self) -> usize { match self { Self::Sword => 0, Self::Spear => 1, Self::Archer => 2, Self::Cavalry => 3, Self::General => 4 } }
}

// combat
pub const CHARGE_SECS: f64 = 13.0; // charge bonus decays linearly over this
pub const CHARGE_NEAR: f64 = 30.0; // need a run-up: no point-blank "charges"
pub const CHARGE_FAR: f64 = 85.0; // ~34 m at 2.5 wu/m — the researched arm distance
pub const FLANK_BONUS: f64 = 5.0;
pub const REAR_BONUS: f64 = 7.0;
pub const ACQUIRE: f64 = 12.0; // an engaged soldier walks at enemies inside this
// missiles
pub const ARCHER_RANGE: f64 = 110.0;
pub const ARROW_SPEED: f64 = 40.0;
pub const ARROW_KILL_R: f64 = 3.0; // landing k-NN(1) must be inside this to hit
pub const MISSILE_ATK: f64 = 13.0; // arrow lethality vs the victim's defense
pub const AMMO: u32 = 24; // per soldier (design: ~20–30)
pub const VOLLEY_COOLDOWN: f64 = 6.0;
// morale
pub const GENERAL_AURA: f64 = 50.0; // the "+1/star within ~50 m" radius
pub const AURA_BONUS: f64 = 3.0; // our general is a flat 3-star
pub const ROUT_AT: f64 = 0.0; // morale at/below → rout
pub const RALLY_AT: f64 = 4.0; // routing + morale back above + no enemy near → rally
pub const SHATTER_FLOOR: f64 = -10.0; // routing this deep never returns
pub const MAX_ROUTS: u32 = 3; // third rout = shattered
pub const ROUTER_RADIUS: f64 = 80.0; // chain-rout / enemy-rout morale radius

/// `P(kill) = 1.9% × 1.2^cf`, cf clamped ±20 — the community-mined RTW/M2TW
/// melee roll, once per ~1 s swing.
pub fn kill_prob(cf: f64) -> f64 { (0.019 * 1.2f64.powf(cf.clamp(-20.0, 20.0))).min(1.0) }

/// Remaining charge bonus: full at impact, gone [`CHARGE_SECS`] later (hence
/// cavalry cycle-charging).
pub fn charge_bonus(kind: RKind, charge_t: f64, now: f64) -> f64 {
    kind.charge() * (1.0 - (now - charge_t) / CHARGE_SECS).clamp(0.0, 1.0)
}

// -------------------------------------------------------------- sector classify

/// Where an attacker stands relative to a victim's **regiment facing** —
/// front/left/right/rear quadrants (±45° each). Flank +5, rear +7; braced
/// spears only brace the front.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sector { Front, Left, Right, Rear }

/// `(dx, dz)` is the victim→attacker offset; `facing` the victim regiment's
/// heading (forward = `(cos, sin)` in XZ). Dot/cross beats atan2 here — the
/// brute-force test recomputes it with angles to cross-check the convention.
pub fn sector(facing: f64, dx: f64, dz: f64) -> Sector {
    let (fx, fz) = (facing.cos(), facing.sin());
    let (dot, cross) = (fx * dx + fz * dz, fx * dz - fz * dx);
    if dot.abs() >= cross.abs() { if dot >= 0.0 { Sector::Front } else { Sector::Rear } }
    else if cross > 0.0 { Sector::Left } else { Sector::Right }
}

// ----------------------------------------------------------- regiments/soldiers

/// Regiment state machine. `Rallying` is the reform pause between a recovered
/// rout and rejoining the line; `Shattered` regiments never come back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RState { Advance, Engage, Charging, Routing, Shattered, Rallying }

impl RState {
    /// Still a fighting formation (counts toward "army not yet broken").
    pub fn fighting(self) -> bool { matches!(self, RState::Advance | RState::Engage | RState::Charging | RState::Rallying) }
}

#[derive(Clone)]
pub struct Regiment {
    pub faction: Faction,
    pub kind: RKind,
    pub centre: Point3,
    pub facing: f64,
    pub state: RState,
    pub morale: f64,
    pub files: u32,
    pub ranks: u32,
    pub spacing: f64,
    /// Starting strength (casualty % base) and the live count.
    pub n0: u32,
    pub alive: u32,
    /// Soldier ids of this regiment (fixed at deploy; slots back-fill inside).
    pub members: Vec<u32>,
    /// Cached order from the army brain (XZ walk/charge target, ~1 Hz).
    pub order: (f64, f64),
    /// Enemy regiment within ~130 wu — soldiers of cold regiments re-decide at
    /// 15 Hz only (the horde decision-bucket lever).
    pub hot: bool,
    pub routs: u32,
    charge_t: f64,
    morale_t: f64,
    volley_t: f64,
    disengage_t: f64,
    /// Soldiers holding a melee target this frame (drives Engage transitions).
    engaged: u32,
    /// Flank/rear hits taken since the last morale tick (→ the flanked penalty).
    flank_hits: u32,
}

#[derive(Clone)]
pub struct Soldier {
    pub p: Point3,
    pub faction: Faction,
    pub regiment: u32,
    /// Formation grid index — file = slot % files, rank = slot / files.
    pub slot: u32,
    pub hp: f64,
    pub ammo: u32,
    pub cooldown: f64,
    vel: (f64, f64),
    /// Melee target decided this frame (index id) — resolved serially in apply.
    melee: Option<u32>,
    moved: bool,
}
impl Soldier { pub fn alive(&self) -> bool { self.hp > 0.0 } }

/// The lightweight item in the soldier index (the siege `IUnit` decoupling):
/// the parallel decide pass holds `&Tree3<ISoldier>` while mutating `Vec<Soldier>`.
#[derive(Clone, Copy)]
pub struct ISoldier { pub id: u32, pub faction: Faction, pub regiment: u32, pub p: Point3 }
impl Positioned3 for ISoldier { fn position(&self) -> Point3 { self.p } }
impl ISoldier {
    fn of(i: usize, s: &Soldier) -> ISoldier { ISoldier { id: i as u32, faction: s.faction, regiment: s.regiment, p: s.p } }
}

/// An in-flight arrow: physically simple ballistics — launch point, landing
/// point (scatter already applied), launch time, flight seconds. The renderer
/// draws [`Arrow::pos`]; landing resolves with **no friend/foe check**.
#[derive(Clone, Copy)]
pub struct Arrow { pub from: Point3, pub to: Point3, pub t0: f64, pub flight: f64, pub faction: Faction }
impl Arrow {
    /// Position at time `now`: straight lerp + a parabolic arc over friendlies.
    pub fn pos(&self, now: f64) -> Point3 {
        let u = ((now - self.t0) / self.flight).clamp(0.0, 1.0);
        let apex = 3.0 + self.flight * ARROW_SPEED * 0.12;
        Point3::new(self.from.x + (self.to.x - self.from.x) * u, self.from.y + (self.to.y - self.from.y) * u + apex * 4.0 * u * (1.0 - u), self.from.z + (self.to.z - self.from.z) * u)
    }
}

/// A soldier's world-space formation slot — recomputed from (centre, facing,
/// grid index): rigid slots on a rotating frame give wheeling's outer-file
/// sweep for free. **No index query** (the cheap contrast to boids).
pub fn slot_pos(r: &Regiment, slot: u32) -> (f64, f64) {
    let files = r.files.max(1);
    let (fi, ra) = ((slot % files) as f64, (slot / files) as f64);
    let lx = (fi - (files as f64 - 1.0) / 2.0) * r.spacing;
    let lz = ((r.ranks as f64 - 1.0) / 2.0 - ra) * r.spacing; // rank 0 = front
    let (fx, fz) = (r.facing.cos(), r.facing.sin());
    (r.centre.x + fz * lx + fx * lz, r.centre.z - fx * lx + fz * lz)
}

// ------------------------------------------------------------------ deployment

const GENERAL_N: usize = 16; // the bodyguard squadron

/// One regiment + its soldiers, spawned standing on their slots.
#[allow(clippy::too_many_arguments)] // a deploy-time plumbing fn; a struct would be noise
fn add_regiment(regs: &mut Vec<Regiment>, soldiers: &mut Vec<Soldier>, rng: &mut Rng, fac: Faction, kind: RKind, n: usize, x: f64, z: f64, facing: f64, seed: f64) {
    let ranks = kind.ranks();
    let files = (n as u32).div_ceil(ranks).max(1);
    let mut r = Regiment {
        faction: fac, kind, centre: Point3::new(x, ground_h(x, z, seed), z), facing,
        state: RState::Advance, morale: kind.base_morale(), files, ranks, spacing: kind.spacing(),
        n0: n as u32, alive: n as u32, members: Vec::with_capacity(n),
        order: (x, z), hot: false, routs: 0, charge_t: -1e9,
        morale_t: rng.range(0.2, 1.2), volley_t: rng.range(0.0, VOLLEY_COOLDOWN), // staggered ticks
        disengage_t: 0.0, engaged: 0, flank_hits: 0,
    };
    let rid = regs.len() as u32;
    for k in 0..n {
        let (sx, sz) = slot_pos(&r, k as u32);
        let (sx, sz) = (sx.clamp(MARGIN, WORLD - MARGIN), sz.clamp(MARGIN, WORLD - MARGIN));
        r.members.push(soldiers.len() as u32);
        soldiers.push(Soldier {
            p: Point3::new(sx, ground_h(sx, sz, seed), sz), faction: fac, regiment: rid, slot: k as u32,
            hp: 1.0, ammo: if kind == RKind::Archer { AMMO } else { 0 }, cooldown: rng.range(0.0, 1.0),
            vel: (0.0, 0.0), melee: None, moved: false,
        });
    }
    regs.push(r);
}

/// The historical line, per side: sword line front, spears second, archers
/// behind, cavalry on the wings, the general's squadron at the back-centre —
/// the CA "Group Formations" block-tree, hand-rolled.
fn deploy(rng: &mut Rng, per_side: usize, seed: f64) -> (Vec<Regiment>, Vec<Soldier>) {
    let (mut regs, mut soldiers) = (Vec::new(), Vec::new());
    let cx = WORLD / 2.0;
    let reg_size = (per_side / 26).clamp(40, 160); // ~20–40 regiments at the default pop
    let cav_size = (reg_size / 2).clamp(24, 100); // TW cavalry runs half infantry strength
    let n_line = ((per_side.saturating_sub(GENERAL_N)) / reg_size).max(4);
    let n_sword = (n_line * 35 / 100).max(1);
    let n_spear = (n_line * 25 / 100).max(1);
    let n_arch = (n_line * 20 / 100).max(1);
    let n_cav = (n_line.saturating_sub(n_sword + n_spear + n_arch)).max(2); // ≥1 per wing
    let row_pitch = |kind: RKind, n: usize| (n as u32).div_ceil(kind.ranks()).max(1) as f64 * kind.spacing() + 16.0;
    for fac in Faction::ALL {
        let red = fac == Faction::Red;
        let (z0, fwd, facing) = if red { (WORLD * 0.34, 1.0, std::f64::consts::FRAC_PI_2) } else { (WORLD * 0.66, -1.0, -std::f64::consts::FRAC_PI_2) };
        let row = |regs: &mut Vec<Regiment>, soldiers: &mut Vec<Soldier>, rng: &mut Rng, kind: RKind, count: usize, size: usize, back: f64| {
            let pitch = row_pitch(kind, size);
            for k in 0..count {
                let x = cx + (k as f64 - (count as f64 - 1.0) / 2.0) * pitch;
                add_regiment(regs, soldiers, rng, fac, kind, size, x, z0 - fwd * back, facing, seed);
            }
        };
        row(&mut regs, &mut soldiers, rng, RKind::Sword, n_sword, reg_size, 0.0);
        row(&mut regs, &mut soldiers, rng, RKind::Spear, n_spear, reg_size, 34.0);
        row(&mut regs, &mut soldiers, rng, RKind::Archer, n_arch, reg_size, 64.0);
        // Cavalry wings: split left/right beyond the sword line's edge.
        let half_line = n_sword as f64 * row_pitch(RKind::Sword, reg_size) / 2.0;
        let cav_pitch = row_pitch(RKind::Cavalry, cav_size);
        for k in 0..n_cav {
            let side = if k % 2 == 0 { 1.0 } else { -1.0 };
            let x = cx + side * (half_line + 40.0 + (k / 2) as f64 * cav_pitch);
            add_regiment(&mut regs, &mut soldiers, rng, fac, RKind::Cavalry, cav_size, x, z0 - fwd * 10.0, facing, seed);
        }
        add_regiment(&mut regs, &mut soldiers, rng, fac, RKind::General, GENERAL_N, cx, z0 - fwd * 92.0, facing, seed);
    }
    (regs, soldiers)
}

// ----------------------------------------------------------------------- sim

/// A brain-tick snapshot of one regiment — the ~60-entry table every brute
/// scan below walks (honest brute force: at this N the index loses).
struct RSnap { fac: Faction, state: RState, x: f64, z: f64, alive: u32, facing: f64 }

pub struct Formations {
    pub soldiers: Vec<Soldier>,
    pub regiments: Vec<Regiment>,
    /// ONE shared index over ALL soldiers, both factions — queries filter by
    /// the item's `faction`/`regiment` (the siege pattern).
    pub index: Tree3<ISoldier>,
    handles: Vec<Option<ItemRef>>,
    pub arrows: Vec<Arrow>,
    /// Fallen soldiers for the renderer: (position, kind, faction, time of death).
    pub corpses: Vec<(Point3, RKind, Faction, f64)>,
    /// Kill credit per faction index (lifetime across runs, like horde's).
    pub kills: [u64; 2],
    /// Routers that escaped off their own map edge.
    pub fled: u64,
    /// Set when an army breaks: (time it happened, winner). Resets ~12 s later.
    pub game_over: Option<(f64, Faction)>,
    pub run: u32,
    pub rng: Rng,
    pub now: f64,
    frame: u64,
    pub seed: f64,
    base_seed: u64,
    per_side: usize,
    /// Regiments inside their own general's aura (refreshed each slow tick).
    aura: Vec<bool>,
    general_dead: [bool; 2],
    slow_t: f64,
}

impl Formations {
    pub fn new(seed: u64, per_side: usize) -> Formations {
        let fseed = (seed % 100_000) as f64 * 0.017 + 0.31;
        let mut rng = Rng::new(seed | 1);
        let (regiments, soldiers) = deploy(&mut rng, per_side, fseed);
        let world = Aabb::new(0.0, -8.0, 0.0, WORLD, SKY, WORLD);
        let (n, nr) = (soldiers.len(), regiments.len());
        let mut f = Formations {
            soldiers, regiments,
            index: Tree3::new(world, 8), handles: vec![None; n],
            arrows: Vec::new(), corpses: Vec::new(),
            kills: [0, 0], fled: 0, game_over: None, run: 1,
            rng, now: 0.0, frame: 0, seed: fseed, base_seed: seed, per_side,
            aura: vec![false; nr], general_dead: [false, false], slow_t: 0.0,
        };
        f.sync_index(); // index ready before the first step (brains/tests query it)
        f
    }

    /// Full run reset (an army broke, ~12 s of aftermath elapsed): fresh
    /// armies on a new seed, next run number; kill counters persist.
    fn reset(&mut self) {
        let next = Formations::new(self.base_seed.wrapping_add(self.run as u64 * 7919), self.per_side);
        let (run, kills, fled) = (self.run + 1, self.kills, self.fled);
        *self = next;
        self.run = run;
        self.kills = kills;
        self.fled = fled;
    }

    /// (red, blue) soldiers alive.
    pub fn counts(&self) -> (usize, usize) {
        let (mut r, mut b) = (0, 0);
        for s in &self.soldiers { if s.alive() { if s.faction == Faction::Red { r += 1; } else { b += 1; } } }
        (r, b)
    }

    /// (red, blue) regiments still in a fighting state.
    pub fn standing(&self) -> (usize, usize) {
        let f = |fac: Faction| self.regiments.iter().filter(|r| r.faction == fac && r.alive > 0 && r.state.fighting()).count();
        (f(Faction::Red), f(Faction::Blue))
    }

    /// The winner, once an army is broken (all regiments routed/shattered/dead).
    pub fn outcome(&self) -> Option<Faction> { self.game_over.map(|(_, w)| w) }

    /// Per-regiment banner line for the renderer:
    /// (centre, facing, faction, kind, state, strength fraction).
    pub fn banners(&self) -> Vec<(Point3, f64, Faction, RKind, RState, f32)> {
        self.regiments.iter().filter(|r| r.alive > 0)
            .map(|r| (r.centre, r.facing, r.faction, r.kind, r.state, r.alive as f32 / r.n0.max(1) as f32))
            .collect()
    }

    /// Keep the soldier index in sync **without rebuilding** (the measured
    /// house rule): movers `update_ref` in place (O(1) while they stay in
    /// their leaf), the unmoved cost nothing, deaths already `remove_ref`d.
    pub fn sync_index(&mut self) {
        for (i, s) in self.soldiers.iter_mut().enumerate() {
            if !s.alive() { s.moved = false; continue; }
            match (self.handles[i], s.moved) {
                (None, _) => { self.handles[i] = self.index.insert_ref(ISoldier::of(i, s)); s.moved = false; }
                (Some(_), false) => {}
                (Some(h), true) => {
                    let it = ISoldier::of(i, s);
                    if !self.index.update_ref(h, |o| *o = it) { self.handles[i] = self.index.insert_ref(it); }
                    s.moved = false;
                }
            }
        }
    }

    /// A soldier leaves the field: drop him from the index by handle (O(1)),
    /// decrement his regiment. `corpse` = false for routers escaping the edge.
    fn fell(&mut self, id: usize, corpse: bool) {
        if !self.soldiers[id].alive() { return; }
        let (p, fac, rid) = (self.soldiers[id].p, self.soldiers[id].faction, self.soldiers[id].regiment as usize);
        if corpse {
            self.corpses.push((p, self.regiments[rid].kind, fac, self.now));
            if self.corpses.len() > 30_000 { self.corpses.drain(0..5_000); }
        }
        if let Some(h) = self.handles[id].take() { self.index.remove_ref(h); }
        self.soldiers[id].hp = 0.0;
        self.soldiers[id].moved = false;
        self.regiments[rid].alive = self.regiments[rid].alive.saturating_sub(1);
    }

    /// Is the charge corridor from regiment `rid`'s centre to `(tx, tz)` free
    /// of interposed *friendlies*? A thick `Tree3::raycast` attacker→target:
    /// the first body on the lane that isn't ours decides — an enemy means the
    /// lane ends at the target, a friend means we'd trample our own line.
    pub fn lane_clear(&self, rid: usize, tx: f64, tz: f64) -> bool {
        let r = &self.regiments[rid];
        let from = Point3::new(r.centre.x, r.centre.y + 1.5, r.centre.z);
        let (dx, dz) = (tx - from.x, tz - from.z);
        let dy = ground_h(tx, tz, self.seed) + 1.5 - from.y;
        let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
        let dir = Point3::new(dx / len, dy / len, dz / len);
        for (_, it) in self.index.raycast(from, dir, len - 6.0, 3.5) {
            if it.regiment == rid as u32 { continue; } // our own front ranks sit on the ray
            return it.faction != r.faction;
        }
        true
    }

    /// The ~1 Hz layer: army brains (orders, charges, pursuit), general auras,
    /// hot flags, victory. All serial; all regiment-level scans are brute
    /// force over the ~60 centres (see [`RSnap`]).
    fn slow_tick(&mut self) {
        let now = self.now;
        let snap: Vec<RSnap> = self.regiments.iter().map(|r| RSnap { fac: r.faction, state: r.state, x: r.centre.x, z: r.centre.z, alive: r.alive, facing: r.facing }).collect();
        // General auras: ONE sphere cull per side — literally the aura case.
        self.aura.iter_mut().for_each(|a| *a = false);
        for fac in Faction::ALL {
            let Some(g) = self.regiments.iter().position(|r| r.faction == fac && r.kind == RKind::General) else { continue };
            if self.regiments[g].alive == 0 { self.general_dead[fac.index()] = true; continue; }
            let c = self.regiments[g].centre;
            for it in self.index.cull(&Sphere3::new(c.x, c.y, c.z, GENERAL_AURA)) {
                if it.faction == fac { self.aura[it.regiment as usize] = true; }
            }
        }
        // Army brains: one pass computes each regiment's order; a second pass
        // applies (the charge check needs `&self` for the corridor raycast).
        let mut orders: Vec<(usize, (f64, f64), bool, bool)> = Vec::new(); // (rid, target, charge, hot)
        for (rid, r) in self.regiments.iter().enumerate() {
            if r.alive == 0 { continue; }
            let (mut near_st, mut near_rt): (Option<(f64, usize)>, Option<(f64, usize)>) = (None, None);
            for (j, o) in snap.iter().enumerate() {
                if o.fac == r.faction || o.alive == 0 { continue; }
                let d = ((o.x - r.centre.x).powi(2) + (o.z - r.centre.z).powi(2)).sqrt();
                let slot = if matches!(o.state, RState::Routing | RState::Shattered) { &mut near_rt } else { &mut near_st };
                if slot.map(|(b, _)| d < b).unwrap_or(true) { *slot = Some((d, j)); }
            }
            let hot = near_st.map(|(d, _)| d < 130.0).unwrap_or(false) || near_rt.map(|(d, _)| d < 130.0).unwrap_or(false);
            let edge = if r.faction == Faction::Red { 8.0 } else { WORLD - 8.0 };
            let (target, charge) = match r.state {
                RState::Routing | RState::Shattered => ((r.centre.x, edge), false), // flee your own edge
                RState::Rallying => ((r.centre.x, r.centre.z), false), // stand and reform
                _ => match (r.kind, near_st, near_rt) {
                    // Cavalry pursuit: run the nearest routers down (free kills).
                    (RKind::Cavalry, _, Some((d, j))) if d < 260.0 => ((snap[j].x, snap[j].z), false),
                    // Cavalry seeks the flank: aim BEHIND the enemy block; the
                    // charge order (below) only arms when the lane is clear.
                    (RKind::Cavalry, Some((d, j)), _) => {
                        let arm = r.state == RState::Advance && d > CHARGE_NEAR && d < CHARGE_FAR && self.lane_clear(rid, snap[j].x, snap[j].z);
                        if arm { ((snap[j].x, snap[j].z), true) }
                        else { ((snap[j].x - snap[j].facing.cos() * 30.0, snap[j].z - snap[j].facing.sin() * 30.0), false) }
                    }
                    // The general keeps station with his army (the aura is the job).
                    (RKind::General, Some(_), _) => {
                        let own: Vec<&RSnap> = snap.iter().filter(|o| o.fac == r.faction && o.alive > 0 && o.state.fighting()).collect();
                        let n = own.len().max(1) as f64;
                        ((own.iter().map(|o| o.x).sum::<f64>() / n, own.iter().map(|o| o.z).sum::<f64>() / n), false)
                    }
                    (_, Some((d, j)), _) => {
                        let arm = r.kind.melee_kind() && r.state == RState::Advance && d > CHARGE_NEAR && d < CHARGE_FAR && self.lane_clear(rid, snap[j].x, snap[j].z);
                        ((snap[j].x, snap[j].z), arm)
                    }
                    (_, None, Some((_, j))) => ((snap[j].x, snap[j].z), false), // mop up
                    (_, None, None) => ((r.centre.x, r.centre.z), false),
                },
            };
            orders.push((rid, target, charge, hot));
        }
        for (rid, target, charge, hot) in orders {
            let r = &mut self.regiments[rid];
            r.order = target;
            r.hot = hot;
            if charge && r.state == RState::Advance { r.state = RState::Charging; r.charge_t = now; }
        }
        // Victory: an army with no fighting regiment left is broken.
        if self.game_over.is_none() {
            let up = |fac: Faction| self.regiments.iter().any(|r| r.faction == fac && r.alive > 0 && r.state.fighting());
            match (up(Faction::Red), up(Faction::Blue)) {
                (true, true) => {}
                (true, false) => self.game_over = Some((now, Faction::Red)),
                (false, true) => self.game_over = Some((now, Faction::Blue)),
                (false, false) => { let (r, b) = self.counts(); self.game_over = Some((now, if b > r { Faction::Blue } else { Faction::Red })); }
            }
        }
    }

    /// One regiment's ~1 Hz morale evaluation — the design table: base +
    /// general aura − casualties (10/50/80% → −2/−8/−12) − friendly routers
    /// nearby (chain routs) + enemy routers nearby − flanked, then the state
    /// transitions (rout / rally / shatter). Also back-fills formation slots
    /// so the grid stays packed from the front rank.
    fn morale_tick(&mut self, rid: usize) {
        // Slot back-fill: alive members, in slot order, take slots 0..alive.
        let mut alive_slots: Vec<(u32, u32)> = self.regiments[rid].members.iter()
            .filter(|&&id| self.soldiers[id as usize].alive())
            .map(|&id| (self.soldiers[id as usize].slot, id)).collect();
        alive_slots.sort_unstable();
        for (k, &(_, id)) in alive_slots.iter().enumerate() { self.soldiers[id as usize].slot = k as u32; }
        let r = &self.regiments[rid];
        if r.alive == 0 || r.state == RState::Shattered { return; }
        let cas = 1.0 - r.alive as f64 / r.n0.max(1) as f64;
        let mut m = r.kind.base_morale();
        if self.aura[rid] { m += AURA_BONUS; }
        if self.general_dead[r.faction.index()] { m -= 4.0; }
        m -= if cas >= 0.8 { 12.0 } else if cas >= 0.5 { 8.0 } else if cas >= 0.1 { 2.0 } else { 0.0 };
        // Routers nearby, both signs — BRUTE FORCE over the ~60 regiment
        // centres: at that N the index loses (docs/CHOOSING.md), so we say so.
        let (mut fr, mut er, mut enemy_d) = (0.0f64, 0.0f64, f64::INFINITY);
        for (j, o) in self.regiments.iter().enumerate() {
            if j == rid || o.alive == 0 { continue; }
            let d = ((o.centre.x - r.centre.x).powi(2) + (o.centre.z - r.centre.z).powi(2)).sqrt();
            if o.faction != r.faction && o.state.fighting() { enemy_d = enemy_d.min(d); }
            if !matches!(o.state, RState::Routing | RState::Shattered) || d > ROUTER_RADIUS { continue; }
            if o.faction == r.faction { fr += 4.0; } else { er += 4.0; }
        }
        m += er.min(8.0) - fr.min(12.0);
        if r.flank_hits >= 3 { m -= 5.0; }
        let r = &mut self.regiments[rid];
        r.morale = m;
        r.flank_hits = 0;
        match r.state {
            RState::Routing if m >= RALLY_AT && enemy_d > 60.0 => r.state = RState::Rallying,
            RState::Rallying => r.state = if m >= RALLY_AT { RState::Advance } else { RState::Routing },
            RState::Advance | RState::Engage | RState::Charging if m <= ROUT_AT => {
                r.routs += 1;
                r.state = if r.routs >= MAX_ROUTS || m <= SHATTER_FLOOR { RState::Shattered } else { RState::Routing };
            }
            _ => {}
        }
    }

    /// One fixed step: keep-index sync → slow tick (~1 Hz) → morale ticks →
    /// **parallel decide** (slot steering + k-NN contact pairing; no rng) →
    /// serial apply (movement, kill rolls, regiment maneuver, volleys, arrow
    /// landings).
    pub fn step(&mut self, dt: f64) {
        self.now += dt;
        self.frame += 1;
        self.sync_index();
        if let Some((t0, _)) = self.game_over { if self.now - t0 > 12.0 { self.reset(); return; } }
        self.slow_t -= dt;
        if self.slow_t <= 0.0 { self.slow_t = 1.0; self.slow_tick(); }
        // Morale ticks (staggered ~1 Hz per regiment).
        let due: Vec<usize> = self.regiments.iter_mut().enumerate()
            .filter_map(|(rid, r)| { r.morale_t -= dt; if r.morale_t <= 0.0 && r.alive > 0 { r.morale_t = 1.0; Some(rid) } else { None } }).collect();
        for rid in due { self.morale_tick(rid); }

        // 1) decide — read-only on the shared index + regiment table; each
        //    soldier writes only itself. Rayon fan-out on native, serial wasm.
        {
            let (index, regiments, frame) = (&self.index, &self.regiments, self.frame);
            let decide_one = |i: usize, s: &mut Soldier| {
                if !s.alive() { return; }
                let r = &regiments[s.regiment as usize];
                // DECISION BUCKETS (the horde lever): far from any enemy a
                // soldier re-decides at 15 Hz staggered by id — his cached vel
                // keeps him marching to a slowly-moving slot in between.
                if !r.hot && (frame + i as u64) % 4 != 0 { return; }
                s.melee = None;
                let routed = matches!(r.state, RState::Routing | RState::Shattered);
                // Formation slot: kinematic, recomputed from (centre, facing,
                // grid index) — no query needed (the cheap contrast to boids).
                let (mut tx, mut tz) = slot_pos(r, s.slot);
                if routed { // rout scatter: loose chaos vs tight blocks reads at a glance
                    let a = i as f64 * 2.399963;
                    tx += a.cos() * (2.0 + (i % 7) as f64); tz += a.sin() * (2.0 + (i % 7) as f64);
                }
                let mut arrive = 0.5;
                let (mut sx, mut sz) = (0.0, 0.0);
                if r.hot && !routed && r.state != RState::Rallying {
                    // Contact pairing: ONE k-NN serves targeting (nearest enemy
                    // in reach → melee pair) and local separation (any body too
                    // close pushes) — routed soldiers skip this: they don't
                    // fight back, which is why pursuers get free kills.
                    let mut foe = false;
                    for (d, it) in index.knn(s.p, 6) {
                        if it.id as usize == i { continue; }
                        if d < 2.0 { let l = d.max(0.2); sx += (s.p.x - it.p.x) / l * (2.0 - d); sz += (s.p.z - it.p.z) / l * (2.0 - d); }
                        if foe || it.faction == s.faction { continue; }
                        foe = true; // knn is sorted: first enemy seen is the nearest
                        if d <= r.kind.reach() { s.melee = Some(it.id); tx = s.p.x; tz = s.p.z; }
                        else if d <= ACQUIRE { tx = it.p.x; tz = it.p.z; arrive = r.kind.reach() * 0.7; }
                    }
                }
                let (dx, dz) = (tx - s.p.x, tz - s.p.z);
                let d = (dx * dx + dz * dz).sqrt();
                let run = if routed { 1.3 } else if d > 8.0 { 1.35 } else { 1.0 }; // routers sprint; laggards trot
                let sp = r.kind.speed() * run;
                let (mut vx, mut vz) = if d > arrive { (dx / d * sp, dz / d * sp) } else { (0.0, 0.0) };
                vx += sx * 3.0; vz += sz * 3.0;
                let l = (vx * vx + vz * vz).sqrt();
                if l > sp { vx = vx / l * sp; vz = vz / l * sp; }
                s.vel = (vx, vz);
            };
            #[cfg(not(target_arch = "wasm32"))]
            self.soldiers.par_iter_mut().enumerate().for_each(|(i, s)| decide_one(i, s));
            #[cfg(target_arch = "wasm32")]
            self.soldiers.iter_mut().enumerate().for_each(|(i, s)| decide_one(i, s));
        }

        // 2) apply, serial. Movement + cooldowns + router escapes first.
        let seed = self.seed;
        let mut escapes: Vec<usize> = Vec::new();
        {
            let regiments = &self.regiments;
            for (i, s) in self.soldiers.iter_mut().enumerate() {
                if !s.alive() { continue; }
                s.cooldown -= dt;
                if s.vel.0 != 0.0 || s.vel.1 != 0.0 {
                    let nx = (s.p.x + s.vel.0 * dt).clamp(MARGIN, WORLD - MARGIN);
                    let nz = (s.p.z + s.vel.1 * dt).clamp(MARGIN, WORLD - MARGIN);
                    s.p = Point3::new(nx, ground_h(nx, nz, seed), nz);
                    s.moved = true;
                }
                // Routers leaving their own map edge are gone for good (TW rule).
                if matches!(regiments[s.regiment as usize].state, RState::Routing | RState::Shattered)
                    && (if s.faction == Faction::Red { s.p.z < 14.0 } else { s.p.z > WORLD - 14.0 }) { escapes.push(i); }
            }
        }
        for i in escapes { self.fled += 1; self.fell(i, false); }

        // 3) melee resolution — the only rng consumer in the combat path
        //    (kill rolls are serial by design: determinism).
        let mut engaged = vec![0u32; self.regiments.len()];
        let mut flanked: Vec<u32> = Vec::new();
        for i in 0..self.soldiers.len() {
            if !self.soldiers[i].alive() { continue; }
            let Some(t) = self.soldiers[i].melee else { continue };
            let rid = self.soldiers[i].regiment as usize;
            engaged[rid] += 1;
            if self.soldiers[i].cooldown > 0.0 { continue; }
            let t = t as usize;
            if !self.soldiers[t].alive() { continue; } // fell to an earlier swing this frame
            let (ap, vp) = (self.soldiers[i].p, self.soldiers[t].p);
            let akind = self.regiments[rid].kind;
            let (dx, dz) = (vp.x - ap.x, vp.z - ap.z);
            if (dx * dx + dz * dz).sqrt() > akind.reach() + 1.0 { continue; } // drifted since decide
            self.soldiers[i].cooldown = self.rng.range(0.9, 1.15); // ~1 swing/s, desynced
            let vrid = self.soldiers[t].regiment as usize;
            let (vkind, vfacing, vstate) = { let v = &self.regiments[vrid]; (v.kind, v.facing, v.state) };
            let vrouted = matches!(vstate, RState::Routing | RState::Shattered);
            // SECTOR CLASSIFY: attacker vs the victim's REGIMENT facing.
            let sec = sector(vfacing, ap.x - vp.x, ap.z - vp.z);
            let flank = match sec { Sector::Front => 0.0, Sector::Rear => REAR_BONUS, _ => FLANK_BONUS };
            if sec != Sector::Front && !vrouted { flanked.push(vrid as u32); }
            let mut cb = charge_bonus(akind, self.regiments[rid].charge_t, self.now);
            // Braced spears nullify a FRONTAL cavalry charge; side/rear charges
            // bypass bracing — flanking beats spear walls by geometry, not stats.
            if vkind == RKind::Spear && sec == Sector::Front && matches!(akind, RKind::Cavalry | RKind::General) && matches!(vstate, RState::Advance | RState::Engage) { cb = 0.0; }
            // Routers don't fight back — pursuers within reach take free kills.
            let p = if vrouted { 1.0 } else { kill_prob(akind.attack() + cb + flank - vkind.defense()) };
            if self.rng.unit() < p { self.kills[self.soldiers[i].faction.index()] += 1; self.fell(t, true); }
        }
        for rid in flanked { self.regiments[rid as usize].flank_hits += 1; }
        for (rid, e) in engaged.into_iter().enumerate() { self.regiments[rid].engaged = e; }

        // 4) regiment maneuver: state upkeep + wheel-and-advance along facing.
        let now = self.now;
        for r in self.regiments.iter_mut() {
            if r.alive == 0 { continue; }
            r.volley_t -= dt;
            let quorum = (r.alive / 16).max(1);
            match r.state {
                RState::Advance | RState::Charging if r.engaged >= quorum => { r.state = RState::Engage; r.disengage_t = 0.0; }
                RState::Charging if now - r.charge_t > CHARGE_SECS => r.state = RState::Advance, // spent, un-caught
                RState::Engage => {
                    if r.engaged == 0 { r.disengage_t += dt; if r.disengage_t > 3.0 { r.state = RState::Advance; } } else { r.disengage_t = 0.0; }
                }
                _ => {}
            }
            let (dx, dz) = (r.order.0 - r.centre.x, r.order.1 - r.centre.z);
            let d = (dx * dx + dz * dz).sqrt();
            if d < 3.0 { continue; }
            let routed = matches!(r.state, RState::Routing | RState::Shattered);
            let want = dz.atan2(dx);
            let mut da = want - r.facing;
            while da > std::f64::consts::PI { da -= std::f64::consts::TAU; }
            while da < -std::f64::consts::PI { da += std::f64::consts::TAU; }
            let w = r.kind.wheel() * if routed { 3.0 } else { 1.0 } * dt; // routers about-face in a panic
            r.facing += da.clamp(-w, w);
            if matches!(r.state, RState::Engage | RState::Rallying) { continue; } // hold the contact line / reform
            let mult = match r.state { RState::Charging => 1.5, RState::Routing | RState::Shattered => 1.3, _ => 1.0 };
            let sp = (r.kind.speed() * mult).min(d / dt);
            let (nx, nz) = ((r.centre.x + r.facing.cos() * sp * dt).clamp(MARGIN, WORLD - MARGIN), (r.centre.z + r.facing.sin() * sp * dt).clamp(MARGIN, WORLD - MARGIN));
            r.centre = Point3::new(nx, ground_h(nx, nz, seed), nz);
        }

        // 5) archer volleys: nearest standing enemy regiment in range (brute
        //    over the centres), one arrow per soldier with ammo, rng scatter
        //    (serial). Arc-over-friendlies is the ballistic model itself.
        let mut volleys: Vec<(usize, f64, f64)> = Vec::new();
        for (rid, r) in self.regiments.iter().enumerate() {
            if r.alive == 0 || r.kind != RKind::Archer || r.volley_t > 0.0 || r.state != RState::Advance { continue; }
            let mut best: Option<(f64, usize)> = None;
            for (j, o) in self.regiments.iter().enumerate() {
                if o.faction == r.faction || o.alive == 0 || !o.state.fighting() { continue; }
                let d = ((o.centre.x - r.centre.x).powi(2) + (o.centre.z - r.centre.z).powi(2)).sqrt();
                if best.map(|(b, _)| d < b).unwrap_or(true) { best = Some((d, j)); }
            }
            if let Some((d, j)) = best { if d <= ARCHER_RANGE { volleys.push((rid, self.regiments[j].centre.x, self.regiments[j].centre.z)); } }
        }
        for (rid, tx, tz) in volleys {
            self.regiments[rid].volley_t = VOLLEY_COOLDOWN;
            let fac = self.regiments[rid].faction;
            let members = self.regiments[rid].members.clone();
            for id in members {
                let s = &mut self.soldiers[id as usize];
                if !s.alive() || s.ammo == 0 { continue; }
                s.ammo -= 1;
                let from = Point3::new(s.p.x, s.p.y + 1.6, s.p.z);
                let (jx, jz) = (self.rng.range(-9.0, 9.0), self.rng.range(-9.0, 9.0)); // volley scatter
                let (lx, lz) = ((tx + jx).clamp(MARGIN, WORLD - MARGIN), (tz + jz).clamp(MARGIN, WORLD - MARGIN));
                let to = Point3::new(lx, ground_h(lx, lz, seed) + 0.2, lz);
                let dist = ((to.x - from.x).powi(2) + (to.z - from.z).powi(2)).sqrt();
                self.arrows.push(Arrow { from, to, t0: now, flight: (dist / ARROW_SPEED).max(0.6), faction: fac });
            }
        }

        // 6) arrow landings: k-NN(1) at the impact point, and whoever stands
        //    there takes the roll — ANY faction; a miss lands where ballistics
        //    says, and that IS the friendly-fire model (no friend/foe check).
        let mut landings: Vec<(Point3, Faction)> = Vec::new();
        self.arrows.retain(|a| { let landed = now >= a.t0 + a.flight; if landed { landings.push((a.to, a.faction)); } !landed });
        for (at, by) in landings {
            let hit = self.index.knn(at, 1).into_iter().next().filter(|(d, _)| *d <= ARROW_KILL_R).map(|(_, it)| it.id as usize);
            let Some(v) = hit else { continue };
            let def = self.regiments[self.soldiers[v].regiment as usize].kind.defense();
            if self.rng.unit() < kill_prob(MISSILE_ATK - def) { self.kills[by.index()] += 1; self.fell(v, true); }
        }
    }
}

// -------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1.0 / 60.0;

    /// Test rig: park a regiment at (x, z) facing `facing`, soldiers on their
    /// slots, and bring the kept index current.
    fn teleport(f: &mut Formations, rid: usize, x: f64, z: f64, facing: f64) {
        let seed = f.seed;
        f.regiments[rid].centre = Point3::new(x, ground_h(x, z, seed), z);
        f.regiments[rid].facing = facing;
        f.regiments[rid].order = (x, z);
        let members = f.regiments[rid].members.clone();
        for id in members {
            let slot = f.soldiers[id as usize].slot;
            let (sx, sz) = slot_pos(&f.regiments[rid], slot);
            let (sx, sz) = (sx.clamp(MARGIN, WORLD - MARGIN), sz.clamp(MARGIN, WORLD - MARGIN));
            f.soldiers[id as usize].p = Point3::new(sx, ground_h(sx, sz, seed), sz);
            f.soldiers[id as usize].moved = true;
        }
        f.sync_index();
    }

    fn find_reg(f: &Formations, fac: Faction, kind: RKind, skip: usize) -> usize {
        f.regiments.iter().enumerate().filter(|(_, r)| r.faction == fac && r.kind == kind).map(|(i, _)| i).nth(skip).expect("roster has the kind")
    }

    #[test]
    fn deployment_forms_lines_and_nothing_stacks() {
        let f = Formations::new(42, 1200);
        // Historical lines: every Red soldier south of every Blue soldier.
        let max_red = f.soldiers.iter().filter(|s| s.faction == Faction::Red).map(|s| s.p.z).fold(f64::MIN, f64::max);
        let min_blue = f.soldiers.iter().filter(|s| s.faction == Faction::Blue).map(|s| s.p.z).fold(f64::MAX, f64::min);
        assert!(max_red < min_blue, "armies must deploy on opposite sides: {max_red} vs {min_blue}");
        // Formation slots keep soldiers apart — brute pairwise inside sampled regiments.
        for rid in (0..f.regiments.len()).step_by(3) {
            let r = &f.regiments[rid];
            for (a, &ia) in r.members.iter().enumerate() {
                for &ib in &r.members[a + 1..] {
                    let (pa, pb) = (f.soldiers[ia as usize].p, f.soldiers[ib as usize].p);
                    let d = ((pa.x - pb.x).powi(2) + (pa.z - pb.z).powi(2)).sqrt();
                    assert!(d >= r.spacing * 0.7, "soldiers stacked at deploy: {d} < spacing {}", r.spacing);
                }
            }
        }
        // Roster sanity: both sides field every kind, exactly one general each.
        for fac in Faction::ALL {
            for kind in [RKind::Sword, RKind::Spear, RKind::Archer, RKind::Cavalry] {
                assert!(f.regiments.iter().any(|r| r.faction == fac && r.kind == kind), "missing {kind:?}");
            }
            assert_eq!(f.regiments.iter().filter(|r| r.faction == fac && r.kind == RKind::General).count(), 1);
        }
    }

    #[test]
    fn sector_classification_matches_brute_angles() {
        // Brute: signed angle victim→attacker vs facing via atan2 — an
        // independent formulation of the same quadrant convention.
        let mut rng = Rng::new(99);
        for _ in 0..500 {
            let facing = rng.range(-8.0, 8.0);
            let (dx, dz) = (rng.range(-5.0, 5.0), rng.range(-5.0, 5.0));
            if dx.abs() + dz.abs() < 1e-3 { continue; }
            let mut a = dz.atan2(dx) - facing;
            while a > std::f64::consts::PI { a -= std::f64::consts::TAU; }
            while a < -std::f64::consts::PI { a += std::f64::consts::TAU; }
            let q = std::f64::consts::FRAC_PI_4;
            if (a.abs() - q).abs() < 1e-9 || (a.abs() - 3.0 * q).abs() < 1e-9 { continue; } // knife-edge tie
            let want = if a.abs() <= q { Sector::Front } else if a.abs() >= 3.0 * q { Sector::Rear } else if a > 0.0 { Sector::Left } else { Sector::Right };
            assert_eq!(sector(facing, dx, dz), want, "facing={facing} d=({dx},{dz}) a={a}");
        }
    }

    #[test]
    fn kill_roll_matches_the_design_curve() {
        assert!((kill_prob(0.0) - 0.019).abs() < 1e-12);
        assert!((kill_prob(5.0) - 0.019 * 1.2f64.powi(5)).abs() < 1e-12);
        assert_eq!(kill_prob(25.0), kill_prob(20.0), "cf clamps at +20");
        assert_eq!(kill_prob(-25.0), kill_prob(-20.0), "cf clamps at -20");
        // charge decays linearly over 13 s
        assert_eq!(charge_bonus(RKind::Cavalry, 0.0, 0.0), 8.0);
        assert!(charge_bonus(RKind::Cavalry, 0.0, CHARGE_SECS / 2.0) - 4.0 < 1e-9);
        assert_eq!(charge_bonus(RKind::Cavalry, 0.0, 20.0), 0.0);
    }

    #[test]
    fn melee_pairs_only_enemies_within_reach() {
        let mut f = Formations::new(7, 800);
        let (a, b) = (find_reg(&f, Faction::Red, RKind::Sword, 0), find_reg(&f, Faction::Blue, RKind::Sword, 0));
        teleport(&mut f, a, 700.0, 700.0, std::f64::consts::FRAC_PI_2);
        teleport(&mut f, b, 700.0, 716.0, -std::f64::consts::FRAC_PI_2);
        for _ in 0..(4.0 / DT) as usize { f.step(DT); }
        let engaged: Vec<usize> = f.soldiers.iter().enumerate().filter(|(_, s)| s.alive() && s.melee.is_some()).map(|(i, _)| i).collect();
        assert!(!engaged.is_empty(), "face-to-face regiments must pair up");
        for i in engaged {
            let t = f.soldiers[i].melee.unwrap() as usize;
            assert!(f.soldiers[i].faction != f.soldiers[t].faction, "melee pair must be enemies");
            let (pa, pb) = (f.soldiers[i].p, f.soldiers[t].p);
            let d = ((pa.x - pb.x).powi(2) + (pa.z - pb.z).powi(2)).sqrt();
            let reach = f.regiments[f.soldiers[i].regiment as usize].kind.reach();
            assert!(d <= reach + 1.5, "pair out of reach: {d} > {reach}"); // +slack: both moved since decide
        }
    }

    #[test]
    fn charge_corridor_blocked_by_a_friendly_does_not_arm() {
        let mut f = Formations::new(11, 800);
        let cav = find_reg(&f, Faction::Red, RKind::Cavalry, 0);
        let foe = find_reg(&f, Faction::Blue, RKind::Sword, 0);
        let ally = find_reg(&f, Faction::Red, RKind::Sword, 0);
        teleport(&mut f, cav, 300.0, 300.0, 0.0);
        teleport(&mut f, foe, 370.0, 300.0, std::f64::consts::PI);
        teleport(&mut f, ally, 600.0, 900.0, 0.0); // parked far away
        let (tx, tz) = (f.regiments[foe].centre.x, f.regiments[foe].centre.z);
        assert!(f.lane_clear(cav, tx, tz), "open lane must be clear");
        f.slow_tick();
        assert_eq!(f.regiments[cav].state, RState::Charging, "clear lane in range must arm the charge");
        // Reset and interpose our own infantry: the corridor raycast hits a
        // friendly body first → no charge order.
        f.regiments[cav].state = RState::Advance;
        f.regiments[cav].charge_t = -1e9;
        teleport(&mut f, ally, 335.0, 300.0, 0.0);
        assert!(!f.lane_clear(cav, tx, tz), "a friendly regiment on the lane must block it");
        f.slow_tick();
        assert!(f.regiments[cav].state != RState::Charging, "blocked corridor must not arm a charge");
    }

    #[test]
    fn morale_decimated_unsupported_routs_while_supported_holds() {
        let mut f = Formations::new(13, 1200);
        let victim = find_reg(&f, Faction::Red, RKind::Sword, 0);
        let control = find_reg(&f, Faction::Red, RKind::Sword, 1);
        // Both at 60% casualties (−8)…
        for rid in [victim, control] {
            let members = f.regiments[rid].members.clone();
            let cut = members.len() * 6 / 10;
            for &id in &members[..cut] { f.fell(id as usize, false); }
        }
        // …but the victim also has two friendly regiments routing beside it
        // (chain-rout −8): 10 − 8 − 8 < 0 → routs.
        let (vc, spear0, spear1) = (f.regiments[victim].centre, find_reg(&f, Faction::Red, RKind::Spear, 0), find_reg(&f, Faction::Red, RKind::Spear, 1));
        for (k, rid) in [spear0, spear1].into_iter().enumerate() {
            teleport(&mut f, rid, vc.x + 30.0 + k as f64 * 20.0, vc.z, std::f64::consts::FRAC_PI_2);
            f.regiments[rid].state = RState::Routing;
        }
        // …while the control stands in the general's aura (+3): 10 − 8 + 3 > 0 → holds.
        let general = find_reg(&f, Faction::Red, RKind::General, 0);
        let cc = f.regiments[control].centre;
        teleport(&mut f, control, 1000.0, 300.0, std::f64::consts::FRAC_PI_2); // away from the routers
        let cc2 = f.regiments[control].centre;
        let _ = (cc, cc2);
        teleport(&mut f, general, 1000.0 + 20.0, 300.0, std::f64::consts::FRAC_PI_2);
        f.slow_tick(); // refresh auras
        f.morale_tick(victim);
        f.morale_tick(control);
        assert_eq!(f.regiments[victim].state, RState::Routing, "decimated + chain-rout must break (morale {})", f.regiments[victim].morale);
        assert_eq!(f.regiments[victim].routs, 1);
        assert!(f.regiments[control].state.fighting(), "same casualties but supported must hold (morale {})", f.regiments[control].morale);
        // The hard floor shatters outright: pile enemy pressure until ≤ −10.
        let members = f.regiments[victim].members.clone();
        for &id in &members { if f.soldiers[id as usize].alive() && f.regiments[victim].alive > f.regiments[victim].n0 / 10 { f.fell(id as usize, false); } }
        f.regiments[victim].state = RState::Advance; // pretend it rallied once
        f.morale_tick(victim); // 90% casualties −12, routers −8 → ≤ −10 → shattered
        assert_eq!(f.regiments[victim].state, RState::Shattered, "below the hard floor never returns (morale {})", f.regiments[victim].morale);
    }

    #[test]
    fn routers_take_free_kills_and_do_not_fight_back() {
        let mut f = Formations::new(17, 800);
        let cav = find_reg(&f, Faction::Red, RKind::Cavalry, 0);
        let prey = find_reg(&f, Faction::Blue, RKind::Sword, 0);
        teleport(&mut f, cav, 700.0, 690.0, std::f64::consts::FRAC_PI_2);
        teleport(&mut f, prey, 700.0, 700.0, -std::f64::consts::FRAC_PI_2);
        f.regiments[prey].state = RState::Routing;
        f.regiments[prey].routs = 1;
        let (cav0, prey0) = (f.regiments[cav].alive, f.regiments[prey].alive);
        for _ in 0..(6.0 / DT) as usize { f.step(DT); }
        assert!(f.regiments[prey].alive < prey0, "pursuers in reach must cut routers down");
        assert_eq!(f.regiments[cav].alive, cav0, "routers don't fight back — the cavalry takes zero losses");
        assert!(f.regiments[prey].members.iter().all(|&id| f.soldiers[id as usize].melee.is_none()), "routing soldiers must never hold a melee target");
    }

    #[test]
    fn archers_volley_and_arrows_kill_friendlies_too() {
        let mut f = Formations::new(19, 800);
        let arch = find_reg(&f, Faction::Red, RKind::Archer, 0);
        let foe = find_reg(&f, Faction::Blue, RKind::Sword, 0);
        teleport(&mut f, arch, 700.0, 600.0, std::f64::consts::FRAC_PI_2);
        teleport(&mut f, foe, 700.0, 690.0, -std::f64::consts::FRAC_PI_2);
        let mut fired = false;
        for _ in 0..(3.0 / DT) as usize { f.step(DT); fired |= !f.arrows.is_empty(); }
        assert!(fired, "archers in range must loose a volley");
        assert!(f.regiments[arch].members.iter().any(|&id| f.soldiers[id as usize].ammo < AMMO), "volleys must spend ammo");
        // Friendly fire: inject arrows landing dead on our OWN sword line —
        // the landing k-NN(1) has no friend/foe check, so friends die.
        let own = find_reg(&f, Faction::Red, RKind::Sword, 0);
        let before = f.regiments[own].alive;
        let targets: Vec<Point3> = f.regiments[own].members.iter().filter(|&&id| f.soldiers[id as usize].alive()).map(|&id| f.soldiers[id as usize].p).collect();
        let now = f.now;
        for k in 0..400 {
            let to = targets[k % targets.len()];
            f.arrows.push(Arrow { from: Point3::new(to.x, to.y + 20.0, to.z - 40.0), to, t0: now, flight: 0.1, faction: Faction::Red });
        }
        for _ in 0..12 { f.step(DT); }
        assert!(f.regiments[own].alive < before, "arrows must kill whoever stands at the landing point — friends included");
    }

    #[test]
    fn keep_index_matches_brute_through_battle_churn() {
        // Movement (update_ref) and deaths (remove_ref) churn the kept index —
        // it must match brute force over live positions at every sample.
        let mut f = Formations::new(31, 700);
        // Slam the two sword lines together so kills happen inside the window.
        let (a, b) = (find_reg(&f, Faction::Red, RKind::Sword, 0), find_reg(&f, Faction::Blue, RKind::Sword, 0));
        teleport(&mut f, a, 700.0, 690.0, std::f64::consts::FRAC_PI_2);
        teleport(&mut f, b, 700.0, 710.0, -std::f64::consts::FRAC_PI_2);
        let alive0 = f.soldiers.iter().filter(|s| s.alive()).count();
        for frame in 0..300 {
            f.step(DT);
            if frame % 30 != 0 { continue; }
            f.sync_index(); // bring the index current with post-apply positions
            let alive = f.soldiers.iter().filter(|s| s.alive()).count();
            assert_eq!(f.index.item_count(), alive, "index count != alive at frame {frame}");
            let q = Sphere3::new(700.0, 0.0, 700.0, 120.0);
            let mut got: Vec<u32> = f.index.cull(&q).iter().map(|it| it.id).collect();
            let mut want: Vec<u32> = f.soldiers.iter().enumerate()
                .filter(|(_, s)| s.alive() && { let (dx, dy, dz) = (s.p.x - 700.0, s.p.y, s.p.z - 700.0); dx * dx + dy * dy + dz * dz <= 120.0 * 120.0 })
                .map(|(i, _)| i as u32).collect();
            got.sort(); want.sort();
            assert_eq!(got, want, "kept index != brute at frame {frame}");
        }
        assert!(f.soldiers.iter().filter(|s| s.alive()).count() < alive0, "the melee must actually kill (churn the index)");
    }

    #[test]
    fn same_seed_same_battle() {
        // No rng in the parallel decide path + serial rolls ⇒ a seed fully
        // determines the battle, thread count and all.
        let (mut a, mut b) = (Formations::new(77, 600), Formations::new(77, 600));
        for _ in 0..600 { a.step(DT); b.step(DT); }
        assert_eq!(a.kills, b.kills);
        assert_eq!(a.arrows.len(), b.arrows.len());
        let pos = |f: &Formations| f.soldiers.iter().flat_map(|s| [s.p.x.to_bits(), s.p.z.to_bits(), s.hp.to_bits()]).collect::<Vec<u64>>();
        assert_eq!(pos(&a), pos(&b), "same seed must replay bit-identically");
        let st = |f: &Formations| f.regiments.iter().map(|r| r.state).collect::<Vec<RState>>();
        assert_eq!(st(&a), st(&b));
    }

    #[test]
    fn a_battle_reaches_an_outcome_and_resets() {
        let mut f = Formations::new(5, 360);
        // Regiment sizing quantizes the request (ranks × files), so compare the
        // reset against the ACTUAL fresh-army size, not the requested 360.
        let (r0, b0) = f.counts();
        let dt = 1.0 / 30.0;
        let mut steps = 0usize;
        while f.game_over.is_none() && f.now < 600.0 { f.step(dt); steps += 1; }
        assert!(f.game_over.is_some(), "a battle must resolve within 600 s (ran {steps} steps, standing {:?})", f.standing());
        let winner = f.outcome().unwrap();
        let loser_up = f.regiments.iter().any(|r| r.faction == winner.other() && r.alive > 0 && r.state.fighting());
        assert!(!loser_up, "the loser must have no fighting regiments left");
        for _ in 0..(13.0 / dt) as usize { f.step(dt); if f.run > 1 { break; } }
        assert_eq!(f.run, 2, "the field must reset ~12 s after the rout");
        assert!(f.game_over.is_none());
        let (r, b) = f.counts();
        assert!(r == r0 && b == b0, "fresh armies after the reset: {r}/{b} (fresh = {r0}/{b0})");
    }
}
