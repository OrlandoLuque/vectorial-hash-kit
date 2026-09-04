//! A workload you can *change while it runs*, so the adaptive index can be watched deciding.
//!
//! `AdaptiveIndex` exists for a workload whose character changes. Every demo in this repo that
//! uses it has a workload that does **not**: `fluid_wgpu`'s SPH tank looks the same in frame
//! 10 000 as in frame 100, and the horde does change but its adaptive arm walks
//! `Brute → KeepTree` and stops. The one varying-load workload we had was
//! `examples/adaptive_vs_pinned`, which is headless, synthetic, and reports a total — and a total
//! cannot distinguish a policy that flapped from one that lagged from one that simply chose
//! wrong. Those three look completely different on a screen.
//!
//! So this is a scene with four knobs, one per boundary the policy reasons about:
//!
//! | knob | crosses | the rule it provokes |
//! | --- | --- | --- |
//! | population | [`Thresholds::brute_max`] | is an index worth having at all? |
//! | queries per item | [`Thresholds::rebuild_query_ratio`] | grid or keep-tree? |
//! | query radius | [`Thresholds::grid_min_hits`] | do the queries FIND anything? |
//! | freeze | [`Thresholds::static_ticks`] | has everything stopped moving? |
//!
//! Graphics-free on purpose, like `horde_sim` and `siege_sim`: the renderer and the tests drive
//! the same [`Lab::step`], so a green test says something about what is on screen.
//!
//! **Pinned arms are the same index with the policy switched off.** [`Lab::bakeoff`] clones the
//! live state, [`AdaptiveIndex2::migrate_to`]s each backend and [`AdaptiveIndex2::freeze`]s it.
//! Building four separate structures by hand would compare four code paths; this compares one
//! code path with and without the decision, which is the thing under test — and it exercises the
//! two API calls that exist precisely so a caller can overrule the policy.

use std::time::Instant;
use vectorial_hash::{AdaptiveIndex2, Backend, Circle, Point, Positioned, Rect, Slot, Thresholds};

/// World size, in world units. Big enough that a radius-120 query is still local.
pub const W: f64 = 1200.0;
pub const H: f64 = 760.0;
/// Hard ceiling on the population slider — the point is to cross thresholds, not to benchmark
/// scale, and everything here has to stay interactive on a laptop.
pub const MAX_N: usize = 20_000;
/// How many frames of backend history the timeline strip keeps.
pub const HISTORY: usize = 600;

pub fn world() -> Rect { Rect::new(0.0, 0.0, W, H) }

/// One indexed agent. `id` is its own, stable across everything; `Slot` is the index's handle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agent {
    pub id: u32,
    pub p: Point,
    pub v: Point,
    /// Was this agent returned by any query on the last step? Purely for the renderer.
    pub hit: bool,
}
impl Positioned for Agent { fn position(&self) -> Point { self.p } }

/// The four knobs, plus pause. Everything the viewer can change.
#[derive(Clone, Copy, Debug)]
pub struct Knobs {
    pub population: usize,
    /// Culls issued per item per step. 0.0 means nobody is asking — where a scan wins however
    /// many items there are.
    pub queries_per_item: f64,
    pub radius: f64,
    /// Fraction of the population that moves each step.
    pub churn: f64,
    /// Nothing moves at all: the build-once backend's regime.
    pub frozen: bool,
    pub paused: bool,
}

impl Default for Knobs {
    fn default() -> Self {
        // Deliberately starts small and quiet, i.e. in BRUTE's regime, so the first thing a
        // viewer does — drag population up — produces a migration rather than nothing.
        Knobs { population: 40, queries_per_item: 0.4, radius: 26.0, churn: 0.5, frozen: false, paused: false }
    }
}

/// What one step cost and found. Times are this machine's; the counts are not.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub maintain_us: f64,
    pub query_us: f64,
    pub queries: usize,
    pub hits: usize,
    /// The policy's own prediction for the last step, against `hits / queries`.
    pub predicted_hits: f64,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.f() * (hi - lo) }
}

/// How long the policy spends WANTING a backend it is not allowed to take yet.
///
/// `#149` blames the adaptive index's 0.70x on detector lag plus the migration's own rebuild, and
/// that was an attribution, never a measurement. This is the half that can be measured **exactly**
/// — it is a count of steps, identical on every machine, so it is worth more here than a
/// millisecond figure from an unfamiliar laptop.
///
/// The quantity is `backend() != recommended()`: the policy has decided, and hysteresis (the hold
/// window, the cooldown) is holding it.
///
/// **`wanting` deliberately duplicates the library's `SwitchStats::near_misses`**, which counts
/// the same event from inside `tick`. I measured it here first and only then found the existing
/// counter — and rather than delete one, the two are cross-checked by a test. A hand-rolled
/// counter that agrees with the library's is a check on both; one that quietly disagrees is a bug
/// in whichever is wrong, and there would be no way to tell without the other.
///
/// What is genuinely new is `lags`: near-misses is one running total, and this is the
/// **distribution** — how long each individual wish waited. A policy that waits 20 steps five
/// times and one that waits 100 once produce the same total and want completely different fixes.
#[derive(Clone, Debug, Default)]
pub struct LagStats {
    /// Steps on which the live backend was not the recommended one.
    pub wanting: u64,
    /// For each migration, how many consecutive steps it had been wanted first.
    pub lags: Vec<u32>,
    /// Consecutive steps the CURRENT unfulfilled wish has lasted.
    pub run: u32,
}

impl LagStats {
    pub fn mean_lag(&self) -> f64 {
        if self.lags.is_empty() { 0.0 } else { self.lags.iter().map(|&l| l as f64).sum::<f64>() / self.lags.len() as f64 }
    }
    pub fn max_lag(&self) -> u32 { self.lags.iter().copied().max().unwrap_or(0) }
    /// Share of all steps spent on a backend the policy had already rejected.
    pub fn wanting_fraction(&self, steps: u64) -> f64 {
        if steps == 0 { 0.0 } else { self.wanting as f64 / steps as f64 }
    }
}

pub struct Lab {
    pub ix: AdaptiveIndex2<Agent>,
    /// The mirror of what the index holds, in `id` order — the brute-force reference the tests
    /// compare against, and the renderer's vertex source.
    pub agents: Vec<Agent>,
    slots: Vec<Slot>,
    pub knobs: Knobs,
    pub stats: FrameStats,
    /// Backend per recent step, oldest first. The timeline strip is this, drawn.
    pub history: Vec<Backend>,
    /// See [`LagStats`] — how much of the run is spent knowing better.
    pub lag: LagStats,
    /// Steps taken. Migrations divided by this is the flap rate.
    pub steps: u64,
    next_id: u32,
    rng: Rng,
}

impl Lab {
    pub fn new(seed: u64) -> Self {
        Self::with_thresholds(seed, Thresholds::from_env())
    }

    pub fn with_thresholds(seed: u64, th: Thresholds) -> Self {
        let mut lab = Lab {
            ix: AdaptiveIndex2::with_thresholds(world(), 8, th),
            agents: Vec::new(),
            slots: Vec::new(),
            knobs: Knobs::default(),
            stats: FrameStats::default(),
            history: Vec::new(),
            lag: LagStats::default(),
            steps: 0,
            next_id: 0,
            rng: Rng(seed | 1),
        };
        lab.resize(lab.knobs.population);
        lab
    }

    /// Grow or shrink to `n`, keeping everyone who stays. Shrinking removes the newest, so
    /// dragging the slider down and back up does not reshuffle the whole scene under the viewer.
    pub fn resize(&mut self, n: usize) {
        let n = n.min(MAX_N);
        while self.agents.len() > n {
            let s = self.slots.pop().expect("slots and agents stay in step");
            self.ix.remove(s);
            self.agents.pop();
        }
        while self.agents.len() < n {
            let a = Agent {
                id: self.next_id,
                p: Point::new(self.rng.range(0.0, W), self.rng.range(0.0, H)),
                v: Point::new(self.rng.range(-90.0, 90.0), self.rng.range(-90.0, 90.0)),
                hit: false,
            };
            self.next_id += 1;
            self.slots.push(self.ix.insert(a));
            self.agents.push(a);
        }
        self.knobs.population = self.agents.len();
    }

    /// One step: move a churn-fraction, issue the query load, let the policy tick.
    pub fn step(&mut self, dt: f64) {
        if self.knobs.paused { return; }
        if self.knobs.population != self.agents.len() { self.resize(self.knobs.population); }

        let n = self.agents.len();
        let moving = if self.knobs.frozen { 0 } else { (n as f64 * self.knobs.churn).round() as usize };

        let t = Instant::now();
        for i in 0..moving {
            let mut a = self.agents[i];
            a.p.x += a.v.x * dt;
            a.p.y += a.v.y * dt;
            // Bounce, so the population stays inside the index's declared world. An agent that
            // escapes its world box is dropped by some backends and kept by others, which the
            // stealth demo learned the hard way: an index only knows what it holds.
            if a.p.x < 0.0 || a.p.x >= W { a.v.x = -a.v.x; a.p.x = a.p.x.clamp(0.0, W - 1e-6); }
            if a.p.y < 0.0 || a.p.y >= H { a.v.y = -a.v.y; a.p.y = a.p.y.clamp(0.0, H - 1e-6); }
            self.agents[i] = a;
            self.ix.update(self.slots[i], |m| *m = a);
        }
        let maintain_us = t.elapsed().as_secs_f64() * 1e6;

        for a in &mut self.agents { a.hit = false; }
        let queries = ((n as f64 * self.knobs.queries_per_item).round() as usize).min(4096);
        let mut hits = 0usize;
        let mut hit_ids: Vec<u32> = Vec::new();
        let t = Instant::now();
        for q in 0..queries {
            // Query centres follow the agents, so the load lands where the data is. Probing
            // empty map would flatter a grid and measure nothing anybody does.
            let c = self.agents[(q * n.max(1) / queries.max(1)).min(n.saturating_sub(1))].p;
            let found = self.ix.cull(&Circle::new(c, self.knobs.radius));
            hits += found.len();
            hit_ids.extend(found.iter().map(|a| a.id));
        }
        let query_us = t.elapsed().as_secs_f64() * 1e6;

        // Marking is separate from querying so the clock above measures the index, not the paint.
        if !hit_ids.is_empty() {
            hit_ids.sort_unstable();
            hit_ids.dedup();
            for a in &mut self.agents { if hit_ids.binary_search(&a.id).is_ok() { a.hit = true; } }
        }

        let before = self.ix.backend();
        self.ix.tick();
        let after = self.ix.backend();
        // Measured AFTER the tick, so a wish granted this step is not also counted as unmet.
        if after != before {
            // It moved: whatever it had been wanting for, that is the lag it paid.
            self.lag.lags.push(self.lag.run);
            self.lag.run = 0;
        } else if after != self.ix.recommended() {
            self.lag.run += 1;
            self.lag.wanting += 1;
        } else {
            self.lag.run = 0;
        }
        self.steps += 1;
        self.history.push(self.ix.backend());
        if self.history.len() > HISTORY { self.history.remove(0); }

        self.stats = FrameStats {
            maintain_us, query_us, queries, hits,
            predicted_hits: self.ix.expected_hits(self.ix.len() as f64),
        };
    }

    /// Mean items a query returned last step, or 0 when nobody asked.
    pub fn mean_hits(&self) -> f64 {
        if self.stats.queries == 0 { 0.0 } else { self.stats.hits as f64 / self.stats.queries as f64 }
    }

    /// Race the live policy against every backend PINNED, from the current state.
    ///
    /// Each arm is a clone of the same index driven through the same script; the pinned ones are
    /// [`AdaptiveIndex2::migrate_to`] plus [`AdaptiveIndex2::freeze`], so the only difference is
    /// whether the policy is allowed to change its mind. Returns `(label, us_per_step)` with the
    /// live arm first.
    ///
    /// **Counterbalanced**: the arms run forward and then in reverse, each keeping its minimum,
    /// so a machine that drifts mid-race cannot favour whoever went first. That is
    /// `docs/MEASURING.md` § 8e applied to five arms, the same way `fluid_wgpu`'s bake-off does.
    pub fn bakeoff(&self, frames: usize) -> Vec<(&'static str, f64)> {
        let arms: [(&'static str, Option<Backend>); 5] = [
            ("ADAPTIVE", None),
            ("BRUTE", Some(Backend::Brute)),
            ("KEEPTREE", Some(Backend::KeepTree)),
            ("GRID", Some(Backend::Grid)),
            ("STATIC", Some(Backend::Static)),
        ];
        let mut best = [f64::INFINITY; 5];
        let order: Vec<usize> = (0..5).chain((0..5).rev()).collect();
        for i in order {
            let (_, pin) = arms[i];
            let mut arm = self.clone_for_race();
            if let Some(b) = pin { arm.ix.migrate_to(b); arm.ix.freeze(); }
            arm.step(1.0 / 60.0); // warm: the first step after a migration pays for the build
            let t = Instant::now();
            for _ in 0..frames { arm.step(1.0 / 60.0); }
            let us = t.elapsed().as_secs_f64() * 1e6 / frames.max(1) as f64;
            best[i] = best[i].min(us);
        }
        arms.iter().zip(best).map(|((l, _), us)| (*l, us)).collect()
    }

    /// A fresh `Lab` holding the same agents, for one arm of a race.
    fn clone_for_race(&self) -> Lab {
        let mut lab = Lab {
            ix: AdaptiveIndex2::with_thresholds(world(), 8, *self.ix.thresholds()),
            agents: Vec::new(), slots: Vec::new(),
            knobs: self.knobs, stats: FrameStats::default(), history: Vec::new(), lag: LagStats::default(),
            steps: 0, next_id: self.next_id, rng: Rng(0xBEEF),
        };
        for a in &self.agents { lab.slots.push(lab.ix.insert(*a)); lab.agents.push(*a); }
        lab
    }

    /// Brute-force reference for a query, for the tests and for anyone who does not trust the
    /// index. Sorted by `id`, which is what the index's canonical order reduces to here.
    pub fn brute(&self, c: Point, r: f64) -> Vec<u32> {
        let shape = Circle::new(c, r);
        let mut v: Vec<u32> = self.agents.iter()
            .filter(|a| { let (dx, dy) = (a.p.x - c.x, a.p.y - c.y); dx * dx + dy * dy <= r * r })
            .map(|a| a.id).collect();
        v.sort_unstable();
        let _ = shape;
        v
    }

    /// The index's answer to the same query, as ids.
    pub fn indexed(&mut self, c: Point, r: f64) -> Vec<u32> {
        let mut v: Vec<u32> = self.ix.cull(&Circle::new(c, r)).iter().map(|a| a.id).collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever the knobs are doing, the index must agree with brute force. This is the gate on
    /// everything else here: a demo that shows a policy switching happily while returning the
    /// wrong neighbours would be worse than no demo.
    #[test]
    fn every_reachable_backend_answers_like_brute_force() {
        let mut seen = std::collections::HashSet::new();
        // Four settings chosen to land in four different regimes, then checked at each.
        let scripts: [(usize, f64, f64, f64, bool); 4] = [
            (30, 0.2, 20.0, 0.5, false),    // small and quiet -> a scan
            (3000, 0.05, 20.0, 1.0, false), // big, churning, few queries -> keep-tree
            (3000, 1.0, 60.0, 0.3, false),  // query storm, wide enough to find things -> grid
            (3000, 0.5, 60.0, 0.0, true),   // frozen -> build-once
        ];
        let mut lab = Lab::with_thresholds(7, Thresholds { hold_ticks: 2, cooldown: 0, static_ticks: 8, ..Default::default() });
        for (n, q, r, churn, frozen) in scripts {
            lab.knobs = Knobs { population: n, queries_per_item: q, radius: r, churn, frozen, paused: false };
            for _ in 0..40 { lab.step(1.0 / 60.0); }
            seen.insert(lab.ix.backend());
            for (i, c) in [Point::new(300.0, 300.0), Point::new(60.0, 700.0), Point::new(900.0, 120.0)].into_iter().enumerate() {
                let want = lab.brute(c, 40.0 + i as f64 * 30.0);
                let got = lab.indexed(c, 40.0 + i as f64 * 30.0);
                assert_eq!(want, got, "backend {:?} disagreed with brute force", lab.ix.backend());
            }
        }
        // Non-vacuity: a script that never left one backend would pass every assertion above
        // while testing a quarter of what it claims to.
        assert!(seen.len() >= 3, "the scripts must reach several backends, saw {seen:?}");
    }

    /// The knobs must actually drive the policy — otherwise the demo is a lava lamp.
    #[test]
    fn the_knobs_move_the_policy() {
        let th = Thresholds { hold_ticks: 2, cooldown: 0, static_ticks: 8, ..Default::default() };
        let mut lab = Lab::with_thresholds(11, th);
        lab.knobs = Knobs { population: 20, queries_per_item: 0.5, radius: 20.0, churn: 0.5, frozen: false, paused: false };
        for _ in 0..30 { lab.step(1.0 / 60.0); }
        let small = lab.ix.backend();

        lab.knobs.population = 4000;
        lab.knobs.radius = 60.0;
        lab.knobs.queries_per_item = 1.0;
        for _ in 0..60 { lab.step(1.0 / 60.0); }
        let loaded = lab.ix.backend();

        lab.knobs.frozen = true;
        for _ in 0..60 { lab.step(1.0 / 60.0); }
        let still = lab.ix.backend();

        assert_eq!(small, Backend::Brute, "20 items should be served by a scan");
        assert_ne!(loaded, Backend::Brute, "4000 items under a query storm should be indexed");
        assert_eq!(still, Backend::Static, "a frozen population should reach the build-once backend");
        assert!(lab.ix.switch_count() >= 2, "only {} migrations over that script", lab.ix.switch_count());
    }

    /// A pinned arm must stay pinned for the whole race, or the bake-off is comparing the policy
    /// against itself and would report five near-identical numbers.
    #[test]
    fn a_pinned_arm_does_not_migrate() {
        let mut lab = Lab::with_thresholds(3, Thresholds { hold_ticks: 2, cooldown: 0, ..Default::default() });
        lab.knobs = Knobs { population: 2000, queries_per_item: 0.5, radius: 40.0, churn: 0.5, frozen: false, paused: false };
        for _ in 0..30 { lab.step(1.0 / 60.0); }

        for pin in [Backend::Brute, Backend::KeepTree, Backend::Grid, Backend::Static] {
            let mut arm = lab.clone_for_race();
            arm.ix.migrate_to(pin);
            arm.ix.freeze();
            let before = arm.ix.switch_count();
            for _ in 0..40 { arm.step(1.0 / 60.0); }
            assert_eq!(arm.ix.backend(), pin, "{pin:?} arm drifted to {:?}", arm.ix.backend());
            assert_eq!(arm.ix.switch_count(), before, "{pin:?} arm migrated while frozen");
            // ...and it must still be correct, since the race would otherwise be timing a lie.
            let c = Point::new(500.0, 400.0);
            assert_eq!(arm.brute(c, 50.0), arm.indexed(c, 50.0), "{pin:?} arm answered wrongly");
        }
    }

    /// Resizing must not corrupt the handle table: shrink and regrow, then check every agent is
    /// still findable through the index at its own position.
    #[test]
    fn resizing_keeps_the_slot_table_honest() {
        let mut lab = Lab::with_thresholds(5, Thresholds::default());
        lab.resize(500);
        for _ in 0..10 { lab.step(1.0 / 60.0); }
        lab.resize(80);
        lab.resize(300);
        for _ in 0..10 { lab.step(1.0 / 60.0); }
        assert_eq!(lab.agents.len(), 300);
        assert_eq!(lab.ix.len(), 300);
        for a in lab.agents.clone() {
            let found = lab.indexed(a.p, 0.5);
            assert!(found.contains(&a.id), "agent {} vanished from the index after resizing", a.id);
        }
    }

    /// ★ The lag counter against the library's own, which counts the same event from inside
    /// `tick`. Neither is evidence on its own; agreeing, they are evidence for both.
    #[test]
    fn the_lag_counter_agrees_with_the_librarys_near_misses() {
        let th = Thresholds { hold_ticks: 4, cooldown: 30, static_ticks: 12, ..Default::default() };
        let mut lab = Lab::with_thresholds(42, th);
        // A script that actually provokes hysteresis: three regime changes, so wishes are formed,
        // held, and eventually granted.
        for (n, q, r, churn, frozen) in [(20usize, 0.3, 20.0, 0.5, false), (3000, 1.0, 60.0, 0.4, false),
                                         (3000, 0.02, 60.0, 1.0, false), (3000, 0.5, 60.0, 0.0, true)] {
            lab.knobs = Knobs { population: n, queries_per_item: q, radius: r, churn, frozen, paused: false };
            for _ in 0..80 { lab.step(1.0 / 60.0); }
        }
        let near = lab.ix.stats().near_misses;
        // Every step spent wanting either preceded a migration (and is in `lags`) or is still
        // outstanding (`run`). Nothing may be lost or double-counted between the two.
        let accounted: u64 = lab.lag.lags.iter().map(|&l| l as u64).sum::<u64>() + lab.lag.run as u64;
        assert_eq!(lab.lag.wanting, near,
                   "the lab counted {} steps of wanting, the library counted {near} near-misses", lab.lag.wanting);
        assert_eq!(accounted, lab.lag.wanting,
                   "lags {:?} + run {} does not account for {} wanting steps", lab.lag.lags, lab.lag.run, lab.lag.wanting);
        // Non-vacuity: a run with no hysteresis pressure would satisfy all of that at zero.
        assert!(near > 10, "the script must actually make the policy wait, saw {near} near-misses");
        assert!(!lab.lag.lags.is_empty(), "no migration ever completed, so no lag was ever paid");
    }

    /// The history feeding the timeline strip must be bounded and must record every step.
    #[test]
    fn history_is_bounded_and_records_each_step() {
        let mut lab = Lab::with_thresholds(9, Thresholds::default());
        lab.knobs.population = 100;
        for _ in 0..20 { lab.step(1.0 / 60.0); }
        assert_eq!(lab.history.len(), 20);
        for _ in 0..HISTORY + 50 { lab.step(1.0 / 60.0); }
        assert_eq!(lab.history.len(), HISTORY, "the strip's buffer must not grow without bound");
        // Pausing must not advance it, or the strip would scroll while nothing happens.
        lab.knobs.paused = true;
        let len = lab.history.len();
        let steps = lab.steps;
        for _ in 0..10 { lab.step(1.0 / 60.0); }
        assert_eq!((lab.history.len(), lab.steps), (len, steps));
    }
}
