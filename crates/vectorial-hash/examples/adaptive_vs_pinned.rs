//! `adaptive_vs_pinned` — is `AdaptiveIndex` worth what it costs?
//!
//! It carries four backends, hysteresis on three boundaries, a calibration file and a
//! property test, and until this bench there was **no evidence it beats picking one structure
//! and leaving it alone**. That is the question, and it is easy to answer dishonestly: any
//! *stationary* workload favours whichever fixed structure suits it, and the adaptive index
//! can only ever pay for its migrations on a workload whose character actually changes.
//!
//! So the script has four acts, and the population and the query load cross the policy's own
//! boundaries during them:
//!
//! 1. **Small and quiet** — a handful of items, few queries. A brute scan should win.
//! 2. **Growing and churning** — population climbs past `brute_max`, everything moves each
//!    frame, queries stay light. The kept tree's regime.
//! 3. **Query storm** — same population, one query per item per frame, well past
//!    `rebuild_query_ratio`. The grid's regime.
//! 4. **Frozen** — nothing moves at all for a long stretch. The build-once tree's regime.
//!
//! Then the same script runs through `AdaptiveIndex` and through each backend **pinned**, and
//! all five totals are reported. The honest outcomes are all interesting: if a pinned backend
//! wins overall, the adaptive index is not paying for itself and the docs should say so; if it
//! wins, the margin is what it is worth.
//!
//! ```bash
//! cargo run -p vectorial-hash --example adaptive_vs_pinned --release
//! ```
//! Env: `AV_N` (peak population), `AV_FRAMES` (frames per act).

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;
use vectorial_hash::{AdaptiveIndex, Aabb, Backend, Point3, Positioned3, Slot, Sphere3, Thresholds};

const W: f64 = 1000.0;

#[derive(Clone, Copy)]
struct P {
    p: Point3,
}
impl Positioned3 for P {
    fn position(&self) -> Point3 {
        self.p
    }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f64) / (1u64 << 24) as f64
    }
    fn r(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
}

/// What one frame of the script asks for: how many items should exist, how many of them move,
/// and how many culls to run. Written out as data so both runs execute exactly the same thing.
#[derive(Clone, Copy)]
struct Frame {
    act: usize,
    want: usize,
    movers: usize,
    culls: usize,
}

/// `cycles` repeats the churn/storm pair, which is the only way this bench can price a
/// migration: at one pass there are two of them in a second and a half, far below what the
/// run-to-run spread can resolve. Cycling drives the policy back and forth across
/// `rebuild_query_ratio` so the migrations multiply while everything else stays identical.
fn script(n: usize, per_act: usize, cycles: usize) -> Vec<Frame> {
    let mut f = Vec::new();
    // Act 1: small and quiet — below any sane brute_max, barely queried.
    for _ in 0..per_act {
        f.push(Frame { act: 0, want: 60, movers: 6, culls: 2 });
    }
    // Act 2: growing past the brute edge, everything moving, few queries.
    for i in 0..per_act {
        let want = 60 + (n - 60) * (i + 1) / per_act;
        f.push(Frame { act: 1, want, movers: want, culls: 8 });
    }
    // Act 3: the query storm — one cull per item, far above rebuild_query_ratio. Repeated with
    // act 2 when `cycles > 1`, so the policy has to migrate each way every time.
    for _ in 0..per_act {
        f.push(Frame { act: 2, want: n, movers: n / 4, culls: n });
    }
    for _ in 1..cycles {
        for _ in 0..per_act { f.push(Frame { act: 1, want: n, movers: n, culls: 8 }); }
        for _ in 0..per_act { f.push(Frame { act: 2, want: n, movers: n / 4, culls: n }); }
    }
    // Act 4: frozen. Nothing moves; a build-once structure should take over.
    for _ in 0..per_act {
        f.push(Frame { act: 3, want: n, movers: 0, culls: n / 8 });
    }
    f
}

/// Run the script once. `pin` forces a backend and disables migration by making every
/// threshold unreachable; `None` lets the policy do its job.
fn run(script: &[Frame], pin: Option<Backend>, warm_start: bool) -> ([f64; 4], Vec<Backend>, usize, u32) {
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
    // Pinning is done through the thresholds rather than by adding a "pinned" mode to the
    // index: the point is to compare against the index carrying ONE structure, with the same
    // item storage and the same call path, so the difference is the policy and nothing else.
    // `scan_budget: 0.0` on every pinned config: the scan rule was added after this bench and
    // silently un-pinned three of the four baselines, which moved their times by up to 25% and
    // would have made the improvement look like whatever it wanted to look like.
    // The detector's smoothing weight, swept from the environment so the lag can be priced
    // rather than argued about. Only the adaptive run is affected; the pinned baselines never
    // consult it.
    let alpha: f64 = std::env::var("AV_ALPHA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let th = match pin {
        None => Thresholds { detector_alpha: alpha, warm_start, ..Default::default() },
        Some(Backend::Brute) => Thresholds { brute_max: usize::MAX, ..Default::default() },
        Some(Backend::KeepTree) => Thresholds { brute_max: 0, scan_budget: 0.0, rebuild_query_ratio: f64::MAX, static_ticks: u32::MAX, ..Default::default() },
        Some(Backend::Grid) => Thresholds { brute_max: 0, scan_budget: 0.0, rebuild_query_ratio: 0.0, static_ticks: u32::MAX, ..Default::default() },
        Some(Backend::Static) => Thresholds { brute_max: 0, scan_budget: 0.0, static_ticks: 0, ..Default::default() },
    };
    let mut ix: AdaptiveIndex<P> = AdaptiveIndex::with_thresholds(world, 16, th);
    let mut rng = Rng(0x5EED_1234);
    let mut slots: Vec<Slot> = Vec::new();
    let mut pos: Vec<Point3> = Vec::new();
    let mut seen: Vec<Backend> = Vec::new();
    let mut sink = 0usize;

    let mut acts = [0.0f64; 4];
    for fr in script {
        let t0 = Instant::now();
        while slots.len() < fr.want {
            let p = Point3::new(rng.r(0.0, W - 0.1), rng.r(0.0, W - 0.1), rng.r(0.0, W - 0.1));
            slots.push(ix.insert(P { p }));
            pos.push(p);
        }
        for i in 0..fr.movers.min(slots.len()) {
            let old = pos[i];
            let np = Point3::new(
                (old.x + rng.r(-30.0, 30.0)).clamp(0.0, W - 0.1),
                (old.y + rng.r(-30.0, 30.0)).clamp(0.0, W - 0.1),
                (old.z + rng.r(-30.0, 30.0)).clamp(0.0, W - 0.1),
            );
            ix.update(slots[i], |c| c.p = np);
            pos[i] = np;
        }
        for c in 0..fr.culls {
            let q = pos[(c * 7919) % pos.len()];
            sink += ix.cull(&Sphere3::new(q.x, q.y, q.z, 40.0)).len();
        }
        let b = ix.tick();
        acts[fr.act] += t0.elapsed().as_secs_f64() * 1e3;
        if seen.last() != Some(&b) {
            seen.push(b);
        }
    }
    std::hint::black_box(sink);
    (acts, seen, ix.switch_count() as usize, ix.warm_starts())
}

fn main() {
    let n: usize = std::env::var("AV_N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let per_act: usize = std::env::var("AV_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let cycles: usize = std::env::var("AV_CYCLES").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let s = script(n, per_act, cycles);

    println!("adaptive vs pinned | peak {n} items | {per_act} frames per act, {} total | {cycles} cycle(s)", s.len());
    println!("acts: 1 small+quiet · 2 growing+churning · 3 query storm (1 cull/item) · 4 frozen\n");

    // Adaptive first and last, so a warm-cache advantage cannot land entirely on it — the
    // interleaving idea from docs/MEASURING.md, applied where compare2 does not fit.
    let (warm, _, _, _) = run(&s, None, true);
    std::hint::black_box(warm);

    let mut rows: Vec<(String, [f64; 4], usize)> = Vec::new();
    for pin in [Some(Backend::Brute), Some(Backend::KeepTree), Some(Backend::Grid), Some(Backend::Static)] {
        let (a, _, _, _) = run(&s, pin, true);
        rows.push((format!("pinned {:?}", pin.unwrap()), a, 0));
    }
    let (a, seen, switches, warmed) = run(&s, None, true);
    rows.push(("AdaptiveIndex".into(), a, switches));
    // The same policy with warm-start migration OFF. Paired inside this process against the row
    // above, because the run-to-run spread of the total (0.53-0.75x against the best pin) is far
    // wider than anything a cheaper migration can contribute: an unpaired before/after here
    // reports noise, in whichever direction it happens to fall.
    let (a_cold, _, sw_cold, _) = run(&s, None, false);
    rows.push(("AdaptiveIndex (cold)".into(), a_cold, sw_cold));

    let total = |a: &[f64; 4]| a.iter().sum::<f64>();
    let best_pinned = rows.iter().take(4).map(|r| total(&r.1)).fold(f64::MAX, f64::min);
    println!("{:<20} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11} {:>9}",
        "index", "1 quiet", "2 churn", "3 storm", "4 frozen", "total ms", "vs best pin", "switches");
    for (name, a, sw) in &rows {
        println!("{name:<20} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>11.1} {:>10.2}x {sw:>9}",
            a[0], a[1], a[2], a[3], total(a), best_pinned / total(a));
        println!("#M {}.total_ms {:.2} ms", name.replace(' ', "_"), total(a));
        for (i, v) in a.iter().enumerate() { println!("#M {}.act{}_ms {v:.2} ms", name.replace(' ', "_"), i + 1); }
    }
    // Per act, which pinned backend would have been right — the thing the policy is trying to
    // guess, shown next to what it actually did.
    println!();
    for act in 0..4 {
        let (mut who, mut best) = ("", f64::MAX);
        for (name, a, _) in rows.iter().take(4) {
            if a[act] < best { best = a[act]; who = name; }
        }
        let adaptive = rows.last().map(|r| r.1[act]).unwrap_or(0.0);
        println!("  act {}: best fixed is {who} ({best:.1} ms) — adaptive spent {adaptive:.1} ms ({:.2}x)",
            act + 1, best / adaptive);
    }
    // The warm-start question on its own terms. The total is the wrong denominator when only
    // the migrations changed, so divide by them.
    // How many of the migrations could ACTUALLY warm-start. Only a grid can hand over an
    // order for free, so a script that never leaves a grid gets none — and then a zero result
    // means "never ran", not "does not help".
    let warm_starts_line = if warmed == 0 {
        format!("0 of {switches} migrations could warm-start: none of them LEFT a grid, and only a
  grid can hand over an order for free. The figure above is therefore two identical code paths
  timed twice — noise, not a verdict. This bench cannot price the feature until a keep-tree can
  also supply an order (see docs/BACKLOG.md).")
    } else {
        format!("{warmed} of {switches} migrations warm-started")
    };
    let warm_total = rows[rows.len() - 2].1.iter().sum::<f64>();
    let cold_total = rows[rows.len() - 1].1.iter().sum::<f64>();
    println!();
    println!("warm-start migration: {warm_total:.1} ms warm vs {cold_total:.1} ms cold, over {switches} migrations");
    println!("  = {:+.2} ms per migration ({:.3}x on the total)",
        (cold_total - warm_total) / switches.max(1) as f64, cold_total / warm_total);
    println!("  {warm_starts_line}");
    // Why this figure is not worth reading, quantitatively rather than as a shrug.
    let build_share = 100.0 * (switches as f64 * 7.0) / warm_total.max(1.0);
    println!("  NOTE: the {switches} migrations are about {build_share:.1}% of this {warm_total:.0} ms run, and");
    println!("  the warm start saves 7-45% OF THAT — so the effect on the total is well under 1%,");
    println!("  against a run-to-run spread of tens of percent. Cycling the script 6x multiplied");
    println!("  the migrations by six and the per-migration figure still read +13.8, +5.3 and -3.5");
    println!("  ms on three consecutive runs. This bench cannot price the warm start and no amount");
    println!("  of repetition inside it will change that; examples/migration_warm_start measures");
    println!("  the build directly (1.07x grid, 1.42x k-d tree, 1.81x tree inserts). What this");
    println!("  bench IS good for is the count above: whether the thing can fire at all.");
    println!("#M warm_start.per_migration_ms {:.3} ms", (cold_total - warm_total) / switches.max(1) as f64);

    println!("\nbackends the policy chose, in order: {seen:?}");
    println!("(vs best pin > 1 means faster than the best fixed choice. The best pin is the one");
    println!("you would have had to know in advance — the adaptive index is worth its complexity");
    println!("only if it is close to it without being told.)");
}
