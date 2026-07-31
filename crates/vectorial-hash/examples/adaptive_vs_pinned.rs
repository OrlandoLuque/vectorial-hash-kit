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

fn script(n: usize, per_act: usize) -> Vec<Frame> {
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
    // Act 3: the query storm — one cull per item, far above rebuild_query_ratio.
    for _ in 0..per_act {
        f.push(Frame { act: 2, want: n, movers: n / 4, culls: n });
    }
    // Act 4: frozen. Nothing moves; a build-once structure should take over.
    for _ in 0..per_act {
        f.push(Frame { act: 3, want: n, movers: 0, culls: n / 8 });
    }
    f
}

/// Run the script once. `pin` forces a backend and disables migration by making every
/// threshold unreachable; `None` lets the policy do its job.
fn run(script: &[Frame], pin: Option<Backend>) -> ([f64; 4], Vec<Backend>, usize) {
    let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
    // Pinning is done through the thresholds rather than by adding a "pinned" mode to the
    // index: the point is to compare against the index carrying ONE structure, with the same
    // item storage and the same call path, so the difference is the policy and nothing else.
    let th = match pin {
        None => Thresholds::default(),
        Some(Backend::Brute) => Thresholds { brute_max: usize::MAX, ..Default::default() },
        Some(Backend::KeepTree) => Thresholds { brute_max: 0, rebuild_query_ratio: f64::MAX, static_ticks: u32::MAX, ..Default::default() },
        Some(Backend::Grid) => Thresholds { brute_max: 0, rebuild_query_ratio: 0.0, static_ticks: u32::MAX, ..Default::default() },
        Some(Backend::Static) => Thresholds { brute_max: 0, static_ticks: 0, ..Default::default() },
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
    (acts, seen, ix.switch_count() as usize)
}

fn main() {
    let n: usize = std::env::var("AV_N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let per_act: usize = std::env::var("AV_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let s = script(n, per_act);

    println!("adaptive vs pinned | peak {n} items | {per_act} frames per act, {} total", s.len());
    println!("acts: 1 small+quiet · 2 growing+churning · 3 query storm (1 cull/item) · 4 frozen\n");

    // Adaptive first and last, so a warm-cache advantage cannot land entirely on it — the
    // interleaving idea from docs/MEASURING.md, applied where compare2 does not fit.
    let (warm, _, _) = run(&s, None);
    std::hint::black_box(warm);

    let mut rows: Vec<(String, [f64; 4], usize)> = Vec::new();
    for pin in [Some(Backend::Brute), Some(Backend::KeepTree), Some(Backend::Grid), Some(Backend::Static)] {
        let (a, _, _) = run(&s, pin);
        rows.push((format!("pinned {:?}", pin.unwrap()), a, 0));
    }
    let (a, seen, switches) = run(&s, None);
    rows.push(("AdaptiveIndex".into(), a, switches));

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
    println!("\nbackends the policy chose, in order: {seen:?}");
    println!("(vs best pin > 1 means faster than the best fixed choice. The best pin is the one");
    println!("you would have had to know in advance — the adaptive index is worth its complexity");
    println!("only if it is close to it without being told.)");
}
