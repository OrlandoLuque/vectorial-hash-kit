//! Deterministic performance **regression gate** — the committed counterpart to
//! the Criterion benches. Criterion gives rich local reports but its baselines
//! live under `target/` and CI runners are too noisy to gate on. This gate is
//! the opposite: a small, fixed, low-variance set of timings (median of many
//! reps) checked against a **committed** baseline, so a real regression can fail
//! a build.
//!
//! ```bash
//! # capture the baseline on this machine (run on a quiet system):
//! cargo run -p vectorial-hash --example regression_gate --release -- --save
//! # later / in CI — compare, exit 1 if any op regressed past the threshold:
//! cargo run -p vectorial-hash --example regression_gate --release
//! cargo run -p vectorial-hash --example regression_gate --release -- --threshold 0.30
//! ```
//!
//! The baseline file is `benches/baseline.tsv` (op<TAB>nanoseconds). It is
//! hardware-specific, and the file says so: every baseline carries a
//! `# machine = ...` line and the gate REFUSES to report a regression against
//! one from elsewhere (or one with no fingerprint at all), printing the table as
//! orientation only. Keep a second machine's numbers beside the committed ones
//! with `--save --local`. The gate compares *ratios*,
//! so it is robust to absolute speed as long as the baseline was taken here.

// The `x < lo || x > hi` bounce test reads clearer than `!(lo..=hi).contains()`.
#![allow(clippy::manual_range_contains)]

use std::hint::black_box;
use std::time::Instant;
use vectorial_hash::{Aabb, Circle, KdTree2, KdTree3, LinearOctree3, LinearQuadTree, MortonGrid, MortonGrid3, Octree3, Point, Point3, Positioned, Positioned3, QuadTree, Rect, Sphere3, Tree, Tree3};

const WORLD: f64 = 512.0;
const N: usize = 20_000;
const IL: usize = 8;
const VISION: f64 = 36.0;
const N_QUERY: usize = 64;
/// 2D query radius. NOT the same as the 3D one on purpose: at r=36 a disc over 20k
/// points in a 512x512 world holds ~310 of them while a sphere in 512^3 holds ~29, so a
/// shared radius would make the 2D ops measure result-vector growth (and be ~15% noisy
/// run to run) instead of the descent the 3D ops measure. r=12 puts both at ~30 hits.
const VISION_2D: f64 = 12.0;
const MARGIN: f64 = 4.0;
const REPS: usize = 40;
const BASELINE: &str = "crates/vectorial-hash/benches/baseline.tsv";

/// Which baseline file this run uses.
///
/// The committed `baseline.tsv` belongs to whichever machine last ran `--save`, and on any other
/// machine it can only ever be orientation. So a machine may keep its own beside it,
/// `baseline.<machine-slug>.tsv`, preferred automatically when present. `$VH_BASELINE` overrides
/// both, for someone who keeps baselines elsewhere entirely.
///
/// The point of `--local` is that a second machine must not silently overwrite the first one's
/// numbers — a baseline is judged against for months, so losing it is expensive and quiet.
fn baseline_path(local: bool) -> String {
    if let Ok(p) = std::env::var("VH_BASELINE") { return p; }
    let mine = format!("crates/vectorial-hash/benches/baseline.{}.tsv", vectorial_hash::machine_slug());
    if local || std::path::Path::new(&mine).exists() { return mine; }
    BASELINE.to_string()
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

#[derive(Clone, Copy)]
struct C3 { id: u32, p: Point3 }
impl Positioned3 for C3 { fn position(&self) -> Point3 { self.p } }

fn items() -> Vec<C3> {
    let mut r = Rng::new(1);
    (0..N).map(|id| C3 { id: id as u32, p: Point3::new(r.range(MARGIN, WORLD - MARGIN), r.range(MARGIN, WORLD - MARGIN), r.range(MARGIN, WORLD - MARGIN)) }).collect()
}
fn vels() -> Vec<(f64, f64, f64)> {
    let mut r = Rng::new(5);
    (0..N).map(|_| { let s = r.range(0.35 * 120.0, 120.0); let (a, b) = (r.range(0.0, std::f64::consts::TAU), r.range(-1.0, 1.0)); let h = (1.0_f64 - b * b).max(0.0).sqrt(); (s * h * a.cos(), s * h * a.sin(), s * b) }).collect()
}
fn queries() -> Vec<Sphere3> {
    let mut r = Rng::new(99);
    (0..N_QUERY).map(|_| Sphere3::new(r.range(0.0, WORLD), r.range(0.0, WORLD), r.range(0.0, WORLD), VISION)).collect()
}
#[derive(Clone, Copy)]
struct C2 { p: Point }
impl Positioned for C2 { fn position(&self) -> Point { self.p } }

fn items2() -> Vec<C2> {
    let mut r = Rng::new(3);
    (0..N).map(|_| C2 { p: Point::new(r.range(MARGIN, WORLD - MARGIN), r.range(MARGIN, WORLD - MARGIN)) }).collect()
}
fn queries2() -> Vec<Circle> {
    let mut r = Rng::new(101);
    (0..N_QUERY).map(|_| Circle::new(Point::new(r.range(0.0, WORLD), r.range(0.0, WORLD)), VISION_2D)).collect()
}

#[inline]
fn step(p: &mut Point3, v: &mut (f64, f64, f64)) -> Point3 {
    let dt = 1.0 / 60.0;
    let mut nx = p.x + v.0 * dt; let mut ny = p.y + v.1 * dt; let mut nz = p.z + v.2 * dt;
    if nx < MARGIN || nx > WORLD - MARGIN { v.0 = -v.0; nx = nx.clamp(MARGIN, WORLD - MARGIN); }
    if ny < MARGIN || ny > WORLD - MARGIN { v.1 = -v.1; ny = ny.clamp(MARGIN, WORLD - MARGIN); }
    if nz < MARGIN || nz > WORLD - MARGIN { v.2 = -v.2; nz = nz.clamp(MARGIN, WORLD - MARGIN); }
    *p = Point3::new(nx, ny, nz); *p
}

/// **Min-of-`REPS`** nanoseconds for one op, after a short warmup. The fastest
/// sample is the truest measure of the code's cost: noise (interrupts,
/// scheduling, turbo dips) only ever *adds* time, so the minimum is the run
/// least disturbed by it — far more stable than the median for microbenchmarks.
/// The closure returns a blackholed accumulator so the work cannot be elided.
fn bench<F: FnMut() -> u64>(mut f: F) -> f64 {
    for _ in 0..5 { black_box(f()); }
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        let acc = f();
        let ns = t.elapsed().as_nanos() as f64;
        black_box(acc);
        if ns < best { best = ns; }
    }
    best
}

/// A fixed, allocation-free CPU workload used to **normalise away global clock
/// scaling** (turbo/thermal/background load) between the baseline run and a
/// later run. Every op is compared as a ratio to this number, so a machine
/// running, say, 1.3× slower overall does not read as a regression — only a
/// *relative* slowdown of the op against the CPU does. Deterministic, so it is
/// itself a stable yardstick.
fn calibrate() -> f64 {
    bench(|| {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for i in 0..4_000_000u64 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(i | 1);
            x ^= x >> 29;
        }
        x
    })
}

fn measure() -> Vec<(&'static str, f64)> {
    let aabb = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let its = items();
    let qs = queries();
    let levels = MortonGrid3::<C3>::levels_for_cell_size(aabb, VISION);
    let mut out = Vec::new();

    out.push(("build_tree3", bench(|| { let mut t = Tree3::<C3>::new(aabb, IL); for it in &its { t.insert(*it); } t.item_count() as u64 })));
    out.push(("build_octree3", bench(|| { let mut t = Octree3::<C3>::new(aabb, IL); for it in &its { t.insert(*it); } t.item_count() as u64 })));
    out.push(("build_morton3", bench(|| { let mut t = MortonGrid3::<C3>::new(aabb, levels); for it in &its { t.insert(*it); } t.item_count() as u64 })));
    // The two build-once structures: they have no incremental-insert story to defend, so
    // they are gated on the build the caller actually uses (bulk) plus their queries.
    out.push(("build_kdtree3", bench(|| KdTree3::from_items(IL, its.clone()).item_count() as u64)));
    out.push(("build_linear_octree3", bench(|| LinearOctree3::from_items(aabb, IL, 12, its.clone()).item_count() as u64)));

    let mut tree3 = Tree3::<C3>::new(aabb, IL); for it in &its { tree3.insert(*it); }
    let mut octree3 = Octree3::<C3>::new(aabb, IL); for it in &its { octree3.insert(*it); }
    let mut morton3 = MortonGrid3::<C3>::new(aabb, levels); for it in &its { morton3.insert(*it); }

    out.push(("cull_tree3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += tree3.cull(s).len() as u64; } n })));
    out.push(("cull_octree3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += octree3.cull(s).len() as u64; } n })));
    out.push(("cull_morton3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += morton3.cull(s).len() as u64; } n })));
    let kd3 = KdTree3::from_items(IL, its.clone());
    let lo3 = LinearOctree3::from_items(aabb, IL, 12, its.clone());
    out.push(("cull_kdtree3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += kd3.cull(s).len() as u64; } n })));
    out.push(("cull_linear_octree3_x64", bench(|| { let mut n = 0u64; for s in &qs { n += lo3.cull(s).len() as u64; } n })));

    let qp: Vec<Point3> = { let mut r = Rng::new(7); (0..256).map(|_| Point3::new(r.range(0.0, WORLD), r.range(0.0, WORLD), r.range(0.0, WORLD))).collect() };
    out.push(("knn_tree3_k16_x256", bench(|| { let mut n = 0u64; for q in &qp { n += tree3.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_kdtree3_k16_x256", bench(|| { let mut n = 0u64; for q in &qp { n += kd3.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_linear_octree3_k16_x256", bench(|| { let mut n = 0u64; for q in &qp { n += lo3.knn(*q, 16).len() as u64; } n })));
    // The grid's k-NN was NOT gated, which is how a 2-21x improvement in it (per-axis
    // expansion, 2026-07-29) passed through this file without leaving a mark. An op nobody
    // gates is an op nobody notices moving, in either direction.
    out.push(("knn_morton3_k16_x256", bench(|| { let mut n = 0u64; for q in &qp { n += morton3.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_octree3_k16_x256", bench(|| { let mut n = 0u64; for q in &qp { n += octree3.knn(*q, 16).len() as u64; } n })));

    // update: one frame of N relocations, predicate vs ItemRef.
    {
        let mut t = Tree3::<C3>::new(aabb, IL); for it in &its { t.insert(*it); }
        let mut pos: Vec<Point3> = its.iter().map(|i| i.p).collect();
        let mut vel = vels();
        out.push(("update_predicate_frame", bench(|| { for id in 0..N { let old = pos[id]; let np = step(&mut pos[id], &mut vel[id]); let cid = id as u32; t.update(old, |c| c.id == cid, |c| c.p = np); } 0 })));
    }
    {
        let mut t = Tree3::<C3>::new(aabb, IL);
        let mut refs = Vec::with_capacity(N); for it in &its { refs.push(t.insert_ref(*it).unwrap()); }
        let mut pos: Vec<Point3> = its.iter().map(|i| i.p).collect();
        let mut vel = vels();
        out.push(("update_ref_frame", bench(|| { for id in 0..N { let np = step(&mut pos[id], &mut vel[id]); t.update_ref(refs[id], |c| c.p = np); } 0 })));
    }

    // ---- 2D. The gate was 3D-only, so a 2D-side regression (the `Tree`/`QuadTree`
    // descent, or the linear quadtree) could not be caught at all.
    let rect = Rect::new(0.0, 0.0, WORLD, WORLD);
    let its2 = items2();
    let qs2 = queries2();
    out.push(("build_tree2", bench(|| { let mut t = Tree::<C2>::new(rect, IL); for it in &its2 { t.insert(*it); } t.item_count() as u64 })));
    out.push(("build_quadtree2", bench(|| { let mut t = QuadTree::<C2>::new(rect, IL); for it in &its2 { t.insert(*it); } t.item_count() as u64 })));
    out.push(("build_linear_quadtree", bench(|| LinearQuadTree::from_items(rect, IL, 14, its2.clone()).item_count() as u64)));

    let mut tree2 = Tree::<C2>::new(rect, IL); for it in &its2 { tree2.insert(*it); }
    let mut quad2 = QuadTree::<C2>::new(rect, IL); for it in &its2 { quad2.insert(*it); }
    let lq2 = LinearQuadTree::from_items(rect, IL, 14, its2.clone());
    out.push(("cull_tree2_x64", bench(|| { let mut n = 0u64; for s in &qs2 { n += tree2.cull(s).len() as u64; } n })));
    out.push(("cull_quadtree2_x64", bench(|| { let mut n = 0u64; for s in &qs2 { n += quad2.cull(s).len() as u64; } n })));
    out.push(("cull_linear_quadtree_x64", bench(|| { let mut n = 0u64; for s in &qs2 { n += lq2.cull(s).len() as u64; } n })));

    // 2D k-NN had NO coverage at all — not one structure. The 2D grid's k-NN changed the same
    // night as the 3D one and this file would have said nothing either way.
    let kd2 = KdTree2::from_items(IL, its2.clone());
    let mlevels = MortonGrid::<C2>::levels_for_cell_size(rect, VISION);
    let mut morton2 = MortonGrid::<C2>::new(rect, mlevels); for it in &its2 { morton2.insert(*it); }
    let qp2: Vec<Point> = { let mut r = Rng::new(11); (0..256).map(|_| Point::new(r.range(0.0, WORLD), r.range(0.0, WORLD))).collect() };
    out.push(("knn_tree2_k16_x256", bench(|| { let mut n = 0u64; for q in &qp2 { n += tree2.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_quadtree2_k16_x256", bench(|| { let mut n = 0u64; for q in &qp2 { n += quad2.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_kdtree2_k16_x256", bench(|| { let mut n = 0u64; for q in &qp2 { n += kd2.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_linear_quadtree_k16_x256", bench(|| { let mut n = 0u64; for q in &qp2 { n += lq2.knn(*q, 16).len() as u64; } n })));
    out.push(("knn_morton2_k16_x256", bench(|| { let mut n = 0u64; for q in &qp2 { n += morton2.knn(*q, 16).len() as u64; } n })));
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let save = argv.iter().any(|a| a == "--save");
    let threshold = argv.iter().position(|a| a == "--threshold").and_then(|i| argv.get(i + 1)).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.25);

    println!("regression gate | N={N} | item_limit={IL} | vision r={VISION} | {N_QUERY} culls | min of {REPS} reps");
    let calib = calibrate();
    let results = measure();

    if save {
        // **Save takes the minimum over several passes**, the same estimator the check uses when
        // confirming a suspected regression. It used to measure once, which is the one place a
        // single reading does the most damage: a baseline is not a datum, it is the reference
        // every future run is judged against, so a bad sample is wrong forever rather than for
        // one run.
        //
        // And a single reading here is a lottery. Ten identical runs of one op on this machine,
        // at 18-32 % background load, spanned **851–1 192 µs — a 1.40× spread**, and two
        // consecutive gate runs read an untouched op at ±3 % and then +74 %. The noise is not
        // proportional to average load; it is episodic, so "wait until the CPU looks free" does
        // not catch it either. A minimum converges on the machine's uncontended floor, which is
        // the quantity that is actually stable, and more passes only sharpen it.
        let passes: usize = argv.iter().position(|a| a == "--passes")
            .and_then(|i| argv.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(5);
        println!("\nbaselining over {passes} passes, keeping the minimum per op");
        let mut best: std::collections::HashMap<&str, f64> = results.iter().copied().collect();
        let mut best_calib = calib;
        for pass in 2..=passes {
            best_calib = best_calib.min(calibrate());
            for (name, ns) in measure() { best.entry(name).and_modify(|b| if ns < *b { *b = ns }).or_insert(ns); }
            println!("  pass {pass}/{passes} done");
        }
        let mut s = String::from("# vectorial-hash regression baseline — op<TAB>nanoseconds (min over passes, each itself min-of-N).\n");
        s.push_str("# _calib is a fixed CPU loop; the gate compares op/_calib ratios to cancel clock scaling.\n");
        s.push_str("# Regenerate with `--save [--passes N]`. More passes is strictly better: each one can only\n");
        s.push_str("# lower a number toward the machine's uncontended floor, never raise it.\n");
        // WHICH machine. Absolute nanoseconds are meaningless anywhere else, and `_calib`
        // normalisation does not rescue them: it cancels clock speed, not cache size or memory
        // bandwidth. Without this line the gate happily judged a laptop against a desktop.
        s.push_str(&vectorial_hash::machine_line());
        s.push_str(&format!("_calib\t{best_calib:.0}\n"));
        for (name, _) in &results { s.push_str(&format!("{name}\t{:.0}\n", best[name])); }
        let path = baseline_path(argv.iter().any(|a| a == "--local"));
        std::fs::write(&path, &s).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
        println!("\nsaved baseline -> {path}");
        println!("  {:<24} {:>12.0} ns (calibration yardstick)", "_calib", best_calib);
        for (name, first) in &results {
            let b = best[name];
            let gain = if b > 0.0 { first / b } else { 1.0 };
            println!("  {name:<24} {b:>12.0} ns  (pass 1 read {first:.0}, {gain:.2}× higher)");
        }
        return;
    }

    let path = baseline_path(false);
    let base = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => { eprintln!("no baseline at {path} — run with --save first."); std::process::exit(2); }
    };
    if path != BASELINE { println!("baseline: {path} (this machine's own, not the committed one)"); }
    let mut baseline = std::collections::HashMap::new();
    for line in base.lines() {
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let mut it = line.split('\t');
        if let (Some(k), Some(v)) = (it.next(), it.next()) { if let Ok(ns) = v.parse::<f64>() { baseline.insert(k.to_string(), ns); } }
    }

    // Normalise current op times by how much the CPU itself shifted since the
    // baseline run: scale > 1 ⟹ machine is slower now ⟹ shrink current numbers.
    // Whose numbers are these? A verdict is only available when the answer is "this machine".
    let prov = vectorial_hash::machine::verdict(&base);
    match &prov {
        vectorial_hash::Provenance::SameMachine => {}
        vectorial_hash::Provenance::OtherMachine(m) => {
            println!("!! this baseline was captured on {m}");
            println!("!! you are on {}", vectorial_hash::machine_id());
            println!("!! absolute times do not transfer between machines — `_calib` cancels clock");
            println!("!! speed, not cache size or memory bandwidth. The table below is ORIENTATION");
            println!("!! ONLY and no regression will be reported. Keep your own alongside it with");
            println!("!! `--save --local`, which never overwrites the committed one.");
            println!("!! or run on the machine above.\n");
        }
        vectorial_hash::Provenance::Unknown => {
            println!("!! this baseline predates machine fingerprinting, so its origin is unknown.");
            println!("!! Treating it as unverified: the table below is ORIENTATION ONLY. Re-save");
            println!("!! it with `--save --local` on the machine you intend to gate with.\n");
        }
    }
    let base_calib = baseline.get("_calib").copied().unwrap_or(calib);
    let scale = base_calib / calib;
    println!("calibration: baseline {base_calib:.0} ns vs current {calib:.0} ns → scale ×{scale:.3} (clock-normalised)\n");
    println!("{:<24} {:>12} {:>12} {:>9}  verdict", "op", "baseline ns", "norm. ns", "delta");
    let mut regressed = Vec::new();
    let mut missing = false;
    for (name, ns) in &results {
        let cur = ns * scale;
        match baseline.get(*name) {
            Some(&b) => {
                let delta = (cur - b) / b;
                let tag = if delta > threshold { regressed.push((*name, delta)); "REGRESSED" }
                    else if delta < -0.10 { "improved" } else { "ok" };
                println!("{name:<24} {b:>12.0} {cur:>12.0} {:>+8.1}%  {tag}", delta * 100.0);
            }
            None => { missing = true; println!("{name:<24} {:>12} {cur:>12.0}      new  (not in baseline)", "—"); }
        }
    }

    let pct = (threshold * 100.0) as i64;
    // A regression is a claim that THIS code got slower. Against a baseline from another machine
    // (or one with no fingerprint at all) the same table is a claim about the hardware, so the
    // run reports what it saw and stops there. Everything above still prints: orientation is
    // useful, a verdict would be a fabrication.
    if !prov.may_judge() {
        if !regressed.is_empty() {
            println!("
{} op(s) are over +{pct}% against that baseline — NOT reported as a", regressed.len());
            println!("regression, because the comparison is not between two runs of this machine:");
            for (n, d) in &regressed { println!("  {n}: {:+.1}%", d * 100.0); }
        }
        println!("
NO VERDICT: baseline provenance is {prov:?}. Re-run with --save here to gate.");
        return;
    }
    // ---------------------------------------------------------------- confirm before failing
    //
    // A single timed pass on a shared desktop is a coin flip, and this gate proved it on itself:
    // two consecutive runs of the SAME binary read `cull_tree3_x64` at -2.8% and then +54.0%, on
    // an op nobody had touched, with the machine at 76% CPU behind a chat client and an editor.
    // A gate that fails on one reading in those conditions is reporting the desktop rather than
    // the commit — and a gate that cries wolf gets ignored, which is worse than not having one.
    //
    // So a suspected regression is re-measured. Each op's estimator is min-of-N already; extra
    // passes simply widen N where it matters. A transient cannot be the minimum of every pass;
    // a real regression is present in all of them.
    if !regressed.is_empty() {
        const CONFIRM_PASSES: usize = 2;
        println!("\n{} op(s) over threshold on the first pass — confirming with {CONFIRM_PASSES} more.", regressed.len());
        println!("(a transient cannot be the minimum of every pass; a real regression is in all of them)");
        let mut best: std::collections::HashMap<&str, f64> = results.iter().map(|(n, ns)| (*n, ns * scale)).collect();
        for pass in 1..=CONFIRM_PASSES {
            let c = calibrate();
            let sc = base_calib / c;
            for (name, ns) in measure() {
                let v = ns * sc;
                best.entry(name).and_modify(|b| if v < *b { *b = v }).or_insert(v);
            }
            println!("  pass {pass} done");
        }
        let before = regressed.len();
        regressed.clear();
        for (name, _) in &results {
            if let (Some(&b), Some(&cur)) = (baseline.get(*name), best.get(*name)) {
                let delta = (cur - b) / b;
                if delta > threshold { regressed.push((*name, delta)); }
            }
        }
        if regressed.is_empty() {
            println!("  none survived — all {before} were transients. This machine is too noisy to");
            println!("  gate on a single pass; treat one-off readings here as unmeasured.");
            println!("\nPASS: no op regressed beyond +{pct}% (confirmed over {} passes).", CONFIRM_PASSES + 1);
            if missing { println!("NOTE: new ops not yet in the baseline — re-run with --save on an IDLE machine."); }
            return;
        }
        println!("\nFAIL: {} op(s) regressed beyond +{pct}% in EVERY pass:", regressed.len());
        for (n, d) in &regressed { println!("  {n}: {:+.1}% (best of {} passes)", d * 100.0, CONFIRM_PASSES + 1); }
        std::process::exit(1);
    }
    if missing { println!("\nNOTE: new ops not yet in the baseline — re-run with --save on an IDLE machine."); }
    println!("\nPASS: no op regressed beyond +{pct}%.");
}
