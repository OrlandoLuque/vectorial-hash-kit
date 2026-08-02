//! `bench-runner` — runs the kit's benchmarks back to back on a **quiet** machine and
//! writes the results out as tables.
//!
//! Every performance number in `docs/` used to be taken by hand, one bench at a time,
//! which makes it hostage to whatever else the box was doing — and that showed: a
//! headline ratio measured on a busy night read 3.06× where three quiet passes say
//! ~2.4×. This exists so the numbers are reproducible instead of anecdotal:
//!
//! - **it builds everything first**, so no compile time lands inside a measured run;
//! - **it waits for the CPU to actually be free** before every pass — measured with a
//!   fixed calibration loop rather than OS performance counters, so it needs no
//!   platform-specific API and directly answers the question that matters ("is someone
//!   stealing my cycles *right now*");
//! - **it repeats** (`--repeat N`) and reports min / median / max / spread per metric,
//!   because a number you have seen once is not a measurement;
//! - **it writes tables to a file**: a Markdown report plus a CSV of every metric, and
//!   the full raw output, under `bench-results/`.
//!
//! Benches opt into the metric tables by printing lines of the form
//! `#M <key> <value> [unit]` (anything else is passed through untouched), so adding a
//! metric is one `println!` and needs no changes here.
//!
//! ```bash
//! cargo run -p bench-runner --release -- --list
//! cargo run -p bench-runner --release -- --group kd --repeat 3
//! cargo run -p bench-runner --release -- --group all --repeat 2
//! ```

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------- the plan

#[derive(Clone, Copy)]
enum Target { Example(&'static str), Bin(&'static str) }
impl Target {
    fn name(&self) -> &'static str { match self { Target::Example(n) | Target::Bin(n) => n } }
    fn flag(&self) -> &'static str { match self { Target::Example(_) => "--example", Target::Bin(_) => "--bin" } }
}

/// What a bench's numbers actually measure. Mixing the two in one table is how you end
/// up comparing an algorithm against a frame that also had to draw 200 000 points.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Headless: the reported numbers are timed around the operation itself, no window,
    /// no GPU, no present. These are the ones to compare algorithms with.
    Algorithm,
    /// A real application loop, render included. Its per-operation metrics are still
    /// timed around the operation (so they are comparable), but anything frame-level
    /// (`fps`) includes drawing and is NOT an algorithmic measurement.
    EndToEnd,
}

#[derive(Clone)]
struct Bench {
    pkg: &'static str,
    target: Target,
    features: &'static str,
    env: Vec<(String, String)>,
    note: &'static str,
    args: &'static [&'static str],
    kind: Kind,
    /// Minutes, not seconds. Skipped unless `--include-slow`, so the default run stays
    /// something you will actually re-run before quoting a number.
    slow: bool,
}

/// Terser constructors — the plan below is a table, and should read like one.
fn b(pkg: &'static str, target: Target, features: &'static str, note: &'static str) -> Bench {
    Bench { pkg, target, features, env: Vec::new(), args: &[], note, kind: Kind::Algorithm, slow: false }
}
fn ba(pkg: &'static str, target: Target, features: &'static str, args: &'static [&'static str], note: &'static str) -> Bench {
    Bench { pkg, target, features, env: Vec::new(), args, note, kind: Kind::Algorithm, slow: false }
}
fn slow(mut x: Bench) -> Bench { x.slow = true; x }
fn e2e(mut x: Bench) -> Bench { x.kind = Kind::EndToEnd; x }
fn env(mut x: Bench, kv: &[(&str, &str)]) -> Bench {
    x.env.extend(kv.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    x
}

/// Expand a bench over a **grid** of environment values — the other dimensions a
/// comparison has besides "method A vs method B": population, distribution, radius,
/// thread count. One entry in, the cartesian product out, each labelled by its cell.
fn matrix(base: Bench, dims: &[(&str, &[&str])]) -> Vec<Bench> {
    let mut out = vec![base];
    for (key, values) in dims {
        let mut next = Vec::with_capacity(out.len() * values.len());
        for b in &out {
            for v in *values { next.push(env(b.clone(), &[(key, v)])); }
        }
        out = next;
    }
    out
}

fn plan(group: &str) -> Vec<Bench> {
    use Target::{Bin, Example};
    const VH: &str = "vectorial-hash";
    const DEMOS: &str = "vectorial-hash-demos";
    const CLI: &str = "vectorial-hash-cli";
    const PAR: &str = "parallel";

    // --- structure comparisons: which index for which data ----------------------
    let core = || vec![
        b(VH, Example("kdtree3_bench"), PAR, "median vs midpoint split, 3D (+ parallel build)"),
        b(VH, Example("linear_quadtree_bench"), PAR, "the 2D set: KdTree2 / QuadTree / LinearQuadTree / Morton"),
        b(VH, Example("linear_octree3_bench"), PAR, "LinearOctree3 vs Octree3 vs Morton"),
        b(VH, Example("decision2d"), "", "2D decision map (maintain vs cull, per structure)"),
        b(VH, Example("threshold_bench"), "", "brute-force crossover (advisor BRUTE_FORCE_MAX)"),
        b(VH, Example("churn_relocation_bench"), "", "relocation rate vs keep-index cost"),
        b(VH, Example("compact_bench"), "", "cache locality of compact()"),
        b(VH, Example("ropes_balance"), "neighbors", "ropes: what the neighbour lists cost to maintain"),
        b(VH, Example("frustum_check"), "", "correctness: frustum cull vs brute force"),
        b(VH, Example("stencil_vs_golden"), "", "correctness: SphereStencil vs an independently generated reference"),
    ];
    // --- the query verbs --------------------------------------------------------
    let query = || vec![
        b(VH, Example("raycast_compare"), "", "2D: capsule cull vs DDA leaf-walk"),
        b(VH, Example("raycast3_compare"), "", "3D: capsule cull vs DDA walk vs first-hit"),
        b(VH, Example("narrowphase_simd"), "", "narrowphase ceiling: scalar vs SoA/SIMD"),
        b(VH, Example("layered_cull_bench"), "", "layered Morton cull (in-memory)"),
        b(VH, Example("visibility_cull_bench"), "", "occlusion-aware visibility culling"),
        b(VH, Example("broadphase_tightness_bench"), "", "does a TIGHTER broadphase pay? (it does not)"),
        b(VH, Example("lbvh_bench"), "", "LBVH from Morton codes (CPU)"),
        b(VH, Example("wide_bvh_bench"), PAR, "wide 8-ary SoA/SIMD BVH node vs the shipping cull"),
        b(VH, Example("compressed_bvh_bench"), "", "quantised (u16) BVH nodes: exact, smaller"),
        b(VH, Example("key_compression_bench"), "", "sorted Morton keys: delta + varint shrink"),
        b(VH, Example("voxel_select_bench"), "", "block selection: naive vs bitmap vs spans vs chunk-skip"),
        b(VH, Example("work_counters"), "", "algorithmic work per query, no clock involved"),
        b(VH, Example("morton_knn_axis_bench"), "", "morton k-NN vs world aspect: what non-cubic cells cost"),
    ];
    // --- GPU --------------------------------------------------------------------
    let gpu = || vec![
        b(DEMOS, Example("gpu_sort_bench"), PAR, "GPU bitonic sort (the negative result)"),
        b(DEMOS, Example("gpu_radix_bench"), PAR, "GPU radix sort vs CPU (8-bit/4-pass)"),
        b(DEMOS, Example("gpu_onesweep_scan_bench"), PAR, "Onesweep in portable WGSL (measured un-implementable)"),
        b(DEMOS, Example("gpu_lbvh_build_bench"), PAR, "a whole LBVH built on the GPU"),
        b(DEMOS, Example("gpu_lbvh_query_bench"), PAR, "GPU LBVH range query + k-NN"),
        b(DEMOS, Example("gpu_spatial_bench"), PAR, "GPU spatial hash / grid"),
        b(DEMOS, Example("gpu_visibility_bench"), PAR, "GPU visibility: broadphase then exact"),
    ];
    // --- whole-simulation workloads ---------------------------------------------
    let sim = || vec![
        b(DEMOS, Example("siege_cpu_bench"), PAR, "siege: rebuild vs bulk vs keep, by thread count"),
        b(DEMOS, Example("bulk_load_bench"), PAR, "Tree3 bulk_load vs bulk_load_par"),
        b(DEMOS, Example("horde_bench"), PAR, "horde: dormant carpet + active front at scale"),
        b(VH, Example("parallel_ai"), PAR, "per-unit AI fan-out (the siege pattern)"),
        ba(DEMOS, Bin("critters_headless"), "", &["--mode", "both"], "2D dynamic workload: updates + culls + churn"),
        slow(ba(DEMOS, Bin("critters3d_headless"), PAR, &["--sweep"], "the 3D decision map sweep (long)")),
        slow(b(DEMOS, Example("horde_balance"), PAR, "horde balance sweep across seeds (long)")),
    ];
    // --- the three application demos, swept over the structures they compare -----
    //
    // **Headless since 2026-08-03.** These entries exist to compare INDEXES, and they used to
    // run the demos with a window and a GPU, so every number also contained a renderer. Two
    // consequences, both real: the group could not run on a machine without a display (i.e. any
    // CI runner), and the measurement moved with what the renderer was doing — the stealth
    // crossover reads ~1 350 agents headless against ~1 100 with drawing, because a linear scan
    // gains more from an idle machine than a pointer-chasing index does.
    //
    // So these are sim-only now, and the `fps` figures they emit are simulation rates rather
    // than frame rates. That is a deliberate change in what is measured, not a tidy-up: if you
    // want to know whether a demo *renders* fast enough, run it and look at it.
    let demos = || {
        let fluid = |idx: &str, note: &'static str| e2e(env(b(DEMOS, Bin("fluid_wgpu"), "", note), &[("FLUID_HEADLESS", "420"), ("FLUID_INDEX", idx)]));
        let cloud = |note: &'static str| e2e(env(b(DEMOS, Bin("pointcloud_wgpu"), "", note), &[("CLOUD_HEADLESS", "1"), ("CLOUD_N", "120000")]));
        let sneak = |n: &str, note: &'static str| e2e(env(b(DEMOS, Bin("stealth_wgpu"), "", note), &[("STEALTH_HEADLESS", "600"), ("STEALTH_CIVS", n)]));
        vec![
            fluid("morton", "fluid neighbours: MortonGrid rebuild"),
            fluid("mortonkeep", "fluid neighbours: MortonGrid kept in place"),
            fluid("keep", "fluid neighbours: kept Tree + ItemRef"),
            fluid("linear", "fluid neighbours: LinearQuadTree rebuild"),
            fluid("adaptive", "fluid neighbours: AdaptiveIndex2 picks its own"),
            // one run covers all three structures and cross-checks their k-NN distances
            cloud("point cloud k-NN: KdTree3 vs Octree3 vs MortonGrid3"),
            sneak("40", "stealth crossover: 40 agents"),
            sneak("160", "stealth crossover: 160 agents"),
            sneak("640", "stealth crossover: 640 agents"),
            sneak("1400", "stealth crossover: 1400 agents (near the crossover)"),
            sneak("2560", "stealth crossover: 2560 agents"),
            sneak("10240", "stealth crossover: 10240 agents"),
            sneak("40000", "stealth crossover: 40000 agents"),
        ]
    };
    // --- sweet spots: the tuning knobs, over the OTHER dimensions too --------------
    // A comparison is not just method A vs method B: the answer moves with population,
    // distribution and query radius, so the knob sweep is run across that grid.
    let sweeps = || {
        let mut v = matrix(b(VH, Example("knob_sweep"), "", "leaf-size / depth / levels optimum"), &[
            ("KS_N", &["50000", "200000"]),
            ("KS_DIST", &["clustered", "uniform"]),
            ("KS_R", &["8", "30"]),
        ]);
        v.push(b(VH, Example("threshold_bench"), "", "index-vs-brute crossover by N"));
        v.push(b(VH, Example("churn_relocation_bench"), "", "keep-index cost by relocation rate"));
        v.push(b(DEMOS, Example("siege_cpu_bench"), PAR, "siege: keep vs rebuild by THREAD COUNT"));
        v.push(b(DEMOS, Example("horde_bench"), PAR, "horde: cost by POPULATION and awake fraction"));
        v
    };
    // --- the template/CLI benchmarks behind the paper's early sections ------------
    let cli = || vec![
        ba(CLI, Bin("vh"), "", &["bench"], "single fixed template, tree vs quadtree"),
        slow(ba(CLI, Bin("vh"), "", &["bench-sizes"], "per-cell-size template selection (long)")),
        ba(CLI, Bin("vh"), "", &["bench-walk"], "tree descent vs neighbour-walk flood fill"),
        ba(CLI, Bin("vh"), "", &["bench-fallback"], "granularity-as-fallback aggregation"),
        ba(CLI, Bin("vh"), "", &["bench-scale"], "figure/grid scale equivalence"),
    ];
    // --- the on-disk cold store (these WRITE FILES under the target dir) ----------
    let cold = || vec![
        b(VH, Example("cold_index_bench"), "", "cold index: structure options"),
        b(VH, Example("cold_layered_bench"), "", "layered cold store on sparse data"),
        b(VH, Example("cold_store_redb"), "", "on-disk B-tree (redb) prototype"),
        slow(b(VH, Example("cold_store_engines"), "", "B-tree (redb) vs LSM (fjall) (long)")),
    ];
    let gate = || vec![
        b(VH, Example("regression_gate"), "", "committed-baseline regression gate"),
    ];
    // Just the two whose numbers the docs quote for the median split.
    let kd = || vec![
        b(VH, Example("kdtree3_bench"), PAR, "median vs midpoint, 3D"),
        b(VH, Example("linear_quadtree_bench"), PAR, "the 2D set incl. KdTree2"),
    ];

    match group {
        "core" => core(), "query" => query(), "gpu" => gpu(), "sim" => sim(),
        "demos" => demos(), "cli" => cli(), "cold" => cold(), "gate" => gate(), "kd" => kd(),
        "sweeps" => sweeps(),
        "all" => {
            let mut v = core();
            for g in [query(), gpu(), sim(), demos(), sweeps(), cli(), cold(), gate()] { v.extend(g); }
            v
        }
        _ => Vec::new(),
    }
}

/// A ratio whose peak-to-peak spread exceeds this is not a number you can quote as if it
/// were exact. 15% is where this repo's own history puts it: the clustered cull ratio read
/// 1.57-3.28 across runs before the comparison was paired, and every figure that survived
/// re-measurement sits well inside this band.
const RATIO_SPREAD_WARN: f64 = 15.0;

const GROUPS: &[&str] = &["core", "query", "gpu", "sim", "demos", "sweeps", "cli", "cold", "gate", "kd", "all"];

// ------------------------------------------------------- is the machine free?

/// A fixed, allocation-free CPU workload. Its *own* runtime is the load signal: if the
/// same loop suddenly takes 30% longer, something else is using the machine. This is
/// deliberately not an OS performance counter — those are platform-specific and (on a
/// localised Windows) even the counter NAMES change, while this measures the thing we
/// actually care about: how much CPU this process can get right now.
fn calibrate() -> Duration {
    let t = Instant::now();
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for i in 0..3_000_000u64 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(i | 1);
        x ^= x >> 29;
    }
    std::hint::black_box(x);
    t.elapsed()
}
fn calib_best(n: usize) -> Duration { (0..n).map(|_| calibrate()).min().unwrap() }

/// Block until the calibration loop runs within `tolerance` of its best-known time.
/// Returns the observed ratio and whether it gave up.
fn wait_for_idle(base: Duration, tolerance: f64, max_wait: Duration, quiet: bool) -> (f64, bool) {
    let start = Instant::now();
    loop {
        let now = calib_best(3);
        let ratio = now.as_secs_f64() / base.as_secs_f64();
        if ratio <= tolerance { return (ratio, false); }
        if start.elapsed() >= max_wait { return (ratio, true); }
        if !quiet { println!("    machine busy: calibration loop {ratio:.2}x its best — waiting…"); }
        std::thread::sleep(Duration::from_secs(5));
    }
}


// ------------------------------------------------------------- processor time

/// Per-process **CPU time** (user + kernel), which is what you actually want to compare:
/// wall-clock time says how long the bench took to finish, CPU time says how much
/// processing it took. If another process steals the machine, wall time inflates and CPU
/// time does not — so `cpu/wall` is also a free honesty check on every measurement.
///
/// No crates: the two OS calls are declared here directly (kernel32 on Windows, libc's
/// getrusage elsewhere). This is why the runner executes the built artifact instead of
/// `cargo run` — CPU time is per process and is NOT inherited from a child, so measuring
/// `cargo` would measure the wrong process.
#[cfg(windows)]
mod cpu {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct FileTime { low: u32, high: u32 }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetProcessTimes(h: std::os::windows::raw::HANDLE, creation: *mut FileTime, exit: *mut FileTime,
                           kernel: *mut FileTime, user: *mut FileTime) -> i32;
    }

    /// Kernel + user seconds the child burned. Valid after it exits, as long as the
    /// handle is still open (i.e. before the `Child` is dropped).
    pub fn child_cpu_secs(child: &Child) -> Option<f64> {
        let (mut c, mut e, mut k, mut u) = (FileTime::default(), FileTime::default(), FileTime::default(), FileTime::default());
        let ok = unsafe { GetProcessTimes(child.as_raw_handle(), &mut c, &mut e, &mut k, &mut u) };
        if ok == 0 { return None; }
        // FILETIME counts 100-nanosecond ticks.
        let secs = |f: FileTime| (((f.high as u64) << 32) | f.low as u64) as f64 * 1e-7;
        Some(secs(k) + secs(u))
    }
    pub fn children_cpu_secs() -> Option<f64> { None } // unix path only
}

#[cfg(unix)]
mod cpu {
    use std::process::Child;

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct TimeVal { sec: i64, usec: i64 }
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct RUsage { utime: TimeVal, stime: TimeVal, rest: [i64; 14] }

    unsafe extern "C" { fn getrusage(who: i32, usage: *mut RUsage) -> i32; }
    const RUSAGE_CHILDREN: i32 = -1;

    /// Cumulative CPU seconds of all reaped children; the runner takes the delta around a
    /// run (it never runs two benches at once, so the delta is that bench).
    pub fn children_cpu_secs() -> Option<f64> {
        let mut u = RUsage::default();
        if unsafe { getrusage(RUSAGE_CHILDREN, &mut u) } != 0 { return None; }
        Some(u.utime.sec as f64 + u.utime.usec as f64 * 1e-6 + u.stime.sec as f64 + u.stime.usec as f64 * 1e-6)
    }
    pub fn child_cpu_secs(_child: &Child) -> Option<f64> { None } // windows path only
}

// ----------------------------------------------------------------- metrics

#[derive(Default)]
struct Series { unit: String, values: Vec<f64> }
impl Series {
    fn min(&self) -> f64 { self.values.iter().cloned().fold(f64::INFINITY, f64::min) }
    fn max(&self) -> f64 { self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    fn median(&self) -> f64 {
        let mut v = self.values.clone();
        v.sort_by(f64::total_cmp);
        if v.is_empty() { return f64::NAN; }
        if v.len() % 2 == 1 { v[v.len() / 2] } else { 0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2]) }
    }
    /// Peak-to-peak as a fraction of the median — how much a single reading could lie.
    fn spread_pct(&self) -> f64 { let m = self.median(); if m == 0.0 { 0.0 } else { (self.max() - self.min()) / m * 100.0 } }
}

/// `#M <key> <value> [unit]` — the opt-in machine-readable line a bench prints.
fn parse_metric(line: &str) -> Option<(String, f64, String)> {
    let rest = line.trim().strip_prefix("#M ")?;
    let mut it = rest.split_whitespace();
    let key = it.next()?.to_string();
    let value: f64 = it.next()?.parse().ok()?;
    Some((key, value, it.next().unwrap_or("").to_string()))
}

// -------------------------------------------------------------------- run

struct Pass { bench: String, pass: usize, ok: bool, seconds: f64, cpu: Option<f64>, busy: bool, load: f64 }

/// `stealth_wgpu [STEALTH_CIVS=640]` / `vh [bench-walk]` — the same binary run several
/// ways needs the variant in its name, in the table and in the metric keys alike.
fn label(b: &Bench) -> String {
    let mut bits: Vec<String> = Vec::new();
    for (k, v) in &b.env { if !k.ends_with("MAX_FRAMES") { bits.push(format!("{k}={v}")); } }
    for a in b.args { bits.push(a.trim_start_matches("--").to_string()); }
    if bits.is_empty() { b.target.name().to_string() } else { format!("{} [{}]", b.target.name(), bits.join(" ")) }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| argv.iter().any(|a| a == name);
    let value = |name: &str| argv.iter().position(|a| a == name).and_then(|i| argv.get(i + 1)).cloned();

    if flag("--help") || flag("-h") {
        println!("bench-runner — run the kit's benchmarks on a quiet machine and write result tables\n");
        println!("  --group <{}>   which set to run (default: core)", GROUPS.join("|"));
        println!("  --repeat <n>              passes per bench (default 1; use 3+ for anything you will quote)");
        println!("  --list                    print the plan and exit");
        println!("  --include-slow            also run the benches marked slow (minutes each)");
        println!("  --only <substring>        keep only benches whose name or note matches");
        println!("  --strict                  exit non-zero if a quoted ratio is unstable (>{RATIO_SPREAD_WARN:.0}% spread)");
        println!("  --no-idle-wait            do not wait for the CPU to be free");
        println!("  --tolerance <f>           'idle' means the calibration loop is within this factor (default 1.15)");
        println!("  --out <dir>               where the report goes (default: bench-results)");
        println!("  --save                    pass --save through to the regression gate (re-baseline)");
        return;
    }

    let group = value("--group").unwrap_or_else(|| "core".into());
    let mut benches = plan(&group);
    if let Some(pat) = value("--only") { benches.retain(|b| b.target.name().contains(&pat) || b.note.contains(&pat)); }
    let skipped_slow = if flag("--include-slow") { 0 } else {
        let before = benches.len();
        benches.retain(|b| !b.slow);
        before - benches.len()
    };
    if benches.is_empty() { eprintln!("unknown group '{group}' — try one of: {}", GROUPS.join(", ")); std::process::exit(2); }
    let repeat: usize = value("--repeat").and_then(|s| s.parse().ok()).unwrap_or(1);
    let tolerance: f64 = value("--tolerance").and_then(|s| s.parse().ok()).unwrap_or(1.15);
    let out_dir = value("--out").unwrap_or_else(|| "bench-results".into());
    let idle_wait = !flag("--no-idle-wait");

    if flag("--list") {
        println!("group '{group}' — {} benches x {repeat} pass(es){}:", benches.len(),
            if skipped_slow > 0 { format!(", {skipped_slow} slow one(s) hidden") } else { String::new() });
        for b in &benches {
            println!("  {:<22} {}{}", b.target.name(), b.note, if b.slow { "  [slow]" } else { "" });
            let feat = if b.features.is_empty() { String::new() } else { format!(" --features {}", b.features) };
            let env: String = b.env.iter().map(|(k, v)| format!(" {k}={v}")).collect();
            println!("      cargo run -p {} {} {} --release{}{}", b.pkg, b.target.flag(), b.target.name(), feat, if env.is_empty() { String::new() } else { format!("   env:{env}") });
        }
        return;
    }

    // ---- environment header (a table of numbers is worthless without it)
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let rustc = capture("rustc", &["--version"]).unwrap_or_else(|| "?".into());
    let commit = capture("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "?".into());
    let subject = capture("git", &["log", "-1", "--format=%s"]).unwrap_or_default();
    let dirty = capture("git", &["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.trim().is_empty()).unwrap_or(false);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);

    println!("bench-runner — group '{group}', {} bench(es) x {repeat} pass(es)", benches.len());
    if skipped_slow > 0 { println!("  ({skipped_slow} slow bench(es) skipped — pass --include-slow to run them)"); }
    println!("  rustc {rustc} | {threads} logical CPUs | commit {commit}{}", if dirty { " (WORKTREE DIRTY)" } else { "" });

    // ---- build first, so no compile time lands inside a measured run
    println!("\nbuilding (so no compile time lands inside a measured run)…");
    let mut feature_sets: Vec<&str> = benches.iter().map(|b| b.features).collect();
    feature_sets.sort_unstable(); feature_sets.dedup();
    for f in feature_sets {
        let mut pkgs: Vec<&str> = benches.iter().filter(|b| b.features == f).map(|b| b.pkg).collect();
        pkgs.sort_unstable(); pkgs.dedup();
        for pkg in pkgs {
            let mut args: Vec<String> = vec!["build".into(), "-q".into(), "-p".into(), pkg.into(), "--release".into()];
            for b in benches.iter().filter(|b| b.features == f && b.pkg == pkg) {
                args.push(b.target.flag().into());
                args.push(b.target.name().into());
            }
            if !f.is_empty() { args.push("--features".into()); args.push(f.into()); }
            println!("  cargo {}", args.join(" "));
            let st = Command::new("cargo").args(&args).status().expect("cargo build");
            if !st.success() { eprintln!("build failed for {pkg} ({f})"); std::process::exit(1); }
        }
    }

    // ---- the calibration reference, taken once while nothing of ours is running
    let base = calib_best(7);
    println!("\ncalibration loop baseline: {:.1} ms (idle = within {tolerance:.2}x)\n", base.as_secs_f64() * 1e3);

    let mut passes: Vec<Pass> = Vec::new();
    let mut metrics: BTreeMap<String, Series> = BTreeMap::new();
    let mut raw = String::new();

    for (i, b) in benches.iter().enumerate() {
        for pass in 1..=repeat {
            let name = b.target.name();
            let label = label(b);
            println!("{}", "=".repeat(74));
            println!("[{}/{}] {label}  pass {pass}/{repeat} — {}", i + 1, benches.len(), b.note);
            println!("{}", "=".repeat(74));
            let (load, gave_up) = if idle_wait { wait_for_idle(base, tolerance, Duration::from_secs(300), false) } else { (1.0, false) };
            if gave_up { println!("    STILL BUSY ({load:.2}x) — running anyway; this pass is marked noisy"); }
            else if idle_wait { println!("    machine free ({load:.2}x baseline)"); }

            // Execute the built artifact directly: cargo's own startup would land in the
            // wall time, and CPU time is per process (a parent does not accumulate its
            // child's), so timing `cargo` would time the wrong process entirely.
            let exe = artifact_path(b);
            let mut args: Vec<String> = b.args.iter().map(|s| s.to_string()).collect();
            if flag("--save") && name == "regression_gate" { args.push("--save".into()); }

            let cpu_before = cpu::children_cpu_secs();
            let t = Instant::now();
            let mut child = Command::new(&exe).args(&args).envs(b.env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
                .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()
                .unwrap_or_else(|e| panic!("cannot run {}: {e}", exe.display()));
            let stdout = child.stdout.take().expect("stdout");
            raw.push_str(&format!("\n### {label} pass {pass}\n\n```\n"));
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some((key, v, unit)) = parse_metric(&line) {
                    let e = metrics.entry(format!("{name}.{key}")).or_default();
                    if e.unit.is_empty() { e.unit = unit; }
                    e.values.push(v);
                }
                println!("  {line}");
                raw.push_str(&line); raw.push('\n');
            }
            let st = child.wait().expect("wait");
            let secs = t.elapsed().as_secs_f64();
            // Windows: ask the (exited but still open) handle. Unix: delta of the reaped
            // children's rusage. Either way this is CPU burned, not time elapsed.
            let cpu_secs = cpu::child_cpu_secs(&child)
                .or_else(|| Some(cpu::children_cpu_secs()? - cpu_before?));
            raw.push_str("```\n");
            match cpu_secs {
                Some(c) => println!("  -> {} in {secs:.1}s wall / {c:.1}s cpu ({:.2}x)", if st.success() { "ok" } else { "FAILED" }, c / secs.max(1e-9)),
                None => println!("  -> {} in {secs:.1}s", if st.success() { "ok" } else { "FAILED" }),
            }
            passes.push(Pass { bench: label, pass, ok: st.success(), seconds: secs, cpu: cpu_secs, busy: gave_up, load });
        }
    }

    // ---- write the tables out
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let stem = format!("{out_dir}/{stamp}-{group}");
    let mut md = String::new();
    md.push_str(&format!("# Benchmark run — group `{group}`\n\n"));
    md.push_str("| | |\n| --- | --- |\n");
    md.push_str(&format!("| unix time | {stamp} |\n| rustc | {rustc} |\n| logical CPUs | {threads} |\n"));
    md.push_str(&format!("| commit | `{commit}` {subject} |\n| worktree | {} |\n", if dirty { "**dirty** — numbers may not match the commit" } else { "clean" }));
    md.push_str(&format!("| slow benches | {} |\n", if skipped_slow > 0 { format!("{skipped_slow} skipped (--include-slow to run)") } else { "none skipped".into() }));
    md.push_str(&format!("| passes per bench | {repeat} |\n| idle gate | {} |\n\n", if idle_wait { format!("calibration loop within {tolerance:.2}x of {:.1} ms", base.as_secs_f64() * 1e3) } else { "disabled".into() }));

    md.push_str("## Runs\n\n**cpu s** is processor time (user + kernel) burned by the bench process, which is what to\ncompare: wall time says how long it took, CPU time says how much processing it took, and\nonly wall time inflates when something else is using the machine. `cpu/wall` above 1 means\nthe bench used several cores; well below 1 on a single-threaded bench means it was starved\nor blocked.\n\n| bench | pass | status | wall s | cpu s | cpu/wall | machine |\n| --- | ---: | --- | ---: | ---: | ---: | --- |\n");
    for p in &passes {
        let (cpu, ratio) = match p.cpu { Some(c) => (format!("{c:.2}"), format!("{:.2}x", c / p.seconds.max(1e-9))), None => ("—".into(), "—".into()) };
        md.push_str(&format!("| `{}` | {} | {} | {:.1} | {} | {} | {} |\n", p.bench, p.pass, if p.ok { "ok" } else { "**FAILED**" }, p.seconds, cpu, ratio,
            if p.busy { format!("**busy {:.2}x**", p.load) } else { format!("{:.2}x", p.load) }));
    }

    if metrics.is_empty() {
        md.push_str("\n## Metrics\n\nNone: no bench in this group prints `#M <key> <value> [unit]` lines yet.\n");
    } else {
        md.push_str("\n## Metrics\n\nAcross the repeated passes. **Spread** is peak-to-peak over the median — how far a\nsingle reading could have been from the truth.\n\n");
        if benches.iter().any(|b| b.kind == Kind::EndToEnd) {
            md.push_str("> This run included **end-to-end** benches (a real application loop). Their\n> per-operation metrics are timed around the operation and are comparable, but any\n> `*.fps` row includes **drawing** and is not an algorithmic measurement — compare\n> algorithms with the headless benches.\n\n");
        }
        md.push_str("| metric | unit | n | min | median | max | spread |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n");
        for (k, s) in &metrics {
            let flag = if unstable_ratio(k, s) { "  (inestable)" } else { "" };
            md.push_str(&format!("| `{k}` | {} | {} | {:.3} | **{:.3}** | {:.3} | {:.1}%{flag} |
", s.unit, s.values.len(), s.min(), s.median(), s.max(), s.spread_pct()));
        }
        let shaky: Vec<&String> = metrics.iter().filter(|(k, s)| unstable_ratio(k, s)).map(|(k, _)| k).collect();
        if !shaky.is_empty() {
            md.push_str(&format!("
> **{} ratio(s) moved more than {RATIO_SPREAD_WARN:.0}% between passes.** Quote them as
> a range or not at all — and consider measuring the pair *interleaved* (`common::compare2`),
> which is what turned this repo's least stable figure from 1.57-3.28 into 2.02-2.28:
", shaky.len()));
            for k in &shaky { md.push_str(&format!("> - `{k}`
")); }
        }
        let mut csv = String::from("metric,unit,n,min,median,max,spread_pct\n");
        for (k, s) in &metrics {
            csv.push_str(&format!("{k},{},{},{:.6},{:.6},{:.6},{:.3}\n", s.unit, s.values.len(), s.min(), s.median(), s.max(), s.spread_pct()));
        }
        write_file(&format!("{stem}-metrics.csv"), &csv);
    }

    md.push_str("\n## Raw output\n");
    md.push_str(&raw);
    write_file(&format!("{stem}.md"), &md);

    let shaky: Vec<String> = metrics.iter().filter(|(k, s)| unstable_ratio(k, s)).map(|(k, _)| k.clone()).collect();
    if !shaky.is_empty() {
        println!("
{} ratio(s) moved more than {RATIO_SPREAD_WARN:.0}% between passes — quote as a range, or", shaky.len());
        println!("measure the pair INTERLEAVED (common::compare2) instead of each side separately:");
        for k in &shaky { println!("  {k}"); }
    }
    let failed = passes.iter().filter(|p| !p.ok).count();
    let noisy = passes.iter().filter(|p| p.busy).count();
    println!("\n{}", "=".repeat(74));
    println!("{} pass(es), {failed} failed, {noisy} run on a busy machine", passes.len());
    println!("report : {stem}.md");
    if !metrics.is_empty() { println!("metrics: {stem}-metrics.csv"); }
    if failed > 0 { std::process::exit(1); }
    // --strict makes the methodology enforceable rather than remembered.
    if flag("--strict") && !shaky.is_empty() {
        eprintln!("--strict: {} unstable ratio(s); refusing to report them as measurements", shaky.len());
        std::process::exit(3);
    }
}

/// Where cargo put the thing we just built. Examples land in `target/release/examples/`,
/// binaries directly in `target/release/`.
fn artifact_path(b: &Bench) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from("target/release");
    if matches!(b.target, Target::Example(_)) { p.push("examples"); }
    p.push(format!("{}{}", b.target.name(), std::env::consts::EXE_SUFFIX));
    p
}

/// A metric is a ratio if its unit or name says so, and unstable if it moved too much
/// between passes to be quoted. Ratios get the gate because ratios are what ends up in
/// documentation.
fn unstable_ratio(key: &str, s: &Series) -> bool {
    // Judge by UNIT, not by name: a metric called `cull_ratio_paired_spread` is reported in
    // percent and is the diagnostic ABOUT a ratio, not a ratio — flagging it for being
    // variable is circular.
    let is_ratio = s.unit == "x" && !key.ends_with("_spread");
    is_ratio && s.values.len() > 1 && s.spread_pct() > RATIO_SPREAD_WARN
}

fn capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}
fn write_file(path: &str, body: &str) {
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
    f.write_all(body.as_bytes()).expect("write");
}
