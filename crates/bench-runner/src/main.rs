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

enum Target { Example(&'static str), Bin(&'static str) }
impl Target {
    fn name(&self) -> &'static str { match self { Target::Example(n) | Target::Bin(n) => n } }
    fn flag(&self) -> &'static str { match self { Target::Example(_) => "--example", Target::Bin(_) => "--bin" } }
}

struct Bench {
    pkg: &'static str,
    target: Target,
    features: &'static str,
    env: &'static [(&'static str, &'static str)],
    note: &'static str,
    args: &'static [&'static str],
    /// Minutes, not seconds. Skipped unless `--include-slow`, so the default run stays
    /// something you will actually re-run before quoting a number.
    slow: bool,
}

/// Terser constructors — the plan below is a table, and should read like one.
const fn b(pkg: &'static str, target: Target, features: &'static str, note: &'static str) -> Bench {
    Bench { pkg, target, features, env: &[], args: &[], note, slow: false }
}
const fn ba(pkg: &'static str, target: Target, features: &'static str, args: &'static [&'static str], note: &'static str) -> Bench {
    Bench { pkg, target, features, env: &[], args, note, slow: false }
}
const fn slow(mut x: Bench) -> Bench { x.slow = true; x }

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
    let demos = || vec![
        Bench { env: &[("FLUID_MAX_FRAMES", "420"), ("FLUID_INDEX", "morton")], ..b(DEMOS, Bin("fluid_wgpu"), "", "fluid neighbours: MortonGrid rebuild") },
        Bench { env: &[("FLUID_MAX_FRAMES", "420"), ("FLUID_INDEX", "keep")], ..b(DEMOS, Bin("fluid_wgpu"), "", "fluid neighbours: kept Tree + ItemRef") },
        Bench { env: &[("FLUID_MAX_FRAMES", "420"), ("FLUID_INDEX", "linear")], ..b(DEMOS, Bin("fluid_wgpu"), "", "fluid neighbours: LinearQuadTree rebuild") },
        Bench { env: &[("CLOUD_MAX_FRAMES", "240"), ("CLOUD_INDEX", "kd")], ..b(DEMOS, Bin("pointcloud_wgpu"), "", "point cloud k-NN: KdTree3") },
        Bench { env: &[("CLOUD_MAX_FRAMES", "240"), ("CLOUD_INDEX", "octree")], ..b(DEMOS, Bin("pointcloud_wgpu"), "", "point cloud k-NN: Octree3") },
        Bench { env: &[("CLOUD_MAX_FRAMES", "240"), ("CLOUD_INDEX", "morton")], ..b(DEMOS, Bin("pointcloud_wgpu"), "", "point cloud k-NN: MortonGrid3") },
        Bench { env: &[("STEALTH_MAX_FRAMES", "600"), ("STEALTH_CIVS", "40")], ..b(DEMOS, Bin("stealth_wgpu"), "", "stealth crossover: 40 agents") },
        Bench { env: &[("STEALTH_MAX_FRAMES", "600"), ("STEALTH_CIVS", "160")], ..b(DEMOS, Bin("stealth_wgpu"), "", "stealth crossover: 160 agents") },
        Bench { env: &[("STEALTH_MAX_FRAMES", "600"), ("STEALTH_CIVS", "640")], ..b(DEMOS, Bin("stealth_wgpu"), "", "stealth crossover: 640 agents") },
        Bench { env: &[("STEALTH_MAX_FRAMES", "600"), ("STEALTH_CIVS", "2560")], ..b(DEMOS, Bin("stealth_wgpu"), "", "stealth crossover: 2560 agents") },
        Bench { env: &[("STEALTH_MAX_FRAMES", "600"), ("STEALTH_CIVS", "10240")], ..b(DEMOS, Bin("stealth_wgpu"), "", "stealth crossover: 10240 agents") },
        Bench { env: &[("STEALTH_MAX_FRAMES", "600"), ("STEALTH_CIVS", "40000")], ..b(DEMOS, Bin("stealth_wgpu"), "", "stealth crossover: 40000 agents") },
    ];
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
        "all" => {
            let mut v = core();
            for g in [query(), gpu(), sim(), demos(), cli(), cold(), gate()] { v.extend(g); }
            v
        }
        _ => Vec::new(),
    }
}

const GROUPS: &[&str] = &["core", "query", "gpu", "sim", "demos", "cli", "cold", "gate", "kd", "all"];

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

struct Pass { bench: String, pass: usize, ok: bool, seconds: f64, busy: bool, load: f64 }

/// `stealth_wgpu [STEALTH_CIVS=640]` / `vh [bench-walk]` — the same binary run several
/// ways needs the variant in its name, in the table and in the metric keys alike.
fn label(b: &Bench) -> String {
    let mut bits: Vec<String> = Vec::new();
    for (k, v) in b.env { if !k.ends_with("MAX_FRAMES") { bits.push(format!("{k}={v}")); } }
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

            let mut args: Vec<String> = vec!["run".into(), "-q".into(), "-p".into(), b.pkg.into(), b.target.flag().into(), name.into(), "--release".into()];
            if !b.features.is_empty() { args.push("--features".into()); args.push(b.features.into()); }
            let extra: Vec<&str> = b.args.iter().copied().chain(if flag("--save") && name == "regression_gate" { Some("--save") } else { None }).collect();
            if !extra.is_empty() { args.push("--".into()); args.extend(extra.iter().map(|s| s.to_string())); }

            let t = Instant::now();
            let mut child = Command::new("cargo").args(&args).envs(b.env.iter().copied())
                .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("spawn cargo");
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
            raw.push_str("```\n");
            let secs = t.elapsed().as_secs_f64();
            println!("  -> {} in {secs:.1}s", if st.success() { "ok" } else { "FAILED" });
            passes.push(Pass { bench: label, pass, ok: st.success(), seconds: secs, busy: gave_up, load });
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

    md.push_str("## Runs\n\n| bench | pass | status | seconds | machine |\n| --- | ---: | --- | ---: | --- |\n");
    for p in &passes {
        md.push_str(&format!("| `{}` | {} | {} | {:.1} | {} |\n", p.bench, p.pass, if p.ok { "ok" } else { "**FAILED**" }, p.seconds,
            if p.busy { format!("**busy {:.2}x**", p.load) } else { format!("{:.2}x", p.load) }));
    }

    if metrics.is_empty() {
        md.push_str("\n## Metrics\n\nNone: no bench in this group prints `#M <key> <value> [unit]` lines yet.\n");
    } else {
        md.push_str("\n## Metrics\n\nAcross the repeated passes. **Spread** is peak-to-peak over the median — how far a\nsingle reading could have been from the truth.\n\n");
        md.push_str("| metric | unit | n | min | median | max | spread |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n");
        for (k, s) in &metrics {
            md.push_str(&format!("| `{k}` | {} | {} | {:.3} | **{:.3}** | {:.3} | {:.1}% |\n", s.unit, s.values.len(), s.min(), s.median(), s.max(), s.spread_pct()));
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

    let failed = passes.iter().filter(|p| !p.ok).count();
    let noisy = passes.iter().filter(|p| p.busy).count();
    println!("\n{}", "=".repeat(74));
    println!("{} pass(es), {failed} failed, {noisy} run on a busy machine", passes.len());
    println!("report : {stem}.md");
    if !metrics.is_empty() { println!("metrics: {stem}-metrics.csv"); }
    if failed > 0 { std::process::exit(1); }
}

fn capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}
fn write_file(path: &str, body: &str) {
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
    f.write_all(body.as_bytes()).expect("write");
}
