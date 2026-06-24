//! Headless critters: run the exact same simulation as the visual demo at
//! full CPU speed (no window, no vsync) and report per-operation statistics.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin critters_headless --release -- \
//!     --mode both --frames 600 --drifters 400 --hunters 400 --pulsars 400 \
//!     --split 3 --merge 3 --dt 0.0167 --seed 42 [--csv out.csv]
//! ```
//!
//! Reported per structure: mean / p50 / p95 of the per-frame totals for
//! movement+update, attack culls (avg per cull), vision culls (avg per
//! cull) and insert+remove, plus wall time, steps/s and the live cull
//! agreement counter in `both` mode.

use std::time::Instant;

use vectorial_hash::UpdateStrategy;
use vectorial_hash_demos::sim::{build_arsenal_scaled, CullStrategy, Mode, Sim, SimParams, ITEM_LIMIT, MAP_W};

fn parse_strategy(s: &str) -> Option<UpdateStrategy> {
    match s {
        "legacy" => Some(UpdateStrategy::Legacy),
        "lca" => Some(UpdateStrategy::Lca),
        "lca-ropes" | "ropes" => Some(UpdateStrategy::LcaRopes),
        _ => None,
    }
}

fn strategy_label(s: UpdateStrategy) -> &'static str {
    match s {
        UpdateStrategy::Legacy => "legacy",
        UpdateStrategy::Lca => "lca",
        UpdateStrategy::LcaRopes => "lca-ropes",
    }
}

fn parse_cull_strategy(s: &str) -> Option<CullStrategy> {
    match s {
        "descent" => Some(CullStrategy::Descent),
        "walk-samet" | "samet" => Some(CullStrategy::WalkSamet),
        "walk-probe" | "probe" => Some(CullStrategy::WalkProbe),
        "walk-ropes" | "ropes" => Some(CullStrategy::WalkRopes),
        _ => None,
    }
}

fn cull_strategy_label(s: CullStrategy) -> &'static str {
    match s {
        CullStrategy::Descent => "descent",
        CullStrategy::WalkSamet => "walk-samet",
        CullStrategy::WalkProbe => "walk-probe",
        CullStrategy::WalkRopes => "walk-ropes",
    }
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    frames: usize,
    warmup: usize,
    dt: f64,
    targets: [usize; 3],
    split: usize,
    merge: usize,
    seed: u64,
    fire: f64,
    respawn: f64,
    csv: Option<String>,
    strategy: UpdateStrategy,
    no_attack: bool,
    figure_scale: f64,
    cull_strategy: CullStrategy,
    agent_radius: f64,
    world: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mode: Mode::Both,
        frames: 600,
        warmup: 120,
        dt: 1.0 / 60.0,
        targets: [400, 400, 400],
        split: ITEM_LIMIT,
        merge: ITEM_LIMIT,
        seed: 42,
        fire: 1.0,
        respawn: 2.5,
        csv: None,
        strategy: UpdateStrategy::default(),
        no_attack: false,
        figure_scale: 1.0,
        cull_strategy: CullStrategy::default(),
        agent_radius: 0.0,
        world: MAP_W,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = argv.get(i + 1).cloned();
        let need = |v: &Option<String>| -> String {
            v.clone().unwrap_or_else(|| panic!("missing value for {key}"))
        };
        match key {
            "--mode" => a.mode = Mode::parse(&need(&val)).expect("mode: binary|quad|both"),
            "--frames" => a.frames = need(&val).parse().unwrap(),
            "--warmup" => a.warmup = need(&val).parse().unwrap(),
            "--dt" => a.dt = need(&val).parse().unwrap(),
            "--drifters" => a.targets[0] = need(&val).parse().unwrap(),
            "--hunters" => a.targets[1] = need(&val).parse().unwrap(),
            "--pulsars" => a.targets[2] = need(&val).parse().unwrap(),
            "--split" => a.split = need(&val).parse().unwrap(),
            "--merge" => a.merge = need(&val).parse().unwrap(),
            "--seed" => a.seed = need(&val).parse().unwrap(),
            "--fire" => a.fire = need(&val).parse().unwrap(),
            "--respawn" => a.respawn = need(&val).parse().unwrap(),
            "--csv" => a.csv = Some(need(&val)),
            "--update-strategy" | "--strategy" => a.strategy = parse_strategy(&need(&val))
                .expect("update-strategy: legacy|lca|lca-ropes"),
            "--no-attack" => { a.no_attack = true; i -= 1; }
            "--figure-scale" => a.figure_scale = need(&val).parse().unwrap(),
            "--agent-radius" => a.agent_radius = need(&val).parse().unwrap(),
            "--world" => a.world = need(&val).parse().unwrap(),
            "--cull-strategy" => a.cull_strategy = parse_cull_strategy(&need(&val))
                .expect("cull-strategy: descent|walk-samet|walk-probe|walk-ropes"),
            other => panic!("unknown argument: {other}"),
        }
        i += 2;
    }
    a.merge = a.merge.min(a.split);
    a
}

#[derive(Default)]
struct SeriesStats {
    samples: Vec<f64>,
}

impl SeriesStats {
    fn push(&mut self, v: f64) {
        self.samples.push(v);
    }
    fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }
    fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut v = self.samples.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((v.len() as f64 - 1.0) * p).round() as usize;
        v[idx]
    }
}

fn main() {
    let args = parse_args();
    println!(
        "headless critters | mode={} | world={}^2 | pop target={}+{}+{}={} | frames={} (+{} warmup) | dt={:.4}s | split>{} merge<={} | seed={} | update={} | cull={}",
        args.mode.name(),
        args.world,
        args.targets[0],
        args.targets[1],
        args.targets[2],
        args.targets.iter().sum::<usize>(),
        args.frames,
        args.warmup,
        args.dt,
        args.split,
        args.merge,
        args.seed,
        strategy_label(args.strategy),
        cull_strategy_label(args.cull_strategy),
    );

    let t0 = Instant::now();
    let arsenal = build_arsenal_scaled(args.figure_scale);
    println!(
        "bank ready in {:.2}s ({} combos, {} unique grids)",
        arsenal.gen_seconds,
        arsenal.bank.entry_count(),
        arsenal.bank.unique_count(),
    );

    let params = SimParams {
        targets: args.targets,
        respawn_delay: args.respawn,
        fire_rate: args.fire,
        no_attack: args.no_attack,
        agent_radius: args.agent_radius,
    };
    let mut sim = Sim::new(args.mode, args.split, args.merge, args.seed);
    if (args.world - sim.world_size()).abs() > 0.5 {
        sim.set_world_size(args.world, args.split, args.merge);
    }
    sim.sims.update_strategy = args.strategy;
    sim.sims.cull_strategy = args.cull_strategy;

    // Warmup: reach the target population and a steady tree shape.
    for _ in 0..args.warmup {
        sim.sims.begin_frame();
        sim.step(args.dt, &arsenal, &params);
        sim.events.clear();
    }

    // Measured run.
    let mut csv: Option<std::fs::File> = args.csv.as_ref().map(|p| {
        use std::io::Write;
        let mut f = std::fs::File::create(p).expect("create csv");
        writeln!(f, "frame,struct,move_us,atk_avg_us,atk_n,vis_avg_us,vis_n,rm_us").unwrap();
        f
    });

    let mut stats: Vec<(&str, [SeriesStats; 4])> = vec![
        ("binary", Default::default()),
        ("quad", Default::default()),
        ("itree", Default::default()),
    ];
    let measure_start = Instant::now();
    for frame in 0..args.frames {
        sim.sims.begin_frame();
        sim.step(args.dt, &arsenal, &params);
        sim.events.clear();

        for (name, ops) in [
            ("binary", sim.sims.t),
            ("quad", sim.sims.q),
            ("itree", sim.sims.it),
        ] {
            let entry = stats.iter_mut().find(|(n, _)| *n == name).unwrap();
            let atk_avg = if ops.atk_n > 0 { ops.atk / ops.atk_n as f64 } else { 0.0 };
            let vis_avg = if ops.vis_n > 0 { ops.vis / ops.vis_n as f64 } else { 0.0 };
            entry.1[0].push(ops.mv);
            entry.1[1].push(atk_avg);
            entry.1[2].push(vis_avg);
            entry.1[3].push(ops.rm);
            if let Some(f) = &mut csv {
                use std::io::Write;
                writeln!(
                    f,
                    "{frame},{name},{:.2},{:.2},{},{:.2},{},{:.2}",
                    ops.mv, atk_avg, ops.atk_n, vis_avg, ops.vis_n, ops.rm,
                )
                .unwrap();
            }
        }
    }
    let wall = measure_start.elapsed().as_secs_f64();

    let alive = sim.sims.item_count();
    println!(
        "\nsimulated {} frames in {:.2}s wall ({:.0} steps/s) | alive {} | kills {} | respawn queue {}",
        args.frames,
        wall,
        args.frames as f64 / wall,
        alive,
        sim.kills,
        sim.respawn_queue_len(),
    );
    if args.mode == Mode::Both {
        println!(
            "cull agreement: {}",
            if sim.sims.mismatches == 0 {
                "all culls agree".to_string()
            } else {
                format!("{} MISMATCHES", sim.sims.mismatches)
            },
        );
    }
    if let Some(t) = &sim.sims.tree {
        println!("binary: {} leaves, {} arena nodes", t.leaf_count(), t.node_count());
    }
    if let Some(q) = &sim.sims.quad {
        println!("quad:   {} leaves, {} arena nodes", q.leaf_count(), q.node_count());
    }
    if let Some(t) = &sim.sims.itree {
        println!("itree:  {} leaves, {} arena nodes", t.leaf_count(), t.node_count());
    }

    println!(
        "\n{:<8} {:<22} {:>10} {:>10} {:>10}",
        "struct", "op (per frame)", "mean", "p50", "p95",
    );
    let labels = [
        "move+update (us)",
        "attack cull avg (us)",
        "vision cull avg (us)",
        "insert+remove (us)",
    ];
    for (name, series) in &stats {
        let active = match (args.mode, *name) {
            (Mode::Binary, "binary") | (Mode::Both, "binary") => true,
            (Mode::Quad, "quad")     | (Mode::Both, "quad")    => true,
            (Mode::IBinary, "itree") => true,
            _ => false,
        };
        if !active {
            continue;
        }
        for (i, label) in labels.iter().enumerate() {
            println!(
                "{:<8} {:<22} {:>10.1} {:>10.1} {:>10.1}",
                name,
                label,
                series[i].mean(),
                series[i].percentile(0.5),
                series[i].percentile(0.95),
            );
        }
    }
    println!("\ntotal wall (incl. bank gen): {:.2}s", t0.elapsed().as_secs_f64());
    if let Some(p) = &args.csv {
        println!("per-frame CSV written to {p}");
    }
}
