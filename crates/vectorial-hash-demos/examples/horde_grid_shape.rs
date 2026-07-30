//! `horde_grid_shape` — does cubic padding help the *horde's* grid, or only the k-NN bench?
//!
//! `examples/morton_knn_axis_bench` (in the library crate) found that a `MortonGrid3` on a
//! non-cubic world can be made 1.65-2.1x faster at k-NN simply by declaring the world as a
//! **cube**: the cell store is a sparse hash, so the empty layers above the data are neither
//! stored nor traversed, and the cells come out cubic instead of slabs.
//!
//! The horde is the extreme case of the geometry that fix addresses — 1800 x 72 x 1800, so
//! `levels = 5` gives cells of **56.25 x 2.25 x 56.25** — but it is NOT the workload that was
//! measured. The k-NN bench asks for 8 neighbours from random points. The horde overwhelmingly
//! asks for a **radius-3 separation cull, once per awake zombie per frame**, plus a handful of
//! big rings (55 / 84 / 110) and two k-NN shapes (k=8 towers, k=48 commander).
//!
//! Those pull in opposite directions, which is why this exists rather than an assumption:
//!
//! - A big ring covers every populated y-layer either way, so both geometries test the same
//!   points — but the slab grid needs ~4x the cell lookups to do it. Cubic should win.
//! - A radius-3 sphere spans ~3 slab cells in y and only ~1 cubic cell. Same lookups-ish, but
//!   the cubic cell holds the WHOLE vertical column, so it should test more points. Cubic
//!   could lose, and this is the query the horde runs thousands of times a frame.
//!
//! The population is not modelled here: it is pulled out of a real `Horde`, so the density and
//! the clustering are the demo's own, and query centres are real unit positions. Note that the
//! wake-up loop barely wakes anything — the sim caps the active front on purpose — which is
//! fine and arguably right: the index holds every unit whether it is awake or asleep, so what
//! is being measured is exactly what the grid actually contains.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example horde_grid_shape --release
//! ```
//! Env: `HGS_POP` (default 50 000), `HGS_Q` (queries per shape, default 400).

use std::cell::Cell;
use std::time::Instant;
use vectorial_hash::{Aabb, MortonGrid3, Point3, Positioned3, Sphere3};
use vectorial_hash_demos::horde_sim::{Horde, SKY, WORLD};

thread_local! { static TESTED: Cell<u64> = const { Cell::new(0) }; }

#[derive(Clone, Copy)]
struct P {
    p: Point3,
}
impl Positioned3 for P {
    fn position(&self) -> Point3 {
        TESTED.with(|c| c.set(c.get() + 1));
        self.p
    }
}

/// Min-of-N wall milliseconds for a batch. The library's cycle harness lives in the other
/// crate; this only ever compares two grids inside one process, back to back, so the
/// interleaving that `compare2` provides is done by the caller's loop order instead.
fn best_ms<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    let mut lo = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        f();
        lo = lo.min(t.elapsed().as_secs_f64() * 1e3);
    }
    lo
}

struct Shape {
    label: &'static str,
    radius: f64,
    knn_k: usize,
}

fn main() {
    let pop: usize = std::env::var("HGS_POP").ok().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let nq: usize = std::env::var("HGS_Q").ok().and_then(|s| s.parse().ok()).unwrap_or(400);

    // A real horde, woken enough that the population is spread the way a fight spreads it
    // rather than sitting in its spawn lattice.
    let mut h = Horde::new(7, pop);
    h.step(1.0 / 60.0);
    for k in 0..(pop / 1500).max(1) {
        let p = h.units[(k * 1499) % h.units.len()].p;
        h.emit_noise(p, 2000.0);
        h.step(1.0 / 60.0);
        let (_, a) = h.counts();
        if a * 10 >= pop {
            break;
        }
    }
    for _ in 0..120 {
        h.step(1.0 / 60.0);
    }
    let items: Vec<P> = h.units.iter().filter(|z| z.alive()).map(|z| P { p: z.p }).collect();
    let (asleep, awake) = h.counts();
    let ys: (f64, f64) = items.iter().fold((f64::MAX, f64::MIN), |(lo, hi), it| (lo.min(it.p.y), hi.max(it.p.y)));
    println!("horde grid shape | {} live units ({asleep} asleep / {awake} awake) | y in [{:.1}, {:.1}]", items.len(), ys.0, ys.1);
    println!("world {WORLD} x {} x {WORLD} (aspect {:.3}) | {nq} queries per shape\n", SKY + 8.0, (SKY + 8.0) / WORLD);

    // Query centres taken from real unit positions — a separation cull is centred on a
    // zombie, and the rings sit where the defence is, which is where the zombies are going.
    let centres: Vec<Point3> = (0..nq).map(|i| items[(i * 7919) % items.len()].p).collect();

    let native = Aabb::new(0.0, -8.0, 0.0, WORLD, SKY + 8.0, WORLD);
    let cube = Aabb::new(0.0, -8.0, 0.0, WORLD, WORLD, WORLD);

    let shapes = [
        Shape { label: "cull r=3 (separation)", radius: 3.0, knn_k: 0 },
        Shape { label: "cull r=55 (guard)", radius: 55.0, knn_k: 0 },
        Shape { label: "cull r=84 (tower ring)", radius: 84.0, knn_k: 0 },
        Shape { label: "cull r=110 (sector)", radius: 110.0, knn_k: 0 },
        Shape { label: "knn k=8 (tower aim)", radius: 0.0, knn_k: 8 },
        Shape { label: "knn k=48 (commander)", radius: 0.0, knn_k: 48 },
    ];

    // levels 5 is what horde_sim ships. 6 and 7 are included because padding to a cube also
    // changes what a level MEANS: the cell side is world_max / 2^levels either way, so the
    // cube's cells are the same size as the slab's x/z, not its y.
    let configs: [(&str, Aabb, u32); 6] = [
        ("native slab  L5", native, 5),
        ("cubic padded L5", cube, 5),
        ("native slab  L6", native, 6),
        ("cubic padded L6", cube, 6),
        ("cubic padded L7", cube, 7),
        ("cubic padded L8", cube, 8),
    ];

    let mut grids = Vec::new();
    for (label, world, lv) in configs {
        let t = Instant::now();
        let mut g = MortonGrid3::new(world, lv);
        for it in &items {
            g.insert(*it);
        }
        let build = t.elapsed().as_secs_f64() * 1e3;
        let cells = (1u32 << lv) as f64;
        grids.push((label, g, build, (world.w / cells, world.h / cells, world.d / cells)));
    }

    println!("{:<17} {:>20} {:>10} {:>11}", "grid", "cell (x,y,z)", "cells", "build ms");
    for (label, g, build, cell) in &grids {
        println!("{label:<17} {:>20} {:>10} {:>11.1}", format!("{:.1},{:.1},{:.1}", cell.0, cell.1, cell.2), g.cell_count(), build);
    }

    for s in &shapes {
        println!("\n{}", s.label);
        println!("  {:<17} {:>14} {:>12} {:>12}", "grid", "tested/query", "ms/query", "vs native L5");
        let mut base: Option<f64> = None;
        let mut reference: Option<Vec<usize>> = None;
        for (label, g, _, _) in &grids {
            TESTED.with(|c| c.set(0));
            let mut answers = Vec::with_capacity(nq);
            for q in &centres {
                let n = if s.knn_k > 0 { g.knn(*q, s.knn_k).len() } else { g.cull(&Sphere3::new(q.x, q.y, q.z, s.radius)).len() };
                answers.push(n);
            }
            let tested = TESTED.with(|c| c.get()) as f64 / nq as f64;
            // Different cell geometry must not mean different answers.
            match &reference {
                None => reference = Some(answers),
                Some(rf) => assert_eq!(rf, &answers, "{label} answered {} differently", s.label),
            }
            let ms = best_ms(5, || {
                for q in &centres {
                    if s.knn_k > 0 {
                        std::hint::black_box(g.knn(*q, s.knn_k).len());
                    } else {
                        std::hint::black_box(g.cull(&Sphere3::new(q.x, q.y, q.z, s.radius)).len());
                    }
                }
            }) / nq as f64;
            let rel = match base {
                None => {
                    base = Some(ms);
                    1.0
                }
                Some(b) => b / ms,
            };
            println!("  {label:<17} {tested:>14.1} {ms:>12.5} {rel:>11.2}x");
            println!("#M {}_{}.tested {tested:.1} n", s.label.replace([' ', '=', '(', ')'], "_"), label.replace(' ', "_"));
            println!("#M {}_{}.ms {ms:.6} ms", s.label.replace([' ', '=', '(', ')'], "_"), label.replace(' ', "_"));
        }
    }

    // ---- and does any of it survive contact with the whole simulation? -------------------
    // The micro numbers above isolate the grid. The sim also runs decide(), movement, combat
    // and the flow field, none of which care about cell shape, so a 1.7x on the index can
    // easily be a rounding error on the frame. Reported so nobody quotes the micro figure as
    // a frame-rate claim. Run it on both sides of the change to see what actually moved.
    {
        use vectorial_hash_demos::horde_sim::ZMode;
        let dt = 1.0 / 60.0;
        let mut ms = [0.0f64; 2];
        for (i, mode) in [ZMode::Tree, ZMode::Morton].into_iter().enumerate() {
            let mut hh = Horde::new(7, pop);
            hh.step(dt);
            hh.set_zmode(mode);
            for _ in 0..30 { hh.step(dt); }
            let t = Instant::now();
            let mut frames = 0u64;
            while t.elapsed().as_secs_f64() < 2.0 { hh.step(dt); frames += 1; }
            ms[i] = t.elapsed().as_secs_f64() * 1e3 / frames as f64;
        }
        println!("\nwhole sim, {pop} units: tree {:.3} ms/step | morton {:.3} ms/step (morton is {:.2}x the tree)",
            ms[0], ms[1], ms[0] / ms[1]);
        println!("#M sim.tree_ms_per_step {:.4} ms", ms[0]);
        println!("#M sim.morton_ms_per_step {:.4} ms", ms[1]);
    }

    println!("\nreading: 'vs native L5' is a speed-up (>1 means faster than what horde_sim ships).");
    println!("The separation cull is the one that matters most by volume - the horde runs it once");
    println!("per awake zombie per frame, while the rings run a few dozen times. A geometry that");
    println!("wins the rings and loses r=3 is not obviously a win; weight by call count.");
}
