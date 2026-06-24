//! 3D culling: true-3D-tree vs projection-indexing (3 × 2D trees), on
//! time AND precision, against a brute-force ground truth.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin tree3d_bench --release -- \
//!     --pop 50000 --item-limit 8 --queries 200 --seed 42
//! ```
//!
//! Four families to answer "which 3D points lie inside this sphere?":
//!
//! 1. **True 3D tree** (`Tree3`): binary split in 3D, sphere classified
//!    against each node box (green/white/yellow), 1×1×1 voxel raster at
//!    leaves. Exact, but a real 3D template bank would be N³ in memory
//!    (here the sphere is analytic, so no bank is needed — best case for
//!    the 3D tree).
//! 2. **Octree** (`Octree3`): the 8-way 2×2×2 split — the same Shape3
//!    machinery, one level doing the work of three binary levels.
//! 3. **Morton / Z-order grid** (`MortonGrid3`): pointer-free. Quantise each
//!    point to an integer cell, pack the cell's bits into a Z-order code,
//!    bucket by code in a hash. A cull visits only the cells overlapping the
//!    query bbox (green/white/yellow per cell). One fixed resolution, no
//!    adaptive depth — the cell size is the whole knob (set here ≈ the mean
//!    query radius).
//! 4. **Projection indexing** (author's idea): three 2D trees on the (x,y),
//!    (x,z), (y,z) projections. Cull each with the sphere's circular
//!    shadow, intersect the candidate id sets, then run the exact 3D test
//!    on survivors. Reuses the 2D machinery, no N³ memory — but the
//!    intersection is a *broadphase* (a superset; the corners of the
//!    three-cylinder intersection that stick out of the sphere are false
//!    positives that the exact test drops). A 1-projection variant culls one
//!    plane and exact-tests its shadow.
//!
//! Reports: ns/query for each, and the projection's false-positive ratio
//! (candidates after intersection ÷ true hits) which drives its exact-test
//! cost. Correctness of every method is gated against brute force.

use std::collections::HashSet;
use std::time::Instant;

use vectorial_hash::{
    Aabb, CellState, MortonGrid3, Octree3, Point, Point3, Positioned, Positioned3, QuadTree, Rect,
    Sphere3, Tree, Tree3, VoxelRaster,
};

const WORLD: f64 = 512.0;

struct Args {
    pop: usize,
    item_limit: usize,
    queries: usize,
    seed: u64,
    rmin: f64,
    rmax: f64,
    stack: bool,
    knn: usize,
}

fn parse_args() -> Args {
    let mut a = Args { pop: 50000, item_limit: 8, queries: 200, seed: 42, rmin: 10.0, rmax: 80.0, stack: false, knn: 0 };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || argv.get(i + 1).cloned().unwrap_or_else(|| panic!("missing value for {key}"));
        match key {
            "--pop" => a.pop = val().parse().unwrap(),
            "--item-limit" => a.item_limit = val().parse().unwrap(),
            "--queries" => a.queries = val().parse().unwrap(),
            "--seed" => a.seed = val().parse().unwrap(),
            "--rmin" => a.rmin = val().parse().unwrap(),
            "--rmax" => a.rmax = val().parse().unwrap(),
            // Stack points in height: cluster the (x,y) projection (few
            // columns, many z per column) so the xy-shadow is dense — the
            // "things stacked in height" regime where the quadtree projection
            // earns its keep.
            "--stack" => { a.stack = true; i -= 1; }
            // Also run a k-nearest-neighbour comparison (Tree3 / Octree3 vs
            // brute) with this k. 0 = skip (sphere culls only).
            "--knn" => a.knn = val().parse().unwrap(),
            other => panic!("unknown argument: {other}"),
        }
        i += 2;
    }
    a
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.max(1)) }
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

// 3D item.
#[derive(Clone, Copy)]
struct I3 { id: u32, p: Point3 }
impl Positioned3 for I3 { fn position(&self) -> Point3 { self.p } }

// 2D projection item (carries the id so we can intersect across planes).
#[derive(Clone, Copy)]
struct I2 { id: u32, p: Point }
impl Positioned for I2 { fn position(&self) -> Point { self.p } }

// 2D circle (the sphere's shadow). No template — bbox reject + exact test,
// i.e. the projection approach reusing the 2D index out of the box.
struct Circle2 { cx: f64, cy: f64, r: f64 }
impl vectorial_hash::Shape for Circle2 {
    fn bounding_box(&self) -> Rect {
        Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r)
    }
    fn contains_point(&self, p: Point) -> bool {
        let dx = p.x - self.cx; let dy = p.y - self.cy;
        dx * dx + dy * dy <= self.r * self.r
    }
}

struct Stats { ns: Vec<f64> }
impl Stats {
    fn new() -> Self { Stats { ns: Vec::new() } }
    fn push(&mut self, v: f64) { self.ns.push(v); }
    fn mean(&self) -> f64 { if self.ns.is_empty() { 0.0 } else { self.ns.iter().sum::<f64>() / self.ns.len() as f64 } }
    fn p95(&self) -> f64 {
        if self.ns.is_empty() { return 0.0; }
        let mut v = self.ns.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 - 1.0) * 0.95).round() as usize]
    }
}

fn main() {
    let args = parse_args();
    println!("tree3d bench | pop={} | item_limit={} | queries={} | world={}^3 | seed={}",
        args.pop, args.item_limit, args.queries, WORLD, args.seed);

    println!("distribution: {}", if args.stack { "STACKED (dense xy, tall columns)" } else { "uniform 3D" });
    let mut rng = Rng::new(args.seed);
    let mut items: Vec<I3> = Vec::with_capacity(args.pop);
    let mut pos: Vec<Point3> = Vec::with_capacity(args.pop);
    for id in 0..args.pop {
        let p = if args.stack {
            // ~64×64 grid of columns; each column holds a tall stack in z.
            // The xy-projection is dense (many ids share a small xy cell).
            let gx = (rng.next() % 64) as f64 * (WORLD / 64.0) + rng.range(0.0, WORLD / 64.0);
            let gy = (rng.next() % 64) as f64 * (WORLD / 64.0) + rng.range(0.0, WORLD / 64.0);
            Point3::new(gx.clamp(2.0, WORLD - 2.0), gy.clamp(2.0, WORLD - 2.0), rng.range(2.0, WORLD - 2.0))
        } else {
            Point3::new(rng.range(2.0, WORLD - 2.0), rng.range(2.0, WORLD - 2.0), rng.range(2.0, WORLD - 2.0))
        };
        items.push(I3 { id: id as u32, p });
        pos.push(p);
    }

    // --- build the structures ---
    let t_build3 = Instant::now();
    let mut tree3 = Tree3::<I3>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), args.item_limit);
    for it in &items { tree3.insert(*it); }
    let build3_ms = t_build3.elapsed().as_secs_f64() * 1e3;

    // Octree (8-way) on the same data — the structural alternative to the
    // binary-3D tree.
    let mut octree = Octree3::<I3>::new(Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD), args.item_limit);
    for it in &items { octree.insert(*it); }

    // Morton / Z-order linear grid (the pointer-free fourth structure). Pick the
    // cell ≈ the mean query radius — coarse enough that a query touches few
    // cells, fine enough that each cell holds few points. The grid has no
    // adaptive depth, so this single resolution is the whole knob.
    let world_aabb = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let morton_target = (args.rmin + args.rmax) * 0.5;
    let morton_levels = MortonGrid3::<I3>::levels_for_cell_size(world_aabb, morton_target);
    let t_buildm = Instant::now();
    let mut morton = MortonGrid3::<I3>::new(world_aabb, morton_levels);
    for it in &items { morton.insert(*it); }
    let buildm_ms = t_buildm.elapsed().as_secs_f64() * 1e3;

    let t_buildp = Instant::now();
    let mut tree_xy = Tree::<I2>::new(Rect::new(0.0, 0.0, WORLD, WORLD), args.item_limit);
    let mut tree_xz = Tree::<I2>::new(Rect::new(0.0, 0.0, WORLD, WORLD), args.item_limit);
    let mut tree_yz = Tree::<I2>::new(Rect::new(0.0, 0.0, WORLD, WORLD), args.item_limit);
    for it in &items {
        tree_xy.insert(I2 { id: it.id, p: Point::new(it.p.x, it.p.y) });
        tree_xz.insert(I2 { id: it.id, p: Point::new(it.p.x, it.p.z) });
        tree_yz.insert(I2 { id: it.id, p: Point::new(it.p.y, it.p.z) });
    }
    // QuadTree on the xy-projection — the structure to keep for "stacked in
    // height" worlds, where the xy-shadow is dense and the quadtree's 4-way
    // split handles density better than the binary tree (BENCHMARKS Results 6).
    let mut quad_xy = QuadTree::<I2>::new(Rect::new(0.0, 0.0, WORLD, WORLD), args.item_limit);
    for it in &items { quad_xy.insert(I2 { id: it.id, p: Point::new(it.p.x, it.p.y) }); }
    let buildp_ms = t_buildp.elapsed().as_secs_f64() * 1e3;

    println!("build: 3D tree {:.1} ms ({} nodes) | octree ({} nodes) | 3×2D trees {:.1} ms ({}+{}+{} nodes)",
        build3_ms, tree3.node_count(), octree.node_count(), buildp_ms,
        tree_xy.node_count(), tree_xz.node_count(), tree_yz.node_count());
    println!("morton grid: {:.1} ms | levels {} ({} cells/axis, cell {:.1}) | {} non-empty cells, {:.2} items/cell",
        buildm_ms, morton_levels, 1u32 << morton_levels, WORLD / (1u64 << morton_levels) as f64,
        morton.cell_count(), morton.item_count() as f64 / morton.cell_count().max(1) as f64);

    // --- queries ---
    let mut s_brute = Stats::new();
    let mut s_tree3 = Stats::new();
    let mut s_octree = Stats::new();
    let mut mismatches_o = 0u64;
    let mut s_morton = Stats::new();
    let mut mismatches_m = 0u64;
    let mut s_proj = Stats::new();
    let mut s_proj_broad = Stats::new(); // projection broadphase only (before exact filter)
    let mut s_proj1 = Stats::new();      // single-projection + exact filter
    let mut s_proj1z = Stats::new();     // single-projection + z-slab reject + exact
    let mut s_proj1r = Stats::new();     // single-projection + voxel-raster narrowphase
    let mut s_proj1q = Stats::new();     // single-projection via quadtree + exact
    let mut total_true = 0u64;
    let mut total_cand = 0u64;
    let mut total_cand1 = 0u64;
    let mut mismatches3 = 0u64;
    let mut mismatchesp = 0u64;
    let mut mismatchesp1 = 0u64;
    let mut mismatchesp1z = 0u64;
    let mut mismatchesp1r = 0u64;
    let mut mismatchesp1q = 0u64;

    for q in 0..args.queries {
        // Radius spread so some queries are small, some large.
        let r = rng.range(args.rmin, args.rmax);
        let cx = rng.range(r, WORLD - r);
        let cy = rng.range(r, WORLD - r);
        let cz = rng.range(r, WORLD - r);

        // Ground truth (brute force).
        let t = Instant::now();
        let mut brute: Vec<u32> = items.iter()
            .filter(|it| { let dx = it.p.x - cx; let dy = it.p.y - cy; let dz = it.p.z - cz; dx*dx+dy*dy+dz*dz <= r*r })
            .map(|it| it.id).collect();
        s_brute.push(t.elapsed().as_secs_f64() * 1e9);
        brute.sort();
        let brute_set: HashSet<u32> = brute.iter().copied().collect();
        total_true += brute.len() as u64;

        // True 3D tree.
        let sphere = Sphere3::new(cx, cy, cz, r).with_raster();
        let t = Instant::now();
        let hits3: Vec<u32> = tree3.cull(&sphere).iter().map(|it| it.id).collect();
        s_tree3.push(t.elapsed().as_secs_f64() * 1e9);
        let set3: HashSet<u32> = hits3.iter().copied().collect();
        if set3 != brute_set { mismatches3 += 1; }

        // Octree (8-way) on the same sphere.
        let t = Instant::now();
        let hits_o: HashSet<u32> = octree.cull(&sphere).iter().map(|it| it.id).collect();
        s_octree.push(t.elapsed().as_secs_f64() * 1e9);
        if hits_o != brute_set { mismatches_o += 1; }

        // Morton / Z-order linear grid on the same sphere.
        let t = Instant::now();
        let hits_m: HashSet<u32> = morton.cull(&sphere).iter().map(|it| it.id).collect();
        s_morton.push(t.elapsed().as_secs_f64() * 1e9);
        if hits_m != brute_set { mismatches_m += 1; }

        // Projection: cull 3 circles, intersect, exact 3D filter.
        let t = Instant::now();
        let cull_xy = tree_xy.cull(&Circle2 { cx, cy, r });
        let cull_xz = tree_xz.cull(&Circle2 { cx: cx, cy: cz, r });
        let cull_yz = tree_yz.cull(&Circle2 { cx: cy, cy: cz, r });
        // Intersect the smallest against the others via id sets.
        let set_xz: HashSet<u32> = cull_xz.iter().map(|i| i.id).collect();
        let set_yz: HashSet<u32> = cull_yz.iter().map(|i| i.id).collect();
        let cand: Vec<u32> = cull_xy.iter().map(|i| i.id)
            .filter(|id| set_xz.contains(id) && set_yz.contains(id))
            .collect();
        let broad_ns = t.elapsed().as_secs_f64() * 1e9;
        s_proj_broad.push(broad_ns);
        // Exact 3D narrowphase on the candidates.
        let projhits: Vec<u32> = cand.iter().copied()
            .filter(|&id| { let p = pos[id as usize]; let dx = p.x-cx; let dy = p.y-cy; let dz = p.z-cz; dx*dx+dy*dy+dz*dz <= r*r })
            .collect();
        s_proj.push(t.elapsed().as_secs_f64() * 1e9);
        let setp: HashSet<u32> = projhits.iter().copied().collect();
        if setp != brute_set { mismatchesp += 1; }
        total_cand += cand.len() as u64;

        // Single-projection broadphase: cull ONE plane (xy), exact-filter
        // its shadow in 3D. Larger candidate set than the 3-way intersect,
        // but no extra culls/hashing — wins when the exact test is cheap.
        let t = Instant::now();
        let cand1 = tree_xy.cull(&Circle2 { cx, cy, r });
        let proj1hits: Vec<u32> = cand1.iter().map(|i| i.id)
            .filter(|&id| { let p = pos[id as usize]; let dx = p.x-cx; let dy = p.y-cy; let dz = p.z-cz; dx*dx+dy*dy+dz*dz <= r*r })
            .collect();
        s_proj1.push(t.elapsed().as_secs_f64() * 1e9);
        let setp1: HashSet<u32> = proj1hits.iter().copied().collect();
        if setp1 != brute_set { mismatchesp1 += 1; }
        total_cand1 += cand1.len() as u64;

        // 1-projection + z-slab reject: a cheap 1D bbox reject on z BEFORE
        // the full distance test (|z - cz| <= r drops candidates outside
        // the sphere's z-extent — exactly the column points the xy-shadow
        // dragged in but the index couldn't prune).
        let t = Instant::now();
        let cand1z = tree_xy.cull(&Circle2 { cx, cy, r });
        let proj1zhits: Vec<u32> = cand1z.iter().map(|i| i.id)
            .filter(|&id| { let p = pos[id as usize]; (p.z - cz).abs() <= r })
            .filter(|&id| { let p = pos[id as usize]; let dx = p.x-cx; let dy = p.y-cy; let dz = p.z-cz; dx*dx+dy*dy+dz*dz <= r*r })
            .collect();
        s_proj1z.push(t.elapsed().as_secs_f64() * 1e9);
        if proj1zhits.iter().copied().collect::<HashSet<u32>>() != brute_set { mismatchesp1z += 1; }

        // 1-projection + voxel-raster narrowphase: instead of the analytic
        // distance test, look the candidate up in the sphere's 1×1×1 raster
        // (In/Out resolve by lookup; Maybe runs exact). For a sphere the
        // exact test is already trivial so this should NOT help — included
        // to confirm the raster only pays when contains_point is expensive.
        let t = Instant::now();
        let cand1r = tree_xy.cull(&Circle2 { cx, cy, r });
        let raster = VoxelRaster::for_sphere(cx, cy, cz, r);
        let proj1rhits: Vec<u32> = cand1r.iter().map(|i| i.id)
            .filter(|&id| {
                let p = pos[id as usize];
                match raster.cell_at_world(p) {
                    CellState::In => true,
                    CellState::Out => false,
                    CellState::Maybe => { let dx=p.x-cx; let dy=p.y-cy; let dz=p.z-cz; dx*dx+dy*dy+dz*dz <= r*r }
                }
            })
            .collect();
        s_proj1r.push(t.elapsed().as_secs_f64() * 1e9);
        if proj1rhits.iter().copied().collect::<HashSet<u32>>() != brute_set { mismatchesp1r += 1; }

        // 1-projection via QuadTree + z-reject + exact (the stacking case).
        let t = Instant::now();
        let cand1q = quad_xy.cull(&Circle2 { cx, cy, r });
        let proj1qhits: Vec<u32> = cand1q.iter().map(|i| i.id)
            .filter(|&id| { let p = pos[id as usize]; (p.z - cz).abs() <= r })
            .filter(|&id| { let p = pos[id as usize]; let dx = p.x-cx; let dy = p.y-cy; let dz = p.z-cz; dx*dx+dy*dy+dz*dz <= r*r })
            .collect();
        s_proj1q.push(t.elapsed().as_secs_f64() * 1e9);
        if proj1qhits.iter().copied().collect::<HashSet<u32>>() != brute_set { mismatchesp1q += 1; }

        let _ = q;
    }

    let fp_ratio = total_cand as f64 / total_true.max(1) as f64;
    let fp_ratio1 = total_cand1 as f64 / total_true.max(1) as f64;
    println!("\nqueries={} | true hits total={} | mean hits/query={:.0}",
        args.queries, total_true, total_true as f64 / args.queries as f64);
    println!("broadphase candidate/true ratio: 3-projection {:.2}x | 1-projection {:.2}x",
        fp_ratio, fp_ratio1);
    let allok = mismatches3 == 0 && mismatches_o == 0 && mismatches_m == 0 && mismatchesp == 0
        && mismatchesp1 == 0 && mismatchesp1z == 0 && mismatchesp1r == 0 && mismatchesp1q == 0;
    println!("correctness vs brute: all methods {}", if allok { "EXACT" } else { "MISMATCH!" });

    let line = |name: &str, s: &Stats| {
        println!("{:<32} {:>11.0} {:>11.0} {:>9.1}x",
            name, s.mean(), s.p95(), s_brute.mean() / s.mean().max(1e-9));
    };
    println!("\n{:<32} {:>11} {:>11} {:>10}", "method", "mean ns/q", "p95 ns/q", "vs brute");
    line("brute force", &s_brute);
    line("true 3D tree (binary)", &s_tree3);
    line("octree (8-way)", &s_octree);
    line("morton grid (Z-order hash)", &s_morton);
    line("3-projection (intersect+exact)", &s_proj);
    line("1-projection (+exact)", &s_proj1);
    line("1-projection +z-reject +exact", &s_proj1z);
    line("1-projection +raster narrowphase", &s_proj1r);
    line("1-projection via quadtree +z+exact", &s_proj1q);

    // --- optional k-nearest-neighbour comparison ---
    if args.knn > 0 {
        let k = args.knn;
        let mut s_kbrute = Stats::new();
        let mut s_ktree3 = Stats::new();
        let mut s_koct = Stats::new();
        let mut kmism3 = 0u64;
        let mut kmismo = 0u64;
        // Distances of `got` (sqrt'd) squared back, compared to the brute k
        // smallest squared distances (unique set even under boundary ties).
        let dmatch = |got: &[(f64, &I3)], bf: &[f64]| -> bool {
            got.len() == bf.len()
                && got.iter().zip(bf).all(|((d, _), b)| (d * d - b).abs() <= 1e-6 * (1.0 + b.abs()))
        };
        for _ in 0..args.queries {
            let q = Point3::new(rng.range(0.0, WORLD), rng.range(0.0, WORLD), rng.range(0.0, WORLD));
            // Brute: every squared distance, sorted, k smallest.
            let t = Instant::now();
            let mut all: Vec<f64> = pos.iter()
                .map(|p| { let dx = p.x - q.x; let dy = p.y - q.y; let dz = p.z - q.z; dx * dx + dy * dy + dz * dz })
                .collect();
            all.sort_by(|a, b| a.total_cmp(b));
            let bf: Vec<f64> = all.into_iter().take(k).collect();
            s_kbrute.push(t.elapsed().as_secs_f64() * 1e9);

            let t = Instant::now();
            let g3 = tree3.knn(q, k);
            s_ktree3.push(t.elapsed().as_secs_f64() * 1e9);
            if !dmatch(&g3, &bf) { kmism3 += 1; }

            let t = Instant::now();
            let go = octree.knn(q, k);
            s_koct.push(t.elapsed().as_secs_f64() * 1e9);
            if !dmatch(&go, &bf) { kmismo += 1; }
        }
        println!("\nk-NN (k={}) over {} queries | correctness vs brute: {}",
            k, args.queries, if kmism3 == 0 && kmismo == 0 { "EXACT" } else { "MISMATCH!" });
        let kline = |name: &str, s: &Stats| {
            println!("{:<32} {:>11.0} {:>11.0} {:>9.1}x",
                name, s.mean(), s.p95(), s_kbrute.mean() / s.mean().max(1e-9));
        };
        println!("{:<32} {:>11} {:>11} {:>10}", "method", "mean ns/q", "p95 ns/q", "vs brute");
        kline("brute force (full sort)", &s_kbrute);
        kline("true 3D tree (binary) knn", &s_ktree3);
        kline("octree (8-way) knn", &s_koct);
    }
}
