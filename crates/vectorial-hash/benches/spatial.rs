//! Criterion benchmarks across the spatial structures.
//!
//! ```bash
//! cargo bench -p vectorial-hash                 # run everything
//! cargo bench -p vectorial-hash -- cull         # filter by name
//! cargo bench -p vectorial-hash -- --save-baseline main   # save a baseline
//! cargo bench -p vectorial-hash -- --baseline main        # compare to it
//! ```
//!
//! These are for *exploration* (rich HTML reports under `target/criterion/`).
//! The committed, deterministic **regression gate** that can fail a build is a
//! separate, lower-variance tool — see `benches/README.md` and the
//! `regression_gate` example.

// The `x < lo || x > hi` bounce test reads clearer than `!(lo..=hi).contains()`.
#![allow(clippy::manual_range_contains)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vectorial_hash::{
    Aabb, MortonGrid3, Octree3, Point, Point3, Positioned, Positioned3, QuadTree, Rect, Shape,
    Shape3, Sphere3, Tree, Tree3,
};

// ----------------------------------------------------------- deterministic rng
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s | 1) }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.unit() * (hi - lo) }
}

// ----------------------------------------------------------------- test items
#[derive(Clone, Copy)]
struct C3 { id: u32, p: Point3 }
impl Positioned3 for C3 { fn position(&self) -> Point3 { self.p } }

#[derive(Clone, Copy)]
struct C2 { p: Point }
impl Positioned for C2 { fn position(&self) -> Point { self.p } }

// 2D disc query (no template) — bbox reject + exact circle test.
struct Disc { cx: f64, cy: f64, r: f64 }
impl Shape for Disc {
    fn bounding_box(&self) -> Rect { Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r) }
    fn contains_point(&self, p: Point) -> bool {
        let dx = p.x - self.cx;
        let dy = p.y - self.cy;
        dx * dx + dy * dy <= self.r * self.r
    }
}

const MARGIN: f64 = 4.0;

fn gen3(n: usize, world: f64, seed: u64) -> Vec<C3> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|id| C3 {
            id: id as u32,
            p: Point3::new(
                rng.range(MARGIN, world - MARGIN),
                rng.range(MARGIN, world - MARGIN),
                rng.range(MARGIN, world - MARGIN),
            ),
        })
        .collect()
}

fn gen2(n: usize, world: f64, seed: u64) -> Vec<C2> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| C2 { p: Point::new(rng.range(MARGIN, world - MARGIN), rng.range(MARGIN, world - MARGIN)) })
        .collect()
}

fn vel3(n: usize, seed: u64, speed: f64) -> Vec<(f64, f64, f64)> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let s = rng.range(0.35 * speed, speed);
            let (a, b) = (rng.range(0.0, std::f64::consts::TAU), rng.range(-1.0, 1.0));
            let h = (1.0_f64 - b * b).max(0.0).sqrt();
            (s * h * a.cos(), s * h * a.sin(), s * b)
        })
        .collect()
}

#[inline]
fn step3(p: &mut Point3, v: &mut (f64, f64, f64), world: f64, dt: f64) -> Point3 {
    let mut nx = p.x + v.0 * dt;
    let mut ny = p.y + v.1 * dt;
    let mut nz = p.z + v.2 * dt;
    if nx < MARGIN || nx > world - MARGIN { v.0 = -v.0; nx = nx.clamp(MARGIN, world - MARGIN); }
    if ny < MARGIN || ny > world - MARGIN { v.1 = -v.1; ny = ny.clamp(MARGIN, world - MARGIN); }
    if nz < MARGIN || nz > world - MARGIN { v.2 = -v.2; nz = nz.clamp(MARGIN, world - MARGIN); }
    *p = Point3::new(nx, ny, nz);
    *p
}

const WORLD: f64 = 512.0;
const N: usize = 20_000;
const IL: usize = 8;
const VISION: f64 = 36.0;
const N_QUERY: usize = 64;

fn queries3(world: f64, seed: u64) -> Vec<Sphere3> {
    let mut rng = Rng::new(seed);
    (0..N_QUERY).map(|_| Sphere3::new(rng.range(0.0, world), rng.range(0.0, world), rng.range(0.0, world), VISION)).collect()
}

// ------------------------------------------------------------------- build
fn bench_build(c: &mut Criterion) {
    let items3 = gen3(N, WORLD, 1);
    let items2 = gen2(N, WORLD, 1);
    let aabb = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let rect = Rect::new(0.0, 0.0, WORLD, WORLD);
    let levels = MortonGrid3::<C3>::levels_for_cell_size(aabb, VISION);

    let mut g = c.benchmark_group("build_20k");
    g.bench_function("tree3", |b| b.iter(|| { let mut t = Tree3::<C3>::new(aabb, IL); for it in &items3 { t.insert(*it); } black_box(&t); }));
    g.bench_function("octree3", |b| b.iter(|| { let mut t = Octree3::<C3>::new(aabb, IL); for it in &items3 { t.insert(*it); } black_box(&t); }));
    g.bench_function("morton3", |b| b.iter(|| { let mut t = MortonGrid3::<C3>::new(aabb, levels); for it in &items3 { t.insert(*it); } black_box(&t); }));
    g.bench_function("tree2", |b| b.iter(|| { let mut t = Tree::<C2>::new(rect, IL); for it in &items2 { t.insert(*it); } black_box(&t); }));
    g.bench_function("quadtree", |b| b.iter(|| { let mut t = QuadTree::<C2>::new(rect, IL); for it in &items2 { t.insert(*it); } black_box(&t); }));
    g.finish();
}

// ------------------------------------------------------------------- cull
fn bench_cull(c: &mut Criterion) {
    let items3 = gen3(N, WORLD, 1);
    let items2 = gen2(N, WORLD, 1);
    let aabb = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let rect = Rect::new(0.0, 0.0, WORLD, WORLD);
    let levels = MortonGrid3::<C3>::levels_for_cell_size(aabb, VISION);
    let q3 = queries3(WORLD, 99);
    let q2: Vec<Disc> = q3.iter().map(|s| { let bb = s.bounding_box(); Disc { cx: bb.x + bb.w * 0.5, cy: bb.y + bb.h * 0.5, r: VISION } }).collect();

    let mut tree3 = Tree3::<C3>::new(aabb, IL); for it in &items3 { tree3.insert(*it); }
    let mut octree3 = Octree3::<C3>::new(aabb, IL); for it in &items3 { octree3.insert(*it); }
    let mut morton3 = MortonGrid3::<C3>::new(aabb, levels); for it in &items3 { morton3.insert(*it); }
    let mut tree2 = Tree::<C2>::new(rect, IL); for it in &items2 { tree2.insert(*it); }
    let mut quad = QuadTree::<C2>::new(rect, IL); for it in &items2 { quad.insert(*it); }

    let mut g = c.benchmark_group("cull_20k_x64");
    g.bench_function("tree3", |b| b.iter(|| { let mut n = 0; for s in &q3 { n += tree3.cull(s).len(); } black_box(n) }));
    g.bench_function("octree3", |b| b.iter(|| { let mut n = 0; for s in &q3 { n += octree3.cull(s).len(); } black_box(n) }));
    g.bench_function("morton3", |b| b.iter(|| { let mut n = 0; for s in &q3 { n += morton3.cull(s).len(); } black_box(n) }));
    g.bench_function("tree2", |b| b.iter(|| { let mut n = 0; for s in &q2 { n += tree2.cull(s).len(); } black_box(n) }));
    g.bench_function("quadtree", |b| b.iter(|| { let mut n = 0; for s in &q2 { n += quad.cull(s).len(); } black_box(n) }));
    g.finish();
}

// ------------------------------------------------------- update: predicate vs ItemRef
fn bench_update(c: &mut Criterion) {
    let items3 = gen3(N, WORLD, 1);
    let aabb = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let dt = 1.0 / 60.0;

    let mut g = c.benchmark_group("update_3d_20k_per_frame");
    g.bench_function("predicate", |b| {
        let mut tree = Tree3::<C3>::new(aabb, IL);
        for it in &items3 { tree.insert(*it); }
        let mut pos: Vec<Point3> = items3.iter().map(|i| i.p).collect();
        let mut vel = vel3(N, 5, 120.0);
        b.iter(|| {
            for id in 0..N {
                let old = pos[id];
                let np = step3(&mut pos[id], &mut vel[id], WORLD, dt);
                let cid = id as u32;
                tree.update(old, |c| c.id == cid, |c| c.p = np);
            }
            black_box(&tree);
        });
    });
    g.bench_function("item_ref", |b| {
        let mut tree = Tree3::<C3>::new(aabb, IL);
        let mut refs = Vec::with_capacity(N);
        for it in &items3 { refs.push(tree.insert_ref(*it).unwrap()); }
        let mut pos: Vec<Point3> = items3.iter().map(|i| i.p).collect();
        let mut vel = vel3(N, 5, 120.0);
        b.iter(|| {
            for id in 0..N {
                let np = step3(&mut pos[id], &mut vel[id], WORLD, dt);
                tree.update_ref(refs[id], |c| c.p = np);
            }
            black_box(&tree);
        });
    });
    g.finish();
}

// ------------------------------------------------------------------- knn
fn bench_knn(c: &mut Criterion) {
    let items3 = gen3(N, WORLD, 1);
    let aabb = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let mut tree3 = Tree3::<C3>::new(aabb, IL); for it in &items3 { tree3.insert(*it); }
    let mut octree3 = Octree3::<C3>::new(aabb, IL); for it in &items3 { octree3.insert(*it); }
    let qs: Vec<Point3> = { let mut r = Rng::new(7); (0..256).map(|_| Point3::new(r.range(0.0, WORLD), r.range(0.0, WORLD), r.range(0.0, WORLD))).collect() };

    let mut g = c.benchmark_group("knn_k16_x256");
    for k in [8usize, 16] {
        g.bench_with_input(BenchmarkId::new("tree3", k), &k, |b, &k| b.iter(|| { let mut n = 0; for q in &qs { n += tree3.knn(*q, k).len(); } black_box(n) }));
        g.bench_with_input(BenchmarkId::new("octree3", k), &k, |b, &k| b.iter(|| { let mut n = 0; for q in &qs { n += octree3.knn(*q, k).len(); } black_box(n) }));
    }
    g.finish();
}

// ------------------------------------------------------------------- raycast
fn bench_raycast(c: &mut Criterion) {
    let items3 = gen3(N, WORLD, 1);
    let aabb = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
    let mut tree3 = Tree3::<C3>::new(aabb, IL);
    for it in &items3 { tree3.insert(*it); }
    // 256 random unit-direction rays across the world.
    let rays: Vec<(Point3, Point3)> = {
        let mut r = Rng::new(13);
        (0..256).map(|_| {
            let o = Point3::new(r.range(0.0, WORLD), r.range(0.0, WORLD), r.range(0.0, WORLD));
            let (dx, dy, dz) = (r.range(-1.0, 1.0), r.range(-1.0, 1.0), r.range(-1.0, 1.0));
            let l = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
            (o, Point3::new(dx / l, dy / l, dz / l))
        }).collect()
    };
    let (max_t, radius) = (WORLD, 4.0);
    let mut g = c.benchmark_group("raycast_20k_x256");
    // All-hits thick raycast (the siege ballista — every unit on the line).
    g.bench_function("tree3_all", |b| b.iter(|| { let mut n = 0; for (o, d) in &rays { n += tree3.raycast(*o, *d, max_t, radius).len(); } black_box(n) }));
    // First-hit DDA (the archer / line-of-sight short-circuit).
    g.bench_function("tree3_dda_first", |b| b.iter(|| { let mut n = 0; for (o, d) in &rays { n += tree3.raycast_dda_first(*o, *d, max_t, radius).is_some() as usize; } black_box(n) }));
    g.finish();
}

criterion_group!(benches, bench_build, bench_cull, bench_update, bench_knn, bench_raycast);
criterion_main!(benches);
