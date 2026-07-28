//! Property-based fuzzing across every structure.
//!
//! Each test drives a randomized sequence of `insert` / `remove` / `update`
//! operations against both the structure and a plain brute-force model (a `Vec`
//! of the live points), then asserts that `cull` (over several query volumes)
//! and `knn` agree with brute force. proptest **shrinks** any failing op
//! sequence to a minimal reproducer.
//!
//! The trees use the O(1) `ItemRef` handle path (`insert_ref` / `remove_ref` /
//! `update_ref`), so the handle bookkeeping is fuzzed too — after every run
//! `item_count()` must still equal the model's live count. The Morton grids have
//! no handle/remove/update surface, so those are insert-only (cull + knn still
//! fuzzed over the inserted set). knn is compared by **distance** (not identity)
//! so exact ties never spuriously fail.

use proptest::prelude::*;
use vectorial_hash::{
    Aabb, IPoint, IPositioned, IRect, IntegerTree, Octree3, Point, Point3, Positioned,
    Positioned3, QuadTree, Rect, Shape, Shape3, Sphere3, Tree, Tree3, ItemRef, MortonGrid, MortonGrid3,
    KdTree3, LinearOctree3, LinearQuadTree, Polyhedron3,
    AdaptiveIndex, Backend, Slot, Thresholds,
};

const W: f64 = 256.0;

// ---- item wrappers ----
#[derive(Clone, Copy, Debug)]
struct M2 { p: Point }
impl Positioned for M2 { fn position(&self) -> Point { self.p } }
#[derive(Clone, Copy, Debug)]
struct M3 { p: Point3 }
impl Positioned3 for M3 { fn position(&self) -> Point3 { self.p } }
#[derive(Clone, Copy, Debug)]
struct MI { p: IPoint }
impl IPositioned for MI { fn position(&self) -> IPoint { self.p } }

// ---- a query disc/sphere ----
struct Disc { cx: f64, cy: f64, r: f64 }
impl Shape for Disc {
    fn bounding_box(&self) -> Rect { Rect::new(self.cx - self.r, self.cy - self.r, 2.0 * self.r, 2.0 * self.r) }
    fn contains_point(&self, p: Point) -> bool { let (dx, dy) = (p.x - self.cx, p.y - self.cy); dx * dx + dy * dy <= self.r * self.r }
}

// ---- op alphabet ----
#[derive(Clone, Debug)]
enum Op { Ins(f64, f64, f64), Rem(usize), Upd(usize, f64, f64, f64) }

fn ops() -> impl Strategy<Value = Vec<Op>> {
    let op = prop_oneof![
        3 => (0.0..W, 0.0..W, 0.0..W).prop_map(|(x, y, z)| Op::Ins(x, y, z)),
        1 => any::<usize>().prop_map(Op::Rem),
        2 => (any::<usize>(), 0.0..W, 0.0..W, 0.0..W).prop_map(|(i, x, y, z)| Op::Upd(i, x, y, z)),
    ];
    prop::collection::vec(op, 0..200)
}

// Query volumes / knn probes reused by every test.
const DISCS: [(f64, f64, f64); 3] = [(128.0, 128.0, 40.0), (60.0, 200.0, 55.0), (20.0, 20.0, 90.0)];
const SPHERES: [(f64, f64, f64, f64); 3] = [(128.0, 128.0, 128.0, 40.0), (60.0, 200.0, 90.0, 55.0), (20.0, 20.0, 20.0, 90.0)];
const KS: [usize; 3] = [1, 5, 12];

// ============================ 3D binary / octree ============================
macro_rules! proptest_tree3 {
    ($name:ident, $ty:ident) => {
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(32))]
            #[test]
            fn $name(ops in ops()) {
                let mut t = $ty::<M3>::new(Aabb::new(0.0, 0.0, 0.0, W, W, W), 6);
                let mut live: Vec<(ItemRef, Point3)> = Vec::new();
                for op in &ops {
                    match *op {
                        Op::Ins(x, y, z) => { let p = Point3::new(x, y, z); if let Some(r) = t.insert_ref(M3 { p }) { live.push((r, p)); } }
                        Op::Rem(i) => { if !live.is_empty() { let j = i % live.len(); let (r, _) = live.swap_remove(j); t.remove_ref(r); } }
                        Op::Upd(i, x, y, z) => { if !live.is_empty() { let j = i % live.len(); let np = Point3::new(x, y, z); let (r, _) = live[j]; if t.update_ref(r, |m| m.p = np) { live[j].1 = np; } else { live.swap_remove(j); } } }
                    }
                }
                prop_assert_eq!(t.item_count(), live.len(), "item_count drifted from model");
                for (cx, cy, cz, rr) in SPHERES {
                    let s = Sphere3::new(cx, cy, cz, rr).with_raster();
                    let mut want: Vec<(u64, u64, u64)> = live.iter().filter(|(_, p)| { let (dx, dy, dz) = (p.x - cx, p.y - cy, p.z - cz); dx * dx + dy * dy + dz * dz <= rr * rr }).map(|(_, p)| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect();
                    let mut got: Vec<(u64, u64, u64)> = t.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect();
                    want.sort(); got.sort();
                    prop_assert_eq!(want, got, "cull != brute for sphere ({},{},{}) r={}", cx, cy, cz, rr);
                }
                for q in [Point3::new(120.0, 120.0, 120.0), Point3::new(30.0, 200.0, 60.0)] {
                    for k in KS {
                        let mut brute: Vec<f64> = live.iter().map(|(_, p)| { let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z); dx * dx + dy * dy + dz * dz }).collect();
                        brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
                        let got: Vec<f64> = t.knn(q, k).iter().map(|(d, _)| d * d).collect();
                        prop_assert_eq!(got.len(), brute.len(), "knn count k={}", k);
                        for (a, b) in got.iter().zip(brute.iter()) { prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "knn dist {} != brute {}", a, b); }
                    }
                }
            }
        }
    };
}
proptest_tree3!(tree3_ops_match_brute, Tree3);
proptest_tree3!(octree3_ops_match_brute, Octree3);

// ============================ 2D binary / quad ============================
macro_rules! proptest_tree2 {
    ($name:ident, $ty:ident) => {
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(32))]
            #[test]
            fn $name(ops in ops()) {
                let mut t = $ty::<M2>::new(Rect::new(0.0, 0.0, W, W), 5);
                let mut live: Vec<(ItemRef, Point)> = Vec::new();
                for op in &ops {
                    match *op {
                        Op::Ins(x, y, _) => { let p = Point::new(x, y); if let Some(r) = t.insert_ref(M2 { p }) { live.push((r, p)); } }
                        Op::Rem(i) => { if !live.is_empty() { let j = i % live.len(); let (r, _) = live.swap_remove(j); t.remove_ref(r); } }
                        Op::Upd(i, x, y, _) => { if !live.is_empty() { let j = i % live.len(); let np = Point::new(x, y); let (r, _) = live[j]; if t.update_ref(r, |m| m.p = np) { live[j].1 = np; } else { live.swap_remove(j); } } }
                    }
                }
                prop_assert_eq!(t.item_count(), live.len(), "item_count drifted from model");
                for (cx, cy, rr) in DISCS {
                    let s = Disc { cx, cy, r: rr };
                    let mut want: Vec<(u64, u64)> = live.iter().filter(|(_, p)| { let (dx, dy) = (p.x - cx, p.y - cy); dx * dx + dy * dy <= rr * rr }).map(|(_, p)| (p.x.to_bits(), p.y.to_bits())).collect();
                    let mut got: Vec<(u64, u64)> = t.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect();
                    want.sort(); got.sort();
                    prop_assert_eq!(want, got, "cull != brute for disc ({},{}) r={}", cx, cy, rr);
                }
                for q in [Point::new(120.0, 120.0), Point::new(30.0, 200.0)] {
                    for k in KS {
                        let mut brute: Vec<f64> = live.iter().map(|(_, p)| { let (dx, dy) = (p.x - q.x, p.y - q.y); dx * dx + dy * dy }).collect();
                        brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
                        let got: Vec<f64> = t.knn(q, k).iter().map(|(d, _)| d * d).collect();
                        prop_assert_eq!(got.len(), brute.len(), "knn count k={}", k);
                        for (a, b) in got.iter().zip(brute.iter()) { prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "knn dist {} != brute {}", a, b); }
                    }
                }
            }
        }
    };
}
proptest_tree2!(tree_ops_match_brute, Tree);
proptest_tree2!(quadtree_ops_match_brute, QuadTree);

// ============================ integer tree ============================
proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn itree_ops_match_brute(ops in ops()) {
        let mut t = IntegerTree::<MI>::new(IRect::new(0, 0, 256, 256), 5);
        let mut live: Vec<(ItemRef, IPoint)> = Vec::new();
        let q = |v: f64| (v as i32).clamp(0, 255);
        for op in &ops {
            match *op {
                Op::Ins(x, y, _) => { let p = IPoint::new(q(x), q(y)); if let Some(r) = t.insert_ref(MI { p }) { live.push((r, p)); } }
                Op::Rem(i) => { if !live.is_empty() { let j = i % live.len(); let (r, _) = live.swap_remove(j); t.remove_ref(r); } }
                Op::Upd(i, x, y, _) => { if !live.is_empty() { let j = i % live.len(); let np = IPoint::new(q(x), q(y)); let (r, _) = live[j]; if t.update_ref(r, |m| m.p = np) { live[j].1 = np; } else { live.swap_remove(j); } } }
            }
        }
        prop_assert_eq!(t.item_count(), live.len(), "item_count drifted from model");
        for (cx, cy, rr) in DISCS {
            let s = Disc { cx, cy, r: rr };
            let mut want: Vec<(i32, i32)> = live.iter().filter(|(_, p)| { let (dx, dy) = (p.x as f64 - cx, p.y as f64 - cy); dx * dx + dy * dy <= rr * rr }).map(|(_, p)| (p.x, p.y)).collect();
            let mut got: Vec<(i32, i32)> = t.cull(&s).iter().map(|m| (m.p.x, m.p.y)).collect();
            want.sort(); got.sort();
            prop_assert_eq!(want, got, "cull != brute for disc ({},{}) r={}", cx, cy, rr);
        }
        for qp in [IPoint::new(120, 120), IPoint::new(30, 200)] {
            for k in KS {
                let mut brute: Vec<f64> = live.iter().map(|(_, p)| { let (dx, dy) = (p.x as f64 - qp.x as f64, p.y as f64 - qp.y as f64); dx * dx + dy * dy }).collect();
                brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
                let got: Vec<f64> = t.knn(qp, k).iter().map(|(d, _)| d * d).collect();
                prop_assert_eq!(got.len(), brute.len(), "knn count k={}", k);
                for (a, b) in got.iter().zip(brute.iter()) { prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "knn dist {} != brute {}", a, b); }
            }
        }
    }
}

// ============================ Morton grids (insert-only) ============================
fn pts() -> impl Strategy<Value = Vec<(f64, f64, f64)>> {
    prop::collection::vec((0.0..W, 0.0..W, 0.0..W), 0..350)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn morton3_cull_knn_match_brute(pts in pts()) {
        let mut g = MortonGrid3::<M3>::new(Aabb::new(0.0, 0.0, 0.0, W, W, W), 5);
        let mut live: Vec<Point3> = Vec::new();
        for (x, y, z) in &pts { let p = Point3::new(*x, *y, *z); if g.insert(M3 { p }) { live.push(p); } }
        prop_assert_eq!(g.item_count(), live.len());
        for (cx, cy, cz, rr) in SPHERES {
            let s = Sphere3::new(cx, cy, cz, rr).with_raster();
            let mut want: Vec<(u64, u64, u64)> = live.iter().filter(|p| { let (dx, dy, dz) = (p.x - cx, p.y - cy, p.z - cz); dx * dx + dy * dy + dz * dz <= rr * rr }).map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect();
            let mut got: Vec<(u64, u64, u64)> = g.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect();
            let mut lay: Vec<(u64, u64, u64)> = g.cull_layered(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect();
            want.sort(); got.sort(); lay.sort();
            prop_assert_eq!(&want, &got, "morton3 cull != brute ({},{},{}) r={}", cx, cy, cz, rr);
            prop_assert_eq!(&got, &lay, "morton3 cull_layered != cull ({},{},{}) r={}", cx, cy, cz, rr);
        }
        for qp in [Point3::new(120.0, 120.0, 120.0), Point3::new(30.0, 200.0, 60.0)] {
            for k in KS {
                let mut brute: Vec<f64> = live.iter().map(|p| { let (dx, dy, dz) = (p.x - qp.x, p.y - qp.y, p.z - qp.z); dx * dx + dy * dy + dz * dz }).collect();
                brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
                let got: Vec<f64> = g.knn(qp, k).iter().map(|(d, _)| d * d).collect();
                prop_assert_eq!(got.len(), brute.len(), "knn count k={}", k);
                for (a, b) in got.iter().zip(brute.iter()) { prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "knn dist {} != brute {}", a, b); }
            }
        }
    }

    #[test]
    fn morton2_cull_knn_match_brute(pts in pts()) {
        let mut g = MortonGrid::<M2>::new(Rect::new(0.0, 0.0, W, W), 5);
        let mut live: Vec<Point> = Vec::new();
        for (x, y, _) in &pts { let p = Point::new(*x, *y); if g.insert(M2 { p }) { live.push(p); } }
        prop_assert_eq!(g.item_count(), live.len());
        for (cx, cy, rr) in DISCS {
            let s = Disc { cx, cy, r: rr };
            let mut want: Vec<(u64, u64)> = live.iter().filter(|p| { let (dx, dy) = (p.x - cx, p.y - cy); dx * dx + dy * dy <= rr * rr }).map(|p| (p.x.to_bits(), p.y.to_bits())).collect();
            let mut got: Vec<(u64, u64)> = g.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect();
            let mut lay: Vec<(u64, u64)> = g.cull_layered(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect();
            want.sort(); got.sort(); lay.sort();
            prop_assert_eq!(&want, &got, "morton2 cull != brute ({},{}) r={}", cx, cy, rr);
            prop_assert_eq!(&got, &lay, "morton2 cull_layered != cull ({},{}) r={}", cx, cy, rr);
        }
        for qp in [Point::new(120.0, 120.0), Point::new(30.0, 200.0)] {
            for k in KS {
                let mut brute: Vec<f64> = live.iter().map(|p| { let (dx, dy) = (p.x - qp.x, p.y - qp.y); dx * dx + dy * dy }).collect();
                brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
                let got: Vec<f64> = g.knn(qp, k).iter().map(|(d, _)| d * d).collect();
                prop_assert_eq!(got.len(), brute.len(), "knn count k={}", k);
                for (a, b) in got.iter().zip(brute.iter()) { prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "knn dist {} != brute {}", a, b); }
            }
        }
    }
}

// ============ build-once structures: KdTree3 / LinearOctree3 / LinearQuadTree ============
// These three have no handle/remove surface, so — like the Morton grids — they're fuzzed
// over a random point set. Beyond cull/knn-vs-brute they get two properties the trees
// can't have: the incremental `insert` path must land in the same place as the bulk
// `from_items` build, and (3D) a **frustum** cull must agree with brute force, which is
// the query verb the stealth demo leans on.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn kdtree3_cull_knn_match_brute(pts in pts(), cap in 1usize..24) {
        let live: Vec<Point3> = pts.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect();
        let t = KdTree3::from_items(cap, live.iter().map(|&p| M3 { p }).collect());
        prop_assert_eq!(t.item_count(), live.len());
        for (cx, cy, cz, rr) in SPHERES {
            let s = Sphere3::new(cx, cy, cz, rr);
            let mut want: Vec<(u64, u64, u64)> = live.iter().filter(|p| { let (dx, dy, dz) = (p.x - cx, p.y - cy, p.z - cz); dx * dx + dy * dy + dz * dz <= rr * rr }).map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect();
            let mut got: Vec<(u64, u64, u64)> = t.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect();
            want.sort(); got.sort();
            prop_assert_eq!(want, got, "kdtree3 cull != brute ({},{},{}) r={}", cx, cy, cz, rr);
        }
        for k in KS {
            let qp = Point3::new(120.0, 120.0, 120.0);
            let mut brute: Vec<f64> = live.iter().map(|p| { let (dx, dy, dz) = (p.x - qp.x, p.y - qp.y, p.z - qp.z); dx * dx + dy * dy + dz * dz }).collect();
            brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
            let got: Vec<f64> = t.knn(qp, k).iter().map(|(d, _)| d * d).collect();
            prop_assert_eq!(got.len(), brute.len(), "kdtree3 knn count k={}", k);
            for (a, b) in got.iter().zip(brute.iter()) { prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "kdtree3 knn dist {} != brute {}", a, b); }
        }
    }

    /// A frustum (six half-spaces) over the adaptive 3D structures — the verb the stealth
    /// demo's view cones use, fuzzed against `contains_point` on every item.
    #[test]
    fn frustum_cull_matches_brute(pts in pts(), fx in 0.0..W, fz in 0.0..W, ang in -3.2..3.2f64) {
        let live: Vec<Point3> = pts.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect();
        let (s, c) = (ang.sin(), ang.cos());
        let quad = |dist: f64, half: f64, vh: f64| {
            let ctr = (fx + c * dist, 110.0, fz + s * dist);
            [Point3::new(ctr.0 + s * half, ctr.1 - vh, ctr.2 - c * half), Point3::new(ctr.0 - s * half, ctr.1 - vh, ctr.2 + c * half),
             Point3::new(ctr.0 - s * half, ctr.1 + vh, ctr.2 + c * half), Point3::new(ctr.0 + s * half, ctr.1 + vh, ctr.2 - c * half)]
        };
        let (n, f) = (quad(4.0, 8.0, 12.0), quad(210.0, 110.0, 95.0));
        let cone = Polyhedron3::from_corners([n[0], n[1], n[2], n[3], f[0], f[1], f[2], f[3]]);
        let mut want: Vec<(u64, u64, u64)> = live.iter().filter(|p| cone.contains_point(**p)).map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect();
        want.sort();
        let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
        let kd = KdTree3::from_items(8, live.iter().map(|&p| M3 { p }).collect());
        let lo = LinearOctree3::from_items(world, 8, 6, live.iter().map(|&p| M3 { p }).collect());
        let t3 = Tree3::bulk_load(world, 8, live.iter().map(|&p| M3 { p }).collect());
        for (label, mut got) in [
            ("kdtree3", kd.cull(&cone).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect::<Vec<_>>()),
            ("linear_octree3", lo.cull(&cone).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect::<Vec<_>>()),
            ("tree3", t3.cull(&cone).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect::<Vec<_>>()),
        ] {
            got.sort();
            prop_assert_eq!(&want, &got, "{} frustum cull != brute at ({}, {}) ang {}", label, fx, fz, ang);
        }
    }

    #[test]
    fn linear_octree3_cull_knn_match_brute(pts in pts(), depth in 2u8..8) {
        let live: Vec<Point3> = pts.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect();
        let world = Aabb::new(0.0, 0.0, 0.0, W, W, W);
        let t = LinearOctree3::from_items(world, 8, depth, live.iter().map(|&p| M3 { p }).collect());
        prop_assert_eq!(t.item_count(), live.len());
        // the incremental path must land in the same place as the bulk build
        let mut inc = LinearOctree3::<M3>::new(world, 8, depth);
        for &p in &live { inc.insert(M3 { p }); }
        for (cx, cy, cz, rr) in SPHERES {
            let s = Sphere3::new(cx, cy, cz, rr);
            let mut want: Vec<(u64, u64, u64)> = live.iter().filter(|p| { let (dx, dy, dz) = (p.x - cx, p.y - cy, p.z - cz); dx * dx + dy * dy + dz * dz <= rr * rr }).map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect();
            let mut got: Vec<(u64, u64, u64)> = t.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect();
            let mut gi: Vec<(u64, u64, u64)> = inc.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect();
            want.sort(); got.sort(); gi.sort();
            prop_assert_eq!(&want, &got, "linear_octree3 cull != brute ({},{},{}) r={}", cx, cy, cz, rr);
            prop_assert_eq!(&got, &gi, "linear_octree3 insert path != from_items");
        }
        for k in KS {
            let qp = Point3::new(30.0, 200.0, 60.0);
            let mut brute: Vec<f64> = live.iter().map(|p| { let (dx, dy, dz) = (p.x - qp.x, p.y - qp.y, p.z - qp.z); dx * dx + dy * dy + dz * dz }).collect();
            brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
            let got: Vec<f64> = t.knn(qp, k).iter().map(|(d, _)| d * d).collect();
            prop_assert_eq!(got.len(), brute.len(), "linear_octree3 knn count k={}", k);
            for (a, b) in got.iter().zip(brute.iter()) { prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "linear_octree3 knn dist {} != brute {}", a, b); }
        }
    }

    #[test]
    fn linear_quadtree_cull_knn_match_brute(pts in pts(), depth in 2u8..9) {
        let live: Vec<Point> = pts.iter().map(|&(x, y, _)| Point::new(x, y)).collect();
        let world = Rect::new(0.0, 0.0, W, W);
        let t = LinearQuadTree::from_items(world, 8, depth, live.iter().map(|&p| M2 { p }).collect());
        prop_assert_eq!(t.item_count(), live.len());
        let mut inc = LinearQuadTree::<M2>::new(world, 8, depth);
        for &p in &live { inc.insert(M2 { p }); }
        for (cx, cy, rr) in DISCS {
            let s = Disc { cx, cy, r: rr };
            let mut want: Vec<(u64, u64)> = live.iter().filter(|p| { let (dx, dy) = (p.x - cx, p.y - cy); dx * dx + dy * dy <= rr * rr }).map(|p| (p.x.to_bits(), p.y.to_bits())).collect();
            let mut got: Vec<(u64, u64)> = t.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect();
            let mut gi: Vec<(u64, u64)> = inc.cull(&s).iter().map(|m| (m.p.x.to_bits(), m.p.y.to_bits())).collect();
            want.sort(); got.sort(); gi.sort();
            prop_assert_eq!(&want, &got, "linear_quadtree cull != brute ({},{}) r={}", cx, cy, rr);
            prop_assert_eq!(&got, &gi, "linear_quadtree insert path != from_items");
        }
        for k in KS {
            let qp = Point::new(120.0, 120.0);
            let mut brute: Vec<f64> = live.iter().map(|p| { let (dx, dy) = (p.x - qp.x, p.y - qp.y); dx * dx + dy * dy }).collect();
            brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
            let got: Vec<f64> = t.knn(qp, k).iter().map(|(d, _)| d * d).collect();
            prop_assert_eq!(got.len(), brute.len(), "linear_quadtree knn count k={}", k);
            for (a, b) in got.iter().zip(brute.iter()) { prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "linear_quadtree knn dist {} != brute {}", a, b); }
        }
    }
}

/// Cull against every probe sphere and compare with brute force over the model. Shared so
/// the adaptive property can re-check after each forced migration without repeating itself.
fn check_cull_matches(ix: &mut AdaptiveIndex<M3>, live: &[Point3]) -> Result<(), TestCaseError> {
    for (cx, cy, cz, rr) in SPHERES {
        let s = Sphere3::new(cx, cy, cz, rr);
        let mut want: Vec<(u64, u64, u64)> = live.iter()
            .filter(|p| { let (dx, dy, dz) = (p.x - cx, p.y - cy, p.z - cz); dx * dx + dy * dy + dz * dz <= rr * rr })
            .map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect();
        let mut got: Vec<(u64, u64, u64)> = ix.cull(&s).iter()
            .map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect();
        want.sort(); got.sort();
        prop_assert_eq!(&want, &got, "cull != brute on backend {:?}", ix.backend());
    }
    Ok(())
}

// ============================ AdaptiveIndex (migrations included) ============================
// The point of this structure is that it CHANGES underneath the caller, so the property is
// not "does this tree answer correctly" but "does the answer stay correct across a change
// of backend the test did not choose and cannot see". Everything is checked against the
// same brute-force model regardless of which structure the policy happens to be holding.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn adaptive_answers_match_brute_across_migrations(ops in ops()) {
        // Thresholds tuned so migrations actually happen inside a short op sequence: the
        // shipped defaults need hundreds of ticks to move, which would make this test
        // exercise one backend and prove nothing.
        let th = Thresholds { brute_max: 40, static_ticks: 6, hold_ticks: 2, cooldown: 0, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(Aabb::new(0.0, 0.0, 0.0, W, W, W), 6, th);
        let mut live: Vec<Point3> = Vec::new();
        let mut backends: Vec<Backend> = vec![ix.backend()];

        for (n, op) in ops.iter().enumerate() {
            match *op {
                Op::Ins(x, y, z) => { let p = Point3::new(x, y, z); ix.insert(M3 { p }); live.push(p); }
                Op::Rem(i) => {
                    // No remove on AdaptiveIndex yet: spend the op on a query instead, which
                    // keeps the op mix (and the query-per-item rate the policy reads) honest.
                    if !live.is_empty() {
                        let q = live[i % live.len()];
                        let _ = ix.cull(&Sphere3::new(q.x, q.y, q.z, 20.0));
                    }
                }
                Op::Upd(i, x, y, z) => {
                    if !live.is_empty() {
                        let j = i % live.len();
                        let np = Point3::new(x, y, z);
                        ix.update(Slot(j as u32), |m| m.p = np);
                        live[j] = np;
                    }
                }
            }
            if n % 3 == 0 { ix.tick(); backends.push(ix.backend()); }

            // The invariant that must hold after EVERY op, whatever backend is loaded.
            prop_assert_eq!(ix.len(), live.len(), "item count drifted on backend {:?}", ix.backend());
        }

        for (cx, cy, cz, rr) in SPHERES {
            let s = Sphere3::new(cx, cy, cz, rr);
            let mut want: Vec<(u64, u64, u64)> = live.iter()
                .filter(|p| { let (dx, dy, dz) = (p.x - cx, p.y - cy, p.z - cz); dx * dx + dy * dy + dz * dz <= rr * rr })
                .map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect();
            let mut got: Vec<(u64, u64, u64)> = ix.cull(&s).iter()
                .map(|m| (m.p.x.to_bits(), m.p.y.to_bits(), m.p.z.to_bits())).collect();
            want.sort(); got.sort();
            prop_assert_eq!(&want, &got, "cull != brute on backend {:?} after {} switches", ix.backend(), ix.switch_count());
        }

        // The random op mix only ever reaches Brute and KeepTree — instrumented, 12 of 24
        // cases migrated and every one of them stopped there. So the other two backends are
        // driven deliberately, and asserted, rather than hoped for: a property that never
        // executes half the code under test is a property about the other half.
        // Above the WIDENED brute edge: brute_max 40 with margin 0.25 keeps the policy on
        // the scan up to 50 items, so a guard of >40 leaves it there and the phase never
        // fires. The hysteresis is behaving; the guard was wrong.
        if live.len() > 80 {
            // Query-heavy: one cull per item per tick is the regime where rebuilding wins.
            for _ in 0..40 {
                for p in live.iter().take(60) { let _ = ix.cull(&Sphere3::new(p.x, p.y, p.z, 8.0)); }
                // one tiny nudge so the tick sees movement (a static workload would take the
                // Static branch instead, which is the phase after this one)
                ix.update(Slot(0), |m| m.p = Point3::new(m.p.x + 0.001, m.p.y, m.p.z));
                ix.tick();
            }
            prop_assert_eq!(ix.backend(), Backend::Grid, "query-heavy phase did not reach the grid");
            check_cull_matches(&mut ix, &live)?;

            // Then everything settles: no movement at all.
            for _ in 0..40 { let _ = ix.cull(&Sphere3::new(60.0, 60.0, 60.0, 30.0)); ix.tick(); }
            prop_assert_eq!(ix.backend(), Backend::Static, "settled phase did not reach the build-once backend");
            check_cull_matches(&mut ix, &live)?;
        }

        for k in KS {
            let q = Point3::new(120.0, 120.0, 120.0);
            let mut brute: Vec<f64> = live.iter()
                .map(|p| { let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z); dx * dx + dy * dy + dz * dz }).collect();
            brute.sort_by(|a, b| a.total_cmp(b)); brute.truncate(k);
            let got: Vec<f64> = ix.knn(q, k).iter().map(|(d, _)| d * d).collect();
            prop_assert_eq!(got.len(), brute.len(), "knn count k={} on backend {:?}", k, ix.backend());
            for (a, b) in got.iter().zip(brute.iter()) {
                prop_assert!((a - b).abs() <= 1e-6 * (1.0 + b), "knn dist {} != brute {}", a, b);
            }
        }
    }

    /// The guarantee hysteresis exists to give: a workload that is not changing must not
    /// keep paying for migrations. Without the hold and the cooldown this flaps every tick
    /// and loses to both candidates.
    #[test]
    fn adaptive_does_not_migrate_under_a_stationary_workload(seed in 0u64..64, pop in 60usize..300) {
        let th = Thresholds { brute_max: 40, hold_ticks: 4, cooldown: 20, ..Default::default() };
        let mut ix = AdaptiveIndex::with_thresholds(Aabb::new(0.0, 0.0, 0.0, W, W, W), 6, th);
        let mut x = seed | 1;
        let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x >> 40) as f64) / (1u64 << 24) as f64 * W };
        for _ in 0..pop { ix.insert(M3 { p: Point3::new(rnd(), rnd(), rnd()) }); }
        // Steady state: same small movement, same query load, every tick.
        for t in 0..200 {
            ix.update(Slot((t % pop) as u32), |m| m.p = Point3::new(m.p.x, m.p.y, (m.p.z + 0.01).min(W)));
            let _ = ix.cull(&Sphere3::new(50.0, 50.0, 50.0, 25.0));
            ix.tick();
        }
        prop_assert!(ix.switch_count() <= 2,
            "stationary workload migrated {} times (pop {})", ix.switch_count(), pop);
    }
}
