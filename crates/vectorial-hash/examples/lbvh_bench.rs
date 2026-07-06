//! LBVH (Linear BVH) prototype + measurement — a BVH built from **Morton codes**
//! (the codes the kit already computes). Sort objects by Z-order, build a binary
//! radix tree over the sorted codes (Karras-style split at the highest differing
//! bit), refit AABBs bottom-up. This is the CPU/serial form; the payoff of the
//! LINEAR construction is that every step is data-parallel → the natural GPU
//! broad-phase (sort + parallel radix-tree build), reusing our Morton encoding.
//!
//! Measures build + sphere-cull vs `Tree3` (bulk_load) at scale, brute-checked.
//!
//! ```bash
//! cargo run -p vectorial-hash --example lbvh_bench --release
//! ```

use std::time::Instant;
use vectorial_hash::{Aabb, Point3, Positioned3, Shape3, Sphere3, Tree3};

const WORLD: f64 = 10_000.0;

fn morton3(x: u32, y: u32, z: u32) -> u64 {
    fn split(mut v: u64) -> u64 { v &= 0x1f_ffff; v = (v | v << 32) & 0x1f00000000ffff; v = (v | v << 16) & 0x1f0000ff0000ff; v = (v | v << 8) & 0x100f00f00f00f00f; v = (v | v << 4) & 0x10c30c30c30c30c3; v = (v | v << 2) & 0x1249249249249249; v }
    split(x as u64) | (split(y as u64) << 1) | (split(z as u64) << 2)
}
fn cell(v: f64) -> u32 { ((v / WORLD) * (1u32 << 21) as f64) as u32 & ((1 << 21) - 1) }

#[derive(Clone, Copy)]
#[allow(dead_code)] // id models an entity handle
struct Obj { id: u32, p: Point3 }
impl Positioned3 for Obj { fn position(&self) -> Point3 { self.p } }

/// Flat BVH node: an AABB (min/max) + either two child node indices (internal)
/// or a point index (leaf, `right == u32::MAX`).
#[derive(Clone, Copy)]
struct BNode { lo: [f64; 3], hi: [f64; 3], left: u32, right: u32 }

struct Lbvh { nodes: Vec<BNode>, root: u32 }

impl Lbvh {
    /// Build from objects: Morton-sort, then a top-down split at the highest
    /// differing bit of the sorted codes (the serial equivalent of Karras'
    /// parallel radix tree; identical topology).
    fn build(objs: &[Obj]) -> Lbvh {
        let mut keyed: Vec<(u64, Point3)> = objs.iter().map(|o| (morton3(cell(o.p.x), cell(o.p.y), cell(o.p.z)), o.p)).collect();
        keyed.sort_unstable_by_key(|k| k.0);
        let mut nodes = Vec::with_capacity(objs.len() * 2);
        let root = Self::build_range(&keyed, 0, keyed.len(), &mut nodes);
        Lbvh { nodes, root }
    }

    fn build_range(k: &[(u64, Point3)], lo: usize, hi: usize, nodes: &mut Vec<BNode>) -> u32 {
        if hi - lo == 1 {
            let p = k[lo].1;
            let id = nodes.len() as u32;
            nodes.push(BNode { lo: [p.x, p.y, p.z], hi: [p.x, p.y, p.z], left: lo as u32, right: u32::MAX });
            return id;
        }
        // split where the top differing bit of code[lo] vs code[hi-1] flips
        let (first, last) = (k[lo].0, k[hi - 1].0);
        let split = if first == last {
            (lo + hi) / 2 // identical codes (dup positions): split down the middle
        } else {
            let common = (first ^ last).leading_zeros();
            let mask = 1u64 << (63 - common);
            // binary search: first index in (lo,hi) whose code has that bit set
            let (mut a, mut b) = (lo, hi - 1);
            while b - a > 1 { let m = (a + b) / 2; if k[m].0 & mask == 0 { a = m; } else { b = m; } }
            b
        };
        let l = Self::build_range(k, lo, split, nodes);
        let r = Self::build_range(k, split, hi, nodes);
        let (ln, rn) = (nodes[l as usize], nodes[r as usize]);
        let id = nodes.len() as u32;
        nodes.push(BNode {
            lo: [ln.lo[0].min(rn.lo[0]), ln.lo[1].min(rn.lo[1]), ln.lo[2].min(rn.lo[2])],
            hi: [ln.hi[0].max(rn.hi[0]), ln.hi[1].max(rn.hi[1]), ln.hi[2].max(rn.hi[2])],
            left: l, right: r,
        });
        id
    }

    /// Sphere cull: descend, pruning subtrees whose AABB misses the sphere.
    fn cull(&self, s: &Sphere3, out: &mut Vec<usize>) {
        let mut stack = vec![self.root];
        let (cx, cy, cz, r) = s.centre_radius();
        let r2 = r * r;
        while let Some(ni) = stack.pop() {
            let n = self.nodes[ni as usize];
            // sphere-vs-AABB: nearest point on the box to the centre
            let nx = cx.clamp(n.lo[0], n.hi[0]); let ny = cy.clamp(n.lo[1], n.hi[1]); let nz = cz.clamp(n.lo[2], n.hi[2]);
            let d2 = (nx - cx).powi(2) + (ny - cy).powi(2) + (nz - cz).powi(2);
            if d2 > r2 { continue; }
            if n.right == u32::MAX { out.push(n.left as usize); } // leaf
            else { stack.push(n.left); stack.push(n.right); }
        }
    }
}

// Sphere3 has no centre accessor in the public API; carry ours.
trait CentreRadius { fn centre_radius(&self) -> (f64, f64, f64, f64); }
impl CentreRadius for Sphere3 { fn centre_radius(&self) -> (f64, f64, f64, f64) { (self.bounding_box().x + self.bounding_box().w / 2.0, self.bounding_box().y + self.bounding_box().h / 2.0, self.bounding_box().z + self.bounding_box().d / 2.0, self.bounding_box().w / 2.0) } }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 { f(); let mut b = f64::MAX; for _ in 0..reps { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); } b }

fn main() {
    println!("LBVH (Morton-built BVH) vs Tree3 | world {WORLD:.0}^3 | bubble r=500\n");
    // correctness on a small set: LBVH cull == brute force.
    {
        let mut r = Rng(1);
        let objs: Vec<Obj> = (0..5000).map(|i| Obj { id: i, p: Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD) }).collect();
        let bvh = Lbvh::build(&objs);
        // map sorted-leaf index back to a position for the check
        let mut keyed: Vec<(u64, Point3)> = objs.iter().map(|o| (morton3(cell(o.p.x), cell(o.p.y), cell(o.p.z)), o.p)).collect();
        keyed.sort_unstable_by_key(|k| k.0);
        for &(cx, cy, cz, rr) in &[(5000.0, 5000.0, 5000.0, 800.0), (1000.0, 2000.0, 9000.0, 1200.0)] {
            let s = Sphere3::new(cx, cy, cz, rr);
            let mut got = Vec::new(); bvh.cull(&s, &mut got);
            let mut gotpos: Vec<(u64, u64, u64)> = got.iter().map(|&li| { let p = keyed[li].1; (p.x.to_bits(), p.y.to_bits(), p.z.to_bits()) }).collect();
            let mut want: Vec<(u64, u64, u64)> = keyed.iter().filter(|(_, p)| s.contains_point(*p)).map(|(_, p)| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect();
            gotpos.sort(); want.sort();
            assert_eq!(gotpos, want, "LBVH cull != brute force");
        }
        println!("LBVH cull == brute force (5k) OK\n");
    }

    println!("{:>9} | {:>20} {:>20} | {:>18} {:>18}", "N", "LBVH build ms", "Tree3 bulk ms", "LBVH cull µs", "Tree3 cull µs");
    for &n in &[100_000usize, 1_000_000] {
        let mut r = Rng(42);
        let objs: Vec<Obj> = (0..n).map(|i| Obj { id: i as u32, p: Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD) }).collect();
        let world = Aabb::new(0.0, 0.0, 0.0, WORLD, WORLD, WORLD);
        let reps = if n >= 1_000_000 { 3 } else { 8 };
        let t_lb = best(reps, || { let b = Lbvh::build(&objs); std::hint::black_box(b.nodes.len()); });
        let t_tb = best(reps, || { let t = Tree3::bulk_load(world, 8, objs.clone()); std::hint::black_box(&t); });
        let bvh = Lbvh::build(&objs);
        let t = Tree3::bulk_load(world, 8, objs.clone());
        let mut rq = Rng(99);
        let qs: Vec<Sphere3> = (0..1000).map(|_| Sphere3::new(rq.unit() * WORLD, rq.unit() * WORLD, rq.unit() * WORLD, 500.0)).collect();
        let mut scratch = Vec::new();
        let t_lc = best(5, || { let mut h = 0; for s in &qs { scratch.clear(); bvh.cull(s, &mut scratch); h += scratch.len(); } std::hint::black_box(h); }) / qs.len() as f64 * 1e6;
        let t_tc = best(5, || { let mut h = 0; for s in &qs { h += t.cull(s).len(); } std::hint::black_box(h); }) / qs.len() as f64 * 1e6;
        println!("{:>9} | {:>17.2}    {:>17.2} | {:>15.2}    {:>15.2}", n, t_lb * 1e3, t_tb * 1e3, t_lc, t_tc);
    }
    println!("\nreading: LBVH build is a Morton sort + a linear tree pass — the point is it's\nfully data-parallel (→ GPU: parallel sort + Karras radix build). On CPU it's a\nfast static build; the query prunes by AABB like any BVH.");
}
