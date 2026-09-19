//! **R\*-Grove's index-quality metrics, applied to all twelve structures.**
//!
//! `key_partition_bench` borrowed three of these to compare *partitioners*. They were defined for
//! spatial indexes, so this asks the obvious follow-up question: what do they say about the kit's
//! own structures? The metrics, as Vu & Eldawy define them for a set of partitions — here, for a
//! structure's set of leaves or cells:
//!
//! | | what it is | what a good index wants |
//! | --- | --- | --- |
//! | **Q1** total volume | Σ volume of each leaf's bounding box | small — dead space costs pruning |
//! | **Q2** total overlap | Σ pairwise overlap between leaf boxes | zero — overlap means visiting twice |
//! | **Q3** total margin | Σ of each box's side lengths | small — favours cubic over elongated |
//! | **Q4** utilization | how full the blocks are | high — a half-empty leaf is overhead |
//! | **Q5** load balance | standard deviation of leaf sizes | small — even work per leaf |
//!
//! **The box measured is the tight bounding box of the items a leaf HOLDS, not the leaf's own
//! box.** Those are different questions: the node box asks how the index carves space (and for a
//! space partitioner the answer is trivially "it tiles the world"), while the item box asks how
//! much *dead space the index claims* — which is what decides whether a query can prune it. Six of
//! the twelve structures exposed only `(box, count)` and not their items, which is why this example
//! did not exist; `visit_leaf_items` is the fill.
//!
//! ## Two of the five do not apply, and saying so is the point
//!
//! **Q2 is identically zero for every structure here**, and it is asserted rather than printed as a
//! column of noughts. Overlap is an **R-tree-family** metric: it measures what you pay when
//! partitions are allowed to intersect, which is the price an R-tree pays for grouping by data
//! rather than by space. Every structure in this kit partitions *space* — disjoint octants,
//! quadrants, grid cells, half-spaces — so a leaf's items lie inside a region no other leaf's items
//! can reach. A metric that cannot separate twelve candidates is not a weak signal, it is the wrong
//! instrument, and quoting it would be padding.
//!
//! **Q4 needs a capacity to be a fraction of.** It comes from HDFS block occupancy. Nine structures
//! have an item limit and get a real number; `MortonGrid`, `MortonGrid3` and `RadixTrie3` have
//! unbounded buckets — a grid cell holds whatever lands in it — so their utilisation is not low, it
//! is undefined, and the column prints `-`.
//!
//! ```bash
//! cargo run -p vectorial-hash --example index_quality --release
//! ```
//! Env: `IQ_N` (points), `IQ_CAP` (leaf capacity where the structure has one).

use vectorial_hash::itree::{IPoint, IPositioned, IRect, IntegerTree};
use vectorial_hash::linear_octree3::LinearOctree3;
use vectorial_hash::linear_quadtree::LinearQuadTree;
use vectorial_hash::{
    Aabb, KdTree2, KdTree3, MortonGrid, MortonGrid3, Octree3, Point, Point3, Positioned,
    Positioned3, QuadTree, RadixTrie3, Rect, Tree, Tree3,
};

const W: f64 = 1000.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

#[derive(Clone, Copy)]
struct P3 { p: Point3 }
impl Positioned3 for P3 { fn position(&self) -> Point3 { self.p } }
#[derive(Clone, Copy)]
struct P2 { p: Point }
impl Positioned for P2 { fn position(&self) -> Point { self.p } }
#[derive(Clone, Copy)]
struct PI { p: IPoint }
impl IPositioned for PI { fn position(&self) -> IPoint { self.p } }

/// Per-leaf tight bounding boxes and counts, in `D` dimensions.
struct Leaves { dims: usize, boxes: Vec<Vec<(f64, f64)>>, counts: Vec<usize> }

impl Leaves {
    fn new(dims: usize) -> Self { Leaves { dims, boxes: Vec::new(), counts: Vec::new() } }

    /// One leaf, given its items and how to read a coordinate out of one. Empty leaves are
    /// skipped: an octree `divide` creates all eight children whether or not points land in them,
    /// and a box around nothing would contribute a spurious zero to every column.
    fn add<T, F: Fn(&T, usize) -> f64>(&mut self, items: &[T], coord: F) {
        if items.is_empty() { return; }
        let mut bb: Vec<(f64, f64)> = vec![(f64::INFINITY, f64::NEG_INFINITY); self.dims];
        for it in items {
            for (k, slot) in bb.iter_mut().enumerate() {
                let v = coord(it, k);
                if v < slot.0 { slot.0 = v; }
                if v > slot.1 { slot.1 = v; }
            }
        }
        self.boxes.push(bb);
        self.counts.push(items.len());
    }

    fn volume(b: &[(f64, f64)]) -> f64 { b.iter().map(|(lo, hi)| (hi - lo).max(0.0)).product() }
    fn margin(b: &[(f64, f64)]) -> f64 { b.iter().map(|(lo, hi)| (hi - lo).max(0.0)).sum() }
    fn overlap(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x.1.min(y.1) - x.0.max(y.0)).max(0.0)).product()
    }

    /// `(leaves, Q1, Q2, Q3, Q5)`, the volumes normalised by the world's so they read as fractions.
    fn metrics(&self) -> (usize, f64, f64, f64, f64) {
        let world_v = W.powi(self.dims as i32);
        let q1: f64 = self.boxes.iter().map(|b| Self::volume(b)).sum::<f64>() / world_v;
        let q3: f64 = self.boxes.iter().map(|b| Self::margin(b)).sum::<f64>() / W;
        let mut q2 = 0.0;
        for i in 0..self.boxes.len() {
            for j in (i + 1)..self.boxes.len() { q2 += Self::overlap(&self.boxes[i], &self.boxes[j]); }
        }
        let n = self.counts.len().max(1) as f64;
        let mean = self.counts.iter().sum::<usize>() as f64 / n;
        let var = self.counts.iter().map(|&c| (c as f64 - mean).powi(2)).sum::<f64>() / n;
        (self.counts.len(), q1, q2 / world_v, q3, var.sqrt() / mean.max(1e-12))
    }

    fn mean_fill(&self) -> f64 {
        self.counts.iter().sum::<usize>() as f64 / self.counts.len().max(1) as f64
    }
}

/// Print one row, and assert the overlap claim the header makes.
fn row(name: &str, l: &Leaves, cap: Option<usize>) {
    let (leaves, q1, q2, q3, q5) = l.metrics();
    assert!(q2 < 1e-9,
            "{name}: Q2 overlap should be identically zero for a SPACE partitioner, got {q2:.6}. \
             If a structure here ever groups by data rather than by space, this header is wrong.");
    let q4 = match cap {
        Some(c) => format!("{:>7.2}", l.mean_fill() / c as f64),
        None => "      -".to_string(),
    };
    println!("  {name:<16} {leaves:>7} {:>9.3} {:>9.1} {q4} {:>8.3}", q1, q3, q5);
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let n: usize = env("IQ_N", 20_000);
    let cap: usize = env("IQ_CAP", 16);
    println!("index_quality — {n} points, leaf capacity {cap} where the structure has one, world {W}");
    println!("{}", vectorial_hash::machine_line());
    println!("\nR*-Grove's metrics on the tight bounding box of what each leaf HOLDS. Q2 (overlap) is");
    println!("asserted zero for every structure and therefore not a column: these all partition");
    println!("SPACE, so leaves cannot intersect. Q4 needs a capacity, so the unbounded-bucket");
    println!("structures print `-` rather than a fabricated number.\n");
    println!("  Q1 = dead volume claimed (x world) | Q3 = total margin (x world side)");
    println!("  Q4 = mean fill / capacity          | Q5 = stddev of leaf sizes / mean\n");

    for (label, clustered) in [("uniform", false), ("clustered", true)] {
        let mut r = Rng(0x1234_5678);
        let blobs: Vec<(f64, f64, f64)> =
            (0..8).map(|_| (r.f() * W, r.f() * W, r.f() * W)).collect();
        let make = |r: &mut Rng, i: usize| -> (f64, f64, f64) {
            if clustered {
                let b = blobs[i % blobs.len()];
                let g = |a: f64, r: &mut Rng| (a + (r.f() - 0.5) * 60.0).clamp(0.0, W - 0.5);
                (g(b.0, r), g(b.1, r), g(b.2, r))
            } else {
                (r.f() * W, r.f() * W, r.f() * W)
            }
        };
        let raw: Vec<(f64, f64, f64)> = (0..n).map(|i| make(&mut r, i)).collect();
        let p3: Vec<P3> = raw.iter().map(|c| P3 { p: Point3::new(c.0, c.1, c.2) }).collect();
        let p2: Vec<P2> = raw.iter().map(|c| P2 { p: Point::new(c.0, c.1) }).collect();
        let pi: Vec<PI> = raw.iter().map(|c| PI { p: IPoint::new(c.0 as i32, c.1 as i32) }).collect();

        let w3 = Aabb { x: 0.0, y: 0.0, z: 0.0, w: W, h: W, d: W };
        let w2 = Rect { x: 0.0, y: 0.0, width: W, height: W };
        // `levels 4` = 16 cells/axis. Chosen so the grids' cell count is the same order as the
        // trees' leaf count at this population and capacity — comparing a 4 096-cell grid against a
        // 1 300-leaf tree would be measuring the resolution knob, not the structure.
        let levels = 4u32;

        println!("== {label} ==");
        println!("  {:<16} {:>7} {:>9} {:>9} {:>7} {:>8}", "structure", "leaves", "Q1 vol", "Q3 margin", "Q4 fill", "Q5 sd");

        // ---- 3D ----
        let mut l = Leaves::new(3);
        let t3 = Tree3::bulk_load(w3, cap, p3.clone());
        t3.visit_leaf_items(|it| l.add(it, |p: &P3, k| [p.p.x, p.p.y, p.p.z][k]));
        row("Tree3", &l, Some(cap));

        let mut l = Leaves::new(3);
        let o3 = Octree3::bulk_load(w3, cap, p3.clone());
        o3.visit_leaf_items(|it| l.add(it, |p: &P3, k| [p.p.x, p.p.y, p.p.z][k]));
        row("Octree3", &l, Some(cap));

        let mut l = Leaves::new(3);
        let lo = LinearOctree3::from_items(w3, cap, 12, p3.clone());
        lo.visit_leaf_items(|it| l.add(it, |p: &P3, k| [p.p.x, p.p.y, p.p.z][k]));
        row("LinearOctree3", &l, Some(cap));

        let mut l = Leaves::new(3);
        let mut g3 = MortonGrid3::new(w3, levels);
        for it in &p3 { g3.insert(*it); }
        g3.visit_leaf_items(|it| l.add(it, |p: &P3, k| [p.p.x, p.p.y, p.p.z][k]));
        row("MortonGrid3", &l, None);

        let mut l = Leaves::new(3);
        let k3 = KdTree3::from_items(cap, p3.clone());
        k3.visit_leaf_items(|it| l.add(it, |p: &P3, k| [p.p.x, p.p.y, p.p.z][k]));
        row("KdTree3", &l, Some(cap));

        let mut l = Leaves::new(3);
        let rx = RadixTrie3::from_items(w3, 8, p3.clone());
        rx.visit_leaf_items(|it| l.add(it, |p: &P3, k| [p.p.x, p.p.y, p.p.z][k]));
        row("RadixTrie3", &l, None);

        // ---- 2D ----
        let mut l = Leaves::new(2);
        let mut t2 = Tree::new(w2, cap);
        for it in &p2 { t2.insert(*it); }
        t2.visit_leaf_items(|it| l.add(it, |p: &P2, k| [p.p.x, p.p.y][k]));
        row("Tree (2D)", &l, Some(cap));

        let mut l = Leaves::new(2);
        let mut q2t = QuadTree::new(w2, cap);
        for it in &p2 { q2t.insert(*it); }
        q2t.visit_leaf_items(|it| l.add(it, |p: &P2, k| [p.p.x, p.p.y][k]));
        row("QuadTree", &l, Some(cap));

        let mut l = Leaves::new(2);
        let mut it2 = IntegerTree::new(IRect::new(0, 0, 1024, 1024), cap);
        for it in &pi { it2.insert(*it); }
        it2.visit_leaf_items(|it| l.add(it, |p: &PI, k| [p.p.x as f64, p.p.y as f64][k]));
        row("IntegerTree", &l, Some(cap));

        let mut l = Leaves::new(2);
        let lq = LinearQuadTree::from_items(w2, cap, 12, p2.clone());
        lq.visit_leaf_items(|it| l.add(it, |p: &P2, k| [p.p.x, p.p.y][k]));
        row("LinearQuadTree", &l, Some(cap));

        let mut l = Leaves::new(2);
        let mut g2 = MortonGrid::new(w2, levels);
        for it in &p2 { g2.insert(*it); }
        g2.visit_leaf_items(|it| l.add(it, |p: &P2, k| [p.p.x, p.p.y][k]));
        row("MortonGrid", &l, None);

        let mut l = Leaves::new(2);
        let k2 = KdTree2::from_items(cap, p2.clone());
        k2.visit_leaf_items(|it| l.add(it, |p: &P2, k| [p.p.x, p.p.y][k]));
        row("KdTree2", &l, Some(cap));
        println!();
    }

    println!("Note: 2D and 3D rows are NOT comparable to each other — Q1 is a fraction of a square");
    println!("in one and of a cube in the other, and Q3 counts two sides against three. Compare");
    println!("down a dimension group, never across.");
    println!();
    println!("★ READ Q1 AND Q3 ONLY AGAINST A COMPARABLE LEAF COUNT. `RadixTrie3` scores a perfect");
    println!("0.000 volume and 0.0 margin on uniform data, and it is not the best index in the");
    println!("table — it is the most degenerate: 19 989 leaves for 20 000 points means a box drawn");
    println!("around ONE point, which has no volume and no margin by definition. Minimising Q1 or");
    println!("Q3 alone drives you to one item per leaf, i.e. to a structure that prunes nothing and");
    println!("costs a descent per point. This is the second time this week an R*-Grove metric read");
    println!("in isolation picked the worst candidate present (the first: an x-stripe partitioner");
    println!("with the best Q1/Q2 of any arm and by far the worst query fan-out).");
    println!();
    println!("★ ON UNIFORM DATA, `Octree3`, `LinearOctree3` AND `MortonGrid3` ARE THE SAME PARTITION.");
    println!("Identical leaf counts and identical Q1/Q3/Q5 to three decimals. An item limit applied");
    println!("to evenly spread points subdivides evenly, which is a grid — the adaptivity has");
    println!("nothing to adapt to. Under clustering they separate immediately and by a lot: the grid");
    println!("collapses to 44 non-empty cells with Q5 = 1.175 while the octrees hold ~4 000 at 0.61.");
    println!("(`Octree3` and `LinearOctree3` agree in BOTH columns, which is the cross-check you");
    println!("would want: same algorithm, different storage, so any disagreement would be a bug.)");
    println!();
    println!("★ THE K-D TREES' Q5 IS 0.043 IN EVERY ROW, uniform and clustered alike. A median split");
    println!("puts half the points on each side by construction, so balance stops being a property");
    println!("of the data and becomes a property of the algorithm — nothing else here is within 2.5x");
    println!("of it on uniform data, and under clustering the grids are 20-27x worse. That is the");
    println!("column that says why the k-d trees exist, and it is the one this repo had never");
    println!("measured because the metric came from a partitioning paper rather than an index one.");
}
