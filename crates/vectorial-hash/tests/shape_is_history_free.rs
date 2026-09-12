//! **When does a maintained structure end up the shape a rebuild would produce?**
//!
//! `examples/restructure_churn` measured this and `docs/CHOOSING.md` states it, but a bench is
//! not a gate: nothing runs it, and the last time this repo let a structural claim live only in
//! prose the claim turned out to be measuring the workload rather than the structure
//! (`MEASURING.md` § 8j). So it is a test, across **every structure that can maintain at all** —
//! nine of the eleven, both dimensions, pointer and keyed.
//!
//! ## The rule, and the two structures that break it
//!
//! A leaf splits when it holds **more** than the limit and merges when it holds **no more** — one
//! threshold, not two — so "subdivided" is equivalent to "holds more than the limit", a statement
//! about the points that are there now rather than about how they arrived. **Provided the split
//! itself is chosen geometrically.** Seven of the nine choose both the split *position* and the
//! split *axis* from the box alone, and they are exactly history-free.
//!
//! `Tree` and `IntegerTree` are **binary**, and for a **square** node they pick the axis by
//! counting which way distributes the items more evenly (`pick_split_by`, and its integer
//! transcription). That count is taken on whatever the node held at the moment it split, so two
//! histories reaching the same point set can disagree about the axis, and the shapes diverge. It
//! is a small effect — a couple of leaves in ~740 — but it is a real one, and it is the same
//! mechanism that stops `KdTree2`/`KdTree3` maintaining at all, in miniature: **a split that asks
//! the data a question remembers the answer.**
//!
//! The seed sweep is not decoration. On the first seed tried, `IntegerTree` drifted and `Tree` did
//! not, which would have read as "the integer tree is the odd one out" — a conclusion about the
//! wrong thing, since the two share the policy verbatim. One seed can only ever tell you that a
//! structure *did* drift, never that it cannot.
//!
//! ## Two guards, because this test could pass vacuously in two different ways
//!
//! A structure nothing moved through would pass trivially, and so would one whose motion never
//! crossed a boundary. Every arm therefore asserts that items really did **cross** leaves, and —
//! under `struct-stats` — that the trees really did **split and merge** while the grids did
//! **neither, exactly**.

use vectorial_hash::itree::{IPoint, IPositioned, IRect, IntegerTree};
use vectorial_hash::linear_octree3::LinearOctree3;
use vectorial_hash::linear_quadtree::LinearQuadTree;
use vectorial_hash::morton3::Crossed;
use vectorial_hash::{
    Aabb, MortonGrid, MortonGrid3, Octree3, Point, Point3, Positioned, Positioned3, QuadTree, Rect,
    Tree, Tree3,
};

const N: usize = 4000;
const FRAMES: usize = 30;
const W: f64 = 256.0;
const CAP: usize = 8;
/// Big enough that a move usually leaves its leaf — the whole point is to disturb the shape.
const STEP: f64 = 24.0;
/// Seeds per arm. Enough that "never drifted" means something; small enough to stay a unit test.
const SEEDS: u64 = 12;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn d(&mut self) -> f64 { (self.f() - 0.5) * 2.0 * STEP }
}

/// Keep a point inside the world by **reflecting** off the wall, not by pinning it to the wall.
///
/// This is load-bearing and was a defect first. `v.clamp(0.5, W - 0.5)` pins every escapee to
/// exactly `0.5` or `W - 0.5`, so in 2D a stream of points piles up on four *exactly coincident*
/// corners — and `divide` refuses to split a node whose items are all at one point while
/// `try_merge_up` will only merge children that between them fit in one leaf. Those are two
/// different predicates, so "became coincident after splitting" is a one-way door and the shape
/// genuinely depends on history. Real, worth knowing, and nothing to do with the property under
/// test — see this file's header. Reflecting removes the manufactured coincidence.
fn clamp(v: f64) -> f64 {
    let (lo, hi) = (0.5, W - 0.5);
    if v < lo { lo + (lo - v).min(hi - lo) } else if v > hi { hi - (v - hi).min(hi - lo) } else { v }
}

/// One `(x, y, z)` per item. The 2D arms read the first two and ignore the third, so every arm at
/// a given seed sees the same motion in the axes it has.
type Triples = Vec<(f64, f64, f64)>;

/// The same motion script for every arm at a given seed, so a failure names the structure rather
/// than the workload, and the arms can be compared to each other.
fn script(seed: u64) -> (Triples, Vec<Triples>) {
    let mut r = Rng(0x5DEECE66D_u64.wrapping_mul(seed.wrapping_add(1)) | 1);
    let start: Triples = (0..N).map(|_| (r.f() * W, r.f() * W, r.f() * W)).collect();
    let frames: Vec<Triples> =
        (0..FRAMES).map(|_| (0..N).map(|_| (r.d(), r.d(), r.d())).collect()).collect();
    (start, frames)
}

#[cfg(feature = "struct-stats")]
fn reset() { let _ = vectorial_hash::restructure::reset(); }
#[cfg(feature = "struct-stats")]
fn taken() -> (u64, u64) { vectorial_hash::restructure::counts() }
#[cfg(not(feature = "struct-stats"))]
fn reset() {}
#[cfg(not(feature = "struct-stats"))]
fn taken() -> (u64, u64) { (0, 0) }

/// The non-vacuity guards: items really moved between leaves, and — under `struct-stats` — the
/// adaptive arms really restructured while the fixed-resolution ones did so **exactly zero** times.
fn guards(arm: &str, restructures: bool, crossings: u64) {
    assert!(crossings > N as u64, "{arm}: only {crossings} boundary crossings in {FRAMES} frames — \
            shapes matching proves nothing if nothing was disturbed");
    let (splits, merges) = taken();
    if !cfg!(feature = "struct-stats") { return; }
    if restructures {
        assert!(splits > 0 && merges > 0,
                "{arm}: expected an adaptive structure to split AND merge, got {splits}/{merges}");
    } else {
        assert_eq!((splits, merges), (0, 0),
                   "{arm}: a fixed-resolution key must never restructure, got {splits} splits and \
                    {merges} merges");
    }
}

/// Every arm reports `(kept, fresh)` per seed; this decides what that means.
fn verdict(arm: &str, shapes: &[(usize, usize)]) {
    let differing: Vec<_> = shapes.iter().enumerate().filter(|(_, (a, b))| a != b).collect();
    let worst = shapes.iter().map(|(a, b)| *a as f64 / *b as f64)
        .fold(1.0_f64, |m, r| if (r - 1.0).abs() > (m - 1.0).abs() { r } else { m });
    println!("{arm:<16} {} / {} seeds drifted, worst {worst:.4}x", differing.len(), shapes.len());
    assert!(differing.is_empty(),
            "{arm}: a maintained structure must have the SAME shape as a rebuild from its own \
             current contents, but {} of {} seeds differ (worst {worst:.4}x); first: {:?}. \
             If this arm's split policy was just made data-dependent, that is the cause — see \
             this file's header.", differing.len(), shapes.len(), differing.first());
}

/// The two binary trees choose a **square** node's split axis from the data, so they are allowed
/// to drift — but only slightly, and the drift must actually be observed somewhere in the sweep.
/// A sweep in which they never drift would mean the policy had been made geometric, and they
/// should then be moved to [`verdict`] rather than left here reading a weaker guarantee.
fn verdict_data_dependent_axis(arm: &str, shapes: &[(usize, usize)]) {
    let differing = shapes.iter().filter(|(a, b)| a != b).count();
    let worst = shapes.iter().map(|(a, b)| *a as f64 / *b as f64)
        .fold(1.0_f64, |m, r| if (r - 1.0).abs() > (m - 1.0).abs() { r } else { m });
    println!("{arm:<16} {differing} / {} seeds drifted, worst {worst:.4}x  (data-dependent axis)",
             shapes.len());
    assert!(differing > 0,
            "{arm} did not drift on any of {} seeds. That is the OPPOSITE of a failure if the \
             square-node axis choice has been made geometric — in which case move this arm onto \
             `verdict` and assert exact equality. It is a failure if the sweep stopped being able \
             to reach the case.", shapes.len());
    assert!((worst - 1.0).abs() < 0.05,
            "{arm}: the data-dependent axis should cost a couple of leaves, not {worst:.4}x");
}

#[derive(Clone, Copy)]
struct P2 { id: u32, p: Point }
impl Positioned for P2 {
    fn position(&self) -> Point { self.p }
}

#[derive(Clone, Copy)]
struct P3 { id: u32, p: Point3 }
impl Positioned3 for P3 {
    fn position(&self) -> Point3 { self.p }
}

#[derive(Clone, Copy)]
struct PI { p: IPoint }
impl IPositioned for PI {
    fn position(&self) -> IPoint { self.p }
}

fn rect() -> Rect { Rect { x: 0.0, y: 0.0, width: W, height: W } }
fn aabb() -> Aabb { Aabb { x: 0.0, y: 0.0, z: 0.0, w: W, h: W, d: W } }

#[test]
fn tree2_shape_vs_a_rebuild() {
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let mut t = Tree::new(rect(), CAP);
        let mut pos: Vec<Point> = start.iter().map(|s| Point::new(s.0, s.1)).collect();
        let refs: Vec<_> = pos.iter().enumerate()
            .map(|(i, p)| t.insert_ref(P2 { id: i as u32, p: *p }).expect("inside the world"))
            .collect();
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = Point::new(clamp(pos[i].x + d.0), clamp(pos[i].y + d.1));
                t.update_ref(refs[i], |m| m.p = np);
                if np != pos[i] { crossings += 1; }
                pos[i] = np;
            }
        }
        let mut fresh = Tree::new(rect(), CAP);
        for (i, p) in pos.iter().enumerate() { fresh.insert_ref(P2 { id: i as u32, p: *p }); }
        guards("Tree", true, crossings);
        shapes.push((t.leaf_count(), fresh.leaf_count()));
    }
    verdict_data_dependent_axis("Tree", &shapes);
}

#[test]
fn integertree_shape_vs_a_rebuild() {
    let iw = W as i32;
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let mut t = IntegerTree::new(IRect::new(0, 0, iw, iw), CAP);
        let mut pos: Vec<IPoint> = start.iter().map(|s| IPoint::new(s.0 as i32, s.1 as i32)).collect();
        let refs: Vec<_> = pos.iter()
            .map(|p| t.insert_ref(PI { p: *p }).expect("inside the world"))
            .collect();
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = IPoint::new((pos[i].x + d.0 as i32).clamp(0, iw - 1),
                                     (pos[i].y + d.1 as i32).clamp(0, iw - 1));
                t.update_ref(refs[i], |m| m.p = np);
                if np != pos[i] { crossings += 1; }
                pos[i] = np;
            }
        }
        let mut fresh = IntegerTree::new(IRect::new(0, 0, iw, iw), CAP);
        for p in pos.iter() { fresh.insert_ref(PI { p: *p }); }
        guards("IntegerTree", true, crossings);
        shapes.push((t.leaf_count(), fresh.leaf_count()));
    }
    verdict_data_dependent_axis("IntegerTree", &shapes);
}

#[test]
fn quadtree_shape_vs_a_rebuild() {
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let mut t = QuadTree::new(rect(), CAP);
        let mut pos: Vec<Point> = start.iter().map(|s| Point::new(s.0, s.1)).collect();
        let refs: Vec<_> = pos.iter().enumerate()
            .map(|(i, p)| t.insert_ref(P2 { id: i as u32, p: *p }).expect("inside the world"))
            .collect();
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = Point::new(clamp(pos[i].x + d.0), clamp(pos[i].y + d.1));
                t.update_ref(refs[i], |m| m.p = np);
                if np != pos[i] { crossings += 1; }
                pos[i] = np;
            }
        }
        let mut fresh = QuadTree::new(rect(), CAP);
        for (i, p) in pos.iter().enumerate() { fresh.insert_ref(P2 { id: i as u32, p: *p }); }
        guards("QuadTree", true, crossings);
        shapes.push((t.leaf_count(), fresh.leaf_count()));
    }
    verdict("QuadTree", &shapes);
}

#[test]
fn tree3_shape_vs_a_rebuild() {
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let mut t = Tree3::new(aabb(), CAP);
        let mut pos: Vec<Point3> = start.iter().map(|s| Point3::new(s.0, s.1, s.2)).collect();
        let refs: Vec<_> = pos.iter().enumerate()
            .map(|(i, p)| t.insert_ref(P3 { id: i as u32, p: *p }).expect("inside the world"))
            .collect();
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = Point3::new(clamp(pos[i].x + d.0), clamp(pos[i].y + d.1), clamp(pos[i].z + d.2));
                t.update_ref(refs[i], |m| m.p = np);
                if np != pos[i] { crossings += 1; }
                pos[i] = np;
            }
        }
        let mut fresh = Tree3::new(aabb(), CAP);
        for (i, p) in pos.iter().enumerate() { fresh.insert_ref(P3 { id: i as u32, p: *p }); }
        guards("Tree3", true, crossings);
        shapes.push((t.leaf_count(), fresh.leaf_count()));
    }
    // Tree3 is binary like `Tree`, but it splits the LONGEST axis with a `>=` tie-break — pure
    // geometry, never a count of the items. That one difference is why it belongs here.
    verdict("Tree3", &shapes);
}

#[test]
fn octree3_shape_vs_a_rebuild() {
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let mut t = Octree3::new(aabb(), CAP);
        let mut pos: Vec<Point3> = start.iter().map(|s| Point3::new(s.0, s.1, s.2)).collect();
        let refs: Vec<_> = pos.iter().enumerate()
            .map(|(i, p)| t.insert_ref(P3 { id: i as u32, p: *p }).expect("inside the world"))
            .collect();
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = Point3::new(clamp(pos[i].x + d.0), clamp(pos[i].y + d.1), clamp(pos[i].z + d.2));
                t.update_ref(refs[i], |m| m.p = np);
                if np != pos[i] { crossings += 1; }
                pos[i] = np;
            }
        }
        let mut fresh = Octree3::new(aabb(), CAP);
        for (i, p) in pos.iter().enumerate() { fresh.insert_ref(P3 { id: i as u32, p: *p }); }
        guards("Octree3", true, crossings);
        shapes.push((t.leaf_count(), fresh.leaf_count()));
    }
    verdict("Octree3", &shapes);
}

#[test]
fn linear_quadtree_shape_vs_a_rebuild() {
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let items: Vec<P2> = start.iter().enumerate()
            .map(|(i, s)| P2 { id: i as u32, p: Point::new(s.0, s.1) }).collect();
        let mut pos: Vec<Point> = items.iter().map(|m| m.p).collect();
        let mut t = LinearQuadTree::from_items(rect(), CAP, 12, items);
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = Point::new(clamp(pos[i].x + d.0), clamp(pos[i].y + d.1));
                let id = i as u32;
                if let Crossed::Moved = t.update(pos[i], |m: &P2| m.id == id, |m: &mut P2| m.p = np) {
                    crossings += 1;
                }
                pos[i] = np;
            }
        }
        let fresh_items: Vec<P2> = pos.iter().enumerate()
            .map(|(i, p)| P2 { id: i as u32, p: *p }).collect();
        let fresh = LinearQuadTree::from_items(rect(), CAP, 12, fresh_items);
        guards("LinearQuadTree", true, crossings);
        shapes.push((t.leaf_count(), fresh.leaf_count()));
    }
    verdict("LinearQuadTree", &shapes);
}

#[test]
fn linear_octree3_shape_vs_a_rebuild() {
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let items: Vec<P3> = start.iter().enumerate()
            .map(|(i, s)| P3 { id: i as u32, p: Point3::new(s.0, s.1, s.2) }).collect();
        let mut pos: Vec<Point3> = items.iter().map(|m| m.p).collect();
        let mut t = LinearOctree3::from_items(aabb(), CAP, 12, items);
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = Point3::new(clamp(pos[i].x + d.0), clamp(pos[i].y + d.1), clamp(pos[i].z + d.2));
                let id = i as u32;
                if let Crossed::Moved = t.update(pos[i], |m: &P3| m.id == id, |m: &mut P3| m.p = np) {
                    crossings += 1;
                }
                pos[i] = np;
            }
        }
        let fresh_items: Vec<P3> = pos.iter().enumerate()
            .map(|(i, p)| P3 { id: i as u32, p: *p }).collect();
        let fresh = LinearOctree3::from_items(aabb(), CAP, 12, fresh_items);
        guards("LinearOctree3", true, crossings);
        shapes.push((t.leaf_count(), fresh.leaf_count()));
    }
    verdict("LinearOctree3", &shapes);
}

#[test]
fn morton_grid_2d_shape_vs_a_rebuild() {
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let mut g = MortonGrid::new(rect(), 4);
        let mut pos: Vec<Point> = start.iter().map(|s| Point::new(s.0, s.1)).collect();
        for (i, p) in pos.iter().enumerate() { g.insert(P2 { id: i as u32, p: *p }); }
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = Point::new(clamp(pos[i].x + d.0), clamp(pos[i].y + d.1));
                let id = i as u32;
                if let Crossed::Moved = g.update(pos[i], |m: &P2| m.id == id, |m: &mut P2| m.p = np) {
                    crossings += 1;
                }
                pos[i] = np;
            }
        }
        let mut fresh = MortonGrid::new(rect(), 4);
        for (i, p) in pos.iter().enumerate() { fresh.insert(P2 { id: i as u32, p: *p }); }
        guards("MortonGrid", false, crossings);
        let (a, b) = (g.occupancy(), fresh.occupancy());
        assert_eq!((a.items, a.max), (b.items, b.max), "MortonGrid: kept {a:?} vs fresh {b:?}");
        shapes.push((a.cells, b.cells));
    }
    verdict("MortonGrid", &shapes);
}

/// **The second mechanism, on purpose: `divide` and `try_merge_up` do not ask the same question.**
///
/// `divide` refuses to split a node whose items are **all at one point** — there is no plane that
/// separates them, so splitting would recurse forever. `try_merge_up` collapses children only when
/// they *between them fit in one leaf*. Those are two different predicates, and the gap between
/// them is a one-way door: a node that split while its points were spread, and whose points then
/// became coincident, holds more than `merge_limit` and can never collapse — while a rebuild from
/// those same points refuses to split it at all and leaves a single leaf.
///
/// This is what made `QuadTree` drift on 2 of 12 seeds before the workload stopped **pinning**
/// escapees to the wall (see [`clamp`]): clamping in 2D piles points onto four exactly-coincident
/// corners. In 3D a point must be clamped in all three axes at once to coincide, which is why the
/// 3D arms never showed it.
///
/// It is **left as it is, deliberately.** Closing it means having `try_merge_up` also merge when
/// the combined items are inseparable, and that test is an O(combined) scan on the branch a
/// rejected merge takes — which is the common branch, on the relocation hot path. A rare shape
/// difference that costs nothing but a little traversal is not worth taxing every item that moves.
/// Recorded here so it is a known property with a reason, rather than a surprise.
#[test]
fn coincident_points_are_a_one_way_door_for_every_pointer_tree() {
    let spread: Vec<Point> = (0..40).map(|i| Point::new(1.0 + i as f64 * 6.0, 1.0 + (i % 7) as f64 * 9.0)).collect();
    let here = Point::new(101.5, 101.5);

    let mut kept = QuadTree::new(rect(), CAP);
    let refs: Vec<_> = spread.iter().enumerate()
        .map(|(i, p)| kept.insert_ref(P2 { id: i as u32, p: *p }).expect("inside")).collect();
    let split_while_spread = kept.leaf_count();
    for r in &refs { kept.update_ref(*r, |m| m.p = here); }

    let mut fresh = QuadTree::new(rect(), CAP);
    for i in 0..spread.len() { fresh.insert_ref(P2 { id: i as u32, p: here }); }

    assert!(split_while_spread > 1, "the setup must actually subdivide first, got {split_while_spread}");
    assert_eq!(fresh.leaf_count(), 1,
               "a rebuild from coincident points must refuse to split at all, got {}", fresh.leaf_count());
    assert!(kept.leaf_count() > fresh.leaf_count(),
            "the one-way door should leave the maintained tree MORE subdivided than a rebuild: \
             kept {} vs fresh {}. If this now reads equal, `try_merge_up` has learned to collapse \
             inseparable children — which is a fix, not a failure: delete this test and move the \
             pointer trees onto exact equality in the sweep above.",
            kept.leaf_count(), fresh.leaf_count());

    // Same answers regardless — the shape differs, the results do not. That is the line that
    // matters, and it is why this is a documented property rather than a bug.
    let s = vectorial_hash::culling::Circle::new(here, 5.0);
    assert_eq!(kept.cull(&s).len(), fresh.cull(&s).len(),
               "shape may differ; the answer may not");
}

#[test]
fn morton_grid_3d_shape_vs_a_rebuild() {
    let mut shapes = Vec::new();
    for seed in 0..SEEDS {
        let (start, frames) = script(seed);
        let mut g = MortonGrid3::new(aabb(), 4);
        let mut pos: Vec<Point3> = start.iter().map(|s| Point3::new(s.0, s.1, s.2)).collect();
        for (i, p) in pos.iter().enumerate() { g.insert(P3 { id: i as u32, p: *p }); }
        reset();
        let mut crossings = 0u64;
        for f in &frames {
            for (i, d) in f.iter().enumerate() {
                let np = Point3::new(clamp(pos[i].x + d.0), clamp(pos[i].y + d.1), clamp(pos[i].z + d.2));
                let id = i as u32;
                if let Crossed::Moved = g.update(pos[i], |m: &P3| m.id == id, |m: &mut P3| m.p = np) {
                    crossings += 1;
                }
                pos[i] = np;
            }
        }
        let mut fresh = MortonGrid3::new(aabb(), 4);
        for (i, p) in pos.iter().enumerate() { fresh.insert(P3 { id: i as u32, p: *p }); }
        guards("MortonGrid3", false, crossings);
        let (a, b) = (g.occupancy(), fresh.occupancy());
        assert_eq!((a.items, a.max), (b.items, b.max), "MortonGrid3: kept {a:?} vs fresh {b:?}");
        shapes.push((a.cells, b.cells));
    }
    verdict("MortonGrid3", &shapes);
}
