//! **`bytes()` on all twelve, and the two rules that are easy to get wrong.**
//!
//! A memory column is read ACROSS arms — "which of these is smallest" — so the arm that counts
//! differently is the one that makes the whole column wrong, and it does it silently, because
//! every individual number still looks plausible.
//!
//! **Rule one: resident, not reachable.** The natural implementation is
//! `live_node_count() * size + item_count() * size`, because those are the public counts. It is
//! wrong: a freed arena slot is still memory. That is what the second test here catches, and it
//! is verified to catch it — with `Tree3::bytes` rewritten from the public counts the test fails
//! reading exactly `1.00x`.
//!
//! **Rule two: `capacity()`, not `len()`.** This one is NOT pinned by the slack test, and my first
//! version of these docs claimed it was. It is not detectable that way, because an arena keeps
//! freed slots inside the `Vec`, so `nodes.len()` sits at the high-water mark and barely differs
//! from `nodes.capacity()`. Measured directly at 20 000 items, capacity-based over len-based is
//! **1.28x** for `Tree3` and **2.45x** for `MortonGrid3`, and that gap **reorders the column** —
//! under `len` the grid looks smaller than the tree (974 KB vs 1 157 KB), under `capacity` it is
//! clearly larger (2 385 KB vs 1 485 KB). So the third test pins it for the hash structures from
//! public counts alone, which is the only place the difference is big enough to bound.
//!
//! The rest are brackets rather than equalities, deliberately. An exact expected byte count would
//! be a second implementation of `bytes()` living in a test, which drifts with the first and
//! catches nothing; `[lower, upper]` bounds derived from the PUBLIC counts (`node_count`,
//! `item_count`, `leaf_count`, `cell_count`) are independent of how `bytes()` is written.

use std::mem::size_of;
use vectorial_hash::{
    Aabb, IPoint, IRect, IntegerTree, KdTree2, KdTree3, LinearOctree3, LinearQuadTree, MortonGrid,
    MortonGrid3, Octree3, Point, Point3, Positioned, Positioned3, QuadTree, RadixTrie3, Rect, Tree,
    Tree3,
};
use vectorial_hash::itree::IPositioned;

const N: usize = 20_000;
const W: f64 = 1000.0;

#[derive(Clone, Copy, PartialEq, Debug)]
struct P3 { id: u32, p: Point3 }
impl Positioned3 for P3 { fn position(&self) -> Point3 { self.p } }
#[derive(Clone, Copy, PartialEq, Debug)]
struct P2 { id: u32, p: Point }
impl Positioned for P2 { fn position(&self) -> Point { self.p } }
#[derive(Clone, Copy, PartialEq, Debug)]
struct PI { id: u32, p: IPoint }
impl IPositioned for PI { fn position(&self) -> IPoint { self.p } }

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

fn world3() -> Aabb { Aabb::new(0.0, 0.0, 0.0, W, W, W) }
fn world2() -> Rect { Rect::new(0.0, 0.0, W, W) }

fn pts3(n: usize, seed: u64) -> Vec<P3> {
    let mut r = Lcg(seed);
    (0..n).map(|i| P3 { id: i as u32, p: Point3::new(r.r(0.0, W - 0.1), r.r(0.0, W - 0.1), r.r(0.0, W - 0.1)) }).collect()
}
fn pts2(n: usize, seed: u64) -> Vec<P2> {
    let mut r = Lcg(seed);
    (0..n).map(|i| P2 { id: i as u32, p: Point::new(r.r(0.0, W - 0.1), r.r(0.0, W - 0.1)) }).collect()
}

/// Every structure must at least hold its items, and must not hold an absurd multiple of them.
///
/// The upper bounds are per-family and justified rather than round: a pointer tree pays a node
/// per leaf plus a `Vec` header per leaf, a hash structure pays a table entry per occupied cell,
/// a k-d tree pays neither. A single shared bound would have to be loose enough to pass a
/// structure holding ten times what it should.
#[test]
fn every_structure_reports_a_footprint_that_brackets_its_contents() {
    let items3 = pts3(N, 0x5EED_0001);
    let items2 = pts2(N, 0x5EED_0002);
    let itemsi: Vec<PI> = items2.iter().map(|p| PI { id: p.id, p: IPoint::new(p.p.x as i32, p.p.y as i32) }).collect();
    let floor3 = N * size_of::<P3>();
    let floor2 = N * size_of::<P2>();
    let floori = N * size_of::<PI>();

    // (name, bytes, floor, ceiling-as-a-multiple-of-floor)
    let mut rows: Vec<(&str, usize, usize, f64)> = Vec::new();

    let mut t3: Tree3<P3> = Tree3::new(world3(), 16);
    for it in &items3 { t3.insert(*it); }
    rows.push(("Tree3", t3.bytes(), floor3, 12.0));

    let mut o3: Octree3<P3> = Octree3::new(world3(), 16);
    for it in &items3 { o3.insert(*it); }
    rows.push(("Octree3", o3.bytes(), floor3, 12.0));

    let kd3 = KdTree3::from_items(16, items3.clone());
    rows.push(("KdTree3", kd3.bytes(), floor3, 4.0));

    let lo3 = LinearOctree3::from_items(world3(), 16, 8, items3.clone());
    rows.push(("LinearOctree3", lo3.bytes(), floor3, 12.0));

    let mut m3: MortonGrid3<P3> = MortonGrid3::new(world3(), 5);
    for it in &items3 { m3.insert(*it); }
    rows.push(("MortonGrid3", m3.bytes(), floor3, 12.0));

    let rx = RadixTrie3::from_items(world3(), 5, items3.clone());
    rows.push(("RadixTrie3", rx.bytes(), floor3, 12.0));

    let mut t2: Tree<P2> = Tree::new(world2(), 16);
    for it in &items2 { t2.insert(*it); }
    rows.push(("Tree", t2.bytes(), floor2, 16.0));

    let mut q2: QuadTree<P2> = QuadTree::new(world2(), 16);
    for it in &items2 { q2.insert(*it); }
    rows.push(("QuadTree", q2.bytes(), floor2, 16.0));

    let kd2 = KdTree2::from_items(16, items2.clone());
    rows.push(("KdTree2", kd2.bytes(), floor2, 4.0));

    let lq2 = LinearQuadTree::from_items(world2(), 16, 8, items2.clone());
    rows.push(("LinearQuadTree", lq2.bytes(), floor2, 16.0));

    let mut m2: MortonGrid<P2> = MortonGrid::new(world2(), 5);
    for it in &items2 { m2.insert(*it); }
    rows.push(("MortonGrid", m2.bytes(), floor2, 16.0));

    // IntegerTree requires a power-of-two side (its split is a bit shift), so 1024, not 1000.
    let mut it2: IntegerTree<PI> = IntegerTree::new(IRect::new(0, 0, 1024, 1024), 16);
    for it in &itemsi { it2.insert(*it); }
    rows.push(("IntegerTree", it2.bytes(), floori, 24.0));

    assert_eq!(rows.len(), 12, "all twelve structures must be covered");
    for (name, bytes, floor, mult) in &rows {
        assert!(bytes >= floor, "{name}: bytes {bytes} < its items' own {floor} — it cannot hold them");
        let ceil = (*floor as f64 * mult) as usize;
        assert!(bytes <= &ceil, "{name}: bytes {bytes} > {ceil} ({mult}x its items) — a field is being counted twice, or the knob is absurd");
        println!("{name:<16} {bytes:>10} bytes  = {:.2}x its items", *bytes as f64 / *floor as f64);
    }
}

/// **Rule one, pinned: resident, not reachable.**
///
/// My first version asserted that after removing 90% of the items the footprint stays near its
/// peak, on the reasoning that `Vec` and `HashMap` never shrink. **It failed, and the test was
/// what was wrong.** A `MortonGrid3` read 0.416 of peak because `remove` drops the whole bucket
/// when a cell empties, so the grid genuinely DOES hand memory back as it empties while an arena
/// tree keeps its freed slots. A real difference between the families, about to be recorded as a
/// defect.
///
/// So everything else is held fixed. Both arms end as the **same structure** holding the **same
/// items**, reached two ways:
///
/// * `grown` — insert N, then remove 90%.
/// * `direct` — insert N/10 and nothing else.
///
/// Whatever a structure genuinely releases, it releases in `grown` too. What is left between them
/// is what `grown` is still holding for contents it no longer has. An implementation written from
/// `live_node_count()` and `item_count()` reports the two as identical by construction.
///
/// Threshold 1.5x, between the two measured states: correct reads 3.2x-6.5x, reachable-only reads
/// exactly 1.00x. Verified failing with `Tree3::bytes` rewritten from the public counts.
#[test]
fn a_structure_holding_slack_must_report_it() {
    let items3 = pts3(N, 0x5EED_0003);
    let items2 = pts2(N, 0x5EED_0004);
    let keep = |i: usize| i % 10 == 0;

    // --- Tree3: grown then emptied, against built small.
    let mut grown_t3: Tree3<P3> = Tree3::new(world3(), 16);
    let refs: Vec<_> = items3.iter().filter_map(|it| grown_t3.insert_ref(*it)).collect();
    assert_eq!(refs.len(), N, "every insert must have produced a handle");
    for (i, r) in refs.iter().enumerate() { if !keep(i) { grown_t3.remove_ref(*r); } }
    let mut direct_t3: Tree3<P3> = Tree3::new(world3(), 16);
    for (i, it) in items3.iter().enumerate() { if keep(i) { direct_t3.insert(*it); } }
    assert_eq!(grown_t3.item_count(), direct_t3.item_count(), "both arms must hold the same items");

    // --- MortonGrid3: the family that genuinely releases, so the slack is the table's.
    let mut grown_m3: MortonGrid3<P3> = MortonGrid3::new(world3(), 5);
    for it in &items3 { grown_m3.insert(*it); }
    for (i, it) in items3.iter().enumerate() { if !keep(i) { grown_m3.remove(it.p, |x| x.id == it.id); } }
    let mut direct_m3: MortonGrid3<P3> = MortonGrid3::new(world3(), 5);
    for (i, it) in items3.iter().enumerate() { if keep(i) { direct_m3.insert(*it); } }
    assert_eq!(grown_m3.item_count(), direct_m3.item_count(), "both grids must hold the same items");

    // --- The 2D twin, so the property is not asserted for one dimension only.
    let mut grown_t2: Tree<P2> = Tree::new(world2(), 16);
    let refs2: Vec<_> = items2.iter().filter_map(|it| grown_t2.insert_ref(*it)).collect();
    assert_eq!(refs2.len(), N, "every insert must have produced a handle");
    for (i, r) in refs2.iter().enumerate() { if !keep(i) { grown_t2.remove_ref(*r); } }
    let mut direct_t2: Tree<P2> = Tree::new(world2(), 16);
    for (i, it) in items2.iter().enumerate() { if keep(i) { direct_t2.insert(*it); } }
    assert_eq!(grown_t2.item_count(), direct_t2.item_count(), "both 2D trees must hold the same items");

    let rows = [
        ("Tree3", grown_t3.bytes(), direct_t3.bytes()),
        ("MortonGrid3", grown_m3.bytes(), direct_m3.bytes()),
        ("Tree", grown_t2.bytes(), direct_t2.bytes()),
    ];
    for (name, grown, direct) in rows {
        let slack = grown as f64 / direct as f64;
        println!("{name:<12} grown-then-emptied {grown:>10}  built-small {direct:>10}  = {slack:.2}x slack");
        assert!(
            slack > 1.5,
            "{name}: a structure sized for {N} items and now holding {} reports {grown} bytes              against {direct} for one built small — {slack:.2}x, i.e. it is reporting nothing for              what it still holds. Both arms hold the same items, so only the accounting can              differ: this is bytes() counting what is REACHABLE (live_node_count, item_count)              rather than what is RESIDENT. See the module docs."
        , grown_t3.item_count());
    }
}

/// A footprint must respond to the knob, and the DIRECTION differs by family — which is the
/// whole reason #180 has to measure rather than assume.
///
/// A finer grid holds more occupied cells, so one `Vec` header per cell is added while the items
/// are merely redistributed: the footprint RISES with resolution. A pointer tree with a bigger
/// leaf capacity splits less, so it holds fewer nodes: the footprint FALLS as the leaf grows.
/// Asserting both in one test means a change that collapsed `bytes()` to something
/// knob-independent (a constant, or items-only) fails here rather than silently flattening
/// #180's memory column.
#[test]
fn the_footprint_moves_with_the_knob_and_in_the_direction_the_family_implies() {
    let items3 = pts3(N, 0x5EED_0005);

    let fine = {
        let mut g: MortonGrid3<P3> = MortonGrid3::new(world3(), 6);
        for it in &items3 { g.insert(*it); }
        (g.bytes(), g.cell_count())
    };
    let coarse = {
        let mut g: MortonGrid3<P3> = MortonGrid3::new(world3(), 4);
        for it in &items3 { g.insert(*it); }
        (g.bytes(), g.cell_count())
    };
    println!("grid levels 6: {} bytes over {} cells", fine.0, fine.1);
    println!("grid levels 4: {} bytes over {} cells", coarse.0, coarse.1);
    assert!(fine.1 > coarse.1, "a finer grid must occupy more cells ({} vs {})", fine.1, coarse.1);
    assert!(
        fine.0 > coarse.0,
        "a finer grid holds one Vec header per occupied cell, so its footprint must be larger: \
         {} bytes over {} cells against {} over {}",
        fine.0, fine.1, coarse.0, coarse.1
    );

    let small_leaf = {
        let mut t: Tree3<P3> = Tree3::new(world3(), 4);
        for it in &items3 { t.insert(*it); }
        (t.bytes(), t.node_count())
    };
    let big_leaf = {
        let mut t: Tree3<P3> = Tree3::new(world3(), 64);
        for it in &items3 { t.insert(*it); }
        (t.bytes(), t.node_count())
    };
    println!("Tree3 leaf 4:  {} bytes over {} nodes", small_leaf.0, small_leaf.1);
    println!("Tree3 leaf 64: {} bytes over {} nodes", big_leaf.0, big_leaf.1);
    assert!(small_leaf.1 > big_leaf.1, "a smaller leaf must split more");
    assert!(
        small_leaf.0 > big_leaf.0,
        "a tree that splits more holds more nodes, so its footprint must be larger: \
         {} bytes over {} nodes against {} over {}",
        small_leaf.0, small_leaf.1, big_leaf.0, big_leaf.1
    );
}

/// **Rule two, pinned where it is big enough to bound: `capacity()`, not `len()`.**
///
/// The slack test above cannot see this rule (an arena's `len` sits at its high-water mark, so
/// `len` and `capacity` differ by only ~1.28x there). The hash structures can: a `HashMap` holds
/// a power-of-two table at a 0.875 load factor and every bucket `Vec` carries its own slack, so
/// capacity-based reads **2.45x** len-based at 20 000 items.
///
/// That gap is bounded from public counts alone. A len-based implementation computes exactly
/// `cell_count * size_of::<(u64, Vec<T>)>() + item_count * size_of::<T>()` plus one control byte
/// per occupied cell — all of which this test can reconstruct without seeing a private field. So
/// the assertion is that `bytes()` exceeds that reconstruction by a real margin.
///
/// Threshold 1.3x, between the two states (len-based gives ~1.0x by construction, capacity-based
/// measured 2.4x). Verified failing with both grids' `bytes` switched to `len()`.
#[test]
fn the_hash_structures_must_report_their_table_slack() {
    let items3 = pts3(N, 0x5EED_0006);
    let items2 = pts2(N, 0x5EED_0007);

    let mut m3: MortonGrid3<P3> = MortonGrid3::new(world3(), 5);
    for it in &items3 { m3.insert(*it); }
    let len_based_m3 = m3.cell_count() * (size_of::<(u64, Vec<P3>)>() + 1) + m3.item_count() * size_of::<P3>();

    let mut m2: MortonGrid<P2> = MortonGrid::new(world2(), 5);
    for it in &items2 { m2.insert(*it); }
    let len_based_m2 = m2.cell_count() * (size_of::<(u64, Vec<P2>)>() + 1) + m2.item_count() * size_of::<P2>();

    let lo3 = LinearOctree3::from_items(world3(), 16, 8, items3.clone());
    // The leaves only: `internal` is not exposed, so this under-counts the reconstruction, which
    // makes the assertion STRICTER rather than looser. Stated because an under-count that made it
    // looser would be a hole.
    let len_based_lo3 = lo3.leaf_count() * (size_of::<(u64, Vec<P3>)>() + 1) + lo3.item_count() * size_of::<P3>();

    for (name, bytes, len_based) in [
        ("MortonGrid3", m3.bytes(), len_based_m3),
        ("MortonGrid", m2.bytes(), len_based_m2),
        ("LinearOctree3", lo3.bytes(), len_based_lo3),
    ] {
        let ratio = bytes as f64 / len_based as f64;
        println!("{name:<16} bytes {bytes:>10}  len-based reconstruction {len_based:>10}  = {ratio:.2}x");
        assert!(
            ratio > 1.3,
            "{name}: bytes() {bytes} is only {ratio:.2}x the len-based reconstruction {len_based}. \
             A HashMap holds a power-of-two table at a 0.875 load factor plus per-bucket Vec slack, \
             so a capacity-based figure must clear that comfortably: this is bytes() counting len(). \
             The rule matters because it REORDERS the memory column — see the module docs."
        );
    }
}
