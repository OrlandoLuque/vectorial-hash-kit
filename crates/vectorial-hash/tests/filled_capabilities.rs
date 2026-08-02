//! Gates for the capability holes filled on 2026-08-02.
//!
//! These live as an *integration* test on purpose: every one of them was a gap in the public
//! surface — a verb one structure had and its sibling did not — so the thing to exercise is the
//! surface a caller sees, not the internals.
//!
//! Each is refereed by brute force where brute force can referee it. Comparing a new code path
//! against an existing one only proves they agree, which is worth much less when the question
//! is whether either is right.
use vectorial_hash::*;

#[derive(Clone, Copy, Debug, PartialEq)]
struct P { id: u32, p: Point }
impl Positioned for P { fn position(&self) -> Point { self.p } }
#[derive(Clone, Copy, Debug)]
struct P3 { id: u32, p: Point3 }
impl Positioned3 for P3 { fn position(&self) -> Point3 { self.p } }

/// Points strictly inside the world box, genuinely scattered.
///
/// The first draft drew them from `sin(a)` and `cos(a)` of the SAME argument, which puts every
/// point on a circle — the probes at the centre of the world then hit nothing and the tests
/// failed on their own non-vacuity guards rather than on a real defect. Independent hashed
/// coordinates instead.
///
/// `Rect` is half-open, so a point landing exactly on `x + width` is OUTSIDE and `insert`
/// rightly refuses it; the 5.0 margin keeps that out of the way of what is being tested here.
fn scatter(i: u32, salt: u64) -> f64 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30; x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27; x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64 * 190.0 + 5.0
}
fn pts(n: u32) -> Vec<P> {
    (0..n).map(|i| P { id: i, p: Point::new(scatter(i, 1), scatter(i, 2)) }).collect()
}
fn pts3(n: u32) -> Vec<P3> {
    (0..n).map(|i| P3 { id: i, p: Point3::new(scatter(i, 1), scatter(i, 2), scatter(i, 3)) }).collect()
}
fn rect() -> Rect { Rect::new(0.0, 0.0, 200.0, 200.0) }

/// `bulk_load` must produce a tree that ANSWERS like one built by repeated `insert` — not
/// merely one holding the same count. Node layout is free to differ; answers are not.
#[test]
fn quadtree_bulk_load_answers_like_repeated_insert() {
    let items = pts(600);
    let mut inserted = QuadTree::<P>::new(rect(), 8);
    for it in &items { assert!(inserted.insert(*it)); }
    let bulk = QuadTree::<P>::bulk_load(rect(), 8, items.clone());
    assert_eq!(bulk.item_count(), inserted.item_count());

    let mut probed = 0;
    for (cx, cy) in [(30.0, 30.0), (100.0, 100.0), (170.0, 60.0), (55.0, 140.0)] {
        let s = Circle::new(Point::new(cx, cy), 40.0);
        let mut a: Vec<u32> = inserted.cull(&s).iter().map(|x| x.id).collect();
        let mut b: Vec<u32> = bulk.cull(&s).iter().map(|x| x.id).collect();
        let mut want: Vec<u32> = items.iter().filter(|x| s.contains_point(x.p)).map(|x| x.id).collect();
        a.sort_unstable(); b.sort_unstable(); want.sort_unstable();
        assert_eq!(a, want, "the insert-built tree disagrees with brute force");
        assert_eq!(b, want, "the bulk-loaded tree disagrees with brute force");
        probed += want.len();
    }
    assert!(probed > 20, "the probes must actually hit things ({probed})");
}

/// Out-of-world items are dropped, exactly as `insert` drops them. This is the contract that
/// stops an index and a linear scan silently answering about different sets (docs/STEALTH.md).
#[test]
fn quadtree_bulk_load_drops_out_of_world_items_like_insert() {
    let mut items = pts(50);
    items.push(P { id: 998, p: Point::new(-10.0, 5.0) });
    items.push(P { id: 999, p: Point::new(5.0, 1e6) });
    let bulk = QuadTree::<P>::bulk_load(rect(), 8, items.clone());
    let mut inserted = QuadTree::<P>::new(rect(), 8);
    let accepted = items.iter().filter(|it| inserted.insert(**it)).count();
    assert_eq!(accepted, 50, "insert must reject the two strays, or this test proves nothing");
    assert_eq!(bulk.item_count(), 50, "bulk_load kept an item insert would have refused");
}

#[test]
fn integertree_bulk_load_answers_like_repeated_insert() {
    #[derive(Clone, Copy)]
    struct I { id: u32, p: IPoint }
    impl IPositioned for I { fn position(&self) -> IPoint { self.p } }
    let items: Vec<I> = (0..500u32)
        .map(|i| I { id: i, p: IPoint::new(scatter(i, 1) as i32, scatter(i, 2) as i32) }).collect();
    let bbox = IRect::new(0, 0, 256, 256); // IntegerTree requires a power-of-two side
    let mut inserted = IntegerTree::<I>::new(bbox, 8);
    for it in &items { assert!(inserted.insert(*it)); }
    let bulk = IntegerTree::<I>::bulk_load(bbox, 8, items.clone());
    assert_eq!(bulk.item_count(), inserted.item_count());

    // `IntegerTree::cull` runs on the float `Shape` machinery, converting at the boundary.
    let s = Circle::new(Point::new(100.0, 100.0), 45.0);
    let mut a: Vec<u32> = inserted.cull(&s).iter().map(|x| x.id).collect();
    let mut b: Vec<u32> = bulk.cull(&s).iter().map(|x| x.id).collect();
    let mut want: Vec<u32> = items.iter()
        .filter(|x| s.contains_point(Point::new(x.p.x as f64, x.p.y as f64))).map(|x| x.id).collect();
    a.sort_unstable(); b.sort_unstable(); want.sort_unstable();
    assert_eq!(a, want, "the insert-built tree disagrees with brute force");
    assert_eq!(b, want, "the bulk-loaded tree disagrees with brute force");
    assert!(!want.is_empty(), "the probe must hit something");
}

/// Compaction must change no answer, keep every handle addressing its own item, and actually
/// reclaim the holes churn left. That last assertion is what stops this passing with an empty
/// body — and the one before it is what stops it passing on a tree that never churned.
#[test]
fn quadtree_compact_reclaims_holes_without_changing_answers() {
    let items = pts(800);
    let mut t = QuadTree::<P>::new(rect(), 4);
    let refs: Vec<ItemRef> = items.iter().map(|it| t.insert_ref(*it).unwrap()).collect();
    for (k, r) in refs.iter().enumerate() {
        if k % 3 == 0 { t.update_ref(*r, |it| it.p = Point::new(1.0 + (k % 7) as f64, 1.0 + (k % 5) as f64)); }
    }
    for (k, r) in refs.iter().enumerate() {
        if k % 3 == 0 { t.update_ref(*r, |it| it.p = items[k].p); }
    }
    let before_nodes = t.node_count();
    let live = t.live_node_count();
    assert!(before_nodes > live, "the churn must leave holes or this proves nothing ({before_nodes} vs {live})");

    let s = Circle::new(Point::new(100.0, 100.0), 45.0);
    let mut want: Vec<u32> = items.iter().filter(|x| s.contains_point(x.p)).map(|x| x.id).collect();
    let mut before: Vec<u32> = t.cull(&s).iter().map(|x| x.id).collect();
    want.sort_unstable(); before.sort_unstable();
    assert_eq!(before, want, "the tree must agree with brute force before compacting");
    assert!(!want.is_empty(), "the probe must hit something");

    t.compact();

    let mut after: Vec<u32> = t.cull(&s).iter().map(|x| x.id).collect();
    after.sort_unstable();
    assert_eq!(after, want, "compaction changed an answer");
    assert_eq!(t.node_count(), t.live_node_count(), "compaction left holes behind");
    assert!(t.node_count() < before_nodes, "compaction reclaimed nothing");
    for (k, r) in refs.iter().enumerate() {
        assert_eq!(t.get_ref(*r).map(|it| it.id), Some(items[k].id), "handle {k} stopped addressing its item");
    }
}

/// The batch verbs must equal the serial loop they wrap — trivial to write, trivially wrong
/// the day a `par_iter` reorders results.
#[test]
fn batch_verbs_equal_the_serial_loop() {
    let items = pts(400);
    let t = QuadTree::<P>::bulk_load(rect(), 8, items.clone());
    let qs: Vec<Point> = (0..12).map(|i| items[i * 17].p).collect();
    let many = t.knn_many(&qs, 5);
    assert_eq!(many.len(), qs.len());
    for (i, got) in many.iter().enumerate() {
        let want = t.knn(qs[i], 5);
        assert_eq!(got.len(), want.len(), "knn_many returned a different count at {i}");
        for (a, b) in got.iter().zip(&want) { assert!((a.0 - b.0).abs() < 1e-12, "knn_many differs at {i}"); }
    }
    assert!(many.iter().any(|v| !v.is_empty()), "the batch must return something");

    let items3 = pts3(400);
    let k3 = KdTree3::<P3>::from_items(8, items3.clone());
    let shapes: Vec<Sphere3> = (0..8).map(|i| { let p = items3[i * 31].p; Sphere3::new(p.x, p.y, p.z, 25.0) }).collect();
    let batch = k3.cull_many(&shapes);
    for (i, got) in batch.iter().enumerate() {
        let mut a: Vec<u32> = got.iter().map(|x| x.id).collect();
        let mut want: Vec<u32> = items3.iter().filter(|x| shapes[i].contains_point(x.p)).map(|x| x.id).collect();
        a.sort_unstable(); want.sort_unstable();
        assert_eq!(a, want, "KdTree3::cull_many disagrees with brute force at {i}");
    }
    assert!(batch.iter().map(|v| v.len()).sum::<usize>() > 8, "the probes must hit things");
}

/// The linear trees' new diagnostics must describe the tree they are on, and iteration must
/// return every item exactly once — including in the ordered form, which must actually order.
#[test]
fn linear_tree_occupancy_and_iteration_are_honest() {
    let items3 = pts3(700);
    let t = LinearOctree3::<P3>::from_items(Aabb::new(0.0, 0.0, 0.0, 200.0, 200.0, 200.0), 8, 12, items3.clone());
    let occ = t.occupancy();
    assert_eq!(occ.items, t.item_count());
    assert!(occ.cells > 1, "700 points must have split into more than one leaf");
    assert!((occ.mean - occ.items as f64 / occ.cells as f64).abs() < 1e-9);
    assert!(occ.max as f64 >= occ.mean, "max cannot be below the mean");

    let mut seen: Vec<u32> = t.iter().map(|x| x.id).collect();
    let mut zorder: Vec<u32> = t.iter_z_order().map(|x| x.id).collect();
    let mut want: Vec<u32> = items3.iter().map(|x| x.id).collect();
    seen.sort_unstable(); zorder.sort_unstable(); want.sort_unstable();
    assert_eq!(seen, want, "iter() lost or duplicated items");
    assert_eq!(zorder, want, "iter_z_order() lost or duplicated items");

    // and it must be an ORDER, not hash order wearing a different name
    let hash_order: Vec<u32> = t.iter().map(|x| x.id).collect();
    let z_order: Vec<u32> = t.iter_z_order().map(|x| x.id).collect();
    assert_ne!(hash_order, z_order, "iter_z_order returned hash order — it is not ordering anything");
    let again: Vec<u32> = t.iter_z_order().map(|x| x.id).collect();
    assert_eq!(z_order, again, "iter_z_order must be deterministic");
}
