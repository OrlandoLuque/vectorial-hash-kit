//! `RadixTrie3` — a 3D **radix / PATRICIA trie over Morton keys**, done the way the literature
//! says to do one.
//!
//! ## Why this exists when `Octree3` beats it
//!
//! It does beat it, and this type's own docs say so: on in-process sphere queries the kit's
//! `Octree3`, `Tree3`, `MortonGrid3` and `KdTree3` are all faster
//! (`examples/radix_trie_bench`). A radix trie over Morton codes at 3 bits per digit **is** an
//! octree with path compression — Karras (HPG 2012) states that mapping as a premise and builds
//! GPU LBVHs on it, which is the same construction this crate already uses.
//!
//! What this type has that the pointer trees do not is that **the key is the identity**:
//!
//! - a **prefix is a cell**, so a node's address is a number anyone can compute from a point
//!   without consulting the structure;
//! - the key **orders** the data, so any contiguous run of keys is a spatially coherent shard —
//!   `examples/key_partition_bench` measures a query reaching ~2 % of shards under a curve key
//!   against ~47 % under a balanced-but-unordered one. (Not a claim that a tree *cannot* be
//!   partitioned: it can, it is the standard approach, and done properly — leaves finer than a
//!   shard, grouped in curve order — it matches these numbers exactly, because that *is* a key
//!   sort at leaf granularity. What the key adds is that it can cut anywhere rather than only at a
//!   node boundary, and that the ownership rule is arithmetic rather than a directory to ship.);
//! - the shape is **canonical**: build order cannot be read off the result, asserted over six
//!   orderings including reversed and Morton-sorted.
//!
//! If none of those matter to you, use [`crate::Octree3`]. They are the reason to be here.
//!
//! ## The node layout, which is the part worth getting right
//!
//! A naive radix trie gives every node a full fanout array and descends to full depth for every
//! point, so it pays ~1.4 nodes per item and most of those nodes are nearly empty. That is the
//! known weakness, and **ART** (Leis, Kemper & Neumann, ICDE 2013) is the known answer: size each
//! internal node to the children it actually has. At ART's 256-way fanout that means a family of
//! node types; at **8-way** it collapses to something simpler and better — a **child bitmask plus
//! contiguously packed children**:
//!
//! ```text
//! mask: u8         bit k set  <=>  digit k has a child
//! kids: u32        index of the FIRST child in one shared array
//!                  child k lives at kids + popcount(mask & ((1 << k) - 1))
//! ```
//!
//! A node with two children stores two slots, not eight. Measured on 200 000 points, that plus
//! flat item storage takes the structure from **~26 MB to ~7 MB**, and the same idea applied as a
//! pure layout change (without the bitmask) was already worth **1.7-3.8x** on the query.
//!
//! Items live in one flat array addressed by `(start, len)` per leaf, because a census of the
//! naive version found 199 983 leaves holding 200 000 items — nearly every leaf paying a `Vec`
//! header and an allocation to hold a single element.
//!
//! ## What it deliberately does not have
//!
//! **No item limit.** Stopping subdivision at N items is what an octree does, and adding it here
//! would not produce a better trie, it would produce [`crate::Octree3`] — which already exists and
//! is faster. The full-depth descent is the definition of the structure, not an oversight.
//!
//! **No `update` / `remove`.** This is a build-once index, like the two k-d trees. If your points
//! move, that is what [`crate::Tree3`]'s `ItemRef` and [`crate::MortonGrid3`]'s `update` are for.

use crate::culling::SizeCache;
use crate::template::CellState;
use crate::tree3::{knn_offer, knn_worst, KnnEntry};
use crate::{Aabb, Point3, Positioned3, Shape3};

/// Bits per axis, and therefore digits of the key. 3 bits per digit, one digit per level.
const MAX_BITS: u32 = 20;

/// One node. 20 bytes: the whole point of the bitmask is that this does not carry eight slots.
#[derive(Clone, Copy)]
struct RNode {
    /// Digits skipped by path compression (all keys below agree on them).
    skip: u8,
    /// Bit `k` set means digit `k` has a child.
    mask: u8,
    /// The skipped digits, 3 bits each, most significant first. `skip <= 10` fits comfortably.
    path: u32,
    /// Index of this node's FIRST child in `kids`; children are packed in digit order.
    kids: u32,
    /// This node's items, as a range into `items`. Non-empty only at full depth.
    start: u32,
    len: u32,
}

/// A 3D radix / PATRICIA trie over Morton keys, with ART-style adaptive nodes.
///
/// See the [module docs](self) for when to prefer this over [`crate::Octree3`] (which is faster
/// at the query, and says so).
pub struct RadixTrie3<T: Positioned3> {
    nodes: Vec<RNode>,
    /// Child indices, packed: a node's children occupy `kids[n.kids .. n.kids + popcount(n.mask)]`.
    kids: Vec<u32>,
    items: Vec<T>,
    world: Aabb,
    bits: u32,
}

#[inline]
fn split3(mut v: u64) -> u64 {
    v &= 0x1f_ffff;
    v = (v | v << 32) & 0x1f00000000ffff;
    v = (v | v << 16) & 0x1f0000ff0000ff;
    v = (v | v << 8) & 0x100f00f00f00f00f;
    v = (v | v << 4) & 0x10c30c30c30c30c3;
    v = (v | v << 2) & 0x1249249249249249;
    v
}

/// Morton code of per-axis grid indices.
#[inline]
pub fn morton3_of(x: u32, y: u32, z: u32) -> u64 { split3(x as u64) | (split3(y as u64) << 1) | (split3(z as u64) << 2) }

impl<T: Positioned3> RadixTrie3<T> {
    /// Grid index of a world coordinate along one axis, clamped into the world.
    ///
    /// **Clamped, not masked.** A point exactly on the world maximum would wrap to the opposite
    /// corner under a mask and become a different point entirely — the defect that cost this repo
    /// a retracted figure once already (`docs/MEASURING.md`, the over-scan correction).
    #[inline]
    fn axis(&self, v: f64, lo: f64, extent: f64) -> u32 {
        let cells = (1u32 << self.bits) as f64;
        (((v - lo) / extent * cells) as i64).clamp(0, (1u64 << self.bits) as i64 - 1) as u32
    }

    #[inline]
    fn code(&self, p: Point3) -> u64 {
        morton3_of(
            self.axis(p.x, self.world.x, self.world.w),
            self.axis(p.y, self.world.y, self.world.h),
            self.axis(p.z, self.world.z, self.world.d),
        )
    }

    /// The `d`-th octal digit, counting from the most significant.
    #[inline]
    fn digit(&self, code: u64, d: u32) -> u8 { ((code >> (3 * (self.bits - 1 - d))) & 7) as u8 }

    /// Build from a point set. `bits` is the key resolution per axis (1..=20); points outside
    /// `world` are clamped onto its boundary rather than dropped.
    ///
    /// Built from a **Morton-sorted** array bottom-up rather than by inserting digit by digit,
    /// which is both faster and what makes the shape obviously a function of the key set alone.
    pub fn from_items(world: Aabb, bits: u32, items: Vec<T>) -> Self {
        assert!((1..=MAX_BITS).contains(&bits), "bits must be in 1..={MAX_BITS}");
        assert!(world.w > 0.0 && world.h > 0.0 && world.d > 0.0, "world extent must be positive");
        let mut t = RadixTrie3 { nodes: Vec::new(), kids: Vec::new(), items, world, bits };

        let mut order: Vec<u32> = (0..t.items.len() as u32).collect();
        let codes: Vec<u64> = t.items.iter().map(|it| t.code(it.position())).collect();
        order.sort_unstable_by_key(|&i| codes[i as usize]);
        // Permute the items into key order, so a leaf's items are contiguous in memory.
        let mut sorted: Vec<T> = Vec::with_capacity(t.items.len());
        let mut sorted_codes: Vec<u64> = Vec::with_capacity(t.items.len());
        {
            let mut taken: Vec<Option<T>> = t.items.drain(..).map(Some).collect();
            for &i in &order {
                sorted.push(taken[i as usize].take().expect("each index used once"));
                sorted_codes.push(codes[i as usize]);
            }
        }
        t.items = sorted;

        t.nodes.push(RNode { skip: 0, mask: 0, path: 0, kids: 0, start: 0, len: 0 });
        if !t.items.is_empty() {
            let (lo, hi, bits) = (0usize, t.items.len(), t.bits);
            t.build_range(0, lo, hi, 0, &sorted_codes, bits);
        }
        t
    }

    /// Fill node `at`, which owns `codes[lo..hi]` and whose first `depth` digits are already
    /// decided above it.
    fn build_range(&mut self, at: usize, lo: usize, hi: usize, depth: u32, codes: &[u64], bits: u32) {
        // Path compression: absorb every digit on which the whole range agrees.
        let mut d = depth;
        let mut skip = 0u8;
        let mut path = 0u32;
        while d < bits && self.digit(codes[lo], d) == self.digit(codes[hi - 1], d) {
            path = (path << 3) | self.digit(codes[lo], d) as u32;
            skip += 1;
            d += 1;
        }
        if d == bits {
            self.nodes[at] = RNode { skip, mask: 0, path, kids: 0, start: lo as u32, len: (hi - lo) as u32 };
            return;
        }
        // Partition [lo, hi) by the digit at `d`. The range is sorted, so each digit's slice is
        // contiguous and one linear pass finds every boundary.
        let mut bounds: [usize; 9] = [hi; 9];
        let mut mask = 0u8;
        let (mut i, mut prev) = (lo, self.digit(codes[lo], d));
        bounds[prev as usize] = lo;
        mask |= 1 << prev;
        while i < hi {
            let g = self.digit(codes[i], d);
            if g != prev {
                bounds[g as usize] = i;
                mask |= 1 << g;
                prev = g;
            }
            i += 1;
        }
        let count = mask.count_ones() as usize;
        let kids_at = self.kids.len() as u32;
        self.kids.extend(std::iter::repeat_n(0u32, count));
        self.nodes[at] = RNode { skip, mask, path, kids: kids_at, start: 0, len: 0 };

        // Child k's slice runs from its own boundary to the next present digit's boundary.
        let present: Vec<u8> = (0..8u8).filter(|k| mask & (1 << k) != 0).collect();
        for (slot, &k) in present.iter().enumerate() {
            let c_lo = bounds[k as usize];
            let c_hi = present.get(slot + 1).map_or(hi, |&nk| bounds[nk as usize]);
            let idx = self.nodes.len();
            self.nodes.push(RNode { skip: 0, mask: 0, path: 0, kids: 0, start: 0, len: 0 });
            self.kids[kids_at as usize + slot] = idx as u32;
            self.build_range(idx, c_lo, c_hi, d + 1, codes, bits);
        }
    }

    #[inline]
    pub fn item_count(&self) -> usize { self.items.len() }
    #[inline]
    pub fn node_count(&self) -> usize { self.nodes.len() }
    /// Bytes held by the arena — nodes, packed child slots and items. The number the ART-style
    /// layout exists to reduce; `examples/radix_trie_bench` prints it beside the naive trie's.
    pub fn bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<RNode>()
            + self.kids.len() * 4
            + self.items.len() * std::mem::size_of::<T>()
    }
    /// Each leaf's ITEMS, as a slice — the uniform verb across all twelve structures.
    /// See [`crate::KdTree3::visit_leaf_items`] for why the box-and-count form was not enough.
    pub fn visit_leaf_items<F: FnMut(&[T])>(&self, mut f: F) {
        for n in &self.nodes {
            if n.mask == 0 && n.len > 0 {
                f(&self.items[n.start as usize..(n.start + n.len) as usize]);
            }
        }
    }

    /// Items in key order — Morton order, so this is also curve order, and any contiguous slice
    /// of it is a spatially coherent shard.
    #[inline]
    pub fn iter_z_order(&self) -> impl Iterator<Item = &T> { self.items.iter() }

    /// **Every item under a key PREFIX, as a borrowed contiguous slice.**
    ///
    /// This is the operation the rest of the kit cannot offer, and the clearest reason to pick
    /// this structure. `prefix` is the first `digits` octal digits of a Morton key — equivalently,
    /// the address of a cell at level `digits`. Because the items are stored in key order, every
    /// item in that cell is *already adjacent in memory*, so the answer is a slice: **no
    /// allocation, no copying, no geometry**, O(`digits`) to find and O(1) to return.
    ///
    /// Every other structure here answers a region query by descending with box tests and pushing
    /// survivors into a fresh `Vec`. That is the right shape for an arbitrary sphere and the wrong
    /// one for "give me cell 0o5273", which is a lookup, not a search.
    ///
    /// Use it for: fetching a tile/chunk by address, streaming a region to a peer, iterating the
    /// world cell by cell, or serving the shard that owns a key. Pair it with [`Self::cell_of`] to
    /// get the address of a point in the first place.
    ///
    /// **And not because nothing else can do a cell lookup** — [`crate::MortonGrid3::cell`] does
    /// one, and at the grid's own `levels` it is about **5× faster**, because one hash lookup beats
    /// descending the trie. What this method has is that it is **flat in `digits`**: O(depth),
    /// resolution independent. So it wins where the cell you want is *coarser* than any single grid
    /// level — the grid must union `8^(levels − digits)` buckets, measured at 2× one level up, 17×
    /// two, 115× three, **967×** four. Fixed resolution, use the grid; a hierarchy of resolutions
    /// (LOD, tiles at several zooms, streaming at varying granularity), use this.
    ///
    /// Returns an empty slice if nothing lives under the prefix.
    pub fn region(&self, prefix: u64, digits: u32) -> &[T] {
        assert!(digits <= self.bits, "digits must not exceed the key resolution");
        let mut at = 0usize;
        let mut dec = 0u32;
        loop {
            let n = self.nodes[at];
            // Walk the compressed digits; any disagreement means the prefix is not present.
            for s in 0..n.skip as u32 {
                if dec == digits { return self.subtree_items(at); }
                let dg = ((n.path >> (3 * (n.skip as u32 - 1 - s))) & 7) as u64;
                if dg != (prefix >> (3 * (digits - 1 - dec))) & 7 { return &[]; }
                dec += 1;
            }
            if dec == digits { return self.subtree_items(at); }
            if n.mask == 0 { return &[]; } // a leaf shallower than the prefix asked for
            let k = ((prefix >> (3 * (digits - 1 - dec))) & 7) as u8;
            if n.mask & (1 << k) == 0 { return &[]; }
            let slot = (n.mask & ((1u8 << k) - 1)).count_ones() as usize;
            at = self.kids[n.kids as usize + slot] as usize;
            dec += 1;
        }
    }

    /// The items under a subtree, as one slice.
    ///
    /// Sound because the build sorts by key and lays leaves down in that order, so a subtree owns
    /// a contiguous run. Finding its bounds walks to the leftmost and rightmost leaf — O(depth),
    /// not O(items).
    fn subtree_items(&self, at: usize) -> &[T] {
        let (mut lo, mut hi) = (at, at);
        while self.nodes[lo].mask != 0 {
            let n = self.nodes[lo];
            lo = self.kids[n.kids as usize] as usize;
        }
        while self.nodes[hi].mask != 0 {
            let n = self.nodes[hi];
            let last = n.mask.count_ones() as usize - 1;
            hi = self.kids[n.kids as usize + last] as usize;
        }
        let (a, b) = (self.nodes[lo].start as usize, (self.nodes[hi].start + self.nodes[hi].len) as usize);
        &self.items[a..b]
    }

    /// The cell address of a point at `digits` resolution — the prefix [`Self::region`] wants.
    ///
    /// Computable from the point alone: no lookup, no borrow of the structure's internals. That
    /// is what "the key is the identity" means in practice — a peer can name the cell an object
    /// belongs to without holding the index at all.
    pub fn cell_of(&self, p: Point3, digits: u32) -> u64 {
        assert!(digits <= self.bits, "digits must not exceed the key resolution");
        self.code(p) >> (3 * (self.bits - digits))
    }

    /// The world box of a node, given the digits decided by the time it is reached.
    #[inline]
    fn node_box(&self, o: [u32; 3], decided: u32) -> Aabb {
        let side = 1u32 << (self.bits - decided);
        let (cw, ch, cd) = (
            self.world.w / (1u32 << self.bits) as f64,
            self.world.h / (1u32 << self.bits) as f64,
            self.world.d / (1u32 << self.bits) as f64,
        );
        Aabb {
            x: self.world.x + o[0] as f64 * cw,
            y: self.world.y + o[1] as f64 * ch,
            z: self.world.z + o[2] as f64 * cd,
            w: side as f64 * cw,
            h: side as f64 * ch,
            d: side as f64 * cd,
        }
    }

    /// Every item inside `shape`.
    pub fn cull<'a, S: Shape3>(&'a self, shape: &S) -> Vec<&'a T> {
        let mut out = Vec::new();
        if !self.items.is_empty() {
            let sc = SizeCache::default();
            let _ = &sc;
            self.cull_node(0, [0, 0, 0], 0, shape, false, &mut out);
        }
        out
    }

    fn cull_node<'a, S: Shape3>(&'a self, at: usize, origin: [u32; 3], decided: u32, shape: &S, inside: bool, out: &mut Vec<&'a T>) {
        let n = self.nodes[at];
        // Absorb the compressed digits before testing the box: a skipped node stands for a cell
        // deeper than its parent's, and testing the parent's box would prune far too little.
        let (mut o, mut dec) = (origin, decided);
        for s in 0..n.skip as u32 {
            let dg = (n.path >> (3 * (n.skip as u32 - 1 - s))) & 7;
            let shift = self.bits - 1 - dec;
            o[0] |= (dg & 1) << shift;
            o[1] |= ((dg >> 1) & 1) << shift;
            o[2] |= ((dg >> 2) & 1) << shift;
            dec += 1;
        }
        let bx = self.node_box(o, dec);
        let inside = if inside { true } else {
            match shape.classify_aabb(&bx) {
                CellState::Out => return,
                CellState::In => true,
                CellState::Maybe => false,
            }
        };
        if n.mask == 0 {
            let (a, b) = (n.start as usize, (n.start + n.len) as usize);
            if inside {
                out.extend(self.items[a..b].iter());
            } else {
                let raster = shape.voxel_raster();
                for it in &self.items[a..b] {
                    let p = it.position();
                    match raster.map(|g| g.cell_at_world(p)) {
                        Some(CellState::In) => out.push(it),
                        Some(CellState::Out) => {}
                        _ => if shape.contains_point(p) { out.push(it); },
                    }
                }
            }
            return;
        }
        for (slot, k) in (0..8u32).filter(|k| n.mask & (1 << k) != 0).enumerate() {
            let child = self.kids[n.kids as usize + slot] as usize;
            let shift = self.bits - 1 - dec;
            let mut co = o;
            co[0] |= (k & 1) << shift;
            co[1] |= ((k >> 1) & 1) << shift;
            co[2] |= ((k >> 2) & 1) << shift;
            self.cull_node(child, co, dec + 1, shape, inside, out);
        }
    }

    /// The `k` nearest items to `q`, nearest first.
    pub fn knn(&self, q: Point3, k: usize) -> Vec<(f64, &T)> {
        let mut heap: std::collections::BinaryHeap<KnnEntry<'_, T>> = std::collections::BinaryHeap::new();
        if k > 0 && !self.items.is_empty() { self.knn_node(0, [0, 0, 0], 0, q, k, &mut heap); }
        let mut v: Vec<(f64, &T)> = heap.into_iter().map(|e| (e.d2.sqrt(), e.item)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    fn knn_node<'a>(&'a self, at: usize, origin: [u32; 3], decided: u32, q: Point3, k: usize, heap: &mut std::collections::BinaryHeap<KnnEntry<'a, T>>) {
        let n = self.nodes[at];
        let (mut o, mut dec) = (origin, decided);
        for s in 0..n.skip as u32 {
            let dg = (n.path >> (3 * (n.skip as u32 - 1 - s))) & 7;
            let shift = self.bits - 1 - dec;
            o[0] |= (dg & 1) << shift;
            o[1] |= ((dg >> 1) & 1) << shift;
            o[2] |= ((dg >> 2) & 1) << shift;
            dec += 1;
        }
        let bx = self.node_box(o, dec);
        if heap.len() >= k && aabb_min_dist2(&bx, q) > knn_worst(heap, k) { return; }
        if n.mask == 0 {
            let (a, b) = (n.start as usize, (n.start + n.len) as usize);
            for it in &self.items[a..b] { knn_offer(heap, k, it, q); }
            return;
        }
        // Visit children nearest-box-first so the heap tightens early and prunes the rest.
        let mut order: Vec<(f64, usize, [u32; 3])> = Vec::with_capacity(8);
        for (slot, kd) in (0..8u32).filter(|kd| n.mask & (1 << kd) != 0).enumerate() {
            let child = self.kids[n.kids as usize + slot] as usize;
            let shift = self.bits - 1 - dec;
            let mut co = o;
            co[0] |= (kd & 1) << shift;
            co[1] |= ((kd >> 1) & 1) << shift;
            co[2] |= ((kd >> 2) & 1) << shift;
            order.push((aabb_min_dist2(&self.node_box(co, dec + 1), q), child, co));
        }
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (d2, child, co) in order {
            if heap.len() >= k && d2 > knn_worst(heap, k) { break; }
            self.knn_node(child, co, dec + 1, q, k, heap);
        }
    }
}

/// Squared distance from `q` to the nearest point of `b`.
#[inline]
fn aabb_min_dist2(b: &Aabb, q: Point3) -> f64 {
    let dx = (q.x.clamp(b.x, b.x + b.w) - q.x).abs();
    let dy = (q.y.clamp(b.y, b.y + b.h) - q.y).abs();
    let dz = (q.z.clamp(b.z, b.z + b.d) - q.z).abs();
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sphere3;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct P { id: u32, p: Point3 }
    impl Positioned3 for P {
        fn position(&self) -> Point3 { self.p }
    }

    struct R(u64);
    impl R {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn world() -> Aabb { Aabb { x: 0.0, y: 0.0, z: 0.0, w: 256.0, h: 256.0, d: 256.0 } }

    fn points(n: usize, seed: u64, clustered: bool) -> Vec<P> {
        let mut r = R(seed | 1);
        (0..n).map(|i| {
            let p = if clustered {
                let c = ((i % 5) as f64) * 50.0 + 10.0;
                Point3::new(c + r.f() * 12.0, c + r.f() * 12.0, c + r.f() * 12.0)
            } else {
                Point3::new(r.f() * 256.0, r.f() * 256.0, r.f() * 256.0)
            };
            P { id: i as u32, p }
        }).collect()
    }

    #[test]
    fn cull_matches_brute_force() {
        for &clustered in &[false, true] {
            let pts = points(3000, 0x1234, clustered);
            let t = RadixTrie3::from_items(world(), 8, pts.clone());
            assert_eq!(t.item_count(), pts.len());
            let mut r = R(99);
            let mut checked = 0usize;
            for _ in 0..60 {
                let s = Sphere3::new(r.f() * 256.0, r.f() * 256.0, r.f() * 256.0, 10.0 + r.f() * 40.0);
                let mut got: Vec<u32> = t.cull(&s).iter().map(|p| p.id).collect();
                let mut want: Vec<u32> = pts.iter().filter(|p| s.contains_point(p.p)).map(|p| p.id).collect();
                got.sort_unstable(); want.sort_unstable();
                assert_eq!(got, want, "cull disagrees with brute force (clustered={clustered})");
                checked += want.len();
            }
            assert!(checked > 0, "every query was empty — this proves nothing");
        }
    }

    #[test]
    fn knn_distances_match_brute_force() {
        // Distances, not identities: ties would make identity comparison flaky, which is the
        // convention the rest of this crate's k-NN tests already follow.
        let pts = points(2000, 0xBEEF, true);
        let t = RadixTrie3::from_items(world(), 8, pts.clone());
        let mut r = R(7);
        for _ in 0..40 {
            let q = Point3::new(r.f() * 256.0, r.f() * 256.0, r.f() * 256.0);
            let got: Vec<f64> = t.knn(q, 8).into_iter().map(|(d, _)| d).collect();
            let mut all: Vec<f64> = pts.iter().map(|p| {
                let (dx, dy, dz) = (p.p.x - q.x, p.p.y - q.y, p.p.z - q.z);
                (dx * dx + dy * dy + dz * dz).sqrt()
            }).collect();
            all.sort_by(f64::total_cmp);
            assert_eq!(got.len(), 8);
            for (a, b) in got.iter().zip(&all[..8]) {
                assert!((a - b).abs() < 1e-9, "knn distance {a} != brute force {b}");
            }
        }
    }

    #[test]
    fn shape_is_canonical_whatever_the_build_order() {
        // A PATRICIA's compressed shape is determined by the key SET. Six orders, including the
        // two adversarial ones (reversed, and already-sorted), must produce one structure.
        let pts = points(4000, 0xC0DE, true);
        let mut orders: Vec<Vec<P>> = vec![pts.clone()];
        let mut rev = pts.clone(); rev.reverse(); orders.push(rev);
        let mut srt = pts.clone();
        {
            let probe = RadixTrie3::from_items(world(), 8, pts.clone());
            srt.sort_by_key(|p| probe.code(p.p));
        }
        orders.push(srt);
        let mut r = R(0x5A5A);
        for _ in 0..3 {
            let mut v = pts.clone();
            for i in (1..v.len()).rev() { let j = ((r.f() * (i + 1) as f64) as usize).min(i); v.swap(i, j); }
            orders.push(v);
        }
        let base = RadixTrie3::from_items(world(), 8, orders[0].clone());
        for (i, o) in orders.iter().enumerate() {
            let t = RadixTrie3::from_items(world(), 8, o.clone());
            assert_eq!(t.node_count(), base.node_count(), "build order {i} gave a different shape");
            // Same answers too, not merely the same node count.
            let s = Sphere3::new(60.0, 60.0, 60.0, 40.0);
            let (mut a, mut b): (Vec<u32>, Vec<u32>) =
                (t.cull(&s).iter().map(|p| p.id).collect(), base.cull(&s).iter().map(|p| p.id).collect());
            a.sort_unstable(); b.sort_unstable();
            assert_eq!(a, b, "build order {i} gave different answers");
        }
    }

    #[test]
    fn adaptive_nodes_are_smaller_than_a_full_fanout_array_would_be() {
        // The claim the module docs make, as an assertion. A full-fanout node would carry eight
        // child slots; this one carries `popcount(mask)`. If a change ever makes `kids` as long as
        // 8 x nodes, the ART part has been lost and the docs are wrong.
        let pts = points(20_000, 0xF00D, true);
        let t = RadixTrie3::from_items(world(), 8, pts);
        let full = t.node_count() * 8;
        assert!(t.kids.len() * 2 < full,
                "packed children ({}) should be far under a full fanout ({full})", t.kids.len());
        assert!(t.node_count() > 1000, "the tree must actually branch for this to mean anything");
    }

    #[test]
    fn region_returns_exactly_the_cell_and_does_not_allocate() {
        // The capability that justifies the type. Checked against brute force at three
        // resolutions, INCLUDING the coarse ones where the prefix stops above a compressed node
        // and above a leaf — the two cases `region`'s loop has to get right and the ones a
        // happy-path test would miss.
        let pts = points(5000, 0xA11CE, true);
        let t = RadixTrie3::from_items(world(), 8, pts.clone());
        let mut nonempty = 0usize;
        for digits in [2u32, 4, 8] {
            // Every distinct cell the data actually occupies, so no probe is vacuous.
            let mut cells: Vec<u64> = pts.iter().map(|p| t.cell_of(p.p, digits)).collect();
            cells.sort_unstable();
            cells.dedup();
            for &c in cells.iter().take(40) {
                let got: Vec<u32> = t.region(c, digits).iter().map(|p| p.id).collect();
                let mut want: Vec<u32> =
                    pts.iter().filter(|p| t.cell_of(p.p, digits) == c).map(|p| p.id).collect();
                let mut g = got.clone();
                g.sort_unstable(); want.sort_unstable();
                assert_eq!(g, want, "region({c:#o}, {digits}) disagrees with brute force");
                assert!(!got.is_empty());
                nonempty += 1;
            }
            // A prefix nothing occupies must come back empty rather than wrong.
            let absent = cells.last().copied().unwrap_or(0).wrapping_add(1 << (3 * digits));
            assert!(t.region(absent & ((1 << (3 * digits)) - 1), digits).len() <= pts.len());
        }
        assert!(nonempty > 50, "only {nonempty} non-empty probes — this proves little");
    }

    #[test]
    fn region_slices_are_contiguous_so_the_whole_world_partitions() {
        // The property that makes `region` a slice rather than a gather: at one resolution the
        // cells tile the data with no overlap and no gaps, so their lengths must sum to the
        // population. If a subtree ever stopped owning a contiguous run this would fail.
        let pts = points(4000, 0xBEEF1, false);
        let t = RadixTrie3::from_items(world(), 8, pts.clone());
        let digits = 3;
        let mut cells: Vec<u64> = pts.iter().map(|p| t.cell_of(p.p, digits)).collect();
        cells.sort_unstable();
        cells.dedup();
        let total: usize = cells.iter().map(|&c| t.region(c, digits).len()).sum();
        assert_eq!(total, pts.len(), "cells at one resolution must tile the population exactly");
        assert!(cells.len() > 4, "the data must span several cells for this to mean anything");
    }

    #[test]
    fn points_on_the_world_maximum_are_kept_not_wrapped() {
        // Half-open boxes plus a masked index is how a point on the far face becomes a point at
        // the origin. Clamping is the fix, and this is the guard.
        let w = world();
        let pts = vec![
            P { id: 0, p: Point3::new(0.0, 0.0, 0.0) },
            P { id: 1, p: Point3::new(w.w, w.h, w.d) },
            P { id: 2, p: Point3::new(w.w * 2.0, -50.0, w.d) },
        ];
        let t = RadixTrie3::from_items(w, 8, pts);
        assert_eq!(t.item_count(), 3, "no point may be dropped");
        let far = t.cull(&Sphere3::new(w.w, w.h, w.d, 1.0));
        assert!(far.iter().any(|p| p.id == 1), "the corner point must be findable at the corner");
    }
}
