//! Exhaustive culling validation campaign.
//!
//! Property/fuzz-style: deterministic seeds generate random scenarios —
//! churned trees (inserts, removes, relocations, merges) × random figures
//! (drop / circle / box / rectangle), scales, angles and integer origins —
//! and **every cull configuration must return exactly the same item set as
//! brute force** (`polygon::is_inside` over every live item):
//!
//! 1. plain shape (bbox + exact geometry only)
//! 2. single fixed grid (`Shape::template_grid`, `classify_region` path)
//! 3. template bank per-cell-size selection, without the 1×1 raster
//! 4. template bank + 1×1 raster
//! 5. `cull_walk` with Samet ascent neighbours
//! 6. `cull_walk` with locate-probe neighbours
//! 7. `cull_walk` with stored ropes (with `--features neighbors`)
//!
//! Boundary cases are seeded on purpose: items on cell-edge lattices and at
//! integer coordinates, box figures with integer dimensions at integer
//! origins (so figure edges coincide exactly with cell edges and items).
//!
//! **Exactness contract**: results are exact for every item farther than
//! ~2×EPSILON (the intersector's 1e-5 tolerance) from the figure's boundary.
//! Items inside that epsilon halo may classify either way: `is_inside`
//! counts on-edge points (within 1e-5) as inside, while bounding boxes and
//! template cells are computed with exact geometry — a discrepancy inherited
//! from the original generator's epsilon-robust predicates. The campaign
//! asserts strict equality and only tolerates mismatches that are provably
//! boundary-fuzzy.
//!
//! Run: `cargo test -p vectorial-hash-templates --test exhaustive_culling`
//! (add `--features neighbors` to cover the ropes strategy, and
//! `-- --ignored` for the long campaign).

// Scenario builders keep an explicit entity-id counter (the id is a payload,
// not just the loop index), so the counter-loop lint doesn't apply.
#![allow(clippy::explicit_counter_loop)]

use vectorial_hash::{
    PlacedTemplate, Point, Positioned, Rect, Shape, TemplateGrid, Tree, WalkNeighbors,
};
use vectorial_hash_templates::adapter::matrix_to_template_grid;
use vectorial_hash_templates::bank::{FigureKey, TemplateBank};
use vectorial_hash_templates::polygon::{
    create_box, create_circle, create_drop, create_square, rotated_copy, scaled_copy, Polygon,
};
use vectorial_hash_templates::templates::{angle_to_radians, get_template_grid_fast};

const WORLD: f64 = 256.0;

#[derive(Clone, Copy, Debug)]
struct It {
    id: u32,
    pos: Point,
}

impl Positioned for It {
    fn position(&self) -> Point {
        self.pos
    }
}

/// xorshift64* — deterministic scenarios.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

struct PlainShape {
    poly: Polygon,
    bbox: Rect,
}
impl Shape for PlainShape {
    fn bounding_box(&self) -> Rect {
        self.bbox
    }
    fn contains_point(&self, p: Point) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
}

struct SingleGridShape {
    poly: Polygon,
    bbox: Rect,
    grid: TemplateGrid,
}
impl Shape for SingleGridShape {
    fn bounding_box(&self) -> Rect {
        self.bbox
    }
    fn contains_point(&self, p: Point) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
    fn template_grid(&self) -> Option<&TemplateGrid> {
        Some(&self.grid)
    }
}

struct BankShape<'a> {
    bank: &'a TemplateBank,
    figure: FigureKey,
    angle: f64,
    origin: (i64, i64),
    poly: Polygon,
    bbox: Rect,
    raster: Option<PlacedTemplate>,
}
impl Shape for BankShape<'_> {
    fn bounding_box(&self) -> Rect {
        self.bbox
    }
    fn contains_point(&self, p: Point) -> bool {
        self.poly.is_inside(p.x, p.y)
    }
    fn template_for_cell(&self, cell_w: f64, cell_h: f64) -> Option<PlacedTemplate> {
        if cell_w.fract() != 0.0 || cell_h.fract() != 0.0 {
            return None;
        }
        // Aggregated fallback is exact, so the campaign also covers it.
        self.bank.placed_for_or_aggregated(
            &self.figure,
            cell_w as u32,
            cell_h as u32,
            self.angle,
            self.origin,
        )
    }
    fn point_template(&self) -> Option<&PlacedTemplate> {
        self.raster.as_ref()
    }
}

fn poly_bbox(poly: &Polygon) -> Rect {
    Rect::new(
        poly.x_min,
        poly.y_min,
        poly.x_max - poly.x_min,
        poly.y_max - poly.y_min,
    )
}

fn sorted_ids(items: Vec<&It>) -> Vec<u32> {
    let mut ids: Vec<u32> = items.into_iter().map(|i| i.id).collect();
    ids.sort_unstable();
    ids
}

/// True when `p` sits within `tol` of the polygon boundary: probing points
/// around it yields both inside and outside answers.
fn is_boundary_fuzzy(poly: &Polygon, p: Point, tol: f64) -> bool {
    let probes = [
        (-tol, 0.0), (tol, 0.0), (0.0, -tol), (0.0, tol),
        (-tol, -tol), (tol, tol), (-tol, tol), (tol, -tol),
    ];
    let mut inside = 0;
    let mut outside = 0;
    for (dx, dy) in probes {
        if poly.is_inside(p.x + dx, p.y + dy) {
            inside += 1;
        } else {
            outside += 1;
        }
    }
    inside > 0 && outside > 0
}

/// Strict equality, except items provably on the epsilon halo of the
/// boundary, which may classify either way (see the contract above).
fn assert_matches(
    name: &str,
    got: Vec<u32>,
    expected: &[u32],
    alive: &[It],
    poly: &Polygon,
    ctx: &str,
) {
    if got == expected {
        return;
    }
    let gs: std::collections::HashSet<u32> = got.iter().copied().collect();
    let es: std::collections::HashSet<u32> = expected.iter().copied().collect();
    for id in gs.symmetric_difference(&es) {
        let p = alive
            .iter()
            .find(|it| it.id == *id)
            .unwrap_or_else(|| panic!("{name} | {ctx}: unknown id {id}"))
            .pos;
        assert!(
            is_boundary_fuzzy(poly, p, 2e-5),
            "{name} | {ctx}: item {id} at {p:?} misclassified and NOT boundary-fuzzy\n  got {got:?}\n  expected {expected:?}",
        );
    }
}

/// One full random scenario; panics with seed context on any mismatch.
fn run_scenario(seed: u64) {
    let mut rng = Rng(seed.max(1));

    // --- tree with churn ---
    let item_limit = 1 + rng.below(6) as usize;
    let merge_limit = 1 + rng.below(item_limit as u64) as usize;
    let mut tree: Tree<It> =
        Tree::with_limits(Rect::new(0.0, 0.0, WORLD, WORLD), item_limit, merge_limit);
    let mut alive: Vec<It> = Vec::new();
    let mut next_id = 0u32;

    let n = 50 + rng.below(300) as usize;
    for _ in 0..n {
        let pos = match rng.below(100) {
            // Cell-edge lattice (multiples of 8): collides with split lines.
            0..=14 => Point::new(
                ((rng.below(33) * 8) as f64).min(WORLD - 1e-9),
                ((rng.below(33) * 8) as f64).min(WORLD - 1e-9),
            ),
            // Integer coordinates: collide with figure edges at integer origins.
            15..=29 => Point::new(rng.below(256) as f64, rng.below(256) as f64),
            // Uniform floats.
            _ => Point::new(rng.unit() * WORLD, rng.unit() * WORLD),
        };
        let it = It { id: next_id, pos };
        next_id += 1;
        if tree.insert(it) {
            alive.push(it);
        }
    }

    // Churn: removals and relocations exercise merge-up and re-insertion.
    let churn = alive.len() / 3;
    for _ in 0..churn {
        if alive.is_empty() {
            break;
        }
        let idx = rng.below(alive.len() as u64) as usize;
        if rng.below(2) == 0 {
            let victim = alive.swap_remove(idx);
            let removed = tree.remove(victim.pos, |c| c.id == victim.id);
            assert!(removed.is_some(), "seed {seed}: failed to remove {victim:?}");
        } else {
            let new_pos = Point::new(rng.unit() * WORLD, rng.unit() * WORLD);
            let old = alive[idx];
            let ok = tree.update(old.pos, |c| c.id == old.id, |c| c.pos = new_pos);
            assert!(ok, "seed {seed}: failed to update {old:?}");
            alive[idx].pos = new_pos;
        }
    }

    // --- random figure ---
    let kind = rng.below(4);
    let angle = if kind == 1 { 0.0 } else { (rng.below(24) as f64) * 15.0 };
    let (figure, base, dims): (u32, Polygon, Vec<f64>) = match kind {
        0 => {
            let s = 20.0 + rng.unit() * 100.0;
            let base = scaled_copy(&create_drop(0.2, 0.8), s, s);
            (0, base, vec![0.2 * s, 0.8 * s])
        }
        1 => {
            let r = 10.0 + rng.unit() * 60.0;
            let base = scaled_copy(&create_circle(1.0), r, r);
            (1, base, vec![r])
        }
        2 => {
            // Box with integer side: edges collide with lattices and items.
            let side = (16 + rng.below(80)) as f64;
            (2, create_box(side), vec![side])
        }
        _ => {
            let w = 10.0 + rng.unit() * 90.0;
            let h = 10.0 + rng.unit() * 90.0;
            let base = scaled_copy(&create_square(0.0, 0.0, 1.0, 1.0), w, h);
            (3, base, vec![w, h])
        }
    };
    let figure_key = FigureKey::new(figure, &dims);
    let rotated = rotated_copy(&base, angle_to_radians(angle));
    let origin = (
        40 + rng.below(176) as i64, // keeps most figures inside the world
        40 + rng.below(176) as i64,
    );
    let mut poly = rotated.clone();
    poly.move_by(origin.0 as f64, origin.1 as f64);
    let bbox = poly_bbox(&poly);

    // --- template bank for this figure + angle ---
    let mut bank = TemplateBank::new();
    for &(w, h) in &[(4u32, 4u32), (8, 8), (16, 16), (8, 16), (16, 8), (1, 1)] {
        bank.generate_size(&figure_key, &base, &[angle], w, h);
    }
    let raster = bank.placed_raster(&figure_key, angle, origin);
    assert!(raster.is_some(), "seed {seed}: raster missing");

    // --- oracle: brute force over every live item ---
    let mut expected: Vec<u32> = alive
        .iter()
        .filter(|it| poly.is_inside(it.pos.x, it.pos.y))
        .map(|it| it.id)
        .collect();
    expected.sort_unstable();

    let ctx = format!(
        "seed {seed}: kind {kind} dims {dims:?} angle {angle} origin {origin:?} \
         items {} limits {item_limit}/{merge_limit}",
        alive.len(),
    );

    // 1. plain
    let plain = PlainShape { poly: poly.clone(), bbox };
    assert_matches("plain", sorted_ids(tree.cull(&plain)), &expected, &alive, &poly, &ctx);

    // 2. single fixed grid (8px cells over the moved polygon)
    let cell = 8i64;
    let gx = [
        (poly.x_min / cell as f64).floor() as i64,
        (poly.x_max / cell as f64).ceil() as i64,
    ];
    let gy = [
        (poly.y_min / cell as f64).floor() as i64,
        (poly.y_max / cell as f64).ceil() as i64,
    ];
    let m = get_template_grid_fast(gx[0], gy[0], gx[1], gy[1], cell, cell, &poly);
    let grid = matrix_to_template_grid(
        &m,
        Point::new(gx[0] as f64 * cell as f64, gy[0] as f64 * cell as f64),
        cell as f64,
        cell as f64,
    );
    let single = SingleGridShape { poly: poly.clone(), bbox, grid };
    assert_matches("single-grid", sorted_ids(tree.cull(&single)), &expected, &alive, &poly, &ctx);

    // 3. bank without raster
    let bank_plain = BankShape {
        bank: &bank,
        figure: figure_key.clone(),
        angle,
        origin,
        poly: poly.clone(),
        bbox,
        raster: None,
    };
    assert_matches("bank no-raster", sorted_ids(tree.cull(&bank_plain)), &expected, &alive, &poly, &ctx);

    // 4. bank + raster
    let bank_full = BankShape { raster: raster.clone(), ..bank_plain };
    assert_matches("bank+raster", sorted_ids(tree.cull(&bank_full)), &expected, &alive, &poly, &ctx);

    // 5-7. flood-fill walk, every neighbour strategy. The seed must lie
    // inside the figure: use the vertex centroid (all our figures are convex).
    let (mut cx, mut cy) = (0.0, 0.0);
    for v in &poly.vertices {
        cx += v.x;
        cy += v.y;
    }
    let centroid = Point::new(cx / poly.vertices.len() as f64, cy / poly.vertices.len() as f64);
    if poly.is_inside(centroid.x, centroid.y)
        && Rect::new(0.0, 0.0, WORLD, WORLD).contains(centroid)
    {
        let strategies: Vec<(&str, WalkNeighbors)> = vec![
            ("walk-samet", WalkNeighbors::Samet),
            ("walk-probe", WalkNeighbors::Probe),
            #[cfg(feature = "neighbors")]
            ("walk-ropes", WalkNeighbors::Ropes),
        ];
        for (name, strategy) in strategies {
            assert_matches(
                name,
                sorted_ids(tree.cull_walk(&bank_full, centroid, strategy)),
                &expected,
                &alive,
                &poly,
                &ctx,
            );
        }
    }
}

#[test]
fn culling_campaign_quick() {
    for seed in 1..=40 {
        run_scenario(seed);
    }
}

/// Long campaign: `cargo test -p vectorial-hash-templates --test
/// exhaustive_culling -- --ignored` (and `--features neighbors`).
#[test]
#[ignore]
fn culling_campaign_long() {
    for seed in 1..=2000 {
        run_scenario(seed);
    }
}
