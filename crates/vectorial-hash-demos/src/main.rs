use vectorial_hash::{Point, Positioned, Rect, Shape, TemplateGrid, Tree};
use vectorial_hash_templates::adapter::matrix_to_template_grid;
use vectorial_hash_templates::polygon::{create_drop, rotated_copy, scaled_copy, Polygon};
use vectorial_hash_templates::templates::{
    angle_to_radians, get_template_grid_fast, TemplateStore,
};

fn main() {
    demo_dedup_8_symmetry();
    println!();
    demo_end_to_end_cull();
}

/// Original demo: feed a polygon at 4 angles into the dedup store and watch the
/// 8-symmetry collapse fold variants into a single canonical template.
fn demo_dedup_8_symmetry() {
    println!("== Demo 1: tiny in-memory template generation ==");

    let poly = create_drop(0.2, 0.8);
    let scaled = scaled_copy(&poly, 64.0, 64.0);
    let store = TemplateStore::new();
    let grid: i64 = 16;

    let angles = [0.0, 45.0, 90.0, 135.0];
    for angle in angles {
        let rotated = rotated_copy(&scaled, angle_to_radians(angle));
        let gxr = [
            (rotated.x_min / grid as f64).floor() as i64,
            (rotated.x_max / grid as f64).ceil() as i64,
        ];
        let gyr = [
            (rotated.y_min / grid as f64).floor() as i64,
            (rotated.y_max / grid as f64).ceil() as i64,
        ];

        let tpl = get_template_grid_fast(gxr[0], gyr[0], gxr[1], gyr[1], grid, grid, &rotated);
        let (id, op, is_new) = store.store_dedup(&tpl, &format!("drop-a{}", angle));
        println!("  angle {:>5.1}deg -> id {} via {} (new: {})", angle, id, op, is_new);
    }

    println!("Unique templates: {}", store.template_count());
}

/// End-to-end demo: build a real polygon, derive its TemplateGrid via the
/// generator + adapter, wrap both in a `Shape`, and cull a `Tree` of synthetic
/// points. Cross-check that the cull result matches a brute-force point-in-poly
/// sweep so we know the full pipeline is correct.
fn demo_end_to_end_cull() {
    println!("== Demo 2: polygon -> template -> Shape -> Tree::cull ==");

    // 1. Polygon (scale 64, no rotation), then shifted into positive space so
    //    everything sits in a clean root rect.
    let poly = scaled_copy(&create_drop(0.2, 0.8), 64.0, 64.0);
    let mut poly = poly;
    poly.move_by(100.0, 30.0);
    println!(
        "  polygon bbox: x[{:.1},{:.1}] y[{:.1},{:.1}]",
        poly.x_min, poly.x_max, poly.y_min, poly.y_max
    );

    // 2. Generator-side template (Matrix of OUT/MAYBE/IN cells).
    let cell: i64 = 8;
    let gxr = [
        (poly.x_min / cell as f64).floor() as i64,
        (poly.x_max / cell as f64).ceil() as i64,
    ];
    let gyr = [
        (poly.y_min / cell as f64).floor() as i64,
        (poly.y_max / cell as f64).ceil() as i64,
    ];
    let matrix = get_template_grid_fast(gxr[0], gyr[0], gxr[1], gyr[1], cell, cell, &poly);

    // 3. Adapter: Matrix -> runtime TemplateGrid.
    let anchor = Point::new(gxr[0] as f64 * cell as f64, gyr[0] as f64 * cell as f64);
    let grid =
        matrix_to_template_grid(&matrix, anchor, cell as f64, cell as f64);
    println!(
        "  template grid: {}x{} cells of {}x{} anchored at ({:.1}, {:.1})",
        grid.cols, grid.rows, grid.cell_w, grid.cell_h, grid.origin_x, grid.origin_y
    );

    // 4. Shape wraps both: polygon for the per-point fallback, grid for the
    //    green/yellow/white short-circuit at internal tree nodes.
    let shape = PolygonShape { poly: poly.clone(), grid };

    // 5. Populate a Tree with a dense lattice of points across the polygon
    //    bbox plus some outside, then cull.
    let root = Rect::new(0.0, 0.0, 256.0, 256.0);
    let mut tree: Tree<Pt> = Tree::new(root, 8);
    let mut total = 0;
    for ix in 0..40 {
        for iy in 0..40 {
            let p = Point::new(ix as f64 * 6.0 + 3.0, iy as f64 * 6.0 + 3.0);
            tree.insert(Pt(p));
            total += 1;
        }
    }

    let hits = tree.cull(&shape);

    // 6. Brute-force check: point-in-polygon for every inserted point.
    let mut brute = Vec::new();
    for ix in 0..40 {
        for iy in 0..40 {
            let p = Point::new(ix as f64 * 6.0 + 3.0, iy as f64 * 6.0 + 3.0);
            if shape.poly.is_inside(p.x, p.y) {
                brute.push(p);
            }
        }
    }

    println!(
        "  tree size: {} pts | cull hits: {} | brute-force inside: {}",
        total,
        hits.len(),
        brute.len()
    );

    let mut cull_pts: Vec<Point> = hits.iter().map(|p| p.0).collect();
    cull_pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    brute.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));

    assert_eq!(cull_pts, brute, "cull result must match brute-force inside-test");
    println!("  cull == brute-force: OK");
}

#[derive(Clone, Copy)]
struct Pt(Point);

impl Positioned for Pt {
    fn position(&self) -> Point {
        self.0
    }
}

struct PolygonShape {
    poly: Polygon,
    grid: TemplateGrid,
}

impl Shape for PolygonShape {
    fn bounding_box(&self) -> Rect {
        Rect::new(
            self.poly.x_min,
            self.poly.y_min,
            self.poly.x_max - self.poly.x_min,
            self.poly.y_max - self.poly.y_min,
        )
    }

    fn contains_point(&self, p: Point) -> bool {
        self.poly.is_inside(p.x, p.y)
    }

    fn template_grid(&self) -> Option<&TemplateGrid> {
        Some(&self.grid)
    }
}
