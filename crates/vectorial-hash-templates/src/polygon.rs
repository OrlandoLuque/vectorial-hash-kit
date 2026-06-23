use crate::vertex::Vertex;
use crate::intersector;

/// A polygon represented as a Vec of vertices (contiguous memory, cache-friendly)
#[derive(Clone, Debug)]
pub struct Polygon {
    pub vertices: Vec<Vertex>,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl Polygon {
    pub fn new() -> Self {
        Polygon {
            vertices: Vec::new(),
            x_min: f64::MAX, x_max: f64::MIN,
            y_min: f64::MAX, y_max: f64::MIN,
        }
    }

    pub fn addv(&mut self, x: f64, y: f64) {
        self.vertices.push(Vertex::new(x, y));
        self.update_bounds(x, y);
    }

    pub fn addv_arc(&mut self, x: f64, y: f64, xc: f64, yc: f64, d: i8) {
        self.vertices.push(Vertex::new_with_arc(x, y, xc, yc, d));
        // Arc extents will be computed on first recalc_bounds call
        self.update_bounds(x, y);
    }

    /// Must be called after all vertices are added for arc-containing polygons
    pub fn finalize_bounds(&mut self) {
        if self.vertices.iter().any(|v| v.seg.d != 0) {
            self.recalc_bounds();
        }
    }

    fn update_bounds(&mut self, x: f64, y: f64) {
        if x < self.x_min { self.x_min = x; }
        if x > self.x_max { self.x_max = x; }
        if y < self.y_min { self.y_min = y; }
        if y > self.y_max { self.y_max = y; }
    }

    pub fn recalc_bounds(&mut self) {
        self.x_min = f64::MAX; self.x_max = f64::MIN;
        self.y_min = f64::MAX; self.y_max = f64::MIN;
        let n = self.vertices.len();
        for i in 0..n {
            let x = self.vertices[i].x;
            let y = self.vertices[i].y;
            if x < self.x_min { self.x_min = x; }
            if x > self.x_max { self.x_max = x; }
            if y < self.y_min { self.y_min = y; }
            if y > self.y_max { self.y_max = y; }

            // Extend bounds for arc segments
            if self.vertices[i].seg.d != 0 {
                let j = (i + 1) % n;
                let xc = self.vertices[i].seg.xc;
                let yc = self.vertices[i].seg.yc;
                let r = intersector::dist(self.vertices[i].x, self.vertices[i].y, xc, yc);
                // Check if cardinal directions (0°, 90°, 180°, 270°) fall within the arc
                let a_start = intersector::angle(xc, yc, self.vertices[i].x, self.vertices[i].y);
                let a_end = intersector::angle(xc, yc, self.vertices[j].x, self.vertices[j].y);
                let cardinal_angles = [0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, 3.0 * std::f64::consts::FRAC_PI_2];
                let cardinal_offsets = [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)];
                for k in 0..4 {
                    let ca = cardinal_angles[k];
                    let in_arc = if self.vertices[i].seg.d == -1 {
                        // Clockwise: swap start/end for range check
                        if a_end <= a_start { ca >= a_end && ca <= a_start }
                        else { ca >= a_end || ca <= a_start }
                    } else {
                        if a_end >= a_start { ca >= a_start && ca <= a_end }
                        else { ca >= a_start || ca <= a_end }
                    };
                    if in_arc {
                        let px = xc + cardinal_offsets[k].0;
                        let py = yc + cardinal_offsets[k].1;
                        if px < self.x_min { self.x_min = px; }
                        if px > self.x_max { self.x_max = px; }
                        if py < self.y_min { self.y_min = py; }
                        if py > self.y_max { self.y_max = py; }
                    }
                }
            }
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get the next vertex index (wrapping)
    #[inline]
    fn next_idx(&self, i: usize) -> usize {
        (i + 1) % self.vertices.len()
    }

    pub fn move_by(&mut self, dx: f64, dy: f64) {
        for v in &mut self.vertices {
            v.x += dx;
            v.y += dy;
            if v.seg.d != 0 {
                v.seg.xc += dx;
                v.seg.yc += dy;
            }
        }
        self.x_min += dx; self.x_max += dx;
        self.y_min += dy; self.y_max += dy;
    }

    pub fn scale(&mut self, sx: f64, sy: f64) {
        for v in &mut self.vertices {
            v.x *= sx;
            v.y *= sy;
            if v.seg.d != 0 {
                v.seg.xc *= sx;
                v.seg.yc *= sy;
            }
        }
        self.recalc_bounds();
    }

    pub fn rotate(&mut self, xr: f64, yr: f64, angle: f64) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        for v in &mut self.vertices {
            let x = v.x - xr;
            let y = v.y - yr;
            v.x = x * cos_a - y * sin_a + xr;
            v.y = x * sin_a + y * cos_a + yr;
            if v.seg.d != 0 {
                let cx = v.seg.xc - xr;
                let cy = v.seg.yc - yr;
                v.seg.xc = cx * cos_a - cy * sin_a + xr;
                v.seg.yc = cx * sin_a + cy * cos_a + yr;
            }
        }
        self.recalc_bounds();
    }


    /// Bounding rectangle as a Polygon
    pub fn brect(&self) -> Polygon {
        let mut p = Polygon::new();
        p.addv(self.x_min, self.y_min);
        p.addv(self.x_max, self.y_min);
        p.addv(self.x_max, self.y_max);
        p.addv(self.x_min, self.y_max);
        p
    }

    /// Legacy `is_inside`: the original PHP-port logic without the
    /// ray-degeneracy guard. Kept ONLY to A/B compare against the current
    /// implementation in regression-investigation tests.
    #[doc(hidden)]
    pub fn is_inside_legacy(&self, vx: f64, vy: f64) -> bool {
        let test_point = Vertex::new(vx, vy);
        let infinity = Vertex::new(-10_000_000.0, vy);
        let n = self.vertices.len();
        for i in 0..n {
            let j = self.next_idx(i);
            let q = &self.vertices[i];
            let r = &self.vertices[j];
            let on_edge = if q.seg.d == 0 {
                intersector::intersection(&test_point, &test_point, q, r)
            } else {
                intersector::line_arc_intersection(&test_point, &test_point, q, r, false)
            };
            if !on_edge.is_empty() {
                return true;
            }
        }
        self.winding_is_odd(&infinity, &test_point)
    }

    /// Check if a point is inside this polygon using winding number ray-casting.
    /// Faithfully ports the PHP isInside() logic including edge-on-point checks
    /// and special arc handling.
    ///
    /// Robustness: the classic horizontal ray degenerates when it passes
    /// within EPSILON of a vertex (e.g. a square rotated 135° puts two
    /// vertices at the test point's exact height) or nearly tangent to an
    /// arc — crossing counts then become unstable and the same geometric
    /// configuration can answer differently at different float offsets. We
    /// therefore pick a ray direction that stays clear of every vertex and
    /// arc tangency before counting; the first candidate is the original
    /// horizontal ray, so clean cases behave exactly as before.
    pub fn is_inside(&self, vx: f64, vy: f64) -> bool {
        let test_point = Vertex::new(vx, vy);
        let n = self.vertices.len();

        // 1) On the boundary counts as inside.
        for i in 0..n {
            let j = self.next_idx(i);
            let q = &self.vertices[i];
            let r = &self.vertices[j];
            let on_edge = if q.seg.d == 0 {
                intersector::intersection(&test_point, &test_point, q, r)
            } else {
                intersector::line_arc_intersection(&test_point, &test_point, q, r, false)
            };
            if !on_edge.is_empty() {
                return true;
            }
        }

        // 2) Winding count with a numerically safe ray. k = 0 reproduces
        // the original horizontal ray byte-for-byte (origin at world x =
        // -1e7) so clean cases behave EXACTLY as the legacy code — moving
        // the origin even by a translation shifts float epsilons in the
        // line-segment-parameter check and can flip a borderline arc
        // intersection between "in segment" and "out of segment".
        for k in 0..8u32 {
            let infinity = if k == 0 {
                Vertex::new(-10_000_000.0, vy)
            } else {
                let ang = k as f64 * 0.39996;
                Vertex::new(
                    vx - 10_000_000.0 * ang.cos(),
                    vy - 10_000_000.0 * ang.sin(),
                )
            };
            if !self.ray_is_degenerate(&infinity, &test_point) {
                return self.winding_is_odd(&infinity, &test_point);
            }
        }
        // Every probe grazed something (pathological): legacy behaviour.
        let infinity = Vertex::new(-10_000_000.0, vy);
        self.winding_is_odd(&infinity, &test_point)
    }

    /// A ray is degenerate when it passes within ~EPSILON of any vertex or
    /// almost tangent to any arc: intersection counts are then unreliable.
    fn ray_is_degenerate(&self, infinity: &Vertex, test_point: &Vertex) -> bool {
        let tol = 2.0 * intersector::EPSILON;
        for v in &self.vertices {
            if intersector::dist_point_to_line(v, infinity, test_point) <= tol {
                return true;
            }
            if v.seg.d != 0 {
                let centre = Vertex::new(v.seg.xc, v.seg.yc);
                let radius = intersector::dist(v.seg.xc, v.seg.yc, v.x, v.y);
                let d = intersector::dist_point_to_line(&centre, infinity, test_point);
                if (d - radius).abs() <= tol {
                    return true;
                }
            }
        }
        false
    }

    /// Original PHP winding logic, parameterized by the ray start.
    ///
    /// KNOWN LIMITATION (root cause traced 2026-06-23): on polygons with
    /// mixed line/arc edges (e.g. a Minkowski-inflated drop) this miscounts
    /// when the ray passes *near* (not exactly through) a vertex where a line
    /// edge meets an arc edge. Both edges then report a crossing close to the
    /// shared vertex, and the vertex-dedup heuristic below
    /// (`q_intercepts`/`r_intercepts`/`is_vertical_vertex`) fails to merge
    /// them because the vertex is > EPSILON off the ray — so the crossing is
    /// double-counted, flipping the inside/outside result. Repro:
    /// `tests/dilation.rs::dilation_is_inside_ray_casting_is_unreliable`
    /// (≈0.028% of a grid, along the horizontal lines that graze the inflated
    /// arcs). The robust fix is a **half-open crossing rule** (count an edge
    /// iff exactly one endpoint is strictly above the ray) applied uniformly
    /// to line and arc edges, so a shared vertex is owned by exactly one
    /// edge. That rework is fingerprint-critical (`completely_contains` feeds
    /// template generation through `is_inside`), so it must be validated
    /// against `fingerprint_regression` + `verify_88_ray_fix_templates`.
    /// Until then, dilation queries use `Polygon::within_dilation` (distance
    /// on the *original* polygon), which avoids this path entirely.
    pub(crate) fn winding_is_odd(&self, infinity: &Vertex, test_point: &Vertex) -> bool {
        let n = self.vertices.len();
        let mut winding_number = 0;
        for i in 0..n {
            let j = self.next_idx(i);
            let q = &self.vertices[i];
            let r = &self.vertices[j];

            let int = if q.seg.d == 0 {
                intersector::intersection(infinity, test_point, q, r)
            } else {
                intersector::line_arc_intersection(infinity, test_point, q, r, true)
            };

            if int.len() == 2 && q.seg.d != 0 {
                // Arc with 2 intersections: check if endpoints are on the ray
                let q_intercepts = !intersector::intersection(infinity, test_point, q, q).is_empty();
                let r_intercepts = !intersector::intersection(infinity, test_point, r, r).is_empty();
                if q_intercepts ^ r_intercepts {
                    winding_number += 1;
                }
            } else if int.len() == 1 {
                // Single intersection: check vertex cases
                let q_intercepts = !intersector::intersection(infinity, test_point, q, q).is_empty();
                let r_intercepts = !intersector::intersection(infinity, test_point, r, r).is_empty();
                if (!q_intercepts && !r_intercepts)
                    || (q_intercepts && self.is_vertical_vertex(i))
                {
                    winding_number += 1;
                }
            }
        }
        winding_number % 2 == 1
    }

    /// Determine the vertical direction of the edge starting at vertex index t.
    /// Returns 1 (going up), -1 (going down), or 0 (horizontal).
    /// For arcs, uses angle/cosine to determine the direction at the vertex.
    fn vertical_direction(&self, t: usize, point_for_arc: Option<usize>) -> i32 {
        let v = &self.vertices[t];
        let n = self.next_idx(t);
        let next = &self.vertices[n];
        if v.seg.d == 0 {
            // Line segment: compare Y values
            if v.y < next.y { 1 }
            else if v.y > next.y { -1 }
            else { 0 }
        } else {
            // Arc segment: use angle/cosine
            let a = if let Some(pi) = point_for_arc {
                intersector::angle(v.seg.xc, v.seg.yc, self.vertices[pi].x, self.vertices[pi].y)
            } else {
                intersector::angle(v.seg.xc, v.seg.yc, v.x, v.y)
            };
            let cos_a = a.cos();
            let td = v.seg.d as i32;
            if cos_a.abs() < 1e-10 {
                let sin_a = a.sin();
                let effective_cos = if let Some(pi) = point_for_arc {
                    if self.vertices[n].equals(&self.vertices[pi]) {
                        sin_a * td as f64
                    } else {
                        sin_a * td as f64
                    }
                } else {
                    -sin_a * td as f64
                };
                (if effective_cos > 0.0 { 1 } else { -1 }) * td
            } else {
                (if cos_a > 0.0 { 1 } else { -1 }) * td
            }
        }
    }

    /// Check if the edge at vertex t is a horizontal line
    fn is_horizontal_line(&self, t: usize) -> bool {
        let v = &self.vertices[t];
        v.seg.d == 0 && v.y == self.vertices[self.next_idx(t)].y
    }

    /// Check if vertex at index i is a "vertical vertex".
    /// A vertical vertex is one where the incoming and outgoing edges
    /// both go in the same vertical direction (both up or both down).
    /// Horizontal edges are skipped when looking backward.
    fn is_vertical_vertex(&self, i: usize) -> bool {
        let n = self.vertices.len();
        // Walk backward, skipping horizontal lines
        let mut prev = if i == 0 { n - 1 } else { i - 1 };
        let mut prev_post = i;
        while self.is_horizontal_line(prev) {
            prev_post = prev;
            prev = if prev == 0 { n - 1 } else { prev - 1 };
        }
        let prev_direction = self.vertical_direction(prev, Some(prev_post));
        let direction = self.vertical_direction(i, None);
        (prev_direction == 1 && direction == 1) || (prev_direction == -1 && direction == -1)
    }

    /// Check if this polygon completely contains another polygon
    pub fn completely_contains(&self, other: &Polygon) -> bool {
        // Fast bounding box rejection (bounds include arc extents)
        if other.x_min < self.x_min || other.x_max > self.x_max
            || other.y_min < self.y_min || other.y_max > self.y_max {
            return false;
        }
        // All vertices of other must be inside this
        for v in &other.vertices {
            if !self.is_inside(v.x, v.y) {
                return false; // Early return
            }
        }

        // No edge crossings (except at shared vertices)
        let n_self = self.vertices.len();
        let n_other = other.vertices.len();
        for i in 0..n_self {
            let si = self.next_idx(i);
            for j in 0..n_other {
                let oj = other.next_idx(j);
                let ints = self.edge_intersection(i, si, other, j, oj);
                if ints.len() == 1 {
                    let int = &ints[0];
                    let sv = &self.vertices[i];
                    let sn = &self.vertices[si];
                    let cv = &other.vertices[j];
                    let cn = &other.vertices[oj];
                    if sv.seg.d == 0 && cv.seg.d == 0
                        && !(int.equals(sv) || int.equals(sn) || int.equals(cv) || int.equals(cn))
                    {
                        return false;
                    }
                } else if ints.len() == 2 {
                    if self.vertices[i].seg.d != 0 || other.vertices[j].seg.d != 0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check that no edges of self cross edges of other (assumes vertices already checked).
    /// Returns true if no problematic crossings found.
    pub fn no_edge_crossing(&self, other: &Polygon) -> bool {
        let n_self = self.vertices.len();
        let n_other = other.vertices.len();
        for i in 0..n_self {
            let si = self.next_idx(i);
            for j in 0..n_other {
                let oj = other.next_idx(j);
                let ints = self.edge_intersection(i, si, other, j, oj);
                if ints.len() == 1 {
                    let int = &ints[0];
                    let sv = &self.vertices[i];
                    let sn = &self.vertices[si];
                    let cv = &other.vertices[j];
                    let cn = &other.vertices[oj];
                    if sv.seg.d == 0 && cv.seg.d == 0
                        && !(int.equals(sv) || int.equals(sn) || int.equals(cv) || int.equals(cn))
                    {
                        return false;
                    }
                } else if ints.len() == 2 {
                    if self.vertices[i].seg.d != 0 || other.vertices[j].seg.d != 0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if this polygon intersects another polygon
    pub fn is_poly_intersect(&self, other: &Polygon) -> bool {
        // Fast bounding box rejection
        if self.x_max < other.x_min || self.x_min > other.x_max
            || self.y_max < other.y_min || self.y_min > other.y_max {
            return false;
        }

        let n_self = self.vertices.len();
        let n_other = other.vertices.len();
        for i in 0..n_self {
            let si = self.next_idx(i);
            for j in 0..n_other {
                let oj = other.next_idx(j);
                if !self.edge_intersection(i, si, other, j, oj).is_empty() {
                    return true; // Early return
                }
            }
        }
        false
    }

    /// Compute intersection between an edge of self and an edge of other
    fn edge_intersection(&self, i: usize, ni: usize, other: &Polygon, j: usize, nj: usize) -> Vec<Vertex> {
        let p1 = &self.vertices[i];
        let p2 = &self.vertices[ni];
        let q1 = &other.vertices[j];
        let q2 = &other.vertices[nj];

        let pt = p1.seg.d;
        let qt = q1.seg.d;

        if pt == 0 && qt == 0 {
            // Line/Line
            intersector::intersection(p1, p2, q1, q2)
        } else if pt == 0 && qt != 0 {
            // Line/Arc
            intersector::line_arc_intersection(p1, p2, q1, q2, false)
        } else if pt != 0 && qt == 0 {
            // Arc/Line
            intersector::line_arc_intersection(q1, q2, p1, p2, false)
        } else {
            // Arc/Arc - simplified: treat as line/line approximation for now
            // Full arc/arc intersection from the PHP code is complex
            intersector::intersection(p1, p2, q1, q2)
        }
    }
}

// === Factory methods ===

pub fn create_drop(width: f64, height: f64) -> Polygon {
    let mut p = Polygon::new();
    p.addv_arc(-width, height, 0.0, height, -1);
    p.addv(width, height);
    p.addv(0.0, 0.0);
    p.finalize_bounds();
    p
}

pub fn create_circle(radius: f64) -> Polygon {
    let mut p = Polygon::new();
    p.addv_arc(0.0, -radius, 0.0, 0.0, -1);
    p.addv_arc(0.0, radius, 0.0, 0.0, -1);
    p.finalize_bounds();
    p
}

pub fn create_box(side: f64) -> Polygon {
    let half = side / 2.0;
    create_square(-half, -half, half, half)
}

pub fn create_square(sx: f64, sy: f64, ex: f64, ey: f64) -> Polygon {
    let mut p = Polygon::new();
    p.addv(sx, sy);
    p.addv(ex, sy);
    p.addv(ex, ey);
    p.addv(sx, ey);
    p
}

/// Minkowski dilation of a **convex** polygon by radius `r`: the returned
/// polygon contains exactly the points within distance `r` of the original
/// (boundary included). Line edges shift outward by `r`, existing arcs grow
/// from radius `R` to `R + r` (same centre), and each corner gains a joining
/// arc of radius `r` centred on the original vertex.
///
/// This is the "index dilation" device for items with circular extent: a
/// figure F hits an item of body radius `r` iff the item's *centre* lies
/// inside `inflated_convex(F, r)` — so the spatial index keeps storing
/// points, and only the template bank gains an inflated set per radius.
///
/// Concave polygons are not supported (offsets would self-intersect).
pub fn inflated_convex(poly: &Polygon, r: f64) -> Polygon {
    assert!(r > 0.0, "inflation radius must be positive");
    let n = poly.vertices.len();
    assert!(n >= 2, "need at least two vertices");

    // Orientation from the signed area of the vertex loop (arc bulges do
    // not change the winding of a convex figure).
    let mut signed2 = 0.0;
    for i in 0..n {
        let a = &poly.vertices[i];
        let b = &poly.vertices[(i + 1) % n];
        signed2 += a.x * b.y - b.x * a.y;
    }
    let ccw = signed2 > 0.0;
    // Corner arcs turn the same way the polygon winds.
    let corner_d: i8 = if ccw { 1 } else { -1 };

    // Offset endpoints of every edge.
    struct OffsetEdge {
        start: (f64, f64),
        end: (f64, f64),
        // None = line; Some((xc, yc, d)) = arc with grown radius.
        arc: Option<(f64, f64, i8)>,
    }
    let mut edges: Vec<OffsetEdge> = Vec::with_capacity(n);
    for i in 0..n {
        let v = &poly.vertices[i];
        let w = &poly.vertices[(i + 1) % n];
        if v.seg.d == 0 {
            let dx = w.x - v.x;
            let dy = w.y - v.y;
            let len = (dx * dx + dy * dy).sqrt();
            assert!(len > 0.0, "degenerate edge");
            // Outward normal: right of travel for CW, left for CCW.
            let (nx, ny) = if ccw {
                (dy / len, -dx / len)
            } else {
                (-dy / len, dx / len)
            };
            edges.push(OffsetEdge {
                start: (v.x + nx * r, v.y + ny * r),
                end: (w.x + nx * r, w.y + ny * r),
                arc: None,
            });
        } else {
            // Convex arc: centre on the interior side; growing the radius
            // offsets it outward. Endpoints move radially away from centre.
            let (xc, yc) = (v.seg.xc, v.seg.yc);
            let radius = intersector::dist(xc, yc, v.x, v.y);
            let k = (radius + r) / radius;
            edges.push(OffsetEdge {
                start: (xc + (v.x - xc) * k, yc + (v.y - yc) * k),
                end: (xc + (w.x - xc) * k, yc + (w.y - yc) * k),
                arc: Some((xc, yc, v.seg.d)),
            });
        }
    }

    // Stitch: each offset edge, then a corner arc (centred on the original
    // shared vertex) bridging to the next offset edge — skipped when the
    // endpoints already coincide (tangent-continuous joins, e.g. circles).
    let mut out = Polygon::new();
    for i in 0..n {
        let e = &edges[i];
        match e.arc {
            None => out.addv(e.start.0, e.start.1),
            Some((xc, yc, d)) => out.addv_arc(e.start.0, e.start.1, xc, yc, d),
        }
        let next_start = edges[(i + 1) % n].start;
        let gap = ((e.end.0 - next_start.0).powi(2) + (e.end.1 - next_start.1).powi(2)).sqrt();
        if gap > intersector::EPSILON {
            let corner = &poly.vertices[(i + 1) % n];
            out.addv_arc(e.end.0, e.end.1, corner.x, corner.y, corner_d);
        }
    }
    out.finalize_bounds();
    out
}

impl Polygon {
    /// Shortest distance from `(px, py)` to this polygon's boundary
    /// (line edges and arcs). Unsigned — interior and exterior points at
    /// the same offset return the same value.
    ///
    /// This is the robust primitive behind the **dilation narrowphase**:
    /// a point is within the Minkowski dilation of this polygon by `r` iff
    /// `self.is_inside(p) || self.dist_to_boundary(p) <= r`. Using the
    /// *original* polygon's `is_inside` (few segments, reliable) plus a
    /// distance check sidesteps the ray-casting fragility that
    /// `is_inside` exhibits on many-arc *inflated* polygons — see
    /// `inflated_convex` and `tests/dilation.rs`.
    pub fn dist_to_boundary(&self, px: f64, py: f64) -> f64 {
        let n = self.vertices.len();
        let mut best = f64::MAX;
        for i in 0..n {
            let v = &self.vertices[i];
            let w = &self.vertices[(i + 1) % n];
            let d = if v.seg.d == 0 {
                dist_point_segment(px, py, v.x, v.y, w.x, w.y)
            } else {
                let (xc, yc) = (v.seg.xc, v.seg.yc);
                let radius = intersector::dist(xc, yc, v.x, v.y);
                let pa = intersector::angle(xc, yc, px, py);
                let mut a1 = intersector::angle(xc, yc, v.x, v.y);
                let mut a2 = intersector::angle(xc, yc, w.x, w.y);
                if v.seg.d == -1 {
                    std::mem::swap(&mut a1, &mut a2);
                }
                let in_span = if a2 >= a1 { pa >= a1 && pa <= a2 } else { pa >= a1 || pa <= a2 };
                if in_span {
                    (intersector::dist(xc, yc, px, py) - radius).abs()
                } else {
                    intersector::dist(px, py, v.x, v.y).min(intersector::dist(px, py, w.x, w.y))
                }
            };
            best = best.min(d);
        }
        best
    }

    /// Robust dilation membership: is `(px, py)` within the Minkowski
    /// dilation of this polygon by radius `r`? Equivalent to
    /// `is_inside(p) || dist_to_boundary(p) <= r`, but computed without
    /// ever building or ray-casting the inflated polygon — the production
    /// **narrowphase** for an agent of body radius `r` (used on `Maybe`
    /// raster pixels; `In`/`Out` pixels are resolved by the precomputed
    /// inflated raster lookup and never reach here).
    pub fn within_dilation(&self, r: f64, px: f64, py: f64) -> bool {
        self.is_inside(px, py) || self.dist_to_boundary(px, py) <= r
    }
}

fn dist_point_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return intersector::dist(px, py, ax, ay);
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
    intersector::dist(px, py, ax + t * dx, ay + t * dy)
}

pub fn scaled_copy(poly: &Polygon, sx: f64, sy: f64) -> Polygon {
    let mut p = poly.clone();
    p.scale(sx, sy);
    p
}

pub fn rotated_copy(poly: &Polygon, angle: f64) -> Polygon {
    let mut p = poly.clone();
    p.rotate(0.0, 0.0, angle);
    p
}
