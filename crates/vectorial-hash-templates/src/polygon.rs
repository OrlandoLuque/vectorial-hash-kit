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

    /// Check if a point is inside this polygon using winding number ray-casting.
    /// Faithfully ports the PHP isInside() logic including edge-on-point checks
    /// and special arc handling.
    pub fn is_inside(&self, vx: f64, vy: f64) -> bool {
        let test_point = Vertex::new(vx, vy);
        let infinity = Vertex::new(-10_000_000.0, vy);
        let n = self.vertices.len();
        let mut winding_number = 0;

        for i in 0..n {
            let j = self.next_idx(i);
            let q = &self.vertices[i];
            let r = &self.vertices[j];

            // If the point lies exactly on an edge, it's inside
            let on_edge = if q.seg.d == 0 {
                intersector::intersection(&test_point, &test_point, q, r)
            } else {
                intersector::line_arc_intersection(&test_point, &test_point, q, r, false)
            };
            if !on_edge.is_empty() {
                return true;
            }

            // Cast ray from infinity to test_point
            let int = if q.seg.d == 0 {
                intersector::intersection(&infinity, &test_point, q, r)
            } else {
                intersector::line_arc_intersection(&infinity, &test_point, q, r, true)
            };

            if int.len() == 2 && q.seg.d != 0 {
                // Arc with 2 intersections: check if endpoints are on the ray
                let q_intercepts = !intersector::intersection(&infinity, &test_point, q, q).is_empty();
                let r_intercepts = !intersector::intersection(&infinity, &test_point, r, r).is_empty();
                if q_intercepts ^ r_intercepts {
                    winding_number += 1;
                }
            } else if int.len() == 1 {
                // Single intersection: check vertex cases
                let q_intercepts = !intersector::intersection(&infinity, &test_point, q, q).is_empty();
                let r_intercepts = !intersector::intersection(&infinity, &test_point, r, r).is_empty();
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
