use crate::vertex::Vertex;

pub const EPSILON: f64 = 0.00001;

/// Whether to use epsilon-based comparisons (true) or exact (false)
static mut EPSILON_MODE: bool = false;

pub fn set_epsilon_mode(mode: bool) {
    unsafe { EPSILON_MODE = mode; }
}

pub fn is_epsilon_mode() -> bool {
    unsafe { EPSILON_MODE }
}

#[inline]
pub fn is_zero(v: f64) -> bool {
    if is_epsilon_mode() { v.abs() < EPSILON } else { v == 0.0 }
}

#[inline]
pub fn is_equal(a: f64, b: f64) -> bool {
    if is_epsilon_mode() { (a - b).abs() < EPSILON } else { a == b }
}

pub fn dist(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
}

pub fn angle(xc: f64, yc: f64, x1: f64, y1: f64) -> f64 {
    let d = dist(xc, yc, x1, y1);
    if !is_zero(d) {
        if ((y1 - yc) / d).asin() >= 0.0 {
            ((x1 - xc) / d).acos()
        } else {
            2.0 * std::f64::consts::PI - ((x1 - xc) / d).acos()
        }
    } else {
        0.0
    }
}

/// Line-line segment intersection
pub fn intersection(a1: &Vertex, a2: &Vertex, b1: &Vertex, b2: &Vertex) -> Vec<Vertex> {
    // Both points
    if a1.equals(a2) && b1.equals(b2) {
        return if a1.equals(b1) { vec![a1.clone()] } else { vec![] };
    }
    // b is a point
    if b1.equals(b2) {
        return if point_on_line(b1, a1, a2) { vec![b1.clone()] } else { vec![] };
    }
    // a is a point
    if a1.equals(a2) {
        return if point_on_line(a1, b1, b2) { vec![a1.clone()] } else { vec![] };
    }

    let ua_t = (b2.x - b1.x) * (a1.y - b1.y) - (b2.y - b1.y) * (a1.x - b1.x);
    let ub_t = (a2.x - a1.x) * (a1.y - b1.y) - (a2.y - a1.y) * (a1.x - b1.x);
    let u_b = (b2.y - b1.y) * (a2.x - a1.x) - (b2.x - b1.x) * (a2.y - a1.y);

    if !(-EPSILON < u_b && u_b < EPSILON) {
        let ua = ua_t / u_b;
        let ub = ub_t / u_b;
        if (0.0..=1.0).contains(&ua) && (0.0..=1.0).contains(&ub) {
            vec![Vertex::new(
                a1.x + ua * (a2.x - a1.x),
                a1.y + ua * (a2.y - a1.y),
            )]
        } else {
            vec![]
        }
    } else {
        // Parallel or coincident
        if (-EPSILON < ua_t && ua_t < EPSILON) || (-EPSILON < ub_t && ub_t < EPSILON) {
            if a1.equals(a2) {
                one_d_intersection(b1, b2, a1, a2)
            } else {
                one_d_intersection(a1, a2, b1, b2)
            }
        } else {
            vec![]
        }
    }
}

fn overlap_intervals(ub1: f64, ub2: f64) -> Vec<f64> {
    let l = ub1.min(ub2);
    let r = ub1.max(ub2);
    let a = 0.0_f64.max(l);
    let b = 1.0_f64.min(r);
    if a > b {
        vec![]
    } else if a == b {
        vec![a]
    } else {
        vec![a, b]
    }
}

fn one_d_intersection(a1: &Vertex, a2: &Vertex, b1: &Vertex, b2: &Vertex) -> Vec<Vertex> {
    let denomx = a2.x - a1.x;
    let denomy = a2.y - a1.y;
    let (ub1, ub2) = if denomx.abs() > denomy.abs() {
        ((b1.x - a1.x) / denomx, (b2.x - a1.x) / denomx)
    } else {
        ((b1.y - a1.y) / denomy, (b2.y - a1.y) / denomy)
    };

    overlap_intervals(ub1, ub2)
        .iter()
        .map(|f| Vertex::new(a2.x * f + a1.x * (1.0 - f), a2.y * f + a1.y * (1.0 - f)))
        .collect()
}

fn point_on_line(p: &Vertex, a1: &Vertex, a2: &Vertex) -> bool {
    let d = dist_from_seg(p, a1, a2);
    d < EPSILON
        && (p.x >= a1.x.min(a2.x) && p.x <= a1.x.max(a2.x))
        && (p.y >= a1.y.min(a2.y) && p.y <= a1.y.max(a2.y))
}

fn dist_from_seg(p: &Vertex, q0: &Vertex, q1: &Vertex) -> f64 {
    let dx21 = q1.x - q0.x;
    let dy21 = q1.y - q0.y;
    let dx10 = q0.x - p.x;
    let dy10 = q0.y - p.y;
    let seg_length = (dx21 * dx21 + dy21 * dy21).sqrt();
    if seg_length < EPSILON {
        return f64::MAX;
    }
    (dx21 * dy10 - dx10 * dy21).abs() / seg_length
}

/// Line-circle intersection
pub fn line_circle_intersection(l1: &Vertex, l2: &Vertex, c: &Vertex, radius: f64) -> Vec<Vertex> {
    let dx = l2.x - l1.x;
    let dy = l2.y - l1.y;
    let a = dx * dx + dy * dy;
    let b = 2.0 * (dx * (l1.x - c.x) + dy * (l1.y - c.y));
    let cc = (l1.x - c.x).powi(2) + (l1.y - c.y).powi(2) - radius * radius;
    let det = b * b - 4.0 * a * cc;

    if a <= 0.0000001 || det < 0.0 {
        vec![]
    } else if is_zero(det) {
        let t = -b / (2.0 * a);
        vec![Vertex::new(l1.x + t * dx, l1.y + t * dy)]
    } else {
        let t1 = (-b + det.sqrt()) / (2.0 * a);
        let t2 = (-b - det.sqrt()) / (2.0 * a);
        vec![
            Vertex::new(l1.x + t1 * dx, l1.y + t1 * dy),
            Vertex::new(l1.x + t2 * dx, l1.y + t2 * dy),
        ]
    }
}

/// Line-arc intersection (uses cached radius and angles from Segment)
pub fn line_arc_intersection(
    l1: &Vertex, l2: &Vertex,
    a1: &Vertex, a2: &Vertex,
    ignore_touch: bool,
) -> Vec<Vertex> {
    let xc = a1.seg.xc;
    let yc = a1.seg.yc;
    let radius = dist(xc, yc, a1.x, a1.y);
    let circle_ints = line_circle_intersection(l1, l2, &Vertex::new(xc, yc), radius);

    let touch = circle_ints.len() == 1
        && !a1.roughly_equals(&circle_ints[0])
        && !a2.roughly_equals(&circle_ints[0]);
    if touch && ignore_touch {
        return vec![];
    }

    let mut arc_angle1 = angle(xc, yc, a1.x, a1.y);
    let mut arc_angle2 = angle(xc, yc, a2.x, a2.y);
    if a1.seg.d == -1 {
        std::mem::swap(&mut arc_angle1, &mut arc_angle2);
    }

    let mut result = Vec::new();
    for mut int in circle_ints {
        if int.roughly_equals(a1) {
            int = a1.clone();
        }
        if int.roughly_equals(a2) {
            int = a2.clone();
        }
        if int.is_inside(l1, l2) {
            // Only this angle() call can't be cached (intersection point varies)
            let int_angle = angle(xc, yc, int.x, int.y);
            let in_arc = if arc_angle2 >= arc_angle1 {
                int_angle >= arc_angle1 && int_angle <= arc_angle2
            } else {
                int_angle <= arc_angle2 || int_angle >= arc_angle1
            };
            if in_arc {
                result.push(int);
            }
        }
    }
    result
}
