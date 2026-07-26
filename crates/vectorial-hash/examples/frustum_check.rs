//! Is `Tree3::cull(&Polyhedron3)` exact? Brute-force cross-check on frustums built the
//! way a view cone is: a small near quad, a large far quad, `from_corners`.
use vectorial_hash::{Aabb, Point3, Polyhedron3, Positioned3, Shape3, Tree3};

#[derive(Clone, Copy)]
struct P { id: u32, p: Point3 }
impl Positioned3 for P { fn position(&self) -> Point3 { self.p } }

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f64) / (1u64 << 24) as f64 }
    fn r(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.f() }
}

fn cone(eye: [f64; 3], face: f64, near: f64, far: f64, fov: f64) -> Polyhedron3 {
    let (s, c) = (face.sin(), face.cos());
    let (fx, fz) = (c, s);
    let (rx, rz) = (-s, c);
    let quad = |dist: f64, half: f64, vh: f64| {
        let ctr = [eye[0] + fx * dist, eye[1], eye[2] + fz * dist];
        [
            Point3::new(ctr[0] - rx * half, ctr[1] - vh, ctr[2] - rz * half),
            Point3::new(ctr[0] + rx * half, ctr[1] - vh, ctr[2] + rz * half),
            Point3::new(ctr[0] + rx * half, ctr[1] + vh, ctr[2] + rz * half),
            Point3::new(ctr[0] - rx * half, ctr[1] + vh, ctr[2] - rz * half),
        ]
    };
    let n = quad(near, near * (fov * 0.5).tan() + 3.0, 7.0);
    let f = quad(far, far * (fov * 0.5).tan(), 34.0);
    Polyhedron3::from_corners([n[0], n[1], n[2], n[3], f[0], f[1], f[2], f[3]])
}

fn main() {
    let mut r = Rng(0x9E37_79B9);
    let world = Aabb::new(-10.0, -10.0, -10.0, 920.0, 220.0, 920.0);
    let pts: Vec<P> = (0..4000).map(|i| P { id: i, p: Point3::new(r.r(20.0, 880.0), r.r(0.0, 40.0), r.r(20.0, 880.0)) }).collect();
    let tree = Tree3::bulk_load(world, 8, pts.clone());

    let (mut bad_cases, mut miss, mut extra, mut checked) = (0u32, 0u32, 0u32, 0u32);
    let mut worst: Option<(u32, f64, [f64; 3])> = None;
    for _ in 0..400 {
        let eye = [r.r(60.0, 840.0), 9.0, r.r(60.0, 840.0)];
        let k = cone(eye, r.r(-3.2, 3.2), 6.0, 300.0, 1.15);
        let got: std::collections::HashSet<u32> = tree.cull(&k).iter().map(|p| p.id).collect();
        let want: std::collections::HashSet<u32> = pts.iter().filter(|p| k.contains_point(p.p)).map(|p| p.id).collect();
        checked += 1;
        if got != want {
            bad_cases += 1;
            miss += want.difference(&got).count() as u32;
            extra += got.difference(&want).count() as u32;
            if worst.is_none() {
                if let Some(&id) = want.difference(&got).next() {
                    let p = pts[id as usize].p;
                    // how deep inside is the missed point, in geometric units?
                    let d = k.planes.iter().map(|&(nx, ny, nz, dd)| { let m = (nx * nx + ny * ny + nz * nz).sqrt(); (dd - (nx * p.x + ny * p.y + nz * p.z)) / m }).fold(f64::INFINITY, f64::min);
                    worst = Some((id, d, [p.x, p.y, p.z]));
                    let b = k.bounding_box();
                    println!("  missed id {} at ({:.1}, {:.1}, {:.1}) — {:.3} units inside the nearest face", id, p.x, p.y, p.z, d);
                    println!("  cone bbox: x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}", b.x, b.x_max(), b.y, b.y_max(), b.z, b.z_max());
                    println!("  world box: x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}", world.x, world.x_max(), world.y, world.y_max(), world.z, world.z_max());
                }
            }
        }
    }
    println!("frustum_check: {} cones, {} disagreed — {} points missed by cull, {} extra", checked, bad_cases, miss, extra);
    if let Some((id, d, p)) = worst { println!("first miss: id {} at {:?}, {:.3} units inside", id, p, d); }
}
