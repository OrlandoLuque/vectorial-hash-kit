//! fluid_wgpu — an **interactive 2D SPH fluid** you can stir with the mouse (or a
//! finger on a touch screen), whose neighbour search runs through the kit's 2D
//! structures — live-switchable so you can watch the index cost move.
//!
//! Smoothed-particle hydrodynamics is *the* archetypal spatial-index workload: every
//! particle needs the ones within the kernel radius `H`, **every step**, and the fluid
//! is by nature strongly clustered (that's what a fluid IS). So the demo is also an
//! honest head-to-head: `M` cycles the index between
//!   · **MortonGrid** — flat Z-order grid, rebuilt each step,
//!   · **Tree + ItemRef** — the binary tree KEPT and relocated in place (O(1)/particle),
//!   · **LinearQuadTree** — the adaptive pointer-free quadtree, rebuilt each step,
//! and the HUD bars break the frame into *maintain* (build/relocate) vs *query*
//! (the neighbour culls) vs *physics*, so the trade-off is visible rather than asserted.
//!
//! Physics: **Position Based Fluids** (Macklin & Müller 2013) — density constraints
//! solved iteratively, which stays stable at dt = 1/60 where the classic Müller-2003
//! equation-of-state formulation needs a sub-millisecond step.
//!
//! Controls: **hold left mouse / drag a finger** to stir · `M` index · `[` `]` particles
//! · `P` pause · `R` reset · `G` gravity flip.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --bin fluid_wgpu --release
//! ```
#![cfg(any(not(target_arch = "wasm32"), feature = "web-wgpu"))]
#![cfg_attr(target_arch = "wasm32", no_main)]

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use winit::{event::*, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::WindowBuilder};
use vectorial_hash::morton3::Crossed;
use vectorial_hash::{AdaptiveIndex2, Backend, Slot};
use vectorial_hash::tree::Crossing2;
use vectorial_hash::{Circle, ItemRef, LinearQuadTree, MortonGrid, Point, Positioned, Rect, Tree};

// ---- world -----------------------------------------------------------------
const WW: f32 = 1200.0; // tank width  (world units)
const WH: f32 = 780.0;  // tank height

// ---- PBF (Macklin & Muller 2013, "Position Based Fluids") -------------------
// Why PBF and not the classic Muller-2003 EOS formulation: the stiff equation of
// state needs a sub-millisecond time step to stay stable, and its constants must be
// mutually consistent (mass, rest density, kernel radius) or the density estimate
// silently collapses. PBF is unconditionally stable at dt = 1/60 with a handful of
// constraint iterations, which is what an interactive demo actually needs.
const H: f32 = 22.0;                 // kernel radius - also the neighbour query radius
const HSQ: f32 = H * H;
const SPACING: f32 = H * 0.45;       // initial lattice spacing
const DT: f32 = 1.0 / 60.0;
const GRAV: f32 = -900.0;
const ITERS: usize = 3;              // constraint solver iterations per step
// Relaxation epsilon. The paper's 600 is in ITS units (rho0 = 1000, h = 0.1); here the
// kernel scale makes sum|grad C|^2 ~ 1e-2, so a literal 300 swamps the denominator and
// lambda collapses to zero - the solver silently stops solving and the fluid compresses
// to 13x rest (caught by the smoke test's density ratio). It only exists to avoid a
// divide-by-zero, so keep it negligible against the real term.
const RELAX: f32 = 1e-9;
const SCORR_K: f32 = 0.0001;         // artificial pressure: stops particle clumping
const SCORR_N: i32 = 4;
const SCORR_DQ: f32 = 0.2 * H;
// Position-based solvers inject energy through v = (q - p)/dt: a big constraint
// correction becomes a big velocity. Two standard brakes keep that in check — an
// over-relaxation factor (< 1) and a hard cap on how far one iteration may move a
// particle. Without them the density held at 1.0x for ~20 frames while vmax quietly
// ran to 3x free-fall, and the fluid then tore itself apart (smoke test, frame sweep).
const SOR: f32 = 0.30;               // correction gain per iteration
const MAX_CORR: f32 = 0.30 * SPACING; // per-iteration position clamp
const XSPH: f32 = 0.02;              // velocity smoothing (a little cohesion)
const VMAX: f32 = 1600.0;            // clamp: a stirred blob can not go ballistic
const EPS: f32 = H * 0.5;            // wall stand-off
const PI: f32 = std::f32::consts::PI;
// 2D kernels: poly6 for density, spiky gradient for the constraint gradient.
const POLY6: f32 = 4.0 / (PI * H * H * H * H * H * H * H * H);
const SPIKY_G: f32 = -30.0 / (PI * H * H * H * H * H);

#[inline] fn w_poly6(r2: f32) -> f32 { if r2 >= HSQ { 0.0 } else { let d = HSQ - r2; POLY6 * d * d * d } }
/// Magnitude of the spiky gradient along r-hat; the caller multiplies by the unit vector.
#[inline] fn w_spiky_grad(r: f32) -> f32 { if r >= H || r <= 1e-6 { 0.0 } else { SPIKY_G * (H - r) * (H - r) } }

const MAX_N: usize = 12_000;
const STIR_R: f32 = 90.0;            // radius the cursor pushes within

// ---- gpu types --------------------------------------------------------------
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Cam { vp: [[f32; 4]; 4] }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Inst { x: f32, y: f32, r: f32, heat: f32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UiVertex { pos: [f32; 2], color: [f32; 4] }

fn push_quad(v: &mut Vec<UiVertex>, px: f32, py: f32, w: f32, h: f32, color: [f32; 4], sw: f32, sh: f32) {
    let x0 = px / sw * 2.0 - 1.0; let x1 = (px + w) / sw * 2.0 - 1.0;
    let y0 = 1.0 - py / sh * 2.0; let y1 = 1.0 - (py + h) / sh * 2.0;
    let c = color;
    for p in [[x0, y0], [x1, y0], [x0, y1], [x0, y1], [x1, y0], [x1, y1]] { v.push(UiVertex { pos: p, color: c }); }
}

/// 3×5 bitmap font — enough for the HUD's letters and digits.
fn glyph(c: char) -> [&'static str; 5] {
    match c {
        '0' => ["111", "101", "101", "101", "111"], '1' => ["010", "110", "010", "010", "111"],
        '2' => ["111", "001", "111", "100", "111"], '3' => ["111", "001", "111", "001", "111"],
        '4' => ["101", "101", "111", "001", "001"], '5' => ["111", "100", "111", "001", "111"],
        '6' => ["111", "100", "111", "101", "111"], '7' => ["111", "001", "010", "010", "010"],
        '8' => ["111", "101", "111", "101", "111"], '9' => ["111", "101", "111", "001", "111"],
        'A' => ["111", "101", "111", "101", "101"], 'B' => ["110", "101", "110", "101", "110"],
        'C' => ["111", "100", "100", "100", "111"], 'D' => ["110", "101", "101", "101", "110"],
        'E' => ["111", "100", "110", "100", "111"], 'F' => ["111", "100", "110", "100", "100"],
        'G' => ["111", "100", "101", "101", "111"], 'H' => ["101", "101", "111", "101", "101"],
        'I' => ["111", "010", "010", "010", "111"], 'K' => ["101", "101", "110", "101", "101"],
        'L' => ["100", "100", "100", "100", "111"], 'M' => ["101", "111", "111", "101", "101"],
        'N' => ["101", "111", "111", "111", "101"], 'O' => ["111", "101", "101", "101", "111"],
        'P' => ["111", "101", "111", "100", "100"], 'Q' => ["111", "101", "101", "111", "011"],
        'R' => ["111", "101", "111", "110", "101"], 'S' => ["111", "100", "111", "001", "111"],
        'T' => ["111", "010", "010", "010", "010"], 'U' => ["101", "101", "101", "101", "111"],
        'V' => ["101", "101", "101", "101", "010"], 'W' => ["101", "101", "111", "111", "101"],
        'X' => ["101", "101", "010", "101", "101"], 'Y' => ["101", "101", "010", "010", "010"],
        'Z' => ["111", "001", "010", "100", "111"], '.' => ["000", "000", "000", "000", "010"],
        '/' => ["001", "001", "010", "100", "100"], '-' => ["000", "000", "111", "000", "000"],
        '%' => ["101", "001", "010", "100", "101"], ':' => ["000", "010", "000", "010", "000"],
        _ => ["000", "000", "000", "000", "000"],
    }
}
fn push_text(v: &mut Vec<UiVertex>, x: f32, y: f32, px: f32, color: [f32; 4], text: &str, sw: f32, sh: f32) {
    let mut cx = x;
    for c in text.chars() {
        for (row, bits) in glyph(c.to_ascii_uppercase()).iter().enumerate() {
            for (col, ch) in bits.char_indices() {
                if ch == '1' { push_quad(v, cx + col as f32 * px, y + row as f32 * px, px, px, color, sw, sh); }
            }
        }
        cx += 4.0 * px;
    }
}

// ---- the indexed particle ---------------------------------------------------
#[derive(Clone, Copy)]
struct FP { id: u32, p: Point }
impl Positioned for FP { fn position(&self) -> Point { self.p } }

#[derive(Clone, Copy, PartialEq, Eq)]
enum Idx { Morton, MortonKeep, TreeKeep, Linear, Adaptive }
impl Idx {
    fn next(self) -> Self { match self { Idx::Morton => Idx::MortonKeep, Idx::MortonKeep => Idx::TreeKeep, Idx::TreeKeep => Idx::Linear, Idx::Linear => Idx::Adaptive, Idx::Adaptive => Idx::Morton } }
    fn label(self) -> &'static str {
        match self { Idx::Morton => "MORTONGRID REBUILD", Idx::MortonKeep => "MORTONGRID KEEP", Idx::TreeKeep => "TREE KEEP-INDEX", Idx::Linear => "LINEARQUADTREE REBUILD", Idx::Adaptive => "ADAPTIVEINDEX2 (picks its own)" }
    }
}

/// The live index. Morton and the linear quadtree have no in-place handle, so their
/// "maintain" is a full rebuild; the binary tree relocates through its `ItemRef`.
enum Index {
    Morton(MortonGrid<FP>),
    /// The same grid, **maintained in place** with `MortonGrid::update`.
    ///
    /// It exists because without it this demo's comparison was rigged. The adaptive index
    /// picks the grid and KEEPS it, and it was being timed against a grid that REBUILDS — so
    /// the gap it showed was the keep path, not the policy. If a comparison makes the clever
    /// thing look good by denying the simple thing an optimisation the simple thing could
    /// have, the comparison is measuring the denial.
    MortonKeep { g: MortonGrid<FP>, last: Vec<Point> },
    Keep { tree: Tree<FP>, refs: Vec<ItemRef> },
    Linear(LinearQuadTree<FP>),
    /// The index that chooses for itself. It is here to be MEASURED against the three fixed
    /// choices on a workload none of them was tuned for — SPH asks one neighbour query per
    /// particle per frame, which is five times `rebuild_query_ratio`, so the policy has a real
    /// decision rather than a foregone one.
    Adaptive { ix: AdaptiveIndex2<FP>, slots: Vec<Slot> },
}

impl Index {
    fn build(kind: Idx, px: &[f32], py: &[f32]) -> Index {
        let rect = Rect::new(0.0, 0.0, WW as f64, WH as f64);
        let items = || (0..px.len()).map(|i| FP { id: i as u32, p: Point::new(px[i] as f64, py[i] as f64) });
        match kind {
            // one grid cell ≈ the kernel radius: the classic SPH bucket size, and MEASURED
            // better than the query diameter here (331-336 fps against 291-293) — the obvious
            // port of the adaptive index'''s apparent choice made things worse, so whatever it
            // is doing better is not simply a larger cell.
            Idx::Morton => { let lv = MortonGrid::<FP>::levels_for_cell_size(rect, H as f64); let mut g = MortonGrid::new(rect, lv); for it in items() { g.insert(it); } Index::Morton(g) }
            Idx::MortonKeep => {
                let lv = MortonGrid::<FP>::levels_for_cell_size(rect, H as f64);
                let mut g = MortonGrid::new(rect, lv);
                let mut last = Vec::with_capacity(px.len());
                for it in items() { last.push(it.p); g.insert(it); }
                Index::MortonKeep { g, last }
            }
            Idx::Linear => Index::Linear(LinearQuadTree::from_items(rect, 12, 16, items().collect())),
            Idx::TreeKeep => {
                let mut tree = Tree::<FP>::new(rect, 12);
                let refs = items().map(|it| tree.insert_ref(it).expect("in world")).collect();
                Index::Keep { tree, refs }
            }
            Idx::Adaptive => {
                let mut ix = AdaptiveIndex2::new(rect, 12);
                let slots = items().map(|it| ix.insert(it)).collect();
                Index::Adaptive { ix, slots }
            }
        }
    }
    /// Bring the index up to date with the new positions — a rebuild, or the O(1)
    /// per-particle relocation for the kept tree. This is the "maintain" bar.
    fn maintain(&mut self, kind: Idx, px: &[f32], py: &[f32]) -> (u64, u64) {
        match self {
            Index::Keep { tree, refs } => {
                // `_tracked` so the demo can report the RELOCATION RATE: how often a move
                // actually leaves its leaf. This is the workload that contradicts "keep the
                // index", so the number the advisor would have judged it by is worth having.
                let mut moved = 0u64;
                for i in 0..px.len() {
                    let np = Point::new(px[i] as f64, py[i] as f64);
                    if let Crossing2::Moved { .. } = tree.update_ref_tracked(refs[i], |it| it.p = np) { moved += 1; }
                }
                (px.len() as u64, moved)
            }
            Index::MortonKeep { g, last } => {
                let mut moved = 0u64;
                for i in 0..px.len() {
                    let np = Point::new(px[i] as f64, py[i] as f64);
                    let id = i as u32;
                    if let Crossed::Moved = g.update(last[i], |it| it.id == id, |it| it.p = np) { moved += 1; }
                    last[i] = np;
                }
                (px.len() as u64, moved)
            }
            Index::Adaptive { ix, slots } => {
                for i in 0..px.len() {
                    let np = Point::new(px[i] as f64, py[i] as f64);
                    ix.update(slots[i], |it| it.p = np);
                }
                ix.tick();
                (px.len() as u64, 0)
            }
            _ => { *self = Index::build(kind, px, py); (px.len() as u64, 0) }
        }
    }

    /// Does the index still hold every particle? A structure that silently dropped some would
    /// answer faster and wrongly, and the fps column cannot tell the difference — the trap
    /// docs/MEASURING.md records from the point-cloud demo ("an index only knows what it
    /// holds"). Returns (held, expected).
    fn held(&mut self, n: usize) -> (usize, usize) {
        // One cull big enough to sweep the whole tank: whatever it returns is what the index
        // actually holds.
        let all = Circle::new(Point::new(WW as f64 * 0.5, WH as f64 * 0.5), (WW + WH) as f64);
        match self {
            Index::Adaptive { ix, .. } => (ix.cull(&all).len(), n),
            Index::MortonKeep { g, .. } => (g.item_count(), n),
            Index::Morton(g) => (g.item_count(), n),
            _ => (n, n),
        }
    }

    /// Which structure the adaptive index is currently holding, for the HUD. `None` for the
    /// fixed choices, which have nothing to report.
    fn chosen(&self) -> Option<&'static str> {
        match self {
            Index::Adaptive { ix, .. } => Some(match ix.backend() {
                Backend::Brute => "scan", Backend::KeepTree => "tree",
                Backend::Grid => "grid", Backend::Static => "kdtree",
            }),
            _ => None,
        }
    }
    /// Neighbours of `q` within `H`, appended as ids. This is the "query" bar.
    fn neighbours(&self, q: Point, out: &mut Vec<u32>) {
        let c = Circle::new(q, H as f64);
        match self {
            Index::Morton(g) => out.extend(g.cull(&c).iter().map(|f| f.id)),
            Index::Keep { tree, .. } => out.extend(tree.cull(&c).iter().map(|f| f.id)),
            Index::MortonKeep { g, .. } => out.extend(g.cull(&c).iter().map(|f| f.id)),
            Index::Linear(t) => out.extend(t.cull(&c).iter().map(|f| f.id)),
            // `cull` takes &mut self here because the adaptive index may refresh a stale
            // backend before answering — the one place its API differs from a fixed structure.
            Index::Adaptive { .. } => unreachable!("adaptive neighbours go through neighbours_mut"),
        }
    }

    /// The adaptive index needs `&mut` to answer (it may rebuild a stale backend first), so it
    /// gets its own entry point rather than forcing every fixed structure to take `&mut`.
    fn neighbours_mut(&mut self, q: Point, out: &mut Vec<u32>) {
        match self {
            Index::Adaptive { ix, .. } => {
                let c = Circle::new(q, H as f64);
                out.extend(ix.cull(&c).iter().map(|f| f.id));
            }
            other => other.neighbours(q, out),
        }
    }
}

// ---- the fluid --------------------------------------------------------------
struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u32 << 24) as f32 }
    fn range(&mut self, a: f32, b: f32) -> f32 { a + (b - a) * self.f() }
}

struct Fluid {
    px: Vec<f32>, py: Vec<f32>,       // positions
    qx: Vec<f32>, qy: Vec<f32>,       // predicted positions (the PBF working set)
    vx: Vec<f32>, vy: Vec<f32>,
    lam: Vec<f32>,                    // per-particle constraint multiplier
    dx: Vec<f32>, dy: Vec<f32>,       // position correction of the current iteration
    rho: Vec<f32>,
    nbr: Vec<u32>, nbr_start: Vec<u32>,   // CSR neighbour lists, rebuilt once per step
    rho0: f32,                        // rest density, DERIVED from the initial packing
    grav: f32,
    /// Movements and real leaf-crossings, so the demo can report the relocation rate the
    /// advisor's rule is written in terms of.
    acc_moves: u64,
    acc_relocs: u64,
}

impl Fluid {
    /// A dam-break block in the left half - the standard SPH opening shot.
    fn new(n: usize) -> Fluid {
        let mut r = Rng(0x5EED_1234);
        let (mut px, mut py) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let cols = ((n as f32).sqrt() * 0.8).ceil().max(1.0) as usize;
        for i in 0..n {
            let (cx, cy) = ((i % cols) as f32, (i / cols) as f32);
            px.push(EPS + 8.0 + cx * SPACING + r.range(-0.3, 0.3));
            py.push(EPS + 8.0 + cy * SPACING + r.range(-0.3, 0.3));
        }
        let z = vec![0.0; n];
        let mut f = Fluid { acc_moves: 0, acc_relocs: 0,
            qx: px.clone(), qy: py.clone(), px, py,
            vx: z.clone(), vy: z.clone(), lam: z.clone(), dx: z.clone(), dy: z.clone(), rho: z,
            nbr: Vec::new(), nbr_start: Vec::new(), rho0: 1.0, grav: GRAV,
        };
        // Rest density is DERIVED, not guessed: it is the density the initial lattice
        // actually has (unit mass per particle). Guessing it - the classic tutorial
        // constants - is exactly how an SPH demo ends up with rho = 0 and particles at
        // Mach 1000, which is what the smoke test caught here.
        f.rho0 = f.lattice_density();
        f
    }
    fn n(&self) -> usize { self.px.len() }

    /// Density of a deep-interior particle in the initial lattice (unit mass).
    fn lattice_density(&self) -> f32 {
        let n = self.n();
        if n == 0 { return 1.0; }
        // the particle closest to the block centroid has a full neighbourhood
        let (mut sx, mut sy) = (0.0f32, 0.0f32);
        for i in 0..n { sx += self.px[i]; sy += self.py[i]; }
        let (cx, cy) = (sx / n as f32, sy / n as f32);
        let mut best = (f32::MAX, 0usize);
        for i in 0..n { let d = (self.px[i] - cx).powi(2) + (self.py[i] - cy).powi(2); if d < best.0 { best = (d, i); } }
        let i = best.1;
        let mut rho = 0.0;
        for j in 0..n {
            let (ex, ey) = (self.px[j] - self.px[i], self.py[j] - self.py[i]);
            rho += w_poly6(ex * ex + ey * ey);
        }
        rho.max(1e-6)
    }

    /// Keep a particle in the tank. The stand-off is jittered per particle: clamping
    /// everything to exactly EPS lines them up in a one-particle-wide column welded to
    /// the wall (visible in the first screenshot), and the density solver then pushes
    /// that column along the wall. A sub-particle spread breaks the alignment.
    #[inline]
    fn clamp_box(i: usize, x: &mut f32, y: &mut f32) {
        let j = (i % 5) as f32 * 0.55;
        *x = x.clamp(EPS + j, WW - EPS - j);
        *y = y.clamp(EPS + j, WH - EPS - j);
    }

    /// One PBF step. Returns (maintain us, query us, physics us) so the HUD can show
    /// where the frame actually goes.
    fn step(&mut self, index: &mut Index, kind: Idx) -> (f32, f32, f32) {
        let n = self.n();
        if n == 0 { return (0.0, 0.0, 0.0); }
        // ---- predict: external forces, then a tentative position
        for i in 0..n {
            self.vy[i] += DT * self.grav;
            let (mut qx, mut qy) = (self.px[i] + DT * self.vx[i], self.py[i] + DT * self.vy[i]);
            Self::clamp_box(i, &mut qx, &mut qy);
            self.qx[i] = qx; self.qy[i] = qy;
        }

        // ---- the index sees the PREDICTED positions (that is where contacts happen)
        let t0 = Instant::now();
        let (moves, relocs) = index.maintain(kind, &self.qx, &self.qy);
        self.acc_moves += moves;
        self.acc_relocs += relocs;
        let t1 = Instant::now();

        // ---- ONE neighbour pass per step, reused by every solver iteration
        self.nbr.clear();
        self.nbr_start.clear();
        self.nbr_start.reserve(n + 1);
        for i in 0..n {
            self.nbr_start.push(self.nbr.len() as u32);
            index.neighbours_mut(Point::new(self.qx[i] as f64, self.qy[i] as f64), &mut self.nbr);
        }
        self.nbr_start.push(self.nbr.len() as u32);
        let t2 = Instant::now();

        // ---- constraint solve
        let wdq = w_poly6(SCORR_DQ * SCORR_DQ).max(1e-12);
        for _ in 0..ITERS {
            for i in 0..n {
                let (s, e) = (self.nbr_start[i] as usize, self.nbr_start[i + 1] as usize);
                let (mut rho, mut sum_grad2, mut gix, mut giy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for &j in &self.nbr[s..e] {
                    let j = j as usize;
                    let (ex, ey) = (self.qx[i] - self.qx[j], self.qy[i] - self.qy[j]);
                    let r2 = ex * ex + ey * ey;
                    rho += w_poly6(r2);
                    if j == i { continue; }
                    let r = r2.sqrt();
                    let g = w_spiky_grad(r) / self.rho0;
                    if g == 0.0 { continue; }
                    let (gx, gy) = (g * ex / r, g * ey / r);
                    gix += gx; giy += gy;           // gradient w.r.t. particle i
                    sum_grad2 += gx * gx + gy * gy; // gradient w.r.t. each neighbour j
                }
                self.rho[i] = rho;
                sum_grad2 += gix * gix + giy * giy;
                // Free-surface fluid: only resolve OVER-density. Letting an
                // under-dense (surface) particle pull its neighbours in makes the
                // sheet collapse and, with a tiny epsilon, blow up.
                let c = (rho / self.rho0 - 1.0).max(0.0);
                self.lam[i] = -c / (sum_grad2 + RELAX);
            }
            for i in 0..n {
                let (s, e) = (self.nbr_start[i] as usize, self.nbr_start[i + 1] as usize);
                let (mut cx, mut cy) = (0.0f32, 0.0f32);
                for &j in &self.nbr[s..e] {
                    let j = j as usize;
                    if j == i { continue; }
                    let (ex, ey) = (self.qx[i] - self.qx[j], self.qy[i] - self.qy[j]);
                    let r2 = ex * ex + ey * ey;
                    let r = r2.sqrt();
                    let g = w_spiky_grad(r);
                    if g == 0.0 { continue; }
                    // artificial pressure: a tiny repulsion that stops particles
                    // stacking into clumps and gives a surface-tension look
                    let ratio = w_poly6(r2) / wdq;
                    let scorr = -SCORR_K * ratio.powi(SCORR_N);
                    let m = (self.lam[i] + self.lam[j] + scorr) * g / self.rho0;
                    cx += m * ex / r; cy += m * ey / r;
                }
                self.dx[i] = cx; self.dy[i] = cy;
            }
            for i in 0..n {
                let (mut cx, mut cy) = (self.dx[i] * SOR, self.dy[i] * SOR);
                let l = (cx * cx + cy * cy).sqrt();
                if l > MAX_CORR { let k = MAX_CORR / l; cx *= k; cy *= k; }
                self.qx[i] += cx;
                self.qy[i] += cy;
                let (mut x, mut y) = (self.qx[i], self.qy[i]);
                Self::clamp_box(i, &mut x, &mut y);
                self.qx[i] = x; self.qy[i] = y;
            }
        }

        // ---- velocity update (+ XSPH smoothing) and commit
        for i in 0..n {
            self.vx[i] = ((self.qx[i] - self.px[i]) / DT).clamp(-VMAX, VMAX);
            self.vy[i] = ((self.qy[i] - self.py[i]) / DT).clamp(-VMAX, VMAX);
        }
        for i in 0..n {
            let (s, e) = (self.nbr_start[i] as usize, self.nbr_start[i + 1] as usize);
            let (mut ax, mut ay) = (0.0f32, 0.0f32);
            for &j in &self.nbr[s..e] {
                let j = j as usize;
                if j == i { continue; }
                let (ex, ey) = (self.qx[i] - self.qx[j], self.qy[i] - self.qy[j]);
                let w = w_poly6(ex * ex + ey * ey);
                ax += (self.vx[j] - self.vx[i]) * w; ay += (self.vy[j] - self.vy[i]) * w;
            }
            self.dx[i] = XSPH / self.rho0 * ax;
            self.dy[i] = XSPH / self.rho0 * ay;
        }
        for i in 0..n {
            self.vx[i] = (self.vx[i] + self.dx[i]).clamp(-VMAX, VMAX);
            self.vy[i] = (self.vy[i] + self.dy[i]).clamp(-VMAX, VMAX);
            self.px[i] = self.qx[i]; self.py[i] = self.qy[i];
        }
        let t3 = Instant::now();
        let us = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1e6;
        (us(t0, t1), us(t1, t2), us(t2, t3))
    }

    /// The user's finger/cursor: shove everything near (x,y) along the drag.
    fn stir(&mut self, x: f32, y: f32, dx: f32, dy: f32) {
        let l = (dx * dx + dy * dy).sqrt();
        if l < 1e-4 { return; }
        let (ux, uy) = (dx / l, dy / l);
        let push = (l * 26.0).min(900.0);
        for i in 0..self.n() {
            let (ex, ey) = (self.px[i] - x, self.py[i] - y);
            let d2 = ex * ex + ey * ey;
            if d2 > STIR_R * STIR_R { continue; }
            let fall = 1.0 - (d2.sqrt() / STIR_R);
            self.vx[i] = (self.vx[i] + ux * push * fall).clamp(-VMAX, VMAX);
            self.vy[i] = (self.vy[i] + uy * push * fall).clamp(-VMAX, VMAX);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // `$FLUID_HEADLESS=<frames>` runs the simulation with no window, no GPU and no wgpu adapter
    // at all, then prints the same per-phase costs the HUD shows. The point is that this demo's
    // `M` key races five neighbour indexes against each other on a real workload, and that
    // comparison was only observable by a human watching a HUD — so it could not be run in CI,
    // on a machine without a display, or over a sweep of populations.
    //
    // It shares the exact simulation path (`Fluid::step`), not a copy of it: a headless mode
    // that reimplements the loop measures the reimplementation.
    if let Some(frames) = std::env::var("FLUID_HEADLESS").ok().and_then(|s| s.parse::<usize>().ok()) {
        headless(frames);
        return;
    }
    pollster::block_on(run());
}

/// Run `frames` simulation steps with no renderer and report the per-phase means.
///
/// Reports **means over the run, not the last frame** — the stealth demo learned that the hard
/// way, when a batch of three passes landed one reading on a frame that had not stepped and
/// printed a clean, plausible zero.
#[cfg(not(target_arch = "wasm32"))]
fn headless(frames: usize) {
    let n = std::env::var("FLUID_N").ok().and_then(|s| s.parse().ok()).unwrap_or(2200usize).min(MAX_N);
    let kind = match std::env::var("FLUID_INDEX").ok().as_deref() {
        Some("keep") | Some("tree") => Idx::TreeKeep,
        Some("linear") | Some("lqt") => Idx::Linear,
        Some("adaptive") | Some("auto") => Idx::Adaptive,
        Some("mortonkeep") | Some("gridkeep") => Idx::MortonKeep,
        _ => Idx::Morton,
    };
    let mut fluid = Fluid::new(n);
    let mut index = Index::build(kind, &fluid.px, &fluid.py);

    // A few unmeasured steps first: the first frame pays for every bucket the index has never
    // allocated, which is a build cost wearing a maintain cost's clothes.
    let warmup = (frames / 10).clamp(1, 60);
    for _ in 0..warmup { fluid.step(&mut index, kind); }

    let (mut acc_m, mut acc_q, mut acc_p) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..frames {
        let (m, q, p) = fluid.step(&mut index, kind);
        acc_m += m as f64; acc_q += q as f64; acc_p += p as f64;
    }
    let f = frames.max(1) as f64;
    let (m, q, p) = (acc_m / f, acc_q / f, acc_p / f);
    let picked = index.chosen().map(|c| format!(" -> {c}")).unwrap_or_default();
    println!("fluid headless | {n} particles | {frames} frames (+{warmup} warmup) | index: {}{picked}", kind.label());
    // `Fluid::step` reports MICROSECONDS (see its `us` closure) — labelling these ms was the
    // first thing this mode got wrong, and it read as 3 420 ms per frame without complaining.
    let frame_us = m + q + p;
    println!("  maintain {m:.1} us | query {q:.1} us | physics {p:.1} us | frame {frame_us:.1} us ({:.0} fps sim-only)",
        1e6 / frame_us.max(1e-9));
    // A stable machine-readable key per arm, so a sweep over `FLUID_INDEX` produces one row
    // per index rather than five rows all called "fluid".
    let tag = match kind {
        Idx::Morton => "morton", Idx::MortonKeep => "mortonkeep", Idx::TreeKeep => "treekeep",
        Idx::Linear => "linear", Idx::Adaptive => "adaptive",
    };
    println!("#M fluid_{tag}.maintain_us {m:.2} us");
    println!("#M fluid_{tag}.query_us {q:.2} us");
    println!("#M fluid_{tag}.physics_us {p:.2} us");
    println!("#M fluid_{tag}.frame_us {frame_us:.2} us");
    println!("#M fluid_{tag}.sim_fps {:.0} n", 1e6 / frame_us.max(1e-9));
}

#[cfg(target_arch = "wasm32")]
fn headless(_frames: usize) {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() { console_error_panic_hook::set_once(); wasm_bindgen_futures::spawn_local(run()); }

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("vectorial-hash fluid SPH").with_inner_size(winit::dpi::LogicalSize::new(1300, 900)).build(&event_loop).unwrap());
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas().expect("canvas");
        let _ = canvas.set_attribute("style", "width:100vw;height:100vh;display:block;touch-action:none");
        web_sys::window().and_then(|w| w.document()).and_then(|d| d.body()).expect("body").append_child(&canvas.into()).expect("append canvas");
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false }).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits() }, None).await.unwrap();
    let size = window.inner_size();
    let dpr = window.scale_factor() as f32;
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: wgpu::PresentMode::AutoNoVsync, desired_maximum_frame_latency: 2, alpha_mode: caps.alpha_modes[0], view_formats: vec![] };
    surface.configure(&device, &config);

    // ---- state
    let mut n = std::env::var("FLUID_N").ok().and_then(|s| s.parse().ok()).unwrap_or(2200usize).min(MAX_N);
    let mut kind = match std::env::var("FLUID_INDEX").ok().as_deref() {
        Some("keep") | Some("tree") => Idx::TreeKeep,
        Some("linear") | Some("lqt") => Idx::Linear,
        Some("adaptive") | Some("auto") => Idx::Adaptive,
        Some("mortonkeep") | Some("gridkeep") => Idx::MortonKeep,
        _ => Idx::Morton,
    };
    let mut fluid = Fluid::new(n);
    let mut index = Index::build(kind, &fluid.px, &fluid.py);
    let substeps = std::env::var("FLUID_SUBSTEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1usize); // PBF is stable at one step per frame

    // ---- gpu resources
    let inst_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("inst"), size: (MAX_N * std::mem::size_of::<Inst>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("cam"), size: std::mem::size_of::<Cam>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cam_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_b.as_entire_binding() }] });
    let rmod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(RENDER.into()) });
    let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&cam_bgl], push_constant_ranges: &[] });
    let render_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("drops"), layout: Some(&rpl),
        vertex: wgpu::VertexState { module: &rmod, entry_point: "vs", compilation_options: Default::default(), buffers: &[
            wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Inst>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x4] },
        ] },
        fragment: Some(wgpu::FragmentState { module: &rmod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None, multisample: Default::default(), multiview: None,
    });
    let ui_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()) });
    let ui_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], push_constant_ranges: &[] });
    let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ui"), layout: Some(&ui_pl),
        vertex: wgpu::VertexState { module: &ui_mod, entry_point: "vs", compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<UiVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4] }] },
        fragment: Some(wgpu::FragmentState { module: &ui_mod, entry_point: "fs", compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None, multisample: Default::default(), multiview: None,
    });
    let ui_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ui"), size: (60_000 * std::mem::size_of::<UiVertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let smoke: Option<u64> = std::env::var("FLUID_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    let (mut maint_us, mut query_us, mut phys_us) = (0.0f32, 0.0f32, 0.0f32);
    let (mut paused, mut frame, mut fps) = (false, 0u64, 0.0f32);
    let mut last = Instant::now();
    let (mut stirring, mut cursor, mut prev_cursor) = (false, (0.0f32, 0.0f32), (0.0f32, 0.0f32));
    let mut inst: Vec<Inst> = Vec::with_capacity(MAX_N);

    let _ = event_loop.run(move |event, elwt| {
        // Screen (physical px) → world, matching the aspect-preserving ortho below.
        let to_world = |sx: f32, sy: f32, cw: f32, ch: f32| {
            let (vw, vh) = view_box(cw, ch);
            let (cx, cy) = (WW * 0.5, WH * 0.5);
            (cx - vw * 0.5 + sx / cw * vw, cy + vh * 0.5 - sy / ch * vh)
        };
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(s) => { config.width = s.width.max(1); config.height = s.height.max(1); surface.configure(&device, &config); }
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    stirring = state == ElementState::Pressed;
                    prev_cursor = cursor;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    cursor = to_world(position.x as f32, position.y as f32, config.width as f32, config.height as f32);
                }
                // Touch: a finger drag stirs exactly like the mouse (the whole point of
                // the demo on a phone), so treat it as press/move/release.
                WindowEvent::Touch(t) => {
                    let p = to_world(t.location.x as f32, t.location.y as f32, config.width as f32, config.height as f32);
                    match t.phase {
                        TouchPhase::Started => { stirring = true; cursor = p; prev_cursor = p; }
                        TouchPhase::Moved => { cursor = p; }
                        TouchPhase::Ended | TouchPhase::Cancelled => { stirring = false; }
                    }
                }
                WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(c), state: ElementState::Pressed, .. }, .. } => match c {
                    KeyCode::KeyM => { kind = kind.next(); index = Index::build(kind, &fluid.px, &fluid.py); }
                    KeyCode::KeyP => paused = !paused,
                    KeyCode::KeyR => { fluid = Fluid::new(n); index = Index::build(kind, &fluid.px, &fluid.py); }
                    KeyCode::KeyG => fluid.grav = -fluid.grav,
                    KeyCode::BracketRight | KeyCode::BracketLeft => {
                        n = if c == KeyCode::BracketRight { (n + 400).min(MAX_N) } else { n.saturating_sub(400).max(200) };
                        fluid = Fluid::new(n);
                        index = Index::build(kind, &fluid.px, &fluid.py);
                    }
                    _ => {}
                },
                WindowEvent::RedrawRequested => {
                    let fdt = { let d = last.elapsed().as_secs_f32().min(0.05); last = Instant::now(); d };
                    fps = if fps == 0.0 { 1.0 / fdt } else { fps * 0.9 + 0.1 / fdt };
                    frame += 1;

                    if !paused {
                        let (mut a, mut b, mut c2) = (0.0, 0.0, 0.0);
                        for _ in 0..substeps { let (x, y, z) = fluid.step(&mut index, kind); a += x; b += y; c2 += z; }
                        if stirring {
                            let (dx, dy) = (cursor.0 - prev_cursor.0, cursor.1 - prev_cursor.1);
                            fluid.stir(cursor.0, cursor.1, dx, dy);
                        }
                        prev_cursor = cursor;
                        // smooth so the bars are readable rather than jittery
                        maint_us = maint_us * 0.85 + a * 0.15;
                        query_us = query_us * 0.85 + b * 0.15;
                        phys_us = phys_us * 0.85 + c2 * 0.15;
                    }

                    // ---- camera: fit the tank, preserving aspect
                    let (cw, ch) = (config.width as f32, config.height as f32);
                    let (vw, vh) = view_box(cw, ch);
                    let (cx, cy) = (WW * 0.5, WH * 0.5);
                    let proj = Mat4::orthographic_rh(cx - vw * 0.5, cx + vw * 0.5, cy - vh * 0.5, cy + vh * 0.5, -1.0, 1.0);
                    queue.write_buffer(&cam_b, 0, bytemuck::bytes_of(&Cam { vp: proj.to_cols_array_2d() }));

                    // ---- instances: colour by speed (deep blue → foam white)
                    inst.clear();
                    for i in 0..fluid.n() {
                        let sp = (fluid.vx[i] * fluid.vx[i] + fluid.vy[i] * fluid.vy[i]).sqrt();
                        inst.push(Inst { x: fluid.px[i], y: fluid.py[i], r: H * 0.42, heat: (sp / 900.0).min(1.0) });
                    }
                    queue.write_buffer(&inst_b, 0, bytemuck::cast_slice(&inst));

                    // ---- HUD: where the frame goes + what's driving it
                    let tp = 3.0 * dpr.clamp(1.0, 3.0);
                    let mut ui: Vec<UiVertex> = Vec::new();
                    let pad = 6.0 * tp;
                    // Panel rows: title, hint, then one (label + bar) block per phase.
                    // Each block owns 11*tp so the label never lands on the bar above.
                    let (row, bar_h, bar_w) = (11.0 * tp, 4.0 * tp, 100.0 * tp);
                    let y0 = pad + 16.0 * tp;
                    push_quad(&mut ui, pad, pad, 110.0 * tp, y0 - pad + 3.0 * row, [0.03, 0.06, 0.12, 0.62], cw, ch);
                    push_text(&mut ui, pad + 3.0 * tp, pad + 2.0 * tp, tp, [1.0, 1.0, 1.0, 1.0], &format!("{} {:.0}FPS", kind.label(), fps), cw, ch);
                    push_text(&mut ui, pad + 3.0 * tp, pad + 9.0 * tp, tp * 0.8, [0.72, 0.80, 0.95, 0.9], &format!("{} DROPS - DRAG TO STIR - M INDEX", fluid.n()), cw, ch);
                    let total = (maint_us + query_us + phys_us).max(1.0);
                    let bars = [([0.40, 0.70, 1.0, 0.95], maint_us, "MAINTAIN"), ([1.0, 0.72, 0.30, 0.95], query_us, "QUERY"), ([0.45, 0.95, 0.55, 0.95], phys_us, "PHYSICS")];
                    for (i, (col, us, label)) in bars.iter().enumerate() {
                        let ty = y0 + i as f32 * row;
                        push_text(&mut ui, pad + 3.0 * tp, ty, tp * 0.8, [0.85, 0.9, 1.0, 0.95], &format!("{label} {:.2}MS", us / 1000.0), cw, ch);
                        let by = ty + 5.0 * tp;
                        push_quad(&mut ui, pad + 3.0 * tp, by, bar_w, bar_h, [0.13, 0.15, 0.22, 0.85], cw, ch);
                        push_quad(&mut ui, pad + 3.0 * tp, by, bar_w * (us / total), bar_h, *col, cw, ch);
                    }
                    queue.write_buffer(&ui_buf, 0, bytemuck::cast_slice(&ui));
                    let ui_count = ui.len() as u32;

                    let frame_tex = match surface.get_current_texture() { Ok(f) => f, Err(_) => { surface.configure(&device, &config); return; } };
                    let view_tex = frame_tex.texture.create_view(&Default::default());
                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("main"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view_tex, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.03, b: 0.06, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                        });
                        rp.set_pipeline(&render_pipe);
                        rp.set_bind_group(0, &cam_bg, &[]);
                        rp.set_vertex_buffer(0, inst_b.slice(..));
                        rp.draw(0..6, 0..inst.len() as u32);
                        if ui_count > 0 { rp.set_pipeline(&ui_pipeline); rp.set_vertex_buffer(0, ui_buf.slice(..)); rp.draw(0..ui_count, 0..1); }
                    }
                    queue.submit(Some(enc.finish()));
                    frame_tex.present();

                    if let Some(max) = smoke {
                        if frame >= max {
                            // Self-check: I can't see the window, so the smoke run reports the
                            // health of the SIMULATION too — a blown-up SPH shows as NaNs, a
                            // collapsed mean density, or a runaway top speed.
                            let n_f = fluid.n().max(1) as f32;
                            // PBF normalises by rho0, so the ABSOLUTE density is an
                            // arbitrary kernel scale — the meaningful health number is
                            // the ratio to rest: ~1.0 means incompressible, >>1 clumped,
                            // <<1 blown apart.
                            let ratio = |r: f32| r / fluid.rho0;
                            let mean = ratio(fluid.rho.iter().sum::<f32>() / n_f);
                            let hi = ratio(fluid.rho.iter().copied().fold(0.0f32, f32::max));
                            let vmax = fluid.vx.iter().zip(fluid.vy.iter()).map(|(a, b)| (a * a + b * b).sqrt()).fold(0.0f32, f32::max);
                            let bad = fluid.px.iter().chain(fluid.py.iter()).chain(fluid.vx.iter()).any(|v| !v.is_finite());
                            let inside = fluid.px.iter().zip(fluid.py.iter()).filter(|(x, y)| **x >= 0.0 && **x <= WW && **y >= 0.0 && **y <= WH).count();
                            // What the adaptive index settled on. Reporting the choice matters
                            // as much as the time: an index that picks well once and never
                            // migrates is a different (and cheaper) claim than one that keeps
                            // adapting, and only this line tells them apart.
                            let picked = index.chosen().map(|c| format!(" -> {c}")).unwrap_or_default();
                            let (held, want) = index.held(fluid.n());
                            if held != want { println!("  !! index holds {held} of {want} particles — it is answering about a different set"); }
                            println!("fluid_wgpu end-to-end: {:.1} fps avg ({} drops, {}{}, maintain {:.2} ms / query {:.2} ms / physics {:.2} ms per frame)",
                                fps, fluid.n(), kind.label(), picked, maint_us / 1000.0, query_us / 1000.0, phys_us / 1000.0);
                            // machine-readable for `bench-runner` (keyed by index, so a
                            // sweep over FLUID_INDEX lands in one comparable table)
                            let tag = kind.label().split_whitespace().next().unwrap_or("?").to_lowercase();
                            println!("#M {tag}.maintain {:.4} ms", maint_us / 1000.0);
                            println!("#M {tag}.query {:.4} ms", query_us / 1000.0);
                            println!("#M {tag}.physics {:.4} ms", phys_us / 1000.0);
                            println!("#M {tag}.fps {fps:.1} fps");
                            // The numbers the advisor's rules are written in terms of, for
                            // the one workload in this repo where keeping the index loses.
                            let (mv, rl) = (fluid.acc_moves.max(1), fluid.acc_relocs);
                            let reloc_rate = rl as f64 / mv as f64;
                            let q_per_item = 1.0; // PBF asks for the neighbours of every particle, every step
                            println!("#M {tag}.relocation_rate {reloc_rate:.4} frac");
                            println!("#M {tag}.queries_per_item {q_per_item:.3} q");
                            println!("  advisor check: relocation rate {:.1}% (HIGH_RELOCATION {:.0}%) | queries per item {:.2} (rebuild_query_ratio {:.2})",
                                reloc_rate * 100.0, vectorial_hash::advisor::HIGH_RELOCATION * 100.0,
                                q_per_item, vectorial_hash::Thresholds::default().rebuild_query_ratio);
                            println!("  sim health: density {mean:.2}x rest (peak {hi:.2}x) - vmax {vmax:.0} - in-tank {inside}/{} - finite {}",
                                fluid.n(), !bad);
                            elwt.exit();
                        }
                    }
                    window.request_redraw();
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    });
}

/// The visible world box for a viewport, preserving the tank's aspect (letterboxed).
fn view_box(cw: f32, ch: f32) -> (f32, f32) {
    let (aw, at) = (cw / ch.max(1.0), WW / WH);
    if aw > at { (WH * aw, WH) } else { (WW, WW / aw.max(1e-3)) }
}

const RENDER: &str = r#"
struct Cam { vp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VO { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) heat: f32 };
@vertex
fn vs(@location(0) inst: vec4<f32>, @builtin(vertex_index) vi: u32) -> VO {
    var corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    let c = corners[vi];
    // the quad is built in WORLD units, so the ortho matrix handles aspect for us
    let world = inst.xy + c * inst.z;
    var o: VO;
    o.clip = cam.vp * vec4<f32>(world, 0.0, 1.0);
    o.uv = c;
    o.heat = inst.w;
    return o;
}
@fragment
fn fs(v: VO) -> @location(0) vec4<f32> {
    let d = dot(v.uv, v.uv);
    if (d > 1.0) { discard; }
    // soft edge + a fake specular lift so the blob reads as liquid, not a flat disc
    let edge = smoothstep(1.0, 0.35, d);
    let deep = vec3<f32>(0.10, 0.35, 0.85);
    let foam = vec3<f32>(0.85, 0.95, 1.0);
    let col = mix(deep, foam, clamp(v.heat, 0.0, 1.0)) + vec3<f32>(0.18) * pow(1.0 - d, 3.0);
    return vec4<f32>(col, edge);
}
"#;

const UI_SHADER: &str = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) color: vec4<f32>) -> VOut { var o: VOut; o.clip = vec4<f32>(p, 0.0, 1.0); o.color = color; return o; }
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"#;
