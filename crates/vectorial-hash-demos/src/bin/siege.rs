//! `siege` — a medieval-battlefield showcase for vectorial-hash in 3D.
//!
//! Two castles at opposite corners of a procedurally-generated battlefield
//! (rolling hills + a central volcano) sally their armies; the two factions
//! advance, clash in the middle, and respawn from their keep when they fall —
//! a continuous battle that never empties. The point is not the graphics (rough
//! by design — billboarded instanced spheres) but the **workload**: every unit,
//! every frame, runs read-only spatial queries on a single shared `Tree3`:
//!
//! - **targeting** — `knn` to find the nearest *enemy* (filter the k-NN result
//!   by faction);
//! - **area attacks** — the dragon's fire-breath is a `Sphere3` `cull` (one
//!   query, every enemy caught takes damage);
//! - (next layers: archer line-of-fire via `raycast`, smoke that blocks it,
//!   boids formations, and the parallel per-unit AI pass.)
//!
//! Combat uses the **parallel-safe split**: a *decide* pass reads the index and
//! writes each unit's intent into *its own* fields only (no cross-unit writes,
//! so it parallelises with `par_iter_mut`); a serial *apply* pass then resolves
//! damage, deaths and respawns. See `docs/PARALLEL.md` § "Per-unit AI".
//!
//! Run: `cargo run -p vectorial-hash-demos --bin siege --release`
//!  - drag left mouse: orbit the camera; scroll: zoom
//!  - `[` / `]`: smaller / larger armies (rebuild)
//!  - `P`: pause / resume the simulation

use macroquad::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use vectorial_hash::{Aabb, Point3, Tree3};
use vectorial_hash_demos::instanced3d::{EffectInstance, InstancedRenderer, ModelGpu};
use vectorial_hash_demos::model::{load_glb, load_glb_clip};
// The whole simulation — Unit/Kind/Faction, decide/apply, terrain, projectiles,
// effects, the volcano, the model→bytes map — lives in the shared `siege_sim`
// module so this macroquad renderer and the wgpu one stay in lockstep. This file
// is render-only (macroquad mesh/draw code, the main loop, the UI).
use vectorial_hash_demos::siege_sim::*;

// --------------------------------------------------------------- render config

/// `Point3` → macroquad `Vec3` (for drawing sim positions).
fn v3(p: Point3) -> Vec3 { vec3(p.x as f32, p.y as f32, p.z as f32) }
/// `[f32;3]` → macroquad `Vec3` (for drawing Fx endpoints).
fn a3(a: [f32; 3]) -> Vec3 { vec3(a[0], a[1], a[2]) }

/// Per-(faction,kind) baked clips: movement (or idle for riders) + attack.
struct UnitAnim { mov: Vec<ModelGpu>, atk: Vec<ModelGpu> }

// ── Everything simulation-side (Rng, terrain, Faction/model_for/Kind, Unit/IUnit,
//    smoke, Fx + projectiles, spawning, decide/apply, the volcano) moved to the
//    shared `siege_sim` module, imported above. What remains is render-only. ──

// ----------------------------------------------------------------- rendering

/// Procedural "animation" offset for a unit's model this frame (cheap, scales to
/// the whole army — no skeletal skinning): a walk-bounce while moving (idle
/// breathe when stopped) plus a forward lunge during an attack. `h` is the
/// model height; returns a world-space translation to add to the unit's base.
fn anim_offset(u: &Unit, now: f64, h: f32) -> Vec3 {
    let moving = u.vel.0 * u.vel.0 + u.vel.2 * u.vel.2 > 1.0;
    let t = now as f32 * 9.0 + u.phase;
    // Small now that the skeleton itself animates — just a touch of extra life.
    let bob = if moving { t.sin().abs() * 0.03 * h } else { (now as f32 * 1.6 + u.phase).sin() * 0.015 * h };
    let prog = if u.atk_anim > 0.0 { 1.0 - u.atk_anim / ATK_ANIM_LEN } else { 0.0 };
    let lunge = (prog * std::f32::consts::PI).sin() * h * 0.16;
    vec3(0.0, bob, 0.0) + vec3(u.face.sin(), 0.0, u.face.cos()) * lunge
}

/// Build the terrain once as smooth triangle `Mesh` **chunks**. Each vertex sits
/// at its true height (so the surface is continuous, not stepped like the old
/// per-tile cubes), and **lambert shading is baked into the vertex colour** — the
/// slope normal dotted with a sun direction — because macroquad's `draw_mesh` is
/// unlit. That shading is what makes the relief readable when the terrain is
/// otherwise one flat green. Lava (crater + flow) is emissive: drawn full-bright.
///
/// Chunked because macroquad clamps any single drawcall at 10 000 verts / 5 000
/// indices; a 6×6 grid of 25-cell chunks (676 verts / 3 750 indices each) stays
/// under both caps.
fn build_terrain_chunks() -> Vec<Mesh> {
    const RES: usize = 150;
    const CHUNK: usize = 25; // cells/side: (CHUNK+1)²=676 verts, CHUNK²·6=3 750 indices
    let step = WORLD / RES as f64;
    let light = vec3(-0.45, 0.84, -0.30).normalize();
    let nchunks = RES / CHUNK;
    let mut meshes = Vec::with_capacity(nchunks * nchunks);
    for cz in 0..nchunks {
        for cx in 0..nchunks {
            let (ix0, iz0) = (cx * CHUNK, cz * CHUNK);
            let mut vertices = Vec::with_capacity((CHUNK + 1) * (CHUNK + 1));
            for jz in 0..=CHUNK {
                for jx in 0..=CHUNK {
                    let (x, z) = ((ix0 + jx) as f64 * step, (iz0 + jz) as f64 * step);
                    let h = terrain_height(x, z);
                    // Heightfield normal via central differences.
                    let hx = terrain_height(x + step, z) - terrain_height(x - step, z);
                    let hz = terrain_height(x, z + step) - terrain_height(x, z - step);
                    let n = vec3((-hx / (2.0 * step)) as f32, 1.0, (-hz / (2.0 * step)) as f32).normalize();
                    let (base, emissive) = terrain_surface(x, z, h);
                    let col = if emissive {
                        Color::new(base[0], base[1], base[2], 1.0)
                    } else {
                        let b = 0.32 + 0.68 * n.dot(light).max(0.0); // ambient + diffuse
                        Color::new(base[0] * b, base[1] * b, base[2] * b, 1.0)
                    };
                    vertices.push(Vertex::new(x as f32, h as f32, z as f32, 0.0, 0.0, col));
                }
            }
            let w = (CHUNK + 1) as u16;
            let mut indices: Vec<u16> = Vec::with_capacity(CHUNK * CHUNK * 6);
            for jz in 0..CHUNK as u16 {
                for jx in 0..CHUNK as u16 {
                    let a = jz * w + jx;
                    indices.extend_from_slice(&[a, a + w, a + 1, a + 1, a + w, a + w + 1]);
                }
            }
            meshes.push(Mesh { vertices, indices, texture: None });
        }
    }
    meshes
}

/// Voxel (blocky) terrain. Each grid cell is a flat-topped prism at its true
/// height, with vertical cliff walls dropping to any lower neighbour — the
/// stepped "voxel" look, watertight, units sit on the tops. Baked per-corner
/// ambient occlusion (darker where taller neighbours crowd a corner) with the
/// quad-flip fix from MAP_DESIGN.md, plus the elevation colour ramp. Chunked so
/// each mesh stays under macroquad's per-draw-call index cap. The cell heights
/// are a grid → alterable (crater carving) is a future step (BACKLOG).
fn build_voxel_chunks(craters: &[(f32, f32, f32)]) -> Vec<Mesh> {
    const VRES: usize = 80;  // cells/side: cell ≈ 10 world units → visibly blocky
    const CHUNK: usize = 10; // cells/side per mesh: ≤100 cells · ≤30 idx = <5000
    let step = WORLD / VRES as f64;
    let light = vec3(-0.45, 0.84, -0.30).normalize();
    // Carve craters (x, z, radius) as a smooth bowl out of a cell's height.
    let carve = |x: f64, z: f64, h: f64| -> f64 {
        let mut h = h;
        for &(cx, cz, cr) in craters {
            let d = (((x - cx as f64).powi(2) + (z - cz as f64).powi(2)).sqrt()) as f32;
            if d < cr { h -= (cr * 0.45 * (1.0 - d / cr)) as f64; } // cone bowl, ~0.45·r deep
        }
        h
    };
    // Precompute the cell-centre heights once for O(1) neighbour AO lookups.
    let heights: Vec<f64> = (0..VRES * VRES).map(|k| {
        let (i, j) = (k % VRES, k / VRES);
        let (x, z) = ((i as f64 + 0.5) * step, (j as f64 + 0.5) * step);
        carve(x, z, terrain_height(x, z))
    }).collect();
    let hc = |i: i32, j: i32| -> f64 {
        if i < 0 || j < 0 || i >= VRES as i32 || j >= VRES as i32 { -1000.0 } else { heights[j as usize * VRES + i as usize] }
    };
    let nchunks = VRES.div_ceil(CHUNK);
    let mut meshes = Vec::with_capacity(nchunks * nchunks);
    for cz in 0..nchunks {
        for cx in 0..nchunks {
            let (i0, i1) = (cx * CHUNK, ((cx + 1) * CHUNK).min(VRES));
            let (j0, j1) = (cz * CHUNK, ((cz + 1) * CHUNK).min(VRES));
            let mut vertices: Vec<Vertex> = Vec::new();
            let mut indices: Vec<u16> = Vec::new();
            for j in j0..j1 {
                for i in i0..i1 {
                    let (ii, jj) = (i as i32, j as i32);
                    let h = hc(ii, jj);
                    let (x0, z0) = (i as f64 * step, j as f64 * step);
                    let (x1, z1) = (x0 + step, z0 + step);
                    let (base, emissive) = terrain_surface(x0 + step * 0.5, z0 + step * 0.5, h);
                    let col = |b: f32| Color::new((base[0] * b).min(1.0), (base[1] * b).min(1.0), (base[2] * b).min(1.0), 1.0);
                    // Per-corner AO: a corner darkens for each taller neighbour cell
                    // touching it (edge neighbours + the diagonal if an edge is up).
                    let ao = |dx: i32, dz: i32| -> f32 {
                        if emissive { return 1.0; }
                        let up = h + step * 0.5;
                        let (s1, s2) = (hc(ii + dx, jj) > up, hc(ii, jj + dz) > up);
                        let sc = hc(ii + dx, jj + dz) > up;
                        let n = s1 as i32 + s2 as i32 + (sc && (s1 || s2)) as i32;
                        1.0 - 0.16 * n as f32
                    };
                    let topb = if emissive { 1.0 } else { 0.32 + 0.68 * light.y.max(0.0) as f32 };
                    let corners = [(x0, z0, ao(-1, -1)), (x1, z0, ao(1, -1)), (x1, z1, ao(1, 1)), (x0, z1, ao(-1, 1))];
                    let bi = vertices.len() as u16;
                    for (vx, vz, a) in corners { vertices.push(Vertex::new(vx as f32, h as f32, vz as f32, 0.0, 0.0, col(topb * a))); }
                    // Quad-flip: split along the diagonal joining the brighter pair
                    // so AO interpolates symmetrically (no dark-corner triangle leak).
                    if corners[0].2 + corners[2].2 >= corners[1].2 + corners[3].2 {
                        indices.extend_from_slice(&[bi, bi + 1, bi + 2, bi, bi + 2, bi + 3]);
                    } else {
                        indices.extend_from_slice(&[bi + 1, bi + 2, bi + 3, bi + 1, bi + 3, bi]);
                    }
                    // Cliff walls: one quad per lower neighbour, top at h down to it.
                    let sideb = if emissive { 1.0 } else { 0.32 + 0.68 * 0.42 };
                    for (dx, dz, e0, e1) in [(-1i32, 0i32, (x0, z1), (x0, z0)), (1, 0, (x1, z0), (x1, z1)), (0, -1, (x0, z0), (x1, z0)), (0, 1, (x1, z1), (x0, z1))] {
                        let hn = hc(ii + dx, jj + dz);
                        if hn < h - 0.01 {
                            let bottom = hn.max(-25.0) as f32;
                            let (top, bot) = (col(sideb), col(sideb * 0.65));
                            let si = vertices.len() as u16;
                            vertices.push(Vertex::new(e0.0 as f32, h as f32, e0.1 as f32, 0.0, 0.0, top));
                            vertices.push(Vertex::new(e1.0 as f32, h as f32, e1.1 as f32, 0.0, 0.0, top));
                            vertices.push(Vertex::new(e1.0 as f32, bottom, e1.1 as f32, 0.0, 0.0, bot));
                            vertices.push(Vertex::new(e0.0 as f32, bottom, e0.1 as f32, 0.0, 0.0, bot));
                            indices.extend_from_slice(&[si, si + 1, si + 2, si, si + 2, si + 3]);
                        }
                    }
                }
            }
            if !vertices.is_empty() { meshes.push(Mesh { vertices, indices, texture: None }); }
        }
    }
    meshes
}

/// Wooden bridge decks across the river (a plank + two rails per crossing).
fn draw_bridges() {
    let deck = Color::new(0.36, 0.24, 0.13, 1.0);
    let rail = Color::new(0.28, 0.18, 0.10, 1.0);
    for &bz in &BRIDGE_Z {
        let bx = river_center_x(bz);
        let y = 9.0f32;
        draw_cube(vec3(bx as f32, y, bz as f32), vec3((BRIDGE_HALF_W * 2.0) as f32, 2.5, (BRIDGE_HALF_D * 2.0) as f32), None, deck);
        for s in [-1.0, 1.0] {
            draw_cube(vec3(bx as f32, y + 3.0, (bz + s * BRIDGE_HALF_D) as f32), vec3((BRIDGE_HALF_W * 2.0) as f32, 4.0, 1.5), None, rail);
        }
    }
}

/// Draw the transient combat effects (the visible part of each attack): arrow /
/// bolt / lightning streaks, a healer's spark, and an expanding AoE ring. Immediate
/// 3D lines, faded by age.
fn draw_effects(effects: &[Fx], now: f64) {
    for f in effects {
        let age = ((now - f.born) / Fx::life(f.kind)).clamp(0.0, 1.0) as f32;
        let (a, b) = (a3(f.a), a3(f.b)); // Fx endpoints are graphics-free [f32;3]
        match f.kind {
            FxKind::Arrow => draw_line_3d(a, b, Color::new(0.96, 0.90, 0.45, 1.0)),
            FxKind::Bolt => draw_line_3d(a, b, Color::new(1.0, 0.58, 0.16, 1.0)),
            FxKind::Lightning => draw_line_3d(a, b, Color::new(0.62, 0.86, 1.0, 1.0)),
            FxKind::Spark => draw_line_3d(a, a + vec3(0.0, 7.0, 0.0), Color::new(0.40, 1.0, 0.55, 1.0)),
            FxKind::Ring => {
                let r = 8.0 + 26.0 * age; // expanding shockwave
                let col = Color::new(1.0, 0.45, 0.12, 1.0 - age);
                let n = 22;
                let mut prev = a + vec3(r, 1.5, 0.0);
                for i in 1..=n {
                    let t = i as f32 / n as f32 * std::f32::consts::TAU;
                    let p = a + vec3(r * t.cos(), 1.5, r * t.sin());
                    draw_line_3d(prev, p, col);
                    prev = p;
                }
            }
        }
    }
}

/// A minimal screen-space slider over an integer range `min..=max`. Draggable
/// handle; updates `*value` and sets `*dragging` so the caller can suppress
/// camera-orbit while it's held.
#[allow(clippy::too_many_arguments)]
fn int_slider(x: f32, y: f32, w: f32, label: &str, value: &mut usize, min: usize, max: usize, dragging: &mut bool) {
    draw_rectangle(x, y - 3.0, w, 6.0, Color::new(0.30, 0.30, 0.36, 1.0));
    let span = (max - min).max(1) as f32;
    let hx = x + ((*value).saturating_sub(min) as f32 / span) * w;
    draw_circle(hx, y, 9.0, WHITE);
    let (mx, my) = mouse_position();
    let over = mx >= x - 12.0 && mx <= x + w + 12.0 && (my - y).abs() < 16.0;
    if is_mouse_button_pressed(MouseButton::Left) && over { *dragging = true; }
    if !is_mouse_button_down(MouseButton::Left) { *dragging = false; }
    if *dragging {
        let nt = ((mx - x) / w).clamp(0.0, 1.0);
        *value = min + (nt * span).round() as usize;
    }
    draw_text(format!("{label}: {value}"), x, y - 14.0, 20.0, WHITE);
}

fn window_conf() -> Conf {
    Conf {
        window_title: "vectorial-hash — siege".to_owned(),
        window_width: 1600,
        window_height: 1000,
        platform: macroquad::miniquad::conf::Platform { swap_interval: Some(0), ..Default::default() },
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Per-run seed: $SIEGE_SEED (reproducible) or the wall clock (varies). Drives
    // both the map (terrain noise offset) and the army composition.
    let seed = std::env::var("SIEGE_SEED").ok().and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| (macroquad::miniquad::date::now() * 1000.0) as u64);
    set_map_seed((seed % 100_000) as f64 * 0.01);
    let mut rng = Rng::new(seed | 1);
    let mut per_faction = PER_FACTION; // live army size per side (population slider)
    let mut cur_pop = per_faction;
    let mut pop_drag = false;
    let mut units = spawn_army(&mut rng, per_faction);
    let world = Aabb::new(0.0, 0.0, 0.0, WORLD, SKY, WORLD);
    let mut index = Tree3::<IUnit>::new(world, 8);
    // Smoke lives in its own index so archer/ballista shots can raycast it.
    let mut smoke: Vec<Puff> = Vec::new();
    let mut effects: Vec<Fx> = Vec::new(); // transient combat visuals
    let mut projectiles: Vec<Projectile> = Vec::new(); // arcing cannonballs / lava bombs
    let mut smoke_index = Tree3::<Puff>::new(world, 8);

    // Camera orbit state — looking down on the battlefield centre.
    let mut yaw: f32 = 0.9;
    let mut pitch: f32 = 0.75;
    let mut dist: f32 = 900.0;
    let observer = vec3((WORLD * 0.5) as f32, 60.0, (WORLD * 0.5) as f32);
    let mut last_mouse = mouse_position();

    let mut renderer = {
        let gl = unsafe { get_internal_gl() };
        InstancedRenderer::new(gl.quad_context)
    };

    // Load + upload a glTF model per (faction, kind) — pirates vs undead. While
    // loading, derive each one's world-space body radius from its own XZ footprint
    // × render height, so the space a unit occupies (separation) matches what's
    // drawn. Indexed [faction][kind].
    let mut body_radius = [[4.0f64; 8]; 2];
    let mut frames_mov = [[1usize; 8]; 2]; // movement-clip frame count per (f, k)
    let mut frames_atk = [[1usize; 8]; 2]; // attack-clip frame count per (f, k)
    let models: Vec<((Faction, Kind), UnitAnim)> = {
        let gl = unsafe { get_internal_gl() };
        let mut v = Vec::with_capacity(16);
        for &f in &[Faction::Red, Faction::Blue] {
            for &k in &Kind::ALL {
                // Riders sit on the horse → their "movement" clip is the idle (no
                // running legs); everyone else walks/runs/flies.
                let mov_prefs = if k == Kind::Knight { IDLE_PREFS } else { MOVE_PREFS };
                let mov_cpu = load_glb_clip(model_for(f, k), ANIM_FRAMES, mov_prefs);
                let atk_cpu = load_glb_clip(model_for(f, k), ATTACK_FRAMES, ATTACK_PREFS);
                let (_, sc) = k.model_tweak(f);
                body_radius[f.index()][k.index()] = (mov_cpu[0].footprint * k.model_height() * sc) as f64;
                frames_mov[f.index()][k.index()] = mov_cpu.len();
                frames_atk[f.index()][k.index()] = atk_cpu.len();
                let mov = mov_cpu.iter().map(|m| renderer.upload_model(gl.quad_context, &m.vertices, &m.indices)).collect();
                let atk = atk_cpu.iter().map(|m| renderer.upload_model(gl.quad_context, &m.vertices, &m.indices)).collect();
                v.push(((f, k), UnitAnim { mov, atk }));
            }
        }
        v
    };
    // The knight is cavalry: the rider model is raised onto this (shared) horse,
    // which is bigger than the rider — so the knight's footprint is the horse's.
    let horse: Vec<ModelGpu> = {
        let gl = unsafe { get_internal_gl() };
        let cpu = load_glb_clip(include_bytes!("../../assets/siege/models/horse.glb"), ANIM_FRAMES, MOVE_PREFS);
        let hr = (cpu[0].footprint * Kind::Knight.model_height()) as f64;
        body_radius[0][Kind::Knight.index()] = hr;
        body_radius[1][Kind::Knight.index()] = hr;
        cpu.iter().map(|m| renderer.upload_model(gl.quad_context, &m.vertices, &m.indices)).collect()
    };
    // Castle model (static) for the two faction keeps, replacing the cube blocks.
    let castle = {
        let gl = unsafe { get_internal_gl() };
        let m = load_glb(include_bytes!("../../assets/siege/models/castle.glb"));
        renderer.upload_model(gl.quad_context, &m.vertices, &m.indices)
    };

    // Terrain. Voxel (blocky) by default; `V` toggles the smooth heightfield live
    // ($SIEGE_SMOOTH=1 starts smooth). The voxel mesh is *alterable*: cannon and
    // lava-bomb impacts carve craters into it (rebuilt, throttled, when dirty).
    let mut smooth = std::env::var("SIEGE_SMOOTH").is_ok();
    let mut craters: Vec<(f32, f32, f32)> = Vec::new(); // (x, z, radius), capped
    let mut terrain_dirty = false; // a crater landed → rebuild the voxel mesh soon
    let mut rebuild_t = 0.0f64; // throttle: at most one rebuild per REBUILD_EVERY
    const REBUILD_EVERY: f64 = 0.4;
    let mut terrain_chunks = if smooth { build_terrain_chunks() } else { build_voxel_chunks(&craters) };
    let mut now = 0.0f64; // simulation clock
    let mut paused = false;
    let mut volcano = Volcano::new(); // crater plume + eruption timers (shared sim)

    // Live thread-count control (native). The decide pass runs inside a rayon
    // pool sized by the slider; dragging the slider blocks camera-orbit.
    #[cfg(not(target_arch = "wasm32"))]
    let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    #[cfg(not(target_arch = "wasm32"))]
    let (mut n_threads, mut cur_threads) = (max_threads, max_threads);
    #[cfg(not(target_arch = "wasm32"))]
    let mut pool = rayon::ThreadPoolBuilder::new().num_threads(cur_threads).build().unwrap();
    // Mutated only by the (native-only) thread slider; immutable on wasm.
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut slider_drag = false;

    // Headless smoke hook: run N frames then exit (CI / startup-panic check).
    let max_frames: Option<u64> = std::env::var("SIEGE_MAX_FRAMES").ok().and_then(|s| s.parse().ok());
    let mut frame_no: u64 = 0;

    loop {
        // ----- input: orbit / zoom / controls -----
        let mp = mouse_position();
        if is_mouse_button_down(MouseButton::Left) && !slider_drag && !pop_drag {
            yaw += (mp.0 - last_mouse.0) * 0.01;
            pitch = (pitch + (mp.1 - last_mouse.1) * 0.01).clamp(0.05, 1.50);
        }
        last_mouse = mp;
        let wheel = mouse_wheel().1;
        if wheel != 0.0 { dist = (dist - wheel * 0.5).clamp(200.0, 1600.0); }
        if is_key_pressed(KeyCode::P) { paused = !paused; }
        if is_key_pressed(KeyCode::V) { // toggle smooth ↔ voxel terrain, rebuild now
            smooth = !smooth;
            terrain_chunks = if smooth { build_terrain_chunks() } else { build_voxel_chunks(&craters) };
        }
        // Rebuild the army when the population slider changed (or on `[`/`]`).
        if is_key_pressed(KeyCode::RightBracket) { per_faction = (per_faction + 100).min(2000); }
        if is_key_pressed(KeyCode::LeftBracket) { per_faction = per_faction.saturating_sub(100).max(20); }
        if per_faction != cur_pop {
            units = spawn_army(&mut rng, per_faction);
            cur_pop = per_faction;
        }

        let dt = (get_frame_time() as f64).min(0.05); // clamp huge hitches

        // ----- simulation step -----
        if !paused {
            now += dt;

            // Rebuild the index from this frame's live positions. The build is
            // serial and cheap; the queries (decide) are the parallel part.
            index.clear();
            for (i, u) in units.iter().enumerate() {
                if u.alive() { index.insert(IUnit { id: i as u32, faction: u.faction, p: u.p, health: (u.hp / u.kind.max_hp()) as f32, face: u.face }); }
            }
            // Rebuild the smoke index from last frame's live puffs.
            smoke_index.clear();
            for s in &smoke { smoke_index.insert(*s); }

            // Decide (read-only on both indices) then apply (serial resolution).
            // The decide pass fans out over the rayon pool (native) — each unit
            // mutates only itself while reading the shared indices. wasm: serial.
            #[cfg(not(target_arch = "wasm32"))]
            {
                if cur_threads != n_threads {
                    pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
                    cur_threads = n_threads;
                }
                let (idx, smk, br) = (&index, &smoke_index, &body_radius);
                pool.install(|| units.par_iter_mut().enumerate().for_each(|(i, u)| decide(u, i as u32, idx, smk, br)));
            }
            #[cfg(target_arch = "wasm32")]
            for i in 0..units.len() { decide(&mut units[i], i as u32, &index, &smoke_index, &body_radius); }
            let impacts = apply(&mut units, &mut smoke, &mut effects, &mut projectiles, &mut rng, dt, now);

            // Volcano: constant crater plume + the occasional eruption (lava
            // streaks, smoke burst, arcing lava bombs) — all in the shared sim.
            volcano_step(&mut volcano, &mut smoke, &mut effects, &mut projectiles, &mut rng, dt, now);

            // Alterable terrain: each ground impact carves a crater into the voxel
            // mesh. Cap the list and rebuild throttled (the voxel rebuild isn't free).
            for (ip, r) in impacts {
                craters.push((ip.x as f32, ip.z as f32, (r * 0.85) as f32));
                terrain_dirty = true;
            }
            if craters.len() > 64 { let drop = craters.len() - 64; craters.drain(0..drop); }
            rebuild_t -= dt;
            if terrain_dirty && !smooth && rebuild_t <= 0.0 {
                terrain_chunks = build_voxel_chunks(&craters);
                terrain_dirty = false;
                rebuild_t = REBUILD_EVERY;
            }
        }

        // ----- render 3D -----
        clear_background(Color::new(0.55, 0.68, 0.85, 1.0)); // sky
        let eye = observer + vec3(
            dist * pitch.cos() * yaw.cos(),
            dist * pitch.sin(),
            dist * pitch.cos() * yaw.sin(),
        );
        let cam = Camera3D { position: eye, up: vec3(0.0, 1.0, 0.0), target: observer, ..Default::default() };
        set_camera(&cam);
        let mvp = cam.matrix();

        for m in &terrain_chunks { draw_mesh(m); }
        draw_bridges(); // castles are drawn as instanced models in the gl block below
        draw_effects(&effects, now);

        // Units → one instanced draw per (faction, kind), using its glTF model.
        // Group live units into [faction][kind] buckets of model matrices
        // (place · face · scale, plus the procedural animation offset). Knights
        // are cavalry: a horse at the feet + the rider raised onto its back.
        // [faction][kind][frame] instance buckets — units split by which baked
        // animation frame they show this instant, so each frame instances together.
        // Separate movement and attack clips: a unit that's mid-attack shows the
        // attack clip, else the movement (or idle, for riders) clip. Buckets are
        // [faction][kind][frame] per clip.
        let mut mov_b: [[Vec<Vec<EffectInstance>>; 8]; 2] =
            std::array::from_fn(|fi| std::array::from_fn(|ki| vec![Vec::new(); frames_mov[fi][ki]]));
        let mut atk_b: [[Vec<Vec<EffectInstance>>; 8]; 2] =
            std::array::from_fn(|fi| std::array::from_fn(|ki| vec![Vec::new(); frames_atk[fi][ki]]));
        let mut horses: Vec<Vec<EffectInstance>> = vec![Vec::new(); horse.len()];
        let (mut red, mut blue) = (0usize, 0usize);
        for u in units.iter() {
            if !u.alive() { continue; }
            match u.faction { Faction::Red => red += 1, Faction::Blue => blue += 1 }
            let feet_y = (u.p.y - u.kind.radius() as f64) as f32; // drop sphere centre to ground
            let (yaw_off, sc) = u.kind.model_tweak(u.faction);
            let h = u.kind.model_height() * sc;
            let yaw = u.face + yaw_off;
            let base = vec3(u.p.x as f32, feet_y, u.p.z as f32) + anim_offset(u, now, h);
            let tint = faction_tint(u.faction);
            let (fi, ki) = (u.faction.index(), u.kind.index());
            let inst = if u.kind == Kind::Knight {
                let hh = h; // horse height (= knight model height)
                let horse_m = Mat4::from_translation(base) * Mat4::from_rotation_y(u.face) * Mat4::from_scale(Vec3::splat(hh));
                horses[anim_frame(u, now, horse.len())].push(EffectInstance::new(horse_m, tint));
                let rider = Mat4::from_translation(base + vec3(0.0, hh * 0.5, 0.0)) * Mat4::from_rotation_y(yaw) * Mat4::from_scale(Vec3::splat(hh * 0.72));
                EffectInstance::new(rider, tint)
            } else {
                let m = Mat4::from_translation(base) * Mat4::from_rotation_y(yaw) * Mat4::from_scale(Vec3::splat(h));
                EffectInstance::new(m, tint)
            };
            if u.atk_anim > 0.0 {
                atk_b[fi][ki][attack_frame(u.atk_anim, frames_atk[fi][ki])].push(inst);
            } else {
                mov_b[fi][ki][anim_frame(u, now, frames_mov[fi][ki])].push(inst);
            }
        }
        {
            let gl = unsafe { get_internal_gl() };
            let light = vec3(-0.45, 0.84, -0.30).normalize();
            for ((f, k), anim) in &models {
                for (fr, gpu) in anim.mov.iter().enumerate() { renderer.draw_models(gl.quad_context, gpu, &mov_b[f.index()][k.index()][fr], mvp, light); }
                for (fr, gpu) in anim.atk.iter().enumerate() { renderer.draw_models(gl.quad_context, gpu, &atk_b[f.index()][k.index()][fr], mvp, light); }
            }
            for (fr, gpu) in horse.iter().enumerate() {
                renderer.draw_models(gl.quad_context, gpu, &horses[fr], mvp, light); // cavalry mounts
            }
            // Castles — one model per faction keep, facing the map centre.
            let mut castle_inst = Vec::with_capacity(2);
            for f in [Faction::Red, Faction::Blue] {
                let (cx, cz) = f.castle();
                let yaw = (WORLD * 0.5 - cx).atan2(WORLD * 0.5 - cz) as f32;
                let m = Mat4::from_translation(vec3(cx as f32, terrain_height(cx, cz) as f32, cz as f32)) * Mat4::from_rotation_y(yaw) * Mat4::from_scale(Vec3::splat(62.0));
                castle_inst.push(EffectInstance::new(m, faction_tint(f)));
            }
            renderer.draw_models(gl.quad_context, &castle, &castle_inst, mvp, light);
        }
        // Projectiles: small spheres arcing through the air (cannonballs / lava).
        for pr in &projectiles {
            let (col, rad) = match pr.kind {
                ProjKind::Cannon => (Color::new(0.16, 0.15, 0.17, 1.0), 3.6),
                ProjKind::LavaRock => (Color::new(1.0, 0.45, 0.10, 1.0), 4.2),
            };
            draw_sphere(v3(pr.p), rad, None, col);
        }
        // Smoke: each cloud is a few translucent billows that rise, spread and
        // fade as they age — so it reads as smoke yet you still see the fight
        // through it. Deterministic offsets (from the spawn point) keep each puff
        // stable frame to frame.
        for s in &smoke {
            let age = ((now - s.born) / SMOKE_LIFE).clamp(0.0, 1.0) as f32;
            let base_r = SMOKE_R as f32 * (0.42 + 0.55 * age);
            let centre = vec3(s.p.x as f32, s.p.y as f32 + age * 22.0, s.p.z as f32); // rises
            let seed = (s.p.x * 0.13 + s.p.z * 0.71) as f32;
            for k in 0..2 {
                let a = seed + k as f32 * 2.39996; // ~golden-angle spread
                let off = vec3(a.sin(), (a * 1.7).sin() * 0.35 + 0.25, a.cos()) * base_r * 0.55;
                let rr = base_r * (0.5 + 0.22 * (a * 2.1).cos().abs());
                let alpha = (0.12 * (1.0 - age) + 0.02) * if k == 0 { 1.2 } else { 0.85 };
                draw_sphere(centre + off, rr, None, Color::new(0.80, 0.80, 0.85, alpha));
            }
        }

        // ----- HUD -----
        set_default_camera();
        draw_text("vectorial-hash — SIEGE", 16.0, 28.0, 30.0, WHITE);
        draw_text(format!("fps {}", get_fps()), 16.0, 54.0, 22.0, LIGHTGRAY);
        draw_text(format!("Red {red}"), 16.0, 80.0, 24.0, Color::new(0.95, 0.4, 0.35, 1.0));
        draw_text(format!("Blue {blue}"), 16.0, 104.0, 24.0, Color::new(0.45, 0.6, 1.0, 1.0));
        let (lead, lcol) = match red.cmp(&blue) {
            std::cmp::Ordering::Greater => ("Red leads", Color::new(0.95, 0.4, 0.35, 1.0)),
            std::cmp::Ordering::Less => ("Blue leads", Color::new(0.45, 0.6, 1.0, 1.0)),
            std::cmp::Ordering::Equal => ("even", LIGHTGRAY),
        };
        draw_text(lead, 16.0, 128.0, 22.0, lcol);
        // Live population slider (per faction) — the spatial-index stress lever.
        int_slider(20.0, 150.0, 220.0, "army/side", &mut per_faction, 20, 2000, &mut pop_drag);
        // Live thread-count slider drives the parallel AI pass (native only).
        #[cfg(not(target_arch = "wasm32"))]
        int_slider(20.0, 192.0, 220.0, "threads", &mut n_threads, 1, max_threads, &mut slider_drag);
        draw_text(
            "drag: orbit  scroll: zoom  P: pause  [ ]: \u{00b1}pop  V: voxel/smooth",
            16.0, screen_height() - 18.0, 20.0, LIGHTGRAY,
        );
        if paused { draw_text("PAUSED", screen_width() * 0.5 - 50.0, 40.0, 36.0, YELLOW); }

        next_frame().await;
        frame_no += 1;
        if let Some(m) = max_frames { if frame_no >= m { break; } }
    }
}
