//! How wide are the ring models ACTUALLY, in world units?
//!
//! `building_tweak` sets each model's world HEIGHT (the loader normalises every glb to height 1)
//! and the layout in `horde_sim::ring_footprint` sets how far apart to place them. Those two
//! numbers have to agree, and until now the bridge between them was a code comment claiming
//! "wall aspect 3.5, but the model bbox is ~1/2 decorative overhang". Nothing computed it, and
//! `model_dims` — the function that comment names — does not exist.
//!
//! The user's report is that the walls look piled on top of each other. If the models really are
//! ~2x as wide as the spacing, that is not an opinion about taste, it is arithmetic. So: load the
//! real assets, read the real bounding boxes, and print the world width each one occupies.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example model_footprints --release
//! ```
use vectorial_hash_demos::horde_sim::{ring_footprint, SKind};

/// Mirrors `horde_wgpu::building_tweak`'s scale column (the model's world height).
fn world_height(k: SKind) -> f32 {
    match k {
        SKind::Wall => 4.6, SKind::Gate => 9.0, SKind::Tower => 9.0,
        SKind::House => 6.5, SKind::Storehouse => 7.5, SKind::CommandCenter => 40.0,
    }
}

fn main() {
    let dir = std::path::Path::new("crates/vectorial-hash-demos/assets/siege/models");
    let files = [
        (SKind::Wall, "wall.glb"), (SKind::Gate, "gate.glb"), (SKind::Tower, "tower.glb"),
        (SKind::House, "house.glb"), (SKind::Storehouse, "storehouse.glb"),
    ];
    println!("ring model footprints — measured from the glb, not from a comment\n");
    println!("{:>12} {:>10} {:>12} {:>12} {:>12} {:>10}",
             "kind", "height", "norm.foot", "world width", "layout says", "ratio");

    for (kind, file) in files {
        let path = dir.join(file);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => { println!("{:>12}  cannot read {}: {e}", format!("{kind:?}"), path.display()); continue; }
        };
        let m = vectorial_hash_demos::model::load_glb(&bytes);
        let h = world_height(kind);
        // `footprint` is HALF the larger XZ extent at unit height, so the full width at this
        // model's world height is 2 * footprint * height.
        let world_w = 2.0 * m.footprint * h;
        let layout = ring_footprint(kind) as f32;
        println!("{:>12} {:>10.2} {:>12.3} {:>12.2} {:>12.2} {:>9.2}x",
                 format!("{kind:?}"), h, m.footprint, world_w, layout, world_w / layout);
    }

    // And what the layout does with those widths, on the real base.
    let h = vectorial_hash_demos::horde_sim::Horde::new(7, 2000);
    let (cx, cz) = (900.0f64, 900.0f64);
    let mut ring: Vec<(f64, f64)> = h.structures.iter()
        .filter(|s| matches!(s.kind, SKind::Wall | SKind::Gate | SKind::Tower))
        .map(|s| ((s.p.z - cz).atan2(s.p.x - cx).rem_euclid(std::f64::consts::TAU) * 150.0,
                  ring_footprint(s.kind) * 0.5))
        .collect();
    ring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let circ = std::f64::consts::TAU * 150.0;
    let (mut worst_gap, mut worst_ov) = (f64::MIN, f64::MIN);
    for i in 0..ring.len() {
        let (a, ha) = ring[i];
        let (b, hb) = ring[(i + 1) % ring.len()];
        let d = if i + 1 == ring.len() { b + circ - a } else { b - a };
        let ov = ha + hb - d;
        worst_gap = worst_gap.max(-ov);
        worst_ov = worst_ov.max(ov);
    }
    let n = |k: SKind| h.structures.iter().filter(|s| s.kind == k).count();
    println!("
the ring as laid out: {} walls · {} gates · {} towers over {:.0} wu of arc",
             n(SKind::Wall), n(SKind::Gate), n(SKind::Tower), circ);
    println!("worst overlap {:.2} wu · worst hole {:.2} wu", worst_ov, worst_gap.max(0.0));

    println!();
    println!("`world width` is what the renderer draws. `layout says` is the arc the sim reserves");
    println!("for that piece. A ratio near 1.0 means the pieces tile; a ratio of 2 means every");
    println!("piece is drawn twice as wide as its slot, so each one buries half of both of its");
    println!("neighbours -- which is what a rampart 'apelotonado' looks like, and no amount of");
    println!("re-deriving the SPACING can fix it while the widths disagree.");
}
