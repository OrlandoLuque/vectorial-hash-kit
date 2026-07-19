//! Prints each model's normalised footprint (the loader scales every model to
//! height = 1, so `footprint × 2` is its width-to-height aspect). Used to size the
//! horde building models correctly — a wide, short wall at "scale = height" comes
//! out far too wide.
use vectorial_hash_demos::model::load_glb;

fn main() {
    let dir = "crates/vectorial-hash-demos/assets/siege/models";
    for name in ["wall", "gate", "tower", "house", "storehouse", "cannon", "castle", "zombie", "skeleton_a", "slime", "skeleton_sword", "bat"] {
        match std::fs::read(format!("{dir}/{name}.glb")) {
            Ok(bytes) => {
                let m = load_glb(&bytes);
                println!("{name:16} footprint(½ @ h=1) = {:.3}  → full width ≈ {:.2}× its height", m.footprint, m.footprint * 2.0);
            }
            Err(_) => println!("{name:16} (not found)"),
        }
    }
}
