//! Which way does each unit model FACE, according to its own walk cycle?
//!
//! `ztweak` carries a per-class yaw correction that is zero for every zombie except the slime,
//! where it has now been guessed three times from a verbal report (-PI/2, then 0, then PI) and
//! reported wrong each time. Guessing a fourth is not a plan, so this asks the asset.
//!
//! **The signal.** In a walk cycle the legs swing fore-and-aft: vertices move much more along
//! the direction of travel than across it. So for each model, take the CPU-baked frames, and for
//! every vertex measure how much it moves along X and along Z across the cycle. The axis that
//! accumulates more motion is the model's walking axis. A model authored facing a different way
//! than its siblings will disagree with them, and that disagreement is the correction to apply.
//!
//! Two things this deliberately does NOT claim:
//!
//! - It finds the walking **axis**, not the **sign** — a model facing +X and one facing -X give
//!   the same answer. That is still useful: it separates "rotated 90 degrees" (an axis swap,
//!   which shows up here) from "rotated 180" (which does not).
//! - A blob has no legs. If the slime's walk is a squash-and-stretch in place, the anisotropy
//!   will be weak and the reading is inconclusive — which is itself worth knowing, because it
//!   would mean the constant cannot be recovered from the geometry and only an eye can settle it.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example model_facing --release
//! ```
use vectorial_hash_demos::siege_sim::{ANIM_FRAMES, MOVE_PREFS};

const UNITS: [(&str, &str); 6] = [
    ("Walker",  "zombie.glb"),
    ("Runner",  "skeleton_a.glb"),
    ("Chubby",  "slime.glb"),      // the one under suspicion
    ("Venom",   "skeleton_sword.glb"),
    ("Harpy",   "bat.glb"),
    ("Ranger",  "anne.glb"),       // a defender: never reported as wrong, so it is the control
];

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/siege/models");
    println!("walking axis by model — measured from the baked walk cycle\n");
    println!("{:>10} {:>12} {:>12} {:>10}   {}", "class", "motion X", "motion Z", "X/Z", "reading");

    for (name, file) in UNITS {
        let bytes = match std::fs::read(dir.join(file)) {
            Ok(b) => b,
            Err(e) => { println!("{name:>10}  cannot read {file}: {e}"); continue; }
        };
        let frames = vectorial_hash_demos::model::load_glb_clip(&bytes, ANIM_FRAMES, MOVE_PREFS);
        if frames.len() < 2 { println!("{name:>10}  only {} frame(s) — no cycle to read", frames.len()); continue; }

        // Per vertex, the spread of its position across the cycle, summed over vertices. The
        // mean is subtracted per vertex so a model that simply sits off-centre reads zero.
        let n = frames[0].vertices.len();
        let (mut mx, mut mz) = (0.0f64, 0.0f64);
        for v in 0..n {
            let (mut sx, mut sz) = (0.0f64, 0.0f64);
            for f in &frames { sx += f.vertices[v].pos[0] as f64; sz += f.vertices[v].pos[2] as f64; }
            let (ax, az) = (sx / frames.len() as f64, sz / frames.len() as f64);
            for f in &frames {
                mx += (f.vertices[v].pos[0] as f64 - ax).abs();
                mz += (f.vertices[v].pos[2] as f64 - az).abs();
            }
        }
        let (mx, mz) = (mx / n as f64, mz / n as f64);
        let ratio = if mz > 1e-9 { mx / mz } else { f64::INFINITY };
        // Well clear of 1.0 either way is a real axis; near 1.0 means the motion is isotropic
        // and this method has nothing to say.
        let reading = if ratio > 1.35 { "walks along X" }
            else if ratio < 0.74 { "walks along Z" }
            else { "INCONCLUSIVE (motion is isotropic — no legs to read)" };
        println!("{name:>10} {mx:>12.4} {mz:>12.4} {ratio:>10.2}   {reading}");
    }

    println!();
    println!("Compare the zombie classes against each other and against Ranger, which is a");
    println!("defender and has never been reported as facing wrong. A class that walks along a");
    println!("different axis than the rest is rotated 90 degrees in the asset, and that is what");
    println!("`ztweak`'s per-class yaw exists to undo. If the slime reads INCONCLUSIVE, the");
    println!("geometry cannot answer it and the live knob (Y/U in horde_wgpu) is the only way.");
}
