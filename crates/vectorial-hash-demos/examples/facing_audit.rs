//! Do the zombies FACE the way they are walking? — the measurement, after two wrong guesses.
//!
//! The user reported (2026-08-04) that among moving zombies "unos avanzan mirando de frente,
//! otros mirando a un lado, y otros van de espaldas". Two hypotheses were tried by reading code
//! and both were wrong in an instructive way:
//!
//! 1. **The impostor billboard was a quarter-turn off.** True — the far-LOD path passed
//!    `-yaw` where the skinned path uses `-yaw + PI/2` — and fixing it changed nothing the user
//!    could see. Refuted by a control the code could not have given: rendering the same scene
//!    with `$HORDE_NOLOD=1`, i.e. every unit as a real skinned model with no billboard anywhere,
//!    and finding the symptom still there.
//! 2. **Reading it off a screenshot.** Useless here, and worth saying why: inside the base the
//!    units converge on the Command Center, so units a few metres apart genuinely have different
//!    headings. A picture of a crowd cannot distinguish "wrong" from "varied".
//!
//! So this asks the only question that separates them, in numbers: for each unit that actually
//! MOVED this frame, does `face` match the direction it moved in? The renderer turns the model
//! by `face` and nothing else, so any disagreement here is visible as a unit walking sideways.
//!
//! ```bash
//! cargo run -p vectorial-hash-demos --example facing_audit --release
//! ```
use vectorial_hash_demos::horde_sim::Horde;

fn main() {
    let mut h = Horde::new(7, 20_000);
    let woke = h.wake_all();
    println!("facing audit — {woke} zombies woken, then marched\n");
    println!("{:>7} {:>9} {:>9} {:>10} {:>10} {:>10} {:>9}",
             "frame", "moved", "median°", "p90°", ">15° off", ">90° off", "worst°");

    for f in 0..=240 {
        // snapshot before the step so the displacement is the real one
        let before: Vec<(f64, f64, f32)> = h.units.iter().map(|z| (z.p.x, z.p.z, z.face)).collect();
        h.step(1.0 / 60.0);

        if f % 60 != 0 || f == 0 { continue; }
        let mut errs: Vec<f64> = Vec::new();
        for (z, (px, pz, _)) in h.units.iter().zip(before.iter()) {
            let (dx, dz) = (z.p.x - px, z.p.z - pz);
            let d = (dx * dx + dz * dz).sqrt();
            if d < 1e-3 { continue; }                       // did not move: nothing to check
            let travelled = dz.atan2(dx);
            // smallest signed angle between the heading it FACES and the one it WALKED
            let mut e = (z.face as f64 - travelled).abs() % std::f64::consts::TAU;
            if e > std::f64::consts::PI { e = std::f64::consts::TAU - e; }
            errs.push(e.to_degrees());
        }
        if errs.is_empty() { println!("{f:>7} {:>9} (nobody moved)", 0); continue; }
        errs.sort_by(f64::total_cmp);
        let med = errs[errs.len() / 2];
        let p90 = errs[errs.len() * 9 / 10];
        let bad15 = errs.iter().filter(|&&e| e > 15.0).count();
        let bad90 = errs.iter().filter(|&&e| e > 90.0).count();
        println!("{f:>7} {:>9} {:>9.1} {:>10.1} {:>9.1}% {:>9.1}% {:>9.1}",
                 errs.len(), med, p90,
                 100.0 * bad15 as f64 / errs.len() as f64,
                 100.0 * bad90 as f64 / errs.len() as f64,
                 errs[errs.len() - 1]);
    }

    println!();
    println!("`face` is set from the position delta whenever a unit actually moves, so a unit");
    println!("that has moved THIS frame should read ~0 degrees off. A large median means the");
    println!("renderer's model rotation disagrees with the sim; a small median with a fat tail");
    println!("means a subset is stale -- most likely units that were shoved by a neighbour");
    println!("rather than moving under their own steam, since only their own step writes face.");
}
