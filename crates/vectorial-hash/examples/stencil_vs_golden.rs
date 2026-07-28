//! Cross-check `SphereStencil` against the reference sets generated for the Minecraft mod
//! team by a completely separate path (Python, world-space brute force, two independent
//! computations that had to agree before the file was written).
//!
//! A reference that shares the implementation's assumptions validates nothing; this one
//! does not share them, which is the only reason it is worth running.
use vectorial_hash::{Alignment, SphereStencil};

fn main() {
    // (alignment, radius, expected full, expected partial) — from _dev/sphere_golden/INDEX.txt
    let cases: &[(&str, Alignment, f64, usize, usize)] = &[
        ("center", Alignment::CellCentre, 2.0, 7, 74),
        ("center", Alignment::CellCentre, 4.0, 147, 314),
        ("center", Alignment::CellCentre, 6.0, 611, 674),
        ("center", Alignment::CellCentre, 8.0, 1599, 1250),
        ("center", Alignment::CellCentre, 12.0, 5935, 2690),
        ("center", Alignment::CellCentre, 16.0, 14915, 4874),
        ("center", Alignment::CellCentre, 24.0, 52587, 10826),
        ("center", Alignment::CellCentre, 32.0, 127883, 19370),
        ("corner", Alignment::CellCorner, 2.0, 8, 80),
        ("corner", Alignment::CellCorner, 4.0, 136, 296),
        ("corner", Alignment::CellCorner, 6.0, 624, 680),
        ("corner", Alignment::CellCorner, 8.0, 1568, 1184),
        ("corner", Alignment::CellCorner, 12.0, 5904, 2648),
        ("corner", Alignment::CellCorner, 16.0, 14784, 4784),
        ("corner", Alignment::CellCorner, 24.0, 52544, 10760),
        ("corner", Alignment::CellCorner, 32.0, 127632, 19256),
        ("free", Alignment::Phase(0.3125, 0.75, 0.5625), 2.0, 8, 80),
        ("free", Alignment::Phase(0.3125, 0.75, 0.5625), 4.0, 140, 305),
        ("free", Alignment::Phase(0.3125, 0.75, 0.5625), 6.0, 609, 689),
        ("free", Alignment::Phase(0.3125, 0.75, 0.5625), 8.0, 1592, 1208),
        ("free", Alignment::Phase(0.3125, 0.75, 0.5625), 12.0, 5950, 2718),
        ("free", Alignment::Phase(0.3125, 0.75, 0.5625), 16.0, 14860, 4828),
        ("free", Alignment::Phase(0.3125, 0.75, 0.5625), 24.0, 52647, 10872),
        ("free", Alignment::Phase(0.3125, 0.75, 0.5625), 32.0, 127833, 19321),
    ];
    let (mut ok, mut bad) = (0, 0);
    for (name, align, r, want_full, want_part) in cases {
        let s = SphereStencil::new(*r, *align);
        let (f, p) = (s.full_count(), s.partial_count());
        if f == *want_full && p == *want_part {
            ok += 1;
        } else {
            bad += 1;
            println!("  MISMATCH {name} r={r}: full {f} (want {want_full}), partial {p} (want {want_part})");
        }
    }
    println!("stencil vs golden: {ok} agree, {bad} differ (reference generated independently, in another language)");
    println!("#M stencil_golden.agree {ok} n");
    println!("#M stencil_golden.differ {bad} n");
    if bad > 0 { std::process::exit(1); }
}
