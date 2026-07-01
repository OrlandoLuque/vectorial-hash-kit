//! Snapshot test for the deterministic template fingerprint.
//!
//! The fingerprint is one line per `(figure, angle, cell, offset)` configuration:
//! a platform-independent label followed by the encoded template *bytes*. The
//! labels and the row *set* are fully portable; the bytes, however, come from
//! floating-point geometry (`sin`/`cos` in the rotation, `asin`/`acos` in the
//! intersector), and those transcendental functions are **not** bit-identical
//! across platforms/libm versions — a handful of cells at horizontal/vertical
//! tangent configurations can flip. So a byte-for-byte fixture generated on one
//! machine cannot be a hard gate on a different CI runner without false
//! failures. This test therefore checks, in layers:
//!
//! 1. **Determinism** (everywhere, hard): `generate()` twice must be identical —
//!    guards against any unordered iteration (HashMap) creeping into the path.
//! 2. **Structure** (everywhere, hard): the row *count* must equal the fixture's
//!    — a real pipeline regression changes which configurations exist / the
//!    encoding length, independent of libm.
//! 3. **Bytes**: exact match when `VH_FINGERPRINT_STRICT=1` (the reference
//!    platform / fixture-regen workflow — this is where *subtle* byte
//!    regressions are caught, since there libm noise is zero). Otherwise only a
//!    *wholesale* break fails (differing rows over [`GROSS_FRACTION`]): a subtle
//!    real regression and cross-platform libm noise are the same tiny magnitude
//!    (the 2026-06 tangent fix touched 0.6 % of cells), so a non-reference
//!    runner can't tell them apart — it defers subtle byte checks to the
//!    reference platform and only guards against a catastrophic break here.
//!
//! Regenerate the fixture (on the reference platform) with:
//!
//! ```bash
//! cargo run -p vectorial-hash-cli --release -- templates-fingerprint \
//!   > crates/vectorial-hash-templates/tests/fixtures/template_fingerprint.txt
//! ```

use vectorial_hash_templates::fingerprint;

const FIXTURE: &str = include_str!("fixtures/template_fingerprint.txt");

/// Fraction of differing rows that counts as a *wholesale* break on a
/// non-reference platform (a broken encoder / geometry predicate flips ~all
/// rows). Set well above any plausible cross-platform libm noise (which is a
/// fraction of a percent) and well above a subtle real regression (also tiny —
/// those are caught exactly on the reference platform via VH_FINGERPRINT_STRICT).
const GROSS_FRACTION: f64 = 0.25;

#[test]
fn template_fingerprint_matches_fixture() {
    let actual = fingerprint::generate();

    // (1) Determinism — platform-independent, always a hard gate.
    assert_eq!(actual, fingerprint::generate(), "template fingerprint generation is non-deterministic");

    let actual_lines: Vec<&str> = actual.lines().collect();
    let fixture_lines: Vec<&str> = FIXTURE.lines().collect();

    // (2) Structure — the row set is portable; a count mismatch is a real change.
    assert_eq!(
        actual_lines.len(),
        fixture_lines.len(),
        "template fingerprint row count changed (fixture={}, actual={}) — a real pipeline change, not libm noise; regenerate the fixture if intentional",
        fixture_lines.len(),
        actual_lines.len(),
    );

    // Rows are emitted in a fixed order, so row i lines up with fixture row i.
    // Also assert the platform-independent *label* (everything up to " : ")
    // matches — that must never differ; only the trailing bytes may.
    let mut differing = 0usize;
    let mut report = String::new();
    let limit = 8;
    for (i, (a, f)) in actual_lines.iter().zip(fixture_lines.iter()).enumerate() {
        if a != f {
            let a_label = a.split(" : ").next().unwrap_or(a);
            let f_label = f.split(" : ").next().unwrap_or(f);
            assert_eq!(a_label, f_label, "fingerprint row {} label diverged (structure, not libm): fixture={:?} actual={:?}", i + 1, f_label, a_label);
            differing += 1;
            if differing <= limit {
                report.push_str(&format!("row {}:\n  fixture: {}\n  actual : {}\n", i + 1, f, a));
            }
        }
    }

    if differing == 0 {
        return;
    }

    let frac = differing as f64 / fixture_lines.len().max(1) as f64;
    let strict = std::env::var("VH_FINGERPRINT_STRICT").is_ok();
    let extra = differing.saturating_sub(limit);
    if extra > 0 {
        report.push_str(&format!("... and {extra} more differing rows.\n"));
    }

    if strict || frac > GROSS_FRACTION {
        let mode = if strict { "strict (VH_FINGERPRINT_STRICT set)".to_string() } else { format!("{:.1}% of rows differ — a wholesale break (over the {:.0}% threshold)", frac * 100.0, GROSS_FRACTION * 100.0) };
        panic!(
            "template fingerprint diverged from tests/fixtures/template_fingerprint.txt [{mode}]\n\n{report}\n\
             If intentional, regenerate the fixture on the reference platform with:\n  \
             cargo run -p vectorial-hash-cli --release -- templates-fingerprint \\\n    \
             > crates/vectorial-hash-templates/tests/fixtures/template_fingerprint.txt"
        );
    }

    // A small divergence on a non-reference platform is expected libm noise (or
    // a subtle change that must be reviewed exactly on the reference platform):
    // pass, but make the drift visible.
    eprintln!(
        "note: {differing} of {} fingerprint rows differ from the fixture ({:.3}% — under the {:.0}% wholesale-break threshold; run VH_FINGERPRINT_STRICT=1 on the reference platform for an exact check)",
        fixture_lines.len(),
        frac * 100.0,
        GROSS_FRACTION * 100.0,
    );
}
