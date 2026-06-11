//! Snapshot test: regenerate the deterministic template fingerprint and
//! compare it byte-for-byte against the versioned fixture. Any unintended
//! change in the template-generation pipeline (geometry predicates,
//! intersector tolerances, encoding) makes this test fail and points at the
//! exact templates that differ.
//!
//! Updating the fixture is an explicit step:
//!
//! ```bash
//! cargo run -p vectorial-hash-cli --release -- templates-fingerprint \
//!   > crates/vectorial-hash-templates/tests/fixtures/template_fingerprint.txt
//! ```
//!
//! The diff will then show in the commit, so every change to the
//! precomputed template bytes is reviewed on purpose.

use vectorial_hash_templates::fingerprint;

const FIXTURE: &str =
    include_str!("fixtures/template_fingerprint.txt");

#[test]
fn template_fingerprint_matches_fixture() {
    let actual = fingerprint::generate();
    if actual == FIXTURE {
        return;
    }
    // Build a focused report on the first few differing lines so the test
    // output points at the problem without dumping ~14k lines.
    let mut report = String::from(
        "template fingerprint diverged from \
         tests/fixtures/template_fingerprint.txt\n\n",
    );
    let actual_lines: Vec<&str> = actual.lines().collect();
    let fixture_lines: Vec<&str> = FIXTURE.lines().collect();
    let mut differing = 0usize;
    let limit = 8;
    for (i, (a, f)) in actual_lines.iter().zip(fixture_lines.iter()).enumerate() {
        if a != f {
            differing += 1;
            if differing <= limit {
                report.push_str(&format!(
                    "line {}:\n  fixture: {}\n  actual : {}\n",
                    i + 1,
                    f,
                    a,
                ));
            }
        }
    }
    if actual_lines.len() != fixture_lines.len() {
        report.push_str(&format!(
            "\nline counts disagree: fixture={} actual={}\n",
            fixture_lines.len(),
            actual_lines.len(),
        ));
    }
    let extra = differing.saturating_sub(limit);
    if extra > 0 {
        report.push_str(&format!("\n... and {extra} more differing lines.\n"));
    }
    report.push_str(
        "\nIf the change is intentional, regenerate the fixture with:\n  \
         cargo run -p vectorial-hash-cli --release -- templates-fingerprint \\\n    \
         > crates/vectorial-hash-templates/tests/fixtures/template_fingerprint.txt\n",
    );
    panic!("{report}");
}
