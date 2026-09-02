//! Provenance for numbers that are only true on the machine that produced them.
//!
//! Two files in this repo hold measured values: `benches/baseline.tsv` (the regression gate's
//! reference times) and `calibrations/*.txt` (the adaptive index's thresholds). Both are
//! hardware-specific and **neither recorded which hardware**. The gate's own docs say to treat
//! cross-machine numbers as orientation only and CI marks its run informational for exactly that
//! reason — but the knowledge lived in prose, so running the gate locally on a different machine
//! produced a table of confident verdicts about nothing.
//!
//! Clock normalisation does not rescue it. `_calib` is a fixed CPU loop, so it cancels *clock
//! speed*; cache sizes, memory bandwidth and core counts survive the division untouched, and
//! those are most of what separates two machines on a spatial-index workload.
//!
//! The fingerprint is deliberately coarse and dependency-free: OS, architecture, logical core
//! count, host name. It cannot tell two identical laptops apart and does not try to — the job is
//! to catch "this baseline came from the desktop and you are on the laptop", which is the mistake
//! that actually happens.

/// A coarse identity for the machine taking a measurement, e.g. `windows/x86_64 8c LANDER-PC`.
///
/// Uses only `std`. The core count comes from [`std::thread::available_parallelism`], which
/// respects container CPU limits, so a CI runner reports the cores it may actually use rather
/// than the cores the host owns.
pub fn machine_id() -> String {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    format!("{}/{} {}c {}", std::env::consts::OS, std::env::consts::ARCH, cores, host())
}

/// Best-effort host name. `HOSTNAME` is frequently not exported to child processes on Unix, so
/// `/etc/hostname` is the fallback before giving up — an unknown host degrades the fingerprint
/// to OS/arch/cores, which still separates an 8-core laptop from a 24-core desktop.
fn host() -> String {
    if let Ok(h) = std::env::var("COMPUTERNAME") { if !h.is_empty() { return h; } }
    if let Ok(h) = std::env::var("HOSTNAME") { if !h.is_empty() { return h; } }
    match std::fs::read_to_string("/etc/hostname") {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => "unknown-host".to_string(),
    }
}

/// The fingerprint reduced to something safe to put in a filename, e.g.
/// `windows-x86_64-8c-lander-pc`.
///
/// This is what lets one repo hold a baseline per machine instead of one baseline that is wrong
/// everywhere but home. Lowercased and reduced to `[a-z0-9-]` so it behaves the same on a
/// case-insensitive filesystem as on a case-sensitive one.
pub fn machine_slug() -> String {
    let mut out = String::new();
    for c in machine_id().chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() { out.push(c); }
        else if !out.ends_with('-') { out.push('-'); }
    }
    out.trim_matches('-').to_string()
}

/// The line to write into a file of measured values. Accepted back by [`machine_of`].
pub fn machine_line() -> String { format!("# machine = {}\n", machine_id()) }

/// Read the fingerprint out of a file's text, if it has one.
///
/// Accepts the line with or without a leading `#`, so the same helper serves the TSV baseline
/// (where it must be a comment) and the calibration format (whose parser strips `#` and would
/// silently drop it either way). Files written before this existed return `None`, and a caller
/// must treat that as "unknown", never as "matches".
pub fn machine_of(text: &str) -> Option<String> {
    text.lines().find_map(|l| {
        let l = l.trim().trim_start_matches('#').trim();
        l.strip_prefix("machine")?.trim_start().strip_prefix('=').map(|v| v.trim().to_string())
    })
}

/// Compare a file's fingerprint against this machine. `None` means the file predates the
/// fingerprint, which is a distinct answer from "different" and is reported as such.
pub fn verdict(text: &str) -> Provenance {
    match machine_of(text) {
        None => Provenance::Unknown,
        Some(m) if m == machine_id() => Provenance::SameMachine,
        Some(m) => Provenance::OtherMachine(m),
    }
}

/// What a file of measured numbers is worth on the machine reading it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provenance {
    /// Captured here. Comparisons are meaningful.
    SameMachine,
    /// Captured elsewhere. Times are orientation only; a pass/fail verdict is not available.
    OtherMachine(String),
    /// No fingerprint recorded — an older file. Treat as unverified, not as a match.
    Unknown,
}

impl Provenance {
    /// Whether a pass/fail judgement on absolute times may be made from this file.
    pub fn may_judge(&self) -> bool { matches!(self, Provenance::SameMachine) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_survives_being_a_comment() {
        let line = machine_line();
        assert_eq!(machine_of(&line).as_deref(), Some(machine_id().as_str()));
        // The calibration format strips `#`; the TSV keeps it. One helper, both files.
        assert_eq!(machine_of(&line.replace("# ", "")).as_deref(), Some(machine_id().as_str()));
        // Embedded in a real file, among lines it must not mistake for the fingerprint.
        let file = format!("# a header\nbrute_max = 64\n{line}machine_learning = 3\ncull_tree3\t120\n");
        assert_eq!(machine_of(&file).as_deref(), Some(machine_id().as_str()));
    }

    #[test]
    fn a_missing_fingerprint_is_unknown_not_a_match() {
        // The distinction that matters: every file written before this module existed has no
        // fingerprint, and silently treating that as "same machine" would leave the whole
        // problem in place for exactly the files most likely to be stale.
        assert_eq!(verdict("cull_tree3\t120\n"), Provenance::Unknown);
        assert!(!Provenance::Unknown.may_judge());
        assert!(Provenance::SameMachine.may_judge());
        assert!(!Provenance::OtherMachine("other".into()).may_judge());
    }

    #[test]
    fn a_different_machine_is_named_not_just_flagged() {
        let foreign = "# machine = linux/aarch64 96c some-server\n";
        match verdict(foreign) {
            Provenance::OtherMachine(m) => assert!(m.contains("some-server"), "got {m}"),
            other => panic!("expected OtherMachine, got {other:?}"),
        }
        // Non-vacuity: this machine must not coincidentally BE that one.
        assert_ne!(machine_id(), "linux/aarch64 96c some-server");
    }

    #[test]
    fn the_slug_is_a_safe_filename_and_still_distinguishes() {
        let slug = machine_slug();
        assert!(slug.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "unsafe characters in {slug}");
        assert!(!slug.starts_with('-') && !slug.ends_with('-'), "{slug}");
        assert!(!slug.contains("--"), "{slug}");
        // It must still carry the discriminating parts, or per-machine files would collide.
        // Compare against the SANITISED arch: `x86_64` becomes `x86-64` on the way through, and
        // asserting the raw form is how this test first failed — correctly.
        let arch: String = std::env::consts::ARCH.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
        assert!(slug.contains(std::env::consts::OS) && slug.contains(&arch), "{slug} lacks {arch}");
    }

    #[test]
    fn the_fingerprint_is_not_degenerate() {
        // A fingerprint that is constant across machines would pass every test above while
        // protecting nobody. Assert it carries the three things it promises.
        let id = machine_id();
        assert!(id.contains(std::env::consts::OS) && id.contains(std::env::consts::ARCH), "{id}");
        assert!(id.contains('c'), "core count missing from {id}");
        assert!(id.len() > 8, "suspiciously short fingerprint: {id}");
    }
}
