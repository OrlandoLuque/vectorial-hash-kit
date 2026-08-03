//! Shared bench harness: **processor cycles**, overhead-corrected, cache-explicit.
//!
//! Four problems, in the order they bite:
//!
//! **1. Wall time answers the wrong question.** It says how long you waited, so a browser
//! stealing a core reports a regression that does not exist. A processor clock only ticks
//! while this process runs.
//!
//! **2. The obvious processor clock is too coarse.** `GetProcessTimes` / `getrusage`
//! report in system timer ticks — **~15.6 ms on Windows** — so a 0.3 ms cull measures as
//! *zero*. The classic fix is to repeat the operation until the interval is hundreds of
//! ticks and divide, and it works, but on a 15.6 ms tick "hundreds of ticks" is seconds
//! **per sample**: measured here, it turned one bench into a >10-minute run. So we use a
//! fine clock instead and keep auto-repetition for the genuinely tiny operations.
//!
//! **3. Cycles, not seconds.** `QueryProcessCycleTime` (Windows) counts **CPU cycles
//! attributed to this process** at cycle resolution; Linux's `CLOCK_PROCESS_CPUTIME_ID`
//! is nanosecond-resolution CPU time. Cycles are also *frequency-invariant*, so turbo and
//! thermal drift do not move them — which is one fewer thing that can fake a regression.
//! Converting cycles to milliseconds needs a rate, and **that calibration is the fragile
//! part**: measured over a fixed wall interval on a 3x-oversubscribed machine, the process
//! gets a fraction of the interval, the rate reads low, and everything inflates by the
//! reciprocal — it turned a 0.37 ms cull into 5.5 ms. Calibrating as the **best** of
//! several short trials fixes it (at least one slice runs on a full core), and ratios
//! between structures never touch the rate at all.
//!
//! **4. The measurement is not free.** Reading the clock costs cycles, and so does the
//! repetition loop. Both are measured once against an empty closure and subtracted, which
//! matters exactly when the operation is small — the case where a benchmark is most
//! likely to be reporting its own harness.
//!
//! **Cache is a choice, not an accident.** Repeating an operation leaves its data hot in
//! L1/L2, so `measure` reports the **warm** cost — the right question for something a
//! frame does thousands of times. `measure_cold` evicts the caches between reps and
//! reports the **first-touch** cost — the right question for something a frame does once,
//! and typically several times larger. Quoting one for the other is a real error, so they
//! are separate functions with separate names rather than a flag.
//!
//! What none of this removes: another process still evicts your cache lines and eats
//! memory bandwidth, so under load the same work genuinely takes more cycles (measured:
//! ~1.7x under 3x oversubscription). Publish from an idle run — `bench-runner` gates on
//! one — and use this to make a run taken *while you keep working* worth looking at.
//!
//! Included via `#[path = "common/mod.rs"] mod common;` — a subdirectory, so cargo does
//! not build it as an example of its own. No dependencies: the OS calls are declared here.
#![allow(dead_code)]

use std::sync::OnceLock;

// ------------------------------------------------------------------ the clock

#[cfg(windows)]
mod imp {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> std::os::windows::raw::HANDLE;
        fn QueryProcessCycleTime(h: std::os::windows::raw::HANDLE, cycles: *mut u64) -> i32;
    }
    /// CPU cycles attributed to this process. Does not advance while descheduled.
    pub fn cycles() -> u64 {
        let mut c = 0u64;
        if unsafe { QueryProcessCycleTime(GetCurrentProcess(), &mut c) } == 0 { 0 } else { c }
    }
    pub const NATIVE_CYCLES: bool = true;
}

#[cfg(unix)]
mod imp {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct TimeSpec { sec: i64, nsec: i64 }
    unsafe extern "C" { fn clock_gettime(clk: i32, tp: *mut TimeSpec) -> i32; }
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;
    /// Nanoseconds of process CPU time, reported as "cycles" of a 1 GHz virtual clock so
    /// the units above stay uniform; `rate()` then converts exactly.
    pub fn cycles() -> u64 {
        let mut t = TimeSpec::default();
        if unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut t) } != 0 { return 0; }
        (t.sec as u64).wrapping_mul(1_000_000_000).wrapping_add(t.nsec as u64)
    }
    pub const NATIVE_CYCLES: bool = false;
}

/// Process CPU cycles (Windows) or CPU nanoseconds (Unix) — see [`rate`].
#[inline]
pub fn cycles() -> u64 { imp::cycles() }

/// Ticks of [`cycles`] per second, best-of-seven so a busy machine cannot deflate it.
/// Only ever used to *display* milliseconds; every ratio in a bench is cycles over cycles
/// and never touches it.
pub fn rate() -> f64 {
    static RATE: OnceLock<f64> = OnceLock::new();
    *RATE.get_or_init(|| {
        if !imp::NATIVE_CYCLES { return 1e9; } // unix: the "cycles" above are nanoseconds
        let mut best = 0.0f64;
        for _ in 0..7 {
            let (c0, t0) = (cycles(), std::time::Instant::now());
            let mut x = 0x9E37_79B9_7F4A_7C15u64;
            while t0.elapsed().as_millis() < 15 {
                for i in 0..100_000u64 { x = x.wrapping_mul(6364136223846793005).wrapping_add(i | 1); x ^= x >> 29; }
            }
            std::hint::black_box(x);
            let (dc, dt) = ((cycles() - c0) as f64, t0.elapsed().as_secs_f64());
            if dc > 0.0 && dt > 0.0 { best = best.max(dc / dt); }
        }
        if best > 0.0 { best } else { 1e9 }
    })
}

/// Cost of the harness itself: two clock reads plus one loop iteration, in cycles.
/// Subtracted from every measurement, which only matters when the operation is small —
/// precisely when a benchmark is most at risk of reporting its own timing code.
pub fn overhead_cycles() -> f64 {
    static OH: OnceLock<u64> = OnceLock::new();
    *OH.get_or_init(|| {
        let reps = 1000usize;
        let mut lo = u64::MAX;
        for _ in 0..20 {
            let c = cycles();
            for _ in 0..reps { std::hint::black_box(0u64); }
            lo = lo.min((cycles() - c) / reps as u64);
        }
        lo
    }) as f64
}

// ------------------------------------------------------------------ measuring

/// One measured operation.
pub struct Sample {
    /// Cycles per call, harness overhead removed. The comparable, frequency-invariant number.
    pub cycles: f64,
    /// The same, converted for display. Depends on [`rate`]; ratios should use `cycles`.
    pub ms: f64,
    /// How many calls one timed interval had to contain for the clock to be precise.
    pub reps: usize,
}

fn run_samples<F: FnMut()>(samples: usize, reps: usize, f: &mut F) -> f64 {
    let mut lo = u64::MAX;
    for _ in 0..samples {
        let c = cycles();
        for _ in 0..reps { f(); }
        lo = lo.min(cycles() - c);
    }
    lo as f64
}

/// **Warm** cost of `f`: repeated back to back, so its data stays in cache. This is the
/// right question for an operation a frame performs many times.
///
/// Auto-scales the repetition count until one timed interval spans enough clock ticks to
/// be precise (the trick that makes a coarse clock usable, and still worth doing on a
/// fine one when the operation is sub-microsecond), then subtracts the harness overhead.
pub fn measure<F: FnMut()>(samples: usize, mut f: F) -> Sample {
    let target = 200_000.0; // cycles per timed interval: ~50 us at 4 GHz
    let mut reps = 1usize;
    loop {
        let d = run_samples(1, reps, &mut f);
        if d >= target || reps >= 1 << 20 { break; }
        let factor = if d > 0.0 { (target / d).clamp(2.0, 16.0) } else { 16.0 };
        reps = ((reps as f64 * factor).ceil() as usize).max(reps + 1);
    }
    let total = run_samples(samples, reps, &mut f);
    let per = (total / reps as f64 - overhead_cycles()).max(0.0);
    Sample { cycles: per, ms: per / rate() * 1e3, reps }
}

/// **Cold** cost of `f`: the caches are evicted before every call, so this is the
/// first-touch number — the right question for an operation a frame performs once. It is
/// normally several times the warm figure, and quoting one for the other is a real error.
///
/// Eviction is a streaming write over a buffer larger than a typical L3 (32 MB). That is
/// itself work, so it sits *outside* the timed interval; each call is timed on its own,
/// which also means this cannot use auto-repetition and is noisier by nature.
pub fn measure_cold<F: FnMut()>(samples: usize, mut f: F) -> Sample {
    let mut flush = vec![0u8; 32 << 20];
    let mut lo = u64::MAX;
    for _ in 0..samples.max(3) {
        for (i, b) in flush.iter_mut().enumerate() { *b = i as u8; }
        std::hint::black_box(&flush);
        let c = cycles();
        f();
        lo = lo.min(cycles() - c);
    }
    let per = (lo as f64 - overhead_cycles()).max(0.0);
    Sample { cycles: per, ms: per / rate() * 1e3, reps: 1 }
}

/// Two alternatives, measured **interleaved in time**, reported as a ratio.
///
/// This exists because measuring A fully and then B fully is not a fair comparison at this
/// scale: the same cull measured at three different points of one process reported 1.89x
/// or 2.45x against the same rival, purely from cache state, run order and clock drift.
/// Whichever runs second inherits a machine the first one warmed (or dirtied).
///
/// The fix is to pair them in time and aggregate the RATIOS, not the times. Each round runs
/// `A B B A` — the palindrome cancels any first-order drift within the round, since A's two
/// samples straddle B's — and the round yields one ratio. The reported figure is the
/// **median of the per-round ratios**, which is robust to a slow drift that would move both
/// absolutes together, and the spread tells you whether the ratio itself was stable or you
/// are quoting noise.
///
/// Returns `(a_cycles, b_cycles, ratio_b_over_a, ratio_spread_pct)`.
pub fn compare2<A: FnMut(), B: FnMut()>(rounds: usize, mut a: A, mut b: B) -> (f64, f64, f64, f64) {
    // One warm-up of each, discarded: the first touch of either data set is a different
    // measurement (see `measure_cold`) and would land entirely on whoever went first.
    let _ = measure(1, &mut a);
    let _ = measure(1, &mut b);

    let (mut a_best, mut b_best) = (f64::INFINITY, f64::INFINITY);
    let mut ratios: Vec<f64> = Vec::with_capacity(rounds.max(1));
    for _ in 0..rounds.max(1) {
        let a1 = measure(1, &mut a).cycles;
        let b1 = measure(1, &mut b).cycles;
        let b2 = measure(1, &mut b).cycles;
        let a2 = measure(1, &mut a).cycles;
        let (ar, br) = (a1.min(a2), b1.min(b2));
        a_best = a_best.min(ar);
        b_best = b_best.min(br);
        if ar > 0.0 { ratios.push(br / ar); }
    }
    ratios.sort_by(f64::total_cmp);
    let median = if ratios.is_empty() { f64::NAN } else { ratios[ratios.len() / 2] };
    let spread = match (ratios.first(), ratios.last()) {
        (Some(lo), Some(hi)) if median != 0.0 => (hi - lo) / median * 100.0,
        _ => f64::NAN,
    };
    (a_best, b_best, median, spread)
}

/// Min-of-`runs` **wall** milliseconds — for anything whose point is elapsed time
/// (parallel speed-ups, frame budgets), where CPU time would sum over threads.
pub fn wall_ms<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    let mut lo = f64::INFINITY;
    for _ in 0..runs {
        let t = std::time::Instant::now();
        f();
        lo = lo.min(t.elapsed().as_secs_f64());
    }
    lo * 1e3
}

/// Warm CPU milliseconds, for benches that just want the one number.
pub fn cpu_ms<F: FnMut()>(samples: usize, f: F) -> f64 { measure(samples, f).ms }

/// A B B A for operations that **consume** their input — bulk builds, mainly.
///
/// [`compare2`] takes closures of no arguments, so the natural way to bench a build that eats a
/// `Vec` is `|| build(items.clone())`. That puts a large allocation, a memcpy and a free inside
/// the clock, and it is not the harmless constant it looks like. Measured on
/// `IntegerTree::bulk_load_par` at 500k items: clone-inside reported **0.85x** (parallel slower),
/// clone-outside reports **2.65x** (parallel much faster) — the same code, the same machine, the
/// same round structure, opposite conclusions. See `docs/MEASURING.md` § 8g.
///
/// So: inputs are cloned UP FRONT, outside the clock, and each timed call consumes one. Arms are
/// interleaved `A B B A` per round so first-order drift cancels within the round, and each arm is
/// reported as its minimum, which is the estimator § 8e argues for on an episodically noisy box.
///
/// Returns `(a_us, b_us, speedup_a_over_b)`.
pub fn abba<T: Clone, A: FnMut(Vec<T>), B: FnMut(Vec<T>)>(rounds: usize, items: &[T], mut a: A, mut b: B) -> (f64, f64, f64) {
    fn one<T>(f: &mut dyn FnMut(Vec<T>), input: Vec<T>) -> f64 {
        let t = std::time::Instant::now();
        f(input);
        t.elapsed().as_secs_f64() * 1e6
    }
    // Warm both: the first touch of either path is a different measurement, and rayon builds its
    // worker pool lazily on the first `join` — charging that to round one would flatter serial.
    one(&mut a, items.to_vec());
    one(&mut b, items.to_vec());
    let (mut abest, mut bbest) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..rounds.max(1) {
        let a1 = one(&mut a, items.to_vec());
        let b1 = one(&mut b, items.to_vec());
        let b2 = one(&mut b, items.to_vec());
        let a2 = one(&mut a, items.to_vec());
        abest = abest.min(a1.min(a2));
        bbest = bbest.min(b1.min(b2));
    }
    (abest, bbest, abest / bbest)
}
