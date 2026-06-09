# vectorial-hash-demos

Runnable demos for the [`vectorial-hash-kit`](../../README.md) workspace. Not published to crates.io (`publish = false`).

## Demos

The default `cargo run -p vectorial-hash-demos` runs a small in-memory generation over a "drop" polygon at four angles (0°, 45°, 90°, 135°) on a 16-cell grid, showing how the 8-symmetry dedup collapses rotations onto canonical templates.

Expected output:

```
  angle   0.0deg -> id 1 via eq (new: true)
  angle  45.0deg -> id 2 via eq (new: true)
  angle  90.0deg -> id 1 via rCC (new: false)
  angle 135.0deg -> id 2 via rC  (new: false)
Unique templates: 2
```

## Run

```bash
cargo run -p vectorial-hash-demos
```
