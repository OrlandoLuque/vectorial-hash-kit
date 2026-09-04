# `adaptive_lab` — change the workload with your hands, watch the index change its mind

```bash
cargo run -p vectorial-hash-demos --bin adaptive_lab_wgpu --release
LAB_HEADLESS=1 cargo run -p vectorial-hash-demos --bin adaptive_lab_wgpu --release
```

`AdaptiveIndex` exists for a workload whose character **changes**. Until this demo, nothing in the
repo let you watch that happen:

| | load varies? | visible? |
| --- | --- | --- |
| `examples/adaptive_vs_pinned` | yes, four acts | no — headless, and it reports a *total* |
| `fluid_wgpu` | no — an SPH tank looks the same in frame 10 000 as in frame 100 | yes |
| the horde | yes, but its adaptive arm only ever walks `Brute → KeepTree` | yes |

A total is the problem. **Flapping, lag, and a plainly wrong choice all look identical in a
total** — one number, slightly too big. On a screen they look nothing alike, which is the entire
argument for building this rather than a sixth bench.

## The four knobs are the four boundaries

| control | knob | the threshold it crosses |
| --- | --- | --- |
| `[` `]` | population | `brute_max` — is an index worth having at all? |
| `,` `.` | queries per item | `rebuild_query_ratio` — grid, or keep-tree? |
| `-` `=` | query radius | `grid_min_hits` — do the queries *find* anything? |
| `;` `'` | churn | (no threshold; it sets what maintenance costs) |
| `F` | freeze | `static_ticks` — has everything stopped moving? |

`P` pause · `R` reset · `A` autopilot (walks the five acts) · `C` bake-off · sliders drag with
the mouse. `$LAB_N`, `$LAB_Q`, `$LAB_R`, `$LAB_CHURN` set the opening state, so a bug report can
be a command line rather than a list of instructions.

The **population slider is logarithmic**. `brute_max` defaults to 64, which on a linear 8–20 000
track is two pixels from the left end: the entire brute-scan regime would be unreachable with a
mouse.

## The strip is the demo

One column per step along the bottom, coloured by the backend that was live — grey brute, blue
keep-tree, amber grid, green build-once. Running the autopilot draws the whole argument:

```
............TTTTTTTTTTTTTTTTTTTTTTGGGGGGGGGGGGGGTTTTTTTTTTTTTTTTGGGGGGGGGGGGGGGGGSSSSSSSS
 quiet       grown, churning       query storm    SAME storm,      wide again       frozen
 tiny        few queries           wide radius    NARROW radius
```

The fourth band is the one worth staring at. **Nothing changed but the radius** — same 4 000
agents, same one-cull-per-item — and the policy left the grid. Its queries were finding 1.2 items
each, and a grid pays a hash lookup per cell whether or not the cell holds anything. That is
`Thresholds::grid_min_hits`, and this is the first place it can be *seen*.

## What the HUD says, and why those numbers are next to each other

```
BUILD-ONCE
5 MIG  102 NEAR  148FPS
N 4000  Q/I 0.40  MV/I 0.00
HITS 22.2 PREDICT 22.0
MAINTAIN 0US
QUERY 4759US
```

`HITS` against `PREDICT` is the policy's input against the truth, side by side. A rule can have a
correct threshold and still decide wrongly if what it is comparing against the threshold is wrong
— which is exactly what happened to `grid_min_hits`, shipped disabled for a day because
`expected_hits` was estimating density from the declared world volume and reading **7.9× low** on
slab-shaped data ([`MEASURING.md`](MEASURING.md), `AdaptiveIndex::expected_hits`). Those two
numbers agreeing on screen is that fix, visible.

`NEAR` — hysteresis crossings that did **not** migrate — is how lag shows up as a number. 102 of
them over 570 steps.

## The lag is real and the demo found it

The first version of the headless script gave the frozen act 90 steps and it ended on **GRID**,
not build-once. That is not a bug: the default `cooldown` is **120 ticks**, so the policy wanted
to move and was not allowed to yet. At 240 steps it arrives. On screen it is a band of amber that
turns green late; in the counter it is `NEAR` climbing while nothing changes.

## `C` — is the choice actually right?

The HUD can say what the policy *chose*. Only a race can say whether that was correct, so `C`
clones the live state and runs every backend **pinned** — `migrate_to` + `freeze`, i.e. the same
code path with the decision switched off, rather than four separately-built structures. Arms run
forward then in reverse, each keeping its minimum ([`MEASURING.md`](MEASURING.md) § 8e).

On the final (frozen) state of the autopilot run, this laptop:

| arm | µs/step |
| --- | ---: |
| ADAPTIVE | 9 064 |
| BRUTE | 22 504 |
| KEEPTREE | 10 049 |
| **GRID** | **8 550** |
| STATIC | 8 869 |

The policy is **6 % behind** the best fixed choice — and the best fixed choice is the one you
could only have made by knowing how the run would end. That is the honest case for the layer,
stated the way the module docs state it: *insurance, not optimisation*. Treat the ordering as the
result and not the 6 %: these arms are a few percent apart on a machine whose noise is episodic.

## Why the tests are in `adaptive_lab`, not in the binary

`src/adaptive_lab.rs` is graphics-free, like `horde_sim` and `siege_sim`, and the renderer and the
tests drive the same `Lab::step`. Five tests, brute-force gated:

- every reachable backend answers **identically to brute force** — the gate on all the rest;
- the knobs actually move the policy (asserting `≥ 2` migrations, so it cannot pass by standing
  still);
- a **pinned arm does not migrate**, or the bake-off would be racing the policy against itself;
- resizing keeps the slot table honest — shrink to 80, regrow to 300, every agent still findable;
- the history buffer is bounded and does not advance while paused.
