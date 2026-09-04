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

`P` pause · `R` reset · `A` autopilot · `C` bake-off · sliders drag with the mouse.

The autopilot **cycles** the five acts, and they have different lengths (110 · 130 · 130 · 130 ·
260 frames) because they are not equally interesting: the frozen one needs longer than the default
**120-tick cooldown** or `BUILD-ONCE` only gets a sliver of strip before the act ends. The first
version clamped to the last act instead of wrapping, which parked the demo in the frozen state
forever — nothing moving, and it reads as a hang rather than as the build-once regime. Turning the
autopilot off also thaws, or `A` hands back a scene that is still frozen and looks broken. `$LAB_N`, `$LAB_Q`, `$LAB_R`, `$LAB_CHURN` set the opening state, so a bug report can
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

## Two defects the running demo found that no test would have

Both were reported by watching it, and both were in the *simulation*, not the display:

**Churn moved a fixed prefix.** `for i in 0..moving` picks the same agents every step, so
`churn = 0.3` meant *30 % of the population is alive and 70 % are statues* rather than *each agent
moves 30 % of the time*. On screen the field froze one act at a time and never came back. It also
quietly flattered the measurement: relocating the same slots every step has cache and leaf
locality a real 30 % churn does not, so the maintain column was timing a friendlier workload than
the knob claimed. The moving window rotates now, and a test asserts everyone moves within a
bounded number of steps, that one step still moves *about* the right fraction, and that `F` still
freezes everything.

**Query centres enumerated the agents.** Centres were `agents[q * n / queries]`, which at one cull
per item is exactly `agents[q]` — every agent the centre of its own query, lighting itself, so the
whole field went yellow whatever the radius. Centres sweep now, and the dot colour is a
**normalised count** rather than a flag: a boolean saturates precisely where the interesting
behaviour lives. At radius 4 you now see a mostly-cold field with scattered warm dots, which is
what `HITS 1.2` looks like.

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

Five runs of the headless script on this laptop, final (frozen) state:

| arm | µs/step, across five runs |
| --- | --- |
| ADAPTIVE | 6 331 · 6 621 · 7 668 · 9 064 · 9 477 |
| BRUTE | 15 531 · 16 490 · 21 276 · 22 043 · 22 504 |
| KEEPTREE | 6 897 · 7 700 · 9 587 · 9 672 · 10 049 |
| GRID | 5 415 · 5 799 · 6 858 · 7 644 · 8 550 |
| STATIC | 4 887 · 5 147 · 5 212 · 7 213 · 8 869 |

**Read the ordering, not the numbers.** The adaptive arm lands at **0.64 · 0.76 · 0.78 · 0.82 ·
0.94** of the best fixed choice — median 0.78, and a spread far too wide to quote a figure from.
Even *which* arm is best moves: `STATIC` wins four of the five, `GRID` the other. This machine's
noise is episodic (`MEASURING.md` § 8e), so a single run of this table is a story.

What *is* stable across all five is the thing the layer is actually for: **`BRUTE` costs 2.4–3.4×
the best**, every time. The policy is behind the choice you could only have made knowing how the
run ends, and far ahead of the choice a reasonable person might have pinned instead. That is
*insurance, not optimisation*, demonstrated rather than asserted.

## The lag, counted

`#149` blames the adaptive index's shortfall on detector lag plus the migration's own rebuild.
The lab measures the first half **exactly**, because it is a count of steps and not a duration:

```
5 migrations over 570 steps, 102 near-misses
lag: 102 of 570 steps (17.9%) held on a backend the policy had already rejected
     20 steps of wanting before a typical migration, worst 90
```

Nearly **a fifth of the run** is spent on a structure the policy has already decided against — the
hold window and the 120-tick cooldown holding it there. The worst single wish waited **90 steps**.

`$LAB_TRACE=<path>` writes one CSV row per step — `step, held, wanted, items, q_per_item,
mv_per_item, predicted_hits, mean_hits, maintain_us, query_us` — so the lag is arithmetic rather
than eyeballing:

```bash
LAB_HEADLESS=1 LAB_TRACE=lab.csv cargo run -p vectorial-hash-demos --bin adaptive_lab_wgpu --release
awk -F, 'NR>1 && $2!=$3' lab.csv | wc -l    # 102
```

Buffered and written once, never appended per step: a file write inside the loop is setup inside
the clock ([`MEASURING.md`](MEASURING.md) § 8g), which has already inverted one result here.

That `102` is **three independent counts agreeing** — the lab's own `LagStats`, the library's
`SwitchStats::near_misses`, and now `held != wanted` rows in the CSV. Deliberately so: the lab counts the same event
independently and a test asserts the two agree, so each is a check on the other. What the library
counter cannot give is the **distribution** — a policy that waits 20 steps five times and one that
waits 100 steps once produce the same total and want completely different fixes.

## Why the tests are in `adaptive_lab`, not in the binary

`src/adaptive_lab.rs` is graphics-free, like `horde_sim` and `siege_sim`, and the renderer and the
tests drive the same `Lab::step`. Six tests, brute-force gated:

- every reachable backend answers **identically to brute force** — the gate on all the rest;
- the knobs actually move the policy (asserting `≥ 2` migrations, so it cannot pass by standing
  still);
- a **pinned arm does not migrate**, or the bake-off would be racing the policy against itself;
- resizing keeps the slot table honest — shrink to 80, regrow to 300, every agent still findable;
- the history buffer is bounded and does not advance while paused;
- **the lag counter agrees with the library own near-miss counter**, and the per-migration
  distribution accounts for every waiting step. Two counters of the same event, written
  independently: agreeing, each is evidence for the other; disagreeing, one is a bug and there
  would be no way to tell which without the other.
