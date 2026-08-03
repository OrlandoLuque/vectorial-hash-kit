#!/usr/bin/env bash
# Can every measured number in the docs still be re-run?
#
# On 2026-07-29 a headline figure went stale without anyone touching the sentence that stated
# it: THREE_D.md and the README said LinearOctree3's clustered k-NN was ~5x MortonGrid3's, which
# was true when written — and then the grid's k-NN gained per-axis expansion and became 3.6x
# faster, leaving the claim wrong by a factor of three. Nothing broke. No test failed. The doc
# simply described a world that had moved.
#
# `check-web-fresh.sh` handles the same failure for published artefacts. This is its twin for
# prose, and it takes the same view: the value is in looking SYSTEMATICALLY rather than at the
# cases someone happens to remember.
#
#   scripts/check-docs-numbers.sh          # report; exit 1 if anything fails
#   scripts/check-docs-numbers.sh --list   # report only, always exit 0
#
# What it can decide mechanically:
#   (1) every `--example NAME` / `--bin NAME` a doc tells you to run still exists;
#   (2) every relative .md link resolves;
#   (3) every `#anchor` on a cross-doc link matches a heading in the target;
#   (4) every section of a REFERENCE doc that quotes a measurement names a way to reproduce it.
#
# What it deliberately does NOT do is re-run the benches and diff the numbers. That is the only
# check that would prove a figure current, and it cannot be a gate: several take minutes, and
# MEASURING.md section 8e records that this machine's noise is EPISODIC — two runs of an
# untouched op read +-3% and then +74%. A gate that fails on noise gets disabled within a week.
# So this enforces the precondition instead: a number whose source is named can be re-checked by
# hand in one command, and a number whose source is not named cannot be checked at all.
#
# Historical logs (BACKLOG.md, CLAUDE.md) are exempt from (4) by design. An entry dated
# 2026-07-24 saying "115 lib tests green" is a record of that day, not a claim about today;
# rewriting history to match the present is how a log stops being evidence.
set -uo pipefail
cd "$(dirname "$0")/.."

# Docs that describe the CURRENT state of the kit. A number here reads as "this is what it does".
REFERENCE_DOCS=(README.md docs/THREE_D.md docs/CHOOSING.md docs/PARALLEL.md docs/MEASURING.md
                docs/UPDATE_STRATEGIES.md)
# Docs that are dated records. Numbers here are history and must NOT be updated to match today.
LOG_DOCS=(docs/BACKLOG.md CLAUDE.md)

list_only=0
[ "${1:-}" = "--list" ] && list_only=1
fail=0
note() { printf '  %s\n' "$1"; }

# ------------------------------------------------------------------ (1) runnable citations
echo "== commands the docs tell you to run =="
ls crates/*/examples/*.rs crates/*/src/bin/*.rs crates/*/benches/*.rs 2>/dev/null \
  | sed -E 's#.*/##; s#\.rs$##' | sort -u > /tmp/_dn_real.txt
missing=0
while read -r name; do
  [ -z "$name" ] && continue
  if ! grep -qx "$name" /tmp/_dn_real.txt; then
    note "MISSING: docs cite '$name' but no example/bin/bench by that name exists"
    missing=1; fail=1
  fi
done < <(grep -rhoE '(--example|--bin|--bench) [a-zA-Z0-9_]+' README.md docs/*.md crates/*/README.md 2>/dev/null \
         | awk '{print $2}' | sort -u)
[ "$missing" = 0 ] && note "every cited example/bin/bench exists"

# ------------------------------------------------------------------ (2)+(3) links and anchors
echo "== cross-document links =="
badlink=0
for f in README.md docs/*.md; do
  [ -f "$f" ] || continue
  d=$(dirname "$f")
  while read -r link; do
    [ -z "$link" ] && continue
    target=${link%%#*}
    anchor=${link#*#}
    [ "$anchor" = "$link" ] && anchor=""
    if [ ! -f "$d/$target" ]; then
      note "BROKEN: $f -> $target"; badlink=1; fail=1; continue
    fi
    if [ -n "$anchor" ]; then
      # GitHub slugs: lowercase, spaces to hyphens, drop everything else.
      if ! grep -hE '^#{1,6} ' "$d/$target" \
        | sed -E 's/^#+ //' | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9 _-]//g; s/ /-/g' | grep -qx "$anchor"; then
        note "BROKEN ANCHOR: $f -> $target#$anchor"; badlink=1; fail=1
      fi
    fi
  done < <(grep -ohE '\]\([A-Za-z0-9_./-]+\.md(#[A-Za-z0-9_-]+)?\)' "$f" | sed -E 's/^\]\(//; s/\)$//')
done
[ "$badlink" = 0 ] && note "every relative link and anchor resolves"

# ------------------------------------------------------------------ (4) reproducible numbers
#
# Walk each reference doc section by section. A section "quotes a measurement" if it contains a
# ratio (1.42x), a timing (3.7 ms, 840 us) or a throughput. It "names a source" if it, or the
# section heading above it, mentions a runnable thing: an example/bin/bench name, a cargo
# command, or a `tests/`+`examples/` path. Sections that measure without naming a source are
# what this reports — not because the number is wrong, but because nobody can find out.
echo "== can each quoted measurement be reproduced? =="
unsourced=0
for f in "${REFERENCE_DOCS[@]}"; do
  [ -f "$f" ] || continue
  awk -v FILE="$f" -v NAMES="$(paste -sd' ' /tmp/_dn_real.txt)" '
    BEGIN { nn = split(NAMES, runnable, " ") }
    # "Names a source" means: mentions something you can actually run. Deciding that against the
    # REAL set of examples/bins/benches beats matching a phrasing convention — the first version
    # of this check demanded "cargo run --example x" and flagged a dozen sections that named
    # their source perfectly well as `critters3d_headless --parallel` or as a path to the file.
    # The docs were right and the check was wrong, which is the failure mode to design against.
    function sourced(s,   i) {
      if (s ~ /(crates\/[a-z-]+\/(src\/bin|examples|benches)\/|cargo run|cargo bench|cargo test)/) return 1
      for (i = 1; i <= nn; i++) if (index(s, runnable[i]) > 0) return 1
      return 0
    }
    # A subsection inherits its source from the section above it: docs are written as
    # "## X (cargo run --example x)" followed by "### the table". Requiring every heading to
    # repeat the command would flag correct prose, and a check that cries wolf gets ignored.
    function ancestor_sourced(  i) {
      for (i = 0; i <= level; i++) if (src[i]) return 1
      return 0
    }
    function flush(  has_num) {
      if (buf == "") { return }
      has_num = (buf ~ /[0-9]\.?[0-9]*[[:space:]]*[x×]([^a-zA-Z]|$)/) ||
                (buf ~ /[0-9][[:space:]]*(ms|us|µs|ns)([^a-zA-Z]|$)/)
      if (sourced(buf)) src[level] = 1
      if (has_num && !ancestor_sourced()) {
        printf "  UNSOURCED: %s:%d  section \"%s\"\n", FILE, hline, htitle
        bad++
      }
      buf = ""
    }
    /^#{1,6} / {
      flush()
      htitle = $0; sub(/^#+ /, "", htitle); hline = NR
      newlevel = length($1)                       # "##" -> 2
      for (i = newlevel; i <= 6; i++) src[i] = 0  # a new sibling does not inherit its predecessor
      level = newlevel
      if (sourced($0)) src[level] = 1             # the heading itself may name the command
      next
    }
    { buf = buf "\n" $0 }
    END { flush(); exit (bad > 0 ? 1 : 0) }
  ' "$f" || { unsourced=1; fail=1; }
done
[ "$unsourced" = 0 ] && note "every section quoting a measurement names a way to re-run it"

# ------------------------------------------------------------------ the part a human must do
echo "== not checked here =="
note "whether the numbers are still TRUE — re-run the named bench. This checks only that you can."
note "log docs (${LOG_DOCS[*]}) are exempt: a dated entry is a record, not a claim about today."

echo
if [ "$list_only" = 1 ]; then exit 0; fi
if [ "$fail" = 1 ]; then
  echo "Something in the docs cannot be reproduced or no longer resolves."
  exit 1
fi
echo "every documented measurement names a way to re-run it, and every link resolves."
