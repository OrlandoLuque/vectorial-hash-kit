//! `SphereStencil` — the right shape of "template" for a **dense uniform grid**.
//!
//! The rest of the kit indexes *sparse points in space*: a query descends a tree and asks
//! each node box a question. A voxel world is the opposite problem — the data is a dense
//! array, positions are integers, and the array itself is the index — so the useful
//! precomputation is different, and the difference is measurable.
//!
//! **A membership bitmap loses.** Precomputing "is this cell inside the sphere" as one bit
//! per cell and looking it up is *slower than recomputing it*: measured in
//! `examples/voxel_select_bench`, 15–20% slower than a naive `dx²+dy²+dz² <= r²` at every
//! radius. It trades three multiplies for a lookup that misses cache. The rule it teaches:
//!
//! > A precomputed template pays in proportion to the **cost of the question it removes**.
//!
//! Point-in-polygon is expensive, and the kit's 2D templates measure 4–19× there.
//! Point-in-sphere is three multiplies, so the only template worth having is one that
//! removes the **loop structure**, not one that answers membership.
//!
//! **That template is a run table.** A sphere meets each `(dy, dz)` row of a grid in exactly
//! **one contiguous run of x** — so the table is `O(r²)` instead of `O(r³)`, and the inner
//! loop becomes a straight walk over adjacent memory with no test and no branch. Measured
//! ~1.8× at r=8 and ~2.0× at r=32 over the naive triple loop, and ~2.9× when whole empty
//! chunks are skipped as well.
//!
//! ## Two tables, because the shell is not the ends of the runs
//!
//! It is tempting to say "the interior of the run is fully covered, its two ends are the
//! partial cells". That is **wrong**, and by a lot: at r=16 it mislabels 716 partial cells
//! as full and finds only a third of the shell. It misses the *caps* — a cell in the middle
//! of the top row is on the boundary through its **top face**, not through the ends of its
//! row.
//!
//! So each row carries two runs: the cells whose **farthest corner** is inside (fully
//! covered) and the cells the sphere merely **touches**. Both conditions are monotone in
//! `|dx|`, so both stay contiguous — the shell is the difference between them, and rows near
//! the poles correctly come out as "all shell, no interior".
//!
//! ## Alignment is a precondition, not a detail
//!
//! Where the sphere's centre sits *within* a cell changes the answer, and a wrong choice
//! produces a set that is self-consistent, symmetric, and shifted by half a cell. Hence
//! [`Alignment`] is explicit and has no default.

/// Where the sphere's centre sits relative to the cell lattice. Cell `k` spans `[k, k+1)`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Alignment {
    /// At the centre of a cell (`world = k + 0.5`). The usual choice for an explosion
    /// centred on a block.
    CellCentre,
    /// At the corner where eight cells meet (`world = k`).
    CellCorner,
    /// An arbitrary sub-cell offset in `[0, 1)` per axis, for a centre that is not snapped.
    /// Costs a table per distinct phase, so quantise before reaching for it.
    Phase(f64, f64, f64),
}

impl Alignment {
    /// The centre in world coordinates, taking the origin cell as `(0, 0, 0)`.
    fn centre(self) -> (f64, f64, f64) {
        match self {
            Alignment::CellCentre => (0.5, 0.5, 0.5),
            Alignment::CellCorner => (0.0, 0.0, 0.0),
            Alignment::Phase(x, y, z) => (x, y, z),
        }
    }
}

/// One `(dy, dz)` row of the stencil: two contiguous runs of `x`, in cell offsets from the
/// origin cell.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Row {
    pub dy: i32,
    pub dz: i32,
    /// Cells the sphere touches at all: `touch.0 ..= touch.1`. Never empty (a row that the
    /// sphere misses entirely is not emitted).
    pub touch: (i32, i32),
    /// Cells lying entirely inside: `full.0 ..= full.1`. `None` for a row that the sphere
    /// only grazes — every cell in it is shell, which is exactly the polar-cap case a
    /// single-table stencil gets wrong.
    pub full: Option<(i32, i32)>,
}

impl Row {
    /// The shell of this row: up to two runs, the parts of `touch` outside `full`.
    pub fn partial(&self) -> [Option<(i32, i32)>; 2] {
        match self.full {
            None => [Some(self.touch), None],
            Some((f0, f1)) => [
                (self.touch.0 < f0).then_some((self.touch.0, f0 - 1)),
                (f1 < self.touch.1).then_some((f1 + 1, self.touch.1)),
            ],
        }
    }
}

/// A precomputed sphere, as contiguous runs per row. Build once per (radius, alignment) and
/// cache it: the build is `O(r²)` and a program uses a handful of radii.
#[derive(Clone, Debug)]
pub struct SphereStencil {
    radius: f64,
    align: Alignment,
    rows: Vec<Row>,
}

impl SphereStencil {
    /// Build the stencil. `radius` is in cells.
    pub fn new(radius: f64, align: Alignment) -> SphereStencil {
        let (ex, ey, ez) = align.centre();
        let r2 = radius * radius;
        let reach = radius.ceil() as i32 + 2;
        // Per axis, the distance from the centre to a cell's nearest point and to its
        // farthest corner. Cell k spans [k, k+1).
        let near = |k: i32, e: f64| { let (lo, hi) = (k as f64, k as f64 + 1.0); (lo - e).max(e - hi).max(0.0).powi(2) };
        let far = |k: i32, e: f64| { let (lo, hi) = (k as f64, k as f64 + 1.0); (lo - e).abs().max((hi - e).abs()).powi(2) };

        let mut rows = Vec::new();
        for dz in -reach..=reach {
            for dy in -reach..=reach {
                let rest_near = r2 - near(dy, ey) - near(dz, ez);
                if rest_near < 0.0 { continue; } // the sphere misses this row entirely
                let rest_far = r2 - far(dy, ey) - far(dz, ez);
                // Both conditions are monotone in the distance along x, so each is one run:
                // scan outward from the cell containing the centre.
                let mut touch = None::<(i32, i32)>;
                let mut full = None::<(i32, i32)>;
                for dx in -reach..=reach {
                    if near(dx, ex) <= rest_near {
                        touch = Some(match touch { None => (dx, dx), Some((a, _)) => (a, dx) });
                    }
                    if rest_far >= 0.0 && far(dx, ex) <= rest_far {
                        full = Some(match full { None => (dx, dx), Some((a, _)) => (a, dx) });
                    }
                }
                if let Some(t) = touch { rows.push(Row { dy, dz, touch: t, full }); }
            }
        }
        SphereStencil { radius, align, rows }
    }

    pub fn radius(&self) -> f64 { self.radius }
    pub fn alignment(&self) -> Alignment { self.align }
    /// The rows, each a pair of contiguous runs. Iterate these instead of testing cells.
    pub fn rows(&self) -> &[Row] { &self.rows }
    /// Cells lying entirely inside the sphere.
    pub fn full_count(&self) -> usize {
        self.rows.iter().filter_map(|r| r.full).map(|(a, b)| (b - a + 1) as usize).sum()
    }
    /// Cells the sphere touches without containing — the shell.
    pub fn partial_count(&self) -> usize {
        self.rows.iter().map(|r| {
            let t = (r.touch.1 - r.touch.0 + 1) as usize;
            let f = r.full.map_or(0, |(a, b)| (b - a + 1) as usize);
            t - f
        }).sum()
    }
    /// Every touched cell, for callers that want the whole set rather than the runs. The
    /// runs are the point of the type; this exists for tests and for one-off uses.
    pub fn cells(&self) -> impl Iterator<Item = (i32, i32, i32, bool)> + '_ {
        self.rows.iter().flat_map(|r| {
            (r.touch.0..=r.touch.1).map(move |dx| {
                let full = r.full.is_some_and(|(a, b)| a <= dx && dx <= b);
                (dx, r.dy, r.dz, full)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (fully covered cells, shell cells) — named so the comparison signatures stay readable.
    type Split = (Vec<(i32, i32, i32)>, Vec<(i32, i32, i32)>);

    /// World-space truth: classify every cell in the bounding box directly. Deliberately
    /// written from the geometry rather than from the stencil's algebra, so it is an
    /// independent path and not a restatement.
    fn brute(align: Alignment, r: f64) -> Split {
        let (ex, ey, ez) = align.centre();
        let reach = r.ceil() as i32 + 2;
        let (mut full, mut part) = (Vec::new(), Vec::new());
        for dz in -reach..=reach {
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let d = |k: i32, e: f64| (k as f64, k as f64 + 1.0, e);
                    let far: f64 = [d(dx, ex), d(dy, ey), d(dz, ez)].iter()
                        .map(|(lo, hi, e)| (lo - e).abs().max((hi - e).abs()).powi(2)).sum();
                    let near: f64 = [d(dx, ex), d(dy, ey), d(dz, ez)].iter()
                        .map(|(lo, hi, e)| (lo - e).max(e - hi).max(0.0).powi(2)).sum();
                    if far <= r * r { full.push((dx, dy, dz)); }
                    else if near <= r * r { part.push((dx, dy, dz)); }
                }
            }
        }
        (full, part)
    }

    fn from_stencil(s: &SphereStencil) -> Split {
        let (mut full, mut part) = (Vec::new(), Vec::new());
        for (x, y, z, is_full) in s.cells() {
            if is_full { full.push((x, y, z)); } else { part.push((x, y, z)); }
        }
        full.sort_unstable(); part.sort_unstable();
        (full, part)
    }

    #[test]
    fn matches_brute_force_in_every_alignment() {
        for align in [Alignment::CellCentre, Alignment::CellCorner, Alignment::Phase(0.3125, 0.75, 0.5625)] {
            for r in [2.0, 5.0, 8.0, 12.5, 16.0] {
                let s = SphereStencil::new(r, align);
                let (mut bf, mut bp) = brute(align, r);
                bf.sort_unstable(); bp.sort_unstable();
                let (sf, sp) = from_stencil(&s);
                assert_eq!(bf, sf, "full set differs, {align:?} r={r}");
                assert_eq!(bp, sp, "partial set differs, {align:?} r={r}");
                assert_eq!(s.full_count(), bf.len());
                assert_eq!(s.partial_count(), bp.len());
            }
        }
    }

    /// The property the whole design rests on: one contiguous run per row, for both the
    /// touched cells and the fully-covered ones. If this ever fails for a shape, that shape
    /// needs a list of runs per row instead of a pair.
    #[test]
    fn every_row_is_two_contiguous_runs() {
        for align in [Alignment::CellCentre, Alignment::CellCorner, Alignment::Phase(0.1, 0.9, 0.5)] {
            let s = SphereStencil::new(9.0, align);
            for row in s.rows() {
                assert!(row.touch.0 <= row.touch.1, "empty touch run emitted");
                if let Some((a, b)) = row.full {
                    assert!(a <= b, "empty full run emitted");
                    assert!(row.touch.0 <= a && b <= row.touch.1, "full run escapes the touch run");
                }
                // and the shell is what is left over, with nothing counted twice
                let shell: usize = row.partial().iter().flatten().map(|(a, b)| (b - a + 1) as usize).sum();
                let touch = (row.touch.1 - row.touch.0 + 1) as usize;
                let full = row.full.map_or(0, |(a, b)| (b - a + 1) as usize);
                assert_eq!(shell + full, touch, "row {row:?} does not partition");
            }
        }
    }

    /// The mistake this type exists to prevent: "the run's two ends are the partial cells".
    /// It misses the polar caps, where a whole row is shell and has no interior at all.
    #[test]
    fn polar_rows_are_all_shell() {
        let s = SphereStencil::new(8.0, Alignment::CellCentre);
        let capless = s.rows().iter().filter(|r| r.full.is_none()).count();
        assert!(capless > 0, "no all-shell rows: the two-table split would be pointless");
        // and those rows still contain cells, i.e. they are real rows the naive rule would
        // have declared fully covered
        assert!(s.rows().iter().filter(|r| r.full.is_none()).all(|r| r.touch.0 <= r.touch.1));
    }

    /// Alignment is a precondition: the same radius gives different sets, all of them valid.
    #[test]
    fn alignments_disagree_and_that_is_the_point() {
        let c = SphereStencil::new(8.0, Alignment::CellCentre);
        let k = SphereStencil::new(8.0, Alignment::CellCorner);
        assert_ne!(c.full_count(), k.full_count(),
            "if these matched, a half-cell shift would be undetectable");
    }
}
