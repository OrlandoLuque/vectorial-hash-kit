//! Bridge between this crate's generation pipeline and `vectorial-hash`'s
//! runtime `TemplateGrid`.
//!
//! The generator side speaks `Matrix` (col-major `Vec<Vec<u8>>`, values
//! `OUT`/`MAYBE`/`IN`). The runtime side speaks
//! [`vectorial_hash::TemplateGrid`] (row-major `Vec<CellState>` with anchor
//! and cell size baked in). This module provides:
//!
//! - [`matrix_to_template_grid`]: lossless conversion to the runtime layout.
//! - [`apply_inverse_op`]: undo one of the 8 symmetry ops the dedup pipeline
//!   uses, so a canonical template can be re-oriented to the query position.
//! - [`decode_binary`]: best-effort inverse of [`crate::matrix::bin_code`]
//!   (lossy at the trailing edge — see its docs).
//! - [`TemplateKey`] / [`TemplateIndex`]: hash-keyed runtime index. O(1)
//!   lookup on the cull hot path; templates are held decoded in RAM by
//!   design (time-over-memory tradeoff for indexing).

use std::collections::HashMap;

use vectorial_hash::{CellState, Point, TemplateGrid};

use crate::matrix::{self, Matrix};
use crate::templates::{IN, MAYBE, OUT};

/// Convert a generator-side [`Matrix`] (col-major, `OUT`/`MAYBE`/`IN`) into a
/// runtime [`TemplateGrid`] (row-major, [`CellState`]). The grid is anchored
/// at `anchor` with cells of size `cell_w` × `cell_h`.
pub fn matrix_to_template_grid(
    m: &Matrix,
    anchor: Point,
    cell_w: f64,
    cell_h: f64,
) -> TemplateGrid {
    let (cols, rows) = matrix::dimensions(m);
    let mut cells = Vec::with_capacity(cols * rows);
    for y in 0..rows {
        for x in 0..cols {
            cells.push(cell_state_from_u8(m[x][y]));
        }
    }
    TemplateGrid::new(
        anchor,
        cell_w,
        cell_h,
        cols as u32,
        rows as u32,
        cells,
    )
}

fn cell_state_from_u8(v: u8) -> CellState {
    match v {
        OUT => CellState::Out,
        MAYBE => CellState::Maybe,
        IN => CellState::In,
        _ => CellState::Out,
    }
}

/// Apply the inverse of one of the 8 symmetry ops used by the dedup pipeline
/// (`eq`, `rCC`, `rC`, `r180`, `fLR`, `fTB`, `fTLBR`, `fTRBL`).
///
/// Returns `None` if `op` isn't one of the recognised names.
///
/// Pairings (each op is its own inverse except for the two 90° rotations):
/// - `eq` ⇔ `eq`
/// - `rCC` ⇔ `rC`
/// - `r180`, `fLR`, `fTB`, `fTLBR`, `fTRBL` are involutions.
pub fn apply_inverse_op(op: &str, m: &Matrix) -> Option<Matrix> {
    match op {
        "eq" => Some(matrix::equal(m)),
        "rCC" => Some(matrix::rotate_clockwise_90(m)),
        "rC" => Some(matrix::rotate_counter_clockwise_90(m)),
        "r180" => Some(matrix::rotate_180(m)),
        "fLR" => Some(matrix::flip_lr(m)),
        "fTB" => Some(matrix::flip_tb(m)),
        "fTLBR" => Some(matrix::flip_tlbr(m)),
        "fTRBL" => Some(matrix::flip_trbl(m)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Input has fewer than the 2 header bytes.
    TooShort,
    /// Input header promises more cells than the payload contains.
    Truncated,
}

/// Best-effort decoder for the binary template format produced by
/// [`crate::matrix::bin_code`].
///
/// Format: 2-byte header (`cols`, `rows`) + 2 bits per cell in row-major
/// order, top 2 bits of each byte = first cell.
///
/// **Lossy at the tail**: the encoder preserves a PHP bug where the final
/// partial byte (1–3 trailing cells when `cols * rows` isn't divisible by 4)
/// isn't flushed. This decoder fills those missing cells with `OUT`. For
/// round-trip fidelity, store matrices directly via
/// [`matrix_to_template_grid`] instead of going through `bin_code`.
pub fn decode_binary(bytes: &[u8]) -> Result<Matrix, DecodeError> {
    if bytes.len() < 2 {
        return Err(DecodeError::TooShort);
    }
    let cols = bytes[0] as usize;
    let rows = bytes[1] as usize;
    if cols == 0 || rows == 0 {
        return Ok(vec![Vec::new(); cols]);
    }
    let full_bytes = (cols * rows) / 4;
    if bytes.len() < 2 + full_bytes {
        return Err(DecodeError::Truncated);
    }
    let payload = &bytes[2..2 + full_bytes];

    let mut m: Matrix = vec![vec![OUT; rows]; cols];
    let mut idx = 0usize;
    'outer: for y in 0..rows {
        for x in 0..cols {
            let byte_idx = idx / 4;
            if byte_idx >= payload.len() {
                // PHP-bug tail: leave remaining cells as OUT.
                break 'outer;
            }
            let pair_in_byte = idx % 4;
            let shift = (3 - pair_in_byte) * 2;
            m[x][y] = (payload[byte_idx] >> shift) & 0b11;
            idx += 1;
        }
    }
    Ok(m)
}

/// Stable key used to address a precomputed template at runtime.
///
/// Floats are stored as their bit pattern so the key is `Hash + Eq` and
/// distinct-but-equal floats hash identically. Template parameters should
/// never be `NaN`; the bit-pattern key doesn't try to handle that case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemplateKey {
    pub polygon_id: u32,
    pub scale_bits: u64,
    pub angle_bits: u64,
    pub grid_size: i64,
    pub x: i32,
    pub y: i32,
}

impl TemplateKey {
    pub fn new(polygon_id: u32, scale: f64, angle: f64, grid_size: i64, x: i32, y: i32) -> Self {
        Self {
            polygon_id,
            scale_bits: scale.to_bits(),
            angle_bits: angle.to_bits(),
            grid_size,
            x,
            y,
        }
    }

    pub fn scale(&self) -> f64 { f64::from_bits(self.scale_bits) }
    pub fn angle(&self) -> f64 { f64::from_bits(self.angle_bits) }
}

/// Hash-keyed runtime index of decoded templates.
///
/// O(1) lookup on the cull hot path. Each entry owns a fully-decoded
/// [`TemplateGrid`] — no per-lookup transformation, no per-cell arithmetic
/// beyond what `classify_region` already does. This is the explicit
/// time-over-memory tradeoff for indexing.
#[derive(Default)]
pub struct TemplateIndex {
    map: HashMap<TemplateKey, TemplateGrid>,
}

impl TemplateIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { map: HashMap::with_capacity(capacity) }
    }

    pub fn insert(&mut self, key: TemplateKey, grid: TemplateGrid) -> Option<TemplateGrid> {
        self.map.insert(key, grid)
    }

    pub fn get(&self, key: &TemplateKey) -> Option<&TemplateGrid> {
        self.map.get(key)
    }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    pub fn iter(&self) -> impl Iterator<Item = (&TemplateKey, &TemplateGrid)> {
        self.map.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::bin_code;
    use vectorial_hash::CellState as CS;

    fn matrix_3x2_mixed() -> Matrix {
        // 3 cols × 2 rows. Visually (origin top-left, y growing downward):
        //   y=0: IN    MAYBE  OUT
        //   y=1: OUT   IN     MAYBE
        vec![
            vec![IN, OUT],
            vec![MAYBE, IN],
            vec![OUT, MAYBE],
        ]
    }

    #[test]
    fn matrix_to_template_grid_preserves_layout() {
        let m = matrix_3x2_mixed();
        let g = matrix_to_template_grid(&m, Point::new(0.0, 0.0), 10.0, 10.0);
        assert_eq!(g.cols, 3);
        assert_eq!(g.rows, 2);
        assert_eq!(g.cell(0, 0), CS::In);
        assert_eq!(g.cell(1, 0), CS::Maybe);
        assert_eq!(g.cell(2, 0), CS::Out);
        assert_eq!(g.cell(0, 1), CS::Out);
        assert_eq!(g.cell(1, 1), CS::In);
        assert_eq!(g.cell(2, 1), CS::Maybe);
    }

    #[test]
    fn binary_round_trip_works_when_cells_divisible_by_4() {
        // 2×2 = 4 cells, exactly one byte payload — no PHP-bug tail.
        let m: Matrix = vec![vec![IN, MAYBE], vec![OUT, IN]];
        let encoded = bin_code(&m);
        assert_eq!(encoded.len(), 2 + 1);
        let decoded = decode_binary(&encoded).unwrap();
        assert_eq!(decoded, m);
    }

    #[test]
    fn binary_round_trip_handles_php_bug_tail() {
        // 1 × 3 = 3 cells, all in the PHP-bug zone (no full byte ever flushed).
        let m: Matrix = vec![vec![IN, MAYBE, OUT]];
        let encoded = bin_code(&m);
        assert_eq!(encoded.len(), 2);
        let decoded = decode_binary(&encoded).unwrap();
        // Decoder fills the unknown tail with OUT.
        assert_eq!(decoded, vec![vec![OUT, OUT, OUT]]);
    }

    #[test]
    fn binary_decoder_rejects_too_short_input() {
        assert_eq!(decode_binary(&[]), Err(DecodeError::TooShort));
        assert_eq!(decode_binary(&[5]), Err(DecodeError::TooShort));
    }

    #[test]
    fn binary_decoder_rejects_truncated_payload() {
        // Header promises 4×4 = 16 cells = 4 full bytes, but payload has 0.
        let r = decode_binary(&[4, 4]);
        assert_eq!(r, Err(DecodeError::Truncated));
    }

    #[test]
    fn inverse_ops_round_trip_for_all_symmetries() {
        let m = matrix_3x2_mixed();
        for (op_name, forward) in [
            ("eq", matrix::equal(&m)),
            ("rCC", matrix::rotate_counter_clockwise_90(&m)),
            ("rC", matrix::rotate_clockwise_90(&m)),
            ("r180", matrix::rotate_180(&m)),
            ("fLR", matrix::flip_lr(&m)),
            ("fTB", matrix::flip_tb(&m)),
            ("fTLBR", matrix::flip_tlbr(&m)),
            ("fTRBL", matrix::flip_trbl(&m)),
        ] {
            let back = apply_inverse_op(op_name, &forward).unwrap();
            assert_eq!(back, m, "inverse of {} should restore the original", op_name);
        }
    }

    #[test]
    fn inverse_op_returns_none_for_unknown_name() {
        let m = matrix_3x2_mixed();
        assert!(apply_inverse_op("nope", &m).is_none());
    }

    #[test]
    fn template_index_hash_lookup_distinguishes_keys() {
        let m = matrix_3x2_mixed();
        let grid = matrix_to_template_grid(&m, Point::new(0.0, 0.0), 10.0, 10.0);

        let mut idx = TemplateIndex::new();
        let k = TemplateKey::new(1, 128.0, 70.5, 16, 3, 5);
        idx.insert(k, grid);

        assert!(idx.get(&k).is_some());
        // Different polygon → miss.
        assert!(idx.get(&TemplateKey::new(2, 128.0, 70.5, 16, 3, 5)).is_none());
        // Different position → miss.
        assert!(idx.get(&TemplateKey::new(1, 128.0, 70.5, 16, 3, 6)).is_none());
        // Same float values constructed independently → hit (bit-identical).
        let k2 = TemplateKey::new(1, 128.0, 70.5, 16, 3, 5);
        assert!(idx.get(&k2).is_some());
    }

    #[test]
    fn template_index_accessors_reflect_inserts() {
        let m = matrix_3x2_mixed();
        let grid = matrix_to_template_grid(&m, Point::new(0.0, 0.0), 10.0, 10.0);

        let mut idx = TemplateIndex::with_capacity(4);
        assert!(idx.is_empty());
        idx.insert(TemplateKey::new(1, 1.0, 0.0, 8, 0, 0), grid);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.iter().count(), 1);
    }
}
