//! A screen-space overlay for the wgpu demos: coloured quads and a tiny bitmap font.
//!
//! wgpu draws no text, so every demo here that wants a label has to build one out of triangles.
//! `fluid_wgpu` grew a 3×5 font for its own HUD; this is that code lifted so a second demo does
//! not copy it. A duplicated glyph table is the kind of thing that drifts silently — one copy
//! gains a character, the other renders a blank, and nobody notices until a label reads `AD PTIVE`.
//!
//! Everything is in **pixels from the top-left**, converted to NDC on the way in, so callers can
//! lay a HUD out in the coordinates they actually think in.

// These are drawing primitives whose arguments are a genuinely flat list — a rectangle, a
// colour, and the surface to map into. Bundling them into a `Rect`/`Style`/`Surface` triple
// would make every call site longer to read and every layout tweak a two-line edit, for no
// safety gained: they are all `f32` in the same units and the compiler cannot help either way.
#![allow(clippy::too_many_arguments)]

use bytemuck::{Pod, Zeroable};

/// One overlay vertex: NDC position plus a flat colour.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
}

/// A filled rectangle at `(px, py)` pixels from the top-left of a `sw × sh` surface.
pub fn push_quad(v: &mut Vec<UiVertex>, px: f32, py: f32, w: f32, h: f32, color: [f32; 4], sw: f32, sh: f32) {
    let x0 = px / sw * 2.0 - 1.0; let x1 = (px + w) / sw * 2.0 - 1.0;
    let y0 = 1.0 - py / sh * 2.0; let y1 = 1.0 - (py + h) / sh * 2.0;
    let c = color;
    for p in [[x0, y0], [x1, y0], [x0, y1], [x0, y1], [x1, y0], [x1, y1]] { v.push(UiVertex { pos: p, color: c }); }
}

/// A 1px-ish outline, as four quads. Cheap, and it makes a slider track readable against any
/// background — which a filled rectangle alone is not.
pub fn push_frame(v: &mut Vec<UiVertex>, px: f32, py: f32, w: f32, h: f32, t: f32, color: [f32; 4], sw: f32, sh: f32) {
    push_quad(v, px, py, w, t, color, sw, sh);
    push_quad(v, px, py + h - t, w, t, color, sw, sh);
    push_quad(v, px, py, t, h, color, sw, sh);
    push_quad(v, px + w - t, py, t, h, color, sw, sh);
}

/// 3×5 bitmap font — upper case, digits, and the punctuation the HUDs actually use.
///
/// Unknown characters render blank rather than panicking: a HUD is not worth a crash, and a gap
/// in a label is self-describing. Add glyphs here rather than in a demo.
pub fn glyph(c: char) -> [&'static str; 5] {
    match c {
        '0' => ["111", "101", "101", "101", "111"], '1' => ["010", "110", "010", "010", "111"],
        '2' => ["111", "001", "111", "100", "111"], '3' => ["111", "001", "111", "001", "111"],
        '4' => ["101", "101", "111", "001", "001"], '5' => ["111", "100", "111", "001", "111"],
        '6' => ["111", "100", "111", "101", "111"], '7' => ["111", "001", "010", "010", "010"],
        '8' => ["111", "101", "111", "101", "111"], '9' => ["111", "101", "111", "001", "111"],
        'A' => ["111", "101", "111", "101", "101"], 'B' => ["110", "101", "110", "101", "110"],
        'C' => ["111", "100", "100", "100", "111"], 'D' => ["110", "101", "101", "101", "110"],
        'E' => ["111", "100", "110", "100", "111"], 'F' => ["111", "100", "110", "100", "100"],
        'G' => ["111", "100", "101", "101", "111"], 'H' => ["101", "101", "111", "101", "101"],
        'I' => ["111", "010", "010", "010", "111"], 'J' => ["001", "001", "001", "101", "111"],
        'K' => ["101", "101", "110", "101", "101"],
        'L' => ["100", "100", "100", "100", "111"], 'M' => ["101", "111", "111", "101", "101"],
        'N' => ["101", "111", "111", "111", "101"], 'O' => ["111", "101", "101", "101", "111"],
        'P' => ["111", "101", "111", "100", "100"], 'Q' => ["111", "101", "101", "111", "011"],
        'R' => ["111", "101", "111", "110", "101"], 'S' => ["111", "100", "111", "001", "111"],
        'T' => ["111", "010", "010", "010", "010"], 'U' => ["101", "101", "101", "101", "111"],
        'V' => ["101", "101", "101", "101", "010"], 'W' => ["101", "101", "111", "111", "101"],
        'X' => ["101", "101", "010", "101", "101"], 'Y' => ["101", "101", "010", "010", "010"],
        'Z' => ["111", "001", "010", "100", "111"], '.' => ["000", "000", "000", "000", "010"],
        '/' => ["001", "001", "010", "100", "100"], '-' => ["000", "000", "111", "000", "000"],
        '%' => ["101", "001", "010", "100", "101"], ':' => ["000", "010", "000", "010", "000"],
        '+' => ["000", "010", "111", "010", "000"], '=' => ["000", "111", "000", "111", "000"],
        ',' => ["000", "000", "000", "010", "100"], '!' => ["010", "010", "010", "000", "010"],
        '?' => ["111", "001", "011", "000", "010"], '<' => ["001", "010", "100", "010", "001"],
        '>' => ["100", "010", "001", "010", "100"], '(' => ["001", "010", "010", "010", "001"],
        ')' => ["100", "010", "010", "010", "100"], '*' => ["101", "010", "111", "010", "101"],
        '#' => ["101", "111", "101", "111", "101"], '_' => ["000", "000", "000", "000", "111"],
        _ => ["000", "000", "000", "000", "000"],
    }
}

/// Draw `text` at `(x, y)` pixels, one glyph pixel being `px` screen pixels. Lower case is
/// folded to upper — the font has one case, and silently uppercasing beats rendering blanks.
pub fn push_text(v: &mut Vec<UiVertex>, x: f32, y: f32, px: f32, color: [f32; 4], text: &str, sw: f32, sh: f32) {
    let mut cx = x;
    for c in text.chars() {
        for (row, bits) in glyph(c.to_ascii_uppercase()).iter().enumerate() {
            for (col, ch) in bits.char_indices() {
                if ch == '1' { push_quad(v, cx + col as f32 * px, y + row as f32 * px, px, px, color, sw, sh); }
            }
        }
        cx += 4.0 * px;
    }
}

/// Width in pixels of what [`push_text`] would draw — for right-aligning, or for sizing a
/// background plate so a label stays readable over a busy scene.
pub fn text_width(text: &str, px: f32) -> f32 {
    if text.is_empty() { 0.0 } else { text.chars().count() as f32 * 4.0 * px - px }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_is_three_by_five_and_only_ones_and_zeros() {
        // A malformed row would render as a shifted or clipped character and look like a font
        // bug rather than a data bug. Checking the table is cheaper than reading it.
        for c in ' '..='~' {
            let g = glyph(c);
            for row in g {
                assert_eq!(row.len(), 3, "glyph {c:?} has a row of width {}", row.len());
                assert!(row.chars().all(|b| b == '0' || b == '1'), "glyph {c:?} row {row:?}");
            }
        }
    }

    #[test]
    fn the_characters_the_huds_use_are_all_present() {
        // Non-vacuity for the test above, which passes just as happily on an all-blank table.
        let needed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:/-%+=,";
        for c in needed.chars() {
            assert!(glyph(c).iter().any(|r| r.contains('1')), "{c:?} renders blank");
        }
        // And a character nobody defined must be blank rather than something else's shape.
        assert!(glyph('\u{263A}').iter().all(|r| !r.contains('1')));
    }

    #[test]
    fn a_quad_lands_where_it_was_asked_to() {
        // Top-left of an 800x600 surface must map to NDC (-1, +1); the y flip is the part that
        // is easy to get backwards and impossible to see in a screenshot of a symmetric HUD.
        let mut v = Vec::new();
        push_quad(&mut v, 0.0, 0.0, 800.0, 600.0, [1.0; 4], 800.0, 600.0);
        assert_eq!(v.len(), 6);
        let xs: Vec<f32> = v.iter().map(|q| q.pos[0]).collect();
        let ys: Vec<f32> = v.iter().map(|q| q.pos[1]).collect();
        assert!((xs.iter().cloned().fold(f32::MAX, f32::min) + 1.0).abs() < 1e-6);
        assert!((ys.iter().cloned().fold(f32::MIN, f32::max) - 1.0).abs() < 1e-6);
        assert!((ys.iter().cloned().fold(f32::MAX, f32::min) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn text_width_matches_what_push_text_draws() {
        // The two are used together for right-aligned HUD rows, and a mismatch shows up as
        // labels creeping off the edge only at certain string lengths.
        for s in ["", "A", "AB", "12.3%", "ADAPTIVE"] {
            let mut v = Vec::new();
            push_text(&mut v, 0.0, 0.0, 2.0, [1.0; 4], s, 400.0, 400.0);
            let right = v.iter().map(|q| q.pos[0]).fold(f32::MIN, f32::max);
            if s.is_empty() { assert!(v.is_empty()); continue; }
            let px_right = (right + 1.0) / 2.0 * 400.0;
            assert!((px_right - text_width(s, 2.0)).abs() <= 2.0, "{s:?}: drew to {px_right}, said {}", text_width(s, 2.0));
        }
    }
}
