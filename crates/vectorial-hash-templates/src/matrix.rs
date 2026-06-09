/// 2D matrix operations and binary encoding for template deduplication

pub type Matrix = Vec<Vec<u8>>; // matrix[x][y] = 0 (OUT), 1 (MAYBE), 2 (IN)

pub fn dimensions(m: &Matrix) -> (usize, usize) {
    if m.is_empty() { return (0, 0); }
    (m.len(), m[0].len())
}

pub fn equal(m: &Matrix) -> Matrix {
    m.clone()
}

pub fn rotate_clockwise_90(m: &Matrix) -> Matrix {
    let (tx, ty) = dimensions(m);
    let mut r = vec![vec![0u8; tx]; ty];
    for x in 0..tx {
        for y in 0..ty {
            r[ty - y - 1][x] = m[x][y];
        }
    }
    r
}

pub fn rotate_counter_clockwise_90(m: &Matrix) -> Matrix {
    let (tx, ty) = dimensions(m);
    let mut r = vec![vec![0u8; tx]; ty]; // Note: dimensions swap
    for x in 0..tx {
        for y in 0..ty {
            r[y][tx - x - 1] = m[x][y];
        }
    }
    r
}

pub fn rotate_180(m: &Matrix) -> Matrix {
    let (tx, ty) = dimensions(m);
    let mut r = vec![vec![0u8; ty]; tx];
    for x in 0..tx {
        for y in 0..ty {
            r[tx - x - 1][ty - y - 1] = m[x][y];
        }
    }
    r
}

pub fn flip_lr(m: &Matrix) -> Matrix {
    let (tx, ty) = dimensions(m);
    let mut r = vec![vec![0u8; ty]; tx];
    for x in 0..tx {
        for y in 0..ty {
            r[tx - x - 1][y] = m[x][y];
        }
    }
    r
}

pub fn flip_tb(m: &Matrix) -> Matrix {
    let (tx, ty) = dimensions(m);
    let mut r = vec![vec![0u8; ty]; tx];
    for x in 0..tx {
        for y in 0..ty {
            r[x][ty - y - 1] = m[x][y];
        }
    }
    r
}

pub fn flip_tlbr(m: &Matrix) -> Matrix {
    let (tx, ty) = dimensions(m);
    let mut r = vec![vec![0u8; tx]; ty];
    for x in 0..tx {
        for y in 0..ty {
            r[ty - y - 1][tx - x - 1] = m[x][y];
        }
    }
    r
}

pub fn flip_trbl(m: &Matrix) -> Matrix {
    let (tx, ty) = dimensions(m);
    let mut r = vec![vec![0u8; tx]; ty];
    for x in 0..tx {
        for y in 0..ty {
            r[y][x] = m[x][y];
        }
    }
    r
}

/// All 8 symmetry transformations
pub const TRANSFORM_NAMES: [&str; 8] = ["eq", "rCC", "rC", "r180", "fLR", "fTB", "fTLBR", "fTRBL"];

pub fn all_transforms(m: &Matrix) -> Vec<Matrix> {
    vec![
        equal(m),
        rotate_clockwise_90(m),
        rotate_counter_clockwise_90(m),
        rotate_180(m),
        flip_lr(m),
        flip_tb(m),
        flip_tlbr(m),
        flip_trbl(m),
    ]
}

/// Binary encoding: 2 bytes header (width, height) + 2 bits per cell
pub fn bin_code(m: &Matrix) -> Vec<u8> {
    let (tx, ty) = dimensions(m);
    let mut result = Vec::with_capacity(2 + (tx * ty + 3) / 4);
    result.push(tx as u8);
    result.push(ty as u8);

    let mut byte = 0u8;
    let mut pair = 0;
    for y in 0..ty {
        for x in 0..tx {
            byte = (byte << 2) | m[x][y];
            pair += 1;
            if pair == 4 {
                result.push(byte);
                byte = 0;
                pair = 0;
            }
        }
    }
    // Match PHP behavior: don't flush remaining bits when cells aren't divisible by 4
    // This is technically a bug in PHP but we replicate it for hash compatibility
    result
}

pub fn to_string(m: &Matrix) -> String {
    let (tx, ty) = dimensions(m);
    let mut s = String::new();
    for y in 0..ty {
        for x in 0..tx {
            s.push('\t');
            s.push_str(&m[x][y].to_string());
        }
        s.push('\n');
    }
    s
}
