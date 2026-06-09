//! 2D geometry primitives used by the spatial tree.

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Axis-aligned rectangle. Used for cell bounds and shape bounding boxes.
///
/// Half-open in both axes: `[x, x + width)` × `[y, y + height)`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub const fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }

    pub fn x_max(&self) -> f64 { self.x + self.width }
    pub fn y_max(&self) -> f64 { self.y + self.height }

    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.x && p.x < self.x_max() && p.y >= self.y && p.y < self.y_max()
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        !(other.x >= self.x_max()
            || other.x_max() <= self.x
            || other.y >= self.y_max()
            || other.y_max() <= self.y)
    }
}
