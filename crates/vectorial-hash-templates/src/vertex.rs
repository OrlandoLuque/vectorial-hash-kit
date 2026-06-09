/// Segment between vertices: line (d=0) or arc (d=-1 clockwise, d=1 counter-clockwise)
#[derive(Clone, Debug)]
pub struct Segment {
    pub xc: f64,
    pub yc: f64,
    pub d: i8, // -1 = clockwise arc, 0 = line, 1 = counter-clockwise arc
}

impl Default for Segment {
    fn default() -> Self {
        Segment { xc: 0.0, yc: 0.0, d: 0 }
    }
}

/// A vertex in a polygon, stored in a Vec-based structure (not linked list)
#[derive(Clone, Debug)]
pub struct Vertex {
    pub x: f64,
    pub y: f64,
    pub seg: Segment, // segment data for the edge starting at this vertex
}

impl Vertex {
    pub fn new(x: f64, y: f64) -> Self {
        Vertex { x, y, seg: Segment::default() }
    }

    pub fn new_with_arc(x: f64, y: f64, xc: f64, yc: f64, d: i8) -> Self {
        Vertex { x, y, seg: Segment { xc, yc, d } }
    }

    pub fn roughly_equals(&self, other: &Vertex) -> bool {
        (self.x - other.x).abs() < 0.001 && (self.y - other.y).abs() < 0.001
    }

    pub fn equals(&self, other: &Vertex) -> bool {
        self.x == other.x && self.y == other.y
    }

    pub fn equals_xy(&self, x: f64, y: f64) -> bool {
        self.x == x && self.y == y
    }

    /// Check if this point is between two other points (on a line segment)
    pub fn is_inside(&self, a: &Vertex, b: &Vertex) -> bool {
        let min_x = a.x.min(b.x);
        let max_x = a.x.max(b.x);
        let min_y = a.y.min(b.y);
        let max_y = a.y.max(b.y);
        self.x >= min_x && self.x <= max_x && self.y >= min_y && self.y <= max_y
    }
}
