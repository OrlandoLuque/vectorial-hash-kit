//! Cold-store prototype on **real on-disk storage** (redb, a pure-Rust embedded
//! B-tree KV) — the missing "cold / persistence index" layer, done for real.
//!
//! Design (the recommendation from the cold-index analysis): a **sorted
//! space-filling-curve key over a B-tree**. Here the key is a coarse Morton
//! CELL code (big-endian bytes so lexicographic = Z-order), the value a blob of
//! the cell's objects. A spatial AoI query = **enumerate the box's cells and
//! probe each** (NOT a naive `[min..max]` range scan — that's the pathological
//! trap the in-memory bench showed). This is what a persistent, unbounded,
//! streaming universe index looks like; measured against the in-memory grid.
//!
//! Measures: write throughput to disk, on-disk file size, and AoI query latency
//! COLD (fresh open, page cache empty-ish) vs WARM (cache hot).
//!
//! ```bash
//! cargo run -p vectorial-hash --example cold_store_redb --release
//! ```

use std::time::Instant;
use redb::{Database, TableDefinition};
use vectorial_hash::{Point3, Shape3, Sphere3};

const WORLD: f64 = 10_000.0;
const LV: u32 = 5; // coarse cell level: 32 cells/axis, cell ≈ 312 wu
const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("cold_objects");

fn morton3(x: u32, y: u32, z: u32) -> u64 {
    fn split(mut v: u64) -> u64 { v &= 0x1f_ffff; v = (v | v << 32) & 0x1f00000000ffff; v = (v | v << 16) & 0x1f0000ff0000ff; v = (v | v << 8) & 0x100f00f00f00f00f; v = (v | v << 4) & 0x10c30c30c30c30c3; v = (v | v << 2) & 0x1249249249249249; v }
    split(x as u64) | (split(y as u64) << 1) | (split(z as u64) << 2)
}
fn ccell(v: f64) -> u32 { ((v / WORLD) * (1u32 << LV) as f64) as u32 & ((1 << LV) - 1) }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let path = std::env::temp_dir().join("vh_cold_store.redb");
    let _ = std::fs::remove_file(&path);
    println!("redb cold-store prototype | {n} objects | world {WORLD:.0}^3 | cell level {LV}\n");

    // ---- generate + group by cell (a per-cell blob = the value)
    let mut r = Rng(42);
    let pts: Vec<(u32, Point3)> = (0..n).map(|i| (i as u32, Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD))).collect();
    let mut by_cell: std::collections::HashMap<u64, Vec<(u32, Point3)>> = std::collections::HashMap::new();
    for &(id, p) in &pts { by_cell.entry(morton3(ccell(p.x), ccell(p.y), ccell(p.z))).or_default().push((id, p)); }
    let cells = by_cell.len();

    // ---- WRITE to disk (one transaction, one blob per cell)
    let t = Instant::now();
    {
        let db = Database::create(&path)?;
        let wtx = db.begin_write()?;
        {
            let mut table = wtx.open_table(TABLE)?;
            let mut buf: Vec<u8> = Vec::new();
            for (&code, objs) in &by_cell {
                buf.clear();
                for &(id, p) in objs {
                    buf.extend_from_slice(&id.to_le_bytes());
                    buf.extend_from_slice(&p.x.to_le_bytes());
                    buf.extend_from_slice(&p.y.to_le_bytes());
                    buf.extend_from_slice(&p.z.to_le_bytes());
                }
                table.insert(code, buf.as_slice())?;
            }
        }
        wtx.commit()?;
    }
    let wsecs = t.elapsed().as_secs_f64();
    let size_mb = std::fs::metadata(&path)?.len() as f64 / 1e6;
    println!("WRITE: {:.2} s  ({:.1} M objects/s, {cells} cells)   file = {:.1} MB on disk", wsecs, n as f64 / wsecs / 1e6, size_mb);

    // ---- QUERY: enumerate box cells, probe each, sphere-filter. Cold = fresh
    // DB open (page cache not primed by our writes); warm = repeated.
    let bubble = 500.0f64;
    let mut rq = Rng(99);
    let queries: Vec<(f64, f64, f64)> = (0..1000).map(|_| (rq.unit() * WORLD, rq.unit() * WORLD, rq.unit() * WORLD)).collect();
    let run = |db: &Database| -> (usize, f64) {
        let rtx = db.begin_read().unwrap();
        let table = rtx.open_table(TABLE).unwrap();
        let t = Instant::now();
        let mut hits = 0usize;
        for &c in &queries {
            let (x0, x1, y0, y1, z0, z1) = (ccell(c.0 - bubble), ccell(c.0 + bubble), ccell(c.1 - bubble), ccell(c.1 + bubble), ccell(c.2 - bubble), ccell(c.2 + bubble));
            let sph = Sphere3::new(c.0, c.1, c.2, bubble);
            for iz in z0..=z1 { for iy in y0..=y1 { for ix in x0..=x1 {
                if let Some(v) = table.get(morton3(ix, iy, iz)).unwrap() {
                    let b = v.value();
                    let mut off = 0;
                    while off + 28 <= b.len() {
                        let px = f64::from_le_bytes(b[off + 4..off + 12].try_into().unwrap());
                        let py = f64::from_le_bytes(b[off + 12..off + 20].try_into().unwrap());
                        let pz = f64::from_le_bytes(b[off + 20..off + 28].try_into().unwrap());
                        if sph.contains_point(Point3::new(px, py, pz)) { hits += 1; }
                        off += 28;
                    }
                }
            }}}
        }
        (hits, t.elapsed().as_secs_f64())
    };
    // COLD: open a fresh handle (redb re-reads pages from disk on first touch).
    let db_cold = Database::open(&path)?;
    let (h0, cold) = run(&db_cold);
    // WARM: same handle, cache primed.
    let mut warm = f64::MAX;
    for _ in 0..5 { let (_, s) = run(&db_cold); warm = warm.min(s); }
    println!("QUERY (1000 AoI bubbles, {} avg hits):", h0 / queries.len());
    println!("   cold (fresh open): {:>8.1} µs/query", cold / queries.len() as f64 * 1e6);
    println!("   warm (cache hot):  {:>8.1} µs/query", warm / queries.len() as f64 * 1e6);
    println!("\nfor comparison (in-memory, same query): BTree cell-probe ≈ 5.8 µs, MortonGrid3 ≈ 8.8 µs (see cold_index_bench).");

    let _ = std::fs::remove_file(&path);
    println!("\ndone. (temp db removed)");
    Ok(())
}
