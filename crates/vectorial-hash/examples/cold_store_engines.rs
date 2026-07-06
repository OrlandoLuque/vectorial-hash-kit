//! Cold-store engine comparison — B-tree (redb) vs LSM (fjall) for the on-disk
//! sorted-Morton-key cold index. redb is a copy-on-write **B-tree**
//! (read-optimized, in-place); fjall is an **LSM tree** (write-optimized:
//! buffer + flush sorted segments + background compaction). Same data, same
//! cell-blob layout, keyed by big-endian Morton cell code (lexicographic =
//! Z-order). We measure the write path (throughput + on-disk size) and the AoI
//! query (cell-probe range) — the trade-off is write vs read.
//!
//! ```bash
//! cargo run -p vectorial-hash --example cold_store_engines --release
//! ```

use std::time::Instant;
use vectorial_hash::{Point3, Shape3, Sphere3};

const WORLD: f64 = 10_000.0;
const LV: u32 = 5;

fn morton3(x: u32, y: u32, z: u32) -> u64 {
    fn split(mut v: u64) -> u64 { v &= 0x1f_ffff; v = (v | v << 32) & 0x1f00000000ffff; v = (v | v << 16) & 0x1f0000ff0000ff; v = (v | v << 8) & 0x100f00f00f00f00f; v = (v | v << 4) & 0x10c30c30c30c30c3; v = (v | v << 2) & 0x1249249249249249; v }
    split(x as u64) | (split(y as u64) << 1) | (split(z as u64) << 2)
}
fn ccell(v: f64) -> u32 { ((v / WORLD) * (1u32 << LV) as f64) as u32 & ((1 << LV) - 1) }

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 } }

fn dir_size(p: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(p) { for e in rd.flatten() { let m = e.metadata().unwrap(); total += if m.is_dir() { dir_size(&e.path()) } else { m.len() }; } }
    total
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    println!("cold-store engine comparison | {n} objects | world {WORLD:.0}^3 | cell level {LV}\n");

    // group objects by cell → one blob per cell (28 B/object)
    let mut r = Rng(42);
    let mut by_cell: std::collections::HashMap<u64, Vec<u8>> = std::collections::HashMap::new();
    for i in 0..n {
        let p = Point3::new(r.unit() * WORLD, r.unit() * WORLD, r.unit() * WORLD);
        let b = by_cell.entry(morton3(ccell(p.x), ccell(p.y), ccell(p.z))).or_default();
        b.extend_from_slice(&(i as u32).to_le_bytes());
        b.extend_from_slice(&p.x.to_le_bytes()); b.extend_from_slice(&p.y.to_le_bytes()); b.extend_from_slice(&p.z.to_le_bytes());
    }
    let mut cells: Vec<(u64, Vec<u8>)> = by_cell.into_iter().collect();
    cells.sort_by_key(|c| c.0); // write in sorted key order (fair to both engines)
    let ncells = cells.len();

    let bubble = 500.0f64;
    let mut rq = Rng(99);
    let queries: Vec<(f64, f64, f64)> = (0..1000).map(|_| (rq.unit() * WORLD, rq.unit() * WORLD, rq.unit() * WORLD)).collect();
    let count_hits = |blob: &[u8], sph: &Sphere3| -> usize {
        let mut hits = 0; let mut off = 0;
        while off + 28 <= blob.len() {
            let px = f64::from_le_bytes(blob[off + 4..off + 12].try_into().unwrap());
            let py = f64::from_le_bytes(blob[off + 12..off + 20].try_into().unwrap());
            let pz = f64::from_le_bytes(blob[off + 20..off + 28].try_into().unwrap());
            if sph.contains_point(Point3::new(px, py, pz)) { hits += 1; }
            off += 28;
        }
        hits
    };

    println!("{:>10} | {:>18} | {:>10} | {:>16} {:>16}", "engine", "write (M obj/s)", "size MB", "query cold µs", "query warm µs");

    // ---------- redb (B-tree) ----------
    {
        use redb::{Database, TableDefinition};
        const T: TableDefinition<u64, &[u8]> = TableDefinition::new("c");
        let path = std::env::temp_dir().join("vh_engine_redb.redb");
        let _ = std::fs::remove_file(&path);
        let t = Instant::now();
        { let db = Database::create(&path)?; let wtx = db.begin_write()?; { let mut tab = wtx.open_table(T)?; for (k, v) in &cells { tab.insert(*k, v.as_slice())?; } } wtx.commit()?; }
        let w = n as f64 / t.elapsed().as_secs_f64() / 1e6;
        let size = std::fs::metadata(&path)?.len() as f64 / 1e6;
        let run = |db: &Database| -> (usize, f64) {
            let rtx = db.begin_read().unwrap(); let tab = rtx.open_table(T).unwrap();
            let t = Instant::now(); let mut hits = 0;
            for &c in &queries { let (x0,x1,y0,y1,z0,z1)=(ccell(c.0-bubble),ccell(c.0+bubble),ccell(c.1-bubble),ccell(c.1+bubble),ccell(c.2-bubble),ccell(c.2+bubble)); let s=Sphere3::new(c.0,c.1,c.2,bubble);
                for iz in z0..=z1 { for iy in y0..=y1 { for ix in x0..=x1 { if let Some(v)=tab.get(morton3(ix,iy,iz)).unwrap() { hits += count_hits(v.value(), &s); } }}} }
            (hits, t.elapsed().as_secs_f64())
        };
        let dbc = Database::open(&path)?; let (_, cold) = run(&dbc);
        let mut warm = f64::MAX; for _ in 0..5 { warm = warm.min(run(&dbc).1); }
        println!("{:>10} | {:>18.1} | {:>10.1} | {:>16.1} {:>16.1}", "redb (B)", w, size, cold / queries.len() as f64 * 1e6, warm / queries.len() as f64 * 1e6);
        let _ = std::fs::remove_file(&path);
    }

    // ---------- fjall (LSM) ----------
    {
        use fjall::{Config, PartitionCreateOptions, PersistMode};
        let path = std::env::temp_dir().join("vh_engine_fjall");
        let _ = std::fs::remove_dir_all(&path);
        let t = Instant::now();
        let ks = Config::new(&path).open()?;
        let part = ks.open_partition("c", PartitionCreateOptions::default())?;
        for (k, v) in &cells { part.insert(k.to_be_bytes(), v.as_slice())?; }
        ks.persist(PersistMode::SyncAll)?;
        let w = n as f64 / t.elapsed().as_secs_f64() / 1e6;
        let size = dir_size(&path) as f64 / 1e6;
        // fresh reopen for cold
        drop(part); drop(ks);
        let ks = Config::new(&path).open()?;
        let part = ks.open_partition("c", PartitionCreateOptions::default())?;
        let run2 = |part: &fjall::PartitionHandle| -> (usize, f64) {
            let t = Instant::now(); let mut hits = 0;
            for &c in &queries { let (x0,x1,y0,y1,z0,z1)=(ccell(c.0-bubble),ccell(c.0+bubble),ccell(c.1-bubble),ccell(c.1+bubble),ccell(c.2-bubble),ccell(c.2+bubble)); let s=Sphere3::new(c.0,c.1,c.2,bubble);
                for iz in z0..=z1 { for iy in y0..=y1 { for ix in x0..=x1 { if let Some(v)=part.get(morton3(ix,iy,iz).to_be_bytes()).unwrap() { hits += count_hits(&v, &s); } }}} }
            (hits, t.elapsed().as_secs_f64())
        };
        let (_, cold) = run2(&part);
        let mut warm = f64::MAX; for _ in 0..5 { warm = warm.min(run2(&part).1); }
        println!("{:>10} | {:>18.1} | {:>10.1} | {:>16.1} {:>16.1}", "fjall (LSM)", w, size, cold / queries.len() as f64 * 1e6, warm / queries.len() as f64 * 1e6);
        drop(part); drop(ks);
        let _ = std::fs::remove_dir_all(&path);
    }

    println!("\n({ncells} cells written in sorted key order.)\nreading: B-tree (redb) favours reads/space; LSM (fjall) favours write throughput\n(sequential flush + compaction) at some read/space amplification. Pick by the\ncold store's real write:read ratio.");
    Ok(())
}
