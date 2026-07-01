//! Little-endian byte IO shared by every structure's `serialize` /
//! `deserialize`. Dependency-free (no `serde` crate) so the format is a fixed,
//! documented byte layout the caller drives with a per-item read/write closure.
//!
//! Each structure writes a 4-byte magic + 1-byte version, its build params, then
//! its arena (nodes / cells) exactly — a load rebuilds the *same* tree without
//! re-inserting, so `ItemRef` handles survive the round trip. See
//! [`crate::Tree3::serialize`] for the reference implementation these mirror.

use crate::geom::Rect;
use crate::tree3::Aabb;
use std::io::{self, Read, Write};

pub(crate) fn corrupt(msg: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, msg) }

pub(crate) fn w_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> { w.write_all(&[v]) }
pub(crate) fn w_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
pub(crate) fn w_i32<W: Write>(w: &mut W, v: i32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
pub(crate) fn w_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
pub(crate) fn w_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }

pub(crate) fn r_u8<R: Read>(r: &mut R) -> io::Result<u8> { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0]) }
pub(crate) fn r_u32<R: Read>(r: &mut R) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
pub(crate) fn r_i32<R: Read>(r: &mut R) -> io::Result<i32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(i32::from_le_bytes(b)) }
pub(crate) fn r_u64<R: Read>(r: &mut R) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
pub(crate) fn r_f64<R: Read>(r: &mut R) -> io::Result<f64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(f64::from_le_bytes(b)) }

pub(crate) fn w_aabb<W: Write>(w: &mut W, b: &Aabb) -> io::Result<()> {
    w_f64(w, b.x)?; w_f64(w, b.y)?; w_f64(w, b.z)?;
    w_f64(w, b.w)?; w_f64(w, b.h)?; w_f64(w, b.d)
}
pub(crate) fn r_aabb<R: Read>(r: &mut R) -> io::Result<Aabb> {
    Ok(Aabb::new(r_f64(r)?, r_f64(r)?, r_f64(r)?, r_f64(r)?, r_f64(r)?, r_f64(r)?))
}

pub(crate) fn w_rect<W: Write>(w: &mut W, b: &Rect) -> io::Result<()> {
    w_f64(w, b.x)?; w_f64(w, b.y)?; w_f64(w, b.width)?; w_f64(w, b.height)
}
pub(crate) fn r_rect<R: Read>(r: &mut R) -> io::Result<Rect> {
    Ok(Rect::new(r_f64(r)?, r_f64(r)?, r_f64(r)?, r_f64(r)?))
}
