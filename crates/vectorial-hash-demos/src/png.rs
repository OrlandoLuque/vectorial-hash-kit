//! A minimal PNG writer — enough to save a screenshot, and nothing more.
//!
//! Why hand-rolled rather than the `image` crate: this exists so a headless run can *look at
//! itself*, and the demos crate already carries wgpu, winit and glam into a wasm build that
//! cares about size. A dependency whose only job is to serialise one RGBA buffer is not worth
//! the compile time or the wasm bytes, and the format's uncompressed path is about sixty lines.
//!
//! Deliberately not compressed: the deflate stream is written as **stored blocks**, so a
//! screenshot is roughly width×height×4 bytes. That is a few megabytes for a 1080p frame, which
//! is fine for something a developer looks at once and deletes, and it keeps the whole encoder
//! auditable — the alternative is a Huffman implementation nobody will read.

fn crc32(data: &[u8]) -> u32 {
    // Table-free: 8 shifts per byte is irrelevant next to writing the file.
    let mut c: u32 = 0xFFFF_FFFF;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let mut crc_in = Vec::with_capacity(4 + body.len());
    crc_in.extend_from_slice(kind);
    crc_in.extend_from_slice(body);
    out.extend_from_slice(&crc_in);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}

/// Encode tightly-packed RGBA8 rows (`w * h * 4` bytes) as a PNG.
pub fn encode_rgba(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), (w as usize) * (h as usize) * 4, "rgba buffer must be w*h*4");

    // Scanlines, each prefixed by filter type 0 (None). Filtering would shrink the file; it
    // would also mean implementing the filters, and this stream is not compressed anyway.
    let mut raw = Vec::with_capacity((h as usize) * (1 + (w as usize) * 4));
    for y in 0..h as usize {
        raw.push(0);
        let row = y * (w as usize) * 4;
        raw.extend_from_slice(&rgba[row..row + (w as usize) * 4]);
    }

    // zlib: 0x78 0x01 (deflate, 32K window, no preset dict), then stored blocks of <= 65535,
    // then the adler of the RAW data.
    let mut z = vec![0x78u8, 0x01];
    for (i, block) in raw.chunks(65535).enumerate() {
        let last = (i + 1) * 65535 >= raw.len();
        z.push(if last { 1 } else { 0 });                       // BFINAL, BTYPE=00 (stored)
        z.extend_from_slice(&(block.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = Vec::with_capacity(z.len() + 128);
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);                   // 8-bit, RGBA, deflate, no filter, no interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header a PNG decoder reads first, and the two checksums that make the rest legal.
    /// This cannot prove a viewer will open it, but every one of these being wrong is a way it
    /// would silently fail to.
    #[test]
    fn encodes_a_structurally_valid_png() {
        let (w, h) = (3u32, 2u32);
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i * 7) as u8).collect();
        let png = encode_rgba(w, h, &rgba);

        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10], "signature");
        // IHDR: length 13, then the tag, then w/h
        assert_eq!(&png[8..12], &13u32.to_be_bytes());
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..20], &w.to_be_bytes());
        assert_eq!(&png[20..24], &h.to_be_bytes());
        assert_eq!(png[24], 8, "bit depth");
        assert_eq!(png[25], 6, "colour type RGBA");
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");

        // Every chunk's CRC must check out — walk them the way a decoder does.
        let mut i = 8;
        let mut seen = 0;
        while i + 8 <= png.len() {
            let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
            let body_end = i + 8 + len;
            let crc = u32::from_be_bytes(png[body_end..body_end + 4].try_into().unwrap());
            assert_eq!(crc32(&png[i + 4..body_end]), crc, "chunk crc at byte {i}");
            i = body_end + 4;
            seen += 1;
        }
        assert_eq!(seen, 3, "IHDR, IDAT, IEND");
        assert_eq!(i, png.len(), "no trailing bytes");
    }

    /// The stored-block framing is the part most likely to be wrong: LEN and ~LEN must agree,
    /// exactly one block may be final, and the adler must cover the raw scanlines.
    #[test]
    fn the_deflate_stream_is_well_formed_across_a_block_boundary() {
        // Wide enough that the scanlines exceed one 65 535-byte stored block.
        let (w, h) = (5000u32, 4u32);
        let rgba = vec![0xABu8; (w * h * 4) as usize];
        let png = encode_rgba(w, h, &rgba);

        // pull IDAT out
        let mut i = 8;
        let mut idat: Vec<u8> = Vec::new();
        while i + 8 <= png.len() {
            let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
            if &png[i + 4..i + 8] == b"IDAT" { idat = png[i + 8..i + 8 + len].to_vec(); }
            i = i + 8 + len + 4;
        }
        assert!(!idat.is_empty(), "IDAT must exist");
        assert_eq!(&idat[..2], &[0x78, 0x01], "zlib header");

        let mut p = 2;
        let mut finals = 0;
        let mut raw_len = 0usize;
        loop {
            let bfinal = idat[p] & 1;
            assert_eq!(idat[p] >> 1, 0, "BTYPE must be 00 (stored)");
            let len = u16::from_le_bytes(idat[p + 1..p + 3].try_into().unwrap());
            let nlen = u16::from_le_bytes(idat[p + 3..p + 5].try_into().unwrap());
            assert_eq!(nlen, !len, "NLEN must be the ones-complement of LEN");
            raw_len += len as usize;
            p += 5 + len as usize;
            if bfinal == 1 { finals += 1; break; }
        }
        assert_eq!(finals, 1, "exactly one final block");
        assert!(raw_len > 65535, "this case must actually cross a block boundary ({raw_len})");
        assert_eq!(raw_len, (h as usize) * (1 + (w as usize) * 4), "every scanline, each with its filter byte");
        assert_eq!(p + 4, idat.len(), "adler32 is the last four bytes");
    }
}
