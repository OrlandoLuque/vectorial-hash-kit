//! Headless screenshots for the wgpu demos — so a session can look at its own geometry.
//!
//! Every visual question this project has asked ("which way does the slime face?", "does the wall
//! actually meet the tower?", "is the building scale right?") has ended the same way: queued until
//! a human was in front of the window. That is a slow loop for questions that are simply about
//! *correctness*, and it does not scale to eight demos.
//!
//! This is the reusable half of what `horde_wgpu` grew first. A demo adopts it in three places:
//!
//! ```ignore
//! // 1. state:  shot: Option<Shot>,          Shot::from_env("SIEGE")
//! // 2. render: let target = Target::begin(&device, &config, &surface, &mut self.shot);
//! //            let view = target.view();            // render into this as usual
//! // 3. end:    target.finish(&device, &queue, encoder);   // presents, or writes the PNG and exits
//! ```
//!
//! **Why offscreen rather than grabbing the swapchain.** A shot renders into its own
//! `COPY_SRC` texture, so it does not depend on the surface format allowing copies and behaves
//! identically whether or not a window is up — including on a machine with no display at all.
//!
//! **Two traps, both of which produce a picture that looks *almost* right** rather than an obvious
//! failure, which is why they are handled here once instead of in each demo:
//!
//! - `copy_texture_to_buffer` requires rows padded to `COPY_BYTES_PER_ROW_ALIGNMENT` (256). Read
//!   it unpadded and the image shears progressively down the frame.
//! - The Windows swapchain is BGRA and PNG is RGBA. Skip the swizzle and you get a perfectly
//!   believable frame with an orange sky.
//!
//! The module compiles everywhere so demos need no `cfg` of their own; only the *saving* is
//! native, because [`crate::png`] is not in the web build and a browser has no file to write to.
//! On the web `Shot::from_env` always returns `None` (wasm has no environment), so the offscreen
//! path is unreachable rather than merely unused.

/// A pending screenshot: where to write it, and how many frames to simulate first.
pub struct Shot {
    path: String,
    /// Frames still to run before the shot. A demo usually needs a few dozen so the world has
    /// settled — a screenshot of frame 0 is a picture of the initial conditions, not the sim.
    after: u32,
}

impl Shot {
    /// Read `$<PREFIX>_SHOT` (the output path) and `$<PREFIX>_SHOT_AFTER` (frames to wait,
    /// default 90). Returns `None` when the demo is running normally.
    ///
    /// `$SHOT_W` (not prefixed — it applies to whichever demo is shooting) downscales the result
    /// before writing, for a web thumbnail: the encoder stores uncompressed, so a full window is
    /// megabytes.
    pub fn from_env(prefix: &str) -> Option<Shot> {
        let path = std::env::var(format!("{prefix}_SHOT")).ok()?;
        let after = std::env::var(format!("{prefix}_SHOT_AFTER")).ok()
            .and_then(|s| s.parse().ok()).unwrap_or(90);
        Some(Shot { path, after })
    }
}

/// Where this frame is going: the swapchain as usual, or an offscreen texture to be saved.
pub enum Target {
    Screen(wgpu::SurfaceTexture, wgpu::TextureView),
    Offscreen { tex: wgpu::Texture, view: wgpu::TextureView, path: String, format: wgpu::TextureFormat, size: (u32, u32) },
    /// No surface and no shot — a headless frame with nothing to draw into.
    None,
}

impl Target {
    /// Decide this frame's target, counting down any pending shot. Call once per frame, before
    /// creating the encoder.
    pub fn begin(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, surface: Option<&wgpu::Surface<'static>>, shot: &mut Option<Shot>) -> Target {
        let due = matches!(shot.as_ref(), Some(s) if s.after == 0);
        if let Some(s) = shot.as_mut() { if s.after > 0 { s.after -= 1; } }

        if due {
            let (width, height) = (config.width, config.height);
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shot"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let path = shot.as_ref().map(|s| s.path.clone()).unwrap_or_default();
            return Target::Offscreen { tex, view, path, format: config.format, size: (width, height) };
        }
        match surface.map(|s| s.get_current_texture()) {
            Some(Ok(frame)) => {
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                Target::Screen(frame, view)
            }
            _ => Target::None,
        }
    }

    /// The view to render into. `None` means skip this frame entirely.
    pub fn view(&self) -> Option<&wgpu::TextureView> {
        match self {
            Target::Screen(_, v) => Some(v),
            Target::Offscreen { view, .. } => Some(view),
            Target::None => None,
        }
    }

    /// Submit, and either present or save. **Saving exits the process** — a shot run is a shot
    /// run, and continuing would render a second frame nobody asked for.
    pub fn finish(self, device: &wgpu::Device, queue: &wgpu::Queue, mut enc: wgpu::CommandEncoder) {
        match self {
            Target::None => { queue.submit(Some(enc.finish())); }
            Target::Screen(frame, _) => { queue.submit(Some(enc.finish())); frame.present(); }
            Target::Offscreen { tex, path, format, size: (w, h), .. } => {
                let unpadded = w * 4;
                let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("shot-readback"), size: (padded * h) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                enc.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout {
                        offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) } },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
                queue.submit(Some(enc.finish()));

                #[cfg(feature = "web-wgpu")]
                { let _ = (buf, unpadded, path, format); }
                #[cfg(not(feature = "web-wgpu"))]
                {
                let slice = buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
                device.poll(wgpu::Maintain::Wait);
                match rx.recv() {
                    Ok(Ok(())) => {
                        let data = slice.get_mapped_range();
                        let bgra = matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
                        let mut rgba = Vec::with_capacity((unpadded * h) as usize);
                        for y in 0..h as usize {
                            let row = &data[y * padded as usize..y * padded as usize + unpadded as usize];
                            for px in row.chunks_exact(4) {
                                if bgra { rgba.extend_from_slice(&[px[2], px[1], px[0], 255]); }
                                else { rgba.extend_from_slice(&[px[0], px[1], px[2], 255]); }
                            }
                        }
                        drop(data);
                        buf.unmap();
                        // `$<PREFIX>_SHOT_W` downscales before writing. The PNG encoder here
                        // stores raw deflate blocks (no compression — the right trade for a frame
                        // a developer looks at once), so a full-window shot is megabytes and
                        // unusable as a web thumbnail. Nearest-neighbour is enough: this is a
                        // thumbnail, and a box filter would only make dark UI text mushier.
                        let (w, h, rgba) = match std::env::var("SHOT_W").ok().and_then(|s| s.parse::<u32>().ok()) {
                            Some(tw) if tw > 0 && tw < w => {
                                let th = (h as f64 * tw as f64 / w as f64).round().max(1.0) as u32;
                                let mut out = Vec::with_capacity((tw * th * 4) as usize);
                                for y in 0..th {
                                    let sy = (y as u64 * h as u64 / th as u64) as usize;
                                    for x in 0..tw {
                                        let sx = (x as u64 * w as u64 / tw as u64) as usize;
                                        let i = (sy * w as usize + sx) * 4;
                                        out.extend_from_slice(&rgba[i..i + 4]);
                                    }
                                }
                                (tw, th, out)
                            }
                            _ => (w, h, rgba),
                        };
                        match std::fs::write(&path, crate::png::encode_rgba(w, h, &rgba)) {
                            Ok(()) => println!("shot -> {path} ({w}x{h})"),
                            Err(e) => eprintln!("shot: cannot write {path}: {e}"),
                        }
                    }
                    other => eprintln!("shot: readback failed ({other:?})"),
                }
                std::process::exit(0);
                }
            }
        }
    }
}
