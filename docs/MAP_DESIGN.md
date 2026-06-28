# Siege map — design + look (research notes)

Condensed from a web research pass (full sources at the bottom). The goal is a
battlefield that is both **designed well** (rivers as choke-points, bridges,
biomes shaping the fight) and **gorgeous** on macroquad's WebGL1 / ES 1.00 stack
(and reusable on the wgpu road). Voxel terrain (alterable) is the chosen base.

> **Status (2026-06-28).** Moves 2–4 are **landed** in `siege.rs`
> (`build_voxel_chunks`): blocky flat-top cells + cliff walls, **baked corner AO
> with the quad-flip fix** (move 2), the **height colour ramp** (move 3), chunked
> under the drawcall cap. Rivers + bridges are in (carved channel, wade speed,
> bridge decks). **Still research-only:** true greedy meshing (move 1 — current
> mesher is per-cell culled faces, not merged quads), runtime **alteration**
> (crater carving — the cells are already a grid, so it's a remesh-dirty-chunks
> step), and the slope ramp / translucent water plane. See BACKLOG "Siege".

## The 5 highest-leverage moves (priority order)

1. **Blocky voxels (culled → greedy-meshed cubes), not smooth.** On WebGL1 with
   no compute shaders, a CPU greedy mesher per chunk is the pragmatic, pretty,
   trivially-destructible choice. Smooth (marching cubes / dual contouring) buys
   Astroneer craters but costs 3-5× the mesher + real normals + harder AO.
   Decision: **blocky terrain, a smooth translucent water plane on top.**
2. **Bake vertex Ambient Occlusion into the mesh** — the single biggest
   visual-per-line win (this is Minecraft "smooth lighting" / Townscaper solidity).
3. **Colour by height + slope via two colour ramps, baked into vertex colours**
   (no textures — perfect for WebGL1). Flat-shaded faces + a height gradient = the
   stylized low-poly look.
4. **Chunk the world (16³, 1-voxel padding) and re-mesh only dirty chunks** on
   edit → runtime carving is cheap.
5. **Carve craters by clearing voxels in a sphere, then re-mesh touched chunks.**
   Wire cannon impacts + volcano bombs to it.

## Meshing

- **Greedy meshing** (0fps): reduce 3D → 2D, sweep each axis in 2 directions (6
  passes), build a per-slice mask, merge equal-attribute cells into maximal
  rectangles. Quad count for an 8³ solid: naive 3072 → culled 384 → **greedy 6**.
  Two faces merge only if they share material **AND** AO value.
- **Chunk layout** (block-mesh-rs): 16³ chunk in an **18³ padded array** (1-voxel
  border from neighbours) so the mesher reads face neighbours + AO corners with no
  cross-chunk branching. Dense `Vec<u8>` palette indices per chunk; sparse
  `HashMap<IVec3, Chunk>` (or a flat Vec for a fixed map). Re-mesh dirty chunks
  only (sub-ms for 16³; rayon batches them — native only, wasm stays serial).

## Vertex AO (do this first)

For each of a face's 4 vertices, look at 3 neighbour voxels in the plane above the
face — the two edge-adjacent (`side1`,`side2`) + the diagonal `corner` (1 if solid):

```
fn vertex_ao(side1, side2, corner) -> u8 {
    if side1 == 1 && side2 == 1 { 0 }       // boxed-in corner = darkest
    else { 3 - (side1 + side2 + corner) }   // 4 levels 0..3
}
```

Map levels → brightness `[0.45, 0.65, 0.85, 1.0]`, multiplied into the vertex
colour. **Anisotropy fix (must-have):** a quad's 4 AO corners differ across its
two triangles; **flip the triangulation** when `a00 + a11 > a01 + a10` (else ugly
diagonal seams — this is the "looks broken vs looks AAA" line).

## Colour ramps (no textures)

Per vertex: normalized **height** + **slope** (`up·normal`); sample a height ramp
(sand→grass→rock→snow) and a slope ramp (grass-on-flat vs rock-on-cliff); blend by
a height-keyed curve (cliffs always rocky, snow = high + flat, beach near water).
Final ≈ `ramp × ao × (ambient + N·L sun)`. Flat shading + the gradient restores the
richness flat shading loses.

- **Water/rivers:** one translucent flat plane at sea level (bluish, ~0.6 alpha),
  optional shimmer. Reads great against blocky banks.
- **Lava:** same but **emissive** (bright orange, don't multiply by N·L; add a
  glow), dark crust ramp, vent brightest.

## Designing the map (not just looks)

- **Domain warping** for organic shapes: `noise(p + k·noise(p + offset))`, two
  levels → meandering coastlines/ridges. Apply to the base heightfield first.
- **Worley/Voronoi** for biome cells + rocky cracking + ridges (F1 distance).
  **Poisson-disk** for even-but-random feature placement (trees, rocks, spawn
  jitter, bridge candidates) with a minimum spacing.
- **Hydraulic erosion → rivers (particle hydrology, Nick McDonald):** drop water
  particles `{pos, speed, volume, sediment}`; each descends the gradient, **erodes**
  on steep slopes (capacity ∝ speed·slope·water), **deposits** when flat. A
  time-averaged **stream map** `stream = 0.99·stream + 0.01·visited` reveals where
  water keeps flowing = **the river** (carve those deeper, lay the water plane).
  Pools flood-fill to lakes that overflow + drain.
- **Rivers + bridges as choke-points (the gameplay payoff):** rivers = high-stream
  cells. Place **bridges** where (a) stream is high, (b) the banks are close +
  similar height, (c) it lies on the castle→castle path. Poisson-sample crossing
  candidates, score by narrowness + path relevance, place the best few. That gives
  forests/rivers/bridges as choke-points + LoS cover *by construction*.

## Performance on WebGL1 / ES 1.00

- No compute, no `gl_VertexID`, instancing only via `ANGLE_instanced_arrays`
  (which miniquad uses). **Terrain = CPU-greedy-meshed vertex buffers per chunk**
  (NOT instanced cubes — instancing can't carry baked AO / merged faces). Re-upload
  only dirty chunks. **Instancing is for units/props** (thousands of soldiers,
  trees, arrows).
- **Frustum cull at chunk granularity.** LOD likely unnecessary for a bounded
  battlefield; if needed, Transvoxel (smooth) or larger far chunks (blocky).
- A 256×64×256 world is comfortable; greedy meshing makes interior voxels free.

## Best references

- Greedy meshing: <https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/> (+ part 2)
- Vertex AO (formula + flip): <https://0fps.net/2013/07/03/ambient-occlusion-for-minecraft-like-worlds/> · <https://playspacefarer.com/ambient-occlusion/>
- Rust mesher to mirror: <https://github.com/bonsairobo/block-mesh-rs>
- Rivers/lakes/erosion: <https://nickmcd.me/2020/04/15/procedural-hydrology/>
- Low-poly colour ramps: <https://www.pinwheelstud.io/post/how-to-color-low-poly-terrain-with-gradients-and-curves>
- Smooth/LOD if ever needed: <https://transvoxel.org/> · voxel.wiki: <https://voxel.wiki/wiki/surface-extraction/>
