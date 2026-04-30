# Houdini-side workflow for Gaussian Splat Fracture rendering

The simulation pipeline writes a Houdini-readable per-frame point
cloud (`.geo.gz` JSON) under `<prompt_dir>/houdini_export/`.  This
folder is the bridge between the Python simulator and Houdini's
Karma path-traced renderer.

## Per-point attributes written

| name          | size | semantics                                  |
|---------------|------|--------------------------------------------|
| `P`           |  3   | world-space position                       |
| `Cd`          |  3   | base RGB (sigmoided DC SH * material LUT)  |
| `Alpha`       |  1   | sigmoided opacity                          |
| `scale`       |  3   | per-axis Gaussian sigmas                   |
| `pscale`      |  1   | mean sigma fallback                        |
| `orient`      |  4   | Houdini quaternion `(i, j, k, s)`          |
| `fragment_id` |  1   | physical fragment label (int)              |
| `damage`      |  1   | AT2 c field value                          |
| `N`           |  3   | crack normal (zero where undefined)        |

## Quick path: instance ellipsoids and Karma path-trace

1. **Open Houdini** (any version with Copy-to-Points 2.0).
2. **Run the scene builder** in *Windows -> Python Source Editor*::

       exec(open("/home/chayo/Desktop/gaussian_phase_field/houdini/build_gpf_scene.py").read())
       node = build_gpf_scene(
           "/home/chayo/Desktop/gaussian_phase_field/output/at2_sentence_50k_long_v3/"
           "same_object_same_impact_different_sentence/"
           "03_soda_lime_glass_object_shattering_into_many_sharp_radial_cracks/houdini_export"
       )

   This creates `/obj/gpf_03_..../` with:
   - `file_in` (File SOP) -- pointed at one frame initially.
   - `attribwrangle_prep` -- VEX cleanup (orient normalize, alpha cull).
   - `ellipsoid_template` (low-poly sphere).
   - `copy_to_points` -- per-point ellipsoid instancing using
     `scale`, `pscale`, `orient`.
   - `OUT_render` -- the SOP output.

3. **Sequence playback**: open `file_in`'s **File** parameter and
   replace the literal frame path with a `$F`-templated pattern, e.g.

       /home/chayo/.../houdini_export/frame_$F4_impact_$F4.geo.gz

   (Or use a `Python SOP` to pick the file by clamped index since
   `loop_frame` and `impact+` are coupled in our naming.)

4. **Material**: assign Principled Shader via `/mat/`.  Recommended
   per-material starting points (Houdini PBR):

   | family            | base color   | metallic | roughness | transmission | IOR |
   |-------------------|--------------|---------:|----------:|-------------:|----:|
   | sharp_brittle (glass) | `Cd`     | 0.0      | 0.05      | 0.95         | 1.5 |
   | sharp_brittle (ice)   | `Cd`     | 0.0      | 0.10      | 0.85         | 1.31|
   | brittle_moderate (ceramic) | `Cd`| 0.0      | 0.55      | 0.0          | 1.5 |
   | rough_quasi_brittle (concrete) | `Cd` | 0.0 | 0.85    | 0.0          | 1.5 |
   | neutral (steel)   | `(0.7,0.7,0.7)` | 1.0  | 0.30      | 0.0          | 2.5 |
   | diffuse (rubber)  | `(0.1,0.1,0.1)` | 0.0  | 0.85      | 0.0          | 1.5 |

   Shader uses `@Cd` for albedo input, drives roughness/metallic from
   constants per-prompt or from `fragment_id`-based lookup.

5. **Lighting**: drop a `Sky Dome` light, attach an HDRI
   (interior/studio softbox HDRIs like Poly Haven `studio_small_03`
   work well).  Add a key `Distant Light` if you want sharper shadow
   strokes.

6. **Render**: in `/stage`, drop a `Karma` render settings node;
   path-trace at 64-256 spp for finals, 8 spp for previews.

7. **Output**: USD render ROP to EXR sequence; ffmpeg-compose to mp4
   in post.

## Optional: anisotropic Gaussian volume rendering (true 3DGS)

Instead of ellipsoid instancing, evaluate each splat as an
anisotropic 3D Gaussian density field via a **Volume VOP**:

```
density(x) = sum_i alpha_i * exp(-0.5 * (x - p_i)^T Sigma_i^-1 (x - p_i))
Sigma_i = R_i diag(scale_i^2) R_i^T   // R_i from orient_i
```

Karma volume primitive + VEX `xnoise` integration along ray gives a
photoreal "splat blob" appearance.  Slower than ellipsoid instancing
but matches the 3DGS look more closely.

We don't need this for paper-grade visuals -- ellipsoid instancing
+ Karma PBR is sufficient.  Keep volume rendering as a fallback if
ellipsoid look is unconvincing.

## File size & disk footprint

- Per-frame `.geo.gz`: ~3.9 MB at 50K points (gzipped JSON).
- 200 frames per prompt -> ~780 MB.
- 11 prompts -> ~8.6 GB.

For a final paper run, keep the export only for the prompts going
into figures/video (5-7 hero prompts, ~5 GB).

## Troubleshooting

- **"orient ignored by Copy-to-Points"**: ensure the attrib is `vector4`
  with name `orient`.  The exporter writes `(i, j, k, s)` order so
  Houdini reads it directly without permutation.

- **"splats look like spheres, not flat"**: check `scale` attribute
  is `vector3` (per-axis).  If only `pscale` exists, splats are
  uniform.

- **"colors look gray"**: `Cd` is sigmoided DC SH.  If material LUT
  isn't applied, drop a wrangle that does
  `@Cd *= chramp("color_lut", @damage)` or simply replace `@Cd` with
  a constant material-conditioned color in the shader.
