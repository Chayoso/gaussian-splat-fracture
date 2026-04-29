# Gaussian-Manifold Fracture V1 Baseline

Date: 2026-04-28

The active implementation is back to the v1 Gaussian surface-graph fracture
pipeline. The experimental replacement branch has been removed from active code.

## Goal

Run CLIP/material/sentence-conditioned fracture experiments before Gaussian
rendering using raw matplotlib/Y-Z validation. The current baseline keeps crack
state on the Gaussian surface graph and uses MPM only as the driving physics
source.

## Active Pipeline

```text
sentence
-> CLIP material prior
-> scaled MPM material parameters and v1 runtime overrides
-> gravity/drop or external-force MPM step
-> PhysicsProjector maps tensile drive/stress cues to Gaussian surface nodes
-> CrackFront advances tip-based cracks on the surface graph
-> GaussianFractureField updates surface damage/opening
-> crack/opening/cut/closure state feeds immediate narrow-band volumetric damage
-> GraphFragmentManager detects graph-connected fragments
-> bounded physical release drift separates closure-born fragments
-> GaussianCrackVisualizer / matplotlib validation writes outputs
```

## Active Code

- `src/core/manifold_simulator.py`: v1 runtime simulator.
- `src/fracture/crack_front.py`: explicit crack-tip propagation.
- `src/fracture/tip_based_fracture_field.py`: surface damage/opening field.
- `src/fracture/graph_fragment_manager.py`: graph fragment labeling.
- `scripts/validate_material_sentence_gravity.py`: CLIP/material gravity sweep.
- `scripts/validate_sentence_materials.py`: surface sentence/material validation.
- `scripts/inspect_gravity_crack_progression.py`: frame-wise crack/fragment inspection.
- `scripts/build_yz_progression_media.py`: Y-Z snapshot and mp4 media generation.

## Current Rule

Fragments are v1 graph fragments. They are detected by graph connectivity and
damage/cut support on the Gaussian surface graph. This is the accepted baseline
for the current experiments even though it is less physical than a true
volume-coupled phase-field fracture method.

Strict brittle shatter is now phase-approved closure-only.  During impact,
surface crack/cut/closure logic can propose a fragment patch, but the patch is
accepted only when the local narrow-band phase/volume proxy also supports
detachment.  The accepted sequence is:

```text
impact/contact
-> phase/contact gate raises local crack drive
-> crack tips seed, advance, and branch on the surface graph
-> crack/cut corridors propose a closure patch
-> narrow-band phase/volume proxy approves the patch
-> fragment label is born and receives bounded release motion
```

The old non-causal release fallbacks have been removed from the active code.
Accepted strict validation runs now use crack-connected closure/cascade patches
plus narrow-band phase approval for fragment birth.

Rough quasi-brittle `chunky_crumble` uses the same strict rule. It promotes
phase-supported crack-cascade patches once propagated crack/cut corridors and
local narrow-band damage are strong enough. This is the current v1.5 answer to
the concrete bottleneck: fragments are still born from crack/cut evidence, while
phase-field damage weakens the local volume proxy so chunks can detach instead
of remaining as painted cracks on the surface.

CLIP material selection is still top-k blending, but object/material phrase
hints now reweight the top-k list before physics scaling.  This prevents
contact/context words such as concrete ground from dominating prompts whose
object is steel, rubber, glass, or ice.

## Known Limitations

- The phase-field constitutive model degrades stress and now feeds immediate
  narrow-band volume damage from crack/opening/cut/closure state, but
  crack/fragment birth is still controlled by the surface graph fracture
  manager.
- This is not full volumetric PFF-MPM. It is a surface-manifold approximation
  where narrow-band phase/volume damage approves surface crack closure instead
  of solving a full volume fracture topology.
- Crack-to-fragment causality must be judged from crack-front event logs,
  graph cut masks, per-frame `progression_metrics`, and Y-Z validation plots.
- Fragment motion is bounded release drift plus shape-matching/contact response
  with material/style lateral bias and small spin-like velocity.  It is not a
  full per-fragment rigid-body dynamics solver.
- Rubber and other diffuse materials should avoid brittle fragment release.
- Brittle prompts should show crack propagation, branch events, and fragment
  labels in gravity validation.
- Gravity validation should always report phase seed/advance/cut gates together
  with fragment labels and physical total/lateral/drop motion.

## Validation Target

Run 10K to 50K particle sweeps at 64 grid first. Use 128 grid only after the v1
baseline behavior is stable enough to justify the extra cost.

Current 50K/64-grid raw validation:

- Report: `output/phase_approved_birth_sweep_50k_v3/progression_sweep_report.md`
- Y-Z montage: `output/phase_approved_birth_sweep_50k_v3/yz_media/yz_sentence_result_montage.png`
- Y-Z MP4: `output/phase_approved_birth_sweep_50k_v3/yz_media/yz_sentence_result.mp4`
- MP4 verification: 10 frames at 1 fps, 1240x1622.
- All seven prompt classes pass:
  glass radial, glass spiderweb, ceramic single crack, concrete chunks, ice
  radial, rubber no-fracture, and steel no-fracture.

Validation summary:

| prompt class | expected mode | result | first birth | final fragments | released ratio | phase birth |
|---|---|---|---:|---:|---:|---:|
| glass radial | crack-connected fragment | PASS | 0 | 552 | 0.594 | 0.908 |
| glass spiderweb | crack-connected fragment | PASS | 0 | 47 | 0.123 | 0.906 |
| ceramic single crack | crack split/no detach | PASS | -1 | 0 | 0.000 | 0.000 |
| concrete chunks | crack-connected fragment | PASS | 2 | 12 | 0.072 | 0.966 |
| ice radial | crack-connected fragment | PASS | 0 | 539 | 0.608 | 0.908 |
| rubber no fracture | no fragment | PASS | -1 | 0 | 0.000 | 0.000 |
| steel denting | no fragment | PASS | -1 | 0 | 0.000 | 0.000 |

Additional 50K checks:

- Chunked symmetric eigensolve keeps the stress projection stable at 50K.
- Fragment birth/phase approval stats are accumulated across all fracture-burst
  detections in a visible frame, so impact+0 births retain their causal phase
  score instead of being overwritten by a later same-frame no-birth detect.
- The v3 50K ordering matches the intended material behavior: radial glass/ice
  shatter immediately, spiderweb glass releases less, concrete chunks later,
  ceramic stays crack-only, and rubber/steel suppress fragments.

## Material And Style Control Points

The active pipeline already has multiple material/style control points.  The
next validation task is to prove that these controls produce visible,
measurable differences rather than only different parameter tables.

```text
CLIP top-k + lexical material hints
-> material family and physical MPM scaling
-> sentence style runtime override
-> phase seed/advance/cut gates
-> crack-front seed, advance, branching, and closure
-> phase-approved fragment birth
-> shape matching, contact, and bounded release drift
```

Current control mechanisms:

- `MaterialPriorAdapter` maps material words to a family:
  `sharp_brittle`, `brittle_moderate`, `rough_quasi_brittle`,
  `diffuse_damage`, or `neutral_reference`.
- Sentence style rules choose crack morphology:
  `radial_shatter`, `spiderweb_branching`, `chunky_crumble`,
  `single_smooth`, or `diffuse_microcrack`.
- Family/style runtime presets change crack onset, front speed, branch density,
  cut/closure thresholds, phase feedback floors, fragment release caps, and
  post-fragment motion coefficients.
- `ManifoldSimulator` uses material-dependent phase gates and impact burst
  steps, then accumulates frame-level phase-approved birth metrics.
- Post-fragment motion is material-aware through physical release drift,
  lateral bias, spin gain, shape matching, contact restitution/friction, and
  soft squash/rebound for diffuse materials.

Expected visual deltas by material:

| material class | crack motion | fragment birth | post-motion target |
|---|---|---|---|
| sharp brittle glass/ice | fast impact-local radial growth, many branch tips | immediate phase-approved shatter | many light pieces, visible lateral separation, low rebound |
| spiderweb glass | connected local branch network, lower extent than radial | immediate but lower released ratio | local separation, smaller spread than radial |
| brittle ceramic | one dominant smooth crack, low branch density | no detach unless closure is complete | crack/split readout, little scatter |
| rough concrete | slower, thicker, noisy crack/cut band | delayed small chunk birth | heavy chunks, downward/support-loss motion, low lateral throw |
| rubber/diffuse | no crack-front propagation, broad damage only | no crack-connected fragment | squash and rebound, no detached labels |
| steel/neutral denting | suppressed crack motion, high threshold | no crack-connected fragment | dent/tilt/settle, minimal damage visualization |

Evidence metrics to report with every matched visual:

- crack motion: crack step, visited count, tip count, branch event count,
  branch event frame span, branch angle mean/std;
- fragment causality: first birth frame, birth phase score, birth cvol max,
  causal boundary support ratio, closure candidate count;
- topology: final fragment count, released ratio, largest fragment ratio;
- post-motion: physical release displacement, lateral release displacement,
  fragment drop, lateral spread, rigid angular speed, soft squash/rebound stats;
- controls: CLIP top-k, material family, sentence style, runtime release mode.

SIGGRAPH Asia positioning:

- Claim controllable, language-conditioned fracture behavior for Gaussian
  Splats, backed by explicit crack-front causality and phase-approved fragment
  birth.
- Do not claim full volumetric PFF-MPM or exact rigid-body fragment dynamics.
- Use raw matplotlib/Y-Z plots as causal evidence and high-quality Gaussian
  renders as the presentation layer.

Renderer dependency note, 2026-04-29:

- In the active `diffmpm_v2.3.0` conda environment, the available Gaussian
  rasterizer module is `diff_gauss`.
- `diff_gauss` resolves from
  `/home/chayo/Desktop/Shape-morphing-binder/gaussian-splatting/submodules/diff-gaussian-rasterization/diff_gauss`.
- The old direct 3DGS imports `scene.gaussian_model` and `gaussian_renderer`
  are not available in this checkout/environment and should not be used for
  new render work.
- Photorealistic render integration should go through
  `src/renderer/core/renderer.py` (`GSRenderer3DGS`), which wraps
  `diff_gauss`, or through a local fallback splat renderer only when debugging
  simulation output without the rasterizer.

## Current Module Layout

The active implementation is modularized around the current v1.5 pipeline:

```text
CLIP/material prior
-> MPM gravity/contact
-> PhysicsProjector
-> ManifoldSimulator frame orchestration
-> GaussianFractureField + CrackFront
-> GraphFragmentManager
-> raw diagnostics / Gaussian rendering
```

Key code boundaries:

- `src/core/manifold_simulator.py`
  - Owns simulator initialization and frame orchestration.
  - Mixins under `src/core/simulator_mixins/` own surface binding,
    frame-level fragment stats, fracture-drive construction, runtime profiles,
    render-fragment offsets, and post-fragment physics.
- `src/fracture/graph_fragment_manager.py`
  - Owns manager state and `detect_fragments`.
  - Fragment mixins own phase approval, cut-field construction, strict closure
    patch extraction, open/release patch extraction, component/support
    analysis, and connected-component/impulse utilities.
- `src/diagnostics/raw_graph_plot.py`
  - Owns raw matplotlib crack/fragment PNG generation.
  - The progression script now focuses on running prompts and collecting
    metrics instead of embedding all plotting code.

The refactor preserves the current algorithm: fragment birth is still proposed
by crack/cut closure and approved by local narrow-band phase/volume evidence.
It does not introduce full volumetric PFF-MPM or per-fragment rigid-body
dynamics.

Refactor verification:

- Compile passed across core, fracture, diagnostics, validation scripts, and
  smoke test.
- 2K impact smoke passed through CLIP, gravity/contact, crack propagation,
  phase-approved fragment birth, fragment-manager mixins, and raw plotting:
  `output/refactor_modularization_impact_smoke_2k/progression_report.md`.

Previous 10K/64-grid raw validation:

- Report: `output/phase_approved_birth_sweep_10k_v2/progression_sweep_report.md`
- Y-Z montage: `output/phase_approved_birth_sweep_10k_v2/yz_media/yz_sentence_result_montage.png`
- Y-Z MP4: `output/phase_approved_birth_sweep_10k_v2/yz_media/yz_sentence_result.mp4`
- MP4 verification: 10 frames at 1 fps.
- All seven prompt classes pass:
  glass radial, glass spiderweb, ceramic single crack, concrete chunks, ice
  radial, rubber no-fracture, and steel no-fracture.

Validation summary:

| prompt class | expected mode | result | first birth | final fragments | released ratio | phase birth |
|---|---|---|---:|---:|---:|---:|
| glass radial | crack-connected fragment | PASS | 0 | 244 | 0.644 | 0.951 |
| glass spiderweb | crack-connected fragment | PASS | 0 | 44 | 0.161 | 0.834 |
| ceramic single crack | crack split/no detach | PASS | -1 | 0 | 0.000 | 0.000 |
| concrete chunks | crack-connected fragment | PASS | 1 | 14 | 0.090 | 0.896 |
| ice radial | crack-connected fragment | PASS | 0 | 232 | 0.642 | 0.873 |
| rubber no fracture | no fragment | PASS | -1 | 0 | 0.000 | 0.000 |
| steel denting | no fragment | PASS | -1 | 0 | 0.000 | 0.000 |

SIGGRAPH evidence harness:

- `scripts/run_siggraph_evidence.py` now standardizes matched prompt suites,
  runtime override ablations, mesh/config copies, Y-Z montage generation, and
  suite-level `evidence_report.md` files.
- Completed 10K evidence supports the current paper direction:
  sentence/style control, same radial sentence across materials, and crack
  style interpolation all produce measurable fragment/release differences.
- Completed smoke mesh repro on bunny/spot/truck preserves the intended
  material ordering: glass radial shatter > concrete chunks > rubber no
  fragment.
- Ablation support exists through `--override-json`, including no phase
  approval, no crack-front branching, no narrow-band volume feedback, and no
  CLIP material prior.  The first quick10K ablation shows no-branch and
  no-CLIP clearly degrade the result; no narrow-band feedback weakens release;
  no phase approval is inconclusive on easy glass radial.
- A separate controlled phase-gate stress ablation tightens the runtime phase
  approval threshold and weakens volume feedback for a rough concrete prompt.
  With phase approval enabled it rejects surface-only closure (`0` fragments);
  with the gate bypassed it births `12` fragments at impact+1.  This is a
  causal stress test for the gate, not a normal quality sweep.

Current retained outputs:

- `output/phase_approved_birth_sweep_10k_v2` is the accepted 10K reference.
- `output/phase_approved_birth_sweep_50k_v3` is the accepted 50K reference.
- `output/siggraph_evidence_sentence_style_quick10k_v1`
- `output/siggraph_evidence_material_radial_quick10k_v1`
- `output/siggraph_evidence_style_interpolation_quick10k_v1`
- `output/siggraph_evidence_mesh_repro_smoke2k_v2`
- `output/siggraph_evidence_ablation_quick10k_v1`
- `output/siggraph_evidence_phase_gate_stress_quick10k_v4`
- Older smoke/probe/superseded output directories were deleted after the
  phase-approved-birth sweep.
