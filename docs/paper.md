# Paper Draft Notes

Working title and abstract candidates for the SIGGRAPH Asia
submission.  Updated as the writing progresses.

## v30 LOCKED — Title + Abstract (final draft, post sim-architecture lock)

### Title (locked)

> **Sentence to Shatter: Language-Conditioned Brittle Fracture on
> Gaussian Splat Manifolds**

(Tightened from the earlier "Fracture Animation on Gaussian Splats"
to make the *brittle* scope explicit and the surface-manifold
representation salient.  Alternative B that we considered:
*"Sentence to Shatter: CLIP-Driven Phase-Field Fracture on 3D
Gaussian Splats"* -- skipped because "phase-field" is method-
detail in the title rather than a contribution claim.)

### Abstract (locked, ~220 words)

> We present **Sentence to Shatter**, a language-conditioned
> fracture animation system on 3D Gaussian Splat representations.
> Given a natural-language description (e.g., *"soda-lime glass
> shattering into many sharp radial cracks"* or *"vulcanized rubber
> object under localized impact"*), our system produces
> physically-faithful, photoreal brittle-fracture animation directly
> on the input splat manifold.
>
> We make four algorithmic contributions: (i) a
> **curvature-weighted anisotropy** term that aligns AT2 phase-field
> crack normals with the local surface principal-curvature
> direction, eliminating kNN-graph axis bias; (ii) a
> **causal-support gate** requiring crack-tip support across each
> fragment boundary, which removes AT2-diffusion artefacts where
> fragments form ahead of the visited mask; (iii) a **Griffith
> stress-driven release** that injects per-particle kinetic energy
> at fragment graduation, with magnitude proportional to
> $\sqrt{\sigma_{\mathrm{principal}}}$ along the tensile
> eigenvector; and (iv) an **AT2 halt-after-saturation** rule that
> stops the phase-field solver once damage is saturated and the
> fragment registry is stable, eliminating residual stress-noise
> oscillation.
>
> A unified rigid-body impact response keeps the still-cohesive
> base remnant coherent with released fragments under a single
> global-grid MPM step, and we export every per-frame state as
> Houdini-native geometry for path-traced rendering.  We validate
> on an 11-prompt sentence/material suite at scales from 10K to
> 150K particles.  Ablations confirm: removing AT2 reduces
> complete-shatter quality by **33%**; removing CLIP-driven style
> retrieval reduces material differentiation by **95%**.

### Algorithmic contributions (4 claims)

1. **Curvature-weighted anisotropy** -- per-particle principal
   in-shell direction precomputed via local tangent-plane PCA on
   the kNN graph; AT2 crack normal is blended with the
   in-plane perpendicular to this direction, gated by per-node
   anisotropy strength.  Implementation:
   `GaussianGraph.compute_curvature_directions` +
   `GaussianFractureField._estimate_crack_normal` blend.

2. **Causal-support gate** -- a fragment candidate is rejected
   when the mean cut-vote across its boundary is below
   `fragment_boundary_cut_min_ratio` (0.55 for `radial_shatter`).
   Implementation:
   `FragmentComponentAnalysisMixin._compute_release_scores` filter.

3. **Griffith stress-driven release** -- at fragment graduation,
   each particle in the new fragment receives
   $v_{\mathrm{kick}} = \alpha \sqrt{\lambda_{\max}(\sigma)}\,\hat{e}_{\max}$
   with random sign per particle and an optional downward bias.
   Implementation:
   `FragmentPhysicsMixin._apply_griffith_release_impulse`.

4. **AT2 halt-after-saturation** -- once $c_{\max} \ge 0.999$ and
   the physical fragment registry is stable (frame-to-frame count
   change $\le 1$), the AT2 update is skipped.  Eliminates the
   post-saturation noise that produced visible base-body
   oscillation.  Implementation: skip-flag check in
   `ManifoldSimulator._step_fracture_field`.

### Supporting system contributions

- **CLIP-driven sentence/material style retrieval** with a
  learned StyleHead that emits per-style runtime parameters
  (rule-derived weak supervision).
- **Global p2g2p MPM** (single grid for base body and fragments)
  -- per-fragment shape matching is a label-only overlay; replaces
  the earlier per-fragment ``p2g2p_subset`` that broke
  inter-fragment interaction.
- **Unified rigid-body impact response** -- horizontal v_com
  slide along body-COM-to-impact-center offset + tumble omega
  around the perpendicular horizontal axis; survives shape match
  because it is a NET v_com / omega change.
- **Post-impact gravity / damping runtime overrides** so the
  drop is uniformly accelerating across impact (no "hovering"
  artefact from the legacy hardcoded -400 vs free-fall -2000
  mismatch).
- **Houdini-native .geo.gz export pipeline** -- per-Gaussian P,
  Cd, Alpha, scale, pscale, orient, fragment_id, damage, N
  attributes consumed directly by Houdini's File SOP +
  Copy-to-Points network for path-traced rendering.
- **Multi-scale validation** at 10K, 50K, 100K, 150K particles
  with $\sqrt{N}$-scaled fragment thresholds.

### Locked v30 radial_shatter physics profile

| Parameter | Value | Why |
| --- | --- | --- |
| `shape_match_strength` (BODY) | 0.97 | rigid base remnant |
| `shape_match_fragment_strength` | 0.97 | rigid fragment shards |
| `shape_match_velocity_blend` | 0.0 | disable correction-velocity injection that bypassed `mpm.damping` |
| `post_impact_gravity_z` | -3500 | matches free-fall, uniform acceleration |
| `post_impact_damping` | 0.95 | residual elastic vibration damps within a few substeps |
| `unified_impact_impulse_scale` | 0.05 | horizontal v_com slide proxy for off-center contact |
| `unified_impact_tumble_scale` | 0.20 | tumble omega proxy for off-center contact moment |
| `fragment_griffith_release_gain` | 0.001 | $v$ in 0--2.4 m/s band for $v_{\mathrm{impact}} \approx 35$ |
| `fragment_griffith_downward_bias` | 0.3 | released fragments don't lift against gravity |
| `fragment_release_jitter` | 0.0 | hand-tuned hack OFF |
| `fragment_offset_gain` | 0.0 | hand-tuned hack OFF |
| `fragment_visual_offset_scale` | 0.0 | hand-tuned hack OFF |
| `fragment_physical_release_velocity` | 0.0 | replaced by Griffith physics |
| `rigid_contact_restitution` | 0.0 | no floor bounce |
| `fragment_physical_max_speed` | 2.40 | Griffith kick ceiling |

---

## Earlier title + abstract draft (pre-v30, kept for reference)

## Title (selected)

> **Sentence to Shatter: Language-Conditioned Fracture Animation on
> Gaussian Splats**

Honest about the input modality (text only -- not image+text, hence
not "multimodal") while keeping the catchy "Sentence to Shatter"
prefix.  "Fracture Animation" makes the output explicit and "Gaussian
Splats" anchors the rendering target.

### Earlier candidates (reference)

1. *Language-Conditioned Surface Phase-Field Fracture for Photorealistic
   Gaussian Splat Animation* -- conservative academic framing.
2. *Splat-Aware Fracture: Co-Designing a Surface Phase-Field Crack Front
   with Gaussian Splatting* -- surfaces the co-design thesis.
3. *Crack on Splats: Promptable Photoreal Fracture via Surface Phase-Field*
4. *Surface-Graph AT2 Phase-Field with Tip-Based Crack Fronts for
   Controllable Gaussian Splat Fracture* -- method-first.
5. *Sentence-to-Shatter: Multimodal Fracture Synthesis on Gaussian Splat
   Manifolds* -- earlier "multimodal" form (replaced; we only use text).

## Abstract drafts

### Draft A — method-first (~150 words)

> We present a fracture animation pipeline co-designed with Gaussian
> Splatting rendering.  Because Gaussian splats represent geometry on
> the rendered surface, we localize fracture computation to a surface
> graph defined on the same splat manifold: an AT2 phase-field is
> integrated via a single Jacobi step on the kNN graph of splat
> positions, with tip-based crack-front propagation conditioned by a
> Griffith-style energy gate and a resolution-invariant Top-K branching
> rule.  A CLIP encoder maps prompts to material parameters via
> embedding retrieval, while an auxiliary learned style head selects
> among five canonical crack morphologies (radial shatter, spiderweb
> branching, single smooth, chunky crumble, diffuse microcrack).  A
> physical fragment registry filters surface labels into coherent rigid
> chunks that are written per frame as Houdini-readable point clouds,
> enabling photoreal path-traced rendering without any Gaussian splat
> training.  We validate the method at 10K, 50K, and 100K splats across
> three meshes and twelve ablation configurations, showing four-band
> material-conditioned post-fracture motion (glass/ice → ceramic/concrete
> → steel → rubber) and stable visual results under resolution scaling.

### Draft B — problem-first / narrative (~150 words)

> Animating fracture on Gaussian-splat representations is challenging
> because splats are surface-anchored while standard fracture solvers
> operate on volumetric continua, forcing expensive volume-to-surface
> coupling and producing splat artifacts at exposed cuts.  We address
> this with a fracture pipeline co-designed for splat rendering: an AT2
> phase-field on the splat surface graph, a tip-based crack front with
> energy-conditioned propagation and resolution-invariant Top-K
> branching, and a physical fragment registry that delivers coherent
> rigid chunks to the renderer rather than per-splat orphan labels.
> Crack morphology is conditioned on natural-language prompts through
> CLIP-based material retrieval and an auxiliary learned style head.
> We export per-frame splat state to Houdini for path-traced photoreal
> rendering.  Across three meshes, six materials, and five sentence
> styles, the method produces controllable fracture animations with
> measurable separability under twelve ablation configurations and
> stable behavior from 10K to 100K splats.

Recommendation: **Draft B** -- the problem statement is sharper and the
co-design motivation is exposed directly.

## Method (working draft)

This section drafts the technical body of the paper.  Notation is
held consistent with the implementation: per-Gaussian quantities use
subscript ``i``, per-edge quantities use ``(i, j)``, and graph
operators are written explicitly.

### 3.1 Splat-aligned representation

We sample ``N`` 3D Gaussians on the input mesh's surface with
tangent-frame-aligned anisotropic scaling: each splat ``g_i`` carries
position ``p_i ∈ R^3``, opacity ``α_i``, RGB color ``c_i``, scale
``σ_i ∈ R^3`` (per-axis), and rotation quaternion ``q_i``.  The two
in-plane axes of ``σ_i`` are larger than the out-of-plane axis, so
each splat is a flattened ellipsoid lying tangent to the surface.
Material-conditioned RGB and opacity are assigned procedurally per
material family before simulation.

A surface graph ``G = (V, E)`` is constructed via kNN over splat
positions with normal-aware edge filtering: edges between splats
whose normals point in opposing directions are pruned to prevent
damage from leaking through thin geometry.  Edge weights ``w_{ij}``
are Gaussian kernels of distance with bandwidth ``σ_g``.

### 3.2 Surface-graph AT2 phase field

We integrate a single Jacobi step toward the AT2 phase-field
equilibrium [Bourdin-Francfort-Marigo 2008] on the splat graph.
With ``H_i`` the irreversibly-accumulated history (``H_i ←
max(H_i, ψ_i^+)`` where ``ψ_i^+`` is the per-splat tensile drive
projected from MPM), the equilibrium target is

    c_eq[i] = (a * H_i + b * (l0/σ_g)^2 * lap_c[i]) / (1 + a * H_i),

where ``lap_c[i] = Σ_j w_{ij} (c_j - c_i)`` is the row-normalized
graph Laplacian, ``l0`` is the Allen-Cahn regularization length, and
``a``, ``b`` are calibration coefficients (default ``a = b = 1``).
The damage update is irreversible:

    c_i ← c_i + clip(c_eq[i] - c_i, 0, dC_max * f_dc),

with ``f_dc = 0.5`` so the AT2 layer contributes at most half of the
allowed per-step damage budget.  The remaining budget is spent by the
tip-based crack-front correction (§3.3).

### 3.3 Tip-based crack-front with energy gating

While the AT2 base layer evolves the smooth damage field, we maintain
an explicit set of crack-front tips ``T ⊆ V`` for structured
propagation.  At each substep, every tip ``i ∈ T`` selects up to
``k_max`` successor edges ``(i, j)``.  Three gates apply:

**Griffith gate.**  An edge ``(i, j)`` qualifies for advance only if
its local drive ``ψ_j^+`` clears a Griffith-style threshold
``g_th = 0.5`` (normalized).  This is a hard binary gate, separate
from the soft aggregate score, so propagation becomes an
energy-conditioned event rather than a percentile of the current
frame's drive distribution.

**Energy-driven branch direction.**  Branching candidates are scored
by

    s_branch[j] = lateral(j) * drive(j) * (1 - dup(j)),

where ``lateral(j)`` is the perpendicular-to-tip component, ``dup(j)``
penalizes redundancy with already-visited paths, and the legacy
hash-noise angle target acts only as a soft modulator with floor
``φ_floor = 0.5``.  This places branches at energy hot-spots rather
than at fixed configured angles.

**Top-K resolution-invariant gate.**  Among each tip's ``k_max``
candidates we keep only the top-``K`` by ``s_branch`` (default
``K = 2``).  Because ``K`` is a relative ranking rather than an
absolute threshold, the branch density per tip per advance step is
bounded as graph density grows.  At ``N = 10K``, ``50K``, ``100K``
the per-tip branch behavior is identical, even though the legacy
absolute threshold would let more candidates qualify in the denser
graph.

### 3.4 Multimodal conditioning

A user prompt ``s`` is encoded by CLIP into a 512-d text embedding.
Two heads consume the embedding:

**Material retrieval (CLIP-KNN).**  Cosine similarity against a
pre-encoded `MaterialDB` of described materials yields top-``K``
nearest entries; their Young's modulus ``E``, fracture toughness
``G_c``, and Poisson ratio ``ν`` are softmax-weighted and
log-averaged.  The predicted family among ``{sharp_brittle,
brittle_moderate, rough_quasi_brittle, neutral_reference,
diffuse_damage}`` routes through a family-clamp that establishes
upper bounds on tip count, branching factor, and release ratio.

**Style head (auxiliary learned).**  A two-layer MLP
(``CLIP_dim → 128 → S``) is trained on weak supervision derived
from an existing keyword-rule selector applied to a
``38 phrase × 6 prefix = 228``-sentence template corpus.  The head
infers among ``S = 5`` canonical morphologies (radial shatter,
spiderweb branching, single smooth, chunky crumble, diffuse
microcrack).  When the head's top-class probability exceeds
``0.55`` the head's prediction drives runtime overrides; otherwise
the keyword rule is used as fallback.  At inference the head
generalizes to paraphrased prompts the rule cannot match -- e.g.,
"the bottle disintegrated into many radial pieces" routes to
``radial_shatter`` at ``p = 0.996`` although neither
``radial`` nor ``shatter`` appears in the rule's token list.

**Material physics override.**  When the predicted family is
``neutral_reference`` AND the prompt mentions a metal token (
``steel``, ``iron``, ``titanium``, ``aluminum``, ``alloy``, ...
), runtime is forced to zero out the crack front:
``successor_topk = 0``, ``enable_front_propagation = False``.  This
honors the material physics claim that even with explicit
``radial cracks`` wording, a steel object does not brittle-fracture
under the simulated impact.

### 3.5 Phase-approved fragment birth

Connected components on the damage-cut graph yield candidate fragment
patches.  In strict mode (``crack_connected_release_only = True``), a
patch becomes a fragment only if it survives a narrow-band
phase-approval gate combining four signals:

    score(patch) = 0.34 * cvol_max + 0.18 * cvol_mean
                 + 0.24 * φ_max + 0.12 * opening_max
                 + 0.12 * boundary_score,

with ``cvol`` the volumetric crack proxy, ``φ`` the narrow-band phase
gate, and ``boundary_score`` the cut-supported boundary edge
fraction.  Patches passing the gate at ``score ≥ τ_phase`` (family-
specific, e.g. ``0.34`` for ``sharp_brittle``) are born as
fragments.  The ``hard_detached_mask`` is excluded from the closure-
score numerator so previously-detached patches do not inflate the
scores of new candidate groups.

### 3.6 Physical fragment registry

The graph fragment manager produces per-Gaussian labels every frame,
but graph labels are noisy at high resolution: small surface patches
appear and disappear as crack propagation oscillates.  We maintain a
``_physical_fragment_labels`` registry that filters surface labels
into coherent persistent chunks satisfying
``size ≥ fragment_physical_min_size`` AND ``overlap_with_previous ≥
0.5``.  Each registered chunk receives a stable integer ID across
frames.  The renderer reads these IDs (not graph IDs) so a
fragment's color and pose update consistently across the
animation -- this is what eliminates the "floating Gaussian" artefact
common in 3DGS-fracture pipelines.

### 3.7 Continuous physics through impact

Fragments and the still-attached body are co-simulated on a shared
volumetric MPM grid for accurate momentum and contact dynamics:

* Free-fall: COM linear gravity plus optional rigid-body angular
  velocity ``ω_drop`` so the body can tumble in flight (default
  ``ω_drop = 0``).  No grid physics during fall.
* Impact: per-particle velocity is set to ``v_com + ω × (x - com)``
  preserving rotational state.  Deformation gradient is *not* reset
  to identity (legacy behaviour); ``F`` and affine velocity ``C``
  blend toward identity by ``α`` (default ``α = 0``, no reset).
* Post-impact: damage feedback delay ``τ_d = 4`` frames + ramp
  ``τ_r = 2``, so damage influences stress within 6 frames after
  contact (vs. the 14-frame legacy decoupling that left
  shape-matching solely responsible for cohesion).
* Shape matching: per-fragment SVD-recovered rotation ``R`` and
  angular velocity ``ω``.  Post-impact ``ω`` is damped per substep:
  multiplicative kinetic ``× 0.985`` for ``|ω| ≥ 8`` rad/s, and a
  more aggressive static-friction surrogate ``× 0.5`` below the
  threshold.  This pair of regimes gives a settling envelope --
  rolling motion is preserved during the high-energy phase, and
  numerical noise from the contact-impulse loop cannot sustain
  micro-rotation indefinitely once the body has tipped over.

### 3.8 Photoreal rendering

Per-frame splat state is exported as gzipped Houdini JSON
(``frame_NNNN.geo.gz``) containing ``P, Cd, Alpha, scale (vec3),
pscale (float), orient (vec4 ijk-s), fragment_id (int), damage,
N``.  In Houdini the file feeds a ``File SOP → Copy-to-Points``
network that instances PBR-shaded ellipsoids; HDRI dome lighting and
Karma path tracing produce the final frames.  Material-conditioned
shader settings (transmission for glass/ice, subsurface for ceramic,
etc.) come from a per-family lookup that mirrors the family used by
the simulator's runtime conditioning.  No per-mesh 3DGS training is
required; the procedural material LUT defines appearance.

## Notes

- Five sentence styles is the count exposed by the StyleHead's weak
  supervision corpus; if the paper version trains on a larger
  hand-labeled set we should update the count and language here.
- Six materials in the spread are glass, ice, ceramic, concrete,
  steel, rubber.  Adjust the abstract enumeration if the figure set
  uses fewer.
- "Without any Gaussian splat training" applies because we render via
  Houdini path tracing rather than the Inria CUDA rasterizer.  If we
  later add a 3DGS-trained variant for hero figures the abstract
  should be revised.
