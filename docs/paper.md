# Paper Draft Notes

Working title, abstract, and pipeline for the SIGGRAPH Asia
submission.  Reflects the current frozen architecture (v25b: Voronoi
pre-fracture + Griffith-bounded Mode-I + impact-energy-budgeted
scaling + spatial-CC fragment refinement).  Older drafts (v12, v30)
are kept in *paper_archive.md* for reference.

## Title (working)

> **Sentence to Shatter: Language-Conditioned Brittle Fracture on
> Gaussian Splat Manifolds**

Kept neutral so the framing extends past gravity drops to horizontal
impacts, hammer strikes, explosions, and any other impulse source
the validation suite eventually covers.

## Main contribution (one sentence)

> **The first language-conditioned brittle fracture system on 3D
> Gaussian Splat representations, with a dual-channel CLIP / StyleHead
> conditioning that orthogonally controls material physics and crack
> morphology, supported by an energy-conserving Voronoi + Griffith
> fracture pipeline operating natively on the splat substrate.**

The two pieces this paper actually owns:

1. **Splat-native brittle fracture.**  Concurrent splat-physics works
   (PhysGaussian, Spring-Gaus, GaussianForge) animate elastic and
   plastic deformation but do not model brittle fracture; volumetric
   MPM fracture (AnisoMPM, CD-MPM) handles damage but produces
   tetrahedral / grid output, not splats.  We are, to our knowledge,
   the first to demonstrate brittle fracture **directly on the splat
   manifold** — no mesh-to-grid retraining, no tet conversion.

2. **Dual-channel sentence conditioning.**  CLIP+MaterialDB recovers
   *what the body is made of* ($E$, $G_c$, $\nu$, family bounds),
   while a learned StyleHead MLP independently selects *how the
   cracks should look* (radial shatter, spiderweb branching, single
   smooth, chunky crumble, diffuse microcrack).  Adversarial prompts
   like *"rubber object that shatters into glass shards"* dissociate
   the channels and reveal the architecture needs both heads — a
   single CLIP–LUT cannot represent that orthogonality.

The mechanical fracture pipeline is the supporting infrastructure
that makes (1) and (2) physically faithful; it is not claimed as
the headline contribution.

## Abstract (LOCKED — submission version)

```latex
\begin{abstract}
We address the problem of controlling material-dependent fracture
animations in 3D Gaussian Splat scenes from natural-language prompts.
The same instruction to ``break'' should produce different crack
patterns, fragment distributions, and damage behaviors for glass,
concrete, and rubber, depending on material properties and impact
conditions.  However, recent 3DGS dynamics and fracture methods do
not expose such fracture behavior as a directly prompt-controllable
interface.  We propose a splat-compatible framework that translates
natural-language material and fracture descriptions into material-
constrained fracture controls.

Our key idea is to decompose each prompt into physical feasibility
and morphological bias.  The physics channel uses CLIP--MaterialDB
retrieval to estimate material properties and family constraints,
defining the feasible fracture response.  The morphology channel
uses a learned StyleHead MLP to predict controls for crack seeding,
anisotropy, and breakage biases.  Thus, morphology does not override
material response; it modulates crack and fragment patterns within
the regime allowed by the material channel.  This separation keeps
the two roles disentangled under conflicting prompts, such as rubber
described with glass-like shattering.

The controls are realized as 3DGS fracture animations without mesh
reconstruction or tetrahedralization.  Input Gaussians become
Gaussian-derived particles integrated on a shared MPM grid.  At
impact, we construct a Voronoi cell-and-bond network adapted to
impact location and energy, and use a Griffith energy budget to
limit normal bond opening within material-imposed bounds.  Through
visual comparisons against reference fracture imagery, comparisons
with related 3DGS fracture methods, dual-channel ablations, and
fragment statistics, we show material-distinct and prompt-controllable
3DGS fracture behaviors across materials, morphologies, conflicts,
and impact conditions.
\end{abstract}
```

## Pipeline (overview)

The system runs as a seven-stage chain on a single shared MPM grid.

### 1. Input

A 3D Gaussian Splat scene with per-Gaussian position $p_i$,
covariance $\Sigma_i$, opacity $\alpha_i$, and SH-encoded RGB $c_i$,
plus a natural-language prompt $s$.  No 3DGS retraining is required;
all simulation operates on a reusable particle representation
sampled from the input mesh.

### 2. Sentence conditioning — CLIP material + learned StyleHead

CLIP encodes $s$ to a 512-d embedding which feeds two **independent**
heads:

* **Material physics channel.**  Cosine retrieval over `MaterialDB`
  yields the top-$K$ nearest material entries; their $E$, $G_c$,
  $\nu$ are softmax-weighted and log-averaged.  The predicted family
  ($\{$`sharp_brittle`, `brittle_moderate`, `rough_quasi_brittle`,
  `neutral_reference`, `diffuse_damage`$\}$) imposes hard caps:
  *e.g.*, a metal-token + `neutral_reference` prediction force-
  disables crack propagation (steel does not brittle-fracture under
  this regime regardless of phrasing).

* **Crack-style channel.**  A learned StyleHead MLP
  $\mathrm{CLIP}_{512}\!\to\!128\!\to\!5$ trained on weak supervision
  (228-sentence template corpus) selects one of five canonical
  morphologies.  Above a confidence threshold $p\!\geq\!0.55$ its
  prediction sets cell-seed distribution, anisotropy axis, bond
  threshold, and force-shrink fraction; otherwise a keyword rule
  is the fallback.

The product *(material × style)* lets adversarial pairings expose
the dual-channel structure: *"rubber that shatters into glass
shards"* still produces rubber-style minor damage because the
metal/rubber family clamp dominates over the style request.

### 3. Free fall on a shared MPM grid

The body integrates gravity on the shared MPM grid without contact.
COM linear velocity accumulates; deformation gradient $F$ is
preserved across substeps.  An optional initial $\omega_{\mathrm{drop}}$
allows a tumbling drop (default zero).  The same grid will host any
post-contact impulse source (gravity, hammer, lateral push), so
nothing is gravity-specific below.

### 4. Impact $\to$ Voronoi tessellation

At first contact we record the impact speed $v_*$ and form
$k_e = \mathrm{clip}(v_*/v_{\mathrm{ref}}, k_{\min}, 1)$.  We seed
$N_{\mathrm{cells}} = \max(4, \lfloor N_{\mathrm{full}}\,k_e^6 \rfloor)$
Voronoi cells (impact-biased distribution + optional anisotropy
axis), assign each MPM particle to its nearest seed, and build the
bond graph between adjacent cells.  The effective bond threshold
$\tau_b^{\mathrm{eff}} = \tau_b + \sqrt{1\!-\!k_e}\,(1\!-\!\tau_b)$
makes soft impacts nearly unbreakable.

### 5. Stress-wave bond breakage + Griffith Mode-I release

Each frame, a wave radius $r_w \mathrel{+}= c_w\,k_e^2$ grows from
the impact center.  A bond $(a,b)$ breaks when its midpoint lies
inside $r_w$ AND its accumulated stress damage exceeds
$\tau_b^{\mathrm{eff}}$.  On break, paired Newton-third-law kicks of
magnitude
$v_{\mathrm{open}} = \sqrt{G_c\,A_{\mathrm{bond}}/m_{\mathrm{cell}}}$
open the bond along the inter-cell direction.  The Griffith bound
ties opening velocity to material toughness, so released KE never
exceeds physical surface energy and falling motion stays dominant.

### 6. Spatial-CC fragment refinement + temporally stable ids

Each frame we run particle-level connected components within each
cell-component, in *current* world space, with $\varepsilon$ set to
$0.025\!\times\!\mathrm{bbox}$.  Spatially-disjoint sub-components
get their own fragment ids (so a chunk that drifts away from the
main mass renders as a separate piece, not a recoloured patch of
its origin cell).  Cluster ids are stabilised across frames via
majority-overlap inheritance: a cluster keeps its render colour as
long as $\geq\!50\%$ of its particles came from the same prior id.

### 7. Continued MPM + render export

The body continues on the shared MPM grid with slip boundary
conditions on lateral walls and floor.  Per-fragment shape matching
keeps each chunk rigid via SVD-recovered $R$.  Per-frame splat state
is exported as gzipped Houdini JSON (`frame_NNNN.geo.gz`) carrying
$P, C_d, \alpha, \mathtt{scale}, \mathtt{orient}, \mathtt{fragment\_id},
\mathtt{damage}$ for path-traced Karma rendering.

## Algorithmic contributions

### Headline (claimed in §1)

1. **First brittle fracture on Gaussian Splat manifolds.**  No
   tetrahedralisation, no mesh-to-grid retraining; the Gaussians are
   sampled to a shared MPM grid, broken via Voronoi cells, and
   exported back as splats for path-traced rendering.  Concurrent
   splat-physics work covers elastic / plastic deformation only.

2. **Dual-channel sentence conditioning.**  A CLIP-MaterialDB
   nearest-neighbour lookup *and* a learned StyleHead MLP
   independently set material physics and crack morphology from one
   prompt, with adversarial prompts dissociating the channels.
   Implementation: `MaterialPriorAdapter.predict_sentence_style`
   (style head) + `MaterialPriorAdapter.build_material_prior`
   (material retrieval).

### Supporting mechanics

3. **Voronoi pre-fracture substrate.**  An impact-time cell-and-bond
   network on top of the shared global MPM grid; replaces per-fragment
   grids and emergent-only fragment detection.  Implementation:
   `VoronoiDecomposer` + `_init_voronoi_tessellation`.

4. **Griffith-bounded Mode-I bond opening.**  Released opening
   velocity $v_{\mathrm{open}}=\sqrt{G_c A_{\mathrm{bond}}/m_{\mathrm{cell}}}$
   is energy-conserving by construction, so the kick is tiny relative
   to body velocity at hard impact.  Implementation:
   `_apply_bond_opening_kick`.

5. **Stress-wave bond-breakage gating.**  Wave radius grows each
   frame from the impact center; bonds outside the front are excluded
   regardless of accumulated damage, producing a temporally-coherent
   crack front.  Implementation:
   `VoronoiDecomposer.update_bond_breakage(wave_speed_per_frame)`.

6. **Impact-energy budgeted scaling.**  One factor $k_e$ scales the
   entire cascade with monomial exponents derived from a shared KE
   budget — $n_{\mathrm{cells}}\!\propto\!k_e^6$,
   $\tau_b^{\mathrm{eff}}\!\propto\!\sqrt{1-k_e}$,
   $\{c_w, r_{\mathrm{shock}}\}\!\propto\!k_e^2$, rigid kick $\propto\!k_e^6$
   — so no per-scene retuning is required across impulse sources.

7. **Spatial-CC fragment refinement with stable ids.**  Per-frame
   particle-level connected components with majority-overlap id
   inheritance; splits Voronoi cells whose particles drift apart while
   keeping each fragment's render colour stable across the animation.
   Implementation: `_spatial_split_fragments`.

## Supporting system contributions

* **Slip BC at the MPM grid level** — `grid_update` zeros normal
  velocity into lateral walls and floor (aligned with
  `gravity_drop_ground_z`); eliminates wall-stuck artifacts and
  Z-tunneling without per-fragment grids.

* **Houdini-native `.geo.gz` export pipeline** — per-Gaussian
  $P, C_d, \alpha, \mathtt{scale}, \mathtt{pscale},
  \mathtt{orient}, \mathtt{fragment\_id}, \mathtt{damage}, N$
  attributes consumed directly by Houdini's File SOP +
  Copy-to-Points network for Karma path-traced rendering.

* **Multi-density validation** at 2K, 10K, 15K particles, with
  $\sqrt{N}$-scaled fragment thresholds; metric heatmaps confirm
  monotonic scatter gradient across $z \in \{0.22, 0.50, 0.80\}$.

## Validation summary (locked at v25b)

| Drop height | $v_*$ | $k_e$ | $n_{\mathrm{cells}}$ | $n_{\mathrm{fragments}}$ | bbox spread | rigid kick |
|---|---|---|---|---|---|---|
| z = 0.22 | 18.2 | 0.26 | 4 → 8 | 6  | 1.10 | 0.000 |
| z = 0.50 | 47.9 | 0.68 | 25     | 23 | 2.50 | 0.25  |
| z = 0.80 | 66.2 | 0.94 | 178    | 95 | 2.92 | 2.36  |

(15K particles per run, frozen seed=1234, soda-lime glass prompt.)
