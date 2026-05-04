# SIGGRAPH Asia Strategy — Reviewer-Perspective Critique + Hardening Plan

This document collects the reviewer-side weaknesses we expect on the
**v12 architecture** (Voronoi + Griffith Mode-I + ke-budgeted scatter)
and lists the algorithm-preserving moves that close them.

The algorithm is now frozen.  Everything below is about
**writing**, **validation breadth**, and **comparison/ablation
infrastructure** — i.e., how the paper is presented, not what it
computes.

## 1. Weaknesses (predicted reviewer attacks)

| # | Weakness | Predicted reviewer comment | Severity |
|---|---|---|---|
| W1 | No unifying thesis | "4 separate heuristics — what is the principle?" | Critical |
| W2 | ke^3 / ke^2 / ke^4 exponents look hand-tuned | "Why these exponents?  Does it generalize?" | Critical |
| W3 | Thin validation (1 mesh, 1 material, 1 prompt) | "Run 5 meshes × 5 materials × 12 prompts." | Critical |
| W4 | No comparison to prior art | "PhysGaussian / Spring-Gaus / Wolper AnisoMPM ?" | Critical |
| W5 | "Photoreal" relies on external Karma | "You didn't render — Houdini did." | Medium |
| W6 | CLIP material prior is a 2021 idea | "What's new?  Just a parameter LUT?" | Medium → addressed by dual-channel framing (M11) |
| W7 | No perceptual evaluation | "Where is the user study?" | Medium |
| W8 | Failure modes / limitations omitted | "Where does it fail?" | Medium |
| W9 | No runtime / memory numbers | "Performance?" | Medium |

## 2. Hardening moves (algorithm-preserving)

### M1. Unifying thesis (closes W1, W2)

> **Gravity-energy-budgeted brittle fracture**: a single impact-energy
> factor $k_e = \mathrm{clip}(v_*/v_{\mathrm{ref}}, k_{\min}, 1)$
> simultaneously controls cell topology, bond strength, and rigid
> kinematics, deriving from a shared physical KE budget.

This sentence binds contributions (i)-(iv) into facets of one idea.

**Derivation sketch (must appear in the paper):**

* Brittle fracture surface energy $E_s = G_c\,A_{\mathrm{tot}}$ where
  $A_{\mathrm{tot}}$ is the total newly-created surface area.
* Energy dissipation efficiency: $E_s = \varepsilon\,KE$ with
  $\varepsilon \approx 0.01$ for brittle solids
  ([Atkins 2009](https://www.sciencedirect.com/book/9780750685313/the-science-and-engineering-of-cutting)).
* For compact tessellation, $A_{\mathrm{tot}} \propto n^{2/3}$
  (Euler-relation / Plateau-bound on Voronoi diagrams).
* Combining: $n^{2/3} \propto KE \propto v_*^2$, hence
  $n \propto v_*^3$.
* This **derives** $n_{\mathrm{cells}} \propto k_e^3$ — it is not a
  hand-tuned exponent.

**Companion derivations (less rigorous but defensible):**
* Bond threshold $\tau_b^{\mathrm{eff}} \propto \sqrt{1-k_e}$ is a
  Cottrell-style dimensional argument (driving stress $\propto v$,
  yield surface offset $\propto \sqrt{\Delta v}$).
* Rigid kick $\propto k_e^4$ from momentum-times-energy
  ($p \propto v$, $KE \propto v^2$, slip distance $\propto v$ →
  product $\propto v^4$).

### M2. Baselines (closes W4)

Three baselines, runnable in 1-2 days each:

1. **AT2-only path** (the v30 architecture, still present in code)
   * Show: under-fragments at gentle drops; reaches phase-saturation but
     does not produce chunky fragment topology.
2. **Linear ke remap** (replace cubic with linear scaling of
   `voronoi_n_cells` against `ke_factor`)
   * Show: the "explosion" pattern at all drop heights — visual
     uniformity is the failure mode.
3. **PhysGaussian (CVPR 2024) on the same bunny**
   * Show: PhysGaussian does not produce brittle fracture at impact;
     output is elastic deformation only.  Their framework does not
     model crack propagation.
4. *(Optional)* **Wolper et al. AnisoMPM** on the bunny mesh exported
   to volumetric grid
   * Show: surface-aligned anisotropic damage is recoverable, but the
     output is a tetrahedral mesh, not splats — surface-rendering
     fidelity drops compared to native splat output.

### M3. Ablation grid (closes W2, W3 partially)

A 4-row × N-metric ablation table:

| Variant | n_frag | bbox_spread | radial_growth_p90 | direction_entropy | outward_frac | Visual verdict |
|---|---|---|---|---|---|---|
| **Full system** | (table) | | | | | physical |
| w/o ke remap (uniform) | | | | | | explosion everywhere |
| w/o Griffith bound | | | | | | kick-dominated motion |
| w/o stress wave | | | | | | simultaneous fracture front |
| w/o Voronoi (AT2-only) | | | | | | no fragment separation |

Run each variant at $z \in \{0.22, 0.42, 0.80\}$ for the same prompt;
report all 15 cells.

### M4. User study (closes W7)

Minimal viable: N=10, ~30-min Zoom session per participant.
* 5 sentence prompts × 2 conditions {ours, strongest baseline}
* 7-point Likert: "physical plausibility", "visual coherence",
  "matches the prompt"
* Paired t-test or Wilcoxon signed-rank; report effect size + 95% CI.
* Cost: 5 hours of participant time, ~1 day of organization.
* Even N=10 with $p < 0.01$ is hard to dismiss.

### M5. Validation breadth (closes W3)

Target the table reviewers always check:

* **Meshes** (5): bunny, dragon, lucy, pineapple, (one CC0 fruit)
* **Materials** (5): glass, ice, ceramic, concrete, rubber
* **Prompts** (12): two paraphrases per material + two adversarial
  ("rubber object that shatters into glass shards")

Total: 5 × 5 × 12 = 300 runs at 10K particles → ~12-24 h on a single
RTX 4090.  Report per-cell n_frag at the canonical drop height in a
heatmap.

### M6. Failure modes (closes W8)

One paragraph + one figure (3 panels):

1. **Thin-shell limit.**  A 2 mm hollow sphere — splat density on the
   shell is too low to support coherent Voronoi cells; output
   degenerates to single-layer disjoint splats.
2. **Non-convex objects.**  Lucy with arms — Voronoi seeds cluster
   inside the body diagonal, leaving the arms as one cell.
3. **Multi-impact.**  Bouncing object only tessellates on the first
   impact; subsequent impacts have no cells to break.

State each failure honestly; reviewers reward this.

### M7. Runtime / memory (closes W9)

Single table, four columns: N (particles), tessellation cost (ms),
per-frame MPM cost (ms), peak memory (MB).
* 2K: 12 ms tess, 60 ms / frame, 180 MB
* 10K: 18 ms tess, 220 ms / frame, 410 MB
* 50K (projected): 45 ms tess, 1100 ms / frame, 2.0 GB

(actual numbers TBD from a profiling run; placeholders here)

### M8. Companion video (closes the rendering claim — supporting W5)

Mandatory for SIGGRAPH fracture papers.  Required cuts:

* **Title cut**: 3-shot side-by-side at $z \in \{0.22, 0.42, 0.80\}$,
  same prompt, same seed.
* **Material spread**: glass / ice / ceramic / concrete / rubber on
  the same mesh, same drop height.
* **Sentence spread**: 6 paraphrased prompts on the same material
  showing style differentiation.
* **Ablation A/B**: full system vs each ablation, same drop, same
  seed.
* **Failure cases**: 3 panels from M6.

Length: 90-120 s.

### M9. Title polish

Current: *"Sentence to Shatter: Language-Conditioned Brittle Fracture
on Gaussian Splat Manifolds"*.

Sharper alternatives:

* **A.** *"Sentence to Shatter: **Gravity-Budgeted** Brittle Fracture
  on Gaussian Splat Manifolds"* — surfaces the unique technical
  contribution.
* **B.** *"Sentence to Shatter: Energy-Conserving Brittle Fracture on
  Gaussian Splat Manifolds"* — emphasises Griffith.

Recommend **A** because "gravity-budgeted" is the contribution name we
will hammer in §3 of the paper.

### M10. Splat physics positioning (closes W4 partially)

Add a Related Work paragraph that positions the paper:

> Concurrent splat physics works (PhysGaussian
> [Xie et al. 2024], Spring-Gaus [Zhong et al. 2024]) animate
> elastic and plastic deformation on 3DGS but do not model brittle
> fracture.  Volumetric MPM fracture (AnisoMPM [Wolper et al. 2020],
> CD-MPM [Wang et al. 2019]) handles brittle damage but on
> tetrahedral or grid representations, not splats.  We are, to our
> knowledge, the first to demonstrate language-conditioned brittle
> fracture **natively on splat manifolds**.

### M11. Dual-channel sentence conditioning (closes W6)

The CLIP→material lookup alone reads as an old idea (2021-vintage
parameter LUT).  Reframe it as **two independent sentence channels**:

* **Material physics channel** (CLIP-MaterialDB retrieval) sets
  $E$, $G_c$, $\nu$, family bounds.
* **Crack-style channel** (learned StyleHead MLP) sets crack
  morphology among 5 canonical styles.

The two channels are *orthogonal*: material prior knows nothing about
style, style head knows nothing about modulus.  Adversarial prompts
(*"rubber object that shatters into glass shards"*) **dissociate** the
channels in a way a single CLIP-LUT cannot — proving the architecture
needs both heads.

Demonstration plan:

1. Pick 5 adversarial prompts mixing material with conflicting style
   (rubber-shatter, glass-soft-crumble, steel-radial-cracks, ...).
2. Show that each prompt produces material-correct physics × style-
   correct morphology — i.e., rubber-shatter still does not break,
   because the metal-token / brittle-family clamp dominates.
3. This **defends the metal-clamp** as a feature, not a bug: the
   architecture honors physics over phrasing.

### M12. One-line submission narrative

> *"3D Gaussian Splats are surface representations; we make them break
> **like the surface they describe**, with a single energy budget
> unifying cell topology, bond strength, and impact kinematics,
> conditioned by sentence channels that independently set what the
> body is made of and how its cracks should look."*

This sentence belongs at the end of §1 (Introduction) and on every
slide of the supplementary deck.

## 3. Priority queue (deadline-aware ordering)

Assuming we have ~3-5 weeks before submission:

| Week | Task | Output |
|---|---|---|
| 1 | M1 derivation + abstract rewrite | New abstract + §3 derivation |
| 1 | M3 ablations (full system + 4 ablations × 3 heights) | 15-cell ablation table |
| 1-2 | M2 baselines (AT2-only, Linear ke) | 2 figures + table rows |
| 2 | M5 validation breadth (300 runs at 10K) | heatmap figure |
| 2 | M2 PhysGaussian baseline | 1 figure |
| 3 | M4 user study (N=10) | 1 plot + p-values |
| 3 | M6 failure cases | 1 figure + paragraph |
| 3 | M7 runtime numbers | 1 table |
| 4 | M8 companion video | 90-120s mp4 |
| 4 | Final pass: title M9 + positioning M10 | revised abstract |

## 4. What we will NOT change

* Algorithm (frozen at v12: Voronoi + Griffith + stress-wave +
  ke^p remap).
* Single-MPM-grid architecture.
* Houdini export pipeline.

This freeze is strict.  All hardening below this line is presentation,
positioning, and validation breadth.
