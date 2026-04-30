# Paper Draft Notes

Working title and abstract candidates for the SIGGRAPH Asia
submission.  Updated as the writing progresses.

## Candidate titles

### Primary candidates

1. **Language-Conditioned Surface Phase-Field Fracture for Photorealistic
   Gaussian Splat Animation**
   _Standard academic phrasing; safest framing._

2. **Splat-Aware Fracture: Co-Designing a Surface Phase-Field Crack Front
   with Gaussian Splatting**
   _Surfaces the co-design thesis directly in the title._

### Alternative phrasings

3. **Crack on Splats: Promptable Photoreal Fracture via Surface Phase-Field**
   _Short and catchy; opens with a verb._

4. **Surface-Graph AT2 Phase-Field with Tip-Based Crack Fronts for
   Controllable Gaussian Splat Fracture**
   _Method-first, very explicit about what is novel._

5. **Sentence-to-Shatter: Multimodal Fracture Synthesis on Gaussian Splat
   Manifolds**
   _Most narrative; emphasizes language conditioning._

Recommendation: **#1** for a conservative submission, **#2** when the
co-design thesis is strong enough to lead with.

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
