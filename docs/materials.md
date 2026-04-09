# Material-Aware Fracture Mechanics + Fragment Separation
## Graphics-Oriented Design Specification
### Built on Existing Pretrained CLIP Material Retrieval

---

## 1. Document Goal

This document specifies the next stage of the fracture system after the successful transition to:

- Gaussian-manifold fracture representation,
- explicit crack-tip front propagation,
- local damage band formation,
- and delayed feedback to MPM.

The next target is:

> **material-aware fracture mechanics + fragment separation**

Importantly, this document assumes that a **pretrained CLIP-based material database and retrieval pipeline already exist**.

That means the next step is **not** to build a new material predictor from scratch.
Instead, the goal is to:

1. reuse the existing CLIP material retrieval output,
2. translate that output into fracture-relevant priors,
3. condition crack growth, crack band behavior, opening, and fragment separation on those priors.

This is the most natural next stage from a graphics and simulation perspective.

---

## 2. Current System Status

At the current stage, the system already supports:

- physically grounded crack initiation cues from MPM,
- explicit crack-tip front propagation on the Gaussian manifold,
- local damage band formation around the active front,
- delayed constitutive feedback to avoid premature lower collapse,
- and fracture rendering through 3DGS.

This is already a strong structural improvement over a pure damage-diffusion or direct volumetric-to-surface projection pipeline.

However, at this point the fracture process is still mostly **geometry- and impact-driven**.

The next major limitation is:

> the crack path may be plausible, but it is not yet sufficiently **material-specific**.

In other words, the system can decide **where** a crack goes,
but not yet fully **how this specific material should crack**.

That is the core motivation for the next stage.

---

## 3. Why Material Awareness Matters

From a graphics expert’s perspective, fracture realism is not determined only by:

- object shape,
- impact condition,
- or stress concentration.

It is equally determined by **material identity** and **fracture style**.

For example:

- ceramic tends to form narrow, sharp, brittle cracks,
- glass tends to split abruptly with strong opening,
- concrete tends to create rough and fragmented patterns,
- rubber may accumulate deformation/damage without clean fragment separation,
- wood often shows directional fracture behavior.

So if the system only uses one universal crack-growth law, it will miss a major part of visible fracture realism.

That means the next logical step is to make the fracture model **material-aware at the level of crack behavior**, not only at the level of constitutive physics.

---

## 4. Why Existing CLIP Material Retrieval Is Already Enough

Since a pretrained CLIP material database is already implemented, the system already has a semantic entry point for material awareness.

That is extremely valuable.

The current CLIP retrieval system likely already provides:
- material identity,
- similarity scores,
- and possibly physical priors such as `E`, `nu`, `Gc`, `rho`.

From a graphics/systems perspective, this is sufficient to move forward.

The missing piece is not a better material classifier.
The missing piece is:

> **an adapter that converts CLIP material priors into fracture-behavior priors.**

This is the key design move.

---

## 5. Core Design Principle

The design should separate two levels of material representation:

### 5.1 Physics material parameters
These are parameters such as:
- `E`
- `nu`
- `Gc`
- `rho`

They influence:
- deformation,
- stiffness,
- energy scale,
- and the physical side of simulation.

### 5.2 Fracture behavior parameters
These are parameters such as:
- crack initiation threshold,
- front growth gain,
- band width,
- opening gain,
- edge break rate,
- branching tendency.

They influence:
- how the crack visually propagates,
- how narrow or broad the crack band is,
- how easily fragments separate,
- and how the fracture looks.

This distinction is very important.

A CLIP material retrieval system does not need to predict exact fracture behavior directly.
It only needs to provide a **semantic material prior** from which fracture-style parameters can be derived.

That framing is both practical and research-sound.

---

## 6. Recommended Architecture

The next-stage architecture should be:

```text
Text / Prompt / Optional Image
    -> CLIP Material Retrieval (existing pretrained system)
    -> Material Prior Adapter (new)
    -> Physics Parameters + Fracture Behavior Parameters
    -> MPM Simulation
    -> Crack Initiation / Tip Growth / Band Fill
    -> Edge Weakening / Fragment Separation
    -> 3DGS Rendering
```

The key new block is:

> **Material Prior Adapter**

This is the central implementation bridge.

---

## 7. What the Material Prior Adapter Should Do

## 7.1 Input
The adapter should consume the existing CLIP retrieval result.

Possible input forms:
- top-1 material label,
- top-k retrieval list,
- similarity-weighted material candidates.

Example:
```python
[
    {"name": "ceramic", "score": 0.62, "E": ..., "nu": ..., "Gc": ...},
    {"name": "plaster", "score": 0.24, "E": ..., "nu": ..., "Gc": ...},
    {"name": "concrete", "score": 0.14, "E": ..., "nu": ..., "Gc": ...},
]
```

## 7.2 Output
The adapter should produce:

### physics prior
```python
physics_prior = {
    "E": ...,
    "nu": ...,
    "Gc": ...,
    "rho": ...,
}
```

### fracture prior
```python
fracture_prior = {
    "tau_init": ...,
    "growth_gain": ...,
    "band_width": ...,
    "band_fill_gain": ...,
    "open_gain": ...,
    "split_threshold": ...,
    "edge_break_rate": ...,
    "branching_bias": ...,
    "anisotropy_strength": ...,
}
```

This is the output that the rest of the fracture system should consume.

---

## 8. Why an Adapter Layer Is the Correct Design

From a graphics pipeline perspective, an adapter layer is the correct design for three reasons.

### 8.1 It decouples semantic recognition from fracture tuning
The CLIP system can stay stable as a pretrained material prior system.

The fracture system can then evolve independently.

This is much cleaner than trying to force CLIP itself to predict all fracture behavior directly.

### 8.2 It makes artistic / visual / physical balancing easier
Fracture realism in graphics often depends on both:
- physically motivated parameters,
- and visually meaningful control parameters.

The adapter is the right place to mix those.

### 8.3 It supports top-k uncertainty naturally
If CLIP retrieval is ambiguous, the adapter can blend multiple material priors softly instead of forcing a hard one-class decision.

That makes the whole pipeline more robust.

---

## 9. Material-Aware Parameters That Matter Most

At the first implementation stage, the most important material-aware parameters are these:

### 9.1 `tau_init`
Controls how easily crack seeds form.

Examples:
- glass / ceramic -> lower threshold,
- rubber -> higher threshold.

---

### 9.2 `growth_gain`
Controls how aggressively crack tips advance.

Examples:
- brittle materials -> larger gain,
- tough / compliant materials -> lower gain.

---

### 9.3 `band_width`
Controls how thick the local damage band becomes.

Examples:
- brittle fracture -> narrow band,
- diffuse damage / tearing -> broader band.

---

### 9.4 `band_fill_gain`
Controls how much the damage band thickens around the active front.

Examples:
- concrete / plaster -> rougher and fuller local damage region,
- glass -> thinner band.

---

### 9.5 `open_gain`
Controls how strongly visible opening occurs after propagation.

Examples:
- glass / ceramic -> higher gain,
- rubber -> lower gain.

---

### 9.6 `split_threshold`
Controls when a local crack becomes visually separated enough to split or detach.

Examples:
- brittle materials -> lower threshold,
- ductile-like materials -> higher threshold.

---

### 9.7 `edge_break_rate`
Controls how fast connectivity breaks along edges.

This is especially important for fragment separation.

---

### 9.8 `branching_bias`
Controls how likely cracks are to branch rather than remain singular and sharp.

Examples:
- concrete -> somewhat higher branching tendency,
- glass -> lower branching tendency.

---

## 10. Recommended Initial Material Style Table

A very practical starting point is a rule-based style database.

Example:

```python
STYLE_DB = {
    "glass": {
        "tau_init": 0.20,
        "growth_gain": 1.40,
        "band_width": 1,
        "band_fill_gain": 0.10,
        "open_gain": 1.50,
        "split_threshold": 0.30,
        "edge_break_rate": 1.60,
        "branching_bias": 0.10,
        "anisotropy_strength": 0.20,
    },
    "ceramic": {
        "tau_init": 0.25,
        "growth_gain": 1.20,
        "band_width": 1,
        "band_fill_gain": 0.20,
        "open_gain": 1.30,
        "split_threshold": 0.35,
        "edge_break_rate": 1.40,
        "branching_bias": 0.20,
        "anisotropy_strength": 0.20,
    },
    "concrete": {
        "tau_init": 0.30,
        "growth_gain": 0.95,
        "band_width": 2,
        "band_fill_gain": 0.45,
        "open_gain": 0.90,
        "split_threshold": 0.45,
        "edge_break_rate": 1.00,
        "branching_bias": 0.55,
        "anisotropy_strength": 0.10,
    },
    "rubber": {
        "tau_init": 0.55,
        "growth_gain": 0.40,
        "band_width": 3,
        "band_fill_gain": 0.80,
        "open_gain": 0.20,
        "split_threshold": 0.80,
        "edge_break_rate": 0.20,
        "branching_bias": 0.00,
        "anisotropy_strength": 0.05,
    },
}
```

This is not meant to be final truth.
It is a strong practical baseline.

From a graphics professor’s perspective, this is the correct starting point:
- simple,
- interpretable,
- debuggable,
- and visually meaningful.

---

## 11. How to Use Top-k Retrieval Correctly

If the existing CLIP system returns top-k candidates, the adapter should not discard them.

Instead, it should softly combine them.

### Example
If the retrieval is:
- ceramic: 0.62
- plaster: 0.24
- concrete: 0.14

then the fracture prior can be computed as:
- weighted average of style parameters,
- or weighted average of physical + style priors separately.

This is preferable to using only the top-1 result because:

- it preserves semantic uncertainty,
- it avoids hard discontinuities in fracture behavior,
- and it often yields smoother, more believable outcomes.

This is a very strong design choice in practice.

---

## 12. File-Level Implementation Plan

Below is the recommended file-level specification.

---

# 12.1 `src/ml/material_predictor.py`

## Current role
This file likely already performs:
- CLIP encoding,
- retrieval from the material database,
- and top-k material matching.

## Recommended change
Keep this file mostly unchanged.

It should remain the semantic material retrieval system.

## Output expectation
Ensure it returns:
- material label(s),
- similarity score(s),
- and any available physical properties.

That is enough for the next stage.

---

# 12.2 `src/ml/material_prior_adapter.py` *(new recommended file)*

## New role
This file should convert CLIP retrieval results into:
- physics priors,
- fracture behavior priors.

This is the most important new material-aware file.

## Inputs
- top-k retrieval list from CLIP,
- optional manual overrides,
- optional region/part labels.

## Outputs
```python
material_prior = {
    "physics": {...},
    "fracture": {...}
}
```

## Recommended functions
- `build_physics_prior(topk_materials)`
- `build_fracture_prior(topk_materials)`
- `blend_material_priors(topk_materials, scores)`
- `assign_prior_to_object(...)`
- `assign_prior_to_region(...)`
- `assign_prior_to_gaussians(...)`

## Why this file matters
Without this file, the existing CLIP retrieval remains disconnected from the actual fracture behavior.

With this file, the whole system becomes materially grounded.

---

# 12.3 `src/fracture/crack_front.py`

## Current role
This file (or equivalent module) should already control tip-based propagation.

## New role
Now the front-advance logic should become **material-aware**.

## Material-aware parameters used here
- `tau_init`
- `growth_gain`
- `branching_bias`
- `anisotropy_strength`

## What should change
### initiation
Crack seeding threshold should depend on material.

### propagation
Front advance score should be scaled or biased by material-aware growth gain.

### branching
Materials with stronger fragmentation tendencies may allow:
- more successor candidates,
- weaker branching penalty,
- or different top-k successor rules.

## Why this matters
This is where the system decides:
- not only where the crack goes,
- but how aggressively and in what style it propagates.

That is central to material-aware fracture.

---

# 12.4 `src/fracture/gaussian_fracture_field.py`

## Current role
This file maintains local damage band behavior around the active front.

## New role
It should now become **material-aware local fracture band control**.

## Material-aware parameters used here
- `band_width`
- `band_fill_gain`

## What should change
When a crack front advances:
- brittle material -> thinner local band
- rougher / more diffuse material -> wider local band and stronger fill

## Why this matters
This is one of the most visible material-specific effects.

Two cracks with the same path but different band structure can look like completely different materials.

---

# 12.5 `src/visualization/gaussian_updater.py`

## Current role
This file updates visible Gaussian rendering attributes.

## New role
This file should make visible fracture behavior **material-aware**.

## Material-aware parameters used here
- `open_gain`
- `split_threshold`

## What should change
### opening
Visible opening should be more aggressive for brittle materials.

### splitting
Gaussian split / local separation should trigger earlier for materials that are supposed to fragment more easily.

### appearance
Opacity falloff, local crack sharpness, and fracture visibility can also be made slightly material-dependent.

## Why this matters
This is the stage where the user actually sees the material difference.

---

# 12.6 `src/core/fragment_manager.py` or `src/fracture/edge_fragmentation.py`

## Current problem
If fragment separation is based only on node damage threshold, it can become noisy or unstable.

## New role
Fragment separation should be based on **edge weakening and connectivity break**, not only scalar damage.

## Material-aware parameter used here
- `edge_break_rate`

## Recommended design
Maintain an edge-integrity state:
```python
edge_state[i, j] = {
    "integrity": ...,
    "broken": ...,
}
```

Edge integrity should weaken based on:
- damage near the edge,
- crack-front passage through the edge,
- opening magnitude,
- material-specific break rate.

## Why this matters
Fragments are fundamentally about **connectivity**.
Graph-based connectivity is a better representation than scalar thresholding alone.

---

# 12.7 `src/pipeline/fracture_pipeline.py`

## New role
This file should orchestrate the material-aware version of the full pipeline.

## Recommended order
```text
1. CLIP material retrieval
2. Material prior adaptation
3. Physics simulation (MPM)
4. Crack initiation and propagation
5. Local band update
6. Opening / split update
7. Edge weakening
8. Connected-component fragment labeling
9. Rendering
```

## Why this order is correct
It matches the logical progression:

- identify material,
- determine fracture style,
- grow crack,
- weaken connectivity,
- separate fragments,
- render the result.

This is the right causal structure.

---

## 13. Fragment Separation: Correct Graphics View

From a graphics and simulation perspective, fragment separation should not be the first thing that happens.

It should be the **result** of:
- a crack path,
- a local damage band,
- and edge weakening over time.

That means the correct ordering is:

```text
material-aware crack growth
    -> edge weakening
    -> connectivity break
    -> fragment labeling
    -> fragment-aware rendering / motion
```

This ordering matters a lot.

If fragment separation is introduced too early, the system tends to:
- explode into noisy small pieces,
- detach regions before a plausible crack path is visible,
- or lose continuity in the rendered fracture.

That is not desirable.

---

## 14. Recommended Fragment Separation Stages

## Stage 1 — fragment labels only
At first, compute only:
- broken edges,
- connected components,
- fragment IDs.

No independent rigid-body motion yet.

This lets you verify:
- whether cracks actually disconnect the graph,
- whether fragment counts are reasonable,
- whether the split follows the crack path.

---

## Stage 2 — rendering-level separation
Use fragment labels only for rendering:
- slight relative offsets,
- local opening,
- visual decoupling between fragments.

This already gives strong visual improvement.

---

## Stage 3 — fragment-wise motion
Only later, if needed, add:
- fragment center-of-mass,
- fragment velocity,
- fragment-wise transform updates,
- or rigid approximation.

This should come after the graph-based fragmentation is stable.

---

## 15. Why This Roadmap Is Correct

From a graphics professor’s perspective, this order is correct because:

### 15.1 Fracture style must come before fragmentation style
Different materials separate differently because they crack differently first.

### 15.2 Path quality determines fragment quality
If the crack path is not material-specific, the fragments will not look material-specific either.

### 15.3 Renderer-native systems benefit from gradual decoupling
Immediate rigid-body fragment separation often looks worse than progressive graph disconnection followed by controlled rendering separation.

This is especially true in Gaussian-manifold-based systems.

---

## 16. What the Next Evaluation Goal Should Be

The next milestone should not be “more realistic physics” in the abstract.

It should be something concrete and visible:

> **For the same geometry and impact condition, changing only the material prompt should produce visibly different crack growth and fragment separation behavior.**

Examples:
- `"ceramic bunny"` -> sharp narrow crack, earlier split
- `"rubber bunny"` -> diffuse band, little separation
- `"glass bunny"` -> abrupt opening, strong fragmentation
- `"concrete bunny"` -> rougher, slightly more branching fracture

If this works, the system already demonstrates strong material awareness.

That is the right next milestone.

---

## 17. What Not to Do

At this stage, do **not** start by:
- rewriting the CLIP system,
- retraining a new material model,
- making fragment motion fully rigid-body immediately,
- or overcomplicating constitutive coupling.

The current CLIP retrieval is already enough.

The missing piece is not better retrieval.
The missing piece is better **translation from semantic material prior into fracture behavior**.

That is the correct problem to solve now.

---

## 18. Final Summary

### Current advantage
A pretrained CLIP material DB already exists.

### Main next step
Add an adapter layer that translates CLIP retrieval output into:
- physics priors,
- fracture behavior priors.

### Material-aware fracture should influence
- crack initiation,
- tip growth,
- band width,
- opening,
- splitting,
- and edge break rate.

### Fragment separation should be driven by
- edge weakening,
- graph disconnection,
- and connected-component labeling.

### Correct graphics-oriented roadmap
```text
CLIP retrieval
    -> material prior adapter
    -> material-aware crack growth
    -> material-aware band / opening
    -> edge weakening
    -> fragment separation
    -> rendering
```

### One-sentence conclusion
Since the pretrained CLIP material retrieval is already in place, the correct next step is not to redesign material recognition, but to translate that semantic prior into a material-aware crack-growth and fragment-separation model that produces visibly different fracture behaviors across materials.
