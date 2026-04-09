# Graphics-Oriented Roadmap for CLIP-Guided Material-Aware Fracture
## Step-by-Step Implementation Direction, System Design, and Experimental Plan
### Gaussian-Manifold Fracture + Crack-Tip Propagation + Material Family Interpretation + Fragment Separation

---

## 1. Purpose of This Document

This document summarizes the **next implementation direction** of the current fracture system from a **graphics-oriented perspective**.

The goal is not only to improve physical plausibility, but to make the system produce **visually convincing, materially distinct fracture behavior** under a renderer-native formulation.

This document assumes the following are already in place:

- pretrained CLIP-based material retrieval / material DB,
- Gaussian-manifold fracture representation,
- explicit crack-tip front propagation,
- local damage band around the active front,
- delayed feedback from fracture state to MPM,
- and 3DGS-based rendering.

The purpose now is to define the next stage clearly:

> **Translate CLIP semantic material priors into fracture behavior families, and then implement family-specific crack growth, local band formation, opening, fragment separation, and rendering.**

This document explains:
1. how the current system should be interpreted,
2. what the next architectural step is,
3. how each file should evolve,
4. what order implementation should follow,
5. and what experiments should be run to validate progress.

---

## 2. Current System Interpretation

The most important conceptual point is this:

> The material identity currently comes from **CLIP-based semantic retrieval**, not from exact ground-truth fracture mechanics identification.

That means the current material labels such as:

- glass
- ceramic
- concrete
- rubber

should **not** be treated as exact physical truth.

They should instead be treated as:

> **semantic material priors**

In practice, that means:
- CLIP does not tell us the exact fracture law,
- CLIP tells us which material family the object is *most similar to*,
- and the fracture system must interpret that prior into a believable graphics-side fracture mode.

So the next challenge is **not material recognition**.

It is:

> **semantic material interpretation**

This changes how the whole system should be designed.

---

## 3. Correct Graphics Framing

From a graphics expert’s perspective, the fracture pipeline should no longer be framed as:

```text
CLIP material label -> exact material property -> universal fracture solver
```

Instead, it should be framed as:

```text
CLIP semantic prior
    -> fracture family interpretation
    -> family-specific crack growth
    -> family-specific local band / opening / fragmentation
    -> family-specific rendering emphasis
```

This is the correct framing because:
- CLIP retrieval is approximate and semantic,
- fracture appearance is highly material-dependent,
- the current solver is renderer-native rather than a strict classical continuum fracture solver,
- and visible graphics quality depends strongly on how fracture is *presented*, not only how it is *computed*.

---

## 4. Core Problem to Solve Next

The current implementation has already solved a major problem:
- crack initiation exists,
- explicit crack-tip front propagation exists,
- and the system is no longer purely a volumetric phase-field projection pipeline.

However, the system still has a major limitation:

> It can decide **where cracks go**, but not yet strongly **how different materials should fracture in distinct visual modes**.

That is why current outputs may show:
- some difference between materials,
- but not yet the correct graphics-level distinction between:
  - glass-like,
  - concrete-like,
  - rubber-like,
  - and neutral/reference behavior.

So the next stage is not “make everything sharper.”
It is:

> **make the fracture mode family-dependent.**

---

## 5. The Right Abstraction: Fracture Families

The correct next abstraction is to introduce **fracture families**.

Instead of mapping CLIP labels directly to fixed expectations, we should first translate them into a smaller set of graphics-oriented fracture families.

### Recommended fracture families

| Family | Typical CLIP prior | Visual interpretation |
|---|---|---|
| `sharp_brittle` | glass-like | thin, sharp, dominant main cracks, strong opening, clean split |
| `brittle_moderate` | ceramic-like | brittle cracks, but less extreme opening/splitting than glass |
| `rough_quasi_brittle` | concrete-like, plaster-like | rougher crack network, broader band, visible branching |
| `diffuse_damage` | rubber-like, soft-like | little to no explicit crack path, mostly diffuse damage/deformation |
| `neutral_reference` | default / uncertain | non-aggressive, material-neutral comparison baseline |

This family representation is important because it allows the system to:
- remain robust to CLIP ambiguity,
- avoid overcommitting to exact material laws,
- and implement visually meaningful fracture behavior.

---

## 6. Overall Architecture Going Forward

The recommended pipeline is:

```text
Text / Prompt / Optional Image
    -> CLIP Material Retrieval
    -> Material Prior Adapter
    -> Fracture Family Selection / Blending
    -> Physics Prior + Fracture Prior
    -> MPM Simulation
    -> Crack Initiation
    -> Tip-Based Propagation
    -> Local Band Formation
    -> Opening / Split
    -> Edge Weakening
    -> Connected-Component Fragment Labeling
    -> Family-Aware Rendering
```

The critical new idea is:

> **Material Prior Adapter + Fracture Family Layer**

This is the bridge between CLIP and the fracture system.

---

## 7. Design Philosophy: Three Layers

The system should now be understood in three layers.

### Layer A — Semantic material prior
This is the CLIP retrieval layer.
It tells us what the object is semantically similar to.

Examples:
- glass-like
- ceramic-like
- concrete-like
- rubber-like

### Layer B — Fracture family interpretation
This is the new adapter layer.
It translates semantic priors into fracture families and family-specific parameters.

Examples:
- glass-like -> `sharp_brittle`
- concrete-like -> `rough_quasi_brittle`
- rubber-like -> `diffuse_damage`

### Layer C — Family-aware fracture simulation and rendering
This layer performs:
- crack front evolution,
- local band formation,
- opening and splitting,
- edge weakening,
- fragment separation,
- and rendering.

The same generic solver should **not** simply be shared by every material with small parameter changes.
Some families require different modes entirely.

This is especially true for rubber-like materials.

---

## 8. Family-by-Family Graphics Target

Below is the correct graphics target for each family.

### 8.1 `sharp_brittle`
**Typical prior:** glass-like

#### Graphics target
- few or moderate number of crack paths,
- one or a few very dominant main cracks,
- very thin crack core,
- strong opening,
- fast split,
- clean separation.

#### Important note
The target is **not necessarily more crack paths**.
The target is often:
- **fewer but much more salient cracks**.

In other words:
- path count may decrease,
- but crack saliency, opening, and split should increase.

---

### 8.2 `brittle_moderate`
**Typical prior:** ceramic-like

#### Graphics target
- brittle crack appearance,
- relatively thin cracks,
- moderate opening,
- less extreme than glass,
- still visibly sharper than concrete-like fracture.

This family is a useful middle point.

---

### 8.3 `rough_quasi_brittle`
**Typical prior:** concrete-like, plaster-like

#### Graphics target
- broader band,
- rougher shell,
- more visible paths,
- more branching,
- larger damaged zone,
- fragmentation may occur but in a rougher / less clean way.

This family is allowed to have more visible path richness than glass-like fracture.

That is not a bug.

The key is:
- concrete-like fracture should be rougher and broader,
- while glass-like fracture should be sharper and more dominant.

---

### 8.4 `diffuse_damage`
**Typical prior:** rubber-like

#### Graphics target
- little to no explicit crack path,
- no visible crack-tip trace,
- mostly diffuse deformation or diffuse damage,
- almost no opening,
- almost no clean fragment separation.

This family should **not** just be “weak brittle fracture.”
It should be a different mode.

This is extremely important.

---

### 8.5 `neutral_reference`
**Typical prior:** default / uncertain

#### Graphics target
- balanced, non-aggressive behavior,
- no strong commitment to any material family,
- a proper baseline for comparison.

This baseline should not already look concrete-like or quasi-brittle.
It must stay neutral.

---

## 9. Step-by-Step Implementation Roadmap

Below is the recommended implementation order.

This order is important.
It is designed to maximize clarity, debugging ease, and graphics usefulness.

### Step 1 — Stabilize the semantic material interpretation layer

#### Goal
Convert existing CLIP retrieval into family-level fracture priors.

#### Why this is first
Right now the system already has semantic material information, but it is not yet translated into the correct graphics fracture mode.

#### What to implement
- add `material_prior_adapter.py`,
- define fracture families,
- define family presets,
- support top-k weighted blending,
- return both physics priors and fracture priors.

#### Expected outcome
Instead of directly using “glass / concrete / rubber” as exact labels, the system now works in terms of:
- `sharp_brittle`
- `rough_quasi_brittle`
- `diffuse_damage`
- etc.

This gives the whole system a much cleaner structure.

---

### Step 2 — Fix family mode selection for rubber-like materials

#### Goal
Move rubber-like priors out of the brittle front-propagation family.

#### Why this matters
Rubber-like output still showing visited/tip traces is almost certainly the wrong mode.
Graphics-wise, rubber should not look like a suppressed brittle crack.

#### What to implement
For `diffuse_damage` family:
- disable explicit crack front propagation,
- disable or strongly restrict impact seed activation,
- set material drive floor to 0 or near 0,
- disable visited/tips rendering,
- render only diffuse damage / deformation-like response,
- prevent early edge break / fragment split.

#### Expected outcome
Rubber-like prompts should no longer show a clean crack path.
Instead, they should show diffuse response.

This is the highest-priority family correction.

---

### Step 3 — Increase glass-like crack saliency in the right way

#### Goal
Make `sharp_brittle` results look like fewer but stronger main cracks.

#### Why this matters
The current error for glass-like output is often not “too many paths.”
It is:
- insufficient crack dominance,
- insufficient opening,
- insufficient split saliency,
- and too much residual band/visited visibility.

#### What to implement
For `sharp_brittle` family:
- reduce branching bias,
- keep successor count low,
- increase tip/core damage importance,
- increase opening gain,
- increase split sensitivity,
- reduce broad halo / band fill,
- reduce visited trace saliency,
- increase opacity reduction or shell contrast near crack core.

#### Expected outcome
Glass-like fracture should become visually dominant through:
- a clearer main crack,
- stronger opening,
- and earlier split,
not necessarily through more path count.

---

### Step 4 — Recalibrate concrete-like / rough quasi-brittle family

#### Goal
Preserve the strengths of rough quasi-brittle fracture without letting it dominate every comparison.

#### Why this matters
Concrete-like outputs often naturally produce more visible path richness.
That is acceptable.
But they should not erase family distinction by making every fracture look equally salient.

#### What to implement
For `rough_quasi_brittle` family:
- keep wider band,
- keep more branching,
- keep rougher shell,
- but do not over-boost opening or split,
- keep the emphasis on rough network and band richness rather than clean dramatic opening.

#### Expected outcome
Concrete-like fracture should look:
- broader,
- rougher,
- more branched,
but not more “dominant” than glass in the same way.

---

### Step 5 — Redefine the baseline as neutral

#### Goal
Make the default reference truly material-neutral.

#### Why this matters
If the baseline already looks quasi-brittle/concrete-like, then all comparisons become harder to read.

#### What to implement
Define a `neutral_reference` preset with:
- lower growth aggressiveness,
- reduced band fill,
- reduced branching,
- moderate or low opening,
- moderate or low fragment separation tendency.

#### Expected outcome
The baseline becomes a real comparison reference rather than a hidden concrete-like default.

This is very important for future figures and experimental interpretation.

---

### Step 6 — Make the crack front family-aware

#### Goal
Use different crack-front behavior for different families.

#### Why this matters
Crack topology is a major part of material appearance.

#### What to implement
Inside `crack_front.py`:
- `sharp_brittle`:
  - low branching,
  - low successor count,
  - stronger path continuity,
  - high saliency of main path.
- `brittle_moderate`:
  - similar but less extreme.
- `rough_quasi_brittle`:
  - more branching allowed,
  - rougher continuation,
  - larger local front influence.
- `diffuse_damage`:
  - no explicit front.

#### Expected outcome
Material families begin to differ at the path-formation level, not just at rendering.

---

### Step 7 — Make local band formation family-aware

#### Goal
Let `gaussian_fracture_field.py` represent different local fracture bands for different families.

#### Why this matters
The local damage band is one of the most visible parts of material behavior.

#### What to implement
- `sharp_brittle`:
  - thin band,
  - low halo,
  - focused around core/front.
- `brittle_moderate`:
  - thin to moderate band.
- `rough_quasi_brittle`:
  - broad band,
  - stronger fill,
  - rougher neighborhood activation.
- `diffuse_damage`:
  - broad, soft, non-crack-like response.

#### Expected outcome
Two fractures with similar crack paths can still look materially different because the local band looks different.

---

### Step 8 — Make rendering composition family-aware

#### Goal
Stop using one material-agnostic crack shell composition for all families.

#### Why this matters
A rendering rule like:

```text
crack_shell = crack_band | crack_visited | crack_tips
```

tends to favor families with more path traces.
That is why rough concrete-like behavior can visually dominate.

#### What to implement
In `gaussian_updater.py`, use different shell composition per family:

##### `sharp_brittle`
- emphasize core / tip / opening,
- suppress visited trace,
- suppress weak band halo,
- strong opacity drop near crack core.

##### `brittle_moderate`
- similar but less extreme.

##### `rough_quasi_brittle`
- emphasize band, rough shell, branching trace.

##### `diffuse_damage`
- no explicit crack shell,
- only diffuse damage appearance.

##### `neutral_reference`
- balanced, moderate shell.

#### Expected outcome
The renderer finally shows material difference in the correct graphics language.

---

### Step 9 — Add family-aware fragment separation

#### Goal
Turn crack paths into material-dependent fragment separation behavior.

#### Why this matters
Fragments are not just damage.
They are connectivity outcomes.

#### What to implement
Use edge-based integrity / connectivity:

- maintain edge integrity,
- weaken edges through damage + tip passage + opening,
- use `edge_break_rate` per family,
- compute connected components as fragments.

Family behavior:

##### `sharp_brittle`
- faster, cleaner edge break,
- earlier split,
- clean fragment separation.

##### `brittle_moderate`
- moderate separation.

##### `rough_quasi_brittle`
- rougher, more fragmented, but less clean.

##### `diffuse_damage`
- almost no edge break.

#### Expected outcome
Fragment separation becomes part of the material family distinction.

---

### Step 10 — Only then tune physical / style parameters

#### Goal
Use parameter sweeps after the structural family system is correct.

#### Why this matters
Tuning before the family abstraction is correct can hide the real problem.

#### What to tune later
- `E`
- `Gc`
- `tau_init`
- `growth_gain`
- `band_width`
- `open_gain`
- `split_threshold`
- `edge_break_rate`
- feedback delay schedule

#### Expected outcome
Parameter tuning becomes meaningful because the system already has the correct family structure.

---

## 10. File-by-File Implementation Direction

### `src/ml/material_predictor.py`

#### Keep mostly unchanged
This file should remain the existing pretrained CLIP material retrieval module.

#### Required output
It should provide:
- top-k candidates,
- scores,
- material names,
- and any known physical priors.

#### Graphics interpretation
This file is the semantic prior source, not the final fracture controller.

---

### `src/ml/material_prior_adapter.py` *(new or expanded)*

#### Main role
Translate CLIP retrieval results into:
- fracture family,
- physics priors,
- fracture priors.

#### Required outputs
```python
{
    "family": ...,
    "physics": {
        "E": ...,
        "nu": ...,
        "Gc": ...,
        "rho": ...
    },
    "fracture": {
        "tau_init": ...,
        "growth_gain": ...,
        "band_width": ...,
        "band_fill_gain": ...,
        "open_gain": ...,
        "split_threshold": ...,
        "edge_break_rate": ...,
        "branching_bias": ...,
        "anisotropy_strength": ...
    }
}
```

#### Important recommendation
Support:
- hard family selection,
- soft parameter blending,
- and a neutral fallback.

This is the most important new file.

---

### `src/fracture/crack_front.py`

#### Main role
Drive crack-tip propagation.

#### New requirement
Use family-aware propagation rules.

#### Key family knobs
- branching bias,
- successor count,
- continuity penalty,
- growth gain,
- anisotropy strength.

#### Special rule
For `diffuse_damage`, crack front should be disabled or bypassed.

---

### `src/fracture/gaussian_fracture_field.py`

#### Main role
Maintain local damage band around the front.

#### New requirement
Use family-aware local band rules.

#### Key family knobs
- band width,
- band fill,
- local shell thickness,
- core/halo balance.

This file should remain a local field module, not a global propagation driver.

---

### `src/visualization/gaussian_updater.py`

#### Main role
Convert solver state into visible fracture.

#### New requirement
Family-aware shell composition and saliency weighting.

#### Key graphics knobs
- core weight,
- tip weight,
- visited trace weight,
- opening weight,
- opacity reduction,
- split emphasis.

This file should express family differences strongly.

---

### `src/core/fragment_manager.py` or `src/fracture/edge_fragmentation.py`

#### Main role
Compute fragment separation from graph connectivity.

#### New requirement
Family-aware edge weakening.

#### Key family knobs
- edge break rate,
- edge weakening gain,
- split threshold.

This file is responsible for material-aware separation outcomes.

---

### `src/core/manifold_simulator.py`

#### Main role
Coordinate MPM coupling and fracture feedback schedule.

#### New requirement
Preserve delayed coupling while family-aware fracture evolves.

#### Why this still matters
Even after family-aware fracture is added, premature lower collapse can still kill useful propagation.

This file should continue to keep early feedback delayed.

---

### `src/pipeline/fracture_pipeline.py`

#### Main role
Orchestrate the full pipeline.

#### Recommended execution order
```text
1. CLIP retrieval
2. Material prior adaptation
3. Family selection / blending
4. MPM simulation
5. Crack initiation / propagation (if family supports it)
6. Local band update
7. Opening / split update
8. Edge weakening
9. Fragment labeling
10. Family-aware rendering
```

This causal order is important.

---

## 11. Recommended Experimental Plan

The experiments should now be designed around **family distinction**, not only generic fracture quality.

### Experiment Set A — Family separation under fixed geometry
Use:
- same object,
- same impact,
- same camera,
- same simulation length,
- only change material prompt.

#### Target prompts
- glass bunny
- ceramic bunny
- concrete bunny
- rubber bunny
- default / neutral

#### What to measure
- path count,
- main crack saliency,
- opening magnitude,
- split onset,
- fragment count,
- visible band width,
- visited trace visibility.

#### Expected results
- `glass-like` -> fewer but more dominant cracks, stronger opening
- `ceramic-like` -> moderate brittle crack
- `concrete-like` -> rougher and broader crack network
- `rubber-like` -> no explicit crack path
- `neutral` -> balanced comparison point

This should be the first major milestone.

---

### Experiment Set B — Baseline redefinition
Test:
- old baseline,
- new neutral baseline.

#### Goal
Verify that the baseline no longer behaves like hidden quasi-brittle concrete-lite.

#### Why it matters
A neutral baseline will make all material comparisons much easier to interpret in figures and demos.

---

### Experiment Set C — Top-k family blending
Use ambiguous prompts and compare:
- top-1 family selection,
- top-k weighted blending.

#### Goal
Check whether blended priors produce smoother and more believable transitions.

This is especially useful for semantically ambiguous objects.

---

### Experiment Set D — Fragment separation validation
For each family, measure:
- number of broken edges,
- number of connected components,
- timing of fragment split,
- visual separation quality.

#### Expected family pattern
- `sharp_brittle` -> earlier clean separation
- `rough_quasi_brittle` -> rougher multi-fragment behavior
- `diffuse_damage` -> little to no separation

---

### Experiment Set E — Ablation studies
Ablate the family-aware components one by one:

1. without family-aware crack front
2. without family-aware band
3. without family-aware shell rendering
4. without family-aware edge break
5. without neutral baseline reset

#### Goal
Demonstrate which component is responsible for which family distinction.

This is especially important for paper-quality argumentation.

---

## 12. Recommended Evaluation Table

A practical internal evaluation table can be:

| Prompt | Family | Path Count | Main Crack Saliency | Opening | Band Width | Fragment Count | Visual Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| glass bunny | sharp_brittle | low-mid | very high | high | thin | mid-high clean | good / bad |
| ceramic bunny | brittle_moderate | mid | high | mid | thin-mid | mid | good / bad |
| concrete bunny | rough_quasi_brittle | mid-high | mid | mid | broad | mid-high rough | good / bad |
| rubber bunny | diffuse_damage | ~0 | very low | very low | diffuse | ~0 | good / bad |
| default | neutral_reference | low-mid | mid-low | mid-low | mid | low-mid | good / bad |

This should be filled repeatedly during development.

---

## 13. What Not to Do

At this stage, avoid the following mistakes:

### Do not
- interpret CLIP labels as exact fracture truth,
- use one universal crack-shell rendering rule for all materials,
- keep rubber inside brittle crack mode and just suppress it,
- keep the baseline concrete-like,
- tune only physical stiffness parameters and expect family distinction to appear automatically,
- judge glass only by path count.

### Instead
- interpret CLIP output semantically,
- build a fracture family layer,
- make the family affect solver, band, shell, and fragmentation,
- and evaluate material distinction in graphics terms.

---

## 14. Final Roadmap Summary

The correct next implementation direction is:

### Phase 1
- add material prior adapter,
- introduce fracture families,
- separate semantic material interpretation from solver behavior.

### Phase 2
- fix rubber-like family as diffuse damage mode,
- strengthen glass-like family through main crack saliency,
- reset the baseline to neutral.

### Phase 3
- make crack front, local band, rendering shell, and fragment separation all family-aware.

### Phase 4
- run fixed-geometry material-comparison experiments,
- validate family separation,
- then tune parameters.

---

## 15. Final One-Paragraph Conclusion

From a graphics perspective, the system should now move away from treating CLIP material labels as exact physical material identities and instead use them as semantic priors that select a fracture behavior family. The implementation should therefore be structured around a material-prior adapter, family-aware crack-tip propagation, family-aware local band formation, family-aware shell rendering, and family-aware edge-based fragment separation. The immediate next steps should prioritize: (1) moving rubber-like prompts into a diffuse non-crack mode, (2) making glass-like fracture produce fewer but much more dominant and open cracks, and (3) redefining the current baseline as a truly neutral reference. Once those family distinctions are stable, parameter tuning and fragment experiments become meaningful and visually interpretable.
