# Physics-Informed Gaussian-Manifold Fracture Pipeline
## CLIP Material Prior + MPM Physics + Gaussian Fracture Field + 3DGS Rendering

---

## 1. Goal

We want a pipeline that integrates:

1. **CLIP-based material understanding** from text (and optionally image cues),
2. **MPM-based physical simulation** for deformation and impact response,
3. **Gaussian-manifold fracture evolution** for renderer-native crack propagation,
4. **3D Gaussian Splatting (3DGS)** for visible fracture rendering.

The main design objective is:

> Keep physics as the source of fracture-driving signals,  
> but move fracture state evolution and visible crack behavior into the Gaussian domain itself.

This avoids forcing a volumetric fracture field to remain the only visible truth when the final representation is already Gaussian-native.

---

## 2. Current Pipeline Summary

The current system is roughly:

```text
Config (YAML)
  -> Mesh loading
  -> Dual point cloud generation (volume + surface)
  -> MPM initialization
  -> Gaussian initialization
  -> Frame loop:
       1. MPM P2G2P update
       2. AT2 phase-field damage evolution
       3. Volume-to-surface damage projection
       4. Gaussian appearance update
       5. Fragment detection (optional)
       6. Rendering
  -> Video generation
```

### Current major modules
- `config/`
- `preprocessing/`
- `mpm_core/`
- `constitutive_models/phase_field.py`
- `constitutive_models/damage_mapper.py`
- `core/fragment_manager.py`
- `visualization/gaussian_updater.py`
- `ml/material_predictor.py`
- `pipeline/fracture_pipeline.py`

---

## 3. Why the Current Pipeline Has Structural Problems

The current bottleneck is not just tuning.  
It is a **representation mismatch**.

### 3.1 Fracture is solved in volume, but displayed in Gaussian space
AT2 phase-field naturally lives in a volumetric domain.
However, 3DGS is a rendering representation, not a volumetric fracture representation.

So the current logic becomes:

```text
volumetric damage field
    -> projected to surface / Gaussian domain
    -> interpreted as visible crack
```

This causes ambiguity.

### 3.2 Projection creates artifact opportunities
Even if the internal damage field is physically plausible, the visible crack can still be wrong because of:

- incorrect surface association,
- KNN or nearest-neighbor mapping error,
- coordinate mismatch,
- lack of crack opening information,
- over-reliance on scalar damage only.

As a result, a crack that should visually emerge near the impact/contact region may instead appear elsewhere.

### 3.3 Current phase-field behavior is already close to regularized damage propagation
In practice, the implemented update often behaves like:

```text
physics signal -> damage activation -> local spreading -> rendered crack-like region
```

This means the current system is already conceptually close to:

- local damage activation,
- neighborhood propagation,
- regularized smoothing,
- threshold-based interpretation.

That makes it a good candidate for migration into a graph-based Gaussian domain.

### 3.4 Scalar damage alone is insufficient
A realistic visible crack requires more than a single damage scalar.
It needs:

- damage magnitude,
- crack direction,
- opening amount,
- connectivity break,
- fragment separation.

A projected scalar damage map can darken a region, but it cannot fully explain visible fracture opening.

---

## 4. Why Gaussian-Manifold Fracture is a Better Fit

The key idea is simple:

> If the final object is represented and rendered as Gaussians,  
> then fracture should also evolve in the Gaussian domain.

### Main advantages

#### 4.1 Same domain for fracture and rendering
No more forced volume-to-surface crack translation.

#### 4.2 Better conceptual consistency
The fracture field is no longer hidden in one representation and displayed in another.

#### 4.3 Easier visible crack control
Gaussian-native fracture can directly drive:

- opacity attenuation,
- covariance deformation,
- Gaussian splitting,
- crack opening,
- fragment-wise motion.

#### 4.4 Better debugging
When artifacts happen, the issue is easier to localize:
- graph propagation,
- opening rule,
- split logic,
- fragment connectivity.

#### 4.5 Better integration with semantic priors
Gaussian-wise material priors from CLIP can directly influence crack behavior.

---

## 5. Core New Design

The recommended architecture is:

```text
Text / Prompt / Optional Image
    -> CLIP Material Predictor
    -> Material Prior Assignment
    -> Mesh / Point Cloud / Gaussian Initialization
    -> MPM Physics Update
    -> Physics-to-Gaussian Signal Projection
    -> Gaussian-Manifold Fracture Evolution
    -> Gaussian Split / Opening / Fragment Update
    -> 3DGS Rendering
```

---

## 6. Division of Responsibility

### 6.1 CLIP / Material side
Responsible for:
- semantic material understanding,
- retrieving likely material properties,
- producing fracture-related priors.

### 6.2 MPM side
Responsible for:
- deformation,
- impact/contact response,
- strain/tensile energy,
- dynamic motion.

### 6.3 Gaussian fracture side
Responsible for:
- damage state,
- local graph propagation,
- irreversibility,
- crack direction,
- opening magnitude,
- fragment connectivity.

### 6.4 3DGS rendering side
Responsible for:
- visible crack cues,
- opacity reduction,
- Gaussian split/opening,
- fragment-aware visual updates.

---

## 7. New Pipeline in Detail

## Stage A. Input and material prior estimation

### A.1 Input
Possible inputs:
- text prompt,
- optional image cues,
- optional object part segmentation,
- optional mesh/object metadata.

### A.2 CLIP-based material prediction
The current CLIP pipeline can be reused:

- `ml/clip_encoder.py`
- `ml/material_db.py`
- `ml/material_predictor.py`

The predictor should output not only classical physical parameters but also fracture/render priors.

### Recommended output format
```python
material_prior = {
    "E": ...,
    "nu": ...,
    "Gc": ...,
    "rho": ...,
    "brittleness": ...,
    "damage_spread": ...,
    "split_sensitivity": ...,
    "open_gain": ...,
    "anisotropy_strength": ...,
}
```

### Interpretation
- `E`, `nu`, `Gc`, `rho`: physics-side material parameters
- `brittleness`, `damage_spread`, `split_sensitivity`, `open_gain`, `anisotropy_strength`: fracture/render priors

These do not all need to be strictly physical constants.
Some can be treated as learned or heuristic semantic priors.

---

## Stage B. Representation initialization

### B.1 Geometry initialization
From mesh / point cloud:
- initialize MPM particles,
- initialize Gaussian splats,
- optionally initialize region labels.

### B.2 Material assignment
Material priors can be assigned at different levels:

#### Object-level
All Gaussians share one material prior.

#### Region-level
Different Gaussian subsets get different priors.

#### Gaussian-level soft assignment
Each Gaussian receives a mixture over candidate materials.

Example:
```text
Gaussian i:
  0.6 ceramic
  0.3 plaster
  0.1 concrete
```

This yields soft material vectors per Gaussian.

---

## Stage C. MPM physics update

The MPM engine remains in charge of physical response.

### Role of MPM
- update deformation,
- compute stress/strain,
- simulate impact/contact,
- produce fracture-driving signals.

### Candidate fracture-driving signals
Examples:
- tensile strain energy,
- principal stretch,
- impact impulse,
- stress concentration,
- strain rate.

These signals do not directly define visible cracks.
Instead, they serve as **physics-informed fracture activation signals**.

---

## Stage D. Physics-to-Gaussian projection

This is the bridge between MPM and the Gaussian fracture field.

### Purpose
Map physics quantities from particles/grid to Gaussians.

### Suggested module rename
Instead of `damage_mapper.py`, use something like:

- `physics_to_gaussian_projector.py`

### Output example
For each Gaussian `i`:
```python
physics_signal_i = {
    "drive": ...,
    "principal_dir": ...,
    "deformation": ...,
}
```

### Notes
This stage is **not** a volume-to-surface damage truth conversion anymore.
It is only a projection of physics cues into the Gaussian domain.

That is a major conceptual improvement.

---

## Stage E. Gaussian-manifold fracture evolution

This is the new core.

### E.1 Gaussian fracture state
For each Gaussian `i`, maintain:

```python
state_i = {
    "c": ...,      # damage magnitude
    "H": ...,      # irreversible history / accumulated drive
    "n": ...,      # crack normal or dominant fracture direction
    "a": ...,      # opening magnitude
    "f": ...,      # fragment label
}
```

### Meaning
- `c`: how damaged the Gaussian is
- `H`: max-accumulated local fracture drive
- `n`: used for split/opening direction
- `a`: visible crack separation amount
- `f`: fragment membership

---

### E.2 Gaussian graph construction
Construct a neighborhood graph over Gaussians using:
- kNN,
- or radius-based neighbors.

Edge weights can depend on:
- Euclidean distance,
- covariance overlap,
- normal compatibility,
- local deformation alignment,
- material similarity.

---

### E.3 History update
Irreversibility should be preserved.

For each Gaussian:
```text
H_i <- max(H_i, drive_i)
```

This is analogous to history accumulation in phase-field fracture.

---

### E.4 Damage update
A simple graph-based update can be:

```text
c_i^{t+1} = max(
    c_i^t,
    c_i^t + eta_i * drive_i + lambda_i * sum_j w_ij (c_j^t - c_i^t)
)
```

### Interpretation
- `eta_i`: material-aware activation gain
- `drive_i`: physics-informed fracture source
- `lambda_i`: local propagation / smoothing strength
- `w_ij`: graph coupling weights
- `max(...)`: irreversibility

This reflects the spirit of phase-field-like regularized fracture propagation while remaining renderer-native.

---

### E.5 Directionality / anisotropy
To avoid purely radial blur, propagation must include directional preference.

Possible directional cues:
- principal stress direction from MPM,
- graph damage gradient,
- local deformation direction,
- material anisotropy prior.

This is especially important for producing crack-like patterns instead of stain-like diffusion.

---

### E.6 Opening magnitude
Visible crack opening should depend on:
- damage level,
- local displacement difference,
- crack normal,
- material prior.

A simple conceptual form:
```text
a_i = open_gain_i * opening_measure_i * c_i
```

If a Gaussian is highly damaged and the local relative motion is strong, it should visibly separate.

---

### E.7 Edge weakening and fragment formation
As damage grows, graph connectivity should weaken.

Example concept:
```text
edge_strength_ij = exp(-beta_ij * max(c_i, c_j))
```

or:
```text
edge_strength_ij = (1 - c_i)(1 - c_j)
```

When graph connectivity breaks, connected components define fragments.

This allows:
- fragment labels,
- fragment-wise transforms,
- more realistic visible separation.

---

## Stage F. Gaussian visual fracture update

This stage directly modifies renderable Gaussian attributes.

### Suggested module
- `visualization/gaussian_updater.py`

### Inputs
- damage `c_i`
- crack normal `n_i`
- opening `a_i`
- fragment label `f_i`
- optional material priors

### Operations
Possible updates include:

#### Opacity attenuation
More damage -> less opacity

#### Covariance flattening
Shape Gaussians to emphasize fracture surfaces

#### Gaussian split
If damage and opening are sufficiently high:
- duplicate Gaussian,
- move copies along `± n_i`,
- reduce overlap.

#### Fragment-wise transform
Apply fragment-specific motion or decoupled updates after connectivity breaks.

This stage is what turns the fracture field into a visible crack.

---

## Stage G. Rendering

Rendering remains 3DGS-based.

However, now the renderer receives fracture-aware Gaussian states directly, instead of trying to infer cracks from a projected scalar damage map.

This is the core advantage of the new design.

---

## 8. How CLIP Material Properties Fit Into This

Yes — the existing CLIP material pipeline can be integrated naturally.

### Existing role
Currently, CLIP material prediction likely provides:
- `E`
- `nu`
- `Gc`

### Expanded role
In the new architecture, it can additionally provide:
- brittleness prior,
- damage spread prior,
- split sensitivity,
- open gain,
- anisotropy bias.

That means CLIP no longer only initializes volumetric constitutive parameters.
It also shapes visible fracture behavior in Gaussian space.

---

## 9. Example Mapping from Material Priors to Fracture Behavior

| Material prior | Gaussian fracture effect |
|---|---|
| low `Gc` | damage grows more easily |
| high brittleness | crack opens earlier, split threshold lower |
| high damage spread | broader local propagation |
| low damage spread | sharper crack band |
| high open gain | stronger visible separation |
| anisotropy strength | more directional crack propagation |

Examples:

- **ceramic** -> sharp cracks, fast split, brittle opening
- **glass** -> abrupt fracture, high separation sensitivity
- **rubber** -> broad damage, less opening
- **concrete** -> rough fragmented crack pattern
- **wood** -> direction-biased propagation

This is where semantic material understanding becomes especially powerful.

---

## 10. Recommended Refactor of Current Project Structure

### Current structure
```text
src/
├── config/
├── preprocessing/
├── mpm_core/
├── constitutive_models/
├── core/
├── engine/
├── visualization/
├── rendering/
├── ml/
└── pipeline/
```

### Suggested future structure
```text
src/
├── config/
│   └── fracture_gaussian.yaml

├── preprocessing/
│   ├── mesh_loader.py
│   ├── gaussian_initializer.py
│   └── region_labeler.py

├── mpm_core/
│   ├── mpm_model.py
│   ├── mpm_pipeline.py
│   └── set_boundary_conditions.py

├── ml/
│   ├── clip_encoder.py
│   ├── material_db.py
│   ├── material_predictor.py
│   └── material_prior_adapter.py

├── projection/
│   └── physics_to_gaussian_projector.py

├── gaussian_fracture/
│   ├── gaussian_graph.py
│   ├── gaussian_fracture_field.py
│   ├── crack_direction.py
│   ├── opening_estimator.py
│   └── edge_fragmentation.py

├── core/
│   ├── hybrid_simulator.py
│   └── fragment_manager.py

├── visualization/
│   └── gaussian_updater.py

├── rendering/
│   ├── gaussian_renderer.py
│   └── video_export.py

├── engine/
│   └── forward_engine.py

└── pipeline/
    └── fracture_pipeline.py
```

---

## 11. Suggested Role of Each New/Modified Module

### `ml/material_prior_adapter.py`
Converts CLIP retrieval output into:
- physical parameters,
- fracture behavior priors.

### `projection/physics_to_gaussian_projector.py`
Projects physics cues from MPM into Gaussian nodes.

### `gaussian_fracture/gaussian_graph.py`
Builds Gaussian neighborhood graph and edge weights.

### `gaussian_fracture/gaussian_fracture_field.py`
Maintains:
- damage,
- history,
- graph propagation,
- irreversibility.

### `gaussian_fracture/crack_direction.py`
Estimates crack normals / directional propagation cues.

### `gaussian_fracture/opening_estimator.py`
Computes visible opening magnitude.

### `gaussian_fracture/edge_fragmentation.py`
Weakens graph connectivity and extracts fragments.

### `visualization/gaussian_updater.py`
Applies visual fracture operations to 3DGS.

### `core/hybrid_simulator.py`
Orchestrates:
- MPM update
- signal projection
- Gaussian fracture evolution
- Gaussian visual update
- rendering

---

## 12. Recommended Frame Loop

```text
For each frame:
    1. Run MPM substeps
    2. Compute fracture-driving physics signals
    3. Project signals to Gaussian nodes
    4. Update Gaussian history H_i
    5. Update Gaussian damage c_i on graph
    6. Estimate crack direction n_i
    7. Estimate opening a_i
    8. Update graph connectivity / fragments
    9. Update Gaussian rendering state
   10. Render frame
```

This is much cleaner than:
- volume fracture solve,
- scalar damage projection,
- crack inference in rendering.

---

## 13. Why This Architecture is Strong

### Conceptually
It aligns:
- representation,
- fracture state,
- rendering.

### Practically
It reduces the number of fragile translation steps.

### Semantically
It gives CLIP material priors a richer downstream role.

### Visually
It allows visible crack opening and fragmentation to be renderer-native.

### Research-wise
It supports a compelling story:

> CLIP provides material semantics,  
> MPM provides physics,  
> and the Gaussian manifold provides fracture evolution and visible crack rendering.

---

## 14. Final Recommendation

The recommended architecture is:

> **CLIP material prior -> MPM physics drive -> Gaussian-manifold fracture evolution -> 3DGS rendering**

This should be treated as a:

- physics-informed,
- phase-field-inspired,
- Gaussian-native fracture framework.

Not as a strict replacement for classical continuum volumetric AT2 fracture,
but as a better fit for the actual representational and rendering structure of the current project.

---

## 15. One-Sentence Summary

Instead of solving fracture in volume and later trying to make it look correct in 3DGS,  
use CLIP to initialize material-aware fracture priors, use MPM to provide physics-driven activation, and let the Gaussian manifold itself become the native domain where fracture evolves and becomes visible.
