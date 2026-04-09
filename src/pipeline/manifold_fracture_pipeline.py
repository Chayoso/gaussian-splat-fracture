"""
ManifoldFracturePipeline — Unified entry point for Gaussian-manifold fracture.

Two modes:
  1. CLIP mode (material_source="clip"):
     text → CLIP → MaterialPredictor → (E, Gc, nu) → ManifoldSimulator
  2. Config mode (material_source="config"):
     YAML config → (E, Gc, nu) directly → ManifoldSimulator

Both modes use the ManifoldSimulator (Gaussian-manifold fracture field).
The CLIP mode adds automatic material parameter prediction from text.
Config mode is for direct parameter control and simulation verification.

Usage:
    # CLIP mode
    pipeline = ManifoldFracturePipeline("configs/gravity_drop_manifold.yaml", material_source="clip")
    result = pipeline.run("ceramic mug", num_frames=100)

    # Config mode (default)
    pipeline = ManifoldFracturePipeline("configs/gravity_drop_manifold.yaml", material_source="config")
    result = pipeline.run(num_frames=100)
"""

import torch
from typing import List, Optional
from pathlib import Path

from src.engine.forward_engine import ForwardEngine


# Physical → MPM parameter scaling (same as fracture_pipeline.py)
MPM_GC_BASE = 60000.0
MPM_E_BASE = 1.5e7
PHYS_GC_REF = 100.0    # physical Gc of reference (concrete, J/m²)

# Material category → RGB tint
MATERIAL_TINTS = {
    "ice": (0.70, 0.85, 1.00),
    "glass": (0.70, 0.85, 1.00),
    "ceramic": (1.00, 0.88, 0.75),
    "concrete": (0.65, 0.65, 0.65),
    "stone": (0.70, 0.65, 0.60),
    "metal": (0.75, 0.80, 0.85),
    "wood": (0.75, 0.55, 0.35),
    "polymer": (0.90, 0.90, 0.80),
    "biological": (0.90, 0.82, 0.72),
    "food": (0.65, 0.45, 0.25),
    "composite": (0.25, 0.25, 0.25),
    "other": (0.80, 0.85, 0.90),
}


class ManifoldFracturePipeline:
    """
    Unified pipeline for Gaussian-manifold fracture simulation.

    Supports two material parameter sources:
      - "clip": CLIP-based material prediction from text
      - "config": Direct parameters from YAML config
    """

    def __init__(
        self,
        config_path: str,
        material_source: str = "config",
        fast_mode: bool = False,
        db_path: str = None,
        clip_model: str = "ViT-B/32",
        transformer_checkpoint: str = None,
    ):
        """
        Args:
            config_path: Path to simulation config YAML
                         (must have simulation.use_manifold: true)
            material_source: "config" or "clip"
            fast_mode: Reduce resolution for faster iteration
            db_path: Path to material_db.json (CLIP mode only)
            clip_model: CLIP model variant (CLIP mode only)
            transformer_checkpoint: Trained transformer weights (CLIP mode only)
        """
        self.material_source = material_source
        self.config_path = config_path

        print(f"\n{'='*60}")
        print(f"Manifold Fracture Pipeline")
        print(f"{'='*60}")
        print(f"  Material source: {material_source}")
        print(f"  Config: {config_path}")

        # Always create the engine (caches mesh/particles)
        self.engine = ForwardEngine(config_path, fast_mode=fast_mode)

        # CLIP predictor (lazy-loaded only when needed)
        self._predictor = None
        self._clip_model = clip_model
        self._db_path = db_path
        self._transformer_checkpoint = transformer_checkpoint

        if material_source == "clip":
            self._init_clip_predictor()

        print(f"{'='*60}")
        print(f"Pipeline ready! (mode={material_source})")
        print(f"{'='*60}\n")

    def _init_clip_predictor(self):
        """Lazy-initialize CLIP material predictor."""
        if self._predictor is not None:
            return

        from src.ml.material_predictor import MaterialPredictor

        self._predictor = MaterialPredictor(
            mode="clip_knn",
            db_path=self._db_path,
            clip_model=self._clip_model,
            transformer_checkpoint=self._transformer_checkpoint,
        )

    def run(
        self,
        text: str = None,
        num_frames: int = None,
        use_transformer: bool = False,
        save_frames: bool = True,
        return_frames: bool = True,
        override_params: dict = None,
        **kwargs,
    ) -> dict:
        """
        Run simulation.

        Args:
            text: Material description (required for CLIP mode, ignored for config mode)
            num_frames: Override frame count
            use_transformer: Use transformer head (CLIP mode only)
            save_frames: Save PNG frames
            return_frames: Return frame tensors
            override_params: Override material params (MPM-space)
            **kwargs: Additional ForwardEngine.simulate() kwargs

        Returns:
            dict with:
                "params": material parameters used
                "frames": rendered frame tensors (if return_frames)
                "material_source": "clip" or "config"
                "top_k": top-K materials (CLIP mode only)
        """
        if self.material_source == "clip":
            return self._run_clip(
                text=text,
                num_frames=num_frames,
                use_transformer=use_transformer,
                save_frames=save_frames,
                return_frames=return_frames,
                override_params=override_params,
                **kwargs,
            )
        else:
            return self._run_config(
                num_frames=num_frames,
                save_frames=save_frames,
                return_frames=return_frames,
                override_params=override_params,
                **kwargs,
            )

    def _run_config(
        self,
        num_frames: int = None,
        save_frames: bool = True,
        return_frames: bool = True,
        override_params: dict = None,
        **kwargs,
    ) -> dict:
        """Run with parameters from YAML config (no CLIP)."""
        E = override_params.get("E") if override_params else None
        Gc = override_params.get("Gc") if override_params else None
        nu = override_params.get("nu") if override_params else None

        params = {
            "E": E or self.engine.base_config.material.youngs_modulus,
            "Gc": Gc or self.engine.base_config.material.Gc,
            "nu": nu or self.engine.base_config.material.poissons_ratio,
        }
        print(f"\n[Pipeline:config] E={params['E']:.2e}, Gc={params['Gc']:.1f}, nu={params['nu']:.3f}")

        frames = self.engine.simulate(
            E=E, Gc=Gc, nu=nu,
            num_frames=num_frames,
            save_frames=save_frames,
            return_frames=return_frames,
            **kwargs,
        )

        return {
            "params": params,
            "frames": frames,
            "material_source": "config",
        }

    def _run_clip(
        self,
        text: str = None,
        num_frames: int = None,
        use_transformer: bool = False,
        save_frames: bool = True,
        return_frames: bool = True,
        override_params: dict = None,
        **kwargs,
    ) -> dict:
        """Run with CLIP-predicted material parameters."""
        if text is None:
            raise ValueError("text is required for CLIP mode. "
                             "Use material_source='config' for manual params.")

        self._init_clip_predictor()

        if use_transformer and self._transformer_checkpoint:
            self._predictor.mode = "clip_transformer"
        else:
            self._predictor.mode = "clip_knn"

        # Predict material params
        params = self._predictor.predict(text)

        print(f"\n[Pipeline:clip] Material: '{text}'")
        print(f"  Physical: E={params['E']:.2e}, Gc={params['Gc']:.1f}, "
              f"nu={params['nu']:.3f}, density={params['density']:.0f}")
        print(f"  Top-K: {params['top_k_names']}")

        # Scale to MPM space
        params["E_physical"] = params["E"]
        params["Gc_physical"] = params["Gc"]
        params["E"] = MPM_E_BASE
        params["Gc"] = MPM_GC_BASE * (params["Gc_physical"] / PHYS_GC_REF)
        print(f"  MPM scaled: E={params['E']:.2e}, Gc={params['Gc']:.1f}")

        # Apply overrides
        if override_params:
            for key, val in override_params.items():
                if key in params:
                    params[key] = val
            print(f"  Overrides: {override_params}")

        # Material texture from top-1 category
        top_entry = self._predictor.db[
            self._predictor.db.get_names().index(params["top_k_names"][0])]
        mat_category = top_entry.category
        print(f"  Texture: {mat_category}")

        # Run simulation
        frames = self.engine.simulate(
            E=params["E"],
            Gc=params["Gc"],
            nu=params["nu"],
            num_frames=num_frames,
            save_frames=save_frames,
            return_frames=return_frames,
            material_texture=mat_category,
            **kwargs,
        )

        return {
            "params": params,
            "frames": frames,
            "material_source": "clip",
            "top_k": params.get("top_k_names", []),
            "material_category": mat_category,
        }

    def predict_only(self, text: str) -> dict:
        """Predict material parameters without running simulation (CLIP mode only)."""
        if self.material_source != "clip":
            raise ValueError("predict_only requires material_source='clip'")
        self._init_clip_predictor()
        return self._predictor.predict(text)
