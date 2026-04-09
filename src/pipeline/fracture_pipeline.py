"""
FracturePipeline — text-to-fracture end-to-end.

Usage:
    pipeline = FracturePipeline("configs/gravity_drop_test.yaml")

    # Forward only (KNN)
    frames = pipeline.run("ceramic mug", num_frames=100)

    # With transformer
    frames = pipeline.run("ceramic mug", num_frames=100, use_transformer=True)
"""

import torch
from typing import List, Optional

from src.engine.forward_engine import ForwardEngine
from src.ml.material_predictor import MaterialPredictor

# ---- Physical → MPM parameter scaling ----
# MPM operates in normalized [0,1]³ space. Physical Gc/E values
# must be scaled to match. Reference: concrete at MPM Gc=60000, E=1.5e7
# maps to physical Gc=100 J/m², E=3e10 Pa.
MPM_GC_BASE = 60000.0     # MPM Gc for reference material
MPM_E_BASE = 1.5e7        # MPM E (fixed for all materials)
PHYS_GC_REF = 100.0       # Physical Gc of reference (concrete)
PHYS_E_REF = 3.0e10       # Physical E of reference (concrete)

# Material category → RGB tint for rendering
MATERIAL_TINTS = {
    "ice": (0.70, 0.85, 1.00),        # icy blue
    "glass": (0.70, 0.85, 1.00),      # clear blue
    "ceramic": (1.00, 0.88, 0.75),    # warm terracotta
    "concrete": (0.65, 0.65, 0.65),   # dark gray
    "stone": (0.70, 0.65, 0.60),      # warm brown-gray
    "metal": (0.75, 0.80, 0.85),      # cool metallic
    "wood": (0.75, 0.55, 0.35),       # warm brown
    "polymer": (0.90, 0.90, 0.80),    # yellowish white
    "biological": (0.90, 0.82, 0.72), # bone/ivory
    "food": (0.65, 0.45, 0.25),       # dark brown
    "composite": (0.25, 0.25, 0.25),  # very dark
    "other": (0.80, 0.85, 0.90),      # light blue-gray
}


class FracturePipeline:
    """End-to-end text → material params → fracture simulation."""

    def __init__(self, config_path: str, fast_mode: bool = False,
                 db_path: str = None, clip_model: str = "ViT-B/32",
                 transformer_checkpoint: str = None):
        """
        Args:
            config_path: Base simulation config YAML
            fast_mode: Reduce particles/resolution for faster iteration
            db_path: Path to material_db.json (default: data/material_db.json)
            clip_model: CLIP variant (ViT-B/32 recommended)
            transformer_checkpoint: Optional trained transformer weights
        """
        print(f"\n{'='*60}")
        print(f"Initializing Fracture Pipeline")
        print(f"{'='*60}")

        self.engine = ForwardEngine(config_path, fast_mode=fast_mode)

        self.predictor = MaterialPredictor(
            mode="clip_knn",
            db_path=db_path,
            clip_model=clip_model,
            transformer_checkpoint=transformer_checkpoint,
        )

        self._transformer_checkpoint = transformer_checkpoint
        print(f"{'='*60}")
        print(f"Pipeline ready!")
        print(f"{'='*60}\n")

    def run(self, text: str, num_frames: int = 100,
            use_transformer: bool = False,
            save_frames: bool = True,
            return_frames: bool = True,
            override_params: dict = None,
            **kwargs) -> dict:
        """
        Run full pipeline: text → material params → simulation.

        Args:
            text: Material description (e.g. "ceramic mug", "glass bottle")
            num_frames: Number of simulation frames
            use_transformer: Use transformer head instead of KNN
            save_frames: Save PNG frames to disk
            return_frames: Return frame tensors
            override_params: Override predicted params (e.g. {"Gc": 50})
            **kwargs: Additional ForwardEngine.simulate() args

        Returns:
            dict with:
                "params": predicted material parameters
                "frames": list of (3,H,W) tensors (if return_frames)
                "top_k": top-K similar materials from DB
        """
        # Step 1: Predict material parameters
        if use_transformer and self._transformer_checkpoint:
            self.predictor.mode = "clip_transformer"
        else:
            self.predictor.mode = "clip_knn"

        params = self.predictor.predict(text)

        print(f"\n[Pipeline] Material: '{text}'")
        print(f"  Physical: E={params['E']:.2e}, Gc={params['Gc']:.1f}, "
              f"nu={params['nu']:.3f}, density={params['density']:.0f}")
        print(f"  Top-K: {params['top_k_names']}")
        print(f"  Scores: {[f'{s:.3f}' for s in params['top_k_scores']]}")

        # Scale physical params → MPM normalized space
        # E: fixed at MPM_E_BASE (stiffness ratio handled by Gc scaling)
        # Gc: scale proportionally from reference
        # nu: direct (dimensionless, same in both spaces)
        params["E_physical"] = params["E"]
        params["Gc_physical"] = params["Gc"]
        params["E"] = MPM_E_BASE
        params["Gc"] = MPM_GC_BASE * (params["Gc_physical"] / PHYS_GC_REF)
        print(f"  MPM scaled: E={params['E']:.2e}, Gc={params['Gc']:.1f}")

        # Apply overrides if provided (overrides are in MPM space)
        if override_params:
            for key, val in override_params.items():
                if key in params:
                    params[key] = val
            print(f"  Overrides: {override_params}")

        # Determine material category for texture
        top_entry = self.predictor.db[
            self.predictor.db.get_names().index(params["top_k_names"][0])]
        mat_category = top_entry.category
        print(f"  Texture: {mat_category}")

        # Step 2: Run simulation with material texture
        frames = self.engine.simulate(
            E=params["E"],
            Gc=params["Gc"],
            nu=params["nu"],
            num_frames=num_frames,
            save_frames=save_frames,
            return_frames=return_frames,
            material_texture=mat_category,
            **kwargs
        )

        return {
            "params": params,
            "frames": frames,
            "top_k": params.get("top_k_names", []),
        }

    def predict_only(self, text: str) -> dict:
        """Predict material parameters without running simulation."""
        return self.predictor.predict(text)
