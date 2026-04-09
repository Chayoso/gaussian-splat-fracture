"""
Entry point for the Gaussian-manifold fracture pipeline.

Two modes:
  1. Config mode (default): material params from YAML config
     python run_manifold.py --config configs/gravity_drop_manifold.yaml

  2. CLIP mode: material params from text description via CLIP
     python run_manifold.py --config configs/gravity_drop_manifold.yaml --clip "ceramic mug"

Examples:
    # Config mode — direct parameter control
    python run_manifold.py --config configs/gravity_drop_manifold.yaml
    python run_manifold.py --config configs/gravity_drop_manifold.yaml --E 1.5e7 --Gc 60000

    # CLIP mode — automatic material prediction
    python run_manifold.py --config configs/gravity_drop_manifold.yaml --clip "glass bottle"
    python run_manifold.py --config configs/gravity_drop_manifold.yaml --clip "ceramic mug" --clip-transformer

    # Fast mode
    python run_manifold.py --config configs/gravity_drop_manifold.yaml --fast-mode --frames 50
"""

import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian-manifold fracture simulation"
    )
    parser.add_argument(
        "--config",
        default="configs/gravity_drop_manifold.yaml",
        help="Path to config (must have simulation.use_manifold: true)",
    )
    parser.add_argument("--frames", type=int, default=None, help="Override total frames")
    parser.add_argument("--fast-mode", action="store_true", help="Fast mode (fewer particles)")
    parser.add_argument("--no-save-frames", action="store_true", help="Skip PNG/video output")
    parser.add_argument("--tag", default=None, help="Experiment tag for output dir")
    parser.add_argument("--experiment-dir", default=None, help="Explicit output directory")
    parser.add_argument("--E", type=float, default=None, help="Override Young's modulus")
    parser.add_argument("--Gc", type=float, default=None, help="Override fracture toughness")
    parser.add_argument("--nu", type=float, default=None, help="Override Poisson ratio")

    # CLIP mode
    parser.add_argument(
        "--clip", type=str, default=None, metavar="TEXT",
        help="Enable CLIP mode: predict material from text description "
             "(e.g., --clip 'ceramic mug')",
    )
    parser.add_argument(
        "--clip-transformer", action="store_true",
        help="Use transformer head for CLIP prediction (requires trained weights)",
    )
    parser.add_argument(
        "--clip-model", type=str, default="ViT-B/32",
        help="CLIP model variant (default: ViT-B/32)",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to material_db.json (default: data/material_db.json)",
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Only predict material params, do not run simulation",
    )

    args = parser.parse_args()

    # Determine material source
    material_source = "clip" if args.clip else "config"

    # Experiment directory
    config_stem = Path(args.config).stem
    if args.experiment_dir is not None:
        exp_dir = Path(args.experiment_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = f"_clip" if material_source == "clip" else ""
        suffix = f"_{args.tag}" if args.tag else ""
        exp_dir = Path("output") / "experiments" / f"{stamp}_{config_stem}{mode_tag}{suffix}"

    print(f"[Manifold Run] Mode: {material_source}")
    print(f"[Manifold Run] Config: {args.config}")
    print(f"[Manifold Run] Experiment directory: {exp_dir}")

    from src.pipeline.manifold_fracture_pipeline import ManifoldFracturePipeline

    pipeline = ManifoldFracturePipeline(
        config_path=args.config,
        material_source=material_source,
        fast_mode=args.fast_mode,
        db_path=args.db_path,
        clip_model=args.clip_model,
    )

    # Predict-only mode (CLIP)
    if args.predict_only:
        if material_source != "clip":
            print("Error: --predict-only requires --clip 'description'")
            return
        params = pipeline.predict_only(args.clip)
        print(f"\nPredicted material parameters:")
        print(f"  E  = {params['E']:.2e} Pa")
        print(f"  Gc = {params['Gc']:.1f} J/m²")
        print(f"  nu = {params['nu']:.3f}")
        print(f"  density = {params['density']:.0f} kg/m³")
        print(f"  Top-K: {params['top_k_names']}")
        return

    # Build overrides
    overrides = {
        "output.frame_dir": str(exp_dir / "frames"),
        "output.checkpoint_dir": str(exp_dir / "checkpoints"),
        "output.statistics_log": str(exp_dir / "statistics.csv"),
        "output.video_path": str(exp_dir / "manifold_fracture.mp4"),
    }

    # Override params (config mode) or additional overrides (CLIP mode)
    override_params = {}
    if args.E is not None:
        override_params["E"] = args.E
    if args.Gc is not None:
        override_params["Gc"] = args.Gc
    if args.nu is not None:
        override_params["nu"] = args.nu

    result = pipeline.run(
        text=args.clip,
        num_frames=args.frames,
        use_transformer=args.clip_transformer,
        save_frames=not args.no_save_frames,
        return_frames=False,
        override_params=override_params if override_params else None,
        **overrides,
    )

    print(f"\n[Done] Material source: {result['material_source']}")
    print(f"  Params: E={result['params']['E']:.2e}, Gc={result['params']['Gc']:.1f}")
    if 'top_k' in result:
        print(f"  Top-K: {result['top_k']}")


if __name__ == "__main__":
    main()
