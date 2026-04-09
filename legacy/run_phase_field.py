import argparse
from datetime import datetime
from pathlib import Path

from src.engine.forward_engine import ForwardEngine


def main():
    parser = argparse.ArgumentParser(
        description="Run the pure phase-field pipeline without touching legacy run.py"
    )
    parser.add_argument(
        "--config",
        default="configs/gravity_drop_pure_pf_100k.yaml",
        help="Path to pure phase-field config",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override total frame count",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use ForwardEngine fast mode for quicker debugging",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Do not save PNG frames or video",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional experiment tag appended to the auto-created output directory",
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Explicit experiment output directory. If omitted, create output/experiments/<timestamp>_<config>[_tag]",
    )
    parser.add_argument("--E", type=float, default=None, help="Override Young's modulus")
    parser.add_argument("--Gc", type=float, default=None, help="Override fracture toughness")
    parser.add_argument("--nu", type=float, default=None, help="Override Poisson ratio")
    args = parser.parse_args()

    config_stem = Path(args.config).stem
    if args.experiment_dir is not None:
        exp_dir = Path(args.experiment_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{args.tag}" if args.tag else ""
        exp_dir = Path("output") / "experiments" / f"{stamp}_{config_stem}{suffix}"

    overrides = {
        "output.frame_dir": str(exp_dir / "frames"),
        "output.checkpoint_dir": str(exp_dir / "checkpoints"),
        "output.statistics_log": str(exp_dir / "statistics.csv"),
        "output.video_path": str(exp_dir / "gravity_drop_pure_pf.mp4"),
    }

    print(f"[PF Run] Experiment directory: {exp_dir}")

    engine = ForwardEngine(args.config, fast_mode=args.fast_mode)
    engine.simulate(
        E=args.E,
        Gc=args.Gc,
        nu=args.nu,
        num_frames=args.frames,
        save_frames=not args.no_save_frames,
        return_frames=False,
        **overrides,
    )


if __name__ == "__main__":
    main()
