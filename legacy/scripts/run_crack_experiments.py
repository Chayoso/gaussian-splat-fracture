"""Run reproducible medium-scale crack/fracture experiment sweeps."""

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run import (
    load_config,
    setup_mesh,
    setup_mpm,
    setup_loading,
    setup_elasticity,
    setup_gaussians,
    setup_simulator,
)
from src.config.loader import apply_overrides_dict
from src.core.material_presets import resolve_material_preset, validate_l0
from src.engine.loading_transforms import apply_loading_transforms
from scripts.visualize_crack_heatmap import visualize_crack_heatmap


DEFAULT_CASES = {
    "baseline": {},
    "longer_paths": {
        "phase_field.max_nucleation_per_frame": 2,
        "phase_field.branch_probability": 0.60,
        "phase_field.crack_tip_speed": 18.0,
        "phase_field.material_rate_scale": 1.20,
        "phase_field.surface_band_depth_max": 0.10,
    },
    "shatter_bias": {
        "phase_field.max_nucleation_per_frame": 3,
        "phase_field.branch_probability": 0.80,
        "phase_field.crack_tip_speed": 17.0,
        "phase_field.material_rate_scale": 1.10,
        "phase_field.surface_band_depth_max": 0.12,
        "phase_field.fragment_damage_threshold": 0.35,
        "phase_field.fragment_cgrid_threshold": 0.35,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run medium-scale crack experiment sweep")
    parser.add_argument("--config", default="configs/gravity_drop_test.yaml")
    parser.add_argument("--output-dir", default="output/experiments/crack_sweep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--particles", type=int, default=30000)
    parser.add_argument("--grids", type=int, default=64)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--at2-iters", type=int, default=4)
    parser.add_argument("--fixed-gravity-z", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["baseline", "longer_paths", "shatter_bias"],
        choices=sorted(DEFAULT_CASES.keys()),
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(base_config_path: str, args, case_name: str):
    config = load_config(base_config_path)
    config = resolve_material_preset(config)
    config = validate_l0(config)

    overrides = {
        "particles.target_count": args.particles,
        "mpm.num_grids": args.grids,
        "rendering.total_frames": args.frames,
        "rendering.physics_substeps": args.substeps,
        "phase_field.at2_iters_per_substep": args.at2_iters,
        "simulation.save_frames": False,
        "simulation.save_checkpoint": False,
        "output.verbose": True,
    }
    if args.device is not None:
        overrides["device.type"] = args.device

    overrides.update(DEFAULT_CASES[case_name])
    config = apply_overrides_dict(config, overrides)
    return config


def ensure_device(config) -> torch.device:
    device_type = config.device.type
    if device_type == "cuda" and not torch.cuda.is_available():
        device_type = "cpu"
    device = torch.device(device_type)
    if device_type == "cuda" and hasattr(config.device, "gpu_id"):
        torch.cuda.set_device(config.device.gpu_id)
    return device


def run_case(case_name: str, args, output_dir: Path):
    config = build_config(args.config, args, case_name)
    device = ensure_device(config)

    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    volume_pcd, surface_pcd, surface_mask, mesh_meta = setup_mesh(config)
    mpm_model = setup_mpm(config, volume_pcd, device)
    loading_params = setup_loading(config, mpm_model, device)
    if args.fixed_gravity_z is not None:
        mpm_model.gravity = torch.tensor([0.0, 0.0, float(args.fixed_gravity_z)], device=device)
        print(f"[Experiment] Overrode gravity_z to {float(args.fixed_gravity_z):.1f}")
    elasticity = setup_elasticity(config, device)
    gaussians = setup_gaussians(config, surface_pcd, device, mesh_meta=mesh_meta)
    simulator = setup_simulator(
        config, mpm_model, gaussians, elasticity,
        surface_mask, device, loading_params=loading_params
    )

    if volume_pcd.normals is not None:
        all_normals = torch.from_numpy(np.asarray(volume_pcd.normals)).float().to(device)
        simulator.visualizer.set_initial_normals(all_normals)

    simulator.initialize(torch.from_numpy(np.asarray(volume_pcd.points)).float().to(device))
    apply_loading_transforms(config, simulator, loading_params, device)

    records = []
    frag_ckpt = None
    final_ckpt = case_dir / "final_state.pt"

    for frame in range(config.rendering.total_frames):
        simulator.step_rendering()
        path_count = len(getattr(simulator, "crack_paths", []))
        path_points = int(sum(int(p.shape[0]) for p in getattr(simulator, "crack_paths", []) if p is not None))
        frag_count = int(getattr(simulator.fragment_manager, "n_fragments", 0)) if simulator.fragment_manager is not None else 0
        cmax = float(simulator.c_vol.max().item())
        cmean = float(simulator.c_vol.mean().item())
        surface_cmax = float(simulator.c_vol[simulator.surface_mask].max().item())
        records.append({
            "frame": int(simulator.frame_count),
            "cmax": cmax,
            "cmean": cmean,
            "surface_cmax": surface_cmax,
            "path_count": path_count,
            "path_points": path_points,
            "fragmentation_active": bool(simulator.fragmentation_active),
            "fragment_count": frag_count,
        })

        if frag_ckpt is None and simulator.fragmentation_active and frag_count > 1:
            frag_ckpt = case_dir / f"fragmentation_frame_{simulator.frame_count:02d}.pt"
            simulator.save_state(str(frag_ckpt))

    simulator.save_state(str(final_ckpt))

    heatmap_target = frag_ckpt if frag_ckpt is not None else final_ckpt
    heatmap_path = None
    try:
        heatmap_path = visualize_crack_heatmap(str(heatmap_target), str(case_dir))
    except ValueError as exc:
        print(f"[Experiment] Heatmap skipped for {case_name}: {exc}")

    summary = {
        "case": case_name,
        "device": str(device),
        "config": {
            "particles": int(config.particles.target_count),
            "grids": int(config.mpm.num_grids),
            "frames": int(config.rendering.total_frames),
            "substeps": int(config.rendering.physics_substeps),
            "at2_iters": int(config.phase_field.at2_iters_per_substep),
            "gravity_z": float(mpm_model.gravity[2].item()),
        },
        "overrides": DEFAULT_CASES[case_name],
        "fragment_checkpoint": str(frag_ckpt) if frag_ckpt is not None else None,
        "final_checkpoint": str(final_ckpt),
        "heatmap": str(heatmap_path) if heatmap_path is not None else None,
        "records": records,
    }

    summary_path = case_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del simulator, gaussians, elasticity, mpm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    summaries = []
    for idx, case_name in enumerate(args.cases):
        case_seed = args.seed + idx
        set_seed(case_seed)
        print(f"\n[Experiment] Running case: {case_name} (seed={case_seed})")
        summary = run_case(case_name, args, output_dir)
        summaries.append(summary)

    aggregate_path = output_dir / "aggregate_summary.json"
    aggregate_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\n[Experiment] Wrote aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
