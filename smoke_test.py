"""
Smoke test: verify crack propagation on the Gaussian manifold.

No 3DGS rendering — just MPM physics + fracture field + matplotlib scatter.
Outputs PNG frames showing damage field evolution on surface Gaussians.

Usage:
    python smoke_test.py
    python smoke_test.py --frames 30 --fast
"""

import sys, os, argparse, time
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gaussian-splatting"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.config.loader import load_config, apply_overrides_dict
from src.mpm_core.mpm_pipeline import create_mpm_model, configure_loading
from src.constitutive_models.model_factory import create_elasticity_model
from src.preprocessing.mesh_converter import MeshToPointCloudConverter
from src.core.coordinate_mapper import CoordinateMapper
from src.core.manifold_simulator import ManifoldSimulator
from src.core.material_presets import resolve_material_preset, validate_l0
from src.engine.loading_transforms import apply_loading_transforms
from src.visualization.gaussian_updater import GaussianCrackVisualizer
from src.ml.material_predictor import MaterialPredictor
from src.ml.material_prior_adapter import MaterialPriorAdapter

from omegaconf import OmegaConf


class DummyGaussians:
    """Minimal stand-in for GaussianModel — no CUDA rasterizer needed."""
    def __init__(self, N, device='cuda'):
        self._xyz = torch.nn.Parameter(torch.zeros(N, 3, device=device))
        self._features_dc = torch.nn.Parameter(torch.zeros(N, 1, 3, device=device))
        self._features_rest = torch.nn.Parameter(torch.zeros(N, 15, 3, device=device))
        self._opacity = torch.nn.Parameter(torch.zeros(N, 1, device=device))
        self._scaling = torch.nn.Parameter(torch.zeros(N, 3, device=device))
        self._rotation = torch.nn.Parameter(
            torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).expand(N, 4).clone())


class DummyVisualizer:
    """No-op visualizer — we plot with matplotlib instead."""
    def __init__(self):
        self.damage_threshold = 0.3
    def set_initial_normals(self, normals):
        pass
    def update_gaussians(self, *args, **kwargs):
        pass


def plot_damage_frame(x_mpm, surface_mask, damage, frame, out_dir, title_extra=""):
    """3-view scatter plot of surface Gaussians colored by damage."""
    x = x_mpm[surface_mask].cpu().numpy()
    c = damage.cpu().numpy() if damage is not None else np.zeros(x.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    views = [
        ("X-Y (top)", 0, 1),
        ("X-Z (front)", 0, 2),
        ("Y-Z (side)", 1, 2),
    ]

    for ax, (title, i, j) in zip(axes, views):
        sc = ax.scatter(x[:, i], x[:, j], c=c, cmap='hot', s=0.5,
                        vmin=0, vmax=1, alpha=0.8)
        ax.set_xlabel('XYZ'[i])
        ax.set_ylabel('XYZ'[j])
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.colorbar(sc, ax=axes, label='Damage c', shrink=0.6)
    fig.suptitle(f"Frame {frame}  |  c_max={c.max():.4f}  c_mean={c.mean():.4f}  "
                 f"cracked(>0.3)={(c > 0.3).sum()}{title_extra}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"crack_{frame:04d}.png", dpi=100)
    plt.close()


def plot_fracture_frame(positions, damage, frame, out_dir, title_extra="", visited=None, tips=None):
    """3-view scatter plot of Gaussian positions with crack overlays."""
    x = positions.cpu().numpy()
    c = damage.cpu().numpy() if damage is not None else np.zeros(x.shape[0])
    v = visited.cpu().numpy() if visited is not None else np.zeros(x.shape[0], dtype=bool)
    t = tips.cpu().numpy() if tips is not None else np.zeros(x.shape[0], dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("X-Y (top)", 0, 1),
        ("X-Z (front)", 0, 2),
        ("Y-Z (side)", 1, 2),
    ]

    for ax, (title, i, j) in zip(axes, views):
        sc = ax.scatter(x[:, i], x[:, j], c=c, cmap='hot', s=0.8,
                        vmin=0, vmax=1, alpha=0.85)
        if v.any():
            ax.scatter(x[v, i], x[v, j], c='#34d399', s=2.0, alpha=0.55, linewidths=0)
        if t.any():
            ax.scatter(x[t, i], x[t, j], c='#38bdf8', s=14.0, alpha=1.0,
                       marker='x', linewidths=0.7)
        ax.set_xlabel('XYZ'[i])
        ax.set_ylabel('XYZ'[j])
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.colorbar(sc, ax=axes, label='Damage c', shrink=0.6)
    fig.suptitle(
        f"Frame {frame}  |  c_max={c.max():.4f}  c_mean={c.mean():.4f}  "
        f"visited={int(v.sum())}  tips={int(t.sum())}{title_extra}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"crack_{frame:04d}.png", dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Manifold fracture smoke test")
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--fast", action="store_true", help="50k particles, 64 grid")
    parser.add_argument("--out", default="output/smoke_test")
    parser.add_argument("--clip", type=str, default=None, help="Material prompt for CLIP-driven priors")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Config ---
    config = load_config(args.config)
    if args.fast:
        OmegaConf.update(config, "particles.target_count", 50000)
        OmegaConf.update(config, "mpm.num_grids", 64)
        OmegaConf.update(config, "rendering.physics_substeps", 8)

    material_prior = None
    if args.clip:
        predictor = MaterialPredictor(
            mode="clip_knn",
            db_path=args.db_path,
            clip_model=args.clip_model,
        )
        adapter = MaterialPriorAdapter()
        raw_params = predictor.predict(args.clip)
        topk_entries = [
            predictor.db.get_entry_by_name(name)
            for name in raw_params["top_k_names"]
        ]
        material_prior = adapter.build_material_prior(
            topk_entries,
            raw_params["top_k_scores"],
        )
        scaled = adapter.scale_physics_to_mpm(
            material_prior["physics"],
            base_density=float(config.material.density),
        )
        runtime_overrides = {
            "E": scaled["E"],
            "Gc": scaled["Gc"],
            "nu": scaled["nu"],
            "density": scaled["density"],
            **material_prior["runtime"],
        }
        config = apply_overrides_dict(config, runtime_overrides)
        print(f"[SmokeTest:clip] Material: '{args.clip}'")
        print(f"  Top-K: {raw_params['top_k_names']}")
        print(f"  Weights: {[f'{w:.3f}' for w in material_prior['weights']]}")
        print(
            f"  MPM scaled: E={scaled['E']:.2e}, Gc={scaled['Gc']:.1f}, "
            f"density={scaled['density']:.1f}"
        )
        print(
            f"  Fracture prior: tau={material_prior['fracture']['tau_init']:.2f}, "
            f"growth={material_prior['fracture']['growth_gain']:.2f}, "
            f"band={material_prior['fracture']['band_width']:.2f}, "
            f"open={material_prior['fracture']['open_gain']:.2f}"
        )
        print(f"  Family: {material_prior['family']}")

    config = resolve_material_preset(config)
    config = validate_l0(config)

    # --- Mesh → Point Clouds ---
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("Generating point clouds...")
    converter = MeshToPointCloudConverter(
        mesh_path=config.mesh.path,
        target_particle_count=config.particles.target_count,
        surface_sample_ratio=config.particles.get('surface_ratio', 0.5),
        use_poisson=config.particles.get('use_poisson_sampling', False),
        normalize_to_unit_cube=config.particles.get('normalize_to_unit_cube', True),
    )
    volume_pcd, surface_pcd, surface_mask_np = converter.convert()
    volume_points = np.asarray(volume_pcd.points).copy()
    volume_normals = np.asarray(volume_pcd.normals).copy()
    N_total = volume_points.shape[0]
    N_surf = surface_mask_np.sum()
    print(f"Particles: {N_total} total, {N_surf} surface")

    # --- MPM + Elasticity ---
    mpm_model = create_mpm_model(config, volume_pcd, device)
    loading_params = configure_loading(config, mpm_model, device)
    elasticity = create_elasticity_model(config, device)

    # --- Coord mapper ---
    coord_mapper = CoordinateMapper(
        mpm_bounds=(0.0, 1.0),
        world_center=np.array(list(config.coordinate_mapping.world_center)),
        world_scale=float(config.coordinate_mapping.world_scale),
        device=str(device),
    )

    # --- Dummy Gaussians + Visualizer (no rendering) ---
    gaussians = DummyGaussians(N_surf, device)
    gs_cfg = config.gaussian_splatting
    visualizer = GaussianCrackVisualizer(
        damage_threshold=float(gs_cfg.get('damage_threshold', 0.3)),
        device=str(device),
        crack_color=tuple(gs_cfg.get('crack_color', [0.6, 0.08, 0.08])),
        crack_opacity_reduction=float(gs_cfg.get('crack_opacity_reduction', 0.70)),
        crack_max_opening=float(gs_cfg.get('crack_max_opening', 0.010)),
        crack_gap_fraction=float(gs_cfg.get('crack_gap_fraction', 0.35)),
        crack_edge_darken=float(gs_cfg.get('crack_edge_darken', 0.75)),
        crack_red_accent=float(gs_cfg.get('crack_red_accent', 0.10)),
        crack_tip_scale_boost=float(gs_cfg.get('crack_tip_scale_boost', 0.20)),
        crack_tip_opacity_boost=float(gs_cfg.get('crack_tip_opacity_boost', 0.10)),
        material_family=str(gs_cfg.get('material_family', 'neutral_reference')),
        crack_band_weight=float(gs_cfg.get('crack_band_weight', 0.80)),
        crack_visited_weight=float(gs_cfg.get('crack_visited_weight', 0.45)),
        crack_tip_weight=float(gs_cfg.get('crack_tip_weight', 0.95)),
        crack_core_weight=float(gs_cfg.get('crack_core_weight', 1.00)),
        damage_scale_shrink=float(gs_cfg.get('damage_scale_shrink', 0.50)),
        damage_center_opacity_reduction=float(gs_cfg.get('damage_center_opacity_reduction', 0.70)),
        diffuse_damage_strength=float(gs_cfg.get('diffuse_damage_strength', 0.12)),
    )
    surface_mask = torch.from_numpy(surface_mask_np).bool().to(device)

    # --- Fracture params ---
    pf_params = OmegaConf.to_container(config.phase_field, resolve=True)
    manifold_cfg = OmegaConf.to_container(
        config.get('manifold', {}), resolve=True) if hasattr(config, 'manifold') else {}
    fracture_params = {**pf_params, **manifold_cfg}

    # --- Seismic ---
    seismic_params = {}
    if hasattr(config, 'seismic'):
        seismic_params = {
            'enabled': config.seismic.get('enabled', False),
            'amplitude': float(config.seismic.get('amplitude', 0)),
            'frequency': float(config.seismic.get('frequency', 50)),
            'direction': list(config.seismic.get('direction', [1, 0, 0])),
            'ramp_time': float(config.seismic.get('ramp_time', 0.01)),
        }

    # --- ManifoldSimulator ---
    simulator = ManifoldSimulator(
        mpm_model=mpm_model,
        gaussians=gaussians,
        elasticity_module=elasticity,
        coord_mapper=coord_mapper,
        visualizer=visualizer,
        surface_mask=surface_mask,
        physics_substeps=config.rendering.physics_substeps,
        fracture_params=fracture_params,
        simulation_mode=config.simulation.mode,
        seismic_params=seismic_params,
    )

    # Initialize simulation state
    all_normals = torch.from_numpy(volume_normals).float().to(device)
    init_pos = torch.from_numpy(volume_points).float().to(device)
    simulator.initialize(init_pos)

    # Pass surface normals for manifold-aware graph
    simulator.set_surface_normals(all_normals)

    apply_loading_transforms(config, simulator, loading_params, device)

    # --- Main loop ---
    total_frames = args.frames
    print(f"\n{'='*50}")
    print(f"Running {total_frames} frames (smoke test, no rendering)")
    print(f"{'='*50}\n")

    t0 = time.time()
    for frame in range(total_frames):
        simulator._render_frame = frame
        simulator.step_rendering()

        # Plot every 5 frames or at key moments
        ff = simulator.fracture_field
        if frame % 5 == 0 or frame == total_frames - 1:
            damage = ff.c if ff.c is not None else torch.zeros(N_surf, device=device)
            plot_fracture_frame(
                gaussians._xyz.data.detach(), damage,
                frame, out_dir,
                title_extra=f"  |  H_max={ff.H.max():.2e}" if ff.H is not None else "",
                visited=ff.crack_front.visited_mask if hasattr(ff, 'crack_front') else None,
                tips=ff.crack_front.tip_mask if hasattr(ff, 'crack_front') else None,
            )

        elapsed = time.time() - t0
        c_max = ff.c.max().item() if ff.c is not None else 0
        if frame % 5 == 0 or frame == total_frames - 1:
            print(f"Frame {frame:3d}/{total_frames}  "
                  f"c_max={c_max:.4f}  "
                  f"elapsed={elapsed:.1f}s")

    # --- Summary ---
    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Smoke test complete: {total_frames} frames in {elapsed:.1f}s")
    print(f"Output: {out_dir}")
    if material_prior is not None:
        print(f"  material_category = {material_prior['dominant_category']}")
        print(f"  family = {material_prior['family']}")
    ff = simulator.fracture_field
    if ff.c is not None:
        c = ff.c
        print(f"  c_max  = {c.max():.4f}")
        print(f"  c_mean = {c.mean():.4f}")
        print(f"  cracked (c>0.3): {(c > 0.3).sum().item()}/{c.shape[0]}")
        print(f"  cracked (c>0.8): {(c > 0.8).sum().item()}/{c.shape[0]}")
    if ff.H is not None:
        print(f"  H_max  = {ff.H.max():.2e}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
