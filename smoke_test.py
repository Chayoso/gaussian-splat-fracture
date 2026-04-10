"""
Smoke test: verify crack propagation on the Gaussian manifold.

No 3DGS rendering — just MPM physics + fracture field + matplotlib scatter.
Outputs PNG frames showing damage field evolution on surface Gaussians.

Usage:
    python smoke_test.py
    python smoke_test.py --frames 120 --fast
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


def plot_fracture_frame(
    positions,
    damage,
    frame,
    out_dir,
    file_prefix="crack",
    title_extra="",
    visited=None,
    tips=None,
    fragment_ids=None,
    n_fragments=0,
    shard_mask=None,
):
    """3-view scatter plot of Gaussian positions with crack overlays."""
    x = positions.cpu().numpy()
    c = damage.cpu().numpy() if damage is not None else np.zeros(x.shape[0])
    v = visited.cpu().numpy() if visited is not None else np.zeros(x.shape[0], dtype=bool)
    t = tips.cpu().numpy() if tips is not None else np.zeros(x.shape[0], dtype=bool)
    frag = fragment_ids.cpu().numpy() if fragment_ids is not None else None
    shard = shard_mask.cpu().numpy() if shard_mask is not None else np.zeros(x.shape[0], dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("X-Y (top)", 0, 1),
        ("X-Z (front)", 0, 2),
        ("Y-Z (side)", 1, 2),
    ]

    for ax, (title, i, j) in zip(axes, views):
        alpha_base = 0.85 if not (frag is not None and n_fragments > 1) else 0.42
        point_size = 0.8 if not (frag is not None and n_fragments > 1) else 0.65
        sc = ax.scatter(x[:, i], x[:, j], c=c, cmap='hot', s=point_size,
                        vmin=0, vmax=1, alpha=alpha_base)
        if frag is not None and n_fragments > 1:
            detached_ids = [frag_id for frag_id in np.unique(frag) if frag_id > 0]
            for frag_id in detached_ids:
                mask = frag == frag_id
                ax.scatter(
                    x[mask, i],
                    x[mask, j],
                    c=np.full(mask.sum(), frag_id),
                    cmap='tab20',
                    s=7.0,
                    alpha=0.96,
                    linewidths=0,
                    vmin=1,
                    vmax=max(n_fragments - 1, 1),
                )
                com = x[mask][:, [i, j]].mean(axis=0)
                ax.scatter(
                    [com[0]], [com[1]],
                    c='white', edgecolors='black', s=28, linewidths=0.7,
                )
        if v.any():
            ax.scatter(x[v, i], x[v, j], c='#34d399', s=2.0, alpha=0.55, linewidths=0)
        if shard.any():
            ax.scatter(x[shard, i], x[shard, j], c='#fbbf24', s=6.0, alpha=0.65, linewidths=0)
        if t.any():
            ax.scatter(x[t, i], x[t, j], c='#38bdf8', s=14.0, alpha=1.0,
                       marker='x', linewidths=0.7)
        ax.set_xlabel('XYZ'[i])
        ax.set_ylabel('XYZ'[j])
        ax.set_title(title)
        ax.set_aspect('equal')
        margin = 0.04
        ax.set_xlim(x[:, i].min() - margin, x[:, i].max() + margin)
        ax.set_ylim(x[:, j].min() - margin, x[:, j].max() + margin)

    fig.colorbar(sc, ax=axes, label='Damage c', shrink=0.6)
    fig.suptitle(
        f"Frame {frame}  |  c_max={c.max():.4f}  c_mean={c.mean():.4f}  "
        f"visited={int(v.sum())}  tips={int(t.sum())}  shards={int(shard.sum())}{title_extra}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_{frame:04d}.png", dpi=100)
    plt.close()


def plot_fragment_frame(positions, fragment_ids, frame, out_dir, title_extra="", file_prefix="fragment"):
    """3-view scatter plot of fragment labels for detach debugging."""
    if fragment_ids is None:
        return

    x = positions.cpu().numpy()
    frag = fragment_ids.cpu().numpy()
    n_fragments = int(frag.max()) + 1 if frag.size > 0 else 0
    if n_fragments <= 1:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("X-Y (top)", 0, 1),
        ("X-Z (front)", 0, 2),
        ("Y-Z (side)", 1, 2),
    ]
    cmap = plt.get_cmap('tab20')
    detached_ids = [frag_id for frag_id in np.unique(frag) if frag_id > 0]

    for ax, (title, i, j) in zip(axes, views):
        base_mask = frag == 0
        ax.scatter(
            x[base_mask, i], x[base_mask, j],
            c='#1f2937', s=0.6, alpha=0.12, linewidths=0,
        )
        for frag_id in detached_ids:
            mask = frag == frag_id
            color = cmap((frag_id - 1) % 20)
            ax.scatter(
                x[mask, i], x[mask, j],
                color=[color], s=7.5, alpha=0.96, linewidths=0,
            )
            com = x[mask][:, [i, j]].mean(axis=0)
            ax.scatter(
                [com[0]], [com[1]],
                c='white', edgecolors='black', s=26, linewidths=0.6,
            )
        ax.set_xlabel('XYZ'[i])
        ax.set_ylabel('XYZ'[j])
        ax.set_title(title)
        ax.set_aspect('equal')
        margin = 0.04
        ax.set_xlim(x[:, i].min() - margin, x[:, i].max() + margin)
        ax.set_ylim(x[:, j].min() - margin, x[:, j].max() + margin)

    fig.suptitle(f"Fragment Labels Frame {frame} | n_frags={n_fragments}{title_extra}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_{frame:04d}.png", dpi=100)
    plt.close()


def plot_opening_frame(positions, opening, frame, out_dir, title_extra=""):
    """3-view scatter plot of crack opening."""
    if opening is None:
        return

    x = positions.cpu().numpy()
    a = opening.cpu().numpy()
    vmax = max(float(a.max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("X-Y (top)", 0, 1),
        ("X-Z (front)", 0, 2),
        ("Y-Z (side)", 1, 2),
    ]

    for ax, (title, i, j) in zip(axes, views):
        sc = ax.scatter(
            x[:, i], x[:, j], c=a, cmap='magma',
            s=0.8, alpha=0.85, linewidths=0,
            vmin=0.0, vmax=vmax,
        )
        ax.set_xlabel('XYZ'[i])
        ax.set_ylabel('XYZ'[j])
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.colorbar(sc, ax=axes, label='Opening a', shrink=0.6)
    fig.suptitle(f"Opening Frame {frame} | a_max={a.max():.5f}{title_extra}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"opening_{frame:04d}.png", dpi=100)
    plt.close()


def plot_cut_debug_frame(
    positions,
    graph,
    cut_core_mask,
    cut_edge_mask,
    frame,
    out_dir,
    title_extra="",
):
    """3-view plot of cut-core nodes and cut-edge midpoints."""
    if cut_core_mask is None or cut_edge_mask is None or graph is None or graph.knn_idx is None:
        return

    cut_core = cut_core_mask.bool()
    edge_mask = cut_edge_mask.bool()
    if not bool(cut_core.any()) and not bool(edge_mask.any()):
        return

    x = positions.cpu().numpy()
    core = cut_core.cpu().numpy()
    edge_rows, edge_cols = torch.where(edge_mask)
    mid = None
    if edge_rows.numel() > 0:
        nbr_idx = graph.knn_idx[edge_rows, edge_cols]
        mid_t = 0.5 * (positions[edge_rows] + positions[nbr_idx])
        if mid_t.shape[0] > 4000:
            step = max(mid_t.shape[0] // 4000, 1)
            mid_t = mid_t[::step]
        mid = mid_t.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("X-Y (top)", 0, 1),
        ("X-Z (front)", 0, 2),
        ("Y-Z (side)", 1, 2),
    ]

    for ax, (title, i, j) in zip(axes, views):
        ax.scatter(x[:, i], x[:, j], c='#1f2937', s=0.5, alpha=0.12, linewidths=0)
        if mid is not None and mid.shape[0] > 0:
            ax.scatter(mid[:, i], mid[:, j], c='#fde047', s=2.0, alpha=0.65, linewidths=0)
        if core.any():
            ax.scatter(x[core, i], x[core, j], c='#ec4899', s=7.0, alpha=0.90, linewidths=0)
        ax.set_xlabel('XYZ'[i])
        ax.set_ylabel('XYZ'[j])
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(
        f"Cut Debug Frame {frame} | cut_core={int(core.sum())} "
        f"cut_edges={int(edge_rows.numel())}{title_extra}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"cut_{frame:04d}.png", dpi=100)
    plt.close()


def plot_closure_debug_frame(
    positions,
    graph,
    closure_candidate_mask,
    closure_boundary_mask,
    frame,
    out_dir,
    title_extra="",
):
    """Visualize raw closure candidates before fragment release logic."""
    if (closure_candidate_mask is None or closure_boundary_mask is None
            or graph is None or graph.knn_idx is None):
        return

    candidate = closure_candidate_mask.bool()
    boundary = closure_boundary_mask.bool()
    if not bool(candidate.any()) and not bool(boundary.any()):
        return

    x = positions.cpu().numpy()
    cand = candidate.cpu().numpy()
    edge_rows, edge_cols = torch.where(boundary)
    mid = None
    if edge_rows.numel() > 0:
        nbr_idx = graph.knn_idx[edge_rows, edge_cols]
        mid_t = 0.5 * (positions[edge_rows] + positions[nbr_idx])
        if mid_t.shape[0] > 5000:
            step = max(mid_t.shape[0] // 5000, 1)
            mid_t = mid_t[::step]
        mid = mid_t.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("X-Y (top)", 0, 1),
        ("X-Z (front)", 0, 2),
        ("Y-Z (side)", 1, 2),
    ]

    for ax, (title, i, j) in zip(axes, views):
        ax.scatter(x[:, i], x[:, j], c='#111827', s=0.45, alpha=0.12, linewidths=0)
        if mid is not None and mid.shape[0] > 0:
            ax.scatter(mid[:, i], mid[:, j], c='#f59e0b', s=2.4, alpha=0.72, linewidths=0)
        if cand.any():
            ax.scatter(x[cand, i], x[cand, j], c='#2563eb', s=3.2, alpha=0.90, linewidths=0)
        ax.set_xlabel('XYZ'[i])
        ax.set_ylabel('XYZ'[j])
        ax.set_title(title)
        ax.set_aspect('equal')
        margin = 0.04
        ax.set_xlim(x[:, i].min() - margin, x[:, i].max() + margin)
        ax.set_ylim(x[:, j].min() - margin, x[:, j].max() + margin)

    fig.suptitle(
        f"Closure Debug Frame {frame} | closure_nodes={int(cand.sum())} "
        f"closure_edges={int(edge_rows.numel())}{title_extra}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"closure_{frame:04d}.png", dpi=100)
    plt.close()


def plot_edge_damage_histogram(
    edge_damage,
    frame,
    out_dir,
    break_threshold,
    broken_edges=0,
    total_edges=0,
    label="edge_damage",
):
    """Histogram of effective edge damage with current break threshold."""
    if edge_damage is None:
        return
    values = edge_damage.detach().cpu().numpy().reshape(-1)
    if values.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(values, bins=60, range=(0.0, 1.0), color="#2563eb", alpha=0.85)
    ax.axvline(float(break_threshold), color="#ef4444", linestyle="--", linewidth=2.0)
    q50, q75, q90, q95, q99 = np.quantile(values, [0.50, 0.75, 0.90, 0.95, 0.99])
    ax.set_xlabel("effective edge damage")
    ax.set_ylabel("count")
    ax.set_title(
        f"Frame {frame} {label}\n"
        f"thr={break_threshold:.3f} broken={broken_edges}/{total_edges} "
        f"q90={q90:.3f} q95={q95:.3f} q99={q99:.3f}"
    )
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.15)
    text = (
        f"q50={q50:.3f}\n"
        f"q75={q75:.3f}\n"
        f"q90={q90:.3f}\n"
        f"q95={q95:.3f}\n"
        f"q99={q99:.3f}"
    )
    ax.text(
        0.98, 0.96, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"edge_hist_{label}_{frame:04d}.png", dpi=120)
    plt.close()


def summarize_edge_damage(edge_damage):
    values = edge_damage.detach().cpu().numpy().reshape(-1)
    if values.size == 0:
        return None
    q50, q75, q90, q95, q99 = np.quantile(values, [0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "q95": float(q95),
        "q99": float(q99),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Manifold fracture smoke test")
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--frames", type=int, default=120)
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
        split_gap_gain=float(gs_cfg.get('split_gap_gain', 1.0)),
        fragment_shell_gain=float(gs_cfg.get('fragment_shell_gain', 1.0)),
        fragment_contrast_gain=float(gs_cfg.get('fragment_contrast_gain', 1.0)),
        debris_darkening=float(gs_cfg.get('debris_darkening', 0.20)),
        shard_scale_gain=float(gs_cfg.get('shard_scale_gain', 1.0)),
        shard_opacity_gain=float(gs_cfg.get('shard_opacity_gain', 1.0)),
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
    first_split_frame = None
    last_n_frags = 1
    max_n_frags = 1
    split_frame_count = 0
    current_split_run = 0
    longest_split_run = 0
    first_cut_frame = None
    max_cut_edges = 0
    max_cross_edge_breaks = 0
    max_detached_distance = 0.0
    max_mean_detached_distance = 0.0
    max_physical_detached_distance = 0.0
    max_physical_mean_detached_distance = 0.0
    max_physical_fragment_drop = 0.0
    max_visible_shards = 0
    max_split_gap_visibility = 0.0
    max_fragment_shell_contrast = 0.0
    max_authoritative_cut_nodes = 0
    max_support_lost_components = 0
    max_support_loss_score = 0.0
    first_closure_frame = None
    max_closure_candidate_count = 0
    max_closure_candidate_nodes = 0
    max_closure_score = 0.0
    last_closure_candidate_count = 0
    edge_hist_snapshots = {}
    for frame in range(total_frames):
        simulator._render_frame = frame
        simulator.step_rendering()
        stats = simulator.get_statistics()
        n_frags = int(stats.get("n_fragments", 0))
        broken_edges = int(stats.get("broken_edges", 0))
        raw_components = int(stats.get("raw_components", 1))
        promoted_components = int(stats.get("promoted_components", 0))
        primary_promoted_components = int(stats.get("primary_promoted_components", 0))
        fallback_promoted_components = int(stats.get("fallback_promoted_components", 0))
        cut_core_nodes = int(stats.get("cut_core_nodes", 0))
        cut_edges = int(stats.get("cut_edges", 0))
        cut_corridor_edges = int(stats.get("cut_corridor_edges", 0))
        cross_edge_breaks = int(stats.get("cross_edge_breaks", 0))
        authoritative_cut_nodes = int(stats.get("authoritative_cut_nodes", 0))
        authoritative_cut_score_max = float(stats.get("authoritative_cut_score_max", 0.0))
        support_lost_components = int(stats.get("support_lost_components", 0))
        support_loss_score_max = float(stats.get("support_loss_score_max", 0.0))
        release_candidate_count = int(stats.get("release_candidate_count", 0))
        boundary_cut_ratio_q50 = float(stats.get("boundary_cut_ratio_q50", 0.0))
        boundary_cut_ratio_q90 = float(stats.get("boundary_cut_ratio_q90", 0.0))
        boundary_cut_ratio_max = float(stats.get("boundary_cut_ratio_max", 0.0))
        components_above_primary = int(stats.get("components_above_primary", 0))
        components_above_fallback = int(stats.get("components_above_fallback", 0))
        absorbed_components = int(stats.get("absorbed_components", 0))
        closure_candidate_count = int(stats.get("closure_candidate_count", 0))
        closure_candidate_nodes = int(stats.get("closure_candidate_nodes", 0))
        closure_score_max = float(stats.get("closure_score_max", 0.0))
        closure_candidate_sizes = stats.get("closure_candidate_sizes", [])
        detached_distance = float(stats.get("detached_distance", 0.0))
        mean_detached_distance = float(stats.get("mean_detached_distance", 0.0))
        physical_detached_distance = float(stats.get("physical_detached_distance", 0.0))
        physical_mean_detached_distance = float(stats.get("physical_mean_detached_distance", 0.0))
        physical_fragment_drop = float(stats.get("physical_fragment_drop", 0.0))
        visible_shards = int(stats.get("visible_shard_count", 0))
        split_gap_visibility = float(stats.get("split_gap_visibility", 0.0))
        fragment_shell_contrast = float(stats.get("fragment_shell_contrast", 0.0))
        shard_persistence = float(stats.get("shard_persistence", 0.0))
        top_component_sizes = stats.get("top_component_sizes", [])
        gm = simulator.fragment_manager
        if n_frags > 1 and first_split_frame is None:
            first_split_frame = frame
        if cut_edges > 0 and first_cut_frame is None:
            first_cut_frame = frame
        if closure_candidate_count > 0 and first_closure_frame is None:
            first_closure_frame = frame
        if n_frags > 1:
            split_frame_count += 1
            current_split_run += 1
            longest_split_run = max(longest_split_run, current_split_run)
        else:
            current_split_run = 0
        max_n_frags = max(max_n_frags, n_frags)
        max_cut_edges = max(max_cut_edges, cut_edges)
        max_cross_edge_breaks = max(max_cross_edge_breaks, cross_edge_breaks)
        max_detached_distance = max(max_detached_distance, detached_distance)
        max_mean_detached_distance = max(max_mean_detached_distance, mean_detached_distance)
        max_physical_detached_distance = max(max_physical_detached_distance, physical_detached_distance)
        max_physical_mean_detached_distance = max(max_physical_mean_detached_distance, physical_mean_detached_distance)
        max_physical_fragment_drop = max(max_physical_fragment_drop, physical_fragment_drop)
        max_visible_shards = max(max_visible_shards, visible_shards)
        max_split_gap_visibility = max(max_split_gap_visibility, split_gap_visibility)
        max_fragment_shell_contrast = max(max_fragment_shell_contrast, fragment_shell_contrast)
        max_authoritative_cut_nodes = max(max_authoritative_cut_nodes, authoritative_cut_nodes)
        max_support_lost_components = max(max_support_lost_components, support_lost_components)
        max_support_loss_score = max(max_support_loss_score, support_loss_score_max)
        max_closure_candidate_count = max(max_closure_candidate_count, closure_candidate_count)
        max_closure_candidate_nodes = max(max_closure_candidate_nodes, closure_candidate_nodes)
        max_closure_score = max(max_closure_score, closure_score_max)
        if gm is not None and gm.last_effective_edge_damage is not None:
            if first_cut_frame == frame and "first_cut" not in edge_hist_snapshots:
                edge_hist_snapshots["first_cut"] = {
                    "frame": frame,
                    "edge_damage": gm.last_effective_edge_damage.clone(),
                    "threshold": gm.last_edge_damage_break_threshold,
                    "broken_edges": gm.last_broken_edges,
                    "total_edges": gm.last_total_edges,
                }
            if first_split_frame == frame and "first_split" not in edge_hist_snapshots:
                edge_hist_snapshots["first_split"] = {
                    "frame": frame,
                    "edge_damage": gm.last_effective_edge_damage.clone(),
                    "threshold": gm.last_edge_damage_break_threshold,
                    "broken_edges": gm.last_broken_edges,
                    "total_edges": gm.last_total_edges,
                }
        if n_frags != last_n_frags:
            print(
                f"[SmokeTest:fragments] frame={frame} "
                f"n_frags={n_frags} raw_components={raw_components} "
                f"broken_edges={broken_edges} promoted={promoted_components} "
                f"primary={primary_promoted_components} fallback={fallback_promoted_components} "
                f"cut_core={cut_core_nodes} auth_cut={authoritative_cut_nodes} "
                f"cut_edges={cut_edges} corridor={cut_corridor_edges} cross_breaks={cross_edge_breaks} "
                f"cut_q90={boundary_cut_ratio_q90:.2f} cut_max={boundary_cut_ratio_max:.2f} "
                f"aboveP={components_above_primary} aboveF={components_above_fallback} "
                f"absorbed={absorbed_components} "
                f"support_lost={support_lost_components} "
                f"release={release_candidate_count} "
                f"detach={detached_distance:.4f} phys={physical_detached_distance:.4f} "
                f"top_sizes={top_component_sizes[:4]}"
            )
            last_n_frags = n_frags
        if closure_candidate_count != last_closure_candidate_count:
            print(
                f"[SmokeTest:closure] frame={frame} "
                f"closure_count={closure_candidate_count} closure_nodes={closure_candidate_nodes} "
                f"closure_score={closure_score_max:.2f} sizes={closure_candidate_sizes[:4]}"
            )
            last_closure_candidate_count = closure_candidate_count

        # Plot every 5 frames or at key moments
        ff = simulator.fracture_field
        if frame % 5 == 0 or frame == total_frames - 1:
            damage = ff.c if ff.c is not None else torch.zeros(N_surf, device=device)
            frag_ids = None
            if (simulator.fragment_manager is not None
                    and simulator.fragment_manager.fragment_ids is not None):
                frag_ids = simulator.fragment_manager.fragment_ids
            physical_frag_ids = frag_ids
            if (getattr(simulator, "_physical_fragment_labels", None) is not None
                    and getattr(simulator, "_surface_indices", None) is not None
                    and bool((simulator._physical_fragment_labels > 0).any())):
                physical_frag_ids = simulator._physical_fragment_labels[simulator._surface_indices]
            render_state = simulator._last_render_state or {}
            render_positions = render_state.get("positions", gaussians._xyz.data.detach())
            render_damage = render_state.get("damage", damage)
            render_frag = render_state.get("fragment_ids", frag_ids)
            render_visited = render_state.get(
                "crack_visited",
                ff.crack_front.visited_mask if hasattr(ff, 'crack_front') else None,
            )
            render_tips = render_state.get(
                "crack_tips",
                ff.crack_front.tip_mask if hasattr(ff, 'crack_front') else None,
            )
            render_shards = render_state.get("shard_mask", None)
            render_opening = render_state.get("opening", ff.a if ff.a is not None else None)
            plot_fracture_frame(
                render_positions, render_damage,
                frame, out_dir,
                file_prefix="crack",
                    title_extra=(
                        f"  |  H_max={ff.H.max():.2e}  n_frags={n_frags}  broken={broken_edges}"
                        f"  q90={boundary_cut_ratio_q90:.2f}  shards={visible_shards}  gap={split_gap_visibility:.2f}"
                        if ff.H is not None else f"  |  n_frags={n_frags}  broken={broken_edges}"
                    ),
                visited=render_visited,
                tips=render_tips,
                fragment_ids=render_frag,
                n_fragments=n_frags,
                shard_mask=render_shards,
            )
            if render_frag is not None and n_frags > 1:
                plot_fragment_frame(
                    render_positions,
                    render_frag,
                    frame,
                    out_dir,
                    title_extra=(
                        f"  broken={broken_edges}  detach={detached_distance:.4f}"
                        f"  phys={physical_detached_distance:.4f}"
                    ),
                    file_prefix="fragment",
                )
            if physical_frag_ids is not None and n_frags > 1 and (frame % 10 == 0 or frame == total_frames - 1):
                physical_positions = simulator.mapper.mpm_to_world(
                    simulator.x_mpm[simulator.surface_mask]
                )
                n_phys = min(
                    physical_positions.shape[0],
                    damage.shape[0],
                    physical_frag_ids.shape[0],
                )
                plot_fracture_frame(
                    physical_positions[:n_phys],
                    damage[:n_phys],
                    frame,
                    out_dir,
                    file_prefix="physical_crack",
                    title_extra=(
                        f"  |  n_frags={n_frags}  cut_edges={cut_edges}"
                        f"  phys_detach={physical_detached_distance:.4f}"
                        f"  drop={physical_fragment_drop:.4f}"
                    ),
                    visited=(
                        ff.crack_front.visited_mask[:n_phys]
                        if hasattr(ff, 'crack_front') and ff.crack_front.visited_mask is not None
                        else None
                    ),
                    tips=(
                        ff.crack_front.tip_mask[:n_phys]
                        if hasattr(ff, 'crack_front') and ff.crack_front.tip_mask is not None
                        else None
                    ),
                    fragment_ids=physical_frag_ids[:n_phys],
                    n_fragments=n_frags,
                    shard_mask=None,
                )
                plot_fragment_frame(
                    physical_positions[:n_phys],
                    physical_frag_ids[:n_phys],
                    frame,
                    out_dir,
                    title_extra=(
                        f"  cut_edges={cut_edges}  phys_detach={physical_detached_distance:.4f}"
                        f"  drop={physical_fragment_drop:.4f}"
                    ),
                    file_prefix="physical_fragment",
                )
            plot_opening_frame(
                render_positions,
                render_opening,
                frame,
                out_dir,
                title_extra=(
                    f"  cut_edges={cut_edges}  shell={fragment_shell_contrast:.3f}"
                    f"  support={support_loss_score_max:.2f}  persist={shard_persistence:.1f}"
                ),
            )
            if simulator.fragment_manager is not None:
                base_positions = render_positions[:damage.shape[0]]
                plot_cut_debug_frame(
                    base_positions,
                    simulator.graph,
                    simulator.fragment_manager.last_cut_core_mask,
                    simulator.fragment_manager.last_cut_edge_mask,
                    frame,
                    out_dir,
                    title_extra=(
                        f"  cross_breaks={cross_edge_breaks} "
                        f"detach={detached_distance:.4f}"
                    ),
                )
                if closure_candidate_count > 0 or frame == total_frames - 1:
                    plot_closure_debug_frame(
                        base_positions,
                        simulator.graph,
                        simulator.fragment_manager.last_closure_candidate_mask,
                        simulator.fragment_manager.last_closure_boundary_mask,
                        frame,
                        out_dir,
                        title_extra=(
                            f"  closure_count={closure_candidate_count}"
                            f"  score={closure_score_max:.2f}"
                        ),
                    )

        elapsed = time.time() - t0
        c_max = ff.c.max().item() if ff.c is not None else 0
        if frame % 5 == 0 or frame == total_frames - 1:
            print(f"Frame {frame:3d}/{total_frames}  "
                  f"c_max={c_max:.4f}  n_frags={n_frags}  cut_edges={cut_edges}  "
                  f"cut_q90={boundary_cut_ratio_q90:.2f}  "
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
    print(f"  first_split_frame = {first_split_frame}")
    print(f"  first_cut_frame = {first_cut_frame}")
    print(f"  first_closure_frame = {first_closure_frame}")
    print(f"  split_frame_count = {split_frame_count}")
    print(f"  longest_split_run = {longest_split_run}")
    print(f"  max_n_frags = {max_n_frags}")
    print(f"  max_cut_edges = {max_cut_edges}")
    print(f"  max_cross_edge_breaks = {max_cross_edge_breaks}")
    print(f"  max_authoritative_cut_nodes = {max_authoritative_cut_nodes}")
    print(f"  max_support_lost_components = {max_support_lost_components}")
    print(f"  max_support_loss_score = {max_support_loss_score:.4f}")
    print(f"  max_closure_candidate_count = {max_closure_candidate_count}")
    print(f"  max_closure_candidate_nodes = {max_closure_candidate_nodes}")
    print(f"  max_closure_score = {max_closure_score:.4f}")
    print(f"  max_detached_distance = {max_detached_distance:.4f}")
    print(f"  max_mean_detached_distance = {max_mean_detached_distance:.4f}")
    print(f"  max_physical_detached_distance = {max_physical_detached_distance:.4f}")
    print(f"  max_physical_mean_detached_distance = {max_physical_mean_detached_distance:.4f}")
    print(f"  max_physical_fragment_drop = {max_physical_fragment_drop:.4f}")
    print(f"  max_visible_shards = {max_visible_shards}")
    print(f"  max_split_gap_visibility = {max_split_gap_visibility:.4f}")
    print(f"  max_fragment_shell_contrast = {max_fragment_shell_contrast:.4f}")
    if simulator.fragment_manager is not None:
        gm = simulator.fragment_manager
        if gm.last_effective_edge_damage is not None:
            edge_hist_snapshots["final"] = {
                "frame": total_frames - 1,
                "edge_damage": gm.last_effective_edge_damage.clone(),
                "threshold": gm.last_edge_damage_break_threshold,
                "broken_edges": gm.last_broken_edges,
                "total_edges": gm.last_total_edges,
            }
        print(f"  final_n_frags = {simulator.fragment_manager.n_fragments}")
        if (getattr(simulator, "_physical_fragment_labels", None) is not None
                and bool((simulator._physical_fragment_labels > 0).any())):
            print(f"  final_physical_n_frags = {int(simulator._physical_fragment_labels.max().item()) + 1}")
        print(f"  broken_edges = {simulator.fragment_manager.last_broken_edges}")
        print(f"  raw_components = {simulator.fragment_manager.last_raw_components}")
        print(f"  promoted_components = {simulator.fragment_manager.last_promoted_components}")
        print(f"  primary_promoted_components = {simulator.fragment_manager.last_primary_promoted_components}")
        print(f"  fallback_promoted_components = {simulator.fragment_manager.last_fallback_promoted_components}")
        print(f"  cut_core_nodes = {simulator.fragment_manager.last_cut_core_nodes}")
        print(f"  cut_edges = {simulator.fragment_manager.last_cut_edges}")
        print(f"  cut_corridor_edges = {simulator.fragment_manager.last_cut_corridor_edges}")
        print(f"  cross_edge_breaks = {simulator.fragment_manager.last_cross_edge_breaks}")
        print(f"  authoritative_cut_nodes = {simulator.fragment_manager.last_authoritative_cut_nodes}")
        print(f"  authoritative_cut_score_max = {simulator.fragment_manager.last_authoritative_cut_score_max:.4f}")
        print(f"  support_lost_components = {simulator.fragment_manager.last_support_lost_components}")
        print(f"  support_loss_score_max = {simulator.fragment_manager.last_support_loss_score_max:.4f}")
        print(f"  release_candidate_count = {simulator.fragment_manager.last_release_candidate_count}")
        print(f"  boundary_cut_ratio_q50 = {simulator.fragment_manager.last_boundary_cut_ratio_q50:.4f}")
        print(f"  boundary_cut_ratio_q90 = {simulator.fragment_manager.last_boundary_cut_ratio_q90:.4f}")
        print(f"  boundary_cut_ratio_max = {simulator.fragment_manager.last_boundary_cut_ratio_max:.4f}")
        print(f"  components_above_primary = {simulator.fragment_manager.last_components_above_primary}")
        print(f"  components_above_fallback = {simulator.fragment_manager.last_components_above_fallback}")
        print(f"  absorbed_components = {simulator.fragment_manager.last_absorbed_components}")
        print(f"  closure_candidate_count = {simulator.fragment_manager.last_closure_candidate_count}")
        print(f"  closure_candidate_nodes = {simulator.fragment_manager.last_closure_candidate_nodes}")
        print(f"  closure_score_max = {simulator.fragment_manager.last_closure_score_max:.4f}")
        print(f"  closure_candidate_sizes = {simulator.fragment_manager.last_closure_candidate_sizes[:8]}")
        print(f"  top_component_sizes = {simulator.fragment_manager.last_top_component_sizes}")
    for label, snap in edge_hist_snapshots.items():
        summary = summarize_edge_damage(snap["edge_damage"])
        if summary is not None:
            print(
                f"  edge_hist[{label}] frame={snap['frame']} "
                f"thr={snap['threshold']:.4f} "
                f"mean={summary['mean']:.4f} "
                f"q50={summary['q50']:.4f} q75={summary['q75']:.4f} "
                f"q90={summary['q90']:.4f} q95={summary['q95']:.4f} q99={summary['q99']:.4f} "
                f"max={summary['max']:.4f}"
            )
        plot_edge_damage_histogram(
            snap["edge_damage"],
            frame=snap["frame"],
            out_dir=out_dir,
            break_threshold=snap["threshold"],
            broken_edges=snap["broken_edges"],
            total_edges=snap["total_edges"],
            label=label,
        )
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
