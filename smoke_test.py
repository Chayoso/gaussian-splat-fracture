"""
Smoke test: verify crack propagation on the Gaussian manifold.

No 3DGS rendering — just MPM physics + fracture field + matplotlib scatter.
Outputs PNG frames showing damage field evolution on surface Gaussians.

Usage:
    python smoke_test.py
    python smoke_test.py --surface-only --clip "glass bottle" --frames 80
    python smoke_test.py --frames 120 --particles 50000
    python smoke_test.py --frames 120 --particles 150000
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
from src.fracture.graph_builder import GaussianGraph
from src.fracture.graph_fragment_manager import GraphFragmentManager
from src.fracture.tip_based_fracture_field import GaussianFractureField
from src.fracture.crack_front import CrackFront
from src.fracture.surface_crack_driver import SurfaceCrackDriver
from src.engine.loading_transforms import apply_loading_transforms
from src.visualization.gaussian_updater import GaussianCrackVisualizer
from src.ml.material_predictor import MaterialPredictor
from src.ml.material_prior_adapter import FAMILY_RUNTIME_PRESETS, MaterialPriorAdapter

from omegaconf import OmegaConf


SURFACE_COLLAPSE_OVERRIDES = {
    "manifold.material_family": "rough_quasi_brittle",
    "manifold.enable_front_propagation": True,
    "manifold.material_drive_floor": 0.11,
    "manifold.damage_source_scale": 0.72,
    "manifold.damage_spread": 0.62,
    "manifold.drive_quantile": 0.45,
    "manifold.front_threshold": 0.02,
    "manifold.tip_propagation_scale": 1.30,
    "manifold.front_substeps": 4,
    "manifold.tau_init": 0.16,
    "manifold.growth_gain": 1.45,
    "manifold.band_width": 2.35,
    "manifold.band_fill_gain": 0.80,
    "manifold.open_gain": 1.35,
    "manifold.branching_bias": 0.85,
    "manifold.successor_topk": 5,
    "manifold.min_successor_score": 0.12,
    "manifold.branch_drive_threshold": 0.22,
    "manifold.branch_score_ratio": 0.70,
    "manifold.max_branching_tips": 48,
    "manifold.fragment_detect_every": 4,
    "manifold.fragment_damage_threshold": 0.28,
    "manifold.min_fragment_particles": 12,
    "manifold.fragment_opening_weight": 0.68,
    "manifold.fragment_active_tip_weight": 0.34,
    "manifold.fragment_recent_front_weight": 0.32,
    "manifold.fragment_pair_break_weight": 0.24,
    "manifold.fragment_edge_memory_decay": 0.994,
    "manifold.fragment_edge_memory_weight": 0.98,
    "manifold.fragment_cut_diffusion_alpha": 0.58,
    "manifold.fragment_cut_diffusion_iters": 1,
    "manifold.fragment_cut_cos_gate_tangent": 0.36,
    "manifold.fragment_cut_cos_gate_normal": 0.30,
    "manifold.fragment_primary_cut_ratio": 0.38,
    "manifold.fragment_fallback_cut_ratio": 0.22,
    "manifold.fragment_min_boundary_edges": 8,
    "manifold.fragment_persistent_min_size": 4,
    "manifold.fragment_component_hysteresis": 0.18,
    "manifold.fragment_post_split_threshold_scale": 0.70,
    "manifold.cut_surface_enable": True,
    "manifold.cut_vote_strength": 1.15,
    "manifold.tau_cross": 0.38,
    "manifold.tau_tangent": 0.58,
    "manifold.cut_core_damage_threshold": 0.075,
    "manifold.cut_core_opening_threshold": 0.045,
    "manifold.cut_hard_break_threshold": 0.16,
    "manifold.authoritative_cut_decay": 0.992,
    "manifold.authoritative_cut_threshold": 0.085,
    "manifold.support_loss_enable": True,
    "manifold.support_anchor_quantile": 0.18,
    "manifold.support_release_threshold": 0.36,
    "manifold.support_promote_min_size": 4,
    "manifold.support_overlap_threshold": 0.035,
    "manifold.open_crack_release_enable": True,
    "manifold.open_crack_release_threshold": 0.32,
    "manifold.open_crack_release_max_patches": 6,
    "manifold.collapse_fast_path": True,
    "manifold.collapse_vector_path": True,
    "manifold.collapse_fast_start_frame": 4,
    "manifold.collapse_fast_threshold": 0.30,
    "manifold.collapse_fast_min_threshold": 0.055,
    "manifold.collapse_fast_threshold_decay": 0.004,
    "manifold.collapse_fast_patch_radius": 0.065,
    "manifold.collapse_fast_patch_core_radius": 0.024,
    "manifold.collapse_fast_min_size": 18,
    "manifold.collapse_fast_max_size_ratio": 0.040,
    "manifold.collapse_fast_patches_per_frame": 10,
    "manifold.collapse_fragment_gap": 0.020,
    "manifold.collapse_fragment_speed": 0.010,
    "manifold.collapse_fragment_gravity": 0.000045,
    "manifold.collapse_fragment_fall_cap": 0.42,
    "manifold.collapse_fragment_max_offset": 0.24,
    "manifold.collapse_fragment_spin_gain": 0.22,
    "gaussian_splatting.material_family": "rough_quasi_brittle",
}


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
    if cut_edge_mask is None or graph is None or graph.knn_idx is None:
        return

    if cut_core_mask is None:
        cut_core = torch.zeros(
            positions.shape[0],
            dtype=torch.bool,
            device=positions.device,
        )
    else:
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


def make_crack_front(fracture_params, device):
    return CrackFront(
        seed_quantile=fracture_params.get('seed_quantile', 0.995),
        max_seed_points=fracture_params.get('max_seed_points', 2),
        min_seed_spacing=fracture_params.get('min_seed_spacing', 0.04),
        successor_topk=fracture_params.get('successor_topk', 2),
        min_successor_score=fracture_params.get('min_successor_score', 0.25),
        drive_weight=fracture_params.get('drive_weight', 0.40),
        distance_weight=fracture_params.get('distance_weight', 0.12),
        align_weight=fracture_params.get('align_weight', 0.24),
        tangent_weight=fracture_params.get('tangent_weight', 0.16),
        continuity_weight=fracture_params.get('continuity_weight', 0.10),
        radial_weight=fracture_params.get('radial_weight', 0.16),
        lift_weight=fracture_params.get('lift_weight', 0.40),
        max_tip_age=fracture_params.get('max_tip_age', 2),
        revisit_drive_threshold=fracture_params.get('revisit_drive_threshold', 0.8),
        branch_score_ratio=fracture_params.get('branch_score_ratio', 0.97),
        branch_drive_threshold=fracture_params.get('branch_drive_threshold', 0.70),
        max_branching_tips=fracture_params.get('max_branching_tips', 12),
        tau_init=fracture_params.get('tau_init', 0.30),
        growth_gain=fracture_params.get('growth_gain', 1.0),
        branching_bias=fracture_params.get('branching_bias', 0.20),
        anisotropy_strength=fracture_params.get('anisotropy_strength', 0.10),
        material_family=fracture_params.get('material_family', 'neutral_reference'),
        device=str(device),
    )


def make_surface_fracture_field(fracture_params, graph, device):
    return GaussianFractureField(
        Gc=fracture_params.get('Gc', 60000.0),
        l0=fracture_params.get('l0', 0.025),
        dC_max=fracture_params.get('dC_max', 0.015),
        warmup_frames=fracture_params.get('warmup_frames', 5),
        aniso_ratio=fracture_params.get('aniso_ratio', 3.0),
        opening_scale=fracture_params.get('opening_scale', 0.02),
        damage_source_scale=fracture_params.get('damage_source_scale', 0.35),
        damage_spread=fracture_params.get('damage_spread', 0.18),
        drive_quantile=fracture_params.get('drive_quantile', 0.90),
        front_threshold=fracture_params.get('front_threshold', 0.05),
        radial_bias=fracture_params.get('radial_bias', 2.5),
        tip_propagation_scale=fracture_params.get('tip_propagation_scale', 0.75),
        front_substeps=fracture_params.get('front_substeps', 2),
        tau_init=fracture_params.get('tau_init', 0.30),
        growth_gain=fracture_params.get('growth_gain', 1.0),
        band_width=fracture_params.get('band_width', 1.5),
        band_fill_gain=fracture_params.get('band_fill_gain', 0.30),
        open_gain=fracture_params.get('open_gain', 1.0),
        material_family=fracture_params.get('material_family', 'neutral_reference'),
        enable_front_propagation=fracture_params.get('enable_front_propagation', True),
        material_drive_floor=fracture_params.get('material_drive_floor', None),
        diffuse_damage_gain=fracture_params.get('diffuse_damage_gain', 0.16),
        diffuse_neighborhood_steps=fracture_params.get('diffuse_neighborhood_steps', 2),
        graph=graph,
        crack_front=make_crack_front(fracture_params, device),
        device=str(device),
    )


def make_fragment_manager(fracture_params, device):
    if not fracture_params.get('fragmentation_enabled', False):
        return None
    return GraphFragmentManager(
        damage_threshold=fracture_params.get('fragment_damage_threshold', 0.5),
        min_fragment_size=fracture_params.get('min_fragment_particles', 20),
        edge_break_rate=fracture_params.get('edge_break_rate', 1.0),
        opening_weight=fracture_params.get('fragment_opening_weight', 0.35),
        active_tip_weight=fracture_params.get('fragment_active_tip_weight', 0.18),
        recent_front_weight=fracture_params.get('fragment_recent_front_weight', 0.12),
        pair_break_weight=fracture_params.get('fragment_pair_break_weight', 0.10),
        edge_memory_decay=fracture_params.get('fragment_edge_memory_decay', 0.97),
        edge_memory_weight=fracture_params.get('fragment_edge_memory_weight', 0.72),
        cut_diffusion_alpha=fracture_params.get('fragment_cut_diffusion_alpha', 0.0),
        cut_diffusion_iters=fracture_params.get('fragment_cut_diffusion_iters', 0),
        cut_cos_gate_tangent=fracture_params.get('fragment_cut_cos_gate_tangent', 0.5),
        cut_cos_gate_normal=fracture_params.get('fragment_cut_cos_gate_normal', 0.4),
        primary_cut_ratio=fracture_params.get('fragment_primary_cut_ratio', 0.75),
        fallback_cut_ratio=fracture_params.get('fragment_fallback_cut_ratio', 0.55),
        min_boundary_edges=fracture_params.get('fragment_min_boundary_edges', 12),
        detached_node_decay=fracture_params.get('fragment_detached_node_decay', 0.95),
        persistent_min_fragment_size=fracture_params.get('fragment_persistent_min_size', 8),
        component_hysteresis=fracture_params.get('fragment_component_hysteresis', 0.35),
        post_split_threshold_scale=fracture_params.get('fragment_post_split_threshold_scale', 0.92),
        cut_surface_enable=fracture_params.get('cut_surface_enable', False),
        cut_vote_strength=fracture_params.get('cut_vote_strength', 0.0),
        tau_cross=fracture_params.get('tau_cross', 0.60),
        tau_tangent=fracture_params.get('tau_tangent', 0.45),
        cut_core_damage_threshold=fracture_params.get('cut_core_damage_threshold', 0.18),
        cut_core_opening_threshold=fracture_params.get('cut_core_opening_threshold', 0.16),
        cut_hard_break_threshold=fracture_params.get('cut_hard_break_threshold', 0.42),
        authoritative_cut_decay=fracture_params.get('authoritative_cut_decay', 0.96),
        authoritative_cut_threshold=fracture_params.get('authoritative_cut_threshold', 0.20),
        support_loss_enable=fracture_params.get('support_loss_enable', True),
        support_anchor_quantile=fracture_params.get('support_anchor_quantile', 0.10),
        support_release_threshold=fracture_params.get('support_release_threshold', 0.56),
        support_promote_min_size=fracture_params.get('support_promote_min_size', 6),
        support_overlap_threshold=fracture_params.get('support_overlap_threshold', 0.10),
        open_crack_release_enable=fracture_params.get('open_crack_release_enable', True),
        open_crack_release_threshold=fracture_params.get('open_crack_release_threshold', 0.0),
        open_crack_release_max_patches=fracture_params.get('open_crack_release_max_patches', 2),
        material_family=fracture_params.get('material_family', 'neutral_reference'),
        device=str(device),
    )


def make_fast_collapse_state(num_nodes, device):
    return {
        "labels": torch.zeros(num_nodes, dtype=torch.long, device=device),
        "next_id": 1,
        "released_nodes": 0,
        "last_release_max": 0.0,
        "motion": {},
    }


def update_fast_collapse_labels_from_fields(
    positions,
    damage,
    opening,
    visited,
    tips,
    state,
    fracture_params,
    frame,
):
    labels = state["labels"]
    start_frame = int(fracture_params.get('collapse_fast_start_frame', 4))
    if frame < start_frame:
        return labels, int(labels.max().item()) + 1, 0

    c = damage.clamp(0.0, 1.0)
    opening_scale = torch.quantile(opening.detach(), 0.90).clamp(min=1e-8)
    opening_norm = (opening / opening_scale).clamp(0.0, 1.0)
    elapsed = max(frame - start_frame, 0)
    threshold = max(
        float(fracture_params.get('collapse_fast_min_threshold', 0.055)),
        float(fracture_params.get('collapse_fast_threshold', 0.30))
        - elapsed * float(fracture_params.get('collapse_fast_threshold_decay', 0.004)),
    )
    release = (
        0.48 * c
        + 0.24 * opening_norm
        + 0.20 * visited.float()
        + 0.08 * tips.float()
    ).clamp(0.0, 1.0)
    state["last_release_max"] = float(release.max().item())

    unassigned = labels == 0
    candidate_idx = torch.where(unassigned & (release >= threshold))[0]
    if candidate_idx.numel() == 0:
        return labels, int(labels.max().item()) + 1, 0

    per_frame = max(int(fracture_params.get('collapse_fast_patches_per_frame', 8)), 1)
    order = release[candidate_idx].argsort(descending=True)
    seed_idx = candidate_idx[order[: min(candidate_idx.numel(), per_frame * 2)]]

    bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
    diag = float(bbox_extent.norm().item())
    radius = max(0.008, float(fracture_params.get('collapse_fast_patch_radius', 0.060)) * diag)
    core_radius = max(0.004, float(fracture_params.get('collapse_fast_patch_core_radius', 0.020)) * diag)
    min_size = max(int(fracture_params.get('collapse_fast_min_size', 16)), 1)
    max_size = max(
        min_size,
        int(round(float(fracture_params.get('collapse_fast_max_size_ratio', 0.035)) * positions.shape[0])),
    )

    created = 0
    for seed in seed_idx.tolist():
        if created >= per_frame:
            break
        if labels[seed] != 0:
            continue
        seed_pos = positions[seed]
        dist = torch.norm(positions - seed_pos.unsqueeze(0), dim=1)
        patch = (
            (labels == 0)
            & (dist <= radius)
            & ((release >= 0.38 * threshold) | (dist <= core_radius))
        )
        patch_size = int(patch.sum().item())
        if patch_size < min_size:
            continue
        if patch_size > max_size:
            patch_idx = torch.where(patch)[0]
            keep = patch_idx[dist[patch_idx].argsort()[:max_size]]
            patch.zero_()
            patch[keep] = True
            patch[seed] = True

        labels[patch] = int(state["next_id"])
        state["next_id"] += 1
        created += 1

    state["released_nodes"] = int((labels > 0).sum().item())
    return labels, int(labels.max().item()) + 1, created


def apply_fragment_motion_offsets(
    positions,
    labels,
    state,
    frame,
    fracture_params,
):
    if labels is None or not bool((labels > 0).any()):
        return positions

    motion = state.setdefault("motion", {})
    out = positions.clone()
    base_com = positions[labels == 0].mean(dim=0) if bool((labels == 0).any()) else positions.mean(dim=0)
    gravity = float(fracture_params.get('collapse_fragment_gravity', 0.0018))
    fall_cap = float(fracture_params.get('collapse_fragment_fall_cap', 0.42))
    speed = float(fracture_params.get('collapse_fragment_speed', 0.010))
    gap = float(fracture_params.get('collapse_fragment_gap', 0.018))
    max_offset = float(fracture_params.get('collapse_fragment_max_offset', 0.24))
    spin_gain = float(fracture_params.get('collapse_fragment_spin_gain', 0.18))

    for frag_id in labels.unique(sorted=True).tolist():
        if frag_id <= 0:
            continue
        mask = labels == frag_id
        if not bool(mask.any()):
            continue
        if frag_id not in motion:
            com = positions[mask].mean(dim=0)
            direction = com - base_com
            if direction.norm() <= 1e-8:
                direction = torch.tensor(
                    [
                        torch.sin(torch.tensor(float(frag_id), device=positions.device)),
                        torch.cos(torch.tensor(float(frag_id) * 1.7, device=positions.device)),
                        torch.tensor(0.25, device=positions.device),
                    ],
                    dtype=positions.dtype,
                    device=positions.device,
                )
            direction = direction / direction.norm().clamp(min=1e-8)
            jitter = torch.tensor(
                [
                    0.35 * torch.sin(torch.tensor(float(frag_id) * 12.989, device=positions.device)),
                    0.35 * torch.sin(torch.tensor(float(frag_id) * 78.233, device=positions.device)),
                    0.18 * torch.cos(torch.tensor(float(frag_id) * 37.719, device=positions.device)),
                ],
                dtype=positions.dtype,
                device=positions.device,
            )
            direction = direction + jitter
            direction[2] += 0.12
            direction = direction / direction.norm().clamp(min=1e-8)
            motion[frag_id] = {
                "birth": int(frame),
                "direction": direction,
                "phase": float(frag_id) * 0.37,
            }

        frag_state = motion[frag_id]
        age = max(frame - int(frag_state["birth"]), 0)
        ramp = min(age / 12.0, 1.0)
        direction = frag_state["direction"]
        travel = min(max_offset, gap + speed * age)
        offset = ramp * travel * direction
        offset = offset.clone()
        offset[2] -= min(fall_cap, gravity * float(age * age))

        local = positions[mask]
        com = local.mean(dim=0)
        rel = local - com.unsqueeze(0)
        spin_phase = float(frag_state["phase"]) + 0.11 * age
        spin_axis = torch.tensor(
            [torch.sin(torch.tensor(spin_phase, device=positions.device)), 0.35, torch.cos(torch.tensor(spin_phase, device=positions.device))],
            dtype=positions.dtype,
            device=positions.device,
        )
        spin_axis = spin_axis / spin_axis.norm().clamp(min=1e-8)
        spin = spin_gain * ramp * min(age / 30.0, 1.0)
        spin_offset = spin * torch.cross(
            spin_axis.unsqueeze(0).expand_as(rel),
            rel,
            dim=1,
        )
        out[mask] = local + offset.unsqueeze(0) + spin_offset

    return out


def update_fast_collapse_fragments(
    positions,
    fracture_field,
    state,
    fracture_params,
    frame,
):
    labels = state["labels"]
    start_frame = int(fracture_params.get('collapse_fast_start_frame', 4))
    if frame < start_frame:
        return labels, int(labels.max().item()) + 1, 0

    c = fracture_field.c.clamp(0.0, 1.0)
    opening = fracture_field.a if fracture_field.a is not None else torch.zeros_like(c)
    opening_scale = torch.quantile(opening.detach(), 0.90).clamp(min=1e-8)
    opening_norm = (opening / opening_scale).clamp(0.0, 1.0)

    crack_front = fracture_field.crack_front
    visited = (
        crack_front.visited_mask.float()
        if crack_front is not None and crack_front.visited_mask is not None
        else torch.zeros_like(c)
    )
    tips = (
        crack_front.tip_mask.float()
        if crack_front is not None and crack_front.tip_mask is not None
        else torch.zeros_like(c)
    )

    elapsed = max(frame - start_frame, 0)
    threshold = max(
        float(fracture_params.get('collapse_fast_min_threshold', 0.055)),
        float(fracture_params.get('collapse_fast_threshold', 0.30))
        - elapsed * float(fracture_params.get('collapse_fast_threshold_decay', 0.004)),
    )
    release = (
        0.48 * c
        + 0.24 * opening_norm
        + 0.20 * visited
        + 0.08 * tips
    ).clamp(0.0, 1.0)
    state["last_release_max"] = float(release.max().item())

    unassigned = labels == 0
    candidate_idx = torch.where(unassigned & (release >= threshold))[0]
    if candidate_idx.numel() == 0:
        return labels, int(labels.max().item()) + 1, 0

    per_frame = max(int(fracture_params.get('collapse_fast_patches_per_frame', 8)), 1)
    order = release[candidate_idx].argsort(descending=True)
    seed_idx = candidate_idx[order[: min(candidate_idx.numel(), per_frame * 3)]]

    bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
    diag = float(bbox_extent.norm().item())
    radius = max(0.008, float(fracture_params.get('collapse_fast_patch_radius', 0.060)) * diag)
    core_radius = max(0.004, float(fracture_params.get('collapse_fast_patch_core_radius', 0.020)) * diag)
    min_size = max(int(fracture_params.get('collapse_fast_min_size', 16)), 1)
    max_size = max(
        min_size,
        int(round(float(fracture_params.get('collapse_fast_max_size_ratio', 0.035)) * positions.shape[0])),
    )

    created = 0
    for seed in seed_idx.tolist():
        if created >= per_frame:
            break
        if labels[seed] != 0:
            continue
        seed_pos = positions[seed]
        dist = torch.norm(positions - seed_pos.unsqueeze(0), dim=1)
        patch = (
            (labels == 0)
            & (dist <= radius)
            & ((release >= 0.42 * threshold) | (dist <= core_radius))
        )
        patch_size = int(patch.sum().item())
        if patch_size < min_size:
            continue
        if patch_size > max_size:
            patch_idx = torch.where(patch)[0]
            keep = patch_idx[dist[patch_idx].argsort()[:max_size]]
            patch.zero_()
            patch[keep] = True
            patch[seed] = True

        labels[patch] = int(state["next_id"])
        state["next_id"] += 1
        created += 1

    state["released_nodes"] = int((labels > 0).sum().item())
    return labels, int(labels.max().item()) + 1, created


def run_vector_collapse_smoke(
    positions,
    normals,
    fracture_params,
    frames,
    out_dir,
    device,
    plot_every=10,
):
    material_family = fracture_params.get('material_family', 'rough_quasi_brittle')
    driver = SurfaceCrackDriver(material_family=material_family)
    state = make_fast_collapse_state(positions.shape[0], device)
    damage = torch.zeros(positions.shape[0], dtype=positions.dtype, device=device)
    opening = torch.zeros_like(damage)
    visited = torch.zeros(positions.shape[0], dtype=torch.bool, device=device)
    tips = torch.zeros_like(visited)

    seed_center = driver.default_impact_center(positions)
    seed_dist = torch.norm(positions - seed_center.unsqueeze(0), dim=1)
    seed = torch.exp(-0.5 * (seed_dist / max(driver.params.impact_radius, 1e-6)) ** 2)
    damage = torch.maximum(damage, 0.22 * seed.clamp(0.0, 1.0))

    noise = torch.sin(
        41.0 * positions[:, 0]
        - 23.0 * positions[:, 1]
        + 17.0 * positions[:, 2]
    )
    noise = (0.5 + 0.5 * noise).clamp(0.0, 1.0)

    print(f"\n{'='*50}")
    print(f"Running {frames} frames (vectorized surface collapse smoke)")
    print(f"Family: {material_family}")
    print(f"{'='*50}\n")

    plot_every = max(1, int(plot_every))
    first_split_frame = None
    max_n_frags = 1
    max_released_nodes = 0
    t0 = time.time()

    for frame in range(frames):
        drive = driver.build(positions, frame, normals=normals)
        growth = drive["growth_drive"].clamp(0.0, 1.0)
        init_score = drive["init_score"].clamp(0.0, 1.0)
        tip_threshold = torch.quantile(growth.detach(), 0.992)
        tips = growth >= tip_threshold
        visit_threshold = max(0.16, 0.72 - 0.006 * frame)
        visited |= (growth >= visit_threshold)

        collapse_phase = max(0.0, min(1.0, (frame - 18) / max(frames - 18, 1)))
        global_damage = collapse_phase * (0.08 + 0.22 * noise)
        damage = torch.maximum(damage, 0.18 * init_score)
        damage = (
            damage
            + 0.060 * growth
            + 0.018 * visited.float()
            + 0.012 * tips.float()
        ).clamp(0.0, 1.0)
        damage = torch.maximum(damage, global_damage.clamp(0.0, 0.42))
        opening = 0.045 * (damage ** 2)

        labels, n_frags, created = update_fast_collapse_labels_from_fields(
            positions=positions,
            damage=damage,
            opening=opening,
            visited=visited,
            tips=tips,
            state=state,
            fracture_params=fracture_params,
            frame=frame,
        )
        if n_frags > 1 and first_split_frame is None:
            first_split_frame = frame
        max_n_frags = max(max_n_frags, n_frags)
        max_released_nodes = max(max_released_nodes, int(state["released_nodes"]))

        if frame % plot_every == 0 or frame == frames - 1:
            render_positions = apply_fragment_motion_offsets(
                positions=positions,
                labels=labels,
                state=state,
                frame=frame,
                fracture_params=fracture_params,
            )
            plot_fracture_frame(
                render_positions,
                damage,
                frame,
                out_dir,
                file_prefix="surface_crack",
                title_extra=f"  |  displaced n_frags={n_frags} released={state['released_nodes']}",
                visited=visited,
                tips=tips,
                fragment_ids=labels,
                n_fragments=n_frags,
            )
            plot_opening_frame(
                render_positions,
                opening,
                frame,
                out_dir,
                title_extra=f"  |  vector_collapse",
            )
            if n_frags > 1:
                plot_fragment_frame(
                    render_positions,
                    labels,
                    frame,
                    out_dir,
                    title_extra=f"  |  released={state['released_nodes']} new={created}",
                    file_prefix="surface_fragment",
                )
            elapsed = time.time() - t0
            print(
                f"Frame {frame:3d}/{frames} "
                f"c_max={float(damage.max().item()):.4f} "
                f"n_frags={n_frags} released={state['released_nodes']} "
                f"elapsed={elapsed:.1f}s"
            )

    print(f"\n{'='*50}")
    print("Vector collapse smoke complete")
    print(f"Output: {out_dir}")
    print(f"  family = {material_family}")
    print(f"  first_split_frame = {first_split_frame}")
    print(f"  max_n_frags = {max_n_frags}")
    print(f"  max_released_nodes = {max_released_nodes}")
    print(f"{'='*50}\n")


def run_surface_only_smoke(
    positions,
    normals,
    fracture_params,
    frames,
    out_dir,
    device,
    plot_every=5,
    fragment_every=None,
):
    material_family = fracture_params.get('material_family', 'neutral_reference')
    graph = GaussianGraph(
        k=fracture_params.get('graph_k', 12),
        sigma=fracture_params.get('graph_sigma', 0.03),
        rebuild_every=fracture_params.get('graph_rebuild_every', 5),
        device=str(device),
    )
    if normals is not None:
        graph.set_normals(torch.nn.functional.normalize(normals, dim=-1))

    fracture_field = make_surface_fracture_field(fracture_params, graph, device)
    fracture_field.initialize(positions.shape[0])
    fast_collapse = bool(fracture_params.get('collapse_fast_path', False))
    fragment_manager = None if fast_collapse else make_fragment_manager(fracture_params, device)
    fast_collapse_state = (
        make_fast_collapse_state(positions.shape[0], device)
        if fast_collapse else None
    )
    driver = SurfaceCrackDriver(material_family=material_family)
    seed_center = driver.default_impact_center(positions)
    fracture_field.seed_damage(
        positions=positions,
        center=seed_center,
        radius=driver.params.impact_radius,
        magnitude=float(fracture_params.get('impact_seed_magnitude', 0.18)),
        H_multiplier=0.0,
    )

    print(f"\n{'='*50}")
    print(f"Running {frames} frames (surface-only matplotlib smoke)")
    print(f"Family: {material_family}")
    if fast_collapse:
        print("Mode: fast collapse release")
    print(f"{'='*50}\n")

    plot_every = max(1, int(plot_every))
    detect_every = max(
        1,
        int(
            fragment_every
            if fragment_every is not None
            else fracture_params.get('fragment_detect_every', 2)
        ),
    )

    first_split_frame = None
    max_n_frags = 1
    max_cut_edges = 0
    max_closure_candidate_count = 0
    max_released_nodes = 0
    t0 = time.time()

    for frame in range(frames):
        drive = driver.build(positions, frame, normals=normals)
        fracture_field.update(
            positions=positions,
            init_score=drive["init_score"],
            growth_drive=drive["growth_drive"],
            growth_dir=drive["growth_dir"],
            F_gaussian=None,
            impact_center=drive["impact_center"],
        )

        n_frags = (
            int(fragment_manager.n_fragments)
            if fragment_manager is not None and fragment_manager.fragment_ids is not None
            else 1
        )
        fast_fragment_ids = None
        created_fast = 0
        if fast_collapse:
            fast_fragment_ids, n_frags, created_fast = update_fast_collapse_fragments(
                positions=positions,
                fracture_field=fracture_field,
                state=fast_collapse_state,
                fracture_params=fracture_params,
                frame=frame,
            )
            max_released_nodes = max(
                max_released_nodes,
                int(fast_collapse_state["released_nodes"]),
            )
            if n_frags > 1 and first_split_frame is None:
                first_split_frame = frame
        elif fragment_manager is not None:
            should_detect = (
                fragment_manager.fragment_ids is None
                or frame % detect_every == 0
                or frame == frames - 1
            )
            if should_detect:
                crack_front = fracture_field.crack_front
                tip_mask = crack_front.tip_mask if crack_front is not None else None
                recent_front_mask = None
                if crack_front is not None and crack_front.visited_mask is not None:
                    recent_front_mask = crack_front.visited_mask & (fracture_field.c > 0.12)
                n_frags = fragment_manager.detect_fragments(
                    graph,
                    fracture_field.c,
                    positions=positions,
                    opening=fracture_field.a,
                    active_tip_mask=tip_mask,
                    recent_front_mask=recent_front_mask,
                    crack_normal=fracture_field.n,
                    crack_tangent=crack_front.growth_dir if crack_front is not None else None,
                )
                max_cut_edges = max(max_cut_edges, fragment_manager.last_cut_edges)
                max_closure_candidate_count = max(
                    max_closure_candidate_count,
                    fragment_manager.last_closure_candidate_count,
                )
                if n_frags > 1 and first_split_frame is None:
                    first_split_frame = frame

        max_n_frags = max(max_n_frags, n_frags)

        if frame % plot_every == 0 or frame == frames - 1:
            crack_front = fracture_field.crack_front
            plot_fracture_frame(
                positions,
                fracture_field.c,
                frame,
                out_dir,
                file_prefix="surface_crack",
                title_extra=f"  |  n_frags={n_frags}",
                visited=crack_front.visited_mask if crack_front is not None else None,
                tips=crack_front.tip_mask if crack_front is not None else None,
                fragment_ids=(
                    fast_fragment_ids
                    if fast_collapse
                    else fragment_manager.fragment_ids if fragment_manager is not None else None
                ),
                n_fragments=n_frags,
            )
            plot_opening_frame(
                positions,
                fracture_field.a,
                frame,
                out_dir,
                title_extra=f"  |  family={material_family}",
            )
            if fast_collapse and n_frags > 1:
                plot_fragment_frame(
                    positions,
                    fast_fragment_ids,
                    frame,
                    out_dir,
                    title_extra=(
                        f"  |  released={fast_collapse_state['released_nodes']} "
                        f"new={created_fast}"
                    ),
                    file_prefix="surface_fragment",
                )
            if (not fast_collapse) and fragment_manager is not None:
                plot_cut_debug_frame(
                    positions,
                    graph,
                    fragment_manager.last_cut_core_mask,
                    fragment_manager.last_cut_edge_mask,
                    frame,
                    out_dir,
                    title_extra=(
                        f"  cut_edges={fragment_manager.last_cut_edges} "
                        f"cross={fragment_manager.last_cross_edge_breaks}"
                    ),
                )
                if n_frags > 1:
                    plot_fragment_frame(
                        positions,
                        fragment_manager.fragment_ids,
                        frame,
                        out_dir,
                        title_extra=f"  |  family={material_family}",
                        file_prefix="surface_fragment",
                    )
                if fragment_manager.last_closure_candidate_count > 0 or frame == frames - 1:
                    plot_closure_debug_frame(
                        positions,
                        graph,
                        fragment_manager.last_closure_candidate_mask,
                        fragment_manager.last_closure_boundary_mask,
                        frame,
                        out_dir,
                        title_extra=(
                            f"  closure={fragment_manager.last_closure_candidate_count} "
                            f"score={fragment_manager.last_closure_score_max:.2f}"
                        ),
                    )

        if frame % plot_every == 0 or frame == frames - 1:
            elapsed = time.time() - t0
            print(
                f"Frame {frame:3d}/{frames} "
                f"c_max={float(fracture_field.c.max().item()):.4f} "
                f"n_frags={n_frags} cut_edges={max_cut_edges} "
                f"released={max_released_nodes} "
                f"elapsed={elapsed:.1f}s"
            )

    print(f"\n{'='*50}")
    print("Surface-only smoke complete")
    print(f"Output: {out_dir}")
    print(f"  family = {material_family}")
    print(f"  first_split_frame = {first_split_frame}")
    print(f"  max_n_frags = {max_n_frags}")
    print(f"  max_cut_edges = {max_cut_edges}")
    print(f"  max_closure_candidate_count = {max_closure_candidate_count}")
    print(f"  max_released_nodes = {max_released_nodes}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Manifold fracture smoke test")
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--particles",
        type=int,
        default=50000,
        help="Particle budget for matplotlib smoke tests (default: 50k)",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=150000,
        help="Hard cap for smoke-test particles (default: 150k)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Compatibility alias for --particles 50000 --mpm-grids 64 --substeps 8",
    )
    parser.add_argument("--mpm-grids", type=int, default=64)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help="Run the new surface graph driver without MPM substeps",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=5,
        help="Matplotlib diagnostic frame interval for smoke tests",
    )
    parser.add_argument(
        "--fragment-every",
        type=int,
        default=None,
        help="Override fragment connected-component detection interval",
    )
    parser.add_argument("--out", default="output/smoke_test")
    parser.add_argument("--clip", type=str, default=None, help="Material prompt for CLIP-driven priors")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument(
        "--family",
        choices=sorted(FAMILY_RUNTIME_PRESETS.keys()),
        default=None,
        help="Direct material-family runtime preset override for fracture smoke tests",
    )
    parser.add_argument(
        "--collapse",
        action="store_true",
        help="Surface-only stress test preset that aggressively releases fragments",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Config ---
    config = load_config(args.config)
    particle_budget = 50000 if args.fast else int(args.particles)
    particle_budget = max(1000, min(particle_budget, int(args.max_particles)))
    OmegaConf.update(config, "particles.target_count", particle_budget)
    OmegaConf.update(config, "mpm.num_grids", int(args.mpm_grids))
    OmegaConf.update(config, "rendering.physics_substeps", int(args.substeps))
    print(
        f"[SmokeTest] particle_budget={particle_budget} "
        f"max_particles={int(args.max_particles)} "
        f"grid={int(args.mpm_grids)} substeps={int(args.substeps)}"
    )

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

    if args.family:
        config = apply_overrides_dict(config, FAMILY_RUNTIME_PRESETS[args.family])
        print(f"[SmokeTest:family] Applied family preset: {args.family}")
    if args.collapse:
        config = apply_overrides_dict(config, SURFACE_COLLAPSE_OVERRIDES)
        print("[SmokeTest:collapse] Applied aggressive surface-collapse preset")

    config = resolve_material_preset(config)
    config = validate_l0(config)

    # --- Mesh → Point Clouds ---
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("Generating point clouds...")
    surface_sample_ratio = (
        1.0
        if args.surface_only
        else config.particles.get('surface_ratio', 0.5)
    )
    converter = MeshToPointCloudConverter(
        mesh_path=config.mesh.path,
        target_particle_count=config.particles.target_count,
        surface_sample_ratio=surface_sample_ratio,
        use_poisson=config.particles.get('use_poisson_sampling', False),
        normalize_to_unit_cube=config.particles.get('normalize_to_unit_cube', True),
    )
    volume_pcd, surface_pcd, surface_mask_np = converter.convert()
    volume_points = np.asarray(volume_pcd.points).copy()
    volume_normals = np.asarray(volume_pcd.normals).copy()
    N_total = volume_points.shape[0]
    N_surf = surface_mask_np.sum()
    print(f"Particles: {N_total} total, {N_surf} surface")

    # --- Fracture params ---
    pf_params = OmegaConf.to_container(config.phase_field, resolve=True)
    manifold_cfg = OmegaConf.to_container(
        config.get('manifold', {}), resolve=True) if hasattr(config, 'manifold') else {}
    fracture_params = {
        **pf_params,
        **manifold_cfg,
        "Gc": float(config.material.Gc),
        "l0": float(config.material.l0),
    }

    if args.surface_only:
        x_surf = torch.from_numpy(np.asarray(surface_pcd.points).copy()).float().to(device)
        n_surf = torch.from_numpy(np.asarray(surface_pcd.normals).copy()).float().to(device)
        print(
            f"Surface-only mode: using {x_surf.shape[0]} surface graph nodes "
            f"and no volume/MPM particles"
        )
        if fracture_params.get('collapse_vector_path', False):
            run_vector_collapse_smoke(
                positions=x_surf,
                normals=n_surf,
                fracture_params=fracture_params,
                frames=args.frames,
                out_dir=out_dir,
                device=device,
                plot_every=args.plot_every,
            )
            return
        run_surface_only_smoke(
            positions=x_surf,
            normals=n_surf,
            fracture_params=fracture_params,
            frames=args.frames,
            out_dir=out_dir,
            device=device,
            plot_every=args.plot_every,
            fragment_every=args.fragment_every,
        )
        return

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
