"""Validate sentence-conditioned material priors and crack morphology.

This script is intentionally matplotlib/render-free by default. It answers:

    sentence -> CLIP retrieval -> material/fracture prior -> surface crack metrics

The goal is to verify that different text prompts produce different material
parameters and different fracture morphology before any Gaussian rendering pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import apply_overrides_dict, load_config
from src.core.material_presets import resolve_material_preset, validate_l0
from src.fracture.graph_builder import GaussianGraph
from src.fracture.surface_crack_driver import SurfaceCrackDriver
from src.ml.material_predictor import MaterialPredictor
from src.ml.material_prior_adapter import MaterialPriorAdapter
from src.preprocessing.mesh_converter import MeshToPointCloudConverter

from smoke_test import (
    make_fragment_manager,
    make_surface_fracture_field,
)


DEFAULT_PROMPTS = [
    "thin glass bottle shattering into sharp clean cracks",
    "porcelain ceramic mug cracking with a few brittle splits",
    "rough concrete block crumbling into irregular chunks",
    "soft rubber ball deforming without visible fracture",
    "dry sandstone statue breaking into gritty rough pieces",
]


def _flatten_for_csv(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value_f) >= 1.0e4 or (0.0 < abs(value_f) < 1.0e-2):
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def _release_mode_for_row(row: dict) -> str:
    family = str(row.get("family", ""))
    style = str(row.get("sentence_style", row.get("crack_style", "")))
    prompt = str(row.get("prompt", "")).lower()
    if (
        family == "diffuse_damage"
        or (family == "neutral_reference" and ("steel" in prompt or "metal" in prompt))
        or style == "diffuse_microcrack"
        or "without visible fracture" in prompt
        or "no visible fracture" in prompt
        or "without brittle fracture" in prompt
        or "denting" in prompt
    ):
        return "no_fragment"
    if style == "single_smooth":
        return "crack_split"
    if bool(row.get("runtime_crack_connected_release_only", False)):
        return "crack_connected_fragment"
    if family == "sharp_brittle" and style == "radial_shatter":
        return "complete_shatter"
    if family == "sharp_brittle" and style == "spiderweb_branching":
        return "fragmented_web"
    if family == "rough_quasi_brittle" and style == "chunky_crumble":
        return "chunk_release"
    if family in {"sharp_brittle", "brittle_moderate", "rough_quasi_brittle"}:
        return "partial_fragment"
    return "crack_only"


def _release_verdict_for_row(row: dict) -> tuple[str, str]:
    mode = _release_mode_for_row(row)
    max_frags = max(
        int(row.get("max_n_fragments", row.get("max_n_frags", 0)) or 0),
        int(row.get("final_component_count", 0) or 0),
    )
    final_frags = max(
        int(row.get("final_n_fragments", row.get("final_fragment_label_count", max_frags)) or 0),
        int(row.get("final_component_count", 0) or 0),
    )
    released = float(row.get("final_released_node_ratio", 0.0) or 0.0)
    largest = float(row.get("final_largest_fragment_ratio", 1.0) or 1.0)
    detach = float(row.get("max_physical_detached_distance", row.get("physical_detached_distance", 0.0)) or 0.0)
    drop = float(row.get("max_physical_fragment_drop", row.get("physical_fragment_drop", 0.0)) or 0.0)
    moved = max(detach, drop)
    family = str(row.get("family", ""))
    style = str(row.get("sentence_style", row.get("crack_style", "")))
    bcut = float(row.get("final_bcut", row.get("fragment_boundary_cut_support_ratio", 0.0)) or 0.0)
    first_detach = int(row.get("first_detach_since_impact", -1) or -1)
    branch_events = int(row.get("branch_event_count", row.get("final_branch_event_count", 0)) or 0)
    nonclosure_patches = int(row.get("max_open_release_patches", 0) or 0)
    has_motion_metrics = (
        "max_physical_detached_distance" in row
        or "max_physical_fragment_drop" in row
        or "physical_detached_distance" in row
        or "physical_fragment_drop" in row
    )

    # Resolution scaling: thresholds were calibrated at 10K particles.
    # At higher resolutions a structurally identical fracture pattern
    # produces proportionally more graph-level fragment labels because
    # the surface graph is denser.  We scale fragment-count thresholds
    # with sqrt(N/10000) so a 50K run is judged on the same physical
    # bar as a 10K run.  Read total node count from the row's
    # `gravity_particles` field (injected by the inspect script at row
    # build time); fall back to 10000 (no scaling) when absent.
    n_particles = float(row.get("gravity_particles", row.get("n_particles", 10000)) or 10000)
    res_scale = max((n_particles / 10000.0) ** 0.5, 1.0)

    passed = False
    if mode == "crack_connected_fragment":
        if family == "sharp_brittle" and style == "radial_shatter":
            min_frag_radial = max(int(round(24 * res_scale)), 24)
            min_branch_radial = max(int(round(32 * res_scale)), 32)
            passed = (
                max_frags >= min_frag_radial
                and released >= 0.35
                and released <= 0.68
                and largest <= 0.65
                and bcut >= 0.35
                and branch_events >= min_branch_radial
                and nonclosure_patches == 0
                and (first_detach < 0 or first_detach <= 2)
                and (moved <= 0.35 or not has_motion_metrics)
            )
        else:
            passed = (
                max_frags >= 2
                and released >= 0.01
                and released <= 0.55
                and largest >= 0.40
                and branch_events >= 2
                and nonclosure_patches == 0
                and (moved <= 0.35 or not has_motion_metrics)
            )
    elif mode == "complete_shatter":
        min_frag = max(int(round(48 * res_scale)), 48)
        passed = max_frags >= min_frag and released >= 0.40 and largest <= 0.70 and (moved >= 0.04 or not has_motion_metrics)
    elif mode == "fragmented_web":
        min_frag = max(int(round(16 * res_scale)), 16)
        passed = max_frags >= min_frag and released >= 0.12 and largest <= 0.88 and (moved >= 0.025 or not has_motion_metrics)
    elif mode == "chunk_release":
        min_frag = max(int(round(8 * res_scale)), 8)
        passed = max_frags >= min_frag and released >= 0.08 and largest <= 0.92 and (moved >= 0.02 or not has_motion_metrics)
    elif mode == "crack_split":
        max_cap = max(int(round(12 * res_scale)), 12)
        passed = 1 <= max_frags <= max_cap and largest >= 0.50 and nonclosure_patches == 0
    elif mode == "partial_fragment":
        passed = max_frags >= 2 and released >= 0.03
    elif mode == "no_fragment":
        # Tolerance scales mildly with resolution for graph-label noise
        # at 50K+ where a single rogue patch can survive numerically.
        max_residual_frags = max(int(round(1 * res_scale)), 1)
        passed = (
            final_frags <= max_residual_frags
            and released <= 0.01
            and moved <= 0.01
            and nonclosure_patches == 0
        )
    else:
        max_cap = max(int(round(4 * res_scale)), 4)
        passed = max_frags <= max_cap
    return mode, "PASS" if passed else "FAIL"


def _short_prompt(prompt: str, max_len: int = 42) -> str:
    prompt = " ".join(str(prompt).split())
    if len(prompt) <= max_len:
        return prompt
    return prompt[: max_len - 3] + "..."


def _slug(text: str, max_len: int = 64) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower())
    text = text.strip("_")
    return (text or "prompt")[:max_len]


def _save_final_crack_plot(
    positions: torch.Tensor,
    damage: torch.Tensor,
    visited: torch.Tensor,
    tips: torch.Tensor,
    out_path: Path,
    title: str,
    fragment_ids: torch.Tensor | None = None,
    parent_index: torch.Tensor | None = None,
    activation_step: torch.Tensor | None = None,
    cracked_threshold: float = 0.30,
    draw_cross_fragment_segments: bool = True,
) -> None:
    """Save a compact final-frame crack morphology diagnostic."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot:warn] matplotlib unavailable: {exc}", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = positions.detach().float().cpu().numpy()
    c = damage.detach().float().cpu().numpy()
    visited_np = visited.detach().bool().cpu().numpy()
    tips_np = tips.detach().bool().cpu().numpy()
    cracked_np = c > float(cracked_threshold)
    frag_np = None
    if fragment_ids is not None:
        frag_np = fragment_ids[: positions.shape[0]].detach().long().cpu().numpy()
    segment_np = None
    if parent_index is not None:
        parent = parent_index[: positions.shape[0]].detach().long().cpu()
        valid = visited.detach().bool().cpu() & (parent >= 0) & (parent < positions.shape[0])
        child_idx = torch.where(valid)[0]
        if child_idx.numel() > 0:
            step_np = None
            if activation_step is not None:
                step_np = activation_step[: positions.shape[0]].detach().long().cpu()[child_idx].numpy()
            segment_np = (
                child_idx.numpy(),
                parent[child_idx].numpy(),
                step_np,
            )

    views = [
        ("X-Y top", 0, 1, "X", "Y"),
        ("X-Z front", 0, 2, "X", "Z"),
        ("Y-Z side", 1, 2, "Y", "Z"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    fig.suptitle(title, fontsize=11)
    scatter_for_colorbar = None

    for ax, (view_name, i, j, xlabel, ylabel) in zip(axes, views):
        ax.scatter(pos[:, i], pos[:, j], s=0.18, c="#d5d8dc", alpha=0.32, linewidths=0)

        if frag_np is not None and (frag_np > 0).any():
            released = frag_np > 0
            ax.scatter(
                pos[released, i],
                pos[released, j],
                s=0.70,
                c=frag_np[released],
                cmap="tab20",
                alpha=0.62,
                linewidths=0,
            )

        if cracked_np.any():
            scatter_for_colorbar = ax.scatter(
                pos[cracked_np, i],
                pos[cracked_np, j],
                s=1.20,
                c=c[cracked_np],
                cmap="inferno",
                vmin=float(cracked_threshold),
                vmax=max(float(c.max()), float(cracked_threshold) + 1e-4),
                alpha=0.92,
                linewidths=0,
            )

        if visited_np.any():
            ax.scatter(
                pos[visited_np, i],
                pos[visited_np, j],
                s=3.2,
                facecolors="none",
                edgecolors="#1f77b4",
                linewidths=0.25,
                alpha=0.78,
            )
        if segment_np is not None:
            child_idx, parent_idx, step_np = segment_np
            norm = None
            cmap = None
            if step_np is not None and step_np.size > 0 and step_np.max() > step_np.min():
                norm = matplotlib.colors.Normalize(vmin=float(step_np.min()), vmax=float(step_np.max()))
                cmap = plt.get_cmap("viridis")
            for seg_i, (child, parent) in enumerate(zip(child_idx, parent_idx)):
                if (
                    not draw_cross_fragment_segments
                    and frag_np is not None
                    and child < frag_np.shape[0]
                    and parent < frag_np.shape[0]
                    and frag_np[child] != frag_np[parent]
                ):
                    continue
                color = "#0b4f9c"
                alpha = 0.18
                if norm is not None and cmap is not None:
                    color = cmap(norm(float(step_np[seg_i])))
                    alpha = 0.32
                ax.plot(
                    [pos[parent, i], pos[child, i]],
                    [pos[parent, j], pos[child, j]],
                    color=color,
                    alpha=alpha,
                    linewidth=0.35,
                    solid_capstyle="round",
                )

        if tips_np.any():
            ax.scatter(
                pos[tips_np, i],
                pos[tips_np, j],
                s=18,
                c="#00d7ff",
                marker="x",
                linewidths=0.8,
            )

        ax.set_title(view_name, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.12, linewidth=0.5)

    if scatter_for_colorbar is not None:
        fig.colorbar(scatter_for_colorbar, ax=axes, shrink=0.78, label="damage c")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _normalized_pairwise(rows: list[dict], keys: list[str]) -> list[dict]:
    valid_keys = [key for key in keys if all(key in row for row in rows)]
    if not valid_keys or len(rows) < 2:
        return []

    matrix = np.asarray(
        [[float(row.get(key, 0.0)) for key in valid_keys] for row in rows],
        dtype=np.float64,
    )
    denom = matrix.std(axis=0)
    denom[denom < 1e-9] = 1.0
    matrix = (matrix - matrix.mean(axis=0)) / denom

    distances = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            distances.append({
                "a": _short_prompt(rows[i]["prompt"], 28),
                "b": _short_prompt(rows[j]["prompt"], 28),
                "distance": float(np.linalg.norm(matrix[i] - matrix[j])),
            })
    distances.sort(key=lambda item: item["distance"], reverse=True)
    return distances


def _write_markdown_report(rows: list[dict], path: Path, no_sim: bool) -> None:
    prior_cols = [
        "prompt",
        "family",
        "sentence_style",
        "top1",
        "E_raw",
        "Gc_raw",
        "tau_init",
        "growth_gain",
        "band_width",
        "open_gain",
        "branching_bias",
    ]
    metric_cols = [
        "prompt",
        "c_max",
        "cracked_count",
        "visited_count",
        "tip_count",
        "tip_event_count",
        "branch_event_count",
        "branch_event_angle_mean_deg",
        "leaf_count",
        "junction_count",
        "max_n_frags",
        "final_released_node_ratio",
        "final_largest_fragment_ratio",
        "final_nonbase_fragment_min_size",
        "expected_release_mode",
        "fragment_release_verdict",
        "max_cut_edges",
        "max_open_release_patches",
        "branchiness",
        "visited_angular_coverage",
        "cracked_span_x",
        "cracked_span_y",
        "cracked_span_z",
        "final_fragment_entropy",
    ]
    prior_keys = [
        "E_raw",
        "Gc_raw",
        "nu_raw",
        "density_raw",
        "tau_init",
        "growth_gain",
        "band_width",
        "open_gain",
        "edge_break_rate",
        "branching_bias",
        "anisotropy_strength",
    ]
    metric_keys = [
        "c_max",
        "c_mean",
        "opening_max",
        "visited_count",
        "tip_count",
        "leaf_count",
        "junction_count",
        "max_children",
        "mean_children",
        "cracked_count",
        "weak_count",
        "max_n_frags",
        "max_cut_edges",
        "branchiness",
        "tip_event_count",
        "branch_event_count",
        "branch_event_frame_count",
        "branch_event_angle_mean_deg",
        "branch_event_angle_std_deg",
        "visited_angular_coverage",
        "visited_angular_entropy",
        "cracked_angular_coverage",
        "cracked_angular_entropy",
        "cracked_span_x",
        "cracked_span_y",
        "cracked_span_z",
        "final_released_node_ratio",
        "final_largest_fragment_ratio",
        "final_nonbase_fragment_min_size",
        "final_nonbase_fragment_mean_size",
        "final_fragment_entropy",
    ]

    families = sorted({str(row.get("family", "")) for row in rows})
    cracked_values = [float(row.get("cracked_count", 0.0)) for row in rows if "cracked_count" in row]
    cmax_values = [float(row.get("c_max", 0.0)) for row in rows if "c_max" in row]
    visited_values = [float(row.get("visited_count", 0.0)) for row in rows if "visited_count" in row]
    frag_values = [float(row.get("max_n_frags", 0.0)) for row in rows if "max_n_frags" in row]
    junction_values = [float(row.get("junction_count", 0.0)) for row in rows if "junction_count" in row]

    lines = [
        "# Sentence Material Validation",
        "",
        "## Verdict",
        "",
        f"- prompts: {len(rows)}",
        f"- distinct material families: {len(families)} ({', '.join(families)})",
    ]
    if not no_sim and cracked_values:
        verdict_values = [str(row.get("fragment_release_verdict", "")) for row in rows]
        fail_count = sum(1 for value in verdict_values if value == "FAIL")
        lines.extend([
            f"- cracked_count range: {_fmt(min(cracked_values), 0)} to {_fmt(max(cracked_values), 0)}",
            f"- c_max range: {_fmt(min(cmax_values))} to {_fmt(max(cmax_values))}",
            f"- fragment release verdicts: {len(rows) - fail_count} pass / {fail_count} fail",
        ])
        morphology_changed = (
            (max(cracked_values) - min(cracked_values) >= 25.0)
            or (visited_values and max(visited_values) - min(visited_values) >= 100.0)
            or (frag_values and max(frag_values) - min(frag_values) >= 3.0)
            or (junction_values and max(junction_values) - min(junction_values) >= 8.0)
        )
        if morphology_changed:
            lines.append("- result: sentence conditioning changes both material priors and crack morphology metrics.")
        else:
            lines.append("- result: material or morphology separation is weak; inspect top-k retrieval and style mapping.")
    else:
        lines.append("- result: prior-only run; morphology metrics were skipped.")

    def table(cols: list[str], active_rows: list[dict]) -> list[str]:
        out = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in active_rows:
            cells = []
            for col in cols:
                if col == "prompt":
                    cells.append(_short_prompt(row.get(col, "")))
                else:
                    cells.append(_fmt(row.get(col, "")))
            out.append("| " + " | ".join(cells) + " |")
        return out

    lines.extend(["", "## Material Priors", ""])
    lines.extend(table(prior_cols, rows))

    if not no_sim and "c_max" in rows[0]:
        lines.extend(["", "## Crack Morphology", ""])
        lines.extend(table(metric_cols, rows))

    prior_distances = _normalized_pairwise(rows, prior_keys)
    if prior_distances:
        lines.extend(["", "## Largest Prior Separations", ""])
        lines.extend(["| prompt A | prompt B | normalized distance |", "| --- | --- | --- |"])
        for item in prior_distances[:5]:
            lines.append(f"| {item['a']} | {item['b']} | {_fmt(item['distance'])} |")

    if not no_sim:
        metric_distances = _normalized_pairwise(rows, metric_keys)
        if metric_distances:
            lines.extend(["", "## Largest Morphology Separations", ""])
            lines.extend(["| prompt A | prompt B | normalized distance |", "| --- | --- | --- |"])
            for item in metric_distances[:5]:
                lines.append(f"| {item['a']} | {item['b']} | {_fmt(item['distance'])} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _runtime_fracture_params(config, material_prior, scaled):
    runtime_overrides = {
        "E": scaled["E"],
        "Gc": scaled["Gc"],
        "nu": scaled["nu"],
        "density": scaled["density"],
        **material_prior["runtime"],
    }
    cfg = apply_overrides_dict(config.copy(), runtime_overrides)
    cfg = resolve_material_preset(cfg)
    cfg = validate_l0(cfg)

    pf_params = OmegaConf.to_container(cfg.phase_field, resolve=True)
    manifold_cfg = OmegaConf.to_container(cfg.get("manifold", {}), resolve=True)
    return {
        **pf_params,
        **manifold_cfg,
        "Gc": float(cfg.material.Gc),
        "l0": float(cfg.material.l0),
    }


def _runtime_release_row(params: dict) -> dict:
    return {
        "runtime_crack_connected_release_only": bool(params.get("crack_connected_release_only", False)),
        "runtime_strict_closure_max_released_ratio": float(params.get("strict_closure_max_released_ratio", 0.0)),
        "runtime_open_crack_release_enable": bool(params.get("open_crack_release_enable", True)),
        "runtime_impact_release_gain": float(params.get("impact_release_gain", 1.0)),
    }


def _surface_points(config, particles: int):
    converter = MeshToPointCloudConverter(
        mesh_path=config.mesh.path,
        target_particle_count=int(particles),
        surface_sample_ratio=1.0,
        use_poisson=False,
        normalize_to_unit_cube=config.particles.get("normalize_to_unit_cube", True),
    )
    _, surface_pcd, _ = converter.convert()
    return (
        np.asarray(surface_pcd.points).copy(),
        np.asarray(surface_pcd.normals).copy(),
    )


def _bbox_metrics(positions: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict:
    if mask is None or not bool(mask.any()):
        return {
            f"{prefix}_count": 0,
            f"{prefix}_span_x": 0.0,
            f"{prefix}_span_y": 0.0,
            f"{prefix}_span_z": 0.0,
            f"{prefix}_radial_std": 0.0,
        }
    pts = positions[mask]
    span = pts.max(dim=0).values - pts.min(dim=0).values
    center = pts.mean(dim=0, keepdim=True)
    radial = torch.norm(pts - center, dim=1)
    return {
        f"{prefix}_count": int(mask.sum().item()),
        f"{prefix}_span_x": float(span[0].item()),
        f"{prefix}_span_y": float(span[1].item()),
        f"{prefix}_span_z": float(span[2].item()),
        f"{prefix}_radial_std": float(radial.std(unbiased=False).item()) if radial.numel() > 1 else 0.0,
    }


def _angular_metrics(
    positions: torch.Tensor,
    mask: torch.Tensor,
    center: torch.Tensor,
    prefix: str,
    bins: int = 24,
) -> dict:
    if mask is None or not bool(mask.any()):
        return {
            f"{prefix}_angular_coverage": 0.0,
            f"{prefix}_angular_entropy": 0.0,
        }
    rel = positions[mask, :2] - center[:2].unsqueeze(0)
    if rel.numel() == 0:
        return {
            f"{prefix}_angular_coverage": 0.0,
            f"{prefix}_angular_entropy": 0.0,
        }
    theta = torch.atan2(rel[:, 1], rel[:, 0])
    two_pi = float(2.0 * torch.pi)
    bin_pos = ((theta + torch.pi) / two_pi * int(bins)).floor().long()
    bin_pos = bin_pos.clamp(0, int(bins) - 1)
    hist = torch.bincount(bin_pos, minlength=int(bins)).float()
    occupied = hist > 0
    prob = hist / hist.sum().clamp(min=1e-8)
    prob = prob[prob > 0]
    entropy = -(prob * prob.log()).sum()
    entropy = entropy / torch.log(torch.tensor(float(bins), device=positions.device))
    return {
        f"{prefix}_angular_coverage": float(occupied.float().mean().item()),
        f"{prefix}_angular_entropy": float(entropy.item()),
    }


def _front_topology_metrics(crack_front, count: int, device: torch.device) -> dict:
    if (
        crack_front is None
        or getattr(crack_front, "visited_mask", None) is None
        or getattr(crack_front, "parent_index", None) is None
    ):
        return {
            "leaf_count": 0,
            "junction_count": 0,
            "max_children": 0,
            "mean_children": 0.0,
        }

    visited = crack_front.visited_mask[:count]
    parent = crack_front.parent_index[:count]
    valid_child = visited & (parent >= 0) & (parent < count)
    child_count = torch.zeros(count, dtype=torch.long, device=device)
    if bool(valid_child.any()):
        ones = torch.ones(int(valid_child.sum().item()), dtype=torch.long, device=device)
        child_count.scatter_add_(0, parent[valid_child], ones)

    if int(visited.sum().item()) <= 0:
        return {
            "leaf_count": 0,
            "junction_count": 0,
            "max_children": 0,
            "mean_children": 0.0,
        }

    leaf = visited & (child_count == 0)
    junction = visited & (child_count >= 2)
    return {
        "leaf_count": int(leaf.sum().item()),
        "junction_count": int(junction.sum().item()),
        "max_children": int(child_count[visited].max().item()),
        "mean_children": float(child_count[visited].float().mean().item()),
    }


def _branch_angle_metrics(crack_front, positions: torch.Tensor, count: int) -> dict:
    if (
        crack_front is None
        or getattr(crack_front, "visited_mask", None) is None
        or getattr(crack_front, "parent_index", None) is None
    ):
        return {
            "branch_angle_count": 0,
            "branch_angle_mean_deg": 0.0,
            "branch_angle_std_deg": 0.0,
        }

    visited = crack_front.visited_mask[:count]
    parent = crack_front.parent_index[:count]
    valid_child = visited & (parent >= 0) & (parent < count)
    if not bool(valid_child.any()):
        return {
            "branch_angle_count": 0,
            "branch_angle_mean_deg": 0.0,
            "branch_angle_std_deg": 0.0,
        }

    child_indices = torch.where(valid_child)[0]
    angles = []
    for p_t in torch.unique(parent[valid_child]).tolist():
        p = int(p_t)
        children = child_indices[parent[child_indices] == p]
        if children.numel() < 2:
            continue
        vec = positions[children] - positions[p].unsqueeze(0)
        vec = vec / vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos = (vec @ vec.T).clamp(-1.0, 1.0)
        upper = torch.triu(torch.ones_like(cos, dtype=torch.bool), diagonal=1)
        if bool(upper.any()):
            angles.append(torch.rad2deg(torch.acos(cos[upper])))

    if not angles:
        return {
            "branch_angle_count": 0,
            "branch_angle_mean_deg": 0.0,
            "branch_angle_std_deg": 0.0,
        }
    all_angles = torch.cat(angles)
    return {
        "branch_angle_count": int(all_angles.numel()),
        "branch_angle_mean_deg": float(all_angles.mean().item()),
        "branch_angle_std_deg": (
            float(all_angles.std(unbiased=False).item()) if all_angles.numel() > 1 else 0.0
        ),
    }


def _tip_event_metrics(events: list[dict]) -> dict:
    if not events:
        return {
            "tip_event_count": 0,
            "seed_event_count": 0,
            "advance_event_count": 0,
            "branch_event_count": 0,
            "tip_event_frame_span": 0,
            "branch_event_frame_count": 0,
            "branch_event_angle_mean_deg": 0.0,
            "branch_event_angle_std_deg": 0.0,
            "branch_event_closure_max": 0.0,
        }

    kinds = [str(event.get("kind", "")) for event in events]
    frames = [int(event.get("frame", 0)) for event in events]
    branch_angles = [
        float(event.get("branch_angle_deg", 0.0))
        for event in events
        if str(event.get("kind", "")) == "branch"
    ]
    branch_closure = [
        float(event.get("closure_score", 0.0))
        for event in events
        if str(event.get("kind", "")) == "branch"
    ]
    angle_np = np.asarray(branch_angles, dtype=np.float64)
    return {
        "tip_event_count": int(len(events)),
        "seed_event_count": int(sum(kind == "seed" for kind in kinds)),
        "advance_event_count": int(sum(kind == "advance" for kind in kinds)),
        "branch_event_count": int(sum(kind == "branch" for kind in kinds)),
        "tip_event_frame_span": int(max(frames) - min(frames)) if frames else 0,
        "branch_event_frame_count": int(len({
            frame for frame, kind in zip(frames, kinds) if kind == "branch"
        })),
        "branch_event_angle_mean_deg": float(angle_np.mean()) if angle_np.size else 0.0,
        "branch_event_angle_std_deg": float(angle_np.std()) if angle_np.size > 1 else 0.0,
        "branch_event_closure_max": float(max(branch_closure)) if branch_closure else 0.0,
    }


def _fragment_label_metrics(fragment_ids: torch.Tensor | None, count: int) -> dict:
    if fragment_ids is None:
        return {
            "final_fragment_label_count": 1,
            "final_nonbase_fragment_count": 0,
            "final_released_node_ratio": 0.0,
            "final_largest_fragment_ratio": 1.0,
            "final_fragment_entropy": 0.0,
            "final_nonbase_fragment_min_size": 0,
            "final_nonbase_fragment_mean_size": 0.0,
        }

    labels = fragment_ids[:count].detach().long()
    if labels.numel() == 0:
        return {
            "final_fragment_label_count": 1,
            "final_nonbase_fragment_count": 0,
            "final_released_node_ratio": 0.0,
            "final_largest_fragment_ratio": 1.0,
            "final_fragment_entropy": 0.0,
            "final_nonbase_fragment_min_size": 0,
            "final_nonbase_fragment_mean_size": 0.0,
        }

    unique, counts = labels.unique(sorted=True, return_counts=True)
    total = float(labels.numel())
    released = float((labels > 0).sum().item()) / max(total, 1.0)
    probs = counts.float() / max(total, 1.0)
    entropy = 0.0
    if probs.numel() > 1:
        entropy_t = -(probs * probs.clamp(min=1e-8).log()).sum()
        entropy = float((entropy_t / torch.log(torch.tensor(float(probs.numel()), device=labels.device))).item())
    nonbase_counts = counts[unique > 0]
    return {
        "final_fragment_label_count": int(unique.numel()),
        "final_nonbase_fragment_count": int((unique > 0).sum().item()),
        "final_released_node_ratio": released,
        "final_largest_fragment_ratio": float(counts.max().item() / max(total, 1.0)),
        "final_fragment_entropy": entropy,
        "final_nonbase_fragment_min_size": (
            int(nonbase_counts.min().item()) if nonbase_counts.numel() > 0 else 0
        ),
        "final_nonbase_fragment_mean_size": (
            float(nonbase_counts.float().mean().item()) if nonbase_counts.numel() > 0 else 0.0
        ),
    }


@torch.no_grad()
def simulate_prompt_metrics(
    positions_np: np.ndarray,
    normals_np: np.ndarray,
    fracture_params: dict,
    frames: int,
    device: torch.device,
    fragment_every: int,
    plot_path: Path | None = None,
    plot_title: str | None = None,
    event_log_path: Path | None = None,
) -> dict:
    positions = torch.from_numpy(positions_np).float().to(device)
    normals = torch.from_numpy(normals_np).float().to(device)
    normals = torch.nn.functional.normalize(normals, dim=-1)

    graph = GaussianGraph(
        k=fracture_params.get("graph_k", 12),
        sigma=fracture_params.get("graph_sigma", 0.03),
        rebuild_every=fracture_params.get("graph_rebuild_every", 5),
        device=str(device),
    )
    graph.set_normals(normals)

    fracture_field = make_surface_fracture_field(fracture_params, graph, device)
    fracture_field.initialize(positions.shape[0])

    family = fracture_params.get("material_family", "neutral_reference")
    style = fracture_params.get("sentence_style", fracture_params.get("crack_style", "material_default"))
    driver = SurfaceCrackDriver(material_family=family, crack_style=style)
    seed_center = driver.default_impact_center(positions)
    fracture_field.seed_damage(
        positions=positions,
        center=seed_center,
        radius=driver.params.impact_radius,
        magnitude=float(fracture_params.get("impact_seed_magnitude", 0.18)),
        H_multiplier=0.0,
    )

    fragment_manager = make_fragment_manager(fracture_params, device)
    max_cut_edges = 0
    max_closure = 0
    max_open_release_patches = 0
    max_open_release_nodes = 0
    max_n_frags = 1

    for frame in range(int(frames)):
        drive = driver.build(positions, frame, normals=normals)
        fracture_field.update(
            positions=positions,
            init_score=drive["init_score"],
            growth_drive=drive["growth_drive"],
            growth_dir=drive["growth_dir"],
            F_gaussian=None,
            impact_center=drive["impact_center"],
        )

        if fragment_manager is not None and (
            frame % max(int(fragment_every), 1) == 0 or frame == frames - 1
        ):
            crack_front = fracture_field.crack_front
            recent_front_mask = None
            if crack_front is not None and crack_front.visited_mask is not None:
                recent_front_mask = crack_front.visited_mask & (fracture_field.c > 0.12)
            n_frags = fragment_manager.detect_fragments(
                graph,
                fracture_field.c,
                positions=positions,
                opening=fracture_field.a,
                active_tip_mask=crack_front.tip_mask if crack_front is not None else None,
                recent_front_mask=recent_front_mask,
                crack_normal=fracture_field.n,
                crack_tangent=crack_front.growth_dir if crack_front is not None else None,
            )
            max_n_frags = max(max_n_frags, int(n_frags))
            max_cut_edges = max(max_cut_edges, int(fragment_manager.last_cut_edges))
            max_closure = max(max_closure, int(fragment_manager.last_closure_candidate_count))
            max_open_release_patches = max(
                max_open_release_patches,
                int(getattr(fragment_manager, "last_open_release_patches", 0)),
            )
            max_open_release_nodes = max(
                max_open_release_nodes,
                int(getattr(fragment_manager, "last_open_release_nodes", 0)),
            )

    c = fracture_field.c
    opening = fracture_field.a if fracture_field.a is not None else torch.zeros_like(c)
    crack_front = fracture_field.crack_front
    visited = (
        crack_front.visited_mask
        if crack_front is not None and crack_front.visited_mask is not None
        else torch.zeros_like(c, dtype=torch.bool)
    )
    tips = (
        crack_front.tip_mask
        if crack_front is not None and crack_front.tip_mask is not None
        else torch.zeros_like(c, dtype=torch.bool)
    )
    cracked = c > 0.30
    weak = c > 0.12

    metrics = {
        "c_max": float(c.max().item()),
        "c_mean": float(c.mean().item()),
        "opening_max": float(opening.max().item()),
        "opening_mean": float(opening.mean().item()),
        "visited_count": int(visited.sum().item()),
        "tip_count": int(tips.sum().item()),
        "cracked_count": int(cracked.sum().item()),
        "weak_count": int(weak.sum().item()),
        "max_n_frags": int(max_n_frags),
        "max_cut_edges": int(max_cut_edges),
        "max_closure_candidates": int(max_closure),
        "max_open_release_patches": int(max_open_release_patches),
        "max_open_release_nodes": int(max_open_release_nodes),
        "branchiness": float(tips.sum().item() / max(int(visited.sum().item()), 1)),
    }
    metrics.update(_front_topology_metrics(crack_front, int(c.shape[0]), positions.device))
    metrics.update(_branch_angle_metrics(crack_front, positions, int(c.shape[0])))
    tip_events = (
        crack_front.export_event_log()
        if crack_front is not None and hasattr(crack_front, "export_event_log")
        else []
    )
    metrics.update(_tip_event_metrics(tip_events))
    if event_log_path is not None:
        event_log_path = Path(event_log_path)
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log_path.write_text(json.dumps(tip_events, indent=2), encoding="utf-8")
    metrics.update(_bbox_metrics(positions, cracked, "cracked"))
    metrics.update(_bbox_metrics(positions, visited, "visited"))
    metrics.update(_angular_metrics(positions, cracked, seed_center, "cracked"))
    metrics.update(_angular_metrics(positions, visited, seed_center, "visited"))

    fragment_ids = (
        fragment_manager.fragment_ids
        if fragment_manager is not None and fragment_manager.fragment_ids is not None
        else None
    )
    metrics.update(_fragment_label_metrics(fragment_ids, int(c.shape[0])))

    if plot_path is not None:
        _save_final_crack_plot(
            positions=positions,
            damage=c,
            visited=visited,
            tips=tips,
            out_path=Path(plot_path),
            title=plot_title or "final crack morphology",
            fragment_ids=fragment_ids,
            parent_index=crack_front.parent_index if crack_front is not None else None,
            activation_step=crack_front.activation_step if crack_front is not None else None,
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch validate CLIP sentence material priors.")
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional UTF-8 text file with one prompt per line. Overrides --prompts.",
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--particles", type=int, default=8000)
    parser.add_argument("--frames", type=int, default=28)
    parser.add_argument("--fragment-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="output/sentence_material_validation")
    parser.add_argument("--no-sim", action="store_true", help="Only run CLIP/material-prior retrieval.")
    parser.add_argument(
        "--no-final-plots",
        dest="plot_final",
        action="store_false",
        help="Disable final-frame matplotlib morphology PNGs.",
    )
    parser.set_defaults(plot_final=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    prompts = list(args.prompts)
    if args.prompts_file:
        prompt_path = Path(args.prompts_file)
        prompts = [
            line.strip().lstrip("\ufeff")
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip().lstrip("\ufeff") and not line.lstrip("\ufeff").lstrip().startswith("#")
        ]
    if not prompts:
        raise ValueError("No prompts provided.")

    config = load_config(args.config)
    OmegaConf.update(config, "particles.target_count", int(args.particles))
    OmegaConf.update(config, "mpm.num_grids", 64)
    config = resolve_material_preset(config)
    config = validate_l0(config)

    predictor = MaterialPredictor(
        mode="clip_knn",
        db_path=args.db_path,
        clip_model=args.clip_model,
        device=str(device),
    )
    adapter = MaterialPriorAdapter()

    positions_np = normals_np = None
    if not args.no_sim:
        positions_np, normals_np = _surface_points(config, int(args.particles))

    rows = []
    t0 = time.time()
    for prompt in prompts:
        raw = predictor.predict(prompt)
        topk_entries = [predictor.db.get_entry_by_name(name) for name in raw["top_k_names"]]
        material_prior = adapter.build_material_prior(topk_entries, raw["top_k_scores"], text=prompt)
        material_prior = adapter.apply_sentence_style(material_prior, prompt)
        scaled = adapter.scale_physics_to_mpm(
            material_prior["physics"],
            family=material_prior["family"],
            base_density=float(config.material.density),
        )

        fracture_params = _runtime_fracture_params(config, material_prior, scaled)
        row = {
            "prompt": prompt,
            "family": material_prior["family"],
            "sentence_style": material_prior.get("sentence_style", "material_default"),
            "dominant_category": material_prior["dominant_category"],
            "top1": material_prior["top_k"][0]["name"] if material_prior["top_k"] else "",
            "top1_weight": material_prior["top_k"][0]["weight"] if material_prior["top_k"] else 0.0,
            "top_k": material_prior["top_k"],
            "family_scores": material_prior["family_scores"],
            "E_raw": material_prior["physics"]["E"],
            "Gc_raw": material_prior["physics"]["Gc"],
            "nu_raw": material_prior["physics"]["nu"],
            "density_raw": material_prior["physics"]["density"],
            "E_mpm": scaled["E"],
            "Gc_mpm": scaled["Gc"],
            "nu_mpm": scaled["nu"],
            "density_mpm": scaled["density"],
            "tau_init": material_prior["fracture"]["tau_init"],
            "growth_gain": material_prior["fracture"]["growth_gain"],
            "band_width": material_prior["fracture"]["band_width"],
            "open_gain": material_prior["fracture"]["open_gain"],
            "edge_break_rate": material_prior["fracture"]["edge_break_rate"],
            "branching_bias": material_prior["fracture"]["branching_bias"],
            "anisotropy_strength": material_prior["fracture"]["anisotropy_strength"],
        }
        row.update(_runtime_release_row(fracture_params))

        if not args.no_sim:
            print(f"\n[validate] {prompt!r} -> family={row['family']} top1={row['top1']}")
            plot_path = None
            if args.plot_final:
                plot_path = out_dir / "final_plots" / f"{len(rows):02d}_{_slug(prompt)}.png"
                row["final_plot"] = str(plot_path)
            event_log_path = out_dir / "event_logs" / f"{len(rows):02d}_{_slug(prompt)}.json"
            row["tip_event_log"] = str(event_log_path)
            metrics = simulate_prompt_metrics(
                positions_np,
                normals_np,
                fracture_params,
                frames=int(args.frames),
                device=device,
                fragment_every=int(args.fragment_every),
                plot_path=plot_path,
                plot_title=f"{prompt} | {row['family']} / {row['sentence_style']}",
                event_log_path=event_log_path,
            )
            row.update(metrics)

        row["expected_release_mode"] = _release_mode_for_row(row)
        if not args.no_sim:
            _, row["fragment_release_verdict"] = _release_verdict_for_row(row)
        rows.append(row)

    json_path = out_dir / "sentence_material_validation.json"
    csv_path = out_dir / "sentence_material_validation.csv"
    report_path = out_dir / "sentence_material_validation.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_for_csv(row))
    _write_markdown_report(rows, report_path, no_sim=bool(args.no_sim))

    print("\nSentence material validation complete")
    print(f"Output JSON: {json_path}")
    print(f"Output CSV:  {csv_path}")
    print(f"Report:      {report_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    for row in rows:
        sim_text = ""
        if "c_max" in row:
                sim_text = (
                    f" c_max={row['c_max']:.3f}"
                    f" cracked={row['cracked_count']}"
                    f" frags={row['max_n_frags']}"
                    f" branch_events={row.get('branch_event_count', 0)}"
                    f" cut={row['max_cut_edges']}"
                )
        print(
            f"- {row['prompt']} -> {row['family']} / {row['top1']} "
            f"E={row['E_raw']:.2e} Gc={row['Gc_raw']:.1f}{sim_text}"
        )


if __name__ == "__main__":
    main()
