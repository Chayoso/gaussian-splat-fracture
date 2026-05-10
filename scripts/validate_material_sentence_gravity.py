"""Validate CLIP material prompts with surface and gravity-drop metrics.

Outputs:
  - surface_material_*: CLIP prior + surface crack morphology by material prompt
  - surface_sentence_*: same material, different crack-shape wording
  - gravity_material_*: no-render gravity-drop fragment/crack stats
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_sentence_materials import (  # noqa: E402
    _angular_metrics,
    _branch_angle_metrics,
    _flatten_for_csv,
    _fragment_label_metrics,
    _front_topology_metrics,
    _release_mode_for_row,
    _release_verdict_for_row,
    _runtime_fracture_params,
    _runtime_release_row,
    _save_final_crack_plot,
    _slug,
    _surface_points,
    _tip_event_metrics,
    _write_markdown_report,
    simulate_prompt_metrics,
)
from src.config.loader import load_config  # noqa: E402
from src.core.material_presets import resolve_material_preset, validate_l0  # noqa: E402
from src.ml.material_predictor import MaterialPredictor  # noqa: E402
from src.ml.material_prior_adapter import MaterialPriorAdapter  # noqa: E402
from src.pipeline.manifold_fracture_pipeline import ManifoldFracturePipeline  # noqa: E402


DEFAULT_MATERIAL_PROMPTS = [
    "thin soda-lime glass bottle shattering into sharp clean cracks",
    "porcelain ceramic mug cracking into a few brittle splits",
    "rough concrete block crumbling into irregular chunks",
    "dry sandstone statue breaking into gritty rough pieces",
    "vulcanized rubber ball deforming without visible fracture",
    "hardwood oak block splitting along the grain",
    "structural steel block denting without brittle fracture",
    "clear ice sphere cracking with radial brittle fractures",
]

DEFAULT_SENTENCE_PROMPTS = [
    "glass bottle with one long smooth crack",
    "glass bottle with spiderweb branching cracks",
    "glass bottle shattering into many sharp radial cracks",
    "glass bottle with diffuse tiny surface scratches",
    "concrete block splitting with one clean fracture line",
    "concrete block crumbling into rough granular chunks",
    "concrete block with wide branching cracks",
    "concrete block with shallow diffuse microcracks",
]

DEFAULT_GRAVITY_PROMPTS = [
    "thin soda-lime glass bottle shattering into sharp clean cracks",
    "porcelain ceramic mug cracking into a few brittle splits",
    "rough concrete block crumbling into irregular chunks",
    "vulcanized rubber ball deforming without visible fracture",
]


def _read_prompts(path: str | None, defaults: list[str]) -> list[str]:
    if path is None:
        return list(defaults)
    prompt_path = Path(path)
    return [
        line.strip().lstrip("\ufeff")
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip().lstrip("\ufeff")
        and not line.lstrip("\ufeff").lstrip().startswith("#")
    ]


def _write_rows(rows: list[dict], prefix: str, out_dir: Path) -> tuple[Path, Path]:
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_for_csv(row))
    return json_path, csv_path


def _predict_row(
    prompt: str,
    predictor: MaterialPredictor,
    adapter: MaterialPriorAdapter,
    config,
) -> tuple[dict, dict, dict]:
    raw = predictor.predict(prompt)
    topk_entries = [predictor.db.get_entry_by_name(name) for name in raw["top_k_names"]]
    material_prior = adapter.build_material_prior(topk_entries, raw["top_k_scores"], text=prompt)
    material_prior = adapter.apply_sentence_style(material_prior, prompt)
    scaled = adapter.scale_physics_to_mpm(
        material_prior["physics"],
        family=material_prior["family"],
        base_density=float(config.material.density),
    )
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
    return row, material_prior, scaled


@torch.no_grad()
def run_surface_sweep(
    prompts: list[str],
    prefix: str,
    args,
    out_dir: Path,
    device: torch.device,
) -> list[dict]:
    config = load_config(args.config)
    OmegaConf.update(config, "particles.target_count", int(args.surface_particles))
    OmegaConf.update(config, "particles.surface_ratio", 1.0)
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
    positions_np, normals_np = _surface_points(config, int(args.surface_particles))

    rows = []
    for idx, prompt in enumerate(prompts):
        row, material_prior, scaled = _predict_row(prompt, predictor, adapter, config)
        fracture_params = _runtime_fracture_params(config, material_prior, scaled)
        row.update(_runtime_release_row(fracture_params))
        print(f"[surface:{prefix}] {prompt!r} -> {row['family']} / {row['top1']}", flush=True)
        plot_path = None
        if bool(getattr(args, "plot_final", True)):
            plot_path = out_dir / "final_plots" / prefix / f"{idx:02d}_{_slug(prompt)}.png"
            row["final_plot"] = str(plot_path)
        event_log_path = out_dir / "event_logs" / prefix / f"{idx:02d}_{_slug(prompt)}.json"
        row["tip_event_log"] = str(event_log_path)
        metrics = simulate_prompt_metrics(
            positions_np,
            normals_np,
            fracture_params,
            frames=int(args.surface_frames),
            device=device,
            fragment_every=int(args.fragment_every),
            plot_path=plot_path,
            plot_title=f"{prefix}: {prompt} | {row['family']} / {row['sentence_style']}",
            event_log_path=event_log_path,
        )
        row.update(metrics)
        row["expected_release_mode"], row["fragment_release_verdict"] = _release_verdict_for_row(row)
        rows.append(row)

    _write_rows(rows, prefix, out_dir)
    report_path = out_dir / f"{prefix}.md"
    _write_markdown_report(rows, report_path, no_sim=False)
    return rows


def _save_gravity_config(args, out_dir: Path) -> Path:
    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "particles.target_count", int(args.gravity_particles), merge=False)
    OmegaConf.update(cfg, "particles.surface_ratio", 1.0, merge=False)
    OmegaConf.update(cfg, "mpm.num_grids", int(args.gravity_grids), merge=False)
    OmegaConf.update(cfg, "rendering.total_frames", int(args.gravity_frames), merge=False)
    OmegaConf.update(cfg, "rendering.physics_substeps", int(args.physics_substeps), merge=False)
    OmegaConf.update(cfg, "rendering.render_frames", [], merge=False)
    OmegaConf.update(cfg, "rendering.image_width", 256, merge=False)
    OmegaConf.update(cfg, "rendering.image_height", 256, merge=False)
    OmegaConf.update(cfg, "simulation.save_frames", False, merge=False)
    OmegaConf.update(cfg, "simulation.save_checkpoint", False, merge=False)
    OmegaConf.update(cfg, "output.frame_dir", str(out_dir / "frames_unused"), merge=False)
    OmegaConf.update(cfg, "output.checkpoint_dir", str(out_dir / "checkpoints_unused"), merge=False)
    OmegaConf.update(cfg, "output.make_video", False, merge=False)
    OmegaConf.update(cfg, "loading.drop_center_z", float(args.drop_center_z), merge=False)
    OmegaConf.update(cfg, "loading.fixed_gravity_z", float(args.gravity_z), merge=False)
    path = out_dir / "gravity_validation_config.yaml"
    OmegaConf.save(cfg, path)
    return path


def _summarize_history(history: list[dict]) -> dict:
    if not history:
        return {
            "impact_frame": None,
            "final_frame": None,
            "max_n_fragments": 0,
            "final_n_fragments": 0,
        }
    impact_frames = [
        int(row.get("loop_frame", row.get("frame", 0)))
        for row in history
        if bool(row.get("gravity_contacted", False))
    ]
    max_row = lambda key, default=0.0: max(float(row.get(key, default)) for row in history)
    final = history[-1]
    return {
        "impact_frame": min(impact_frames) if impact_frames else None,
        "final_frame": int(final.get("loop_frame", final.get("frame", 0))),
        "max_n_fragments": int(max(int(row.get("n_fragments", 0)) for row in history)),
        "final_n_fragments": int(final.get("n_fragments", 0)),
        "max_n_cracked": int(max(int(row.get("n_cracked", 0)) for row in history)),
        "final_n_cracked": int(final.get("n_cracked", 0)),
        "max_c_max": max_row("c_max"),
        "final_c_max": float(final.get("c_max", 0.0)),
        "max_cut_edges": int(max(int(row.get("cut_edges", 0)) for row in history)),
        "max_broken_edges": int(max(int(row.get("broken_edges", 0)) for row in history)),
        "max_release_candidates": int(max(int(row.get("release_candidate_count", 0)) for row in history)),
        "max_hard_detached_nodes": int(max(int(row.get("hard_detached_nodes", 0)) for row in history)),
        "final_hard_detached_nodes": int(final.get("hard_detached_nodes", 0)),
        "max_detached_boundary_edges": int(max(int(row.get("detached_boundary_edges", 0)) for row in history)),
        "final_detached_boundary_edges": int(final.get("detached_boundary_edges", 0)),
        "max_open_release_patches": int(max(int(row.get("open_release_patches", 0)) for row in history)),
        "max_open_release_nodes": int(max(int(row.get("open_release_nodes", 0)) for row in history)),
        "max_open_release_score": max_row("open_release_score_max"),
        "max_catastrophic_release_patches": int(max(int(row.get("catastrophic_release_patches", 0)) for row in history)),
        "max_catastrophic_release_nodes": int(max(int(row.get("catastrophic_release_nodes", 0)) for row in history)),
        "max_catastrophic_release_score": max_row("catastrophic_release_score_max"),
        "max_secondary_shatter_patches": int(max(int(row.get("secondary_shatter_patches", 0)) for row in history)),
        "max_secondary_shatter_nodes": int(max(int(row.get("secondary_shatter_nodes", 0)) for row in history)),
        "max_secondary_shatter_score": max_row("secondary_shatter_score_max"),
        "max_impact_shatter_patches": int(max(int(row.get("impact_shatter_patches", 0)) for row in history)),
        "max_impact_shatter_nodes": int(max(int(row.get("impact_shatter_nodes", 0)) for row in history)),
        "max_impact_shatter_score": max_row("impact_shatter_score_max"),
        "max_physical_fragment_drop": max_row("physical_fragment_drop"),
        "max_physical_detached_distance": max_row("physical_detached_distance"),
        "max_physical_release_displacement": max_row("physical_release_displacement"),
        "max_physical_lateral_release_displacement": max_row("physical_lateral_release_displacement"),
        "max_physical_fragment_lateral_spread": max_row("physical_fragment_lateral_spread"),
        "max_volumetric_damage": max_row("volumetric_damage_max"),
        "final_volumetric_damage": float(final.get("volumetric_damage_max", 0.0)),
        "max_detached_distance": max_row("detached_distance"),
        "final_z_min": float(final.get("z_min", 0.0)),
        "final_z_com": float(final.get("z_com", 0.0)),
        "final_v_com_z": float(final.get("v_com_z", 0.0)),
        "impact_release_gain": max_row("impact_release_gain", 1.0),
    }


def _fragment_boundary_support_metrics(simulator, fragment_ids: torch.Tensor | None, n: int) -> dict:
    """Measure whether final fragment label boundaries have crack/cut support."""
    if simulator is None or fragment_ids is None:
        return {
            "final_fragment_boundary_edges": 0,
            "final_causal_supported_fragment_boundary_edges": 0,
            "final_fragment_boundary_support_ratio": 0.0,
            "final_detached_boundary_edges": 0,
        }
    graph = getattr(simulator, "graph", None)
    knn_idx = getattr(graph, "knn_idx", None) if graph is not None else None
    if knn_idx is None:
        return {
            "final_fragment_boundary_edges": 0,
            "final_causal_supported_fragment_boundary_edges": 0,
            "final_fragment_boundary_support_ratio": 0.0,
            "final_detached_boundary_edges": 0,
        }

    n = min(int(n), int(fragment_ids.shape[0]), int(knn_idx.shape[0]))
    frag = fragment_ids[:n].detach().long()
    knn = knn_idx[:n].detach().long()
    valid = (knn >= 0) & (knn < n)
    frag_i = frag.unsqueeze(1).expand_as(knn)
    frag_j = torch.zeros_like(knn)
    frag_j[valid] = frag[knn[valid]]
    boundary = valid & (frag_i != frag_j) & ((frag_i > 0) | (frag_j > 0))

    # Graph-derived cut/closure mask augmentation removed with
    # GraphFragmentManager; causal support now reduces to fragment-
    # boundary edges only.
    supported = boundary
    boundary_edges = int(boundary.sum().item())
    supported_edges = int(supported.sum().item())
    return {
        "final_fragment_boundary_edges": boundary_edges,
        "final_causal_supported_fragment_boundary_edges": supported_edges,
        "final_fragment_boundary_support_ratio": supported_edges / max(boundary_edges, 1),
        "final_detached_boundary_edges": 0,
    }


def _bbox_metrics_tensor(positions: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict:
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
        f"{prefix}_radial_std": (
            float(radial.std(unbiased=False).item()) if radial.numel() > 1 else 0.0
        ),
    }


@torch.no_grad()
def _final_simulator_crack_metrics(
    simulator,
    plot_path: Path | None = None,
    plot_title: str | None = None,
    event_log_path: Path | None = None,
) -> dict:
    """Read final crack-front metrics directly from the no-render simulator."""
    ff = getattr(simulator, "fracture_field", None)
    if ff is None or getattr(ff, "c", None) is None:
        return {}

    c = ff.c.detach()
    opening = ff.a.detach() if getattr(ff, "a", None) is not None else torch.zeros_like(c)
    front = getattr(ff, "crack_front", None)
    visited = (
        front.visited_mask.detach()
        if front is not None and getattr(front, "visited_mask", None) is not None
        else torch.zeros_like(c, dtype=torch.bool)
    )
    tips = (
        front.tip_mask.detach()
        if front is not None and getattr(front, "tip_mask", None) is not None
        else torch.zeros_like(c, dtype=torch.bool)
    )

    positions = None
    if getattr(simulator, "x_mpm", None) is not None and getattr(simulator, "surface_mask", None) is not None:
        positions = simulator.mapper.mpm_to_world(simulator.x_mpm[simulator.surface_mask]).detach()
    if positions is None:
        positions = simulator.gaussians._xyz.detach()

    n = min(int(c.shape[0]), int(positions.shape[0]))
    c = c[:n]
    opening = opening[:n]
    visited = visited[:n]
    tips = tips[:n]
    positions = positions[:n]
    cracked = c > 0.30
    weak = c > 0.12

    metrics = {
        "final_c_mean": float(c.mean().item()),
        "final_opening_max": float(opening.max().item()),
        "final_opening_mean": float(opening.mean().item()),
        "final_visited_count": int(visited.sum().item()),
        "final_tip_count": int(tips.sum().item()),
        "final_weak_count": int(weak.sum().item()),
        "final_branchiness": float(tips.sum().item() / max(int(visited.sum().item()), 1)),
    }
    metrics.update({
        f"final_{key}": value
        for key, value in _front_topology_metrics(front, int(c.shape[0]), positions.device).items()
    })
    metrics.update({
        f"final_{key}": value
        for key, value in _branch_angle_metrics(front, positions, int(c.shape[0])).items()
    })
    tip_events = (
        front.export_event_log()
        if front is not None and hasattr(front, "export_event_log")
        else []
    )
    event_metrics = _tip_event_metrics(tip_events)
    metrics.update({
        f"final_{key}": value
        for key, value in event_metrics.items()
    })
    if event_log_path is not None:
        event_log_path = Path(event_log_path)
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log_path.write_text(json.dumps(tip_events, indent=2), encoding="utf-8")
    metrics.update(_bbox_metrics_tensor(positions, cracked, "final_cracked"))
    metrics.update(_bbox_metrics_tensor(positions, visited, "final_visited"))
    center = positions.mean(dim=0)
    impact_center = getattr(simulator, "_impact_center", None)
    if impact_center is not None:
        try:
            center = simulator.mapper.mpm_to_world(impact_center.unsqueeze(0)).squeeze(0).detach()
        except Exception:
            center = positions.mean(dim=0)
    metrics.update(_angular_metrics(positions, cracked, center, "final_cracked"))
    metrics.update(_angular_metrics(positions, visited, center, "final_visited"))

    # Source of truth: physical voronoi fragment labels.
    fragment_ids = None
    physical_labels = getattr(simulator, "_physical_fragment_labels", None)
    surface_indices = getattr(simulator, "_surface_indices", None)
    if physical_labels is not None and surface_indices is not None:
        surf_phys = physical_labels[surface_indices][:n].detach()
        if bool((surf_phys > 0).any()):
            fragment_ids = surf_phys
            metrics.update({
                f"physical_{key}": value
                for key, value in _fragment_label_metrics(surf_phys, int(surf_phys.shape[0])).items()
            })
    metrics.update(_fragment_label_metrics(fragment_ids, int(c.shape[0])))
    metrics.update(_fragment_boundary_support_metrics(simulator, fragment_ids, int(c.shape[0])))

    if plot_path is not None:
        _save_final_crack_plot(
            positions=positions,
            damage=c,
            visited=visited,
            tips=tips,
            out_path=Path(plot_path),
            title=plot_title or "gravity final crack morphology",
            fragment_ids=fragment_ids,
            parent_index=front.parent_index if front is not None else None,
            activation_step=front.activation_step if front is not None else None,
        )
    return metrics


def _write_gravity_report(rows: list[dict], out_dir: Path) -> None:
    lines = [
        "# Gravity Material Validation",
        "",
        "No-render gravity-drop run. CLIP predicts material priors, then the object falls under gravity and reports crack/fragment metrics.",
        "",
        "| prompt | family | style | mode | verdict | strict | top1 | impact | frags max/final | rel | largest | bcut | hard det | saved boundary | cracked max/final | visited | tips | tip events | branch events | junction | max child | angle mean/std | branch | c_max | cut_edges | open p/n | cat p/n | sec p/n | gain | scatter |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("error"):
            lines.append(
                "| {prompt} | ERROR |  |  | FAIL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |".format(
                    prompt=str(row.get("prompt", ""))[:46],
                )
            )
            continue
        lines.append(
            "| {prompt} | {family} | {style} | {mode} | {verdict} | {strict} | {top1} | {impact} | {maxf}/{finalf} | "
            "{rel:.3f} | {largest:.3f} | {bcut:.3f} | {hard_det} | {saved_boundary} | {maxc}/{finalc} | {visited} | {tips} | {events} | {branches} | {junction} | {maxchildren} | "
            "{angle_mean:.1f}/{angle_std:.1f} | {branch:.3f} | {cmax:.3f} | {cut} | {openp}/{openn} | "
            "{catp}/{catn} | {secp}/{secn} | {gain:.2f} | {scatter:.4f} |".format(
                prompt=row["prompt"][:46],
                family=row["family"],
                style=row.get("sentence_style", "material_default"),
                mode=row.get("expected_release_mode", _release_mode_for_row(row)),
                verdict=row.get("fragment_release_verdict", ""),
                strict="Y" if bool(row.get("runtime_crack_connected_release_only", False)) else "",
                top1=row["top1"],
                impact="" if row.get("impact_frame") is None else row["impact_frame"],
                maxf=row.get("max_n_fragments", 0),
                finalf=row.get("final_n_fragments", 0),
                rel=float(row.get("final_released_node_ratio", 0.0)),
                largest=float(row.get("final_largest_fragment_ratio", 1.0)),
                bcut=float(row.get("final_fragment_boundary_support_ratio", 0.0)),
                hard_det=row.get("final_hard_detached_nodes", 0),
                saved_boundary=row.get("final_detached_boundary_edges", 0),
                maxc=row.get("max_n_cracked", 0),
                finalc=row.get("final_n_cracked", 0),
                visited=row.get("final_visited_count", 0),
                tips=row.get("final_tip_count", 0),
                events=row.get("final_tip_event_count", 0),
                branches=row.get("final_branch_event_count", 0),
                junction=row.get("final_junction_count", 0),
                maxchildren=row.get("final_max_children", 0),
                angle_mean=float(row.get("final_branch_angle_mean_deg", 0.0)),
                angle_std=float(row.get("final_branch_angle_std_deg", 0.0)),
                branch=float(row.get("final_branchiness", 0.0)),
                cmax=float(row.get("max_c_max", 0.0)),
                cut=row.get("max_cut_edges", 0),
                openp=row.get("max_open_release_patches", 0),
                openn=row.get("max_open_release_nodes", 0),
                catp=row.get("max_catastrophic_release_patches", 0),
                catn=row.get("max_catastrophic_release_nodes", 0),
                secp=row.get("max_secondary_shatter_patches", 0),
                secn=row.get("max_secondary_shatter_nodes", 0),
                gain=float(row.get("impact_release_gain", 1.0)),
                scatter=float(row.get("max_physical_release_displacement", 0.0)),
            )
        )
    lines.append("")
    lines.append("Interpretation: CLIP chooses material/style parameters. Fragment birth should stay closure-only: open/cat/sec must remain 0, and `bcut` should be nonzero when fragments appear. `scatter` is post-fragment physical release displacement.")
    (out_dir / "gravity_material_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gravity_sweep(
    prompts: list[str],
    args,
    out_dir: Path,
) -> list[dict]:
    config_path = _save_gravity_config(args, out_dir)
    pipeline = ManifoldFracturePipeline(
        str(config_path),
        material_source="clip",
        fast_mode=False,
        db_path=args.db_path,
        clip_model=args.clip_model,
    )
    rows = []
    for idx, prompt in enumerate(prompts):
        t0 = time.time()
        print(f"[gravity] {prompt!r}", flush=True)
        try:
            result = pipeline.run(
                prompt,
                num_frames=int(args.gravity_frames),
                save_frames=False,
                return_frames=False,
                override_params={
                    "rendering.render_frames": [],
                    "output.make_video": False,
                    "simulation.save_checkpoint": False,
                    "manifold.fragment_detect_every": max(int(args.fragment_every), 1),
                },
            )
        except Exception as exc:
            rows.append({
                "prompt": prompt,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": time.time() - t0,
            })
            _write_rows(rows, "gravity_material_validation", out_dir)
            _write_gravity_report(rows, out_dir)
            print(f"[gravity:error] {prompt!r}: {exc}", flush=True)
            continue
        history = list(getattr(pipeline.engine, "last_stats_history", []))
        prior = result["material_prior"]
        row = {
            "prompt": prompt,
            "family": result["fracture_family"],
            "sentence_style": prior.get("sentence_style", "material_default"),
            "dominant_category": result["material_category"],
            "top1": prior["top_k"][0]["name"] if prior["top_k"] else "",
            "top_k": prior["top_k"],
            "E_raw": prior["physics"]["E"],
            "Gc_raw": prior["physics"]["Gc"],
            "nu_raw": prior["physics"]["nu"],
            "density_raw": prior["physics"]["density"],
            "E_mpm": result["params"]["E"],
            "Gc_mpm": result["params"]["Gc"],
            "nu_mpm": result["params"]["nu"],
            "density_mpm": result["params"]["density"],
            "elapsed_sec": time.time() - t0,
        }
        row.update(_runtime_release_row(result["params"]))
        row.update(_summarize_history(history))
        simulator = getattr(pipeline.engine, "last_simulator", None)
        if simulator is not None:
            row["runtime_crack_connected_release_only"] = bool(getattr(simulator, "crack_connected_release_only", False))
        plot_path = None
        if bool(getattr(args, "plot_final", True)):
            plot_path = out_dir / "final_plots" / "gravity_material_validation" / f"{idx:02d}_{_slug(prompt)}.png"
            row["final_plot"] = str(plot_path)
        event_log_path = out_dir / "event_logs" / "gravity_material_validation" / f"{idx:02d}_{_slug(prompt)}.json"
        row["tip_event_log"] = str(event_log_path)
        row.update(_final_simulator_crack_metrics(
            simulator,
            plot_path=plot_path,
            plot_title=f"gravity: {prompt} | {row['family']} / {row['sentence_style']}",
            event_log_path=event_log_path,
        ))
        row["expected_release_mode"], row["fragment_release_verdict"] = _release_verdict_for_row(row)
        rows.append(row)
        _write_rows(rows, "gravity_material_validation", out_dir)
        _write_gravity_report(rows, out_dir)
        torch.cuda.empty_cache()

    _write_rows(rows, "gravity_material_validation", out_dir)
    _write_gravity_report(rows, out_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CLIP material and sentence fracture behavior.")
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--out", default="output/material_sentence_validation_20260425")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--material-prompts-file", default=None)
    parser.add_argument("--sentence-prompts-file", default=None)
    parser.add_argument("--gravity-prompts-file", default=None)
    parser.add_argument("--surface-particles", type=int, default=10000)
    parser.add_argument("--surface-frames", type=int, default=24)
    parser.add_argument("--fragment-every", type=int, default=4)
    parser.add_argument("--gravity-particles", type=int, default=8000)
    parser.add_argument("--gravity-frames", type=int, default=64)
    parser.add_argument("--gravity-grids", type=int, default=64)
    parser.add_argument("--physics-substeps", type=int, default=3)
    parser.add_argument("--drop-center-z", type=float, default=0.42)
    parser.add_argument("--gravity-z", type=float, default=-3500.0)
    parser.add_argument("--skip-surface", action="store_true")
    parser.add_argument("--skip-gravity", action="store_true")
    parser.add_argument(
        "--no-final-plots",
        dest="plot_final",
        action="store_false",
        help="Disable final-frame matplotlib morphology PNGs.",
    )
    parser.set_defaults(plot_final=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    material_prompts = _read_prompts(args.material_prompts_file, DEFAULT_MATERIAL_PROMPTS)
    sentence_prompts = _read_prompts(args.sentence_prompts_file, DEFAULT_SENTENCE_PROMPTS)
    gravity_prompts = _read_prompts(args.gravity_prompts_file, DEFAULT_GRAVITY_PROMPTS)

    (out_dir / "material_prompts.txt").write_text("\n".join(material_prompts) + "\n", encoding="utf-8")
    (out_dir / "sentence_prompts.txt").write_text("\n".join(sentence_prompts) + "\n", encoding="utf-8")
    (out_dir / "gravity_prompts.txt").write_text("\n".join(gravity_prompts) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    if not args.skip_surface:
        run_surface_sweep(material_prompts, "surface_material_validation", args, out_dir, device)
        run_surface_sweep(sentence_prompts, "surface_sentence_validation", args, out_dir, device)
    if not args.skip_gravity:
        run_gravity_sweep(gravity_prompts, args, out_dir)

    manifest = {
        "out_dir": str(out_dir),
        "surface_particles": int(args.surface_particles),
        "surface_frames": int(args.surface_frames),
        "gravity_particles": int(args.gravity_particles),
        "gravity_frames": int(args.gravity_frames),
        "physics_substeps": int(args.physics_substeps),
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nValidation complete: {out_dir}")
    print(f"Elapsed: {manifest['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
