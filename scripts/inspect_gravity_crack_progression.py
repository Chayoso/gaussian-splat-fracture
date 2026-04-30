"""Inspect crack and fragment progression for one gravity-drop prompt.

This saves matplotlib crack/fragment snapshots at fixed frame intervals after
impact, plus JSON/CSV/Markdown timeline metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_material_sentence_gravity import _save_gravity_config, _summarize_history  # noqa: E402
from scripts.validate_sentence_materials import (  # noqa: E402
    _flatten_for_csv,
    _fragment_label_metrics,
    _release_verdict_for_row,
    _runtime_release_row,
    _slug,
    _tip_event_metrics,
)
from src.pipeline.manifold_fracture_pipeline import ManifoldFracturePipeline  # noqa: E402
from src.diagnostics.raw_graph_plot import save_raw_graph_diagnostic as _save_raw_graph_diagnostic  # noqa: E402
from src.diagnostics.physical_fragment_plot import save_physical_diagnostic as _save_physical_diagnostic  # noqa: E402
from src.diagnostics.houdini_export import export_simulator_state as _export_houdini_geo  # noqa: E402
from src.utils.knn import knn_search  # noqa: E402


DEFAULT_PROMPT = "thin glass bottle shattering into localized connected radial cracks"


def _euler_to_rotation_matrix_np(rot_deg: list[float]) -> np.ndarray:
    rot_rad = [np.radians(float(r)) for r in rot_deg]
    cx, sx = np.cos(rot_rad[0]), np.sin(rot_rad[0])
    cy, sy = np.cos(rot_rad[1]), np.sin(rot_rad[1])
    cz, sz = np.cos(rot_rad[2]), np.sin(rot_rad[2])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _apply_gravity_reference_transform(
    vertices: np.ndarray,
    surface_points: np.ndarray,
    config,
) -> tuple[np.ndarray, np.ndarray]:
    """Match src.engine.loading_transforms for diagnostic mesh references."""
    loading = getattr(config, "loading", {})
    if loading.get("type", None) != "gravity_drop":
        return vertices.astype(np.float32), surface_points.astype(np.float32)

    center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    drop_scale = float(loading.get("drop_scale", 0.33))
    verts = center + (vertices.astype(np.float32) - center) * drop_scale
    surf = center + (surface_points.astype(np.float32) - center) * drop_scale

    drop_center_z = float(loading.get("drop_center_z", 0.7))
    z_current_center = 0.5 * (float(surf[:, 2].min()) + float(surf[:, 2].max()))
    z_shift = drop_center_z - z_current_center
    verts[:, 2] += z_shift
    surf[:, 2] += z_shift

    drop_rotation = loading.get("drop_rotation", None)
    if drop_rotation is not None:
        rot = _euler_to_rotation_matrix_np([float(v) for v in drop_rotation])
        obj_center = surf.mean(axis=0)
        verts = obj_center + (verts - obj_center) @ rot.T
        surf = obj_center + (surf - obj_center) @ rot.T
    return verts.astype(np.float32), surf.astype(np.float32)


def _load_normalized_mesh(config) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh_path = Path(str(config.mesh.path))
    if not mesh_path.is_absolute():
        mesh_path = PROJECT_ROOT / mesh_path
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if bool(config.particles.get("normalize_to_unit_cube", True)):
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        center = (bbox_min + bbox_max) * 0.5
        scale = float((bbox_max - bbox_min).max())
        vertices = (vertices - center) / scale * 0.95 + 0.5
    return vertices.astype(np.float32), faces.astype(np.int64)


def _nearest_indices(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    _, idx = knn_search(
        torch.from_numpy(query.astype(np.float32)),
        torch.from_numpy(database.astype(np.float32)),
        1,
    )
    return idx[:, 0].cpu().numpy().astype(np.int64)


def _knn_indices(query: np.ndarray, database: np.ndarray, k: int) -> np.ndarray:
    _, idx = knn_search(
        torch.from_numpy(query.astype(np.float32)),
        torch.from_numpy(database.astype(np.float32)),
        int(k),
    )
    return idx.cpu().numpy().astype(np.int64)


def _build_mesh_visual_state(config, surface_points: np.ndarray) -> dict:
    vertices, faces = _load_normalized_mesh(config)
    vertices_ref, surface_ref = _apply_gravity_reference_transform(
        vertices,
        surface_points,
        config,
    )
    vertex_anchor = _nearest_indices(vertices_ref, surface_ref)
    face_centroids = vertices_ref[faces].mean(axis=1)
    face_anchor = _nearest_indices(face_centroids, surface_ref)
    face_anchors = _knn_indices(face_centroids, surface_ref, 7)

    face_ids = np.concatenate(
        [
            np.arange(faces.shape[0], dtype=np.int64),
            np.arange(faces.shape[0], dtype=np.int64),
            np.arange(faces.shape[0], dtype=np.int64),
        ]
    )
    raw_edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    raw_edges = np.sort(raw_edges, axis=1)
    unique_edges, inverse = np.unique(raw_edges, axis=0, return_inverse=True)
    edge_faces = np.full((unique_edges.shape[0], 2), -1, dtype=np.int64)
    for edge_row, face_id in zip(inverse, face_ids):
        if edge_faces[edge_row, 0] < 0:
            edge_faces[edge_row, 0] = face_id
        elif edge_faces[edge_row, 1] < 0 and edge_faces[edge_row, 0] != face_id:
            edge_faces[edge_row, 1] = face_id

    return {
        "vertices_ref": vertices_ref,
        "surface_ref": surface_ref,
        "faces": faces,
        "vertex_anchor": vertex_anchor,
        "face_anchor": face_anchor,
        "face_anchors": face_anchors,
        "edges": unique_edges,
        "edge_faces": edge_faces,
    }


def _read_prompts(path: str | None, default_prompt: str) -> list[str]:
    if path is None:
        return [default_prompt]
    prompt_path = Path(path)
    return [
        line.strip().lstrip("\ufeff")
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip().lstrip("\ufeff")
        and not line.lstrip("\ufeff").lstrip().startswith("#")
    ]


def _read_override_json(value: str | None) -> dict:
    if value is None:
        return {}
    raw = value.strip()
    if not raw:
        return {}
    candidate = Path(raw)
    if candidate.exists():
        raw = candidate.read_text(encoding="utf-8")
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("--override-json must decode to an object")
    return loaded


def _surface_state(
    simulator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object | None]:
    ff = getattr(simulator, "fracture_field", None)
    if ff is None or getattr(ff, "c", None) is None:
        raise RuntimeError("simulator has no fracture field state")

    c = ff.c.detach()
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

    if getattr(simulator, "x_mpm", None) is not None and getattr(simulator, "surface_mask", None) is not None:
        positions = simulator.mapper.mpm_to_world(simulator.x_mpm[simulator.surface_mask]).detach()
    else:
        positions = simulator.gaussians._xyz.detach()

    n = min(int(c.shape[0]), int(positions.shape[0]))
    return positions[:n], c[:n], visited[:n], tips[:n], front


def _fragment_ids(simulator, n: int) -> torch.Tensor | None:
    manager = getattr(simulator, "fragment_manager", None)
    if manager is None or getattr(manager, "fragment_ids", None) is None:
        return None
    return manager.fragment_ids[:n].detach()


def _edge_boundary_metrics(
    *,
    fragment_ids: torch.Tensor | None,
    knn_idx: torch.Tensor | None,
    cut_edge_mask: torch.Tensor | None,
    closure_boundary_mask: torch.Tensor | None,
    detached_boundary_mask: torch.Tensor | None,
    n: int,
) -> dict:
    if fragment_ids is None or knn_idx is None:
        return {
            "fragment_boundary_edges": 0,
            "cut_supported_fragment_boundary_edges": 0,
            "causal_supported_fragment_boundary_edges": 0,
            "closure_boundary_edges": 0,
            "detached_boundary_edges": 0,
            "fragment_boundary_cut_support_ratio": 0.0,
        }
    frag = fragment_ids[:n].detach().long()
    knn = knn_idx[:n].detach().long()
    valid = (knn >= 0) & (knn < n)
    frag_i = frag.unsqueeze(1).expand_as(knn)
    frag_j = torch.zeros_like(knn)
    frag_j[valid] = frag[knn[valid]]
    boundary = valid & (frag_i != frag_j) & ((frag_i > 0) | (frag_j > 0))
    cut_supported = torch.zeros_like(boundary)
    if cut_edge_mask is not None and cut_edge_mask.shape == knn.shape:
        cut_supported = boundary & cut_edge_mask.detach().bool()[:n]
    closure_edges = torch.zeros_like(boundary)
    if closure_boundary_mask is not None and closure_boundary_mask.shape == knn.shape:
        closure_edges = closure_boundary_mask.detach().bool()[:n] & valid
    detached_edges = torch.zeros_like(boundary)
    if detached_boundary_mask is not None and detached_boundary_mask.shape == knn.shape:
        detached_edges = detached_boundary_mask.detach().bool()[:n] & valid
    causal_supported = boundary & (cut_supported | closure_edges | detached_edges)

    boundary_count = int(boundary.sum().item())
    causal_count = int(causal_supported.sum().item())
    return {
        "fragment_boundary_edges": boundary_count,
        "cut_supported_fragment_boundary_edges": int(cut_supported.sum().item()),
        "causal_supported_fragment_boundary_edges": causal_count,
        "closure_boundary_edges": int(closure_edges.sum().item()),
        "detached_boundary_edges": int(detached_edges.sum().item()),
        "fragment_boundary_cut_support_ratio": (
            causal_count / max(boundary_count, 1)
        ),
    }


def _snapshot_metrics(
    *,
    frame: int,
    impact_frame: int,
    simulator,
    stats: dict,
    prompt: str,
    out_dir: Path,
    previous_event_count: int,
    mesh_visual_state: dict | None = None,
    save_plot: bool = True,
) -> tuple[dict, int]:
    positions, c, visited, tips, front = _surface_state(simulator)
    fragment_ids = _fragment_ids(simulator, int(c.shape[0]))
    events = (
        front.export_event_log()
        if front is not None and hasattr(front, "export_event_log")
        else []
    )
    new_events = events[previous_event_count:]
    event_metrics = _tip_event_metrics(events)
    new_event_metrics = _tip_event_metrics(new_events)
    fragment_metrics = _fragment_label_metrics(fragment_ids, int(c.shape[0]))
    manager = getattr(simulator, "fragment_manager", None)
    graph = getattr(simulator, "graph", None)
    knn_idx = getattr(graph, "knn_idx", None) if graph is not None else None
    cut_edge_mask = (
        getattr(manager, "last_cut_edge_mask", None)
        if manager is not None else None
    )
    closure_boundary_mask = (
        getattr(manager, "last_closure_boundary_mask", None)
        if manager is not None else None
    )
    detached_boundary_mask = (
        getattr(manager, "detached_boundary_mask", None)
        if manager is not None else None
    )
    boundary_metrics = _edge_boundary_metrics(
        fragment_ids=fragment_ids,
        knn_idx=knn_idx,
        cut_edge_mask=cut_edge_mask,
        closure_boundary_mask=closure_boundary_mask,
        detached_boundary_mask=detached_boundary_mask,
        n=int(c.shape[0]),
    )
    crack_frame = int(getattr(front, "_advance_step", 0)) if front is not None else 0

    base_name = f"frame_{frame:04d}_impact_{frame - impact_frame:03d}"
    plot_path = out_dir / "snapshots" / f"{base_name}_raw_graph.png"
    physical_plot_path = out_dir / "snapshots" / f"{base_name}_physical.png"
    if save_plot:
        _save_raw_graph_diagnostic(
            positions=positions,
            damage=c,
            visited=visited,
            tips=tips,
            out_path=plot_path,
            title=(
                f"{prompt} | raw graph loop={frame} impact+{frame - impact_frame} "
                f"crack_step={crack_frame}"
            ),
            fragment_ids=fragment_ids,
            parent_index=front.parent_index if front is not None else None,
            activation_step=front.activation_step if front is not None else None,
            knn_idx=knn_idx,
            cut_edge_mask=cut_edge_mask,
            closure_boundary_mask=closure_boundary_mask,
            detached_boundary_mask=detached_boundary_mask,
            mesh_visual_state=mesh_visual_state,
            material_family=str(getattr(simulator, "material_family", "")),
        )

        # Physical-fragment view: actual MPM particle positions + the
        # filtered `_physical_fragment_labels` registry.  This view
        # avoids floating Gaussians because every node is either base
        # body (label 0) or a member of a coherent persistent chunk.
        physical_positions = None
        physical_frag_ids = None
        x_mpm = getattr(simulator, "x_mpm", None)
        surface_mask = getattr(simulator, "surface_mask", None)
        mapper = getattr(simulator, "mapper", None)
        physical_labels = getattr(simulator, "_physical_fragment_labels", None)
        surface_indices = getattr(simulator, "_surface_indices", None)
        if (
            x_mpm is not None
            and surface_mask is not None
            and mapper is not None
        ):
            try:
                physical_positions = mapper.mpm_to_world(x_mpm[surface_mask])
            except Exception:
                physical_positions = None
        if (
            physical_labels is not None
            and surface_indices is not None
            and physical_positions is not None
        ):
            try:
                physical_frag_ids = physical_labels[surface_indices]
            except Exception:
                physical_frag_ids = None

        if physical_positions is not None:
            _save_physical_diagnostic(
                positions=physical_positions[: c.shape[0]],
                damage=c,
                physical_fragment_ids=(
                    physical_frag_ids[: c.shape[0]]
                    if physical_frag_ids is not None
                    else None
                ),
                visited=visited,
                tips=tips,
                out_path=physical_plot_path,
                title=(
                    f"{prompt} | physical view loop={frame} "
                    f"impact+{frame - impact_frame}"
                ),
            )

        # Houdini-readable per-frame state (.geo.gz JSON).  Each snapshot
        # carries the per-Gaussian attributes a Houdini Copy-to-Points or
        # Volume Path Trace network needs: P, Cd, Alpha, scale (3-axis),
        # pscale, orient (quaternion in Houdini ijk-s convention),
        # fragment_id, damage, and N (crack normal).
        houdini_dir = out_dir / "houdini_export"
        houdini_path = houdini_dir / f"{base_name}.geo"
        try:
            _export_houdini_geo(
                houdini_path,
                simulator=simulator,
                compress=True,
            )
        except Exception as exc:  # pragma: no cover -- diagnostic best-effort
            print(f"[houdini_export] failed at frame {frame}: {exc}")

    row = {
        "loop_frame": int(frame),
        "since_impact": int(frame - impact_frame),
        "crack_step": crack_frame,
        "plot": str(plot_path) if save_plot else "",
        "c_max": float(c.max().item()),
        "c_mean": float(c.mean().item()),
        "phase_y_max": float(stats.get("phase_y_max", 0.0)),
        "phase_y_mean": float(stats.get("phase_y_mean", 0.0)),
        "phase_seed_gate_max": float(stats.get("phase_seed_gate_max", 0.0)),
        "phase_advance_gate_max": float(stats.get("phase_advance_gate_max", 0.0)),
        "phase_advance_gate_mean": float(stats.get("phase_advance_gate_mean", 0.0)),
        "phase_cut_gate_max": float(stats.get("phase_cut_gate_max", 0.0)),
        "phase_cut_edges": int(stats.get("phase_cut_edges", 0)),
        "volumetric_damage_max": float(stats.get("volumetric_damage_max", 0.0)),
        "volumetric_damage_mean": float(stats.get("volumetric_damage_mean", 0.0)),
        "surface_volume_damage_proxy_max": float(stats.get("surface_volume_damage_proxy_max", 0.0)),
        "surface_volume_damage_proxy_mean": float(stats.get("surface_volume_damage_proxy_mean", 0.0)),
        "rigid_angular_speed_max": float(stats.get("rigid_angular_speed_max", 0.0)),
        "rigid_angular_speed_mean": float(stats.get("rigid_angular_speed_mean", 0.0)),
        "cracked_count": int((c > 0.30).sum().item()),
        "visited_count": int(visited.sum().item()),
        "tip_count": int(tips.sum().item()),
        "n_fragments": int(stats.get("n_fragments", 1)),
        "hard_detached_nodes": int(stats.get("hard_detached_nodes", 0)),
        "new_detached_nodes": int(stats.get("new_detached_nodes", 0)),
        "cut_edges": int(stats.get("cut_edges", 0)),
        "closure_candidate_count": int(stats.get("closure_candidate_count", 0)),
        "closure_candidate_nodes": int(stats.get("closure_candidate_nodes", 0)),
        "closure_score_max": float(stats.get("closure_score_max", 0.0)),
        "closure_candidate_sizes": list(stats.get("closure_candidate_sizes", [])),
        "fragment_boundary_edges": int(boundary_metrics["fragment_boundary_edges"]),
        "cut_supported_fragment_boundary_edges": int(boundary_metrics["cut_supported_fragment_boundary_edges"]),
        "causal_supported_fragment_boundary_edges": int(boundary_metrics["causal_supported_fragment_boundary_edges"]),
        "closure_boundary_edges": int(boundary_metrics["closure_boundary_edges"]),
        "detached_boundary_edges": int(boundary_metrics["detached_boundary_edges"]),
        "fragment_boundary_cut_support_ratio": float(boundary_metrics["fragment_boundary_cut_support_ratio"]),
        "detached_boundary_edges_stat": int(stats.get("detached_boundary_edges", 0)),
        "support_lost_components": int(stats.get("support_lost_components", 0)),
        "release_candidate_count": int(stats.get("release_candidate_count", 0)),
        "boundary_cut_ratio_max": float(stats.get("boundary_cut_ratio_max", 0.0)),
        "authoritative_cut_nodes": int(stats.get("authoritative_cut_nodes", 0)),
        "open_release_patches": int(stats.get("open_release_patches", 0)),
        "impact_closure_patches": int(stats.get("impact_closure_patches", 0)),
        "impact_closure_nodes": int(stats.get("impact_closure_nodes", 0)),
        "impact_closure_score_max": float(stats.get("impact_closure_score_max", 0.0)),
        "phase_candidate_patches": int(stats.get("phase_candidate_patches", 0)),
        "phase_approved_patches": int(stats.get("phase_approved_patches", 0)),
        "phase_rejected_patches": int(stats.get("phase_rejected_patches", 0)),
        "phase_approved_nodes": int(stats.get("phase_approved_nodes", 0)),
        "phase_approval_score_max": float(stats.get("phase_approval_score_max", 0.0)),
        "phase_approval_score_mean": float(stats.get("phase_approval_score_mean", 0.0)),
        "phase_approval_cvol_max": float(stats.get("phase_approval_cvol_max", 0.0)),
        "phase_approval_gate_max": float(stats.get("phase_approval_gate_max", 0.0)),
        "pseudo_thickness_mass": float(stats.get("pseudo_thickness_mass", 0.0)),
        "birth_phase_score": float(stats.get("birth_phase_score", 0.0)),
        "birth_cvol_max": float(stats.get("birth_cvol_max", 0.0)),
        "birth_phase_gate_max": float(stats.get("birth_phase_gate_max", 0.0)),
        "birth_phase_approved": bool(stats.get("birth_phase_approved", False)),
        "tip_event_count": int(event_metrics["tip_event_count"]),
        "branch_event_count": int(event_metrics["branch_event_count"]),
        "branch_event_frame_count": int(event_metrics["branch_event_frame_count"]),
        "new_tip_events": int(new_event_metrics["tip_event_count"]),
        "new_branch_events": int(new_event_metrics["branch_event_count"]),
        "new_branch_event_frames": int(new_event_metrics["branch_event_frame_count"]),
        "branch_angle_mean_deg": float(event_metrics["branch_event_angle_mean_deg"]),
        "branch_angle_std_deg": float(event_metrics["branch_event_angle_std_deg"]),
        "released_node_ratio": float(fragment_metrics["final_released_node_ratio"]),
        "largest_fragment_ratio": float(fragment_metrics["final_largest_fragment_ratio"]),
        "fragment_label_count": int(fragment_metrics["final_fragment_label_count"]),
        "nonbase_fragment_count": int(fragment_metrics["final_nonbase_fragment_count"]),
        "nonbase_fragment_min_size": int(fragment_metrics["final_nonbase_fragment_min_size"]),
        "nonbase_fragment_mean_size": float(fragment_metrics["final_nonbase_fragment_mean_size"]),
        "detached_distance": float(stats.get("detached_distance", 0.0)),
        "physical_detached_distance": float(stats.get("physical_detached_distance", 0.0)),
        "physical_release_displacement": float(stats.get("physical_release_displacement", 0.0)),
        "physical_lateral_release_displacement": float(stats.get("physical_lateral_release_displacement", 0.0)),
        "physical_fragment_lateral_spread": float(stats.get("physical_fragment_lateral_spread", 0.0)),
        "physical_fragment_drop": float(stats.get("physical_fragment_drop", 0.0)),
        "top_component_sizes": list(stats.get("top_component_sizes", [])),
    }
    if manager is not None:
        row.update({
            "strict": bool(getattr(manager, "crack_connected_release_only", False)),
            "open_release_enabled": bool(getattr(manager, "open_crack_release_enable", False)),
        })
    return row, len(events)


def _write_rows(rows: list[dict], out_dir: Path, *, stem: str = "progression_snapshots") -> None:
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_for_csv(row))


def _annotate_birth_causality(rows: list[dict]) -> dict:
    """Mark the first snapshot where a fragment label appears and record support."""
    previous_detached = 0
    birth_row = None
    for row in rows:
        detached = int(row.get("nonbase_fragment_count", 0) or 0)
        new_detached = int(row.get("new_detached_nodes", 0) or 0)
        is_birth = detached > previous_detached or (detached > 0 and new_detached > 0)
        row["fragment_birth_event"] = bool(is_birth)
        row["fragment_birth_cut_edges"] = (
            int(row.get("causal_supported_fragment_boundary_edges", 0) or 0)
            if is_birth else 0
        )
        row["fragment_birth_closure_score"] = (
            float(row.get("closure_score_max", 0.0) or 0.0)
            if is_birth else 0.0
        )
        row["fragment_birth_boundary_cut_support_ratio"] = (
            float(row.get("fragment_boundary_cut_support_ratio", 0.0) or 0.0)
            if is_birth else 0.0
        )
        row["fragment_birth_phase_score"] = (
            float(row.get("birth_phase_score", row.get("phase_approval_score_max", 0.0)) or 0.0)
            if is_birth else 0.0
        )
        row["fragment_birth_cvol_max"] = (
            float(row.get("birth_cvol_max", row.get("surface_volume_damage_proxy_max", 0.0)) or 0.0)
            if is_birth else 0.0
        )
        row["fragment_birth_phase_approved"] = (
            bool(row.get("birth_phase_approved", False) or row.get("phase_approved_patches", 0))
            if is_birth else False
        )
        row["branch_events_before_birth"] = (
            int(row.get("branch_event_count", 0) or 0)
            if is_birth else 0
        )
        if birth_row is None and is_birth:
            birth_row = row
        previous_detached = max(previous_detached, detached)

    if birth_row is None:
        return {
            "first_detach_frame": -1,
            "first_detach_since_impact": -1,
            "fragment_birth_cut_edges": 0,
            "fragment_birth_closure_score": 0.0,
            "fragment_birth_boundary_cut_support_ratio": 0.0,
            "fragment_birth_phase_score": 0.0,
            "fragment_birth_cvol_max": 0.0,
            "fragment_birth_phase_approved": False,
            "branch_events_before_detach": 0,
        }

    return {
        "first_detach_frame": int(birth_row.get("loop_frame", -1)),
        "first_detach_since_impact": int(birth_row.get("since_impact", -1)),
        "fragment_birth_cut_edges": int(birth_row.get("fragment_birth_cut_edges", 0) or 0),
        "fragment_birth_closure_score": float(birth_row.get("fragment_birth_closure_score", 0.0) or 0.0),
        "fragment_birth_boundary_cut_support_ratio": float(
            birth_row.get("fragment_birth_boundary_cut_support_ratio", 0.0) or 0.0
        ),
        "fragment_birth_phase_score": float(
            birth_row.get("fragment_birth_phase_score", 0.0) or 0.0
        ),
        "fragment_birth_cvol_max": float(
            birth_row.get("fragment_birth_cvol_max", 0.0) or 0.0
        ),
        "fragment_birth_phase_approved": bool(
            birth_row.get("fragment_birth_phase_approved", False)
        ),
        "branch_events_before_detach": int(birth_row.get("branch_events_before_birth", 0) or 0),
    }


def _write_report(
    *,
    rows: list[dict],
    summary_row: dict,
    prompt: str,
    out_dir: Path,
    elapsed_sec: float,
    metric_rows: list[dict] | None = None,
) -> None:
    diagnostic_rows = metric_rows if metric_rows is not None else rows
    lines = [
        "# Gravity Crack Progression",
        "",
        f"- prompt: `{prompt}`",
        f"- elapsed_sec: {elapsed_sec:.1f}",
        f"- verdict: `{summary_row.get('fragment_release_verdict', '')}`",
        f"- mode: `{summary_row.get('expected_release_mode', '')}`",
        f"- strict: `{summary_row.get('runtime_crack_connected_release_only', False)}`",
        f"- first_detach_since_impact: `{summary_row.get('first_detach_since_impact', -1)}`",
        f"- birth_cut_edges: `{summary_row.get('fragment_birth_cut_edges', 0)}`",
        f"- birth_closure_score: `{float(summary_row.get('fragment_birth_closure_score', 0.0)):.3f}`",
        f"- birth_bcut: `{float(summary_row.get('fragment_birth_boundary_cut_support_ratio', 0.0)):.3f}`",
        f"- birth_phase_score: `{float(summary_row.get('fragment_birth_phase_score', 0.0)):.3f}`",
        f"- birth_cvol_max: `{float(summary_row.get('fragment_birth_cvol_max', 0.0)):.3f}`",
        f"- birth_phase_approved: `{bool(summary_row.get('fragment_birth_phase_approved', False))}`",
        "",
        "## Timeline",
        "",
        "| loop | impact+ | crack step | c_max | Ymax | seed/adv/cut gate | cvol/proxy | phase birth | omega | cracked | visited | tips | labels | detached | hard det | new det | rel | largest | closure | causal edges | bcut | birth | phys open/lat/drop | new branches | total branches | impact/open | raw graph |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        plot_rel = Path(row["plot"]).relative_to(out_dir)
        lines.append(
            "| {loop} | {impact} | {crack} | {cmax:.3f} | {ymax:.3f} | {seed_gate:.2f}/{gate:.2f}/{cut_gate:.2f} | {cvol:.3f}/{cproxy:.3f} | {phase_birth} | {omega:.3f} | {cracked} | {visited} | {tips} | "
            "{labels} | {detached} | {hard_det} | {new_det} | {rel:.3f} | {largest:.3f} | {closure:.3f} | {causal_edges} | {bcut:.3f} | {birth} | {phys_open} | "
            "{new_branch} | {branch} | {impactp}/{openp} | [{plot}]({plot}) |".format(
                loop=row["loop_frame"],
                impact=row["since_impact"],
                crack=row["crack_step"],
                cmax=float(row["c_max"]),
                ymax=float(row.get("phase_y_max", 0.0)),
                seed_gate=float(row.get("phase_seed_gate_max", 0.0)),
                gate=float(row.get("phase_advance_gate_max", 0.0)),
                cut_gate=float(row.get("phase_cut_gate_max", 0.0)),
                cvol=float(row.get("volumetric_damage_max", 0.0)),
                cproxy=float(row.get("surface_volume_damage_proxy_max", 0.0)),
                phase_birth=(
                    f"{float(row.get('phase_approval_score_max', 0.0)):.2f}/"
                    f"{int(row.get('phase_approved_patches', 0))}/"
                    f"{int(row.get('phase_rejected_patches', 0))}"
                ),
                omega=float(row.get("rigid_angular_speed_max", 0.0)),
                cracked=row["cracked_count"],
                visited=row["visited_count"],
                tips=row["tip_count"],
                labels=row.get("fragment_label_count", row["n_fragments"]),
                detached=row.get("nonbase_fragment_count", 0),
                hard_det=row.get("hard_detached_nodes", 0),
                new_det=row.get("new_detached_nodes", 0),
                rel=float(row["released_node_ratio"]),
                largest=float(row["largest_fragment_ratio"]),
                closure=float(row["closure_score_max"]),
                causal_edges=int(row.get("causal_supported_fragment_boundary_edges", 0) or 0),
                bcut=float(row.get("fragment_boundary_cut_support_ratio", 0.0)),
                birth="Y" if bool(row.get("fragment_birth_event", False)) else "",
                phys_open=(
                    f"{float(row.get('physical_release_displacement', 0.0)):.4f}/"
                    f"{float(row.get('physical_lateral_release_displacement', 0.0)):.4f}/"
                    f"{float(row.get('physical_fragment_drop', 0.0)):.4f}"
                ),
                new_branch=row["new_branch_events"],
                branch=row["branch_event_count"],
                impactp=row.get("impact_closure_patches", 0),
                openp=row["open_release_patches"],
                plot=str(plot_rel),
            )
        )

    nonclosure = max(
        int(row["open_release_patches"])
        for row in diagnostic_rows
    ) if diagnostic_rows else 0
    impact_closure = max((int(row.get("impact_closure_patches", 0)) for row in diagnostic_rows), default=0)
    branch_frames = max((int(row["branch_event_frame_count"]) for row in diagnostic_rows), default=0)
    max_phys_open = max((float(row.get("physical_release_displacement", 0.0)) for row in diagnostic_rows), default=0.0)
    max_phys_lat = max((float(row.get("physical_lateral_release_displacement", 0.0)) for row in diagnostic_rows), default=0.0)
    max_cvol = max((float(row.get("volumetric_damage_max", 0.0)) for row in diagnostic_rows), default=0.0)
    max_proxy = max((float(row.get("surface_volume_damage_proxy_max", 0.0)) for row in diagnostic_rows), default=0.0)
    max_phase_score = max((float(row.get("phase_approval_score_max", 0.0)) for row in diagnostic_rows), default=0.0)
    lines.extend([
        "",
        "## Readout",
        "",
        f"- branch tips appeared over `{branch_frames}` crack-front steps.",
        f"- max impact closure patches: `{impact_closure}`.",
        f"- max non-closure release patches: `{nonclosure}`.",
        f"- max physical release motion: `{max_phys_open:.4f}` total, `{max_phys_lat:.4f}` lateral.",
        f"- max narrow-band volumetric damage: `{max_cvol:.3f}`.",
        f"- max surface volume proxy: `{max_proxy:.3f}`; max patch phase approval score: `{max_phase_score:.3f}`.",
        f"- first fragment birth: impact+`{summary_row.get('first_detach_since_impact', -1)}` with `{summary_row.get('fragment_birth_cut_edges', 0)}` causal boundary edges, phase score `{float(summary_row.get('fragment_birth_phase_score', 0.0)):.3f}`, cvol proxy `{float(summary_row.get('fragment_birth_cvol_max', 0.0)):.3f}`, and `{summary_row.get('branch_events_before_detach', 0)}` accumulated branch events.",
        "- `bcut` is the fraction of fragment-label boundary edges supported by current cut, current closure, or saved birth-time detached boundary edges.",
        "- Top row emphasizes causal cut, closure, and detached-boundary edges; low-alpha parent edges are retained only as propagation context.",
        "- Bottom row transfers detached particle labels onto actual OBJ faces, omits mixed boundary faces from the base mesh, and draws each detached label with one rigid piece transform.",
        "- Visualization is raw-only: MPM particle states drive crack/fragment labels, while OBJ faces are used only as the diagnostic display shell.",
        "- Fragment labels are expected only once closure candidates and cut boundaries become strong enough.",
        "- Use the linked PNGs to verify branch tips emerge during propagation instead of only forming a final perimeter ring.",
        "",
    ])
    (out_dir / "progression_report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_prompt_progression(
    *,
    pipeline: ManifoldFracturePipeline,
    args: argparse.Namespace,
    prompt: str,
    out_dir: Path,
    mesh_visual_state: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    snapshot_rows: list[dict] = []
    metric_rows: list[dict] = []
    impact_frame: int | None = None
    previous_event_count = 0

    def callback(frame: int, simulator, stats: dict) -> None:
        nonlocal impact_frame, previous_event_count
        if not bool(stats.get("gravity_contacted", False)):
            return
        if impact_frame is None:
            impact_frame = int(frame)
        since_impact = int(frame) - int(impact_frame)
        is_snapshot = (
            since_impact == 0
            or since_impact % max(int(args.snapshot_stride), 1) == 0
            or int(frame) == int(args.gravity_frames) - 1
        )
        row, previous_event_count = _snapshot_metrics(
            frame=int(frame),
            impact_frame=int(impact_frame),
            simulator=simulator,
            stats=stats,
            prompt=prompt,
            out_dir=out_dir,
            previous_event_count=previous_event_count,
            mesh_visual_state=mesh_visual_state,
            save_plot=is_snapshot,
        )
        metric_rows.append(row)
        if not is_snapshot:
            return
        snapshot_rows.append(row)
        _write_rows(snapshot_rows, out_dir, stem="progression_snapshots")
        _write_rows(metric_rows, out_dir, stem="progression_metrics")
        print(
            "[snapshot] frame={frame} impact+{impact} crack={crack} "
            "labels={labels} detached={detached} hard_det={hard_det} new_det={new_det} "
            "branches={branches} new={new}".format(
                frame=row["loop_frame"],
                impact=row["since_impact"],
                crack=row["crack_step"],
                labels=row.get("fragment_label_count", row["n_fragments"]),
                detached=row.get("nonbase_fragment_count", 0),
                hard_det=row.get("hard_detached_nodes", 0),
                new_det=row.get("new_detached_nodes", 0),
                branches=row["branch_event_count"],
                new=row["new_branch_events"],
            ),
            flush=True,
        )

    t0 = time.time()
    result = pipeline.run(
        prompt,
        num_frames=int(args.gravity_frames),
        save_frames=False,
        return_frames=False,
        diagnostic_callback=callback,
        override_params={
            "rendering.render_frames": [],
            "output.make_video": False,
            "simulation.save_checkpoint": False,
            "manifold.fragment_detect_every": max(int(args.fragment_every), 1),
            **getattr(args, "runtime_overrides", {}),
        },
    )

    history = list(getattr(pipeline.engine, "last_stats_history", []))
    prior = result["material_prior"]
    summary_row = {
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
        "snapshot_count": len(snapshot_rows),
        "progression_report": str(out_dir / "progression_report.md"),
    }
    summary_row.update(_runtime_release_row(result["params"]))
    summary_row.update(_summarize_history(history))
    simulator = getattr(pipeline.engine, "last_simulator", None)
    manager = getattr(simulator, "fragment_manager", None) if simulator is not None else None
    if manager is not None:
        summary_row.update({
            "runtime_crack_connected_release_only": bool(getattr(manager, "crack_connected_release_only", False)),
            "runtime_phase_approval_enable": bool(getattr(manager, "phase_approval_enable", True)),
            "runtime_open_crack_release_enable": bool(getattr(manager, "open_crack_release_enable", True)),
        })
    _write_rows(metric_rows, out_dir, stem="progression_metrics")
    if snapshot_rows:
        _annotate_birth_causality(snapshot_rows)
        birth_summary = _annotate_birth_causality(metric_rows)
        summary_row.update(birth_summary)
        _write_rows(snapshot_rows, out_dir, stem="progression_snapshots")
        _write_rows(metric_rows, out_dir, stem="progression_metrics")
        final = snapshot_rows[-1]
        summary_row.update({
            "final_snapshot": final["plot"],
            "final_released_node_ratio": final["released_node_ratio"],
            "final_largest_fragment_ratio": final["largest_fragment_ratio"],
            "final_component_count": final.get("fragment_label_count", final.get("n_fragments", 0)),
            "final_nonbase_fragment_count": final.get("nonbase_fragment_count", 0),
            "final_branch_event_count": final["branch_event_count"],
            "final_hard_detached_nodes": final.get("hard_detached_nodes", 0),
            "final_new_detached_nodes": final.get("new_detached_nodes", 0),
            "final_bcut": final.get("fragment_boundary_cut_support_ratio", 0.0),
            "final_phase_approval_score": final.get("phase_approval_score_max", 0.0),
            "final_phase_approved_patches": final.get("phase_approved_patches", 0),
            "final_phase_rejected_patches": final.get("phase_rejected_patches", 0),
            "final_surface_volume_damage_proxy": final.get("surface_volume_damage_proxy_max", 0.0),
            "max_phase_approval_score": max(
                (float(row.get("phase_approval_score_max", 0.0)) for row in metric_rows),
                default=0.0,
            ),
            "max_surface_volume_damage_proxy": max(
                (float(row.get("surface_volume_damage_proxy_max", 0.0)) for row in metric_rows),
                default=0.0,
            ),
            "final_physical_release_displacement": final["physical_release_displacement"],
            "final_physical_lateral_release_displacement": final.get(
                "physical_lateral_release_displacement", 0.0
            ),
            "final_physical_fragment_lateral_spread": final.get(
                "physical_fragment_lateral_spread", 0.0
            ),
            "final_physical_fragment_drop": final["physical_fragment_drop"],
        })
    else:
        summary_row.update(_annotate_birth_causality(snapshot_rows))
    summary_row["expected_release_mode"], summary_row["fragment_release_verdict"] = _release_verdict_for_row(summary_row)

    (out_dir / "summary.json").write_text(json.dumps(summary_row, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(
        rows=snapshot_rows,
        summary_row=summary_row,
        prompt=prompt,
        out_dir=out_dir,
        elapsed_sec=time.time() - t0,
        metric_rows=metric_rows,
    )
    print(f"\nProgression report: {out_dir / 'progression_report.md'}", flush=True)
    return summary_row


def _write_sweep_report(rows: list[dict], out_dir: Path) -> None:
    lines = [
        "# Gravity Crack Progression Sweep",
        "",
        "Each prompt directory contains raw matplotlib snapshots at the configured cadence. Every PNG shows crack propagation and fragment surface patches together.",
        "",
        "| prompt | family | style | verdict | strict | snapshots | detached | rel | bcut | phase | birth | open | report | final snapshot |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        if row.get("error"):
            lines.append(
                "| {prompt} | ERROR |  | FAIL |  | 0 | 0 | 0.000 | 0.000 | 0.000 | -1 |  |  |  |".format(
                    prompt=str(row.get("prompt", ""))[:52],
                )
            )
            continue
        report = Path(row.get("progression_report", ""))
        final_snapshot = Path(row.get("final_snapshot", ""))
        try:
            report_rel = report.relative_to(out_dir)
        except ValueError:
            report_rel = report
        try:
            final_rel = final_snapshot.relative_to(out_dir)
        except ValueError:
            final_rel = final_snapshot
        lines.append(
            "| {prompt} | {family} | {style} | {verdict} | {strict} | {snapshots} | {detached} | "
            "{rel:.3f} | {bcut:.3f} | {phase:.3f} | {birth} | {openp} | [{report}]({report}) | [{final}]({final}) |".format(
                prompt=str(row.get("prompt", ""))[:52],
                family=row.get("family", ""),
                style=row.get("sentence_style", ""),
                verdict=row.get("fragment_release_verdict", ""),
                strict="Y" if bool(row.get("runtime_crack_connected_release_only", False)) else "",
                snapshots=int(row.get("snapshot_count", 0)),
                detached=int(row.get("final_nonbase_fragment_count", 0) or 0),
                rel=float(row.get("final_released_node_ratio", 0.0) or 0.0),
                bcut=float(row.get("final_bcut", 0.0) or 0.0),
                phase=float(row.get("fragment_birth_phase_score", row.get("max_phase_approval_score", 0.0)) or 0.0),
                birth=int(
                    row["first_detach_since_impact"]
                    if row.get("first_detach_since_impact") is not None
                    else -1
                ),
                openp=int(row.get("max_open_release_patches", 0) or 0),
                report=str(report_rel),
                final=str(final_rel),
            )
        )
    (out_dir / "progression_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save 5-frame crack/fragment progression snapshots.")
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--out", default="output/gravity_crack_progression_10k_brittle_v1")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gravity-particles", type=int, default=10000)
    parser.add_argument("--gravity-frames", type=int, default=56)
    parser.add_argument("--gravity-grids", type=int, default=64)
    parser.add_argument("--physics-substeps", type=int, default=3)
    parser.add_argument("--fragment-every", type=int, default=1)
    parser.add_argument("--snapshot-stride", type=int, default=5)
    parser.add_argument("--drop-center-z", type=float, default=0.42)
    parser.add_argument("--gravity-z", type=float, default=-3500.0)
    parser.add_argument(
        "--override-json",
        default=None,
        help=(
            "JSON object or path to a JSON object of ForwardEngine/runtime "
            "overrides, applied after CLIP material/style runtime presets."
        ),
    )
    args = parser.parse_args()
    args.runtime_overrides = _read_override_json(args.override_json)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(args.prompts_file, args.prompt)
    (out_dir / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    config_args = SimpleNamespace(
        config=args.config,
        gravity_particles=args.gravity_particles,
        gravity_grids=args.gravity_grids,
        gravity_frames=args.gravity_frames,
        physics_substeps=args.physics_substeps,
        drop_center_z=args.drop_center_z,
        gravity_z=args.gravity_z,
    )
    config_path = _save_gravity_config(config_args, out_dir)

    t0 = time.time()
    pipeline = ManifoldFracturePipeline(
        str(config_path),
        material_source="clip",
        fast_mode=False,
        db_path=args.db_path,
        clip_model=args.clip_model,
    )
    surface_points = np.asarray(pipeline.engine._surface_pcd.points, dtype=np.float32).copy()
    mesh_visual_state = _build_mesh_visual_state(pipeline.engine.base_config, surface_points)
    print(
        "[mesh-face] using actual OBJ faces: "
        f"faces={mesh_visual_state['faces'].shape[0]} vertices={mesh_visual_state['vertices_ref'].shape[0]}",
        flush=True,
    )
    summaries: list[dict] = []
    for idx, prompt in enumerate(prompts):
        prompt_dir = (
            out_dir / f"{idx:02d}_{_slug(prompt)}"
            if len(prompts) > 1
            else out_dir
        )
        print(f"\n[progression:{idx + 1}/{len(prompts)}] {prompt!r}", flush=True)
        try:
            summary = _run_prompt_progression(
                pipeline=pipeline,
                args=args,
                prompt=prompt,
                out_dir=prompt_dir,
                mesh_visual_state=mesh_visual_state,
            )
        except Exception as exc:
            summary = {
                "prompt": prompt,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": time.time() - t0,
            }
            print(f"[progression:error] {prompt!r}: {exc}", flush=True)
        summaries.append(summary)
        (out_dir / "progression_sweep_summary.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_sweep_report(summaries, out_dir)
        torch.cuda.empty_cache()

    print(f"\nProgression sweep report: {out_dir / 'progression_sweep_report.md'}", flush=True)
    print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
