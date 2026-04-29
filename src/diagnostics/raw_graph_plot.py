"""Matplotlib diagnostics for raw manifold crack and fragment state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_raw_graph_diagnostic(
    *,
    positions: torch.Tensor,
    damage: torch.Tensor,
    visited: torch.Tensor,
    tips: torch.Tensor,
    out_path: Path,
    title: str,
    fragment_ids: torch.Tensor | None = None,
    parent_index: torch.Tensor | None = None,
    activation_step: torch.Tensor | None = None,
    knn_idx: torch.Tensor | None = None,
    cut_edge_mask: torch.Tensor | None = None,
    closure_boundary_mask: torch.Tensor | None = None,
    detached_boundary_mask: torch.Tensor | None = None,
    mesh_visual_state: dict | None = None,
    material_family: str = "",
    cracked_threshold: float = 0.30,
) -> None:
    """Save one raw-only diagnostic with crack and fragment graphs separated."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.collections import LineCollection, PolyCollection
    except Exception as exc:
        print(f"[plot:warn] matplotlib unavailable: {exc}", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = positions.detach().float().cpu().numpy()
    c = damage.detach().float().cpu().numpy()
    visited_np = visited.detach().bool().cpu().numpy()
    tips_np = tips.detach().bool().cpu().numpy()
    cracked_np = c > float(cracked_threshold)
    frag_np = (
        fragment_ids[: positions.shape[0]].detach().long().cpu().numpy()
        if fragment_ids is not None
        else None
    )
    knn_np = (
        knn_idx[: positions.shape[0]].detach().long().cpu().numpy()
        if knn_idx is not None
        else None
    )
    cut_edge_np = (
        cut_edge_mask[: positions.shape[0]].detach().bool().cpu().numpy()
        if cut_edge_mask is not None and knn_idx is not None
        else None
    )
    closure_edge_np = (
        closure_boundary_mask[: positions.shape[0]].detach().bool().cpu().numpy()
        if closure_boundary_mask is not None and knn_idx is not None
        else None
    )
    detached_edge_np = (
        detached_boundary_mask[: positions.shape[0]].detach().bool().cpu().numpy()
        if detached_boundary_mask is not None and knn_idx is not None
        else None
    )
    segment_np = None
    if parent_index is not None:
        parent = parent_index[: positions.shape[0]].detach().long().cpu()
        valid = visited.detach().bool().cpu() & (parent >= 0) & (parent < positions.shape[0])
        child_idx = torch.where(valid)[0]
        if child_idx.numel() > 0:
            step_np = None
            if activation_step is not None:
                step_np = activation_step[: positions.shape[0]].detach().long().cpu()[child_idx].numpy()
            segment_np = (child_idx.numpy(), parent[child_idx].numpy(), step_np)

    mesh_face_vertices = None
    mesh_faces = None
    mesh_face_frag = None
    mesh_face_base_mask = None
    mesh_face_has_detached = None
    mesh_edges = None
    mesh_edge_faces = None
    if mesh_visual_state is not None:
        try:
            vertices_ref = np.asarray(mesh_visual_state["vertices_ref"], dtype=np.float32)
            surface_ref = np.asarray(mesh_visual_state["surface_ref"], dtype=np.float32)
            mesh_faces = np.asarray(mesh_visual_state["faces"], dtype=np.int64)
            vertex_anchor = np.asarray(mesh_visual_state["vertex_anchor"], dtype=np.int64)
            face_anchor = np.asarray(mesh_visual_state["face_anchor"], dtype=np.int64)
            face_anchors = np.asarray(
                mesh_visual_state.get("face_anchors", face_anchor[:, None]),
                dtype=np.int64,
            )
            mesh_edges = np.asarray(mesh_visual_state["edges"], dtype=np.int64)
            mesh_edge_faces = np.asarray(mesh_visual_state["edge_faces"], dtype=np.int64)

            n_surface = min(pos.shape[0], surface_ref.shape[0])
            disp = np.zeros_like(surface_ref, dtype=np.float32)
            disp[:n_surface] = pos[:n_surface].astype(np.float32) - surface_ref[:n_surface]
            face_anchors = np.clip(face_anchors, 0, max(pos.shape[0] - 1, 0))
            vertex_anchor = np.clip(vertex_anchor, 0, max(n_surface - 1, 0))
            anchored_vertices = vertices_ref.astype(np.float32).copy()
            if n_surface > 0 and vertex_anchor.size == anchored_vertices.shape[0]:
                anchored_vertices += disp[vertex_anchor]
            deform_base_mesh = str(material_family) == "diffuse_damage"

            if frag_np is not None:
                neighbor_labels = frag_np[face_anchors]
                mesh_face_frag = np.zeros(neighbor_labels.shape[0], dtype=np.int64)
                for row_idx, labels in enumerate(neighbor_labels):
                    labels = labels[labels > 0]
                    if labels.size == 0:
                        continue
                    values, counts = np.unique(labels, return_counts=True)
                    best = int(np.argmax(counts))
                    if counts[best] >= max(2, int(np.ceil(0.42 * neighbor_labels.shape[1]))):
                        mesh_face_frag[row_idx] = int(values[best])
                mesh_face_has_detached = np.any(neighbor_labels > 0, axis=1)
                mesh_face_base_mask = mesh_face_frag <= 0
            else:
                mesh_face_frag = np.zeros(mesh_faces.shape[0], dtype=np.int64)
                neighbor_labels = np.zeros(face_anchors.shape, dtype=np.int64)
                mesh_face_has_detached = np.zeros(mesh_faces.shape[0], dtype=bool)
                mesh_face_base_mask = np.ones(mesh_faces.shape[0], dtype=bool)

            def _fit_piece_transform(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                mask = mask[:n_surface]
                if not np.any(mask):
                    mask = np.ones(n_surface, dtype=bool)
                rest_pts = surface_ref[:n_surface][mask].astype(np.float64)
                cur_pts = pos[:n_surface][mask].astype(np.float64)
                rest_com = rest_pts.mean(axis=0)
                cur_com = cur_pts.mean(axis=0)
                if rest_pts.shape[0] < 3:
                    return rest_com.astype(np.float32), cur_com.astype(np.float32), np.eye(3, dtype=np.float32)
                rest_centered = rest_pts - rest_com
                cur_centered = cur_pts - cur_com
                try:
                    u, _, vh = np.linalg.svd(rest_centered.T @ cur_centered)
                    rot = u @ vh
                    if np.linalg.det(rot) < 0.0:
                        u[:, -1] *= -1.0
                        rot = u @ vh
                except np.linalg.LinAlgError:
                    rot = np.eye(3, dtype=np.float64)
                return rest_com.astype(np.float32), cur_com.astype(np.float32), rot.astype(np.float32)

            label_transforms: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            if frag_np is not None:
                base_particle_mask = frag_np[:n_surface] <= 0
                label_transforms[0] = _fit_piece_transform(base_particle_mask)
                for label in [int(v) for v in np.unique(mesh_face_frag) if int(v) > 0]:
                    label_transforms[label] = _fit_piece_transform(frag_np[:n_surface] == label)
            else:
                label_transforms[0] = _fit_piece_transform(np.ones(n_surface, dtype=bool))

            mesh_face_vertices = np.empty((mesh_faces.shape[0], 3, 3), dtype=np.float32)
            ref_face_vertices = vertices_ref[mesh_faces].astype(np.float32)
            anchored_face_vertices = anchored_vertices[mesh_faces].astype(np.float32)
            for label in [0] + [int(v) for v in np.unique(mesh_face_frag) if int(v) > 0]:
                rest_com, cur_com, rot = label_transforms.get(label, label_transforms[0])
                face_mask = mesh_face_frag == label if label > 0 else mesh_face_frag <= 0
                if not np.any(face_mask):
                    continue
                if label == 0 and deform_base_mesh:
                    mesh_face_vertices[face_mask] = anchored_face_vertices[face_mask]
                    continue
                mesh_face_vertices[face_mask] = (
                    (ref_face_vertices[face_mask] - rest_com.reshape(1, 1, 3)) @ rot
                    + cur_com.reshape(1, 1, 3)
                )
        except Exception as exc:
            print(f"[plot:warn] mesh-face diagnostic disabled: {exc}", flush=True)
            mesh_face_vertices = None
            mesh_faces = None
            mesh_face_frag = None
            mesh_face_base_mask = None
            mesh_face_has_detached = None
            mesh_edges = None
            mesh_edge_faces = None

    views = [
        ("X-Y top", 0, 1, "X", "Y"),
        ("X-Z front", 0, 2, "X", "Z"),
        ("Y-Z side", 1, 2, "Y", "Z"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.7), constrained_layout=True)
    fig.suptitle(title, fontsize=11)

    def _setup_axis(ax, view_name: str, xlabel: str, ylabel: str) -> None:
        ax.set_title(view_name, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.12, linewidth=0.5)

    def _draw_mesh_faces(
        ax,
        i: int,
        j: int,
        face_mask: np.ndarray,
        *,
        color,
        alpha: float,
        zorder: int,
        edge_color="none",
        edge_width: float = 0.0,
    ) -> None:
        if mesh_face_vertices is None or mesh_faces is None:
            return
        face_idx = np.where(face_mask)[0]
        if face_idx.size == 0:
            return
        polys = mesh_face_vertices[face_idx][:, :, [i, j]]
        ax.add_collection(
            PolyCollection(
                polys,
                facecolors=[color],
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=alpha,
                zorder=zorder,
            )
        )
        ax.update_datalim(polys.reshape(-1, 2))
        ax.autoscale_view()

    def _draw_mesh_background(ax, i: int, j: int, *, alpha: float, zorder: int) -> bool:
        if mesh_faces is None:
            return False
        _draw_mesh_faces(
            ax,
            i,
            j,
            np.ones(mesh_faces.shape[0], dtype=bool),
            color="#bfc6ce",
            alpha=alpha,
            zorder=zorder,
            edge_color="none",
        )
        return True

    def _plot_actual_fragment_mesh(ax, i: int, j: int) -> bool:
        if mesh_faces is None or mesh_face_frag is None:
            return False
        base_mask = (
            mesh_face_base_mask
            if mesh_face_base_mask is not None
            else mesh_face_frag <= 0
        )
        _draw_mesh_faces(
            ax,
            i,
            j,
            base_mask,
            color="#c6ccd2",
            alpha=0.16,
            zorder=1,
            edge_color="none",
        )

        labels = [int(label) for label in np.unique(mesh_face_frag) if int(label) > 0]
        if not labels:
            ax.text(
                0.5, 0.5, "no detached fragments",
                transform=ax.transAxes,
                ha="center", va="center", color="#555555", fontsize=10,
            )
            return True

        cmap = plt.get_cmap("tab20")
        for label_idx, label in enumerate(labels):
            _draw_mesh_faces(
                ax,
                i,
                j,
                mesh_face_frag == label,
                color=cmap(label_idx % cmap.N),
                alpha=0.46,
                zorder=4,
                edge_color="none",
            )
        return True

    def _unique_projected_points(idx: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        pts = pos[idx][:, [i, j]]
        if pts.shape[0] == 0:
            return idx, pts
        rounded = np.round(pts, decimals=6)
        _, unique = np.unique(rounded, axis=0, return_index=True)
        unique = np.sort(unique)
        return idx[unique], pts[unique]

    def _draw_patch_surface(
        ax,
        idx: np.ndarray,
        i: int,
        j: int,
        *,
        color,
        max_points: int,
        face_alpha: float,
        mesh_alpha: float,
        boundary_alpha: float,
        point_alpha: float,
        boundary_width: float,
        zorder: int,
        crack_boundary_color=None,
    ) -> None:
        if idx.size == 0:
            return
        if idx.size > max_points:
            keep = np.linspace(0, idx.size - 1, max_points).astype(np.int64)
            idx = idx[keep]
        idx, pts = _unique_projected_points(idx, i, j)

        if pts.shape[0] < 3:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=1.2, color=color, alpha=point_alpha,
                linewidths=0, zorder=zorder + 3,
            )
            return
        extent = pts.max(axis=0) - pts.min(axis=0)
        if float(np.linalg.norm(extent)) < 1e-6:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=1.2, color=color, alpha=point_alpha,
                linewidths=0, zorder=zorder + 3,
            )
            return

        try:
            tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
        except Exception:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=1.0, color=color, alpha=point_alpha,
                linewidths=0, zorder=zorder + 3,
            )
            return

        triangles = tri.triangles
        if triangles.shape[0] == 0:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=1.0, color=color, alpha=point_alpha,
                linewidths=0, zorder=zorder + 3,
            )
            return

        tri_pts = pts[triangles]
        e01 = np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 1], axis=1)
        e12 = np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 2], axis=1)
        e20 = np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 0], axis=1)
        all_edges = np.concatenate([e01, e12, e20])
        local_scale = max(float(np.quantile(all_edges, 0.60)), 1e-6)
        diag = max(float(np.linalg.norm(extent)), 1e-6)
        max_allowed = min(max(2.8 * local_scale, 0.018 * diag), 0.18 * diag)
        long_mask = np.maximum.reduce([e01, e12, e20]) > max_allowed
        valid_triangles = triangles[~long_mask]
        if valid_triangles.shape[0] == 0:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=1.0, color=color, alpha=point_alpha,
                linewidths=0, zorder=zorder + 3,
            )
            return

        ax.add_collection(
            PolyCollection(
                pts[valid_triangles],
                facecolors=[color],
                edgecolors="none",
                alpha=face_alpha,
                zorder=zorder,
            )
        )

        edges = np.concatenate(
            [
                valid_triangles[:, [0, 1]],
                valid_triangles[:, [1, 2]],
                valid_triangles[:, [2, 0]],
            ],
            axis=0,
        )
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        mesh_edges = unique_edges[counts > 1]
        if mesh_edges.shape[0] > 1800:
            keep = np.linspace(0, mesh_edges.shape[0] - 1, 1800).astype(np.int64)
            mesh_edges = mesh_edges[keep]
        if mesh_edges.shape[0] > 0:
            ax.add_collection(
                LineCollection(
                    pts[mesh_edges],
                    colors=[color],
                    linewidths=0.16,
                    alpha=mesh_alpha,
                    zorder=zorder + 1,
                )
            )
        if boundary_edges.shape[0] > 0:
            line_color = crack_boundary_color if crack_boundary_color is not None else color
            ax.add_collection(
                LineCollection(
                    pts[boundary_edges],
                    colors=[line_color],
                    linewidths=boundary_width,
                    alpha=boundary_alpha,
                    zorder=zorder + 2,
                )
            )
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=0.60, color=color, alpha=point_alpha,
            linewidths=0, zorder=zorder + 3,
        )

    def _plot_fragment_surface(ax, i: int, j: int) -> None:
        base_idx = (
            np.where(frag_np <= 0)[0]
            if frag_np is not None
            else np.arange(pos.shape[0], dtype=np.int64)
        )
        _draw_patch_surface(
            ax,
            base_idx,
            i,
            j,
            color="#aeb6bf",
            max_points=1600,
            face_alpha=0.10,
            mesh_alpha=0.10,
            boundary_alpha=0.38,
            point_alpha=0.12,
            boundary_width=0.45,
            zorder=1,
        )

        if frag_np is None or not (frag_np > 0).any():
            ax.text(
                0.5, 0.5, "no detached fragments",
                transform=ax.transAxes,
                ha="center", va="center", color="#555555", fontsize=10,
            )
            return

        labels = [int(label) for label in np.unique(frag_np) if int(label) > 0]
        cmap = plt.get_cmap("tab20")
        for label_idx, label in enumerate(labels):
            idx = np.where(frag_np == label)[0]
            _draw_patch_surface(
                ax,
                idx,
                i,
                j,
                color=cmap(label_idx % cmap.N),
                max_points=900,
                face_alpha=0.22,
                mesh_alpha=0.20,
                boundary_alpha=0.10,
                point_alpha=0.62,
                boundary_width=0.35,
                zorder=4,
            )

    def _masked_knn_segments(
        mask_np: np.ndarray | None,
        i: int,
        j: int,
        *,
        max_edges: int,
    ) -> np.ndarray:
        if knn_np is None or mask_np is None:
            return np.empty((0, 2, 2), dtype=np.float32)
        n = min(knn_np.shape[0], mask_np.shape[0], pos.shape[0])
        if n <= 0:
            return np.empty((0, 2, 2), dtype=np.float32)
        src = np.repeat(np.arange(n, dtype=np.int64), knn_np.shape[1])
        dst = knn_np[:n].reshape(-1)
        valid = (
            mask_np[:n].reshape(-1)
            & (dst >= 0)
            & (dst < pos.shape[0])
            & (src != dst)
        )
        if not np.any(valid):
            return np.empty((0, 2, 2), dtype=np.float32)
        a = np.minimum(src[valid], dst[valid])
        b = np.maximum(src[valid], dst[valid])
        pairs = np.unique(np.stack([a, b], axis=1), axis=0)
        if pairs.shape[0] > max_edges:
            keep = np.linspace(0, pairs.shape[0] - 1, max_edges).astype(np.int64)
            pairs = pairs[keep]
        return np.stack(
            [
                np.stack([pos[pairs[:, 0], i], pos[pairs[:, 0], j]], axis=1),
                np.stack([pos[pairs[:, 1], i], pos[pairs[:, 1], j]], axis=1),
            ],
            axis=1,
        ).astype(np.float32)

    def _draw_causal_edges(ax, i: int, j: int) -> None:
        edge_specs = [
            (cut_edge_np, "#1f5a99", 0.42, 0.38, 4),
            (closure_edge_np, "#2d8f6f", 0.58, 0.56, 5),
            (detached_edge_np, "#6f4ca5", 0.62, 0.48, 6),
        ]
        for mask, color, width, alpha, zorder in edge_specs:
            segments = _masked_knn_segments(mask, i, j, max_edges=2600)
            if segments.shape[0] == 0:
                continue
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=[color],
                    linewidths=width,
                    alpha=alpha,
                    zorder=zorder,
                )
            )

    for col, (view_name, i, j, xlabel, ylabel) in enumerate(views):
        ax = axes[0, col]
        if not _draw_mesh_background(ax, i, j, alpha=0.10, zorder=0):
            ax.scatter(pos[:, i], pos[:, j], s=0.12, c="#d5d8dc", alpha=0.16, linewidths=0)
        if segment_np is not None:
            child_idx, parent_idx, step_np = segment_np
            max_segments = 1600
            if child_idx.shape[0] > max_segments:
                keep = np.linspace(0, child_idx.shape[0] - 1, max_segments).astype(np.int64)
                child_idx = child_idx[keep]
                parent_idx = parent_idx[keep]
                step_view = step_np[keep] if step_np is not None else None
            else:
                step_view = step_np
            segments = np.stack(
                [
                    np.stack([pos[parent_idx, i], pos[parent_idx, j]], axis=1),
                    np.stack([pos[child_idx, i], pos[child_idx, j]], axis=1),
                ],
                axis=1,
            )
            if step_view is not None and step_view.size > 0 and step_view.max() > step_view.min():
                norm = matplotlib.colors.Normalize(vmin=float(step_view.min()), vmax=float(step_view.max()))
                collection = LineCollection(
                    segments,
                    cmap="viridis",
                    norm=norm,
                    linewidths=0.20,
                    alpha=0.16,
                )
                collection.set_array(step_view.astype(np.float32))
            else:
                collection = LineCollection(segments, colors="#0b4f9c", linewidths=0.18, alpha=0.14)
            ax.add_collection(collection)
        _draw_causal_edges(ax, i, j)
        if cracked_np.any():
            ax.scatter(pos[cracked_np, i], pos[cracked_np, j], s=0.42, c=c[cracked_np],
                       cmap="inferno", alpha=0.18, linewidths=0, zorder=3)
        if tips_np.any():
            ax.scatter(pos[tips_np, i], pos[tips_np, j], s=15, c="#00d7ff",
                       marker="x", linewidths=0.75)
        _setup_axis(ax, f"Causal crack graph | {view_name}", xlabel, ylabel)

    for col, (view_name, i, j, xlabel, ylabel) in enumerate(views):
        ax = axes[1, col]
        if not _plot_actual_fragment_mesh(ax, i, j):
            ax.scatter(pos[:, i], pos[:, j], s=0.12, c="#d5d8dc", alpha=0.14, linewidths=0)
            _plot_fragment_surface(ax, i, j)
        _setup_axis(ax, f"Fragment surface | {view_name}", xlabel, ylabel)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

