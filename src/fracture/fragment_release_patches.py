"""Open-crack release patch extraction for graph fragment detection."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .graph_builder import GaussianGraph


class FragmentReleasePatchMixin:
    """Extract crack-connected open-release patches."""

    def _open_release_params(self) -> dict:
        if self.material_family == "sharp_brittle":
            params = {
                "seed_threshold": 0.52,
                "release_threshold": 0.54,
                "candidate_score": 0.10,
                "radius_scale": 0.040,
                "seed_radius_scale": 0.014,
                "min_edges": max(96, 6 * self.min_boundary_edges),
                "min_contact_edges": max(24, 2 * self.min_boundary_edges),
                "min_size": max(8, self.support_promote_min_size),
                "max_size_ratio": 0.026,
                "max_patches": self.open_crack_release_max_patches,
            }
        elif self.material_family == "brittle_moderate":
            params = {
                "seed_threshold": 0.50,
                "release_threshold": 0.56,
                "candidate_score": 0.09,
                "radius_scale": 0.047,
                "seed_radius_scale": 0.018,
                "min_edges": max(112, 6 * self.min_boundary_edges),
                "min_contact_edges": max(28, 2 * self.min_boundary_edges),
                "min_size": max(14, self.support_promote_min_size),
                "max_size_ratio": 0.035,
                "max_patches": self.open_crack_release_max_patches,
            }
        elif self.material_family == "rough_quasi_brittle":
            params = {
                "seed_threshold": 0.45,
                "release_threshold": 0.50,
                "candidate_score": 0.075,
                "radius_scale": 0.060,
                "seed_radius_scale": 0.024,
                "min_edges": max(128, 5 * self.min_boundary_edges),
                "min_contact_edges": max(32, 2 * self.min_boundary_edges),
                "min_size": max(24, self.support_promote_min_size),
                "max_size_ratio": 0.050,
                "max_patches": self.open_crack_release_max_patches,
            }
        else:
            params = {
                "seed_threshold": 0.58,
                "release_threshold": 0.64,
                "candidate_score": 0.12,
                "radius_scale": 0.034,
                "seed_radius_scale": 0.012,
                "min_edges": max(160, 8 * self.min_boundary_edges),
                "min_contact_edges": max(40, 2 * self.min_boundary_edges),
                "min_size": max(18, self.support_promote_min_size),
                "max_size_ratio": 0.020,
                "max_patches": 0,
            }

        style = self.crack_style
        if style == "radial_shatter" and self.material_family == "sharp_brittle":
            params.update({
                "seed_threshold": 0.28,
                "release_threshold": 0.29,
                "candidate_score": 0.035,
                "radius_scale": 0.060,
                "seed_radius_scale": 0.020,
                "min_edges": max(36, 3 * self.min_boundary_edges),
                "min_contact_edges": max(5, self.min_boundary_edges // 2),
                "min_size": max(3, min(self.support_promote_min_size, 4)),
                "max_size_ratio": 0.085,
                "max_patches": max(params["max_patches"], 20),
            })
        elif style == "spiderweb_branching" and self.material_family in {"sharp_brittle", "brittle_moderate"}:
            params.update({
                "seed_threshold": 0.43,
                "release_threshold": 0.47,
                "radius_scale": 0.050,
                "seed_radius_scale": 0.018,
                "min_edges": max(72, 4 * self.min_boundary_edges),
                "min_contact_edges": max(16, self.min_boundary_edges),
                "min_size": max(8, self.support_promote_min_size),
                "max_size_ratio": 0.040,
                "max_patches": max(params["max_patches"], 6),
            })
        elif style == "chunky_crumble" and self.material_family == "rough_quasi_brittle":
            params.update({
                "seed_threshold": 0.36,
                "release_threshold": 0.40,
                "candidate_score": 0.055,
                "radius_scale": 0.070,
                "seed_radius_scale": 0.030,
                "min_edges": max(72, 3 * self.min_boundary_edges),
                "min_contact_edges": max(16, self.min_boundary_edges),
                "min_size": max(12, self.support_promote_min_size),
                "max_size_ratio": 0.070,
                "max_patches": max(params["max_patches"], 8),
            })
        elif style == "single_smooth":
            params.update({
                "seed_threshold": 0.44,
                "release_threshold": 0.46,
                "candidate_score": 0.080,
                "radius_scale": max(params["radius_scale"], 0.060),
                "seed_radius_scale": max(params["seed_radius_scale"], 0.018),
                "min_contact_edges": max(12, self.min_boundary_edges),
                "max_size_ratio": max(params["max_size_ratio"], 0.070),
                "max_patches": min(max(params["max_patches"], 1), 2),
            })
        elif style == "diffuse_microcrack":
            params["max_patches"] = 0

        intensity = max(self.brittle_release_intensity, 1e-3)
        if intensity != 1.0 and params["max_patches"] > 0:
            loosen = min(max(intensity, 0.35), 3.00)
            params["seed_threshold"] = max(0.14, params["seed_threshold"] / (0.70 + 0.30 * loosen))
            params["release_threshold"] = max(0.14, params["release_threshold"] / (0.64 + 0.36 * loosen))
            params["radius_scale"] *= min(1.55, 0.86 + 0.14 * loosen)
            params["seed_radius_scale"] *= min(1.35, 0.92 + 0.08 * loosen)
            params["max_size_ratio"] *= min(1.70, 0.80 + 0.20 * loosen)
            params["max_patches"] = max(
                0,
                int(round(float(params["max_patches"]) * min(loosen, 1.70))),
            )

        return params

    def _extract_open_crack_release_patches(
        self,
        positions: Optional[Tensor],
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        damage: Tensor,
        opening: Optional[Tensor],
        active_tip_mask: Optional[Tensor],
        recent_front_mask: Optional[Tensor],
        used_mask: Optional[Tensor] = None,
    ) -> List[dict]:
        """Promote local crack-neighborhood collapse without requiring a closed ring.

        This is a surface-graph substitute for stiffness-loss collapse. A long,
        high-confidence crack corridor can release a small local patch even when
        the crack path has not topologically closed into a loop.
        """
        if (
            not self.open_crack_release_enable
            or self.material_family == "diffuse_damage"
            or positions is None
            or graph.knn_idx is None
            or corridor_edge_mask is None
        ):
            return []

        params = self._open_release_params()
        max_patches = int(params["max_patches"])
        if max_patches <= 0:
            return []

        edge_rows, edge_cols = torch.where(corridor_edge_mask)
        if edge_rows.numel() < int(params["min_edges"]):
            return []

        nbr_idx = graph.knn_idx[edge_rows, edge_cols]
        N = positions.shape[0]
        dtype = damage.dtype
        device = positions.device
        edge_count = torch.zeros(N, dtype=dtype, device=device)
        ones = torch.ones(edge_rows.shape[0], dtype=dtype, device=device)
        edge_count.index_add_(0, edge_rows, ones)
        edge_count.index_add_(0, nbr_idx, ones)
        edge_density = (edge_count / edge_count.max().clamp(min=1.0)).clamp(0.0, 1.0)
        cut_node_mask = edge_count > 0

        if opening is not None:
            opening_scale = torch.quantile(opening.detach(), 0.90).clamp(min=1e-8)
            opening_norm = (opening / opening_scale).clamp(0.0, 1.0)
        else:
            opening_norm = torch.zeros_like(damage)

        front_score = torch.zeros_like(damage)
        if recent_front_mask is not None:
            front_score = torch.maximum(front_score, recent_front_mask.float())
        if active_tip_mask is not None:
            front_score = torch.maximum(front_score, active_tip_mask.float())

        release_field = (
            0.48 * edge_density
            + 0.26 * damage.clamp(0.0, 1.0)
            + 0.18 * opening_norm
            + 0.08 * front_score
        ).clamp(0.0, 1.0)
        if self.crack_style == "radial_shatter" and self.material_family == "sharp_brittle":
            radial_release = (
                0.62 * edge_density
                + 0.25 * damage.clamp(0.0, 1.0)
                + 0.08 * opening_norm
                + 0.12 * front_score
            ).clamp(0.0, 1.0)
            release_field = torch.maximum(release_field, radial_release)

        if used_mask is None:
            used = torch.zeros(N, dtype=torch.bool, device=device)
        else:
            used = used_mask.clone()

        seed_threshold = (
            self.open_crack_release_threshold
            if self.open_crack_release_threshold > 0.0
            else float(params["seed_threshold"])
        )
        seed_candidates = cut_node_mask & (~used) & (release_field >= seed_threshold)
        if not bool(seed_candidates.any()):
            relaxed = seed_threshold * 0.86
            if float(release_field[cut_node_mask].max().item()) < relaxed:
                return []
            top_count = min(32, int(cut_node_mask.sum().item()))
            seed_idx = torch.where(cut_node_mask & (~used))[0]
            if seed_idx.numel() == 0:
                return []
            top_local = release_field[seed_idx].topk(min(top_count, seed_idx.numel())).indices
            seed_candidates = torch.zeros(N, dtype=torch.bool, device=device)
            seed_candidates[seed_idx[top_local]] = True

        ordered_seed_idx = torch.where(seed_candidates)[0]
        seed_order = release_field[ordered_seed_idx].argsort(descending=True)
        max_seed_trials = max(16, max_patches * 4)
        ordered_seed_idx = ordered_seed_idx[seed_order[:max_seed_trials]]

        bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
        diag = float(bbox_extent.norm().item())
        radius = max(0.012, float(params["radius_scale"]) * max(diag, 1e-6))
        seed_radius = max(0.006, float(params["seed_radius_scale"]) * max(diag, 1e-6))
        max_patch_size = max(
            int(params["min_size"]),
            int(round(float(params["max_size_ratio"]) * N)),
        )

        patches: List[dict] = []
        seed_patch_budget = max_patches
        if self.crack_style == "radial_shatter" and self.material_family == "sharp_brittle":
            seed_patch_budget = min(max_patches, 8)
        for seed in ordered_seed_idx.tolist():
            if len(patches) >= seed_patch_budget:
                break
            if bool(used[seed]):
                continue
            patch = self._build_open_crack_release_patch(
                seed_index=int(seed),
                positions=positions,
                graph=graph,
                corridor_edge_mask=corridor_edge_mask,
                release_field=release_field,
                cut_node_mask=cut_node_mask,
                used_mask=used,
                radius=radius,
                seed_radius=seed_radius,
                min_patch_size=int(params["min_size"]),
                max_patch_size=max_patch_size,
                min_contact_edges=int(params["min_contact_edges"]),
                release_threshold=float(params["release_threshold"]),
                candidate_score=float(params["candidate_score"]),
            )
            if patch is None:
                continue
            used |= patch["mask"]
            patches.append(patch)

        if (
            len(patches) < max_patches
            and self.crack_style == "radial_shatter"
            and self.material_family == "sharp_brittle"
        ):
            for patch in self._extract_radial_shatter_sector_patches(
                positions=positions,
                graph=graph,
                corridor_edge_mask=corridor_edge_mask,
                release_field=release_field,
                cut_node_mask=cut_node_mask,
                damage=damage,
                front_score=front_score,
                used_mask=used,
                max_patches=max_patches - len(patches),
                min_patch_size=int(params["min_size"]),
                max_patch_size=max_patch_size,
                min_contact_edges=int(params["min_contact_edges"]),
                release_threshold=float(params["release_threshold"]),
                candidate_score=float(params["candidate_score"]),
            ):
                used |= patch["mask"]
                patches.append(patch)

        self.last_open_release_patches = len(patches)
        self.last_open_release_nodes = int(sum(int(patch["size"]) for patch in patches))
        self.last_open_release_score_max = (
            max(float(patch.get("release_score", 0.0)) for patch in patches)
            if patches else 0.0
        )
        return patches

    def _extract_radial_shatter_sector_patches(
        self,
        positions: Tensor,
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        release_field: Tensor,
        cut_node_mask: Tensor,
        damage: Tensor,
        front_score: Tensor,
        used_mask: Tensor,
        max_patches: int,
        min_patch_size: int,
        max_patch_size: int,
        min_contact_edges: int,
        release_threshold: float,
        candidate_score: float,
    ) -> List[dict]:
        """Promote radial-shatter sectors when no clean closed ring exists."""
        if max_patches <= 0:
            return []

        gate = max(float(candidate_score), 0.50 * float(release_threshold), 0.10)
        candidate_mask = (
            (~used_mask)
            & (release_field >= gate)
            & (
                cut_node_mask
                | (damage >= max(0.12, 0.55 * float(release_threshold)))
                | (front_score > 0.0)
            )
        )
        if int(candidate_mask.sum().item()) < int(min_patch_size):
            return []

        center_mask = candidate_mask & cut_node_mask
        if not bool(center_mask.any()):
            center_mask = candidate_mask
        center = positions[center_mask].mean(dim=0)
        rel = positions - center.unsqueeze(0)
        theta = torch.atan2(rel[:, 1], rel[:, 0])
        theta01 = ((theta + math.pi) / (2.0 * math.pi)).clamp(0.0, 0.999999)

        planar_r = rel[:, :2].norm(dim=1)
        if bool(candidate_mask.any()):
            r_scale = torch.quantile(planar_r[candidate_mask].detach(), 0.88).clamp(min=1e-6)
        else:
            r_scale = planar_r.max().clamp(min=1e-6)
        radial_outer = planar_r >= 0.48 * r_scale

        sector_count = max(8, min(int(max_patches), 24))
        band_count = 2 if max_patches >= 14 else 1
        sector_idx = torch.floor(theta01 * sector_count).long().clamp(0, sector_count - 1)
        if band_count > 1:
            group_id = sector_idx + sector_count * radial_outer.long()
        else:
            group_id = sector_idx

        groups = group_id[candidate_mask].unique(sorted=False)
        patch_candidates = []
        for gid_t in groups.tolist():
            mask = candidate_mask & (group_id == int(gid_t))
            size = int(mask.sum().item())
            if size < int(min_patch_size):
                continue
            if size > int(max_patch_size):
                idx = torch.where(mask)[0]
                top = release_field[idx].topk(int(max_patch_size)).indices
                keep = idx[top]
                mask = torch.zeros_like(mask)
                mask[keep] = True
                size = int(mask.sum().item())

            patch_side = mask.unsqueeze(1)
            neighbor_side = mask[graph.knn_idx]
            cross_boundary = patch_side ^ neighbor_side
            patch_boundary_mask = corridor_edge_mask & cross_boundary
            contact_mask = corridor_edge_mask & (patch_side | neighbor_side)
            boundary_edges = int(patch_boundary_mask.sum().item())
            contact_edges = int(contact_mask.sum().item())
            mean_score = float(release_field[mask].mean().item())
            peak_score = float(release_field[mask].max().item())
            weak_contact_ok = (
                max(boundary_edges, contact_edges) >= max(2, int(min_contact_edges) // 2)
                or mean_score >= float(release_threshold) + 0.04
            )
            if not weak_contact_ok:
                continue

            contact_score = min(1.0, contact_edges / max(float(min_contact_edges * 2), 1.0))
            boundary_score = min(1.0, boundary_edges / max(float(min_contact_edges * 2), 1.0))
            release_score = (
                0.45 * peak_score
                + 0.25 * mean_score
                + 0.20 * contact_score
                + 0.10 * boundary_score
            )
            if release_score < max(0.14, 0.86 * float(release_threshold)):
                continue

            boundary_mask = patch_boundary_mask if boundary_edges > 0 else contact_mask
            patch_candidates.append({
                "mask": mask,
                "boundary_mask": boundary_mask,
                "size": size,
                "seed_size": size,
                "cut_ratio": contact_score,
                "center": positions[mask].mean(dim=0),
                "seed_center": positions[mask].mean(dim=0),
                "plane_normal": torch.zeros(3, dtype=positions.dtype, device=positions.device),
                "closure_score": 0.0,
                "boundary_score": boundary_score,
                "release_score": float(min(1.0, release_score)),
                "support_lost": True,
                "open_release": True,
                "sector_release": True,
            })

        patch_candidates.sort(
            key=lambda item: (
                float(item.get("release_score", 0.0)),
                int(item.get("size", 0)),
            ),
            reverse=True,
        )
        return patch_candidates[:max_patches]

    def _build_open_crack_release_patch(
        self,
        seed_index: int,
        positions: Tensor,
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        release_field: Tensor,
        cut_node_mask: Tensor,
        used_mask: Tensor,
        radius: float,
        seed_radius: float,
        min_patch_size: int,
        max_patch_size: int,
        min_contact_edges: int,
        release_threshold: float,
        candidate_score: float,
    ) -> Optional[dict]:
        seed_pos = positions[seed_index]
        dist = torch.norm(positions - seed_pos.unsqueeze(0), dim=1)
        candidate_mask = (
            (dist <= radius)
            & (~used_mask)
            & ((release_field >= candidate_score) | cut_node_mask)
        )
        seed_mask = (
            (dist <= seed_radius)
            & cut_node_mask
            & (~used_mask)
            & (release_field >= max(0.10, 0.72 * release_threshold))
        )
        seed_mask[seed_index] = True
        seed_size = int(seed_mask.sum().item())
        if seed_size <= 0:
            return None

        patch_mask = seed_mask.clone()
        allowed_edge = (
            (~corridor_edge_mask)
            & candidate_mask.unsqueeze(1)
            & candidate_mask[graph.knn_idx]
        )
        for _ in range(5):
            row_in = patch_mask.unsqueeze(1).expand_as(graph.knn_idx)
            col_in = patch_mask[graph.knn_idx]
            touch = allowed_edge & (row_in | col_in)
            if not bool(touch.any()):
                break
            expanded = patch_mask.clone()
            expanded |= torch.any(touch, dim=1)
            expanded[graph.knn_idx[touch]] = True
            expanded &= candidate_mask
            if int(expanded.sum().item()) > max_patch_size:
                patch_idx = torch.where(expanded)[0]
                patch_dist = dist[patch_idx]
                keep = patch_idx[patch_dist.argsort()[:max_patch_size]]
                expanded.zero_()
                expanded[keep] = True
                expanded[seed_index] = True
                patch_mask = expanded
                break
            delta = expanded & (~patch_mask)
            patch_mask = expanded
            if not bool(delta.any()):
                break

        patch_size = int(patch_mask.sum().item())
        if patch_size < min_patch_size:
            return None

        patch_side = patch_mask.unsqueeze(1)
        neighbor_side = patch_mask[graph.knn_idx]
        cross_boundary = patch_side ^ neighbor_side
        patch_boundary_mask = corridor_edge_mask & cross_boundary
        contact_mask = corridor_edge_mask & (patch_side | neighbor_side)
        boundary_edges = int(patch_boundary_mask.sum().item())
        contact_edges = int(contact_mask.sum().item())
        if max(boundary_edges, contact_edges) < min_contact_edges:
            return None

        patch_score = float(release_field[patch_mask].mean().item())
        peak_score = float(release_field[patch_mask].max().item())
        contact_score = min(1.0, contact_edges / max(float(min_contact_edges * 3), 1.0))
        boundary_score = min(1.0, boundary_edges / max(float(min_contact_edges * 2), 1.0))
        release_score = (
            0.46 * peak_score
            + 0.26 * patch_score
            + 0.18 * contact_score
            + 0.10 * boundary_score
        )
        if release_score < release_threshold:
            return None

        boundary_mask = patch_boundary_mask if boundary_edges > 0 else contact_mask
        return {
            "mask": patch_mask,
            "boundary_mask": boundary_mask,
            "size": patch_size,
            "seed_size": seed_size,
            "cut_ratio": contact_score,
            "center": positions[patch_mask].mean(dim=0),
            "seed_center": seed_pos,
            "plane_normal": torch.zeros(3, dtype=positions.dtype, device=positions.device),
            "closure_score": 0.0,
            "boundary_score": boundary_score,
            "release_score": float(min(1.0, release_score)),
            "support_lost": True,
            "open_release": True,
        }

