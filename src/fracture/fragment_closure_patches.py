"""Closure patch extraction for graph fragment detection."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .graph_builder import GaussianGraph


class FragmentClosurePatchMixin:
    """Extract crack-connected closure/ring/cascade patch candidates."""

    def _closure_params(self) -> Tuple[int, float, float]:
        if self.crack_connected_release_only:
            if self.material_family == "sharp_brittle":
                return 14, 0.18, 0.42
            if self.material_family == "brittle_moderate":
                return 16, 0.18, 0.46
            if self.material_family == "rough_quasi_brittle":
                return 14, 0.14, 0.48
            return 16, 0.18, 0.50
        if self.material_family == "sharp_brittle":
            return 18, 0.22, 0.58
        if self.material_family == "brittle_moderate":
            return 20, 0.20, 0.60
        if self.material_family == "rough_quasi_brittle":
            return 16, 0.16, 0.52
        return 16, 0.18, 0.58

    def _explicit_closure_params(self) -> Tuple[float, int, float, float]:
        if self.crack_connected_release_only:
            if self.material_family == "sharp_brittle":
                return 0.42, max(10, min(self.min_fragment_size, 18)), 1.16, 0.060
            if self.material_family == "brittle_moderate":
                return 0.48, max(18, min(self.min_fragment_size, 28)), 1.12, 0.060
            if self.material_family == "rough_quasi_brittle":
                return 0.50, max(24, min(self.min_fragment_size, 40)), 1.10, 0.070
            return 0.50, max(20, self.min_fragment_size), 1.10, 0.060
        if self.material_family == "sharp_brittle":
            return 0.68, max(28, self.min_fragment_size), 1.04, 0.040
        if self.material_family == "brittle_moderate":
            return 0.72, max(40, self.min_fragment_size), 1.08, 0.050
        if self.material_family == "rough_quasi_brittle":
            return 0.70, max(72, 2 * self.min_fragment_size), 1.12, 0.060
        return 0.72, max(48, self.min_fragment_size), 1.08, 0.050

    def _strict_closure_release_cap(self) -> float:
        cap = max(0.0, min(float(self.strict_closure_max_released_ratio), 1.0))
        if not self.crack_connected_release_only:
            return 1.0
        if self.material_family == "sharp_brittle":
            return max(0.08, min(cap, 0.68))
        if self.material_family == "brittle_moderate":
            return max(0.06, min(cap, 0.46))
        if self.material_family == "rough_quasi_brittle":
            return max(0.05, min(cap, 0.42))
        return max(0.04, min(cap, 0.35))

    def _strict_patch_required_cut_ratio(self) -> float:
        """Minimum crack/cut support for a strict closure fragment boundary."""
        if not self.crack_connected_release_only:
            return max(0.32, 0.85 * self.fallback_cut_ratio)
        if self.material_family == "sharp_brittle":
            return max(0.70, 0.98 * self.fallback_cut_ratio)
        if self.material_family == "rough_quasi_brittle":
            return max(0.55, 0.98 * self.fallback_cut_ratio)
        return max(0.62, 0.98 * self.fallback_cut_ratio)

    def _compute_group_closure_scores(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        component_group_map: dict,
        group_ids: set,
        excluded_mask: Optional[Tensor] = None,
    ) -> dict:
        if positions is None or graph.knn_idx is None or not group_ids:
            return {}

        angle_bins, compactness_target, _ = self._closure_params()
        two_pi = float(2.0 * torch.pi)
        scores = {}
        excluded = (
            excluded_mask.to(device=labels.device, dtype=torch.bool)
            if excluded_mask is not None
            else None
        )

        for group_id in sorted(group_ids):
            in_group = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
            group_size = 0
            for old_label in labels.unique().tolist():
                if component_group_map.get(old_label, old_label) != group_id:
                    continue
                mask = labels == old_label
                if excluded is not None:
                    mask = mask & (~excluded)
                in_group |= mask
                group_size += int(mask.sum().item())
            if group_size <= 0:
                scores[group_id] = 0.0
                continue

            boundary_mask = corridor_edge_mask & (in_group.unsqueeze(1) ^ in_group[graph.knn_idx])
            edge_rows, edge_cols = torch.where(boundary_mask)
            if edge_rows.numel() < max(self.min_boundary_edges, 8):
                scores[group_id] = 0.0
                continue

            nbr_idx = graph.knn_idx[edge_rows, edge_cols]
            mids = 0.5 * (positions[edge_rows] + positions[nbr_idx])
            center = positions[in_group].mean(dim=0, keepdim=True)
            centered = mids - center
            if centered.shape[0] < 6:
                scores[group_id] = 0.0
                continue

            cov = centered.T @ centered / float(max(centered.shape[0] - 1, 1))
            try:
                _, eigvecs = torch.linalg.eigh(cov)
            except RuntimeError:
                scores[group_id] = 0.0
                continue
            basis = eigvecs[:, -2:]
            uv = centered @ basis
            radii = uv.norm(dim=1)
            mean_radius = float(radii.mean().item())
            if mean_radius < 1e-6:
                scores[group_id] = 0.0
                continue

            valid = radii > (0.15 * mean_radius)
            if int(valid.sum().item()) < 6:
                scores[group_id] = 0.0
                continue
            uv = uv[valid]
            radii = radii[valid]
            angles = torch.atan2(uv[:, 1], uv[:, 0])

            bin_pos = ((angles + torch.pi) / (2.0 * torch.pi) * angle_bins).floor().long()
            bin_pos = bin_pos.clamp(0, angle_bins - 1)
            occupied = torch.bincount(bin_pos, minlength=angle_bins) > 0
            angular_coverage = float(occupied.float().mean().item())

            sorted_angles = torch.sort(angles).values
            wrapped = torch.cat([sorted_angles, sorted_angles[:1] + two_pi])
            gaps = wrapped[1:] - wrapped[:-1]
            max_gap = float(gaps.max().item()) if gaps.numel() > 0 else two_pi
            gap_score = max(0.0, min(1.0, 1.0 - max_gap / two_pi))

            compactness = float(group_size) / float(max(int(edge_rows.numel()), 1))
            compactness_score = max(0.0, min(1.0, compactness / max(compactness_target, 1e-6)))

            radial_cv = float(radii.std(unbiased=False).item()) / max(float(radii.mean().item()), 1e-6)
            radial_score = max(0.0, min(1.0, 1.0 - radial_cv / 0.85))

            closure_score = (
                0.42 * angular_coverage
                + 0.33 * gap_score
                + 0.17 * compactness_score
                + 0.08 * radial_score
            )
            scores[group_id] = float(max(0.0, min(1.0, closure_score)))

        return scores

    def _build_explicit_patch_from_group(
        self,
        labels: Tensor,
        positions: Tensor,
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        members: List[int],
        min_patch_size: int,
        inflate_scale: float,
        slab_scale: float,
        excluded_mask: Optional[Tensor] = None,
    ) -> Optional[dict]:
        if graph.knn_idx is None or positions is None or not members:
            return None

        seed_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        excluded = (
            excluded_mask.to(device=labels.device, dtype=torch.bool)
            if excluded_mask is not None
            else torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        )
        for old_label in members:
            seed_mask |= (labels == old_label)
        seed_mask &= ~excluded
        seed_size = int(seed_mask.sum().item())
        if self.crack_connected_release_only:
            seed_min = 4 if self.material_family == "sharp_brittle" else 6
        else:
            seed_min = max(self.min_boundary_edges, 8)
        if seed_size < seed_min:
            return None

        active_pair = ~(excluded.unsqueeze(1) | excluded[graph.knn_idx])
        boundary_mask = (
            corridor_edge_mask
            & active_pair
            & (seed_mask.unsqueeze(1) ^ seed_mask[graph.knn_idx])
        )
        edge_rows, edge_cols = torch.where(boundary_mask)
        if edge_rows.numel() < max(self.min_boundary_edges, 10):
            return None

        nbr_idx = graph.knn_idx[edge_rows, edge_cols]
        boundary_mids = 0.5 * (positions[edge_rows] + positions[nbr_idx])
        boundary_nodes = torch.unique(torch.cat([edge_rows, nbr_idx], dim=0))
        if boundary_mids.shape[0] < 8:
            return None

        center = boundary_mids.mean(dim=0)
        centered = boundary_mids - center.unsqueeze(0)
        cov = centered.T @ centered / float(max(centered.shape[0] - 1, 1))
        try:
            _, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError:
            return None
        basis = eigvecs[:, -2:]
        plane_normal = eigvecs[:, 0]
        uv_boundary = centered @ basis
        radii = uv_boundary.norm(dim=1)
        mean_radius = float(radii.mean().item())
        if mean_radius < 1e-5:
            return None

        angle_bins = max(self._closure_params()[0] * 2, 24)
        angles = torch.atan2(uv_boundary[:, 1], uv_boundary[:, 0])
        bin_pos = ((angles + torch.pi) / (2.0 * torch.pi) * angle_bins).floor().long()
        bin_pos = bin_pos.clamp(0, angle_bins - 1)

        radius_bins = torch.zeros(angle_bins, dtype=positions.dtype, device=positions.device)
        occupied = torch.zeros(angle_bins, dtype=torch.bool, device=positions.device)
        for b in range(angle_bins):
            mask_b = bin_pos == b
            if bool(mask_b.any()):
                radius_bins[b] = radii[mask_b].max()
                occupied[b] = True
        if int(occupied.sum().item()) < max(angle_bins // 3, 8):
            return None

        filled_bins = radius_bins.clone()
        occ_idx = torch.where(occupied)[0]
        for b in range(angle_bins):
            if occupied[b]:
                continue
            circular_dist = torch.minimum(
                (occ_idx - b).abs(),
                angle_bins - (occ_idx - b).abs(),
            )
            nearest = occ_idx[int(torch.argmin(circular_dist).item())]
            filled_bins[b] = radius_bins[nearest]
        for _ in range(2):
            filled_bins = torch.maximum(
                filled_bins,
                0.5 * (torch.roll(filled_bins, 1) + torch.roll(filled_bins, -1)),
            )
        filled_bins = inflate_scale * filled_bins

        seed_plane_dist = ((positions[seed_mask] - center.unsqueeze(0)) @ plane_normal).abs()
        slab = max(
            float(torch.quantile(seed_plane_dist, 0.90).item()) * 1.8,
            slab_scale,
        )
        uv_all = (positions - center.unsqueeze(0)) @ basis
        node_radii = uv_all.norm(dim=1)
        node_angles = torch.atan2(uv_all[:, 1], uv_all[:, 0])
        node_bins = ((node_angles + torch.pi) / (2.0 * torch.pi) * angle_bins).floor().long()
        node_bins = node_bins.clamp(0, angle_bins - 1)
        radius_limit = filled_bins[node_bins]
        plane_dist = ((positions - center.unsqueeze(0)) @ plane_normal).abs()
        inside_proj = (
            (node_radii <= radius_limit.clamp(min=1e-6))
            & (plane_dist <= slab)
        )
        candidate_mask = (inside_proj | seed_mask) & (~excluded)
        candidate_mask[boundary_nodes] = True

        patch_mask = seed_mask.clone()
        allowed_edge = (~corridor_edge_mask) & candidate_mask.unsqueeze(1) & candidate_mask[graph.knn_idx]
        for _ in range(32):
            row_in = patch_mask.unsqueeze(1).expand_as(graph.knn_idx)
            col_in = patch_mask[graph.knn_idx]
            touch = allowed_edge & (row_in | col_in)
            if not bool(touch.any()):
                break
            expanded = patch_mask.clone()
            expanded |= torch.any(touch, dim=1)
            expanded[graph.knn_idx[touch]] = True
            delta = expanded & (~patch_mask)
            patch_mask = expanded
            if not bool(delta.any()):
                break

        patch_size = int(patch_mask.sum().item())
        if patch_size < min_patch_size or patch_size <= seed_size:
            return None

        cross_boundary = (
            (patch_mask.unsqueeze(1) ^ patch_mask[graph.knn_idx])
            & active_pair
        )
        boundary_edges = int(cross_boundary.sum().item())
        if boundary_edges < max(self.min_boundary_edges, 12):
            return None
        patch_boundary_mask = corridor_edge_mask & cross_boundary
        patch_cut_ratio = float(patch_boundary_mask[cross_boundary].float().mean().item()) if bool(cross_boundary.any()) else 0.0
        required_cut_ratio = self._strict_patch_required_cut_ratio()
        if patch_cut_ratio < required_cut_ratio:
            return None

        patch_center = positions[patch_mask].mean(dim=0)
        seed_center = positions[seed_mask].mean(dim=0)
        return {
            "mask": patch_mask,
            "boundary_mask": patch_boundary_mask,
            "size": patch_size,
            "seed_size": seed_size,
            "cut_ratio": patch_cut_ratio,
            "center": patch_center,
            "seed_center": seed_center,
            "plane_normal": plane_normal,
        }

    @staticmethod
    def _combined_patch_mask(
        patches: List[dict],
        node_count: int,
        device: torch.device,
        excluded_mask: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if not patches and excluded_mask is None:
            return None
        mask = torch.zeros(node_count, dtype=torch.bool, device=device)
        if excluded_mask is not None:
            mask |= excluded_mask.to(device=device, dtype=torch.bool)
        for patch in patches:
            mask |= patch["mask"]
        return mask

    def _collect_fragment_patches(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        damage: Tensor,
        opening: Optional[Tensor],
        active_tip_mask: Optional[Tensor],
        recent_front_mask: Optional[Tensor],
        grouped_boundary_labels: set,
        grouped_boundary_scores: dict,
        group_closure_scores: dict,
        members_by_group: dict,
        excluded_mask: Optional[Tensor] = None,
    ) -> List[dict]:
        patches = self._extract_explicit_closure_patches(
            labels=labels,
            positions=positions,
            graph=graph,
            corridor_edge_mask=corridor_edge_mask,
            grouped_boundary_labels=grouped_boundary_labels,
            grouped_boundary_scores=grouped_boundary_scores,
            group_closure_scores=group_closure_scores,
            members_by_group=members_by_group,
            excluded_mask=excluded_mask,
        )

        if self.crack_connected_release_only:
            patches.extend(
                self._extract_strict_closure_local_patches(
                    labels=labels,
                    positions=positions,
                    graph=graph,
                    corridor_edge_mask=corridor_edge_mask,
                    damage=damage,
                    opening=opening,
                    active_tip_mask=active_tip_mask,
                    recent_front_mask=recent_front_mask,
                    grouped_boundary_labels=grouped_boundary_labels,
                    grouped_boundary_scores=grouped_boundary_scores,
                    group_closure_scores=group_closure_scores,
                    members_by_group=members_by_group,
                    used_mask=self._combined_patch_mask(
                        patches,
                        labels.shape[0],
                        labels.device,
                        excluded_mask=excluded_mask,
                    ),
                )
            )
            patches.extend(
                self._extract_strict_ring_closure_patches(
                    positions=positions,
                    graph=graph,
                    corridor_edge_mask=corridor_edge_mask,
                    damage=damage,
                    opening=opening,
                    active_tip_mask=active_tip_mask,
                    recent_front_mask=recent_front_mask,
                    used_mask=self._combined_patch_mask(
                        patches,
                        labels.shape[0],
                        labels.device,
                        excluded_mask=excluded_mask,
                    ),
                )
            )
            patches.extend(
                self._extract_strict_crack_cascade_patches(
                    positions=positions,
                    graph=graph,
                    corridor_edge_mask=corridor_edge_mask,
                    damage=damage,
                    opening=opening,
                    active_tip_mask=active_tip_mask,
                    recent_front_mask=recent_front_mask,
                    group_closure_scores=group_closure_scores,
                    used_mask=self._combined_patch_mask(
                        patches,
                        labels.shape[0],
                        labels.device,
                        excluded_mask=excluded_mask,
                    ),
                    has_seed_patch=bool(patches),
                )
            )
            patches.extend(
                self._extract_strict_impact_closure_patches(
                    positions=positions,
                    graph=graph,
                    corridor_edge_mask=corridor_edge_mask,
                    damage=damage,
                    opening=opening,
                    active_tip_mask=active_tip_mask,
                    recent_front_mask=recent_front_mask,
                    used_mask=self._combined_patch_mask(
                        patches,
                        labels.shape[0],
                        labels.device,
                        excluded_mask=excluded_mask,
                    ),
                )
            )
            return patches

        patches.extend(
            self._extract_open_crack_release_patches(
                positions=positions,
                graph=graph,
                corridor_edge_mask=corridor_edge_mask,
                damage=damage,
                opening=opening,
                active_tip_mask=active_tip_mask,
                recent_front_mask=recent_front_mask,
                used_mask=self._combined_patch_mask(
                    patches,
                    labels.shape[0],
                    labels.device,
                    excluded_mask=excluded_mask,
                ),
            )
        )
        return patches

    def _extract_explicit_closure_patches(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        grouped_boundary_labels: set,
        grouped_boundary_scores: dict,
        group_closure_scores: dict,
        members_by_group: dict,
        excluded_mask: Optional[Tensor] = None,
    ) -> List[dict]:
        if positions is None or graph.knn_idx is None or not grouped_boundary_labels:
            return []

        closure_threshold, min_patch_size, inflate_scale, slab_scale = self._explicit_closure_params()
        patches: List[dict] = []
        used_mask = (
            excluded_mask.to(device=labels.device, dtype=torch.bool).clone()
            if excluded_mask is not None
            else torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        )
        ordered_groups = sorted(
            grouped_boundary_labels,
            key=lambda gid: (
                float(group_closure_scores.get(gid, 0.0)),
                float(grouped_boundary_scores.get(gid, 0.0)),
            ),
            reverse=True,
        )
        for group_id in ordered_groups:
            closure_score = float(group_closure_scores.get(group_id, 0.0))
            boundary_score = float(grouped_boundary_scores.get(group_id, 0.0))
            if closure_score < closure_threshold:
                continue
            if boundary_score < max(self.fallback_cut_ratio, 0.30):
                continue
            members = members_by_group.get(group_id, [])
            patch = self._build_explicit_patch_from_group(
                labels=labels,
                positions=positions,
                graph=graph,
                corridor_edge_mask=corridor_edge_mask,
                members=members,
                min_patch_size=min_patch_size,
                inflate_scale=inflate_scale,
                slab_scale=slab_scale,
                excluded_mask=excluded_mask,
            )
            if patch is None:
                continue
            patch_mask = patch["mask"] & (~used_mask)
            patch_size = int(patch_mask.sum().item())
            if patch_size < min_patch_size:
                continue
            patch["mask"] = patch_mask
            patch["group_id"] = group_id
            patch["closure_score"] = closure_score
            patch["boundary_score"] = boundary_score
            patch["release_score"] = float(min(1.0, max(boundary_score, 0.52 + 0.42 * closure_score)))
            patch["support_lost"] = bool(
                self.material_family == "rough_quasi_brittle"
                or closure_score >= 0.88
            )
            used_mask |= patch_mask
            patches.append(patch)

        return patches

    def _extract_strict_closure_local_patches(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        damage: Tensor,
        opening: Optional[Tensor],
        active_tip_mask: Optional[Tensor],
        recent_front_mask: Optional[Tensor],
        grouped_boundary_labels: set,
        grouped_boundary_scores: dict,
        group_closure_scores: dict,
        members_by_group: dict,
        used_mask: Optional[Tensor] = None,
    ) -> List[dict]:
        if (
            not self.crack_connected_release_only
            or positions is None
            or graph.knn_idx is None
            or not grouped_boundary_labels
        ):
            return []

        closure_threshold, min_patch_size, _, slab_scale = self._explicit_closure_params()
        threshold = max(0.32, 0.82 * closure_threshold)
        bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
        diag = float(bbox_extent.norm().item())
        radius = max(0.030, 0.038 * max(diag, 1e-6))
        max_size = max(min_patch_size, int(round(0.055 * labels.shape[0])))
        cut_degree = corridor_edge_mask.float().sum(dim=1)
        cut_node = cut_degree >= max(2.0, float(self.min_boundary_edges) / 5.0)
        crack_node = damage > max(0.10, 0.45 * self.damage_threshold)
        if opening is not None:
            crack_node |= opening > max(0.015, 0.25 * float(opening.max().item()))
        if active_tip_mask is not None:
            crack_node |= active_tip_mask.bool()
        if recent_front_mask is not None:
            crack_node |= recent_front_mask.bool()

        occupied = (
            used_mask.clone()
            if used_mask is not None
            else torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        )
        patches: List[dict] = []
        ordered_groups = sorted(
            grouped_boundary_labels,
            key=lambda gid: (
                float(group_closure_scores.get(gid, 0.0)),
                float(grouped_boundary_scores.get(gid, 0.0)),
            ),
            reverse=True,
        )
        max_patches = 7 if self.material_family == "sharp_brittle" else 5
        for group_id in ordered_groups:
            if len(patches) >= max_patches:
                break
            closure_score = float(group_closure_scores.get(group_id, 0.0))
            boundary_score = float(grouped_boundary_scores.get(group_id, 0.0))
            if closure_score < threshold:
                continue
            seed_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
            for old_label in members_by_group.get(group_id, []):
                seed_mask |= (labels == old_label)
            if not bool(seed_mask.any()):
                continue
            seed_mask &= crack_node | cut_node
            seed_mask &= ~occupied
            if not bool(seed_mask.any()):
                continue

            center = positions[seed_mask].mean(dim=0)
            local_radius = radius * (1.0 + 0.55 * min(max(closure_score, 0.0), 1.0))
            rel = positions - center.unsqueeze(0)
            slab = max(slab_scale, 0.035)
            patch_mask = (rel.norm(dim=1) <= local_radius) & (rel[:, 2].abs() <= slab)
            patch_mask |= seed_mask
            patch_mask &= ~occupied
            patch_size = int(patch_mask.sum().item())
            if patch_size < min_patch_size:
                continue
            if patch_size > max_size:
                dist = rel.norm(dim=1)
                order = torch.argsort(dist)
                trimmed = torch.zeros_like(patch_mask)
                selected = order[patch_mask[order]][:max_size]
                trimmed[selected] = True
                patch_mask = trimmed | seed_mask
                patch_mask &= ~occupied
                patch_size = int(patch_mask.sum().item())
                if patch_size < min_patch_size:
                    continue

            active_pair = ~(occupied.unsqueeze(1) | occupied[graph.knn_idx])
            cross_boundary = (
                (patch_mask.unsqueeze(1) ^ patch_mask[graph.knn_idx])
                & active_pair
            )
            boundary_edges = int(cross_boundary.sum().item())
            if boundary_edges < max(self.min_boundary_edges, 8):
                continue
            patch_boundary_mask = corridor_edge_mask & cross_boundary
            patch_cut_ratio = (
                float(patch_boundary_mask[cross_boundary].float().mean().item())
                if bool(cross_boundary.any()) else 0.0
            )
            if patch_cut_ratio < self._strict_patch_required_cut_ratio():
                continue
            patch_crack_ratio = float((patch_mask & crack_node).float().sum().item()) / max(patch_size, 1)
            if patch_crack_ratio < 0.08:
                continue

            occupied |= patch_mask
            patches.append({
                "mask": patch_mask,
                "boundary_mask": patch_boundary_mask,
                "size": patch_size,
                "seed_size": int(seed_mask.sum().item()),
                "cut_ratio": patch_cut_ratio,
                "center": positions[patch_mask].mean(dim=0),
                "seed_center": center,
                "plane_normal": torch.tensor([0.0, 0.0, 1.0], device=positions.device, dtype=positions.dtype),
                "group_id": group_id,
                "closure_score": closure_score,
                "boundary_score": max(boundary_score, patch_cut_ratio),
                "release_score": float(min(1.0, max(patch_cut_ratio, 0.42 + 0.45 * closure_score))),
                "support_lost": False,
                "strict_closure_patch": True,
            })

        return patches

    def _extract_strict_ring_closure_patches(
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
        """Promote a local fragment from an actual ring of cut edges.

        Connected components can stay glued when a crack loop is visually clear
        but not every kNN bridge has been cut.  In strict mode we still require
        ring evidence: cut-edge midpoints must angularly surround the candidate
        center before a patch is released.
        """
        if (
            not self.crack_connected_release_only
            or positions is None
            or graph.knn_idx is None
            or self.material_family == "diffuse_damage"
        ):
            return []

        N = int(positions.shape[0])
        active_pair = torch.ones_like(corridor_edge_mask, dtype=torch.bool)
        occupied = (
            used_mask.to(device=positions.device, dtype=torch.bool).clone()
            if used_mask is not None
            else torch.zeros(N, dtype=torch.bool, device=positions.device)
        )
        if bool(occupied.any()):
            active_pair &= ~(occupied.unsqueeze(1) | occupied[graph.knn_idx])

        cut_edges = corridor_edge_mask.bool() & active_pair
        edge_rows, edge_cols = torch.where(cut_edges)
        min_edges = max(self.min_boundary_edges * 4, 32)
        if edge_rows.numel() < min_edges:
            return []

        nbr_idx = graph.knn_idx[edge_rows, edge_cols]
        edge_mids = 0.5 * (positions[edge_rows] + positions[nbr_idx])
        cut_degree = cut_edges.float().sum(dim=1)
        cut_nodes = torch.unique(torch.cat([edge_rows, nbr_idx], dim=0))

        opening_norm = torch.zeros_like(damage)
        if opening is not None and bool(opening.max() > 0.0):
            opening_scale = torch.quantile(opening.detach(), 0.85).clamp(min=1e-8)
            opening_norm = (opening / opening_scale).clamp(0.0, 1.0)

        damage_norm = damage.clamp(0.0, 1.0)
        degree_norm = (cut_degree / cut_degree.max().clamp(min=1.0)).clamp(0.0, 1.0)
        support = 0.48 * damage_norm + 0.34 * degree_norm + 0.18 * opening_norm
        if active_tip_mask is not None:
            support = torch.maximum(support, active_tip_mask.float() * 0.72)
        if recent_front_mask is not None:
            support = torch.maximum(support, recent_front_mask.float() * 0.58)
        support[occupied] = 0.0

        viable = cut_nodes[(~occupied[cut_nodes]) & (support[cut_nodes] > 0.05)]
        if viable.numel() == 0:
            return []
        candidate_count = min(24, int(viable.numel()))
        _, order = torch.topk(support[viable], k=candidate_count)
        candidates = viable[order]

        bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
        diag = float(bbox_extent.norm().item())
        base_radius = max(0.034, 0.095 * max(diag, 1e-6))
        radius_factors = (0.70, 0.90, 1.12, 1.35)
        if self.material_family == "sharp_brittle":
            max_patches = 2
            max_size_ratio = 0.070
        elif self.material_family == "rough_quasi_brittle":
            max_patches = 2
            max_size_ratio = 0.075
        else:
            max_patches = 2
            max_size_ratio = 0.055
        min_patch_size = max(self._effective_persistent_min_size(N), 14)
        max_patch_size = max(
            min_patch_size,
            int(round(max_size_ratio * float(positions.shape[0]))),
        )
        closure_threshold = 0.54
        if self.material_family == "sharp_brittle":
            closure_threshold = 0.50
        elif self.material_family == "rough_quasi_brittle":
            closure_threshold = 0.52

        patches: List[dict] = []
        for center_idx in candidates.tolist():
            if len(patches) >= max_patches:
                break
            if bool(occupied[int(center_idx)].item()):
                continue
            center = positions[int(center_idx)]
            best_patch = None
            best_score = -1.0

            for factor in radius_factors:
                radius = float(base_radius * factor)
                rel_edges = edge_mids - center.unsqueeze(0)
                edge_dist = rel_edges.norm(dim=1)
                local_edge_mask = (edge_dist <= radius) & (edge_dist >= 0.18 * radius)
                if int(local_edge_mask.sum().item()) < min_edges:
                    continue

                local_edges = rel_edges[local_edge_mask]
                cov = local_edges.transpose(0, 1) @ local_edges / float(
                    max(local_edges.shape[0] - 1, 1)
                )
                try:
                    _, eigvecs = torch.linalg.eigh(cov)
                except RuntimeError:
                    continue
                basis = eigvecs[:, -2:]
                plane_normal = eigvecs[:, 0]
                uv = local_edges @ basis
                local_radii = uv.norm(dim=1)
                valid = local_radii > (0.15 * local_radii.mean().clamp(min=1e-8))
                if int(valid.sum().item()) < min_edges:
                    continue
                uv = uv[valid]
                local_radii = local_radii[valid]
                angles = torch.atan2(uv[:, 1], uv[:, 0])
                angle_bins = max(self._closure_params()[0] * 2, 24)
                bin_pos = ((angles + torch.pi) / (2.0 * torch.pi) * angle_bins).floor().long()
                bin_pos = bin_pos.clamp(0, angle_bins - 1)
                occupied_bins = torch.bincount(bin_pos, minlength=angle_bins) > 0
                angular_coverage = float(occupied_bins.float().mean().item())
                sorted_angles = torch.sort(angles).values
                if sorted_angles.numel() > 1:
                    gaps = torch.diff(sorted_angles)
                    wrap_gap = (sorted_angles[0] + 2.0 * torch.pi) - sorted_angles[-1]
                    max_gap = float(torch.cat([gaps, wrap_gap.view(1)]).max().item())
                else:
                    max_gap = float(2.0 * torch.pi)
                gap_score = max(0.0, min(1.0, 1.0 - max_gap / float(2.0 * torch.pi)))
                radial_cv = float(local_radii.std(unbiased=False).item()) / max(
                    float(local_radii.mean().item()),
                    1e-6,
                )
                radial_score = max(0.0, min(1.0, 1.0 - radial_cv / 0.95))
                closure_score = (
                    0.52 * angular_coverage
                    + 0.34 * gap_score
                    + 0.14 * radial_score
                )
                if closure_score < closure_threshold:
                    continue

                ring_radius = float(torch.quantile(local_radii.detach(), 0.70).item())
                if ring_radius <= 1e-5:
                    continue
                rel_nodes = positions - center.unsqueeze(0)
                uv_nodes = rel_nodes @ basis
                node_radii = uv_nodes.norm(dim=1)
                plane_dist = (rel_nodes @ plane_normal).abs()
                slab = max(0.032, 0.22 * radius)
                patch_mask = (
                    (node_radii <= 0.92 * ring_radius)
                    & (plane_dist <= slab)
                    & (~occupied)
                )
                patch_size = int(patch_mask.sum().item())
                if patch_size < min_patch_size:
                    continue
                if patch_size > max_patch_size:
                    order_nodes = torch.argsort(node_radii + 0.35 * plane_dist)
                    trimmed = torch.zeros_like(patch_mask)
                    selected = order_nodes[patch_mask[order_nodes]][:max_patch_size]
                    trimmed[selected] = True
                    patch_mask = trimmed & (~occupied)
                    patch_size = int(patch_mask.sum().item())
                    if patch_size < min_patch_size:
                        continue

                cross_boundary = (
                    (patch_mask.unsqueeze(1) ^ patch_mask[graph.knn_idx])
                    & active_pair
                )
                boundary_edges = int(cross_boundary.sum().item())
                if boundary_edges < max(self.min_boundary_edges, 12):
                    continue
                patch_boundary_mask = corridor_edge_mask & cross_boundary
                cut_ratio = (
                    float(patch_boundary_mask[cross_boundary].float().mean().item())
                    if bool(cross_boundary.any())
                    else 0.0
                )
                required_cut_ratio = max(0.34, 0.62 * self._strict_patch_required_cut_ratio())
                if cut_ratio < required_cut_ratio and closure_score < (closure_threshold + 0.16):
                    continue
                crack_ratio = float((patch_mask & (support > 0.10)).float().mean().item())
                if crack_ratio < 0.04:
                    continue
                score = 0.58 * closure_score + 0.30 * min(cut_ratio, 1.0) + 0.12 * crack_ratio
                if score <= best_score:
                    continue
                best_score = score
                best_patch = {
                    "mask": patch_mask,
                    "boundary_mask": patch_boundary_mask,
                    "size": patch_size,
                    "seed_size": int((patch_mask & (support > 0.10)).sum().item()),
                    "cut_ratio": cut_ratio,
                    "center": positions[patch_mask].mean(dim=0),
                    "seed_center": center,
                    "plane_normal": plane_normal,
                    "closure_score": closure_score,
                    "boundary_score": cut_ratio,
                    "release_score": float(min(1.0, max(cut_ratio, 0.46 + 0.42 * closure_score))),
                    "support_lost": self.material_family == "rough_quasi_brittle",
                    "strict_ring_patch": True,
                }

            if best_patch is None:
                continue
            occupied |= best_patch["mask"]
            patches.append(best_patch)

        return patches

    def _extract_strict_crack_cascade_patches(
        self,
        positions: Optional[Tensor],
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        damage: Tensor,
        opening: Optional[Tensor],
        active_tip_mask: Optional[Tensor],
        recent_front_mask: Optional[Tensor],
        group_closure_scores: dict,
        used_mask: Optional[Tensor] = None,
        has_seed_patch: bool = False,
    ) -> List[dict]:
        """Release strict crack-corridor patches only after propagated cracks close locally.

        This replaces the old graphical shatter fallback for strict mode.  It can
        raise breakup density, but each patch still needs cut-edge contact and
        local angular support from the propagated crack network.  Rough
        quasi-brittle chunks may also use this path once the phase/damage field
        has made the crack corridor volumetrically weak; the birth remains
        crack-connected rather than an arbitrary damage patch.
        """
        if (
            not self.crack_connected_release_only
            or self.material_family not in {"sharp_brittle", "rough_quasi_brittle"}
            or positions is None
            or graph.knn_idx is None
            or corridor_edge_mask is None
        ):
            return []

        closure_max = max([float(value) for value in group_closure_scores.values()] or [0.0])
        damage_max = float(damage.max().item()) if damage.numel() else 0.0
        is_rough_crumble = (
            self.material_family == "rough_quasi_brittle"
            and self.crack_style == "chunky_crumble"
        )
        if self.material_family == "sharp_brittle":
            if not has_seed_patch and closure_max < 0.50:
                return []
        elif not is_rough_crumble:
            return []
        elif not has_seed_patch and closure_max < 0.42 and damage_max < 0.42:
            return []

        N = int(positions.shape[0])
        if N <= 0:
            return []
        occupied = (
            used_mask.to(device=positions.device, dtype=torch.bool).clone()
            if used_mask is not None
            else torch.zeros(N, dtype=torch.bool, device=positions.device)
        )
        detached_boundary = (
            self.detached_boundary_mask.to(device=positions.device, dtype=torch.bool)
            if (
                self.detached_boundary_mask is not None
                and self.detached_boundary_mask.shape == corridor_edge_mask.shape
            )
            else torch.zeros_like(corridor_edge_mask, dtype=torch.bool)
        )
        evidence_edge_mask = corridor_edge_mask.bool() | detached_boundary
        active_pair = ~(occupied.unsqueeze(1) & occupied[graph.knn_idx])
        cut_edges = evidence_edge_mask & active_pair
        edge_rows, edge_cols = torch.where(cut_edges)
        min_edges = (
            max(48, self.min_boundary_edges * 3)
            if is_rough_crumble
            else max(64, self.min_boundary_edges * 4)
        )
        if edge_rows.numel() < min_edges:
            return []

        nbr_idx = graph.knn_idx[edge_rows, edge_cols]
        edge_mids = 0.5 * (positions[edge_rows] + positions[nbr_idx])
        dtype = damage.dtype
        edge_count = torch.zeros(N, dtype=dtype, device=positions.device)
        ones = torch.ones(edge_rows.shape[0], dtype=dtype, device=positions.device)
        edge_count.index_add_(0, edge_rows, ones)
        edge_count.index_add_(0, nbr_idx, ones)
        edge_density = (edge_count / edge_count.max().clamp(min=1.0)).clamp(0.0, 1.0)
        cut_node_mask = edge_count > 0

        opening_norm = torch.zeros_like(damage)
        if opening is not None and bool(opening.max() > 0.0):
            opening_scale = torch.quantile(opening.detach(), 0.88).clamp(min=1e-8)
            opening_norm = (opening / opening_scale).clamp(0.0, 1.0)

        front_score = torch.zeros_like(damage)
        if recent_front_mask is not None:
            front_score = torch.maximum(front_score, recent_front_mask.float())
        if active_tip_mask is not None:
            front_score = torch.maximum(front_score, active_tip_mask.float())

        release_field = (
            0.50 * edge_density
            + 0.25 * damage.clamp(0.0, 1.0)
            + 0.15 * opening_norm
            + 0.10 * front_score
        ).clamp(0.0, 1.0)
        release_field[occupied] = 0.0

        if is_rough_crumble:
            seed_threshold = 0.18
        else:
            seed_threshold = 0.20 if self.crack_style == "radial_shatter" else 0.26
        seed_mask = cut_node_mask & (~occupied) & (release_field >= seed_threshold)
        if not bool(seed_mask.any()):
            viable = torch.where(cut_node_mask & (~occupied))[0]
            if viable.numel() == 0:
                return []
            top_k = min(max(24, self.min_boundary_edges * 3), int(viable.numel()))
            top = release_field[viable].topk(top_k).indices
            seed_mask = torch.zeros(N, dtype=torch.bool, device=positions.device)
            seed_mask[viable[top]] = True

        seed_idx = torch.where(seed_mask)[0]
        order = release_field[seed_idx].argsort(descending=True)
        seed_trials = 80 if is_rough_crumble else 64
        seed_idx = seed_idx[order[: min(seed_trials, seed_idx.numel())]]

        bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
        diag = float(bbox_extent.norm().item())
        base_radius = max(
            0.024 if is_rough_crumble else 0.020,
            (0.076 if is_rough_crumble else 0.066) * max(diag, 1e-6),
        )
        radius_factors = (0.78, 1.02, 1.28, 1.55) if is_rough_crumble else (0.74, 0.96, 1.18)
        min_patch_size = max(self._effective_persistent_min_size(N), 12)
        max_patch_size = max(
            min_patch_size,
            int(round((0.070 if is_rough_crumble else 0.055) * N)),
        )
        if is_rough_crumble:
            max_patches = 8
        else:
            max_patches = 16 if self.crack_style == "radial_shatter" else 10
        patches: List[dict] = []

        for seed in seed_idx.tolist():
            if len(patches) >= max_patches:
                break
            if bool(occupied[int(seed)].item()):
                continue
            seed_pos = positions[int(seed)]
            best_patch = None
            best_score = -1.0

            for factor in radius_factors:
                radius = float(base_radius * factor)
                rel_edges = edge_mids - seed_pos.unsqueeze(0)
                edge_dist = rel_edges.norm(dim=1)
                local_edge_mask = (edge_dist <= radius) & (edge_dist >= 0.16 * radius)
                local_min_edges = (
                    max(18, self.min_boundary_edges * 2)
                    if is_rough_crumble
                    else max(24, self.min_boundary_edges * 2)
                )
                if int(local_edge_mask.sum().item()) < local_min_edges:
                    continue

                local_edges = rel_edges[local_edge_mask]
                cov = local_edges.transpose(0, 1) @ local_edges / float(max(local_edges.shape[0] - 1, 1))
                try:
                    _, eigvecs = torch.linalg.eigh(cov)
                except RuntimeError:
                    continue
                basis = eigvecs[:, -2:]
                plane_normal = eigvecs[:, 0]
                uv_edges = local_edges @ basis
                edge_radii = uv_edges.norm(dim=1)
                valid_edges = edge_radii > (0.14 * edge_radii.mean().clamp(min=1e-8))
                if int(valid_edges.sum().item()) < local_min_edges:
                    continue
                uv_edges = uv_edges[valid_edges]
                edge_radii = edge_radii[valid_edges]
                angles = torch.atan2(uv_edges[:, 1], uv_edges[:, 0])
                angle_bins = 28
                bin_pos = ((angles + torch.pi) / (2.0 * torch.pi) * angle_bins).floor().long()
                bin_pos = bin_pos.clamp(0, angle_bins - 1)
                angular_coverage = float(
                    (torch.bincount(bin_pos, minlength=angle_bins) > 0).float().mean().item()
                )
                sorted_angles = torch.sort(angles).values
                gaps = torch.diff(sorted_angles)
                wrap_gap = (sorted_angles[0] + 2.0 * torch.pi) - sorted_angles[-1]
                max_gap = float(torch.cat([gaps, wrap_gap.view(1)]).max().item())
                gap_score = max(0.0, min(1.0, 1.0 - max_gap / float(2.0 * torch.pi)))
                ring_score = 0.56 * angular_coverage + 0.44 * gap_score
                if is_rough_crumble:
                    ring_threshold = 0.34
                else:
                    ring_threshold = 0.42
                if ring_score < ring_threshold:
                    continue

                ring_radius = float(torch.quantile(edge_radii.detach(), 0.66).item())
                if ring_radius <= 1e-5:
                    continue
                rel_nodes = positions - seed_pos.unsqueeze(0)
                uv_nodes = rel_nodes @ basis
                node_radii = uv_nodes.norm(dim=1)
                plane_dist = (rel_nodes @ plane_normal).abs()
                patch_mask = (
                    (~occupied)
                    & (node_radii <= 0.90 * ring_radius)
                    & (plane_dist <= max(0.026, 0.22 * radius))
                    & (
                        (release_field >= max(0.10, 0.45 * seed_threshold))
                        | cut_node_mask
                        | (damage >= 0.18 if is_rough_crumble else torch.zeros_like(damage, dtype=torch.bool))
                    )
                )
                patch_size = int(patch_mask.sum().item())
                if patch_size < min_patch_size:
                    continue
                if patch_size > max_patch_size:
                    patch_idx = torch.where(patch_mask)[0]
                    keep_order = (
                        node_radii[patch_idx]
                        + 0.35 * plane_dist[patch_idx]
                        - 0.25 * release_field[patch_idx]
                    ).argsort()
                    keep = patch_idx[keep_order[:max_patch_size]]
                    patch_mask.zero_()
                    patch_mask[keep] = True
                    patch_size = int(patch_mask.sum().item())
                    if patch_size < min_patch_size:
                        continue

                cross_boundary = (
                    (patch_mask.unsqueeze(1) ^ patch_mask[graph.knn_idx])
                    & active_pair
                )
                boundary_edges = int(cross_boundary.sum().item())
                if boundary_edges < max(12, self.min_boundary_edges):
                    continue
                patch_boundary_mask = corridor_edge_mask & cross_boundary
                contact_mask = corridor_edge_mask & (
                    patch_mask.unsqueeze(1) | patch_mask[graph.knn_idx]
                )
                boundary_cut_edges = int(patch_boundary_mask.sum().item())
                contact_edges = int(contact_mask.sum().item())
                cut_ratio = (
                    float(patch_boundary_mask[cross_boundary].float().mean().item())
                    if bool(cross_boundary.any()) else 0.0
                )
                contact_ratio = min(1.0, contact_edges / max(float(patch_size), 1.0))
                required_cut = (
                    max(0.24, 0.42 * self._strict_patch_required_cut_ratio())
                    if is_rough_crumble
                    else max(0.34, 0.50 * self._strict_patch_required_cut_ratio())
                )
                if cut_ratio < required_cut and contact_edges < max(18, self.min_boundary_edges * 2):
                    continue
                patch_score = float(release_field[patch_mask].mean().item())
                peak_score = float(release_field[patch_mask].max().item())
                release_score = (
                    0.36 * ring_score
                    + 0.28 * min(cut_ratio, 1.0)
                    + 0.20 * min(contact_ratio, 1.0)
                    + 0.16 * peak_score
                )
                if is_rough_crumble:
                    release_threshold = 0.36
                else:
                    release_threshold = 0.42
                if release_score < release_threshold or release_score <= best_score:
                    continue
                boundary_mask = patch_boundary_mask if boundary_cut_edges > 0 else contact_mask
                best_score = release_score
                best_patch = {
                    "mask": patch_mask,
                    "boundary_mask": boundary_mask,
                    "size": patch_size,
                    "seed_size": int((patch_mask & cut_node_mask).sum().item()),
                    "cut_ratio": max(cut_ratio, contact_ratio),
                    "center": positions[patch_mask].mean(dim=0),
                    "seed_center": seed_pos,
                    "plane_normal": plane_normal,
                    "closure_score": ring_score,
                    "boundary_score": max(cut_ratio, contact_ratio),
                    "release_score": float(min(1.0, release_score)),
                    "support_lost": bool(is_rough_crumble),
                    "strict_closure_patch": True,
                    "strict_cascade_patch": True,
                }

            if best_patch is None:
                continue
            occupied |= best_patch["mask"]
            patches.append(best_patch)

        return patches

    def _extract_strict_impact_closure_patches(
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
        """Early sharp-brittle closure cells grown from the crack corridor.

        This is not a particle-spray fallback.  Patches are only born while the
        impact burst is active, and every patch needs support from the current
        crack/cut corridor or from boundaries of already detached strict patches.
        """
        if (
            not bool(getattr(self, "impact_closure_active", False))
            or not self.crack_connected_release_only
            or self.material_family != "sharp_brittle"
            or positions is None
            or graph.knn_idx is None
            or corridor_edge_mask is None
        ):
            return []

        N = int(positions.shape[0])
        if N <= 0:
            return []
        device = positions.device
        dtype = damage.dtype
        occupied = (
            used_mask.to(device=device, dtype=torch.bool).clone()
            if used_mask is not None
            else torch.zeros(N, dtype=torch.bool, device=device)
        )

        target_ratio = float(getattr(self, "impact_closure_target_ratio", 0.0))
        target_ratio = min(max(target_ratio, 0.0), self._strict_closure_release_cap())
        ratio_min_size = int(round(float(N) * float(getattr(
            self, "impact_closure_min_size_ratio", 0.0))))
        min_patch_size = max(self._effective_persistent_min_size(N), ratio_min_size, 10)
        max_released_nodes = int(round(target_ratio * float(N)))
        remaining_budget = max_released_nodes - int(occupied.sum().item())
        if remaining_budget < min_patch_size:
            return []

        detached_boundary = (
            self.detached_boundary_mask.to(device=device, dtype=torch.bool)
            if (
                self.detached_boundary_mask is not None
                and self.detached_boundary_mask.shape == corridor_edge_mask.shape
            )
            else torch.zeros_like(corridor_edge_mask, dtype=torch.bool)
        )
        evidence_edge_mask = corridor_edge_mask.bool() | detached_boundary
        active_pair = ~(occupied.unsqueeze(1) & occupied[graph.knn_idx])
        cut_edges = evidence_edge_mask & active_pair
        edge_rows, edge_cols = torch.where(cut_edges)
        if edge_rows.numel() < max(72, self.min_boundary_edges * 5):
            return []

        nbr_idx = graph.knn_idx[edge_rows, edge_cols]
        edge_mids = 0.5 * (positions[edge_rows] + positions[nbr_idx])
        edge_count = torch.zeros(N, dtype=dtype, device=device)
        ones = torch.ones(edge_rows.shape[0], dtype=dtype, device=device)
        edge_count.index_add_(0, edge_rows, ones)
        edge_count.index_add_(0, nbr_idx, ones)
        edge_density = (edge_count / edge_count.max().clamp(min=1.0)).clamp(0.0, 1.0)
        crack_support = edge_density
        for _ in range(2):
            crack_support = torch.maximum(
                crack_support,
                0.82 * graph.weighted_neighbor_max(crack_support),
            ).clamp(0.0, 1.0)
        cut_node_mask = edge_count > 0

        opening_norm = torch.zeros_like(damage)
        if opening is not None and bool(opening.max() > 0.0):
            opening_scale = torch.quantile(opening.detach(), 0.88).clamp(min=1e-8)
            opening_norm = (opening / opening_scale).clamp(0.0, 1.0)

        front_score = torch.zeros_like(damage)
        if recent_front_mask is not None:
            front_score = torch.maximum(front_score, recent_front_mask.float())
        if active_tip_mask is not None:
            front_score = torch.maximum(front_score, active_tip_mask.float())

        release_field = (
            0.44 * crack_support
            + 0.24 * damage.clamp(0.0, 1.0)
            + 0.18 * opening_norm
            + 0.14 * front_score
        ).clamp(0.0, 1.0)
        release_field[occupied] = 0.0

        weights = (edge_density + 0.40 * damage.clamp(0.0, 1.0) + 0.32 * front_score).detach()
        if bool((weights > 1e-6).any()):
            center = (positions * weights.unsqueeze(1)).sum(dim=0) / weights.sum().clamp(min=1e-8)
        else:
            center = edge_mids.mean(dim=0)

        centered_edges = edge_mids - center.unsqueeze(0)
        cov = centered_edges.transpose(0, 1) @ centered_edges / float(max(centered_edges.shape[0] - 1, 1))
        try:
            _, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError:
            return []
        basis = eigvecs[:, -2:]
        plane_normal = eigvecs[:, 0]
        centered = positions - center.unsqueeze(0)
        uv = centered @ basis
        radial = uv.norm(dim=1)
        theta = torch.atan2(uv[:, 1], uv[:, 0])
        plane_dist = (centered @ plane_normal).abs()
        supported_radial = radial[cut_node_mask] if bool(cut_node_mask.any()) else radial
        radial_scale = torch.quantile(supported_radial.detach(), 0.98).clamp(min=1e-8)
        radial_norm = (radial / radial_scale).clamp(0.0, 1.0)
        plane_scale = torch.quantile(plane_dist.detach(), 0.90).clamp(min=1e-8)
        plane_norm = (plane_dist / plane_scale).clamp(0.0, 1.0)

        support_threshold = 0.055 if self.crack_style == "radial_shatter" else 0.075
        release_threshold = 0.075 if self.crack_style == "radial_shatter" else 0.095
        candidate = (
            (~occupied)
            & (crack_support >= support_threshold)
            & (release_field >= release_threshold)
        )
        if int(candidate.sum().item()) < min_patch_size:
            viable = torch.where((~occupied) & (crack_support >= 0.025))[0]
            if viable.numel() < min_patch_size:
                return []
            top_k = min(max(min_patch_size * 8, remaining_budget), int(viable.numel()))
            top = release_field[viable].topk(top_k).indices
            candidate = torch.zeros(N, dtype=torch.bool, device=device)
            candidate[viable[top]] = True

        sectors = int(getattr(self, "impact_closure_sector_count", 0) or 0)
        if sectors <= 0:
            sectors = 40 if self.crack_style == "radial_shatter" else 28
        bands = int(getattr(self, "impact_closure_band_count", 0) or 0)
        if bands <= 0:
            bands = 4 if self.crack_style == "radial_shatter" else 3
        layers = int(getattr(self, "impact_closure_layer_count", 0) or 0)
        if layers <= 0:
            layers = 2
        sector_id = (((theta + math.pi) / (2.0 * math.pi)) * sectors).floor().long()
        sector_id = sector_id.clamp(0, sectors - 1)
        band_id = (radial_norm * bands).floor().long().clamp(0, bands - 1)
        layer_id = (plane_norm * layers).floor().long().clamp(0, layers - 1)
        group_id = sector_id + sectors * (band_id + bands * layer_id)
        unique_groups = group_id[candidate].unique(sorted=False)
        if unique_groups.numel() == 0:
            return []

        default_max_size_ratio = 0.016 if self.crack_style == "radial_shatter" else 0.022
        configured_max_size_ratio = float(getattr(
            self, "impact_closure_max_size_ratio", 0.0) or 0.0)
        max_size_ratio = configured_max_size_ratio if configured_max_size_ratio > 0.0 else default_max_size_ratio
        max_patch_size = max(min_patch_size, int(round(max_size_ratio * N)))
        max_patches = int(getattr(self, "impact_closure_max_patches", 0) or 0)
        if max_patches <= 0:
            return []
        max_patches = min(max_patches, 104 if self.crack_style == "radial_shatter" else 56)

        group_candidates = []
        for gid in unique_groups.tolist():
            mask = candidate & (group_id == int(gid))
            size = int(mask.sum().item())
            if size < min_patch_size:
                continue
            patch_side = mask.unsqueeze(1)
            cross_boundary = (patch_side ^ mask[graph.knn_idx]) & active_pair
            boundary_edges = int(cross_boundary.sum().item())
            if boundary_edges < max(10, self.min_boundary_edges):
                continue
            evidence_boundary = evidence_edge_mask & cross_boundary
            evidence_contact = evidence_edge_mask & (patch_side | mask[graph.knn_idx])
            boundary_cut_edges = int(evidence_boundary.sum().item())
            contact_edges = int(evidence_contact.sum().item())
            cut_ratio = boundary_cut_edges / max(float(boundary_edges), 1.0)
            contact_ratio = contact_edges / max(float(size), 1.0)
            if cut_ratio < 0.10 and contact_ratio < 0.72:
                continue
            mean_score = float(release_field[mask].mean().item())
            peak_score = float(release_field[mask].max().item())
            support_score = min(1.0, 0.60 * cut_ratio + 0.40 * min(contact_ratio, 1.0))
            size_score = min(1.0, size / max(float(max_patch_size), 1.0))
            group_candidates.append((
                0.38 * peak_score + 0.30 * mean_score + 0.22 * support_score + 0.10 * size_score,
                int(gid),
                cut_ratio,
                contact_ratio,
            ))
        group_candidates.sort(reverse=True)

        patches: List[dict] = []
        released_now = 0
        used = occupied.clone()
        for _, gid, _, _ in group_candidates[: max(max_patches * 2, 1)]:
            if len(patches) >= max_patches or released_now >= remaining_budget:
                break
            patch_mask = candidate & (group_id == gid) & (~used)
            patch_size = int(patch_mask.sum().item())
            if patch_size < min_patch_size:
                continue
            allowed_size = min(max_patch_size, remaining_budget - released_now)
            if allowed_size < min_patch_size:
                break
            if patch_size > allowed_size:
                patch_idx = torch.where(patch_mask)[0]
                local_score = (
                    release_field[patch_idx]
                    + 0.16 * crack_support[patch_idx]
                    - 0.05 * radial_norm[patch_idx]
                )
                keep = patch_idx[local_score.topk(int(allowed_size)).indices]
                patch_mask.zero_()
                patch_mask[keep] = True
                patch_size = int(patch_mask.sum().item())
                if patch_size < min_patch_size:
                    continue

            patch_side = patch_mask.unsqueeze(1)
            cross_boundary = (patch_side ^ patch_mask[graph.knn_idx]) & active_pair
            boundary_edges = int(cross_boundary.sum().item())
            if boundary_edges < max(10, self.min_boundary_edges):
                continue
            evidence_boundary = evidence_edge_mask & cross_boundary
            evidence_contact = evidence_edge_mask & (patch_side | patch_mask[graph.knn_idx])
            boundary_cut_edges = int(evidence_boundary.sum().item())
            contact_edges = int(evidence_contact.sum().item())
            cut_ratio = boundary_cut_edges / max(float(boundary_edges), 1.0)
            contact_ratio = contact_edges / max(float(patch_size), 1.0)
            if cut_ratio < 0.08 and contact_ratio < 0.62:
                continue

            boundary_mask = evidence_boundary | (evidence_contact & cross_boundary)
            if not bool(boundary_mask.any()):
                boundary_mask = evidence_contact
            patch_score = float(release_field[patch_mask].mean().item())
            peak_score = float(release_field[patch_mask].max().item())
            support_score = min(1.0, 0.56 * cut_ratio + 0.44 * min(contact_ratio, 1.0))
            release_score = min(1.0, 0.42 * peak_score + 0.32 * patch_score + 0.26 * support_score)
            patch_indices = torch.where(patch_mask)[0]
            seed_index = int(patch_indices[release_field[patch_indices].argmax()].item())
            patches.append({
                "mask": patch_mask,
                "boundary_mask": boundary_mask,
                "size": patch_size,
                "seed_size": int((patch_mask & cut_node_mask).sum().item()),
                "cut_ratio": max(cut_ratio, min(contact_ratio, 1.0)),
                "center": positions[patch_mask].mean(dim=0),
                "seed_center": positions[seed_index],
                "plane_normal": plane_normal,
                "closure_score": support_score,
                "boundary_score": max(cut_ratio, min(contact_ratio, 1.0)),
                "release_score": float(max(0.68, release_score)),
                "support_lost": False,
                "strict_closure_patch": True,
                "strict_impact_closure_patch": True,
            })
            used |= patch_mask
            released_now += patch_size

        self.last_impact_closure_patches = len(patches)
        self.last_impact_closure_nodes = int(sum(int(patch["size"]) for patch in patches))
        self.last_impact_closure_score_max = (
            max(float(patch.get("release_score", 0.0)) for patch in patches)
            if patches else 0.0
        )
        return patches
