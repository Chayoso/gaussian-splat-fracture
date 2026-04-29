"""Phase-field approval logic for graph-based fragment birth."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from .graph_builder import GaussianGraph


class FragmentPhaseApprovalMixin:
    """Gate surface closure candidates with narrow-band phase damage evidence."""

    def _reset_phase_approval_stats(self) -> None:
        self.last_phase_candidate_patches = 0
        self.last_phase_approved_patches = 0
        self.last_phase_rejected_patches = 0
        self.last_phase_approved_nodes = 0
        self.last_phase_approval_score_max = 0.0
        self.last_phase_approval_score_mean = 0.0
        self.last_phase_approval_cvol_max = 0.0
        self.last_phase_approval_gate_max = 0.0
        self.last_pseudo_thickness_mass = 0.0
        self.last_birth_phase_score = 0.0
        self.last_birth_cvol_max = 0.0
        self.last_birth_phase_gate_max = 0.0
        self.last_birth_phase_approved = False

    def _phase_approval_threshold(self) -> float:
        if self.material_family == "sharp_brittle":
            base = 0.34
            if bool(getattr(self, "impact_closure_active", False)):
                base = 0.28
            return self._phase_approval_threshold_with_runtime_offset(base)
        if self.material_family == "rough_quasi_brittle":
            return self._phase_approval_threshold_with_runtime_offset(0.36)
        if self.material_family == "brittle_moderate":
            return self._phase_approval_threshold_with_runtime_offset(0.40)
        return self._phase_approval_threshold_with_runtime_offset(0.46)

    def _phase_approval_threshold_with_runtime_offset(self, base: float) -> float:
        scale = float(getattr(self, "phase_approval_threshold_scale", 1.0))
        offset = float(getattr(self, "phase_approval_threshold_offset", 0.0))
        return float(max(0.02, min(base * scale + offset, 0.95)))

    def _phase_approval_metrics(
        self,
        *,
        patch: dict,
        graph: GaussianGraph,
        damage: Tensor,
        opening: Optional[Tensor],
        phase_gate: Optional[Tensor],
        volume_damage: Optional[Tensor],
    ) -> dict:
        patch_mask = patch["mask"].bool()
        patch_size = int(patch_mask.sum().item())
        device = damage.device
        dtype = damage.dtype
        if patch_size <= 0:
            return {
                "phase_score": 0.0,
                "phase_approved": False,
                "phase_cvol_max": 0.0,
                "phase_cvol_mean": 0.0,
                "phase_gate_max": 0.0,
                "phase_gate_mean": 0.0,
                "phase_opening_max": 0.0,
                "pseudo_thickness_mass": 0.0,
            }

        phase_values = (
            phase_gate.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            if phase_gate is not None
            else damage.clamp(0.0, 1.0)
        )
        volume_values = (
            volume_damage.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            if volume_damage is not None
            else torch.maximum(damage.clamp(0.0, 1.0), 0.55 * phase_values)
        )

        boundary_nodes = None
        boundary_mask = patch.get("boundary_mask")
        if (
            boundary_mask is not None
            and graph.knn_idx is not None
            and boundary_mask.shape == graph.knn_idx.shape
        ):
            rows, cols = torch.where(boundary_mask.bool())
            if rows.numel() > 0:
                nbrs = graph.knn_idx[rows, cols]
                valid = (nbrs >= 0) & (nbrs < damage.shape[0])
                if bool(valid.any()):
                    boundary_nodes = torch.unique(torch.cat([rows[valid], nbrs[valid]], dim=0))

        eval_mask = patch_mask.clone()
        if boundary_nodes is not None and boundary_nodes.numel() > 0:
            eval_mask[boundary_nodes] = True
        eval_mask = eval_mask[: damage.shape[0]]
        if not bool(eval_mask.any()):
            eval_mask = patch_mask[: damage.shape[0]]

        cvol_local = volume_values[: damage.shape[0]][eval_mask]
        phase_local = phase_values[: damage.shape[0]][eval_mask]
        cvol_patch = volume_values[: damage.shape[0]][patch_mask[: damage.shape[0]]]
        phase_patch = phase_values[: damage.shape[0]][patch_mask[: damage.shape[0]]]
        cvol_max = float(cvol_local.max().item()) if cvol_local.numel() else 0.0
        cvol_mean = float(cvol_patch.mean().item()) if cvol_patch.numel() else 0.0
        phase_max = float(phase_local.max().item()) if phase_local.numel() else 0.0
        phase_mean = float(phase_patch.mean().item()) if phase_patch.numel() else 0.0

        opening_max = 0.0
        if opening is not None and opening.numel() > 0:
            local_opening = opening.to(device=device, dtype=dtype)[: damage.shape[0]].clamp(min=0.0)
            if bool(local_opening.max() > 0.0):
                opening_scale = torch.quantile(local_opening.detach(), 0.85).clamp(min=1e-6)
                opening_norm = (local_opening / opening_scale).clamp(0.0, 1.0)
                vals = opening_norm[eval_mask]
                opening_max = float(vals.max().item()) if vals.numel() else 0.0

        boundary_score = float(
            max(
                float(patch.get("cut_ratio", 0.0)),
                float(patch.get("boundary_score", 0.0)),
                float(patch.get("closure_score", 0.0)) * 0.82,
            )
        )
        phase_score = (
            0.34 * cvol_max
            + 0.18 * cvol_mean
            + 0.24 * phase_max
            + 0.12 * opening_max
            + 0.12 * min(max(boundary_score, 0.0), 1.0)
        )
        # Strong crack/cut support with a high phase gate is enough for impact
        # birth; otherwise the local volume proxy must carry the patch.
        phase_score = max(
            phase_score,
            0.56 * phase_max + 0.28 * cvol_mean + 0.16 * min(boundary_score, 1.0),
        )
        threshold = self._phase_approval_threshold()
        approved = bool(
            phase_score >= threshold
            and (
                cvol_max >= 0.18
                or phase_max >= 0.58
                or (opening_max >= 0.45 and boundary_score >= 0.35)
            )
        )
        pseudo_mass = float(patch_size) * (
            1.0
            + 1.75 * max(cvol_mean, 0.0)
            + 0.85 * max(phase_mean, 0.0)
            + 0.35 * min(max(boundary_score, 0.0), 1.0)
        )
        return {
            "phase_score": float(max(0.0, min(phase_score, 1.0))),
            "phase_approved": approved,
            "phase_cvol_max": cvol_max,
            "phase_cvol_mean": cvol_mean,
            "phase_gate_max": phase_max,
            "phase_gate_mean": phase_mean,
            "phase_opening_max": opening_max,
            "pseudo_thickness_mass": pseudo_mass,
        }

    def _filter_phase_approved_patches(
        self,
        *,
        patches: List[dict],
        graph: GaussianGraph,
        damage: Tensor,
        opening: Optional[Tensor],
        phase_gate: Optional[Tensor],
        volume_damage: Optional[Tensor],
    ) -> List[dict]:
        self.last_phase_candidate_patches = len(patches)
        if not patches:
            return []
        phase_approval_enabled = bool(getattr(self, "phase_approval_enable", True))
        approved: List[dict] = []
        rejected = 0
        for patch in patches:
            metrics = self._phase_approval_metrics(
                patch=patch,
                graph=graph,
                damage=damage,
                opening=opening,
                phase_gate=phase_gate,
                volume_damage=volume_damage,
            )
            metrics["phase_approval_bypassed"] = not phase_approval_enabled
            patch.update(metrics)
            if bool(metrics["phase_approved"]) or not phase_approval_enabled:
                approved.append(patch)
            else:
                rejected += 1
        self.last_phase_rejected_patches = rejected
        return approved

    def _supports_cut_surface(self) -> bool:
        return self.cut_surface_enable and self.material_family in {
            "sharp_brittle",
            "brittle_moderate",
            "rough_quasi_brittle",
        }

    def _effective_impact_release_gain(self) -> float:
        """Runtime impact/external-force gain used only for release promotion."""
        return max(0.35, min(float(getattr(self, "impact_release_gain", 1.0)), 2.75))

    def _authoritative_cut_threshold(self) -> float:
        thresh = self.authoritative_cut_threshold
        if self.material_family == "sharp_brittle":
            thresh *= 0.82
        elif self.material_family == "rough_quasi_brittle":
            thresh *= 0.92
        elif self.material_family == "brittle_moderate":
            thresh *= 1.05
        return float(thresh)
