"""Render-only Gaussian splitter for visible fracture aftermath."""

import math
from typing import Dict, Optional

import torch
from torch import Tensor


class GaussianSplitter:
    """
    Split and deform Gaussians based on fracture state.

    Operations (applied in order):
        1. Covariance deformation: flatten Gaussians along crack normal
        2. Opacity modulation: reduce opacity in high-damage regions
        3. Gaussian split: duplicate high-damage Gaussians, offset along ±n
    """

    def __init__(
        self,
        split_threshold: float = 0.8,
        opacity_threshold: float = 0.6,
        flatten_threshold: float = 0.3,
        max_flatten: float = 0.5,
        max_opacity_reduction: float = 0.8,
        split_offset_scale: float = 1.5,
        device: str = "cuda",
    ):
        self.split_threshold = float(split_threshold)
        self.opacity_threshold = float(opacity_threshold)
        self.flatten_threshold = float(flatten_threshold)
        self.max_flatten = float(max_flatten)
        self.max_opacity_reduction = float(max_opacity_reduction)
        self.split_offset_scale = float(split_offset_scale)
        self.device = torch.device(device)

        self._base_count: int = 0
        self._active_parent_mask: Optional[Tensor] = None
        self._spawn_frame: Optional[Tensor] = None
        self._parent_side: Optional[Tensor] = None
        self._last_append_parent_idx: Optional[Tensor] = None
        self._last_split_mask: Optional[Tensor] = None
        self._last_metrics: Dict[str, float] = {
            "visible_shard_count": 0,
            "split_gap_visibility": 0.0,
            "fragment_shell_contrast": 0.0,
            "shard_persistence": 0.0,
        }

    def _ensure_registry(self, n_base: int) -> None:
        if self._active_parent_mask is not None and self._base_count == n_base:
            return
        self._base_count = int(n_base)
        self._active_parent_mask = torch.zeros(
            n_base, dtype=torch.bool, device=self.device
        )
        self._spawn_frame = torch.full(
            (n_base,), -1, dtype=torch.long, device=self.device
        )
        self._parent_side = torch.zeros(
            n_base, dtype=torch.float32, device=self.device
        )
        self._last_append_parent_idx = None
        self._last_split_mask = None
        self._last_metrics = {
            "visible_shard_count": 0,
            "split_gap_visibility": 0.0,
            "fragment_shell_contrast": 0.0,
            "shard_persistence": 0.0,
        }

    def compose_render_state(
        self,
        positions: Tensor,
        damage: Optional[Tensor],
        opening: Optional[Tensor] = None,
        fragment_ids: Optional[Tensor] = None,
        crack_tips: Optional[Tensor] = None,
        crack_visited: Optional[Tensor] = None,
        debris_mask: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """Compose base + render-only shard auxiliary state for plotting."""
        n_base = int(damage.shape[0]) if damage is not None else int(positions.shape[0])
        if damage is None:
            damage = torch.zeros(n_base, device=positions.device)
        if opening is None:
            opening = torch.zeros(n_base, device=positions.device)
        if fragment_ids is None:
            fragment_ids = torch.zeros(n_base, dtype=torch.long, device=positions.device)
        if crack_tips is None:
            crack_tips = torch.zeros(n_base, dtype=torch.bool, device=positions.device)
        if crack_visited is None:
            crack_visited = torch.zeros(n_base, dtype=torch.bool, device=positions.device)
        if debris_mask is None:
            debris_mask = torch.zeros(n_base, dtype=torch.bool, device=positions.device)

        base_shard_mask = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
        parent_idx = self._last_append_parent_idx
        if parent_idx is None or parent_idx.numel() == 0:
            return {
                "positions": positions,
                "damage": damage,
                "opening": opening,
                "fragment_ids": fragment_ids,
                "crack_tips": crack_tips,
                "crack_visited": crack_visited,
                "debris_mask": debris_mask,
                "shard_mask": base_shard_mask,
                **self._last_metrics,
            }

        appended = parent_idx.numel()
        positions_already_extended = positions.shape[0] == (n_base + appended)
        shard_damage = damage[parent_idx].clamp(min=self.split_threshold)
        shard_opening = opening[parent_idx]
        shard_fragment_ids = fragment_ids[parent_idx]
        shard_tips = torch.zeros(appended, dtype=torch.bool, device=positions.device)
        shard_visited = crack_visited[parent_idx]
        shard_debris = torch.ones(appended, dtype=torch.bool, device=positions.device)

        if positions_already_extended:
            shard_mask = base_shard_mask
            shard_mask[n_base:] = True
            out_damage = torch.cat([damage, shard_damage], dim=0)
            out_opening = torch.cat([opening, shard_opening], dim=0)
            out_fragment_ids = torch.cat([fragment_ids, shard_fragment_ids], dim=0)
            out_tips = torch.cat([crack_tips, shard_tips], dim=0)
            out_visited = torch.cat([crack_visited, shard_visited], dim=0)
            out_debris = torch.cat([debris_mask, shard_debris], dim=0)
        else:
            shard_mask = torch.cat(
                [base_shard_mask, torch.ones(appended, dtype=torch.bool, device=positions.device)],
                dim=0,
            )
            out_damage = torch.cat([damage, shard_damage], dim=0)
            out_opening = torch.cat([opening, shard_opening], dim=0)
            out_fragment_ids = torch.cat([fragment_ids, shard_fragment_ids], dim=0)
            out_tips = torch.cat([crack_tips, shard_tips], dim=0)
            out_visited = torch.cat([crack_visited, shard_visited], dim=0)
            out_debris = torch.cat([debris_mask, shard_debris], dim=0)

        return {
            "positions": positions,
            "damage": out_damage,
            "opening": out_opening,
            "fragment_ids": out_fragment_ids,
            "crack_tips": out_tips,
            "crack_visited": out_visited,
            "debris_mask": out_debris,
            "shard_mask": shard_mask,
            **self._last_metrics,
        }

    @staticmethod
    def _safe_normalize(v: Tensor) -> Tensor:
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return v / norm

    @staticmethod
    def _append_param_data(param, extra: Tensor) -> None:
        param.data = torch.cat([param.data, extra], dim=0)

    def apply_deformation(
        self,
        gaussians,
        c: Tensor,
        n: Tensor,
        a: Tensor,
        F_per_gaussian: Optional[Tensor] = None,
    ) -> None:
        """Compatibility hook; base deformation is handled by the visualizer."""
        _ = (gaussians, c, n, a, F_per_gaussian)

    def _family_target_count(
        self,
        n_base: int,
        candidate_count: int,
        material_family: str,
        shard_count_scale: float,
    ) -> int:
        if candidate_count <= 0:
            return 0
        if material_family == "sharp_brittle":
            base_fraction = 0.0011
            min_count = 8
        elif material_family == "rough_quasi_brittle":
            base_fraction = 0.0022
            min_count = 18
        else:
            base_fraction = 0.0010
            min_count = 6
        target = int(round(n_base * base_fraction * max(shard_count_scale, 0.15)))
        target = max(min_count, target)
        return min(candidate_count, target)

    def split_gaussians(
        self,
        gaussians,
        c: Tensor,
        n: Tensor,
        a: Tensor,
        fragment_ids: Optional[Tensor] = None,
        crack_tips: Optional[Tensor] = None,
        crack_visited: Optional[Tensor] = None,
        impact_center: Optional[Tensor] = None,
        frame: int = 0,
        shard_enable: bool = False,
        shard_count_scale: float = 1.0,
        debris_motion_gain: float = 0.0,
        fragment_offset_gain: float = 1.0,
        material_family: str = "neutral_reference",
        shard_scale_gain: float = 1.0,
        shard_opacity_gain: float = 1.0,
        debris_darkening: float = 0.2,
        split_event_boost: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Split high-damage Gaussians into two halves.

        For each Gaussian with c > split_threshold:
            1. Create two copies offset along ±n by a_i
            2. Halve the scale perpendicular to n
            3. Reduce opacity of both halves

        Args:
            gaussians: 3DGS GaussianModel
            c: (N,) damage
            n: (N, 3) crack normals
            a: (N,) opening magnitudes

        Returns:
            split_info: dict with mapping from old to new indices
        """
        n_base = int(c.shape[0])
        self._ensure_registry(n_base)
        self._last_append_parent_idx = None
        self._last_split_mask = torch.zeros(
            n_base, dtype=torch.bool, device=c.device
        )
        self._last_metrics = {
            "visible_shard_count": 0,
            "split_gap_visibility": 0.0,
            "fragment_shell_contrast": 0.0,
            "shard_persistence": 0.0,
        }

        if not shard_enable or n_base <= 0:
            return {
                "n_split": 0,
                "split_mask": self._last_split_mask,
                "split_idx": torch.empty(0, dtype=torch.long, device=c.device),
            }

        crack_tips = crack_tips if crack_tips is not None else torch.zeros_like(
            c, dtype=torch.bool
        )
        crack_visited = crack_visited if crack_visited is not None else torch.zeros_like(
            c, dtype=torch.bool
        )
        fragment_ids = (
            fragment_ids
            if fragment_ids is not None
            else torch.zeros(n_base, dtype=torch.long, device=c.device)
        )

        c_norm = ((c - self.split_threshold) / max(1.0 - self.split_threshold, 1e-6)).clamp(0.0, 1.0)
        a_scale = torch.quantile(a.detach(), 0.85).clamp(min=1e-6)
        a_norm = (a / a_scale).clamp(0.0, 1.0)
        candidate_mask = (c > self.split_threshold) & (a > 1e-4)
        if material_family == "sharp_brittle":
            candidate_mask &= (crack_visited | crack_tips | (fragment_ids > 0))
        elif material_family == "rough_quasi_brittle":
            candidate_mask &= ((fragment_ids > 0) | crack_visited | (a > 0.55 * a_scale))
        else:
            candidate_mask &= crack_tips

        candidate_idx = torch.where(candidate_mask)[0]
        if candidate_idx.numel() == 0:
            self._active_parent_mask.zero_()
            self._spawn_frame.fill_(-1)
            self._parent_side.zero_()
            return {
                "n_split": 0,
                "split_mask": self._last_split_mask,
                "split_idx": torch.empty(0, dtype=torch.long, device=c.device),
            }

        score = 0.55 * c_norm + 0.45 * a_norm
        score = score + 0.18 * crack_tips.float()
        score = score + 0.12 * crack_visited.float()
        score = score + 0.20 * (fragment_ids > 0).float()

        target_count = self._family_target_count(
            n_base, candidate_idx.numel(), material_family, shard_count_scale
        )
        if target_count <= 0:
            return {
                "n_split": 0,
                "split_mask": self._last_split_mask,
                "split_idx": torch.empty(0, dtype=torch.long, device=c.device),
            }

        candidate_score = score[candidate_idx]
        if candidate_idx.numel() > target_count:
            topk = torch.topk(candidate_score, k=target_count, largest=True).indices
            split_idx = candidate_idx[topk]
        else:
            split_idx = candidate_idx
        split_idx = split_idx.unique(sorted=True)

        keep_mask = self._active_parent_mask & (
            (c > 0.82 * self.split_threshold) | (a > 0.45 * a_scale)
        )
        self._active_parent_mask = keep_mask
        self._active_parent_mask[split_idx] = True
        new_idx = split_idx[self._spawn_frame[split_idx] < 0]
        if new_idx.numel() > 0:
            self._spawn_frame[new_idx] = int(frame)
            hash_val = torch.sin(new_idx.float() * 12.9898)
            self._parent_side[new_idx] = torch.where(
                hash_val >= 0.0,
                torch.ones_like(hash_val),
                -torch.ones_like(hash_val),
            )

        active_idx = torch.where(self._active_parent_mask)[0]
        if active_idx.numel() == 0:
            return {
                "n_split": 0,
                "split_mask": self._last_split_mask,
                "split_idx": torch.empty(0, dtype=torch.long, device=c.device),
            }
        self._last_split_mask[active_idx] = True

        xyz_parent = gaussians._xyz.data[:n_base][active_idx]
        n_parent = self._safe_normalize(n[active_idx])
        a_parent = a[active_idx].clamp(min=1e-4)
        side = self._parent_side[active_idx].unsqueeze(1)

        if impact_center is None:
            impact_center = xyz_parent.mean(dim=0)
        radial = self._safe_normalize(xyz_parent - impact_center.unsqueeze(0))
        tangent = self._safe_normalize(torch.cross(n_parent, radial, dim=1))
        spawn_age = (frame - self._spawn_frame[active_idx]).float().clamp(min=0.0)

        if material_family == "sharp_brittle":
            radial_mix = 0.28
            tangent_mix = 0.10
            age_gain = 0.12
            scale_mult = 0.82 / max(shard_scale_gain, 1e-3)
            opacity_mult = min(1.0, 0.92 * shard_opacity_gain)
        else:
            radial_mix = 0.46
            tangent_mix = 0.22
            age_gain = 0.20
            scale_mult = 0.94 / max(shard_scale_gain, 1e-3)
            opacity_mult = min(1.0, 0.88 * shard_opacity_gain)

        event_boost = max(float(split_event_boost), 1.0)
        offset_boost = 1.0 + 0.18 * (event_boost - 1.0)
        motion_boost = 1.0 + 0.30 * (event_boost - 1.0)
        split_offset = (
            self.split_offset_scale
            * fragment_offset_gain
            * offset_boost
            * a_parent.unsqueeze(1)
            * n_parent
            * side
        )
        debris_dir = self._safe_normalize(
            (1.0 - radial_mix - tangent_mix) * n_parent * side
            + radial_mix * radial
            + tangent_mix * tangent
        )
        debris_mag = debris_motion_gain * motion_boost * a_parent * (1.0 + age_gain * spawn_age)
        new_xyz = xyz_parent + split_offset + debris_dir * debris_mag.unsqueeze(1)

        new_features_dc = gaussians._features_dc.data[:n_base][active_idx].clone()
        new_features_rest = gaussians._features_rest.data[:n_base][active_idx].clone()
        new_opacity = gaussians._opacity.data[:n_base][active_idx].clone()
        new_scaling = gaussians._scaling.data[:n_base][active_idx].clone()
        new_rotation = gaussians._rotation.data[:n_base][active_idx].clone()

        new_scaling = new_scaling + math.log(max(scale_mult, 0.12))
        prob = torch.sigmoid(new_opacity)
        new_prob = (prob * opacity_mult).clamp(1e-6, 1.0 - 1e-6)
        new_opacity = torch.log(new_prob / (1.0 - new_prob))
        darken = 1.0 - debris_darkening
        if material_family == "sharp_brittle":
            new_features_dc[:, 0, :] = new_features_dc[:, 0, :] * darken + 0.04
        else:
            new_features_dc[:, 0, :] = new_features_dc[:, 0, :] * max(darken - 0.06, 0.45)

        self._append_param_data(gaussians._xyz, new_xyz)
        self._append_param_data(gaussians._features_dc, new_features_dc)
        self._append_param_data(gaussians._features_rest, new_features_rest)
        self._append_param_data(gaussians._opacity, new_opacity)
        self._append_param_data(gaussians._scaling, new_scaling)
        self._append_param_data(gaussians._rotation, new_rotation)

        gap = (new_xyz - xyz_parent).norm(dim=1)
        base_scale = torch.exp(gaussians._scaling.data[:n_base][active_idx]).mean(dim=1)
        gap_visibility = (gap / (base_scale + 1e-6)).mean().item()
        shell_contrast = (
            (gaussians._features_dc.data[:n_base][active_idx, 0, :]
             - new_features_dc[:, 0, :]).abs().mean().item()
        )
        shard_persistence = spawn_age.mean().item() + 1.0
        self._last_append_parent_idx = active_idx
        self._last_metrics = {
            "visible_shard_count": int(active_idx.numel()),
            "split_gap_visibility": float(gap_visibility),
            "fragment_shell_contrast": float(shell_contrast),
            "shard_persistence": float(shard_persistence),
        }

        return {
            "n_split": int(active_idx.numel()),
            "split_mask": self._last_split_mask,
            "split_idx": active_idx,
            "new_start_idx": n_base,
        }

    @staticmethod
    def _quat_to_rotmat(q: Tensor) -> Tensor:
        """Convert wxyz quaternions to rotation matrices.

        Args:
            q: (N, 4) quaternions in wxyz order

        Returns:
            R: (N, 3, 3) rotation matrices
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)

        return R

    def extend_fracture_state(
        self,
        fracture_field,
        split_info: Dict[str, Tensor],
    ) -> None:
        """Compatibility no-op: fracture state stays on the base manifold."""
        _ = (fracture_field, split_info)
