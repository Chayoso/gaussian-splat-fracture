"""
Gaussian Splitter

Phase 3: Split high-damage Gaussians into two halves along the crack normal,
creating visible crack openings. Also handles covariance deformation for
partially-damaged Gaussians.

This makes cracks visible as actual geometric openings rather than just
opacity reduction or darkening.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple, Dict
import math


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
        """
        Args:
            split_threshold: damage above which Gaussians are split
            opacity_threshold: damage above which opacity is reduced
            flatten_threshold: damage above which covariance is flattened
            max_flatten: maximum scale reduction along crack normal
            max_opacity_reduction: maximum opacity reduction factor
            split_offset_scale: split offset as multiple of Gaussian scale
            device: torch device
        """
        self.split_threshold = split_threshold
        self.opacity_threshold = opacity_threshold
        self.flatten_threshold = flatten_threshold
        self.max_flatten = max_flatten
        self.max_opacity_reduction = max_opacity_reduction
        self.split_offset_scale = split_offset_scale
        self.device = torch.device(device)

        # Track which Gaussians have been split (to avoid re-splitting)
        self._split_mask: Optional[Tensor] = None
        self._original_count: int = 0

    def apply_deformation(
        self,
        gaussians,
        c: Tensor,
        n: Tensor,
        a: Tensor,
        F_per_gaussian: Optional[Tensor] = None,
    ) -> None:
        """
        Apply crack-induced deformation to Gaussians (in-place).

        This modifies Gaussian scale, rotation, and opacity based on
        the fracture state. Does NOT split Gaussians (see split_gaussians).

        Args:
            gaussians: 3DGS GaussianModel
            c: (N,) damage per Gaussian
            n: (N, 3) crack normal per Gaussian
            a: (N,) opening magnitude per Gaussian
            F_per_gaussian: (N, 3, 3) deformation gradient (optional)
        """
        N = c.shape[0]

        # 1. Covariance flattening along crack normal
        flatten_mask = c > self.flatten_threshold
        if flatten_mask.any():
            self._flatten_along_normal(gaussians, c, n, flatten_mask)

        # 2. Opacity reduction at crack center
        opacity_mask = c > self.opacity_threshold
        if opacity_mask.any():
            self._reduce_opacity(gaussians, c, opacity_mask)

    def _flatten_along_normal(
        self,
        gaussians,
        c: Tensor,
        n: Tensor,
        mask: Tensor,
    ) -> None:
        """
        Flatten Gaussian covariance along crack normal direction.

        Projects the Gaussian scale onto the crack normal and reduces
        it proportional to damage. This makes the crack appear as a
        thin gap rather than a blurry region.
        """
        # Damage intensity for flattened Gaussians
        t = ((c[mask] - self.flatten_threshold)
             / (1.0 - self.flatten_threshold)).clamp(0.0, 1.0)
        flatten_factor = 1.0 - self.max_flatten * (t ** 2)  # (M,)

        # Get current log-scale and rotation
        log_scale = gaussians._scaling.data[mask]   # (M, 3)
        q = gaussians._rotation.data[mask]           # (M, 4) wxyz

        # Convert quaternion to rotation matrix
        R = self._quat_to_rotmat(q)  # (M, 3, 3)

        # Project crack normal into Gaussian local frame
        n_local = torch.bmm(R.transpose(1, 2), n[mask].unsqueeze(2)).squeeze(2)  # (M, 3)

        # Find which local axis is most aligned with crack normal
        alignment = n_local.abs()  # (M, 3)

        # Reduce scale along the most-aligned axis
        # Soft blending: scale reduction proportional to alignment
        scale_reduction = torch.log(flatten_factor.unsqueeze(1)) * alignment  # (M, 3)
        gaussians._scaling.data[mask] = log_scale + scale_reduction

    def _reduce_opacity(
        self,
        gaussians,
        c: Tensor,
        mask: Tensor,
    ) -> None:
        """Reduce opacity for high-damage Gaussians."""
        t = ((c[mask] - self.opacity_threshold)
             / (1.0 - self.opacity_threshold)).clamp(0.0, 1.0)
        opacity_mult = 1.0 - self.max_opacity_reduction * (t ** 2)

        # Apply in logit space
        cur_logit = gaussians._opacity.data[mask]
        cur_prob = torch.sigmoid(cur_logit)
        new_prob = (cur_prob * opacity_mult.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
        gaussians._opacity.data[mask] = torch.log(new_prob / (1.0 - new_prob))

    def split_gaussians(
        self,
        gaussians,
        c: Tensor,
        n: Tensor,
        a: Tensor,
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
        N_orig = c.shape[0]
        split_mask = (c > self.split_threshold) & (a > 1e-4)

        # Track previously split Gaussians to avoid double-splitting
        if self._split_mask is not None and self._split_mask.shape[0] == N_orig:
            split_mask = split_mask & ~self._split_mask

        n_split = split_mask.sum().item()
        if n_split == 0:
            return {'n_split': 0, 'split_mask': split_mask}

        split_idx = torch.where(split_mask)[0]

        # Get properties of Gaussians to split
        xyz = gaussians._xyz.data[split_idx]          # (M, 3)
        normals = n[split_idx]                          # (M, 3)
        opening = a[split_idx]                          # (M,)

        # Compute offset along crack normal
        offset = normals * opening.unsqueeze(1) * self.split_offset_scale  # (M, 3)

        # Create two halves: shift along +n and -n
        xyz_plus = xyz + offset
        xyz_minus = xyz - offset

        # Update positions: replace originals with one half, append the other
        gaussians._xyz.data[split_idx] = xyz_plus

        # Clone all properties for the new half
        new_xyz = xyz_minus
        new_features_dc = gaussians._features_dc.data[split_idx].clone()
        new_features_rest = gaussians._features_rest.data[split_idx].clone()
        new_opacity = gaussians._opacity.data[split_idx].clone()
        new_scaling = gaussians._scaling.data[split_idx].clone()
        new_rotation = gaussians._rotation.data[split_idx].clone()

        # Reduce scale perpendicular to crack for both halves
        # (makes the crack gap visible)
        scale_reduction = torch.log(torch.tensor(0.7, device=self.device))
        gaussians._scaling.data[split_idx] += scale_reduction
        new_scaling += scale_reduction

        # Reduce opacity for both halves (crack edge effect)
        opacity_factor = 0.85
        for logit in [gaussians._opacity.data[split_idx], new_opacity]:
            prob = torch.sigmoid(logit)
            new_prob = (prob * opacity_factor).clamp(1e-6, 1 - 1e-6)
            logit.copy_(torch.log(new_prob / (1.0 - new_prob)))

        # Append new Gaussians
        gaussians._xyz.data = torch.cat([gaussians._xyz.data, new_xyz], dim=0)
        gaussians._features_dc.data = torch.cat(
            [gaussians._features_dc.data, new_features_dc], dim=0)
        gaussians._features_rest.data = torch.cat(
            [gaussians._features_rest.data, new_features_rest], dim=0)
        gaussians._opacity.data = torch.cat(
            [gaussians._opacity.data, new_opacity], dim=0)
        gaussians._scaling.data = torch.cat(
            [gaussians._scaling.data, new_scaling], dim=0)
        gaussians._rotation.data = torch.cat(
            [gaussians._rotation.data, new_rotation], dim=0)

        # Update split tracking
        N_new = gaussians._xyz.data.shape[0]
        new_split_mask = torch.zeros(N_new, dtype=torch.bool, device=self.device)
        if self._split_mask is not None and self._split_mask.shape[0] == N_orig:
            new_split_mask[:N_orig] = self._split_mask
        new_split_mask[split_idx] = True
        new_split_mask[N_orig:] = True
        self._split_mask = new_split_mask

        print(f"[Splitter] Split {n_split} Gaussians → "
              f"{N_orig} → {N_new} total")

        return {
            'n_split': n_split,
            'split_mask': split_mask,
            'split_idx': split_idx,
            'new_start_idx': N_orig,
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
        """
        Extend fracture field state arrays after Gaussian split.

        New Gaussians inherit their parent's fracture state.

        Args:
            fracture_field: GaussianFractureField instance
            split_info: output from split_gaussians()
        """
        n_split = split_info['n_split']
        if n_split == 0:
            return

        split_idx = split_info['split_idx']

        # Extend all state tensors by appending copies from split parents
        fracture_field.c = torch.cat([fracture_field.c,
                                       fracture_field.c[split_idx]])
        fracture_field.H = torch.cat([fracture_field.H,
                                       fracture_field.H[split_idx]])
        fracture_field.n = torch.cat([fracture_field.n,
                                       fracture_field.n[split_idx]])
        fracture_field.a = torch.cat([fracture_field.a,
                                       fracture_field.a[split_idx]])
        fracture_field.f = torch.cat([fracture_field.f,
                                       fracture_field.f[split_idx]])

        if hasattr(fracture_field, "crack_front"):
            crack_front = fracture_field.crack_front
            if crack_front.tip_mask is not None:
                crack_front.tip_mask = torch.cat(
                    [crack_front.tip_mask, crack_front.tip_mask[split_idx]])
                crack_front.visited_mask = torch.cat(
                    [crack_front.visited_mask, crack_front.visited_mask[split_idx]])
                crack_front.parent_index = torch.cat(
                    [crack_front.parent_index, crack_front.parent_index[split_idx]])
                crack_front.tip_age = torch.cat(
                    [crack_front.tip_age, crack_front.tip_age[split_idx]])
                crack_front.growth_dir = torch.cat(
                    [crack_front.growth_dir, crack_front.growth_dir[split_idx]])
