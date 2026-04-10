"""Gaussian crack visualization via AT2 phase-field damage."""

import torch
from torch import Tensor


class GaussianCrackVisualizer:
    """
    Crack visualization on Gaussian Splats using AT2 phase-field damage.

    Maps c_surface ∈ [0,1] (from AT2 PDE) to Gaussian scale and opacity.
    Polylines drive crack propagation physics; c_surface drives rendering.
    """

    def __init__(
        self,
        damage_threshold: float = 0.3,
        device: str = "cuda",
        light_dir: tuple = (0.4, 0.3, 0.8),
        crack_color: tuple = (0.6, 0.08, 0.08),
        crack_opacity_reduction: float = 0.70,
        crack_max_opening: float = 0.010,
        crack_gap_fraction: float = 0.35,
        crack_edge_darken: float = 0.75,
        crack_red_accent: float = 0.10,
        crack_tip_scale_boost: float = 0.20,
        crack_tip_opacity_boost: float = 0.10,
        material_family: str = "neutral_reference",
        crack_band_weight: float = 0.80,
        crack_visited_weight: float = 0.45,
        crack_tip_weight: float = 0.95,
        crack_core_weight: float = 1.00,
        split_gap_gain: float = 1.0,
        fragment_shell_gain: float = 1.0,
        fragment_contrast_gain: float = 1.0,
        debris_darkening: float = 0.20,
        shard_scale_gain: float = 1.0,
        shard_opacity_gain: float = 1.0,
        damage_scale_shrink: float = 0.50,
        damage_center_opacity_reduction: float = 0.70,
        diffuse_damage_strength: float = 0.12,
    ):
        self.damage_threshold = damage_threshold
        self.device = device
        self.crack_opacity_reduction = crack_opacity_reduction
        self.crack_max_opening = crack_max_opening
        self.crack_gap_fraction = crack_gap_fraction
        self.crack_edge_darken = crack_edge_darken
        self.crack_red_accent = crack_red_accent
        self.crack_tip_scale_boost = crack_tip_scale_boost
        self.crack_tip_opacity_boost = crack_tip_opacity_boost
        self.material_family = str(material_family)
        self.crack_band_weight = float(crack_band_weight)
        self.crack_visited_weight = float(crack_visited_weight)
        self.crack_tip_weight = float(crack_tip_weight)
        self.crack_core_weight = float(crack_core_weight)
        self.split_gap_gain = float(split_gap_gain)
        self.fragment_shell_gain = float(fragment_shell_gain)
        self.fragment_contrast_gain = float(fragment_contrast_gain)
        self.debris_darkening = float(debris_darkening)
        self.shard_scale_gain = float(shard_scale_gain)
        self.shard_opacity_gain = float(shard_opacity_gain)
        self.damage_scale_shrink = float(damage_scale_shrink)
        self.damage_center_opacity_reduction = float(damage_center_opacity_reduction)
        self.diffuse_damage_strength = float(diffuse_damage_strength)
        self.crack_color = torch.tensor(crack_color, dtype=torch.float32, device=device)

        # Light direction for dynamic diffuse shading
        ld = torch.tensor(light_dir, dtype=torch.float32, device=device)
        self.light_dir = ld / ld.norm()

        self._original_dc = None
        self._original_rest = None
        self._original_opacity = None
        self._original_scaling = None
        self._original_rotation = None
        self._base_count = None
        self._initial_normals = None  # stored on first call

        print(f"[GaussianCrackVisualizer] Initialized (AT2 damage mode)")
        print(f"  - Damage threshold: {damage_threshold}")
        print(f"  - Device: {device}")

    @torch.no_grad()
    def _apply_damage_visualization(self, gaussians, c_surface: Tensor):
        """Visualize AT2 damage field on Gaussian Splats.

        Only affects the narrow crack band. Fragments stay sharp and visible.
        - Scale shrinkage only for very high damage (crack gap)
        - Opacity reduction only at crack center (c > 0.85)
        """
        thresh = self.damage_threshold

        # Scale shrinkage: only for high-damage crack region
        crack_band = c_surface > thresh
        if crack_band.any():
            t = ((c_surface[crack_band] - thresh)
                 / (1.0 - thresh)).clamp(0.0, 1.0)
            # Moderate shrinkage (50% max) — keeps fragments solid
            scale_mult = 1.0 - self.damage_scale_shrink * (t ** 3)
            gaussians._scaling.data[crack_band] += torch.log(
                scale_mult.unsqueeze(1).clamp(min=0.05))

        # Opacity reduction: only at crack center (very high damage)
        crack_center = c_surface > 0.85
        if crack_center.any():
            t_o = ((c_surface[crack_center] - 0.85) / 0.15).clamp(0.0, 1.0)
            opacity_mult = 1.0 - self.damage_center_opacity_reduction * (t_o ** 2)
            cur_prob = torch.sigmoid(gaussians._opacity.data[crack_center])
            new_prob = (cur_prob * opacity_mult.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
            gaussians._opacity.data[crack_center] = torch.log(
                new_prob / (1.0 - new_prob))

    @torch.no_grad()
    def _apply_deformation_gradient(self, gaussians, F_per_gaussian: Tensor):
        """Apply deformation gradient to Gaussian scale and rotation.

        Polar decomposition: F = R · S
        - R → rotate Gaussian orientation (quaternion multiplication)
        - S → stretch Gaussian scales (log-space addition)

        Args:
            gaussians: 3DGS Gaussian model
            F_per_gaussian: (K, 3, 3) deformation gradient per Gaussian
        """
        K = F_per_gaussian.shape[0]
        device = F_per_gaussian.device

        # Polar decomposition via SVD: F = U * diag(sigma) * Vt
        # R = U * Vt, S = V * diag(sigma) * Vt
        U, sigma, Vt = torch.linalg.svd(F_per_gaussian)

        # Ensure proper rotation (det > 0)
        det_sign = torch.det(U @ Vt).sign().unsqueeze(-1).unsqueeze(-1)
        U_fixed = U.clone()
        U_fixed[:, :, -1] *= det_sign.squeeze(-1)
        sigma_fixed = sigma.clone()
        sigma_fixed[:, -1] *= det_sign.squeeze(-1).squeeze(-1)

        R_mat = U_fixed @ Vt  # (K, 3, 3) rotation matrices

        # Apply stretch to log-scales
        log_stretch = torch.log(sigma_fixed.clamp(min=0.1, max=3.0))  # clamp for stability
        gaussians._scaling.data += log_stretch

        # Convert R to quaternion and multiply with existing rotation
        # Batch rotation matrix to quaternion
        q_rot = self._rotmat_to_quat_batch(R_mat)  # (K, 4) wxyz

        # Hamilton product: q_rot * q_old
        q_old = gaussians._rotation.data  # (K, 4) wxyz
        q_new = self._quat_multiply(q_rot, q_old)
        gaussians._rotation.data = q_new

    @staticmethod
    def _rotmat_to_quat_batch(R: Tensor) -> Tensor:
        """Convert batch of rotation matrices to quaternions (wxyz).

        Args:
            R: (K, 3, 3) rotation matrices
        Returns:
            q: (K, 4) quaternions in wxyz order
        """
        K = R.shape[0]
        q = torch.zeros(K, 4, device=R.device, dtype=R.dtype)

        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        # Case 1: tr > 0
        mask1 = tr > 0
        if mask1.any():
            s = torch.sqrt(tr[mask1] + 1.0) * 2.0
            q[mask1, 0] = 0.25 * s
            q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
            q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
            q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

        # Case 2: R[0,0] largest
        mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        if mask2.any():
            s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
            q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
            q[mask2, 1] = 0.25 * s
            q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
            q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

        # Case 3: R[1,1] largest
        mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
        if mask3.any():
            s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
            q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
            q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
            q[mask3, 2] = 0.25 * s
            q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

        # Case 4: R[2,2] largest
        mask4 = ~mask1 & ~mask2 & ~mask3
        if mask4.any():
            s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0
            q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
            q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
            q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
            q[mask4, 3] = 0.25 * s

        # Normalize
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        return q

    @staticmethod
    def _quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
        """Hamilton product q1 * q2 (both in wxyz format)."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def _restore_base_state(self, gaussians, x_world: Tensor, preserve_original: bool) -> None:
        """Restore the base Gaussian set before applying per-frame render effects."""
        if preserve_original and self._original_dc is None:
            self._original_dc = gaussians._features_dc.data.clone()
            self._original_rest = gaussians._features_rest.data.clone()
            self._original_opacity = gaussians._opacity.data.clone()
            self._original_scaling = gaussians._scaling.data.clone()
            self._original_rotation = gaussians._rotation.data.clone()
            self._base_count = int(x_world.shape[0])

        if self._base_count is None:
            self._base_count = int(x_world.shape[0])

        base_n = int(self._base_count)
        if gaussians._xyz.data.shape[0] != base_n:
            gaussians._xyz.data = gaussians._xyz.data[:base_n].clone()
            gaussians._features_dc.data = gaussians._features_dc.data[:base_n].clone()
            gaussians._features_rest.data = gaussians._features_rest.data[:base_n].clone()
            gaussians._opacity.data = gaussians._opacity.data[:base_n].clone()
            gaussians._scaling.data = gaussians._scaling.data[:base_n].clone()
            gaussians._rotation.data = gaussians._rotation.data[:base_n].clone()

        gaussians._xyz.data = x_world
        gaussians._features_dc.data.copy_(self._original_dc)
        gaussians._features_rest.data.copy_(self._original_rest)
        gaussians._opacity.data.copy_(self._original_opacity)
        gaussians._scaling.data.copy_(self._original_scaling)
        gaussians._rotation.data.copy_(self._original_rotation)

    @torch.no_grad()
    def _apply_fragment_shell_styling(
        self,
        gaussians,
        c_surface: Tensor,
        fragment_ids: Tensor = None,
        shard_mask: Tensor = None,
    ) -> None:
        if c_surface is None:
            return
        if fragment_ids is None and shard_mask is None:
            return

        detached_mask = torch.zeros_like(c_surface, dtype=torch.bool)
        if fragment_ids is not None:
            detached_mask |= fragment_ids > 0
        if shard_mask is not None:
            detached_mask |= shard_mask
        if not bool(detached_mask.any()):
            return

        thresh = self.damage_threshold
        c_norm = ((c_surface - thresh) / max(1.0 - thresh, 1e-6)).clamp(0.0, 1.0)
        shell_strength = (0.35 + 0.65 * c_norm[detached_mask]) * self.fragment_shell_gain
        shell_strength = shell_strength.clamp(0.0, 1.0)

        darken = 1.0 - self.debris_darkening * shell_strength.unsqueeze(1)
        gaussians._features_dc.data[detached_mask, 0, :] *= darken
        gaussians._features_dc.data[detached_mask, 0, 0] += (
            0.06 * self.fragment_contrast_gain * shell_strength
        )
        gaussians._scaling.data[detached_mask] += torch.log(
            (1.0 + 0.06 * self.fragment_shell_gain * shell_strength)
            .unsqueeze(1)
            .clamp(min=0.90)
        )

        if shard_mask is not None and bool(shard_mask.any()):
            gaussians._scaling.data[shard_mask] += torch.log(
                torch.full(
                    (int(shard_mask.sum().item()), 1),
                    0.88 / max(self.shard_scale_gain, 1e-3),
                    device=gaussians._scaling.device,
                )
            )
            cur_prob = torch.sigmoid(gaussians._opacity.data[shard_mask])
            new_prob = (
                cur_prob * min(1.0, 0.92 * self.shard_opacity_gain)
            ).clamp(1e-6, 1.0 - 1e-6)
            gaussians._opacity.data[shard_mask] = torch.log(new_prob / (1.0 - new_prob))

    def set_initial_normals(self, normals: Tensor):
        """Store initial surface normals for dynamic lighting.

        Args:
            normals: (N_surf, 3) unit normals from mesh sampling
        """
        self._initial_normals = normals.to(self.device).float()

    @torch.no_grad()
    def _apply_dynamic_lighting(self, gaussians, F_per_gaussian: Tensor, c_surface: Tensor = None):
        """Recompute diffuse shading based on rotated normals.

        After fragments rotate, the baked-in shading no longer matches.
        Extract R from F via polar decomposition, rotate initial normals,
        recompute N·L, and update DC color.
        """
        if self._initial_normals is None:
            return

        N = gaussians._xyz.shape[0]
        normals = self._initial_normals[:N]

        if F_per_gaussian is not None and F_per_gaussian.shape[0] == N:
            # Polar decomposition: F = R · S → extract R
            U, sigma, Vt = torch.linalg.svd(F_per_gaussian)
            det_sign = torch.det(U @ Vt).sign().unsqueeze(-1).unsqueeze(-1)
            U_fixed = U.clone()
            U_fixed[:, :, -1] *= det_sign.squeeze(-1)
            R_mat = U_fixed @ Vt  # (N, 3, 3)

            # Rotate normals: n' = R @ n
            normals = torch.bmm(R_mat, normals.unsqueeze(-1)).squeeze(-1)
            normals = torch.nn.functional.normalize(normals, dim=-1)

        # Flip all back-facing normals toward camera (two-sided lighting)
        if hasattr(self, '_camera_pos') and self._camera_pos is not None:
            view_dir = self._camera_pos.unsqueeze(0) - gaussians._xyz.data[:N]
            view_dir = torch.nn.functional.normalize(view_dir, dim=-1)
            back_facing = (normals * view_dir).sum(dim=-1) < 0
            normals = normals.clone()
            normals[back_facing] = -normals[back_facing]

        # Diffuse: N · L
        ndotl = (normals * self.light_dir).sum(dim=-1).clamp(0.0, 1.0)

        # Ambient + diffuse → gray value [0.25, 0.85]
        gray = 0.25 + 0.6 * ndotl

        # Convert to SH DC: color = SH_C0 * dc + 0.5 → dc = (color - 0.5) / SH_C0
        SH_C0 = 0.28209479177387814
        dc_val = (gray - 0.5) / SH_C0  # (N,)

        # Update features_dc: shape (N, 1, 3)
        gaussians._features_dc.data[:, 0, :] = dc_val.unsqueeze(-1).expand(-1, 3)

    def update_gaussians(
        self,
        gaussians,
        c_surface: Tensor,
        x_world: Tensor,
        preserve_original: bool = True,
        debris_mask: Tensor = None,
        F_per_gaussian: Tensor = None,
        camera_pos: Tensor = None,
        crack_normals: Tensor = None,
        crack_opening: Tensor = None,
        crack_tips: Tensor = None,
        crack_visited: Tensor = None,
        fragment_ids: Tensor = None,
        shard_mask: Tensor = None,
    ):
        """Update Gaussian properties each frame.

        Args:
            gaussians:        3DGS Gaussian model
            c_surface:        (N_surf,) AT2 damage values ∈ [0,1]
            x_world:          (N_surf, 3) current surface positions in world space
            preserve_original: cache original Gaussian properties on first call
            debris_mask:      (N_surf,) bool — small fragment Gaussians to hide
            F_per_gaussian:   (N_surf, 3, 3) deformation gradient per Gaussian
            crack_normals:    (N_surf, 3) crack normal directions (from manifold fracture)
            crack_opening:    (N_surf,) crack opening magnitudes (from manifold fracture)
        """
        # Store camera position for back-face normal flipping
        self._camera_pos = camera_pos

        self._restore_base_state(gaussians, x_world, preserve_original)

        # Apply deformation gradient to scale and rotation
        if F_per_gaussian is not None:
            self._apply_deformation_gradient(gaussians, F_per_gaussian)

        # Dynamic lighting: recompute shading from rotated normals
        self._apply_dynamic_lighting(gaussians, F_per_gaussian, c_surface)

        # Debris: mild shrinkage + darkening (keep visible)
        if debris_mask is not None and debris_mask.any():
            gaussians._scaling.data[debris_mask] -= 0.25 * self.fragment_shell_gain
            gaussians._features_dc.data[debris_mask] *= max(0.35, 1.0 - self.debris_darkening)

        # Manifold fracture visualization (crack normals + opening)
        if crack_normals is not None and crack_opening is not None:
            if self.material_family == "diffuse_damage":
                self._apply_damage_visualization(gaussians, c_surface)
            else:
                self._apply_manifold_crack_visualization(
                    gaussians, c_surface, crack_normals, crack_opening,
                    crack_tips=crack_tips, crack_visited=crack_visited)
        else:
            # Legacy: scalar damage visualization
            has_damage = (c_surface is not None
                          and c_surface.max() > self.damage_threshold)
            if has_damage:
                self._apply_damage_visualization(gaussians, c_surface)

        self._apply_fragment_shell_styling(
            gaussians,
            c_surface,
            fragment_ids=fragment_ids,
            shard_mask=shard_mask,
        )

    @torch.no_grad()
    def _apply_manifold_crack_visualization(
        self,
        gaussians,
        c_surface: Tensor,
        crack_normals: Tensor,
        crack_opening: Tensor,
        crack_tips: Tensor = None,
        crack_visited: Tensor = None,
    ):
        """Visualize cracks using manifold fracture state.

        Uses crack normal to flatten Gaussians along the crack direction
        and crack opening to create visible gap geometry.

        Args:
            gaussians: 3DGS GaussianModel
            c_surface: (N,) damage values
            crack_normals: (N, 3) crack normal directions
            crack_opening: (N,) opening magnitudes
        """
        if c_surface is None:
            return

        N = c_surface.shape[0]
        thresh = self.damage_threshold
        crack_tips = crack_tips if crack_tips is not None else torch.zeros(
            N, dtype=torch.bool, device=c_surface.device)
        crack_visited = crack_visited if crack_visited is not None else torch.zeros(
            N, dtype=torch.bool, device=c_surface.device)
        crack_band = c_surface > thresh
        c_norm = ((c_surface - thresh) / max(1.0 - thresh, 1e-6)).clamp(0.0, 1.0)
        tip_strength = crack_tips.float() * self.crack_tip_weight
        visited_strength = crack_visited.float() * self.crack_visited_weight
        core_strength = (c_surface > max(thresh, 0.55)).float() * self.crack_core_weight
        band_strength = c_norm * self.crack_band_weight
        shell_strength = torch.maximum(
            torch.maximum(band_strength, visited_strength),
            torch.maximum(tip_strength, core_strength),
        )
        crack_shell = shell_strength > 0.12

        # --- 1. Covariance flattening along crack normal ---
        flatten_mask = crack_shell
        if flatten_mask.any() and crack_normals is not None:
            t = shell_strength[flatten_mask].clamp(0.0, 1.0)
            tip_boost = crack_tips[flatten_mask].float() * self.crack_tip_scale_boost
            flatten_factor = 1.0 - (0.48 + tip_boost) * (t.clamp(min=0.12) ** 2)
            flatten_factor = flatten_factor.clamp(min=0.08)

            # Get Gaussian local frame
            q = gaussians._rotation.data[flatten_mask]
            R = self._quat_to_rotmat_batch(q)  # (M, 3, 3)

            # Project crack normal into local frame
            n_local = torch.bmm(
                R.transpose(1, 2),
                crack_normals[flatten_mask].unsqueeze(2)
            ).squeeze(2)  # (M, 3)

            # Scale reduction proportional to alignment with crack normal
            alignment = n_local.abs()  # (M, 3)
            scale_reduction = torch.log(
                flatten_factor.unsqueeze(1).clamp(min=0.05)) * alignment
            gaussians._scaling.data[flatten_mask] += scale_reduction

        # --- 2. Position offset along crack normal (opening) ---
        open_mask = ((c_surface > 0.35) | crack_tips) & (crack_opening > 1e-4)
        if open_mask.any() and crack_normals is not None:
            # Offset Gaussians slightly along crack normal
            # Creates a visible gap effect
            opening_mag = crack_opening[open_mask].clamp(min=0.0, max=self.crack_max_opening)
            tip_boost = 1.0 + crack_tips[open_mask].float() * (0.30 + self.crack_tip_scale_boost)
            offset = crack_normals[open_mask] * opening_mag.unsqueeze(1) * tip_boost.unsqueeze(1)
            # Alternate sign based on position hash for two-sided opening
            pos_hash = gaussians._xyz.data[open_mask].sum(dim=1)
            sign = torch.where(pos_hash.frac() > 0.5,
                               torch.ones_like(pos_hash),
                               -torch.ones_like(pos_hash))
            gaussians._xyz.data[open_mask] += (
                offset
                * sign.unsqueeze(1)
                * self.crack_gap_fraction
                * self.split_gap_gain
            )

        # --- 2.5. Crack-path darkening / tint ---
        if crack_shell.any():
            tint_strength = shell_strength[crack_shell].clamp(0.0, 1.0)
            cur = gaussians._features_dc.data[crack_shell, 0, :]
            darken = 1.0 - (
                self.crack_edge_darken
                * self.fragment_contrast_gain
                * tint_strength.unsqueeze(1)
            )
            tinted = cur * darken
            color_bias = self.crack_color.unsqueeze(0) * (
                self.crack_red_accent
                * self.fragment_contrast_gain
                * tint_strength.unsqueeze(1)
            )
            gaussians._features_dc.data[crack_shell, 0, :] = tinted + color_bias

        # --- 3. Opacity reduction at crack center ---
        crack_center = (c_surface > 0.65) | crack_tips
        if crack_center.any():
            t_o = shell_strength[crack_center].clamp(0.0, 1.0)
            tip_boost = crack_tips[crack_center].float() * self.crack_tip_opacity_boost
            opacity_mult = 1.0 - (self.crack_opacity_reduction + tip_boost) * (t_o.clamp(min=0.15) ** 2)
            opacity_mult = opacity_mult.clamp(min=0.05, max=1.0)
            cur_prob = torch.sigmoid(gaussians._opacity.data[crack_center])
            new_prob = (cur_prob * opacity_mult.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
            gaussians._opacity.data[crack_center] = torch.log(
                new_prob / (1.0 - new_prob))

    @staticmethod
    def _quat_to_rotmat_batch(q: Tensor) -> Tensor:
        """Convert wxyz quaternions to rotation matrices."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)
