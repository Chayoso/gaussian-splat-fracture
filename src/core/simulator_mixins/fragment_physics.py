"""FragmentPhysicsMixin for ManifoldSimulator.

This module is behavior-preserving extraction from manifold_simulator.py.
"""

from typing import Optional

import torch
from torch import Tensor


class FragmentPhysicsMixin:
    def _ensure_physical_fragment_registry(self, count: int, device) -> None:
        if (self._physical_fragment_labels is not None
                and self._physical_fragment_labels.shape[0] == count
                and self._physical_fragment_labels.device == device):
            return
        self._physical_fragment_labels = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self._physical_fragment_states = {}
        self._next_physical_fragment_id = 1

    def _update_physical_fragment_registry(self, raw_particle_labels: Tensor) -> Tensor:
        """Persist released particle chunks even if graph labels later merge back."""
        self._ensure_physical_fragment_registry(raw_particle_labels.shape[0], raw_particle_labels.device)
        persistent = self._physical_fragment_labels.clone()
        active_raw = raw_particle_labels.unique(sorted=True)

        for raw_id in active_raw.tolist():
            if raw_id <= 0:
                continue
            mask = raw_particle_labels == raw_id
            raw_size = int(mask.sum().item())
            if raw_size < self.fragment_physical_min_size:
                continue

            overlap_vals, overlap_counts = persistent[mask].unique(return_counts=True)
            persistent_label = None
            if overlap_vals.numel() > 0:
                overlap_mask = overlap_vals > 0
                if bool(overlap_mask.any()):
                    overlap_vals = overlap_vals[overlap_mask]
                    overlap_counts = overlap_counts[overlap_mask]
                    best_idx = int(torch.argmax(overlap_counts).item())
                    best_count = int(overlap_counts[best_idx].item())
                    if best_count / max(raw_size, 1) >= self.fragment_physical_overlap_threshold:
                        persistent_label = int(overlap_vals[best_idx].item())

            if persistent_label is None:
                persistent_label = self._next_physical_fragment_id
                self._next_physical_fragment_id += 1
                base_mask = raw_particle_labels == 0
                if bool(base_mask.any()):
                    birth_base_com = self.mapper.mpm_to_world(
                        self.x_mpm[base_mask]
                    ).mean(dim=0).detach().clone()
                else:
                    birth_base_com = self.mapper.mpm_to_world(
                        self.x_mpm
                    ).mean(dim=0).detach().clone()
                birth_com = self.mapper.mpm_to_world(
                    self.x_mpm[mask]
                ).mean(dim=0).detach().clone()
                self._physical_fragment_states[persistent_label] = {
                    "age": 0,
                    "release_score": 0.0,
                    "support_lost": False,
                    "birth_com": birth_com,
                    "birth_base_com": birth_base_com,
                    "release_dir": None,
                    "spin_axis": None,
                }

            persistent[mask] = persistent_label
            state = self._physical_fragment_states.get(
                persistent_label,
                {"age": 0, "release_score": 0.0, "support_lost": False},
            )
            release_score = 0.0
            support_lost = False
            if (self.fragment_manager is not None
                    and raw_id < len(self.fragment_manager.fragment_release_scores)):
                release_score = float(self.fragment_manager.fragment_release_scores[raw_id])
            if (self.fragment_manager is not None
                    and raw_id < len(self.fragment_manager.fragment_support_lost)):
                support_lost = bool(self.fragment_manager.fragment_support_lost[raw_id])
            meta = (
                self.fragment_manager.detached_fragment_meta.get(int(raw_id), {})
                if self.fragment_manager is not None else {}
            )
            pseudo_mass = float(meta.get("pseudo_thickness_mass", raw_size))
            phase_score = float(meta.get("phase_score", 0.0))
            state["release_score"] = max(float(state.get("release_score", 0.0)), release_score)
            state["support_lost"] = bool(state.get("support_lost", False) or support_lost)
            state["pseudo_thickness_mass"] = max(
                float(state.get("pseudo_thickness_mass", 0.0)),
                pseudo_mass,
            )
            state["phase_score"] = max(float(state.get("phase_score", 0.0)), phase_score)
            state["age"] = int(state.get("age", 0)) + 1
            self._physical_fragment_states[persistent_label] = state

        if self.crack_connected_release_only:
            for label in persistent.unique(sorted=True).tolist():
                if label <= 0:
                    continue
                mask = persistent == label
                if int(mask.sum().item()) < self.fragment_physical_min_size:
                    persistent[mask] = 0
                    self._physical_fragment_states.pop(int(label), None)

        self._physical_fragment_labels = persistent
        if bool((persistent > 0).any()):
            return persistent
        return raw_particle_labels

    def _step_fragmented_physics(self, stress: Tensor, dt: float):
        """Per-fragment MPM physics."""
        # Map Gaussian fragments back to MPM particles
        surf_frag_ids = self.fragment_manager.fragment_ids
        raw_mpm_frag_ids = self._map_surface_labels_to_particles(surf_frag_ids)
        mpm_frag_ids = self._update_physical_fragment_registry(raw_mpm_frag_ids)

        # Per-fragment P2G2P
        for frag_id in mpm_frag_ids.unique(sorted=True).tolist():
            frag_mask = mpm_frag_ids == frag_id
            frag_idx = torch.where(frag_mask)[0]
            if len(frag_idx) < 10:
                self.v_mpm[frag_idx] += dt * self.mpm.gravity.unsqueeze(0)
                self.x_mpm[frag_idx] += self.v_mpm[frag_idx] * dt
                self.x_mpm[frag_idx] = self.x_mpm[frag_idx].clamp(
                    self.mpm.clip_bound, 1.0 - self.mpm.clip_bound)
                continue
            self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p_subset(
                self.x_mpm, self.v_mpm, self.C, self.F, stress, frag_idx)

        self._apply_physical_fragment_release_drift(mpm_frag_ids, dt)
        self.mpm.time += dt

    def _apply_physical_fragment_release_drift(self, mpm_frag_ids: Tensor, dt: float) -> None:
        """Apply a small physical gap / release drift to support-lost fragments."""
        if self.fragment_manager is None or self.fragment_manager.n_fragments <= 1:
            return
        if self.fragment_physical_release_frames <= 0:
            return
        if self.fragment_physical_gap_scale <= 0.0 and self.fragment_physical_release_velocity <= 0.0:
            return
        if self._fragment_activation_frame < 0:
            return

        frames_since = max(self.frame_count - self._fragment_activation_frame, 0)
        if frames_since > self.fragment_physical_release_frames:
            return
        taper = 1.0 - (frames_since / max(float(self.fragment_physical_release_frames), 1.0))
        base_mask = mpm_frag_ids == 0
        if not bool(base_mask.any()):
            return
        base_com = self.x_mpm[base_mask].mean(dim=0)
        strict_mode = bool(self.crack_connected_release_only)
        release_gate = 0.58 if strict_mode else 0.72

        for frag_id in mpm_frag_ids.unique(sorted=True).tolist():
            if frag_id <= 0:
                continue
            mask = mpm_frag_ids == frag_id
            if int(mask.sum().item()) < 8:
                continue
            state = self._physical_fragment_states.get(frag_id, {})
            release_score = float(state.get("release_score", 0.0))
            support_lost = bool(state.get("support_lost", False))
            if not support_lost and release_score < release_gate:
                continue

            frag_com = self.x_mpm[mask].mean(dim=0)
            direction = frag_com - base_com
            lateral = direction.clone()
            lateral[2] = 0.0
            if float(lateral.norm().item()) < 1e-8 and hasattr(self, "_impact_center"):
                lateral = frag_com - self._impact_center.to(frag_com.device)
                lateral[2] = 0.0
            lateral = self._safe_vector_normalize(lateral.unsqueeze(0)).squeeze(0)
            if strict_mode:
                direction[2] -= 0.35 * self.fragment_physical_downward_bias * (
                    0.40 + 0.60 * release_score
                )
            else:
                direction[2] -= self.fragment_physical_downward_bias * (0.65 + 0.85 * release_score)
            direction = self._safe_vector_normalize(direction.unsqueeze(0)).squeeze(0)
            if float(direction.norm().item()) < 1e-8:
                direction = torch.tensor([0.0, 0.0, -1.0], device=self.x_mpm.device)
            lateral_bias = max(0.0, min(float(self.fragment_physical_lateral_bias), 0.85))
            if lateral_bias > 0.0 and float(lateral.norm().item()) > 1e-8:
                direction = self._safe_vector_normalize(
                    ((1.0 - lateral_bias) * direction + lateral_bias * lateral).unsqueeze(0)
                ).squeeze(0)
            stored_dir = state.get("release_dir")
            if stored_dir is None:
                state["release_dir"] = direction.detach().clone()
            else:
                direction = self._safe_vector_normalize(
                    (0.86 * stored_dir.to(direction.device) + 0.14 * direction).unsqueeze(0)
                ).squeeze(0)
                state["release_dir"] = direction.detach().clone()

            up = torch.tensor([0.0, 0.0, 1.0], device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            spin_axis = state.get("spin_axis")
            if spin_axis is None:
                spin_axis = torch.cross(direction, up, dim=0)
                if float(spin_axis.norm().item()) < 1e-8:
                    spin_axis = torch.tensor([1.0, 0.0, 0.0], device=self.x_mpm.device, dtype=self.x_mpm.dtype)
                spin_axis = self._safe_vector_normalize(spin_axis.unsqueeze(0)).squeeze(0)
                state["spin_axis"] = spin_axis.detach().clone()
            else:
                spin_axis = self._safe_vector_normalize(spin_axis.to(self.x_mpm.device).unsqueeze(0)).squeeze(0)

            gap_step = self.fragment_physical_gap_scale * taper * (1.0 + 1.25 * release_score)
            vel_step = self.fragment_physical_release_velocity * taper * (0.55 + 0.95 * release_score)
            pseudo_mass = max(float(state.get("pseudo_thickness_mass", int(mask.sum().item()))), 1.0)
            mass_scale = max(0.62, min((float(mask.sum().item()) / pseudo_mass) ** 0.25, 1.18))
            gap_step *= mass_scale
            vel_step *= mass_scale
            if strict_mode:
                gap_step *= 1.35
                vel_step *= 0.55
            if gap_step > 0.0:
                self.x_mpm[mask] = self.x_mpm[mask] + direction.unsqueeze(0) * gap_step
            if vel_step > 0.0:
                self.v_mpm[mask] = self.v_mpm[mask] + direction.unsqueeze(0) * vel_step
            spin_gain = max(0.0, float(self.fragment_physical_spin_gain))
            if spin_gain > 0.0 and (gap_step > 0.0 or vel_step > 0.0):
                rel = self.x_mpm[mask] - frag_com.unsqueeze(0)
                radius = rel.norm(dim=1).quantile(0.75).clamp(min=1e-6)
                spin_speed = spin_gain * taper * (0.35 * gap_step / max(float(dt), 1e-8) + vel_step)
                spin_vel = torch.cross(
                    spin_axis.unsqueeze(0).expand_as(rel),
                    rel,
                    dim=1,
                ) * (spin_speed / radius)
                max_spin = max(0.0, float(self.fragment_physical_max_speed)) * 0.45
                if max_spin > 0.0:
                    spin_norm = spin_vel.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    spin_vel = spin_vel * torch.clamp(max_spin / spin_norm, max=1.0)
                self.v_mpm[mask] = self.v_mpm[mask] + spin_vel
            max_speed = max(0.0, float(self.fragment_physical_max_speed))
            if max_speed > 0.0:
                speeds = self.v_mpm[mask].norm(dim=1, keepdim=True).clamp(min=1e-8)
                self.v_mpm[mask] = self.v_mpm[mask] * torch.clamp(max_speed / speeds, max=1.0)
            self._physical_fragment_states[frag_id] = state

        self.x_mpm = self.x_mpm.clamp(self.mpm.clip_bound, 1.0 - self.mpm.clip_bound)

    def _current_mpm_fragment_labels(self) -> Optional[Tensor]:
        if (
            self._physical_fragment_labels is not None
            and self._physical_fragment_labels.shape[0] == self.x_mpm.shape[0]
            and bool((self._physical_fragment_labels > 0).any())
        ):
            return self._physical_fragment_labels
        if (
            self.fragment_manager is not None
            and getattr(self.fragment_manager, "fragment_ids", None) is not None
            and bool((self.fragment_manager.fragment_ids > 0).any())
        ):
            return self._map_surface_labels_to_particles(self.fragment_manager.fragment_ids)
        return None

    @staticmethod
    def _component_angular_velocity(rel: Tensor, vel_rel: Tensor) -> Tensor:
        device = rel.device
        dtype = rel.dtype
        eye = torch.eye(3, device=device, dtype=dtype)
        rr = (rel * rel).sum(dim=1)
        inertia = (rr.sum() * eye) - rel.transpose(0, 1) @ rel
        inertia = inertia + 1e-6 * max(float(rel.shape[0]), 1.0) * eye
        angular_momentum = torch.cross(rel, vel_rel, dim=1).sum(dim=0)
        try:
            return torch.linalg.solve(inertia, angular_momentum)
        except RuntimeError:
            return torch.zeros(3, device=device, dtype=dtype)

    def _apply_rigid_contact_impulse(
        self,
        current: Tensor,
        velocity: Tensor,
        com: Tensor,
        v_com: Tensor,
        omega: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not (
            self.rigid_contact_enabled
            and self._gravity_drop
            and self._gravity_drop_contacted
        ):
            return v_com, omega

        ground_z = float(self._gravity_drop_ground_z)
        band = max(float(self.rigid_contact_band) * float(self.mpm.dx), 1e-4)
        contact = current[:, 2] <= (ground_z + band)
        if not bool(contact.any()):
            return v_com, omega

        n = torch.tensor([0.0, 0.0, 1.0], device=current.device, dtype=current.dtype)
        contact_point = current[contact].mean(dim=0)
        r = contact_point - com
        v_contact = v_com + torch.cross(omega, r, dim=0)
        vn = torch.minimum(v_contact.dot(n), v_com.dot(n))
        if float(vn.item()) >= -1e-5:
            return v_com, omega

        rel = current - com.unsqueeze(0)
        eye = torch.eye(3, device=current.device, dtype=current.dtype)
        rr = (rel * rel).sum(dim=1)
        inertia = (rr.sum() * eye) - rel.transpose(0, 1) @ rel
        inertia = inertia + 1e-6 * max(float(current.shape[0]), 1.0) * eye
        rn = torch.cross(r, n, dim=0)
        try:
            inv_i_rn = torch.linalg.solve(inertia, rn)
        except RuntimeError:
            inv_i_rn = torch.zeros_like(rn)

        mass = max(float(current.shape[0]), 1.0)
        angular_term = n.dot(torch.cross(inv_i_rn, r, dim=0))
        denom = (1.0 / mass) + max(float(angular_term.item()), 0.0) + 1e-6
        restitution = max(0.0, min(float(self.rigid_contact_restitution), 0.85))
        impulse_mag = float((-(1.0 + restitution) * vn / denom).item())
        impulse = impulse_mag * n
        v_com = v_com + impulse / mass
        try:
            delta_omega = torch.linalg.solve(inertia, torch.cross(r, impulse, dim=0))
        except RuntimeError:
            delta_omega = torch.zeros_like(omega)
        omega = omega + self.rigid_contact_angular_gain * delta_omega

        tangent = v_contact - v_contact.dot(n) * n
        tangent_norm = float(tangent.norm().item())
        if tangent_norm > 1e-6 and self.rigid_contact_friction > 0.0:
            friction_mag = min(
                self.rigid_contact_friction * impulse_mag,
                tangent_norm * mass,
            )
            friction_impulse = -friction_mag * tangent / tangent_norm
            v_com = v_com + friction_impulse / mass
            try:
                delta_omega = torch.linalg.solve(inertia, torch.cross(r, friction_impulse, dim=0))
            except RuntimeError:
                delta_omega = torch.zeros_like(omega)
            omega = omega + self.rigid_contact_angular_gain * delta_omega

        return v_com, omega

    def _apply_soft_elastic_contact_response(self, dt: float) -> None:
        if not (
            self.material_family == "diffuse_damage"
            and self._gravity_drop
            and self._gravity_drop_contacted
            and self.soft_contact_rebound > 0.0
            and self.soft_contact_rebound_frames > 0
            and self.x_mpm is not None
            and self.v_mpm is not None
        ):
            return

        frames_since = int(getattr(self, "_impact_frame_count", 0))
        if frames_since > self.soft_contact_rebound_frames:
            return

        ground_z = float(self._gravity_drop_ground_z)
        band = max(1.25 * float(self.mpm.dx), 1e-4)
        contact = self.x_mpm[:, 2] <= ground_z + band
        if not bool(contact.any()):
            return

        phase = 1.0 - float(frames_since) / max(float(self.soft_contact_rebound_frames), 1.0)
        phase = max(0.0, min(phase, 1.0))
        contact_down = float(torch.clamp(-self.v_mpm[contact, 2].mean(), min=0.0).item())
        impact_speed = max(float(getattr(self, "_soft_impact_speed", 0.0)), contact_down)
        if impact_speed <= 1e-6:
            return

        boost = float(self.soft_contact_rebound) * phase * max(contact_down, 0.24 * impact_speed)
        boost = min(boost, 0.34 * float(self.mpm.dx) / max(float(dt), 1e-8))
        if boost <= 0.0:
            return

        self.v_mpm[contact, 2] = torch.maximum(
            self.v_mpm[contact, 2],
            torch.full_like(self.v_mpm[contact, 2], boost),
        )
        body_fraction = max(0.0, min(float(self.soft_contact_rebound_body_fraction), 1.0))
        if body_fraction > 0.0:
            self.v_mpm[:, 2] = self.v_mpm[:, 2] + body_fraction * boost

        floor = ground_z + 0.18 * float(self.mpm.dx)
        penetration = floor - float(self.x_mpm[:, 2].min().item())
        if penetration > 0.0:
            self.x_mpm[:, 2] = self.x_mpm[:, 2] + min(penetration, 0.45 * float(self.mpm.dx))

    def _apply_soft_elastic_squash(self, dt: float) -> None:
        if not (
            self.material_family == "diffuse_damage"
            and self._gravity_drop
            and self._gravity_drop_contacted
            and self.soft_contact_squash > 0.0
            and self.soft_contact_squash_frames > 0
            and self.x_mpm is not None
            and self.v_mpm is not None
        ):
            return

        frames_since = int(getattr(self, "_impact_frame_count", 0))
        if frames_since > self.soft_contact_squash_frames:
            return

        ground_z = float(self._gravity_drop_ground_z)
        band = max(3.0 * float(self.mpm.dx), 1e-4)
        z = self.x_mpm[:, 2]
        contact = z <= ground_z + band
        if not bool(contact.any()):
            return

        phase = 1.0 - float(frames_since) / max(float(self.soft_contact_squash_frames), 1.0)
        phase = max(0.0, min(phase, 1.0))
        impact_speed = max(float(getattr(self, "_soft_impact_speed", 0.0)), 1e-6)
        speed_scale = max(0.0, min(impact_speed / 35.0, 1.0))
        squash = max(0.0, min(float(self.soft_contact_squash) * phase * speed_scale, 0.24))
        if squash <= 0.0:
            return

        old = self.x_mpm.clone()
        com_xy = self.x_mpm[:, :2].mean(dim=0, keepdim=True)
        height = (z.max() - z.min()).clamp(min=1e-6)
        lower01 = ((z - ground_z) / (0.55 * height)).clamp(0.0, 1.0)
        influence = (1.0 - lower01).pow(1.7)
        influence = torch.maximum(influence, contact.float())
        z_rel = (self.x_mpm[:, 2] - ground_z).clamp(min=0.0)
        self.x_mpm[:, 2] = ground_z + z_rel * (1.0 - squash * influence)
        xy_rel = self.x_mpm[:, :2] - com_xy
        self.x_mpm[:, :2] = com_xy + xy_rel * (1.0 + 0.42 * squash * influence.unsqueeze(1))
        floor = ground_z + 0.12 * float(self.mpm.dx)
        below = self.x_mpm[:, 2] < floor
        if bool(below.any()):
            self.x_mpm[below, 2] = floor
        self.v_mpm = self.v_mpm + 0.18 * (self.x_mpm - old) / max(float(dt), 1e-8)

    def _shape_match_component(self, idx: Tensor, strength: float, dt: float) -> float:
        if idx.numel() < self.shape_match_min_particles:
            return 0.0
        rest_all = self._shape_match_rest_positions
        if rest_all is None or rest_all.shape[0] != self.x_mpm.shape[0]:
            return 0.0

        rest = rest_all[idx]
        current = self.x_mpm[idx]
        rest_com = rest.mean(dim=0)
        current_com = current.mean(dim=0)
        rest_centered = rest - rest_com
        current_centered = current - current_com
        if float(rest_centered.norm().item()) < 1e-8:
            return 0.0

        try:
            cov = rest_centered.transpose(0, 1) @ current_centered
            u, _, vh = torch.linalg.svd(cov)
            rot = u @ vh
            if torch.det(rot) < 0.0:
                u = u.clone()
                u[:, -1] *= -1.0
                rot = u @ vh
        except RuntimeError:
            return 0.0

        target = rest_centered @ rot + current_com
        strength = max(0.0, min(float(strength), 1.0))
        if strength <= 0.0:
            return 0.0

        velocity = self.v_mpm[idx]
        v_com = velocity.mean(dim=0)
        omega = self._component_angular_velocity(current_centered, velocity - v_com.unsqueeze(0))
        v_com, omega = self._apply_rigid_contact_impulse(
            current=current,
            velocity=velocity,
            com=current_com,
            v_com=v_com,
            omega=omega,
        )
        # Post-impact angular damping (ground friction surrogate).
        # Without this the SVD-recovered omega is re-injected every
        # substep via rigid_v and the body rotates indefinitely after
        # landing.  Free-fall rotation is conserved (no damping until
        # contact).
        if (
            self._gravity_drop
            and self._gravity_drop_contacted
            and self.shape_match_angular_damping < 1.0
        ):
            omega = omega * float(self.shape_match_angular_damping)

        dt_eff = max(float(dt), 1e-8)
        rel_target = target - current_com.unsqueeze(0)
        angular_step = torch.cross(
            omega.unsqueeze(0).expand_as(rel_target),
            rel_target,
            dim=1,
        ) * dt_eff
        target = target + angular_step

        if self._gravity_drop and self._gravity_drop_contacted:
            floor = float(self._gravity_drop_ground_z) + 0.35 * float(self.mpm.dx)
            penetration = floor - float(target[:, 2].min().item())
            if penetration > 0.0:
                target[:, 2] = target[:, 2] + min(penetration, 2.0 * float(self.mpm.dx))

        old = self.x_mpm[idx]
        new = old.lerp(target, strength)
        self.x_mpm[idx] = new
        if self.v_mpm is not None and self.mpm.dt > 0.0:
            rel_new = new - new.mean(dim=0, keepdim=True)
            rigid_v = v_com.unsqueeze(0) + torch.cross(
                omega.unsqueeze(0).expand_as(rel_new),
                rel_new,
                dim=1,
            )
            correction_v = (new - old) / max(float(self.mpm.dt), 1e-8)
            blend = max(0.0, min(self.shape_match_velocity_blend + 0.45 * strength, 1.0))
            self.v_mpm[idx] = (
                (1.0 - blend) * self.v_mpm[idx]
                + blend * rigid_v
                + 0.15 * self.shape_match_velocity_blend * correction_v
            )
        return float(omega.norm().item())

    def _apply_shape_matching(self, dt: float) -> None:
        """Keep surface-MPM bodies/fragments cohesive without particle scatter."""
        if (
            not self.shape_matching_enabled
            or self.x_mpm is None
            or self._shape_match_rest_positions is None
            or self._shape_match_rest_positions.shape[0] != self.x_mpm.shape[0]
        ):
            return

        labels = self._current_mpm_fragment_labels()
        if labels is None:
            if self._gravity_drop and not self._gravity_drop_contacted:
                return
            idx = torch.arange(self.x_mpm.shape[0], device=self.x_mpm.device)
            strength = self._body_shape_match_strength()
            angular_speed = self._shape_match_component(idx, strength, dt)
            self._last_rigid_angular_speed_max = angular_speed
            self._last_rigid_angular_speed_mean = angular_speed
            return

        labels = labels.to(device=self.x_mpm.device, dtype=torch.long)
        angular_speeds = []
        for label in labels.unique(sorted=True).tolist():
            mask = labels == int(label)
            idx = torch.where(mask)[0]
            if idx.numel() < self.shape_match_min_particles:
                continue
            strength = (
                self.shape_match_fragment_strength
                if int(label) > 0
                else self._body_shape_match_strength()
            )
            angular_speeds.append(self._shape_match_component(idx, strength, dt))
        if angular_speeds:
            self._last_rigid_angular_speed_max = max(angular_speeds)
            self._last_rigid_angular_speed_mean = sum(angular_speeds) / max(len(angular_speeds), 1)

    def _body_shape_match_strength(self) -> float:
        """Return component rigidity for the still-attached body.

        Localized crack damage should weaken stress along the cut corridor, but
        it should not make the whole uncut body behave like a soft blob.  Only
        diffuse-damage materials globally soften the base component.
        """
        if self.material_family != "diffuse_damage":
            return self.shape_match_strength
        if self.fracture_field.c is not None and bool(self.fracture_field.c.max() > 0.25):
            return self.shape_match_damaged_strength
        return self.shape_match_strength

    # ================================================================
    # Fracture step (Gaussian manifold)
    # ================================================================
