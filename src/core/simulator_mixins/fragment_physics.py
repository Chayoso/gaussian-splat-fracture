"""FragmentPhysicsMixin for ManifoldSimulator.

This module is behavior-preserving extraction from manifold_simulator.py.
"""

import math
from typing import Optional

import torch
from torch import Tensor


class FragmentPhysicsMixin:
    def _rigid_handoff_enabled(self) -> bool:
        if not bool(getattr(self, "fracture_cfg", {}).get(
                "use_physical_fragment_authority", False)):
            return False
        mode = str(getattr(self, "detached_fragment_dynamics", "mpm_shape_match"))
        return mode in {"rigid_handoff", "rigid", "detached_rigid"}

    def _ensure_physical_fragment_registry(self, count: int, device) -> None:
        if (self._physical_fragment_labels is not None
                and self._physical_fragment_labels.shape[0] == count
                and self._physical_fragment_labels.device == device):
            return
        self._physical_fragment_labels = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self._physical_fragment_states = {}
        self._rigid_fragment_states = {}
        self._next_physical_fragment_id = 1

    def _update_physical_fragment_registry(self, raw_particle_labels: Tensor) -> Tensor:
        """Return the current Voronoi-derived persistent fragment labels.

        Voronoi is the only fragment-tracking mechanism: cell-bond
        connected components are computed each frame in
        ``_step_voronoi_fracture`` and written directly to
        ``self._physical_fragment_labels``.  This routine just makes
        them available to the simulator's per-fragment shape-match /
        kinetic-drift bookkeeping (which still iterates labels as
        before).
        """
        self._ensure_physical_fragment_registry(
            raw_particle_labels.shape[0], raw_particle_labels.device
        )
        return self._physical_fragment_labels.clone()

    def _fragmented_physics_labels(self) -> Tensor:
        """Return label overlay for fragment-aware post-P2G2P hooks."""
        use_physical = bool(self.fracture_cfg.get(
            'use_physical_fragment_authority', False))
        if use_physical:
            labels = self._physical_fragment_labels
            if (
                labels is not None
                and labels.shape[0] == self.x_mpm.shape[0]
                and labels.device == self.x_mpm.device
            ):
                return labels
            return torch.zeros(
                self.x_mpm.shape[0],
                dtype=torch.long,
                device=self.x_mpm.device,
            )

        if (
            self.fragment_manager is not None
            and getattr(self.fragment_manager, "fragment_ids", None) is not None
        ):
            return self._map_surface_labels_to_particles(
                self.fragment_manager.fragment_ids)
        return torch.zeros(
            self.x_mpm.shape[0],
            dtype=torch.long,
            device=self.x_mpm.device,
        )

    def _step_fragmented_physics(self, stress: Tensor, dt: float):
        """Single global MPM step with per-fragment label tracking.

        Earlier this routine ran one ``p2g2p_subset`` per fragment label,
        which gave each fragment its own isolated grid and broke
        inter-fragment interaction completely (fragments tunneled
        through each other and the base body kept its own decoupled
        dynamics).  Reviewer-defensible behaviour: a single global
        p2g2p over all particles, with the per-fragment registry update
        + per-fragment shape matching applied as a label-only overlay.
        """
        mpm_frag_ids = self._fragmented_physics_labels()
        rigid_enabled = self._rigid_handoff_enabled()
        rigid_mask = None
        if rigid_enabled:
            self._sync_rigid_fragment_states(mpm_frag_ids)
            rigid_mask = self._rigid_fragment_particle_mask(mpm_frag_ids)

        # Global P2G2P over all particles (base body + fragments share
        # the same grid).  Inter-fragment forces propagate naturally
        # through the grid, and base + fragments inherit the same
        # post-impact velocity field.
        # Save pre-p2g2p fragment z-velocity so we can apply per-particle
        # bounce on the floor.  The MPM grid uses a "slip" BC at the
        # ground plane which zeros out the normal velocity component as
        # part of p2g2p, so by the time the per-fragment shape-match
        # rigid contact impulse runs the z-velocity is already 0 and the
        # restitution kick has nothing to bounce off of (this is why
        # fragment ground bounces were invisible despite restitution=0.5
        # in style profiles).  We restore the bounce after p2g2p by
        # detecting fragment particles that approach the floor and
        # rewriting v_z = -restitution * v_z_pre.
        v_z_pre_for_bounce = None
        floor_restitution = float(getattr(self, "fragment_floor_restitution", 0.0))
        if floor_restitution > 0.0 and getattr(self, "_gravity_drop_contacted", False):
            v_z_pre_for_bounce = self.v_mpm[:, 2].clone()

        if rigid_enabled and rigid_mask is not None and bool(rigid_mask.any()):
            active_mask = ~rigid_mask
            self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p_masked(
                self.x_mpm, self.v_mpm, self.C, self.F, stress, active_mask)
        else:
            self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p(
                self.x_mpm, self.v_mpm, self.C, self.F, stress)

        if v_z_pre_for_bounce is not None:
            ground_z = float(getattr(self, "_gravity_drop_ground_z", 0.05))
            band = max(2.0 * float(self.mpm.dx), 1e-4)
            at_floor = self.x_mpm[:, 2] <= ground_z + band
            was_descending = v_z_pre_for_bounce < -1e-3
            is_fragment = mpm_frag_ids > 0
            bounce_mask = at_floor & was_descending & is_fragment
            if bool(bounce_mask.any()):
                self.v_mpm[bounce_mask, 2] = (
                    -floor_restitution * v_z_pre_for_bounce[bounce_mask]
                )

        # Fragment-only damping override.  The global mpm.damping
        # (post_impact_damping=0.95 per substep for brittle materials)
        # is tuned to suppress base-body elastic vibration after impact
        # but also bleeds the velocity of detached fragments, so a
        # per-fragment rigid release impulse decays to background
        # within a few substeps and the visual reads as soft "powder
        # falling" rather than glass-like "shards flying".  Fragments
        # are shape-matched to rigid (no elastic vibration to suppress)
        # so we can safely undo the damping for fragment-labeled
        # particles only -- their effective per-substep damping is
        # 1.0, the base body keeps its 0.95.
        damping = float(self.mpm.damping)
        if (not rigid_enabled) and 0.0 < damping < 1.0:
            mask = mpm_frag_ids > 0
            if bool(mask.any()):
                self.v_mpm[mask] = self.v_mpm[mask] / damping

        if rigid_enabled and rigid_mask is not None and bool(rigid_mask.any()):
            nonrigid_frag_ids = torch.where(
                rigid_mask,
                torch.zeros_like(mpm_frag_ids),
                mpm_frag_ids,
            )
            self._apply_fragment_extra_gravity(nonrigid_frag_ids, dt)
            self._integrate_rigid_fragments(dt)
        else:
            self._apply_fragment_extra_gravity(mpm_frag_ids, dt)
            self._apply_physical_fragment_release_drift(mpm_frag_ids, dt)

    @torch.no_grad()
    def _apply_fragment_extra_gravity(self, mpm_frag_ids: Tensor, dt: float) -> None:
        """Per-substep extra downward acceleration applied to fragment
        particles only.

        The global ``post_impact_gravity_z`` + ``post_impact_damping``
        (default 0.95 / substep for brittle materials) are tuned for the
        cohesive base body: aggressive damping kills residual elastic
        vibration after impact.  That damping also bleeds the gravity-
        induced velocity of small detached fragments so they visually
        appear to fall under a weaker effective gravity than the base.

        This adds a per-substep extra downward velocity impulse to
        fragment-labeled particles only -- analogous in spirit to the
        runtime ``post_impact_gravity_z`` override but applied
        additively per fragment.  Base body falling kinematics are
        unchanged.  ``fragment_extra_gravity_z = 0`` disables.
        """
        extra_g = float(getattr(self, "fragment_extra_gravity_z", 0.0))
        if abs(extra_g) < 1e-8:
            return
        if mpm_frag_ids is None:
            return
        mask = mpm_frag_ids > 0
        if not bool(mask.any()):
            return
        self.v_mpm[mask, 2] = self.v_mpm[mask, 2] + extra_g * float(dt)

    def _apply_physical_fragment_release_drift(self, mpm_frag_ids: Tensor, dt: float) -> None:
        """Apply a small physical gap / release drift to support-lost fragments."""
        use_physical_authority = bool(getattr(self, "fracture_cfg", {}).get(
            "use_physical_fragment_authority", False))
        if use_physical_authority:
            if mpm_frag_ids is None or not bool((mpm_frag_ids > 0).any()):
                return
        elif self.fragment_manager is None or self.fragment_manager.n_fragments <= 1:
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
            jitter_mag = max(0.0, float(getattr(self, "fragment_release_jitter", 0.0)))
            if stored_dir is None:
                # Per-fragment deterministic-but-distinct perturbation:
                # seed an RNG with the fragment id so each fragment gets
                # a UNIQUE direction jitter, breaking the lockstep
                # "elastic-breathing" look where every fragment moves
                # along the same radial-out + downward path.
                if jitter_mag > 0.0:
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(int(frag_id) * 9173 + 17)
                    rand3 = torch.randn(3, generator=gen).to(direction.device).to(direction.dtype)
                    rand3 = rand3 / rand3.norm().clamp(min=1e-8)
                    direction = self._safe_vector_normalize(
                        ((1.0 - jitter_mag) * direction + jitter_mag * rand3).unsqueeze(0)
                    ).squeeze(0)
                # The release impulse is a post-impact separation aid,
                # not an upward launch.  Jitter is 3D, so it can undo the
                # downward bias unless we re-project it after blending.
                if self.fragment_physical_downward_bias > 0.0:
                    max_up = -0.015 * min(float(self.fragment_physical_downward_bias), 1.0)
                    direction[2] = torch.minimum(
                        direction[2],
                        torch.tensor(max_up, device=direction.device, dtype=direction.dtype),
                    )
                    direction = self._safe_vector_normalize(direction.unsqueeze(0)).squeeze(0)
                state["release_dir"] = direction.detach().clone()
            else:
                direction = self._safe_vector_normalize(
                    (0.86 * stored_dir.to(direction.device) + 0.14 * direction).unsqueeze(0)
                ).squeeze(0)
                if self.fragment_physical_downward_bias > 0.0:
                    max_up = -0.015 * min(float(self.fragment_physical_downward_bias), 1.0)
                    direction[2] = torch.minimum(
                        direction[2],
                        torch.tensor(max_up, device=direction.device, dtype=direction.dtype),
                    )
                    direction = self._safe_vector_normalize(direction.unsqueeze(0)).squeeze(0)
                state["release_dir"] = direction.detach().clone()

            up = torch.tensor([0.0, 0.0, 1.0], device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            spin_axis = state.get("spin_axis")
            if spin_axis is None:
                spin_axis = torch.cross(direction, up, dim=0)
                if float(spin_axis.norm().item()) < 1e-8:
                    spin_axis = torch.tensor([1.0, 0.0, 0.0], device=self.x_mpm.device, dtype=self.x_mpm.dtype)
                # Per-fragment random tilt of the spin axis so fragments
                # tumble around different axes instead of all spinning
                # around the same horizontal-tangent direction.
                if jitter_mag > 0.0:
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(int(frag_id) * 9173 + 53)
                    rand3 = torch.randn(3, generator=gen).to(spin_axis.device).to(spin_axis.dtype)
                    rand3 = rand3 / rand3.norm().clamp(min=1e-8)
                    spin_axis = self._safe_vector_normalize(
                        ((1.0 - 0.85 * jitter_mag) * spin_axis + 0.85 * jitter_mag * rand3).unsqueeze(0)
                    ).squeeze(0)
                else:
                    spin_axis = self._safe_vector_normalize(spin_axis.unsqueeze(0)).squeeze(0)
                state["spin_axis"] = spin_axis.detach().clone()
            else:
                spin_axis = self._safe_vector_normalize(spin_axis.to(self.x_mpm.device).unsqueeze(0)).squeeze(0)

            # Per-fragment random scale on the release magnitudes so
            # fragments don't all reach peak velocity at the same instant.
            speed_scale = 1.0
            if jitter_mag > 0.0:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(frag_id) * 9173 + 91)
                # Uniform multiplier in [1 - jitter_mag, 1 + jitter_mag].
                speed_scale = float(1.0 + jitter_mag * (2.0 * torch.rand(1, generator=gen).item() - 1.0))
                state["speed_scale"] = speed_scale
            else:
                speed_scale = float(state.get("speed_scale", 1.0))

            gap_step = (self.fragment_physical_gap_scale * taper
                        * (1.0 + 1.25 * release_score) * speed_scale)
            vel_step = (self.fragment_physical_release_velocity * taper
                        * (0.55 + 0.95 * release_score) * speed_scale)
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
        use_physical = bool(self.fracture_cfg.get(
            'use_physical_fragment_authority', False))
        if (
            use_physical
            and
            self._physical_fragment_labels is not None
            and self._physical_fragment_labels.shape[0] == self.x_mpm.shape[0]
            and bool((self._physical_fragment_labels > 0).any())
        ):
            return self._physical_fragment_labels
        if use_physical:
            return None
        if (
            self.fragment_manager is not None
            and getattr(self.fragment_manager, "fragment_ids", None) is not None
            and bool((self.fragment_manager.fragment_ids > 0).any())
        ):
            return self._map_surface_labels_to_particles(self.fragment_manager.fragment_ids)
        return None

    def _rigid_fragment_particle_mask(self, labels: Optional[Tensor] = None) -> Tensor:
        mask = torch.zeros(
            self.x_mpm.shape[0],
            dtype=torch.bool,
            device=self.x_mpm.device,
        )
        states = getattr(self, "_rigid_fragment_states", {})
        if not states:
            return mask
        for state in states.values():
            idx = state.get("indices")
            if idx is None:
                continue
            idx = idx.to(device=self.x_mpm.device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < self.x_mpm.shape[0])]
            if idx.numel() > 0:
                mask[idx] = True
        return mask

    def _sync_rigid_fragment_states(self, labels: Optional[Tensor] = None) -> None:
        if not self._rigid_handoff_enabled():
            return
        if self.x_mpm is None or self.v_mpm is None:
            return
        if labels is None:
            labels = self._physical_fragment_labels
        if labels is None or labels.shape[0] != self.x_mpm.shape[0]:
            return
        if not hasattr(self, "_rigid_fragment_states"):
            self._rigid_fragment_states = {}

        min_particles = max(1, int(getattr(
            self, "rigid_handoff_min_particles",
            getattr(self, "fragment_physical_min_size", 1),
        )))
        attach_tiny = bool(getattr(
            self, "rigid_handoff_attach_tiny_to_nearest_fragment", True))
        tiny_to_base = bool(getattr(
            self, "rigid_handoff_merge_tiny_fragments", False))

        states = self._rigid_fragment_states
        counts = {
            int(lab): int((labels == int(lab)).sum().item())
            for lab in labels.unique(sorted=True).tolist()
            if int(lab) > 0
        }
        large_labels = [
            lab for lab, count in counts.items()
            if count >= min_particles
        ]
        if attach_tiny and large_labels:
            large_com = {}
            for lab in large_labels:
                lab_mask = labels == lab
                if bool(lab_mask.any()):
                    large_com[lab] = self.x_mpm[lab_mask].mean(dim=0)
            for lab, count in list(counts.items()):
                if count >= min_particles:
                    continue
                idx = torch.where(labels == lab)[0]
                if idx.numel() == 0 or not large_com:
                    continue
                tiny_com = self.x_mpm[idx].mean(dim=0)
                target = min(
                    large_com,
                    key=lambda cand: float((large_com[cand] - tiny_com).norm().item()),
                )
                old_state = states.pop(int(lab), None)
                if old_state is not None:
                    old_idx = old_state.get("indices")
                    if old_idx is not None:
                        old_idx = old_idx.to(device=self.x_mpm.device, dtype=torch.long)
                        old_idx = old_idx[
                            (old_idx >= 0) & (old_idx < self.x_mpm.shape[0])
                        ]
                        if old_idx.numel() > 0:
                            idx = torch.unique(torch.cat([idx, old_idx]))
                labels[idx] = int(target)
                if int(target) in states:
                    self._absorb_rigid_fragment_indices(int(target), idx)

        for lab in labels.unique(sorted=True).tolist():
            lab_int = int(lab)
            if lab_int <= 0 or lab_int in self._rigid_fragment_states:
                continue
            idx = torch.where(labels == lab_int)[0]
            if idx.numel() < min_particles:
                if tiny_to_base:
                    states.pop(lab_int, None)
                    labels[idx] = 0
                continue
            self._capture_rigid_fragment_state(lab_int, idx)

        present_labels = {
            int(lab) for lab in labels.unique(sorted=True).tolist()
            if int(lab) > 0
        }
        for lab in list(states.keys()):
            if int(lab) not in present_labels:
                states.pop(int(lab), None)

    def _absorb_rigid_fragment_indices(self, label: int, idx: Tensor) -> None:
        state = getattr(self, "_rigid_fragment_states", {}).get(int(label))
        if state is None or idx.numel() == 0:
            return
        idx = idx.to(device=self.x_mpm.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < self.x_mpm.shape[0])]
        if idx.numel() == 0:
            return
        existing = state["indices"].to(device=self.x_mpm.device, dtype=torch.long)
        keep = ~torch.isin(idx, existing)
        idx = idx[keep]
        if idx.numel() == 0:
            return
        R = state["R"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
        com = state["com"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
        local_new = (self.x_mpm[idx].detach() - com.unsqueeze(0)) @ R
        state["indices"] = torch.cat([existing, idx.detach().clone()])
        state["local"] = torch.cat([
            state["local"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype),
            local_new.detach().clone(),
        ], dim=0)
        self._refresh_rigid_fragment_mass_properties(state)

    def _capture_rigid_fragment_state(self, label: int, idx: Tensor) -> None:
        idx = idx.to(device=self.x_mpm.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < self.x_mpm.shape[0])]
        if idx.numel() == 0:
            return
        x = self.x_mpm[idx].detach()
        v = self.v_mpm[idx].detach()
        com = x.mean(dim=0)
        local = x - com.unsqueeze(0)
        v_com = v.mean(dim=0)
        base_v_com = None
        base_com = None
        labels = getattr(self, "_physical_fragment_labels", None)
        if labels is not None and labels.shape[0] == self.v_mpm.shape[0]:
            base_mask = labels.to(device=self.v_mpm.device) == 0
            if bool(base_mask.any()):
                base_com = self.x_mpm[base_mask].mean(dim=0).detach()
                base_v_com = self.v_mpm[base_mask].mean(dim=0).detach()
        lateral_cap = float(getattr(
            self, "rigid_handoff_birth_lateral_velocity_cap", 0.0))
        inherited_lateral_scale = max(0.0, min(float(getattr(
            self,
            "rigid_handoff_birth_inherited_lateral_velocity_scale",
            1.0,
        )), 1.0))
        if base_v_com is not None and inherited_lateral_scale < 1.0:
            rel_xy = v_com[:2] - base_v_com[:2]
            v_com[:2] = base_v_com[:2] + inherited_lateral_scale * rel_xy
        elif base_v_com is None and inherited_lateral_scale < 1.0:
            v_com[:2] = inherited_lateral_scale * v_com[:2]
        if lateral_cap > 0.0:
            if base_v_com is None:
                v_xy = v_com[:2]
                v_xy_norm = float(v_xy.norm().item())
                if v_xy_norm > lateral_cap:
                    v_com[:2] = v_xy * (lateral_cap / max(v_xy_norm, 1e-8))
            else:
                rel_xy = v_com[:2] - base_v_com[:2]
                rel_xy_norm = float(rel_xy.norm().item())
                if rel_xy_norm > lateral_cap:
                    v_com[:2] = base_v_com[:2] + rel_xy * (
                        lateral_cap / max(rel_xy_norm, 1e-8))
        birth_down = float(getattr(
            self, "rigid_handoff_birth_downward_velocity", 0.0))
        if birth_down > 0.0 and bool(getattr(self, "_gravity_drop_contacted", False)):
            base_vz = (
                base_v_com[2] if base_v_com is not None
                else torch.zeros((), device=v_com.device, dtype=v_com.dtype)
            )
            target_vz = torch.tensor(
                -birth_down, device=v_com.device, dtype=v_com.dtype) + base_vz
            v_com[2] = torch.minimum(v_com[2], target_vz)
        if bool(getattr(self, "fracture_cfg", {}).get(
                "rigid_handoff_inherit_particle_spin", False)):
            omega = self._component_angular_velocity(local, v - v_com.unsqueeze(0))
        elif bool(getattr(self, "fracture_cfg", {}).get(
                "rigid_handoff_inherit_body_spin", False)):
            omega_src = getattr(self, "_omega_com", None)
            if omega_src is None:
                omega = torch.zeros(3, device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            else:
                omega = omega_src.to(
                    device=self.x_mpm.device,
                    dtype=self.x_mpm.dtype,
                ).detach().clone()
        else:
            # Capture only detachment-local angular momentum.  Full parent
            # body spin is subtracted so all shards do not inherit the same
            # perpetual tumble, but residual intra-fragment angular velocity
            # from crack opening / MPM contact is preserved.
            raw_omega = self._component_angular_velocity(
                local, v - v_com.unsqueeze(0))
            parent_omega = getattr(self, "_omega_com", None)
            if parent_omega is None:
                parent_omega = torch.zeros(
                    3, device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            else:
                parent_omega = parent_omega.to(
                    device=self.x_mpm.device,
                    dtype=self.x_mpm.dtype,
                )
            omega = (
                raw_omega - parent_omega
            ) * float(getattr(
                self, "rigid_handoff_birth_angular_velocity_scale", 0.0))
            body_spin_scale = float(getattr(
                self, "rigid_handoff_birth_body_angular_velocity_scale", 0.0))
            if body_spin_scale > 0.0:
                omega = omega + parent_omega * body_spin_scale

            tumble_gain = float(getattr(
                self, "rigid_handoff_birth_tumble_gain", 0.0))
            if tumble_gain > 0.0 and base_v_com is not None:
                rel_v = v_com - base_v_com
                speed = float(rel_v.norm().item())
                if speed > 1e-6 and float(local.norm().item()) > 1e-8:
                    try:
                        _, _, vh = torch.linalg.svd(local, full_matrices=False)
                        major_axis = vh[0]
                    except RuntimeError:
                        major_axis = None
                    if major_axis is not None:
                        motion_dir = rel_v / rel_v.norm().clamp(min=1e-8)
                        axis = torch.cross(major_axis, motion_dir, dim=0)
                        if float(axis.norm().item()) < 1e-8 and base_com is not None:
                            radial = com - base_com
                            if float(radial.norm().item()) > 1e-8:
                                axis = torch.cross(
                                    radial / radial.norm().clamp(min=1e-8),
                                    motion_dir,
                                    dim=0,
                                )
                        if float(axis.norm().item()) > 1e-8:
                            axis = axis / axis.norm().clamp(min=1e-8)
                            radius = local.norm(dim=1).quantile(0.75).clamp(
                                min=float(self.mpm.dx))
                            omega = omega + axis * (
                                tumble_gain * speed / radius)
            omega_cap = float(getattr(
                self, "rigid_handoff_birth_angular_velocity_cap", 0.0))
            omega_norm = float(omega.norm().item())
            if omega_cap > 0.0 and omega_norm > omega_cap:
                omega = omega * (omega_cap / max(omega_norm, 1e-8))
        dtype = self.x_mpm.dtype
        device = self.x_mpm.device
        eye = torch.eye(3, device=device, dtype=dtype)
        mass_per_particle = float(getattr(self.mpm, "p_mass", 1.0))
        mass = max(mass_per_particle * float(idx.numel()), 1e-12)
        rr = (local * local).sum(dim=1)
        inertia = mass_per_particle * ((rr.sum() * eye) - local.transpose(0, 1) @ local)
        inertia = inertia + 1e-8 * max(float(idx.numel()), 1.0) * eye
        try:
            inertia_inv = torch.linalg.inv(inertia)
        except RuntimeError:
            inertia_inv = torch.linalg.pinv(inertia)
        radius = local.norm(dim=1).max().clamp(min=float(self.mpm.dx))
        self._rigid_fragment_states[int(label)] = {
            "indices": idx.detach().clone(),
            "local": local.detach().clone(),
            "R": eye.detach().clone(),
            "com": com.detach().clone(),
            "birth_com": com.detach().clone(),
            "v_com": v_com.detach().clone(),
            "omega": omega.detach().clone(),
            "mass": mass,
            "inertia_body_inv": inertia_inv.detach().clone(),
            "radius": float(radius.item()),
            "birth_time": float(getattr(self.mpm, "time", 0.0)),
            "birth_physics_step": int(getattr(self, "_physics_step", 0)),
        }

    def _refresh_rigid_fragment_mass_properties(self, state: dict) -> None:
        local = state["local"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
        dtype = self.x_mpm.dtype
        device = self.x_mpm.device
        eye = torch.eye(3, device=device, dtype=dtype)
        mass_per_particle = float(getattr(self.mpm, "p_mass", 1.0))
        mass = max(mass_per_particle * float(local.shape[0]), 1e-12)
        rr = (local * local).sum(dim=1)
        inertia = mass_per_particle * ((rr.sum() * eye) - local.transpose(0, 1) @ local)
        inertia = inertia + 1e-8 * max(float(local.shape[0]), 1.0) * eye
        try:
            inertia_inv = torch.linalg.inv(inertia)
        except RuntimeError:
            inertia_inv = torch.linalg.pinv(inertia)
        state["mass"] = mass
        state["inertia_body_inv"] = inertia_inv.detach().clone()
        state["radius"] = float(local.norm(dim=1).max().clamp(min=float(self.mpm.dx)).item())

    @staticmethod
    def _rodrigues(axis_angle: Tensor) -> Tensor:
        device = axis_angle.device
        dtype = axis_angle.dtype
        eye = torch.eye(3, device=device, dtype=dtype)
        theta = axis_angle.norm()
        if float(theta.item()) < 1e-9:
            return eye
        axis = axis_angle / theta.clamp(min=1e-12)
        x, y, z = axis[0], axis[1], axis[2]
        K = torch.stack([
            torch.stack([torch.zeros((), device=device, dtype=dtype), -z, y]),
            torch.stack([z, torch.zeros((), device=device, dtype=dtype), -x]),
            torch.stack([-y, x, torch.zeros((), device=device, dtype=dtype)]),
        ])
        return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)

    def _rigid_world_inertia_inv(self, state: dict) -> Tensor:
        R = state["R"]
        return R @ state["inertia_body_inv"] @ R.transpose(0, 1)

    def _apply_floor_contact_to_rigid_state(self, state: dict, dt: float) -> None:
        ground_z = float(getattr(self, "_gravity_drop_ground_z", 0.0))
        if ground_z <= 0.0:
            return
        local = state["local"]
        R = state["R"]
        rel = local @ R.transpose(0, 1)
        pos = state["com"].unsqueeze(0) + rel
        min_z = float(pos[:, 2].min().item())
        if min_z < ground_z:
            state["com"][2] = state["com"][2] + (ground_z - min_z)
            pos[:, 2] = pos[:, 2] + (ground_z - min_z)

        band = max(float(getattr(self, "rigid_contact_band", 1.5)) * float(self.mpm.dx), 1e-6)
        contact_mask = pos[:, 2] <= ground_z + band
        if not bool(contact_mask.any()):
            return

        n = torch.tensor([0.0, 0.0, 1.0], device=rel.device, dtype=rel.dtype)
        contact_pos = pos[contact_mask]
        depth = (ground_z + band - contact_pos[:, 2]).clamp(min=0.0)
        if float(depth.sum().item()) > 1e-8:
            weights = depth / depth.sum().clamp(min=1e-8)
            contact_point = (contact_pos * weights.unsqueeze(1)).sum(dim=0)
        else:
            contact_point = contact_pos.mean(dim=0)
        r = contact_point - state["com"]

        inv_m = 1.0 / max(float(state["mass"]), 1e-12)
        e = max(0.0, min(float(getattr(
            self, "rigid_handoff_restitution",
            getattr(self, "rigid_contact_restitution", 0.0),
        )), 0.95))
        torque_enabled = bool(getattr(
            self, "rigid_handoff_floor_contact_torque", True))
        angular_gain = max(0.0, float(getattr(
            self,
            "rigid_handoff_floor_contact_angular_gain",
            getattr(self, "rigid_contact_angular_gain", 0.0),
        )))
        inertia_inv = None
        omega = state.get("omega")
        if torque_enabled and omega is not None and float(r.norm().item()) > 1e-8:
            try:
                inertia_inv = self._rigid_world_inertia_inv(state)
            except RuntimeError:
                inertia_inv = None

        if inertia_inv is not None:
            v_contact = state["v_com"] + torch.cross(omega, r, dim=0)
            vn = torch.minimum(v_contact.dot(n), state["v_com"].dot(n))
        else:
            v_contact = state["v_com"]
            vn = torch.dot(state["v_com"], n)

        normal_impulse_mag = 0.0
        if float(vn.item()) < 0.0:
            denom = inv_m
            if inertia_inv is not None:
                rn = torch.cross(r, n, dim=0)
                angular_term = n.dot(torch.cross(inertia_inv @ rn, r, dim=0))
                denom = denom + max(float(angular_term.item()), 0.0)
            denom = max(float(denom), 1e-8)
            normal_impulse_mag = -(1.0 + e) * float(vn.item()) / denom
            impulse = normal_impulse_mag * n
            state["v_com"] = state["v_com"] + impulse * inv_m
            # Keep the normal support impulse torque-free.  With only a
            # particle-cloud contact patch, the normal lever arm is too noisy
            # for small shards and creates hundreds of rad/s of fake spin.
            # Tangential friction below is the resolved source of contact
            # angular momentum.

        # Coulomb floor friction for rigid handoff fragments.  This is a
        # contact impulse, not free-space fragment damping: it only acts
        # while a fragment has particles in the floor contact band.  The
        # normal load includes the collision impulse plus the per-step
        # support impulse from gravity, so resting fragments can settle
        # instead of sliding forever on a frictionless plane.
        gravity = self.mpm.gravity.to(device=rel.device, dtype=rel.dtype)
        mass = max(float(state["mass"]), 1e-12)
        support_impulse_mag = (
            max(0.0, -float(torch.dot(gravity, n).item()))
            * mass
            * max(float(dt), 0.0)
        )
        mu = max(0.0, float(getattr(
            self, "rigid_handoff_floor_friction",
            getattr(self, "rigid_contact_friction", 0.0),
        )))
        if mu > 0.0:
            friction_cap = mu * (normal_impulse_mag + support_impulse_mag)
            if friction_cap > 0.0:
                omega = state.get("omega")
                if inertia_inv is not None and omega is not None:
                    v_contact = state["v_com"] + torch.cross(omega, r, dim=0)
                else:
                    v_contact = state["v_com"]
                v_n = torch.dot(v_contact, n) * n
                v_t = v_contact - v_n
                v_t_norm = float(v_t.norm().item())
                if v_t_norm > 1e-8:
                    stop_impulse_mag = mass * v_t_norm
                    applied = min(stop_impulse_mag, friction_cap)
                    friction_impulse = -applied * (
                        v_t / v_t.norm().clamp(min=1e-8))
                    state["v_com"] = state["v_com"] + friction_impulse * inv_m
                    if inertia_inv is not None and angular_gain > 0.0:
                        state["omega"] = state["omega"] + angular_gain * (
                            inertia_inv @ torch.cross(r, friction_impulse, dim=0))

        # Rolling/contact angular friction.  This is contact-only: airborne
        # shards keep their birth angular momentum, but pieces resting on the
        # floor lose spin through the support impulse instead of spinning
        # forever.
        mu_roll = max(0.0, float(getattr(
            self, "rigid_handoff_floor_angular_friction", 0.0)))
        omega = state.get("omega")
        if mu_roll > 0.0 and omega is not None:
            omega_norm = float(omega.norm().item())
            if omega_norm > 1e-8:
                support_accel = max(0.0, -float(torch.dot(gravity, n).item()))
                radius = max(float(state.get("radius", float(self.mpm.dx))),
                             float(self.mpm.dx))
                domega = mu_roll * support_accel * max(float(dt), 0.0) / radius
                if domega > 0.0:
                    state["omega"] = omega * max(0.0, 1.0 - domega / omega_norm)

    def _solve_rigid_fragment_pair_contacts(self) -> None:
        states = getattr(self, "_rigid_fragment_states", {})
        if len(states) < 2:
            return
        labels = list(states.keys())
        e = max(0.0, min(float(getattr(
            self, "rigid_handoff_restitution",
            getattr(self, "rigid_contact_restitution", 0.0),
        )), 0.95))
        for i, lab_a in enumerate(labels):
            a = states[lab_a]
            for lab_b in labels[i + 1:]:
                b = states[lab_b]
                delta = b["com"] - a["com"]
                dist = delta.norm()
                ra = float(a["radius"])
                rb = float(b["radius"])
                target = ra + rb
                # Bounding spheres are a broad phase, not the shard mesh.
                # Immediately after fracture, adjacent shards can have very
                # overlapping spheres even though their surfaces are only
                # touching along the crack.  Resolving that overlap to
                # ``ra + rb`` creates the visual "explosion".  Treat the
                # birth COM distance as the non-penetrating rest separation
                # for sibling fragments; contact only corrects motion that
                # compresses them beyond that rest distance plus a small slop.
                birth_a = a.get("birth_com")
                birth_b = b.get("birth_com")
                if birth_a is not None and birth_b is not None:
                    birth_delta = (
                        birth_b.to(device=delta.device, dtype=delta.dtype)
                        - birth_a.to(device=delta.device, dtype=delta.dtype)
                    )
                    slop = max(
                        float(getattr(
                            self,
                            "rigid_handoff_pair_contact_birth_slop",
                            2.0,
                        )) * float(self.mpm.dx),
                        1e-6,
                    )
                    # Treat the birth separation as the rest distance and
                    # allow a small tolerance before applying broad-phase
                    # sphere correction.  Using birth_distance + slop pushes
                    # every sibling pair apart immediately after fracture,
                    # which makes fully-pulverized objects expand even when
                    # explicit release velocity is disabled.
                    target = min(
                        target,
                        max(float(birth_delta.norm().item()) - slop, 0.0),
                    )
                if float(dist.item()) >= target:
                    continue
                if float(dist.item()) < 1e-8:
                    n = torch.tensor([1.0, 0.0, 0.0], device=delta.device, dtype=delta.dtype)
                    dist_val = 1e-8
                else:
                    n = delta / dist.clamp(min=1e-8)
                    dist_val = float(dist.item())
                penetration = target - dist_val
                inv_ma = 1.0 / max(float(a["mass"]), 1e-12)
                inv_mb = 1.0 / max(float(b["mass"]), 1e-12)
                inv_sum = inv_ma + inv_mb
                if inv_sum <= 1e-12:
                    continue
                a["com"] = a["com"] - n * (penetration * inv_ma / inv_sum)
                b["com"] = b["com"] + n * (penetration * inv_mb / inv_sum)

                # This is a bounding-sphere contact, not a resolved mesh
                # contact point.  Using r = n * radius as a lever arm
                # creates synthetic torque and can spin fragments forever
                # when many shards overlap.  Resolve pair contact at the
                # centers of mass and leave angular velocity unchanged.
                rel_v = b["v_com"] - a["v_com"]
                vn = torch.dot(rel_v, n)
                if float(vn.item()) >= 0.0:
                    continue
                j = -(1.0 + e) * float(vn.item()) / inv_sum
                impulse = j * n
                a["v_com"] = a["v_com"] - impulse * inv_ma
                b["v_com"] = b["v_com"] + impulse * inv_mb

    def _integrate_rigid_fragments(self, dt: float) -> None:
        states = getattr(self, "_rigid_fragment_states", {})
        if not states:
            return
        dt = float(dt)
        gravity = self.mpm.gravity.to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
        for state in states.values():
            state["v_com"] = state["v_com"] + gravity * dt
            dR = self._rodrigues(state["omega"] * dt)
            state["R"] = dR @ state["R"]
            try:
                u, _, vh = torch.linalg.svd(state["R"])
                state["R"] = u @ vh
            except RuntimeError:
                pass
            state["com"] = state["com"] + state["v_com"] * dt
            self._apply_floor_contact_to_rigid_state(state, dt)
        self._solve_rigid_fragment_pair_contacts()
        self._write_rigid_fragment_particles_from_state()

    def _write_rigid_fragment_particles_from_state(self) -> None:
        states = getattr(self, "_rigid_fragment_states", {})
        if not states or self.x_mpm is None or self.v_mpm is None:
            return
        eye = torch.eye(3, device=self.x_mpm.device, dtype=self.x_mpm.dtype)
        for lab, state in states.items():
            idx = state["indices"].to(device=self.x_mpm.device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < self.x_mpm.shape[0])]
            if idx.numel() == 0:
                continue
            local = state["local"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            R = state["R"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            rel = local @ R.transpose(0, 1)
            pos = state["com"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype).unsqueeze(0) + rel
            omega = state["omega"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            v_com = state["v_com"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            vel = v_com.unsqueeze(0) + torch.cross(
                omega.unsqueeze(0).expand_as(rel),
                rel,
                dim=1,
            )
            self.x_mpm[idx] = pos.clamp(self.mpm.clip_bound, 1.0 - self.mpm.clip_bound)
            self.v_mpm[idx] = vel
            if self.C is not None and self.C.shape[0] == self.x_mpm.shape[0]:
                self.C[idx] = torch.zeros_like(self.C[idx])
            if self.F is not None and self.F.shape[0] == self.x_mpm.shape[0]:
                self.F[idx] = eye.unsqueeze(0).expand(idx.numel(), 3, 3)
            if (self._physical_fragment_labels is not None
                    and self._physical_fragment_labels.shape[0] == self.x_mpm.shape[0]):
                self._physical_fragment_labels[idx] = int(lab)

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
        physical_authority = bool(getattr(self, "fracture_cfg", {}).get(
            "use_physical_fragment_authority", False))
        contact_torque_enabled = bool(getattr(self, "fracture_cfg", {}).get(
            "shape_match_contact_torque_enabled",
            not physical_authority,
        ))
        if not contact_torque_enabled:
            vn_com = v_com.dot(n)
            mass = max(float(current.shape[0]), 1.0)
            restitution = max(0.0, min(float(self.rigid_contact_restitution), 0.85))
            normal_impulse_mag = 0.0
            if float(vn_com.item()) < -1e-5:
                normal_impulse_mag = float((-(1.0 + restitution) * vn_com * mass).item())
                v_com = v_com + (normal_impulse_mag / mass) * n

            friction = max(0.0, float(self.rigid_contact_friction))
            if friction > 0.0:
                gravity = self.mpm.gravity.to(device=current.device, dtype=current.dtype)
                support_impulse_mag = (
                    max(0.0, -float(torch.dot(gravity, n).item()))
                    * mass
                    * max(float(self.mpm.dt), 0.0)
                )
                friction_cap = friction * (normal_impulse_mag + support_impulse_mag)
                v_t = v_com - v_com.dot(n) * n
                v_t_norm = float(v_t.norm().item())
                if friction_cap > 0.0 and v_t_norm > 1e-8:
                    applied = min(mass * v_t_norm, friction_cap)
                    v_com = v_com - (applied / mass) * (
                        v_t / v_t.norm().clamp(min=1e-8)
                    )
            return v_com, omega

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

    def _shape_match_component(self, idx: Tensor, strength: float, dt: float,
                               is_fragment: bool = False,
                               label: int = 0) -> float:
        """Shape-match a connected component back toward its rest pose.

        Step 4a (pipeline rewrite, 2026-05-09): when ``is_fragment`` is
        True and ``manifold.shape_match_fragment_position_pull`` is
        disabled, the position pull (``x.lerp(target, strength)``) is
        skipped for this component.  The velocity-injection block still
        runs so cohesion is maintained through ``v``.  Step 4b will
        further attenuate the velocity injection on fragments.
        """
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
            # Procrustes: optimal R such that current ≈ rest @ R + COM.
            # cov = rest^T @ current; SVD U Σ V^T ⇒ rot = U V^T.
            cov = rest_centered.transpose(0, 1) @ current_centered
            u, _, vh = torch.linalg.svd(cov)
            rot = u @ vh
            if torch.det(rot) < 0.0:
                u = u.clone()
                u[:, -1] *= -1.0
                rot = u @ vh

            # Optional Müller 2005 affine extension: optimal linear
            # transform A that maps centered rest to centered current
            # via least-squares.  Polar-decomposed via SVD into
            # rotation × symmetric stretch.  T = (1-β)·R + β·A blends
            # rigid (β=0) and full affine (β=1).
            affine_blend = float(getattr(self, 'shape_match_affine_blend', 0.0))
            affine_blend = max(0.0, min(affine_blend, 1.0))
            if affine_blend > 0.0:
                P_mat = current_centered.transpose(0, 1) @ rest_centered  # 3x3
                Q_mat = rest_centered.transpose(0, 1) @ rest_centered     # 3x3
                eye3 = torch.eye(3, device=Q_mat.device, dtype=Q_mat.dtype)
                Q_reg = Q_mat + 1e-6 * eye3 * float(Q_mat.diag().abs().max())
                A_col = P_mat @ torch.linalg.inv(Q_reg)  # column-form A
                # Clamp singular values to prevent runaway stretch.
                # SVD: A = U Σ V^T.  We clamp Σ to [0.78, 1.0] so
                # deformation is COMPRESSION-ONLY (max 22% squash per
                # axis, no expansion).  This is the correct rubber
                # impact regime: bunny squashes when hitting floor,
                # never stretches taller than rest.  Without this,
                # the affine fit captures impact stretch (top falling
                # while bottom held) and amplifies it each frame.
                u_a, s_a, vh_a = torch.linalg.svd(A_col)
                # Configurable SV clamp (default [0.78, 1.0]).  Lower
                # min allows more dramatic per-axis squash (rubber-
                # like elasticity).  max=1.0 forbids stretching past
                # rest size to prevent vertical run-away.
                sv_min = float(getattr(self, 'shape_match_sv_min', 0.78))
                sv_max = float(getattr(self, 'shape_match_sv_max', 1.00))
                s_a = torch.clamp(s_a, min=sv_min, max=sv_max)
                A_col = u_a @ torch.diag(s_a) @ vh_a
                # In code's row-form convention, target = rest @ R_row.
                # A in row form is A_col.T (rest_row @ A_col.T = current_row).
                A_row = A_col.transpose(0, 1)
                # Blend rigid (rot) with affine (A_row).
                rot = (1.0 - affine_blend) * rot + affine_blend * A_row
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
        # Two-regime model:
        # - Above `shape_match_static_omega` (rad/s): kinetic friction,
        #   mild multiplicative decay per substep -- preserves a falling
        #   body's natural tipping/rolling.
        # - Below threshold: static friction surrogate, aggressive decay
        #   (omega *= shape_match_static_damping, default 0.5).  This is
        #   what prevents the body from spinning indefinitely once it
        #   has tipped over and lies flat: the contact-impulse loop
        #   keeps injecting tiny torques from numerical noise, and
        #   without a static threshold the body never reaches rest.
        # Free-fall rotation is preserved (no damping before contact).
        if (
            self._gravity_drop
            and self._gravity_drop_contacted
        ):
            omega_norm = float(omega.norm().item())
            if omega_norm < float(self.shape_match_static_omega):
                omega = omega * float(self.shape_match_static_damping)
            elif self.shape_match_angular_damping < 1.0:
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
        # Step 4a: skip position pull on detached fragments when the
        # corresponding flag is disabled.  Velocity injection still
        # runs below so cohesion is preserved through v.
        #
        # Guard: Step 4a is only meaningful when fragment identity is
        # owned by the voronoi authority (Step 2).  Without that, the
        # incoming label may be a graph-fragment id whose physical
        # separation is not yet established, and skipping the pull
        # would bleed cohesion from still-connected pieces.  We
        # therefore require ``use_physical_fragment_authority`` to be
        # True before honouring the skip.
        position_pull_enabled = bool(self.fracture_cfg.get(
            'shape_match_fragment_position_pull', True))
        physical_authority = bool(self.fracture_cfg.get(
            'use_physical_fragment_authority', False))
        skip_pull = (is_fragment
                     and not position_pull_enabled
                     and physical_authority)
        if skip_pull:
            new = old
        else:
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
            apply_handoff = is_fragment and physical_authority
            if apply_handoff:
                smoothing_steps = max(0, int(self.fracture_cfg.get(
                    'fragment_handoff_smoothing_substeps', 0)))
                birth_steps = getattr(
                    self, '_physical_fragment_birth_physics_step', {})
                birth_step = birth_steps.get(int(label), None)
                age_steps = (
                    int(getattr(self, '_physics_step', 0)) - int(birth_step)
                    if birth_step is not None else smoothing_steps
                )
                if smoothing_steps > 0 and age_steps < smoothing_steps:
                    w = 1.0 - max(float(age_steps), 0.0) / float(smoothing_steps)
                    v_target = v_com.unsqueeze(0).expand_as(self.v_mpm[idx])
                    self.v_mpm[idx] = (
                        (1.0 - w) * self.v_mpm[idx] + w * v_target
                    )
                else:
                    s = float(self.fracture_cfg.get(
                        'shape_match_fragment_velocity_blend_scale', 1.0))
                    s = max(0.0, min(s, 1.0))
                    self.v_mpm[idx] = (
                        (1.0 - s * blend) * self.v_mpm[idx]
                        + s * blend * rigid_v
                        + 0.15 * s * self.shape_match_velocity_blend * correction_v
                    )
            else:
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
        rigid_states = getattr(self, "_rigid_fragment_states", {})
        angular_speeds = [
            float(state.get("omega", torch.zeros(3, device=self.x_mpm.device)).norm().item())
            for state in rigid_states.values()
            if state.get("omega") is not None
        ]
        for label in labels.unique(sorted=True).tolist():
            mask = labels == int(label)
            idx = torch.where(mask)[0]
            if idx.numel() < self.shape_match_min_particles:
                continue
            is_fragment = int(label) > 0
            if is_fragment and int(label) in rigid_states:
                continue
            strength = (
                self.shape_match_fragment_strength
                if is_fragment
                else self._body_shape_match_strength()
            )
            angular_speeds.append(self._shape_match_component(
                idx, strength, dt, is_fragment=is_fragment, label=int(label)))
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
