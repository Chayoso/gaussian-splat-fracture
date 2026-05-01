"""RenderFragmentMixin for ManifoldSimulator.

This module is behavior-preserving extraction from manifold_simulator.py.
"""

from typing import Optional

import torch
from torch import Tensor


class RenderFragmentMixin:
    def _current_fragment_impulse_strength(self) -> float:
        if not self.fragmentation_active or self._fragment_activation_frame < 0:
            return self.fragment_impulse_strength
        frames_since = max(self.frame_count - self._fragment_activation_frame, 0)
        if frames_since >= self.fragment_impulse_boost_frames:
            return self.fragment_impulse_strength
        decay = self.fragment_impulse_decay ** frames_since
        return self.fragment_impulse_strength * self.fragment_event_boost * max(decay, 0.45)

    def _current_fragment_visual_boost(self) -> float:
        if not self.fragmentation_active or self._fragment_activation_frame < 0:
            return 1.0
        frames_since = max(self.frame_count - self._fragment_activation_frame, 0)
        if frames_since >= self.fragment_impulse_boost_frames:
            return 1.0
        decay = self.fragment_impulse_decay ** frames_since
        return 1.0 + (self.fragment_event_boost - 1.0) * max(decay, 0.50)

    def _ensure_render_fragment_registry(self, count: int, device) -> None:
        if (self._render_fragment_labels is not None
                and self._render_fragment_labels.shape[0] == count
                and self._render_fragment_labels.device == device):
            return
        self._render_fragment_labels = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self._render_fragment_states = {}
        self._next_render_fragment_id = 1

    def _advance_render_fragment_states(self) -> None:
        if not self._render_fragment_states:
            return
        gravity_step = float(self.fragment_render_gravity)
        damp_z = float(self.fragment_render_damping)
        damp_xy = float(self.fragment_render_lateral_damping)
        max_detach = 0.45
        if self.crack_connected_release_only:
            gravity_step *= max(0.0, float(getattr(self, "fragment_render_strict_gravity_scale", 0.0)))
            damp_z = min(damp_z, float(getattr(self, "fragment_render_strict_damping", 0.90)))
            damp_xy = min(damp_xy, float(getattr(self, "fragment_render_strict_lateral_damping", 0.90)))
            max_detach = max(0.0, float(getattr(self, "fragment_render_strict_max_detach", 0.085)))
        if self.material_family == "rough_quasi_brittle":
            max_detach = 0.55
        elif self.material_family == "brittle_moderate":
            max_detach = 0.40
        for state in self._render_fragment_states.values():
            vel = state["velocity"]
            vel = vel.clone()
            release_score = float(state.get("release_score", 0.0))
            support_lost = bool(state.get("support_lost", False))
            local_gravity = gravity_step * (1.0 + (0.85 * release_score if support_lost else 0.0))
            local_damp_z = max(0.90, damp_z - (0.05 * release_score if support_lost else 0.0))
            vel[:2] *= damp_xy
            vel[2] = vel[2] * local_damp_z - local_gravity
            state["velocity"] = vel
            new_offset = state["offset"] + vel
            offset_norm = float(new_offset.norm().item())
            if offset_norm > max_detach:
                new_offset = new_offset * (max_detach / max(offset_norm, 1e-8))
                vel = vel * 0.55
                state["velocity"] = vel
            state["offset"] = new_offset
            state["age"] = int(state["age"]) + 1

    def _register_render_fragments(
        self,
        positions: Tensor,
        fragment_ids: Optional[Tensor],
        opening: Optional[Tensor] = None,
    ) -> None:
        if fragment_ids is None:
            return
        if self._render_fragment_labels is None:
            self._ensure_render_fragment_registry(positions.shape[0], positions.device)

        active_labels = fragment_ids.unique(sorted=True)
        impact_center_world = None
        if hasattr(self, '_impact_center'):
            impact_center_world = self.mapper.mpm_to_world(
                self._impact_center.unsqueeze(0)
            ).squeeze(0)
        else:
            impact_center_world = positions.mean(dim=0)
        base_mask = fragment_ids == 0
        base_com = positions[base_mask].mean(dim=0) if bool(base_mask.any()) else impact_center_world

        for frag_id in active_labels.tolist():
            if frag_id <= 0:
                continue
            mask = fragment_ids == frag_id
            frag_size = int(mask.sum().item())
            if frag_size < self.fragment_render_min_size:
                continue

            overlap_vals, overlap_counts = self._render_fragment_labels[mask].unique(
                return_counts=True
            )
            persistent_label = None
            if overlap_vals.numel() > 0:
                overlap_mask = overlap_vals > 0
                if bool(overlap_mask.any()):
                    overlap_vals = overlap_vals[overlap_mask]
                    overlap_counts = overlap_counts[overlap_mask]
                    best_idx = int(torch.argmax(overlap_counts).item())
                    best_count = int(overlap_counts[best_idx].item())
                    if best_count / max(frag_size, 1) >= self.fragment_render_overlap_threshold:
                        persistent_label = int(overlap_vals[best_idx].item())

            frag_pos = positions[mask]
            frag_com = frag_pos.mean(dim=0)
            release_score = 0.0
            support_lost = False
            if (self.fragment_manager is not None
                    and frag_id < len(self.fragment_manager.fragment_release_scores)):
                release_score = float(self.fragment_manager.fragment_release_scores[frag_id])
            if (self.fragment_manager is not None
                    and frag_id < len(self.fragment_manager.fragment_support_lost)):
                support_lost = bool(self.fragment_manager.fragment_support_lost[frag_id])

            direction = frag_com - impact_center_world
            if support_lost:
                direction = frag_com - base_com
                direction[2] -= 0.28 + 0.48 * release_score
            else:
                direction[2] += 0.22 * self.fragment_upward_bias
            direction = self._safe_vector_normalize(direction.unsqueeze(0)).squeeze(0)
            if float(direction.norm().item()) < 1e-8:
                direction = torch.tensor([0.0, 0.0, -1.0 if support_lost else 1.0], device=positions.device)

            size_ratio = frag_size / max(float(positions.shape[0]), 1.0)
            size_gain = max(0.85, 1.70 - 10.0 * size_ratio)
            opening_mag = 0.0
            if opening is not None:
                opening_mag = float(opening[mask].mean().item())
            event_gain = self._current_fragment_visual_boost()
            base_gap = (
                self.fragment_visual_offset_scale
                * self.fragment_offset_gain
                * self.fragment_render_gap_scale
                * size_gain
                * max(event_gain, 1.0)
                * (1.0 + 4.0 * opening_mag)
            )
            vel_mag = (
                self.fragment_visual_offset_scale
                * self.fragment_render_velocity_scale
                * size_gain
                * (0.55 + 0.35 * max(event_gain - 1.0, 0.0) + 6.0 * opening_mag)
            )
            if support_lost:
                base_gap *= 1.0 + 0.95 * release_score
                vel_mag *= 1.0 + 1.35 * release_score
            if self.crack_connected_release_only:
                motion_boost = max(0.0, float(getattr(self, "fragment_render_strict_motion_boost", 1.0)))
                base_gap *= float(getattr(self, "fragment_render_strict_gap_scale", 0.68)) * motion_boost
                vel_mag *= float(getattr(self, "fragment_render_strict_velocity_scale", 0.12)) * motion_boost

            if persistent_label is None:
                persistent_label = self._next_render_fragment_id
                self._next_render_fragment_id += 1
                self._render_fragment_states[persistent_label] = {
                    "offset": direction * base_gap,
                    "velocity": direction * vel_mag,
                    "age": 0,
                    "release_score": release_score,
                    "support_lost": support_lost,
                }
            else:
                state = self._render_fragment_states.get(persistent_label)
                if state is not None:
                    target_vel = direction * vel_mag
                    state["velocity"] = 0.72 * state["velocity"] + 0.28 * target_vel
                    offset_step = 0.04 * base_gap if self.crack_connected_release_only else 0.20 * base_gap
                    state["offset"] = state["offset"] + direction * offset_step
                    state["release_score"] = max(float(state.get("release_score", 0.0)), release_score)
                    state["support_lost"] = bool(state.get("support_lost", False) or support_lost)

            self._render_fragment_labels[mask] = persistent_label

    def _apply_persistent_fragment_separation(
        self,
        positions: Tensor,
        fragment_ids: Optional[Tensor],
        opening: Optional[Tensor] = None,
    ) -> Tensor:
        self._ensure_render_fragment_registry(positions.shape[0], positions.device)
        if fragment_ids is not None and bool((fragment_ids > 0).any()):
            self._register_render_fragments(positions, fragment_ids, opening)

        render_ids = self._render_fragment_labels.clone()
        out = positions.clone()
        if self._render_fragment_states:
            for persistent_label, state in self._render_fragment_states.items():
                mask = render_ids == persistent_label
                if not bool(mask.any()):
                    continue
                out[mask] = out[mask] + state["offset"].unsqueeze(0)
        self._advance_render_fragment_states()
        return out, render_ids
