"""FractureDriveMixin for ManifoldSimulator.

This module is behavior-preserving extraction from manifold_simulator.py.
"""

import math
from typing import Dict, Optional

import torch
from torch import Tensor


class FractureDriveMixin:
    def _phase_y_gate(self, phase_y: Tensor, threshold: float) -> Tensor:
        if not self.phase_y_coupling_enabled:
            return torch.ones_like(phase_y)
        threshold = max(float(threshold), 1e-8)
        gate = (phase_y / threshold).clamp(0.0, 1.0)
        if self.phase_y_gate_width > 0.0:
            exponent = max(0.35, min(1.75, float(self.phase_y_gate_width)))
            gate = gate.pow(exponent)
        return gate.clamp(0.0, 1.0)

    def _damage_memory_gate(self, damage: Optional[Tensor]) -> Tensor:
        """Allow existing localized crack damage to keep the narrow-band gate open.

        The thin-shell proxy does not have a true volume phase field, so pure
        Griffith-Y can lag immediately after contact.  Seeded contact damage is
        the local phase-field memory in that case: it should not create
        fragments alone, but it must allow the crack front to advance and build
        a cut surface while the stress field catches up.
        """
        ref = self._last_phase_y_gauss
        if ref is None:
            if damage is None:
                return torch.empty(0, device=self.x_mpm.device if self.x_mpm is not None else self.mpm.gravity.device)
            ref = damage
        if damage is None or self.material_family == "diffuse_damage":
            return torch.zeros_like(ref)
        c = damage[: ref.shape[0]].to(device=ref.device, dtype=ref.dtype).clamp(0.0, 1.0)
        thresholds = {
            "sharp_brittle": 0.16,
            "brittle_moderate": 0.18,
            "rough_quasi_brittle": 0.15,
            "neutral_reference": 0.30,
        }
        threshold = thresholds.get(self.material_family, 0.30)
        return (c / max(threshold, 1e-6)).clamp(0.0, 1.0).pow(0.72)

    @staticmethod
    def _robust_normalize(values: Tensor, quantile: float = 0.95) -> Tensor:
        """Normalize a nonnegative field by a robust upper quantile."""
        q = float(min(max(quantile, 0.5), 0.999))
        scale = torch.quantile(values.detach(), q).clamp(min=1e-8)
        return (values / scale).clamp(0.0, 2.0)

    def _build_effective_fracture_drive(
        self,
        psi_mpm: Tensor,
        stress: Optional[Tensor],
    ) -> Tensor:
        """
        Build a fracture drive that is not locked to pure contact tension.

        Mix tensile energy with deviatoric stress, principal tensile stress,
        and relative kinetic activity so impact waves can continue driving
        a narrow crack band after the initial contact patch.
        """
        drive = self.drive_tension_weight * self._robust_normalize(psi_mpm, 0.90)

        if stress is not None:
            S = 0.5 * (stress + stress.transpose(1, 2))
            S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e8, 1e8)
            tr = torch.diagonal(S, dim1=1, dim2=2).sum(dim=1) / 3.0
            I = torch.eye(3, device=S.device).unsqueeze(0)
            dev = S - tr.view(-1, 1, 1) * I
            von_mises = torch.sqrt(
                torch.clamp(1.5 * (dev ** 2).sum(dim=(1, 2)), min=0.0))
            try:
                sigma1 = torch.linalg.eigvalsh(S)[:, -1].clamp(min=0.0)
            except RuntimeError:
                sigma1 = torch.diagonal(S, dim1=1, dim2=2).max(dim=1).values.clamp(min=0.0)

            drive = drive + (
                self.drive_shear_weight
                * self._robust_normalize(von_mises, 0.92)
            )
            drive = drive + (
                self.drive_principal_weight
                * self._robust_normalize(sigma1, 0.92)
            )

        if self.v_mpm is not None:
            v_rel = self.v_mpm - self.v_mpm.mean(dim=0, keepdim=True)
            speed_rel = v_rel.norm(dim=1)
            drive = drive + (
                self.drive_kinetic_weight
                * self._robust_normalize(speed_rel, 0.95)
            )

        return drive

    @staticmethod
    def _safe_vector_normalize(vectors: Tensor) -> Tensor:
        norm = vectors.norm(dim=1, keepdim=True)
        return torch.where(
            norm > 1e-8,
            vectors / norm.clamp(min=1e-8),
            torch.zeros_like(vectors),
        )

    def _build_growth_direction(
        self,
        stress_dir: Optional[Tensor],
        x_surf_world: Tensor,
        impact_center_world: Optional[Tensor],
    ) -> Tensor:
        if stress_dir is None:
            growth_dir = torch.zeros_like(x_surf_world)
        else:
            growth_dir = self._safe_vector_normalize(stress_dir)

        if impact_center_world is not None:
            radial = self._safe_vector_normalize(
                x_surf_world - impact_center_world.unsqueeze(0))
            if stress_dir is None:
                growth_dir = radial
            else:
                growth_dir = self._safe_vector_normalize(0.35 * growth_dir + 0.65 * radial)

        return growth_dir

    def _apply_sentence_crack_style_drive(
        self,
        x_surf_world: Tensor,
        init_score: Tensor,
        growth_drive: Tensor,
        growth_dir: Tensor,
        impact_center_world: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Shape impact drive into sentence-level crack motifs.

        This is a graphics-side controller layered on top of physical stress:
        it only biases where the surface crack front can run. Material priors
        still control whether those cracks can detach fragments.
        """
        style = self.crack_style
        if (
            impact_center_world is None
            or style in {"material_default", "diffuse_microcrack"}
            or x_surf_world.numel() == 0
        ):
            if style == "diffuse_microcrack":
                return 0.45 * init_score, 0.38 * growth_drive, growth_dir
            return init_score, growth_drive, growth_dir

        rel = x_surf_world - impact_center_world.unsqueeze(0)
        extent = x_surf_world.max(dim=0).values - x_surf_world.min(dim=0).values
        diag = extent.norm().clamp(min=1e-6)
        planar = rel[:, :2]
        planar_r = planar.norm(dim=1).clamp(min=1e-8)
        r_norm = (planar_r / (0.45 * diag)).clamp(0.0, 1.0)
        theta = torch.atan2(planar[:, 1], planar[:, 0])
        radial = torch.zeros_like(rel)
        radial[:, :2] = planar / planar_r.unsqueeze(1).clamp(min=1e-8)
        upward = torch.zeros_like(rel)
        upward[:, 2] = 1.0
        tangent = torch.zeros_like(rel)
        tangent[:, 0] = -radial[:, 1]
        tangent[:, 1] = radial[:, 0]

        if style == "radial_shatter":
            ray_count = 14.0
            phase = 0.45
            ray = (0.5 + 0.5 * torch.cos(ray_count * theta + phase)).clamp(0.0, 1.0)
            ray = ray.pow(2.3)
            rings = (0.5 + 0.5 * torch.cos(22.0 * r_norm + 0.35)).clamp(0.0, 1.0)
            rings = rings.pow(2.0)
            near = torch.exp(-0.5 * (planar_r / (0.15 * diag)).pow(2.0))
            far_gain = (0.16 + 0.84 * r_norm).clamp(0.0, 1.0)
            ray_drive = (ray * far_gain + 0.34 * near).clamp(0.0, 1.0)
            ring_gate = (r_norm > 0.18).float()
            ring_drive = (rings * ring_gate * (0.35 + 0.65 * ray)).clamp(0.0, 1.0)
            growth_drive = torch.maximum(growth_drive, 0.94 * ray_drive)
            growth_drive = torch.maximum(growth_drive, 0.03 * ring_drive)
            init_score = torch.maximum(init_score, 0.74 * near * (0.30 + 0.70 * ray))
            tangent_sign = torch.sign(torch.sin(ray_count * theta + phase))
            tangent_sign = torch.where(
                tangent_sign.abs() > 0,
                tangent_sign,
                torch.ones_like(tangent_sign),
            )
            tangent = tangent * tangent_sign.unsqueeze(1)
            ring_mix = (0.08 * ring_gate * rings).unsqueeze(1)
            growth_dir = self._safe_vector_normalize(
                (1.55 - 0.18 * ring_mix) * radial
                + 0.08 * ring_mix * tangent
                + 0.16 * upward
            )
        elif style == "spiderweb_branching":
            ray = (0.5 + 0.5 * torch.cos(9.0 * theta + 0.25)).clamp(0.0, 1.0).pow(2.0)
            rings = (0.5 + 0.5 * torch.cos(30.0 * r_norm + 0.35)).clamp(0.0, 1.0).pow(2.2)
            web = (0.92 * ray + 0.08 * rings) * (0.16 + 0.84 * r_norm)
            near = torch.exp(-0.5 * (planar_r / (0.16 * diag)).pow(2.0))
            growth_drive = torch.maximum(growth_drive, 0.70 * web.clamp(0.0, 1.0))
            init_score = torch.maximum(init_score, 0.48 * near * (0.45 + 0.55 * ray))
            tangent_sign = torch.sign(torch.sin(9.0 * theta + 0.25))
            tangent_sign = torch.where(
                tangent_sign.abs() > 0,
                tangent_sign,
                torch.ones_like(tangent_sign),
            )
            tangent = tangent * tangent_sign.unsqueeze(1)
            ring_mix = (0.06 + 0.34 * rings).unsqueeze(1)
            ray_mix = (0.35 + 0.65 * ray).unsqueeze(1)
            growth_dir = self._safe_vector_normalize(
                0.92 * ray_mix * radial
                + 0.08 * ring_mix * tangent
                + 0.18 * upward
            )
        elif style == "single_smooth":
            angle = torch.tensor(0.35, dtype=x_surf_world.dtype, device=x_surf_world.device)
            axis = torch.stack([torch.cos(angle), torch.sin(angle)])
            tangent = planar @ axis
            cross = planar[:, 0] * axis[1] - planar[:, 1] * axis[0]
            width = 0.045 * diag
            line = torch.exp(-0.5 * (cross / width).pow(2.0))
            forward = (tangent > (-0.10 * diag)).float()
            line_drive = line * forward * (0.25 + 0.75 * r_norm)
            growth_drive = torch.maximum(0.38 * growth_drive, 0.70 * line_drive)
            near = torch.exp(-0.5 * (planar_r / (0.12 * diag)).pow(2.0))
            init_score = torch.maximum(init_score, 0.36 * near * line)
            axis3 = torch.zeros_like(rel)
            axis3[:, 0] = axis[0]
            axis3[:, 1] = axis[1]
            growth_dir = self._safe_vector_normalize(axis3 + 0.16 * upward)
        elif style == "chunky_crumble":
            noise = torch.sin(
                31.0 * x_surf_world[:, 0]
                - 17.0 * x_surf_world[:, 1]
                + 23.0 * x_surf_world[:, 2]
            )
            grain = (0.5 + 0.5 * noise).clamp(0.0, 1.0)
            broad = torch.exp(-0.5 * (planar_r / (0.26 * diag)).pow(2.0))
            crumble = (0.45 * broad + 0.55 * grain * (0.25 + 0.75 * r_norm)).clamp(0.0, 1.0)
            growth_drive = torch.maximum(growth_drive, 0.58 * crumble)
            init_score = torch.maximum(init_score, 0.28 * broad * (0.55 + 0.45 * grain))
            growth_dir = self._safe_vector_normalize(0.60 * radial + 0.38 * tangent + 0.18 * upward)

        return init_score.clamp(0.0, 1.0), growth_drive.clamp(0.0, 1.0), growth_dir
