"""Procedural surface-only crack drive for Gaussian-manifold smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class SurfaceCrackDriverParams:
    impact_radius: float = 0.070
    front_speed: float = 0.010
    front_width: float = 0.045
    seed_strength: float = 1.00
    drive_strength: float = 1.00
    upward_bias: float = 0.35
    roughness: float = 0.08
    branch_bias: float = 0.15


FAMILY_SURFACE_PARAMS = {
    "sharp_brittle": SurfaceCrackDriverParams(
        impact_radius=0.052,
        front_speed=0.013,
        front_width=0.026,
        seed_strength=1.10,
        drive_strength=1.12,
        upward_bias=0.42,
        roughness=0.035,
        branch_bias=0.04,
    ),
    "brittle_moderate": SurfaceCrackDriverParams(
        impact_radius=0.060,
        front_speed=0.010,
        front_width=0.035,
        seed_strength=0.95,
        drive_strength=0.92,
        upward_bias=0.34,
        roughness=0.060,
        branch_bias=0.10,
    ),
    "rough_quasi_brittle": SurfaceCrackDriverParams(
        impact_radius=0.080,
        front_speed=0.009,
        front_width=0.058,
        seed_strength=0.90,
        drive_strength=1.04,
        upward_bias=0.26,
        roughness=0.140,
        branch_bias=0.32,
    ),
    "diffuse_damage": SurfaceCrackDriverParams(
        impact_radius=0.105,
        front_speed=0.004,
        front_width=0.085,
        seed_strength=0.35,
        drive_strength=0.28,
        upward_bias=0.12,
        roughness=0.040,
        branch_bias=0.00,
    ),
    "neutral_reference": SurfaceCrackDriverParams(),
}


class SurfaceCrackDriver:
    """Builds surface crack drive fields without volume physics.

    The driver is intentionally simple and deterministic. It provides a
    material-conditioned activation wave on the surface graph, while
    GaussianFractureField remains responsible for crack-front propagation,
    damage history, normals, and opening.
    """

    def __init__(
        self,
        material_family: str = "neutral_reference",
        crack_style: str = "material_default",
        impact_center: Optional[Tensor] = None,
        params: Optional[SurfaceCrackDriverParams] = None,
    ):
        self.material_family = str(material_family)
        self.crack_style = str(crack_style)
        self.params = params or FAMILY_SURFACE_PARAMS.get(
            self.material_family,
            FAMILY_SURFACE_PARAMS["neutral_reference"],
        )
        self.impact_center = impact_center

    @staticmethod
    def _safe_normalize(vectors: Tensor) -> Tensor:
        norm = vectors.norm(dim=-1, keepdim=True)
        return torch.where(
            norm > 1e-8,
            vectors / norm.clamp(min=1e-8),
            torch.zeros_like(vectors),
        )

    @staticmethod
    def default_impact_center(positions: Tensor) -> Tensor:
        z = positions[:, 2]
        z_min = z.min()
        z_max = z.max()
        low = z <= z_min + 0.08 * (z_max - z_min).clamp(min=1e-6)
        if bool(low.any()):
            center = positions[low].mean(dim=0)
        else:
            center = positions.mean(dim=0)
            center[2] = z_min
        return center

    def _apply_style_drive(
        self,
        positions: Tensor,
        center: Tensor,
        init_score: Tensor,
        growth_drive: Tensor,
        growth_dir: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        style = self.crack_style
        if style in {"material_default", "diffuse_microcrack"} or positions.numel() == 0:
            if style == "diffuse_microcrack":
                return 0.45 * init_score, 0.38 * growth_drive, growth_dir
            return init_score, growth_drive, growth_dir

        rel = positions - center.unsqueeze(0)
        extent = positions.max(dim=0).values - positions.min(dim=0).values
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
            near = torch.exp(-0.5 * (planar_r / (0.15 * diag)).pow(2.0))
            far_gain = (0.16 + 0.84 * r_norm).clamp(0.0, 1.0)
            ray_drive = (ray * far_gain + 0.34 * near).clamp(0.0, 1.0)
            init_score = torch.maximum(init_score, 0.74 * near * (0.30 + 0.70 * ray))
            growth_drive = torch.maximum(growth_drive, 0.94 * ray_drive)
            growth_dir = self._safe_normalize(1.70 * radial + 0.18 * upward)

        elif style == "spiderweb_branching":
            spoke = (0.5 + 0.5 * torch.cos(9.0 * theta + 0.25)).clamp(0.0, 1.0).pow(2.0)
            ring = (0.5 + 0.5 * torch.cos(30.0 * r_norm + 0.35)).clamp(0.0, 1.0).pow(2.2)
            web = (0.92 * spoke + 0.08 * ring) * (0.16 + 0.84 * r_norm)
            near = torch.exp(-0.5 * (planar_r / (0.16 * diag)).pow(2.0))
            init_score = torch.maximum(init_score, 0.48 * near * (0.45 + 0.55 * spoke))
            growth_drive = torch.maximum(growth_drive, 0.70 * web.clamp(0.0, 1.0))
            tangent_sign = torch.sign(torch.sin(9.0 * theta + 0.25))
            tangent_sign = torch.where(
                tangent_sign.abs() > 0,
                tangent_sign,
                torch.ones_like(tangent_sign),
            )
            tangent = tangent * tangent_sign.unsqueeze(1)
            ring_mix = (0.06 + 0.34 * ring).unsqueeze(1)
            spoke_mix = (0.35 + 0.65 * spoke).unsqueeze(1)
            growth_dir = self._safe_normalize(
                0.92 * spoke_mix * radial
                + 0.08 * ring_mix * tangent
                + 0.18 * upward
            )

        elif style == "single_smooth":
            angle = torch.tensor(0.35, dtype=positions.dtype, device=positions.device)
            axis2 = torch.stack([torch.cos(angle), torch.sin(angle)])
            tangent_pos = planar @ axis2
            cross = planar[:, 0] * axis2[1] - planar[:, 1] * axis2[0]
            width = 0.045 * diag
            line = torch.exp(-0.5 * (cross / width).pow(2.0))
            forward = (tangent_pos > (-0.10 * diag)).float()
            line_drive = line * forward * (0.20 + 0.80 * r_norm)
            init_score = torch.maximum(init_score, 0.42 * line)
            growth_drive = torch.maximum(0.36 * growth_drive, 0.78 * line_drive)
            axis = torch.zeros_like(rel)
            axis[:, 0] = axis2[0]
            axis[:, 1] = axis2[1]
            growth_dir = self._safe_normalize(axis + 0.16 * upward)

        elif style == "chunky_crumble":
            noise = torch.sin(
                31.0 * positions[:, 0]
                - 17.0 * positions[:, 1]
                + 23.0 * positions[:, 2]
            )
            grain = (0.5 + 0.5 * noise).clamp(0.0, 1.0)
            broad = torch.exp(-0.5 * (planar_r / (0.26 * diag)).pow(2.0))
            crumble = (0.45 * broad + 0.55 * grain * (0.25 + 0.75 * r_norm)).clamp(0.0, 1.0)
            init_score = torch.maximum(init_score, 0.30 * broad * (0.55 + 0.45 * grain))
            growth_drive = torch.maximum(growth_drive, 0.60 * crumble)
            growth_dir = self._safe_normalize(0.60 * radial + 0.38 * tangent + 0.18 * upward)

        return init_score.clamp(0.0, 1.0), growth_drive.clamp(0.0, 1.0), growth_dir

    def build(
        self,
        positions: Tensor,
        frame: int,
        normals: Optional[Tensor] = None,
    ) -> dict:
        params = self.params
        center = (
            self.impact_center.to(positions.device)
            if self.impact_center is not None
            else self.default_impact_center(positions)
        )

        rel = positions - center.unsqueeze(0)
        dist = rel.norm(dim=1).clamp(min=1e-8)
        radial = self._safe_normalize(rel)

        z = positions[:, 2]
        height = (z.max() - z.min()).clamp(min=1e-6)
        z_norm = ((z - z.min()) / height).clamp(0.0, 1.0)

        front_radius = params.impact_radius + params.front_speed * float(frame)
        seed = torch.exp(-0.5 * (dist / max(params.impact_radius, 1e-6)) ** 2)
        wave = torch.exp(
            -0.5 * ((dist - front_radius) / max(params.front_width, 1e-6)) ** 2
        )

        upward = torch.zeros_like(radial)
        upward[:, 2] = 1.0
        growth_dir = self._safe_normalize(
            radial + params.upward_bias * upward
        )

        if normals is not None and normals.shape == positions.shape:
            normals = self._safe_normalize(normals)
            normal_gate = 1.0 - (growth_dir * normals).sum(dim=1).abs().clamp(0.0, 0.85)
        else:
            normal_gate = torch.ones_like(dist)

        noise = torch.sin(
            37.17 * positions[:, 0]
            + 19.91 * positions[:, 1]
            + 11.47 * positions[:, 2]
            + 0.37 * float(frame)
        )
        noise = 1.0 + params.roughness * noise
        branch = 1.0 + params.branch_bias * torch.sin(
            53.0 * positions[:, 0] - 29.0 * positions[:, 1]
        )

        init_score = (
            params.seed_strength
            * seed
            * (0.60 + 0.40 * normal_gate)
            * noise
        ).clamp(0.0, 1.0)
        growth_drive = (
            params.drive_strength
            * (0.25 * seed + 0.75 * wave)
            * (0.45 + 0.55 * z_norm)
            * (0.65 + 0.35 * normal_gate)
            * branch
            * noise
        ).clamp(0.0, 1.0)

        init_score, growth_drive, growth_dir = self._apply_style_drive(
            positions,
            center,
            init_score,
            growth_drive,
            growth_dir,
        )

        if self.material_family == "diffuse_damage":
            growth_drive = 0.45 * growth_drive
            init_score = 0.35 * init_score

        return {
            "init_score": init_score,
            "growth_drive": growth_drive,
            "growth_dir": growth_dir,
            "impact_center": center,
        }
