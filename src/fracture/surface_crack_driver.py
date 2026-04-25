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
        impact_center: Optional[Tensor] = None,
        params: Optional[SurfaceCrackDriverParams] = None,
    ):
        self.material_family = str(material_family)
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

        if self.material_family == "diffuse_damage":
            growth_drive = 0.45 * growth_drive
            init_score = 0.35 * init_score

        return {
            "init_score": init_score,
            "growth_drive": growth_drive,
            "growth_dir": growth_dir,
            "impact_center": center,
        }
