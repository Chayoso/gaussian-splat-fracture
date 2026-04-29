"""
Tip-based Gaussian fracture field.

Fracture growth is represented as an explicit crack front on the Gaussian
surface graph. The damage field is only updated near that front, which keeps
the crack narrow and makes upward propagation possible when the projected
physics cues support it.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional

from .crack_front import CrackFront
from .graph_builder import GaussianGraph


class GaussianFractureField:
    """Tip-driven fracture state on Gaussian nodes."""

    def __init__(
        self,
        Gc: float = 60000.0,
        l0: float = 0.025,
        dC_max: float = 0.015,
        warmup_frames: int = 5,
        aniso_ratio: float = 3.0,
        opening_scale: float = 0.02,
        damage_source_scale: float = 0.35,
        damage_spread: float = 0.18,
        drive_quantile: float = 0.90,
        front_threshold: float = 0.05,
        radial_bias: float = 2.5,
        tip_propagation_scale: float = 0.75,
        front_substeps: int = 2,
        tau_init: float = 0.30,
        growth_gain: float = 1.00,
        band_width: float = 1.50,
        band_fill_gain: float = 0.30,
        open_gain: float = 1.00,
        material_family: str = "neutral_reference",
        enable_front_propagation: bool = True,
        material_drive_floor: Optional[float] = None,
        diffuse_damage_gain: float = 0.16,
        diffuse_neighborhood_steps: int = 2,
        graph: Optional[GaussianGraph] = None,
        crack_front: Optional[CrackFront] = None,
        device: str = "cuda",
    ):
        self.Gc = Gc
        self.l0 = l0
        self.dC_max = dC_max
        self.warmup_frames = warmup_frames
        self.aniso_ratio = aniso_ratio
        self.opening_scale = opening_scale
        self.damage_source_scale = damage_source_scale
        self.damage_spread = damage_spread
        self.drive_quantile = drive_quantile
        self.front_threshold = front_threshold
        self.radial_bias = radial_bias
        self.tip_propagation_scale = tip_propagation_scale
        self.front_substeps = max(int(front_substeps), 1)
        self.tau_init = tau_init
        self.growth_gain = growth_gain
        self.band_width = band_width
        self.band_fill_gain = band_fill_gain
        self.open_gain = open_gain
        self.material_family = str(material_family)
        self.enable_front_propagation = bool(enable_front_propagation) and self.material_family != "diffuse_damage"
        self.material_drive_floor = (
            float(material_drive_floor)
            if material_drive_floor is not None
            else max(
                0.02,
                min(
                    0.28,
                    0.05 + 0.22 * (self.growth_gain - 1.0) + 0.30 * (0.30 - self.tau_init),
                ),
            )
        )
        if self.material_family == "diffuse_damage":
            self.material_drive_floor = 0.0
        self.diffuse_damage_gain = float(diffuse_damage_gain)
        self.diffuse_neighborhood_steps = max(int(diffuse_neighborhood_steps), 1)
        self.device = torch.device(device)

        self.graph = graph or GaussianGraph(device=device)
        self.crack_front = crack_front or CrackFront(device=device)

        self.c: Optional[Tensor] = None
        self.H: Optional[Tensor] = None
        self.n: Optional[Tensor] = None
        self.a: Optional[Tensor] = None
        self.f: Optional[Tensor] = None
        self.seed_center: Optional[Tensor] = None

        self._frame_count: int = 0
        self._initialized: bool = False
        self._phase_gate: Optional[Tensor] = None
        self._seed_phase_gate: Optional[Tensor] = None

    def initialize(self, N: int) -> None:
        device = self.device
        self.c = torch.zeros(N, device=device)
        self.H = torch.zeros(N, device=device)
        self.n = torch.zeros(N, 3, device=device)
        self.a = torch.zeros(N, device=device)
        self.f = torch.zeros(N, dtype=torch.long, device=device)
        self.seed_center = None
        self._frame_count = 0
        self._initialized = True
        self._phase_gate = torch.zeros(N, device=device)
        self._seed_phase_gate = torch.zeros(N, device=device)
        self.crack_front.initialize(N, device)

    @staticmethod
    def _safe_normalize(v: Tensor) -> Tensor:
        if v.numel() == 0:
            return v
        if v.ndim == 1:
            scale = v.max().clamp(min=1e-8)
            return (v / scale).clamp(0.0, 1.0)
        norm = v.norm(dim=1, keepdim=True)
        return torch.where(norm > 1e-8, v / norm.clamp(min=1e-8), torch.zeros_like(v))

    @staticmethod
    def _apply_gain(values: Tensor, gain: float) -> Tensor:
        gain = max(float(gain), 1e-4)
        return 1.0 - torch.pow(1.0 - values.clamp(0.0, 1.0), gain)

    def _normalize_drive(self, values: Tensor, quantile: Optional[float] = None) -> Tensor:
        if values.numel() == 0:
            return values
        q = float(min(max(quantile if quantile is not None else self.drive_quantile, 0.5), 0.999))
        floor = torch.quantile(values.detach(), q)
        span = (values.max() - floor).clamp(min=1e-8)
        return ((values - floor) / span).clamp(0.0, 1.0)

    def update(
        self,
        positions: Tensor,
        init_score: Tensor,
        growth_drive: Tensor,
        growth_dir: Optional[Tensor] = None,
        F_gaussian: Optional[Tensor] = None,
        impact_center: Optional[Tensor] = None,
        phase_gate: Optional[Tensor] = None,
        seed_phase_gate: Optional[Tensor] = None,
    ) -> None:
        N = positions.shape[0]
        if not self._initialized or self.c.shape[0] != N:
            self.initialize(N)

        self._frame_count += 1
        self.graph.build(positions)

        if impact_center is not None:
            self.seed_center = impact_center.detach().clone()

        init_score = self._normalize_drive(init_score.clamp(min=0.0), quantile=0.96)
        growth_drive = self._normalize_drive(growth_drive.clamp(min=0.0))
        growth_drive = self._apply_gain(growth_drive, self.growth_gain)
        if phase_gate is None:
            phase_gate = torch.ones_like(growth_drive)
        else:
            phase_gate = phase_gate.to(device=positions.device, dtype=growth_drive.dtype)
            phase_gate = phase_gate[:N].clamp(0.0, 1.0)
        if seed_phase_gate is None:
            seed_phase_gate = phase_gate
        else:
            seed_phase_gate = seed_phase_gate.to(device=positions.device, dtype=init_score.dtype)
            seed_phase_gate = seed_phase_gate[:N].clamp(0.0, 1.0)
        self._phase_gate = phase_gate.detach()
        self._seed_phase_gate = seed_phase_gate.detach()
        init_score = init_score * seed_phase_gate
        growth_drive = growth_drive * phase_gate
        if growth_dir is None:
            growth_dir = torch.zeros(N, 3, device=positions.device)
        growth_dir = self._safe_normalize(growth_dir)

        self.H = torch.maximum(self.H, growth_drive)
        if self._frame_count <= self.warmup_frames:
            return

        seeded = 0
        advanced = 0
        if not self.enable_front_propagation:
            self._update_diffuse_damage(init_score, growth_drive)
        else:
            if not self.crack_front.has_active_tips():
                seeded = self._seed_front(positions, init_score, growth_drive, growth_dir)
            else:
                max_substeps = 4
                if self.material_family == "sharp_brittle":
                    max_substeps = 32
                elif self.material_family == "brittle_moderate":
                    max_substeps = 16
                elif self.material_family == "rough_quasi_brittle":
                    max_substeps = 10
                substeps = max(
                    1,
                    min(
                        max_substeps,
                        int(round(self.front_substeps * max(0.75, self.growth_gain))),
                    ),
                )
                burst_damage_updates = (
                    self.material_family == "sharp_brittle"
                    and substeps > 4
                )
                for substep_idx in range(substeps):
                    if not self.crack_front.has_active_tips():
                        break
                    advanced_now = self.crack_front.advance(
                        self.graph,
                        positions,
                        growth_drive,
                        growth_dir,
                        impact_center=self.seed_center,
                    )
                    advanced += advanced_now
                    if (
                        burst_damage_updates
                        and advanced_now > 0
                        and (substep_idx + 1) % 3 == 0
                    ):
                        self._update_damage_band(init_score, growth_drive)
                if not self.crack_front.has_active_tips():
                    seeded = self._seed_front(positions, init_score, growth_drive, growth_dir)
            self._update_damage_band(init_score, growth_drive)
        self._estimate_crack_normal(positions, growth_dir)
        self._compute_opening(F_gaussian)

        if self._frame_count % 10 == 0 or self._frame_count < 5:
            tip_count = int(self.crack_front.tip_mask.sum().item())
            visited_count = int(self.crack_front.visited_mask.sum().item())
            cracked_count = int((self.c > 0.3).sum().item())
            print(
                f"[GFF] frame={self._frame_count} "
                f"c_max={self.c.max():.4f} c_mean={self.c.mean():.4f} "
                f"tips={tip_count} visited={visited_count} "
                f"seeded={seeded} advanced={advanced} cracked={cracked_count}"
            )

    def _seed_front(
        self,
        positions: Tensor,
        init_score: Tensor,
        growth_drive: Tensor,
        growth_dir: Tensor,
    ) -> int:
        front_support = torch.maximum(self.c, self.graph.weighted_neighbor_max(self.c))
        front_support = self._safe_normalize(front_support)
        candidate_score = (
            0.65 * init_score
            + 0.35 * front_support
            + 0.20 * growth_drive * (front_support > self.front_threshold).float()
        ).clamp(0.0, 1.0)
        candidate_score = self._apply_gain(
            candidate_score,
            0.60 + 0.60 * self.growth_gain,
        )
        phase_gate = (
            self._seed_phase_gate
            if self._seed_phase_gate is not None and self._seed_phase_gate.shape[0] == candidate_score.shape[0]
            else torch.ones_like(candidate_score)
        )
        candidate_score = candidate_score * phase_gate
        candidate_score = candidate_score * (front_support > 0.5 * self.front_threshold).float()
        return self.crack_front.seed_from_scores(
            candidate_score,
            positions,
            growth_dir,
            impact_center=self.seed_center,
        )

    def _update_damage_band(self, init_score: Tensor, growth_drive: Tensor) -> None:
        tip_mask = self.crack_front.tip_mask
        visited_mask = self.crack_front.visited_mask
        if tip_mask is None or visited_mask is None or not bool(visited_mask.any()):
            return

        tip_f = tip_mask.float()
        visited_f = visited_mask.float()
        phase_gate = (
            self._phase_gate
            if self._phase_gate is not None and self._phase_gate.shape[0] == growth_drive.shape[0]
            else torch.ones_like(growth_drive)
        ).clamp(0.0, 1.0)
        style = getattr(self.crack_front, "crack_style", "material_default")
        band_field = tip_f.clone()
        band_hops = max(1, int(round(self.band_width)))
        if self.material_family == "sharp_brittle":
            band_hops = min(band_hops, 1)
        for _ in range(band_hops):
            nbr_band = band_field[self.graph.knn_idx]
            band_field = torch.maximum(
                band_field,
                (self.graph.weights * nbr_band).sum(dim=1).clamp(0.0, 1.0),
            )
        tip_ring = (band_field * (1.0 - tip_f)).clamp(0.0, 1.0)

        parent_mask = torch.zeros_like(tip_f)
        active_tip_idx = torch.where(tip_mask)[0]
        if active_tip_idx.numel() > 0:
            parent_idx = self.crack_front.parent_index[active_tip_idx]
            valid_parent = parent_idx >= 0
            if valid_parent.any():
                parent_mask[parent_idx[valid_parent]] = 1.0

        drive_local = torch.maximum(
            growth_drive.clamp(0.0, 1.0),
            torch.full_like(growth_drive, self.material_drive_floor) * phase_gate,
        )
        init_local = init_score.clamp(0.0, 1.0)

        tip_floor = (
            0.16 + 0.08 * self.open_gain + 0.10 * drive_local
        ).clamp(0.0, 0.38) * tip_f * phase_gate
        core_floor = (
            0.12 + 0.05 * self.open_gain + 0.08 * drive_local
        ).clamp(0.0, 0.30) * parent_mask * phase_gate
        self.c = torch.maximum(self.c, tip_floor)
        self.c = torch.maximum(self.c, core_floor)

        growth_boost = 0.65 + 0.35 * self.growth_gain
        brittle_boost = 0.85 + 0.25 * self.open_gain
        fill_boost = 0.30 + 0.90 * self.band_fill_gain
        dc_tip = (
            self.damage_source_scale
            * growth_boost
            * brittle_boost
            * (0.45 + 0.55 * drive_local)
            * tip_f
            * phase_gate
        )
        dc_core = (
            0.65
            * self.tip_propagation_scale
            * growth_boost
            * brittle_boost
            * (0.25 + 0.75 * drive_local)
            * parent_mask
            * phase_gate
        )
        dc_band = (
            self.damage_spread
            * fill_boost
            * (0.15 + 0.85 * drive_local)
            * tip_ring
            * (1.0 - tip_f)
            * phase_gate
        )
        dc_seed = (
            0.20
            * self.damage_source_scale
            * (0.50 + 0.50 * self.band_fill_gain)
            * init_local
            * tip_ring
            * (1.0 - visited_f)
            * phase_gate
        )

        if self.material_family == "sharp_brittle":
            tip_floor = tip_floor * 1.22
            core_floor = core_floor * 1.18
            dc_tip = dc_tip * 1.18
            dc_core = dc_core * 1.18
            dc_band = dc_band * 0.56
            dc_seed = dc_seed * 0.48
            if style in {"radial_shatter", "spiderweb_branching"}:
                dc_band = dc_band * 0.72
                dc_seed = dc_seed * 0.58
            elif style == "single_smooth":
                dc_band = dc_band * 0.54
                dc_seed = dc_seed * 0.45
        elif self.material_family == "brittle_moderate":
            dc_tip = dc_tip * 1.08
            dc_core = dc_core * 1.05
            dc_band = dc_band * 0.82
            dc_seed = dc_seed * 0.75
        elif self.material_family == "rough_quasi_brittle":
            tip_floor = tip_floor * 0.92
            dc_tip = dc_tip * 0.96
            dc_core = dc_core * 0.90
            dc_band = dc_band * 1.28
            dc_seed = dc_seed * 1.22
        elif self.material_family == "neutral_reference":
            dc_band = dc_band * 0.88

        dc = (dc_tip + dc_core + dc_band + dc_seed).clamp(0.0, self.dC_max)
        self.c = (self.c + dc).clamp(0.0, 1.0)
        self.H = torch.maximum(self.H, drive_local * torch.maximum(tip_f, tip_ring))

    def _update_diffuse_damage(self, init_score: Tensor, growth_drive: Tensor) -> None:
        base = (0.45 * init_score.clamp(0.0, 1.0) + 0.55 * growth_drive.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        phase_gate = (
            self._phase_gate
            if self._phase_gate is not None and self._phase_gate.shape[0] == base.shape[0]
            else torch.ones_like(base)
        ).clamp(0.0, 1.0)
        base = base * phase_gate
        base = self._apply_gain(base, max(0.45, 0.65 * self.growth_gain))

        field = base
        for _ in range(self.diffuse_neighborhood_steps):
            nbr = field[self.graph.knn_idx]
            field = torch.maximum(
                field,
                (self.graph.weights * nbr).sum(dim=1).clamp(0.0, 1.0),
            )

        neighbor_field = field[self.graph.knn_idx]
        smooth = 0.55 * field + 0.45 * (self.graph.weights * neighbor_field).sum(dim=1)
        dc = (
            self.dC_max
            * self.diffuse_damage_gain
            * (0.20 + 0.80 * smooth)
        ).clamp(0.0, 0.65 * self.dC_max)
        cap = min(0.28, 0.20 + 0.18 * self.diffuse_damage_gain)
        self.c = torch.maximum(self.c, 0.08 * smooth)
        self.c = (self.c + dc).clamp(0.0, cap)
        self.H = torch.maximum(self.H, smooth)

    def _estimate_crack_normal(
        self,
        positions: Tensor,
        growth_dir_hint: Optional[Tensor] = None,
    ) -> None:
        damaged = self.c > 0.05
        if not damaged.any():
            return

        grad_c = self.graph.graph_gradient(self.c, positions)
        grad_mag = grad_c.norm(dim=1)
        has_grad = damaged & (grad_mag > 1e-4)
        if has_grad.any():
            self.n[has_grad] = grad_c[has_grad] / grad_mag[has_grad].unsqueeze(1)

        remaining_idx = torch.where(damaged & ~has_grad)[0]
        if remaining_idx.numel() == 0:
            return

        normals = getattr(self.graph, "_normals", None)
        if growth_dir_hint is not None and normals is not None and normals.shape[0] == positions.shape[0]:
            opening_hint = torch.cross(
                normals[remaining_idx],
                growth_dir_hint[remaining_idx],
                dim=1,
            )
            opening_mag = opening_hint.norm(dim=1, keepdim=True)
            valid = opening_mag.squeeze(1) > 1e-8
            if valid.any():
                self.n[remaining_idx[valid]] = opening_hint[valid] / opening_mag[valid].clamp(min=1e-8)
                remaining_idx = remaining_idx[~valid]

        if remaining_idx.numel() == 0 or growth_dir_hint is None:
            return

        hint_norm = growth_dir_hint[remaining_idx].norm(dim=1, keepdim=True)
        valid = hint_norm.squeeze(1) > 1e-8
        if valid.any():
            self.n[remaining_idx[valid]] = (
                growth_dir_hint[remaining_idx[valid]]
                / hint_norm[valid].clamp(min=1e-8)
            )

    def _compute_opening(self, F_gaussian: Optional[Tensor] = None) -> None:
        if self.material_family == "diffuse_damage":
            self.a = 0.05 * self.opening_scale * (self.c ** 2)
            return
        if F_gaussian is not None and self.n is not None:
            Fn = torch.bmm(F_gaussian, self.n.unsqueeze(2)).squeeze(2)
            stretch = Fn.norm(dim=1)
            tensile_stretch = (stretch - 1.0).clamp(min=0.0)
            self.a = (
                self.opening_scale
                * self.open_gain
                * (self.c ** 2)
                * (1.0 + 4.0 * tensile_stretch)
            )
        else:
            self.a = self.opening_scale * self.open_gain * (self.c ** 2)

        if self.material_family == "sharp_brittle":
            self.a = self.a * 1.35
        elif self.material_family == "brittle_moderate":
            self.a = self.a * 1.12
        elif self.material_family == "rough_quasi_brittle":
            self.a = self.a * 0.88
        elif self.material_family == "neutral_reference":
            self.a = self.a * 0.90

        if self.crack_front.tip_mask is not None:
            self.a = self.a * (
                1.0 + (0.15 + 0.10 * self.open_gain) * self.crack_front.tip_mask.float()
            )

    def seed_damage(
        self,
        positions: Tensor,
        center: Tensor,
        radius: float,
        magnitude: float = 0.8,
        H_multiplier: float = 0.0,
    ) -> None:
        if not self._initialized:
            self.initialize(positions.shape[0])

        radius = max(float(radius), 1e-6)
        dist = (positions - center.unsqueeze(0)).norm(dim=1)
        t = (1.0 - dist / radius).clamp(0.0, 1.0)
        influence = t * t

        c_seed = influence * magnitude
        self.c = torch.maximum(self.c, c_seed)
        self.seed_center = center.detach().clone()

        if H_multiplier > 0.0:
            H_ref = self.Gc / (2.0 * self.l0)
            H_seed = influence * H_ref * H_multiplier
            self.H = torch.maximum(self.H, H_seed)

        n_seeded = int((c_seed > 1e-6).sum().item())
        print(
            f"[GFF] Seeded damage: {n_seeded} Gaussians, "
            f"c_max={self.c.max():.3f} H_max={self.H.max():.1f}"
        )

    def get_state(self) -> dict:
        return {
            "c": self.c,
            "H": self.H,
            "n": self.n,
            "a": self.a,
            "f": self.f,
            "seed_center": self.seed_center,
            "frame_count": self._frame_count,
            "crack_front": self.crack_front.get_state(),
        }

    def load_state(self, state: dict) -> None:
        self.c = state["c"]
        self.H = state["H"]
        self.n = state["n"]
        self.a = state["a"]
        self.f = state.get("f", torch.zeros_like(self.c, dtype=torch.long))
        self.seed_center = state.get("seed_center", None)
        self._frame_count = state.get("frame_count", 0)
        self._initialized = True
        if "crack_front" in state:
            self.crack_front.load_state(state["crack_front"])
        else:
            self.crack_front.initialize(self.c.shape[0], self.c.device)
