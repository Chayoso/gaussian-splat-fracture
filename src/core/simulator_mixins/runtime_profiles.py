"""RuntimeProfilesMixin for ManifoldSimulator.

This module is behavior-preserving extraction from manifold_simulator.py.
"""

from typing import Dict


class RuntimeProfilesMixin:
    @staticmethod
    def _default_phase_y_thresholds(material_family: str) -> Dict[str, float]:
        """Soft Griffith gates for the thin surface phase-field proxy."""
        family = str(material_family)
        defaults = {
            "sharp_brittle": {"seed": 0.22, "advance": 0.14, "cut": 0.24, "width": 0.36},
            "brittle_moderate": {"seed": 0.34, "advance": 0.22, "cut": 0.36, "width": 0.40},
            "rough_quasi_brittle": {"seed": 0.30, "advance": 0.20, "cut": 0.32, "width": 0.44},
            "neutral_reference": {"seed": 0.42, "advance": 0.30, "cut": 0.46, "width": 0.46},
            "diffuse_damage": {"seed": 0.78, "advance": 0.56, "cut": 1.20, "width": 0.60},
        }
        return defaults.get(family, defaults["neutral_reference"])

    @staticmethod
    def _default_impact_fracture_burst(material_family: str) -> Dict[str, int]:
        """Internal crack-solver iterations per visible frame after impact."""
        family = str(material_family)
        defaults = {
            "sharp_brittle": {"frames": 2, "steps": 8, "front_substeps": 28},
            "brittle_moderate": {"frames": 2, "steps": 5, "front_substeps": 12},
            "rough_quasi_brittle": {"frames": 2, "steps": 4, "front_substeps": 8},
            "neutral_reference": {"frames": 1, "steps": 2, "front_substeps": 4},
            "diffuse_damage": {"frames": 0, "steps": 1, "front_substeps": 1},
        }
        return defaults.get(family, defaults["neutral_reference"])

    def _fracture_iterations_this_frame(self) -> int:
        if not (
            self._gravity_drop
            and self._gravity_drop_contacted
            and self.impact_fracture_burst_frames > 0
        ):
            return 1
        frames_since = int(getattr(self, "_impact_frame_count", 0))
        if frames_since >= self.impact_fracture_burst_frames:
            return 1
        return max(1, int(self.impact_fracture_burst_steps))

    @staticmethod
    def _default_shape_matching_params(material_family: str) -> Dict[str, float]:
        """Cohesive shell/fragment rigidity for the surface-particle MPM proxy."""
        family = str(material_family)
        defaults = {
            "sharp_brittle": {
                "body": 0.90, "fragment": 0.97, "damaged": 0.78,
                "restitution": 0.06, "angular_gain": 0.85, "friction": 0.16,
                "fragment_lateral_bias": 0.34, "fragment_spin_gain": 0.18,
                "fragment_max_speed": 0.42,
            },
            "brittle_moderate": {
                "body": 0.86, "fragment": 0.95, "damaged": 0.74,
                "restitution": 0.08, "angular_gain": 0.80, "friction": 0.18,
                "fragment_lateral_bias": 0.22, "fragment_spin_gain": 0.12,
                "fragment_max_speed": 0.36,
            },
            "rough_quasi_brittle": {
                "body": 0.86, "fragment": 0.96, "damaged": 0.76,
                "restitution": 0.05, "angular_gain": 0.72, "friction": 0.22,
                "fragment_lateral_bias": 0.18, "fragment_spin_gain": 0.08,
                "fragment_max_speed": 0.32,
            },
            "neutral_reference": {
                "body": 0.84, "fragment": 0.94, "damaged": 0.66,
                "restitution": 0.20, "angular_gain": 0.92, "friction": 0.26,
                "fragment_lateral_bias": 0.06, "fragment_spin_gain": 0.04,
                "fragment_max_speed": 0.30,
            },
            "diffuse_damage": {
                "body": 0.18, "fragment": 0.28, "damaged": 0.10,
                "restitution": 0.72, "angular_gain": 0.22, "friction": 0.04,
                "soft_rebound": 0.48, "soft_rebound_frames": 14,
                "soft_rebound_body_fraction": 0.22,
                "soft_squash": 0.18, "soft_squash_frames": 12,
                "fragment_lateral_bias": 0.0, "fragment_spin_gain": 0.0,
                "fragment_max_speed": 0.24,
            },
        }
        return defaults.get(family, defaults["neutral_reference"])
