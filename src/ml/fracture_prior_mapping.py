"""Pure data-transformation: fracture prior dict → runtime override dict.

Extracted from ``MaterialPriorAdapter.fracture_prior_to_runtime_overrides``
because the function is a flat ~200-line parameter-mapping table that
doesn't depend on any class state.  Keeping it here makes the
adapter file shorter and lets the mapping be reviewed / tuned without
touching the rest of the prior-adapter machinery.

Usage:

    from src.ml.fracture_prior_mapping import (
        fracture_prior_to_runtime_overrides as compute_overrides,
    )

    runtime = compute_overrides(
        prior, family,
        family_runtime_presets=FAMILY_RUNTIME_PRESETS,
        enforce_crack_fn=MaterialPriorAdapter._enforce_crack_connected_fragment_runtime,
    )

The two callable / dict dependencies are passed in explicitly so this
module has zero imports from ``material_prior_adapter`` (avoiding a
circular dependency).
"""

from __future__ import annotations

from typing import Callable, Dict


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def fracture_prior_to_runtime_overrides(
    fracture_prior: Dict[str, float],
    family: str = "neutral_reference",
    *,
    family_runtime_presets: Dict[str, Dict[str, object]],
    enforce_crack_fn: Callable[[Dict[str, object], str], Dict[str, object]],
) -> Dict[str, object]:
    """Translate a fracture prior dict into runtime override keys.

    Args:
        fracture_prior:  Fracture prior with keys ``tau_init``,
            ``growth_gain``, ``band_width``, ``band_fill_gain``,
            ``open_gain``, ``split_threshold``, ``edge_break_rate``,
            ``branching_bias``, ``anisotropy_strength``.
        family:  Material family used to apply per-family clamps.
        family_runtime_presets:  ``MaterialPriorAdapter`` family preset
            dict (passed in to avoid circular imports).
        enforce_crack_fn:  Callable applying the crack-connected
            fragment-runtime constraints.

    Returns:
        A flat dict of ``"manifold.*"`` and ``"gaussian_splatting.*"``
        runtime override keys ready to be merged with the simulator's
        config.
    """
    tau = float(fracture_prior["tau_init"])
    growth = float(fracture_prior["growth_gain"])
    band_width = float(fracture_prior["band_width"])
    band_fill = float(fracture_prior["band_fill_gain"])
    open_gain = float(fracture_prior["open_gain"])
    split_threshold = float(fracture_prior["split_threshold"])
    edge_break_rate = float(fracture_prior["edge_break_rate"])
    branching = float(fracture_prior["branching_bias"])
    anisotropy = float(fracture_prior["anisotropy_strength"])

    successor_topk = 1 if branching < 0.18 else (2 if branching < 0.60 else 3)
    max_branching_tips = int(round(6 + 18 * branching))
    branch_score_ratio = _clamp(0.99 - 0.18 * branching, 0.75, 0.99)
    branch_drive_threshold = _clamp(0.82 - 0.42 * branching, 0.18, 0.85)
    damage_spread = _clamp(0.08 + 0.08 * (band_width - 1.0) + 0.48 * band_fill, 0.08, 0.95)
    damage_source_scale = _clamp(0.18 + 0.22 * growth + 0.10 * band_fill, 0.12, 0.70)
    tip_propagation_scale = _clamp(0.25 + 0.50 * growth, 0.20, 1.20)
    opening_scale = _clamp(0.006 + 0.018 * open_gain, 0.004, 0.050)
    dC_max = _clamp(0.014 + 0.010 * growth + 0.005 * open_gain, 0.018, 0.045)
    drive_quantile = _clamp(0.95 - 0.10 * branching - 0.10 * (band_width - 1.0), 0.55, 0.98)
    front_threshold = _clamp(0.02 + 0.16 * tau, 0.02, 0.18)
    gaussian_damage_threshold = _clamp(0.28 - 0.08 * open_gain + 0.03 * band_width, 0.12, 0.30)
    crack_max_opening = _clamp(0.008 + 0.010 * open_gain, 0.006, 0.035)
    crack_gap_fraction = _clamp(0.25 + 0.20 * open_gain, 0.20, 0.75)
    crack_opacity_reduction = _clamp(0.48 + 0.24 * open_gain, 0.30, 0.95)
    crack_edge_darken = _clamp(0.65 + 0.35 * open_gain, 0.40, 1.40)
    crack_red_accent = _clamp(0.06 + 0.10 * growth + 0.10 * branching, 0.06, 0.35)
    impact_seed_magnitude = _clamp(0.14 + 0.08 * (1.0 - tau) + 0.02 * open_gain, 0.14, 0.28)
    split_gap_gain = _clamp(0.85 + 0.32 * open_gain, 0.70, 1.60)
    fragment_shell_gain = _clamp(0.80 + 0.26 * band_width + 0.12 * branching, 0.80, 1.60)
    fragment_contrast_gain = _clamp(0.82 + 0.18 * growth + 0.16 * open_gain, 0.75, 1.55)
    debris_darkening = _clamp(0.08 + 0.16 * band_fill + 0.10 * branching, 0.04, 0.42)
    shard_scale_gain = _clamp(0.85 + 0.20 * open_gain + 0.15 * branching, 0.75, 1.50)
    shard_opacity_gain = _clamp(0.82 + 0.16 * growth + 0.10 * open_gain, 0.75, 1.35)
    shard_count_scale = _clamp(0.18 + 0.55 * edge_break_rate + 0.30 * branching, 0.0, 1.60)
    fragment_offset_gain = _clamp(0.82 + 0.18 * open_gain + 0.12 * edge_break_rate, 0.80, 1.60)
    debris_motion_gain = _clamp(0.06 + 0.30 * branching + 0.22 * max(edge_break_rate - 0.8, 0.0), 0.0, 1.60)
    fragment_damage_threshold = _clamp(
        0.58 * split_threshold + 0.12 / max(edge_break_rate, 1e-6),
        0.18,
        0.78,
    )

    if family == "sharp_brittle":
        successor_topk = 1
        max_branching_tips = min(max_branching_tips, 8)
        branch_score_ratio = max(branch_score_ratio, 0.92)
        branch_drive_threshold = max(branch_drive_threshold, 0.62)
        damage_spread *= 0.65
        damage_source_scale *= 1.12
        tip_propagation_scale *= 1.05
        opening_scale *= 1.35
        dC_max *= 0.92
        drive_quantile = max(drive_quantile, 0.78)
        gaussian_damage_threshold = min(gaussian_damage_threshold + 0.03, 0.34)
        crack_max_opening *= 1.35
        crack_gap_fraction *= 1.16
        crack_opacity_reduction = min(crack_opacity_reduction + 0.08, 0.98)
        crack_edge_darken = min(crack_edge_darken + 0.12, 1.55)
        crack_red_accent = max(crack_red_accent - 0.03, 0.05)
        fragment_damage_threshold *= 0.82
        split_gap_gain *= 1.18
        fragment_shell_gain *= 1.08
        fragment_contrast_gain *= 1.18
        debris_darkening *= 0.85
        shard_scale_gain *= 1.08
        shard_opacity_gain *= 1.08
        shard_count_scale *= 0.85
        fragment_offset_gain *= 1.10
        debris_motion_gain *= 0.85
    elif family == "brittle_moderate":
        successor_topk = min(successor_topk, 2)
        max_branching_tips = min(max_branching_tips, 10)
        branch_score_ratio = max(branch_score_ratio, 0.88)
        branch_drive_threshold = max(branch_drive_threshold, 0.48)
        damage_spread *= 0.84
        opening_scale *= 1.10
        crack_gap_fraction *= 1.05
        fragment_damage_threshold *= 0.92
        split_gap_gain *= 1.02
        fragment_shell_gain *= 0.96
        fragment_contrast_gain *= 0.95
        debris_darkening *= 0.90
        shard_scale_gain *= 0.92
        shard_opacity_gain *= 0.95
        shard_count_scale *= 0.15
        fragment_offset_gain *= 0.98
        debris_motion_gain *= 0.20
    elif family == "rough_quasi_brittle":
        successor_topk = max(successor_topk, 2)
        max_branching_tips = max(max_branching_tips, 16)
        branch_score_ratio = max(branch_score_ratio - 0.06, 0.72)
        branch_drive_threshold = max(branch_drive_threshold - 0.10, 0.18)
        damage_spread *= 1.22
        damage_source_scale *= 0.92
        opening_scale *= 0.88
        gaussian_damage_threshold = max(gaussian_damage_threshold - 0.04, 0.10)
        crack_gap_fraction *= 0.92
        crack_opacity_reduction = max(crack_opacity_reduction - 0.10, 0.28)
        crack_red_accent = min(crack_red_accent + 0.05, 0.40)
        fragment_damage_threshold *= 1.04
        split_gap_gain *= 1.06
        fragment_shell_gain *= 1.24
        fragment_contrast_gain *= 1.08
        debris_darkening *= 1.28
        shard_scale_gain *= 1.18
        shard_opacity_gain *= 0.96
        shard_count_scale *= 1.28
        fragment_offset_gain *= 1.12
        debris_motion_gain *= 1.30
    elif family == "diffuse_damage":
        successor_topk = 0
        max_branching_tips = 0
        branch_score_ratio = 0.999
        branch_drive_threshold = 0.99
        damage_spread *= 1.36
        damage_source_scale *= 0.55
        tip_propagation_scale *= 0.30
        opening_scale *= 0.14
        dC_max *= 0.58
        drive_quantile = min(drive_quantile + 0.12, 0.995)
        front_threshold = 0.999
        gaussian_damage_threshold = min(gaussian_damage_threshold + 0.08, 0.36)
        crack_max_opening *= 0.25
        crack_gap_fraction *= 0.30
        crack_opacity_reduction *= 0.20
        crack_edge_darken *= 0.35
        crack_red_accent = 0.0
        impact_seed_magnitude *= 0.40
        fragment_damage_threshold = min(fragment_damage_threshold + 0.18, 0.92)
        split_gap_gain *= 0.35
        fragment_shell_gain = 0.0
        fragment_contrast_gain = 0.0
        debris_darkening = 0.0
        shard_scale_gain = 1.0
        shard_opacity_gain = 1.0
        shard_count_scale = 0.0
        fragment_offset_gain = 1.0
        debris_motion_gain = 0.0
    else:
        successor_topk = min(max(successor_topk, 1), 2)
        max_branching_tips = min(max_branching_tips, 12)
        damage_spread *= 0.92
        opening_scale *= 0.90
        crack_gap_fraction *= 0.90
        fragment_damage_threshold *= 0.96
        split_gap_gain *= 0.92
        fragment_shell_gain *= 0.96
        fragment_contrast_gain *= 0.96
        shard_count_scale *= 0.40
        debris_motion_gain *= 0.35

    runtime: Dict[str, object] = {
        "manifold.tau_init": tau,
        "manifold.growth_gain": growth,
        "manifold.band_width": band_width,
        "manifold.band_fill_gain": band_fill,
        "manifold.open_gain": open_gain,
        "manifold.edge_break_rate": edge_break_rate,
        "manifold.branching_bias": branching,
        "manifold.anisotropy_strength": anisotropy,
        "manifold.successor_topk": successor_topk,
        "manifold.max_branching_tips": max_branching_tips,
        "manifold.branch_score_ratio": branch_score_ratio,
        "manifold.branch_drive_threshold": branch_drive_threshold,
        "manifold.damage_spread": damage_spread,
        "manifold.damage_source_scale": damage_source_scale,
        "manifold.tip_propagation_scale": tip_propagation_scale,
        "manifold.opening_scale": opening_scale,
        "manifold.dC_max": dC_max,
        "manifold.drive_quantile": drive_quantile,
        "manifold.front_threshold": front_threshold,
        "manifold.impact_seed_magnitude": impact_seed_magnitude,
        "manifold.split_threshold": split_threshold,
        "manifold.opacity_threshold": _clamp(split_threshold * 0.82, 0.20, 0.88),
        "manifold.flatten_threshold": _clamp(split_threshold * 0.58, 0.16, 0.70),
        "manifold.split_offset_scale": _clamp(0.80 + 0.90 * open_gain, 0.70, 2.40),
        "manifold.fragment_damage_threshold": fragment_damage_threshold,
        "manifold.shard_count_scale": shard_count_scale,
        "manifold.fragment_offset_gain": fragment_offset_gain,
        "manifold.debris_motion_gain": debris_motion_gain,
        "gaussian_splatting.damage_threshold": gaussian_damage_threshold,
        "gaussian_splatting.crack_max_opening": crack_max_opening,
        "gaussian_splatting.crack_gap_fraction": crack_gap_fraction,
        "gaussian_splatting.crack_opacity_reduction": crack_opacity_reduction,
        "gaussian_splatting.crack_edge_darken": crack_edge_darken,
        "gaussian_splatting.crack_red_accent": crack_red_accent,
        "gaussian_splatting.split_gap_gain": split_gap_gain,
        "gaussian_splatting.fragment_shell_gain": fragment_shell_gain,
        "gaussian_splatting.fragment_contrast_gain": fragment_contrast_gain,
        "gaussian_splatting.debris_darkening": debris_darkening,
        "gaussian_splatting.shard_scale_gain": shard_scale_gain,
        "gaussian_splatting.shard_opacity_gain": shard_opacity_gain,
    }
    runtime.update(family_runtime_presets.get(
        family, family_runtime_presets["neutral_reference"]))
    runtime = enforce_crack_fn(runtime, family)
    return runtime
