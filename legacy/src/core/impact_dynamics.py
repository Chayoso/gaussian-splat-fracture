"""Impact and refinement helpers for HybridCrackSimulator."""

from __future__ import annotations

import math

import torch

from src.core.crack_evolution import _get_effective_max_total_cracks, get_material_shatter_scale


def handle_ground_impact(sim):
    """Handle the moment of ground contact in gravity drop mode."""
    step = sim._physics_step
    sim._gravity_drop_contacted = True
    sim.v_mpm[:] = sim._v_com.unsqueeze(0)
    v_impact = sim._v_com[2].item()
    z_min = sim.x_mpm[:, 2].min().item()
    print(f"  [IMPACT] Ground contact at step {step}! "
          f"v_impact={v_impact:.3f} z_min={z_min:.4f}")

    n_particles = sim.F.shape[0]
    sim.F = torch.eye(3, device=sim.F.device).unsqueeze(0).expand(n_particles, 3, 3).clone()
    sim.C = torch.zeros_like(sim.C)
    if hasattr(sim, 'c_grid'):
        delattr(sim, 'c_grid')
    if hasattr(sim, 'grid_occupied'):
        delattr(sim, 'grid_occupied')
    sim.frame_count = 0
    sim._hybrid_step = 0
    sim._impact_frame_count = 0

    z_vals = sim.x_mpm[:, 2]
    z_min_val = z_vals.min().item()
    z_max_val = z_vals.max().item()
    obj_height = z_max_val - z_min_val

    z_threshold = z_min_val + obj_height * 0.05
    bottom_mask = z_vals < z_threshold
    impact_center = sim.x_mpm[bottom_mask].mean(dim=0)

    sim._impact_center = impact_center
    radius_frac = float(sim.pf_params.get("impact_seed_radius_frac", 0.08))
    depth_frac = float(sim.pf_params.get("impact_seed_depth_frac", 0.06))
    sim._impact_radius = obj_height * radius_frac
    sim._impact_seed_radius_xy = max(obj_height * radius_frac, 1e-6)
    sim._impact_seed_depth = max(obj_height * depth_frac, 1e-6)
    sim._impact_bottom_z = z_min_val
    sim._impact_contact_normal = torch.tensor(
        [0.0, 0.0, 1.0], device=sim.x_mpm.device, dtype=sim.x_mpm.dtype
    )

    Gc = getattr(sim.elasticity, 'Gc', 200.0)
    l0 = getattr(sim.elasticity, 'l0', 0.035)
    H_ref = Gc / (2.0 * l0)

    xy_delta = sim.x_mpm[:, :2] - impact_center[:2].unsqueeze(0)
    dist_xy = xy_delta.norm(dim=1)
    dz_above = (sim.x_mpm[:, 2] - impact_center[2]).clamp(min=0.0)
    tight_radius = sim._impact_seed_radius_xy
    tight_depth = sim._impact_seed_depth
    h_seed_mult = float(sim.pf_params.get("impact_h_seed_multiplier", 4.5))
    H_tight = (
        torch.exp(-0.5 * (dist_xy / tight_radius) ** 2) *
        torch.exp(-0.5 * (dz_above / tight_depth) ** 2) *
        (h_seed_mult * H_ref)
    )
    H_tight[dz_above > (1.8 * tight_depth)] = 0.0
    nuc_frac = sim.pf_params.get('nucleation_fraction', 0.3)
    H_tight[H_tight < nuc_frac * H_ref] = 0.0

    if not hasattr(sim, '_history_H'):
        sim._history_H = H_tight
    else:
        sim._history_H = torch.maximum(sim._history_H, H_tight)

    print(f"  [IMPACT-H] Localized seed: {(H_tight > 0.0).sum().item()} particles")
    print(f"    center=[{impact_center[0]:.3f},{impact_center[1]:.3f},{impact_center[2]:.3f}] "
          f"r_xy={tight_radius:.4f} depth={tight_depth:.4f} H_ref={H_ref:.1f}")

    base_radial = float(sim.pf_params.get("impact_radial_seeds", 2))
    radial_cap = int(sim.pf_params.get("impact_radial_seed_cap", 2))
    mat_scale = get_material_shatter_scale(sim)
    n_radial = int(math.ceil(base_radial * (0.95 + 0.15 * mat_scale)))
    n_radial = max(1, min(radial_cap, n_radial))
    max_total_cracks = _get_effective_max_total_cracks(sim)
    n_radial = min(n_radial, max_total_cracks)
    upward_bias = float(sim.pf_params.get("impact_seed_upward_bias", 1.0))
    lateral_weight = float(sim.pf_params.get("impact_seed_lateral_weight", 0.35))
    principal_axis = torch.tensor(
        [1.0, 0.0], device=sim.x_mpm.device, dtype=sim.x_mpm.dtype
    )
    if bottom_mask.sum().item() >= 3:
        xy = sim.x_mpm[bottom_mask][:, :2]
        xy_centered = xy - xy.mean(dim=0, keepdim=True)
        cov = xy_centered.T @ xy_centered
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            principal_axis = eigvecs[:, torch.argmax(eigvals)]
        except RuntimeError:
            principal_axis = torch.tensor(
                [1.0, 0.0], device=sim.x_mpm.device, dtype=sim.x_mpm.dtype
            )
    principal_axis = principal_axis / (principal_axis.norm() + 1e-8)
    if not hasattr(sim, 'crack_paths'):
        sim.crack_paths = []
        sim.crack_dirs = []
        sim._branch_count = {}
    direction_signs = [0.0] if n_radial == 1 else [-1.0, 1.0]
    for sign in direction_signs[:n_radial]:
        init_dir = torch.tensor(
            [
                float(principal_axis[0].item()) * lateral_weight * sign,
                float(principal_axis[1].item()) * lateral_weight * sign,
                upward_bias,
            ],
            device=sim.x_mpm.device,
            dtype=sim.x_mpm.dtype,
        )
        init_dir = init_dir / (init_dir.norm() + 1e-8)
        sim.crack_paths.append(impact_center.unsqueeze(0).clone())
        sim.crack_dirs.append(init_dir)
    print(f"  [IMPACT-CRACK] Created {n_radial} crack seeds at impact "
          f"(principal-axis upward backbone)")

    post_g = -400.0
    g_vec = sim.mpm.gravity.clone()
    g_vec[:] = 0.0
    g_vec[2] = post_g
    sim.mpm.gravity = g_vec
    sim.mpm.damping = 0.975
    sim._post_impact_gravity_restored = False
    print(f"  [POST-IMPACT] gravity=[0,0,{post_g}] damping=0.975")
