"""Rendering-step helpers for HybridCrackSimulator."""

from __future__ import annotations

import math

import torch

from src.core.crack_evolution import get_material_shatter_scale


def _get_material_burst_iters(sim, frames_since: int) -> int:
    """Material-aware impact burst schedule.

    Lower Gc -> more burst iterations.
    Higher Gc -> fewer burst iterations.
    """
    if not sim.pf_params.get("impact_burst_enabled", True):
        return 0

    schedule = sim.pf_params.get("impact_burst_schedule", [12, 6, 3, 1])
    if frames_since < 0 or frames_since >= len(schedule):
        return 0

    base_iters = int(schedule[frames_since])
    mat_scale = get_material_shatter_scale(sim)
    burst_scale = float(sim.pf_params.get("impact_burst_material_scale", 0.9))
    eff_scale = 1.0 + (mat_scale - 1.0) * burst_scale
    min_iters = int(sim.pf_params.get("impact_burst_min_iters", 1))
    max_iters = int(sim.pf_params.get("impact_burst_max_iters", 20))
    burst_iters = int(math.ceil(base_iters * eff_scale))
    burst_iters = max(min_iters, min(max_iters, burst_iters))
    return burst_iters


def run_frame_dynamics(sim) -> bool:
    """Advance one frame of physics/crack dynamics and return refinement mode."""
    if sim.crack_only:
        for _ in range(sim.substeps):
            sim.step_crack_only(sim.mpm.dt)
        return False

    # Adaptive dt physics substeps
    dt_base = sim.mpm.dt
    cfl_target = 0.4
    dt_min = dt_base / 32

    for _ in range(sim.substeps):
        dt_current = dt_base
        last_cfl = getattr(sim, '_last_cfl', 0.0)
        if last_cfl > cfl_target and sim._gravity_drop_contacted:
            scale = cfl_target / (last_cfl + 1e-8)
            dt_current = max(dt_base * scale, dt_min)
            n_sub = max(1, int(math.ceil(dt_base / dt_current)))
            dt_current = dt_base / n_sub
            if n_sub > 1:
                if not hasattr(sim, '_adaptive_dt_logged') or sim._physics_step % 20 == 0:
                    print(f"  [ADAPTIVE-DT] CFL={last_cfl:.3f} -> {n_sub} sub-steps "
                          f"(dt={dt_current:.2e})", flush=True)
                    sim._adaptive_dt_logged = True
            orig_dt = sim.mpm.dt
            sim.mpm.dt = dt_current
            for _s in range(n_sub):
                sim.step_physics(dt_current)
            sim.mpm.dt = orig_dt
        else:
            sim.step_physics(dt_current)

    # Burst mode: crack iterations at impact
    if (sim._gravity_drop and sim._gravity_drop_contacted
            and hasattr(sim, '_impact_frame_count')):
        frames_since = sim._impact_frame_count
        burst_iters = _get_material_burst_iters(sim, frames_since)
        if burst_iters > 0:
            mat_scale = get_material_shatter_scale(sim)
            print(f"  [BURST] Impact frame+{frames_since}: "
                  f"running {burst_iters} crack iterations "
                  f"(mat_scale={mat_scale:.2f})", flush=True)
            for _ in range(burst_iters):
                sim.step_hybrid_crack(sim.mpm.dt)
    else:
        sim.step_hybrid_crack(sim.mpm.dt)

    apply_fragment_rigid_projection(sim)
    return False


def apply_fragment_rigid_projection(sim):
    """Suppress internal wobble while preserving fragment ballistic motion."""
    if not (sim.fragmentation_active and sim.fragment_manager.n_fragments > 1):
        return

    rigid_alpha = float(sim.pf_params.get("fragment_rigid_alpha", 0.5))
    small_alpha = float(sim.pf_params.get("fragment_small_rigid_alpha", 0.15))
    min_count = int(sim.fragment_manager.min_fragment_particles)
    omega_max = float(sim.pf_params.get("fragment_rigid_omega_max", 35.0))
    reg_eps = float(sim.pf_params.get("fragment_inertia_reg", 1.0e-6))
    slender_ratio_thresh = float(sim.pf_params.get("fragment_slender_ratio_threshold", 6.0))
    p_mass = sim.mpm.p_mass
    for frag_idx in sim.fragment_manager.fragment_particle_indices:
        n_frag = len(frag_idx)
        if n_frag < 8:
            continue
        x_frag = sim.x_mpm[frag_idx]
        v_frag = sim.v_mpm[frag_idx]
        com = x_frag.mean(dim=0)
        v_com = v_frag.mean(dim=0)
        r = x_frag - com.unsqueeze(0)

        cov = (r.transpose(0, 1) @ r) / max(n_frag, 1)
        try:
            eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
            slender_ratio = float((eigvals[-1] / eigvals[0]).item())
        except Exception:
            slender_ratio = 1.0

        alpha = rigid_alpha
        if n_frag < min_count or slender_ratio > slender_ratio_thresh:
            alpha = small_alpha
        if alpha <= 0.0:
            continue

        v_rel = v_frag - v_com.unsqueeze(0)
        L = torch.cross(r, v_rel, dim=1).sum(dim=0) * p_mass
        r_sq = (r * r).sum(dim=1)
        I_tensor = torch.zeros(3, 3, device=r.device)
        I_tensor[0, 0] = (r_sq - r[:, 0] ** 2).sum()
        I_tensor[1, 1] = (r_sq - r[:, 1] ** 2).sum()
        I_tensor[2, 2] = (r_sq - r[:, 2] ** 2).sum()
        I_tensor[0, 1] = I_tensor[1, 0] = -(r[:, 0] * r[:, 1]).sum()
        I_tensor[0, 2] = I_tensor[2, 0] = -(r[:, 0] * r[:, 2]).sum()
        I_tensor[1, 2] = I_tensor[2, 1] = -(r[:, 1] * r[:, 2]).sum()
        I_tensor *= p_mass
        I_tensor = I_tensor + reg_eps * torch.eye(3, device=r.device, dtype=r.dtype)
        try:
            omega = torch.linalg.solve(I_tensor, L)
        except Exception:
            omega = torch.zeros(3, device=r.device)
        omega_norm = float(omega.norm().item())
        if omega_norm > omega_max:
            omega = omega * (omega_max / max(omega_norm, 1e-8))
        omega_exp = omega.unsqueeze(0).expand_as(r)
        v_rigid = v_com.unsqueeze(0) + torch.cross(omega_exp, r, dim=1)
        sim.v_mpm[frag_idx] = alpha * v_rigid + (1.0 - alpha) * v_frag


def post_frame_maintenance(sim, impact_refine: bool):
    """Run post-dynamics maintenance such as gravity restore and fragment checks."""
    empty_cache_interval = int(sim.pf_params.get("empty_cache_interval", 0))
    if empty_cache_interval > 0 and torch.cuda.is_available():
        frame_idx = int(getattr(sim, "frame_count", 0))
        if frame_idx % empty_cache_interval == 0:
            torch.cuda.empty_cache()

    if (sim._gravity_drop and sim._gravity_drop_contacted
            and hasattr(sim, '_impact_frame_count')
            and not getattr(sim, '_post_impact_gravity_restored', True)):
        if sim._impact_frame_count >= 10:
            sim._post_impact_gravity_restored = True
            print("  [GRAVITY] Maintaining gravity (adaptive dt handles CFL)")

    frag_detect_frames = (1, 2, 3, 5, 10, 20, 40)
    if (hasattr(sim, '_impact_frame_count')
                and sim._impact_frame_count in frag_detect_frames):
            sim._maybe_detect_fragments()


def update_surface_gaussians(sim):
    """Project damage to surface and update Gaussian state."""
    x_surf_mpm = sim.x_mpm[sim.surface_mask]
    ref_center = sim.x_mpm.mean(dim=0)
    if (getattr(sim, '_gravity_drop', False)
            and getattr(sim, '_gravity_drop_contacted', False)
            and hasattr(sim, '_impact_center')):
        impact_proj_frames = int(sim.pf_params.get("impact_projection_frames", 12))
        impact_frame = int(getattr(sim, '_impact_frame_count', 0))
        if impact_frame < impact_proj_frames:
            blend = max(0.0, 1.0 - impact_frame / max(impact_proj_frames, 1))
            ref_center = (
                blend * sim._impact_center.to(device=x_surf_mpm.device, dtype=x_surf_mpm.dtype)
                + (1.0 - blend) * ref_center
            )
    inward_dirs = ref_center.unsqueeze(0) - x_surf_mpm
    c_surf = sim.damage_mapper.project_damage(
        sim.c_vol, sim.x_mpm, x_surf_mpm, sim.surface_mask,
        inward_dirs=inward_dirs)
    x_surf_world = sim.mapper.mpm_to_world(x_surf_mpm)

    F_surf = sim.F[sim.surface_mask]

    if sim._ply_direct:
        disp = x_surf_world - sim._surf_init_world
        ply_disp = disp[sim._ply_to_surface]
        x_ply_world = sim._ply_init_xyz + ply_disp
        c_ply = c_surf[sim._ply_to_surface]
        F_ply = F_surf[sim._ply_to_surface]
    else:
        x_ply_world = x_surf_world
        c_ply = c_surf
        F_ply = F_surf

    debris_mask = _build_debris_mask(sim, x_surf_world)

    sim.visualizer.update_gaussians(
        sim.gaussians, c_ply, x_ply_world,
        preserve_original=True,
        debris_mask=debris_mask,
        F_per_gaussian=F_ply,
        camera_pos=getattr(sim, '_camera_pos', None))


def _build_debris_mask(sim, x_surf_world):
    """Build debris mask so small fragments can be hidden or stylized."""
    if not (sim.fragmentation_active
            and sim.fragment_manager is not None
            and sim.fragment_manager.n_fragments > 1):
        return None

    frag_ids = sim.fragment_manager.fragment_ids
    surf_frag_ids = frag_ids[sim.surface_mask]
    if sim._ply_direct:
        surf_debris = torch.zeros(x_surf_world.shape[0],
                                  dtype=torch.bool, device=x_surf_world.device)
        for frag_idx in sim.fragment_manager.fragment_particle_indices:
            if (len(frag_idx) == 0
                    or len(frag_idx) >= sim.fragment_manager.min_fragment_particles):
                continue
            frag_label = frag_ids[frag_idx[0]].item()
            surf_debris |= (surf_frag_ids == frag_label)
        return surf_debris[sim._ply_to_surface]

    debris_mask = torch.zeros(x_surf_world.shape[0],
                              dtype=torch.bool, device=x_surf_world.device)
    for frag_idx in sim.fragment_manager.fragment_particle_indices:
        if (len(frag_idx) == 0
                or len(frag_idx) >= sim.fragment_manager.min_fragment_particles):
            continue
        frag_label = frag_ids[frag_idx[0]].item()
        debris_mask |= (surf_frag_ids == frag_label)
    return debris_mask
