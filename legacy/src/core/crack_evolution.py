"""Crack evolution helpers extracted from HybridCrackSimulator.

These helpers operate on the simulator instance directly to preserve
behavior while reducing the size of the main orchestration class.
"""

from __future__ import annotations

import math

import torch


def _get_effective_max_total_cracks(sim) -> int:
    """Material-aware crack count cap with a stricter early-impact budget."""
    mat_scale = get_material_shatter_scale(sim)
    base_max = int(sim.pf_params.get('max_total_cracks', 5))
    crack_count_scale = float(sim.pf_params.get("material_crack_count_scale", 0.7))
    eff_max = max(
        2,
        int(math.ceil(base_max * (1.0 + (mat_scale - 1.0) * crack_count_scale)))
    )

    if (getattr(sim, '_gravity_drop', False)
            and getattr(sim, '_gravity_drop_contacted', False)):
        frame_idx = int(getattr(sim, '_impact_frame_count', 0))
        cap_frames = int(sim.pf_params.get("impact_path_cap_frames", 3))
        if frame_idx < cap_frames:
            early_base = int(sim.pf_params.get("impact_early_max_cracks", 18))
            early_scale = float(sim.pf_params.get("impact_early_crack_scale", 0.35))
            early_cap = max(
                4,
                int(math.ceil(early_base * (1.0 + (mat_scale - 1.0) * early_scale)))
            )
            eff_max = min(eff_max, early_cap)

    return eff_max


def _ensure_crack_runtime_state(sim):
    """Keep auxiliary per-path state aligned with crack_paths."""
    n_paths = len(getattr(sim, 'crack_paths', []))

    if not hasattr(sim, 'crack_active'):
        sim.crack_active = [True] * n_paths
    if not hasattr(sim, 'crack_fail_count'):
        sim.crack_fail_count = [0] * n_paths
    if not hasattr(sim, 'crack_seed_pos'):
        sim.crack_seed_pos = [
            path[0].detach().clone() if path.shape[0] > 0 else None
            for path in sim.crack_paths
        ]

    if len(sim.crack_active) < n_paths:
        sim.crack_active.extend([True] * (n_paths - len(sim.crack_active)))
    else:
        sim.crack_active = sim.crack_active[:n_paths]

    if len(sim.crack_fail_count) < n_paths:
        sim.crack_fail_count.extend([0] * (n_paths - len(sim.crack_fail_count)))
    else:
        sim.crack_fail_count = sim.crack_fail_count[:n_paths]

    if len(sim.crack_seed_pos) < n_paths:
        sim.crack_seed_pos.extend(
            path[0].detach().clone() if path.shape[0] > 0 else None
            for path in sim.crack_paths[len(sim.crack_seed_pos):]
        )
    else:
        sim.crack_seed_pos = sim.crack_seed_pos[:n_paths]


def _append_crack_path(sim, path: torch.Tensor, init_dir):
    """Append a new crack path and keep auxiliary runtime state in sync."""
    if not hasattr(sim, 'crack_paths'):
        sim.crack_paths = []
        sim.crack_dirs = []
    sim.crack_paths.append(path)
    sim.crack_dirs.append(init_dir)
    if hasattr(sim, 'crack_active'):
        sim.crack_active.append(True)
    if hasattr(sim, 'crack_fail_count'):
        sim.crack_fail_count.append(0)
    if hasattr(sim, 'crack_seed_pos'):
        seed = path[0].detach().clone() if path.shape[0] > 0 else None
        sim.crack_seed_pos.append(seed)


def _tip_within_front_envelope(sim, point: torch.Tensor) -> bool:
    """Limit early impact propagation to a growing front envelope."""
    if not (getattr(sim, '_gravity_drop', False)
            and getattr(sim, '_gravity_drop_contacted', False)
            and hasattr(sim, '_impact_center')):
        return True

    frame_idx = int(getattr(sim, '_impact_frame_count', 0))
    envelope_frames = int(sim.pf_params.get("impact_front_envelope_frames", 3))
    if frame_idx >= envelope_frames:
        return True

    center = sim._impact_center.to(device=point.device, dtype=point.dtype)
    radius_xy = float(getattr(sim, '_impact_seed_radius_xy', getattr(sim, '_impact_radius', 0.0)))
    depth = float(getattr(sim, '_impact_seed_depth', getattr(sim, '_impact_radius', 0.0)))
    if radius_xy <= 0.0 or depth <= 0.0:
        return True

    mat_scale = get_material_shatter_scale(sim)
    scale = max(0.75, min(1.35, mat_scale))
    r_base = float(sim.pf_params.get("impact_front_radius_base_scale", 1.8))
    r_growth = float(sim.pf_params.get("impact_front_radius_growth_scale", 1.0))
    z_base = float(sim.pf_params.get("impact_front_depth_base_scale", 3.0))
    z_growth = float(sim.pf_params.get("impact_front_depth_growth_scale", 2.5))
    z_below = float(sim.pf_params.get("impact_front_below_scale", 0.6))

    rel = point - center
    dist_xy = float(rel[:2].norm().item())
    dz = float(rel[2].item())

    max_r = radius_xy * scale * (r_base + r_growth * frame_idx)
    max_z = depth * scale * (z_base + z_growth * frame_idx)
    min_z = -depth * z_below
    return (dist_xy <= max_r) and (min_z <= dz <= max_z)


def _tip_candidate_valid(sim, point: torch.Tensor, H_ref: float) -> bool:
    """Check if a crack tip candidate is physically admissible."""
    if not _tip_within_front_envelope(sim, point):
        return False

    n = sim.mpm.num_grids
    gi = (point * n).long().clamp(0, n - 1)
    if not sim.grid_occupied[gi[0], gi[1], gi[2]]:
        return False

    h_gate_frac = float(sim.pf_params.get("crack_tip_h_fraction", 0.03))
    c_gate_min = float(sim.pf_params.get("crack_tip_c_min", 0.02))
    H_grid = getattr(sim, '_H_grid_physics', sim.H_grid)
    H_local = float(H_grid[gi[0], gi[1], gi[2]].item())
    c_local = float(sim.c_grid[gi[0], gi[1], gi[2]].item()) if hasattr(sim, 'c_grid') else 0.0
    return (H_local >= h_gate_frac * H_ref) or (c_local >= c_gate_min)


def _front_band_score(sim, point: torch.Tensor, direction: torch.Tensor,
                      smooth_dir: torch.Tensor, H_ref: float) -> float:
    """Score a candidate tip inside an AT2/H-admissible narrow front band."""
    if not _tip_within_front_envelope(sim, point):
        return -1e9

    n = sim.mpm.num_grids
    gi = (point * n).long().clamp(0, n - 1)
    if not sim.grid_occupied[gi[0], gi[1], gi[2]]:
        return -1e9

    H_grid = getattr(sim, '_H_grid_physics', sim.H_grid)
    H_local = float(H_grid[gi[0], gi[1], gi[2]].item())
    c_local = float(sim.c_grid[gi[0], gi[1], gi[2]].item()) if hasattr(sim, 'c_grid') else 0.0

    h_gate_frac = float(sim.pf_params.get("crack_tip_h_fraction", 0.03))
    c_gate_min = float(sim.pf_params.get("crack_tip_c_min", 0.02))
    if H_local < h_gate_frac * H_ref and c_local < c_gate_min:
        return -1e9

    h_cap = float(sim.pf_params.get("front_h_score_cap", 3.0))
    h_score = min(H_local / max(H_ref, 1e-12), h_cap) / max(h_cap, 1e-12)

    c_target = float(sim.pf_params.get("front_c_target", 0.16))
    c_width = float(sim.pf_params.get("front_c_bandwidth", 0.12))
    c_band_score = max(0.0, 1.0 - abs(c_local - c_target) / max(c_width, 1e-6))

    if c_local < c_gate_min:
        c_band_score *= 0.35

    align = float((direction * smooth_dir).sum().item())
    align_score = 0.5 * (align + 1.0)

    w_h = float(sim.pf_params.get("front_h_weight", 0.45))
    w_c = float(sim.pf_params.get("front_c_weight", 0.40))
    w_align = float(sim.pf_params.get("front_align_weight", 0.15))
    return w_h * h_score + w_c * c_band_score + w_align * align_score


def _build_impact_zone_mask(sim, n: int, device, occ=None, radius_scale: float = 1.0,
                            depth_scale: float = 1.0):
    """Build a floor-contact-localized mask near the impact patch."""
    if not hasattr(sim, '_impact_center'):
        return None

    radius_xy = float(getattr(sim, '_impact_seed_radius_xy', getattr(sim, '_impact_radius', 0.0)))
    depth = float(getattr(sim, '_impact_seed_depth', getattr(sim, '_impact_radius', 0.0)))
    if radius_xy <= 0.0 or depth <= 0.0:
        return None

    coords_1d = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    gx, gy, gz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    center = sim._impact_center.to(device=device, dtype=torch.float32)
    dist_xy = torch.sqrt((gx - center[0]) ** 2 + (gy - center[1]) ** 2)
    dz_above = (gz - center[2]).clamp(min=0.0)

    impact_zone = (
        (dist_xy <= radius_xy * radius_scale) &
        (dz_above <= depth * depth_scale)
    )
    if occ is not None:
        impact_zone = impact_zone & occ
    return impact_zone


def _build_front_scaffold_grids(sim, n: int, device):
    """Build narrow front-core and front-tube grids from active crack path tails."""
    if not hasattr(sim, 'crack_paths') or len(sim.crack_paths) == 0:
        return None, None

    tail_points = max(1, int(sim.pf_params.get("front_core_tail_points", 3)))
    front_core = torch.zeros(n, n, n, device=device, dtype=torch.float32)

    for path_idx, path in enumerate(sim.crack_paths):
        if hasattr(sim, 'crack_active') and not sim.crack_active[path_idx]:
            continue
        if path.shape[0] == 0:
            continue
        tail = path[-tail_points:] if path.shape[0] > tail_points else path
        gi = (tail * n).long().clamp(0, n - 1)
        front_core[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0

    if front_core.max().item() <= 0.0:
        return front_core, front_core.clone()

    radius_cells = max(0, int(sim.pf_params.get("front_tube_radius_cells", 1)))
    if radius_cells == 0:
        front_tube = front_core.clone()
    else:
        k = 2 * radius_cells + 1
        front_tube = torch.nn.functional.max_pool3d(
            front_core.unsqueeze(0).unsqueeze(0),
            kernel_size=k,
            stride=1,
            padding=radius_cells,
        )[0, 0]

    occ = getattr(sim, 'grid_occupied', None)
    if occ is not None:
        front_core = front_core * occ.float()
        front_tube = front_tube * occ.float()
    return front_core, front_tube


def get_material_shatter_scale(sim) -> float:
    """Bounded brittle-response scale from fracture toughness.

    Lower Gc -> larger scale (more brittle release).
    Higher Gc -> smaller scale (more resistant to violent shatter).
    """
    Gc = max(float(getattr(sim.elasticity, 'Gc', 30.0)), 1e-8)
    gc_ref = float(sim.pf_params.get("impact_shatter_gc_ref", 30.0))
    alpha = float(sim.pf_params.get("impact_shatter_gc_alpha", 0.35))
    scale_min = float(sim.pf_params.get("impact_shatter_scale_min", 0.6))
    scale_max = float(sim.pf_params.get("impact_shatter_scale_max", 1.6))
    scale = (gc_ref / Gc) ** alpha
    return max(scale_min, min(scale_max, scale))


def _get_impact_acceleration(sim):
    """Return impact-window crack acceleration gains.

    This is the default impact behavior, not an optional overlay.
    """
    gains = {
        "nucleation": 1.0,
        "tip_speed": 1.0,
        "at2": 1.0,
    }

    if not (getattr(sim, '_gravity_drop', False) and getattr(sim, '_gravity_drop_contacted', False)):
        return gains
    if not hasattr(sim, '_impact_frame_count'):
        return gains

    frames = int(sim.pf_params.get("impact_refine_frames", 4))
    frame_idx = int(getattr(sim, '_impact_frame_count', 0))
    if frame_idx >= frames:
        return gains

    decay = float(sim.pf_params.get("impact_accel_decay", 0.72))
    ramp = decay ** frame_idx
    gains["nucleation"] = 1.0 + (float(sim.pf_params.get("impact_nucleation_gain", 2.4)) - 1.0) * ramp
    gains["tip_speed"] = 1.0 + (float(sim.pf_params.get("impact_tip_speed_gain", 1.9)) - 1.0) * ramp
    gains["at2"] = 1.0 + (float(sim.pf_params.get("impact_at2_gain", 1.6)) - 1.0) * ramp
    return gains


def _spawn_impact_shatter_seeds(sim, device):
    """Spawn a few extra shallow seeds near contact for a punchier initial shatter."""
    if not (getattr(sim, '_gravity_drop', False) and getattr(sim, '_gravity_drop_contacted', False)):
        return 0
    if not hasattr(sim, '_impact_center'):
        return 0

    frame_idx = int(getattr(sim, '_impact_frame_count', 0))
    shatter_frames = int(sim.pf_params.get("impact_shatter_frames", 2))
    if frame_idx >= shatter_frames:
        return 0

    last_seed_frame = getattr(sim, '_impact_shatter_seed_frame', -1)
    if last_seed_frame == frame_idx:
        return 0

    max_total_cracks = _get_effective_max_total_cracks(sim)
    remaining = max_total_cracks - len(sim.crack_paths)
    if remaining <= 0:
        return 0

    mat_scale = get_material_shatter_scale(sim)
    base_extra = float(sim.pf_params.get("impact_shatter_extra_seeds", 6))
    n_extra = min(int(math.ceil(base_extra * mat_scale)), remaining)
    if n_extra <= 0:
        return 0

    center = sim._impact_center.to(device=device, dtype=sim.x_mpm.dtype)
    radius = float(getattr(sim, '_impact_seed_radius_xy', getattr(sim, '_impact_radius', 0.02)))
    depth = float(getattr(sim, '_impact_seed_depth', radius))
    ring_radius = max(radius * float(sim.pf_params.get("impact_shatter_radius_scale", 0.65)), 1e-5)
    z_offset = max(depth * float(sim.pf_params.get("impact_shatter_depth_scale", 0.35)), 0.0)
    upward_bias = float(sim.pf_params.get("impact_shatter_upward_bias", 0.25))
    angle_jitter = float(sim.pf_params.get("impact_shatter_angle_jitter", 0.18))

    spawned = 0
    for idx in range(n_extra):
        angle = (2.0 * math.pi * idx) / max(n_extra, 1)
        angle += (torch.rand(1, device=device).item() - 0.5) * 2.0 * angle_jitter
        radial = torch.tensor(
            [math.cos(angle), math.sin(angle), 0.0],
            device=device,
            dtype=center.dtype,
        )
        seed_pos = center + radial * ring_radius
        seed_pos[2] = min(center[2] + z_offset, 1.0 - 1e-4)
        seed_pos = seed_pos.clamp(1e-4, 1.0 - 1e-4)

        init_dir = radial.clone()
        init_dir[2] = upward_bias
        init_dir = init_dir / (init_dir.norm() + 1e-8)
        _append_crack_path(sim, seed_pos.unsqueeze(0), init_dir)
        spawned += 1

    sim._impact_shatter_seed_frame = frame_idx
    return spawned


@torch.no_grad()
def step_hybrid_crack(sim, dt: float):
    """Hybrid crack: nucleation + tip advance + AT2 PDE + geometric damage."""
    if not hasattr(sim, 'c_grid'):
        sim._init_grid_infrastructure()
    if not hasattr(sim, '_hybrid_step'):
        sim._hybrid_step = 0
    if not hasattr(sim, '_crack_dt_ref'):
        sim._crack_dt_ref = max(float(dt), 1e-8)
    if not hasattr(sim, '_nucleation_budget'):
        sim._nucleation_budget = 0.0

    n = sim.mpm.num_grids
    dx = sim.mpm.dx
    device = sim.x_mpm.device
    step = sim._hybrid_step
    dt_ratio = max(float(dt) / max(sim._crack_dt_ref, 1e-8), 1e-4)

    Gc = getattr(sim.elasticity, 'Gc', 30.0)
    l0 = getattr(sim.elasticity, 'l0', 0.025)
    H_ref = Gc / (2.0 * l0)
    max_nuc = sim.pf_params.get('max_nucleation_per_frame', 1)
    nucleation_frac = sim.pf_params.get('nucleation_fraction', 0.3)
    min_spacing = sim.pf_params.get('nucleation_min_spacing', 8)
    crack_tip_speed = sim.pf_params.get('crack_tip_speed', 1.5)
    max_total_cracks = _get_effective_max_total_cracks(sim)
    impact_accel = _get_impact_acceleration(sim)
    max_nuc_eff = max_nuc * impact_accel["nucleation"]
    crack_tip_speed_eff = crack_tip_speed * impact_accel["tip_speed"]

    if not hasattr(sim, 'crack_paths'):
        sim.crack_paths = []
        sim.crack_dirs = []
    _ensure_crack_runtime_state(sim)

    shatter_new = _spawn_impact_shatter_seeds(sim, device)

    if hasattr(sim, '_history_H'):
        sim.H_grid = sim._bin_particles_to_grid(sim._history_H)
    else:
        sim.H_grid = torch.zeros(n, n, n, device=device)

    sim._H_grid_physics = sim.H_grid.clone()

    H_crack_mult = float(sim.pf_params.get("crack_history_h_multiplier", 0.0))
    H_crack = H_ref * H_crack_mult
    if sim.crack_paths and H_crack_mult > 0.0:
        history_h_points = max(1, int(sim.pf_params.get("crack_history_h_points", 1)))
        for path_idx, path in enumerate(sim.crack_paths):
            if hasattr(sim, 'crack_active') and not sim.crack_active[path_idx]:
                continue
            tail = path[-history_h_points:] if path.shape[0] > history_h_points else path
            gi = (tail * n).long().clamp(0, n - 1)
            current_H = sim.H_grid[gi[:, 0], gi[:, 1], gi[:, 2]]
            sim.H_grid[gi[:, 0], gi[:, 1], gi[:, 2]] = torch.maximum(
                current_H, torch.full_like(current_H, H_crack))

    if hasattr(sim, '_last_stress') and sim._last_stress is not None:
        sim._compute_aniso_diffusion()

    n_new = 0
    if len(sim.crack_paths) < max_total_cracks:
        crack_mask = torch.zeros(n, n, n, device=device, dtype=torch.bool)
        for path in sim.crack_paths:
            for pt in path:
                gi = (pt * n).long().clamp(0, n - 1)
                lo = (gi - min_spacing).clamp(min=0)
                hi = (gi + min_spacing + 1).clamp(max=n)
                crack_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True

        occ = sim.grid_occupied
        interior = (occ[1:-1, 1:-1, 1:-1] &
                    occ[2:, 1:-1, 1:-1] & occ[:-2, 1:-1, 1:-1] &
                    occ[1:-1, 2:, 1:-1] & occ[1:-1, :-2, 1:-1] &
                    occ[1:-1, 1:-1, 2:] & occ[1:-1, 1:-1, :-2])
        interior_full = torch.zeros_like(occ)
        interior_full[1:-1, 1:-1, 1:-1] = interior

        impact_zone = None
        if (getattr(sim, '_gravity_drop', False) and
                getattr(sim, '_gravity_drop_contacted', False)):
            impact_zone = _build_impact_zone_mask(
                sim, n, device, occ=occ, radius_scale=1.35, depth_scale=2.25
            )
            if impact_zone is not None:
                interior_full = interior_full | impact_zone

        H_nuc_grid = getattr(sim, '_H_grid_physics', sim.H_grid)
        candidate_mask = ((H_nuc_grid > nucleation_frac * H_ref) &
                          interior_full & ~crack_mask)
        impact_only_frames = int(sim.pf_params.get("impact_nucleation_only_frames", 2))
        if (impact_zone is not None and
                int(getattr(sim, '_impact_frame_count', 0)) < impact_only_frames):
            localized_mask = candidate_mask & impact_zone
            if localized_mask.any():
                candidate_mask = localized_mask
        n_candidates = candidate_mask.sum().item()

        if n_candidates > 0 and max_nuc_eff > 0:
            H_score = H_nuc_grid.clone()
            H_score[~candidate_mask] = 0.0
            if impact_zone is not None:
                impact_priority = float(sim.pf_params.get("impact_nucleation_priority", 1.5))
                H_score[impact_zone & candidate_mask] *= impact_priority
            sim._nucleation_budget += max_nuc_eff * dt_ratio
            k_new = min(int(sim._nucleation_budget), n_candidates)
            if k_new <= 0 and sim._nucleation_budget > 0.0:
                trigger_prob = min(sim._nucleation_budget, 1.0)
                if torch.rand(1, device=device).item() < trigger_prob:
                    k_new = min(1, n_candidates)

            remaining_budget = max_total_cracks - len(sim.crack_paths)
            if remaining_budget <= 0:
                k_new = 0
            else:
                k_new = min(k_new, remaining_budget)

            if (getattr(sim, '_gravity_drop', False)
                    and getattr(sim, '_gravity_drop_contacted', False)):
                frame_idx = int(getattr(sim, '_impact_frame_count', 0))
                nuc_cap_frames = int(sim.pf_params.get("impact_nucleation_cap_frames", 3))
                if frame_idx < nuc_cap_frames:
                    early_nuc_cap = int(sim.pf_params.get("impact_max_new_paths_per_step", 2))
                    k_new = min(k_new, max(1, early_nuc_cap))

            if k_new > 0:
                _, topk_flat = H_score.view(-1).topk(k_new)
                sim._nucleation_budget = max(sim._nucleation_budget - k_new, 0.0)
            else:
                topk_flat = []

            for flat_idx in topk_flat:
                fi = flat_idx.item()
                i0 = fi // (n * n)
                i1 = (fi % (n * n)) // n
                i2 = fi % n
                pos = torch.tensor(
                    [(i0 + 0.5) / n, (i1 + 0.5) / n, (i2 + 0.5) / n],
                    device=device, dtype=torch.float32)
                _append_crack_path(sim, pos.unsqueeze(0), None)
                if (getattr(sim, '_gravity_drop', False) and
                        getattr(sim, '_gravity_drop_contacted', False) and
                        hasattr(sim, '_impact_center')):
                    delta_xy = pos[:2] - sim._impact_center[:2].to(device=device, dtype=pos.dtype)
                    if delta_xy.norm() > 1e-6:
                        init_dir = torch.tensor(
                            [delta_xy[0].item(), delta_xy[1].item(),
                             float(sim.pf_params.get("impact_seed_upward_bias", 0.20))],
                            device=device,
                            dtype=pos.dtype,
                        )
                    else:
                        angle = 2.0 * math.pi * torch.rand(1, device=device).item()
                        init_dir = torch.tensor(
                            [math.cos(angle), math.sin(angle),
                             float(sim.pf_params.get("impact_seed_upward_bias", 0.20))],
                            device=device,
                            dtype=pos.dtype,
                        )
                    init_dir = init_dir / (init_dir.norm() + 1e-8)
                    sim.crack_dirs[-1] = init_dir
                else:
                    sim.crack_dirs[-1] = None
                n_new += 1
                if step < 50:
                    print(f"  [NUC] New crack at grid=({i0},{i1},{i2}) "
                          f"pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
                          f"H={sim.H_grid[i0,i1,i2]:.1f}", flush=True)

    advance_crack_tips(sim, dx, n, device, crack_tip_speed_eff, H_ref, step, dt_ratio)

    front_core_grid, front_tube_grid = _build_front_scaffold_grids(sim, n, device)
    sim._front_core_grid = front_core_grid
    sim._front_tube_grid = front_tube_grid

    if (front_tube_grid is not None
            and getattr(sim, '_gravity_drop', False)
            and getattr(sim, '_gravity_drop_contacted', False)):
        frame_idx = int(getattr(sim, '_impact_frame_count', 0))
        focus_frames = int(sim.pf_params.get("front_focus_frames", 6))
        if frame_idx < focus_frames:
            outside_decay = float(sim.pf_params.get("impact_outside_front_h_decay", 0.35))
            tube_mask = front_tube_grid > 0.5
            sim.H_grid = torch.where(tube_mask, sim.H_grid, sim.H_grid * outside_decay)

    c_old = sim.c_vol.clone()

    if step == 0 and sim.c_vol.max() > 0:
        c_from_vol = sim._bin_particles_to_grid(sim.c_vol)
        sim.c_grid = torch.maximum(sim.c_grid, c_from_vol)
    if getattr(sim, '_at2_iters_override', None) is not None:
        base_at2_iters = sim._at2_iters_override
    else:
        base_at2_iters = 50 if step < 3 else 30
    H_phys_grid = getattr(sim, '_H_grid_physics', sim.H_grid)
    H_phys_max = float(H_phys_grid.max().item()) if H_phys_grid.numel() > 0 else 0.0
    h_ratio = H_phys_max / max(H_ref, 1e-12)
    impact_h_iter_scale = float(sim.pf_params.get("impact_h_iter_scale", 0.6))
    impact_h_iter_cap = float(sim.pf_params.get("impact_h_iter_cap", 2.5))
    h_iter_mult = 1.0 + impact_h_iter_scale * max(0.0, min(h_ratio - 1.0, impact_h_iter_cap))
    base_at2_iters = max(1, int(math.ceil(base_at2_iters * impact_accel["at2"] * h_iter_mult)))
    at2_iters = max(1, int(math.ceil(base_at2_iters * dt_ratio)))
    solve_at2_phase_field(sim, n_iters=at2_iters, crack_paths=None)
    sim._gather_grid_to_particles()

    assign_crack_damage(sim, crack_width=sim.pf_params.get('crack_width', 0.004))
    sim.c_vol = torch.maximum(sim.c_vol, c_old)

    if step < 5 or step % 20 == 0:
        n_paths = len(sim.crack_paths)
        total_pts = sum(p.shape[0] for p in sim.crack_paths)
        max_len = max((p.shape[0] for p in sim.crack_paths), default=0)
        n_cracked = (sim.c_vol > 0.3).sum().item()
        dc = sim.c_vol - c_old
        print(f"  [AT2 {step:3d}] paths={n_paths} pts={total_pts} "
              f"max_seg={max_len} c_vol={sim.c_vol.max():.4f} "
              f"c_grid={sim.c_grid.max():.4f}({(sim.c_grid > 0.3).sum().item()}cells) "
              f"cracked(>0.3)={n_cracked} dc_max={dc.max():.6f} "
              f"H_max={sim.H_grid.max():.2e} H_ref={H_ref:.2e} "
              f"nuc_new={n_new} shatter_new={shatter_new}", flush=True)

    sim._hybrid_step += 1


@torch.no_grad()
def advance_crack_tips(sim, dx, n, device, crack_tip_speed, H_ref, step, dt_ratio):
    """Advance crack tips as a constrained narrow-front tracker.

    Front motion is allowed only inside an AT2/H-admissible band and is
    limited to local one-cell-scale moves so the tracker follows the
    volumetric front instead of drawing global polylines.
    """
    _ensure_crack_runtime_state(sim)
    ema_alpha = 0.20
    min_step_dist = max(0.05 * dx, 0.20 * dx * dt_ratio)

    branch_angle = sim.pf_params.get('branch_angle', 35.0)
    branch_min_len = sim.pf_params.get('branch_min_length', 6)
    branch_prob = sim.pf_params.get('branch_probability', 0.3)
    mat_scale = get_material_shatter_scale(sim)
    branch_scale = float(sim.pf_params.get("material_branch_scale", 0.6))
    branch_prob *= max(0.35, min(2.2, 1.0 + (mat_scale - 1.0) * branch_scale))
    branch_prob_step = min(max(branch_prob * dt_ratio, 0.0), 1.0)
    max_branches = sim.pf_params.get('max_branches_per_path', 1)
    max_total_cracks = _get_effective_max_total_cracks(sim)
    if not hasattr(sim, '_branch_count'):
        sim._branch_count = {}
    max_failures = int(sim.pf_params.get("crack_path_max_failures", 4))
    min_h_continue = float(sim.pf_params.get("crack_path_min_h_fraction", 0.02))
    if (getattr(sim, '_gravity_drop', False)
            and getattr(sim, '_gravity_drop_contacted', False)):
        frame_idx = int(getattr(sim, '_impact_frame_count', 0))
        freeze_frames = int(sim.pf_params.get("impact_branch_freeze_frames", 6))
        if frame_idx < freeze_frames:
            branch_prob_step = 0.0
    if getattr(sim, 'fragmentation_active', False):
        branch_prob_step *= float(sim.pf_params.get("post_fragment_branch_scale", 0.10))

    pending_branches = []

    for path_idx in range(len(sim.crack_paths)):
        if not sim.crack_active[path_idx]:
            continue
        path = sim.crack_paths[path_idx]
        tip = path[-1]

        gi = (tip * n).long().clamp(1, n - 2)
        i, j, k = gi[0].item(), gi[1].item(), gi[2].item()
        H_phys = sim._H_grid_physics
        H_local = H_phys[i, j, k].item()
        grad_H = torch.zeros(3, device=device)
        grad_H[0] = (H_phys[min(i + 1, n - 1), j, k] - H_phys[max(i - 1, 0), j, k]) / (2 * dx)
        grad_H[1] = (H_phys[i, min(j + 1, n - 1), k] - H_phys[i, max(j - 1, 0), k]) / (2 * dx)
        grad_H[2] = (H_phys[i, j, min(k + 1, n - 1)] - H_phys[i, j, max(k - 1, 0)]) / (2 * dx)

        raw_dir = -grad_H
        if hasattr(sim, '_n1_grid') and sim._n1_grid is not None:
            n1 = sim._n1_grid[i, j, k]
            n1_mag = n1.norm()
            if n1_mag > 1e-6:
                n1 = n1 / n1_mag
                raw_dir = grad_H - (grad_H * n1).sum() * n1

        raw_mag = raw_dir.norm()
        if raw_mag > 1e-8:
            raw_dir = raw_dir / raw_mag
        else:
            if sim.crack_dirs[path_idx] is not None:
                raw_dir = sim.crack_dirs[path_idx]
            elif path.shape[0] >= 2:
                raw_dir = path[-1] - path[-2]
                if raw_dir.norm() < 1e-8:
                    continue
                raw_dir = raw_dir / raw_dir.norm()
            else:
                continue

        if sim.crack_dirs[path_idx] is None:
            smooth_dir = raw_dir
        else:
            smooth_dir = (1.0 - ema_alpha) * sim.crack_dirs[path_idx] + ema_alpha * raw_dir
            sm = smooth_dir.norm()
            smooth_dir = raw_dir if sm < 1e-8 else smooth_dir / sm
        sim.crack_dirs[path_idx] = smooth_dir

        speed_scale = H_local / (H_ref + 1e-12)
        if speed_scale < 0.1:
            sim.crack_fail_count[path_idx] += 1
            if H_local < min_h_continue * H_ref and sim.crack_fail_count[path_idx] >= max_failures:
                sim.crack_active[path_idx] = False
            continue
        max_step_cells = float(sim.pf_params.get("front_max_step_cells", 1.25))
        speed = min(
            crack_tip_speed * dx * min(speed_scale, 5.0) * dt_ratio,
            max_step_cells * dx * max(dt_ratio, 1.0)
        )
        if (getattr(sim, '_gravity_drop', False)
                and getattr(sim, '_gravity_drop_contacted', False)
                and hasattr(sim, '_impact_center')):
            frame_idx = int(getattr(sim, '_impact_frame_count', 0))
            shatter_frames = int(sim.pf_params.get("impact_shatter_frames", 2))
            if frame_idx < shatter_frames:
                impact_center = sim._impact_center.to(device=device, dtype=tip.dtype)
                dx_xy = tip[:2] - impact_center[:2]
                dz_above = max((tip[2] - impact_center[2]).item(), 0.0)
                local_radius = float(getattr(sim, '_impact_seed_radius_xy', getattr(sim, '_impact_radius', dx)))
                local_depth = float(getattr(sim, '_impact_seed_depth', getattr(sim, '_impact_radius', dx)))
                if (dx_xy.norm().item() <= 1.8 * local_radius and
                        dz_above <= 3.0 * local_depth):
                    base_gain = float(sim.pf_params.get("impact_shatter_tip_gain", 1.4))
                    mat_scale = get_material_shatter_scale(sim)
                    speed *= 1.0 + (base_gain - 1.0) * mat_scale
        best_tip = None
        best_dir = None
        best_score = -1e9
        candidate_dirs = [smooth_dir, raw_dir]
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    alt_dir = torch.tensor([float(di), float(dj), float(dk)], device=device)
                    alt_dir = alt_dir / alt_dir.norm()
                    candidate_dirs.append(alt_dir)

        forward_cos_min = float(sim.pf_params.get("front_forward_cos_min", 0.10))
        filtered_dirs = []
        for cand_dir in candidate_dirs:
            if float((cand_dir * smooth_dir).sum().item()) >= forward_cos_min:
                filtered_dirs.append(cand_dir)
        if filtered_dirs:
            candidate_dirs = filtered_dirs

        for cand_dir in candidate_dirs:
            cand_tip = (tip + speed * cand_dir).clamp(dx, 1.0 - dx)
            score = _front_band_score(sim, cand_tip, cand_dir, smooth_dir, H_ref)
            if score > best_score:
                best_score = score
                best_tip = cand_tip
                best_dir = cand_dir

        if best_tip is None or best_score < 0.0:
            sim.crack_fail_count[path_idx] += 1
            if sim.crack_fail_count[path_idx] >= max_failures:
                sim.crack_active[path_idx] = False
            continue

        new_tip = best_tip
        sim.crack_dirs[path_idx] = best_dir

        if (new_tip - tip).norm() > min_step_dist:
            sim.crack_paths[path_idx] = torch.cat([path, new_tip.unsqueeze(0)], dim=0)
            sim.crack_fail_count[path_idx] = 0

        n_branches_so_far = sim._branch_count.get(path_idx, 0)
        path_len = sim.crack_paths[path_idx].shape[0]
        can_branch = (path_len >= branch_min_len and
                      n_branches_so_far < max_branches and
                      len(sim.crack_paths) + len(pending_branches) * 2 < max_total_cracks and
                      H_local > 0.3 * H_ref)
        if can_branch and torch.rand(1).item() < branch_prob_step:
            parent_dir = sim.crack_dirs[path_idx]
            if parent_dir is not None:
                dir1 = sim._rotate_direction(parent_dir, branch_angle)
                dir2 = sim._rotate_direction(parent_dir, -branch_angle)
                pending_branches.append((new_tip.clone(), dir1, dir2))
                sim._branch_count[path_idx] = n_branches_so_far + 1

    for tip_pos, dir1, dir2 in pending_branches:
        _append_crack_path(sim, tip_pos.unsqueeze(0), dir1)
        _append_crack_path(sim, tip_pos.unsqueeze(0), dir2)
        if step < 50:
            print(f"  [BRANCH] New fork at [{tip_pos[0]:.3f},{tip_pos[1]:.3f},{tip_pos[2]:.3f}] "
                  f"angle={branch_angle}", flush=True)


@torch.no_grad()
def solve_at2_phase_field(sim, n_iters: int = 30, crack_paths: list = None):
    """Solve AT2 phase field via Jacobi iteration."""
    n = sim.mpm.num_grids
    dx = sim.mpm.dx
    device = sim.c_grid.device

    Gc = getattr(sim.elasticity, 'Gc', 30.0)
    l0 = getattr(sim.elasticity, 'l0', 0.025)
    H_ref = Gc / (2.0 * l0)

    Gc_over_l0 = Gc / l0
    Gc_l0_over_dx2 = Gc * l0 / (dx * dx)
    H2 = 2.0 * sim.H_grid
    front_core = getattr(sim, '_front_core_grid', None)
    front_tube = getattr(sim, '_front_tube_grid', None)
    if front_tube is not None:
        front_seed_frac = float(sim.pf_params.get("front_h_seed_fraction", 0.75))
        H2 = torch.maximum(H2, 2.0 * front_seed_frac * H_ref * front_tube.float())
        n_iters += int(sim.pf_params.get("front_extra_iters", 12))
    diag = H2 + Gc_over_l0 + 6.0 * Gc_l0_over_dx2

    crack_bc = None
    if crack_paths:
        crack_raw = torch.zeros(n, n, n, device=device)
        tip_tail_points = max(1, int(sim.pf_params.get("front_bc_tail_points", 2)))
        for path_idx, path in enumerate(crack_paths):
            if hasattr(sim, 'crack_active') and not sim.crack_active[path_idx]:
                continue
            tail = path[-tip_tail_points:] if path.shape[0] > tip_tail_points else path
            gi = (tail * n).long().clamp(0, n - 1)
            crack_raw[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0
        crack_bc = torch.nn.functional.max_pool3d(
            crack_raw.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        )[0, 0] > 0.5

    c_old = sim.c_grid.clone()
    c = sim.c_grid.clone()
    occ = sim.grid_occupied.float()

    for _ in range(n_iters):
        cp = torch.nn.functional.pad(
            c.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1, 1, 1), mode='replicate'
        )[0, 0]

        nbr_sum = (cp[2:, 1:-1, 1:-1] + cp[:-2, 1:-1, 1:-1] +
                   cp[1:-1, 2:, 1:-1] + cp[1:-1, :-2, 1:-1] +
                   cp[1:-1, 1:-1, 2:] + cp[1:-1, 1:-1, :-2])

        c = (H2 + Gc_l0_over_dx2 * nbr_sum) / (diag + 1e-12)
        c = c.clamp(0.0, 1.0) * occ
        c = torch.maximum(c, c_old)
        front_bc = float(sim.pf_params.get("front_bc_value", 0.35))
        if crack_bc is not None:
            c = torch.maximum(c, front_bc * crack_bc.float())
        if front_core is not None:
            c = torch.maximum(c, front_bc * front_core.float())

    sim.c_grid = c


@torch.no_grad()
def assign_crack_damage(sim, crack_width: float = 0.025):
    """Assign geometric damage near the active crack front only.

    Keep fast crack-front propagation, but avoid turning the full historical
    polyline trace into immediate volumetric damage. Recent path segments and
    the current tip get priority; older trail damage only survives if the
    underlying H/AT2 state is already active there.
    """
    if not sim.crack_paths:
        return

    positions = sim.x_mpm
    min_dist = torch.full((positions.shape[0],), float('inf'), device=positions.device)
    tip_min_dist = torch.full((positions.shape[0],), float('inf'), device=positions.device)

    tail_points = max(2, int(sim.pf_params.get("crack_damage_tail_points", 5)))
    tip_width_scale = float(sim.pf_params.get("crack_damage_tip_width_scale", 1.35))
    h_gate_frac = float(sim.pf_params.get("crack_damage_h_fraction", 0.08))
    persist_thresh = float(sim.pf_params.get("crack_damage_persist_threshold", 0.18))

    for path_idx, path in enumerate(sim.crack_paths):
        if hasattr(sim, 'crack_active') and not sim.crack_active[path_idx]:
            continue
        tail = path[-tail_points:] if path.shape[0] > tail_points else path
        if tail.shape[0] == 1:
            dist = (positions - tail[0]).norm(dim=1)
        else:
            dist = sim._point_to_polyline_dist(positions, tail)
        min_dist = torch.minimum(min_dist, dist)
        tip_dist = (positions - path[-1]).norm(dim=1)
        tip_min_dist = torch.minimum(tip_min_dist, tip_dist)

    c_tail = (1.0 - (min_dist / crack_width)).clamp(0.0, 1.0)
    c_tip = (1.0 - (tip_min_dist / (crack_width * tip_width_scale))).clamp(0.0, 1.0)
    c_geom = torch.maximum(c_tail, c_tip)

    if hasattr(sim, '_history_H'):
        Gc = max(float(getattr(sim.elasticity, 'Gc', 30.0)), 1e-8)
        l0 = max(float(getattr(sim.elasticity, 'l0', 0.025)), 1e-8)
        H_ref = Gc / (2.0 * l0)
        h_gate = sim._history_H > (h_gate_frac * H_ref)
    else:
        h_gate = torch.zeros_like(sim.c_vol, dtype=torch.bool)

    # Keep the current front visible, but only keep historical trail damage
    # where the underlying AT2/H state is already active.
    c_front_gate = sim.c_vol > float(sim.pf_params.get("crack_damage_c_front_gate", 0.14))
    gate = h_gate | c_front_gate | (sim.c_vol > persist_thresh) | (c_tip > 0.05)
    c_geom = torch.where(gate, c_geom, torch.zeros_like(c_geom))
    geom_scale = float(sim.pf_params.get("crack_damage_geom_scale", 0.65))
    sim.c_vol = torch.maximum(sim.c_vol, geom_scale * c_geom)
