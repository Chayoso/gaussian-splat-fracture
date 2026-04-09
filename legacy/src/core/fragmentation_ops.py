"""Fragment detection helpers for HybridCrackSimulator."""

from __future__ import annotations

import math

import torch

from src.core.crack_evolution import get_material_shatter_scale


def maybe_detect_fragments(sim):
    """Run fragment detection if conditions are met."""
    if not (sim.fragment_manager is not None
            and sim._gravity_drop and sim._gravity_drop_contacted
            and hasattr(sim, 'grid_occupied')
            and hasattr(sim, 'crack_paths') and len(sim.crack_paths) > 0):
        return

    n_frags = detect_fragments_from_crack_planes(sim)

    if n_frags > 1:
        first_detection = not sim.fragmentation_active
        sim.fragmentation_active = True

        if first_detection and hasattr(sim, '_impact_center'):
            ic = sim._impact_center.unsqueeze(0)
            total_particles = sum(len(fi) for fi in sim.fragment_manager.fragment_particle_indices)
            frame_idx = int(getattr(sim, '_impact_frame_count', 0))
            shatter_frames = int(sim.pf_params.get("fragment_first_separation_frames", 4))
            early_shatter = frame_idx < shatter_frames
            mat_scale = get_material_shatter_scale(sim)
            if early_shatter:
                impulse_base = float(sim.pf_params.get("fragment_first_separation_impulse", 4.0))
                impulse_base *= mat_scale
                upward_bias = float(sim.pf_params.get("fragment_first_separation_upward_bias", 0.9))
                normal_bias = float(sim.pf_params.get("fragment_first_normal_bias", 1.0))
            else:
                impulse_base = float(sim.pf_params.get("fragment_separation_impulse", 2.5))
                upward_bias = float(sim.pf_params.get("fragment_separation_upward_bias", 0.7))
                normal_bias = float(sim.pf_params.get("fragment_normal_bias", 0.35))
            contact_normal = getattr(sim, '_impact_contact_normal', None)
            for frag_idx in sim.fragment_manager.fragment_particle_indices:
                n_p = len(frag_idx)
                if n_p < 10:
                    continue
                com = sim.x_mpm[frag_idx].mean(dim=0, keepdim=True)
                direction = com - ic
                dist = direction.norm() + 1e-8
                direction = direction / dist
                direction[0, 2] += upward_bias
                if contact_normal is not None:
                    direction = direction + normal_bias * contact_normal.view(1, 3)
                direction = direction / (direction.norm() + 1e-8)
                size_ratio = n_p / (total_particles + 1e-8)
                if early_shatter:
                    impulse_strength = impulse_base * max(0.45, min(1.15, size_ratio * 6.0))
                else:
                    impulse_strength = impulse_base * max(0.3, min(1.0, size_ratio * 5.0))
                sim.v_mpm[frag_idx] += impulse_strength * direction
            mode = "first-shatter" if early_shatter else "standard"
            print(f"  [SEPARATION] Applied {mode} impulse to {n_frags} fragments")



@torch.no_grad()
def detect_fragments_from_crack_planes(sim) -> int:
    """Build planar crack surfaces from polylines and run CC."""
    n = sim.mpm.num_grids
    dx = 1.0 / n
    device = sim.x_mpm.device
    frame_idx = int(getattr(sim, '_impact_frame_count', 0))
    shatter_frames = int(sim.pf_params.get("impact_shatter_frames", 2))
    early_shatter = frame_idx < shatter_frames

    impact_center = getattr(sim, '_impact_center', sim.x_mpm.mean(dim=0))
    ic_xy = impact_center[:2]
    Gc = max(float(getattr(sim.elasticity, 'Gc', 30.0)), 1e-8)
    l0 = max(float(getattr(sim.elasticity, 'l0', 0.025)), 1e-8)
    H_ref = Gc / (2.0 * l0)
    tail_points = max(2, int(sim.pf_params.get("fragment_plane_tail_points", 4)))
    tip_h_frac = float(sim.pf_params.get("fragment_plane_h_fraction", 0.08))
    tip_c_min = float(sim.pf_params.get("fragment_plane_c_min", 0.08))

    coords = (torch.arange(n, device=device).float() + 0.5) / n

    damage_grid = torch.zeros(n, n, n, device=device)
    thickness_gain = float(sim.pf_params.get("impact_shatter_plane_thickness_gain", 1.35))
    plane_thickness = (2.5 * dx) * (thickness_gain if early_shatter else 1.0)

    seen_angles = []
    unique_planes = []

    def stamp_plane(angle: float, z_lo: float, z_hi: float):
        normal_xy = torch.tensor(
            [-math.sin(angle), math.cos(angle)],
            device=device,
            dtype=torch.float32,
        )
        cell_x = coords
        cell_y = coords
        dist_x = cell_x.unsqueeze(1) - ic_xy[0]
        dist_y = cell_y.unsqueeze(0) - ic_xy[1]
        signed_dist = dist_x * normal_xy[0] + dist_y * normal_xy[1]
        plane_mask_2d = signed_dist.abs() < (plane_thickness / 2)
        z_lo_idx = max(0, int(z_lo * n))
        z_hi_idx = min(n, int(z_hi * n) + 1)
        plane_mask_3d = plane_mask_2d.unsqueeze(2).expand(n, n, n).clone()
        plane_mask_3d[:, :, :z_lo_idx] = False
        plane_mask_3d[:, :, z_hi_idx:] = False
        damage_grid[plane_mask_3d] = 1.0

    for path_idx, path in enumerate(sim.crack_paths):
        if hasattr(sim, 'crack_active') and not sim.crack_active[path_idx]:
            continue
        if path.shape[0] < 2:
            continue
        tip = path[-1]
        gi = (tip * n).long().clamp(0, n - 1)
        H_grid = getattr(sim, '_H_grid_physics', sim.H_grid)
        H_tip = float(H_grid[gi[0], gi[1], gi[2]].item())
        c_tip = float(sim.c_grid[gi[0], gi[1], gi[2]].item()) if hasattr(sim, 'c_grid') else 0.0
        if H_tip < tip_h_frac * H_ref and c_tip < tip_c_min:
            continue

        tail = path[-tail_points:] if path.shape[0] > tail_points else path
        path_center = tail.mean(dim=0)
        radial = path_center[:2] - ic_xy
        if radial.norm() < 1e-6:
            continue
        radial = radial / radial.norm()
        angle = torch.atan2(radial[1], radial[0]).item()

        is_dup = False
        for sa in seen_angles:
            diff = abs(angle - sa)
            diff = min(diff, 2 * 3.14159 - diff)
            if diff < 0.087:
                is_dup = True
                break
        if is_dup:
            continue
        seen_angles.append(angle)

        z_vals = tail[:, 2]
        z_lo_pad = 0.08 * (1.15 if early_shatter else 1.0)
        z_hi_pad = 0.15 * (1.35 if early_shatter else 1.0)
        z_lo = max(0.0, z_vals.min().item() - z_lo_pad)
        z_hi = min(1.0, z_vals.max().item() + z_hi_pad)
        stamp_plane(angle, z_lo, z_hi)

        unique_planes.append(angle)

    damage_grid = damage_grid * sim.grid_occupied.float()

    n_crack_cells = (damage_grid > 0.5).sum().item()
    n_occupied = sim.grid_occupied.sum().item()
    print(f"  [FRAGMENT] Built {len(unique_planes)} crack planes, "
          f"{n_crack_cells}/{n_occupied} cells marked as crack")

    n_frags = sim.fragment_manager.detect_fragments(
        damage_grid, sim.grid_occupied, sim.x_mpm, sim.mpm)

    print(f"  [FRAGMENT] CC detected {n_frags} fragments")
    for k in range(min(n_frags, 20)):
        nk = len(sim.fragment_manager.fragment_particle_indices[k])
        print(f"    Fragment {k}: {nk} particles")

    return n_frags
