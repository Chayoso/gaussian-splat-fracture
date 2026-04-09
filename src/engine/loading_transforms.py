"""
Loading transforms — gravity drop and seismic object positioning.

Applies scale, rotation, and z-shift to MPM particles and Gaussians
before simulation begins. Extracted from run.py main().
"""

import math
import numpy as np
import torch
from omegaconf import OmegaConf


def apply_loading_transforms(config: OmegaConf, simulator, loading_params: dict,
                             device: torch.device):
    """
    Apply pre-simulation transforms based on loading type.

    For gravity_drop: scale down, reposition, rotate, enable drop mode.
    For seismic: optional object rescaling.

    Args:
        config: Full simulation configuration
        simulator: HybridCrackSimulator instance
        loading_params: dict from configure_loading()
        device: PyTorch device
    """
    if loading_params.get("type") == "gravity_drop":
        _apply_gravity_drop(config, simulator, device)
    elif loading_params.get("type") == "seismic":
        _apply_seismic_rescale(config, simulator, device)


def _apply_gravity_drop(config, simulator, device):
    """Scale, shift, rotate object for gravity drop, then enable drop mode."""
    ground_z = 0.1
    if hasattr(config, 'ground_plane') and config.ground_plane.get('enabled', False):
        ground_z = float(config.ground_plane.get('point', [0.5, 0.5, 0.1])[2])

    drop_scale = config.loading.get('drop_scale', 0.33)
    drop_center_z = config.loading.get('drop_center_z', 0.7)

    # Scale all MPM positions around center [0.5, 0.5, 0.5]
    center = torch.tensor([0.5, 0.5, 0.5], device=device)
    simulator.x_mpm = center + (simulator.x_mpm - center) * drop_scale

    # Shift z so object center is at drop_center_z
    z_current_center = (simulator.x_mpm[:, 2].min().item() +
                        simulator.x_mpm[:, 2].max().item()) / 2
    z_shift = drop_center_z - z_current_center
    simulator.x_mpm[:, 2] += z_shift

    # Apply drop rotation (Euler XYZ degrees)
    drop_rotation = config.loading.get('drop_rotation', None)
    R_mat = None
    if drop_rotation is not None:
        rot_deg = [float(r) for r in drop_rotation]
        R_mat = _euler_to_rotation_matrix(rot_deg, device)
        obj_center = simulator.x_mpm.mean(dim=0)
        simulator.x_mpm = obj_center + (simulator.x_mpm - obj_center) @ R_mat.T
        print(f"  - Drop rotation: {rot_deg} deg")

    # Adjust Gaussian splats for pretrained PLY
    if config.gaussian_splatting.get("pretrained_ply", None) is not None:
        _adjust_pretrained_gaussians(simulator, drop_scale, z_shift,
                                     drop_rotation, R_mat, device)

    z_min = simulator.x_mpm[:, 2].min().item()
    z_max = simulator.x_mpm[:, 2].max().item()
    print(f"[GravityDrop] Rescaled object: scale={drop_scale}, "
          f"center_z={drop_center_z}")
    print(f"  - z range: [{z_min:.4f}, {z_max:.4f}]")
    print(f"  - Fall distance to ground: {z_min - ground_z:.4f}")

    simulator.enable_gravity_drop(ground_z=ground_z)


def _apply_seismic_rescale(config, simulator, device):
    """Optional object rescaling for seismic loading."""
    obj_scale = config.loading.get('object_scale', None)
    if obj_scale is None:
        return

    obj_scale = float(obj_scale)
    obj_center = list(config.loading.get('object_center', [0.5, 0.5, 0.35]))

    cube_center = torch.tensor([0.5, 0.5, 0.5], device=device,
                               dtype=simulator.x_mpm.dtype)
    simulator.x_mpm = cube_center + (simulator.x_mpm - cube_center) * obj_scale

    current_center = simulator.x_mpm.mean(dim=0)
    target_center = torch.tensor(obj_center, device=device,
                                 dtype=simulator.x_mpm.dtype)
    simulator.x_mpm += (target_center - current_center)

    z_min = simulator.x_mpm[:, 2].min().item()
    z_max = simulator.x_mpm[:, 2].max().item()
    print(f"[Seismic] Rescaled object: scale={obj_scale}, center={obj_center}")
    print(f"  - z range: [{z_min:.4f}, {z_max:.4f}]")
    print(f"  - Gap to ground: "
          f"{z_min - float(config.ground_plane.get('point', [0,0,0.05])[2]):.4f}")


def _euler_to_rotation_matrix(rot_deg, device):
    """Build rotation matrix from Euler XYZ angles (degrees)."""
    rot_rad = [np.radians(r) for r in rot_deg]
    cx, sx = np.cos(rot_rad[0]), np.sin(rot_rad[0])
    cy, sy = np.cos(rot_rad[1]), np.sin(rot_rad[1])
    cz, sz = np.cos(rot_rad[2]), np.sin(rot_rad[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return torch.tensor(Rz @ Ry @ Rx, dtype=torch.float32, device=device)


def _rotmat_to_quat(R_np):
    """Convert 3x3 rotation matrix (numpy) to quaternion [w, x, y, z]."""
    tr = R_np[0, 0] + R_np[1, 1] + R_np[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return [0.25 * s,
                (R_np[2, 1] - R_np[1, 2]) / s,
                (R_np[0, 2] - R_np[2, 0]) / s,
                (R_np[1, 0] - R_np[0, 1]) / s]
    elif R_np[0, 0] > R_np[1, 1] and R_np[0, 0] > R_np[2, 2]:
        s = np.sqrt(1.0 + R_np[0, 0] - R_np[1, 1] - R_np[2, 2]) * 2
        return [(R_np[2, 1] - R_np[1, 2]) / s,
                0.25 * s,
                (R_np[0, 1] + R_np[1, 0]) / s,
                (R_np[0, 2] + R_np[2, 0]) / s]
    elif R_np[1, 1] > R_np[2, 2]:
        s = np.sqrt(1.0 + R_np[1, 1] - R_np[0, 0] - R_np[2, 2]) * 2
        return [(R_np[0, 2] - R_np[2, 0]) / s,
                (R_np[0, 1] + R_np[1, 0]) / s,
                0.25 * s,
                (R_np[1, 2] + R_np[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R_np[2, 2] - R_np[0, 0] - R_np[1, 1]) * 2
        return [(R_np[1, 0] - R_np[0, 1]) / s,
                (R_np[0, 2] + R_np[2, 0]) / s,
                (R_np[1, 2] + R_np[2, 1]) / s,
                0.25 * s]


def _quat_multiply_batch(q_rot, q_old):
    """Hamilton product q_rot * q_old. q_rot: (4,), q_old: (K, 4). wxyz format."""
    w1, x1, y1, z1 = q_rot[0], q_rot[1], q_rot[2], q_rot[3]
    w2, x2, y2, z2 = q_old[:, 0], q_old[:, 1], q_old[:, 2], q_old[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _adjust_pretrained_gaussians(simulator, drop_scale, z_shift,
                                  drop_rotation, R_mat, device):
    """Adjust pretrained PLY Gaussian scales, positions, and rotations."""
    simulator.gaussians._scaling.data += math.log(drop_scale)
    print(f"  - Pretrained splat scales adjusted by log({drop_scale})="
          f"{math.log(drop_scale):.4f}")

    if getattr(simulator, '_ply_direct', False):
        ply_center = torch.tensor([0.5, 0.5, 0.5], device=device)
        simulator._ply_init_xyz = (
            ply_center + (simulator._ply_init_xyz - ply_center) * drop_scale)
        simulator._ply_init_xyz[:, 2] += z_shift

        if drop_rotation is not None and R_mat is not None:
            ply_obj_center = simulator._ply_init_xyz.mean(dim=0)
            simulator._ply_init_xyz = (
                ply_obj_center + (simulator._ply_init_xyz - ply_obj_center) @ R_mat.T)

            # Rotate Gaussian quaternions
            R_np = R_mat.cpu().numpy()
            q_rot = torch.tensor(_rotmat_to_quat(R_np),
                                 dtype=torch.float32, device=device)
            simulator.gaussians._rotation.data = _quat_multiply_batch(
                q_rot, simulator.gaussians._rotation.data)

        simulator._surf_init_world = simulator.mapper.mpm_to_world(
            simulator.x_mpm[simulator.surface_mask])
