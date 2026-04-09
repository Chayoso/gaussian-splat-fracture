"""
MPM Pipeline — initialization and loading configuration.

Extracted from run.py to enable programmatic reuse in ForwardEngine.
"""

import torch
import numpy as np
from omegaconf import OmegaConf

from src.mpm_core.mpm_model import MPMModel
from src.mpm_core.set_boundary_conditions import add_surface_collider


def create_mpm_model(config: OmegaConf, volume_pcd, device: torch.device) -> MPMModel:
    """
    Initialize MPM simulation model from config and point cloud.

    Args:
        config: Full simulation configuration
        volume_pcd: BasicPointCloud with .points (N, 3)
        device: PyTorch device

    Returns:
        MPMModel instance with grid, particles, and boundary conditions
    """
    sim_params = OmegaConf.create({
        "num_grids": config.mpm.num_grids,
        "dt": config.mpm.dt,
        "gravity": config.mpm.gravity,
        "clip_bound": config.mpm.clip_bound,
        "damping": config.mpm.damping
    })

    material_params = OmegaConf.create({
        "center": config.material.center,
        "size": config.material.size,
        "rho": config.material.density
    })

    mpm_model = MPMModel(
        sim_params=sim_params,
        material_params=material_params,
        init_pos=torch.from_numpy(volume_pcd.points).float().to(device),
        device=device
    )

    if hasattr(config.mpm, 'particle_chunk'):
        mpm_model.particle_chunk = config.mpm.particle_chunk

    # Add ground plane collision if configured
    if hasattr(config, 'ground_plane') and config.ground_plane.get('enabled', False):
        gp = config.ground_plane
        add_surface_collider(
            model=mpm_model,
            point=list(gp.get('point', [0.5, 0.1, 0.5])),
            normal=list(gp.get('normal', [0.0, 1.0, 0.0])),
            surface=gp.get('surface_type', 'slip'),
            friction=float(gp.get('friction', 0.5)),
        )
        print(f"  - Ground plane: point={list(gp.point)}, "
              f"normal={list(gp.normal)}, type={gp.surface_type}")

    return mpm_model


def configure_loading(config: OmegaConf, mpm_model: MPMModel,
                      device: torch.device) -> dict:
    """
    Configure loading conditions (gravity_drop, compression, seismic, point_impact).

    Modifies mpm_model in-place (gravity, damping, boundary conditions).

    Args:
        config: Full simulation configuration
        mpm_model: MPMModel to configure
        device: PyTorch device

    Returns:
        dict with keys:
          - "type": loading type string
          - "seismic_override": bool or None
    """
    loading_type = None
    if hasattr(config, 'loading'):
        loading_type = config.loading.get('type', None)

    # Backward compatibility: infer from legacy flags
    if loading_type is None:
        if hasattr(config, 'seismic') and config.seismic.get('enabled', False):
            loading_type = "seismic"
        elif hasattr(config, 'external_force') and config.external_force.get('enabled', False):
            loading_type = "point_impact"
        else:
            loading_type = "seismic"

    print(f"\n[Loading] Type: {loading_type}")
    result = {"type": loading_type}

    if loading_type == "gravity_drop":
        _configure_gravity_drop(config, mpm_model, device, result)
    elif loading_type == "compression":
        _configure_compression(config, mpm_model, result)
    elif loading_type == "point_impact":
        result["seismic_override"] = False
        print(f"  - Using external_force config for point impact")
        print(f"  - Seismic: OFF (point_impact mode)")
    elif loading_type == "seismic":
        result["seismic_override"] = None
        print(f"  - Using seismic config (existing behavior)")
    else:
        raise ValueError(
            f"Unknown loading type: '{loading_type}'. "
            f"Options: seismic, gravity_drop, point_impact, compression"
        )

    return result


def _configure_gravity_drop(config, mpm_model, device, result):
    """Set up gravity drop mode: gravity, damping, ground plane, warmup."""
    fixed_gravity = config.loading.get('fixed_gravity_z', None)
    if fixed_gravity is None:
        total_frames = config.rendering.get('total_frames', 100)
        substeps = config.rendering.get('physics_substeps', 10)
        dt = float(config.mpm.dt)
        total_time = total_frames * substeps * dt
        target_fall = 0.35

        g_needed = 2.0 * target_fall / (total_time ** 2) if total_time > 0 else 500.0
        g_needed = max(g_needed, 200.0)
        g_needed = min(g_needed, 5000.0)
    else:
        g_needed = abs(float(fixed_gravity))

    mpm_model.gravity = torch.tensor([0.0, 0.0, -g_needed], device=device)
    if fixed_gravity is None:
        print(f"  - Effective gravity: [0, 0, {-g_needed:.1f}]  (sim_time={total_time:.4f}s)")
    else:
        print(f"  - Fixed gravity override: [0, 0, {-g_needed:.1f}]")

    mpm_model.damping = 0.9995
    print(f"  - Damping: {mpm_model.damping} (near-free-fall for impact)")

    # Auto-enable ground plane if not already configured
    if not (hasattr(config, 'ground_plane') and config.ground_plane.get('enabled', False)):
        add_surface_collider(
            model=mpm_model,
            point=[0.5, 0.5, 0.1],
            normal=[0.0, 0.0, 1.0],
            surface="slip",
            friction=0.5,
        )
        print(f"  - Auto-enabled ground plane at z=0.1 (slip)")

    warmup_val = config.phase_field.get('warmup_frames', 3)
    warmup_val = min(warmup_val, 10)
    OmegaConf.update(config, "phase_field.warmup_frames", warmup_val)
    print(f"  - warmup_frames={warmup_val} (resets at impact)")

    result["seismic_override"] = False
    print(f"  - Seismic: OFF (gravity_drop mode)")


def _configure_compression(config, mpm_model, result):
    """Set up compression loading: two plates moving inward."""
    from src.mpm_core.set_boundary_conditions import set_velocity_on_cuboid

    comp = config.loading.get('compression', {})
    axis = int(comp.get('axis', 0))
    speed = float(comp.get('speed', 0.5))
    start_pos = float(comp.get('start_position', 0.1))
    end_pos = float(comp.get('end_position', 0.9))

    # Lower plate
    lower_point = [0.5, 0.5, 0.5]
    lower_point[axis] = start_pos
    lower_size = [0.5, 0.5, 0.5]
    lower_size[axis] = 0.02
    lower_vel = [0.0, 0.0, 0.0]
    lower_vel[axis] = speed

    set_velocity_on_cuboid(
        model=mpm_model, point=lower_point,
        size=lower_size, velocity=lower_vel,
    )

    # Upper plate
    upper_point = [0.5, 0.5, 0.5]
    upper_point[axis] = end_pos
    upper_size = [0.5, 0.5, 0.5]
    upper_size[axis] = 0.02
    upper_vel = [0.0, 0.0, 0.0]
    upper_vel[axis] = -speed

    set_velocity_on_cuboid(
        model=mpm_model, point=upper_point,
        size=upper_size, velocity=upper_vel,
    )

    result["seismic_override"] = False
    axis_names = ["X", "Y", "Z"]
    print(f"  - Compression axis: {axis_names[axis]}")
    print(f"  - Plate speed: {speed}")
    print(f"  - Plates: [{start_pos}] --> <-- [{end_pos}]")
    print(f"  - Seismic: OFF (compression mode)")
