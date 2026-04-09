"""
Main Entry Point for MPM + Gaussian Splatting Crack Simulation

This script orchestrates the entire simulation pipeline:
1. Load configuration from YAML
2. Convert 3D mesh to point clouds
3. Initialize MPM physics with Phase Field
4. Initialize Gaussian Splats for rendering
5. Run simulation with external forces
6. Generate output video

Usage:
    python run.py --config configs/simulation_config.yaml
    python run.py --config my_config.yaml --mesh assets/meshes/bunny.obj
"""

import torch
import numpy as np
import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Optional
import time
import csv

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gaussian-splatting"))

# Project imports
from src.preprocessing.mesh_converter import MeshToPointCloudConverter
from src.core.coordinate_mapper import CoordinateMapper
from src.core.hybrid_simulator import HybridCrackSimulator
from src.constitutive_models.damage_mapper import VolumetricToSurfaceDamageMapper
from src.visualization.gaussian_updater import GaussianCrackVisualizer
from src.mpm_core.mpm_model import MPMModel
from src.constitutive_models.physical_constitutive_models import (
    PhaseFieldElasticity,
    CorotatedPhaseFieldElasticity
)

# Gaussian Splatting imports
try:
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from scene.cameras import Camera, MiniCam
    from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix
    from utils.sh_utils import SH2RGB
    from torchvision.utils import save_image
except ImportError as e:
    print(f"[Warning] Gaussian Splatting modules not fully available: {e}")
    GaussianModel = None


# ============================================================================
# Configuration & Setup
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MPM Crack Simulation with Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --config configs/simulation_config.yaml
  python run.py --config my_config.yaml --mesh assets/meshes/bunny.obj
  python run.py --config configs/simulation_config.yaml --frames 500
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/simulation_config.yaml",
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--mesh", "-m",
        type=str,
        default=None,
        help="Override mesh path from config"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Override output video path from config"
    )

    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=None,
        help="Override total frames from config"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video generation"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


def load_config(config_path: str) -> OmegaConf:
    """
    Load and validate YAML configuration

    Args:
        config_path: Path to YAML config file

    Returns:
        OmegaConf configuration object
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Validate required fields
    required_sections = [
        "simulation", "mesh", "particles", "mpm",
        "material", "phase_field", "gaussian_splatting", "rendering"
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    print(f"[Config] Loaded configuration from: {config_path}")
    print(f"  - Simulation: {config.simulation.name}")
    print(f"  - Mesh: {config.mesh.path}")
    print(f"  - Particles: {config.particles.target_count}")
    print(f"  - Frames: {config.rendering.total_frames}")

    return config


def apply_cli_overrides(config: OmegaConf, args) -> OmegaConf:
    """
    Apply command-line argument overrides to config

    Args:
        config: Base configuration
        args: Parsed command-line arguments

    Returns:
        Modified configuration
    """
    if args.mesh is not None:
        config.mesh.path = args.mesh
        print(f"[Config] Override mesh: {args.mesh}")

    if args.output is not None:
        config.output.video_path = args.output
        print(f"[Config] Override output: {args.output}")

    if args.frames is not None:
        config.rendering.total_frames = args.frames
        print(f"[Config] Override frames: {args.frames}")

    if args.device is not None:
        config.device.type = args.device
        print(f"[Config] Override device: {args.device}")

    return config


# ============================================================================
# Mesh & Point Cloud Processing
# ============================================================================

def setup_mesh(config: OmegaConf):
    """
    Load or generate mesh, convert to point clouds

    Args:
        config: Simulation configuration

    Returns:
        Tuple of (volume_pcd, surface_pcd, surface_mask)
    """
    print(f"\n{'='*60}")
    print(f"Step 1: Mesh Processing")
    print(f"{'='*60}")

    mesh_path = Path(config.mesh.path)

    # Check if mesh exists, generate if needed
    if not mesh_path.exists() and config.mesh.auto_generate_if_missing:
        print(f"[Mesh] File not found: {mesh_path}")
        print(f"[Mesh] Generating test mesh: {config.mesh.test_mesh_type}")

        import open3d as o3d

        if config.mesh.test_mesh_type == "sphere":
            mesh = o3d.geometry.TriangleMesh.create_sphere(
                radius=config.mesh.test_mesh_radius
            )
        elif config.mesh.test_mesh_type == "cube":
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=config.mesh.test_mesh_radius * 2,
                height=config.mesh.test_mesh_radius * 2,
                depth=config.mesh.test_mesh_radius * 2
            )
        elif config.mesh.test_mesh_type == "torus":
            mesh = o3d.geometry.TriangleMesh.create_torus(
                torus_radius=config.mesh.test_mesh_radius,
                tube_radius=config.mesh.test_mesh_radius * 0.3
            )
        else:
            raise ValueError(f"Unknown test mesh type: {config.mesh.test_mesh_type}")

        mesh.compute_vertex_normals()

        # Save to assets
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"[Mesh] Generated and saved: {mesh_path}")

    # Convert mesh to point clouds
    converter = MeshToPointCloudConverter(
        mesh_path=str(mesh_path),
        target_particle_count=config.particles.target_count,
        surface_sample_ratio=config.particles.surface_ratio,
        use_poisson=config.particles.use_poisson_sampling,
        poisson_depth=config.particles.poisson_depth,
        normalize_to_unit_cube=config.particles.normalize_to_unit_cube
    )

    volume_pcd, surface_pcd, surface_mask = converter.convert()

    # Expose mesh normalization metadata for PLY alignment
    mesh_meta = {
        'center': converter._original_center,
        'scale': converter._original_scale,
    }

    return volume_pcd, surface_pcd, surface_mask, mesh_meta


# ============================================================================
# Simulation Components Initialization
# ============================================================================

def setup_mpm(config: OmegaConf, volume_pcd, device: torch.device):
    """
    Initialize MPM simulation model

    Args:
        config: Simulation configuration
        volume_pcd: Volumetric point cloud
        device: PyTorch device

    Returns:
        MPMModel instance
    """
    print(f"\n{'='*60}")
    print(f"Step 2: MPM Initialization")
    print(f"{'='*60}")

    # Create MPM configuration
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

    # Set chunk size if specified
    if hasattr(config.mpm, 'particle_chunk'):
        mpm_model.particle_chunk = config.mpm.particle_chunk

    # Add ground plane collision if configured
    if hasattr(config, 'ground_plane') and config.ground_plane.get('enabled', False):
        from src.mpm_core.set_boundary_conditions import add_surface_collider
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


def setup_elasticity(config: OmegaConf, device: torch.device):
    """
    Initialize elasticity model based on config.

    Args:
        config: Simulation configuration
        device: PyTorch device

    Returns:
        Elasticity module (PhaseFieldElasticity or CorotatedPhaseFieldElasticity)
    """
    print(f"\n{'='*60}")
    print(f"Step 3: Elasticity Model")
    print(f"{'='*60}")

    model_type = config.material.get("constitutive_model", "phase_field")

    Gc = float(config.material.get("Gc", 100.0))
    l0 = float(config.material.get("l0", 0.03))

    if model_type == "corotated_phase_field":
        elasticity = CorotatedPhaseFieldElasticity(Gc=Gc, l0=l0).to(device)
    else:
        elasticity = PhaseFieldElasticity().to(device)
        elasticity.Gc = Gc
        elasticity.l0 = l0

    # Override buffer values with config parameters
    elasticity.log_E = torch.log(torch.tensor([config.material.youngs_modulus], device=device))
    elasticity.nu = torch.tensor([config.material.poissons_ratio], device=device)

    # Set damage degradation exponent (accessed via getattr in forward())
    elasticity.damage_exp = config.material.degradation_exponent

    print(f"  - Model: {model_type}")
    print(f"  - Young's modulus: {config.material.youngs_modulus:.2e}")
    print(f"  - Poisson's ratio: {config.material.poissons_ratio}")
    print(f"  - Gc: {Gc:.1f}, l0: {l0:.4f}")
    print(f"  - Degradation exponent: {config.material.degradation_exponent}")

    return elasticity


def setup_gaussians(config: OmegaConf, surface_pcd, device: torch.device,
                    mesh_meta: dict = None):
    """
    Initialize Gaussian Splatting model

    Args:
        config: Simulation configuration
        surface_pcd: Surface point cloud
        device: PyTorch device

    Returns:
        GaussianModel instance
    """
    print(f"\n{'='*60}")
    print(f"Step 4: Gaussian Splats Initialization")
    print(f"{'='*60}")

    if GaussianModel is None:
        raise ImportError("GaussianModel not available. Install Gaussian Splatting submodules.")

    # Compute camera parameters for scale initialization
    cam_distance = config.rendering.camera.distance
    cam_elevation = np.radians(config.rendering.camera.elevation)
    cam_azimuth = np.radians(config.rendering.camera.azimuth)
    cam_fov = np.radians(config.rendering.camera.fov)
    img_width = config.rendering.image_width
    img_height = config.rendering.image_height
    aspect_ratio = img_width / img_height
    fov_y = cam_fov
    fov_x = 2 * np.arctan(np.tan(fov_y / 2) * aspect_ratio)

    print(f"\n[Camera Parameters for Scale Init]")
    print(f"  - Distance: {cam_distance:.3f}")
    print(f"  - Image: {img_width}x{img_height}")
    print(f"  - FoV: {np.degrees(fov_x):.1f}° x {np.degrees(fov_y):.1f}°")

    gaussians = GaussianModel(sh_degree=config.gaussian_splatting.sh_degree)

    pretrained_ply = config.gaussian_splatting.get("pretrained_ply", None)

    if pretrained_ply is not None:
        # Pre-trained 3DGS .ply → photorealistic appearance
        from src.preprocessing.ply_loader import PretrainedPlyLoader

        scale_mult = config.gaussian_splatting.get("pretrained_scale_multiplier", 1.0)
        use_direct = config.gaussian_splatting.get("ply_direct", False)
        fg_distance = config.gaussian_splatting.get("ply_fg_distance", 0.08)

        loader = PretrainedPlyLoader(
            ply_path=pretrained_ply,
            sh_degree=config.gaussian_splatting.sh_degree,
        )
        ply_data = loader.load_raw_ply()

        # Align PLY to mesh coordinate system (uses same center/scale as mesh)
        mesh_center = mesh_meta.get('center') if mesh_meta else None
        mesh_scale = mesh_meta.get('scale') if mesh_meta else None
        ply_xyz_norm, scale_factor = loader.normalize_positions(
            ply_data['xyz'], mesh_center=mesh_center, mesh_scale=mesh_scale)

        surface_xyz = np.asarray(surface_pcd.points)

        if use_direct:
            # Direct PLY mode: use foreground PLY Gaussians directly
            fg_mask, _ = loader.filter_foreground(
                ply_xyz_norm, surface_xyz, max_distance=fg_distance)
            ply_to_surface = loader.create_direct_gaussians(
                gaussians, ply_data, ply_xyz_norm, surface_xyz,
                scale_factor, scale_multiplier=scale_mult,
                fg_mask=fg_mask)
            # Store mapping for displacement tracking
            gaussians._ply_to_surface = ply_to_surface
            gaussians._ply_direct_mode = True
            print(f"  - Direct PLY mode: {fg_mask.sum()} foreground Gaussians")
        else:
            # Matched mode: one Gaussian per surface particle
            match_indices, distances = loader.match_to_surface_particles(
                ply_xyz_norm, surface_xyz)
            loader.create_matched_gaussians(
                gaussians, surface_pcd, ply_data, match_indices,
                scale_factor, scale_multiplier=scale_mult)
            gaussians._ply_direct_mode = False

        print(f"  - Loaded pretrained PLY: {pretrained_ply}")
    else:
        # Procedural gray splats from mesh
        gaussians.create_from_pcd(
            surface_pcd,
            cam_infos=[],
            spatial_lr_scale=1.0,
            camera_distance=cam_distance,
            image_width=img_width,
            fov_x=fov_x
        )

    # Move to device
    gaussians._xyz = gaussians._xyz.to(device)
    gaussians._features_dc = gaussians._features_dc.to(device)
    gaussians._features_rest = gaussians._features_rest.to(device)
    gaussians._opacity = gaussians._opacity.to(device)
    gaussians._scaling = gaussians._scaling.to(device)
    gaussians._rotation = gaussians._rotation.to(device)

    # Ensure _normal exists (load_ply/pretrained path may not set it)
    if not hasattr(gaussians, '_normal') or gaussians._normal is None:
        gaussians._normal = torch.zeros(
            (gaussians._xyz.shape[0], 3), device=device
        )

    print(f"  - Surface Gaussians: {gaussians._xyz.shape[0]}")
    print(f"  - SH degree: {config.gaussian_splatting.sh_degree}")

    return gaussians


def setup_simulator(
    config: OmegaConf,
    mpm_model,
    gaussians,
    elasticity,
    surface_mask,
    device: torch.device,
    loading_params: dict = None
):
    """
    Create hybrid MPM + Gaussian Splats simulator

    Args:
        config: Simulation configuration
        mpm_model: MPM model
        gaussians: Gaussian model
        elasticity: Elasticity module
        surface_mask: Surface particle mask
        device: PyTorch device

    Returns:
        HybridCrackSimulator instance
    """
    print(f"\n{'='*60}")
    print(f"Step 5: Hybrid Simulator Setup")
    print(f"{'='*60}")

    # Coordinate mapper
    mapper = CoordinateMapper(
        world_center=np.array(config.coordinate_mapping.world_center),
        world_scale=config.coordinate_mapping.world_scale,
        device=device
    )

    # Damage mapper
    damage_mapper = VolumetricToSurfaceDamageMapper(
        projection_method=config.damage_projection.method,
        k_neighbors=config.damage_projection.k_neighbors,
        influence_radius=config.damage_projection.influence_radius,
        damage_threshold=config.damage_projection.damage_threshold,
        use_faiss=config.damage_projection.use_faiss,
        device=device
    )

    # Visualizer
    visualizer = GaussianCrackVisualizer(
        damage_threshold=config.gaussian_splatting.get("damage_threshold", 0.3),
        device=device
    )

    # Phase field parameters (AT2 uses Gc/l0 from elasticity, only need warmup + rate limiter)
    phase_field_params = {
        "warmup_frames": config.phase_field.get("warmup_frames", 5),
        "dC_max": config.phase_field.get("dC_max", 0.02),
        # Crack-tip tracking parameters
        "crack_tip_speed": config.phase_field.get("crack_tip_speed", 1.5),
        "crack_width": config.phase_field.get("crack_width", 0.025),
        "max_total_cracks": config.phase_field.get("max_total_cracks", 5),
        # Anisotropic diffusion (for stress eigenvector computation)
        "crack_aniso_ratio": config.phase_field.get("crack_aniso_ratio", 200.0),
        # Nucleation parameters
        "nucleation_fraction": config.phase_field.get("nucleation_fraction", 0.3),
        "max_nucleation_per_frame": config.phase_field.get("max_nucleation_per_frame", 1),
        "nucleation_min_spacing": config.phase_field.get("nucleation_min_spacing", 8),
        # Fragmentation parameters
        "fragmentation_enabled": config.phase_field.get("fragmentation_enabled", False),
        "fragment_damage_threshold": config.phase_field.get("fragment_damage_threshold", 0.5),
        "min_fragment_particles": config.phase_field.get("min_fragment_particles", 50),
    }

    # Seismic loading parameters (earthquake ground motion)
    seismic_params = {}
    if hasattr(config, 'seismic'):
        seismic_enabled = config.seismic.get("enabled", False)
        # Loading type can override seismic enable/disable
        if loading_params and loading_params.get("seismic_override") is not None:
            seismic_enabled = loading_params["seismic_override"]
        seismic_params = {
            "enabled": seismic_enabled,
            "amplitude": float(config.seismic.get("amplitude", 1000.0)),
            "frequency": float(config.seismic.get("frequency", 80.0)),
            "direction": list(config.seismic.get("direction", [1.0, 0.0, 0.0])),
            "ramp_time": float(config.seismic.get("ramp_time", 0.005)),
        }

    # Create simulator
    simulator = HybridCrackSimulator(
        mpm_model=mpm_model,
        gaussians=gaussians,
        elasticity_module=elasticity,
        coord_mapper=mapper,
        damage_mapper=damage_mapper,
        visualizer=visualizer,
        surface_mask=torch.from_numpy(surface_mask).to(device),
        physics_substeps=config.rendering.physics_substeps,
        phase_field_params=phase_field_params,
        simulation_mode=config.simulation.get("mode", "crack_only"),
        seismic_params=seismic_params
    )

    return simulator


# ============================================================================
# Rendering
# ============================================================================

@torch.no_grad()
def _depth_to_normal(depth: torch.Tensor, camera) -> torch.Tensor:
    """Compute screen-space normals from depth map via finite differences.

    Works for ALL visible surfaces — external and crack interior alike.

    Args:
        depth: (1, H, W) depth map from rasterizer
        camera: MiniCam with camera intrinsics

    Returns:
        normal: (3, H, W) world-space unit normals
    """
    d = depth.squeeze(0)  # (H, W)
    H, W = d.shape

    # Finite differences (central where possible)
    dz_dx = torch.zeros_like(d)
    dz_dy = torch.zeros_like(d)
    dz_dx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / 2.0
    dz_dy[1:-1, :] = (d[2:, :] - d[:-2, :]) / 2.0

    # Pixel-to-world scale: at depth z, 1 pixel = z / focal
    fx = camera.image_width / (2.0 * np.tan(camera.FoVx / 2.0))
    fy = camera.image_height / (2.0 * np.tan(camera.FoVy / 2.0))

    # Normal = (-dz/dx / fx, -dz/dy / fy, 1), then normalize
    # This gives view-space normals
    nx = -dz_dx / (fx + 1e-8)
    ny = -dz_dy / (fy + 1e-8)
    nz = torch.ones_like(d)

    normal = torch.stack([nx, ny, nz], dim=0)  # (3, H, W)
    norm = normal.norm(dim=0, keepdim=True).clamp(min=1e-8)
    normal = normal / norm

    # Transform view-space normals to world-space using inverse view rotation
    # camera has world_view_transform (4x4)
    view_mat = camera.world_view_transform.T  # column-major to row-major
    R_view = view_mat[:3, :3]  # (3, 3) view rotation
    R_inv = R_view.T  # inverse rotation = transpose

    # Reshape for batch matmul: (3, H*W) -> (H*W, 3) -> matmul -> (3, H, W)
    n_flat = normal.reshape(3, -1).T  # (H*W, 3)
    n_world = (n_flat @ R_inv.T).T.reshape(3, H, W)

    # Zero out background (depth == 0)
    mask = (d > 0).unsqueeze(0).float()
    n_world = n_world * mask

    return n_world


def setup_camera(config: OmegaConf):
    """
    Create rendering camera using proper camera system.

    Args:
        config: Simulation configuration

    Returns:
        MiniCam object with proper intrinsics and projection
    """
    from src.renderer.camera.config import make_matrices_from_yaml

    # Build camera configuration dictionary for the proper camera system
    cam_config = config.rendering.camera
    width = config.rendering.image_width
    height = config.rendering.image_height

    # Convert orbital parameters (distance, elevation, azimuth) to lookat parameters
    elev_rad = np.radians(cam_config.elevation)
    azim_rad = np.radians(cam_config.azimuth)
    distance = cam_config.distance
    # Camera target: configurable, defaults to mesh center [0.5, 0.5, 0.5]
    target_cfg = cam_config.get('target', [0.5, 0.5, 0.5])
    target = np.array(list(target_cfg))

    # Compute eye position from orbital parameters
    x = target[0] + distance * np.cos(elev_rad) * np.cos(azim_rad)
    y = target[1] + distance * np.cos(elev_rad) * np.sin(azim_rad)
    z = target[2] + distance * np.sin(elev_rad)
    eye = [float(x), float(y), float(z)]

    # Compute intrinsic parameters from FOV
    fov_deg = cam_config.fov
    fov_rad = np.radians(fov_deg)
    # fx = width / (2 * tan(fov_x / 2))
    # For square FOV: use width as reference
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # Assuming square pixels
    cx = width / 2.0
    cy = height / 2.0

    # Build proper camera configuration
    camera_yaml = {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "znear": 0.01,
        "zfar": 100.0,
        "lookat": {
            "eye": eye,
            "target": target.tolist(),
            "up": [0.0, 0.0, 1.0]  # Z-up
        }
    }

    print(f"\n[Camera Setup]")
    print(f"  - Eye: [{eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f}]")
    print(f"  - Target: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
    print(f"  - Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"  - FOV: {fov_deg}°")

    # Use proper camera system to build matrices
    w, h, tanfovx, tanfovy, view_matrix, proj_matrix, camera_position = make_matrices_from_yaml(camera_yaml)

    # Convert to PyTorch tensors
    world_view_transform = torch.from_numpy(view_matrix).cuda()
    full_proj_transform = torch.from_numpy(proj_matrix).cuda()

    # Compute FOV angles from tan_half_fov
    fov_x = 2.0 * np.arctan(tanfovx)
    fov_y = 2.0 * np.arctan(tanfovy)

    # Camera position as tensor
    camera_center_tensor = torch.from_numpy(camera_position).cuda()

    # Use MiniCam (simpler, no image data required)
    camera = MiniCam(
        width=w,
        height=h,
        fovy=fov_y,
        fovx=fov_x,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform
    )

    return camera


# ============================================================================
# Simulation Loop
# ============================================================================

def run_simulation(config: OmegaConf, simulator, camera, args):
    """
    Main simulation loop

    Args:
        config: Simulation configuration
        simulator: Hybrid simulator
        camera: Rendering camera
        args: Command-line arguments
    """
    print(f"\n{'='*60}")
    print(f"Step 6: Running Simulation")
    print(f"{'='*60}")

    # Setup output directories
    output_dir = Path(config.output.video_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = Path(config.output.frame_dir)
    if config.simulation.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config.output.checkpoint_dir)
    if config.simulation.save_checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup rendering
    device = torch.device(config.device.type)
    bg_color = torch.tensor(config.rendering.background_color, device=device)

    # Create pipeline parameters for Gaussian Splatting renderer
    pipe = type('obj', (object,), {
        'convert_SHs_python': False,  # Let rasterizer handle SH to RGB conversion
        'compute_cov3D_python': False,
        'debug': True,  # Enable debug mode to see C++ errors
        'antialiasing': False
    })()

    # Apply impact and initialize crack energy field
    if config.external_force.enabled:
        print(f"\n[Impact] Finding front-facing surface particle for crack initiation...")

        # Test render to find visible Gaussians
        test_rendering = render(camera, simulator.gaussians, pipe, bg_color)
        visible_radii = test_rendering['radii']
        visible_mask = visible_radii > 0
        visible_indices = torch.where(visible_mask)[0]

        print(f"  - Total Gaussians: {len(visible_radii)}")
        print(f"  - Visible from camera: {len(visible_indices)}")

        if len(visible_indices) > 0:
            # Find a front-facing particle as impact center
            visible_positions = simulator.gaussians.get_xyz[visible_indices]
            visible_normals = simulator.gaussians.get_normal[visible_indices]

            cam_pos = camera.camera_center
            view_dirs = cam_pos - visible_positions
            view_dirs = view_dirs / (view_dirs.norm(dim=1, keepdim=True) + 1e-8)
            facing_scores = (visible_normals * view_dirs).sum(dim=1)

            front_facing = facing_scores > 0.3
            if front_facing.sum() > 0:
                front_facing_indices = visible_indices[front_facing]
                center_idx = front_facing_indices[torch.randint(0, len(front_facing_indices), (1,))].item()
                print(f"  - Sampled from {len(front_facing_indices)} front-facing particles")
            else:
                center_idx = visible_indices[facing_scores.argmax()].item()
                print(f"  - Picked best facing particle")

            center_pos = simulator.gaussians.get_xyz[center_idx:center_idx+1]
            print(f"  - Impact center index: {center_idx}")
            print(f"  - Impact position (world): {center_pos[0].detach().cpu().numpy()}")

            # Convert to MPM space
            center_mpm = simulator.mapper.world_to_mpm(center_pos)

            # Initialize based on simulation mode
            sim_mode = config.simulation.get("mode", "crack_only")
            if sim_mode == "crack_only":
                simulator.initialize_crack_energy(
                    impact_center_mpm=center_mpm[0],
                    impact_energy=config.external_force.magnitude,
                    impact_radius=config.external_force.radius
                )
            else:
                # Impact direction: camera → object (inward)
                impact_dir = center_pos[0] - cam_pos
                impact_dir = impact_dir / (impact_dir.norm() + 1e-8)
                # Convert direction to MPM space (same rotation, just scale)
                impact_dir_mpm = simulator.mapper.world_to_mpm(
                    center_pos + impact_dir.unsqueeze(0) * 0.01
                ) - center_mpm
                impact_dir_mpm = impact_dir_mpm[0] / (impact_dir_mpm[0].norm() + 1e-8)

                simulator.initialize_deformation_impact(
                    impact_center_mpm=center_mpm[0],
                    impact_energy=config.external_force.magnitude,
                    impact_radius=config.external_force.radius,
                    impact_direction=impact_dir_mpm
                )
        else:
            print(f"  - WARNING: No visible Gaussians found! Skipping impact.")

    # Apply pre-notch if configured
    if hasattr(config, 'pre_notch') and config.pre_notch.get('enabled', False):
        notches = config.pre_notch.get('notches', [])
        if notches:
            print(f"\n[Pre-notch] Seeding {len(notches)} notch(es) in body...")
            notch_list = []
            for n in notches:
                notch_list.append({
                    'start': list(n['start']),
                    'end': list(n['end']),
                    'damage': n.get('damage', 0.9)
                })
            simulator.apply_pre_notch(notch_list)

    # Statistics logging
    stats_log = []
    log_file = Path(config.output.statistics_log)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Pass camera position to simulator for back-face normal flipping
    simulator._camera_pos = camera.camera_center

    # Simulation loop
    render_interval = 1  # Render every frame for debugging
    print(f"\nSimulating {config.rendering.total_frames} frames (render every {render_interval})...")
    print(f"{'='*60}")

    start_time = time.time()

    # Enable diagnostic data saving at key frames
    simulator._save_diagnostics = True
    simulator._output_dir = config.output.get('frame_dir', 'output/frames').replace('/frames', '')
    # Diagnostic at rendering frames: pre-impact, impact, post-impact, settled
    simulator._diag_frames = {0, 10, 20, 30, 40, 44, 45, 46, 47, 48, 50, 55, 60, 70, 80, 90, 99}

    for frame in range(config.rendering.total_frames):
        # Pass rendering frame number for diagnostics (no reset overlap)
        simulator._render_frame = frame
        # Physics + Gaussian update (always runs)
        simulator.step_rendering()

        # Determine if we should render this frame
        should_render = (frame % render_interval == 0) or (frame == config.rendering.total_frames - 1)

        # Skip rendering for intermediate frames (physics only)
        if not should_render:
            # Still collect stats and log periodically
            if frame % config.output.log_interval == 0:
                stats = simulator.get_statistics()
                stats["frame"] = frame
                stats_log.append(stats)
                elapsed = time.time() - start_time
                eta = elapsed / (frame + 1) * (config.rendering.total_frames - frame - 1)
                print(f"Frame {frame:04d}/{config.rendering.total_frames}: "
                      f"c_max={stats['c_max']:.4f}, "
                      f"c_mean={stats['c_mean']:.4f}, "
                      f"cracked={stats['n_cracked']}/{stats['n_particles']}, "
                      f"fps={(frame+1)/elapsed:.1f}, "
                      f"ETA={eta/60:.1f}min", flush=True)
            continue

        print(f"\n--- Rendering frame {frame} ---")

        # Debug: Print Gaussian stats on first frame
        if frame == 0:
            print(f"\n[Debug] Gaussian Stats:")
            print(f"  - Positions: {simulator.gaussians.get_xyz.shape}, range [{simulator.gaussians.get_xyz.min().item():.3f}, {simulator.gaussians.get_xyz.max().item():.3f}]")
            print(f"  - Opacity: {simulator.gaussians.get_opacity.shape}, mean {simulator.gaussians.get_opacity.mean().item():.3f}")
            print(f"  - Scales: {simulator.gaussians.get_scaling.shape}, mean {simulator.gaussians.get_scaling.mean().item():.3f}")
            print(f"  - Features DC: {simulator.gaussians.get_features_dc.shape}, mean {simulator.gaussians.get_features_dc.mean().item():.3f}")
            print(f"  - Features Rest: {simulator.gaussians.get_features_rest.shape}, mean {simulator.gaussians.get_features_rest.mean().item():.3f}")
            print(f"\n[Debug] Camera:")
            print(f"  - Position: {camera.camera_center}")
            print(f"  - FoV: {np.degrees(camera.FoVx):.1f}° x {np.degrees(camera.FoVy):.1f}°")
            print(f"  - Image: {camera.image_width}x{camera.image_height}")

            # Calculate mesh bounds and screen projection
            gauss_pos = simulator.gaussians.get_xyz.detach().cpu().numpy()
            mesh_min = gauss_pos.min(axis=0)
            mesh_max = gauss_pos.max(axis=0)
            mesh_center = (mesh_min + mesh_max) / 2
            mesh_extent = mesh_max - mesh_min

            print(f"\n[Debug] Mesh Bounds:")
            print(f"  - Min: [{mesh_min[0]:.3f}, {mesh_min[1]:.3f}, {mesh_min[2]:.3f}]")
            print(f"  - Max: [{mesh_max[0]:.3f}, {mesh_max[1]:.3f}, {mesh_max[2]:.3f}]")
            print(f"  - Center: [{mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f}]")
            print(f"  - Extent: [{mesh_extent[0]:.3f}, {mesh_extent[1]:.3f}, {mesh_extent[2]:.3f}]")

            # Calculate distance from camera to mesh center
            cam_pos = camera.camera_center.detach().cpu().numpy()
            dist_to_mesh = np.linalg.norm(cam_pos - mesh_center)
            print(f"  - Distance from camera to mesh center: {dist_to_mesh:.3f}")

            # Calculate angular size of mesh
            max_extent = mesh_extent.max()
            angular_size_rad = 2 * np.arctan(max_extent / (2 * dist_to_mesh))
            angular_size_deg = np.degrees(angular_size_rad)
            fov_deg = np.degrees(camera.FoVx)
            screen_coverage = (angular_size_deg / fov_deg) * 100

            print(f"  - Mesh max extent: {max_extent:.3f} units")
            print(f"  - Angular size: {angular_size_deg:.1f}° (covers {screen_coverage:.1f}% of FoV)")
            print(f"  - Recommended distance for 80% coverage: {max_extent / (2 * np.tan(np.radians(fov_deg * 0.8 / 2))):.3f}")

            # Debug: Check view space z-coordinates to understand frustum culling
            print(f"\n[Debug] View Space Coordinates (checking p_view.z for frustum culling):")
            viewmatrix = camera.world_view_transform.detach().cpu().numpy()
            gauss_pos = simulator.gaussians.get_xyz.detach().cpu().numpy()
            print(f"  - Viewmatrix shape: {viewmatrix.shape}")
            print(f"  - Viewmatrix:\n{viewmatrix}")

            # Check 10 sample Gaussians
            sample_idx = np.linspace(0, len(gauss_pos)-1, min(10, len(gauss_pos)), dtype=int)
            print(f"\n  - Sample Gaussians (checking if p_view.z <= 0.2 causes culling):")
            for i in sample_idx:
                pos_hom = np.append(gauss_pos[i], 1.0)  # [x, y, z, 1]
                p_view_hom = viewmatrix @ pos_hom  # 4x4 @ 4 = 4
                p_view = p_view_hom[:3]  # [x, y, z] in view space
                culled = "CULLED" if p_view[2] <= 0.2 else "visible"
                print(f"    [{i:5d}] world={gauss_pos[i]}, view.z={p_view[2]:8.3f} ({culled})")

            print(f"\n  - Culling threshold: p_view.z <= 0.2 (see auxiliary.h line 155)")

        # Render (colors already modified by visualizer in step_rendering)
        rendering = render(
            camera,
            simulator.gaussians,
            pipe,
            bg_color
        )
        image = rendering["render"]
        depth = rendering["depth"]

        # Depth-to-normal post-process shading
        # Works for ALL visible surfaces (external + crack interior)
        object_mask = (depth > 0).float()  # (1, H, W)
        if object_mask.sum() > 0:
            normal_from_depth = _depth_to_normal(depth, camera)

            cam_pos = camera.camera_center
            scene_center = torch.tensor([0.5, 0.5, 0.5], device='cuda')
            light_dir = scene_center - cam_pos
            light_dir = light_dir / (light_dir.norm() + 1e-8)

            light_dir_reshaped = light_dir.view(3, 1, 1)
            diffuse = (normal_from_depth * light_dir_reshaped).sum(
                dim=0, keepdim=True).clamp(0.0, 1.0)

            ambient = 0.85
            lit_intensity = ambient + (1.0 - ambient) * diffuse

            image_lit = image * lit_intensity
            bg_intensity = 0.92
            image_bg = image * bg_intensity

            image = image_bg * (1.0 - object_mask) + image_lit * object_mask
            image = torch.clamp(image, 0.0, 1.0)

        if frame == 0:
            radii = rendering['radii']
            n_rendered = (radii > 0).sum().item()
            n_total = radii.shape[0]
            render_percent = 100.0 * n_rendered / n_total

            print(f"\n[Debug] Rendering Results:")
            print(f"  - Image shape: {image.shape}")
            print(f"  - Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
            print(f"  - Depth shape: {depth.shape}")
            print(f"  - Depth range: [{depth.min().item():.3f}, {depth.max().item():.3f}]")
            print(f"  - Depth non-zero pixels: {(depth > 0).sum().item()} / {depth.numel()}")
            if object_mask.sum() > 0:
                n = normal_from_depth
                print(f"  - Normal shape: {n.shape}")
                print(f"  - Normal range: [{n.min().item():.3f}, {n.max().item():.3f}]")
                print(f"  - Normal non-zero pixels: {(n.abs() > 0.01).sum().item()} / {n.numel()}")
                print(f"  - Normal mean (XYZ): [{n[0].mean().item():.3f}, {n[1].mean().item():.3f}, {n[2].mean().item():.3f}]")
            print(f"\n[Gaussian Rendering Statistics]")
            print(f"  - Total Gaussians: {n_total}")
            print(f"  - Rendered (radii > 0): {n_rendered} ({render_percent:.1f}%)")
            print(f"  - Culled: {n_total - n_rendered} ({100 - render_percent:.1f}%)")
            if n_rendered > 0:
                print(f"  - Radii range (rendered only): {radii[radii > 0].min().item():.1f} to {radii[radii > 0].max().item():.1f} pixels")
                # Show which Gaussians are rendering
                rendered_indices = torch.where(radii > 0)[0]
                print(f"  - Rendered Gaussian indices: {rendered_indices.cpu().numpy()[:10]}")  # Show first 10
                gauss_pos = simulator.gaussians.get_xyz.detach().cpu().numpy()
                for idx in rendered_indices[:5]:  # Show first 5
                    pos = gauss_pos[idx]
                    print(f"    - Gaussian {idx}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], radius={radii[idx].item():.1f}px")

            # Analyze why Gaussians are culled
            print(f"\n[Culling Analysis]")
            gauss_scales_linear = simulator.gaussians.get_scaling.detach().cpu().numpy()  # Already linear (exp applied)
            gauss_scales_log = simulator.gaussians._scaling.detach().cpu().numpy()  # Raw log values
            gauss_opacity = simulator.gaussians.get_opacity.detach().cpu().numpy()
            print(f"  - Scale (linear, world units): min={gauss_scales_linear.min():.6f}, max={gauss_scales_linear.max():.6f}, mean={gauss_scales_linear.mean():.6f}")
            print(f"  - Scale (log space): min={gauss_scales_log.min():.3f}, max={gauss_scales_log.max():.3f}, mean={gauss_scales_log.mean():.3f}")
            print(f"  - Opacity (sigmoid): min={gauss_opacity.min():.3f}, max={gauss_opacity.max():.3f}, mean={gauss_opacity.mean():.3f}")

            # Check if low rendering is due to scale or culling
            if render_percent < 50:
                print(f"\n  [WARNING] Less than 50% of Gaussians are rendering!")
                print(f"  Possible causes:")
                print(f"    1. Scales too small -> increase target_pixel_coverage or camera distance")
                print(f"    2. Frustum culling -> adjust camera position/orientation")
                print(f"    3. Opacity too low -> increase initial opacity")
            elif render_percent > 95:
                print(f"\n  [OK] Good: >95% of Gaussians are rendering")

        # Save frame if requested
        if config.simulation.save_frames:
            save_image(image, frame_dir / f"frame_{frame:04d}.png")

        # Save checkpoint if requested
        if config.simulation.save_checkpoint:
            if frame % config.simulation.checkpoint_interval == 0 and frame > 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{frame:04d}.pth"
                simulator.save_state(str(checkpoint_path))

        # Collect statistics
        stats = simulator.get_statistics()
        stats["frame"] = frame
        stats_log.append(stats)

        # Log progress
        if frame % config.output.log_interval == 0 or frame == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (frame + 1) * (config.rendering.total_frames - frame - 1)

            print(f"Frame {frame:04d}/{config.rendering.total_frames}: "
                  f"c_max={stats['c_max']:.4f}, "
                  f"c_mean={stats['c_mean']:.4f}, "
                  f"cracked={stats['n_cracked']}/{stats['n_particles']}, "
                  f"fps={stats['fps']:.1f}, "
                  f"ETA={eta/60:.1f}min")

    total_time = time.time() - start_time
    print(f"{'='*60}")
    print(f"Simulation complete! Total time: {total_time/60:.1f} min")

    # Save statistics to CSV
    if stats_log:
        with open(log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats_log[0].keys())
            writer.writeheader()
            writer.writerows(stats_log)
        print(f"[Output] Statistics saved: {log_file}")

    # Generate video
    if not args.no_video and config.simulation.save_frames:
        print(f"\n[Output] Generating video...")
        create_video(frame_dir, config.output.video_path, config.rendering.fps)


def create_video(frame_dir: Path, output_path: str, fps: int):
    """
    Create H.264 mp4 video from saved frames (VS Code compatible).

    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
    """
    import subprocess

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        print(f"[Warning] No frames found in {frame_dir}")
        return

    output_path = str(output_path)

    # Try ffmpeg H.264 (VS Code compatible)
    try:
        frame_pattern = str(frame_dir / "frame_%04d.png")
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-movflags", "+faststart",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"[Output] Video created (H.264): {output_path}")
            return
        else:
            print(f"[Warning] ffmpeg failed: {result.stderr[-300:]}")
    except Exception as e:
        print(f"[Warning] ffmpeg unavailable: {e}")

    # Fallback: OpenCV mp4v
    try:
        import cv2
        first_frame = cv2.imread(str(frames[0]))
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for i, frame_path in enumerate(frames):
            out.write(cv2.imread(str(frame_path)))
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(frames)} frames...")
        out.release()
        print(f"[Output] Video created (mp4v fallback): {output_path}")
    except Exception as e:
        print(f"[Error] Video creation failed: {e}")


# ============================================================================
# Loading Conditions
# ============================================================================

def setup_loading(config: OmegaConf, mpm_model, device: torch.device):
    """
    Configure loading conditions based on config.loading.type.

    Sets up gravity, ground plane, compression plates, etc. on the MPM model.
    Returns a dict that may override seismic enable/disable.

    Backward compatible: if no loading section exists, infers from legacy flags.
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
        # Gravity scaling for MPM normalized space [0,1]³ (Z-down gravity):
        # Total sim time = frames × substeps × dt (e.g., 200×10×5e-5 = 0.1s)
        # To fall ~0.4 units (z=0.7 to ground z=0.05) in sim time:
        #   d = 0.5 * g_eff * t² → g_eff = 2d/t²
        total_frames = config.rendering.get('total_frames', 100)
        substeps = config.rendering.get('physics_substeps', 10)
        dt = float(config.mpm.dt)
        total_time = total_frames * substeps * dt
        target_fall = 0.35  # units to fall in normalized space

        # Compute needed gravity: d = 0.5 * g * t²  → g = 2d/t²
        g_needed = 2.0 * target_fall / (total_time ** 2) if total_time > 0 else 500.0
        g_needed = max(g_needed, 200.0)  # minimum effective gravity
        g_needed = min(g_needed, 5000.0)  # safety cap

        # Gravity in -Z direction (Z-up coordinate system, matching camera)
        mpm_model.gravity = torch.tensor([0.0, 0.0, -g_needed], device=device)
        print(f"  - Effective gravity: [0, 0, {-g_needed:.1f}]  (sim_time={total_time:.4f}s)")

        # Reduce damping for gravity_drop: preserve kinetic energy for impact
        # Original damping (0.98) = 2% energy loss per substep → very overdamped
        # Use 0.9995 for near-free-fall until impact
        mpm_model.damping = 0.9995
        print(f"  - Damping: {mpm_model.damping} (near-free-fall for impact)")

        # Auto-enable ground plane if not already configured (Z-normal)
        if not (hasattr(config, 'ground_plane') and config.ground_plane.get('enabled', False)):
            from src.mpm_core.set_boundary_conditions import add_surface_collider
            add_surface_collider(
                model=mpm_model,
                point=[0.5, 0.5, 0.1],
                normal=[0.0, 0.0, 1.0],
                surface="slip",
                friction=0.5,
            )
            print(f"  - Auto-enabled ground plane at z=0.1 (slip)")

        # Warmup: frame_count resets at impact, so use a short warmup (3-5 frames)
        # to let initial impact stress build before nucleating cracks
        warmup_val = config.phase_field.get('warmup_frames', 3)
        warmup_val = min(warmup_val, 10)  # cap at 10 for gravity_drop
        OmegaConf.update(config, "phase_field.warmup_frames", warmup_val)
        print(f"  - warmup_frames={warmup_val} (resets at impact)")

        # Disable seismic for gravity_drop
        result["seismic_override"] = False
        print(f"  - Seismic: OFF (gravity_drop mode)")

    elif loading_type == "compression":
        comp = config.loading.get('compression', {})
        axis = int(comp.get('axis', 0))
        speed = float(comp.get('speed', 0.5))
        start_pos = float(comp.get('start_position', 0.1))
        end_pos = float(comp.get('end_position', 0.9))

        from src.mpm_core.set_boundary_conditions import set_velocity_on_cuboid

        # Lower plate: moves in +axis direction
        lower_point = [0.5, 0.5, 0.5]
        lower_point[axis] = start_pos
        lower_size = [0.5, 0.5, 0.5]
        lower_size[axis] = 0.02
        lower_vel = [0.0, 0.0, 0.0]
        lower_vel[axis] = speed

        set_velocity_on_cuboid(
            model=mpm_model,
            point=lower_point,
            size=lower_size,
            velocity=lower_vel,
        )

        # Upper plate: moves in -axis direction
        upper_point = [0.5, 0.5, 0.5]
        upper_point[axis] = end_pos
        upper_size = [0.5, 0.5, 0.5]
        upper_size[axis] = 0.02
        upper_vel = [0.0, 0.0, 0.0]
        upper_vel[axis] = -speed

        set_velocity_on_cuboid(
            model=mpm_model,
            point=upper_point,
            size=upper_size,
            velocity=upper_vel,
        )

        result["seismic_override"] = False
        axis_names = ["X", "Y", "Z"]
        print(f"  - Compression axis: {axis_names[axis]}")
        print(f"  - Plate speed: {speed}")
        print(f"  - Plates: [{start_pos}] --> <-- [{end_pos}]")
        print(f"  - Seismic: OFF (compression mode)")

    elif loading_type == "point_impact":
        result["seismic_override"] = False
        print(f"  - Using external_force config for point impact")
        print(f"  - Seismic: OFF (point_impact mode)")

    elif loading_type == "seismic":
        # Default: use existing seismic system
        result["seismic_override"] = None  # don't override
        print(f"  - Using seismic config (existing behavior)")

    else:
        raise ValueError(
            f"Unknown loading type: '{loading_type}'. "
            f"Options: seismic, gravity_drop, point_impact, compression"
        )

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()

    # Deterministic seeding for reproducibility
    seed = getattr(args, 'seed', 42) or 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load and apply configuration
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Apply material preset (if specified)
    from src.core.material_presets import resolve_material_preset, validate_l0
    config = resolve_material_preset(config)
    config = validate_l0(config)

    # Setup device
    device_type = config.device.type
    if device_type == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device_type = "cpu"

    device = torch.device(device_type)
    if device_type == "cuda" and hasattr(config.device, 'gpu_id'):
        torch.cuda.set_device(config.device.gpu_id)

    print(f"\n{'='*60}")
    print(f"MPM + Gaussian Splatting Crack Simulation")
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    try:
        # Pipeline execution
        volume_pcd, surface_pcd, surface_mask, mesh_meta = setup_mesh(config)
        mpm_model = setup_mpm(config, volume_pcd, device)
        loading_params = setup_loading(config, mpm_model, device)
        elasticity = setup_elasticity(config, device)
        gaussians = setup_gaussians(config, surface_pcd, device, mesh_meta=mesh_meta)

        simulator = setup_simulator(
            config, mpm_model, gaussians, elasticity,
            surface_mask, device, loading_params=loading_params
        )

        # Pass ALL particle normals (surface + volume) for dynamic lighting
        # Surface: mesh normals, Volume: radial from centroid
        if volume_pcd.normals is not None:
            all_normals = torch.from_numpy(
                np.asarray(volume_pcd.normals)).float().to(device)
            simulator.visualizer.set_initial_normals(all_normals)

        # Initialize simulation
        simulator.initialize(torch.from_numpy(volume_pcd.points).float().to(device))

        # Enable gravity drop mode if loading type is gravity_drop
        # Z-up coordinate system: gravity in -Z, ground plane at z=ground_z
        if loading_params.get("type") == "gravity_drop":
            ground_z = 0.1  # default
            if hasattr(config, 'ground_plane') and config.ground_plane.get('enabled', False):
                ground_z = float(config.ground_plane.get('point', [0.5, 0.5, 0.1])[2])

            # Scale down and reposition object for drop:
            # Object fills [0.025, 0.975] → need to shrink to ~1/3 and raise above ground
            drop_scale = config.loading.get('drop_scale', 0.33)
            drop_center_z = config.loading.get('drop_center_z', 0.7)

            # Scale all MPM positions around center [0.5, 0.5, 0.5]
            center = torch.tensor([0.5, 0.5, 0.5], device=device)
            simulator.x_mpm = center + (simulator.x_mpm - center) * drop_scale
            # Shift z so object center is at drop_center_z
            z_current_center = (simulator.x_mpm[:, 2].min().item() + simulator.x_mpm[:, 2].max().item()) / 2
            z_shift = drop_center_z - z_current_center
            simulator.x_mpm[:, 2] += z_shift

            # Apply drop rotation (Euler XYZ degrees) around object center
            drop_rotation = config.loading.get('drop_rotation', None)
            if drop_rotation is not None:
                rot_deg = [float(r) for r in drop_rotation]
                rot_rad = [np.radians(r) for r in rot_deg]
                # Build rotation matrix (XYZ Euler)
                cx, sx = np.cos(rot_rad[0]), np.sin(rot_rad[0])
                cy, sy = np.cos(rot_rad[1]), np.sin(rot_rad[1])
                cz, sz = np.cos(rot_rad[2]), np.sin(rot_rad[2])
                Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
                Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
                Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
                R_mat = torch.tensor(Rz @ Ry @ Rx, dtype=torch.float32, device=device)

                obj_center = simulator.x_mpm.mean(dim=0)
                simulator.x_mpm = obj_center + (simulator.x_mpm - obj_center) @ R_mat.T
                print(f"  - Drop rotation: {rot_deg} deg")

            # Adjust Gaussian splat scales for the object shrinkage
            if config.gaussian_splatting.get("pretrained_ply", None) is not None:
                import math as _math
                simulator.gaussians._scaling.data += _math.log(drop_scale)
                print(f"  - Pretrained splat scales adjusted by log({drop_scale})="
                      f"{_math.log(drop_scale):.4f}")

                # Direct PLY mode: rescale PLY initial positions to match MPM
                if getattr(simulator, '_ply_direct', False):
                    ply_center = torch.tensor([0.5, 0.5, 0.5], device=device)
                    simulator._ply_init_xyz = (
                        ply_center + (simulator._ply_init_xyz - ply_center) * drop_scale)
                    simulator._ply_init_xyz[:, 2] += z_shift
                    # Apply same rotation to PLY positions and quaternions
                    if drop_rotation is not None:
                        ply_obj_center = simulator._ply_init_xyz.mean(dim=0)
                        simulator._ply_init_xyz = (
                            ply_obj_center + (simulator._ply_init_xyz - ply_obj_center) @ R_mat.T)
                        # Rotate Gaussian quaternions: q_new = q_rot * q_old
                        # Convert rotation matrix to quaternion
                        R = R_mat.cpu().numpy()
                        tr = R[0,0] + R[1,1] + R[2,2]
                        if tr > 0:
                            s = np.sqrt(tr + 1.0) * 2
                            qw = 0.25 * s
                            qx = (R[2,1] - R[1,2]) / s
                            qy = (R[0,2] - R[2,0]) / s
                            qz = (R[1,0] - R[0,1]) / s
                        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                            qw = (R[2,1] - R[1,2]) / s
                            qx = 0.25 * s
                            qy = (R[0,1] + R[1,0]) / s
                            qz = (R[0,2] + R[2,0]) / s
                        elif R[1,1] > R[2,2]:
                            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                            qw = (R[0,2] - R[2,0]) / s
                            qx = (R[0,1] + R[1,0]) / s
                            qy = 0.25 * s
                            qz = (R[1,2] + R[2,1]) / s
                        else:
                            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                            qw = (R[1,0] - R[0,1]) / s
                            qx = (R[0,2] + R[2,0]) / s
                            qy = (R[1,2] + R[2,1]) / s
                            qz = 0.25 * s
                        q_rot = torch.tensor([qw, qx, qy, qz],
                                             dtype=torch.float32, device=device)
                        # Hamilton product: q_rot * q_old for each Gaussian
                        q_old = simulator.gaussians._rotation.data  # (K, 4) wxyz
                        w1, x1, y1, z1 = q_rot[0], q_rot[1], q_rot[2], q_rot[3]
                        w2, x2, y2, z2 = q_old[:,0], q_old[:,1], q_old[:,2], q_old[:,3]
                        q_new = torch.stack([
                            w1*w2 - x1*x2 - y1*y2 - z1*z2,
                            w1*x2 + x1*w2 + y1*z2 - z1*y2,
                            w1*y2 - x1*z2 + y1*w2 + z1*x2,
                            w1*z2 + x1*y2 - y1*x2 + z1*w2,
                        ], dim=-1)
                        simulator.gaussians._rotation.data = q_new
                    # Update surface initial positions
                    simulator._surf_init_world = simulator.mapper.mpm_to_world(
                        simulator.x_mpm[simulator.surface_mask])

            z_min = simulator.x_mpm[:, 2].min().item()
            z_max = simulator.x_mpm[:, 2].max().item()
            print(f"[GravityDrop] Rescaled object: scale={drop_scale}, "
                  f"center_z={drop_center_z}")
            print(f"  - z range: [{z_min:.4f}, {z_max:.4f}]")
            print(f"  - Fall distance to ground: {z_min - ground_z:.4f}")

            simulator.enable_gravity_drop(ground_z=ground_z)

        elif loading_params.get("type") == "seismic":
            obj_scale = config.loading.get('object_scale', None)
            if obj_scale is not None:
                obj_scale = float(obj_scale)
                obj_center = list(config.loading.get('object_center', [0.5, 0.5, 0.35]))
                # Scale around unit-cube center [0.5, 0.5, 0.5], then translate
                cube_center = torch.tensor([0.5, 0.5, 0.5], device=device,
                                           dtype=simulator.x_mpm.dtype)
                simulator.x_mpm = cube_center + (simulator.x_mpm - cube_center) * obj_scale
                # Shift so object center-of-mass lands on obj_center
                current_center = simulator.x_mpm.mean(dim=0)
                target_center = torch.tensor(obj_center, device=device,
                                             dtype=simulator.x_mpm.dtype)
                simulator.x_mpm += (target_center - current_center)
                z_min = simulator.x_mpm[:, 2].min().item()
                z_max = simulator.x_mpm[:, 2].max().item()
                print(f"[Seismic] Rescaled object: scale={obj_scale}, center={obj_center}")
                print(f"  - z range: [{z_min:.4f}, {z_max:.4f}]")
                print(f"  - Gap to ground: {z_min - float(config.ground_plane.get('point', [0,0,0.05])[2]):.4f}")

        # Setup camera
        camera = setup_camera(config)

        # Run simulation
        run_simulation(config, simulator, camera, args)

        print(f"\n{'='*60}")
        print(f"Simulation Complete!")
        print(f"{'='*60}")
        print(f"Output: {config.output.video_path}")
        print(f"Frames: {config.output.frame_dir}")
        print(f"Statistics: {config.output.statistics_log}")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n[Info] Simulation interrupted by user")
    except Exception as e:
        print(f"\n[Error] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
