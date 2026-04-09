"""
Static render test: Load pretrained PLY Gaussians and render a single frame.
No MPM physics, no drop, no rotation — pure 3DGS rendering quality check.
"""

import torch
import numpy as np
import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gaussian-splatting"))

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from src.renderer.camera.config import make_matrices_from_yaml
from torchvision.utils import save_image


def build_camera(distance=0.85, elevation=25.0, azimuth=30.0, fov=50.0,
                 target=(0.5, 0.5, 0.5), width=1920, height=1920):
    """Build a MiniCam using the same pipeline as run.py setup_camera."""
    elev_rad = np.radians(elevation)
    azim_rad = np.radians(azimuth)
    fov_rad = np.radians(fov)

    x = target[0] + distance * np.cos(elev_rad) * np.cos(azim_rad)
    y = target[1] + distance * np.cos(elev_rad) * np.sin(azim_rad)
    z = target[2] + distance * np.sin(elev_rad)
    eye = [float(x), float(y), float(z)]

    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    camera_yaml = {
        "width": width, "height": height,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "znear": 0.01, "zfar": 100.0,
        "lookat": {
            "eye": eye,
            "target": list(target),
            "up": [0.0, 0.0, 1.0]
        }
    }

    w, h, tanfovx, tanfovy, view_matrix, proj_matrix, camera_position = make_matrices_from_yaml(camera_yaml)
    world_view_transform = torch.from_numpy(view_matrix).cuda()
    full_proj_transform = torch.from_numpy(proj_matrix).cuda()
    fov_x = 2.0 * np.arctan(tanfovx)
    fov_y = 2.0 * np.arctan(tanfovy)

    cam = MiniCam(w, h, fov_y, fov_x, 0.01, 100.0, world_view_transform, full_proj_transform)
    print(f"Camera: eye={eye}, center={cam.camera_center.cpu().numpy()}")
    return cam


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    ply_path = "assets/trained/lego/point_cloud.ply"
    sh_degree = 3
    out_dir = Path("output/static_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load PLY
    from src.preprocessing.ply_loader import PretrainedPlyLoader
    loader = PretrainedPlyLoader(ply_path, sh_degree=sh_degree)
    ply_data = loader.load_raw_ply()
    ply_xyz_norm, scale_factor = loader.normalize_positions(ply_data['xyz'])

    M = ply_xyz_norm.shape[0]
    print(f"\n=== Direct PLY rendering: {M} Gaussians ===")

    gaussians = GaussianModel(sh_degree=sh_degree)

    gaussians._xyz = torch.nn.Parameter(
        torch.tensor(ply_xyz_norm, dtype=torch.float, device=device))
    gaussians._features_dc = torch.nn.Parameter(
        torch.tensor(ply_data['features_dc'], dtype=torch.float, device=device
        ).transpose(1, 2).contiguous())
    gaussians._features_rest = torch.nn.Parameter(
        torch.tensor(ply_data['features_rest'], dtype=torch.float, device=device
        ).transpose(1, 2).contiguous())
    gaussians._opacity = torch.nn.Parameter(
        torch.tensor(ply_data['opacity'], dtype=torch.float, device=device))

    scale_adj = math.log(scale_factor)
    scales = ply_data['scaling'] + scale_adj
    gaussians._scaling = torch.nn.Parameter(
        torch.tensor(scales, dtype=torch.float, device=device))
    gaussians._rotation = torch.nn.Parameter(
        torch.tensor(ply_data['rotation'], dtype=torch.float, device=device))

    gaussians.active_sh_degree = sh_degree
    gaussians.max_radii2D = torch.zeros(M, device=device)
    gaussians.exposure_mapping = {}
    gaussians.pretrained_exposures = None

    s_lin = np.exp(scales)
    print(f"Scale (log): min={scales.min():.3f}, max={scales.max():.3f}, median={np.median(scales):.3f}")
    print(f"Scale (linear): min={s_lin.min():.6f}, max={s_lin.max():.6f}, mean={s_lin.mean():.6f}")

    center = ply_xyz_norm.mean(axis=0)
    print(f"PLY center: {center}")

    class PipeArgs:
        debug = False
        convert_SHs_python = False
        compute_cov3D_python = False
        antialiasing = False
    pipe = PipeArgs()
    bg = torch.ones(3, device=device)

    # Direct PLY rendering at multiple distances
    for dist in [1.5, 2.0, 2.5]:
        cam = build_camera(distance=dist, elevation=25.0, azimuth=30.0, fov=50.0,
                           target=tuple(center), width=1920, height=1920)
        result = render(cam, gaussians, pipe, bg)
        image = result["render"]
        radii = result["radii"]
        visible = (radii > 0).sum().item()
        if visible > 0:
            print(f"[dist={dist}] {visible}/{M} visible, radii {radii[radii>0].min().item()}-{radii[radii>0].max().item()}px")
        else:
            print(f"[dist={dist}] 0/{M} visible!")
        save_image(image, str(out_dir / f"direct_ply_d{dist:.1f}.png"))

    # Surface-matched rendering
    print(f"\n=== Surface-matched rendering ===")
    from src.preprocessing.mesh_converter import MeshToPointCloudConverter
    converter = MeshToPointCloudConverter(
        mesh_path="assets/meshes/lego.obj",
        target_particle_count=300000,
        surface_sample_ratio=0.7,
        normalize_to_unit_cube=True
    )
    volume_pcd, surface_pcd, surface_mask = converter.convert()

    match_indices, distances = loader.match_to_surface_particles(
        ply_xyz_norm, np.asarray(surface_pcd.points))

    gaussians2 = GaussianModel(sh_degree=sh_degree)
    loader.create_matched_gaussians(
        gaussians2, surface_pcd, ply_data, match_indices,
        scale_factor, scale_multiplier=1.0)

    N = surface_pcd.points.shape[0]
    gaussians2.max_radii2D = torch.zeros(N, device=device)
    gaussians2.exposure_mapping = {}
    gaussians2.pretrained_exposures = None

    center2 = np.asarray(surface_pcd.points).mean(axis=0)
    for dist in [1.5, 2.0]:
        cam2 = build_camera(distance=dist, elevation=25.0, azimuth=30.0, fov=50.0,
                            target=tuple(center2), width=1920, height=1920)
        result2 = render(cam2, gaussians2, pipe, bg)
        image2 = result2["render"]
        radii2 = result2["radii"]
        visible2 = (radii2 > 0).sum().item()
        if visible2 > 0:
            print(f"[matched d={dist}] {visible2}/{N} visible, radii {radii2[radii2>0].min().item()}-{radii2[radii2>0].max().item()}px")
        else:
            print(f"[matched d={dist}] 0/{N} visible!")
        save_image(image2, str(out_dir / f"matched_d{dist:.1f}.png"))

    print(f"\nDone! Check {out_dir}/")


if __name__ == "__main__":
    main()
