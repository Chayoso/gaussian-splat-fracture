"""Quick camera angle comparison - renders frame 0 at multiple azimuths."""
import torch, numpy as np, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gaussian-splatting"))

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from src.renderer.camera.config import make_matrices_from_yaml
from src.preprocessing.ply_loader import PretrainedPlyLoader
from torchvision.utils import save_image
from scipy.spatial.transform import Rotation as ScipyR
import math

torch.cuda.set_device(0)

# Load PLY
loader = PretrainedPlyLoader("assets/trained/lego/point_cloud.ply", sh_degree=3)
ply_data = loader.load_raw_ply()
ply_xyz_norm, scale_factor = loader.normalize_positions(ply_data['xyz'])
M = ply_xyz_norm.shape[0]

# Apply drop transforms: scale=0.33, rotation=[30,0,15], center_z=0.7
drop_scale = 0.33
center = np.array([0.5, 0.5, 0.5])
xyz = center + (ply_xyz_norm - center) * drop_scale

# Z shift
z_center = (xyz[:, 2].min() + xyz[:, 2].max()) / 2
xyz[:, 2] += 0.7 - z_center

# Rotation
R_drop = ScipyR.from_euler('xyz', [30, 0, 15], degrees=True)
R_mat = R_drop.as_matrix()
xyz = center + (xyz - center) @ R_mat.T

# Build gaussians
gaussians = GaussianModel(sh_degree=3)
gaussians._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
gaussians._features_dc = torch.nn.Parameter(
    torch.tensor(ply_data['features_dc'], dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
rest = ply_data['features_rest']
gaussians._features_rest = torch.nn.Parameter(
    torch.tensor(rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
gaussians._opacity = torch.nn.Parameter(
    torch.tensor(ply_data['opacity'], dtype=torch.float, device="cuda"))
scales = ply_data['scaling'] + math.log(scale_factor) + math.log(drop_scale)
gaussians._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))

# Rotate quaternions
q_drop = R_drop.as_quat()  # [x,y,z,w]
q_wxyz = torch.tensor([q_drop[3], q_drop[0], q_drop[1], q_drop[2]], dtype=torch.float, device="cuda")
rots = torch.tensor(ply_data['rotation'], dtype=torch.float, device="cuda")
w1, x1, y1, z1 = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
w2, x2, y2, z2 = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
new_rots = torch.stack([
    w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
    w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)
gaussians._rotation = torch.nn.Parameter(new_rots)
gaussians.active_sh_degree = 3
gaussians.max_radii2D = torch.zeros(M, device="cuda")
gaussians.exposure_mapping = {}
gaussians.pretrained_exposures = None

class PipeArgs:
    debug = False; convert_SHs_python = False; compute_cov3D_python = False; antialiasing = False
pipe = PipeArgs()
bg = torch.ones(3, device="cuda")

# Object center for camera target
obj_center = xyz.mean(axis=0)
print(f"Object center: {obj_center}")

out = Path("output/camera_test")
out.mkdir(parents=True, exist_ok=True)

az = -15
elev = 10
dist = 0.75

# Fixed camera based on drop_z=0.7 (current setting)
cam_target = obj_center  # already computed at drop_z=0.7
eye_x = cam_target[0] + dist * np.cos(np.radians(elev)) * np.cos(np.radians(az))
eye_y = cam_target[1] + dist * np.cos(np.radians(elev)) * np.sin(np.radians(az))
eye_z = cam_target[2] + dist * np.sin(np.radians(elev))

camera_yaml = {
    "width": 960, "height": 960,
    "fx": 1029.0, "fy": 1029.0, "cx": 480.0, "cy": 480.0,
    "znear": 0.01, "zfar": 100.0,
    "lookat": {"eye": [float(eye_x), float(eye_y), float(eye_z)],
               "target": cam_target.tolist(), "up": [0, 0, 1]}
}
w, h, tfx, tfy, vm, pm, cp = make_matrices_from_yaml(camera_yaml)
wvt = torch.from_numpy(vm).cuda()
fpt = torch.from_numpy(pm).cuda()
cam = MiniCam(w, h, 2*np.arctan(tfy), 2*np.arctan(tfx), 0.01, 100.0, wvt, fpt)
print(f"Fixed camera: eye=[{eye_x:.3f},{eye_y:.3f},{eye_z:.3f}] target={cam_target}")

for drop_z in [0.6, 0.7, 0.8, 0.9, 1.0]:
    # Recompute positions with different drop_center_z
    xyz_t = center + (ply_xyz_norm - center) * drop_scale
    z_c = (xyz_t[:, 2].min() + xyz_t[:, 2].max()) / 2
    xyz_t[:, 2] += drop_z - z_c
    xyz_t = center + (xyz_t - center) @ R_mat.T

    # Update Gaussian positions only (camera stays fixed)
    gaussians._xyz.data = torch.tensor(xyz_t, dtype=torch.float, device="cuda")
    oc = xyz_t.mean(axis=0)

    result = render(cam, gaussians, pipe, bg)
    save_image(result["render"], str(out / f"dropz_{drop_z:.1f}.png"))
    v = (result["radii"] > 0).sum().item()
    print(f"drop_z={drop_z:.1f}: obj_center=[{oc[0]:.3f},{oc[1]:.3f},{oc[2]:.3f}] {v}/{M} visible")

print(f"\nDone! Check {out}/")
