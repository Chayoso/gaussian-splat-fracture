"""
Rendering post-processing utilities.

- Depth-to-normal computation for screen-space shading
- Video creation from frame sequences
"""

import torch
import numpy as np
from pathlib import Path


def depth_to_normal(depth: torch.Tensor, camera) -> torch.Tensor:
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
    nx = -dz_dx / (fx + 1e-8)
    ny = -dz_dy / (fy + 1e-8)
    nz = torch.ones_like(d)

    normal = torch.stack([nx, ny, nz], dim=0)  # (3, H, W)
    norm = normal.norm(dim=0, keepdim=True).clamp(min=1e-8)
    normal = normal / norm

    # Transform view-space normals to world-space using inverse view rotation
    view_mat = camera.world_view_transform.T  # column-major to row-major
    R_view = view_mat[:3, :3]
    R_inv = R_view.T

    n_flat = normal.reshape(3, -1).T  # (H*W, 3)
    n_world = (n_flat @ R_inv.T).T.reshape(3, H, W)

    # Zero out background (depth == 0)
    mask = (d > 0).unsqueeze(0).float()
    n_world = n_world * mask

    return n_world


def create_video(frame_dir: Path, output_path: str, fps: int):
    """
    Create H.264 mp4 video from saved frames (VS Code compatible).

    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
    """
    import subprocess

    frames = sorted(Path(frame_dir).glob("frame_*.png"))
    if not frames:
        print(f"[Warning] No frames found in {frame_dir}")
        return

    output_path = str(output_path)

    # Try ffmpeg H.264
    try:
        frame_pattern = str(Path(frame_dir) / "frame_%04d.png")
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
