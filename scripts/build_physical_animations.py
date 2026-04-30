"""Build physical-view animations from sweep snapshots.

For each prompt directory under a sweep root (e.g.
``output/at2_core_50k_v3``), collect the ``frame_*_physical.png``
snapshots that ``inspect_gravity_crack_progression`` writes and encode
them into an mp4 (`<prompt_dir>/physical.mp4`).  Also builds a
top-level montage mp4 that places all prompts of one suite-run side
by side for at-a-glance visual comparison.

Usage:
    python scripts/build_physical_animations.py output/at2_core_50k_v3
    python scripts/build_physical_animations.py output/at2_mesh_50k_v1
    python scripts/build_physical_animations.py output/at2_100k_radial_probe_v1
    python scripts/build_physical_animations.py output/at2_ablation_10k_v1

By default uses 2 fps (slow enough to read each frame); pass
``--fps 6`` for a smoother animation.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise SystemExit("ffmpeg not found on PATH")
    return binary


def _physical_snapshots(prompt_dir: Path) -> List[Path]:
    snapshots = sorted((prompt_dir / "snapshots").glob("frame_*_physical.png"))
    return snapshots


def _make_mp4(frames: List[Path], out_mp4: Path, fps: int) -> bool:
    if not frames:
        return False
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Build a temporary frame index for ffmpeg by writing a concat list.
    list_path = out_mp4.parent / f".{out_mp4.stem}.frames.txt"
    with list_path.open("w") as f:
        for frame in frames:
            f.write(f"file '{frame.resolve()}'\n")
            f.write(f"duration {1.0 / max(fps, 1):.4f}\n")
        # ffmpeg concat needs the last file repeated without duration
        f.write(f"file '{frames[-1].resolve()}'\n")
    cmd = [
        _ffmpeg(),
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-crf", "20",
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        list_path.unlink(missing_ok=True)
    return out_mp4.exists()


def _find_prompt_dirs(root: Path) -> List[Path]:
    """Return all leaf directories that contain a `snapshots/` folder
    with at least one `*_physical.png`."""
    out: List[Path] = []
    for p in sorted(root.rglob("snapshots")):
        if not p.is_dir():
            continue
        if any(p.glob("frame_*_physical.png")):
            out.append(p.parent)
    return out


def _short_label(prompt_dir: Path, root: Path) -> str:
    rel = prompt_dir.relative_to(root)
    name = str(rel).replace("/", "_")
    return name[:48]


def _make_montage_mp4(
    prompt_dirs: List[Path],
    root: Path,
    out_mp4: Path,
    fps: int,
    max_cols: int = 3,
    cell_width: int = 720,
    cell_height: int = 240,
) -> bool:
    """Stitch multiple per-prompt physical snapshot streams into one mp4."""
    if not prompt_dirs:
        return False
    streams = [_physical_snapshots(d) for d in prompt_dirs]
    streams = [s for s in streams if s]
    if not streams:
        return False

    n = len(streams)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    title_h = 28
    label_h = 18
    width = cols * cell_width
    height = title_h + rows * (cell_height + label_h)
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    frame_dir = out_mp4.parent / f".{out_mp4.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    max_steps = max(len(s) for s in streams)
    for step in range(max_steps):
        canvas = Image.new("RGB", (width, height), "white")
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(canvas)
            try:
                font_small = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_title = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except OSError:
                font_small = ImageFont.load_default()
                font_title = font_small
            draw.text((12, 6), f"physical view  |  step {step + 1}/{max_steps}",
                      fill=(20, 24, 28), font=font_title)
        except Exception:
            font_small = None

        for idx, (prompt_dir, snaps) in enumerate(zip(prompt_dirs, streams)):
            col = idx % cols
            row = idx // cols
            x = col * cell_width
            y = title_h + row * (cell_height + label_h)
            label = _short_label(prompt_dir, root)
            if font_small is not None:
                draw.text((x + 8, y), label, fill=(40, 44, 50), font=font_small)
            snap_idx = min(step, len(snaps) - 1)
            try:
                img = Image.open(snaps[snap_idx]).convert("RGB")
                img.thumbnail((cell_width - 4, cell_height - 4))
                canvas.paste(img, (x + 2, y + label_h + 2))
            except Exception:
                pass

        canvas.save(frame_dir / f"frame_{step:04d}.png")

    cmd = [
        _ffmpeg(),
        "-y",
        "-framerate", str(max(fps, 1)),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-crf", "20",
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return out_mp4.exists()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build per-prompt physical mp4 + suite montage mp4.")
    parser.add_argument("root", help="sweep output root, e.g. output/at2_core_50k_v3")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--no-montage", action="store_true")
    parser.add_argument("--max-cols", type=int, default=3)
    args = parser.parse_args(argv[1:])

    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    prompt_dirs = _find_prompt_dirs(root)
    if not prompt_dirs:
        print(f"no physical snapshots under {root}", file=sys.stderr)
        return 1

    print(f"found {len(prompt_dirs)} prompt directories with physical snapshots")
    for prompt_dir in prompt_dirs:
        snaps = _physical_snapshots(prompt_dir)
        out_mp4 = prompt_dir / "physical.mp4"
        ok = _make_mp4(snaps, out_mp4, args.fps)
        flag = "OK" if ok else "skip"
        print(f"  [{flag}] {prompt_dir.relative_to(root)} -- {len(snaps)} frames -> physical.mp4")

    # Group prompt dirs by their immediate parent (suite-run) and build
    # one montage per group.
    if not args.no_montage:
        groups: dict[Path, List[Path]] = {}
        for prompt_dir in prompt_dirs:
            groups.setdefault(prompt_dir.parent, []).append(prompt_dir)
        for parent, dirs in groups.items():
            if len(dirs) < 2:
                continue
            montage_path = parent / "physical_montage.mp4"
            ok = _make_montage_mp4(dirs, root, montage_path, args.fps,
                                    max_cols=int(args.max_cols))
            flag = "OK" if ok else "skip"
            print(f"  [{flag}] montage -> {montage_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
