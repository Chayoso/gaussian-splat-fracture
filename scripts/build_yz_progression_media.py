"""Build Y-Z-only comparison media from progression snapshot outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT = _load_font(17)
FONT_SMALL = _load_font(14)
FONT_BOLD = _load_font(18, bold=True)


def _short(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[: max(width - 3, 1)] + "..."


def _read_prompt(prompt_dir: Path) -> str:
    prompt_path = prompt_dir / "prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return prompt_dir.name


def _resolve_plot_path(prompt_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = prompt_dir / value
    if candidate.exists():
        return candidate
    candidate = prompt_dir.parent / value
    return candidate


def _load_prompt_runs(root: Path) -> list[dict]:
    runs = []
    if (root / "progression_snapshots.json").exists():
        prompt_dirs = [root]
    else:
        prompt_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for prompt_dir in prompt_dirs:
        snapshots_path = prompt_dir / "progression_snapshots.json"
        if not snapshots_path.exists():
            continue
        rows = json.loads(snapshots_path.read_text(encoding="utf-8"))
        if not rows:
            continue
        summary_path = prompt_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append({
            "dir": prompt_dir,
            "run_name": prompt_dir.name,
            "prompt": _read_prompt(prompt_dir),
            "rows": rows,
            "summary": summary,
        })
    return runs


def _limit_rows(rows: list[dict], max_steps: int | None) -> list[dict]:
    if max_steps is None or max_steps <= 0 or len(rows) <= max_steps:
        return rows
    if max_steps == 1:
        return [rows[-1]]
    indices = [
        round(i * (len(rows) - 1) / (max_steps - 1))
        for i in range(max_steps)
    ]
    return [rows[int(idx)] for idx in indices]


def _apply_step_limit(runs: list[dict], max_steps: int | None) -> list[dict]:
    if max_steps is None or max_steps <= 0:
        return runs
    limited = []
    for run in runs:
        run_copy = dict(run)
        run_copy["rows"] = _limit_rows(list(run["rows"]), max_steps)
        limited.append(run_copy)
    return limited


def _raw_yz_halves(image_path: Path) -> tuple[Image.Image, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    x0 = int(w * 0.655)
    x1 = int(w * 0.985)
    top = image.crop((x0, int(h * 0.060), x1, int(h * 0.505)))
    bottom = image.crop((x0, int(h * 0.535), x1, int(h * 0.985)))
    return top, bottom


def _make_yz_cell(
    image_path: Path,
    *,
    label: str,
    cell_width: int,
    cell_height: int,
) -> Image.Image:
    top, bottom = _raw_yz_halves(image_path)
    pad = 8
    label_h = 26
    panel_w = (cell_width - 3 * pad) // 2
    panel_h = cell_height - label_h - 2 * pad
    top.thumbnail((panel_w, panel_h), Image.LANCZOS)
    bottom.thumbnail((panel_w, panel_h), Image.LANCZOS)

    cell = Image.new("RGB", (cell_width, cell_height), "white")
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, 0, cell_width - 1, cell_height - 1), outline=(214, 219, 224), width=1)
    draw.text((pad, 5), label, fill=(32, 36, 40), font=FONT_SMALL)
    draw.text((pad, label_h), "crack", fill=(32, 72, 128), font=FONT_SMALL)
    draw.text((pad * 2 + panel_w, label_h), "fragment", fill=(92, 62, 118), font=FONT_SMALL)

    y_top = label_h + pad + 16
    cell.paste(top, (pad, y_top))
    cell.paste(bottom, (pad * 2 + panel_w, y_top))
    return cell


def _blank_cell(text: str, cell_width: int, cell_height: int) -> Image.Image:
    cell = Image.new("RGB", (cell_width, cell_height), (248, 249, 250))
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, 0, cell_width - 1, cell_height - 1), outline=(214, 219, 224), width=1)
    draw.text((12, 12), text, fill=(108, 117, 125), font=FONT_SMALL)
    return cell


def _row_label(run: dict) -> str:
    summary = run["summary"]
    parts = []
    if (run["dir"] / "prompts.txt").exists():
        parts.append(_short(run.get("run_name", run["dir"].name), 42))
    parts.extend([
        _short(run["prompt"], 42),
        f"{summary.get('family', '')} / {summary.get('sentence_style', '')}",
        f"{summary.get('fragment_release_verdict', '')} detached={summary.get('final_nonbase_fragment_count', 0)}",
        f"rel={float(summary.get('final_released_node_ratio', 0.0) or 0.0):.3f} bcut={float(summary.get('final_bcut', 0.0) or 0.0):.3f}",
    ])
    return "\n".join(parts)


def _make_montage(runs: list[dict], out_path: Path, cell_width: int, cell_height: int) -> None:
    max_steps = max(len(run["rows"]) for run in runs)
    label_w = 310
    header_h = 72
    width = label_w + max_steps * cell_width
    height = header_h + len(runs) * cell_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Y-Z Crack And Fragment Progression", fill=(24, 28, 33), font=FONT_BOLD)

    for step in range(max_steps):
        draw.text(
            (label_w + step * cell_width + 10, 44),
            f"snapshot {step}",
            fill=(52, 58, 64),
            font=FONT_SMALL,
        )

    for row_i, run in enumerate(runs):
        y = header_h + row_i * cell_height
        draw.rectangle((0, y, label_w - 1, y + cell_height - 1), fill=(248, 249, 250), outline=(222, 226, 230))
        draw.multiline_text((12, y + 12), _row_label(run), fill=(24, 28, 33), font=FONT_SMALL, spacing=4)
        for step in range(max_steps):
            x = label_w + step * cell_width
            if step >= len(run["rows"]):
                cell = _blank_cell("no snapshot", cell_width, cell_height)
            else:
                snap = run["rows"][step]
                plot = _resolve_plot_path(run["dir"], str(snap["plot"]))
                label = f"impact+{snap.get('since_impact', step)}  frame {snap.get('loop_frame', '')}"
                cell = _make_yz_cell(plot, label=label, cell_width=cell_width, cell_height=cell_height)
            canvas.paste(cell, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _make_video_frames(
    runs: list[dict],
    frame_dir: Path,
    cell_width: int,
    cell_height: int,
) -> int:
    frame_dir.mkdir(parents=True, exist_ok=True)
    max_steps = max(len(run["rows"]) for run in runs)
    cols = 2
    rows = (len(runs) + cols - 1) // cols
    title_h = 54
    label_h = 62
    width = cols * cell_width
    height = title_h + rows * (label_h + cell_height)
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    for step in range(max_steps):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 14), f"Y-Z progression | snapshot {step}", fill=(24, 28, 33), font=FONT_BOLD)
        for idx, run in enumerate(runs):
            col = idx % cols
            row = idx // cols
            x = col * cell_width
            y = title_h + row * (label_h + cell_height)
            draw.rectangle((x, y, x + cell_width - 1, y + label_h - 1), fill=(248, 249, 250), outline=(222, 226, 230))
            draw.multiline_text(
                (x + 10, y + 8),
                _row_label(run),
                fill=(24, 28, 33),
                font=FONT_SMALL,
                spacing=3,
            )
            if step >= len(run["rows"]):
                cell = _blank_cell("no snapshot", cell_width, cell_height)
            else:
                snap = run["rows"][step]
                plot = _resolve_plot_path(run["dir"], str(snap["plot"]))
                label = f"impact+{snap.get('since_impact', step)}  frame {snap.get('loop_frame', '')}"
                cell = _make_yz_cell(plot, label=label, cell_width=cell_width, cell_height=cell_height)
            canvas.paste(cell, (x, y + label_h))
        canvas.save(frame_dir / f"frame_{step:03d}.png")
    return max_steps


def _encode_mp4(frame_dir: Path, mp4_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%03d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Y-Z montage and MP4 from progression snapshots.")
    parser.add_argument("--progression-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cell-width", type=int, default=430)
    parser.add_argument("--cell-height", type=int, default=250)
    parser.add_argument("--video-cell-width", type=int, default=620)
    parser.add_argument("--video-cell-height", type=int, default=330)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.progression_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / "yz_media"
    runs = _load_prompt_runs(root)
    if not runs:
        raise SystemExit(f"No progression prompt runs found under {root}")
    runs = _apply_step_limit(runs, int(args.max_steps) if int(args.max_steps) > 0 else None)

    montage_path = out_dir / "yz_sentence_result_montage.png"
    _make_montage(runs, montage_path, int(args.cell_width), int(args.cell_height))

    frame_dir = out_dir / "video_frames"
    _make_video_frames(
        runs,
        frame_dir,
        int(args.video_cell_width),
        int(args.video_cell_height),
    )
    mp4_path = out_dir / "yz_sentence_result.mp4"
    _encode_mp4(frame_dir, mp4_path, max(int(args.fps), 1))

    manifest = {
        "progression_dir": str(root),
        "runs": len(runs),
        "montage": str(montage_path),
        "mp4": str(mp4_path),
        "fps": max(int(args.fps), 1),
        "max_steps": int(args.max_steps),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
