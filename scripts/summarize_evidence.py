"""Summarize SIGGRAPH evidence sweep into a single comparison table.

Reads `evidence_report.md` and per-prompt `progression_metrics.csv` files
from a `run_siggraph_evidence.py --out <root>` directory and writes
`<root>/summary.md` and `<root>/summary.csv` with the columns most
useful for paper figures: prompt, family, style, verdict, n_fragments,
released ratio, boundary-cut support, phase birth, scatter_max,
lateral_max, fragment_drop_max, detached_distance_max.

Usage:
    python scripts/summarize_evidence.py output/at2_core_50k_v1
    python scripts/summarize_evidence.py output/at2_mesh_50k_v1
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRIC_COLUMNS = (
    ("n_frags", "n_fragments", "last"),
    ("released", "released_node_ratio", "last"),
    ("bcut", "fragment_boundary_cut_support_ratio", "last"),
    ("birth_phase", "birth_phase_score", "max"),
    ("birth_cvol", "birth_cvol_max", "max"),
    ("scatter_max", "physical_release_displacement", "max"),
    ("lat_max", "physical_lateral_release_displacement", "max"),
    ("drop_max", "physical_fragment_drop", "max"),
    ("det_max", "physical_detached_distance", "max"),
    ("branch_events", "branch_event_count", "max"),
    ("closure_score", "closure_score_max", "max"),
)


def _read_csv_metric(csv_path: Path, column: str, agg: str) -> float:
    if not csv_path.exists():
        return float("nan")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or column not in rows[0]:
        return float("nan")
    try:
        values = [float(r[column] or 0.0) for r in rows]
    except ValueError:
        return float("nan")
    if not values:
        return float("nan")
    if agg == "last":
        return values[-1]
    if agg == "max":
        return max(values)
    if agg == "min":
        return min(values)
    if agg == "mean":
        return sum(values) / len(values)
    raise ValueError(f"unknown agg: {agg}")


def _collect_prompt_dirs(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        for prompt_dir in sorted(run_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            metrics_csv = prompt_dir / "progression_metrics.csv"
            if not metrics_csv.exists():
                continue
            prompt_path = prompt_dir / "prompt.txt"
            prompt_text = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else prompt_dir.name
            row: Dict[str, Any] = {
                "run": run_dir.name,
                "prompt_slug": prompt_dir.name,
                "prompt": prompt_text,
            }
            for short, col, agg in METRIC_COLUMNS:
                row[short] = _read_csv_metric(metrics_csv, col, agg)
            out.append(row)
    return out


def _format_md(rows: Iterable[Dict[str, Any]]) -> str:
    rows = list(rows)
    if not rows:
        return "_no rows_\n"
    headers = ["run", "prompt"] + [c[0] for c in METRIC_COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        cells = [str(r["run"])[:32], str(r["prompt"])[:60]]
        for short, _col, _agg in METRIC_COLUMNS:
            v = r.get(short, float("nan"))
            cells.append(f"{v:.0f}" if short == "n_frags" or short == "branch_events"
                         else f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    rows = _collect_prompt_dirs(root)
    md = _format_md(rows)
    (root / "summary.md").write_text(
        f"# Evidence summary: {root.name}\n\n{md}\n", encoding="utf-8",
    )
    if rows:
        with (root / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {root / 'summary.md'} and {root / 'summary.csv'} "
          f"({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
