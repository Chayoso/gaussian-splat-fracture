"""Multi-GPU launcher: flatten suite-level prompt lists into per-prompt
jobs and distribute across N GPUs.

`run_siggraph_evidence.py` itself loops prompts sequentially within a
suite -- on a 4-GPU server that wastes 75% of the compute.  This
launcher reads the same suite definitions, flattens them into
(prompt, mesh, overrides, tier_overrides) jobs, and runs each job on
its own GPU via ``CUDA_VISIBLE_DEVICES``.

Usage::

    python scripts/server_launch_chain.py \\
        --suites sentence_style material_radial mesh_repro ablation \\
        --tiers final50k_long final50k_long final50k_long quick10k \\
        --gpus 0,1,2,3 \\
        --out-root output/at2_server_v1

Or a single-suite parallel run::

    python scripts/server_launch_chain.py \\
        --suites mesh_repro --tiers final50k_long --gpus 0,1,2,3 \\
        --out-root output/at2_mesh_long_v1
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

# Make the repo root importable so the suite definitions are reused.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_siggraph_evidence import (  # type: ignore  # noqa: E402
    EvidenceRun,
    TIERS,
    _suite_runs,
    _slug,
)


@dataclass
class Job:
    suite: str
    run_name: str
    prompt_idx: int
    prompt: str
    mesh: str
    tier_name: str
    overrides: dict
    tier_overrides: dict
    out_dir: Path

    def label(self) -> str:
        return f"{self.suite}/{self.run_name}/{self.prompt_idx:02d}_{_slug(self.prompt)[:32]}"


def flatten_jobs(
    suites: List[str],
    tiers: List[str],
    out_root: Path,
) -> List[Job]:
    if len(tiers) == 1 and len(suites) > 1:
        tiers = tiers * len(suites)
    if len(tiers) != len(suites):
        raise ValueError("--tiers must be either length 1 or match --suites")

    jobs: List[Job] = []
    for suite, tier_name in zip(suites, tiers):
        if tier_name not in TIERS:
            raise ValueError(f"unknown tier: {tier_name}")
        runs: List[EvidenceRun] = _suite_runs(suite)
        for run in runs:
            for idx, prompt in enumerate(run.prompts):
                run_dir = out_root / suite / run.name
                prompt_dir = run_dir / f"{idx:02d}_{_slug(prompt)}"
                jobs.append(Job(
                    suite=suite,
                    run_name=run.name,
                    prompt_idx=idx,
                    prompt=prompt,
                    mesh=run.mesh,
                    tier_name=tier_name,
                    overrides=dict(run.overrides),
                    tier_overrides=dict(run.tier_overrides),
                    out_dir=prompt_dir,
                ))
    return jobs


def _build_overrides_json(overrides: dict, path: Path) -> Optional[Path]:
    if not overrides:
        return None
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    return path


def _build_config_with_mesh(
    base_config: Path,
    mesh: str,
    out_dir: Path,
) -> Path:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(base_config)
    OmegaConf.update(cfg, "mesh.path", mesh, merge=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "evidence_config.yaml"
    OmegaConf.save(cfg, cfg_path)
    return cfg_path


def _run_one_job(
    job: Job,
    gpu_id: int,
    base_config: Path,
    python_bin: str,
    inspector: str,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    job.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _build_config_with_mesh(base_config, job.mesh, job.out_dir)

    overrides_path = _build_overrides_json(
        job.overrides, job.out_dir / "runtime_overrides.json"
    )

    tier = TIERS[job.tier_name]
    if job.tier_overrides:
        tier_kwargs = dict(tier.__dict__)
        tier_kwargs.update(job.tier_overrides)
        from scripts.run_siggraph_evidence import Tier  # type: ignore
        tier = Tier(**tier_kwargs)

    cmd = [
        python_bin,
        inspector,
        "--config", str(cfg_path),
        "--prompt", job.prompt,
        "--out", str(job.out_dir),
        "--gravity-particles", str(int(tier.particles)),
        "--gravity-frames", str(int(tier.frames)),
        "--gravity-grids", str(int(tier.grids)),
        "--physics-substeps", str(int(tier.substeps)),
        "--snapshot-stride", str(int(tier.snapshot_stride)),
        "--drop-center-z", str(float(tier.drop_center_z)),
        "--gravity-z", str(float(tier.gravity_z)),
    ]
    if overrides_path is not None:
        cmd.extend(["--override-json", str(overrides_path)])

    log_path = job.out_dir / "run.log"
    with log_path.open("w") as logf:
        logf.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n")
        logf.write(f"# cwd={PROJECT_ROOT}\n")
        logf.write(f"# cmd={' '.join(cmd)}\n\n")
        logf.flush()
        rc = subprocess.run(
            cmd, env=env, cwd=str(PROJECT_ROOT),
            stdout=logf, stderr=subprocess.STDOUT,
        ).returncode
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-GPU launcher for SIGGRAPH evidence suites.")
    parser.add_argument("--suites", nargs="+", required=True,
                        choices=["sentence_style", "material_radial",
                                 "style_interpolation", "mesh_repro",
                                 "siggraph_teaser", "ablation",
                                 "phase_gate_stress"])
    parser.add_argument("--tiers", nargs="+", required=True)
    parser.add_argument("--gpus", default="0",
                        help="comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--python", default=shutil.which("python") or sys.executable)
    parser.add_argument("--inspector",
                        default="scripts/inspect_gravity_crack_progression.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    base_config = (PROJECT_ROOT / args.config).resolve()
    inspector_path = (PROJECT_ROOT / args.inspector).resolve()

    gpu_ids = [int(g.strip()) for g in str(args.gpus).split(",") if g.strip()]
    if not gpu_ids:
        raise SystemExit("at least one GPU id required")

    jobs = flatten_jobs(args.suites, args.tiers, out_root)
    print(f"[launcher] {len(jobs)} jobs across {len(gpu_ids)} GPU(s) "
          f"({gpu_ids}) -> {out_root}")
    for j in jobs:
        print(f"  - {j.label()}  [{j.tier_name}]  mesh={Path(j.mesh).stem}")

    if args.dry_run:
        return 0

    # Producer-consumer with one worker thread per GPU.
    job_q: "queue.Queue[Optional[Job]]" = queue.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in gpu_ids:
        job_q.put(None)  # sentinel per worker

    results: dict[str, int] = {}
    results_lock = threading.Lock()

    def worker(gpu_id: int):
        while True:
            j = job_q.get()
            if j is None:
                job_q.task_done()
                return
            t0 = time.time()
            print(f"[gpu {gpu_id}] start {j.label()}", flush=True)
            try:
                rc = _run_one_job(j, gpu_id, base_config,
                                  args.python, str(inspector_path))
            except Exception as exc:
                print(f"[gpu {gpu_id}] EXCEPTION {j.label()}: {exc}", flush=True)
                rc = 99
            dt = time.time() - t0
            tag = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"[gpu {gpu_id}] {tag} {j.label()}  ({dt/60:.1f} min)",
                  flush=True)
            with results_lock:
                results[j.label()] = rc
            job_q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in gpu_ids]
    for t in threads:
        t.start()
    job_q.join()
    for t in threads:
        t.join()

    failed = [k for k, v in results.items() if v != 0]
    print(f"\n[launcher] {len(results) - len(failed)}/{len(results)} jobs OK")
    if failed:
        print(f"[launcher] failures:")
        for k in failed:
            print(f"  - {k}  rc={results[k]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
