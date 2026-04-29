"""Run SIGGRAPH-style controllability and causality evidence suites.

This is an orchestration layer around inspect_gravity_crack_progression.py.
It does not change the v1.5 algorithm; it standardizes prompt matrices,
mesh/config copies, ablation overrides, and Y-Z media generation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Tier:
    particles: int
    frames: int
    grids: int
    substeps: int
    snapshot_stride: int
    drop_center_z: float
    gravity_z: float
    media_max_steps: int


@dataclass(frozen=True)
class EvidenceRun:
    name: str
    prompts: list[str]
    mesh: str = "assets/meshes/bunny.obj"
    overrides: dict[str, Any] = field(default_factory=dict)
    tier_overrides: dict[str, Any] = field(default_factory=dict)
    build_media: bool = True


TIERS = {
    "smoke2k": Tier(
        particles=2000,
        frames=36,
        grids=32,
        substeps=1,
        snapshot_stride=6,
        drop_center_z=0.25,
        gravity_z=-5000.0,
        media_max_steps=6,
    ),
    "quick10k": Tier(
        particles=10000,
        frames=52,
        grids=64,
        substeps=3,
        snapshot_stride=2,
        drop_center_z=0.42,
        gravity_z=-3500.0,
        media_max_steps=10,
    ),
    "final50k": Tier(
        particles=50000,
        frames=52,
        grids=64,
        substeps=3,
        snapshot_stride=2,
        drop_center_z=0.42,
        gravity_z=-3500.0,
        media_max_steps=10,
    ),
}


SENTENCE_STYLE_PROMPTS = [
    "soda-lime glass object with one long smooth crack",
    "soda-lime glass object with sparse branching cracks",
    "soda-lime glass object forming dense spiderweb branching cracks",
    "soda-lime glass object shattering into many sharp radial cracks",
    "soda-lime glass object with diffuse tiny surface scratches and no visible fracture",
]


MATERIAL_RADIAL_PROMPTS = [
    "soda-lime glass object under localized radial crack impact",
    "porcelain ceramic object under localized radial crack impact",
    "rough concrete object under localized radial crack impact",
    "clear ice object under localized radial crack impact",
    "vulcanized rubber object under localized radial crack impact",
    "structural steel object under localized radial crack impact",
]


STYLE_INTERPOLATION_PROMPTS = [
    "soda-lime glass object with one clean smooth crack",
    "soda-lime glass object with a few branching cracks",
    "soda-lime glass object with connected spiderweb branching cracks",
    "soda-lime glass object shattering into dense radial branching cracks",
]


MESH_REPRO_PROMPTS = [
    "soda-lime glass object shattering into many sharp radial cracks",
    "rough concrete object crumbling into irregular chunks",
    "vulcanized rubber object deforming without visible fracture",
]


SIGGRAPH_TEASER_PROMPTS = [
    "soda-lime glass object with one long smooth crack",
    "soda-lime glass object forming dense spiderweb branching cracks",
    "soda-lime glass object shattering into many sharp radial cracks",
    "rough concrete object crumbling into irregular chunks",
    "vulcanized rubber object deforming without visible fracture",
    "structural steel object denting without visible fracture",
]


ABLATION_PROMPT = "soda-lime glass object shattering into many sharp radial cracks"
PHASE_GATE_STRESS_PROMPTS = [
    "rough concrete object crumbling into irregular chunks",
]


ABLATION_OVERRIDES = {
    "baseline": {},
    "no_phase_approval": {
        "manifold.phase_approval_enable": False,
    },
    "no_phase_cc_modulation": {
        "manifold.phase_cc_modulation_enable": False,
    },
    "no_phase_total": {
        "manifold.phase_approval_enable": False,
        "manifold.phase_cc_modulation_enable": False,
    },
    "no_at2_jacobi": {
        "manifold.at2_jacobi_enable": False,
        "manifold.at2_drive_gain": 0.0,
        "manifold.at2_reg_gain": 0.0,
    },
    "no_griffith_gate": {
        "manifold.growth_griffith_threshold": 0.0,
    },
    "branch_direction_angle": {
        "manifold.branch_direction_mode": "angle",
    },
    "branch_direction_hybrid": {
        "manifold.branch_direction_mode": "hybrid",
    },
    "no_crack_front_branching": {
        "manifold.successor_topk": 1,
        "manifold.branching_bias": 0.0,
        "manifold.branch_score_ratio": 1.10,
        "manifold.branch_drive_threshold": 1.05,
        "manifold.max_branching_tips": 1,
    },
    "no_narrow_band_volume_feedback": {
        "manifold.volumetric_cut_damage_scale": 0.0,
        "manifold.volumetric_auth_damage_floor": 0.0,
        "manifold.volumetric_detached_damage_floor": 0.0,
        "manifold.crack_volume_feedback_gain": 0.0,
        "manifold.crack_volume_opening_gain": 0.0,
        "manifold.crack_volume_visited_floor": 0.0,
        "manifold.crack_volume_tip_floor": 0.0,
        "manifold.crack_volume_interior_scale": 0.0,
        "manifold.crack_volume_immediate_feedback": 0.0,
    },
    "no_clip_material_prior": {
        "manifold.material_family": "neutral_reference",
        "gaussian_splatting.material_family": "neutral_reference",
        "manifold.material_drive_floor": 0.025,
        "manifold.damage_source_scale": 0.35,
        "manifold.front_threshold": 0.05,
        "manifold.tip_propagation_scale": 0.75,
        "manifold.front_substeps": 2,
        "manifold.tau_init": 0.30,
        "manifold.growth_gain": 0.92,
        "manifold.band_width": 1.25,
        "manifold.band_fill_gain": 0.24,
        "manifold.open_gain": 0.88,
        "manifold.edge_break_rate": 0.92,
        "manifold.branching_bias": 0.16,
        "manifold.strict_closure_max_released_ratio": 0.35,
    },
    "legacy_F_reset": {
        "manifold.impact_F_reset_alpha": 1.0,
    },
    "legacy_damage_delay": {
        "manifold.damage_feedback_delay_frames": 8,
        "manifold.damage_feedback_ramp_frames": 6,
    },
}


PHASE_GATE_STRESS_BASE_OVERRIDES = {
    "manifold.phase_approval_threshold_scale": 2.0,
    "manifold.phase_approval_threshold_offset": 0.25,
    "manifold.volumetric_cut_damage_scale": 0.18,
    "manifold.volumetric_auth_damage_floor": 0.28,
    "manifold.volumetric_detached_damage_floor": 0.38,
    "manifold.crack_volume_feedback_gain": 0.12,
    "manifold.crack_volume_opening_gain": 0.10,
    "manifold.crack_volume_visited_floor": 0.03,
    "manifold.crack_volume_tip_floor": 0.04,
    "manifold.crack_volume_interior_scale": 0.22,
    "manifold.cut_vote_strength": 0.72,
    "manifold.fragment_primary_cut_ratio": 0.42,
    "manifold.fragment_fallback_cut_ratio": 0.34,
    "manifold.fragment_cut_diffusion_alpha": 0.35,
    "manifold.fragment_cut_diffusion_iters": 2,
}


PHASE_GATE_STRESS_OVERRIDES = {
    "phase_gate_on": PHASE_GATE_STRESS_BASE_OVERRIDES,
    "phase_gate_bypassed": {
        **PHASE_GATE_STRESS_BASE_OVERRIDES,
        "manifold.phase_approval_enable": False,
    },
}


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    out = "_".join(part for part in out.split("_") if part)
    return out[:96] or "run"


def _write_prompts(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def _write_config(base_config: Path, out_dir: Path, mesh: str) -> Path:
    cfg = OmegaConf.load(base_config)
    OmegaConf.update(cfg, "mesh.path", mesh, merge=False)
    config_path = out_dir / "evidence_config.yaml"
    OmegaConf.save(cfg, config_path)
    return config_path


def _write_overrides(out_dir: Path, overrides: dict[str, Any]) -> str | None:
    if not overrides:
        return None
    path = out_dir / "runtime_overrides.json"
    path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    return str(path)


def _with_tier_overrides(tier: Tier, overrides: dict[str, Any]) -> Tier:
    if not overrides:
        return tier
    values = dict(tier.__dict__)
    values.update(overrides)
    return Tier(**values)


def _suite_runs(suite: str) -> list[EvidenceRun]:
    if suite == "sentence_style":
        return [
            EvidenceRun(
                name="same_object_same_impact_different_sentence",
                prompts=SENTENCE_STYLE_PROMPTS,
            )
        ]
    if suite == "material_radial":
        return [
            EvidenceRun(
                name="same_sentence_different_material",
                prompts=MATERIAL_RADIAL_PROMPTS,
            )
        ]
    if suite == "style_interpolation":
        return [
            EvidenceRun(
                name="crack_style_interpolation",
                prompts=STYLE_INTERPOLATION_PROMPTS,
            )
        ]
    if suite == "mesh_repro":
        return [
            EvidenceRun(
                name=f"mesh_repro_{Path(mesh).stem}",
                prompts=MESH_REPRO_PROMPTS,
                mesh=mesh,
                tier_overrides={"frames": 72},
            )
            for mesh in [
                "assets/meshes/bunny.obj",
                "assets/meshes/spot.obj",
                "assets/meshes/truck.obj",
            ]
        ]
    if suite == "siggraph_teaser":
        return [
            EvidenceRun(
                name="siggraph_teaser_controllability",
                prompts=SIGGRAPH_TEASER_PROMPTS,
            )
        ]
    if suite == "ablation":
        return [
            EvidenceRun(
                name=f"ablation_{name}",
                prompts=[ABLATION_PROMPT],
                overrides=overrides,
                build_media=False,
            )
            for name, overrides in ABLATION_OVERRIDES.items()
        ]
    if suite == "phase_gate_stress":
        return [
            EvidenceRun(
                name=f"phase_gate_stress_{name}",
                prompts=PHASE_GATE_STRESS_PROMPTS,
                overrides=overrides,
            )
            for name, overrides in PHASE_GATE_STRESS_OVERRIDES.items()
        ]
    if suite == "core":
        return _suite_runs("sentence_style") + _suite_runs("material_radial")
    if suite == "all":
        return (
            _suite_runs("sentence_style")
            + _suite_runs("material_radial")
            + _suite_runs("style_interpolation")
            + _suite_runs("siggraph_teaser")
            + _suite_runs("ablation")
            + _suite_runs("phase_gate_stress")
            + _suite_runs("mesh_repro")
        )
    raise ValueError(f"unknown suite: {suite}")


def _run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _build_yz_media(
    progression_dir: Path,
    tier: Tier,
    dry_run: bool,
    out_dir: Path | None = None,
) -> Path:
    media_cmd = [
        sys.executable,
        "scripts/build_yz_progression_media.py",
        "--progression-dir",
        str(progression_dir),
        "--max-steps",
        str(tier.media_max_steps),
        "--fps",
        "1",
    ]
    if out_dir is not None:
        media_cmd.extend(["--out-dir", str(out_dir)])
    _run_cmd(media_cmd, PROJECT_ROOT, dry_run)
    return (out_dir if out_dir is not None else progression_dir / "yz_media") / "manifest.json"


def _run_evidence(run: EvidenceRun, tier: Tier, args, root: Path) -> dict:
    tier = _with_tier_overrides(tier, run.tier_overrides)
    run_dir = root / run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = run_dir / "prompts.txt"
    _write_prompts(prompt_path, run.prompts)
    config_path = _write_config(Path(args.config), run_dir, run.mesh)
    override_path = _write_overrides(run_dir, run.overrides)

    progression_cmd = [
        sys.executable,
        "scripts/inspect_gravity_crack_progression.py",
        "--config",
        str(config_path),
        "--prompts-file",
        str(prompt_path),
        "--out",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--gravity-particles",
        str(tier.particles),
        "--gravity-frames",
        str(tier.frames),
        "--gravity-grids",
        str(tier.grids),
        "--physics-substeps",
        str(tier.substeps),
        "--fragment-every",
        "1",
        "--snapshot-stride",
        str(tier.snapshot_stride),
        "--drop-center-z",
        str(tier.drop_center_z),
        "--gravity-z",
        str(tier.gravity_z),
    ]
    if override_path is not None:
        progression_cmd.extend(["--override-json", override_path])
    _run_cmd(progression_cmd, PROJECT_ROOT, bool(args.dry_run))

    media_manifest = None
    if run.build_media:
        media_manifest = _build_yz_media(run_dir, tier, bool(args.dry_run))

    return {
        "name": run.name,
        "mesh": run.mesh,
        "prompts": len(run.prompts),
        "dir": str(run_dir),
        "summary": str(run_dir / "progression_sweep_summary.json"),
        "report": str(run_dir / "progression_sweep_report.md"),
        "media_manifest": str(media_manifest) if media_manifest else "",
        "overrides": run.overrides,
        "tier_overrides": run.tier_overrides,
    }


def _load_run_summaries(item: dict) -> list[dict]:
    summary_path = Path(item["summary"])
    if not summary_path.exists():
        return []
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_evidence_report(root: Path, manifest: dict) -> None:
    lines = [
        "# SIGGRAPH Evidence Run",
        "",
        f"- tier: `{manifest['tier']}`",
        f"- suite: `{manifest['suite']}`",
        f"- elapsed_sec: `{manifest.get('elapsed_sec', 0.0):.1f}`",
        "",
        "## Runs",
        "",
        "| run | mesh | prompts | report | media |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in manifest["runs"]:
        report = Path(item["report"])
        media = Path(item["media_manifest"]).parent / "yz_sentence_result_montage.png" if item["media_manifest"] else None
        report_rel = report.relative_to(root) if report.exists() else report
        media_rel = media.relative_to(root) if media is not None and media.exists() else ""
        lines.append(
            f"| {item['name']} | `{item['mesh']}` | {item['prompts']} | "
            f"[report]({report_rel}) | "
            f"{f'[montage]({media_rel})' if media_rel else ''} |"
        )

    suite_media = manifest.get("suite_media_manifest", "")
    if suite_media:
        suite_montage = Path(suite_media).parent / "yz_sentence_result_montage.png"
        suite_rel = suite_montage.relative_to(root) if suite_montage.exists() else suite_montage
        lines.extend([
            "",
            f"Suite media: [montage]({suite_rel})",
        ])

    lines.extend([
        "",
        "## Summary Rows",
        "",
        "| run | prompt | family/style | verdict | birth | frags | rel | bcut | phase | top1 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for item in manifest["runs"]:
        for row in _load_run_summaries(item):
            if "error" in row:
                lines.append(
                    f"| {item['name']} | {row.get('prompt', '')} | error | "
                    f"`{row['error']}` | -1 | 0 | 0 | 0 | 0 |  |"
                )
                continue
            lines.append(
                "| {run} | {prompt} | `{family}` / `{style}` | `{verdict}` | "
                "{birth} | {frags} | {rel} | {bcut} | {phase} | {top1} |".format(
                    run=item["name"],
                    prompt=row.get("prompt", ""),
                    family=row.get("family", ""),
                    style=row.get("sentence_style", ""),
                    verdict=row.get("fragment_release_verdict", ""),
                    birth=row.get("first_detach_since_impact", -1),
                    frags=row.get("final_nonbase_fragment_count", 0),
                    rel=_fmt(float(row.get("final_released_node_ratio", 0.0) or 0.0)),
                    bcut=_fmt(float(row.get("final_bcut", 0.0) or 0.0)),
                    phase=_fmt(float(row.get("fragment_birth_phase_score", 0.0) or 0.0)),
                    top1=row.get("top1", ""),
                )
            )
    (root / "evidence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SIGGRAPH controllability evidence suites.")
    parser.add_argument(
        "--suite",
        choices=[
            "sentence_style",
            "material_radial",
            "style_interpolation",
            "mesh_repro",
            "siggraph_teaser",
            "ablation",
            "phase_gate_stress",
            "core",
            "all",
        ],
        default="sentence_style",
    )
    parser.add_argument("--tier", choices=sorted(TIERS), default="quick10k")
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tier = TIERS[args.tier]
    out_root = Path(args.out) if args.out else Path("output") / f"siggraph_evidence_{args.suite}_{args.tier}"
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    manifest = {
        "suite": args.suite,
        "tier": args.tier,
        "tier_params": tier.__dict__,
        "runs": [],
    }
    for run in _suite_runs(args.suite):
        manifest["runs"].append(_run_evidence(run, tier, args, out_root))
        manifest["elapsed_sec"] = time.time() - t0
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _write_evidence_report(out_root, manifest)

    if args.suite in {"ablation", "phase_gate_stress"}:
        manifest["suite_media_manifest"] = str(
            _build_yz_media(out_root, tier, bool(args.dry_run))
        )

    manifest["elapsed_sec"] = time.time() - t0
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_evidence_report(out_root, manifest)
    print(f"\nEvidence report: {out_root / 'evidence_report.md'}", flush=True)


if __name__ == "__main__":
    main()
