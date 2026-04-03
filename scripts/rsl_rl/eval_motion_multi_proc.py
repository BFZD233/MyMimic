"""Run multiple motion evaluations by launching one eval process per motion.

This avoids Isaac runtime hangs observed when repeatedly creating environments
in a single process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-process motion batch evaluation")
    parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0")
    parser.add_argument("--motion_yaml", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_motions", type=int, default=8)
    parser.add_argument("--selection", type=str, choices=("head", "spaced", "random"), default="spaced")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--obs_mode", type=str, default="twist2_1432")
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--future_steps", type=str, default="0")
    parser.add_argument("--timeout_per_case", type=int, default=900, help="seconds")
    parser.add_argument(
        "--kit_args",
        type=str,
        default="--/app/hangDetector/enabled=false --/app/hangDetector/timeout=3600 "
        "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false "
        "--/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=1 --/physics/cudaDevice=1",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--livestream", type=int, default=0)
    return parser.parse_args()


def load_motion_entries(motion_yaml: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = yaml.safe_load(motion_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid motion yaml mapping: {motion_yaml}")
    root_path = payload.get("root_path", None)
    if not isinstance(root_path, str) or not root_path:
        raise ValueError(f"Invalid root_path in {motion_yaml}")
    motions = payload.get("motions", None)
    if not isinstance(motions, list) or len(motions) == 0:
        raise ValueError(f"No motions found in {motion_yaml}")

    entries: list[dict[str, Any]] = []
    root = Path(root_path).expanduser().resolve()
    for idx, item in enumerate(motions):
        if not isinstance(item, dict):
            continue
        file_rel = item.get("file", None)
        if not isinstance(file_rel, str) or not file_rel:
            continue
        file_abs = (root / file_rel).resolve()
        if not file_abs.is_file():
            continue
        entries.append(
            {
                "yaml_index": idx,
                "file_rel": file_rel,
                "file_abs": str(file_abs),
                "weight": float(item.get("weight", 1.0)),
                "description": str(item.get("description", "")),
            }
        )
    if not entries:
        raise ValueError(f"No valid existing motion files found from {motion_yaml}")
    return root_path, entries


def pick_entries(entries: list[dict[str, Any]], num_motions: int, selection: str, seed: int) -> list[dict[str, Any]]:
    num = max(1, min(num_motions, len(entries)))
    if num == len(entries):
        return entries
    if selection == "head":
        return entries[:num]
    if selection == "random":
        rng = random.Random(seed)
        chosen = rng.sample(entries, num)
        return sorted(chosen, key=lambda x: int(x["yaml_index"]))

    # spaced
    if num == 1:
        idxs = [0]
    else:
        idxs = [round(i * (len(entries) - 1) / float(num - 1)) for i in range(num)]
    dedup: list[int] = []
    for idx in idxs:
        if idx not in dedup:
            dedup.append(idx)
    while len(dedup) < num:
        for idx in range(len(entries)):
            if idx not in dedup:
                dedup.append(idx)
            if len(dedup) >= num:
                break
    return [entries[i] for i in dedup[:num]]


def write_case_motion_yaml(path: Path, root_path: str, entry: dict[str, Any]) -> None:
    payload = {
        "root_path": root_path,
        "motions": [
            {
                "file": entry["file_rel"],
                "weight": float(entry.get("weight", 1.0)),
                "description": str(entry.get("description", "")),
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    motion_yaml = Path(args.motion_yaml).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not motion_yaml.is_file():
        raise FileNotFoundError(f"motion_yaml does not exist: {motion_yaml}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt_path does not exist: {ckpt_path}")

    root_path, entries = load_motion_entries(motion_yaml)
    selected = pick_entries(entries, args.num_motions, args.selection, args.seed)
    (out_dir / "inputs").mkdir(exist_ok=True, parents=True)
    (out_dir / "cases").mkdir(exist_ok=True, parents=True)

    # Keep child Isaac processes on a consistent runtime-lib setup to avoid
    # picking incompatible base-conda libstdc++ and flaky matplotlib cache paths.
    child_env = os.environ.copy()
    env_root = Path(sys.executable).resolve().parent.parent
    env_lib = str(env_root / "lib")
    ld_prefix = f"{env_lib}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
    old_ld = child_env.get("LD_LIBRARY_PATH", "")
    child_env["LD_LIBRARY_PATH"] = f"{ld_prefix}:{old_ld}" if old_ld else ld_prefix
    if "VK_ICD_FILENAMES" not in child_env and Path("/etc/vulkan/icd.d/nvidia_icd.json").is_file():
        child_env["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    child_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    run_meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "task": args.task,
        "motion_yaml": str(motion_yaml),
        "ckpt_path": str(ckpt_path),
        "num_motions_requested": args.num_motions,
        "num_motions_selected": len(selected),
        "selection": args.selection,
        "seed": args.seed,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "device": args.device,
        "obs_mode": args.obs_mode,
        "history_len": args.history_len,
        "future_steps": args.future_steps,
        "timeout_per_case": args.timeout_per_case,
        "kit_args": args.kit_args,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=True), encoding="utf-8")
    (out_dir / "selected_motions.json").write_text(json.dumps(selected, indent=2, ensure_ascii=True), encoding="utf-8")

    case_results: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    for i, entry in enumerate(selected):
        case_name = f"{i:03d}__{Path(entry['file_rel']).name}"
        case_dir = out_dir / "cases" / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_yaml = out_dir / "inputs" / f"{i:03d}__motion.yaml"
        case_log = case_dir / "run.log"
        write_case_motion_yaml(case_yaml, root_path, entry)

        cmd = [sys.executable, "scripts/rsl_rl/eval_motion_batch.py"]
        if args.headless:
            cmd.append("--headless")
        cmd += [
            "--task",
            args.task,
            "--device",
            args.device,
            "--livestream",
            str(args.livestream),
            "--kit_args",
            args.kit_args,
            "--motion_yaml",
            str(case_yaml),
            "--num_motions",
            "1",
            "--selection",
            "head",
            "--steps",
            str(args.steps),
            "--warmup_steps",
            str(args.warmup_steps),
            "--obs_mode",
            args.obs_mode,
            "--history_len",
            str(args.history_len),
            "--future_steps",
            args.future_steps,
            "--ckpt_path",
            str(ckpt_path),
            "--output_dir",
            str(case_dir),
        ]

        start = time.time()
        status = "ok"
        return_code = 0
        error_msg = ""
        print(f"[INFO] [{i + 1}/{len(selected)}] case={case_name}", flush=True)
        with case_log.open("w", encoding="utf-8") as lf:
            lf.write(f"CMD: {' '.join(cmd)}\n\n")
            lf.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    env=child_env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_per_case,
                    check=False,
                )
                return_code = int(proc.returncode)
                if return_code != 0:
                    status = "failed"
                    error_msg = f"return_code={return_code}"
            except subprocess.TimeoutExpired:
                status = "timeout"
                return_code = 124
                error_msg = "timeout"

        elapsed = time.time() - start
        row: dict[str, Any] = {
            "case_index": i,
            "case_name": case_name,
            "yaml_index": int(entry["yaml_index"]),
            "file_rel": entry["file_rel"],
            "file_abs": entry["file_abs"],
            "status": status,
            "return_code": return_code,
            "elapsed_sec": round(elapsed, 3),
            "log_file": str(case_log),
            "case_dir": str(case_dir),
        }

        summary_json = case_dir / "summary.json"
        if status == "ok" and summary_json.is_file():
            try:
                payload = json.loads(summary_json.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    metrics = payload[0]
                    # Flatten selected numeric metrics into case summary.
                    for k in (
                        "mean_reward",
                        "mean_error_anchor_pos",
                        "mean_error_anchor_rot",
                        "mean_error_body_pos",
                        "mean_error_body_rot",
                        "mean_error_joint_pos",
                        "mean_error_joint_vel",
                        "done_events_post_warmup",
                    ):
                        if k in metrics:
                            row[k] = metrics[k]
                    success_rows.append(row.copy())
            except Exception as exc:
                row["status"] = "failed_parse_summary"
                row["error"] = str(exc)
        elif status != "ok":
            row["error"] = error_msg

        case_results.append(row)
        (out_dir / "case_status.json").write_text(json.dumps(case_results, indent=2, ensure_ascii=True), encoding="utf-8")

    write_csv(out_dir / "case_status.csv", case_results)
    write_csv(out_dir / "aggregate_summary.csv", success_rows)
    (out_dir / "aggregate_summary.json").write_text(json.dumps(success_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    headline: dict[str, Any] = {
        "num_cases": len(case_results),
        "num_success": len(success_rows),
        "num_failed": len(case_results) - len(success_rows),
    }
    if success_rows:
        denom = float(len(success_rows))
        case_denom = float(len(case_results))
        for k in (
            "mean_reward",
            "mean_error_anchor_pos",
            "mean_error_anchor_rot",
            "mean_error_body_pos",
            "mean_error_body_rot",
            "mean_error_joint_pos",
            "mean_error_joint_vel",
            "done_events_post_warmup",
        ):
            vals = [float(r[k]) for r in success_rows if k in r]
            if vals:
                headline[f"avg_{k}"] = sum(vals) / float(len(vals))
                headline[f"max_{k}"] = max(vals)
                headline[f"min_{k}"] = min(vals)
        headline["success_rate"] = float(len(success_rows)) / case_denom

    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[INFO] Done. Output: {out_dir}", flush=True)
    print(f"[INFO] Success: {headline['num_success']} / {headline['num_cases']}", flush=True)
    return 0 if headline["num_success"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
