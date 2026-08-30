#!/usr/bin/env python3
"""Run training or inference jobs from a JSONL manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "pls_train_classifier.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def add_value(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def training_command(spec: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(TRAIN_SCRIPT)]
    mapping = {
        "model_type": "--model_type",
        "model_size": "--model_size",
        "experiment_id": "--experiment_id",
        "training_regime": "--training_regime",
        "data_path": "--data_path",
        "num_samples": "--num_samples",
        "epochs": "--epochs",
        "batch_size": "--batch_size",
        "learning_rate": "--learning_rate",
        "num_warmup_steps": "--num_warmup_steps",
        "seed": "--seed",
        "seed_profile": "--seed_profile",
        "position_id_policy": "--position_id_policy",
        "pretrained_model_name_or_path": "--pretrained_model_name_or_path",
        "pretrained_model_revision": "--pretrained_model_revision",
        "output_root": "--output_root",
        "run_id": "--run_id",
        "run_name": "--run_name",
        "resume": "--resume",
        "strategy": "--strategy",
        "early_stopping_patience": "--early_stopping_patience",
    }
    for key, flag in mapping.items():
        add_value(command, flag, spec.get(key))
    if spec.get("use_test_as_val"):
        command.append("--use_test_as_val")
    if spec.get("ignore_pad_boundary_loss"):
        command.append("--ignore_pad_boundary_loss")
    return command


def display(command: list[str]) -> str:
    import shlex
    return shlex.join(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job-type", choices=["train", "inference"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl(Path(args.manifest))
    if args.job_type:
        jobs = [job for job in jobs if job.get("job_type") == args.job_type]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    failures = 0
    for index, spec in enumerate(jobs, start=1):
        job_type = spec.get("job_type")
        identifier = spec.get("run_id") or spec.get("job_id") or f"row-{index}"
        print(f"[{index}/{len(jobs)}] {identifier}")
        if job_type == "train":
            command = training_command(spec)
            print(display(command))
            if not args.dry_run:
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode:
                    failures += 1
        elif job_type == "inference":
            if args.dry_run:
                print("run inference job from manifest row")
            else:
                from pls_run_inference_context import run_inference_job
                try:
                    run_inference_job(spec)
                except Exception as error:
                    print(f"inference failed: {error}", file=sys.stderr)
                    failures += 1
        else:
            print(f"unsupported job type: {job_type}", file=sys.stderr)
            failures += 1
        if failures and not args.continue_on_error:
            return 1
    print(f"completed jobs={len(jobs)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
