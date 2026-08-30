#!/usr/bin/env python3
"""Expand a YAML sweep into deterministic train and inference job specs.

Idempotency principle: job IDs are hashes of canonical job specs. Re-running the
planner after editing the config only adds genuinely new jobs; completed jobs are
recognized by artifact _SUCCESS markers.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pls.common import artifact_valid, ensure_dir, load_yaml, make_job_id, read_json, stable_hash, write_json_atomic, write_jsonl_atomic


def _product_dict(d: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(d.keys())
    for vals in itertools.product(*(d[k] for k in keys)):
        yield dict(zip(keys, vals))


def train_run_name(spec: Dict[str, Any]) -> str:
    ns = spec.get("num_samples")
    ns_tag = f"ns{ns}" if ns else "nsFull"
    return f"{spec.get('dataset', spec.get('data_path'))}_{spec['model_type']}_{spec['model_size']}_seed{spec['seed']}_{ns_tag}"


def build_train_jobs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = cfg["experiment"].get("artifact_root", "artifacts")
    tcfg = cfg.get("training", {})
    if not tcfg.get("enabled", True):
        return []
    train_batch_sizes = tcfg.get("train_batch_sizes", [tcfg.get("batch_size", 32)])
    if not tcfg.get("treat_train_batch_size_as_axis", False):
        train_batch_sizes = [train_batch_sizes[0]]
    grid = {
        "dataset": tcfg.get("datasets", []),
        "model_type": tcfg.get("model_types", []),
        "model_size": tcfg.get("model_sizes", []),
        "seed": tcfg.get("seeds", [40]),
        "num_samples": tcfg.get("num_samples", [None]),
        "batch_size": train_batch_sizes,
    }
    jobs = []
    for x in _product_dict(grid):
        spec = {
            "job_type": "train",
            "output_root": root,
            "data_path": x["dataset"],
            "model_type": x["model_type"],
            "model_size": x["model_size"],
            "seed": int(x["seed"]),
            "seed_profile": tcfg.get("seed_profile", "strict"),
            "num_samples": x.get("num_samples"),
            "epochs": int(tcfg.get("epochs", 20)),
            "batch_size": int(x["batch_size"]),
            "learning_rate": float(tcfg.get("learning_rate", 2e-5)),
            "num_warmup_steps": int(tcfg.get("num_warmup_steps", 0)),
            "use_test_as_val": bool(tcfg.get("use_test_as_val", False)),
            "resume": tcfg.get("resume", "auto"),
            "strategy": tcfg.get("strategy", "auto"),
            "early_stopping_patience": tcfg.get("early_stopping_patience"),
        }
        spec["run_name"] = train_run_name(spec)
        spec["run_id"] = make_job_id("train", spec)
        jobs.append(spec)
    return jobs


def _pick_checkpoint(train_dir: Path, selection: str) -> Optional[str]:
    if not train_dir.exists():
        return None
    if selection == "last":
        p = train_dir / "checkpoints" / "last.ckpt"
        return str(p) if p.exists() else None
    best_json = train_dir / "best_checkpoint.json"
    if best_json.exists():
        p = read_json(best_json).get("best_model_path")
        if p and Path(p).exists():
            return p
    ckpt_dir = train_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = [p for p in ckpt_dir.glob("*.ckpt") if not p.name.startswith("last")]
    if not candidates:
        return None
    def wf1(path: Path) -> float:
        m = re.search(r"val_weighted_f1=([\d.]+)", path.name)
        return float(m.group(1)) if m else -1.0
    return str(max(candidates, key=wf1))


def build_inference_jobs(cfg: Dict[str, Any], train_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    root = cfg["experiment"].get("artifact_root", "artifacts")
    icfg = cfg.get("inference", {})
    if not icfg.get("enabled", True):
        return []
    checkpoint_selection = icfg.get("checkpoints", {}).get("selection", "best_val_weighted_f1")
    order_policies = icfg.get("order_policies", [{"name": "original", "type": "identity"}])
    jobs = []
    for train in train_jobs:
        train_dir = Path(root) / "train" / train["run_id"]
        ckpt = _pick_checkpoint(train_dir, checkpoint_selection)
        if not ckpt and not icfg.get("plan_without_checkpoint", False):
            continue
        grid = {
            "split": icfg.get("splits", ["test"]),
            "batch_size": icfg.get("batch_sizes", [1]),
            "order_policy": order_policies,
            "runtime": icfg.get("runtimes", [{"name": "pytorch_fp32", "backend": "pytorch", "dtype": "fp32", "quantization": "none", "seed_profile": "production_like"}]),
            "shard_id": list(range(int(icfg.get("num_shards", 1)))),
        }
        for x in _product_dict(grid):
            runtime = x["runtime"]
            if runtime.get("backend", "pytorch") != "pytorch":
                continue
            spec = {
                "job_type": "inference",
                "output_root": root,
                "train_run_id": train["run_id"],
                "train_dir": str(train_dir),
                "checkpoint_path": ckpt or "",
                "dataset": train["data_path"],
                "split": x["split"],
                "batch_size": int(x["batch_size"]),
                "order_policy": x["order_policy"],
                "dtype": runtime.get("dtype", "fp32"),
                "quantization": runtime.get("quantization", "none"),
                "device": runtime.get("device"),
                "seed": int(runtime.get("seed", train.get("seed", 0))),
                "seed_profile": runtime.get("seed_profile", "production_like"),
                "runtime_name": runtime.get("name", "pytorch"),
                "shard_id": int(x["shard_id"]),
                "num_shards": int(icfg.get("num_shards", 1)),
                "limit": icfg.get("limit"),
            }
            spec["job_id"] = make_job_id("infer", spec)
            jobs.append(spec)
    return jobs


def is_complete(spec: Dict[str, Any]) -> bool:
    root = Path(spec["output_root"])
    if spec["job_type"] == "train":
        return artifact_valid(root / "train" / spec["run_id"], ["train_config.json"])
    if spec["job_type"] == "inference":
        return artifact_valid(root / "inference" / spec["job_id"], ["predictions.jsonl", "job_spec.json"])
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--write_all", action="store_true", help="Write all jobs, not only missing")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = Path(cfg["experiment"].get("artifact_root", "artifacts"))
    manifest_dir = root / "manifests"
    ensure_dir(manifest_dir)
    train_jobs = build_train_jobs(cfg)
    infer_jobs = build_inference_jobs(cfg, train_jobs)
    if args.write_all:
        train_out, infer_out = train_jobs, infer_jobs
    else:
        train_out = [j for j in train_jobs if not is_complete(j)]
        infer_out = [j for j in infer_jobs if not is_complete(j)]
    write_jsonl_atomic(train_out, manifest_dir / "desired_train_jobs.jsonl")
    write_jsonl_atomic(infer_out, manifest_dir / "desired_inference_jobs.jsonl")
    write_json_atomic(
        {
            "train_total": len(train_jobs),
            "train_missing": len(train_out),
            "inference_total_plannable": len(infer_jobs),
            "inference_missing": len(infer_out),
            "config": args.config,
        },
        manifest_dir / "plan_summary.json",
    )
    print(f"Train jobs: total={len(train_jobs)} missing={len(train_out)}")
    print(f"Inference jobs: plannable={len(infer_jobs)} missing={len(infer_out)}")
    print(f"Wrote manifests to {manifest_dir}")


if __name__ == "__main__":
    main()
