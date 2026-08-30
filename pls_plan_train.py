#!/usr/bin/env python3
"""pls_plan_train.py : training checkpoint-grid planner with budget + dry-run.

Expands a config's checkpoint grid (datasets x model_types x model_sizes x seeds)
into training job specs and writes a manifest the GPU pool can run. Reuses the
already-tested `build_train_jobs` from `pls_plan_jobs` so the train-spec schema
never diverges from what `pls_worker.build_command` expects.

Adds what the old planner lacked: a hard job-count budget, `--explain-counts`,
`--dry-run`, and CLI filters so you can carve out a subset (e.g. the from-scratch
small/med/large grid) for an early training release stage while the modern/LoRA track is
still being built.

By default only *missing* jobs (no valid `_SUCCESS`) are written, so re-running
after a partial run adds only the remaining checkpoints.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

from pls.common import ensure_dir, load_yaml, write_json_atomic, write_jsonl_atomic
from pls_plan_jobs import build_train_jobs, is_complete


def _filtered(cfg: Dict[str, Any], sizes: List[str] | None, families: List[str] | None,
             datasets: List[str] | None) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    t = cfg.setdefault("training", {})
    if sizes:
        t["model_sizes"] = [s for s in t.get("model_sizes", []) if s in sizes] or sizes
    if families:
        t["model_types"] = [f for f in t.get("model_types", []) if f in families] or families
    if datasets:
        t["datasets"] = [d for d in t.get("datasets", []) if d in datasets] or datasets
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--write-manifest", help="Path to write the train manifest (JSONL)")
    ap.add_argument("--explain-counts", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-all", action="store_true", help="Write all jobs, not only missing")
    ap.add_argument("--max-jobs", type=int, default=None, help="Budget; default from config training.budget_max_jobs or 500")
    ap.add_argument("--override-budget", help="Reason to bypass the budget (recorded)")
    ap.add_argument("--sizes", help="CSV filter of model_sizes (e.g. small,med,large)")
    ap.add_argument("--families", help="CSV filter of model_types")
    ap.add_argument("--datasets", help="CSV filter of datasets")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = _filtered(
        cfg,
        args.sizes.split(",") if args.sizes else None,
        args.families.split(",") if args.families else None,
        args.datasets.split(",") if args.datasets else None,
    )
    t = cfg.get("training", {})
    root = Path(cfg["experiment"]["artifact_root"])
    budget = args.max_jobs or int(t.get("budget_max_jobs", 500))

    jobs = build_train_jobs(cfg)
    missing = [j for j in jobs if not is_complete(j)]

    grid = {
        "datasets": t.get("datasets", []),
        "model_types": t.get("model_types", []),
        "model_sizes": t.get("model_sizes", []),
        "seeds": t.get("seeds", [40]),
    }
    print("=" * 72)
    print("TRAIN CHECKPOINT GRID")
    print("=" * 72)
    print(f"grid            : {grid}")
    print(f"artifact_root   : {root}")
    print(f"total jobs      : {len(jobs)}")
    print(f"already complete: {len(jobs) - len(missing)}")
    print(f"missing (to run): {len(missing)}")
    print(f"budget (max)    : {budget}")

    to_write = jobs if args.write_all else missing
    if len(to_write) > budget:
        if args.override_budget:
            print(f"[BUDGET OVERRIDE] {len(to_write)} > {budget}  reason={args.override_budget!r}")
        else:
            print(f"[BUDGET EXCEEDED] {len(to_write)} > {budget} : refusing to write. "
                  f"Use --override-budget '<reason>' or --max-jobs.")
            return 3
    else:
        print("[BUDGET OK]")

    if args.explain_counts:
        return 0
    if args.dry_run:
        print("[DRY-RUN] manifest not written")
        return 0
    if not args.write_manifest:
        print("No --write-manifest given; nothing written.")
        return 0
    out = Path(args.write_manifest)
    ensure_dir(out.parent)
    write_jsonl_atomic(to_write, out)
    write_json_atomic(
        {"total": len(jobs), "missing": len(missing), "written": len(to_write),
         "grid": grid, "budget": budget, "override_reason": args.override_budget},
        out.parent / "train_plan_summary.json",
    )
    print(f"\nWrote {len(to_write)} train jobs -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
