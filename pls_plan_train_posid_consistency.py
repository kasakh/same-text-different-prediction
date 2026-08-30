#!/usr/bin/env python3
"""pls_plan_train_posid_consistency.py : consistency-training experiment planner.

Trains GPT-2-family checkpoints with position_id_policy=mask_derived at TRAIN
time (module-internal, so inference inherits it). Closes the causal loop opened
by the position-ID diagnostic with a falsifiable prediction: on these
checkpoints, mask-derived serving shows flips=0 AND accuracy recovered to the
static/default level of conventionally-trained models; the explicit "arange"
serving arm becomes the unstable one.

Grid: {pure-gen-GPT2, pseudo-gen-GPT2} x med x seed 42 x 5 datasets = 10 runs.
Output root: artifacts_posid_consistency (separate root).
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pls.common import make_job_id, write_jsonl_atomic

DATASETS = ["sst2", "sst5", "emotion", "ag_news", "imdb"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="./outputs/position_consistency")
    ap.add_argument("--write-manifest", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    jobs = []
    for mtype in ("pure-gen-GPT2", "pseudo-gen-GPT2"):
        for ds in DATASETS:
            spec = {
                "job_type": "train", "model_type": mtype, "model_size": "med",
                "data_path": ds, "dataset": ds, "epochs": 60, "batch_size": 32,
                "learning_rate": 2e-5, "num_warmup_steps": 0, "seed": args.seed,
                "seed_profile": "strict", "num_samples": None,
                "early_stopping_patience": 10, "use_test_as_val": False,
                "position_id_policy": "mask_derived",
                "output_root": args.output_root,
            }
            spec["run_id"] = "train_" + make_job_id("train", spec)[6:]
            spec["run_name"] = f"{ds}_{mtype}_med_seed{args.seed}_maskpos"
            jobs.append(spec)
    print(f"jobs: {len(jobs)}")
    if args.write_manifest:
        write_jsonl_atomic(jobs, args.write_manifest)
        print(f"wrote {len(jobs)} -> {args.write_manifest}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
