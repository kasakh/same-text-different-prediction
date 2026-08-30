#!/usr/bin/env python3
"""pls_plan_posid_diag.py : GPT-2 padding/position-ID causal diagnostic planner.

The main-study padding finding (mean 0.10, max 0.82 flips, GPT-2 families only)
is mechanistically attributed to LEFT-padding + absolute position embeddings
with default position IDs: real-token positions depend on padded length.
This diagnostic makes that causal via intervention:

  position_id_policy = default       (arange over padded length; train-consistent)
  position_id_policy = mask_derived  (cumsum(attention_mask)-1; real tokens at 0..L-1
                                      regardless of padded length)

Grid: {pure-gen-GPT2, pseudo-gen-GPT2, BERT(control)} x med x seed42 x 5 datasets
      x {dynamic_longest, static_max_length} x {default, mask_derived}
      target-anchored, composition=random, position=first, 2000 stratified targets.

Predictions:
  * GPT-2 + default:      static-vs-dynamic flips ~0.10-0.24 (reproduces finding)
  * GPT-2 + mask_derived: static-vs-dynamic flips collapse to numeric floor (~1e-4)
  * BERT (right-padded):  ~0 everywhere (mask_derived == default for right padding)
  * accuracy per arm is recorded: mask_derived is train-INCONSISTENT for GPT-2
    (training saw real tokens at high positions), so accuracy may drop : that
    outcome redirects the mitigation claim to train/serve consistency.

Output root: artifacts_diag_position (separate from frozen paper roots).
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pls_plan_v2 import unique_checkpoints_from_manifest
from pls.common import make_job_id, write_jsonl_atomic as write_jsonl

FAMILIES = {"pure-gen-GPT2", "pseudo-gen-GPT2", "BERT"}
DATASETS = ["sst2", "sst5", "emotion", "ag_news", "imdb"]
PADDINGS = ["dynamic_longest", "static_max_length"]
POSID_POLICIES = ["default", "mask_derived"]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-from", help="Inference manifest containing checkpoint bindings")
    ap.add_argument("--train-root", default=None,
                    help="discover checkpoints from <root>/train/*/ instead of an inference manifest")
    ap.add_argument("--output-root", default="./outputs/position_diagnostic")
    ap.add_argument("--write-manifest", default=None)
    ap.add_argument("--reference-predictions", default=None,
                    help="Optional frozen reference_predictions.parquet path")
    ap.add_argument("--size", default="med")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-targets", type=int, default=2000)
    ap.add_argument("--posid-policies", default="default,mask_derived",
                    help="comma pair; use arange,mask_derived for consistency-trained checkpoints")
    ap.add_argument("--families", default=None,
                    help="comma list; default = the built-in family set")
    ap.add_argument("--no-ref-margins", action="store_true",
                    help="omit reference_predictions_path (label-only stratification)")
    args = ap.parse_args()

    if not args.train_root and not args.checkpoints_from:
        ap.error("provide --train-root or --checkpoints-from")

    if args.train_root:
        ckpts = []
        for td in sorted(Path(args.train_root, "train").iterdir()):
            if not (td / "_SUCCESS").exists():
                continue
            best = json.load(open(td / "best_checkpoint.json"))["best_model_path"]
            tc = json.load(open(td / "train_config.json"))
            ckpts.append({"train_run_id": td.name, "train_dir": str(td),
                          "checkpoint_path": best, "dataset": tc["data_path"],
                          "seed": int(tc.get("seed", -1))})
    else:
        ckpts = unique_checkpoints_from_manifest(args.checkpoints_from)
    families = set(args.families.split(",")) if args.families else FAMILIES
    policies = args.posid_policies.split(",")
    assert len(policies) == 2, "--posid-policies must be a comma pair"
    picked = []
    for c in ckpts:
        tc = json.load(open(Path(c["train_dir"]).resolve() / "train_config.json"))
        if (tc.get("model_type") in families and tc.get("model_size") == args.size
                and int(tc.get("seed", -1)) == args.seed and c["dataset"] in DATASETS):
            c["model_type"] = tc["model_type"]
            picked.append(c)
    assert len(picked) == len(families) * len(DATASETS), \
        f"expected {len(families)*len(DATASETS)} checkpoints, found {len(picked)}"

    jobs = []
    for c in picked:
        for pad in PADDINGS:
            for pol in policies:
                spec = {
                    "job_type": "inference", "output_root": args.output_root,
                    "train_run_id": c["train_run_id"], "train_dir": c["train_dir"],
                    "checkpoint_path": c["checkpoint_path"], "dataset": c["dataset"],
                    "split": "test", "mode": "target_anchored",
                    "composition_policy": "random", "batch_position": "first",
                    "padding_policy": pad, "position_id_policy": pol,
                    "batch_size": 8, "max_targets": args.max_targets,
                    "static_max_length": 512, "target_sampling": "stratified",
                    "reference_predictions_path": None if args.no_ref_margins else args.reference_predictions,
                    "dtype": "fp32", "quantization": "none", "device": "cuda",
                    "seed": 0, "seed_profile": "production_like",
                    "order_policy": {"name": "target_anchored", "type": "target_anchored"},
                    "runtime_name": "ta_fp32_none_cuda",
                    "shard_id": 0, "num_shards": 1, "limit": None,
                    "context_kind": "posid_diagnostic", "varied_axis": "padding_x_position_ids",
                    "model_type": c["model_type"],
                    "repeat_id": 0, "repeats": 1,
                }
                spec["job_id"] = make_job_id("infer", spec)
                jobs.append(spec)

    by_fam = {}
    for j in jobs:
        by_fam[j["model_type"]] = by_fam.get(j["model_type"], 0) + 1
    print(f"checkpoints: {len(picked)}  jobs: {len(jobs)}  by_family: {by_fam}")
    if args.write_manifest:
        write_jsonl(jobs, args.write_manifest)
        print(f"wrote {len(jobs)} jobs -> {args.write_manifest}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
