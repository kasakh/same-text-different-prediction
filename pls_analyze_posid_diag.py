#!/usr/bin/env python3
"""pls_analyze_posid_diag.py : analysis for the padding/position-ID causal diagnostic.

Strictness contract :
  * ALL manifest jobs must have _SUCCESS (no silent continuation); --allow-partial
    downgrades to a warning for exploratory peeking only.
  * All 4 arms of a cell (2 padding x 2 posid) must exist with IDENTICAL target
    sets and IDENTICAL peer_example_ids per target : otherwise the comparison is
    confounded and the cell is REJECTED (hard error without --allow-partial).

Reported per (family, dataset):
  * padding flip rate (static vs dynamic) per posid policy, with bootstrap 95% CI
  * posid flip rate (default vs mask_derived) per padding policy  [label drift]
  * probability drift (mean TV distance) and logit drift (mean max|delta|)
  * accuracy per arm (4 arms)
  * all of the above restricted to targets whose dynamic/static padded lengths
    actually differ (the only targets where the mechanism CAN act)

Causal claim test: flips(GPT2, default) >> flips(GPT2, mask_derived) ~= numeric floor.
Control: flips(BERT, *) ~= 0 and accuracy unchanged.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

def make_arms(pol_a, pol_b):
    return [("static_max_length", pol_a), ("dynamic_longest", pol_a),
            ("static_max_length", pol_b), ("dynamic_longest", pol_b)]


def load_run(run_dir):
    rows = {}
    with open(run_dir / "predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[r["example_id"]] = r
    return rows


def bootstrap_ci(flips_bool, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a flip-rate over per-target booleans."""
    if not flips_bool:
        return (float("nan"), float("nan"))
    if sum(flips_bool) == 0:
        # Zero-flip cells: bootstrap of all-zeros degenerately yields [0,0], which
        # is wrong for population inference. Use the exact binomial (Clopper-
        # Pearson) upper bound: for x=0, upper = 1-(alpha/2)^(1/n) (~3.69/n @95%).
        # The empirical estimate remains 0.
        n = len(flips_bool)
        return (0.0, 1.0 - (alpha / 2) ** (1.0 / n))
    rng = random.Random(seed)
    n = len(flips_bool)
    stats = []
    for _ in range(n_boot):
        s = sum(flips_bool[rng.randrange(n)] for _ in range(n))
        stats.append(s / n)
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (lo, hi)


def tv_distance(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def max_logit_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def pair_stats(rows_a, rows_b, ids, tag, seed=0):
    """Flip rate + CI + prob/logit drift between two arms over given target ids."""
    if not ids:
        return {"tag": tag, "n": 0}
    flips = [rows_a[e]["pred_class_index"] != rows_b[e]["pred_class_index"] for e in ids]
    fr = sum(flips) / len(flips)
    lo, hi = bootstrap_ci(flips, seed=seed)
    tv = sum(tv_distance(rows_a[e]["probs"], rows_b[e]["probs"]) for e in ids) / len(ids)
    ld = sum(max_logit_diff(rows_a[e]["raw_logits"], rows_b[e]["raw_logits"]) for e in ids) / len(ids)
    return {"tag": tag, "n": len(ids), "flip_rate": fr, "flip_ci_low": lo, "flip_ci_high": hi,
            "mean_tv_prob_drift": tv, "mean_max_logit_drift": ld}


def accuracy(rows, ids):
    return sum(1 for e in ids if rows[e]["pred_class_index"] == rows[e]["true_class_index"]) / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./outputs/position_diagnostic")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--policies", default="default,mask_derived",
                    help="comma pair: naive_policy,intervention_policy (e.g. arange,mask_derived)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="exploratory peeking only: tolerate missing jobs / cells (results NOT causal-grade)")
    args = ap.parse_args()

    pol_a, pol_b = args.policies.split(",")
    ARMS = make_arms(pol_a, pol_b)
    specs = [json.loads(l) for l in open(args.manifest)]
    inf_root = Path(args.root) / "inference"

    # ---- gate 1: ALL jobs succeeded ----
    missing = [s["job_id"] for s in specs if not (inf_root / s["job_id"] / "_SUCCESS").exists()]
    if missing:
        msg = f"{len(missing)}/{len(specs)} jobs missing _SUCCESS (e.g. {missing[:5]})"
        if not args.allow_partial:
            print(f"FAIL completeness gate: {msg}\nRerun the manifest or pass --allow-partial to peek.", file=sys.stderr)
            return 2
        print(f"WARNING (allow-partial): {msg}", file=sys.stderr)

    cells = defaultdict(dict)  # (family, dataset) -> {(padding, posid): rows}
    for s in specs:
        rd = inf_root / s["job_id"]
        if not (rd / "_SUCCESS").exists():
            continue
        cells[(s["model_type"], s["dataset"])][(s["padding_policy"], s["position_id_policy"])] = load_run(rd)

    # ---- gate 2: per-cell arm completeness + identical targets + identical peers ----
    valid_cells = {}
    for key, arms in sorted(cells.items()):
        fam, ds = key
        problems = []
        if set(arms) != set(ARMS):
            problems.append(f"arms present: {sorted(arms)}")
        else:
            id_sets = {a: set(r) for a, r in arms.items()}
            base = id_sets[ARMS[0]]
            if any(s != base for s in id_sets.values()):
                sizes = {a: len(s) for a, s in id_sets.items()}
                problems.append(f"target sets differ across arms: {sizes}")
            else:
                ref_arm = arms[ARMS[0]]
                for a in ARMS[1:]:
                    bad = [e for e in base
                           if arms[a][e].get("peer_example_ids") != ref_arm[e].get("peer_example_ids")]
                    if bad:
                        problems.append(f"peer sets differ vs {a} for {len(bad)} targets (e.g. {bad[:3]})")
                        break
        if problems:
            msg = f"cell ({fam},{ds}) REJECTED: {'; '.join(problems)}"
            if not args.allow_partial:
                print(f"FAIL isolation gate: {msg}", file=sys.stderr)
                return 2
            print(f"WARNING (allow-partial): {msg}", file=sys.stderr)
            continue
        valid_cells[key] = arms

    out = {"cells": [], "gates": {"jobs_total": len(specs), "jobs_missing": len(missing),
                                  "cells_valid": len(valid_cells), "cells_total": len(cells),
                                  "causal_grade": not missing and len(valid_cells) == len(cells)}}

    hdr = (f"{'family':<17}{'dataset':<9}{'comparison':<34}{'n':>5} {'flip':>8} "
           f"{'ci95':>17} {'tv_prob':>8} {'logitD':>8}")
    print(hdr)
    print("-" * len(hdr))
    for (fam, ds), arms in sorted(valid_cells.items()):
        ids_all = sorted(arms[ARMS[0]])
        # targets where the mechanism CAN act: dynamic padded length != static (under default)
        st, dy = arms[("static_max_length", pol_a)], arms[("dynamic_longest", pol_a)]
        ids_diff = [e for e in ids_all
                    if st[e]["padded_sequence_length"] != dy[e]["padded_sequence_length"]]

        cell = {"family": fam, "dataset": ds, "n_targets": len(ids_all),
                "n_targets_padlen_differs": len(ids_diff),
                "accuracy": {f"{p}/{q}": accuracy(arms[(p, q)], ids_all) for p, q in ARMS},
                "comparisons": []}
        comps = [
            # padding axis (the finding), per posid policy
            (("static_max_length", pol_a), ("dynamic_longest", pol_a), f"padding | posid={pol_a}"),
            (("static_max_length", pol_b), ("dynamic_longest", pol_b), f"padding | posid={pol_b}"),
            # posid axis (label drift of the intervention itself), per padding
            (("static_max_length", pol_a), ("static_max_length", pol_b), "posid | padding=static"),
            (("dynamic_longest", pol_a), ("dynamic_longest", pol_b), "posid | padding=dynamic"),
        ]
        for a, b, tag in comps:
            for subset, ids in (("all", ids_all), ("padlen_differs", ids_diff)):
                st_ = pair_stats(arms[a], arms[b], ids, f"{tag} [{subset}]")
                cell["comparisons"].append(st_)
                if st_["n"]:
                    print(f"{fam:<17}{ds:<9}{st_['tag']:<34}{st_['n']:>5} {st_['flip_rate']:>8.5f} "
                          f"[{st_['flip_ci_low']:.4f},{st_['flip_ci_high']:.4f}] "
                          f"{st_['mean_tv_prob_drift']:>8.5f} {st_['mean_max_logit_drift']:>8.3f}")
        out["cells"].append(cell)

    # ---- verdict block ----
    def agg(fam, tag_prefix):
        rs = [c1["flip_rate"] for c in out["cells"] if c["family"] == fam
              for c1 in c["comparisons"] if c1.get("n") and c1["tag"].startswith(tag_prefix) and c1["tag"].endswith("[all]")]
        return sum(rs) / len(rs) if rs else None

    print("\nVERDICT (mean padding flip rate across datasets, all targets):")
    for fam in ("pure-gen-GPT2", "pseudo-gen-GPT2", "BERT"):
        d = agg(fam, f"padding | posid={pol_a}")
        m = agg(fam, f"padding | posid={pol_b}")
        if d is None or m is None:
            continue
        reduction = "n/a" if d == 0 else f"{(1 - m / d) * 100:.1f}%"
        print(f"  {fam:<17} default={d:.5f}  mask_derived={m:.5f}  reduction={reduction}")
    print("\nACCURACY (watch mask_derived arms for train-inconsistency cost):")
    for c in out["cells"]:
        acc = c["accuracy"]
        print(f"  {c['family']:<17}{c['dataset']:<9} " +
              "  ".join(f"{k}={v:.4f}" for k, v in acc.items()))
    if not out["gates"]["causal_grade"]:
        print("\nNOTE: --allow-partial run : NOT causal-grade; rerun full manifest for claims.")

    dst = Path(args.root) / "posid_diag_results.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
