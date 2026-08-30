#!/usr/bin/env python3
"""pls_metrics_v2.py : metrics pipeline: frozen reference-margin + comparison-policy metrics.

Reads completed inference contexts (each an `inference/<job_id>/{job_spec.json,
predictions.jsonl}` dir), then:

1. Selects and FREEZES one reference context per (train_run_id, dataset, split)
   and writes `reference_predictions.parquet` with the frozen `reference_margin`.
2. Classifies context PAIRS by the metric-comparison policy (exactly one serving
   axis differs, the rest held fixed per `metric_comparison_policy.md`) into:
   batch_invariance / order_invariance / precision_drift / quantization_drift /
   hardware_drift  (position/composition/padding are handled once those axes
   exist in the specs : the classifier already keys on them).
3. Computes per-pair stability metrics: flip_rate (+bootstrap CI), n_flips,
   mean_prob_drift, mean_margin_drift, mean_max_abs_logit_diff, and McNemar's
   test on correctness. Writes one parquet per comparison family + an
   `all_pairwise_diagnostics.parquet`.
4. Computes `flip_by_margin.parquet` (flip rate of each context vs its frozen
   reference, bucketed by the FROZEN reference margin : the H2 mechanism).
5. Computes `context_metrics.parquet` (per-context accuracy / macro-F1 /
   weighted-F1 / mean-confidence / ECE).

Invariants (acceptance): fp32-vs-bf16 appears ONLY in precision_drift; batch/order
comparisons hold dtype+quant+device+backend+seed fixed; every comparison aligns on
`example_id`, never row order; margin buckets use the FROZEN reference margin.

Rerun-safe: writes to <artifacts_root>/metrics_v2/<metrics_id>/ with _SUCCESS.
"""
from __future__ import annotations

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

from pls.common import ensure_dir, mark_success, read_json, read_jsonl, write_json_atomic

# axis keys used to identify a context
BASE_KEYS = ["train_run_id", "dataset", "split"]
# A valid single-axis comparison differs on EXACTLY one of these and matches on all
# the rest (base + every other serving/runtime axis, INCLUDING seed & seed_profile).
SERVING_AXES = ["batch_size", "order_name", "batch_position", "composition_policy", "padding_policy"]
RUNTIME_AXES = ["dtype", "quantization", "device", "backend", "execution_profile", "seed_profile", "seed",
                "position_id_policy", "repeat_id"]
# position_id_policy / repeat_id have NO family mapping: pairs differing on them
# are correctly excluded from single-axis families (analyzed by the dedicated
# posid diagnostic / repeat tooling), never silently misclassified.
ALL_AXES = SERVING_AXES + RUNTIME_AXES
AXIS_TO_FAMILY = {
    "batch_size": "batch_invariance", "order_name": "order_invariance",
    "batch_position": "position_invariance", "composition_policy": "composition_invariance",
    "padding_policy": "padding_invariance",
    "dtype": "precision_drift", "quantization": "quantization_drift",
    "device": "hardware_drift", "backend": "backend_drift",
    "execution_profile": "determinism_drift",
}
# recognized multi-axis bundles: reported as their OWN family, never single-axis
BUNDLE_TO_FAMILY = {frozenset({"quantization", "device"}): "deployment_profile_drift"}

MARGIN_BUCKETS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 1.01)]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _norm_device(d: Any) -> str:
    d = (str(d) if d is not None else "").lower()
    if d == "" or d.startswith("cuda"):
        return "cuda"
    return d


def load_contexts(artifacts_root: Path, limit_checkpoints: Optional[int] = None
                  ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Return (contexts_df, per_context_predictions).

    contexts_df: one row per context (axes + context_id + n_examples).
    per_context_predictions: context_id -> DataFrame[example_id, pred_label,
        true_label, top1_prob, top2_prob, margin, probs, raw_logits, num_labels].
    """
    inf_root = artifacts_root / "inference"
    ctx_rows: List[Dict[str, Any]] = []
    preds: Dict[str, pd.DataFrame] = {}
    allowed_ckpts: Optional[set] = None

    dirs = sorted(p for p in inf_root.iterdir() if p.is_dir())
    for d in dirs:
        spec_p = d / "job_spec.json"
        pred_p = d / "predictions.jsonl"
        succ = d / "_SUCCESS"
        if not (spec_p.exists() and pred_p.exists() and succ.exists()):
            continue
        spec = read_json(spec_p)
        trid = spec.get("train_run_id")
        if limit_checkpoints is not None:
            if allowed_ckpts is None:
                allowed_ckpts = set()
            if trid not in allowed_ckpts and len(allowed_ckpts) >= limit_checkpoints:
                continue
            allowed_ckpts.add(trid)

        op = spec.get("order_policy", {}) or {}
        cid = spec.get("job_id") or d.name
        rows = list(read_jsonl(pred_p))
        if not rows:
            continue
        pdf = pd.DataFrame(rows)
        keep = [c for c in ["example_id", "pred_label", "true_label", "top1_prob",
                            "top2_prob", "margin", "probs", "raw_logits", "num_labels"] if c in pdf.columns]
        pdf = pdf[keep].copy()
        pdf["example_id"] = pdf["example_id"].astype(str)
        pdf = pdf.drop_duplicates("example_id").set_index("example_id")
        preds[cid] = pdf

        ctx_rows.append({
            "context_id": cid,
            "train_run_id": trid,
            "dataset": spec.get("dataset"),
            "split": spec.get("split", "test"),
            "batch_size": spec.get("batch_size"),
            "order_name": op.get("name", IDENTITY_ORDER_NAME),
            "order_type": op.get("type", "identity"),
            "dtype": spec.get("dtype", "fp32"),
            "quantization": spec.get("quantization", "none"),
            "device": _norm_device(spec.get("device")),
            "backend": spec.get("backend", "pytorch"),
            "execution_profile": spec.get("execution_profile"),
            "seed_profile": spec.get("seed_profile", "production_like"),
            "seed": spec.get("seed", 0),
            # new axes (present only once the inference script emits them)
            "batch_position": spec.get("batch_position"),
            "composition_policy": spec.get("composition_policy"),
            "padding_policy": spec.get("padding_policy"),
            "position_id_policy": spec.get("position_id_policy", "default"),
            "repeat_id": spec.get("repeat_id", 0),
            "context_kind": spec.get("context_kind"),
            "n_examples": len(pdf),
        })
    return pd.DataFrame(ctx_rows), preds


# --------------------------------------------------------------------------- #
# reference selection + frozen margin
# --------------------------------------------------------------------------- #
# Canonical name for the identity (no-reordering) order policy. The paper anchor
# pins order_name == this value everywhere. Older code defaulted a nameless
# identity policy to "original", which silently degrades otherwise-valid groups.
IDENTITY_ORDER_NAME = "identity"

DEFAULT_REFERENCE_CELL = {
    "dtype": "fp32", "quantization": "none", "device": "cuda",
    "backend": "pytorch", "order_type": "identity", "order_name": IDENTITY_ORDER_NAME,
    "batch_size": 32, "seed_profile": "production_like", "reference_repeat_id": 0,
}


def select_reference(group: pd.DataFrame, ref_cell: Dict[str, Any], exploratory: bool) -> Tuple[Optional[str], str]:
    """Return (context_id, status) for one (train_run_id, dataset, split) group.

    STRICT (paper) mode: the anchor must match EXACTLY ONE context.
      * 0 matches -> (None, "no_reference")
      * >1 matches -> (None, "ambiguous")  : we REFUSE to silently tie-break by
        context_id; pin more axes (seed/repeat/order_name) to disambiguate.
    EXPLORATORY mode: tie-break or nearest-batch fallback is allowed (status says
    which). `reference_repeat_id` matches the `repeat_id` column when present.
    """
    m = group
    for k, v in ref_cell.items():
        col = "repeat_id" if k == "reference_repeat_id" else k
        if col in group.columns:
            m = m[m[col] == v]
    if len(m) == 1:
        return (m.iloc[0]["context_id"], "ok")
    if len(m) > 1:
        if not exploratory:
            return (None, "ambiguous")
        return (m.sort_values("context_id").iloc[0]["context_id"], "ok_exploratory_tiebreak")
    if not exploratory:
        return (None, "no_reference")
    cand = group[(group["dtype"] == "fp32") & (group["quantization"] == "none")
                 & (group["device"] == "cuda") & (group["order_type"] == "identity")]
    if cand.empty:
        cand = group
    cand = cand.copy()
    cand["batch_dist"] = (cand["batch_size"].fillna(10**9).astype(float) - ref_cell.get("batch_size", 32)).abs()
    return (cand.sort_values(["batch_dist", "context_id"]).iloc[0]["context_id"], "ok_exploratory_fallback")


def _degraded_diag(grp: pd.DataFrame, ref_cell: Dict[str, Any], key) -> Dict[str, Any]:
    """Diagnostic for a group with no exact reference: which anchor fields could not
    be matched (required vs available) and what anchor combos DO exist, so the
    planner/config can be fixed and the group re-run."""
    unmatched = {}
    for k, v in ref_cell.items():
        if k in grp.columns:
            present = set(grp[k].dropna().tolist())
            if v not in present:
                unmatched[k] = {"required": v, "available": sorted(str(x) for x in present)}
    anchor_fields = [k for k in ref_cell if k in grp.columns]
    combos = grp[anchor_fields].astype(str).drop_duplicates().to_dict("records") if anchor_fields else []
    return {
        "train_run_id": key[0], "dataset": key[1], "split": key[2],
        "n_candidate_contexts": int(len(grp)),
        "unmatched_anchor_fields": unmatched,
        "available_anchor_combos": combos,
        "candidate_context_ids": [str(x) for x in grp["context_id"].tolist()[:50]],
    }


def build_reference(contexts: pd.DataFrame, preds: Dict[str, pd.DataFrame],
                    ref_cell: Dict[str, Any], exploratory: bool
                    ) -> Tuple[pd.DataFrame, Dict[Tuple, Optional[str]], List[List[Any]]]:
    """Freeze one reference prediction per (ckpt, dataset, split). Groups without an
    exact reference (strict mode) are marked degraded (ref = None) and excluded from
    reference_predictions / drift / flip-by-margin."""
    ref_map: Dict[Tuple, Optional[str]] = {}
    ref_rows: List[pd.DataFrame] = []
    degraded: List[List[Any]] = []
    for key, grp in contexts.groupby(BASE_KEYS):
        cid, status = select_reference(grp, ref_cell, exploratory)
        ref_map[tuple(key)] = cid
        if cid is None:
            d = _degraded_diag(grp, ref_cell, key)
            d["status"] = status
            degraded.append(d)
            continue
        p = preds[cid]
        ref_rows.append(pd.DataFrame({
            "train_run_id": key[0], "dataset": key[1], "split": key[2],
            "example_id": p.index,
            "reference_context_id": cid,
            "reference_predicted_label": p["pred_label"].values,
            "reference_top1_probability": p["top1_prob"].values,
            "reference_top2_probability": p["top2_prob"].values,
            "reference_margin": p["margin"].values,
            "reference_correct": (p["pred_label"].values == p["true_label"].values),
            "num_labels": p["num_labels"].values if "num_labels" in p else np.nan,
        }))
    ref_df = pd.concat(ref_rows, ignore_index=True) if ref_rows else pd.DataFrame()
    return ref_df, ref_map, degraded


# --------------------------------------------------------------------------- #
# pair classification (faithful to metric_comparison_policy.md)
# --------------------------------------------------------------------------- #
def _same(a, b, keys) -> bool:
    return all(a[k] == b[k] for k in keys)


def _diff(a, b, keys) -> bool:
    return all(a[k] != b[k] for k in keys)


def _axis_eq(x, y) -> bool:
    if x == y:
        return True
    try:
        return bool(pd.isna(x) and pd.isna(y))
    except (TypeError, ValueError):
        return False


def classify_pair(a: pd.Series, b: pd.Series) -> Optional[Tuple[str, str]]:
    """Generic single-axis classifier.

    A pair is valid for a family iff EXACTLY ONE axis differs (and it maps to a
    family); every other base/serving/runtime axis : including seed & seed_profile
    : must match. A recognized multi-axis bundle (currently {quantization, device})
    is reported as its own bundled family (deployment_profile_drift), NEVER as
    single-axis drift. Everything else (>1 unrecognized diff, or a diff on a
    hold-fixed-only axis like seed) returns None and is excluded.
    """
    if not _same(a, b, BASE_KEYS):
        return None
    diffs = {ax for ax in ALL_AXES if not _axis_eq(a.get(ax), b.get(ax))}
    if not diffs:
        return None
    if len(diffs) == 1:
        ax = next(iter(diffs))
        fam = AXIS_TO_FAMILY.get(ax)
        if fam is None:
            return None  # e.g. a pure seed / seed_profile difference : not a studied axis
        return (fam, "order_policy" if ax == "order_name" else ax)
    fam = BUNDLE_TO_FAMILY.get(frozenset(diffs))
    if fam is not None:
        return (fam, "+".join(sorted(diffs)))
    return None


def _axis_val(rec, axis: str) -> str:
    if axis == "order_policy":
        return str(rec.get("order_name"))
    if "+" in axis:
        return "|".join(f"{ax}={rec.get(ax)}" for ax in axis.split("+"))
    return str(rec.get(axis))


# --------------------------------------------------------------------------- #
# per-pair metrics
# --------------------------------------------------------------------------- #
def _probs_matrix(s: pd.Series) -> np.ndarray:
    return np.array([np.asarray(x, dtype=float) for x in s.values])


def bootstrap_flip_ci(flips: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    if n_boot <= 0 or len(flips) == 0:
        return (float("nan"), float("nan"))
    n = len(flips)
    idx = rng.integers(0, n, size=(n_boot, n))
    rates = flips[idx].mean(axis=1)
    return (float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)))


def mcnemar_correctness(a_correct: np.ndarray, b_correct: np.ndarray) -> Tuple[float, float]:
    """SECONDARY evidence: did context A vs B change WHICH examples are right/wrong?
    (Not a test of label instability.)"""
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    if b + c == 0:
        return (0.0, 1.0)
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
    return (float(stat), float(chi2.sf(stat, df=1)))


def mcnemar_label_binary(pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[float, float]:
    """Binary label-direction McNemar: is label switching between A and B
    directionally asymmetric? (0->1 vs 1->0)."""
    labs = sorted(set(pred_a.tolist()) | set(pred_b.tolist()))
    if len(labs) != 2:
        return (float("nan"), float("nan"))
    x, y = labs
    n_xy = int(np.sum((pred_a == x) & (pred_b == y)))
    n_yx = int(np.sum((pred_a == y) & (pred_b == x)))
    if n_xy + n_yx == 0:
        return (0.0, 1.0)
    stat = (abs(n_xy - n_yx) - 1) ** 2 / (n_xy + n_yx)
    return (float(stat), float(chi2.sf(stat, df=1)))


def bowker_symmetry(pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[float, float, int]:
    """Multiclass Bowker test of symmetry (generalizes McNemar to k x k): is the
    label-change contingency table symmetric (no systematic directional shift)?"""
    labs = sorted(set(pred_a.tolist()) | set(pred_b.tolist()))
    k = len(labs)
    if k < 2:
        return (float("nan"), float("nan"), 0)
    idx = {l: i for i, l in enumerate(labs)}
    N = np.zeros((k, k), dtype=float)
    for pa, pb in zip(pred_a.tolist(), pred_b.tolist()):
        N[idx[pa], idx[pb]] += 1
    stat, df = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            s = N[i, j] + N[j, i]
            if s > 0:
                stat += (N[i, j] - N[j, i]) ** 2 / s
                df += 1
    if df == 0:
        return (0.0, 1.0, 0)
    return (float(stat), float(chi2.sf(stat, df)), df)


def pair_metrics(pa: pd.DataFrame, pb: pd.DataFrame, n_boot: int,
                 rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    common = pa.index.intersection(pb.index)
    if len(common) == 0:
        return None
    a = pa.loc[common]
    b = pb.loc[common]
    pred_a = a["pred_label"].values
    pred_b = b["pred_label"].values
    flips = (pred_a != pred_b).astype(float)
    prob_a = _probs_matrix(a["probs"])
    prob_b = _probs_matrix(b["probs"])
    prob_drift = np.max(np.abs(prob_a - prob_b), axis=1)
    margin_drift = np.abs(a["margin"].values - b["margin"].values)
    if "raw_logits" in a and "raw_logits" in b:
        la = _probs_matrix(a["raw_logits"]); lb = _probs_matrix(b["raw_logits"])
        logit_drift = float(np.mean(np.max(np.abs(la - lb), axis=1)))
    else:
        logit_drift = float("nan")
    a_correct = (pred_a == a["true_label"].values)
    b_correct = (pred_b == b["true_label"].values)
    # PRIMARY instability: flip rate + bootstrap CI.
    ci_lo, ci_hi = bootstrap_flip_ci(flips, n_boot, rng)
    # SECONDARY: correctness-change McNemar (kept, renamed) + proper label-instability test.
    mc_corr_stat, mc_corr_p = mcnemar_correctness(a_correct, b_correct)
    num_labels = prob_a.shape[1]
    if num_labels == 2:
        mlab_stat, mlab_p = mcnemar_label_binary(pred_a, pred_b)
        bowker_stat, bowker_p, bowker_df = float("nan"), float("nan"), 0
    else:
        mlab_stat, mlab_p = float("nan"), float("nan")
        bowker_stat, bowker_p, bowker_df = bowker_symmetry(pred_a, pred_b)
    return {
        "n_examples": int(len(common)),
        "n_flips": int(flips.sum()),
        "flip_rate": float(flips.mean()),
        "flip_rate_ci_low": ci_lo,
        "flip_rate_ci_high": ci_hi,
        "mean_prob_drift": float(prob_drift.mean()),
        "max_prob_drift": float(prob_drift.max()),
        "mean_margin_drift": float(margin_drift.mean()),
        "mean_max_abs_logit_diff": logit_drift,
        # secondary label-instability significance
        "mcnemar_label_stat": mlab_stat,
        "mcnemar_label_p": mlab_p,
        "bowker_stat": bowker_stat,
        "bowker_p": bowker_p,
        "bowker_df": bowker_df,
        # secondary correctness-change (NOT label instability)
        "mcnemar_correctness_stat": mc_corr_stat,
        "mcnemar_correctness_p": mc_corr_p,
    }


# --------------------------------------------------------------------------- #
# flip-by-margin (frozen reference)
# --------------------------------------------------------------------------- #
def flip_by_margin(contexts: pd.DataFrame, preds: Dict[str, pd.DataFrame],
                   ref_map: Dict[Tuple, str]) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for _, ctx in contexts.iterrows():
        key = (ctx["train_run_id"], ctx["dataset"], ctx["split"])
        ref_cid = ref_map.get(key)
        if ref_cid is None or ctx["context_id"] == ref_cid:
            continue
        ref = preds[ref_cid]
        cur = preds[ctx["context_id"]]
        common = ref.index.intersection(cur.index)
        if len(common) == 0:
            continue
        r = ref.loc[common]
        c = cur.loc[common]
        flip = (r["pred_label"].values != c["pred_label"].values).astype(float)
        rmargin = r["margin"].values
        for lo, hi in MARGIN_BUCKETS:
            m = (rmargin >= lo) & (rmargin < hi)
            if m.any():
                out.append({
                    "bucket": f"[{lo},{hi})",
                    "bucket_lo": lo,
                    "n_examples": int(m.sum()),
                    "n_flips": int(flip[m].sum()),
                })
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).groupby(["bucket", "bucket_lo"], as_index=False).sum()
    df["flip_rate"] = df["n_flips"] / df["n_examples"]
    return df.sort_values("bucket_lo")


# --------------------------------------------------------------------------- #
# per-context utility metrics
# --------------------------------------------------------------------------- #
def context_metrics(contexts: pd.DataFrame, preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, f1_score
    rows = []
    for _, ctx in contexts.iterrows():
        p = preds[ctx["context_id"]]
        y = p["true_label"].values
        yhat = p["pred_label"].values
        conf = p["top1_prob"].values
        correct = (yhat == y).astype(float)
        # ECE (15 bins)
        bins = np.linspace(0, 1, 16)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        rows.append({
            "context_id": ctx["context_id"],
            "train_run_id": ctx["train_run_id"], "dataset": ctx["dataset"],
            "dtype": ctx["dtype"], "quantization": ctx["quantization"],
            "batch_size": ctx["batch_size"], "order_name": ctx["order_name"],
            "device": ctx["device"], "seed_profile": ctx["seed_profile"],
            "accuracy": float(accuracy_score(y, yhat)),
            "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, yhat, average="weighted", zero_division=0)),
            "mean_confidence": float(conf.mean()),
            "ece": float(ece),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-root", required=True, help="Root containing completed inference contexts")
    ap.add_argument("--metrics-id", default=None, help="Output subdir under <root>/metrics_v2/ (default: UTC)")
    ap.add_argument("--reference-batch-size", type=int, default=32)
    ap.add_argument("--reference-seed-profile", default="production_like")
    ap.add_argument("--exploratory", action="store_true",
                    help="Allow nearest-batch reference fallback (exploration only; NOT the frozen paper artifact)")
    ap.add_argument("--require-complete-manifest", default=None, metavar="MANIFEST",
                    help="fail (exit 2) unless EVERY inference job in this manifest has "
                         "_SUCCESS under the artifacts root. Closes the silent-skip gap: "
                         "the loader ignores unfinished contexts, so without this gate "
                         "metrics can succeed with missing spokes/checkpoints.")
    ap.add_argument("--require-strict-reference", action="store_true",
                    help="Final study gate: require EXACTLY ONE reference per (train_run_id, dataset, split). "
                         "If any group has 0 or >1 candidate references, write diagnostics and FAIL the job (exit 2).")
    ap.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples for flip-rate CI (0 to disable)")
    ap.add_argument("--limit-checkpoints", type=int, default=None, help="Only load N checkpoints (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.require_complete_manifest:
        inf_root_gate = Path(args.artifacts_root) / "inference"
        want = [j for j in read_jsonl(args.require_complete_manifest)
                if j.get("job_type") == "inference"]
        missing = [j["job_id"] for j in want
                   if not (inf_root_gate / j["job_id"] / "_SUCCESS").exists()]
        if missing:
            print(f"[completeness gate] FAIL: {len(missing)}/{len(want)} manifest jobs "
                  f"missing _SUCCESS (e.g. {missing[:5]})", file=sys.stderr)
            return 2
        print(f"[completeness gate] OK: {len(want)}/{len(want)} manifest jobs complete")

    if args.require_strict_reference and args.exploratory:
        print("ERROR: --require-strict-reference is incompatible with --exploratory "
              "(the gate enforces the frozen strict anchor; exploratory permits fallback).")
        return 2
    protocol = "exploratory" if args.exploratory else "strict_paper"
    ref_cell = dict(DEFAULT_REFERENCE_CELL)
    ref_cell["batch_size"] = args.reference_batch_size
    ref_cell["seed_profile"] = args.reference_seed_profile

    root = Path(args.artifacts_root)
    metrics_id = args.metrics_id or time.strftime("metrics_%Y-%m-%dT%H-%M-%S", time.gmtime())
    out_dir = root / "metrics_v2" / metrics_id
    ensure_dir(out_dir)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print(f"Loading contexts from {root/'inference'} ...")
    contexts, preds = load_contexts(root, args.limit_checkpoints)
    if contexts.empty:
        print("No valid inference contexts found.")
        return 1
    print(f"  {len(contexts)} contexts, {contexts['train_run_id'].nunique()} checkpoints "
          f"({time.time()-t0:.1f}s)")

    # 1) frozen reference margin (strict exact anchor; degraded groups excluded)
    ref_df, ref_map, degraded = build_reference(contexts, preds, ref_cell, args.exploratory)
    if not ref_df.empty:
        ref_df["protocol"] = protocol
        ref_df.to_parquet(out_dir / "reference_predictions.parquet", index=False)
    n_groups = len(ref_map)
    print(f"  reference_predictions: {len(ref_df)} rows over {n_groups - len(degraded)}/{n_groups} groups "
          f"({len(degraded)} degraded)")
    if degraded:
        write_json_atomic(degraded, out_dir / "degraded_groups.json")
        pd.DataFrame([{
            "train_run_id": g["train_run_id"], "dataset": g["dataset"], "split": g["split"],
            "n_candidate_contexts": g["n_candidate_contexts"],
            "unmatched_anchor_fields": json.dumps(g["unmatched_anchor_fields"]),
        } for g in degraded]).to_parquet(out_dir / "degraded_groups.parquet", index=False)
        tag = "exploratory fallback used" if args.exploratory else "STRICT: no exact reference cell"
        keys = [(g["train_run_id"], g["dataset"], g["split"]) for g in degraded[:5]]
        print(f"  degraded groups ({tag}) e.g.: {keys}{' ...' if len(degraded) > 5 else ''}")
        print(f"  wrote degraded_groups.json/.parquet ({len(degraded)} groups)")

    if args.require_strict_reference and degraded:
        gate = {
            "gate": "require_strict_reference",
            "protocol": protocol,
            "status": "FAILED",
            "rule": "exactly one reference per (train_run_id, dataset, split)",
            "n_groups": int(n_groups),
            "n_ok_groups": int(n_groups - len(degraded)),
            "n_degraded_groups": int(len(degraded)),
            "reference_cell": ref_cell,
            "metrics_id": metrics_id,
            "degraded_groups": degraded,
        }
        write_json_atomic(gate, out_dir / "strict_reference_gate.json")
        print("=" * 72)
        print(f"STRICT REFERENCE GATE FAILED: {len(degraded)}/{n_groups} groups lack exactly one reference.")
        print(f"  diagnostics: {out_dir/'strict_reference_gate.json'} + degraded_groups.json/.parquet")
        print("  Common cause: order_name mismatch (anchor requires 'identity', not 'original').")
        print("  Fix the planner/anchor and re-run. Exiting 2 (final-run gate).")
        print("=" * 72)
        return 2

    # 2+3) classify pairs + metrics, grouped by comparison family
    fam_rows: Dict[str, List[Dict[str, Any]]] = {}
    ctx_by_base = {k: g for k, g in contexts.groupby(BASE_KEYS)}
    n_pairs = 0
    n_skipped_degraded = 0
    for key, grp in ctx_by_base.items():
        if ref_map.get(key) is None:      # degraded group -> no paper metrics
            n_skipped_degraded += 1
            continue
        recs = grp.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                cls = classify_pair(pd.Series(a), pd.Series(b))
                if cls is None:
                    continue
                comp_type, axis = cls
                m = pair_metrics(preds[a["context_id"]], preds[b["context_id"]], args.bootstrap, rng)
                if m is None:
                    continue
                row = {
                    "protocol": protocol,
                    "comparison_type": comp_type, "varied_axis": axis,
                    "train_run_id": a["train_run_id"], "dataset": a["dataset"], "split": a["split"],
                    "context_a": a["context_id"], "context_b": b["context_id"],
                    "value_a": _axis_val(a, axis),
                    "value_b": _axis_val(b, axis),
                    **m,
                }
                fam_rows.setdefault(comp_type, []).append(row)
                n_pairs += 1
    print(f"  classified {n_pairs} valid single-axis pairs into {len(fam_rows)} families")

    all_rows: List[Dict[str, Any]] = []
    for comp_type, rows in fam_rows.items():
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / f"{comp_type}_metrics.parquet", index=False)
        all_rows.extend(rows)
    if all_rows:
        pd.DataFrame(all_rows).to_parquet(out_dir / "all_pairwise_diagnostics.parquet", index=False)

    # 4) flip-by-margin (frozen reference)
    fbm = flip_by_margin(contexts, preds, ref_map)
    if not fbm.empty:
        fbm["protocol"] = protocol
        fbm.to_parquet(out_dir / "flip_by_margin.parquet", index=False)

    # 5) per-context utility
    cm = context_metrics(contexts, preds)
    cm["protocol"] = protocol
    cm.to_parquet(out_dir / "context_metrics.parquet", index=False)

    # summary
    fam_summary = {ct: {
        "pairs": int(len(rows)),
        "mean_flip_rate": float(np.mean([r["flip_rate"] for r in rows])),
        "max_flip_rate": float(np.max([r["flip_rate"] for r in rows])),
        "pairs_with_flips": int(sum(r["n_flips"] > 0 for r in rows)),
    } for ct, rows in fam_rows.items()}
    summary = {
        "artifacts_root": str(root),
        "metrics_id": metrics_id,
        "n_contexts": int(len(contexts)),
        "n_checkpoints": int(contexts["train_run_id"].nunique()),
        "n_pairs": n_pairs,
        "reference_cell": ref_cell,
        "protocol": protocol,
        "strict_reference": (not args.exploratory),
        "require_strict_reference_gate": bool(args.require_strict_reference),
        "degraded_groups": degraded,
        "n_degraded_groups": len(degraded),
        "n_skipped_degraded_groups": n_skipped_degraded,
        "bootstrap": args.bootstrap,
        "families": fam_summary,
        "flip_by_margin": fbm.to_dict("records") if not fbm.empty else [],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    write_json_atomic(summary, out_dir / "summary.json")
    mark_success(out_dir, {"metrics_id": metrics_id, "n_pairs": n_pairs})

    print("=" * 72)
    print(f"DONE ({summary['elapsed_sec']}s) -> {out_dir}")
    for ct, s in fam_summary.items():
        print(f"  {ct:26s} pairs={s['pairs']:5d} mean_flip={s['mean_flip_rate']:.5f} "
              f"max_flip={s['max_flip_rate']:.5f} pairs_with_flips={s['pairs_with_flips']}")
    if not fbm.empty:
        print("  flip_by_margin (frozen reference margin):")
        for _, r in fbm.iterrows():
            print(f"    {r['bucket']:14s} n={int(r['n_examples']):8d} n_flips={int(r['n_flips']):7d} flip_rate={r['flip_rate']:.3e}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
