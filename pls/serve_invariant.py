"""Shared, CPU-only helpers for ServeInvariant pilot analysis."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def load_predictions(run_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load one target-anchored run, rejecting duplicate example IDs."""
    rows: Dict[str, Dict[str, Any]] = {}
    path = Path(run_dir) / "predictions.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_id in rows:
                raise ValueError(f"duplicate example_id {example_id} in {path}")
            rows[example_id] = row
    return rows


def clopper_pearson_upper_zero(n: int, alpha: float = 0.05) -> float:
    """Two-sided exact-binomial upper confidence limit for zero events."""
    return 1.0 - (alpha / 2.0) ** (1.0 / n) if n else float("nan")


def rate_interval(
    events: int, n: int, alpha: float = 0.05
) -> tuple[float, float, str]:
    """Exact upper limit at zero; Wilson score interval otherwise."""
    if n <= 0:
        return float("nan"), float("nan"), "undefined"
    if events == 0:
        return 0.0, clopper_pearson_upper_zero(n, alpha), "clopper_pearson"
    p = events / n
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denom
    )
    return max(0.0, center - radius), min(1.0, center + radius), "wilson"


def event_stats(events: Sequence[bool], tag: str) -> Dict[str, Any]:
    n = len(events)
    n_events = sum(bool(x) for x in events)
    lo, hi, method = rate_interval(n_events, n)
    return {
        "tag": tag,
        "n": n,
        "n_flips": n_events,
        "flip_rate": n_events / n if n else float("nan"),
        "flip_ci_low": lo,
        "flip_ci_high": hi,
        "flip_ci_method": method,
    }


def tv_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"probability vector lengths differ: {len(a)} != {len(b)}")
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"logit vector lengths differ: {len(a)} != {len(b)}")
    return max(abs(x - y) for x, y in zip(a, b))


def pair_stats(
    rows_a: Mapping[str, Mapping[str, Any]],
    rows_b: Mapping[str, Mapping[str, Any]],
    ids: Iterable[str],
    tag: str,
) -> Dict[str, Any]:
    selected = list(ids)
    flips = [
        rows_a[example_id]["pred_class_index"]
        != rows_b[example_id]["pred_class_index"]
        for example_id in selected
    ]
    out = event_stats(flips, tag)
    if selected:
        out["mean_tv_prob_drift"] = sum(
            tv_distance(rows_a[e]["probs"], rows_b[e]["probs"])
            for e in selected
        ) / len(selected)
        out["mean_max_logit_drift"] = sum(
            max_abs_diff(rows_a[e]["raw_logits"], rows_b[e]["raw_logits"])
            for e in selected
        ) / len(selected)
    else:
        out["mean_tv_prob_drift"] = float("nan")
        out["mean_max_logit_drift"] = float("nan")
    return out


def accuracy(
    rows: Mapping[str, Mapping[str, Any]], ids: Iterable[str]
) -> float:
    selected = list(ids)
    if not selected:
        return float("nan")
    return sum(
        rows[e]["pred_class_index"] == rows[e]["true_class_index"]
        for e in selected
    ) / len(selected)


def cluster_bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for the mean, resampling checkpoint-level cells."""
    clean = [float(x) for x in values if not math.isnan(float(x))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], clean[0]
    rng = random.Random(seed)
    n = len(clean)
    stats = [
        sum(clean[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    ]
    stats.sort()
    lo = stats[int((alpha / 2.0) * n_boot)]
    hi = stats[min(int((1.0 - alpha / 2.0) * n_boot), n_boot - 1)]
    return lo, hi


def aggregate_cell_stats(
    rows: Sequence[Mapping[str, Any]],
    tag: str,
    n_boot: int = 5000,
    seed: int = 0,
) -> Dict[str, Any]:
    selected = [row for row in rows if row.get("tag") == tag and row.get("n", 0)]
    n = sum(int(row["n"]) for row in selected)
    events = sum(int(row["n_flips"]) for row in selected)
    lo, hi, method = rate_interval(events, n)
    rates = [float(row["flip_rate"]) for row in selected]
    cluster_lo, cluster_hi = cluster_bootstrap_ci(rates, n_boot=n_boot, seed=seed)
    return {
        "tag": tag,
        "cells": len(selected),
        "n": n,
        "n_flips": events,
        "micro_flip_rate": events / n if n else float("nan"),
        "micro_ci_low": lo,
        "micro_ci_high": hi,
        "micro_ci_method": method,
        "mean_cell_flip_rate": sum(rates) / len(rates) if rates else float("nan"),
        "cell_cluster_bootstrap_ci_low": cluster_lo,
        "cell_cluster_bootstrap_ci_high": cluster_hi,
    }
