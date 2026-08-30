#!/usr/bin/env python3
"""Read-only focused score-drift reanalysis of strict metric roots.

The analyzer deliberately reconstructs direct comparison pairs from inference
contexts. Pair eligibility is delegated to ``pls_metrics_v2.classify_pair``;
strict references are rebuilt with ``load_contexts`` and ``build_reference``.
No aggregate produced by the original metrics run is treated as row-level
evidence.

Orientation is deterministic and directional:

* the paper/reference value is the baseline when it is present in a pair;
* otherwise a documented family-specific semantic order is used;
* any remaining tie is resolved by a canonical value and then context ID.

Margins and certificates use the oriented baseline. The primary margin is the
top-two raw-logit gap ``gamma``. The secondary margin is the top-two
probability gap. An example is empirically certified when
``gamma > 2 * max(abs(logits_comparison - logits_baseline))``.

Outputs are written only under an explicit ``--output-dir`` and never inside a
source ``metrics_v2`` tree.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from pls.common import read_json, read_jsonl, write_json_atomic, write_text_atomic


try:
    from pls_metrics_v2 import (
        BASE_KEYS,
        DEFAULT_REFERENCE_CELL,
        build_reference,
        classify_pair,
        load_contexts,
    )
except ModuleNotFoundError as import_error:
    # pls_metrics_v2 imports scipy only for legacy significance tests. The three
    # paper APIs used here do not touch chi2, so permit this CPU-only analyzer to
    # load in minimal environments without silently replacing any called test.
    if import_error.name not in {"scipy", "scipy.stats"}:
        raise

    class _UnavailableChiSquared:
        @staticmethod
        def sf(*_args: Any, **_kwargs: Any) -> float:
            raise RuntimeError("scipy is required for chi-squared significance tests")

    scipy_module = types.ModuleType("scipy")
    scipy_stats_module = types.ModuleType("scipy.stats")
    scipy_stats_module.chi2 = _UnavailableChiSquared()
    scipy_module.stats = scipy_stats_module
    inserted_scipy = "scipy" not in sys.modules
    inserted_stats = "scipy.stats" not in sys.modules
    sys.modules.setdefault("scipy", scipy_module)
    sys.modules.setdefault("scipy.stats", scipy_stats_module)
    try:
        from pls_metrics_v2 import (
            BASE_KEYS,
            DEFAULT_REFERENCE_CELL,
            build_reference,
            classify_pair,
            load_contexts,
        )
    finally:
        if inserted_stats:
            sys.modules.pop("scipy.stats", None)
        if inserted_scipy:
            sys.modules.pop("scipy", None)


ANALYSIS_VERSION = "tier_a_a2_a4_v1"
QUANTILE_BIN_COUNT = 5

# Absolute bins are left-closed and right-open. The final bin is unbounded.
ABSOLUTE_MARGIN_BUCKETS: Dict[str, tuple[tuple[float, float], ...]] = {
    "logit_top2": (
        (0.0, 0.01),
        (0.01, 0.05),
        (0.05, 0.10),
        (0.10, 0.25),
        (0.25, 0.50),
        (0.50, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (4.0, math.inf),
    ),
    "probability_top2": (
        (0.0, 0.01),
        (0.01, 0.05),
        (0.05, 0.10),
        (0.10, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, math.inf),
    ),
}

# Upper bounds for compact, bounded-memory score-drift distributions. Bin zero
# is the exact point mass at zero; later intervals are (previous, upper].
DRIFT_HISTOGRAM_UPPERS: Dict[str, np.ndarray] = {
    "true_total_variation": np.asarray(
        [0.0] + [10.0**power for power in range(-12, 1)] + [math.inf],
        dtype=float,
    ),
    "max_abs_logit_drift": np.asarray(
        [0.0] + [10.0**power for power in range(-12, 5)] + [math.inf],
        dtype=float,
    ),
}

AXIS_PREFERENCES: Dict[str, tuple[Any, ...]] = {
    "batch_size": (32,),
    "order_policy": ("identity",),
    "batch_position": ("first", "middle", "last"),
    "composition_policy": (
        "random",
        "same_label",
        "different_label",
        "short",
        "long",
        "low_margin",
        "high_margin",
    ),
    "padding_policy": ("dynamic_longest", "static_max_length"),
    "dtype": ("fp32", "bf16", "fp16"),
    "quantization": ("none", "dynamic_int8_cpu"),
    "device": ("cuda", "cpu"),
    "backend": ("pytorch",),
    "execution_profile": (None, "deterministic", "production_like"),
}

FAMILY_ORIENTATION_TEXT: Dict[str, str] = {
    "batch_invariance": (
        "batch_size=reference batch first; otherwise ascending numeric batch size"
    ),
    "order_invariance": "identity first; otherwise canonical order-policy name",
    "position_invariance": "first, middle, last; then canonical value",
    "composition_invariance": (
        "random first, then same_label, different_label, short, long, "
        "low_margin, high_margin; then canonical value"
    ),
    "padding_invariance": (
        "dynamic_longest first, then static_max_length; then canonical value"
    ),
    "precision_drift": "fp32 first, then bf16, fp16; then canonical value",
    "quantization_drift": "none first, then dynamic_int8_cpu; then canonical value",
    "hardware_drift": "cuda first, then cpu; then canonical value",
    "backend_drift": "pytorch first; then canonical value",
    "determinism_drift": (
        "unset/default first, then deterministic, production_like; "
        "then canonical value"
    ),
    "deployment_profile_drift": (
        "component-wise baseline order: cuda before cpu and none before "
        "dynamic_int8_cpu; then canonical bundled value"
    ),
}


class AnalysisError(RuntimeError):
    """A protocol or data-integrity failure."""


@dataclass(frozen=True)
class Source:
    artifacts_root: Path
    metrics_root: Optional[Path]
    source_id: str


@dataclass
class AlignedPair:
    example_ids: np.ndarray
    pred_baseline: np.ndarray
    pred_comparison: np.ndarray
    true_labels: np.ndarray
    flips: np.ndarray
    harmful: np.ndarray
    beneficial: np.ndarray
    tv_drift: np.ndarray
    max_logit_drift: np.ndarray
    logit_margin: np.ndarray
    probability_margin: np.ndarray
    certified: np.ndarray
    task_type: str
    n_only_baseline: int
    n_only_comparison: int


def _as_float_vector(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains a non-finite value")
    return arr


def total_variation(
    probs_a: Sequence[float], probs_b: Sequence[float]
) -> float:
    """True discrete total variation: one half of the probability L1 distance."""
    a = _as_float_vector(probs_a, "probs_a")
    b = _as_float_vector(probs_b, "probs_b")
    if a.shape != b.shape:
        raise ValueError(f"probability vector lengths differ: {a.size} != {b.size}")
    return float(0.5 * np.abs(a - b).sum())


def max_abs_logit_drift(
    logits_a: Sequence[float], logits_b: Sequence[float]
) -> float:
    """L-infinity distance between two raw-logit vectors."""
    a = _as_float_vector(logits_a, "logits_a")
    b = _as_float_vector(logits_b, "logits_b")
    if a.shape != b.shape:
        raise ValueError(f"logit vector lengths differ: {a.size} != {b.size}")
    return float(np.abs(a - b).max())


def top_two_margin(values: Sequence[float]) -> float:
    """Largest minus second-largest value, requiring at least two classes."""
    arr = _as_float_vector(values, "values")
    if arr.size < 2:
        raise ValueError("top-two margin requires at least two classes")
    top = np.partition(arr, -2)[-2:]
    return float(top.max() - top.min())


def empirical_certificate(
    baseline_logits: Sequence[float], comparison_logits: Sequence[float]
) -> tuple[float, float, bool]:
    """Return ``(gamma, delta_linf, gamma > 2*delta_linf)``."""
    gamma = top_two_margin(baseline_logits)
    drift = max_abs_logit_drift(baseline_logits, comparison_logits)
    return gamma, drift, bool(gamma > 2.0 * drift)


def correctness_transition_counts(
    pred_baseline: Sequence[Any],
    pred_comparison: Sequence[Any],
    true_labels: Sequence[Any],
) -> Dict[str, int]:
    """Directional correctness counts under baseline -> comparison orientation."""
    baseline = np.asarray(pred_baseline)
    comparison = np.asarray(pred_comparison)
    truth = np.asarray(true_labels)
    if baseline.shape != comparison.shape or baseline.shape != truth.shape:
        raise ValueError("prediction and truth arrays must have identical shapes")
    baseline_correct = baseline == truth
    comparison_correct = comparison == truth
    harmful = baseline_correct & ~comparison_correct
    beneficial = ~baseline_correct & comparison_correct
    return {
        "correct_to_wrong": int(harmful.sum()),
        "wrong_to_correct": int(beneficial.sum()),
        "net_correct": int(beneficial.sum() - harmful.sum()),
    }


def assign_absolute_margin_bins(
    values: Sequence[float], margin_type: str
) -> np.ndarray:
    """Assign left-closed absolute margin buckets; invalid negatives map to -1."""
    if margin_type not in ABSOLUTE_MARGIN_BUCKETS:
        raise ValueError(f"unknown margin type: {margin_type}")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("margin values must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError("margin values contain a non-finite value")
    lowers = np.asarray(
        [lower for lower, _ in ABSOLUTE_MARGIN_BUCKETS[margin_type]], dtype=float
    )
    bins = np.searchsorted(lowers, arr, side="right") - 1
    bins[arr < 0.0] = -1
    return bins.astype(int)


def assign_within_cell_quantile_bins(
    values: Sequence[float], n_bins: int = QUANTILE_BIN_COUNT
) -> np.ndarray:
    """Assign deterministic within-cell empirical quantiles.

    Average ranks keep tied margins together. Thus bins can be empty when a
    cell is small or has many ties, which is preferable to splitting equal
    margins by row order.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("margin values must be one-dimensional")
    if arr.size == 0:
        return np.asarray([], dtype=int)
    if not np.all(np.isfinite(arr)):
        raise ValueError("margin values contain a non-finite value")
    ranks = pd.Series(arr).rank(method="average").to_numpy(dtype=float)
    bins = np.floor((ranks - 1.0) * n_bins / arr.size).astype(int)
    return np.clip(bins, 0, n_bins - 1)


def _axis_column(axis: str) -> str:
    return "order_name" if axis == "order_policy" else axis


def _equal_axis_value(a: Any, b: Any) -> bool:
    if a == b:
        return True
    try:
        return bool(pd.isna(a) and pd.isna(b))
    except (TypeError, ValueError):
        return False


def _canonical_value(value: Any) -> str:
    if value is None:
        return "<none>"
    try:
        if pd.isna(value):
            return "<none>"
    except (TypeError, ValueError):
        pass
    return str(value)


def _scalar_orientation_key(
    record: Mapping[str, Any],
    axis: str,
    reference_cell: Mapping[str, Any],
) -> tuple[Any, ...]:
    column = _axis_column(axis)
    value = record.get(column)
    preferences = list(AXIS_PREFERENCES.get(axis, ()))
    if axis == "batch_size":
        preferences = [reference_cell.get("batch_size", 32)]
    elif axis == "order_policy":
        preferences = [
            reference_cell.get("order_name", DEFAULT_REFERENCE_CELL["order_name"])
        ]
    elif axis in reference_cell:
        preferred = reference_cell.get(axis)
        preferences = [preferred] + [
            item for item in preferences if not _equal_axis_value(item, preferred)
        ]

    preference_index = len(preferences)
    for index, preferred in enumerate(preferences):
        if _equal_axis_value(value, preferred):
            preference_index = index
            break

    numeric_flag = 1
    numeric_value = 0.0
    if axis == "batch_size":
        try:
            numeric_flag = 0
            numeric_value = float(value)
        except (TypeError, ValueError):
            pass
    return (
        preference_index,
        numeric_flag,
        numeric_value,
        _canonical_value(value),
    )


def orientation_key(
    record: Mapping[str, Any],
    axis: str,
    reference_cell: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, ...]:
    """Family-semantic orientation key for a classified varied axis."""
    ref = dict(DEFAULT_REFERENCE_CELL)
    if reference_cell:
        ref.update(reference_cell)
    axes = axis.split("+")
    return tuple(_scalar_orientation_key(record, item, ref) for item in axes)


def orient_pair(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    axis: str,
    reference_cell: Optional[Mapping[str, Any]] = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return ``(baseline, comparison)`` independent of discovery order."""
    key_a = orientation_key(a, axis, reference_cell)
    key_b = orientation_key(b, axis, reference_cell)
    if key_a == key_b:
        key_a = (*key_a, str(a.get("context_id", "")))
        key_b = (*key_b, str(b.get("context_id", "")))
    return (a, b) if key_a <= key_b else (b, a)


def _matrix(series: pd.Series, name: str) -> np.ndarray:
    rows = [_as_float_vector(value, name) for value in series.to_numpy()]
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise AnalysisError(f"{name} vectors have inconsistent lengths")
    return np.vstack(rows)


def _validate_probabilities(probs: np.ndarray, context_id: str) -> None:
    if np.any(probs < -1e-7) or np.any(probs > 1.0 + 1e-7):
        raise AnalysisError(f"invalid probability value in context {context_id}")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise AnalysisError(f"probabilities do not sum to one in context {context_id}")


def align_and_measure_pair(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    baseline_context_id: str = "baseline",
    comparison_context_id: str = "comparison",
) -> AlignedPair:
    """Align a direct pair by ``example_id`` and calculate all score-drift row metrics."""
    required = {"pred_label", "true_label", "probs", "raw_logits"}
    for frame, context_id in (
        (baseline, baseline_context_id),
        (comparison, comparison_context_id),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise AnalysisError(
                f"context {context_id} lacks required columns: {sorted(missing)}"
            )
        if not frame.index.is_unique:
            raise AnalysisError(f"context {context_id} has duplicate example_id values")

    baseline_ids = set(baseline.index)
    comparison_ids = set(comparison.index)
    common = sorted(baseline_ids & comparison_ids, key=str)
    if not common:
        raise AnalysisError(
            f"contexts {baseline_context_id} and {comparison_context_id} "
            "have no common example_id"
        )
    a = baseline.loc[common]
    b = comparison.loc[common]
    truth_a = a["true_label"].to_numpy()
    truth_b = b["true_label"].to_numpy()
    if not np.array_equal(truth_a, truth_b):
        raise AnalysisError(
            f"true labels disagree after example_id alignment for "
            f"{baseline_context_id} vs {comparison_context_id}"
        )

    probs_a = _matrix(a["probs"], "probs")
    probs_b = _matrix(b["probs"], "probs")
    logits_a = _matrix(a["raw_logits"], "raw_logits")
    logits_b = _matrix(b["raw_logits"], "raw_logits")
    if probs_a.shape != probs_b.shape:
        raise AnalysisError("probability matrix shapes differ across a direct pair")
    if logits_a.shape != logits_b.shape or logits_a.shape != probs_a.shape:
        raise AnalysisError("logit/probability class dimensions differ across a pair")
    if probs_a.shape[1] < 2:
        raise AnalysisError("score-drift analysis requires at least two classes")
    _validate_probabilities(probs_a, baseline_context_id)
    _validate_probabilities(probs_b, comparison_context_id)

    pred_a = a["pred_label"].to_numpy()
    pred_b = b["pred_label"].to_numpy()
    flips = pred_a != pred_b
    correct_a = pred_a == truth_a
    correct_b = pred_b == truth_a
    harmful = correct_a & ~correct_b
    beneficial = ~correct_a & correct_b
    tv = 0.5 * np.abs(probs_a - probs_b).sum(axis=1)
    logit_drift = np.abs(logits_a - logits_b).max(axis=1)
    logit_sorted = np.partition(logits_a, -2, axis=1)[:, -2:]
    logit_margin = logit_sorted.max(axis=1) - logit_sorted.min(axis=1)
    prob_sorted = np.partition(probs_a, -2, axis=1)[:, -2:]
    probability_margin = prob_sorted.max(axis=1) - prob_sorted.min(axis=1)
    certified = logit_margin > 2.0 * logit_drift

    return AlignedPair(
        example_ids=np.asarray([str(value) for value in common], dtype=object),
        pred_baseline=pred_a,
        pred_comparison=pred_b,
        true_labels=truth_a,
        flips=flips,
        harmful=harmful,
        beneficial=beneficial,
        tv_drift=tv,
        max_logit_drift=logit_drift,
        logit_margin=logit_margin,
        probability_margin=probability_margin,
        certified=certified,
        task_type="binary" if probs_a.shape[1] == 2 else "multiclass",
        n_only_baseline=len(baseline_ids - comparison_ids),
        n_only_comparison=len(comparison_ids - baseline_ids),
    )


def checkpoint_clustered_bootstrap_ci(
    cluster_counts: Mapping[str, tuple[float, float]],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for a ratio, resampling checkpoint clusters.

    Each mapping value is ``(numerator, denominator)``. The bootstrap estimand
    is the pooled ratio after checkpoint resampling, so unequal example counts
    remain represented while within-checkpoint dependence is preserved.
    """
    clean = [
        (float(numerator), float(denominator))
        for numerator, denominator in cluster_counts.values()
        if float(denominator) > 0.0
    ]
    if not clean or n_boot <= 0:
        return float("nan"), float("nan")
    numerators = np.asarray([value[0] for value in clean], dtype=float)
    denominators = np.asarray([value[1] for value in clean], dtype=float)
    if len(clean) == 1:
        point = float(numerators[0] / denominators[0])
        return point, point
    rng = np.random.default_rng(seed)
    results = np.empty(n_boot, dtype=float)
    cluster_count = len(clean)
    batch_size = 256
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        indices = rng.integers(
            0, cluster_count, size=(stop - start, cluster_count)
        )
        sampled_numerators = numerators[indices].sum(axis=1)
        sampled_denominators = denominators[indices].sum(axis=1)
        results[start:stop] = sampled_numerators / sampled_denominators
    return (
        float(np.quantile(results, alpha / 2.0)),
        float(np.quantile(results, 1.0 - alpha / 2.0)),
    )


def _derived_seed(seed: int, key: Any) -> int:
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") + int(seed)) % (2**63 - 1)


def _axis_value(record: Mapping[str, Any], axis: str) -> str:
    if "+" in axis:
        return "|".join(
            f"{item}={_canonical_value(record.get(_axis_column(item)))}"
            for item in axis.split("+")
        )
    return _canonical_value(record.get(_axis_column(axis)))


def _distribution_fields(prefix: str, values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(values.max()),
    }


def _json_label(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


class ReanalysisAccumulator:
    """Bounded aggregate state; only one pair's example arrays exist at a time."""

    def __init__(self, n_boot: int, seed: int, quantile_bins: int) -> None:
        self.n_boot = n_boot
        self.seed = seed
        self.quantile_bins = quantile_bins
        self.pair_rows: list[Dict[str, Any]] = []
        self.family_pair_counts: Counter[str] = Counter()
        self.family_clusters: MutableMapping[
            str, MutableMapping[str, Counter[str]]
        ] = defaultdict(lambda: defaultdict(Counter))
        self.label_counts: Counter[tuple[str, str, str, str]] = Counter()
        self.curve_clusters: MutableMapping[
            tuple[str, str, str, str, int],
            MutableMapping[str, list[int]],
        ] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.curve_cells: Counter[tuple[str, str, str, str, int]] = Counter()
        self.curve_combinations: set[tuple[str, str, str, str]] = set()
        self.drift_moments: MutableMapping[
            tuple[str, str, str, str], Dict[str, float]
        ] = defaultdict(
            lambda: {
                "n": 0.0,
                "sum": 0.0,
                "sumsq": 0.0,
                "min": math.inf,
                "max": -math.inf,
            }
        )
        self.drift_clusters: MutableMapping[
            tuple[str, str, str, str],
            MutableMapping[str, list[float]],
        ] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        self.drift_histograms: Counter[tuple[str, str, str, str, int]] = Counter()

    def add_pair(
        self,
        source: Source,
        family: str,
        axis: str,
        baseline_record: Mapping[str, Any],
        comparison_record: Mapping[str, Any],
        measured: AlignedPair,
        reference_cell: Mapping[str, Any],
    ) -> None:
        cluster = f"{source.source_id}::{baseline_record['train_run_id']}"
        n = int(measured.flips.size)
        flips = int(measured.flips.sum())
        harmful = int(measured.harmful.sum())
        beneficial = int(measured.beneficial.sum())
        certified = int(measured.certified.sum())
        uncertified_mask = ~measured.certified
        uncertified = int(uncertified_mask.sum())
        uncertified_flips = int((measured.flips & uncertified_mask).sum())
        certified_flips = int((measured.flips & measured.certified).sum())

        family_cluster = self.family_clusters[family][cluster]
        family_cluster.update(
            {
                "n": n,
                "flips": flips,
                "harmful": harmful,
                "beneficial": beneficial,
                "certified": certified,
                "uncertified": uncertified,
                "uncertified_flips": uncertified_flips,
                "certified_flips": certified_flips,
            }
        )
        self.family_pair_counts[family] += 1

        stable = ~measured.flips
        pair_row: Dict[str, Any] = {
            "analysis_version": ANALYSIS_VERSION,
            "source_id": source.source_id,
            "artifacts_root": str(source.artifacts_root),
            "comparison_family": family,
            "varied_axis": axis,
            "orientation_rule": FAMILY_ORIENTATION_TEXT.get(
                family, "canonical varied-axis value, then context ID"
            ),
            "train_run_id": baseline_record["train_run_id"],
            "dataset": baseline_record["dataset"],
            "split": baseline_record["split"],
            "task_type": measured.task_type,
            "baseline_context_id": baseline_record["context_id"],
            "comparison_context_id": comparison_record["context_id"],
            "baseline_value": _axis_value(baseline_record, axis),
            "comparison_value": _axis_value(comparison_record, axis),
            "n_aligned_examples": n,
            "n_only_baseline": measured.n_only_baseline,
            "n_only_comparison": measured.n_only_comparison,
            "n_top1_flips": flips,
            "top1_flip_rate": flips / n,
            "correct_to_wrong": harmful,
            "wrong_to_correct": beneficial,
            "net_accuracy_delta": (beneficial - harmful) / n,
            "certificate_covered": certified,
            "certificate_coverage": certified / n,
            "uncertified_examples": uncertified,
            "uncertified_flips": uncertified_flips,
            "uncertified_flip_rate": (
                uncertified_flips / uncertified if uncertified else float("nan")
            ),
            "certified_flips": certified_flips,
            **_distribution_fields(
                "true_total_variation", measured.tv_drift
            ),
            **_distribution_fields(
                "true_total_variation_zero_top1_flip",
                measured.tv_drift[stable],
            ),
            **_distribution_fields(
                "max_abs_logit_drift", measured.max_logit_drift
            ),
            **_distribution_fields(
                "max_abs_logit_drift_zero_top1_flip",
                measured.max_logit_drift[stable],
            ),
            **_distribution_fields("logit_margin", measured.logit_margin),
            **_distribution_fields(
                "probability_margin", measured.probability_margin
            ),
        }
        self.pair_rows.append(pair_row)

        changed = np.flatnonzero(measured.flips)
        for index in changed:
            self.label_counts[
                (
                    family,
                    measured.task_type,
                    _json_label(measured.pred_baseline[index]),
                    _json_label(measured.pred_comparison[index]),
                )
            ] += 1

        for task_group in ("all", measured.task_type):
            self._add_drift(
                family,
                task_group,
                "all_examples",
                "true_total_variation",
                measured.tv_drift,
                cluster,
            )
            self._add_drift(
                family,
                task_group,
                "zero_top1_flip",
                "true_total_variation",
                measured.tv_drift[stable],
                cluster,
            )
            self._add_drift(
                family,
                task_group,
                "all_examples",
                "max_abs_logit_drift",
                measured.max_logit_drift,
                cluster,
            )
            self._add_drift(
                family,
                task_group,
                "zero_top1_flip",
                "max_abs_logit_drift",
                measured.max_logit_drift[stable],
                cluster,
            )

        for margin_type, margins in (
            ("logit_top2", measured.logit_margin),
            ("probability_top2", measured.probability_margin),
        ):
            absolute = assign_absolute_margin_bins(margins, margin_type)
            quantiles = assign_within_cell_quantile_bins(
                margins, self.quantile_bins
            )
            self._add_curve_bins(
                family,
                measured.task_type,
                margin_type,
                "absolute",
                absolute,
                measured.flips,
                cluster,
            )
            self._add_curve_bins(
                family,
                measured.task_type,
                margin_type,
                "within_cell_quantile",
                quantiles,
                measured.flips,
                cluster,
            )

    def _add_curve_bins(
        self,
        family: str,
        task_type: str,
        margin_type: str,
        binning: str,
        bins: np.ndarray,
        flips: np.ndarray,
        cluster: str,
    ) -> None:
        combination = (family, task_type, margin_type, binning)
        self.curve_combinations.add(combination)
        for bucket_index in np.unique(bins):
            if bucket_index < 0:
                continue
            mask = bins == bucket_index
            key = (*combination, int(bucket_index))
            counts = self.curve_clusters[key][cluster]
            counts[0] += int(flips[mask].sum())
            counts[1] += int(mask.sum())
            self.curve_cells[key] += 1

    def _add_drift(
        self,
        family: str,
        task_type: str,
        subset: str,
        metric: str,
        values: np.ndarray,
        cluster: str,
    ) -> None:
        if values.size == 0:
            return
        key = (family, task_type, subset, metric)
        moment = self.drift_moments[key]
        moment["n"] += float(values.size)
        moment["sum"] += float(values.sum())
        moment["sumsq"] += float(np.square(values).sum())
        moment["min"] = min(moment["min"], float(values.min()))
        moment["max"] = max(moment["max"], float(values.max()))
        cluster_values = self.drift_clusters[key][cluster]
        cluster_values[0] += float(values.sum())
        cluster_values[1] += float(values.size)
        upper_bounds = DRIFT_HISTOGRAM_UPPERS[metric]
        indices = np.searchsorted(upper_bounds, values, side="left")
        counts = np.bincount(indices, minlength=len(upper_bounds))
        for index, count in enumerate(counts):
            if count:
                self.drift_histograms[(*key, index)] += int(count)

    def pair_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.pair_rows)

    def family_frame(self) -> pd.DataFrame:
        rows = []
        for family in sorted(self.family_clusters):
            clusters = self.family_clusters[family]
            totals: Counter[str] = Counter()
            for counts in clusters.values():
                totals.update(counts)
            n = int(totals["n"])

            def ratio_ci(numerator: str, denominator: str = "n"):
                data = {
                    cluster: (counts[numerator], counts[denominator])
                    for cluster, counts in clusters.items()
                }
                return checkpoint_clustered_bootstrap_ci(
                    data,
                    self.n_boot,
                    _derived_seed(
                        self.seed, ("family", family, numerator, denominator)
                    ),
                )

            flip_lo, flip_hi = ratio_ci("flips")
            harmful_lo, harmful_hi = ratio_ci("harmful")
            beneficial_lo, beneficial_hi = ratio_ci("beneficial")
            net_data = {
                cluster: (
                    counts["beneficial"] - counts["harmful"],
                    counts["n"],
                )
                for cluster, counts in clusters.items()
            }
            net_lo, net_hi = checkpoint_clustered_bootstrap_ci(
                net_data,
                self.n_boot,
                _derived_seed(self.seed, ("family", family, "net_accuracy")),
            )
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "comparison_family": family,
                    "n_pairs": int(self.family_pair_counts[family]),
                    "n_checkpoints": len(clusters),
                    "n_pair_examples": n,
                    "n_top1_flips": int(totals["flips"]),
                    "top1_flip_rate": totals["flips"] / n if n else float("nan"),
                    "top1_flip_ci_low": flip_lo,
                    "top1_flip_ci_high": flip_hi,
                    "correct_to_wrong": int(totals["harmful"]),
                    "correct_to_wrong_rate": (
                        totals["harmful"] / n if n else float("nan")
                    ),
                    "correct_to_wrong_ci_low": harmful_lo,
                    "correct_to_wrong_ci_high": harmful_hi,
                    "wrong_to_correct": int(totals["beneficial"]),
                    "wrong_to_correct_rate": (
                        totals["beneficial"] / n if n else float("nan")
                    ),
                    "wrong_to_correct_ci_low": beneficial_lo,
                    "wrong_to_correct_ci_high": beneficial_hi,
                    "net_accuracy_delta": (
                        (totals["beneficial"] - totals["harmful"]) / n
                        if n
                        else float("nan")
                    ),
                    "net_accuracy_delta_ci_low": net_lo,
                    "net_accuracy_delta_ci_high": net_hi,
                }
            )
        return pd.DataFrame(rows)

    def certificate_frame(self) -> pd.DataFrame:
        rows = []
        for family in sorted(self.family_clusters):
            clusters = self.family_clusters[family]
            totals: Counter[str] = Counter()
            for counts in clusters.values():
                totals.update(counts)

            def ci(numerator: str, denominator: str):
                return checkpoint_clustered_bootstrap_ci(
                    {
                        cluster: (counts[numerator], counts[denominator])
                        for cluster, counts in clusters.items()
                    },
                    self.n_boot,
                    _derived_seed(
                        self.seed, ("certificate", family, numerator, denominator)
                    ),
                )

            coverage_lo, coverage_hi = ci("certified", "n")
            uncertified_lo, uncertified_hi = ci(
                "uncertified_flips", "uncertified"
            )
            n = int(totals["n"])
            uncertified = int(totals["uncertified"])
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "comparison_family": family,
                    "certificate_rule": (
                        "baseline top-two logit margin gamma > "
                        "2 * max_abs_logit_drift"
                    ),
                    "n_checkpoints": len(clusters),
                    "n_pair_examples": n,
                    "certified_examples": int(totals["certified"]),
                    "certificate_coverage": (
                        totals["certified"] / n if n else float("nan")
                    ),
                    "certificate_coverage_ci_low": coverage_lo,
                    "certificate_coverage_ci_high": coverage_hi,
                    "uncertified_examples": uncertified,
                    "uncertified_flips": int(totals["uncertified_flips"]),
                    "uncertified_flip_rate": (
                        totals["uncertified_flips"] / uncertified
                        if uncertified
                        else float("nan")
                    ),
                    "uncertified_flip_rate_ci_low": uncertified_lo,
                    "uncertified_flip_rate_ci_high": uncertified_hi,
                    "certified_flips": int(totals["certified_flips"]),
                }
            )
        return pd.DataFrame(rows)

    def labels_frame(self) -> pd.DataFrame:
        rows = [
            {
                "analysis_version": ANALYSIS_VERSION,
                "comparison_family": family,
                "task_type": task_type,
                "baseline_label_json": baseline_label,
                "comparison_label_json": comparison_label,
                "direction": f"{baseline_label}->{comparison_label}",
                "n_flips": count,
            }
            for (
                family,
                task_type,
                baseline_label,
                comparison_label,
            ), count in sorted(self.label_counts.items())
        ]
        return pd.DataFrame(rows)

    def curves_frame(self) -> pd.DataFrame:
        rows = []
        for combination in sorted(self.curve_combinations):
            family, task_type, margin_type, binning = combination
            bucket_count = (
                len(ABSOLUTE_MARGIN_BUCKETS[margin_type])
                if binning == "absolute"
                else self.quantile_bins
            )
            for bucket_index in range(bucket_count):
                key = (*combination, bucket_index)
                clusters = self.curve_clusters.get(key, {})
                n = sum(counts[1] for counts in clusters.values())
                flips = sum(counts[0] for counts in clusters.values())
                ci_low, ci_high = checkpoint_clustered_bootstrap_ci(
                    {
                        cluster: (counts[0], counts[1])
                        for cluster, counts in clusters.items()
                    },
                    self.n_boot,
                    _derived_seed(self.seed, ("curve", key)),
                )
                if binning == "absolute":
                    lower, upper = ABSOLUTE_MARGIN_BUCKETS[margin_type][
                        bucket_index
                    ]
                    bucket = (
                        f"[{lower:g},{upper:g})"
                        if math.isfinite(upper)
                        else f"[{lower:g},inf)"
                    )
                    quantile_low = float("nan")
                    quantile_high = float("nan")
                else:
                    lower = float("nan")
                    upper = float("nan")
                    quantile_low = bucket_index / self.quantile_bins
                    quantile_high = (bucket_index + 1) / self.quantile_bins
                    bucket = (
                        f"Q{bucket_index + 1} "
                        f"[{quantile_low:.1f},{quantile_high:.1f})"
                    )
                cluster_rates = [
                    counts[0] / counts[1]
                    for counts in clusters.values()
                    if counts[1]
                ]
                rows.append(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "comparison_family": family,
                        "task_type": task_type,
                        "margin_type": margin_type,
                        "margin_role": "oriented_baseline_top_two",
                        "binning": binning,
                        "bucket_index": bucket_index,
                        "bucket": bucket,
                        "bucket_lower": lower,
                        "bucket_upper": upper,
                        "quantile_lower": quantile_low,
                        "quantile_upper": quantile_high,
                        "n_cells": int(self.curve_cells.get(key, 0)),
                        "n_checkpoints": len(clusters),
                        "n_pair_examples": int(n),
                        "n_top1_flips": int(flips),
                        "top1_flip_rate": flips / n if n else float("nan"),
                        "checkpoint_mean_flip_rate": (
                            float(np.mean(cluster_rates))
                            if cluster_rates
                            else float("nan")
                        ),
                        "top1_flip_cluster_ci_low": ci_low,
                        "top1_flip_cluster_ci_high": ci_high,
                    }
                )
        return pd.DataFrame(rows)

    def drift_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_rows = []
        for key in sorted(self.drift_moments):
            family, task_type, subset, metric = key
            moment = self.drift_moments[key]
            n = int(moment["n"])
            mean = moment["sum"] / n
            variance = max(0.0, moment["sumsq"] / n - mean * mean)
            ci_low, ci_high = checkpoint_clustered_bootstrap_ci(
                {
                    cluster: (values[0], values[1])
                    for cluster, values in self.drift_clusters[key].items()
                },
                self.n_boot,
                _derived_seed(self.seed, ("drift", key)),
            )
            summary_rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "comparison_family": family,
                    "task_type": task_type,
                    "subset": subset,
                    "metric": metric,
                    "n_pair_examples": n,
                    "n_checkpoints": len(self.drift_clusters[key]),
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "min": moment["min"],
                    "max": moment["max"],
                    "mean_cluster_ci_low": ci_low,
                    "mean_cluster_ci_high": ci_high,
                    "distribution_artifact": "drift_distributions.parquet",
                }
            )

        histogram_rows = []
        for key_with_index, count in sorted(self.drift_histograms.items()):
            family, task_type, subset, metric, index = key_with_index
            uppers = DRIFT_HISTOGRAM_UPPERS[metric]
            upper = float(uppers[index])
            lower = 0.0 if index == 0 else float(uppers[index - 1])
            if index == 0:
                bucket = "[0,0]"
            elif math.isfinite(upper):
                bucket = f"({lower:g},{upper:g}]"
            else:
                bucket = f"({lower:g},inf)"
            histogram_rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "comparison_family": family,
                    "task_type": task_type,
                    "subset": subset,
                    "metric": metric,
                    "bucket_index": index,
                    "bucket": bucket,
                    "bucket_lower_exclusive": lower,
                    "bucket_upper_inclusive": upper,
                    "n_pair_examples": int(count),
                }
            )
        return pd.DataFrame(summary_rows), pd.DataFrame(histogram_rows)


def _source_id(path: Path) -> str:
    resolved = str(path.resolve())
    suffix = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:10]
    return f"{path.name}-{suffix}"


def source_from_metrics_root(metrics_root: str | Path) -> Source:
    metrics = Path(metrics_root).expanduser().resolve()
    if metrics.parent.name != "metrics_v2":
        raise AnalysisError(
            f"metrics root must be <artifacts_root>/metrics_v2/<metrics_id>: {metrics}"
        )
    artifacts = metrics.parent.parent
    summary_path = metrics / "summary.json"
    if not summary_path.exists():
        raise AnalysisError(f"strict metrics root lacks summary.json: {metrics}")
    summary = read_json(summary_path)
    if summary.get("protocol") != "strict_paper" or not summary.get(
        "strict_reference", False
    ):
        raise AnalysisError(f"metrics root is not marked strict_paper: {metrics}")
    return Source(artifacts, metrics, _source_id(metrics))


def source_from_artifacts_root(artifacts_root: str | Path) -> Source:
    artifacts = Path(artifacts_root).expanduser().resolve()
    return Source(artifacts, None, _source_id(artifacts))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_location(output_dir: Path, sources: Sequence[Source]) -> None:
    output = output_dir.expanduser().resolve()
    for source in sources:
        metrics_tree = (source.artifacts_root / "metrics_v2").resolve()
        if _is_relative_to(output, metrics_tree):
            raise AnalysisError(
                f"output directory must be outside source metrics tree {metrics_tree}"
            )
        if source.metrics_root and (
            output == source.metrics_root
            or _is_relative_to(source.metrics_root, output)
        ):
            raise AnalysisError(
                f"output directory may not contain source metrics root "
                f"{source.metrics_root}"
            )


def check_complete_manifest(
    artifacts_root: Path, manifest: str | Path
) -> Dict[str, Any]:
    jobs = [
        job for job in read_jsonl(manifest) if job.get("job_type") == "inference"
    ]
    if not jobs:
        raise AnalysisError(f"manifest contains no inference jobs: {manifest}")
    malformed = [job for job in jobs if not job.get("job_id")]
    if malformed:
        raise AnalysisError(f"manifest has inference jobs without job_id: {manifest}")
    inference_root = artifacts_root / "inference"
    missing = [
        str(job["job_id"])
        for job in jobs
        if not (inference_root / str(job["job_id"]) / "_SUCCESS").exists()
    ]
    if missing:
        raise AnalysisError(
            f"complete-manifest gate failed for {artifacts_root}: "
            f"{len(missing)}/{len(jobs)} missing _SUCCESS; examples={missing[:5]}"
        )
    return {
        "manifest": str(Path(manifest).resolve()),
        "n_inference_jobs": len(jobs),
        "n_complete": len(jobs),
        "status": "passed",
    }


def assess_target_length(
    source: Source, allowed_context_ids: Optional[set[str]] = None
) -> Dict[str, Any]:
    """Check whether realized padded width exists in ordinary prediction rows."""
    ordinary = 0
    missing_width = 0
    examples: list[str] = []
    inference_root = source.artifacts_root / "inference"
    if inference_root.exists():
        for run_dir in sorted(path for path in inference_root.iterdir() if path.is_dir()):
            spec_path = run_dir / "job_spec.json"
            predictions_path = run_dir / "predictions.jsonl"
            if not (
                spec_path.exists()
                and predictions_path.exists()
                and (run_dir / "_SUCCESS").exists()
            ):
                continue
            spec = read_json(spec_path)
            context_id = str(spec.get("job_id") or run_dir.name)
            if (
                allowed_context_ids is not None
                and context_id not in allowed_context_ids
            ):
                continue
            if (
                spec.get("mode") == "target_anchored"
                or spec.get("composition_policy") is not None
            ):
                continue
            ordinary += 1
            first_row: Optional[Dict[str, Any]] = None
            for row in read_jsonl(predictions_path):
                first_row = row
                break
            if first_row is None or "padded_sequence_length" not in first_row:
                missing_width += 1
                if len(examples) < 5:
                    examples.append(context_id)

    if missing_width:
        return {
            "source_id": source.source_id,
            "status": "unavailable",
            "ordinary_contexts_checked": ordinary,
            "ordinary_contexts_missing_realized_padded_width": missing_width,
            "example_context_ids": examples,
            "statement": (
                "Target-length analysis unavailable: ordinary prediction rows lack realized padded "
                "width (`padded_sequence_length`), so realized-shape mediation "
                "cannot be reconstructed post hoc."
            ),
        }
    return {
        "source_id": source.source_id,
        "status": "not_analyzed",
        "ordinary_contexts_checked": ordinary,
        "ordinary_contexts_missing_realized_padded_width": 0,
        "statement": (
            "Target-length analysis is outside this score-drift analyzer; no ordinary context with a "
            "missing realized padded width was detected."
        ),
    }


def _strict_reference_gate(
    source: Source,
    contexts: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    reference_cell: Mapping[str, Any],
) -> tuple[Dict[tuple[Any, ...], Optional[str]], Dict[str, Any]]:
    reference_rows, reference_map, degraded = build_reference(
        contexts, predictions, dict(reference_cell), exploratory=False
    )
    del reference_rows
    if degraded:
        examples = [
            {
                "train_run_id": item["train_run_id"],
                "dataset": item["dataset"],
                "split": item["split"],
                "status": item["status"],
            }
            for item in degraded[:5]
        ]
        raise AnalysisError(
            f"strict unique-reference gate failed for {source.artifacts_root}: "
            f"{len(degraded)}/{len(reference_map)} groups; examples={examples}"
        )
    return reference_map, {
        "status": "passed",
        "rule": "exactly one strict reference per (train_run_id, dataset, split)",
        "n_groups": len(reference_map),
        "n_degraded_groups": 0,
    }


def _restrict_to_recorded_metrics_contexts(
    source: Source,
    contexts: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Pin reanalysis to the contexts recorded by an existing metrics root."""
    if source.metrics_root is None:
        return contexts, predictions
    context_path = source.metrics_root / "context_metrics.parquet"
    if not context_path.exists():
        raise AnalysisError(
            f"strict metrics root lacks context_metrics.parquet: "
            f"{source.metrics_root}"
        )
    recorded = pd.read_parquet(context_path, columns=["context_id"])
    if recorded["context_id"].isna().any() or not recorded["context_id"].is_unique:
        raise AnalysisError(
            f"context_metrics.parquet has null or duplicate context_id values: "
            f"{context_path}"
        )
    expected = {str(value) for value in recorded["context_id"]}
    loaded = {str(value) for value in contexts["context_id"]}
    missing = sorted(expected - loaded)
    if missing:
        raise AnalysisError(
            f"{len(missing)} contexts recorded by {source.metrics_root} are no "
            f"longer loadable from inference/; examples={missing[:5]}"
        )
    selected = contexts[contexts["context_id"].astype(str).isin(expected)].copy()
    selected_predictions = {
        context_id: predictions[context_id]
        for context_id in selected["context_id"].astype(str)
    }
    return selected, selected_predictions


def analyze_sources(
    sources: Sequence[Source],
    reference_cell: Mapping[str, Any],
    n_boot: int,
    seed: int,
    quantile_bins: int,
) -> tuple[ReanalysisAccumulator, list[Dict[str, Any]], list[Dict[str, Any]]]:
    accumulator = ReanalysisAccumulator(n_boot, seed, quantile_bins)
    source_summaries: list[Dict[str, Any]] = []
    target_length_rows: list[Dict[str, Any]] = []

    for source in sources:
        contexts, predictions = load_contexts(source.artifacts_root)
        if contexts.empty:
            raise AnalysisError(
                f"no complete inference contexts under {source.artifacts_root}"
            )
        contexts, predictions = _restrict_to_recorded_metrics_contexts(
            source, contexts, predictions
        )
        reference_map, gate = _strict_reference_gate(
            source, contexts, predictions, reference_cell
        )
        target_length_rows.append(
            assess_target_length(
                source,
                {str(value) for value in contexts["context_id"].to_numpy()},
            )
        )
        pair_count_before = len(accumulator.pair_rows)
        context_groups = contexts.groupby(BASE_KEYS, dropna=False, sort=True)
        for raw_key, group in context_groups:
            key = tuple(raw_key) if isinstance(raw_key, tuple) else (raw_key,)
            if reference_map.get(key) is None:
                raise AnalysisError(
                    f"internal strict-reference inconsistency for group {key}"
                )
            records = sorted(
                group.to_dict("records"), key=lambda item: str(item["context_id"])
            )
            for left_index in range(len(records)):
                for right_index in range(left_index + 1, len(records)):
                    left = records[left_index]
                    right = records[right_index]
                    classification = classify_pair(
                        pd.Series(left), pd.Series(right)
                    )
                    if classification is None:
                        continue
                    family, axis = classification
                    baseline, comparison = orient_pair(
                        left, right, axis, reference_cell
                    )
                    measured = align_and_measure_pair(
                        predictions[baseline["context_id"]],
                        predictions[comparison["context_id"]],
                        str(baseline["context_id"]),
                        str(comparison["context_id"]),
                    )
                    accumulator.add_pair(
                        source,
                        family,
                        axis,
                        baseline,
                        comparison,
                        measured,
                        reference_cell,
                    )

        source_summaries.append(
            {
                "source_id": source.source_id,
                "artifacts_root": str(source.artifacts_root),
                "metrics_root": (
                    str(source.metrics_root) if source.metrics_root else None
                ),
                "n_contexts": int(len(contexts)),
                "n_checkpoints": int(contexts["train_run_id"].nunique()),
                "n_valid_direct_pairs": (
                    len(accumulator.pair_rows) - pair_count_before
                ),
                "strict_reference_gate": gate,
            }
        )
        del contexts, predictions, reference_map
        gc.collect()
    return accumulator, source_summaries, target_length_rows


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_rate(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.6g}" if math.isfinite(number) else "n/a"


def build_markdown_report(
    summary: Mapping[str, Any],
    family_frame: pd.DataFrame,
    certificate_frame: pd.DataFrame,
) -> str:
    lines = [
        "# Focused Score-Drift Reanalysis",
        "",
        "All direct pairs were reconstructed with `pls_metrics_v2.classify_pair`, "
        "aligned by `example_id`, and oriented baseline to comparison before "
        "directional statistics were computed.",
        "",
        "## Baseline Orientation",
        "",
        "| comparison family | deterministic rule |",
        "|---|---|",
    ]
    for family, rule in sorted(FAMILY_ORIENTATION_TEXT.items()):
        lines.append(f"| `{family}` | {rule} |")

    lines.extend(
        [
            "",
            "The paper reference value is preferred whenever present. For "
            "batch-size comparisons, the configured reference batch replaces "
            "the default value in the rule above. Remaining ties use canonical "
            "values and context IDs.",
            "",
            "## Directional Accuracy",
            "",
            "| family | pairs | checkpoint clusters | pair-examples | flips | "
            "correct->wrong | wrong->correct | net accuracy delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in family_frame.sort_values("comparison_family").iterrows():
        lines.append(
            f"| `{row['comparison_family']}` | {int(row['n_pairs'])} | "
            f"{int(row['n_checkpoints'])} | {int(row['n_pair_examples'])} | "
            f"{int(row['n_top1_flips'])} | {int(row['correct_to_wrong'])} | "
            f"{int(row['wrong_to_correct'])} | "
            f"{_format_rate(row['net_accuracy_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Empirical Certificate",
            "",
            "Coverage uses the strict certificate "
            "`gamma > 2 * ||delta logits||_infinity`, where `gamma` is the "
            "oriented baseline top-two logit gap.",
            "",
            "| family | coverage | 95% checkpoint-cluster CI | uncertified "
            "flip rate | certified flips |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in certificate_frame.sort_values("comparison_family").iterrows():
        lines.append(
            f"| `{row['comparison_family']}` | "
            f"{_format_rate(row['certificate_coverage'])} | "
            f"[{_format_rate(row['certificate_coverage_ci_low'])}, "
            f"{_format_rate(row['certificate_coverage_ci_high'])}] | "
            f"{_format_rate(row['uncertified_flip_rate'])} | "
            f"{int(row['certified_flips'])} |"
        )

    lines.extend(["", "## Margin Curves", ""])
    lines.append(
        "The primary curves use oriented-baseline top-two raw-logit margins. "
        "Secondary curves use top-two probability margins. Both are emitted as "
        "absolute buckets and within-pair-cell empirical quantiles, split into "
        "binary and multiclass tasks. Confidence intervals resample checkpoint "
        "clusters and recompute the pooled rate."
    )

    lines.extend(["", "## Target-Length Availability", ""])
    for item in summary["target_length"]:
        lines.append(f"- `{item['source_id']}`: {item['statement']}")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `pair_metrics.parquet`: exact per-direct-pair score-drift metrics and "
            "pair-level score quantiles, including zero-top1-flip subsets.",
            "- `family_summary.parquet`: family directional accuracy and "
            "checkpoint-clustered intervals.",
            "- `label_direction_counts.parquet`: oriented top-1 label changes.",
            "- `drift_summary.parquet` and `drift_distributions.parquet`: exact "
            "moments plus bounded-memory log-scale distributions.",
            "- `margin_curves.parquet`: primary/secondary absolute and "
            "within-cell-quantile curves.",
            "- `certificate_summary.parquet`: empirical certificate coverage "
            "and flip rate among uncertified examples.",
            "- `summary.json`: protocol, gates, sources, and caveats.",
            "",
        ]
    )
    return "\n".join(lines)


def run_reanalysis(
    sources: Sequence[Source],
    output_dir: str | Path,
    manifests: Optional[Sequence[str | Path]] = None,
    reference_cell: Optional[Mapping[str, Any]] = None,
    n_boot: int = 2000,
    seed: int = 0,
    quantile_bins: int = QUANTILE_BIN_COUNT,
) -> Dict[str, Any]:
    if not sources:
        raise AnalysisError("at least one source is required")
    if n_boot < 0:
        raise AnalysisError("bootstrap replicate count cannot be negative")
    if quantile_bins <= 0:
        raise AnalysisError("quantile bin count must be positive")
    output = Path(output_dir).expanduser().resolve()
    validate_output_location(output, sources)

    manifests = list(manifests or [])
    if manifests and len(manifests) != len(sources):
        raise AnalysisError(
            "provide exactly one complete manifest per source, in source order"
        )
    manifest_gates = [
        check_complete_manifest(source.artifacts_root, manifest)
        for source, manifest in zip(sources, manifests)
    ]

    ref = dict(DEFAULT_REFERENCE_CELL)
    if reference_cell:
        ref.update(reference_cell)
    started = time.time()
    accumulator, source_summaries, target_length_rows = analyze_sources(
        sources, ref, n_boot, seed, quantile_bins
    )
    if not accumulator.pair_rows:
        raise AnalysisError("no valid direct comparison pairs were found")

    pair_frame = accumulator.pair_frame()
    family_frame = accumulator.family_frame()
    certificate_frame = accumulator.certificate_frame()
    label_frame = accumulator.labels_frame()
    curve_frame = accumulator.curves_frame()
    drift_summary, drift_distributions = accumulator.drift_frames()

    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "pair_metrics": pair_frame,
        "family_summary": family_frame,
        "certificate_summary": certificate_frame,
        "label_direction_counts": label_frame,
        "margin_curves": curve_frame,
        "drift_summary": drift_summary,
        "drift_distributions": drift_distributions,
    }
    for name, frame in artifacts.items():
        _atomic_parquet(frame, output / f"{name}.parquet")

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "protocol": "strict_paper_direct_pairs",
        "pair_eligibility": "pls_metrics_v2.classify_pair",
        "alignment_key": "example_id",
        "orientation": FAMILY_ORIENTATION_TEXT,
        "reference_cell": ref,
        "strict_unique_reference_gate": "required and passed for every source",
        "complete_manifest_gates": manifest_gates,
        "sources": source_summaries,
        "n_sources": len(sources),
        "n_valid_direct_pairs": len(pair_frame),
        "n_comparison_families": int(
            pair_frame["comparison_family"].nunique()
        ),
        "bootstrap": {
            "unit": "checkpoint",
            "replicates": n_boot,
            "seed": seed,
            "interval": "percentile_95",
            "rate_estimand": "pooled numerator/denominator after cluster resampling",
        },
        "margin_curves": {
            "primary": "oriented baseline top-two raw-logit margin",
            "secondary": "oriented baseline top-two probability margin",
            "absolute_buckets": {
                key: [
                    [lower, upper if math.isfinite(upper) else None]
                    for lower, upper in buckets
                ]
                for key, buckets in ABSOLUTE_MARGIN_BUCKETS.items()
            },
            "within_cell_quantile_bins": quantile_bins,
            "within_cell_definition": (
                "average empirical ranks within each oriented direct pair; "
                "ties remain together"
            ),
        },
        "certificate_rule": (
            "gamma > 2 * ||comparison_logits - baseline_logits||_infinity"
        ),
        "drift_distribution_note": (
            "Family-level distributions use fixed log-scale histograms for "
            "bounded memory; pair_metrics.parquet contains exact pair quantiles."
        ),
        "target_length": target_length_rows,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": sorted(f"{name}.parquet" for name in artifacts),
    }
    safe_summary = _json_safe(summary)
    write_json_atomic(safe_summary, output / "summary.json")
    write_text_atomic(
        build_markdown_report(safe_summary, family_frame, certificate_frame),
        output / "report.md",
    )
    write_text_atomic(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "n_valid_direct_pairs": len(pair_frame),
            },
            sort_keys=True,
        )
        + "\n",
        output / "_SUCCESS",
    )
    return safe_summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metrics-root",
        action="append",
        default=[],
        help=(
            "Existing strict <artifacts>/metrics_v2/<metrics_id> root. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        action="append",
        default=[],
        help=(
            "Artifact root containing inference/; strict references are still "
            "rebuilt and gated. May be repeated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Separate destination; paths inside source metrics_v2 trees are rejected.",
    )
    parser.add_argument(
        "--require-complete-manifest",
        action="append",
        default=[],
        metavar="JSONL",
        help="One inference manifest per source, in source order.",
    )
    parser.add_argument("--reference-batch-size", type=int, default=32)
    parser.add_argument(
        "--reference-seed-profile", default="production_like"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Checkpoint-cluster bootstrap replicates; zero disables intervals.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quantile-bins", type=int, default=QUANTILE_BIN_COUNT
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        sources = [
            source_from_metrics_root(path) for path in args.metrics_root
        ] + [
            source_from_artifacts_root(path) for path in args.artifacts_root
        ]
        reference_cell = dict(DEFAULT_REFERENCE_CELL)
        reference_cell["batch_size"] = args.reference_batch_size
        reference_cell["seed_profile"] = args.reference_seed_profile
        summary = run_reanalysis(
            sources=sources,
            output_dir=args.output_dir,
            manifests=args.require_complete_manifest,
            reference_cell=reference_cell,
            n_boot=args.bootstrap,
            seed=args.seed,
            quantile_bins=args.quantile_bins,
        )
    except (AnalysisError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {summary['n_valid_direct_pairs']} direct pairs across "
        f"{summary['n_comparison_families']} families to "
        f"{Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
