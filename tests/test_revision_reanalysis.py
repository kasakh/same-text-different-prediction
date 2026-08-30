"""Focused tests for score-drift reanalysis."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ModuleNotFoundError:
    HAS_PARQUET = False

from pls_revision_reanalysis import (
    align_and_measure_pair,
    assign_absolute_margin_bins,
    assign_within_cell_quantile_bins,
    classify_pair,
    correctness_transition_counts,
    empirical_certificate,
    orient_pair,
    run_reanalysis,
    source_from_artifacts_root,
    total_variation,
)


def prediction_frame(
    example_ids,
    predictions,
    truths,
    probabilities,
    logits,
):
    return pd.DataFrame(
        {
            "example_id": [str(value) for value in example_ids],
            "pred_label": predictions,
            "true_label": truths,
            "probs": probabilities,
            "raw_logits": logits,
        }
    ).set_index("example_id")


def context(context_id: str, **overrides):
    row = {
        "context_id": context_id,
        "train_run_id": "checkpoint",
        "dataset": "sst2",
        "split": "test",
        "batch_size": 32,
        "order_name": "identity",
        "batch_position": None,
        "composition_policy": None,
        "padding_policy": None,
        "dtype": "fp32",
        "quantization": "none",
        "device": "cuda",
        "backend": "pytorch",
        "execution_profile": None,
        "seed_profile": "production_like",
        "seed": 42,
        "position_id_policy": "default",
        "repeat_id": 0,
    }
    row.update(overrides)
    return row


class TestScoreDrift(unittest.TestCase):
    def test_true_total_variation_is_half_l1(self):
        self.assertAlmostEqual(
            total_variation([0.70, 0.20, 0.10], [0.40, 0.40, 0.20]),
            0.30,
        )

    def test_zero_top1_flip_can_have_nonzero_score_drift(self):
        baseline = prediction_frame(
            ["x"],
            [0],
            [0],
            [[0.70, 0.30]],
            [[2.0, 1.0]],
        )
        comparison = prediction_frame(
            ["x"],
            [0],
            [0],
            [[0.60, 0.40]],
            [[1.8, 1.1]],
        )
        measured = align_and_measure_pair(baseline, comparison)
        self.assertFalse(measured.flips[0])
        self.assertAlmostEqual(measured.tv_drift[0], 0.10)
        self.assertAlmostEqual(measured.max_logit_drift[0], 0.20)


class TestDirectionalAccuracy(unittest.TestCase):
    def test_harmful_and_beneficial_are_oriented(self):
        counts = correctness_transition_counts(
            pred_baseline=[0, 1, 0],
            pred_comparison=[1, 1, 1],
            true_labels=[0, 0, 1],
        )
        self.assertEqual(counts["correct_to_wrong"], 1)
        self.assertEqual(counts["wrong_to_correct"], 1)
        self.assertEqual(counts["net_correct"], 0)

        baseline = prediction_frame(
            ["beneficial", "harmful"],
            [0, 0],
            [1, 0],
            [[0.7, 0.3], [0.7, 0.3]],
            [[2.0, 1.0], [2.0, 1.0]],
        )
        comparison = prediction_frame(
            ["harmful", "beneficial"],
            [1, 1],
            [0, 1],
            [[0.3, 0.7], [0.3, 0.7]],
            [[1.0, 2.0], [1.0, 2.0]],
        )
        measured = align_and_measure_pair(baseline, comparison)
        self.assertEqual(int(measured.harmful.sum()), 1)
        self.assertEqual(int(measured.beneficial.sum()), 1)


class TestCertificate(unittest.TestCase):
    def test_certificate_uses_strict_gamma_greater_than_twice_linf(self):
        gamma, drift, covered = empirical_certificate(
            [3.0, 1.0], [2.7, 1.2]
        )
        self.assertAlmostEqual(gamma, 2.0)
        self.assertAlmostEqual(drift, 0.3)
        self.assertTrue(covered)

        gamma, drift, covered = empirical_certificate(
            [1.0, 0.0], [0.5, 0.5]
        )
        self.assertAlmostEqual(gamma, 1.0)
        self.assertAlmostEqual(drift, 0.5)
        self.assertFalse(covered, "equality must remain uncertified")


class TestMarginBins(unittest.TestCase):
    def test_absolute_buckets_are_left_closed(self):
        probability = assign_absolute_margin_bins(
            [0.0, 0.009, 0.01, 0.05, 0.75, 1.0],
            "probability_top2",
        )
        np.testing.assert_array_equal(probability, [0, 0, 1, 2, 6, 6])

        logits = assign_absolute_margin_bins(
            [0.0, 0.25, 1.0, 4.0, 100.0], "logit_top2"
        )
        np.testing.assert_array_equal(logits, [0, 4, 6, 8, 8])

    def test_within_cell_quantiles_are_deterministic_and_keep_ties(self):
        np.testing.assert_array_equal(
            assign_within_cell_quantile_bins([0, 1, 2, 3, 4], 5),
            [0, 1, 2, 3, 4],
        )
        np.testing.assert_array_equal(
            assign_within_cell_quantile_bins([0.2] * 5, 5),
            [2, 2, 2, 2, 2],
        )


class TestPairOrientation(unittest.TestCase):
    def test_precision_orientation_is_independent_of_discovery_order(self):
        fp32 = context("z-reference", dtype="fp32")
        bf16 = context("a-spoke", dtype="bf16")
        family, axis = classify_pair(pd.Series(bf16), pd.Series(fp32))
        self.assertEqual(family, "precision_drift")
        baseline, comparison = orient_pair(bf16, fp32, axis)
        self.assertEqual(baseline["dtype"], "fp32")
        self.assertEqual(comparison["dtype"], "bf16")

    def test_reference_batch_precedes_other_values_then_numeric_order_applies(self):
        batch_1 = context("batch-1", batch_size=1)
        batch_32 = context("batch-32", batch_size=32)
        batch_64 = context("batch-64", batch_size=64)

        _, axis = classify_pair(pd.Series(batch_1), pd.Series(batch_32))
        baseline, comparison = orient_pair(batch_1, batch_32, axis)
        self.assertEqual(baseline["batch_size"], 32)
        self.assertEqual(comparison["batch_size"], 1)

        _, axis = classify_pair(pd.Series(batch_64), pd.Series(batch_1))
        baseline, comparison = orient_pair(batch_64, batch_1, axis)
        self.assertEqual(baseline["batch_size"], 1)
        self.assertEqual(comparison["batch_size"], 64)


@unittest.skipUnless(HAS_PARQUET, "pyarrow is required for parquet output")
class TestEndToEnd(unittest.TestCase):
    def test_strict_direct_pair_outputs_and_target_length_caveat(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            artifacts = base / "artifacts"
            inference = artifacts / "inference"

            reference = context("reference")
            spoke = context("spoke", dtype="bf16")
            rows = {
                "reference": [
                    {
                        "example_id": "a",
                        "pred_label": 0,
                        "true_label": 0,
                        "top1_prob": 0.8,
                        "top2_prob": 0.2,
                        "margin": 0.6,
                        "probs": [0.8, 0.2],
                        "raw_logits": [2.0, 0.0],
                        "num_labels": 2,
                    },
                    {
                        "example_id": "b",
                        "pred_label": 0,
                        "true_label": 1,
                        "top1_prob": 0.525,
                        "top2_prob": 0.475,
                        "margin": 0.05,
                        "probs": [0.525, 0.475],
                        "raw_logits": [1.0, 0.9],
                        "num_labels": 2,
                    },
                ],
                "spoke": [
                    {
                        "example_id": "b",
                        "pred_label": 0,
                        "true_label": 1,
                        "top1_prob": 0.53,
                        "top2_prob": 0.47,
                        "margin": 0.06,
                        "probs": [0.53, 0.47],
                        "raw_logits": [1.01, 0.89],
                        "num_labels": 2,
                    },
                    {
                        "example_id": "a",
                        "pred_label": 1,
                        "true_label": 0,
                        "top1_prob": 0.8,
                        "top2_prob": 0.2,
                        "margin": 0.6,
                        "probs": [0.2, 0.8],
                        "raw_logits": [0.0, 2.0],
                        "num_labels": 2,
                    },
                ],
            }
            for record in (reference, spoke):
                run_dir = inference / record["context_id"]
                run_dir.mkdir(parents=True)
                spec = dict(record)
                spec["job_id"] = record["context_id"]
                spec["order_policy"] = {
                    "name": spec.pop("order_name"),
                    "type": "identity",
                }
                (run_dir / "job_spec.json").write_text(
                    json.dumps(spec), encoding="utf-8"
                )
                (run_dir / "predictions.jsonl").write_text(
                    "".join(
                        json.dumps(row) + "\n"
                        for row in rows[record["context_id"]]
                    ),
                    encoding="utf-8",
                )
                (run_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")

            output = base / "revision-output"
            summary = run_reanalysis(
                [source_from_artifacts_root(artifacts)],
                output,
                n_boot=20,
                seed=4,
            )
            self.assertEqual(summary["n_valid_direct_pairs"], 1)
            self.assertEqual(summary["target_length"][0]["status"], "unavailable")
            family = pd.read_parquet(output / "family_summary.parquet").iloc[0]
            self.assertEqual(family["comparison_family"], "precision_drift")
            self.assertEqual(int(family["correct_to_wrong"]), 1)
            self.assertAlmostEqual(float(family["net_accuracy_delta"]), -0.5)
            self.assertTrue((output / "margin_curves.parquet").exists())
            self.assertIn(
                "Target-length analysis unavailable",
                (output / "report.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
