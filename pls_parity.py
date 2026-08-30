#!/usr/bin/env python3
"""Compare matched prediction artifacts from two serving configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def enrich(spec: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(spec)
    train_dir = spec.get("train_dir")
    if train_dir:
        config_path = Path(train_dir) / "train_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("model_type", "model_size", "seed"):
                enriched.setdefault(key, config.get(key))
            enriched.setdefault("dataset", config.get("data_path"))
    return enriched


def index_manifest(path: Path, context_kind: str | None, key_fields: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in read_jsonl(path):
        if raw.get("job_type") != "inference":
            continue
        if context_kind and raw.get("context_kind") != context_kind:
            continue
        spec = enrich(raw)
        key = tuple(spec.get(field) for field in key_fields)
        if any(value is None for value in key):
            raise ValueError(f"manifest row lacks a parity key field: {key_fields}")
        if key in indexed:
            raise ValueError(f"duplicate parity cell: {key}")
        indexed[key] = spec
    return indexed


def load_predictions(root: Path, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    run_dir = root / "inference" / str(spec["job_id"])
    if not (run_dir / "_SUCCESS").exists():
        raise FileNotFoundError(f"incomplete inference artifact: {run_dir}")
    rows = read_jsonl(run_dir / "predictions.jsonl")
    indexed = {str(row["example_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate example identifiers in {run_dir}")
    return indexed


def metrics(truth: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--comparison-manifest", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--baseline-context-kind", default="reference_cell")
    parser.add_argument("--comparison-context-kind", default=None)
    parser.add_argument("--key-fields", default="model_type,dataset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    key_fields = [field.strip() for field in args.key_fields.split(",") if field.strip()]
    baseline = index_manifest(Path(args.baseline_manifest), args.baseline_context_kind, key_fields)
    comparison = index_manifest(Path(args.comparison_manifest), args.comparison_context_kind, key_fields)
    if set(baseline) != set(comparison):
        missing = sorted(set(baseline) ^ set(comparison), key=str)
        raise ValueError(f"parity manifest coverage differs for {len(missing)} cells")

    rows = []
    for key in sorted(baseline, key=str):
        left = load_predictions(Path(args.baseline_root), baseline[key])
        right = load_predictions(Path(args.comparison_root), comparison[key])
        if set(left) != set(right):
            raise ValueError(f"example coverage differs for cell {key}")
        example_ids = sorted(left)
        truth = [int(left[item]["true_class_index"]) for item in example_ids]
        if truth != [int(right[item]["true_class_index"]) for item in example_ids]:
            raise ValueError(f"true labels differ for cell {key}")
        left_pred = [int(left[item]["pred_class_index"]) for item in example_ids]
        right_pred = [int(right[item]["pred_class_index"]) for item in example_ids]
        left_metrics = metrics(truth, left_pred)
        right_metrics = metrics(truth, right_pred)
        rows.append({
            "key": dict(zip(key_fields, key)),
            "n": len(example_ids),
            "baseline": left_metrics,
            "comparison": right_metrics,
            "delta": {name: right_metrics[name] - left_metrics[name] for name in left_metrics},
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} parity cells to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
