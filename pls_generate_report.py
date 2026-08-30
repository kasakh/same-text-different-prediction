#!/usr/bin/env python3
"""Generate neutral summary tables from a metrics directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXCLUDED_TABLES = {"all_pairwise_diagnostics_metrics"}


def family_name(path: Path) -> str:
    return path.name.removesuffix("_metrics.parquet")


def summarize(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    required = {"flip_rate", "n_flips", "n_examples"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    total_examples = int(frame["n_examples"].sum())
    total_flips = int(frame["n_flips"].sum())
    result: dict[str, object] = {
        "family": family_name(path),
        "pairs": int(len(frame)),
        "pair_examples": total_examples,
        "flips": total_flips,
        "micro_flip_rate": total_flips / total_examples if total_examples else None,
        "mean_pair_flip_rate": float(frame["flip_rate"].mean()) if len(frame) else None,
        "max_pair_flip_rate": float(frame["flip_rate"].max()) if len(frame) else None,
    }
    for column in ("mean_prob_drift", "mean_tv_prob_drift", "mean_max_abs_logit_diff", "mean_max_logit_drift"):
        if column in frame.columns:
            result[column] = float(frame[column].mean())
    return result


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = ["family", "pairs", "pair_examples", "flips", "micro_flip_rate", "mean_pair_flip_rate", "max_pair_flip_rate"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--position-results")
    parser.add_argument("--parity-results")
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root)
    files = [path for path in sorted(metrics_root.glob("*_metrics.parquet")) if family_name(path) not in EXCLUDED_TABLES]
    if not files:
        raise FileNotFoundError(f"no family metric tables under {metrics_root}")
    rows = sorted((summarize(path) for path in files), key=lambda row: str(row["family"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "family_summary.csv", index=False)
    table = markdown_table(rows)
    (output_dir / "family_summary.md").write_text("# Serving-context summary\n\n" + table, encoding="utf-8")

    report = ["# Text Classifier Non-Determinism", "", "## Aggregate metrics", "", table.rstrip(), ""]
    if args.position_results:
        position = json.loads(Path(args.position_results).read_text(encoding="utf-8"))
        report.extend(["## Position diagnostic", "", f"Validated cells: {position.get('gates', {}).get('cells_valid', 'unknown')}", ""])
    if args.parity_results:
        parity = json.loads(Path(args.parity_results).read_text(encoding="utf-8"))
        report.extend(["## Quality parity", "", f"Matched cells: {len(parity)}", ""])
    report.extend(["## Scope", "", "This report summarizes supplied artifacts. It does not infer missing cells or merge structural position changes with numerical execution changes.", ""])
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} family rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
