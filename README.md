# Text Classifier Non-Determinism

This repository is an anonymous reproducibility package for a 180-checkpoint study of text-classifier non-determinism under changes in serving context. The repository uses non-determinism as its public-facing name while distinguishing two technical questions. Same-context run-to-run nondeterminism asks whether an identical execution changes across repetitions. Serving-context non-invariance asks whether outputs change when the checkpoint, text, and tokenizer stay fixed but hidden serving conditions change. The study also separates structural changes to represented token positions from numerical execution changes that preserve the represented input.

Exact model dimensions, library versions, hardware, determinism settings,
and quantization APIs are consolidated in `REPRODUCIBILITY.md`.

## Included study

The main grid contains 180 independently trained checkpoints:

- Datasets: `SetFit/sst2`, `SetFit/sst5`, `SetFit/emotion`, `SetFit/ag_news`, and `SetFit/imdb`.
- Classifier formulations: BERT sequence classification, masked-language-model label scoring, causal label-token scoring, and causal sequence scoring.
- Model sizes: small, medium, and large configurations defined in `pls_train_classifier.py`.
- Seeds: 40, 41, and 42.

The product is `5 datasets x 4 formulations x 3 sizes x 3 seeds = 180`.

## Exact environment

The recorded execution stack used Python 3.12.13, CUDA 13.0, and the exact direct package versions in `requirements.txt`. Create an isolated environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The accelerator wheel index in `requirements.txt` is public. For CPU-only planning and analysis, replace the accelerator-specific torch line with the matching public CPU build while retaining torch 2.11.0.

Set cache roots per invocation when needed:

```bash
HF_HOME=./cache/huggingface TOKENIZERS_PARALLELISM=false python pls_plan_train.py --config configs/study_180.yaml --explain-counts
```

## Public model and dataset bindings

GPT-2 tokenizer and pretrained-weight access use `openai-community/gpt2` at immutable revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Every trained checkpoint stores its tokenizer and label mapping. Inference requires those checkpoint-bundled assets and does not fall back to an unpinned tokenizer.

The staged source retained public dataset identifiers but did not retain immutable dataset commit identifiers or an immutable BERT tokenizer commit. To prevent a false reproducibility claim, this package does not invent them. Supply verified dataset commits in a JSON mapping and enable strict checking:

```json
{
  "ag_news": "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT",
  "emotion": "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT",
  "imdb": "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT",
  "sst2": "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT",
  "sst5": "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT"
}
```

```bash
TCI_DATASET_REVISIONS=configs/dataset_revisions.json TCI_REQUIRE_DATASET_REVISIONS=1 python scripts/run_manifest.py --manifest outputs/study_180/manifests/train_jobs.jsonl --job-type train
```

This missing provenance is the only source-level blocker to bit-for-bit reconstruction of dataset acquisition. Existing local JSON dataset snapshots can instead be passed as dataset values and are hashed by the training pipeline.

## Offline planning smoke test

These commands do not load datasets or models:

```bash
python pls_plan_train.py --config configs/smoke.yaml --explain-counts
python pls_plan_train.py --config configs/study_180.yaml --write-all --explain-counts
python pls_plan_v2.py --config configs/serving_context.yaml --checkpoints-from examples/checkpoints.example.jsonl --output-root outputs/planning_smoke --dry-run --explain-counts
```

Expected training planner result:

```text
total jobs      : 180
missing (to run): 180
```

The missing count decreases when valid completed checkpoints already exist under the configured output root.

## Plan and reproduce the 180 checkpoints

Write the deterministic training manifest:

```bash
python pls_plan_train.py --config configs/study_180.yaml --write-all --write-manifest outputs/study_180/manifests/train_jobs.jsonl
```

Inspect two commands without running them:

```bash
python scripts/run_manifest.py --manifest outputs/study_180/manifests/train_jobs.jsonl --job-type train --limit 2 --dry-run
```

Run all training jobs sequentially:

```bash
python scripts/run_manifest.py --manifest outputs/study_180/manifests/train_jobs.jsonl --job-type train
```

Each training job writes `train_config.json`, `label_mapping.json`, a tokenizer directory, checkpoints, a best-checkpoint binding, environment metadata, and a success marker under `outputs/study_180/train/`.

## Plan and run serving contexts

After training completes, create reference inference bindings:

```bash
python pls_plan_jobs.py --config configs/study_180.yaml --write_all
```

Expand those bindings into the reference cell, one-axis contexts, bounded interactions, and target-anchored position/composition/padding contexts:

```bash
python pls_plan_v2.py --config configs/serving_context.yaml --checkpoints-from outputs/study_180/manifests/desired_inference_jobs.jsonl --output-root outputs/study_180 --write-manifest outputs/study_180/manifests/serving_context_jobs.jsonl
```

Run the contexts:

```bash
python scripts/run_manifest.py --manifest outputs/study_180/manifests/serving_context_jobs.jsonl --job-type inference
```

Expected inference outputs are one directory per job under `outputs/study_180/inference/`. Each completed directory contains the exact job specification, predictions with raw logits and probabilities, an environment fingerprint, status metadata, and a success marker.

## Metrics and score-drift reanalysis

Build strictly matched single-axis comparisons:

```bash
python pls_metrics_v2.py --artifacts-root outputs/study_180 --metrics-id main --require-complete-manifest outputs/study_180/manifests/serving_context_jobs.jsonl --require-strict-reference
```

Compute true probability total variation, raw-logit drift, directional accuracy changes, margin strata, and an empirical margin certificate:

```bash
python pls_revision_reanalysis.py --metrics-root outputs/study_180/metrics_v2/main --output-dir outputs/study_180/score_drift --require-complete-manifest outputs/study_180/manifests/serving_context_jobs.jsonl
```

Generate configurable tables without embedded result paths:

```bash
python pls_generate_report.py --metrics-root outputs/study_180/metrics_v2/main --output-dir outputs/study_180/report
```

## Position consistency and diagnostic workflow

Plan the 10 medium-size consistency-training checkpoints:

```bash
python pls_plan_train_posid_consistency.py --output-root outputs/position_consistency --write-manifest outputs/position_consistency/manifests/train_jobs.jsonl
python scripts/run_manifest.py --manifest outputs/position_consistency/manifests/train_jobs.jsonl --job-type train
```

Plan the remove-and-restore diagnostic from an explicit checkpoint manifest or training root:

```bash
python pls_plan_posid_diag.py --train-root outputs/position_consistency --output-root outputs/position_diagnostic --write-manifest outputs/position_diagnostic/manifests/jobs.jsonl --posid-policies arange,mask_derived --no-ref-margins
python scripts/run_manifest.py --manifest outputs/position_diagnostic/manifests/jobs.jsonl --job-type inference
python pls_analyze_posid_diag.py --root outputs/position_diagnostic --manifest outputs/position_diagnostic/manifests/jobs.jsonl --policies arange,mask_derived
```

For margin-stratified target selection, pass `--reference-predictions` to the planner instead of `--no-ref-margins`.

## Quality parity

`pls_parity.py` compares any two matched inference manifests. It requires identical cell coverage, example identifiers, and true labels:

```bash
python pls_parity.py --baseline-manifest outputs/study_180/manifests/serving_context_jobs.jsonl --comparison-manifest outputs/position_consistency/manifests/full_test_jobs.jsonl --baseline-root outputs/study_180 --comparison-root outputs/position_consistency --comparison-context-kind consistency_full_test --output outputs/position_consistency/parity.json
```

## Validation

Run the offline checks from the repository root:

```bash
bash scripts/check.sh
```

The check compiles Python files, runs discoverable offline unit tests, verifies the 180-job count, and scans all content and filenames for identifying strings, secret material, private hosts, absolute paths, binary payloads, and forbidden dash characters.

## Focused experiment scope

Included execution scope is the generic training, inference, one-axis serving-context planning, target-anchored position/composition/padding planning, metrics, score-drift reanalysis, position-rule consistency diagnostic, quality parity, and configurable report generation pipeline.

Specialized backend compilation, alternate-runtime export, quantization-only peer experiments, invariant-kernel experiments, generation accumulation, target-length studies, systems scheduling, native position-free models, online verification, and operator attribution are not included. Their staged drivers depended on private filesystem roots, additional environments, unbundled base scripts, or result-specific assumptions. Publishing them unchanged would create nonfunctional or misleading entry points.

## Limitations

- No checkpoints, datasets, raw predictions, generated job manifests, run identifiers, or result tables are included.
- Full execution requires public dataset/model access or a prepared public cache and suitable accelerator hardware.
- Public dataset revisions and the BERT tokenizer revision must be supplied because the staged source did not record them.
- The sequential manifest runner is a reproducibility reference, not a cluster scheduler.
- The exact direct package stack is pinned. Platform drivers and transitive system libraries remain platform dependent and are recorded at run time.
- Batch-invariant kernels address numerical batch-shape changes under one execution configuration. They do not remove structural token-position changes or cross-format and cross-hardware differences.
