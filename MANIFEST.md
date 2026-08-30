# File Manifest

Provenance categories:

- Adapted core source: copied from staged study code, then sanitized and parameterized.
- Adapted test: copied from staged tests, then sanitized or made dependency-aware.
- Sanitized configuration: derived from staged study configuration with relative roots.
- New release glue: written for this reproducibility package.
- New release metadata: written for packaging and documentation.
- Placeholder example: synthetic paths and identifiers only.

| File | Provenance category | Purpose |
|---|---|---|
| `.gitignore` | New release metadata | Excludes environments, caches, checkpoints, binaries, and generated outputs. |
| `MANIFEST.md` | New release metadata | Complete release inventory and provenance categories. |
| `README.md` | New release metadata | Non-determinism framing, setup, commands, outputs, scope, and limitations. |
| `pyproject.toml` | New release metadata | Non-determinism package metadata and command entry points. |
| `requirements.txt` | New release metadata | Exact direct package versions from the recorded execution stack. |
| `configs/serving_context.yaml` | Sanitized configuration | One-axis serving-context and target-anchored design for 180 checkpoints. |
| `configs/smoke.yaml` | Sanitized configuration | One-job planning and training smoke configuration. |
| `configs/study_180.yaml` | Sanitized configuration | Main five-dataset, four-formulation, three-size, three-seed grid. |
| `examples/checkpoints.example.jsonl` | Placeholder example | Synthetic checkpoint binding rows for planner inspection. |
| `pls/__init__.py` | Adapted core source | Package marker. |
| `pls/common.py` | Adapted core source | Hashing, atomic I/O, environment capture, and seed helpers. |
| `pls/data.py` | Adapted core source | Public dataset loading, explicit revision binding, ordering, and splitting. |
| `pls/serve_invariant.py` | Adapted core source | CPU-only flip, drift, accuracy, and interval helpers. |
| `pls/target_batch.py` | Adapted core source | Deterministic peer selection, target placement, and padded-width control. |
| `pls_analyze_posid_diag.py` | Adapted core source | Completeness-gated position-rule diagnostic analysis. |
| `pls_generate_report.py` | New release glue | Configurable metric table and neutral report generation. |
| `pls_metrics_v2.py` | Adapted core source | Strict reference selection and matched comparison metrics. |
| `pls_parity.py` | New release glue | Configurable matched quality-parity comparison. |
| `pls_plan_jobs.py` | Adapted core source | Deterministic training and reference-inference job expansion. |
| `pls_plan_posid_diag.py` | Adapted core source | Position-rule remove-and-restore diagnostic planning. |
| `pls_plan_train.py` | Adapted core source | Budgeted training-grid planner. |
| `pls_plan_train_posid_consistency.py` | Adapted core source | Position-consistency training planner. |
| `pls_plan_v2.py` | Adapted core source | Reference, one-axis, interaction, and target-anchored context planner. |
| `pls_revision_reanalysis.py` | Adapted core source | Score drift, total variation, directional accuracy, margin, and certificate analysis. |
| `pls_run_inference_context.py` | Adapted core source | Checkpoint-bound standard and target-anchored inference. |
| `pls_train_classifier.py` | Adapted core source | Four classifier formulations and checkpoint-bundled preprocessing assets. |
| `scripts/check.sh` | New release glue | Fixed offline validation sequence. |
| `scripts/check_anonymity.py` | New release glue | Content, filename, secret, path, dash, and binary scan. |
| `scripts/run_manifest.py` | New release glue | Sequential training and inference manifest runner. |
| `tests/__init__.py` | Adapted test | Test package marker. |
| `tests/test_planners.py` | New release glue | 180-job and serving-design count tests. |
| `tests/test_position_ids.py` | Adapted test | Position-ID behavior and wrapper regression tests. |
| `tests/test_revision_reanalysis.py` | Adapted test | Score-drift and end-to-end artifact analysis tests. |
| `tests/test_target_batch.py` | Adapted test | Pure CPU controlled-batch construction tests. |

No checkpoints, datasets, predictions, result tables, generated job manifests, run identifiers, paper files, binary payloads, or private records are included.
