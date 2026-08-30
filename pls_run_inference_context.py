#!/usr/bin/env python3
"""Run one offline PyTorch inference context for label-stability sweeps.

Input can be a full JSON job spec or CLI args. A context is intentionally
explicit: same checkpoint, same dataset split, configurable batch size/order,
dtype, quantization placeholder, device, seed profile, and shard.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import BertTokenizer, GPT2Tokenizer

from pls.common import (
    apply_seed_profile,
    artifact_valid,
    canonical_json,
    ensure_dir,
    environment_fingerprint,
    mark_success,
    read_json,
    sha256_file,
    sha256_tree,
    to_jsonable,
    write_json_atomic,
    write_jsonl_atomic,
)
from pls.data import (
    apply_order_policy,
    batch_iter,
    examples_sha256,
    load_examples,
    shard_rows,
)
from pls_train_classifier import (
    DEFAULT_PRETRAINED_GPT2_MODEL,
    DEFAULT_PRETRAINED_GPT2_REVISION,
    BertModule,
    GPTModule,
    MLMModule,
    TextDataset,
)
from pls.target_batch import build_context


def _percentile(values: List[float], quantile: float) -> float:
    """Linear-interpolated percentile without adding a NumPy dependency."""
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_tokenizer(train_dir: Path, model_type: str):
    tok_dir = train_dir / "tokenizer"
    if tok_dir.exists():
        if model_type in ["BERT", "pseudo-gen-MLM"]:
            return BertTokenizer.from_pretrained(str(tok_dir))
        tok = GPT2Tokenizer.from_pretrained(
            str(tok_dir), padding_side="left"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    raise FileNotFoundError(
        f"checkpoint tokenizer directory is missing: {tok_dir}"
    )


def _load_label_mapping(train_dir: Path, model_type: str, labels_from_config: Optional[List[int]] = None, tokenizer=None) -> Dict[str, Any]:
    mapping_path = train_dir / "label_mapping.json"
    if mapping_path.exists():
        return read_json(mapping_path)
    if labels_from_config is None:
        raise FileNotFoundError(f"No label_mapping.json in {train_dir} and no labels in train_config")
    labels = [int(x) for x in labels_from_config]
    label_token_ids = None
    if model_type in ["pseudo-gen-MLM", "pure-gen-GPT2", "pseudo-gen-GPT2"]:
        label_token_ids = [tokenizer.convert_tokens_to_ids(f"[label_{label}]") for label in labels]
    return {"labels": labels, "label_token_ids": label_token_ids}


def _validate_pretrained_inference_binding(
    spec: Dict[str, Any],
    train_dir: Path,
    checkpoint_path: str,
    train_config: Dict[str, Any],
) -> None:
    """Bind a B2 inference cell to its exact training artifact identity."""
    if spec.get("experiment_id") != "pretrained_gpt2_b2_v1":
        return
    fixed_spec = {
        "model_type": "pseudo-gen-GPT2",
        "model_size": "pretrained",
        "pretrained_model_name_or_path": DEFAULT_PRETRAINED_GPT2_MODEL,
        "pretrained_model_revision": DEFAULT_PRETRAINED_GPT2_REVISION,
    }
    if any(spec.get(field) != value for field, value in fixed_spec.items()):
        raise ValueError(
            "pretrained inference spec differs from the fixed B2 model"
        )
    if spec.get("training_regime") not in (
        "conventional",
        "consistency",
    ):
        raise ValueError("pretrained inference has an unknown training regime")
    expected = {
        "experiment_id": spec.get("experiment_id"),
        "training_regime": spec.get("training_regime"),
        "run_id": spec.get("train_run_id"),
        "data_path": spec.get("dataset"),
        "seed": spec.get("train_seed"),
        "model_type": spec.get("model_type"),
        "model_size": spec.get("model_size"),
        "pretrained_model_name_or_path": spec.get(
            "pretrained_model_name_or_path"
        ),
        "pretrained_model_revision": spec.get(
            "pretrained_model_revision"
        ),
    }
    mismatches = [
        field
        for field, value in expected.items()
        if type(train_config.get(field)) is not type(value)
        or train_config.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            "pretrained inference spec does not match train_config.json: "
            f"{mismatches}"
        )
    expected_training = {
        "seed_profile": "strict",
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "num_warmup_steps": 0,
        "num_samples": None,
        "early_stopping_patience": 2,
        "use_test_as_val": False,
        "ignore_pad_boundary_loss": True,
        "resolved_pretrained_model_revision": (
            DEFAULT_PRETRAINED_GPT2_REVISION
        ),
        "position_id_policy": (
            "default"
            if spec.get("training_regime") == "conventional"
            else "mask_derived"
        ),
        "resolved_training_position_id_policy": (
            "arange"
            if spec.get("training_regime") == "conventional"
            else "mask_derived"
        ),
    }
    invalid_training = [
        field
        for field, value in expected_training.items()
        if type(train_config.get(field)) is not type(value)
        or train_config.get(field) != value
    ]
    hashes = (
        train_config.get("train_data_sha256"),
        train_config.get("validation_data_sha256"),
    )
    if invalid_training or any(
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise ValueError(
            "pretrained training contract is invalid for inference: "
            f"{invalid_training}"
        )
    root = Path(str(spec["output_root"])).expanduser().resolve()
    expected_train_dir = (
        root / "train" / str(spec["train_run_id"])
    ).resolve()
    if train_dir.expanduser().resolve() != expected_train_dir:
        raise ValueError(
            "pretrained inference train_dir is not the canonical run "
            "directory"
        )
    configured_root = Path(
        str(train_config.get("output_root", ""))
    ).expanduser().resolve()
    if configured_root != root:
        raise ValueError(
            "pretrained inference output_root differs from training"
        )
    if (
        spec.get("position_id_policy") not in ("arange", "mask_derived")
        or spec.get("padding_policy")
        not in ("static_max_length", "dynamic_longest")
        or spec.get("static_max_length") != 512
        or spec.get("batch_size") != 8
        or spec.get("max_targets") != 2000
    ):
        raise ValueError(
            "pretrained inference context differs from the fixed B2 design"
        )
    best = read_json(train_dir / "best_checkpoint.json")
    bound_checkpoint = Path(
        str(best.get("best_model_path", ""))
    ).expanduser().resolve()
    if Path(checkpoint_path).expanduser().resolve() != bound_checkpoint:
        raise ValueError(
            "pretrained inference checkpoint differs from training metadata"
        )


class _ModernInferenceWrapper(torch.nn.Module):
    """Thin eval wrapper over a MERGED modern model (LoRA already folded in, no
    peft dependency). Exposes the same forward contract as the legacy Lightning
    modules: forward(input_ids, attention_mask, labels=None, position_ids=None)
    -> object with .raw_logits / .probs."""

    def __init__(self, hf_model, model_type, label_token_ids=None):
        super().__init__()
        self.model = hf_model
        self.model_type = model_type
        self.label_token_ids = label_token_ids

    def forward(self, input_ids, attention_mask, labels=None, position_ids=None):
        if self.model_type == "modern-decoder":
            # final-position logits only + no KV cache (finding 1)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, logits_to_keep=1, use_cache=False)
            raw_logits = out.logits[:, -1, :][:, self.label_token_ids]
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids)
            raw_logits = out.logits
        return type("ModelOutput", (), {"loss": None, "raw_logits": raw_logits,
                                        "probs": torch.softmax(raw_logits, dim=-1)})()


def _load_model(
    checkpoint_path: str,
    train_dir: Path,
    map_location: str,
    device: torch.device,
    expected_spec: Optional[Dict[str, Any]] = None,
):
    train_config = read_json(train_dir / "train_config.json")
    model_type = train_config["model_type"]
    model_size = train_config["model_size"]
    is_pretrained_gpt2 = (
        model_size == "pretrained"
        and model_type in ("pure-gen-GPT2", "pseudo-gen-GPT2")
    )
    if expected_spec is not None:
        _validate_pretrained_inference_binding(
            expected_spec,
            train_dir,
            checkpoint_path,
            train_config,
        )
    if train_config.get("modern"):
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        merged_dir = train_dir / "merged_model"
        tokenizer = AutoTokenizer.from_pretrained(merged_dir)
        mapping = read_json(train_dir / "label_mapping.json")
        if model_type == "modern-decoder":
            hf = AutoModelForCausalLM.from_pretrained(merged_dir)
        else:
            hf = AutoModelForSequenceClassification.from_pretrained(merged_dir)
        model = _ModernInferenceWrapper(hf, model_type, mapping.get("label_token_ids"))
        model.eval().to(device)
        return model, tokenizer, train_config, mapping
    if is_pretrained_gpt2:
        if not (train_dir / "tokenizer").is_dir():
            raise ValueError(
                "pretrained checkpoint is missing its tokenizer directory"
            )
        if not (train_dir / "label_mapping.json").exists():
            raise ValueError(
                "pretrained checkpoint is missing label_mapping.json"
            )
    tokenizer = _load_tokenizer(train_dir, model_type)
    mapping = _load_label_mapping(train_dir, model_type, train_config.get("labels"), tokenizer)
    labels = [int(x) for x in mapping["labels"]]
    label_token_ids = mapping.get("label_token_ids")
    vocab_size = int(train_config.get("vocab_size", len(tokenizer)))
    if is_pretrained_gpt2:
        tokenizer_dir = train_dir / "tokenizer"
        mapping_path = train_dir / "label_mapping.json"
        expected_tokenizer_hash = train_config.get(
            "tokenizer_artifact_sha256"
        )
        if not tokenizer_dir.is_dir() or not expected_tokenizer_hash:
            raise ValueError(
                "pretrained checkpoint requires a bound tokenizer artifact"
            )
        actual_tokenizer_hash = sha256_tree(tokenizer_dir)
        if actual_tokenizer_hash != expected_tokenizer_hash:
            raise ValueError(
                "pretrained tokenizer artifact hash mismatch: "
                f"expected {expected_tokenizer_hash}, "
                f"got {actual_tokenizer_hash}"
            )
        expected_mapping_hash = train_config.get(
            "label_mapping_artifact_sha256"
        )
        if (
            not expected_mapping_hash
            or sha256_file(mapping_path) != expected_mapping_hash
        ):
            raise ValueError("pretrained label-mapping artifact hash mismatch")
        if mapping.get("labels") != train_config.get("labels") or mapping.get(
            "label_token_ids"
        ) != train_config.get("label_token_ids"):
            raise ValueError(
                "pretrained label mapping differs from train_config.json"
            )
        revision = train_config.get("pretrained_model_revision")
        resolved_revision = train_config.get(
            "resolved_pretrained_model_revision"
        )
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in revision
            )
            or resolved_revision != revision
        ):
            raise ValueError(
                "pretrained checkpoint lacks a resolved immutable revision"
            )
        contract = train_config.get("preprocessing_contract")
        if (
            not isinstance(contract, dict)
            or contract.get("padding_side") != "left"
            or contract.get("sequence_length") != 512
            or contract.get("text_truncation_length") != 510
            or contract.get("classification_interface")
            != "pseudo_gen_next_label_token_v1"
            or contract.get("ignore_pad_boundary_loss") is not True
            or tokenizer.padding_side != contract.get("padding_side")
            or tokenizer.pad_token_id != contract.get("pad_token_id")
            or tokenizer.pad_token != contract.get("pad_token")
        ):
            raise ValueError(
                "pretrained checkpoint preprocessing contract is invalid"
            )
        best_metadata_path = train_dir / "best_checkpoint.json"
        if not best_metadata_path.exists():
            raise ValueError(
                "pretrained checkpoint requires best_checkpoint.json"
            )
        best_metadata = read_json(best_metadata_path)
        expected_checkpoint = Path(
            str(best_metadata.get("best_model_path", ""))
        ).resolve()
        actual_checkpoint = Path(checkpoint_path).resolve()
        if expected_checkpoint != actual_checkpoint:
            raise ValueError(
                "pretrained checkpoint_path does not match the bound best "
                f"checkpoint: {actual_checkpoint} != {expected_checkpoint}"
            )
        expected_checkpoint_hash = best_metadata.get(
            "best_checkpoint_sha256"
        )
        if (
            not expected_checkpoint_hash
            or sha256_file(actual_checkpoint) != expected_checkpoint_hash
        ):
            raise ValueError("pretrained checkpoint artifact hash mismatch")
        success_path = train_dir / "success.json"
        if not success_path.exists():
            raise ValueError(
                "pretrained checkpoint requires success.json provenance"
            )
        success = read_json(success_path)
        if (
            success.get("best_model_path")
            != best_metadata.get("best_model_path")
            or success.get("best_checkpoint_sha256")
            != expected_checkpoint_hash
        ):
            raise ValueError(
                "pretrained success metadata differs from the bound "
                "checkpoint"
            )
    if model_type == "BERT":
        model = BertModule.load_from_checkpoint(checkpoint_path, map_location=map_location)
    elif model_type == "pseudo-gen-MLM":
        model = MLMModule.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            label_token_ids=label_token_ids,
            vocab_size=vocab_size,
        )
    else:
        checkpoint_overrides = {
            "label_token_ids": label_token_ids,
            "vocab_size": vocab_size,
        }
        if is_pretrained_gpt2:
            pretrained_config = train_config.get("pretrained_config")
            if not pretrained_config:
                raise ValueError(
                    "pretrained checkpoint is missing pretrained_config "
                    f"in {train_dir / 'train_config.json'}"
                )
            checkpoint_overrides.update(
                {
                    "initialize_from_pretrained": False,
                    "pretrained_config": pretrained_config,
                    "pretrained_model_name_or_path": train_config.get(
                        "pretrained_model_name_or_path"
                    ),
                    "pretrained_model_revision": train_config.get(
                        "pretrained_model_revision"
                    ),
                }
            )
        model = GPTModule.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            **checkpoint_overrides,
        )
    model.eval().to(device)
    return model, tokenizer, train_config, mapping


def _apply_dtype(model, dtype: str, device: torch.device):
    dtype = dtype.lower()
    if dtype == "fp32":
        return model.float()
    if dtype == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 inference requested on non-CUDA device")
        return model.half()
    if dtype == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 inference requested on non-CUDA device")
        return model.to(dtype=torch.bfloat16)
    raise ValueError(f"Unsupported dtype: {dtype}")


def _maybe_quantize(model, quantization: str, device: torch.device):
    quantization = (quantization or "none").lower()
    if quantization == "none":
        return model
    if quantization == "dynamic_int8_cpu":
        if device.type != "cpu":
            raise ValueError("dynamic_int8_cpu quantization requires --device cpu")
        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    raise ValueError(f"Unsupported quantization: {quantization}")


def _collate(dataset: TextDataset, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    items = [dataset[i] for i in range(len(rows))]
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0].keys()}


def _maybe_position_ids(batch, policy):
    """position_id_policy axis (padding/position diagnostic):
      default      -> None (model uses arange over the padded length; the
                      training-consistent behavior for this codebase)
      mask_derived -> positions count REAL tokens only: cumsum(attention_mask)-1
                      clamped at 0, so left-padded real tokens always get
                      positions 0..L-1 regardless of padded length."""
    if policy in (None, "default"):
        return None
    if policy == "mask_derived":
        am = batch["attention_mask"]
        pos = am.long().cumsum(-1) - 1
        return pos.clamp(min=0)
    if policy == "arange":
        # EXPLICIT naive absolute positions. Needed for consistency-trained
        # checkpoints (module-internal default = mask_derived): passing None
        # there would NOT give arange behavior, so the diagnostic naive arm
        # must force it.
        am = batch["attention_mask"]
        return torch.arange(am.size(1), device=am.device).unsqueeze(0).expand(am.size(0), -1)
    raise ValueError(f"unknown position_id_policy: {policy}")


def _top2(probs: torch.Tensor):
    values, indices = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
    if probs.shape[-1] == 1:
        return indices[:, 0], values[:, 0], indices[:, 0], torch.zeros_like(values[:, 0])
    return indices[:, 0], values[:, 0], indices[:, 1], values[:, 1]


def _natural_length(item: Dict[str, "torch.Tensor"]) -> int:
    """Real (unpadded) token count of a TextDataset item = attention_mask sum."""
    return int(item["attention_mask"].sum().item())


def _load_reference_margins(spec: Dict[str, Any]) -> Dict[str, float]:
    """Load frozen reference margins for this (train_run_id, dataset, split) from a
    Frozen reference_predictions.parquet. Required only for low/high_margin
    composition. Path via spec['reference_predictions_path']."""
    path = spec.get("reference_predictions_path")
    if not path:
        raise ValueError("composition_policy low/high_margin needs spec['reference_predictions_path']")
    import pandas as pd
    df = pd.read_parquet(path)
    m = df[(df["train_run_id"] == spec["train_run_id"]) & (df["dataset"] == spec["dataset"])
           & (df["split"] == spec.get("split", "test"))]
    return {str(r.example_id): float(r.reference_margin) for r in m.itertuples()}


def _stratified_targets(pool: List[Dict[str, Any]], n: int, seed: int,
                        margins: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Deterministic stratified sample of n targets by label (and reference-margin
    bucket when margins are available). Proportional per-stratum allocation with a
    seeded shuffle inside each stratum. Use for final-claim sweeps instead of the
    dev-only pool[:n] prefix."""
    import random as _random
    if n >= len(pool):
        return list(pool)

    def mbin(m):
        if m is None:
            return "na"
        for lo, hi, name in [(0, 0.01, "m0"), (0.01, 0.05, "m1"), (0.05, 0.10, "m2"),
                             (0.10, 0.25, "m3"), (0.25, 1.01, "m4")]:
            if lo <= m < hi:
                return name
        return "m4"

    buckets: Dict[Any, List[Dict[str, Any]]] = {}
    for it in pool:
        key = (it["label"], mbin((margins or {}).get(it["example_id"]))) if margins else (it["label"],)
        buckets.setdefault(key, []).append(it)

    total = len(pool)
    out: List[Dict[str, Any]] = []
    chosen = set()
    for k in sorted(buckets, key=str):
        b = sorted(buckets[k], key=lambda x: x["example_id"])
        _random.Random(f"{seed}:{k}").shuffle(b)
        take = int(round(n * len(buckets[k]) / total))
        for it in b[:take]:
            out.append(it); chosen.add(it["example_id"])
    if len(out) < n:  # deterministic fill to exactly n
        for it in sorted(pool, key=lambda x: x["example_id"]):
            if it["example_id"] not in chosen:
                out.append(it); chosen.add(it["example_id"])
                if len(out) >= n:
                    break
    return out[:n]


def _run_target_anchored(spec, model, tokenizer, train_config, mapping,
                         device, device_str, dtype, quantization, out_dir, predictions_path) -> Path:
    """Target-anchored inference: for each target, build ONE controlled batch
    (composition x position x padding), run it, and record the target's row."""
    job_id = spec["job_id"]
    model_type = train_config["model_type"]
    labels = [int(x) for x in mapping["labels"]]
    label_to_class_index = {label: i for i, label in enumerate(labels)}
    split = spec.get("split", "test")

    rows = load_examples(spec["dataset"], split=split, limit=spec.get("limit"))
    dataset_split_sha256 = examples_sha256(rows)
    id_to_row = {str(r["example_id"]): r for r in rows}

    static_max = int(spec.get("static_max_length", 512))  # BERT/GPT2 max positions
    meas = TextDataset([r["text"] for r in rows], [r["label"] for r in rows], tokenizer,
                       max_length=static_max, model_type=model_type)
    id_to_length = {str(rows[i]["example_id"]): _natural_length(meas[i]) for i in range(len(rows))}

    comp = spec.get("composition_policy", "random")
    pool = [{"example_id": str(r["example_id"]), "label": int(r["label"]),
             "length": id_to_length[str(r["example_id"])]} for r in rows]
    if comp in ("low_margin", "high_margin"):
        margins = _load_reference_margins(spec)  # raises if reference_predictions_path missing
        if not margins:
            raise ValueError(f"{comp} composition: reference_predictions_path yielded NO margins for "
                             f"train_run_id={spec.get('train_run_id')} dataset={spec.get('dataset')} "
                             f"split={split} : cannot form margin-based batches.")
        for it in pool:
            it["reference_margin"] = margins.get(it["example_id"])
        pool = [it for it in pool if it.get("reference_margin") is not None]
        if len(pool) < int(spec.get("batch_size", 8)):
            raise ValueError(f"{comp} composition: only {len(pool)} examples have reference margins "
                             f"(need >= batch_size); check reference_predictions_path coverage.")
    pool_by_id = {it["example_id"]: it for it in pool}

    target_ids = spec.get("target_ids")
    max_targets = spec.get("max_targets")
    sampling = spec.get("target_sampling", "head")
    if target_ids:
        targets = [pool_by_id[t] for t in map(str, target_ids) if t in pool_by_id]
    elif not max_targets or sampling == "all":
        targets = pool                                    # every example (final reduced grid)
    elif sampling == "stratified":
        strat_margins = None
        if spec.get("reference_predictions_path"):
            try:
                strat_margins = _load_reference_margins(spec)
            except Exception:
                strat_margins = None
        targets = _stratified_targets(pool, int(max_targets), int(spec.get("seed", 0)), strat_margins)
    else:  # "head" : deterministic prefix; DEV ONLY, not for final claims
        targets = pool[: int(max_targets)]

    batch_size = int(spec.get("batch_size", 8))
    position = spec.get("batch_position", "first")
    padding_policy = spec.get("padding_policy", "dynamic_longest")
    pos_id_policy = spec.get("position_id_policy", "default")
    seed = int(spec.get("seed", 0))
    measure_efficiency = bool(spec.get("measure_efficiency", False))
    warmup_targets = (
        int(spec.get("efficiency_warmup_targets", 0))
        if measure_efficiency
        else 0
    )
    if warmup_targets < 0:
        raise ValueError("efficiency_warmup_targets must be nonnegative")
    memory_measurement_scope = spec.get(
        "memory_measurement_scope",
        "model_baseline_plus_measured_batches_after_warmup_release",
    )
    if measure_efficiency and memory_measurement_scope != (
        "model_baseline_plus_measured_batches_after_warmup_release"
    ):
        raise ValueError(
            f"unknown memory_measurement_scope: {memory_measurement_scope}"
        )
    forward_events = []
    padded_tokens = 0

    def prepare_target(target):
        ctx = build_context(
            target,
            pool,
            batch_size,
            comp,
            position,
            padding_policy,
            id_to_length,
            seed=seed,
            max_length=static_max,
        )
        member_rows = [id_to_row[m] for m in ctx["member_ids"]]
        local_ds = TextDataset(
            [r["text"] for r in member_rows],
            [r["label"] for r in member_rows],
            tokenizer,
            max_length=ctx["padded_sequence_length"],
            model_type=model_type,
        )
        batch = _collate(local_ds, member_rows)
        return ctx, {k: v.to(device) for k, v in batch.items()}

    warmup_batch = None
    warmup_position_ids = None
    with torch.no_grad():
        for target in targets[:warmup_targets]:
            _, warmup_batch = prepare_target(target)
            warmup_position_ids = _maybe_position_ids(
                warmup_batch, pos_id_policy
            )
            if warmup_position_ids is not None:
                warmup_batch["position_ids"] = warmup_position_ids
            model(**warmup_batch)
    if measure_efficiency and device.type == "cuda":
        torch.cuda.synchronize(device)
        del warmup_batch
        del warmup_position_ids
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        forward_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in targets
        ]

    all_preds: List[Dict[str, Any]] = []
    start = time.perf_counter()
    with torch.no_grad():
        for target_number, target in enumerate(targets):
            ctx, batch = prepare_target(target)
            padded_tokens += (
                int(ctx["padded_sequence_length"])
                * int(ctx["realized_batch_size"])
            )
            if measure_efficiency and device.type == "cuda":
                start_event, end_event = forward_events[target_number]
                start_event.record()
                _pos = _maybe_position_ids(batch, pos_id_policy)
                if _pos is not None:
                    batch["position_ids"] = _pos
                outputs = model(**batch)
                end_event.record()
            else:
                _pos = _maybe_position_ids(batch, pos_id_policy)
                if _pos is not None:
                    batch["position_ids"] = _pos
                outputs = model(**batch)
            raw_logits = outputs.raw_logits.detach().float().cpu()
            probs = torch.softmax(raw_logits, dim=-1)
            top1_idx, top1_prob, top2_idx, top2_prob = _top2(probs)
            j = ctx["target_index"]
            trow = id_to_row[target["example_id"]]
            pred_class_index = int(top1_idx[j].item())
            true_label = int(trow["label"])
            all_preds.append({
                "example_id": target["example_id"],
                "original_index": int(trow.get("original_index", -1)),
                "true_label": true_label,
                "true_class_index": label_to_class_index.get(true_label, -1),
                "pred_label": int(labels[pred_class_index]),
                "pred_class_index": pred_class_index,
                "top1_prob": float(top1_prob[j].item()),
                "top2_class_index": int(top2_idx[j].item()),
                "top2_prob": float(top2_prob[j].item()),
                "margin": float((top1_prob[j] - top2_prob[j]).item()),
                "raw_logits": to_jsonable(raw_logits[j]),
                "probs": to_jsonable(probs[j]),
                "num_labels": len(labels),
                "labels": labels,
                "job_id": job_id,
                "train_run_id": spec.get("train_run_id"),
                "checkpoint_path": spec["checkpoint_path"],
                "dataset": spec["dataset"],
                "dataset_split_sha256": dataset_split_sha256,
                "split": split,
                "batch_size": ctx["realized_batch_size"],
                "order_policy_name": "target_anchored",
                "order_policy_type": "target_anchored",
                "dtype": dtype,
                "quantization": quantization,
                "device": device_str,
                "seed_profile": spec.get("seed_profile", "production_like"),
                "position_id_policy": spec.get("position_id_policy", "default"),
                "train_seed": spec.get("train_seed"),
                "seed": seed,
                "shard_id": int(spec.get("shard_id", 0)),
                "num_shards": int(spec.get("num_shards", 1)),
                # target-anchored axes
                "batch_position": position,
                "composition_policy": comp,
                "padding_policy": padding_policy,
                "padded_sequence_length": ctx["padded_sequence_length"],
                "target_index": j,
                "realized_batch_size": ctx["realized_batch_size"],
                "peer_example_ids": ctx["peer_example_ids"],
            })

    if measure_efficiency and device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    write_jsonl_atomic(all_preds, predictions_path)
    status = {
        "status": "SUCCEEDED",
        "time": time.time(),
        "rows": len(all_preds),
        "elapsed_sec": elapsed,
        "mode": "target_anchored",
        "dataset_split_sha256": dataset_split_sha256,
    }
    if measure_efficiency:
        status.update(
            {
                "padded_tokens": padded_tokens,
                "efficiency_warmup_targets": min(
                    warmup_targets, len(targets)
                ),
                "forward_latency_scope": (
                    "position_ids_plus_model_forward"
                    if device.type == "cuda"
                    else "unavailable"
                ),
                "memory_measurement_scope": memory_measurement_scope,
                "warmup_tensors_released_before_memory_measurement": (
                    device.type == "cuda"
                ),
                "target_examples_per_sec": (
                    len(all_preds) / elapsed if elapsed > 0 else None
                ),
                "padded_tokens_per_sec": (
                    padded_tokens / elapsed if elapsed > 0 else None
                ),
            }
        )
        if device.type == "cuda":
            latencies = [
                start_event.elapsed_time(end_event)
                for start_event, end_event in forward_events
            ]
            status.update(
                {
                    "forward_latency_samples": len(latencies),
                    "forward_latency_ms_p50": _percentile(latencies, 0.50),
                    "forward_latency_ms_p90": _percentile(latencies, 0.90),
                    "forward_latency_ms_p99": _percentile(latencies, 0.99),
                    "peak_cuda_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_cuda_memory_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                }
            )
    write_json_atomic(status, out_dir / "status.json")
    success = {
        key: status[key]
        for key in (
            "rows",
            "elapsed_sec",
            "mode",
            "dataset_split_sha256",
            "padded_tokens",
            "target_examples_per_sec",
            "padded_tokens_per_sec",
            "efficiency_warmup_targets",
            "forward_latency_scope",
            "memory_measurement_scope",
            "warmup_tensors_released_before_memory_measurement",
            "forward_latency_samples",
            "forward_latency_ms_p50",
            "forward_latency_ms_p90",
            "forward_latency_ms_p99",
            "peak_cuda_memory_allocated_bytes",
            "peak_cuda_memory_reserved_bytes",
        )
        if key in status
    }
    mark_success(out_dir, success)
    print(f"[OK] {job_id} (target_anchored): wrote {len(all_preds)} target rows to {predictions_path}")
    return out_dir


def run_inference_job(spec: Dict[str, Any]) -> Path:
    output_root = Path(spec["output_root"])
    job_id = spec["job_id"]
    out_dir = output_root / "inference" / job_id
    predictions_path = out_dir / "predictions.jsonl"
    if artifact_valid(out_dir, expected_files=["predictions.jsonl", "job_spec.json"]):
        stored_spec = read_json(out_dir / "job_spec.json")
        if canonical_json(stored_spec) != canonical_json(spec):
            raise ValueError(
                f"completed artifact spec differs for {job_id}; refusing "
                "to reuse stale predictions"
            )
        print(f"[SKIP] {job_id} already complete: {out_dir}")
        return out_dir

    ensure_dir(out_dir)
    write_json_atomic(spec, out_dir / "job_spec.json")
    write_json_atomic({"status": "RUNNING", "time": time.time()}, out_dir / "status.json")
    write_json_atomic(environment_fingerprint(), out_dir / "environment.json")

    seed = int(spec.get("seed", 0))
    apply_seed_profile(seed, spec.get("seed_profile", "production_like"))

    device_str = spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    dtype = spec.get("dtype", "fp32")
    quantization = spec.get("quantization", "none")

    train_dir = Path(spec["train_dir"])
    checkpoint_path = spec["checkpoint_path"]
    model, tokenizer, train_config, mapping = _load_model(
        checkpoint_path,
        train_dir,
        "cpu",
        device,
        expected_spec=spec,
    )
    model = _apply_dtype(model, dtype, device)
    model = _maybe_quantize(model, quantization, device)

    if spec.get("mode") == "target_anchored" or spec.get("composition_policy") is not None:
        return _run_target_anchored(spec, model, tokenizer, train_config, mapping,
                                    device, device_str, dtype, quantization, out_dir, predictions_path)

    rows = load_examples(spec["dataset"], split=spec.get("split", "test"), limit=spec.get("limit"))
    rows = apply_order_policy(rows, spec.get("order_policy", {"name": "identity", "type": "identity"}))
    rows = shard_rows(rows, int(spec.get("shard_id", 0)), int(spec.get("num_shards", 1)))

    labels = [int(x) for x in mapping["labels"]]
    label_to_class_index = {label: i for i, label in enumerate(labels)}
    ds = TextDataset([r["text"] for r in rows], [r["label"] for r in rows], tokenizer, model_type=train_config["model_type"])

    batch_size = int(spec.get("batch_size", 1))
    all_preds: List[Dict[str, Any]] = []
    global_pos = 0
    start = time.time()
    with torch.no_grad():
        for batch_id, batch_rows in enumerate(batch_iter(rows, batch_size)):
            local_ds = TextDataset([r["text"] for r in batch_rows], [r["label"] for r in batch_rows], tokenizer, model_type=train_config["model_type"])
            batch = _collate(local_ds, batch_rows)
            batch = {k: v.to(device) for k, v in batch.items()}
            _pos = _maybe_position_ids(batch, spec.get("position_id_policy", "default"))
            if _pos is not None:
                batch["position_ids"] = _pos
            outputs = model(**batch)
            raw_logits = outputs.raw_logits.detach().float().cpu()
            probs = torch.softmax(raw_logits, dim=-1)
            top1_idx, top1_prob, top2_idx, top2_prob = _top2(probs)
            for j, r in enumerate(batch_rows):
                pred_class_index = int(top1_idx[j].item())
                pred_label = labels[pred_class_index]
                true_label = int(r["label"])
                true_class_index = label_to_class_index.get(true_label, -1)
                all_preds.append(
                    {
                        "example_id": str(r["example_id"]),
                        "original_index": int(r["original_index"]),
                        "context_position": global_pos,
                        "batch_id": batch_id,
                        "position_in_batch": j,
                        "true_label": true_label,
                        "true_class_index": true_class_index,
                        "pred_label": int(pred_label),
                        "pred_class_index": pred_class_index,
                        "top1_prob": float(top1_prob[j].item()),
                        "top2_class_index": int(top2_idx[j].item()),
                        "top2_prob": float(top2_prob[j].item()),
                        "margin": float((top1_prob[j] - top2_prob[j]).item()),
                        "raw_logits": to_jsonable(raw_logits[j]),
                        "probs": to_jsonable(probs[j]),
                        "num_labels": len(labels),
                        "labels": labels,
                        "job_id": job_id,
                        "train_run_id": spec.get("train_run_id"),
                        "checkpoint_path": checkpoint_path,
                        "dataset": spec["dataset"],
                        "split": spec.get("split", "test"),
                        "batch_size": batch_size,
                        "order_policy_name": spec.get("order_policy", {}).get("name", "identity"),
                        "order_policy_type": spec.get("order_policy", {}).get("type", "identity"),
                        "dtype": dtype,
                        "quantization": quantization,
                        "device": device_str,
                        "seed_profile": spec.get("seed_profile", "production_like"),
                "position_id_policy": spec.get("position_id_policy", "default"),
                        "train_seed": spec.get("train_seed"),
                        "seed": seed,
                        "shard_id": int(spec.get("shard_id", 0)),
                        "num_shards": int(spec.get("num_shards", 1)),
                    }
                )
                global_pos += 1

    write_jsonl_atomic(all_preds, predictions_path)
    elapsed = time.time() - start
    write_json_atomic({"status": "SUCCEEDED", "time": time.time(), "rows": len(all_preds), "elapsed_sec": elapsed}, out_dir / "status.json")
    mark_success(out_dir, {"rows": len(all_preds), "elapsed_sec": elapsed})
    print(f"[OK] {job_id}: wrote {len(all_preds)} rows to {predictions_path}")
    return out_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job_spec", help="Path to inference job JSON spec")
    p.add_argument("--output_root", default="artifacts")
    p.add_argument("--job_id")
    p.add_argument("--train_run_id")
    p.add_argument("--train_dir")
    p.add_argument("--checkpoint_path")
    p.add_argument("--dataset")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--order_policy_json", default='{"name":"identity","type":"identity"}')
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--quantization", default="none")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed_profile", default="production_like", choices=["strict", "production_like", "none"])
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.job_spec:
        spec = read_json(args.job_spec)
    else:
        if not args.job_id:
            raise ValueError("--job_id is required when --job_spec is not provided")
        spec = {
            "job_id": args.job_id,
            "output_root": args.output_root,
            "train_run_id": args.train_run_id,
            "train_dir": args.train_dir,
            "checkpoint_path": args.checkpoint_path,
            "dataset": args.dataset,
            "split": args.split,
            "batch_size": args.batch_size,
            "order_policy": json.loads(args.order_policy_json),
            "dtype": args.dtype,
            "quantization": args.quantization,
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "seed": args.seed,
            "seed_profile": args.seed_profile,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "limit": args.limit,
        }
    run_inference_job(spec)


if __name__ == "__main__":
    main()
