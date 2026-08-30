#!/usr/bin/env python3
"""Dataset loading and configurable ordering for label-stability experiments."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .common import read_json, sha256_json

# Dataset keys aligned with the EMNLP Gen-vs-Disc paper and the existing code.
# Prefer SetFit-hosted mirrors where available so benchmark inputs stay consistent.
DATASET_CONFIGS: Dict[str, Dict[str, Optional[str]]] = {
    "sst2": {"hf_name": "SetFit/sst2", "hf_config": None, "text_field": "text", "source": "SetFit"},
    "sst5": {"hf_name": "SetFit/sst5", "hf_config": None, "text_field": "text", "source": "SetFit"},
    "emotion": {"hf_name": "SetFit/emotion", "hf_config": None, "text_field": "text", "source": "SetFit"},
    "imdb": {"hf_name": "SetFit/imdb", "hf_config": None, "text_field": "text", "source": "SetFit"},
    "ag_news": {"hf_name": "SetFit/ag_news", "hf_config": None, "text_field": "text", "source": "SetFit"},
    "tweet_eval_sentiment": {"hf_name": "tweet_eval", "hf_config": "sentiment", "text_field": "text", "source": "Hugging Face"},
    "twitter_financial": {"hf_name": "zeroshot/twitter-financial-news-sentiment", "hf_config": None, "text_field": "text", "source": "Hugging Face"},
    "multiclass_sentiment": {"hf_name": "Sp1786/multiclass-sentiment-analysis-dataset", "hf_config": None, "text_field": "text", "source": "Hugging Face"},
}


def dataset_revision(dataset_key: str) -> Optional[str]:
    """Return an explicit public snapshot revision when configured.

    Set TCI_DATASET_REVISIONS to a JSON file mapping dataset keys to immutable
    public commit identifiers. Set TCI_REQUIRE_DATASET_REVISIONS=1 to reject
    unpinned public dataset access. Local JSON datasets do not use this setting.
    """
    revision_file = os.environ.get("TCI_DATASET_REVISIONS")
    revision = None
    if revision_file:
        mapping = read_json(revision_file)
        revision = mapping.get(dataset_key)
    if os.environ.get("TCI_REQUIRE_DATASET_REVISIONS") == "1" and not revision:
        raise ValueError(f"missing public dataset revision for {dataset_key}")
    if revision is not None and (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError(f"dataset revision for {dataset_key} must be a 40-character commit")
    return revision



def _load_hf_dataset(dataset_key: str, split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    if dataset_key not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset key: {dataset_key}. Known: {sorted(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[dataset_key]
    name = cfg["hf_name"]
    config = cfg["hf_config"]
    text_field = cfg["text_field"]
    ds = load_dataset(name, config, revision=dataset_revision(dataset_key))
    if split not in ds:
        if split == "test" and "validation" in ds:
            split = "validation"
        elif split == "validation" and "test" in ds:
            split = "test"
        else:
            raise KeyError(f"Split {split!r} not available in {dataset_key}. Available: {list(ds.keys())}")
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds[split]):
        if text_field not in ex or "label" not in ex:
            raise KeyError(f"Expected fields {text_field!r} and 'label' in {dataset_key}/{split}; saw {list(ex.keys())}")
        rows.append({"example_id": str(ex.get("id", i)), "text": str(ex[text_field]), "label": int(ex["label"]), "original_index": i})
    return rows


def _load_json_dataset(path: str | Path, split: str) -> List[Dict[str, Any]]:
    obj = read_json(path)
    data = obj[split] if isinstance(obj, dict) and split in obj else obj
    if not isinstance(data, list):
        raise ValueError(f"JSON dataset must be list or dict containing split {split!r}")
    rows = []
    for i, ex in enumerate(data):
        text = ex.get("text") or ex.get("sentence") or ex.get("content")
        if text is None:
            raise KeyError(f"JSON row {i} lacks text/sentence/content field")
        label = ex.get("label")
        if label is None:
            raise KeyError(f"JSON row {i} lacks label field")
        rows.append({"example_id": str(ex.get("id", i)), "text": str(text), "label": int(label), "original_index": i})
    return rows


def load_examples(dataset: str, split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if Path(dataset).exists():
        rows = _load_json_dataset(dataset, split)
    else:
        rows = _load_hf_dataset(dataset, split)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def examples_sha256(rows: Iterable[Dict[str, Any]]) -> str:
    """Hash the exact ordered examples consumed by a run."""
    canonical_rows = [
        {
            "example_id": str(row["example_id"]),
            "original_index": int(row["original_index"]),
            "text": str(row["text"]),
            "label": int(row["label"]),
        }
        for row in rows
    ]
    return sha256_json(canonical_rows)


def get_label_set(dataset: str) -> List[int]:
    rows = load_examples(dataset, split="train", limit=None)
    return sorted({int(r["label"]) for r in rows})


def get_dataset_source(dataset: str) -> str:
    if Path(dataset).exists():
        return "local_json"
    if dataset not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset key: {dataset}. Known: {sorted(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[dataset]
    source = cfg.get("source") or "Hugging Face"
    hf_name = cfg["hf_name"]
    hf_config = cfg.get("hf_config")
    if hf_config:
        return f"{source} ({hf_name}/{hf_config})"
    return f"{source} ({hf_name})"


def _has_real_validation(dataset: str) -> bool:
    """True only if the dataset genuinely ships a 'validation' split.

    Guards against the loader's test->validation substitution: we must NOT treat
    the test set as a validation set (that would leak test into early stopping /
    checkpoint selection). For local JSON, a top-level 'validation' key counts.
    """
    if Path(dataset).exists():
        try:
            obj = read_json(dataset)
            return isinstance(obj, dict) and "validation" in obj
        except Exception:
            return False
    cfg = DATASET_CONFIGS.get(dataset)
    if not cfg:
        return False
    try:
        from datasets import get_dataset_split_names

        return "validation" in get_dataset_split_names(
            cfg["hf_name"], cfg.get("hf_config"), revision=dataset_revision(dataset)
        )
    except Exception:
        return False


def load_train_val_data(
    dataset: str,
    use_test_as_val: bool = False,
    val_split_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Return (train_texts, train_labels, val_texts, val_labels).

    Validation source, in priority order:
      1. use_test_as_val=True  -> the test split (debug/backward-compat ONLY; leaks
         test into model selection : do not use for reported results).
      2. a REAL 'validation' split when the dataset has one (sst2/sst5/emotion).
      3. otherwise a stratified carve-out from TRAIN (fixed seed) : never test
         (this is the correct held-out path for ag_news/imdb, which lack a val
         split). The test set stays untouched for evaluation.
    """
    train_rows = load_examples(dataset, split="train")
    if use_test_as_val:
        val_rows = load_examples(dataset, split="test")
    elif _has_real_validation(dataset):
        val_rows = load_examples(dataset, split="validation")
    else:
        from sklearn.model_selection import train_test_split

        train_rows, val_rows = train_test_split(
            train_rows,
            test_size=val_split_size,
            random_state=random_state,
            stratify=[int(row["label"]) for row in train_rows],
        )
    train_texts = [str(row["text"]) for row in train_rows]
    train_labels = [int(row["label"]) for row in train_rows]
    val_texts = [str(row["text"]) for row in val_rows]
    val_labels = [int(row["label"]) for row in val_rows]
    return train_texts, train_labels, val_texts, val_labels


def apply_order_policy(rows: List[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a reordered copy of rows according to a config policy.

    Supported policies:
      {name: original, type: identity}
      {name: reverse, type: reverse}
      {name: seeded_shuffle_0, type: seeded_shuffle, seed: 123}
      {name: explicit_perm, type: explicit_order_file, path: ...}

    explicit_order_file can contain either a JSON list of example_ids or a JSON list
    of integer original indices.
    """
    ptype = policy.get("type", policy.get("name", "identity"))
    out = list(rows)
    if ptype in {"identity", "original"}:
        return out
    if ptype == "reverse":
        return list(reversed(out))
    if ptype == "seeded_shuffle":
        rng = random.Random(int(policy.get("seed", 0)))
        rng.shuffle(out)
        return out
    if ptype == "explicit_order_file":
        order_path = policy["path"]
        order = read_json(order_path)
        if not isinstance(order, list):
            raise ValueError(f"Order file {order_path} must contain a JSON list")
        by_id = {r["example_id"]: r for r in rows}
        by_index = {int(r["original_index"]): r for r in rows}
        reordered = []
        for item in order:
            key = str(item)
            if key in by_id:
                reordered.append(by_id[key])
            elif isinstance(item, int) and item in by_index:
                reordered.append(by_index[item])
            else:
                raise KeyError(f"Order item {item!r} not found as example_id or original_index")
        if len(reordered) != len(rows):
            raise ValueError(f"Order file length {len(reordered)} != dataset length {len(rows)}")
        return reordered
    raise ValueError(f"Unsupported order policy type: {ptype}")


def shard_rows(rows: List[Dict[str, Any]], shard_id: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return rows
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id {shard_id}; num_shards={num_shards}")
    return [r for i, r in enumerate(rows) if i % num_shards == shard_id]


def batch_iter(rows: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]
