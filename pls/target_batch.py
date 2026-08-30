#!/usr/bin/env python3
"""pls/target_batch.py : target-anchored batch assembly (pure, CPU-only, testable).

The core of context builder. Given a POOL of candidate examples and a TARGET example,
build a controlled batch in which exactly ONE property varies at a time:

  * composition  : WHICH peers are co-batched with the target
  * position     : WHERE the target sits inside the batch (first/middle/last)
  * padding      : the padded tensor length (dynamic vs static vs multiple-of-8)

Decomposed into three independent, deterministic primitives so the planner /
inference layer can build matched contexts that isolate a single axis:

  select_peers(target, pool, policy, n_peers, ...)  -> peer item list
  place_target(peer_ids, target_id, position)       -> (member_ids, target_index)
  compute_padded_length(lengths, policy, ...)        -> int

No torch, no tokenizer, no GPU. Items are plain dicts:
  {"example_id": str, "label": int, "length": int, "reference_margin": float?}
`length` is the tokenized length (computed upstream); `reference_margin` is the
FROZEN reference margin (metrics pipeline) and is only needed for low/high_margin.

Determinism: every selection is reproducible given `seed`; ties broken by
`example_id` so results are stable across runs and machines.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

COMPOSITION_POLICIES = [
    "random", "short", "long", "mixed_length",
    "same_label", "different_label", "low_margin", "high_margin",
]
POSITION_POLICIES = ["first", "middle", "last"]
PADDING_POLICIES = ["dynamic_longest", "static_max_length", "dynamic_pad_to_multiple_8", "length_bucketed"]


def _eid(x: Dict[str, Any]) -> str:
    return str(x["example_id"])


def select_peers(
    target: Dict[str, Any],
    pool: Sequence[Dict[str, Any]],
    policy: str,
    n_peers: int,
    seed: int = 0,
    length_bucket: int = 32,
) -> List[Dict[str, Any]]:
    """Return n_peers peer items for the target per the composition policy.

    The target is always excluded. If fewer than n_peers candidates satisfy a
    label-constrained policy, returns as many as are available (caller may then
    reduce the batch size and MUST record the realized batch size).
    """
    if policy not in COMPOSITION_POLICIES:
        raise ValueError(f"Unknown composition policy: {policy}")
    tgt_id = _eid(target)
    cands = [x for x in pool if _eid(x) != tgt_id]
    # stable base order for reproducibility
    cands = sorted(cands, key=_eid)
    if n_peers <= 0 or not cands:
        return []
    k = min(n_peers, len(cands))

    if policy == "random":
        rng = random.Random(f"{seed}:{tgt_id}")
        return rng.sample(cands, k)
    if policy == "short":
        return sorted(cands, key=lambda x: (x["length"], _eid(x)))[:k]
    if policy == "long":
        return sorted(cands, key=lambda x: (-x["length"], _eid(x)))[:k]
    if policy == "mixed_length":
        by_len = sorted(cands, key=lambda x: (x["length"], _eid(x)))
        # evenly spaced picks across the length distribution
        idx = [round(i * (len(by_len) - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]
        seen, out = set(), []
        for i in idx:
            j = i
            while j in seen and j < len(by_len) - 1:
                j += 1
            seen.add(j)
            out.append(by_len[j])
        return out[:k]
    if policy == "same_label":
        same = [x for x in cands if x["label"] == target["label"]]
        return same[:k]
    if policy == "different_label":
        diff = [x for x in cands if x["label"] != target["label"]]
        return diff[:k]
    if policy in ("low_margin", "high_margin"):
        with_m = [x for x in cands if x.get("reference_margin") is not None]
        if not with_m:
            raise ValueError(f"{policy} requires reference_margin on pool items")
        rev = policy == "high_margin"
        return sorted(with_m, key=lambda x: (x["reference_margin"], _eid(x)), reverse=rev)[:k]
    raise ValueError(f"Unhandled policy: {policy}")  # pragma: no cover


def place_target(peer_ids: Sequence[str], target_id: str, position: str) -> tuple[List[str], int]:
    """Insert the target among peers at first/middle/last. Returns (members, idx)."""
    if position not in POSITION_POLICIES:
        raise ValueError(f"Unknown position policy: {position}")
    peers = list(peer_ids)
    if position == "first":
        idx = 0
    elif position == "last":
        idx = len(peers)
    else:  # middle
        idx = len(peers) // 2
    members = peers[:idx] + [target_id] + peers[idx:]
    return members, idx


def compute_padded_length(
    lengths: Sequence[int],
    policy: str,
    max_length: int = 128,
    multiple: int = 8,
    length_bucket: int = 32,
) -> int:
    """Padded sequence length for the batch under the padding policy.

    Real tokenized lengths are capped at `max_length` (truncation) first.
    """
    if policy not in PADDING_POLICIES:
        raise ValueError(f"Unknown padding policy: {policy}")
    if not lengths:
        return 0
    capped = [min(int(l), max_length) for l in lengths]
    longest = max(capped)
    if policy == "static_max_length":
        return int(max_length)
    if policy == "dynamic_longest":
        return int(longest)
    if policy == "dynamic_pad_to_multiple_8":
        return int(min(math.ceil(longest / multiple) * multiple, max_length))
    if policy == "length_bucketed":
        return int(min(math.ceil(longest / length_bucket) * length_bucket, max_length))
    raise ValueError(f"Unhandled padding policy: {policy}")  # pragma: no cover


def build_context(
    target: Dict[str, Any],
    pool: Sequence[Dict[str, Any]],
    batch_size: int,
    composition_policy: str,
    position: str,
    padding_policy: str,
    id_to_length: Dict[str, int],
    seed: int = 0,
    max_length: int = 128,
) -> Dict[str, Any]:
    """Convenience: assemble a full target-anchored context descriptor.

    Returns member_ids (ordered), target_position, peer_ids, padded_length, and
    the realized batch size (may be < batch_size if peers ran short).
    """
    peers = select_peers(target, pool, composition_policy, batch_size - 1, seed=seed)
    peer_ids = [_eid(p) for p in peers]
    members, tgt_idx = place_target(peer_ids, _eid(target), position)
    lengths = [id_to_length[m] for m in members]
    padded = compute_padded_length(lengths, padding_policy, max_length=max_length)
    return {
        "target_example_id": _eid(target),
        "member_ids": members,
        "peer_example_ids": peer_ids,
        "batch_position": position,
        "target_index": tgt_idx,
        "composition_policy": composition_policy,
        "padding_policy": padding_policy,
        "padded_sequence_length": padded,
        "realized_batch_size": len(members),
    }
