#!/usr/bin/env python3
"""pls_plan_v2.py : reference-cell + one-axis-spoke inference planner (reference design).

Implements Section 5 of the execution brief: instead of an unrestricted Cartesian
product over serving axes, every checkpoint gets ONE canonical reference cell plus
one-axis-at-a-time spokes, an optional small predeclared interaction block, and a
HARD job-count budget that fails planning before enqueue when exceeded.

Supported scope with the included inference script: the axes the current `pls_run_inference_context.py` already supports :
  * batch_size
  * dtype (fp32/bf16/fp16)
  * quantization (none / dynamic_int8_cpu -> device cpu)
  * order_policy (identity / reverse / seeded_shuffle)

Not emitted because they need additional runtimes or assets:
  * target-anchored position / composition / padding / tensor-shape spokes
  * hardware / backend spokes (need a 2nd GPU / TPU / vLLM / SGLang)
  * frozen analytical reference-margin artifact
  * modern-model track (ModernBERT / Qwen)
These are COUNTED in --explain-counts (for budgeting the full run) but not emitted.

Two count views:
  * projected  = full checkpoint grid (datasets x families x sizes x seeds) x spokes
                 -> the number the FULL study will cost; used for budgeting.
  * emitted    = reference+spokes for checkpoints that ACTUALLY exist on disk
                 (extracted from --checkpoints-from) -> runnable now.

Emitted specs use the exact schema `pls_worker.build_command` expects, plus two
harmless metadata fields (`context_kind`, `varied_axis`) for the comparison-policy
layer. Every emitted context is labelled reference_cell | single_axis_spoke |
predeclared_interaction (brief §5.7 requires no unlabelled cross-product context).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pls.common import (
    ensure_dir,
    load_yaml,
    make_job_id,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)

# --------------------------------------------------------------------------- #
# defaults (mirror the reference experimental_design block; overridable by --config)
# --------------------------------------------------------------------------- #
DEFAULT_DESIGN: Dict[str, Any] = {
    "reference_cell": {
        "backend": "pytorch",
        "seed_profile": "production_like",
        "dtype": "fp32",
        "quantization": "none",
        "device": "cuda",
        "batch_size": 32,
        "order_policy": {"name": "identity", "type": "identity"},
        "split": "test",
        "reference_repeat_id": 0,
    },
    # runnable-now spokes (one axis at a time; reference value excluded)
    "serving_spokes": {
        "batch_size": [1, 2, 4, 8, 16, 64],
        "dtype": ["bf16", "fp16"],
        "quantization": ["dynamic_int8_cpu"],   # -> device cpu
        "order_policy": ["reverse", "seeded_shuffle_0", "seeded_shuffle_1"],
    },
    # counted for budgeting but NOT emitted (need extra HW / serving engines)
    "deferred_spokes": {
        "hardware": ["cpu", "secondary_gpu"],
        "backend": ["vllm", "sglang"],
    },
    # target-anchored spokes (context builder): position / composition / padding, one axis
    # at a time. build_context uses a FIXED seed so the peer set matches across
    # position & padding variants (the controlled-comparison discipline). One
    # forward pass per target, so max_targets bounds cost.
    "target_anchored": {
        "enabled": True,
        "batch_size": 8,
        "max_targets": 2000,
        "target_sampling": "stratified",   # study default; "all"=full grid, "head"=dev only
        "reference_predictions_path": None,  # required for low/high_margin composition
        "seed": 0,
        "static_max_length": 512,
        "reference": {"composition_policy": "random", "batch_position": "first", "padding_policy": "dynamic_longest"},
        "position_spokes": ["middle", "last"],
        "composition_spokes": ["short", "long", "mixed_length"],
        "padding_spokes": ["static_max_length"],
        "composition_padding": "static_max_length",  # composition spokes use FIXED shape
    },
    "interaction_block": {
        "enabled": True,
        "max_contexts_per_checkpoint": 12,
        "interactions": [
            {"axes": ["batch_size", "dtype"], "levels": {"batch_size": [1, 64], "dtype": ["bf16"]}},
        ],
    },
    "repeats": 1,   # production_like repeats measure the same-context repeatability floor
    "budget": {
        "serving_max_logical_contexts": 10000,
        "serving_max_physical_jobs": 50000,
        "require_explicit_override": True,
    },
    # projected checkpoint grid for the FULL study (used only by --explain-counts)
    "checkpoint_grid": {
        "datasets": ["sst2", "sst5", "emotion", "ag_news", "imdb"],
        "families": ["encoder_classifier", "mlm_classifier", "decoder_label_token", "decoder_autoregressive"],
        "sizes": ["small", "med", "large"],
        "seeds": [40, 41, 42],
    },
}

ORDER_TYPE_MAP = {
    "identity": {"type": "identity"},
    "reverse": {"type": "reverse"},
    "seeded_shuffle_0": {"type": "seeded_shuffle", "seed": 0},
    "seeded_shuffle_1": {"type": "seeded_shuffle", "seed": 1},
}


def order_policy_from_name(name: str) -> Optional[Dict[str, Any]]:
    if isinstance(name, dict):
        return name
    spec = ORDER_TYPE_MAP.get(name)
    if spec is None:
        return None
    return {"name": name, **spec}


# --------------------------------------------------------------------------- #
# config merge
# --------------------------------------------------------------------------- #
def load_design(config_path: Optional[str]) -> Dict[str, Any]:
    design = copy.deepcopy(DEFAULT_DESIGN)
    if not config_path:
        return design
    cfg = load_yaml(config_path)
    ed = cfg.get("experimental_design", cfg)  # allow either a bare block or a full config
    for key in ("reference_cell", "serving_spokes", "deferred_spokes", "target_anchored", "interaction_block", "budget", "checkpoint_grid"):
        if key in ed:
            if isinstance(design.get(key), dict) and isinstance(ed[key], dict):
                design[key].update(ed[key])
            else:
                design[key] = ed[key]
    if "repeats" in ed:
        design["repeats"] = ed["repeats"]
    return design


# --------------------------------------------------------------------------- #
# checkpoint discovery (for emission)
# --------------------------------------------------------------------------- #
def unique_checkpoints_from_manifest(path: str) -> List[Dict[str, Any]]:
    """Extract unique (train_run_id, train_dir, checkpoint_path, dataset, seed) tuples."""
    seen: Dict[str, Dict[str, Any]] = {}
    for j in read_jsonl(path):
        if j.get("job_type") not in (None, "inference"):
            continue
        trid = j.get("train_run_id")
        if not trid or trid in seen:
            continue
        if not j.get("checkpoint_path"):
            continue
        seen[trid] = {
            "train_run_id": trid,
            "train_dir": j.get("train_dir"),
            "checkpoint_path": j.get("checkpoint_path"),
            "dataset": j.get("dataset"),
            "seed": j.get("seed", 0),
        }
    return list(seen.values())


# --------------------------------------------------------------------------- #
# spoke context construction (per checkpoint)
# --------------------------------------------------------------------------- #
def _cell(ref: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    c = dict(ref)
    c.update(overrides)
    return c


def build_contexts(design: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of serving contexts (reference + runnable spokes + interactions).

    Each context is a dict of serving-axis values + context_kind + varied_axis.
    Reference values are never re-scheduled as their own spoke.
    """
    ref = design["reference_cell"]
    ref_order = ref.get("order_policy", {"name": "identity", "type": "identity"})
    contexts: List[Dict[str, Any]] = []

    # 1) reference cell
    base = {
        "batch_size": ref["batch_size"],
        "dtype": ref["dtype"],
        "quantization": ref["quantization"],
        "device": ref["device"],
        "order_policy": ref_order,
        "split": ref.get("split", "test"),
        "seed_profile": ref.get("seed_profile", "production_like"),
        "context_kind": "reference_cell",
        "varied_axis": "none",
    }
    contexts.append(base)

    sp = design["serving_spokes"]

    # 2) batch-size spokes
    for b in sp.get("batch_size", []):
        if b == ref["batch_size"]:
            continue
        contexts.append(_cell(base, batch_size=b, context_kind="single_axis_spoke", varied_axis="batch_size"))

    # 3) dtype spokes
    for dt in sp.get("dtype", []):
        if dt == ref["dtype"]:
            continue
        contexts.append(_cell(base, dtype=dt, context_kind="single_axis_spoke", varied_axis="dtype"))

    # 4) quantization spokes (dynamic_int8_cpu forces device cpu)
    for q in sp.get("quantization", []):
        if q == ref["quantization"]:
            continue
        dev = "cpu" if q == "dynamic_int8_cpu" else ref["device"]
        contexts.append(_cell(base, quantization=q, device=dev, context_kind="single_axis_spoke", varied_axis="quantization"))

    # 5) order spokes
    for o in sp.get("order_policy", []):
        op = order_policy_from_name(o)
        if op is None:
            continue  # e.g. explicit_order_file : not supported in this release
        if op.get("name") == ref_order.get("name"):
            continue
        contexts.append(_cell(base, order_policy=op, context_kind="single_axis_spoke", varied_axis="order_policy"))

    # 6) predeclared interaction block (bounded)
    ib = design.get("interaction_block", {})
    if ib.get("enabled"):
        budget = int(ib.get("max_contexts_per_checkpoint", 12))
        added = 0
        for inter in ib.get("interactions", []):
            axes = inter["axes"]
            levels = inter["levels"]
            import itertools as _it
            for combo in _it.product(*(levels[a] for a in axes)):
                if added >= budget:
                    break
                ov = dict(zip(axes, combo))
                # skip if it equals the reference on all interaction axes
                if all(ov.get(a) == ref.get(a) for a in axes):
                    continue
                if "order_policy" in ov:
                    ov["order_policy"] = order_policy_from_name(ov["order_policy"])
                contexts.append(_cell(base, **ov, context_kind="predeclared_interaction",
                                      varied_axis="+".join(axes)))
                added += 1
    return contexts


def build_target_anchored_contexts(design: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Target-anchored contexts (context builder) with FAMILY-SPECIFIC anchors so each axis
    is isolated:
      * position family:    comp=random, pad=dynamic_longest, vary batch_position
                            (same peers+shape -> pure position)
      * padding family:     comp=random, pos=first, vary padding_policy
                            (same members -> pure shape)
      * composition family: pos=first, pad=STATIC_MAX_LENGTH, vary composition
                            (fixed shape regardless of peers -> PURE composition;
                             avoids the composition+shape confound that
                             dynamic_longest introduces)
    A fixed target-anchored seed keeps peer selection reproducible so position &
    padding variants share their peer set. Contexts are de-duplicated on
    (composition, position, padding)."""
    ta = design.get("target_anchored", {})
    if not ta.get("enabled"):
        return []
    rt = design["reference_cell"]
    r = ta.get("reference", {})
    a_comp = r.get("composition_policy", "random")
    a_pos = r.get("batch_position", "first")
    dyn = r.get("padding_policy", "dynamic_longest")
    comp_pad = ta.get("composition_padding", "static_max_length")
    common = {
        "mode": "target_anchored",
        "batch_size": ta.get("batch_size", 8),
        "max_targets": ta.get("max_targets"),
        "target_sampling": ta.get("target_sampling", "head"),
        "reference_predictions_path": ta.get("reference_predictions_path"),
        "static_max_length": ta.get("static_max_length", 512),
        "seed": ta.get("seed", 0),
        "dtype": rt["dtype"], "quantization": rt["quantization"], "device": rt["device"],
        "seed_profile": rt.get("seed_profile", "production_like"),
        "split": rt.get("split", "test"),
    }
    by_key = {}
    order = []

    def add(comp, pos, pad, kind, axis, role):
        k = (comp, pos, pad)
        if k in by_key:
            if role not in by_key[k]["roles"]:
                by_key[k]["roles"].append(role)
            return
        by_key[k] = {**common, "composition_policy": comp, "batch_position": pos,
                     "padding_policy": pad, "context_kind": kind, "varied_axis": axis,
                     "roles": [role]}
        order.append(k)

    # shared dynamic anchor (serves the position + padding families)
    add(a_comp, a_pos, dyn, "target_anchored_reference", "none", "position_padding_reference")
    for p in ta.get("position_spokes", []):
        if p != a_pos:
            add(a_comp, p, dyn, "target_anchored_spoke", "batch_position", "position_spoke")
    for pad in ta.get("padding_spokes", []):
        if pad != dyn:
            add(a_comp, a_pos, pad, "target_anchored_spoke", "padding_policy", "padding_spoke")
    # composition family under STATIC shape (isolates composition from tensor shape).
    # NOTE: (random/first/static) is de-duped with the padding spoke; it then carries
    # BOTH roles [padding_spoke, composition_reference] (context_kind stays the first).
    add(a_comp, a_pos, comp_pad, "target_anchored_composition_reference", "none", "composition_reference")
    for c in ta.get("composition_spokes", []):
        if c != a_comp:
            add(c, a_pos, comp_pad, "target_anchored_spoke", "composition_policy", "composition_spoke")
    return [by_key[k] for k in order]


def context_to_job(ctx: Dict[str, Any], ckpt: Dict[str, Any], output_root: str,
                   repeat_id: int, repeats: int) -> Dict[str, Any]:
    if ctx.get("mode") == "target_anchored":
        ta = {
            "job_type": "inference", "output_root": output_root,
            "train_run_id": ckpt["train_run_id"], "train_dir": ckpt["train_dir"],
            "checkpoint_path": ckpt["checkpoint_path"], "dataset": ckpt["dataset"],
            "split": ctx["split"], "mode": "target_anchored",
            "composition_policy": ctx["composition_policy"],
            "batch_position": ctx["batch_position"],
            "padding_policy": ctx["padding_policy"],
            "batch_size": int(ctx["batch_size"]),
            "max_targets": ctx.get("max_targets"),
            "static_max_length": int(ctx.get("static_max_length", 512)),
            "target_sampling": ctx.get("target_sampling", "head"),
            "reference_predictions_path": ctx.get("reference_predictions_path"),
            "roles": ctx.get("roles"),
            "dtype": ctx["dtype"], "quantization": ctx["quantization"], "device": ctx["device"],
            "seed": int(ctx.get("seed", 0)),
            "seed_profile": ctx["seed_profile"],
            "order_policy": {"name": "target_anchored", "type": "target_anchored"},
            "runtime_name": f"ta_{ctx['dtype']}_{ctx['quantization']}_{ctx['device']}",
            "shard_id": 0, "num_shards": 1, "limit": None,
            "context_kind": ctx["context_kind"], "varied_axis": ctx["varied_axis"],
            "repeat_id": repeat_id, "repeats": repeats,
        }
        ta["job_id"] = make_job_id("infer", ta)
        return ta
    spec: Dict[str, Any] = {
        "job_type": "inference",
        "output_root": output_root,
        "train_run_id": ckpt["train_run_id"],
        "train_dir": ckpt["train_dir"],
        "checkpoint_path": ckpt["checkpoint_path"],
        "dataset": ckpt["dataset"],
        "split": ctx["split"],
        "batch_size": int(ctx["batch_size"]),
        "order_policy": ctx["order_policy"],
        "dtype": ctx["dtype"],
        "quantization": ctx["quantization"],
        "device": ctx["device"],
        "seed": int(ckpt.get("seed", 0)),
        "seed_profile": ctx["seed_profile"],
        "runtime_name": f"{ctx['dtype']}_{ctx['quantization']}_{ctx['device']}",
        "shard_id": 0,
        "num_shards": 1,
        "limit": None,
        # metadata for the comparison-policy layer (worker ignores unknown keys)
        "context_kind": ctx["context_kind"],
        "varied_axis": ctx["varied_axis"],
        "repeat_id": repeat_id,
        "repeats": repeats,
    }
    spec["job_id"] = make_job_id("infer", spec)
    return spec


# --------------------------------------------------------------------------- #
# counting / budget
# --------------------------------------------------------------------------- #
def per_checkpoint_context_count(design: Dict[str, Any], include_deferred: bool) -> Dict[str, int]:
    ref = design["reference_cell"]
    sp = design["serving_spokes"]
    counts: Dict[str, int] = {"reference_cell": 1}
    counts["batch_size"] = len([b for b in sp.get("batch_size", []) if b != ref["batch_size"]])
    counts["dtype"] = len([d for d in sp.get("dtype", []) if d != ref["dtype"]])
    counts["quantization"] = len([q for q in sp.get("quantization", []) if q != ref["quantization"]])
    counts["order_policy"] = len([o for o in sp.get("order_policy", [])
                                  if order_policy_from_name(o) is not None])
    ib = design.get("interaction_block", {})
    inter = 0
    if ib.get("enabled"):
        import itertools as _it
        for it in ib.get("interactions", []):
            inter += len(list(_it.product(*(it["levels"][a] for a in it["axes"]))))
        inter = min(inter, int(ib.get("max_contexts_per_checkpoint", 12)))
    counts["predeclared_interaction"] = inter
    ta = design.get("target_anchored", {})
    if ta.get("enabled"):
        r = ta.get("reference", {})
        counts["ta_reference"] = 1
        counts["ta_position"] = len([p for p in ta.get("position_spokes", []) if p != r.get("batch_position", "first")])
        counts["ta_composition"] = len([c for c in ta.get("composition_spokes", []) if c != r.get("composition_policy", "random")])
        counts["ta_padding"] = len([p for p in ta.get("padding_spokes", []) if p != r.get("padding_policy", "dynamic_longest")])
    if include_deferred:
        dsp = design.get("deferred_spokes", {})
        for k, v in dsp.items():
            counts[f"deferred_{k}"] = len(v)
    return counts


def projected_checkpoint_count(design: Dict[str, Any]) -> int:
    g = design.get("checkpoint_grid", {})
    n = 1
    for k in ("datasets", "families", "sizes", "seeds"):
        n *= max(1, len(g.get(k, [])))
    return n


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML with experimental_design (optional; defaults mirror reference)")
    ap.add_argument("--checkpoints-from", help="Existing inference manifest to extract real checkpoints from (for emission)")
    ap.add_argument("--output-root", help="Artifact root for emitted specs (default: reuse checkpoints' root or ./outputs/serving_context)")
    ap.add_argument("--write-manifest", help="Path to write the emitted serving-context manifest (JSONL)")
    ap.add_argument("--explain-counts", action="store_true", help="Print projected + emitted job counts and exit")
    ap.add_argument("--dry-run", action="store_true", help="Do everything except write the manifest")
    ap.add_argument("--override-budget", help="Reason string to bypass the hard budget (recorded in the manifest meta)")
    args = ap.parse_args()

    design = load_design(args.config)
    contexts = build_contexts(design)
    ta_contexts = build_target_anchored_contexts(design)
    all_contexts = contexts + ta_contexts
    repeats = int(design.get("repeats", 1))
    budget = design.get("budget", {})

    # ---- projected (full study) counts ----
    proj_ckpts = projected_checkpoint_count(design)
    pc = per_checkpoint_context_count(design, include_deferred=True)
    runnable_per_ckpt = sum(v for k, v in pc.items() if not k.startswith("deferred_"))
    deferred_per_ckpt = sum(v for k, v in pc.items() if k.startswith("deferred_"))
    proj_logical = proj_ckpts * (runnable_per_ckpt + deferred_per_ckpt)
    proj_physical = proj_logical * repeats

    print("=" * 72)
    print("PROJECTED (full checkpoint grid : for budgeting the full study)")
    print("=" * 72)
    print(f"checkpoint grid          : {design['checkpoint_grid']}")
    print(f"projected checkpoints    : {proj_ckpts}")
    print(f"contexts / checkpoint    : {pc}")
    print(f"  runnable-now / ckpt    : {runnable_per_ckpt}")
    print(f"  deferred / ckpt        : {deferred_per_ckpt}")
    print(f"repeats                  : {repeats}")
    print(f"projected logical ctx    : {proj_logical}")
    print(f"projected physical jobs  : {proj_physical}")

    # ---- emitted (existing checkpoints) ----
    emitted: List[Dict[str, Any]] = []
    ckpts: List[Dict[str, Any]] = []
    if args.checkpoints_from:
        ckpts = unique_checkpoints_from_manifest(args.checkpoints_from)
        out_root = args.output_root or "./outputs/serving_context"
        for ck in ckpts:
            for ctx in all_contexts:
                for r in range(repeats):
                    emitted.append(context_to_job(ctx, ck, out_root, r, repeats))

    print()
    print("=" * 72)
    print("EMITTED (checkpoints that exist on disk : runnable now)")
    print("=" * 72)
    if args.checkpoints_from:
        by_kind: Dict[str, int] = {}
        by_axis: Dict[str, int] = {}
        for j in emitted:
            by_kind[j["context_kind"]] = by_kind.get(j["context_kind"], 0) + 1
            by_axis[j["varied_axis"]] = by_axis.get(j["varied_axis"], 0) + 1
        print(f"discovered checkpoints   : {len(ckpts)}")
        print(f"runnable contexts / ckpt : {len(all_contexts)} (serving={len(contexts)} + target_anchored={len(ta_contexts)})  (x repeats={repeats})")
        print(f"emitted physical jobs    : {len(emitted)}")
        print(f"  by context_kind        : {by_kind}")
        print(f"  by varied_axis         : {by_axis}")
    else:
        print("(no --checkpoints-from given; nothing emitted)")

    # ---- budget enforcement (on emitted physical jobs) ----
    max_phys = int(budget.get("serving_max_physical_jobs", 50000))
    max_logical = int(budget.get("serving_max_logical_contexts", 10000))
    emitted_logical = len(ckpts) * len(all_contexts)
    over = []
    if emitted_logical > max_logical:
        over.append(f"logical {emitted_logical} > {max_logical}")
    if len(emitted) > max_phys:
        over.append(f"physical {len(emitted)} > {max_phys}")
    print()
    print(f"budget: max_logical={max_logical} max_physical={max_phys}")
    if over:
        if args.override_budget:
            print(f"[BUDGET OVERRIDE] {', '.join(over)}  reason={args.override_budget!r}")
        else:
            print(f"[BUDGET EXCEEDED] {', '.join(over)} : refusing to write manifest. "
                  f"Use --override-budget '<reason>' to force.")
            return 3
    else:
        print("[BUDGET OK]")

    if args.explain_counts:
        return 0

    if args.write_manifest and not args.dry_run:
        if not emitted:
            print("Nothing to emit (need --checkpoints-from). Not writing.")
            return 1
        out = Path(args.write_manifest)
        ensure_dir(out.parent)
        write_jsonl_atomic(emitted, out)
        write_json_atomic(
            {
                "projected_checkpoints": proj_ckpts,
                "projected_physical_jobs": proj_physical,
                "emitted_checkpoints": len(ckpts),
                "emitted_physical_jobs": len(emitted),
                "repeats": repeats,
                "budget": budget,
                "override_reason": args.override_budget,
                "runnable_axes": ["batch_size", "dtype", "quantization", "order_policy"],
                "deferred_axes": list(design.get("deferred_spokes", {}).keys()) + ["frozen_reference_margin", "modern_model_track"],
            },
            out.parent / "serving_plan_summary.json",
        )
        print(f"\nWrote {len(emitted)} jobs -> {out}")
    elif args.dry_run:
        print("\n[DRY-RUN] manifest not written")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
