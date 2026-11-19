"""VL-I experiment: reuse VL-64's real pretrained native numerical realizer.

The runner does NOT invoke Transformers model.forward, a model-serving endpoint,
or any alternative hidden inference route. It explicitly reuses the pinned
VL-64 typed-DAG / numeric operator implementation and its separate NumPy
reference. VL-I adds intake, intention-frontier, result binding and an
independent *structural* receipt check; it does not yet add a new shortcut.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .ground import (
    _canonical, _digest, _hex64, _require, lift_observation,
    intention_frontier, verify_frontier, verify_ground,
)

RUN_SCHEMA = "vl-i.model-realization.v1"


def bind_realization(ground, frontier, upstream, tokens):
    """Bind a successful full-model VL-64 native numerical result.

    The numerical comparison is performed by VL-64's actual execution and
    separately implemented oracle. This function does not recompute logits.
    """
    verify_ground(ground)
    _require(isinstance(upstream, dict), "Missing upstream numerical result")
    _require(isinstance(tokens, (tuple, list)) and 1 <= len(tokens) <= 32
             and all(type(x) is int and 0 <= x < ground["interpretation"]["architecture"]["vocab"]
                     for x in tokens), "Invalid model input token IDs")
    source, interpretation = ground["source"], ground["interpretation"]
    _require(frontier["dag_digest"] == interpretation["dag_digest"]
             and frontier["targets"] == ["vocab_logits"],
             "A partial Ground cannot produce full model logits")
    _require(upstream.get("schema") == "vl64.real-pretrained-30-layer-run.v1"
             and upstream.get("status") == "REAL_PRETRAINED_30_LAYER_LOGITS_INDEPENDENTLY_VERIFIED"
             and upstream.get("model") == source["model"]
             and upstream.get("revision") == source["revision"]
             and upstream.get("dag_digest") == interpretation["dag_digest"]
             and upstream.get("tokens") == list(tokens), "Upstream numerical/source contract mismatch")
    arch = interpretation["architecture"]
    _require(upstream.get("layers_completed") == arch["layers"]
             and upstream.get("total_declared_layers") == arch["layers"]
             and upstream.get("full_vocab_size") == arch["vocab"],
             "Incomplete pretrained decoder")
    _require(type(upstream.get("selected_next_token")) is int
             and 0 <= upstream["selected_next_token"] < arch["vocab"],
             "Invalid selected token")
    _require(_hex64(upstream.get("checkpoint_digest"))
             and _hex64(upstream.get("logits_f64_sha256"))
             and _hex64(upstream.get("result_receipt_digest")),
             "Missing pinned numerical or recovery witness")
    _require(type(upstream.get("independent_logit_max_abs_error")) in (int, float)
             and 0 <= upstream["independent_logit_max_abs_error"] <= 2e-4,
             "Independent logit comparison failed")
    for field in ("range_reads_including_logits", "weight_bytes_including_logits"):
        _require(type(upstream.get(field)) is int and upstream[field] > 0,
                 "Missing actual pretrained weight ranges: " + field)
    record = {
        "schema": RUN_SCHEMA,
        "experiment": "VL-I-001",
        "tier": "B2_VL_NATIVE_REUSED_VL64_OPERATOR_IMPLEMENTATION",
        "scope": "provided_token_ids_full_pretrained_forward_no_tokenizer_or_multi_token_generation",
        "ground_digest": ground["ground_digest"],
        "frontier_digest": frontier["frontier_digest"],
        "source_model": source["model"],
        "source_revision": source["revision"],
        "typed_dag_digest": interpretation["dag_digest"],
        "input_token_ids": list(tokens),
        "source_payload_bytes_read_during_intake": 0,
        "source_payload_bytes_read_during_realization": upstream["weight_bytes_including_logits"],
        "range_reads": upstream["range_reads_including_logits"],
        "layers_completed": upstream["layers_completed"],
        "final_vocab_size": upstream["full_vocab_size"],
        "selected_next_token": upstream["selected_next_token"],
        "full_logits_f64_sha256": upstream["logits_f64_sha256"],
        "independent_logit_max_abs_error": upstream["independent_logit_max_abs_error"],
        "causal_checkpoint_digest": upstream["checkpoint_digest"],
        "upstream_numerical_receipt_digest": upstream["result_receipt_digest"],
        "result_status": "REUSED_VL64_NATIVE_NUMERICAL_EXECUTION_REPORTED_VERIFIED",
        "independent_vli_status": "STRUCTURAL_CONTRACT_CHECKED_NUMERICAL_ORACLE_NOT_REEXECUTED",
    }
    record["realization_digest"] = _digest(RUN_SCHEMA, record)
    return record


def verify_realization(ground, frontier, record):
    """Check VL-I's digest, source and typed-result consistency.

    This is NOT an independent re-execution of the full model's mathematics.
    """
    verify_ground(ground)
    _require(isinstance(frontier, dict) and
             frontier.get("frontier_digest") == record.get("frontier_digest"),
             "Missing/altered intention frontier binding")
    digest = record.get("realization_digest")
    _require(_hex64(digest) and digest == _digest(RUN_SCHEMA, {
        k: v for k, v in record.items() if k != "realization_digest"}),
        "Altered realization receipt")
    _require(record.get("schema") == RUN_SCHEMA and
             record.get("ground_digest") == ground["ground_digest"] and
             record.get("source_model") == ground["source"]["model"] and
             record.get("source_revision") == ground["source"]["revision"] and
             record.get("typed_dag_digest") == ground["interpretation"]["dag_digest"] and
             record.get("frontier_digest") == frontier["frontier_digest"] and
             record.get("layers_completed") == ground["interpretation"]["architecture"]["layers"] and
             record.get("final_vocab_size") == ground["interpretation"]["architecture"]["vocab"],
             "Realization source/architecture disagreement")
    _require(record.get("tier") == "B2_VL_NATIVE_REUSED_VL64_OPERATOR_IMPLEMENTATION"
             and record.get("result_status") ==
             "REUSED_VL64_NATIVE_NUMERICAL_EXECUTION_REPORTED_VERIFIED",
             "Unqualified execution-class claim")
    return True


def run_pinned_decoder(observed, tokens, *, state_dir, result_path,
                       native_runner=None):
    """Lift → intention frontier → real VL-64 pretrained execution → binding.

    Live run requires a qualified pinned Hugging Face source. Missing network
    or incomplete tensor ranges must raise; no fixture success is fabricated.
    """
    ground, plan = lift_observation(observed)
    frontier = intention_frontier(plan, ["vocab_logits"])
    verify_frontier(frontier, plan)
    _require(len(frontier["required_nodes"]) == ground["interpretation"]["node_count"],
             "Full-model logits must depend on complete decoder")
    if native_runner is None:
        try:
            from experiments.model_fabric.full_decoder_lab import run
        except ImportError as exc:
            raise RuntimeError("Pinned VL-64 native decoder is not installed") from exc
        native_runner = run
    upstream = native_runner(tokens=tuple(tokens),
                             state_dir=str(state_dir),
                             result_path=str(result_path))
    realization = bind_realization(ground, frontier, upstream, tokens)
    verify_realization(ground, frontier, realization)
    return {"ground": ground, "frontier": frontier,
            "realization": realization, "upstream": upstream}
