"""Archival regression of original QUALIFIED-008 trained eight-layer checkpoint.

The archive binds a successful remote original-weight experiment, not an
independent numerical oracle. Comparing to historical QUALIFIED-007
prevents a later run from silently changing its first four numerical laws.
"""
from __future__ import annotations

import json
from pathlib import Path

from vli.ground import _hex64
from vli.large_model_comparison import SOURCES

BASE=Path(__file__).resolve().parents[1]/"experiments/qwen_all_heads/datasets"
EIGHT=BASE/"2026-11-19-real-original-qwen-first-eight-architecture-directed.json"
FOUR=BASE/"2026-11-19-real-original-qwen-first-hybrid-block.json"


def test_original_eight_are_exact_trained_layers_and_not_a_decoder_claim():
    x=json.loads(EIGHT.read_text("utf8"))
    s=SOURCES["qwen-27b"]
    assert x["schema"]=="vl-i.saved-original-qwen-architecture-first-eight.v1"
    assert x["source_artifact_id"]==14598659140
    assert x["model"]==s["model"] and x["revision"]==s["revision"]
    assert x["full_architecture_layer_count"]==64
    assert x["executed_original_layers"]==8
    assert x["executed_causal_tokens"]==[63204]
    assert x["hidden_width"]==5120
    assert x["layer_types"]==["GDN","GDN","GDN","GATED_GQA"]*2
    assert _hex64(x["source_config_sha256"])
    assert _hex64(x["source_architecture_plan_digest"])
    assert not x["config_external_sha256_pin_supplied"]
    assert x["all_predecessor_f64_handoffs_match"]
    assert x["first_four_exact_outputs_reproduce_qualified_007"]
    assert not x["independent_original_weight_pytorch_parity"]
    assert not x["full_64_decoder_layers_executed"]
    assert not x["vocabulary_logits_computed"]
    p=x["original_position_0"]
    assert p["token_id"]==63204
    for name in ("output_f32_digests","output_f64_digests",
                 "causal_state_digests"):
        assert len(p[name])==8
        assert all(_hex64(z) for z in p[name])
    assert _hex64(p["last_semantic_head"])
    assert _hex64(p["input_f64_digest"])
    assert len(set(p["output_f64_digests"]))==8
    assert len(set(p["causal_state_digests"]))==8


def test_original_layer_zero_through_three_have_exact_prior_independent_digest_parity():
    current=json.loads(EIGHT.read_text("utf8"))
    prior=json.loads(FOUR.read_text("utf8"))
    p=current["original_position_0"]
    old=prior["positions"][0]
    assert p["input_f64_digest"]==old["exact_f64_predecessor_input"]
    assert p["output_f32_digests"][:4]==old["full_original_output_f32_digests"]
    assert p["output_f64_digests"][:4]==old["exact_f64_layer_outputs"]
    assert p["causal_state_digests"][:4]==old["causal_state_digests"]


def test_original_eight_keep_source_and_recovery_costs_explicit():
    r=json.loads(EIGHT.read_text("utf8"))
    t=r["source_traffic"]
    assert t["remote_https_206_requests"]==825
    assert t["remote_http_bytes"]==6918617224
    assert t["logical_64k_reads"]==93151
    assert t["logical_range_bytes"]==6089267544
    assert t["source_block_budget_bytes"]==134217728
    assert t["model_weights_stored_in_VL64_core_bytes"]==0
    c=r["checkpoint"]
    assert _hex64(c["externally_pinned_manifest_digest"])
    assert c["per_layer_state_pages"]==[100,100,100,1,100,100,100,1]
    assert c["state_f64_bytes"]==39256064
    assert c["fresh_navigator_restore_same_process"]
    assert c["separate_process_kill_recovery_only_on_fixtures"]
