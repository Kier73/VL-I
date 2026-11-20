"""Source-tracked original two-token Qwen temporal PyTorch result.

This is a regression of a completed, artifact-linked original pretrained
job, not a claim that the pytest fixture itself executes 27B weights.
"""
from __future__ import annotations

import json
from pathlib import Path
import math

from vli.ground import _hex64
from vli.large_model_comparison import SOURCES

DATA=Path(__file__).resolve().parents[1]/"experiments/qwen_all_heads/datasets"
CURRENT=DATA/"2026-11-19-real-original-qwen-two-token-trained-attention-parity.json"
EIGHT=DATA/"2026-11-19-real-original-qwen-first-eight-architecture-directed.json"
FOUR=DATA/"2026-11-19-real-original-qwen-first-hybrid-block.json"


def test_original_both_trained_tokens_numerical_parity_and_scope_are_evidenced():
    x=json.loads(CURRENT.read_text("utf8"))
    source=SOURCES["qwen-27b"]
    assert x["schema"]=="vl-i.saved-original-qwen-two-token-trained-temporal-parity.v1"
    assert x["source_artifact_id"]==14601034410
    assert x["status"]=="REAL_ORIGINAL_QWEN_TWO_CAUSAL_TOKENS_TORCH_F64_ATTENTION_PARITY_VERIFIED"
    assert x["model"]==source["model"]
    assert x["revision"]==source["revision"]
    assert x["original_tokens"]==[63204,628]
    assert x["hidden_width"]==5120
    assert x["query_heads"]==24 and x["kv_heads"]==4
    assert x["original_complete_predecessor_layers"]==list(range(7))
    assert x["independently_compared_trained_attention_layer"]==7
    assert len(x["positions"])==2
    assert x["independent_full_attention_PyTorch_F64_parity_both_tokens"]
    assert not x["independent_official_HF_BF16_full_runtime"]
    assert not x["full_64_layer_decoder_executed"]
    assert not x["vocabulary_logits_computed"]
    assert not x["VL64_core_modified"]
    for pos,receipt in enumerate(x["positions"]):
        assert receipt["causal_position"]==pos
        assert receipt["token_id"]==x["original_tokens"][pos]
        assert receipt["historical_KV_positions"]==pos+1
        assert len(receipt["complete_predecessor_F64_digests"])==7
        assert all(_hex64(s) for s in receipt["complete_predecessor_F64_digests"])
        assert all(_hex64(receipt[name]) for name in (
            "source_predecessor_layer6_F64_digest",
            "native_whole_layer7_F64_digest",
            "independent_torch_whole_layer7_F64_digest"))
        assert receipt["source_predecessor_layer6_F64_digest"]==receipt[
            "complete_predecessor_F64_digests"][-1]
        assert receipt["max_absolute_F64_error"]<=1e-5
        assert receipt["max_scaled_F64_error"]<=2e-7


def test_historical_original_token_outputs_reproduce_separate_qualified_experiments():
    new=json.loads(CURRENT.read_text("utf8"))
    original_eight=json.loads(EIGHT.read_text("utf8"))
    original_four=json.loads(FOUR.read_text("utf8"))
    assert new["source_config_sha256"]==original_eight["source_config_sha256"]
    assert new["architecture_plan_digest"]==original_eight["source_architecture_plan_digest"]
    assert new["positions"][0]["complete_predecessor_F64_digests"]==original_eight[
        "original_position_0"]["output_f64_digests"][:7]
    assert new["positions"][0]["native_whole_layer7_F64_digest"]==original_eight[
        "original_position_0"]["output_f64_digests"][7]
    assert new["positions"][1]["complete_predecessor_F64_digests"][:4]==original_four[
        "positions"][1]["exact_f64_layer_outputs"]
    assert new["original_layer0_to_3_second_token_exactly_matches_QUALIFIED_007"]
    assert new["original_layer0_to_7_first_token_exactly_matches_QUALIFIED_008"]


def test_trained_attention_is_nonuniform_and_has_nonzero_prior_value_contribution():
    r=json.loads(CURRENT.read_text("utf8"))
    initial,second=r["positions"]
    assert initial["max_deviation_from_uniform_attention"]==0
    assert initial["max_attention_context_difference_from_current_value"]==0
    assert second["max_deviation_from_uniform_attention"]==0.4972924196653088
    assert second["max_attention_context_difference_from_current_value"]==4.862222537731979
    probs=second["previous_token_attention_probability_by_query_head"]
    assert len(probs)==24
    assert all(0<p<1 and math.isfinite(p) for p in probs)
    assert min(probs)==0.0027075803346912294
    assert max(probs)==0.9539322042725595
    assert any(p>0.9 for p in probs) and any(p<0.1 for p in probs)
    assert r["real_original_trained_nonuniform_temporal_query_key_selection"]
    assert r["real_original_trained_prior_value_contribution"]


def test_bounded_source_traffic_and_numerical_error_measurements_are_exact():
    r=json.loads(CURRENT.read_text("utf8"))
    a,b=r["positions"]
    assert a["max_absolute_F64_error"]==7.105427357601002e-15
    assert b["max_absolute_F64_error"]==5.273559366969494e-16
    t=r["original_source_traffic"]
    assert t["remote_https_206_blocks"]==1734
    assert t["remote_http_bytes"]==14537908768
    assert t["logical_64k_ranges"]==208800
    assert t["logical_range_bytes"]==13665107192
    assert t["worker_model_host_memory_budget_bytes"]==134217728
    assert t["model_weight_bytes_in_VL64_core"]==0
