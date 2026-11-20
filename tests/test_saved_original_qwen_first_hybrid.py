"""Archived real original Qwen first hybrid-block success, not a PyTorch oracle.

Receipt was produced by original public checkpoint job 44664814333.
The source artifact contains all individual trained operation witnesses.
"""
from __future__ import annotations

import json
from pathlib import Path

from vli.ground import _hex64
from vli.large_model_comparison import SOURCES

DATA=Path(__file__).resolve().parents[1]/"experiments/qwen_all_heads/datasets"
CURRENT=DATA/"2026-11-19-real-original-qwen-first-hybrid-block.json"
PREVIOUS=DATA/"2026-11-19-real-original-qwen-first-two-complete-layers.json"


def test_original_first_hybrid_four_layers_and_two_causal_tokens_are_evidenced():
    r=json.loads(CURRENT.read_text("utf8"))
    source=SOURCES["qwen-27b"]
    assert r["schema"]=="vl-i.saved-original-qwen-first-hybrid-block.v1"
    assert r["source_artifact_id"]==14597722183
    assert r["model"]==source["model"]
    assert r["revision"]==source["revision"]
    assert r["model_advertised_payload_bytes"]==source["advertised_bytes"]
    assert r["layer_types"]==["GDN","GDN","GDN","GATED_GQA"]
    assert r["hidden_width"]==5120
    assert r["complete_pretrained_layer_operations"]==8
    assert r["original_token_ids"]==[63204,628]
    assert len(r["positions"])==2
    for idx,p in enumerate(r["positions"]):
        assert p["position"]==idx
        assert p["token_id"]==r["original_token_ids"][idx]
        assert _hex64(p["exact_f64_predecessor_input"])
        assert len(p["full_original_output_f32_digests"])==4
        assert len(p["exact_f64_layer_outputs"])==4
        assert len(p["causal_state_digests"])==4
        assert all(_hex64(x) for k in (
            "full_original_output_f32_digests","exact_f64_layer_outputs",
            "causal_state_digests") for x in p[k])
        assert _hex64(p["causal_head"])
        assert p["kv_history_positions"]==idx+1
        assert p["original_trained_bytes_per_layer"]==[
            766443520,766443520,766443520,744488960]
    assert r["positions"][0]["causal_head"]!=r["positions"][1]["causal_head"]
    assert r["positions"][0]["exact_f64_layer_outputs"][-1]!=r["positions"][1]["exact_f64_layer_outputs"][-1]
    assert r["every_interlayer_f64_output_to_input_matches"] is True
    assert r["independent_real_weight_pytorch_reference_verified"] is False
    assert r["full_original_64_layer_decoder_executed"] is False
    assert r["vocabulary_logits_computed"] is False


def test_exact_original_layer_zero_and_one_reproduce_prior_qualified_006():
    current=json.loads(CURRENT.read_text("utf8"))
    previous=json.loads(PREVIOUS.read_text("utf8"))
    for idx,(a,b) in enumerate(zip(current["positions"],previous["positions"])):
        assert a["token_id"]==b["token_id"]
        for layer in (0,1):
            assert a["full_original_output_f32_digests"][layer]==b[
                f"layer_{layer}_output_f32_digest"]
            assert a["exact_f64_layer_outputs"][layer]==b[
                f"layer_{layer}_output_f64_digest"]
            assert a["causal_state_digests"][layer]==b[
                f"layer_{layer}_recurrent_digest"]
        if idx==0:
            assert a["exact_f64_predecessor_input"]==b["layer_0_input_f64_digest"]
    assert current["first_two_layers_reproduce_qualified_006_exactly"]


def test_bounded_host_traffic_and_four_independent_checkpoint_groups():
    a=json.loads(CURRENT.read_text("utf8"))
    t=a["worker_source_traffic"]
    assert t["remote_https_206_requests"]==775
    assert t["remote_http_bytes"]==6501171200
    assert t["logical_64k_reads"]==93044
    assert t["logical_64k_reads"]>t["remote_https_206_requests"]
    assert t["source_block_budget_bytes"]==134217728
    assert t["model_weights_stored_in_VL64_core_bytes"]==0
    cp=a["checkpoint"]
    assert _hex64(cp["externally_pinned_digest"])
    assert cp["per_layer_pages"]==[100,100,100,1]
    assert cp["state_f64_bytes"]==19628032
    assert len(cp["per_layer_digest"])==4
    assert all(_hex64(s) for s in cp["per_layer_digest"])
    assert len(set(cp["per_layer_digest"]))==4
    assert cp["fresh_navigator_and_executor_recovery"] is True
    assert cp["independent_OS_process_kill_recovery_on_real_original_weights"] is False
    assert cp["independent_OS_process_kill_recovery_on_fixtures"] is True
