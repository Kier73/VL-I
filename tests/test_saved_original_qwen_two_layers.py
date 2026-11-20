"""Immutable observed original-weight two-layer receipt regression.

This archive test does not rerun the full 27B checkpoint and is not an
independent real-weight numerical reference. Full provenance is in the
linked GitHub Actions artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

from vli.ground import _hex64
from vli.large_model_comparison import SOURCES

DIR=Path(__file__).resolve().parents[1]/"experiments/qwen_all_heads/datasets"
ARCHIVE=DIR/"2026-11-19-real-original-qwen-first-two-complete-layers.json"


def test_original_two_layer_trained_chain_has_full_causal_provenance():
    d=json.loads(ARCHIVE.read_text("utf8"))
    source=SOURCES["qwen-27b"]
    assert d["schema"]=="vl-i.saved-original-qwen-two-causal-layers.v1"
    assert d["model"]==source["model"] and d["revision"]==source["revision"]
    assert d["source_artifact_id"]==14594509188
    assert d["layer_coordinates"]==[0,1] and d["causal_positions"]==2
    assert d["original_token_ids"]==[63204,628]
    assert len(d["positions"])==2
    for number,p in enumerate(d["positions"]):
        assert p["position"]==number
        assert p["token_id"]==d["original_token_ids"][number]
        assert p["layer_0_output_f64_digest"]==p["layer_1_input_f64_digest"]
        assert _hex64(p["layer_1_output_f64_digest"])
        assert _hex64(p["layer_1_output_f32_digest"])
        assert _hex64(p["causal_chain_head"])
        assert p["layer_0_output_f64_digest"]!=p["layer_1_output_f64_digest"]
    assert d["positions"][0]["causal_chain_head"]!=d["positions"][1]["causal_chain_head"]
    assert d["positions"][0]["layer_1_output_f64_digest"]!=d["positions"][1]["layer_1_output_f64_digest"]
    assert not d["full_64_layer_decoder_completed"]
    assert not d["vocabulary_logits_computed"]
    assert not d["independent_original_weight_pytorch_oracle"]


def test_layer_zero_pretrained_output_and_recurrence_reproduce_prior_experiment():
    current=json.loads(ARCHIVE.read_text("utf8"))
    original=json.loads((DIR/"2026-11-19-real-original-qwen-first-complete-layer.json").read_text("utf8"))
    recurrent=json.loads((DIR/"2026-11-19-real-original-qwen-48-heads.json").read_text("utf8"))
    first=current["positions"][0]
    assert first["layer_0_output_f32_digest"]==original["first_layer"]["complete_output_digest"]
    assert first["layer_0_recurrent_digest"]==original["first_layer"]["gdn_recurrent_state_digest"]
    assert first["layer_0_recurrent_digest"]==recurrent["trained_positions"][0]["recurrent_state_digest"]
    assert current["positions"][1]["layer_0_recurrent_digest"]==recurrent["trained_positions"][1]["recurrent_state_digest"]
    assert current["original_first_layer_prior_digest_exactly_reproduced"] is True
    assert current["causal_checkpoint"]["per_layer_state_checkpoint_digests"][0]==recurrent["cross_observer_checkpoint"]["pinned_digest"]


def test_full_original_source_traffic_and_independent_layer_checkpoint():
    d=json.loads(ARCHIVE.read_text("utf8"))
    c=d["causal_checkpoint"]
    assert c["position"]==1
    assert _hex64(c["externally_pinned_manifest_digest"])
    assert all(_hex64(x) for x in c["per_layer_state_checkpoint_digests"])
    assert c["per_layer_state_checkpoint_digests"][0]!=c["per_layer_state_checkpoint_digests"][1]
    assert c["state_pages_per_layer"]==[100,100]
    assert c["state_f64_bytes"]==13074432
    assert c["fresh_navigator_and_executor_recovery"]
    assert not c["separate_os_process_recovery"]
    traffic=d["model_worker_traffic"]
    assert traffic["logical_64k_ranges"]==46866
    assert traffic["actual_https_206_requests"]==391
    assert traffic["remote_https_bytes"]==3279945728
    assert traffic["source_block_budget_bytes"]==134217728
    assert traffic["model_weights_stored_in_vl64_interpreter_bytes"]==0
    assert d["original_weight_bytes_per_layer_token"]==766443520
