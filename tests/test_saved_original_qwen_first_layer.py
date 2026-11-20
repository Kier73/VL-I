"""Archived actual first complete original Qwen GDN+SwiGLU layer evidence.

Receipt regression is NOT an independent pretrained numeric oracle. That
requires separate weight-backed reference execution; fixture scalar oracles
were run in the surrounding tested source package.
"""
from __future__ import annotations

import json
from pathlib import Path
from vli.ground import _hex64
from vli.large_model_comparison import SOURCES

ARCHIVE=(Path(__file__).resolve().parents[1]/
    "experiments/qwen_all_heads/datasets/2026-11-19-real-original-qwen-first-complete-layer.json")


def test_original_full_layer_receipt_has_complete_output_and_original_identity():
    a=json.loads(ARCHIVE.read_text('utf-8'))
    source=SOURCES["qwen-27b"]
    assert a["schema"]=="vl-i.saved-original-qwen-complete-layer.v1"
    assert a["source_artifact_id"]==14593501856
    assert a["model"]==source["model"]
    assert a["checkpoint_revision"]==source["revision"]
    assert a["model_advertised_payload_bytes"]==source["advertised_bytes"]
    assert a["original_token_id"]==63204
    assert _hex64(a["tokenizer_sha256"])
    assert _hex64(a["original_embedding_sha256"])
    layer=a["first_layer"]
    assert layer["layer"]==0
    assert layer["position"]==0
    assert layer["status"]=="REAL_PINNED_QWEN_COMPLETE_GDN_LAYER_NO_INDEPENDENT_FULL_ORACLE"
    assert layer["complete_output_width"]==5120
    assert layer["qualified_weight_bytes"]==766443520
    assert layer["qualified_range_count"]==11707
    assert _hex64(layer["complete_output_digest"])
    assert layer["numeric_full_layer_reference"]=="NOT_VERIFIED_INDEPENDENTLY_ON_REAL_WEIGHTS"
    assert layer["math_validated_against_scalar_reference_in_independent_fixtures"]


def test_first_complete_layer_retains_exact_slices_of_prior_all_head_state():
    a=json.loads(ARCHIVE.read_text('utf-8'))
    sibling=json.loads((ARCHIVE.parent/
        "2026-11-19-real-original-qwen-48-heads.json").read_text("utf-8"))
    ref=sibling["trained_positions"][0]
    layer=a["first_layer"]
    assert layer["gdn_recurrent_state_digest"]==ref["recurrent_state_digest"]
    assert layer["convolution_state_digest"]==ref["convolution_digest"]
    assert layer["cross_experiment_exact_recurrent_digest_match"] is True
    assert layer["cross_experiment_exact_convolution_digest_match"] is True
    assert layer["worker_cpu_seconds"]>0
    assert layer["worker_wall_seconds"]>=layer["worker_cpu_seconds"]
    host=a["host"]
    assert host["source_https_206_blocks"]==99
    assert host["source_remote_bytes"]==830472192
    assert host["logical_vl_range_reads"]==11719
    assert host["logical_vl_range_reads"]>host["source_https_206_blocks"]
    assert host["source_block_resident_limit_bytes"]==134217728
    assert host["vl_persistent_weight_bytes"]==0
    assert not a["full_original_decoder_executed"]
    assert not a["complete_language_logits_executed"]
