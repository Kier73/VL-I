"""Independent original pretrained PyTorch F64 oracle archival evidence.

This validates a saved GitHub Actions source result, not a new runtime
recomputation. It deliberately refuses to confuse single-position
mathematical parity with BF16 or multi-token trained attention parity.
"""
from __future__ import annotations

import json
from pathlib import Path

from vli.ground import _hex64
from vli.large_model_comparison import SOURCES

BASE=Path(__file__).resolve().parents[1]/"experiments/qwen_all_heads/datasets"
PARITY=BASE/"2026-11-19-real-original-qwen-layer7-independent-pytorch-parity.json"
EIGHT=BASE/"2026-11-19-real-original-qwen-first-eight-architecture-directed.json"


def test_independent_original_trained_layer7_pytorch_agreement_is_measured():
    a=json.loads(PARITY.read_text("utf8"))
    src=SOURCES["qwen-27b"]
    assert a["schema"]=="vl-i.saved-original-qwen-independent-torch-math-parity.v1"
    assert a["source_artifact_id"]==14599529220
    assert a["status"]=="REAL_ORIGINAL_TRAINED_QWEN_LAYER7_TORCH_F64_PARITY_VERIFIED"
    assert a["model"]==src["model"]
    assert a["revision"]==src["revision"]
    assert a["original_token_id"]==63204
    assert a["original_trained_predecessor_layers"]==list(range(7))
    assert a["independent_target_layer"]==7
    assert a["original_hidden_width"]==5120
    assert a["independent_pytorch_reference"]["math_runtime"]=="independent_torch_float64"
    assert a["independent_pytorch_reference"]["trained_query_heads"]==24
    assert a["independent_pytorch_reference"]["trained_kv_heads"]==4
    assert a["independent_pytorch_reference"]["context_positions"]==1
    assert a["max_abs_numeric_error"]==1.7763568394002505e-15
    assert a["max_scaled_numeric_error"]==1.4432899320127035e-15
    assert a["rmse_numeric_error"]==3.1732128636098387e-16
    assert a["max_abs_numeric_error"]<=1e-5
    assert a["max_scaled_numeric_error"]<=2e-7
    assert a["independent_mathematical_real_trained_layer_parity_verified"]
    assert not a["official_HF_BF16_runtime_parity"]
    assert not a["multi_token_trained_attention_history_parity"]
    assert not a["full_64_layer_decoder"]
    assert not a["generated_text"]


def test_new_real_torch_reference_binds_exact_pretrained_eight_layer_source():
    parity=json.loads(PARITY.read_text("utf8"))
    prior=json.loads(EIGHT.read_text("utf8"))
    for name in (
        "original_source_config_sha256","architecture_plan_digest",
        "original_source_index_ground_digest","original_layer6_F64_digest",
        "qualified_original_layer7_F64_digest",
        "independent_PyTorch_F64_digest"):
        assert _hex64(parity[name]),name
    assert parity["original_source_config_sha256"]==prior["source_config_sha256"]
    assert parity["architecture_plan_digest"]==prior["source_architecture_plan_digest"]
    assert parity["original_layer6_F64_digest"]==prior["original_position_0"]["output_f64_digests"][6]
    assert parity["qualified_original_layer7_F64_digest"]==prior["original_position_0"]["output_f64_digests"][7]
    assert parity["prior_original_eight_layer7_F64_digest"]==prior["original_position_0"]["output_f64_digests"][7]
    assert parity["independent_PyTorch_F64_digest"]!=parity["qualified_original_layer7_F64_digest"]
    assert not parity["reference_and_qualified_F64_output_digest_equal"]


def test_original_torch_reference_has_real_weight_and_network_cost():
    parity=json.loads(PARITY.read_text("utf8"))
    r=parity["independent_pytorch_reference"]
    t=parity["original_model_host_traffic"]
    assert r["source_weight_bytes"]==744488960
    assert r["logical_64k_reads"]==11371
    assert t["remote_https_206_blocks"]==884
    assert t["remote_http_bytes"]==7411560720
    assert t["logical_64k_ranges"]==104444
    assert t["logical_range_bytes"]==6832968952
    assert t["resident_model_host_budget_bytes"]==134217728
    assert t["model_weights_stored_in_VL_ground_bytes"]==0
