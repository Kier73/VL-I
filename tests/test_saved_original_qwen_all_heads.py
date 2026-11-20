"""Regression of real Qwen3.8-27B all-head causal observations and work metrics.

The archive is historical evidence; these tests do not independently repeat
the 356 MB HTTPS traffic or attest checkpoint bytes beyond its source reports.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from vli.ground import _hex64
from vli.large_model_comparison import SOURCES


PATH=(Path(__file__).resolve().parents[1]/
      "experiments/qwen_all_heads/datasets/2026-11-19-real-original-qwen-48-heads.json")


def test_original_full_scale_qwen_and_complete_head_topology():
    a=json.loads(PATH.read_text("utf-8"))
    s=SOURCES["qwen-27b"]
    assert a["schema"]=="vl-i.saved-qwen-original-48-head-causal-evidence.v1"
    assert a["model"]==s["model"]
    assert a["revision"]==s["revision"]
    assert a["original_payload_advertised_bytes"]==s["advertised_bytes"]
    assert a["model_input_grade"]=="SAME_PINNED_CHECKPOINT_TOKENIZER"
    assert a["native_paragraph_ids_first_two"]==[63204,628]
    assert a["source_artifact_id"]==14593556159
    assert not a["all_layer_output_rows_completed"]
    assert not a["full_model_logits_generated"]
    assert len(a["trained_positions"])==2
    for n,r in enumerate(a["trained_positions"]):
        assert r["position"]==n
        assert r["token_id"]==a["native_paragraph_ids_first_two"][n]
        assert r["head_count"]==48
        assert r["key_head_count"]==r["query_head_count"]==16
        assert r["full_recurrent_output_width"]==6144
        assert r["qualified_range_count"]==2578
        assert r["trained_weight_bytes"]==168779776
        assert 0<=r["reference_error"]<=2e-7
        assert math.isfinite(r["out_proj_row0"])
        assert r["worker_cpu_seconds"]>0
        assert r["worker_wall_seconds"]>=r["worker_cpu_seconds"]
        for key in ("recurrent_state_digest","convolution_digest","all_head_output_digest"):
            assert _hex64(r[key])


def test_original_range_coalescing_and_exact_paged_restart():
    r=json.loads(PATH.read_text("utf-8"))
    traffic=r["host_traffic"]
    assert traffic["remote_https_206_blocks"]==170
    assert traffic["logical_64k_ranges"]==5180
    assert traffic["remote_http_bytes"]==356515840
    assert traffic["logical_range_bytes"]==337876832
    assert traffic["reused_logical_observations"]==5010
    assert traffic["resident_model_host_budget_bytes"]==67108864
    assert traffic["model_weights_inside_VL_ground_bytes"]==0
    assert _hex64(traffic["network_observation_head"])
    cp=r["cross_observer_checkpoint"]
    assert cp["state_pages"]==100
    assert cp["state_bytes"]==6537216
    assert _hex64(cp["pinned_digest"])
    assert cp["restored_into_new_qualified_navigator"]
    first,second=r["trained_positions"]
    assert first["recurrent_state_digest"]!=second["recurrent_state_digest"]
    assert first["convolution_digest"]!=second["convolution_digest"]
    assert first["all_head_output_digest"]!=second["all_head_output_digest"]
    assert first["out_proj_row0"]!=second["out_proj_row0"]
