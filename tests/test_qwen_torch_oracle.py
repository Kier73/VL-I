"""QUALIFIED-009: independent PyTorch GQA reference against original-topology fixtures.

This is not a copy of the VL-I attention implementation: trained matrix
multiplication, normalization, rotary phases, grouped attention, and SwiGLU
are independently performed by PyTorch. The numerical source navigator is
shared to preserve exactly the original safetensors bytes and identity.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

torch=pytest.importorskip("torch")
from vli.qwen_gated_attention import QwenQualifiedGatedAttention
from vli.qwen_architecture_navigator import QwenArchitectureCausalPrefix
from vli.qwen_torch_oracle import OriginalQwenTorchGQA
from vli.qwen_layer_chain import _f64_transfer_digest
from test_qwen_architecture_navigator import nav_eight_layers,plan_for


def _native(nav):
    return QwenQualifiedGatedAttention(
        nav,layer=7,query_heads=2,kv_heads=1,
        head_dim=2,rotary_dim=2,rope_theta=10_000_000.0)


def test_independent_torch_matches_full_attention_and_causal_history():
    nav=nav_eight_layers()
    plan=plan_for(nav)
    oracle=OriginalQwenTorchGQA(nav,plan,layer=7)
    native=_native(nav)
    reference_inputs=([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3])
    for pos,x in enumerate(reference_inputs):
        want,independent=oracle.next(list(x))
        got,actual=native.next(list(x))
        assert got==pytest.approx(want,abs=2e-10)
        assert oracle.position==native.position==pos+1
        assert independent["causal_kv_positions"]==pos+1
        assert actual["kv_history_positions"]==pos+1
        assert independent["complete_output_width"]==len(x)
        assert independent["torch_f64_output_digest"]==_f64_transfer_digest(want)
        assert independent["math_runtime"]=="independent_torch_float64"
        assert not independent["official_huggingface_bf16_runtime_parity"]


def test_real_intermediate_fixture_preceded_by_seven_full_pretrained_form_layers():
    nav=nav_eight_layers()
    plan=plan_for(nav)
    prefix=QwenArchitectureCausalPrefix(
        nav,plan,stop_layer=7,max_positions=64)
    native=_native(nav)
    independent=OriginalQwenTorchGQA(nav,plan,layer=7)
    for tok,x in ((63204,[.5,-.2,.3]),(628,[-.1,.2,.8])):
        intermediate,receipt=prefix.step(x,token_id=tok)
        assert receipt["complete_prefix_layers"]==7
        ref_out,ref_report=independent.next(intermediate)
        actual_out,act_report=native.next(intermediate)
        assert ref_out==pytest.approx(actual_out,abs=2e-10)
        assert len(receipt["layer_witnesses"])==7
        assert act_report["complete_Qwen_transformer_layer"]
        assert act_report["layer"]==7
        assert ref_report["new_source_weight_payload_bytes"]>0


def test_independent_reference_does_not_commit_state_after_source_failure(monkeypatch):
    nav=nav_eight_layers()
    oracle=OriginalQwenTorchGQA(nav,plan_for(nav),layer=7)
    oracle.next([.5,-.2,.3])
    old=oracle.position
    old_history=[(a.clone(),b.clone()) for a,b in oracle.history]
    valid=oracle._linear
    def reject(name,x,*,output_width):
        if name=="mlp.down_proj.weight":
            raise OSError("original source byte-range host departed")
        return valid(name,x,output_width=output_width)
    monkeypatch.setattr(oracle,"_linear",reject)
    with pytest.raises(OSError,match="departed"):
        oracle.next([-.1,.2,.8])
    assert oracle.position==old
    for pair,original in zip(oracle.history,old_history):
        assert all(torch.equal(x,y) for x,y in zip(pair,original))


def test_torch_oracle_requires_matching_admitted_source_plan():
    nav=nav_eight_layers()
    plan=plan_for(nav)
    with pytest.raises(ValueError,match="does not declare"):
        OriginalQwenTorchGQA(nav,plan,layer=6)
    changed=deepcopy(plan)
    changed["geometry"]["rotary_dim"]=4
    with pytest.raises(ValueError,match="digest"):
        OriginalQwenTorchGQA(nav,changed,layer=7)
