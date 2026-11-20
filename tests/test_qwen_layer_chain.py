"""Independent scalar and adversarial QUALIFIED-006 causal chain tests.

These fixtures validate actual layer-0-output -> layer-1-input composition;
they do NOT assert numerical equivalence to a separate real 27B runtime.
"""
from __future__ import annotations

from copy import deepcopy
import struct

import pytest

from experiments.model_fabric.complete_native_layers import (
    QwenQualifiedLayer, LocalStatePages,
)
from vli.qwen_all_heads import QwenVectorizedCompleteLayer
from vli.qwen_layer_chain import QwenCausalGDNChain, _f64_transfer_digest
from test_architecture_probe import nav_for
from test_complete_native_layers import qwen_fixture


def two_layer_fixture():
    # Reproduce the exact F32 source tensor geometry of the independent
    # upstream scalar Qwen fixture, at *two* numbered pretrained coordinates.
    _,matrices,vectors,conv=qwen_fixture()
    tensors={}
    for layer in (0,1):
        prefix=f"model.language_model.layers.{layer}."
        for name,m in matrices.items():
            raw=[float(v) for row in m for v in row]
            tensors[prefix+name]=("F32",[len(m),len(m[0])],
                struct.pack("<"+str(len(raw))+"f",*raw))
        for name,v in vectors.items():
            tensors[prefix+name]=("F32",[len(v)],
                struct.pack("<"+str(len(v))+"f",*v))
        raw=[float(v) for row in conv for v in row]
        tensors[prefix+"linear_attn.conv1d.weight"]=(
            "F32",[len(conv),1,len(conv[0])],
            struct.pack("<"+str(len(raw))+"f",*raw))
    return nav_for(tensors)


def executor(nav, cls):
    return [cls(nav,layer=idx,key_heads=1,value_heads=2,
                key_dim=2,value_dim=2) for idx in (0,1)]


def test_second_layer_consumes_exact_actual_output_and_matches_scalar():
    fast=QwenCausalGDNChain(executor(two_layer_fixture(),QwenVectorizedCompleteLayer))
    scalar=executor(two_layer_fixture(),QwenQualifiedLayer)
    for pos,hidden in enumerate(([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3])):
        trained,receipt=fast.step(list(hidden),token_id=pos)
        ref0,_=scalar[0].next(list(hidden))
        ref1,_=scalar[1].next(ref0)
        assert trained==pytest.approx(ref1,abs=2e-10)
        first,second=receipt["layer_witnesses"]
        assert receipt["position"]==pos
        assert receipt["all_layers_completed"] is True
        assert receipt["original_decoder_completed"] is False
        assert receipt["vocabulary_logits_computed"] is False
        assert first["output_f64_digest"]==second["input_f64_digest"]
        assert first["input_f64_digest"]==_f64_transfer_digest(hidden)
        assert second["output_f64_digest"]==_f64_transfer_digest(trained)
        assert first["output_f64_digest"]!=first["input_f64_digest"]
        assert receipt["chain_head"]==fast.head
    assert fast.position==3
    assert [r.position for r in fast.runners]==[3,3]


def test_both_causal_layers_recover_from_independently_pinned_pages(tmp_path):
    continuous=QwenCausalGDNChain(
        executor(two_layer_fixture(),QwenVectorizedCompleteLayer))
    one,_=continuous.step([.2,.4,-.1],token_id=63204)
    assert len(one)==3
    store=LocalStatePages(tmp_path)
    checkpoint=continuous.checkpoint(store,input_sequence_digest="a"*64)
    assert len(checkpoint["layer_checkpoints"])==2
    assert all(len(s["causal_pages"])>=1 for s in checkpoint["layer_checkpoints"])
    assert checkpoint["layer_checkpoints"][0]["layer"]==0
    assert checkpoint["layer_checkpoints"][1]["layer"]==1
    fresh=QwenCausalGDNChain.restore(
        executor(two_layer_fixture(),QwenVectorizedCompleteLayer),
        store,checkpoint,expected_digest=checkpoint["checkpoint_digest"],
        expected_input_sequence_digest="a"*64)
    assert fresh.position==1
    assert fresh.head==continuous.head
    assert fresh.last_output_digest==continuous.last_output_digest
    for token,hidden in ((628,[-.1,.6,.7]),(123,[.3,-.2,.1])):
        a,receipt_a=continuous.step(hidden,token_id=token)
        b,receipt_b=fresh.step(hidden,token_id=token)
        assert a==b
        assert receipt_a["chain_head"]==receipt_b["chain_head"]
        for r_a,r_b in zip(continuous.runners,fresh.runners):
            assert r_a.convolution_history==r_b.convolution_history
            assert r_a.recurrent_state==r_b.recurrent_state
    assert fresh.position==3


def test_checkpoint_denies_stale_digest_wrong_lineage_and_broken_pages(tmp_path):
    chain=QwenCausalGDNChain(
        executor(two_layer_fixture(),QwenVectorizedCompleteLayer))
    chain.step([.3,.1,-.2])
    store=LocalStatePages(tmp_path)
    saved=chain.checkpoint(store,input_sequence_digest="c"*64)
    def restore(record,digest=None,lineage="c"*64):
        return QwenCausalGDNChain.restore(
            executor(two_layer_fixture(),QwenVectorizedCompleteLayer),
            store,record,expected_digest=digest or saved["checkpoint_digest"],
            expected_input_sequence_digest=lineage)
    with pytest.raises(ValueError,match="Unpinned|altered"):
        restore({**saved,"position":2})
    with pytest.raises(ValueError,match="lineage|Wrong"):
        restore(saved,lineage="d"*64)
    path=tmp_path/saved["layer_checkpoints"][1]["causal_pages"][0]["sha256"]
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="Corrupted"):
        restore(saved)


def test_failed_second_layer_rolls_back_every_causal_state(monkeypatch):
    chain=QwenCausalGDNChain(
        executor(two_layer_fixture(),QwenVectorizedCompleteLayer))
    chain.step([.5,.2,-.1])
    state=deepcopy([(r.position,r.convolution_history,r.recurrent_state)
                    for r in chain.runners])
    old_head,old_digest=chain.head,chain.last_output_digest
    def absent(_input):
        raise ValueError("downstream worker departed")
    monkeypatch.setattr(chain.runners[1],"next",absent)
    with pytest.raises(ValueError,match="departed"):
        chain.step([-.2,.3,.4])
    assert [(r.position,r.convolution_history,r.recurrent_state)
            for r in chain.runners]==state
    assert chain.position==1 and chain.head==old_head
    assert chain.last_output_digest==old_digest


def test_invalid_layer_result_or_input_is_never_committed(monkeypatch):
    chain=QwenCausalGDNChain(
        executor(two_layer_fixture(),QwenVectorizedCompleteLayer))
    with pytest.raises(ValueError,match="Unqualified"):
        chain.step([float("nan"),.1,.2])
    original=chain.runners[1].next
    def altered(x):
        out,receipt=original(x)
        return out,{**receipt,"complete_Qwen_transformer_layer":False}
    monkeypatch.setattr(chain.runners[1],"next",altered)
    with pytest.raises(ValueError,match="contradictory"):
        chain.step([.1,.2,.3])
    assert chain.position==0 and chain.head==chain.genesis
    assert [r.position for r in chain.runners]==[0,0]


def test_missing_original_layer_refused_before_claiming_a_chain():
    nav,*_=qwen_fixture()
    first=QwenVectorizedCompleteLayer(nav,layer=0,key_heads=1,
                                      value_heads=2,key_dim=2,value_dim=2)
    with pytest.raises(ValueError,match="Incomplete pinned"):
        QwenVectorizedCompleteLayer(nav,layer=1,key_heads=1,
                                    value_heads=2,key_dim=2,value_dim=2)
    with pytest.raises(ValueError,match="At least two"):
        QwenCausalGDNChain([first])


def test_requested_original_interlayer_observations_have_exact_native_handoff_receipts():
    chain=QwenCausalGDNChain(
        executor(two_layer_fixture(),QwenVectorizedCompleteLayer))
    out,received=chain.step([.5,-.2,.3],token_id=73,
                            capture_intermediates=True)
    captured=received["receiver_local_intermediates"]
    assert len(captured)==2
    assert all(type(v) is tuple and len(v)==3 for v in captured)
    assert received["layer_witnesses"][0]["output_f64_digest"]==(
        _f64_transfer_digest(captured[0]))
    assert received["layer_witnesses"][1]["input_f64_digest"]==(
        _f64_transfer_digest(captured[0]))
    assert received["layer_witnesses"][1]["output_f64_digest"]==(
        _f64_transfer_digest(captured[1]))
    assert captured[1]==tuple(out)
    assert received["all_layers_completed"]
    assert not received["original_decoder_completed"]
    with pytest.raises(ValueError):
        chain.step([.3,.2,.1],token_id=74,capture_intermediates="yes")
    assert chain.position==1

