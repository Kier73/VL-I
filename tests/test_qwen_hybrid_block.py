"""QUALIFIED-007: independent scalar hybrid-attention and recovery contracts."""
from __future__ import annotations

from copy import deepcopy
import math
import struct

import pytest

from experiments.model_fabric.complete_native_layers import (
    LocalStatePages,QwenQualifiedLayer,
)
from vli.qwen_all_heads import QwenVectorizedCompleteLayer
from vli.qwen_gated_attention import (
    QwenQualifiedGatedAttention,rotate_text_rope,
    export_attention_kv,restore_attention_kv,
)
from vli.qwen_hybrid_block import QwenQualifiedHybridBlock
from vli.qwen_layer_chain import _f64_transfer_digest
from test_architecture_probe import nav_for
from test_complete_native_layers import qwen_fixture


def attention_specs():
    """Small but nonconstant original-topology tensor fixture, all F32."""
    width,heads,kvs,dim,intermediate=3,2,1,2,4
    def mat(rows,cols,scale=1):
        return [[(((i+1)*(j+3))%13-6)*scale/64
                 for j in range(cols)] for i in range(rows)]
    return {
      "input_layernorm.weight":[0.125,-0.0625,0.25],
      "post_attention_layernorm.weight":[-0.125,0.25,0.0625],
      "self_attn.q_norm.weight":[0.125,-0.0625],
      "self_attn.k_norm.weight":[-0.125,0.25],
      "self_attn.q_proj.weight":mat(2*heads*dim,width),
      "self_attn.k_proj.weight":mat(kvs*dim,width,2),
      "self_attn.v_proj.weight":mat(kvs*dim,width,2),
      "self_attn.o_proj.weight":mat(width,heads*dim,3),
      "mlp.gate_proj.weight":mat(intermediate,width,2),
      "mlp.up_proj.weight":mat(intermediate,width,3),
      "mlp.down_proj.weight":mat(width,intermediate,2),
    }


def nav_hybrid():
    _,matrices,vectors,conv=qwen_fixture()
    specs={}
    for layer in range(3):
        prefix=f"model.language_model.layers.{layer}."
        for name,m in matrices.items():
            xs=[z for row in m for z in row]
            specs[prefix+name]=("F32",[len(m),len(m[0])],
                struct.pack("<"+str(len(xs))+"f",*xs))
        for name,v in vectors.items():
            specs[prefix+name]=("F32",[len(v)],
                struct.pack("<"+str(len(v))+"f",*v))
        xs=[z for row in conv for z in row]
        specs[prefix+"linear_attn.conv1d.weight"]=(
            "F32",[len(conv),1,len(conv[0])],
            struct.pack("<"+str(len(xs))+"f",*xs))
    prefix="model.language_model.layers.3."
    for name,entries in attention_specs().items():
        if isinstance(entries[0],list):
            xs=[z for row in entries for z in row]
            shape=[len(entries),len(entries[0])]
        else:
            xs=entries;shape=[len(xs)]
        specs[prefix+name]=("F32",shape,
            struct.pack("<"+str(len(xs))+"f",*xs))
    return nav_for(specs)


def runners(nav,*,vectorized=True):
    cls=QwenVectorizedCompleteLayer if vectorized else QwenQualifiedLayer
    result=[cls(nav,layer=i,key_heads=1,value_heads=2,
                key_dim=2,value_dim=2) for i in range(3)]
    result.append(QwenQualifiedGatedAttention(
        nav,layer=3,query_heads=2,kv_heads=1,head_dim=2,
        rotary_dim=2,rope_theta=10000000.0))
    return result


def reference_attention(hidden,prior,position):
    """Separate scalar oracle, not invoking QwenQualifiedGatedAttention math."""
    src=attention_specs()
    def norm(x,w):
        scale=1/math.sqrt(math.fsum(v*v for v in x)/len(x)+1e-6)
        return [v*scale*(1+z) for v,z in zip(x,w)]
    def linear(name,vector):
        return [math.fsum(a*b for a,b in zip(row,vector))
                for row in src[name]]
    def sigmoid(v):
        if v>=0:return 1/(1+math.exp(-v))
        z=math.exp(v)
        return z/(1+z)
    def rope(z):
        a=position
        return [z[0]*math.cos(a)-z[1]*math.sin(a),
                z[1]*math.cos(a)+z[0]*math.sin(a)]
    x=norm(hidden,src["input_layernorm.weight"])
    qg=linear("self_attn.q_proj.weight",x)
    keys=rope(norm(linear("self_attn.k_proj.weight",x),
                   src["self_attn.k_norm.weight"]))
    vals=linear("self_attn.v_proj.weight",x)
    timeline=prior+[(keys,vals)]
    heads=[]
    for h in range(2):
        q=rope(norm(qg[h*4:h*4+2],src["self_attn.q_norm.weight"]))
        gate=qg[h*4+2:h*4+4]
        logits=[math.fsum(a*b for a,b in zip(q,k))/math.sqrt(2)
                for k,_ in timeline]
        largest=max(logits)
        exp=[math.exp(z-largest) for z in logits]
        weights=[z/math.fsum(exp) for z in exp]
        values=[math.fsum(p*v[i] for p,(_,v) in zip(weights,timeline))
                for i in range(2)]
        heads.extend(s*sigmoid(g) for s,g in zip(values,gate))
    attn=linear("self_attn.o_proj.weight",heads)
    residual=[a+b for a,b in zip(hidden,attn)]
    feed=norm(residual,src["post_attention_layernorm.weight"])
    g=linear("mlp.gate_proj.weight",feed)
    u=linear("mlp.up_proj.weight",feed)
    f=[v*sigmoid(v)*t for v,t in zip(g,u)]
    down=linear("mlp.down_proj.weight",f)
    return [a+b for a,b in zip(residual,down)],timeline


def test_text_rope_rotates_only_selected_dimensions():
    z=[.25,.75,1.5,-.5]
    got=rotate_text_rope(z,2,2,10000000.0)
    assert got[:2]==pytest.approx([
        .25*math.cos(2)-.75*math.sin(2),
        .75*math.cos(2)+.25*math.sin(2)])
    assert got[2:]==[1.5,-.5]


def test_gated_attention_matches_independent_scalar_reference_across_positions():
    nav=nav_hybrid()
    runner=runners(nav)[-1]
    previous=[]
    for idx,hidden in enumerate(([.3,.1,-.4],[-.2,.6,.5],[.4,-.3,.8])):
        got,report=runner.next(list(hidden))
        want,previous=reference_attention(hidden,previous,idx)
        assert got==pytest.approx(want,abs=2e-11)
        assert report["position"]==idx
        assert report["trained_query_heads"]==2
        assert report["trained_kv_heads"]==1
        assert report["kv_history_positions"]==idx+1
        assert report["complete_Qwen_transformer_layer"]
        assert len(report["full_attention_kv_state_digest"])==64


def test_real_chain_hands_exact_predecessor_output_across_three_boundaries():
    chain=QwenQualifiedHybridBlock(runners(nav_hybrid()))
    scalar=runners(nav_hybrid(),vectorized=False)
    history=[]
    for idx,hidden in enumerate(([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3])):
        got,receipt=chain.step(list(hidden),token_id=idx)
        reference=list(hidden)
        for gdn in scalar[:3]:
            reference,_=gdn.next(reference)
        reference,history=reference_attention(reference,history,idx)
        assert got==pytest.approx(reference,abs=2e-10)
        witnesses=receipt["layer_witnesses"]
        assert [w["kind"] for w in witnesses]==["GDN","GDN","GDN","GATED_GQA"]
        assert [w["layer"] for w in witnesses]==[0,1,2,3]
        assert witnesses[0]["input_f64_digest"]==_f64_transfer_digest(hidden)
        for a,b in zip(witnesses,witnesses[1:]):
            assert a["output_f64_digest"]==b["input_f64_digest"]
        assert witnesses[-1]["output_f64_digest"]==_f64_transfer_digest(got)
        assert receipt["all_four_original_layers_completed"]
        assert not receipt["original_decoder_completed"]
        assert chain.position==idx+1


def test_first_hybrid_block_paged_restart_matches_continuous(tmp_path):
    original=QwenQualifiedHybridBlock(runners(nav_hybrid()))
    original.step([.5,-.2,.3],token_id=63204)
    store=LocalStatePages(tmp_path)
    cp=original.checkpoint(store,input_sequence_digest="a"*64)
    assert len(cp["layer_checkpoints"])==4
    assert [p["layer"] for p in cp["layer_checkpoints"]]==[0,1,2,3]
    assert all(p.get("causal_pages") for p in cp["layer_checkpoints"][:3])
    assert cp["layer_checkpoints"][3]["kv_pages"]
    recovered=QwenQualifiedHybridBlock.restore(
        runners(nav_hybrid()),store,cp,
        expected_digest=cp["checkpoint_digest"],
        expected_input_sequence_digest="a"*64)
    assert recovered.head==original.head
    for idx,hidden in enumerate(([-.1,.2,.8],[.7,.6,-.3]),start=1):
        a,receipt_a=original.step(hidden,token_id=idx)
        b,receipt_b=recovered.step(hidden,token_id=idx)
        assert a==b
        assert receipt_a["chain_head"]==receipt_b["chain_head"]
        for r,s in zip(original.runners,recovered.runners):
            if isinstance(r,QwenQualifiedGatedAttention):
                assert r.kv_keys==s.kv_keys
                assert r.kv_values==s.kv_values
            else:
                assert r.recurrent_state==s.recurrent_state
                assert r.convolution_history==s.convolution_history


def test_failed_attention_downstream_rolls_back_all_causal_states(monkeypatch):
    chain=QwenQualifiedHybridBlock(runners(nav_hybrid()))
    chain.step([.3,.1,-.4])
    saved=deepcopy(chain._snapshots())
    old_head=chain.head
    orig=chain.runners[3]._projection
    def fail(part,values):
        if part=="mlp.down_proj.weight":
            raise ValueError("original source host departed")
        return orig(part,values)
    monkeypatch.setattr(chain.runners[3],"_projection",fail)
    with pytest.raises(ValueError,match="departed"):
        chain.step([-.2,.6,.5])
    assert chain._snapshots()==saved
    assert chain.head==old_head and chain.position==1


def test_refuse_incompatible_kv_manifest_and_corruption(tmp_path):
    chain=QwenQualifiedHybridBlock(runners(nav_hybrid()))
    chain.step([.3,.1,-.4])
    store=LocalStatePages(tmp_path)
    cp=chain.checkpoint(store,input_sequence_digest="b"*64)
    def restore(record, lineage="b"*64):
        return QwenQualifiedHybridBlock.restore(
            runners(nav_hybrid()),store,record,
            expected_digest=cp["checkpoint_digest"],
            expected_input_sequence_digest=lineage)
    with pytest.raises(ValueError,match="Modified|unpinned"):
        restore({**cp,"position":2})
    with pytest.raises(ValueError,match="lineage|Wrong"):
        restore(cp,lineage="c"*64)
    page=cp["layer_checkpoints"][-1]["kv_pages"][0]
    (tmp_path/page["sha256"]).write_bytes(b"tampered")
    with pytest.raises(ValueError,match="Corrupt|corrupt"):
        restore(cp)


def test_wrong_attention_geometry_or_missing_source_is_rejected():
    nav,*_=qwen_fixture()
    with pytest.raises(ValueError,match="Incomplete original attention"):
        QwenQualifiedGatedAttention(nav,layer=3,query_heads=2,
                                    kv_heads=1,head_dim=2,rotary_dim=2)
    nav=nav_hybrid()
    with pytest.raises(ValueError,match="Unsupported"):
        QwenQualifiedGatedAttention(nav,layer=3,query_heads=2,
                                    kv_heads=3,head_dim=2,rotary_dim=2)
    with pytest.raises(ValueError,match="budget"):
        QwenQualifiedGatedAttention(nav,layer=3,query_heads=2,kv_heads=1,
                                    head_dim=2,rotary_dim=2,budget_bytes=1)
