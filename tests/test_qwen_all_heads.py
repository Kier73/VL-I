"""Qualified all-head Qwen host: independent and full native-head regressions."""
from __future__ import annotations

import copy
import math
import struct
from unittest.mock import patch

import pytest

from experiments.model_fabric.architecture_ops import state_digest
from experiments.model_fabric.complete_native_layers import (
    QwenQualifiedLayer, LocalStatePages,
    export_qwen_causal_ground, restore_qwen_causal_ground,
)
from vli.qwen_all_heads import QwenAllHeadsRecurrentHost
from vli.qualified_projection_host import fast_rows

from test_architecture_probe import nav_for
from test_complete_native_layers import qwen_fixture


def max_nested_numeric_error(a,b):
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        assert len(a)==len(b)
        return max((max_nested_numeric_error(x,y) for x,y in zip(a,b)),
                   default=0.0)
    return abs(a-b)


def test_qualified_all_head_recurrence_agrees_with_scalar_native_layer():
    one,*_=qwen_fixture()
    two,*_=qwen_fixture()
    host=QwenAllHeadsRecurrentHost(one,layer=0,key_heads=1,
                                  value_heads=2,key_dim=2,value_dim=2)
    reference=QwenQualifiedLayer(two,layer=0,key_heads=1,
                                 value_heads=2,key_dim=2,value_dim=2)
    for n,x in enumerate(([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3])):
        observed=host.advance_recurrent(list(x))
        reference.next(list(x))
        assert observed["position"]==n
        # NumPy BLAS and scalar fsum have different reduction orders.
        # Compare independent arithmetic numerically; use exact state digests
        # only when restoring the same host backend.
        assert max_nested_numeric_error(
            host.recurrent_state,reference.recurrent_state)<2e-10
        assert max_nested_numeric_error(
            host.convolution_history,reference.convolution_history)<2e-10
        assert observed["max_independent_reference_error"]<=2e-7
        assert observed["new_weight_payload_bytes"]>0
        assert observed["new_qualified_ranges"]>0
        assert observed["complete_Qwen_transformer_layer"] is False
    assert host.position==3


def qwen_48_fixture():
    key_heads,value_heads,kdim,vdim,width=16,48,16,16,3
    mixed=2*key_heads*kdim+value_heads*vdim
    specs={
        'input_layernorm.weight':([width],[0.]*width),
        'linear_attn.in_proj_qkv.weight':([mixed,width],[.03125]*(mixed*width)),
        'linear_attn.in_proj_z.weight':([value_heads*vdim,width],
                                       [.03125]*(value_heads*vdim*width)),
        'linear_attn.in_proj_a.weight':([value_heads,width],
                                       [.03125]*(value_heads*width)),
        'linear_attn.in_proj_b.weight':([value_heads,width],
                                       [.03125]*(value_heads*width)),
        'linear_attn.conv1d.weight':([mixed,1,2],[.25,.5]*mixed),
        'linear_attn.A_log':([value_heads],[0.]*value_heads),
        'linear_attn.dt_bias':([value_heads],[0.]*value_heads),
        'linear_attn.norm.weight':([vdim],[1.]*vdim),
        'linear_attn.out_proj.weight':([width,value_heads*vdim],
                                      [.0078125]*(width*value_heads*vdim)),
        'post_attention_layernorm.weight':([width],[0.]*width),
        'mlp.gate_proj.weight':([4,width],[.0625]*(4*width)),
        'mlp.up_proj.weight':([4,width],[.03125]*(4*width)),
        'mlp.down_proj.weight':([width,4],[.03125]*(width*4)),
    }
    prefix='model.language_model.layers.0.'
    tensors={prefix+name:('F32',shape,
            struct.pack('<'+str(len(v))+'f',*v))
            for name,(shape,v) in specs.items()}
    return nav_for(tensors)


def test_48_heads_and_paged_host_recovery_without_any_missing_value_head(tmp_path):
    nav=qwen_48_fixture()
    h=QwenAllHeadsRecurrentHost(nav,layer=0,
                               key_heads=16,value_heads=48,
                               key_dim=16,value_dim=16)
    one=h.advance_recurrent([.1,.2,.3])
    assert one["trained_query_heads"]==16
    assert one["trained_key_heads"]==16
    assert one["trained_value_heads"]==48
    assert one["full_recurrent_output_width"]==48*16
    assert len(h.recurrent_state)==48
    assert len(h.recurrent_state[0])==16
    assert len(h.recurrent_state[0][0])==16
    pages=LocalStatePages(tmp_path)
    cp=export_qwen_causal_ground(h,pages,input_sequence_digest='b'*64)
    assert len(cp['causal_pages'])>=2
    saved=QwenAllHeadsRecurrentHost(qwen_48_fixture(),layer=0,
                                   key_heads=16,value_heads=48,
                                   key_dim=16,value_dim=16)
    restore_qwen_causal_ground(
        saved,pages,cp,expected_digest=cp['checkpoint_digest'],
        expected_input_sequence_digest='b'*64)
    a=h.advance_recurrent([.3,.2,.1])
    b=saved.advance_recurrent([.3,.2,.1])
    assert a["full_recurrent_state_digest"]==b["full_recurrent_state_digest"]
    assert a["output_projection_row0"]==b["output_projection_row0"]
    assert a["full_convolution_state_digest"]==b["full_convolution_state_digest"]
    assert a["position"]==1


def test_host_refuses_delegated_math_failure_before_causal_commit():
    nav,*_=qwen_fixture()
    host=QwenAllHeadsRecurrentHost(nav,layer=0,key_heads=1,
                                  value_heads=2,key_dim=2,value_dim=2)
    original=fast_rows
    def deny(nav,name,x,**kwargs):
        if name.endswith('out_proj.weight'):
            raise ValueError('failed trained projection at model host')
        return original(nav,name,x,**kwargs)
    with patch('vli.qwen_all_heads.fast_rows',side_effect=deny):
        with pytest.raises(ValueError,match='failed trained projection'):
            host.advance_recurrent([.1,.2,.3])
    assert host.position==0 and host.recurrent_state is None
    assert host.convolution_history is None


def test_host_refuses_bad_original_token_width_and_nonfinite_vector():
    nav,*_=qwen_fixture()
    host=QwenAllHeadsRecurrentHost(nav,layer=0,key_heads=1,
                                  value_heads=2,key_dim=2,value_dim=2)
    for bad in ([.1,.2],[.1,float('nan'),.2]):
        with pytest.raises(ValueError):
            host.advance_recurrent(bad)
    assert host.position==0


def test_model_host_full_layer_unchanged_and_can_be_executed_with_same_source():
    nav,*_=qwen_fixture()
    host=QwenQualifiedLayer(nav,layer=0,key_heads=1,
                            value_heads=2,key_dim=2,value_dim=2)
    result,receipt=host.next([.1,.2,.3])
    assert len(result)==3
    assert receipt['scope'].startswith('full GDN layer')
    assert receipt['layer']==0


def test_vectorized_complete_layer_matches_independent_scalar_layer():
    from vli.qwen_all_heads import QwenVectorizedCompleteLayer
    one,*_=qwen_fixture()
    two,*_=qwen_fixture()
    fast=QwenVectorizedCompleteLayer(one,layer=0,key_heads=1,
                                     value_heads=2,key_dim=2,value_dim=2)
    scalar=QwenQualifiedLayer(two,layer=0,key_heads=1,
                              value_heads=2,key_dim=2,value_dim=2)
    for index,x in enumerate(([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3])):
        accelerated,receipt=fast.next(list(x))
        reference,_=scalar.next(list(x))
        assert accelerated==pytest.approx(reference,abs=2e-12)
        assert receipt["position"]==index
        assert receipt["complete_Qwen_transformer_layer"] is True
        assert receipt["complete_original_64_layer_decoder"] is False
        assert max_nested_numeric_error(
            fast.recurrent_state,scalar.recurrent_state)<2e-10
        assert max_nested_numeric_error(
            fast.convolution_history,scalar.convolution_history)<2e-10


def test_complete_layer_rejects_downstream_projection_error_without_advancing():
    from vli.qwen_all_heads import QwenVectorizedCompleteLayer
    nav,*_=qwen_fixture()
    fast=QwenVectorizedCompleteLayer(nav,layer=0,key_heads=1,
                                     value_heads=2,key_dim=2,value_dim=2)
    real=fast_rows
    def reject(nav,name,x,**kwargs):
        if name.endswith("mlp.down_proj.weight"):
            raise ValueError("Qualified model host absent for FFN")
        return real(nav,name,x,**kwargs)
    with patch("vli.qwen_all_heads.fast_rows",side_effect=reject):
        with pytest.raises(ValueError,match="host absent"):
            fast.next([.1,.2,.3])
    assert fast.position==0
    assert fast.recurrent_state is None
    assert fast.convolution_history is None
