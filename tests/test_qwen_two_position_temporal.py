"""QUALIFIED-010: temporal attention is not first-position softmax parity.

The experiment must distinguish historical learned query-key selection
from trivial single-key attention, including with exact predecessor
states produced by all seven prior original-topology trained layers.
"""
from __future__ import annotations

import pytest

torch=pytest.importorskip("torch")

from vli.qwen_architecture_navigator import QwenArchitectureCausalPrefix
from vli.qwen_gated_attention import QwenQualifiedGatedAttention
from vli.qwen_torch_oracle import OriginalQwenTorchGQA
from vli.qwen_layer_chain import _f64_transfer_digest
from test_qwen_architecture_navigator import nav_eight_layers,plan_for


def test_two_token_historical_attention_from_full_seven_layer_inputs():
    nav=nav_eight_layers()
    plan=plan_for(nav)
    seven=QwenArchitectureCausalPrefix(nav,plan,stop_layer=7,max_positions=64)
    native=QwenQualifiedGatedAttention(
        nav,layer=7,query_heads=2,kv_heads=1,
        head_dim=2,rotary_dim=2,rope_theta=10_000_000.0)
    reference=OriginalQwenTorchGQA(nav,plan,layer=7)
    outputs=[]
    for position,(token,x) in enumerate((
        (63204,[.5,-.2,.3]),(628,[-.1,.2,.8]))):
        hidden6,prefix=seven.step(x,token_id=token)
        got,actual=native.next(hidden6)
        want,independent=reference.next(hidden6)
        assert got==pytest.approx(want,abs=2e-10)
        assert len(got)==3
        assert _f64_transfer_digest(hidden6)==prefix["layer_witnesses"][-1][
            "output_f64_digest"]
        assert len(independent["attention_probabilities_per_query_head"])==2
        assert independent["causal_kv_positions"]==position+1
        assert actual["kv_history_positions"]==position+1
        for probabilities in independent["attention_probabilities_per_query_head"]:
            assert len(probabilities)==position+1
            assert sum(probabilities)==pytest.approx(1.0,abs=1e-12)
            assert all(0.0<=z<=1.0 for z in probabilities)
        if position==0:
            assert independent["max_deviation_from_uniform_attention"]==0
            assert independent["max_context_difference_from_current_token_values"]==0
        else:
            assert independent["max_deviation_from_uniform_attention"]>0
            assert independent["max_context_difference_from_current_token_values"]>0
        outputs.append(_f64_transfer_digest(got))
    assert outputs[0]!=outputs[1]


def test_separate_attention_heads_have_full_history_probabilities():
    nav=nav_eight_layers()
    plan=plan_for(nav)
    ref=OriginalQwenTorchGQA(nav,plan,layer=7)
    _,first=ref.next([.2,.4,-.1])
    _,second=ref.next([.8,.3,.7])
    assert len(first["attention_probabilities_per_query_head"])==2
    assert all(row==[1.0] for row in first["attention_probabilities_per_query_head"])
    assert all(len(row)==2 for row in second["attention_probabilities_per_query_head"])
    assert second["causal_kv_positions"]==2
    assert second["max_context_difference_from_current_token_values"]>0
