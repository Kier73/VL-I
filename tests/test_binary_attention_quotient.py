"""Original Qwen 2-token learned softmax as an exact, falsifiable quotient.

Coordinates are formed from real original trained head probabilities
archived by QUALIFIED-010. Algebraic reconstruction is verified
independently from the native attention implementation.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from vli.binary_attention_quotient import (
    binary_softmax_source_digest,admit_binary_attention_quotient,
    reconstruct_binary_attention,realize_binary_context,
)

ARCHIVE=(Path(__file__).resolve().parents[1]/
    "experiments/qwen_all_heads/datasets/"
    "2026-11-19-real-original-qwen-two-token-trained-attention-parity.json")


def real_two_token_probability_pairs():
    observed=json.loads(ARCHIVE.read_text("utf8"))
    x=observed["positions"][1]["previous_token_attention_probability_by_query_head"]
    return [[p,1.0-p] for p in x]


def test_all_24_original_qwen_heads_admit_one_exact_coordinate_each():
    observations=real_two_token_probability_pairs()
    source_pin=binary_softmax_source_digest(observations)
    quotient=admit_binary_attention_quotient(
        observations,expected_source_digest=source_pin)
    assert quotient["head_count"]==24
    assert quotient["causal_history_positions"]==2
    assert quotient["original_probability_scalar_count"]==48
    assert quotient["quotient_probability_coordinate_count"]==24
    assert quotient["max_probability_reconstruction_error"]<3e-15
    assert quotient["not_a_claim_of_eliminated_weight_projections"]
    assert quotient["not_a_claim_of_full_attention_state_compression"]
    for h,row in enumerate(observations):
        recovered=reconstruct_binary_attention(quotient,head=h)
        assert recovered==pytest.approx(row,abs=3e-15)
        assert math.isclose(sum(recovered),1,abs_tol=3e-15)
    assert min(x[0] for x in observations)<0.01
    assert max(x[0] for x in observations)>0.95


def test_binary_score_gap_is_exact_two_logit_softmax_relation():
    # This is an independent algebraic source, not VL-I attention math.
    query=[.4,-.6,.8,.2]
    past=[-.2,.1,.9,.7]
    present=[.6,-.5,-.7,.1]
    score_p=sum(a*b for a,b in zip(query,past))/math.sqrt(len(query))
    score_n=sum(a*b for a,b in zip(query,present))/math.sqrt(len(query))
    mass=math.exp(score_p-max(score_p,score_n))
    other=math.exp(score_n-max(score_p,score_n))
    probabilities=[[mass/(mass+other),other/(mass+other)]]
    quotient=admit_binary_attention_quotient(
        probabilities,expected_source_digest=binary_softmax_source_digest(probabilities))
    delta=quotient["head_score_difference_coordinates"][0]
    assert delta==pytest.approx(score_p-score_n,abs=1e-14)
    assert reconstruct_binary_attention(quotient,head=0)==pytest.approx(
        probabilities[0],abs=1e-14)


def test_attended_values_are_exactly_reconstructed_from_single_head_coordinate():
    observations=real_two_token_probability_pairs()
    quotient=admit_binary_attention_quotient(
        observations,expected_source_digest=binary_softmax_source_digest(observations))
    earlier=[.25,-1.5,.125,-.5]
    now=[-.125,.2,1.0,.8]
    for head,pair in enumerate(observations):
        expected=[pair[0]*a+pair[1]*b for a,b in zip(earlier,now)]
        actual=realize_binary_context(
            quotient,head=head,prior_values=earlier,current_values=now)
        assert actual==pytest.approx(expected,abs=3e-15)


def test_observer_pin_and_mutated_coordinate_rejected():
    observed=real_two_token_probability_pairs()
    good=binary_softmax_source_digest(observed)
    with pytest.raises(ValueError,match="pin"):
        admit_binary_attention_quotient(observed,expected_source_digest="a"*64)
    quotient=admit_binary_attention_quotient(observed,expected_source_digest=good)
    changed=deepcopy(quotient)
    changed["head_score_difference_coordinates"][0]+=1.0
    with pytest.raises(ValueError,match="Unadmitted"):
        reconstruct_binary_attention(changed,head=0)
    with pytest.raises(ValueError,match="Unadmitted"):
        reconstruct_binary_attention(quotient,head=24)
    with pytest.raises(ValueError,match="Missing"):
        realize_binary_context(quotient,head=0,prior_values=[1.0],current_values=[])
    observed[0]=[1.0,0.0]
    with pytest.raises(ValueError,match="Unqualified"):
        binary_softmax_source_digest(observed)
