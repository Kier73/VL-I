"""NAV-002: exact, independently challenged Markov-state quotient proofs.

These tests use qualified Qwen *topology fixtures*, not original 27B
pretrained embedding rows or verified tokenizer-to-input mappings.
"""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from experiments.model_fabric.complete_native_layers import LocalStatePages
from vli.counterfactual_navigation import CounterfactualCausalGround
from vli.exact_state_quotient import ExactCausalStateQuotient
from test_counterfactual_navigation import (
    two_factory, hybrid_factory, advance, A, B, C, D,
)


@pytest.mark.parametrize("factory", [two_factory, hybrid_factory])
def test_distinct_history_heads_share_exact_numerical_state_and_predict_new_future(tmp_path, factory):
    ground = CounterfactualCausalGround(factory, LocalStatePages(tmp_path))
    _, first, _ = advance(ground, ground.root_coordinate, A, 11)
    _, twin, _ = advance(ground, ground.root_coordinate, A, 12)
    observed, successor, _ = advance(ground, first["coordinate"], B, 20)
    assert first["coordinate"] != twin["coordinate"]
    assert first["causal_head"] != twin["causal_head"]
    assert first["checkpoint_digest"] != twin["checkpoint_digest"]
    quotient = ExactCausalStateQuotient(ground, expected_head=ground.head)
    assert quotient.keys[first["coordinate"]] == quotient.keys[twin["coordinate"]]
    assert quotient.metrics()["merged_history_state_witnesses"] >= 1
    assert quotient.keys[first["coordinate"]] != quotient.keys[successor["coordinate"]]
    initial_head = ground.head
    initial_count = ground.executed_transitions
    proof, outputs = quotient.navigate(
        twin["coordinate"], [(21,B)],
        expected_parent_digest=twin["node_digest"], expected_head=initial_head)
    assert outputs[0]["numerical_output"] == observed
    assert outputs[0]["was_originally_executed"] is False
    assert outputs[0]["coordinate"] not in ground.nodes
    assert proof["steps"][0]["representative_coordinate"] == successor["coordinate"]
    assert proof["steps"][0]["parent_exact_state_key"] == quotient.keys[twin["coordinate"]]
    assert proof["native_trained_transitions_executed_by_navigation"] == 0
    assert ground.head == initial_head and ground.executed_transitions == initial_count
    assert quotient.verify(proof,expected_digest=proof["proof_digest"],
                           expected_head=initial_head,
                           expected_parent_digest=twin["node_digest"]) == outputs
    # Independent fresh qualified mathematical execution, AFTER prediction.
    oracle = factory()
    oracle.step(A,token_id=12)
    oracle_output,_ = oracle.step(B,token_id=21)
    assert oracle_output == outputs[0]["numerical_output"]
    # Source-specific chain heads differ, even when numerical outputs agree.
    measured, new_branch,_ = advance(ground, twin["coordinate"], B, 21)
    assert measured == observed
    assert new_branch["causal_head"] != successor["causal_head"]


@pytest.mark.parametrize("factory", [two_factory,hybrid_factory])
def test_four_step_label_histories_navigate_16_virtual_futures_from_four_trained_edges(
        tmp_path, factory):
    ground = CounterfactualCausalGround(factory, LocalStatePages(tmp_path))
    past = ground.root_coordinate
    vectors = (A,B,C,D)
    native_outputs = []
    for j, x in enumerate(vectors):
        out, record, _ = advance(ground,past,x,10+j)
        native_outputs.append(out)
        past = record["coordinate"]
    assert ground.executed_transitions == 4
    pinned = ground.head
    quotient = ExactCausalStateQuotient(ground,expected_head=pinned)
    results = set()
    virtual_count = 0
    for labels in range(16):
        actions = [(10+2*depth+(labels >> depth & 1), vector)
                   for depth,vector in enumerate(vectors)]
        proof, outputs = quotient.navigate(
            ground.root_coordinate, actions,
            expected_parent_digest=ground.root_digest, expected_head=pinned)
        assert [o["numerical_output"] for o in outputs] == native_outputs
        assert len(proof["steps"]) == 4
        assert all(step["trained_layer_executions_for_this_navigation"] == 0
                   for step in proof["steps"])
        assert quotient.verify(proof,expected_digest=proof["proof_digest"],
                               expected_parent_digest=ground.root_digest,
                               expected_head=pinned) == outputs
        results.add(outputs[-1]["coordinate"])
        virtual_count += sum(not item["was_originally_executed"] for item in outputs)
    assert len(results) == 16
    assert virtual_count > 0
    assert ground.executed_transitions == 4
    assert len(ground.nodes) == 4
    assert ground.head == pinned
    assert quotient.metrics()["unique_observed_numerical_relations"] == 4
    # This is specifically a numerical input-preserving *relabeling* family,
    # NOT evidence that 16 different natural-language token histories match.


@pytest.mark.parametrize("factory", [two_factory, hybrid_factory])
def test_same_coarse_visible_output_does_not_qualify_different_state(tmp_path,factory):
    ground = CounterfactualCausalGround(factory,LocalStatePages(tmp_path))
    a_output,a,_ = advance(ground,ground.root_coordinate,A,1)
    b_output,b,_ = advance(ground,ground.root_coordinate,C,2)
    _, observed, _ = advance(ground,a["coordinate"],B,3)
    # They collapse under a deliberately low-information observer projection.
    visible = lambda x: int(abs(x[0])//1e30)
    assert visible(a_output) == visible(b_output) == 0
    quotient = ExactCausalStateQuotient(ground,expected_head=ground.head)
    assert quotient.keys[a["coordinate"]] != quotient.keys[b["coordinate"]]
    with pytest.raises(ValueError,match="UNRESOLVED"):
        quotient.navigate(b["coordinate"], [(3,B)],
            expected_parent_digest=b["node_digest"],expected_head=ground.head)
    assert observed["coordinate"] in ground.nodes


def test_unknown_future_input_is_not_predicted_even_from_same_exact_state(tmp_path):
    ground=CounterfactualCausalGround(two_factory,LocalStatePages(tmp_path))
    _,a,_=advance(ground,ground.root_coordinate,A,1)
    _,b,_=advance(ground,ground.root_coordinate,A,2)
    _,observed,_=advance(ground,a["coordinate"],B,3)
    quotient=ExactCausalStateQuotient(ground,expected_head=ground.head)
    with pytest.raises(ValueError,match="UNRESOLVED"):
        quotient.navigate(b["coordinate"],[(4,D)],
            expected_parent_digest=b["node_digest"],expected_head=ground.head)
    assert observed["coordinate"] in ground.nodes


def test_full_state_pages_required_after_initial_quotient_admission(tmp_path):
    ground=CounterfactualCausalGround(hybrid_factory,LocalStatePages(tmp_path))
    _,a,_=advance(ground,ground.root_coordinate,A,1)
    _,b,_=advance(ground,ground.root_coordinate,A,2)
    _,child,_=advance(ground,a["coordinate"],B,3)
    quotient=ExactCausalStateQuotient(ground,expected_head=ground.head)
    proof, outputs=quotient.navigate(
        b["coordinate"],[(4,B)],
        expected_parent_digest=b["node_digest"],expected_head=ground.head)
    assert outputs[0]["numerical_output"]
    kv=child["checkpoint"]["layer_checkpoints"][-1]["kv_pages"][0]
    (tmp_path/kv["sha256"]).write_bytes(b"forged")
    with pytest.raises((ValueError,OSError),match="Corrupt|corrupt"):
        quotient.navigate(b["coordinate"],[(4,B)],
            expected_parent_digest=b["node_digest"],expected_head=ground.head)
    with pytest.raises((ValueError,OSError),match="Corrupt|corrupt"):
        quotient.verify(proof,expected_digest=proof["proof_digest"],
            expected_head=ground.head,expected_parent_digest=b["node_digest"])


def test_stale_authority_revised_proofs_and_wrong_fingerprint_refuse(tmp_path):
    ground=CounterfactualCausalGround(two_factory,LocalStatePages(tmp_path))
    _,a,_=advance(ground,ground.root_coordinate,A,1)
    _,b,_=advance(ground,ground.root_coordinate,A,2)
    _,child,_=advance(ground,a["coordinate"],B,3)
    head=ground.head
    quotient=ExactCausalStateQuotient(ground,expected_head=head)
    proof, _ = quotient.navigate(b["coordinate"],[(4,B)],
        expected_parent_digest=b["node_digest"], expected_head=head)
    changes=[
        {**proof,"implementation_fingerprint":"0"*64},
        {**proof,"steps":[{**proof["steps"][0],"result_f64_digest":"0"*64}]},
        {**proof,"initial_parent":a["coordinate"]},
        {**proof,"native_trained_transitions_executed_by_navigation":1},
    ]
    for altered in changes:
        with pytest.raises(ValueError,match="Unpinned|altered"):
            quotient.verify(altered,expected_digest=proof["proof_digest"],
                expected_head=head,expected_parent_digest=b["node_digest"])
    with pytest.raises(ValueError,match="Stale"):
        quotient.navigate(b["coordinate"],[(4,B)],
            expected_parent_digest=b["node_digest"],expected_head="0"*64)
    with pytest.raises(ValueError,match="Stale"):
        quotient.navigate(b["coordinate"],[(4,B)],
            expected_parent_digest=a["node_digest"],expected_head=head)
    assert quotient.verify(proof, expected_digest=proof["proof_digest"],
                           expected_parent_digest=b["node_digest"],
                           expected_head=head)
    advance(ground,ground.root_coordinate,D,8)
    with pytest.raises(ValueError,match="Stale"):
        quotient.verify(proof,expected_digest=proof["proof_digest"],
            expected_parent_digest=b["node_digest"],expected_head=head)


def test_newly_executed_duplicate_relation_must_agree_exactly(tmp_path):
    ground=CounterfactualCausalGround(two_factory,LocalStatePages(tmp_path))
    _,a,_=advance(ground,ground.root_coordinate,A,1)
    _,b,_=advance(ground,ground.root_coordinate,A,2)
    _,one,_=advance(ground,a["coordinate"],B,3)
    _,two,_=advance(ground,b["coordinate"],B,4)
    assert one["causal_head"] != two["causal_head"]
    quotient=ExactCausalStateQuotient(ground,expected_head=ground.head)
    assert quotient.metrics()["duplicate_relation_consistency_checks"] >= 2
    assert quotient.keys[one["coordinate"]] == quotient.keys[two["coordinate"]]


def test_snapshot_recovery_can_verify_previously_unvisited_numerical_path(tmp_path):
    pages=LocalStatePages(tmp_path)
    ground=CounterfactualCausalGround(hybrid_factory,pages)
    _,a,_=advance(ground,ground.root_coordinate,A,1)
    _,b,_=advance(ground,ground.root_coordinate,A,2)
    _,child,_=advance(ground,a["coordinate"],B,3)
    source_head=ground.head
    quotient=ExactCausalStateQuotient(ground,expected_head=source_head)
    proof, outputs=quotient.navigate(b["coordinate"],[(4,B)],
        expected_parent_digest=b["node_digest"],expected_head=source_head)
    snapshot=json.loads(json.dumps(ground.snapshot()))
    new_ground=CounterfactualCausalGround.restore(
        hybrid_factory,pages,snapshot,expected_digest=snapshot["snapshot_digest"])
    recovered=ExactCausalStateQuotient(new_ground,expected_head=source_head)
    assert recovered.verify(json.loads(json.dumps(proof)),
        expected_digest=proof["proof_digest"],expected_head=source_head,
        expected_parent_digest=b["node_digest"])==outputs
    assert recovered.keys[b["coordinate"]] == quotient.keys[b["coordinate"]]


def test_no_premature_promotion_of_unvisited_branch_to_execution_record(tmp_path):
    ground=CounterfactualCausalGround(two_factory,LocalStatePages(tmp_path))
    _,a,_=advance(ground,ground.root_coordinate,A,1)
    _,b,_=advance(ground,ground.root_coordinate,A,2)
    _,_,_=advance(ground,a["coordinate"],B,3)
    quotient=ExactCausalStateQuotient(ground,expected_head=ground.head)
    proof, outputs=quotient.navigate(
        b["coordinate"],[(4,B)],expected_parent_digest=b["node_digest"],
        expected_head=ground.head)
    target=proof["steps"][0]["target_coordinate"]
    assert target not in ground.nodes
    assert outputs[0]["was_originally_executed"] is False
    assert ground.metrics()["new_trained_transitions_this_receiver"]==3
    with pytest.raises(ValueError,match="unknown observed state"):
        ground.observe(target,expected_head=ground.head)


def test_invalid_specifications_or_nonfinite_hidden_are_refused(tmp_path):
    ground=CounterfactualCausalGround(two_factory,LocalStatePages(tmp_path))
    _,a,_=advance(ground,ground.root_coordinate,A,1)
    quotient=ExactCausalStateQuotient(ground,expected_head=ground.head)
    with pytest.raises(ValueError,match="Unqualified"):
        quotient.navigate(a["coordinate"],[(2,[float("nan"),0,1])],
            expected_parent_digest=a["node_digest"],expected_head=ground.head)
    with pytest.raises(ValueError,match="dimension"):
        quotient.navigate(a["coordinate"],[(2,[1.0])],
            expected_parent_digest=a["node_digest"],expected_head=ground.head)
    with pytest.raises(ValueError,match="finite"):
        quotient.navigate(a["coordinate"],[(2,C)]*65,
            expected_parent_digest=a["node_digest"],expected_head=ground.head)
    with pytest.raises(ValueError,match="UNRESOLVED"):
        quotient.navigate(a["coordinate"],[(2,C)],
            expected_parent_digest=a["node_digest"],expected_head=ground.head)
