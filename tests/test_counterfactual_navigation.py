"""NAV-001: independent causal parity and adversarial branching contracts."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
from experiments.model_fabric.complete_native_layers import LocalStatePages

from vli.counterfactual_navigation import CounterfactualCausalGround
from vli.qwen_layer_chain import QwenCausalGDNChain
from vli.qwen_all_heads import QwenVectorizedCompleteLayer
from vli.qwen_hybrid_block import QwenQualifiedHybridBlock
from test_qwen_layer_chain import two_layer_fixture, executor
from test_qwen_hybrid_block import nav_hybrid, runners

A = [.5, -.2, .3]
B = [-.1, .2, .8]
C = [.7, .6, -.3]
D = [.4, -.6, .2]


def two_factory():
    return QwenCausalGDNChain(executor(two_layer_fixture(), QwenVectorizedCompleteLayer))


def hybrid_factory():
    return QwenQualifiedHybridBlock(runners(nav_hybrid()))


def advance(nav, parent, hidden, token):
    node = nav.nodes.get(parent)
    return nav.advance(parent, hidden, token_id=token,
                       expected_parent_digest=(node["node_digest"] if node else nav.root_digest),
                       expected_head=nav.head)


@pytest.mark.parametrize("factory", [two_factory, hybrid_factory])
def test_forks_equal_independently_uninterrupted_trained_layer_math(tmp_path, factory):
    nav = CounterfactualCausalGround(factory, LocalStatePages(tmp_path))
    prefix, anchor, mode = advance(nav, nav.root_coordinate, A, 63204)
    assert mode == "NEW_TRAINED_TRANSITION"
    left, lnode, _ = advance(nav, anchor["coordinate"], B, 628)
    right, rnode, _ = advance(nav, anchor["coordinate"], C, 729)
    assert lnode["coordinate"] != rnode["coordinate"]
    assert lnode["input_sequence_digest"] != rnode["input_sequence_digest"]
    assert lnode["previous_causal_head"] == rnode["previous_causal_head"] == anchor["causal_head"]
    assert lnode["depth"] == rnode["depth"] == 2
    for hidden, output, record in ((B, left, lnode), (C, right, rnode)):
        independent = factory()
        first, _ = independent.step(A, token_id=63204)
        expected, receipt = independent.step(hidden, token_id=record["token_id"])
        assert prefix == first
        assert output == expected
        assert receipt["chain_head"] == record["causal_head"]
        # Transport/range heads differ after rehoming; mathematical transfer
        # digests, not observer-local acquisition history, must agree.
        for a, b in zip(receipt["layer_witnesses"], record["receipt"]["layer_witnesses"]):
            assert a["input_f64_digest"] == b["input_f64_digest"]
            assert a["output_f64_digest"] == b["output_f64_digest"]
    assert nav.executed_transitions == 3
    assert nav.observe(lnode["coordinate"], expected_head=nav.head) == left
    assert nav.observe(rnode["coordinate"], expected_head=nav.head) == right


@pytest.mark.parametrize("factory", [two_factory, hybrid_factory])
def test_exact_repeat_navigates_retained_future_not_another_transition(tmp_path, factory):
    nav = CounterfactualCausalGround(factory, LocalStatePages(tmp_path))
    _, anchor, _ = advance(nav, nav.root_coordinate, A, 1)
    output, record, _ = advance(nav, anchor["coordinate"], B, 2)
    head, count = nav.head, nav.executed_transitions
    again, same, mode = advance(nav, anchor["coordinate"], B, 2)
    assert (again, same) == (output, record)
    assert mode == "RETAINED_NO_TRAINED_TRANSITION"
    assert nav.head == head and nav.executed_transitions == count
    assert nav.reused_transitions == 1
    _, alternate, _ = advance(nav, anchor["coordinate"], C, 3)
    assert alternate["coordinate"] != record["coordinate"]
    assert nav.observe(record["coordinate"], expected_head=nav.head) == output


@pytest.mark.parametrize("factory", [two_factory, hybrid_factory])
def test_identical_token_and_input_do_not_erase_distinct_parent_histories(tmp_path, factory):
    nav = CounterfactualCausalGround(factory, LocalStatePages(tmp_path))
    _, p1, _ = advance(nav, nav.root_coordinate, A, 10)
    _, p2, _ = advance(nav, nav.root_coordinate, D, 10)
    assert p1["coordinate"] != p2["coordinate"]
    _, a, _ = advance(nav, p1["coordinate"], C, 11)
    _, b, _ = advance(nav, p2["coordinate"], C, 11)
    assert a["coordinate"] != b["coordinate"]
    assert a["input_sequence_digest"] != b["input_sequence_digest"]
    assert a["previous_causal_head"] != b["previous_causal_head"]
    assert a["checkpoint_digest"] != b["checkpoint_digest"]


@pytest.mark.parametrize("factory", [two_factory, hybrid_factory])
def test_checkpoint_recovers_both_futures_and_extends_one_without_changing_other(tmp_path, factory):
    pages = LocalStatePages(tmp_path)
    original = CounterfactualCausalGround(factory, pages)
    _, p, _ = advance(original, original.root_coordinate, A, 10)
    _, left, _ = advance(original, p["coordinate"], B, 11)
    _, right, _ = advance(original, p["coordinate"], C, 12)
    snapshot = original.snapshot()
    recovered = CounterfactualCausalGround.restore(
        factory, pages, json.loads(json.dumps(snapshot)),
        expected_digest=snapshot["snapshot_digest"])
    assert recovered.head == original.head
    assert recovered.nodes == original.nodes
    output, next_node, status = advance(recovered, left["coordinate"], D, 13)
    assert status == "NEW_TRAINED_TRANSITION"
    reference = factory()
    for hidden, token in ((A, 10), (B, 11)):
        reference.step(hidden, token_id=token)
    expected, receipt = reference.step(D, token_id=13)
    assert output == expected
    assert next_node["causal_head"] == receipt["chain_head"]
    assert recovered.observe(right["coordinate"], expected_head=recovered.head) == original.observe(
        right["coordinate"], expected_head=original.head)


def test_stale_heads_modified_snapshots_and_substituted_parents_are_refused(tmp_path):
    nav = CounterfactualCausalGround(two_factory, LocalStatePages(tmp_path))
    _, p, _ = advance(nav, nav.root_coordinate, A, 1)
    _, q, _ = advance(nav, p["coordinate"], B, 2)
    snapshot = nav.snapshot()
    for forged in (
        {**deepcopy(snapshot), "nodes": list(reversed(deepcopy(snapshot["nodes"])))},
        {**deepcopy(snapshot), "nodes": [
            deepcopy(snapshot["nodes"][0]), {**deepcopy(snapshot["nodes"][1]),
                                            "parent": nav.root_coordinate}]},
    ):
        with pytest.raises(ValueError, match="unpinned|Modified"):
            CounterfactualCausalGround.restore(two_factory, nav.pages, forged,
                                               expected_digest=snapshot["snapshot_digest"])
    with pytest.raises(ValueError, match="unpinned|Modified"):
        CounterfactualCausalGround.restore(two_factory, nav.pages, snapshot,
                                           expected_digest="f"*64)
    with pytest.raises(ValueError, match="Stale"):
        nav.advance(q["coordinate"], C, token_id=3,
                    expected_head=nav.head, expected_parent_digest=p["node_digest"])
    with pytest.raises(ValueError, match="Stale"):
        nav.observe(p["coordinate"], expected_head="e"*64)
    assert nav.head == snapshot["forest_head"]


def test_corrupt_kv_page_refused_despite_original_snapshot_pin(tmp_path):
    pages = LocalStatePages(tmp_path)
    nav = CounterfactualCausalGround(hybrid_factory, pages)
    _, p, _ = advance(nav, nav.root_coordinate, A, 1)
    _, q, _ = advance(nav, p["coordinate"], B, 2)
    snapshot = nav.snapshot()
    kv = q["checkpoint"]["layer_checkpoints"][-1]["kv_pages"][0]
    (tmp_path / kv["sha256"]).write_bytes(b"adversarial")
    with pytest.raises((ValueError, OSError), match="Corrupt|corrupt|Invalid"):
        CounterfactualCausalGround.restore(hybrid_factory, pages, snapshot,
                                           expected_digest=snapshot["snapshot_digest"], strict=True)
    assert nav.observe(p["coordinate"], expected_head=nav.head)


def test_failed_storage_after_trained_execution_does_not_publish_partial_state(tmp_path, monkeypatch):
    pages = LocalStatePages(tmp_path)
    nav = CounterfactualCausalGround(two_factory, pages)
    _, parent, _ = advance(nav, nav.root_coordinate, A, 1)
    old_head, old_nodes = nav.head, deepcopy(nav.nodes)
    original = pages.put
    def broken(data):
        raise ValueError("disconnected page host")
    monkeypatch.setattr(pages, "put", broken)
    with pytest.raises(ValueError, match="disconnected"):
        advance(nav, parent["coordinate"], B, 2)
    assert nav.head == old_head and nav.nodes == old_nodes
    monkeypatch.setattr(pages, "put", original)
    assert nav.observe(parent["coordinate"], expected_head=old_head)
    assert advance(nav, parent["coordinate"], B, 2)[2] == "NEW_TRAINED_TRANSITION"


def test_complete_binary_frontier_shares_prefixes_exactly(tmp_path):
    nav = CounterfactualCausalGround(two_factory, LocalStatePages(tmp_path), max_nodes=100)
    current = [(nav.root_coordinate, [])]
    for depth in range(4):
        children = []
        for parent, history in current:
            for tag, hidden in ((0, B), (1, C)):
                _, node, status = advance(nav, parent, hidden, 100+depth*2+tag)
                assert status == "NEW_TRAINED_TRANSITION"
                children.append((node["coordinate"], history+[(hidden, 100+depth*2+tag)]))
        current = children
    assert len(current) == 16
    assert len(nav.nodes) == 30  # distinct tree edges, not 64 replay steps
    assert nav.executed_transitions == 30
    assert sum(len(history) for _, history in current) == 64
    leaf, history = current[-1]
    independent = two_factory()
    for hidden, token in history:
        expected, receipt = independent.step(hidden, token_id=token)
    assert nav.observe(leaf, expected_head=nav.head) == expected
    assert nav.nodes[leaf]["causal_head"] == receipt["chain_head"]


def test_rehashed_in_memory_node_cannot_outrun_externally_pinned_forest(tmp_path):
    from vli.counterfactual_navigation import NODE
    from vli.ground import _digest

    nav = CounterfactualCausalGround(two_factory, LocalStatePages(tmp_path))
    _, first, _ = advance(nav, nav.root_coordinate, A, 1)
    pinned = nav.head
    # Simulate an adversary with write access to node memory who tries to
    # replace a receipt and recompute *its own* local content hashes.
    stored = nav.nodes[first["coordinate"]]
    stored["receipt"]["layer_witnesses"][-1]["output_f64_digest"] = "0"*64
    stored["receipt_digest"] = _digest(NODE, stored["receipt"])
    stored["node_digest"] = _digest(
        NODE, {k:v for k,v in stored.items() if k != "node_digest"})
    with pytest.raises(ValueError, match="Committed causal forest"):
        nav.observe(first["coordinate"], expected_head=pinned)
    with pytest.raises(ValueError, match="Committed causal forest"):
        nav.snapshot()
    with pytest.raises(ValueError, match="Committed causal forest"):
        nav.advance(first["coordinate"], B, token_id=2,
                    expected_parent_digest=first["node_digest"],
                    expected_head=pinned)


def test_refuse_bad_input_or_excess_nodes(tmp_path):
    nav = CounterfactualCausalGround(two_factory, LocalStatePages(tmp_path), max_nodes=1)
    with pytest.raises(ValueError, match="identifier"):
        advance(nav, nav.root_coordinate, A, -1)
    with pytest.raises(ValueError, match="Unqualified"):
        advance(nav, nav.root_coordinate, [float("nan"), 0, 1], 1)
    with pytest.raises(ValueError, match="width"):
        advance(nav, nav.root_coordinate, [0.0], 1)
    _, p, _ = advance(nav, nav.root_coordinate, A, 1)
    with pytest.raises(ValueError, match="Maximum"):
        advance(nav, p["coordinate"], B, 2)
    assert len(nav.nodes) == 1
