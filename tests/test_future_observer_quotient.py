"""NAV-003: exhaustive observer-relative equivalence and negative proof tests.

These use source-qualified Qwen operator-topology F32 fixtures, NOT
original 27B pretrained tokenizer embeddings or a complete decoder.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.model_fabric.complete_native_layers import LocalStatePages
from vli.counterfactual_navigation import CounterfactualCausalGround
from vli.future_observer_quotient import (
    FiniteFutureObserverGround, observer_spec,
    EQUIVALENT, REFUTED, _readout,
)
from vli.ground import _digest
from test_counterfactual_navigation import (
    two_factory,hybrid_factory,advance,A,B,C,D,
)


def experiment(tmp_path, factory, *, horizon=2,threshold=0.0,actions=None):
    ground=CounterfactualCausalGround(factory,LocalStatePages(tmp_path))
    _,left,_=advance(ground,ground.root_coordinate,A,1)
    _,right,_=advance(ground,ground.root_coordinate,D,2)
    assert left["coordinate"]!=right["coordinate"]
    observer=observer_spec(3,[(2,threshold)],observer="scalar-sign-reader")
    alphabet=actions or [("negative",3,B),("positive",4,C)]
    navigator=FiniteFutureObserverGround(ground)
    cert=navigator.prove(left["coordinate"],right["coordinate"],
         expected_left_digest=left["node_digest"],
         expected_right_digest=right["node_digest"],
         expected_head=ground.head,
         observer=observer,actions=alphabet,horizon=horizon)
    return ground,navigator,left,right,cert


@pytest.mark.parametrize("factory",[two_factory,hybrid_factory])
def test_distinct_complete_neural_states_are_equivalent_for_bounded_intention(
        tmp_path,factory):
    ground,navigator,left,right,cert=experiment(tmp_path,factory)
    assert cert["initial_exact_state_keys"][0]!=cert["initial_exact_state_keys"][1]
    assert cert["verdict"]==EQUIVALENT
    assert cert["complete_exhaustive_frontier"] is True
    assert cert["declared_frontier_pairs"]==6
    assert len(cert["rows"])==6
    assert all(r["equal"] for r in cert["rows"])
    assert cert["new_trained_transitions_for_evidence"]==12
    assert len(ground.nodes)==14
    assert all(r["left_coordinate"]!=r["right_coordinate"] for r in cert["rows"])
    head=ground.head
    checked=navigator.verify(cert,expected_digest=cert["certificate_digest"],
          expected_head=head,expected_left_digest=left["node_digest"],
          expected_right_digest=right["node_digest"])
    assert checked["witness_pairs_verified"]==6
    assert checked["counterexample"] is None
    native=ground.executed_transitions
    virtual=navigator.navigate(cert,[("negative",31,B),("positive",32,C)],
        anchor="right",expected_certificate_digest=cert["certificate_digest"],
        expected_head=head,expected_anchor_digest=right["node_digest"])
    assert virtual["states"][0]["coordinate"] not in ground.nodes
    assert virtual["states"][1]["coordinate"] not in ground.nodes
    assert virtual["states"][0]["entire_numerical_state_reconstructed"] is False
    observed=navigator.verify_navigation(cert,virtual,
        expected_navigation_digest=virtual["navigation_digest"],
        expected_certificate_digest=cert["certificate_digest"],
        expected_head=head,expected_anchor_digest=right["node_digest"])
    independent=factory()
    independent.step(D,token_id=2)
    out1,_=independent.step(B,token_id=31)
    out2,_=independent.step(C,token_id=32)
    assert observed==[_readout(cert["observer"],out1),
                      _readout(cert["observer"],out2)]
    assert ground.executed_transitions==native and ground.head==head


@pytest.mark.parametrize("factory",[two_factory,hybrid_factory])
def test_counterexample_refines_insufficient_observer_scope(tmp_path,factory):
    ground,navigator,left,right,coarse=experiment(tmp_path,factory)
    assert coarse["verdict"]==EQUIVALENT
    # Choose a more discriminating threshold from the original observed
    # numerical reference (post-hoc calibration is explicitly labeled).
    row=next(r for r in coarse["rows"]
        if ground._inspect(ground.nodes[r["left_coordinate"]],verify_checkpoint=True)!=
           ground._inspect(ground.nodes[r["right_coordinate"]],verify_checkpoint=True))
    one=ground._inspect(ground.nodes[row["left_coordinate"]],verify_checkpoint=True)
    two=ground._inspect(ground.nodes[row["right_coordinate"]],verify_checkpoint=True)
    channel=next(i for i,(a,b) in enumerate(zip(one,two)) if a!=b)
    threshold=max(one[channel],two[channel])
    sharp=observer_spec(3,[(channel,threshold)],observer="calibrated-counterexample")
    refined=navigator.prove(left["coordinate"],right["coordinate"],
        expected_left_digest=left["node_digest"],
        expected_right_digest=right["node_digest"],
        expected_head=ground.head,
        observer=sharp,actions=[("negative",3,B),("positive",4,C)],horizon=2)
    assert refined["verdict"]==REFUTED
    assert refined["complete_exhaustive_frontier"] is False
    assert refined["first_counterexample"] is not None
    assert refined["first_counterexample"]["left_observation"]!=refined["first_counterexample"]["right_observation"]
    assert all(r["equal"] for r in refined["rows"][:-1])
    validated=navigator.verify(refined,expected_digest=refined["certificate_digest"],
        expected_head=ground.head,expected_left_digest=left["node_digest"],
        expected_right_digest=right["node_digest"])
    assert validated["verdict"]==REFUTED
    with pytest.raises(ValueError,match="Refuted"):
        navigator.navigate(refined,[("negative",99,B)],anchor="left",
            expected_certificate_digest=refined["certificate_digest"],
            expected_head=ground.head,expected_anchor_digest=left["node_digest"])


def test_root_output_collision_is_not_a_future_equivalence_certificate(tmp_path):
    # Even if current coarse observations match, a sharper observer may
    # distinguish the first causal continuation. The verifier must find it.
    ground=CounterfactualCausalGround(two_factory,LocalStatePages(tmp_path))
    a,left,_=advance(ground,ground.root_coordinate,A,1)
    b,right,_=advance(ground,ground.root_coordinate,D,2)
    assert a!=b  # genuine different neural outputs after genesis
    from vli.future_observer_quotient import _readout
    assert _readout(observer_spec(3,[(0,0.0)]),a)==_readout(
        observer_spec(3,[(0,0.0)]),b)
    observed_a,anode,_=advance(ground,left["coordinate"],B,3)
    observed_b,bnode,_=advance(ground,right["coordinate"],B,3)
    assert observed_a!=observed_b
    channel=next(i for i,(u,v) in enumerate(zip(observed_a,observed_b)) if u!=v)
    selected=max(observed_a[channel],observed_b[channel])
    spec=observer_spec(3,[(channel,selected)])
    navigator=FiniteFutureObserverGround(ground)
    cert=navigator.prove(left["coordinate"],right["coordinate"],
        expected_left_digest=left["node_digest"],expected_right_digest=right["node_digest"],
        expected_head=ground.head,observer=spec,actions=[("negative",3,B)],horizon=1)
    assert cert["verdict"]==REFUTED
    assert cert["first_counterexample"]["path"]==["negative"]
    assert len(cert["rows"])==1


def test_finite_horizon_and_explicit_action_alphabet_are_enforced(tmp_path):
    ground,navigator,left,right,cert=experiment(tmp_path,two_factory,horizon=1)
    assert cert["verdict"]==EQUIVALENT
    head=ground.head
    with pytest.raises(ValueError,match="virtual future navigation request"):
        navigator.navigate(cert,[("negative",5,B),("positive",6,C)],
            anchor="left",expected_certificate_digest=cert["certificate_digest"],
            expected_head=head,expected_anchor_digest=left["node_digest"])
    with pytest.raises(ValueError,match="outside certified"):
        navigator.navigate(cert,[("negative",5,D)],
            anchor="left",expected_certificate_digest=cert["certificate_digest"],
            expected_head=head,expected_anchor_digest=left["node_digest"])
    with pytest.raises(ValueError,match="Unsupported"):
        navigator.navigate(cert,[("unseen",5,B)],
            anchor="left",expected_certificate_digest=cert["certificate_digest"],
            expected_head=head,expected_anchor_digest=left["node_digest"])
    with pytest.raises(ValueError,match="horizon"):
        navigator.prove(left["coordinate"],right["coordinate"],
            expected_left_digest=left["node_digest"],expected_right_digest=right["node_digest"],
            expected_head=head,observer=cert["observer"],
            actions=[("negative",3,B)],horizon=5)


def test_forged_certificate_and_stale_forest_are_refused(tmp_path):
    ground,navigator,left,right,cert=experiment(tmp_path,two_factory)
    head=ground.head
    modifications=[
        {**deepcopy(cert),"verdict":REFUTED},
        {**deepcopy(cert),"horizon":1},
        {**deepcopy(cert),"rows":deepcopy(cert["rows"][:-1])},
        {**deepcopy(cert),"observer":observer_spec(3,[(0,1000.0)])},
        {**deepcopy(cert),"initial_exact_state_keys":["0"*64,"1"*64]},
    ]
    for changed in modifications:
        with pytest.raises(ValueError,match="Unpinned|altered"):
            navigator.verify(changed,expected_digest=cert["certificate_digest"],
                expected_head=head,expected_left_digest=left["node_digest"],
                expected_right_digest=right["node_digest"])
    with pytest.raises(ValueError,match="Unpinned|stale"):
        navigator.verify(cert,expected_digest="f"*64,expected_head=head,
            expected_left_digest=left["node_digest"],
            expected_right_digest=right["node_digest"])
    receipt=navigator.navigate(cert,[("negative",15,B)],anchor="right",
         expected_certificate_digest=cert["certificate_digest"],
         expected_head=head,expected_anchor_digest=right["node_digest"])
    with pytest.raises(ValueError,match="Stale|stale"):
        navigator.navigate(cert,[("negative",15,B)],anchor="right",
            expected_certificate_digest=cert["certificate_digest"],
            expected_head=head,expected_anchor_digest=left["node_digest"])
    changed=deepcopy(receipt);changed["states"][0]["observation"]=[9]
    with pytest.raises(ValueError,match="Unpinned|altered"):
        navigator.verify_navigation(cert,changed,
            expected_navigation_digest=receipt["navigation_digest"],
            expected_certificate_digest=cert["certificate_digest"],
            expected_head=head,expected_anchor_digest=right["node_digest"])
    advance(ground,ground.root_coordinate,B,99)
    with pytest.raises(ValueError,match="Unpinned|stale"):
        navigator.verify(cert,expected_digest=cert["certificate_digest"],
            expected_head=head,expected_left_digest=left["node_digest"],
            expected_right_digest=right["node_digest"])


def test_corrupt_attention_pages_deny_qualified_future_reuse(tmp_path):
    ground,navigator,left,right,cert=experiment(tmp_path,hybrid_factory)
    head=ground.head
    receipt=navigator.navigate(cert,[("negative",42,B)],anchor="left",
        expected_certificate_digest=cert["certificate_digest"],
        expected_head=head,expected_anchor_digest=left["node_digest"])
    page=ground.nodes[cert["rows"][0]["right_coordinate"]]["checkpoint"]["layer_checkpoints"][-1]["kv_pages"][0]
    (tmp_path/page["sha256"]).write_bytes(b"malicious")
    with pytest.raises((ValueError,OSError),match="Corrupt|corrupt"):
        navigator.verify(cert,expected_digest=cert["certificate_digest"],
            expected_head=head,expected_left_digest=left["node_digest"],
            expected_right_digest=right["node_digest"])
    with pytest.raises((ValueError,OSError),match="Corrupt|corrupt"):
        navigator.verify_navigation(cert,receipt,
            expected_navigation_digest=receipt["navigation_digest"],
            expected_certificate_digest=cert["certificate_digest"],
            expected_head=head,expected_anchor_digest=left["node_digest"])


def test_cross_process_restore_verifies_categorical_future_proof(tmp_path):
    child=r'''
import json,sys,time
from pathlib import Path
from experiments.model_fabric.complete_native_layers import LocalStatePages
from vli.counterfactual_navigation import CounterfactualCausalGround
from vli.future_observer_quotient import FiniteFutureObserverGround,observer_spec
from test_counterfactual_navigation import hybrid_factory,advance,A,B,C,D

root=Path(sys.argv[2]);pages=LocalStatePages(root)
if sys.argv[1]=="writer":
    g=CounterfactualCausalGround(hybrid_factory,pages)
    _,a,_=advance(g,g.root_coordinate,A,1)
    _,b,_=advance(g,g.root_coordinate,D,2)
    nav=FiniteFutureObserverGround(g)
    cert=nav.prove(a["coordinate"],b["coordinate"],
        expected_left_digest=a["node_digest"],expected_right_digest=b["node_digest"],
        expected_head=g.head,observer=observer_spec(3,[(2,0.0)]),
        actions=[("negative",3,B),("positive",4,C)],horizon=2)
    assert cert["verdict"]=="CERTIFIED_FINITE_HORIZON"
    response=nav.navigate(cert,[("negative",100,B),("positive",101,C)],
        anchor="right",expected_certificate_digest=cert["certificate_digest"],
        expected_head=g.head,expected_anchor_digest=b["node_digest"])
    snap=g.snapshot()
    for name,item in (("snap",snap),("cert",cert),("prediction",response)):
        (root/(name+".json")).write_text(json.dumps(item),encoding="utf8")
    print(json.dumps({"snapshot":snap["snapshot_digest"],
         "certificate":cert["certificate_digest"],
         "navigation":response["navigation_digest"],
         "forest":g.head,"left":a["node_digest"],"right":b["node_digest"]}),flush=True)
    while True:time.sleep(1)
elif sys.argv[1]=="reader":
    pins=json.loads(sys.argv[3])
    load=lambda name:json.loads((root/(name+".json")).read_text(encoding="utf8"))
    snap,cert,pred=(load("snap"),load("cert"),load("prediction"))
    g=CounterfactualCausalGround.restore(hybrid_factory,pages,snap,
        expected_digest=pins["snapshot"])
    verifier=FiniteFutureObserverGround(g)
    detail=verifier.verify(cert,expected_digest=pins["certificate"],
        expected_head=pins["forest"],
        expected_left_digest=pins["left"],expected_right_digest=pins["right"])
    observations=verifier.verify_navigation(cert,pred,
        expected_navigation_digest=pins["navigation"],
        expected_certificate_digest=pins["certificate"],
        expected_head=pins["forest"],expected_anchor_digest=pins["right"])
    assert detail["verdict"]=="CERTIFIED_FINITE_HORIZON"
    assert pred["states"][-1]["coordinate"] not in g.nodes
    print(json.dumps({"observations":observations,"evidence":detail,
                      "actual_trained_nodes":len(g.nodes)}),flush=True)
'''
    env=os.environ.copy()
    env["PYTHONPATH"]=str(Path(__file__).resolve().parent)+os.pathsep+env.get("PYTHONPATH","")
    cmd=[sys.executable,"-u","-c",child]
    writer=subprocess.Popen(cmd+["writer",str(tmp_path)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,text=True)
    try:
        first=writer.stdout.readline().strip()
        assert first.startswith("{"), (
            first,writer.poll(),writer.stderr.read() if writer.poll() is not None else "")
        pins=json.loads(first)
    finally:
        writer.kill();writer.wait(timeout=15)
        writer.stdout.close();writer.stderr.close()
    assert writer.returncode!=0
    resumed=subprocess.run(cmd+["reader",str(tmp_path),json.dumps(pins)],
        capture_output=True,text=True,env=env,check=True,timeout=80)
    result=json.loads(resumed.stdout.strip())
    assert result["evidence"]["witness_pairs_verified"]==6
    assert result["actual_trained_nodes"]==14
    assert len(result["observations"])==2


@pytest.mark.parametrize("factory",[two_factory,hybrid_factory])
def test_independent_scalar_oracles_replay_all_certified_futures(tmp_path,factory):
    """Every certificate witness is checked against different numerical math.

    The GDN reference executes upstream original scalar qualified operators,
    not VL-I's fast NumPy QwenVectorizedCompleteLayer. The hybrid reference
    separately executes a hand-written scalar gated-attention/softmax oracle.
    """
    from experiments.model_fabric.complete_native_layers import QwenQualifiedLayer
    from test_qwen_layer_chain import two_layer_fixture,executor
    from test_qwen_hybrid_block import (
        nav_hybrid,runners,reference_attention,
    )
    ground,navigator,left,right,cert=experiment(tmp_path,factory)
    assert cert["verdict"]==EQUIVALENT
    action_vectors={"negative":B,"positive":C}
    def scalar_reference(prefix, names):
        if factory is two_factory:
            native=executor(two_layer_fixture(),QwenQualifiedLayer)
        else:
            native=runners(nav_hybrid(),vectorized=False)[:3]
        timeline=[]
        output=None
        for hidden in [prefix]+[action_vectors[x] for x in names]:
            output=list(hidden)
            for layer in native:
                output,_=layer.next(output)
            if factory is hybrid_factory:
                output,timeline=reference_attention(output,timeline,len(timeline))
        return output
    for row in cert["rows"]:
        for parent, side in ((A,"left"),(D,"right")):
            native=scalar_reference(parent,row["path"])
            retained=ground._inspect(
                ground.nodes[row[side+"_coordinate"]],verify_checkpoint=True)
            assert retained==pytest.approx(native,abs=2e-10)
            assert _readout(cert["observer"],native)==row[side+"_observation"]
    assert len(cert["rows"])==6


def test_self_rehashed_false_certificate_fails_independent_evidence_replay(tmp_path):
    """A fresh digest is not mathematical authority, even if a caller trusts it."""
    from vli.future_observer_quotient import CERTIFICATE, EQUIVALENT
    ground,navigator,left,right,valid=experiment(tmp_path,two_factory)
    assert valid["verdict"]==EQUIVALENT
    attempts=[]
    reduced=deepcopy(valid)
    reduced["rows"].pop()
    attempts.append(reduced)
    altered=deepcopy(valid)
    altered["rows"][0]["left_observation"]=[9]
    attempts.append(altered)
    changed_observer=deepcopy(valid)
    changed_observer["observer"]=observer_spec(3,[(0,1e12)])
    attempts.append(changed_observer)
    for forged in attempts:
        forged["certificate_digest"]=_digest(CERTIFICATE,{
            key:value for key,value in forged.items() if key!="certificate_digest"})
        with pytest.raises(ValueError,match="Missing|Counterexample|forged|Altered|contradictory"):
            navigator.verify(forged,expected_digest=forged["certificate_digest"],
                expected_head=ground.head,expected_left_digest=left["node_digest"],
                expected_right_digest=right["node_digest"])


def test_invalid_observer_mutation_and_oversized_frontier_refused(tmp_path):
    from vli.future_observer_quotient import action_alphabet
    for channels in ([(0,float("nan"))],[(0,0.0),(0,1.0)],[(1000,0.0)]):
        with pytest.raises(ValueError,match="Observer|observer"):
            observer_spec(3,channels)
    with pytest.raises(ValueError,match="Duplicate"):
        action_alphabet(3,[("same",3,B),("same",4,C)])
    ground=CounterfactualCausalGround(two_factory,LocalStatePages(tmp_path),max_nodes=8)
    _,a,_=advance(ground,ground.root_coordinate,A,1)
    _,b,_=advance(ground,ground.root_coordinate,D,2)
    nav=FiniteFutureObserverGround(ground)
    with pytest.raises(ValueError,match="Insufficient"):
        nav.prove(a["coordinate"],b["coordinate"],
             expected_left_digest=a["node_digest"],expected_right_digest=b["node_digest"],
             expected_head=ground.head,observer=observer_spec(3,[(0,0.0)]),
             actions=[("negative",3,B),("positive",4,C)],horizon=3)
