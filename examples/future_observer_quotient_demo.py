"""NAV-003 measured bounded observer-future equivalence on qualified Qwen fixtures.

Real Qwen operator topology with synthetic F32 source tensors only.
The true 27B pretrained checkpoint, tokenizer and complete decoder are NOT run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter_ns

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tests"))
from experiments.model_fabric.complete_native_layers import LocalStatePages
from test_counterfactual_navigation import (
    two_factory,hybrid_factory,advance,A,B,C,D,
)
from vli.counterfactual_navigation import CounterfactualCausalGround
from vli.future_observer_quotient import (
    FiniteFutureObserverGround, observer_spec, _readout,
    EQUIVALENT, REFUTED,
)


def experiment(mode):
    if mode not in ("two_gdn","gdn_gdn_gdn_gqa"):
        raise ValueError("Unqualified original Qwen numerical geometry fixture")
    factory=two_factory if mode=="two_gdn" else hybrid_factory
    with tempfile.TemporaryDirectory(prefix="nav003-") as tmp:
        start=perf_counter_ns()
        ground=CounterfactualCausalGround(factory,LocalStatePages(tmp))
        left_input,left,_=advance(ground,ground.root_coordinate,A,1)
        right_input,right,_=advance(ground,ground.root_coordinate,D,2)
        startup_ns=perf_counter_ns()-start
        observer=observer_spec(3,[(2,0.0)],observer="horizon-two-coordinate-two-sign")
        alphabet=[("negative",3,B),("positive",4,C)]
        navigator=FiniteFutureObserverGround(ground)
        start=perf_counter_ns()
        certificate=navigator.prove(
            left["coordinate"],right["coordinate"],
            expected_left_digest=left["node_digest"],
            expected_right_digest=right["node_digest"],
            expected_head=ground.head,
            observer=observer,actions=alphabet,horizon=2)
        proof_ns=perf_counter_ns()-start
        assert certificate["verdict"]==EQUIVALENT,certificate["first_counterexample"]
        assert certificate["initial_exact_state_keys"][0]!=certificate["initial_exact_state_keys"][1]
        assert certificate["new_trained_transitions_for_evidence"]==12
        assert len(certificate["rows"])==6 and len(ground.nodes)==14
        head=ground.head
        count=ground.executed_transitions

        start=perf_counter_ns()
        checked=navigator.verify(
            certificate,expected_digest=certificate["certificate_digest"],
            expected_head=head,
            expected_left_digest=left["node_digest"],
            expected_right_digest=right["node_digest"])
        proof_verify_ns=perf_counter_ns()-start
        assert checked["witness_pairs_verified"]==6

        start=perf_counter_ns()
        navigation=navigator.navigate(certificate,
            [("negative",100,B),("positive",101,C)],
            anchor="right",expected_certificate_digest=certificate["certificate_digest"],
            expected_head=head,expected_anchor_digest=right["node_digest"])
        navigate_ns=perf_counter_ns()-start
        assert all(s["coordinate"] not in ground.nodes for s in navigation["states"])

        start=perf_counter_ns()
        observations=navigator.verify_navigation(certificate,navigation,
            expected_navigation_digest=navigation["navigation_digest"],
            expected_certificate_digest=certificate["certificate_digest"],
            expected_head=head,expected_anchor_digest=right["node_digest"])
        navigation_verify_ns=perf_counter_ns()-start
        assert ground.executed_transitions==count and ground.head==head

        # Independently run the queried hidden input sequence AFTER
        # the observer result has been formed, as a numerical negative control.
        start=perf_counter_ns()
        independent=factory()
        independent.step(D,token_id=2)
        output1,_=independent.step(B,token_id=100)
        output2,_=independent.step(C,token_id=101)
        independent_oracle_ns=perf_counter_ns()-start
        assert observations==[_readout(observer,output1),_readout(observer,output2)]

        # A different observer notices the difference. The original
        # exact state has not changed; the requested distinction has.
        start=perf_counter_ns()
        negative=navigator.prove(
            left["coordinate"],right["coordinate"],
            expected_left_digest=left["node_digest"],
            expected_right_digest=right["node_digest"],
            expected_head=ground.head,
            observer=observer_spec(3,[(0,0.0)],observer="coordinate-zero-sign-control"),
            actions=alphabet,horizon=2)
        negative_ns=perf_counter_ns()-start
        assert negative["verdict"]==REFUTED
        assert negative["first_counterexample"] is not None
        assert negative["new_trained_transitions_for_evidence"]==0
        assert ground.executed_transitions==count and ground.head==head

        return {
            "schema":"vl-i.nav003-measured-observer-future-evidence.v1",
            "source_scope":"synthetic_F32_original_Qwen_operator_topology_fixture",
            "trained_geometry":mode,
            "original_pretrained_weight_bytes_read":0,
            "original_token_embedding_identity_verified":False,
            "full_decoder_completed":False,
            "source_root":ground.source_root,
            "initial_neural_state_keys_different":True,
            "observer_A":"sign_of_full_output_coordinate_2",
            "observer_B":"sign_of_full_output_coordinate_0",
            "observer_A_verdict":certificate["verdict"],
            "observer_B_verdict":negative["verdict"],
            "observer_B_first_counterexample":negative["first_counterexample"]["path"],
            "complete_action_alphabet":["negative","positive"],
            "certified_future_horizon":2,
            "source_branch_count":2,
            "actual_source_trained_transitions":ground.executed_transitions,
            "trained_transitions_for_positive_certificate":certificate[
                "new_trained_transitions_for_evidence"],
            "trained_transitions_for_negative_control":negative[
                "new_trained_transitions_for_evidence"],
            "original_observation_pairs_compared":len(certificate["rows"]),
            "virtual_output_observations":observations,
            "virtual_coordinates_unexecuted":True,
            "new_trained_transitions_during_navigation":0,
            "independent_original_geometry_oracle_equal":True,
            "real_source_forest_head":head,
            "positive_certificate_digest":certificate["certificate_digest"],
            "virtual_navigation_digest":navigation["navigation_digest"],
            "wall_ms":{
                "source_startup_and_initial_two_states":round(startup_ns/1e6,5),
                "complete_future_trained_evidence_plus_admission":round(proof_ns/1e6,5),
                "independent_certificate_reverification":round(proof_verify_ns/1e6,5),
                "virtual_navigation_with_source_checks":round(navigate_ns/1e6,5),
                "independent_virtual_receipt_reverification":round(navigation_verify_ns/1e6,5),
                "independent_direct_two_step_causal_oracle_with_prefix":round(
                    independent_oracle_ns/1e6,5),
                "negative_observer_counterexample_check":round(negative_ns/1e6,5),
            }
        }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("two_gdn","gdn_gdn_gdn_gqa"),
                        default="gdn_gdn_gdn_gqa")
    parser.add_argument("--out")
    args=parser.parse_args()
    result=experiment(args.mode)
    if args.out:
        path=Path(args.out)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
