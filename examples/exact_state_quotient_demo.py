"""NAV-002: measured exact-Qwen-topology fixture counterfactual experiment.

This fixture uses synthetic F32 tensors in original Qwen operator geometry.
It does NOT access the original 27B snapshot or demonstrate text generation.
Run under the pinned VL-64 installation used by NAV-002 CI.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter_ns

# This experiment intentionally imports the separate qualified numerical
# fixture generators; none are original Qwen pretrained weight bytes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from experiments.model_fabric.complete_native_layers import LocalStatePages
from test_counterfactual_navigation import (
    two_factory, hybrid_factory, advance, A, B, C, D,
)
from vli.counterfactual_navigation import CounterfactualCausalGround
from vli.exact_state_quotient import ExactCausalStateQuotient


def experiment(mode):
    if mode not in ("two_gdn", "gdn_gdn_gdn_gqa"):
        raise ValueError("Unqualified Qwen model fixture geometry")
    factory = two_factory if mode == "two_gdn" else hybrid_factory
    vectors = (A, B, C, D)
    with tempfile.TemporaryDirectory(prefix="vli-nav002-") as root:
        t = perf_counter_ns()
        ground = CounterfactualCausalGround(factory, LocalStatePages(root))
        position = ground.root_coordinate
        original = []
        for depth, vector in enumerate(vectors):
            output, node, status = advance(ground, position, vector, 10+depth)
            assert status == "NEW_TRAINED_TRANSITION"
            original.append(output)
            position = node["coordinate"]
        trained_ns = perf_counter_ns() - t
        assert ground.executed_transitions == 4
        pinned_forest = ground.head

        t = perf_counter_ns()
        quotient = ExactCausalStateQuotient(ground, expected_head=pinned_forest)
        qualification_ns = perf_counter_ns() - t

        t = perf_counter_ns()
        predictions = []
        for leaf in range(16):
            labels = [10+2*depth+(leaf >> depth & 1)
                      for depth in range(4)]
            path = list(zip(labels,vectors))
            proof, observations = quotient.navigate(
                ground.root_coordinate,path,
                expected_parent_digest=ground.root_digest,
                expected_head=pinned_forest)
            assert [x["numerical_output"] for x in observations] == original
            predictions.append((leaf,labels,proof,observations))
        navigate_ns = perf_counter_ns() - t

        t = perf_counter_ns()
        verified = []
        for leaf,labels,proof,observations in predictions:
            assert quotient.verify(
                proof, expected_digest=proof["proof_digest"],
                expected_parent_digest=ground.root_digest,
                expected_head=pinned_forest) == observations
            verified.append(sha256(
                json.dumps(observations[-1]["numerical_output"],
                    separators=(",",":"),allow_nan=False).encode("utf8")).hexdigest())
        verification_ns = perf_counter_ns() - t

        # The independent oracle executes each distinct label history from
        # genesis. It is intentionally NOT the same worker as the proof path.
        t = perf_counter_ns()
        total_reference_transitions = 0
        for leaf,labels,proof,observations in predictions:
            independent = factory()
            for depth,(token,vector) in enumerate(zip(labels,vectors)):
                out,_ = independent.step(vector, token_id=token)
                assert out == observations[depth]["numerical_output"]
                total_reference_transitions += 1
        independent_replay_ns = perf_counter_ns() - t
        assert total_reference_transitions == 64
        assert ground.executed_transitions == 4 and ground.head == pinned_forest
        assert len(set(p["steps"][-1]["target_coordinate"]
                       for _,_,p,_ in predictions)) == 16
        assert len(set(verified)) == 1   # identical numerical inputs at every step
        return {
            "schema":"vl-i.exact-markov-quotient-fixture-evidence.v1",
            "status":"VERIFIED_EXACT_NUMERICAL_MARKOV_FUTURES",
            "model_source":"synthetic_F32_qualified_original_operator_topology_fixture",
            "original_pretrained_27b_weight_bytes_read":0,
            "trained_geometry":mode,
            "model_root":ground.source_root,
            "forest_head":ground.head,
            "operator_fingerprint":quotient.fingerprint,
            "original_trained_transitions":ground.executed_transitions,
            "full_history_reference_transitions":total_reference_transitions,
            "virtual_histories":len(predictions),
            "distinct_virtual_leaf_coordinates":16,
            "distinct_leaf_numerical_outputs":len(set(verified)),
            "same_F64_input_vectors_for_every_virtual_leaf":True,
            "token_ID_labels_not_verified_pretrained_embeddings":True,
            "quotient":quotient.metrics(),
            "timing_ns":{
                "initial_actual_trained_transitions":trained_ns,
                "full_source_state_qualification":qualification_ns,
                "all_virtual_navigations":navigate_ns,
                "all_proof_reverification":verification_ns,
                "total_amortization_including_acquisition":(
                    trained_ns+qualification_ns+navigate_ns+verification_ns),
                "independent_full_history_reference_replay":independent_replay_ns,
            },
            "numerical_oracle_matches_all_64_reference_transitions":True,
            "full_model_decoder_completed":False,
        }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("two_gdn","gdn_gdn_gdn_gqa"),
                   default="gdn_gdn_gdn_gqa")
    p.add_argument("--out",default=None)
    args=p.parse_args()
    result=experiment(args.mode)
    if args.out:
        out=Path(args.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
