"""SHAPE-003 benchmark: partial black-box observations build a generative neural Ground.

Learner: IFS COG-008 membership ledger and COG-007 observation table.
Reference authority: independent VLD finite Mealy source, unavailable to the
learner. Realizer: VLD generated neural model + VL-I response-state renderer.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

from vli.learned_morphology import (
    LearnedObserverGround, LearnedMorphologyConversation, LearnedObserverIntentionRouter
)
from vli.intent_topology import TopologyWitness


def run():
    raw=(Path(__file__).resolve().parents[1]/
         "experiments/shape002/morphology_model.json").read_bytes()
    source_digest=sha256(raw).hexdigest()
    groups={}
    t0=perf_counter_ns()
    for name,channels in (("phase",("phase",)),("micro",("micro",)),
                          ("full",("phase","micro"))):
        model=LearnedObserverGround(raw,channels)
        groups[name]=model
    t1=perf_counter_ns()
    snapshots={}
    for name,g in groups.items():
        proof=g.certificate
        snapshots[name]={
            "channels":list(g.channels),
            "generated_model_digest":g.neural_ground.manifest["ground_digest"],
            "generated_weights_digest":g.neural_ground.weights_digest,
            "qualified_neural_states":len(g.machine.representatives),
            "hidden_units":len(g.neural_ground.neural.hidden_bias),
            "revision_rounds":len(g.revision_history),
            "counterexample_traces":[row["counterexample"]
                                     for row in g.revision_history
                                     if row["counterexample"] is not None],
            "ifs_black_box_membership_calls":g.oracle.external_calls,
            "ifs_actual_source_transition_steps_for_membership":g.source.direct_transition_steps,
            "independent_full_reference_edges_across_revisions":
                sum(round_info["checked_full_source_edges"]
                    for round_info in g.revision_history),
            "ifs_ledger_head":g.oracle.head,
            "ifs_ledger_digest":g.ledger["digest"],
            "ifs_checkpoint_digest":g.ifs_checkpoint["digest"],
            "independent_reference_edges_final_check":g.verdict["checked_product_edges"],
            "full_source_state_edges_for_reference":len(g.reference.states)*len(g.reference.actions),
            "proof_head":proof["proof_digest"],
            "learn_compile_ns":g.learning_ns,
        }
    session=LearnedMorphologyConversation((
        groups["phase"],groups["micro"],groups["full"]))
    user_interactions=(
        ("phase","advance","Show only the phase"),
        ("micro","inspect","Now reveal the microstate"),
        ("full","advance","Please show both kinds of state"),
        ("phase","reset","Reset the session; show just the phase"),
    )
    outputs=[]
    for key,action,prompt in user_interactions:
        resp=session.respond(groups[key],prompt=prompt,action=action)
        e=resp["event"]
        rendered=resp["response"]["projections"]
        outputs.append({
            "observer_channels":list(groups[key].channels),
            "coordinate":e["position"],"action":action,
            "causal_head":e["event_digest"],
            "observed_values":e["observed_values"],
            "new_text_sha256":sha256(rendered[0]["content"].encode()).hexdigest(),
            "new_generated_program_sha256":sha256(rendered[1]["content"].encode()).hexdigest(),
            "token_count":len(rendered[2]["content"]),
            "generated_neural_steps":resp["response"]["neural_steps_executed_this_formation"],
        })
    witnessed=(
        TopologyWitness("report only the phase observation",("phase",)),
        TopologyWitness("show the current phase state",("phase",)),
        TopologyWitness("report only the micro observation",("micro",)),
        TopologyWitness("show the current micro state",("micro",)),
        TopologyWitness("report both phase and micro observations",("phase","micro")),
        TopologyWitness("show the complete state with both channels",("phase","micro")),
    )
    observer_session=LearnedMorphologyConversation(
        (groups["phase"],groups["micro"],groups["full"]))
    router=LearnedObserverIntentionRouter(observer_session)
    requested=router.propose(
        witnessed,"report only the phase observation")
    narrowed=router.accept_and_respond(
        witnessed,requested,
        expected_proposal_head=requested["proposal_digest"],action="advance",
        views=("text","tokens"))
    revised=router.propose(
        witnessed,"report both phase and micro observations")
    widened=router.accept_and_respond(
        witnessed,revised,
        expected_proposal_head=revised["proposal_digest"],action="inspect",
        views=("text","code","tokens"))
    original=session.checkpoint()
    replay_start=perf_counter_ns()
    restored=LearnedMorphologyConversation.restore(
        (groups["phase"],groups["micro"],groups["full"]),
        original,expected_head=original["checkpoint_digest"])
    replay_ns=perf_counter_ns()-replay_start
    assert restored.checkpoint()==original
    assert restored.history==session.history
    return {
        "schema":"vl-i.shape003-observed-counterexample-learning.v1",
        "source_sha256":source_digest,
        "reference_description_bytes":len(raw),
        "source_entirely_hidden_from_ifs_candidate_constructor":True,
        "learner":"IFS_COG008_observation_only_membership_with_COG007_table",
        "independent_certifier":"VLD_finite_reference_whole_reachable_product",
        "generated_neural_realizer":"VLD_compile_fsm_to_neural_via_VLI_COORD002",
        "observer_models":snapshots,
        "model_acquisition_and_compilation_ns":t1-t0,
        "interactions":outputs,
        "prompt_to_learned_shape": {
            "first_intention_status":requested["status"],
            "first_intention_evidence_tier":
                requested["ecpu_intention"]["ecpu_prediction"]["evidence_tier"],
            "first_learned_channels":narrowed["event"]["observer_channels"],
            "first_learned_values":narrowed["event"]["observed_values"],
            "first_response_position":narrowed["event"]["position"],
            "followup_learned_channels":widened["event"]["observer_channels"],
            "followup_observed_values":widened["event"]["observed_values"],
            "followup_response_position":widened["event"]["position"],
            "causal_symbolic_history":list(observer_session.history),
            "status":"source_bound_E0_intention_plus_exact_independent_model_certificate",
        },
        "causal_history_actions":list(session.history),
        "restored_checkpoint_digest":original["checkpoint_digest"],
        "source_free_learning_ledger_recovery_available":True,
        "replay_verification_ns":replay_ns,
        "not_claimed":[
            "general open-domain language intelligence",
            "recovery of pretrained parameters from an unknown checkpoint header",
            "learned behavior outside the declared action alphabet and finite source",
            "global certification from partial membership witnesses alone",
            "zero cost to probe, certify, compile, render or maintain history",
            "real biological neural network simulation",
        ],
    }


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="artifacts/shape003-learned-morphology.json")
    opts=ap.parse_args(argv)
    result=run()
    output=Path(opts.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VL_I_SHAPE003 "+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
