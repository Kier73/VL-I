"""SHAPE-002 live benchmark: two intentions, three neural shapes, genuine new code/tokens.

The source is a complete 12-state machine description, not a pretrained model
and not a biological neural simulator. VLD proposes the observer quotient;
all original Mealy outputs and actions are verified *before* it is compiled
to a new, separately sourced ReLU neural network.

Report cold preparation separately from qualified warm query and rendering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from vli.morphological_generation import (
    MorphologySourceGround, MorphologySession, select_qualified_morphology,
)
from vli.morphological_intent import MorphologyIntentionRouter
from vli.intent_topology import TopologyWitness


def run():
    source_path=Path(__file__).resolve().parents[1]/"experiments/shape002/morphology_model.json"
    raw=source_path.read_bytes()
    t0=perf_counter_ns()
    source=MorphologySourceGround(raw)
    t1=perf_counter_ns()
    shapes={
        "phase":source.observer(("phase",)),
        "micro":source.observer(("micro",)),
        "full":source.observer(("phase","micro")),
    }
    measurements={}
    for name,shape in shapes.items():
        measurements[name]={
            "qualified_channel_scope":list(shape.channels),
            "source_states":len(source.system.states),
            "realized_states":shape.manifest["qualified_state_dimension"],
            "generated_hidden_neurons":shape.manifest["generated_hidden_units"],
            "model_ground_digest":shape.generated_model.manifest["ground_digest"],
            "neural_weights_digest":shape.generated_model.weights_digest,
            "exact_source_cells_checked":shape.proof["checked_original_cells"],
            "proof_digest":shape.proof["proof_digest"],
            "preparation":shape.timings_ns,
        }
    cold_phase=select_qualified_morphology(
        source,tuple(shapes.values()),requested_channels=("phase",),
        expected_queries=8,warm_sample_count=256)
    warm_phase=select_qualified_morphology(
        source,tuple(shapes.values()),requested_channels=("phase",),
        expected_queries=10000,warm_sample_count=256)
    micro_view=select_qualified_morphology(
        source,tuple(shapes.values()),requested_channels=("micro",),
        expected_queries=1000,warm_sample_count=256)
    session=MorphologySession(source)
    words=[
        ("phase","advance","Only tell me the observable phase"),
        ("micro","inspect","Now tell me the independent microstate"),
        ("full","advance","Now reveal the entire computable state"),
        ("phase","reset","Return to the initial phase of the described Ground"),
    ]
    rendered=[]
    for name,action,prompt in words:
        result=session.step(shapes[name],action=action,prompt=prompt)
        out=result["observation"]
        rendered.append({
            "position":out["coordinate"],"view":name,
            "ground_identity":out["morphology_ground_digest"],
            "source_state":result["turn"]["source_state"],
            "next_full_source_state":result["turn"]["next_full_source_state"],
            "observable_values":out["observed_channel_values"],
            "generated_views":[v["kind"] for v in out["generated_views"]],
            "new_generated_text_bytes":len(out["generated_views"][0]["content"].encode("utf-8")),
            "render_ns":out["generate_and_render_ns"],
            "generated_neural_steps":out["generated_neural_steps"],
        })
    # A separate conversation uses E-CPU's source-qualified witness
    # prototypes to route the external prompt to a topology, without an
    # observer manually specifying which channel to use.
    proposal_session=MorphologySession(source)
    router=MorphologyIntentionRouter(source)
    witnesses=(
        TopologyWitness("report only the phase observation",("phase",)),
        TopologyWitness("show the current phase state",("phase",)),
        TopologyWitness("report only the micro observation",("micro",)),
        TopologyWitness("show the current micro state",("micro",)),
        TopologyWitness("report both phase and micro observations",("phase","micro")),
        TopologyWitness("show the complete state with both channels",("phase","micro")),
    )
    proposed=router.propose(
        proposal_session,witnesses,"report only the phase observation",
        min_confidence=0.35,min_margin=0.001)
    admitted=router.accept(
        proposal_session,witnesses,proposed,
        expected_proposal_head=proposed["proposal_digest"])
    initial_response=proposal_session.step(
        source.observer(admitted),action="advance",
        prompt="report only the phase observation")
    refined=router.propose(
        proposal_session,witnesses,"report both phase and micro observations",
        min_confidence=0.35,min_margin=0.001)
    refined_channels=router.accept(
        proposal_session,witnesses,refined,
        expected_proposal_head=refined["proposal_digest"])
    completed_response=proposal_session.step(
        source.observer(refined_channels),action="inspect",
        prompt="report both phase and micro observations")
    saved=session.checkpoint()
    t2=perf_counter_ns()
    recovered=MorphologySession.restore(source,tuple(shapes.values()),saved,
                                        expected_checkpoint_head=saved["checkpoint_digest"])
    t3=perf_counter_ns()
    assert saved==recovered.checkpoint()
    assert recovered.state==session.state and recovered.head==session.head
    return {
        "schema":"vl-i.shape002-cost-qualified-generative-morphology.v1",
        "source":source.manifest,
        "source_admission_ns":t1-t0,
        "shape_models":measurements,
        "cold_phase_intention_cost_selection":cold_phase,
        "repeated_phase_intention_cost_selection":warm_phase,
        "micro_intention_cost_selection":micro_view,
        "different_intention_refuses_phase_only_for_micro":
            micro_view["options"][0]["status"]=="REFUSED_INSUFFICIENT_OBSERVER_SCOPE",
        "prompt_driven_shape_revision": {
            "initial_prompt_proposal_head":proposed["proposal_digest"],
            "initial_evidence_tier":proposed["ecpu_intention_proposal"]["ecpu_prediction"]["evidence_tier"],
            "first_observer_channels":list(admitted),
            "first_observer_states":shapes["phase"].manifest["qualified_state_dimension"],
            "first_response_state":initial_response["observation"]["coordinate"],
            "next_prompt_proposal_head":refined["proposal_digest"],
            "next_observer_channels":list(refined_channels),
            "next_observer_states":shapes["full"].manifest["qualified_state_dimension"],
            "next_observer_observed_values":completed_response["observation"]["observed_channel_values"],
            "next_response_state":completed_response["observation"]["coordinate"],
            "source_state_retained_across_morphology_change":proposal_session.state,
        },
        "turns":rendered,
        "all_turn_coordinates_unique":len({r["position"] for r in rendered})==len(rendered),
        "checkpoint_digest":saved["checkpoint_digest"],
        "full_source_state_after_context_changes":recovered.state,
        "replay_verification_ns":t3-t2,
        "not_claimed":[
            "comparable language-model intelligence or pretrained LLM speedup",
            "biological neural firing reproduction",
            "zero source or proof computation",
            "automatic natural-language comprehension",
            "irrevocable erasure of hidden source state",
            "guaranteed host-dependent runtime advantage for all workloads",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="artifacts/shape002-morphology.json")
    opts=p.parse_args(argv)
    result=run()
    target=Path(opts.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VL_I_SHAPE002 "+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
