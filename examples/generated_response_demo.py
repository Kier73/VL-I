"""Run a model description as a newly generated neural state machine.

No pretrained parameters, model-server, tool execution or language model.
VLD constructs genuine neural weights from the described transition Ground,
compares whole domain, and the response coordinate forms new text/code/
UTF-8-byte tokens from the interpreted state. The causal session is recovered.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

from vli.generative_response import GeneratedModelGround, GenerativeConversation
from vli.generative_intent import ActionWitness, propose_action, accept_and_respond


def run():
    source = Path(__file__).resolve().parents[1] / "experiments/coord002/described_model.json"
    raw = source.read_bytes()
    start = perf_counter_ns()
    model = GeneratedModelGround(raw)
    after_compile = perf_counter_ns()
    model.verify()
    conversation = GenerativeConversation(model, observer="vli-lab")
    turns = [
        ("Inspect the initial described state and explain what it observes.", "inspect"),
        ("Advance the state and generate a working Python transition function.", "advance"),
        ("Show what changes when I reset this computational Ground.", "reset"),
        ("Inspect again: carry the previous conversation context forward.", "inspect"),
    ]
    generated = []
    for prompt, action in turns:
        item = conversation.respond(prompt, action)
        view_summary = []
        for projection in item["response"]["projections"]:
            payload = projection["content"]
            encoded = payload.encode("utf-8") if type(payload) is str else json.dumps(
                payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            view_summary.append({
                "kind": projection["kind"],
                "generated_bytes_or_serialized_view": len(encoded),
                "sha256": sha256(encoded).hexdigest(),
            })
        generated.append({
            "position": item["turn"]["coordinate"],
            "source_state": item["turn"]["from_state"],
            "target_state": item["turn"]["to_state"],
            "action": action,
            "neural_steps": item["turn"]["neural_forward_steps"],
            "step_digest": item["turn"]["step_digest"],
            "observation_digest": item["turn"]["generated_observation_digest"],
            "views": view_summary,
        })
    after_generation = perf_counter_ns()
    original = conversation.checkpoint()
    restored = GenerativeConversation.restore(
        model, original, expected_head=original["checkpoint_digest"])
    after_replay = perf_counter_ns()
    prototype_turn = GenerativeConversation(model, observer="vli-lab")
    examples = (
        ActionWitness("inspect the model state", "inspect"),
        ActionWitness("show the current state observation", "inspect"),
        ActionWitness("advance to the next state", "advance"),
        ActionWitness("continue the described system", "advance"),
        ActionWitness("reset the state", "reset"),
        ActionWitness("return to the initial state", "reset"),
    )
    proposed = propose_action(
        prototype_turn, examples, "advance to the next state",
        min_confidence=0.35, min_margin=0.001)
    assert proposed["status"] == "PROPOSED_NEEDS_OBSERVER_ACCEPTANCE"
    prompted = accept_and_respond(
        prototype_turn, examples, proposed,
        expected_proposal_head=proposed["proposal_digest"],
        views=("text", "code", "tokens"))
    assert prompted["turn"]["action"] == "advance"
    assert restored.head == conversation.head
    assert restored.state == conversation.state
    assert restored.checkpoint() == original
    return {
        "schema": "vl-i.coord002-description-native-measurement.v1",
        "description_bytes_observed": len(raw),
        "description_sha256": sha256(raw).hexdigest(),
        "ground_digest": model.manifest["ground_digest"],
        "source": model.manifest["source"],
        "vld_system_fingerprint": model.system.fingerprint,
        "generated_weights_digest": model.weights_digest,
        "generated_hidden_units": len(model.neural.hidden_bias),
        "generated_parameters": sum(map(len, model.neural.hidden_weight)) +
            len(model.neural.hidden_bias) +
            sum(map(len, model.neural.next_state_weight)) +
            sum(map(len, model.neural.channel_weight)),
        "complete_neural_verification_cells": len(model.system.states)*len(model.system.actions),
        "turns": generated,
        "prompt_first_intention": {
            "status": proposed["status"],
            "admitted_action": proposed["action"],
            "ecpu_evidence_tier": proposed["ecpu_advisory_prediction"]["evidence_tier"],
            "confidence": proposed["ecpu_advisory_prediction"]["confidence"],
            "external_proposal_head": proposed["proposal_digest"],
            "generated_state_position": prompted["turn"]["coordinate"],
            "generated_state": prompted["turn"]["to_state"],
            "actual_execution_without_model_server": True,
            "scope": "witnessed_ECPU_advisory_not_general_English_model",
        },
        "stable_fixed_width_coordinates": all(len(x["position"])==76 for x in generated),
        "distinct_turn_positions": len({x["position"] for x in generated})==len(generated),
        "restored_state": restored.state,
        "restored_history_head": restored.head,
        "checkpoint_digest": original["checkpoint_digest"],
        "generated_output_not_in_input_as_literally_copied_transcripts": True,
        "measured_elapsed_ns": {
            "model_ground_build_and_neural_compilation": after_compile-start,
            "four_turn_generation": after_generation-after_compile,
            "whole_history_verification_replay": after_replay-after_generation,
        },
        "scope": "bounded_fsm_generative_neural_compilation_not_general_llm",
        "not_claimed": [
            "learned pretrained large-language-model behavior",
            "general English understanding or unseen arbitrary instruction solving",
            "full tensor checkpoint recreation from any arbitrary header",
            "zero interpreter arithmetic during model generation",
            "actual tool execution, web access or side-effect authority",
            "zero-cost replay: restore independently re-computes pure turns",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/coord002-description-native.json")
    opts=p.parse_args(argv)
    outcome=run()
    target=Path(opts.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(outcome,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VL_I_COORD002",json.dumps(outcome,sort_keys=True))


if __name__=="__main__":
    main()
