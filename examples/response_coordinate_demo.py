"""Exercise an actual intent-relative response coordinate over real repository bytes.

This uses authentic E-CPU and VL-I source docs as *supplied content*. The
response-state coordinate selects and progressively renders the requested
page/code/tool-proposal without running a language model, calling a tool, or
claiming to generate unseen trained-model tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from vli.response_state import (
    SourceSlot, ResponseSourceGround, ResponseFormation, compile_intention
)
from vli.intent_topology import (
    TopologyWitness, propose_response_topology, admit_response_topology
)


def run():
    repo = Path(__file__).resolve().parents[1]
    page = (repo / "upstream/ecpu/docs/COMPUTATIONAL_STATE_SPACE_ARCHITECTURE.md").read_bytes()
    code = (repo / "src/vli/interpreter_ground.py").read_bytes()
    proposal = json.dumps({
        "tool": "inspect_state", "arguments": {"scope": "read-only", "source": "E-CPU"}
    }, sort_keys=True).encode("utf-8")
    # The content Ground is not a trained text generator; it retains real
    # supplied source bytes and a typed proposal, with explicit dependencies.
    ground = ResponseSourceGround((
        SourceSlot("page", "page", page),
        SourceSlot("code", "code", code, dependencies=("page",)),
        SourceSlot("tool_proposal", "tool_call", proposal, dependencies=("code",)),
    ))
    intent = compile_intention(
        "Show the E-CPU state-space page and VL-I interpreter code, then form a "
        "read-only tool proposal as part of the same observable response.",
        slots=("page","code","tool_proposal"), observer="vl-i-observer",
        history=({"role":"user","content":"A coordinate is a position of a rendered state."},),
    )
    start = perf_counter_ns()
    formed = ResponseFormation(ground, intent)
    initial = formed.observe()
    frames = [initial]
    # One causal frontier per observational stage; exact same position.
    formed.advance(max_nodes=1)
    frames.append(formed.observe())
    formed.advance(max_nodes=1)
    frames.append(formed.observe())
    checkpoint = formed.checkpoint()
    restarted = ResponseFormation(
        ground,intent, checkpoint=checkpoint,
        expected_head=checkpoint["checkpoint_digest"])
    while not restarted.dag.is_complete:
        restarted.advance(max_nodes=1)
        frames.append(restarted.observe())
    final=frames[-1]
    duration = perf_counter_ns()-start
    witnesses = (
        TopologyWitness("show python source code", ("code",)),
        TopologyWitness("display the interpreter source code", ("code",)),
        TopologyWitness("render the markdown page", ("page",)),
        TopologyWitness("display the documentation page", ("page",)),
        TopologyWitness("prepare a read-only tool proposal", ("tool_proposal",)),
        TopologyWitness("show the tool proposal", ("tool_proposal",)),
    )
    suggestion = propose_response_topology(
        ground, witnesses, "display the interpreter source code",
        min_confidence=0.4, min_margin=0.01)
    proposed_state = None
    if suggestion["status"] == "PROPOSED_NEEDS_OBSERVER_APPROVAL":
        accepted_intent = admit_response_topology(
            ground, witnesses, suggestion,
            expected_proposal_head=suggestion["proposal_digest"],
            observer="qualified-proposal-observer")
        proposed_state = ResponseFormation(ground, accepted_intent).position
    return {
        "schema": "vl-i.response-coordinate-experiment.v1",
        "coordinate": final["coordinate"],
        "coordinate_root": final["coordinate_root"],
        "source_ground_digest": ground.manifest["ground_digest"],
        "intention_digest": intent["intent_digest"],
        "state": final["state"],
        "view_kinds": [p["kind"] for p in final["projections"]],
        "rendered_source_sha256": {p["slot"]:p["source_sha256"] for p in final["projections"]},
        "selected_source_bytes": sum(len(ground.slots[n].content)
                                     for n in intent["selected_slots"]),
        "coordinate_string_length": len(final["coordinate"]),
        "steps": len(frames)-1,
        "formation_statuses": [f["state"] for f in frames],
        "observed_slot_counts": [len(f["observed_slots"]) for f in frames],
        "identical_position_across_formation": len({f["coordinate"] for f in frames})==1,
        "restored_checkpoint_head": checkpoint["checkpoint_digest"],
        "final_dag_head": final["dag_state_digest"],
        "inert_tool_call_only": final["projections"][-1]["status"]=="INERT_PROPOSAL",
        "wall_time_ns_including_snapshot_replay": duration,
        "ecpu_intention_topology_advisory": {
            "status": suggestion["status"],
            "proposed_slots": suggestion["proposed_slots"],
            "confidence": suggestion["ecpu_prediction"]["confidence"],
            "proposal_head": suggestion["proposal_digest"],
            "accepted_coordinate": (proposed_state["coordinate"]
                                    if proposed_state is not None else None),
            "scope": "caller_pinned_advisory_not_general_prompt_interpretation",
        },
        "not_claimed": [
            "general natural language intention interpretation",
            "new free-text content generation beyond supplied Ground",
            "execution of proposed tool calls",
            "zero physical interpretation/observation cost",
            "universal computability from any arbitrary short description",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="artifacts/response-coordinate.json")
    args=p.parse_args(argv)
    result=run()
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VLI_RESPONSE_COORDINATE_EXPERIMENT "+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
