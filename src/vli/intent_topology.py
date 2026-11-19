"""Observed-prompt -> proposed response topology through real E-CPU prototype laws.

This is an advisory, witness-formed intention selector, not a general language
model. E-CPU's IntentPrototypeModel provides the only classifier; the VL-I
adapter binds its training witnesses, source Ground, model snapshot, and
uncertainty. A caller must independently pin/accept the proposal before
turning it into a response-state coordinate. No tool is executed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ground import _digest, _hex64, _require
from .response_state import ResponseSourceGround, compile_intention

SCHEMA = "vl-i.intention-topology-proposal.v1"


@dataclass(frozen=True)
class TopologyWitness:
    prompt: str
    slots: tuple[str, ...]

    def __post_init__(self):
        _require(type(self.prompt) is str and 0 < len(self.prompt.encode("utf-8")) <= 8192,
                 "Invalid witnessed intention")
        _require(type(self.slots) is tuple and self.slots and
                 len(set(self.slots)) == len(self.slots),
                 "Invalid witnessed topology")


def _prompt_state(prompt: str) -> dict:
    # Actual token shape and similarity are E-CPU's, not locally invented.
    return {"operation": prompt, "resource": "/virtual-ground/response",
            "payload": {"result_kind": "renderable-state", "intention": prompt}}


def _induce(source, witnesses, prompt, *, min_confidence, min_margin):
    from mailcpu.intent_embedding import IntentEmbedding, IntentExample, IntentPrototypeModel

    _require(isinstance(source, ResponseSourceGround), "Missing response Ground")
    _require(type(witnesses) is tuple and len(witnesses) >= 2
             and all(isinstance(w, TopologyWitness) for w in witnesses)
             and type(prompt) is str and 0 < len(prompt.encode("utf-8")) <= 8192,
             "Unqualified source or training witnesses")
    _require(type(min_confidence) in (float,int) and
             0.0 <= min_confidence <= 1.0 and
             type(min_margin) in (float,int) and 0.0 <= min_margin <= 1.0,
             "Invalid advisory uncertainty threshold")
    groups = sorted({w.slots for w in witnesses})
    _require(2 <= len(groups) <= 32, "Require independently contrasted output topologies")
    for w in witnesses:
        source.closure(w.slots)
    labels = {group: f"topology{i:03d}" for i,group in enumerate(groups)}
    examples = tuple(IntentExample(labels[w.slots], _prompt_state(w.prompt))
                     for w in witnesses)
    model = IntentPrototypeModel.fit(examples, encoder=IntentEmbedding(dimension=256))
    prediction = model.predict(_prompt_state(prompt))
    by_label = {name: slots for slots,name in labels.items()}
    probs = sorted(prediction.probabilities.values(),reverse=True)
    margin = probs[0] - probs[1]
    accepted = (prediction.confidence >= min_confidence and margin >= min_margin)
    proposal = {
        "schema": SCHEMA, "source_ground_digest": source.manifest["ground_digest"],
        "prompt": prompt, "witnesses": [
            {"prompt": w.prompt, "slots": list(w.slots)} for w in witnesses],
        "witness_digest": _digest("vl-i.topology-witnesses.v1", [
            {"prompt": w.prompt, "slots": list(w.slots)} for w in witnesses]),
        "ecpu_prototype_snapshot_digest": _digest(
            "vl-i.ecpu-intent-prototype.v1",model.snapshot()),
        "ecpu_prediction": prediction.to_dict(),
        "min_confidence": float(min_confidence),
        "min_margin": float(min_margin),
        "observed_margin": margin,
        "proposed_slots": list(by_label[prediction.domain]) if accepted else [],
        "status": "PROPOSED_NEEDS_OBSERVER_APPROVAL" if accepted else "INSUFFICIENT_INTENT_EVIDENCE",
        "scope": "witness_conditioned_structural_advisory_not_general_natural_language",
    }
    proposal["proposal_digest"] = _digest(SCHEMA, proposal)
    return proposal


def propose_response_topology(source: ResponseSourceGround,
                              witnesses: tuple[TopologyWitness, ...],
                              prompt: str, *, min_confidence: float = 0.55,
                              min_margin: float = 0.05) -> dict:
    """E-CPU learned structural prototypes select candidate observer topology.

    Caller supplies true witness labels; the classifier cannot discover
    previously unknown output kinds or authenticate claimed training examples.
    """
    return _induce(source, witnesses, prompt,
                   min_confidence=min_confidence, min_margin=min_margin)


def admit_response_topology(source: ResponseSourceGround,
                            witnesses: tuple[TopologyWitness, ...],
                            proposal: dict, *, expected_proposal_head: str,
                            observer: str, history: tuple[dict, ...] = ()) -> dict:
    """Accept after caller pins the actual proposal; no side-effect authority."""
    _require(type(proposal) is dict and proposal.get("schema") == SCHEMA
             and _hex64(expected_proposal_head)
             and proposal.get("proposal_digest") == expected_proposal_head,
             "Unpinned or changed topology proposal")
    independent = _induce(source, witnesses, proposal.get("prompt"),
                          min_confidence=proposal.get("min_confidence"),
                          min_margin=proposal.get("min_margin"))
    _require(proposal == independent, "E-CPU evidence or Ground topology changed")
    _require(proposal["status"] == "PROPOSED_NEEDS_OBSERVER_APPROVAL",
             "Intention has insufficient evidence for topology")
    return compile_intention(proposal["prompt"],
                             slots=tuple(proposal["proposed_slots"]),
                             observer=observer, history=history)
