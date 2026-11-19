"""COORD-002: observer prompt -> state-specific, qualified generative action.

E-CPU's existing structural IntentPrototypeModel learns advisory intent/action
relations from caller-witnessed examples. No new classifier or trained language
backend is introduced. A proposal cannot trigger a Ground transition without
external acceptance of its full fingerprint at the current causal state.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ground import _digest, _hex64, _require
from .generative_response import GenerativeConversation, GeneratedModelGround

SCHEMA = "vl-i.generated-action-intention.v1"


@dataclass(frozen=True)
class ActionWitness:
    prompt: str
    action: str

    def __post_init__(self):
        _require(type(self.prompt) is str and 0 < len(self.prompt.encode("utf-8")) <= 8192
                 and type(self.action) is str and self.action,
                 "Invalid witnessed generative intention")


def _prompt_state(prompt: str) -> dict:
    return {
        "operation": prompt,
        "resource": "/virtual-layer/description-native-model",
        "payload": {"observer_intention": prompt, "target": "state-transition"},
    }


def _propose(conversation, witnesses, prompt, *, min_confidence, min_margin):
    from mailcpu.intent_embedding import IntentEmbedding, IntentExample, IntentPrototypeModel

    _require(isinstance(conversation, GenerativeConversation)
             and type(witnesses) is tuple and 4 <= len(witnesses) <= 128
             and all(isinstance(w, ActionWitness) for w in witnesses),
             "No qualified witnessed action descriptions")
    _require(type(prompt) is str and 0 < len(prompt.encode("utf-8")) <= 8192,
             "Missing bounded prompt")
    _require(type(min_confidence) in (int, float) and
             type(min_margin) in (int, float) and
             0.0 <= min_confidence <= 1.0 and
             0.0 <= min_margin <= 1.0,
             "Invalid acceptance thresholds")
    labels = {w.action for w in witnesses}
    _require(2 <= len(labels) <= len(conversation.ground.system.actions)
             and labels.issubset(set(conversation.ground.system.actions)),
             "Witness action labels not contained in model description")
    for name in labels:
        _require(sum(w.action == name for w in witnesses) >= 2,
                 "Need at least two witnesses for each candidate action")
    model = IntentPrototypeModel.fit(
        tuple(IntentExample(w.action, _prompt_state(w.prompt)) for w in witnesses),
        encoder=IntentEmbedding(dimension=256))
    predicted = model.predict(_prompt_state(prompt))
    ordered = sorted(predicted.probabilities.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    admitted = predicted.confidence >= min_confidence and margin >= min_margin
    body = {
        "schema": SCHEMA,
        "model_ground_digest": conversation.ground.manifest["ground_digest"],
        "prior_conversation_head": conversation.head,
        "current_state": conversation.state,
        "prompt": prompt,
        "witnesses": [{"prompt": w.prompt, "action": w.action} for w in witnesses],
        "witness_digest": _digest(
            "vl-i.witnessed-generative-actions.v1",
            [{"prompt": w.prompt, "action": w.action} for w in witnesses]),
        "ecpu_prototype_digest": _digest("vl-i.generative-action-prototype.v1",
                                        model.snapshot()),
        "ecpu_advisory_prediction": predicted.to_dict(),
        "min_confidence": float(min_confidence),
        "min_margin": float(min_margin),
        "prediction_margin": margin,
        "action": predicted.domain if admitted else None,
        "status": ("PROPOSED_NEEDS_OBSERVER_ACCEPTANCE"
                   if admitted else "INSUFFICIENT_INTENTION_EVIDENCE"),
        "scope": "E0_observer_witnessed_advisory_not_general_language_understanding",
    }
    body["proposal_digest"] = _digest(SCHEMA, body)
    return body


def propose_action(conversation: GenerativeConversation,
                   witnesses: tuple[ActionWitness, ...], prompt: str,
                   *, min_confidence: float = 0.40,
                   min_margin: float = 0.01):
    """Suggest a known Ground action for the observer's text, without executing."""
    return _propose(conversation, witnesses, prompt,
                    min_confidence=min_confidence, min_margin=min_margin)


def accept_and_respond(conversation: GenerativeConversation,
                       witnesses: tuple[ActionWitness, ...], proposal: dict,
                       *, expected_proposal_head: str,
                       views: tuple[str, ...] = ("text", "page", "code", "tokens")):
    """Observer-approved, state-fenced action transition through GenerativeConversation."""
    _require(type(proposal) is dict and proposal.get("schema") == SCHEMA
             and _hex64(expected_proposal_head)
             and proposal.get("proposal_digest") == expected_proposal_head,
             "Missing externally pinned generative intention proposal")
    independent = _propose(conversation, witnesses, proposal.get("prompt"),
                           min_confidence=proposal.get("min_confidence"),
                           min_margin=proposal.get("min_margin"))
    _require(proposal == independent,
             "Description, causal state, witnesses or proposal have changed")
    _require(proposal["status"] == "PROPOSED_NEEDS_OBSERVER_ACCEPTANCE"
             and proposal["action"] in conversation.ground.system.actions,
             "Cannot promote uncertain or unadmitted observer intention")
    return conversation.respond(proposal["prompt"], proposal["action"], views=views)
