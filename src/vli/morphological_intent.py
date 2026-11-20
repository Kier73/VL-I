"""SHAPE-002: externally pinned prompt -> observer morphogenesis.

Reuses E-CPU's existing witnessed IntentPrototypeModel via VL-I
intent_topology; it does not implement its own classifier. A prompt is an
advisory E0 proposal until the exact source Ground and causal head are
externally pinned by an observer. The returned channels determine which
VLD-qualified morphology is allowed to generate a response.
"""
from __future__ import annotations

from .ground import _digest, _hex64, _require
from .intent_topology import (
    TopologyWitness, propose_response_topology, admit_response_topology,
)
from .morphological_generation import (
    MorphologySourceGround, MorphologySession,
)
from .response_state import ResponseSourceGround, SourceSlot

SCHEMA="vl-i.morphology-intention-proposal.v1"


class MorphologyIntentionRouter:
    def __init__(self, source: MorphologySourceGround):
        _require(isinstance(source,MorphologySourceGround),
                 "Missing original model Ground")
        self.source=source
        # Each observer-lens source byte Ground includes the model identity.
        # The existing E-CPU/VL-I intent-prototype pathway then binds the
        # proposal to the true source, not a generic unqualified label.
        self.lenses=ResponseSourceGround(tuple(
            SourceSlot(c,"text",
                       ("qualified-observer:"+source.manifest["ground_digest"]+
                        ":"+c).encode("utf-8"))
            for c in source.system.channels
        ))

    def propose(self, session: MorphologySession,
                witnesses: tuple[TopologyWitness,...], prompt: str, *,
                min_confidence:float=0.35,
                min_margin:float=0.001):
        _require(isinstance(session,MorphologySession)
                 and session.source is self.source,
                 "Cannot propose across a different causal model")
        ecpu=propose_response_topology(
            self.lenses,witnesses,prompt,
            min_confidence=min_confidence,min_margin=min_margin)
        body={
            "schema":SCHEMA,
            "original_model_ground_digest":self.source.manifest["ground_digest"],
            "causal_session_head":session.head,
            "full_source_state":session.state,
            "ecpu_intention_proposal":ecpu,
            "status":ecpu["status"],
            "scope":"qualified_E0_advisory_only_without_tool_or_state_authority",
        }
        body["proposal_digest"]=_digest(SCHEMA,body)
        return body

    def accept(self, session:MorphologySession, witnesses:tuple[TopologyWitness,...],
               proposal:dict, *, expected_proposal_head:str,
               observer:str="vli-observer"):
        _require(isinstance(session,MorphologySession)
                 and session.source is self.source
                 and type(proposal) is dict and
                 proposal.get("schema")==SCHEMA
                 and _hex64(expected_proposal_head)
                 and proposal.get("proposal_digest")==expected_proposal_head
                 and _digest(SCHEMA,{k:v for k,v in proposal.items()
                                     if k!="proposal_digest"})==expected_proposal_head
                 and proposal.get("original_model_ground_digest")==
                     self.source.manifest["ground_digest"]
                 and proposal.get("causal_session_head")==session.head
                 and proposal.get("full_source_state")==session.state,
                 "Observer proposal missing, forged or stale for this source state")
        ecpu=proposal.get("ecpu_intention_proposal")
        typed=admit_response_topology(
            self.lenses,witnesses,ecpu,
            expected_proposal_head=ecpu["proposal_digest"],
            observer=observer)
        _require(proposal["status"]=="PROPOSED_NEEDS_OBSERVER_APPROVAL",
                 "Unqualified observer intentions must not become actions")
        channels=tuple(typed["selected_slots"])
        _require(channels and all(c in self.source.system.channels
                                  for c in channels),
                 "E-CPU topology lies outside model's observer channels")
        return channels
