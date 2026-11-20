"""SHAPE-004: verified structural memory for black-box observation histories.

Receiver-owned prefix states are retained as *actual source Ground
coordinates*, not learned model-state numbers. IFS COG-008 continues to
receive only its documented action/output membership interface; no hidden
state is returned to the learner. VLD's original deterministic finite
transition Ground is the authority, including snapshot-restore verification.

Caching is admitted only for pure, complete, deterministic transition
semantics. This does not cache arbitrary effectful tools or remote services.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Sequence

from .ground import _digest, _hex64, _require
from .response_state import _strict_json

SCHEMA="vl-i.observation-prefix-memory.v1"
MAX_NODES=20000
MAX_WORD=64


class StructuralPrefixMemory:
    """Shared, receiver-owned structural memory for one pinned transition source."""

    def __init__(self, source_bytes: bytes):
        from vld.signal.transitions import GroundTransitionSystem
        _require(type(source_bytes) is bytes and
                 0<len(source_bytes)<=1048576,
                 "Missing bounded original Ground source")
        source=GroundTransitionSystem.from_dict(_strict_json(source_bytes))
        _require(source.complete and len(source.states)<=64
                 and len(source.actions)<=16,
                 "Prefix snapshots require complete pure deterministic Ground")
        self.reference=source
        self.source_sha256=sha256(source_bytes).hexdigest()
        self.source_fingerprint=source.fingerprint
        # The admitted VLD transition system is a frozen dataclass; retain
        # its original identity rather than rehashing every source cell at
        # every black-box membership query.
        self._reference_anchor=source
        self.alphabet=tuple(source.actions)
        # Prefix -> (opaque source state for receiver, full output at last edge).
        # Only the source owner reads this; the learner sees projected outputs.
        self._nodes={(): (source.initial_state, None)}
        self.evaluated_transitions=0
        self.naive_transition_steps=0
        self.reused_prefix_edges=0
        self.membership_queries=0
        self.restoration_validation_steps=0

    @property
    def unique_nodes(self):
        return len(self._nodes)

    def _validate(self,word,channels):
        _require(type(word) in (list,tuple) and len(word)<=MAX_WORD
                 and all(type(a) is str and a in self.alphabet for a in word),
                 "Unadmitted cached observation history")
        _require(type(channels) is tuple and channels
                 and len(channels)==len(set(channels))
                 and all(type(c) is str and c in self.reference.channels for c in channels),
                 "Unqualified observer cannot read a hidden channel")
        _require(self.reference is self._reference_anchor,
                 "Admitted reference object was replaced under structural memory")
        return tuple(word),tuple(self.reference.channels.index(c) for c in channels)

    def observe(self,word:Sequence[str],channels:tuple[str,...]):
        """Return exactly the typed observer outputs of one source action word."""
        path,positions=self._validate(word,channels)
        missing=sum(path[:j] not in self._nodes for j in range(1,len(path)+1))
        _require(len(self._nodes)+missing<=MAX_NODES,
                 "Observed structural memory capacity exceeded")
        out=[]
        created=0
        for j,action in enumerate(path,1):
            prefix=path[:j]
            if prefix not in self._nodes:
                parent=self._nodes[path[:j-1]][0]
                result=self.reference.step(parent,action)
                self._nodes[prefix]=(result.target,tuple(result.output))
                created+=1
            projected=tuple(self._nodes[prefix][1][i] for i in positions)
            out.append(projected)
        self.evaluated_transitions+=created
        self.reused_prefix_edges+=len(path)-created
        self.naive_transition_steps+=len(path)
        self.membership_queries+=1
        return tuple(out)

    def checkpoint(self):
        _require(self.reference is self._reference_anchor and
                 self.reference.fingerprint==self.source_fingerprint,
                 "Original Ground identity changed before durable memory publication")
        entries=[]
        for word in sorted(self._nodes,key=lambda w:(len(w),w)):
            state,output=self._nodes[word]
            entries.append({
                "path":list(word),"source_state":state,
                "output":list(output) if output is not None else None})
        body={
            "schema":SCHEMA,"source_sha256":self.source_sha256,
            "source_fingerprint":self.source_fingerprint,
            "state_count":len(self.reference.states),
            "actions":list(self.alphabet),
            "entries":entries,
        }
        body["checkpoint_digest"]=_digest(SCHEMA,body)
        return body

    @classmethod
    def restore(cls,source_bytes,checkpoint,*,expected_head):
        _require(type(checkpoint) is dict and _hex64(expected_head)
                 and checkpoint.get("schema")==SCHEMA
                 and checkpoint.get("checkpoint_digest")==expected_head
                 and _digest(SCHEMA,{
                     k:v for k,v in checkpoint.items() if k!="checkpoint_digest"})==
                     expected_head,
                 "Missing independent receiver pin or changed structural memory")
        fresh=cls(source_bytes)
        _require(checkpoint.get("source_sha256")==fresh.source_sha256
                 and checkpoint.get("source_fingerprint")==fresh.source_fingerprint
                 and checkpoint.get("state_count")==len(fresh.reference.states)
                 and checkpoint.get("actions")==list(fresh.alphabet)
                 and type(checkpoint.get("entries")) is list
                 and 1<=len(checkpoint["entries"])<=MAX_NODES,
                 "Wrong original Ground or invalid structural-memory schema")
        restored={}
        for entry in checkpoint["entries"]:
            _require(type(entry) is dict and
                     set(entry)=={"path","source_state","output"}
                     and type(entry["path"]) is list
                     and len(entry["path"])<=MAX_WORD
                     and all(type(a) is str and a in fresh.alphabet
                             for a in entry["path"]),
                     "Malformed state prefix node")
            word=tuple(entry["path"])
            _require(word not in restored,"Duplicate structural-memory prefix")
            if not word:
                _require(entry["source_state"]==fresh.reference.initial_state and
                         entry["output"] is None,
                         "Wrong initial state in structural memory")
                restored[word]=(entry["source_state"],None)
                continue
            _require(word[:-1] in restored,
                     "Structural-memory child is missing causal parent")
            parent=restored[word[:-1]][0]
            oracle=fresh.reference.step(parent,word[-1])
            fresh.restoration_validation_steps+=1
            _require(entry["source_state"]==oracle.target and
                     entry["output"]==list(oracle.output),
                     "Cached source state/output was not justified by original model")
            restored[word]=(oracle.target,tuple(oracle.output))
        _require(()
                 in restored and len(restored)==len(checkpoint["entries"]),
                 "Missing root or duplicate prefix")
        fresh._nodes=restored
        return fresh

    def stats(self):
        return {
            "source_sha256":self.source_sha256,
            "source_fingerprint":self.source_fingerprint,
            "membership_queries":self.membership_queries,
            "naive_transition_steps":self.naive_transition_steps,
            "actually_evaluated_source_transitions":self.evaluated_transitions,
            "reused_prefix_edges":self.reused_prefix_edges,
            "unique_cached_state_prefixes":self.unique_nodes,
            "restoration_source_validation_transitions":self.restoration_validation_steps,
            "transition_work_saved_excluding_memory_lookup":(
                self.naive_transition_steps-self.evaluated_transitions),
            "work_scope":"pure_deterministic_source_traces_not_remote_effectful_tools",
        }
