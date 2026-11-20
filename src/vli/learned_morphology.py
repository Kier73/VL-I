"""SHAPE-003: learn observer-relative intelligence shape from black-box behavior.

Use the ORIGINAL IFS COG-008 ObservedOracle (source-visible ONLY through
membership action/output traces), ORIGINAL IFS COG-007 observation-table
candidate constructor and counterexample revision, VLD complete Mealy
reference as an INDEPENDENT verifier, and VLD's existing neural synthesizer
via COORD-002 to produce genuinely new response-state content.

The independently trusted verifier can see the original finite source; the
learner does NOT receive its state graph, transitions, or weights. This
separation is intentional. A finitely challenged hypothesis is *never*
promoted to a certified Ground without whole-declared-domain verification.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter_ns
import json

from .ground import _digest, _hex64, _require
from .response_state import _strict_json
from .generative_response import GeneratedModelGround, GenerativeFormation

SCHEMA = "vl-i.learned-observer-ground.v1"
CERT_SCHEMA = "vl-i.learned-observer-certification.v1"
TURN_SCHEMA = "vl-i.learned-observer-conversation-turn.v1"
SESSION_SCHEMA = "vl-i.learned-observer-conversation-snapshot.v1"


class BlackBoxObserver:
    """Separate probe interface: no state graph exposed to IFS learner."""

    def __init__(self, reference, *, channels: tuple[str, ...], source_digest: str):
        _require(type(channels) is tuple and channels and
                 len(channels)==len(set(channels)) and
                 all(c in reference.channels for c in channels),
                 "Black-box observation contract cannot access that channel")
        self._reference=reference
        self._positions=tuple(reference.channels.index(c) for c in channels)
        self.channels=channels
        self.alphabet=tuple(reference.actions)
        self.source_root=_digest("vl-i.observer-black-box-source.v1",{
            "original_source":source_digest,
            "vld_fingerprint":reference.fingerprint,
            "channels":list(channels),
        })
        self.direct_transition_steps=0

    def query(self, actions):
        _require(type(actions) in (tuple,list) and len(actions)<=64
                 and all(a in self.alphabet for a in actions),
                 "Unadmitted black-box path")
        s=self._reference.initial_state
        outputs=[]
        for a in actions:
            step=self._reference.step(s,a)
            s=step.target
            outputs.append(tuple(step.output[i] for i in self._positions))
            self.direct_transition_steps+=1
        return tuple(outputs)


def independent_finite_verdict(reference, machine, channels: tuple[str,...]):
    """Whole-reachable-domain conformance receiver, NOT called by IFS learner.

    BFS over (original hidden source state, candidate predictive state). A
    returned shortest action word is counterexample evidence; there is no
    access to the hidden model from the candidate construction algorithm.
    """
    _require(tuple(reference.actions)==machine.alphabet and
             all(c in reference.channels for c in channels),
             "Reference and learned model alphabets or channels disagree")
    projection=tuple(reference.channels.index(c) for c in channels)
    queue=deque([(reference.initial_state,0,())])
    visited={(reference.initial_state,0)}
    compared=0
    while queue:
        source_state, predictive_state, prefix=queue.popleft()
        for i,action in enumerate(reference.actions):
            real=reference.step(source_state,action)
            intended=tuple(real.output[j] for j in projection)
            proposed=machine.emissions[predictive_state][i]
            compared+=1
            if proposed!=intended:
                return {"exact":False,"counterexample":prefix+(action,),
                        "checked_product_states":len(visited),
                        "checked_product_edges":compared,
                        "source_observer_channels":list(channels)}
            nxt=(real.target,machine.transitions[predictive_state][i])
            if nxt not in visited:
                _require(len(visited)<10000,
                         "Finite model verifier exceeded declared product budget")
                visited.add(nxt)
                queue.append((nxt[0],nxt[1],prefix+(action,)))
    return {"exact":True,"counterexample":None,
            "checked_product_states":len(visited),
            "checked_product_edges":compared,
            "source_observer_channels":list(channels)}


class LearnedObserverGround:
    """An IFS-learned model promoted only after an independent exact check."""

    def __init__(self, reference_source: bytes, channels: tuple[str,...],
                 *, seed_suffix: tuple[str,...] = ("reset",),
                 max_membership: int = 5000, max_revisions: int = 32):
        from vld.signal.transitions import GroundTransitionSystem
        from partial_observation import ObservedOracle, fit_table, first_mismatch
        from predictive_state import ActiveMachineLearner

        _require(type(reference_source) is bytes and 0<len(reference_source)<=1048576,
                 "Missing black-box source origin")
        reference=GroundTransitionSystem.from_dict(_strict_json(reference_source))
        _require(reference.complete and 2<=len(reference.states)<=64
                 and 2<=len(reference.actions)<=16
                 and 1<=len(reference.channels)<=3,
                 "Need closed finite external conformance contract")
        _require(type(seed_suffix) is tuple and len(seed_suffix)==1
                 and seed_suffix[0] in reference.actions and
                 type(max_membership) is int and 8<=max_membership<=100000
                 and type(max_revisions) is int and 1<=max_revisions<=64,
                 "Invalid bounded observational learning budget")
        self.reference=reference  # Verifier scope only, not given to learner.
        self.source_digest=sha256(reference_source).hexdigest()
        self.channels=channels
        self.source=BlackBoxObserver(reference,channels=channels,
                                     source_digest=self.source_digest)
        self.original_source_bytes=len(reference_source)
        start=perf_counter_ns()
        oracle=ObservedOracle(self.source,max_queries=max_membership)
        learning=ActiveMachineLearner(oracle)
        # A deliberately incomplete first observer lens; it will mistake
        # distinguishable states for equal until a real counterexample arrives.
        learning.E=[seed_suffix]
        history=[]
        candidate=None
        verdict=None
        for round_index in range(max_revisions):
            candidate=fit_table(learning,max_rounds=2000)
            verdict=independent_finite_verdict(reference,candidate,channels)
            row={
                "round":round_index,
                "candidate_states":len(candidate.representatives),
                "candidate_fingerprint":candidate.fingerprint,
                "exact":verdict["exact"],
                "checked_full_source_edges":verdict["checked_product_edges"],
                "counterexample":list(verdict["counterexample"])
                    if verdict["counterexample"] is not None else None,
                "observed_membership_queries":oracle.external_calls,
                "observed_ledger_head":oracle.head,
            }
            history.append(row)
            if verdict["exact"]:
                break
            path=verdict["counterexample"]
            observed=first_mismatch(candidate,oracle,path)
            _require(observed is not None and observed==path,
                     "Source verifier counterexample not independently reproducible")
            # IFS-COG-007 refines rows from actual observed continuations.
            if path not in learning.counterexamples:
                learning.counterexamples.append(path)
            for i in range(len(path)):
                tail=path[i:]
                if tail not in learning.E:
                    learning.E.append(tail)
        _require(verdict is not None and verdict["exact"],
                 "No exact candidate under finite counterexample budget")
        _require(len(candidate.representatives)>=2 and
                 len(candidate.representatives)<=16,
                 "Learned model exceeds qualified VLD neural synthesizer capacity")
        self.oracle=oracle
        self.learning=learning
        self.machine=candidate
        self.revision_history=tuple(history)
        self.verdict=verdict
        self.learning_ns=perf_counter_ns()-start
        self.ledger=oracle.document()
        # IFS COG-008 keeps the observed ledger and candidate reconstruction
        # independent from global equivalence. We use that original checkpoint
        # authority, with a *separate* external exact-verification certificate.
        from partial_observation import checkpoint as ifs_checkpoint, recover as ifs_recover
        self.ifs_checkpoint=ifs_checkpoint(oracle,learning,candidate,{
            "status":"provisional_finitely_challenged",
            "global_equivalence":"not_established_by_observations_alone",
            "external_verified":False,
        })
        reconstructed,_=ifs_recover(
            self.ifs_checkpoint,
            expected_checkpoint_digest=self.ifs_checkpoint["digest"],
            expected_ledger_head=oracle.head,
            expected_source_root=oracle.source_root)
        _require(reconstructed.document()==candidate.document(),
                 "IFS failed source-free reconstruction from actual observations")
        proof={
            "schema":CERT_SCHEMA,
            "source_sha256":self.source_digest,
            "source_vld_fingerprint":reference.fingerprint,
            "observer_channels":list(channels),
            "black_box_contract_root":self.source.source_root,
            "ifs_ledger_head":oracle.head,
            "ifs_ledger_digest":self.ledger["digest"],
            "ifs_checkpoint_digest":self.ifs_checkpoint["digest"],
            "learned_machine_fingerprint":candidate.fingerprint,
            "candidate_count":len(history),
            "revision_trace":list(history),
            "exact_finite_domain":verdict,
            "assertion":"independent_exhaustive_product_reference_equivalence",
        }
        proof["proof_digest"]=_digest(CERT_SCHEMA,proof)
        self.certificate=proof
        self.neural_ground=GeneratedModelGround(self._neural_description())
        self.manifest={
            "schema":SCHEMA,
            "source_sha256":self.source_digest,
            "source_vld_fingerprint":reference.fingerprint,
            "observer_channels":list(channels),
            "ifs_ledger_head":oracle.head,
            "certificate_digest":proof["proof_digest"],
            "learned_predictive_states":len(candidate.representatives),
            "generated_weights_digest":self.neural_ground.weights_digest,
            "generated_model_digest":self.neural_ground.manifest["ground_digest"],
            "source_acquisition":"IFS_observation_only_membership_then_independent_receiver_qualification",
        }
        self.manifest["ground_digest"]=_digest(SCHEMA,self.manifest)

    def _neural_description(self):
        import json
        actions=list(self.machine.alphabet)
        states=[f"q{i:03d}" for i in range(len(self.machine.representatives))]
        rows=[]
        for i,source in enumerate(states):
            for j,action in enumerate(actions):
                rows.append({
                    "source":source,"action":action,
                    "outcomes":[{"target":states[self.machine.transitions[i][j]],
                                 "output":list(self.machine.emissions[i][j])}],
                    "support_count":1,
                    "evidence":[self.oracle.head,self.machine.fingerprint,
                                self.source.source_root],
                })
        doc={
            "format":"vld.transition-system.v1",
            "initial_state":states[0],
            "states":states,"actions":actions,
            "channels":list(self.channels),
            "observations":rows,
            "metadata":{
                "learned_ifs_hypothesis":self.machine.fingerprint,
                "ifs_witness_ledger":self.ledger["digest"],
                "independent_exact_certificate":self.certificate["proof_digest"],
                "source_contract":self.source.source_root,
                "scope":"typed_observer_output_exact_finite_domain_only",
            },
        }
        return json.dumps(doc,sort_keys=True,separators=(",", ":")).encode("utf-8")

    def check(self, *, expected_certificate_digest: str):
        from partial_observation import recover
        _require(_hex64(expected_certificate_digest)
                 and expected_certificate_digest==self.certificate["proof_digest"]
                 and expected_certificate_digest==_digest(CERT_SCHEMA,{
                     k:v for k,v in self.certificate.items() if k!="proof_digest"})
                 and self.manifest["certificate_digest"]==expected_certificate_digest
                 and self.reference.fingerprint==self.manifest["source_vld_fingerprint"]
                 and self.machine.fingerprint==
                     self.certificate["learned_machine_fingerprint"]
                 and self.neural_ground.manifest["ground_digest"]==
                     self.manifest["generated_model_digest"]
                 and self.manifest["ground_digest"]==_digest(SCHEMA,{
                     k:v for k,v in self.manifest.items() if k!="ground_digest"}),
                 "Changed model, proof or declared observed Ground")
        restored,_=recover(
            self.ifs_checkpoint,
            expected_checkpoint_digest=self.certificate["ifs_checkpoint_digest"],
            expected_ledger_head=self.certificate["ifs_ledger_head"],
            expected_source_root=self.source.source_root)
        _require(restored.document()==self.machine.document(),
                 "Learned model differs from hash-linked original observation ledger")
        independent=independent_finite_verdict(self.reference,self.machine,self.channels)
        _require(independent==self.verdict and independent["exact"],
                 "Learned observer capability has lost certified source equivalence")
        return True

    def model_state_after(self, history):
        _require(type(history) is tuple and len(history)<=48
                 and all(a in self.machine.alphabet for a in history),
                 "Unqualified external history")
        return f"q{self.machine.state_after(history):03d}"


class LearnedMorphologyConversation:
    """Change observer models while preserving symbolic action/causal history."""

    def __init__(self, models: tuple[LearnedObserverGround,...]):
        _require(type(models) is tuple and len(models)>=2 and
                 all(isinstance(m,LearnedObserverGround) for m in models)
                 and len({m.manifest["ground_digest"] for m in models})==len(models)
                 and len({m.source_digest for m in models})==1
                 and len({tuple(m.reference.actions) for m in models})==1,
                 "All learned views must share one original causal model")
        self.models=models
        self.history:tuple[str,...]=()
        self.head=_digest(SESSION_SCHEMA,{
            "source_sha256":models[0].source_digest,
            "admitted_shapes":[m.manifest["ground_digest"] for m in models],
            "genesis":"observer_specific_shape_revision",
        })
        self.events=[]

    def respond(self, model: LearnedObserverGround, *, prompt: str, action: str,
                views:tuple[str,...]=("text","code","tokens")):
        _require(model in self.models and len(self.history)<40
                 and action in model.machine.alphabet,
                 "Unknown observer model or unadmitted action")
        model.check(expected_certificate_digest=model.certificate["proof_digest"])
        state=model.model_state_after(self.history)
        form=GenerativeFormation(
            model.neural_ground,state=state,action=action,
            prompt=prompt,history_head=self.head,
            observer="observer-shape-learning",
            views=views)
        pre=form.observe()
        _require(pre["phase"]=="LOCATED" and
                 pre["neural_steps_executed_this_formation"]==0,
                 "Missing intention-address before neural generation")
        done=form.finish()
        original_state=model.reference.initial_state
        for a in self.history:
            original_state=model.reference.step(original_state,a).target
        original_step=model.reference.step(original_state,action)
        projected=tuple(original_step.output[model.reference.channels.index(c)]
                        for c in model.channels)
        _require(tuple(done["step_receipt"]["channel_output"])==projected
                 and done["step_receipt"]["target_state"]==
                     model.model_state_after(self.history+(action,)),
                 "Generated response conflicts with independent original observation")
        event={
            "schema":TURN_SCHEMA,
            "source_sha256":model.source_digest,
            "prior_head":self.head,
            "observer_ground":model.manifest["ground_digest"],
            "observer_channels":list(model.channels),
            "history":list(self.history),
            "prompt":prompt,"action":action,
            "position":done["coordinate"],
            "observed_values":list(projected),
            "target_learned_state":done["step_receipt"]["target_state"],
            "render_digest":done["observation_digest"],
            "rendered_view_kinds":[v["kind"] for v in done["projections"]],
        }
        event["event_digest"]=_digest(TURN_SCHEMA,event)
        self.events.append(event)
        self.history=self.history+(action,)
        self.head=event["event_digest"]
        return {"event":event,"response":done}

    def checkpoint(self):
        body={
            "schema":SESSION_SCHEMA,
            "source_sha256":self.models[0].source_digest,
            "admitted_models":[m.manifest["ground_digest"] for m in self.models],
            "symbolic_history":list(self.history),
            "head":self.head,
            "events":list(self.events),
        }
        body["checkpoint_digest"]=_digest(SESSION_SCHEMA,body)
        return body

    @classmethod
    def restore(cls, models, saved:dict, *, expected_head:str):
        _require(type(saved) is dict and _hex64(expected_head)
                 and saved.get("schema")==SESSION_SCHEMA
                 and saved.get("checkpoint_digest")==expected_head
                 and _digest(SESSION_SCHEMA,{
                     k:v for k,v in saved.items() if k!="checkpoint_digest"})==expected_head,
                 "Untrusted conversational state checkpoint")
        restored=cls(models)
        _require(saved["source_sha256"]==restored.models[0].source_digest
                 and saved["admitted_models"]==
                     [m.manifest["ground_digest"] for m in restored.models]
                 and type(saved["events"]) is list and len(saved["events"])<=40,
                 "Changed or absent learned morphology authorities")
        by_id={m.manifest["ground_digest"]:m for m in models}
        for row in saved["events"]:
            _require(type(row) is dict and row.get("observer_ground") in by_id,
                     "Unadmitted recovered observer state")
            new=restored.respond(
                by_id[row["observer_ground"]],
                prompt=row["prompt"],action=row["action"],
                views=tuple(row["rendered_view_kinds"]))
            _require(new["event"]==row,
                     "Recovered observation is not the same semantic state")
        _require(restored.head==saved["head"]
                 and list(restored.history)==saved["symbolic_history"],
                 "Restored causal state changed")
        return restored



class LearnedObserverIntentionRouter:
    """Reuse E-CPU's witnessed intent-to-topology on *learned* observer models.

    The intention proposal remains E0 advisory until the exact proposal head
    is externally approved. The chosen model is separately certified by its
    IFS observation ledger, independent source conformance and VLD compiler.
    This adapter does not add an independent classifier or execute tools.
    """

    def __init__(self, conversation: LearnedMorphologyConversation):
        from .response_state import ResponseSourceGround, SourceSlot
        _require(isinstance(conversation,LearnedMorphologyConversation),
                 "Missing causal learned observer Ground")
        self.conversation=conversation
        root=conversation.models[0].source_digest
        names=tuple(c for m in conversation.models for c in m.channels)
        names=tuple(sorted(set(names)))
        self.lenses=ResponseSourceGround(tuple(
            SourceSlot(c,"text",("learned-observer:"+root+":"+c).encode("utf-8"))
            for c in names
        ))

    def propose(self, witnesses, prompt, *, min_confidence=0.35,
                min_margin=0.001):
        from .intent_topology import propose_response_topology
        session=self.conversation
        p=propose_response_topology(
            self.lenses,witnesses,prompt,
            min_confidence=min_confidence,min_margin=min_margin)
        base={
            "schema":"vl-i.learned-observer-intention.v1",
            "original_source_sha256":session.models[0].source_digest,
            "causal_head":session.head,
            "history":list(session.history),
            "admitted_model_digests":[m.manifest["ground_digest"] for m in session.models],
            "observed_ledger_heads":[m.oracle.head for m in session.models],
            "ecpu_intention":p,
            "status":p["status"],
            "scope":"E0_advisory_requires_receiver_pin_and_independent_model_proof",
        }
        base["proposal_digest"]=_digest(base["schema"],base)
        return base

    def accept(self, witnesses, proposal, *, expected_proposal_head:str):
        from .intent_topology import admit_response_topology
        session=self.conversation
        _require(type(proposal) is dict
                 and proposal.get("schema")=="vl-i.learned-observer-intention.v1"
                 and _hex64(expected_proposal_head)
                 and proposal.get("proposal_digest")==expected_proposal_head
                 and _digest(proposal["schema"],{
                     k:v for k,v in proposal.items() if k!="proposal_digest"})==
                     expected_proposal_head
                 and proposal.get("original_source_sha256")==
                     session.models[0].source_digest
                 and proposal.get("causal_head")==session.head
                 and proposal.get("history")==list(session.history)
                 and proposal.get("admitted_model_digests")==[
                     m.manifest["ground_digest"] for m in session.models]
                 and proposal.get("observed_ledger_heads")==[
                     m.oracle.head for m in session.models],
                 "E-CPU observer proposal refers to another learned model or causal state")
        e=proposal.get("ecpu_intention")
        _require(type(e) is dict and
                 proposal.get("status")=="PROPOSED_NEEDS_OBSERVER_APPROVAL",
                 "Uncertain observer intention has no admission")
        typed=admit_response_topology(
            self.lenses,witnesses,e,
            expected_proposal_head=e["proposal_digest"],
            observer="learned-shape-router")
        channels=tuple(typed["selected_slots"])
        eligible=[m for m in session.models if m.channels==channels]
        _require(len(eligible)==1,
                 "No independently verified learned model for requested channels")
        model=eligible[0]
        model.check(expected_certificate_digest=model.certificate["proof_digest"])
        return model

    def accept_and_respond(self, witnesses, proposal,
                           *, expected_proposal_head:str, action:str,
                           views:tuple[str,...]=("text","code","tokens")):
        model=self.accept(witnesses,proposal,
                          expected_proposal_head=expected_proposal_head)
        return self.conversation.respond(
            model,prompt=proposal["ecpu_intention"]["prompt"],
            action=action,views=views)
