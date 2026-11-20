"""SHAPE-002: intention-qualified morphology of a generative model Ground.

A position selects an observer's *semantic distinctions*, not a numeric state.
VLD's original Moore/Myhill-Nerode analysis proposes an observer quotient;
a separate exhaustive Mealy-output congruence proof qualifies it. The
quotient model is synthesized as actual neural weights through COORD-002's
VLD compiler. Same source, changed intention => a different shaped model,
address, generated code/text/tokens, and required interpreter work.

No algorithm can obtain an unspecified observation from a deliberately
erased distinction. Full-model equivalence is conditional on the observer
channels and action alphabet admitted in the proof.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from time import perf_counter_ns
from typing import Mapping

from .ground import _digest, _hex64, _require
from .response_state import _ecpu_verify, _strict_json
from .generative_response import GeneratedModelGround, GenerativeFormation

SOURCE_SCHEMA = "vl-i.morphological-model-ground.v1"
PROOF_SCHEMA = "vl-i.morphological-quotient-proof.v1"
POSITION_SCHEMA = "vl-i.morphological-state-position.v1"
OBS_SCHEMA = "vl-i.morphological-state-observation.v1"
MAX_SOURCE_BYTES = 1_048_576


def _canonical_bytes(value) -> bytes:
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class MorphologySourceGround:
    """Admit one complete VLD state machine without *requiring* full neural synthesis."""

    def __init__(self, source_bytes: bytes):
        from vld.signal.transitions import GroundTransitionSystem
        # Qualify the compiler backend at source-admission/bootstrap time:
        # every alternative model must be compared under the same initialized
        # interpreter, rather than making the first candidate alone pay its
        # Python module import cost. Source admission timing charges this work.
        from vld.models.procedural import compile_fsm_to_neural
        _require(type(source_bytes) is bytes and 0 < len(source_bytes) <= MAX_SOURCE_BYTES,
                 "Missing or excessive descriptive Ground")
        source = GroundTransitionSystem.from_dict(_strict_json(source_bytes))
        _require(source.complete and 2 <= len(source.states) <= 16
                 and 1 <= len(source.actions) <= 16
                 and 1 <= len(source.channels) <= 8,
                 "Need a complete, bounded deterministic neural Ground")
        self.source_bytes = source_bytes
        self.system = source
        self.ecpu_chunk_digests = tuple(_ecpu_verify(source_bytes))
        self.manifest = {
            "schema": SOURCE_SCHEMA,
            "source_sha256": sha256(source_bytes).hexdigest(),
            "source_bytes": len(source_bytes),
            "ecpu_chunk_digests": list(self.ecpu_chunk_digests),
            "vld_transition_fingerprint": source.fingerprint,
            "states": list(source.states),
            "actions": list(source.actions),
            "channels": list(source.channels),
            "evidence_scope": "actual_description_and_exact_all_cell_source",
        }
        self.manifest["ground_digest"] = _digest(SOURCE_SCHEMA, self.manifest)

    def observer(self, channels: tuple[str, ...]) -> "ObserverMorphology":
        return ObserverMorphology(self, channels=channels)


class ObserverMorphology:
    """Proof-carrying, observer-specific quotient; native VLD proposal and neural realizer."""

    def __init__(self, source: MorphologySourceGround, *, channels: tuple[str, ...]):
        from vld.triad import analyze_finite_ground
        _require(isinstance(source, MorphologySourceGround)
                 and type(channels) is tuple and channels
                 and len(channels) == len(set(channels))
                 and all(type(c) is str and c in source.system.channels for c in channels),
                 "Observer requests unavailable or duplicate semantic channels")
        self.source, self.channels = source, channels
        system=source.system
        positions = tuple(system.channels.index(c) for c in channels)
        self.positions = positions
        transitions = {
            (s,a): (system.step(s,a).target,)
            for s in system.states for a in system.actions
        }
        # VLD triad's state projection considers state-dependent observable
        # signatures, distinct from actual transition outputs. The separate
        # Mealy congruence check below covers the *emitted* channels too.
        # Mealy observer distinguishes all immediate outputs for every
        # admitted action; VLD then refines for all future action histories.
        # Starting from the complete output signature is essential: using
        # only one action could merge states that differ on another action.
        observer = {
            s: repr(tuple((a, tuple(system.step(s,a).output[i]
                                   for i in positions)) for a in system.actions))
            for s in system.states
        }
        t0 = perf_counter_ns()
        report=analyze_finite_ground(system.states,system.actions,transitions,observer)
        report.require_executable_deterministic()
        t1 = perf_counter_ns()
        classes = report.quotient_classes
        block = {state: f"q{index:03d}" for index,members in enumerate(classes)
                 for state in members}
        _require(len(block)==len(system.states), "VLD partition did not cover source")
        class_names=tuple(f"q{index:03d}" for index in range(len(classes)))
        checks=0
        instructions = []
        for index,members in enumerate(classes):
            for action in system.actions:
                # Independent whole-domain Mealy check: VLD's state observer
                # quotient must preserve BOTH next class and projected
                # transition output for every state/action in the class.
                outcomes=set()
                for state in members:
                    full=system.step(state,action)
                    projected=tuple(full.output[i] for i in positions)
                    outcomes.add((block[full.target],projected))
                    checks+=1
                _require(len(outcomes)==1,
                         "Candidate loses an observer distinction in action "+action)
                target,output=next(iter(outcomes))
                instructions.append({
                    "source":class_names[index], "action":action,
                    "outcomes":[{"target":target,"output":list(output)}],
                    "support_count":len(members),
                    "evidence":[source.manifest["ground_digest"],
                                "complete-Mealy-cell-congruence-v1"],
                })
        _require(checks == len(system.states)*len(system.actions),
                 "Not all original transition cells proved")
        t2=perf_counter_ns()
        proof={
            "schema":PROOF_SCHEMA,
            "source_ground_digest":source.manifest["ground_digest"],
            "vld_transition_fingerprint":system.fingerprint,
            "observer_channels":list(channels),
            "source_states":len(system.states),
            "observer_classes":len(classes),
            "state_to_class":{s:block[s] for s in sorted(block)},
            "classes":[list(members) for members in classes],
            "checked_original_cells":checks,
            "projected_transition_cells":instructions,
            "assurance":"VLD_predictive_partition_plus_independent_all_state_action_Mealy_congruence",
        }
        proof["proof_digest"]=_digest(PROOF_SCHEMA,proof)
        self.proof=proof
        self.class_map=block
        self.classes=classes
        self.timings_ns={
            "vld_predictive_partition":t1-t0,
            "independent_full_cell_Mealy_proof":t2-t1,
        }
        reduced={
            "format":"vld.transition-system.v1",
            "states":list(class_names), "actions":list(system.actions),
            "channels":list(channels),
            "initial_state":block[system.initial_state],
            "metadata":{
                "source_ground_digest":source.manifest["ground_digest"],
                "source_vld_system":system.fingerprint,
                "observer_proof_digest":proof["proof_digest"],
                "observer_channel_contract":list(channels),
                "semantic_law":"exact-output-congruence-not-LLM-inference",
            },
            "observations":instructions,
        }
        self.description_bytes=_canonical_bytes(reduced)
        t3=perf_counter_ns()
        self.generated_model=GeneratedModelGround(self.description_bytes)
        t4=perf_counter_ns()
        self.timings_ns["description_neural_compilation"]=t4-t3
        _require(tuple(self.generated_model.system.channels)==channels,
                 "Generated neural model changed observer outputs")
        self.manifest={
            "schema":"vl-i.observer-morphology-ground.v1",
            "source_ground_digest":source.manifest["ground_digest"],
            "observer_channels":list(channels),
            "proof_digest":proof["proof_digest"],
            "qualified_state_dimension":len(classes),
            "generated_model_digest":self.generated_model.manifest["ground_digest"],
            "generated_weights_digest":self.generated_model.weights_digest,
            "generated_hidden_units":len(self.generated_model.neural.hidden_bias),
        }
        self.manifest["ground_digest"]=_digest(self.manifest["schema"],self.manifest)

    def _assert_qualified(self, *, expected_proof_digest: str):
        _require(_hex64(expected_proof_digest)
                 and self.proof["proof_digest"]==expected_proof_digest
                 and self.proof["proof_digest"]==_digest(PROOF_SCHEMA,{
                     k:v for k,v in self.proof.items() if k!="proof_digest"})
                 and self.manifest["proof_digest"]==self.proof["proof_digest"]
                 and self.manifest["source_ground_digest"]==
                 self.source.manifest["ground_digest"]
                 and self.source.system.fingerprint==
                 self.source.manifest["vld_transition_fingerprint"]
                 and self.generated_model.manifest["ground_digest"]==
                 self.manifest["generated_model_digest"]
                 and self.generated_model.weights_digest==
                 self.manifest["generated_weights_digest"]
                 and self.manifest["ground_digest"]==_digest(
                    self.manifest["schema"],{
                        k:v for k,v in self.manifest.items() if k!="ground_digest"}),
                 "Source, proof, generated model or observer authority changed")

    def compare_full_cell(self, state: str, action: str):
        _require(state in self.source.system.states
                 and action in self.source.system.actions,
                 "Unknown source-state/intention relation")
        a=self.source.system.step(state,action)
        expected=(self.class_map[a.target],
                  tuple(a.output[i] for i in self.positions))
        derived=self.generated_model.generated(self.class_map[state],action)
        _require(expected==derived,
                 "Reduced neural model disagrees with full source semantics")
        return expected

    def position(self, *, state: str, action: str, prompt: str, observer: str,
                 causal_head: str, expected_proof_digest: str):
        self._assert_qualified(expected_proof_digest=expected_proof_digest)
        _require(state in self.source.system.states
                 and action in self.source.system.actions
                 and type(prompt) is str and 0<len(prompt)<=8192
                 and type(observer) is str and observer and len(observer)<=80
                 and _hex64(causal_head),
                 "Invalid intention or past state lineage")
        # The state coordinate binds the *model* and the exact current causal
        # state, but the result/value is not yet evaluated.
        body={
            "schema":POSITION_SCHEMA,
            "full_source_ground_digest":self.source.manifest["ground_digest"],
            "morphology_ground_digest":self.manifest["ground_digest"],
            "observer_proof_digest":self.proof["proof_digest"],
            "requested_channels":list(self.channels),
            "observer":observer, "source_state":state,
            "intention_action":action,"prompt":prompt,
            "causal_head":causal_head,
            "scope":"state_position_pre_observation_not_numeric_output",
        }
        position_digest=_digest(POSITION_SCHEMA,body)
        return {**body,"position_digest":position_digest,
                "coordinate":"state://vli/"+position_digest}

    def observe(self, position, *, expected_proof_digest: str,
                source_verify_every_cell: bool=False,
                views: tuple[str,...]=("text","code","tokens")):
        self._assert_qualified(expected_proof_digest=expected_proof_digest)
        _require(type(position) is dict and position.get("schema")==POSITION_SCHEMA
                 and position.get("morphology_ground_digest")==
                     self.manifest["ground_digest"]
                 and position.get("requested_channels")==list(self.channels)
                 and position.get("full_source_ground_digest")==
                     self.source.manifest["ground_digest"]
                 and position.get("position_digest")==_digest(POSITION_SCHEMA,{
                    k:v for k,v in position.items() if k not in
                    ("position_digest","coordinate")})
                 and position.get("coordinate")==
                     "state://vli/"+position["position_digest"],
                 "State coordinate changed or refers to a different observer")
        if source_verify_every_cell:
            # An independent receiver may re-run the exact full source proof
            # at every admission; its cost is reported, not silently excluded.
            for s in self.source.system.states:
                for a in self.source.system.actions:
                    self.compare_full_cell(s,a)
        state=position["source_state"]
        action=position["intention_action"]
        qstate=self.class_map[state]
        t0=perf_counter_ns()
        formation=GenerativeFormation(
            self.generated_model,state=qstate,action=action,
            prompt=position["prompt"],history_head=position["causal_head"],
            observer=position["observer"],views=views)
        before=formation.observe()
        _require(before["phase"]=="LOCATED"
                 and before["neural_steps_executed_this_formation"]==0,
                 "Expected a coordinate before the neural state was generated")
        final=formation.finish()
        t1=perf_counter_ns()
        full=self.source.system.step(state,action)
        expected=(self.class_map[full.target],
                  tuple(full.output[i] for i in self.positions))
        step=final["step_receipt"]
        _require((step["target_state"],tuple(step["channel_output"]))==expected,
                 "Rendered generated state differs from independent full source")
        result={
            "schema":OBS_SCHEMA,
            "coordinate":position["coordinate"],
            "position_digest":position["position_digest"],
            "proof_digest":self.proof["proof_digest"],
            "source_ground_digest":self.source.manifest["ground_digest"],
            "morphology_ground_digest":self.manifest["ground_digest"],
            "requested_channels":list(self.channels),
            "source_state":state,
            "source_target_class":expected[0],
            "observed_channel_values":list(expected[1]),
            "generated_neural_step":step,
            "generated_views":final["projections"],
            "inner_coordinate":final["coordinate"],
            "generated_neural_steps":formation.generative_step_count,
            "generate_and_render_ns":t1-t0,
            "source_recheck_all_cells":source_verify_every_cell,
            "scope":"qualified_observer_output_not_general_LLM",
        }
        # Host timings are non-semantic and must not alter the state identity
        # or make independently replayed causal history divergent.
        result["observation_digest"]=_digest(OBS_SCHEMA,{
            k:v for k,v in result.items() if k!="generate_and_render_ns"})
        return result


class MorphologySession:
    """Switch between qualified shapes without pretending discarded state is gone.

    The authoritative full source-state and causal head survive each observer
    projection, permitting finer later intentions. A quotient-only receiver
    without that state would have to recover the missing distinction.
    """

    def __init__(self, source: MorphologySourceGround):
        _require(isinstance(source,MorphologySourceGround), "Missing full source")
        self.source=source
        self.state=source.system.initial_state
        self.head=_digest("vl-i.morphological-session-genesis.v1",
                          source.manifest["ground_digest"])
        self.turns=[]

    def step(self, shape: ObserverMorphology, *, action: str, prompt: str,
             observer: str="vli-observer",
             views: tuple[str,...]=("text","code","tokens")):
        _require(shape.source is self.source,
                 "Morphology does not belong to this source")
        position=shape.position(state=self.state,action=action,prompt=prompt,
                                observer=observer,causal_head=self.head,
                                expected_proof_digest=shape.proof["proof_digest"])
        result=shape.observe(position,
                             expected_proof_digest=shape.proof["proof_digest"],
                             views=views)
        full=self.source.system.step(self.state,action)
        event={
            "schema":"vl-i.morphological-session-turn.v1",
            "previous":self.head,
            "original_ground_digest":self.source.manifest["ground_digest"],
            "observer_shape":shape.manifest["ground_digest"],
            "position":position["coordinate"],
            "prompt":prompt,
            "action":action,
            "observer":observer,
            "requested_views":list(views),
            "source_state":self.state,
            "next_full_source_state":full.target,
            "observed_channels":list(shape.channels),
            "observed_values":list(full.output[i] for i in shape.positions),
            "response_digest":result["observation_digest"],
        }
        event["turn_digest"]=_digest(event["schema"],event)
        self.head=event["turn_digest"]
        self.state=full.target
        self.turns.append(event)
        return {"position":position,"observation":result,"turn":event}

    @classmethod
    def restore(cls, source: MorphologySourceGround,
                shapes: tuple[ObserverMorphology, ...], checkpoint: dict,
                *, expected_checkpoint_head: str):
        """Independent pure-history replay; no prior neural observation is trusted.

        Retains the full source state, so later intentions can legitimately
        refine a previously coarser observer. Replay cost is explicit.
        """
        _require(type(checkpoint) is dict and _hex64(expected_checkpoint_head)
                 and checkpoint.get("schema")==
                    "vl-i.morphological-session-checkpoint.v1"
                 and checkpoint.get("checkpoint_digest")==expected_checkpoint_head
                 and expected_checkpoint_head==_digest(
                    checkpoint["schema"],{k:v for k,v in checkpoint.items()
                                          if k!="checkpoint_digest"})
                 and checkpoint.get("source_ground_digest")==
                     source.manifest["ground_digest"]
                 and type(checkpoint.get("turns")) is list
                 and len(checkpoint["turns"])<=48,
                 "Changed or unauthenticated morphology session")
        by_digest = {s.manifest["ground_digest"]:s for s in shapes
                     if isinstance(s,ObserverMorphology) and s.source is source}
        _require(len(by_digest)==len(shapes),"Missing or repeated observer shape")
        new=cls(source)
        for step in checkpoint["turns"]:
            _require(type(step) is dict and step.get("schema")==
                     "vl-i.morphological-session-turn.v1" and
                     step.get("observer_shape") in by_digest,
                     "Unqualified historical shape transition")
            shaped=by_digest[step["observer_shape"]]
            reproduced=new.step(shaped,action=step["action"],
                                prompt=step["prompt"],observer=step["observer"],
                                views=tuple(step["requested_views"]))
            _require(reproduced["turn"]==step,
                     "Historical semantic operation differs from certified source")
        _require(new.state==checkpoint["state"] and
                 new.head==checkpoint["head"],
                 "Recovered state differs from external history")
        return new

    def checkpoint(self):
        body={
            "schema":"vl-i.morphological-session-checkpoint.v1",
            "source_ground_digest":self.source.manifest["ground_digest"],
            "state":self.state,"head":self.head,
            "turns":[dict(x) for x in self.turns],
        }
        body["checkpoint_digest"]=_digest(body["schema"],body)
        return body



def select_qualified_morphology(
    source: MorphologySourceGround,
    candidates: tuple[ObserverMorphology, ...],
    *, requested_channels: tuple[str, ...],
    expected_queries: int,
    warm_sample_count: int = 128,
):
    """Choose by measured estimated *cold and warm* cost, not dimension alone.

    All candidates are already independently qualified against the same full
    Ground. A candidate omitting requested semantic channels cannot be used.
    Prepared-model timings and repeated neural+local-oracle work are kept
    separate. This is a workload-specific estimate, not a promised speedup
    across hosts, hardware or trained LLMs.
    """
    _require(isinstance(source, MorphologySourceGround)
             and type(candidates) is tuple and 1 <= len(candidates) <= 8
             and all(isinstance(c,ObserverMorphology) and c.source is source
                     for c in candidates), "Missing common source authority")
    _require(type(requested_channels) is tuple and requested_channels
             and all(type(c) is str and c in source.system.channels
                     for c in requested_channels)
             and len(requested_channels)==len(set(requested_channels))
             and type(expected_queries) is int and 1 <= expected_queries <= 1_000_000
             and type(warm_sample_count) is int and 4 <= warm_sample_count <= 4096,
             "Invalid requested observer scope or cost budget")
    options=[]
    states=source.system.states
    actions=source.system.actions
    for shape in candidates:
        admitted=set(requested_channels).issubset(set(shape.channels))
        if not admitted:
            options.append({
                "channels":list(shape.channels),"status":"REFUSED_INSUFFICIENT_OBSERVER_SCOPE",
                "proof_digest":shape.proof["proof_digest"],
            })
            continue
        shape._assert_qualified(expected_proof_digest=shape.proof["proof_digest"])
        samples=[(states[i%len(states)], actions[(i//len(states))%len(actions)])
                 for i in range(warm_sample_count)]
        t0=perf_counter_ns()
        observed=[]
        for s,a in samples:
            observed.append(shape.generated_model.generated(shape.class_map[s],a))
        t1=perf_counter_ns()
        for (s,a),result in zip(samples,observed):
            full=source.system.step(s,a)
            oracle=(shape.class_map[full.target],
                    tuple(full.output[j] for j in shape.positions))
            _require(result==oracle,"Timed neural sample failed external source comparison")
        prep=sum(shape.timings_ns.values())
        query_ns=(t1-t0)/warm_sample_count
        options.append({
            "channels":list(shape.channels),
            "status":"QUALIFIED",
            "proof_digest":shape.proof["proof_digest"],
            "source_state_dimension":len(states),
            "realized_state_dimension":shape.manifest["qualified_state_dimension"],
            "neural_hidden_units":shape.manifest["generated_hidden_units"],
            "preparation_ns":prep,
            "neural_and_local_oracle_ns_per_query":query_ns,
            "estimated_total_cold_ns":prep+expected_queries*query_ns,
            "estimated_total_warm_ns":expected_queries*query_ns,
            "warm_samples":warm_sample_count,
            "estimated_query_count":expected_queries,
        })
    admitted=[x for x in options if x["status"]=="QUALIFIED"]
    _require(admitted, "No proven morphology preserves requested semantic channels")
    cold=min(admitted,key=lambda x:(x["estimated_total_cold_ns"],x["realized_state_dimension"]))
    warm=min(admitted,key=lambda x:(x["estimated_total_warm_ns"],x["realized_state_dimension"]))
    body={
        "schema":"vl-i.morphology-selection-evidence.v1",
        "source_ground_digest":source.manifest["ground_digest"],
        "requested_channels":list(requested_channels),
        "expected_queries":expected_queries,
        "options":options,
        "cold_selection_channels":cold["channels"],
        "warm_selection_channels":warm["channels"],
        "cost_scope":"single_host_Python_finite_ground_measured_samples_no_energy_or_transport",
        "qualification":"source_wide_all_action_Mealy_observation_congruence",
    }
    body["selection_digest"]=_digest(body["schema"],body)
    return body
