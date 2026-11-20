"""TIME-004: bind an observed environment to a constructed reflective processor.

Uses the *actual* Navigable-Compute NCI/NCS/NCR-002 implementations, with their
independent checkers, rather than copying/reimplementing their semantics.

The task is intentionally narrow: navigate a finite ordered Ground of source-
observed positions to the final source observation. The processor interprets
its task program and its encoded evaluator (bounded depth two). This is
neither an execution of VL-I's floating-point/neural law through NCR-002 nor
a proof of future physical/environmental behavior.

NCR-002 is optional and should be installed at a pinned reviewed commit.
"""
from __future__ import annotations

from .ground import _digest, _hex64, _require
from .temporal_intelligence import (
    TemporalIntelligenceGround, verify_temporal_snapshot,
)
from .evolving_environment_language import (
    EvolvingEnvironmentLanguage, verify_environment_language,
    verify_language_migration,
)

BRIDGE = "vl-i.source-observed-reflective-processor.v1"
SUCCESSION = "vl-i.reflective-processor-succession.v1"
MAX_BRIDGE_SAMPLES = 8


def _ncr():
    try:
        from navigable_compute.invariant_space import InvariantDescription
        from navigable_compute.self_space import build_endogenous_processor
        from navigable_compute.self_verify import verify_endogenous_processor
        from navigable_compute.reflective_architecture import (
            construct_reflective_architecture, architecture_run,
            encode_source,
        )
        from navigable_compute.reflective_architecture_verify import (
            verify_architecture, verify_architecture_execution,
        )
    except ImportError as exc:
        raise RuntimeError("NCR-002 requires installed pinned Navigable-Compute") from exc
    return (InvariantDescription, build_endogenous_processor,
            verify_endogenous_processor, construct_reflective_architecture,
            architecture_run, encode_source, verify_architecture,
            verify_architecture_execution)


def _semantic_ground(capsule, n):
    """A time-position geometry derived from source order, NOT model dynamics.

    The intention (reach last verified observation), L1 adjacency, identity
    symmetry and preserved label are explicitly supplied observer invariants.
    No trained model forward pass or physical effect is claimed.
    """
    InvariantDescription, *_ = _ncr()
    return InvariantDescription(
        coordinates=tuple((i,0) for i in range(n)),
        symmetry=tuple(range(n)),
        labels=tuple("source-observed:"+capsule["family"] for _ in range(n)),
        initial=tuple(range(n)),
        goal=(n-1,),
        forbidden=(),
        max_displacement=1,
    )


def construct_reflective_environment_processor(
    snapshot, capsule, *, expected_head, expected_identity,
    representation_coordinate=0,
):
    """Prove task-ground construction + same-language interpreter reflection."""
    _require(type(representation_coordinate) is int and
             0 <= representation_coordinate < 120,
             "Unknown reflective architecture coordinate")
    verify_environment_language(snapshot, capsule,
                 expected_head=expected_head,expected_identity=expected_identity)
    rows=snapshot["events"][capsule["epoch_first"]:capsule["epoch_last"]+1]
    n=len(rows)
    _require(3<=n<=MAX_BRIDGE_SAMPLES,
             "Constructive reflection requires 3..8 genuine source observations")
    (InvariantDescription, build_endogenous_processor,
     verify_endogenous_processor, construct_reflective_architecture,
     architecture_run, encode_source, verify_architecture,
     verify_architecture_execution)=_ncr()
    desc=_semantic_ground(capsule,n)
    machine=build_endogenous_processor(desc, generator_limit=16,
                                       max_operators=128)
    _require(verify_endogenous_processor(desc,machine),
             "Source-derived mathematical processor not independently qualified")
    design=construct_reflective_architecture(
        machine, representation_coordinate=representation_coordinate)
    _require(verify_architecture(machine,design),
             "Constructed self-interpreter failed independent semantic verification")
    _require(machine.generator_opcodes,
             "Task invariant geometry produced no executable law")
    word=(machine.generator_opcodes[0],)*(n-1)
    results=[]
    for depth in (0,1,2):
        receipt=architecture_run(machine,design,word,
                                 machine.states[0],depth=depth)
        _require(verify_architecture_execution(machine,design,receipt),
                 "Independent stack evaluator rejected reflected execution")
        _require(receipt["result_state"]==n-1,
                 "Constructed processor failed target source position")
        results.append(receipt)
    _require(results[0]["result_state"]==results[1]["result_state"]==
             results[2]["result_state"],"Self-interpretation altered task meaning")

    last=rows[-1]
    # The observational value is *retrieved from the authenticated-by-pin
    # source-ledger transcript*, not inferred by the NCI navigation machine.
    body={
        "schema":BRIDGE,
        "source_identity":expected_identity,
        "source_head":expected_head,
        "environment_capsule":capsule["capsule_digest"],
        "source_witnesses":[row["digest"] for row in rows],
        "source_observation_count":n,
        "qualified_family":capsule["family"],
        "declared_intention":"reach-final-recorded-source-position",
        "geometry":"index-L1-adjacency-identity-symmetry",
        "processor_ground_digest":machine.source.digest,
        "generated_algebra_digest":machine.digest,
        "generated_interpreter_digest":design.digest,
        "representation_coordinate":representation_coordinate,
        "primitive_seed":list(design.primitive_seed),
        "task_program":list(word),
        "result_source_index":capsule["epoch_last"],
        "result_recorded_time":last["time"],
        "result_recorded_value":last["values"][0],
        "execution_receipts":results,
        "source_observation_claim":"EXACT_RECORDED_SOURCE_COORDINATE",
        "environment_dynamics_executed":"NOT_CLAIMED",
        "physical_effect_performed":False,
        "bootstrap_generated_without_semantic_contract":False,
        "task_semantics_source":"NCI_NCS_NCR002_INDEPENDENTLY_CHECKED",
    }
    return {**body,"bridge_digest":_digest(BRIDGE,body)}


def verify_reflective_environment_processor(
    snapshot,capsule,bridge,*,expected_head,expected_identity,
):
    """Independent source + NCI/NCS/NCR verification and exact receipt replay."""
    _require(type(bridge) is dict and bridge.get("schema")==BRIDGE and
             type(bridge.get("representation_coordinate")) is int,
             "Malformed claimed reflective processor")
    expected=construct_reflective_environment_processor(
        snapshot,capsule,expected_head=expected_head,
        expected_identity=expected_identity,
        representation_coordinate=bridge["representation_coordinate"])
    _require(bridge==expected,"Claimed reflective processor cannot be reconstructed")
    return True


def certify_reflective_interpreter_succession(
    old_snapshot,old_capsule,old_bridge,new_snapshot,new_capsule,new_bridge,
    migration,*,old_head,new_head,identity,
):
    """Two proven reflective processors linked by a proven unmodified history.

    No claim is made that their operators are isomorphic, their model
    trajectories coincide, or either predicts the external environment.
    """
    verify_language_migration(old_snapshot,old_capsule,new_snapshot,new_capsule,
                              migration,before_head=old_head,after_head=new_head,
                              identity=identity)
    verify_reflective_environment_processor(
        old_snapshot,old_capsule,old_bridge,
        expected_head=old_head,expected_identity=identity)
    verify_reflective_environment_processor(
        new_snapshot,new_capsule,new_bridge,
        expected_head=new_head,expected_identity=identity)
    body={"schema":SUCCESSION,"identity":identity,
          "old_source_head":old_head,"new_source_head":new_head,
          "old_reflective_processor":old_bridge["bridge_digest"],
          "new_reflective_processor":new_bridge["bridge_digest"],
          "language_migration_digest":migration["migration_digest"],
          "old_interpreter":old_bridge["generated_interpreter_digest"],
          "new_interpreter":new_bridge["generated_interpreter_digest"],
          "preservation":"ONLY_SOURCE_OBSERVED_PREFIX",
          "future_behavior_equivalence":"NOT_ESTABLISHED",
          "reflective_execution":"VERIFIED_BOUNDED_DEPTH_TWO",
          "physical_environment_execution":"NOT_ESTABLISHED"}
    return {**body,"succession_digest":_digest(SUCCESSION,body)}


def verify_reflective_interpreter_succession(
    old_snapshot,old_capsule,old_bridge,new_snapshot,new_capsule,new_bridge,
    migration,succession,*,old_head,new_head,identity,
):
    expected=certify_reflective_interpreter_succession(
        old_snapshot,old_capsule,old_bridge,
        new_snapshot,new_capsule,new_bridge,migration,
        old_head=old_head,new_head=new_head,identity=identity)
    _require(succession==expected,
             "Reflected interpreter succession is not independently reconstructable")
    return True
