"""TIME-006: compile a source-derived numerical law into a reflective task CPU.

The mathematical state of the finite machine is an exact rational dynamical
state (time, current value, prior value) derived from an independently signed
TIME-005 source ledger and a qualified TIME-003 law. The mathematical
difference equation generates a bounded transition table and binary
jump vocabulary. The real NCR-002 constructed evaluator executes that
compiled table at reflection depth 0/1/2.

This is an extensional finite model processor: arithmetic happens when
constructing/qualifying the transition table; the reflective interpreter
executes its verified semantic transition table, not primitive unbounded
Fraction arithmetic. All future states are labeled UNVERIFIED projections.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256

from .ground import _canonical, _digest, _require
from .temporal_intelligence import _fraction, _encode
from .evolving_environment_language import (
    EvolvingEnvironmentLanguage,_interpret,_reference,
    verify_environment_language,
)
from .authenticated_temporal_source import (
    verified_source_journal, _packet_digest, _check_checkpoint_against_signed_source,
)
from .reflective_environment_processor import _ncr, _semantic_ground

SCHEMA="vl-i.authenticated-numerical-reflective-processor.v1"
MAX_HORIZON=12
MAX_DIGITS=110


def _safe_fraction(q):
    _require(isinstance(q,Fraction)
             and len(str(abs(q.numerator)))<=MAX_DIGITS
             and len(str(q.denominator))<=MAX_DIGITS,
             "Generated exact rational state exceeds finite construction budget")
    return _encode(q)


def _step_model(family,coeffs,k,current,previous):
    """Separate mathematical transition route, not the AST or closed form."""
    q=tuple(_fraction(x) for x in coeffs)
    if family=="affine":
        return current+q[1]
    if family=="geometric":
        return current*q[1]
    if family=="quadratic":
        return current+q[1]+k*q[2]
    if family=="recurrence2":
        _require(len(q)==4,"Malformed learned second-order recurrence")
        return q[1] if k==0 else q[2]*current+q[3]*previous
    raise ValueError("No numerical semantics for the qualified family")


def _compile_numeric_states(snapshot,capsule,horizon):
    """Construct exact finite trajectory via the actual difference equation.

    Forward source-model evaluation and independently written closed form
    MUST agree at each coordinate. A model's future output is not a source
    observation, even when both implementations calculate the same value.
    """
    first,last=capsule["epoch_first"],capsule["epoch_last"]
    observed=last-first+1
    _require(observed>=3 and observed-1<=horizon<=MAX_HORIZON,
             "Finite numerical horizon must include every qualified source sample")
    _require(type(capsule["coefficients"]) is list
             and capsule["family"] in (
                 "affine","geometric","quadratic","recurrence2"),
             "Unadmitted generated numeric model")
    origin=_fraction(capsule["time_origin"])
    delta=_fraction(capsule["time_step"])
    _require(delta>0,"Numerical source step must advance")
    coeffs=capsule["coefficients"]
    current=_fraction(coeffs[0])
    previous=None
    states=[]
    for k in range(horizon+1):
        oracle=_reference(capsule["family"],coeffs,k)
        _require(current==oracle, "Exact difference operator disagrees with algebraic oracle")
        # The separately generated TIME-003 AST also must realize each state.
        executable,_=_interpret(capsule["program"],k)
        _require(executable==current,"Generated original arithmetic AST disagrees")
        global_time=_safe_fraction(origin+k*delta)
        observed_here=k<observed
        if observed_here:
            event=snapshot["events"][first+k]
            _require(event["time"]==global_time and
                     event["values"]==[_safe_fraction(current)],
                     "Compiled finite numeric state contradicted source-signature history")
        status=("EXACT_RECORDED_SOURCE_STATE" if observed_here
                else "UNVERIFIED_FUTURE_MODEL_STATE")
        states.append({"step":k,"time":global_time,
                       "value":_safe_fraction(current),
                       "previous_value":_safe_fraction(previous) if previous is not None
                                         else None,
                       "status":status,
                       "source_witness":(snapshot["events"][first+k]["digest"]
                                         if observed_here else None)})
        if k<horizon:
            next_value=_step_model(capsule["family"],coeffs,k,current,previous)
            previous,current=current,next_value
    return states


def _make_table(states):
    """Finite compiled semantic ISA, powers-of-two derived from NEXT law.

    The terminal state absorbs extra steps by an explicitly declared finite
    horizon guard. This saturation is an interpreter budget, NOT physical law.
    """
    count=len(states)
    maxstep=count-1
    shifts=[]
    shift=1
    while shift<=maxstep:
        shifts.append(shift)
        shift*=2
    table=tuple(tuple(min(i+stride,maxstep) for i in range(count))
                for stride in shifts)
    # Independently check closure, every jump equals repeated NEXT and
    # every visited exact numerical state satisfies the source-derived law.
    for stride,row in zip(shifts,table):
        for position,target in enumerate(row):
            observed=position
            for _ in range(stride):
                observed=min(observed+1,maxstep)
            _require(target==observed and 0<=target<count,
                     "Generated navigation macro does not implement NEXT semantics")
    return table,tuple(shifts)


def _compile_word(destination,shifts):
    _require(type(destination) is int and 0<=destination<=MAX_HORIZON,
             "Unqualified source-derived target coordinate")
    remaining=destination
    result=[]
    for opcode in reversed(range(len(shifts))):
        if remaining>=shifts[opcode]:
            remaining-=shifts[opcode]
            result.append(opcode)
    _require(remaining==0,"No bounded binary navigation word")
    return tuple(result)


def _ncr_runtime(snapshot,capsule,states,table,word,representation_coordinate):
    """Execute compiled numerical state transitions under constructed NCR-002."""
    (_desc, build_endogenous_processor, verify_endogenous_processor,
     construct_reflective_architecture, _architecture_run,encode_source,
     verify_architecture,_verify_architecture_execution)=_ncr()
    from navigable_compute.reflective_ground import (
        compile_task_program,host_run,
    )
    from navigable_compute.reflective_verify import reference_run

    observed=capsule["epoch_last"]-capsule["epoch_first"]+1
    _require(3<=observed<=8,"NCR-002 source-ground adapter budget exceeded")
    desc=_semantic_ground(capsule,observed)
    machine=build_endogenous_processor(desc,generator_limit=16,
                                       max_operators=128)
    _require(verify_endogenous_processor(desc,machine),
             "Underlying task-derived processor algebra not independently checked")
    architecture=construct_reflective_architecture(
        machine,representation_coordinate=representation_coordinate)
    _require(verify_architecture(machine,architecture),
             "NCR-002 generated interpreter did not pass independent verification")
    task=compile_task_program(machine)
    argument=(word,0)
    input_1=(encode_source(task,architecture.syntax_codebook),
             "main",argument,table)
    input_2=(encode_source(architecture.interpreter,
                           architecture.syntax_codebook),
             "run",input_1,table)
    execution=[]
    for depth in (0,1,2):
        if depth==0:
            original,count=host_run(task,"main",argument,table)
            independent,n=reference_run(task,"main",argument,table)
        else:
            packed=input_1 if depth==1 else input_2
            original,count=host_run(
                architecture.interpreter,"run",packed,table,
                seed_primitives=architecture.primitive_seed)
            independent,n=reference_run(
                architecture.interpreter,"run",packed,table,
                seed_primitives=architecture.primitive_seed)
        _require(original==independent==len(states)-1 and count==n,
                 "Constructed reflective evaluator contradicts independent stack-machine oracle")
        execution.append({"reflection_depth":depth,
                          "result_state_index":original,
                          "expression_steps":count})
    naive,naive_cost=host_run(task,"main",((0,)*(len(states)-1),0),table)
    _require(naive==len(states)-1,"Naive numeric trace disagrees with generated jump machine")
    # The compiled table is more than a position-only navigator: each step
    # is checked against a rational difference equation and yields a typed
    # numeric state, including its full recurrence memory where needed.
    return {
        "nci_ground_digest":machine.source.digest,
        "ncs_semantic_algebra_digest":machine.digest,
        "ncr_reflective_interpreter_digest":architecture.digest,
        "ncr_external_host_primitive_seed":list(architecture.primitive_seed),
        "syntax_coordinate":representation_coordinate,
        "task_instruction_word":list(word),
        "depth_receipts":execution,
        "naive_unrolled_direct_steps":naive_cost,
        "compiled_state_count":len(states),
    }


def construct_authenticated_numerical_processor(
    snapshot,capsule,*,source_journal,source_id,public_key,
    expected_source_head,expected_source_count,
    expected_ground_head,expected_ground_identity,
    horizon,allow_projection=False,
):
    """Only an externally pinned signed source can admit the numeric Ground.

    Future model states require an explicit opt-in and are never returned with
    a source witness. Model arithmetic is compiled *extensionally*, so this
    does not demonstrate an unbounded self-generated arithmetic hardware ISA.
    """
    _require(type(horizon) is int and 0<=horizon<=MAX_HORIZON,
             "Finite source-conditioned machine target is not admitted")
    signed=verified_source_journal(
        source_journal,expected_head=expected_source_head,
        expected_count=expected_source_count,source=source_id,public_key=public_key)
    _require(len(signed)==len(snapshot.get("events",[])),
             "Signed source coverage is incomplete for the claimed numeric Ground")
    _check_checkpoint_against_signed_source(snapshot,signed,source_id)
    verify_environment_language(
        snapshot,capsule,expected_head=expected_ground_head,
        expected_identity=expected_ground_identity)
    _require(capsule["source_head"]==expected_ground_head,
             "Numerical processor source head diverged")
    recorded=capsule["epoch_last"]-capsule["epoch_first"]
    _require(horizon>=recorded,
             "Compiled numerical machine cannot omit qualifying source observations")
    if horizon>recorded:
        _require(allow_projection is True,
                 "Future numeric machine materialization requires explicit opt-in")

    states=_compile_numeric_states(snapshot,capsule,horizon)
    table,shifts=_make_table(states)
    word=_compile_word(horizon,shifts)
    # Gauge selection is source-dependent, NOT an optimization or proof of
    # minimal representation complexity.
    gauge=int(capsule["capsule_digest"],16)%120
    runtime=_ncr_runtime(snapshot,capsule,states,table,word,gauge)

    body={
        "schema":SCHEMA,
        "source_identity":source_id,
        "source_public_key_fingerprint":sha256(bytes.fromhex(public_key)).hexdigest(),
        "trusted_publisher_head":expected_source_head,
        "trusted_publisher_count":expected_source_count,
        "environment_identity":expected_ground_identity,
        "environment_head":expected_ground_head,
        "language_capsule_digest":capsule["capsule_digest"],
        "language_family":capsule["family"],
        "semantic_law_coefficients":list(capsule["coefficients"]),
        "finite_horizon":horizon,
        "jump_strides":list(shifts),
        "compiled_action_table":[list(row) for row in table],
        "compiled_numeric_states":states,
        "result_exact_value":states[-1]["value"],
        "result_time":states[-1]["time"],
        "result_evidence_status":states[-1]["status"],
        "all_recorded_samples_source_signed":True,
        "generated_semantic_processor":runtime,
        "numerical_semantics":"EXACT_RATIONAL_FORWARD_DIFFERENCE_TABLE",
        "runtime_arithmetic":"SOURCE_DERIVED_FINITE_TABLE_LOOKUP",
        "future_projection_is_source_observation":False,
        "terminal_absorption_is_physical_law":False,
        "unbounded_general_intelligence":"NOT_ESTABLISHED",
    }
    return {**body,"numerical_processor_digest":_digest(SCHEMA,body)}


def verify_authenticated_numerical_processor(
    snapshot,capsule,record,*,source_journal,source_id,public_key,
    expected_source_head,expected_source_count,
    expected_ground_head,expected_ground_identity,
):
    """Regenerate all numeric values, table actions, AST and 3 runtime depths."""
    _require(type(record) is dict and record.get("schema")==SCHEMA
             and type(record.get("finite_horizon")) is int,
             "Unadmitted signed numerical execution receipt")
    expected=construct_authenticated_numerical_processor(
        snapshot,capsule,source_journal=source_journal,
        source_id=source_id,public_key=public_key,
        expected_source_head=expected_source_head,
        expected_source_count=expected_source_count,
        expected_ground_head=expected_ground_head,
        expected_ground_identity=expected_ground_identity,
        horizon=record["finite_horizon"],
        allow_projection=record.get("result_evidence_status")==
                         "UNVERIFIED_FUTURE_MODEL_STATE")
    _require(expected==record,
             "Independent numerical compiler/evaluator rejects received processor")
    return True
