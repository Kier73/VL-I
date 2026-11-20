"""TIME-007: navigate a law-generated intelligence state-space by intention.

Four TIME-003 observational models are transformed into exact rational
finite-dimensional linear systems; a requested time is a *coordinate* of the
mathematical state, realized by binary composition of a small operator.
No trajectory table or per-time-step state is constructed.

A separate explicit source-level interpreter (reflective_rational.py)
can realize the same navigation, including bounded interpretation of its
own encoded source. All exact rational host primitives remain declared.

Not a neural intelligence, physical predictor, unbounded proof or
authenticated multi-host quorum. The signed publisher is still an external
source of value claims, not objective physical truth.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256

from .ground import _digest, _require
from .temporal_intelligence import _fraction, _encode
from .evolving_environment_language import (
    _reference, verify_environment_language,
)
from .authenticated_temporal_source import (
    verified_source_journal,_check_checkpoint_against_signed_source,
)

SCHEMA="vl-i.intention-derived-rational-state-space.v1"
NAVIGATION="vl-i.intention-rational-navigation.v1"
MAX_TARGET=1_048_576
MAX_INTEGER_BITS=4096
MAX_QOPS=1500
FAMILY_DIMENSIONS={"affine":2,"geometric":1,"quadratic":3,"recurrence2":2}


def _bounded(value):
    _require(type(value) is Fraction and
             value.numerator.bit_length()<=MAX_INTEGER_BITS and
             value.denominator.bit_length()<=MAX_INTEGER_BITS,
             "Exact state coefficient exceeds finite bit budget")
    return value


def _q(s):
    return _bounded(_fraction(s))


def _encode_vec(a):
    return [_encode(_bounded(x)) for x in a]


def _encode_mat(a):
    return [_encode_vec(row) for row in a]


def _realize(family, coeffs):
    """Construct a minimal *declared* family state, not claim global minimality.

    Source-derived coefficients instantiate a known class of invariant
    state-space realizations, and the output coordinate is explicitly typed.
    """
    c=tuple(_q(v) for v in coeffs)
    one,zero=Fraction(1),Fraction(0)
    if family=="affine":
        _require(len(c)==2,"Affine model requires origin and first difference")
        x,d=c
        return ((one,one),(zero,one)),(x,d),0
    if family=="geometric":
        _require(len(c)==2,"Geometric model requires origin and multiplier")
        x,r=c
        return ((r,),),(x,),0
    if family=="quadratic":
        _require(len(c)==3,"Quadratic model requires second difference")
        x,d,second=c
        return ((one,one,zero),
                (zero,one,one),
                (zero,zero,one)),(x,d,second),0
    if family=="recurrence2":
        _require(len(c)==4,"Order-2 relation requires two seeds and a,b")
        x0,x1,a,b=c
        return ((a,b),(one,zero)),(x1,x0),1
    raise ValueError("No admitted environmental state-space grammar")


def _mul_matrices(A,B,metrics):
    d=len(A)
    _require(d==len(B) and all(len(row)==d for row in A+B),
             "Wrong matrix state dimensionality")
    result=[]
    for i in range(d):
        row=[]
        for j in range(d):
            total=Fraction(0)
            for k in range(d):
                total=_bounded(total+_bounded(A[i][k]*B[k][j]))
                metrics["rational_multiplications"]+=1
                metrics["rational_additions"]+=1
            row.append(total)
        result.append(tuple(row))
    return tuple(result)


def _mul_vector(A,V,metrics):
    d=len(V)
    result=[]
    for i in range(d):
        total=Fraction(0)
        for j in range(d):
            total=_bounded(total+_bounded(A[i][j]*V[j]))
            metrics["rational_multiplications"]+=1
            metrics["rational_additions"]+=1
        result.append(total)
    return tuple(result)


def _identity(d):
    return tuple(tuple(Fraction(int(i==j)) for j in range(d))
                 for i in range(d))


def navigate_relation(A,V,step,output_index):
    """O(log step) matrix compositions, O(1) retained power-state matrices.

    Counts exact field operations; numerator/denominator bit work is reported
    separately as a restriction, not asserted constant-time.
    """
    _require(type(step) is int and 0<=step<=MAX_TARGET,
             "Unadmitted intention coordinate")
    d=len(V)
    _require(1<=d<=3 and type(output_index) is int and
             0<=output_index<d and len(A)==d,
             "Invalid constructed state-space dimension")
    metrics={"matrix_products":0,"rational_multiplications":0,
             "rational_additions":0,"binary_bits":step.bit_length(),
             "expanded_temporal_transitions":0,
             "trajectory_samples_materialized":0}
    exponent=step
    power=A
    acc=_identity(d)
    while exponent:
        if exponent & 1:
            acc=_mul_matrices(acc,power,metrics)
            metrics["matrix_products"]+=1
        exponent>>=1
        if exponent:
            power=_mul_matrices(power,power,metrics)
            metrics["matrix_products"]+=1
    vector=_mul_vector(acc,V,metrics)
    metrics["largest_output_numerator_bits"]=max(
        v.numerator.bit_length() for v in vector)
    metrics["largest_output_denominator_bits"]=max(
        v.denominator.bit_length() for v in vector)
    return vector[output_index],vector,metrics


def _qualify(snapshot,capsule,*,journal_path,source_id,public_key,
             expected_source_head,expected_source_count,
             expected_ground_head,expected_ground_identity):
    packets=verified_source_journal(
        journal_path,expected_head=expected_source_head,
        expected_count=expected_source_count,source=source_id,
        public_key=public_key)
    _require(len(packets)==len(snapshot.get("events",[])),
             "Mathematical Ground not covered by signed source ledger")
    _check_checkpoint_against_signed_source(snapshot,packets,source_id)
    verify_environment_language(
        snapshot,capsule,expected_head=expected_ground_head,
        expected_identity=expected_ground_identity)
    _require(capsule["source_head"]==expected_ground_head,
             "Source head changed during law construction")
    return packets


def construct_intention_space(
    snapshot,capsule,*,journal_path,source_id,public_key,
    expected_source_head,expected_source_count,
    expected_ground_head,expected_ground_identity,
):
    _qualify(snapshot,capsule,journal_path=journal_path,source_id=source_id,
             public_key=public_key,expected_source_head=expected_source_head,
             expected_source_count=expected_source_count,
             expected_ground_head=expected_ground_head,
             expected_ground_identity=expected_ground_identity)
    matrix,seed,output=_realize(capsule["family"],capsule["coefficients"])
    d=len(seed)
    origin=_q(capsule["time_origin"])
    delta=_q(capsule["time_step"])
    _require(delta>0,"Source time coordinate must advance")
    # Validate all recorded source values with the new algebra, INDEPENDENT
    # of the original TIME-003 AST or its compiled expression tree.
    first,last=capsule["epoch_first"],capsule["epoch_last"]
    total_ops=0
    for k,event in enumerate(snapshot["events"][first:last+1]):
        val,_,metrics=navigate_relation(matrix,seed,k,output)
        total_ops+=metrics["rational_multiplications"]+metrics["rational_additions"]
        _require(_encode(_bounded(origin+k*delta))==event["time"]
                 and _encode(val)==event["values"][0]
                 and val==_reference(capsule["family"],capsule["coefficients"],k),
                 "Constructed mathematical space contradicts a source witness")
    body={
        "schema":SCHEMA,
        "source_id":source_id,
        "signed_source_journal_head":expected_source_head,
        "source_public_key_sha256":sha256(bytes.fromhex(public_key)).hexdigest(),
        "signed_source_count":expected_source_count,
        "ground_identity":expected_ground_identity,
        "ground_head":expected_ground_head,
        "capsule_digest":capsule["capsule_digest"],
        "family":capsule["family"],
        "time_origin":capsule["time_origin"],
        "time_step":capsule["time_step"],
        "source_observation_count":last-first+1,
        "source_witnesses":[row["digest"] for row in
             snapshot["events"][first:last+1]],
        "transition_operator":_encode_mat(matrix),
        "initial_state":_encode_vec(seed),
        "output_coordinate":output,
        "dimension":d,
        "attested_sample_reverification_qops":total_ops,
        "model_family_qualified":"SOURCE_HELDOUT_EMPIRICAL_NOT_PHYSICAL_LAW",
        "representation":"EXACT_LINEAR_STATE_SPACE_RELATION",
        "intermediate_trajectory_enumerated":False,
        "source_observation_between_samples":"NOT_ESTABLISHED",
        "unbounded_intelligence":"NOT_ESTABLISHED",
    }
    return {**body,"intention_space_digest":_digest(SCHEMA,body)}


def _compiled(space):
    M=tuple(tuple(_q(v) for v in row)
            for row in space["transition_operator"])
    V=tuple(_q(v) for v in space["initial_state"])
    _require(len(M)==len(V)==space["dimension"],
             "Unqualified mathematical state-space encoding")
    return M,V


def navigate_intention(space,requested_time,*,allow_projection=False,
                       reflect=False):
    """Navigate the immediate requested state; refuse unsupported futures."""
    _require(type(space) is dict and space.get("schema")==SCHEMA and
             type(requested_time) is str and
             space.get("intention_space_digest")==_digest(SCHEMA,{
                 k:v for k,v in space.items() if k!="intention_space_digest"}),
             "Malformed or tampered source-derived Ground")
    requested=_q(requested_time)
    start=_q(space["time_origin"])
    interval=_q(space["time_step"])
    index=(requested-start)/interval
    _require(index.denominator==1 and 0<=index<=MAX_TARGET,
             "Intention coordinate must be aligned and bounded")
    k=int(index)
    recorded=space["source_observation_count"]
    if k>=recorded:
        _require(allow_projection is True,
                 "Future intent requires explicit unverified projection consent")
    A,V=_compiled(space)
    value,state,metrics=navigate_relation(A,V,k,space["output_coordinate"])
    execution={"method":"COMPOSE_STATE_SPACE_RELATION",
               "state_vector":_encode_vec(state),
               "navigation_metrics":metrics}
    if reflect:
        from .reflective_rational import reflect_intention
        execution["source_reflection"]=reflect_intention(
            space,k,expected_value=_encode(value))
    body={
        "schema":NAVIGATION,
        "intention_space_digest":space["intention_space_digest"],
        "requested_time":requested_time,
        "logical_step_coordinate":k,
        "value":_encode(value),
        "evidence_status":(
            "EXACT_RECORDED_SOURCE_COORDINATE" if k<recorded
            else "UNVERIFIED_FUTURE_MODEL_PROJECTION"),
        "calculation":execution,
        "expanded_source_trajectory_steps":0,
        "learned_law_is_physical_truth":False,
    }
    return {**body,"navigation_digest":_digest(NAVIGATION,body)}


def verify_intention_navigation(
    snapshot,capsule,space,result,*,journal_path,source_id,public_key,
    expected_source_head,expected_source_count,
    expected_ground_head,expected_ground_identity,
):
    rebuilt=construct_intention_space(
        snapshot,capsule,journal_path=journal_path,source_id=source_id,
        public_key=public_key,expected_source_head=expected_source_head,
        expected_source_count=expected_source_count,
        expected_ground_head=expected_ground_head,
        expected_ground_identity=expected_ground_identity)
    _require(rebuilt==space,"State-space was not constructed from qualified source")
    _require(type(result) is dict and
             result.get("schema")==NAVIGATION and
             type(result.get("requested_time")) is str,
             "Unqualified navigated intention")
    same=navigate_intention(
        space,result["requested_time"],
        allow_projection=result.get("evidence_status")==
                         "UNVERIFIED_FUTURE_MODEL_PROJECTION",
        reflect=bool(result.get("calculation",{}).get("source_reflection")))
    _require(same==result,
             "Intention response not independently reconstructable")
    return True
