"""TIME-003: construct a changing finite interpreter vocabulary from observations.

A finite, explicitly supplied candidate family can generate DIFFERENT executable
expression trees (affine, geometric, quadratic, order-two recurrence). The
environment supplies coefficients and determines which language qualifies.
This is not unrestricted language invention, neural training or physical proof.

TIME-001 retains immutable sample evidence and independently pinned lineage.
TIME-002 can own the same Ground in one live process. This module never
confuses a sample with an inferred continuation.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

from .ground import _digest, _hex64, _require
from .temporal_intelligence import (
    TemporalIntelligenceGround, _encode, _fraction, verify_temporal_snapshot,
)

CAPSULE = "vl-i.environment-language-capsule.v1"
MIGRATION = "vl-i.environment-language-migration.v1"
ANSWER = "vl-i.environment-language-answer.v1"
MAX_SAMPLES = 128
MAX_STEP = 512
FAMILIES = ("affine", "geometric", "quadratic", "recurrence2")
STATUS = "EMPIRICAL_FIT_WITH_HELDOUT_SAMPLE_NOT_A_UNIVERSAL_LAW"


def _rows(events: list[dict]) -> tuple[tuple[Fraction, Fraction], ...]:
    _require(len(events) <= MAX_SAMPLES, "Language-construction sample budget exhausted")
    result = []
    for event in events:
        _require(type(event) is dict and type(event.get("values")) is list
                 and len(event["values"]) == 1, "Univariate observation required")
        result.append((_fraction(event["time"]), _fraction(event["values"][0])))
    return tuple(result)


def _is_regular(rows) -> bool:
    if len(rows) < 2:
        return True
    dt = rows[1][0] - rows[0][0]
    return dt > 0 and all(rows[j][0] - rows[j-1][0] == dt
                          for j in range(1, len(rows)))


def _inference(rows):
    """Fit a bounded family and hold out at least one real sample.

    Family precedence is a declared minimum-complexity prior, not a proof that
    this language is the unique or best explanation in all possible languages.
    """
    n = len(rows)
    if n < 3 or not _is_regular(rows):
        return None
    ys = [row[1] for row in rows]
    x, y = ys[:2]
    d = y-x
    if all(ys[k] == x+k*d for k in range(n)):
        return ("affine", (x,d), 2)
    if x and all(ys[k] == x*(y/x)**k for k in range(n)):
        return ("geometric", (x,y/x), 2)
    if n >= 4:
        second = ys[2]-2*y+x
        if all(ys[k] == x+k*d+Fraction(k*(k-1),2)*second for k in range(n)):
            return ("quadratic", (x,d,second), 3)
    if n >= 5:
        a0,a1,a2,a3 = ys[:4]
        determinant = a1*a1-a0*a2
        if determinant:
            a = (a2*a1-a0*a3)/determinant
            b = (a1*a3-a2*a2)/determinant
            predecessor, current = a0,a1
            good = True
            for actual in ys[2:]:
                nxt = a*current+b*predecessor
                if actual != nxt:
                    good = False
                    break
                predecessor,current = current,nxt
            if good:
                return ("recurrence2", (a0,a1,a,b), 4)
    return None


def _segment(start, end, fitted):
    return {"first":start, "last":end,
            "status":"QUALIFIED" if fitted else "UNQUALIFIED",
            "family":fitted[0] if fitted else None,
            "coefficients":([_encode(q) for q in fitted[1]] if fitted else None),
            "training_samples":fitted[2] if fitted else 0,
            "heldout_samples":end-start+1-fitted[2] if fitted else 0}


def _segments(rows):
    """Deterministic online segmentation using only each observed prefix.

    An established law is frozen at its last supporting observation once a
    counterexample appears. Unqualified startup windows get a finite budget;
    source evidence is never deleted, even if no candidate law is admitted.
    """
    out = []
    start = 0
    chosen = None
    for end in range(1,len(rows)+1):
        subset = rows[start:end]
        next_fit = _inference(subset)
        if chosen and next_fit is None:
            out.append(_segment(start,end-2,chosen))
            start = end-1
            chosen = None
            continue
        if next_fit:
            chosen = next_fit
        elif end-start >= 5:
            out.append(_segment(start,end-2,None))
            start = end-1
    if start < len(rows):
        out.append(_segment(start,len(rows)-1,chosen))
    return out


def _const(q):
    return ["CONST",_encode(q)]


def _expression(family, coefficients):
    qs = tuple(_fraction(x) for x in coefficients)
    t = ["STEP"]
    if family == "affine":
        x,d = qs
        return ["ADD",_const(x),["MUL",_const(d),t]]
    if family == "geometric":
        x,r = qs
        return ["MUL",_const(x),["POW",_const(r),t]]
    if family == "quadratic":
        x,d,second = qs
        return ["ADD",["ADD",_const(x),["MUL",_const(d),t]],
                ["MUL",_const(second),["CHOOSE2",t]]]
    if family == "recurrence2":
        x0,x1,a,b = qs
        return ["RECURRENCE2",_const(x0),_const(x1),_const(a),_const(b),t]
    raise ValueError("Unknown environment-generated expression language")


def _vocabulary(program):
    if not isinstance(program,list):
        raise ValueError("Malformed generated AST")
    return tuple(sorted({program[0]} | {op for p in program[1:]
                         if isinstance(p,list) for op in _vocabulary(p)}))


def _matrix_mul(a,b):
    return (
        (a[0][0]*b[0][0]+a[0][1]*b[1][0],
         a[0][0]*b[0][1]+a[0][1]*b[1][1]),
        (a[1][0]*b[0][0]+a[1][1]*b[1][0],
         a[1][0]*b[0][1]+a[1][1]*b[1][1]),
    )


def _recurrence_jump(x0,x1,a,b,k):
    if k==0:return x0,0
    if k==1:return x1,0
    mat = ((a,b),(Fraction(1),Fraction(0)))
    acc = ((Fraction(1),Fraction(0)),(Fraction(0),Fraction(1)))
    products = 0
    power = k-1
    while power:
        if power & 1:
            acc = _matrix_mul(acc,mat)
            products += 1
        power >>= 1
        if power:
            mat = _matrix_mul(mat,mat)
            products += 1
    return acc[0][0]*x1+acc[0][1]*x0,products


def _interpret(program, index, *, fuel=24):
    """Fixed trusted arithmetic bootstrap, not an auto-generated interpreter."""
    _require(type(program) is list and program and type(index) is int
             and 0 <= index <= MAX_STEP and fuel > 0,
             "Unadmitted program or temporal coordinate")
    tag=program[0]
    if tag=="STEP":
        _require(len(program)==1, "Malformed STEP")
        return Fraction(index),0
    if tag=="CONST":
        _require(len(program)==2, "Malformed CONST")
        return _fraction(program[1]),0
    if tag in ("ADD","MUL","POW"):
        _require(len(program)==3, "Malformed binary expression")
        a,ca=_interpret(program[1],index,fuel=fuel-1)
        b,cb=_interpret(program[2],index,fuel=fuel-1)
        if tag=="ADD": return a+b,ca+cb+1
        if tag=="MUL": return a*b,ca+cb+1
        _require(b.denominator==1 and 0<=b<=MAX_STEP,
                 "Exponent outside exact navigation scope")
        return a**int(b),ca+cb+int(b).bit_length()
    if tag=="CHOOSE2":
        _require(len(program)==2, "Malformed binomial operator")
        k,c=_interpret(program[1],index,fuel=fuel-1)
        _require(k.denominator==1 and k>=0,"Noninteger binomial coordinate")
        return k*(k-1)/2,c+3
    if tag=="RECURRENCE2":
        _require(len(program)==6, "Malformed recurrence Ground")
        params=[_interpret(x,index,fuel=fuel-1) for x in program[1:]]
        x0,x1,a,b,n=(pair[0] for pair in params)
        _require(n.denominator==1,"Recurrence requires discrete observation index")
        result,products=_recurrence_jump(x0,x1,a,b,int(n))
        return result,sum(pair[1] for pair in params)+products*8+2
    raise ValueError("Unadmitted interpreter opcode")


def _reference(family, coefficients, k):
    """A non-AST independent semantic oracle; recurrence is linear replay."""
    q=tuple(_fraction(x) for x in coefficients)
    if family=="affine":return q[0]+k*q[1]
    if family=="geometric":return q[0]*(q[1]**k)
    if family=="quadratic":return q[0]+k*q[1]+Fraction(k*(k-1),2)*q[2]
    if family=="recurrence2":
        p,c,a,b=q
        if k==0:return p
        for _ in range(1,k):
            p,c=c,a*c+b*p
        return c
    raise ValueError("Unknown independent semantic family")


class EvolvingEnvironmentLanguage:
    """An interpretation view over ONE retained TIME-001 / TIME-002 Ground."""

    def __init__(self, ground: TemporalIntelligenceGround):
        _require(isinstance(ground,TemporalIntelligenceGround)
                 and ground.contract["dimensions"]==1,
                 "A source-qualified, univariate TIME-001 Ground is required")
        self.ground=ground

    def segments(self):
        return deepcopy(_segments(_rows(self.ground.events)))

    def current(self, *, expected_head):
        _require(expected_head==self.ground.head and _hex64(expected_head),
                 "Stale interpreter source head")
        rows=_rows(self.ground.events)
        _require(rows,"No observations to construct a language")
        segments=_segments(rows)
        last=segments[-1]
        if last["status"]!="QUALIFIED":
            return None
        first,last_index=last["first"],last["last"]
        interval=rows[first:last_index+1]
        _require(_is_regular(interval) and last_index-first+1>=3,
                 "Irregular source time; no executable law")
        program=_expression(last["family"],last["coefficients"])
        body={"schema":CAPSULE,"environment_identity":self.ground.identity,
              "source_head":self.ground.head,"source_event_count":len(rows),
              "epoch_first":first,"epoch_last":last_index,
              "time_origin":_encode(interval[0][0]),
              "time_step":_encode(interval[1][0]-interval[0][0]),
              "family":last["family"],
              "coefficients":last["coefficients"],
              "program":program,"vocabulary":list(_vocabulary(program)),
              "training_samples":last["training_samples"],
              "heldout_samples":last["heldout_samples"],
              "source_witnesses":[event["digest"] for event in
                                  self.ground.events[first:last_index+1]],
              "claim":STATUS}
        return {**body,"capsule_digest":_digest(CAPSULE,body)}

    def navigate(self, capsule, time: str, *, expected_head, allow_projection=False):
        _require(capsule==self.current(expected_head=expected_head),
                 "Unqualified or stale generated interpreter")
        _require(type(time) is str, "Canonical temporal coordinate required")
        requested=_fraction(time)
        origin=_fraction(capsule["time_origin"])
        step=_fraction(capsule["time_step"])
        n=(requested-origin)/step
        _require(n.denominator==1 and 0<=n<=MAX_STEP,
                 "Only bounded forward integer sample coordinates are supported")
        k=int(n)
        last=capsule["epoch_last"]-capsule["epoch_first"]
        if k>last:
            _require(allow_projection is True,
                     "Prediction outside observed interval requires opt-in")
            status="UNVERIFIED_FUTURE_PROJECTION"
        else:
            status="RECORDED_SOURCE_COORDINATE"
        value,ops=_interpret(capsule["program"],k)
        _require(value==_reference(capsule["family"],capsule["coefficients"],k),
                 "Generated interpreter disagrees with independent reference")
        if k<=last:
            source=self.ground.events[capsule["epoch_first"]+k]
            _require(source["time"]==time and source["values"]==[_encode(value)],
                     "Generated architecture disagrees with witnessed sample")
        body={"schema":ANSWER,"environment_identity":self.ground.identity,
              "source_head":self.ground.head,
              "capsule_digest":capsule["capsule_digest"],"time":time,
              "value":_encode(value),"status":status,"counted_operations":ops,
              "expanded_source_transitions":0,
              "source_sample_at_coordinate": k<=last}
        return {**body,"answer_digest":_digest(ANSWER,body)}


def verify_environment_language(snapshot,capsule,*,expected_head,expected_identity):
    """Rebuild candidate selection and validate all held-out source witnesses."""
    verify_temporal_snapshot(snapshot,expected_head=expected_head,
                             expected_identity=expected_identity)
    g=TemporalIntelligenceGround.restore(snapshot,expected_head=expected_head,
                                         expected_identity=expected_identity)
    actual=EvolvingEnvironmentLanguage(g).current(expected_head=expected_head)
    _require(type(capsule) is dict and actual is not None and capsule==actual,
             "Interpreter capsule cannot be regenerated from source evidence")
    segment=_rows(g.events)[capsule["epoch_first"]:capsule["epoch_last"]+1]
    _require(capsule["heldout_samples"] >= 1, "No independent sample to challenge model")
    for k,(_,observed) in enumerate(segment):
        _require(_reference(capsule["family"],capsule["coefficients"],k)==observed,
                 "Independent mathematical verifier rejects interpreter")
        result,_=_interpret(capsule["program"],k)
        _require(result==observed, "Executable language contradicts source")
    return True


def verify_environment_answer(snapshot,capsule,answer,*,expected_head,expected_identity):
    verify_environment_language(snapshot,capsule,expected_head=expected_head,
                                expected_identity=expected_identity)
    g=TemporalIntelligenceGround.restore(snapshot,expected_head=expected_head,
                                         expected_identity=expected_identity)
    candidate=EvolvingEnvironmentLanguage(g).navigate(
        capsule,answer["time"],expected_head=expected_head,
        allow_projection=answer.get("status")=="UNVERIFIED_FUTURE_PROJECTION")
    _require(candidate==answer,"Independent answer reconstruction disagrees")
    return True


def certify_language_migration(before_snapshot,before_capsule,after_snapshot,
                               after_capsule,*,before_head,after_head,identity):
    """Preserve historical evidence, never assert incompatible models equivalent.

    The two externally trusted snapshot heads must be independently obtained,
    not copied from the checkpoints. A source prefix is a necessary condition
    for continuation, not proof of network authentication or unbroken uptime.
    """
    verify_environment_language(before_snapshot,before_capsule,
                   expected_head=before_head,expected_identity=identity)
    verify_environment_language(after_snapshot,after_capsule,
                   expected_head=after_head,expected_identity=identity)
    old=before_snapshot["events"]
    new=after_snapshot["events"]
    _require(len(new)>len(old) and new[:len(old)]==old
             and before_head==old[-1]["digest"],
             "Interpretation migration has rewritten or truncated observed history")
    _require(before_capsule["capsule_digest"]!=after_capsule["capsule_digest"],
             "Require a genuinely new source-qualified interpreter capsule")
    body={"schema":MIGRATION,"identity":identity,
          "before_head":before_head,"after_head":after_head,
          "before_interpreter":before_capsule["capsule_digest"],
          "after_interpreter":after_capsule["capsule_digest"],
          "before_family":before_capsule["family"],
          "after_family":after_capsule["family"],
          "retained_source_events":len(old),
          "new_source_events":len(new)-len(old),
          "historical_source_prefix_exact":True,
          "preserved_claim":"PREVIOUSLY_OBSERVED_SAMPLE_SCOPE_ONLY",
          "future_behavioral_equivalence":"NOT_ESTABLISHED",
          "bootstrap_self_hosting":"NOT_DEMONSTRATED"}
    return {**body,"migration_digest":_digest(MIGRATION,body)}


def verify_language_migration(before_snapshot,before_capsule,after_snapshot,
                              after_capsule,migration,*,before_head,after_head,identity):
    expected=certify_language_migration(
        before_snapshot,before_capsule,after_snapshot,after_capsule,
        before_head=before_head,after_head=after_head,identity=identity)
    _require(migration==expected,"Migration receipt cannot be independently reconstructed")
    return True
