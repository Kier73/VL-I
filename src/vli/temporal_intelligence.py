"""TIME-001: an environment-derived interpreter evolving in causal, rational time.

The observed environment supplies exact timestamped vector samples. Each
three-or-more-point affine epoch yields an *empirical* executable law.
Contradicting observations cause a new interpretation epoch, never historical
rewriting. This is a finite laboratory model, not evidence that an unobserved
physical environment follows the inferred law between samples.

The generative program is data interpreted by a fixed minimal AST evaluator;
the evaluator is NOT generated and is NOT self-hosting. A trusted head outside
the checkpoint is required for lineage verification. Hashes are not signatures.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import re

from .ground import _digest, _hex64, _require

SCHEMA = "vl-i.temporal-environment-ground.v1"
EVENT = "vl-i.temporal-observation.v1"
SNAPSHOT = "vl-i.temporal-ground-snapshot.v1"
PROCESSOR = "vl-i.environment-derived-processor.v1"
NAVIGATION = "vl-i.temporal-navigation.v1"
GENESIS = "0" * 64
MAX_OBSERVATIONS = 2048
MAX_DIMENSIONS = 8
_NUM = re.compile(r"-?(?:0|[1-9][0-9]{0,17})/(?:[1-9][0-9]{0,11})\Z")


def _fraction(value: str) -> Fraction:
    """Only canonical, bounded rational coordinates enter the Ground."""
    _require(type(value) is str and _NUM.fullmatch(value) is not None,
             "Use a canonical, bounded rational n/d coordinate")
    n, d = value.split("/")
    q = Fraction(int(n), int(d))
    _require(f"{q.numerator}/{q.denominator}" == value,
             "Rational coordinate must be reduced and canonical")
    return q


def _encode(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _vectors(values, dimensions):
    _require(type(values) in (list, tuple) and len(values) == dimensions,
             "Wrong environmental observation dimension")
    return tuple(_fraction(x) for x in values)


def _derive_epochs(events):
    """Independently derive every epoch from the full unmodified sample stream.

    A mismatch gives a *detection interval*, NOT a known physical switch time.
    The first two points propose a rate; the third independent point qualifies
    it for interpolation over that observed span, not for universal prediction.
    """
    epochs = []
    for i, event in enumerate(events):
        t = _fraction(event["time"])
        v = tuple(_fraction(x) for x in event["values"])
        if not epochs:
            epochs.append({"first": i, "last": i, "rate": None,
                           "uncertain_change_interval": None})
            continue
        e = epochs[-1]
        first = events[e["first"]]
        t0, v0 = _fraction(first["time"]), tuple(_fraction(x) for x in first["values"])
        if e["first"] == e["last"]:
            e["last"] = i
            e["rate"] = tuple((x-y)/(t-t0) for x,y in zip(v,v0))
            continue
        predicted = tuple(x + rate*(t-t0) for x,rate in zip(v0,e["rate"]))
        if v == predicted:
            e["last"] = i
        else:
            e["status"] = "QUALIFIED_SAMPLES_ONLY" if e["last"]-e["first"] >= 2 else "CANDIDATE"
            epochs.append({"first": i, "last": i, "rate": None,
                           "uncertain_change_interval": [
                               events[i-1]["time"], event["time"]]})
    for e in epochs:
        e["status"] = ("QUALIFIED_SAMPLES_ONLY" if e["last"]-e["first"] >= 2
                       else "CANDIDATE" if e["last"] > e["first"] else "INSUFFICIENT")
    return epochs


def _eval_ast(node, env, fuel=24):
    """Fixed, bounded host bootstrap for generated vector arithmetic programs."""
    _require(type(node) is list and node and type(node[0]) is str and fuel > 0,
             "Malformed generated processor expression or exhausted fuel")
    op = node[0]
    if op == "INPUT":
        _require(len(node) == 2 and type(node[1]) is str and node[1] in env,
                 "Unbound processor input")
        return env[node[1]]
    _require(len(node) == 3 and op in ("ADD", "SUB", "MUL"),
             "Unadmitted processor operation")
    left = _eval_ast(node[1], env, fuel-1)
    right = _eval_ast(node[2], env, fuel-1)
    _require(len(left) == len(right), "Processor tensor dimension mismatch")
    return tuple({"ADD": lambda a,b:a+b, "SUB":lambda a,b:a-b,
                  "MUL":lambda a,b:a*b}[op](a,b) for a,b in zip(left,right))


def _program():
    return ["ADD", ["INPUT", "origin"],
            ["MUL", ["INPUT", "rate"],
             ["SUB", ["INPUT", "time"], ["INPUT", "anchor"]]]]


class TemporalIntelligenceGround:
    """One causally continuous identity with time-indexed, mutable interpretations.

    A new host can resume its exact ledger but cannot claim to have observed any
    missing wall-clock interval. No background process is created by this class.
    """

    def __init__(self, environment: str, *, dimensions=1, observer="vli-time"):
        _require(type(environment) is str and 1 <= len(environment) <= 160
                 and type(observer) is str and 1 <= len(observer) <= 100
                 and type(dimensions) is int and 1 <= dimensions <= MAX_DIMENSIONS,
                 "Invalid environment identity or observer scope")
        self.contract = {"environment":environment, "dimensions":dimensions,
                         "observer":observer, "time_domain":"exact_rational",
                         "semantics":"empirical_piecewise_affine_vector"}
        self.identity = _digest(SCHEMA, self.contract)
        self.events = []
        self.head = _digest(EVENT, {"identity":self.identity, "genesis":GENESIS})

    def observe(self, time: str, values, *, source_digest: str, host: str,
                expected_head: str):
        """Commit an observed sample after an external causal-head comparison."""
        _require(expected_head == self.head and _hex64(expected_head),
                 "Stale or missing independent lineage head")
        t = _fraction(time)
        vector = _vectors(values, self.contract["dimensions"])
        _require(_hex64(source_digest), "Source evidence digest is malformed")
        _require(type(host) is str and 1 <= len(host) <= 100,
                 "Missing observing host label")
        _require(len(self.events) < MAX_OBSERVATIONS, "Full observation budget")
        if self.events:
            _require(t > _fraction(self.events[-1]["time"]),
                     "Environmental time must advance strictly")
        body = {"schema":EVENT, "index":len(self.events), "previous":self.head,
                "identity":self.identity, "time":time,
                "values":[_encode(x) for x in vector],
                "source_digest":source_digest, "host":host}
        event = {**body, "digest":_digest(EVENT, body)}
        self.events.append(event)
        self.head = event["digest"]
        return deepcopy(event)

    @property
    def epochs(self):
        return deepcopy(_derive_epochs(self.events))

    def checkpoint(self):
        body = {"schema":SNAPSHOT, "contract":deepcopy(self.contract),
                "identity":self.identity, "events":deepcopy(self.events),
                "head":self.head}
        return {**body, "checkpoint_digest":_digest(SNAPSHOT,body)}

    @classmethod
    def restore(cls, snapshot, *, expected_head, expected_identity):
        verify_temporal_snapshot(snapshot, expected_head=expected_head,
                                 expected_identity=expected_identity)
        c = snapshot["contract"]
        obj = cls(c["environment"], dimensions=c["dimensions"],
                  observer=c["observer"])
        obj.events = deepcopy(snapshot["events"])
        obj.head = expected_head
        return obj

    def compile_epoch(self, epoch_index: int, *, expected_head: str):
        """Construct executable source-conditioned mathematics, not weights."""
        _require(expected_head == self.head and _hex64(expected_head),
                 "Stale source lineage for compiler")
        epochs = self.epochs
        _require(type(epoch_index) is int and 0 <= epoch_index < len(epochs),
                 "Missing environmental epoch")
        e = epochs[epoch_index]
        _require(e["status"] == "QUALIFIED_SAMPLES_ONLY",
                 "Three consistent distinct observations are required")
        first, last = self.events[e["first"]], self.events[e["last"]]
        body = {"schema":PROCESSOR, "identity":self.identity,
                "lineage_head":self.head, "epoch":epoch_index,
                "first_index":e["first"], "last_index":e["last"],
                "anchor":first["time"], "observed_end":last["time"],
                "origin":list(first["values"]),
                "rate":[_encode(x) for x in e["rate"]],
                "witnesses":[event["digest"] for event in
                             self.events[e["first"]:e["last"]+1]],
                "uncertain_change_interval":e["uncertain_change_interval"],
                "program":_program(),
                "claim":"EMPIRICAL_FIT_NOT_PHYSICAL_CONTINUITY_PROOF"}
        return {**body, "processor_digest":_digest(PROCESSOR,body)}

    def navigate(self, time: str, *, epoch_index: int, expected_head: str,
                 allow_projection=False):
        """Observe recorded state or evaluate the epoch's candidate continuous flow.

        Model output between sampled instants is explicitly NOT source evidence.
        Extrapolation requires an opt-in and is marked separately.
        """
        _require(expected_head == self.head and _hex64(expected_head),
                 "Stale temporal navigation identity")
        t = _fraction(time)
        p = self.compile_epoch(epoch_index, expected_head=expected_head)
        start, end = _fraction(p["anchor"]), _fraction(p["observed_end"])
        _require(t >= start, "Cannot navigate before the admitted epoch")
        if t > end:
            _require(allow_projection is True, "Future projection needs explicit consent")
            status = "UNVERIFIED_FUTURE_PROJECTION"
        else:
            status = "UNVERIFIED_INTERPOLATION"
        v = _eval_ast(p["program"],{
            "time":(t,)*self.contract["dimensions"],
            "anchor":(start,)*self.contract["dimensions"],
            "origin":_vectors(p["origin"],self.contract["dimensions"]),
            "rate":_vectors(p["rate"],self.contract["dimensions"]),
        })
        # A second independent arithmetic route checks program semantics.
        direct = tuple(x + rate*(t-start) for x,rate in zip(
            _vectors(p["origin"],self.contract["dimensions"]),
            _vectors(p["rate"],self.contract["dimensions"])))
        _require(v == direct, "Generated interpreter disagrees with exact mathematics")
        for event in self.events[p["first_index"]:p["last_index"]+1]:
            if event["time"] == time:
                _require(tuple(event["values"]) == tuple(_encode(x) for x in v),
                         "Generated processor failed historical source evidence")
                status = "EXACT_RECORDED_OBSERVATION"
                break
        body = {"schema":NAVIGATION, "identity":self.identity,
                "lineage_head":self.head, "processor_digest":p["processor_digest"],
                "epoch":epoch_index, "time":time,
                "values":[_encode(x) for x in v], "status":status,
                "expanded_trajectory_steps":0,
                "source_observation_at_coordinate":status == "EXACT_RECORDED_OBSERVATION"}
        return {**body, "navigation_digest":_digest(NAVIGATION,body)}


def verify_temporal_snapshot(snapshot, *, expected_head, expected_identity):
    """Reconstruct all causal events and segmentation, not just the outer hash.

    The independent caller still MUST authenticate the external source itself.
    Source digests and host names in the sample are not source signatures.
    """
    _require(type(snapshot) is dict and set(snapshot) == {
        "schema","contract","identity","events","head","checkpoint_digest"},
        "Malformed temporal checkpoint fields")
    _require(snapshot["schema"] == SNAPSHOT and _hex64(expected_head)
             and _hex64(expected_identity) and snapshot["head"] == expected_head
             and snapshot["identity"] == expected_identity
             and snapshot["checkpoint_digest"] == _digest(SNAPSHOT,{
                 k:v for k,v in snapshot.items() if k != "checkpoint_digest"}),
             "Temporal checkpoint changed or is not externally pinned")
    c = snapshot["contract"]
    _require(type(c) is dict and set(c) == {
        "environment","dimensions","observer","time_domain","semantics"},
        "Unknown environment contract fields")
    obj = TemporalIntelligenceGround(c["environment"],
             dimensions=c["dimensions"], observer=c["observer"])
    _require(obj.contract == c and obj.identity == expected_identity,
             "Changed environment semantics or dimensions")
    rows = snapshot["events"]
    _require(type(rows) is list and len(rows) <= MAX_OBSERVATIONS,
             "Invalid observed event count")
    for row in rows:
        _require(type(row) is dict and set(row) == {
            "schema","index","previous","identity","time","values",
            "source_digest","host","digest"},
            "Malformed causal observation")
        _require(row["schema"] == EVENT and type(row["index"]) is int
                 and row["index"] == len(obj.events)
                 and row["identity"] == expected_identity
                 and row["previous"] == obj.head
                 and _hex64(row["digest"]),
                 "Causal sequence or predecessor substituted")
        recreated = obj.observe(row["time"], row["values"],
                        source_digest=row["source_digest"], host=row["host"],
                        expected_head=obj.head)
        _require(recreated == row, "Environmental observation was altered")
    _require(obj.head == expected_head, "Observation lineage has changed")
    return True


def verify_temporal_processor(snapshot, processor, *, expected_head, expected_identity):
    """Re-derive all rate coefficients and every witness, including epoch split."""
    obj = TemporalIntelligenceGround.restore(
        snapshot, expected_head=expected_head, expected_identity=expected_identity)
    _require(type(processor) is dict and processor.get("schema") == PROCESSOR
             and type(processor.get("epoch")) is int,
             "Malformed environmental processor")
    expected = obj.compile_epoch(processor["epoch"], expected_head=expected_head)
    _require(processor == expected, "Generated processor contradicts observed history")
    return True


def verify_temporal_navigation(snapshot, nav, *, expected_head, expected_identity):
    """Re-run source-witness derivation and AST/direct semantic agreement."""
    obj = TemporalIntelligenceGround.restore(
        snapshot, expected_head=expected_head, expected_identity=expected_identity)
    _require(type(nav) is dict and nav.get("schema") == NAVIGATION
             and type(nav.get("epoch")) is int,
             "Malformed temporal navigation receipt")
    expected = obj.navigate(nav["time"], epoch_index=nav["epoch"],
              expected_head=expected_head,
              allow_projection=(nav.get("status") == "UNVERIFIED_FUTURE_PROJECTION"))
    _require(nav == expected, "Temporal navigation is not reconstructed")
    return True
