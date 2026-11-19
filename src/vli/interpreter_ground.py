"""Interpreter Self-Ground: lift actual interpreter bytes and semantic contracts.

Integrates (not reimplements):
- E-CPU DataGround exact Z/256Z lenses and OperatorRegistry declarative snapshots.
- VLD ByteLens, SignalGroundSpec, VirtualGroundSpec and SignalStateMachine execution.
- VLD StructuralMemoryRegistry and E-CPU affine reference for non-enumerated orbit reads.

A lifted source/contract is NOT an executable copy of its handlers: a trusted
host must bind an interpreter. E-CPU operator declarations carry no authority.
A receipt records observable state, not free computation in network transit.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping

from .ground import _digest, _hex64, _require

SELF_SCHEMA = "vl-i.interpreter-self-ground.v1"
RADIANCE_SCHEMA = "vl-i.information-radiance-event.v1"
RUN_SCHEMA = "vl-i.interpreter-self-ground-run.v1"
GENESIS = "0" * 64
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 2048


def _hash(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _import_predecessors():
    try:
        from mailcpu.data_ground import DataGround
        from mailcpu.virtual_layer import OperatorRegistry
        from vld.ground import ByteLens, LensKind, VirtualGroundSpec
        from vld.signal import VirtualGroundInterpreter
    except ImportError as exc:
        raise RuntimeError("Install pinned E-CPU and Virtual-Layer-Dynamics for interpreter self-lift") from exc
    return DataGround, OperatorRegistry, ByteLens, LensKind, VirtualGroundSpec, VirtualGroundInterpreter


def lift_interpreter(source_modules: Mapping[str, bytes], operator_registry,
                     signal_interpreter):
    """Lift source bytes + declarative laws, *not* privileged executable handlers.

    Source is an explicitly supplied mapping of actual module bytes (not an
    auto-discovered package). Each chunk receives two independent byte-lens
    checks from E-CPU and VLD, and the complete source identity is retained.
    """
    DataGround, OperatorRegistry, ByteLens, LensKind, VirtualGroundSpec, VirtualGroundInterpreter = _import_predecessors()
    _require(type(source_modules) is dict and source_modules, "At least one interpreter source is required")
    _require(isinstance(operator_registry, OperatorRegistry), "Require actual E-CPU operator registry")
    _require(isinstance(signal_interpreter, VirtualGroundInterpreter), "Require actual VLD signal interpreter")
    snapshot = operator_registry.snapshot()
    rebuilt = OperatorRegistry.from_snapshot(snapshot)
    _require(rebuilt.digest == operator_registry.digest, "E-CPU operator contracts cannot round-trip")
    signal_spec = signal_interpreter.ground
    _require(signal_spec.domain_id == signal_interpreter.backend.domain_id,
             "Interpreter semantic domain disagrees with backend")
    sources = []
    total = 0
    for name, raw in sorted(source_modules.items()):
        _require(isinstance(name, str) and name and len(name) <= 120 and
                 type(raw) is bytes and raw and len(raw) <= MAX_SOURCE_BYTES,
                 "Invalid named interpreter source bytes")
        total += len(raw)
        _require(total <= MAX_SOURCE_BYTES, "Interpreter source admission budget exceeded")
        chunks = []
        for start in range(0, len(raw), 65536):
            part = raw[start:start+65536]
            ecpu = DataGround(part)
            vld_projection = ByteLens(LensKind.RAW_Z256).lift(part)
            _require(ecpu.byte_sha256 == vld_projection.source_sha256
                     and vld_projection.byte_length == len(part),
                     "Predecessor byte Grounds disagree")
            chunks.append({
                "offset": start, "length": len(part), "sha256": ecpu.byte_sha256,
                "vld_byte_lens_receipt": vld_projection.receipt_digest,
                "ecpu_scope": ecpu.source_scope,
            })
        sources.append({"name": name, "size": len(raw), "sha256": _hash(raw), "chunks": chunks})
    ops = signal_spec.instruction_set
    declarative_ground = VirtualGroundSpec(
        ground_id="vl-i.self-interpreter", version="0.1.0",
        language="e-cpu.operator-declarations+vld.signal-ir",
        state_domain=signal_spec.domain_id,
        transitions=tuple("vld:" + op for op in ops) +
                    tuple("ecpu:" + spec.operator_id + "@" + str(spec.version)
                          for spec in operator_registry.specs()),
        invariants=("only supplied source bytes identified", "handlers are not carried by catalog",
                    "source labels alone grant no runtime authority",
                    "observer output requires actual interpreter operation"),
        observers=("raw byte coordinate", "signal transition", "structural memory observation"),
        evidence_scope="source_bytes_and_declarative_contracts_only",
    )
    ground = {
        "schema": SELF_SCHEMA,
        "source_modules": sources,
        "source_bytes_observed": total,
        "ecpu_operator_snapshot": snapshot,
        "ecpu_operator_digest": operator_registry.digest,
        "vld_signal_ground": signal_spec.to_dict(),
        "vld_signal_ground_fingerprint": signal_spec.fingerprint,
        "vld_ground_spec": declarative_ground.to_dict(),
        "vld_ground_fingerprint": declarative_ground.fingerprint,
        "claims": {
            "source": "ALL_SUPPLIED_BYTES_DUAL_LENS_OBSERVED",
            "semantics": "DECLARED_OPERATOR_AND_SIGNAL_GROUND",
            "handlers": "NOT_TRANSPORTED_MUST_BE_BOUND_LOCALLY",
            "self_hosting": "NOT_DEMONSTRATED",
            "runtime_qualification": "NOT_GRANTED_BY_HASHES",
        },
    }
    ground["ground_digest"] = _digest(SELF_SCHEMA, ground)
    return ground


def verify_interpreter_ground(ground, source_modules: Mapping[str, bytes],
                              operator_registry, signal_interpreter):
    """Rebuild semantics and source bytes; rejects content/contract substitution."""
    _require(isinstance(ground, dict) and ground.get("schema") == SELF_SCHEMA
             and _hex64(ground.get("ground_digest")), "Invalid interpreter Ground")
    expected = lift_interpreter(source_modules, operator_registry, signal_interpreter)
    _require(ground == expected, "Interpreter source or semantic Ground changed")
    return True


class RadianceJournal:
    """Hash-chain every accepted information movement and observer projection.

    A chained event is an *evidence-bearing projection*, not energy-free
    computation or source-host authentication. A trusted head is external.
    """

    def __init__(self, interpreter_ground_digest: str):
        _require(_hex64(interpreter_ground_digest), "Invalid interpreter Ground identity")
        self.ground_digest = interpreter_ground_digest
        self.entries: list[dict] = []
        self.head = GENESIS

    def observe(self, kind: str, *, observer: str, source_digest: str,
                result_digest: str, relation: str, counted_units: int = 1):
        _require(isinstance(kind, str) and kind and isinstance(observer, str) and observer
                 and isinstance(relation, str) and relation, "Invalid information relation")
        _require(_hex64(source_digest) and _hex64(result_digest),
                 "Invalid linked source/result evidence")
        _require(type(counted_units) is int and counted_units >= 0,
                 "Invalid observation work counter")
        _require(len(self.entries) < MAX_EVENTS, "Observation journal budget exceeded")
        event = {
            "schema": RADIANCE_SCHEMA,
            "sequence": len(self.entries), "previous": self.head,
            "interpreter_ground_digest": self.ground_digest,
            "kind": kind, "observer": observer,
            "source_digest": source_digest, "result_digest": result_digest,
            "relation": relation, "counted_units": counted_units,
        }
        event["event_digest"] = _digest(RADIANCE_SCHEMA, event)
        self.entries.append(event)
        self.head = event["event_digest"]
        return event

    def snapshot(self):
        return {"schema": "vl-i.radiance-transcript.v1",
                "interpreter_ground_digest": self.ground_digest,
                "entries": [dict(e) for e in self.entries], "head": self.head}


def verify_radiance(snapshot, *, expected_ground_digest, expected_head):
    """Independent chain replay; expected head must come from a trusted context."""
    _require(isinstance(snapshot, dict) and snapshot.get("schema") ==
             "vl-i.radiance-transcript.v1", "Wrong radiance schema")
    _require(_hex64(expected_ground_digest) and _hex64(expected_head),
             "Missing externally pinned radiance head and Ground")
    _require(snapshot.get("interpreter_ground_digest") == expected_ground_digest,
             "Changed interpreter semantics")
    entries = snapshot.get("entries")
    _require(isinstance(entries, list) and len(entries) <= MAX_EVENTS,
             "Malformed radiance events")
    head = GENESIS
    for i, event in enumerate(entries):
        _require(isinstance(event, dict) and
                 event.get("schema") == RADIANCE_SCHEMA and
                 event.get("sequence") == i and type(event.get("sequence")) is int and
                 event.get("previous") == head and
                 event.get("interpreter_ground_digest") == expected_ground_digest,
                 "Radiance sequence or source identity disagrees")
        _require(_hex64(event.get("source_digest")) and
                 _hex64(event.get("result_digest")) and
                 type(event.get("counted_units")) is int and
                 event["counted_units"] >= 0 and
                 all(isinstance(event.get(k), str) and event[k]
                     for k in ("kind", "observer", "relation")), "Malformed radiance observation")
        head = _digest(RADIANCE_SCHEMA, {k:v for k,v in event.items()
                                        if k != "event_digest"})
        _require(event.get("event_digest") == head, "Altered information event")
    _require(snapshot.get("head") == head and head == expected_head,
             "Radiance transcript differs from trusted head")
    return True


def operate_byte_ground(raw: bytes, *, left: int, right: int, operation: str,
                        observer: str, interpreter_ground, signal_interpreter):
    """E-CPU byte coordinates -> VLD interpreted arithmetic -> radiant trace.

    A small concrete self-interpretation building block. The two source bytes
    determine inputs, and the interpreter Ground determines allowed operations.
    This does not implement language generation or use trained weights.
    """
    DataGround, _, _, _, _, VirtualGroundInterpreter = _import_predecessors()
    from vld.signal import Instruction, Program

    _require(isinstance(interpreter_ground, dict) and
             interpreter_ground.get("schema") == SELF_SCHEMA, "Require interpreter Ground")
    _require(isinstance(signal_interpreter, VirtualGroundInterpreter)
             and signal_interpreter.ground.fingerprint ==
             interpreter_ground.get("vld_signal_ground_fingerprint"),
             "Live interpreter differs from lifted Ground")
    _require(type(raw) is bytes and 1 <= len(raw) <= 65536, "Invalid byte source")
    _require(type(left) is int and type(right) is int and
             0 <= left < len(raw) and 0 <= right < len(raw), "Source byte coordinate outside Ground")
    _require(operation in ("ADD", "SUB", "MUL")
             and operation in signal_interpreter.ground.instruction_set,
             "Operation not admitted by lifted interpreter")
    _require(isinstance(observer, str) and observer, "Invalid observer")

    source = DataGround(raw)
    a, b = source.coordinate(left), source.coordinate(right)
    _require(a["source_sha256"] == b["source_sha256"],
             "Operands have different source Grounds")
    backend = signal_interpreter.backend
    _require(getattr(backend, "modulus", None) == 256,
             "Initial byte arithmetic requires an exact Z/256Z realizer")
    program = Program(
        name="vl-i.byte-intention",
        instructions=(
            Instruction("LOAD", {"dst": "a", "address": "left"}),
            Instruction("LOAD", {"dst": "b", "address": "right"}),
            Instruction(operation, {"dst": "result", "left": "a", "right": "b"}),
            Instruction("EMIT", {"src": "result", "label": "observed"}),
            Instruction("HALT"),
        ),
        metadata={"purpose": "interpret-byte-coordinates", "source_sha256": source.byte_sha256},
    )
    result = signal_interpreter.execute(
        program,
        {"left": backend.encode_symbol(a["value"]),
         "right": backend.encode_symbol(b["value"])},
    )
    _require(result.halted and len(result.outputs) == 1, "Incomplete interpreter execution")
    value = result.observed_symbols(backend)["observed"]
    oracle = {"ADD": lambda x,y:(x+y)%256, "SUB": lambda x,y:(x-y)%256,
              "MUL":lambda x,y:(x*y)%256}[operation](a["value"], b["value"])
    _require(value == oracle, "Independent byte-domain algebra disagrees with VLD")
    j = RadianceJournal(interpreter_ground["ground_digest"])
    j.observe("BYTE_SOURCE", observer=observer, source_digest=source.byte_sha256,
              result_digest=_digest("vl-i.byte-coordinate-intention.v1", {
                  "left": left, "right": right, "values": [a["value"], b["value"]]}),
              relation="ECPU.DataGround.coordinate", counted_units=2)
    for event in result.trace:
        trace = event.to_dict(backend)
        j.observe("INTERPRETER_TRANSITION", observer=observer,
                  source_digest=program.fingerprint,
                  result_digest=_digest("vl-i.vld-transition.v1", trace),
                  relation="VLD.VirtualGroundInterpreter." + operation,
                  counted_units=1)
    output_digest = _digest("vl-i.byte-result.v1",
                            {"source": source.byte_sha256, "op": operation,
                             "left": left, "right": right, "value": value})
    j.observe("OBSERVER_OUTPUT", observer=observer, source_digest=program.fingerprint,
              result_digest=output_digest,
              relation="VLD.observed_symbols", counted_units=1)
    transcript = j.snapshot()
    verify_radiance(transcript, expected_ground_digest=interpreter_ground["ground_digest"],
                    expected_head=j.head)
    record = {
        "schema": RUN_SCHEMA, "ground_digest": interpreter_ground["ground_digest"],
        "source_sha256": source.byte_sha256, "source_byte_count": len(raw),
        "observer": observer, "input_coordinates": [left, right],
        "op": operation, "operand_bytes": [a["value"], b["value"]],
        "program_fingerprint": program.fingerprint,
        "value": value, "oracle_value": oracle, "signal_steps": len(result.trace),
        "host_operations": "VLD_SIGNAL_INTERPRETER_ACTUALLY_EXECUTED",
        "transcript": transcript,
        "scope": "exact_resident_byte_relation_not_model_language_generation",
    }
    record["run_digest"] = _digest(RUN_SCHEMA, record)
    return record


def verify_byte_ground_run(record, raw, interpreter_ground, signal_interpreter):
    """Re-execute on original bytes and compare the entire deterministic trace."""
    _require(isinstance(record, dict) and record.get("schema") == RUN_SCHEMA
             and _hex64(record.get("run_digest")) and
             record["run_digest"] == _digest(RUN_SCHEMA, {
                 k:v for k,v in record.items() if k != "run_digest"}),
             "Malformed or altered interpreter report")
    regenerated = operate_byte_ground(raw, left=record["input_coordinates"][0],
                                       right=record["input_coordinates"][1],
                                       operation=record["op"], observer=record["observer"],
                                       interpreter_ground=interpreter_ground,
                                       signal_interpreter=signal_interpreter)
    _require(regenerated == record, "Interpreter replay or evidence disagree")
    return True


def observe_structural_orbit(ground, vld_registry, *, memory_id: str,
                             coordinate: int, expected_value: int, observer: str):
    """Read VLD certified Structural Memory, not by replaying all intermediate states.

    Caller supplies the independently expected value; the VLD registry checks
    its own journal and witness/certificate state. Retain its native registry
    as authority, do not invent a competing memory-verification protocol.
    """
    from vld.memory import StructuralMemoryRegistry
    _require(isinstance(vld_registry, StructuralMemoryRegistry), "Require VLD memory registry")
    _require(isinstance(ground, dict) and _hex64(ground.get("ground_digest")),
             "Missing interpreter identity")
    _require(type(coordinate) is int and coordinate >= 0 and
             type(expected_value) is int, "Invalid qualified orbit coordinate")
    value, receipt = vld_registry.read(memory_id, coordinate)
    valid, reason = vld_registry.verify_read_receipt(receipt, require_active_revision=True)
    _require(valid and value == expected_value, "Unqualified Structural Memory read: " + reason)
    j = RadianceJournal(ground["ground_digest"])
    j.observe("VERIFIED_STRUCTURAL_MEMORY", observer=observer,
              source_digest=receipt.spec_digest, result_digest=receipt.receipt_digest,
              relation="VLD.StructuralMemoryRegistry.read",
              counted_units=1)
    return {
        "memory_id": memory_id, "coordinate": coordinate,
        "value": value, "source_spec_digest": receipt.spec_digest,
        "native_receipt_digest": receipt.receipt_digest,
        "native_journal_event_digest": receipt.journal_event_digest,
        "expanded_intermediate_states": receipt.expanded_intermediate_states,
        "transcript": j.snapshot(),
        "meaning": "qualified_direct_memory_navigation_not_free_arithmetic",
    }


def load_real_interpreter_modules():
    """Read installed predecessor and VL-I interpreter source bytes, no execution."""
    from mailcpu.virtual_layer import VirtualLayerRuntime
    from vld.signal.machine import VirtualGroundInterpreter
    from . import interpreter_ground

    import inspect
    sources = {}
    for name, obj in (
        ("ECPU.VirtualLayerRuntime", VirtualLayerRuntime),
        ("VLD.VirtualGroundInterpreter", VirtualGroundInterpreter),
        ("VLI.InterpreterSelfGround", interpreter_ground),
    ):
        path = inspect.getsourcefile(obj)
        _require(path is not None, "No actual source file for interpreter: " + name)
        sources[name] = Path(path).read_bytes()
    return sources
