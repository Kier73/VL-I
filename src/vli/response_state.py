"""Intent-conditioned response-state coordinates and progressive Ground rendering.

Coordinates are opaque positions in an admitted response-state topology, NOT
numeric values, byte offsets, or scheduler ticks. A coordinate can address a
response/page/code/tool-proposal/tokens bundle of arbitrary *declared* size.
Only requested dependencies are realized; larger outputs still cost resources.

Actual predecessors: E-CPU DataGround exact byte observation and VLD's
checkpointed NavigableComputationDAG. This is an evidence-anchored composition
of supplied content, NOT a general language-model generator or latent
diffusion neural model. A tool proposal is never executed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

from .ground import _digest, _hex64, _require, load_json

SOURCE_SCHEMA = "vl-i.response-source-ground.v1"
INTENT_SCHEMA = "vl-i.response-intention.v1"
POSITION_SCHEMA = "vl-i.response-state-coordinate.v1"
SNAPSHOT_SCHEMA = "vl-i.response-formation-snapshot.v1"
VIEW_SCHEMA = "vl-i.response-observation.v1"
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,79}\Z")
_KINDS = frozenset(("text", "page", "code", "tool_call", "tokens"))
_ROLES = frozenset(("system", "user", "assistant", "tool"))
MAX_SLOTS = 128
MAX_BYTES_PER_SLOT = 8 * 1024 * 1024
MAX_HISTORY = 64


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ecpu_verify(raw: bytes):
    """Consume existing E-CPU interpretation; no duplicate byte algebra."""
    from mailcpu.data_ground import DataGround
    chunks = []
    for start in range(0, len(raw), 65536):
        data = DataGround(raw[start:start + 65536])
        chunks.append(data.byte_sha256)
    return chunks


def _strict_json(raw: bytes):
    def pairs(items):
        obj = {}
        for name, value in items:
            if name in obj:
                raise ValueError("Duplicate key in typed renderer data: " + name)
            obj[name] = value
        return obj
    def constant(value):
        raise ValueError("Invalid JSON constant " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _validate_payload(kind: str, raw: bytes):
    if kind in ("text", "page", "code"):
        raw.decode("utf-8", errors="strict")
    elif kind == "tool_call":
        value = _strict_json(raw)
        _require(type(value) is dict and set(value) == {"tool", "arguments"}
                 and type(value["tool"]) is str and _NAME.fullmatch(value["tool"]) is not None
                 and type(value["arguments"]) is dict,
                 "Tool-call view must be inert, typed proposal data")
    elif kind == "tokens":
        value = _strict_json(raw)
        _require(type(value) is list and
                 all(type(t) is int and t >= 0 for t in value),
                 "Token view must contain nonnegative token IDs")
    else:
        raise ValueError("Unsupported render kind")


@dataclass(frozen=True)
class SourceSlot:
    name: str
    kind: str
    content: bytes
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        _require(type(self.name) is str and _NAME.fullmatch(self.name) is not None,
                 "Invalid response-state slot name")
        _require(self.kind in _KINDS and type(self.content) is bytes and
                 0 < len(self.content) <= MAX_BYTES_PER_SLOT, "Invalid response-slot bytes")
        _require(type(self.dependencies) is tuple and
                 all(type(d) is str and _NAME.fullmatch(d) for d in self.dependencies)
                 and len(self.dependencies) == len(set(self.dependencies))
                 and self.name not in self.dependencies,
                 "Invalid slot dependencies")
        _validate_payload(self.kind, self.content)


class ResponseSourceGround:
    """A capacity description; content remains in source refs, not coordinate."""

    def __init__(self, slots: tuple[SourceSlot, ...]):
        _require(type(slots) is tuple and 1 <= len(slots) <= MAX_SLOTS
                 and all(isinstance(s, SourceSlot) for s in slots),
                 "Invalid source Ground slots")
        self.slots = {s.name: s for s in slots}
        _require(len(self.slots) == len(slots), "Duplicate Ground slot name")
        self._order = self._ordered(self.slots.keys())
        manifest = {
            "schema": SOURCE_SCHEMA, "scope": "caller_supplied_typed_content",
            "slots": [
                {"name": s.name, "kind": s.kind, "content_sha256": _sha(s.content),
                 "content_bytes": len(s.content), "dependencies": list(s.dependencies)}
                for s in sorted(slots, key=lambda x: x.name)
            ],
        }
        manifest["ground_digest"] = _digest(SOURCE_SCHEMA, manifest)
        self.manifest = manifest

    def _ordered(self, names):
        visiting, visited, result = set(), set(), []
        def visit(name):
            _require(name in self.slots, "Unresolved source Ground relation: " + name)
            _require(name not in visiting, "Cycle in response-state topology")
            if name in visited:
                return
            visiting.add(name)
            for d in self.slots[name].dependencies:
                visit(d)
            visiting.remove(name)
            visited.add(name)
            result.append(name)
        for name in sorted(names):
            visit(name)
        return tuple(result)

    def closure(self, selected):
        _require(type(selected) is tuple and 1 <= len(selected) <= MAX_SLOTS
                 and all(type(s) is str for s in selected)
                 and len(set(selected)) == len(selected),
                 "Invalid intention-selected source slots")
        needed = set()
        def walk(n):
            _require(n in self.slots, "Intention asks for a missing Ground slot")
            if n in needed:
                return
            needed.add(n)
            for d in self.slots[n].dependencies:
                walk(d)
        for n in selected:
            walk(n)
        return tuple(n for n in self._order if n in needed)

    def observed_bytes(self, name: str) -> tuple[bytes, tuple[str, ...]]:
        _require(name in self.slots, "Unknown Ground byte source")
        s = self.slots[name]
        chunk_hashes = tuple(_ecpu_verify(s.content))
        _require(_sha(s.content) == next(r["content_sha256"] for r in self.manifest["slots"]
                                      if r["name"] == name), "Changed source Ground")
        return s.content, chunk_hashes


def compile_intention(prompt: str, *, slots: tuple[str, ...], observer: str,
                      history: tuple[dict, ...] = ()) -> dict:
    """Raw prompt retained exactly; explicit view selection is a typed observer lens.

    Turning unrestricted natural-language prompts into these lenses remains a
    separate interpreter/intelligence research challenge.
    """
    _require(type(prompt) is str and 0 < len(prompt.encode("utf-8")) <= 16384,
             "Invalid observer intention")
    _require(type(observer) is str and _NAME.fullmatch(observer) is not None,
             "Invalid observer identity label")
    _require(type(history) is tuple and len(history) <= MAX_HISTORY, "Invalid causal conversation")
    for item in history:
        _require(type(item) is dict and set(item) == {"role", "content"}
                 and item["role"] in _ROLES and type(item["content"]) is str
                 and len(item["content"].encode("utf-8")) <= 16384,
                 "Malformed causal conversation turn")
    # A canonical typed intention is source-independent until resolved to one Ground.
    record = {
        "schema": INTENT_SCHEMA, "prompt": prompt,
        "selected_slots": list(slots), "observer": observer,
        "history": list(history), "scope": "typed_intention_not_free_text_understanding",
    }
    record["intent_digest"] = _digest(INTENT_SCHEMA, record)
    return record


def locate_response_state(source: ResponseSourceGround, intent: dict) -> dict:
    """Return a symbolic topology-bound *state position*, not a numerical state."""
    _require(isinstance(source, ResponseSourceGround) and type(intent) is dict
             and intent.get("schema") == INTENT_SCHEMA
             and _hex64(intent.get("intent_digest"))
             and intent["intent_digest"] == _digest(INTENT_SCHEMA, {
                 k:v for k,v in intent.items() if k != "intent_digest"}),
             "Changed or invalid intention")
    selected = intent.get("selected_slots")
    _require(type(selected) is list, "Invalid selected output topology")
    required = source.closure(tuple(selected))
    # A state coordinate points to the topology formed by the intention:
    # the selected output, dependencies, and causal context, not any scalar.
    body = {
        "schema": POSITION_SCHEMA, "ground_digest": source.manifest["ground_digest"],
        "intent_digest": intent["intent_digest"],
        "selected_slots": selected, "required_slots": list(required),
        "topology": [
            {"slot": name, "kind": source.slots[name].kind,
             "dependencies": list(source.slots[name].dependencies),
             "source_sha256": _sha(source.slots[name].content)}
            for name in required
        ],
        "scope": "position_of_renderable_observer_state_not_numeric_coordinate",
    }
    root = _digest(POSITION_SCHEMA, body)
    return {**body, "coordinate_root": root,
            "coordinate": "state://vli/" + root}


class ResponseFormation:
    """Progressively realize only an intention's dependency topology via VLD DAG."""

    def __init__(self, source: ResponseSourceGround, intent: dict, *,
                 checkpoint: dict | None = None, expected_head: str | None = None):
        from vld.navigation.dag import ComputationNode, NavigableComputationDAG
        self.source, self.intent = source, intent
        self.position = locate_response_state(source, intent)
        root = self.position["coordinate_root"]
        required = self.position["required_slots"]
        self.node_for = {name: f"s{i:04d}" for i,name in enumerate(required)}
        self.name_for = {v:k for k,v in self.node_for.items()}
        desired = self.position["selected_slots"]
        # Source values are NOT stored inside node specifications or the coordinate.
        dag = NavigableComputationDAG(dag_id="vli.response."+root[:32])
        dag.add_node(ComputationNode(
            node_id="outline", revision=1, operation="vli.form.outline",
            resource="vli.response", context_digest=root, dependencies=(),
            payload={"selected": desired, "required": required}))
        for name in required:
            slot = source.slots[name]
            dag.add_node(ComputationNode(
                node_id=self.node_for[name], revision=1,
                operation="vli.form.observed-slot", resource="vli.source-ground",
                context_digest=root,
                dependencies=("outline",) + tuple(self.node_for[d]
                    for d in slot.dependencies),
                payload={"name": name, "kind": slot.kind,
                         "source_sha256": _sha(slot.content),
                         "bytes": len(slot.content)}))
        dag.add_node(ComputationNode(
            node_id="render", revision=1, operation="vli.form.render",
            resource="vli.response", context_digest=root,
            dependencies=tuple(self.node_for[n] for n in required),
            payload={"selected": desired, "coordinate": self.position["coordinate"]}))
        self.materialized_bytes = 0
        self.materialized_chunks = 0
        self._template = dag
        if checkpoint is None:
            self.dag = dag
        else:
            _require(type(checkpoint) is dict and
                     checkpoint.get("schema") == SNAPSHOT_SCHEMA and
                     checkpoint.get("coordinate_root") == root and
                     checkpoint.get("ground_digest") == source.manifest["ground_digest"] and
                     checkpoint.get("intent_digest") == intent["intent_digest"] and
                     _hex64(expected_head) and
                     checkpoint.get("checkpoint_digest") == expected_head,
                     "Missing trusted state-coordinate checkpoint head")
            candidate = NavigableComputationDAG.from_checkpoint(checkpoint["dag"])
            _require(candidate.state_digest == checkpoint["dag"]["checkpoint_digest"]
                     and candidate.nodes == dag.nodes, "Changed topology or VLD state")
            _require(checkpoint["checkpoint_digest"] ==
                     _digest(SNAPSHOT_SCHEMA, {
                         k:v for k,v in checkpoint.items() if k != "checkpoint_digest"}),
                     "Altered response-state envelope")
            # Re-evaluate committed payloads against trusted Ground source.
            for node_id, receipt in candidate.committed.items():
                expected = self._expected(node_id, skip_byte_read=True)
                _require(receipt.value == expected, "Checkpoint claims unrealized source data")
            self.dag = candidate

    def _expected(self, node_id: str, *, skip_byte_read: bool):
        if node_id == "outline":
            return {"coordinate": self.position["coordinate"],
                    "required_slots": self.position["required_slots"],
                    "selected_slots": self.position["selected_slots"]}
        if node_id == "render":
            return {"coordinate": self.position["coordinate"],
                    "selected_slots": self.position["selected_slots"],
                    "source_digests": [self.source.slots[n].__dict__["content"].hex()[:0] +
                        _sha(self.source.slots[n].content)
                        for n in self.position["selected_slots"]],
                    "scope": "render_supplied_verified_ground_not_model_inference"}
        name = self.name_for[node_id]
        s = self.source.slots[name]
        if not skip_byte_read:
            raw, chunks = self.source.observed_bytes(name)
            self.materialized_bytes += len(raw)
            self.materialized_chunks += len(chunks)
        return {"slot": name, "kind": s.kind, "source_sha256": _sha(s.content),
                "declared_bytes": len(s.content), "observed": True}

    def advance(self, *, max_nodes: int = 4) -> dict:
        _require(type(max_nodes) is int and 1 <= max_nodes <= 256,
                 "Invalid formation budget")
        # Ready frontier and VLD dependency receipts are authoritative.
        ready = self.dag.ready_frontier()
        for node in ready[:max_nodes]:
            self.dag.commit_result(node.node_id, self._expected(
                node.node_id, skip_byte_read=False))
        return self.observe()

    def observe(self) -> dict:
        ready_slots = [n for n in self.position["required_slots"]
                       if self.node_for[n] in self.dag.committed]
        projections = []
        # View renders only requested slots. Dependency-only slots are
        # observed/verified but never silently emitted as extra output.
        for name in self.position["selected_slots"]:
            if name not in ready_slots:
                continue
            slot = self.source.slots[name]
            content = slot.content
            if slot.kind in ("text", "page", "code"):
                rendered = content.decode("utf-8")
            else:
                rendered = _strict_json(content)
            projections.append({"slot": name, "kind": slot.kind,
                                "content": rendered, "source_sha256": _sha(content),
                                "status": ("INERT_PROPOSAL" if slot.kind=="tool_call"
                                           else "OBSERVED_CONTENT")})
        result = {
            "schema": VIEW_SCHEMA,
            "coordinate": self.position["coordinate"],
            "coordinate_root": self.position["coordinate_root"],
            "state": "RENDERED" if "render" in self.dag.committed else "FORMING",
            "outline_ready": "outline" in self.dag.committed,
            "required_slots": self.position["required_slots"],
            "observed_slots": ready_slots,
            "remaining_slots": [n for n in self.position["required_slots"]
                                if n not in ready_slots],
            "projections": projections,
            "dag_state_digest": self.dag.state_digest,
            "measured_source_bytes_materialized_this_process": self.materialized_bytes,
            "source_chunks_read_this_process": self.materialized_chunks,
            "origin": "source_bound_progressive_structural_rendering_no_model_generation",
        }
        result["observation_digest"] = _digest(VIEW_SCHEMA, result)
        return result

    def checkpoint(self) -> dict:
        dag = self.dag.export_checkpoint()
        record = {
            "schema": SNAPSHOT_SCHEMA,
            "coordinate_root": self.position["coordinate_root"],
            "ground_digest": self.source.manifest["ground_digest"],
            "intent_digest": self.intent["intent_digest"],
            "dag": dag,
        }
        record["checkpoint_digest"] = _digest(SNAPSHOT_SCHEMA, record)
        return record
