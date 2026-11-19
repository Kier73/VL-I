"""Evidence-scoped model lifting and intention-directed dependency navigation.

VL-I-001 composes the pinned VL-64 typed-pretrained decoder Ground; it does
not invent tensor bytes, prove general source authenticity, or execute neural
operators by parsing a header. External compiler is the numerical graph
authority for this first adapter; this module independently checks the
dependency closure and source/interpretation separation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "vl-i.model-ground.v1"
FRONTIER_SCHEMA = "vl-i.intention-frontier.v1"
_HEX40 = re.compile(r"^[a-f0-9]{40}$")
_HEX64 = re.compile(r"^[a-f0-9]{64}$")
_SIZE = {"BF16": 2, "F32": 4}


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _digest(schema, obj):
    return hashlib.sha256(schema.encode("utf-8") + b"\x00" + _canonical(obj)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hex64(value):
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def load_json(path):
    """Load data without forgiving duplicate keys or NaN/Infinity constants."""
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("Duplicate JSON key: " + k)
            out[k] = v
        return out

    def reject_constant(value):
        raise ValueError("Non-finite JSON constant: " + value)

    raw = Path(path).read_bytes()
    _require(len(raw) <= 64 * 1024 * 1024, "Oversized manifest")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=reject_constant)


def validate_plan(observed, plan):
    """Check a typed graph and source binding *independently* of graph compiler.

    This is a structural check of an already-observed snapshot, not a proof
    that the source contains the declared weights or matches trained behavior.
    """
    _require(isinstance(observed, dict) and isinstance(plan, dict), "Missing manifest/plan")
    _require(observed.get("schema") == "vl64.remote-model-topology-ground.v1",
             "Wrong observation schema")
    _require(plan.get("schema") == "vl64.typed-pretrained-execution-dag.v1",
             "Wrong typed execution DAG")
    name, revision = observed.get("model"), observed.get("pinned_revision")
    _require(isinstance(name, str) and name and "/" in name, "Unqualified model name")
    _require(isinstance(revision, str) and _HEX40.fullmatch(revision),
             "Source revision must be an immutable commit hash")
    _require(plan.get("model") == name and plan.get("revision") == revision,
             "Source/typed-plan identity mismatch")
    _require(plan.get("model_ground_digest") == observed.get("ground_digest")
             and _hex64(observed.get("ground_digest")), "Model-Ground digest mismatch")
    header = observed.get("tensors")
    _require(isinstance(header, dict)
             and header.get("schema") == "vl64.remote-tensor-structure-ground.v1"
             and observed.get("tensor_header_status") == "OBSERVED_COMPLETE_TENSOR_HEADER",
             "Not a complete observed tensor header")
    _require(plan.get("tensor_ground_digest") == header.get("ground_digest")
             and _hex64(header.get("ground_digest")), "Tensor-Ground digest mismatch")
    _require(_hex64(plan.get("dag_digest")), "No typed graph digest")
    _require(plan.get("tensors") and isinstance(plan["tensors"], dict), "Missing tensor addresses")
    _require(isinstance(plan.get("nodes"), list) and plan["nodes"], "Missing graph nodes")

    tensor_list = header.get("tensors")
    _require(isinstance(tensor_list, list) and tensor_list, "No tensor inventory")
    inventory = {}
    intervals = []
    for entry in tensor_list:
        _require(isinstance(entry, dict), "Invalid tensor")
        key, shape, dtype, offset = (entry.get(k) for k in
                                     ("name", "shape", "dtype", "data_offsets"))
        _require(isinstance(key, str) and key and key not in inventory,
                 "Duplicate/invalid tensor name")
        _require(dtype in _SIZE and isinstance(shape, list) and shape and
                 all(type(d) is int and d > 0 for d in shape), "Invalid tensor dtype/shape")
        _require(isinstance(offset, list) and len(offset) == 2 and
                 all(type(n) is int and n >= 0 for n in offset) and
                 offset[1] - offset[0] == math.prod(shape) * _SIZE[dtype],
                 "Invalid tensor byte extent")
        inventory[key] = entry
        intervals.append((offset[0], offset[1]))
    intervals.sort()
    _require(all(intervals[i][1] <= intervals[i+1][0]
                 for i in range(len(intervals)-1)), "Overlapping tensor extents")
    _require(header.get("tensor_count") == len(inventory), "Tensor inventory count mismatch")
    _require(header.get("declared_payload_bytes") == intervals[-1][1],
             "Tensor payload length mismatch")
    _require(set(inventory) == set(plan["tensors"]),
             "Typed graph must bind the exact observed tensor inventory")
    for key, item in plan["tensors"].items():
        # VL-64 compiles a minimal typed address rather than copying
        # header-specific metadata such as 'name' and 'elements'.
        expected = {field: inventory[key][field]
                    for field in ("dtype", "shape", "data_offsets")}
        _require(item == expected, "Typed graph tensor binding altered: " + key)

    seen = set()
    ops = Counter()
    for node in plan["nodes"]:
        _require(isinstance(node, dict) and isinstance(node.get("id"), str) and
                 node["id"] not in seen, "Duplicate/invalid graph node")
        inputs = node.get("inputs")
        _require(isinstance(inputs, list) and
                 all(isinstance(dep, str) and dep in seen for dep in inputs),
                 "Non-topological or unresolved graph dependency")
        weight = node.get("weight")
        _require(weight is None or weight in inventory,
                 "Graph references absent trained tensor")
        op = node.get("op")
        _require(isinstance(op, str) and op, "Unknown graph operation")
        seen.add(node["id"])
        ops[op] += 1
    _require(plan["nodes"][0]["id"] == "tokens" and
             plan["nodes"][-1]["id"] == "vocab_logits",
             "Not a complete token-to-vocabulary graph")
    return dict(sorted(ops.items()))


def lift_observation(observed, *, compiler=None):
    """Compile an observed model as an addressable Ground. Zero weight bytes read."""
    if compiler is None:
        try:
            from experiments.model_fabric.execution_dag import compile_decoder_dag
        except ImportError as exc:
            raise RuntimeError(
                "Pin and install VL-64; this adapter does not duplicate its compiler."
            ) from exc
        compiler = compile_decoder_dag
    plan = compiler(observed)
    ops = validate_plan(observed, plan)
    tensor = observed["tensors"]
    record = {
        "schema": SCHEMA,
        "source": {
            "model": observed["model"],
            "revision": observed["pinned_revision"],
            "observed_config_sha256": observed["config_sha256"],
            "observed_header_sha256": tensor["header_sha256"],
            "model_ground_digest": observed["ground_digest"],
            "tensor_ground_digest": tensor["ground_digest"],
            "source_url": tensor["source"],
            "observation_scope": "metadata_and_safetensors_header_only",
            "payload_bytes_admitted": 0,
            "declared_model_payload_bytes": tensor["declared_payload_bytes"],
        },
        "interpretation": {
            "compiler": "vl64.typed-pretrained-execution-dag.v1",
            "dag_digest": plan["dag_digest"],
            "architecture": plan["architecture"],
            "operations": ops,
            "node_count": len(plan["nodes"]),
            "tensor_count": len(tensor["tensors"]),
            "operator_semantics_scope": "VL64 supported Llama/SwiGLU/RoPE/GQA only",
        },
        "readiness": {
            "structural": "QUALIFIED_BY_VL64_AND_VLI_DEPENDENCY_CHECKS",
            "numerical": "UNRESOLVED_WEIGHT_PAYLOAD_NOT_ACQUIRED",
            "tokenizer": "UNRESOLVED_TOKENIZER_ASSETS_NOT_ACQUIRED",
            "generation": "NOT_EXECUTED",
        },
    }
    record["ground_digest"] = _digest(SCHEMA, record)
    return record, plan


def verify_ground(record):
    _require(isinstance(record, dict) and record.get("schema") == SCHEMA,
             "Wrong VL-I Ground schema")
    digest = record.get("ground_digest")
    _require(_hex64(digest) and digest == _digest(SCHEMA, {
        k: v for k, v in record.items() if k != "ground_digest"}),
        "VL-I Ground digest mismatch")
    source = record.get("source", {})
    _require(isinstance(source, dict)
             and source.get("observation_scope") == "metadata_and_safetensors_header_only"
             and source.get("payload_bytes_admitted") == 0,
             "A header-only Ground cannot admit pretrained weights")
    return True


def intention_frontier(plan, targets):
    """Compute the exact *structural dependency closure* for requested nodes.

    This is graph navigation, not a claim that the selected numerical outputs
    have been executed. For full vocab_logits, the closure should be the whole
    graph; intermediate observations can need fewer original weight ranges.
    """
    _require(isinstance(targets, (tuple, list)) and targets and
             all(isinstance(t, str) for t in targets), "Invalid observation targets")
    node_map = {n["id"]: n for n in plan["nodes"]}
    _require(len(node_map) == len(plan["nodes"]), "Duplicate graph nodes")
    _require(all(t in node_map for t in targets), "Unknown observation target")
    need = set()
    stack = list(targets)
    while stack:
        name = stack.pop()
        if name in need:
            continue
        need.add(name)
        stack.extend(node_map[name]["inputs"])
    selected = [n for n in plan["nodes"] if n["id"] in need]
    weights = {n["weight"] for n in selected if n.get("weight")}
    _require(weights <= plan["tensors"].keys(), "Missing parameter source")
    all_used = {n["weight"] for n in plan["nodes"] if n.get("weight")}
    def cost(keys):
        return sum(math.prod(plan["tensors"][k]["shape"]) *
                   _SIZE[plan["tensors"][k]["dtype"]] for k in keys)
    record = {
        "schema": FRONTIER_SCHEMA,
        "model": plan["model"],
        "revision": plan["revision"],
        "dag_digest": plan["dag_digest"],
        "targets": sorted(set(targets)),
        "required_nodes": [n["id"] for n in selected],
        "required_tensor_names": sorted(weights),
        "required_weight_bytes_declared": cost(weights),
        "full_graph_weight_bytes_declared": cost(all_used),
        "excluded_weight_bytes_declared": cost(all_used - weights),
        "scope": "declared_dependency_closure_no_weight_payload_acquired",
    }
    record["frontier_digest"] = _digest(FRONTIER_SCHEMA, record)
    return record


def verify_frontier(record, plan):
    _require(isinstance(record, dict) and record.get("schema") == FRONTIER_SCHEMA,
             "Wrong frontier schema")
    digest = record.get("frontier_digest")
    _require(_hex64(digest) and digest == _digest(FRONTIER_SCHEMA, {
        k: v for k, v in record.items() if k != "frontier_digest"}),
        "Frontier digest mismatch")
    expected = intention_frontier(plan, record["targets"])
    _require(expected == record, "Frontier dependency evidence mismatch")
    return True
