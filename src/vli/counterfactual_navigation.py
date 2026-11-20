"""NAV-001: exact, pinned branch navigation over qualified Qwen causal layers.

An edge represents one *actually executed* model-worker transition. A
coordinate is allocated before execution and bound to the parent, token,
and exact F64 input. Revisiting an edge retrieves verified retained state;
it never infers an unobserved result.

Caller-supplied hidden vectors are NOT source-verified token embeddings.
Tokenization and pretrained input admission remain separate contracts.
No full-model decoder, logits, or generated language are claimed.
"""
from __future__ import annotations

from copy import deepcopy
import struct

from experiments.model_fabric.complete_native_layers import LocalStatePages
from .ground import _digest, _hex64
from .qwen_layer_chain import QwenCausalGDNChain, _f64_transfer_digest
from .qwen_hybrid_block import QwenQualifiedHybridBlock

ROOT = "vl-i.counterfactual-root.v1"
COORDINATE = "vl-i.counterfactual-coordinate.v1"
LINEAGE = "vl-i.counterfactual-input-lineage.v1"
NODE = "vl-i.counterfactual-node.v1"
FOREST = "vl-i.counterfactual-forest.v1"
SNAPSHOT = "vl-i.counterfactual-snapshot.v1"
MAX_NODES = 256
MAX_DEPTH = 128


def _pack(hidden):
    digest = _f64_transfer_digest(hidden)
    raw = struct.pack("<" + str(len(hidden)) + "d", *(float(x) for x in hidden))
    return raw, digest


def _pages(store, raw):
    return [store.put(raw[i:i+65536]) for i in range(0, len(raw), 65536)]


def _read(store, pages, width, digest):
    if (type(pages) is not list or len(pages) != (width*8+65535)//65536 or
            any(type(p) is not dict for p in pages) or not _hex64(digest)):
        raise ValueError("Malformed retained numerical state pages")
    raw = b"".join(store.get(p) for p in pages)
    if len(raw) != 8*width:
        raise ValueError("Incorrect retained numerical result width")
    values = list(struct.unpack("<" + str(width) + "d", raw))
    if _f64_transfer_digest(values) != digest:
        raise ValueError("Retained numerical result fails exact F64 witness")
    return values


class CounterfactualCausalGround:
    """Persistent DAG of already-executed, original-geometry causal futures.

    The factory creates a fresh two-GDN chain or four-layer hybrid block with
    qualified model-source navigators. A caller must retain expected_head and
    expected_parent_digest independently; self-reported digests aren't authority.
    Token IDs label edges; independent observers must verify token embedding
    provenance before making claims of original-token model computation.
    """

    def __init__(self, factory, pages, *, observer="vli-counterfactual", max_nodes=MAX_NODES):
        if (not callable(factory) or not isinstance(pages, LocalStatePages) or
                type(observer) is not str or not 1 <= len(observer) <= 100 or
                type(max_nodes) is not int or not 1 <= max_nodes <= MAX_NODES):
            raise ValueError("Unqualified observer, model factory or causal page store")
        self.factory, self.pages = factory, pages
        self.observer, self.max_nodes = observer, max_nodes
        initial = self._fresh()
        self.signature = self._signature(initial)
        self.source_root = _digest(ROOT, self.signature)
        self.root_coordinate = "state://vli/" + self.source_root
        self.root_digest = self.source_root
        self.root_lineage = _digest(LINEAGE, {"source_root": self.source_root})
        self.head = _digest(FOREST, {"root": self.root_digest, "observer": observer})
        self.nodes = {}
        self.executed_transitions = 0
        self.reused_transitions = 0

    @staticmethod
    def _signature(runner):
        return {"class": type(runner).__name__, "genesis": runner.genesis,
                "model": runner.model, "revision": runner.revision,
                "index_ground_digest": runner.index_ground_digest,
                "layers": list(runner.layers), "width": runner.width}

    def _fresh(self):
        runner = self.factory()
        if (type(runner) not in (QwenCausalGDNChain, QwenQualifiedHybridBlock) or
                runner.position != 0 or runner.head != runner.genesis or
                runner.last_output_digest is not None):
            raise ValueError("A fresh qualified Qwen causal genesis is required")
        if hasattr(self, "signature") and self._signature(runner) != self.signature:
            raise ValueError("Source, trained geometry or interpreter lineage changed")
        return runner

    def _audit(self):
        """Recompute the authenticated forest before every public read/write.

        A node can live in mutable worker memory; the caller's external
        forest-head pin must still reject a rewritten or reordered node map.
        This is hash verification, not trained numerical re-execution.
        """
        committed = _digest(FOREST, {"root": self.root_digest,
                                    "observer": self.observer})
        seen = set()
        for coordinate, node in self.nodes.items():
            if (type(node) is not dict or node.get("coordinate") != coordinate or
                    (node.get("parent") != self.root_coordinate and
                     node.get("parent") not in seen) or
                    node.get("node_digest") != _digest(
                        NODE, {k:v for k,v in node.items() if k != "node_digest"})):
                raise ValueError("Committed causal forest has altered node ancestry")
            committed = _digest(FOREST, {"previous": committed,
                                       "node_digest": node["node_digest"]})
            seen.add(coordinate)
        if committed != self.head:
            raise ValueError("Committed causal forest differs from external head")

    def _coordinate(self, parent, token_id, input_digest):
        return "state://vli/" + _digest(COORDINATE, {
            "source_root": self.source_root, "parent": parent,
            "token_id": token_id, "input_f64_digest": input_digest})

    def _inspect(self, node, *, verify_checkpoint=True):
        if (type(node) is not dict or set(node) != {
                "schema", "coordinate", "parent", "depth", "token_id",
                "input_f64_digest", "input_sequence_digest", "previous_causal_head",
                "causal_head", "output_f64_digest", "result_pages",
                "checkpoint", "checkpoint_digest", "receipt", "receipt_digest", "node_digest"} or
            node["schema"] != NODE or not _hex64(node["node_digest"]) or
            _digest(NODE, {k:v for k,v in node.items() if k != "node_digest"}) != node["node_digest"]):
            raise ValueError("Tampered or malformed causal branch receipt")
        parent = node["parent"]
        if parent == self.root_coordinate:
            prior, lineage, depth = self.root_digest, self.root_lineage, 0
            previous = self.signature["genesis"]
        else:
            if parent not in self.nodes:
                raise ValueError("Missing ancestral causal branch")
            p = self.nodes[parent]
            prior, lineage, depth, previous = (p["node_digest"],
                p["input_sequence_digest"], p["depth"], p["causal_head"])
        token = node["token_id"]
        if (type(token) is not int or not 0 <= token < 2**31 or
                type(node["depth"]) is not int or node["depth"] != depth+1 or
                node["depth"] > MAX_DEPTH or
                not _hex64(node["input_f64_digest"]) or
                not _hex64(node["output_f64_digest"]) or
                not _hex64(node["causal_head"]) or
                not _hex64(node["receipt_digest"]) or
                not _hex64(node["checkpoint_digest"]) or
                node["previous_causal_head"] != previous or
                node["coordinate"] != self._coordinate(parent,token,node["input_f64_digest"])):
            raise ValueError("Causal branch position, token or lineage changed")
        expected_lineage = _digest(LINEAGE, {
            "parent_lineage": lineage, "parent_digest": prior,
            "token_id": token, "input_f64_digest": node["input_f64_digest"]})
        receipt = node["receipt"]
        if (type(receipt) is not dict or
                node["receipt_digest"] != _digest(NODE, receipt) or
                receipt.get("position") != depth or
                receipt.get("original_token_id") != token or
                receipt.get("previous") != previous or
                receipt.get("chain_head") != node["causal_head"] or
                receipt.get("model") != self.signature["model"] or
                receipt.get("revision") != self.signature["revision"] or
                receipt.get("index_ground_digest") != self.signature["index_ground_digest"] or
                receipt.get("original_decoder_completed") is not False):
            raise ValueError("Contradictory retained numerical transition receipt")
        witnesses = receipt.get("layer_witnesses")
        if (type(witnesses) is not list or
                len(witnesses) != len(self.signature["layers"]) or
                [w.get("layer") for w in witnesses] != self.signature["layers"] or
                any(type(w) is not dict for w in witnesses) or
                any(w.get("position") != depth or
                    not _hex64(w.get("input_f64_digest")) or
                    not _hex64(w.get("output_f64_digest"))
                    for w in witnesses) or
                witnesses[0]["input_f64_digest"] != node["input_f64_digest"] or
                witnesses[-1]["output_f64_digest"] != node["output_f64_digest"] or
                any(a["output_f64_digest"] != b["input_f64_digest"]
                    for a,b in zip(witnesses,witnesses[1:]))):
            raise ValueError("Broken causal predecessor-to-successor F64 transfer")
        cp = node["checkpoint"]
        if (node["input_sequence_digest"] != expected_lineage or
                type(cp) is not dict or cp.get("checkpoint_digest") != node["checkpoint_digest"] or
                cp.get("causal_head") != node["causal_head"] or
                cp.get("last_output_f64_digest") != node["output_f64_digest"] or
                cp.get("position") != node["depth"] or
                cp.get("input_sequence_digest") != expected_lineage or
                cp.get("genesis") != self.signature["genesis"] or
                cp.get("model") != self.signature["model"] or
                cp.get("revision") != self.signature["revision"] or
                cp.get("index_ground_digest") != self.signature["index_ground_digest"] or
                cp.get("layers") != self.signature["layers"]):
            raise ValueError("Causal checkpoint does not belong to its exact branch")
        values = _read(self.pages, node["result_pages"], self.signature["width"],
                       node["output_f64_digest"])
        if verify_checkpoint:
            self._restore(node)
        return values

    def _restore(self, node):
        fresh = self._fresh()
        recovered = type(fresh).restore(fresh.runners, self.pages, node["checkpoint"],
            expected_digest=node["checkpoint_digest"],
            expected_input_sequence_digest=node["input_sequence_digest"])
        if (recovered.head != node["causal_head"] or
                recovered.position != node["depth"] or
                recovered.last_output_digest != node["output_f64_digest"]):
            raise ValueError("Rehomed causal state disagrees with branch witness")
        return recovered

    def advance(self, parent, hidden, *, token_id, expected_parent_digest, expected_head):
        """Execute one new edge or navigate exactly the same retained future."""
        if expected_head != self.head:
            raise ValueError("Stale counterfactual forest authority head")
        self._audit()
        if type(token_id) is not int or not 0 <= token_id < 2**31:
            raise ValueError("Original token identifier is required")
        raw, input_digest = _pack(hidden)
        if len(raw) != self.signature["width"]*8:
            raise ValueError("Input width disagrees with pretrained causal Ground")
        if parent == self.root_coordinate:
            p, parent_digest, lineage, depth = None, self.root_digest, self.root_lineage, 0
            prev_head = self.signature["genesis"]
        else:
            if parent not in self.nodes:
                raise ValueError("Unknown causal predecessor coordinate")
            p = self.nodes[parent]
            self._inspect(p, verify_checkpoint=True)
            parent_digest, lineage, depth, prev_head = (p["node_digest"],
                p["input_sequence_digest"], p["depth"], p["causal_head"])
        if parent_digest != expected_parent_digest:
            raise ValueError("Stale or substituted causal predecessor pin")
        if depth >= MAX_DEPTH:
            raise ValueError("Maximum causal branch depth reached")
        coordinate = self._coordinate(parent, token_id, input_digest)
        if coordinate in self.nodes:
            values = self._inspect(self.nodes[coordinate], verify_checkpoint=True)
            self.reused_transitions += 1
            return values, deepcopy(self.nodes[coordinate]), "RETAINED_NO_TRAINED_TRANSITION"
        if len(self.nodes) >= self.max_nodes:
            raise ValueError("Maximum retained counterfactual states reached")
        chain = self._fresh() if p is None else self._restore(p)
        if chain.head != prev_head or chain.position != depth:
            raise ValueError("Model host recovered a different causal predecessor")
        output, receipt = chain.step(hidden, token_id=token_id)
        output_raw, output_digest = _pack(output)
        if (receipt.get("position") != depth or
                receipt.get("previous") != prev_head or
                receipt.get("chain_head") != chain.head or
                chain.last_output_digest != output_digest or
                (receipt.get("all_layers_completed") is not True and
                 receipt.get("all_four_original_layers_completed") is not True) or
                receipt.get("original_decoder_completed") is not False):
            raise ValueError("Unqualified complete causal layer execution")
        input_lineage = _digest(LINEAGE, {
            "parent_lineage": lineage, "parent_digest": parent_digest,
            "token_id": token_id, "input_f64_digest": input_digest})
        cp = chain.checkpoint(self.pages, input_sequence_digest=input_lineage)
        body = {"schema": NODE, "coordinate": coordinate, "parent": parent,
                "depth": depth+1, "token_id": token_id,
                "input_f64_digest": input_digest,
                "input_sequence_digest": input_lineage,
                "previous_causal_head": prev_head, "causal_head": chain.head,
                "output_f64_digest": output_digest,
                "result_pages": _pages(self.pages, output_raw),
                "checkpoint": cp, "checkpoint_digest": cp["checkpoint_digest"],
                "receipt": deepcopy(receipt), "receipt_digest": _digest(NODE, receipt)}
        node = {**body, "node_digest": _digest(NODE, body)}
        # Commit only after every state page, receipt and recovered state checks.
        # A failed write may leave orphaned content-addressed pages, not an edge.
        self._inspect(node, verify_checkpoint=True)
        new_head = _digest(FOREST, {"previous": self.head,
                                   "node_digest": node["node_digest"]})
        self.nodes[coordinate] = node
        self.head = new_head
        self.executed_transitions += 1
        return output, deepcopy(node), "NEW_TRAINED_TRANSITION"

    def observe(self, coordinate, *, expected_head):
        """Retrieve a retained result with no trained transition arithmetic."""
        if expected_head != self.head or coordinate not in self.nodes:
            raise ValueError("Stale forest head or unknown observed state")
        self._audit()
        return self._inspect(self.nodes[coordinate], verify_checkpoint=False)

    def snapshot(self):
        """Publish a serializable DAG requiring a separately retained digest."""
        self._audit()
        body = {"schema": SNAPSHOT, "source_root": self.source_root,
                "observer": self.observer, "max_nodes": self.max_nodes,
                "forest_head": self.head, "nodes": deepcopy(list(self.nodes.values()))}
        return {**body, "snapshot_digest": _digest(SNAPSHOT, body)}

    @classmethod
    def restore(cls, factory, pages, snapshot, *, expected_digest, strict=True):
        if (type(snapshot) is not dict or not _hex64(expected_digest) or
                snapshot.get("snapshot_digest") != expected_digest or
                _digest(SNAPSHOT, {k:v for k,v in snapshot.items()
                    if k != "snapshot_digest"}) != expected_digest or
                set(snapshot) != {"schema", "source_root", "observer", "max_nodes",
                                 "forest_head", "nodes", "snapshot_digest"} or
                snapshot["schema"] != SNAPSHOT or
                type(snapshot["nodes"]) is not list or
                len(snapshot["nodes"]) > MAX_NODES):
            raise ValueError("Modified or unpinned counterfactual snapshot")
        result = cls(factory, pages, observer=snapshot["observer"],
                     max_nodes=snapshot["max_nodes"])
        if snapshot["source_root"] != result.source_root:
            raise ValueError("Counterfactual source or interpreter identity changed")
        for node in snapshot["nodes"]:
            result._inspect(node, verify_checkpoint=strict)
            if node["coordinate"] in result.nodes:
                raise ValueError("Duplicate causal state coordinate")
            result.nodes[node["coordinate"]] = deepcopy(node)
            result.head = _digest(FOREST, {"previous": result.head,
                                        "node_digest": node["node_digest"]})
        if result.head != snapshot["forest_head"]:
            raise ValueError("Recovered causal branch history diverged")
        return result

    def metrics(self):
        return {"qualified_model_scope": self.signature["layers"],
                "retained_states": len(self.nodes),
                "new_trained_transitions_this_receiver": self.executed_transitions,
                "reused_trained_transitions_this_receiver": self.reused_transitions,
                "full_64_layer_decoder_completed": False}
