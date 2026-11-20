"""NAV-002: source- and implementation-bound exact causal Markov quotient.

Two *history coordinates* may hold the same numerical causal machine state.
If complete F64 recurrent/convolution/attention state, position, source and
operator implementation all agree, an already observed transition for one
state can predict the exact numerical output of another, previously unvisited
history for the SAME F64 input. This does NOT imply token-ID/embedding parity
or safe reuse across merely similar/rounded observations.

A proof navigates an existing mathematical transition relation; it does
not insert a synthetic "executed" edge into NAV-001's original evidence DAG.
"""
from __future__ import annotations

from hashlib import sha256
import inspect
import sys

import numpy as np

from experiments.model_fabric.complete_native_layers import QwenQualifiedLayer
from experiments.model_fabric.architecture_ops import (
    depthwise_causal_conv_step, qwen_gates, qwen_gated_delta_step,
    qwen_gated_rms_norm,
)
from experiments.model_fabric.trained_subgraphs import (
    centered_rms_norm, qwen_gated_delta_layer,
)
from .ground import _digest, _hex64
from .counterfactual_navigation import CounterfactualCausalGround, _pack
from .qwen_layer_chain import QwenCausalGDNChain
from .qwen_hybrid_block import QwenQualifiedHybridBlock
from .qwen_all_heads import QwenVectorizedCompleteLayer
from .qwen_gated_attention import QwenQualifiedGatedAttention, rotate_text_rope
from .qualified_projection_host import fast_rows

IMPLEMENTATION = "vl-i.exact-qwen-markov-implementation.v1"
STATE = "vl-i.exact-qwen-markov-state.v1"
PROOF = "vl-i.exact-qwen-markov-navigation-proof.v1"
STEP = "vl-i.exact-qwen-markov-navigation-step.v1"
MAX_PREDICTION_STEPS = 64


def _implementation_fingerprint():
    """Bind reuse to the concrete deterministic numerical operation sources.

    This is a same-implementation guard, not a formal proof that arbitrary
    Python/native numerical code is pure or deterministic.
    """
    operators = (
        QwenQualifiedLayer.next,
        QwenVectorizedCompleteLayer.next,
        QwenQualifiedGatedAttention.next,
        QwenCausalGDNChain.step,
        QwenQualifiedHybridBlock.step,
        fast_rows,
        centered_rms_norm,
        qwen_gated_delta_layer,
        depthwise_causal_conv_step,
        qwen_gates,
        qwen_gated_delta_step,
        qwen_gated_rms_norm,
        rotate_text_rope,
    )
    try:
        sources = [inspect.getsource(op) for op in operators]
    except (TypeError, OSError) as exc:
        raise ValueError("Qualified numerical interpreter source unavailable") from exc
    return _digest(IMPLEMENTATION, {
        "operator_sources": sources,
        "python_version": list(sys.version_info[:3]),
        "numpy_version": np.__version__,
        "floating_point": "python-binary64-and-qualified-numpy-worker",
    })


def _fresh_origin_is_empty(ground):
    chain = ground._fresh()
    for runner in chain.runners:
        if (runner.position != 0 or
                (isinstance(runner, QwenQualifiedLayer) and
                 (runner.convolution_history not in (None, []) or
                  runner.recurrent_state not in (None, []))) or
                (isinstance(runner, QwenQualifiedGatedAttention) and
                 (runner.kv_keys != [] or runner.kv_values != []))):
            raise ValueError("Undeclared nonempty genesis causal state")
    return chain


def _state_key(ground, node, fingerprint):
    """Exact state bytes, independent of ancestry and transport receipt heads.

    Re-admission of the full existing worker checkpoint is a prerequisite:
    these bytes are not blindly trusted because their page hashes match.
    """
    parts = node["checkpoint"]["layer_checkpoints"]
    if (type(parts) is not list or
            len(parts) != len(ground.signature["layers"])):
        raise ValueError("Incomplete source-qualified Markov state components")
    admitted = []
    total_bytes = 0
    for layer, cp in zip(ground.signature["layers"], parts):
        if (type(cp) is not dict or cp.get("layer") != layer or
                cp.get("position") != node["depth"]):
            raise ValueError("Wrong exact causal state layer/position")
        if cp.get("schema") == "vl64.qualified-native-orbit.v1":
            descriptor_key = "causal_pages"
            fields = ("key_heads", "value_heads", "key_dim", "value_dim",
                      "convolution_history_rows", "state_f64_bytes",
                      "causal_state_digest")
            extent = "state_f64_bytes"
        elif cp.get("schema") == "vl-i.qualified-original-attention-kv.v1":
            descriptor_key = "kv_pages"
            fields = ("query_heads", "kv_heads", "head_dim", "rotary_dim",
                      "rope_theta", "kv_f64_bytes", "kv_state_digest")
            extent = "kv_f64_bytes"
        else:
            raise ValueError("Unrecognized trained causal state equation")
        if type(cp.get(descriptor_key)) is not list or not cp[descriptor_key]:
            raise ValueError("Missing qualified causal state pages")
        raw = b"".join(ground.pages.get(part) for part in cp[descriptor_key])
        if type(cp.get(extent)) is not int or len(raw) != cp[extent]:
            raise ValueError("Incomplete exact recurrent or attention state bytes")
        total_bytes += len(raw)
        admitted.append({
            "layer": layer,
            "state_schema": cp["schema"],
            "geometry": {key:cp[key] for key in fields},
            "raw_f64_sha256": sha256(raw).hexdigest(),
        })
    body = {"source_root": ground.source_root, "implementation": fingerprint,
            "position": node["depth"], "parts": admitted}
    return _digest(STATE, body), total_bytes


class ExactCausalStateQuotient:
    """A certifying navigator over *previously executed* Qwen causal edges.

    A caller's externally trusted NAV-001 forest head fixes the evidence set.
    Unlike NAV-001, this can describe the numerical result at a coordinate
    that has NEVER been executed, if identical complete machine state and
    identical next-step F64 input witness an existing trained transition.

    The source history coordinate remains distinct. This class never
    publishes unexecuted trained arithmetic as an executed NAV-001 node.
    """

    def __init__(self, ground, *, expected_head):
        if not isinstance(ground, CounterfactualCausalGround):
            raise ValueError("Qualified NAV-001 causal execution forest required")
        self.ground = ground
        self.head = expected_head
        self._check_head()
        _fresh_origin_is_empty(ground)
        self.fingerprint = _implementation_fingerprint()
        self.root_key = _digest(STATE, {
            "source_root": ground.source_root,
            "implementation": self.fingerprint,
            "position": 0, "parts": [], "genesis": ground.signature["genesis"]})
        self.keys = {ground.root_coordinate: self.root_key}
        self.edges = {}
        self.page_bytes_verified = 0
        self.duplicate_equivalence_witnesses = 0
        self.numeric_relation_collisions_checked = 0
        # Topological NAV-001 insertion order is part of the authenticated head.
        # Every checkpoint is independently restored before its state key is used.
        for coordinate, node in ground.nodes.items():
            ground._inspect(node, verify_checkpoint=True)
            key, consumed = _state_key(ground, node, self.fingerprint)
            self.keys[coordinate] = key
            self.page_bytes_verified += consumed
            parent_key = self.keys.get(node["parent"])
            if parent_key is None:
                raise ValueError("Missing original Markov causal predecessor")
            relation = (parent_key, node["input_f64_digest"])
            if relation in self.edges:
                prior = ground.nodes[self.edges[relation]]
                # A deterministic state/input may be reached under different
                # provenance, but MUST not produce contradictory future state.
                if (prior["output_f64_digest"] != node["output_f64_digest"] or
                        self.keys[prior["coordinate"]] != key):
                    raise ValueError("Counterexample to exact Markov transition equivalence")
                self.numeric_relation_collisions_checked += 1
            else:
                self.edges[relation] = coordinate
        groups = {}
        for coordinate, key in self.keys.items():
            groups.setdefault(key, []).append(coordinate)
        self.classes = {k:tuple(v) for k,v in groups.items()}
        self.duplicate_equivalence_witnesses = sum(
            len(v)-1 for v in self.classes.values())

    def _check_head(self):
        if not _hex64(self.head) or self.ground.head != self.head:
            raise ValueError("Stale or unpinned original trained causal forest head")
        self.ground._audit()

    def _anchor(self, parent, digest):
        ground = self.ground
        if parent == ground.root_coordinate:
            actual_digest = ground.root_digest
        else:
            if parent not in ground.nodes:
                raise ValueError("Unknown exact causal anchor coordinate")
            actual_digest = ground.nodes[parent]["node_digest"]
        if digest != actual_digest:
            raise ValueError("Stale or substituted exact Markov causal parent pin")
        return self.keys[parent]

    def _resolve(self, parent, parent_digest, specifications):
        self._check_head()
        state_key = self._anchor(parent, parent_digest)
        if (type(specifications) not in (tuple, list) or
                not 1 <= len(specifications) <= MAX_PREDICTION_STEPS):
            raise ValueError("Invalid finite counterfactual navigation extent")
        steps = []
        current = parent
        previous = parent_digest
        original = self.ground
        for index, (token_id, input_digest) in enumerate(specifications):
            if (type(token_id) is not int or not 0 <= token_id < 2**31 or
                    not _hex64(input_digest)):
                raise ValueError("Malformed exact input or token-label intention")
            source = self.edges.get((state_key, input_digest))
            if source is None:
                raise ValueError("UNRESOLVED: no equivalent trained future was observed")
            node = original.nodes[source]
            next_key = self.keys[source]
            projected = original._coordinate(current, token_id, input_digest)
            if projected in original.nodes:
                actual = original.nodes[projected]
                if (actual["output_f64_digest"] != node["output_f64_digest"] or
                        self.keys[projected] != next_key):
                    raise ValueError("Known original causal coordinate contradicts quotient")
            body = {
                "schema": STEP, "index": index, "logical_parent": current,
                "logical_predecessor": previous,
                "target_coordinate": projected, "token_id": token_id,
                "input_f64_digest": input_digest,
                "parent_exact_state_key": state_key,
                "representative_coordinate": source,
                "representative_node_digest": node["node_digest"],
                "representative_parent": node["parent"],
                "result_f64_digest": node["output_f64_digest"],
                "next_exact_state_key": next_key,
                "trained_layer_executions_for_this_navigation": 0,
            }
            step = {**body, "step_digest": _digest(STEP, body)}
            steps.append(step)
            current, previous, state_key = projected, step["step_digest"], next_key
        body = {
            "schema": PROOF, "source_root": original.source_root,
            "forest_head": self.head,
            "implementation_fingerprint": self.fingerprint,
            "initial_parent": parent,
            "initial_parent_digest": parent_digest,
            "steps": steps,
            "scope": "exact_markov_state_same_f64_inputs_and_fixed_interpreter_only",
            "native_trained_transitions_executed_by_navigation": 0,
            "original_decoder_completed": False,
        }
        return {**body, "proof_digest": _digest(PROOF, body)}

    def navigate(self, parent, actions, *, expected_parent_digest, expected_head):
        """Locate a previously *unexecuted* future by exact known state law.

        Action = (token-ID label, whole F64 hidden input). Token-ID does not
        authenticate this input as the original embedding of that token.
        """
        if expected_head != self.head:
            raise ValueError("Stale externally pinned predictive navigation head")
        specs = []
        if type(actions) not in (tuple, list) or not actions:
            raise ValueError("Counterfactual action sequence required")
        for action in actions:
            if not isinstance(action, (tuple,list)) or len(action) != 2:
                raise ValueError("Malformed original-coordinate navigation action")
            token, hidden = action
            raw, digest = _pack(hidden)
            if len(raw) != self.ground.signature["width"]*8:
                raise ValueError("Wrong pretrained hidden dimension")
            specs.append((token,digest))
        proof = self._resolve(parent,expected_parent_digest,specs)
        return proof, self._outputs(proof)

    def _outputs(self, proof):
        return [
            {"coordinate":step["target_coordinate"],
             "numerical_output":self.ground._inspect(
                 self.ground.nodes[step["representative_coordinate"]],
                 verify_checkpoint=False),
             "source_observation_coordinate":step["representative_coordinate"],
             "was_originally_executed":step["target_coordinate"] in self.ground.nodes}
            for step in proof["steps"]
        ]

    def verify(self, proof, *, expected_digest, expected_head,
               expected_parent_digest):
        """Independently reconstruct every navigated equivalence witness."""
        self._check_head()
        if (expected_head != self.head or type(proof) is not dict or
                not _hex64(expected_digest) or
                proof.get("proof_digest") != expected_digest or
                _digest(PROOF, {k:v for k,v in proof.items()
                         if k != "proof_digest"}) != expected_digest or
                type(proof.get("steps")) is not list):
            raise ValueError("Unpinned or altered exact Markov navigation proof")
        try:
            specs = [(step["token_id"],step["input_f64_digest"])
                     for step in proof["steps"]]
            rebuilt = self._resolve(proof["initial_parent"],
                                    expected_parent_digest, specs)
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed exact numerical navigation witness") from exc
        if rebuilt != proof:
            raise ValueError("Exact state-bisimulation proof disagrees with observed Ground")
        return self._outputs(proof)

    def metrics(self):
        return {
            "original_qualified_transitions":len(self.ground.nodes),
            "unique_exact_markov_states":len(self.classes),
            "merged_history_state_witnesses":self.duplicate_equivalence_witnesses,
            "unique_observed_numerical_relations":len(self.edges),
            "duplicate_relation_consistency_checks":self.numeric_relation_collisions_checked,
            "full_causal_page_bytes_hashed_to_admit_quotient":self.page_bytes_verified,
            "new_trained_arithmetic_during_navigate":0,
            "unseen_input_vectors_predictable":False,
            "whole_model_decoder_completed":False,
        }
