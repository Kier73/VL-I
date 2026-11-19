"""SHAPE-001: source-pinned connectome shape -> observer-qualified VL-I Ground.

Compose the *original* IFS connectome source parser and observer quotient,
VLD's navigable response renderer, and VL-I's intention-bound coordinates.

The 448-label OpenWorm wiring diagram is an observed heterogenous structure.
Its synthetic expansion is **not** a biological neural model or actual trained
language-model architecture. Source identity and the synthetic quotient are
kept separate. A low-dimensional observer quotient is admitted only after the
original IFS receiver verifies the complete PM=QP relation over every edge.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter_ns
from typing import Any
import json

from .ground import _digest, _hex64, _require
from .response_state import SourceSlot, ResponseSourceGround, ResponseFormation, compile_intention

SCHEMA = "vl-i.connectome-observer-shape.v1"
POSITION = "vl-i.connectome-observer-state-position.v1"
RESULT = "vl-i.connectome-observer-state-observation.v1"


class ConnectomeObserverShape:
    """Admit exact source and prove a smaller observer-specific computation."""

    def __init__(self, source_bytes: bytes, *, groups: int = 8,
                 copies: int = 16, seed: int = 73, original_source: bool = True):
        from connectome_ground import ConnectomeGround
        from observer_quotient import ExpandedSyntheticGround, discover, verify
        _require(type(source_bytes) is bytes and 0 < len(source_bytes) <= 2_000_000,
                 "Missing or excessive raw topology input")
        t0 = perf_counter_ns()
        self.source = ConnectomeGround.from_csv(source_bytes,
                                                require_original=original_source)
        t1 = perf_counter_ns()
        self.synthetic = ExpandedSyntheticGround(self.source, groups=groups,
                                                 copies=copies, seed=seed)
        t2 = perf_counter_ns()
        self.certificate = discover(self.synthetic, horizon=32)
        t3 = perf_counter_ns()
        self.audited = verify(self.synthetic, self.certificate,
                             replay_observations=True)
        t4 = perf_counter_ns()
        self.source_sha256 = sha256(source_bytes).hexdigest()
        self.source_bytes = source_bytes
        self.build_timings_ns = {
            "original_source_admission": t1-t0,
            "synthetic_shape_construction": t2-t1,
            "observer_signature_discovery": t3-t2,
            "independent_full_edge_proof": t4-t3,
        }
        self.dimension = len(self.audited[1])
        self.manifest = {
            "schema": SCHEMA,
            "original_openworm_source": original_source,
            "original_source_git_blob": self.source.source_blob,
            "source_sha256": self.source_sha256,
            "source_bytes": len(source_bytes),
            "source_label_count": len(self.source.nodes),
            "source_synapse_records": len(self.source.edges),
            "source_ground_fingerprint": self.source.fingerprint,
            "expanded_synthetic_ground": self.synthetic.fingerprint,
            "expanded_microstates": self.synthetic.n,
            "expanded_directed_relations": 8*self.synthetic.n,
            "observer_projection": self.certificate["readout"],
            "qualified_quotient_certificate": self.certificate["digest"],
            "qualified_observer_states": self.dimension,
            "proof": "IFS_exact_receiver_full_edge_PM_equals_QP",
            "domain": "GF_1000000007_synthetic_transitions",
        }
        self.manifest["ground_digest"] = _digest(SCHEMA, self.manifest)

    def verify(self, *, certificate=None):
        from observer_quotient import verify
        audited = verify(self.synthetic, self.certificate if certificate is None
                         else certificate, replay_observations=True)
        _require(audited == self.audited, "Observer Ground differs from its source proof")
        return True

    def position(self, *, source_micro: int, logical_tick: int, observer: str,
                 prompt: str, views: tuple[str, ...] = ("text", "page", "tokens")) -> dict:
        _require(type(source_micro) is int and 0 <= source_micro < self.synthetic.n
                 and type(logical_tick) is int and 0 <= logical_tick <= 2**56+73,
                 "Source coordinate/tick lies outside declared synthetic topology")
        _require(type(observer) is str and observer and len(observer) <= 80
                 and type(prompt) is str and 0 < len(prompt) <= 8192
                 and type(views) is tuple and views and
                 len(views) == len(set(views)) and
                 set(views).issubset({"text", "page", "tokens"}),
                 "Invalid intention and observation contract")
        body = {
            "schema": POSITION,
            "model_ground_digest": self.manifest["ground_digest"],
            "source_microstate": source_micro,
            "logical_tick": logical_tick,
            "observer": observer, "intention": prompt,
            "requested_views": list(views),
            "scope": "observer_relative_state_address_not_numeric_value",
        }
        root = _digest(POSITION, body)
        return {**body, "position_digest": root,
                "coordinate": "state://vli/" + root}

    def observe(self, position: dict, *, verify_short_horizon: bool = True) -> dict:
        """Interpret an intention using qualified k-state topology, then render it.

        A short-horizon full synthetic microstate computation is an *independent
        comparison*, not a substitute for the source-wide proof. Long-horizon
        observations use IFS's existing certified navigation machinery.
        """
        from observer_quotient import quotient_operator, quotient_at, micro_observation
        _require(type(position) is dict and position.get("schema") == POSITION
                 and position.get("model_ground_digest") == self.manifest["ground_digest"]
                 and position.get("position_digest") ==
                 _digest(POSITION, {k:v for k,v in position.items()
                                   if k not in ("position_digest", "coordinate")})
                 and position.get("coordinate") ==
                 "state://vli/" + position["position_digest"],
                 "Altered or missing state-position contract")
        source_micro, tick = position["source_microstate"], position["logical_tick"]
        t0 = perf_counter_ns()
        operator, source_class = quotient_operator(
            self.synthetic, self.certificate, source_micro, trusted=self.audited)
        if tick <= 16 and verify_short_horizon:
            # Full original synthetic network vs reduced quotient: same observer
            # vector at the requested tick; exact full-edge proof already holds
            # for all possible initial states and all logical ticks.
            x = [0] * self.dimension
            x[source_class] = 1
            visited_quotient_edges = 0
            for _ in range(tick):
                visited_quotient_edges += sum(len(operator.column_edges[i])
                                              for i, value in enumerate(x) if value)
                x = operator.step(x)
            t1 = perf_counter_ns()
            independent = micro_observation(self.synthetic, self.audited[0],
                                            source_micro, tick)
            t2 = perf_counter_ns()
            _require(x == independent,
                     "Original expanded microstates contradict admitted quotient")
            value = x
            navigation = {
                "method": "exact_quotient_sparse_steps",
                "observer_state_vector": x,
                "microstate_oracle_equal": True,
                "verified_full_microstate_edge_trajectory": True,
                "quotient_edge_operations_for_query": visited_quotient_edges,
                "full_microstate_edge_visits_upper_bound": 8*self.synthetic.n*tick,
                "quotient_execution_ns": t1-t0,
                "independent_microstate_reference_ns": t2-t1,
                "source_microstate_vectors_realized_by_quotient": 0,
            }
        else:
            # No full microstate trajectory is evaluated for the far-future
            # observation. A reusable independent global PM=QP proof qualifies
            # the compressed neural chart in the original IFS code.
            report = quotient_at(self.synthetic, self.certificate, source_micro,
                                 tick, depth=min(4,self.dimension-1),
                                 trusted=self.audited)
            t1 = perf_counter_ns()
            value = report["neural_result"]["nonlinear_readout"]
            navigation = {
                "method": "IFS_verified_distant_observer_quotient",
                "readout_mod_p": value,
                "neural_chart_receipt_digest": report["digest"],
                "polynomial_products": report["neural_result"]["polynomial_products"],
                "source_microstate_vectors_realized_by_quotient": report[
                    "microstate_vectors_realized_for_query"],
                "navigation_ns": t1-t0,
                "no_short_horizon_micro_oracle_run": True,
            }

        # The *state position* is not output value; the response is formed
        # from qualified interpretation and source/observer-specific laws.
        text = (
            f"The intention '{position['intention']}' observes a synthetic "
            f"connectome-shaped Ground with {self.synthetic.n} microstates "
            f"through an exactly qualified {self.dimension}-state observer "
            f"topology at logical tick {tick}. The result is {value}. "
            f"Source Git blob: {self.source.source_blob}. "
            "This is a constructed computational state-space, not a "
            "biological prediction or a pretrained model inference."
        )
        page = (
            f"# Intention-relative connectome-shaped state\n\n"
            f"Coordinate: {position['coordinate']}\n\n"
            f"Observer: {position['observer']}\n\n"
            f"Requested tick: {tick}\n\n"
            f"Full synthetic state dimension: {self.synthetic.n}\n\n"
            f"Verified observable dimension: {self.dimension}\n\n"
            f"Observable: {value}\n\n"
            f"Full edge proof: {self.certificate['digest']}\n"
        )
        payloads = {
            "text": text.encode("utf-8"),
            "page": page.encode("utf-8"),
            "tokens": json.dumps(list(text.encode("utf-8")),
                                 separators=(",", ":")).encode("utf-8"),
        }
        views = tuple(position["requested_views"])
        source = ResponseSourceGround(tuple(SourceSlot(v, v, payloads[v])
                                            for v in views))
        intent = compile_intention(
            position["intention"], slots=views, observer=position["observer"])
        formation = ResponseFormation(source, intent)
        while not formation.dag.is_complete:
            formation.advance(max_nodes=8)
        result = {
            "schema": RESULT,
            "coordinate": position["coordinate"],
            "position_digest": position["position_digest"],
            "model_ground_digest": self.manifest["ground_digest"],
            "qualified_quotient_digest": self.certificate["digest"],
            "value": value,
            "navigation": navigation,
            "source_render_ground_digest": source.manifest["ground_digest"],
            "views": formation.observe()["projections"],
            "rendered_bytes_observed": formation.materialized_bytes,
            "scope": "source_pinned_synthetic_connectome_shape_not_neural_language",
        }
        result["observation_digest"] = _digest(RESULT, result)
        return result
