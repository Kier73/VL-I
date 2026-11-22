"""Independent TDL-004 verifier for admitted E-CPU route and VGI refinement.

Rechecks source/plan/cost identities, current SAB leases and permissions, typed
path reports, their original handler-oracle flags and VGI finalized evidence.
No attempt is made to treat unkeyed digests as cryptographic authentication.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .cost_aware_refinement_verify import verify_cost_plan, verify_vgi_refinement_receipt


def _digest(obj: dict) -> str:
    return sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


def verify_fabric_refinement(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    acquisition, runtime, result,
) -> bool:
    """Audit native path reports and finalized VGI event result without replay."""
    try:
        from mailcpu.substrate_bootstrap import (
            AdaptiveBootstrap, BootstrapRefused, BootstrapTask,
            canonical, digest, verify_bootstrap_report,
        )
        from .cost_aware_refinement import ObservationOffer
        from vgi.evidence import TransitionEvidenceLedger
        if not isinstance(runtime, AdaptiveBootstrap):
            return False
        if (type(result) is not dict or set(result) != {
            "schema", "fabric_plan_digest", "epoch_digest",
            "mathematical_plan_digest", "declared_total_quanta",
            "measured_elapsed_ns", "measured_application_bytes",
            "path_reports", "vgi_ledger", "vgi_ledger_digest",
            "refinement_receipt", "authority_scope", "digest",
        }):
            return False
        if _digest({k:v for k,v in result.items() if k != "digest"}) != result["digest"]:
            return False
        args = (old_source, old_quotient, old_bundle,
                new_source, new_quotient, new_bundle)
        plan = acquisition.mathematical_plan
        if not verify_cost_plan(*args, plan):
            return False
        expected_epoch = _digest({
            "schema": "vl-i.fabric-refinement-epoch.v1",
            "old_source": old_source.digest,
            "new_source": new_source.digest,
            "old_interpreter": old_bundle["digest"],
            "new_interpreter": new_bundle["digest"],
            "old_native_decimal": str(acquisition.old_native_state),
            "inventory": runtime.inventory.fingerprint,
            "policy": runtime.policy.fingerprint,
            "coordinate": acquisition.coordinate,
        })
        if (result["schema"] != "vl-i.fabric-refinement.v1"
                or acquisition.epoch_digest != expected_epoch
                or result["epoch_digest"] != expected_epoch
                or result["mathematical_plan_digest"] != plan["digest"]
                or result["fabric_plan_digest"] != acquisition.fingerprint
                or result["authority_scope"] !=
                "receiver-owned-sab-handler/carrier+single-VGI-adapter"
                or type(result["declared_total_quanta"]) is not int
                or result["declared_total_quanta"] != plan["total_cost"]
                or type(result["measured_elapsed_ns"]) is not int
                or type(result["measured_application_bytes"]) is not int
                or result["measured_elapsed_ns"] < 0
                or result["measured_application_bytes"] < 0):
            return False
        # Recompute ALL available candidate routes, not only the selected
        # ones: omitting a cheaper admissible candidate must fail the audit.
        candidates = acquisition.declared_offers
        if (type(candidates) is not tuple or len(candidates) > 12
                or any(not isinstance(o, ObservationOffer) for o in candidates)
                or len({o.offer_id for o in candidates}) != len(candidates)):
            return False
        effective = []
        expected_route_by_id = {}
        for offered in sorted(candidates, key=lambda o:o.offer_id):
            if offered.row >= 1 << old_source.width:
                return False
            operation = "refinement.observe." + offered.offer_id + ".v1"
            if any(key[1] == operation and key[0] != offered.locus
                   for key in runtime.realizers):
                return False
            expected_task = BootstrapTask(
                intention_id="observe-" + offered.offer_id,
                ground_digest=expected_epoch,
                source=acquisition.ingress, input_kind="refinement.request.v1",
                target=acquisition.collector, output_kind="refinement.observation.v1." + offered.offer_id,
                value={
                    "epoch_digest": expected_epoch,
                    "offer_id": offered.offer_id,
                    "row_decimal": str(offered.row),
                    "old_native_decimal": str(acquisition.old_native_state),
                },
                max_quanta=100_000, max_steps=16, max_bytes=8192,
                require_oracle=True,
            )
            try:
                route = runtime.plan(expected_task, coordinate=acquisition.coordinate)
            except BootstrapRefused:
                continue
            realizers = [step for step in route.steps if step.kind == "realize"]
            if (len(realizers) != 1 or realizers[0].locus != offered.locus
                    or realizers[0].operation != operation
                    or realizers[0].quanta != offered.cost):
                continue
            effective.append({
                "offer_id": offered.offer_id, "row_decimal": str(offered.row),
                "cost": route.estimated_quanta, "locus": offered.locus,
            })
            expected_route_by_id[offered.offer_id] = (expected_task, route)
        if plan["offers"] != effective:
            return False

        routes = acquisition.routes
        reports = result["path_reports"]
        if (type(reports) is not list or len(reports) != len(routes)
                or len(routes) != plan["observation_count"]
                or [r.offer.offer_id for r in routes] != plan["selected_offer_ids"]):
            return False
        actual_bits = []
        elapsed_ns = 0
        payload_bytes = 0
        for route, record in zip(routes, reports):
            offered = route.offer
            task = route.task
            if offered.offer_id not in expected_route_by_id:
                return False
            expected_task, expected_route = expected_route_by_id[offered.offer_id]
            if (task.fingerprint != expected_task.fingerprint
                    or route.plan.fingerprint != expected_route.fingerprint):
                return False
            current_plan = runtime.plan(task, coordinate=acquisition.coordinate)
            if (current_plan.fingerprint != route.plan.fingerprint
                    or task.ground_digest != expected_epoch
                    or task.source != acquisition.ingress
                    or task.target != acquisition.collector
                    or task.value != {
                        "epoch_digest": expected_epoch,
                        "offer_id": offered.offer_id,
                        "row_decimal": str(offered.row),
                        "old_native_decimal": str(acquisition.old_native_state),
                    }):
                return False
            realizations = [s for s in route.plan.steps if s.kind == "realize"]
            if (len(realizations) != 1 or realizations[0].locus != offered.locus
                    or realizations[0].operation != "refinement.observe." + offered.offer_id + ".v1"
                    or realizations[0].quanta != offered.cost):
                return False
            if (type(record) is not dict or set(record) != {
                    "offer_id", "bit", "report", "elapsed_ns",
                    "application_input_bytes", "application_output_bytes",
                } or record["offer_id"] != offered.offer_id
                or type(record["bit"]) is not int or record["bit"] not in (0,1)
                or type(record["elapsed_ns"]) is not int or record["elapsed_ns"] < 0):
                return False
            report = record["report"]
            if not verify_bootstrap_report(report, task, route.plan):
                return False
            value = report["output"]
            if (type(value) is not dict or set(value) != {
                    "request_digest", "offer_id", "row_decimal", "bit"
                } or value != {
                    "request_digest": digest(task.value),
                    "offer_id": offered.offer_id,
                    "row_decimal": str(offered.row),
                    "bit": record["bit"],
                }):
                return False
            input_bytes = len(canonical(task.value))
            output_bytes = len(canonical(value))
            if (record["application_input_bytes"] != input_bytes
                    or record["application_output_bytes"] != output_bytes):
                return False
            actual_bits.append(record["bit"])
            elapsed_ns += record["elapsed_ns"]
            payload_bytes += input_bytes + output_bytes
        if (elapsed_ns != result["measured_elapsed_ns"]
                or payload_bytes != result["measured_application_bytes"]):
            return False
        ledger = None
        if routes:
            if type(result["vgi_ledger"]) is not dict:
                return False
            ledger = TransitionEvidenceLedger.from_dict(result["vgi_ledger"])
            if (ledger.fingerprint != result["vgi_ledger_digest"]
                    or ledger.policy.allowed_sources != ("receiver-admitted-fabric",)
                    or ledger.policy.quorum != 1 or ledger.policy.strict is not True):
                return False
            for offer_id, bit in zip(plan["selected_offer_ids"], actual_bits):
                observed = ledger.transition_observation("refinement", offer_id)
                if (len(observed.outcomes) != 1
                        or observed.outcomes[0].target != "refinement"
                        or observed.outcomes[0].output != (bit,)):
                    return False
            # Bind every VGI encounter to the E-CPU execution report that
            # triggered this receiver-local evidence adapter.
            by_id = {x["offer_id"]: x["report"]["report_digest"] for x in reports}
            for item in ledger.evidence:
                if (item.action not in by_id
                        or item.source != "receiver-admitted-fabric"
                        or item.source_state != "refinement"):
                    return False
                # Actual cross-reference to report identity is preserved in
                # the associated Encounter, which VGI rechecks on ingestion;
                # the serialized evidence retains encounter ID and message ID.
        elif result["vgi_ledger"] is not None or result["vgi_ledger_digest"] is not None:
            return False
        return verify_vgi_refinement_receipt(
            *args, plan, result["refinement_receipt"], ledger=ledger,
        )
    except (TypeError, ValueError, KeyError, AttributeError, IndexError, OverflowError):
        return False
