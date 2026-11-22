"""TDL-006 independent SAB route / native IOS peer identity record verifier.

No source producer planning function is called. Every possible candidate row
and required peer is re-planned with separate typed task construction. This
checks the authoritative mathematical plan and current receiver-local graph.
Exported JSON still does NOT prove historical TLS identity; only the producer's
live MeasurementPeerClient call checks the actual peer certificate.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re

from .cost_aware_refinement_verify import verify_cost_plan
from .authenticated_refinement_verify import verify_authenticated_refinement_record

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_DNS = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,125}[a-z0-9])?\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _hash(value):
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


def _name_pair(a, b):
    return ("refinement.mtls." + a + "." + b + ".v1",
            "refinement.peer-dispatch.result.v1." + a + "." + b)


def _task(offer, peer, epoch, old, ingress, collector):
    from mailcpu.substrate_bootstrap import BootstrapTask
    _, kind = _name_pair(offer.offer_id, peer.peer_id)
    return BootstrapTask(
        intention_id="observe-" + offer.offer_id + "-via-" + peer.peer_id,
        ground_digest=epoch,
        source=ingress, input_kind="refinement.peer-dispatch.request.v1",
        target=collector, output_kind=kind,
        value={
            "epoch_digest": epoch, "offer_id": offer.offer_id,
            "row_decimal": str(offer.row), "peer_id": peer.peer_id,
            "expected_dns": peer.expected_dns,
            "old_native_decimal": str(old),
        },
        max_quanta=100_000,
        max_steps=16, max_bytes=8192, require_oracle=True,
    )


def _expected_output(task):
    from mailcpu.substrate_bootstrap import digest
    value = task.value
    return {
        "request_digest": digest(value),
        "epoch_digest": value["epoch_digest"],
        "offer_id": value["offer_id"], "row_decimal": value["row_decimal"],
        "peer_id": value["peer_id"], "expected_dns": value["expected_dns"],
    }


def verify_routed_authenticated_plan(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    acquisition, runtime,
) -> bool:
    """Independently recompute all available peer tasks and least-cost rows."""
    try:
        from mailcpu.substrate_bootstrap import (
            AdaptiveBootstrap, BootstrapRefused,
        )
        from mailcpu.temporal import SchedulerReading
        from .cost_aware_refinement import ObservationOffer
        if not isinstance(runtime, AdaptiveBootstrap):
            return False
        old = acquisition.old_native_state
        coord = acquisition.coordinate
        if (type(old) is not int or not 0 <= old < (1 << old_quotient.dimension)
                or type(coord) is not int or coord < 0):
            return False
        if (old_source.width != new_source.width
                or old_source.origin != new_source.origin
                or old_source.generators != new_source.generators):
            return False
        peers = acquisition.pinned_peers
        if (type(peers) is not tuple or not 2 <= len(peers) <= 8
                or peers != tuple(sorted(peers, key=lambda p: p.peer_id))
                or len({p.peer_id for p in peers}) != len(peers)
                or len({p.expected_dns for p in peers}) != len(peers)
                or any(type(p.peer_id) is not str or _NAME.fullmatch(p.peer_id) is None
                       or type(p.expected_dns) is not str
                       or _DNS.fullmatch(p.expected_dns) is None for p in peers)):
            return False
        offers = acquisition.declared_offers
        if (type(offers) is not tuple or len(offers) > 12
                or any(not isinstance(o, ObservationOffer)
                       or o.row >= 1 << old_source.width for o in offers)
                or len({o.offer_id for o in offers}) != len(offers)):
            return False
        authority = acquisition.authority_digest
        nonce = acquisition.challenge_nonce
        sched = acquisition.scheduler
        if (type(authority) is not str or _HEX.fullmatch(authority) is None
                or type(nonce) is not str or _HEX.fullmatch(nonce) is None
                or not isinstance(sched, SchedulerReading)
                or type(acquisition.ingress) is not str
                or type(acquisition.collector) is not str):
            return False
        epoch = _hash({
            "schema": "vl-i.routed-authenticated-epoch.v1",
            "old_source": old_source.digest, "new_source": new_source.digest,
            "old_interpreter": old_bundle["digest"],
            "new_interpreter": new_bundle["digest"],
            "old_native_decimal": str(old),
            "inventory": runtime.inventory.fingerprint,
            "policy": runtime.policy.fingerprint,
            "coordinate": coord,
            "peers": [{
                "peer_id": p.peer_id, "expected_dns": p.expected_dns,
            } for p in peers],
            "authority_digest": authority,
            "scheduler": {
                "scheduler_id": sched.scheduler_id, "tick": sched.tick,
            },
            "challenge_nonce": nonce,
        })
        if acquisition.epoch_digest != epoch:
            return False
        from mailcpu.substrate_bootstrap import BootstrapPlan
        full_candidates = []
        eligible = {}
        for offer in sorted(offers, key=lambda x: x.offer_id):
            resolved = {}
            for peer in peers:
                task = _task(offer, peer, epoch, old,
                             acquisition.ingress, acquisition.collector)
                operation, _ = _name_pair(offer.offer_id, peer.peer_id)
                try:
                    plan = runtime.plan(task, coordinate=coord)
                except BootstrapRefused:
                    break
                realize = [x for x in plan.steps if x.kind == "realize"]
                if (len(realize) != 1 or realize[0].locus != peer.peer_id
                        or realize[0].operation != operation
                        or realize[0].quanta != offer.cost):
                    break
                resolved[peer.peer_id] = (task, plan)
            if len(resolved) != len(peers):
                continue
            price = sum(x[1].estimated_quanta for x in resolved.values())
            if not 1 <= price <= 1_000_000:
                continue
            full_candidates.append({
                "offer_id": offer.offer_id, "row_decimal": str(offer.row),
                "cost": price, "locus": offer.locus,
            })
            for peer in peers:
                eligible[offer.offer_id, peer.peer_id] = resolved[peer.peer_id]
        math_plan = acquisition.mathematical_plan
        if not verify_cost_plan(
            old_source, old_quotient, old_bundle,
            new_source, new_quotient, new_bundle, math_plan,
        ):
            return False
        if math_plan["offers"] != full_candidates:
            return False
        selected_keys = tuple((o, p.peer_id)
                              for o in math_plan["selected_offer_ids"]
                              for p in peers)
        if len(acquisition.requests) != len(selected_keys):
            return False
        for pair, (offer_id, peer_id) in zip(acquisition.requests, selected_keys):
            if (pair.offer.offer_id != offer_id
                    or pair.peer.peer_id != peer_id
                    or (offer_id, peer_id) not in eligible):
                return False
            task, plan = eligible[offer_id, peer_id]
            if (pair.offer != next(o for o in offers if o.offer_id == offer_id)
                    or pair.peer != next(p for p in peers if p.peer_id == peer_id)
                    or pair.task.fingerprint != task.fingerprint
                    or pair.plan.fingerprint != plan.fingerprint):
                return False
        return acquisition.fingerprint == _hash({
            "schema": "vl-i.routed-authenticated-plan.v1",
            "mathematical_plan_digest": math_plan["digest"],
            "epoch_digest": epoch,
            "old_native_decimal": str(old), "coordinate": coord,
            "ingress": acquisition.ingress, "collector": acquisition.collector,
            "offers": [o.encode() for o in offers],
            "peers": [{
                "peer_id": p.peer_id, "expected_dns": p.expected_dns,
            } for p in peers],
            "authority_digest": authority,
            "scheduler": {"scheduler_id": sched.scheduler_id, "tick": sched.tick},
            "challenge_nonce": nonce,
            "requests": [{
                "offer_id": k[0], "peer_id": k[1],
                "task_digest": eligible[k][0].fingerprint,
                "route_digest": eligible[k][1].fingerprint,
            } for k in selected_keys],
        })
    except (AttributeError, ValueError, TypeError, KeyError, OverflowError,
            StopIteration):
        return False


def verify_routed_authenticated_result(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    acquisition, runtime, result,
) -> bool:
    """Recheck selected SAB reports and an independently qualified TDL-005 record.

    No live TLS capability is provided to this verifier; exported report
    consistency must never be presented as portable network authentication.
    """
    try:
        from mailcpu.substrate_bootstrap import verify_bootstrap_report
        args = (old_source, old_quotient, old_bundle,
                new_source, new_quotient, new_bundle)
        if not verify_routed_authenticated_plan(*args, acquisition, runtime):
            return False
        if (type(result) is not dict or set(result) != {
            "schema", "plan_digest", "epoch_digest",
            "mathematical_plan_digest", "declared_total_route_quanta",
            "dispatch_reports", "authenticated_record",
            "authority_scope", "digest",
        } or _hash({k: v for k, v in result.items() if k != "digest"})
               != result["digest"]):
            return False
        plan = acquisition.mathematical_plan
        if (result["schema"] != "vl-i.routed-authenticated-refinement.v1"
                or result["plan_digest"] != acquisition.fingerprint
                or result["epoch_digest"] != acquisition.epoch_digest
                or result["mathematical_plan_digest"] != plan["digest"]
                or type(result["declared_total_route_quanta"]) is not int
                or result["declared_total_route_quanta"] != plan["total_cost"]
                or result["authority_scope"] !=
                "sab-local-dispatch-admission+ios003-live-peer-tls+vgi-strict-quorum"):
            return False
        reports = result["dispatch_reports"]
        if type(reports) is not list or len(reports) != len(acquisition.requests):
            return False
        for expected, item in zip(acquisition.requests, reports):
            if (type(item) is not dict or set(item) != {
                "offer_id", "peer_id", "task_digest", "route_digest", "report",
            } or item["offer_id"] != expected.offer.offer_id
                    or item["peer_id"] != expected.peer.peer_id
                    or item["task_digest"] != expected.task.fingerprint
                    or item["route_digest"] != expected.plan.fingerprint
                    or not verify_bootstrap_report(
                        item["report"], expected.task, expected.plan
                    ) or item["report"]["output"] != _expected_output(expected.task)):
                return False
        record = result["authenticated_record"]
        if not verify_authenticated_refinement_record(*args, plan, record):
            return False
        return (
            record["old_native_decimal"] == str(acquisition.old_native_state)
            and record["plan_digest"] == plan["digest"]
            and record["required_sources"] == [
                p.peer_id for p in acquisition.pinned_peers
            ]
            and record["authority_digest"] == acquisition.authority_digest
            and record["scheduler"] == {
                "scheduler_id": acquisition.scheduler.scheduler_id,
                "tick": acquisition.scheduler.tick,
            }
            and record["challenge_nonce"] == acquisition.challenge_nonce
            and set((x["offer_id"], x["peer_id"]) for x in record["native_receipts"])
                == set((r.offer.offer_id, r.peer.peer_id)
                       for r in acquisition.requests)
        )
    except (AttributeError, ValueError, TypeError, KeyError, OverflowError,
            IndexError):
        return False
