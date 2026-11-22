"""TDL-005 independent *record consistency* verifier.

Reconstruct source-row IOS-002 contracts, request identities, VGI quorum and
the VL-I coordinate. Deliberately makes NO claim of verifying a historical TLS
session from exported JSON. Live peer authentication happens only inside the
MeasurementPeerClient used by authenticated_refinement.acquire_*.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re

from .cost_aware_refinement_verify import verify_cost_plan, verify_vgi_refinement_receipt


def _digest(body) -> str:
    return sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


def verify_authenticated_refinement_record(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle, plan, result,
) -> bool:
    """Check exact derivations without accepting report hashes as signatures."""
    try:
        from mailcpu.intention_sufficiency import (
            FiniteGround, FiniteState, MeasurementCapability, plan_sufficiency,
        )
        from mailcpu.intention_observer_bridge import issue_measurement_requests
        from mailcpu.observer_orbit import ObserverContract
        from mailcpu.semantic_translation import (
            ObservationContract, RepresentationRef, canonical_value_digest,
        )
        from mailcpu.temporal import SchedulerReading
        from vgi.evidence import TransitionEvidenceLedger

        args = (old_source, old_quotient, old_bundle,
                new_source, new_quotient, new_bundle)
        if not verify_cost_plan(*args, plan):
            return False
        keys = {
            "schema", "plan_digest", "observation_context",
            "old_native_decimal", "required_sources", "authority_digest",
            "scheduler", "challenge_nonce", "native_receipts",
            "vgi_ledger", "vgi_ledger_digest", "refinement_receipt",
            "scope", "digest",
        }
        if (type(result) is not dict or set(result) != keys
                or result["schema"] != "vl-i.ios003-vgi-refinement.v1"
                or result["scope"] !=
                "live-ios003-mtls-receiver-verified;exported-digests-not-source-signatures"
                or _digest({k: v for k, v in result.items() if k != "digest"})
                   != result["digest"]
                or result["plan_digest"] != plan["digest"]):
            return False
        old = result["old_native_decimal"]
        if (type(old) is not str or re.fullmatch(r"(0|[1-9][0-9]*)", old) is None
                or int(old) >= (1 << old_quotient.dimension)):
            return False
        context = _digest({
            "schema": "vl-i.refinement-observation-context.v1",
            "plan_digest": plan["digest"], "old_native_decimal": old,
        })
        if result["observation_context"] != context:
            return False
        peers = result["required_sources"]
        if (type(peers) is not list or not 2 <= len(peers) <= 8
                or peers != sorted(set(peers))
                or any(type(p) is not str or not p for p in peers)):
            return False
        nonce = result["challenge_nonce"]
        if (type(nonce) is not str or
                re.fullmatch(r"[0-9a-f]{64}", nonce) is None):
            return False
        scheduler_raw = result["scheduler"]
        if type(scheduler_raw) is not dict or set(scheduler_raw) != {"scheduler_id", "tick"}:
            return False
        scheduler = SchedulerReading(
            scheduler_raw["scheduler_id"], scheduler_raw["tick"],
        )
        authority = result["authority_digest"]
        expected_count = len(peers) * plan["observation_count"]
        native = result["native_receipts"]
        if type(native) is not list or len(native) != expected_count:
            return False
        expected = {}
        by_id = {o["offer_id"]: o for o in plan["offers"]}
        for offer_id in plan["selected_offer_ids"]:
            offer = by_id[offer_id]
            row = offer["row_decimal"]
            for peer_id in peers:
                feature = "refinement." + offer_id
                subject = RepresentationRef.from_payload(
                    "vl-i.refinement.source", 1, {
                        "source_sha256": plan["old_source_sha256"],
                        "new_source_sha256": plan["new_source_sha256"],
                        "old_native_decimal": old,
                        "plan_digest": plan["digest"],
                    },
                )
                ground = RepresentationRef.from_payload(
                    "vl-i.refinement.row-law", 1, {
                        "source_sha256": plan["old_source_sha256"],
                        "row_decimal": row,
                        "offer_id": offer_id,
                        "plan_digest": plan["digest"],
                    },
                )
                public = ObservationContract.from_definition(
                    "refinement-public", 1, {"public": 0},
                )
                target = ObservationContract.from_definition(
                    "refinement-target", 1, {"target": feature},
                )
                observation = ObservationContract.from_definition(
                    feature, 1, {
                        "source_sha256": plan["old_source_sha256"],
                        "row_decimal": row,
                        "offer_id": offer_id,
                        "plan_digest": plan["digest"],
                    },
                )
                contract = ObserverContract(
                    peer_id, 1, subject, ground, observation, context,
                )
                model = FiniteGround(
                    ground.content_digest, public.projection_digest,
                    target.projection_digest, context,
                    (
                        FiniteState.create("0", 0, 0, {feature: 0}),
                        FiniteState.create("1", 0, 1, {feature: 1}),
                    ),
                )
                capabilities = (MeasurementCapability(
                    feature, contract.digest, contract.observer_id, 1,
                ),)
                plan_native = plan_sufficiency(model, 0, capabilities, authority)
                challenge = canonical_value_digest({
                    "schema": "vl-i.refinement-ios003-challenge.v1",
                    "nonce": nonce, "context": context,
                    "peer_id": peer_id, "offer_id": offer_id,
                })
                request = issue_measurement_requests(
                    model, 0, capabilities, authority, plan_native,
                    {feature: contract}, subject, scheduler, challenge,
                )[0]
                expected[offer_id, peer_id] = {
                    "feature_id": feature,
                    "observer_contract_digest": contract.digest,
                    "challenge_digest": challenge,
                    "ios_request_digest": request["request_digest"],
                    "model_digest": model.digest,
                }
        seen = set()
        for entry in native:
            if (type(entry) is not dict or set(entry) != {
                "offer_id", "peer_id", "feature_id", "bit",
                "observer_contract_digest", "observer_receipt_digest",
                "challenge_digest", "ios_resolution_digest",
                "ios_request_digest", "model_digest",
            }):
                return False
            key = (entry["offer_id"], entry["peer_id"])
            if key not in expected or key in seen:
                return False
            seen.add(key)
            for k, value in expected[key].items():
                if entry[k] != value:
                    return False
            if (type(entry["bit"]) is not int or entry["bit"] not in (0, 1)
                    or any(type(entry[k]) is not str or
                           re.fullmatch(r"[0-9a-f]{64}", entry[k]) is None
                           for k in ("observer_receipt_digest", "ios_resolution_digest"))):
                return False
        if seen != set(expected):
            return False
        ledger = TransitionEvidenceLedger.from_dict(result["vgi_ledger"])
        if (ledger.fingerprint != result["vgi_ledger_digest"]
                or ledger.policy.ground_id != "pfi-refinement-observation"
                or ledger.policy.context_digest != context
                or ledger.policy.allowed_sources != tuple(peers)
                or ledger.policy.quorum != len(peers)
                or ledger.policy.strict is not True
                or len(ledger.evidence) != expected_count):
            return False
        evidence_seen = set()
        for item in ledger.evidence:
            key = (item.action, item.source)
            if (key not in expected or key in evidence_seen
                    or item.source_state != "refinement"
                    or item.status != "known"
                    or item.channel_id != "refinement.ios003.measurements"
                    or item.protocol_id != "refinement.ios003.bit/1"
                    or item.transport_id != "ios003-receiver-live-tls"
                    or item.source_model_fingerprint != expected[key]["model_digest"]
                    or item.correlation_id is not None
                    or item.frame_id != expected[key]["ios_request_digest"]):
                return False
            claim = next(e for e in native
                         if e["offer_id"] == item.action and e["peer_id"] == item.source)
            if (len(item.outcomes) != 1
                    or item.outcomes[0].target != "refinement"
                    or item.outcomes[0].output != (claim["bit"],)):
                return False
            evidence_seen.add(key)
        if evidence_seen != set(expected):
            return False
        for offer_id in plan["selected_offer_ids"]:
            if ledger.decision("refinement", offer_id).status != "AUTHORIZED":
                return False
        return verify_vgi_refinement_receipt(
            *args, plan, result["refinement_receipt"], ledger=ledger,
        )
    except (TypeError, ValueError, KeyError, IndexError, AttributeError,
            OverflowError, StopIteration):
        return False
