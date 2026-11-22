"""TDL-005: acquire observer-specific bits through E-CPU IOS-003 live mTLS.

Only a live MeasurementPeerClient.resolve call can introduce a peer-attributed
observation into this adapter. Native ObserverReceipt digests in exported JSON
do not prove historical TLS identity; this module intentionally issues no
offline authentication certificate or private authority token.

The receiver owns source allowlists, source Ground semantics and VGI policy.
IOS-003 owns transport authentication, request/receipt checks, and durable
hosted measurement completion.
"""
from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Mapping

from .cost_aware_refinement import (
    GROUND_ID, SOURCE_STATE, acquire_vgi_refinement, observation_context,
)
from .cost_aware_refinement_verify import verify_cost_plan
from .problem_formed_interpreter import _hash

SCHEMA = "vl-i.ios003-vgi-refinement.v1"
SCOPE = "live-ios003-mtls-receiver-verified;exported-digests-not-source-signatures"
CHANNEL = "refinement.ios003.measurements"
PROTOCOL = "refinement.ios003.bit/1"


@dataclass(frozen=True)
class PeerMeasurementSpec:
    """Preinstalled IOS-003 model and exact observer contract for one source/row."""
    peer_id: str
    offer_id: str
    feature_id: str
    model: object
    public_value: int
    capabilities: tuple
    plan: dict
    contract: object
    subject: object
    scheduler: object
    challenge_digest: str
    authority_digest: str


class AuthenticatedRefinementRefused(ValueError):
    """A reading failed before a qualified processor state was released."""


def _offer_row(plan: dict, offer_id: str) -> int:
    if (type(plan) is not dict or type(offer_id) is not str
            or offer_id not in plan.get("selected_offer_ids", ())):
        raise ValueError("Observation was not selected for this intention")
    for offer in plan["offers"]:
        if offer["offer_id"] == offer_id:
            return int(offer["row_decimal"])
    raise ValueError("Selected observation is absent from plan catalogue")


def prepare_peer_measurement(
    plan: dict, *, old_native_state: int, offer_id: str,
    peer_id: str, authority_digest: str, scheduler,
    challenge_nonce: str,
) -> PeerMeasurementSpec:
    """Build IOS-002/003's exact native Ground, contract and request plan.

    The finite two-state IOS model describes the VALUE DOMAIN of one row;
    the row itself remains the separately admitted symbolic source-law
    contract. This does not enumerate an n-bit source's hidden states.
    """
    from mailcpu.intention_sufficiency import (
        FiniteGround, FiniteState, MeasurementCapability, plan_sufficiency,
    )
    from mailcpu.observer_orbit import ObserverContract
    from mailcpu.semantic_translation import (
        ObservationContract, RepresentationRef, canonical_value_digest,
    )
    from mailcpu.temporal import SchedulerReading
    if (type(peer_id) is not str or not peer_id or not isinstance(scheduler, SchedulerReading)
            or type(challenge_nonce) is not str or len(challenge_nonce) != 64
            or any(x not in "0123456789abcdef" for x in challenge_nonce)):
        raise ValueError("Exact peer, scheduler and 256-bit challenge are required")
    row = _offer_row(plan, offer_id)
    context = observation_context(plan, old_native_state=old_native_state)
    feature_id = "refinement." + offer_id
    subject = RepresentationRef.from_payload(
        "vl-i.refinement.source", 1, {
            "source_sha256": plan["old_source_sha256"],
            "new_source_sha256": plan["new_source_sha256"],
            "old_native_decimal": str(old_native_state),
            "plan_digest": plan["digest"],
        },
    )
    ground = RepresentationRef.from_payload(
        "vl-i.refinement.row-law", 1, {
            "source_sha256": plan["old_source_sha256"],
            "row_decimal": str(row),
            "offer_id": offer_id,
            "plan_digest": plan["digest"],
        },
    )
    public = ObservationContract.from_definition(
        "refinement-public", 1, {"public": 0},
    )
    target = ObservationContract.from_definition(
        "refinement-target", 1, {"target": feature_id},
    )
    observation = ObservationContract.from_definition(
        feature_id, 1, {
            "source_sha256": plan["old_source_sha256"],
            "row_decimal": str(row),
            "offer_id": offer_id,
            "plan_digest": plan["digest"],
        },
    )
    contract = ObserverContract(peer_id, 1, subject, ground, observation, context)
    model = FiniteGround(
        ground.content_digest, public.projection_digest,
        target.projection_digest, context,
        (
            FiniteState.create("0", 0, 0, {feature_id: 0}),
            FiniteState.create("1", 0, 1, {feature_id: 1}),
        ),
    )
    capabilities = (MeasurementCapability(
        feature_id, contract.digest, contract.observer_id, 1,
    ),)
    candidate = plan_sufficiency(model, 0, capabilities, authority_digest)
    if (candidate["status"] != "MEASURE"
            or candidate["selected_features"] != [feature_id]):
        raise ValueError("Native IOS-002 plan did not select the exact observer")
    challenge = canonical_value_digest({
        "schema": "vl-i.refinement-ios003-challenge.v1",
        "nonce": challenge_nonce, "context": context,
        "peer_id": peer_id, "offer_id": offer_id,
    })
    return PeerMeasurementSpec(
        peer_id, offer_id, feature_id, model, 0, capabilities,
        candidate, contract, subject, scheduler, challenge, authority_digest,
    )


def acquire_authenticated_refinement(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle, plan,
    *, old_native_state: int, clients: Mapping[str, object],
    required_sources: tuple[str, ...], authority_digest: str,
    scheduler, challenge_nonce: str | None = None,
) -> dict:
    """Perform synchronous native mTLS reads, then strict VGI source quorum.

    For every selected row, ALL pinned live peers must independently supply
    the same GF(2) value. Incomplete or conflicting readings never emit a
    processor-state receipt. The optional caller nonce permits an exact
    idempotent IOS-003 recovery test; ordinary calls generate a fresh nonce.
    """
    from mailcpu.intention_measurement_fabric import MeasurementPeerClient
    from mailcpu.observer_orbit import verify_observer_receipt
    from mailcpu.semantic_translation import canonical_value_digest
    from mailcpu.temporal import SchedulerReading
    from vgi.channels import ChannelContract, ChannelExchangeLedger, ChannelMessage
    from vgi.encounters import EncounterProvenance
    from vgi.evidence import (
        TransitionEvidenceLedger, TransitionEvidencePolicy, TransportFrameReceipt,
    )
    args = (old_source, old_quotient, old_bundle, new_source, new_quotient, new_bundle)
    if not verify_cost_plan(*args, plan):
        raise AuthenticatedRefinementRefused("Source-law refinement plan was not admitted")
    if (type(required_sources) is not tuple or len(required_sources) < 2
            or len(required_sources) > 8 or len(set(required_sources)) != len(required_sources)
            or type(clients) is not dict or set(clients) != set(required_sources)
            or any(not isinstance(clients[s], MeasurementPeerClient)
                   or clients[s].provider_id != s for s in required_sources)
            or not isinstance(scheduler, SchedulerReading)):
        raise AuthenticatedRefinementRefused("Independent pinned IOS-003 peer sessions are required")
    if challenge_nonce is None:
        challenge_nonce = secrets.token_hex(32)
    context = observation_context(plan, old_native_state=old_native_state)
    ordered_sources = tuple(sorted(required_sources))
    native: list[dict] = []
    for offer_id in plan["selected_offer_ids"]:
        for peer_id in ordered_sources:
            spec = prepare_peer_measurement(
                plan, old_native_state=old_native_state, offer_id=offer_id,
                peer_id=peer_id, authority_digest=authority_digest,
                scheduler=scheduler, challenge_nonce=challenge_nonce,
            )
            client = clients[peer_id]
            # resolve() admits only receipts obtained on this exact
            # mutually authenticated, CA-verified, peer-pinned live stream.
            try:
                report, receipts = client.resolve(
                    spec.model, spec.public_value, spec.capabilities,
                    spec.authority_digest, spec.plan,
                    {spec.feature_id: spec.contract}, spec.subject,
                    spec.scheduler, spec.challenge_digest,
                )
            except Exception as exc:
                raise AuthenticatedRefinementRefused(
                    f"Authenticated observation failed for {offer_id}/{peer_id}"
                ) from exc
            receipt = receipts.get(spec.feature_id)
            bit = None if receipt is None else receipt.projection
            if (not verify_observer_receipt(spec.contract, receipt)
                    or type(bit) is not int or bit not in (0, 1)
                    or report["resolution"]["result"] != bit
                    or report["resolution"]["remaining_candidates"] != 1
                    or report["measurement_receipts"][0]["observer_receipt_digest"]
                       != receipt.digest):
                raise AuthenticatedRefinementRefused(
                    "Peer reading differs from native IOS-003 observer contract"
                )
            native.append({
                "offer_id": offer_id, "peer_id": peer_id,
                "feature_id": spec.feature_id, "bit": bit,
                "observer_contract_digest": spec.contract.digest,
                "observer_receipt_digest": receipt.digest,
                "challenge_digest": spec.challenge_digest,
                "ios_resolution_digest": report["bridge_digest"],
                "ios_request_digest": receipt.input_digest,
                "model_digest": spec.model.digest,
            })

    policy = TransitionEvidencePolicy(
        GROUND_ID, context, ordered_sources, len(ordered_sources), True,
    )
    ledger = TransitionEvidenceLedger(policy)
    channel = ChannelContract(
        CHANNEL, PROTOCOL,
        observation_schema="vgi.transition-observation/1",
    )
    messages = []
    for index, item in enumerate(native):
        messages.append(ChannelMessage(
            f"ios003-observation-{index}", CHANNEL, PROTOCOL,
            "observation", {
                "status": "known", "source_state": SOURCE_STATE,
                "action": item["offer_id"], "target": SOURCE_STATE,
                "output": [item["bit"]],
                "ground_fingerprint": item["model_digest"],
            },
            EncounterProvenance(
                item["peer_id"],
                f"ios003-live://{item['observer_receipt_digest']}",
                collector="receiver-verified-tls-session",
            ),
        ))
    exchange = ChannelExchangeLedger()
    for message in messages:
        exchange = exchange.record_observation(channel, message)
    encounters = exchange.to_encounter_ledger()
    for index, message in enumerate(messages, 1):
        frame = TransportFrameReceipt.capture(
            message, transport_id="ios003-receiver-live-tls", sequence=index,
            frame_id=native[index - 1]["ios_request_digest"],
        )
        ledger = ledger.ingest_channel_message(
            exchange, message.message_id, frame, encounters,
        )
    for offer_id in plan["selected_offer_ids"]:
        ledger = ledger.finalize(SOURCE_STATE, offer_id)
        if ledger.decision(SOURCE_STATE, offer_id).status != "AUTHORIZED":
            raise AuthenticatedRefinementRefused(
                "VGI strict source quorum refused or disagreed on observation"
            )

    receipt = acquire_vgi_refinement(
        *args, plan, old_native_state=old_native_state, ledger=ledger,
    )
    body = {
        "schema": SCHEMA,
        "plan_digest": plan["digest"],
        "observation_context": context,
        "old_native_decimal": str(old_native_state),
        "required_sources": list(ordered_sources),
        "authority_digest": authority_digest,
        "scheduler": {
            "scheduler_id": scheduler.scheduler_id, "tick": scheduler.tick,
        },
        "challenge_nonce": challenge_nonce,
        "native_receipts": native,
        "vgi_ledger": ledger.to_dict(),
        "vgi_ledger_digest": ledger.fingerprint,
        "refinement_receipt": receipt,
        "scope": SCOPE,
    }
    result = {**body, "digest": _hash(body)}
    from .authenticated_refinement_verify import verify_authenticated_refinement_record
    if not verify_authenticated_refinement_record(*args, plan, result):
        raise AuthenticatedRefinementRefused("Independent record consistency audit refused")
    return result
