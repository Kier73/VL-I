"""TDL-006: bind SAB-001 admitted endpoint to IOS-003 live mTLS peer.

The SAB-001 path is receiver-owned *dispatch admission* with locally bound
carriers. It is NOT falsely described as the network route used by IOS-003.
The exact endpoint ID must agree with the peer pinned by the live TLS socket,
and one IOS-003 measurement per (selected row, pinned peer) is subsequently
required for a VGI strict quorum.

No new scheduler, TLS stack, source authority, or VGI consensus is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .cost_aware_refinement import (
    ObservationOffer, plan_cost_aware_observations, observation_context,
    _scope_check,
)
from .cost_aware_refinement_verify import verify_cost_plan
from .problem_formed_interpreter import _hash
from .authenticated_refinement import acquire_authenticated_refinement

SCHEMA = "vl-i.routed-authenticated-refinement.v1"
PLAN_SCHEMA = "vl-i.routed-authenticated-plan.v1"
EPOCH_SCHEMA = "vl-i.routed-authenticated-epoch.v1"
SCOPE = "sab-local-dispatch-admission+ios003-live-peer-tls+vgi-strict-quorum"
INPUT_KIND = "refinement.peer-dispatch.request.v1"
MAX_OFFERS = 12
MAX_PEERS = 8
MAX_QUANTA = 100_000
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_DNS = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,125}[a-z0-9])?\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PinnedPeer:
    """Exact receiver-authorized resource and mTLS endpoint identity."""
    peer_id: str
    expected_dns: str

    def __post_init__(self):
        if (type(self.peer_id) is not str or _NAME.fullmatch(self.peer_id) is None
                or type(self.expected_dns) is not str
                or _DNS.fullmatch(self.expected_dns) is None):
            raise ValueError("Pinned peer and DNS identity must be exact, bounded names")

    def to_dict(self):
        return {"peer_id": self.peer_id, "expected_dns": self.expected_dns}


@dataclass(frozen=True)
class RoutedPeerRequest:
    """One admitted SAB route to the same peer required by IOS-003."""
    offer: ObservationOffer
    peer: PinnedPeer
    task: object
    plan: object


@dataclass(frozen=True)
class RoutedAuthenticatedPlan:
    """Epoch-fenced receiver-local route catalogue and selected peer tasks."""
    mathematical_plan: dict
    epoch_digest: str
    old_native_state: int
    coordinate: int
    ingress: str
    collector: str
    declared_offers: tuple[ObservationOffer, ...]
    pinned_peers: tuple[PinnedPeer, ...]
    authority_digest: str
    scheduler: object
    challenge_nonce: str
    requests: tuple[RoutedPeerRequest, ...]

    @property
    def fingerprint(self):
        return _hash({
            "schema": PLAN_SCHEMA,
            "mathematical_plan_digest": self.mathematical_plan["digest"],
            "epoch_digest": self.epoch_digest,
            "old_native_decimal": str(self.old_native_state),
            "coordinate": self.coordinate,
            "ingress": self.ingress, "collector": self.collector,
            "offers": [o.encode() for o in self.declared_offers],
            "peers": [p.to_dict() for p in self.pinned_peers],
            "authority_digest": self.authority_digest,
            "scheduler": {
                "scheduler_id": self.scheduler.scheduler_id,
                "tick": self.scheduler.tick,
            },
            "challenge_nonce": self.challenge_nonce,
            "requests": [{
                "offer_id": pair.offer.offer_id, "peer_id": pair.peer.peer_id,
                "task_digest": pair.task.fingerprint,
                "route_digest": pair.plan.fingerprint,
            } for pair in self.requests],
        })


def observer_operation(offer_id: str, peer_id: str) -> str:
    """Bind both source question and peer to the exact receiver operation."""
    return "refinement.mtls." + offer_id + "." + peer_id + ".v1"


def observer_output_kind(offer_id: str, peer_id: str) -> str:
    """Avoid a cheaper, different question or peer satisfying a typed route."""
    return "refinement.peer-dispatch.result.v1." + offer_id + "." + peer_id


def _validated_inputs(args, offers, pinned_peers, runtime, *, old_native_state,
                      coordinate, authority_digest, scheduler, challenge_nonce):
    from mailcpu.substrate_bootstrap import AdaptiveBootstrap
    from mailcpu.temporal import SchedulerReading
    _scope_check(*args)
    if not isinstance(runtime, AdaptiveBootstrap):
        raise ValueError("Receiver-owned E-CPU SAB runtime required")
    if (type(offers) is not tuple or len(offers) > MAX_OFFERS
            or any(not isinstance(o, ObservationOffer) for o in offers)
            or len({o.offer_id for o in offers}) != len(offers)
            or any(o.row >= (1 << args[0].width) for o in offers)):
        raise ValueError("Bounded unique source-row offers required")
    if (type(pinned_peers) is not tuple
            or not 2 <= len(pinned_peers) <= MAX_PEERS
            or any(not isinstance(p, PinnedPeer) for p in pinned_peers)
            or len({p.peer_id for p in pinned_peers}) != len(pinned_peers)
            or len({p.expected_dns for p in pinned_peers}) != len(pinned_peers)):
        raise ValueError("Two or more different pinned peer/DNS identities required")
    if (type(old_native_state) is not int
            or not 0 <= old_native_state < (1 << args[1].dimension)
            or type(coordinate) is not int or coordinate < 0
            or not isinstance(scheduler, SchedulerReading)
            or type(authority_digest) is not str or _HEX.fullmatch(authority_digest) is None
            or type(challenge_nonce) is not str or _HEX.fullmatch(challenge_nonce) is None):
        raise ValueError("Canonical old state, coordinate, scheduler, authority and fresh nonce required")


def _epoch(args, runtime, *, old_native_state, coordinate,
           peers, authority_digest, scheduler, challenge_nonce):
    return _hash({
        "schema": EPOCH_SCHEMA,
        "old_source": args[0].digest, "new_source": args[3].digest,
        "old_interpreter": args[2]["digest"],
        "new_interpreter": args[5]["digest"],
        "old_native_decimal": str(old_native_state),
        "inventory": runtime.inventory.fingerprint,
        "policy": runtime.policy.fingerprint,
        "coordinate": coordinate,
        "peers": [p.to_dict() for p in peers],
        "authority_digest": authority_digest,
        "scheduler": {
            "scheduler_id": scheduler.scheduler_id,
            "tick": scheduler.tick,
        },
        "challenge_nonce": challenge_nonce,
    })


def _task(offer, peer, *, epoch, old_native_state, ingress, collector):
    from mailcpu.substrate_bootstrap import BootstrapTask
    return BootstrapTask(
        intention_id="observe-" + offer.offer_id + "-via-" + peer.peer_id,
        ground_digest=epoch,
        source=ingress, input_kind=INPUT_KIND,
        target=collector,
        output_kind=observer_output_kind(offer.offer_id, peer.peer_id),
        value={
            "epoch_digest": epoch, "offer_id": offer.offer_id,
            "row_decimal": str(offer.row), "peer_id": peer.peer_id,
            "expected_dns": peer.expected_dns,
            "old_native_decimal": str(old_native_state),
        },
        max_quanta=MAX_QUANTA,
        max_steps=16, max_bytes=8192, require_oracle=True,
    )


def _qualify(offer, peer, task, runtime, coordinate):
    """Use SAB-001's *actual* least-cost path and require exact operator/peer."""
    from mailcpu.substrate_bootstrap import BootstrapRefused
    try:
        route = runtime.plan(task, coordinate=coordinate)
    except BootstrapRefused:
        return None
    operations = [x for x in route.steps if x.kind == "realize"]
    if (len(operations) != 1
            or operations[0].locus != peer.peer_id
            or operations[0].operation != observer_operation(
                offer.offer_id, peer.peer_id
            )
            or operations[0].quanta != offer.cost):
        return None
    return route


def plan_routed_authenticated_refinement(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    offers: tuple[ObservationOffer, ...],
    runtime,
    *,
    pinned_peers: tuple[PinnedPeer, ...],
    ingress: str, collector: str,
    old_native_state: int,
    coordinate: int,
    authority_digest: str,
    scheduler,
    challenge_nonce: str,
) -> RoutedAuthenticatedPlan:
    """Minimize sum of full receiver-admitted route quanta across all peers.

    An offered row is usable only if *every* required peer has a distinct
    SAB-admitted route to the exact operation and resource identity. Candidate
    sets are bounded; TDL-003 proves exact source-row sufficiency and optimum.
    """
    args = (old_source, old_quotient, old_bundle,
            new_source, new_quotient, new_bundle)
    _validated_inputs(args, offers, pinned_peers, runtime,
                      old_native_state=old_native_state, coordinate=coordinate,
                      authority_digest=authority_digest, scheduler=scheduler,
                      challenge_nonce=challenge_nonce)
    peers = tuple(sorted(pinned_peers, key=lambda x: x.peer_id))
    epoch = _epoch(
        args, runtime, old_native_state=old_native_state,
        coordinate=coordinate, peers=peers,
        authority_digest=authority_digest,
        scheduler=scheduler, challenge_nonce=challenge_nonce,
    )
    available = []
    by_key = {}
    for offer in sorted(offers, key=lambda x: x.offer_id):
        admitted = []
        for peer in peers:
            task = _task(
                offer, peer, epoch=epoch,
                old_native_state=old_native_state,
                ingress=ingress, collector=collector,
            )
            route = _qualify(offer, peer, task, runtime, coordinate)
            if route is None:
                break
            admitted.append(RoutedPeerRequest(offer, peer, task, route))
        if len(admitted) != len(peers):
            continue
        price = sum(item.plan.estimated_quanta for item in admitted)
        if not 1 <= price <= 1_000_000:
            continue
        available.append(ObservationOffer(
            offer.offer_id, offer.row, price, offer.locus,
        ))
        for item in admitted:
            by_key[offer.offer_id, item.peer.peer_id] = item
    math_plan = plan_cost_aware_observations(*args, tuple(available))
    if not verify_cost_plan(*args, math_plan):
        raise ValueError("Independent mathematical cost audit refused")
    selected = tuple(
        by_key[offer_id, peer.peer_id]
        for offer_id in math_plan["selected_offer_ids"]
        for peer in peers
    )
    return RoutedAuthenticatedPlan(
        math_plan, epoch, old_native_state, coordinate, ingress, collector,
        offers, peers, authority_digest, scheduler, challenge_nonce, selected,
    )


def _dispatch_output(task):
    """The SAB handler authorizes dispatch; IOS-003 supplies the actual bit."""
    from mailcpu.substrate_bootstrap import digest
    value = task.value
    return {
        "request_digest": digest(value),
        "epoch_digest": value["epoch_digest"],
        "offer_id": value["offer_id"],
        "row_decimal": value["row_decimal"],
        "peer_id": value["peer_id"],
        "expected_dns": value["expected_dns"],
    }


def execute_routed_authenticated_refinement(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    acquisition: RoutedAuthenticatedPlan, runtime,
    *, clients: dict,
) -> dict:
    """Use admitted dispatch paths, then IOS live TLS, then strict VGI quorum.

    This function does not attest that SAB's locally bound carrier channels
    are the physical IP links carrying the IOS TLS frames. Its guarantee is
    that each authorized dispatch endpoint is exactly the mTLS-pinned peer.
    """
    from mailcpu.intention_measurement_fabric import (
        MeasurementPeerClient, _bound_transport,
    )
    from mailcpu.substrate_bootstrap import verify_bootstrap_report, BootstrapRefused
    from .routed_authenticated_verify import (
        verify_routed_authenticated_plan,
        verify_routed_authenticated_result,
    )
    args = (old_source, old_quotient, old_bundle,
            new_source, new_quotient, new_bundle)
    if (not isinstance(acquisition, RoutedAuthenticatedPlan)
            or not verify_routed_authenticated_plan(*args, acquisition, runtime)):
        raise ValueError("Receiver route/epoch plan is not currently admitted")
    if type(clients) is not dict or set(clients) != {
        peer.peer_id for peer in acquisition.pinned_peers
    }:
        raise ValueError("Exact pinned live peer catalogue required")
    for peer in acquisition.pinned_peers:
        client = clients[peer.peer_id]
        if not isinstance(client, MeasurementPeerClient):
            raise ValueError("An exported receipt cannot replace a live IOS client")
        # This re-checks actual TLS protocol, CERT_REQUIRED, peer transport
        # endpoints and CA-verified certificate DNS SAN, not merely client IDs.
        _bound_transport(
            client.transport, client.peer_id, peer.peer_id, peer.expected_dns,
        )

    dispatch_reports = []
    for request in acquisition.requests:
        # Each task represents a receiver-authorized dispatch to one exact
        # peer-specific observer operation, not an unverified measurement bit.
        if (runtime.plan(request.task, coordinate=acquisition.coordinate).fingerprint
                != request.plan.fingerprint):
            raise ValueError("Peer dispatch route changed")
        report = runtime.execute(
            request.task, request.plan, coordinate=acquisition.coordinate,
        )
        if (not verify_bootstrap_report(
                report, request.task, request.plan,
            ) or report["output"] != _dispatch_output(request.task)):
            raise ValueError("Admitted peer dispatch report differs from exact request")
        dispatch_reports.append({
            "offer_id": request.offer.offer_id,
            "peer_id": request.peer.peer_id,
            "task_digest": request.task.fingerprint,
            "route_digest": request.plan.fingerprint,
            "report": report,
        })
    # Revalidate the complete available route catalogue *after* dispatch;
    # a revised lease/policy must not silently authorize an old intention.
    if not verify_routed_authenticated_plan(*args, acquisition, runtime):
        raise ValueError("Receiver authority/route changed before measurement")
    for peer in acquisition.pinned_peers:
        client = clients[peer.peer_id]
        _bound_transport(
            client.transport, client.peer_id, peer.peer_id, peer.expected_dns,
        )
    # Native IOS-003 resolves on live authenticated channels. The TDL-005
    # verifier and VGI policy independently reconstruct the read results.
    authenticated = acquire_authenticated_refinement(
        *args, acquisition.mathematical_plan,
        old_native_state=acquisition.old_native_state,
        clients=clients,
        required_sources=tuple(peer.peer_id for peer in acquisition.pinned_peers),
        authority_digest=acquisition.authority_digest,
        scheduler=acquisition.scheduler,
        challenge_nonce=acquisition.challenge_nonce,
    )
    if not verify_routed_authenticated_plan(*args, acquisition, runtime):
        raise ValueError("Receiver authority/route changed before commitment")
    body = {
        "schema": SCHEMA,
        "plan_digest": acquisition.fingerprint,
        "epoch_digest": acquisition.epoch_digest,
        "mathematical_plan_digest": acquisition.mathematical_plan["digest"],
        "declared_total_route_quanta": acquisition.mathematical_plan["total_cost"],
        "dispatch_reports": dispatch_reports,
        "authenticated_record": authenticated,
        "authority_scope": SCOPE,
    }
    result = {**body, "digest": _hash(body)}
    if not verify_routed_authenticated_result(*args, acquisition, runtime, result):
        raise ValueError("Independent route-to-peer and VGI receipt audit refused")
    return result
