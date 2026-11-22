"""TDL-006: admitted SAB endpoint must equal live IOS-003 mTLS identity.

Real CA-verified TLS clients/servers from TDL-005 are composed with real
E-CPU SubstrateObservation/AdaptiveBootstrap route and handler/carrier APIs.
The mock substrate carriers are receiver-local admission; only the IOS
client's native socket constitutes the live TLS measurement channel.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import shutil
from pathlib import Path
import tempfile
import unittest

from mailcpu.substrate_bootstrap import (
    AdaptiveBootstrap, BootstrapPolicy, CarrierBinding, ChannelObservation,
    OperationOffer, RealizerBinding, ResourceObservation, SubstrateObservation,
    digest,
)
from mailcpu.temporal import SchedulerReading
from vli.cost_aware_refinement import ObservationOffer
from vli.problem_formed_interpreter import _hash
from vli.routed_authenticated_refinement import (
    PinnedPeer, plan_routed_authenticated_refinement,
    execute_routed_authenticated_refinement,
    observer_operation, observer_output_kind,
    INPUT_KIND,
)
from vli.routed_authenticated_verify import (
    verify_routed_authenticated_plan, verify_routed_authenticated_result,
)
from test_authenticated_refinement import (
    _grounds, _certificates, LivePeers, AUTH, SCHED, NONCE, PEERS,
)

PINNED = tuple(PinnedPeer(p, p + ".fabric.test") for p in PEERS)
OFFERS = (
    ObservationOffer("direct", 2, 20, "observation-family"),
    ObservationOffer("cheap-left", 6, 1, "observation-family"),
    ObservationOffer("cheap-right", 4, 1, "observation-family"),
)


def observed_runtime(*, revoke=(), lease_end=20, malicious_dispatch=None):
    """Receiver-owned execution routes to each exact pinned peer.

    One peer's dispatch must traverse an intermediate relay and is costlier.
    The returned response is a *dispatch authorization*, not a measured bit.
    """
    def operation_offer(offer, peer):
        return OperationOffer(
            observer_operation(offer.offer_id, peer),
            INPUT_KIND, observer_output_kind(offer.offer_id, peer),
            offer.cost,
        )

    def resource(name, offers=(), expires=20):
        return ResourceObservation(
            name, "receiver-observed-locus", 1, expires,
            "tdl006-receiver-prober", digest({
                "resource": name, "source_revision": 1,
            }), tuple(offers),
        )

    def channel(cid, source, dest, kind):
        return ChannelObservation(
            cid, source, dest, (kind,), 1, 1, 20,
            digest({"channel": cid, "source_revision": 1}), 8192,
        )

    channels = [
        channel("ingress-left", "ingress", "peer-left", INPUT_KIND),
        channel("ingress-relay", "ingress", "relay", INPUT_KIND),
        channel("relay-right", "relay", "peer-right", INPUT_KIND),
    ]
    for offer in OFFERS:
        for peer in PEERS:
            kind = observer_output_kind(offer.offer_id, peer)
            channels.append(
                channel("return-" + offer.offer_id + "-" + peer,
                        peer, "collector", kind),
            )
    inventory = SubstrateObservation((
        resource("ingress"),
        resource("collector"),
        resource("relay"),
        resource("peer-left", (
            operation_offer(o, "peer-left") for o in OFFERS
        )),
        resource("peer-right", (
            operation_offer(o, "peer-right") for o in OFFERS
        ), expires=lease_end),
    ), tuple(channels))

    bound = []
    for offer in OFFERS:
        for peer in PEERS:
            operation = observer_operation(offer.offer_id, peer)

            def perform(request, offer=offer, peer=peer):
                answer = {
                    "request_digest": digest(request),
                    "epoch_digest": request["epoch_digest"],
                    "offer_id": offer.offer_id,
                    "row_decimal": str(offer.row),
                    "peer_id": peer,
                    "expected_dns": peer + ".fabric.test",
                }
                if malicious_dispatch == (offer.offer_id, peer):
                    answer["peer_id"] = "wrong-endpoint"
                return answer

            def independent(previous, resulting, offer=offer, peer=peer):
                return (
                    type(resulting) is dict
                    and resulting == {
                        "request_digest": digest(previous),
                        "epoch_digest": previous["epoch_digest"],
                        "offer_id": offer.offer_id,
                        "row_decimal": str(offer.row),
                        "peer_id": peer,
                        "expected_dns": peer + ".fabric.test",
                    }
                )

            bound.append(RealizerBinding(
                peer, operation, perform,
                digest({"source": "trusted-local-dispatch", "operation": operation}),
                independent,
            ))
    policy = BootstrapPolicy(
        frozenset(
            (p, observer_operation(o.offer_id, p))
            for o in OFFERS for p in PEERS
            if (o.offer_id, p) not in revoke
        ),
        frozenset(ch.channel_id for ch in channels),
        "tdl006-receiver-owned-authority",
    )
    carriers = tuple(CarrierBinding(
        ch.channel_id, lambda frame, tick: frame,
        digest({"carrier": ch.channel_id, "role": "receiver-bound"}),
    ) for ch in channels)
    return AdaptiveBootstrap(inventory, policy, tuple(bound), carriers)


def create_plan(args, runtime, *, state=5, coordinate=3,
                peers=PINNED, nonce=NONCE, offers=OFFERS,
                authority=AUTH, scheduler=SCHED):
    return plan_routed_authenticated_refinement(
        *args, offers, runtime,
        pinned_peers=peers, ingress="ingress", collector="collector",
        old_native_state=args[1].project(state), coordinate=coordinate,
        authority_digest=authority, scheduler=scheduler,
        challenge_nonce=nonce,
    )


@unittest.skipUnless(shutil.which("openssl"), "OpenSSL CLI required for live TLS")
class RoutedMTLSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cert_dir = tempfile.TemporaryDirectory(prefix="tdl006-tls-certs-")
        cls.tls_contexts = _certificates(Path(cls.cert_dir.name))
        cls.args = _grounds()

    @classmethod
    def tearDownClass(cls):
        cls.cert_dir.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tdl006-tls-ledgers-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def live(self, plan, *, state=5, other=None, reads=None):
        return LivePeers(
            root=self.root, contexts=self.tls_contexts,
            plan=plan.mathematical_plan,
            old_z=plan.old_native_state,
            states={"peer-left": state, "peer-right":
                    state if other is None else other},
            reads=reads,
        )

    def test_exact_admitted_peer_pair_matches_live_tls_sources_and_VGI(self):
        for state in (0, 5, 7):
            runtime = observed_runtime()
            plan = create_plan(self.args, runtime, state=state,
                               nonce=f"{state+1:064x}")
            self.assertTrue(verify_routed_authenticated_plan(
                *self.args, plan, runtime,
            ))
            self.assertEqual(
                plan.mathematical_plan["selected_offer_ids"],
                ["cheap-left", "cheap-right"],
            )
            self.assertEqual(plan.mathematical_plan["total_cost"], 14)
            self.assertEqual(len(plan.requests), 4)
            self.assertEqual(
                [(r.offer.offer_id, r.peer.peer_id,
                  r.plan.estimated_quanta) for r in plan.requests],
                [
                    ("cheap-left", "peer-left", 3),
                    ("cheap-left", "peer-right", 4),
                    ("cheap-right", "peer-left", 3),
                    ("cheap-right", "peer-right", 4),
                ],
            )
            with self.live(plan, state=state) as peers:
                result = execute_routed_authenticated_refinement(
                    *self.args, plan, runtime, clients=peers.clients,
                )
            self.assertTrue(verify_routed_authenticated_result(
                *self.args, plan, runtime, result,
            ))
            self.assertEqual(
                int(result["authenticated_record"]["refinement_receipt"]
                    ["new_native_decimal"]),
                self.args[4].project(state),
            )
            self.assertEqual(
                {(x["offer_id"], x["peer_id"])
                 for x in result["authenticated_record"]["native_receipts"]},
                {(r.offer.offer_id, r.peer.peer_id) for r in plan.requests},
            )
            for item in result["dispatch_reports"]:
                self.assertTrue(item["report"]["receiver_oracle_checked_steps"])
                self.assertFalse(item["report"]["physical_independence_attested"])

    def test_swapped_live_tls_peers_cannot_satisfy_admitted_route(self):
        runtime = observed_runtime()
        plan = create_plan(self.args, runtime)
        reads = []
        with self.live(plan, reads=reads) as connected:
            swapped = {
                "peer-left": connected.clients["peer-right"],
                "peer-right": connected.clients["peer-left"],
            }
            with self.assertRaises(ValueError):
                execute_routed_authenticated_refinement(
                    *self.args, plan, runtime, clients=swapped,
                )
        self.assertEqual(reads, [])

    def test_certificate_dns_differs_from_receiver_admitted_endpoint(self):
        runtime = observed_runtime()
        pinned = (
            PINNED[0],
            PinnedPeer("peer-right", "wrong-peer.fabric.test"),
        )
        plan = create_plan(self.args, runtime, peers=pinned)
        reads = []
        with self.live(plan, reads=reads) as connected:
            with self.assertRaises(ValueError):
                execute_routed_authenticated_refinement(
                    *self.args, plan, runtime, clients=connected.clients,
                )
        self.assertEqual(reads, [])

    def test_lease_expiration_refuses_stale_epoch_and_replans(self):
        active = observed_runtime()
        initial = create_plan(self.args, active)
        expired = observed_runtime(lease_end=2)
        self.assertFalse(verify_routed_authenticated_plan(
            *self.args, initial, expired,
        ))
        with self.assertRaises(ValueError):
            execute_routed_authenticated_refinement(
                *self.args, initial, expired, clients={},
            )
        with self.assertRaisesRegex(ValueError, "span"):
            create_plan(self.args, expired, coordinate=3)

    def test_revoked_peer_specific_offer_switches_to_direct_basis(self):
        restricted = observed_runtime(
            revoke=(("cheap-right", "peer-right"),),
        )
        plan = create_plan(self.args, restricted)
        self.assertEqual(
            plan.mathematical_plan["selected_offer_ids"], ["direct"],
        )
        self.assertEqual(plan.mathematical_plan["total_cost"], 45)
        with self.live(plan, state=5) as connected:
            result = execute_routed_authenticated_refinement(
                *self.args, plan, restricted, clients=connected.clients,
            )
        self.assertTrue(verify_routed_authenticated_result(
            *self.args, plan, restricted, result,
        ))
        self.assertEqual(
            int(result["authenticated_record"]["refinement_receipt"]
                ["new_native_decimal"]),
            self.args[4].project(5),
        )

    def test_policy_revision_after_planning_refuses_before_tls_read(self):
        active = observed_runtime()
        plan = create_plan(self.args, active)
        restricted = observed_runtime(
            revoke=(("cheap-left", "peer-right"),),
        )
        reads = []
        with self.live(plan, reads=reads) as connected:
            with self.assertRaises(ValueError):
                execute_routed_authenticated_refinement(
                    *self.args, plan, restricted, clients=connected.clients,
                )
        self.assertEqual(reads, [])

    def test_wrong_receiver_dispatch_oracle_prevents_tls_read(self):
        tampered = observed_runtime(
            malicious_dispatch=("cheap-left", "peer-left"),
        )
        plan = create_plan(self.args, tampered)
        reads = []
        with self.live(plan, reads=reads) as connected:
            with self.assertRaises(Exception):
                execute_routed_authenticated_refinement(
                    *self.args, plan, tampered, clients=connected.clients,
                )
        self.assertEqual(reads, [])

    def test_conflicting_peer_bits_cannot_finalize_processor_state(self):
        runtime = observed_runtime()
        plan = create_plan(self.args, runtime)
        with self.live(plan, state=5, other=1) as connected:
            with self.assertRaises(ValueError):
                execute_routed_authenticated_refinement(
                    *self.args, plan, runtime, clients=connected.clients,
                )

    def test_challenge_reconnect_uses_durable_measurement_not_second_read(self):
        runtime = observed_runtime()
        plan = create_plan(self.args, runtime)
        reads = []
        with self.live(plan, reads=reads) as first:
            initial = execute_routed_authenticated_refinement(
                *self.args, plan, runtime, clients=first.clients,
            )
        self.assertEqual(len(reads), 4)
        with self.live(plan, reads=reads) as recovered:
            second = execute_routed_authenticated_refinement(
                *self.args, plan, runtime, clients=recovered.clients,
            )
        self.assertEqual(len(reads), 4)
        self.assertEqual(
            initial["authenticated_record"]["native_receipts"],
            second["authenticated_record"]["native_receipts"],
        )
        self.assertTrue(verify_routed_authenticated_result(
            *self.args, plan, runtime, second,
        ))

    def test_forged_endpoint_and_quorum_record_rejected_after_rehash(self):
        runtime = observed_runtime()
        plan = create_plan(self.args, runtime)
        with self.live(plan) as connected:
            original = execute_routed_authenticated_refinement(
                *self.args, plan, runtime, clients=connected.clients,
            )
        for case in ("dispatch_peer", "route_cost", "native_peer",
                     "auth_digest", "vgi_evidence", "epoch"):
            forged = copy.deepcopy(original)
            if case == "dispatch_peer":
                forged["dispatch_reports"][0]["peer_id"] = "imposter"
            elif case == "route_cost":
                forged["declared_total_route_quanta"] = 2
            elif case == "native_peer":
                forged["authenticated_record"]["native_receipts"][0][
                    "peer_id"] = "imposter"
                forged["authenticated_record"]["digest"] = _hash({
                    k: v for k, v in forged["authenticated_record"].items()
                    if k != "digest"
                })
            elif case == "auth_digest":
                forged["authenticated_record"]["plan_digest"] = "0" * 64
                forged["authenticated_record"]["digest"] = _hash({
                    k: v for k, v in forged["authenticated_record"].items()
                    if k != "digest"
                })
            elif case == "vgi_evidence":
                forged["authenticated_record"]["vgi_ledger_digest"] = "0" * 64
                forged["authenticated_record"]["digest"] = _hash({
                    k: v for k, v in forged["authenticated_record"].items()
                    if k != "digest"
                })
            else:
                forged["epoch_digest"] = "0" * 64
            forged["digest"] = _hash({
                k: v for k, v in forged.items() if k != "digest"
            })
            with self.subTest(case=case):
                self.assertFalse(verify_routed_authenticated_result(
                    *self.args, plan, runtime, forged,
                ))


if __name__ == "__main__":
    unittest.main()
