"""TDL-005: actual IOS-003 mTLS loopback peers -> VGI strict quorum -> VL-I.

Two independent TLS certificate identities are exercised on live connections.
Servers run in local threads; they are NOT independently administered physical
hosts. All native measurement contracts and durable completion use E-CPU's
existing production APIs, rather than mock transport/forged receipt objects.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest

from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from vli.problem_formed_interpreter import construct, _hash
from vli.cost_aware_refinement import (
    ObservationOffer, plan_cost_aware_observations,
)
from vli.authenticated_refinement import (
    acquire_authenticated_refinement, prepare_peer_measurement,
    AuthenticatedRefinementRefused,
)
from vli.authenticated_refinement_verify import verify_authenticated_refinement_record
from mailcpu.intention_measurement_fabric import (
    HostedMeasurement, MeasurementPeerClient, MeasurementPeerServer,
    MeasurementReceiptLedger,
)
from mailcpu.semantic_translation import canonical_value_digest
from mailcpu.socket_transport import SocketTransport
from mailcpu.temporal import SchedulerReading
from vgi.evidence import TransitionEvidenceLedger

AUTH = canonical_value_digest({"receiver": "tdl005-authority-epoch-1"})
SCHED = SchedulerReading("tdl005-observer-clock", 5)
NONCE = "1" * 64
PEERS = ("peer-left", "peer-right")


def _grounds():
    rows = tuple(1 << i for i in range(4))
    operators = (AffineMap(rows, 0),)
    old = SymbolicSource(4, operators, (1,), "tdl005-same-source-law")
    new = SymbolicSource(4, operators, (1, 2), "tdl005-same-source-law")
    old_q, old_bundle = construct(old)
    new_q, new_bundle = construct(new)
    return (old, old_q, old_bundle, new, new_q, new_bundle)


def _plan(args):
    return plan_cost_aware_observations(*args, (
        ObservationOffer("direct", 2, 10, "peer-direct"),
        ObservationOffer("cheap-left", 6, 1, "peer-left"),
        ObservationOffer("cheap-right", 4, 1, "peer-right"),
    ))


def _certificates(root: Path):
    def run(*args):
        done = subprocess.run(
            ["openssl", *args], cwd=root, text=True, capture_output=True,
            timeout=30,
        )
        if done.returncode:
            raise RuntimeError(done.stderr)
    ca_key, ca_cert = root / "ca.key", root / "ca.pem"
    run("req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(ca_key), "-out", str(ca_cert), "-days", "1",
        "-subj", "/CN=TDL005 Local Test CA",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    certificates = {}
    for name in ("peer-left", "peer-right", "receiver"):
        dns = name + ".fabric.test"
        key, csr, cert, ext = (
            root / f"{name}.{suffix}" for suffix in ("key", "csr", "pem", "ext")
        )
        run("req", "-new", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(csr),
            "-subj", f"/CN={dns}")
        ext.write_text(
            f"subjectAltName=DNS:{dns}\n"
            "extendedKeyUsage=clientAuth,serverAuth\n"
            "basicConstraints=critical,CA:FALSE\n"
            "keyUsage=critical,digitalSignature,keyEncipherment\n",
            encoding="ascii",
        )
        run("x509", "-req", "-in", str(csr), "-CA", str(ca_cert),
            "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(cert),
            "-days", "1", "-extfile", str(ext))
        certificates[name] = (cert, key)

    servers = {}
    for name in PEERS:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(*map(str, certificates[name]))
        context.load_verify_locations(cafile=str(ca_cert))
        context.verify_mode = ssl.CERT_REQUIRED
        servers[name] = context
    client = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH, cafile=str(ca_cert),
    )
    client.minimum_version = ssl.TLSVersion.TLSv1_2
    client.load_cert_chain(*map(str, certificates["receiver"]))
    return servers, client


class LivePeers:
    """One mTLS session per peer, reusing durable files across restarts."""

    def __init__(self, *, root, contexts, plan, old_z, states,
                 reads=None, false_output=None):
        self.root = root
        self.servers, self.client_context = contexts
        self.plan, self.old_z, self.states = plan, old_z, states
        self.reads = reads if reads is not None else []
        self.false_output = false_output
        self.clients = {}
        self.listeners = {}
        self.threads = []
        self.errors = []
        self.stop = threading.Event()

    def __enter__(self):
        try:
            for peer in PEERS:
                listener = SocketTransport.listen("127.0.0.1", 0)
                listener.settimeout(6)
                self.listeners[peer] = listener
                installed = {}
                for offer_id in self.plan["selected_offer_ids"]:
                    spec = prepare_peer_measurement(
                        self.plan, old_native_state=self.old_z,
                        offer_id=offer_id, peer_id=peer,
                        authority_digest=AUTH, scheduler=SCHED,
                        challenge_nonce=NONCE,
                    )
                    row = next(int(x["row_decimal"]) for x in self.plan["offers"]
                               if x["offer_id"] == offer_id)

                    def observe(_request, row=row, peer=peer, offer_id=offer_id):
                        self.reads.append((peer, offer_id))
                        if self.false_output == (peer, offer_id):
                            return 2
                        return (self.states[peer] & row).bit_count() & 1

                    installed[spec.feature_id] = HostedMeasurement(
                        spec.contract, spec.model.digest,
                        lambda: AUTH,
                        lambda req, caller, model=spec.model: (
                            caller == "receiver"
                            and req["model_digest"] == model.digest
                            and req["authority_digest"] == AUTH
                        ),
                        observe, (0, 1),
                    )

                def serve(peer=peer, listener=listener, installed=installed):
                    ledger = MeasurementReceiptLedger(
                        self.root / f"{peer}-durable.sqlite",
                    )
                    server = None
                    try:
                        server = MeasurementPeerServer.accept_mutual_tls(
                            listener, self.servers[peer], peer, "receiver",
                            "receiver.fabric.test", installed, ledger,
                        )
                        while not self.stop.is_set() and not server.transport.closed:
                            server.service_once()
                            time.sleep(.002)
                    except Exception as exc:
                        if not self.stop.is_set():
                            self.errors.append(exc)
                    finally:
                        if server is not None:
                            server.close()
                        ledger.close()

                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                self.threads.append(thread)
                self.clients[peer] = MeasurementPeerClient.connect_mutual_tls(
                    "127.0.0.1", listener.getsockname()[1],
                    self.client_context, "receiver", peer,
                    peer + ".fabric.test", timeout=4.0,
                )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, _typ, _val, _tb):
        self.stop.set()
        for client in self.clients.values():
            client.close()
        for listener in self.listeners.values():
            listener.close()
        for thread in self.threads:
            thread.join(timeout=7)
        if any(thread.is_alive() for thread in self.threads):
            raise AssertionError("Live IOS-003 receiver test left a server running")
        if self.errors:
            raise self.errors[0]


@unittest.skipUnless(shutil.which("openssl"), "OpenSSL CLI required for live mutual-TLS fixture")
class AuthenticatedRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certs = tempfile.TemporaryDirectory(prefix="tdl005-certs-")
        cls.contexts = _certificates(Path(cls.certs.name))
        cls.args = _grounds()
        cls.plan = _plan(cls.args)
        assert cls.plan["selected_offer_ids"] == ["cheap-left", "cheap-right"]

    @classmethod
    def tearDownClass(cls):
        cls.certs.cleanup()

    def setUp(self):
        self.storage = tempfile.TemporaryDirectory(prefix="tdl005-ledger-")
        self.root = Path(self.storage.name)

    def tearDown(self):
        self.storage.cleanup()

    def peers(self, *, state=5, other=None, old_z=None, reads=None, false_output=None):
        if old_z is None:
            old_z = self.args[1].project(state)
        return LivePeers(
            root=self.root, contexts=self.contexts, plan=self.plan,
            old_z=old_z, states={
                "peer-left": state, "peer-right": state if other is None else other,
            }, reads=reads, false_output=false_output,
        )

    def acquire(self, clients, *, state=5, old_z=None, nonce=NONCE):
        if old_z is None:
            old_z = self.args[1].project(state)
        return acquire_authenticated_refinement(
            *self.args, self.plan, old_native_state=old_z,
            clients=clients, required_sources=PEERS,
            authority_digest=AUTH, scheduler=SCHED, challenge_nonce=nonce,
        )

    def test_two_real_mtls_peer_identities_form_strict_vgi_quorum(self):
        for state in (0, 5, 7):
            with self.peers(state=state) as group:
                result = self.acquire(group.clients, state=state)
                self.assertTrue(verify_authenticated_refinement_record(
                    *self.args, self.plan, result,
                ))
                self.assertEqual(
                    int(result["refinement_receipt"]["new_native_decimal"]),
                    self.args[4].project(state),
                )
                self.assertEqual(result["required_sources"], list(PEERS))
                self.assertEqual(len(result["native_receipts"]), 4)
                ledger = TransitionEvidenceLedger.from_dict(result["vgi_ledger"])
                for offer_id in self.plan["selected_offer_ids"]:
                    decision = ledger.decision("refinement", offer_id)
                    self.assertEqual(decision.status, "AUTHORIZED")
                    self.assertEqual(decision.agreeing_sources, PEERS)
                    self.assertEqual(decision.quorum, 2)

    def test_conflicting_authenticated_peer_bits_refuse_processor_state(self):
        with self.peers(state=5, other=1) as group:
            with self.assertRaisesRegex(
                AuthenticatedRefinementRefused, "quorum refused or disagreed"
            ):
                self.acquire(group.clients, state=5)

    def test_stale_observer_context_cannot_migrate_processor(self):
        with self.peers(state=5) as group:
            with self.assertRaises(AuthenticatedRefinementRefused):
                self.acquire(group.clients, state=5, old_z=0)

    def test_receipt_export_cannot_replace_a_live_client(self):
        with self.peers(state=5) as group:
            result = self.acquire(group.clients, state=5)
            forged = {"peer-left": result["native_receipts"], "peer-right": group.clients["peer-right"]}
            with self.assertRaisesRegex(
                AuthenticatedRefinementRefused, "pinned IOS-003"
            ):
                self.acquire(forged, state=5)

    def test_host_output_outside_admitted_bit_domain_refuses(self):
        with self.peers(state=5, false_output=("peer-left", "cheap-left")) as group:
            with self.assertRaises(AuthenticatedRefinementRefused):
                self.acquire(group.clients, state=5)

    def test_reconnect_idempotent_challenge_uses_durable_ios_receipts(self):
        reads = []
        with self.peers(state=5, reads=reads) as first:
            earlier = self.acquire(first.clients, state=5)
        self.assertEqual(len(reads), 4)
        with self.peers(state=5, reads=reads) as second:
            later = self.acquire(second.clients, state=5)
        self.assertEqual(len(reads), 4)
        self.assertEqual(earlier["native_receipts"], later["native_receipts"])
        self.assertTrue(verify_authenticated_refinement_record(
            *self.args, self.plan, later,
        ))

    def test_new_nonce_creates_new_source_measurement_epoch(self):
        reads = []
        with self.peers(state=5, reads=reads) as group:
            first = self.acquire(group.clients, state=5, nonce="2"*64)
            second = self.acquire(group.clients, state=5, nonce="3"*64)
        self.assertEqual(len(reads), 8)
        self.assertNotEqual(first["native_receipts"], second["native_receipts"])
        self.assertEqual(
            first["refinement_receipt"]["new_native_decimal"],
            second["refinement_receipt"]["new_native_decimal"],
        )

    def test_independent_record_rejects_rehashed_peer_or_bit_forgery(self):
        with self.peers(state=5) as group:
            valid = self.acquire(group.clients, state=5)
        for mutation in ("bit", "peer", "challenge", "ledger", "state"):
            forged = copy.deepcopy(valid)
            if mutation == "bit":
                forged["native_receipts"][0]["bit"] ^= 1
            elif mutation == "peer":
                forged["native_receipts"][0]["peer_id"] = "arbitrary-host"
            elif mutation == "challenge":
                forged["native_receipts"][0]["challenge_digest"] = "0" * 64
            elif mutation == "ledger":
                forged["vgi_ledger_digest"] = "0" * 64
            else:
                forged["old_native_decimal"] = "0"
            forged["digest"] = _hash({
                k: v for k, v in forged.items() if k != "digest"
            })
            with self.subTest(mutation=mutation):
                self.assertFalse(verify_authenticated_refinement_record(
                    *self.args, self.plan, forged,
                ))

    def test_peer_identity_is_distinct_from_observation_feature(self):
        spec_a = prepare_peer_measurement(
            self.plan, old_native_state=1, offer_id="cheap-left",
            peer_id="peer-left", authority_digest=AUTH,
            scheduler=SCHED, challenge_nonce=NONCE,
        )
        spec_b = prepare_peer_measurement(
            self.plan, old_native_state=1, offer_id="cheap-left",
            peer_id="peer-right", authority_digest=AUTH,
            scheduler=SCHED, challenge_nonce=NONCE,
        )
        self.assertEqual(spec_a.model.digest, spec_b.model.digest)
        self.assertNotEqual(spec_a.contract.digest, spec_b.contract.digest)
        self.assertNotEqual(spec_a.challenge_digest, spec_b.challenge_digest)
        self.assertEqual(spec_a.feature_id, spec_b.feature_id)


if __name__ == "__main__":
    unittest.main()
