"""TIME-009: real E-CPU IPS-008 receiver and source-pinned rational Ground join.

The source and E-CPU native stage are independent machines. Matching one
observed integer is an evidence join, NOT native rational law execution.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.authenticated_temporal_source import (
    _genesis, public_key_hex, sign_source_observation,
    append_signed_observation, resume_signed_observer,
)
from vli.discovered_state_space import (
    SCHEMA, construct_discovered_space, verify_discovered_space,
)
from vli.temporal_native_effect_bridge import apply_observed_native_effect
from vli.ground import load_json, _digest

ECPU_TESTS = os.environ.get("VLI_ECPU_TEST_ROOT")
if ECPU_TESTS:
    sys.path.insert(0, ECPU_TESTS)


def _sequence(seed, coefficients, n):
    rows = list(seed)
    while len(rows) < n:
        rows.append(sum(coefficients[j]*rows[-j-1]
                        for j in range(len(coefficients))))
    return rows


@unittest.skipUnless(ECPU_TESTS and Path(ECPU_TESTS).is_dir(),
                     "requires checkout of pinned E-CPU native test fixtures")
class TemporalNativeEffectTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ecpu = self.root / "native"
        self.ecpu.mkdir()
        self.key = Ed25519PrivateKey.generate()
        self.public = public_key_hex(self.key)
        self.source = "time009-publisher"
        self.head = _genesis(self.source, self.public)
        self.count = 0
        self.journal = self.root / "signed.jsonl"
        self.checkpoint = self.root / "observer.json"

    def source_ground(self, new_values, *, previous_space=None):
        for v in new_values:
            packet = sign_source_observation(
                self.key, source=self.source, sequence=self.count,
                time=f"{self.count}/1", value=f"{int(v)}/1")
            self.head, self.count = append_signed_observation(
                self.journal, packet, expected_head=self.head,
                expected_count=self.count, source=self.source,
                public_key=self.public)
        run = resume_signed_observer(
            journal_path=self.journal, checkpoint_path=self.checkpoint,
            source=self.source, public_key=self.public,
            expected_source_head=self.head,
            expected_source_count=self.count,
            environment="time009-signature-and-revision",
            host="vl-i-source-observer")
        snapshot = load_json(self.checkpoint)
        contract = dict(
            journal_path=self.journal, source_id=self.source,
            public_key=self.public,
            expected_source_head=self.head,
            expected_source_count=self.count,
            expected_ground_head=run["result_ground_head"],
            expected_ground_identity=run["observer_identity"])
        start = 0 if previous_space is None else previous_space["signed_source_count"]
        space = construct_discovered_space(
            snapshot, **contract, training_count=8, validation_count=2,
            test_count=2, epoch_start=start, previous_space=previous_space)
        self.assertTrue(verify_discovered_space(
            snapshot, space, previous_space=previous_space, **contract))
        return snapshot, space, contract

    def bind(self, snapshot, space, contract, lineage, head, receiver, *,
             time, previous_space=None, expected_space=None,
             expected_root=None):
        from test_effect_receiver_service import RESOURCE
        return apply_observed_native_effect(
            snapshot, space, source_contract=contract,
            expected_space_digest=(space["intention_space_digest"] if
                                   expected_space is None else expected_space),
            expected_native_root=(head.manifest.digest if
                                  expected_root is None else expected_root),
            intended_time=time, native_lineage=lineage, native_head=head,
            native_receiver=receiver, native_resource=RESOURCE,
            previous_space=previous_space)

    def test_native_local_receiver_never_uses_a_source_hash_as_authority(self):
        from test_effect_receiver_service import _published_first, RESOURCE
        from mailcpu.fenced_effect_receiver import FencedGroundEffectReceiver

        first12 = list(range(14, 26))
        snap, space, contract = self.source_ground(first12)
        self.assertEqual(space["dimension"],2)
        lineage, first, _ = _published_first(self.ecpu)
        receiver = FencedGroundEffectReceiver(
            self.ecpu / "receiver.sqlite", resource=RESOURCE,
            plan_digest=lineage.plan.digest)
        receiver.install(lineage, first)
        first_result = self.bind(
            snap, space, contract, lineage, first, receiver, time="2/1")
        self.assertEqual(first_result["source_ground_observed_rational"],"16/1")
        self.assertEqual(first_result["native_effect_receipt"]["value"],16)
        self.assertTrue(first_result["native_receiver_is_only_effect_authority"])
        self.assertFalse(first_result["mathematical_law_native_execution_established"])

        # Native receiver deduplication is real; a second call does not
        # execute an effect again, and the VLI receipt is deterministic.
        self.assertEqual(self.bind(
            snap,space,contract,lineage,first,receiver,time="2/1"),
            first_result)
        self.assertEqual(receiver.snapshot()["accepted_effects"],1)

        with self.assertRaises(ValueError):
            self.bind(snap,space,contract,lineage,first,receiver,
                      time="2/1", expected_space="0"*64)
        with self.assertRaises(ValueError):
            self.bind(snap,space,contract,lineage,first,receiver,
                      time="2/1", expected_root="0"*64)
        with self.assertRaises(ValueError):
            self.bind(snap,space,contract,lineage,first,receiver,
                      time="20/1")
        # Exact authentic signed observation 17 is not equal to original
        # E-CPU stage 16. The native stage must not be misdescribed as
        # having computed an arbitrary source law.
        with self.assertRaises(ValueError):
            self.bind(snap,space,contract,lineage,first,receiver,time="3/1")

        falsified = deepcopy(space)
        falsified["coefficients"][0] = "3/1"
        falsified["intention_space_digest"] = _digest(
            SCHEMA, {k:v for k,v in falsified.items()
                     if k!="intention_space_digest"})
        with self.assertRaises(ValueError):
            self.bind(snap, falsified, contract,lineage,first,receiver,
                      time="2/1",expected_space=falsified["intention_space_digest"])
        self.assertEqual(receiver.snapshot()["accepted_effects"],1)

    @unittest.skipUnless(shutil.which("openssl"), "real E-CPU mTLS requires OpenSSL CLI")
    def test_signed_regime_change_and_real_mtls_receiver_hard_kill_rehome(self):
        from test_effect_receiver_service import (
            _published_first, _start, _client, _kill,
        )
        from test_fenced_generated_ground import _host, _lineage, _restart
        from mailcpu.fenced_effect_receiver import EffectFencingRefused

        old_snapshot, old, old_contract = self.source_ground(
            list(range(14,26)))
        self.assertEqual(old["dimension"],2)
        # Preserve the immutable source prefix independently of the live
        # append-only publisher journal which will later grow.
        self.old_journal = self.root / "old-signed.jsonl"
        shutil.copyfile(self.journal, self.old_journal)
        old_contract = {**old_contract,"journal_path":self.old_journal}

        lineage, first, envelope = _published_first(self.ecpu)
        certs = __import__("test_recovery_lineage_service")._make_certificates(self.ecpu)
        receiver_process, info = _start(self.ecpu, certs)
        recovered = None
        try:
            client = _client(info,certs)
            client.activate(first.manifest.digest)
            initial = self.bind(old_snapshot,old,old_contract,
                                lineage,first,client,time="2/1")
            self.assertEqual(initial["native_effect_receipt"]["value"],16)
            replacement = _host(self.ecpu,"time009-replacement")
            barrier = client.prepare(first.manifest.digest,replacement.locus)
            self.assertEqual(barrier["source_root"],first.manifest.digest)
            with self.assertRaises(EffectFencingRefused):
                self.bind(old_snapshot,old,old_contract,
                          lineage,first,client,time="2/1")

            _kill(receiver_process)
            recovered, info2 = _start(self.ecpu,certs)
            client2 = _client(info2,certs)
            self.assertEqual(client2.snapshot()["pending_handoff_digest"],
                             barrier["barrier_digest"])
            with self.assertRaises(EffectFencingRefused):
                self.bind(old_snapshot,old,old_contract,
                          lineage,first,client2,time="2/1")

            # A new environmental regime supplies an order-3 rather than
            # order-2 processor. Its exact signed source and old source
            # witnesses remain linked; no old future claim is grandfathered.
            new_values = _sequence([14,15,16],[1,0,1],12)
            new_snapshot, revised, new_contract = self.source_ground(
                new_values, previous_space=old)
            self.assertEqual(revised["dimension"],3)
            self.assertEqual(revised["epoch_start"],12)
            self.assertEqual(revised["previous_space_digest"],
                             old["intention_space_digest"])
            self.assertNotEqual(revised["intention_space_digest"],
                                old["intention_space_digest"])
            # Even the new, correct value 16 cannot reach the frozen old root.
            with self.assertRaises(EffectFencingRefused):
                self.bind(new_snapshot,revised,new_contract,
                          lineage,first,client2,time="14/1",previous_space=old)

            continued = _lineage(self.ecpu)
            self.assertEqual(continued.restore_latest().manifest.digest,
                             first.manifest.digest)
            assigned = continued.assign(first,replacement)
            client2.activate(assigned.manifest.digest)
            with self.assertRaises(ValueError):
                self.bind(old_snapshot,old,old_contract,
                          continued,first,client2,time="2/1")
            new_os,new_pid = _restart(envelope)
            latest = continued.publish(assigned, continued.compute(
                assigned,replacement,new_os,new_pid))
            client2.activate(latest.manifest.digest)
            result = self.bind(new_snapshot,revised,new_contract,
                               continued,latest,client2,time="14/1",
                               previous_space=old)
            self.assertEqual(result["source_ground_observed_rational"],"16/1")
            self.assertEqual(result["native_epoch"],barrier["target_epoch"])
            self.assertNotEqual(result["trusted_source_ground_digest"],
                                initial["trusted_source_ground_digest"])
            self.assertNotEqual(result["trusted_native_published_root"],
                                initial["trusted_native_published_root"])
            self.assertEqual(client2.snapshot()["accepted_effects"],2)

            # Wrong live TLS owner identity must fail, despite a genuine
            # mathematical Ground and correct native observation.
            wrong = _client(info2,certs,certificate="C")
            with self.assertRaises(Exception):
                self.bind(new_snapshot,revised,new_contract,
                          continued,latest,wrong,time="14/1",
                          previous_space=old)
            self.assertEqual(client2.snapshot()["accepted_effects"],2)
        finally:
            _kill(receiver_process)
            if recovered is not None:
                _kill(recovered)


if __name__=="__main__":
    unittest.main()
