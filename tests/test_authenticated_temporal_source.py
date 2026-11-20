"""TIME-005A source-authenticated process-death and checkpoint recovery controls."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.ground import _canonical, _digest, load_json
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.evolving_environment_language import (
    EvolvingEnvironmentLanguage,certify_language_migration,
    verify_environment_language,verify_language_migration,
)
from vli.authenticated_temporal_source import (
    SIGNED_SCHEMA, _genesis, _extend, _packet_digest,
    append_signed_observation, public_key_hex,
    sign_source_observation, verified_source_journal, _signature_body,
    verify_source_signature, resume_signed_observer,
)

SOURCE="time005:signed-declared-software-sequence"
ENV="time005:continuous-observer-station"


class SignedSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        self.journal=self.path/"source.jsonl"
        self.snapshot=self.path/"checkpoint.json"
        self.key=Ed25519PrivateKey.generate()
        self.public=public_key_hex(self.key)
        self.head=_genesis(SOURCE,self.public)
        self.count=0

    def emit(self,value):
        packet=sign_source_observation(
             self.key,source=SOURCE,sequence=self.count,
             time=f"{self.count}/1",value=f"{value}/1")
        self.head,self.count=append_signed_observation(
             self.journal,packet,expected_head=self.head,
             expected_count=self.count,source=SOURCE,public_key=self.public)
        return packet

    def run_worker(self,host,*,before=None,after=None,expect=0):
        cmd=[sys.executable,"-m","vli.authenticated_temporal_source",
             "--journal",str(self.journal),
             "--checkpoint",str(self.snapshot),
             "--source",SOURCE,"--public-key",self.public,
             "--head",self.head,"--count",str(self.count),
             "--environment",ENV,"--host",host]
        if before is not None:
            cmd+=["--crash-before",str(before)]
        if after is not None:
            cmd+=["--crash-after",str(after)]
        completed=subprocess.run(
            cmd,capture_output=True,text=True,timeout=45)
        self.assertEqual(completed.returncode,expect,
             f"worker failed: {completed.stderr[-3000:]}")
        return json.loads(completed.stdout) if expect==0 else None

    def test_source_signature_checks_real_measured_clocks(self):
        first=self.emit(1)
        second=self.emit(3)
        self.assertEqual(first["schema"],SIGNED_SCHEMA)
        self.assertGreaterEqual(second["captured_monotonic_ns"],
                                first["captured_monotonic_ns"])
        self.assertGreater(first["captured_wall_ns"],0)
        self.assertTrue(verify_source_signature(
            first,public_key=self.public,source=SOURCE))
        packets=verified_source_journal(self.journal,
            expected_head=self.head,expected_count=self.count,
            source=SOURCE,public_key=self.public)
        self.assertEqual(packets,[first,second])

    def test_actual_process_death_before_commit_and_after_commit(self):
        """Distinct real OS PIDs resume source without reapplying saved effects."""
        for x in [1,3,5]:
            self.emit(x)
        self.run_worker("observer-host-A",before=2,expect=71)
        saved_2=load_json(self.snapshot)
        self.assertEqual(len(saved_2["events"]),2)
        b=self.run_worker("observer-host-B")
        self.assertEqual(b["restored_source_count"],2)
        self.assertEqual(b["newly_admitted_in_this_process"],1)
        self.assertEqual(b["source_events_replayed_after_durable_commit"],0)
        self.assertEqual(b["active_environment_family"],"affine")
        before=load_json(self.snapshot)
        prior=TemporalIntelligenceGround.restore(
            before,expected_head=before["head"],
            expected_identity=b["observer_identity"])
        old_lang=EvolvingEnvironmentLanguage(prior).current(expected_head=prior.head)
        self.assertTrue(verify_environment_language(
            before,old_lang,expected_head=before["head"],
            expected_identity=prior.identity))
        for x in [100,101,104,109]:
            self.emit(x)
        self.run_worker("observer-host-B",after=5,expect=72)
        saved_6=load_json(self.snapshot)
        self.assertEqual(len(saved_6["events"]),6)
        c=self.run_worker("observer-host-C")
        self.assertNotEqual(b["process_id"],c["process_id"])
        self.assertEqual(c["restored_source_count"],6)
        self.assertEqual(c["newly_admitted_in_this_process"],1)
        self.assertEqual(c["authenticated_observation_count"],7)
        self.assertEqual(c["source_events_replayed_after_durable_commit"],0)
        self.assertEqual(c["active_environment_family"],"quadratic")
        self.assertEqual(c["unobserved_intervals_between_samples"],6)
        self.assertEqual(len(c["measured_source_inter_sample_gaps_ns"]),6)
        self.assertTrue(all(x>=0 for x in c["measured_source_inter_sample_gaps_ns"]))
        after=load_json(self.snapshot)
        recovered=TemporalIntelligenceGround.restore(
            after,expected_head=after["head"],
            expected_identity=b["observer_identity"])
        new_lang=EvolvingEnvironmentLanguage(recovered).current(
            expected_head=recovered.head)
        self.assertEqual([old_lang["family"],new_lang["family"]],
                         ["affine","quadratic"])
        migration=certify_language_migration(
            before,old_lang,after,new_lang,
            before_head=before["head"],after_head=after["head"],
            identity=recovered.identity)
        self.assertTrue(verify_language_migration(
            before,old_lang,after,new_lang,migration,
            before_head=before["head"],after_head=after["head"],
            identity=recovered.identity))
        self.assertEqual(len({row["source_digest"] for row in recovered.events}),7)
        self.assertEqual([row["host"] for row in recovered.events],
                         ["observer-host-A"]*2+["observer-host-B"]*4+
                         ["observer-host-C"])
        no_op=self.run_worker("observer-host-D")
        self.assertEqual(no_op["newly_admitted_in_this_process"],0)
        self.assertEqual(no_op["result_ground_head"],c["result_ground_head"])

    def test_external_forced_os_process_kill_after_fsync_is_recoverable(self):
        """Parent calls Popen.kill on a real child suspended after fsync."""
        for x in [1,3,5]:
            self.emit(x)
        marker=self.path/"ready-after-commit.json"
        cmd=[sys.executable,"-m","vli.authenticated_temporal_source",
             "--journal",str(self.journal),
             "--checkpoint",str(self.snapshot),
             "--source",SOURCE,"--public-key",self.public,
             "--head",self.head,"--count",str(self.count),
             "--environment",ENV,"--host","killed-external-worker",
             "--pause-after","1","--pause-marker",str(marker)]
        child=subprocess.Popen(cmd,stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+30
            while not marker.exists():
                self.assertIsNone(child.poll(),
                    "Observer unexpectedly terminated before forced kill point")
                if time.monotonic()>deadline:
                    self.fail("Worker did not acknowledge its durable crash point")
                time.sleep(0.025)
            signal=load_json(marker)
            self.assertEqual(signal["committed_index"],1)
            persisted=load_json(self.snapshot)
            self.assertEqual(len(persisted["events"]),2)
            self.assertEqual(signal["ground_head"],persisted["head"])
            child.kill()  # actual parent-initiated OS-process termination
            _,stderr=child.communicate(timeout=15)
            self.assertNotEqual(child.returncode,0,stderr)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=15)
        recovered=self.run_worker("recovered-external-worker")
        self.assertEqual(recovered["restored_source_count"],2)
        self.assertEqual(recovered["newly_admitted_in_this_process"],1)
        self.assertEqual(recovered["authenticated_observation_count"],3)
        self.assertEqual(recovered["source_events_replayed_after_durable_commit"],0)
        self.assertEqual(len(load_json(self.snapshot)["events"]),3)

    def test_hard_crash_before_first_sample_requires_full_recovery(self):
        self.emit(7)
        self.run_worker("A",before=0,expect=71)
        self.assertFalse(self.snapshot.exists())
        b=self.run_worker("B")
        self.assertEqual(b["restored_source_count"],0)
        self.assertEqual(b["newly_admitted_in_this_process"],1)
        self.assertEqual(b["authenticated_observation_count"],1)
        self.assertIsNone(b["active_environment_family"])

    def test_stale_publisher_head_rejected_before_checkpoint_mutation(self):
        self.emit(1)
        old_head=self.head
        self.emit(3)
        with self.assertRaises(ValueError):
            resume_signed_observer(
                journal_path=self.journal,checkpoint_path=self.snapshot,
                source=SOURCE,public_key=self.public,
                expected_source_head=old_head,expected_source_count=2,
                environment=ENV,host="B")
        self.assertFalse(self.snapshot.exists())

    def test_corrupt_source_with_recomputed_chain_is_not_authenticated(self):
        first=self.emit(1)
        other=self.emit(3)
        entries=self.journal.read_text().splitlines()
        changed=deepcopy(other)
        changed["values"]=["999/1"]
        self.journal.write_text(entries[0]+"\n"+
                                _canonical(changed).decode()+"\n")
        forged_head=_extend(_extend(_genesis(SOURCE,self.public),first),
                            changed)
        with self.assertRaisesRegex(ValueError,"signature"):
            verified_source_journal(self.journal,expected_head=forged_head,
                expected_count=2,source=SOURCE,public_key=self.public)

    def test_authoritative_publisher_head_detects_truncation(self):
        self.emit(1)
        self.emit(2)
        self.journal.write_bytes(self.journal.read_bytes().splitlines()[0]+b"\n")
        with self.assertRaises(ValueError):
            verified_source_journal(self.journal,expected_head=self.head,
                expected_count=self.count,source=SOURCE,public_key=self.public)

    def test_sequence_or_replay_and_torn_source_rows_rejected(self):
        self.emit(1)
        self.emit(2)
        lines=self.journal.read_bytes().splitlines()
        for replacement in (lines[1]+b"\n"+lines[0]+b"\n",
                            lines[0]+b"\n"+lines[0]+b"\n",
                            lines[0]+b"\n"+lines[1]):
            with self.subTest(replacement=replacement[:20]):
                self.journal.write_bytes(replacement)
                with self.assertRaises(ValueError):
                    verified_source_journal(self.journal,expected_head=self.head,
                        expected_count=2,source=SOURCE,public_key=self.public)

    def test_equal_source_clock_ticks_are_valid_but_regression_is_refused(self):
        """Finite OS monotonic clock precision must not fabricate chronology."""
        first=self.emit(1)
        raw=sign_source_observation(
            self.key,source=SOURCE,sequence=1,time="1/1",value="2/1")
        same={**raw,"captured_monotonic_ns":first["captured_monotonic_ns"]}
        body={k:v for k,v in same.items() if k!="signature"}
        same["signature"]=self.key.sign(_signature_body(body)).hex()
        self.head,self.count=append_signed_observation(
            self.journal,same,expected_head=self.head,
            expected_count=self.count,source=SOURCE,public_key=self.public)
        verified=verified_source_journal(
            self.journal,expected_head=self.head,
            expected_count=self.count,source=SOURCE,public_key=self.public)
        self.assertEqual(verified[0]["captured_monotonic_ns"],
                         verified[1]["captured_monotonic_ns"])
        worker=self.run_worker("equal-clock")
        self.assertEqual(worker["measured_source_inter_sample_gaps_ns"],[0])
        regressed={**same,"sequence":2,"time":"2/1",
                   "captured_monotonic_ns":first["captured_monotonic_ns"]-1}
        contents={k:v for k,v in regressed.items() if k!="signature"}
        regressed["signature"]=self.key.sign(_signature_body(contents)).hex()
        with self.assertRaisesRegex(ValueError,"strictly advancing"):
            append_signed_observation(
                self.journal,regressed,expected_head=self.head,
                expected_count=self.count,source=SOURCE,public_key=self.public)

    def test_wrong_public_key_or_wrong_source_rejected(self):
        self.emit(1)
        stranger=public_key_hex(Ed25519PrivateKey.generate())
        for key,src in ((stranger,SOURCE),(self.public,"different-source")):
            with self.subTest(source=src),self.assertRaises(ValueError):
                verified_source_journal(self.journal,expected_head=self.head,
                     expected_count=self.count,source=src,public_key=key)

    def test_forged_checkpoint_rehashed_internally_fails_signed_check(self):
        for x in [1,3,5]:
            self.emit(x)
        self.run_worker("host-A")
        snapshot=load_json(self.snapshot)
        replacement=deepcopy(snapshot)
        replacement["events"][1]["values"]=["999/1"]
        # Recompute the internal causal ledger to mimic a successful rehash
        g=TemporalIntelligenceGround(ENV)
        for row in replacement["events"]:
            g.observe(row["time"],row["values"],source_digest=row["source_digest"],
                      host=row["host"],expected_head=g.head)
        fake=g.checkpoint()
        self.snapshot.write_bytes(_canonical(fake)+b"\n")
        with self.assertRaisesRegex(ValueError,"signed source"):
            self.run_recovery_local()

    def test_future_invented_checkpoint_observations_refused(self):
        for x in [1,3,5]:
            self.emit(x)
        self.run_worker("host-A")
        valid=load_json(self.snapshot)
        g=TemporalIntelligenceGround.restore(valid,expected_head=valid["head"],
                                              expected_identity=valid["identity"])
        g.observe("3/1",["7/1"],source_digest="0"*64,host="forged",
                  expected_head=g.head)
        self.snapshot.write_bytes(_canonical(g.checkpoint())+b"\n")
        with self.assertRaisesRegex(ValueError,"outside publisher"):
            self.run_recovery_local()

    def test_no_fake_source_signatures_and_no_physical_uptime_claim(self):
        self.emit(1)
        event=verified_source_journal(
            self.journal,expected_head=self.head,expected_count=self.count,
            source=SOURCE,public_key=self.public)[0]
        self.assertNotIn("private_key",event)
        result=self.run_worker("observer-host")
        self.assertEqual(result["physical_continuity_across_crash"],
                         "NOT_OBSERVED")
        self.assertEqual(result["physical_source_value_truth"],
                         "NOT_PROVEN_BY_SIGNATURE")
        self.assertEqual(result["external_quorum_fencing"],"NOT_IMPLEMENTED")
        self.assertEqual(result["authenticated_observation_count"],1)

    def run_recovery_local(self):
        return resume_signed_observer(
            journal_path=self.journal,checkpoint_path=self.snapshot,
            source=SOURCE,public_key=self.public,
            expected_source_head=self.head,expected_source_count=self.count,
            environment=ENV,host="B")


if __name__ == "__main__":
    unittest.main()
