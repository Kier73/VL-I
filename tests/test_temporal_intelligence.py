"""TIME-001: continuity, drift, source provenance and independent-refusal tests."""
import copy
from hashlib import sha256
import unittest

from vli.ground import _digest
from vli.temporal_intelligence import (
    NAVIGATION, PROCESSOR, SNAPSHOT, TemporalIntelligenceGround,
    verify_temporal_navigation, verify_temporal_processor,
    verify_temporal_snapshot,
)


def observe(g, t, x, host="host-A"):
    values = [x] if isinstance(x, str) else x
    raw = f"{g.identity}|{t}|{','.join(values)}".encode()
    return g.observe(t, values, source_digest=sha256(raw).hexdigest(),
                     host=host, expected_head=g.head)


def stable(n=5):
    g = TemporalIntelligenceGround("sensor:finite-laboratory")
    for i in range(n):
        observe(g, f"{i}/1", f"{2*i+1}/1")
    return g


class TemporalGroundTests(unittest.TestCase):
    def test_environment_constructs_executable_law(self):
        g = stable()
        self.assertEqual(g.epochs[0]["status"], "QUALIFIED_SAMPLES_ONLY")
        p = g.compile_epoch(0, expected_head=g.head)
        self.assertEqual(p["rate"], ["2/1"])
        self.assertEqual(p["origin"], ["1/1"])
        self.assertTrue(verify_temporal_processor(g.checkpoint(),p,
                        expected_head=g.head,expected_identity=g.identity))
        n = g.navigate("5/2",epoch_index=0,expected_head=g.head)
        self.assertEqual(n["values"], ["6/1"])
        self.assertEqual(n["status"], "UNVERIFIED_INTERPOLATION")
        self.assertEqual(n["expanded_trajectory_steps"], 0)
        self.assertTrue(verify_temporal_navigation(g.checkpoint(),n,
                        expected_head=g.head,expected_identity=g.identity))

    def test_recorded_coordinates_are_not_fabricated(self):
        g = stable()
        n = g.navigate("3/1",epoch_index=0,expected_head=g.head)
        self.assertEqual(n["status"], "EXACT_RECORDED_OBSERVATION")
        self.assertTrue(n["source_observation_at_coordinate"])
        self.assertFalse(g.navigate("5/2",epoch_index=0,
                         expected_head=g.head)["source_observation_at_coordinate"])

    def test_changing_environment_creates_new_interpreter_epoch(self):
        g = stable()
        original = g.compile_epoch(0,expected_head=g.head)
        observe(g,"5/1","30/1")
        self.assertEqual(len(g.epochs), 2)
        self.assertEqual(g.epochs[1]["uncertain_change_interval"],["4/1","5/1"])
        with self.assertRaises(ValueError):
            g.compile_epoch(1,expected_head=g.head)
        observe(g,"6/1","31/1")
        with self.assertRaises(ValueError):
            g.compile_epoch(1,expected_head=g.head)
        observe(g,"7/1","32/1")
        self.assertEqual([x["status"] for x in g.epochs],
                         ["QUALIFIED_SAMPLES_ONLY","QUALIFIED_SAMPLES_ONLY"])
        new = g.compile_epoch(1,expected_head=g.head)
        self.assertEqual(new["rate"],["1/1"])
        self.assertNotEqual(original["processor_digest"],new["processor_digest"])
        # Historical model can be rederived under the newest lineage head.
        old = g.compile_epoch(0,expected_head=g.head)
        self.assertEqual(old["rate"],["2/1"])
        self.assertEqual(g.navigate("5/2",epoch_index=0,expected_head=g.head)["values"],
                         ["6/1"])
        self.assertEqual(g.navigate("11/2",epoch_index=1,expected_head=g.head)["values"],
                         ["61/2"])
        with self.assertRaises(ValueError):
            g.navigate("9/2",epoch_index=0,expected_head=g.head)
        with self.assertRaises(ValueError):
            g.navigate("9/2",epoch_index=1,expected_head=g.head)

    def test_continuity_survives_rehome_without_replaying_source(self):
        g=stable()
        pinned_head,pinned_id=g.head,g.identity
        snap=g.checkpoint()
        recovered=TemporalIntelligenceGround.restore(snap,
                        expected_head=pinned_head,expected_identity=pinned_id)
        self.assertEqual(recovered.checkpoint(),snap)
        observe(recovered,"5/1","11/1",host="different-host-B")
        self.assertEqual(recovered.identity,pinned_id)
        self.assertNotEqual(recovered.head,pinned_head)
        self.assertEqual(len(recovered.epochs),1)
        self.assertEqual(recovered.events[-1]["host"],"different-host-B")
        self.assertTrue(verify_temporal_snapshot(recovered.checkpoint(),
                        expected_head=recovered.head,expected_identity=pinned_id))

    def test_refuses_session_local_checkpoint_as_authority(self):
        g=stable()
        snap=g.checkpoint()
        with self.assertRaises(ValueError):
            TemporalIntelligenceGround.restore(snap,expected_head="0"*64,
                                               expected_identity=g.identity)
        with self.assertRaises(ValueError):
            TemporalIntelligenceGround.restore(snap,expected_head=g.head,
                                               expected_identity="1"*64)
        with self.assertRaises(ValueError):
            observe_wrong_head(g)

    def test_source_evidence_is_not_external_authentication(self):
        g=stable(2)
        with self.assertRaises(ValueError):
            g.observe("2/1",["5/1"],source_digest="source-self-report",
                      host="host-A",expected_head=g.head)
        with self.assertRaises(ValueError):
            g.observe("2/1",["5/1"],source_digest="e"*64,
                      host="",expected_head=g.head)

    def test_uncertainty_and_projection_refusal(self):
        g=stable()
        with self.assertRaises(ValueError):
            g.navigate("10/1",epoch_index=0,expected_head=g.head)
        n=g.navigate("10/1",epoch_index=0,expected_head=g.head,
                     allow_projection=True)
        self.assertEqual(n["values"],["21/1"])
        self.assertEqual(n["status"],"UNVERIFIED_FUTURE_PROJECTION")
        self.assertTrue(verify_temporal_navigation(g.checkpoint(),n,
                        expected_head=g.head,expected_identity=g.identity))

    def test_incomplete_epoch_cannot_be_promoted(self):
        g=TemporalIntelligenceGround("unknown-environment")
        observe(g,"0/1","1/1")
        observe(g,"1/1","3/1")
        with self.assertRaises(ValueError):
            g.compile_epoch(0,expected_head=g.head)
        observe(g,"2/1","100/1")
        self.assertEqual(len(g.epochs),2)
        self.assertEqual(g.epochs[0]["status"],"CANDIDATE")

    def test_exact_multidimensional_fractional_time(self):
        g=TemporalIntelligenceGround("vector-field",dimensions=2)
        observe(g,"0/1",["0/1","1/1"])
        observe(g,"1/2",["1/1","0/1"])
        observe(g,"1/1",["2/1","-1/1"])
        self.assertEqual(g.compile_epoch(0,expected_head=g.head)["rate"],
                         ["2/1","-2/1"])
        n=g.navigate("3/4",epoch_index=0,expected_head=g.head)
        self.assertEqual(n["values"],["3/2","-1/2"])

    def test_noncanonical_time_and_bounded_domain_refused(self):
        g=TemporalIntelligenceGround("environment")
        for invalid in ("0.5","1/2/3","2/4","01/1","-0/1","1/0","1/9999999999999"):
            with self.subTest(time=invalid), self.assertRaises(ValueError):
                observe(g,invalid,"1/1")
        observe(g,"1/1","1/1")
        with self.assertRaises(ValueError):
            observe(g,"1/1","2/1")
        with self.assertRaises(ValueError):
            observe(g,"0/1","2/1")

    def test_tampered_snapshot_rejected_even_after_digest_edit(self):
        g=stable()
        s=g.checkpoint()
        forged=copy.deepcopy(s)
        forged["events"][2]["values"]=["999/1"]
        forged["events"][2]["digest"]=_digest(forged["events"][2]["schema"],{
            k:v for k,v in forged["events"][2].items() if k!="digest"})
        forged["checkpoint_digest"]=_digest(SNAPSHOT,{
            k:v for k,v in forged.items() if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            verify_temporal_snapshot(forged,expected_head=g.head,
                                     expected_identity=g.identity)
        bad=copy.deepcopy(s)
        bad["events"][3]["previous"]="a"*64
        bad["checkpoint_digest"]=_digest(SNAPSHOT,{
            k:v for k,v in bad.items() if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            verify_temporal_snapshot(bad,expected_head=g.head,
                                     expected_identity=g.identity)

    def test_generated_semantics_checked_not_just_hashes(self):
        g=stable()
        snap=g.checkpoint()
        p=g.compile_epoch(0,expected_head=g.head)
        forged=copy.deepcopy(p)
        forged["rate"]=["3/1"]
        forged["processor_digest"]=_digest(PROCESSOR,{
            k:v for k,v in forged.items() if k!="processor_digest"})
        with self.assertRaises(ValueError):
            verify_temporal_processor(snap,forged,expected_head=g.head,
                                      expected_identity=g.identity)
        n=g.navigate("3/2",epoch_index=0,expected_head=g.head)
        changed=copy.deepcopy(n)
        changed["values"]=["999/1"]
        changed["navigation_digest"]=_digest(NAVIGATION,{
            k:v for k,v in changed.items() if k!="navigation_digest"})
        with self.assertRaises(ValueError):
            verify_temporal_navigation(snap,changed,expected_head=g.head,
                                      expected_identity=g.identity)

    def test_monotone_head_and_bounded_observed_history(self):
        g=stable()
        original=g.head
        with self.assertRaises(ValueError):
            g.observe("5/1",["11/1"],source_digest="b"*64,
                      host="host-A",expected_head="0"*64)
        self.assertEqual(g.head,original)
        self.assertEqual(len(g.events),5)


def observe_wrong_head(g):
    return g.observe("5/1",["11/1"],source_digest="a"*64,
                     host="host-A",expected_head="0"*64)


if __name__ == "__main__":
    unittest.main()
