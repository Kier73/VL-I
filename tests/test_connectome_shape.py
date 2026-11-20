"""SHAPE-001: real original-source connectome observation with IFS quotient.

The original OpenWorm edge list is Git-pinned and exact-source checked by
IFS. Its morphology is not reinterpreted as a trained biological network.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from vli.connectome_shape import ConnectomeObserverShape
from observer_quotient import QuotientRefused, verify, micro_observation

ORIGINAL = (Path(__file__).resolve().parents[1] / "upstream/ifs" /
            "experiments/real_connectome_ground/data/herm_full_edgelist.csv")
FIXTURE = b"""Source,Target,Weight,Type
AWCL,ASER,1,chemical
AWCL,AWCR,11,chemical
AWCL,AIYL,22,chemical
AWCL,AIBL,12,chemical
ASER,AIYL,9,chemical
ASER,AIBR,32,chemical
ASIL,AIYL,3,chemical
ASIL,AWCL,3,chemical
AWCR,AIYL,12,chemical
AFDL,AIYL,25,chemical
"""


class TestConnectomeShapedObservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ground = ConnectomeObserverShape(
            FIXTURE, groups=8, copies=16, seed=73, original_source=False)

    def test_qualifies_observer_relative_exact_quotient(self):
        g = self.ground
        self.assertEqual(g.synthetic.n, 128)
        self.assertLess(g.dimension,g.synthetic.n)
        self.assertEqual(g.manifest["source_synapse_records"],10)
        self.assertFalse(g.manifest["original_openworm_source"])
        self.assertEqual(g.manifest["expanded_directed_relations"],1024)
        self.assertTrue(g.verify())
        self.assertEqual(len(g.audited[0]),128)
        self.assertEqual(len(g.audited[1]),g.dimension)

    def test_locates_position_before_any_quotient_execution(self):
        g=self.ground
        p=g.position(source_micro=17,logical_tick=6,observer="test",
                     prompt="Observe the future dynamics",views=("text","tokens"))
        self.assertTrue(p["coordinate"].startswith("state://vli/"))
        self.assertEqual(len(p["coordinate"]),76)
        self.assertNotIn("value",p)
        self.assertNotIn("source_microstate_vectors",p)
        out=g.observe(p)
        self.assertEqual(out["coordinate"],p["coordinate"])
        self.assertEqual(out["navigation"]["method"],"exact_quotient_sparse_steps")
        self.assertTrue(out["navigation"]["microstate_oracle_equal"])
        self.assertEqual(out["navigation"]["source_microstate_vectors_realized_by_quotient"],0)
        self.assertEqual(len(out["views"]),2)
        self.assertIn("connectome-shaped Ground",out["views"][0]["content"])
        self.assertEqual(bytes(out["views"][1]["content"]).decode("utf-8"),
                         out["views"][0]["content"])

    def test_all_short_horizon_source_vertices_and_ticks(self):
        g=self.ground
        for source_micro,tick in ((0,0),(1,1),(17,3),(63,7),(127,9)):
            with self.subTest(source_micro=source_micro,tick=tick):
                p=g.position(source_micro=source_micro,logical_tick=tick,
                             observer="source-observer",prompt="Inspect")
                result=g.observe(p)
                self.assertEqual(result["value"],
                                 micro_observation(g.synthetic,g.audited[0],
                                                   source_micro,tick))

    def test_intention_and_observer_change_opaque_position(self):
        g=self.ground
        base=g.position(source_micro=2,logical_tick=5,observer="observer1",
                        prompt="Observe")
        changed=[
            g.position(source_micro=2,logical_tick=6,observer="observer1",
                       prompt="Observe"),
            g.position(source_micro=3,logical_tick=5,observer="observer1",
                       prompt="Observe"),
            g.position(source_micro=2,logical_tick=5,observer="observer2",
                       prompt="Observe"),
            g.position(source_micro=2,logical_tick=5,observer="observer1",
                       prompt="Describe"),
        ]
        for p in changed:
            self.assertNotEqual(p["coordinate"],base["coordinate"])
            self.assertEqual(len(p["coordinate"]),len(base["coordinate"]))

    def test_far_observer_navigation_does_not_realize_microstate_trajectories(self):
        g=self.ground
        far=2**40+73
        p=g.position(source_micro=3,logical_tick=far,observer="future",
                     prompt="Observe at a far logical time",views=("page",))
        out=g.observe(p)
        self.assertEqual(out["coordinate"],p["coordinate"])
        self.assertEqual(out["navigation"]["method"],
                         "IFS_verified_distant_observer_quotient")
        self.assertEqual(out["navigation"]["source_microstate_vectors_realized_by_quotient"],0)
        self.assertGreater(out["navigation"]["polynomial_products"],0)
        self.assertEqual(out["views"][0]["kind"],"page")
        self.assertEqual(out["navigation"]["readout_mod_p"],out["value"])

    def test_forged_partition_and_metadata_fail_even_if_digests_are_changed(self):
        from temporal_ground import hash_object
        g=self.ground
        forged=copy.deepcopy(g.certificate)
        forged["quotient_columns"][0][0][1] += 1
        forged["digest"]=hash_object({k:v for k,v in forged.items() if k!="digest"})
        with self.assertRaises(QuotientRefused):
            g.verify(certificate=forged)
        bad=copy.deepcopy(g.certificate)
        bad["expanded"]["seed"] += 1
        bad["digest"]=hash_object({k:v for k,v in bad.items() if k!="digest"})
        with self.assertRaises(QuotientRefused):
            g.verify(certificate=bad)

    def test_changed_stored_proof_cannot_be_reused_as_a_valid_navigable_shape(self):
        from temporal_ground import hash_object
        g=ConnectomeObserverShape(
            FIXTURE,groups=8,copies=16,seed=73,original_source=False)
        position=g.position(source_micro=2,logical_tick=3,
                            observer="proof-observer",prompt="Observe")
        g.certificate["quotient_columns"][0][0][1] += 1
        # Even replacing the in-memory head cannot change an already
        # admitted, externally addressable model-Ground identity.
        g.certificate["digest"]=hash_object({
            k:v for k,v in g.certificate.items() if k!="digest"})
        with self.assertRaises(ValueError):
            g.observe(position)

    def test_in_memory_partition_not_caller_mutable(self):
        g=self.ground
        self.assertIsInstance(g.audited[0],tuple)
        with self.assertRaises(TypeError):
            g.audited[0][0] = 999

    def test_position_tamper_and_unadmitted_observer_refuse(self):
        g=self.ground
        pos=g.position(source_micro=0,logical_tick=2,observer="test",prompt="Observe")
        changed=copy.deepcopy(pos)
        changed["logical_tick"]=3
        with self.assertRaises(ValueError):
            g.observe(changed)
        with self.assertRaises(ValueError):
            g.position(source_micro=-1,logical_tick=2,observer="test",prompt="Observe")
        with self.assertRaises(ValueError):
            g.position(source_micro=0,logical_tick=2,observer="test",prompt="Observe",
                       views=("tool_call",))

    def test_changed_description_changes_shape_ground(self):
        changed=FIXTURE.replace(b"AWCL,AWCR,11,chemical",b"AWCL,AWCR,12,chemical")
        alt=ConnectomeObserverShape(changed,groups=8,copies=16,original_source=False)
        self.assertNotEqual(alt.manifest["ground_digest"],self.ground.manifest["ground_digest"])
        p1=self.ground.position(source_micro=1,logical_tick=3,observer="test",prompt="Observe")
        p2=alt.position(source_micro=1,logical_tick=3,observer="test",prompt="Observe")
        self.assertNotEqual(p1["coordinate"],p2["coordinate"])


class TestPinnedOriginalTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ORIGINAL.is_file():
            raise AssertionError("Pinned original source missing; checkout IFS as CI workflow does")
        cls.source=ORIGINAL.read_bytes()
        cls.ground=ConnectomeObserverShape(cls.source,groups=8,copies=16,
                                           seed=73,original_source=True)

    def test_real_source_admission_and_synthetic_scope(self):
        g=self.ground
        self.assertEqual(g.manifest["original_source_git_blob"],
                         "89ffc2a2d9e1934ddc462d51032acead5e4cfc91")
        self.assertEqual(g.manifest["source_label_count"],448)
        self.assertEqual(g.manifest["source_synapse_records"],7379)
        self.assertTrue(g.manifest["original_openworm_source"])
        self.assertLess(g.dimension,g.synthetic.n)
        self.assertTrue(g.verify())

    def test_real_source_observer_proof_and_render(self):
        g=self.ground
        p=g.position(source_micro=0,logical_tick=7,observer="real-source",
                     prompt="Observe the source-informed synthetic Ground")
        out=g.observe(p)
        self.assertTrue(out["navigation"]["microstate_oracle_equal"])
        self.assertEqual(out["coordinate"],p["coordinate"])
        self.assertEqual([x["kind"] for x in out["views"]],["text","page","tokens"])
        self.assertEqual(out["navigation"]["source_microstate_vectors_realized_by_quotient"],0)

    def test_real_source_modification_refused(self):
        damaged=self.source.replace(b"I1L , I2L           ,10,chemical",
                                    b"I1L , I2L           ,11,chemical",1)
        self.assertNotEqual(damaged,self.source)
        with self.assertRaises(ValueError):
            ConnectomeObserverShape(damaged,groups=8,copies=16,original_source=True)


if __name__=="__main__":
    unittest.main()
