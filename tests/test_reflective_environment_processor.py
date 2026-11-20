"""TIME-004: prove and falsify the real NCR-002 bridge under pinned sources."""
from copy import deepcopy
from hashlib import sha256
import unittest

from vli.ground import _digest
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.evolving_environment_language import (
    EvolvingEnvironmentLanguage,certify_language_migration,
)
from vli.reflective_environment_processor import (
    BRIDGE, SUCCESSION, construct_reflective_environment_processor,
    verify_reflective_environment_processor,
    certify_reflective_interpreter_succession,
    verify_reflective_interpreter_succession,
)


def feed(g,values,*,offset=0,host="A"):
    for n,x in enumerate(values,offset):
        val=f"{x}/1"
        g.observe(f"{n}/1",[val],
                  source_digest=sha256(f"recorded:{n}:{x}".encode()).hexdigest(),
                  host=host,expected_head=g.head)


def source(g,coordinate=0):
    snapshot=g.checkpoint()
    capsule=EvolvingEnvironmentLanguage(g).current(expected_head=g.head)
    assert capsule
    bridge=construct_reflective_environment_processor(
        snapshot,capsule,expected_head=g.head,
        expected_identity=g.identity,representation_coordinate=coordinate)
    return snapshot,capsule,bridge


class ConstructedEnvironmentInterpreterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world=TemporalIntelligenceGround("verified:reflective-source")
        feed(cls.world,[1,3,5])
        cls.snapshot,cls.capsule,cls.bridge=source(cls.world)

    def test_environment_source_generates_real_verified_self_interpreter(self):
        b=self.bridge
        self.assertEqual(b["source_observation_count"],3)
        self.assertEqual(b["qualified_family"],"affine")
        self.assertEqual(b["result_source_index"],2)
        self.assertEqual(b["result_recorded_value"],"5/1")
        self.assertEqual(b["source_observation_claim"],
                         "EXACT_RECORDED_SOURCE_COORDINATE")
        self.assertEqual(b["environment_dynamics_executed"],"NOT_CLAIMED")
        self.assertFalse(b["physical_effect_performed"])
        self.assertEqual(b["primitive_seed"],["at","cons","eq","rest"])
        self.assertEqual([r["reflection_depth"] for r in b["execution_receipts"]],
                         [0,1,2])
        self.assertEqual(len({r["result_state"] for r in b["execution_receipts"]}),1)
        self.assertLess(b["execution_receipts"][0]["expression_steps"],
                        b["execution_receipts"][1]["expression_steps"])
        self.assertLess(b["execution_receipts"][1]["expression_steps"],
                        b["execution_receipts"][2]["expression_steps"])
        self.assertTrue(verify_reflective_environment_processor(
              self.snapshot,self.capsule,b,
              expected_head=self.snapshot["head"],
              expected_identity=self.world.identity))

    def test_representation_changes_source_encoding_not_observed_value(self):
        s,c,other=source(self.world,representation_coordinate=17)
        self.assertNotEqual(other["generated_interpreter_digest"],
                            self.bridge["generated_interpreter_digest"])
        self.assertEqual(other["result_recorded_value"],self.bridge["result_recorded_value"])
        self.assertEqual(other["generated_algebra_digest"],
                         self.bridge["generated_algebra_digest"])
        self.assertTrue(verify_reflective_environment_processor(
            s,c,other,expected_head=s["head"],expected_identity=self.world.identity))

    def test_live_environment_change_preserves_two_independently_qualified_processors(self):
        g=TemporalIntelligenceGround("verified:reflective-transition")
        feed(g,[1,3,5])
        s0,p0,b0=source(g)
        feed(g,[100,101,104,109],offset=3,host="B")
        s1,p1,b1=source(g,coordinate=17)
        self.assertEqual([p0["family"],p1["family"]],["affine","quadratic"])
        self.assertEqual([b0["result_recorded_value"],b1["result_recorded_value"]],
                         ["5/1","109/1"])
        migration=certify_language_migration(s0,p0,s1,p1,
                 before_head=s0["head"],after_head=s1["head"],identity=g.identity)
        succession=certify_reflective_interpreter_succession(
            s0,p0,b0,s1,p1,b1,migration,
            old_head=s0["head"],new_head=s1["head"],identity=g.identity)
        self.assertEqual(succession["preservation"],"ONLY_SOURCE_OBSERVED_PREFIX")
        self.assertEqual(succession["future_behavior_equivalence"],
                         "NOT_ESTABLISHED")
        self.assertNotEqual(succession["old_reflective_processor"],
                            succession["new_reflective_processor"])
        self.assertTrue(verify_reflective_interpreter_succession(
            s0,p0,b0,s1,p1,b1,migration,succession,
            old_head=s0["head"],new_head=s1["head"],identity=g.identity))
        with self.assertRaises(ValueError):
            verify_reflective_interpreter_succession(
                s0,p0,b0,s1,p1,b1,migration,
                {**succession,"future_behavior_equivalence":"CERTIFIED"},
                old_head=s0["head"],new_head=s1["head"],identity=g.identity)

    def test_forged_reflective_execution_refused_after_outer_hash_rewrite(self):
        bad=deepcopy(self.bridge)
        bad["execution_receipts"][2]["result_state"]=0
        bad["bridge_digest"]=_digest(BRIDGE,{
           k:v for k,v in bad.items() if k!="bridge_digest"})
        with self.assertRaises(ValueError):
            verify_reflective_environment_processor(
                self.snapshot,self.capsule,bad,
                expected_head=self.snapshot["head"],
                expected_identity=self.world.identity)

    def test_stale_head_and_substituted_model_source_refused(self):
        with self.assertRaises(ValueError):
            construct_reflective_environment_processor(
                self.snapshot,self.capsule,
                expected_head="0"*64,expected_identity=self.world.identity)
        wrong=deepcopy(self.capsule)
        wrong["family"]="geometric"
        wrong["capsule_digest"]=_digest(wrong["schema"],{
            k:v for k,v in wrong.items() if k!="capsule_digest"})
        with self.assertRaises(ValueError):
            construct_reflective_environment_processor(
                self.snapshot,wrong,
                expected_head=self.snapshot["head"],
                expected_identity=self.world.identity)

    def test_refuses_unqualified_or_overbudget_source_space(self):
        g=TemporalIntelligenceGround("verified:too-few")
        feed(g,[1,3])
        with self.assertRaises(ValueError):
            construct_reflective_environment_processor(
                g.checkpoint(),None,expected_head=g.head,
                expected_identity=g.identity)
        big=TemporalIntelligenceGround("verified:too-many")
        feed(big,list(range(9)))
        s,c=big.checkpoint(),EvolvingEnvironmentLanguage(big).current(expected_head=big.head)
        with self.assertRaises(ValueError):
            construct_reflective_environment_processor(
                s,c,expected_head=big.head,expected_identity=big.identity)

    def test_refuses_unadmitted_architecture_coordinate(self):
        for index in (-1,120,True,1.5):
            with self.subTest(coordinate=index),self.assertRaises(ValueError):
                construct_reflective_environment_processor(
                    self.snapshot,self.capsule,
                    expected_head=self.snapshot["head"],
                    expected_identity=self.world.identity,
                    representation_coordinate=index)


if __name__=="__main__":
    unittest.main()
