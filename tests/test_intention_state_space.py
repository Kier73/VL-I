"""TIME-007: signed, held-out qualified intentional navigation without replay."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.ground import _digest,load_json
from vli.authenticated_temporal_source import (
    _genesis,public_key_hex,sign_source_observation,
    append_signed_observation,resume_signed_observer,
)
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.evolving_environment_language import EvolvingEnvironmentLanguage
from vli.intention_state_space import (
    SCHEMA, construct_intention_space,navigate_intention,
    verify_intention_navigation,
)
from vli.reflective_rational import (
    SEED, compile_intention_program,compile_self_interpreter,
    _codebook,_encode_source,host_run,verify_run,reflect_intention,
)


class IntentionStateSpaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.key=Ed25519PrivateKey.generate()
        self.public=public_key_hex(self.key)
        self.counter=0

    def fixture(self,label,values,*,denominator=1):
        folder=self.root/f"{label}-{self.counter}"
        self.counter+=1
        folder.mkdir()
        journal=folder/"source.jsonl"
        checkpoint=folder/"checkpoint.json"
        source="time007:"+label
        head=_genesis(source,self.public)
        for i,value in enumerate(values):
            q=Fraction(i,denominator)
            p=sign_source_observation(
                self.key,source=source,sequence=i,
                time=f"{q.numerator}/{q.denominator}",
                value=f"{value}/1")
            head,n=append_signed_observation(
                journal,p,expected_head=head,expected_count=i,
                source=source,public_key=self.public)
            self.assertEqual(n,i+1)
        observer=resume_signed_observer(
            journal_path=journal,checkpoint_path=checkpoint,
            source=source,public_key=self.public,
            expected_source_head=head,expected_source_count=len(values),
            environment="time007:"+label,host="signed-source-worker")
        snapshot=load_json(checkpoint)
        ground=TemporalIntelligenceGround.restore(
            snapshot,expected_head=snapshot["head"],
            expected_identity=observer["observer_identity"])
        capsule=EvolvingEnvironmentLanguage(ground).current(
            expected_head=ground.head)
        self.assertIsNotNone(capsule)
        args={
            "journal_path":journal,"source_id":source,"public_key":self.public,
            "expected_source_head":head,"expected_source_count":len(values),
            "expected_ground_head":ground.head,
            "expected_ground_identity":ground.identity,
        }
        space=construct_intention_space(snapshot,capsule,**args)
        return snapshot,capsule,space,args

    def test_four_environment_families_converge_to_compact_state_spatial_language(self):
        cases=[
          ("affine",[1,3,5],2,1000000,"2000001/1"),
          ("geometric",[2,6,18],1,100,"103075504146402266207914947899402/1"),
          ("quadratic",[0,1,4,9],3,1000000,"1000000000000/1"),
          ("recurrence2",[1,1,2,3,5],2,100,"573147844013817084101/1"),
        ]
        # Check long-range values with independent direct mathematical oracles.
        for family,values,dim,k,expected in cases:
            with self.subTest(family=family):
                snap,capsule,space,args=self.fixture(family,values)
                self.assertEqual(space["family"],family)
                self.assertEqual(space["dimension"],dim)
                self.assertEqual(len(space["transition_operator"]),dim)
                self.assertLessEqual(len(space["initial_state"]),3)
                self.assertFalse(space["intermediate_trajectory_enumerated"])
                result=navigate_intention(space,f"{k}/1",allow_projection=True)
                if family=="geometric":
                    self.assertEqual(result["value"],f"{2*3**k}/1")
                else:
                    self.assertEqual(result["value"],expected)
                self.assertEqual(result["evidence_status"],
                                 "UNVERIFIED_FUTURE_MODEL_PROJECTION")
                self.assertEqual(result["expanded_source_trajectory_steps"],0)
                stats=result["calculation"]["navigation_metrics"]
                self.assertEqual(stats["trajectory_samples_materialized"],0)
                self.assertEqual(stats["expanded_temporal_transitions"],0)
                self.assertLessEqual(stats["matrix_products"],2*k.bit_length())
                self.assertLessEqual(stats["binary_bits"],20)
                self.assertTrue(verify_intention_navigation(
                    snap,capsule,space,result,**args))

    def test_intention_is_coordinate_not_constructed_trace(self):
        snap,cap,space,args=self.fixture("quadratic-local",[0,1,4,9])
        observed=navigate_intention(space,"3/1")
        self.assertEqual(observed["value"],"9/1")
        self.assertEqual(observed["evidence_status"],
                         "EXACT_RECORDED_SOURCE_COORDINATE")
        far=navigate_intention(space,"1000000/1",allow_projection=True)
        self.assertEqual(far["value"],"1000000000000/1")
        self.assertEqual(far["calculation"]["navigation_metrics"]
                         ["trajectory_samples_materialized"],0)
        self.assertLess(far["calculation"]["navigation_metrics"]
                        ["matrix_products"],50)
        self.assertTrue(verify_intention_navigation(
            snap,cap,space,far,**args))

    def test_exact_half_interval_environment_coordinates(self):
        snap,cap,space,args=self.fixture("affine-half",[1,3,5],denominator=2)
        self.assertEqual(space["time_step"],"1/2")
        current=navigate_intention(space,"1/1")
        self.assertEqual(current["value"],"5/1")
        future=navigate_intention(space,"100/1",allow_projection=True)
        self.assertEqual(future["logical_step_coordinate"],200)
        self.assertEqual(future["value"],"401/1")
        with self.assertRaises(ValueError):
            navigate_intention(space,"3/4",allow_projection=True)

    def test_unverified_projection_opt_in_and_bounded_target(self):
        _,_,space,_=self.fixture("consent",[1,3,5])
        with self.assertRaisesRegex(ValueError,"opt"):
            navigate_intention(space,"10/1")
        for candidate in ("-1/1","1048577/1","1/3"):
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):
                navigate_intention(space,candidate,allow_projection=True)

    def test_signed_source_authority_and_rehashed_geometry_refusal(self):
        snap,cap,space,args=self.fixture("source-fencing",[1,3,5])
        for corrupt in (
             {"public_key":public_key_hex(Ed25519PrivateKey.generate())},
             {"expected_source_head":"0"*64},
             {"source_id":"foreign-source"},
             {"expected_ground_head":"a"*64},
        ):
            with self.subTest(corrupt=tuple(corrupt)),self.assertRaises(ValueError):
                construct_intention_space(snap,cap,**{**args,**corrupt})
        forged=deepcopy(space)
        forged["transition_operator"][0][1]="7/1"
        forged["intention_space_digest"]=_digest(SCHEMA,{
            k:v for k,v in forged.items() if k!="intention_space_digest"})
        with self.assertRaises(ValueError):
            verify_intention_navigation(
                snap,cap,forged,navigate_intention(
                    forged,"10/1",allow_projection=True),**args)

    def test_no_false_numeric_output_after_rehash(self):
        snap,cap,space,args=self.fixture("forged-output",[0,1,4,9])
        record=navigate_intention(space,"6/1",allow_projection=True)
        bad=deepcopy(record)
        bad["value"]="999/1"
        bad["navigation_digest"]=_digest(bad["schema"],{
            k:v for k,v in bad.items() if k!="navigation_digest"})
        with self.assertRaises(ValueError):
            verify_intention_navigation(snap,cap,space,bad,**args)

    def test_environment_representation_is_family_specific(self):
        for family,values,dimension,output in [
            ("affine",[1,3,5],2,0),
            ("geometric",[2,6,18],1,0),
            ("quadratic",[0,1,4,9],3,0),
            ("recurrence2",[1,1,2,3,5],2,1),
        ]:
            with self.subTest(family=family):
                _,_,space,_=self.fixture(family,values)
                self.assertEqual(space["output_coordinate"],output)
                self.assertEqual(space["dimension"],dimension)

    def test_reflective_arithmetic_compiles_from_dimensions_not_target_trajectory(self):
        for d in (1,2,3):
            program=compile_intention_program(d,0)
            self.assertEqual(tuple(name for name,_ in program),
                             ("matmul","matvec","power","main"))
            source=repr(program)
            self.assertNotIn("matrix_power",source)
            self.assertNotIn("eval(",source)
            self.assertLess(len(source),24000)
        self.assertEqual(SEED,("at","cons","eq","rest","qadd","qmul","odd","half"))

    def test_independent_source_interpreter_agrees_at_zero_and_integer_steps(self):
        snap,cap,space,args=self.fixture("reflected-geo",[2,6,18])
        for index in (0,1,2,3,7,17):
            with self.subTest(index=index):
                expected=navigate_intention(
                    space,f"{index}/1",allow_projection=True if index>=3 else False)
                certificate=reflect_intention(
                    space,index,expected_value=expected["value"],depth=1)
                self.assertEqual([x["depth"] for x in certificate["reflection_receipts"]],
                                 [0,1])
                self.assertTrue(all(x["host_and_independent_continuation_agree"]
                                    for x in certificate["reflection_receipts"]))
                self.assertEqual(certificate["trajectory_states_constructed"],0)

    def test_same_language_evaluator_interprets_its_own_encoded_source_twice(self):
        _,_,space,_=self.fixture("self-evaluating",[2,6,18])
        record=reflect_intention(space,2,expected_value="18/1",depth=2)
        receipts=record["reflection_receipts"]
        self.assertEqual([x["depth"] for x in receipts],[0,1,2])
        self.assertLess(receipts[0]["expression_steps"],
                        receipts[1]["expression_steps"])
        self.assertLess(receipts[1]["expression_steps"],
                        receipts[2]["expression_steps"])
        self.assertTrue(record["primitive_arithmetic_is_external_seed"])
        self.assertFalse(record["reflector_is_unbounded_hardware"])

    def test_quadratic_math_really_runs_through_reflective_source(self):
        _,_,space,_=self.fixture("reflected-square",[0,1,4,9])
        result=reflect_intention(space,3,expected_value="9/1",depth=1)
        self.assertEqual(result["state_dimension"],3)
        self.assertEqual(result["reflection_receipts"][1]["result_exact_rational"],"9/1")

    def test_counterexample_does_not_silently_keep_previous_interpreter(self):
        snap,cap,space,args=self.fixture("switch", [1,3,5,7,100,101,104,109])
        self.assertEqual(space["family"],"quadratic")
        self.assertEqual(space["source_observation_count"],4)
        self.assertEqual(space["source_witnesses"][-1],snap["events"][-1]["digest"])
        outcome=navigate_intention(space,"8/1",allow_projection=True)
        self.assertEqual(outcome["value"],"116/1")
        self.assertEqual(outcome["evidence_status"],
                         "UNVERIFIED_FUTURE_MODEL_PROJECTION")
        self.assertEqual(outcome["logical_step_coordinate"],4)

    def test_source_language_rejects_missing_primitives_and_false_program(self):
        _,_,space,_=self.fixture("illegal-primitives",[2,6,18])
        code=_codebook(space["intention_space_digest"])
        interpreter=compile_self_interpreter(code)
        fake=(("main",("prim","matrix_power",())),)
        with self.assertRaises(ValueError):
            host_run(fake,"main",())
        with self.assertRaises(ValueError):
            verify_run(fake,"main",())
        # Mutating a generated opcode in the *encoded* source must fail.
        original=compile_intention_program(1,0)
        encoded=list(_encode_source(original,code))
        name,node=encoded[-1]
        encoded[-1]=(name,(999,)+node[1:])
        with self.assertRaises(ValueError):
            verify_run(interpreter,"run",(tuple(encoded),"main",(0,(((1,1),),),((1,1),))))


if __name__=="__main__":
    unittest.main()
