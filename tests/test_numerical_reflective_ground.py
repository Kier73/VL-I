"""TIME-006: signed-source exact numerical operators in genuine NCR-002 reflection."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.ground import _digest,load_json
from vli.authenticated_temporal_source import (
    _genesis,append_signed_observation,public_key_hex,
    sign_source_observation,resume_signed_observer,
)
from vli.evolving_environment_language import (
    EvolvingEnvironmentLanguage,
)
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.numerical_reflective_ground import (
    SCHEMA, construct_authenticated_numerical_processor,
    verify_authenticated_numerical_processor,
    _step_model,
)


class SignedNumericalReflectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)
        self.signer=Ed25519PrivateKey.generate()
        self.public=public_key_hex(self.signer)
        self.source="signed-numeric-source"

    def fixture(self,values,*,name,horizon,time_denominator=1):
        root=self.directory/name
        root.mkdir(parents=True,exist_ok=True)
        journal=root/"events.jsonl"
        checkpoint=root/"observer.json"
        head=_genesis(self.source,self.public)
        for i,value in enumerate(values):
            packet=sign_source_observation(
                self.signer,source=self.source,sequence=i,
                time=f"{i}/{time_denominator}",
                value=f"{value}/1")
            head,count=append_signed_observation(
                journal,packet,expected_head=head,expected_count=i,
                source=self.source,public_key=self.public)
            self.assertEqual(count,i+1)
        environment="time006:"+name
        result=resume_signed_observer(
            journal_path=journal,checkpoint_path=checkpoint,
            source=self.source,public_key=self.public,
            expected_source_head=head,expected_source_count=len(values),
            environment=environment,host="verified-source-process")
        snapshot=load_json(checkpoint)
        ground=TemporalIntelligenceGround.restore(
            snapshot,expected_head=snapshot["head"],
            expected_identity=result["observer_identity"])
        capsule=EvolvingEnvironmentLanguage(ground).current(
            expected_head=ground.head)
        self.assertIsNotNone(capsule)
        args=dict(
            source_journal=journal,source_id=self.source,public_key=self.public,
            expected_source_head=head,expected_source_count=len(values),
            expected_ground_head=ground.head,
            expected_ground_identity=ground.identity,
        )
        record=construct_authenticated_numerical_processor(
            snapshot,capsule,horizon=horizon,
            allow_projection=horizon>len(values)-1,**args)
        self.assertTrue(verify_authenticated_numerical_processor(
            snapshot,capsule,record,**args))
        return snapshot,capsule,record,args

    def test_four_distinct_environmental_numeric_languages(self):
        examples=[
            ("affine",[1,3,5],10,"21/1"),
            ("geometric",[2,6,18],8,"13122/1"),
            ("quadratic",[0,1,4,9],8,"64/1"),
            ("recurrence2",[1,1,2,3,5],10,"89/1"),
        ]
        digests=set()
        for family,values,horizon,answer in examples:
            with self.subTest(family=family):
                snap,capsule,receipt,arguments=self.fixture(
                    values,name=family,horizon=horizon)
                self.assertEqual(capsule["family"],family)
                self.assertEqual(receipt["language_family"],family)
                self.assertEqual(receipt["result_exact_value"],answer)
                self.assertEqual(receipt["result_evidence_status"],
                                 "UNVERIFIED_FUTURE_MODEL_STATE")
                self.assertEqual(receipt["runtime_arithmetic"],
                                 "SOURCE_DERIVED_FINITE_TABLE_LOOKUP")
                self.assertEqual(receipt["compiled_numeric_states"][0]["status"],
                                 "EXACT_RECORDED_SOURCE_STATE")
                self.assertEqual(len(receipt["compiled_numeric_states"]),horizon+1)
                self.assertEqual(
                    sum(s["status"]=="EXACT_RECORDED_SOURCE_STATE"
                        for s in receipt["compiled_numeric_states"]),len(values))
                self.assertTrue(all(
                    s["source_witness"] is None for s in
                    receipt["compiled_numeric_states"][len(values):]))
                self.assertEqual([r["reflection_depth"] for r in
                    receipt["generated_semantic_processor"]["depth_receipts"]],
                    [0,1,2])
                self.assertEqual(len(set(r["result_state_index"]
                    for r in receipt["generated_semantic_processor"]["depth_receipts"])),1)
                self.assertEqual(
                    receipt["generated_semantic_processor"]["depth_receipts"][0]["result_state_index"],
                    horizon)
                self.assertLess(
                    receipt["generated_semantic_processor"]["depth_receipts"][0]["expression_steps"],
                    receipt["generated_semantic_processor"]["naive_unrolled_direct_steps"])
                self.assertEqual(len(receipt["generated_semantic_processor"]["task_instruction_word"]),
                                 horizon.bit_count())
                digests.add(receipt["numerical_processor_digest"])
        self.assertEqual(len(digests),4)

    def test_horizon_exact_source_coordinate_never_claims_future(self):
        snap,language,record,args=self.fixture(
            [2,6,18],name="recorded",horizon=2)
        self.assertEqual(record["result_evidence_status"],
                         "EXACT_RECORDED_SOURCE_STATE")
        self.assertEqual(record["result_exact_value"],"18/1")
        self.assertTrue(verify_authenticated_numerical_processor(
            snap,language,record,**args))

    def test_half_interval_time_coordinates_are_retained_exactly(self):
        snap,language,record,args=self.fixture(
            [1,3,5],name="half",horizon=6,time_denominator=2)
        self.assertEqual(record["result_time"],"3/1")
        self.assertEqual(record["result_exact_value"],"13/1")
        self.assertEqual(record["compiled_numeric_states"][1]["time"],"1/2")

    def test_compiled_state_memory_matters_when_values_repeat(self):
        snap,language,record,args=self.fixture(
            [1,1,2,3,5],name="memory",horizon=7)
        first,second=record["compiled_numeric_states"][:2]
        self.assertEqual(first["value"],second["value"])
        self.assertNotEqual(first["step"],second["step"])
        self.assertIsNone(first["previous_value"])
        self.assertEqual(second["previous_value"],"1/1")
        self.assertEqual(record["result_exact_value"],"21/1")

    def test_unauthorized_future_projection_is_explicitly_refused(self):
        snap,language,record,args=self.fixture(
            [1,3,5],name="projection-refusal",horizon=8)
        with self.assertRaisesRegex(ValueError,"opt-in"):
            construct_authenticated_numerical_processor(
                snap,language,horizon=10,allow_projection=False,**args)
        with self.assertRaises(ValueError):
            construct_authenticated_numerical_processor(
                snap,language,horizon=1,**args)
        with self.assertRaises(ValueError):
            construct_authenticated_numerical_processor(
                snap,language,horizon=13,allow_projection=True,**args)

    def test_independent_verifier_rejects_rehashed_wrong_numeric_table(self):
        snap,language,record,args=self.fixture(
            [1,3,5],name="tamper-table",horizon=8)
        forged=deepcopy(record)
        forged["compiled_action_table"][0][0]=7
        forged["numerical_processor_digest"]=_digest(SCHEMA,{
            k:v for k,v in forged.items() if k!="numerical_processor_digest"})
        with self.assertRaises(ValueError):
            verify_authenticated_numerical_processor(
                snap,language,forged,**args)

    def test_independent_verifier_rejects_wrong_future_rational_value(self):
        snap,language,record,args=self.fixture(
            [1,3,5],name="tamper-future",horizon=8)
        forged=deepcopy(record)
        forged["compiled_numeric_states"][-1]["value"]="9999/1"
        forged["result_exact_value"]="9999/1"
        forged["numerical_processor_digest"]=_digest(SCHEMA,{
            k:v for k,v in forged.items() if k!="numerical_processor_digest"})
        with self.assertRaises(ValueError):
            verify_authenticated_numerical_processor(
                snap,language,forged,**args)

    def test_source_key_and_chain_pin_are_actual_admission_authorities(self):
        snap,language,record,args=self.fixture(
            [2,6,18],name="foreign-signer",horizon=5)
        other=public_key_hex(Ed25519PrivateKey.generate())
        with self.assertRaises(ValueError):
            construct_authenticated_numerical_processor(
                snap,language,horizon=5,allow_projection=True,
                **{**args,"public_key":other})
        with self.assertRaises(ValueError):
            construct_authenticated_numerical_processor(
                snap,language,horizon=5,allow_projection=True,
                **{**args,"expected_source_head":"0"*64})
        with self.assertRaises(ValueError):
            construct_authenticated_numerical_processor(
                snap,language,horizon=5,allow_projection=True,
                **{**args,"source_id":"untrusted-source"})

    def test_unqualified_or_counterexample_source_cannot_emit_machine(self):
        root=self.directory/"no-law"
        root.mkdir(parents=True)
        signed_path=root/"source.jsonl"
        snap_path=root/"checkpoint.json"
        head=_genesis(self.source,self.public)
        data=[0,1,3,8,19]
        for i,value in enumerate(data):
            packet=sign_source_observation(self.signer,source=self.source,
                  sequence=i,time=f"{i}/1",value=f"{value}/1")
            head,_=append_signed_observation(
                signed_path,packet,expected_head=head,expected_count=i,
                source=self.source,public_key=self.public)
        report=resume_signed_observer(
            journal_path=signed_path,checkpoint_path=snap_path,
            source=self.source,public_key=self.public,
            expected_source_head=head,expected_source_count=len(data),
            environment="time006:no-law",host="host")
        snap=load_json(snap_path)
        ground=TemporalIntelligenceGround.restore(snap,
            expected_head=snap["head"],expected_identity=report["observer_identity"])
        self.assertIsNone(EvolvingEnvironmentLanguage(ground).current(
            expected_head=ground.head))
        with self.assertRaises(ValueError):
            construct_authenticated_numerical_processor(
                snap,None,source_journal=signed_path,
                source_id=self.source,public_key=self.public,
                expected_source_head=head,expected_source_count=len(data),
                expected_ground_head=ground.head,
                expected_ground_identity=ground.identity,
                horizon=5,allow_projection=True)

    def test_difference_equations_are_independent_of_source_ast(self):
        from fractions import Fraction
        self.assertEqual(_step_model("quadratic",["0/1","1/1","2/1"],
            3,Fraction(9),Fraction(4)),Fraction(16))
        self.assertEqual(_step_model("recurrence2",["1/1","1/1","1/1","1/1"],
            4,Fraction(5),Fraction(3)),Fraction(8))
        self.assertEqual(_step_model("geometric",["2/1","3/1"],
            2,Fraction(18),Fraction(6)),Fraction(54))


if __name__=="__main__":
    unittest.main()
