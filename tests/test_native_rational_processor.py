"""TIME-010: execute source-discovered laws through original E-CPU Virtual Layer."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.authenticated_temporal_source import (
    _genesis,public_key_hex,sign_source_observation,
    append_signed_observation,resume_signed_observer,
)
from vli.discovered_state_space import (
    construct_discovered_space,verify_discovered_space,
    navigate_discovered,SCHEMA as SOURCE_SCHEMA,
)
from vli.ground import _digest,load_json
from vli.native_rational_processor import (
    execute_native_rational,FencedNativeRationalProcessor,
    SCHEMA as NATIVE_SCHEMA,CANDIDATE,OPERATOR,
)


def sequence(seed,coefficients,count):
    vals=list(seed)
    while len(vals)<count:
        vals.append(sum(coefficients[j]*vals[-1-j]
                        for j in range(len(coefficients))))
    return vals


class Time010Tests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.key=Ed25519PrivateKey.generate()
        self.public=public_key_hex(self.key)

    def prepare(self, values, *, label):
        work=self.root/label
        work.mkdir()
        source="time010:"+label
        journal=work/"source.jsonl"
        checkpoint=work/"checkpoint.json"
        head=_genesis(source,self.public)
        for idx,value in enumerate(values):
            q=Fraction(value)
            packet=sign_source_observation(
                self.key,source=source,sequence=idx,time=f"{idx}/1",
                value=f"{q.numerator}/{q.denominator}")
            head,count=append_signed_observation(
                journal,packet,expected_head=head,
                expected_count=idx,source=source,public_key=self.public)
            self.assertEqual(count,idx+1)
        observer=resume_signed_observer(
            journal_path=journal,checkpoint_path=checkpoint,
            source=source,public_key=self.public,
            expected_source_head=head,expected_source_count=len(values),
            environment="time010-test:"+label,host="signed-external-observer")
        snapshot=load_json(checkpoint)
        source_contract={
            "journal_path":str(journal),"source_id":source,
            "public_key":self.public,"expected_source_head":head,
            "expected_source_count":len(values),
            "expected_ground_head":observer["result_ground_head"],
            "expected_ground_identity":observer["observer_identity"],
        }
        space=construct_discovered_space(
            snapshot,**source_contract,
            training_count=len(values)-4,
            validation_count=2,test_count=2)
        self.assertTrue(verify_discovered_space(
            snapshot,space,**source_contract))
        return snapshot,space,source_contract

    def execute(self,snapshot,space,contract,*,time,**extra):
        return execute_native_rational(
            snapshot,space,source_contract=contract,
            expected_space_digest=space["intention_space_digest"],
            intended_time=time,**extra)

    def test_dimension_four_is_genuinely_executed_in_native_hypervisor(self):
        vals=sequence([1,1,2,3],[1,1,0,1],14)
        snapshot,space,contract=self.prepare(vals,label="dimension-four")
        self.assertEqual(space["dimension"],4)
        native=self.execute(snapshot,space,contract,time="12/1",host="receiver-x")
        self.assertEqual(native["native_result"]["exact_value"],f"{vals[12]}/1")
        self.assertEqual(native["native_result"]["exact_state"],
                         navigate_discovered(space,"12/1")["calculation"]
                         ["state_vector"])
        self.assertEqual(native["native_result"]["evaluator"],
                         "NATIVE_ECPU_EXACT_BINARY_MATRIX_OPERATOR")
        self.assertEqual(native["native_operator_id"],OPERATOR)
        self.assertEqual(native["native_effect"]["kind"],
                         "virtual_layer.operator_result")
        self.assertTrue(native["native_mathematical_law_executed"])
        self.assertEqual(native["receiver_handler_invocations"],1)
        self.assertTrue(native["local_kernel_journal_verified"])
        self.assertFalse(native["external_resource_effect_committed"])
        self.assertFalse(native["remote_hardware_attested"])
        self.assertLessEqual(native["native_result"]["matrix_products"],8)
        independent=self.execute(snapshot,space,contract,time="12/1",
                                 host="receiver-y")
        self.assertEqual(independent["native_result"],native["native_result"])
        self.assertNotEqual(independent["host"],native["host"])

    def test_future_semantics_and_capability_denial(self):
        vals=sequence([1,2,3],[1,0,1],14)
        snap,space,contract=self.prepare(vals,label="permission")
        self.assertEqual(space["dimension"],3)
        with self.assertRaises(ValueError):
            self.execute(snap,space,contract,time="30/1")
        projected=self.execute(snap,space,contract,time="30/1",
                               allow_projection=True)
        self.assertEqual(projected["evidence_status"],
                         "UNVERIFIED_FUTURE_MODEL_PROJECTION")
        self.assertEqual(projected["native_result"]["exact_value"],
                         navigate_discovered(space,"30/1",
                             allow_projection=True)["value"])
        with self.assertRaises(ValueError):
            self.execute(snap,space,contract,time="11/1",grant_compute=False)
        self.assertNotEqual(projected["native_result"]["exact_value"],
                            navigate_discovered(space,"8/1")["value"])

    def test_source_semantics_tampering_cannot_rebind_native_handler(self):
        vals=sequence([1,2,3],[1,0,1],14)
        snap,space,contract=self.prepare(vals,label="tamper")
        good=self.execute(snap,space,contract,time="12/1")
        wrong=deepcopy(space)
        wrong["coefficients"][0]="2/1"
        wrong["intention_space_digest"]=_digest(
            SOURCE_SCHEMA,{k:v for k,v in wrong.items()
                           if k!="intention_space_digest"})
        with self.assertRaises(ValueError):
            execute_native_rational(snap,wrong,
                source_contract=contract,
                expected_space_digest=wrong["intention_space_digest"],
                intended_time="12/1")
        with self.assertRaises(ValueError):
            execute_native_rational(snap,space,
                source_contract={**contract,"expected_source_head":"0"*64},
                expected_space_digest=space["intention_space_digest"],
                intended_time="12/1")
        self.assertEqual(good["receiver_handler_invocations"],1)

    def test_quorum_reassigns_pure_native_result_and_refuses_old_host(self):
        vals=sequence([1,2,3],[1,0,1],14)
        snap,space,contract=self.prepare(vals,label="claims")
        root=self.root/"claim-state"
        def processor():
            return FencedNativeRationalProcessor(
                root,snap,space,source_contract=contract,
                expected_space_digest=space["intention_space_digest"],
                intended_time="12/1",job="resume-native")
        owner=processor()
        ticket=owner.claim(peer="member-a",host="native-host-a")
        claim_id=ticket.claim.claim_id
        first=owner.calculate(claim_id,peer="member-a",host="native-host-a")
        self.assertEqual(first["native_execution"]["receiver_handler_invocations"],1)
        self.assertEqual(first["result_exact_rational"],f"{vals[12]}/1")
        changed=owner.reassign(claim_id,peer="member-b",host="native-host-b",
            failure_evidence_digest="f"*64)
        self.assertEqual(changed.claim.current_assignment.epoch,2)
        with self.assertRaises(ValueError):
            owner.publish(first)
        with self.assertRaises(ValueError):
            owner.calculate(claim_id,peer="member-a",host="native-host-a")
        reopened=processor()
        second=reopened.calculate(claim_id,peer="member-b",host="native-host-b")
        self.assertEqual(second["assignment_epoch"],2)
        self.assertNotEqual(first["native_execution"]["host"],
                            second["native_execution"]["host"])
        forged=deepcopy(second)
        forged["result_exact_rational"]="500/1"
        forged["candidate_digest"]=_digest(
            CANDIDATE,{k:v for k,v in forged.items()
                       if k!="candidate_digest"})
        with self.assertRaises(ValueError):
            reopened.publish(forged)
        published=reopened.publish(second)
        self.assertEqual(published["result_exact_rational"],f"{vals[12]}/1")
        self.assertTrue(published["source_law_executed_by_native_hypervisor"])
        self.assertTrue(published["fenced_pure_result_published"])
        self.assertEqual(published["assignment_epoch"],2)
        self.assertFalse(published["external_effect_performed"])
        self.assertEqual(processor().verify_published(claim_id),published)
        with self.assertRaises(ValueError):
            processor().publish(first)

    def test_fenced_result_recovery_across_new_python_process(self):
        vals=sequence([2,3],[1,1],12)
        snap,space,contract=self.prepare(vals,label="hard-process")
        root=self.root/"hard-claim"
        owner=FencedNativeRationalProcessor(
            root,snap,space,source_contract=contract,
            expected_space_digest=space["intention_space_digest"],
            intended_time="10/1",job="subprocess-verified")
        ticket=owner.claim(peer="member-a",host="worker-a")
        claim_id=ticket.claim.claim_id
        old=owner.calculate(claim_id,peer="member-a",host="worker-a")
        owner.reassign(claim_id,peer="member-b",host="worker-b",
                       failure_evidence_digest="a"*64)
        with self.assertRaises(ValueError):
            owner.publish(old)
        newer=owner.calculate(claim_id,peer="member-b",host="worker-b")
        published=owner.publish(newer)
        args={
            "root":str(root),"snapshot":snap,"space":space,
            "source_contract":contract,
            "expected_space_digest":space["intention_space_digest"],
            "intended_time":"10/1","job":"subprocess-verified",
            "claim_id":claim_id,
        }
        encoded=self.root/"worker-args.json"
        encoded.write_text(json.dumps(args),encoding="utf-8")
        code=(
            "import json,sys;"
            "from vli.native_rational_processor import FencedNativeRationalProcessor;"
            "a=json.load(open(sys.argv[1],encoding='utf-8'));"
            "claim=a.pop('claim_id');"
            "p=FencedNativeRationalProcessor(**a);"
            "print(json.dumps(p.verify_published(claim),sort_keys=True))"
        )
        child=subprocess.run([sys.executable,"-c",code,str(encoded)],
                             capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(child.stdout),published)

    def test_reopened_processor_refuses_different_source_law_identity(self):
        vals=sequence([1,2,3],[1,0,1],14)
        snap,space,contract=self.prepare(vals,label="revision-a")
        root=self.root/"one-namespace"
        FencedNativeRationalProcessor(
            root,snap,space,source_contract=contract,
            expected_space_digest=space["intention_space_digest"],
            intended_time="12/1")
        with self.assertRaisesRegex(ValueError,"different mathematical"):
            FencedNativeRationalProcessor(
                root,snap,space,source_contract=contract,
                expected_space_digest=space["intention_space_digest"],
                intended_time="11/1")


if __name__=="__main__":
    unittest.main()
