"""PFI-005: source-equivalent SAT interpreter revision, native host execution.

These tests use the actual SFI-003 meta-SAT proof constructor and the existing
PFI-004/ E-CPU claim store, Hypervisor, FIFO, DAG and recovery lineage.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_recursive_sat_compiler import REPAIRABLE
from morphic_sat import canonical,digest
from vli.problem_formed_revision import (
    create_revision_source,verify_revision_source,RevisedSatFabric,
    morphic_runtime,translate_source_witness,verify_translated_witness)
from vli.problem_formed_sat import DirectSatFabric


def qualified_source():
    return create_revision_source(REPAIRABLE,6)


def execute_run(root,source,revision):
    obj=RevisedSatFabric(root,source,revision=revision,
                         maximum_free=6,partition_size=8,job="same-source-goal")
    obj.dispatch()
    for i in range(len(obj.program["partitions"])):
        token=obj.claim(i,peer="peer-a",host="host-a")
        candidate=obj.calculate(i,token.claim.claim_id,peer="peer-a",host="host-a")
        obj.publish(candidate)
        # Preserve the same grounded machine after its process-local object dies.
        obj=RevisedSatFabric(root,source,revision=revision,
                             maximum_free=6,partition_size=8,job="same-source-goal")
    return obj.finish(),obj


class RevisionContinuity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=qualified_source()

    def test_direct_self_application_is_proof_checked(self):
        source=self.source
        proof=verify_revision_source(source)
        self.assertTrue(proof["source_equivalence_verified"])
        self.assertEqual(proof["verified_register_reduction"],1)
        self.assertEqual(len(source["baseline"]["registers"]),5)
        self.assertEqual(len(source["refined"]["registers"]),4)
        self.assertEqual(source["meta_report"]["outcome"],
                         "IMPROVED_BY_OWN_SAT_INTERPRETER")

    def test_register_coordinate_translation_preserves_real_source_witness(self):
        source=self.source
        satisfiers=[v for v in range(1<<6)
                    if all(any(((v>>(abs(l)-1))&1)==int(l>0) for l in c)
                           for c in source["source_clauses"])]
        self.assertEqual(len(satisfiers),5)
        for bits in satisfiers:
            c=translate_source_witness(source,bits)
            self.assertTrue(verify_translated_witness(source,c))
            self.assertEqual(c["source_assignment_decimal"],str(bits))
            self.assertNotEqual(c["baseline_ground_sha256"],c["refined_ground_sha256"])

    def test_native_fabric_executes_both_source_equivalent_processors(self):
        with tempfile.TemporaryDirectory() as root:
            before,a=execute_run(Path(root)/"before",self.source,"baseline")
            after,b=execute_run(Path(root)/"after",self.source,"refined")
            self.assertEqual(before["status"],"SAT")
            self.assertEqual(after["status"],"SAT")
            self.assertEqual(before["count"],after["count"])
            self.assertEqual(after["count"],5)
            self.assertEqual(before["evaluated"],32)
            self.assertEqual(after["evaluated"],16)
            self.assertEqual(len(a.program["partitions"]),4)
            self.assertEqual(len(b.program["partitions"]),2)
            self.assertEqual(a.program["source_sha256"],b.program["source_sha256"])
            self.assertNotEqual(a.program["digest"],b.program["digest"])
            self.assertNotEqual(a.identity,b.identity)

    def test_late_current_result_cannot_publish_after_native_reassignment(self):
        with tempfile.TemporaryDirectory() as root:
            obj=RevisedSatFabric(root,self.source,maximum_free=6,partition_size=8)
            obj.dispatch()
            token=obj.claim(0,peer="peer-a",host="host-a")
            claim=token.claim.claim_id
            old=obj.calculate(0,claim,peer="peer-a",host="host-a")
            new=obj.reassign(0,claim,peer="peer-b",host="host-b",
                             failure_evidence_digest=digest({"terminated":True}))
            self.assertEqual(new.claim.current_assignment.epoch,2)
            with self.assertRaises(ValueError):
                obj.publish(old)
            current=obj.calculate(0,claim,peer="peer-b",host="host-b")
            obj.publish(current)
            self.assertEqual(obj.receipts["0"]["host"],"host-b")
            self.assertEqual(obj.receipts["0"]["assignment_epoch"],2)

    def test_prior_revision_checkpoint_cannot_be_resumed_as_new_revision(self):
        with tempfile.TemporaryDirectory() as root:
            baseline=RevisedSatFabric(root,self.source,revision="baseline",
                                      maximum_free=6,partition_size=8)
            baseline.dispatch()
            with self.assertRaisesRegex(ValueError,"Changed SAT interpreter epoch"):
                RevisedSatFabric(root,self.source,revision="refined",
                                 maximum_free=6,partition_size=8)
            self.assertEqual(baseline.dag.committed_nodes,("dispatch",))

    def test_old_receipts_do_not_certify_different_processor_coordinates(self):
        before=morphic_runtime(revision="baseline")
        after=morphic_runtime(revision="refined")
        pg,_=before[0](self.source,maximum_free=6,partition_size=8)
        rg,_=after[0](self.source,maximum_free=6,partition_size=8)
        part=before[2](pg,0)
        self.assertTrue(before[3](pg,part))
        with self.assertRaises(ValueError):
            after[3](rg,part)
        with self.assertRaises(ValueError):
            after[4](rg,[part],source_receipt=self.source)

    def test_strict_partial_work_refusal(self):
        run=morphic_runtime()
        pg,_=run[0](self.source,maximum_free=6,partition_size=8)
        first=run[2](pg,0)
        with self.assertRaises(ValueError):
            run[4](pg,[first],source_receipt=self.source)
        bad=copy.deepcopy(first)
        bad["models"]+=1
        bad["digest"]=digest({k:v for k,v in bad.items() if k!="digest"})
        with self.assertRaises(ValueError):
            run[3](pg,bad)

    def test_source_and_meta_forgery_rejected_even_if_rehashed(self):
        for key,transform in [
            ("refined",lambda x:x["instructions"][0]["table"].__setitem__(0,1-x["instructions"][0]["table"][0])),
            ("meta_report",lambda x:x.update(refined_registers=x["refined_registers"]+1)),
            ("source_clauses",lambda x:x[0].__setitem__(0,-x[0][0])),
        ]:
            bad=copy.deepcopy(self.source)
            transform(bad[key])
            if key=="meta_report":
                bad[key]["digest"]=digest({k:v for k,v in bad[key].items() if k!="digest"})
            if key=="refined":
                bad[key]["digest"]=digest({k:v for k,v in bad[key].items() if k!="digest"})
            bad["digest"]=digest({k:v for k,v in bad.items() if k!="digest"})
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_revision_source(bad)

    def test_forged_coordinate_translation_rejected(self):
        witness=next(bits for bits in range(1<<6) if all(
            any(((bits>>(abs(l)-1))&1)==int(l>0) for l in c)
            for c in self.source["source_clauses"]))
        valid=translate_source_witness(self.source,witness)
        bad=copy.deepcopy(valid)
        bad["coordinates"]["refined"]=str(int(bad["coordinates"]["refined"])^1)
        bad["digest"]=digest({k:v for k,v in bad.items() if k!="digest"})
        with self.assertRaises(ValueError):
            verify_translated_witness(self.source,bad)

    def test_capacity_refusal_does_not_infer_unsat(self):
        runtime=morphic_runtime()
        with self.assertRaisesRegex(ValueError,"cannot cover"):
            runtime[0](self.source,maximum_free=3,partition_size=2)
        with self.assertRaises(ValueError):
            create_revision_source([(1,2),(1,-2)],2)

    def test_run_specific_timings_do_not_change_semantic_processor_identity(self):
        alternate=copy.deepcopy(self.source)
        alternate["meta_report"]["elapsed_ms"]+=123.125
        alternate["meta_report"]["digest"]=digest({k:v for k,v in alternate["meta_report"].items()
                                                   if k!="digest"})
        alternate["digest"]=digest({k:v for k,v in alternate.items() if k!="digest"})
        self.assertNotEqual(alternate["digest"],self.source["digest"])
        self.assertTrue(verify_revision_source(alternate)["source_equivalence_verified"])
        runtime=morphic_runtime(revision="refined")
        first,_=runtime[0](self.source,maximum_free=6,partition_size=8)
        second,_=runtime[0](alternate,maximum_free=6,partition_size=8)
        self.assertEqual(first,second)

    def test_hard_exited_revised_worker_and_fresh_source_reconstruction(self):
        with tempfile.TemporaryDirectory() as root:
            obj=RevisedSatFabric(root,self.source,maximum_free=6,partition_size=8)
            obj.dispatch()
            token=obj.claim(0,peer="peer-a",host="host-a")
            claim=token.claim.claim_id
            # A second process *rebuilds* the proof and exact semantic program.
            # It calculates but hard exits without publishing. The original
            # process has no access to that worker's in-memory state.
            script=r"""
import os,sys
from pathlib import Path
from test_recursive_sat_compiler import REPAIRABLE
from vli.problem_formed_revision import create_revision_source,RevisedSatFabric
from vli.problem_formed_claims import _atomic_json
root=Path(sys.argv[1]);claim=sys.argv[2]
source=create_revision_source(REPAIRABLE,6)
run=RevisedSatFabric(root,source,maximum_free=6,partition_size=8)
candidate=run.calculate(0,claim,peer="peer-a",host="host-a")
_atomic_json(root/'suspended.json',candidate)
os._exit(0)
"""
            completed=subprocess.run([sys.executable,"-c",script,str(root),claim],
                                     capture_output=True,text=True,timeout=90,
                                     env=os.environ.copy())
            self.assertEqual(completed.returncode,0,completed.stderr)
            old=json.loads((Path(root)/"suspended.json").read_text())
            reconstructed=create_revision_source(REPAIRABLE,6)
            restarted=RevisedSatFabric(root,reconstructed,maximum_free=6,partition_size=8)
            self.assertEqual(restarted.identity,obj.identity)
            restarted.reassign(0,claim,peer="peer-b",host="host-b",
                               failure_evidence_digest=digest({"hard_exit":True}))
            with self.assertRaises(ValueError):
                restarted.publish(old)
            current=restarted.calculate(0,claim,peer="peer-b",host="host-b")
            restarted.publish(current)
            self.assertEqual(restarted.receipts["0"]["host"],"host-b")

    def test_baseline_pfi004_adapter_default_unmodified(self):
        from algebraic_ground import encode_xor
        from derived_ground import compile_proven_ground
        with tempfile.TemporaryDirectory() as root:
            src=compile_proven_ground(encode_xor(0b111,1),3)
            obj=DirectSatFabric(root,src,maximum_free=3,partition_size=8)
            self.assertEqual(obj.program["source_variables"],3)
            obj.dispatch()
            token=obj.claim(0,peer="peer-a",host="host-a")
            result=obj.calculate(0,token.claim.claim_id,peer="peer-a",host="host-a")
            obj.publish(result)
            self.assertEqual(obj.finish()["count"],4)


if __name__=="__main__":
    unittest.main()
