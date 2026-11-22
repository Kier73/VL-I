"""Cross-repository SAT microarchitecture, native E-CPU, and quorum-fenced rehome."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algebraic_ground import encode_xor
from derived_ground import compile_proven_ground,split_guard
from frontier_ground import random_3cnf,brute_count
from self_forming_sat import digest
from vli.problem_formed_sat import DirectSatFabric


def proof():
    return compile_proven_ground(split_guard(split_guard(
        encode_xor(0b1111,1),guard_id=5),guard_id=6),6)


def fixture(root,source=None,intention='#SAT',capacity=8):
    return DirectSatFabric(root,proof() if source is None else source,
          intention=intention,maximum_free=12,partition_size=capacity,job='sat-native-ground')


def complete_stage(run,i,host='host-a',peer='peer-a'):
    token=run.claim(i,peer=peer,host=host)
    candidate=run.calculate(i,token.claim.claim_id,peer=peer,host=host)
    return run.publish(candidate)


class NativeSAT(unittest.TestCase):
    def test_sat_and_count_through_real_native_operators_and_fabric(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture(root)
            self.assertEqual(run.program['source_variables'],6)
            self.assertEqual(run.program['free_dimension'],5)
            self.assertEqual(len(run.program['partitions']),4)
            run.dispatch()
            self.assertEqual(len(run.fabric.snapshot()['queues']),4)
            for i in range(4):
                result=complete_stage(run,i,host='host-'+('a' if i%2 else 'b'),
                                           peer='peer-'+('a' if i%2 else 'b'))
                self.assertEqual(result['status'],'published')
                run=fixture(root)
            result=run.finish()
            self.assertEqual(result['status'],'SAT')
            self.assertEqual(result['count'],32)
            self.assertEqual(run.result(),result)
            self.assertEqual(run.fabric.snapshot()['queues'],[])

    def test_stale_old_host_correct_result_cannot_publish(self):
        with tempfile.TemporaryDirectory() as root:
            obj=fixture(root);obj.dispatch()
            token=obj.claim(0,peer='peer-a',host='host-a')
            claim=token.claim.claim_id
            old=obj.calculate(0,claim,peer='peer-a',host='host-a')
            self.assertGreater(old['result']['models'],0)
            moved=obj.reassign(0,claim,peer='peer-b',host='host-b',
                failure_evidence_digest=digest({'old-host':'killed'}))
            self.assertEqual(moved.claim.current_assignment.epoch,2)
            with self.assertRaises(ValueError):obj.publish(old)
            with self.assertRaises(Exception):obj.calculate(0,claim,peer='peer-a',host='host-a')
            self.assertEqual(obj.store.get(claim).claim.status.value,'assigned')
            current=obj.calculate(0,claim,peer='peer-b',host='host-b')
            obj.publish(current)
            self.assertEqual(obj.receipts['0']['host'],'host-b')
            self.assertEqual(obj.receipts['0']['assignment_epoch'],2)
            self.assertEqual(obj.dag.committed_nodes,('dispatch','part-0'))

    def test_missing_capability_leaves_queue_and_quorum_unmodified(self):
        with tempfile.TemporaryDirectory() as root:
            obj=fixture(root);obj.dispatch()
            token=obj.claim(0,peer='peer-a',host='host-a')
            before=obj.fabric.snapshot()
            with self.assertRaisesRegex(ValueError,'Hypervisor refused'):
                obj.calculate(0,token.claim.claim_id,peer='peer-a',host='host-a',grant_compute=False)
            self.assertEqual(obj.fabric.snapshot(),before)
            self.assertEqual(obj.store.get(token.claim.claim_id).claim.status.value,'assigned')

    def test_completed_quorum_before_dag_crash_reconciles_without_competing_result(self):
        with tempfile.TemporaryDirectory() as root:
            obj=fixture(root);obj.dispatch()
            ticket=obj.claim(0,peer='peer-a',host='host-a')
            candidate=obj.calculate(0,ticket.claim.claim_id,peer='peer-a',host='host-a')
            with patch.object(DirectSatFabric,'reconcile',side_effect=RuntimeError('simulate checkpoint death')):
                with self.assertRaisesRegex(RuntimeError,'checkpoint death'):
                    obj.publish(candidate)
            new=fixture(root)
            self.assertEqual(new.dag.committed_nodes,('dispatch',))
            new.reconcile(0,ticket.claim.claim_id)
            self.assertEqual(new.receipts['0']['result'],candidate['result'])
            self.assertEqual(new.store.get(ticket.claim.claim_id).claim.status.value,'completed')

    def test_changed_intention_and_forged_work_refused(self):
        with tempfile.TemporaryDirectory() as root:
            obj=fixture(root);obj.dispatch()
            with self.assertRaises(ValueError):fixture(root,intention='SAT')
            ticket=obj.claim(0,peer='peer-a',host='host-a')
            c=obj.calculate(0,ticket.claim.claim_id,peer='peer-a',host='host-a')
            bad=copy.deepcopy(c)
            bad['result']['models']+=1
            bad['result']['sha256']=digest({k:v for k,v in bad['result'].items() if k!='sha256'})
            bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
            with self.assertRaises(ValueError):obj.publish(bad)
            self.assertEqual(obj.store.get(ticket.claim.claim_id).claim.status.value,'assigned')

    def test_generic_source_still_needs_full_search(self):
        with tempfile.TemporaryDirectory() as root:
            cls=random_3cnf(8,25,79)
            source=compile_proven_ground(cls,8)
            obj=fixture(root,source=source,capacity=32)
            self.assertEqual(obj.program['free_dimension'],8)
            obj.dispatch()
            for i in range(len(obj.program['partitions'])):
                complete_stage(obj,i)
            answer=obj.finish()
            self.assertEqual(answer['count'],brute_count(cls,8))
            self.assertEqual(answer['evaluated'],256)

    def test_subprocess_candidate_does_not_commit_before_new_host_rehome(self):
        with tempfile.TemporaryDirectory() as root:
            obj=fixture(root);obj.dispatch()
            token=obj.claim(0,peer='peer-a',host='host-a')
            claim=token.claim.claim_id
            # A genuinely independent Python process runs real E-CPU receiver work
            # and exits without calling publication or altering the DAG.
            script=r'''
import json,os,sys
from pathlib import Path
from algebraic_ground import encode_xor
from derived_ground import compile_proven_ground,split_guard
from vli.problem_formed_sat import DirectSatFabric
from vli.problem_formed_claims import _atomic_json
root=Path(sys.argv[1]);claim=sys.argv[2]
source=compile_proven_ground(split_guard(split_guard(encode_xor(0b1111,1),guard_id=5),guard_id=6),6)
obj=DirectSatFabric(root,source,intention='#SAT',maximum_free=12,partition_size=8,job='sat-native-ground')
c=obj.calculate(0,claim,peer='peer-a',host='host-a')
_atomic_json(root/'suspended.json',c)
os._exit(0)
'''
            ret=subprocess.run([sys.executable,'-c',script,str(root),claim],
                               capture_output=True,text=True,timeout=60)
            self.assertEqual(ret.returncode,0,ret.stderr)
            old=json.loads((Path(root)/'suspended.json').read_text())
            replacement=fixture(root)
            replacement.reassign(0,claim,peer='peer-b',host='host-b',
                              failure_evidence_digest=digest({'reason':'worker process death'}))
            with self.assertRaises(ValueError):replacement.publish(old)
            fresh=replacement.calculate(0,claim,peer='peer-b',host='host-b')
            replacement.publish(fresh)
            self.assertEqual(replacement.receipts['0']['host'],'host-b')

if __name__=='__main__':unittest.main()
