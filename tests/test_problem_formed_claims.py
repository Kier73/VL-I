"""PFI-003 independent-process publication vs calculation experiments."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mailcpu import TransferableExecutionClaimError
from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from vli.problem_formed_interpreter import construct
from vli.problem_formed_claims import FencedFormedProcessor, _hash_record


def problem():
    n=128
    rows=[1<<i for i in range(n)]
    rows[0]=3
    return SymbolicSource(n,(AffineMap(tuple(rows),1),),(1,),
                          'pfi003-declared-binary-affine-source')


def fixture(root):
    src=problem();q,b=construct(src)
    return FencedFormedProcessor(root,src,q,b,initial=3,
                                 gates_per_host=2,job='formed-identity')


def advance_stage(obj, name='host-a', peer='peer-a'):
    ticket=obj.claim(host=name,peer=peer)
    claim=ticket.claim.claim_id
    c=obj.calculate(claim,peer=peer,host=name)
    return obj.publish(c)


class FormedClaimsTests(unittest.TestCase):
    def test_entire_generated_fabric_with_quorum_publication(self):
        with tempfile.TemporaryDirectory() as root:
            obj=fixture(root)
            self.assertEqual(obj.local_step(),'dispatch')
            stage_count=len(obj.run.steps)
            for i in range(stage_count):
                result=advance_stage(obj,name=('host-a' if i%2==0 else 'host-b'),
                                      peer=('peer-a' if i%2==0 else 'peer-b'))
                self.assertEqual(result['status'],'published')
                self.assertEqual(result['stage'],i)
                obj=fixture(root)
            self.assertEqual(obj.local_step(),'finish')
            self.assertEqual(obj.result()['observer_result'],
                             obj.source.observe(obj.source.generators[0].apply(3)))
            self.assertEqual(obj.result()['executed_host_stages'],stage_count)
            self.assertEqual(len(obj.run.receipts),stage_count)

    def test_reassignment_fences_correct_stale_result(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture(root);run.local_step()
            issued=run.claim(peer='peer-a',host='host-a')
            claim=issued.claim.claim_id
            old=run.calculate(claim,peer='peer-a',host='host-a')
            moved=run.reassign(claim,peer='peer-b',host='host-b',
                 failure_evidence_digest=_hash_record({'failure':'old host killed'}))
            self.assertEqual(moved.claim.current_assignment.epoch,2)
            with self.assertRaises(TransferableExecutionClaimError):
                issued.verify_current(run.coordinator)
            with self.assertRaises(ValueError):
                run.publish(old)
            with self.assertRaises(TransferableExecutionClaimError):
                run.calculate(claim,peer='peer-a',host='host-a')
            current=run.calculate(claim,peer='peer-b',host='host-b')
            result=run.publish(current)
            self.assertEqual(result['stage'],0)
            self.assertEqual(run.run.receipts[0]['host'],'host-b')
            self.assertEqual(run.store.get(claim).claim.current_assignment.epoch,2)

    def test_without_local_hypervisor_capability_no_candidate_or_published_stage(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture(root);run.local_step()
            claim=run.claim(peer='peer-a',host='host-a').claim.claim_id
            checkpoint=run.run.snapshot()
            with self.assertRaisesRegex(ValueError,'Hypervisor refused'):
                run.calculate(claim,peer='peer-a',host='host-a',grant_compute=False)
            self.assertEqual(run.run.snapshot(),checkpoint)
            self.assertEqual(run.store.get(claim).claim.status.value,'assigned')

    def test_after_native_quorum_completion_checkpoint_crash_is_reconciled(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture(root);run.local_step()
            claim=run.claim(peer='peer-a',host='host-a').claim.claim_id
            c=run.calculate(claim,peer='peer-a',host='host-a')
            with patch.object(FencedFormedProcessor,'reconcile',side_effect=RuntimeError('simulated kill')):
                with self.assertRaisesRegex(RuntimeError,'simulated kill'):
                    run.publish(c)
            reopened=fixture(root)
            self.assertEqual(len(reopened.run.receipts),0)
            result=reopened.reconcile()
            self.assertEqual(result['status'],'published')
            self.assertEqual(reopened.run.receipts[0]['host'],'host-a')
            self.assertEqual(reopened.store.get(claim).claim.status.value,'completed')

    def test_candidate_tampering_even_rehashed_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture(root);run.local_step()
            claim=run.claim(peer='peer-a',host='host-a').claim.claim_id
            original=run.calculate(claim,peer='peer-a',host='host-a')
            for key,value in [('output',{'result':99}),('stage',13),
                              ('problem_epoch','0'*64),('host','stale-host')]:
                bad=copy.deepcopy(original);bad[key]=value
                bad['digest']=_hash_record({k:v for k,v in bad.items() if k!='digest'})
                with self.assertRaises(ValueError,msg=key):
                    run.publish(bad)
            self.assertEqual(run.store.get(claim).claim.status.value,'assigned')

    def test_cannot_open_old_ground_under_new_observer(self):
        with tempfile.TemporaryDirectory() as root:
            run=fixture(root);run.local_step()
            s=problem()
            altered=SymbolicSource(s.width,s.generators,(1,4),s.origin)
            q,b=construct(altered)
            with self.assertRaises(ValueError):
                FencedFormedProcessor(root,altered,q,b,initial=3,
                                      gates_per_host=2,job='formed-identity')

    def test_killed_subprocess_candidate_has_no_publication(self):
        script=Path(__file__).resolve().parents[1]/'examples'/'pfi_fenced_rehome.py'
        def cmd(root,op,**kwargs):
            args=[sys.executable,str(script),str(root),op]
            for k,v in kwargs.items():
                args.extend(['--'+k.replace('_','-'),str(v)])
            return json.loads(subprocess.check_output(args,text=True,timeout=30))
        with tempfile.TemporaryDirectory() as root:
            cmd(root,'dispatch')
            assigned=cmd(root,'claim',host='host-a',peer='peer-a')
            p=subprocess.Popen([sys.executable,str(script),str(root),
                   'hold-after-calculate','--host','host-a','--peer','peer-a',
                   '--claim-id',assigned['claim_id']],
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                signal=p.stdout.readline().strip()
                self.assertTrue(json.loads(signal)['candidate_digest'])
            finally:
                p.kill();p.wait(timeout=10)
            moved=cmd(root,'reassign',claim_id=assigned['claim_id'],
                      host='host-b',peer='peer-b')
            self.assertEqual(moved['epoch'],2)
            old=subprocess.run([sys.executable,str(script),str(root),'publish'],
                    capture_output=True,text=True,timeout=30)
            self.assertNotEqual(old.returncode,0)
            self.assertFalse(cmd(root,'status')['complete'])
            cmd(root,'calculate',claim_id=assigned['claim_id'],host='host-b',peer='peer-b')
            done=cmd(root,'publish')
            self.assertEqual(done['stage'],0)
            self.assertEqual(fixture(root).run.receipts[0]['host'],'host-b')


if __name__=='__main__':
    unittest.main()
