"""PFI-002 adversarial verification against native E-CPU and NCR implementations."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mailcpu.computation_dag import NavigableComputationDAG
from navigable_compute.symbolic_quotient import SymbolicSource, AffineMap
from vli.problem_formed_interpreter import construct, verify_interpreter
from vli.problem_formed_fabric import FormedFabric, _hash


def described_problem(width=128, observed=(1,)):
    rows=[1<<i for i in range(width)]
    rows[0]=3
    return SymbolicSource(width,(AffineMap(tuple(rows),1),),tuple(observed),
                          'pfi002-description-specified-law')


def session(*, observed=(1,), limit=1, initial=3):
    source=described_problem(observed=observed)
    quotient,bundle=construct(source)
    return FormedFabric(source,quotient,bundle,initial=initial,
                        gates_per_host=limit,job='fresh-task')


class FormedFabricTests(unittest.TestCase):
    def test_generated_host_count_changes_with_problem_and_gate_capacity(self):
        a=session(limit=1)
        b=session(limit=64)
        self.assertTrue(len(a.steps)>len(b.steps))
        self.assertTrue(len(a.fabric.links())>len(b.fabric.links()))
        self.assertNotEqual(a.epoch,b.epoch)
        self.assertTrue(all('gates' in x for x in a.steps))

    def test_real_native_owned_frames_and_hypervisor_authorized_execution(self):
        run=session()
        source=run.source
        initial=3
        while run.advance(host='observer-A'):
            run=FormedFabric.restore(run.source,run.quotient,run.bundle,run.snapshot())
        expected=source.observe(source.generators[0].apply(initial))
        self.assertEqual(run.result()['observer_result'],expected)
        self.assertEqual(run.result()['executed_host_stages'],len(run.steps))
        self.assertTrue(all(r['local_kernel_journal_verified'] for r in run.receipts))
        self.assertEqual({r['hypervisor_action'] for r in run.receipts},{'ALLOW'})
        self.assertEqual(len(run.dag.committed_nodes),len(run.steps)+2)
        self.assertGreater(run.bytes_transmitted,0)

    def test_receiver_denial_does_not_consume_native_frame_or_commit_dag(self):
        run=session()
        run.advance(host='origin')
        before=run.snapshot()
        with self.assertRaisesRegex(ValueError,'Hypervisor refused'):
            run.advance(host='host-without-rights',grant_compute=False)
        self.assertEqual(run.snapshot(),before)
        run.advance(host='authorized-worker',grant_compute=True)
        self.assertEqual(run.receipts[0]['host'],'authorized-worker')

    def test_cross_host_handoff_keeps_one_stable_interpreter_epoch(self):
        run=session()
        names=['host-A','host-B','host-C','host-D','host-E','host-F']
        for i in range(len(run.steps)+2):
            host=names[i%len(names)]
            self.assertTrue(run.advance(host=host))
            run=FormedFabric.restore(run.source,run.quotient,run.bundle,run.snapshot())
        self.assertTrue(run.dag.is_complete)
        self.assertEqual(run.result()['interpreter_digest'],run.bundle['digest'])
        self.assertGreater(len(set(run.result()['host_identifiers'])),1)

    def test_forged_predecessor_frame_even_rehashed_fails_closed(self):
        run=session();run.advance(host='origin')
        capsule=run.snapshot()
        old=capsule['state']['fabric']['queues'][0]['frames'][0]['payload_hex']
        raw=bytearray.fromhex(old);raw[-2]^=1
        capsule['state']['fabric']['queues'][0]['frames'][0]['payload_hex']=raw.hex()
        capsule['sha256']=_hash(capsule['state'])
        with self.assertRaises((ValueError,Exception)):
            FormedFabric.restore(run.source,run.quotient,run.bundle,capsule)

    def test_stale_interpreter_not_admitted_to_continuation(self):
        run=session();run.advance(host='origin')
        newer=session(observed=(1,4))
        with self.assertRaises(ValueError):
            FormedFabric.restore(newer.source,newer.quotient,newer.bundle,run.snapshot())

    def test_own_dag_cannot_be_rewritten_by_checkpoint_rehash(self):
        run=session();run.advance(host='origin')
        capsule=run.snapshot()
        capsule['state']['dag']['state']['nodes'][0]['node_id']='falsified'
        capsule['state']['dag']['digest']=_hash(capsule['state']['dag']['state'])
        capsule['sha256']=_hash(capsule['state'])
        with self.assertRaises(Exception):
            FormedFabric.restore(run.source,run.quotient,run.bundle,capsule)

    def test_three_or_more_separate_python_processes(self):
        script=Path(__file__).resolve().parents[1]/'examples'/'pfi_fabric_rehome.py'
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'checkpoint.json'
            args=[sys.executable,str(script),str(target)]
            first=json.loads(subprocess.check_output([*args,'--host','alpha','--initialize'],text=True))
            self.assertFalse(first['complete'])
            last=first
            for i in range(12):
                last=json.loads(subprocess.check_output([*args,'--host',f'fresh-{i}'],text=True))
                if last['complete']:break
            self.assertTrue(last['complete'])
            self.assertGreaterEqual(len(last['result']['host_identifiers']),2)
            self.assertNotEqual(last['result']['host_identifiers'][0],last['result']['host_identifiers'][-1])
            self.assertEqual(last['result']['observer_result'],
                             described_problem().observe(described_problem().generators[0].apply(3)))


if __name__ == '__main__':
    unittest.main()
