"""PBG-009: native VL-64 Qwen evidence + real E-CPU quorum claim authority.

Each E-CPU authority has an independent file journal. Tests establish live
Merkle ticket invalidation, old-host fencing, result recovery, and process race.
The native Qwen fixture contains synthetic F32 weights, NOT pretrained 27B.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

from mailcpu import (
    ExecutionClaimStatus, RecoveryLineageAuthority, RecoveryLineageCoordinator,
    SeedCommitMembership, TransferableExecutionClaimStore,
    TransferableExecutionClaimError,
)
from vli.quorum_neural_observation import (
    QuorumNeuralObservation, ObservationRefused,
    verified_observation_operator, semantic_identity,
)
from vli.active_neural_acquisition import select_native_run
from vli.durable_neural_acquisition import _canonical, _seal, _tape_record, _restore_tape, _path, _unpath
from vli.neural_trace_ground import WitnessedNeuralTraceEnsemble
from test_durable_neural_acquisition import source
from test_active_neural_acquisition_native import OPTIONS
from test_neural_trace_ground_native import fresh_chain

GENESIS = '0'*64
ORIGIN = ('peer-original','host-original')
RECOVERED = ('peer-replacement','host-replacement')
IMPL_A = 'a'*64
IMPL_B = 'b'*64
FAILURE = 'c'*64


def authority(root):
    membership=SeedCommitMembership('pbg009-neural-source-quorum',1,('A','B','C'),
                                    'A',1,GENESIS)
    peers={name:RecoveryLineageAuthority(membership,name,root/'authorities'/(name+'.log'))
           for name in membership.member_ids}
    coordinator=RecoveryLineageCoordinator(membership,peers)
    store=TransferableExecutionClaimStore(root/'claims',coordinator)
    return store,coordinator


def recovered_actor(root):
    store,coordinator=authority(root)
    return QuorumNeuralObservation(root/'observation',store,coordinator)


def archive_fixture(root,parent,candidates,plan):
    (root/'fixture.json').write_bytes(_canonical({
        'tapes':[_tape_record(t) for t in parent.tapes],
        'candidates':[_path(c) for c in candidates],
        'nonce':plan.nonce,
    }))


def load_fixture(root):
    raw=json.loads((root/'fixture.json').read_text())
    parent=WitnessedNeuralTraceEnsemble(tuple(_restore_tape(t) for t in raw['tapes']))
    candidates=tuple(_unpath(x) for x in raw['candidates'])
    plan=select_native_run(parent,candidates,nonce=raw['nonce'])
    return parent,candidates,plan


def competing_replacement(root,owner,barrier):
    from mailcpu.transferable_execution_claim import TransferableExecutionClaimError
    parent,candidates,plan=load_fixture(root)
    actor=recovered_actor(root)
    (root/('ready-'+owner)).write_text('ready')
    deadline=time.monotonic()+15
    while not barrier.exists():
        if time.monotonic()>deadline:
            raise AssertionError('Race barrier failed')
        time.sleep(.005)
    try:
        claimed=actor.reassign(parent,candidates,plan,expected_epoch=1,
                               peer=owner,host='host-'+owner,
                               implementation=IMPL_B,
                               failure_evidence_digest=FAILURE)
        accepted=actor.publish(parent,candidates,plan,
                               peer=owner,host='host-'+owner,implementation=IMPL_B)
        return {'status':'accepted', 'claim_id':accepted['claim_id'],
                'owner':accepted['owner_peer_id'],
                'epoch':claimed.claim.current_assignment.epoch,
                'new_neural_calls':accepted['new_neural_calls_during_restore']}
    except (TransferableExecutionClaimError, ObservationRefused) as exc:
        return {'status':'superseded','why':type(exc).__name__}


class ECPUQuorumNeuralRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=source()  # genuine trained two-layer F32 Qwen fixture

    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.store,self.coordinator=authority(self.root)
        self.actor=QuorumNeuralObservation(self.root/'observation',self.store,self.coordinator)
        self.plan=select_native_run(self.parent,OPTIONS,nonce='9'*64)
        self.bundle=self.actor.issue(self.parent,OPTIONS,self.plan,
                                     peer=ORIGIN[0],host=ORIGIN[1],implementation=IMPL_A)

    def prepare(self):
        calls=[]
        def factory():
            calls.append(1)
            return fresh_chain()
        prepared=self.actor.prepare(self.parent,OPTIONS,self.plan,factory,
                                    peer=ORIGIN[0],host=ORIGIN[1],implementation=IMPL_A)
        self.assertEqual(len(calls),1)
        return prepared

    def test_merkle_current_target_scope_and_immutable_semantic_identity(self):
        self.assertEqual(self.bundle.claim.current_assignment.epoch,1)
        self.assertEqual(self.bundle.claim.status,ExecutionClaimStatus.ASSIGNED)
        self.assertEqual(self.bundle.claim.computational_ground_fingerprint,self.parent.source_digest)
        self.assertEqual(self.bundle.claim.child_request_digest,
                         semantic_identity(self.parent,OPTIONS,self.plan)['child_request_digest'])
        self.assertEqual(self.bundle.claim.operator_spec.to_dict(),verified_observation_operator().to_dict())
        with self.assertRaises(TransferableExecutionClaimError):
            self.bundle.verify_for_target(self.coordinator,
                **semantic_identity(self.parent,OPTIONS,self.plan),
                operator_spec=verified_observation_operator(),
                owner_peer_id='wrong-peer',owner_host_id=ORIGIN[1],
                implementation_fingerprint=IMPL_A)
        with self.assertRaises((ObservationRefused,ValueError)):
            self.actor.prepare(self.parent,tuple(reversed(OPTIONS)),self.plan,
                               lambda:None,peer=ORIGIN[0],host=ORIGIN[1],implementation=IMPL_A)

    def test_original_host_lost_prepared_result_recovered_under_new_quorum_epoch(self):
        prepared=self.prepare()
        replacement=recovered_actor(self.root)  # brand-new coordinator and store objects
        moved=replacement.reassign(self.parent,OPTIONS,self.plan,expected_epoch=1,
                                   peer=RECOVERED[0],host=RECOVERED[1],
                                   implementation=IMPL_B,failure_evidence_digest=FAILURE)
        self.assertEqual(moved.claim.claim_id,self.bundle.claim.claim_id)
        self.assertEqual(moved.claim.current_assignment.epoch,2)
        with self.assertRaises(TransferableExecutionClaimError):
            self.bundle.verify_current(self.coordinator) # old proof at generation 1
        with self.assertRaises((ObservationRefused,TransferableExecutionClaimError)):
            self.actor.publish(self.parent,OPTIONS,self.plan,
                               peer=ORIGIN[0],host=ORIGIN[1],implementation=IMPL_A)
        child,restored=replacement.restore_prepared(self.parent,OPTIONS,self.plan)
        self.assertEqual(restored['digest'],prepared['digest'])
        self.assertEqual(len(child.tapes),len(self.parent.tapes)+1)
        result=replacement.publish(self.parent,OPTIONS,self.plan,
                                   peer=RECOVERED[0],host=RECOVERED[1],implementation=IMPL_B)
        self.assertEqual(result['status'],'QUORUM_NEURAL_OBSERVATION_ACCEPTED')
        self.assertEqual(result['assignment_epoch'],2)
        self.assertEqual(result['new_neural_calls_during_restore'],0)
        self.assertEqual(self.store.get(result['claim_id']).claim.status,ExecutionClaimStatus.COMPLETED)
        self.assertEqual(replacement.publish(self.parent,OPTIONS,self.plan,
                         peer=RECOVERED[0],host=RECOVERED[1],implementation=IMPL_B),result)
        with self.assertRaises(TransferableExecutionClaimError):
            replacement.reassign(self.parent,OPTIONS,self.plan,expected_epoch=2,
                                 peer='peer-c',host='host-c',implementation='e'*64,
                                 failure_evidence_digest='f'*64)

    def test_crash_after_quorum_completion_before_outbox_recovers_without_second_claim(self):
        self.prepare()
        import vli.quorum_neural_observation as mod
        original=mod._atomic_once
        def crash_on_outbox(path,record):
            if path.name=='accepted.json':raise RuntimeError('hard-crash-cut-after-quorum-commit')
            return original(path,record)
        mod._atomic_once=crash_on_outbox
        try:
            with self.assertRaisesRegex(RuntimeError,'hard-crash-cut'):
                self.actor.publish(self.parent,OPTIONS,self.plan,
                                   peer=ORIGIN[0],host=ORIGIN[1],implementation=IMPL_A)
        finally:
            mod._atomic_once=original
        claim=self.store.get(self.bundle.claim.claim_id)
        self.assertEqual(claim.claim.status,ExecutionClaimStatus.COMPLETED)
        replacement=recovered_actor(self.root)
        accepted=replacement.publish(self.parent,OPTIONS,self.plan,
                                     peer=ORIGIN[0],host=ORIGIN[1],implementation=IMPL_A)
        self.assertEqual(accepted['accepted_result_digest'],claim.claim.result_digest)
        self.assertEqual(accepted['completion_generation'],claim.claim.completion_generation)
        self.assertEqual(self.store.high_water[0],2) # issue plus completion only

    def test_tampered_prepared_source_cannot_commit_despite_valid_ownership(self):
        self.prepare()
        prepared=self.root/'observation'/'prepared-1.json'
        obj=json.loads(prepared.read_text())
        obj['tape']['labels'][0]='2'
        prepared.write_bytes(_canonical(_seal({k:v for k,v in obj.items() if k!='digest'})))
        with self.assertRaises((ObservationRefused,ValueError)):
            self.actor.publish(self.parent,OPTIONS,self.plan,
                               peer=ORIGIN[0],host=ORIGIN[1],implementation=IMPL_A)
        self.assertEqual(self.store.get(self.bundle.claim.claim_id).claim.status,
                         ExecutionClaimStatus.ASSIGNED)

    def test_reassign_before_any_native_result_can_run_once_on_replacement(self):
        replacement=recovered_actor(self.root)
        moved=replacement.reassign(self.parent,OPTIONS,self.plan,expected_epoch=1,
                                   peer=RECOVERED[0],host=RECOVERED[1],
                                   implementation=IMPL_B,failure_evidence_digest=FAILURE)
        calls=[]
        prepared=replacement.prepare(self.parent,OPTIONS,self.plan,
                            lambda:(calls.append('only-replacement') or fresh_chain()),
                            peer=RECOVERED[0],host=RECOVERED[1],implementation=IMPL_B)
        self.assertEqual(calls,['only-replacement'])
        self.assertEqual(prepared['epoch'],2)
        accepted=replacement.publish(self.parent,OPTIONS,self.plan,
                            peer=RECOVERED[0],host=RECOVERED[1],implementation=IMPL_B)
        self.assertEqual(accepted['assignment_epoch'],2)

    def test_two_processes_race_same_quorum_generation_one_winner(self):
        if os.name=='nt':
            self.skipTest('Quorum race runs in Linux job; Windows tests durable restart')
        self.prepare()
        archive_fixture(self.root,self.parent,OPTIONS,self.plan)
        barrier=self.root/'release'
        script='''import json,pathlib,sys\nfrom test_quorum_neural_observation import competing_replacement\nprint(json.dumps(competing_replacement(pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3]))),flush=True)\n'''
        processes=[]
        try:
            for owner in ('peer-replace-a','peer-replace-b'):
                processes.append(subprocess.Popen([sys.executable,'-c',script,
                      str(self.root),owner,str(barrier)],stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE,text=True,
                      env={**os.environ,'PYTHONPATH':os.pathsep.join(x for x in sys.path if x)}))
            deadline=time.monotonic()+20
            while not all((self.root/('ready-'+name)).exists() for name in
                          ('peer-replace-a','peer-replace-b')):
                if time.monotonic()>deadline:
                    raise AssertionError('No two ready replacement processes')
                time.sleep(.01)
            barrier.write_text('continue')
            observations=[]
            for proc in processes:
                out,err=proc.communicate(timeout=35)
                if proc.returncode:raise AssertionError('Replacement crashed: '+err)
                observations.append(json.loads(out))
            self.assertEqual(sum(x['status']=='accepted' for x in observations),1)
            self.assertEqual(sum(x['status']=='superseded' for x in observations),1)
            winner=next(x for x in observations if x['status']=='accepted')
            self.assertEqual(winner['new_neural_calls'],0)
            verifier=recovered_actor(self.root)
            accepted=verifier.verify_publication(self.parent,OPTIONS,self.plan)
            self.assertEqual(accepted['owner_peer_id'],winner['owner'])
            self.assertEqual(accepted['assignment_epoch'],2)
            self.assertEqual(self.store.high_water[0],3)  # issue + reassign + complete
        finally:
            barrier.write_text('continue')
            for proc in processes:
                if proc.poll() is None:proc.kill();proc.wait(timeout=5)

if __name__=='__main__':unittest.main()
