"""PBG-008 real IFS SQLite authority and source-witnessed PBG-007 recovery.

Requires the pinned IFS ProofActionRouteEpochLedger, E-CPU, VL-64 and PBG007.
Separate child interpreters race one durable SQLite owner generation.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

from vli.active_neural_acquisition import select_native_run
from vli.durable_neural_acquisition import (
    DurableNeuralObservation, ObservationRefused, _seal, _canonical,
    _tape_record, _restore_tape, _path, _unpath,
)
from vli.neural_trace_ground import WitnessedNeuralTraceEnsemble, capture_native_tape, NeuralObserver
from test_active_neural_acquisition_native import OPTIONS
from test_neural_trace_ground_native import PATHS, fresh_chain


def source():
    observer=NeuralObserver(0,0.)
    return WitnessedNeuralTraceEnsemble(tuple(capture_native_tape(fresh_chain(),p,observer)
                                      for p in PATHS[:2]))


def _archive_parent(folder,parent,candidates,plan):
    body={'tapes':[_tape_record(t) for t in parent.tapes],
          'catalogue':[_path(x) for x in candidates], 'nonce':plan.nonce}
    (folder/'fixture.json').write_bytes(_canonical(body))


# Run by a FRESH Python process with no new Qwen-layer execution.
def _child(folder,owner,barrier):
    from proof_action_route_epoch import ProofActionRouteEpochError
    fixture=json.loads((folder/'fixture.json').read_text())
    parent=WitnessedNeuralTraceEnsemble(tuple(_restore_tape(t) for t in fixture['tapes']))
    candidates=tuple(_unpath(x) for x in fixture['catalogue'])
    plan=select_native_run(parent,candidates,nonce=fixture['nonce'])
    actor=DurableNeuralObservation(folder)
    (folder/('ready-'+owner)).write_text('ready')
    while not barrier.exists():time.sleep(.005)
    try:
        ident=actor.claim(new_owner=owner,expected_owner='original-observer',expected_generation=1)
        published=actor.publish(parent,candidates,plan,
                                owner_id=ident.owner_id,generation=ident.generation)
        return {'status':'published','ticket':published['execution_ticket'],
                'owner':published['authority_owner'],
                'new_native_calls':published['native_calls_during_restore']}
    except (ProofActionRouteEpochError,ObservationRefused) as e:
        return {'status':'superseded','reason':type(e).__name__}


class DurableNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=source()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.ledger=DurableNeuralObservation(self.root)
        self.plan=select_native_run(self.parent,OPTIONS,nonce='8'*64)
        self.original=self.ledger.issue(self.parent,OPTIONS,self.plan,
                                        owner_id='original-observer')

    def _prepared(self):
        call=[]
        def factory():
            call.append(1)
            return fresh_chain()
        self.ledger.prepare(self.parent,OPTIONS,self.plan,factory,
                            owner_id='original-observer',generation=1)
        self.assertEqual(len(call),1)

    def test_prepared_result_survives_crash_and_rehomes_without_native_reexecution(self):
        self._prepared()
        replacement=DurableNeuralObservation(self.root)
        claimed=replacement.claim(new_owner='replacement-observer',
                                  expected_owner='original-observer',expected_generation=1)
        self.assertEqual(claimed.generation,2)
        with self.assertRaises(ObservationRefused):
            self.ledger.publish(self.parent,OPTIONS,self.plan,
                                owner_id='original-observer',generation=1)
        restored,prepared=replacement.restore_prepared(self.parent,OPTIONS,self.plan)
        self.assertEqual(len(restored.tapes),len(self.parent.tapes)+1)
        self.assertEqual(prepared['child_witness'],restored.native_witness_digest)
        receipt=replacement.publish(self.parent,OPTIONS,self.plan,
                                     owner_id='replacement-observer',generation=2)
        self.assertEqual(receipt['status'],'RECOVERED_OBSERVATION_PUBLISHED')
        self.assertEqual(receipt['native_calls_during_restore'],0)
        self.assertEqual(receipt['owner_generation'],2)
        self.assertEqual(len([x for x in replacement._ledger().entries
                 if x['event_type']=='execution_authorized']),1)
        self.assertEqual(replacement.verify_publication(self.parent,OPTIONS,self.plan),receipt)
        self.assertEqual(replacement.publish(self.parent,OPTIONS,self.plan,
                       owner_id='replacement-observer',generation=2),receipt)

    def test_changed_source_or_catalogue_is_refused_after_recovery(self):
        self._prepared()
        with self.assertRaises((ObservationRefused,ValueError)):
            self.ledger.restore_prepared(self.parent,tuple(reversed(OPTIONS)),self.plan)
        with self.assertRaises((ObservationRefused,ValueError)):
            self.ledger.restore_prepared(self.parent,OPTIONS,
                                         replace(self.plan,selected_digest='0'*64))
        record=json.loads((self.root/'prepared.json').read_text())
        record['tape']['labels'][0]='2'
        (self.root/'prepared.json').write_bytes(_canonical(_seal({k:v for k,v in record.items() if k!='digest'})))
        with self.assertRaises((ObservationRefused,ValueError)):
            self.ledger.restore_prepared(self.parent,OPTIONS,self.plan)

    def test_two_independent_replacement_processes_race_cas(self):
        if os.name=='nt':
            self.skipTest('Separate-process race covered by Linux CI; SQLite WAL Windows restart is tested separately')
        self._prepared()
        _archive_parent(self.root,self.parent,OPTIONS,self.plan)
        barrier=self.root/'release'
        code='''import json, pathlib,sys
from test_durable_neural_acquisition import _child
result=_child(pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3]))
print(json.dumps(result),flush=True)
'''
        ps=[]
        for name in ['replacement-a','replacement-b']:
            ps.append(subprocess.Popen([sys.executable,'-c',code,str(self.root),name,str(barrier)],
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                   env={**os.environ,'PYTHONPATH':os.pathsep.join(x for x in sys.path if x)}))
        deadline=time.monotonic()+15
        try:
            while not all((self.root/('ready-'+name)).exists()
                           for name in ('replacement-a','replacement-b')):
                if time.monotonic()>deadline:
                    raise AssertionError('Independent contenders did not reach barrier')
                time.sleep(.01)
            barrier.write_text('go')
            results=[]
            for p in ps:
                stdout,stderr=p.communicate(timeout=30)
                if p.returncode!=0:raise AssertionError('child failure: '+stderr)
                results.append(json.loads(stdout))
            self.assertEqual([r['status'] for r in results].count('published'),1)
            self.assertEqual([r['status'] for r in results].count('superseded'),1)
            self.assertEqual(results[0].get('new_native_calls',0),0)
            verified=self.ledger.verify_publication(self.parent,OPTIONS,self.plan)
            self.assertIn(verified['authority_owner'],('replacement-a','replacement-b'))
            self.assertEqual(len([x for x in self.ledger._ledger().entries
                             if x['event_type']=='execution_authorized']),1)
        finally:
            barrier.write_text('go')
            for p in ps:
                if p.poll() is None:p.kill();p.wait(timeout=5)

    def test_result_before_publication_refuses_authority_forgery(self):
        self._prepared()
        with self.assertRaises(ObservationRefused):
            self.ledger.verify_publication(self.parent,OPTIONS,self.plan)
        with self.assertRaises(ObservationRefused):
            self.ledger.publish(self.parent,OPTIONS,self.plan,
                                owner_id='replacement-pretender',generation=2)
        # An owner claim after the old owner authorized publication does not
        # invent a second logical ticket: authority event remains one.
        first=self.ledger.publish(self.parent,OPTIONS,self.plan,
                                  owner_id='original-observer',generation=1)
        self.assertEqual(first['owner_generation'],1)
        self.ledger.claim(new_owner='replacement-after-commit',
                          expected_owner='original-observer',expected_generation=1)
        second=self.ledger.publish(self.parent,OPTIONS,self.plan,
                          owner_id='replacement-after-commit',generation=2)
        self.assertEqual(first['execution_ticket'],second['execution_ticket'])
        self.assertEqual(first['publication_digest'],second['publication_digest'])
        self.assertEqual(second['owner_generation'],1)

if __name__=='__main__':
    unittest.main()
