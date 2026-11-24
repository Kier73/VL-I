"""Actual VL-64 synthetic-weight native causal receipts + CIG-style sufficiency.

This does not execute the full original 27B pretrained model. The test runs a
real qualified two-layer F32 fixture and selects readouts from its exact F64
intermediate receiver outputs, never from fabricated future neural states.
"""
from copy import deepcopy
from dataclasses import replace
import unittest

from vli.observer_sufficiency import (
    SourceBit, from_neural, compile_for_neural, verify_for_neural,
    export_cig017_source, suggest_next_action,
)
from vli.neural_trace_ground import (NeuralObserver, capture_native_tape,
                                     WitnessedNeuralTraceEnsemble,
                                     reframe_native_tape)
from test_neural_trace_ground_native import PATHS, fresh_chain


class NativeSufficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tapes=tuple(capture_native_tape(fresh_chain(), path, NeuralObserver(0,0.))
                        for path in PATHS)
        cls.ensemble=WitnessedNeuralTraceEnsemble(cls.tapes)
        cls.width=len(cls.tapes[0].inputs[0][1])

    def goal(self):
        # Use only original witness values from the last common causal step.
        for c in range(self.width):
            vals=sorted(set(float(t.reports[1]['receiver_local_intermediates'][1][c])
                            for t in self.ensemble.tapes))
            if len(vals)>1:
                return SourceBit(1,c,vals[1])
        raise AssertionError('Actual fixture had no distinguishing source output')

    def test_native_goal_sufficiency_reuses_existing_f64_receipts(self):
        expected=self.ensemble.native_witness_digest
        table,proof=compile_for_neural(self.ensemble,self.goal(),
                          expected_witness=expected,coordinates=tuple(range(self.width)))
        audit=verify_for_neural(self.ensemble,self.goal(),proof,
                          expected_witness=expected,coordinates=tuple(range(self.width)))
        self.assertEqual(audit['status'],'INDEPENDENTLY_VERIFIED')
        self.assertEqual(audit['native_trained_calls'],0)
        self.assertEqual(table.source_digest,self.ensemble.source_digest)
        self.assertEqual(len(table.tape_witnesses),len(PATHS))
        self.assertEqual(proof['native_trained_calls'],0)
        self.assertIn(suggest_next_action(table,proof)['action'],(
          'REUSE_VERIFIED_GROUND','REFINE_FROM_RETAINED_RECEIPTS',
          'EXPAND_OBSERVER_CATALOGUE_OR_PBG009_SOURCE'))
        self.assertEqual(self.ensemble.native_witness_digest,expected)
        self.assertTrue(all(t.verify() for t in self.tapes))

    def test_budget_and_changed_source_are_refused_without_new_source_execution(self):
        original=self.ensemble
        table,proof=compile_for_neural(original,self.goal(),
            expected_witness=original.native_witness_digest,coordinates=(0,),max_steps=0)
        self.assertTrue(verify_for_neural(original,self.goal(),proof,
            expected_witness=original.native_witness_digest,coordinates=(0,),max_steps=0))
        with self.assertRaises(ValueError):
            verify_for_neural(original,self.goal(),proof,expected_witness='0'*64,
                              coordinates=(0,),max_steps=0)
        new=WitnessedNeuralTraceEnsemble(tuple(
            reframe_native_tape(t,NeuralObserver(1,0.)) for t in self.tapes))
        self.assertNotEqual(new.native_witness_digest,original.native_witness_digest)
        with self.assertRaises(ValueError):
            verify_for_neural(new,self.goal(),proof,
                   expected_witness=original.native_witness_digest,coordinates=(0,),max_steps=0)

    def test_native_f64_tamper_refused_before_observer_language_admission(self):
        report=deepcopy(self.tapes[0].reports[0]);raw=list(report['receiver_local_intermediates'][1])
        raw[0]+=1.0;report['receiver_local_intermediates'][1]=tuple(raw)
        forged=replace(self.tapes[0],reports=(report,*self.tapes[0].reports[1:]))
        with self.assertRaises(ValueError):
            WitnessedNeuralTraceEnsemble((forged,*self.tapes[1:]))

    def test_export_source_does_not_include_unseen_neural_transitions(self):
        table=from_neural(self.ensemble,self.goal(),coordinates=tuple(range(self.width)))
        n,source,q,f=export_cig017_source(table,tuple(range(min(2,len(table.features)))))
        self.assertEqual(n,len(table.targets)+len(q)+1)
        self.assertEqual(len(source),1+len(table.targets)*(len(table.targets)-1)//2+
                         len(table.targets)*(len(q)+1))
        self.assertTrue(all(1<=abs(l)<=n for clause in source for l in clause))


if __name__=='__main__':
    unittest.main()
