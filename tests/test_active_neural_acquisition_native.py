"""PBG-007 genuine Qwen-shaped causal native acquisition with PBG-005/006.

Runs the actual VL-64 trained two-layer F32 *fixture*, never the full
pretrained Qwen 27B checkpoint. Receiver-local E-CPU/VGI fixtures supply
strict source quorum for the subsequently selected trace query.
"""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
import unittest

from vli.active_neural_acquisition import (
    Candidate, ActiveTraceObserver, distinguish, select_native_run,
    acquire_native_run, select_observer_attention, apply_observer_attention,
)
from vli.neural_trace_ground import (
    NeuralObserver, capture_native_tape, WitnessedNeuralTraceEnsemble,
    ReceiverQualifiedNeuralReader, ALPHABET,
)
from vli.source_hankel_ground import HankelLearner, QueryAtlas
from vli.source_hankel_fabric import NativeSourceReader
from test_neural_trace_ground_native import PATHS, fresh_chain
from test_source_hankel_fabric import PEERS, admitted_sources

# Novel input paths are explicit declared hypotheticals, not tokenizer-derived.
OPTIONS=(
 ((903, (.51,-.22,.31)), (904, (.04,.08,.18))),
 ((905, (1.4,-.9,1.1)), (906, (-.7,.45,-.88))),
)


def learn_ensemble(source,reader=None):
    atlas=QueryAtlas(ALPHABET,source.source_digest,
                     source.read if reader is None else reader,max_queries=12000)
    learner=HankelLearner(atlas)
    g=learner.build()
    for i in range(20):
        proof=source.certify(g)
        if proof['status']=='EQUIVALENT_ALL_WORDS':
            return g,atlas
        g=learner.counterexample(proof['word'])
    raise AssertionError('Finite witnessed native tape law did not converge')


class NativeActiveAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observer=NeuralObserver(0,0.)
        cls.tapes=tuple(capture_native_tape(fresh_chain(),path,cls.observer)
                        for path in PATHS[:2])
        cls.original=WitnessedNeuralTraceEnsemble(cls.tapes)

    def test_intention_selects_real_native_observer_coordinate_without_new_execution(self):
        original=self.original
        plan=select_observer_attention(original,(0,))
        self.assertGreater(plan.distinction,0)
        self.assertGreaterEqual(plan.coordinate,0)
        self.assertLess(plan.position,min(t.length for t in original.tapes))
        reframed,receipt=apply_observer_attention(original,plan)
        self.assertEqual(receipt['status'],'OBSERVER_GROUND_REFRAMED')
        self.assertEqual(receipt['original_neural_trained_calls'],0)
        self.assertEqual(receipt['source_receipts_reused'],sum(t.length for t in original.tapes))
        self.assertEqual(tuple(t.reports for t in reframed.tapes),
                         tuple(t.reports for t in original.tapes))
        self.assertNotEqual(reframed.native_witness_digest,original.native_witness_digest)
        self.assertTrue(all(t.verify() for t in reframed.tapes))
        with self.assertRaisesRegex(ValueError,'Stale|modified'):
            apply_observer_attention(original,replace(plan,threshold=12345.))
        learned,_=learn_ensemble(reframed)
        self.assertEqual(reframed.certify(learned)['status'],'EQUIVALENT_ALL_WORDS')

    def test_select_one_new_qwen_causal_run_and_relearn_source_without_reexecuting_old(self):
        old=self.original
        before=old.native_witness_digest
        plan=select_native_run(old,OPTIONS,nonce='1'*64)
        calls=[]
        def factory():
            calls.append('one-fresh-native-source')
            return fresh_chain()
        self.assertEqual(len(calls),0)
        self.assertEqual(plan.cost,2)
        new,receipt=acquire_native_run(old,OPTIONS,plan,factory)
        self.assertEqual(len(calls),1)
        self.assertEqual(receipt['selected_native_transitions'],2)
        self.assertEqual(receipt['unselected_native_transitions'],0)
        self.assertEqual(receipt['native_calls_during_selection'],0)
        self.assertEqual(len(new.tapes),len(old.tapes)+1)
        self.assertEqual(new.tapes[:len(old.tapes)],old.tapes)
        self.assertEqual(before,old.native_witness_digest)
        self.assertNotEqual(new.native_witness_digest,before)
        self.assertNotEqual(new.source_digest,old.source_digest)
        self.assertTrue(new.tapes[-1].verify())
        learned,atlas=learn_ensemble(new)
        self.assertEqual(new.certify(learned)['status'],'EQUIVALENT_ALL_WORDS')
        self.assertGreater(len(atlas.cache),0)
        # The retained original tapes remain valid on the original source.
        self.assertTrue(all(t.verify() for t in old.tapes))

    def test_pinned_parent_stale_plan_and_unqualified_changed_chain_refused(self):
        plan=select_native_run(self.original,OPTIONS,nonce='2'*64)
        tried=[]
        def should_not_be_called():
            tried.append(1)
            return fresh_chain()
        forged=replace(plan,selected_digest='0'*64)
        with self.assertRaisesRegex(ValueError,'Stale|substituted'):
            acquire_native_run(self.original,OPTIONS,forged,should_not_be_called)
        self.assertEqual(tried,[])
        def changed_chain():
            wrong=fresh_chain()
            wrong.step([.2,.1,.3],token_id=1)
            return wrong
        with self.assertRaisesRegex(ValueError,'source differs'):
            acquire_native_run(self.original,OPTIONS,plan,changed_chain)

    def test_native_evidence_directed_hypothesis_probe_through_ecpu_and_vgi(self):
        original=self.original
        new,record=acquire_native_run(original,OPTIONS,
                    select_native_run(original,OPTIONS,nonce='3'*64),fresh_chain)
        ground_a,_=learn_ensemble(original)
        ground_b,_=learn_ensemble(new)
        difference=distinguish(ground_a,ground_b)
        self.assertEqual(difference['status'],'CANDIDATE_DISTINCTION')
        word=difference['word']
        # Candidate predictions are NOT source certifications. The current
        # experiment chooses an actual source read through PBG-005's receiver.
        native=NativeSourceReader(admitted_sources(new.source),PEERS,
                                  new.source_digest,ALPHABET,batch_size=32,
                                  nonce_factory=lambda:'a'*64)
        wrapped=ReceiverQualifiedNeuralReader(new,native)
        atlas=QueryAtlas(ALPHABET,new.source_digest,wrapped,max_queries=12000)
        learner=HankelLearner(atlas)
        controller=ActiveTraceObserver(atlas,learner,
            (Candidate('old-neural-traces',ground_a,Q(1)),
             Candidate('new-neural-traces',ground_b,Q(1))))
        plan=controller.propose(nonce='4'*64)
        self.assertEqual(plan.word,word)
        self.assertEqual(native.lineage,[])
        result=controller.execute(plan)
        self.assertEqual(result['observed'],new.probability(word))
        self.assertEqual(result['matching_forecasts'],('new-neural-traces',))
        self.assertEqual(native.lineage[0]['quorum'],'AUTHORIZED')
        self.assertEqual(native.lineage[0]['source_ids'],('host-A','host-B'))
        self.assertNotEqual(result['source_receipt_digest'],new.source_digest)
        self.assertEqual(result['native_trained_calls_during_source_trace_query'],0)
        # The full current source is still independently available for audit.
        g=learner.ground
        for _ in range(20):
            proof=new.certify(g)
            if proof['status']=='EQUIVALENT_ALL_WORDS':break
            g=learner.counterexample(proof['word'])
        else: self.fail('No source-derived verified neural-tape Ground')
        self.assertEqual(new.certify(g)['status'],'EQUIVALENT_ALL_WORDS')


if __name__=='__main__':unittest.main()
