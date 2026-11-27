"""PBG-006: actual native two-layer Qwen causal reports -> exact trace law.

Requires the existing VL-64 trained-operator fixture, E-CPU receiver fabric,
and VGI strict quorum. It does not load an original pretrained 27B checkpoint.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
import unittest

from test_qwen_layer_chain import two_layer_fixture, executor
from vli.qwen_all_heads import QwenVectorizedCompleteLayer
from vli.qwen_layer_chain import QwenCausalGDNChain
from vli.neural_trace_ground import (
    NeuralObserver, capture_native_tape, reframe_native_tape,
    WitnessedNeuralTraceEnsemble, ReceiverQualifiedNeuralReader, ALPHABET,
)
from vli.source_hankel_ground import (
    QueryAtlas, HankelLearner, source_trace_probability,
)
from vli.source_hankel_fabric import NativeSourceReader
from test_source_hankel_fabric import PEERS, admitted_sources


PATHS=(
    ((63204,(.5,-.2,.3)),(628,(-.1,.2,.8)),(1761,(.7,.6,-.3))),
    ((1761,(.7,.6,-.3)),(33303,(.3,-.4,.65))),
    ((628,(-.1,.2,.8)),(1761,(.7,.6,-.3)),
     (33303,(.3,-.4,.65)),(423,(-.4,.5,-.2))),
)


def fresh_chain():
    return QwenCausalGDNChain(
        executor(two_layer_fixture(),QwenVectorizedCompleteLayer))


def discover(exp, reader=None):
    atlas=QueryAtlas(ALPHABET,exp.source_digest,
                     exp.read if reader is None else reader,max_queries=12000)
    learner=HankelLearner(atlas)
    start=learner.build()
    states=[start]
    for _ in range(20):
        proof=exp.certify(states[-1])
        if proof['status']=='EQUIVALENT_ALL_WORDS':
            return states,atlas
        states.append(learner.counterexample(proof['word']))
    raise AssertionError('No exact source-qualified trace law after 20 revisions')


class NativeCausalTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observer=NeuralObserver(0,0.)
        cls.tapes=tuple(capture_native_tape(fresh_chain(),path,cls.observer)
                        for path in PATHS)
        cls.exp=WitnessedNeuralTraceEnsemble(cls.tapes)

    def test_genuine_two_layer_state_handoff_and_native_reexecution(self):
        self.assertEqual(len(self.exp.tapes),3)
        self.assertEqual(sum(t.length for t in self.exp.tapes),9)
        for tape in self.tapes:
            self.assertTrue(tape.verify())
            self.assertEqual(tape.labels,tuple(tape.labels))
            for i,report in enumerate(tape.reports):
                self.assertEqual(report['position'],i)
                self.assertTrue(report['all_layers_completed'])
                first,second=report['layer_witnesses']
                self.assertEqual(first['output_f64_digest'],second['input_f64_digest'])
                self.assertFalse(report['vocabulary_logits_computed'])
        self.assertTrue(self.exp.independently_reexecute(fresh_chain))

    def test_actual_neural_output_creates_exact_source_discovered_by_hankel(self):
        series,atlas=discover(self.exp)
        self.assertGreater(len(series),1)
        learned=series[-1]
        self.assertGreater(learned.rank,series[0].rank)
        self.assertEqual(self.exp.certify(learned)['status'],'EQUIVALENT_ALL_WORDS')
        self.assertEqual(learned.source_digest,self.exp.source_digest)
        self.assertGreater(len(atlas.cache),20)
        # Held-out combinations not used as original Hankel pivot words.
        for length in range(5):
            for word in product(ALPHABET,repeat=length):
                self.assertEqual(learned.probability(word),self.exp.probability(word))
        self.assertEqual(self.exp.probability(('act:step',)*4+('sense:readout:2',)),1)

    def test_new_observer_reuses_actual_native_causal_bytes_without_reexecution(self):
        values=[float(r['receiver_local_intermediates'][1][0])
                for tape in self.tapes for r in tape.reports]
        self.assertTrue(all(map(lambda x: abs(x)<1e10,values)))
        # Choose a different coordinate cut that necessarily changes a symbol.
        high=max(values)+1.
        low=min(values)-1.
        previous=set(label for tape in self.tapes for label in tape.labels)
        threshold=high if '1' in previous else low
        reframed=tuple(reframe_native_tape(t,NeuralObserver(0,threshold))
                       for t in self.tapes)
        new=WitnessedNeuralTraceEnsemble(reframed)
        self.assertNotEqual(self.exp.native_witness_digest,new.native_witness_digest)
        self.assertNotEqual(self.exp.source_digest,new.source_digest)
        for old,after in zip(self.tapes,reframed):
            self.assertEqual(old.reports,after.reports)
            self.assertEqual(old.inputs,after.inputs)
        self.assertEqual(new.probability(()),1)
        self.assertEqual(new.probability(('act:step',)*4+('sense:readout:2',)),1)

    def test_rehashed_forged_native_receipt_and_wrong_genesis_refuse(self):
        tape=self.tapes[0]
        forged=deepcopy(tape.reports[0])
        forged['receiver_local_intermediates'][0]=(999.,*forged['receiver_local_intermediates'][0][1:])
        broken=replace(tape,reports=(forged,*tape.reports[1:]),witness_digest='')
        broken=replace(broken,witness_digest=__import__('vli.source_hankel_ground',
                        fromlist=['digest']).digest(broken.material()))
        with self.assertRaises(ValueError):broken.verify()
        with self.assertRaises(ValueError):replace(tape,genesis='0'*64).verify()
        forged=replace(tape,labels=tuple('1' if x=='0' else '0' for x in tape.labels))
        with self.assertRaises(ValueError):forged.verify()

    def test_receiver_qualified_ecpu_vgi_reads_native_causal_source(self):
        native=NativeSourceReader(admitted_sources(self.exp.source),PEERS,
                                  self.exp.source_digest,ALPHABET,batch_size=32,
                                  nonce_factory=lambda:'a'*64)
        wrapped=ReceiverQualifiedNeuralReader(self.exp,native)
        queries=((),('sense:readout:0',),('sense:readout:1',),
                 ('act:step','sense:readout:0'),
                 ('act:step','act:step','sense:readout:2'))
        observations=wrapped(queries)
        self.assertEqual(tuple(x.probability for x in observations),
                         tuple(self.exp.probability(w) for w in queries))
        self.assertTrue(all(x.receipt_digest!=x.source_digest for x in observations))
        self.assertEqual(native.lineage[-1]['quorum'],'AUTHORIZED')
        self.assertEqual(native.lineage[-1]['source_ids'],('host-A','host-B'))
        # Reuse same receiver-quorum reader to derive an actual learned Ground.
        _,atlas=discover(self.exp,wrapped)
        self.assertGreater(len(atlas.cache),len(queries))

    def test_disagreeing_receiver_hosts_cannot_publish_neural_trace(self):
        readings=[float(r['receiver_local_intermediates'][1][0])
                  for t in self.tapes for r in t.reports]
        prior={label for tape in self.tapes for label in tape.labels}
        other_threshold=(max(readings)+1.) if '1' in prior else (min(readings)-1.)
        altered=tuple(reframe_native_tape(t,NeuralObserver(0,other_threshold))
                      for t in self.tapes)
        alternate=WitnessedNeuralTraceEnsemble(altered)
        # Inject a second hosted source with a different, individually valid
        # trace law. E-CPU still accepts receiver-local handler validity, but
        # VGI strict quorum rejects the different complete readout.
        native=NativeSourceReader(admitted_sources(self.exp.source,
                                  wrong_source=alternate.source),PEERS,
                                  self.exp.source_digest,ALPHABET,
                                  nonce_factory=lambda:'b'*64)
        wrapped=ReceiverQualifiedNeuralReader(self.exp,native)
        for event in ('sense:readout:0','sense:readout:1'):
            if self.exp.probability((event,))!=alternate.probability((event,)):
                with self.assertRaisesRegex(ValueError,'VGI.*CONFLICT'):
                    wrapped(((event,),))
                break
        else:
            self.fail('Test source never produced a distinct native readout')


if __name__=='__main__': unittest.main()
