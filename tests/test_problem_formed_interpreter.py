"""PFI-001: problem-derived interpreter, source proofs, adaptive epochs, E-CPU DAG."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vli.problem_formed_interpreter import (
    _hash, _network, _symbolic_network, certify_succession,
    construct, create_ecpu_dag, execute_network, realize, run_one_ecpu_frontier,
    verify_interpreter, verify_realization, verify_succession,
)
from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from mailcpu.computation_dag import NavigableComputationDAG, ComputationDAGError


def problem(width=128, *, outputs=(1,), change_law=False):
    """No supplied ISA, quotient, processor registers or output microcode."""
    rows1 = [1 << i for i in range(width)]
    rows1[0] = 3                    # x0' = x0 XOR x1 XOR 1
    rows2 = [1 << i for i in range(width)]
    rows2[0], rows2[1] = 2, 1      # swap the two coupled bits
    if change_law:
        rows1[2] ^= 1 << 4
    return SymbolicSource(width, (
        AffineMap(tuple(rows1), 1), AffineMap(tuple(rows2), 0)),
        tuple(outputs), 'problem-specified-declared-dynamics')


class FormedInterpreterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = problem()
        cls.quotient, cls.bundle = construct(cls.source)

    def test_source_intention_constructs_small_distinct_interpreter(self):
        self.assertEqual(self.source.width, 128)
        self.assertEqual(self.quotient.dimension, 2)
        self.assertEqual(self.bundle['register_bits'], 2)
        self.assertEqual(len(self.bundle['transitions']), 2)
        self.assertTrue(verify_interpreter(self.source, self.quotient, self.bundle))
        self.assertTrue(self.bundle['primitive_set'])
        self.assertLess(sum(len(x['nodes']) for x in self.bundle['transitions']), 10)

    def test_adversarial_broken_microcode_rehashed_still_refused(self):
        broken = copy.deepcopy(self.bundle)
        program = broken['transitions'][0]
        program['outputs'][0] = 'zero'
        broken['digest'] = _hash({k:v for k,v in broken.items() if k != 'digest'})
        self.assertFalse(verify_interpreter(self.source, self.quotient, broken))

    def test_injected_op_rejected_even_with_matching_digest(self):
        broken = copy.deepcopy(self.bundle)
        circuit = broken['transitions'][0]
        circuit['nodes'].append({'id': 'g' + str(len(circuit['nodes'])),
                                 'op': 'privileged-io', 'args':['x0']})
        circuit['primitive_set'].append('privileged-io')
        broken['primitive_set'].append('privileged-io')
        broken['digest'] = _hash({k:v for k,v in broken.items() if k != 'digest'})
        self.assertFalse(verify_interpreter(self.source, self.quotient, broken))

    def test_source_and_generated_machine_small_words_agree(self):
        for initial in (0, 1, 2, 5, 6, 128, (1 << 127) | 5):
            for word in ((), (0,), (1,), (0, 1), (1, 0, 0)):
                for repeats in (0, 1, 2, 7):
                    state = initial
                    for _ in range(repeats):
                        for op in word:
                            state = self.source.generators[op].apply(state)
                    record = realize(self.source, self.quotient, self.bundle,
                                     initial, word, repeats)
                    self.assertEqual(int(record['observed_decimal']), self.source.observe(state))
                    self.assertTrue(verify_realization(self.source, self.quotient,
                                                       self.bundle, record))

    def test_large_time_uses_composed_ground_not_trajectory(self):
        result = realize(self.source, self.quotient, self.bundle, 3, (0,1), 10**18+137)
        self.assertEqual(result['strategy'], 'composed-native-ground-navigation')
        self.assertEqual(result['cost']['constructed_gate_steps'], 0)
        self.assertLess(result['cost']['ncr_operator_products'], 150)
        self.assertTrue(verify_realization(self.source, self.quotient,self.bundle,result))

    def test_intention_refinement_changes_processor_dimension_and_epoch(self):
        wider = problem(outputs=(1, 4))
        q2, b2 = construct(wider)
        self.assertEqual(q2.dimension, 3)
        self.assertNotEqual(self.bundle['digest'], b2['digest'])
        anchor = (1 << 94) | 5
        old_native = self.quotient.project(anchor)
        with self.assertRaisesRegex(ValueError, 'needs source-state anchor'):
            certify_succession(self.source,self.quotient,self.bundle,wider,q2,b2,
                              old_native_state=old_native)
        migration = certify_succession(self.source,self.quotient,self.bundle,
                                      wider,q2,b2,old_native_state=old_native,
                                      source_anchor=anchor)
        self.assertEqual(migration['transition'], 'declared-source-anchor-required')
        self.assertEqual(int(migration['new_native_decimal']), q2.project(anchor))
        self.assertTrue(verify_succession(self.source,self.quotient,self.bundle,wider,q2,b2,migration))
        forged = dict(migration, new_native_decimal='0')
        forged['digest'] = _hash({k:v for k,v in forged.items() if k != 'digest'})
        self.assertFalse(verify_succession(self.source,self.quotient,self.bundle,wider,q2,b2,forged))

    def test_intention_coarsening_derives_map_without_source_anchor(self):
        source2 = problem(outputs=(1, 4))
        q2,b2 = construct(source2)
        x = (1 << 100) | 7
        migration = certify_succession(source2,q2,b2,
                                      self.source,self.quotient,self.bundle,
                                      old_native_state=q2.project(x))
        self.assertEqual(migration['transition'], 'mathematically-derived-quotient-map')
        self.assertIsNone(migration['source_anchor_decimal'])
        self.assertEqual(int(migration['new_native_decimal']), self.quotient.project(x))
        self.assertTrue(verify_succession(source2,q2,b2,self.source,self.quotient,
                                         self.bundle,migration))

    def test_causal_continuation_after_mathematically_qualified_reformation(self):
        # Reach a genuine source state before changing the observer, then
        # continue the same generator law under the freshly constructed CPU.
        source_state = 0b10101101
        for _ in range(27):
            for action in (0, 1):
                source_state = self.source.generators[action].apply(source_state)
        old_native = self.quotient.project(source_state)
        changed = problem(outputs=(1, 4))
        new_q, new_cpu = construct(changed)
        certificate = certify_succession(
            self.source, self.quotient, self.bundle,
            changed, new_q, new_cpu,
            old_native_state=old_native, source_anchor=source_state)
        self.assertTrue(verify_succession(
            self.source,self.quotient,self.bundle,changed,new_q,new_cpu,certificate))
        resumed_source = source_state
        for _ in range(45):
            for action in (1, 0):
                resumed_source = changed.generators[action].apply(resumed_source)
        continuation = realize(changed,new_q,new_cpu,source_state,(1, 0),45)
        self.assertEqual(int(continuation['observed_decimal']),
                         changed.observe(resumed_source))
        self.assertEqual(int(certificate['new_native_decimal']),
                         new_q.project(source_state))

    def test_changed_dynamics_refuses_reusing_intention_epoch(self):
        source2 = problem(outputs=(1,), change_law=True)
        q2,b2 = construct(source2)
        with self.assertRaisesRegex(ValueError,'Different source-law epoch'):
            certify_succession(self.source,self.quotient,self.bundle,
                              source2,q2,b2,old_native_state=0,source_anchor=0)

    def test_minimal_quotient_refuses_budget_below_required_width(self):
        with self.assertRaises(ValueError):
            construct(self.source,max_dimension=1)

    def test_adversarial_shift_register_uses_all_source_bits(self):
        width = 24
        shifted = tuple(1 << (i+1) if i<width-1 else 1 << (width-1)
                        for i in range(width))
        source = SymbolicSource(width,(AffineMap(shifted,0),),(1,),
                                'long-shift-counterexample')
        with self.assertRaisesRegex(ValueError, 'exceeds declared dimension'):
            construct(source,max_dimension=8)
        quotient, bundle = construct(source,max_dimension=width)
        self.assertEqual(quotient.dimension,width)
        self.assertTrue(verify_interpreter(source, quotient, bundle))

    def test_native_ecpu_dag_checkpoint_and_host_resumption(self):
        dag = create_ecpu_dag(self.bundle,initial=7,word=(0,1),repetitions=100003)
        self.assertEqual(tuple(x.node_id for x in dag.ready_nodes()),('ground',))
        with self.assertRaises(ComputationDAGError):
            dag.commit('navigate',{'result':'invalid'})
        self.assertTrue(run_one_ecpu_frontier(self.source,self.quotient,self.bundle,dag))
        checkpoint = dag.snapshot()
        restored = NavigableComputationDAG.from_snapshot(checkpoint)
        self.assertEqual(tuple(x.node_id for x in restored.ready_nodes()), ('navigate',))
        self.assertTrue(run_one_ecpu_frontier(self.source,self.quotient,self.bundle,restored))
        second_checkpoint = restored.snapshot()
        again = NavigableComputationDAG.from_snapshot(second_checkpoint)
        self.assertTrue(run_one_ecpu_frontier(self.source,self.quotient,self.bundle,again))
        self.assertTrue(again.is_complete)
        out = again.result('observe')
        expected = realize(self.source,self.quotient,self.bundle,7,(0,1),100003)
        self.assertEqual(int(out['observed_decimal']),int(expected['observed_decimal']))
        self.assertEqual(out['receipt_digest'],expected['digest'])

    def test_retargeted_interpreter_cannot_resume_original_native_dag(self):
        dag = create_ecpu_dag(self.bundle,initial=1,word=(0,),repetitions=7)
        other = problem(outputs=(1,4)); quotient,bundle = construct(other)
        with self.assertRaisesRegex(ValueError,'Stale generated-processor identity'):
            run_one_ecpu_frontier(other, quotient, bundle, dag)

    def test_native_dag_checkpoint_tamper_rejected(self):
        dag = create_ecpu_dag(self.bundle,initial=2,word=(0,),repetitions=5)
        run_one_ecpu_frontier(self.source,self.quotient,self.bundle,dag)
        tampered = copy.deepcopy(dag.snapshot())
        tampered['state']['nodes'][0]['node_id'] = 'forged'
        with self.assertRaises((ValueError, ComputationDAGError, KeyError)):
            NavigableComputationDAG.from_snapshot(tampered)

    def test_three_independent_host_processes_continue_native_dag(self):
        script = Path(__file__).resolve().parents[1] / 'examples' / 'pfi_host_rehome.py'
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.json'
            first = None
            for phase in range(3):
                run = subprocess.run([sys.executable, str(script), str(checkpoint), str(phase)],
                                     check=True, capture_output=True, text=True)
                report = json.loads(run.stdout)
                self.assertEqual(report['stage'], phase)
                if first is None:
                    first = report['interpreter_digest']
                self.assertEqual(report['interpreter_digest'],first)
                self.assertEqual(report['complete'], phase == 2)
            self.assertIsNotNone(report['observed_decimal'])
            self.assertEqual(json.loads(checkpoint.read_text())['stage_completed'], 2)

    def test_circuit_symbolic_reference_without_live_backend(self):
        rows=(0b11,0b10,0b100)
        prog=_network(rows,1,3)
        self.assertEqual(_symbolic_network(prog),(rows,1))
        for z in range(8):
            expected = (((z&3).bit_count()&1) ^ 1) | (z & 0b110)
            self.assertEqual(execute_network(prog,z),expected)


if __name__ == '__main__':
    unittest.main()
