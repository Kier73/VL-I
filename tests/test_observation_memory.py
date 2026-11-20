"""SHAPE-004: source-identity-qualified observer-host structural-memory execution.

Tests actual IFS CO G-008 black-box learning over full source replay versus a
receiver-owned prefix state-trie; IFS learner is unchanged. Verify that
prefix memory only saves *pure original source transitions*, not semantic
evidence or independent model verification.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from vli.learned_morphology import LearnedObserverGround, BlackBoxObserver
from vli.observation_memory import StructuralPrefixMemory

RAW=(Path(__file__).resolve().parents[1]/
     "experiments/shape002/morphology_model.json")


class TestReceiverStructuralPrefixMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=RAW.read_bytes()

    def test_prefix_cache_exact_for_all_queries_vs_actual_uncached_source(self):
        memory=StructuralPrefixMemory(self.raw)
        uncached=BlackBoxObserver(
            memory.reference,channels=("phase",),
            source_digest=memory.source_sha256)
        cached=BlackBoxObserver(
            memory.reference,channels=("phase",),
            source_digest=memory.source_sha256,structural_memory=memory)
        words=(
            ("advance",),("advance","inspect"),
            ("advance","advance","inspect"),("advance","reset"),
            ("advance","advance","inspect"),("advance",),
            ("advance","advance","reset","advance"),
            ("advance","advance","inspect","advance","reset"),
        )
        for w in words:
            self.assertEqual(cached.query(w),uncached.query(w))
        self.assertGreater(uncached.direct_transition_steps,
                           cached.direct_transition_steps)
        self.assertEqual(memory.evaluated_transitions,
                         cached.direct_transition_steps)
        self.assertEqual(memory.naive_transition_steps,
                         uncached.direct_transition_steps)
        self.assertGreater(memory.reused_prefix_edges,0)

    def test_cross_observer_queries_reuse_hidden_source_state_without_leaking_channels(self):
        m=StructuralPrefixMemory(self.raw)
        phase=BlackBoxObserver(m.reference,channels=("phase",),
                               source_digest=m.source_sha256,structural_memory=m)
        micro=BlackBoxObserver(m.reference,channels=("micro",),
                               source_digest=m.source_sha256,structural_memory=m)
        path=("advance","advance","inspect")
        one=phase.query(path)
        before=m.evaluated_transitions
        two=micro.query(path)
        self.assertEqual(m.evaluated_transitions,before)
        self.assertEqual(phase.direct_transition_steps,len(path))
        self.assertEqual(micro.direct_transition_steps,0)
        self.assertEqual(one,((1,),(2,),(2,)))
        self.assertEqual(two,((1,),(2,),(2,)))
        # Distinct observer outputs from same cached hidden source sequence.
        other=("advance",)*4
        self.assertEqual(phase.query(other)[-1],(1,))
        self.assertEqual(micro.query(other)[-1],(0,))

    def test_learned_neural_ground_and_ledger_identical_to_uncached_baseline(self):
        raw=self.raw
        memory=StructuralPrefixMemory(raw)
        for scope in (("phase",),("micro",),("phase","micro")):
            with self.subTest(observer=scope):
                full=LearnedObserverGround(raw,scope)
                cached=LearnedObserverGround(raw,scope,structural_memory=memory)
                self.assertEqual(full.machine.document(),cached.machine.document())
                self.assertEqual(full.verdict,cached.verdict)
                self.assertEqual(full.ifs_checkpoint,cached.ifs_checkpoint)
                self.assertEqual(full.manifest,cached.manifest)
                self.assertEqual(full.neural_ground.weights_digest,
                                 cached.neural_ground.weights_digest)
                self.assertLess(cached.source.direct_transition_steps,
                                full.source.direct_transition_steps)
                self.assertTrue(cached.check(
                    expected_certificate_digest=cached.certificate["proof_digest"]))

    def test_total_actual_source_work_is_lower_across_three_learned_shapes(self):
        raw=self.raw
        m=StructuralPrefixMemory(raw)
        source_ops_naive=0
        source_ops_cached=0
        for scope in (("phase",),("micro",),("phase","micro")):
            baseline=LearnedObserverGround(raw,scope)
            source_ops_naive+=baseline.source.direct_transition_steps
            qualified=LearnedObserverGround(raw,scope,structural_memory=m)
            source_ops_cached+=qualified.source.direct_transition_steps
        self.assertEqual(source_ops_cached,m.evaluated_transitions)
        self.assertLess(source_ops_cached,source_ops_naive)
        self.assertEqual(m.naive_transition_steps,source_ops_naive)
        self.assertEqual(m.reused_prefix_edges,
                         source_ops_naive-source_ops_cached)
        self.assertGreater(m.unique_nodes,1)

    def test_receiver_memory_checkpoint_restores_with_original_source_independent_check(self):
        m=StructuralPrefixMemory(self.raw)
        word=("advance","advance","inspect")
        direct=m.observe(word,("phase",))
        saved=m.checkpoint()
        copy_memory=StructuralPrefixMemory.restore(
            self.raw,saved,expected_head=saved["checkpoint_digest"])
        self.assertEqual(copy_memory.checkpoint(),saved)
        self.assertEqual(copy_memory.restoration_validation_steps,
                         len(saved["entries"])-1)
        self.assertEqual(copy_memory.observe(word,("phase",)),direct)
        self.assertEqual(copy_memory.evaluated_transitions,0)
        self.assertEqual(copy_memory.observe(word,("micro",))[-1],(2,))

    def test_wrong_external_head_and_forged_state_refuse(self):
        m=StructuralPrefixMemory(self.raw)
        m.observe(("advance","inspect"),("phase",))
        saved=m.checkpoint()
        with self.assertRaises(ValueError):
            StructuralPrefixMemory.restore(
                self.raw,saved,expected_head="0"*64)
        forgery=copy.deepcopy(saved)
        forgery["entries"][-1]["source_state"]="p0m0"
        with self.assertRaises(ValueError):
            StructuralPrefixMemory.restore(
                self.raw,forgery,expected_head=saved["checkpoint_digest"])

    def test_rehashed_wrong_state_even_with_matching_new_external_pin_refused(self):
        from vli.ground import _digest
        m=StructuralPrefixMemory(self.raw)
        m.observe(("advance","inspect"),("phase",))
        saved=m.checkpoint()
        forged=copy.deepcopy(saved)
        forged["entries"][-1]["output"]=[9,9]
        forged["checkpoint_digest"]=_digest(
            forged["schema"],
            {k:v for k,v in forged.items() if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            StructuralPrefixMemory.restore(
                self.raw,forged,
                expected_head=forged["checkpoint_digest"])

    def test_changed_source_epoch_cannot_reuse_cached_state(self):
        m=StructuralPrefixMemory(self.raw)
        saved=m.checkpoint()
        revised=json.loads(self.raw)
        revised["metadata"]["epoch"]="changed source"
        raw=json.dumps(revised,sort_keys=True).encode("utf-8")
        with self.assertRaises(ValueError):
            StructuralPrefixMemory.restore(
                raw,saved,expected_head=saved["checkpoint_digest"])
        from vld.signal.transitions import GroundTransitionSystem
        source=GroundTransitionSystem.from_dict(revised)
        with self.assertRaises(ValueError):
            BlackBoxObserver(
                source,channels=("phase",),
                source_digest=m.source_sha256,structural_memory=m)

    def test_unadmitted_actions_and_channels_refuse_before_cache_mutation(self):
        m=StructuralPrefixMemory(self.raw)
        original=m.unique_nodes
        with self.assertRaises(ValueError):
            m.observe(("invalid",),("phase",))
        with self.assertRaises(ValueError):
            m.observe(("advance",),("secret",))
        with self.assertRaises(ValueError):
            m.observe(("advance",)*65,("phase",))
        self.assertEqual(m.unique_nodes,original)


if __name__=="__main__":
    unittest.main()
