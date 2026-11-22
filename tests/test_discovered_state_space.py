"""TIME-008: authenticated unknown-order discovery, independent audit and refusal."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.ground import _digest, load_json
from vli.authenticated_temporal_source import (
    _genesis, public_key_hex, sign_source_observation,
    append_signed_observation, resume_signed_observer,
)
from vli.discovered_state_space import (
    SCHEMA, construct_discovered_space, verify_discovered_space,
    navigate_discovered, verify_discovered_navigation,
)


def sequence(seed, coefficients, length):
    values = list(seed)
    while len(values) < length:
        values.append(sum(coefficients[j] * values[-1-j]
                          for j in range(len(coefficients))))
    return values


class DiscoveredProcessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.key = Ed25519PrivateKey.generate()
        self.public = public_key_hex(self.key)
        self.source = "time008-signed-external-publisher"
        self.head = _genesis(self.source, self.public)
        self.count = 0
        self.journal = self.folder / "source.jsonl"
        self.checkpoint = self.folder / "observer.json"

    def observe(self, values, *, irregular_at=None):
        for value in values:
            time = Fraction(self.count + (1 if self.count == irregular_at else 0))
            packet = sign_source_observation(
                self.key, source=self.source, sequence=self.count,
                time=f"{time.numerator}/{time.denominator}",
                value=f"{value}/1")
            self.head, self.count = append_signed_observation(
                self.journal, packet, expected_head=self.head,
                expected_count=self.count, source=self.source,
                public_key=self.public)
        observer = resume_signed_observer(
            journal_path=self.journal, checkpoint_path=self.checkpoint,
            source=self.source, public_key=self.public,
            expected_source_head=self.head,
            expected_source_count=self.count,
            environment="time008-independent-environment",
            host="separate-signed-observer-process")
        snap = load_json(self.checkpoint)
        args = dict(
            journal_path=self.journal, source_id=self.source,
            public_key=self.public, expected_source_head=self.head,
            expected_source_count=self.count,
            expected_ground_head=observer["result_ground_head"],
            expected_ground_identity=observer["observer_identity"])
        return snap, args

    def make(self, values, *, train, validation=2, test=2, max_order=4):
        snap, args = self.observe(values)
        space = construct_discovered_space(
            snap, **args, training_count=train,
            validation_count=validation, test_count=test,
            max_order=max_order)
        self.assertTrue(verify_discovered_space(snap, space, **args))
        return snap, space, args

    def test_dimension_three_is_discovered_without_family_label(self):
        values = sequence([1, 2, 3], [1, 0, 1], 14)
        snap, space, args = self.make(values, train=10)
        self.assertEqual(space["dimension"], 3)
        self.assertEqual(space["coefficients"], ["1/1", "0/1", "1/1"])
        self.assertEqual(space["heldout_validation"], "EXACT_MATCH")
        self.assertEqual(space["untouched_test"], "EXACT_MATCH")
        self.assertEqual(space["source_interpreter_grammar_is_external_seed"], True)
        for position in (2, 5, 12, 13):
            result = navigate_discovered(space, f"{position}/1",
                                         reflect=position == 12, depth=0)
            self.assertEqual(result["value"], f"{values[position]}/1")
            self.assertTrue(verify_discovered_navigation(
                snap, space, result, **args))
            self.assertEqual(result["calculation"]["source_steps_replayed"], 0)
        historical = navigate_discovered(space, "13/1", output="lag:1")
        self.assertEqual(historical["value"], f"{values[12]}/1")
        self.assertTrue(verify_discovered_navigation(
            snap, space, historical, **args))
        self.assertLessEqual(navigate_discovered(
            space, "100/1", allow_projection=True)["calculation"]
            ["navigation_metrics"]["matrix_products"], 16)
        with self.assertRaises(ValueError):
            navigate_discovered(space, "100/1")
        with self.assertRaises(ValueError):
            navigate_discovered(space, "100/1", output="invent-actuation",
                                allow_projection=True)

    def test_dimension_four_and_independent_reflective_execution(self):
        values = sequence([1, 1, 2, 3], [1, 1, 0, 1], 14)
        snap, space, args = self.make(values, train=10)
        self.assertEqual(space["dimension"], 4)
        self.assertEqual(space["qualified_observation_alphabet"],
                         ["current", "lag:1", "lag:2", "lag:3"])
        result = navigate_discovered(space, "10/1", reflect=True, depth=0)
        self.assertEqual(result["value"], f"{values[10]}/1")
        self.assertEqual(result["calculation"]["source_reflection"]
                         ["reflection_receipts"][0]
                         ["host_and_independent_continuation_agree"], True)
        self.assertTrue(verify_discovered_navigation(
            snap, space, result, **args))

    def test_depth_two_self_interpretation_survives_new_discovery(self):
        values = sequence([2, 3], [1, 1], 12)
        snap, space, args = self.make(values, train=8)
        self.assertEqual(space["dimension"], 2)
        result = navigate_discovered(space, "9/1", reflect=True, depth=2)
        self.assertEqual(result["value"], "144/1")
        self.assertEqual([r["depth"] for r in result["calculation"]
                          ["source_reflection"]["reflection_receipts"]],
                         [0, 1, 2])
        self.assertTrue(verify_discovered_navigation(
            snap, space, result, **args))

    def test_insufficient_rank_overfit_and_unseen_future_are_refused(self):
        values = [0] * 12
        snap, args = self.observe(values)
        with self.assertRaisesRegex(ValueError, "identifiable"):
            construct_discovered_space(
                snap, **args, training_count=8,
                validation_count=2, test_count=2)
        # A non-recurrent polynomial source exceeds the fixed low-order budget.
        with self.assertRaises(ValueError):
            # Same signed stream is zero: refuse order=1 rather than infer anything.
            construct_discovered_space(
                snap, **args, training_count=8, validation_count=2,
                test_count=2, max_order=1)

    def test_true_chronological_holdout_rejects_accidental_fit(self):
        vals = sequence([1, 2, 3], [1, 0, 1], 12)
        vals[9] += 17  # authenticated but contradicts training-fitted law
        snap, args = self.observe(vals)
        with self.assertRaisesRegex(ValueError, "validation"):
            construct_discovered_space(
                snap, **args, training_count=8,
                validation_count=2, test_count=2, max_order=3)

    def test_independent_source_and_result_forgery_refusal(self):
        vals = sequence([2, 3], [1, 1], 12)
        snap, space, args = self.make(vals, train=8)
        forged = deepcopy(space)
        forged["coefficients"][0] = "2/1"
        forged["intention_space_digest"] = _digest(
            SCHEMA, {k:v for k,v in forged.items()
                     if k != "intention_space_digest"})
        with self.assertRaises(ValueError):
            verify_discovered_space(snap, forged, **args)
        bad = deepcopy(snap)
        bad["events"][4]["values"][0] = "999/1"
        with self.assertRaises(ValueError):
            verify_discovered_space(bad, space, **args)
        stale = dict(args, expected_source_head="0"*64)
        with self.assertRaises(ValueError):
            verify_discovered_space(snap, space, **stale)
        answer = navigate_discovered(space, "8/1")
        tampered = deepcopy(answer)
        tampered["value"] = "123/1"
        tampered["navigation_digest"] = _digest(
            tampered["schema"], {k:v for k,v in tampered.items()
                                 if k != "navigation_digest"})
        with self.assertRaises(ValueError):
            verify_discovered_navigation(snap, space, tampered, **args)

    def test_signed_source_revision_retains_old_epoch_but_no_future_equivalence(self):
        first = sequence([2, 3], [1, 1], 12)
        snap0, old, args0 = self.make(first, train=8)
        second = sequence([1, 2, 3], [1, 0, 1], 12)
        snap1, args1 = self.observe(second)
        fresh = construct_discovered_space(
            snap1, **args1, training_count=8,
            validation_count=2, test_count=2,
            epoch_start=12, previous_space=old)
        self.assertEqual(fresh["dimension"], 3)
        self.assertEqual(fresh["previous_space_digest"],
                         old["intention_space_digest"])
        self.assertEqual(fresh["source_witnesses"][:12],
                         old["source_witnesses"])
        self.assertTrue(verify_discovered_space(
            snap1, fresh, previous_space=old, **args1))
        self.assertEqual(navigate_discovered(fresh, "14/1")["value"], "3/1")
        self.assertEqual(navigate_discovered(old, "10/1")["value"],
                         f"{first[10]}/1")
        wrong_old = deepcopy(old)
        wrong_old["source_witnesses"][2] = "0"*64
        with self.assertRaises(ValueError):
            verify_discovered_space(snap1, fresh, previous_space=wrong_old,
                                    **args1)


if __name__ == "__main__":
    unittest.main()
