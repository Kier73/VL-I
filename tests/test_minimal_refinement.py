"""TDL-002: exact minimal refinement witnesses, independent source-row audit."""
from __future__ import annotations

import copy
import unittest

from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource

from vli.problem_formed_interpreter import construct, _hash, certify_succession
from vli.minimal_refinement import (
    plan_minimal_refinement, witness_from_source, certify_minimal_refinement,
)
from vli.minimal_refinement_verify import (
    verify_refinement_plan, verify_refinement_receipt,
)


def source_for(outputs=(1,), width=128, *, different_law=False):
    rows1 = [1 << i for i in range(width)]
    rows1[0] = 3
    rows2 = [1 << i for i in range(width)]
    rows2[0], rows2[1] = 2, 1
    if different_law:
        rows1[2] ^= 1 << 4
    return SymbolicSource(
        width,
        (AffineMap(tuple(rows1), 1), AffineMap(tuple(rows2), 0)),
        tuple(outputs),
        "tdl002-stationary-law",
    )


def pair(old_outputs=(1,), new_outputs=(1, 4)):
    old_source = source_for(old_outputs)
    new_source = source_for(new_outputs)
    old_q, old_bundle = construct(old_source)
    new_q, new_bundle = construct(new_source)
    return old_source, old_q, old_bundle, new_source, new_q, new_bundle


class MinimalRefinementTests(unittest.TestCase):
    def test_one_auxiliary_bit_replaces_full_128_bit_anchor(self):
        old_source, old_q, old_bundle, new_source, new_q, new_bundle = pair()
        args = old_source, old_q, old_bundle, new_source, new_q, new_bundle
        plan = plan_minimal_refinement(*args)
        self.assertEqual(plan["source_width"], 128)
        self.assertEqual(plan["old_bits"], 2)
        self.assertEqual(plan["new_bits"], 3)
        self.assertEqual(plan["additional_bits"], 1)
        self.assertEqual(len(plan["witness_source_rows_decimal"]), 1)
        self.assertTrue(verify_refinement_plan(*args, plan))
        for x in (0, 1, 2, 5, 7, (1 << 94) | 5, (1 << 127) | 5):
            old_z = old_q.project(x)
            witness = witness_from_source(old_source, plan, x)
            receipt = certify_minimal_refinement(
                plan, old_native_state=old_z, witness_bits=witness
            )
            self.assertTrue(verify_refinement_receipt(*args, plan, receipt))
            self.assertEqual(int(receipt["new_native_decimal"]), new_q.project(x))
            self.assertEqual(int(receipt["witness_decimal"]), witness)
            self.assertNotIn("source_anchor_decimal", receipt)
        # The existing full-anchor path is retained for compatibility.
        with self.assertRaisesRegex(ValueError, "needs source-state anchor"):
            certify_succession(*args, old_native_state=old_q.project(5))

    def test_coarsening_uses_zero_auxiliary_bits(self):
        args = pair((1, 4), (1,))
        old_source, old_q, _, _, new_q, _ = args
        plan = plan_minimal_refinement(*args)
        self.assertEqual(plan["additional_bits"], 0)
        for x in (0, 1, 2, 3, 5, 6, (1 << 100) | 7):
            receipt = certify_minimal_refinement(
                plan, old_native_state=old_q.project(x)
            )
            self.assertEqual(receipt["witness_decimal"], "0")
            self.assertTrue(verify_refinement_receipt(*args, plan, receipt))
            self.assertEqual(int(receipt["new_native_decimal"]), new_q.project(x))
            self.assertEqual(witness_from_source(old_source, plan, x), 0)

    def test_incomparable_observation_spans_use_conditional_bits(self):
        args = pair((1,), (4,))
        plan = plan_minimal_refinement(*args)
        self.assertGreater(plan["additional_bits"], 0)
        old_source, old_q, _, _, new_q, _ = args
        for x in (0, 1, 2, 4, 5, 15, (1 << 126) | 7):
            witness = witness_from_source(old_source, plan, x)
            receipt = certify_minimal_refinement(
                plan, old_native_state=old_q.project(x), witness_bits=witness
            )
            self.assertTrue(verify_refinement_receipt(*args, plan, receipt))
            self.assertEqual(int(receipt["new_native_decimal"]), new_q.project(x))

    def test_exact_independent_row_audit_rejects_rehashed_forgery(self):
        args = pair()
        plan = plan_minimal_refinement(*args)
        forged = copy.deepcopy(plan)
        forged["new_from_old_and_witness_rows_decimal"][0] = "0"
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_refinement_plan(*args, forged))

        forged = copy.deepcopy(plan)
        forged["witness_source_rows_decimal"][0] = "0"
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_refinement_plan(*args, forged))

        forged = copy.deepcopy(plan)
        forged["additional_bits"] = 0
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_refinement_plan(*args, forged))

    def test_receipt_requires_actual_bit_domain_and_exact_mapping(self):
        args = pair()
        old_source, old_q, _, _, _, _ = args
        plan = plan_minimal_refinement(*args)
        with self.assertRaisesRegex(ValueError, "additional observer bits"):
            certify_minimal_refinement(plan, old_native_state=0)
        with self.assertRaises(ValueError):
            certify_minimal_refinement(plan, old_native_state=True, witness_bits=0)
        with self.assertRaises(ValueError):
            certify_minimal_refinement(plan, old_native_state=0, witness_bits=True)
        with self.assertRaises(ValueError):
            certify_minimal_refinement(plan, old_native_state=0, witness_bits=2)
        with self.assertRaises(ValueError):
            witness_from_source(old_source, plan, 1 << old_source.width)
        x = (1 << 94) | 5
        receipt = certify_minimal_refinement(
            plan, old_native_state=old_q.project(x),
            witness_bits=witness_from_source(old_source, plan, x)
        )
        forged = copy.deepcopy(receipt)
        forged["new_native_decimal"] = str(int(receipt["new_native_decimal"]) ^ 1)
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_refinement_receipt(*args, plan, forged))
        forged = copy.deepcopy(receipt)
        forged["old_native_decimal"] = "00"
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_refinement_receipt(*args, plan, forged))

    def test_source_epoch_and_interpreter_identity_are_bound(self):
        args = pair()
        old_source, old_q, old_bundle, _, _, _ = args
        changed = source_for((1, 4), different_law=True)
        new_q, new_bundle = construct(changed)
        with self.assertRaisesRegex(ValueError, "Different source-law epoch"):
            plan_minimal_refinement(
                old_source, old_q, old_bundle, changed, new_q, new_bundle
            )
        plan = plan_minimal_refinement(*args)
        receipt = certify_minimal_refinement(
            plan, old_native_state=old_q.project(5), witness_bits=1
        )
        broken = copy.deepcopy(receipt)
        broken["to_digest"] = "0" * 64
        broken["digest"] = _hash({k: v for k, v in broken.items() if k != "digest"})
        self.assertFalse(verify_refinement_receipt(*args, plan, broken))
        self.assertFalse(verify_refinement_plan(
            old_source, old_q, old_bundle, changed, new_q, new_bundle, plan
        ))

    def test_exhaustive_small_source_observer_translations(self):
        width = 5
        shift = tuple(1 << (i + 1) if i + 1 < width else 0
                      for i in range(width))
        identity = tuple(1 << i for i in range(width))
        outputs = ((1,), (2,), (4,), (1, 4), (1, 2, 4))
        prepared = {}
        for observer in outputs:
            source = SymbolicSource(
                width,
                (AffineMap(shift, 0), AffineMap(identity, 1)),
                observer,
                "tdl002-five-bit-exhaustive",
            )
            quotient, bundle = construct(source)
            prepared[observer] = source, quotient, bundle
        cases = 0
        for old_fields in outputs:
            for new_fields in outputs:
                args = (*prepared[old_fields], *prepared[new_fields])
                plan = plan_minimal_refinement(*args)
                self.assertTrue(verify_refinement_plan(*args, plan))
                for x in range(1 << width):
                    old_source, old_q, _, _, new_q, _ = args
                    witness = witness_from_source(old_source, plan, x)
                    receipt = certify_minimal_refinement(
                        plan, old_native_state=old_q.project(x),
                        witness_bits=witness,
                    )
                    self.assertTrue(verify_refinement_receipt(*args, plan, receipt))
                    self.assertEqual(int(receipt["new_native_decimal"]), new_q.project(x))
                    cases += 1
        self.assertEqual(cases, 25 * 32)


if __name__ == "__main__":
    unittest.main()
