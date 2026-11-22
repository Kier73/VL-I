"""TDL-002: independent finite Moore and symbolic affine quotient commutation.

The fixture converts a DECLARED small affine source into a total finite Moore
table. This checks equality of separately generated observer equivalences
and execution results; it is not a new external-ground admission mechanism.
"""
from __future__ import annotations

from itertools import product
import unittest

from mailcpu.intention_formed_fabric import (
    FiniteObserverLaw, form_observer_machine, verify_formed_machine,
)
from navigable_compute.symbolic_quotient import (
    AffineMap, SymbolicSource, apply_rows,
)
from vli.problem_formed_interpreter import construct, verify_interpreter


def affine_fixture(outputs):
    width = 5
    # Two distinct operators: shift observability left through the source
    # registers and toggle bit zero. Both have exact GF(2) semantics.
    shift = tuple(1 << (i + 1) if i + 1 < width else 0
                  for i in range(width))
    identity = tuple(1 << i for i in range(width))
    return SymbolicSource(
        width,
        (AffineMap(shift, 0), AffineMap(identity, 1)),
        tuple(outputs),
        "tdl002-finite-affine-bridge",
    )


def finite_moore_from_affine(source):
    n = 1 << source.width
    return FiniteObserverLaw(
        states=tuple(f"s{i}" for i in range(n)),
        actions=("shift", "toggle"),
        fields=("observer",),
        transitions=tuple(
            tuple(generator.apply(x) for generator in source.generators)
            for x in range(n)
        ),
        observations=tuple((source.observe(x),) for x in range(n)),
    )


class TranslationCommutationTests(unittest.TestCase):
    def test_finite_partition_equals_affine_observer_coordinate(self):
        checked_pairs = 0
        checked_words = 0
        for output_rows in ((1,), (2,), (4,), (1, 4), (1, 2, 4)):
            source = affine_fixture(output_rows)
            quotient, bundle = construct(source)
            self.assertTrue(verify_interpreter(source, quotient, bundle))
            finite = finite_moore_from_affine(source)
            formed = form_observer_machine(finite, ("observer",))
            self.assertTrue(verify_formed_machine(finite, formed))

            for x in range(1 << source.width):
                for y in range(1 << source.width):
                    self.assertEqual(
                        formed.state_groups[x] == formed.state_groups[y],
                        quotient.project(x) == quotient.project(y),
                    )
                    checked_pairs += 1

            # Two independently generated state-transition realizations
            # commute with the original source observation on short words.
            words = ((), (0,), (1,), (0, 1), (1, 0), (0, 1, 1))
            for x in range(1 << source.width):
                for word in words:
                    true_x = x
                    native_z = quotient.project(x)
                    for action in word:
                        true_x = source.generators[action].apply(true_x)
                        native_z = quotient.operators[action].apply(native_z)
                    native_output = apply_rows(quotient.observer_decoder, native_z)
                    finite_output = formed.apply({
                        "initial_state": f"s{x}",
                        "word": [finite.actions[a] for a in word],
                    })["observation"]["observer"]
                    self.assertEqual(native_output, source.observe(true_x))
                    self.assertEqual(finite_output, source.observe(true_x))
                    checked_words += 1

            self.assertEqual(len(formed.groups), 1 << quotient.dimension)
        self.assertEqual(checked_pairs, 5 * 32 * 32)
        self.assertEqual(checked_words, 5 * 32 * 6)


if __name__ == "__main__":
    unittest.main()
