"""Real-source E-CPU + VLD interpreter self-Ground and information-radiance tests.

No model weights or model forward pass: this is an operational structural
experiment with actual predecessor code, not a mock imported backend.
"""
from __future__ import annotations

import copy
import unittest

from vli.interpreter_ground import (
    GENESIS, RadianceJournal, _hash, lift_interpreter,
    derive_orbit_from_interpreter_description, load_real_interpreter_modules,
    observe_structural_orbit, rebind_interpreter_from_ground,
    operate_byte_ground, verify_byte_ground_run,
    verify_interpreter_ground, verify_radiance,
)
from mailcpu.virtual_layer import OperatorRegistry, OperatorSpec, VirtualLayerError
from mailcpu.ground_native_realization import AffineGround, GroundIntention, independent_observation
from vld.signal import ModularRingBackend, SignalGroundSpec, VirtualGroundInterpreter
from vld.memory import StructuralMemoryRegistry


def setup_interpreter():
    registry = OperatorRegistry()
    registry.declare(OperatorSpec(
        operator_id="vli.byte.relation", version=1,
        description="Declarative byte Ground relation; handler is host-owned",
        input_kind="object", output_kind="object",
        substrates=("cpu",), deterministic=True, effect_mode="pure",
    ))
    backend = ModularRingBackend(bit_width=8)
    signal = SignalGroundSpec.for_backend(backend, ground_id="vli.byte-ground")
    interpreter = VirtualGroundInterpreter(signal, backend)
    return registry, interpreter


class TestInterpreterSelfGround(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = load_real_interpreter_modules()

    def setUp(self):
        self.registry, self.interpreter = setup_interpreter()
        self.ground = lift_interpreter(self.modules, self.registry, self.interpreter)

    def test_lifts_real_own_and_predecessor_source_bytes(self):
        names = sorted(self.modules)
        self.assertEqual(names, ["ECPU.VirtualLayerRuntime", "VLD.VirtualGroundInterpreter",
                                  "VLI.InterpreterSelfGround"])
        self.assertEqual(self.ground["source_bytes_observed"],
                         sum(len(raw) for raw in self.modules.values()))
        self.assertTrue(any(len(row["chunks"]) > 1
                            for row in self.ground["source_modules"]))
        self.assertTrue(verify_interpreter_ground(
            self.ground, self.modules, self.registry, self.interpreter))
        self.assertEqual(self.ground["claims"]["self_hosting"], "NOT_DEMONSTRATED")
        self.assertEqual(self.ground["claims"]["handlers"],
                         "NOT_TRANSPORTED_MUST_BE_BOUND_LOCALLY")
        self.assertEqual(len(self.ground["ecpu_operator_snapshot"]["operators"]), 1)

    def test_contract_is_portable_but_unbound(self):
        fresh = OperatorRegistry.from_snapshot(self.ground["ecpu_operator_snapshot"])
        self.assertEqual(fresh.digest, self.registry.digest)
        with self.assertRaises(VirtualLayerError):
            fresh.resolve("vli.byte.relation", 1)

    def test_actual_vld_signal_interpreter_executes_byte_intention(self):
        record = operate_byte_ground(
            bytes((3, 5, 250)), left=0, right=1, operation="MUL",
            observer="observer-A", interpreter_ground=self.ground,
            signal_interpreter=self.interpreter)
        self.assertEqual(record["value"], 15)
        self.assertTrue(record["signal_steps"] >= 5)
        self.assertEqual(len(record["transcript"]["entries"]),
                         record["signal_steps"] + 2)
        self.assertTrue(verify_byte_ground_run(
            record, bytes((3,5,250)), self.ground, self.interpreter))
        self.assertTrue(verify_radiance(
            record["transcript"], expected_ground_digest=self.ground["ground_digest"],
            expected_head=record["transcript"]["head"]))

    def test_interpreter_operates_its_own_source_bytes(self):
        # Not a model token output: interpret two coordinate bytes from the
        # actual installed interpreter source using the VLD operational Ground.
        raw = self.modules["VLI.InterpreterSelfGround"][:65536]
        record = operate_byte_ground(raw, left=0, right=len(raw)-1,
                                     operation="ADD", observer="self-observer",
                                     interpreter_ground=self.ground,
                                     signal_interpreter=self.interpreter)
        self.assertEqual(record["value"], (raw[0] + raw[-1]) % 256)
        self.assertEqual(record["source_sha256"], _hash(raw))

    def test_same_interpreter_ground_rejects_changed_source(self):
        changed = dict(self.modules)
        raw = changed["VLI.InterpreterSelfGround"]
        changed["VLI.InterpreterSelfGround"] = b"!" + raw[1:]
        with self.assertRaises(ValueError):
            verify_interpreter_ground(self.ground, changed, self.registry, self.interpreter)

    def test_same_source_rejects_different_operator_catalog(self):
        other = OperatorRegistry()
        other.declare(OperatorSpec(
            operator_id="vli.byte.unrelated", version=1,
            description="Another declared relation",
            input_kind="object", output_kind="object", substrates=("cpu",),
        ))
        with self.assertRaises(ValueError):
            verify_interpreter_ground(self.ground, self.modules, other, self.interpreter)

    def test_refuses_invalid_coordinate_or_unadmitted_arithmetic(self):
        for args in ({"left":0, "right":30, "operation":"ADD"},
                     {"left":0, "right":1, "operation":"UNTRUSTED"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                operate_byte_ground(b"\x01\x03", observer="test",
                                    interpreter_ground=self.ground,
                                    signal_interpreter=self.interpreter, **args)

    def test_replay_refuses_changed_source_bytes(self):
        record = operate_byte_ground(b"\x01\x03", left=0, right=1, operation="ADD",
                                     observer="test", interpreter_ground=self.ground,
                                     signal_interpreter=self.interpreter)
        with self.assertRaises(ValueError):
            verify_byte_ground_run(record, b"\x01\x04", self.ground, self.interpreter)

    def test_radiance_refuses_changed_or_dropped_events(self):
        original = operate_byte_ground(b"\x03\x09", left=0, right=1, operation="ADD",
                                       observer="test", interpreter_ground=self.ground,
                                       signal_interpreter=self.interpreter)
        report = original["transcript"]
        head = report["head"]
        changed = copy.deepcopy(report)
        changed["entries"][2]["relation"] = "forged"
        with self.assertRaises(ValueError):
            verify_radiance(changed, expected_ground_digest=self.ground["ground_digest"],
                            expected_head=head)
        dropped = copy.deepcopy(report)
        dropped["entries"].pop(1)
        with self.assertRaises(ValueError):
            verify_radiance(dropped, expected_ground_digest=self.ground["ground_digest"],
                            expected_head=head)
        with self.assertRaises(ValueError):
            verify_radiance(report, expected_ground_digest=self.ground["ground_digest"],
                            expected_head=GENESIS)

    def test_verified_structural_memory_read_without_tick_enumeration(self):
        # VLD certifies the sampled generative law; E-CPU supplies an
        # independently implemented affine-orbit evaluator as a cross-check.
        samples = [(7 + 3 * t) % 256 for t in range(12)]
        memory = StructuralMemoryRegistry()
        spec, certificate = memory.certify_and_admit(
            "vli-desc-interpreter-orbit", samples,
            declared_modulus=256, domain="Z/256Z",
            context={"interpreter_ground_digest": self.ground["ground_digest"]},
        )
        coordinate = 1_000_000_003
        ecpu_reference = independent_observation(GroundIntention(
            request_id="vli-orbit-observation", ground=AffineGround(
                a=1, c=3, modulus=256, seed=7), coordinate=coordinate))
        result = observe_structural_orbit(
            self.ground, memory, memory_id="vli-desc-interpreter-orbit",
            coordinate=coordinate, expected_value=ecpu_reference,
            observer="structural-memory-observer")
        self.assertEqual(result["value"], ecpu_reference)
        self.assertEqual(result["expanded_intermediate_states"], 0)
        self.assertTrue(verify_radiance(
            result["transcript"], expected_ground_digest=self.ground["ground_digest"],
            expected_head=result["transcript"]["head"]))
        with self.assertRaises(ValueError):
            observe_structural_orbit(
                self.ground, memory, memory_id="vli-desc-interpreter-orbit",
                coordinate=coordinate, expected_value=(ecpu_reference+1) % 256,
                observer="structural-memory-observer")

    def test_rebind_from_lifted_ground_with_original_predecessor_runtime(self):
        declared, restored = rebind_interpreter_from_ground(
            self.ground, self.modules,
            expected_ground_digest=self.ground["ground_digest"])
        self.assertIsNot(restored, self.interpreter)
        self.assertEqual(restored.ground.fingerprint,
                         self.interpreter.ground.fingerprint)
        self.assertEqual(declared.digest, self.registry.digest)
        with self.assertRaises(VirtualLayerError):
            declared.resolve("vli.byte.relation")
        result = operate_byte_ground(b"\x07\x0b", left=0, right=1,
                                     operation="MUL", observer="restored-host",
                                     interpreter_ground=self.ground,
                                     signal_interpreter=restored)
        self.assertEqual(result["value"], 77)

    def test_rebinding_requires_trusted_external_ground_head(self):
        with self.assertRaises(ValueError):
            rebind_interpreter_from_ground(self.ground, self.modules,
                                           expected_ground_digest="0"*64)
        changed = dict(self.modules)
        changed["VLI.InterpreterSelfGround"] = b"forged"
        with self.assertRaises(ValueError):
            rebind_interpreter_from_ground(
                self.ground, changed, expected_ground_digest=self.ground["ground_digest"])

    def test_rebind_rejects_altered_signal_algebra(self):
        changed = copy.deepcopy(self.ground)
        changed["vld_signal_ground"]["domain_id"] = "unsupported.algebra"
        with self.assertRaises(ValueError):
            rebind_interpreter_from_ground(changed, self.modules,
                                           expected_ground_digest=self.ground["ground_digest"])

    def test_interpreter_description_generates_navigable_memory_law(self):
        source = "VLI.InterpreterSelfGround"
        coordinate = 9_876_543_210
        reading = derive_orbit_from_interpreter_description(
            self.ground, self.modules, source_name=source,
            coordinate=coordinate, observer="law-observer")
        raw = self.modules[source]
        seed, step = raw[0], raw[1] % 255 + 1
        self.assertEqual(reading["value"], (seed+coordinate*step)%256)
        self.assertEqual(reading["source_byte_values"], [raw[0], raw[1]])
        self.assertEqual(reading["sampled_witness_count"], 12)
        self.assertEqual(reading["expanded_intermediate_states"], 0)
        self.assertTrue(verify_radiance(
            reading["transcript"],
            expected_ground_digest=self.ground["ground_digest"],
            expected_head=reading["transcript"]["head"]))

    def test_source_derived_law_refuses_replaced_interpreter_description(self):
        changed = dict(self.modules)
        changed["VLI.InterpreterSelfGround"] = b"different source bytes"
        with self.assertRaises(ValueError):
            derive_orbit_from_interpreter_description(
                self.ground, changed, source_name="VLI.InterpreterSelfGround",
                coordinate=10**10, observer="law-observer")

    def test_observer_scopes_are_distinct_from_source_authority(self):
        # The Ground is source-observed, but never claims a trusted executable
        # was reconstructed on an independent host.
        claims = self.ground["claims"]
        self.assertNotEqual(claims["source"], "SELF_HOSTING_DEMONSTRATED")
        self.assertEqual(claims["runtime_qualification"], "NOT_GRANTED_BY_HASHES")


if __name__ == "__main__":
    unittest.main()
