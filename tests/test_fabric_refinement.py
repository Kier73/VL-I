"""TDL-004: E-CPU typed host/route paths -> VGI evidence -> VL-I state."""
from __future__ import annotations

from dataclasses import replace
import copy
import unittest

from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from vli.problem_formed_interpreter import construct, _hash
from vli.cost_aware_refinement import ObservationOffer
from vli.fabric_refinement import (
    INPUT, OUTPUT, offer_operation, offer_output_kind,
    plan_fabric_refinement, execute_fabric_refinement,
)
from vli.fabric_refinement_verify import verify_fabric_refinement
from mailcpu.substrate_bootstrap import (
    AdaptiveBootstrap, BootstrapPolicy, BootstrapRefused,
    CarrierBinding, ChannelObservation, OperationOffer, RealizerBinding,
    ResourceObservation, SubstrateObservation, digest,
)


def grounds():
    identity = tuple(1 << i for i in range(4))
    operators = (AffineMap(identity, 0),)
    old = SymbolicSource(4, operators, (1,), "tdl004-stationary-law")
    new = SymbolicSource(4, operators, (1, 2), "tdl004-stationary-law")
    old_q, old_bundle = construct(old)
    new_q, new_bundle = construct(new)
    return old, old_q, old_bundle, new, new_q, new_bundle


OFFERS = (
    ObservationOffer("direct", 0b0010, 10, "direct"),
    ObservationOffer("cheap-left", 0b0110, 1, "cheap-left"),
    ObservationOffer("cheap-right", 0b0100, 1, "cheap-right"),
)


def runtime(source_state=0b0111, *, broken_handler=None, deny=(), expiry=4):
    """Build receiver-local, path-distinct observed E-CPU fabrics.

    The right observer path has an intermediate relay. A local fixture
    source-state is closed over by handlers; it is not sent with tasks.
    Oracle closures calculate parity independently from the handler result.
    """
    def res(name, operation=None, expires=20):
        offers = () if operation is None else (operation,)
        return ResourceObservation(
            name, "receiver-observed-locus", 1, expires, "receiver-prober",
            digest({"host": name, "epoch": 1}), offers,
        )

    def edge(name, start, end, kind):
        return ChannelObservation(
            name, start, end, (kind,), 1, 1, 20,
            digest({"channel": name, "epoch": 1}), 8192,
        )

    channels = (
        edge("to-left", "start", "cheap-left", INPUT),
        edge("left-back", "cheap-left", "collector", offer_output_kind("cheap-left")),
        edge("to-relay", "start", "relay", INPUT),
        edge("relay-right", "relay", "cheap-right", INPUT),
        edge("right-back", "cheap-right", "collector", offer_output_kind("cheap-right")),
        edge("to-direct", "start", "direct", INPUT),
        edge("direct-back", "direct", "collector", offer_output_kind("direct")),
    )
    inventory = SubstrateObservation((
        res("start"),
        res("collector"),
        res("relay"),
        res("direct", OperationOffer(offer_operation("direct"), INPUT, offer_output_kind("direct"), 10)),
        res("cheap-left", OperationOffer(offer_operation("cheap-left"), INPUT, offer_output_kind("cheap-left"), 1)),
        res("cheap-right", OperationOffer(offer_operation("cheap-right"), INPUT, offer_output_kind("cheap-right"), 1),
            expires=expiry),
    ), channels)

    bindings = []
    for offer in OFFERS:
        def handle(value, offer=offer):
            bit = ((offer.row & source_state).bit_count() & 1)
            if broken_handler == offer.offer_id:
                bit ^= 1
            return {
                "request_digest": digest(value),
                "offer_id": offer.offer_id,
                "row_decimal": str(offer.row),
                "bit": bit,
            }

        def oracle(previous, value, offer=offer):
            return (
                type(value) is dict
                and set(value) == {"request_digest", "offer_id", "row_decimal", "bit"}
                and value["request_digest"] == digest(previous)
                and value["offer_id"] == offer.offer_id
                and value["row_decimal"] == str(offer.row)
                and type(value["bit"]) is int
                and value["bit"] == ((offer.row & source_state).bit_count() & 1)
            )

        bindings.append(RealizerBinding(
            offer.locus, offer_operation(offer.offer_id),
            handle, digest({"handler": offer.offer_id}), oracle,
        ))
    carriers = [CarrierBinding(
        c.channel_id, lambda data, coordinate: data,
        digest({"carrier": c.channel_id, "receiver": "local"}),
    ) for c in channels]
    policy = BootstrapPolicy(
        frozenset(
            (o.locus, offer_operation(o.offer_id)) for o in OFFERS
            if o.offer_id not in deny
        ),
        frozenset(c.channel_id for c in channels),
        "receiver-owned-tdl004",
    )
    return AdaptiveBootstrap(inventory, policy, bindings, carriers)


class FabricObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = grounds()

    def plan(self, instance, state=0b0111, coordinate=3, offers=OFFERS):
        return plan_fabric_refinement(
            *self.args, offers, instance,
            ingress="start", collector="collector",
            old_native_state=self.args[1].project(state),
            coordinate=coordinate,
        )

    def test_two_multihost_paths_beat_direct_and_are_vgi_qualified(self):
        for state in range(16):
            fabric = runtime(source_state=state)
            acquisition = self.plan(fabric, state=state)
            mathematical = acquisition.mathematical_plan
            self.assertEqual(mathematical["minimum_extra_rank"], 1)
            self.assertEqual(mathematical["selected_offer_ids"],
                             ["cheap-left", "cheap-right"])
            self.assertEqual(mathematical["total_cost"], 7)
            self.assertEqual(
                [route.plan.estimated_quanta for route in acquisition.routes],
                [3, 4],
            )
            self.assertEqual(
                [step.operation for step in acquisition.routes[1].plan.steps],
                ["to-relay", "relay-right",
                 offer_operation("cheap-right"), "right-back"],
            )
            result = execute_fabric_refinement(*self.args, acquisition, fabric)
            self.assertTrue(verify_fabric_refinement(*self.args, acquisition, fabric, result))
            self.assertEqual(int(result["refinement_receipt"]["new_native_decimal"]),
                             self.args[4].project(state))
            self.assertEqual(len(result["path_reports"]), 2)
            self.assertEqual(result["declared_total_quanta"], 7)
            self.assertGreater(result["measured_application_bytes"], 0)
            self.assertGreaterEqual(result["measured_elapsed_ns"], 0)
            for report in result["path_reports"]:
                self.assertFalse(report["report"]["physical_independence_attested"])
                self.assertEqual(report["report"]["receiver_oracle_checked_steps"], 1)

    def test_lease_expiration_reforms_processor_route_without_semantic_loss(self):
        fabric = runtime(source_state=5, expiry=4)
        old_acquisition = self.plan(fabric, state=5, coordinate=3)
        new_acquisition = self.plan(fabric, state=5, coordinate=10)
        self.assertEqual(old_acquisition.mathematical_plan["total_cost"], 7)
        self.assertEqual(new_acquisition.mathematical_plan["selected_offer_ids"],
                         ["direct"])
        self.assertEqual(new_acquisition.mathematical_plan["total_cost"], 12)
        self.assertNotEqual(old_acquisition.fingerprint, new_acquisition.fingerprint)
        result = execute_fabric_refinement(*self.args, new_acquisition, fabric)
        self.assertEqual(int(result["refinement_receipt"]["new_native_decimal"]),
                         self.args[4].project(5))

    def test_policy_revocation_and_missing_receivers_refuse_or_replan(self):
        fabric = runtime(source_state=3, deny=("cheap-right",))
        plan = self.plan(fabric, state=3)
        self.assertEqual(plan.mathematical_plan["selected_offer_ids"], ["direct"])
        self.assertEqual(plan.mathematical_plan["total_cost"], 12)
        result = execute_fabric_refinement(*self.args, plan, fabric)
        self.assertTrue(verify_fabric_refinement(*self.args, plan, fabric, result))
        # Revoking after planning changes the local authority fingerprint.
        newly_revoked = runtime(source_state=3, deny=("cheap-right", "direct"))
        with self.assertRaises(ValueError):
            execute_fabric_refinement(*self.args, plan, newly_revoked)
        with self.assertRaisesRegex(ValueError, "span"):
            self.plan(newly_revoked, state=3)

    def test_receiver_oracle_refuses_false_handler_output(self):
        fabric = runtime(source_state=5, broken_handler="cheap-left")
        plan = self.plan(fabric, state=5)
        with self.assertRaises(BootstrapRefused):
            execute_fabric_refinement(*self.args, plan, fabric)

    def test_all_bare_advertisements_without_handler_fail_to_supply_witness(self):
        fabric = runtime(source_state=3)
        advertisement_only = AdaptiveBootstrap(
            fabric.inventory, fabric.policy, (), tuple(fabric.carriers.values())
        )
        with self.assertRaisesRegex(ValueError, "span"):
            self.plan(advertisement_only, state=3)

    def test_receipt_tamper_is_refused_even_if_outer_hash_is_recomputed(self):
        fabric = runtime(source_state=5)
        plan = self.plan(fabric, state=5)
        result = execute_fabric_refinement(*self.args, plan, fabric)
        forged = copy.deepcopy(result)
        forged["path_reports"][0]["bit"] ^= 1
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_fabric_refinement(*self.args, plan, fabric, forged))
        forged = copy.deepcopy(result)
        forged["declared_total_quanta"] = 2
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_fabric_refinement(*self.args, plan, fabric, forged))
        forged = copy.deepcopy(result)
        forged["refinement_receipt"]["new_native_decimal"] = "0"
        forged["refinement_receipt"]["digest"] = _hash({
            k: v for k, v in forged["refinement_receipt"].items() if k != "digest"
        })
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_fabric_refinement(*self.args, plan, fabric, forged))

    def test_zero_extra_information_executes_without_hosts(self):
        old, q_old, old_b, _, _, _ = self.args
        new = SymbolicSource(
            4, old.generators, (1,), old.origin,
        )
        q_new, new_b = construct(new)
        args = (old, q_old, old_b, new, q_new, new_b)
        fabric = runtime(source_state=7, deny=tuple(o.offer_id for o in OFFERS))
        acq = plan_fabric_refinement(
            *args, (), fabric, ingress="start", collector="collector",
            old_native_state=q_old.project(7), coordinate=3,
        )
        self.assertEqual(acq.routes, ())
        self.assertEqual(acq.mathematical_plan["total_cost"], 0)
        result = execute_fabric_refinement(*args, acq, fabric)
        self.assertTrue(verify_fabric_refinement(*args, acq, fabric, result))
        self.assertIsNone(result["vgi_ledger"])
        self.assertEqual(result["refinement_receipt"]["new_native_decimal"],
                         str(q_new.project(7)))


if __name__ == "__main__":
    unittest.main()
