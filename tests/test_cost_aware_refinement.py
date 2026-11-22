"""TDL-003: optimal observation selection and actual VGI evidence-finalization bridge."""
from __future__ import annotations

import copy
import unittest

from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from vli.problem_formed_interpreter import construct, _hash
from vli.cost_aware_refinement import (
    ObservationOffer, plan_cost_aware_observations, observation_context,
    acquire_vgi_refinement, GROUND_ID, SOURCE_STATE,
)
from vli.cost_aware_refinement_verify import (
    verify_cost_plan, verify_vgi_refinement_receipt,
)
from vgi.channels import ChannelContract, ChannelExchangeLedger, ChannelMessage
from vgi.encounters import EncounterProvenance
from vgi.evidence import (
    TransitionEvidenceLedger, TransitionEvidencePolicy, TransportFrameReceipt,
)


def grounds(*, old_outputs=(1,), new_outputs=(1, 2), width=4, shift=False):
    operator = tuple(1 << i for i in range(width))
    if shift:
        operator = tuple(1 << (i + 1) if i + 1 < width else 0
                         for i in range(width))
    transition = (AffineMap(operator, 0),)
    old = SymbolicSource(width, transition, tuple(old_outputs), "tdl003-same-law")
    new = SymbolicSource(width, transition, tuple(new_outputs), "tdl003-same-law")
    old_q, old_bundle = construct(old)
    new_q, new_bundle = construct(new)
    return (old, old_q, old_bundle, new, new_q, new_bundle)


def offers():
    return (
        ObservationOffer("direct", 0b0010, 10, "observer-A"),
        ObservationOffer("cheap-left", 0b0110, 1, "observer-B"),
        ObservationOffer("cheap-right", 0b0100, 1, "observer-C"),
    )


def observed_ledger(plan, old_z, bits, *,
                    missing=(), dissent=(), unknown=(), finalize=True, context=None):
    """Use VGI's real channel, encounter, transport, quorum and finalization APIs."""
    context = context or observation_context(plan, old_native_state=old_z)
    policy = TransitionEvidencePolicy(
        GROUND_ID, context, ("sensor-A", "sensor-B"), 2, True
    )
    ledger = TransitionEvidenceLedger(policy)
    contract = ChannelContract(
        "refinement.bus", "refinement.frame/1",
        observation_schema="vgi.transition-observation/1",
    )
    messages = []
    count = 0
    for offer_id, bit in zip(plan["selected_offer_ids"], bits):
        for source in ("sensor-A", "sensor-B"):
            if (offer_id, source) in missing:
                continue
            status = "unknown" if (offer_id, source) in unknown else "known"
            if (offer_id, source) in dissent:
                value = 1 ^ bit
            else:
                value = bit
            payload = {
                "status": status, "source_state": SOURCE_STATE, "action": offer_id,
            }
            if status == "known":
                payload.update({"target": SOURCE_STATE, "output": [value]})
            messages.append(ChannelMessage(
                f"obs-{count}", "refinement.bus", "refinement.frame/1", "observation",
                payload,
                EncounterProvenance(source, f"capture://{source}/{count}", collector="fixture"),
            ))
            count += 1
    exchange = ChannelExchangeLedger()
    for message in messages:
        exchange = exchange.record_observation(contract, message)
    encounters = exchange.to_encounter_ledger()
    for sequence, message in enumerate(messages, 1):
        frame = TransportFrameReceipt.capture(
            message, transport_id="tdl003-fixture-transport",
            sequence=sequence, received_at="2026-11-22T20:00:00Z",
        )
        ledger = ledger.ingest_channel_message(exchange, message.message_id, frame, encounters)
    if finalize:
        for offer_id in plan["selected_offer_ids"]:
            if any(record.action == offer_id for record in ledger.evidence):
                ledger = ledger.finalize(SOURCE_STATE, offer_id)
    return ledger


class CostAwareEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = grounds()
        cls.plan = plan_cost_aware_observations(*cls.args, offers())

    def test_cost_optimum_uses_two_rows_outside_target_instead_of_one(self):
        plan = self.plan
        self.assertTrue(verify_cost_plan(*self.args, plan))
        self.assertEqual(plan["minimum_extra_rank"], 1)
        self.assertEqual(plan["observation_count"], 2)
        self.assertEqual(plan["total_cost"], 2)
        self.assertEqual(plan["selected_offer_ids"], ["cheap-left", "cheap-right"])
        old, old_q, _, _, new_q, _ = self.args
        for x in range(1 << old.width):
            observations = [((int(row) & x).bit_count() & 1)
                            for row in plan["selected_rows_decimal"]]
            ledger = observed_ledger(plan, old_q.project(x), observations)
            receipt = acquire_vgi_refinement(
                *self.args, plan, old_native_state=old_q.project(x), ledger=ledger
            )
            self.assertTrue(verify_vgi_refinement_receipt(
                *self.args, plan, receipt, ledger=ledger
            ))
            self.assertEqual(int(receipt["new_native_decimal"]), new_q.project(x))

    def test_pending_missing_and_unknown_are_never_promoted(self):
        plan = self.plan
        args = self.args
        x = 0b0111
        old_z = args[1].project(x)
        bits = [((int(row) & x).bit_count() & 1)
                for row in plan["selected_rows_decimal"]]
        for choices in (
            {"finalize": False},
            {"missing": ((plan["selected_offer_ids"][0], "sensor-B"),)},
            {"unknown": ((plan["selected_offer_ids"][0], "sensor-B"),)},
            {"dissent": ((plan["selected_offer_ids"][0], "sensor-B"),)},
        ):
            ledger = observed_ledger(plan, old_z, bits, **choices)
            with self.subTest(choices=choices):
                with self.assertRaises(ValueError):
                    acquire_vgi_refinement(
                        *args, plan, old_native_state=old_z, ledger=ledger
                    )

    def test_ledger_context_and_mutated_receipts_fail_independent_audit(self):
        plan, args = self.plan, self.args
        x = 0b0111
        old_z = args[1].project(x)
        bits = [((int(row) & x).bit_count() & 1)
                for row in plan["selected_rows_decimal"]]
        wrong_context = "0" * 64
        ledger = observed_ledger(plan, old_z, bits, context=wrong_context)
        with self.assertRaisesRegex(ValueError, "context"):
            acquire_vgi_refinement(*args, plan, old_native_state=old_z, ledger=ledger)
        ledger = observed_ledger(plan, old_z, bits)
        receipt = acquire_vgi_refinement(
            *args, plan, old_native_state=old_z, ledger=ledger
        )
        for field, changed in (
            ("new_native_decimal", str(int(receipt["new_native_decimal"]) ^ 1)),
            ("old_native_decimal", str(old_z ^ 1)),
            ("ledger_digest", "0" * 64),
            ("observation_context", "0" * 64),
        ):
            forged = copy.deepcopy(receipt)
            forged[field] = changed
            forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
            self.assertFalse(verify_vgi_refinement_receipt(
                *args, plan, forged, ledger=ledger
            ))
        forged = copy.deepcopy(receipt)
        forged["selected_observations"][0]["bit"] ^= 1
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_vgi_refinement_receipt(
            *args, plan, forged, ledger=ledger
        ))

    def test_zero_cost_when_old_coordinate_already_sufficient(self):
        args = grounds(old_outputs=(1, 2), new_outputs=(1,))
        plan = plan_cost_aware_observations(*args, ())
        self.assertEqual(plan["total_cost"], 0)
        self.assertEqual(plan["observation_count"], 0)
        self.assertEqual(plan["minimum_extra_rank"], 0)
        for x in range(16):
            old_z = args[1].project(x)
            receipt = acquire_vgi_refinement(
                *args, plan, old_native_state=old_z
            )
            self.assertTrue(verify_vgi_refinement_receipt(
                *args, plan, receipt,
            ))
            self.assertEqual(int(receipt["new_native_decimal"]), args[4].project(x))
        with self.assertRaises(ValueError):
            acquire_vgi_refinement(
                *args, plan, old_native_state=0,
                ledger=TransitionEvidenceLedger(
                    TransitionEvidencePolicy(
                        GROUND_ID, observation_context(plan, old_native_state=0),
                        ("sensor-A",), 1, True,
                    )
                ),
            )

    def test_unspanned_catalogue_and_invalid_offer_domains_refuse(self):
        with self.assertRaisesRegex(ValueError, "do not span"):
            plan_cost_aware_observations(
                *self.args, (ObservationOffer("irrelevant", 0b1000, 1, "observer"),)
            )
        with self.assertRaises(ValueError):
            ObservationOffer("invalid", 1, True, "observer")
        with self.assertRaises(ValueError):
            ObservationOffer("invalid", 0, 1, "observer")
        with self.assertRaises(ValueError):
            plan_cost_aware_observations(
                *self.args, tuple(ObservationOffer(
                    f"o{i}", 1, 1, "observer"
                ) for i in range(13)),
            )
        with self.assertRaises(ValueError):
            plan_cost_aware_observations(
                *self.args, (ObservationOffer("duplicate", 2, 1, "a"),
                             ObservationOffer("duplicate", 4, 1, "b")),
            )

    def test_rehashed_suboptimal_and_invalid_mapping_are_refused(self):
        plan = self.plan
        forged = copy.deepcopy(plan)
        forged["selected_offer_ids"] = ["direct"]
        forged["selected_rows_decimal"] = ["2"]
        forged["observation_count"] = 1
        forged["total_cost"] = 10
        # Derive exact direct-row mapping on this simple identity fixture.
        forged["mapping_rows_decimal"] = ["1", "2"]
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_cost_plan(*self.args, forged))
        forged = copy.deepcopy(plan)
        forged["mapping_rows_decimal"][0] = "0"
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_cost_plan(*self.args, forged))
        forged = copy.deepcopy(plan)
        forged["minimum_extra_rank"] = 0
        forged["digest"] = _hash({k: v for k, v in forged.items() if k != "digest"})
        self.assertFalse(verify_cost_plan(*self.args, forged))

    def test_new_intention_and_source_epoch_binding(self):
        plan = self.plan
        args = grounds(old_outputs=(1,), new_outputs=(1, 4))
        self.assertFalse(verify_cost_plan(*args, plan))
        with self.assertRaisesRegex(ValueError, "Different source-law epoch"):
            bad = grounds(old_outputs=(1,), new_outputs=(1, 2), shift=True)
            plan_cost_aware_observations(
                *self.args[:3], *bad[3:], offers()
            )


if __name__ == "__main__":
    unittest.main()
