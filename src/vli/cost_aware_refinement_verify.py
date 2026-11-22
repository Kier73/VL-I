"""Independent TDL-003 exact-cost planning and VGI-refinement receipt audit.

This module does not import or call the observation-plan producer.
Rank inclusion is checked with pivot elimination; candidate optimality is
enumerated via combinations, independently of the producer's bit-mask search.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
import re

from .problem_formed_interpreter import verify_interpreter

PLAN_SCHEMA = "vl-i.cost-aware-observation-plan.v1"
RECEIPT_SCHEMA = "vl-i.vgi-qualified-refinement.v1"
GROUND_ID = "pfi-refinement-observation"
SOURCE_STATE = "refinement"
SCOPE = "declared-source-row;VGI-policy-qualified;source-authentication-external"


def _digest(body: dict) -> str:
    return sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def _decimal(value, maximum_bits: int) -> int:
    if (type(value) is not str or re.fullmatch(r"(0|[1-9][0-9]*)", value) is None):
        raise ValueError("Noncanonical decimal")
    number = int(value)
    if number.bit_length() > maximum_bits:
        raise ValueError("Out-of-domain decimal")
    return number


def _rank(rows) -> int:
    pivots = {}
    for item in rows:
        value = item
        while value:
            p = value.bit_length() - 1
            if p in pivots:
                value ^= pivots[p]
            else:
                pivots[p] = value
                break
    return len(pivots)


def _project(rows, x) -> int:
    return sum(((row & x).bit_count() & 1) << i
               for i, row in enumerate(rows))


def verify_cost_plan(old_source, old_quotient, old_bundle,
                     new_source, new_quotient, new_bundle, plan) -> bool:
    """Independently verify exact minimum cost, rank, and row decoding."""
    try:
        required = {
            "schema", "old_source_sha256", "new_source_sha256",
            "from_digest", "to_digest", "source_width", "old_bits", "new_bits",
            "minimum_extra_rank", "offers", "selected_offer_ids",
            "selected_rows_decimal", "mapping_rows_decimal",
            "observation_count", "total_cost", "scope", "digest",
        }
        if (type(plan) is not dict or set(plan) != required
                or plan["schema"] != PLAN_SCHEMA or plan["scope"] != SCOPE
                or _digest({k: v for k, v in plan.items() if k != "digest"}) != plan["digest"]):
            return False
        if not (verify_interpreter(old_source, old_quotient, old_bundle)
                and verify_interpreter(new_source, new_quotient, new_bundle)):
            return False
        if (old_source.width != new_source.width
                or old_source.generators != new_source.generators
                or old_source.origin != new_source.origin):
            return False
        old_rows = tuple(old_quotient.basis_rows)
        new_rows = tuple(new_quotient.basis_rows)
        if (plan["old_source_sha256"] != old_source.digest
                or plan["new_source_sha256"] != new_source.digest
                or plan["from_digest"] != old_bundle["digest"]
                or plan["to_digest"] != new_bundle["digest"]
                or type(plan["source_width"]) is not int
                or plan["source_width"] != old_source.width
                or type(plan["old_bits"]) is not int
                or plan["old_bits"] != old_quotient.dimension
                or type(plan["new_bits"]) is not int
                or plan["new_bits"] != new_quotient.dimension
                or type(plan["minimum_extra_rank"]) is not int
                or type(plan["total_cost"]) is not int
                or type(plan["observation_count"]) is not int):
            return False
        expected_deficit = (_rank(old_rows + new_rows) - _rank(old_rows))
        if plan["minimum_extra_rank"] != expected_deficit:
            return False
        offers = plan["offers"]
        if type(offers) is not list or len(offers) > 12:
            return False
        normalized = []
        for offer in offers:
            if (type(offer) is not dict or set(offer) != {
                    "offer_id", "row_decimal", "cost", "locus"
                } or type(offer["offer_id"]) is not str
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", offer["offer_id"]) is None
                or type(offer["locus"]) is not str
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:/-]{0,127}", offer["locus"]) is None
                or type(offer["cost"]) is not int or
                not 1 <= offer["cost"] <= 1_000_000):
                return False
            row = _decimal(offer["row_decimal"], old_source.width)
            if row == 0:
                return False
            normalized.append((offer["offer_id"], row, offer["cost"]))
        if (normalized != sorted(normalized, key=lambda x: x[0])
                or len(set(item[0] for item in normalized)) != len(normalized)):
            return False

        selected = plan["selected_offer_ids"]
        row_strings = plan["selected_rows_decimal"]
        mapping = plan["mapping_rows_decimal"]
        if (type(selected) is not list or type(row_strings) is not list
                or type(mapping) is not list or len(mapping) != len(new_rows)
                or len(selected) != len(row_strings)
                or len(selected) != plan["observation_count"]):
            return False
        chosen = tuple(item for item in normalized if item[0] in selected)
        if ([item[0] for item in chosen] != selected
                or [str(item[1]) for item in chosen] != row_strings):
            return False
        if sum(x[2] for x in chosen) != plan["total_cost"]:
            return False

        def enough(subset):
            basis = old_rows + tuple(v[1] for v in subset)
            return all(_rank(basis + (row,)) == _rank(basis) for row in new_rows)

        if not enough(chosen):
            return False
        # Independent exhaustive oracle: combinations ordered by cardinality.
        best = None
        for count in range(len(normalized) + 1):
            for subset in combinations(normalized, count):
                if not enough(subset):
                    continue
                score = (sum(v[2] for v in subset), count, tuple(v[0] for v in subset))
                if best is None or score < best:
                    best = score
        if best is None or best != (plan["total_cost"], len(chosen), tuple(selected)):
            return False

        combined = old_rows + tuple(v[1] for v in chosen)
        codes = tuple(_decimal(value, len(combined)) for value in mapping)
        for source_row, code in zip(new_rows, codes):
            decoded_row = 0
            for j, basis_row in enumerate(combined):
                if (code >> j) & 1:
                    decoded_row ^= basis_row
            if decoded_row != source_row:
                return False
        return True
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return False


def _context(plan: dict, old: int) -> str:
    return _digest({
        "schema": "vl-i.refinement-observation-context.v1",
        "plan_digest": plan["digest"], "old_native_decimal": str(old),
    })


def verify_vgi_refinement_receipt(old_source, old_quotient, old_bundle,
                                  new_source, new_quotient, new_bundle,
                                  plan, receipt, *, ledger=None) -> bool:
    """Check VGI-finalized evidence, exact receipt identity and new coordinate."""
    try:
        if not verify_cost_plan(old_source, old_quotient, old_bundle,
                                new_source, new_quotient, new_bundle, plan):
            return False
        required = {
            "schema", "plan_digest", "old_native_decimal", "new_native_decimal",
            "observation_context", "selected_observations", "ledger_digest",
            "policy_digest", "total_cost", "scope", "digest",
        }
        if (type(receipt) is not dict or set(receipt) != required
                or receipt["schema"] != RECEIPT_SCHEMA or receipt["scope"] != SCOPE
                or _digest({k: v for k, v in receipt.items() if k != "digest"}) != receipt["digest"]
                or receipt["plan_digest"] != plan["digest"]
                or receipt["total_cost"] != plan["total_cost"]):
            return False
        old = _decimal(receipt["old_native_decimal"], plan["old_bits"])
        claimed_new = _decimal(receipt["new_native_decimal"], plan["new_bits"])
        ctx = _context(plan, old)
        if receipt["observation_context"] != ctx:
            return False
        selected = plan["selected_offer_ids"]
        if type(receipt["selected_observations"]) is not list:
            return False
        if len(receipt["selected_observations"]) != len(selected):
            return False
        expected = []
        if selected:
            from vgi.evidence import TransitionEvidenceLedger
            if not isinstance(ledger, TransitionEvidenceLedger):
                return False
            # Revalidate the VGI immutable evidence-finalization records.
            ledger = TransitionEvidenceLedger.from_dict(ledger.to_dict())
            if (ledger.policy.ground_id != GROUND_ID
                    or ledger.policy.context_digest != ctx
                    or ledger.policy.strict is not True
                    or ledger.fingerprint != receipt["ledger_digest"]
                    or ledger.policy.fingerprint != receipt["policy_digest"]):
                return False
            for offer_id in selected:
                decision = ledger.decision(SOURCE_STATE, offer_id)
                if decision.status != "AUTHORIZED":
                    return False
                actual = ledger.transition_observation(SOURCE_STATE, offer_id)
                if len(actual.outcomes) != 1:
                    return False
                observed = actual.outcomes[0]
                if (observed.target != SOURCE_STATE or len(observed.output) != 1
                        or type(observed.output[0]) is not int
                        or observed.output[0] not in (0, 1)):
                    return False
                expected.append({
                    "offer_id": offer_id, "bit": observed.output[0],
                    "decision_digest": decision.digest,
                    "evidence_ids": list(actual.evidence),
                })
        elif (ledger is not None or receipt["ledger_digest"] is not None
              or receipt["policy_digest"] is not None):
            return False
        if receipt["selected_observations"] != expected:
            return False
        bits = sum(row["bit"] << i for i, row in enumerate(expected))
        combined = old | (bits << plan["old_bits"])
        return claimed_new == _project(
            tuple(int(x) for x in plan["mapping_rows_decimal"]), combined,
        )
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError, IndexError):
        return False
