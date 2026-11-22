"""Independent TDL-002 GF(2) refinement-plan and receipt verifier.

No imports from the minimal-refinement producer; rederive the chosen complement
by rank and check every source-space row identity. Digests are for integrity,
not claims of independent physical sensing or external authorization.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .problem_formed_interpreter import verify_interpreter

PLAN_SCHEMA = "vl-i.minimal-refinement-plan.v1"
RECEIPT_SCHEMA = "vl-i.minimal-refinement-receipt.v1"
SCOPE = "declared-affine-source;auxiliary-bits-not-sensor-attested"


def _digest(payload: dict) -> str:
    return sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


def _decimal(value, bits: int) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("Expected canonical decimal string")
    number = int(value)
    if str(number) != value or number.bit_length() > bits:
        raise ValueError("Noncanonical or out-of-domain integer")
    return number


def _rank(rows) -> int:
    """Compute independent rank via highest-set-bit pivot elimination."""
    pivots: dict[int, int] = {}
    for original in rows:
        value = original
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)


def _project(rows, value: int) -> int:
    return sum(((row & value).bit_count() & 1) << i
               for i, row in enumerate(rows))


def verify_refinement_plan(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle, plan,
) -> bool:
    """Recheck completeness and rank-optimality without calling the planner."""
    try:
        if (type(plan) is not dict or set(plan) != {
            "schema", "old_source_sha256", "new_source_sha256",
            "from_digest", "to_digest", "source_width", "old_bits", "new_bits",
            "additional_bits", "witness_source_rows_decimal",
            "new_from_old_and_witness_rows_decimal", "scope", "digest",
        } or plan["schema"] != PLAN_SCHEMA or plan["scope"] != SCOPE):
            return False
        if _digest({k: v for k, v in plan.items() if k != "digest"}) != plan["digest"]:
            return False
        if (not verify_interpreter(old_source, old_quotient, old_bundle)
                or not verify_interpreter(new_source, new_quotient, new_bundle)):
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
                or plan["old_bits"] != len(old_rows)
                or type(plan["new_bits"]) is not int
                or plan["new_bits"] != len(new_rows)
                or type(plan["additional_bits"]) is not int
                or type(plan["witness_source_rows_decimal"]) is not list
                or type(plan["new_from_old_and_witness_rows_decimal"]) is not list):
            return False
        d = plan["additional_bits"]
        if d < 0 or d > len(new_rows):
            return False
        if len(plan["witness_source_rows_decimal"]) != d:
            return False
        if len(plan["new_from_old_and_witness_rows_decimal"]) != len(new_rows):
            return False
        witnesses = tuple(_decimal(row, old_source.width)
                          for row in plan["witness_source_rows_decimal"])
        maps = tuple(_decimal(row, len(old_rows) + d)
                     for row in plan["new_from_old_and_witness_rows_decimal"])
        # Independently recompute deterministic greedy complement using ranks.
        union = list(old_rows)
        expected = []
        for row in new_rows:
            if _rank(union + [row]) > _rank(union):
                expected.append(row)
                union.append(row)
        if tuple(expected) != witnesses:
            return False
        if (d != _rank(old_rows + new_rows) - _rank(old_rows)
                or _rank(old_rows + witnesses) != _rank(old_rows) + d):
            return False
        # Every new output covector must be exactly represented, for ALL x.
        row_source = old_rows + witnesses
        for desired, code in zip(new_rows, maps):
            actual = 0
            for index, basis_row in enumerate(row_source):
                if (code >> index) & 1:
                    actual ^= basis_row
            if actual != desired:
                return False
        return True
    except (AttributeError, KeyError, ValueError, TypeError, OverflowError):
        return False


def verify_refinement_receipt(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    plan, receipt,
) -> bool:
    """Verify receipt, exact reconstruction, and source-independent plan."""
    try:
        if not verify_refinement_plan(
            old_source, old_quotient, old_bundle,
            new_source, new_quotient, new_bundle, plan,
        ):
            return False
        if (type(receipt) is not dict or set(receipt) != {
            "schema", "plan_digest", "from_digest", "to_digest",
            "old_native_decimal", "witness_decimal", "new_native_decimal",
            "additional_bits", "scope", "digest",
        } or receipt["schema"] != RECEIPT_SCHEMA or receipt["scope"] != SCOPE):
            return False
        if _digest({k: v for k, v in receipt.items() if k != "digest"}) != receipt["digest"]:
            return False
        if (receipt["plan_digest"] != plan["digest"]
                or receipt["from_digest"] != old_bundle["digest"]
                or receipt["to_digest"] != new_bundle["digest"]
                or type(receipt["additional_bits"]) is not int
                or receipt["additional_bits"] != plan["additional_bits"]):
            return False
        old_z = _decimal(receipt["old_native_decimal"], plan["old_bits"])
        extra = _decimal(receipt["witness_decimal"], plan["additional_bits"])
        new_z = _decimal(receipt["new_native_decimal"], plan["new_bits"])
        # This is an exact check of coordinate reconstruction, not a claim that
        # witness bits were authenticated or measured by a physical sensor.
        expected = _project(
            tuple(int(v) for v in plan["new_from_old_and_witness_rows_decimal"]),
            old_z | (extra << plan["old_bits"]),
        )
        return expected == new_z
    except (AttributeError, KeyError, ValueError, TypeError, OverflowError):
        return False
