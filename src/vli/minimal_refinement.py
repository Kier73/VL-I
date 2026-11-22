"""TDL-002: minimal auxiliary-bit witness for exact affine intention succession.

A refinement plan is an explicitly checked GF(2) linear mapping. It does not
authenticate a physical source state; an observer supplies the missing bits.
Existing PFI-001 anchor-based succession remains available unchanged.
"""
from __future__ import annotations

from .problem_formed_interpreter import verify_interpreter, _hash

PLAN_SCHEMA = "vl-i.minimal-refinement-plan.v1"
RECEIPT_SCHEMA = "vl-i.minimal-refinement-receipt.v1"
SCOPE = "declared-affine-source;auxiliary-bits-not-sensor-attested"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _express(target: int, rows: tuple[int, ...]) -> int | None:
    """Return coefficients representing target using source rows over GF(2).

    Pivots retain both a reduced row and its coefficient mask. This algorithm
    does not rely on the NCR basis being ordered in any particular way.
    """
    pivots: dict[int, tuple[int, int]] = {}
    for index, original in enumerate(rows):
        value, code = original, 1 << index
        for pivot in sorted(pivots, reverse=True):
            if value & (1 << pivot):
                prior, representation = pivots[pivot]
                value ^= prior
                code ^= representation
        if value:
            pivots[value.bit_length() - 1] = (value, code)
    value, code = target, 0
    for pivot in sorted(pivots, reverse=True):
        if value & (1 << pivot):
            prior, representation = pivots[pivot]
            value ^= prior
            code ^= representation
    return code if value == 0 else None


def plan_minimal_refinement(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
) -> dict:
    """Find the smallest complement of the old observer row space.

    The selected additional covectors are source-space rows of the new
    observer quotient. Their values plus the retained old coordinate
    determine every new coordinate for the exact same source-law epoch.
    """
    _require(
        verify_interpreter(old_source, old_quotient, old_bundle)
        and verify_interpreter(new_source, new_quotient, new_bundle),
        "Both interpreters require independent mathematical admission",
    )
    _require(
        old_source.width == new_source.width
        and old_source.generators == new_source.generators
        and old_source.origin == new_source.origin,
        "Different source-law epoch",
    )
    old_rows = tuple(old_quotient.basis_rows)
    new_rows = tuple(new_quotient.basis_rows)
    union = list(old_rows)
    complement: list[int] = []
    for row in new_rows:
        if _express(row, tuple(union)) is None:
            complement.append(row)
            union.append(row)
    projections = tuple(_express(row, tuple(union)) for row in new_rows)
    _require(all(row is not None for row in projections), "Unclosed refinement representation")
    body = {
        "schema": PLAN_SCHEMA,
        "old_source_sha256": old_source.digest,
        "new_source_sha256": new_source.digest,
        "from_digest": old_bundle["digest"],
        "to_digest": new_bundle["digest"],
        "source_width": old_source.width,
        "old_bits": old_quotient.dimension,
        "new_bits": new_quotient.dimension,
        "additional_bits": len(complement),
        "witness_source_rows_decimal": [str(row) for row in complement],
        "new_from_old_and_witness_rows_decimal": [str(row) for row in projections],
        "scope": SCOPE,
    }
    plan = {**body, "digest": _hash(body)}
    from .minimal_refinement_verify import verify_refinement_plan
    _require(
        verify_refinement_plan(
            old_source, old_quotient, old_bundle,
            new_source, new_quotient, new_bundle, plan,
        ),
        "Independent refinement-plan audit refused",
    )
    return plan


def witness_from_source(source, plan: dict, source_state: int) -> int:
    """Observe the d selected source-space bits without retaining full x."""
    _require(type(plan) is dict and source.digest == plan.get("old_source_sha256"),
             "Witness source identity differs from plan")
    _require(type(source_state) is int and 0 <= source_state < (1 << source.width),
             "Source anchor outside mathematical domain")
    rows = tuple(int(row) for row in plan["witness_source_rows_decimal"])
    _require(len(rows) == plan["additional_bits"], "Malformed witness rows")
    return sum(((row & source_state).bit_count() & 1) << i
               for i, row in enumerate(rows))


def certify_minimal_refinement(
    plan: dict, *, old_native_state: int, witness_bits: int | None = None,
) -> dict:
    """Transform old native state using exactly d auxiliary measured bits.

    This producer only applies a previously checked plan. Independently
    verify the returned certificate against both source/quotient Grounds
    before using it as a computational continuation.
    """
    _require(type(plan) is dict and plan.get("schema") == PLAN_SCHEMA,
             "Missing refinement plan")
    _require(type(old_native_state) is int and
             0 <= old_native_state < (1 << plan["old_bits"]),
             "Old native state outside quotient")
    d = plan["additional_bits"]
    if witness_bits is None and d == 0:
        witness_bits = 0
    _require(type(witness_bits) is int and 0 <= witness_bits < (1 << d),
             "Refinement requires exactly the additional observer bits")
    combined = old_native_state | (witness_bits << plan["old_bits"])
    new_native = sum(
        ((int(row) & combined).bit_count() & 1) << i
        for i, row in enumerate(plan["new_from_old_and_witness_rows_decimal"])
    )
    body = {
        "schema": RECEIPT_SCHEMA,
        "plan_digest": plan["digest"],
        "from_digest": plan["from_digest"],
        "to_digest": plan["to_digest"],
        "old_native_decimal": str(old_native_state),
        "witness_decimal": str(witness_bits),
        "new_native_decimal": str(new_native),
        "additional_bits": d,
        "scope": SCOPE,
    }
    return {**body, "digest": _hash(body)}
