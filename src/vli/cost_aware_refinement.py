"""TDL-003: bounded exact-cost observation planning and VGI-qualified acquisition.

The mathematical row plan can be formed before observation values are known.
An accepted VGI ledger supplies policy-finalized bit observations; its sources
are receiver-attributed and remain distinct from cryptographic attestation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re

from .minimal_refinement import _express
from .problem_formed_interpreter import _hash, verify_interpreter

SCHEMA = "vl-i.cost-aware-observation-plan.v1"
RECEIPT = "vl-i.vgi-qualified-refinement.v1"
GROUND_ID = "pfi-refinement-observation"
SOURCE_STATE = "refinement"
SCOPE = "declared-source-row;VGI-policy-qualified;source-authentication-external"
MAX_OFFERS = 12
MAX_COST = 1_000_000


@dataclass(frozen=True)
class ObservationOffer:
    """One source GF(2) parity row offered by a named observation locus."""
    offer_id: str
    row: int
    cost: int
    locus: str

    def __post_init__(self) -> None:
        if (type(self.offer_id) is not str or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_-]{0,63}", self.offer_id)):
            raise ValueError("Invalid observation offer ID")
        if type(self.locus) is not str or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_.:/-]{0,127}", self.locus):
            raise ValueError("Invalid observer locus")
        if type(self.row) is not int or self.row <= 0:
            raise ValueError("An observation offer needs a nonzero GF(2) row")
        if type(self.cost) is not int or not 1 <= self.cost <= MAX_COST:
            raise ValueError("Observation cost must be a bounded positive integer")

    def encode(self) -> dict:
        return {
            "offer_id": self.offer_id, "row_decimal": str(self.row),
            "cost": self.cost, "locus": self.locus,
        }


def _scope_check(old_source, old_quotient, old_bundle,
                 new_source, new_quotient, new_bundle) -> None:
    if not (
        verify_interpreter(old_source, old_quotient, old_bundle)
        and verify_interpreter(new_source, new_quotient, new_bundle)
    ):
        raise ValueError("Both interpreters must pass mathematical admission")
    if not (
        old_source.width == new_source.width
        and old_source.generators == new_source.generators
        and old_source.origin == new_source.origin
    ):
        raise ValueError("Different source-law epoch")


def _covers(old_rows: tuple[int, ...], new_rows: tuple[int, ...],
            selected: tuple[ObservationOffer, ...]) -> bool:
    basis = old_rows + tuple(offer.row for offer in selected)
    return all(_express(row, basis) is not None for row in new_rows)


def plan_cost_aware_observations(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    offers: tuple[ObservationOffer, ...],
) -> dict:
    """Find globally minimum total offered cost on a declared <=12-offer set.

    The offers may be outside the target quotient span. Two cheap independent
    external rows can reconstruct an expensive desired row through XOR, so
    minimizing number of measured bits and cost are different objectives.
    """
    _scope_check(old_source, old_quotient, old_bundle,
                 new_source, new_quotient, new_bundle)
    if (type(offers) is not tuple or len(offers) > MAX_OFFERS
            or any(not isinstance(v, ObservationOffer) or
                   v.row >= (1 << old_source.width) for v in offers)):
        raise ValueError("Observation offers must be bounded, typed source rows")
    ordered = tuple(sorted(offers, key=lambda item: item.offer_id))
    if len({item.offer_id for item in ordered}) != len(ordered):
        raise ValueError("Duplicate offer identity")

    old_rows = tuple(old_quotient.basis_rows)
    new_rows = tuple(new_quotient.basis_rows)
    best: tuple[tuple, tuple[ObservationOffer, ...]] | None = None
    # Exhaustive exact search; budget is explicit and never described as
    # polynomial-time optimal across unbounded observer catalogues.
    for mask in range(1 << len(ordered)):
        chosen = tuple(row for j, row in enumerate(ordered) if mask & (1 << j))
        score = (sum(item.cost for item in chosen), len(chosen),
                 tuple(item.offer_id for item in chosen))
        if best is not None and score >= best[0]:
            continue
        if _covers(old_rows, new_rows, chosen):
            best = score, chosen
    if best is None:
        raise ValueError("Available observer offers do not span the new intention")

    chosen = best[1]
    basis = old_rows + tuple(item.row for item in chosen)
    code_rows = [_express(row, basis) for row in new_rows]
    if any(x is None for x in code_rows):
        raise AssertionError("Selected observer basis does not reconstruct new coordinate")

    union = list(old_rows)
    deficit = 0
    for row in new_rows:
        if _express(row, tuple(union)) is None:
            deficit += 1
            union.append(row)

    body = {
        "schema": SCHEMA,
        "old_source_sha256": old_source.digest,
        "new_source_sha256": new_source.digest,
        "from_digest": old_bundle["digest"],
        "to_digest": new_bundle["digest"],
        "source_width": old_source.width,
        "old_bits": old_quotient.dimension,
        "new_bits": new_quotient.dimension,
        "minimum_extra_rank": deficit,
        "offers": [v.encode() for v in ordered],
        "selected_offer_ids": [v.offer_id for v in chosen],
        "selected_rows_decimal": [str(v.row) for v in chosen],
        "mapping_rows_decimal": [str(x) for x in code_rows],
        "observation_count": len(chosen),
        "total_cost": best[0][0],
        "scope": SCOPE,
    }
    plan = {**body, "digest": _hash(body)}
    from .cost_aware_refinement_verify import verify_cost_plan
    if not verify_cost_plan(
        old_source, old_quotient, old_bundle,
        new_source, new_quotient, new_bundle, plan,
    ):
        raise ValueError("Independent optimal-cost observation audit refused")
    return plan


def observation_context(plan: dict, *, old_native_state: int) -> str:
    """Bind a VGI Ground revision to exact plan and retained native coordinate."""
    if (type(plan) is not dict or plan.get("schema") != SCHEMA
            or type(old_native_state) is not int
            or not 0 <= old_native_state < (1 << plan["old_bits"])):
        raise ValueError("Invalid observer context")
    return _hash({
        "schema": "vl-i.refinement-observation-context.v1",
        "plan_digest": plan["digest"],
        "old_native_decimal": str(old_native_state),
    })


def acquire_vgi_refinement(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle, plan,
    *, old_native_state: int, ledger=None,
) -> dict:
    """Read only finalized VGI transition outcomes and reconstruct new state.

    A policy/context match and VGI finalization are enforced, but external
    sensors and their transport authority remain delegated to the receiver.
    """
    from .cost_aware_refinement_verify import verify_cost_plan
    if not verify_cost_plan(
        old_source, old_quotient, old_bundle,
        new_source, new_quotient, new_bundle, plan,
    ):
        raise ValueError("Unverified observation plan")
    context = observation_context(plan, old_native_state=old_native_state)
    observations: list[dict] = []
    if plan["observation_count"] == 0:
        if ledger is not None:
            raise ValueError("Zero-observation plan must not claim external evidence")
        ledger_digest = None
        policy_digest = None
    else:
        from vgi.evidence import TransitionEvidenceLedger
        if not isinstance(ledger, TransitionEvidenceLedger):
            raise ValueError("Observation acquisition requires a VGI evidence ledger")
        if (ledger.policy.ground_id != GROUND_ID
                or ledger.policy.context_digest != context
                or ledger.policy.strict is not True):
            raise ValueError("VGI Ground, state context or strict policy does not match")
        # Force the VGI ledger's own finalization/evidence reconstruction.
        ledger = TransitionEvidenceLedger.from_dict(ledger.to_dict())
        for offer_id in plan["selected_offer_ids"]:
            try:
                decision = ledger.decision(SOURCE_STATE, offer_id)
                if decision.status != "AUTHORIZED":
                    raise ValueError("VGI observation is not finalized AUTHORIZED")
                transition = ledger.transition_observation(SOURCE_STATE, offer_id)
                outcome = transition.outcomes[0]
                if (outcome.target != SOURCE_STATE or len(outcome.output) != 1
                        or type(outcome.output[0]) is not int
                        or outcome.output[0] not in (0, 1)):
                    raise ValueError("VGI outcome is not an exact source parity bit")
                observations.append({
                    "offer_id": offer_id, "bit": outcome.output[0],
                    "decision_digest": decision.digest,
                    "evidence_ids": list(transition.evidence),
                })
            except (KeyError, IndexError) as exc:
                raise ValueError("Unadmitted VGI refinement observation") from exc
        ledger_digest, policy_digest = ledger.fingerprint, ledger.policy.fingerprint

    witness = sum(item["bit"] << i for i, item in enumerate(observations))
    combined = old_native_state | (witness << plan["old_bits"])
    value = sum(
        ((int(row) & combined).bit_count() & 1) << i
        for i, row in enumerate(plan["mapping_rows_decimal"])
    )
    body = {
        "schema": RECEIPT,
        "plan_digest": plan["digest"],
        "old_native_decimal": str(old_native_state),
        "new_native_decimal": str(value),
        "observation_context": context,
        "selected_observations": observations,
        "ledger_digest": ledger_digest,
        "policy_digest": policy_digest,
        "total_cost": plan["total_cost"],
        "scope": SCOPE,
    }
    receipt = {**body, "digest": _hash(body)}
    from .cost_aware_refinement_verify import verify_vgi_refinement_receipt
    if not verify_vgi_refinement_receipt(
        old_source, old_quotient, old_bundle,
        new_source, new_quotient, new_bundle, plan, receipt, ledger=ledger,
    ):
        raise ValueError("Independent VGI-qualified refinement audit refused")
    return receipt
