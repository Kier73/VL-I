"""TIME-009: join independently authenticated TIME-008 observations to E-CPU effects.

This adapter DOES NOT interpret the discovered rational program on E-CPU.
It independently checks the signed source-relative observation and accepts
only a matching, *already quorum-published* native E-CPU IPS-008 result from
a receiver whose separate authority is still current. An exact trusted pair
(space digest, native published root) scopes the attempted effect.

Neither a source signature nor a model digest is a capability. E-CPU owns
mTLS identities, durable handoff barriers, quorum freshness and atomic effects.
"""
from __future__ import annotations

from fractions import Fraction

from .ground import _digest, _require
from .temporal_intelligence import _fraction
from .discovered_state_space import (
    verify_discovered_space, navigate_discovered,
)

SCHEMA = "vl-i.time009.signed-ground-native-effect-join.v1"


def _native():
    """Import E-CPU only for opted-in native authority integrations."""
    from mailcpu.fenced_generated_ground import (
        FencedGroundHead, FencedGroundLineage,
    )
    from mailcpu.fenced_effect_receiver import (
        FencedGroundEffectReceiver, published_effect,
    )
    from mailcpu.effect_receiver_service import AuthenticatedGroundEffectClient
    return (FencedGroundHead, FencedGroundLineage,
            FencedGroundEffectReceiver, AuthenticatedGroundEffectClient,
            published_effect)


def apply_observed_native_effect(
        snapshot, space, *, source_contract,
        expected_space_digest, expected_native_root,
        intended_time, native_lineage, native_head, native_receiver,
        native_resource, previous_space=None):
    """Require BOTH independent proof authorities before one native effect.

    The source is independently checked against the publisher's expected
    Ed25519 head and the receiver's externally pinned Ground checkpoint.
    The caller must separately pin the exact law and E-CPU published root;
    that pair is an application policy input, NOT inferred from a source hash.

    The claimed value must correspond to an actually signed historical
    coordinate. Predictions of future measurements cannot control effects.
    A native E-CPU IPS-007/008 receiver remains the ONLY effect committer.
    It rechecks quorum/plan/epoch/host and serializes the write with fencing.
    """
    _require(type(source_contract) is dict and
             type(expected_space_digest) is str and
             type(expected_native_root) is str and
             type(intended_time) is str and
             type(native_resource) is str,
             "Trusted two-lineage effect policy is incomplete")
    _require(type(space) is dict and
             space.get("intention_space_digest") == expected_space_digest,
             "Application policy does not select this signed processor Ground")
    verify_discovered_space(
        snapshot, space, previous_space=previous_space, **source_contract)
    observed = navigate_discovered(
        space, intended_time, output="current", allow_projection=False)
    _require(observed["evidence_status"] == "SOURCE_OBSERVED_COORDINATE",
             "Only actually authenticated source coordinates may request effects")
    value = _fraction(observed["value"])
    _require(value.denominator == 1,
             "Native integer resource cannot realize a noninteger rational")
    (Head, Lineage, Local, Client, native_effect) = _native()
    _require(isinstance(native_lineage, Lineage) and
             isinstance(native_head, Head) and
             isinstance(native_receiver, (Local, Client)),
             "Only an existing E-CPU native authority and receiver can commit")
    _require(native_head.manifest.digest == expected_native_root,
             "Application policy pins another native quorum publication")
    # Original E-CPU IPS-005 verifies majority high-water and the *entire*
    # original generated-ground execution trace. The current root must be
    # independently read, not accepted from an old caller checkpoint.
    latest = native_lineage.restore_latest()
    _require(latest.manifest.digest == expected_native_root,
             "Old native worker is no longer the currently published owner")
    native_lineage.restore(native_head)
    claim = native_effect(native_head, native_resource)
    _require(claim.value == int(value) and
             claim.manifest_digest == expected_native_root and
             claim.plan_digest == native_lineage.plan.digest,
             "Original E-CPU stage observation and source Ground disagree")

    # The snapshot is diagnostic only: the real receiver does a fresh
    # current epoch/installed root check atomically in apply(). A PREPARE
    # freeze blocks this even when the previous publication was valid.
    state = native_receiver.snapshot()
    _require(state["installed_root"] == expected_native_root,
             "Effect receiver has not activated the selected native root")
    committed = native_receiver.apply(claim)
    _require(type(committed) is dict and
             committed.get("committed") is True and
             committed.get("root") == expected_native_root and
             committed.get("receipt_digest") == claim.receipt_digest and
             committed.get("operation_id") == claim.operation_id and
             type(committed.get("value")) is int and
             committed["value"] == int(value),
             "Native effect receiver did not acknowledge the exact publication")

    body = {
        "schema": SCHEMA,
        "trusted_source_ground_digest": expected_space_digest,
        "trusted_native_published_root": expected_native_root,
        "source_id": space["source_id"],
        "source_signed_journal_head": space["signed_source_journal_head"],
        "source_public_key_sha256": space["source_public_key_sha256"],
        "source_observation_coordinate": intended_time,
        "source_ground_observed_rational": observed["value"],
        "source_navigation_digest": observed["navigation_digest"],
        "source_ground_revision_predecessor": space["previous_space_digest"],
        "native_plan_digest": claim.plan_digest,
        "native_generation": claim.generation,
        "native_epoch": claim.epoch,
        "native_published_host": claim.host_id,
        "native_stage_receipt_digest": claim.receipt_digest,
        "native_receiver_operation_id": claim.operation_id,
        "native_effect_receipt": committed,
        "native_receiver_is_only_effect_authority": True,
        "mathematical_law_native_execution_established": False,
        "physical_source_truth_established": False,
        "remote_host_hardware_attested": False,
    }
    return {**body, "join_digest": _digest(SCHEMA, body)}
