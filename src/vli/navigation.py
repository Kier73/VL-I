"""Certified exact-observation navigation: the first intentionally narrow B3 seed.

A retained, independent-model-compared result can be addressed by the exact
same pretrained Model Ground and input token sequence. Navigation retrieves a
previously observed next-token result with no new weight ranges or neural
forward pass. This does NOT predict new prompts, prove universal model
equivalence or make the original cold model computation disappear.

Caller must retain the expected record digest outside the untrusted cache.
SHA-256 alone is integrity relative to a trusted digest, not authentication.
"""
from __future__ import annotations

from .ground import _digest, _require, _hex64, verify_ground
from .realization import verify_realization

SCHEMA = "vl-i.exact-continuation-address.v1"
OBS_SCHEMA = "vl-i.exact-continuation-observation.v1"


def commit_continuation(ground, frontier, realization):
    """Create an address and receipt for one *already observed* greedy result."""
    verify_ground(ground)
    verify_realization(ground, frontier, realization)
    _require(frontier.get("targets") == ["vocab_logits"], "Only full logits may certify next token")
    _require(realization.get("independent_logit_max_abs_error", 1) <= 2e-4,
             "No qualified numerical reference comparison")
    address = {
        "schema": SCHEMA,
        "model_ground_digest": ground["ground_digest"],
        "source_model": ground["source"]["model"],
        "source_revision": ground["source"]["revision"],
        "typed_dag_digest": ground["interpretation"]["dag_digest"],
        "frontier_digest": frontier["frontier_digest"],
        "input_token_ids": list(realization["input_token_ids"]),
        "observation": "greedy_next_token_exact_input",
        "precision": "VL64_F64_realization_from_original_pretrained_BF16",
    }
    address["address_digest"] = _digest(SCHEMA, address)
    record = {
        "schema": OBS_SCHEMA,
        "address": address,
        "selected_next_token": realization["selected_next_token"],
        "upstream_full_logits_digest": realization["full_logits_f64_sha256"],
        "verified_native_realization_digest": realization["realization_digest"],
        "source_weight_bytes_in_original_realization":
            realization["source_payload_bytes_read_during_realization"],
        "source_ranges_in_original_realization": realization["range_reads"],
        "scope": "exact_pinned_input_observation_reuse_not_unseen_prompt_prediction",
    }
    record["observation_digest"] = _digest(OBS_SCHEMA, record)
    return record


def navigate_continuation(ground, frontier, record, *, input_token_ids,
                          expected_observation_digest):
    """Navigate to a retained result, with externally pinned record integrity.

    Ordinary hash/lookup computation still occurs. Zero refers only to
    *new trained-weight reads* and *new neural forward operations*.
    """
    verify_ground(ground)
    _require(isinstance(record, dict) and record.get("schema") == OBS_SCHEMA,
             "No retained continuation observation")
    _require(_hex64(expected_observation_digest), "Missing independent expected record head")
    actual = _digest(OBS_SCHEMA, {k:v for k,v in record.items()
                                if k != "observation_digest"})
    _require(record.get("observation_digest") == actual
             and actual == expected_observation_digest,
             "Stale or altered retained observation")
    address = record.get("address")
    _require(isinstance(address, dict) and address.get("schema") == SCHEMA,
             "Invalid continuation address")
    a_digest = _digest(SCHEMA, {k:v for k,v in address.items()
                               if k != "address_digest"})
    _require(address.get("address_digest") == a_digest, "Altered continuation address")
    _require(address.get("model_ground_digest") == ground["ground_digest"]
             and address.get("source_model") == ground["source"]["model"]
             and address.get("source_revision") == ground["source"]["revision"]
             and address.get("typed_dag_digest") == ground["interpretation"]["dag_digest"]
             and address.get("frontier_digest") == frontier.get("frontier_digest")
             and frontier.get("targets") == ["vocab_logits"],
             "Model/intention relation is no longer the admitted source Ground")
    _require(type(input_token_ids) in (tuple, list)
             and list(input_token_ids) == address.get("input_token_ids"),
             "Unobserved continuation cannot be inferred from retained result")
    _require(address.get("observation") == "greedy_next_token_exact_input"
             and address.get("precision") == "VL64_F64_realization_from_original_pretrained_BF16",
             "Unsupported decoding or precision change")
    vocab = ground["interpretation"]["architecture"]["vocab"]
    selected = record.get("selected_next_token")
    _require(type(selected) is int and 0 <= selected < vocab
             and _hex64(record.get("upstream_full_logits_digest"))
             and _hex64(record.get("verified_native_realization_digest")),
             "No qualified retained output")
    return {
        "status": "RETAINED_EXACT_OBSERVATION",
        "tier": "B3_EXACT_OBSERVATION_NAVIGATION_ONLY",
        "selected_next_token": selected,
        "address_digest": address["address_digest"],
        "evidence_head": actual,
        "new_pretrained_weight_bytes_acquired": 0,
        "new_neural_forward_operations": 0,
        "avoided_original_range_reads": record["source_ranges_in_original_realization"],
        "scope": "same_model_same_input_same_precision_greedy_result_reuse",
    }
