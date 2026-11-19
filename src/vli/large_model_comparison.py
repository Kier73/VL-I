"""Evidence-gated large-model comparison: VL-64 B1 vs VL-I B2/B3.

No model weight or provider credential is needed to inspect *existing* evidence.
An archived partial operator receipt is not full pretrained inference. A provider
chat alias does not attest the exact underlying checkpoint. A fixture is not a
live model observation. These distinctions are executable admission conditions.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from .ground import _digest, _hex64, _require, load_json

SCHEMA = "vl-i.large-model-comparison.v1"
GROUND_SCHEMA = "vl-i.large-model-evidence-ground.v1"
BASELINE_COMMIT = "fea0d29d5a8a1edb03d9c66edd5ca4b15a1395d3"
ARCHIVE_DIR = "experiments/model_fabric/datasets"
SOURCES = {
    "qwen-27b": {
        "archive_key": "qwen", "model": "Qwen/Qwen3.8-27B",
        "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        "index_sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
        "index_ground_digest": "059c388b0818e297d54b9ce322312cb93eab30920f0ba250bcbe46839b6f36f3",
        "type": "qwen3_5", "layers": 64, "tensors": 1199, "shards": 18,
        "advertised_bytes": 55562855904,
    },
    "deepseek-v41-flash": {
        "archive_key": "deepseek", "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "index_sha256": "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
        "index_ground_digest": "a449b268199162da03bd2d5b9ac",
        "type": "deepseek_v41", "layers": 40, "tensors": 96085, "shards": 48,
        "advertised_bytes": 510286023000,
    },
}
# Correct the full upstream DeepSeek index ground identity rather than allowing
# a shortened or self-asserted digest to pass.
SOURCES["deepseek-v41-flash"]["index_ground_digest"] = (
    "a449b268199162da03bd2d5b8d8471297d630e7f9446864fc8df86257cb5b9ac"
)
TASKS = ("paragraph", "python", "tool_json")
_OPERATORS = {
    "qwen-27b": [
        ("tokenizer", "Exact pinned tokenizer and native chat-template parity", False),
        ("embedding", "Full trained embedding of this prompt's token IDs", False),
        ("gdn", "All 48 Gated DeltaNet layers, complete 48-head recurrent state and FFNs", False),
        ("full_attention", "All 16 full-attention layers with native RoPE/gating and FFNs", False),
        ("logits", "Final norm, vocabulary head and exact full-token logits", False),
        ("decode", "Causal continuation, deterministic decoding and stopping", False),
    ],
    "deepseek-v41-flash": [
        ("tokenizer", "Exact pinned tokenizer and native chat-template parity", False),
        ("embedding", "Full trained embedding of this prompt's token IDs", False),
        ("attention", "All 40 compressed/sparse attention layers and causal state", False),
        ("moe", "Full six selected FP4 experts plus shared experts at each token/layer", False),
        ("engram_mhc", "Engram, mHC and all additional declared model transitions", False),
        ("logits", "Final norm, vocabulary head and exact full-token logits", False),
        ("decode", "Causal continuation, deterministic decoding and stopping", False),
    ],
}


def _plain_numeric(value, label):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "Unqualified numeric evidence: " + label)
    return float(value)


def _match_identity(value, model, src):
    _require(isinstance(value, dict) and
             value.get("model") == src["model"] and
             value.get("revision") == src["revision"],
             "Actual model/revision differs from pinned large-scale Ground")


def admit_archived_subgraphs(structural, trained, replays):
    """Qualify prior *reports* for provenance and strictly partial numeric scope.

    This checks observed report contracts against externally pinned VL-64
    coordinates; it does NOT independently recompute the pretrained mathematics
    from the archived weights, which the reports do not contain.
    """
    _require(structural.get("schema") == "vl64.architecture-scale-observations.v1"
             and trained.get("schema") == "vl64.saved-large-trained-subgraph-evidence.v1"
             and replays.get("schema") == "vl64.real-pretrained-glance-replay-archive.v1",
             "Unexpected archived pretrained evidence schema")
    result = {}
    for route, src in SOURCES.items():
        key = src["archive_key"]
        snapshot = structural["models"][key]["report"]
        calc = trained[key]
        replay = replays["observations"][key]
        for row in (snapshot, calc, replay):
            _match_identity(row, route, src)
        _require(snapshot.get("model_type") == src["type"]
                 and snapshot.get("layers") == src["layers"]
                 and snapshot.get("mapped_tensors") == src["tensors"]
                 and snapshot.get("shards") == src["shards"]
                 and snapshot.get("advertised_payload_bytes") == src["advertised_bytes"]
                 and snapshot.get("index_sha256") == src["index_sha256"]
                 and snapshot.get("index_digest") == src["index_ground_digest"],
                 "Pinned full-scale structural inventory changed")
        _require(calc.get("index_sha256") == src["index_sha256"]
                 and calc.get("index_ground_digest") == src["index_ground_digest"]
                 and type(calc.get("range_count")) is int and calc["range_count"] > 0
                 and type(calc.get("weight_bytes")) is int and calc["weight_bytes"] > 0
                 and _hex64(calc.get("range_chain")),
                 "Missing actual trained numerical range provenance")
        err = _plain_numeric(calc.get("max_abs_independent_error"), "operator reference")
        _require(err <= 2e-5 and calc.get("input_scope") == "synthetic_hidden",
                 "Reference mismatch or a falsely attributed real prompt")
        _require(replay.get("status") == "REAL_PINNED_GLANCE_NUMERICAL_REPLAY_VERIFIED"
                 and _hex64(replay.get("portable_ground_head"))
                 and type(replay.get("original_remote_ranges")) is int
                 and replay["original_remote_ranges"] > 0
                 and replay.get("replayed_remote_ranges") == 0,
                 "Missing strict retained-observation replay")
        if key == "qwen":
            _require(snapshot.get("layer_families") ==
                     {"linear_attention": 48, "full_attention": 16}
                     and calc.get("status") ==
                     "REAL_QWEN_GATED_DELTA_TRAINED_HEAD_VERIFIED"
                     and calc.get("layer") == 0 and calc.get("value_head") == 0
                     and type(calc.get("output_row0_head_contribution")) in (float, int)
                     and replay.get("original_observation", {}).get("output") ==
                     replay.get("retained_replay", {}).get("output"),
                     "Qwen partial recurrent head does not match archived reference")
            evidence = ["layer_0_single_gated_delta_head", "partial_out_projection"]
            replay_status = "QWEN_ONE_TRAINED_HEAD_REPLAYED"
            weight_bytes = calc["weight_bytes"]
        else:
            slices = calc.get("selected_fp4_slices", {})
            selected = calc.get("selected_experts")
            _require(snapshot.get("layer_families") == {"compressed_sparse_attention_moe": 40}
                     and calc.get("status") == "REAL_DEEPSEEK_TRAINED_384_ROUTER_VERIFIED"
                     and slices.get("status") ==
                     "REAL_DEEPSEEK_SELECTED_FP4_SWIGLU_SLICES_VERIFIED"
                     and selected == slices.get("selected_experts")
                     and isinstance(selected, list) and len(selected) == 6
                     and len(set(selected)) == 6
                     and len(slices.get("slices", [])) == 6
                     and {z.get("expert") for z in slices["slices"]} == set(selected)
                     and type(slices.get("weight_bytes")) is int
                     and slices["weight_bytes"] >= calc["weight_bytes"]
                     and _plain_numeric(slices.get("max_abs_independent_error"),
                         "FP4 independent reference") <= 2e-5
                     and replay.get("original_observation", {}).get("selected_experts")
                     == replay.get("retained_replay", {}).get("selected_experts")
                     == selected,
                     "DeepSeek selected-expert numeric observation incomplete")
            evidence = ["layer_0_384_way_router", "six_selected_expert_fp4_row_fragments"]
            replay_status = "DEEPSEEK_SELECTED_ROW_FRAGMENTS_REPLAYED"
            weight_bytes = slices["weight_bytes"]
        info = {
            "schema": GROUND_SCHEMA, "route": route,
            "model": src["model"], "pinned_revision": src["revision"],
            "index_sha256": src["index_sha256"], "index_ground_digest": src["index_ground_digest"],
            "architecture": src["type"], "layers": src["layers"],
            "full_model_advertised_weight_bytes": src["advertised_bytes"],
            "archived_trained_weight_bytes_read": weight_bytes,
            "numeric_observation_scope": "synthetic_hidden_partial_subgraph",
            "upstream_reported_reference_error": err,
            "numeric_replay": replay_status, "observed_partial_relations": evidence,
            "pretrained_forward_completed": False,
            "complete_full_vocab_logits_observed": False,
            "original_remote_range_count": replay["original_remote_ranges"],
            "replay_additional_remote_range_count": 0,
            "source_commit": BASELINE_COMMIT,
            "source_trust": "immutable_GitHub_commit_not_independent_weight_signature",
        }
        info["ground_digest"] = _digest(GROUND_SCHEMA, info)
        result[route] = info
    return result


def _verify_b1_report(b1):
    """Admit the B1 report without promoting protocol-only fixtures to inference."""
    from experiments.model_fabric.operational_ground import digest as source_digest
    if b1 is None:
        return "NO_PROVIDER_RUN", {}
    _require(isinstance(b1, dict)
             and b1.get("schema") == "vl64.remote-large-cross-process-benchmark.v1"
             and b1.get("report_digest") == source_digest(
                 b1["schema"], {k:v for k,v in b1.items() if k != "report_digest"}),
             "Unqualified B1 benchmark provenance")
    mode = b1.get("mode")
    status = b1.get("status")
    _require((mode, status) in (
        ("FIXTURE_ONLY", "CONTROL_PLANE_FIXTURE_BENCHMARK"),
        ("LIVE_PROVIDER", "REAL_PROVIDER_CHAT_BENCHMARK")),
        "B1 cannot assert live inference from a fixture")
    records = b1.get("instances")
    _require(isinstance(records, list) and len(records) == len(SOURCES)
             and {z.get("route") for z in records} == set(SOURCES),
             "Missing one independently isolated large-model route")
    for record in records:
        _require(record.get("cross_process_checkpoint_verified") is True
                 and record.get("external_provider_KV_recovered") is False
                 and len(record.get("epochs", [])) == 2
                 and record["epochs"][0].get("stage") == "start"
                 and record["epochs"][1].get("stage") == "continue"
                 and record["epochs"][0].get("mode") ==
                     ("LIVE_PROVIDER" if mode == "LIVE_PROVIDER" else "FIXTURE_ONLY"),
                 "Unqualified B1 recovery or provider-mode observation")
    return ("LIVE_B1_PROVIDER_OBSERVED" if mode == "LIVE_PROVIDER"
            else "B1_CONTROL_FIXTURE_ONLY"), {r["route"]: r for r in records}


def comparison(structural, trained, replays, *, b1=None):
    """Compute a falsifiable, task-aligned gap report for *unscaled* models.

    B1 is opaque chat serving; B2 is only partial trained math on synthetic
    input. Until both use the same exact snapshot, tokenizer, prompt and
    complete outputs, declaring parity or a speedup is prohibited.
    """
    from experiments.model_fabric.remote_large_fleet import TASKS, ground, intention
    source_grounds = admit_archived_subgraphs(structural, trained, replays)
    b1_grade, observed_b1 = _verify_b1_report(b1)
    result = {
        "schema": SCHEMA,
        "status": "QUALIFIED_EVIDENCE_GAPS_NOT_FULL_MODEL_PARITY",
        "large_models_preserved": True,
        "vl64_reference_commit": BASELINE_COMMIT,
        "b1_observation_grade": b1_grade,
        "b2_execution_grade": "REAL_PRETRAINED_PARTIAL_SUBGRAPHS_ONLY",
        "same_model_same_intention_full_output_parity": "UNRESOLVED",
        "models": {},
        "comparison_scope": "B1 remote chat vs B2 partial pretrained operator probes; not directly comparable compute costs",
    }
    for route, source in source_grounds.items():
        obligations = [
            {"operation": operation, "required_evidence": why,
             "resolved_by_archived_B2_evidence": proved}
            for operation, why, proved in _OPERATORS[route]
        ]
        # The archived B2 reads contain genuine trained bytes but do not cover
        # the complete prompt-dependent input or output graph.
        _require(all(not o["resolved_by_archived_B2_evidence"] for o in obligations),
                 "Do not accidentally promote a partial head/expert to a full layer")
        tasks = []
        for kind in ("paragraph", "python", "tool_json"):
            spec = intention(ground(route), kind,
                 [{"role": "user", "content": TASKS[kind]["prompt"]}], max_tokens=128)
            tasks.append({
                "kind": kind, "exact_b1_intention_digest": spec["intent_digest"],
                "prompt_sha256": _digest("vl-i.large-model-intent-prompt.v1", TASKS[kind]["prompt"]),
                "b1_result_available": b1_grade == "LIVE_B1_PROVIDER_OBSERVED",
                "b2_real_prompt_tokens_observed": False,
                "b2_complete_logits_observed": False,
                "b2_response_quality_measured": False,
                "parity": "UNRESOLVED_DIFFERENT_OBSERVATION_SCOPE",
            })
        b1_metrics = None
        if route in observed_b1:
            r = observed_b1[route]
            b1_metrics = {
                "mode": b1_grade,
                "vl_interpreter_cpu_seconds": r.get("sum_interpreter_process_cpu_seconds"),
                "external_model_request_wall_seconds": r.get("sum_provider_wall_seconds"),
                "completion_tokens_per_provider_wall_second": r.get(
                    "completion_tokens_per_provider_wall_second"),
                "checkpoint_continuity_observed": True,
                "provider_KV_state_recovered": False,
                "underlying_pinned_weight_snapshot_attested": False,
            }
        result["models"][route] = {
            "ground": source,
            "task_intentions": tasks,
            "b1_metrics": b1_metrics,
            "b2_missing_required_operations": obligations,
            "nearest_unresolved_frontier": {
                "first": "pinned_tokenizer_and_chat_template",
                "then": "prompt_embeddings_and_all_model_layers",
                "terminal": "complete_vocab_logits_then_multitoken_generation",
            },
            "eligibility_for_B2_full_model": False,
            "eligibility_for_B3_predictive_shortcut": False,
            "possible_comparable_metric": (
                "none_until_same_immutable_snapshot_prompt_and_complete_output_observed"),
        }
    result["report_digest"] = _digest(SCHEMA, result)
    return result


def run(vl64_root, *, b1_path=None, require_pinned_checkout=True):
    path = Path(vl64_root)
    if require_pinned_checkout:
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("Missing immutable upstream VL-64 checkout") from exc
        _require(sha == BASELINE_COMMIT, "Pinned VL-64 source commit differs")
    structural = load_json(path / ARCHIVE_DIR / "2026-11-19-large-model-frontier-observations.json")
    trained = load_json(path / ARCHIVE_DIR / "2026-11-19-qwen-deepseek-trained-subgraphs.json")
    replays = load_json(path / ARCHIVE_DIR / "2026-11-19-real-trained-weight-glance-replays.json")
    b1 = load_json(b1_path) if b1_path is not None else None
    return comparison(structural, trained, replays, b1=b1)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ground exact 27B/DeepSeek B1 vs partial VL-native B2 evidence")
    parser.add_argument("--vl64-root", required=True)
    parser.add_argument("--b1-report", help="Optional B1 cross-process fixture or live receipt")
    parser.add_argument("--out", default="-")
    args = parser.parse_args(argv)
    try:
        report = run(args.vl64_root, b1_path=args.b1_report)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, "Evidence refused: " + type(exc).__name__ + ": " + str(exc) + "\n")
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.out == "-":
        print(payload, end="")
    else:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
