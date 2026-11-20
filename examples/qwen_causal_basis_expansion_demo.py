"""SHAPE-010: native causal four-state simplex, fixture or original Qwen27B.

The original run is opt-in and acquires the pinned source weights. It uses
the same complete native VL-64 two-layer source transition as SHAPE-009.
No output here claims independent PyTorch complete-layer model parity.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

from vli.causal_hidden_manifold import CausalHiddenManifoldGround, TOL
from vli.causal_basis_expansion import ExpandedCausalGround
from vli.large_model_comparison import SOURCES
from vli.qwen_layer_chain import QwenCausalGDNChain

IDS = (63204, 628, 1761, 33303)


def fixture_sources():
    from test_qwen_layer_chain import two_layer_fixture, executor
    from vli.qwen_all_heads import QwenVectorizedCompleteLayer

    nav = two_layer_fixture()
    chain = QwenCausalGDNChain(executor(nav, QwenVectorizedCompleteLayer))
    reports = []
    for token_id, hidden in zip(IDS, (
            [.5,-.2,.3], [-.1,.2,.8], [.7,.6,-.3], [.3,-.4,.65])):
        _, receipt = chain.step(hidden, token_id=token_id,
                                capture_intermediates=True)
        reports.append(receipt)
    norm_raw, desc = nav.get_raw(
        "model.language_model.layers.1.input_layernorm.weight")
    parent = CausalHiddenManifoldGround(
        reports=reports[:3], native_norm_raw=norm_raw,
        native_norm_dtype=desc["dtype"], model=nav.index["model"],
        revision=nav.index["revision"],
        index_sha256=nav.index["index_sha256"],
        index_ground_digest=nav.index["ground_digest"])
    return parent, reports[3], {
        "mode": "FABRICATED_THREE_WIDE_NATIVE_TWO_LAYER_FIXTURE",
        "original_pretrained": False, "source_token_ids": list(IDS),
        "original_complete_decoder": False,
        "independent_pytorch_layer_oracle": False}


def original_sources():
    from experiments.model_fabric.scaled_model_fabric import (
        QualifiedShardNavigator, MAX_INDEX_BYTES)
    from experiments.model_fabric.scaled_model_lab import _url, _fetch_metadata
    from vli.operative_model import ground as model_ground
    from vli.pinned_text_ground import (
        admit, text_intention, tokenize, validate_observation,
        MAX_ASSET_BYTES, TOKENIZER_SOURCES)
    from vli.token_embedding_ground import token_intention, observe_token_row
    from vli.qualified_projection_host import HOST_NAME, BoundedModelRangeHost
    from vli.qwen_all_heads import QwenVectorizedCompleteLayer
    from examples.qwen_layer_chain_frontier import DEFAULT_TEXT

    cfg = SOURCES["qwen-27b"]
    model, revision = cfg["model"], cfg["revision"]
    started = perf_counter_ns()
    index = _fetch_metadata(
        _url(model, revision, "model.safetensors.index.json"),
        MAX_INDEX_BYTES)
    if sha256(index).hexdigest() != cfg["index_sha256"]:
        raise ValueError("Original Qwen pinned index changed")
    host = BoundedModelRangeHost(
        model=model, revision=revision, block_bytes=8*1024*1024,
        budget_bytes=128*1024*1024)
    n = QualifiedShardNavigator(
        model=model, revision=revision, index_raw=index,
        range_host=host, approved_hosts={HOST_NAME})
    model_g = model_ground("qwen-27b", n.index)
    tok = TOKENIZER_SOURCES["qwen-27b"]
    if tok["repository"] != model or tok["revision"] != revision:
        raise ValueError("Original Qwen token source differs")
    tokenizer_url = ("https://huggingface.co/" + model + "/resolve/" +
                     revision + "/tokenizer.json")
    raw_tokenizer = _fetch_metadata(tokenizer_url, MAX_ASSET_BYTES)
    text_ground, tokenizer = admit(model_g, raw_tokenizer)
    intention = text_intention(text_ground, DEFAULT_TEXT)
    observed = tokenize(text_ground, tokenizer, intention)
    validate_observation(model_g, text_ground, intention, observed)
    token_ids = tuple(observed["token_ids"][:4])
    if token_ids != IDS:
        raise ValueError("Original pretrained tokenization differs from SHAPE-009")
    chain = QwenCausalGDNChain([
        QwenVectorizedCompleteLayer(
            n, layer=idx, key_heads=16, value_heads=48,
            key_dim=128, value_dim=128, budget_bytes=2_000_000_000)
        for idx in (0,1)])
    reports = []
    for token_id in token_ids:
        embedding, embedding_witness = observe_token_row(
            n, model_g, token_intention(model_g, token_id))
        _, receipt = chain.step(
            embedding, token_id=token_id, capture_intermediates=True)
        if receipt["original_token_id"] != token_id:
            raise ValueError("Original source token embedding not bound to receipt")
        reports.append(receipt)
    norm_raw, desc = n.get_raw(
        "model.language_model.layers.1.input_layernorm.weight")
    source_end = perf_counter_ns()
    parent = CausalHiddenManifoldGround(
        reports=reports[:3], native_norm_raw=norm_raw,
        native_norm_dtype=desc["dtype"], model=n.index["model"],
        revision=n.index["revision"],
        index_sha256=n.index["index_sha256"],
        index_ground_digest=n.index["ground_digest"])
    return parent, reports[3], {
        "mode": "PINNED_ORIGINAL_QWEN27B_FOUR_NATIVE_TWO_LAYER_POSITIONS",
        "original_pretrained": True,
        "model": model, "revision": revision,
        "source_token_ids": list(token_ids),
        "tokenizer_sha256": text_ground["asset_sha256"],
        "tokenized_intention_digest": observed["observation_digest"],
        "final_original_two_layer_chain_head": chain.head,
        "source_bytes_and_ranges": host.stats(),
        "original_source_and_four_causal_steps_ns": source_end-started,
        "original_complete_decoder": False,
        "independent_pytorch_layer_oracle": False,
        "native_complete_layers_0_and_1": True}


def verify(parent, fourth, metadata):
    old = parent.checkpoint()
    start = perf_counter_ns()
    grown = ExpandedCausalGround(
        parent, fourth, expected_parent_checkpoint_digest=old["checkpoint_digest"])
    admission_ns = perf_counter_ns()-start
    assert grown.original_pretrained == metadata["original_pretrained"]
    assert grown.parent_checkpoint == old
    sparse = tuple(dict.fromkeys((0,1,2,grown.width//2,grown.width-1)))
    unseen = ((.13,.27,.11),(.23,.17,.41),(.08,.11,.73))
    staged = []
    for p in unseen:
        start=perf_counter_ns()
        result=grown.generate(*p,sparse,expected_head=grown.head)
        elapsed=perf_counter_ns()-start
        staged.append((p,result,elapsed))
    heldouts = []
    for p, formed, elapsed in staged:
        start=perf_counter_ns()
        actual=grown.native_reference(*p,sparse)
        native_ns=perf_counter_ns()-start
        err=max(abs(a-b) for a,b in zip(actual,formed["values"]))
        if err>TOL:
            raise ValueError("New causal source simplex fails held-out native reference")
        heldouts.append({
            "unseen_simplex_intention": list(p),
            "requested_indices": list(sparse),
            "position": formed["position"],
            "generated_before_native_reference": True,
            "generated_ns": elapsed, "native_reference_ns": native_ns,
            "max_native_output_error": err,
            "native_calls_during_generation": 0})
    saved=grown.checkpoint()
    start=perf_counter_ns()
    restored=ExpandedCausalGround.restore(
        saved,expected_head=saved["checkpoint_digest"])
    recovery_ns=perf_counter_ns()-start
    future=(.19,.31,.22)
    fresh=restored.generate(*future,sparse,expected_head=restored.head)
    new_err=max(abs(a-b) for a,b in zip(
        fresh["values"],grown.native_reference(*future,sparse)))
    if new_err>TOL:
        raise ValueError("Source-free restored Ground cannot generate unseen simplex")
    full=(.21,.14,.33)
    start=perf_counter_ns()
    formed=restored.generate(*full,tuple(range(grown.width)),
                             expected_head=restored.head)
    full_generated_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    native=grown.native_reference(*full,tuple(range(grown.width)))
    native_full_ns=perf_counter_ns()-start
    full_err=max(abs(a-b) for a,b in zip(formed["values"],native))
    if full_err>TOL:
        raise ValueError("Full-width normalization control contradicts Ground")
    return {
        "schema":"vl-i.shape010-original-causal-rank-three.v1",
        **metadata,
        "status":"MEASURED",
        "source_revision":4,
        "original_parent_ground_unchanged":parent.checkpoint()==old,
        "parent_ground_digest":parent.ground_digest,
        "parent_checkpoint_digest":old["checkpoint_digest"],
        "new_ground_digest":grown.ground_digest,
        "new_ground_head":grown.head,
        "parent_qualifier":parent.proof["admitted_family"],
        "old_rank_two_residual":parent.off_line_residual,
        "real_fourth_observation_rank_expansion":
            grown.rank_probe["relative_residual"],
        "true_fourth_original_causal_head":
            grown.fourth_witness["chain_head"],
        "true_fourth_layer0_f64_digest":
            grown.fourth_witness["layer0_output_f64_digest"],
        "source_ECPU_page_manifest":
            grown.fourth_object["manifest_digest"],
        "fit_coefficients":list(grown.coefficients),
        "independent_gram_coefficients":
            grown.proof["independent_gram_coefficients"],
        "hypothesis_scores":
            grown.proof["candidate_max_output_error"],
        "invalid_energy_candidates":
            grown.proof["invalid_energy_candidates"],
        "native_training_and_challenge_normalizations":14,
        "source_and_native_fit_admission_ns":admission_ns,
        "held_out_generated_intentions":heldouts,
        "source_free_checkpoint_digest":saved["checkpoint_digest"],
        "checkpoint_serialized_bytes":len(json.dumps(saved)),
        "source_free_recovery_ns":recovery_ns,
        "source_free_unseen_mixture":list(future),
        "recovered_unseen_native_error":new_err,
        "full_width_negative_control":{
            "coordinates":grown.width,
            "generated_ns":full_generated_ns,
            "native_full_normalization_ns":native_full_ns,
            "max_error":full_err,
            "generated_slower":full_generated_ns>native_full_ns},
        "native_calls_during_generated_observer_projections":0,
        "future_causal_state_generated":False,
        "complete_layer_generated":False,
        "independent_pytorch_whole_layer_parity":False}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("fixture","live"),required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        sources=fixture_sources if args.mode=="fixture" else original_sources
        result=verify(*sources())
    except (OSError,TimeoutError,RuntimeError,ValueError,MemoryError,
            ImportError,OverflowError,AssertionError) as exc:
        result={"schema":"vl-i.shape010-original-causal-rank-three.v1",
                "status":"UNRESOLVED","mode":args.mode,
                "reason":str(exc)[:400],
                "error_type":type(exc).__name__,
                "original_pretrained_rank_three_not_claimed":True}
        target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print("VL_I_SHAPE010 "+json.dumps(result,sort_keys=True))
        return 1
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("VL_I_SHAPE010 "+json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
