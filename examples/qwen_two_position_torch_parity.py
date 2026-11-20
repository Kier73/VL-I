"""QUALIFIED-010: two *genuine original* trained tokens and causal attention.

This is the first original-checkpoint PyTorch comparison where a
pretrained Qwen layer has >1 distinct causal key/value positions.
Its inputs come from fully recomputed original layers 0..6 on both
native paragraph token IDs, with exact prior trained digest controls.

A strong result needs:
  * exact replay of historical pretrained prefix digests,
  * independent full-layer PyTorch F64 numerical parity at positions 0+1,
  * an observed non-uniform learned causal attention distribution at pos1,
  * nonzero historical V contribution to the resulting attention context.
No final decoder, HF BF16 runtime or networking-native arithmetic claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vli.large_model_comparison import SOURCES
from vli.operative_model import ground
from vli.pinned_text_ground import (
    admit,text_intention,tokenize,validate_observation,
    TOKENIZER_SOURCES,MAX_ASSET_BYTES,
)
from vli.token_embedding_ground import token_intention,observe_token_row
from vli.qualified_projection_host import HOST_NAME,BoundedModelRangeHost
from vli.qwen_architecture_navigator import (
    admit_indexed_qwen_topology,QwenArchitectureCausalPrefix,
)
from vli.qwen_gated_attention import QwenQualifiedGatedAttention
from vli.qwen_torch_oracle import OriginalQwenTorchGQA
from vli.qwen_layer_chain import _f64_transfer_digest
from examples.qwen_model_host_frontier import DEFAULT_TEXT


def run(*,text=DEFAULT_TEXT,
        block_bytes=8*1024*1024,host_budget_bytes=128*1024*1024,
        layer_budget_bytes=2_000_000_000):
    import torch
    from experiments.model_fabric.scaled_model_fabric import (
        QualifiedShardNavigator,MAX_INDEX_BYTES)
    from experiments.model_fabric.scaled_model_lab import _url,_fetch_metadata
    torch.set_num_threads(1)
    source=SOURCES["qwen-27b"]
    model,revision=source["model"],source["revision"]
    raw_index=_fetch_metadata(_url(model,revision,"model.safetensors.index.json"),
                              MAX_INDEX_BYTES)
    raw_config=_fetch_metadata(_url(model,revision,"config.json"),65536)
    host=BoundedModelRangeHost(
        model=model,revision=revision,block_bytes=block_bytes,
        budget_bytes=host_budget_bytes)
    nav=QualifiedShardNavigator(
        model=model,revision=revision,index_raw=raw_index,
        range_host=host,approved_hosts={HOST_NAME})
    plan=admit_indexed_qwen_topology(
        nav,raw_config,expected_config_sha256=source.get("config_sha256"))
    if (plan["layer_count"]!=64 or
        [r["kind"] for r in plan["layers"][:8]]!=
        ["GDN","GDN","GDN","GATED_GQA"]*2):
        raise ValueError("Wrong source-indexed original hybrid topology")
    g=ground("qwen-27b",nav.index)
    tok=TOKENIZER_SOURCES["qwen-27b"]
    if tok["repository"]!=model or tok["revision"]!=revision:
        raise ValueError("Original model tokenizer evidence changed")
    tokenizer_url=(
        f"https://huggingface.co/{model}/resolve/{revision}/tokenizer.json")
    raw_tok=_fetch_metadata(tokenizer_url,MAX_ASSET_BYTES)
    tg,tokenizer=admit(g,raw_tok)
    intention=text_intention(tg,text)
    observed=tokenize(tg,tokenizer,intention)
    validate_observation(g,tg,intention,observed)
    selected=observed["token_ids"][:2]
    if selected!=[63204,628]:
        raise ValueError("Pinned original paragraph tokenization changed")

    root=Path(__file__).resolve().parents[1]/"experiments/qwen_all_heads/datasets"
    earlier_eight=json.loads((root/
        "2026-11-19-real-original-qwen-first-eight-architecture-directed.json"
        ).read_text("utf8"))
    earlier_four=json.loads((root/
        "2026-11-19-real-original-qwen-first-hybrid-block.json"
        ).read_text("utf8"))
    if (earlier_eight["model"]!=model or
        earlier_eight["revision"]!=revision or
        earlier_eight["source_config_sha256"]!=plan["config_sha256"] or
        earlier_eight["source_architecture_plan_digest"]!=plan["plan_digest"]):
        raise ValueError("Earlier trained source identity or plan diverged")

    geom=plan["geometry"]
    seven=QwenArchitectureCausalPrefix(
        nav,plan,stop_layer=7,layer_budget_bytes=layer_budget_bytes)
    qualified=QwenQualifiedGatedAttention(
        nav,layer=7,query_heads=geom["query_heads"],kv_heads=geom["kv_heads"],
        head_dim=geom["head_dim"],rotary_dim=geom["rotary_dim"],
        rope_theta=geom["rope_theta"],eps=geom["eps"],
        budget_bytes=layer_budget_bytes)
    independent=OriginalQwenTorchGQA(nav,plan,layer=7)
    positions=[]
    for position,token_id in enumerate(selected):
        row,embedding=observe_token_row(
            nav,g,token_intention(g,token_id))
        h6,prefix_receipt=seven.step(row,token_id=token_id)
        f64_prefix=[w["output_f64_digest"]
                    for w in prefix_receipt["layer_witnesses"]]
        if len(f64_prefix)!=7:
            raise ValueError("Missing fully executed original trained predecessor")
        if position==0:
            if (f64_prefix!=earlier_eight["original_position_0"][
                "output_f64_digests"][:7] or
                _f64_transfer_digest(row)!=earlier_eight[
                "original_position_0"]["input_f64_digest"]):
                raise ValueError("Historical full original seven-layer result diverged")
        elif f64_prefix[:4]!=earlier_four["positions"][1]["exact_f64_layer_outputs"]:
            raise ValueError("Historical original second-token layers 0–3 diverged")

        torch_result,torch_receipt=independent.next(h6)
        vl_result,vl_receipt=qualified.next(h6)
        if (len(torch_result)!=geom["hidden"] or
            len(vl_result)!=geom["hidden"] or
            independent.position!=position+1 or
            qualified.position!=position+1 or
            torch_receipt["causal_kv_positions"]!=position+1 or
            vl_receipt["kv_history_positions"]!=position+1):
            raise ValueError("Incomplete independently qualified causal attention")
        vl_digest=_f64_transfer_digest(vl_result)
        if position==0 and vl_digest!=earlier_eight[
                "original_position_0"]["output_f64_digests"][7]:
            raise ValueError("Original first-token layer7 no longer reproduces QUALIFIED-008")
        errors=[abs(a-b) for a,b in zip(vl_result,torch_result)]
        max_abs=max(errors)
        max_scaled=max(e/max(1.0,abs(a),abs(b))
                       for e,a,b in zip(errors,vl_result,torch_result))
        rmse=math.sqrt(math.fsum(e*e for e in errors)/len(errors))
        if (not all(math.isfinite(v) for v in (max_abs,max_scaled,rmse)) or
            max_abs>1e-5 or max_scaled>2e-7):
            raise ValueError(
                f"Independent trained attention parity failed at causal position {position}: "
                f"abs={max_abs:.8g}, scaled={max_scaled:.8g}")

        probabilities=torch_receipt["attention_probabilities_per_query_head"]
        if (len(probabilities)!=geom["query_heads"] or
            any(len(p)!=position+1 or
                not math.isclose(sum(p),1.0,rel_tol=0,abs_tol=2e-12)
                for p in probabilities)):
            raise ValueError("Independent historical key/value attention evidence incomplete")
        if position==1 and (
            torch_receipt["max_deviation_from_uniform_attention"]<=1e-12 or
            torch_receipt["max_context_difference_from_current_token_values"]<=1e-12):
            raise ValueError("Real trained query-key history did not exhibit distinct selection")
        positions.append({
            "position":position,"original_token_id":token_id,
            "original_embedding_sha256":embedding["embedding_sha256"],
            "original_seven_layer_outputs_F64":f64_prefix,
            "original_layer6_F64_digest":_f64_transfer_digest(h6),
            "native_full_layer7_F64_digest":vl_digest,
            "pytorch_full_layer7_F64_digest":torch_receipt["torch_f64_output_digest"],
            "native_full_layer7_F32_digest":vl_receipt["complete_layer_output_digest"],
            "max_abs_numeric_error":max_abs,
            "max_scaled_numeric_error":max_scaled,
            "rmse_numeric_error":rmse,
            "independent_historical_key_value_positions":position+1,
            "independent_trained_query_heads":torch_receipt["query_heads"],
            "independent_trained_kv_heads":torch_receipt["kv_heads"],
            "attention_probability_by_query_head":probabilities,
            "max_deviation_from_uniform_attention":
                torch_receipt["max_deviation_from_uniform_attention"],
            "max_context_difference_from_current_token_values":
                torch_receipt["max_context_difference_from_current_token_values"],
            "qualified_layer7_source_weight_bytes":vl_receipt["new_weight_payload_bytes"],
            "pytorch_layer7_source_weight_bytes":
                torch_receipt["new_source_weight_payload_bytes"],
        })
    return {
        "schema":"vl-i.original-qwen-real-trained-two-token-independent-torch.v1",
        "status":"REAL_ORIGINAL_QWEN_TWO_CAUSAL_TOKENS_TORCH_F64_ATTENTION_PARITY_VERIFIED",
        "model":model,"revision":revision,
        "source_config_sha256":plan["config_sha256"],
        "architecture_plan_digest":plan["plan_digest"],
        "source_index_ground_digest":plan["index_ground_digest"],
        "source_tokenizer_sha256":tg["asset_sha256"],
        "original_native_token_ids":selected,
        "trained_predecessor_layers":list(range(7)),
        "independently_compared_full_attention_layer":7,
        "trained_hidden_width":geom["hidden"],
        "positions":positions,
        "original_source_traffic":host.stats(),
        "real_trained_temporal_key_selection_observed":True,
        "real_trained_prior_value_contribution_observed":True,
        "independent_original_pytorch_F64_parity":True,
        "official_HF_BF16_runtime_parity":False,
        "full_64_layer_decoder_completed":False,
        "vocabulary_logits_computed":False,
        "VL64_core_modified":False,
        "scope":"Two actual original Qwen tokens through seven predecessor layers and full trained layer7 GQA with independently compared causal Q/K softmax and FFN, no final model inference",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--block-bytes",type=int,default=8*1024*1024)
    parser.add_argument("--host-budget-bytes",type=int,default=128*1024*1024)
    parser.add_argument("--layer-budget-bytes",type=int,default=2_000_000_000)
    args=parser.parse_args(argv)
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run(block_bytes=args.block_bytes,
                   host_budget_bytes=args.host_budget_bytes,
                   layer_budget_bytes=args.layer_budget_bytes)
    except (ValueError,OSError,TimeoutError,MemoryError,RuntimeError,
            OverflowError,ImportError,AssertionError) as exc:
        result={
            "schema":"vl-i.original-qwen-real-trained-two-token-independent-torch.v1",
            "status":"UNRESOLVED",
            "error_type":type(exc).__name__,
            "error":str(exc)[:500],
            "no_claim_of_original_two_token_torch_parity":True}
    target.write_text(json.dumps(result,indent=2,sort_keys=True,
                       allow_nan=False)+"\n",encoding="utf8")
    print("VL_I_ORIGINAL_CAUSAL_QWEN_TORCH_TWO_TOKENS",result["status"])
    return 0 if result["status"].startswith("REAL_") else 2


if __name__=="__main__":
    raise SystemExit(main())
