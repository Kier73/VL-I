"""QUALIFIED-009: independently check original Qwen's layer-7 trained math.

Walk the actual original token through full layers 0..6, then compare
complete trained gated-attention layer 7 from:
 (A) the existing VL-I qualified F64 source range execution, and
 (B) an independent PyTorch F64 implementation with its own matrix
     multiplication, normalization, rotary phases, grouped causal
     softmax, learned gating, and feed-forward calculations.

Both read original revision-pinned source tensor bytes. This is NOT an
official HF BF16 whole-model execution or a language-generation claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vli.ground import _digest
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
    model,rev=source["model"],source["revision"]
    index=_fetch_metadata(_url(model,rev,"model.safetensors.index.json"),
                          MAX_INDEX_BYTES)
    config=_fetch_metadata(_url(model,rev,"config.json"),65536)
    host=BoundedModelRangeHost(
        model=model,revision=rev,block_bytes=block_bytes,
        budget_bytes=host_budget_bytes)
    nav=QualifiedShardNavigator(
        model=model,revision=rev,index_raw=index,range_host=host,
        approved_hosts={HOST_NAME})
    plan=admit_indexed_qwen_topology(
        nav,config,expected_config_sha256=source.get("config_sha256"))
    if (plan["layer_count"]!=source["layers"] or
        [p["kind"] for p in plan["layers"][:8]]!=
        ["GDN","GDN","GDN","GATED_GQA"]*2):
        raise ValueError("Original source no longer qualifies this eight-layer path")
    source_ground=ground("qwen-27b",nav.index)
    t=TOKENIZER_SOURCES["qwen-27b"]
    if t["repository"]!=model or t["revision"]!=rev:
        raise ValueError("Original model/tokenizer source identity changed")
    token_raw=_fetch_metadata(
        f"https://huggingface.co/{model}/resolve/{rev}/tokenizer.json",
        MAX_ASSET_BYTES)
    tg,tokenizer=admit(source_ground,token_raw)
    intent=text_intention(tg,text)
    observed=tokenize(tg,tokenizer,intent)
    validate_observation(source_ground,tg,intent,observed)
    original_token=observed["token_ids"][0]
    if original_token!=63204:
        raise ValueError("Independent original trained example token changed")

    archived=json.loads((
        Path(__file__).resolve().parents[1]/
        "experiments/qwen_all_heads/datasets/"
        "2026-11-19-real-original-qwen-first-eight-architecture-directed.json"
        ).read_text("utf8"))
    if (archived["model"]!=model or archived["revision"]!=rev or
        archived["source_config_sha256"]!=plan["config_sha256"] or
        archived["source_architecture_plan_digest"]!=plan["plan_digest"]):
        raise ValueError("Historical original trained source/plan changed")

    trained_row,embedding_evidence=observe_token_row(
        nav,source_ground,token_intention(source_ground,original_token))
    seven=QwenArchitectureCausalPrefix(
        nav,plan,stop_layer=7,layer_budget_bytes=layer_budget_bytes)
    h6,seven_receipt=seven.step(trained_row,token_id=original_token)
    archived_w=archived["original_position_0"]
    for witness,expected in zip(
        seven_receipt["layer_witnesses"],
        archived_w["output_f64_digests"][:7]):
        if witness["output_f64_digest"]!=expected:
            raise ValueError("Original pretrained seven-layer prefix changed")
    before=_f64_transfer_digest(h6)
    if before!=archived_w["output_f64_digests"][6]:
        raise ValueError("Original pre-attention hidden differs from archive")

    g=plan["geometry"]
    qualified=QwenQualifiedGatedAttention(
        nav,layer=7,query_heads=g["query_heads"],kv_heads=g["kv_heads"],
        head_dim=g["head_dim"],rotary_dim=g["rotary_dim"],
        rope_theta=g["rope_theta"],eps=g["eps"],
        budget_bytes=layer_budget_bytes)
    independent=OriginalQwenTorchGQA(nav,plan,layer=7)
    # The same list of actual trained F64 inputs is passed to each
    # independently implemented whole attention/FFN executor.
    torch_output,torch_report=independent.next(h6)
    vl_output,vl_report=qualified.next(h6)
    vl_digest=_f64_transfer_digest(vl_output)
    if (vl_digest!=archived_w["output_f64_digests"][7] or
        vl_report["complete_layer_output_digest"]!=
        archived_w["output_f32_digests"][7]):
        raise ValueError("Qualified original layer 7 diverged from earlier trained evidence")
    if (len(torch_output)!=g["hidden"] or
        len(vl_output)!=g["hidden"] or
        independent.position!=1 or qualified.position!=1):
        raise ValueError("Independent reference did not complete original trained width")
    errors=[abs(a-b) for a,b in zip(vl_output,torch_output)]
    max_abs=max(errors)
    max_scaled=max(error/max(1.0,abs(a),abs(b))
                   for error,a,b in zip(errors,vl_output,torch_output))
    rmse=math.sqrt(math.fsum(z*z for z in errors)/len(errors))
    if (not all(math.isfinite(x) for x in (max_abs,max_scaled,rmse)) or
        max_abs>1e-5 or max_scaled>2e-7):
        raise ValueError(
            f"Independent real-weight F64 parity falsified: abs={max_abs:.8g}, "
            f"scaled={max_scaled:.8g}")
    return {
        "schema":"vl-i.original-qwen-independent-torch-math-parity.v1",
        "status":"REAL_ORIGINAL_TRAINED_QWEN_LAYER7_TORCH_F64_PARITY_VERIFIED",
        "model":model,"revision":rev,
        "source_config_sha256":plan["config_sha256"],
        "source_index_ground_digest":plan["index_ground_digest"],
        "admitted_architecture_plan_digest":plan["plan_digest"],
        "actual_original_token_id":original_token,
        "original_embedding_sha256":embedding_evidence["embedding_sha256"],
        "independent_input_from_full_trained_layers":list(range(7)),
        "original_hidden_width":g["hidden"],
        "hidden6_f64_digest":before,
        "previous_original_eight_layer_f64_digest":archived_w["output_f64_digests"][7],
        "vl_original_layer7_f64_digest":vl_digest,
        "torch_independent_layer7_f64_digest":torch_report["torch_f64_output_digest"],
        "max_abs_numeric_error":max_abs,
        "max_scaled_numeric_error":max_scaled,
        "rmse_numeric_error":rmse,
        "reference":torch_report,
        "vl_layer7_source_receipt":vl_report,
        "original_trained_prefix_receipts":seven_receipt["layer_witnesses"],
        "model_host_traffic":host.stats(),
        "original_model_bf16_runtime_equivalence_tested":False,
        "independent_f64_pytorch_trained_operator_parity":True,
        "full_original_decoder_completed":False,
        "vocabulary_logits_computed":False,
        "vl64_core_modified":False,
        "scope":"Independent complete mathematical PyTorch F64 reference on original trained layer 7 and actual seven-layer token-derived input; not HF BF16 runtime equivalence",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--block-bytes",type=int,default=8*1024*1024)
    parser.add_argument("--host-budget-bytes",type=int,default=128*1024*1024)
    parser.add_argument("--layer-budget-bytes",type=int,default=2_000_000_000)
    args=parser.parse_args(argv)
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run(block_bytes=args.block_bytes,
                   host_budget_bytes=args.host_budget_bytes,
                   layer_budget_bytes=args.layer_budget_bytes)
    except (ValueError,OSError,TimeoutError,MemoryError,RuntimeError,
            OverflowError,ImportError,AssertionError) as exc:
        result={
            "schema":"vl-i.original-qwen-independent-torch-math-parity.v1",
            "status":"UNRESOLVED",
            "error_type":type(exc).__name__,
            "error":str(exc)[:500],
            "no_claim_of_original_weight_pytorch_parity":True}
    p.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
                 encoding="utf8")
    print("VL_I_INDEPENDENT_REAL_ORIGINAL_QWEN_TORCH",result["status"])
    return 0 if result["status"].startswith("REAL_") else 2


if __name__=="__main__":
    raise SystemExit(main())
