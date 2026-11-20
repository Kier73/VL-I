"""QUALIFIED-007 original Qwen3.8-27B first hybrid block experiment.

The unmodified source is pinned to the same original 27B checkpoint used
in QUALIFIED-005/006. The original tokenizer selects real paragraph IDs.
Each original layer receives the preceding layer's **actual** output.
No full decoder/logit/native-network-arithmetic claim is made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from vli.large_model_comparison import SOURCES
from vli.operative_model import ground
from vli.pinned_text_ground import (
    admit,text_intention,tokenize,validate_observation,
    MAX_ASSET_BYTES,TOKENIZER_SOURCES,
)
from vli.token_embedding_ground import token_intention,observe_token_row
from vli.qualified_projection_host import HOST_NAME,BoundedModelRangeHost
from vli.qwen_all_heads import QwenVectorizedCompleteLayer
from vli.qwen_gated_attention import QwenQualifiedGatedAttention
from vli.qwen_layer_chain import QwenCausalGDNChain
from vli.qwen_hybrid_block import QwenQualifiedHybridBlock
from examples.qwen_model_host_frontier import DEFAULT_TEXT


def run(*,mode="hybrid4",positions=2,text=DEFAULT_TEXT,
        block_bytes=8*1024*1024,host_budget_bytes=128*1024*1024,
        layer_budget_bytes=2_000_000_000):
    from experiments.model_fabric.scaled_model_fabric import (
        QualifiedShardNavigator,MAX_INDEX_BYTES)
    from experiments.model_fabric.scaled_model_lab import _url,_fetch_metadata
    from experiments.model_fabric.complete_native_layers import LocalStatePages

    if mode not in ("gdn3","hybrid4") or type(positions) is not int or not 1<=positions<=2:
        raise ValueError("Unqualified hybrid execution mode or token extent")
    source=SOURCES["qwen-27b"]
    model,revision=source["model"],source["revision"]
    index=_fetch_metadata(_url(model,revision,"model.safetensors.index.json"),
                          MAX_INDEX_BYTES)
    host=BoundedModelRangeHost(model=model,revision=revision,
        block_bytes=block_bytes,budget_bytes=host_budget_bytes)
    def nav():
        return QualifiedShardNavigator(model=model,revision=revision,
            index_raw=index,range_host=host,approved_hosts={HOST_NAME})
    def chain(qualified):
        layers=[QwenVectorizedCompleteLayer(
            qualified,layer=i,key_heads=16,value_heads=48,
            key_dim=128,value_dim=128,budget_bytes=layer_budget_bytes)
            for i in range(3)]
        if mode=="gdn3":
            return QwenCausalGDNChain(layers)
        layers.append(QwenQualifiedGatedAttention(
            qualified,layer=3,query_heads=24,kv_heads=4,head_dim=256,
            rotary_dim=64,rope_theta=10_000_000.0,
            budget_bytes=layer_budget_bytes))
        return QwenQualifiedHybridBlock(layers)

    navigator=nav()
    g=ground("qwen-27b",navigator.index)
    tok_source=TOKENIZER_SOURCES["qwen-27b"]
    if (tok_source["repository"]!=model or tok_source["revision"]!=revision or
        tok_source["filename"]!="tokenizer.json"):
        raise ValueError("Tokenizer identity changed from original checkpoint")
    tokenizer_url=(f"https://huggingface.co/{model}/resolve/"
                   f"{revision}/tokenizer.json")
    tok_raw=_fetch_metadata(tokenizer_url,MAX_ASSET_BYTES)
    tg,tokenizer=admit(g,tok_raw)
    intent=text_intention(tg,text)
    observed=tokenize(tg,tokenizer,intent)
    validate_observation(g,tg,intent,observed)
    if observed["token_count"]<positions:
        raise ValueError("Insufficient original paragraph token positions")
    active=chain(navigator)
    count=3 if mode=="gdn3" else 4
    records=[];checkpoint=None
    with tempfile.TemporaryDirectory(prefix="vli-qualified-hybrid-") as folder:
        store=LocalStatePages(folder)
        for idx,token_id in enumerate(observed["token_ids"][:positions]):
            embedding,embedding_receipt=observe_token_row(
                navigator,g,token_intention(g,token_id))
            out,receipt=active.step(embedding,token_id=token_id)
            witnesses=receipt["layer_witnesses"]
            if (len(out)!=5120 or len(witnesses)!=count or
                [w["layer"] for w in witnesses]!=list(range(count)) or
                any(left["output_f64_digest"]!=right["input_f64_digest"]
                    for left,right in zip(witnesses,witnesses[1:]))):
                raise ValueError("Incomplete original pretrained hybrid transition")
            records.append({
                "position":idx,"original_token_id":token_id,
                "embedding_sha256":embedding_receipt["embedding_sha256"],
                "embedding_payload_bytes":embedding_receipt["embedding_bytes"],
                "complete_hybrid_transition":receipt})
            if idx==0 and positions==2:
                pinned=active.checkpoint(
                    store,input_sequence_digest=observed["observation_digest"])
                manifests=pinned["layer_checkpoints"]
                checkpoint={
                    "externally_pinned_checkpoint_digest":pinned["checkpoint_digest"],
                    "position":pinned["position"],
                    "layer_digests":[p["checkpoint_digest"] for p in manifests],
                    "per_layer_state_pages":[len(p.get("causal_pages",
                        p.get("kv_pages",[]))) for p in manifests],
                    "state_f64_bytes":sum(p.get("state_f64_bytes",
                        p.get("kv_f64_bytes",0)) for p in manifests),
                    "separate_GDN_and_KV_state_authorities":mode=="hybrid4",
                    "fresh_worker_navigator_recovery":True,
                    "separate_OS_process_recovery":False}
                restored=type(active).restore(
                    chain(nav()).runners,store,pinned,
                    expected_digest=pinned["checkpoint_digest"],
                    expected_input_sequence_digest=observed["observation_digest"])
                if restored.position!=active.position or restored.head!=active.head:
                    raise ValueError("Hybrid checkpoint failed its external authority pin")
                active=restored
                navigator=restored.runners[0].nav
    return {
        "schema":"vl-i.qwen-first-original-hybrid-block.v1",
        "status":("REAL_PINNED_QWEN_FIRST_THREE_GDN_LAYERS"
                  if mode=="gdn3" else
                  "REAL_PINNED_QWEN_FIRST_COMPLETE_HYBRID_BLOCK"),
        "mode":mode,"original_model":model,"revision":revision,
        "source_index_sha256":source["index_sha256"],
        "original_model_advertised_payload_bytes":source["advertised_bytes"],
        "tokenizer_source_grade":tg["source_grade"],
        "tokenizer_sha256":tg["asset_sha256"],
        "original_prompt_sha256":observed["text_sha256"],
        "original_native_token_ids":observed["token_ids"],
        "trained_positions":records,
        "trained_layer_coordinates":list(range(count)),
        "last_completed_position":active.position,
        "final_output_f64_digest":active.last_output_digest,
        "final_causal_head":active.head,
        "cross_observer_checkpoint":checkpoint,
        "model_host_traffic":host.stats(),
        "VL64_core_changed":False,
        "VL64_persistent_model_weight_bytes":0,
        "original_worker_arithmetic_CPU":True,
        "independent_full_original_pytorch_oracle":False,
        "full_64_layer_decoder_executed":False,
        "vocabulary_logits_computed":False,
        "scope":"original trained GDN -> first gated attention block with exact causal continuity, not full model inference",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("gdn3","hybrid4"),default="hybrid4")
    parser.add_argument("--positions",type=int,default=2)
    parser.add_argument("--block-bytes",type=int,default=8*1024*1024)
    parser.add_argument("--host-budget-bytes",type=int,default=128*1024*1024)
    parser.add_argument("--layer-budget-bytes",type=int,default=2_000_000_000)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run(mode=args.mode,positions=args.positions,
             block_bytes=args.block_bytes,host_budget_bytes=args.host_budget_bytes,
             layer_budget_bytes=args.layer_budget_bytes)
    except (ValueError,OSError,TimeoutError,MemoryError,RuntimeError,
            OverflowError,ImportError) as exc:
        result={"schema":"vl-i.qwen-first-original-hybrid-block.v1",
                "status":"UNRESOLVED","mode":args.mode,
                "error_type":type(exc).__name__,"error":str(exc)[:300],
                "no_claim_of_full_original_pretrained_hybrid_block":True}
    target.write_text(json.dumps(result,indent=2,sort_keys=True,
                      allow_nan=False)+"\n",encoding="utf8")
    print("VL_I_QWEN_FIRST_HYBRID_BLOCK",result["status"])
    return 0 if result["status"].startswith("REAL_") else 2


if __name__=="__main__":
    raise SystemExit(main())
