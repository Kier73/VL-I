"""Execute original Qwen text through a bounded model worker, not VL64 core.

Modes:
  recurrent: all 16Q/16K/48V trained GDN heads + one actual output row,
             across two literal-text tokens with externally pinned state.
  complete: complete original GDN+SwiGLU transformer layer for one token,
             including all output coordinates (expensive optional experiment).

The tokenizer and original 27B weights stay at their qualified sources.
The worker executes arithmetic on its CPU: *not* communication-only compute.
The 64 KiB VL interpreter is not modified. No full 64-layer inference claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from vli.large_model_comparison import SOURCES
from vli.operative_model import ground
from vli.pinned_text_ground import (
    admit, text_intention, tokenize, validate_observation,
    MAX_ASSET_BYTES, TOKENIZER_SOURCES)
from vli.token_embedding_ground import token_intention,observe_token_row
from vli.qualified_projection_host import HOST_NAME, BoundedModelRangeHost
from vli.qwen_all_heads import (
    QwenAllHeadsRecurrentHost,QwenVectorizedCompleteLayer,
)


DEFAULT_TEXT=(
    "Communication can support computation when information is interpreted "
    "through a virtual layer."
)


def run(*, mode="recurrent", positions=2, text=DEFAULT_TEXT,
        block_bytes=2*1024*1024, host_budget_bytes=64*1024*1024,
        layer_budget_bytes=2_000_000_000):
    from experiments.model_fabric.scaled_model_fabric import (
        QualifiedShardNavigator,MAX_INDEX_BYTES)
    from experiments.model_fabric.scaled_model_lab import _url,_fetch_metadata
    from experiments.model_fabric.complete_native_layers import (
        LocalStatePages,export_qwen_causal_ground,restore_qwen_causal_ground)
    if mode not in ("recurrent","complete") or type(positions) is not int or not 1<=positions<=2:
        raise ValueError("Only bounded original Qwen layer modes are qualified")
    src=SOURCES["qwen-27b"]
    model,revision=src["model"],src["revision"]
    raw_index=_fetch_metadata(_url(model,revision,"model.safetensors.index.json"),
                              MAX_INDEX_BYTES)
    host=BoundedModelRangeHost(
        model=model,revision=revision,block_bytes=block_bytes,
        budget_bytes=host_budget_bytes)
    def navigator():
        return QualifiedShardNavigator(
            model=model,revision=revision,index_raw=raw_index,
            range_host=host,approved_hosts={HOST_NAME})
    nav=navigator()
    g=ground("qwen-27b",nav.index)
    tok_src=TOKENIZER_SOURCES["qwen-27b"]
    if (tok_src["repository"]!=model or tok_src["revision"]!=revision
            or tok_src["filename"]!="tokenizer.json"):
        raise ValueError("Qwen tokenizer is not the exact source revision")
    tokenizer_url=(f"https://huggingface.co/{model}/resolve/"
                   f"{revision}/tokenizer.json")
    tok_raw=_fetch_metadata(tokenizer_url,MAX_ASSET_BYTES)
    tg,tokenizer=admit(g,tok_raw)
    intent=text_intention(tg,text)
    observed=tokenize(tg,tokenizer,intent)
    validate_observation(g,tg,intent,observed)
    if observed["token_count"]<positions:
        raise ValueError("Too few real original prompt tokens")
    def executor(nav):
        cls=(QwenAllHeadsRecurrentHost if mode=="recurrent"
             else QwenVectorizedCompleteLayer)
        return cls(nav,layer=0,key_heads=16,value_heads=48,
                   key_dim=128,value_dim=128,budget_bytes=layer_budget_bytes)
    active=executor(nav)
    records=[]
    with tempfile.TemporaryDirectory(prefix="vli-qwen-model-host-") as folder:
        store=LocalStatePages(folder)
        checkpoint_receipt=None
        for idx,token_id in enumerate(observed["token_ids"][:positions]):
            row,embedding=observe_token_row(nav,g,token_intention(g,token_id))
            if mode=="recurrent":
                calc=active.advance_recurrent(row)
                assert calc["trained_value_heads"]==48
                assert calc["trained_key_heads"]==16
                result={"kind":"all_48_native_recurrent_heads",
                        "trained":calc}
            else:
                out,calc=active.next(row)
                assert len(out)==5120
                result={"kind":"original_entire_gdn_swiglu_layer",
                        "trained":calc}
            records.append({"position":idx,"original_vocabulary_token_id":token_id,
                            "trained_embedding_sha256":embedding["embedding_sha256"],
                            "trained_embedding_payload_bytes":embedding["embedding_bytes"],
                            "computation":result})
            if idx==0 and positions>1:
                lineage=observed["observation_digest"]
                checkpoint=export_qwen_causal_ground(
                    active,store,input_sequence_digest=lineage)
                checkpoint_receipt={
                    "externally_pinned_checkpoint_digest":checkpoint["checkpoint_digest"],
                    "page_count":len(checkpoint["causal_pages"]),
                    "state_f64_bytes":checkpoint["state_f64_bytes"],
                    "saved_observer_position":active.position,
                }
                fresh=executor(navigator())
                restore_qwen_causal_ground(
                    fresh,store,checkpoint,
                    expected_digest=checkpoint["checkpoint_digest"],
                    expected_input_sequence_digest=lineage)
                if fresh.position!=active.position:
                    raise ValueError("Cross-observer causal checkpoint failed to restore")
                active=fresh;nav=fresh.nav
    stats=host.stats()
    return {"schema":"vl-i.qualified-model-host-recurrent-frontier.v1",
            "status":("REAL_PINNED_QWEN_48_RECURRENT_HEADS_OBSERVED"
                if mode=="recurrent" else
                "REAL_PINNED_QWEN_COMPLETE_GDN_LAYER_NO_INDEPENDENT_FULL_ORACLE"),
            "mode":mode,"original_model":model,"revision":revision,
            "source_index_sha256":src["index_sha256"],
            "original_model_advertised_payload_bytes":src["advertised_bytes"],
            "tokenizer_source_grade":tg["source_grade"],
            "tokenizer_sha256":tg["asset_sha256"],
            "original_literal_prompt_sha256":observed["text_sha256"],
            "native_token_count":observed["token_count"],
            "native_token_ids":observed["token_ids"],
            "trained_executed_positions":records,
            "cross_observer_checkpoint":checkpoint_receipt,
            "model_host_traffic":stats,
            "VL_interpreter_persistent_weight_bytes":0,
            "model_host_arithmetic_on_CPU":True,
            "all_64_decoder_layers_executed":False,
            "next_token_logits_executed":False,
            "scope":("all real GDN recurrent heads plus selected output row"
                     if mode=="recurrent" else
                     "full trained first GDN+SwiGLU layer but no independent whole-layer pretrained reference"),
           }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("recurrent","complete"),default="recurrent")
    parser.add_argument("--positions",type=int,default=2)
    parser.add_argument("--block-bytes",type=int,default=2*1024*1024)
    parser.add_argument("--host-budget-bytes",type=int,default=64*1024*1024)
    parser.add_argument("--layer-budget-bytes",type=int,default=2_000_000_000)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    try:
        receipt=run(mode=args.mode,positions=args.positions,
             block_bytes=args.block_bytes,
             host_budget_bytes=args.host_budget_bytes,
             layer_budget_bytes=args.layer_budget_bytes)
    except (ValueError,OSError,TimeoutError,MemoryError,
            RuntimeError,OverflowError,ImportError) as e:
        receipt={"schema":"vl-i.qualified-model-host-recurrent-frontier.v1",
                 "status":"UNRESOLVED","mode":args.mode,
                 "error_type":type(e).__name__,
                 "error":str(e)[:300],
                 "no_claim_of_pretrained_numerical_completion":True}
    target.write_text(json.dumps(receipt,indent=2,sort_keys=True,
                                 allow_nan=False)+"\n",encoding="utf8")
    print("VL_I_ORIGINAL_QWEN_HOST_FRONTIER",receipt["status"])
    return 0 if receipt["status"].startswith("REAL_") else 2


if __name__=="__main__":
    raise SystemExit(main())
