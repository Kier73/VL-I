"""QUALIFIED-006: feed an original Qwen paragraph through layer 0 -> layer 1.

Opt-in original 27B numerical work, not a fixture or a model generation claim.
Both layers see unchanged pinned original weights; layer 1 receives layer 0's
actual 5,120-element output, and both retain their own causal histories.

Example:
  PYTHONPATH=upstream/vl64 python -m examples.qwen_layer_chain_frontier \\
      --positions 2 --out artifacts/qwen-first-two-layers.json
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
from vli.qwen_layer_chain import QwenCausalGDNChain
from examples.qwen_model_host_frontier import DEFAULT_TEXT


def run(*, positions=2, text=DEFAULT_TEXT,
        block_bytes=8*1024*1024, host_budget_bytes=128*1024*1024,
        layer_budget_bytes=2_000_000_000):
    from experiments.model_fabric.scaled_model_fabric import (
        QualifiedShardNavigator,MAX_INDEX_BYTES)
    from experiments.model_fabric.scaled_model_lab import _url,_fetch_metadata
    from experiments.model_fabric.complete_native_layers import LocalStatePages

    if type(positions) is not int or not 1<=positions<=2:
        raise ValueError("At most two original paragraph tokens in this experiment")
    source=SOURCES["qwen-27b"]
    model,revision=source["model"],source["revision"]
    raw_index=_fetch_metadata(_url(model,revision,"model.safetensors.index.json"),
                              MAX_INDEX_BYTES)
    host=BoundedModelRangeHost(
        model=model,revision=revision,
        block_bytes=block_bytes,budget_bytes=host_budget_bytes)
    def navigator():
        return QualifiedShardNavigator(
            model=model,revision=revision,index_raw=raw_index,
            range_host=host,approved_hosts={HOST_NAME})
    def executors(nav):
        return [QwenVectorizedCompleteLayer(
            nav,layer=i,key_heads=16,value_heads=48,
            key_dim=128,value_dim=128,budget_bytes=layer_budget_bytes)
            for i in (0,1)]

    nav=navigator()
    g=ground("qwen-27b",nav.index)
    tok_source=TOKENIZER_SOURCES["qwen-27b"]
    if (tok_source["repository"]!=model or
        tok_source["revision"]!=revision or
        tok_source["filename"]!="tokenizer.json"):
        raise ValueError("Original tokenizer and weight Ground revisions differ")
    raw_tok=_fetch_metadata(
        _url(model,revision,"tokenizer.json"),MAX_ASSET_BYTES)
    token_ground,tokenizer=admit(g,raw_tok)
    intent=text_intention(token_ground,text)
    observed=tokenize(token_ground,tokenizer,intent)
    validate_observation(g,token_ground,intent,observed)
    if observed["token_count"]<positions:
        raise ValueError("No original tokenizer tokens at required positions")

    chain=QwenCausalGDNChain(executors(nav))
    records=[]
    checkpoint_receipt=None
    with tempfile.TemporaryDirectory(prefix="vli-qualified-gdn-chain-") as root:
        store=LocalStatePages(root)
        for idx,token_id in enumerate(observed["token_ids"][:positions]):
            row,embedding=observe_token_row(nav,g,token_intention(g,token_id))
            out,transition=chain.step(row,token_id=token_id)
            if (len(out)!=5120 or len(transition["layer_witnesses"])!=2 or
                transition["layer_witnesses"][0]["output_f64_digest"]!=
                transition["layer_witnesses"][1]["input_f64_digest"]):
                raise ValueError("Trained layer transition did not compose")
            records.append({
                "position":idx,"original_token_id":token_id,
                "trained_embedding_sha256":embedding["embedding_sha256"],
                "trained_embedding_bytes":embedding["embedding_bytes"],
                "causal_transition":transition,
            })
            if idx==0 and positions==2:
                pinned=chain.checkpoint(
                    store,input_sequence_digest=observed["observation_digest"])
                checkpoint_receipt={
                    "checkpoint_digest":pinned["checkpoint_digest"],
                    "position":pinned["position"],
                    "per_layer_pinned_digests":[
                        p["checkpoint_digest"]
                        for p in pinned["layer_checkpoints"]],
                    "per_layer_state_pages":[
                        len(p["causal_pages"])
                        for p in pinned["layer_checkpoints"]],
                    "state_f64_bytes":sum(
                        p["state_f64_bytes"]
                        for p in pinned["layer_checkpoints"]),
                    "separate_layer_0_and_1_states":True,
                    "fresh_navigator_and_executor_recovery":True,
                    "new_host_process_recovery":False,
                }
                restored=QwenCausalGDNChain.restore(
                    executors(navigator()),store,pinned,
                    expected_digest=pinned["checkpoint_digest"],
                    expected_input_sequence_digest=observed["observation_digest"])
                if restored.position!=chain.position or restored.head!=chain.head:
                    raise ValueError("Layer chain failed externally pinned recovery")
                chain=restored
                nav=restored.runners[0].nav
    return {
        "schema":"vl-i.original-qwen-two-complete-layers.v1",
        "status":"REAL_PINNED_QWEN_TWO_CAUSAL_GDN_LAYERS_NO_INDEPENDENT_REAL_ORACLE",
        "original_model":model,"revision":revision,
        "source_index_sha256":source["index_sha256"],
        "model_advertised_payload_bytes":source["advertised_bytes"],
        "tokenizer_source_grade":token_ground["source_grade"],
        "tokenizer_sha256":token_ground["asset_sha256"],
        "original_literal_prompt_sha256":observed["text_sha256"],
        "native_token_ids":observed["token_ids"],
        "trained_positions":records,
        "trained_layer_coordinates":[0,1],
        "last_completed_causal_position":chain.position,
        "final_complete_layer_output_f64_digest":chain.last_output_digest,
        "final_causal_chain_head":chain.head,
        "cross_observer_checkpoint":checkpoint_receipt,
        "model_host_traffic":host.stats(),
        "VL64_interpreter_source_changed":False,
        "VL64_interpreter_persistent_weight_bytes":0,
        "model_host_arithmetic_on_CPU":True,
        "independent_real_pretrained_whole_layer_reference":False,
        "all_64_decoder_layers_executed":False,
        "next_token_logits_executed":False,
        "scope":"two consecutive original GDN+SwiGLU layers with exact numerical handoff, not full decoder inference",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions",type=int,default=2)
    parser.add_argument("--block-bytes",type=int,default=8*1024*1024)
    parser.add_argument("--host-budget-bytes",type=int,default=128*1024*1024)
    parser.add_argument("--layer-budget-bytes",type=int,default=2_000_000_000)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=run(positions=args.positions,block_bytes=args.block_bytes,
                   host_budget_bytes=args.host_budget_bytes,
                   layer_budget_bytes=args.layer_budget_bytes)
    except (ValueError,OSError,TimeoutError,MemoryError,RuntimeError,
            OverflowError,ImportError) as e:
        report={"schema":"vl-i.original-qwen-two-complete-layers.v1",
                "status":"UNRESOLVED","error_type":type(e).__name__,
                "error":str(e)[:300],
                "no_claim_of_pretrained_two_layer_execution":True}
    path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",
                    encoding="utf8")
    print("VL_I_QWEN_FIRST_TWO_CAUSAL_LAYERS",report["status"])
    return 0 if report["status"].startswith("REAL_") else 2


if __name__=="__main__":
    raise SystemExit(main())
