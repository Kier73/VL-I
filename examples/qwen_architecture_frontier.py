"""QUALIFIED-008: original-trained Qwen decoder prefix from observed architecture.

Fetch the *same immutable original model revision* for config, index,
tokenizer and every weight row. Construct the actual needed operator for
each declared indexed layer, without manually specifying a 3:1 schedule.

Example:
  PYTHONPATH=upstream/vl64 python -m examples.qwen_architecture_frontier \\
      --layers 8 --positions 1 --out artifacts/qwen-original-eight-layers.json
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
    TOKENIZER_SOURCES,MAX_ASSET_BYTES,
)
from vli.token_embedding_ground import token_intention,observe_token_row
from vli.qualified_projection_host import HOST_NAME,BoundedModelRangeHost
from vli.qwen_architecture_navigator import (
    admit_indexed_qwen_topology,QwenArchitectureCausalPrefix,
)
from examples.qwen_model_host_frontier import DEFAULT_TEXT


def run(*,layers=8,positions=1,text=DEFAULT_TEXT,
        block_bytes=8*1024*1024,host_budget_bytes=128*1024*1024,
        layer_budget_bytes=2_000_000_000):
    from experiments.model_fabric.scaled_model_fabric import (
        QualifiedShardNavigator,MAX_INDEX_BYTES)
    from experiments.model_fabric.scaled_model_lab import _url,_fetch_metadata
    from experiments.model_fabric.complete_native_layers import LocalStatePages

    if (type(layers) is not int or not 1<=layers<=64 or
        type(positions) is not int or not 1<=positions<=2):
        raise ValueError("Original model experiment must have explicit finite extent")
    source=SOURCES["qwen-27b"]
    model,rev=source["model"],source["revision"]
    raw_index=_fetch_metadata(_url(model,rev,"model.safetensors.index.json"),
                              MAX_INDEX_BYTES)
    raw_config=_fetch_metadata(_url(model,rev,"config.json"),65536)
    host=BoundedModelRangeHost(model=model,revision=rev,
          block_bytes=block_bytes,budget_bytes=host_budget_bytes)
    def navigator():
        return QualifiedShardNavigator(model=model,revision=rev,
            index_raw=raw_index,range_host=host,approved_hosts={HOST_NAME})
    nav=navigator()
    plan=admit_indexed_qwen_topology(
        nav,raw_config,expected_config_sha256=source.get("config_sha256"))
    if plan["layer_count"]!=source["layers"]:
        raise ValueError("Trained original decoder layer count changed")
    source_ground=ground("qwen-27b",nav.index)
    tok=TOKENIZER_SOURCES["qwen-27b"]
    if (tok["repository"]!=model or tok["revision"]!=rev or
        tok["filename"]!="tokenizer.json"):
        raise ValueError("Original tokenizer revision/source do not agree")
    token_raw=_fetch_metadata(
        f"https://huggingface.co/{model}/resolve/{rev}/tokenizer.json",
        MAX_ASSET_BYTES)
    text_ground,tokenizer=admit(source_ground,token_raw)
    intention=text_intention(text_ground,text)
    observed=tokenize(text_ground,tokenizer,intention)
    validate_observation(source_ground,text_ground,intention,observed)
    if observed["token_count"]<positions:
        raise ValueError("Missing original text token positions")

    active=QwenArchitectureCausalPrefix(
        nav,plan,stop_layer=layers,layer_budget_bytes=layer_budget_bytes)
    historical=json.loads((Path(__file__).resolve().parents[1]/
        "experiments/qwen_all_heads/datasets/"
        "2026-11-19-real-original-qwen-first-hybrid-block.json").read_text())
    records=[]
    checkpoint=None
    with tempfile.TemporaryDirectory(prefix="vli-architecture-orbit-") as folder:
        store=LocalStatePages(folder)
        for position,token_id in enumerate(observed["token_ids"][:positions]):
            embedding,embedding_receipt=observe_token_row(
                nav,source_ground,token_intention(source_ground,token_id))
            result,receipt=active.step(embedding,token_id=token_id)
            witnesses=receipt["layer_witnesses"]
            if (len(result)!=plan["geometry"]["hidden"] or
                len(witnesses)!=layers or
                [r["layer"] for r in witnesses]!=list(range(layers)) or
                [r["kind"] for r in witnesses]!=[
                    p["kind"] for p in plan["layers"][:layers]] or
                any(a["output_f64_digest"]!=b["input_f64_digest"]
                    for a,b in zip(witnesses,witnesses[1:]))):
                raise ValueError("Incomplete verified source-indexed causal execution")
            if layers>=4 and position<2:
                old=historical["positions"][position]
                if (old["token_id"]!=token_id or
                    [w["output_f64_digest"] for w in witnesses[:4]]!=
                    old["exact_f64_layer_outputs"]):
                    raise ValueError(
                        "Earlier independent original four-layer execution diverged")
            records.append({
                "position":position,"original_token_id":token_id,
                "trained_embedding_sha256":embedding_receipt["embedding_sha256"],
                "trained_embedding_bytes":embedding_receipt["embedding_bytes"],
                "architecture_derived_transition":receipt,
                "prior_qualified_007_four_layer_prefix_reproduced":(
                    layers>=4 and position<2)})
            if position==0:
                pinned=active.checkpoint(
                    store,input_sequence_digest=observed["observation_digest"])
                pieces=pinned["layer_checkpoints"]
                checkpoint={
                    "checkpoint_digest":pinned["checkpoint_digest"],
                    "config_sha256":plan["config_sha256"],
                    "architecture_plan_digest":plan["plan_digest"],
                    "layer_digests":[p["checkpoint_digest"] for p in pieces],
                    "layer_kinds":[x["kind"] for x in plan["layers"][:layers]],
                    "state_pages":[len(p.get("causal_pages",p.get("kv_pages",[])))
                                   for p in pieces],
                    "state_f64_bytes":sum(p.get("state_f64_bytes",
                        p.get("kv_f64_bytes",0)) for p in pieces),
                    "fresh_source_navigator_restore":True,
                    "different_OS_process_on_original_model":False}
                fresh=navigator()
                fresh_plan=admit_indexed_qwen_topology(
                    fresh,raw_config,expected_config_sha256=plan["config_sha256"])
                recovered=QwenArchitectureCausalPrefix.restore(
                    fresh,fresh_plan,store,pinned,
                    expected_digest=pinned["checkpoint_digest"],
                    expected_input_sequence_digest=observed["observation_digest"],
                    expected_plan_digest=plan["plan_digest"],
                    layer_budget_bytes=layer_budget_bytes)
                if recovered.head!=active.head or recovered.position!=active.position:
                    raise ValueError("Indexed architecture restore changed causal head")
                active=recovered
                nav=fresh
    return {
        "schema":"vl-i.original-qwen-architecture-directed-prefix.v1",
        "status":"REAL_PINNED_QWEN_ARCHITECTURE_DIRECTED_PREFIX",
        "model":model,"revision":rev,
        "source_index_sha256":source["index_sha256"],
        "source_index_ground_digest":plan["index_ground_digest"],
        "source_config_sha256":plan["config_sha256"],
        "source_config_external_sha256_pin":plan["external_config_sha256_pin_supplied"],
        "architecture_plan_digest":plan["plan_digest"],
        "full_indexed_architecture_layer_types":[x["kind"] for x in plan["layers"]],
        "source_architecture_layers":plan["layer_count"],
        "materialized_original_layers":layers,
        "original_token_ids":observed["token_ids"],
        "original_prompt_sha256":observed["text_sha256"],
        "tokenizer_sha256":text_ground["asset_sha256"],
        "completed_original_positions":active.position,
        "trained_positions":records,
        "checkpoint":checkpoint,
        "last_f64_output_digest":active.last_output_digest,
        "last_semantic_head":active.head,
        "model_host_traffic":host.stats(),
        "vl64_source_modified":False,
        "vl64_persistent_model_weight_bytes":0,
        "worker_cpu_mathematics":True,
        "independent_original_weight_pytorch_parity":False,
        "original_all_64_layer_decoder_complete":layers==64,
        "original_final_norm_and_vocabulary_logits_complete":False,
        "scope":"text-only original pretrained Qwen decoder prefix; no final logits or generation",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers",type=int,default=8)
    p.add_argument("--positions",type=int,default=1)
    p.add_argument("--block-bytes",type=int,default=8*1024*1024)
    p.add_argument("--host-budget-bytes",type=int,default=128*1024*1024)
    p.add_argument("--layer-budget-bytes",type=int,default=2_000_000_000)
    p.add_argument("--out",required=True)
    args=p.parse_args(argv)
    target=Path(args.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        receipt=run(layers=args.layers,positions=args.positions,
            block_bytes=args.block_bytes,
            host_budget_bytes=args.host_budget_bytes,
            layer_budget_bytes=args.layer_budget_bytes)
    except (ValueError,OSError,TimeoutError,MemoryError,RuntimeError,
            OverflowError,ImportError) as exc:
        receipt={"schema":"vl-i.original-qwen-architecture-directed-prefix.v1",
                 "status":"UNRESOLVED","error_type":type(exc).__name__,
                 "error":str(exc)[:300],
                 "no_claim_of_original_eight_layer_execution":True}
    target.write_text(json.dumps(receipt,indent=2,sort_keys=True,
        allow_nan=False)+"\n",encoding="utf8")
    print("VL_I_ARCHITECTURE_DIRECTED_QWEN",receipt["status"])
    return 0 if receipt["status"].startswith("REAL_") else 2


if __name__=="__main__":
    raise SystemExit(main())
