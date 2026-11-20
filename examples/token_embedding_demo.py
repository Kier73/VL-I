"""Original checkpoint token-row → partial trained operator → constructed views.

Input is an explicit original-vocabulary token ID, not a claimed natural
language tokenizer transcript. The exact BF16/F16/F32 row is range-observed
from the pinned, full-scale weight source. No model is resized or substituted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vli.operative_model import ground
from vli.token_embedding_ground import (
    TokenEmbeddingSession, token_intention, SCOPE,
)
from vli.large_model_comparison import SOURCES


def run(route, *, tokens=(100,101), render=True):
    from experiments.model_fabric.scaled_model_lab import (
        _url,_fetch_metadata,pinned_hf_range_host)
    from experiments.model_fabric.scaled_model_fabric import (
        MAX_INDEX_BYTES,QualifiedShardNavigator)
    if route not in SOURCES or type(tokens) not in (list,tuple) or not tokens:
        raise ValueError("Pinned original model and explicit token IDs required")
    cfg=SOURCES[route]
    model,rev=cfg["model"],cfg["revision"]
    idx=_fetch_metadata(_url(model,rev,"model.safetensors.index.json"),
                        MAX_INDEX_BYTES)
    nav=QualifiedShardNavigator(model=model,revision=rev,index_raw=idx,
         range_host=pinned_hf_range_host(model,rev),
         approved_hosts={"pinned-huggingface-https-range"})
    source=ground(route,nav.index)
    s=TokenEmbeddingSession(source,observer="vli-token-lab")
    responses=[]; restoration_verified=False
    for step,token_id in enumerate(tokens):
        if step==1:
            cp=s.checkpoint()
            restored=TokenEmbeddingSession.restore(
                source,cp,expected_head=cp["checkpoint_digest"])
            restoration_verified=restored.head==s.head
            s=restored
        intent=token_intention(source,token_id)
        output=s.operate(nav,intent,expected_head=s.head,render=render)
        witness=output["trained"]["embedding_witness"]
        responses.append({
            "token_id":token_id,"coordinate":output["coordinate"],
            "embedding_tensor":witness["tensor"],
            "embedding_sha256":witness["embedding_sha256"],
            "embedding_bytes":witness["embedding_bytes"],
            "embedding_dtype":witness["dtype"],
            "trained_operator":output["trained"]["numerical_result"]["kind"],
            "constructed":output["constructed"],
            "new_weight_bytes":output["trained"]["new_weight_bytes"],
            "new_range_reads":output["trained"]["new_ranges"],
            "range_chain":output["trained"]["range_chain_head"],
            "render_state":(output["rendered"]["state"] if output["rendered"]
                            else "UNRENDERED"),
            "native_tokenizer_applied":False,
            "pretrained_full_decoder_completed":False,
            "causal_scope":output["trained"]["causal_scope"],
        })
    if len(tokens)>1 and responses[0]["embedding_sha256"]==responses[1]["embedding_sha256"]:
        raise ValueError("Distinct original token rows lack distinct observed bytes")
    return {
      "schema":"vl-i.pinned-original-token-embedding-lab.v1",
      "status":"REAL_ORIGINAL_CHECKPOINT_TOKEN_EMBEDDING_PARTIAL_OPERATION",
      "model":model,"revision":rev,"route":route,
      "full_model_advertised_weight_bytes":cfg["advertised_bytes"],
      "index_sha256":cfg["index_sha256"],
      "per_token":responses,"restoration_verified":restoration_verified,
      "trained_weight_bytes_stored_inside_ground":0,
      "original_vocab_ids_explicit_not_text_tokenized":True,
      "all_layers_completed":False,"complete_logits_computed":False,
      "causal_input_grade":SCOPE[route],
      "scope":"exact_trained_embedding_rows_then_partial_native_model_relations",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=tuple(SOURCES),required=True)
    parser.add_argument("--tokens",type=int,nargs="+",default=[100,101])
    parser.add_argument("--no-render",action="store_true")
    parser.add_argument("--out",required=True)
    a=parser.parse_args(argv)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=run(a.model,tokens=a.tokens,render=not a.no_render)
    except (ValueError,OSError,TimeoutError,ImportError,RuntimeError) as exc:
        report={"schema":"vl-i.pinned-original-token-embedding-lab.v1",
                "status":"UNRESOLVED","model_route":a.model,
                "error_type":type(exc).__name__,
                "no_claim_of_completed_token_execution":True}
    path.write_text(json.dumps(report,sort_keys=True,indent=2,
                     allow_nan=False)+"\n",encoding="utf-8")
    print("ORIGINAL_TOKEN_EMBEDDING_STATUS",report["status"])
    return 0 if report["status"].startswith("REAL_") else 2


if __name__=="__main__":
    raise SystemExit(main())
