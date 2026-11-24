"""NOC-001: pretrained SmolLM nonlinear observer cut against held-out prompts.

An OFFLINE opt-in experiment. Requires the exact locally provisioned pinned
SmolLM-135M snapshot already used by smollm_pairwise_readout. Never requests
or downloads weights, and never calls a model on the proposed Virtual Layer
cut path. The separate unrestricted FP32 model call is a validation oracle.

    OPENBLAS_NUM_THREADS=1 python -m examples.smollm_nonlinear_cut \
        --snapshot /existing/pinned/SmolLM-135M --out artifacts/noc001.json
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import tokenizers

from examples.smollm_pairwise_readout import (
    MODEL,REVISION,HASHES,YES,NO,WIDTH,INTERMEDIATE,
    TailBoundary,capture_input,compile_readout,
)
from vli.nonlinear_observer_cut import compile_nonlinear_cut, evaluate_cut

# Fixed before evaluating the oracle; none are from the original 85-position
# SmolLM readout experiment. This is held-out relative to the original suite,
# not training/validation/test separation for discovering the row-bound rule.
HELDOUT_TEXTS = (
    "The telescope records a distant star. The researcher answers",
    "To determine if the statement is correct, we first inspect the evidence:",
    "A baker warmed the oven before mixing the ingredients.",
    "The programmer checked the invariant after every state transition.",
)


def capture_cut(model, ids):
    """Execute the source causal prefix only to the final MLP *input*."""
    last=model.model.layers[-1]
    u,x=[],[]
    with ExitStack() as hooks:
        hooks.enter_context(last.post_attention_layernorm.register_forward_pre_hook(capture_input(u)))
        hooks.enter_context(last.mlp.register_forward_pre_hook(capture_input(x,stop=True)))
        try:
            model(input_ids=torch.tensor([ids],dtype=torch.long),use_cache=False)
        except TailBoundary:
            pass
        else:
            raise RuntimeError("Source model did not stop before final nonlinear MLP")
    if len(u)!=1 or len(x)!=1 or u[0].shape!=x[0].shape or u[0].shape[1]!=WIDTH:
        raise ValueError("Unqualified last-layer causal observation geometry")
    return u[0],x[0]


def run(snapshot):
    a,compiled_b=compile_readout(snapshot)  # verifies SHA256 of each pinned file
    tokenizer=tokenizers.Tokenizer.from_file(str(snapshot/"tokenizer.json"))
    torch.set_num_threads(1)
    model=AutoModelForCausalLM.from_pretrained(
        snapshot,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,torch_dtype=torch.float32,
        attn_implementation="eager").cpu().eval()
    mlp=model.model.layers[-1].mlp
    if (model.config.model_type!="llama" or len(model.model.layers)!=30 or
            mlp.gate_proj.bias is not None or mlp.up_proj.bias is not None or
            mlp.down_proj.bias is not None or model.lm_head.bias is not None):
        raise ValueError("Trained source model topology does not match compiled Ground")
    source = lambda layer:layer.weight.detach().cpu().numpy().astype(np.float64)
    compiled = compile_nonlinear_cut(a, source(mlp.down_proj),
                                      source(mlp.gate_proj),source(mlp.up_proj))
    if compiled.width!=WIDTH or compiled.channels!=INTERMEDIATE or not np.allclose(
            compiled.b,compiled_b,atol=2e-10,rtol=1e-10):
        raise ValueError("Compiled readout differs from independently source-pinned direction")
    records=[]
    with torch.inference_mode():
        for text in HELDOUT_TEXTS:
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            if not ids:
                raise ValueError("Unqualified empty held-out source prompt")
            u,x=capture_cut(model,ids)
            cuts=[evaluate_cut(compiled,x_i,u_i) for x_i,u_i in zip(x,u)]
            # The independent reference is deliberately *after* all cut decisions.
            full=model(input_ids=torch.tensor([ids],dtype=torch.long),use_cache=False).logits[0]
            if full.shape[0]!=len(cuts) or not torch.isfinite(full).all():
                raise ValueError("Bad independent full-model reference")
            margins=(full[:,YES].double()-full[:,NO].double()).cpu().numpy()
            for i,(cut,margin) in enumerate(zip(cuts,margins)):
                oracle_sign=int(np.sign(margin))
                records.append({"prompt":text,"position":i,"token_id":ids[i],
                    "oracle_fp32_yes_minus_no":float(margin),"oracle_fp32_sign":oracle_sign,
                    "observer_cut":cut,
                    "agrees_with_fp32_oracle": (cut["sign"]==oracle_sign
                        if cut["sign"] is not None else None)})
    decided=[r for r in records if r["observer_cut"]["sign"] is not None]
    total_mlp_channels=sum(r["observer_cut"]["channels_evaluated"] for r in records)
    total_possible=len(records)*INTERMEDIATE
    return {"experiment":"NOC-001","scope":"local-pinned-trained-Smollm135m; real-arithmetic padded bounds vs independent CPU FP32 model",
            "model":MODEL,"revision":REVISION,"file_sha256":HASHES,
            "runtime":{"python":platform.python_version(),"torch":torch.__version__,
                       "numpy":np.__version__,"threads":torch.get_num_threads()},
            "heldout_texts":list(HELDOUT_TEXTS),
            "statistics":{"positions":len(records),"decided":len(decided),
                "correct_decided":sum(bool(r["agrees_with_fp32_oracle"]) for r in decided),
                "incorrect_decided":sum(r["agrees_with_fp32_oracle"] is False for r in decided),
                "channels_evaluated":total_mlp_channels,
                "channels_full_feedforward":total_possible,
                "gate_and_up_dot_products_avoided":2*(total_possible-total_mlp_channels),
                "ratio_feedforward_gate_up_evaluations":total_mlp_channels/total_possible},
            "cost_scope":"Logical gate/up dot-product counts on final block; bound evaluation and model prefix are additional. Independent full-model reference calls excluded from candidate path but included in experiment costs; physical latency not measured.",
            "records":records}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    result=run(args.snapshot)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(result["statistics"],sort_keys=True))
    return int(result["statistics"]["incorrect_decided"]>0)

if __name__=="__main__":
    raise SystemExit(main())
