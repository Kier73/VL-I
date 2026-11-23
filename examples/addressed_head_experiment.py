"""Challenge a portable trained Qwen head on unseen contexts and a new token address."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

PAGE = "https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0"
ALPHABET = [63204, 628, 1761, 33303]
NEW = 948


def reference(g, weights, tokens):
    """Run the forward dense state S←fS+βk(v−(fS)ᵀk)ᵀ with PyTorch F64 on original operands."""
    w = {name: torch.from_numpy(array) for name,array in weights.items()}
    x = w["embedding"][[g["alphabet"].index(t) for t in tokens]]
    x = x * torch.rsqrt(x.square().mean(dim=-1,keepdim=True) + g["eps"]) * (1+w["norm"])
    k, v, taps = g["key_width"], g["value_width"], g["conv_width"]
    channels = 2*k+v
    pre = F.linear(x,w["qkv"]).T.unsqueeze(0)
    mixed = F.silu(F.conv1d(F.pad(pre,(taps-1,0)),w["conv"].unsqueeze(1),groups=channels))[0].T
    queries, keys, values = mixed.split((k,k,v),dim=-1)
    queries = queries * torch.rsqrt(queries.square().sum(-1,keepdim=True)+1e-6) / k**0.5
    keys = keys * torch.rsqrt(keys.square().sum(-1,keepdim=True)+1e-6)
    betas = torch.sigmoid(x@w["b"])
    fades = torch.exp(-w["alog"].exp() * F.softplus(x@w["a"]+w["dt"]))
    state = torch.zeros((k,v),dtype=torch.float64)
    for key,value,beta,fade in zip(keys,values,betas,fades):
        state = fade*state
        state = state + torch.outer(key,beta*(value-key@state))
    core = queries[-1]@state
    out = core*torch.rsqrt(core.square().mean()+g["eps"])*w["gain"]*F.silu(w["z"]@x[-1])
    return out, float(w["row"]@out)


def main():
    """Accept if every coordinate has error ≤ τ, two independent pins match, and history matters."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vl64",required=True,type=Path)
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args = p.parse_args()
    upstream = args.vl64.resolve()
    sys.path[:0] = [str(upstream),str(upstream/"src")]
    from experiments.model_fabric.lift_delta_head import lift, package
    g, data, source, _ = lift(PAGE, ALPHABET)
    archive, expanded = package(g,data)
    path = args.archive.resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(archive)
    # Freeze executable bytes before forming any evaluation contexts or reference outputs.
    sha = hashlib.sha256(archive).hexdigest()
    dense_ground, _, _, weights = lift(PAGE, ALPHABET+[NEW])
    torch.set_num_threads(1)
    prior = Path(__file__).resolve().parents[1]/"experiments/pinned_text/datasets/2026-11-19-real-paragraph-token-causal-results.json"
    anchors = json.loads(prior.read_text())["original_sources"]["qwen-27b"]["observed_steps"]
    rng = random.Random(73027)
    suffix = [628,1761,33303,63204,628]
    contexts = [[63204], [63204,628], [63204]+suffix, [33303]+suffix,
                [rng.choice(ALPHABET) for _ in range(32)], [628]*16,
                [63204,NEW,628,NEW]]
    report = dict(ground=g, source=source, archive_sha256=sha, archive_bytes=len(archive),
                  expanded_code_and_data_bytes=expanded, tolerance=1e-10, observations=[])
    with tempfile.TemporaryDirectory() as tmp:
        for i, tokens in enumerate(contexts):
            flags = ["--resolve"] if NEW in tokens else []
            cmd = [sys.executable,"-I","-S",str(path),*flags,*map(str,tokens)]
            result = json.loads(subprocess.check_output(cmd,cwd=tmp,text=True))
            vec,row = reference(dense_ground,weights,tokens)
            error = float((vec-torch.tensor(result["head_output"],dtype=torch.float64)).abs().max())
            row_error = abs(row-result["row_contribution"])
            anchor_error = abs(row-anchors[i]["selected_head_out_row0_contribution"]) if i<2 else None
            report["observations"].append(dict(tokens=tokens,row=result["row_contribution"],
                max_vector_error=error,row_error=row_error,historical_pin_error=anchor_error,
                trained_bytes_read=result["trained_bytes_read"],passed=max(error,row_error,anchor_error or 0)<=1e-10
                and (result["trained_bytes_read"]>0)==bool(flags)))
        refusal = subprocess.run([sys.executable,"-I","-S",str(path),str(NEW)],cwd=tmp,capture_output=True,text=True)
    rows = report["observations"]
    report["same_suffix_row_difference"] = abs(rows[2]["row"]-rows[3]["row"])
    report["uncached_offline_refused"] = refusal.returncode!=0 and "outside compiled alphabet" in refusal.stderr
    report["passed"] = (all(r["passed"] for r in rows) and report["same_suffix_row_difference"]>1e-10
                        and report["uncached_offline_refused"] and hashlib.sha256(path.read_bytes()).hexdigest()==sha)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("passed","archive_bytes","expanded_code_and_data_bytes","same_suffix_row_difference")}))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
