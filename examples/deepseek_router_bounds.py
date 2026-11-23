"""Bound the six winning DeepSeek text experts using exact dyadic residual energies."""
import argparse
from contextlib import closing
from decimal import Decimal, localcontext
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "08c8657f02291e11b6f6b63e0e16014f8e748518"
GATE = "layers.0.ffn.gate."
EXPERTS, WIDTH, TOP, BLOCK = 384, 5120, 6, 128
TOKENS = (42, 75, 63204)


def bf16(raw):
    """BF16 bits b denote the F32 value whose bit representation is b<<16."""
    return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")


def dyadics(values):
    """Express x_i=X_i/d exactly, where d is the largest power-of-two denominator."""
    pairs = [float(x).as_integer_ratio() for x in values]
    d = max(q for _, q in pairs)
    return [p*(d//q) for p,q in pairs], d


def norm2(xs):
    """E(x)=Σ_i x_i² in unbounded integer arithmetic."""
    return sum(x*x for x in xs)


def score_endpoint(num, den, bias, upper):
    """Outward enclosure of f(num/den)+bias, f(t)=sqrt(ln(1+exp(t))), by monotonicity."""
    # Decimal exp, ln, sqrt are correctly rounded to nearest. Widen after
    # every operation, including division and addition, by one decimal ULP.
    with localcontext() as ctx:
        ctx.prec = 50
        widen = ctx.next_plus if upper else ctx.next_minus
        t = widen(Decimal(num)/Decimal(den))
        ex = widen(t.exp())
        sp = widen(widen(1+ex).ln())
        root = widen(max(Decimal(0), sp).sqrt())
        return widen(root + bias)


def row_bounds(s, rw, rx, den, bias):
    """|Σ_unread W_i X_i| ≤ sqrt(rw·rx); ceil of that root is an exact integer bound."""
    square = rw*rx
    radius = math.isqrt(square)
    radius += radius*radius != square
    return (score_endpoint(s-radius,den,bias,False), score_endpoint(s+radius,den,bias,True))


def compile_norms(path):
    """Retain only one scale d_e and E_e=Σ(d_e w_ei)² per trained expert row."""
    summary = []
    with path.open("rb") as f:
        for _ in range(EXPERTS):
            w,d = dyadics(bf16(f.read(WIDTH*2)))
            summary.append((d,norm2(w)))
    return summary


def route(path, summary, bias, probe):
    """Drop e only when U_e < kth-largest L; return a set only when exactly k survive."""
    x, dx = dyadics(probe)
    blocks = [(start, norm2(x[start:start+BLOCK])) for start in range(0,WIDTH,BLOCK)]
    blocks.sort(key=lambda pair: -pair[1])
    rx = norm2(x)
    sums = [0]*EXPERTS
    rw = [energy for _,energy in summary]
    active = list(range(EXPERTS))
    trace = []
    reads = 0
    with path.open("rb") as f:
        for step in range(len(blocks)+1):
            bounds = {e: row_bounds(sums[e],rw[e],rx,summary[e][0]*dx,bias[e]) for e in active}
            threshold = sorted((v[0] for v in bounds.values()),reverse=True)[TOP-1]
            active = [e for e in active if bounds[e][1] >= threshold]
            trace.append(len(active))
            if len(active)==TOP or step==len(blocks):
                break
            start,energy = blocks[step]
            rx -= energy
            segment = x[start:start+BLOCK]
            for e in active:
                f.seek(2*(e*WIDTH+start))
                raw = f.read(BLOCK*2)
                if len(raw)!=2*BLOCK:
                    raise ValueError("Incomplete trained row block")
                ints,d = dyadics(bf16(raw))
                scale,rem = divmod(summary[e][0],d)
                if rem:
                    raise ValueError("Summary source differs from row encoding")
                sums[e] += scale*sum(w*z for w,z in zip(ints,segment))
                rw[e] -= scale*scale*norm2(ints)
                reads += 1
    return {"certified_set":active if len(active)==TOP else None, "survivors":trace,
            "query_file_reads":reads, "query_gate_bytes":reads*2*BLOCK}


def acquire(files, parser):
    """Map each selected tensor interval [l,h) to [header+l,header+h) at the pinned revision."""
    index = files.metadata("model.safetensors.index.json",16<<20)
    config = files.metadata("config.json",1<<20)
    c = json.loads(config)["text_config"]
    if (c["hidden_size"],c["n_routed_experts"],c["num_experts_per_tok"],c["scoring_func"],
            c.get("gate_temp",1)) != (WIDTH,EXPERTS,TOP,"sqrtsoftplus",1):
        raise ValueError("Unexpected pinned router semantics")
    mapping = json.loads(index)["weight_map"]
    headers = {}
    report = {"index_sha256":sha256(index).hexdigest(), "config_sha256":sha256(config).hexdigest(),
              "metadata_bytes":len(index)+len(config), "slices":{}}

    def tensor(key, dtype, shape, row=None):
        """For row r, read offset o+r·width·sizeof(dtype); otherwise acquire the whole tensor."""
        shard = mapping[key]
        if shard not in headers:
            lead,size = files.read_range(shard,0,8)
            n = int.from_bytes(lead,"little")
            if not 2 <= n <= 2**20:
                raise ValueError("Unexpected header extent")
            body,total = files.read_range(shard,8,n)
            if size!=total:
                raise ValueError("Source length changed")
            _,entries = parser(lead+body,filename=shard,file_size=size,
                               expected_names={k for k,v in mapping.items() if v==shard})
            headers[shard] = n+8, entries, size
            report["metadata_bytes"] += n+8
        prefix,entries,size = headers[shard]
        desc = entries[key]
        if desc["dtype"]!=dtype or desc["shape"]!=shape:
            raise ValueError(f"Unexpected {key} geometry")
        lo,hi = desc["data_offsets"]
        if row is not None:
            width = (hi-lo)//shape[0]
            if not 0<=row<shape[0]:
                raise ValueError("Invalid probe row")
            lo,hi = lo+row*width,lo+(row+1)*width
        raw,total = files.read_range(shard,prefix+lo,hi-lo)
        if size!=total or len(raw)!=hi-lo:
            raise ValueError("Incomplete acquisition")
        report["slices"][key if row is None else f"{key}[{row}]"] = {
            "shard":shard,"offset":prefix+lo,"bytes":len(raw),"sha256":sha256(raw).hexdigest()}
        return raw

    gate = tensor(GATE+"weight","BF16",[EXPERTS,WIDTH])
    bias = tensor(GATE+"bias","F32",[EXPERTS])
    probes = [tensor("embed.weight","BF16",[c["vocab_size"],WIDTH],row=t) for t in TOKENS]
    return gate,bias,probes,report


def main():
    """Commit certified sets before dense FP32 routing; agreement compares sets, not mixture weights."""
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vl64",required=True,type=Path,help="VL-64 checkout at bb0aac8 or compatible")
    parser.add_argument("--out",required=True,type=Path)
    args = parser.parse_args()
    upstream = args.vl64.resolve()
    sys.path[:0] = [str(upstream),str(upstream/"src")]
    from experiments.model_fabric.local_seed import HubFiles
    from experiments.model_fabric.scaled_model_fabric import admit_shard_header
    with closing(HubFiles(MODEL,REVISION)) as files:
        gate,raw_bias,probes,source = acquire(files,admit_shard_header)
    biases = np.frombuffer(raw_bias,"<f4")
    decimals = [Decimal.from_float(float(b)) for b in biases]
    with TemporaryDirectory() as temp:
        path = Path(temp)/"gate.bf16"
        path.write_bytes(gate)
        summary = compile_norms(path)
        candidates = [route(path,summary,decimals,bf16(raw)) for raw in probes]
    torch.set_num_threads(1)
    w = torch.from_numpy(bf16(gate).reshape(EXPERTS,WIDTH))
    for token,probe,trial in zip(TOKENS,probes,candidates):
        dot = torch.mv(w,torch.from_numpy(bf16(probe)))
        scores = torch.nn.functional.softplus(dot).sqrt() + torch.tensor(biases)
        ranking = scores.topk(TOP+1)
        reference = sorted(ranking.indices[:TOP].tolist())
        trial.update(embedding_id=token, native_fp32_set=reference, agrees=trial["certified_set"]==reference,
                     fp32_selection_margin=float(ranking.values[TOP-1]-ranking.values[TOP]))
    report = {"model":MODEL,"revision":REVISION,"layer":0,"mode":"text-selection-bias",
              "input_scope":"actual raw embedding rows supplied at router boundary; no causal prefix executed",
              "source":source,"gate_bytes":len(gate),"cold_summary_gate_bytes_read":len(gate),
              "summary_json_bytes":len(json.dumps(summary,separators=(",",":")).encode()),
              "block_coordinates":BLOCK,"query_network_bytes":0,"torch":str(torch.__version__),"trials":candidates}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print([(t["embedding_id"],t["agrees"],t["query_gate_bytes"]) for t in candidates])
    return int(any(not t["agrees"] for t in candidates))


if __name__=="__main__":
    raise SystemExit(main())
