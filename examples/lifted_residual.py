"""Validate the full first Qwen attention residual at held-out prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from time import perf_counter

import torch

from .composed_attention_row import acquire,dense
from .run_composed_row import head_outputs
from vli.grouped_projection import residual


def main():
    """Compare r=E[t]+W_o concat(z_h) across all coordinates against dense CPU FP64."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vl64",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    root=args.vl64.resolve()
    sys.path[:0]=[str(root),str(root/"src")]
    from experiments.model_fabric.lift_delta_head import PageFiles,Snapshot
    m=json.loads(args.manifest.read_bytes())
    with PageFiles(m["page"]) as files:
        snapshot=Snapshot(files)
        cfg=snapshot.config["text_config"]
        raw,_,total,calls=acquire(snapshot,None)
    typ,weight=raw["out"]
    h=cfg["hidden_size"]
    if typ!="BF16" or m["heads"]!=cfg["linear_num_value_heads"]:
        raise ValueError("Unsupported projection ground")
    w=torch.frombuffer(bytearray(weight),dtype=torch.bfloat16).double().reshape(h,-1)
    width=w.shape[1]
    torch.set_num_threads(1)
    records=[]
    for tokens in ([1761,628,33303],[63204,33303,1761,628]):
        z=[v for output in head_outputs(m,args.directory,tokens) for v in output["head_output"]]
        etype,eb=raw[f"token{tokens[-1]}"]
        if etype!="BF16":
            raise ValueError("Unsupported embedding dtype")
        embedding=[struct.unpack("<f",b"\x00\x00"+word)[0] for word, in struct.iter_unpack("2s",eb)]
        started=perf_counter()
        values,products=residual(weight,z,embedding)
        elapsed=perf_counter()-started
        _,heads=dense(cfg,dict(raw,out=(typ,weight[:2*width])),tokens)
        expected=w@heads.flatten()+torch.frombuffer(bytearray(eb),dtype=torch.bfloat16).double()
        error=float((torch.tensor(values,dtype=torch.float64)-expected).abs().max())
        records.append(dict(tokens=tokens,coordinates=h,grouped_products=products,dense_products=h*width,
                            projection_seconds=elapsed,max_abs_error=error,
                            residual_f64le_sha256=hashlib.sha256(struct.pack(f"<{h}d",*values)).hexdigest()))
    result=dict(page=m["page"],head_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                matrix_bytes=len(weight),matrix_sha256=hashlib.sha256(weight).hexdigest(),
                joint_validation_trained_bytes=total,joint_validation_requests=calls,
                tolerance=1e-10,observations=records,passed=all(r["max_abs_error"]<=1e-10 for r in records))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))
    return int(not result["passed"])


if __name__=="__main__":
    raise SystemExit(main())
