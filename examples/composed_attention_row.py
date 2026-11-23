"""Compile Qwen layer zero into head programs and challenge their summed output row."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from .addressed_head_experiment import PAGE, ALPHABET


def acquire(snapshot, row):
    """Fetch the union of required intervals once: full projections, E[t] rows, and W_out[row,:]."""
    cfg = snapshot.config["text_config"]
    hidden = cfg["hidden_size"]
    output_width = cfg["linear_num_value_heads"]*cfg["linear_value_head_dim"]
    prefix = "model.language_model.layers.0."
    raw, cache = {}, {}
    total = calls = 0
    specs = [(f"token{t}", "model.language_model.embed_tokens.weight", t*hidden, hidden) for t in ALPHABET]
    specs.append(("norm", prefix+"input_layernorm.weight",0,None))
    for key, name in (("qkv","in_proj_qkv.weight"),("z","in_proj_z.weight"),("a","in_proj_a.weight"),
                      ("b","in_proj_b.weight"),("conv","conv1d.weight"),("alog","A_log"),
                      ("dt","dt_bias"),("gain","norm.weight"),("out","out_proj.weight")):
        start,count = (row*output_width,output_width) if key=="out" else (0,None)
        specs.append((key,prefix+"linear_attn."+name,start,count))
    dtype = {t["name"]: t["dtype"] for t in snapshot.ground["tensors"]["tensors"]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key,name,first,count in specs:
            shard,header,_,lo,hi,size = snapshot.locations[name]
            encoding = dtype[name]
            width = {"BF16":2,"F32":4,"F16":2}[encoding]
            start = header+lo+first*width
            length = hi-lo if count is None else count*width
            if not header+lo <= start < start+length <= header+hi:
                raise ValueError("Invalid operand span")
            jobs = [pool.submit(snapshot.files.read_range,shard,start+j,min(4<<20,length-j))
                    for j in range(0,length,4<<20)]
            chunks = [job.result() for job in jobs]
            data = b"".join(piece for piece,_ in chunks)
            if len(data)!=length or any(n!=size for _,n in chunks):
                raise ValueError("Incomplete source observation")
            cache.setdefault(shard,[]).append((start,data,size))
            raw[key] = (encoding,data)
            total += length
            calls += len(jobs)

    def read(shard,start,length):
        """Restrict the observed span [b,b+m) to [a,a+n) ⊆ [b,b+m)."""
        for base,data,size in cache[shard]:
            offset = start-base
            if 0 <= offset and offset+length <= len(data):
                return data[offset:offset+length],size
        raise ValueError("Requested an unobserved operand")
    return raw,read,total,calls


def dense(cfg, raw, tokens):
    """Project the unsliced matrices, update all S_h forward, then dot one W_out row with concat_h(z_h)."""
    def tensor(key):
        """Decode the hosted little-endian tensor into FP64 reference coordinates."""
        dtype,b = raw[key]
        typ = {"BF16":torch.bfloat16,"F32":torch.float32,"F16":torch.float16}[dtype]
        return torch.frombuffer(bytearray(b),dtype=typ).double()
    x = torch.stack([tensor(f"token{t}") for t in tokens])
    eps = cfg["rms_norm_eps"]
    x = x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*(1+tensor("norm"))
    def project(key):
        """Y = X Wᵀ for the entire original projection."""
        return F.linear(x,tensor(key).reshape(-1,x.shape[-1]))
    h, groups = cfg["linear_num_value_heads"],cfg["linear_num_key_heads"]
    k,v,taps = cfg["linear_key_head_dim"],cfg["linear_value_head_dim"],cfg["linear_conv_kernel_dim"]
    projected = project("qkv").T.unsqueeze(0)
    conv = F.conv1d(F.pad(projected,(taps-1,0)),tensor("conv").reshape(-1,1,taps),groups=projected.shape[1])
    q,key,value = F.silu(conv)[0].T.split([groups*k,groups*k,h*v],dim=1)
    def unit(a):
        """Broadcast normalized key-group vectors to h value heads."""
        a = a.reshape(-1,groups,k)
        return (a*torch.rsqrt(a.square().sum(-1,keepdim=True)+1e-6)).repeat_interleave(h//groups,dim=1)
    q,key,value = unit(q)/math.sqrt(k),unit(key),value.reshape(-1,h,v)
    beta = project("b").sigmoid()
    fade = (-tensor("alog").exp()*F.softplus(project("a")+tensor("dt"))).exp()
    state = torch.zeros(h,k,v,dtype=torch.float64)
    for i in range(len(tokens)):
        state *= fade[i,:,None,None]
        delta = beta[i,:,None]*(value[i]-torch.einsum("hk,hkv->hv",key[i],state))
        state += key[i,:,:,None]*delta[:,None,:]
    y = torch.einsum("hk,hkv->hv",q[-1],state)
    y *= torch.rsqrt(y.square().mean(-1,keepdim=True)+eps)*tensor("gain")*F.silu(project("z")[-1].reshape(h,v))
    return float(tensor("out")@y.flatten()),y


def main():
    """Accept iff all head vectors and their summed row agree with the full projection to 10⁻¹⁰."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vl64",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--row",type=int,default=0)
    args=p.parse_args()
    root=args.vl64.resolve()
    sys.path[:0]=[str(root),str(root/"src")]
    from experiments.model_fabric.lift_delta_head import PageFiles,Snapshot,lift_from_snapshot,package
    from experiments.model_fabric.addressed_delta import read_ground,evaluate
    torch.set_num_threads(1)
    args.directory.mkdir(parents=True,exist_ok=True)
    programs,hashes,sizes=[],[],[]
    logical=0
    with PageFiles(PAGE) as files:
        snapshot=Snapshot(files)
        cfg=snapshot.config["text_config"]
        raw,read,total,calls=acquire(snapshot,args.row)
        for head in range(cfg["linear_num_value_heads"]):
            g,data,source,_=lift_from_snapshot(snapshot,PAGE,ALPHABET,head,args.row,read)
            blob,expanded=package(g,data)
            path=args.directory/f"head-{head:02}.pyz"
            path.write_bytes(blob)
            programs.append(read_ground(path))
            hashes.append(hashlib.sha256(blob).hexdigest())
            sizes.append((len(blob),expanded))
            logical+=source["trained_bytes"]
    observations=[]
    for tokens in ([628,33303,1761],[33303,628,63204,1761],[628,33303,63204,1761,628,33303]):
        outputs=[evaluate(program,tokens) for program in programs]
        row=math.fsum(item["row_contribution"] for item in outputs)
        expected,vectors=dense(cfg,raw,tokens)
        head_error=float((vectors-torch.tensor([o["head_output"] for o in outputs],dtype=torch.float64)).abs().max())
        observations.append(dict(tokens=tokens,row=row,reference=expected,abs_error=abs(row-expected),
                                 max_head_coordinate_error=head_error))
    passed=all(max(o["abs_error"],o["max_head_coordinate_error"])<=1e-10 for o in observations)
    result=dict(page=PAGE,row=args.row,alphabet=ALPHABET,heads=len(programs),trained_http_bytes=total,
                trained_requests=calls,metadata_bytes=snapshot.metadata_bytes,logical_head_operand_bytes=logical,
                total_archive_bytes=sum(x for x,_ in sizes),max_expanded_worker_bytes=max(y for _,y in sizes),
                archive_sha256=hashes,observations=observations,tolerance=1e-10,passed=passed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(observations,flush=True)
    return int(not passed)


if __name__=="__main__":
    raise SystemExit(main())
