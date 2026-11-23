"""Materialize one model-addressed Qwen first-attention residual using standard-library workers."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

from .run_composed_row import head_outputs
from vli.grouped_projection import residual


def main():
    """Resolve (model, revision, layer=0, prefix) → encoded r ∈ ℝ^hidden, little-endian FP64."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vl64",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("tokens",nargs="+",type=int)
    args=p.parse_args()
    root=args.vl64.resolve()
    sys.path[:0]=[str(root),str(root/"src")]
    from experiments.model_fabric.local_seed import Snapshot
    from experiments.model_fabric.page_files import PageFiles
    m=json.loads(args.manifest.read_bytes())
    z=[v for out in head_outputs(m,args.directory,args.tokens) for v in out["head_output"]]
    with PageFiles(m["page"]) as files:
        snap=Snapshot(files)
        cfg=snap.config["text_config"]
        h=cfg["hidden_size"]
        descriptors={t["name"]:t for t in snap.ground["tensors"]["tensors"]}
        def fetch(name,shape,start,count):
            """Bind byte interval [header+lo+2s,header+lo+2(s+n)) to a BF16 tensor."""
            shard,header,_,lo,hi,size=snap.locations[name]
            tensor=descriptors[name]
            if (tensor["dtype"]!="BF16" or tensor["shape"]!=shape or start<0 or
                    2*(start+count)>hi-lo):
                raise ValueError("Operand identity or extent disagrees with plan")
            data=bytearray()
            while len(data)<2*count:
                n=min(4<<20,2*count-len(data))
                piece,actual=files.read_range(shard,header+lo+2*start+len(data),n)
                if actual!=size or len(piece)!=n:
                    raise ValueError("Incomplete source interval")
                data.extend(piece)
            return data
        matrix=fetch("model.language_model.layers.0.linear_attn.out_proj.weight",[h,len(z)],0,h*len(z))
        embedding=fetch("model.language_model.embed_tokens.weight",[cfg["vocab_size"],h],args.tokens[-1]*h,h)
    e=[struct.unpack("<f",b"\x00\x00"+word)[0] for word, in struct.iter_unpack("2s",embedding)]
    values,products=residual(matrix,z,e)
    encoded=struct.pack(f"<{h}d",*values)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(encoded)
    print(json.dumps(dict(tokens=args.tokens,coordinates=h,trained_bytes=len(matrix)+len(embedding),
                          grouped_products=products,residual_f64le_sha256=hashlib.sha256(encoded).hexdigest())))


if __name__=="__main__":
    main()
