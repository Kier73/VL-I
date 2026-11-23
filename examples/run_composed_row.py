"""Read a trained attention row for a new prefix using only the retained head programs."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys


def observe(manifest, directory, tokens):
    """W[row,:]·concat(y_h) = Σ_h W[row,h]·y_h with each compiled h consumed exactly once."""
    from experiments.model_fabric.addressed_delta import read_ground,evaluate
    hashes=manifest["archive_sha256"]
    if len(hashes)!=manifest["heads"] or not hashes:
        raise ValueError("Incomplete head inventory")
    contributions=[]
    for head,expected in enumerate(hashes):
        blob=(directory/f"head-{head:02}.pyz").read_bytes()
        if hashlib.sha256(blob).hexdigest()!=expected:
            raise ValueError(f"Archive identity changed for head {head}")
        program=read_ground(io.BytesIO(blob))
        g=program[0]
        if (g["head"]!=head or g["output_row"]!=manifest["row"] or
                g["alphabet"]!=sorted(manifest["alphabet"]) or g["layer"]!=0 or
                manifest["page"]!=f"https://huggingface.co/{g['model']}/tree/{g['revision']}"):
            raise ValueError("Inconsistent compiled coordinate")
        contributions.append(evaluate(program,tokens)["row_contribution"])
    return math.fsum(contributions)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vl64",type=Path,required=True)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("tokens",type=int,nargs="+")
    args=parser.parse_args()
    root=args.vl64.resolve()
    sys.path[:0]=[str(root),str(root/"src")]
    print(observe(json.loads(args.manifest.read_bytes()),args.directory,args.tokens))
