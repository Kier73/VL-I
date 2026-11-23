"""Find a distinguishing suffix on the previously verified Qwen head archive."""
import argparse
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sys


def probe(archive, runtime):
    """Find h,j,a with Q(y_h)=Q(y_j) and Q(y_ha)≠Q(y_ja), Q(y)=round(10⁴y)."""
    program = runtime.read_ground(archive)
    alphabet = program[0]["alphabet"]
    buckets = {}
    for history in product(alphabet, repeat=2):
        y = runtime.evaluate(program, history)["row_contribution"]
        buckets.setdefault(round(10000*y), []).append((history, y))
    for bucket, members in buckets.items():
        for (h,y), (j,z) in combinations(members, 2):
            for a in alphabet:
                after = [runtime.evaluate(program, prefix+(a,))["row_contribution"] for prefix in (h,j)]
                if round(10000*after[0]) != round(10000*after[1]):
                    return dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                histories=[h,j],before=[y,z],before_bin=bucket,token=a,after=after,
                                after_bins=[round(10000*v) for v in after])
    raise ValueError("No distinguishing suffix within the tested domain")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vl64", type=Path, required=True)
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    root = args.vl64.resolve()
    sys.path[:0] = [str(root), str(root/"src")]
    from experiments.model_fabric import addressed_delta
    print(json.dumps(probe(args.archive,addressed_delta), indent=2))
