"""Qualify an isolated VL-64 interpreter on the same checkpoint from two sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def invoke(seed, arguments):
    """Read a receipt from the isolated process implementing O=F(source,ids)."""
    run = subprocess.run([sys.executable,"-I","-S",str(seed),*arguments],check=True,
                         capture_output=True,text=True)
    return json.loads(run.stdout)


def main():
    """Require O_page=O_file and compare O_file against an independent dense evaluation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=Path,required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--page",required=True)
    p.add_argument("--tokens",type=int,nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    report = dict(source_page=a.page,seed_sha256=hashlib.sha256(a.seed.read_bytes()).hexdigest(),
                  seed=invoke(a.seed,["seed-verify"]))
    with TemporaryDirectory() as tmp:
        outputs = []
        for mode,source in (("snapshot",a.snapshot),("page",a.page)):
            dest = Path(tmp)/(mode+".json")
            report[mode] = invoke(a.seed,["--"+mode,str(source),"--tokens",*map(str,a.tokens),"--out",str(dest)])
            outputs.append(dest.read_bytes())
        report["identical_observations"] = outputs[0]==outputs[1]
        import torch
        from .check_external_transformer import check
        torch.set_num_threads(1)
        report["reference"] = check(a.snapshot,a.tokens,json.loads(outputs[0]))
        report["passed"] = report["identical_observations"] and report["reference"]["passed"]
    a.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
