"""Replay pinned public model descriptors against the same guarded VL-64 dispatcher."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen


def summarize(seed,survey,out):
    """Bind H(seed), H(survey) and H(rows) to the observed status histogram."""
    blob = out.read_bytes()
    counts = Counter(json.loads(line)["result"]["status"] for line in blob.splitlines())
    summary = dict(counts=dict(counts),trained_bytes_read=0,
                   seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                   prior_survey_sha256=hashlib.sha256(survey.read_bytes()).hexdigest(),
                   output_sha256=hashlib.sha256(blob).hexdigest())
    out.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return counts


def main():
    """Check SHA256(D)=pinned hash for each unique observed D, then run with zero trained bytes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=Path,required=True)
    p.add_argument("--survey",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.seed.resolve()),str(Path(__file__).resolve().parent)]
    from architecture_dry_run import probe,Descriptor
    from vld.external_transformer import run
    from vld.page_files import PageFiles
    cases = {}
    for row in json.loads(a.survey.read_bytes())["rows"]:
        if row["status"]=="observed":
            d = row["observation"]
            key = (d["model"],d["revision"],d["descriptor"])
            record = cases.setdefault(key,dict(d,axes=[]))
            record["axes"].append([row["axis"],row["key"]])

    def measure(key):
        """Compare execution on the verified descriptor with the admitted reader's first stop."""
        model,rev,name = key
        row = cases[key]
        url = f"https://huggingface.co/{model}/resolve/{rev}/{name}"
        row["url"] = url
        try:
            with urlopen(url,timeout=20) as response:
                raw = response.read(row["bytes"]+1)
            if len(raw)!=row["bytes"] or hashlib.sha256(raw).hexdigest()!=row["sha256"]:
                raise ValueError("Pinned descriptor identity differs")
            doc = json.loads(raw)
            pair = (doc.get("model_type"),tuple(doc.get("architectures") or []))
            if name=="config.json" and pair in {("bert",("BertModel",)),("gpt2",("GPT2LMHeadModel",)),
                                                ("resnet",("ResNetForImageClassification",))}:
                row["basis"] = "actual_page_and_headers"
                with PageFiles(f"https://huggingface.co/{model}/tree/{rev}") as source:
                    row["result"] = probe(run,source,image=pair[0]=="resnet")
            else:
                row["basis"] = "actual_descriptor_empty_tensor_inventory"
                row["result"] = probe(run,Descriptor(doc if name=="config.json" else None))
        except (OSError,URLError,ValueError) as exc:
            row["result"] = dict(status="unresolved_source",reason=str(exc))
        return row

    with a.out.open("w",encoding="utf-8",newline="\n") as out, ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(measure,sorted(cases)):
            out.write(json.dumps(row,sort_keys=True)+"\n")
    print(summarize(a.seed,a.survey,a.out))


if __name__=="__main__":
    main()
