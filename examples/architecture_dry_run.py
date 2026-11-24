"""Challenge the shipped external-model dispatcher without supplying trained payload bytes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys


class PayloadRequested(Exception):
    """Boundary at the first [start,end) inside a trained payload."""


class HeaderGuard:
    def __init__(self, source):
        """Authorize only S[0:8] and S[8:8+header_length] for each shard S."""
        self.source = source
        self.model, self.revision, self.authority = source.model, source.revision, source.authority
        self.headers = {}
        self.header_bytes = 0

    def metadata(self,name,limit):
        """Return D[name] only for the three JSON descriptors used by the seed."""
        if name not in ("config.json","model.safetensors.index.json","preprocessor_config.json"):
            raise AssertionError(name)
        return self.source.metadata(name,limit)

    def read_range(self,name,start,length):
        """Observe the header; raise before forwarding any payload request to the provider."""
        if start==0 and length==8:
            data,total = self.source.read_range(name,start,length)
            self.headers[name], = struct.unpack("<Q",data)
        elif start==8 and name in self.headers and length==self.headers[name]:
            data,total = self.source.read_range(name,start,length)
        else:
            raise PayloadRequested(name,start,length)
        self.header_bytes += len(data)
        return data,total


class Descriptor:
    """Registration-only fixture with a valid empty Safetensors inventory."""
    model,revision,authority = "registration","registry","metadata-fixture"

    def __init__(self,config):
        """D(config.json)=config, or D(config.json)=∅ for a pipeline-directory layout."""
        self.config = config

    def metadata(self,name,limit):
        """Supply only the known registration fields, never synthesized weight descriptors."""
        if name=="config.json" and self.config is not None:
            return json.dumps(self.config).encode()
        raise FileNotFoundError(name)

    def read_range(self,name,start,length):
        """S=LE64(2)||'{}' has header extent 10 and payload extent zero."""
        blob = struct.pack("<Q",2)+b"{}"
        return blob[start:start+length],len(blob)


def probe(run, source, image=False):
    """Classify the observed first stop of run(D,input) with payload-byte budget B=0."""
    guard = HeaderGuard(source)
    try:
        run(guard, None if image else [0], rgb=bytes(224*224*3) if image else None)
    except PayloadRequested as stop:
        return dict(status="first_weight_requested",request=list(stop.args),header_bytes=guard.header_bytes)
    except (KeyError,ValueError,FileNotFoundError) as error:
        return dict(status="refused",exception=type(error).__name__,reason=str(error),header_bytes=guard.header_bytes)
    raise AssertionError("Unexpected completion without a trained operand")


LIVE = {
    "distilbert/distilgpt2":"2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "sentence-transformers/all-MiniLM-L6-v2":"1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "microsoft/resnet-18":"65a5785d9156231087c481e0c7dd33a5ff6f7e3e",
    "microsoft/resnet-50":"34c2154c194f829b11125337b98c8f5f9965ff19",
    "timm/resnet18.a1_in1k":"491b427b45c94c7fb0e78b5474cc919aff584bbf",
    "openai/whisper-tiny":"169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "google/vit-base-patch16-224":"3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
    "stable-diffusion-v1-5/stable-diffusion-v1-5":"451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    "state-spaces/mamba-130m-hf":"1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
}


def main():
    """For each registration r record its source, observed first stop and next development."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=Path,required=True)
    p.add_argument("--archives",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.seed.resolve()),str(Path(__file__).resolve().parent)]
    from vld.external_transformer import run
    from vld.page_files import PageFiles
    from dry_run_inventory import inventory, PINS
    a.out.mkdir(parents=True,exist_ok=True)
    live = {}
    for model,rev in LIVE.items():
        url = f"https://huggingface.co/{model}/tree/{rev}"
        try:
            with PageFiles(url) as source:
                result = probe(run,source,image="resnet" in model or "vit-" in model)
        except (OSError,ValueError) as error:
            result = dict(status="source_unavailable",exception=type(error).__name__,reason=str(error))
        live[model] = dict(page=url,**result)
    controls = {("bert","BertModel"):"sentence-transformers/all-MiniLM-L6-v2",
                ("gpt2","GPT2LMHeadModel"):"distilbert/distilgpt2",
                ("resnet","ResNetForImageClassification"):"microsoft/resnet-18"}
    counts = Counter()
    records = inventory(a.archives)
    with (a.out/"inventory.jsonl").open("w",encoding="utf-8",newline="\n") as out:
        for row in records:
            family,cls = row["family"],row["entry"]
            provider = row["catalogue"]
            model = controls.get((family,cls)) if provider=="transformers" else None
            if model:
                result = live[model]
            else:
                config = {"model_type":family,"architectures":[cls] if cls else []}
                if provider=="timm":
                    config = {"architecture":cls}
                if provider in ("diffusers","sklearn"):
                    config = None
                result = probe(run,Descriptor(config))
            row.update(probe_kind="pinned_model" if model else "registration_fixture",result=result)
            # This public class identifier triggered repository push protection. Keep the
            # source-coordinate witness (pinned file,line) instead of publishing the text.
            if family=="mistral3" and row.get("tasks") and len(cls)==32:
                row["entry"] = None
                row["entry_record"] = "See pinned upstream class declaration; literal omitted after publication scan."
                row["result"] = {k:v for k,v in result.items() if k!="reason"}
                row["result"]["reason_category"] = "unimplemented model or input semantics"
            counts[provider,result["status"]] += 1
            out.write(json.dumps(row,sort_keys=True,ensure_ascii=True)+"\n")
    summary = dict(seed_sha256=hashlib.sha256(a.seed.read_bytes()).hexdigest(),
                   inventory_sha256=hashlib.sha256((a.out/"inventory.jsonl").read_bytes()).hexdigest(),
                   pins=PINS,live=live,counts={f"{k}/{s}":n for (k,s),n in sorted(counts.items())},
                   trained_bytes_read=0)
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(summary["counts"])


if __name__=="__main__":
    main()
