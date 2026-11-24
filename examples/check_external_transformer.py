"""Compare all emitted coordinates to the corresponding pretrained Transformers model."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, BertModel, GPT2LMHeadModel


def check(directory, tokens, observed):
    """Require max|VL(ids)−reference(ids)|≤10⁻⁸ on every named output."""
    c = AutoConfig.from_pretrained(directory, local_files_only=True)
    cls = {"bert":BertModel, "gpt2":GPT2LMHeadModel}[c.model_type]
    model = cls.from_pretrained(directory, local_files_only=True, attn_implementation="eager",
                                torch_dtype=torch.float64).eval()
    with torch.no_grad():
        result = model(torch.tensor([tokens]))
    expected = ({"logits":result.logits[0,-1]} if c.model_type=="gpt2" else
                {"last_hidden_state":result.last_hidden_state[0], "pooler_output":result.pooler_output[0]})
    if observed.keys() != expected.keys():
        raise ValueError("Named observations differ")
    errors = {}
    for name, ref in expected.items():
        output = torch.tensor(observed[name], dtype=torch.float64)
        if output.shape != ref.shape:
            raise ValueError(f"Output geometry differs for {name}")
        errors[name] = (ref-output).abs().max().item()
    report = dict(coordinates=sum(ref.numel() for ref in expected.values()),max_abs_error=errors,
                  tolerance=1e-8,passed=all(e<=1e-8 for e in errors.values()))
    if "logits" in expected:
        report["reference_next_token"] = expected["logits"].argmax().item()
        logits = observed["logits"]
        first, second = sorted(range(len(logits)),key=logits.__getitem__,reverse=True)[:2]
        report["vl_next_token"] = first
        # For this measured pair of vectors, gap_ref ≥ gap_VL − 2||z_VL−z_ref||∞.
        report["top_two_gap"] = logits[first]-logits[second]
        report["observed_reference_gap_lower_bound"] = report["top_two_gap"]-2*errors["logits"]
        report["passed"] &= report["reference_next_token"] == report["vl_next_token"]
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--observation",type=Path,required=True)
    p.add_argument("--tokens",type=int,nargs="+",required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    report = check(args.snapshot,args.tokens,json.loads(args.observation.read_bytes()))
    print(json.dumps(report,allow_nan=False))
    raise SystemExit(int(not report["passed"]))
