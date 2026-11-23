"""Compare a page-discovered VL-64 continuation to a separately loaded pretrained decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    """Accept iff argmax(VL)=argmax(HF) and max_i |VL_i−HF_i|≤2·10⁻⁴."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vl64", type=Path, required=True)
    parser.add_argument("--reference-snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.vl64.resolve()
    sys.path[:0] = [str(root), str(root / "src")]
    from experiments.model_fabric.model_seed_reference import MODEL, REVISION, PINS, file_digest
    from tokenizers import Tokenizer
    import torch
    import transformers

    snapshot = args.reference_snapshot
    if any(file_digest(snapshot / name) != pin for name, pin in PINS.items()):
        raise ValueError("Local reference differs from pinned weights or config")
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    archive = root / "artifacts/VL64-model.pyz"
    seed_hash = file_digest(archive)
    command = [sys.executable, "-I", "-S", str(archive), "run", "--search", MODEL,
               "--stage-bytes", "540000000", "--include-logits", "--tokens", *map(str, tokens)]
    report = json.loads(subprocess.check_output(command, text=True))
    if (report["source_revision"] != REVISION or report["model"] != MODEL
            or report["staged_sha256"] != {"model.safetensors": PINS["model.safetensors"]}
            or report["input_ids"] != tokens or file_digest(archive) != seed_hash):
        raise ValueError("Candidate source identity or input differs from reference")
    torch.set_num_threads(1)
    reference = transformers.AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32, attn_implementation="eager").eval()
    with torch.inference_mode():
        expected = reference(torch.tensor([tokens]), use_cache=False).logits[0, -1].numpy()
    actual, = np.asarray(report.pop("logits"), dtype="<f8")
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Invalid vocabulary geometry or numerical result")
    error = float(np.max(np.abs(actual - expected)))
    winner = int(expected.argmax())
    agrees = report["generated_ids"] == [winner] and int(actual.argmax()) == winner and error <= 2e-4
    report.update(prompt=prompt, tokenizer_sha256=file_digest(snapshot / "tokenizer.json"),
                  tokenizer_scope="external tokenizer for shared reference and seed input IDs",
                  seed_sha256=seed_hash, seed_bytes=archive.stat().st_size,
                  continuation=tokenizer.decode(report["generated_ids"]), reference_id=winner,
                  torch=str(torch.__version__), transformers=transformers.__version__,
                  vocabulary=actual.size, max_abs_logit_error=error, tolerance=2e-4, agrees=agrees,
                  logits_f64le_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(not agrees)


if __name__ == "__main__":
    raise SystemExit(main())
