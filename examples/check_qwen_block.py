"""Compare a VL block handoff with an independent dense calculation from token IDs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import torch
import torch.nn.functional as F

from .composed_attention_row import acquire, dense


def source_tensor(directory, key, inventory):
    """Decode BF16(W) to binary64 with the source geometry after H(bytes)=H(W)."""
    data = bytearray((directory / key).read_bytes())
    entry = inventory[key]
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Changed source operand {key}")
    return torch.frombuffer(data, dtype=torch.bfloat16).reshape(entry["shape"])


def main():
    """Require max_i |VL_i(tokens)−Dense_i(tokens)| ≤ 1e−10 for the supplied handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vl64", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--case", nargs=3, action="append", required=True,
                        metavar=("COMMA_SEPARATED_IDS", "INPUT_F64", "OUTPUT_F64"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.vl64.resolve()
    sys.path[:0] = [str(root), str(root / "src")]
    from experiments.model_fabric.local_seed import Snapshot
    from experiments.model_fabric.page_files import PageFiles

    torch.set_num_threads(1)
    manifest = json.loads((args.directory / "manifest.json").read_bytes())
    with PageFiles(manifest["page"]) as files:
        snapshot = Snapshot(files)
        cfg = snapshot.config["text_config"]
        for key, entry in manifest["operands"].items():
            name = f"model.language_model.layers.0.{key}.weight"
            shard, header, _, lo, hi, _ = snapshot.locations[name]
            if (shard, header+lo, hi-lo) != (entry["shard"], entry["start"], entry["bytes"]):
                raise ValueError(f"Operand address mismatch for {key}")
        raw, _, attention_bytes, _ = acquire(snapshot, None)
    hidden = cfg["hidden_size"]
    ops = manifest["operands"]
    # W_o ∈ ℝ^(hidden × (heads·head_dim)).
    dtype, weights = raw["out"]
    projection = torch.frombuffer(bytearray(weights), dtype=torch.bfloat16).double().reshape(hidden, -1)
    attention = dict(raw, out=(dtype, weights[:2*projection.shape[1]]))
    records = []
    for ids, input_path, output_path in args.case:
        tokens = list(map(int, ids.split(",")))
        _, heads = dense(cfg, attention, tokens)
        embedding = torch.frombuffer(bytearray(raw[f"token{tokens[-1]}"][1]), dtype=torch.bfloat16).double()
        r = embedding + projection @ heads.flatten()
        started = perf_counter()
        gain = 1 + source_tensor(args.directory, "post_attention_layernorm", ops).double()
        normalized = F.rms_norm(r, (hidden,), gain, cfg["rms_norm_eps"])
        gate = source_tensor(args.directory, "mlp.gate_proj", ops).double() @ normalized
        up = source_tensor(args.directory, "mlp.up_proj", ops).double() @ normalized
        expected = r + source_tensor(args.directory, "mlp.down_proj", ops).double() @ (F.silu(gate)*up)
        entry = dict(tokens=tokens, dense_mlp_and_io_seconds=perf_counter()-started)
        for label, path, reference in (("input", input_path, r), ("output", output_path, expected)):
            data = bytearray(Path(path).read_bytes())
            if len(data) != 8*hidden:
                raise ValueError("Unexpected handoff width")
            candidate = torch.frombuffer(data, dtype=torch.float64)
            error = (candidate-reference).abs().max().item()
            entry[label+"_max_abs_error"] = error if math.isfinite(error) else "nonfinite"
            entry[label+"_sha256"] = hashlib.sha256(data).hexdigest()
        records.append(entry)
    passed = all(isinstance(e[k], float) and e[k] <= 1e-10 for e in records
                 for k in ("input_max_abs_error", "output_max_abs_error"))
    report = dict(**manifest, attention_reference_bytes=attention_bytes,
                  cases=records, tolerance=1e-10, passed=passed)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
