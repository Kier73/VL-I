"""Resolve the four trained operands for Qwen's first SwiGLU sublayer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def form(snapshot, directory):
    """Bind each W ∈ BF16^(m×n) to the ordered concatenation of its source intervals."""
    cfg = snapshot.config["text_config"]
    hidden, intermediate = cfg["hidden_size"], cfg["intermediate_size"]
    if cfg["model_type"] != "qwen3_5_text" or cfg["hidden_act"] != "silu":
        raise ValueError("Expected Qwen3.5 text RMSNorm and SiLU semantics")
    shapes = {"post_attention_layernorm": [hidden], "mlp.gate_proj": [intermediate, hidden],
              "mlp.up_proj": [intermediate, hidden], "mlp.down_proj": [hidden, intermediate]}
    descriptors = {t["name"]: t for t in snapshot.ground["tensors"]["tensors"]}
    inventory = {}
    for key, shape in shapes.items():
        name = "model.language_model.layers.0." + key + ".weight"
        if name.removesuffix(".weight") + ".bias" in descriptors:
            raise ValueError(f"Unexpected additive bias for {name}")
        tensor = descriptors[name]
        shard, header, _, lo, hi, size = snapshot.locations[name]
        if tensor["shape"] != shape or tensor["dtype"] != "BF16" or hi-lo != 2*math.prod(shape):
            raise ValueError(f"Unsupported trained operand {name}")

        def read(offset):
            """Read S[header+lo+i : header+lo+i+n] for n=min(chunk,hi-lo-i)."""
            length = min(4 << 20, hi-lo-offset)
            data, total = snapshot.files.read_range(shard, header+lo+offset, length)
            if total != size or len(data) != length:
                raise ValueError(f"Unexpected range for {name}")
            return data

        digest = hashlib.sha256()
        with (directory / key).open("wb") as out, ThreadPoolExecutor(max_workers=4) as pool:
            for data in pool.map(read, range(0, hi-lo, 4 << 20)):
                digest.update(data)
                out.write(data)
        inventory[key] = dict(shape=shape, sha256=digest.hexdigest(), shard=shard,
                              start=header+lo, bytes=hi-lo)
    return dict(hidden=hidden, intermediate=intermediate, epsilon=cfg["rms_norm_eps"],
                operands=inventory)


def main():
    """Publish one immutable directory only after ∀W, |bytes(W)|=2·prod(shape(W))."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vl64", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.vl64.resolve()
    sys.path[:0] = [str(root), str(root / "src")]
    from experiments.model_fabric.local_seed import Snapshot
    from experiments.model_fabric.page_files import PageFiles

    page = json.loads(args.manifest.read_bytes())["page"]
    args.directory.parent.mkdir(parents=True, exist_ok=True)
    if args.directory.exists():
        raise FileExistsError(args.directory)
    with TemporaryDirectory(dir=args.directory.parent) as tmp, PageFiles(page) as files:
        if page != f"https://huggingface.co/{files.model}/tree/{files.revision}":
            raise ValueError("Handoff requires an immutable model revision")
        staged = Path(tmp) / "operands"
        staged.mkdir()
        manifest = form(Snapshot(files), staged)
        manifest["page"] = page
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staged.rename(args.directory)


if __name__ == "__main__":
    main()
