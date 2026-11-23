"""Continue a lifted Qwen layer-zero attention residual through its trained SwiGLU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from time import perf_counter

from vli.grouped_projection import residual


def project(directory, key, operand, vector, bias):
    """Compute y=b+Wx in row tiles and require H(tiles in row order)=H(W)."""
    rows, cols = operand["shape"]
    if (rows, cols) != (len(bias), len(vector)):
        raise ValueError(f"Dimension mismatch in {key}")
    tile_rows = max(1, (1 << 20) // (2*cols))
    output = []
    products = 0
    digest = hashlib.sha256()
    with (directory / key).open("rb") as stream:
        for start in range(0, rows, tile_rows):
            offsets = bias[start:start+tile_rows]
            tile = stream.read(2*cols*len(offsets))
            digest.update(tile)
            part, count = residual(tile, vector, offsets)
            output.extend(part)
            products += count
        if stream.read(1) or digest.hexdigest() != operand["sha256"]:
            raise ValueError(f"Source identity mismatch in {key}")
    return output, products


def silu(x):
    """SiLU(x)=x·σ(x), with exp evaluated on a nonpositive argument."""
    if x >= 0:
        return x / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return x * exp_x / (1 + exp_x)


def advance(directory, manifest, state, projection=project):
    """r' = r + D[SiLU(Gn) ⊙ Un], n = (1+w) ⊙ r / sqrt(mean(r²)+ε)."""
    ops = manifest["operands"]
    hidden = manifest["hidden"]
    intermediate = manifest["intermediate"]
    norm = (directory / "post_attention_layernorm").read_bytes()
    if len(norm) != 2*hidden or hashlib.sha256(norm).hexdigest() != ops["post_attention_layernorm"]["sha256"]:
        raise ValueError("Normalization weight identity mismatch")
    scale = 1 / math.sqrt(math.fsum(x*x for x in state)/hidden + manifest["epsilon"])
    gain = (struct.unpack("<f", bytes(2)+word)[0] for word, in struct.iter_unpack("2s", norm))
    normalized = [x*scale*(1+w) for x, w in zip(state, gain, strict=True)]
    zeros = [0.] * intermediate
    gate, ng = projection(directory, "mlp.gate_proj", ops["mlp.gate_proj"], normalized, zeros)
    up, nu = projection(directory, "mlp.up_proj", ops["mlp.up_proj"], normalized, zeros)
    mixed = [silu(g)*u for g, u in zip(gate, up, strict=True)]
    output, nd = projection(directory, "mlp.down_proj", ops["mlp.down_proj"], mixed, state)
    return output, ng+nu+nd


def main():
    """Bind the input and output by SHA256(pack_le64(r)) at the sublayer boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_bytes())
    encoded = args.input.read_bytes()
    state = struct.unpack(f"<{manifest['hidden']}d", encoded)
    started = perf_counter()
    output, products = advance(args.directory, manifest, state)
    elapsed = perf_counter() - started
    result = struct.pack(f"<{len(output)}d", *output)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(result)
    print(json.dumps(dict(input_sha256=hashlib.sha256(encoded).hexdigest(),
                          output_sha256=hashlib.sha256(result).hexdigest(), grouped_products=products,
                          dense_products=3*manifest["hidden"]*manifest["intermediate"],
                          projection_and_io_seconds=elapsed)))


if __name__ == "__main__":
    main()
