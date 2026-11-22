"""Compare two trained readouts at every position of fixed SmolLM-135M prompts.

For h=u+Dv, RMSNorm(h)=g⊙h/r, r=sqrt(mean(h²)+ε)>0. With tied, bias-free
readout rows, sign(logit_yes−logit_no)=sign(a·h)=sign(a·u+b·v), where
a=(w_yes−w_no)⊙g and b=Dᵀa. These identities are over the reals. This
experiment compares FP64 evaluations to CPU FP32 logits, including ties.

Run with a local snapshot directory and an output path:
    python -m examples.smollm_pairwise_readout --snapshot SNAPSHOT --out RESULT.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import numpy as np
import safetensors
from safetensors import safe_open
import tokenizers
import torch
import transformers
from transformers import AutoModelForCausalLM

MODEL = "HuggingFaceTB/SmolLM-135M"
REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
HASHES = {
    "model.safetensors": "c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093",
    "config.json": "a1fe6f43e20f7a6c6dbc6380222af9526b5cef262446391a281c038249e3e3b7",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
}
YES, NO = 9805, 787
VOCAB, WIDTH, INTERMEDIATE = 49152, 576, 1536
TEXTS = (
    "The sky is blue on a clear day. If you agree, answer yes or no.",
    "Can a square circle exist in Euclidean geometry? The answer is",
    "She shook her head and said no. When asked again, she said",
    "He nodded and said yes. When asked again, he said",
    "There are no tickets left for the concert, and the box office is closed.",
    "We keep a notebook of each question and write down the answer.",
)


def compile_readout(snapshot):
    """After SHA256(file)=pin, compile a=(w_y−w_n)⊙g and b=Dᵀa from F32 slices."""
    for name, expected in HASHES.items():
        with (snapshot / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch for {name}")
    with safe_open(snapshot / "model.safetensors", framework="numpy") as file:
        rows = file.get_slice("model.embed_tokens.weight")
        gain = file.get_slice("model.norm.weight")
        down = file.get_slice("model.layers.29.mlp.down_proj.weight")
        for view, shape in ((rows, [VOCAB, WIDTH]), (gain, [WIDTH]),
                            (down, [WIDTH, INTERMEDIATE])):
            if view.get_shape() != shape or view.get_dtype() != "F32":
                raise ValueError(f"Expected F32 tensor with shape {shape}")
        a = (rows[YES].astype(np.float64) - rows[NO].astype(np.float64)) * gain[:]
        b = down[:].astype(np.float64).T @ a
    if not all(np.isfinite(x).all() for x in (a, b)):
        raise ValueError("Nonfinite compiled direction")
    return a, b


class TailBoundary(Exception):
    """Stop at v=SiLU(W_gate n)⊙(W_up n), immediately before h=u+Dv."""


def capture_input(destination, *, stop=False):
    """Capture X=arg₀(module); stop=True cuts the DAG at this input edge."""
    def hook(module, args):
        """For X∈R^(1×T×d), retain X[0,:,:] in FP64 without changing its values."""
        (x,) = args
        if x.ndim != 3 or x.shape[0] != 1 or x.dtype != torch.float32:
            raise ValueError("Unexpected activation shape or dtype")
        destination.append(x[0].detach().to(torch.float64).numpy())
        if stop:
            raise TailBoundary
    return hook


def read_text(model, ids, a, b):
    """Compare sign(Ua+Vb) from an interrupted prefix and sign(Ha) against L_y−L_n."""
    tokens = torch.tensor([ids], dtype=torch.long)
    last = model.model.layers[-1]
    u, v, h = [], [], []
    with ExitStack() as hooks:
        hooks.enter_context(last.post_attention_layernorm.register_forward_pre_hook(capture_input(u)))
        hooks.enter_context(last.mlp.down_proj.register_forward_pre_hook(capture_input(v, stop=True)))
        try:
            model(input_ids=tokens, use_cache=False)
        except TailBoundary:
            pass
        else:
            raise RuntimeError("Candidate reached the vocabulary head without hitting the cut")
    if len(u) != 1 or len(v) != 1:
        raise RuntimeError("Expected exactly one residual and one down-projection input")
    # Candidate computation completes before the independent unrestricted head runs.
    early = u[0] @ a + v[0] @ b
    with model.model.norm.register_forward_pre_hook(capture_input(h)):
        logits = model(input_ids=tokens, use_cache=False).logits[0]
    if len(h) != 1 or logits.shape != (len(ids), VOCAB) or logits.dtype != torch.float32:
        raise RuntimeError("Unexpected reference output")
    late = h[0] @ a
    # Widen individual FP32 oracle logits before forming their difference.
    margin = (logits[:, YES].double() - logits[:, NO].double()).numpy()
    if not all(np.isfinite(x).all() for x in (early, late, margin)) or not torch.isfinite(logits).all():
        raise ValueError("Nonfinite observation")
    return {
        "input_token_ids": ids,
        "early_unnormalized_score": early.tolist(),
        "final_hidden_unnormalized_score": late.tolist(),
        "early_sign": np.sign(early).astype(int).tolist(),
        "final_hidden_sign": np.sign(late).astype(int).tolist(),
        "oracle_yes_minus_no": margin.tolist(),
        "unrestricted_argmax_token_ids": logits.argmax(dim=1).tolist(),
    }


def summarize(records):
    """For positions i, tally signs, |{i:s_i≠r_i}|, min_{Δ_i≠0}|Δ_i| and argmax outside the pair."""
    margin = np.concatenate([r["oracle_yes_minus_no"] for r in records])
    reference = np.sign(margin).astype(int)
    summary = {"positions": len(margin)}
    for name in ("early_sign", "final_hidden_sign", "oracle_sign"):
        values = reference if name == "oracle_sign" else np.concatenate([r[name] for r in records])
        counts = Counter(values)
        summary[name] = {"yes": counts[1], "no": counts[-1], "tie_abstain": counts[0]}
        if name != "oracle_sign":
            summary[name]["disagreements_including_ties"] = int(np.count_nonzero(values != reference))
            summary[name]["opposite_nonzero_signs"] = int(np.count_nonzero(values * reference == -1))
    nonzero = np.abs(margin[margin != 0])
    summary["minimum_nonzero_absolute_oracle_margin"] = float(nonzero.min()) if len(nonzero) else None
    # In real arithmetic, (Ua+Vb)−Ha=0. The FP32 residual and FP64 pullback can round differently.
    early = np.concatenate([r["early_unnormalized_score"] for r in records])
    late = np.concatenate([r["final_hidden_unnormalized_score"] for r in records])
    summary["maximum_absolute_early_minus_final_hidden_score"] = float(np.max(np.abs(early - late)))
    summary["unrestricted_argmax_outside_pair"] = sum(
        token not in (YES, NO) for r in records for token in r["unrestricted_argmax_token_ids"])
    return summary


def synthetic_head_controls(model):
    """Test exact tie a₀a₁−a₁a₀=0 and near-tie a₀(a₁₇+2⁻³¹)−a₁₇a₀=a₀2⁻³¹>0."""
    cases = (
        ("exact_tie", (0, 1), (Fraction(145691, 1048576), Fraction(-26455, 65536)), Fraction(0)),
        ("positive_near_tie", (0, 17), (Fraction(14884865, 2147483648), Fraction(-26455, 65536)),
         Fraction(26455, 140737488355328)),
    )
    w = model.lm_head.weight
    g = model.model.norm.weight
    results = []
    for name, indices, values, predicted in cases:
        # FP32 scalars embed exactly in the rationals; form aᵢ=(w_yi−w_ni)gᵢ without rounding.
        coefficients = [(Fraction(w[YES, i].item()) - Fraction(w[NO, i].item())) * Fraction(g[i].item())
                        for i in indices]
        exact = sum(ai * hi for ai, hi in zip(coefficients, values, strict=True))
        if exact != predicted:
            raise ValueError(f"Pinned weights disagree with the rational construction for {name}")
        hidden = torch.zeros((1, 1, WIDTH), dtype=torch.float32)
        for i, x in zip(indices, values, strict=True):
            hidden[0, 0, i] = float(x)
            if Fraction(hidden[0, 0, i].item()) != x:
                raise ValueError("The constructed coordinate is not exactly representable in FP32")
        logits = model.lm_head(model.model.norm(hidden))[0, 0]
        yes, no = logits[YES].item(), logits[NO].item()
        results.append({
            "case": name, "scope": "Constructed final residual excluded from causal-position counts.",
            "shape": list(hidden.shape), "nonzero_indices": list(indices),
            "residual_values_exact": [str(x) for x in values], "all_other_residual_values": 0,
            "direction_at_indices_exact": [str(x) for x in coefficients],
            "exact_unnormalized_score": str(exact), "exact_sign": (exact > 0) - (exact < 0),
            "fp32_yes_logit": yes, "fp32_no_logit": no, "fp32_logits_difference_in_float64": yes - no,
            "fp32_sign": (yes > no) - (yes < no),
        })
    return results


def main():
    """Fix (model, texts, y, n) before evaluating both observers at every causal prefix x₁:ₜ."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    a, b = compile_readout(args.snapshot)
    tokenizer = tokenizers.Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
    for piece, token_id in ((" yes", YES), (" no", NO)):
        if tokenizer.encode(piece, add_special_tokens=False).ids != [token_id]:
            raise ValueError(f"Unexpected pinned tokenization of {piece!r}")
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation="eager").cpu().eval()
    down = model.model.layers[-1].mlp.down_proj
    if (model.config.model_type != "llama" or len(model.model.layers) != 30
            or model.config.rms_norm_eps <= 0 or model.lm_head.bias is not None or down.bias is not None
            or model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()):
        raise ValueError("Model does not implement the pinned tied, bias-free RMS readout")
    with torch.inference_mode():
        records = [{"text": text, **read_text(model, tokenizer.encode(text, add_special_tokens=False).ids, a, b)}
                   for text in TEXTS]
        controls = synthetic_head_controls(model)
    result = {
        "model": MODEL, "revision": REVISION, "file_sha256": HASHES,
        "source_file_bytes_hashed": {name: (args.snapshot / name).stat().st_size for name in HASHES},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "numpy": np.__version__,
                    "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__,
                    "device": "cpu", "attention": "eager", "oracle_dtype": "float32",
                    "readout_dtype": "float64", "torch_threads": torch.get_num_threads(),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "pair": {"yes": YES, "no": NO}, "special_tokens_added": False,
        "sign_encoding": {"1": " yes", "-1": " no", "0": "tie/abstain"},
        "cost_scope": {
            "logical_f32_bytes_two_rows_and_gain": 3 * WIDTH * 4,
            "logical_f32_bytes_last_down_projection": WIDTH * INTERMEDIATE * 4,
            "cold_direction_elementwise_subtractions_and_products_each": WIDTH,
            "cold_down_transpose_matvec_multiply_accumulates": WIDTH * INTERMEDIATE,
            "retained_fp64_direction_bytes": a.nbytes + b.nbytes,
            "early_warm_dot_product_lengths": [WIDTH, INTERMEDIATE],
            "late_warm_dot_product_lengths": [WIDTH],
            "direct_two_row_head_dot_product_lengths_after_rmsnorm": [WIDTH, WIDTH],
            "reference_down_and_full_head_matvec_multiply_accumulates_per_position":
                [WIDTH * INTERMEDIATE, WIDTH * VOCAB],
            "scope": "One full local model load, whole-file identity scans, and two CPU prefix runs per text. "
                     "Early pass executes 29 complete layers plus last attention, norms, gate and up projections. "
                     "It stops before last down projection, residual addition, final norm and vocabulary head. "
                     "The full oracle pass also supplies the final residual for the late observer. "
                     "Counts describe mathematical tail work. Physical I/O, peak memory and latency are unmeasured.",
        },
        "summary": summarize(records), "texts": records, "synthetic_head_controls": controls,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return int(any(result["summary"][key]["disagreements_including_ties"]
                   for key in ("early_sign", "final_hidden_sign")))


if __name__ == "__main__":
    raise SystemExit(main())
