"""Compare a certified, partial trained readout with each checkpoint's complete native output head."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path


# Fixed before execution: checkpoint, revision, reference class, and input modality.
MODELS = [
    ("bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594", "BertForMaskedLM", "masked"),
    ("facebook/opt-125m", "1f9886ce095904096e22b0f4d9e7ba932fa7df2a", "OPTForCausalLM", "causal"),
    ("google/vit-base-patch16-224", "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3", "ViTForImageClassification", "vision"),
    ("gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "GPT2LMHeadModel", "causal"),
    ("HuggingFaceTB/SmolLM-135M", "1d461723eec654e65efdc40cf49301c89c0c92f4", "LlamaForCausalLM", "causal"),
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "Qwen3ForCausalLM", "causal"),
    ("HuggingFaceTB/SmolVLM-256M-Instruct", "7e3e67edbbed1bf9888184d9df282b700a323964", "Idefics3ForConditionalGeneration", "text-only"),
    ("state-spaces/mamba-130m-hf", "1e76775f628fbf1350fbe4dbb3d971ba64af25a1", "MambaForCausalLM", "causal"),
    ("google/flan-t5-small", "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab", "T5ForConditionalGeneration", "encoder-decoder"),
    ("RWKV/rwkv-4-169m-pile", "54646c56ad6b5d4283918ef47561e01b777f7575", "RwkvForCausalLM", "causal"),
]


def integerize(numbers):
    """Finite binary floats are dyadic: x_i=X_i/d with d=max_i denominator(x_i)."""
    ratios = [x.as_integer_ratio() for x in numbers]
    d = max(q for _, q in ratios)
    return [p * (d // q) for p, q in ratios], d


def compile_rows(left, right, bias_left, bias_right):
    """Compile a=W_i−W_j with bias as a final coefficient, z'=(z,1), ordered by |a_k|."""
    if len(left) != len(right) or not left:
        raise ValueError("trained row widths must agree and be nonzero")
    w, d = integerize([*left, bias_left, *right, bias_right])
    width = len(left) + 1
    a = [w[k] - w[k+width] for k in range(width)]
    order = sorted(range(width), key=lambda k: -abs(a[k]))
    return a, order, sum(v*v for v in a), d


def navigate(compiled, state):
    """If S²>(Σ_remaining A²)(Σ_remaining Z²), Cauchy–Schwarz certifies sign(S+tail)=sign(S)."""
    a, order, energy_a, _ = compiled
    z, _ = integerize([*state, 1.0])
    if len(z) != len(a):
        raise ValueError("state and trained direction dimensions differ")
    energy_z = sum(v*v for v in z)
    score = 0
    for count, k in enumerate(order, start=1):
        score += a[k]*z[k]
        energy_z -= z[k]*z[k]
        energy_a -= a[k]*a[k]
        if score*score > energy_z*energy_a:
            break
    return (score > 0) - (score < 0), count


def load_reference(transformers, torch, folder, class_name):
    """Instantiate f_W with all required trained tensors; tied OPT stores W_input=W_head once."""
    cls = getattr(transformers, class_name)
    if class_name == "OPTForCausalLM":
        from safetensors.torch import load_file
        config = transformers.AutoConfig.from_pretrained(folder, local_files_only=True)
        weights = load_file(str(folder / "model.safetensors"))
        key = "model.decoder.embed_tokens.weight"
        if config.tie_word_embeddings and key not in weights:
            weights[key] = weights["lm_head.weight"]
        model = cls(config)
        model.load_state_dict(weights, strict=True)
        return model.eval()
    model, info = cls.from_pretrained(folder, use_safetensors=True, local_files_only=True,
                                      torch_dtype=torch.float32, output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Incomplete pretrained reference parameters")
    return model.cpu().eval()


def run_one(spec):
    """For each observed native affine input z, compare navigate(a,z) to sign(native_i−native_j)."""
    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
    import torch
    import transformers
    from huggingface_hub.constants import HF_HUB_CACHE
    name, revision, class_name, mode = spec
    row = dict(model=name, revision=revision, reference_class=class_name, mode=mode)
    try:
        torch.set_num_threads(1)
        folder = Path(HF_HUB_CACHE) / ("models--" + name.replace("/", "--")) / "snapshots" / revision
        for fname in ("config.json", "model.safetensors"):
            with (folder / fname).open("rb") as f:
                row[fname + "_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        reference = load_reference(transformers, torch, folder, class_name)
        head = reference.classifier if mode == "vision" else reference.get_output_embeddings()
        if not isinstance(head, torch.nn.Linear):
            raise ValueError("output module is not an affine head")
        i, j = (0, 1) if mode == "vision" else (42, 75)
        bias = head.bias
        compiled = compile_rows(head.weight[i].tolist(), head.weight[j].tolist(),
                                0.0 if bias is None else bias[i].item(), 0.0 if bias is None else bias[j].item())
        decisions = []

        class ReadoutComplete(Exception):
            """The cut z→sign(a·z') has answered the intention."""

        def stop_at_head(module, args):
            """Compute sign(a·z') at the actual input edge and terminate before Wz+b."""
            decisions.extend(navigate(compiled, state)
                             for state in args[0].detach().flatten(end_dim=-2).tolist())
            raise ReadoutComplete

        ids = torch.tensor([[42, 75, 123, 17, 256, 113], [113, 256, 17, 123, 75, 42]])
        inputs = {"input_ids": ids}
        if mode == "vision":
            n = reference.config.image_size
            image = torch.linspace(-1, 1, 3*n*n).reshape(3, n, n)
            inputs = {"pixel_values": torch.stack([image, image.flip([1])])}
        elif mode == "encoder-decoder":
            inputs["decoder_input_ids"] = ids.flip([1])
        elif mode == "masked":
            ids[:, 2] = 103
        with torch.inference_mode():
            with head.register_forward_pre_hook(stop_at_head):
                try:
                    reference(**inputs)
                except ReadoutComplete:
                    pass
                else:
                    raise ValueError("Candidate missed the output-head boundary")
            logits = reference(**inputs).logits.flatten(end_dim=-2)
        if len(decisions) != len(logits) or not torch.isfinite(logits).all():
            raise ValueError("reference head invocation differs from the admitted contract")
        trials = []
        for (sign, terms), values in zip(decisions, logits):
            gap = values[i].item() - values[j].item()
            trials.append(dict(sign=sign, native_gap=gap, terms=terms, matches=sign == ((gap>0)-(gap<0))))
        row.update(status="agreed" if all(t["matches"] for t in trials) else "disagreement",
                   pair=[i,j], width_including_bias=len(compiled[0]), trials=trials,
                   torch_version=torch.__version__, transformers_version=transformers.__version__)
    except Exception as error:
        row.update(status="unresolved", reason=f"{type(error).__name__}: {error}")
    return row


def main():
    """Evaluate the fixed panel concurrently and retain every model in the result denominator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(1,2,3), default=2)
    parser.add_argument("--acquire", action="store_true", help="Fetch pinned config and Safetensors before testing")
    args = parser.parse_args()
    if args.acquire:
        from huggingface_hub import hf_hub_download
        for model, revision, _, _ in MODELS:
            for filename in ("config.json", "model.safetensors"):
                try:
                    hf_hub_download(model, filename, revision=revision)
                except OSError as error:
                    print(f"{model}/{filename}: {error}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_one, MODELS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print([(r["model"], r["status"]) for r in rows])
    return int(any(r["status"] != "agreed" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
