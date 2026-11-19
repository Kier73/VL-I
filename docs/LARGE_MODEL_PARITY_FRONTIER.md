# VL-I B1 ↔ B2/B3: same-intention full-scale qualification

**2026-11-19.** This is an executable comparison protocol, not a claim that
VL-native paragraphs or code already run through the complete original Qwen
or DeepSeek checkpoints. It joins the measured control-plane model fleet in
[VL-64](https://github.com/Kier73/VL-64) to the native mathematical Grounds
developed here and in VL-64. **No model is replaced, shrunk, or re-trained.**

## The two original model sources

| Original model | Pinned *observed* weight revision | Advertised source payload | Structure |
| --- | --- | ---: | --- |
| Qwen/Qwen3.8-27B | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | 55,562,855,904 bytes | 64 decoder layers; 48 gated-delta/linear attention, 16 full attention |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | `1bae0e1088993b9b82de08f59de5be5a34e42170` | 510,286,023,000 bytes | 40-layer hybrid model; compressed sparse attention, routed MoE, Engram and mHC |

Both source/index digests, observed tensor counts, pretrained-weight range witnesses
and *actual* independently compared numerical **partial subgraphs** are imported
from immutable VL-64 commit
[`fea0d29`](https://github.com/Kier73/VL-64/commit/fea0d29d5a8a1edb03d9c66edd5ca4b15a1395d3).
Git commit and hash checks are integrity/provenance relative to these pinned
sources, **not independent signatures of every model byte**.

## What the implementation actually checks

`src/vli/large_model_comparison.py` admits three distinct sources:

1. **B1**: VL-64 cross-process inference-control receipts, with the actual
   paragraph, Python code and approved `add_numbers` agent intentions. For
   offline fixtures, there are **zero trained-model invocations**. Live B1
   requires real provider credentials and returns only chat content, reported
   usage, wall time and controller CPU: no raw logits, model KV or attestable
   exact served model revision.
2. **B2 partial numerical evidence**: Qwen's trained layer-0 recurrent head
   and partial output projection, and DeepSeek's 384-way trained router with
   six selected FP4 expert row fragments. These came from genuine pinned
   pretrained bytes and have archived independent-reference error claims;
   VL-I checks the receipts structurally, **not by re-executing the missing
   weight matrix from those receipts**. The test explicitly calls this
   `REAL_PRETRAINED_PARTIAL_SUBGRAPHS_ONLY`.
3. **B3 partial observation navigation**: the admitted Qwen/DeepSeek trained
   fragments were reconstructed offline from verified retained observations
   without extra remote reads. This is *reuse of already observed partial
   calculations*, not prediction of unobserved language responses.

The model-runner in this comparison performs **no neural arithmetic** and
makes **no provider calls**. It discovers the required *proof/observation
frontier* before a claimed same-intention comparison is eligible.

The comparison binds the same B1 intent digest, task prompt, exact model
route, and original large-scale pinned source. It explicitly marks these
missing obligations: pinned tokenizer/template parity, prompt-dependent
embeddings, complete attention/FFN layers, causal state, full logits, multi-token
decoding, and verified task outputs. It does **not** convert one model's
synthetic-hidden partial subgraph into an answer to a different natural-language
prompt.

## Why numerical parity is currently unresolved

A B1 provider chat response does not independently establish which exact
revision the provider served. Even a capable provider returning excellent
paragraphs, Python, and approved tool decisions would not by itself verify
the B2 path. A real B2 comparison requires the *same original model snapshot*
and tokenizer, the *same prompt token IDs and decoding policy*, and independently
verified complete logits or selected tokens. Different numerical precisions and
rounding contracts must be compared within a declared tolerance.

Likewise, interpreter CPU milliseconds **cannot** be compared to partial B2
trained-weight byte count as if those measured the same operation. The report
keeps both measures but leaves speedup and full-model parity unresolved.

## Run on a CPU-only observer

```bash
# Check out the immutable upstream model evidence; Git enforces the exact SHA.
git clone https://github.com/Kier73/VL-64 upstream/vl64
git -C upstream/vl64 checkout fea0d29d5a8a1edb03d9c66edd5ca4b15a1395d3
python -m pip install -e . -e upstream/vl64 pytest
python -m pytest -q tests/test_large_model_comparison.py
python -m experiments.model_fabric.remote_large_recovery_benchmark \
  --fixture --max-tokens 64 --working-dir artifacts/control-epochs \
  --out artifacts/b1-control.json
python -m vli.large_model_comparison \
  --vl64-root upstream/vl64 --b1-report artifacts/b1-control.json \
  --out artifacts/b1-b2-comparison.json
```

GitHub Actions performs the same experiment on Python 3.11 and 3.13, retains
the JSON report, and tests adversarial mutations: wrong revision/index, fake
full Qwen forward, missing DeepSeek FP4 slices, altered reference error,
non-zero replay reads and a fixture masquerading as live inference.

## Next hypothesis-driven B2 work

The first useful promotion is **one complete and independently checked
pretrained layer** on its real checkpoint, with declared tokenizer, hidden
input and numerical tolerance. Then acquire the complete model's causal
state and unscaled decoder. Make each newly admitted transition available
to VL-I's Ground navigator and to VL-64's minimal interpreter ABI; avoid
adding competing executable authorities or copying 510 GB of weights into
the interpreter.

For the strict same-intention comparison, run real B1 only when the provider
is available and subsequently prove that B2 observes the identical model
snapshot and prompt contract. If revision equivalence cannot be established,
that is an explicit scientific result, not a reason to fabricate a match.

The goal remains the larger one: discover qualified navigable relations that
replace conventional work on B3, while retaining full model capability.
