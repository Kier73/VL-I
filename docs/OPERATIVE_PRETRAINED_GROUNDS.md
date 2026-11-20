# INTERACT-002 — trained relations as constructive, renderable Grounds

**Development target:** from a catalogue of admitted model tensor locations to
*actual interaction with trained computation* and the formation of a response
from that computation.

**Implementations:** [operative_model.py](../src/vli/operative_model.py),
[operative_trained_demo.py](../examples/operative_trained_demo.py).
**Independent controls:** [test_operative_model.py](../tests/test_operative_model.py).
**Reproducibility:** [vli-operative-model.yml](../.github/workflows/vli-operative-model.yml).

## An intention causes trained work

A Ground is not its list of tensors. The interpreter starts with an immutable
large-model source identity and a qualified typed operator, then an observer
selects an **input procedure** (sin/cos, amplitude, frequency, phase) and the
requested output views. The resulting coordinate is allocated *before* any
weight tensor is read or any result is computed.

```text
Original-size pretrained tensor source + verified index
   ↓
Admitted operative Ground / real typed operator, not model API delegation
   ↓
Observer intention + causal head + procedural hidden-state input recipe
   ↓
Future state coordinate (no output value or tensor bytes stored here)
   ↓
Qualified VL-64 range reads (≤64 KiB each), original weight scales/geometry
   ↓
Native trained operator + independent numerical expression
   ↓
Construct from resulting numeric state → JSON, text, page, Python, byte tokens
   ↓
VLD response DAG + E-CPU byte Ground for requested rendered views
   ↓
Hash-bound receipt and recoverable causal interpreter history
```

**New input is not fabricated pretrained data.** The analytic vector is an
observer-selected synthetic hidden input, not a natural-language prompt's
tokenized embedding. It lets us test whether changing an intention actually
changes the model's trained numerical response, rather than selecting a
different entry from a catalogue. The full underlying model scale remains
exactly as observed.

## Two original-scale executable relations

| Model | Native pretrained operation | What it creates | Still missing |
| --- | --- | --- | --- |
| Qwen/Qwen3.8-27B | Real BF16 input norm and 48 rows each of the learned `in_proj_a`, `in_proj_b`; trained `A_log`, `dt_bias`; `qwen_gates` produces all 48 beta/decay gates | Numeric gate vector, mean gate, count above 0.5, verified output views and generated pure recomposition program | Q/K/V projections, conv and complete recurrent head/layer, tokenization, logits and full language output |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | Real trained 384-way router; choose **six** experts by trained scores + bias; request only their native MXFP4 gate/up/down first-row contributions and E8M0 scales | Six selected expert identities and partial weighted numerical contribution, independently decoded FP4 references, rendered output views and generated pure recomposition program | Remaining expert matrix rows, shared expert, complete MoE, attention/Engram/mHC, tokenization and logits |

The Qwen source is revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`; DeepSeek uses
`1bae0e1088993b9b82de08f59de5be5a34e42170`.
Sources, matrix dimensions and indexes are admitted through immutable
VL-64 evidence, pinned at
[`fea0d29`](https://github.com/Kier73/VL-64/commit/fea0d29d5a8a1edb03d9c66edd5ca4b15a1395d3).
Neither checkpoint is reduced, substituted or quantized by this experiment.
The DeepSeek FP4 encoding and trained scales are preserved.

## Construction, not just exposition

`TrainedOperationSession.operate()` performs the original trained numeric
relation before it builds outputs. It reports measured trained weight bytes,
qualified range count, independent activation/decoder reference error and
an execution-chain digest. `_compose` computes a downstream result from
the trained result, such as how many learned gates exceed a threshold or
a selected-expert weighted contribution.

The generated Python is a pure function of the *observed numeric result*.
It reproduces the derived result in a **separate Python process**. It does
not secretly contain an entire pretrained checkpoint or claim to generate
full next-token logits. The existing E-CPU/VLD response Ground can then
render the actual newly constructed result as Markdown, text, code and
UTF-8 byte tokens. Here tokens are output byte tokens, **not the model's BPE
tokenization**.

This is a narrower but real bridge between **observation → trained
computation → constructed artifact**, not simply metadata → artifact.

## Causal continuation and limits of evidence

The opaque coordinate is bound to the source Ground, input recipe and
previous causal head, before the trained operator runs. The interpreter
commits a transcript event only after numerical verification and requested
rendering succeed. A subsequent new intention may choose a different input
and therefore produce a different result while retaining the causal
history. A separate process can restore the checkpoint with an externally
pinned SHA-256 digest without downloading the original checkpoint again.

Restore verifies the event and intention chain; it does **not** independently
recompute the old operator outputs without re-reading the corresponding
weights. A hash relative to a trusted external head is not a signature
of the remote model host. Selected-source receipts are not proof of a
complete decoder or general semantic natural-language ability.

## Verification and live execution

The offline suite tests the actual constructive architecture through mocked
qualified range functions (these are **fixtures**, not claimed pretrained
inference). It exercises changes of input, all 48 Qwen gates, 384/6 DeepSeek
routing, selected-only expert arithmetic, independent generated code,
recovery, wrong revisions, nonexistent trained weights, altered intentions,
and VLD rendering against original E-CPU byte Grounds.

```bash
# Tests require the pinned VL-64/E-CPU/VLD predecessor source environments
PYTHONPATH=upstream/vl64 python -m unittest discover -s tests -p test_operative_model.py -v

# Real pretrained experiments: large checkpoints stay at the pinned HTTP host.
# Only required qualified operator ranges are streamed, never the whole model.
PYTHONPATH=upstream/vl64 python -m examples.operative_trained_demo \
    --model qwen-27b --out artifacts/qwen-operative.json
PYTHONPATH=upstream/vl64 python -m examples.operative_trained_demo \
    --model deepseek-v41-flash --out artifacts/deepseek-operative.json
```

Live GitHub Actions uses real Hugging Face range reads and retains a
machine-readable result (or explicit **UNRESOLVED**). Both routes preserve the
original model revision; no provider authentication token is needed for
public pinned weights.

## What should be promoted back to VL-64?

Not a second tensor framework and not the large research code. The
potential minimal interpreter primitive is:

`(qualified Ground, causal head, operation, intention) → next state + receipt`

Its only essential additional contracts are typed input extent, qualified
operator provenance, position-before-realization, output view selection, and
atomic receipt publication. These belong in VL-64 **only after** experiments
show which parts are general, small and actually improve inference or
recoverability. The existing 64 KiB core remains untouched.

The next frontier is replacing observer-generated analytic hidden inputs with
the **actual tokenizer/prefill hidden state of an unmodified original model**,
then navigating every required layer until complete pretrained logits are
realized. Only then can this pathway be evaluated for paragraphs, code and
autonomous action at the full original model scale.
