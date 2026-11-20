# QUALIFIED-010 — Original trained Qwen historical attention, independently checked

The independent original-weight QUALIFIED-009 PyTorch result was
exceptionally close: maximum F64 error 1.78 × 10^-15 on 5,120 actual
layer-7 output coordinates after seven original trained predecessor
layers. However, it was measured at **causal position zero**. A
one-position softmax has exactly one key/value and must assign it
weight 1. Independent full-layer parity there does **not** test
meaningful trained query/key selection between competing causal states.

This experiment is explicitly designed to falsify that weaker inference.

## The question

Can two independently implemented whole-layer mathematical systems
take the **actual Qwen3.8-27B source-trained neural states of two
different paragraph tokens** and agree at layer 7 *after the second
token encounters a nontrivial history of learned keys and values*?

The source is the original pinned
`Qwen/Qwen3.8-27B` revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
The same source-admitted original architecture plan and tokenizer
determine text-token IDs `63204` and `628`.

## Train-derived causal path

Both tokens pass independently through the seven *complete* original
trained layers 0–6. Those layers retain their GDN/convolution states and
earlier attention KV states across the two positions. Their original
math is unchanged from the source-indexed QUALIFIED-008 execution.

At the 7→8 handoff (zero-based layer 7), each 5,120-wide hidden
vector is supplied to two whole-layer executors:

- VL-I's `QwenQualifiedGatedAttention`, using the existing qualified
  original source range and F64 NumPy/scalar mathematics;
- `OriginalQwenTorchGQA`, using independent CPU PyTorch F64
  source decoding, projections, norm, RoPE, grouped causal softmax,
  output sigmoid gate, original FFN and residual math.

Each retains **its own** layer-7 causal KV history. Neither may reuse
the other's computed attention value.

The first token must reproduce all historical exact F64 layer
digests from QUALIFIED-008. The second token must reproduce all
four original first-hybrid-block output digests from QUALIFIED-007,
whose two-token original-weight experiment preceded this work.

## Evidence that tests the actual missing mechanism

The independent reference reports all query-head softmax probabilities
for all admitted historical KV positions, as well as:

- The maximum deviation of any learned attention probability from
  equal uniform weighting across available history.
- The maximum absolute difference between attention's weighted
  historical V context and simply selecting the *current* V vector.

At the first token, both distinctions must be exactly zero, because
there is only one KV position. At the second, the example must show
a nonzero learned distribution difference **and** a nonzero historical
value contribution. Otherwise it cannot publish the strong claimed
temporal parity status, even if final output vectors numerically match.

At both original trained positions, the independent PyTorch and VL-I
whole-layer outputs must agree on all 5,120 coordinates within
`max_abs <= 1e-5` and
`max_scaled <= 2e-7`. The source-range identity, exact input/output
digests, true original tokens, query-head count, KV-head count, and
network traffic are retained in the evidence.

## Run and evidence status

`tests/test_qwen_two_position_temporal.py` is an independent causal
fixture probe. The original pretrained run itself is explicitly
separate and source-expensive:

```bash
PYTHONPATH=upstream/vl64:upstream/vl64/tests:tests \
  python -m pytest -q tests/test_qwen_two_position_temporal.py

PYTHONPATH=upstream/vl64:upstream/vl64/tests:tests \
  python -m examples.qwen_two_position_torch_parity \
    --block-bytes 8388608 \
    --host-budget-bytes 134217728 \
    --layer-budget-bytes 2000000000 \
    --out artifacts/qwen-original-two-token-temporal-parity.json
```

**This original checkpoint experiment must not be claimed until a live
original-trained job produces a successful exact-source receipt.**
Synthetic fixture agreement alone is insufficient.

Even a successful two-position original-weight F64 mathematical
reference is **not** a full original 64-layer decoder, Hugging Face
BF16 runtime parity, generated language, hardware independence, or
network-native floating-point arithmetic. VL-64 is untouched.

## Next research step

If the real trained temporal comparison passes, investigate
independent BF16 kernel-level parity and trained attention effects
over longer genuine paragraphs. More ambitious: learn which causal
history distinctions are actually necessary to realize a requested
output, then test a compressed Ground/quotient against the same
independent pretrained numerical authority.
