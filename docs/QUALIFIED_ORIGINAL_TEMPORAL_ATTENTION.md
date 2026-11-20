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

## Verified original pretrained outcome

The exact original model's two-token source job
[44673275797](https://github.com/Kier73/VL-I/actions/runs/44673275797)
**passed**. The unabridged receipt is artifact
[14601034410](https://github.com/Kier73/VL-I/actions/runs/44673275797/artifacts/14601034410);
the compact source-traceable archive and reproducible regressions live at:

- `experiments/qwen_all_heads/datasets/2026-11-19-real-original-qwen-two-token-trained-attention-parity.json`
- `tests/test_saved_original_qwen_temporal_parity.py`

Original source: Qwen3.8-27B revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Both genuine paragraph
tokens `63204` and `628` were executed through all seven prior
original trained layers, retaining genuine independent recurrent
and earlier-attention histories.

Measured complete layer-7 (5,120-wide) numerical agreement:

| Original causal position | Available KV positions | Maximum absolute PyTorch F64 error | Historical attention evidence |
| --- | ---: | ---: | --- |
| First token `63204` | 1 | 7.105427357601002 × 10^-15 | Exactly one softmax entry: 1.0 |
| Second token `628` | 2 | 5.273559366969494 × 10^-16 | Nonuniform learned two-token attention |

At the second original trained position:

- There are **24 independently observed query-head distributions** with
  four shared KV heads.
- Previous-token attention probabilities range from
  **0.0027075803346912294** to **0.9539322042725595**.
- Maximum absolute deviation from uniform two-position attention
  is **0.4972924196653088**, clearly nontrivial trained query–key
  selection.
- Maximum historical value-context difference from the current-token-only
  value is **4.862222537731979**.
- The source host observed **1,734 HTTPS 206 responses** containing
  **14,537,908,768 bytes**, servicing **208,800 bounded 64 KiB logical
  range requests**. The external cache was limited to 128 MiB; VL-64
  retained no model weights.

Layer 0–7 exact F64 digests at the first token reproduced QUALIFIED-008;
layer 0–3 exact F64 digests at the second token reproduced QUALIFIED-007.
These controls exclude an arbitrary synthetic hidden state or silently
changed source from the accepted experiment.

### Research curiosity: a one-coordinate Ground for two-key attention

Two causal keys have only **one normalized attention degree of freedom**
per query head. If `s_prior` and `s_now` are the trained query–key
scores after their usual scaling, then:

```text
delta = s_prior - s_now
p_prior = 1 / (1 + exp(-delta))
p_now = 1 - p_prior
attended_V = V_now + p_prior * (V_prior - V_now)
```

This is an exact algebraic identity before ordinary floating-point
roundoff. The new `binary_attention_quotient.py` records just
**one log-odds coordinate per head** rather than two probabilities.
The source-attested, originally observed 24 probabilities are used
in `tests/test_binary_attention_quotient.py` to verify reconstruction
with sub-picounit errors and refusal of altered source observation pins.

This **does not** eliminate trained Q/K dot products, values, or the
original complete model architecture. What it does establish is an
interpretable, one-dimensional structural coordinate for the exact
two-position attention relation. The conditional V realization can be
reconstructed from this coordinate and its two value endpoints.

The practical research direction is to discover reusable, witnessed
quotient coordinates for longer genuine causal histories, and measure
**actual saved work** versus the unmodified trained model. A structural
description alone is not yet a demonstrated computational speedup.
