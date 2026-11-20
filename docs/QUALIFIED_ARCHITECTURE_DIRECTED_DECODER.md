# QUALIFIED-008 — Source-indexed architecture navigation across hybrid blocks

**Hypothesis:** the model can provide its own operator ordering as an
admitted, verifiable description, instead of requiring a human-written
sequence of Gated DeltaNet and full-attention operations.

This work moves VL-I from `QwenQualifiedHybridBlock` (four hand-specified
runners) to `QwenArchitectureCausalPrefix` (N successive runners selected
from an immutable source-derived architecture). The separate
**VL-64 64 KiB interpreter remains unchanged**.

## Two independent source observations, one admitted plan

1. **Original `config.json`** is requested from
   `https://huggingface.co/Qwen/Qwen3.8-27B/resolve/<commit>/config.json`
   at the same immutable revision as the trained weight index. The
   interpreter verifies the config size, syntax (no duplicate keys,
   NaN or Infinity), text operator type, layer count, exact layer
   schedule, head dimensions, M-RoPE section geometry and normalization.
   It computes a SHA-256 fingerprint. An externally provided SHA-256
   pin, when available, is required to match exactly.
2. **Original `model.safetensors.index.json` and all referenced shard
   headers** are independently admitted by the pinned VL-64
   `QualifiedShardNavigator`. The adapter checks *all* declared
   layer coordinates, verifies exact required operator tensor names
   and rejects unexpected opposite-operator tensors. Importantly, it
   verifies **every required trained matrix and vector shape and dtype
   directly in the qualified source headers** before acknowledging
   the plan. A config cannot silently declare a mathematical operator
   that the source index cannot render.

The plan includes model/revision, weight-index Ground, exact config SHA,
and an explicit list of all 64 layer types and geometry, and is itself
digest-bound. Neither the config nor a tensor header alone proves that
the whole neural model has actually been executed.

## Architecture-directed causal path

```text
config.json (immutable revision) ───┐
                                    ├─ SHA-256 bound admitted indexed plan
weight index → qualified headers ────┘
                                           ↓
original token 63204 → embedding → GDN0 → GDN1 → GDN2 → GQA3
                                                 ↓ exact f64 predecessor output
                                  → GDN4 → GDN5 → GDN6 → GQA7
                                    ↓
                            exact 5120-coordinate H7[0]
```

No manually encoded `if layer % 4 == 3` is required in the
model-worker selector. Instead the approved layer kind at each
coordinate supplies the execution adapter:

- **GDN:** existing `QwenVectorizedCompleteLayer`, using the pinned
  VL-64 exact Gated DeltaNet + SwiGLU math and resident host vectorization.
- **Gated GQA:** existing `QwenQualifiedGatedAttention`, implementing
  trained Q/K/V, per-head norms, text M-RoPE, causal softmax, sigmoid
  output gating, residual and complete trained SwiGLU.

The per-layer exact F64 output/input digests must agree across
**all** predecessor edges. Every layer remains required to publish
its complete original hidden width and causal state digest.

## Multiple independent kinds of memory

For the eight-layer example there are **six independently recoverable
GDN convolution/recurrent states** and **two distinct causal KV
histories**. The KV histories are kept apart: attention layer 7 cannot
reuse layer 3's keys, even when both share identical numerical shapes.

A checkpoint publishes a separate content-addressed 64 KiB page
inventory per layer under a single plan- and source-bound
externally pinned manifest. A different model revision, config digest,
topology, layer count, input sequence, causal position, or page contents
is rejected. On an unsuccessful downstream token, all semantic
states are returned to the previously accepted causal position.
Network reads are audit observations and do not roll back.

## Falsification suites

`tests/test_qwen_architecture_navigator.py` constructs a complete
**two-hybrid-block, eight-layer original-topology fixture** with distinct
source matrices for all six GDN and two attention operators. It
independently reconstructs the expected result using the scalar
`QwenQualifiedLayer` for every GDN and a separate scalar
attention/FFN oracle, across three causal positions.

It also verifies:

- The first architecture-selected four-layer prefix agrees with
  independently executed `QwenQualifiedHybridBlock`.
- Every exact F64 interlayer handoff agrees.
- The two attention layers preserve independently advancing histories.
- Eight complete causal state receipts survive a fresh-executor
  checkpoint/restore, and subsequent output/head digests agree with
  continuous execution.
- Wrong config SHA, missing tensors, duplicated JSON keys, invalid
  geometry, wrong layer topology, corrupted page, mismatched external
  plan pin and failure in the **eighth** trained layer all reject
  without promoting partial results.

An existing independent OS-process kill and recovery fixture continues
to test the first hybrid block. The eight-layer test so far exercises
fresh navigators and executors in the same OS process; it does not
advertise separate-process recovery on real 27B weights.

## Original pretrained checkpoint experiment

```bash
PYTHONPATH=upstream/vl64:upstream/vl64/tests:tests \
  python -m pytest -q tests/test_qwen_architecture_navigator.py

PYTHONPATH=upstream/vl64:upstream/vl64/tests:tests \
  python -m examples.qwen_architecture_frontier \
    --layers 8 --positions 1 \
    --block-bytes 8388608 \
    --host-budget-bytes 134217728 \
    --layer-budget-bytes 2000000000 \
    --out artifacts/original-architecture-eight.json
```

The live original-weight experiment is **not qualified by code review
alone**. It succeeds only after the exact original-token row traverses
all eight original full-width layers, and **reproduces every exact F64
output digest of layers 0–3 recorded in QUALIFIED-007**. It must publish
the previously unseen exact outputs and state of layers 4–7.
A 64-layer architecture description is *not* a 64-layer numerical
execution receipt.

## Intended next inference frontier

With architecture-directed selection and causal state navigation,
the next research problem is not another manually coded hybrid
block. It is independent PyTorch parity on the original pretrained
weights, prefill and OS-process recovery through a much longer prefix,
then final norm, vocabulary head, and observable generated text.

None of this proves network-native floating-point arithmetic or that
the physical model-worker host disappears. The distinction between
a communication-path navigator and the numerical substrate remains
explicit in every accepted receipt.
