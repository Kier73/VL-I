# QUALIFIED-006 — Exact two-layer original Qwen causal composition

This continuation advances the Qwen/Qwen3.8-27B original checkpoint
beyond QUALIFIED-005's complete layer-0 execution without altering VL-64.

## Concrete navigation relation

For each tokenizer-admitted paragraph token, read exactly one pretrained
embedding row, execute **all** Gated DeltaNet + SwiGLU mathematics of layer
0, and feed its **actual** 5,120-dimensional numerical output directly
into pretrained layer 1. A separately qualified complete layer-1 output
is produced. The same model revision and index Ground are required.

```text
pinned original token ID
  → source-qualified embedding row E[t]
  → GDN + SwiGLU layer 0 with causal state S0[t]
  → exact 5,120-value intermediate hidden output H0[t]
  → GDN + SwiGLU layer 1 with separate causal state S1[t]
  → exact 5,120-value output H1[t]
```

There is **no** independent hidden-vector selection for layer 1. In a
consecutive causal input sequence, each trained layer retains its own
recurrent and convolution state. Both are checkpointed after a completed
token and independently restored into fresh navigators and executors.

Each transfer witness hashes the *actual f64 numbers passed in Python*,
rather than merely the f32-rounded digest exposed by previous stage
receipts. A layer's output f64 digest must equal its successor's input
f64 digest. The existing f32 digest remains available for backward
comparison to QUALIFIED-005 archival results.

## Commit, failure, and recovery

`QwenCausalGDNChain` validates contiguous layer coordinates, model and
revision, index Ground, input geometry, output width and finite values,
and both complete-layer receipts. A token transition commits its hash
chain only after **every** layer has succeeded. On failed downstream
arithmetic, incomplete output, or contradictory receipt, previously
advanced layer-0 and layer-1 *semantic* causal states are rolled back.

Acquired HTTP/qualified-range observations remain historical observations
and are not retroactively erased. Host external side effects (if any)
are not transacted by this prototype.

The checkpoint is a manifest containing separate, existing
`export_qwen_causal_ground` states for each layer, plus the exact
layer-coordinate sequence, model/revision, index Ground, input-sequence
lineage, chain head, final output digest and an externally pinned digest.
`restore` refuses wrong lineage, changed manifest, wrong layer or
revision, corrupt pages or inconsistent causal positions.

The worker and original layer states are not inside the 64 KiB
interpreter. This first recovery test **recreates independent executors
and navigators within the worker process**, not a separately restarted
operating-system process.

## Verification

```bash
python -m pip install -e . -e upstream/vl64 'numpy>=1.26' 'pytest>=8'
PYTHONPATH=upstream/vl64:upstream/vl64/tests \
  python -m pytest -q tests/test_qwen_layer_chain.py

# Original full-size weights, original pinned tokenizer:
PYTHONPATH=upstream/vl64 python -m examples.qwen_layer_chain_frontier \
  --positions 2 --host-budget-bytes 134217728 --block-bytes 8388608 \
  --out artifacts/qwen-first-two-layers.json
```

The independent fixture reference executes the upstream
`QwenQualifiedLayer` **scalar** implementation on two layers and
multiple causal positions. The vectorized chain must match it
numerically, then survive checkpoint/restart, source failures,
tampering, and stale recovery pins.

The original checkpoint job is a separate opt-in/CI experiment with
original-size qualified weight ranges. **The presence of a passing
fixture suite does not constitute a completed original trained-layer-1
run.** The original pretrained experiment must succeed and preserve a
receipt before claiming that achievement.

## Scientific qualifications

This adapter handles the model's first two Gated DeltaNet layers only.
The hybrid architecture later requires **full-attention** adapters,
then the remaining 64-layer backbone, final norm and output head
before logits or generated text can be claimed.

Current mathematical authority is the upstream Qwen GDN equations, with
independent fixture scalar parity. The original 27B trained output has
not yet been cross-checked against a separately executed Hugging Face /
PyTorch reference with identical weights, dtypes, tokenization, and
input state. Work remains to test the layer chain on separate operating-
system model-host processes and verify identical causal receipts after
that kind of rehoming.

A significant research question is now executable: can an observer
navigate successive original trained states while the interpreter retains
only compact intention, state coordinates, receipts and source authority?
This experiment tests composition and recovery; it **does not** establish
that network transport performs the numerical arithmetic or that it is
cheaper than conventional inference.
