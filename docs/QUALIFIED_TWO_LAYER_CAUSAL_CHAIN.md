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

## Successful original checkpoint observation (QUALIFIED-006)

The source-backed [GitHub Actions run 44660157399](https://github.com/Kier73/VL-I/actions/runs/44660157399)
**passed** the full original trained two-layer execution and both
Python 3.11/3.13 scalar-oracle regression jobs. Original model revision:
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`.

Two original paragraph tokens, **63204 and 628**, advanced both complete
trained GDN+SwiGLU layers: four complete layer executions. On **both**
positions the exact 5,120-value f64 handoff witness from layer 0
**equals** the input witness for layer 1. Both layer-1 outputs have
complete, distinct f32/f64 digests. The **first original layer-0 f32
output digest exactly reproduces** QUALIFIED-005's archived first-layer
receipt: `b87fe04766701c2433a23b5d40e892c6d84a909c976d4ce18bd842edc8555aa8`.
The layer-0 recurrent digests for both positions also match the
original all-48-head experiment.

Observed bounded host cost:

| Observation | Original two-layer causal run |
| --- | ---: |
| Completed original trained layer operations | **4** |
| Qualified original tensor bytes, each layer/token | **766,443,520** |
| Logical <=64 KiB range reads | **46,866** |
| Actual pinned HTTPS 206 requests | **391** |
| Total source HTTPS bytes | **3,279,945,728** |
| Source block resident budget | **128 MiB** |
| Recovered GDN causal pages | **200** (100 per layer) |
| Recovered F64 mathematical state | **13,074,432 bytes** |

Both layer states were independently recovered into **fresh navigators
and executors within the same worker process** after the first token,
using two content-addressed state-page sets and a separately pinned
manifest head. This is not yet process-kill/recovery across an operating
system host boundary.

The summarized exact digests, original source identity and testable
observations are archived at
[`2026-11-19-real-original-qwen-first-two-complete-layers.json`](../experiments/qwen_all_heads/datasets/2026-11-19-real-original-qwen-first-two-complete-layers.json).
The original full receipt is the Actions artifact
`vli-qualified-original-qwen-two-complete-causal-layers`
(artifact ID `14594509188`).

**Reproducibility:** the expensive real-weight job is now manual opt-in
through workflow dispatch, to prevent 3.28 GB of redundant remote
source traffic on every ordinary pull-request change. The two
independent scalar regression matrices and immutable artifact tests
continue to run automatically.

This result establishes original-weight causal composition across
**two** complete layers, not independently checked whole-pretrained-layer
equivalence to the canonical HF framework, nor a full 64-layer
forward pass, logits, generated text, or communication-native floating
point operations.
