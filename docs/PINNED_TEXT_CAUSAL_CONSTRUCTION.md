# INTERACT-004 — Pinned natural-language input to trained causal state

**Goal:** make the state-space navigator take an ordinary paragraph, use the
actual model-vocabulary tokenizer, and execute an original trained relation
from the resulting token embeddings. Do not make the 64 KiB VL-64 interpreter
store the tokenizer, weights, or the 27B/DeepSeek model.

**Source:** [pinned_text_ground.py](../src/vli/pinned_text_ground.py),
[qwen_causal_head.py](../src/vli/qwen_causal_head.py),
[experiment](../examples/pinned_text_causal_demo.py).
**Tests:** [pinned text and causal state](../tests/test_pinned_text_causal.py).

## New constructive path

```text
Observer supplies real text
   ↓
Exact source-commit tokenizer bytes (admitted into a tokenizer Ground)
   ↓
Native tokenizer executes its BPE/normalizer/decoder over the text
   ↓
Token ID + offset sequence, tokenizer asset hash and intention receipt
   ↓
Original full-sized safetensors index and exact embedding row read
   ↓
Qwen layer-0: full trained 48-gate vector + selected GDN head
  • native Q/K/V and Z projections; depthwise causal convolution
  • A/B gates, delta recurrent state and gated normalization
  • one output-projection row's head contribution
  • carry recurrent and convolution state between consecutive text tokens
  • independently checked recurrence and cross-adapter beta agreement
   ↓
Derived numerical state and hash-qualified causal checkpoint
```

This is **trained computation from actual natural-language input** rather
than an external model API or synthetic hidden vector. The original
unchanged checkpoint remains remote; only individual qualified ranges
cross the observer boundary.

The Qwen experiment is much stronger than INTERACT-003's independent
single-token gate probe because **the second token's head computation
inherits the first token's trained state**. The selected head is a
complete mathematical *head transition*, but not the complete layer:
the remaining 47 heads, full layer output projection, residual/FFN, all
subsequent layers, final normalization, and vocabulary logits are still
needed for actual pretrained language generation.

## Two source-provenance grades — do not conflate them

| Original model | Tokenizer source | Qualification |
| --- | --- | --- |
| Qwen/Qwen3.8-27B at `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | `tokenizer.json` in **same** Qwen model revision | `SAME_PINNED_CHECKPOINT_TOKENIZER` |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 at `1bae0e1088993b9b82de08f59de5be5a34e42170` | `tokenizer.json` from **related** `deepseek-ai/DeepSeek-V4.1-Flash`, revision `dba1be0a40aa45a94ad051997016db3960a90277` | `RELATED_FLASH_RELEASE_NOT_PROVEN_IDENTICAL_TO_1017` |

The pinned DeepSeek 1017 repo supplies a V4.1
`encoding/encoding.py` message-formatter reference but does not include
the root `tokenizer.json` used by this Rust tokenizer path. The related
Flash tokenizer is **not** an attested 1017 token-ID mapping in this
experiment. Its computed IDs are a qualified **cross-release tokenizer
hypothesis** until directly compared with the original 1017 serving
implementation. DeepSeek's embedding-fed MoE remains a **controlled probe**
rather than its actual post-attention/mHC expert input. Both facts are
explicitly retained in every receipt.

Neither route here runs the original chat template. The exact
native tokenizer encodes the *literal observer paragraph*, which can
be round-trip decoded; chat-role markup, control tokens, and assistant
continuation are a separate upcoming Ground.

## Concrete verification

- Tokenizer bytes are fetched at an immutable commit; the Ground retains
  source model identity, tokenizer source revision, SHA-256 and vocabulary
  size. The large asset remains outside the VL Ground.
- The Rust tokenizer executes from the admitted serialization and must
  round-trip the literal text exactly without adding chat special tokens.
  Its exact model-vocabulary token IDs, character spans and text digest
  are recorded, not invented.
- Each selected original embedding row is range-admitted with an independent
  BF16/F16/F32 decoder, a source-witness chain, exact dimensions and SHA.
- On Qwen, the native head retains a 3-step depthwise convolution history and
  a 128×128 trained recurrent state. These states are only committed if
  the full qualified head operation and independent reference succeed.
- The second position executes with the first position's actual state.
  Regression tests compare against a fresh-state second-token evaluation
  and independently calculate two recurrent output channels.
- A separately pinned checkpoint allows the mathematical head state to be
  restored without repeating already committed computation. This checkpoint
  is **VL-I research state**, not a new 64 KiB-VL core allocation.

For deepseek, the related tokenizer maps input text into trained
embedding rows, then uses the existing 384-expert trained router and the
six actually selected FP4 first-neuron slices. It does **not** invent
the missing original causal attention/mHC states.

## Run the experiment

```bash
# Use pinned VL-64 research source (same as prior experiments).
PYTHONPATH=upstream/vl64 python -m pip install -e . 'tokenizers>=0.20,<1'
PYTHONPATH=upstream/vl64 python -m unittest discover \
  -s tests -p test_pinned_text_causal.py -v

PYTHONPATH=upstream/vl64 python -m examples.pinned_text_causal_demo \
  --model qwen-27b --steps 2 --out artifacts/qwen-text-causal.json

PYTHONPATH=upstream/vl64 python -m examples.pinned_text_causal_demo \
  --model deepseek-v41-flash --steps 2 --out artifacts/deepseek-text-probe.json
```

Two GitHub Actions Python-version jobs validate the offline contract
and separate optional real-weight jobs reproduce the public-checkpoint
measurements. Any missing source or unqualified numerical operation
produces `UNRESOLVED`, not a fabricated model answer.

## The next causal-model goal

**A verified first complete pretrained Qwen layer**, not only a selected
head: all 48 value heads + all 16 key heads, complete out-projection and
SwiGLU residual, with cross-token checkpoints and independent reference.
Then continue through its alternating GDN/full-attention stack, producing
original full-vocabulary logits for real model-token continuation.

The second research route is DeepSeek's correct
`tokenizer → embedding → attention/mHC → routed FFN` causal input,
not merely its trained but mismatched embedding-fed router experiment.

The scale constraint remains in VL-64: it should eventually carry the
**minimal interpretation relation** determined by these experiments.
VL-I retains the detailed mathematical research and original checkpoint
experiments. Neither repo should mislabel partial trained states as
useful natural-language model inference before the missing layers execute.

## First real paragraph-driven result, original checkpoints

[GitHub Actions run 44652875978](https://github.com/Kier73/VL-I/actions/runs/44652875978)
succeeded for both public-checkpoint model routes. The human-supplied literal
paragraph was:

> Communication can support computation when information is interpreted through a virtual layer.

Both tokenizer sources returned **13 native IDs** with an exact decode
round-trip. The ID lists differed, as they should for different tokenizer
sources. We executed **the first two** original embedding IDs in each
trained path; no entire 13-token paragraph forward pass is claimed.

| Model | Native trained token ID | Trained numerical output |
| --- | --- | --- |
| Qwen3.8-27B | `63204` | Gated-delta head 0 output-row0 contribution `-0.003374717951` |
| Qwen3.8-27B | `628` | Causally continued head 0 output-row0 contribution `0.002284792043` |
| DeepSeek-V4.1-Flash-1017 (related tokenizer) | `75557` | FP4 selected experts `270,332,155,258,156,32`; weighted partial `0.000129327881` |
| DeepSeek-V4.1-Flash-1017 (related tokenizer) | `588` | FP4 selected experts `85,12,61,43,251,144`; weighted partial `0.000019968239` |

The Qwen independent scalar recurrence errors were `0` and
`6.776263578034403e-21`, respectively. Its 48-gate beta means also
changed from `0.712909898786` to `0.684901588467`. The selected Qwen
recurrent-state and convolution-state digests changed between positions.
The actual Qwen tokenizer source SHA-256 was
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.

The related DeepSeek tokenizer source SHA-256 was
`c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`.
**It is not qualified as the exact 1017 serving tokenizer**, and its
embedding-fed router is still not the model's correct post-attention MoE.
Neither real-model result includes the requested text generated by the model.

The complete trained observation receipts are preserved in the GitHub Actions
artifacts; regression-grounded selected fields are in
[`2026-11-19-real-paragraph-token-causal-results.json`](../experiments/pinned_text/datasets/2026-11-19-real-paragraph-token-causal-results.json).

### Execution substrate, explicitly

These experiments run qualified pretrained mathematical operators on the
GitHub Actions CPU host while receiving original weights over authenticated
HTTPS range connections. They show **native VL mathematical execution and
observer-driven construction**, not yet the envisioned hardware-agnostic
model-host computation inside communication paths. Separating the model
processor from the lightweight interpreter and measuring the transport/
compute split remains necessary to prove that stronger goal.
