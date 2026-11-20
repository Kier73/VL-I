# QUALIFIED-007 — Original Qwen first hybrid block: GDN × 3 → gated GQA

This experiment takes QUALIFIED-006's two causal original-weight layers
and attempts the next, architecturally novel frontier:
**complete original layer 0 → 1 → 2 → full-attention layer 3**.
It stays entirely within VL-I's model worker and does not change VL-64.

## Original pinned model, not an invented architecture

Source identity: `Qwen/Qwen3.8-27B`,
revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
The model card and `text_config.layer_types` document 64 alternating
hybrid layers: `GDN,GDN,GDN,gated GQA` repeated 16 times.

For the original model, layers 0–2 use 16 Q/K and 48 V GDN heads, each
with key/value dimensions of 128. Layer 3 uses **24 query heads, 4
key/value heads, 256-wide heads, 64 rotary dimensions, rotary base 1e7**,
a gated Q projection and learned zero-centered Q/K RMSNorm, followed
by a full SwiGLU feed-forward residual. The full hidden width remains
5,120, with an intermediate feed-forward width of 17,408.

The attention operator matches the reference Transformer sequence for
text positions: zero-centered RMSNorm; original pretrained Q, K, V
matrices; separate head RMSNorm for Q and K; first-64-dimensional
rotate-half RoPE; full causal grouped-query softmax over all prior
positions; elementwise sigmoid gating of heads; O-projection;
residual; post-attention zero-centered norm; trained SwiGLU and final
residual. A text-only input uses the same token position on all three
M-RoPE axes, so the rotary angles reduce to the standard text RoPE.
This does not implement images or arbitrary multimodal position axes.

Layer 3's KV history differs from GDN's fixed-size delta/convolution
state. A causal attention result retains its own ordered key/value
heads and **cannot** be replaced by the GDN state of another layer.

## Exact causal composition

```text
pinned paragraph token and original embedding
  → pretrained GDN layer 0, state S0[t]
  → exact 5,120 F64 numerical output H0[t]
  → pretrained GDN layer 1, state S1[t]
  → exact H1[t]
  → pretrained GDN layer 2, state S2[t]
  → exact H2[t]
  → pretrained gated GQA layer 3, KV history K3[≤t], V3[≤t]
  → exact 5,120 F64 numerical output H3[t]
```

Every original-weight projection passes through
`BoundedModelRangeHost` and the original
`QualifiedShardNavigator` 64 KiB logical request contract. The
worker coalesces HTTPS source blocks separately. It does **not** claim
that HTTPS itself computes matrix products.

The coordinator verifies each layer's qualified receipt, model revision,
Ground identity, causal position, complete output width, state digest
and exact F64 handoff. Layer-0 output digest must match layer-1 input;
similarly layer 1→2 and layer 2→3. A semantic causal head includes
original trained numerical/state evidence but excludes observer-local
network read chronology. Every read still has its own source provenance.

A downstream error rolls back all four *semantic causal states* to the
last fully accepted token position. It does not retract historical
network reads or promise distributed atomic side effects.

## Recovery and falsification

Three separately page-stored GDN states and the fourth layer's KV state
are placed under a single externally pinned, model-bound checkpoint.
Manifest and page digests must agree; stale/mismatched lineage, corrupted
pages, wrong layer ordering, absent weight tensors, nonfinite results,
invalid head shapes and failed downstream projections are all refused.

`tests/test_qwen_hybrid_block.py` exercises independent scalar attention
mathematics over multiple causal positions, tests the complete four-layer
hand-off against the existing independent scalar three-GDN oracle plus
the separate attention oracle, and tests restart, tampering and rollback. A separate
`tests/test_qwen_hybrid_process_recovery.py` test constructs and pins
the four-layer checkpoint in one Python operating-system process,
**forcibly terminates that process**, then starts a second independent
process and verifies byte-for-byte output and exact causal-head parity
against an uninterrupted worker. The process-kill test currently runs
on controlled original-topology mathematical fixtures; a real-checkpoint
process kill/recovery test is a distinct next experiment.

## Commands

```bash
python -m pip install -e . -e upstream/vl64 'numpy>=1.26' 'pytest>=8'
PYTHONPATH=upstream/vl64:upstream/vl64/tests \
  python -m pytest -q tests/test_qwen_hybrid_block.py

# Original checkpoint, only the 3 GDN layers
PYTHONPATH=upstream/vl64 python -m examples.qwen_hybrid_block_frontier \
  --mode gdn3 --positions 2 --out artifacts/qwen-original-first-three-gdn.json

# Original full hybrid block, including layer 3's gated attention
PYTHONPATH=upstream/vl64 python -m examples.qwen_hybrid_block_frontier \
  --mode hybrid4 --positions 2 --block-bytes 8388608 \
  --host-budget-bytes 134217728 --layer-budget-bytes 2000000000 \
  --out artifacts/qwen-original-first-hybrid-block.json
```

The original-weight run and its receipts are not accepted until the
separate GitHub Actions job succeeds. The original model requires
substantial HTTP weight traffic, and source memory is bounded on the
worker. Numeric compatibility with an independently executed
**real pretrained** Hugging Face/PyTorch whole-layer reference is not
yet claimed merely from fixture tests.

## Research objective

This experiment tests the first change of **state-space topology** inside
a single model: from recurrent GDN matrices to growing temporal KV
history, with no loss of the original trained numerical path.
A successful experiment establishes four original trained layer
transitions, not a complete 64-layer decoder, final vocabulary logits,
generated text, physical substrate independence, or network-native
arithmetic. The next step is independent real-weight reference parity,
separate process rehosting, and bounded KV navigation over long prefixes.

## Original pretrained first hybrid block — confirmed success

[GitHub Actions run 44664814333](https://github.com/Kier73/VL-I/actions/runs/44664814333)
**passed** both Python 3.11 and 3.13 independent scalar/recovery jobs and
the separately executed **original trained 27B first hybrid block**.

Both original natural-language paragraph tokens `63204`, `628`
were advanced across **all four full 5,120-wide trained layers**,
yielding eight complete pretrained layer executions:
three GDN layers and the first gated full-attention layer.
Their state and output digests are distinct as expected.
Every F64 layer output was passed exactly to its successor.
The first two layers, both tokens, **reproduce the QUALIFIED-006
trained F64/F32 outputs and recurrent digests exactly**.

| Measured source-backed property | Original experiment |
| --- | ---: |
| Full original trained layer operations | **8** |
| Trained bytes each GDN layer/token | **766,443,520** |
| Trained bytes attention layer/token | **744,488,960** |
| Qualified logical <=64 KiB ranges | **93,044** |
| Actual HTTPS 206 source requests | **775** |
| Remote HTTPS bytes | **6,501,171,200** |
| Source RAM block budget on worker | **128 MiB** |
| GDN state pages at first-token checkpoint | **300** |
| Full-attention KV pages at checkpoint | **1** |
| Combined checkpoint state F64 bytes | **19,628,032** |

The full pretrained run observed one complete attention KV position,
checkpointed that historical state together with all three GDN states
under one externally pinned digest, recovered into fresh qualified
navigators/executors, and then advanced the second token with two KV
positions. The separately exercised OS-process-kill recovery is on
fixture weights, not a claim that a full original-weight worker was
physically restarted.

The source artifact is
[GitHub Actions artifact 14597722183](https://github.com/Kier73/VL-I/actions/runs/44664814333/artifacts/14597722183).
The independently regression-tested exact digest summary is archived at
[`2026-11-19-real-original-qwen-first-hybrid-block.json`](../experiments/qwen_all_heads/datasets/2026-11-19-real-original-qwen-first-hybrid-block.json).

**Qualification:** This is four original trained layers executed with
VL-I's numeric semantics, not an independent PyTorch parity result on
the complete original 27B checkpoint; nor does it establish the remaining
60 original decoder layers, final norm, logits, generated text or
communication-native arithmetic. The bandwidth-heavy original run is
manual opt-in after this completed observation, while scalar and
archived provenance tests continue automatically.
