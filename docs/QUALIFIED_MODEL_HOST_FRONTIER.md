# QUALIFIED-005 — All Qwen recurrent heads at a bounded model host

The original VL-64 source and its 64 KiB core remain unchanged. This
VL-I experiment asks an operational question: **can the interpreter navigate
a full trained state while the weight-bearing arithmetic and finite source
buffer live at a separate host abstraction?** This is not a claim that the
network medium itself performs the linear algebra.

## What was slow

VL-64 already had a mathematically complete Qwen GDN+SwiGLU layer in
`experiments/model_fabric/complete_native_layers.py`. Its reference-grade
`observed_rows` used Python `math.fsum` for every output and frequently
requested tiny HTTPS ranges. A prior original-checkpoint full-layer workflow
was cancelled after approximately 45 minutes. It cannot be counted as an
executed full pretrained layer.

The new implementation **does not replace its math or the model**. It makes
a host-side alternative to repeated independent HTTP glances and scalar
matvecs, behind exactly the same qualified-range interface.

## Two execution authorities, kept distinct

```text
VL-I / eventual 64KiB VL-64 interpreter
  • choose original pinned model + input intention
  • admit exact tokenizer receipt and source index Ground
  • navigate causal coordinate, request trained operation
  • receive small numerical outputs and evidence receipts
               │
               ▼
Bounded model worker / original weight host
  • fetch original unmodified pretrained blocks by pinned HTTPS 206
  • verify Content-Range, source model/revision, block lengths and hashes
  • retain only bounded model-host RAM, never VL persistent model weights
  • respond to *logical VL range reads* of at most 65,536 bytes
  • execute vectorized BF16/F16/F32 matrix-vector projections
  • independently compare against the original scalar projection laws
  • preserve complete trained causal state for later token positions
```

`BoundedModelRangeHost` has two separate metrics: **actual HTTPS 206 blocks**
and **logical VL64 range reads**. Several logical requests may reuse one
authenticated source block; they are not falsely counted as separate remote
downloads. Its bounded LRU holds bytes only at the *model worker* and may
evict them without changing the mathematical answer. The returned VL64
`QualifiedShardNavigator` replies retain exact model, revision, shard,
tensor extent and SHA-256 witnesses.

The block source has an independent hash-chain report. The 64 KiB logical
range chain remains controlled by the original VL64 interpreter contract;
its own code is not altered. HTTP transport itself only conveys data: the
actual matrix products still consume CPU on the worker in this experiment.

## First original-scale trained numerical target

`QwenAllHeadsRecurrentHost` executes the native Qwen3.8-27B GDN
**all-head recurrent subgraph**, not only one selected head:

- 16 distinct trained Q heads and 16 K heads, group-mapped to 48 V heads
- every required pretrained Q/K/V, Z, A and B projection row
- the complete 10,240-channel depthwise causal convolution
- all 48 learned beta and log-decay gates
- all 48 native 128 × 128 recurrent state matrices
- all 48 gated normalized value-head outputs (6,144 values)
- a real trained output-projection row combining **all** head outputs
- an independent scalar check for both ends of the head array and
  independently recomposed original output-projection row zero
- a paged, externally digest-pinned causal checkpoint and restart in a
  fresh navigator, preserving the same qualified original model revision

The output of this subgraph is **not** the complete first transformer layer;
the remaining 5,119 output-projection rows and the full SwiGLU feed-forward
residual have not been produced by this particular execution.

`QwenVectorizedCompleteLayer` also offers the original VL64
**complete GDN+SwiGLU** mathematical evaluator using the same qualified
vectorized host projections. Its independent *fixture* tests compare three
causal positions to the original scalar layer. A separate opt-in
`--mode complete` public-checkpoint run can execute all trained first-layer
parameters and produce its entire 5,120-dimensional output. **No complete
pretrained-layer result exists until that live run succeeds.**

### Original model and real text

The model stays **Qwen/Qwen3.8-27B**, pinned at
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`, with its original
55,562,855,904-byte advertised total weight payload. Its tokenizer is
loaded from the same exact model revision; its first two native token IDs
come from the full paragraph already used in INTERACT-004.

All tensor reads are admitted by the existing exact source-index and
safetensors-header contracts. The worker never fabricates Q/K/V or silently
substitutes a smaller pretrained model.

## Falsification conditions

The native math is accepted only when every vector and matrix is
dimension-qualified; the returned full-head tensor contains the expected
6,144 values; both independent scalar recurrence and dense projection
references agree; original source block authentication and every logical
range SHA pass; no missing layer component can silently substitute zero;
and a failed later projection must not advance causal state.

An independent checkpoint head is required for recovery. Incorrect
upstream model revision, stale causal head, missing or malformed bytes,
wrong HTTPS status/Content-Range, nonfinite weights or activations and
oversized memory/range requests are explicitly refused.

A successful all-head subgraph receipt **cannot** be promoted into complete
first-layer or full-language inference. Full-model logits, role-formatted
conversations, code generation and model-operated agents remain subsequent
tasks.

## Commands

```bash
# Install a bounded numeric host backend, no entire model framework.
python -m pip install -e . -e upstream/vl64 'numpy>=1.26' \
    'tokenizers>=0.20,<1' 'pytest>=8'

PYTHONPATH=upstream/vl64:upstream/vl64/tests \
    python -m pytest -q tests/test_qualified_projection_host.py \
    tests/test_qwen_all_heads.py

# Original checkpoint: all 48 trained recurrent heads across two text tokens.
PYTHONPATH=upstream/vl64 python -m examples.qwen_model_host_frontier \
    --mode recurrent --positions 2 \
    --out artifacts/qwen-all-heads.json

# Opt-in expensive experiment: entire original GDN+SwiGLU layer, one token.
PYTHONPATH=upstream/vl64 python -m examples.qwen_model_host_frontier \
    --mode complete --positions 1 \
    --host-budget-bytes 134217728 --block-bytes 8388608 \
    --out artifacts/qwen-entire-first-layer.json
```

This is a real **cost separation experiment**. A qualified observation can
feed many VL-local causal state transitions. The host's remote request count,
logical-range count, byte traffic, RAM budget and CPU time make it possible
to measure whether moving compute toward admitted model weights reduces work
compared with streaming each matrix row through an interpreting CPU.

The next optimization should keep the full model worker **outside the
interpreter process** and compare equivalent trained numerical receipts
across independent host lifetimes. Only abstractions that prove essential
for general navigation should be distilled into the unchanged 64 KiB VL64
core.
