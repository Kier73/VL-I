# INTERACT-003 — Actual pretrained token rows as operative Virtual Grounds

**Purpose:** replace INTERACT-002's analytic synthetic hidden inputs with **real
learned token embeddings** read from each unchanged, original-scale model.
The model weight bytes live only at their qualified source. A small interpreted
Ground carries the source identity, an intended token coordinate, the
selected native relation, evidence receipts, and the requested output forms.

This is the first step from source weight data to native trained causal
construction; it does **not** yet form a full Qwen or DeepSeek model response.

## The exact experiment

1. Admit the same index/hash/revision as the original real-weight experiments.
2. Locate an opaque state coordinate from the Ground, previous causal head,
   and a *specific native vocabulary token ID* **before** reading its embedding.
3. Obtain only one corresponding trained `embed.weight` row through VL-64's
   existing qualified safetensors 64 KiB-range reader; validate width, dtype,
   byte count, source identity, and an independently decoded reference.
4. Execute the corresponding existing native partial trained model relation.
5. From the **computed** result form text, JSON, Markdown, a standalone
   executable Python result-composition law, and byte-token observation.
6. Render the selected views through the existing E-CPU/VLD response DAG.
7. Commit a hash-bound causal event; allow a new interpreter process to
   resume without needing to retain the model parameters.

The Ground **never stores or synthesizes the embedding table**. Its operator
receives the selected row transiently and its committed receipt stores only
the row's original tensor name, source-range witness and SHA-256 digest.
There is no standalone model engine or opaque provider chatbot endpoint
on this experimental path.

### Full-scale source and applicability

| Model | Native embedding source | Executed relation | Causal qualification |
| --- | --- | --- | --- |
| **Qwen/Qwen3.8-27B** | `model.language_model.embed_tokens.weight` | Actual learned embedding → layer-0 input RMSNorm → 48 learned gate and decay relations | **Genuine initial-token embedding input to Qwen layer 0**, partial trained numerical computation. Full Gated DeltaNet QKV and subsequent layers remain to be composed. |
| **deepseek-ai/DeepSeek-V4.1-Flash-1017** | `embed.weight` | Actual learned embedding → 384 trained gate logits → six selected trained FP4 expert first-row fragments | **Embedding-fed controlled probe, not the native DeepSeek layer-0 MoE input**: attention and mHC transitions lie between token embeddings and its routed MoE. No valid full-model prefill/decoder state is implied. |

Both use the same original pinned immutable checkpoint revisions and original
embedding widths. The experiment does **not** replace a large model with a
small surrogate, does **not** load a whole table, and does **not** count a
remote HTTP weight range as native CPU-free work.

### Why explicit token IDs?

This experiment deliberately uses **100 and 101** as two externally specified
original-vocabulary row coordinates. Such an ID can locate a genuine trained
embedding without having a reliable interpretation as the tokenization of
an arbitrary user paragraph.

A future tokenization Ground must separately qualify the *same pinned
checkpoint's original tokenizer assets*, exact chat template, token order and
special-token treatment before promoting these IDs into observed text
semantics. A later DeepSeek attention/mHC Ground must then produce the **true**
MoE input hidden state, rather than feeding its embedding directly to the
expert router. Only after the full decoder produces logits may we claim
pretrained next-token inference.

### Validated interpreter mechanisms

- Distinct token IDs select distinct actual weight rows, causal coordinates,
  row digests and downstream numerical results.
- Invalid token, missing embedding, incompatible shape, nonfinite hidden
  state, truncated trained row, wrong full model revision and altered intent
  are rejected rather than silently changing the computation.
- Both model families use the existing qualified BF16/F16/F32 decoders
  and original FP4/E8M0 scales in their native mathematical adapters.
- Native renderer builds requested views from the **new computed state**;
  independently executed generated Python checks the constructed result.
- Checkpoint continuity is hash-qualified; restoration verifies original
  committed event identity without re-downloading any trained parameters.

### Run

```bash
# Interpreter and independent tests (no actual model download)
PYTHONPATH=upstream/vl64 python -m unittest discover \
    -s tests -p test_token_embedding_ground.py -v

# Optional public pinned checkpoint experiments (original scale retained)
PYTHONPATH=upstream/vl64 python -m examples.token_embedding_demo \
    --model qwen-27b --tokens 100 101 \
    --out artifacts/token-qwen.json
PYTHONPATH=upstream/vl64 python -m examples.token_embedding_demo \
    --model deepseek-v41-flash --tokens 100 101 \
    --out artifacts/token-deepseek.json
```

GitHub Actions tests the interpreter on Python 3.11 and 3.13 and runs
the exact original checkpoint numerical route on PR/explicit dispatch.
Live results are archived separately from fixtures; failures leave an
explicit `UNRESOLVED` result, not a fabricated native output.

### Next intended structural realization

`pinned tokenizer → original token embeddings → complete first-layer
causal state → next trained layer ... → full vocabulary logits → next token`

That is the operational Ground to pursue, not merely a larger catalogue
of weights. Every new transition should be executable, independently
qualified, recoverable, and expressive as the requested output.

The VL-64 core is unchanged; VL-I is free to explore the richer relations
until an irreducible general interpreter primitive can be extracted within
the 64 KiB budget.
