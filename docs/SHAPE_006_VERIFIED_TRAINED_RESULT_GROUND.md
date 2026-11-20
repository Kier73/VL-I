# SHAPE-006 — A verified trained result is a navigable computational state

**Status:** executable original-VL-64-trained-math result memory over the **existing E-CPU immutable object store**. The model has not been downscaled in the real-source mode. Original pretrained values are acquired only from a pinned original checkpoint, never guessed from metadata.

**Code:** [trained_result_memory.py](../src/vli/trained_result_memory.py)  
**Reproducible benchmark:** [trained_result_memory_demo.py](../examples/trained_result_memory_demo.py)  
**Adversarial tests:** [test_trained_result_memory.py](../tests/test_trained_result_memory.py)  
**Workflow:** [vli-shape006.yml](../.github/workflows/vli-shape006.yml)

## 1. What has changed

Earlier:
- SHAPE-003 learned the observer-specific shape of a finite computational Ground from actual black-box observations.
- SHAPE-004 retained verified source-state prefixes, avoiding repeated source transitions.
- SHAPE-005 placed *actual original pretrained weight bytes* under original VL-64's source-qualified range protocol using E-CPU's immutable object storage. A successful [live original Qwen3.8-27B experiment](../experiments/pretrained_range_memory/results/2026-11-19-qwen-original-range-memory.json) established three source HTTP ranges and 55,640 original transmitted bytes, then zero repeat HTTP requests. **It still executed the numerical operator on every response.**

**SHAPE-006 now retains the *result* of a real trained numerical operator** under a source/implementation/input/intent-qualified identity. A repeated, identical intention navigates to that prior result rather than reexecuting the trained mathematics. The result remains independently bound to the original observed weight bytes and their E-CPU memory checkpoint. Another intention, even with the same source bytes, requires new arithmetic. The generated response position is not a numeric machine-state index: it binds the observer, original source Ground, causal predecessor and intended computation before its result exists.

This is one material operational step toward **interpreting and navigating states rather than repeatedly evaluating every upstream transition**.

## 2. The original trained operator reused, not simulated

SHAPE-006 uses:

- **VL-64's original \`_decode_float\` tensor-byte interpreter** for original BF16/F32 weights.
- **VL-64's original \`centered_rms_norm\`**, the Qwen3.5 zero-centered RMSNorm relation: \(\mathrm{RMSNorm}(x,w)_i = x_i (1+w_i)/\sqrt{\sum_j x_j^2/n + 10^{-6}}\).
- **VL-I INTERACT-002's existing \`_input\` interpretation** for intentional, parameterized sinusoidal hidden states, not arbitrary pretrained prompt token embeddings.
- **VL-64's original \`QualifiedShardNavigator\`** and exact shard-range/source-witness contract.
- **E-CPU's original \`ComputationalObjectStore\` + \`StructuralMemoryPageBacking\`** for original weight-range and trained result bytes.

An independent algebraic implementation rechecks **each of the \(n\) output coordinates** before any result is admitted into E-CPU. The complete output vector is stored as binary64 values in the original E-CPU content-addressed object format; the response may render a concise projection (width, digest, mean, first/last eight and independent error).

In the **real original pretrained** mode, the native operator consumes the entire original **5,120-dimensional Qwen layer-0 input normalization tensor**. The original model index must exactly match the pinned Qwen/Qwen3.8-27B checkpoint revision, original index SHA-256 and original index Ground digest. No 64-state/small model substitute can qualify as real pretrained evidence.

In the separate **test fixture** mode, the model-mimicking index contains an invented 64-element BF16 tensor and a local TCP receiver. Every output is explicitly marked **synthetic fixture**, even if the fixture borrows the actual model's path and revision labels. The fixture is a protocol verification only.

## 3. The identity that permits navigation

A result is admitted only when all these fields agree:

| Evidence/identity | Why it matters |
| --- | --- |
| Original checkpoint model, revision, index bytes + Ground | Prevent changing weights by changing the named model |
| Original VL-64 qualified weight range and source witness | Proves specific bytes were observed through a caller-qualified source |
| Source-observation checkpoint digest | Prevent silently rebinding the result to a changed set of source observations |
| Native trained operator and Python/interpreter source-code fingerprint | Prevent reusing a result under different arithmetic or input interpretation |
| Observer intention digest | Prevent a different stated target, waveform, phase or view being treated as identical |
| Exact numerical input vector digest | Captures the actual hidden vector consumed during original execution |
| Complete trained output digest | Identifies the realized result, not a prediction |
| E-CPU content-addressed result manifest/page hashes | Prevent altered retained bytes being replayed |
| Independent original-run numerical error | Records the original full-vector algebraic check |
| Externally pinned checkpoint head + causal event chain | Prevent undetected replay of a different observer history |

A **causal response coordinate** is selected from original source identity + observer + intention + predecessor *before* numerical realization. The same trained value can therefore appear at different observer-history coordinates. Position identity and numerical content identity serve different purposes.

### Fresh versus retained

**Fresh:**
1. Construct position from intention and causal head.
2. Validate model/index/implementation and original source host.
3. Read source-qualified trained weight ranges through VL-64 and SHAPE-005.
4. Generate intended hidden vector using the existing VL-I interpreter.
5. Execute the original VL-64 trained operator; independently verify every output component.
6. Retain the entire output, exact source checkpoint and result receipt using E-CPU's original byte-object store.
7. Render a response from the verified result and commit one causally linked event.

**Same intention:** construct a *new* response-state coordinate, revalidate source identity and proof, fetch the exact E-CPU output object, verify all its pages/manifest and digests, and render the result **with zero source range reads and zero new original trained arithmetic**.

**Changed intention:** preserve original source-memory relations but invoke the original trained operator on the newly generated input. The resulting output digest differs. No unproven interpolation, arbitrary language generation or unknown tensor value is inserted.

**After restart:** the external expected checkpoint head is mandatory. Rebuild original E-CPU byte memory and result memory from their snapshots. Both must independently verify all immutable object bytes and source/implementation receipt contracts. The history is validated against each recorded result and its original causal coordinate. The original source need not be present for the exact same previously qualified result.

## 4. Measured initial synthetic TCP test

The initial [SHAPE-006 Actions run 44655081829](https://github.com/Kier73/VL-I/actions/runs/44655081829) passed **13 tests**, including changed source/intent/implementation, altered E-CPU pages, forged numerical receipts, rehashed checkpoints and cold-to-warm restore. (Later tests additionally challenge forged causal coordinates and false reuse claims.)

The test uses a real loopback HTTP TCP byte-range server, but with **synthetic BF16 weights**, not the actual model.

| Intent / work item | First | Identical repeat | Changed intention |
| --- | ---: | ---: | ---: |
| Actual source TCP requests | 3 | **0** | **0** |
| Original source bytes | 245 | **0** | **0** |
| Native VL-64 trained operator executions | **1** | **0** | **1** |
| Result E-CPU CAS use | publish | verify+read | publish different result |
| End-to-end single-run observation time | **3.914 ms** | **0.922 ms** | **1.217 ms** |
| Causal response coordinate | New | **Different position** | **Different position** |

The warm observation **did not execute** the pretrained-style normalization law. A differently phrased intention, or a changed numerical phase, is not accidentally considered equal. The result can be recovered from the durable E-CPU object checkpoint with neither source network nor trained arithmetic.

**Performance caveat:** one run of a synthetic 64-element operator is not a meaningful prediction of performance for a full real model, and the warm path still pays E-CPU memory checks, source provenance checks and output rendering. For extremely cheap arithmetic, retaining/checking outputs can be *slower* than reexecuting them. The recorded operation counts and wall times are separate evidence, not interchangeable metrics.

## 5. Original full-size model live experiment

The same code supports:

\`\`\`bash
python -m examples.trained_result_memory_demo --mode live \
  --out artifacts/shape006-real-qwen.json
\`\`\`

It retrieves the original pinned **Qwen/Qwen3.8-27B** index, verifies the full-scale model identity, and executes original VL-64 native Qwen trained normalization on its entire original 5,120-dimensional input tensor. The next identical intention must re-use the **full binary64 result vector** without either original HTTPS traffic or original trained arithmetic. A third intention changes phase and frequency, and must compute a new distinct vector using the retained original pretrained weights. Finally the result and source memory are reconstructed after a source-free restart.

The CI workflow attempts this real original-size experiment on a merged main push or explicit manual dispatch. If HTTPS fails or source evidence changes, its status must remain \`UNRESOLVED\`; the local fixture is **not** acceptable substitute evidence.

**Do not conflate:** a real pretrained *single normalization operator* with original full-model decoder inference, language-model next-token parity or universal interpreter execution. In particular, an analytic sinusoidal hidden input is not a trained token embedding or genuine conversation history. The research now has a truthful, executable result-reuse path to extend to the full original trained layer graph.

## 6. Where to take the interpreter

The hard next stage is to lift *reusable mathematical laws* into Grounds that can address many future result states **without** retaining each full output vector separately:

- Build source- and scope-qualified result positions for the existing complete E-CPU model orbit/layer handoff, including real predecessor hidden-state digests.
- Reuse exact numerical results when the original trained input, source and operator match, even when the surrounding dialogue history and rendered view differ. Avoid confusing **semantic equivalence** with mere repeated requests.
- Mine observed partial Grounds for **symbolic or generative relations** that can be independently proved over whole families of intentions, not merely memoize one output.
- Compare total original trained arithmetic, repeated network reads, structural memory, proof, checkpoint, rendering and energy on the original model scale.
- Use VGI/IFS active attention to determine which operations deserve observation and retention by their expected future information value.
- Pursue an interpreter whose own semantics are reconstructable/navigable from verified Grounds, with increasingly little repeated conventional evaluation.

**The principle:** the response state exists as an *intention-addressed position*; computing or navigating to its realization is a choice the interpreter makes under evidence. SHAPE-006 now makes that choice operational for one qualified trained subgraph.
