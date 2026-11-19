# VL-I research plan — from observed model data to Virtual Layer Intelligence

**Plan version:** 0.1 — 2026-11-19  
**Nature:** an executable research programme, not a claim that the future milestones have passed.  
**Core hypothesis:** A trained model's information, transitions and causal state can be represented as a navigable Virtual Ground. A qualified VL interpreter can produce the model's useful functionality through that representation, and may eventually realize chosen observables more effectively than conventional layer-by-layer model execution.

The goal is not to replace a large model with a smaller one. It is to change the **way the original model's structure and trained information are interpreted and realized**.

## Research charter

A development increment is useful if it advances at least one of four things:

1. **Model addressability:** more of an actual model's original data and semantics can be observed, reconstructed and identified.
2. **Ground fidelity:** the lifted representation preserves more verified behavior and meaningful state transitions.
3. **Qualified navigation:** an intention can select a less costly and/or more capable route to the same declared observation.
4. **Continuity and generality:** the result works across requests, model families, sources, scales or observer-hosts with retained evidence.

Every increment must have a reproducible demonstration or explicit negative/control result.

## Working definitions

- **Observation** — specific acquired bytes, pixels or reported behavior, with provenance and exact coverage.
- **Description / Virtual Ground** — a versioned interpretation mapping observations into parameters, states, relations and possible transitions.
- **Intent** — an observer-specified result with input, output, authority, cost and fidelity constraints.
- **Realization** — a concrete result obtained by an authorized VL operation, an admitted relation or a baseline backend, with a declared path type.
- **Verified** — only what independent tests, proof obligations or measurement scopes actually establish.
- **Predictable / unpredictable** — relative to a given context and observer, with observed uncertainty and missing information explicitly tracked.

A large state space is not inherently a better description. A Ground earns promotion by predicting *more relevant independently observed distinctions per unit cost* and preserving the behavior requested by the intention.

## Immediate project order

### Phase A — Establish an independently reproducible model intake

**Candidate:** pinned pretrained SmolLM2-135M checkpoint already used by the [VL-64 complete decoder experiment](https://github.com/Kier73/VL-64/blob/main/docs/REAL_FULL_PRETRAINED_DECODER_OBSERVATION.md). This is an initial validation fixture, **not** a cap on eventual model size.

**Implementation actions:**
- Pin exact upstream revisions in a lock manifest: VL-64, E-CPU, IFS, VGI, VLD; record source license and permitted access.
- Import model config, tokenizer files, Safetensors inventory and graph; preserve distinctions between actual bytes, header-only references and inferred semantics.
- Build an adapter around existing HFR NSD and HFG/weight Ground contracts instead of copying model internals.
- Define a model-independent transcript for one observational input and one decoded continuation.

**Pass gate:** A verifiable model identity, complete typed graph, exact tensor-to-byte addressing for the chosen operation, tokenizer identity, and separately audited provenance. Missing or altered bytes cause explicit refusal.

**Output:** model Ground manifest, observations, adapter tests and independent verifier.

### Phase B — VL-native complete pretrained forward pass

**Implementation actions:**
- Reuse VL-64 qualified range readers and VLD numerical realizer ideas under a single VL-I interface.
- Express embeddings, all decoder blocks, norm, final head and next-token selection as typed VL relations.
- Make the *experimental VL path* call only VL-native operators. Do not use Transformers/PyTorch model.forward or a hosted model API there. They remain isolated baselines.
- Run both direct numerical and independently implemented reference computations on the same pinned weight bits and input IDs.

**Pass gate:** all layers and final logits are independently compared under declared precision/tolerance. A selected token ID must match, or the mismatch must be diagnosed. Missing operators or weights refuse. Count actual work, bytes read and memory used.

**Output:** exact-input, full-model numeric transcript plus cost and verifier receipts.

### Phase C — Tokenization, complete multi-token behavior and intention

**Implementation actions:**
- Import the model's actual tokenizer assets, preprocessing and native chat template when applicable; distinguish plain role formatting from trained template semantics.
- Implement prefill → logits → selection/sampling → next-input → incremental decode in the VL, carrying real KV or recurrent memory.
- Support paragraph generation, code output and structured tool proposals from *instruction-tuned* full pretrained models, not fabricated canned responses.
- Keep tool invocation caller-authorized and external to untrusted model content.

**Pass gate:** independent multi-token greedy equivalence over several fixed prompts; deterministic sampling tests where seeds and numeric kernels permit; parser/format checks; no tool side effect absent explicit authority. Count generated tokens, failures and model-native semantics.

**Output:** text, code and authorized-action test corpus with complete causal traces.

### Phase D — Recoverable model state and multiple VL instances

**Implementation actions:**
- Reuse E-CPU state lineage and qualified communication channels.
- Checkpoint prefix tokens, position, RNG/sampling state, KV/recurrent values, model-ground revision, weight identity and reference outputs.
- Resume on a separate process/observer-host with verified checkpoint head and a newer authority epoch. Do not replay already committed semantic effects.
- Run two independent model Grounds or separate sessions with isolated memory and policy.

**Pass gate:** identical continuation relative to uninterrupted reference where deterministic; tampered checkpoints, wrong model or stale epoch refused; per-session isolation; throughput/recovery overhead measured.

**Output:** independent recovery verifier, two-model and multi-session evidence.

### Phase E — Predictive description formation and the first genuine navigation advantage

**The experimental heart of VL-I.** Having established numerical parity, search for a model-derived relation that changes **how much computation is required** for a fixed observable.

Candidates:
- reuse certified causal prefixes or whole subgraph results under identical inputs and snapshots;
- compose parameter-independent algebraic laws across compatible operator sequences;
- discover prediction-supported equivalence classes and quotient states under a specific observation contract;
- exploit sparse expert choice only after exact qualified router observation;
- use low-rank, symmetry or factorization relations only when resulting contract fidelity is independently established;
- active selection of the next source range or computation needed to determine an intention.

**Pass gate:** held-out prompts and adversarial counterexamples, reference-equivalence constraints, nontrivial baseline, exact saved operations and end-to-end resource measurements. Separate mathematical proofs from finite empirical evidence. Report refusals and coverage, not only successful outputs.

**Output:** two comparable execution paths for the same model/intention and an explicitly measured Pareto improvement or negative result.

### Phase F — Large-model and architecture-general scale

**Implementation actions:**
- Apply identical Ground schema to a second dense family, a sparse MoE family and a hybrid-attention architecture using actual immutable model identities.
- Maintain exact original weight data and trained-model scale by default; record all quantized reference formats *as they actually exist*, without forced extra model compression.
- Stream only necessary ranges, support out-of-core and distributed pages, but do not claim that streaming all the weights is free.
- Qualify each operator family (e.g. attention variants, expert routing, scale factors and model-specific state) before declaring execution.

**Pass gate:** at least two distinct families with full-model, full-tokenizer multi-token test parity; larger-model experiments differentiated as architecture-only / subgraph / full-model / continuous-decode. Resource and access dependencies documented.

**Output:** scale and generality matrix across local, Hub and multi-host reads.

### Phase G — Ground acquired from more than files

**Implementation actions:**
- Support images/screenshots of model pages and observations as evidence-scoped, uncertain capture sources.
- Cross-check visually inferred model facts against official machine-readable pinned metadata when available.
- Use E-CPU / VL-64 generic byte fabric and live observation to discover missing descriptions.
- Develop discriminating experiments that can request the next visual capture, model file, weight range or output probe.

**Pass gate:** known-data demonstration where partial images produce *candidate* architecture facts and a later byte-level source either validates or contradicts them. The system never invents unobserved weight payloads.

**Output:** grounded multimodal observation adapter and evidence ledger.

### Phase H — Compact projection into VL-64

Extract only proven interpreter relations into the size-constrained companion.

**Pass gate:** maintain the published 65,536-byte source-package ceiling, pass VL-64 test/verification suite, preserve result contract and supply exact size evidence. Keep model snapshots, range acquisition, external adapters and large research harnesses outside the compact interpreter.

**Output:** small interoperable VL seed with verifiable full-model external data access.

## The first research experiment to start now

**VL-I-001: observed-data full-model Ground parity.**

Use an immutable full pretrained decoder snapshot with a known existing numerical reference. Stage the experiment:

| Stage | Required observation | Refusal example |
| --- | --- | --- |
| 1. Read | complete config, tokenizer provenance, indexed header/weight scope | incorrect source revision |
| 2. Lift | typed graph and parameter-address Ground | missing/ambiguous operator semantics |
| 3. Realize | full VL-native operator graph and output logits | any absent required trained tensor |
| 4. Compare | independent logits and tokens across multiple inputs | reference beyond declared tolerance |
| 5. Continue | genuine prefill and multiple new tokens | incorrect KV or position lineage |
| 6. Recover | resume without repeated committed effects | stale/mismatched checkpoint root |
| 7. Optimize | alternate Ground with held-out cost/accuracy improvement | shortcut that only fits the training trace |

This experiment is complete only when actual results and executable independent checks are published. Existing predecessor results provide test fixtures and implementation references, **not inherited VL-I pass status**.

## Parallel workstreams for independent agents

| Stream | Responsibility | What to hand off |
| --- | --- | --- |
| Source & provenance | live/local/HF/capture readers, snapshot pins, exact scopes | immutable manifest + negative tests |
| Lift & semantics | NSD, graph IR, Ground interpreter, model-family contracts | typed operator Ground + unresolved obligations |
| Numerical realization | VL-native tensor, attention, MoE and recurrent relations | independent reference-comparison harness |
| Intention & tokens | tokenizer, role template, sampling, structured output | reproducible multi-token sessions |
| State & fabric | KV/prefix checkpoint, source-range cache, transport, host recovery | independent crash/rehome transcripts |
| Predictive theory | equivalences, active attention, new algebraic shortcuts | held-out evidence, counterexamples and saved work |
| Verification & cost | independent receipt checker, energy/work/I/O measurements | machine-readable evidence and status report |

Workstreams must use a common model ID, Ground receipt protocol and experiment IDs. Avoid inventing incompatible abstractions.

## Milestone declaration rules

**Proposed → Fixture-tested → Numerically realized → Independently reproduced → Contract-qualified → Generalized to additional model families**.

Do not mark an entire milestone complete when only one substage passed. An open problem should be written as an actionable experiment, not as a permanent prohibition.

A claimed gain should say *over which baseline, for which model/input/precision/host, with which costs included*. A candidate that saves arithmetic by asking a conventional backend to compute the answer is not a VL-only inference gain.

## Success criteria

At maturity, VL-I should be able to receive a real model description and data source, produce a versioned Ground, accept human intentions, and run the original pretrained model's relevant behavior through VL-native interpretation and navigable transitions, preserving exact source identity, causal memory, authority, reproducibility and evidence.

The **breakthrough** sought is not simply "a model ran." It is an independently validated method by which navigating the model's representation **changes the efficiency or capability of realizing its trained computation**, possibly without the conventional per-layer/per-token sequence in qualified cases.

See [Architecture](ARCHITECTURE.md), [Source Inventory](SOURCE_INVENTORY.md) and [Experiment Protocol](EXPERIMENT_PROTOCOL.md).
