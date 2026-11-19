# VL-I research plan — from observed model data to Virtual Layer Intelligence

**Plan version:** 0.2 — 2026-11-19  
**Nature:** an executable research programme, not a claim that the future milestones have passed.  
**Core hypothesis:** A model's **description itself** can be an operational and generative Virtual Ground. Interpreting and navigating that Ground can produce meaningful tokens, dialogue, programs, models and actions **without conventional neural-model execution**. This includes deriving or generating model weights from metadata, inventing alternate neural mathematics and discovering what functionality can emerge from even a file header. Faithful reproduction of original trained-model behavior is an additional research objective, not a gate on description-native intelligence.

The goal is not simply a smaller model or faster kernel. It is to **replace conventional model execution** by discovering how model descriptions, generative algebraic laws, structure, observer context and intention can operate as the model. A conventional pretrained implementation remains a useful comparator when relevant, never the limit of which experiments are allowed.

## Research charter

A development increment is useful if it advances at least one of these research capacities:

1. **Description-native operation:** a header, metadata record, image, architecture description or arbitrary byte representation becomes an active Ground that can respond to intention.
2. **Generative model synthesis:** alternative weights, token relations, state laws or architectures are constructed from a description and evaluated as operational systems.
3. **New computational mathematics:** reduce or replace dependency on inherited layer-by-layer neural execution through new predictive, compositional or algebraic relations.
4. **Model addressability and source fidelity:** actual pretrained snapshots can also be observed, reconstructed and compared where exact behavior is the research claim.
5. **Continuity, generality and evidence:** the Ground can support changing requests, dialogue, code, actions, recovery and additional observer-hosts with measurable work and explicit uncertainty.

Every increment must have a reproducible demonstration or explicit negative/control result.

## Working definitions

- **Observation** — specific acquired bytes, pixels or reported behavior, with provenance and exact coverage.
- **Description / Virtual Ground** — a versioned interpretation mapping observations into parameters, states, relations and possible transitions.
- **Intent** — an observer-specified result with input, output, authority, cost and fidelity constraints.
- **Realization** — a concrete result obtained by an authorized VL operation, an admitted relation or a baseline backend, with a declared path type.
- **Verified** — only what independent tests, proof obligations or measurement scopes actually establish.
- **Predictable / unpredictable** — relative to a given context and observer, with observed uncertainty and missing information explicitly tracked.

A Ground is judged by the **capability it realizes under intention**, its predictive value, the distinctions it preserves and its measured resources. The goal is to discover the most productive description, not merely the most faithful recitation of today's model operators.

## Immediate revolutionary programme — parallel to trained-snapshot integration

**Description is not a passive prerequisite.** Investigate the following experiments *now*; conventional model parity is not a prerequisite. Each outcome must be reported under its own contract and evaluated on held-out intentions.

### DNO-001 — Run a model description as an interactive Ground

Start with model config, header, metadata, model card or screenshot alone. Construct a Virtual Ground that can accept observer intentions, produce tokens and context-sensitive responses, retain conversational state and revise its interpretation. Generalize beyond answering static configuration questions. Use a declared generative law and log what information the input description, interpreter, context and memory each contribute.

**Test:** independent unseen prompts, different descriptions, ablations, reproducible state transitions and comparison to null/template baselines. This is not automatically reproduction of the original model's trained behavior.

### DNO-002 — Model weights and functions from metadata

Given a header/config with zero parameter-payload bytes, search for a procedural law that generates functional parameters or equivalent operations. Test both **exact checkpoint reconstruction** (when source data are available to validate) and **new model formation** (measured by behavior on independent tasks).

**Test:** validity of synthesized parameter data or functions, output-quality changes under controlled metadata edits, measured extra information consumed, relation to any observed pretrained snapshot and counterexamples.

### DNO-003 — Redefine inference mathematics

Find new models for operations normally expressed as learned matrix/tensor transformations: predictive quotient states, relational composition, symbolic/temporal navigation, generative parameter Grounds, exact or approximate functional equivalence, and entirely novel schemes. Avoid treating the inherited Transformer graph as the definition of computation.

**Test:** held-out tasks and full accounting of interpretation, observation, storage, verification and actual computation; report which capabilities emerge and where conventional operations were eliminated or reexpressed.

### DNO-004 — Description-native conversational intelligence

Aim for a multi-turn VL-I interaction: a human asks, changes goals, requests paragraphs and code, and authorizes a bounded action. The Virtual Layer interprets a model description and navigates its formed Ground to produce contextually useful continuations, persisting memory without running a conventional model backend.

**Test:** long-horizon continuity, instruction changes, code checks, constrained tool decisions and independent capability comparison. A templated or random token stream is not sufficient.

See [Description Is Operation](DESCRIPTION_IS_OPERATION.md) for the mathematical research thesis and [Experiment Protocol](EXPERIMENT_PROTOCOL.md) for the separate evidence contracts.

## Complementary pretrained-equivalence programme

The following established phases continue as an *independent* experimental track. They are powerful references and reuse sources, **not prerequisites for DNO-001..004**.


### Phase A — Establish an independently reproducible model intake

**Candidate:** pinned pretrained SmolLM2-135M checkpoint already used by the [VL-64 complete decoder experiment](https://github.com/Kier73/VL-64/blob/main/docs/REAL_FULL_PRETRAINED_DECODER_OBSERVATION.md). This is an initial validation fixture, **not** a cap on eventual model size.

**Implementation actions:**
- Pin exact upstream revisions in a lock manifest: VL-64, E-CPU, IFS, VGI, VLD; record source license and permitted access.
- Import model config, tokenizer files, Safetensors inventory and graph; preserve distinctions between actual bytes, header-only references and inferred semantics.
- Build an adapter around existing HFR NSD and HFG/weight Ground contracts instead of copying model internals.
- Define a model-independent transcript for one observational input and one decoded continuation.

**Pass gate for source fidelity:** a verifiable model identity, typed graph, exact tensor-to-byte addressing and audited provenance. Missing source bytes refuse an exact checkpoint claim; they do **not** prohibit creating and evaluating a new generative Ground from the description.

**Output:** model Ground manifest, observations, adapter tests and independent verifier.

### Phase B — VL-native complete pretrained forward pass

**Implementation actions:**
- Reuse VL-64 qualified range readers and VLD numerical realizer ideas under a single VL-I interface.
- Express embeddings, all decoder blocks, norm, final head and next-token selection as typed VL relations.
- Make the *experimental VL path* call only VL-native operators. Do not use Transformers/PyTorch model.forward or a hosted model API there. They remain isolated baselines.
- Run both direct numerical and independently implemented reference computations on the same pinned weight bits and input IDs.

**Pass gate for trained-source equivalence:** all layers and final logits independently compared under declared precision/tolerance, or a precisely declared alternative functional-equivalence contract. Missing original weights refuse a claim of exact source parity; novel generated models remain eligible for their own behavioral experiments. Count actual work, bytes read and memory used.

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

### Phase E — Predictive description formation and a navigation advantage

This track tests alternative laws against a particular trained checkpoint *after parity is available for that comparison*. The broader description-native and new-mathematics research above does not need to wait for numerical parity.

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

## The first research experiments

**DNO-001: description-native interaction** starts directly from metadata/header bytes and intention, seeking operative model behavior without original weight-payload access. In parallel, **VL-I-001: observed-data full-model Ground parity** remains a reference-equivalence track.

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

Each experiment is complete only when actual results and executable checks are published. Predecessor results provide valuable evidence and machinery, not automatic new VL-I pass status. **Do not gate successful DNO-001 outcomes on matching the original pretrained-model logits; compare the claims under the correct contract.**

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

A claimed gain should state the reference, task, input, interpretation, host and all included costs. A conventional backend may be used as a separately metered comparator, but its output must not be silently credited to a description-native VL. New generated functionality must be measured on independent tasks rather than dismissed for not matching a pretrained checkpoint.

## Success criteria

At maturity, VL-I should be able to receive a real model description and data source, produce a versioned Ground, accept human intentions, and run the original pretrained model's relevant behavior through VL-native interpretation and navigable transitions, preserving exact source identity, causal memory, authority, reproducibility and evidence.

The **breakthrough** sought is a Virtual Layer in which **operating a description** provides interactive intelligence: metadata can form useful mathematical models, headers can seed token-producing Grounds, and novel interpretation/navigation can replace conventional model inference. The benchmark is meaningful observer-directed functionality, with exact pretrained equivalence tested separately when claimed.

See [Architecture](ARCHITECTURE.md), [Source Inventory](SOURCE_INVENTORY.md) and [Experiment Protocol](EXPERIMENT_PROTOCOL.md).
