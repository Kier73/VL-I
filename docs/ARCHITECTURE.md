# VL-I architecture — the model as a navigable Ground

**Design status:** proposed VL-I composition contract, not yet a VL-I executable implementation. Reuse predecessor authorities and receipts rather than declaring a new one.

## 1. Position of VL-I

**Input:** a model source and an observer intention. **Output:** a qualified realization of the requested trained-model behavior, or an explicit unresolved/refusal state, with inspectable evidence.

VL-I's central boundary is unusual. The language model is *not* necessarily loaded into and invoked as an independent inference framework. It is represented as an addressable, versioned mathematical object whose operations and parameters are interpreted by the Virtual Layer. A physical observer-host runs the VL and the VL's qualified realizers. Whether this can be more efficient than a conventional model runner is the experimental question.

The underlying trained model remains unchanged unless an experiment explicitly admits a different snapshot or separately proves a contract-specific behavioral equivalence. Model size is not silently reduced to fit an interpreter's size.

## 2. Four identities that must never collapse

1. **Source identity:** repository/path, immutable revision, file identity and observed byte range; a captured screenshot has its own capture provenance and pixel scope.
2. **Architecture identity:** family, config, tokenizer contract, graph semantics, tensor schema, operator versions and unsupported components.
3. **Parameter-snapshot identity:** exact content-digest identities of the original weights, distinguished from storage chunks, range locations, and generated replacements.
4. **Execution identity:** observer intention, input bytes/token IDs, Ground revisions, realizer versions, precision, causal-state root, output commitments and authority receipt.

A config-derived schema can exist without parameter identity; a parameter identity can exist without verified interpretation; a successful transition can exist without proof of all possible future outputs.

## 3. The stages

~~~text
     EXTERNAL OBSERVER                    MODEL INFORMATION
     language / code / action             screenshots / files / Hub / streams
                 |                                      |
         intention contract                     evidence-scoped readers
                 |                                      |
                 +-----------------+--------------------+
                                   |
                                   v
                          virtual Ground graph
                     provenance / topology / weights
                     tokenizer / temporal causal state
                                   |
                      prediction and distinction engine
                       candidate descriptions & costs
                                   |
                         VL navigation coordinator
                      typed dependency and policy gates
                                   |
                   relation selection + qualified realizer
                   algebraic shortcut OR VL-native operator
                                   |
                   verified outputs / causal-state commit
                                   |
                           observer continuation
~~~

**A. Read / Observe.** Independent adapters acquire exact bytes or explicitly partial observations. Supported paths should eventually include:
- local immutable snapshots and host files, memory buffers, page stores;
- pinned Hugging Face model records (config, tokenizer assets, index, Safetensors headers and qualified payload ranges);
- image/screenshots of model pages, tensor visualizations, system observations or model outputs, retaining capture-time and visual interpretation uncertainty;
- E-CPU authenticated communication paths, content-addressed pages and model-host data.

Screenshots are not clandestine model snapshots: an image may justify a candidate configuration or visual property, but cannot prove the presence of unspecified tensors or model weights.

**B. Describe / Lift.** Construct model topology and tensor addressing using HFR NSD, typed neural IR, E-CPU HFModelGround and VGI ProofGround adapters where applicable. A description may be *structurally consistent* before it becomes *numerically executable*. Dependencies must retain order (embeddings, norm, QKV/attention, residual, MLP/MoE, LM head, sampling) and family-specific semantics.

**C. Bind actual data.** Resolve requested tensor coordinates to complete and verified original parameter bytes or a genuinely reproducible, content-verified generative law. MWS-001 and MWS-002 are strong foundations. Partial ranges may suffice for a specific qualified subgraph. They do not suffice for an unqualified full-model output.

**D. Interpret intention.** Bind prompt bytes, tokenizer and chat template identity, selected decoding mode, structured output needs, stop conditions, sampling seed, time/byte/compute budgets and authorized side-effect tools. Translation of intent may choose a shorter *path* but may not silently change the model's requested semantics.

**E. Navigate and realize.** VL-I chooses one of three explicitly marked paths:
- **VL-native exact/precision-declared relation:** execute typed model operators within the VL realizer using verified parameter ranges; compare to an independent oracle.
- **Evidence-backed navigable relation:** compose an admitted algebraic transformation, cached continuation, equivalent subgraph, or proven functional quotient that produces the same contract-observable result without replaying all original operations.
- **Reference/hosted comparison:** an external framework or model-server call, *only* for baseline or compatibility, never credited as a VL-only inference result.

**F. Continue.** Commit token/logit observations, causal memory, source/ground identities, exact operation scopes, and independently checkable receipts. Recover only from verified checkpoint roots and source data. Failed integrity, authority, precision or dependency checks return *unresolved*; they never invent outputs.

## 4. The Virtual Ground record (proposed fields)

~~~yaml
model_ground:
  schema: vl-i.model-ground.v0 # proposal; not yet implemented
  source:
    kind: local_snapshot | pinned_hub | captured_image | byte_stream
    identifier: ...
    immutable_revision: ...
    observed_scopes: [...]
    provenance: ...
  architecture:
    declared_family: ...
    graph_commitment: ...
    operator_semantics: ...
    tensor_inventory: ...
    unresolved_topology: [...]
  parameters:
    snapshot_commitment: ...        # absent for metadata-only evidence
    tensors: [...]                 # source coordinates and verified byte digests
    realizers: [stored, delta, admitted_structural_law]
  tokenizer:
    assets_and_digests: [...]
    byte_to_token_semantics: ...
    template: ...
  computation:
    input_contract: ...
    causal_state_contract: ...
    output_and_sampling_contract: ...
  verification:
    source_status: ...
    structural_status: ...
    numerical_status: ...
    behavioral_scope: ...
    proof_obligations: [...]
  navigation:
    admitted_relations: [...]
    candidate_descriptions: [...]
    measured_costs: [...]
~~~

Field names are design targets, not claims that existing repositories expose this exact schema. The adapter should preserve each upstream record's native digest and verification meaning.

## 5. The mathematical question

Let a model Ground be G with state space S, admissible transition relation T, observation map O, and intention contract C. A conventional decoder evaluates a sequence of operators to reach a new state.

VL-I seeks a *qualified* relation N_C: S → S' that reaches a state with the same permitted observation:

\[
O_C(N_C(s)) = O_C(T^{*}(s))
\]

where T* denotes the conventional model computation for the same input and declared numerical semantics. A candidate N_C is not admitted merely because a finite test set matches. Evidence may include exact symbolic proof, complete finite-state verification, or a precisely labeled empirical tolerance and input scope.

A predictive Ground can also learn an observer-relative quotient. Two states are equivalent under C only to the extent that no permitted future continuation distinguishes them. Testing finitely many continuations gives **finite-test evidence**, not automatic universal equivalence.

**The optimization target** is *validated information-realization efficiency*, not merely number of generated state labels. For selected contract C, compare output accuracy/coverage, distinct meaningful states reached, source bytes required, realizer work, end-to-end latency, memory, network traffic, and recovery cost. A candidate cannot buy an apparently superior score by refusing most inputs or silently changing precision.

## 6. Information classification and active attention

For each model component or transition, record:

- **Known deterministic:** a verified mapping (e.g. a tensor address, a certified affine transform, or a complete deterministic operator under fixed input/precision).
- **Predictively supported:** an observed pattern with specified held-out performance and uncertainty.
- **Condition-dependent / stochastic:** sampling, adaptive routing, or source behavior requiring additional context or seed.
- **Unknown or missing:** unobserved weights, ambiguous visual interpretation, unsupported operator or unavailable host data.
- **Distinction-preserving or distinction-losing:** defined relative to specific observer projections, not an absolute claim of information creation or annihilation.

An active-attention planner should choose the next byte range, test input, or output observation that most reduces uncertainty *per measured cost*. This is inspired by VGI partial inquiry and IFS observation-equivalence work. The planner must not turn a guess into an admitted fact.

## 7. Model operational semantics

A complete decoder requires explicit coverage of:
- tokenizer and chat-template version; byte-level reversible or declared-lossy text mapping;
- embeddings / positional representation; true family-specific attention, routing, recurrence, activation and normalization;
- model precision and tensor decoding, including quantization companions when model data actually uses them;
- prompt prefill, causal masking, incremental decode, KV or recurrent state, truncation and stopping;
- logits and selection/sampling; correct deterministic seed, stable token ID and output reconstruction;
- per-session isolation, checkpoint roots, state movement and exact-once side-effect authorization.

These are computation obligations. An observed header or a complete typed DAG does not discharge them.

## 8. Relation to the constrained VL-64 core

The VL-I research engine is **unconstrained in source size** so it can host readers, proof search, numerical experiments, precision analysis, benchmarking and federated integration. Successful concepts may be ported to VL-64 **only after** preserving its 65,536-byte source budget.

A small interpreter is compatible with large model data: the source-code budget is not the storage, memory, transferred byte or operation budget for weights. The ordinary host may remain a thin observer/realizer, but physical arithmetic, I/O and energy must still be accounted for.

## 9. Fail-closed guarantees

- A source or snapshot revision mismatch invalidates attached observations until requalified.
- Digest equality of a header is not digest equality of its weight payload.
- A structural graph check is not a numerical execution proof.
- A tokenizer mismatch is not a harmless formatting change.
- A realized partial layer is not full-model generation.
- An opaque inference endpoint is not a native VL-only result.
- A change in model weights/precision changes the semantic contract unless equivalence is independently established.
- Model-emitted commands have **no authority** without explicit caller-supplied tools and receiver policy.
- An unavailable host route, incomplete evidence or uncertain observation yields refusal / unresolved evidence, not fabricated success.

## 10. Engineering first step

Build **one adapter**, not another core. Given an already pinned and experimentally observed pretrained decoder from VL-64 / IFS:

1. Import its exact model inventory and typed operator graph.
2. Expose source/parameter identity and an independently verifiable Ground receipt.
3. Execute one full forward pass through **VL-native** typed operations without using external model-forward code in that path.
4. Compare every checkpointed layer and final logits/token selection to a separate reference.
5. Demonstrate a saved/recovered continuation and record full cost accounting.
6. Propose the first alternate Ground only after establishing reproducible parity.

See [Research Plan](RESEARCH_PLAN.md) and [Experiment Protocol](EXPERIMENT_PROTOCOL.md).
