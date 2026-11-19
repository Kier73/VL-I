# VL-I architecture — the model as a navigable Ground

**Design status:** architectural research programme. Parts of the source-snapshot integration are implemented; the description-native architecture is a new, higher-ambition research direction. Reuse predecessor authority/evidence components while allowing entirely new computational forms.

## 1. Position of VL-I

**Input:** any model description (including bytes, header, image, metadata or full snapshot), an observer's intention and optional memory. **Output:** an operational Ground capable of continuing a conversation, deriving representations, generating tokens, navigating state or answering the intention; report observed behavior, provenance and evidence. Exact reproduction of source-trained behavior is an independent *optional* objective.

The central proposition is **description is operation**. A language model can be understood as a family of mathematical states and relations, constructed or inferred from any available description and interpreted by the Virtual Layer. The architecture should pursue meaningful behavior even before full source weights, a tokenizer or a conventional neural forward graph is available. The deeper aim is to replace inherited model execution with novel generative, predictive and navigational mathematics. A capable observer-host runs the VL interpreter; which mathematical relations should exist inside that interpreter is deliberately open for research.

**Two independent outputs are legitimate:** (1) reproducing the trained source's exact or contract-specific behavior while preserving its identity; and (2) creating a *new capable model Ground* from the same description. Do not silently relabel the second as the first. Neither path needs to impose an artificially small architecture or model size.

## 1A. Description-native operational path (the desired architecture)

~~~text
       ANY MODEL DESCRIPTION
   header | metadata | prose | code | image | screenshot | weights
                      |
           interpreted into bytes
                      |
            generative Ground
        laws / possible parameters / states
        structural memory / uncertainties
                      |
        external observer intention
          dialogue / code / action
                      |
           VL Ground navigation
     compose / synthesize / predict / observe
                      |
       output tokens, model behavior,
       or a qualified request for more evidence
                      |
         retain context and revise Ground
~~~

This path is a **first-class target**, not a fallback after failed pretrained execution. The header or model description participates directly as a computational input. It can seed an interpreter, suggest mathematical laws, generate an alternative parameter model, or guide a new operational Ground without downloading any source-trained weights. A description-native output must not masquerade as an exact pretrained checkpoint result unless that equivalence is actually established.

See [Description Is Operation](DESCRIPTION_IS_OPERATION.md) for an independent set of experimental hypotheses.

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

**B. Describe / Lift.** Translate observed bytes into one or more candidate *operational* Grounds. A Ground may be conventional model topology (HFR/NSD, E-CPU, VGI) or a new algebra, generative parameter law, symbolic token state, time-navigation relation, or description-native conversational machine. Preserve source evidence and testable assumptions; do not require a conventional layer graph before experimentation begins.

**C. Realize the model's capacity.** Discover which information is needed for the selected intention. The Ground may interpret header fields directly, synthesize novel parameters, instantiate an algebraic law, reuse retained states or read exact source-trained tensors. MWS-001/002 provide excellent exact-snapshot mechanisms, but they are not prerequisites for a separate **description-generated model**. Exact source fidelity is checked only when that is the claim.

**D. Interpret intention.** Bind prompt bytes, tokenizer and chat template identity, selected decoding mode, structured output needs, stop conditions, sampling seed, time/byte/compute budgets and authorized side-effect tools. Translation of intent may choose a shorter *path* but may not silently change the model's requested semantics.

**E. Navigate and realize.** VL-I can take distinct explicitly marked paths:
- **Description-native generative relation:** interpret available metadata, header, image or other description, form an operative Ground and produce context-dependent tokens, relations, parameters or actions—without claiming access to unseen source weights.
- **Constructive model synthesis:** generate a new mathematical model/parameter Ground from descriptions and test its abilities independently.
- **Novel neural mathematics:** select a nonstandard algebra, predictive quotient, time/state navigation law, structural memory composition or other discovered interpreter relation.
- **VL-native faithful realization:** execute reference-equivalent pretrained model operations under the VL interpreter, with admitted source parameters and independent numerical comparison.
- **Reference/hosted comparison:** a conventional model implementation remains a baseline for source-equivalence tests, never secretly substituted for the VL-only experimental path.

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
    realizers: [stored, delta, admitted_structural_law, synthesized_parameters, description_native_relation]
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

## 4A. Ground contracts must permit new generated models

A *source-fidelity* record and a *description-native* record are both valid but answer different questions.

- **Source-fidelity record:** exact source/revision/parameter identity, declared comparison target, numerical or behavioral equivalence evidence.
- **Generated-model record:** description identity, interpreter/generative-law identities, constructed parameters or rules, initial context/memory, output capability measured on independent requests, and the relationship (if any) to an external trained model.
- **Navigation-law record:** the observer projection, conditions for use, held-out distinguishing evidence, cost and behavior. It need not retain the original conventional operator sequence when the desired behavior is independently supported.

A header producing tokens is therefore an admissible **generated-model experiment** immediately; it only becomes a claim of **original-model inference** upon satisfying the separately declared source-fidelity contract.

## 5. The mathematical question

Let a model Ground be G with state space S, admissible transition relation T, observation map O, and intention contract C. A conventional decoder evaluates a sequence of operators to reach a new state.

VL-I seeks a *qualified* relation N_C: S → S' that reaches a state with the same permitted observation:

\[
O_C(N_C(s)) = O_C(T^{*}(s))
\]

where T* denotes one **optional** conventional reference computation for an experiment aiming at parity. This equality is not the governing criterion for all VL-I work: a description-generated Ground may define *new* meaningful transitions, and must be evaluated against its own observer intention and independent behavioral tests. For exact pretrained claims, retain independent reference checks and their declared evidence scope.

A predictive Ground can also learn an observer-relative quotient. Two states are equivalent under C only to the extent that no permitted future continuation distinguishes them. Testing finitely many continuations gives **finite-test evidence**, not automatic universal equivalence.

**The optimization target** is *validated information-realization efficiency*, not merely number of generated state labels. For selected contract C, compare output accuracy/coverage, distinct meaningful states reached, source bytes required, realizer work, end-to-end latency, memory, network traffic, and recovery cost. A candidate cannot buy an apparently superior score by refusing most inputs or silently changing precision.

## 6. Information classification and active attention

For each model component or transition, record:

- **Known deterministic:** a verified mapping (e.g. a tensor address, a certified affine transform, or a complete deterministic operator under fixed input/precision).
- **Predictively supported:** an observed pattern with specified held-out performance and uncertainty.
- **Condition-dependent / stochastic:** sampling, adaptive routing, or source behavior requiring additional context or seed.
- **Unknown or missing for a source-equivalence claim:** unobserved checkpoint weights, ambiguous visual interpretation, unsupported conventional operator or unavailable host data. These may motivate generative alternatives rather than prevent exploration.
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
