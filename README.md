# VL-I — Virtual Layer Intelligence

**Lift the description of a language model into a Virtual Layer. Navigate its model data as a computational state space. Realize useful language-model behavior through the Virtual Layer itself.**

**Status: research charter and design programme.** This repository begins with an ambitious target, not a claim that the target has already been achieved.

## The endeavor

A Virtual Layer observes bytes, constructs an interpretation of their variables and transformations, and expresses them as a **Virtual Ground**: a versioned, testable description of possible states, transitions, observations, and intentions.

Rather than merely imitating a machine, the Layer seeks descriptions that make desired states *more addressable, predictable, reconstructable, and inexpensive to realize*. It can discover stable relations, choose what to observe next, retain structural memory, distinguish genuine laws from coincidences, and navigate qualified transitions.

VL-I applies that endeavor to **large language models**.

> **Research goal:** Read a model's data and architecture; lift its tokenizer, parameters, operations, causal memory, and output behavior into a Virtual Ground; then realize its trained functionality by navigating that Ground. In the intended architecture, the only general-purpose programme a CPU or other observer-host needs to run is the Virtual Layer interpreter. The model does not run as a separately invoked model runtime.

The premise is not that model weights can be guessed from metadata, that neural mathematics disappears, or that a file header produces tokens. The research is whether increasingly powerful descriptions and verified navigation relations can **replace enough of conventional model execution** to make the virtual realization measurably more effective.

**Anything expressible in bytes can enter as an observation.** What those bytes mean, and whether they suffice to realize a model, must be established through evidence.

## The central research question

Given an actual model snapshot, an intention, and an observer contract:

**Can VL-I discover or compose a predictive representation of the model that realizes the requested behavior with less interpretation work, less data movement, better continuity, or greater portability than executing the conventional model graph, without losing the trained functionality claimed by the contract?**

This includes the more radical possibility of navigating directly to selected intermediate or terminal model states using certified relations rather than evaluating every conventional operator. It is a hypothesis to investigate, not an assumed universal shortcut.

### What "only the Virtual Layer is computed" means

- The **model** is a pinned data-and-law Ground: architecture, exact parameters or validated alternative parameter realizations, tokenizer, attention and recurrence rules, KV or other causal state, sampling and stopping semantics.
- The **Virtual Layer** receives external intention, selects and composes certified transitions, realizes numerical or symbolic relations where required, and verifies and commits results. Its realizer may use a host's ordinary instructions; those instructions count as VL work and must be measured.
- **No opaque model server, hidden Transformers forward pass, or outsourced model inference** may be used on the experimental *VL-only* path. Such implementations are welcome as independent reference baselines and must be labeled separately.
- A reference implementation, a VL-mediated host-model run, a VL-native numerical model realization, and a *navigation-shortcut* realization are four different evidence tiers. Do not interchange their claims.

The physical host still performs the operations required to interpret and realize the Layer. Substrate independence means a common **semantic contract across qualified hosts**, not an unmeasured assertion that arbitrary physical resources are unnecessary.

## The model-to-intention path

~~~text
Observed source
  local model files | pinned Hugging Face snapshot | bytes/ranges
  verified captures | image or screenshot (partial observations)
             |
             v
Observation and provenance
  exact scope, hashes, source revision, format and confidence
             |
             v
Model Ground construction
  architecture + typed operator relations + tensor coordinates
  trained parameter realization + tokenizer + causal state
             |
             v
Predictive description selection
  exact laws | compositional relations | verified abstractions
  unresolved/nonpredictable regions preserved explicitly
             |
             v
External intention
  conversation | paragraph | code | tool proposal | typed result
             |
             v
VL intention interpretation and navigation
  choose qualified path, materialize only needed information
  realize missing operations in the VL, retain causal memory
             |
             v
Verified continuation
  tokens / logits / text / structured tool decision
  evidence, uncertainty, costs, recoverable checkpoint
~~~

A screenshot may reveal a tensor name, a configuration value, a tokenizer setting, or a published model result. It **cannot certify unseen weight bits**. Source scope is always part of the Ground; a metadata-only Ground is not silently promoted into runnable full-model inference.

## Description, prediction, and information

We seek descriptions of a state space that preserve **the greatest useful capacity to realize distinct behavior under an intention**, not the most states enumerated for their own sake.

Candidate Grounds are compared on:

1. **Expressiveness:** which inputs, outputs, states, and possible continuations are represented?
2. **Predictive validity:** what portion of independent held-out observations is correctly predicted? What remains unknown, contradictory, or stochastic?
3. **Distinguishability:** which apparently different states are equivalent for the requested observations, and which distinctions matter under future continuation?
4. **Information effects:** which transitions create new observable distinctions, preserve information, or lose distinctions under a declared projection? Measure conditional entropy and prediction surprise rather than treating "information creation" as unbounded entropy from nowhere.
5. **Realization cost:** source bytes, retained parameter bytes, arithmetic/interpretation work, wall time, memory, energy where measurable, and network transfer.
6. **Reuse and recovery:** whether a description, cache, or certified transition survives movement between observer-hosts without repeating an already committed semantic effect.

A concise description that fits the training trace but fails a held-out continuation is not a better Ground. A compact hash with no means to recover unpredictable parameters is not model compression.

## What we inherit — do not rebuild it blindly

VL-I is a **focused integration and research environment**. It reuses existing work and preserves its provenance rather than duplicating whole runtimes.

| Source | Capabilities to investigate and compose |
| --- | --- |
| [VL-64](https://github.com/Kier73/VL-64) | 65,536-byte source-budget interpreter; generic byte Grounds, observer intention, typed pretrained DAGs, bounded Safetensors ranges, 30-layer pretrained decoder experiment, cross-process recovery, model-instance orchestration |
| [Virtual-Layer-Dynamics](https://github.com/Kier73/Virtual-Layer-Dynamics) | Consolidated Ground/navigation mechanics, numerical model realizers, sparse MoE/frontier operations, precision receipts, model arbitration and hosted source glances |
| [E-CPU](https://github.com/Kier73/E-CPU) | Structural Memory and predictive selection, model-weight snapshots and generative Grounds, HF model/layer Grounds, Hypervisor authority, VirtualLayerRuntime, execution DAG, communication fabric, durable causal state |
| [In-Formation-Space](https://github.com/Kier73/In-Formation-Space) | HFR neural-structure and Safetensors reader, typed graph IR, independent verification, neural observation experiments, pretrained crash continuity and model-to-model comparisons |
| [Virtual-Ground-Interpreter](https://github.com/Kier73/Virtual-Ground-Interpreter) | Three-valued observation, partial-attention navigation, finite-state and algebraic methods, structural-memory work, AND/OR ProofGround obligations for neural topologies |
| [E-CPU Virtual Processor](https://github.com/Kier73/E-CPU-Virtual-Processor) | Portable observer-host identity, a grounded operational interface and recovery-ledger precedent |

Specific implementation paths and what each does **and does not establish** are in [Source Inventory](docs/SOURCE_INVENTORY.md).

## Research trajectory

**First demonstrate one end-to-end, independently reproducible path.** Do not multiply architectures, build parallel authorities, or introduce a new model framework before that path has an honest comparison.

- **R0 — Grounded observation:** pinned source + model config + tokenizer identity + complete tensor inventory; exact observed-byte scope and explicit unknowns.
- **R1 — Lift:** canonical, typed Model Ground with parameter addresses, operations, causal dependencies and proof obligations; compare two independent interpretations.
- **R2 — VL-native reference parity:** verified full pretrained forward pass, logits and token IDs through VL-realized operations, with no hidden external model execution; compare to a separate standard-model reference.
- **R3 — Generative continuity:** prefill, incremental decode, KV/causal-state reuse, multi-token text and code, cancellation, checkpoint/restart and host change.
- **R4 — Predictive navigation:** learn and validate alternate laws, quotients, partial realization and directly navigable subgraphs. Measure exact saved work and provide counterexamples.
- **R5 — Scale and generality:** retain full-scale pretrained model identity and weights; investigate dense, MoE, hybrid attention and multimodal variants without obligatory model downscaling.
- **R6 — Many intentions and instances:** isolated model Grounds, multiple VL instances, typed outputs, authorized tool actions and throughput/recovery across observer-hosts.
- **R7 — Portable realization:** characterize what can live in the 64 KiB VL-64 interpreter and what belongs to external data, adapters, pages or host capabilities.

Full dependencies, deliverables and test gates: [Research Plan](docs/RESEARCH_PLAN.md).

**VL-I is not constrained to 64 KiB.** VL-64 remains the constrained companion and eventual target for proven minimal relations. VL-I's model data is not to be silently quantized, truncated, approximated or scaled down to satisfy that budget.

## How to begin

**First implementation task:** [VL-I-001 — Lift actual pretrained model data and verify VL-native full-model generation](https://github.com/Kier73/VL-I/issues/1). Its checklist begins with source identity and independent parity, then advances to multi-token generation, recovery and the first genuine shortcut experiment.

Start with the [Architecture](docs/ARCHITECTURE.md), [Source Inventory](docs/SOURCE_INVENTORY.md) and [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md). The first experiment should reuse an already pinned, observed, full-layer decoder case, retain exact pretrained weights, and compare the same input under:

1. conventional independently implemented model execution (reference);
2. VL-native realization of the full model graph;
3. a Ground-selected alternative transition or reuse path **only when independently qualified**.

Record precise numerical and token-level equivalence, data-source completeness, wall-clock time, memory, I/O, bytes moved, work performed, and what was skipped. Then repeat for multiple tokens, architectures, and host transitions.

The first successful result is not "the VL knows the model." It is a checkable **observed model → admitted description → intention → VL realization → verified model output** chain, with measurable cost and counterexamples.

## Research principles

**Observe before interpreting. Prove or measure before promoting. Preserve exact identities. Distinguish intended ability from observed result. Navigate before iterating when qualified. Never fabricate missing model information.**

This is an open research endeavor. Strong negative results—proof that a supposed shortcut fails, that a model's behavior is not determined by accessible bytes, or that a proposed quotient loses necessary distinctions—are valuable findings. The goal is to build a Virtual Layer that becomes progressively more capable by learning which descriptions actually work.

**Repository state:** charter and documentation first; VL-I-native executable claims begin only when corresponding code, test cases and evidence are added.
