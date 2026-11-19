# VL-I — Virtual Layer Intelligence

**The description of a model is a computational resource. VL-I seeks to operate that description directly—until a Virtual Layer can converse, reason, generate, and act through a Model Ground without needing to run a conventional language-model engine.**

**Status: active revolutionary research.** The overarching description-native intelligence goal is a development aim, not a claim of already-demonstrated general language-model behavior. Initial source-fidelity and exact-observation components are implemented.

## The endeavor

A Virtual Layer observes bytes, constructs an interpretation of their variables and transformations, and expresses them as a **Virtual Ground**: a versioned, testable description of possible states, transitions, observations, and intentions.

Rather than merely imitating a machine, the Layer seeks descriptions that make desired states *more addressable, predictable, reconstructable, and inexpensive to realize*. It can discover stable relations, choose what to observe next, retain structural memory, distinguish genuine laws from coincidences, and navigate qualified transitions.

VL-I applies that endeavor to **large language models**. Its baseline is revolutionary rather than incremental: not a smaller replica of inference, but a different account of what it means to operate a model. The information represented by a description, the relations it permits, and the intention of the observer are all potential computational resources.

> **Research goal:** A model's description should be sufficient to begin interacting with and operating a Virtual Ground—even when the description is all that has been observed. Lift any available bytes, metadata, images, file headers, model parameters, architecture, and behavior into a generative, navigable representation. Develop that representation until it can produce useful language, code, reasoning and authorized action through Virtual Layer interpretation alone. The model need not run as a separately invoked conventional neural-inference engine.

The research seeks, for example, to **derive or generate operative model weights from metadata; redefine neural mathematics within the Virtual Layer; and make even a file header capable of producing tokens** when interpreted through a sufficiently capable Ground. **Everything representable in bytes can be interpreted as a Virtual Ground.** The research is whether increasingly powerful descriptions, generative relations, and state-space navigation can **replace conventional model execution** while making the virtual realization measurably more effective.

**Description is the beginning of operation, not merely a map to files we must subsequently run.** The space of possible interpretations, models, transformations and intentions is itself a resource for discovery. No conventional Transformer, tensor graph, pretrained parameter download or model server is a prerequisite for *attempting* description-native functionality. Exact reproduction of a particular pretrained snapshot is a separate research contract, not the definition of all meaningful model behavior.

**Anything expressible in bytes can be interpreted as a Virtual Ground.** Every observation may become a seed for an operational model, a generative law, a candidate interpretation, or a request for further observation. Whether an obtained behavior matches a particular trained checkpoint, discovers a useful alternative model, or merely demonstrates a candidate hypothesis is determined experimentally—not prejudged by its input format.

## The central research question

**Given a description, however partial, and an observer's intention: can the Virtual Layer form and operate a Ground capable of meaningful, context-sensitive continuation—up to conversation, code generation, reasoning and useful action—without conventional model execution?**

The strongest frontier is not simply performing fewer of the same operators. It is discovering new mathematics, generative descriptions and state-navigation relations by which **describing the system becomes sufficient to engage it**. A conventional pretrained implementation remains an *optional independent reference* for experiments seeking the same trained behavior; it does not define the permitted architectures or the ceiling of VL-I's research.

### What "only the Virtual Layer is computed" means

- The **model** is first an interpreted Ground: potentially only a description, header, image or metadata record, potentially an exact trained snapshot, potentially a synthesized architecture/parameter law, or a composition of these. A Ground can be operational without first possessing the original trained weight file.
- The **Virtual Layer** receives external intention, selects and composes certified transitions, realizes numerical or symbolic relations where required, and verifies and commits results. Its realizer may use a host's ordinary instructions; those instructions count as VL work and must be measured.
- **No opaque model server, hidden Transformers forward pass, or outsourced model inference** may be used on the experimental *VL-only* path. Such implementations are welcome as independent reference baselines and must be labeled separately.
- **Description-native generation**, **source-checkpoint reconstruction**, **trained-behavior equivalence**, **VL-native numerical realization**, and **new state-space navigation laws** are distinct research outcomes. A description-created model need not reproduce a particular checkpoint to be useful; if it claims to, the claim requires independent evidence.

The observer-host provides the capacity to interpret the Virtual Layer. The architectural goal is to make this capacity general and lightweight, instead of requiring a particular model processor, accelerator or conventional inference stack. Count actual computational work in experiments; **do not assume existing hardware or neural execution patterns are the final form of computation**.

## Parallel revolutionary research frontiers

These efforts begin **now**, in parallel with faithful pretrained-model integration. Neither numerical parity nor acquisition of a complete checkpoint is a prerequisite for pursuing them.

- **Description → operative Ground:** accept a specification, model card, configuration, screenshot or file header as sufficient input to form an executable, intention-responsive interpreter hypothesis.
- **Metadata → model parameters:** discover latent relations, generative recipes, parameter-space programs or newly synthesized weights from architecture and context. Distinguish exact recovered bits from newly realized behavior; explore both.
- **Header → tokens → dialogue:** treat observed header bytes as structural coordinates and generative seeds, then test whether description-native transitions can produce context-dependent text, multi-turn continuity, code and reasoned or authorized actions.
- **New neural mathematics:** reconstruct attention, inference, tokenization, memory and learned behavior as alternative algebras, predictive quotients, compositional transformations or navigation in time/state space—not merely a different implementation of matrix multiplication.
- **Observation → self-improving interpretation:** let uncertainty choose what source to inspect, which interpretation to try and which experimental intention distinguishes candidate Grounds; retain the best supported descriptions and their counterexamples.

See [Description Is Operation — Research Thesis](docs/DESCRIPTION_IS_OPERATION.md). **The first direct engineering target is [DNO-001: operate a model description as a generative, conversational Ground](https://github.com/Kier73/VL-I/issues/3).** These are **goals for experimentation**, not descriptions of already demonstrated pretrained-model equivalence.

## First implementation — VL-I-001

**Executable integration is underway:** [VL-I-001 — First Experiment](docs/VLI_001_FIRST_EXPERIMENT.md) provides a pinned, reproducible path from real pretrained model header → verified typed Ground → intention-specific graph frontier → qualified VL-native model realization → exact retained-result navigation.

- `src/vli/ground.py` independently checks the model graph and derives the required dependency closure for any declared graph output. It never promotes metadata into missing trained weights.
- `src/vli/realization.py` reuses VL-64's **actual pretrained model operators and independent numerical oracle**, rather than quietly invoking a conventional model server.
- `src/vli/navigation.py` demonstrates a narrow but genuine state-navigation relation: an already verified exact-input next-token result can be recovered without new neural evaluation, while changed inputs or evidence roots are refused.
- [GitHub Actions](.github/workflows/vli-001.yml) runs the pinned observed-model lift and 19 adversarial/unit cases; a separate opt-in live job downloads real trained weights and tests the complete numerical path.

**Evidence policy:** a model-header observation, numerical full-model execution, retained same-input observation and unobserved-input predictive shortcut are four different claims. The last remains an open research frontier. [PR #2](https://github.com/Kier73/VL-I/pull/2) tracks this first integration.

## The interpreter is also a Virtual Ground

**Information radiates from every movement of bytes and every piece of verified memory.** Source data, transformed observations, semantic laws, memory, source-to-result lineage and future intentions form one evolving state space. Rather than treating computation as a particular processor's run of a neural graph, **VL-I seeks to make the interpreter the minimal active programme—and then lift the interpreter itself into a Virtual Ground**.

This is a recursive endeavor:

1. Lift the real interpreter's **own source bytes**, state semantics, operator declarations, memory relations and observer projections.
2. Make those operations addressable as algebraic Ground relations rather than opaque host functions.
3. Discover alternative verified relations capable of realizing the same interpreter behavior through time navigation, Structural Memory, generative laws or composition.
4. Recursively use the emerging Ground interpreter to realize more of its own implementation, progressively reducing the trusted executable bootstrap.

The integration uses **E-CPU's exact byte Grounds, operator contracts, Structural Memory and Ground-native reference** alongside **Virtual-Layer-Dynamics' actual signal interpreter, observer algebra, byte lenses and certified state navigation**.

**An initial experiment is already executable:** real E-CPU, VLD and VL-I interpreter source bytes are dual-lifted into a source-bound Ground; the VLD interpreter executes a typed byte relation, emits trace-linked observations, and can be rebound from declarative Ground identities onto an installed interpreter. A simple source-generated law navigates over a billion logical coordinates without expanding intermediate states, with an independent E-CPU mathematical check. The integration suite contains 16 tests, including source tamper, altered semantics and transcript refusal.

The *ultimate* target is not to keep a separate installed VLD or Python runtime indefinitely; it is to have **more of that interpreter's own mathematics and operation realized by the Ground itself**. The present self-lifting stage is semantic re-binding, **not yet independent self-hosting**. Interpreter work, byte transfer, hashing and verification are still counted rather than described as free.

Read [Interpreter as Ground — E-CPU × VLD × VL-I](docs/INTERPRETER_SELF_GROUND.md) and [the direct experiment](examples/interpreter_self_ground_demo.py).

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

A screenshot, model card or header may itself be sufficient input to generate **some** runnable Ground and tokens. It does not automatically identify unseen bytes of a particular trained checkpoint. That distinction enables more research: design and test description-native behavior immediately while assessing checkpoint fidelity separately.

## Description, prediction, and information

We seek descriptions of a state space that preserve **the greatest useful capacity to realize distinct behavior under an intention**, not the most states enumerated for their own sake.

Candidate Grounds are compared on:

1. **Expressiveness:** which inputs, outputs, states, and possible continuations are represented?
2. **Predictive validity:** what portion of independent held-out observations is correctly predicted? What remains unknown, contradictory, or stochastic?
3. **Distinguishability:** which apparently different states are equivalent for the requested observations, and which distinctions matter under future continuation?
4. **Information effects:** which transitions create new observable distinctions, preserve information, or lose distinctions under a declared projection? Measure conditional entropy and prediction surprise rather than treating "information creation" as unbounded entropy from nowhere.
5. **Realization cost:** source bytes, retained parameter bytes, arithmetic/interpretation work, wall time, memory, energy where measurable, and network transfer.
6. **Reuse and recovery:** whether a description, cache, or certified transition survives movement between observer-hosts without repeating an already committed semantic effect.

A concise description that fits the training trace but fails a held-out continuation is not a better Ground. A content digest is not automatically a decoder for the exact source parameters, but its bytes and relations can still seed a different generative Ground. Test **what the resulting description actually does** rather than assuming that lack of exact snapshot recovery rules out operational intelligence.

## What we inherit — do not rebuild it blindly

VL-I is an **experimental research environment for new forms of model operation**. It draws upon the other repositories as tools and evidence, without requiring their existing computational machinery to define what novel Grounds may become.

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

**Pursue two experimental frontiers concurrently:** faithfully reproduce observable trained-model behavior through the VL; and build description-native intelligence where metadata or other descriptive bytes can be the entire initial substrate. Reuse existing Ground authorities and evidence contracts, but allow radically new mathematical interpretations and operating architectures.

- **D0 — Description itself becomes operation:** accept incomplete metadata, model cards, headers, images or any byte representation as a Ground seed; emit typed states, generated relations and context-dependent experimental outputs with provenance.
- **D1 — Invent operative mathematics and parameters:** synthesize model components or model weights from descriptions; explore generative laws, alternative inference/tokenization and ground-native dialogue, with independent behavioral measurements.
- **D2 — Description-native conversation:** pursue multi-turn language, code, reasoning and authorized actions with no conventional model-forward path; report the description/seed, actual computation and capability on held-out requests.
- **R0 — Grounded observation:** pinned source + model config + tokenizer identity + complete tensor inventory when available; exact observed-byte scope and explicit unknowns.
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

**Foremost new research track:** [DNO-001 — Operate a model description as a generative, conversational Virtual Ground](https://github.com/Kier73/VL-I/issues/3), independent of original pretrained weights. **Complementary source-equivalence track:** [VL-I-001 — Lift and verify full pretrained-model behavior](https://github.com/Kier73/VL-I/issues/1). These develop in parallel; neither limits the other's possible mathematical architecture.

Start with the [Architecture](docs/ARCHITECTURE.md), [Source Inventory](docs/SOURCE_INVENTORY.md) and [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md). The first experiment should reuse an already pinned, observed, full-layer decoder case, retain exact pretrained weights, and compare the same input under:

1. conventional independently implemented model execution (reference);
2. VL-native realization of the full model graph;
3. a Ground-selected alternative transition or reuse path **only when independently qualified**.

Record precise numerical and token-level equivalence, data-source completeness, wall-clock time, memory, I/O, bytes moved, work performed, and what was skipped. Then repeat for multiple tokens, architectures, and host transitions.

Success has more than one form: **description → Ground → intention → generated useful behavior** with evidence of its actual capabilities; or **observed trained snapshot → Ground → intention → verified trained-model behavior**. The first is not invalid because it was not created from original pretrained weights. The second remains the route for claims of checkpoint fidelity.

## Research principles

**Description is a resource. Interpretation is computation. Intention directs navigation. Innovate beyond inherited neural mathematics. Test what can be generated, not only what can be reproduced. Distinguish discovered behavior from claimed pretrained identity.**

This is an open research endeavor. Strong negative results—proof that a supposed shortcut fails, that a model's behavior is not determined by accessible bytes, or that a proposed quotient loses necessary distinctions—are valuable findings. The goal is to build a Virtual Layer that becomes progressively more capable by learning which descriptions actually work.

**Repository state:** the VL-I-001 integration is executable and independently tested for its specified structural scope. The much broader description-native intelligence and replacement of conventional model inference remain the primary research ambitions, to be advanced through new implementations, experiments and measured results.
