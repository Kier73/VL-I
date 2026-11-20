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

## Coordinates locate response states, not numbers

**A coordinate is the address of a state in an intention-conditioned topology—not a numerical representation of the state.** The response, rendered page, code, emitted tokens, tool proposals/results, causal memory and interpreter lineage belong to the observable structure at that position. The observer's intention constrains which topology must be navigated and which views are relevant.

The response need not be preassembled. The Virtual Layer can first locate a state position, then **progressively form its observable content** as its required Ground relations become available: outline → component observation → evidence-backed composition → text/page/code/tool projection. This is the **diffusion-like formation** research direction, not a claim that a stochastic diffusion network is already operating.

This principle is now exercised in [RESPONSE-001 — Intent as topology and state-position rendering](docs/INTENT_STATE_POSITION_AND_RENDERING.md):

- Actual E-CPU byte Grounds anchor the supplied information.
- VLD's original navigable, checkpointable DAG progressively forms a response, keeping the **same opaque coordinate** across incomplete and complete observations.
- The output can include page, code, text, token IDs and **inert** tool proposals; actual authority remains separate.
- E-CPU's existing \`IntentPrototypeModel\` can now propose the response topology from **witnessed prompts**, with uncertainty and externally pinned acceptance. It is an early intention-navigation mechanism, not general natural-language comprehension.
- The coordinate is fixed-width independent of source payload size. Data acquisition, verification and rendering are **not** assumed scale-free.

**RESPONSE-001** renders supplied source bytes. **COORD-002 now goes further:** it uses a model description to **generate new neural weights, interpret a state transition, and form new code, grounded text, Markdown, and UTF-8 byte tokens from that transition**. The coordinate is positioned *before* the generated output exists. Four causal turns, an independent all-transition neural comparison, E-CPU prompt-to-action proposals, and restart verification have been exercised in CI.

### COORD-002 — A description becomes a generative model Ground

- **3,199 input bytes** describing a complete three-state/nine-transition computational Ground; **zero pretrained weight bytes acquired**.
- VLD constructs **108 neural coefficients** and a **nine-unit ReLU model**, checking all nine source transitions independently. The weights are *generated from the description*, not guessed as the exact values of some unknown pretrained checkpoint.
- A prompt and causal state locate the new response position **before** the output is computed. The Ground then forms a verifiable new state and creates a runnable Python transition program, explanation, Markdown page, token bytes and optional inert tool proposal.
- E-CPU's existing \`IntentPrototypeModel\` can propose a generative action from witnessed prompts; a model- and causal-state-bound caller approval is mandatory, and stale/uncertain prompts refuse.
- Sessions retain causal positions across turns; restart replays pure interpreted relations to independently verify committed history. There is no hidden language-model backend.

**This is an actual description-native generative capability, not an open-ended chatbot.** Its fixed finite-state grammar and formatting laws still constrain what the resulting intelligence can do. Discovering richer predictive and language-producing laws, learned state, general intentions and recursive interpreter operation is our next focus—not a reason to retreat to conventional inference.

Read [COORD-002 — Description-Native Generative Model and Conversation](docs/COORD_002_DESCRIPTION_NATIVE_GENERATION.md), inspect [the executable experiment](examples/generated_response_demo.py), and follow [the original COORD-002 research issue](https://github.com/Kier73/VL-I/issues/9).

### INTERACT-001 — From recognized relations to constructed results

**A Ground is not merely a catalogue of what can be observed.** The latest constructive-data experiment applies qualified, typed transformations to actual source records, produces new values, and forms usable Markdown, JSON, CSV, text, tokens and **independently executable generated Python** from the same intention. An explicitly authorized data edit creates a new source Ground, so a subsequent question and its renderable coordinate reflect the **changed reality**, not stale observed content.

The operation is causal and testable: location of a state position precedes computation, E-CPU and VLD's **existing** response DAG renders the built output, and a pinned checkpoint independently replays prior constructions and revisions. The demonstration works against VL-64's **actual archived, original-scale Qwen and DeepSeek structural observations**, not their unobserved pretrained weight values. This is verified data interaction and artifact formation, **not** full language-model inference or unconstrained natural-language intention understanding.

Read [INTERACT-001 — Constructive Data Grounds](docs/CONSTRUCTIVE_DATA_GROUNDS.md), run [the executable example](examples/constructive_data_demo.py), and see [the Python 3.11/3.13 verification](.github/workflows/vli-constructive-data.yml).

## Malleable computational shape — observer-relative intelligence

A state-space can be vast even when **the distinctions needed by a particular observer are few**. The connectome investigations in [In-Formation-Space](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/real_connectome_ground) and the predictive-quotient, structured-operator and sparse-neural foundations in [Virtual-Layer-Dynamics](https://github.com/Kier73/Virtual-Layer-Dynamics) suggest a path beyond conventional fixed network architecture: **let the intention qualify which state distinctions to preserve, then navigate the smallest Ground that actually supports the requested observation**.

SHAPE-001 is a live cross-repository experiment. It loads the source-pinned OpenWorm **448-label / 7,379-record** connectivity observation, then forms a **separate synthetic** mathematical computational Ground, not a claim about biological neural behavior. IFS' original observer-quotient verifier proves the smaller topology preserves every future observation in its declared finite-field model. VL-I locates an opaque response state and renders its verified readout using VLD.

On a measured 256-microstate synthetic expansion, a **verified eight-state observer quotient** matched the original on five queries, yielding a **32× reduction in state dimension** and a **35.6× query-only Python timing ratio**. The one-time source intake/discovery/proof cost was approximately 65 ms, **more expensive than those five queries combined**. These are encouraging shape-of-computation measurements, not measured pretrained LLM acceleration or free interpretation. The research target is to adaptively form and refine such topologies around *real model intentions and generative Ground relations*, with full cost accounting.

See [SHAPE-001 — Connectome-inspired Observer Grounds](docs/SHAPE_001_CONNECTOME_OBSERVER_GROUND.md) and [the measured cross-repository experiment](examples/connectome_shape_demo.py).

**SHAPE-002 now makes the intelligence model itself malleable.** Starting from a **single 12-state, 36-transition description**, VL-I uses VLD's predictive quotient and an independent all-state/action output-equivalence proof to generate **three different neural models** for three different observers:

- Phase-only: **3 observable states, 9 generated neural hidden units**.
- Microstate-only: **4 observable states, 12 hidden units**.
- Both channels: **12 observable states, 36 hidden units**.

The same external observer can now **change what they request conversationally**. An E-CPU witnessed-intention proposal, accepted with a source- and causal-head-bound digest, chooses a qualified projection; a broader follow-up automatically selects a finer neural Ground while retaining the underlying original state. All models produce newly generated code, text and byte-token responses, not just hidden/masked columns of one large model. Tests cover all original transitions, longer action histories, altered sources, proof forgeries, new distinguishing outputs and recoverable multi-turn shape changes.

A measured GitHub Actions run found approximately **5.08× lower repeated neural-step-plus-local-oracle time** for the three-state phase network than the full twelve-state model (8.29 μs versus 42.07 μs). We corrected an initial cold-cost confound by charging shared compiler imports to the source-admission bootstrap; after that correction the three-state form also required less separate model compilation work. The one-time source/bootstrap cost remains substantial and is not included in query-only speed comparisons. This is bounded description-native model arithmetic, **not** a verified speedup for pretrained language models or a guarantee of cheaper total deployment.

See [SHAPE-002 — Intention-driven neural morphogenesis](docs/SHAPE_002_INTENTION_MORPHOGENESIS.md) and [its measured generative conversation](examples/morphological_generation_demo.py).

**SHAPE-003 now learns the computational shape through observation instead of receiving the original transition table.** The original **IFS COG-008 black-box learner** sees only action/output traces, builds an intentionally underfitted predictive hypothesis, receives real counterexamples from a separate exact finite-state verifier, and refines its learned model. **Only the independently certified candidate** is compiled by VLD into new neural parameters and used to form a VL-I response-state.

On one 12-state source, the learner discovered **three different observer-dependent models (3, 4 and 12 states)** using 30, 39 and 111 actual black-box membership queries respectively. Each started as an incorrect one-state machine; the true distinguishing history caused a correction. The new generated neural models then supported a four-turn conversation switching observer shapes, rendering new code/text/tokens and recovering its causal history from an independently checked checkpoint. **E-CPU's existing witnessed prompt-to-topology system** also chose a learned three-state model from a phase-only request and a newly qualified twelve-state model for a subsequent full-state request, while preserving the true symbolic causal history. The source-state graph was **not available to the learning algorithm**; the external conformance verifier held that authority separately.

**Acquisition is not yet cheap**: 111 membership queries and 838 actual source transitions (including repeated prefixes) for the complete observer exceed the source model's 36 direct transition cells. The other two observer scopes consumed 91 and 138 source transitions respectively. The important milestone is **learning and reconstructing the shape of computation from observation**, not a claim of reduced model-training cost or pretrained LLM capability. The challenge now is learning richer response semantics with fewer observations and reusing them to eliminate substantial conventional model interpretation work.

See [SHAPE-003 — Observation-learned intelligence shape](docs/SHAPE_003_LEARNED_OBSERVER_GROUND.md) and [the executable black-box model-learning experiment](examples/learned_morphology_demo.py).

**SHAPE-004 closes a concrete cost loop:** an observer-host can now keep **verified structural memory of source-state action prefixes** while the IFS black-box learner continues to see only its usual action/output observations. The same three learning tasks originally executed **1,067 source transitions**; with shared, source-pinned prefix memory they executed **136** (**7.85× fewer**), with **exactly identical IFS evidence ledgers, learned observer models and generated VLD neural weights**. The memory retains 137 prefix positions and is reconstructed under a pinned checkpoint by independently rechecking every cached edge.

This is **reduced underlying transition work**, but not a demonstrated total runtime win in this local Python control. Six alternating-order warm trials measured about **34.78 ms uncached versus 37.05 ms cached** across learning and compilation—roughly **6.5% slower** due to cache/source setup and interpreter overhead. We preserve the negative result: a faster computational path must outperform *all* of its costs, not just reduce its mathematical steps. The next useful substrate is a costly observer-host, communication path or actual model operation, where retaining previously verified computation may outweigh the navigation overhead.

Read [SHAPE-004 — Verified structural observation memory](docs/SHAPE_004_VERIFIED_OBSERVATION_MEMORY.md) and [its source-work versus wall-time benchmark](examples/observation_memory_demo.py).

**SHAPE-005 now lifts this memory relation into original-scale model communication.** It directly composes **VL-64's original, source-qualified pretrained tensor range protocol** with **E-CPU's existing content-addressed ComputationalObjectStore and StructuralMemoryPageBacking**. A received tensor/header range is admitted under its exact model, revision, shard, byte position, SHA-256, source observer-host identity and original range witness. Later qualified observers retrieve the identical *actual observed bytes* from a local E-CPU immutable object, with a distinct **PINNED_GLANCE_DERIVATION** source-evidence chain—not a fictional new HTTPS observation. The original trusted host may be recontacted to independently check retained bytes; cross-model, forged/changed bytes or unstored ranges refuse.

A first **real loopback TCP experiment using fabricated BF16 safetensors bytes** measured **three original TCP range requests, then zero on warm reuse**, with the exact same 64-dimensional independently checked VL-64 numerical result. One GitHub Actions run measured **3.83 ms cold versus 0.45 ms retained**, but those are single-run fixture timings, *not* a pretrained-model or statistically established speedup. **The original full-size, pinned Qwen3.8-27B HTTPS experiment subsequently succeeded.** Using the actual original 5,120-dimensional layer-0 normalization weights, VL-64 independently verified a native numerical result. E-CPU retained **55,640 bytes from three qualified HTTPS range reads**, and the same operation was reconstructed with **zero additional original-host requests**. The observed single-run times were **785.3 ms original source versus 6.29 ms retained**, with unchanged numeric result and a successful source-free memory restart. This is a real trained *partial operator* and verified source-byte/navigation improvement, **not a full model forward pass or a 125× LLM speedup**. The original evidence is saved in [SHAPE-005 results](experiments/pretrained_range_memory/results/2026-11-19-qwen-original-range-memory.json).

See [SHAPE-005 — Pretrained observer-host structural memory](docs/SHAPE_005_PRETRAINED_OBSERVER_MEMORY.md) and [the local TCP / live original pretrained range experiment](examples/pretrained_range_memory_demo.py).





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
