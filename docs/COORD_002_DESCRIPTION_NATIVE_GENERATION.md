# COORD-002 — A description forms a neural Ground and generates a response-state

**Status:** executable, independently tested finite-model milestone  
**Date:** 2026-11-19  
**Implementation:** [generative_response.py](../src/vli/generative_response.py), [generative_intent.py](../src/vli/generative_intent.py)  
**Real description:** [described_model.json](../experiments/coord002/described_model.json)  
**Reproduction:** [generated_response_demo.py](../examples/generated_response_demo.py)  
**CI:** [vli-coord002.yml](../.github/workflows/vli-coord002.yml)

> **A coordinate locates a computational state before the output exists. The Virtual Layer interprets the Ground, forms the required state, and makes its contents observable.**

## What this changes

[RESPONSE-001](INTENT_STATE_POSITION_AND_RENDERING.md) located an intent-specific state, then progressively rendered **already supplied** content.

COORD-002 takes a new step: a **complete descriptive model Ground** supplies symbolic state/action/output relations but **no trained parameters, precomputed response text, Python program, or token sequence**. An existing VLD compiler synthesizes new neural weights from those laws. On each intention:

1. Identify the source model, current causal state, intended action, observer and requested output views.
2. **Locate a coordinate before any neural forward step or output generation.** The coordinate is an opaque state-position root, not a scalar prediction or an index into pretrained weights.
3. Execute one **VLD-generated neural transition**, checking its next-state and output-channel results independently against the described transition Ground.
4. Interpret that state into new grounded text, a Markdown page, a complete executable Python transition function, and UTF-8 byte-token values. The program is synthesized from the admitted model transition table; it was not stored as source code in the input.
5. Reuse the existing **E-CPU exact byte Ground** and **VLD navigable computation DAG** to progressively render only the selected output views, with a trace/receipt and an optional inert tool proposal.
6. Commit the causal turn identity; the next prompt begins from the new state. Checkpoints are verified by reconstructing and replaying the deterministic pure Ground relations under an external expected digest.

A **new causal state position** is allocated for each turn. Within a turn, its coordinate remains unchanged between LOCATED, FORMING and RENDERED. The nested response projection has its own Ground address for the generated output bytes; both identities are retained rather than confused.

## The decisive reuse from existing repositories

**Virtual-Layer-Dynamics**, rather than a new VL-I neural framework, supplies [compile_fsm_to_neural](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/985ba5c2fbcc40842aed9f45f42587749620b98e/src/vld/models/procedural.py) and [GroundTransitionSystem](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/985ba5c2fbcc40842aed9f45f42587749620b98e/src/vld/signal/transitions.py). The compiler constructs actual integer-ReLU weights using one-hot state/action encodings and exhaustive whole-domain equivalence checks. It needs **no pretrained model weight file**.

**E-CPU** supplies [DataGround](https://github.com/Kier73/E-CPU/blob/f9ed67c495b00f9d26df47e291b40b41070cb609/mailcpu/data_ground.py) for source-byte observation and [IntentPrototypeModel](https://github.com/Kier73/E-CPU/blob/f9ed67c495b00f9d26df47e291b40b41070cb609/mailcpu/intent_embedding.py) for witness-conditioned advisory prompt/action selection.

**VLD's existing NavigableComputationDAG**, not a new scheduler, tracks partial output observation, dependencies and versioned checkpoints. RESPONSE-001 already integrates it.

**IFS' evidence-compiled cognitive Grounds** and **VGI's self-constructing interpreter** are useful models for the next step: improving acquired language capabilities and discovering new generative relations from observations, rather than keeping action labels fixed forever.

## Reproducible measurement

First successful pinned [COORD-002 GitHub Actions run 44639564158](https://github.com/Kier73/VL-I/actions/runs/44639564158) recorded:

| Observed value | Result |
| --- | ---: |
| Actual descriptive Ground input | 3,199 bytes |
| Trained-model parameters downloaded | **0** |
| Generated neural network | 9 ReLU hidden units |
| Generated integer neural coefficients including biases | **108** |
| Exact neural/reference transition cells verified | **9 of 9** |
| Stateful interaction | **4 distinct turns** |
| New generated views | Text, Markdown, Python code, UTF-8 byte tokens |
| State-position width | **76 characters**, unchanged by response size |
| Causal recovery | Full prior turn chain reconstructed; same final state/head |
| Observed model Ground compilation (single runner) | 455 ms |
| Four-turn generation (single runner) | 40 ms |
| Full-history deterministic verification replay (single runner) | 14 ms |

The millisecond values are one GitHub-hosted Python 3.11 observation, not a stable performance baseline, and they are **not** LLM speedups. The generated neural model's whole-domain verification runs on a small **three-state, three-action system**. It does not establish scaling to arbitrary 30B-parameter pretrained language-model behavior.

The [workflow artifact](https://github.com/Kier73/VL-I/actions/runs/44639564158/artifacts/14585901424) preserves source identity, generated-weight identity, causal response coordinates, view digests, step receipts and measured work. Other effects such as Python startup/setup are not included in the timing intervals.

### How to run

~~~bash
git clone https://github.com/Kier73/VL-I.git
cd VL-I
git clone https://github.com/Kier73/E-CPU.git upstream/ecpu
git clone https://github.com/Kier73/Virtual-Layer-Dynamics.git upstream/vld
git -C upstream/ecpu checkout f9ed67c495b00f9d26df47e291b40b41070cb609
git -C upstream/vld checkout 985ba5c2fbcc40842aed9f45f42587749620b98e
python -m pip install -e . -e upstream/ecpu -e upstream/vld

python -m unittest discover -s tests -p test_generative_response.py -v
python -m examples.generated_response_demo \
  --out artifacts/coord002-description-native.json
~~~

The input [model JSON](../experiments/coord002/described_model.json) is intentionally human-readable. Its nine transition observations specify states, actions and two bounded output channels, not actual neural matrices or response text. Change one admitted transition to produce **different generated neural weights, a different Ground identity and different state-position coordinates**. The tests exercise this control.

The exact generated Python function implements *all admitted transitions*, not just the examples used in one conversation; tests compile and execute it against each independent VLD source oracle cell. All tool outputs are **inert proposals**; none are dispatched.

## Prompt-first navigation

Before this step, the caller supplied the typed action (e.g. \`advance\`) directly. Now \`generative_intent.py\` reuses E-CPU's structural prototypes trained on explicitly witnessed prompts:

~~~python
witnesses = (
    ActionWitness("inspect the model state", "inspect"),
    ActionWitness("show the current state observation", "inspect"),
    ActionWitness("advance to the next state", "advance"),
    ActionWitness("continue the described system", "advance"),
    ActionWitness("reset the state", "reset"),
    ActionWitness("return to the initial state", "reset"),
)
proposal = propose_action(session, witnesses, "advance to the next state")
result = accept_and_respond(
    session, witnesses, proposal,
    expected_proposal_head=proposal["proposal_digest"])
~~~

The proposal carries the **exact model Ground, current causal conversation head, current state, witness set, E-CPU prototype identity, classifier confidence and uncertainty margin**. It is **E0 advisory evidence** requiring external caller acceptance, not autonomous permission for side effects or proof of natural-language understanding. If the session advances in between proposal and acceptance, the old proposal refuses.

A sufficiently uncertain or unrecognized prompt remains unresolved rather than being falsely promoted into model behavior.

## Distinctions that protect the research without narrowing it

- **Yes:** model description bytes were made operational; a real neural network and its parameters were generated without a pretrained weight checkpoint.
- **Yes:** the coordinate was established *before* the transition result and response text were computed, and remained stable throughout response formation.
- **Yes:** genuinely new model-specific program code and grounded text were constructed from the supplied model law and changing causal state.
- **Yes:** a multi-turn state relation and hash-bound recovery were tested against an independent VLD transition oracle.
- **Not yet:** general-language knowledge or meaning comparable to a pretrained LLM. Text is composed using a fixed Ground interpreter/formatting law, not a trained open-domain decoder.
- **Not yet:** model weights of a particular existing pretrained snapshot reconstructed from its file header. Those would require a separately checked source-equivalence contract.
- **Not yet:** arbitrary zero-cost computation. The compiler executes real Python/arithmetic and the new network evaluates generated weights.
- **Not yet:** self-hosting virtual interpretation of the full VLD model synthesizer.

These limits are properties of **this experiment**, not design limits on what new mathematics or model capabilities VL-I may develop.

## Next breakthrough

The important next question is how to turn a finite generative transition Ground into a **compositional language and coding Ground** with expandable state descriptions and observer-relative learning.

Specific research paths:

1. **COORD-003 / predictive token law:** make the content itself a generative computation rather than a fixed response-format compiler: typed grammars, procedural parameterized token decoders, data-driven learned transition systems, transformer Ground primitives, or new state-navigation algebras.
2. **Causal composition:** compose several Grounds into one response state, including different world knowledge, code functions, memory and tool observations, while permitting independent evidence and observer-specific collapse.
3. **Intent beyond witnesses:** learn new intentions from context and counterexamples with VGI and IFS learners, preserve uncertainty and choose which observations are worth obtaining.
4. **Recursive interpreter:** derive more of the generative compiler and response-realizer operations as Ground laws, building on [SELF-002](https://github.com/Kier73/VL-I/issues/6).
5. **Scale and generality:** test unknown model families, larger causal descriptions, genuine tokenizer/weights/head captures and held-out natural-language/code benchmarks with truthful computation accounting.

**The ambition remains revolutionary:** a description should be able to become an operational, intelligent state-space, whose intention-addressed coordinates render meaningful new results without requiring the conventional model runner to execute that intelligence.
