# SHAPE-002 — Observer-conditioned morphogenesis of a generative Ground

**Experiment:** same model description, changing observer intentions, independently certified mathematical shapes, separate generated neural models and causal response states.  
**Status:** executed in [GitHub Actions run 44645020064](https://github.com/Kier73/VL-I/actions/runs/44645020064); 20 tests passed.  
**Code:** `src/vli/morphological_generation.py`, `src/vli/morphological_intent.py`  
**Input:** `experiments/shape002/morphology_model.json`  
**Tests:** `tests/test_morphological_generation.py`  
**Benchmark:** `examples/morphological_generation_demo.py`  
**CI:** `.github/workflows/vli-shape002.yml`

## 1. Why shape matters to a description-native interpreter

The earlier [SHAPE-001](SHAPE_001_CONNECTOME_OBSERVER_GROUND.md) source-pinned experiment drew inspiration from an observed biological connectome, then independently verified a small observer quotient of a **separate synthetic computational Ground** using [IFS RCG-006](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/real_connectome_ground/continuity/observer_quotient/README.md).

SHAPE-002 now asks a more direct VL-I question:

**Can the same generative model be interpreted as different computational structures depending on what an observer needs to know?**

If two internal states yield identical outputs and identical future responses for every admitted intention/action under a declared observer, they may occupy the **same predictive class** in that observer's Ground. This is not merely hiding data after computing the full model. The Virtual Layer can compile a new, smaller generative model *of that observable behavior*.

The original full state remains available as a separate certified authority for later finer observations. If future prompts demand previously suppressed distinctions, a new Ground must be qualified; the interpreter cannot simply hallucinate the discarded state.

## 2. Full-source semantics, not a scalar coordinate trick

The input is an **11,398-byte complete, described model** with 12 states: three phase states crossed with four microstates. It has three actions and two independent output channels. No pretrained weights, prompt/response corpus or external language model are present.

The source transition is a total Mealy law:

- `inspect` keeps the complete state;
- `advance` advances both phase and microstate with their own moduli;
- `reset` returns both to zero.

For each of its 36 state/action cells, the output supplies the resulting phase and microstate.

Let `S` be the full state set, `A` the available actions, `delta` the transition function, `lambda` the emitted output vector, and `C` a selected observer channel set.

An observer-equivalence relation `~C` must obey:

`s ~C t => (project_C(lambda(s,a)) = project_C(lambda(t,a)) AND delta(s,a) ~C delta(t,a))`

for **every** source state in a class and every admitted action `a`.

The implementation calls **VLD's existing** `analyze_finite_ground` to propose the Moore/Myhill–Nerode predictive quotient. It then applies a **separate exhaustive all-state/all-action Mealy output-congruence proof** against the original VLD `GroundTransitionSystem`. This is necessary: a state-only quotient does not by itself prove that emitted text, tokens, tool arguments or other outputs remain unchanged.

Once proven, the reduced transition description is compiled by **VLD's original** `compile_fsm_to_neural`, through the previously implemented COORD-002 `GeneratedModelGround`. Thus, a smaller observer-specific network has its **own generated ReLU weights** and program, without executing a full-size network just to mask its outputs.

The observer's response position is a fixed-width, source/proof/intention/causal-state-bound address. It exists before the generated neural step is evaluated. Its rendered page, code and byte tokens are observations of that position, not numerical coordinates.

## 3. Actual verified morphogenesis

The same 12-state model produced **three different source-qualified generative neural Grounds**:

| Observer intention | Full source states | Proven observable states | Generated ReLU hidden units | Source transition checks |
| --- | ---: | ---: | ---: | ---: |
| Observe phase only | 12 | **3** | **9** | 36/36 |
| Observe microstate only | 12 | **4** | **12** | 36/36 |
| Observe phase and microstate | 12 | **12** | **36** | 36/36 |

These are three separate **generated neural models**, not one fixed model with invisible columns. The observer-specific mathematical descriptions and generated parameter identities are retained as source-bound Grounds.

In addition to checking all individual source transition cells, tests compare every three-action history from all twelve source states under the phase- and micro-only generated models: **648 histories × 3 transitions = 1,944 neural/state observations** checked against the original semantics.

When the prompt changes from phase-only to both phase and microstate, the newly qualified model expands from **3 to 12 states**; it does not reuse the previous three-state Ground to invent missing microstate information.

An intentionally perturbed original model changes one phase output based on a hidden microstate. VLD's predictive analysis then **splits the old three-state observer partition**, independently verifies the revised model, and produces a different Ground identity. Reusing the old shape against a changed source or changed proof is refused.

## 4. Prompt-first observer selection

The new `MorphologyIntentionRouter` reuses the existing **E-CPU structural prototype** through VL-I's previously integrated `intent_topology` path. It does not implement another independent language-model classifier.

Example:

- An observed request to report only phase proposes the **phase** topology.
- The observer externally pins the proposal's digest; only then can it be accepted.
- The system generates a response with the three-state neural Ground, retaining the full causal source state as recoverable reference.
- A follow-up request to report both phase and microstate proposes the **full** topology.
- The source-bound, causal-head-bound proposal must be newly accepted; the full response then includes both channels.

The [observed CI run](https://github.com/Kier73/VL-I/actions/runs/44645020064) produced two distinct 76-character response-position coordinates and correctly revealed [phase=1, micro=1] on the broader second request after first giving a phase-only answer.

These prototypes are **E0, witness-conditioned advisory classification**, not general semantic comprehension or implicit permission to execute tools. Changed sources, forged approval heads, stale session states and uncertain prompts are refused.

## 5. Measured cold and warm costs

The experiment deliberately does not equate smaller neural state dimension with automatic speed.

The same GitHub-hosted Python 3.11 run recorded the following sampled timings:

| Component | Phase quotient | Full state model |
| --- | ---: | ---: |
| Neural hidden units | 9 | 36 |
| Source proof and model preparation (observed) | **48.74 ms** | **3.90 ms** |
| Sampled generated neural step + local VLD oracle | **12.02 μs** | **63.59 μs** |
| Repeated-query *step-only* timing ratio | approximately **5.3×** full/phase | — |

**Important confound:** the phase model was the first compilation in the Python process and includes disproportionate one-time module initialization; this is **not** evidence that generating a smaller network intrinsically costs 12× more to prepare. The current chooser records these particular samples; a more robust cold-cost comparison needs isolated, randomized compilation orders, more repetitions and hardware/energy profiling.

With these particular observations, the cost chooser selected the full model for **eight expected cold phase queries**, but selected the smaller phase-only model for **10,000 expected queries**. It refused to use the phase-only model for a microstate intention even when the latter would be faster on paper.

Costs still **outside** the selector's per-candidate total estimates include acquiring the original source description (one measurement was about 277 ms with Python import/module initialization), generating and timing *all* candidate shapes, caller/observer approval, full source retention, renderer/response output, transport and host energy. Candidate selection and estimator costs must be accounted for in a deployment-level benchmark.

These numbers show a viable mathematical-work difference in a small finite-state model; they are **not pretrained Transformer acceleration, general intelligence, or measured end-to-end energy savings**.

## 6. Causal preservation and verification

The running `MorphologySession` retains:

- original full model Ground identity;
- original full source state, even after a narrow observer projection;
- causal head, source state and observer-projection identity for every turn;
- the new output coordinate, neural model Ground, generated code/text/tokens and exact observed output values.

When the observer changes, a new response position and possibly a newly compiled network are selected. The source does **not** irreversibly discard its extra distinctions merely because a previous observer did not need them.

A checkpoint can be restored under an externally expected digest by independently replaying deterministic pure Ground operations and verifying each causal turn's original observation digest. Host timings are excluded from the **semantic** digest so different CPU scheduling cannot falsify an otherwise identical state.

This is a concrete example of the research principle: **retain the full capacity to distinguish states, yet realize only the distinctions that matter to the current intention**.

## 7. How to reproduce

~~~bash
git clone https://github.com/Kier73/VL-I.git
cd VL-I
git clone https://github.com/Kier73/E-CPU.git upstream/ecpu
git clone https://github.com/Kier73/Virtual-Layer-Dynamics.git upstream/vld
git -C upstream/ecpu checkout f9ed67c495b00f9d26df47e291b40b41070cb609
git -C upstream/vld checkout 985ba5c2fbcc40842aed9f45f42587749620b98e
python -m pip install -e . -e upstream/ecpu -e upstream/vld

python -m unittest discover -s tests -p test_morphological_generation.py -v
python -m examples.morphological_generation_demo \
  --out artifacts/shape002-morphology.json
~~~

The workflow uploads exact source/weight/proof identities, observer-state positions, measured costs, prompt-selection evidence, generated-output statistics and cross-process-style replay digest.

## 8. The more ambitious next experiment

**SHAPE-003: learned quotient shapes for meaningful language and code semantics.** The current mathematical transitions were **fully supplied**. The next system should construct useful shapes from partial evidence, program semantics, tokenizer dynamics, memory and causal states. It should discover the shape **while interpreting the intention**, prove which distinctions it can ignore, and revise that shape when held-out prompts reveal counterexamples.

Important paths already exist:

- IFS [observer quotient discovery](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/real_connectome_ground/continuity/observer_quotient/observer_quotient.py) from temporal observations; leverage source-bound full-edge verification as examples grow.
- VLD [structural linear operators](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/src/vld/accel/structural_ops.py) and [proof-producing intention compiler](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/src/vld/compiler.py) for discovering algebraic alternatives to dense execution.
- IFS [evidence-compiled cognitive Grounds](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/codonic-capability-atlas) and VGI [self-constructing interpreter](https://github.com/Kier73/Virtual-Ground-Interpreter) for learning meaning and refining state discrimination.
- Existing VL-I [COORD-003 generative content track](https://github.com/Kier73/VL-I/issues/11) and [SELF-002 recursive interpreter research](https://github.com/Kier73/VL-I/issues/6).

**The eventual aim is not a tiny finite model. It is a Virtual Layer whose very *shape of intelligence* evolves with the required observations and whose interpreter can exploit newly proved generative laws, rather than inheriting the full cost of conventional model execution.**
