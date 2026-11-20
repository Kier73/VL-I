# SHAPE-003 — Learn the shape before generating the model

**Status:** executable black-box learning and independently certified, observer-dependent neural generation  
**Code:** [learned_morphology.py](../src/vli/learned_morphology.py)  
**Reproducible experiment:** [learned_morphology_demo.py](../examples/learned_morphology_demo.py)  
**Tests:** [test_learned_morphology.py](../tests/test_learned_morphology.py)  
**CI:** [vli-shape003.yml](../.github/workflows/vli-shape003.yml)

## The step beyond SHAPE-002

SHAPE-002 showed that a complete twelve-state model description could be expressed as three observer-specific neural models with different generated weights. Its source transition cells were fully supplied to the morphology analyzer.

**SHAPE-003 does not supply that graph to the learner.** The IFS observer receives only action sequences and their observed output traces. A first, underspecified hypothesis is allowed to be wrong. A separate verifier finds the shortest distinguishing history, the learner gathers that additional evidence and revises its predictive state partition. Only after it passes a **whole-reachable-product check on the independently held finite source** does VL-I use VLD to compile the resulting learned state machine into neural weights.

The intended trajectory is now executable:

\[
\text{unseen source Ground}
\to \text{black-box observation}
\to \text{tentative predictive topology}
\to \text{counterexample}
\to \text{revised observer topology}
\to \text{verified neural parameters}
\to \text{response-state coordinate}
\to \text{new code / text / tokens}.
\]

A coordinate remains the *position of an observer-relative response state*, not the numerical next-state label. It is established before response formation and binds the model, intended action, causal head and observer context.

## Existing implementations actually reused

1. **IFS [COG-008 partial_observation.py](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/codonic-capability-atlas/partial_observation.py):** real \`ObservedOracle\`, \`MembershipWitness\` hash-linked probe ledger, \`fit_table\`, \`first_mismatch\`, checkpoint and source-free recovery. The observer cannot claim its own observational challenges prove universal equivalence.
2. **IFS [COG-007 predictive_state.py](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/codonic-capability-atlas/predictive_state.py):** original \`ActiveMachineLearner\` observation-table constructor, predictive state representatives, closure and consistency revision, and typed \`Machine\` candidate.
3. **VLD [GroundTransitionSystem](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/src/vld/signal/transitions.py):** independent reference of declared finite semantics, unavailable to the IFS candidate learner. The VL-I **verification receiver**, not the learner, has this source authority.
4. **VLD [compile_fsm_to_neural](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/src/vld/models/procedural.py):** actual generated ReLU model parameters through the preexisting VL-I COORD-002 \`GeneratedModelGround\`. This is not a pretrained neural weight snapshot.
5. **VL-I [GenerativeFormation](../src/vli/generative_response.py):** fixed-width state-position address, generated text/code/token content, and VLD-backed progressive response projection.

The proof is explicit: starting from the original initial state and the learned initial predictive class, the independent receiver traverses **every reachable pair** \((s,\hat{s})\) and checks each admitted action's projected output before following both next-state relations. A mismatched output yields a concrete shortest counterexample action word. If no mismatch remains and the finite product closes, the learned machine is exactly equivalent for the **declared finite original source and observer channels**, including every possible future action history in that alphabet.

That does **not** establish equivalence for undisclosed source states, unseen actions, different observer projections, unspecified tools or general language tasks.

## The deliberate falsification

The learner starts with a deliberately weak distinguishing suffix, \`reset\`. Resetting hides the phase and microstate differences, so the observation table proposes a **one-state** machine. It is incorrect.

The separate conformance receiver finds a shortest counterexample, e.g. the action history \`advance, inspect\`. The learner must then observe that actual history through the same black-box interface and incorporate its distinguishing suffixes. The original IFS table learner refines its predictive partition and proposes a new model.

The eventual observed state classes are:

| Observer contract | Underfit first candidate | Verified learned classes | Generated neural hidden units |
| --- | ---: | ---: | ---: |
| Phase-only | 1 | **3** | **9** |
| Microstate-only | 1 | **4** | **12** |
| Full phase+micro | 1 | **12** | **36** |

The exact counterexample and model revision history are retained in the source-bound certificate.

The original reference model contains twelve physical states and 36 declared state/action cells. The learner's smaller predictive machine is **not preloaded with those cells**. It acquires its own trace-specific observations, each bound into the original IFS COG-008 evidence ledger.

## Experimental result

The first successful [GitHub Actions run 44647162211](https://github.com/Kier73/VL-I/actions/runs/44647162211) ran the original pinned IFS, VLD and E-CPU implementations, passed **18 focused tests**, and produced a machine-readable [experimental artifact](https://github.com/Kier73/VL-I/actions/runs/44647162211/artifacts/14589670951).

| Measurement | Phase-only | Microstate-only | Full observer |
| --- | ---: | ---: | ---: |
| Black-box membership queries | **30** | **39** | **111** |
| Distinct modeled observer states | 3 | 4 | 12 |
| Learning/counterexample revision rounds | 2 | 2 | 2 |
| Counterexample | advance → inspect | advance → inspect | advance → inspect |
| Independent full-reference edges in final equivalence check | 36 | 36 | 36 |

The original finite source has **36 directly declared cells**. The 111-query full-observer acquisition is thus *not* evidence of a cheaper model-discovery path. Actual executed source transition steps across membership observations were **91 (phase), 138 (micro), and 838 (full)**, reflecting repeated prefix replay. These costs are **not** neural-generation, output-rendering or independent proof costs; those are measured separately. In the full-observer case, the learner performed considerably more source work than directly enumerating all 36 cells. Optimizing acquisition is a separate frontier, not an established speedup. The value demonstrated here is a **correct learned topology without giving the learner the source transition graph**, not a proven reduction of training/sample acquisition cost.

The original finite source descriptor is 11,398 bytes. The generated model’s actual parameters and code/text/UTF-8 byte-token views are synthesized from the **learned predictive machine**, not from a pretrained checkpoint.

Four causal turns change observer scope from phase-only to micro-only, then to full, then back to phase. All four have distinct 76-character state-position handles. A pinned, hash-linked conversation checkpoint restores the exact prior action history and each semantic render digest.

### Independent evidence and authority

**IFS ledger:** claims only the exact paths queried, source identity, outputs observed and observed hash-linked continuity. Its recovered model remains correctly labeled \`provisional_finitely_challenged\` with global equivalence \`not_established\` based on those witnesses alone.

**External conformance certificate:** separately records the model and reference identity, observer scope, all candidate/refinement rounds, observed ledger and an exhaustive finite-product result. This is a stronger independent assertion for the **particular declared finite system**.

**Generated neural Ground:** contains newly produced, whole-domain checked VLD neural weights and program semantics. Its claims are bounded by the certified observer contract, not interchangeable with arbitrary different observers.

**Conversation memory:** retains symbolic *action histories* so changing from one learned observer model to another cannot incorrectly reuse the old model's numeric class index. This follows IFS' finding that a collapsed state index alone cannot safely recover distinctions for a broader intention. Replay reconstructs the true learned state in the newly selected model from the symbolic history without repeating any external side effects; this prototype has no effectful external calls.

## Prompt-conditioned choice of a learned intelligence shape

The full experiment now includes \`LearnedObserverIntentionRouter\`, composed with the **existing E-CPU \`IntentPrototypeModel\`** through VL-I's \`intent_topology\`. It does not train another independent language model.

From six explicit training witnesses (two examples each for phase-only, micro-only and full-state observation), E-CPU proposes an observer-dependent output topology. This proposal is only **E0 advisory**. A caller must independently pin its exact digest. VL-I then requires:

- identical original source identity and causal conversation head;
- unchanged symbolic action history;
- the same admitted list of learned observer-model identities;
- matching IFS source-observation ledger heads;
- a separately certified learned neural Ground for the proposed output scope.

On the actual run, a prompt asking only for phase selected the **three-state learned model**, generated a state response, and retained its full symbolic causal history. A follow-up prompt requesting both phase and microstate selected the **twelve-state independently learned model**; it recovered the previous hidden distinction and rendered the qualified [1, 1] observation. Both response-position coordinates were distinct.

The protocol explicitly refuses stale proposals, changed membership evidence, forged external heads and insufficient observer intent evidence. **Automatic response-action semantics are not implied by this view proposal**: the external action (\`advance\`, \`inspect\`, etc.) remains independently specified under the admitted action alphabet.

The companion [GitHub Actions run 44647754297](https://github.com/Kier73/VL-I/actions/runs/44647754297) passed **18 tests**, covering this prompt-conditioned path and the original black-box learner. Its [machine-readable artifact](https://github.com/Kier73/VL-I/actions/runs/44647754297/artifacts/14589457688) records query counts, actual black-box transition steps, model/proof digests, response state coordinates and the prompt-induced shape change.

## Reproduce

~~~bash
git clone https://github.com/Kier73/VL-I.git
cd VL-I
git clone https://github.com/Kier73/In-Formation-Space.git upstream/ifs
git clone https://github.com/Kier73/Virtual-Layer-Dynamics.git upstream/vld
git clone https://github.com/Kier73/E-CPU.git upstream/ecpu
git -C upstream/ifs checkout 0c712a83a33d14a0de630e26facd74cb487e102e
git -C upstream/vld checkout 985ba5c2fbcc40842aed9f45f42587749620b98e
git -C upstream/ecpu checkout f9ed67c495b00f9d26df47e291b40b41070cb609
python -m pip install -e . -e upstream/vld -e upstream/ecpu
export PYTHONPATH=upstream/ifs/experiments/codonic-capability-atlas
python -m unittest discover -s tests -p test_learned_morphology.py -v
python -m examples.learned_morphology_demo \
  --out artifacts/shape003-learned-morphology.json
~~~

## Actual next question

Can the interpreter **acquire generative semantic structure from incomplete conversation, code execution, model observations and communication paths**, revise it from meaningful counterexamples, and then generate an *improved* model whose interpretation is cheaper than the conventional original?

We now have an operational composition of the ingredients:

- real observation-only learning and persistent source evidence;
- counterexample-driven model revision;
- exact qualification for one finite source and observer;
- generated neural parameters conditioned on the learned shape;
- causal, multi-view language/code/token response formation.

The research frontier is to make the **semantic structure learned** richer than a small known finite transition system: acquire partial Ground laws from actual programs and externally observed model tokens, generate richer response states without enumerating all transitions, choose high-value probes using E-CPU/VGI partial attention, and *amortize* learning/proof costs over useful future answers. Any claim of replacing neural model execution must be checked on a real original model at its original scale, not inferred from this finite control.

**Founding intuition:** intelligence is not solely the neural geometry inherited from training. The Virtual Layer may infer and revise an *observer-relative geometry of computable behavior*, then use that shape as a Ground for generating the response. The interpreter itself may eventually be a recursively learned, reconstructable part of that Ground.
