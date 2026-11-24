# PBG-002 — Adaptive intention: observe the desired outcome, then change the material

**Status:** implemented finite exact-rational experiment. Independent full-source and reduced-Ground tests are in the CI suite. This extends [PBG-001](OBSERVER_BACKWARD_AND_PREDICTIVE_GROUND.md), [PFI-001](PROBLEM_FORMED_INTERPRETER_PFI001.md), [TDL-003](COST_AWARE_REFINEMENT_TDL003.md), [E-CPU INTENT-001](https://github.com/Kier73/E-CPU/blob/main/docs/INTENT_001_OBSERVER_FORMED_FABRIC.md), and [IFS RE-001](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/response-equivalence).

## Research question

An intention has two distinct mathematical phases:

1. **Intended outcome:** an observer specifies a future condition, before the external state has changed.
2. **Material realization:** permitted observations and interventions navigate the external state toward that condition. A predicted outcome is not itself evidence that an actuator, host, or communication path realized it.

We investigate whether a Virtual Layer can form its own observer-relative state and policy from the *desired future condition*, rather than simulating the original full source-state representation.

The phrase **adaptive perplexity** is treated here as a research metaphor and a measurable goal-event quantity, not a redefinition of conventional language-model perplexity. A goal event with predicted success probability p has surprisal -log₂ p bits and inverse-probability factor 1/p; conventional LM perplexity is an exponential average token negative log-likelihood and is a different statistic. Neither a low-entropy belief nor an abundant stream of sensor readings necessarily improves goal attainability.

## Exact source and compilation

The source is a supplied finite controlled stochastic model with n hidden material states. Each admitted intervention a has row-stochastic rational matrix A_a and cost c_a≥0. Each passive sensor s has outcome matrices S_{s,y}≥0 and cost c_s≥0, satisfying

\[
\sum_y S_{s,y}=I_n.
\]

Thus receiving a sensor reading can update the observer's distribution, but cannot itself alter the unconditional physical state. The goal is a binary event vector g∈{0,1}ⁿ and prior belief α with α·1=1. The ground constructor begins with the columns 1 and g, and closes their span under **every** intervening A_a and sensing S_{s,y}. It obtains basis matrix B and exact rational operators K:

\[
\boxed{A_aB=BK_a,\qquad S_{s,y}B=BK_{s,y}.}
\]

This is a source-global observer-sufficient predictive coordinate system. Its dimension r is the dimension of the smallest common invariant column space containing the stated goal and normalization readout. It can be smaller than n; no finite-horizon word enumeration is used for compilation.

The online state is **z=αB**, not a full source belief α. By basis construction its first coordinate is 1 and its second coordinate is the goal probability. The two operation types differ:

- **Predicted material intervention:** z'=zK_a, preserving the normalization coordinate 1.
- **Condition on a supplied sensor outcome:** p_y=(zK_{s,y})₀; for p_y>0, z'=zK_{s,y}/p_y.

All-horizon preservation of these observations, for the fully declared source, follows from exact commutation; the implementation also independently audits each source/compiled operator. No hidden source-state matrix is needed for online predictions, planning, or session restoration.

## Intent-conditioned policy

Given a bounded budget of h operations, the Virtual Layer can choose to stop, predict a material intervention, or request a sensor and condition its continuation on each possible reading.

The implemented **exact rational** planner maximizes

\[
\boxed{J(\pi)=P(g\text{ at stop}\mid\pi)-\mathbb E_\pi[C],}
\]

where costs are declared success-equivalent units, **not measured host latency or an intrinsic uncertainty penalty**. It compares all choices recursively to h≤4, reusing equal predictive states.

The policy *can* choose sensing because its information changes which subsequent material action is most effective. It does not award sensor data intrinsic value when no improvement in expected goal realization follows. This is consistent with the distinction between pragmatic and epistemic value in active-inference literature, but is a specified finite expected-success planner rather than an implementation of a general free-energy theory.

## Measured exact-rational fixture

Source: eight hidden states = two material types × two success flags × two redundant replica tags. A 20% prior success rate is declared; either intervention sets success for its matching material type and clears success for the opposite type. A type-reading sensor is accurate with probability 3/4 and costs 1/20, while the intervention costs 1/100. A second sensor produces an independent fair coin.

| Metric | Direct intervention (h=1) | Observe type, then condition intervention (h=2) |
| --- | ---: | ---: |
| Goal success probability | 1/2 | 3/4 |
| Expected declared cost | 1/100 | 3/50 |
| Objective = success - cost | 49/100 | 69/100 |
| Goal-event surprisal, bits | 1 | -log₂(3/4) ≈ 0.415 |

**Predictive reduction:** eight source-state coordinates → four goal-sufficient Ground coordinates. Change the intention from material-success to “type 0” and the corresponding processor uses **two** coordinates instead. The Ground fingerprint changes with the goal.

**Why an observation is different from an effect:** the type sensor preserves the prior 1/5 success probability *in expectation* before an intervention. The subsequent type-matched material action raises the predicted goal success probability to 3/4. The independent fair-coin sensor has the same outcome entropy but does not improve the goal policy, and is rejected by the optimizer when it has a positive cost.

**Verification:** independently compare original full eight-state transitions and compiled four-coordinate transitions for all 1+6+36+216=**259** operation words up to length three. The audit covers goal readout and both sensor-outcome distributions; a separate full-state Bellman oracle also agrees with the reduced planner at horizons zero through three, including selected post-observation histories. Modified goals, self-rehashed incorrect operators, mismatched snapshots, invalid non-passive sensors and inadmissible operations are refused.

## Reproduce

\`\`\`sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_intention_predictive_ground.py
python -m examples.adaptive_intention_lab
\`\`\`

- Core: [\`src/vli/intention_predictive_ground.py\`](../src/vli/intention_predictive_ground.py)
- Independent tests: [\`tests/test_intention_predictive_ground.py\`](../tests/test_intention_predictive_ground.py)
- Machine-readable exact-rational lab: [\`examples/adaptive_intention_lab.py\`](../examples/adaptive_intention_lab.py)

All file-pins, outcome labels, goal definitions and operation costs belong to the source/ground identity. Snapshot validation replays the *compiled* Ground, not the original source; the external expected digest must be held independently.

## Active development — translation into living systems

1. **Source discovery rather than supplied laws.** Use VGI's evidence finalization and existing source-observation selectors to construct candidate controlled matrices from admitted measurements, then challenge them out-of-sample. A finite sample cannot certify an arbitrary stochastic law without structural assumptions or error bounds.
2. **Physical E-CPU realization.** Bind suggested sensor and actuator operations to receiver-approved observer-host capabilities and authenticated receipts. \`IntentionSession.apply\` is **a prediction inside a model**, not an external act; even a matching receipt hash cannot by itself prove a physical effect. Compare a measured post-intervention state against the Ground's prediction and record material realization distinctly.
3. **Intention-relative processor migration.** Connect changing goals to PFI-001 / TDL-002 / TDL-003 rank-refinement witnesses, allowing an intention to require further source observations without discarding valid prior structural memory.
4. **Pretrained model observational basis.** Apply the observation/action distinction to real causal pretrained-model experiments (e.g. a language model choosing sensor/tool use vs generating a token). The present experiment establishes a controlled finite mathematical mechanism, **not trained LLM behavior, independent physical action, or a model-independent efficiency law**.

### Mathematical throughline

**An intention can form the relevant predictive coordinate system and select evidence-gathering or state-changing operations within it.** The Virtual Layer performs the interpretation; observer-hosts and material paths must separately qualify and realize effects.
