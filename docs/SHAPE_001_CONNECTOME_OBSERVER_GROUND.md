# SHAPE-001 — Connectome-inspired observer-relative Grounds

**Research question:** can the shape of a computational system be changed to preserve only the distinctions relevant to an observer's intention, rather than mechanically replaying the original internal network?

**Status:** executable research integration, verified against actual In-Formation-Space and VLD predecessor code.  
**Implementation:** `src/vli/connectome_shape.py`  
**Tests:** `tests/test_connectome_shape.py`  
**Measurement:** `examples/connectome_shape_demo.py`  
**CI:** `.github/workflows/vli-connectome-shape.yml`

## The predecessor throughline

The key finding is not simply that sparse graphs use fewer edges.

[In-Formation-Space real connectome research](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/real_connectome_ground) has a source-pinned 448-label, 7,379-synapse OpenWorm wiring Ground and a series of follow-on experiments into sparse transferred states, nonlinear temporal navigation, evidence-qualified host policies and learned **observer-relative quotients**.

[IFS RCG-006 observer quotient](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/real_connectome_ground/continuity/observer_quotient/README.md) learns groups of states with equal future behavior for a particular observer and independently checks an exact whole-edge intertwining equation `P M = Q P` before accepting a lower-dimensional mathematical Ground. Its recorded synthetic stress model contains **131,072 microstates and 1,048,576 directed links**, but only **16 observer-relevant equivalence classes**, with the qualification and discovery costs explicitly included. It does not claim to recover animal neural firing, arbitrary learned language intelligence or a quotient for every observer.

[Virtual-Layer-Dynamics](https://github.com/Kier73/Virtual-Layer-Dynamics) already studies the same shape question through predictive Moore/Myhill–Nerode equivalence, structural operator contraction, sparse MoE realization and navigable computation DAGs. The relevant modules are [`triad.py`](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/src/vld/triad.py), [`accel`](https://github.com/Kier73/Virtual-Layer-Dynamics/tree/main/src/vld/accel), and [`models/frontier.py`](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/src/vld/models/frontier.py).

**The research opportunity for VL-I:** computational shape can itself be a Ground, adapted to the observer's intended result. The shape of computation need not be a permanent copy of today's Transformer architecture.

## What this experiment actually does

VL-I imports the original IFS code; it does **not** add a new parallel quotient-learning authority.

1. Load the exact original source file from IFS, which checks the OpenWorm Git blob `89ffc2a2d9e1934ddc462d51032acead5e4cfc91`, the source's **448 labels**, and its **7,379 observations**.
2. Use the original source's *connectivity statistics* to construct a separate, explicitly **synthetic** sparse computational Ground over a prime field. This new Ground is not the real biological network's trained dynamics.
3. Apply IFS's original temporal-response learner to infer an **observer-specific quotient**. Independently verify it across every synthetic directed edge before admitting its Ground.
4. From the observer's intention, locate a **VL-I state-position coordinate** before calculating the future observation. The coordinate identifies the relevant observable position, not a numerical answer.
5. At short horizons, run both the certified quotient and a full synthetic microstate reference, compare every observer-class value and measure their costs.
6. At distant times, navigate using IFS's certified polynomial/nonlinear Ground without enumerating full microstate trajectories.
7. Form a new observer-dependent page, text and UTF-8 byte-token response through the existing VLD-backed VL-I renderer, keeping the outer state coordinate and original source/quotient evidence.

No code paths claim that the real OpenWorm connectome's unaltered physical/neural dynamics have been learned. The source is a **heterogeneous topology** used to form and test a mathematical model.

## Direct measured result — November 19, 2026

[GitHub Actions SHAPE-001 run 44641636327](https://github.com/Kier73/VL-I/actions/runs/44641636327) passed **11 tests** and uploaded its machine-readable measurement.

| Observable | Measurement |
| --- | ---: |
| Source CSV | 245,463 actual bytes |
| Source synapse records / labels | 7,379 / 448 |
| New synthetic microstate network | 256 states, 2,048 directed relations |
| Independently certified observer quotient | **8 states** |
| State dimension ratio | **32×** |
| Five short test queries | **All exact matches** against original microstate readouts |
| Measured quotient query time (5 combined) | **0.629 ms** |
| Measured full microstate reference time (5 combined) | **22.412 ms** |
| Query-only reference/quotient time ratio | **35.63×** |
| Source admission | 51.3 ms |
| Synthetic Ground construction | 0.95 ms |
| Observation signature discovery | 6.37 ms |
| Whole-edge independent proof | 6.57 ms |
| Combined preparation | **65.18 ms** |
| Far-future logical coordinate | 1,099,511,627,849 |
| Distant navigation measured query time | **2.69 ms**, 44 polynomial products |
| Distant full microstate vectors constructed | **0** |

All timing observations are from one GitHub-hosted Python 3.11 run on a **small synthetic model**; they are not large-model benchmarks or evidence of energy savings. Full source admission and quotient proof were **not free**. The combined five short queries cost about 0.63 ms *after* preparation versus about 22.41 ms for the full microstate reference. Including preparation, the quotient path cost about 65.81 ms, which is **slower for only those five queries**. At identical average query costs, roughly 15 qualifying queries would amortize the one-time preparation, but that is a projection—not a separately measured repeat-run result.

The main value is that **all downstream admissible states** are represented exactly within the smaller observer Ground, not that one Python timing has established an LLM performance advantage.

## The constraint that makes this interesting

An observer-relative quotient is only valid for the observation and future relations under which it was proved. A new task, changed model, additional observer channel or newly meaningful distinction can force the Ground to **split** its previously merged classes.

That is why a promising intelligence architecture may be malleable rather than merely sparse:

- Merge what the observer cannot distinguish.
- Retain what affects future behavior or permitted tool/action semantics.
- Split the topology when an intention reveals a new necessary distinction.
- Pay for observation/proof only where it can justify reusable capabilities.
- Record the complete cost of discovery, verification, memory, communication and realization.

In a language model, the analogous research questions concern attention, KV state, memory, expert routing, code production, user intent and tool history. Nothing here proves that all original pretrained weights can be represented by a small exact observer quotient.

## Next research stage

**SHAPE-002 — Cost-qualified, intention-driven shape selection.** Combine this result with E-CPU's existing intention-sufficiency and path cost machinery and VLD's proof-producing intention compiler. Allow full-graph, sparse-graph, quotient and generative descriptions to compete under observer-specific semantics and full amortized cost.

**SHAPE-003 — Observer-driven morphogenesis.** When a new request distinguishes previously equivalent states, refine/reform the quotient, test the new output on held-out observations and keep both original and revised proof identities.

**COORD-003 connection.** Apply this idea to the content-generating Ground itself: discover which semantic state, syntax/token state or causal memory relations are necessary to form a requested paragraph, code program or tool response. Optimize the **shape of interpretation**, not merely matrix operations.

**Recursive interpreter connection.** The Virtual Layer interpreter's own state and operation laws should be candidates for observer-relative quotient and computational-shape revision, reducing the cost of *interpreting* model information.

The full ambition is that **the observer's intention selects a navigable computational shape**, and that this shape may be far cheaper than the inherited model architecture without losing the observations the intention requires.

**Provenance and rights:** Original OpenWorm data are not rehosted in VL-I. The workflow checks out the pinned IFS source, preserving its upstream [OpenWorm license and attribution](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/real_connectome_ground/data/OPENWORM-LICENSE.txt).
