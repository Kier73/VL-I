# TIME-001 — An environment-derived interpreter with continuous causal lineage

**Status:** exact finite laboratory implementation. Not a self-hosting language model, autonomous long-running service, physical sensor proof, or general solution to continual neural learning.

## Research question

> Can an intelligence's environment generate its computational language, while its interpretations change without its ongoing causal identity restarting?

A continuously evolving intelligence must retain what it observed, what it inferred from that observation, the evidence scope of that inference, when the inference stopped fitting, and how its interpretation changed. This is different from merely putting model weights and context in a checkpoint.

TIME-001 adds [temporal_intelligence.py](../src/vli/temporal_intelligence.py). The first environment-derived language is deliberately small and exact: an interpreter for vector affine flows over rational time. Its **coefficients and executable processor capsule** are constructed from observed source coordinates, not supplied as hidden prior rates. A fixed, bounded host interpreter remains part of the bootstrap.

## 1. State and time

Each observation is a tuple of an externally identified environment, an increasing **exact rational timestamp**, a vector of exact observed rational values, a digest identifying the supplied source evidence, and an observing host label. All accepted observations form one hash-linked causal history.

For an observed epoch, derive a candidate flow from two distinct samples:

\[
\widehat x(t) = x(t_0) + v(t-t_0),\qquad
v = \frac{x(t_1)-x(t_0)}{t_1-t_0}.
\]

A *third* distinct observed point must agree exactly before the Ground admits this as an **empirical qualified processor** for the observed epoch. All subsequent points in the same epoch must agree. The interpreter program is ordinary data:

~~~text
ADD(INPUT(origin),
    MUL(INPUT(rate),
        SUB(INPUT(time), INPUT(anchor))))
~~~

The parameters rate and origin are **derived from the environment's observations**. The five operation nodes and their host evaluator are supplied bootstrap semantics, not generated from scratch. The generated program is executed through the fixed bounded AST evaluator and independently checked by a direct exact-rational mathematical implementation.

For each qualified epoch, a program can address any rational time coordinate within its *sampled span* without replaying earlier observational transitions. This is a **model interpolation**, not proof that a physical world followed that law at unobserved instants.

## 2. Continuity of identity while interpretations change

Let the persistent system identity be G. Each observation extends one causally ordered log:

\[
H_{n+1} = \operatorname{Hash}(\mathrm{schema}, H_n, G,
                        t_{n+1}, x_{n+1}, \mathrm{source}, \mathrm{host}).
\]

A change in environment need not replace G. Instead the representation becomes an ordered set of epochs, each retaining its own empirical law, observed interval and witnesses. A new observation contradicting the current law closes that **interpretation epoch** at its last consistent observation. The disagreeing observation starts a new epoch whose dynamics must again be qualified. Old epochs and their witnesses remain addressable. A change detected between observations at two times is **only known to have occurred somewhere in that interval**; TIME-001 does not invent a precise physical change instant.

For example:

~~~text
time:      0   1   2   3   4   5   6   7
observed:  1   3   5   7   9  30  31  32
host:      A   A   A   A   A   B   B   B
~~~

The first qualified processor has rate 2; the second, following a detected change, has rate 1. Their causal observation history shares a single identity and extends a single head, including a change of observer-host. Recovery on B requires independently trusted expected_head and expected_identity.

The literal time coordinate is **continuous rational mathematical time**, not evidence of a continuously running host process. There may be arbitrary gaps between observations, including periods when no host runs. TIME-001 never asserts that it perceived the environment during those gaps.

## 3. Claims and refusal conditions

| Result | Truth condition |
| --- | --- |
| EXACT_RECORDED_OBSERVATION | Exact time/value pair is present in the externally pinned source-sample ledger |
| UNVERIFIED_INTERPOLATION | Exact execution of empirically fitted law *inside* observed epoch |
| UNVERIFIED_FUTURE_PROJECTION | Fitted law outside observed epoch, explicitly opted into |
| CANDIDATE | Two observations support a unique slope; a third has not qualified it |
| INSUFFICIENT | New epoch has only one observed point |

Refusals cover changed lineage heads, wrong environment identity, changed schema, malformed/noncanonical rational timestamps, duplicate or non-increasing time, unadmitted dimensions, insufficient observations, cross-epoch interpolation, implicit extrapolation, contradictory witness bundles, and forged processor/navigation receipts.

The source evidence digest and host label are **not authenticated signatures**. A malicious source can lie about observations; an attacker controlling both the entire ledger and external expected head can rewrite history. For actual distributed deployment, bind source receipts to E-CPU's authenticated host/provenance and quorum-fenced lineage. This first adapter is deliberately non-effectful.

## 4. Relationship to existing research

- **VL-I SHAPE-003/004:** learned observer morphology and verified structural prefix memory; TIME-001 adds regime-preserving evolution of an interpreting Ground over a chronological stream.
- **VL-I NAV-001/003:** qualified neural state navigation and finite-horizon observer-relative equivalence; TIME-001 navigates a generated empirical *time law*, not original Qwen causal-state equivalence.
- **Navigable-Compute NCI/NCS/NCR:** task invariants construct processor algebra and same-language reflective interpreter; TIME-001 should next lift learned epoch capsules into that already implemented reflection machinery.
- **VLD ContinuousVirtualSystem:** multi-Ground execution, recovery, independent stage verification; TIME-001's lineage and witnesses can become a typed continuous stage.
- **E-CPU/VGI:** authenticate host evidence and distinguish observed data from admitted interpretation using existing authority and Ground protocols.

This complements continual-learning research on nonstationarity: retaining history and adapting the active model are different objectives. We track them separately before adding trainable neural behavior.

## 5. Verification

~~~sh
python -m unittest discover -s tests -p 'test_temporal_intelligence.py' -v
python -m examples.temporal_intelligence_demo
~~~

Tests cover exact multidimensional rational flow, differing generated processor epochs, recovery onto another labeled host, false external pins, unobserved gaps, source digest admission, JSON digest recomputation, recomputed-but-false processor capsules, and projection truth status.

An exact linear fit is not scientific proof of a physical law. A quadratic process sampled at two points can produce the same initial candidate; a third point provides a counterexample test, not universal identification. Piecewise-affine memory is not general intelligence.

## 6. Next construction steps

1. **Source and observer Ground:** adapt real authenticated E-CPU receiver receipts and VGI three-valued observation qualification.
2. **Generated language, not only coefficients:** compile qualified laws into Navigable-Compute's NCS algebra and test NCR-001 same-language self-interpretation of the environment-specific program. Measure evaluator costs.
3. **Adaptive expressivity:** admit competing dynamics (recurrence, switching laws, neural operators and symbolic invariants); use active attention to choose distinguishing observations and retain counterexamples.
4. **Actual continuous hosting:** run a real event pump with monotonic clock, source heartbeat, failure detection and independent checkpoints; distinguish unobserved intervals from present-but-uninformative events.
5. **Preservation without fossilization:** retain old qualified invariants while allowing new observers, action vocabularies, representations and host topologies. Benchmark adaptation lag, invariant retention, memory growth, calibration and actual work.

**First-phase criterion:** the same environment-identified process can derive an executable law, detect that it no longer fits, generate a different law without erasing the old one, and survive a host change while keeping historical claims independently checkable.
