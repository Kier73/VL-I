# TIME-003 — Evolving the computational language while preserving the observer

**Research milestone:** Environment-conditioned generation of multiple executable
interpreter expression languages under a single source/observer identity.

**Status:** finite, exact, evidence-scoped mathematical experiment.
This does **not** demonstrate unrestricted processor invention, neural model
self-training, physically continuous sensing, universal self-interpretation,
or a host-free runtime.

## 1. Why TIME-002 was insufficient

TIME-001 derived an affine rational-time Ground and preserved the source
observation chain. TIME-002 operated one living interpreter across observations
and source-silence events. Neither could select an alternative operator
family when an environment expressed behavior outside affine dynamics.

A living intelligence should be able to revise **the mathematics it is
capable of interpreting**, not simply replace a slope parameter.

TIME-003 introduces
[environment-conditioned language construction](../src/vli/evolving_environment_language.py).
It uses the SAME TIME-001 observation Ground that TIME-002 operates;
there is no second hidden source of truth or invented sample.

## 2. What is constructed

Within one explicitly declared finite semantic search family, a source of
rational-valued scalar samples at equally spaced rational times can qualify
four different symbolic languages:

| Family | Source-derived structure | Novel program vocabulary |
| --- | --- | --- |
| Affine | x(n) = x0 + d n | STEP, CONST, MUL, ADD |
| Geometric | x(n) = x0 r^n | STEP, CONST, POW, MUL |
| Quadratic | x(n) = x0 + d n + c n(n-1)/2 | STEP, CONST, CHOOSE2, MUL, ADD |
| Order-two recurrence | x(n) = a x(n-1) + b x(n-2) | STEP, CONST, RECURRENCE2 |

The constants and selected expression tree are constructed from actual sample
histories. **The family library and trusted AST evaluator are supplied.**
No arbitrary bytecode, remote inference engine, learned neural weight
generation or unrestricted program synthesis is invoked.

A candidate family is admitted only after an additional, independently
observed sample agrees with the parameters derived from its initial training
prefix: affine and geometric require at least 3 samples; quadratic 4;
recurrence 5. This is one finite adversarial check against a simple candidate,
not proof of a universal physical law or robust generalization.

The declared family precedence is part of the contract. It is not a proof
that the selected model is uniquely best among all mathematical languages.

## 3. Continuous-time and regime semantics

Source time is exact rational; TIME-003 only admits **equally spaced**
samples to the current language, and its recurrence/specified operators run
at integral sample offsets. The uniform grid is an assumption that must be
witnessed; arbitrary environmental time cannot be silently coerced into it.

When a previously qualified model fails a new real observation, its epoch is
closed at the last supporting sample. The new observation begins an
unqualified segment until the environment supplies enough evidence for a
new model. Old source events, model witnesses and prior source heads remain
addressable and independently verifiable.

If no family fits after a bounded warm-up, the system marks the interval
**UNQUALIFIED** and searches a fresh segment. This is a refusal, not an
automatic license to claim an invented language.

Each answer explicitly says whether it is retrieval of an exact previously
observed source coordinate or an **UNVERIFIED_FUTURE_PROJECTION**.
Only the latter needs opt-in. A generated recurrence can evaluate a bounded
distant sample coordinate through exponentiation of a 2x2 companion matrix
rather than replaying every prior sample; an independent iterative reference
checks the same output. Counts include mathematical matrix work. Growth
limits and exact rational output sizes remain relevant.

## 4. The new preservation contract: migration, not equivalence

Let a prior source-ledger prefix be L0 with interpreter I0. New observations
extend it to L1 and may construct a different interpreter I1.

The portable migration receipt demands:

- L0 and L1 independently checked against **external trusted heads**.
- The exact event sequence of L0 is a prefix of L1, including source identities,
  timestamps, values, host labels and event digests.
- Both generated interpreter capsules independently reconstructed and
  checked against their own observed training AND held-out samples.
- No presumption that I0 and I1 predict the same future or share full
  internal state. Only the **historically observed sample scope** is preserved.

The migration is therefore a proof of **evidence-preserving interpretation
succession**, NOT a proof of full behavioral equivalence or continuous
subjective experience. It is also NOT an authenticated VLD quorum certificate.

A malicious actor controlling the source, external head pin and host still
can forge an entire physical history. The experiment explicitly relies on
the external head-provenance boundary already defined in TIME-001.

## 5. Demonstration: three languages in one living observer

The controlled environment emits these 13 source samples in a single
TIME-002 running interpreter:

- First: 1,3,5,7 -> **affine** qualified
- Then: 100,101,104,109 -> **quadratic** qualified
- Then: 1,1,2,3,5 -> **order-two recurrence** qualified

The TIME-003 learner owns the same in-memory Ground as the TIME-002 runtime.
It generates three different executable expression trees with different
operator vocabularies and preserves two verified migration receipts. The
final recurrence supports a separately checked prospective coordinate at
global time 40. That forecast is **not** an observation of the world.

Another experiment qualifies a geometric family from samples 2,6,18,54,
producing a **POW** expression absent from the affine language.

All experiments use exact rational arithmetic rather than float tolerance
thresholds and retain source evidence without silently altering earlier events.

## 6. Reproduction

~~~shell
python -m unittest discover -s tests -p test_evolving_environment_language.py -v
python -m examples.evolving_environment_language_demo
~~~

CI also executes the TIME-001 and TIME-002 tests for regression coverage on
Linux and Windows with Python 3.11 and 3.14.

Deliberately negative cases: one- or two-sample overqualification; affine
refusal on quadratic data; recurrence fitting only after held-out fifth sample;
irregular timestamps; unsupported streams; current-head mismatch; corrupted
source values; forged AST and answer despite recomputing digest; reversed
migration; and independent host restoration.

## 7. How this differs from prior art

Classical online change-point detection studies inference about switching
data-generating regimes. TIME-003 is not a Bayesian change-point posterior
and makes no statistical optimality claim:
https://arxiv.org/abs/0710.3742

Classical event sourcing preserves a canonical history so derived projections
can evolve and be rebuilt. TIME-003 applies that distinction to verified
interpretation scopes while retaining an externally pinned Ground identity:
https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing

**The research addition is compositional**: the sensor's observed behavior
determines which executable *mathematical language* is admitted, the live
interpreter changes its active language within one lifetime, and the transfer
from one interpretation to the next carries a claim-specific witness.

## 8. The next ambitious, falsifiable questions

**NCR-002 composition:** Translate the generated TIME-003 expression
semantics into Navigable-Compute's already implemented task-generated
monoid/reflection contract. Qualify a genuine newly constructed evaluator
against independent host semantics, rather than smuggling a Python interpreter
behind the capsule. Import only portable mathematics; retain model/source
provenance separately.

**Neural intelligence:** Move from exact deterministic scalar streams to a
source-qualified recurrent or neural observational Ground. Measure whether
new actions/observations require adding trainable representation operators;
preserve held-out task performance and earlier justified invariants.
A static pattern recognizer is not yet a self-generated intelligence.

**Real continuous sensing and host survival:** Connect TIME-002 to real
authenticated E-CPU receiver observations and VLD's independently signed
quorum-fenced checkpoint store. Run a bounded wall-clock experiment with
a live stream, forced process death, a second process and unmodified
physically observed sample timestamps; distinguish unobserved gaps and
deduplicated effects. Neither resumed computation nor remembered timestamps
would constitute uninterrupted consciousness.

**Cross-interpreter semantics:** Form a stronger refinement relation when
two languages overlap on a declared observer and future action alphabet.
Where preservation cannot be proven, retain both Grounds and the precise
missing proof obligations instead of substituting a digest match.

The gap between a small generated family and a genuinely original
self-interpreting intelligence remains the motivation for the research,
not a completed result.
