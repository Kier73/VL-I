# ACT-001 — Authenticated active rational observer inquiry

**Milestone:** two source-qualified, partially uninformative rational observer
channels identify one shared order-two recurrence through rank-aware probe
selection. This is an executable, bounded experimental transfer of
[VLD TRACE-002's active rank inquiry](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/docs/TRACE_002_ACTIVE_OBSERVER_INQUIRY.md)
to **exact rational values backed by publisher-signed observations**. It
reuses VL-I's TIME-005 Ed25519 journal and TIME-008 Gaussian/Cramer exact
arithmetic; it does not claim the VLD finite-field implementation itself was
imported, or that finite observations prove an external physical law.

## Why separate observers matter

For a stipulated shared order `d`, each channel's next regressor row is
formed solely from its *already admitted* historical values:

`r_t = (y_(t-1), y_(t-2), ..., y_(t-d))`.

Before requesting its next reading `y_t`, VL-I can rank the prospective
row against previously observed, authenticated cross-observer equations.
The selector chooses maximum exact guaranteed one-step **rank gain per
observation cost**, with deterministic cost/channel tie breaks. A zero-gain
`BRIDGE` query is explicitly distinguished; such a query might enable a
future rank-increasing regressor but has **no guaranteed gain**. This is a
greedy, not globally optimal, information acquisition policy.

The signed channels agree in sample-time origin and spacing. Future
measurements and hypothetical outcomes do not enter the selection decision.
Only after a receiver sees a new externally pinned publisher head and checks
the entire Ed25519 journal does its response enter an accepted row. The
record includes the original prior head, exact regressor, signed packet digest,
observed rational response, rank, cost and a chained session identity.

An independent auditor **replays from independently supplied complete
publisher heads**; it reconstructs all initial prefixes and subsequent
observations, checks every selection with determinant-minor rank rather than
the producer's Gaussian pivots, and refuses altered decisions, values,
source heads, costs or session contents. Recomputing a digest never creates a
valid missing signed packet.

## Test: two individually unidentifiable observers

A declared shared recurrence of order two over exact rationals has
coefficients `(3,-2)`. Observer A sees the constant sequence
`1,1,1,1,...` and B sees `1,2,4,8,...`. Neither observer individually
provides an order-two full-rank regressor: each has rank one.

Initially both have exactly two authenticated samples. Costs are one unit
for A, two for B. The planner sees two rank-one offers and selects A
because of its lower cost. After receiving A's newly signed `1`,
A's next row cannot increase rank, whereas B's next row can. It therefore
selects B before seeing B's `4`.

The two signed observations provide independent rows

```
[1 1] [a b]^T = 1
[2 1] [a b]^T = 4
=> [a b] = [3 -2]
```

The exact source-checked law is separately challenged by two *additional,
previously unobserved* signed values from **each** channel. The producer
uses exact Gaussian elimination; the checker uses Cramer's rule and
independently replays each full channel. A signed but contradicting
observation is a genuine model refutation, not a reason to assert the
original predictions as observed reality.

This setup uses **two new acquisition requests** and cost `3` for full
rank. Uniformly acquiring more samples from A alone would never identify
the common two-coefficient recurrence, despite additional work. This is
a designed control, not a measured improvement for arbitrary sources or
large neural-host costs. There is no claim that the source signer's
synthetic mathematical values are measurements of a physical instrument.

## What is still to solve

- **Unknown order:** ACT-001 declares a shared candidate order. TIME-008
  separately discovers a smallest order from a single signed scalar trace,
  but this code does not yet adaptively optimize which order to test across
  heterogeneous observers.
- **Physical source quality:** Signed publisher receipt verifies authorship,
  not truth; separate hardware, external measurement semantics, sensor clock
  alignment and precision proofs remain to be built.
- **Real query costs:** The cost in this experiment is a declared exact
  rational budget, not measured trained-model runtime, energy or transport.
- **Learning morphology:** The common recurrence is a behavioral law, not
  evidence of a novel full pretrained Transformer equivalent. Connecting to
  SHAPE-010 original causal state and VGI active attention requires new
  qualified source events and independent numerical oracles.
- **Multi-host rights:** No effectful actuation, distributed quorum, or
  cross-host authority is created by this Ground.

## Reproduction

```sh
python -m pip install -e .
python -m pip install 'cryptography>=46,<47'
python -m unittest discover -s tests -p test_active_rational_inquiry.py -v
python -m unittest discover -s tests -p test_discovered_state_space.py -v
```

The four-job CI matrix runs Linux/Windows and Python 3.11/3.13.
