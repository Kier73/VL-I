# TIME-007 — Intention selects a coordinate of a generated intelligence space

**Research state:** a bounded source-qualified exact rational environment
Ground, an intention-directed compact mathematical processor, and a
self-interpreter constructed in a declared five-form source language.
No claim of unrestricted artificial intelligence, physical-world future
predictability, unbounded reflection, continuous consciousness, or hardware
freed from external primitives is made.

## 1. Change the question from trajectory to coordinate

The environment is not something an intelligence must completely reproduce
at every instant before responding. Its current input includes an **intention**:
what mathematical state do we want to realize?

A qualified Ground retains a compact relation between observations and
candidate dynamics. The external observer's intention selects one
coordinate of that admitted mathematical state space. TIME-007 computes
that result by composing the relevant transformation directly, without
constructing every earlier temporal state or enumerating a finite
transition table in advance.

This is a significant change relative to TIME-006, which precomputed
a finite numeric trajectory and then offered a binary index navigation
over the compiled table.

The mathematical idea itself is classical: companion matrices, finite
differences, linear state-space realizations and exponentiation by
squaring. The testable research addition is the way VL-I admits the
mathematical form through signed environmental evidence and constructs
a reflective processor from that form, under an explicit provenance
and independent-verification contract.

## 2. Construct four compact mathematical processors

The four TIME-003-qualified law families admit the following finite
rational state-space realizations.

| Law family | Initial state | Transformation | Rendered coordinate |
| --- | --- | --- | --- |
| Affine | (x0, d) | [[1,1],[0,1]] | first |
| Geometric | (x0) | [[r]] | first |
| Quadratic | (x0,d,c) | [[1,1,0],[0,1,1],[0,0,1]] | first |
| Second-order recurrence | (x1,x0) | [[a,b],[1,0]] | second |

The learned source coefficients are substituted into the declared
family architecture. A state at step k is described by A^k times
the retained initial state. The chosen component renders the intended
output. The recurrence's second value is an actual mathematical state
coordinate, not a discarded implicit datum.

The source program does **not** materialize the states at steps
1,2,...,k. It keeps a fixed-dimension operator and performs
binary matrix composition in O(log k) matrix products. It counts
rational additions and multiplications and refuses operands outside
a declared 4096-bit exact arithmetic budget. This logarithmic count
must not be interpreted as constant bit complexity or zero physical
processor work.

An intention can be a distant exact rational coordinate. For example,
a quadratic model qualified by 0,1,4,9 can mathematically render
step 1000000 as 1000000000000 through about 20 binary decision bits,
without constructing a million sample states. This is a **model
projection**, not evidence of what physically happened at that instant.

The classifier is limited to the four supplied model families, and
a held-out source sample is required. New mathematical operator
families have not yet been invented outside that grammar.

## 3. Construct the processor and an interpreter of its own language

The companion rational interpreter emits a source program for:
finite-dimensional matrix multiplication, applying a matrix to a state,
recursive binary powering, and rendering the desired component.

Five source syntax forms are admitted:
literal, argument, conditional, primitive application, and function call.

The same source language can express a generated evaluator with
function lookup, argument-list evaluation, primitive dispatch,
expression evaluation, function invocation and program entry.
A Ground-specific codebook transforms the program and evaluator into
encoded source; the evaluator can read and interpret that encoding,
including an encoded representation of its own evaluator.

Reflection depths are explicitly measured:
0 — direct program execution;
1 — program executed through generated interpreter;
2 — interpreter interpreting its own encoded evaluator while
    interpreting the mathematical program.

The actual host evaluator now uses **heap-resident activation frames**,
rather than Python's recursion stack. A distinct continuation-stack
implementation checks the same program and exact expression-step counts.
This corrects the first CI-discovered deep-reflection resource failure.

The physical seed is **not generated**: it includes tuple selection
and construction, equality and rest, exact rational add/multiply, and
integer half/odd. The semantic grammar itself was supplied. No
generalized neural architecture or universally optimal minimal
primitive basis is claimed. The new exact rational reflective language
is a separate implementation inspired by the earlier NCR-002 work;
upstream NCR-002 uses a different finite semantic operator table and
does not itself implement unrestricted rational arithmetic.

## 4. Causal source evidence does not imply physical future truth

Before admitting the mathematical processor, VL-I verifies each signed
TIME-005 publisher observation against the externally supplied expected
publisher head and count. The observed TIME-001 Ground and TIME-003
capsule must exactly match these records, and the resulting matrix
representation must independently reproduce each recorded sample.

The source publisher can still sign mistaken or fabricated values.
Cryptographic signatures authenticate who made the report, not whether
a physical system behaved that way. Intervals between discrete
readings are not observed. All future coordinates require explicit
projection consent and are labeled unverified.

The source Ground identity is preserved as a verifiable lineage.
The observer's subjective awareness, external actuator authority
and uninterrupted cross-host runtime are not established by it.

## 5. Run and falsify

Install VL-I and the optional cryptographic source verifier.

    python -m pip install -e .
    python -m pip install "cryptography>=46,<47"
    python -m unittest discover -s tests -p test_intention_state_space.py -v
    python -m examples.intention_state_space_demo

The Linux/Windows Python 3.11/3.14 CI additionally runs TIME-005 real
process-kill recovery tests and TIME-003 language-change tests.

Independent refusals cover wrong signer, rolled-back source head,
wrong Ground identity, wrong transform after digest recomputation,
invented output, non-aligned time coordinates, unsupported future
prediction, over-budget coordinates, unknown source opcodes and
a separately checked source-level two-level reflection.

## 6. Research remaining

The next frontier is to allow the environment's **new observations
and invariant violations** to generate a *previously absent*
representation family instead of selecting one of four supplied
templates. This should be assessed on external changing source
observations with chronological held-out tests, typed receipts
and independent mathematical reconstruction.

Another frontier is to optimize reflection itself. A compact physical
mathematical operator may be cheap to navigate while a naive
metainterpreter is costly. Preserve that distinction and measure
source program size, actual host arithmetic, interpreter work and
cross-host recovery separately.

The design question for future agents is now concrete:

**Can an observer transform its currently trusted relation to the
environment into a processor that directly realizes the incoming
intention, without rendering the entire intervening state-space?**
