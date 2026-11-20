# TIME-004 — An environment-derived reflective processor

**Status:** bounded executable bridge between VL-I's verified environmental
observer and Navigable-Compute's *actual* NCI/NCS/NCR-002 processor +
constructed self-interpreter. It is not a self-hosting general neural model.

## The precise question

**Can the observed requirements of an environment construct both an
operational state-space processor and a mathematical evaluator that interprets
an encoded copy of itself, while earlier admitted interpretations remain
available as the environment changes?**

TIME-003 constructs one of four source-conditioned mathematical expression
languages (affine, geometric, quadratic, order-two recurrence). TIME-004
addresses a different, complementary problem: **how can an observer navigate
positions that were genuinely recorded under a qualified model and construct
the smallest explicit finite task processor for that intention?**

This is a direct integration with the already built
[Navigable-Compute NCR-002](https://github.com/Kier73/Navigable-Compute/blob/9572cdea273aa5e22b2f5d4a83296492a941eab5/docs/NCR-002.md),
not a reimplementation of its compiler or proof obligations.

## Source -> invariants -> processor -> self-interpreter

For a TIME-003 empirically qualified capsule and its original TIME-001
observation ledger, select a bounded epoch containing 3..8 genuine samples.
A source-derived task description contains:

- One finite state *per recorded sample position* (not per invented physical state).
- An L1 index geometry with movement only to adjacent recorded positions.
- Identity symmetry and source-family label preservation.
- All admitted recorded positions as legal initial positions.
- **A declared intention** to reach the final recorded source position.

The source samples determine the number of states and the qualified family.
The interpreter does **not** reconstruct hidden causal dynamics from this
geometry. The L1 adjacency and target-final-sample intention are externally
declared observer contracts. They are not asserted as measured physical
actuator relationships.

NCI-001 generates the rank-descending mathematical transition law from these
invariants. NCS-001 closes the law into a finite task-relative semantic
operator algebra. NCR-002 constructs a reflective architecture: it derives a
four-primitive seed in a finite recipe library, constructs the remaining
semantic macros and a source-specific encoding, and emits an interpreter AST.

The task program then runs in three modes:
1. Direct execution of the generated task program.
2. Execution through the generated reflective interpreter.
3. The generated interpreter interpreting **its own encoded source** while
   interpreting the task.

The independently implemented NCR-002 continuation-stack checker verifies
the resulting state and accounted expression steps at all three depths.
This is an actual bounded same-language self-interpretation test, not a
renamed Python eval.

**Source value semantics:** when the task reaches the intended index,
the result value is retrieved from the externally pinned TIME-001 event.
NCI/NCR has navigated the *observation position*, not run the
TIME-003 quadratic, affine or recurrent numerical law. That distinction is
deliberate and testable.

## Continuous change

In the demonstration a single TIME-002 live observer first sees samples
1,3,5 and qualifies an affine TIME-003 language. NCR-002 constructs and
self-interprets a task processor over those three observed positions.

The same live observer then sees 100,101,104,109. TIME-003 qualifies a new
quadratic language, and NCR-002 constructs a different processor and
self-interpreter from the new observation scope.

Both reflective processors are independently qualified. A separate source
migration certificate proves the first source ledger is an exact prefix of
the later one. The reflective succession receipt binds both generated
interpreters and the migration evidence.

This is evidence-preserving **succession**, not full behavioral equivalence
of the two architectures and not continuity of physical host consciousness.
A source name, hash or timestamp is not a remote attestation.

## Reproduce against the exact NCR-002 source

Python 3.11+; VL-I is zero-mandatory-dependency by default. NCR-002 is an
optional, pinned integration dependency for these tests:

~~~sh
python -m pip install -e .
python -m pip install 'git+https://github.com/Kier73/Navigable-Compute.git@9572cdea273aa5e22b2f5d4a83296492a941eab5'
python -m unittest discover -s tests -p test_reflective_environment_processor.py -v
python -m examples.reflective_environment_processor_demo
~~~

CI covers Linux and Windows, Python 3.11 and 3.14, including TIME-003 and
older temporal Ground regressions.

Negative controls refuse false source heads, changed parameter evidence,
forged reflective output even after rehashing, invalid syntax-encoding
coordinates, insufficient source evidence, excessively large Ground
construction and fabricated future-equivalence claims.

## What comes next

- **Operate the generated numerical law itself** through an algebra derived
  from the observed environment, rather than only an index-navigator over
  recorded positions. Prove exact correspondence with independent
  arithmetic, including real recurrence jumps.
- **Let the environment influence semantic architecture**, beyond the
  number of source states and a family label. Search over multiple
  grammars under separate training/validation and negative controls;
  retain the earliest discriminatory evidence.
- **Use actual authenticated E-CPU and VLD hosts**, supported by
  quorum-fenced process recovery, so real source evidence and interpreter
  continuity span hosts without trusting an in-memory Python object.
- **Test neural adaptation**, where the constructed interpreter chooses
  representations that retain held-out task competence over changes in
  environment, rather than simply fitting deterministic scalar traces.

This milestone does not solve the original full question. It creates a
verified composition where **the task constructs the processor**, the
processor's declared semantics support **construction of a reflective
interpreter**, and the interpreter can be changed without rewriting the
observer's recorded history. What remains is to generate and operate richer
semantic machinery from the environment itself.
