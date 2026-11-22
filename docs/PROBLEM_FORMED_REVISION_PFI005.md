# PFI-005 — Source-equivalent SAT interpreter revision on a fenced E-CPU fabric

**Status:** experimental cross-repository integration. The executable implementation is
`src/vli/problem_formed_revision.py`, the existing native host runtime remains
`src/vli/problem_formed_sat.py`, and the test is
`tests/test_problem_formed_revision.py`. The CI workflow pins IFS and E-CPU
source commits. This work does not resolve P versus NP.

## Research question

Can **the original SAT problem generate an improved interpreter using the
existing SAT machinery**, and can a previously qualified E-CPU execution route
realize that new processor without confusing (1) mathematical identity,
(2) interpretation identity, and (3) current right to publish?

This is intentionally an integration of *already implemented* functionality:

- [IFS SFI-002/003](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/pnp-ground-navigation/SFI-003.md) discovers exact source-backed finite Boolean functions and invokes its own executable SAT interpreter to improve a bounded gate selection; `morphic_verifier` independently checks the source/Ground semantics and `recursive_sat_compiler.audit_refinement` rechecks the candidate-pool optimality statement by finite enumeration.
- [VL-I PFI-004](PROBLEM_FORMED_SAT_PFI004.md) already uses E-CPU `OwnedNetworkFabric`, `NavigableComputationDAG`, `VirtualLayerRuntime`, local Hypervisor rights and `TransferableExecutionClaimStore` to execute original CNF SAT partitions under majority-published claims.
- [E-CPU IPS-005/006](https://github.com/Kier73/E-CPU/blob/main/docs/DISTRIBUTED_GENERATED_GROUND_IPS006.md) already separates speculative pure execution, authoritative current assignment and durable recovery. This experiment does **not** create another quorum or a synthetic executor.
- [VGI ProofGround](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/main/virtual_ground_interpreter/docs/PROOFGROUND_NAVIGATION.md) provides a useful future-facing vocabulary for preserving independently verified proof obligations. No VGI ProofGround is falsely claimed to have certified this integration.

## Proven semantics before physical realization

SFI-003 derives two source-equivalent Ground descriptions for the **same**
CNF `F(x)`. Each is an acyclic Boolean gate network plus an exact residual
clause set. Its independently checked decoder maps every free register
coordinate to one full original-source assignment satisfying the extracted
gates, and every original solution is represented by exactly one coordinate.

For each revision `r`:

```
F(x) iff (decoded_gate_relations_r(x) AND residual_r(x))
#SAT(F) = sum_z [F(decode_r(z))]
```

The original CNF source digest is constant. The Ground/interpreter digests,
register counts, partition topology and native claim identities differ.

`create_revision_source` uses the already-existing SFI-003 meta-SAT
constructor and its independent auditor; it **refuses** sources that do not
support an actual verified architectural improvement. This is finite
self-application: the candidate-function pool and the meta-search are bounded,
and the claimed optimum is only within the source-discovered finite pool.

`translate_source_witness` carries a **source-validated** model between old
and new coordinate systems by projecting that same full assignment onto the
new free registers and checking independent reconstruction on *both* Grounds.
It refuses assignments that violate the original CNF. It does not invent a
bijection between arbitrary unexecuted old and new coordinate positions.

## Using the existing native E-CPU implementation

`DirectSatFabric` has one backwards-compatible, optional **trusted local**
runtime adapter. Its default remains the original SFI-001 mathematical
interpreter. `RevisedSatFabric` supplies a seven-function adapter for the
original SFI-002/003 generated Boolean Ground: source-ground formation,
independent reconstruction, real partition evaluation, independently replayed
segment checking, full-coverage aggregation, canonical hashing and encoding.

Native E-CPU packet dispatch, original DAG, per-stage receiver-local
Hypervisor authorization, current majority-backed claim epoch, speculative
candidate handling, durable checkpoint verification, pure-operation replay,
and terminal aggregation all remain in **the original PFI-004 class**. The
adapter does not grant rights or declare a peer's source text executable.

**No unqualified cross-revision claim transfer is attempted.** A new
interpreter creates a distinct semantic work identity. The native checkpoint
for the baseline refuses reopening as the refined program. A segment receipt
for the old coordinate interval cannot be accepted as one for the new
interval. A correct candidate calculated by a worker with a *stale assignment
epoch within the revised Ground* is refused by native E-CPU claim fencing.

Old work cannot be magically reinterpreted as a current quorum claim. This
experiment uses separate native authority roots for concurrently exercised
baseline and revised interpreters. It does **not yet implement** a single
global activation lease that prevents an intentionally still-running old
processor under its *own* independent root from continuing to publish.
A true in-flight cutover requires one shared authority policy that retires
old-Ground assignments before admitting the new ones.

## Controlled direct SAT instance

Use the existing SFI-003 6-variable `REPAIRABLE` source fixture: the
greedy interpreter has 5 free registers; the SAT-refined Ground has 4. The
source has **five** solutions. Both interpreters must recompute the original
CNF under their own distinct coordinate spaces.

| Same original CNF | Baseline | SAT-refined |
| --- | ---: | ---: |
| Free Boolean registers | 5 | 4 |
| Complete COUNT coordinates | 32 | 16 |
| Partitions at 8 coordinates per stage | 4 | 2 |
| Exact #SAT (expected) | 5 | 5 |
| Mathematical source identity | same | same |
| Semantic interpreter / claim identity | different | different |

These are *exact domain sizes*, not yet measured wall-time gains. Constructing
the refined interpreter and independently rechecking its meta-CNF costs real
work. The test runs the **real** E-CPU Hypervisor and native claim store, then
reopens the same Ground from durable state between partitions. Additional
tests cover source/Meta-SAT/ground forgery even after rehashing, source witness
translation, wrong-revision segments, partial coverage, altered segment
counts, capacity refusal, old checkpoint refusal, stale worker publication and
unchanged PFI-004 default execution.

## Reproduction and bounded scope

The matrix workflow `.github/workflows/problem-formed-sat.yml` pins:

- IFS: `d2022bf811dc89a9ae69a266148847bd443fa256` (SFI-003).
- E-CPU: `9064559d6e9b621034953b7135fc991a436d7746`.
- Navigable-Compute: `559a1a813012d13ed37c04e2a117467e34ed3cb1`.

From a checkout with those dependencies installed and
`PYTHONPATH` set to IFS `experiments/pnp-ground-navigation`:

```sh
python -m unittest discover -s VL-I/tests -p test_problem_formed_revision.py -v
python -m unittest discover -s VL-I/tests -p test_problem_formed_sat.py -v
```

Execution occurs on ordinary CPU-backed E-CPU software hosts and a shared
filesystem with three journal files. This is not physical in-fiber computation,
remote adversarial-machine proof, cryptographic attestation of independent
hardware, nor an end-to-end general polynomial SAT algorithm.

**Next research step:** a shared native authority-approved Ground activation
transition (retire old, qualify new), with no window where an obsolete
interpreter could publish as the current semantic computation. Only after a
source equivalence proof and authority transition may old source witnesses be
reused, never old model-count partitions with different coordinate meanings.
