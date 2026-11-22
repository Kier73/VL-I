# TIME-010 — Native execution of a source-discovered rational processor

**Research status:** bounded, source-authenticated real native E-CPU
Virtual Layer execution and quorum-fenced pure result publication. The
research goal is to close the **semantic gap** exposed by TIME-009:
matching a separately produced E-CPU value is not the same thing as
having E-CPU actually execute the law discovered by VL-I.

TIME-010 implements **the latter**. The mathematical law is still a
bounded exact linear rational recurrence qualified by finite source
observations, not an independently proven physical future or a general
neural model. The result publication is a pure computational semantic
effect, **not** the IPS-008 receiver's externally effectful register.

## Source → admitted semantic operator

TIME-008 reconstructs a signed Ed25519 source journal and the complete
TIME-001 observer Ground from externally pinned source heads. It derives
the smallest full-rank rational recurrence (bounded dimension 1–4), then
checks training, validation and untouched test intervals using independent
Cramer's-rule arithmetic. TIME-010 repeats this entire source check
**before every native execution**. A digest alone cannot admit a program.

The real executable law is a finite companion-matrix interpreter
instantiated from the source-approved matrix, initial state and selected
coordinate. TIME-010 registers an `OperatorSpec` for
`vli.rational_companion@1` with the original E-CPU `OperatorRegistry`.
It binds an exact local handler that performs its **own** rational
matrix multiplication and binary exponentiation under a 4096-bit
intermediate budget. This is a different numerical path from the
TIME-007 navigation evaluator and its source-level self-interpreter;
the source-derived transition law is the common specification, not a
precomputed output supplied in the handler.

The receiver creates and admits an original E-CPU `SystemSeedEnvelope`
and `VirtualGroundRef`, locally spawns a kernel process, and processes
a typed `VirtualLayerWorkload`. The original
`VirtualLayerRuntime.execute_operator()` performs receiver-local
Hypervisor capability checks **before** invoking the handler. The
workload carries no source code and cannot grant itself `compute`.
The native resulting
`virtual_layer.operator_result` includes the actual operator contract,
workload and result digests and a host-local event fingerprint.

The native output carries the exact rational value, all inferred
state coordinates, a logarithmic matrix-product count and the
original qualified source and observer heads. TIME-010 demands
independent equality with `navigate_discovered()` and a verified
local kernel journal. Two distinct logical receiver hosts are tested
against the same exact source-derived law.

## Quorum-backed pure semantic publication

TIME-010 deliberately does **not** invent a new consensus or recovery
protocol. Its `FencedNativeRationalProcessor` creates a native E-CPU
`TransferableExecutionClaimStore` backed by the already available
`RecoveryLineageAuthority`, `RecoveryLineageCoordinator` and
`SeedCommitMembership`, with three independently journaled members
in one local filesystem.

Every claim is bound to:

- externally selected source Ground digest and exact signed publisher
  head; qualified operator `OperatorSpec` and implementation identity;
- a fixed intention coordinate, rational output selection, and explicit
  future-model-projection permission; and
- current native E-CPU owner peer, host and assignment epoch.

After an accepted reassignment, an obsolete host's existing genuine
execution candidate cannot publish. A replacement host requalifies the
same source, executes the law in a new E-CPU HeadlessOS/Hypervisor, and
proposes a new candidate. Before quorum completion the owner checks the
native E-CPU claim, exact source semantics, candidate shape and hash,
**and re-executes the pure law through the Hypervisor** to avoid accepting
a source-hashed but fabricated event. The original native claim store
then completes the current result. Reopening the owner reads the
original live majority and independently reconstructs the candidate.

The independently checked candidate is durably staged **before**
the native E-CPU majority commits. This makes the originally discovered
source law and exact native work recoverable if the coordinator is
hard-killed at the particularly dangerous crash boundary *after* native
quorum completion but *before* returning its result. A dedicated subprocess
test uses an immediate `os._exit(73)` after the native majority commit;
the parent reopens independent quorum journals and reconstructs the
accepted source-qualified result. This is a tested pure-result recovery cut,
not an atomic remote actuator transaction.

The proof survives a fresh Python process reading the same immutable
source inputs and durable quorum journals. This is evidence of
**preserved mathematical identity and accepted pure results across
host reassignment**, not a guarantee of uninterrupted physical
computation or a physically distributed majority. Since pure work may
be replayed during validation, no at-most-once physical CPU execution
is claimed. Claim ownership governs result publication only.

## Reproduction

A pinned E-CPU version is installed from
[commit `25047fee`](https://github.com/Kier73/E-CPU/commit/25047fee033b84c6cb80bbcda862741d9792bcf2),
which includes the receiver resource-lifetime fix discovered in TIME-009.

```sh
python -m pip install -e .
python -m pip install -e /path/to/pinned/E-CPU
python -m pip install 'cryptography>=46,<47' 'pytest>=8'
python -m unittest discover -s tests -p test_native_rational_processor.py -v
python -m unittest discover -s tests -p test_discovered_state_space.py -v
```

The new CI matrix runs Linux/Windows × Python 3.11/3.13 and rechecks
E-CPU's upstream transferable claim and Virtual Layer tests.

Adversarial cases include: signed source with self-rehashed changed
coefficients, changed external publisher head, missing Hypervisor
capability, unrequested future projection, different original vs.
replacement host, valid stale old-host candidate, result substitution
with a recomputed digest, wrong source/coordinate on recovery, and
readback of a quorum-completed result from a different Python process.

## Actual progress and still-open proofs

This is a more substantive integration than TIME-009. The native
`vli.rational_companion@1` handler computes the very mathematical
operator that TIME-008 inferred, inside the original E-CPU Hypervisor
runtime, and the original E-CPU quorum owns publication. The previous
stage-by-stage affine E-CPU program is no longer being presented as a
surrogate for this semantic execution.

Remaining research:

1. Use E-CPU's **original IPS-008 mTLS receiver** for a native
   `rational_companion` **effectful** publication, with a plan whose
   quorum and receiver verify the *source-derived operator itself*.
   This requires a compatible native IPS-005/008 plan and admission
   contract; the present pure claim store does not impersonate that.
2. Make semantic source-revision activation atomic with the
   old processor's authority retirement. Presently one source Ground
   identity is pinned for each claim-store root. An independently
   signed revision creates another processor identity; no automatic
   cross-revision authority is inferred.
3. Test separate physical majority authorities and remote worker
   endpoints, including network partitions and one-sided restarts.
   Current testing uses distinct journals and runtime contexts on
   local test hosts/processes.
4. Measure measured source acquisition time, native interpreter
   execution overhead, source verification work and transport,
   against matched mathematical baselines for longer coordinates.
5. Extend the learned mathematical grammar beyond finite scalar
   constant-coefficient recurrence families and test against
   original trained model intermediate states with external oracles.

These are active targets for further development, not claims this
stage proves them.
