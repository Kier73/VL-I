# PFI-003 — A problem-formed interpreter with transferable execution claims

**Status:** bounded integration experiment, 21 November 2026. Builds on [PFI-001](PROBLEM_FORMED_INTERPRETER_PFI001.md) and [PFI-002](PROBLEM_FORMED_FABRIC_PFI002.md). This is not a general autonomous distributed computer, an arbitrary runtime placement compiler, or proof of physical in-network computation.

## What changed

The source problem continues to choose the interpreter architecture, its register dimension, generated XOR gates, active logical host partitions, and dependency structure. PFI-003 adds one new experimentally relevant property: **ownership of a pure stage's accepted result can move between hosts without changing the original source Ground**.

It does this by *composing the existing APIs*, rather than introducing a second scheduler, consensus service, or result ledger:

- `vli.problem_formed_interpreter`: source-bounded exact observer quotient and generated gate network, independently verified by Navigable-Compute.
- `vli.problem_formed_fabric`: real E-CPU `OwnedNetworkFabric` typed byte frames, native `NavigableComputationDAG` commits and local `VirtualLayerRuntime`/Hypervisor operator execution.
- E-CPU `TransferableExecutionClaimStore`: majority-published pure execution claim assignments, strictly increasing epochs, live proof verification, exact semantic/implementation binding and one accepted completion.
- E-CPU `RecoveryLineageAuthority` and `RecoveryLineageCoordinator`: three immutable member identities, three durable local authority journal files, and majority-root verification.

This composition addresses a concrete difference: **compute ownership is not physical ability to compute**. A delayed old host may have derived a correct candidate. After a quorum-reassignment to another host, that correct candidate is nevertheless not eligible to publish the semantic result. The current owner must independently qualify its local receiver, compute, and complete using the live native claim authority.

## Experimental protocol

For one pending generated stage, the parent request is bound to the immutable PFI epoch, interpreter Ground and job; the child request identifies the stage and both expected source-bound frame hashes; the computational address binds the exact generated stage and pure `OperatorSpec`. The `computational_ground_fingerprint` is the interpreter's own digest. E-CPU's claim ID is computed from these bindings, not from the host.

1. The E-CPU fabric dispatches a real typed frame. The source-ground reference semantics and native DAG must predict its exact content.
2. E-CPU's claim store assigns the pending pure stage to `(peer, logical-host, implementation, epoch)` under a fixed-membership majority root.
3. The logical host re-qualifies this exact assignment against the live quorum, binds the stage handler *locally* through a fresh E-CPU `VirtualLayerRuntime` and Hypervisor, and produces a **speculative** candidate. Calculation alone does not consume the E-CPU queued frame or commit the DAG.
4. The candidate is bound to the source frame, stage, interpreter identity, assignment epoch, declared output and local operator receipt. An immutable-by-convention local copy is durably written before completing the claim.
5. E-CPU's unchanged claim store atomically accepts one current assignment epoch and digest through the majority. Only then can PFI-002's local stage be replayed through the Hypervisor, consume the typed queued frame and commit the original DAG. The observer's terminal result remains qualified by the original description.
6. If process death occurs after the quorum commits but before the PFI checkpoint advances, the successor reopens the original Ground, independently validates the durable candidate and majority result, then **re-executes a pure stage** locally and advances the DAG; no side-effect operation is replayed.

The protocol is deliberately scoped to pure deterministic generated stages. A candidate is not a general signed attestation of a remote host; checksum and E-CPU journal verification are local evidence, not a cryptographic identity proof from an independent machine.

## Falsifiable tests

- Full PFI generated pipeline through E-CPU's native fabric and Hypervisor with a distinct live quorum claim per generated stage. The final observed result must equal the original declared source law.
- A speculative old-host candidate must be **correct but unpublishable** after native quorum reassignment to a distinct host. The replacement host completes the same semantic stage without changing the problem-derived interpreter digest.
- A hard-killed independent worker process must not publish. A different process must reopen native authority journals and recover the uncommitted stage; stale candidate publication must fail.
- Missing receiver compute capability must prevent calculation without consuming the queued native E-CPU frame.
- Killing the owner between quorum claim completion and local DAG checkpoint must allow *checked replay of a pure result* without accepting a competing result.
- Recomputed digests must not allow a changed output, stage, host or interpreter epoch to cross the original Ground contract.
- A new future-observer intention must not reuse a checkpoint from the previous processor epoch.

## How to reproduce

Install VL-I, E-CPU and Navigable-Compute (the CI workflow pins the peer source revisions). Then run:

```sh
python -m unittest discover -s tests -p test_problem_formed_claims.py -v
python -m unittest discover -s tests -p test_problem_formed_fabric.py -v
python -m unittest discover -s tests -p test_problem_formed_interpreter.py -v
```

Or inspect the independently spawned command experiment `examples/pfi_fenced_rehome.py`, which offers `dispatch`, `claim`, `hold-after-calculate`, `reassign`, `publish` and `reconcile` modes. It intentionally stores durable state in an explicit temporary filesystem directory rather than assuming persistent process memory.

## Important remaining work

**No authentic remote computational-peer transport was exercised** in this experiment. The generated stage runs in a fresh local E-CPU Hypervisor, while native quorum authority journals are separate files on a **single shared filesystem**, not independent physical hosts or separate mTLS authority processes. The subprocess interruption test proves a real process lifetime interruption, but does not prove failure tolerance against host-machine loss, authentication of remote worker identity, network partition safety under diverse physical machines, or receiver-enforced protection for external side effects. For those, the next integration should reuse E-CPU's existing authenticated `ComputationalFabricForwardingServer`, pinned mTLS identities, `DistributedRecoveryLineageCoordinator`, and IPS-008 receiver handoff barrier; do not replace them with bespoke protocols.

More importantly, this is **result-publication fencing**. A stale worker may continue calculating pure gates; an invalid publish fails, but work and energy already spent are not undone. Equally, a content hash is not a host credential. The PFI-003 wrapper is a sequential-owner experiment; it does not itself serialize competing owners' checkpoint files. Native claim store enforces the accepted claim generation, and a future remote fabric integration must also own receiver-side ordering, exact transport receipt, and live authority checks.

A genuine next-step scientific question is whether PFI's observer-required processor structure can itself inform the route/host placement objective—for example, minimizing *communication cuts* and the number of independently qualified semantic transitions—without discarding necessary future observation distinctions. The native E-CPU computational-fabric candidate resolver is a suitable starting point. No result here resolves P versus NP or infers arbitrary problem-class complexity advantages.

## Sources studied

- [E-CPU transferable execution claim store](https://github.com/Kier73/E-CPU/blob/9d60818a73db8816d5d4f44047d67af81fb1d796/mailcpu/transferable_execution_claim.py)
- [E-CPU existing claim regressions](https://github.com/Kier73/E-CPU/blob/9d60818a73db8816d5d4f44047d67af81fb1d796/tests/test_transferable_execution_claims.py)
- [E-CPU receiver effect fencing IPS-007](https://github.com/Kier73/E-CPU/blob/9d60818a73db8816d5d4f44047d67af81fb1d796/docs/RECEIVER_EFFECT_FENCING_IPS007.md)
- [E-CPU mTLS receiver handoff IPS-008](https://github.com/Kier73/E-CPU/blob/9d60818a73db8816d5d4f44047d67af81fb1d796/docs/RECEIVER_HANDOFF_BARRIER_IPS008.md)
- [Navigable-Compute NCR-006](https://github.com/Kier73/Navigable-Compute/blob/559a1a813012d13ed37c04e2a117467e34ed3cb1/navigable_compute/symbolic_quotient.py)
