# PFI-002 — From problem-formed interpreter to active E-CPU observer-host fabric

**Status:** executable cross-repository integration experiment, 21 November 2026. See CI results on the PR for evidence of execution on fresh hosts. This is a finite declared-affine mathematical experiment, not physical communication-path computing, remote mTLS admission, or a general computational substrate compiler.

## Why this experiment exists

The interpreter should not merely choose a mathematical language. The description of the problem, its available transformations, and its observable obligations should determine **the structure of the interpreter itself**. That structure should then determine a realizable arrangement of active computational hosts. Resource constraints should change the arrangement, without modifying the source problem's semantics.

PFI-001 constructs an exact future-observer-minimal quotient and synthesizes XOR/flip microcode from its induced affine transformations, using Navigable-Compute NCR-006 and an independent symbolic gate verifier. E-CPU provides the substrate, rather than a fabricated second scheduler: `OwnedNetworkFabric`, `NavigableComputationDAG`, and `VirtualLayerRuntime` under receiver-local `HeadlessOS` Hypervisor authorization.

## Concrete construction

For a declared source law `T` and observer `O`, PFI-001 constructs `q` so that for the required action words `w`, `O(T_w(x))` factors through an exact induced processor on `q(x)`. The requested observer, not a prechosen ISA, determines the register count and generated gate circuits. PFI-002 divides **actual generated topologically ordered gates** into sequential host stages using a declared `gates_per_host` capacity. It includes at least one final observation stage. Thus the problem and resource envelope determine the number of hosts and the active transformations assigned to each.

Each host stage:

1. receives a canonical, typed binary packet through an actual E-CPU `OwnedNetworkFabric` FIFO;
2. checks the exact Ground, interpreter epoch, stage index, and expected intermediate bytes;
3. constructs a new `HeadlessOS`, admits a local seed envelope, binds a **trusted local** operator to the stage, and receives the `compute` capability only if allowed;
4. executes `pfi.formed_segment` with `VirtualLayerRuntime.execute_operator`, which runs native Hypervisor authorization and a pure local handler;
5. compares its actual gate result against the independently expected source-derived state, emits the next typed frame, and commits the DAG frontier with a host/authorization receipt.

The final sink consumes a typed observer-result frame and commits a source-equivalent result. The DAG and OwnedNetwork checkpoints preserve the pending frame and already committed stages. Passing those checkpoints into a **different operating-system process** preserves the computational Ground identity while a new receiver constructs and authorizes its own local runtime.

### What is problem-determined versus externally provided

- **Problem-determined:** quotient register dimension, generated microcode, primitive basis, gate dependency order, and observer projection. These derive from the admitted mathematical source and intention via PFI-001 and NCR-006.
- **Jointly determined by the problem and resources:** number of active virtual hosts, number of owned links, gate fragments per host, FIFO payload size, and reconstruction work. `gates_per_host` is an **explicit resource budget**, not a mathematically discovered constant.
- **Externally governed:** actual runtime availability, effect authority, Hypervisor policy, destination process identity, source authenticity, and the physical link or processor supplying arithmetic.

A CPU still executes every gate. A passive communication path does not become an active arithmetic operator merely because it transports a byte frame. The test uses local in-process `OwnedNetworkFabric` frames and **checkpoint transfer** across fresh Python processes, not a network socket between machines.

## Continuity and refusal

The checkpoint records the source-derived interpreter epoch, original target, exact E-CPU DAG snapshot, E-CPU network snapshot with its FIFO and event journal, receiver-local execution receipts, and cumulative communication bytes. The restore path reconstructs the interpreter, recomputes expected stage-by-stage semantic frames, checks DAG node identity and committed-prefix relationships, checks pending frame content, and checks byte accounting.

The receiver **does not consume a frame** until its local Hypervisor has allowed and completed the stage. Missing `compute` rights must leave the whole checkpoint unchanged. A changed observer/derived processor has a different interpreter identity and is refused by the old checkpoint. A separate PFI-001 succession proof can admit a changed observer intention only if its required old state is available; refinement cannot invent missing bits.

This is **boundary-to-boundary checkpoint continuity**, not a proof that an interrupted frame send and DAG commit are one atomic transaction, nor a distributed fencing/quorum guarantee. Signed authority, fault isolation, hard process-kill injection inside a transaction, live intermachine mTLS, host placement across the public Internet, and byte-for-byte physical observation remain further experiments.

## Reproduce

Install `Navigable-Compute`, `E-CPU`, and `VL-I` from their pinned reviewed revisions. From the VL-I repo root:

```sh
python -m unittest discover -s tests -p test_problem_formed_fabric.py -v
python -m unittest discover -s tests -p test_problem_formed_interpreter.py -v
python examples/pfi_fabric_rehome.py checkpoint.json --host origin --initialize
python examples/pfi_fabric_rehome.py checkpoint.json --host receiver-a
python examples/pfi_fabric_rehome.py checkpoint.json --host receiver-b
```

Continue invoking the last command (each invocation is a fresh process) until `complete` is `true`. The CI suite explicitly checks multiple separate processes, generated architecture variation, correct source-derived observations, missing-rights refusal, checkpoint tampering, stale Ground epochs, and host switching.

## Research direction

The next useful step is to move the generated and admitted host stages into E-CPU's existing authenticated computational fabric, with **receiver-local implementation qualification, host loss, quorum-fenced reassignment, and retained source-observer identity**. That is the stronger test of substrate morphogenesis: the requested computation determines not just a logical processor, but qualified computational loci and migration routes in a changing environment.
