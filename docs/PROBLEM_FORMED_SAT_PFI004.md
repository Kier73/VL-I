# PFI-004 — Direct SAT becomes a problem-formed E-CPU computation

**Status:** bounded integration experiment. Combines [SFI-001](https://github.com/Kier73/In-Formation-Space/blob/276e669b4aa1bbd251dd5bb9e76f33bad65e692b/experiments/pnp-ground-navigation/SFI-001.md), [PNG-003](https://github.com/Kier73/In-Formation-Space/blob/276e669b4aa1bbd251dd5bb9e76f33bad65e692b/experiments/pnp-ground-navigation/PNG-003.md) and VL-I PFI-001/002/003. **P versus NP remains unsolved.** No natural-language understanding or physical network-native computation is claimed.

## Research question

Can **the SAT problem itself** form the state representation, operator graph, partition topology and result semantics of a qualified E-CPU computation, and can those semantics survive loss of the currently assigned worker?

Rather than choosing from a menu of fixed language descriptions, the mathematical Ground determines the processor architecture. The PNG-003 proof-carrying CNF compiler discovers exact affine constraints or furnishes a refutation. SFI-001 independently replays its proof and forms a finite processor with rank-dependent registers, source-literal observation gates and intention-conditioned output accumulation. PFI-004 distributes that processor's **source-evaluated work partitions**, not a precomputed answer. Ordinary CPU hosts execute each partition through the already-existing E-CPU `VirtualLayerRuntime`, `OwnedNetworkFabric`, content-addressed `NavigableComputationDAG`, and majority-backed `TransferableExecutionClaimStore`.

A 6-variable doubly guarded parity formula is particularly useful as a first end-to-end target: the source does not directly expose its XOR; contraction reveals it, so the constructed interpreter uses five free-coordinate registers, and four 8-coordinate work partitions cover the entire exact model set (32 assignments). General 8-variable CNF remains eight-dimensional and must evaluate the full 256 candidates. The same adapter can use a 48-variable, eight-free-coordinate instance with more expensive source-ground proof construction.

## Native execution and recovery protocol

1. Verify the original CNF and PNG-003 proof. Construct the SFI-001 machine and source-bound digest from the same full source and intention. Reject an `UNDETERMINED` machine rather than declaring UNSAT; proof-only UNSAT is independently handled in SFI-001, without fictitious host search.
2. Create E-CPU's original native DAG: one dispatch, one node per formed SAT partition, and a final aggregate node requiring **all** partition predecessors. Build the actual E-CPU owned network with one origin->partition FIFO link per generated partition. Dispatch canonical typed bytes referring to each exact source-bound coordinate interval.
3. Native `TransferableExecutionClaimStore` publishes one pure `pfi.sat_source_segment@1` assignment per partition, tied to the parent query, the partition, the original source and machine digests, typed operator contract, host, and implementation fingerprint.
4. The assigned host verifies its **live** majority-backed claim and original frame, constructs a local E-CPU `HeadlessOS` and seeded Ground, binds its pure operator **locally**, and submits a `VirtualLayerWorkload` through local Hypervisor authority. The handler executes `execute_partition`, visiting source assignments, evaluating the **original CNF**, and returning an exact count/witness receipt for its admitted range. An unauthorized receiver cannot calculate a candidate.
5. Computation is not publication. A calculated candidate is *speculative*. Only the current claim epoch and host can have the exact result digest majority-published. The durable candidate is written before native quorum completion; only after majority completion does the original E-CPU DAG consume the real queued frame and publish its node result.
6. If the worker dies with an unpublished candidate, a new host can receive a native reassignment, check fresh authority and recompute the same pure partition. The old candidate, even mathematically valid, fails its epoch/host check. If a process dies after majority completion but before local DAG publication, the new process restores the native E-CPU network and DAG checkpoint and **re-executes pure mathematics**, reconciles the original committed result digest, then continues.
7. The terminal aggregator **independently reruns every source-CNF partition**, demands full partition coverage, and reduces according to the requested SAT/#SAT intention. It cannot report UNSAT based merely on failure to find a solution under an exhausted search budget.

## Test targets

- A real source-derived four-stage search with each stage carried by typed E-CPU FIFO frames, a native computational DAG and a receiver-local `VirtualLayerRuntime` Hypervisor call, then exact #SAT count 32.
- A mathematically correct, old-host speculative result refused after native majority reassignment to another logical host, with the exact source and interpreter identities unchanged.
- A hard-exited worker process after real evaluation but before publication, followed by a fresh Python process restoring the same claim and accepting only the replacement host.
- Recovery across the crash gap **after quorum completion but before local DAG commit**, without a competing result or modified mathematical observation.
- Missing local `compute` capability, changed intention under an old checkpoint, altered/rehashed work result, modified network/DAG snapshots, and incomplete partitions rejected.
- An unstructured eight-variable problem with no compressible parity interface still requires checking 256 assignments and preserving exact count.

## Reproduction and source provenance

SFI-001 experiment source: `Kier73/In-Formation-Space` commit `276e669b4aa1bbd251dd5bb9e76f33bad65e692b` in `experiments/pnp-ground-navigation/`; E-CPU and Navigable-Compute versions are pinned by the CI workflow. Install these three repositories plus VL-I. Put the SFI-001 directory on the Python module path, or invoke the dedicated CI workflow, then:

```sh
python -m unittest discover -s VL-I/tests -p test_problem_formed_sat.py -v
python -m unittest discover -s VL-I/tests -p test_problem_formed_claims.py -v
```

## Interpretation and remaining work

This is a software execution substrate: all stages run CPU mathematics inside E-CPU's ordinary runtime; the typed network is E-CPU's `OwnedNetworkFabric`, not a physical network hop. The native recovery journals are three separate durable files in **one shared filesystem**, not three separately authenticated physical machines. Neither SHA-256 digests nor local execution receipts constitute remote attestation. Native claim fencing governs **result publication** of pure operations, not the ability of a stale worker to continue calculating and consuming energy. The owner wrapper assumes one orderly coordinating process at a time; independent concurrent owners and external effects require further protocols, including E-CPU IPS-008 receiver-acknowledged effect handoff, mutual TLS and fully independent quorum services.

A direct SAT test is a sharp discriminator for P versus NP: this apparatus succeeds efficiently when the proof-constructible affine rank makes d small and can carry its exact semantics across hosts. For general CNF, d can approach n; the formed machine will then perform exponential work or refuse under its explicit budget. Worse, the PNG-003 *source-ground compiler itself already solves or partially solves the input*, so the full cost of deriving its structure must be charged. The experiment is therefore a reproducible investigation into **where the complexity lives**, not a claim to hide it in a special interpreter.

**Next test:** replace the local shared-file quorum and owned FIFO links with E-CPU's existing authenticated computational forwarding service and IPS-008 effect-receiver barrier. Do not introduce a second authority or trust transport contents as a computation. Separately, pursue non-affine interpreter self-formation rather than presupposing that parity is the optimal language for arbitrary SAT.
