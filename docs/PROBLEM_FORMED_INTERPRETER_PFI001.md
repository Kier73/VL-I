# PFI-001 — The problem forms its own interpreter

## Current implementation at a glance

**Source-grounded contract.** `vli.problem_formed_interpreter.construct(source)` accepts an exact, declared binary-affine `SymbolicSource`. It uses Navigable-Compute's NCR-006 quotient to derive native register bits, an XOR/FLIP circuit for each transition and a circuit for the observer projection. `verify_interpreter` symbolically re-evaluates the generated gates and compares their affine rows/bias against the mathematical quotient. `realize` executes a cost-chosen path and compares the result with independently verified NCR navigation; `certify_succession` identifies which changes of observation retain sufficient old state.

### Formal invariants and operational placement

For source transitions \(T_a(x)=A_ax\oplus b_a\) on \(\mathbb F_2^n\) and requested linear observer \(O(x)=Hx\), define

\[
W_F=\operatorname{span}_{\mathbb F_2}\{hA_w\mid h\in\operatorname{rows}(H),w\in A^*\}.
\]

The constructed quotient basis \(B\) has \(r=\dim W_F\) rows; native state \(z=Bx\). Every generated transition and observer gate network implements

\[
BA_a=M_aB,\quad c_a=Bb_a,\quad H=DB,
\qquad z'=M_az\oplus c_a,\quad O(x)=Dz.
\]

**Operational order:** validate and construct NCR quotient → synthesize `xor` / `flip` topological circuits → verify each gate's symbolic affine expression → choose direct gate execution or composed-law navigation using the cost heuristic → independently verify NCR result → decode observer → bind source/interpreter/result digests in the receipt. E-CPU DAG integration is separate: `create_ecpu_dag` creates `ground → navigate → observe`, and `run_one_ecpu_frontier` commits one ready step per call.

### Current syntax

```python
from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from vli.problem_formed_interpreter import construct, verify_interpreter, realize

# Four-bit exact source; action 0 toggles bit 0; observer reads bit 0.
identity_rows = tuple(1 << i for i in range(4))
source = SymbolicSource(
    4, (AffineMap(identity_rows, 1),), (1,), "declared-demo"
)
quotient, bundle = construct(source)
assert verify_interpreter(source, quotient, bundle)
receipt = realize(source, quotient, bundle, 0, (0,), 5)
assert int(receipt["observed_decimal"]) == 1
```

The source is declared mathematical input; no external source discovery is involved in this call. The runnable cross-repository test fixture is [`tests/test_problem_formed_interpreter.py`](../tests/test_problem_formed_interpreter.py); it imports the actual NCR and E-CPU modules.

### Intention succession: exact information requirement

For old/new intentions under the **same source transition-law epoch**, write \(z_o=B_ox\) and \(z_n=B_nx\). The implementation attempts to express every row of \(B_n\) in the row span of \(B_o\):

\[
\exists L\quad B_n=LB_o
\quad\Longleftrightarrow\quad
\operatorname{rowspan}(B_n)\subseteq\operatorname{rowspan}(B_o).
\]

When this holds, `certify_succession` derives \(z_n=Lz_o\) without a source-state anchor. Otherwise, it requires an explicitly declared \(x\) satisfying \(B_ox=z_o\) and computes \(z_n=B_nx\), recording `declared-source-anchor-required`. `verify_succession` independently regenerates the exact certificate, including both interpreter identities. This is the **affine linear-algebra counterpart** of E-CPU INTENT-001's finite partition-refinement criterion.

### TDL-002: minimal refinement witness (implemented)

The [separate minimal-witness module](MINIMAL_REFINEMENT_TDL002.md) now computes the rank deficiency
\(d=\operatorname{rank}([B_{old};B_{new}])-\operatorname{rank}(B_{old})\).
It selects exactly \(d\) additional source-space covectors, derives \(B_{new}=L[B_{old};C]\), and reconstructs the new coordinate from \((z_{old},Cx)\). The independent verifier rederives the rank and exact row identities; it checks receipt bytes and coordinate transformation separately from the caller's measurement authority.

The original full-source-anchor `certify_succession` API remains unchanged. The additional protocol is in `vli.minimal_refinement` and `vli.minimal_refinement_verify`, with a dedicated Linux/Windows workflow and finite E-CPU/NCR commutation test.

**Evidence anchors:** `test_intention_refinement_changes_processor_dimension_and_epoch`, `test_intention_coarsening_derives_map_without_source_anchor`, `test_causal_continuation_after_mathematically_qualified_reformation`, `test_changed_dynamics_refuses_reusing_intention_epoch`, and `test_native_ecpu_dag_checkpoint_and_host_resumption`.

**New construction:** translate an E-CPU finite quotient certificate into a VL-I/NCR affine basis witness when the supplied source has an admitted affine embedding. Test whether the two implementations agree on all future-observation classes and whether a succession witness transfers without retaining the full source state.

**Evidence scope:** current generation and oracle checks are over the stipulated exact binary-affine source law. Three independent local processes exercise E-CPU's native DAG checkpoint continuation; receiver-local execution, physical host authority, and external source attestation are separate integration contracts.



**Question:** Can an intended observation and a declared causal system generate the shape of their own processor—its state registers, allowed operations, transition circuitry and observer—while preserving continuity and executing through an existing computational host?

It treats the interpreter itself as a constructed Ground, whose machine structure is an output of the problem.

## 1. Connection to the actual repositories

The following was inspected, not copied or replaced:

- [VL-I `interpreter_ground.py`](../src/vli/interpreter_ground.py) already lifts actual interpreter source and declarative contracts, distinguishing source evidence from handler authority. [TIME-004](../src/vli/reflective_environment_processor.py) already constructs a bounded reflective processor from observed environment models. PFI-001 asks what the *machine's register space and transition microarchitecture* become when the observer intention changes, rather than selecting an existing family of processors.
- [Navigable-Compute NCR-006](https://github.com/Kier73/Navigable-Compute/blob/main/docs/NCR-006.md) already constructs the **minimal future-observer quotient** of a fully declared binary-affine source. This experiment imports `construct_symbolic_quotient` and its independent checker directly: it does not create a competing quotient constructor.
- [E-CPU native `computation_dag.py`](https://github.com/Kier73/E-CPU/blob/main/mailcpu/computation_dag.py) already manages durable content-addressed dependencies and exact result commits. PFI-001 uses its real `ComputationNode` and `NavigableComputationDAG`, not another scheduler or a mocked API.
- [E-CPU GGE-001](https://github.com/Kier73/E-CPU/blob/main/docs/GENERAL_GROUND_EXECUTION_GGE001.md) cautions against creating parallel authority layers. This prototype has **no** Hypervisor capability grant, route qualification, actual TCP/mTLS traffic, shared host authority or physical link execution. Those remain separate E-CPU admission steps.
- [In-Formation-Space PNG-001/002/003](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/pnp-ground-navigation) motivate the distinction between the cost of describing a result, deriving a representation, and actually realizing it. PFI-001 focuses on the interpreter's form rather than SAT-specific representation selection.

## 2. Mathematical input and generated microarchitecture

The source supplies the transition matrices, biases and requested observer rows, while NCR-006 supplies the independently verified future-observer basis. The processor register count, gate networks, observer decoder and interpreter identity are generated, as formalized in the current implementation contract above.

PFI-001 then **constructs actual gate-network microarchitectures** from the derived mathematics. Every transition and the observer readout gets its own acyclic network of primitive `xor` and `flip` nodes and source-derived output connections. The processor's *number of registers*, *number and wiring of gates*, *effective primitive set*, *operator network*, *observer decoder*, and *digest identity* are outputs of construction. For a degenerate projection, zero gates may be required. Its execution uses these generated networks; for long logical-time intentions it instead uses NCR's independently checked compressed operator-composition semantics.

The microarchitecture is no more privileged than any other description. A separate checker propagates **symbolic GF(2) affine masks through every gate**, proving that the actual generated circuit implements each NCR operator for *all* native states. It is not sample-based simulation equivalence. Tampering and then recomputing the SHA digest cannot create a true circuit-equivalence proof.

**This is endogenous architecture construction within a declared finite affine family**, not spontaneous discovery of an unknown environmental law. The generator and constraint equations are still provided by the problem. The chip, CPU, and source of energy are not generated by the equations.

## 3. Interpretation, execution and continuity

At runtime, an intention `(initial source coordinate, action word, repetitions)` is evaluated under two possible realization structures:

- **Generated gate interpreter** for a small number of steps: the declared source's exact dynamics are executed by the newly constructed gate program.
- **Composed law navigation** for long temporal coordinates: NCR's affine operator compositions reach the requested state in a number of composed operators logarithmic in repetitions (arithmetic bit cost and ground-construction cost remain real).

The system uses a simple, explicit *symbolic operation-count estimate*, not a claimed optimal optimizer or measured hardware benchmark. Either execution route is independently re-evaluated by the existing NCR mathematical verifier. The result includes both the interpreter identity and what was actually executed.

PFI-001 also constructs a **three-node native E-CPU DAG**:

    ground(source) -> navigate(constructed processor) -> observe(required output)

The receiver commits one node each time it is invoked, validating all predecessors and the full interpreter epoch digest before advancing. E-CPU's own `snapshot` and `from_snapshot` serialize and restore the exact committed frontier. A three-subprocess demonstration uses genuinely different local Python processes to perform the successive native DAG transitions, reconstructing the original mathematical source/quotient and interpreter identity on each process. Those are host-local execution steps; no passive link, network, Hypervisor or multi-party authority is claimed.

## 4. Intention succession and information preservation

`certify_succession` applies the row-span criterion in the implementation contract above. A smaller observer quotient can be reconstructed directly from a more informative native coordinate. A finer intention requires additional source information consistent with the old native coordinate; the current API accepts a full declared source-state anchor. Both interpreter digests and the exact transformation kind are recorded and checked by `verify_succession`. When transition generators or source-law origin differ, the old succession epoch is refused.

The [Translation Discovery Log (TDL-001)](https://github.com/Kier73/In-Formation-Space/blob/main/reports/TRANSLATION_DISCOVERY_LOG.md) derives the minimum conditional rank witness that could replace the larger source anchor in a future implementation.

## 5. Falsifiers and cost accounting

The project must not say that observer coordinates make all problems small. A 24-bit shift system whose first bit eventually reveals every bit has an **exact future observer dimension of 24** even though each immediate observation is one bit; the prototype refuses an eight-bit budget. With an alternative 128-bit source containing 126 forever-unobserved bits, the same construction derives two native registers from the exact required observer future.

The important accounted costs are

\[
T_{total}=T_{construction}+T_{circuit\ checking}+T_{execution}+T_{result\ checking}
+T_{checkpoint\ serialization}+T_{host\ transition}.
\]

The demo reports the cost estimate for generated gates versus composition, the *actual* generated node count, and NCR algebraic composition counts. Local `perf_counter` and large-integer bit overhead could be added in follow-up runs; no unconditional speedup claim is made. The source dynamics and observation contract are stipulated, not inferred from the real world.

## 6. Reproduction

Requires locally installed current/pinned revisions of [Navigable-Compute](https://github.com/Kier73/Navigable-Compute), [E-CPU](https://github.com/Kier73/E-CPU), and VL-I. The feature imports its predecessor APIs at use time; the rest of VL-I retains its dependency-light installation.

```bash
python -m unittest discover -s tests -p 'test_problem_formed_interpreter.py' -v
python examples/problem_formed_interpreter_demo.py
python examples/pfi_host_rehome.py /tmp/pfi-checkpoint.json 0
python examples/pfi_host_rehome.py /tmp/pfi-checkpoint.json 1
python examples/pfi_host_rehome.py /tmp/pfi-checkpoint.json 2
```

The dedicated GitHub Actions workflow uses Linux and Windows with Python 3.11 and 3.13. It checks source-to-Ground exactness, generated microcode equivalence, large-time navigation, three genuine independent local process lifetimes, E-CPU DAG checkpoint persistence, stale interpreter rejection, forced source-anchor requirement under refinement, changed causal-law epoch rejection, tamper controls and future-observer rank obstruction.

The scaffold is deliberately not a self-aware agent, a general compiler for arbitrary bytes, a live environmental solver, physical fabric execution, authenticated host migration or unrestricted behavioral equivalence. An independent external sensor may require stronger evidence and authority than this mathematical source contract supplies. A future version should bind the same verified microarchitecture to E-CPU's `VirtualLayerRuntime` with actual receiver-local operator admission and a typed E-CPU host catalogue/route, **without replacing** the existing Hypervisor, classifier or durable channel.

## 7. The language is constructed from the problem

A static interpreter supplies its instruction vocabulary and registers in advance. PFI-001 derives register count, XOR/FLIP operation graphs, observer readout and exact interpreter digest from the declared source law and intention; the generated machinery changes when the observer's required distinctions change.

**Next construction:** extend exact formation beyond binary-affine source laws using NCR-007 invariant-derived nonlinear families, then bind produced circuits through E-CPU's existing receiver-owned admission coordinator. Demonstrate source-backed numerical equivalence, independent host authority and recovery identity under explicit tests.

The quantitative refinement-witness problem and cross-representation verification procedure are tracked in [TDL-001](https://github.com/Kier73/In-Formation-Space/blob/main/reports/TRANSLATION_DISCOVERY_LOG.md).
