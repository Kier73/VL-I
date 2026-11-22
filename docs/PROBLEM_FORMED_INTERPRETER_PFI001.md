# PFI-001 — The problem forms its own interpreter

**Status:** executable integration experiment, 21–22 November 2026. Source-bounded mathematical semantics. No demonstrated universal self-forming general intelligence, neural execution, or computation in passive wires.

**Question:** Can an intended observation and a declared causal system generate the shape of their own processor—its state registers, allowed operations, transition circuitry and observer—while preserving continuity and executing through an existing computational host?

This **does not** search for a named programming language and then invoke a solver. Nor does it posit a second universal runtime. It treats the interpreter itself as a constructed Ground, whose machine structure is an output of the problem.

## 1. Connection to the actual repositories

The following was inspected, not copied or replaced:

- [VL-I `interpreter_ground.py`](../src/vli/interpreter_ground.py) already lifts actual interpreter source and declarative contracts, distinguishing source evidence from handler authority. [TIME-004](../src/vli/reflective_environment_processor.py) already constructs a bounded reflective processor from observed environment models. PFI-001 asks what the *machine's register space and transition microarchitecture* become when the observer intention changes, rather than selecting an existing family of processors.
- [Navigable-Compute NCR-006](https://github.com/Kier73/Navigable-Compute/blob/main/docs/NCR-006.md) already constructs the **minimal future-observer quotient** of a fully declared binary-affine source. This experiment imports `construct_symbolic_quotient` and its independent checker directly: it does not create a competing quotient constructor.
- [E-CPU native `computation_dag.py`](https://github.com/Kier73/E-CPU/blob/main/mailcpu/computation_dag.py) already manages durable content-addressed dependencies and exact result commits. PFI-001 uses its real `ComputationNode` and `NavigableComputationDAG`, not another scheduler or a mocked API.
- [E-CPU GGE-001](https://github.com/Kier73/E-CPU/blob/main/docs/GENERAL_GROUND_EXECUTION_GGE001.md) cautions against creating parallel authority layers. This prototype has **no** Hypervisor capability grant, route qualification, actual TCP/mTLS traffic, shared host authority or physical link execution. Those remain separate E-CPU admission steps.
- [In-Formation-Space PNG-001/002/003](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/pnp-ground-navigation) motivate the distinction between the cost of describing a result, deriving a representation, and actually realizing it. PFI-001 focuses on the interpreter's form rather than SAT-specific representation selection.

## 2. What is supplied, and what is generated

The supplied problem is an exact mathematical description

\[
D=(X=\mathbb F_2^n,\{T_a(x)=A_ax\oplus b_a\}_{a\in A},\;O(x)=Hx,\;\text{intention}).
\]

There is **no supplied processor register layout or instruction-set implementation**. The observer intention names a particular output view and, at execution, an initial state and action-word/time coordinate.

The NCR-006 constructor derives the least future-stable dual observer space

\[
W=\mathrm{span}\{hA_w\mid h\in\mathrm{rows}(H),\;w\in A^*\}
\]

with canonical basis rows `B`. The native state `z=Bx` has `r=dim W` bits; the source can have `2^n` states while the native interpreter only needs `2^r` distinguishable states for all requested future observations. For every generator, it constructs native operations `M_a,c_a` with `BA_a=M_aB`, `c_a=Bb_a`, and readout `H=DB`. NCR's verifier independently checks that **every** possible finite future observer output is preserved under the stipulated source law.

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

## 4. An evolving problem forces the processor to re-form

If the observer changes from `O_old` to `O_new` while the causal law remains fixed, the resulting new `W_new` can have a different rank, hence a different processor.

Let the old native state be `z_old=B_old x` and the new one be `z_new=B_new x`:

- If every new basis row lies in the span of the **old** basis, a source-state-independent quotient map `z_old -> z_new` exists. PFI-001 derives this mapping exactly. The original state can be migrated without reobtaining the full source.
- If `W_new` contains previously invisible distinctions, **there is generally no function** of the old coordinate alone that can produce the new one. PFI-001 explicitly refuses that transition until a *declared source-state anchor* consistent with the old native coordinate is supplied. It uses that anchor to instantiate the new coordinate. This is not authenticated sensing, but it prevents a false mathematical inference from an impoverished old representation.
- If the original source *transition laws* have changed, the succession contract is refused. A changed law requires a new Ground epoch and a separate continuity obligation; it is not silently accepted as a different name for the old processor.

The checkable succession certificate binds both digests, old and new coordinates, transformation kind, and optional source anchor. An adversarially rehashed but incorrect successor coordinate fails independent regeneration.

This makes the compiler's self-change a **semantic** phenomenon: the machine can be reconstituted with more or fewer registers because the intended distinctions changed, but it cannot fabricate lost observations.

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

## 7. Why this is not merely choosing a language

A static interpreter has an instruction vocabulary and registers that exist *before* the input. A PFI-001 instance receives a problem whose transition law and observer are known, discovers the smallest exact future-relevant coordinate subspace for that problem, **forms** its registers and executable transformation graph, and realizes that problem through the resulting constructed machine. When the intended observation changes, its exact mathematical state-space, circuit and identity can change.

That is substantially stronger than selecting a preexisting SAT solver or programming language. It is still a restricted but verifiable example of an adaptable interpreter, not a claimed general path around the P versus NP question: a general source law might have no compact future-observer quotient, and discovering the appropriate Ground could itself cost exponentially more than operating it.

**Next constructive direction:** generate and admit non-affine finite Boolean processor architectures from NCR-007's invariant-derived nonlinear laws, and connect E-CPU `ComputationalExecutionAdmissionCoordinator` to execute the generated circuits under explicit rights, real host identities and typed recovery epochs. An intention should be able to request a different computational structure *and* a qualified host to realize that structure, not assume the path itself performs arbitrary arithmetic.
