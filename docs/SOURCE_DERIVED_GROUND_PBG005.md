# PBG-005 — Source-derived predictive law and witness-driven basis construction

**Implementation:** [`source_hankel_ground.py`](../src/vli/source_hankel_ground.py), [E-CPU/VGI source-probe adapter](../src/vli/source_hankel_fabric.py), [exact construction tests](../tests/test_source_hankel_ground.py), [native integration tests](../tests/test_source_hankel_fabric.py), [runnable demonstration](../examples/source_hankel_demo.py), [four-job workflow](../.github/workflows/pbg005-source-hankel.yml).

## The missing connection resolved

[PBG-004](EVIDENCE_ADAPTIVE_GROUND_PBG004.md) could *admit* another verified stochastic source law when new observations made all prior laws impossible, but the new source's transition matrices still had to be supplied. [SHAPE-010](SHAPE_010_CAUSAL_BASIS_EXPANSION.md) already showed a different route: a source-qualified causal observation can create an additional geometric basis direction and preserve its older revision. The [IFS response-equivalence experiment](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/response-equivalence) demonstrated exact Hankel rank and deceptive delayed counterexamples; [IFS observation-equivalence](https://github.com/Kier73/In-Formation-Space/tree/main/bridges/observation-equivalence) provides transition-congruence refinement. [VGI evidence finalization](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/main/virtual_ground_interpreter/docs/EVIDENCE_AND_RECONSTRUCTION.md) distinguishes authorized from pending/conflicting observations, and [E-CPU SAB-001](https://github.com/Kier73/E-CPU/blob/main/mailcpu/substrate_bootstrap.py) already provides receiver-owned typed routes and independently checked execution reports.

PBG-005 composes those ideas to **learn an exact candidate predictive operator directly from source observations, without passing source transition matrices into its learner**. It then performs a separate source-native mathematical audit, returning either a distinguishing sequence or an all-future certificate for the declared finite source class. If a counterexample is found, the original Ground remains immutable; the learner grows the basis using the exact observed word and its suffixes/prefixes.

## Source interface and intention contract

A source is a fully specified but **not revealed to the learner** finite controlled stochastic law, with rational initial distribution α, row-stochastic interventions A_a, and passive sensor outcome matrices S_{s,y} that sum to identity. The experiment explicitly declares operation tokens:

- `act:<operation>` denotes a **hypothetical source-model intervention transition**, not a physical action.
- `sense:<sensor>:<label>` denotes an observation likelihood operator.

For a word w of operation/event tokens, let M_w be its ordered product and define the exact trace weight

\[
\boxed{f(w)=\alpha M_w\mathbf 1\in\mathbb Q\cap[0,1].}
\]

The **source-query API** returns f(w), bound to the declared source fingerprint, question identity, receiver-owned E-CPU handler report and VGI finalized agreement when using the native adapter. This interface assumes a controlled *model-source* that supports hypothetical probability queries from its named initial condition. A physical uncontrolled environment that only emits one sampled trajectory does **not** supply f(w) for arbitrary counterfactual words. Real pretrained models would require a separately qualified conditional-probability or operator-reading service; no claim of such access is made here.

## Construction: row/suffix Hankel basis, never hidden-source matrices

From an adaptive collection of prefix words P and suffix words S, form the finite Hankel table

\[
H_{p,s}=f(ps).
\]

Choose exact-rational independent P,S rows and columns, with non-singular square H. For every operation a, acquire the **shifted** table

\[
(H_a)_{p,s}=f(pas).
\]

Construct **source-observed** transform matrices and readouts:

\[
\boxed{K_a=H^{-1}H_a,\quad z_0=(f(s))_{s\in S},\quad
c=H^{-1}(f(p))_{p\in P}.}
\]

The candidate's virtual trace law is

\[
\hat f(a_1\cdots a_t)=z_0K_{a_1}\cdots K_{a_t}c.
\]

There is no source M_a, source α or hidden native state in the candidate's construction. Exact rational Gauss-Jordan elimination, all source reader evidence and an immutable Ground digest are retained. The next operation applies only r×r matrices, where r is the currently observed rank. The first candidate is a *hypothesis* until challenged; an arbitrary truncated Hankel table cannot prove an unseen future.

**Witness-directed growth.** On an independently obtained source witness w such that f(w)≠\hat f(w), add every prefix and suffix of w to the candidate table, obtain additional exact measurements, select a new independent Hankel basis, and publish a new Ground digest. This need not always increase r on the first revision; the API refuses a revision that still disagrees with the accepted witness. The old Ground and its observation scope remain unchanged.

## Exact all-future audit for a declared finite source

The learner is intentionally ignorant of native matrices. A **separate** verifier receives the native finite source and candidate. It uses two complementary checks:

1. **Source-global commutation** (stronger): let C be the columns `M_s 1` for learned suffixes s. Check exactly that αC=z₀, Cc=1, and `M_a C=C K_a` for every admitted a. Then every word follows by induction. This is `verify_source_closure`.
2. **Initial-observer equivalence** (potentially more permissive): compose source and learner as a block-diagonal weighted automaton. For every word w retain paired row `v_w=(αM_w,z₀K_w)`. Starting at the empty word, grow an exact-rational basis of their **reachable row space**, adding children only for independent rows. At every basis row check the linear difference readout `αM_w1−z₀K_wc`. If the difference is nonzero, return the earliest-length source counterexample. If the reachable span closes with zero readout on every basis row, it is zero on **every possible word** under the declared finite-source law. Its dimension is at most n+r; the verifier does not exhaustively enumerate an exponentially large corpus.

\[
\boxed{
\bigl[\forall v\in\operatorname{ReachSpan}(\alpha,z_0):\ v(\mathbf1,-c)=0\bigr]
\implies \forall w\in\Sigma^*:\ f(w)=\hat f(w).
}
\]

The latter is `certify_source_equivalence`, returning `COUNTEREXAMPLE` or `EQUIVALENT_ALL_WORDS`. This is an exact linear-algebra proof **conditional on the independently supplied finite rational source model**. It is not a proof that a real physical or neural process has that finite law. The source-native audit has access to hidden matrices because it is the independent verifier, not the learned interpreter.

## Source-qualified E-CPU/VGI reader

The `NativeSourceReader` delegates batched trace queries to **existing** E-CPU `AdaptiveBootstrap` operations: local handler bindings, observed carrier paths, explicit policy allowlists, current leases, source/word-question/nonce-typed `BootstrapTask`s and `verify_bootstrap_report`. It requires exactly two distinct receiver-controlled source loci and the same source fingerprint.

After native calls return exact canonical rational values, the receiver independently encodes **each source's own** SHA-256 of its response/question as an **86-digit base-eight vector**—lossless within VGI's existing Z/8Z observation schema—and uses genuine `ChannelMessage`, `ChannelExchangeLedger`, `EncounterLedger`, `TransportFrameReceipt`, and strict two-source `TransitionEvidenceLedger` finalization. Divergent source values produce distinct VGI outputs and an actual `CONFLICT`; a receiver must **not** replace multiple readings with the first source's digest before finalization. No VGI schema change or custom fabricated `AUTHORIZED` flag is needed. The query atlas caches all source values and the exact E-CPU/VGI event-reference digests, avoiding repeated source reads on basis refinement.

The duplicate observer-hosts here are separate *receiver-provisioned local software bindings*, not independently administered physical hosts. Their same-source output agreement and receiver-local oracle checks qualify the particular experiment's source records; hashes and VGI source names are **not historical remote TLS authentication**. [TDL-005/006](ROUTED_AUTHENTICATED_REFINEMENT_TDL006.md) already implement native pinned mTLS peer identity and routed-observer binding for a different observation family and are the appropriate existing tools for a remote deployment rather than adding a second TLS implementation.

## Executed evidence and benchmarks

### Synthetic delayed-source fixture

Three hidden causal stages; a `step` advances the hidden state, and `goal` reports whether the terminal stage has been reached. The initial source is in state 0. The learner only receives exact word-probability observations.

| Observation | Exact value |
| --- | ---: |
| `P(goal=0)` initially | 1 |
| `P(step, goal=0)` | 1 |
| `P(step, step, goal=0)` | 0 |
| `P(step, step, goal=1)` | 1 |

The first short-window candidate has rank **1** and predicts `P(step,step,goal=0)=1`. The independent paired-row verifier emits that precise **length-3** counterexample. Refining via its prefixes/suffixes expands to rank **3** and the separate native verifier confirms source-global commutation and all-future trace equivalence. The original rank-1 Ground stays unchanged. The demonstration uses **121 unique exact source queries**, grouped into **three construction rounds**, with zero external material operations.

A separate four-stage variant grows from rank 1 to rank **4** when the goal sensor only distinguishes the state after three steps. This checks that the method is not keyed to the three-stage fixture.

### PBG-004 historical interpretation

The native integration test also uses PBG-004's existing eight-state controlled-source fixture with a source-hidden **inverted type detector**. The learned law is formed through exact native trace questions and independent counterexamples, then qualified by the full-source verifier. `HankelObserverSession` reconstructs the previously admitted PBG-004 history (type reading, failed intervention, goal reading) by applying only the learned r×r operators, with **no physical action re-execution**. It then predicts success after the corrected intervention and compares against PBG-002's independent full-source interpreter. This demonstrates a concrete reuse path for historical evidence but does not replace PBG-004's native `QualifiedLaw`/council API; the learned representation is a separate exact interpreter.

### Native test matrix

The test matrix exercises actual E-CPU handler/carrier execution and VGI strict finalization on Linux/Windows Python 3.11/3.14, ordinary-source/alternate-source disagreement, revoked/expired permissions, repeated-query de-duplication, malformed source identities, counterexamples, source-global closure and all-future equivalence. Report query count and receiver route costs separately from the semantic dimension r; neither is automatically an end-to-end LLM inference speedup.

## Reproduction

```sh
python -m pytest -q tests/test_source_hankel_ground.py tests/test_source_hankel_fabric.py
python examples/source_hankel_demo.py
```

The native integration tests need the E-CPU, VGI and Navigable-Compute packages already installed, following this repository's predecessor-pinned PBG-003/004 CI workflows. The standalone demo/math tests use only stdlib plus test runner.

## Active development: translating to real model descriptions

1. **Source-admissible pretrained operator queries.** Use VL-64 and E-CPU's existing source-bound model pages and SHAPE-010's actual causal receipts to expose a bounded `f(w)`-like operation-observation interface. Do **not** reinterpret a file's mere byte existence as an executed future causal transition or fetch/download entire weight tensors by default.
2. **PBG-004 candidate admission from a learned representation.** Give the evidence-adaptive council a verified *behavioral* candidate constructor so it can run the newly learned Hankel operators directly, instead of requiring its canonical `ControlledSource` matrix-based compilation. Preserve the requested goal (read-goal law), operation costs, causal evidence and replay identity.
3. **Open-ended attention and counterexample selection.** Use the existing SHAPE-010 causal residual/witness scorer, IFS exact-rational Hankel counterexample mechanics and VGI partial-state blockers to choose informative source questions per genuine operation/latency cost. The current query construction is finite and not optimal for unbounded alphabets.
4. **Durable external realization.** Keep E-CPU's existing recovery, fencing, checkpoints and terminal rehoming as the authority for true material operations. PBG-005's `step` tokens are source queries/simulated transitions, not externally actuated retries. No new exactly-once physical actuator claim is introduced.

**Throughline:** intention names the future observation to preserve; a source-qualified question yields a new mathematical distinction; the Virtual Layer constructs its own transitions from those observations; and a separate oracle determines the exact claim that may be admitted.
