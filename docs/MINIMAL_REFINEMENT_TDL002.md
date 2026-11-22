# TDL-002 — Minimal information for affine intention succession

**Implemented capability.** Given two independently verified PFI-001 interpreters for the same declared binary-affine source dynamics, construct the smallest linear observation supplement that, together with the old native state, reconstructs the new native state. Produce an integrity-bound plan and a separate state-transfer receipt with an independently rederived rank and source-row audit.

**Source:**
- [`minimal_refinement.py`](../src/vli/minimal_refinement.py): `plan_minimal_refinement`, `witness_from_source`, `certify_minimal_refinement`.
- [`minimal_refinement_verify.py`](../src/vli/minimal_refinement_verify.py): `verify_refinement_plan`, `verify_refinement_receipt`.
- [`problem_formed_interpreter.py`](../src/vli/problem_formed_interpreter.py): retained PFI-001 `construct`, `certify_succession`, `verify_interpreter`.
- [Unit and adversarial tests](../tests/test_minimal_refinement.py); [cross-representation commutation tests](../tests/test_translation_commutation.py).

## Exact contract

Let both intentions share source transitions \(\{T_a(x)=A_ax\oplus b_a\}\), width \(n\), and origin. Their future-observer quotient basis rows are \(B_o\in\mathbb F_2^{r_o\times n}\) and \(B_n\in\mathbb F_2^{r_n\times n}\). The old native coordinate is \(z_o=B_ox\).

Define

\[
\boxed{d=\operatorname{rank}\begin{bmatrix}B_o\\B_n\end{bmatrix}
-\operatorname{rank}(B_o).}
\]

The producer greedily selects \(d\) independent new-basis rows modulo the span of \(B_o\), forming \(C\in\mathbb F_2^{d\times n}\), and solves exact row identities

\[
B_n=L\begin{bmatrix}B_o\\C\end{bmatrix}.
\]

For any source state \(x\), the new processor coordinate is

\[
\boxed{z_n=L\begin{bmatrix}z_o\\w\end{bmatrix},\quad w=Cx\in\mathbb F_2^d.}
\]

**Minimality for the full declared source domain.** For any fixed old coordinate \(z_o\), the possible new coordinates form an affine set of exactly \(2^d\) values. Thus a worst-case \(d\)-bit auxiliary observation is necessary and the constructed \(w\) is sufficient. The proof follows by rank-nullity applied to the joint map \(x\mapsto(B_ox,B_nx)\).

The original `certify_succession` continues to accept a full `source_anchor` when necessary. TDL-002 provides a *separate* narrower protocol, preserving the existing API and provenance.

## Algorithm, in the order currently implemented

1. Independently verify both PFI-001 interpreter Grounds and equality of the declared affine generator/origin epoch.
2. Start with \(B_o\); traverse the new basis rows in declared order, appending a row to \(C\) iff it raises rank.
3. Over GF(2), express every row of \(B_n\) as an exact XOR of rows from \([B_o;C]\). Export the resulting matrix \(L\).
4. Bind source digests, interpreter digests, dimensions, \(d\), row masks, scope and plan digest.
5. Call the independent verifier: rederive rank and greedy complement by a separate elimination algorithm; check each row of \(B_n=L[B_o;C]\) as an exact bit identity.
6. Observe/provide \(w=Cx\) through `witness_from_source` or an independent source adapter; supply old native coordinate and \(w\) to `certify_minimal_refinement`.
7. Verify receipt fields, exact plan and interpreter identities, canonical decimal encoding, digest and \(z_n=L(z_o,w)\) using `verify_refinement_receipt`.

## Current syntax

```python
from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from vli.problem_formed_interpreter import construct
from vli.minimal_refinement import (
    plan_minimal_refinement, witness_from_source, certify_minimal_refinement,
)
from vli.minimal_refinement_verify import verify_refinement_receipt

identity = tuple(1 << i for i in range(4))
operator = (AffineMap(identity, 0),)
old = SymbolicSource(4, operator, (1,), "same-declared-law")
new = SymbolicSource(4, operator, (1, 4), "same-declared-law")
old_q, old_bundle = construct(old)
new_q, new_bundle = construct(new)
args = (old, old_q, old_bundle, new, new_q, new_bundle)

plan = plan_minimal_refinement(*args)
assert plan["additional_bits"] == 1
x = 0b1101                        # Source-side observation for this fixture
missing = witness_from_source(old, plan, x)
receipt = certify_minimal_refinement(
    plan, old_native_state=old_q.project(x), witness_bits=missing,
)
assert verify_refinement_receipt(*args, plan, receipt)
assert int(receipt["new_native_decimal"]) == new_q.project(x)
assert "source_anchor_decimal" not in receipt
```

The witness `missing` is one bit; it is supplied by the caller and checked for mathematical consistency with the plan, **not attested as a physical measurement**. Exact receiver authorization, source authentication and sensor identity are independent of this coordinate-equivalence protocol.

## Cross-representation test

[`test_translation_commutation.py`](../tests/test_translation_commutation.py) translates a fully declared five-bit `SymbolicSource` into a 32-state E-CPU `FiniteObserverLaw` for the same two source operators and requested observer. It constructs both independent quotients and checks

\[
\forall x,y:\quad
\operatorname{ECPUClass}(x)=\operatorname{ECPUClass}(y)
\iff
B x=B y
\]

for five observer selections and \(32^2\) pairs each. It also checks 960 source/quotient action-word executions against original source observations. This is a deliberately small exhaustive integration oracle; source-table enumeration is used **for comparison**, not as a prerequisite of the symbolic affine constructor.

## Verification contracts

| Test | Checked property |
|---|---|
| `test_one_auxiliary_bit_replaces_full_128_bit_anchor` | 128-bit source: a three-register new observer requires only one added bit beyond two retained old registers |
| `test_coarsening_uses_zero_auxiliary_bits` | All new registers can be expressed from old registers |
| `test_incomparable_observation_spans_use_conditional_bits` | Conditional witness works for distinct/incomparable projections |
| `test_exact_independent_row_audit_rejects_rehashed_forgery` | Rehashed wrong source-row identities, witness rows and rank claims fail independent audit |
| `test_receipt_requires_actual_bit_domain_and_exact_mapping` | Boolean, missing, oversized and noncanonical witnesses or receipts rejected |
| `test_source_epoch_and_interpreter_identity_are_bound` | Different source dynamics and stale interpreter digest refused |
| `test_exhaustive_small_source_observer_translations` | 25 observer pairs × 32 source states = 800 independently checked transfers |
| `test_finite_partition_equals_affine_observer_coordinate` | E-CPU Moore-state classes agree with NCR/VL-I affine coordinates on declared five-bit source |

Reproduce from the VL-I repository with Navigable-Compute and E-CPU available as for existing PFI-001:

```sh
python -m unittest discover -s tests -p 'test_minimal_refinement.py' -v
python -m unittest discover -s tests -p 'test_translation_commutation.py' -v
python -m unittest discover -s tests -p 'test_problem_formed_interpreter.py' -v
```

## Implemented next step: TDL-003 cost and evidence qualification

[TDL-003](COST_AWARE_REFINEMENT_TDL003.md) now implements a bounded exact-cost selector over receiver-declared observer offers, derives an exact row decoder, and uses VGI's actual finalized transition-evidence ledger to accept the selected scalar observation values. The minimal-bit protocol below remains a separate mathematical baseline.

## Discovery for the next round

For nonlinear NCR-007 feature coordinates \(q(x)\), the number of additional distinctions in an old observer fiber is

\[
d_{\mathrm{finite}}=\left\lceil \log_2 \max_{z_o}
\left|\{q_n(x):q_o(x)=z_o\}\right| \right\rceil.
\]

This is an information-theoretic worst-case class-index bound, unlike the affine linear-rank formula. An independently qualified compact class selector could make intention succession cheaper even when the nonlinear feature-space rank is relatively large.

**Next construction:** qualify a finite or symbolic nonlinear fiber-image certificate that retains exact observable equivalence and independently checks all affected classes. Compare auxiliary witness bits, certificate bytes and execution cost with the affine implementation and full-source anchors. In parallel, attach a typed observation authority receipt to the \(w\) acquisition boundary.

The cross-repository translation record is maintained in [In-Formation-Space TDL-002](https://github.com/Kier73/In-Formation-Space/blob/main/reports/TRANSLATION_DISCOVERY_LOG_002.md).
