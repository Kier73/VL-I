# NOC-001 + PBG-001 — Observer-first nonlinear cuts and predictive Grounds

**Aim:** construct observer-sufficient computational relations rather than repeatedly executing a source neural architecture. These experiments extend existing VL-I source-fidelity, NAV-003 finite-future observation, PFI-001 processor formation, and E-CPU source-ground/route mechanisms. They make different claims, with different proof obligations.

## Cross-repository starting points

| Mechanism | Reuse |
|---|---|
| VL-I [What a glance must preserve](WHAT_A_GLANCE_MUST_PRESERVE.md) and [source-pinned SmolLM pairwise experiment](../examples/smollm_pairwise_readout.py) | Exact readout pullback a=(w_yes−w_no)⊙g, b=Dᵀa; independent CPU FP32 oracle |
| VL-I [NAV-003](NAV_003_OBSERVER_FUTURE_EQUIVALENCE.md) | Finite-horizon source-qualified future-observation parity; not unrestricted state closure |
| VL-I [PFI-001](PROBLEM_FORMED_INTERPRETER_PFI001.md) and E-CPU [INTENT-001](https://github.com/Kier73/E-CPU/blob/main/docs/INTENT_001_OBSERVER_FORMED_FABRIC.md) | Construct observer-relative processors from admitted deterministic laws |
| IFS [RE-001 response equivalence](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/response-equivalence) | Exact-rational Hankel rank and alternate hidden-state realizations |
| E-CPU [model-weight Ground](https://github.com/Kier73/E-CPU/blob/main/docs/MODEL_WEIGHT_GROUND_MWS001.md) | Source identity and structural coordinates separate from numerical execution |

## Path A — nonlinear cut for a requested pairwise token ranking

The source block uses final attention residual u, normalized feed-forward input x, gate/up matrices G/P, and down matrix D:

\[
v_k=\mathrm{SiLU}(G_kx)(P_kx),\qquad
s=a^\top(u+Dv)=a^\top u+\sum_{k=1}^m b_kv_k,\quad b=D^\top a.
\]

For a set K of evaluated MLP channels, use partial score

\[
s_K=a^\top u+\sum_{k\in K} b_k v_k.
\]

Because |SiLU(t)|≤|t|, omitted channels have a conservative real-arithmetic bound:

\[
R_K(x)=\sum_{k\notin K}|b_k|\,
\min(\|G_k\|_2\|x\|_2,\|G_k\|_1\|x\|_\infty)\,
\min(\|P_k\|_2\|x\|_2,\|P_k\|_1\|x\|_\infty).
\]

Whenever |s_K|>R_K, its real-arithmetic sign is determined without calculating the omitted channels. The interpreter compiles b and channel norms once, ranks channels by available bound, evaluates only selected gate/up rows and reports avoided row dot-products. A caller-controlled numerical guard supports conservative abstention near ties. The NumPy implementation uses padded floating-point bounds, **not** a fully formally verified interval library. An FP32-specific exact certificate would additionally need rigorous per-operation rounding envelopes or a fallback. Consequently **the pretrained FP32 parity claim is decided by an independent full-model oracle**, not by assuming that a real-arithmetic sign identity implies bitwise runtime parity.

Implementation: [`nonlinear_observer_cut.py`](../src/vli/nonlinear_observer_cut.py). Synthetic/oracle tests: [`test_nonlinear_observer_cut.py`](../tests/test_nonlinear_observer_cut.py).

The **offline** [SmolLM NOC-001 experiment](../examples/smollm_nonlinear_cut.py) reuses the original verified file pins (SmolLM-135M revision `1d461723eec654e65efdc40cf49301c89c0c92f4`). It stops source computation at the final nonlinear MLP input, computes all observer-cut decisions, then separately performs unrestricted source FP32 inference on four fixed prompt texts not included in the earlier 85-position experiment. It does not request or download any model weight file.

Run against an already available verified local checkpoint:

```sh
OPENBLAS_NUM_THREADS=1 python -m examples.smollm_nonlinear_cut \
  --snapshot /existing/SmolLM-135M --out artifacts/noc001.json
```

No real-pretrained NOC-001 result is claimed until this source-pinned run succeeds. Baselines should include the earlier b·v cut and direct two-row classification, accounting for source causal prefix, precompilation, bounds, fallbacks, and physical latency. The candidate tail performs 2*k*d gate/up multiply-accumulate groups for k chosen channels rather than 2*m*d for all m. The full baseline also includes downstream projection and head work; a lower channel count alone does not prove wall-time savings.

## Path B — exact rational persistent predictive Ground

This is a *different* mathematical class: a fully declared finite stochastic token generator. Let each token a∈A have a nonnegative rational n×n matrix T_a, let α be an initial row distribution, and require Σ_a T_a 1=1. Its complete token-sequence law is

\[
P(a_1\cdots a_t)=\alpha T_{a_1}\cdots T_{a_t}{\bf1}.
\]

Build the **smallest source-global future-observation column space**

\[
V=\operatorname{span}\{T_w{\bf1}: w\in A^\ast\}.
\]

Construct an exact-rational basis B=[1,b₁,…,bᵣ₋₁] by iterating closure under every T_a, then solve and independently verify

\[
\boxed{T_a B=B K_a \text{ for every } a\in A.}
\]

Now the Virtual Layer can retain only z=αB, its r×r matrices K_a and output e₀. Its next-token probabilities and state update are

\[
p(a\mid h)=(z K_a)e_0,\qquad z'=\frac{z K_a}{p(a\mid h)}.
\]

The dimension r may be substantially less than n. For the declared source these relations preserve **all future token-sequence probabilities**, not merely a tested horizon. This r is the minimal source-global *observable column-space* dimension; a particular initial distribution may admit further reachable-state reduction. Source laws and proofs are computed during construction and verification; the online session needs only the qualified Ground and token intention, never the original n-state source matrices.

The [implementation](../src/vli/predictive_basis_ground.py) uses exact Python Fraction arithmetic, independent re-derived closure, source/ground fingerprints, conditional-token transitions, and snapshots replayed and validated on restore. It rejects wrong source fingerprints, forged transitions even after a new hash, invalid stochastic laws, unknown tokens and zero-probability histories.

The [tests](../tests/test_predictive_basis_ground.py) verify 8 source states → 2 predictive coordinates, exact parity on all 511 binary action words of length 0–8, resumed unseen continuations, changed-source counterexamples that increase required rank, and modified snapshot refusals.

The [benchmark](../examples/predictive_basis_benchmark.py) extends exact comparison to **1,023** words of length 0–9 and a 96-token streaming continuation, reporting compile time and incremental physical Python timings against a sparse rational reference, not a theoretical dense-only baseline.

```sh
python -m pytest -q tests/test_nonlinear_observer_cut.py tests/test_predictive_basis_ground.py
PYTHONPATH=src python -m examples.predictive_basis_benchmark
```

## Current evidence and next qualification

- **NOC-001:** synthetic numerical-oracle and adversarial tests pass. Original trained SmolLM execution is wired but requires its exact local source snapshot. No network data acquisition is part of the runner.
- **PBG-001:** constructed rank-2 predictive law from rank-8 source; 1,023 bounded word-history comparisons passed in the local benchmark, and source-free virtual transitions proceed for a held-out 96-token continuation. All-horizon correctness for this *declared finite source law* follows algebraically from the verified closure, not from extrapolation of those 1,023 samples.
- **Cost:** local structural dimension and timing are measured; the compiled Ground is not yet claimed to improve end-to-end LLM cost. Reference matrices in the fixture are sparse; real performance comparisons must retain sparsity.
- **Next:** bind PBG candidate predictive coordinates to source-qualified neural causal observations and formulate source-identity-preserving evidence acquisition through E-CPU and VGI. A real LLM does **not** supply exact finite stochastic T_a tables by assumption: discovering and independently verifying a useful candidate representation from selected trained observations is the next major experiment. Vary observation horizons and alphabets and require counterexamples to refine the Ground.
