# Classifying a request to the trained Qwen block

E-CPU asks a typed question about a request, gets a proposed execution domain, then checks whether the current state admits the handler. Its separate Ground reuse experiment binds a cheaper evaluation to an observation contract. VL-I translates these two ideas into a small local dispatcher for one observation, the complete layer-zero block output at the final position of a token prefix. The classifier is deterministic. Its answer selects how the VL obtains the vector. [E-CPU classifier implementation](https://github.com/Kier73/E-CPU/blob/d329b8ca6d1f9c94f77e7823a3352cbe66b34707/mailcpu/classifier.py#L910) · [E-CPU Ground reuse experiment](https://github.com/Kier73/E-CPU/blob/d329b8ca6d1f9c94f77e7823a3352cbe66b34707/docs/AI_GROUND_REUSE_AGA002.md)

## Two questions with different lifetimes

**At formation, which columns in a trained row share a coefficient?** Read the original BF16 bytes and check the digest against the staged model manifest. In row $i$, give each distinct bit pattern a local integer $k$. Store a stream of class addresses $I_{ij}$ and a dictionary $B_{ik}$ such that $W_{ij}=B_{i,I_{ij}}$. This classification is independent of context. Each new activation can walk the address stream, accumulate into the named buckets, then apply one trained coefficient per bucket. In real arithmetic the identity is

$$ (Wx)_i=\sum_k B_{ik}\sum_{j:I_{ij}=k}x_j. $$

The interpreter uses binary64 and a fixed summation order. Floating-point parity is tested at the final block output. The program uses a native loop because rebuilding Python dictionaries for every request took about 45 seconds in the earlier experiment. [Coefficient program](../examples/coefficient_plan.cpp) · [Earlier measurements](QWEN_FIRST_BLOCK_HANDOFF.md)

**At execution, can the last answer be reused?** Within a session the model operands, operation and numerical convention are fixed. The dispatcher retains the last input and output as immutable bytes. A byte-for-byte match admits replay. A different residual executes the coefficient program and replaces the entry after successful completion. A nonfinite input is rejected. This rule requires no learned confidence threshold. It applies even when two distinct token histories reach the exact same residual, because this particular sublayer depends only on that residual. A later causal attention layer would additionally need the appropriate history or recurrent state in its key. [Session and admission rule](../examples/classified_qwen.py)

## Cost and scope

There are separate costs for preparing the coefficient program, executing an unfamiliar residual, and returning a repeated result. Replay saves all three matrix projections. On a miss the interpreter still visits each of the 267,386,880 matrix coefficients through its class address and performs 60,999,419 weight multiplications. The three programs occupy 657,092,110 logical bytes for class addresses, dictionaries and row offsets. Allocator capacity, the host interpreter and working memory are additional. The benchmark also retains the original matrices for an in-process native dense comparator. Normal operation releases each original matrix after forming its program.

The benchmark starts after the [source staging step](QWEN_FIRST_BLOCK_HANDOFF.md). It times warm dispatch with the request bytes already in memory, excluding result-file writes. The independently timed native dense path uses the same source matrices, precision, host, normalization and activation code. Both warm paths include the existing normalization weight read and digest check on actual evaluations. The independent end-to-end reference starts from token IDs, recomputes attention, and checks all 5,120 block outputs. PyTorch is needed only for that separate reference. [Route and timing receipt](../experiments/model_pages/2026-11-23-classified-block.json) · [Independent numerical receipt](../experiments/model_pages/2026-11-23-classified-block-oracle.json)

In this WSL run, formation took 4.33 seconds. The median over five misses was 0.234 seconds in the interpreter and 0.231 seconds in the paired native dense comparator. The two consecutive repeats took 8.7 and 56.6 microseconds. The seven requests intentionally include repeats to exercise both branches. Their replay rate says nothing about the frequency of identical residuals in ordinary generation. The compiled library occupies 16,272 bytes alongside the separate Python host and model-specific plan.

A further interpretation experiment would change the requested observation from a full vector to a discrete choice. For example, if a certified region of residual space gives every competing score an upper bound below the chosen score's lower bound, membership in that region is sufficient to dispatch the action on a new context. Such a route needs measured coverage, a sound bound through the nonlinear layers, and a way to charge the cost of classifying the input. The current implementation qualifies replay only on exact state identity.

## Run

Use Linux or WSL with a C++20 compiler, Python and the staged directory from the previous block experiment. Each input is one little-endian FP64 attention residual. Supply multiple cases to reuse the formed session. Append `--benchmark` to retain the dense comparator and check each result against its $10^{-10}$ acceptance threshold.

```sh
g++ -O3 -std=c++20 -ffp-contract=off -shared -fPIC examples/coefficient_plan.cpp -o artifacts/libcoefficient_plan.so
PYTHONPATH=src python3 -S -m examples.classified_qwen \
    --library artifacts/libcoefficient_plan.so --directory artifacts/qwen-mlp0 \
    --case artifacts/a.f64 artifacts/a-block.f64 \
    --case artifacts/a.f64 artifacts/a-repeat.f64 \
    --case artifacts/b.f64 artifacts/b-block.f64
```
