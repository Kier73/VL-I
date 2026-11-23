# Continuing the Qwen attention residual through its feed-forward sublayer

The first Qwen3.8-27B text block has a useful boundary. Attention depends on the history. Once the attention workers produce the residual for a position, the rest of this block depends only on that vector and four immutable trained operands. The same feed-forward program can therefore consume a new residual without recompiling the token-specific head workers. Its input is 5,120 little-endian binary64 values. Its output has the same width and encoding.

For input residual $r$, trained normalization offset $w$, and checkpoint matrices $G,U,D$, the worker evaluates

$$n=(1+w)\odot r/\sqrt{\operatorname{mean}(r^2)+10^{-6}},\qquad
b=r+D\left[\operatorname{SiLU}(Gn)\odot(Un)\right].$$

The intermediate activation has 17,408 coordinates. In each projection the worker groups columns with identical trained BF16 bits within a row, sums the corresponding activations, then multiplies once per group. It processes complete rows in tiles of at most 1 MiB for this checkpoint and verifies the operand digest before returning. The tile budget covers input matrix bytes only. [Implementation](../examples/run_qwen_mlp.py) · [Upstream operator definitions](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

## What the experiment measures

Formation resolves a pinned directory and checks tensor names, shapes, dtypes and extents. The selected bytes occupy 534,784,000 bytes in one shard. Staging downloads each operand once. Each warm run reads every staged operand again, entirely locally. The worker runs with Python site packages disabled. The reference separately reconstructs the original trained attention recurrence from token IDs and uses PyTorch binary64 matrix multiplication, RMSNorm and SiLU through the final addition. Both evaluations interpret the original learned BF16 coefficients in binary64. Production accelerator kernels have their own rounding behavior.

The test compares both the incoming attention residual and outgoing block vector against that reference at all coordinates, with an absolute tolerance of $10^{-10}$. One prefix reuses the previously qualified residual. The other introduces a five-token history ending on the same token, so its token embedding is identical while its context differs. The result applies to the final position of these prefixes. This handoff does not supply all preceding positions to layer one. To continue through the next causal layer, supply the sequence of block outputs or a sufficient recurrent state, and compare future observations after a restart. [Model lift contract](MODEL_LIFT_CONTRACT.md)

The measured maximum block-output errors were $4.44\times10^{-14}$ for the three-token prefix and $1.07\times10^{-14}$ for the five-token prefix. The largest difference between the two candidate outputs was 0.447. Per prefix the grouped method used 60,999,419 weight multiplications across the three matrices, compared with 267,386,880 for the dense equations. Coefficient visits, hash lookups, additions and source reads still occur. [Dense reference and operand identities](../experiments/model_pages/2026-11-23-block-handoff.json) · [Three-token runtime receipt](../experiments/model_pages/2026-11-23-mlp-runtime-first.json) · [Five-token runtime receipt](../experiments/model_pages/2026-11-23-mlp-runtime-fresh.json)

The grouped feed-forward step took 47.30 and 45.02 seconds. Single-threaded PyTorch took 1.61 and 1.62 seconds on the same machine. These per-prefix timings include local operand reads, digest checks and the feed-forward arithmetic. They exclude initial downloads, the attention stage and process startup. Both paths used the OS file cache under ordinary operating conditions. The arithmetic reduction came with a substantial latency penalty in this Python implementation. One candidate for a follow-up experiment is to compile the coefficient groups once and measure the added storage and break-even reuse count against this baseline.

## Reproduction

The [E-CPU classifier translation](CLASSIFIED_QWEN_HANDOFF.md) follows up on grouping costs with a resident coefficient program, a request dispatcher, and a native dense comparator.

Use the sibling VL-64 checkout at `fbc8094` and the head archives from the [formation experiment](COMPOSED_ATTENTION_ROW.md). In the commands below, `artifacts/row.json` is the manifest generated with those archives. Set `PYTHONPATH=src` from the VL-I checkout. The staged operand directory must be new. The candidate commands use the standard library. The last command requires PyTorch and network access for the reference attention operands.

```sh
python -S -m examples.form_qwen_mlp --vl64 ../VL-64 --manifest artifacts/row.json --directory artifacts/qwen-mlp0
python -S -m examples.run_lifted_residual --vl64 ../VL-64 --directory artifacts/qwen-row0 --manifest artifacts/row.json --out artifacts/a.f64 1761 628 33303
python -S -m examples.run_lifted_residual --vl64 ../VL-64 --directory artifacts/qwen-row0 --manifest artifacts/row.json --out artifacts/b.f64 628 1761 628 63204 33303
python -S -m examples.run_qwen_mlp --directory artifacts/qwen-mlp0 --input artifacts/a.f64 --out artifacts/a-block.f64
python -S -m examples.run_qwen_mlp --directory artifacts/qwen-mlp0 --input artifacts/b.f64 --out artifacts/b-block.f64
python -m examples.check_qwen_block --vl64 ../VL-64 --directory artifacts/qwen-mlp0 --case 1761,628,33303 artifacts/a.f64 artifacts/a-block.f64 --case 628,1761,628,63204,33303 artifacts/b.f64 artifacts/b-block.f64 --out artifacts/block-validation.json
```
