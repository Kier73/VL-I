# Can a worker determine DeepSeek’s route from a partial weight read?

**First result.** On three supplied inputs, the interval method identified the same six experts as an independent PyTorch FP32 evaluation of the pinned DeepSeek layer-0 text router. It needed **98.48% to 99.63%** of the gate matrix per query. Counting the one-time summary scan made the three-query run more expensive in local gate bytes than three dense scans. [Recorded source ranges, survivor counts, and oracle results](../experiments/router_bounds/2026-11-22-deepseek-layer0.json)

## What was observed

The worker used VL-64’s Hub range reader and quantized shard-header validator on revision 08c8657f02291e11b6f6b63e0e16014f8e748518. It fetched 7,732,101 metadata bytes, a 384 × 5,120 BF16 gate matrix (3,932,160 bytes), 384 FP32 text-selection biases (1,536 bytes), and three 10,240-byte embedding rows. Token IDs 42, 75 and 63204 were fixed before execution. Their raw embeddings served as router inputs. A real prompt supplies contextual states after preceding trained operations, so causal-route coverage remains open. The observable here is the unordered set of routed experts. Computing their mixture weights and outputs is a further operation. [Published router equations](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/model.py)

## The certificate

For expert $e$, express the observed floats exactly as $w_e=W_e/d_e$ and $x=X/d_x$ with integer vectors and positive dyadic denominators. The cold scan stores $d_e$ and $\sum_i W_{ei}^2$ for each row. On visited coordinates $K$, maintain the integer dot product $s_e$ and unread energies $r_e=\sum_{i\notin K}W_{ei}^2$, $r_x=\sum_{i\notin K}X_i^2$. Cauchy–Schwarz puts the true dot product in

$$
\left[\frac{s_e-\lceil\sqrt{r_e r_x}\rceil}{d_e d_x},
      \frac{s_e+\lceil\sqrt{r_e r_x}\rceil}{d_e d_x}\right].
$$

The router’s selection function $f_e(t)=\sqrt{\log(1+\exp t)}+b_e$ is increasing. Mapping both endpoints through it with outward rounding encloses the real-arithmetic selection score. Let $L_{(6)}$ be the sixth-largest lower bound among surviving rows. Discard $e$ only if $U_e<L_{(6)}$: at least six rows already beat it. Strict inequality preserves contenders tied at the boundary. If ambiguity survives the last block, the worker abstains. The FP32 selection comparison is empirical because the reference rounds its operations differently.

## Measured queries

The worker ranked contiguous 128-coordinate blocks by the supplied input’s squared energy. It read the next block only for experts still in contention. The candidate used a locally staged file, with no network traffic during queries. The dense FP32 oracle ran after all three candidate decisions.

| Embedding ID | First elimination after blocks read (of 40) | Query gate bytes | Gate bytes saved | FP32 sixth–seventh score gap |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 37 | 3,885,056 | 1.20% | 0.001417 |
| 75 | 37 | 3,872,256 | 1.52% | 0.010041 |
| 63204 | 38 | 3,917,568 | 0.37% | 0.030283 |

The 10,723-byte serialized norm summary required one full gate scan. Summary construction plus candidate queries read **15,607,040 local gate bytes**, compared with **11,796,480** for three dense queries on the same staged matrix. These counts exclude the common initial transfer, oracle verification and filesystem caching. Each candidate also issued over 15,000 small file reads. The result establishes numerical agreement on these inputs; wall-time acceleration was not measured.

The largest final routing margin coincided with the latest first elimination. A row norm allows the unread part to align with the input in the worst possible direction. One discriminating follow-up would keep a small shared orthogonal basis $P$ and record each row’s projection. For $P^\top P=I$, the unexplained dot product is bounded by $\|(I-PP^\top)w_e\|\,\|(I-PP^\top)x\|$. Freeze the basis before collecting held-out **causal** states, then charge for its construction, projections and storage. The experiment succeeds economically only if the tighter bounds avoid more work than those steps consume.

## Reproduce

With Python 3.13, NumPy, CPU PyTorch and a VL-64 checkout of commit bb0aac8, run the following from VL-I. A live run needs read access to the pinned public files. The tests exercise interval enclosure, early elimination and an unresolved tie without accessing the network.

~~~sh
python -m examples.deepseek_router_bounds --vl64 ../VL-64 --out artifacts/deepseek-route-bounds.json
python -m pytest -q tests/test_deepseek_router_bounds.py
~~~
