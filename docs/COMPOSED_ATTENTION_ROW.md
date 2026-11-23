# One Qwen attention coordinate across all 48 heads

Starting at the pinned Qwen3.8-27B directory, the compiler shared one tensor map and one copy of the requested trained bytes among 48 head programs. Each worker evaluates its gated value-head vector from the supplied prefix. For output row zero the coordinator computes $y_0=\sum_h W_{0,h} z_h$. This is the complete attention branch at that coordinate, before residual addition. The implementation runs the workers sequentially on one machine.

Three fresh prefixes contained three, four and six tokens from the compiled four-ID alphabet. The independent PyTorch binary64 reference consumed the same original bytes, projected the full unsliced QKV and gate matrices, maintained all 48 dense recurrent states forward in time, and applied the original projection row to the concatenated gated vectors. The maximum row discrepancy was $6.51\times10^{-15}$. Checking all 6,144 head coordinates on each prefix gave a maximum discrepancy of $1.88\times10^{-13}$. The acceptance threshold was $10^{-10}$. [Numerical receipt and component identities](../experiments/model_pages/2026-11-23-composed-row.json)

Cold formation transferred 168,901,056 trained bytes in 52 range requests. The 48 compilers logically requested 255,271,104 bytes. Sharing saved 86,370,048 bytes of repeated operand reads. The tensor metadata cost 267,400 bytes plus the directory pages. All worker archives together occupy 1,299,614 bytes. The largest individual worker expands to 45,478 bytes including data. A fresh process verified their hashes and reproduced the first row value from the archives alone. No wall-time or arithmetic speedup was established.

## Reproduce

Use VL-64 commit `fbc8094` and run the commands below from VL-I. The first command needs NumPy, CPU PyTorch and network access. The second reuses the local workers for any nonempty prefix within the model context over token IDs 628, 1761, 33303 and 63204.

~~~sh
python -m examples.composed_attention_row --vl64 ../VL-64 --directory artifacts/qwen-row0 --out artifacts/row.json
python -m examples.run_composed_row --vl64 ../VL-64 --directory artifacts/qwen-row0 --manifest artifacts/row.json 628 33303 1761
~~~

For propagation into the next sublayer, all 5,120 output coordinates are needed. The reusable intermediate is the concatenation of the 48 gated head vectors. Compute it once per prefix, apply the remaining trained output rows to that vector, then add the embedding residual before feed-forward normalization. This avoids recompiling the recurrent heads separately for each output coordinate.
