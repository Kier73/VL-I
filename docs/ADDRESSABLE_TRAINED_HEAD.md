# Addressing a trained transition in Qwen3.8-27B

The address in this experiment selects a checkpoint revision, layer zero, a recurrent value head, an output-projection row and a token prefix. Its value is the last gated head vector and that head's contribution to the selected row. Layer zero matters because its input before normalization is the trained embedding for each token. A later layer would require the preceding layers' contextual hidden states.

The compiler follows the pinned model directory, checks the config and tensor layouts, and selects the corresponding embedding, projection, convolution, gate and normalization coordinates. It specializes the token-local projections for four IDs. The generated program carries these coefficients plus a recipe for reading other IDs from the checkpoint. An unseen ID requires its original embedding row and the shared projection operands. Newly formed entries are reused within the invocation.

For a normalized key $k_t$, value $v_t$, update gate $b_t$ and decay $f_t$, the recurrent update is

$$S_t=f_t(I-b_t k_t k_t^T)S_{t-1}+b_t k_t v_t^T.$$

To observe $S_t^T q_t$ from zero initial state, start with the scaled query $u=q_t$ and an empty output vector. Visit events backwards. Add $b_i(k_i^T u)v_i$ to the output and replace $u$ by $f_i(u-b_i(k_i^T u)k_i)$. This follows by substitution into the recurrence. The program computes the causal convolution in forward order and applies the original gated normalization and output row after this reverse pass. It keeps per-token events and avoids materializing the dense recurrent state. Long histories incur growing storage and replay cost.

The 27,103-byte executable contained 45,475 bytes of expanded code and data. All seven tested prefixes matched the independent PyTorch binary64 forward recurrence within the prespecified tolerance. The maximum absolute error across the 128 head coordinates was 1.81e-16 or less. Two sequences shared their last five tokens, longer than the four-tap convolution, yet their row contributions differed by 0.0010349561. Six prefixes needed zero further weight reads. The seventh resolved token 948 at runtime and read 5,283,840 trained bytes. Compilation itself read 5,318,148 trained bytes plus metadata. The reference shared the source slices and independently evaluated their numerical operations. Two earlier published row values supplied additional checks. [Measured results](../experiments/model_pages/2026-11-22-addressed-qwen-head.json)

With sibling VL-I and VL-64 checkouts on the `codex/addressed-recurrent-head` branches, run the following from VL-I. Compilation and comparison require NumPy and CPU PyTorch. The generated executable uses the Python standard library.

~~~sh
python -m examples.addressed_head_experiment --vl64 ../VL-64 --archive artifacts/qwen-head0.pyz --out artifacts/head-report.json
python -I -S artifacts/qwen-head0.pyz 63204 628
python -I -S artifacts/qwen-head0.pyz --resolve 63204 948 628 948
~~~

The tested observation ends before summing the other 47 head contributions, applying the residual and feed-forward transformations, and executing the remaining layers. The next compositional test is to sum all 48 contributions to one shared output row and compare that scalar against the original first-layer attention output on fresh prefixes.
