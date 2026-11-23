# Lifting a trained model into the virtual layer

The target is a VL program that can accept a fresh context, advance the state of a particular trained model, and emit its next token or action. At deployment the host supplies immutable bytes, and the VL implements the model transitions and decisions. A build may use a dense reference to challenge the result. Formation, validation and online operation each consume resources and should have separate accounts.

## What a lift has to preserve

Fix a checkpoint revision, operator semantics, numerical convention, input alphabet, context limit and observation. Write the resulting model as a state transition $T(s,a)$ and a readout $R(s)$. State includes everything a continuation can inspect, such as position, convolution history and recurrent or attention memory. For text, include the tokenizer and prompt template in the identity. Given a representation $p$, a virtual transition $V$ and decoder $D$, an exact lift must satisfy on the declared reachable domain

$$v_0=p(s_0),\qquad V(p(s),a)=p(T(s,a)),\qquad D(p(s))=R(s).$$

Induction then covers every admitted continuation. A worker that recomputes from the whole prefix may retain the prefix as its state. A smaller restartable state needs its own closure argument. Matching one residual vector establishes a finite observation. A later token can expose information lost by a proposed state compression. The repository already proves the conditional induction and future-readout result in Lean. Proving a specific implementation satisfies its premises is a separate obligation. [Continuation theorem](../formal/Continuation.lean)

For approximate transitions let $e_t$ bound state error, $L_t$ bound local amplification on a certified reachable neighborhood, and $\delta_t$ bound new operator error. Propagate $e_{t+1}\le L_t e_t+\delta_t$ through each sublayer. If the resulting logit error is at most $\epsilon$ in maximum norm, an observed top-two gap greater than $2\epsilon$ certifies a greedy decision. Sampling needs a specified sampler, numerical convention and random stream. For inverse-CDF sampling, bound the cumulative probabilities around the common random draw before claiming the same token. An agent benchmark should compare emitted tool identifiers and arguments, feed the same recorded tool response to both runs, then compare the next decision. The external tool performs the real-world effect.

## Formation from a public page

1. **Bind the trained identity.** Resolve the directory to an immutable revision. Bind each operand to its shard, dtype, shape, byte range and decoding convention. Record digests for reuse. A page image may supply a starting address. Two checkpoints with identical topology can have different output weights and choose different tokens on the same input. Distinguishing them requires trained evidence.
2. **Lower the architecture.** Turn configured layers into a typed dependency graph. Specify axis order, tied parameters, masks, positions, normalization, recurrence initialization and update order. For quantized weights include scale, block layout and zero point. An unrecognized operator leaves its downstream observation unresolved.
3. **Start from the requested observation.** Trace dependencies backward from a residual, logit, sampled token or structured action. For each live edge, bind original bytes, derive an equivalent expression, or supply an error bound. For a branch or routed expert, record what certifies the decision and when further source reads become necessary.
4. **Emit workers and their composition.** A worker identifies its admitted inputs, immutable operands, local state, equations and output coordinates. The coordinator binds output to input with explicit order and units. Shared immutable operands may be reused across prefixes. Reusing a state requires agreement on the causal history or a proven equivalence of states.
5. **Challenge, then run independently.** Compare the candidate with a reference reading the original trained bytes. Include unseen histories, boundary positions, perturbed weights, restarts and near-tied decisions as applicable. Run the candidate separately with the reference backend unavailable. Record the exact domain, tolerances, mismatches and unimplemented paths.

For an observation over a dense layer, even one unseen coefficient can matter. If $z_j\ne0$, changing only $W_{ij}$ by $d$ changes output $i$ by $dz_j$. Skipping that address therefore needs some other information, such as a previously compiled summary, an exact dependency or an explicit error budget. The research question is which such information remains small across many contexts.

## A measured instance

On the pinned Qwen3.8-27B checkpoint, 48 VL-64 worker programs evaluate layer-zero attention heads for prefixes over four token IDs. VL-I concatenates their 6,144 gated coordinates and computes the first attention residual $r=E[t]+W_o z$. In row $i$, let $J_{ib}$ contain the columns whose original BF16 encoding equals $b$. Distributivity gives

$$r_i=E[t]_i+\sum_b \mathrm{decode}(b)\sum_{j\in J_{ib}}z_j.$$

The program uses binary64 arithmetic, so changing the addition order can change rounding. Over two held-out prefixes the maximum absolute disagreement across all 5,120 coordinates was $7.66\times10^{-13}$ against an independently evaluated dense binary64 recurrence and projection. The original output matrix occupies 62,914,560 bytes. On each prefix the projection visited 31,457,280 coefficients and used 8,041,091 weight multiplications. Dictionary lookups, additions, memory traffic and the head computations still cost time. These counts alone establish neither latency nor energy savings. [Validation receipt](../experiments/model_pages/2026-11-23-lifted-residual.json)

The 48 archives occupy 1,299,614 bytes together. Each expanded worker fits in 64 KiB. This boundary excludes the interpreter, coordinator, source bytes and working memory. The candidate requires an existing set of compiled workers, reads the projection and final token embedding from the pinned host and returns a binary64 vector. This experiment ends before feed-forward normalization, later layers and token selection. [Worker formation and archive identities](COMPOSED_ATTENTION_ROW.md)

## Proposed mechanisms and rejection tests

The standalone run with site packages disabled materialized a 40,960-byte vector for prefix `1761 628 33303`. Its SHA-256 matched the vector from the validation harness. It read 62,924,800 trained bytes beyond the precompiled workers, plus model metadata. [Standalone receipt](../experiments/model_pages/2026-11-23-isolated-residual.json)

| Hypothesis | Experiment that would distinguish progress from a lookup |
| --- | --- |
| Future observations admit a smaller state. Two states may share a representation when every allowed suffix produces the same readout. | Search for histories colliding under a proposed representation, append adversarial suffixes and compare against the source recurrence. A divergent future disproves that representation. Track representation size as alphabet and horizon grow. |
| A downstream request can be transported backward across linear maps by $u^T Wz=(W^T u)^Tz$. | Compile the transformed readout once and reuse it on fresh contexts. At a nonlinear boundary measure how many independent coordinates must be recovered. If they span the original width, charge the full width. |
| A source-derived shared basis $B$ can replace repeated projections, with $\|(W-UB)z\|_2\le\|W-UB\|_2\|z\|_2$. | Sweep rank, bound the residual, and charge storage, formation and online work. Reject a rank when the bound fails the downstream decision margin or the certified execution costs more than the baseline. |
| Context can guide selective expert reads. A lower bound on the weakest selected router score above every excluded upper bound fixes top-$k$. | Challenge close router decisions, account for reads needed to certify the margin and refine ambiguous cases. Compare selected experts and final logits against dense evaluation. Report the frequency and cost of full refinement. |

Report cold formation, oracle validation and warm execution separately. For $N$ requests compare $C_{\rm form}+\sum_{n=1}^N C_{\rm VL}(n)$ with the baseline over the same requests, precision and hardware. Record transferred and retained bytes, peak resident memory, worker count, wall time and arithmetic. The next handoff for this Qwen experiment is a consumer of the actual residual vector that reproduces the trained feed-forward output for fresh prefixes. A generation claim additionally requires every subsequent transition through token selection and feedback.

## Reproduce the residual

Use the [worker formation instructions](COMPOSED_ATTENTION_ROW.md) first. From VL-I set `PYTHONPATH=src` in the environment. Use the manifest produced alongside your workers if regenerating them. The first command writes 5,120 little-endian FP64 numbers with Python site packages disabled. The second uses PyTorch for validation.

```sh
python -S -m examples.run_lifted_residual --vl64 ../VL-64 --directory artifacts/qwen-row0 --manifest artifacts/row.json --out artifacts/residual.f64 1761 628 33303
python -m examples.lifted_residual --vl64 ../VL-64 --directory artifacts/qwen-row0 --manifest artifacts/row.json --out artifacts/residual-validation.json
```
