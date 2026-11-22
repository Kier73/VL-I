# What a glance must preserve

**VL-I asks how little a Virtual Layer needs to see and compute to produce the part of a model’s behavior that someone actually wants.** A model diagram can reveal how information flows. Two learned rows can reveal how a model orders two words. A causal state carries what the model has already read. The opportunity is to combine these kinds of evidence into an executable description, called a *Ground*, and to discover where the usual computation can end early.

## When is a glance enough?

Let $M$ range over the possible trained models, including their tokenizer and numerical conventions. Fix an intention $I$, such as “after this prompt, which token ranks higher, ‘ yes’ or ‘ no’?” Let $Y_I(M)$ be the requested answer and $D(M)$ the evidence available to the interpreter. For an adaptive reader, $D$ includes the queries, the replies, and the identity of any previously acquired state.

An exact answer is determined by a glance precisely when

$$
 D(M_1)=D(M_2) \quad\Longrightarrow\quad Y_I(M_1)=Y_I(M_2).
$$

**Why.** If this holds, assign each observed description the common answer of all its compatible models. This defines $g_I$ on attainable descriptions with $Y_I=g_I\circ D$. Conversely, a function of the same evidence must return the same answer. For numerical answers, two compatible models whose answers differ by more than $2\epsilon$ witness the failure of any uniform error guarantee of $\epsilon$, by the triangle inequality.

This gives an interpreter something concrete to do with uncertainty. If two same-shaped checkpoints could give opposite answers, find the cheapest available observation that separates those possibilities. A screenshot might reveal the family or a revision to look up. A model page might expose a config, a shard index, or a link to an existing certificate. A raw file might supply the few trained coefficients an identity needs. A revision can identify an entire checkpoint while leaving the work of evaluating it ahead. Every lookup, retained summary, and source read belongs in the cost. The repository has machine-readable source readers today. Extracting a claim from pixels and confirming it against source bytes is an open experiment. [Model acquisition plan](RESEARCH_PLAN.md) · [Hosted header reader](https://github.com/Kier73/VL-64/blob/main/experiments/model_fabric/hf_glance.py)

There is a second productive use for exactly the same screenshot or header. Give a description $d$, interpreter $L$, and causal memory $s$ their own transition law $(s',y)=L(d,s,I)$. Then ask what this newly formed system can do on unseen intentions. COORD-002 already compiles a described three-state system into nine ReLU units and checks every supplied transition. Richer language behavior can be tested by changing the description while holding the interpreter and prompts fixed, and by withholding descriptions to measure what the interpreter supplied on its own. [Description-native experiment](COORD_002_DESCRIPTION_NATIVE_GENERATION.md) · [VLD compiler](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/985ba5c2fbcc40842aed9f45f42587749620b98e/src/vld/models/procedural.py)

## What the existing build establishes

| Question | Evidence and its scope |
| --- | --- |
| Can the architecture tell us where to look? | Running VL-I’s inspector on the pinned SmolLM2-135M header identified 17 required operations for the first layer’s feed-forward residual and 454 for vocabulary logits. It observed 272 tensor entries and read zero weight-payload bytes. The corresponding declared weight extents were 63,703,296 and 269,030,016 bytes. These are dependency counts. [Ground implementation](../src/vli/ground.py) · [VL-I-001](VLI_001_FIRST_EXPERIMENT.md) |
| Can those paths carry real trained behavior? | VL-64 records an entire 30-layer SmolLM2-135M evaluation on input IDs 42 and 75, yielding greedy token 33. VL-I’s separate Qwen archive records two genuine paragraph tokens through eight of 64 layers. At position two an independent PyTorch implementation checked layer 7 on the same predecessor input, with maximum absolute error $5.27\times10^{-16}$ across 5,120 coordinates. The Qwen run transferred 14,537,908,768 bytes. [SmolLM2 full-decoder record](https://github.com/Kier73/VL-64/blob/9a08e7246ba4fb046270f88db2d38ab4c19368cb/docs/REAL_FULL_PRETRAINED_DECODER_OBSERVATION.md) · [Qwen temporal receipt](../experiments/qwen_all_heads/datasets/2026-11-19-real-original-qwen-two-token-trained-attention-parity.json) |
| Can a description replace a trained calculation? | SHAPE-009 fitted normalization energy on the triangle between three genuine Qwen causal states. On three new interpolations it produced five coordinates each with maximum error $5.55\times10^{-17}$. Asking for all 5,120 took 9.57 ms against 3.19 ms for the dense reference. The fourth real causal state lay outside the fitted plane with relative residual 0.5343. The failure identifies exactly where a new observation or a different law is needed. [Retained measurements](../experiments/causal_hidden_manifold/results/2026-11-19-original-qwen27b-four-causal-positions.json) |

This connects the public work at the level of mechanisms. In-Formation-Space supplies typed graph descriptions. Virtual-Ground-Interpreter represents unresolved relations and chooses further observations. E-CPU gives trained bytes and stored computations durable identities. VL-64 supplies source-bearing numerical references. Virtual-Layer-Dynamics can compile an explicit law into an executable model. Navigable-Compute and VL-I’s problem-formed interpreter show, for declared binary-affine dynamics, how the observer itself can determine which state registers and transitions must exist. [Source inventory](SOURCE_INVENTORY.md) · [Problem-formed interpreter and upstream construction](PROBLEM_FORMED_INTERPRETER_PFI001.md)

## A trained token preference that can be carried backward

Consider the published **SmolLM-135M** checkpoint. This experiment uses the first-generation model. Take the intention to compare two fixed next-token logits. Let $h$ be the residual entering the final RMS normalization, $g$ its trained gain, and $w_i,w_j$ two learned vocabulary rows in the bias-free output head. In real arithmetic,

$$
 \ell_i-\ell_j =
 \frac{a^\top h}{\sqrt{\|h\|^2/d+\epsilon}},
 \qquad a=(w_i-w_j)\odot g.
$$

The denominator is strictly positive. The sign of one dot product answers the ranking question for **every** $h$, including genuine causal states outside any previously observed span. The magnitude of that dot product has different semantics from a logit gap or a probability.

The architecture yields one more step. The last feed-forward block ends with $h=u+Dv$, where $u$ is the attention residual, $v$ the gated intermediate activation, and $D$ the trained down-projection matrix. Compile $b=D^\top a$ once. Then

$$
 \operatorname{sign}(\ell_i-\ell_j)
   =\operatorname{sign}\!\left(a^\top u+b^\top v\right).
$$

This is the transpose identity applied to the *question being asked*. From $u,v$, the host can stop before constructing the dense final-layer output, applying final RMS normalization, or scoring the vocabulary. The 30-layer causal prefix up to that boundary still supplies $u,v$. Another token pair generally needs its own direction. Floating-point rounding can alter very close rankings, so agreement with a particular runtime is an empirical check. [Experiment source](../examples/smollm_pairwise_readout.py)

**Measured on the pinned checkpoint.** We fixed the two single-token completions “ yes” (9805) and “ no” (787) and six texts before running the comparison. At every one of the **85 prompt positions** the early-stopped pass agreed with the full CPU FP32 model. “ yes” ranked higher 17 times and “ no” 68 times, with zero ties. The smallest absolute reference gap was 0.02166. With all 49,152 tokens eligible, the original model chose a third token at **81 of 85** positions. Each recorded answer concerns the model’s preference at a prompt prefix. [Per-position results, source hashes, and runtime versions](../experiments/pairwise_readout/results/2026-11-21-smollm-135m-yes-no.json)

**A counterexample at the rounding boundary.** Give the genuine checkpoint head a constructed residual whose only nonzero coordinates are $h_0=14884865/2147483648$ and $h_{17}=-26455/65536$. The trained coefficients give $a^\top h=26455/140737488355328>0$, which favors “ yes”. On this machine the original CPU FP32 head produced a logit gap of $-1.1920928955078125\times10^{-7}$, which favors “ no”. This is a head-level stress input. It demonstrates why exact agreement with a chosen runtime needs a numerical margin or a fallback near ties, even when the real-arithmetic identity is proved.

Compiling the directions used 6,912 logical bytes for two trained rows and the gain, plus 3,538,944 bytes for the last down-projection. The two retained FP64 vectors occupy 16,896 bytes. Once $u,v$ are available, two dot products of lengths 576 and 1,536 suffice. Even a baseline that scores only the same two tokens would ordinarily first compute the $576\times1{,}536$ down-projection. The verification harness hashed the complete 538,090,408-byte weight file, loaded the local checkpoint, ran the shortened prefix, and separately ran the full model on the same token IDs. End-to-end speed and physical I/O remain unmeasured.

To reproduce, use the package versions in the receipt, set the environment variable `OPENBLAS_NUM_THREADS` to `1`, and point the command at a local copy of the pinned model. [Checkpoint revision 1d46172](https://huggingface.co/HuggingFaceTB/SmolLM-135M/tree/1d461723eec654e65efdc40cf49301c89c0c92f4)

~~~sh
python -m examples.smollm_pairwise_readout --snapshot "/path/to/SmolLM-135M" --out artifacts/pairwise-readout.json
~~~

## From a readout to a conversation

Matching one answer is weaker than preserving the state needed for the next question. For a fixed model, readout $O$, and allowed continuation words $w$ in a set $A$, two causal states can share a representation if $O(T_w s)=O(T_w s')$ for every $w\in A$, where $T_w$ advances the state. Extending $A$ can split a previously useful class. NAV-003 already exhibits the distinction on finite neural fixtures. The same two states agreed for one coordinate across all tested action histories and diverged for another. Its certified navigation was slower than direct replay. [Observer-future experiment](NAV_003_OBSERVER_FUTURE_EQUIVALENCE.md)

The next experiment can move the stopping point inside the **nonlinear** feed-forward block. Write $v_k=\operatorname{SiLU}(G_kx)(P_kx)$, with $x$ the normalized attention residual and $G_k,P_k$ the gate and up-projection rows. Suppose the interpreter has evaluated only channels in $K$. Its partial score and a conservative bound on the unseen contribution are

$$
 s_K=a^\top u+\sum_{k\in K}b_kv_k,\qquad
 R_K=\|x\|_2^2\sum_{k\notin K}|b_k|\,\|G_k\|_2\,\|P_k\|_2.
$$

The bound follows from Cauchy–Schwarz and $|\operatorname{SiLU}(t)|\le |t|$. The sign is settled in real arithmetic as soon as $|s_K|>R_K$. Otherwise the interpreter can observe another channel. The row norms and an order for reading them can be compiled once. A finite-precision certificate would also have to enclose rounding error. An experiment where almost every channel is required would tell us that a tighter bound is needed.

Freeze the token pair, reading order, prompts, and numerical contract before testing. Compare with a direct two-row readout as well as full vocabulary evaluation. Count prefix work, compile time and bytes, channels examined, fallbacks, and latency on unseen continuations. If the matched baseline costs $B$ per request, the complete alternative averages $Q$ including fallbacks, and incremental setup costs $S$, a saving requires $n(B-Q)>S$ over $n$ requests. Run this comparison separately for wall time and for bytes. A model operation disappears from the cost only when the host actually stops doing it.
