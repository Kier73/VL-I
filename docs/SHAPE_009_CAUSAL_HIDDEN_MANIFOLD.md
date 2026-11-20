# SHAPE-009 — Original causal interlayer states as a generative observer Ground

**Research contract:** The source states are the true outputs of consecutive trained Qwen GDN layer-0 causal transitions, with exact native receipts that confirm each output became the next layer's input. The observer's inferred output law is qualified only for a bounded mathematical family of interpolations of those source states. A real next-token causal state outside that family is a counterexample and must be observed, not fabricated.

**Implementation:** [causal_hidden_manifold.py](../src/vli/causal_hidden_manifold.py)  
**Native layer-chain integration:** [qwen_layer_chain.py](../src/vli/qwen_layer_chain.py)  
**Experiment:** [qwen_causal_manifold_demo.py](../examples/qwen_causal_manifold_demo.py)  
**Adversarial tests:** [test_causal_hidden_manifold.py](../tests/test_causal_hidden_manifold.py)  
**CI:** [vli-shape009.yml](../.github/workflows/vli-shape009.yml)

## Existing computational infrastructure, directly reused

SHAPE-008 succeeded with two real source-qualified token embeddings, but interpolations of static embedding rows are not context-dependent prefill hidden states.

QUALIFIED-006 already operates two complete original-width consecutive Qwen GDN+SwiGLU layers through the original VL-64 native operators, across real token positions. Each layer-0 output is the exact layer-1 input, with separate recurrent histories and an externally pinned checkpoint. Earlier original-scale evidence showed the first two layers but explicitly lacked an independent original-weight PyTorch/Transformers numerical reference.

SHAPE-009 adds a receiver-local intermediate observation port to QwenCausalGDNChain. It captures exact full-width complete layer outputs only after the existing source receipt passes, and rechecks every captured F64 digest against the same layer's native witness. This changes neither the ordinary layer output nor its semantic causal head.

Three consecutive actual layer-0 outputs become source-basis states h0, h1 and h2; each is the actual complete original layer-1 input. Source evidence includes original token ID, model/revision/index Ground, original transition position and predecessor head, both native layer output digests, convolution and recurrent state digests, original range-chain receipts and exact E-CPU content-addressed bytes.

## Discover a smaller computational law from actual contextual states

The source-qualified observer's admitted *mathematical* family is a convex combination of three actual causal states:

\[
x(s,t)=h_0+s(h_1-h_0)+t(h_2-h_0),
\qquad s,t\ge 0,\quad s+t\le 1.
\]

These interpolations are **not** claimed to be real future native causal states. They are valid new numerical inputs to the original trained layer-1 normalization operation under this observer's explicitly admitted mathematical domain.

For the original zero-centered native Qwen normalization, the observer needs the full input energy

\[
E(s,t)=\frac1n\sum_j x_j(s,t)^2
\]

to generate an arbitrary selected output coordinate

\[
y_i(s,t)=\frac{x_i(s,t)(1+w_i)}{\sqrt{E(s,t)+10^{-6}}}.
\]

Rather than supplying the coefficients, SHAPE-009 observes six complete original VL-64 native operator outputs at different source-bound (s,t) points. It infers the energy seen in those output observations and fits four candidate families: constant, affine, separable quadratic and quadratic with cross term. Three further independently observed native states challenge them. Wrong models must fail the output-error tolerance and are preserved as counterexamples. Only a unique, correctly qualified quadratic-with-cross-term law is admitted.

There is an independent algebraic reason a full quadratic is sufficient on the admitted affine plane. The six coefficients follow Gram dot products of the three actual source-basis vectors. That identity is used to **recheck the inferred coefficients**, not to secretly supply coefficients during the original observation-based fitting. It establishes a form within this exact source family, **not** a universal theory for arbitrary GDN recurrent transitions.

Before generating a new output, the observer forms a source-/intention-/causal-head-bound **state-position coordinate**. Only the requested numerical coordinates are formed using the inferred scalar energy law and the relevant original-trained E-CPU coefficient pages. Generating a sparse output does not need to create the entire hidden vector again. Generating the entire vector must incur proportional output costs; the experiment retains a full-output negative control.

A newly observed **fourth actual original causal token position** is explicitly withheld. It is checked against the earlier 2D state plane. An out-of-plane position is a real source-verified counterexample: the interpreter **refuses** to claim the old Ground generates that true future state. This is not a failure of the declared convex-interpolation mathematical law. It shows where that law stops, and where a new genuine causal observation must expand the topological basis.

## Preserving the computational evidence

- The receiver uses the original native VL-64 Qwen GDN complete-layer mathematics, not a substituted 64-element emulator in the original-scale run.
- E-CPU's **original ComputationalObjectStore and StructuralMemoryPageBacking** retain three complete real layer-0 output vectors, original layer-1 trained normalization raw bytes and the resulting trained gains.
- The response Ground stores the actual original source token positions, source/causal receipts, native layer-1 trained normalization weight SHA and original model index identity. No per-request output lookup table is saved.
- Restoring the Ground requires an **externally pinned checkpoint digest**, rechecks all immutable E-CPU pages, reconstructs the native complete-layer semantic causal heads from their full witnesses, verifies original F64 handoff digests, re-derives the trained gains from the retained original norm bytes, independently checks the learned quadratic coefficients and **re-evaluates every previously generated observer response** against its original causal address.
- A recovered interpreter can generate an additional, never-before-requested in-span state **without the original pretrained network source or original native normalization calls**, and the original native operator can independently check that novel output afterward.

A receipt and its digest are **integrity evidence**, not third-party remote authentication. No independent whole-layer PyTorch oracle has yet been exercised on the original 27B checkpoint in this experiment. The original VL-64 layer executor remains the source operator; its numerical parity with a distinct original-model implementation is a separate research milestone.

## Two separate test modes

**Small deterministic native fixture:** actual VL-64 trained GDN+SwiGLU layer code advances a three-dimensional recurrent state through two complete trained layers for four consecutive input tokens. These are genuine causal hidden states of that synthetic small model; the parameters and external input are deliberately fabricated, not original Qwen pretrained evidence. The tests check exact layer-to-layer handoff, non-collinearity of the observed states, learned wrong-hypothesis rejection, held-out numerical correctness, causal position formation, genuine next-source-state refusal, original E-CPU checkpoint recovery and rehashed tampering.

**Full original pretrained Qwen3.8-27B:** the separate live job uses the original checkpoint and original paragraph tokenizer, including previously evidenced token IDs 63204, 628, 1761 and 33303. It executes all four actual causal positions through the complete original native layer 0 and layer 1, retains the first three real layer-0 outputs as the trained source basis and tests the fourth genuine causal output as an out-of-manifold counterexample. All original 5,120 hidden coordinates are preserved under exact F64 source receipts.

The learned normalization Ground generates genuinely **new observer-in-span state positions** and only requested output coordinates before consulting a native trained reference. The live job measures original source-block traffic, native two-layer bootstrap work, source/proof storage, fitting and falsification, sparse generated versus native full normalization, a full-output negative control and recovery. If real original-weight source execution fails, it reports an unresolved result rather than promoting the fixture as original pretrained proof.

## Live original pretrained proof — November 19, 2026

The [main-branch original-Qwen SHAPE-009 run 44668309457](https://github.com/Kier73/VL-I/actions/runs/44668309457) **succeeded** against the full, pinned **Qwen/Qwen3.8-27B** pretrained weights and the same original tokenizer paragraph previously qualified in VL-I. Original token IDs **63204, 628, 1761 and 33303** were executed through **both complete original native 5,120-dimensional GDN+SwiGLU layers 0 and 1**. The layer-0 outputs, exactly matching the native layer-1 inputs, were source-bound by their complete original semantic chain, recurrent/convolution state and F64 output digests.

**No model downscaling or invented token-derived hidden inputs.** The first three source-basis vectors are genuine contextual outputs of the actual native original-model causal trajectory; the fourth is a withheld true causal counterexample.

Six native layer-1 normalization observations provided the quadratic coefficients. Three more native observations challenged the four possible law families:

| Candidate energy law | Maximum counterexample error |
| --- | ---: |
| Constant | 3.1070 |
| Affine | 0.51239 |
| Separable quadratic without cross term | 0.33188 |
| **Complete quadratic with cross term** | **3.55 × 10⁻¹⁵** |

The selected coefficients also independently matched the Gram algebra of the original source-verified contextual state plane. **Three new, previously unobserved in-plane observer intentions** were then assigned distinct causal response coordinates and generated *before* the original VL-64 normalization reference was called.

| Previously unseen in-plane intention (s,t) | Generated sparse observation | Original native full norm reference | Maximum absolute error |
| --- | ---: | ---: | ---: |
| (0.13, 0.27) | **0.529 ms** | 3.023 ms | 0 |
| (0.44, 0.19) | **0.409 ms** | 3.034 ms | 6.94 × 10⁻¹⁸ |
| (0.08, 0.73) | **0.419 ms** | 3.030 ms | 5.55 × 10⁻¹⁷ |

Each sparse request selected five out of 5,120 original output coordinates. There were **zero original pretrained HTTP range reads and zero native normalization calls during the generation of the new outputs**. The original native reference was called **afterward**, and all results satisfied the declared tolerance.

**Actual fourth causal position as a negative semantic control:** original token 33303 truly advanced the original GDN recurrent layer states and yielded a verified layer-0 output. Its relative residual from the affine plane of the first three genuine contextual states was **0.5343**. The old source-qualified Ground **refused to generate this real next causal state**. The second-to-third source-rank expansion had already been observed with a relative residual of **0.5770** against the initial two-state line. These are empirical observations of evolving native causal topology, not proof of a universal low-dimensional model of that topology.

**Source-free recovery:** E-CPU retained three complete 5,120-dimensional original layer-0 output vectors (**122,880 F64 bytes**) and original layer-1 trained normalization weights/gains, with source and causal proof manifests. An **approximately 373 KB** externally pinned checkpoint independently reconstructed original state pages, the original complete native-layer semantic chain, source trained gains, the learned mathematical law and all previous generated outputs in **15.64 ms**. It then generated an additional **unseen in-plane** state whose independent native error was approximately **1.11 × 10⁻¹⁶**.

**Cost and negative performance result:** actual original model source acquisition plus execution of four causal positions through two complete original GDN layers took about **258 seconds**, with **6.459 GB of original remote HTTPS transfer** and **770 HTTP source blocks** serving 93,679 logical 64 KiB range glances. The learned law's nine native normalization witness calls took about **25.23 ms**. Asking the current interpreter to generate **all 5,120 output coordinates** required **9.571 ms**, whereas the native complete normalization reference required **3.191 ms**. Full-output interpretation was approximately **3× slower**. The sparse warm advantage therefore must be amortized against expensive original source observation and does not constitute a full pretrained decoder acceleration.

**Independent-whole-layer oracle caveat:** this run verifies *original pretrained source-qualified native VL-64 complete layers and their original numeric handoffs*, together with independently checked native **layer-1 normalization** laws. It **does not** compare complete original 27B Qwen layers against a separately implemented original-weight PyTorch/Transformers reference. That separate original-model parity check is still required. It also does not run all 64 decoder layers, compute vocabulary logits or produce generated language.

[Machine-readable live original-model evidence](../experiments/causal_hidden_manifold/results/2026-11-19-original-qwen27b-four-causal-positions.json) contains the exact original source/causal and learned-law evidence, counterexample geometry, performance counters, source-free checkpoint and negative controls **without original model weight payloads**.

The original pretrained job is now **explicit opt-in for repeat experiments**, rather than downloading more than 6 GB again after routine documentation changes; the local native fixture remains in CI.

## Next architectural move

The ambitious frontier is not merely interpolating contextual states: it is **learning which source observations allow us to navigate to the next true causal state**. A fourth real observed position outside the admitted plane is a concrete active-attention signal to add a new source direction, then search for higher-order or observer-quotient relations that explain the evolving recurrent topology.

VGI's partial-aware active attention, IFS's black-box observational learning, VLD's proof-producing rewrite system and E-CPU's complete original-layer HFG-003 handoff should be integrated at that level. Only a complete original trained layer with a proper independently qualified receipt can be published as a full native handoff. A learned layer-1 **normalization projection** is not the actual complete layer-1 output.

**Long-term intention:** the minimal interpreter navigates a living graph of verifiable computational states; new observations refine its morphology, and a source- and observer-qualified generative Ground forms response states with work proportional to the requested result, not automatically proportional to every inherited intermediate operation.
