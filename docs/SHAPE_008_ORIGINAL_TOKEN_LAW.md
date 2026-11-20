# SHAPE-008 — Learn a generative operator law from actual token-conditioned states

**Research status:** source-observed degree/parameter inference, independent falsification and generative held-out predictions, integrated with original VL-64 pretrained tensor readers and E-CPU's actual immutable byte store.

**Implementation:** [token_conditioned_law.py](../src/vli/token_conditioned_law.py)  
**Experimental program:** [token_conditioned_law_demo.py](../examples/token_conditioned_law_demo.py)  
**Original-scale and synthetic CI:** [vli-shape008.yml](../.github/workflows/vli-shape008.yml)  
**Adversarial tests:** [test_token_conditioned_law.py](../tests/test_token_conditioned_law.py)

## 1. What changed from SHAPE-007

SHAPE-007 supplied a known finite-trigonometric identity and tested whether it could generate novel **sinusoidal procedural hidden-state** normalization outputs against actual original pretrained Qwen3.8-27B weights. It succeeded for sparse outputs but **did not use token-derived hidden states** and did not infer the law from original operator observations.

SHAPE-008 starts from **two original pretrained token embeddings** instead. The experiment reads token rows through the existing VL-I INTERACT-003 `observe_token_row` original-range observer and the existing VL-64 original model index, using E-CPU ComputationalObjectStore to retain the two actual rows and the real Qwen layer-0 normalization coefficients.

The original token IDs **63204 and 628** were already evidenced in VL-I's [original Qwen paragraph-tokenizer experiment](../experiments/pinned_text/datasets/2026-11-19-real-paragraph-token-causal-results.json). They are original pretrained token embeddings, not synthetic sinusoidal hidden vectors. The experiment does **not** equate an interpolation of those rows with a real Qwen prefill/decoder state: the observer's declared domain is the affine span of the two independently observed embeddings.

Given an intention to locate a point inside that span, the response coordinate is the **position of the requested state**, constructed from original model Ground + two token identities + requested mixture + observer projection + causal predecessor. Its numerical output is formed *after* that address exists.

## 2. Learn a relation from original operator observations

Let (u,vin\mathbb{R}^{5120}) be two original token embedding rows, observed from the real, pinned checkpoint. The observer's admitted input family is:

[
x(t)=(1-t)u+t v, quad t\in[0,1].
]

The native original VL-64 layer-0 normalized output is

[
y_i(t)=\frac{x_i(t)(1+w_i)}{\sqrt{E(t)+\epsilon}},
\qquad E(t)=\frac{1}{n}\sum_j x_j(t)^2,
\quad\epsilon=10^{-6}.
]

**Crucially, candidate coefficients are not specified in the model description.** The learner runs original native VL-64 normalization at three input positions (t=0,\frac12,1), derives scalar normalization energy from its output and the source-qualified input/gain using an observation-only factor estimator, and fits candidate **degree-0, degree-1, and degree-2** polynomials. Independent original native observations at (t=\frac14,\frac34) challenge each candidate. A candidate that fails the declared output error tolerance is rejected; only a unique qualified candidate is promoted.

On the chosen span, an independent mathematical derivation confirms *why* a quadratic can represent the whole energy family:

[
E(t)=\frac{\lVert v-u\rVert^2}{n}t^2
+\frac{2\langle u,v-u\rangle}{n}t
+\frac{\lVert u\rVert^2}{n}.
]

This independent formula is used to **requalify** the inferred coefficient witness against the true retained token rows, not to silently produce the fitted coefficients during acquisition. The law's degree is selected by counterexamples, its numerical coefficients come from actual original native outputs, and the span identity provides structural justification for novel (t).

**Epistemic scope:** this is genuine observation-driven fitting *within a bounded polynomial hypothesis family* chosen by the experiment. It is not unrestricted autonomous mathematical invention, a newly trained general language network, or an inductive proof about arbitrary incoming hidden states. The original source may contain states outside that span; these are deliberately not admitted.

## 3. Generate unseen computational states

After independent source qualification:

1. The observer requests a *new* (t), output channel indices, and exact causal head. It does not supply a desired output vector.
2. VL-I constructs a source- and intention-bound **state-position coordinate before generation**.
3. The interpreter uses the learned quadratic energy and verifies only the relevant original E-CPU token/gain pages.
4. Each requested coordinate is formed by evaluating the learned law. Original pretrained source reads and native VL-64 normalization are **not executed during this prediction**.
5. After predictions have already been committed, the original native VL-64 operator executes on the same new token-embedding mixture. This post-generation reference checks **every requested output coordinate** and records numerical error.
6. A changed mixture creates a different causal position and a previously unobserved output; no per-mixture result table is retained.

E-CPU stores exactly **three reusable full-length source-derived coefficient objects** (two token row vectors, one normalization gain vector) as byte pages and manifests. Neither the restored Ground nor its saved source-witness records need to contain the pretrained checkpoint in full.

### Recovery

An externally pinned recovery reconstructs both the original source-qualified range observations and the E-CPU coefficient objects. It compares each retained object byte-for-byte against the genuine source-observed embedding/norm tensor bytes, recomputes the independent quadratic coefficients from source-derived dot products, verifies the original observation-bound coefficient certificate and replays each recorded response-state position and its generated output digest. Only then can a new, previously unseen mixture be formed without original network access or repeated native normalization.

This is stronger than believing a checksum emitted by the same party that produced the alleged result. The supplied expected checkpoint head remains an external integrity anchor; it is **not** a remote identity authentication system.

## 4. First measured control

[Successful synthetic-TCP experiment, GitHub Actions run 44664713529](https://github.com/Kier73/VL-I/actions/runs/44664713529): 10 adversarial tests passed and a real local HTTP Range receiver served fabricated BF16 token embeddings and normalization weights.

| Control measurement | Observed |
| --- | ---: |
| Synthetic token-row width | 64 |
| Original-style TCP range requests | 5 |
| Native VL-64 observations used to infer/challenge the law | 5 |
| Rejected constant candidate, maximum error | 0.37552 |
| Rejected affine candidate, maximum error | 0.50406 |
| Admitted quadratic candidate, maximum error | 4.44 × 10⁻¹⁶ |
| Unseen token-mixture intentions | 4 |
| Requested output coordinates | 8 per intention |
| Maximum held-out error against later native reference | 6.66 × 10⁻¹⁶ |
| New source requests/native operation calls during new predictions | **0 / 0** |
| Source-free recovery generated another unseen mixture | **Yes** |

In that control, sparse generation took roughly **0.20–0.27 ms** for eight coordinates, with the subsequent full native reference taking roughly **0.68–0.85 ms**. Full 64-coordinate generation took approximately **0.48 ms**, compared with **0.45 ms** native—effectively a tie/slight slowdown on this tiny local workload. These single-run numbers are not representative full-model acceleration evidence.

This fixture is explicitly labeled synthetic. Its model-like paths and revision strings **do not** make its numeric tensors original trained Qwen weights.

## 5. Original full-size pretrained experiment

The same code runs with `--mode live` against the **real original Qwen/Qwen3.8-27B full checkpoint**, pinned to the existing source revision, index digest and original trained token embeddings.

The selected token IDs 63204 and 628 originate in VL-I's previously evidenced tokenizer output for the original paragraph. The input to the tested normalization is a **new affine mixture of their real source-qualified 5,120-element embeddings**, *not* a paragraph prefill hidden state produced by an actual preceding Transformer layer. The previous provenance is retained, rather than using a fabricated token mapping.

The live job performs **five original native normalization observations**, infers and independently challenges the quadratic energy model, generates **four held-out mixture outputs × eight coordinates** before running native references, independently restores the source and coefficient Grounds after process restart, then generates a fifth never-seen mixture. It also performs a complete 5,120-coordinate negative cost control.

The job records cold source/index/embedding acquisition and candidate fitting, both original pretrained token-row SHA-256 values, source HTTPS requests/bytes, sparse generation versus independent native normalization, held-out error, complete-output performance and E-CPU checkpoint size/cost. HTTPS unavailability is reported `UNRESOLVED`, never replaced with a synthetic “pass.”

## Live original Qwen3.8-27B verified result — November 19, 2026

The merged [SHAPE-008 original-source GitHub Actions run 44665097077](https://github.com/Kier73/VL-I/actions/runs/44665097077) succeeded against the full, pinned original pretrained **Qwen/Qwen3.8-27B** checkpoint. It read actual original 5,120-dimensional embedding rows for token IDs **63204 and 628** and the full original trained layer-0 normalization weights. These IDs were independently observed in VL-I's prior original paragraph-tokenizer experiment.

The receiver admitted **7 exact original HTTPS range requests / 76,272 source bytes**. The original checkpoint was not downsized. Its original-model index identity, token-row source SHA-256 and native operator were qualified before any learned output could be admitted.

**Observation-driven law inference:** five native original VL-64 normalization runs were executed on the actual two-token affine input span: three to infer polynomial coefficients from output observations, two to challenge model degree. Maximum challenge errors were **1.69914 (constant)**, **2.36324 (affine)** and **3.55 × 10⁻¹⁵ (quadratic)**. Only the bounded quadratic hypothesis was admitted, with the fitted coefficients independently checked against the mathematical square-energy identity of the source token span.

Then, four *never-before-observed* mixtures produced eight output coordinates each, **before** the original pretrained operator was called for a reference check.

| Unseen mixture (t) | Generative sparse Ground | Original native full-normalization reference | Maximum held-out absolute error |
| --- | ---: | ---: | ---: |
| 0.13 | **0.466 ms** | 18.445 ms | 2.22 × 10⁻¹⁶ |
| 0.37 | **0.381 ms** | 6.798 ms | 8.88 × 10⁻¹⁶ |
| 0.67 | **0.357 ms** | 6.677 ms | 8.88 × 10⁻¹⁶ |
| 0.91 | **0.346 ms** | 6.664 ms | 4.44 × 10⁻¹⁶ |

Every new output was predicted with **zero original HTTPS reads and zero native normalization evaluations during its formation**; the independent native reference executed **after** each output had been committed. The maximum held-out error was **8.88 × 10⁻¹⁶**, and each resulting coordinate occupied a separately causal, original-source/observer/intention-bound state position.

The generated Ground retained three E-CPU original-derived complete coefficient objects: both pretrained token rows and the true layer-0 trained normalization gains, totaling **122,880 binary64 bytes** (excluding content-addressed page geometry). This stores a function's source basis, not an input-to-output lookup table.

**Source-free recovery:** a fully pinned receiver restart independently revalidated the retained original trained token/gain bytes, the fitted law and every previous generated output. Recovering the **~471 KB** serialized checkpoint took **23.04 ms**. Without a source connection and without native original arithmetic during prediction, the recovered Ground formed a **fifth unseen mixture (t = 0.83)**, with independent original native reference error **4.44 × 10⁻¹⁶**.

**Performance accounting and negative control:** original source acquisition and inference preparation together took **2,609.7 ms** in the recorded run, including **6.61 ms** of original native normalization during training. Warm sparse generation can be much quicker than producing a whole original native output, but this advantage is conditional on amortizing those cold costs and asking for only a small observer projection. Requesting all **5,120 coordinates** through the current Ground took **9.298 ms**, **slower** than the **7.019 ms** original full-native reference. These are single-run, GitHub-hosted Python measurements, not a general or full-model speedup claim.

The precise scope is **a quadratic normalization law fitted from original native observations of the affine span of two real pretrained token embeddings**. This is no longer a synthetic hidden-vector input, but also **not** an actual Transformer-attention-conditioned hidden state, complete prefill, layer execution or language generation. The law's polynomial family was **chosen as a bounded hypothesis set**, not discovered with an open-ended search of arbitrary computation.

[Machine-readable original-Qwen evidence](../experiments/token_conditioned_law/results/2026-11-19-original-qwen27b-token-span.json) records source hashes, token identities, learned coefficients, candidate counterexamples, verified unseen outputs, wall time, memory and source-free recovery without copying original pretrained weights into the research report.

## 6. What SHAPE-008 does and does not establish

**If the live original-model verification succeeds, it establishes:** an output-forming Ground with coefficients fitted from native observations, original pretrained token-derived input state, bounded source-based polynomial semantics, unseen output generation, and independent post-prediction numerical verification.

**It does not yet establish:** autonomous discovery of an unlimited class of neural mathematics; inferred whole-network semantics from a few source observations; the exact real hidden state after token contextual attention; full Qwen 27B decoding or next-token parity; multi-host authenticated operation; a universal speedup on full-dimensional outputs.

The next research leap is to widen this from a **two-token affine span** toward *actual attention/MLP output manifolds* and to enlist IFS/VGI active probe selection and VLD's proof-producing compiler to synthesize richer candidate operator laws, with counterexamples, authority and observer scopes preserved at every step. E-CPU HFG-003 complete-layer handoff can participate **only after a real complete source-qualified layer** has been computed and independently verified.
