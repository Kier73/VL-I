# SHAPE-007 — Generative trained-operator Ground for unseen intentions

**Objective:** lift one original pretrained operator's mathematical relation into a source-qualified, observer-dependent Ground that generates numerical results for **previously unobserved input intentions**, rather than returning a previously cached output. This is a genuine *generative family* for a stated numerical input domain, **not** a claim of newly invented neural mathematics, general model interpretation or language intelligence.

**Code:** [generative_operator_ground.py](../src/vli/generative_operator_ground.py)  
**Held-out, source-free experiment:** [generative_operator_demo.py](../examples/generative_operator_demo.py)  
**Adversarial tests:** [test_generative_operator_ground.py](../tests/test_generative_operator_ground.py)  
**CI:** [vli-shape007.yml](../.github/workflows/vli-shape007.yml)

## The computational coordinate is a position

The original trained model has a 5,120-dimensional normalization relation for the first Qwen3.8-27B layer. A conventional native evaluation computes a complete hidden vector and its full mean-squared energy before producing a normalized vector. The observer may only need eight selected output coordinates, or perhaps a structural projection rather than a complete 5,120-element array.

The new Ground binds **original model source identity, native implementation identity, learned/admitted mathematical law, observer intention, requested output coordinates, and predecessor causal head**. It computes a fixed-width response-state position *before* interpreting any new result. Its position is not the numerical index of a hidden state, and a result value is not used to define its address.

The observer may ask the same original model mathematical question with different frequency, phase, amplitude, waveform, output coordinates or previous state. Unlike a memoized result, the Ground forms those new results without running the entire original normalization operator.

### Mathematical realization law

The **existing VL-I INTERACT-002 hidden-state input interpreter** uses an analytic input family:

\[
x_j = A \sin((j+1)\omega+\phi)
\quad\text{or}\quad
x_j = A \cos((j+1)\omega+\phi),\quad 0\leq j<n.
\]

The **original VL-64 Qwen trained operator** is the native zero-centered normalization law:

\[
y_j = \frac{x_j(1+w_j)}
  {\sqrt{(1/n)\sum_{k=0}^{n-1}x_k^2+10^{-6}}}.
\]

The relevant sufficient statistic is the entire hidden vector's mean-squared energy. For the declared sinusoidal family, a known finite trigonometric sum identity gives:

\[
\frac{1}{n}\sum_{j=0}^{n-1}x_j^2 =
\frac{A^2}{2}\left[
1 \mp
\frac{\sin(n\omega)}{n\sin\omega}
\cos((n+1)\omega+2\phi)
\right],
\]

with minus for sine and plus for cosine. The observer can therefore derive the normalization energy with a **bounded constant number of scalar operations** rather than enumerating all \(n\) input squares. It then generates each *requested* coordinate \(y_j\) from an **original source-qualified trained coefficient \(1+w_j\)** held in E-CPU's immutable ComputationalObjectStore. E-CPU's existing \`read_range\` checks only relevant coefficient pages.

If the observer requests \(k\) coordinates rather than the whole array, numerical formation is \(O(k)\) coordinate work plus \(O(1)\) input energy work. Reading and checking original coefficient pages still incurs storage and interpreter work. A full requested vector requires \(O(n)\) output work; **neither the model's dimensions nor memory/verification costs disappear**.

This law is known mathematics and is only applicable to the admitted **analytic sin/cos input family**. It is *not* an independently learned formula for arbitrary token embeddings, arbitrary incoming hidden states or the complete original attention network. The exact program source and its operator/input interpreter fingerprint bind the Ground.

## Reused source authorities

- **[VL-I SHAPE-005](SHAPE_005_PRETRAINED_OBSERVER_MEMORY.md):** the original pretrained model, exact revision/index SHA-256/index Ground, QualifiedShardNavigator source-range witnesses and E-CPU byte object memory. No trained weights are guessed from the header. Remote source authenticity remains the original qualified host's responsibility.
- **[VL-I SHAPE-006](SHAPE_006_VERIFIED_TRAINED_RESULT_GROUND.md):** the existing intention input semantics, native VL-64 trained operator and original E-CPU result-state evidence. Unlike SHAPE-006, this Ground does *not* store one output for each new input.
- **[VL-64 original centered_rms_norm](https://github.com/Kier73/VL-64/blob/main/experiments/model_fabric/trained_subgraphs.py):** used as the independent native numerical witness on training and held-out cases.
- **[E-CPU ComputationalObjectStore](https://github.com/Kier73/E-CPU/blob/main/mailcpu/computational_objects.py)** with **StructuralMemoryPageBacking:** stores **generalizable trained per-coordinate gains**, not a list of expected answers. A source-free observer can reconstruct the coefficients against the true original retained weight bytes and generate a *new* output.
- **[VLD intention-to-realization and E-CPU model-orbit research](https://github.com/Kier73/E-CPU/blob/main/mailcpu/model_orbit_handoff.py):** inform the retained state-position and causal-proof contracts. HFG-003 specifically requires a **complete, independently checked layer native receipt**; the present partial RMSNorm receipt is **not falsely published as a complete layer handoff**.

## Admission: reject inadequate laws before promoting

The interpreter considers **three supplied candidate normalizers**:

1. Constant unit energy;
2. Equidistributed sinusoidal energy \(\approx A^2/2\);
3. Finite-orbit energy identity with exact finite-length phase/frequency dependence.

For each of **four distinct training input intentions** with different waveforms and frequencies, the original VL-64 trained operator computes its complete \(n\)-element reference. Each proposed law is compared against **all \(n\) outputs** for its maximum absolute error. Only an unambiguous candidate meeting the stipulated \(3\times10^{-9}\) error tolerance is admitted; the source-specific scores, weight digest, original source-range snapshot, interpreter and candidate fingerprints are retained.

This is **selection from provided mathematical candidates using witnessed evidence**, not autonomous symbolic discovery or a kernel-level proof that an unknown expression is universally equivalent. The mathematical finite-sum relation itself provides the derivation within its stated domain. An input outside the declared stable parameter domain is refused rather than silently extrapolated.

### Separate held-out verification

After admission, an unseen intention first constructs a state position and **generates requested values without invoking the original native operator**. The reference is then queried independently using actual original VL-64 math. This temporal separation ensures reference results are not used to generate the predictions being tested.

The test suite checks four genuinely unseen frequency/amplitude/phase combinations, different observer indices, both waveforms, full-vector generation, a deliberately wrong approximate law, source epoch changes and stale causal heads.

An independently recovered Ground must do more than match a manifest hash: it reads the **original source-weight bytes** from the restored E-CPU memory, rebuilds their gain coefficients, verifies the saved E-CPU coefficient object bytes against that source, and independently **recomputes the output of each retained causal event** from its stored original intention and position. It then generates a *never-requested-before* new result with no original HTTPS host or original native trained arithmetic.

## Initial control — synthetic 64-dimensional TCP

[The first passing SHAPE-007 CI run 44659786392](https://github.com/Kier73/VL-I/actions/runs/44659786392) exercised **11 adversarial tests** using a **synthetic** BF16 fixture carried over an actual local TCP byte-range server. It is not evidence of pretrained Qwen weight equivalence.

| Metric | Measured result |
| --- | ---: |
| Original source requests during bootstrap | 3 |
| Original bytes during bootstrap | 245 |
| Original native operator witness evaluations | 4 |
| Falsified candidate maximum error: constant unit | 2.61384 |
| Falsified candidate maximum error: equidistributed | 0.39423 |
| Admitted finite-orbit candidate maximum training error | \(4.44\times 10^{-16}\) |
| Four unseen held-out intentions × eight requested coordinates | **32 generated values** |
| Held-out max absolute error, native verification after generation | **0.0** in this synthetic run |
| Native operator executions during generated output | **0** |
| Additional source requests during generated output | **0** |
| Source-free recovery generated a fifth unseen intention | **Yes** |

**Negative cost control:** in the 64-dimensional synthetic benchmark, one eight-coordinate generated observation took approximately **1.4–1.5 ms**, while the full original native reference took around **0.4–0.46 ms** including the locally retained range receiver. The full 64-coordinate generated observation was **2.56 ms**, versus **0.396 ms** for original native reference. The Ground was **slower in this tiny Python workload** because per-coordinate E-CPU page checking costs more than the simple original arithmetic. This negative result must be preserved, not disguised as a speedup.

The value demonstrated is new **unseen computational state generation** from original source coefficients and a qualified low-dimensional law, not lower wall time under every circumstance.

## Original full-size Qwen test

The same executable supports \`--mode live\`, using the original full-sized pinned Qwen3.8-27B checkpoint and its actual original **5,120-element layer-0 trained BF16 normalization tensor**. The real-source workflow executes on \`main\` or explicit manual dispatch. It tests:

- source-bound bootstrap through actual original pretrained HTTPS ranges;
- four **unseen intention** probes requesting only eight distinct coordinates each;
- independently executed whole-operator native references on exactly those unseen inputs **after** predictions are fixed;
- source-free checkpoint recovery followed by a *fifth previously unseen* generative observation, verified against original trained math;
- a full-width 5,120-coordinate output negative cost control, where per-page verification may be slower;
- actual source bytes, trusted source receipts, native reference work, E-CPU storage, memory footprint, held-out errors and wall time.

The original trained model is **not downsized** for that experiment. If live HTTPS is unavailable, results are marked \`UNRESOLVED\` and the synthetic fixture is never substituted as original pretrained proof.

## Original full-size live result — November 19, 2026

The [successful main-branch GitHub Actions run 44660339251](https://github.com/Kier73/VL-I/actions/runs/44660339251) **executed against original, full-scale pinned Qwen3.8-27B trained normalization weights**. It read **55,640 bytes in three original HTTPS range requests** and bound the actual 5,120-element layer-0 trained tensor to E-CPU's content-addressed memory. There was **no model downscaling or pretrained-weight approximation**.

The original native VL-64 operator evaluated four complete 5,120-element *training witnesses* to discriminate three supplied candidate laws. The constant-energy and equidistribution candidates were falsified with maximum deviations of **1.4770** and **0.3614** respectively. The finite-orbit law matched those training witnesses to approximately **4.44 × 10⁻¹⁶** maximum absolute error.

The admitted Ground then formed the outputs of four **previously unseen analytic input intentions**, with eight selected coordinates each, *before* the original native operator was consulted for independent verification.

| Held-out original-Qwen request | Generated sparse state | Native full-normalization reference | Max absolute error |
| --- | ---: | ---: | ---: |
| Unseen sine, low frequency | 2.896 ms | 16.494 ms | 0 |
| Unseen cosine, intermediate frequency | 2.749 ms | 7.188 ms | 4.44 × 10⁻¹⁶ |
| Unseen sine, higher frequency | 2.686 ms | 7.060 ms | 0 |
| Unseen cosine, high frequency | 2.663 ms | 7.164 ms | 0 |

**Source HTTPS requests during these new generated responses: zero. Original native pretrained-operator calls during generating the predictions: zero.** The independent original operator ran afterward as the held-out test oracle. All 32 generated output coordinates were accurate to the stipulated tolerance.

The source-bound generative Ground was also **restored without a connected source** in approximately **13.99 ms**, independently validating its original stored trained coefficients and previous state positions, and then generated a **fifth never-requested input** whose independently checked native error was zero.

**Essential negative control:** generating **all 5,120 output coordinates** using the current per-coordinate E-CPU page reads took **49.178 ms**, whereas the corresponding whole-vector original native reference took **7.481 ms**. For a full output, the current interpretation was **about 6.57× slower**. The finite-series energy computation is cheap; requesting a full result incurs proportional output and per-page verification overhead. This is not a universal operator speedup.

**Cost context:** the original source+Ground preparation took **277.67 ms** in that run, including approximately **30.50 ms** of original native training-witness execution. The reported sparse per-request times are **warm qualified-observer** samples, not cold total amortized cost; single-run timings may vary by host. Including bootstrap, one sparse request is not cheaper than a fresh full native evaluation. The benefit becomes material when a source-qualified Ground is reused for many sparse intentions.

The strongest warranted conclusion is **unseen-input sparse trained operator result generation over the original full-scale source with independently verified numerical correctness**, together with a measured warm observer-specific cost reduction and a measured full-output slowdown. This is neither autonomous mathematical-law invention nor full trained decoder/token prediction.

[Raw, machine-readable original-Qwen held-out evidence](../experiments/generative_operator_ground/results/2026-11-19-original-qwen27b-heldout-generative-law.json) records model source and native operator identities, candidate errors, predicted/verified timings, new causal positions, source-free recovery and negative controls. It embeds **no pretrained weight payloads**.

## Frontier

SHAPE-007 is a bounded **function-shaped Ground** rather than an exact result cache. It establishes that, when the observer's permitted input family has independently qualified sufficient statistics, the interpreter can form new computational states using only the work needed for the observer's requested output.

The next frontier needs genuinely more powerful learned relations:

- Discover candidate laws **from observations** using IFS/VGI active attention and VLD proof-producing rewriting, rather than picking only among supplied known mathematical identities.
- Move from analytic input recipes to **actual token embeddings and prior trained layer outputs**, preserving their source/causal Ground, operator and oracle contracts.
- For full original trained Qwen attention/gated operators, find **observer-relative low-rank/quotient/sparse relations** whose correctness can be independently verified on unseen hidden states, not merely cached responses.
- Integrate E-CPU HFG-003 **complete native layer** handoff once a true whole layer is independently realized; preserve the distinction between a partial source-qualified mathematical relation and full layer execution.
- Measure true end-to-end gains at model scale, including source acquisition, repeated proofs, E-CPU page checks, machine scheduling, transport and output rendering.

**Research ambition:** progressively lift the interpreter itself into generative mathematical and relational Grounds so observer intention can navigate to and form verified response states without repeating the entire inherited trained computational path.
