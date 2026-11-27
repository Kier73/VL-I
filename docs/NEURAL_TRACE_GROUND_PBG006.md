# PBG-006 — A neural causal observation constructs a predictive Ground

**Experiment:** `vli.neural_trace_ground` composes [PBG-005's source-query Hankel learner](SOURCE_DERIVED_GROUND_PBG005.md), [SHAPE-009's source-qualified native causal capture](SHAPE_009_CAUSAL_HIDDEN_MANIFOLD.md), [SHAPE-010's basis-expansion insight](SHAPE_010_CAUSAL_BASIS_EXPANSION.md), and the existing E-CPU SAB-001 + VGI receiver-attested observation route.

## Implemented question

Can the output of **actual trained numerical causal transitions**, observed through an intention-selected projection, supply the source facts from which VL-I constructs a new executable predictive transition algebra without inspecting the trained model's hidden transition matrices?

The answer tested here is a finite, explicit form: one or more resettable original-topology Qwen Gated DeltaNet layer-chain experiments produce genuine F64 causal output receipts. An observer converts those original outputs to a small event language. A uniform mixture of the actually witnessed finite event tapes defines an empirical stochastic process. **PBG-005 learns a minimal initial-source-relative rational predictive realization of that process solely from qualified sequence-probability queries.**

The mathematical target is the *observed finite tape generator*, with an absorbing end event, not unobserved behavior of the full neural network. This distinction matters: the source is a bounded collection of factual trained computations, and the resulting law is exact only under its declared empirical sampling and termination conventions.

## Cross-repository reuse

| Source | Mechanism reused | Why it matters |
|---|---|---|
| VL-64 `experiments.model_fabric.complete_native_layers` | `QwenQualifiedLayer` and original-format trained GDN + SwiGLU execution | Native numerical work and qualified tensor ranges |
| VL-I QUALIFIED-006 `QwenCausalGDNChain` | Complete layer-0 output passed **exactly** into layer-1; F64 digest chain and atomic token transition | Preserves actual causal order, source identity and original-topology transition receipts |
| VL-I SHAPE-009 `_qualified_capture` | Reconstruct chain head, source/model/revision, original token label, complete trained layer receipt and both F64 handoffs | Refuses forged or stale numerical observations before semantic interpretation |
| VL-I SHAPE-010 | Monotone source-witnessed basis expansion, parent Ground identity retention | The new neural observation can change a Ground without retroactively changing its earlier scope |
| VL-I PBG-005 | `QueryAtlas`, `HankelLearner`, paired reachable-span all-word verifier | Derives `K_a` from observation, not from hidden source matrices |
| E-CPU + VGI via PBG-005 | Receiver-owned two-host source-query routes and strict independently finalized VGI observation quorum | Evidence is qualified before an exact source reading enters Hankel learning |
| In-Formation-Space RE-001 | Exact-rational response equivalence versus hidden generator identity | Prevents confusing a minimal observable law with a unique internal neural architecture |

## Causal capture and observer-native reading

For each declared input sequence of exactly qualified F64 vectors and their associated token-ID **labels**, `QwenCausalGDNChain.step(..., capture_intermediates=True)` executes actual trained layer arithmetic and emits a source-bound report. `_qualified_capture` rechecks `previous`, `chain_head`, layer indices, interlayer F64 identity and complete layer widths. The second-layer F64 vector `h_t` supplies a declared categorical observer:

\[
O_{j,\theta}(h_t)=\begin{cases}
1,&h_t[j]\ge\theta\\
0,&h_t[j]<\theta.
\end{cases}
\]

Only validated native report bytes may enter `CausalTape.labels`. The source identity retains the model, pinned revision, native index-Ground digest, causal genesis, original input F64 digests, report hashes, observer coordinate and exact threshold representation. It is preserved separately from the digest of the induced abstract probability law.

A new observer can apply `reframe_native_tape(tape, new_observer)` to the **already captured and revalidated** F64 output. This changes the observer-relative Ground and source evidence digest with **zero new trained-layer transitions**. It does **not** invent a new source position or claim another original neural continuation.

Caller-declared input hidden vectors and token IDs in the fixture are not assumed to be a verified original tokenizer/embedding correspondence. An original pretrained execution must supply that separate qualification.

## Exact rational law induced by qualified neural traces

For `m` independently admitted finite traces `\tau_i = (o_{i,0},...,o_{i,L_i-1})`, choose one trace uniformly at the start. At each position the observer may `act:step` (advance with terminal absorption) or `sense:readout:v` for `v` in `0,1,2`, where `2` is the absorbing end-of-tape reading. The source's *empirical* observable weight for an operation sequence `w` is

\[
P_{\mathrm{tapes}}(w)=\frac{1}{m}\sum_{i=1}^{m}
\mathbf 1\{w\text{ is consistent with }\tau_i\text{ under the declared step/read semantics}\}.
\]

This is an **exact rational**, normalized, finite probabilistic source. Reading does not change its material state, while `step` advances a source-specific position. The learner sees only `SourceObservation(word, probability, source_digest, receipt_digest)` from this finite stochastic trace observer. It is not given individual native vectors, hidden trained matrices, or the finite source's transition arrays.

PBG-005 constructs

\[
H_{p,s}=P_{\mathrm{tapes}}(ps),\quad
H^{(a)}_{p,s}=P_{\mathrm{tapes}}(pas),\qquad
K_a=H^{-1}H^{(a)}.
\]

When the independent source verifier supplies a distinguishing word, that source-observed word is added as a prefix/suffix basis witness. The revised learned `K_a` replaces **only the new Ground revision**, preserving the old Ground's original scope.

### Source-relative all-future mathematical verification

The independent verifier can materialize a finite `ControlledSource` **from the already admitted event tapes** and check the learned law with a paired reachable rational row-space closure. This is stronger than finite random testing: for the declared tape generator, it proves equality of all finite operation-word probabilities from the declared initial distribution. It need not prove equality from every hidden tape position *in isolation*; source-global commutation would impose a stronger, unnecessary condition. It certifies neither the original model's unobserved word probabilities nor its next-token distribution.

## Actual E-CPU and VGI admission path

`ReceiverQualifiedNeuralReader` composes the pre-existing `NativeSourceReader` with the native neural witness. PBG-005's `read_source_batch` binds each exact batch to source, nonce, question digest, E-CPU receiver-owned handler and path, independent local oracle and two strict VGI observers. The adapter verifies each returned exact value against its native F64-derived empirical tape semantics, then issues a receipt digest that incorporates **both** the native E-CPU/VGI receipt and the particular native neural witness fingerprint.

Two E-CPU endpoints are distinct local receiver-provisioned handlers, not independently controlled hardware sensors. The trust scope for live pretrained source bytes remains the authenticated range host and its qualified tensor/capture contract. A SHA-256 digest identifies content; a portable remote source signature or pretrained-model authority cannot be inferred from it.

## Execution and adversarial tests

- Native original-topology two-GDN-layer Qwen execution with exact causal F64 handoffs, checked receiver-native report chain and independent numerical re-execution on a fresh pinned fixture.
- Multiple different causal input histories yield an original-trained **synthetic F32 fixture** trace ensemble; all learned probabilities checked against an independent trace-walk oracle.
- Counterexample-driven source-observed Hankel rank expansion; paired-space all-word proof plus held-out finite prefix checks.
- New observer projection forms a distinct source law without rerunning already completed trained transitions.
- Rehashed tampered full-F64 receiver bytes, false causal genesis and altered categorical readout are refused.
- Native E-CPU SAB-001 two-host operation routes + VGI strict matching quorum qualify source-query answers; conflicting complete source readings refuse admission.
- Existing SHAPE-010, QUALIFIED-006 and PBG-005 tests run alongside the new tests in cross-platform CI.

Run the dedicated [workflow](../.github/workflows/pbg006-native-trace.yml) (Linux/Windows, Python 3.11/3.13). The fixture tests do not fetch full Qwen 27B pretrained weights. The native source library is the same pinned VL-64 implementation used by SHAPE-010 and pre-existing numerical fixtures.

## Next research questions

1. Extend observed tape ensembles into **source-qualified renewal and conditional reset** operators, so intervention and observer history jointly select a representative causal state rather than terminate at the end of each tape.
2. Use SHAPE-010's actual neural F64 structural-memory pages to acquire only the observer-required coordinates from a pinned original pretrained model, avoiding a requirement to download all model weights.
3. Add a qualified active counterexample acquisition policy. When current neural traces cannot distinguish competing laws, request only the source state/action and numerical channel that split them, rather than an arbitrary whole-model inference.
4. Integrate E-CPU's durable authorization and crash-fenced actuation separately from read-only native probes; never equate a numerical receipt with a physical intervention.
