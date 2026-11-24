# PBG-007 — Intention-directed source observation and neural causal acquisition

**Research target.** Move from compiling a predictive Ground from already supplied neural trajectories (PBG-006) to the observer choosing **what source fact to acquire next**, then constructing a new Ground revision from the evidence that actually arrives. Two explicit observation modes are implemented: an exact algebraic discriminator of rival predictive laws, and a source-execution selector that runs only one qualified native Qwen-shaped causal input trajectory. A third mode constructs an intention-specific observer coordinate from existing witnessed F64 readouts without invoking trained numerical execution.

**Code:** [`vli.active_neural_acquisition`](../src/vli/active_neural_acquisition.py). **Tests:** [`exact-rational`](../tests/test_active_neural_acquisition_math.py), [`native two-layer Qwen-shaped fixture`](../tests/test_active_neural_acquisition_native.py). **CI:** [`PBG-007`](../.github/workflows/pbg007-active-neural.yml).

## Reused source authorities

| Existing construction | Reused responsibility |
| --- | --- |
| [PBG-005](SOURCE_DERIVED_GROUND_PBG005.md) | `HankelLearner`, `QueryAtlas`, `LearnedGround`; exact source word weights and independent finite-source audit |
| [PBG-006](NEURAL_TRACE_GROUND_PBG006.md) | `CausalTape`, `capture_native_tape`, native F64 output receipt checks, `WitnessedNeuralTraceEnsemble`, `ReceiverQualifiedNeuralReader` |
| [SHAPE-009/010](SHAPE_010_CAUSAL_BASIS_EXPANSION.md) | `_qualified_capture`, causal head and native interlayer numerical fidelity; witness before extending the structural representation |
| [VL-64 original operation fixture](https://github.com/Kier73/VL-64/blob/main/experiments/model_fabric/complete_native_layers.py) | Trained, original-topology GDN numerical operations; only selected new execution is performed |
| [E-CPU SAB-001](https://github.com/Kier73/E-CPU/blob/main/mailcpu/substrate_bootstrap.py) and [VGI evidence](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/main/virtual_ground_interpreter/docs/EVIDENCE_AND_RECONSTRUCTION.md) | Existing receiver-owned operation admission and strict two-source VGI finalization; no duplicated protocol |
| [VGI Relational Atlas](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/main/virtual_ground_interpreter/docs/RELATIONAL_ATLAS.md), [IFS Response Equivalence](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/response-equivalence/README.md) | Observer-relative distinguishability, horizon-dependent information and avoiding unsupported generator identity |

## 1. Exact active source questions, without source-hidden matrices

Each competing *candidate*, not yet asserted to be true, has the PBG-005 finite weighted-automaton form

\[
\hat P_i(w)=\alpha_i K_{i,w}\,\beta_i, \qquad K_{i,w}=K_{i,a_1}\cdots K_{i,a_m}.
\]

Construct joint reachable rows by BFS using only candidate matrices:

\[
z_\epsilon=(\alpha_i,\alpha_j),\qquad
z_{wa}=z_w\;\operatorname{diag}(K_{i,a},K_{j,a}).
\]

Accept a row in the exact rational span basis if linearly independent. If an accepted row has nonzero difference

\[
D(w)=\hat P_i(w)-\hat P_j(w) \ne 0,
\]

it is a source-free, shortest-depth distinguishing *question*. The BFS contains at most \(r_i+r_j\) independent rows. If all independent rows have zero difference after the reachable span closes, the **two candidate automata** agree on *every* finite word. This is candidate equivalence, not proof that either matches the source. A depth cap that interrupts closure is reported as unresolved, never equivalent.

For 2–8 declared hypotheses with positive rational proposal weights \(q_i\), score each pair's shortest previously unqueried distinguisher by

\[
\mathrm{score}(w;i,j)=\frac{q_iq_j\;|D_{ij}(w)|}
{(\sum_k q_k)^2(1+\sum_{a\in w}c_a)}.
\]

This is a *cost-normalized forecast difference*, **not** asserted to equal Bayesian expected information gain or actual source likelihood. `SourceProbePlan` pins the source, complete QueryAtlas evidence state, candidate identities, observer operation costs, selected word and nonce. `ActiveTraceObserver.execute` re-plans before requesting the word. It admits only `SourceObservation` through the source's configured reader. The PBG-006 native reader requires its existing E-CPU/VGI strict quorum and independently matches the returned value against the native-witnessed ensemble.

If the new observation contradicts the current `HankelLearner.ground`, the actual source-validated word is submitted to `HankelLearner.counterexample` and drives a newly learned Ground revision. The parent law remains immutable. No material actuation and no new trained neural transition occur during these *trace weight queries*. A pre-existing PBG-006 native tape is the experimental source.

## 2. Active native causal acquisition (new numerical work)

A receiver declares a finite 1–32 catalogue of candidate native input *paths* (each at most 12 positions), with exact token-ID **labels** and finite, bounded F64 input vectors. They are not claimed to be original tokenizer embeddings. Before any new numerical execution, an intention-relative source explorer selects one previously unwitnessed path by the explicit heuristic

\[
N(c)=\min_{t\in\mathrm{ObservedTapes}}
\sum_{j=0}^{\min(|c|,|t|)-1}
\frac{\|c_j-t_j\|_2^2}{j+1},\qquad
c^*=\arg\max_{c\in\mathrm{Allowed}\setminus\mathrm{Observed}}
\frac{N(c)}{|c|}.
\]

`NativeRunPlan` binds the exact parent native witness digest, all proposed input path digests, an independent nonce, chosen path, cost and novelty. The plan is recomputed immediately before execution. A fresh receiver-authorized `QwenCausalGDNChain` must match the existing model, revision, index and causal genesis. `capture_native_tape` invokes **only the chosen path** and revalidates every trained causal receipt through SHAPE-009's qualified capture. Existing retained tapes are reused unchanged. The new `WitnessedNeuralTraceEnsemble` carries a distinct native source revision; PBG-005 can build a new predictive Ground over its changed observed trace law.

The score uses **known input coordinates only**. It does *not* predict which unseen neural output will be obtained. Source novelty is a tractable experiment-design heuristic, not a proven measure of expected information; causal neural behavior remains source-qualified only after execution. Recreating the same old `NativeRunPlan` against an old parent could cause another numerical evaluation; the caller must persist an externally authoritative completed request/fencing record to obtain durable idempotency.

## 3. Intention selects the observation coordinate

For two nonempty groups of already executed trajectories, a goal declares which source tapes should be distinguishable. Every existing verified F64 second-layer output can serve as a possible observer coordinate. At a position \(p\) and coordinate \(i\), evaluate thresholds drawn only from observed F64 samples:

\[
\mathrm{sep}(p,i,\tau)=\left|
\Pr[h_{p,i}\ge\tau\mid\mathrm{target}]
-\Pr[h_{p,i}\ge\tau\mid\mathrm{comparison}]
\right|.
\]

Both empirical probabilities are exact fractions of *observed* tape counts. Choose the most discriminating coordinate/threshold per declared read cost; tie-break by position, coordinate and threshold. `apply_observer_attention` requalifies the original SHAPE native receipts, reprojects them using PBG-006 `reframe_native_tape`, and constructs a separate source identity and predictive Ground for the new observer. No pretrained numerical transition is repeated. If no observed native coordinate separates the intention, the procedure refuses rather than hallucinating one.

## 4. Native fixture validation and falsifiers

The mathematical suite checks independently learned competing finite source laws, shortest witness discovery, strict source receipt binding, zero-neural-execution trace queries, Counterexample → Hankel rank growth, stale plan rejection, altered candidate digests, missing observations, and native input-selector semantics. The native integration suite uses the existing VL-64 F32 Qwen-shaped two-GDN-layer fixture and full receiver-local E-CPU/VGI source reader to test:

1. A previously unseen native run is selected without any trained execution; exactly its length's number of trained causal transitions are subsequently executed and fully witnessed.
2. Original source tapes and SHAPE receipts remain unchanged; the new empirical trace law receives a new source identity and an independently equivalent learned PBG-005 Ground.
3. A changed observer coordinate is chosen from *actually witnessed native numerical outputs*, producing a new intention-defined source without trained reexecution.
4. A source-free shortest discriminator between old/new neural-trace models is sent through the existing E-CPU SAB native host and VGI two-source strict quorum. Only the source-qualified returned probability may refine the new Ground.
5. Modified plans, stale source genesis and incorrect native-operation identities are refused before admitting a result.

### Evidence scope

- **Exact theorem**: finite-dimensional *rational learned candidate automata* have a finite paired reachable span; the code either finds a shortest distinguishing word or proves candidate equivalence within the complete declared alphabet. This theorem does not prove equivalence to a pretrained neural model.
- **Native executable evidence**: on CI, synthetic-weight Qwen-shaped complete GDN layer transitions, source-witnessed via the existing exact-F64 two-layer causal chain and native receiver verification. A finite ensemble of executed traces, sampled uniformly, gives an exact discrete source law. Its probabilistic interpretation is a property of that declared empirical ensemble.
- **New neural acquisition**: executes only the selected *declared* input path under the existing source-owned fixture; input novelty does not establish optimal future learning or exact next-token generation.
- **Operational authority**: two receiver-local E-CPU hosts and VGI strict quorum; independently administered remote mTLS and an original pretrained tokenizer/decoder are separate follow-up qualifications.

## Next research frontiers

Connect `select_native_run` to E-CPU authenticated remote host leases/dispatch, preserve a durable input-probe journal across requester restarts, and compare inferred observer-specific predictive distinctions against actual independently verified pretrained Qwen continuations with pinned original tokenizer/embedding inputs. Replace input novelty with calibrated, source-validated active-learning scores only after an independent held-out predictive model exists.
