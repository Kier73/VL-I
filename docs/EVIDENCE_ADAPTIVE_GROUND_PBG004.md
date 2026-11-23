# PBG-004 — Evidence-adaptive intention and source-law revision

**Objective:** allow the Virtual Layer to revise its predictive Ground when independently qualified observations conflict with its intended model, without confusing stochastic surprise with mathematical impossibility, and without re-executing a material action merely to replay an observation history.

**Implementation:** [`vli.evidence_adaptive_ground`](../src/vli/evidence_adaptive_ground.py), [exact-source tests](../tests/test_evidence_adaptive_ground.py), [native E-CPU/VGI integration](../tests/test_evidence_adaptive_fabric.py), and [CI](../.github/workflows/pbg004-evidence-adaptive.yml).

## Reused cross-repository mechanisms

| Existing mechanism | Exact role in PBG-004 |
| --- | --- |
| [PBG-001](OBSERVER_BACKWARD_AND_PREDICTIVE_GROUND.md) | Exact rational source-derived future-observation closure and persistent reduced state |
| [PBG-002](ADAPTIVE_INTENTION_PBG002.md) | Compiled controlled sources satisfying `M B = B K`, exact goal-conditioned planner |
| [PBG-003](OBSERVED_INTENTION_PBG003.md) | Native receiver-owned E-CPU SAB-001 execution, strict two-source VGI finalization, exact source/epoch-specific records |
| [TDL-005](AUTHENTICATED_REFINEMENT_TDL005.md) and [TDL-006](ROUTED_AUTHENTICATED_REFINEMENT_TDL006.md) | Prior mTLS and pinned-peer mechanisms for a future authenticated, remote-source version; **not** newly implemented here |
| [VGI evidence and reconstruction](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/main/virtual_ground_interpreter/docs/EVIDENCE_AND_RECONSTRUCTION.md) | Explicit authorized versus pending/conflicting evidence, with source identity and reconstruction |
| [IFS response equivalence](https://github.com/Kier73/In-Formation-Space/tree/main/experiments/response-equivalence) and [observation-equivalence bridge](https://github.com/Kier73/In-Formation-Space/tree/main/bridges/observation-equivalence) | Observer-relative predictive algebra and future distinctions |

The code does **not** introduce a second transport, reimplement the evidence ledger, execute a trained neural model, or require source-state matrices during online planning.

## 1. Mathematical state: intention-relative posterior over qualified Grounds

Given a fixed goal/observer contract, a finite catalogue of candidate source laws `L_i` is independently checked using PBG-002 `compile_intention_ground` / `verify_intention_ground`. Each provides a predictive Ground with basis `B_i`, reduced matrices `K_{i,a}` and `K_{i,s,y}`, and retained coordinate `z_i`.

The interpreter's state is

\[
\mathcal G_t=\bigl\{(d_i,z_{i,t},w_{i,t})\bigr\}_{i=1}^{m},
\quad \sum_iw_{i,t}=1, \quad m\le8,
\]

where `d_i` is each source/compiled Ground identity. Only the reduced coordinates and law masses, not the eight-state source beliefs, are needed for online inference. The original source matrices are used for construction/qualification of **new** candidates, not for acting on an established candidate.

A receiver-admitted **action** `a` uses

\[
z'_{i}=z_i K_{i,a}, \qquad w'_i=w_i.
\]

An authorized sensor reading `y`, with candidate conditional likelihood
\(p_i(y)=(z_i K_{i,s,y})_0\), uses

\[
\boxed{w'_i=\frac{w_i p_i(y)}{\sum_j w_jp_j(y)},\qquad
 z'_i=\frac{z_iK_{i,s,y}}{p_i(y)}\quad\text{when }p_i(y)>0.}
\]

All values are exact `fractions.Fraction` rationals. Positive-likelihood surprises simply change the relative masses. A zero-likelihood candidate receives posterior zero. **An observed failure with probability 1/4 does not refute a 3/4-success law.** One impossible event is a counterexample only relative to the declared complete finite source law and trusted observation contract.

The intended action policy remains a bounded exact Bellman search maximizing

\[
J(\pi)=\Pr(g\text{ at stop}\mid\pi)-\mathbb E[C],
\]

but now probabilities integrate over the qualified law posterior, not only one guessed model. After new evidence, the active law and chosen intervention can change without building or executing an entirely new neural inference path.

## 2. Evidence before belief: original PBG-003 operations are reused

```text
Current council revision + current PBG-003 contract IntentionSession
  → issue RevisionPlan(current epoch, PBG-003 ScheduledIntent)
  → E-CPU SAB-001 admitted host/path execution; receiver oracle
  → for sensing, VGI ChannelMessage + Encounter + frame + quorum + finalize
  → independently verify original PBG-003 E-CPU/VGI receipt
  → candidate-relative likelihood of *the same admitted outcome*
       ↳ positive: update candidate predictive coordinates and exact posterior
       ↳ zero for some: eliminate those candidates, preserve survivor Grounds
       ↳ zero for all: preserve qualified reading, refuse future planning
  → append causal event identity and increment revision epoch
```

`dispatch` runs PBG-003 against a **shadow** contract session and then feeds its verified result into the council; the live council state advances only on successful receipt verification. A failure after an external action may still leave the material world changed. It must be reconciled against the actual actuator's durable completion log, rather than re-running the action. The present receiver-local simulation retains memory-local nonce idempotency from PBG-003; it does not claim physical exactly-once actuation.

A new revision has a digest over its contract Ground, candidate/source identities, prior masses, retained coordinates, accepted evidence digest chain and version. A scheduled operation binds that epoch. After a revision, old proposals are refused; attempts to reuse an admitted operation nonce are refused. The epoch is a **receiver-side fencing token**, not a substitute for source-host durable fencing.

## 3. Source-law expansion when existing models fail

When all candidates give probability zero to independently admitted `y`, **the Ground is not silently coerced to admit `y`**. PBG-004 retains the actual PBG-003 receipt as `EXPANSION_REQUIRED` and refuses further planning.

A receiver can then supply a new `ControlledSource`, which is compiled and independently verified with the existing PBG-002 machinery. The new candidate must preserve the same goal and exact names/outcome labels/costs of admitted operations. Its reduced state is reconstructed by replaying **only the already accepted source-independent operation and outcome descriptions through the new reduced Ground**:

\[
 z_{\mathrm{new},0} \xrightarrow{\text{retained observations/actions}}
 z_{\mathrm{new},t}.
\]

No E-CPU actuator, model weight payload or physical source is re-executed for that catch-up. If the new model supports the pending observation, PBG-004 consumes that same receipt and resumes planning; otherwise `EXPANSION_STILL_REQUIRED` is retained.

This is **qualified candidate introduction**, not unconstrained discovery of an arbitrary new physical law from one measurement. Inferring the new law from data is a separate future experiment requiring a declared identification family and independent challenge corpus.

## 4. Direct witnessed fixture

Eight hidden source states encode material type, goal-success flag and redundant tag. The receiver's contract Ground uses a 3/4-accurate type-sensor law to admit either reading. Two alternative source-qualified model Grounds have *the same* operations, goal and target space:

- `reported-faithful`: type sensor perfectly reports the material type; prior mass 4/5.
- `sensor-inverted`: type sensor perfectly reverses the material type; prior mass 1/5.

A receiver-held software environment has actual type **0**, but both sensor handlers report type **1**. E-CPU executes the two exact measurement operations and VGI admits the agreeing reading. Both source hypotheses assigned probability 1/2 to this initial type reading, so their law posterior remains at 4/5 and 1/5. The active model therefore selects `apply-1`.

E-CPU's actuator changes the **local material fixture**; its acknowledged action leaves the law weights unchanged. Both post-action goal readers subsequently report **0**. Since the `reported-faithful` hypothesis had predicted goal success with probability 1, it assigns probability **zero** to this measurement. `sensor-inverted` predicted failure with probability 1. Its posterior becomes 1; without any source-state matrix execution, the Virtual Layer now selects `apply-0`. That second, separately admitted actuation makes the fixture's goal **1**, which a further VGI-finalized read confirms.

A separate experiment starts with **only** the faithful law. The same qualified goal-0 receipt is retained as `EXPANSION_REQUIRED`; introducing the source-qualified inverted law reconstructs its state from the prior history and consumes the **original**, pending result. The material revision count does not increment during hypothesis introduction.

Other tests demonstrate why positive-probability failures are only Bayesian evidence. With candidate sensor accuracies 3/4 and 1/4 at equal priors, one post-action goal failure yields posterior masses **1/4 and 3/4**, respectively; neither is eliminated.

## 5. Verification, persistence, and scope

**Source-only tests** verify the reduced Bayesian Bellman policy against an **independent eight-state joint-model planner** for horizons 0–2; exact posterior updates; zero-support rejection; hypothesis addition without re-execution; compact Ground snapshot reconstruction from prior event descriptions; cross-intention refusal; rehashed state forgeries; stale and duplicate epoch refusals.

**Native integration tests** drive the actual E-CPU and VGI APIs via the existing PBG-003 receiver-local fixture: success after model revision; all-candidate impossibility and recovery; conflicting source readings; tampered E-CPU/VGI receipts even if an outer hash is recomputed; and revoked host authority. They also verify causal revision after an already executed material action.

The checkpoint retains the exact admitted event **descriptions and evidence digests**, posterior masses and reduced coordinates. Restoring it with an independently pinned checkpoint digest recomputes every coordinate/likelihood through the compiled Grounds; it does **not** re-run material effects or establish new historical TLS authenticity. Full historical evidence should additionally be retained in durable E-CPU/VGI records; a bare SHA-256 digest is integrity identity, not an external signature.

**Scope:** finite declared rational candidates; receiver-local software observations/actions; exact qualified model laws only under their supplied source contracts. There is no externally independent physical sensor, neural checkpoint inference or universal inference-cost improvement. Multiple compact candidates can together consume more memory than a single full-state model; the cost of model multiplicity is counted separately in future benchmarking.

Run on checkout with installed pinned dependencies:

```sh
python -m pytest -q tests/test_evidence_adaptive_ground.py tests/test_evidence_adaptive_fabric.py
python -m pytest -q tests/test_goal_evidence_fabric.py tests/test_intention_predictive_ground.py
```

## Next construction

1. Make source-host actuation **durably idempotent and epoch fenced**, reusing E-CPU's recovery/fencing patterns; reconcile uncertain effects from a separate post-intervention observer before retrying. A persisted `MeasurementReceiptLedger` supports read-only measurements and is not sufficient by itself for irreversible actuators.
2. Define a finite structural hypothesis grammar and source-qualified active probes that can **generate** a new candidate Ground from observed counterexamples, instead of requiring caller-provided candidate laws.
3. Connect the PBG-004 revision controller to TDL-006's actual pinned live mTLS observer route, and preserve VGI policy/peer identities across host failover.
4. Measure whether observation-conditioned structural revision reduces original pretrained-model inference or tool-action work. Distinguish extra source inspections/compilation, persistent Ground cost, Bayesian planner cost and eliminated original execution.
