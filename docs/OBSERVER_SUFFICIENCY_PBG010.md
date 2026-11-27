# PBG-010 — Source-derived observer sufficiency and counted language extension

**Research connection:** In-Formation-Space CIG-015, CIG-016 and CIG-017, applied to VL-I PBG-006/007 and the PBG-009 E-CPU quorum observation path. Implements a proof-aware *gate before a new neural acquisition*. The existing E-CPU claim/quorum authority remains authoritative for any subsequent execution or publication.

## What has been implemented

- [`vli.observer_sufficiency`](../src/vli/observer_sufficiency.py) reads full **existing, source-qualified** F64 numerical states from original-topology Qwen GDN-shaped PBG-006 `CausalTape` receipts. Each original tape's `verify()` revalidates the complete trained two-layer causal chain, its source identity and original trained input-output handoffs.
- A declared goal `SourceBit(goal_position, coordinate, threshold)` is measured at a *later* already witnessed causal position. The observational catalogue contains only earlier witnessed positions, with a finite exact-cost coordinate catalogue, so a goal output is never silently read as a prior input.
- Each source tape contributes one exact binary row. Retain the source/witness identities, original tape SHA-256s, observation thresholds as exact F64 hex strings, and exact `Fraction` per-coordinate costs. All candidate thresholds are drawn only from already observed F64 values; no model execution or unseen output prediction takes place during feature discovery.
- Compile a greedy (not globally optimal) observation language. At each step select the feature that separates the greatest number of *currently indistinguishable opposite-goal source pairs* per declared exact rational cost. Admission is conditioned on an independent receiver checker. Source evidence is immutable; refinements create separate digest-bound certificates.
- Return either `SUFFICIENT_ON_WITNESSED_ENSEMBLE` plus an executable binary decoder on the realized signature image, `MORE_RETAINED_REFINEMENT_AVAILABLE` when a configured step budget interrupts the process, or `MISSING_OBSERVER_DISTINCTION` with a concrete pair of source-tape IDs that remain indistinguishable despite all admitted features.
- `suggest_next_action` never automatically runs a model, fabric, or claim. It recommends replaying the admitted decoder, completing remaining retained observation refinement, or extending the observer catalogue / considering a fresh PBG-009 E-CPU-quorum acquisition. Acquisition and publication remain independently authorized operations.

## Mathematical statement and independent check

For witnessed source tapes \(X=\{x_1,\ldots,x_n\}\), declared earlier observation vector \(q(x)\in\{0,1\}^k\), and later binary intention/goal \(g(x)\in\{0,1\}\), the exact finite-source sufficiency condition is

\[
\forall x,y\in X:\quad q(x)=q(y)\implies g(x)=g(y).
\]

The receiver checks every pair of retained *source-valid, genuinely witnessed* tapes. A counterexample is the exact pair \((x,y)\) with the same observation code and different goal. This is **not** a certificate over all possible neural hidden states: it is exact for the declared empirical finite ensemble of already executed causal tapes. If the condition holds, \(g\) factors uniquely through the realized observation image; values on unrealized codes are not established.

Following CIG-015/016, let \(H_q(u,b)\) count source tapes in observation fiber \(u\) with goal bit \(b\). For a new source-derived predicate \(f\), the extension conserves *each* count:

\[
H_q(u,b)=H_{(q,f)}((u,0),b)+H_{(q,f)}((u,1),b).
\]

The source verifier computes the entire table with a separate grouping/pairwise routine, checks each cost-ordered step and the source conservation law, checks the complete decoder, binds every source/witness identity, and refuses rehashed forged histograms, altered selection orders, stale source witnesses or changed goal bits. A producer-supplied hash is only a content fingerprint, not an authenticated proof that the underlying physical model was executed.

The score for a candidate retained feature \(f\) at current observation \(q\) is

\[
S(f)=\frac{|\{(i,j):g_i\ne g_j,\ q_i=q_j,\ f_i\ne f_j\}|}{c(f)}.
\]

This is an **exact declared weighted-pair separation heuristic**, not Bayesian information gain and not a proof of minimum feature count. `MAX_TAPES=12`, `MAX_FEATURES=128`, `MAX_STEPS=24`, with up to 24 declared native numerical coordinates. The input source need only supply prior verified causal receipts; the complete originally trained neural operation never runs during this proof construction.

## Direct cross-repository proof interface — actual CIG executable

`export_cig017_source(table,selected)` builds a small canonical **one-hot finite-source CNF**: exactly one selector variable for each originally witnessed neural tape; source clauses tie each selected observation bit and the goal bit to that selector. There are precisely \(n\) satisfying assignments, one per original tape. Thus the original CIG-017 independent compiler and checker can operate on exactly the same source-relative implication:

- For a sufficient source, run actual `global_sufficiency.synthesize` and the independent `verify_global_sufficiency.verify` on the exported original finite-source CNF, retained singleton observation predicates and singleton goal predicate.
- For an insufficient source, the actual CIG-017 paired-source witness recovers two source-valid assignments with identical observation bits and opposite goal bits. The original tapes remain the qualified numerical evidence for those finite-source assignments.
- In a separate control, the actual CIG-015 `Language.ensure` extends a seed observer by the requested goal predicate; its existing `verify_language_extension.check_extension` checks histogram conservation and source-witnessed splits. This is directly executed code in the In-Formation-Space repository, not a reimplementation of its proof checker.

These proofs deliberately concern the *finite exported witnessed ensemble*. The one-hot export itself does not confer authentication on arbitrary data; only `from_neural`, `compile_for_neural` and `verify_for_neural` reconstruct it from original verified source tapes and a separately pinned expected witness identity.

## Connection to E-CPU's PBG-009 ownership and recovery

The PBG-010 certificate binds parent `source_digest`, `native_witness_digest`, the full list of qualified tape identities, the goal and observer-choice costs. A caller operating on a PBG-009 `QuorumNeuralObservation` session can check it before creating another claim. A sufficient observation language yields a source-relative decoder and can avoid a *new* neural source request; missing evidence is passed to the existing PBG-007 selector and PBG-009 claim authority rather than inventing a new owner or consensus protocol.

For any new input acquisition, the E-CPU `TransferableExecutionClaimStore` still owns pure verification/publication result fencing; PBG-009 performs source-receipt validation and live quorum status. A local CIG receipt neither grants permission to execute on a remote host nor bypasses that quorum. The currently integrated TLS and quorum tests run on one CI host, and the new CIG proof does not assert independent physical deployment.

## Reproduction, evidence and next experiments

- Local independently replayed exact mathematical source-fiber suite: **25 passing tests** (Python 3.13), including all sixteen 4-row binary goal patterns, exact one-hot source solutions, source witness mutation, forged counts, changed greedy order, cost tie-breaks, budget cuts and canonical JSON checkpoint restoration.
- The new native source suite runs the actual VL-64 trained two-layer Qwen-shaped **synthetic-F32** fixture using VL-I's existing source witnesses, then reconstructs observer positions and verifies the result without trained-layer reexecution. It also checks altered original F64 receipts and a changed parent source revision.
- The direct CIG suite checks the exact source CNF with CIG-017's real compiler and independent checker, and runs CIG-015's actual retained-language extension. These native/CIG integration tests are authored for CI and are not claimed as locally passed until they actually run.
- The GitHub Actions matrix uses Linux/Windows and Python 3.11/3.13, with pinned E-CPU, VL-64, VGI and IFS commits. CI checks the original PBG-006 source baseline and SHAPE-010 as regressions.

**Next:** Treat the CIG-019 shared proof DAG and CIG-020 adaptive gate evaluation as *resumable structural-memory components* behind an E-CPU transferable claim; cache and prove unchanged decoder subgraphs after a neural source revision, test recovery while an observer-language refinement is partially prepared, and compare fresh native acquisition against cheaper re-observation of retained source coordinates. CIG-021's nonlinear time navigation can join once an explicit source-qualified temporal law exists rather than being inferred from a few neural traces.
