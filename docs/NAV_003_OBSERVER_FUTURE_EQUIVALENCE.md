# NAV-003 — Observer-relative finite future equivalence

## Research ambition

NAV-001 revisits an actually executed causal history. NAV-002
navigates a different history if its COMPLETE neural machine state
is identical to an observed one. NAV-003 asks a deeper question:

**Can two genuinely different neural causal states be equivalent
for a particular observer's future intention?**

We require actual source-qualified numerical execution of all
continuations in a declared finite alphabet and time horizon,
not merely matching current outputs.

## Defined claim

Let S and T be different complete Qwen GDN/recurrent/convolution/KV
states at the same causal position, O be a categorical projection
of full F64 layer output, A be a finite set of exact next hidden
input vectors with separate (unverified) token-ID labels, and
H an explicit finite horizon.

NAV-003 certifies S ~[O,A,H] T only if, **for every** word of actions
over A up to length H, the observed outputs are identical at every
prefix of that word when the real trained Qwen layer transitions
are executed separately from S and T.

This is a finite tested-action observer-equivalence result; it is
**not** an unrestricted strong bisimulation, a proof for arbitrary
hidden vectors, or trained text generation. A passing horizon
H=2 does not imply H=3. Matching observer output does not imply
the complete internal state is reconstructable.

## Implementation and proof authority

The new implementation src/vli/future_observer_quotient.py:

1. Defines versioned observer readouts. A categorical channel
   selects one numerical coordinate and an exact finite F64 threshold.
   The output is 0 below threshold or 1 otherwise.
2. Defines a small bounded action alphabet. Every named action
   binds a precise F64 hidden input, plus a token-ID label. The
   label does NOT prove a matching original tokenizer embedding.
3. Refuses equal-complete-state starting points (NAV-002 already
   covers those), changed source, different causal positions,
   stale source heads, missing parameter evidence and excessive
   proof-frontier work.
4. Executes BOTH real Qwen numerical transitions from each
   competing state for EVERY action prefix, breadth-first. No
   unobserved descendant is silently admitted.
5. Emits the **shortest-depth counterexample** with exact node
   receipts and differing observer outputs whenever a future
   disagrees. On complete coverage only, emits a finite-horizon
   equivalence certificate.
6. Re-verifies the certificate by independently reconstructing
   the full breadth-first evidence tree from NAV-001 nodes,
   full F64 numerical output pages, all layer checkpoint pages,
   observer/source identities and externally pinned head.
7. Answers a NEW virtual history coordinate only after positive
   certification, for a path within the declared horizon and
   exactly the same qualified numeric input vectors. The response
   contains ONLY the observer-selected readout and both supporting
   trained transition receipts, **never a fabricated full hidden
   state or native-execution record**.
8. Supports independent verification of that navigation, including
   after restore of an externally pinned NAV-001 state DAG.

The numerical operator remains source-bound to the existing
qualified two-layer Gated DeltaNet or four-layer hybrid GQA worker.
The source and numerical implementation fingerprint comes from
NAV-002. The external observer/forest/proof pins must be held
outside the untrusted record they authenticate.

## Falsifiable experiments

The fixtures are small F32-trained tensor objects in Qwen's
original operator geometry; they are NOT the original Qwen27B
checkpoint or independently validated real token embeddings.

- Construct two different actual causal states from different
  initial hidden vectors. Validate distinct full neural
  Markov-state digests with NAV-002's exact byte verifier.
- For two explicit next-step actions and two-step horizon,
  enumerate 6 action histories from each state: **12 real
  trained causal transitions** plus the two initial transitions.
- Compare actual categorical observer results at every depth.
  On certification, navigate a previously unexecuted
  label-distinct virtual two-step coordinate. Compare its
  declared projection with an independently executed fresh
  numerical Qwen fixture run, performed after prediction.
- Make the observer sharper with a threshold chosen using
  observed data. This is deliberately **post-hoc calibration**,
  not discovery with held-out input evidence. Require the
  new counterexample to refuse the stronger equivalence.
- Compare an observer whose current root readout matches
  even when the future diverges, and ensure the candidate
  equivalence is REFUTED.
- Corrupt attention KV pages and change authority digests.
  Refuse tampering, new inputs outside the bounded alphabet,
  and queries beyond the certified horizon.
- Kill the process that performed initial state and proof
  admission, then independently restore and re-verify the
  same categorical future in a separate Python process.

The first trial of the coordinate-0 sign observer produced a
real counterexample in both fixture geometries, correctly
refusing equality. This is a negative scientific result.
Other observer projections must still pass the exhaustive
proof before becoming admissible.

## Cost and scope

For a two-action, two-step future, the qualifier performs up to
12 trained transitions *before* new virtual coordinates can
be navigated. Numerical reconstruction, F64 hashing, content
addressed pages, attention KV history, checkpoint restoration
and certificate verification have nonzero work and storage costs.

A successful warm observer-navigation query performs **zero
new trained transitions**, not zero total host operations.
Proof building can cost more than a direct numerical query.
The experiment separately measures cold and warm work,
and must not be presented as a pretrained inference speedup.

A source-verified full Qwen27B checkpoint, complete original
tokenizer/embedding relation, and full model decoder remain
distinct research milestones. A virtual categorical readout is
not a model's exact next token.

## Reproduce

Use upstream VL-64 pinned to revision:

    fea0d29d5a8a1edb03d9c66edd5ca4b15a1395d3

Run with the upstream package and pytest installed:

    PYTHONPATH=upstream/vl64:upstream/vl64/tests python -m pytest -q tests/test_future_observer_quotient.py

The NAV-003 GitHub Actions workflow runs Python 3.11 and 3.13
and all NAV-001/NAV-002 and original causal-layer regressions.

## Research continuation

The next major research question is whether sparse active
attention can replace full finite-action enumeration with
a counterexample-guided proof/refinement process. VGI and IFS
have partial-observation and predictive quotient machinery to
study, but matching current results must never count as proof
of unconstrained future equivalence. Measurement, falsification
and independent future-observation checks remain the authority.
