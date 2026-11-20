# NAV-002 — Exact causal-state quotient: verified future without repeating neural arithmetic

**Status:** experimental numerical navigation, *not* a learned predictor of unseen
hidden vectors or a complete pretrained Qwen decoder. The source-ground
contract and strong negative controls are as important as the positive result.

## Breakthrough being tested

NAV-001 records observed history-coordinates and their true trained
numerical transitions. Returning to the same history reuses its result.

NAV-002 asks a stronger question:

> Can a history coordinate that was NEVER executed nevertheless obtain
> a correct numerical future from an already verified relation, by
> proving that its current computational state is exactly the same
> as that of another history?

The answer is conditional. A Qwen Gated DeltaNet / attention layer's
future numerical output depends on **the complete current Markov state**
(recurrent matrices, convolution tails, full attention KV timeline and
position), **the exact next hidden input** and **the fixed trained
operator/source identity**. The causal history hash, transport receipt,
and token-ID label independently document where that state came from,
but they are not themselves additional numerical inputs to the
current layer equations.

Consequently, two differently named, independently executed histories
can sometimes have the same complete computational state. When that
equality is proven by source-bound bytes, a previously observed
successor operation for one history is a valid numerical relation for
the other under the same exact next F64 hidden input.

This is a narrow **sufficient-state quotient**, not a claim that a
matching text response or rounded output establishes equivalence.

## Formal relation and its exact qualification

For a fixed operator implementation I and model Ground G, let

    S = (position, conv_0, delta_0, ..., conv_2, delta_2, KV_3)

for the four-layer hybrid Qwen source. The two-layer Gated DeltaNet
variant contains only its two convolution/recurrent components.
A history h has Markov projection m(h) = S.

If source identities, implementation fingerprint, position, geometry,
and *all canonical binary64 state bytes* agree, the implementation
forms an equivalence class:

    h1 ~ h2  iff  m(h1) == m(h2)

Given an exact candidate next input x, the fixed deterministic trained
layer operation is F(G,I,S,x) = (S',y). Therefore:

    m(h1) == m(h2) AND x1 == x2
    => F(G,I,m(h1),x1) == F(G,I,m(h2),x2)

This implication concerns the specified **numerical operator** and
complete state. It is conditional on the determinism and correctness
of the admitted interpreter; no Lean/kernel proof of the numerical
Python implementation has yet been constructed.

The separate token-ID labels remain in the causal coordinate and
witnesses, not in the layer's numerical F64 input. In particular,
**changing a token-ID label while reusing exactly the same hidden
vector is NOT equivalent to executing a different actual tokenizer
embedding for the new token ID.** This experiment deliberately uses
that relabeling to create genuinely new history positions with
mathematically identical numerical input. The next stronger
experiment must establish input identity through the original
tokenizer and source-qualified embedding rows.

## Verification procedure

NAV-002 builds an index over **only existing, actually executed
NAV-001 edges**, with an externally pinned forest head:

1. Reconstruct every original model worker checkpoint with the
   existing source-bound GDN and GQA restore verifiers, including
   finite values and exact source geometry.
2. Re-read every content-addressed state page and hash the raw
   binary64 state data, layer geometry, causal position, immutable
   source root, and source-code/runtime implementation fingerprint.
   The input-sequence lineage, history coordinate, and original
   transition head are **not** used to fake a state distinction.
3. Group numerically equal states into exact classes. Refuse any
   contradictory pair of observed transitions with equal complete
   starting state and equal exact input but unequal output or
   next state.
4. For a proposed new history, find an **observed trained transition**
   from the same exact Markov state and the same full binary64 input.
   If no such transition exists, report UNRESOLVED. Do not approximate.
5. Form a new target coordinate using its own distinct history parent,
   intended token label, and exact input; return a multi-step proof
   linking it to the source-observed transition, output bytes and
   next-state fingerprint.
6. Preserve a separate causal proof chain through each *virtual*
   position. Each predicted next state is eligible for another
   relation lookup, so a complete qualified path may navigate
   several previously unvisited history coordinates.
7. Before using a proof, revalidate its relevant original source
   parent and successor state pages. The verifier independently
   reconstructs the proof using the committed evidence index.
   Restored NAV-001 snapshots can regenerate the same proof only
   with their original independently pinned forest root.

**Crucial integrity separation:** The unexecuted numerical result
stays in the NAV-002 proof. It does NOT become a fake executed
NAV-001 transition, and it does not publish a fabricated recurrent
checkpoint or an original causal execution receipt. The coordinate
remains new, its numerical output is justified by a representative
already executed edge, and exact lineage is retained in the proof.

## Falsifiable experiments

The tests use the small, trained-topology Qwen numerical fixtures, not
the original 27B pretrained source.

**Positive** — two different token-label histories with identical
exact input hidden vectors produce different causal history heads,
but identical complete recurrent/KV state pages. An already executed
successor at the first history must predict the exact F64 output at
the other, previously unexecuted history. After prediction is
committed, a separately run native worker must reproduce the output.

**Multiple future steps** — execute only four distinct numerical
input transitions A, B, C, D as a single causal history. Navigate
16 other distinct token-label histories using the same A/B/C/D
hidden inputs at each depth. Their 16 terminal history coordinates
must be distinct. All returned numerical outputs must equal
independent source-qualified execution, and the original forest
must retain exactly four actual trained transitions.

This shows four known transitions **can support** 16 virtual histories.
It does NOT show 16 different natural-language token sequences can be
modeled from those four transitions. These histories intentionally
preserve their numerical input vectors while changing labels.

**Negative** — choose two genuinely different hidden vectors whose
outputs collide under a deliberately coarse/rounded observer
projection. They must remain **different** exact Markov states.
No mathematical successor may be inferred across the false match.
A new input vector with no observed state-specific transition must
also return UNRESOLVED.

**Adversarial** — stale forest head, altered proof, forged
implementation fingerprint, wrong parent, source-page corruption
after admission, and recovery from an independently pinned
source DAG must all be checked.

## Cost accounting

The expected reduction is in **trained numerical transition
executions during navigation**, not necessarily total host work.

Qualifying the quotient requires source-worker instantiation, full
state-page reads, hashing and exact checkpoint restore. Every
prediction verifies relevant state pages again. This work is
intentionally charged. Network traffic to obtain original model
weights (if any) and the initial trained computations are NOT free.

A full-scale Qwen 27B run would also carry large GDN and attention
states. The fixture operation-count comparison cannot establish
a performance gain on the original model, much less for full
decoder token generation.

The one-step equivalence result can be useful even when the
total Python-side proof validation is slower than computing the
tiny test-layer transition. The objective is an auditable new
navigation capability, followed by measurements and optimization.

## Reproduction

Pin the same upstream VL-64 commit as NAV-001 and QUALIFIED-007:

    fea0d29d5a8a1edb03d9c66edd5ca4b15a1395d3

Install VL-I, VL-64, NumPy, pytest, then run:

    PYTHONPATH=upstream/vl64:upstream/vl64/tests python -m pytest -q tests/test_exact_state_quotient.py

GitHub Actions runs Python 3.11 and 3.13, plus the predecessor
NAV-001 and original hybrid mathematical fixture suites.

## Measured fixture cost — GitHub Actions, Python 3.11

A pinned numerical experiment was executed on GitHub-hosted
Python 3.11, against the complete two-GDN-layer and four-layer
GDN/GQA *synthetic operator-topology fixtures*. It compared all
16 virtual history outputs with 64 independent qualified
reference transitions. Every output matched exactly.

After replacing repeated fresh-worker restoration with rechecks
of the **already admitted full state bytes and exact page hashes**,
the source-backed worker-free proof path obtained the following
single-run results (milliseconds):

| Work | Two GDN layers | Three GDN + one gated GQA |
| --- | ---: | ---: |
| First four trained transitions, retaining full state pages | 35.05 | 24.67 |
| One-time quotient admission and complete worker recovery | 13.10 | 14.70 |
| Navigate 16 different virtual histories | 17.82 | 25.41 |
| Independently reverify all 16 proofs | 19.88 | 28.43 |
| **Total including cold admission and re-verification** | **85.86** | **93.20** |
| Independently recompute 16 full histories (64 trained transitions) | 34.95 | 63.55 |

The **full cold-and-verify cost remains higher** than independent
replay on these tiny fixtures, despite eliminating repeated neural
transitions. It would be scientifically wrong to report an end-to-end
speedup from the four-versus-64 transition count.

The warm hybrid navigation-only path (25.41 ms) was faster than
64 independent tiny-fixture trained transitions (63.55 ms), but
this excludes the charged initial observation, quotient admission
and subsequent independent proof re-verification. Results are
single-run, host/OS/runtime-specific, and **not evidence of real
Qwen 27B neural acceleration**.

The core scientific result is different: successful construction of
a **verifiable numerical-result Ground that remains operational
after the admitted model worker is unavailable**. State bytes,
parent identities, source and implementation remain checked,
without invoking a trained forward pass for the newly described
history coordinates. The next scaling experiment should benchmark
the same relation on original-source full-size states and report
both cold and amortized costs.

[Measured initial and optimized verifier run, Python 3.11](https://github.com/Kier73/VL-I/actions/runs/44670584209).
Machine-readable evidence is retained as workflow artifacts
by the NAV-002 CI job.

## Beyond this milestone

This is exact bisimulation in a *strong* form: every internal causal
variable must agree byte-for-byte. The more ambitious next problem
is **observer-relative predictive equivalence between genuinely
different internal states**.

That requires an admitted observation algebra, a declared action
family and horizon, proofs/counterexamples for all relevant future
observations, and an explicit UNRESOLVED state where coverage or
formal refinement fails. At a finite horizon an exact observational
quotient must not masquerade as an unrestricted future-preserving
bisimulation. VGI's partial observation/proof methods and IFS's
observer quotient experiments provide natural tools, but cannot
be used to promote matching current outputs into unqualified
model equivalence.

A second high-value step is to bind the next numerical input to
actual source-qualified tokens and completed upstream transformer
states; until that is done, it is not a 27B language generation result.
