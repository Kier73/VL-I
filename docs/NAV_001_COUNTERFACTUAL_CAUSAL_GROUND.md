# NAV-001 — Counterfactual Causal State Navigation

## The question

Can an observer return to a source-qualified causal prefix of a trained
model, explore several incompatible futures, and later resume any of them
without recomputing the successful prefix?

NAV-001 makes this executable over VL-I's existing Qwen model workers:
the two-layer complete Gated DeltaNet chain (QUALIFIED-006) and the
four-layer GDN/GDN/GDN/gated-GQA hybrid block (QUALIFIED-007). It
composes their existing numerical, source, checkpoint and state-page
contracts instead of copying their implementation into a new engine.

The new structure is a **counterfactual causal DAG**. Its edges are
actually executed trained transitions. Its nodes are exact, source-bound
addresses of states that have been realized. This is NOT a decoder,
unseen-token predictor, or a claim of complete Qwen language generation.

## State-space mathematics

Given an admitted source Ground G and a causal parent p, a proposed
edge has a token-ID label t and a complete finite F64 input x.

- H(x) commits to the **exact little-endian F64 bytes** of x.
- Coordinate C = H(G, p, t, H(x)) exists **before** its output exists.
- The model worker executes its original geometry: (S', y) = F_G(S, x).
- The retained record binds C, its parent, depth, model/revision/index,
  full input and output F64 digests, all per-layer handoff witnesses,
  a causal head, exact state lineage and paged checkpoint.
- Another proposal with the SAME parent, token label and F64 input
  navigates to this already realized node **without a new trained
  transition**.
- Even if the token ID and input are identical, a DIFFERENT causal
  parent produces a different coordinate. The architecture refuses
  false state merging.

A fixed-width coordinate locates a state; it does not contain the
full state, perform its computation, or make unobserved results known.

## Admission, atomicity and recovery

1. Supply a factory producing a **fresh** Qwen qualified two-layer
   chain or four-layer hybrid block. Model name, immutable revision,
   index Ground, layer order, width and genesis must remain unchanged.
2. Every operation receives the caller's separately trusted forest
   head and parent-node digest. An untrusted manifest is not its own
   authentication authority.
3. An existing coordinate reads independently hash-checked output
   pages and verifies its causal receipt and state checkpoint. An
   unexplored coordinate restores the exact parent into a fresh worker,
   executes real model math, and checks every F64 layer handoff.
4. Persist each layer's Gated DeltaNet state and attention KV chronology
   through the already proven source-bound LocalStatePages backend.
   The complete new checkpoint is re-admitted into another fresh worker
   before the new DAG head is published.
5. Snapshots are serializable with their own SHA-256 integrity digest
   and ordered forest head. Restore requires an **externally retained**
   snapshot digest. Strict recovery checks all source-bound numerical
   pages, causal identities and topological ancestry.
6. Failed numerical work, interrupted page storage, tampered state,
   stale heads and forged input ancestry do NOT publish a new edge.
   Orphaned content-addressed pages are possible after interruption.

This is **single-writer local prototype semantics**, not distributed
quorum consensus. Trust in the source-qualified worker and the external
digest pin is still required. The native worker continues to execute
CPU/NumPy trained arithmetic.

**Important source qualification:** a token ID on the edge is a label,
NOT a proof that the supplied x is the original pretrained embedding
of that token. The caller must separately admit and prove the original
tokenizer mapping and embedding observation before claiming an
original-token trained decoder transition. The fixture tests supply
arbitrary small hidden vectors and synthetic source tensors.

## Experiment — 16 possible four-step futures

A full depth-four binary tree has 16 distinct leaves. Re-running each
four-step leaf from the initial state requires **64 causal transitions**.
Computing each unique tree edge once requires **30 transitions**:
2 + 4 + 8 + 16 = 30.

NAV-001 tests the 30-edge realization, verifies a leaf against an
independent uninterrupted qualified execution, checks that revisiting
the same edge does not recompute the trained operator, and recovers
sibling futures without cross-contaminating recurrent or attention
state. The tests cover both two-layer and four-layer worker geometries.

**These are operation-count predictions, not measured end-to-end
speedups.** Extra source reads, model-worker startup, I/O, paging,
hashing and checkpoint validation may outweigh any saved math. At
original model width retained recurrent-state pages can be substantial.

## Run the experiment

With upstream VL-64 checked out at the same pinned revision as the
QUALIFIED-007 workflow, install VL-I, VL-64, NumPy and pytest, then run:

    PYTHONPATH=upstream/vl64:upstream/vl64/tests python -m pytest -q tests/test_counterfactual_navigation.py

The new GitHub Actions workflow runs Python 3.11 and 3.13 and preserves
the existing qualified-layer test gate. The real 27B model requires an
explicit separately budgeted live experiment; normal PR tests DO NOT
download full original checkpoint weights.

## Scientific next step

Bind the branch inputs to the **actual verified original token-embedding
source and a completed decoder's qualified logit selection**. At that
point this DAG can support speculative candidate evaluation, beam search,
exact recoverability and observer-shaped quotient experiments. To
merge different internal states under a cheaper learned Ground, prove
equivalence of ALL relevant future observations, not just one matching
current rendered output. Until then, no inference shortcut for unseen
neural states is claimed.
