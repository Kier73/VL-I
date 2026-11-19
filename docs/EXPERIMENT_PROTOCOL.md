# VL-I experimental protocol — make every claim independently checkable

**Protocol status:** design for the first VL-I research campaign. Measurements and pass results must be produced by a future execution run; the existence of this protocol is not experimental proof.

## Rule zero: compare the *same computation*

Do not compare a tiny surrogate, altered weights, a short prompt, or a metadata-only graph against an unmodified full-model result and call the difference optimization.

Define a **Model Contract** before any run:
- exact model name/source and immutable source revision;
- full original trained snapshot identity and source license/permissions;
- tensor encoding and precision; every quantization format and dequantization scale;
- tokenizer assets/hash, template, input text and/or token IDs;
- family graph and operator versions;
- full output contract (logits, token IDs, text, stopping, and if sampled, RNG policy);
- allowed numerical tolerance and environmental equivalence class;
- request authority and side-effect policy.

Declare what was **fully observed**, what was **interpreted**, what was **executed**, and what was **independently verified**. Use distinct statuses, never a single ambiguous “supported”.

## Four experimental paths

| ID | Meaning | What is allowed | What it can establish |
| --- | --- | --- | --- |
| B0 — reference | conventional independent implementation | PyTorch / Transformers / established numerical oracle in a separate code path | expected behavior and baseline cost for identical model/intention |
| B1 — VL-mediated | VL interprets request and calls independent model host | host model inference must be explicitly labeled and metered | intention/transport/authority value, **not** “only VL computes model” |
| B2 — VL-native | all required model operations represented and realized via VL operators | ordinary machine arithmetic inside VL realizer, fully metered, no hidden model-forward APIs | independent execution semantics using the Virtual Layer |
| B3 — VL-navigation | admitted alternate or composed Ground relation returns equivalent contract-observable result | only model-derived source data, certified relations, declared cached state or verified transformations | possible saved conventional model work; requires proof or held-out qualification and full cost accounting |

Some tasks cannot attain B3. A B2 parity success still advances the goal.

## First experiment: VL-I-001

**Source fixture:** start from the pinned pretrained full-decoder observations recorded by VL-64, and independently re-acquire the necessary source bytes. Preserve original model scale and weights.

**Run sequence:**

1. **Acquisition:** record exact model commit, config, tokenizer assets, index, header and payload coverage. Refuse malformed offsets, overlapping ranges, absent tensors and changed source revision.
2. **Lift:** translate the observed graph to a typed Model Ground and verify every required edge, tensor shape/dtype and family-specific operator rule. Preserve unresolved labels for unsupported semantics.
3. **One forward pass:** same known token IDs through B0 and B2; save per-layer output commitments and full final logit information.
4. **Tokenization:** add actual prompt bytes and model tokenizer to the Ground. Measure reversible/nonreversible text mapping and chat template semantics explicitly.
5. **Multi-token generation:** fixed prompt suite with greedy decoding and complete causal-state continuation. Test paragraph and code generation. Report output quality separately from equality to reference behavior.
6. **Recovery:** terminate the host after a committed layer/token, restore with externally pinned checkpoint identity and recover without duplicating already committed effects.
7. **Candidate navigation:** introduce one reusable/certified shortcut (e.g., unchanged prefix state reuse, algebraic composition, qualified equivalence class, sparse selection) and compare B3 with both B2 and B0.

## Minimal negative controls

- Corrupt one weight byte and assert snapshot/parameter admission refusal.
- Supply a valid header for a different source revision and assert refusal.
- Hide a required tensor and assert full-model output remains unresolved.
- Switch tokenizer/chat template while leaving weights unchanged and assert contract mismatch.
- Replace model-family operator semantics with a superficially same-shaped wrong kernel; check independent logit mismatch.
- Attempt an incorrect cached prefix/position or truncated KV state; reject continuation.
- Reuse a request ID with different intention; reject replay.
- Resume with stale epoch or altered checkpoint root; reject.
- Ask for a side-effect tool not authorized by the caller; refuse it.
- Introduce an alternate Ground that matches the training prompts but fails held-out continuation; do not promote it.
- Show only a screenshot of a model page; the outcome must remain *partial observation*, not completed inference.

Negative controls are **successes of refusal**, not failed experiments.

## Measurement schema

Each run should emit a machine-readable record similar to:

~~~json
{
  "schema": "vl-i.experiment.v0",
  "experiment_id": "VL-I-001",
  "model_source": "<source and immutable revision>",
  "source_observation": {
    "config_digest": "<digest>",
    "tokenizer_digest": "<digest or unresolved>",
    "index_digest": "<digest>",
    "verified_weight_byte_count": 0,
    "expected_weight_byte_count": 0,
    "data_coverage": "partial | complete"
  },
  "ground": {
    "architecture_digest": "<digest>",
    "parameter_snapshot_digest": "<digest or unresolved>",
    "unverified_obligations": []
  },
  "run": {
    "path": "B0 | B1 | B2 | B3",
    "input_digest": "<digest>",
    "precision": "<precision>",
    "new_tokens": 0,
    "selected_token_ids": [],
    "final_logits_digest": "<digest>",
    "numerical_max_abs_error": null,
    "numerical_max_rel_error": null,
    "status": "determined | unresolved | refused"
  },
  "cost": {
    "wall_time_seconds": null,
    "cpu_time_seconds": null,
    "peak_rss_bytes": null,
    "peak_accelerator_bytes": null,
    "network_bytes": null,
    "disk_bytes_read": null,
    "weight_payload_bytes": null,
    "vl_operator_count": null,
    "external_model_forward_calls": 0,
    "energy_joules_when_measurable": null
  },
  "verification": {
    "reference_run_id": "<independent baseline>",
    "receipt_chain_digest": "<digest>",
    "independent_verifier": "<identity>",
    "witness_scope": "<claim actually checked>"
  }
}
~~~

This is an illustrative schema, not a frozen implementation. A zero is not a substitute for an unmeasured null. Zero **external model-forward calls** is mandatory for B2/B3 and must be independently auditable, not simply self-reported.

## The fidelity ladder

1. **Source integrity:** same immutable bytes, not merely filename.
2. **Topology fidelity:** graph dependency order and typed operations independently checked.
3. **Parameter fidelity:** the exact original trained tensors (or content-exact proven reconstruction).
4. **Numerical fidelity:** reference comparison at each required layer/operation and final logits, under declared numeric differences.
5. **Token fidelity:** selected IDs, special tokens, stopping and decoding match reference for fixed inputs.
6. **Continuation fidelity:** multi-token causal state, restart and split-host continuity match.
7. **Behavioral fidelity:** held-out prompts produce the contractually required results under the admitted scope.
8. **Navigation advantage:** full end-to-end cost/coverage comparison on the same contract, with overheads included.

Report the highest established rung; do not infer the next one.

## Generality and source modalities

The same experiment interface should accept:
- **Local model files:** complete local snapshot or integrity-scoped file regions.
- **Pinned Hub source:** immutable revision and per-range acquisition receipts.
- **Other hosts / communication paths:** explicit qualified source capabilities and authenticated transfer.
- **Image/screenshot:** capture provenance, extracted visual observations, uncertainty and required corroborating machine-readable data.

A screenshot of a Hugging Face model page cannot establish the complete parameter snapshot. It may lead the active-observation engine toward a missing byte source, which is a legitimate research result.

For new model families, require new semantic obligations and independent comparisons. Do not assume identical behavior from identical tensor shapes.

## Information and predictive experiments

For every candidate alternate description/shortcut, hold out requests **before** fitting the description. Report:
- valid contract coverage fraction and refusal rate;
- exact/precision-qualified output agreement on held-out inputs;
- number of distinct observed states explained and changed/unchanged state prediction;
- entropy/conditional surprise of the target projection under the observed sample distribution;
- description bytes, generation/reconstruction bytes and verification work;
- baseline and optimized end-to-end time, memory, data traffic and host expense;
- adversarial inputs that distinguish the candidate from reference behavior;
- cost of learning the Ground (amortized over reuse count and separately reported cold-start cost).

Avoid multiplying an apparent algorithmic speedup measured on an affine toy trace into a universal estimate for nonlinear trained neural inference.

## Recovery and security

A valid recovery transcript contains original source/model identities, session and intention digests, exact causal-state commitments, external expected root, owner/epoch handoff and independent verifier evidence. Retried transport delivery does not imply repeated inference semantics. Replaying a tool result must not repeat side effects.

Treat model data and generated instructions as untrusted. Verify file lengths, tensor ranges and parser budgets. Do not execute untrusted pickle checkpoints or model-supplied code just to interpret model structure. Live Hub accesses must respect license, permissions and authentication policies.

## How to publish a result

Each experiment PR should include a one-page report answering:

**Question → Snapshot → Ground → Intended observation → Actual realizer → Independent reference → Result → Costs → Refusals/counterexamples → What this does not establish → Next discriminating experiment.**

Include reproducible scripts, fixtures, commands, tested environment, artifact digests and actual test output. If a live network runner cannot be contacted, mark the experiment **not run**, rather than substituting a prose assertion.

No one needs to prove the entire vision in one commit. Every result should make the next hypothesis easier to test.

## Primary acceptance standard

A complete first VL-I prototype will demonstrate:

**pinned full pretrained model data → independently lifted model Ground → text/tokenized external intention → VL-native full-model multi-token realization → reference-checked output → recoverable causal state → measured resource cost**.

The greater **VL-I breakthrough target** is a later, measured B3 result: reaching equivalent useful trained-model behavior through a qualified state-space navigation relation more efficiently than conventional evaluation.

Both outcomes deserve accurate publication.
