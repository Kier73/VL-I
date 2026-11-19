# VL-I experimental protocol — make every claim independently checkable

**Protocol status:** design for the first VL-I research campaign. Measurements and pass results must be produced by a future execution run; the existence of this protocol is not experimental proof.

## Rule zero: choose the research claim, not a predetermined implementation

VL-I permits **description-native model creation, novel neural mathematics, direct header-to-token operations, and source-trained model reproduction** as separate legitimate goals. For a **source-equivalence claim**, compare the same computation, input and model. For **generated-model functionality**, compare actual capability on independent tasks; the original model's logits are not the definition of success. Never mistake arbitrary output for a capable model, or a newly generated model for exact recovery of a source-trained snapshot.

Define an **Experiment Contract** before any run. In source-equivalence experiments, include the complete original model contract; for a description-native experiment, record an exact description/seed and interpreter-law identity, not an invented full checkpoint:
- exact model name/source and immutable source revision;
- observed byte scope, description and generative-law identity; original trained snapshot identity when relevant, plus source license/permissions;
- tensor encoding and precision; every quantization format and dequantization scale;
- tokenizer assets/hash, template, input text and/or token IDs;
- family graph and operator versions;
- full output contract (logits, token IDs, text, stopping, and if sampled, RNG policy);
- allowed numerical tolerance and environmental equivalence class;
- request authority and side-effect policy.

Declare what was **fully observed**, what was **interpreted**, what was **executed**, and what was **independently verified**. Use distinct statuses, never a single ambiguous “supported”.

## Two research families, both primary

**Description-native invention (D paths)** asks whether the *description itself* can generate and operate a model. It requires a reproducible interpreter and meaningful capability tests, not original-model weight files.

**Checkpoint-fidelity research (B paths)** asks whether an existing trained model can be reproduced or navigated more effectively. It requires separate reference/oracle evidence appropriate to that claim.

### Description-native paths

| ID | Meaning | What to measure |
| --- | --- | --- |
| D0 — Ground from description | Model header, metadata, screenshot or other bytes are lifted to an operational mathematical description | Exact observation scope, candidate laws, generated state space, interactive transitions |
| D1 — Generative parameters | Interpreter constructs new parameters or equivalent computational rules from description + context | Work/bytes/law assumptions, validity, intervention sensitivity, generated model's independent behavior |
| D2 — Header-to-token operation | Description-native interpreter produces context-dependent token continuations | Correct interaction conditioning, diversity, long-run memory, costs, ablations, usefulness versus simple baselines |
| D3 — Description-native intelligence | Multi-turn conversation, code and authorized action through a navigated Ground with no conventional forward pass | Held-out tasks, coherent context, correctness, tool authorization, sustained behavior, recovery and cost |

The D paths are not certified pretrained-model equivalence. A generative Ground is useful if it can reliably do useful work; any claim that it is also the original trained model requires separate B-family evidence.

### Faithful-source and navigation-comparison paths

| ID | Meaning | What is allowed | What it can establish |
| --- | --- | --- | --- |
| B0 — reference | conventional independent implementation | PyTorch / Transformers / established numerical oracle in a separate code path | expected behavior and baseline cost for identical model/intention |
| B1 — VL-mediated | VL interprets request and calls independent model host | host model inference must be explicitly labeled and metered | intention/transport/authority value, **not** “only VL computes model” |
| B2 — VL-native | all required model operations represented and realized via VL operators | ordinary machine arithmetic inside VL realizer, fully metered, no hidden model-forward APIs | independent execution semantics using the Virtual Layer |
| B3 — VL-navigation | admitted alternate or composed Ground relation returns equivalent contract-observable result | only model-derived source data, certified relations, declared cached state or verified transformations | possible saved conventional model work; requires proof or held-out qualification and full cost accounting |

These B paths are a complementary benchmark, not the architectural ceiling of VL-I. Success in description-native D paths need not wait for B2 or B3.

## First description-native experiment: DNO-001

Take a model description (a header/config and optionally a screenshot) **without pretrained parameter-payload acquisition**. Construct an operational Ground and make it answer changing observer intentions via a declared generative law; retain state between turns. Assess whether any generated tokens are actually useful, and identify precisely which parts of the response come from descriptive bytes, interpreter rules, initial memory and external context. Compare with static-template, random and descriptor-ablated controls. Extend beyond static model facts to paragraphs, code and bounded action proposals.

**Pass gate:** at least one independently reproduced general, context-dependent interaction that changes appropriately under held-out intentions or descriptor interventions and retains a consistent state transcript; this is a *first step* rather than proof of general language intelligence. Do not require equivalence to hidden trained-model weights.

See [Description Is Operation](DESCRIPTION_IS_OPERATION.md).

## Complementary pretrained parity experiment: VL-I-001

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
- Show only a screenshot of a model page; source-trained equivalence remains unestablished, but a separate **description-native generative experiment is allowed** and must be measured under its own contract.

Negative controls are **successes of refusal**, not failed experiments.

## Measurement schema

Each run should emit a machine-readable record. The following is for **B-family source-fidelity** experiments; D-family runs must additionally record description seed/hash, interpreter laws, generated parameters (if any), interaction transcript, held-out task results and control comparisons:

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

A screenshot of a model page is already a legitimate **byte observation and potential generative Ground seed**. It does not by itself prove exact reconstruction of any unseen source-trained snapshot. Both facts matter: explore what it **can** produce operationally; independently test source equivalence only when that is the claim.

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

One complete **source-equivalent** VL-I prototype will demonstrate:

**pinned full pretrained model data → independently lifted model Ground → text/tokenized external intention → VL-native full-model multi-token realization → reference-checked output → recoverable causal state → measured resource cost**.

The greater **VL-I breakthrough target** is to make model description sufficient to form an intelligent operational Ground and **replace conventional model execution altogether**—not necessarily by matching its weights, but by realizing useful capability through novel generative and navigable relations. Qualified B3 pretrained-equivalence shortcuts are one complementary form of that breakthrough.

Both outcomes deserve accurate publication.
