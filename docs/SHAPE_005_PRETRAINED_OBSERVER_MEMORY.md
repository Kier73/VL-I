# SHAPE-005 — Observer-host memory for original pretrained tensor ranges

**Goal:** connect the demonstrated observer-relative structural-memory research to *communication carrying real model weight bytes*, using existing E-CPU and VL-64 authorities. **This is not an assertion that cached bytes are equivalent to cached neural execution.**

**Implementation:** [pretrained_range_memory.py](../src/vli/pretrained_range_memory.py)  
**Live + TCP experiment:** [pretrained_range_memory_demo.py](../examples/pretrained_range_memory_demo.py)  
**Tests:** [test_pretrained_range_memory.py](../tests/test_pretrained_range_memory.py)  
**CI:** [vli-shape005.yml](../.github/workflows/vli-shape005.yml)

## The throughline: a state position should retain the observations that formed it

The original predecessor research is already unusually well aligned:

- **[VL-64 QualifiedShardNavigator](https://github.com/Kier73/VL-64/blob/main/experiments/model_fabric/scaled_model_fabric.py)** accepts exact, pinned model/shard/byte-range requests, checks source receipts and records a source-witness hash chain. Its original range vocabulary already distinguishes **fresh host observations** from **retained observations** and **pinned Ground derivation**. There was no need for a duplicate byte-range parser or alternative pretrained-execution authority.
- **[VL-64 pinned Hugging Face range host](https://github.com/Kier73/VL-64/blob/main/experiments/model_fabric/scaled_model_lab.py)** checks actual HTTPS status 206, full Content-Range and exact returned extent against a pinned model revision. This remains the only live original trained weight source in the experiment.
- **[E-CPU ComputationalObjectStore](https://github.com/Kier73/E-CPU/blob/main/mailcpu/computational_objects.py)** already publishes immutable byte objects into content-addressed pages with independent SHA-256 manifest/page checks. **[E-CPU StructuralMemoryPageBacking](https://github.com/Kier73/E-CPU/blob/main/mailcpu/structural_memory_pages.py)** already provides page publication, immutable retrieval and source-bound recovery. We use those directly, not a second homegrown object store.
- **[VL-I INTERACT-002/003](OPERATIVE_PRETRAINED_GROUNDS.md)** already operates selected real trained Qwen and DeepSeek tensor relations in the original checkpoint topology and renders the resulting observation. Those operations need actual trained payload bytes; a header is not sufficient to invent the original trained values.
- **[SHAPE-003/004](SHAPE_004_VERIFIED_OBSERVATION_MEMORY.md)** previously learned observable network shapes, then showed original-source transition work can be avoided through retained causal state. SHAPE-005 puts the same *verified reuse* principle under the real trained-weight range interface.

The important separation is **what has been retained**:

1. Original externally observed source bytes with a pinned exact tensor/byte identity.
2. A local E-CPU immutable byte-object Ground and the precise original source witness.
3. A future VL-64 navigator observation derived from (1) and (2), **not** a fabricated new HTTP receipt or a re-run of the original source.

No layer, attention operator, full pretrained tokenizer or decoder may be marked executed solely because its tensor bytes are available.

## Architectural flow

\`\`\`text
Observer intention: inspect / realize a qualified original model relation
  |
  v
VL-64 QualifiedShardNavigator
  | exact model + revision + index Ground + shard + byte range + purpose
  v
SHAPE-005 E-CPU-backed observer host
  |
  +-- Unseen range -> original pinned HTTPS/qualified receiver callback
  |                    -> check original receipt and actual bytes
  |                    -> E-CPU ComputationalObjectStore / page digest
  |                    -> original source witness + exact Ground identity
  |
  +-- Seen range -> E-CPU immutable byte-object read/verify
                      -> PINNED_GLANCE_DERIVATION
                      -> original source witness + local Ground head
  |
  v
VL-64 independently checks exact range, hash and declared lineage
  |
  v
Bounded original trained operation + new observer response
\`\`\`

This uses no replacement inference backend. The small numerical probe is the original VL-64 pretrained RMSNorm experiment with independently checked arithmetic.

### Authority and recovery

Every retained entry binds the exact model, forty-hex revision, index Ground digest, shard, byte start/end, purpose, original receiver-host identifier, byte SHA-256 and E-CPU object manifest identity. Warm reads are given a **local observer-host identity** (not impersonating the external HTTPS host) and are explicitly marked \`PINNED_GLANCE_DERIVATION\`. The original VL-64 receiver incorporates the original witness digest into its execution's range-chain.

The checkpoint can be restored with **no original remote host connected** if and only if its externally expected digest, model index, E-CPU object manifests, page contents and preserved original witnesses verify. The revived observer may re-emit only **exact retained ranges**. Previously unobserved ranges refuse until a new source is explicitly connected. A caller can separately **requalify** any previously observed range against its upstream host; if the bytes have drifted under an ostensibly unchanged identity, it refuses rather than silently replacing the original observation.

A digest is an **integrity pin**, not a remote identity attestation. In a deployed federated E-CPU, the trusted receiver must qualify the upstream host/path using the existing mutual-TLS, observer and Hypervisor authority contracts. This first adapter accepts a caller-owned allowlist of upstream-host identities; an untrusted test function's assertion that it is a trusted source is not itself an authentication system. The local TCP fixture is deliberately unauthenticated.

## Measured test — local TCP, fabricated source bytes

The first passing [SHAPE-005 workflow run 44651969290](https://github.com/Kier73/VL-I/actions/runs/44651969290) executed **10 focused tests** and a real 127.0.0.1 HTTP TCP range endpoint. The server supplied a small, *fabricated* BF16 safetensors shard with the **same type of index and request schema** as the original Qwen model. Its bytes are **not Qwen weights**, regardless of the Qwen-compatible address fields used by the test.

| Quantity | First fresh observation | Second qualified retained observation |
| --- | ---: | ---: |
| TCP source range requests | 3 | **0** |
| TCP source bytes transferred | 245 | **0** |
| VL-64 qualified numerical output | 64-dimension synthetic RMSNorm | **Identical** |
| Observed end-to-end sample time | **3.83 ms** | **0.45 ms** |
| Source range evidence | Fresh source witness chain | Locally derived provenance chain |

The two wall times are **one illustrative GitHub-hosted run** including first use, Python code and actual local TCP scheduling. They are **not** a statistical speedup, trained-model result, remote Internet latency measurement or energy profile. The benchmark records original E-CPU CAS publication (~0.31 ms) and warm CAS reading (~0.09 ms) separately.

Crucially the test never replaces the range source with a tuple of cached expected tensor values. The **same original VL-64 numerical test** receives the real bytes of the fixture from the memory-backed observer on the second call, checks them, and independently recomputes the same result.

A saved E-CPU-backed memory snapshot occupied roughly **30 KB** of JSON including the retained local content-addressed pages, provenance and external identity. An independently restarted observer rechecked it and then obtained the same numerical result with **no further source calls**.

### Negative and adversarial tests

- Wrong model, revision, index, shard, byte extent or purpose refuses before a source call.
- Memory budget exhaustion refuses before observing the source.
- Changed original range data is rejected during explicit source requalification rather than silently overwriting the retained Ground.
- Wrong externally pinned checkpoint head refuses.
- Changed or rehashed-forged original witness identity refuses.
- Mutated E-CPU page bytes refuse even if the caller recomputes a checkpoint hash; content-addressed page verification checks actual bytes.
- Recovered memory with no remote host cannot manufacture a never-observed tensor range.
- Code does not claim the existence of a full pretrained inference result.

## Original, full-size trained model probe

The same benchmark has a **separate** live mode:

\`\`\`bash
python -m examples.pretrained_range_memory_demo --mode live \
    --out artifacts/shape005-live-qwen.json
\`\`\`

It retrieves the original, unmodified **Qwen/Qwen3.8-27B** shard index at revision \`37561816e66b5b9153b1b64c10968c18c1e6f4a0\`, checks it against the existing VL-I pinned index SHA-256 \`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df\`, then reads an original pretrained normalization tensor using VL-64's source-qualified HTTPS range host. **No smaller model is substituted.** The complete original checkpoint is advertised as **55,562,855,904 bytes**; only exact required ranges are read.

The second navigator instance retrieves the **same original observed bytes** from E-CPU's retained byte objects. The numerical verification must produce the identical dimension and error, no additional remote range requests and a distinct provenance-preserving local range chain. A third instance recovers the retained observation after a source-free checkpoint restoration.

**Execution is independently tracked:** the CI workflow runs the deterministic TCP tests on all PRs, and attempts the original Qwen HTTPS probe once on the merged \`main\` push, or via an explicit \`workflow_dispatch.run_real_ranges\` input. The public source could be rate-limited, reject ranges, change or become unavailable. The optional live job reports \`UNRESOLVED\` and does not turn a failed source request into simulated pretrained proof. It uses an explicit \`continue-on-error\` guard because network availability is not an invariant of the artifact's local semantics.

Until that live run reports \`MEASURED\` with \`trained_original_checkpoint_observed=true\`, **only the synthetic local TCP behavior has been experimentally established in this PR**.

## Next: cache verified computation, not merely bytes

The stored original pretrained shard is a **qualifying input** for trained mathematics; retaining those bytes doesn't remove the work of recomputing a new operator output for a new hidden state. A more ambitious next experiment should:

1. Bind an observer's exact **intention, original weight snapshot, incoming hidden state digest and trained subgraph identity** to a numerically verified operation result.
2. Retain that value only for **identical pure input and exact operator**; a different hidden state must still execute original trained math unless a more general law is independently proved.
3. Reuse existing E-CPU **ModelOrbitHandoff**, ComputationalObjectStore, ObserverReceipt and source/path authorities rather than creating a second standalone model execution runtime.
4. Compare cold and warm *full costs*: remote HTTPS or mTLS requests, E-CPU object/page serialization, independent verification, source bytes, neural arithmetic, end-to-end response latency and proof/checkpoint bytes.
5. Extend beyond exact lookup into observer-relative **generalizable generative relations**, where a small verified description can form many useful future outputs without requiring a conventional neural forward pass.

**Research target:** a Virtual Layer in which the interpreter learns, certifies and navigates retained source and computational states, so an intention can select a response-producing topology rather than replay all intermediate states. This experiment establishes one source-bound memory interface; broader intelligence and actual pretrained decoder replacement remain research questions.
