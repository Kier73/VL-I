# PBG-008 — Durable, owner-fenced neural-source observation

**Build:** `vli.durable_neural_acquisition`; standard-library journal tests; actual IFS/E-CPU/VL-64 integration tests; cross-platform CI.

## Throughline and reuse

An intention now selects a **native causal input path** (PBG-007). The next requirement is to preserve the exact request and its completed source observation when the executing observer disappears, and prevent a displaced observer from publishing a competing accepted result. This build reuses four existing mechanisms rather than constructing another consensus system:

| Existing repository mechanism | Reused role |
| --- | --- |
| [VL-I PBG-007](ACTIVE_NEURAL_COUNTEREXAMPLES_PBG007.md) | `select_native_run` and `acquire_native_run`: source- and nonce-bound selection; one fresh native Qwen-shaped execution; complete causal receipt qualification |
| [VL-I PBG-006](NEURAL_TRACE_GROUND_PBG006.md) / [SHAPE-009](SHAPE_010_CAUSAL_BASIS_EXPANSION.md) | `CausalTape.verify`, qualified interlayer F64 chain and original source identity; `WitnessedNeuralTraceEnsemble` reconstruction |
| [IFS USM-007](https://github.com/Kier73/In-Formation-Space/blob/main/bridges/model-native-operations/model_route_fencing.py) | Actual `ProofActionRouteEpochLedger` + `SQLiteRouteEpochLedgerStore`: request admission, CAS owner-generation claim, stable one-attempt execution ticket, append-only audit |
| [IFS SRH-005](https://github.com/Kier73/In-Formation-Space/blob/main/experiments/portable-realization-capsule/STATEFUL-PREPARED-RESULT-RECOVERY-005.md) | Prepare result before authority publication; reconstruct stored result after process loss; commit without replaying an already witnessed pure result |
| [VL-I TDL-005/006](ROUTED_AUTHENTICATED_REFINEMENT_TDL006.md) | Existing live IOS-003 mTLS source-peer identity checks for the *next* remote-host deployment; no duplicate TLS protocol is asserted here |

## Executable state sequence

```
PBG-007 selected native path + externally retained parent witnessed ensemble
  → PBG-008 exact input catalogue, nonce, parent/source/observer digest
  → immutable fsynced `intent.json`, then IFS open_epoch + admit_request
  → source execution: Qwen native complete two-layer chain
  → revalidate every original F64 interlayer causal witness
  → immutable fsynced `prepared.json` containing complete CausalTape
  → original observer crashes / is killed
  → replacement IFS CAS claim, owner generation advances
  → recovery revalidates prepared bytes and PBG-006 causal witness
  → IFS `record_execution_authorization` (stable one-attempt ticket)
  → immutable `publication.json` bound to ticket + source Ground revision
  → independent verification of the IFS audit event + native witness
```

The IFS `execution_authorized` event is used as the **logical result publication authority gate**: the observed result must already be prepared and verified before this gate is requested. Its `execution_id` is stable under recovery and is the deduplication identity. This is a narrow adaptation of its existing execution-authorization contract, **not** a new generic result-commit primitive, and the prepared-result bytes are kept in a separate fsynced store. No atomic commit across SQLite and the result directory is claimed. On crash after the IFS event but before outbox publication, a replacement with the same pinned source and request can recover the same stable ticket and populate the missing outbox.

### Identities and invariants

For a source witness `G`, chosen input path `c`, catalogue `C`, selection nonce `n` and PBG-007 plan `p`, let

\[
R=H(\text{schema},G,\text{observer},C,c,n,p.\text{proposal_digest}).
\]

`intent.json` is immutable and its digest `R` is admitted in IFS under epoch `p.proposal_digest`, which cannot be reused. Competing owners `o_i` may publish only if the durable IFS authority has their exact `(owner_id, owner_generation)` at the linearizing authorization call. A displaced owner cannot claim a new ticket. The prepared result binds the exact native source root, full output receipt chain, selected input and new Ground source/witness digests. It is independently reconstructed without invoking Qwen training-weight numerical execution.

`publication.json` names `R`, the prepared receipt digest, the stable IFS execution ticket and the new Ground identity. `verify_publication` reconstructs the prepared output and requires **exactly one matching IFS execution-authorization event**. A replay returns the same logical ticket, not a second accepted result.

The source path is F64 hex-serialized, rather than decimal-rounded, and cannot be changed after intent admission. A prepared result that disagrees with the original source, native result or selected path is refused. An attempt to replace an already retained prepared result with different contents is refused.

## Crash and recovery tests

- **Crash after native execution, before source publication**: original prepares the new `CausalTape`; another Python object claims its ownership epoch and restores from F64 and SHAPE receipts, with zero new trained calls.
- **Stale owner**: the original owner cannot publish after the claim and cannot change the exact input/parent identities.
- **Two separate replacement interpreters**: both observe the same first owner generation and race the IFS SQLite compare-and-swap; only one successfully claims and publishes, while exactly one execution ticket remains in the audit trail.
- **Crash between ledger event and publication outbox**: the later owner can re-materialize the same logical publication using the deterministic ticket.
- **Corrupted or rehashed prepared data**: revalidated original causal F64 receipts and source path prevent a changed native result from being accepted.

CI tests use a **small synthetic-weight** trained Qwen-shaped two-layer native fixture with E-CPU and VL-64 source adapters. No original Qwen27B pretrained weights are acquired. Request semantics concern fresh synthetic neural input vectors, not verified original tokenizer embeddings.

## Evidence and next

This step validates a **single-host, single-database fault domain with genuinely separate local OS processes**. IFS SQLite is not a multi-host quorum service. Rechecking the active authority before returning cannot make the two stores ACID or stop an already-authorized external physical actuator; the observation operation here has no external side effect beyond computation and retained output. On Windows, file contents are flushed but directory fsync is not available, so local crash durability has a narrower platform guarantee.

**Next:** substitute the existing E-CPU fixed-membership `TransferableExecutionClaimStore` high-water/quorum for a remote owner authority, bind the actual TDL-006 live mTLS peer certificate to the selected probe and source receipt, and repeat the crash test with physically separate observers. The existing PBG-005 exact-finite-source all-future certificate applies to the resulting *observed trace ensemble*, not all possible future Qwen neural states.
