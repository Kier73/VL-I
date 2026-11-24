# PBG-009 — E-CPU quorum-backed neural observation recovery

## Goal

Keep the *meaning* of an active neural observation constant while its responsible peer/host and physical process change. Publish exactly one semantically accepted result through **E-CPU's existing transferable execution-claim authority**. This replaces the IFS SQLite publication owner used by PBG-008 for this experiment; it does **not** replace E-CPU's quorum or add an independent ownership ledger.

Implemented in [`src/vli/quorum_neural_observation.py`](../src/vli/quorum_neural_observation.py); tests: [`test_quorum_neural_observation.py`](../tests/test_quorum_neural_observation.py) and [`test_quorum_neural_math.py`](../tests/test_quorum_neural_math.py).

## Reused systems and authority

| Existing component | Role |
| --- | --- |
| [E-CPU TransferableExecutionClaimStore](https://github.com/Kier73/E-CPU/blob/main/mailcpu/transferable_execution_claim.py) | Creates portable claims, verifies current Merkle/quorum root, advances assignment epoch, quorum-commits one result digest and typed receipt digest |
| E-CPU RecoveryLineageCoordinator, SeedCommitMembership, durable RecoveryLineageAuthority journals | Single fixed-membership majority authority for committed claim-state generations; no second consensus layer |
| E-CPU OperatorSpec | Full typed, deterministic, pure contract for **prepared observation verification**, checked again at the receiver |
| E-CPU ComputationalFabricPeerClient / ComputationalFabricPeerListener | Direct CA-trusted mutual TLS claim-status refresh scoped to exact authenticated receiving peer and Host; no new TLS protocol |
| [PBG-007](ACTIVE_NEURAL_COUNTEREXAMPLES_PBG007.md) | Selects one native Qwen-shaped input trajectory and emits a source-qualified acquisition receipt |
| [PBG-006](NEURAL_TRACE_GROUND_PBG006.md), [SHAPE-009/010](SHAPE_010_CAUSAL_BASIS_EXPANSION.md), VL-64 | Genuine trained Qwen-shaped numerical execution with source-checked, causal F64 transfer and chain receipts |
| [PBG-008](DURABLE_ACTIVE_NEURAL_ACQUISITION_PBG008.md) | Durable immutable intent and prepared neural receipt serializers / fsynced local outbox; **IFS SQLite route owner authority is not used** |

## Mathematical contract

Let `I` be the exact selected intention, including parent native witness and source, observer, catalogue, selected input path and nonce. Let `V` be the source-qualified prepared neural result. Let `S` be E-CPU's typed pure verifier:

\[
S:\operatorname{VerifyPrepared}(I,V)\longrightarrow (\mathrm{digest}(V),\mathrm{receiptDigest}(V))
\]

The **transferable operator is this deterministic verification**, not the potentially expensive or hardware-sensitive neural execution that created `V`. Neural acquisition remains a separate source operation that may be retried after a crash. No physical exactly-once execution is asserted.

The claim has a host-independent semantic identity:

\[
K=\operatorname{SHA256}(\text{parentRequest},\text{childIntention},\text{address},
       \text{typed verifier},\text{source Ground})
\]

The current assignment is `A_e=(epoch, peer_id, host_id, implementation_fingerprint, state_generation)` with `e=1,2,...`. An E-CPU quorum reassignment preserves `K` and increments `e`. The owner may publish only if `PortableExecutionClaimBundle.verify_for_target()` succeeds against **live** quorum state and the complete operator/parent/child/address/source contract matches. A cached ticket or old Merkle root is insufficient; `store.get(claim_id)` refreshes the ticket just before operation admission and completion.

Once the source-qualified prepared result passes the existing PBG-006/007 receipt checks, the current peer calls `store.complete(claim_id, expected_epoch=e, owner_peer_id=..., owner_host_id=..., implementation_fingerprint=..., result_digest=SHA256(V), child_receipt_digest=...)`. E-CPU's current-state quorum commit is the semantic acceptance point. Identical completion retries are idempotent. Stale epochs, changed results, changed typed receipts, wrong hosts, and changed implementations are refused. The local accepted result file is an outbox, not another commit authority.

## Crash/recovery ordering

1. **Intent preparation:** persist the unchanged observation request in an immutable fsynced file. Create the typed E-CPU claim under existing three-member quorum and persist its claim ID. A crash after claim issue and before pointer persistence is recoverable by repeating identical `store.create`.
2. **Native source acquisition:** only the peer owning the current E-CPU assignment begins the selected Qwen-shaped native run. It checks the current E-CPU ticket again after executing the native transition. Qualified native F64 receipts become a separate immutable `prepared-<epoch>.json`. A crash before prepare may repeat the numerical source acquisition; it cannot commit twice.
3. **Host loss/reassignment:** new peer claims the same K with `store.reassign` using a source-loss evidence digest, which does not independently prove the old peer is dead. Quorum revokes the old epoch's result-acceptance authority. Old Merkle tickets become invalid.
4. **Prepared recovery:** a replacement can reconstruct `V` from retained original causal receipts, verify native witness chain/selected input/observer/source identities without retraining, then complete the current E-CPU claim. Prepared result artifacts retain their own source assignment epoch as evidence; **the quorum controls acceptance**, not the old file.
5. **Completion recovery:** if process dies after E-CPU `complete()` but before local accepted outbox fsync, `publish()` reads current quorum `COMPLETED` status and reconstructs the *same* digest-bound outbox; it does not create a new completion generation.
6. **Independent audit:** reload quorum state and Merkle proof, reconstruct original source receipts, check result/receipt digests and committed completion generation. The resulting `WitnessedNeuralTraceEnsemble` may be passed to PBG-005's exact finite trace learner without any new native numerical calls.

## Independent verification and current scope

The native test suite performs real synthetic-F32-weight Qwen-shaped two-layer execution from VL-64, source receipt verification via SHAPE, three independent recovery-authority journals, immutable claim-state generations, E-CPU Merkle verification, ownership/reassignment/completion, and two independent competing replacement Python processes sharing the durable quorum journals. Specific adversarial tests include stale owner and stale ticket, wrong typed target, altered prepared neural receipt, changed selected catalogue, crash after quorum result commit before outbox, completion idempotency, expired-ticket mTLS refresh, wrong TLS-requested Host, mismatched implementation, and mTLS peer revocation after reassignment.

The native CI topology uses **three durable authority journal identities on one CI machine**, not physically separate peers or WAN reconfiguration. `admit_authenticated_observer()` additionally integrates E-CPU's **real direct mutual-TLS** `ComputationalFabricPeerClient.get_execution_claim_status()` receiver check: the test starts the native `ComputationalFabricPeerListener` over a TLS loopback socket with CA-trusted peer certificates and exact DNS SANs, refreshes a globally expired Merkle ticket, verifies the exact selected source/operator/host, and refuses the former peer after reassignment. This is an authenticated *admission* proof, not proof of remote physical computation or cryptographically authenticated observation-result delivery. For physically separated Hosts, bind the existing `DistributedRecoveryLineageCoordinator`, authenticate the forwarding result path, and perform the same quorum completion at the origin; do not add a second result authority.

## Key falsifiers

- An obsolete assigned Host can complete `K` after quorum has advanced its epoch.
- A changed recorded F64 interlayer receipt, source native chain, or observer can commit under the same semantic claim.
- A crash between result completion and local publication creates a second accepted completion generation.
- Two replacement processes can concurrently reassign the same claim and both commit accepted (possibly different) results.
- A prepared-result file, self-hashed without a valid SHAPE native witness, is accepted by the pure verifier.

If any occurs, the recovery identity property fails. The CI tests make these failure modes executable rather than leaving them as declarations.
