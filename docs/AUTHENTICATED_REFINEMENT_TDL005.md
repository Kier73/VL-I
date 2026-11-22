# TDL-005 — Live authenticated observer evidence for intention succession

**Implementation:** [`vli.authenticated_refinement`](../src/vli/authenticated_refinement.py), [independent record verifier](../src/vli/authenticated_refinement_verify.py), [live mTLS integration tests](../tests/test_authenticated_refinement.py), and [CI matrix](../.github/workflows/tdl005-authenticated-refinement.yml).

[TDL-004](FABRIC_REFINEMENT_TDL004.md) formed observer-specific executable requests over receiver-owned E-CPU routes, then used a single receiver-admitted VGI adapter. This round extends the **measurement admission route** to *multiple peer-specific live mTLS connections* using the existing E-CPU [IOS-003 measurement service](https://github.com/Kier73/E-CPU/blob/main/mailcpu/intention_measurement_fabric.py).

## Source semantics and mathematical contract

A verified old/new VL-I processor pair has coordinate \(z_o=B_ox\), target \(z_n=B_nx\), and a source-row acquisition plan with offered rows \(a_i\). That plan is already verified by `verify_cost_plan` and carries the exact source-law and intention digests.

For each selected row \(a_i\), the receiver requires \(q\ge2\) distinct pinned peer identities \(p\) to return a bit \(b_{i,p}\). The receiver accepts only a complete, strict VGI quorum:

\[
\boxed{\forall i\in S:
\operatorname{AUTHORIZED}_{\mathrm{VGI}}(i)
\ \Longleftrightarrow\
\operatorname{Finalize}\bigl(\{
\operatorname{LiveMTLSRead}(p,i)\mid p\in P\}\bigr)
=\mathrm{AUTHORIZED}.}
\]

Here each live read is produced by **the actual** `MeasurementPeerClient.resolve` method, not by accepting a serialized `ObserverReceipt` from the caller. The IOS-003 client requires CA-verified peer TLS, a pinned expected peer DNS SAN, native `ObserverContract`, challenge/request identity, scheduler coordinate, and a current receiver authority digest. The IOS-003 server requires a preinstalled `HostedMeasurement`, evaluates its receiver-local `admit` callback, and commits deterministic read results to its durable ledger before acknowledgement.

Given admitted agreeing bits \(b_i\), TDL-003's independently checked decoder reconstructs

\[
\boxed{z_n=L\begin{bmatrix}z_o\\b_1\\\vdots\\b_k\end{bmatrix}}
\]

with \(B_n=L[B_o;A_S]\). An absent, refused, inconsistent, or non-bit response cannot produce a new coordinate.

## Two separate identities: mathematical question and peer authority

The shared model for each selected observation has only **two declared bit outcomes**, regardless of the original source width. This does **not** enumerate the source's \(2^n\) states. The source row remains a separate semantic contract:

\[
\mathrm{RowGround}_i=
H(\mathrm{source\ digest}, \mathrm{plan\ digest},
\mathrm{offer\ ID},a_i).
\]

Each pinned provider \(p\) receives an independent `ObserverContract` with its own `observer_id=p`, while the source row and finite output domain remain the same:

\[
\mathrm{contract}_{i,p}
=H(\mathrm{RowGround}_i,\, p,\, \mathrm{intent\ context}).
\]

Its request challenge adds a receiver-supplied random nonce and binds the current `observation_context(plan, old_native_state)`, provider identity, and offer identity. Fresh attempts therefore receive different identities even when an old processor coordinate happens to be equal. Retrying the **same** challenge intentionally uses IOS-003's durable idempotent completion semantics.

## Existing APIs, composed in execution order

```python
from mailcpu.temporal import SchedulerReading
from vli.authenticated_refinement import (
    prepare_peer_measurement, acquire_authenticated_refinement,
)
from vli.authenticated_refinement_verify import (
    verify_authenticated_refinement_record,
)

# args = old/new SymbolicSource, quotient, interpreter bundle (six values)
# plan = independently checked TDL-003 cost-aware plan
# clients maps TWO OR MORE pinned provider IDs to existing live
# MeasurementPeerClient sessions created through connect_mutual_tls().
# Providers preinstall HostedMeasurement specifications matching:
spec = prepare_peer_measurement(
    plan, old_native_state=old_z, offer_id="cheap-left",
    peer_id="peer-left", authority_digest=authority_digest,
    scheduler=SchedulerReading("session", 5),
    challenge_nonce="1" * 64,
)
# Host installs HostedMeasurement(spec.contract, spec.model.digest,
#           authority_now, admit, read, allowed_values=(0, 1)).
result = acquire_authenticated_refinement(
    *args, plan,
    old_native_state=old_z,
    clients=clients,
    required_sources=("peer-left", "peer-right"),
    authority_digest=authority_digest,
    scheduler=SchedulerReading("session", 5),
)
assert verify_authenticated_refinement_record(*args, plan, result)
new_z = int(result["refinement_receipt"]["new_native_decimal"])
```

`acquire_authenticated_refinement` generates a fresh 256-bit challenge nonce by default. The optional `challenge_nonce` parameter exists for deliberate replay/recovery tests; repeating it while the physical source changes reuses the previously completed IOS-003 reading, by design. A distinct observation epoch therefore requires a new challenge.

The complete executable TLS fixture is [`tests/test_authenticated_refinement.py`](../tests/test_authenticated_refinement.py). It provisions a local CA, distinct server certificate identities and a client certificate with OpenSSL; starts two different live TLS endpoints; installs E-CPU `HostedMeasurement` policies for selected source-row projections; and invokes the public E-CPU IOS-003 client/server methods.

The receiver then translates the native IOS-003 admitted readings into VGI's existing `ChannelMessage`, `ChannelExchangeLedger`, `EncounterLedger`, `TransportFrameReceipt`, and strict `TransitionEvidenceLedger`. VGI finalizes each selected row only after receiving matching values from all required peer identities. The accepted `acquire_vgi_refinement` result is a normal TDL-003 coordinate receipt.

## Verification and the trust distinction

| Layer | Verified in the current execution | Interpretation |
|---|---|---|
| Exact source intention | VL-I source and quotient, selected row masks, decoder, plan digest | Mathematical sufficiency under admitted source law |
| Native peer transport | IOS-003's live mTLS client/server, CA validation and DNS SAN binding | Authenticates peers **during the live transport session** |
| Request identity | IOS-002 plan, native model/observer contract, scheduler and challenge digest | Prevents substituting a different question or stale request |
| Host admission | IOS-003 `HostedMeasurement.admit`, allowed bit values and durable completion ledger | Receiver/host-specified permission, reproducible read |
| Evidence qualification | Per-peer VGI source claims, strict quorum, explicit finalization, unchanged source-row map | Only agreement creates a reconstructible bit |
| Result record | Independent record verifier rederives each native contract/request identity, VGI decisions and coordinate | Checks **exported content consistency**, *not* historical TLS authenticity |

**Vital scope:** an exported `ObserverReceipt`, VGI source ID, SHA-256 digest, or `verify_authenticated_refinement_record` success is **not** a portable certificate that the peer authenticated over TLS at some earlier time. The live acquisition function has observed that connection while invoking IOS-003; the offline verifier deliberately provides only mathematical/record-consistency evidence. TLS endpoint identities here are separate local test certificates/threads, not independently administered physical observer hosts.

Two agreeing authenticated transport endpoints also do not prove that their underlying physical measurements are independent or true: a trusted CA, admitted host software, policies and source-law grounding are still part of the receiver's assumptions.

## Observed CI verification

**2026-11-22:** [GitHub Actions run 45071510754](https://github.com/Kier73/VL-I/actions/runs/45071510754) completed successfully in all four combinations of Linux/Windows and Python 3.11/3.13. Each job passed **9** live IOS-003 TLS/quorum/recovery integration tests, **7** TDL-004 E-CPU route regressions and **7** TDL-003 VGI/cost regressions (**23 tests/job, 92 executions**). The tested source head was `c7242e5372fe660dd1edb90ef474b32e2800b2c6`; subsequent changes were documentation/index changes.

The earlier exploratory runs established two concrete implementation requirements: (1) IOS finite `FiniteGround` state rows must use canonical sorted IDs; (2) VGI's observation messages cannot use its reply-only `correlation_id` field. The IOS request identity is now retained as the receiver adapter's `TransportFrameReceipt.frame_id`, with independent ledger-record verification. These corrections are included in the passing source revision.

## Current verification cases

- Two distinct live peer TLS identities, all selected measurements, VGI strict two-source quorum and exactly reconstructed new native state.
- Conflicting peer bit readings reject a new processor state.
- Stale observer/old-state context is refused by IOS-003's installed Ground contract.
- A caller's exported receipt list cannot replace a live `MeasurementPeerClient`.
- Out-of-domain host readings are refused before the receiver admits them.
- Reconnection with the same challenge reuses IOS-003 durable responses without another source read.
- Fresh challenge nonces create new observations even when old native coordinates match.
- Independent verifier rejects rehashed wrong source, bit, challenge, VGI ledger or old state.

Run the dedicated [TDL-005 workflow](../.github/workflows/tdl005-authenticated-refinement.yml) (Linux/Windows, Python 3.11/3.13), or install the pinned E-CPU/NCR/VGI dependencies and execute:

```sh
python -m unittest discover -s tests -p 'test_authenticated_refinement.py' -v
python -m unittest discover -s tests -p 'test_fabric_refinement.py' -v
python -m unittest discover -s tests -p 'test_cost_aware_refinement.py' -v
```

## Translation discoveries and next construction

The observation question is now independently bound to **source-law identity**, **receiver intent**, **live transport peer identity**, **temporal challenge**, and **policy-qualified admission**. These are different representational coordinates; losing one is not equivalent to losing another.

Next: build a composition receipt that binds an IOS-003 transport-authenticated measurement to the concrete TDL-004 SAB route/task and receiver-accepted host identity, and then extend to truly independent processes or machines. To support portable verification of source authentication, use a separately admitted attestation/signature mechanism rather than claiming an unkeyed receipt digest proves it. A larger benchmark should compare independently administered mTLS peers, durable rehome after crash and per-peer cost under quorum, while preserving the source mathematical relation.

[Central research log — TDL-005](https://github.com/Kier73/In-Formation-Space/blob/main/reports/TRANSLATION_DISCOVERY_LOG_005.md).
