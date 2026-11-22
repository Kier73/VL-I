# TDL-004 — Intention-formed host routes for qualified observation

**Current implementation:** a declared observer intention forms a finite set of typed measurement requests; E-CPU SAB-001 admits receiver-owned handlers and observed carrier paths under lease, policy and operation identity; VL-I selects the cheapest information-sufficient combination of **actually plannable** paths; E-CPU executes each selected operation; its receipts become receiver-attributed VGI evidence; only finalized VGI outcomes reconstruct the next processor coordinate.

Source: [`vli.fabric_refinement`](../src/vli/fabric_refinement.py), [independent audit](../src/vli/fabric_refinement_verify.py), [multi-hop integration tests](../tests/test_fabric_refinement.py), [four-job workflow](../.github/workflows/tdl004-fabric-refinement.yml). Previous implementation: [TDL-003 cost-aware observation](COST_AWARE_REFINEMENT_TDL003.md).

## Mathematical contract: admissible source-row information and route cost

For existing and new observer quotient bases \(B_o,B_n\), the intent is still

\[
z_o=B_ox,\quad z_n=B_nx.
\]

Each requested observation \(i\) supplies an exact declared covector \(a_i\in\mathbb F_2^n\) and a designated operation/locus. Its E-CPU task \(T_i\) is a typed request from ingress \(s\) to receiving collector \(t\), requiring a receiver-owned local operation binding and carrier path. Let \(\mathcal P_i(I,P,\tau)\) be the SAB-001 route plans admitted by current inventory \(I\), policy \(P\), and observation coordinate \(\tau\), including all leases and local handler/carrier availability.

For each request, SAB-001 finds the least declared-quanta path \(p_i^*\). A request is available precisely when the route is admitted **and executes its exact identified measurement operation at the declared locus**, with a typed path ending in its observation-specific result representation.

\[
q_i=\mathrm{Quanta}(p_i^*),\quad
\mathcal A_{\rm available}=\{i:p_i^*\text{ exists and matches }(a_i,\operatorname{locus}_i)\}.
\]

The TDL-003 exact-cost solver then minimizes

\[
\boxed{
S^*=\arg\min_{S\subseteq\mathcal A_{\rm available}}
\left(\sum_{i\in S}q_i,\ |S|,\mathrm{IDs}(S)\right)
}
\]

subject to

\[
\operatorname{rowspan}(B_n)\subseteq\operatorname{rowspan}
\begin{bmatrix}B_o\\A_S\end{bmatrix}.
\]

This is exact within the declared finite offer catalogue (at most 12 offers), bounded SAB-001 route graphs, and one independent route per selected offer. **Route costs remain observed-inventory/operation-declared quanta**; actual elapsed local wall time and application payload bytes are recorded separately, not substituted for declared quanta.

## A new invariant discovered by execution: types must express the *question*

The first test showed a subtle mismatch: several different parity measurements were advertised with the same general request/output types. SAB-001 correctly chose the cheapest authorized operation that converted those types—even when its mathematical row was not the row requested by the intention.

Representations therefore must identify *which mathematical observation* the returned bit represents:

```python
from vli.fabric_refinement import offer_operation, offer_output_kind

operation = offer_operation("cheap-left")
# "refinement.observe.cheap-left.v1"

output_type = offer_output_kind("cheap-left")
# "refinement.observation.v1.cheap-left"
```

The receiver's `OperationOffer` declares `accepts="refinement.request.v1"` and `produces=offer_output_kind(offer_id)`. The collector-facing `ChannelObservation` carries the same qualified type. The task explicitly targets that type. The independent verifier reconstructs all candidate tasks, re-plans all admissible paths, checks the complete available offer catalogue, exact operation locus and declared cost, and refuses an invented path or an omitted cheaper candidate.

The binding is still to **declared source row + specific receiver operation**. Independently authenticating the *physical source's actual state* requires a separate remote sensor/attestation mechanism.

## Executable formation, path execution, and evidence finalization

```text
verified PFI old + new source/interpreter Grounds
  -> old-native-state + current inventory/policy/coordinate digest
  -> each candidate source row becomes typed E-CPU BootstrapTask
  -> SAB-001.plan: observed resource offers ∩ receiver policy ∩ local
     handler/carrier bindings ∩ lease validity ∩ path budgets
  -> accept route only if it executes exactly the candidate operation/locus
     and returns its observation-specific output representation
  -> exact TDL-003 rank/cost plan chooses a complete set of routes
  -> SAB-001.execute revalidates each route and invokes bound handler/carrier
  -> verify_bootstrap_report checks typed receipt chain and receiver oracle
  -> local monotonic wall-clock + application payload byte observations
  -> receiver adapter builds a VGI ChannelExchangeLedger, EncounterLedger,
     TransportFrameReceipt, and TransitionEvidenceLedger
  -> explicitly finalize each selected offer event; require AUTHORIZED
  -> independently verify VGI ledger, SAB reports and their bit correspondence
  -> TDL-003 reconstruction computes z_new from z_old and qualified bits
  -> digest-bound fabric realization receipt
```

`FabricRefinement` binds the whole offered catalogue, the chosen path tasks and exact SAB plans. Its epoch digest binds source/interpreter Ground, old native state, inventory and policy fingerprint, and observation coordinate. Re-planning after a lease expiry or policy revision creates a different route identity while preserving the mathematical intention and output where sufficient alternative information exists.

### Native Python call path

```python
from vli.fabric_refinement import (
    plan_fabric_refinement, execute_fabric_refinement,
)
from vli.fabric_refinement_verify import verify_fabric_refinement

# args: the six admitted old/new source, quotient and interpreter Grounds
# offers: tuple[ObservationOffer, ...] with row/cost/locus identifiers
# runtime: AdaptiveBootstrap(inventory, policy, local_handlers, local_carriers)
# old_z: retained prior processor state
acquisition = plan_fabric_refinement(
    *args, offers, runtime,
    ingress="start", collector="collector",
    old_native_state=old_z, coordinate=3,
)
result = execute_fabric_refinement(*args, acquisition, runtime)
assert verify_fabric_refinement(*args, acquisition, runtime, result)
new_z = int(result["refinement_receipt"]["new_native_decimal"])
```

For fully executable construction of `SubstrateObservation`, `OperationOffer`, `BootstrapPolicy`, `RealizerBinding`, `CarrierBinding`, the exact channel graph and VGI admission, see [`tests/test_fabric_refinement.py`](../tests/test_fabric_refinement.py).

### Current experimental network

A four-bit declared source has old observer \(x_0\), new observer \((x_0,x_1)\), and three offered measurements:

| Source-row observation | Operation estimated quanta | SAB-001 admitted route | Route total quanta |
|---|---:|---|---:|
| \(x_1\) (direct) | 10 | ingress → direct → collector | 12 |
| \(x_1\oplus x_2\) (cheap-left) | 1 | ingress → cheap-left → collector | 3 |
| \(x_2\) (cheap-right) | 1 | ingress → relay → cheap-right → collector | 4 |

The information-sufficient route selection uses **cheap-left + cheap-right** for 7 declared quanta instead of the direct route's 12. The outputs' XOR reconstructs the missing \(x_1\) after VGI finalization. In the fixture, expiring the cheap-right resource causes a new plan to select the direct route, preserving the required observation for the same source state. A receiver policy revocation invalidates the old plan.

This is a **receiver-local observed-and-bound multi-hop graph**. Channel drivers in the test use actual E-CPU's carrier byte API with a locally bound byte-identity driver; they are not a proof of geographically separate hosts, mTLS network transport, or remote sensor signature validity.

## Evidence, interpretation, and verification

| Evidence layer | Current independently checked property | Future construction |
|---|---|---|
| Source mathematical sufficiency | TDL-003 independently verifies rank, selected rows and exact map | Nonlinear conditional quotient selection |
| Receiver admission | Actual SAB-001 policy, lease, handler and carrier checks; the full alternative catalogue is re-derived | Authenticated cross-process mTLS host identity |
| Path execution | SAB execution step receipts, typed result and independent receiver-local oracle | Independently controlled observer source and verified physical measurement |
| Observation policy | One receiver-adapter VGI strict policy source; actual VGI event finalizations, digest/lineage | Multi-source independent sensor quorum per requested row |
| Measurement | Locally measured `perf_counter_ns` duration and canonical application payload bytes | Host energy, network bytes and repeated latency distributions |
| Continuity | Form new host/measurement path on lease expiry, preserving new native result | Durable across-process rehome/fencing and recovery of in-flight measurement |

The VGI policy source is **a single receiver-owned adapter**, not a federation of independently authenticated sensors. This distinction is retained in `result["authority_scope"]` and each SAB report's `physical_independence_attested=False`. The physical-source bit is qualified under the declared receiver oracle and VGI local policy, not an external scientific observation theorem. Likewise, a SHA-256 content digest supplies integrity identity rather than proof of who originated the bytes.

The dedicated CI runs the new seven-case integration suite, TDL-003 seven-case regression suite, and TDL-002 seven-case regression suite on Linux/Windows Python 3.11 and 3.13. Results are tracked in the [workflow](../.github/workflows/tdl004-fabric-refinement.yml).

## Next discoveries to test

1. **Authentication as typed evidence:** use E-CPU's mutual-TLS transit fabric and independent host identity to produce the observation frame consumed by the receiver adapter; bind VGI's source membership to authenticated endpoint identity instead of adapter attribution.
2. **Quorum and path costs jointly:** plan at least two independent authenticated observer-hosts for each source row under strict VGI quorum, measure duplicated requests, and account for source correlation/authority.
3. **Moving intention epochs:** allow context-safe renegotiation of the chosen row basis after one host fails between planning and execution; preserve unfinished VGI events as pending rather than turning absence into a bit value.
4. **Information cost versus execution cost:** measure end-to-end elapsed time, payload bytes, authorization overhead and physical energy across repeated runs; determine when a higher number of cheaper observations is actually faster.
5. **Transport semantics and source-law truth:** distinguish a receiver-verified handler computing parity of a declared state from a separately observed external system, and qualify that connection through source/ground witness receipts.

Research translation: [In-Formation-Space TDL-004](https://github.com/Kier73/In-Formation-Space/blob/main/reports/TRANSLATION_DISCOVERY_LOG_004.md).
