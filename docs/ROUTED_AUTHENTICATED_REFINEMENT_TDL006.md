# TDL-006 — A routed observer must be the authenticated observer

**Implementation:** [`vli.routed_authenticated_refinement`](../src/vli/routed_authenticated_refinement.py), [independent verifier](../src/vli/routed_authenticated_verify.py), [live endpoint-binding integration tests](../tests/test_routed_authenticated_refinement.py), and [Linux/Windows verification](../.github/workflows/tdl006-routed-authenticated.yml).

The earlier rounds established separate executable contracts:
- [TDL-004](FABRIC_REFINEMENT_TDL004.md): receiver-owned E-CPU SAB-001 observation routes, typed operation and path authorization, and an exact cost-optimal mathematical basis.
- [TDL-005](AUTHENTICATED_REFINEMENT_TDL005.md): IOS-003 native mutual-TLS measurement over pinned authenticated peers, strict VGI source quorum, and durable native observation challenge/receipt semantics.

TDL-006 composes those contracts so a peer accepted by the routing plan must be **the same peer authenticated by the live measurement session**, for the same source-law epoch, observation, receiver policy and challenge.

## Formal invariant

For old/new observer state \(z_o=B_ox,z_n=B_nx\) and each offered source row \(a_i\), define a bound observer request for every required peer \(p\):

\[
r_{i,p} = \left(a_i,\ \mathrm{offer}_i,\ \mathrm{peerID}_p,\
\mathrm{expectedDNS}_p,\ \mathrm{intentionEpoch},\ \mathrm{challenge}\right).
\]

The route is not accepted on type compatibility alone. It must have **exactly one** realized operation matching the offer and peer, at the specific receiver-observed resource \(p\). The native IOS client must then pass E-CPU's pinned `_bound_transport` checks on a live `SSLSocket` whose CA-verified peer certificate contains the exact DNS SAN for \(p\).

\[
\boxed{
\mathrm{Admit}(r_{i,p})
\land\mathrm{RealizerLocus}(r_{i,p})=p
\land\mathrm{IOSProviderID}=p
\land\mathrm{TLSVerifiedSAN}=\mathrm{DNS}(p)
}
\tag{TDL6-1}
\]

The source row is eligible only if **every** selected pinned peer has an admissible dispatch route for that same mathematical question. Let \(q_{i,p}\) be the receiver-declared quanta of the full SAB route for \(i,p\). Then

\[
q_i^{\rm quorum}=\sum_{p\in P}q_{i,p},\qquad
S^*=\arg\min_{S\subseteq\mathcal A_{\rm fully\ admitted}}
\left(\sum_{i\in S}q_i^{\rm quorum},|S|,\operatorname{IDs}(S)\right)
\]

subject to

\[
\boxed{\operatorname{rowspan}(B_n)\subseteq\operatorname{rowspan}[B_o;A_S].}
\tag{TDL6-2}
\]

Thus a cheap measurement path that can reach only one required peer is **not** counted as an available quorum-qualified feature. The existing TDL-003 finite exact optimizer (up to 12 offered rows) and separate row/rank verifier are reused after deriving the effective all-peer route costs.

The bound cost model includes the sum of *declared receiver-local SAB route quanta* for all required peer dispatches, not measured network latency, TLS setup time, or physical energy. This selection assumes one independent dispatch route per peer/feature. Shared-edge scheduling, source correlation and variable quorum sizes remain separate optimizations.

## Exact execution order

```text
old/new verified SymbolicSource and PFI interpreters
  -> declared ObservationOffer rows and 2..8 PinnedPeer(id, expected DNS)
  -> derive epoch = H(source/interpreter Grounds, retained old native state,
                       receiver inventory+policy, coordinate, peer DNS set,
                       authority, scheduler, receiver challenge nonce)
  -> create BootstrapTask for every (offer, required peer)
     input kind = refinement.peer-dispatch.request.v1
     output kind = refinement.peer-dispatch.result.v1.<offer>.<peer>
     task payload binds row, old state, epoch, peer ID, expected DNS
  -> SAB-001 plan: available only if authorized route realizes exact operation
     at exact peer resource under current leases/policy/handler/carrier binding
  -> TDL-003 optimize complete source-row basis on sum(peer-route quanta)
  -> independently replan the COMPLETE eligible catalogue, reject omitted
     cheaper routes or mismatched rank/row/epoch
  -> require each live IOS-003 MeasurementPeerClient to have exactly the
     receiver-required provider ID and CA-verified TLS peer DNS SAN
  -> execute each admitted SAB dispatch operation, verify native SAB receipts
     and receiver-owned independent dispatch oracle
  -> verify receiver policy, inventory and all eligible routes still current
  -> IOS-003 live MeasurementPeerClient.resolve on SAME required peer set,
     exact selected row Ground, scheduler, challenge and host admission
  -> strict VGI quorum and explicit AUTHORIZED finalization for each row
  -> TDL-003 exact mathematical reconstruction of new native coordinate
  -> recheck admission epoch and independent SAB/IOS/VGI record consistency
```

**Trust semantics of the join:** SAB-001's locally bound carrier graph is a receiver-side *authorization and dispatch topology*. The E-CPU native IOS-003 socket carries the actual mutual-TLS measurement. This construction binds the **authorized endpoint identity** to the **TLS peer identity**. It does *not* establish that the receiver-local SAB channel graph describes the physical network route taken by the TLS packets.

## Python APIs

```python
from vli.routed_authenticated_refinement import (
    PinnedPeer, plan_routed_authenticated_refinement,
    execute_routed_authenticated_refinement,
)
from vli.routed_authenticated_verify import (
    verify_routed_authenticated_plan,
    verify_routed_authenticated_result,
)

# args: six verified PFI-001 source/quotient/interpreter Ground objects
# offers: tuple[ObservationOffer, ...] with source GF(2) row, operation cost, ID
# sab: receiver-owned E-CPU AdaptiveBootstrap with bound handlers/carriers
# clients: dict peer ID -> *live*, CA-verified IOS-003 MeasurementPeerClient
# authority_digest: current, receiver-defined source-authority digest
peers = (
    PinnedPeer("peer-left", "peer-left.fabric.test"),
    PinnedPeer("peer-right", "peer-right.fabric.test"),
)
acquisition = plan_routed_authenticated_refinement(
    *args, offers, sab,
    pinned_peers=peers, ingress="ingress", collector="collector",
    old_native_state=old_z, coordinate=3,
    authority_digest=authority_digest, scheduler=scheduler,
    challenge_nonce=fresh_nonce,  # 64 canonical hex characters
)
assert verify_routed_authenticated_plan(*args, acquisition, sab)
result = execute_routed_authenticated_refinement(
    *args, acquisition, sab, clients=clients,
)
assert verify_routed_authenticated_result(*args, acquisition, sab, result)
new_z = int(result["authenticated_record"]["refinement_receipt"]
            ["new_native_decimal"])
```

`result["dispatch_reports"]` are ordinary E-CPU SAB receipts from a **dispatch authorization** operator, not independently measured source bits. Only `result["authenticated_record"]` contains live IOS-003 results admitted under VGI. The independent result verifier rechecks the complete admitted offer catalogue, route digests, expected dispatch output, native IOS-002 request identities, peer VGI source membership, and reconstructed coordinates.

**Critical distinction:** an exported `result` passes a *record-consistency* verifier, not a portable historical proof of TLS authentication. The live acquisition procedure uses E-CPU's existing TLS certificate checks; offline portable source authenticity would require separately admitted signatures/attestations and preserved trust roots.

## Test fixture

A four-bit declared binary-affine Ground has old observer \(x_0\), new observer \((x_0,x_1)\), and candidate observations \(x_1\), \(x_1\oplus x_2\), and \(x_2\). The receiver-owned SAB inventory has two local observer-locus identities corresponding to two different native IOS-003 TLS peer identities, each provisioned with a distinct certificate. The right-hand admission route has an intermediate relay.

| Observation | SAB cost to left peer | SAB cost to right peer | All-peer declared cost |
|---|---:|---:|---:|
| \(x_1\) direct | 22 | 23 | 45 |
| \(x_1\oplus x_2\) | 3 | 4 | 7 |
| \(x_2\) | 3 | 4 | 7 |

The optimal complete basis takes both cheaper rows, total **14 declared quanta**, and reconstructs \(x_1\) by XOR after two-peer VGI quorum on each. A revoked peer-specific route for \(x_2\) removes that row from the available catalogue and selects the direct \(x_1\) row at cost 45. This is a bounded test Ground with receiver-local admission paths and live TLS sockets on the same test machine.

The [integration suite](../tests/test_routed_authenticated_refinement.py) also checks swapped TLS clients, mismatched pinned certificate DNS, stale inventory/leases, receiver-policy revocation, wrong dispatch oracle, disagreeing peer observations, durable replay on identical challenge, and altered/rehashed dispatch/IOS/VGI certificates. It reuses the real IOS-003 peer fixture built for TDL-005, not synthetic `ObserverReceipt` objects.

## Questions exposed by joining the protocols

1. **Transport-route equivalence:** the current theorem binds endpoints but not the physical path of packets. Build an instrumentation adapter that maps actual IOS-003 transport/channel IDs and bytes to a receiver-observed E-CPU route, then verify path constraints on wire evidence.
2. **Durable admission across process death:** IOS-003 can persist a read-only measurement completion; new work should store the receiver-side dispatch identity/epoch, record a partial VGI ledger, and resume only unfinished peer/row readings with fencing after a process restart.
3. **Quorum-aware routing beyond all-peers:** optimize a sufficient subset of distinct authenticated peers per observation while treating source correlation and unavailable/denied routes as formal constraints.
4. **Portable authenticated provenance:** a native TLS handshake is ephemeral. Source-signed measurement receipts with explicit trust roots, revocation evidence and context-bound nonce could support later offline independent qualification.
5. **Cost of network intelligence:** benchmark measured TLS requests, bytes, retries, wall-time, energy and independent host failure. Compare the exact quanta-optimal source-row basis against the measured fastest one.

[Central translation record: In-Formation-Space TDL-006](https://github.com/Kier73/In-Formation-Space/blob/main/reports/TRANSLATION_DISCOVERY_LOG_006.md).
