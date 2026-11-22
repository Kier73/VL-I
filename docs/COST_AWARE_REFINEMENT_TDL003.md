# TDL-003 — Cost-aware observer acquisition and VGI-finalized processor refinement

**Continuation:** [TDL-004](FABRIC_REFINEMENT_TDL004.md) now composes this mathematical selector with receiver-admitted E-CPU host/channel route planning, actual handler/carrier execution and VGI-qualified reports. Observation-specific output kinds preserve the intended mathematical measurement through the typed path.

**Implementation:** [`vli.cost_aware_refinement`](../src/vli/cost_aware_refinement.py), [independent verifier](../src/vli/cost_aware_refinement_verify.py), [adversarial integration tests](../tests/test_cost_aware_refinement.py).

TDL-002 established the smallest additional **number of linear bits** needed when an intention changes. TDL-003 selects **which offered observations to acquire**, minimizing their declared cost and accepting values only from VGI's existing finalized evidence ledger.

## Current mathematical contract

Let the prior processor have \(z_o=B_ox\); a new intention requests \(z_n=B_nx\). Each `ObservationOffer` declares a nonzero GF(2) row \(a_i\), positive integer acquisition cost \(c_i\), a stable offer ID and descriptive observer locus. The caller declares a finite list \(\mathcal A\) with at most 12 offers.

The implemented optimizer solves **exactly** over this bounded catalogue:

\[
\boxed{S^*=\arg\min_{S\subseteq\mathcal A}
\left(\sum_{i\in S}c_i,\ |S|,\ \operatorname{IDs}(S)\right)}
\]

subject to the exact source-space condition

\[
\operatorname{rowspan}(B_n)
\subseteq\operatorname{rowspan}
\left(\begin{bmatrix}B_o\\a_{i_1}\\\vdots\\a_{i_k}\end{bmatrix}\right).
\]

The score is compared lexicographically: least declared total cost, then fewest measurements, then stable offer-ID order. Bounded exhaustive enumeration uses up to \(2^{12}=4096\) subsets; the verifier uses a separate combinations-based enumeration and checks source-space rank and every reconstruction row.

The minimum number of auxiliary bits without offer constraints remains

\[
d=\operatorname{rank}([B_o;B_n])-\operatorname{rank}(B_o).
\]

**Translation finding:** \(d\) and the cost-optimal count \(k\) can differ. If \(B_o=(x_0)\), the new observer additionally wants \(x_1\), and available offers are \((x_1,\$10)\), \((x_1\oplus x_2,\$1)\), \((x_2,\$1)\), the cheapest solution costs **2 cost units** and makes **two measurements**, even though \(d=1\). Combining their values by XOR reconstructs the missing bit.

The costs are caller-declared abstract units; they are not benchmarked network latency or independently measured prices. The observer-locus field is a declared offer label, not authorization to execute at that host.

## Actual selection API

```python
from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from vli.problem_formed_interpreter import construct
from vli.cost_aware_refinement import (
    ObservationOffer, plan_cost_aware_observations, observation_context,
)

identity = tuple(1 << i for i in range(4))
generators = (AffineMap(identity, 0),)
old = SymbolicSource(4, generators, (1,), "tdl003-declared")
new = SymbolicSource(4, generators, (1, 2), "tdl003-declared")
old_q, old_bundle = construct(old)
new_q, new_bundle = construct(new)
args = (old, old_q, old_bundle, new, new_q, new_bundle)
offers = (
    ObservationOffer("direct", 2, 10, "observer-A"),
    ObservationOffer("cheap-left", 6, 1, "observer-B"),
    ObservationOffer("cheap-right", 4, 1, "observer-C"),
)
plan = plan_cost_aware_observations(*args, offers)
assert plan["minimum_extra_rank"] == 1
assert plan["selected_offer_ids"] == ["cheap-left", "cheap-right"]
assert plan["total_cost"] == 2 and plan["observation_count"] == 2
old_z = old_q.project(0b0111)
context = observation_context(plan, old_native_state=old_z)
```

## VGI observation authorization and reconstruction order

The plan is independent of whether an observation has already occurred. The receiver constructs a `TransitionEvidencePolicy` for `GROUND_ID="pfi-refinement-observation"` and `context_digest=observation_context(plan, old_native_state=z_o)`; its strict source/quorum membership is supplied by the receiver. Each selected offer ID becomes the action of a VGI transition event from `SOURCE_STATE="refinement"` to itself, whose single output lies in \(\{0,1\}\).

The actual pipeline is:

```text
verified PFI old/new Grounds
    -> bounded exact cheapest-offer search + independent cost/row audit
    -> observation_context(plan_digest, old_native_state)
    -> receiver-owned VGI ChannelMessage -> EncounterLedger + TransportFrameReceipt
    -> TransitionEvidenceLedger.ingest_channel_message
    -> raw QUORUM / PENDING / CONFLICT
    -> explicit TransitionEvidenceLedger.finalize for each chosen offer
    -> AUTHORIZED bit only (other statuses refuse)
    -> form w from the k observed bits
    -> z_new = L(old_native_state, w) from exact plan source-row mapping
    -> receipt binds plan, context, VGI policy/ledger digests,
       per-offer decision digests and evidence IDs, final native coordinate
    -> independent verify_cost_plan + verify_vgi_refinement_receipt
```

Current execution syntax after a receiver has ingested real VGI `ChannelMessage` values through matching `TransportFrameReceipt` and `EncounterLedger` objects:

```python
from vli.cost_aware_refinement import acquire_vgi_refinement
from vli.cost_aware_refinement_verify import verify_vgi_refinement_receipt

# ledger is the receiver's finalized TransitionEvidenceLedger with
# policy Ground/context matching the context created above.
receipt = acquire_vgi_refinement(
    *args, plan, old_native_state=old_z, ledger=ledger,
)
assert verify_vgi_refinement_receipt(*args, plan, receipt, ledger=ledger)
```

For the full channel/transport/encounter construction, see [`observed_ledger`](../tests/test_cost_aware_refinement.py) and its test cases. That fixture exercises the exact library calls instead of constructing synthetic “AUTHORIZED” dictionaries. The raw evidence event and its finalization remain inspectable in the VGI ledger. `verify_vgi_refinement_receipt` rechecks them against the plan and canonical context.

## Verification and scope

| Case | Mechanism and evidence target |
|---|---|
| Exact cost optimality | Exhaustive 12-offer finite search, independently enumerated combinations, rank and decode check |
| Two cheap offers beat one direct offer | Candidate rows outside the target observer span, with cost 2 vs cost 10 |
| VGI positive path | Real ChannelExchangeLedger, EncounterLedger, TransportFrameReceipt, two-source quorum, and explicit finalization |
| Uncertain/discordant observations | Missing, unknown, unfinalized and dissenting evidence cannot issue a new native-state receipt |
| Context and provenance | Different old-native coordinate, plan, source Ground or policy digest fails |
| Tampering | Rehashed wrong bit, mapping, rank, price, selected offers, source identity or result refused |
| Zero missing information | Empty offer set and zero-cost direct mathematical projection; no implied evidence claim |

The VGI transport identity is adapter-supplied; its source IDs, frame receipts and policy quorum are evidence and integrity contracts, not independent cryptographic source authentication. The current processor-state reconstruction has **mathematical equality conditional on the supplied, policy-admitted measurement bits**. E-CPU admission of physical observation hosts is a separate execution-layer integration, not replaced by the offered `locus` label.

Run the dedicated [TDL-003 GitHub Actions workflow](../.github/workflows/tdl003-cost-aware-refinement.yml) across Linux/Windows and Python 3.11/3.13, or locally with the pinned E-CPU/NCR/VGI packages installed:

```sh
python -m unittest discover -s tests -p 'test_cost_aware_refinement.py' -v
python -m unittest discover -s tests -p 'test_minimal_refinement.py' -v
python -m unittest discover -s tests -p 'test_translation_commutation.py' -v
```

## Development opportunities revealed by translation

1. **Optimize observation acquisition in actual E-CPU host terms.** Map each declared offer to a receiver-owned computational/observer capability, enforce authenticated host identity and allowed path with `AdaptiveBootstrap`, then replace abstract cost with separately measured resource/transport/authorization costs.
2. **Minimize total cost under quorum.** The current `cost` is supplied per completed observation offer. Once actual observers have distinct prices/reliabilities, joint selection of rows, hosts, source quorum and routes becomes a constrained optimization problem.
3. **Adaptive observation.** When an offer returns unknown/conflict, preserve the VGI ledger and recompute a valid remaining plan under a new context/revision, without treating missing information as a zero bit. VLD TRACE-002's rank-gain inquiry can suggest the next informative query.
4. **Information versus energy.** Test whether a two-measurement cheaper basis actually reduces end-to-end time/bytes/energy relative to one direct observation, with exact receipts. A smaller bit count is not automatically a smaller physical cost.
5. **Cross-representation acquisition.** Verify finite E-CPU quotient transitions before and after a VGI-qualified intention change, using TDL-002's commutative-diagram tests as the independent semantic oracle.

The continuing research record is [TDL-003 in In-Formation-Space](https://github.com/Kier73/In-Formation-Space/blob/main/reports/TRANSLATION_DISCOVERY_LOG_003.md).
