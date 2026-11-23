# PBG-003 — Intention, observation, material operation, observed result

**Executable implementation:** [goal_evidence_fabric.py](../src/vli/goal_evidence_fabric.py), [integration tests](../tests/test_goal_evidence_fabric.py), and [cross-platform CI](../.github/workflows/pbg003-observed-intention.yml).

This continues [PBG-002](ADAPTIVE_INTENTION_PBG002.md) from virtual action selection to actual receiver-owned E-CPU SAB-001 routes and VGI finalized evidence. The material state is a local receiver-held simulated fixture; the mathematical and operational mechanisms are real library calls, while independent physical-source authority remains the next experiment.

## Three different states of an intention

1. **Predicted effect.** The PBG-002 Ground computes the probability that an intervention realizes the desired outcome.
2. **Accepted action.** E-CPU invokes the specifically named receiver-owned handler and rechecks its result under a local operation oracle.
3. **Observed realization.** A *subsequent* measurement of the goal event passes VGI strict two-source finalization. Only that observation can change the Virtual Ground from predicted success to an observed success or failure.

The distinction is mathematical. For a source-global goal-relative predictive basis B, qualified intervention A and sensor-outcome S_y satisfy

\[
AB=BK_A,\quad S_yB=BK_y,\quad z=\alpha B.
\]

A verified action receipt authorizes the **predicted** state transition \(z^-=zK_A\). The following sensor outcome y is admitted through VGI; only then is the conditioned state

\[
\boxed{p_y=(z^-K_y)_0,\qquad z^+=z^-K_y/p_y,\quad p_y>0}
\]

constructed. Neither the action receipt nor a prior model prediction substitutes for the post-effect observation.

## Concrete E-CPU and VGI composition

```text
Current IntentionSession.snapshot and exact source Ground
  → prepare: bind nonce, previous state, inventory, policy and coordinate
  → one typed BootstrapTask per named receiver observer/actuator host
  → SAB-001.plan: local handler/carrier/permission/lease intersection
  → reject substituting another operation or a generic output representation
  → SAB-001.execute with a receiver-owned independent local oracle
  → verify native BootstrapReport source identity and step receipt chain
  → sensing: VGI ChannelMessage → EncounterLedger → TransportFrameReceipt
  → strict two-source policy → explicit finalize → AUTHORIZED only
  → independently bind VGI payload/frame/ground fingerprint to each E-CPU report
  → independently replay the candidate transition using only the predictive Ground
  → commit the state after record verification
  → when testing an intervention, independently sense its goal result afterward
```

The VGI observation event includes the native E-CPU report digest as its `ground_fingerprint`. The verifier reconstructs the complete channel message and frame identity from the native output; a rehashed outer record alone cannot bypass that cross-evidence check. The current source IDs denote separate receiver-provisioned local handlers, not independently administered physical sensors.

### Direct API

```python
from vli.goal_evidence_fabric import (
    Endpoint, prepare, execute, verify_material_outcome,
)
from vli.intention_predictive_ground import IntentionSession

session = IntentionSession(compiled_ground)
observers = (Endpoint("observer-A", "observer-A"),
             Endpoint("observer-B", "observer-B"))
actuator = (Endpoint("actuator-A", "actuator-A"),)

sensor_plan = prepare(session, receiver_runtime, "sense", "read-type",
                      observers, nonce="1"*64, coordinate=2)
sensor_record = execute(sensor_plan, session, receiver_runtime)

action_name = session.plan(horizon=1).name
action_plan = prepare(session, receiver_runtime, "act", action_name,
                      actuator, nonce="2"*64, coordinate=2)
action_record = execute(action_plan, session, receiver_runtime)

goal_plan = prepare(session, receiver_runtime, "sense", "read-goal",
                    observers, nonce="3"*64, coordinate=2)
goal_record = execute(goal_plan, session, receiver_runtime)

outcome = verify_material_outcome(
    compiled_ground, action_plan, goal_plan, receiver_runtime,
    action_record, goal_record)
```

The receiver must explicitly provision an `AdaptiveBootstrap` instance with `RealizerBinding`, `CarrierBinding`, leases, policy and independent handler-oracle callbacks. The full executable fixture constructs this typed runtime.

## Test result and mathematical checkpoints

The test source contains eight material states (two types × two success flags × two redundant tags). The intention forms a four-coordinate Ground. Adding a deterministic post-action goal sensor preserves this dimension.

| Phase | Native execution/evidence | Goal probability held by the Ground |
| --- | --- | ---: |
| Declare intention | Source-qualified predictive Ground | 1/5 |
| Observe type | Two E-CPU host routes and strict VGI evidence reporting type 1 | 1/5 |
| Actuate chosen intervention | E-CPU `apply-1` handler with local receiver oracle | 3/4 (predicted) |
| Measure result | New E-CPU `read-goal` routes and strict VGI quorum reports goal 1 | 1 (observed) |

A separate failure scenario has a misleading but possible type-reading event, the same predicted 3/4 success, and a post-action outcome **0**. The retained Ground then conditions its goal probability to **0**, correctly distinguishing prediction from observation.

Each of two sensor passes uses two three-step E-CPU paths, and the action uses one three-step path: **15 declared route-quanta** across five task executions. These are declared path costs, not wall-clock latency.

The seven integration tests cover the successful complete cycle, disagreeing sensor values, real effect failure, snapshot restoration before intervention, modified/rehashed receipts, policy revocation, and a forged causal connection between an action and a goal reading.

## Continuity, authority and cost

- A sensor conflict, missing permission, stale Ground, changed host policy or wrong output identity refuses an interpreter state update.
- A restored PBG-002 session can resume after an admitted sensor event without restoring the original eight-state distribution.
- A single host-local actuator in the fixture caches a nonce and returns its previous result if the same request is repeated. This is **memory-local idempotency**, not crash-durable exactly-once actuation. The next host-side build must journal action intent before execution, fence stale incarnations and reconcile ambiguous in-flight effects before retry.
- Two named observation sources are distinct software bindings, and their results must agree. Their shared underlying simulated world is intentionally not claimed to supply independently authenticated physical measurement.
- `verify_bootstrap_report` verifies native receipt consistency and records receiver oracle checks; neither an unkeyed SHA digest nor the offline verifier proves historical remote TLS authentication or physical truth.

## Reproduce

The [dedicated workflow](../.github/workflows/pbg003-observed-intention.yml) installs pinned predecessors on Linux and Windows with Python 3.11 and 3.14. It runs the native bridge and PBG-001/PBG-002 and TDL-004 regressions.

```sh
python -m unittest discover -s tests -p test_goal_evidence_fabric.py -v
python -m pytest -q tests/test_intention_predictive_ground.py tests/test_predictive_basis_ground.py
```

## Next experimental frontier

Connect the observation-specific operation to E-CPU's existing live mTLS peer identities (TDL-005/006), then bind a durable actuator commit to an independently qualified material sensor. Use the resulting discrepancies to refine a source-derived predictive Ground, not merely to replay a known finite simulation. Finally, apply that evidence-driven intention loop to pretrained language-model observation and tool/use transitions: model source data become a readable Ground, and the observer's desired outcome determines which source relations to interpret.
