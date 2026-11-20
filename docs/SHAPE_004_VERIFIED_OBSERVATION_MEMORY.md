# SHAPE-004 — Verified structural memory instead of repeated source transitions

**Research question:** Can a Virtual Layer observer-host retain the *causal positions of already observed information* so repeated model acquisition doesn't pay to replay the same source transitions?  
**Implementation:** [observation_memory.py](../src/vli/observation_memory.py)  
**IFS integration:** [learned_morphology.py](../src/vli/learned_morphology.py)  
**Test:** [test_observation_memory.py](../tests/test_observation_memory.py)  
**Executable benchmark:** [observation_memory_demo.py](../examples/observation_memory_demo.py)  
**CI:** [vli-shape004.yml](../.github/workflows/vli-shape004.yml)

## 1. The observation is a state position, not just a value

SHAPE-003 reconstructed observable neural shape from IFS COG-008 black-box membership histories, but it also measured heavy repetition. A learner asking for \`advance, advance, inspect\` after previously observing \`advance, advance\` should not, where the source contract allows it, have to recompute the first two transitions.

SHAPE-004 retains a **source-owned structural memory** whose keys are causal *action-history prefixes*. Each prefix identifies the original source's actual hidden state position and the emitted observation associated with the last admitted transition. These hidden states remain internal to the receiver. The IFS learner continues to see **only its authorized projected action/output observations** and does not receive the original state or internal memory.

\[
\text{observed prefix }(a_1,\dots,a_k)
\mapsto (s_k,\lambda_k)
\]

A new query reuses already-qualified prefixes and executes only the as-yet-unobserved suffix. This is navigation to retained computational states rather than repetition of the original transitions.

### The safety/semantics contract

- **Pure deterministic Ground required.** The original VLD \`GroundTransitionSystem\` is complete and has no external effects. Arbitrary tool calls, HTTP requests, physical sensors and effectful programs are **not** eligible for this shortcut without new capability, replay and effect-authority proofs.
- **Exact source identity.** The structural memory is admitted under the source bytes and original VLD model fingerprint. Changing the source invalidates retained state snapshots.
- **Observer isolation.** The receiver may reuse a hidden state position across observers, but each must receive only the requested qualified channel projection, not all output channels.
- **Independent recovery.** A memory checkpoint is digest-bound to an externally trusted expected head. During restoration it checks **every cached edge** against the original VLD source reference, refusing forged state transitions, altered observations, missing parent prefixes and incompatible model epochs.
- **Honest cost accounting.** Checkpoint validation itself pays one original transition per retained edge. A digest without independent source verification is not claimed to prove the causal correctness of a restored prefix.
- **Unchanged IFS learning.** The exact same observation-only \`ObservedOracle\`, counterexample revision, learned model, ledger and generated VLD neural parameters must result with and without prefix memory.

## 2. Actual verified benchmark

The [GitHub Actions SHAPE-004 run 44648975538](https://github.com/Kier73/VL-I/actions/runs/44648975538) passed **9 focused tests** against pinned original IFS, VLD and E-CPU. It ran the same three full SHAPE-003 learning tasks both with and without receiver-owned structural memory.

| Measurement | Uncached source replay | Shared structural prefix memory |
| --- | ---: | ---: |
| Phase learning / underlying original transitions | 91 | **37** |
| Microstate learning / underlying original transitions | 138 | **11** |
| Full observer learning / underlying original transitions | 838 | **88** |
| **All three tasks combined** | **1,067** | **136** |
| Membership queries | 180 | 180 |
| Original IFS observer evidence ledgers | Baseline | **Exactly identical** |
| Independently qualified learned models | 3 / 4 / 12 states | **Exactly identical** |
| Generated VLD neural parameters | Baseline | **Exactly identical** |

**Source transition work reduction: 7.85×, avoiding 931 original transition evaluations.**

The cache retained **137 prefix state positions**, including its genesis. A persisted snapshot occupied approximately **19 KB** of JSON and required **136 original transition checks** to independently validate on restore. This is a deliberate memory/computation and repeated-observation tradeoff, not free compression.

### Does it improve actual end-to-end wall time?

**Not yet on this local Python example.** After correcting a first-run initialization confound, the experiment ran **six alternating-order warm trials**, including all source parsing, IFS observation learning, source proof and generated neural compilation for the three observer models.

| Warm acquisition + model generation (median) | Timing |
| --- | ---: |
| Uncached baseline | **34.78 ms** |
| Prefix-memory version | **37.05 ms** |
| Baseline / cached ratio | **0.94×** |

The cache was approximately **6.5% slower** in this warm, local implementation despite greatly reducing actual source transitions. Python dictionary/path management, one additional memory-owned source parser, and short inexpensive local VLD transitions outweigh the saved arithmetic for this workload. This is a negative performance result for total local latency, and it is preserved explicitly.

In contrast, when a transition needs actual remote communication, large neural computation or costly causal interpretation, avoiding 931 repeated transitions may be valuable. **That benefit has not been measured on a network, a pretrained model or a real communication fabric yet.**

The corrected experimental benchmark runs both uncached and cached paths in alternating order with warm imports, rather than attributing the first Python import to one candidate and claiming a fictional speedup.

## 3. What is actually demonstrated

- Observer-specific *source state positions* can be retained internally while the public source interface remains ordinary black-box observation.
- **Multiple learned neural Grounds reuse the same original hidden source prefixes**, across phase, microstate and full observers; they retain distinct qualified external observations.
- IFS's source evidence ledger, original hypothesis, counterexample history and source-free restoration are unchanged.
- VLD generates **identical neural weights** and the same novel code/text/token response capability from the learned Grounds.
- A receiver can recover the cached state-prefix topology from a source-bound checkpoint, reverify every original edge and refuse tampering.
- The actual original-source work decreases, although local total interpreter time does not yet decrease.

The distinction is important: **lower work in one computational relation is not automatically lower total system cost**. The interpreter, memory, verifier and observer-host must be designed together.

## 4. Run it directly

~~~bash
git clone https://github.com/Kier73/VL-I.git
cd VL-I
git clone https://github.com/Kier73/In-Formation-Space.git upstream/ifs
git clone https://github.com/Kier73/Virtual-Layer-Dynamics.git upstream/vld
git clone https://github.com/Kier73/E-CPU.git upstream/ecpu
git -C upstream/ifs checkout 0c712a83a33d14a0de630e26facd74cb487e102e
git -C upstream/vld checkout 985ba5c2fbcc40842aed9f45f42587749620b98e
git -C upstream/ecpu checkout f9ed67c495b00f9d26df47e291b40b41070cb609
python -m pip install -e . -e upstream/vld -e upstream/ecpu
export PYTHONPATH=upstream/ifs/experiments/codonic-capability-atlas
python -m unittest discover -s tests -p test_observation_memory.py -v
python -m examples.observation_memory_demo \
  --out artifacts/shape004-observation-memory.json
~~~

## 5. Next: make the saved work matter

The next direct experiment should connect this structural memory to a **more expensive source observer-host**, such as E-CPU's existing verified distributed data/compute paths, VL-I INTERACT-002's full original pretrained model, or actual external code execution. Count network bytes, source transitions, message round trips, memory retention and cross-host provenance. A receiver must not reuse effectful operations merely because their input prefix looks familiar.

Then apply [VGI partial attention](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/main/virtual_ground_interpreter/vgi/partial_attention.py) and [IFS partial-attention navigation](https://github.com/Kier73/In-Formation-Space/blob/main/bridges/observation-path/partial_attention_navigator.py) to choose **which unseen prefixes are worth observing** before building a model. A cheaper observation set is potentially more powerful than memoizing a costly, poorly selected observation set.

The larger goal is still the same: an interpreter whose internal operations increasingly reduce to **navigation among previously verified and newly generated informational relations**, with intelligence-shaped Grounds formed and revised around actual observer intentions.

**This first direct improvement saves real source transitions. It does not yet prove that VL-I is faster than conventional model execution.**
