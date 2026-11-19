# VL-I-001 — First executable model Ground and observation-navigation experiment

**Implementation:** `src/vli/` · **Tests:** `tests/test_vli_001.py` · **CI:** `.github/workflows/vli-001.yml`  
**Source baseline:** `Kier73/VL-64` pinned at commit `9a08e7246ba4fb046270f88db2d38ab4c19368cb`  
**Model observed:** `HuggingFaceTB/SmolLM2-135M` at commit `4e53f736cbb20a9a0f56b4c4bf378d9f306ff915`

## What this actually introduces

1. **Model lifting:** `vli.ground.lift_observation` uses the qualified VL-64 typed-DAG compiler, independently checks its source identity, tensor extents and topological dependencies, then builds a content-digested VL-I Model Ground. The input is an observed pretrained **tensor header**, so this first stage has observed zero parameter payload bytes and **must not claim model inference**.
2. **Intent-frontier navigation:** `vli.ground.intention_frontier` walks the graph backward from a requested result to identify all required operations and original tensor ranges. A full-vocabulary logits intention needs every decoder stage. An intermediate hidden-state observation has a smaller dependency closure. This is exact graph reasoning; reported byte savings are **declared byte extents**, not measured network traffic unless the ranges are actually omitted by the realizer.
3. **Pretrained numerical realization:** `vli.realization.run_pinned_decoder` composes an existing VL-64 full 30-layer numerical run with an independently implemented NumPy reference and binds its model identity, actual byte-range counts, logits digest, chosen next token and checkpoint. No new VL-I neural implementation is invented; this first adapter **reuses the pinned VL-64 operators**, which compute actual trained-model mathematics locally.
4. **First constrained B3 observation relation:** `vli.navigation.commit_continuation` turns an already reference-checked *greedy next-token result for an exact model/input/precision combination* into an addressable structural memory entry. `navigate_continuation` returns the retained observation without another model forward pass or trained-weight acquisition. It refuses a changed input, model Ground or trusted record head.

This final case is exact **memoized observation**, not a novel law that predicts unseen requests. It illustrates the intended direction: meaningful computation can sometimes become state navigation by carrying and validating previously established structure. It also states the cold cost instead of pretending that cost vanished.

## Getting started

Clone both repositories, pin VL-64 exactly, and install the two packages:

~~~bash
git clone https://github.com/Kier73/VL-I.git
cd VL-I
git clone https://github.com/Kier73/VL-64.git upstream/vl64
git -C upstream/vl64 checkout 9a08e7246ba4fb046270f88db2d38ab4c19368cb
python -m pip install -e . -e upstream/vl64 numpy
~~~

Add `upstream/vl64` to `PYTHONPATH` (as the CI does). Run:

~~~bash
python -m unittest discover -s tests -v

python -m vli inspect \
  --manifest upstream/vl64/experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json \
  --out artifacts/pinned-model-ground.json

python -m vli inspect \
  --manifest upstream/vl64/experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json \
  --target layer.0.ffn_residual \
  --out artifacts/partial-frontier.json
~~~

`inspect` does not download trained parameter bytes.

A live model run requires qualified Hugging Face network access and approximately 269 MB of original pretrained weight data. The selected input is **provided token IDs** (42, 75), not demonstrated text tokenization:

~~~bash
python -m vli run \
  --manifest upstream/vl64/experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json \
  --tokens 42 75 \
  --out artifacts/vli-realization.json

python -m vli verify \
  --manifest upstream/vl64/experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json \
  --report artifacts/vli-realization.json

python -m vli remember \
  --manifest upstream/vl64/experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json \
  --report artifacts/vli-realization.json \
  --out artifacts/retained.json
~~~

`remember` emits an exact `VL_I_EXPECTED_OBSERVATION_HEAD`. Retain this head outside the cache and supply it to the navigation command:

~~~bash
python -m vli navigate \
  --manifest upstream/vl64/experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json \
  --retained artifacts/retained.json \
  --expected-head YOUR_TRUSTED_OBSERVATION_HEAD \
  --tokens 42 75
~~~

The SHA-256 head provides integrity relative to a trusted external pin; it is not a digital signature, independent source attestation or proof of all future outputs.

## Verification stages

| Question | Evidence tier | This increment |
| --- | --- | --- |
| Can a real pretrained header become a typed, integrity-bound VL-I Ground? | Structural, no weights | Implemented; GitHub Actions tests and lifts the observed 272-tensor header |
| Can an intention select only the graph dependencies that can affect an output? | Symbolic closure | Implemented and tested with negative controls |
| Can the *whole original pretrained model* run through the Virtual Layer's own numerical operators? | B2 native | Reuses VL-64 full-decoder implementation and independent NumPy oracle; VL-I must verify its own live run separately |
| Can VL-I answer a repeated exact next-token intention without rerunning that model? | B3 exact retained observation | Implemented and tested using mock evidence for the API; a live replay is included in the optional numerical CI job |
| Can VL-I navigate unseen trained-model outputs more efficiently than conventional inference? | B3 generalization | **Open research** — not established by caching an already observed result |
| Can VL-I consume native chat templates and run multi-token paragraph/code generation? | Behavioral realization | **Open**, targeted next |
| Can all of this remain equivalent across large dense/MoE/hybrid checkpoint sizes and observer-hosts? | Generality + recovery | **Open**, target retained at full model scale |

For historical evidence, the pinned VL-64 source contains a complete 30-layer pretrained decoder record and independent layer comparisons. Those upstream receipts are not silently promoted into verified new VL-I numerical outcomes.

## Next experiment: exact causal state reuse, then predictive law discovery

The narrow exact-input cache offers a useful first contrast:

**Full model path:** obtain all required parameter ranges, evaluate all typed operations, independently verify next-token output, commit a causal checkpoint.

**Retained result path:** navigate directly to a previously checked token observation for the identical input/model state/precision and externally pinned evidence head.

The second path can avoid all *new model computation* for this specific read, but its cold origin and storage/verification costs must be included. Next, extend from exact-output memoization to reusable **causal prefix/KV states**, qualified operator subgraph equivalences and genuine predictive descriptions that work across previously unobserved continuations. Each new relation must demonstrate held-out parity or report a counterexample.

No goal is narrowed by this first experiment. The purpose is to establish a hard, reproducible point of departure for deeper navigation.
