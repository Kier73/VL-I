# QUALIFIED-009 — Independent original-weight PyTorch mathematical parity

**Research question:** Does the original pretrained Qwen3.8-27B gated attention
operator that VL-I navigates actually compute the same mathematical result
as a separately implemented PyTorch reference on the *same original trained
hidden state*?

This follows the verified architecture-directed eight-layer trained execution
of QUALIFIED-008, not merely an independent fixture or a random input vector.

## Experiment design

1. Obtain the original immutable Qwen model weight index, config and
   tokenizer at revision
   `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
2. Re-admit the model's source-indexed 64-layer architectural plan with
   `QwenArchitectureCausalPrefix`. Read the same original paragraph and
   train-derived token `63204`.
3. Execute **all seven complete trained original layers 0–6**, including
   the preceding Gated DeltaNet and attention operators. Reject unless
   each F64 hidden-output digest exactly reproduces the independent
   historical QUALIFIED-008 trained numerical observations.
4. Copy the actual layer-6 5,120-wide hidden vector into two distinct,
   independently implemented whole-layer computations at layer 7:
   - VL-I's original `QwenQualifiedGatedAttention`, whose dense source
     projections run NumPy F64 with source-verified range receipts;
   - `OriginalQwenTorchGQA`, which *does not call VL-I's operators*.
     It independently decodes original BF16/F16/F32 source bytes, evaluates
     matrix/vector products via CPU PyTorch, applies separately coded
     zero-centered RMSNorm, per-head Q/K RMSNorm, rotary embeddings,
     causally masked grouped-key/value softmax, learned sigmoid gating,
     original full output projection, original complete SwiGLU, and both
     residual additions.
5. Compare every original hidden output coordinate. The criterion is
   `max_abs <= 1e-5` **and** `max_scaled <= 2e-7` where the
   scaled denominator is `max(1, abs(ref), abs(vli))`. Report error
   metrics, trained input/output SHA digests, original source traffic,
   exact revision/plan authority and the distinct PyTorch backend.
   Reject any change in the existing trained layer-7 digest before
   reporting parity.

The qualified navigator and exact original pretrained source bytes are
*shared input evidence*. **The mathematical operator implementations
are independent.** This is stronger than comparing two instances of the
same VL-I executor; unlike a second model checkpoint, no new source
identity is invented.

## Independent fixtures and failure tests

`tests/test_qwen_torch_oracle.py` runs multistep original-topology
attention fixtures against the existing VL-I operator, including causal
KV history, simulated source departure after completing the attentional
part of a layer, altered source plans, and actual fully executed
seven-layer intermediate fixture states. A failed source range must not
promote any PyTorch reference causal state.

`tests/test_saved_original_qwen_first_eight.py` archives all eight
original trained source digests from GitHub Actions
[44668103895](https://github.com/Kier73/VL-I/actions/runs/44668103895)
and tests exact correspondence with prior QUALIFIED-007 results.
The source evidence also includes 825 qualified source HTTPS 206 requests
and 6,918,617,224 transferred bytes for the eight-layer computation.

Run locally with a CPU installation of PyTorch and the pinned upstream
VL-64 experimental package:

```bash
PYTHONPATH=upstream/vl64:upstream/vl64/tests:tests \
  python -m pytest -q tests/test_qwen_torch_oracle.py \
       tests/test_saved_original_qwen_first_eight.py

PYTHONPATH=upstream/vl64:upstream/vl64/tests:tests \
  python -m examples.qwen_independent_torch_parity \
  --layer-budget-bytes 2000000000 \
  --out artifacts/qualified-009-original-torch-parity.json
```

The original checkpoint comparison is **not accepted until its separate
GitHub Actions experiment succeeds** and the explicit numerical-error
thresholds pass. Failures are useful evidence: a mismatch may localize
the discrepancy to source decoding, PyTorch precision, rotary conventions,
grouped attention order, gate placement, or the trained MLP.

## Interpretation of success

This compares independent PyTorch **float64 mathematical expressions** on
the actual original trained matrices. The official Hugging Face full model
typically uses reduced-precision kernels, casting, fusion and runtime
choices; exact BF16 roundoff equivalence is a separate experiment. Nor
does the PyTorch reference prove network-native arithmetic or that a
physical model-worker is absent.

Neither experiment executes all 64 decoder layers, the final learned
normalization/vocabulary projection, or generated language. VL-64 remains
unchanged.

**Motivation:** before searching for faster virtual or observer-aware
relations, establish which detailed trained transformations are actually
equivalent to an independent mathematical reference. Only then can a
new navigation or quotient law claim preserved trained behavior.
