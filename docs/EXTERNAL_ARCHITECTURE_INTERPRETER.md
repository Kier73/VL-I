# One interpreter, external model data

The [convolutional vision experiment](EXTERNAL_VISION.md) extends the original text build below to ResNet-18. The expanded interpreter uses the same source interface for all three architectures.

A single 20,443-byte VL-64 executable ran the trained DistilGPT2 decoder and MiniLM BERT encoder. It contains 42,771 bytes of expanded Python and runs with the Python standard library. The model weights stay on a filesystem or at a public host between requests. During execution the VL reads a bounded interval, applies the operator locally and releases that interval. It retains architecture metadata, current activations and temporary arithmetic buffers. Each tensor request is at most 4 MiB. The executable size and the range limit measure different things, and neither bounds total working memory. [Standalone executable](https://github.com/Kier73/VL-64/blob/main/artifacts/VL64-external.pyz)

| Trained checkpoint | Observation | Bytes of weights read per run | Maximum error against Transformers in binary64 |
| --- | --- | ---: | ---: |
| DistilGPT2 | All 50,257 next-token logits after three input IDs | 324,523,008 | 7.11 × 10⁻¹⁴ |
| MiniLM-L6-v2 | Four contextual vectors plus BERT's pooler output, 1,920 coordinates | 43,195,392 | 3.11 × 10⁻¹⁵ |

For each checkpoint the same archive produced byte-identical output files from a pinned Hugging Face page and a local copy. The GPT-2 and BERT references ran separately. GPT-2 selected token 13. BERT used one unpadded segment with all positions visible. Its encoder and pooler observations precede the sentence-transformers pooling and normalization pipeline. The MiniLM output also matched through a loopback HTTP directory serving ordinary byte ranges with strong ETags. [Decoder receipt](../experiments/model_pages/2026-11-23-external-distilgpt2.json) · [Encoder receipt](../experiments/model_pages/2026-11-23-external-minilm.json)

## What generalizes

The model adapter owns the equations, including attention direction, normalization placement, trained bias, matrix orientation and readout. The source adapter owns metadata and the operation `read_range(shard,start,length) → (bytes,file_size)`. A model adapter can run over any provider implementing that contract. The built-in providers cover local Safetensors directories, Hugging Face model pages and versioned HTTP(S) directories with exact ranges and strong shard ETags. A custom provider can call `run(files, ids)` directly. A generic HTTP root should identify one immutable release across all shards. A per-file ETag protects that file across reads.

These two models share dot products and embeddings. GPT-2 uses causal attention, pre-normalization and input-major trained projections. BERT uses bidirectional attention, post-normalization and output-major projections. ResNet-18 adds spatial convolution, pooling and inference-time batch normalization. Architecture support comes from binding each configuration to the correct equations and validating the requested outputs.

## Builds in related repositories

- **Virtual-Ground-Interpreter** already has typed BERT operator contracts for bidirectional attention, padding-mask semantics, first-token selection, tanh and erf-GELU. Its eight focused contract tests passed independently. This is a candidate source of admission checks when extending VL-I to padded or multi-segment encoder requests. The present numerical run uses an unpadded segment. [VGI BERT contracts and substitution tests](https://github.com/Kier73/Virtual-Ground-Interpreter/blob/ae3f949b7d0b2aaa42236bc7cc37e9857384b174/virtual_ground_interpreter/tests/test_bert_operator_contracts.py)

- **Virtual-Layer-Dynamics** implements error propagation and a logit-margin test for selective refinement. VL-I now records an empirical version in the decoder reference check. For two complete measured vectors $z,\hat z$, the selected token's lead in the reference is at least the VL top-two gap minus $2\|z-\hat z\|_\infty$. On the tested prefix this lower bound was 0.23760945. Using this to skip reads on future prefixes would require bounds available before the reference calculation. [Precision certificate implementation](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/afe54804bc15ea14b1c51766757edef3aac96817/src/vld/models/precision_cert.py)
- **In-Formation-Space** partitions a GPT-2 feed-forward block by hidden neurons. Each partition can independently emit its contribution to the output. For this interpreter the adaptation would send workers the checkpoint identity and partition coordinates, let each read its own intervals, and verify the sum against the unpartitioned run. Charge shared-input traffic and repeated source reads when testing the benefit. [Tensor-cut implementation](https://github.com/Kier73/In-Formation-Space/blob/e0a0089bae2789b1a7fc769284396658eb38f0a4/experiments/tensor_cut_realization/tensor_cut.py)
- **Navigable-Compute** closes a set of requested observations under state transitions on finite Boolean systems. That gives a criterion for merging classifier states in VL-I. If two contexts receive one state identifier, every admitted continuation must preserve the requested observation. Extending that result to floating-point trained models remains an open research step. [Observer closure experiment](https://github.com/Kier73/Navigable-Compute/blob/66e216e53ab606547ba4a5316ed44044e327b67b/docs/NCR-007.md)

## Run from VL-I

Supply the [VL-64 archive](https://github.com/Kier73/VL-64/blob/main/artifacts/VL64-external.pyz) and an output path. With VL-I installed, the example below computes the complete next-token logit vector. Substitute `--snapshot DIRECTORY` or `--url VERSIONED_DIRECTORY_URL` to change storage. The receipt reports bytes and requests. Repeated invocations reread the required weights.

```sh
vli external --seed VL64-external.pyz --page https://huggingface.co/distilbert/distilgpt2/tree/2290a62682d06624634c1f46a6ad5be0f47f38aa --tokens 15496 11 995 --out logits.json
```
