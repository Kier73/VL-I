# How much trained behavior survives a change of architecture?

The target is a trained model's response to a new input. In the book analogy, the architecture supplies a grammar, the weights supply this particular story, and the observer asks something precise about it. A useful Virtual Layer would find the pages that settle the question and know when it has read enough. A model's name helps locate the book. The test of navigation is whether the observed weights let the Layer reproduce the answer and omit work.

## The public survey

The survey queried **all 811 configuration keys** in one pinned Transformers registry, plus all 52 tasks and 55 library labels returned by the Hub taxonomy. Each query considered up to four repositories ranked by downloads. A registry hit required an immutable checkpoint descriptor whose root `model_type` matched the registry key. Task and library hits required the corresponding Hub listing field and a readable pinned descriptor. These three counts overlap. A task, library, subconfiguration, or particular checkpoint is a different unit from an architecture. [Pinned registry](https://github.com/huggingface/transformers/blob/be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef/src/transformers/models/auto/auto_mappings.py) · [Hub taxonomy](https://huggingface.co/api/models-tags-by-type)

| Survey axis | Queried | Confirmed | Unresolved |
| --- | ---: | ---: | ---: |
| Registry keys | 811 | 626 | 185 |
| Task labels | 52 | 44 | 8 |
| Library labels | 55 | 26 | 29 |

The confirmed rows cover **673 distinct repository/revision pairs**. Of the 185 unresolved registry keys, 158 returned empty listings and 27 had candidates that failed qualification. Some keys describe components inside composite models. Other public projects distribute GGUF, ONNX, custom code, or library-specific manifests. This survey reads root JSON descriptors and records these gaps. Registry coverage measures discoverability under that rule. It costs zero weight payload bytes and supplies no inference result. [Full survey, including every unsuccessful query](../experiments/architecture_survey/2026-11-22-public-descriptions.json)

## A common experiment on trained weights

Across ten pinned checkpoints, the intended observation is which of two output IDs has the higher score. The IDs are fixed in advance within each modality. Their meanings can differ between tokenizers. The language probes use two short, reversed token-ID sequences. BERT receives a mask, T5 receives both encoder and decoder input, SmolVLM receives text alone, and ViT receives two deterministic image tensors. These inputs exercise numerical behavior rather than task accuracy.

An affine output head has rows $w_i,w_j$ and biases $b_i,b_j$. Given the state $z$ entering it, compile the two trained rows into $a=(w_i-w_j,b_i-b_j)$ and append a constant to obtain $z'=(z,1)$. Rank coordinates once by decreasing $|a_k|$. For visited coordinates $K$, the actual score gap lies in

$$
[S-R,S+R],\qquad S=\sum_{k\in K}a_kz'_k,\qquad
R^2=\left(\sum_{k\notin K}a_k^2\right)\left(\sum_{k\notin K}{z'_k}^2\right).
$$

Cauchy–Schwarz gives the bound. When $S^2>R^2$ the sign is settled. Otherwise the interpreter reads another coordinate product, down to exact equality if necessary. The implementation converts the observed dyadic floats to scaled integers and compares squares. This certifies the real affine score of those represented parameters and activations. The independently executed CPU FP32 model is a separate numerical comparison, particularly important near ties.

In the candidate pass a hook intercepts the trained head input, records the Virtual Layer's decision and stops before the head executes. A second pass through the original model computes every output score. The conventional model still supplies the candidate's upstream state. Its full state vector is read once to establish the residual energy bound. The reported coordinate count measures visited score terms, and excludes that preliminary scan. No end-to-end speedup is established. [Trained weights, per-position outcomes and visited-term counts](../experiments/architecture_survey/2026-11-22-trained-readouts.json)

The panel agreed with the unmodified models on **110 of 110 decisions**. The first ID won 50 times and the second 60 times. BERT, OPT, ViT, GPT-2, SmolLM, Qwen3, SmolVLM, Mamba, Flan-T5 and RWKV all reached the same affine observation boundary. Their internal transition laws differ. Across all trials the certificate visited 70,749 of 79,982 possible coordinate products, or 88.5%. Every trial stopped early. The tightest case still visited 767 of 769 coordinates, which makes that input a useful challenge for a cheaper certificate.

## Reproduction

From the VL-I root, run the two commands below. The survey uses the Python standard library. The numerical panel was run with Python 3.13, PyTorch 2.7.0 CPU, Transformers 4.51.3, Hugging Face Hub and Safetensors. Add `--acquire` to the second command to populate the Hub cache from its exact source revisions. The panel otherwise runs offline, with two model workers by default. RWKV uses the public Safetensors conversion revision recorded in the panel.

~~~sh
python -m examples.architecture_survey --out artifacts/public-descriptions.json
python -m examples.multi_model_readout --out artifacts/trained-readouts.json
~~~

The numerical experiment belongs in VL-I's research layer. The existing VL-64 model seed has a stronger, narrower demonstration of full generation on SmolLM. To move the many-model result toward the same goal, the next boundary to test is the operation that produces $z$. An affine residual can carry the comparison backward by a transpose. A nonlinear or recurrent transition needs its own bound and must still preserve the causal state required for the next question.
