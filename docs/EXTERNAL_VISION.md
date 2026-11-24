# A convolutional network from external weights

The same VL-64 archive that interprets GPT-2 and BERT can classify a 224 × 224 RGB crop with the trained Microsoft ResNet-18 checkpoint. The archive is 21,926 bytes and its expanded Python is 46,896 bytes. Running the earlier GPT-2 and BERT local examples with this archive reproduced their recorded output hashes. The image path uses the standard library and the existing filesystem, model-page and HTTP directory readers. [Interpreter archive](https://github.com/Kier73/VL-64/blob/main/artifacts/VL64-external.pyz) · [Pinned checkpoint](https://huggingface.co/microsoft/resnet-18/tree/65a5785d9156231087c481e0c7dd33a5ff6f7e3e)

For a convolution kernel stored as `[output_channel,input_channel,height,width]`, the existing range reader can fetch output filters as rows by flattening the trailing dimensions. Each filter is applied across spatial positions. For stride $s$ and padding $p$, the pre-normalization output is

$$y_{o,u,v}=\sum_{c,i,j}W_{o,c,i,j}\,x_{c,su+i-p,sv+j-p}.$$

The seed interprets the stem, all eight residual blocks, the projection shortcuts at resolution changes, global average pooling and the trained 1,000-way readout. It uses zero padding for convolution and negative infinity for max pooling. Learned batch-normalization parameters and running statistics are read in inference mode. The source serves bytes. The seed performs the calculations and keeps intermediate activations and spatial windows in working memory.

## What was tested

The reference harness decoded a public dog photograph and used the checkpoint's image processor for resizing and cropping. The isolated seed received interleaved RGB8 pixels, read the checkpoint's channel normalization, and computed every class score. The reference independently ran the pretrained Transformers model in binary64 on the same normalized crop. The original crop and a second crop with both spatial axes reversed were each evaluated from a local model directory and the pinned model page. [Original photograph](https://raw.githubusercontent.com/pytorch/hub/c3beaae7d32fca2a23fec30aa7938ef5c9b6e5d5/images/dog.jpg) · [Reference harness](../examples/external_vision_trial.py)

The upright crop selected class 258, Samoyed. Turning it upside down changed the model's decision to class 279, Arctic fox. The VL reproduced both decisions. Its largest absolute logit discrepancies were $1.07\times10^{-14}$ upright and $1.16\times10^{-14}$ rotated. For each crop the page and filesystem runs returned byte-identical output. Each run read 46,796,448 bytes of trained operands in 109 reads, plus metadata. The largest weight read was 4,193,280 bytes. These measurements establish numerical agreement and transfer cost. Peak working memory and end-to-end latency remain unmeasured. [Upright receipt](../experiments/model_pages/2026-11-23-resnet18.json) · [Rotated receipt](../experiments/model_pages/2026-11-23-resnet18-rotated.json)

## Run

Pass a 150,528-byte file containing the already cropped pixels in row-major RGB order. The image option and token option are mutually exclusive. The same command accepts `--snapshot DIRECTORY` in place of `--page URL` for a local checkpoint. The returned JSON file contains logits in checkpoint label order.

```sh
vli external --seed VL64-external.pyz --page https://huggingface.co/microsoft/resnet-18/tree/65a5785d9156231087c481e0c7dd33a5ff6f7e3e --rgb224 crop.rgb --out scores.json
```

To reproduce the crop and the numerical checks directly from the photograph, use the source checkout, a local copy of the pinned model (including `preprocessor_config.json`), Pillow, NumPy and Transformers with PyTorch. Repeat with `--rotate-crop` for the spatial control.

```sh
python -m examples.external_vision_trial --seed VL64-external.pyz --snapshot MODEL_DIRECTORY --page https://huggingface.co/microsoft/resnet-18/tree/65a5785d9156231087c481e0c7dd33a5ff6f7e3e --image dog.jpg --out trial.json
```
