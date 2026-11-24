"""Check all ImageNet scores from a public photograph over local and remote model storage."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification

from .external_source_trial import invoke


def main():
    """Compare F_file(x)=F_page(x) and ||F_file−F_reference||∞≤10⁻⁸ on the same RGB crop."""
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("seed","snapshot","image","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--page",required=True)
    p.add_argument("--rotate-crop",action="store_true",help="Reverse both spatial axes after cropping")
    a = p.parse_args()
    torch.set_num_threads(1)
    processor = AutoImageProcessor.from_pretrained(a.snapshot,local_files_only=True,use_fast=False)
    with Image.open(a.image) as image:
        rgb = processor(image.convert("RGB"),do_normalize=False,do_rescale=False,
                        return_tensors="np")["pixel_values"][0]
    if rgb.shape != (3,224,224) or rgb.dtype != np.uint8:
        raise ValueError("Expected a 224×224 RGB8 crop")
    if a.rotate_crop:
        rgb = rgb[:,::-1,::-1]
    report = dict(page_url=a.page,image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),
                  seed_sha256=hashlib.sha256(a.seed.read_bytes()).hexdigest(),seed=invoke(a.seed,["seed-verify"]))
    report["crop_rotation_degrees"] = 180 if a.rotate_crop else 0
    with TemporaryDirectory() as tmp:
        pixel_path = Path(tmp)/"crop.rgb"
        pixel_path.write_bytes(rgb.transpose(1,2,0).tobytes())
        outputs = []
        for mode,source in (("snapshot",a.snapshot),("page",a.page)):
            dest = Path(tmp)/(mode+".json")
            report[mode] = invoke(a.seed,["--"+mode,str(source),"--rgb224",str(pixel_path),"--out",str(dest)])
            outputs.append(dest.read_bytes())
    report["identical_outputs"] = outputs[0]==outputs[1]
    values = torch.tensor(json.loads(outputs[0])["logits"],dtype=torch.float64)
    model = ResNetForImageClassification.from_pretrained(a.snapshot,local_files_only=True).double().eval()
    mean = torch.tensor(processor.image_mean,dtype=torch.float64)[:,None,None]
    std = torch.tensor(processor.image_std,dtype=torch.float64)[:,None,None]
    pixels = (torch.from_numpy(rgb.copy()).double()/255.-mean)/std
    with torch.inference_mode():
        ref = model(pixels[None]).logits[0]
    if values.shape != ref.shape:
        raise ValueError("Classifier dimensions disagree")
    error = (values-ref).abs().max().item()
    index = values.argmax().item()
    report.update(coordinates=len(values),max_abs_error=error,tolerance=1e-8,
                  vl_top1=index,reference_top1=ref.argmax().item(),label=model.config.id2label[index])
    report["passed"] = report["identical_outputs"] and index==report["reference_top1"] and error<=report["tolerance"]
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report))
    return int(not report["passed"])


if __name__=="__main__":
    raise SystemExit(main())
