"""Repeat two published embedding observations, starting at their public directory pages."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys


# Independent byte pins from ADDRESSABLE_HOSTED_MODELS and the layer-0 router receipt.
CASES = (
    ("Qwen/Qwen3.8-27B", "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
     "model.language_model.embed_tokens.weight",
     "3b6eafbbea137d8ad9abd30e3e376bce9385fb23707ae82e1051920603801047"),
    ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "08c8657f02291e11b6f6b63e0e16014f8e748518",
     "embed.weight", "a38ebed9f015f0c3949cd2bcbc13224ac0d7eea82c8db33218e849f586880987"),
)


def observe(case, *, all_headers=False):
    """BF16 bytes of E[t] = S_j[h+lo+t·2d : h+lo+(t+1)·2d]."""
    from experiments.model_fabric.page_files import PageFiles
    from experiments.model_fabric.scaled_model_fabric import MAX_HEADER_BYTES, admit_index, admit_shard_header
    from experiments.model_fabric.local_seed import INDEX, INDEX_LIMIT

    model, revision, tensor, expected = case
    page = "https://huggingface.co/" + model + "/tree/" + revision
    with closing(PageFiles(page)) as files:
        index = files.metadata(INDEX, INDEX_LIMIT)
        inventory, mapping = admit_index(index, model=model, revision=revision)
        shard = mapping[tensor]
        grouped = defaultdict(set)
        for key, owner in mapping.items():
            grouped[owner].add(key)
        headers = {}
        for owner in sorted(grouped) if all_headers else [shard]:
            lead, size = files.read_range(owner, 0, 8)
            n = int.from_bytes(lead, "little")
            if not 2 <= n <= min(MAX_HEADER_BYTES, size - 8):
                raise ValueError("Invalid shard header extent")
            body, total = files.read_range(owner, 8, n)
            if total != size:
                raise ValueError("Inconsistent shard size")
            head, tensors = admit_shard_header(lead + body, filename=owner, file_size=size,
                                               expected_names=grouped[owner])
            payload = sum(d["data_offsets"][1] - d["data_offsets"][0] for d in tensors.values())
            if payload != size - n - 8:
                raise ValueError("Tensor intervals leave unaccounted bytes")
            headers[owner] = head
            if owner == shard:
                item = tensors[tensor]
        audited_bytes = sum(h["payload_span_bytes"] for h in headers.values())
        if all_headers and inventory["advertised_payload_bytes"] not in (None, audited_bytes):
            raise ValueError("Total payload differs from index")
        row = 63204
        rows, width = item["shape"]
        if item["dtype"] != "BF16" or row >= rows:
            raise ValueError("Embedding coordinate differs from baseline")
        start = item["header_bytes"] + item["data_offsets"][0] + row * width * 2
        raw, _ = files.read_range(shard, start, width * 2)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise ValueError(f"Embedding row diverged for {model}: {digest}")
        return dict(model=model, revision=revision, pages=files.visits, index_bytes=len(index),
                    index=inventory, audited_payload_bytes=audited_bytes, observed_headers=headers,
                    tensor=tensor, shard=shard, row=row, width=width, start=start,
                    trained_bytes=len(raw), sha256=digest, shard_http_bytes=files.transferred)


def main():
    """Accept the two-source observation iff ∀m, SHA256(row_m) = pin_m."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vl64", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all-headers", action="store_true", help="Verify every indexed tensor extent")
    args = parser.parse_args()
    root = args.vl64.resolve()
    sys.path[:0] = [str(root), str(root / "src")]
    with ThreadPoolExecutor(max_workers=len(CASES)) as pool:
        jobs = [pool.submit(observe, case, all_headers=args.all_headers) for case in CASES]
        results = [job.result() for job in jobs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
