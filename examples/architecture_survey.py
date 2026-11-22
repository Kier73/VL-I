"""Discover pinned public model descriptions across registry families, Hub tasks and libraries."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


REGISTRY = "be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef"
ROOT = "https://huggingface.co"
RATE_LOCK = threading.Lock()
NEXT_REQUEST = {}


def fetch(url):
    """Bound |response| to 16 MiB and retry transient responses with a bounded backoff."""
    for attempt in range(3):
        # All workers share the provider's cooldown t_next=max(t_next,t_now+Retry-After).
        bucket = "listing" if "/api/" in url else "descriptor"
        while True:
            with RATE_LOCK:
                delay = NEXT_REQUEST.get(bucket, 0) - time.monotonic()
            if delay <= 0:
                break
            time.sleep(delay)
        try:
            with urlopen(Request(url, headers={"User-Agent": "VL-I-public-architecture-survey/1"}), timeout=20) as response:
                data = response.read(16 * 1024**2 + 1)
            if len(data) > 16 * 1024**2:
                raise ValueError("metadata byte limit")
            return data
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            retry = exc.headers.get("Retry-After", "")
            if exc.code == 429:
                cooldown = int(retry) if retry.isdecimal() else 300
                with RATE_LOCK:
                    NEXT_REQUEST[bucket] = max(NEXT_REQUEST.get(bucket, 0), time.monotonic()+cooldown+1)
            else:
                time.sleep(2**attempt)
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def registry_keys(source):
    """Extract keys K={k:(k,c)∈CONFIG_MAPPING_NAMES} from the pinned literal without executing it."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CONFIG_MAPPING_NAMES" for t in node.targets):
            return sorted(dict(ast.literal_eval(node.value.args[0])))
    raise ValueError("registry layout changed")


def inspect_candidate(model, revision):
    """Bind D to (repository, commit, H(config bytes)); count only declarations present in D."""
    if not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", model) or not re.fullmatch("[a-f0-9]{40}", revision):
        raise ValueError("invalid source coordinate")
    for name in ("config.json", "model_index.json", "adapter_config.json"):
        url = f"{ROOT}/{quote(model)}/resolve/{revision}/{name}"
        try:
            raw = fetch(url)
        except HTTPError as error:
            if error.code == 404:
                continue
            raise
        doc = json.loads(raw)
        if not isinstance(doc, dict):
            raise ValueError("descriptor is not an object")
        classes = doc.get("architectures") or []
        return {"model": model, "revision": revision, "descriptor": name,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "model_type": doc.get("model_type"), "architectures": classes,
                "pipeline_class": doc.get("_class_name"), "adapter_type": doc.get("peft_type")}
    raise ValueError("no supported root descriptor")


def survey(key):
    """For each descriptor class k, inspect up to four ranked candidates until one satisfies k."""
    axis, value = key
    row = {"axis": axis, "key": value, "status": "unresolved", "attempts": []}
    params = [("filter", value), ("sort", "downloads"), ("direction", "-1"), ("limit", 4),
              ("expand", "sha"), ("expand", "config"), ("expand", "library_name"), ("expand", "pipeline_tag")]
    try:
        candidates = json.loads(fetch(ROOT + "/api/models?" + urlencode(params)))
        if not isinstance(candidates, list):
            raise ValueError("invalid listing")
        row["listing_count"] = len(candidates)
        if axis == "registry":
            candidates.sort(key=lambda c: (c.get("config") or {}).get("model_type") != value)
        for candidate in candidates:
            model, revision = candidate["id"], candidate["sha"]
            attempt = {"model": model, "revision": revision}
            row["attempts"].append(attempt)
            try:
                obs = inspect_candidate(model, revision)
                matches = (obs["model_type"] == value if axis == "registry"
                           else candidate.get("pipeline_tag" if axis == "task" else "library_name") == value)
                attempt["observed_type"] = obs["model_type"]
                if matches:
                    row.update(status="observed", observation=obs)
                    break
                attempt["error"] = "candidate declaration differs"
            except (OSError, ValueError, KeyError) as error:
                attempt["error"] = str(error)
    except (OSError, ValueError, KeyError) as error:
        row["error"] = str(error)
    return row


def main():
    """Survey K=K_registry ∪ K_tasks ∪ K_libraries with axis labels and bounded concurrency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--registry-revision", default=REGISTRY, help="40-hex Transformers Git commit")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--resume", action="store_true", help="Retry rate-limited rows in this registry report")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.registry_revision):
        parser.error("registry revision must be a full Git commit")
    url = f"https://raw.githubusercontent.com/huggingface/transformers/{args.registry_revision}/src/transformers/models/auto/auto_mappings.py"
    registry = fetch(url)
    tags = json.loads(fetch(ROOT + "/api/models-tags-by-type"))
    keys = [("registry", k) for k in registry_keys(registry)]
    for axis, tag in (("task", "pipeline_tag"), ("library", "library")):
        keys.extend((axis, k) for k in sorted({v["id"] for v in tags[tag]}))
    retained = {}
    if args.resume:
        prior = json.loads(args.out.read_bytes())
        if prior["registry_sha256"] != hashlib.sha256(registry).hexdigest():
            raise ValueError("registry differs from prior report")
        retained = {(r["axis"],r["key"]): r for r in prior["rows"] if "HTTP Error 429" not in json.dumps(r)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        missing = [key for key in keys if key not in retained]
        retained.update(zip(missing, pool.map(survey, missing)))
    rows = [retained[key] for key in keys]
    counts = {axis: {s: sum(r["axis"] == axis and r["status"] == s for r in rows)
                     for s in ("observed", "unresolved")} for axis in ("registry", "task", "library")}
    report = {"registry_url": url, "registry_sha256": hashlib.sha256(registry).hexdigest(),
              "counts": counts, "rows": rows, "weight_bytes_read": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
