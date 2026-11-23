"""Find model cards by free text and inspect their pinned configurations concurrently."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from urllib.parse import quote


def survey(query, limit):
    """For the first k displayed links m ∈ search(q), resolve (commit_m, config_m)."""
    from experiments.model_fabric.page_files import Page, PageFiles, SITE
    from experiments.model_fabric.local_seed import metadata_object

    class Results(Page):
        def handle_starttag(self, tag, attrs):
            """Bind rank i to the i-th displayed model under the submitted query q."""
            super().handle_starttag(tag, attrs)
            attrs = dict(attrs)
            if attrs.get("data-target") == "ModelList":
                listing = json.loads(attrs["data-props"])["initialValues"]
                if listing["searchQuery"] != query:
                    raise ValueError("Search response differs from submitted query")
                self.models = [entry["id"] for entry in listing["models"]]

    visits = []
    page = Results(SITE + "/models?search=" + quote(query, safe=""), visits)
    if not hasattr(page, "models"):
        raise ValueError("No model search results component")

    def inspect(model):
        """Resolve the structural observation C_m = JSON(bytes(m,r,config.json))."""
        try:
            with PageFiles(page.require("/" + model)) as source:
                config = metadata_object(source.metadata("config.json", 1 << 20))
                return dict(model=model, revision=source.revision, model_type=config.get("model_type"),
                            architectures=config.get("architectures"), pages=source.visits,
                            status="observed_config", trained_bytes=0)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            return dict(model=model, status="unresolved", reason=str(exc))

    with ThreadPoolExecutor(max_workers=min(limit, 4)) as pool:
        observations = list(pool.map(inspect, page.models[:limit]))
    return dict(query=query, search_page=visits[0], displayed=len(page.models), observations=observations)


def main():
    """Emit k≤32 ranked observations. A network or format failure remains an explicit outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vl64", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=4, choices=range(1, 33))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.vl64.resolve()
    sys.path[:0] = [str(root), str(root / "src")]
    result = survey(args.query, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return int(not any(row["status"] == "observed_config" for row in result["observations"]))


if __name__ == "__main__":
    raise SystemExit(main())
