"""Download only pinned HF judgment files absent from an existing mirror."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo", default="LASR-G5/judging")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    token = os.environ["HF_TOKEN"]
    tree = HfApi(token=token).list_repo_tree(
        args.repo, repo_type="dataset", revision=args.revision, recursive=True)
    remote = {item.path for item in tree
              if getattr(item, "path", "").endswith(".eval")
              and item.path.startswith(("pilot/gpqa/", "pilot/swe_bench_mini/"))}
    local = {str(path.relative_to(args.mirror)) for path in args.mirror.glob("pilot/**/*.eval")
             if not path.name.startswith("._")}
    missing = sorted(remote - local)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pinned_paths.txt").write_text("\n".join(sorted(remote)) + "\n")

    def fetch(filename):
        return hf_hub_download(args.repo, filename, repo_type="dataset", token=token,
                               revision=args.revision, local_dir=args.out)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(fetch, missing))
    absent = [filename for filename in missing if not (args.out / filename).exists()]
    if absent:
        raise RuntimeError(f"{len(absent)} downloads missing after completion")
    print(f"downloaded {len(paths)} pinned files to {args.out}")


if __name__ == "__main__":
    main()
