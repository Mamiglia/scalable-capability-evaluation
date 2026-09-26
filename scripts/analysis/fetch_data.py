"""Download the published data the analysis reads, at the revisions in config.yaml (data_sources), and nothing else.

    uv run python scripts/analysis/fetch_data.py            # everything, about 40 GB
    uv run python scripts/analysis/fetch_data.py --dry-run  # how many files and GB per dataset, without downloading

Fetches into paths.data, paths.logs, paths.features and paths.embeddings:
    data        the frozen tables (meta/) and every transcript log they reference for tasks.domains (and
                coding_frontier, the closed-source targets)
    logs        the judge runs of those domains for tasks.all_semesters: pairwise (judge/, judge_channels-*) and
                absolute scoring (direct/, direct_reasoning-hidden/, direct_channels-*, direct_truncated-thirds-*)
    features    all of generality-labs/glm_features
    embeddings  the vectors of those transcript logs
A file already present with the published hash is skipped; every download is checked against it.
Requires read access to the generality-labs datasets (HF_TOKEN, or `hf auth login`).
"""
import argparse
import hashlib
import re
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from analysis.config import load_config

EXTRA_DOMAINS = ["coding_frontier"]  # the closed-source targets of the frontier figure


def published_files(repo: str, revision: str) -> list:
    """Every file of a dataset repository at a revision (path, size, and blob or LFS hash)."""
    tree = HfApi().list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
    return [f for f in tree if hasattr(f, "blob_id") and hasattr(f, "size")]


def transcript_logs(meta: Path, domains: list[str]) -> set[str]:
    """The transcript logs (paths relative to data/) the frozen tables of `domains` reference."""
    paths = set()
    for domain in domains:
        models = pd.read_csv(meta / domain / "models.csv")
        paths |= {p for c in models.columns if c.startswith("path_") for p in models[c].dropna()}
    return paths


def digest(path: Path, lfs: bool, size: int) -> str:
    """The file's hash as the Hub reports it: sha256 for LFS files, the git blob sha1 otherwise."""
    h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def published(path: Path, f) -> bool:
    """Whether the local file at `path` is the published file `f` (same size and hash)."""
    return path.exists() and path.stat().st_size == f.size and digest(path, f.lfs is not None, f.size) == want(f)


def want(f) -> str:
    """The published file's hash: sha256 for LFS files, the git blob sha1 otherwise."""
    return f.lfs.sha256 if f.lfs else f.blob_id


def fetch(repo: str, revision: str, root: Path, wanted, dry_run: bool) -> None:
    """Download the files of `repo` for which `wanted(path)` is true into `root`, skipping those already published.

    wanted: path within the repository -> whether the analysis reads it.
    """
    files = [f for f in published_files(repo, revision) if wanted(f.path)]
    todo = [f for f in files if not published(root / f.path, f)]
    print(f"{repo}: {len(files)} files ({sum(f.size for f in files) / 1e9:.1f} GB), "
          f"{len(todo)} to download ({sum(f.size for f in todo) / 1e9:.1f} GB)", flush=True)
    if dry_run:
        return
    for i, f in enumerate(todo, start=1):
        for attempt in range(5):  # the Hub rate-limits bursts of requests: back off and retry
            try:
                hf_hub_download(repo, f.path, repo_type="dataset", revision=revision, local_dir=root)
                break
            except Exception as error:
                print(f"retrying {f.path}: {error!r}"[:200], flush=True)
                time.sleep(60 * (attempt + 1))
        else:
            raise SystemExit(f"could not download {f.path}")
        if not published(root / f.path, f):
            raise SystemExit(f"{f.path}: hash differs from the published file")
        if i % 200 == 0:
            print(f"{repo}: {i}/{len(todo)}", flush=True)


def main(config: dict, dry_run: bool) -> None:
    sources, paths, tasks = config["data_sources"], config["paths"], config["tasks"]
    domains = tasks["domains"] + EXTRA_DOMAINS
    data = Path(paths["data"])
    fetch(sources["data"]["repo"], sources["data"]["revision"], data, lambda p: p.startswith("meta/"), dry_run)
    logs = transcript_logs(data / "meta", domains) if (data / "meta").is_dir() else set()
    fetch(sources["data"]["repo"], sources["data"]["revision"], data, logs.__contains__, dry_run)
    runs = r"judge|judge_channels-[^/]+|direct|direct_reasoning-hidden|direct_channels-[^/]+|direct_truncated-thirds-[^/]+"
    judged = re.compile(rf"^({runs})/({'|'.join(domains)})/({'|'.join(tasks['all_semesters'])})/")
    fetch(sources["logs"]["repo"], sources["logs"]["revision"], Path(paths["logs"]), lambda p: bool(judged.match(p)), dry_run)
    fetch(sources["features"]["repo"], sources["features"]["revision"], Path(paths["features"]), lambda p: True, dry_run)
    stems = {p.removesuffix(".eval") for p in logs}
    fetch(sources["embeddings"]["repo"], sources["embeddings"]["revision"], Path(paths["embeddings"]),
          lambda p: p.rsplit(".", 2)[0] in stems, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    parser.add_argument("--dry-run", action="store_true", help="count the files and bytes to download, download nothing")
    args = parser.parse_args()
    main(load_config(args.config), args.dry_run)
