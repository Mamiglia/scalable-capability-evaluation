"""
Downloads the published data the analysis reads, at the revisions in config.yaml (data_sources), and nothing else.

    uv run python scripts/analysis/fetch_data.py            # everything, about 40 GB
    uv run python scripts/analysis/fetch_data.py --dry-run  # how many files and GB per dataset, without downloading

Fetches into paths.data and paths.logs:
    data        generality-labs/benchmarks: the frozen tables (meta/) and every transcript log they reference for
                tasks.domains (and coding_frontier, the closed-source targets)
    logs        generality-labs/protocol-outputs: the judge runs of those domains for tasks.all_semesters (pairwise/ and
                absolute/, every view), all of features/, and the embeddings/ vectors of those transcript logs
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

EXTRA_DOMAINS = ["coding_frontier"]  # the closed-source Coding targets of generalisation.py's frontier analysis


def published_files(repo: str, revision: str) -> list:
    """
    Lists every file of a Hugging Face dataset repository at one revision.

    Inputs:
        repo: the dataset repository, e.g. generality-labs/benchmarks.
        revision: the commit to list.
    Outputs:
        list of the Hub's file entries (folders left out), each with path, size, blob_id and lfs (None for a file not
        stored in LFS).
    """
    tree = HfApi().list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
    return [f for f in tree if hasattr(f, "blob_id") and hasattr(f, "size")]


def transcript_logs(meta: Path, domains: list[str]) -> set[str]:
    """
    Collects the transcript logs the frozen model tables of the given domains point to.

    Inputs:
        meta: the frozen tables' folder (<paths.data>/meta); reads <meta>/full/samples.csv, the repeat-0 `log` column.
        domains: domain names.
    Outputs:
        set of log paths relative to paths.data.
    """
    samples = pd.read_csv(meta / "full" / "samples.csv", usecols=["domain", "repeat", "log"], dtype={"log": str})
    return set(samples[samples.domain.isin(domains) & (samples.repeat == 0)].log.dropna())


def digest(path: Path, lfs: bool, size: int) -> str:
    """
    Hashes a local file the way the Hub identifies it: the sha256 of its content for an LFS file, otherwise the git
    blob sha1 (the sha1 of the header "blob <size>" and a null byte, then the content).

    Inputs:
        path: the local file.
        lfs: whether the published file is stored in LFS.
        size: the published file's size in bytes, used in the git blob header.
    Outputs:
        the hex digest.
    """
    h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def published(path: Path, f) -> bool:
    """
    Checks whether a local file is the published one: present, of the same size and with the same hash.

    Inputs:
        path: the local file.
        f: the Hub's entry for the published file (from published_files).
    Outputs:
        True when they match, else False.
    """
    return path.exists() and path.stat().st_size == f.size and digest(path, f.lfs is not None, f.size) == want(f)


def want(f) -> str:
    """
    Returns the published file's hash as the Hub reports it: the LFS sha256 for an LFS file, otherwise the git blob id.

    Inputs:
        f: the Hub's entry for the published file (from published_files).
    Outputs:
        the hex hash.
    """
    return f.lfs.sha256 if f.lfs else f.blob_id


def fetch(repo: str, revision: str, root: Path, wanted, dry_run: bool) -> None:
    """
    Downloads the files of a dataset repository that the analysis reads and that are not already present, trying each
    download up to five times (the Hub rate-limits bursts of requests) and checking its hash; first prints how many
    files and GB there are and how many are to download.

    Inputs:
        repo: the dataset repository.
        revision: the commit to download.
        root: the local folder the repository's paths are written under.
        wanted: path within the repository -> whether the analysis reads it.
        dry_run: print the counts and download nothing.
    Outputs:
        the downloaded files under root. Exits (SystemExit) when a file cannot be downloaded or its hash differs from
        the published one.
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
    """
    Fetches the two datasets: the frozen tables first, then the transcript logs they reference; then the judge runs,
    the features and the embeddings of those transcript logs. On a dry run without the frozen tables on disk, the transcript
    and embedding counts are 0, since the tables that list them are not there yet.

    Inputs:
        config: the analysis configuration (scripts/analysis/config.yaml); uses data_sources, paths and tasks (domains,
            all_semesters).
        dry_run: count the files and bytes to download, download nothing.
    Outputs:
        the files under paths.data and paths.logs.
    """
    sources, paths, tasks = config["data_sources"], config["paths"], config["tasks"]
    domains = tasks["domains"] + EXTRA_DOMAINS
    data = Path(paths["data"])
    fetch(sources["data"]["repo"], sources["data"]["revision"], data, lambda p: p.startswith("meta/"), dry_run)
    logs = transcript_logs(data / "meta", domains) if (data / "meta").is_dir() else set()
    fetch(sources["data"]["repo"], sources["data"]["revision"], data, logs.__contains__, dry_run)
    judged = re.compile(rf"^(pairwise|absolute)/[^/]+/({'|'.join(domains)})/({'|'.join(tasks['all_semesters'])})/")
    stems = {p.removesuffix(".eval") for p in logs}

    def wanted(p: str) -> bool:
        if p.startswith("embeddings/"):
            return p.removeprefix("embeddings/").rsplit(".", 2)[0] in stems
        return p.startswith("features/") or bool(judged.match(p))

    fetch(sources["logs"]["repo"], sources["logs"]["revision"], Path(paths["logs"]), wanted, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    parser.add_argument("--dry-run", action="store_true", help="count the files and bytes to download, download nothing")
    args = parser.parse_args()
    main(load_config(args.config), args.dry_run)
