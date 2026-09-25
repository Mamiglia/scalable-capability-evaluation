"""Export the data one semester task of one domain needs, with machine-identifying strings scrubbed.

    uv run python scripts/export_subset.py cyber 2026-I --out release [--conditions 24d6cc32 9c1d3abb] [--compact]

Writes, mirroring the repository layout so `release/` can be dropped next to the code:
    data/meta/<domain>/                        the frozen tables (every task's columns; small)
    data/<bench>/<hash>/<model>/0/*.eval       the repeat-0 rollouts of the task's references, historical and targets
    glm_features/..., transcript-embeddings/... the deterministic features and embeddings of those rollouts
    logs/judge/<domain>/<semester>/<judge>/<condition>/   pairwise runs of the chosen conditions (default: all)
    logs/direct/<domain>/<semester>/                      absolute-scoring runs
Every .eval is rewritten: samples keep only SAMPLE_FIELDS (the transcript, its score and usage; no event traces,
retried attempts or API call records), headers lose HEADER_DROP (hosting, endpoint and revision details), the
_journal/ copy of the header is dropped, and SCRUB is applied to every remaining byte (home and checkout paths,
account and host names in tracebacks and in the transcripts judges saw).
"""

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import zipfile_zstd  # noqa: F401  (registers the zstd codec the logs are written with)

SCRUB = [  # (pattern, replacement) on raw bytes, most specific first
    (rb"(?:git@github\.com:|https://github\.com/)[\w.-]+/(?:small-judge|scalable-capability-evaluation)(?:\.git)?", b"anonymous"),
    (rb"https://github\.com/Mamiglia/inspect_evals(?:\.git)?", b"https://github.com/UKGovernmentBEIS/inspect_evals.git"),
    (rb"/media/hdd/usr/tao/small-judge[\w-]*", b"/workspace/sce"),
    (rb"/media/hdd/usr/tao", b"/workspace"),
    (rb"/home/tao\b", b"/home/user"),
    (rb"/Users/[\w.-]+/PycharmProjects/[\w.-]+/[\w.-]+", b"/workspace/sce"),  # a collaborator's checkout
    (rb"/Users/[\w.-]+", b"/Users/user"),  # any macOS home
    (rb"small[-_]judge", b"sce"),
    (rb"Mamiglia|generality-labs|LASR-G5", b"anonymous"),
    (rb"/dev/shm/tao-podman", b"/dev/shm/user-podman"),  # the collection host's rootless podman store
    (rb"\bloki\b", b"host"),  # the collection host's name, seen by agents in /etc/hosts
]
# What the readers use (evals.transcripts, protocols.*.from_logs); everything else in a sample is dropped.
SAMPLE_FIELDS = {"id", "epoch", "input", "target", "messages", "output", "scores", "metadata", "model_usage",
                 "attachments", "error", "limit", "uuid", "started_at", "completed_at", "total_time", "working_time"}
HEADER_DROP = {"model_base_url", "runpod_pod_id", "runpod_machine", "runpod_datacenter", "source_path"}
CHECK = re.compile(rb"/media/hdd|/home/tao\b|/Users/(?!user\b)|PycharmProjects|\btao-podman|\bloki\b|Mamiglia|small[-_]judge|generality-labs|LASR-G5")


def scrub(data: bytes) -> bytes:
    for pattern, replacement in SCRUB:
        data = re.sub(pattern, replacement, data)
    return data


def drop_keys(o, keys: set[str]):
    """o without any dict entry named in keys, at any depth; revision commits blanked (they name private commits)."""
    if isinstance(o, dict):
        o = {k: ("" if k == "commit" else drop_keys(v, keys)) for k, v in o.items() if k not in keys}
    elif isinstance(o, list):
        o = [drop_keys(v, keys) for v in o]
    return o


def copy_log(src: Path, dst: Path, keep=None) -> None:
    """dst = src sanitised (module docstring), same member order and compression. keep(name) -> False drops a
    member; with a keep (the compact export), summaries also lose their prompt preview, unread by any analysis."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        finished = "header.json" in zin.namelist()
        for info in zin.infolist():
            name = info.filename
            if (keep and not keep(name)) or (finished and name.startswith("_journal/")):
                continue
            data = zin.read(info)
            if name.startswith("samples/"):
                data = json.dumps({k: v for k, v in json.loads(data).items() if k in SAMPLE_FIELDS}).encode()
            elif name in ("header.json", "_journal/start.json"):
                data = json.dumps(drop_keys(json.loads(data), HEADER_DROP)).encode()
            elif name == "summaries.json" and keep:
                data = json.dumps([s | {"input": ""} for s in json.loads(data)]).encode()
            zout.writestr(info, scrub(data), compress_type=info.compress_type)


def verdicts_only(name: str) -> bool:
    """A judge log reduced to what the analyses read: the header and the sample summaries (scores, metadata)."""
    return name in ("header.json", "summaries.json")


def copy_tree(src: Path, dst: Path, keep=None) -> int:
    """Every file under src into dst (.eval scrubbed, reduced by keep); returns the number of files."""
    n = 0
    for path in sorted(src.rglob("*")):
        if path.is_file():
            target = dst / path.relative_to(src)
            if path.suffix == ".eval":
                copy_log(path, target, keep)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(scrub(path.read_bytes()))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domain")
    ap.add_argument("semester")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", nargs="*", help="pairwise condition hashes to keep (default: every condition)")
    ap.add_argument("--compact", action="store_true", help="only what the forecasts read: rollouts keep this task's train "
                    "items without event traces, judge logs keep their verdict summaries (enough for results.py, "
                    "not for re-judging or re-embedding)")
    args = ap.parse_args()
    out, domain, semester = args.out, args.domain, args.semester

    meta = Path("data/meta") / domain
    print(f"{meta}: {copy_tree(meta, out / meta)} files")
    models = pd.read_csv(meta / "models.csv", index_col="model")
    task = models[models[f"{semester}_role"].isin(["reference", "historical", "target"])]
    paths = [Path(p) for c in models.columns if c.startswith("path_") for p in task[c].dropna()]
    items = pd.read_csv(meta / "items.csv", dtype={"item": str})
    train = items[items[f"{semester}_train"].fillna(False).astype(bool)]
    for path in paths:
        ids = set(train.item[train.bench == path.parts[0]])  # data/<bench>/...: the dir is the bench name
        keep = (lambda name, ids=ids: not name.startswith("_journal/") and
                (not name.startswith("samples/") or name[8:].rsplit("_epoch_", 1)[0] in ids)) if args.compact else None
        copy_log(Path("data") / path, out / "data" / path, keep)
        for cache in ("glm_features", "transcript-embeddings"):  # both mirror data/<log path without .eval>.*
            for extra in (Path(cache) / path.parent).glob(f"{path.stem}.*"):
                (out / extra).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(extra, out / extra)
    print(f"{len(task)} models, {len(paths)} rollout logs")

    for kind in ("judge", "direct"):
        root = Path("logs") / kind / domain / semester
        runs = sorted(p for p in root.glob("*/*") if p.is_dir())
        if kind == "judge" and args.conditions:
            runs = [p for p in runs if p.name in args.conditions]
        for run in runs:
            print(f"{run}: {copy_tree(run, out / run, verdicts_only if args.compact else None)} files", flush=True)

    leaks = [p for p in out.rglob("*.eval") if any(CHECK.search(zipfile.ZipFile(p).read(i)) for i in zipfile.ZipFile(p).infolist())]
    leaks += [p for p in out.rglob("*") if p.is_file() and p.suffix != ".eval" and CHECK.search(p.read_bytes())]
    if leaks:
        raise SystemExit(f"identifying strings remain in {len(leaks)} files, e.g. {leaks[:3]}")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out}: {size / 1e6:.0f} MB, no identifying strings")


if __name__ == "__main__":
    main()
