#!/usr/bin/env python3
"""Import Lyptus Cyber Task Horizons model evals into ``data/``."""

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


BENCHMARK_ALIASES = {"intercode_ctf": "intercode"}
TAGS = {"imported", "lyptus"}
LEGACY_TAG = "imported_from_lyptus"


def read_header(path: Path) -> dict:
    return json.loads(
        subprocess.run(
            ["bsdtar", "-xOf", str(path), "header.json"],
            check=True,
            capture_output=True,
        ).stdout
    )


def normalize_model(name: str) -> str:
    return name.removeprefix("openrouter/").removeprefix("vllm/")


def benchmark(header: dict) -> str:
    name = header["eval"]["task"].rsplit("/", 1)[-1]
    return BENCHMARK_ALIASES.get(name, name)


def task_args_hash(header: dict) -> str:
    args = header["eval"].get("task_args") or {}
    payload = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def normalize_header(header: dict, sample_ids: list[str] | None = None) -> dict:
    header = json.loads(json.dumps(header))
    header["eval"]["model"] = normalize_model(header["eval"]["model"])
    tags = sorted(
        (
            set(header.get("tags") or []) | set(header["eval"].get("tags") or [])
        )
        - {LEGACY_TAG}
        | TAGS
    )
    header["tags"] = tags
    header["eval"]["tags"] = tags
    if sample_ids is not None:
        dataset = header["eval"].setdefault("dataset", {})
        dataset["samples"] = len(sample_ids)
        dataset["sample_ids"] = sample_ids
        results = header.setdefault("results", {})
        results["total_samples"] = len(sample_ids)
        results["completed_samples"] = len(sample_ids)
    return header


def numeric_score(value: object) -> float | None:
    if value == "C":
        return 1.0
    if value == "I":
        return 0.0
    return float(value) if isinstance(value, (int, float)) else None


def reductions(summaries: list[dict]) -> list[dict]:
    by_scorer: dict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        for scorer, score in summary.get("scores", {}).items():
            value = numeric_score(score.get("value"))
            if value is None:
                continue
            by_scorer[scorer].append(
                {
                    key: score[key]
                    for key in ("answer", "explanation", "history")
                    if key in score
                }
                | {"value": value, "sample_id": summary["id"]}
            )
    return [
        {"scorer": scorer, "samples": samples}
        for scorer, samples in sorted(by_scorer.items())
    ]


def update_metrics(header: dict, reduced: list[dict]) -> None:
    by_scorer = {item["scorer"]: item["samples"] for item in reduced}
    for score in header["results"].get("scores", []):
        samples = by_scorer.get(score["scorer"], [])
        values = [sample["value"] for sample in samples]
        score["scored_samples"] = len(values)
        score["unscored_samples"] = header["results"]["completed_samples"] - len(values)
        metrics = score.get("metrics", {})
        if values and "accuracy" in metrics:
            metrics["accuracy"]["value"] = statistics.mean(values)
        if values and "stderr" in metrics:
            metrics["stderr"]["value"] = (
                statistics.stdev(values) / len(values) ** 0.5
                if len(values) > 1
                else 0.0
            )


def source_logs(eval_set: Path) -> list[Path]:
    timestamped = sorted(
        path for path in eval_set.glob("*.eval") if path.name[:4].isdigit()
    )
    return timestamped or sorted(eval_set.glob("*.eval"))


def successful_logs(eval_set: Path) -> list[Path]:
    logs = []
    for path in source_logs(eval_set):
        try:
            if read_header(path).get("status") == "success":
                logs.append(path)
        except (json.JSONDecodeError, subprocess.CalledProcessError):
            continue
    return logs


def extract_archive(path: Path, destination: Path) -> None:
    members = subprocess.run(
        ["bsdtar", "-tf", str(path)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if any(
        PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        for name in members
    ):
        raise ValueError(f"unsafe archive path in {path}")
    subprocess.run(["bsdtar", "-xf", str(path), "-C", str(destination)], check=True)


def archive_samples(path: Path) -> tuple[dict[str, tuple[str, bytes]], dict[str, dict]]:
    samples = {}
    for sample in (path / "samples").glob("*.json"):
        data = sample.read_bytes()
        samples[json.loads(data)["id"]] = (sample.relative_to(path).as_posix(), data)
    summaries_path = path / "summaries.json"
    summaries = (
        {item["id"]: item for item in json.loads(summaries_path.read_bytes())}
        if summaries_path.exists()
        else {}
    )
    return samples, summaries


def write_log(logs: list[Path], output: Path) -> None:
    headers = [read_header(path) for path in logs]
    identities = {
        (
            benchmark(header),
            task_args_hash(header),
            normalize_model(header["eval"]["model"]),
        )
        for header in headers
    }
    if len(identities) != 1:
        raise ValueError(f"incompatible retry logs: {logs}")

    if len(logs) == 1:
        shutil.copyfile(logs[0], output)
        with zipfile.ZipFile(output, "a") as archive:
            archive.filelist = [
                entry for entry in archive.filelist if entry.filename != "header.json"
            ]
            archive.NameToInfo.pop("header.json", None)
            archive.writestr(
                "header.json", json.dumps(normalize_header(headers[0])).encode()
            )
        return

    with tempfile.TemporaryDirectory() as temporary:
        roots = []
        for index, path in enumerate(logs):
            root = Path(temporary) / str(index)
            root.mkdir()
            extract_archive(path, root)
            roots.append(root)

        samples: dict[str, tuple[str, bytes]] = {}
        summaries: dict[str, dict] = {}
        for root in roots:
            next_samples, next_summaries = archive_samples(root)
            samples.update(next_samples)
            summaries.update(next_summaries)
        sample_ids = sorted(samples)
        header = normalize_header(headers[-1], sample_ids)
        with zipfile.ZipFile(
            output, "w", zipfile.ZIP_DEFLATED, allowZip64=True
        ) as destination:
            if set(samples) != set(summaries):
                raise ValueError(f"sample/summary mismatch in {logs}")
            reduced = reductions([summaries[sample_id] for sample_id in sample_ids])
            update_metrics(header, reduced)
            destination.writestr("header.json", json.dumps(header).encode())
            destination.writestr(
                "summaries.json",
                json.dumps([summaries[key] for key in sample_ids]).encode(),
            )
            destination.writestr("reductions.json", json.dumps(reduced).encode())
            for sample_id in sample_ids:
                name, data = samples[sample_id]
                destination.writestr(name, data)


def valid_output(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            header = json.loads(archive.read("header.json"))
            return TAGS <= set(header.get("tags", [])) and TAGS <= set(
                header["eval"].get("tags", [])
            )
    except (KeyError, OSError, zipfile.BadZipFile):
        return False


def import_evals(source: Path, target: Path) -> tuple[int, int]:
    planned = []
    for eval_set in sorted(path for path in source.iterdir() if path.is_dir()):
        logs = successful_logs(eval_set)
        if not logs:
            continue
        header = read_header(logs[-1])
        model = normalize_model(header["eval"]["model"]).rsplit("/", 1)[-1].lower()
        planned.append(
            (benchmark(header), task_args_hash(header), model, eval_set.name, logs)
        )

    repeats: dict[tuple[str, str, str], int] = defaultdict(int)
    imported = skipped = 0
    for bench, args_hash, model, source_id, logs in planned:
        repeat = repeats[bench, args_hash, model]
        repeats[bench, args_hash, model] += 1
        output = (
            target
            / bench
            / args_hash
            / model
            / str(repeat)
            / f"lyptus-{source_id}.eval"
        )
        if valid_output(output):
            skipped += 1
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        part = output.with_suffix(".eval.part")
        write_log(logs, part)
        if not valid_output(part):
            raise ValueError(f"invalid output archive {part}")
        part.replace(output)
        imported += 1
    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Lyptus data/eval_logs directory")
    parser.add_argument("--target", type=Path, default=Path("data"))
    args = parser.parse_args()
    imported, skipped = import_evals(args.source, args.target)
    print(
        f"Imported {imported} Lyptus eval sets; skipped {skipped} valid existing outputs"
    )


if __name__ == "__main__":
    main()
