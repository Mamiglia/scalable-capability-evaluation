#!/usr/bin/env python3
"""Import Epoch AI logs into the repository layout, split by eval epoch."""

import argparse
import csv
import json
import re
import statistics
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


TASK_ARGS_HASH = "a288e1a9"  # sha256 of normalized task arguments: {"cot": true}
TAGS = {"epoch_ai", "imported"}
LEGACY_TAG = "imported_epoch_ai"


def normalize_model(name: str) -> str:
    if not name.startswith("epoch/"):
        return name
    model = name.removeprefix("epoch/")
    if model.startswith("MiniMaxAI/"):
        return model
    providers = {
        "claude-": "anthropic",
        "deepseek-": "deepseek-ai",
        "gemini-": "google",
        "gemma-": "google",
        "glm-": "zai-org",
        "gpt-": "openai",
        "kimi-": "moonshotai",
        "magistral-": "mistralai",
        "o1-": "openai",
        "o3-": "openai",
        "o4-": "openai",
        "qwen": "Qwen",
    }
    provider = next((value for prefix, value in providers.items() if model.lower().startswith(prefix)), None)
    return f"{provider}/{model}" if provider else model


def normalize_eval(header: dict, benchmark: str) -> None:
    task = "inspect_evals/gpqa_diamond" if benchmark == "gpqa" else "otis_mock_aime_2024_2025"
    display = "GPQA Diamond" if benchmark == "gpqa" else "OTIS Mock AIME 2024-2025"
    header["eval"]["task"] = task
    header["eval"]["task_registry_name"] = task
    header["eval"]["task_display_name"] = display
    header["eval"]["task_args"] = {"cot": True}
    header["eval"]["task_args_passed"] = {"cot": True}
    header["eval"]["config"]["epochs"] = 1
    header["eval"]["model"] = normalize_model(header["eval"]["model"])
    tags = sorted(
        (
            set(header.get("tags") or []) | set(header["eval"].get("tags") or [])
        )
        - {LEGACY_TAG}
        | TAGS
    )
    header["eval"]["tags"] = tags
    header["tags"] = tags


def model_names(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="") as f:
        return {
            row["id"]: re.sub(r"[^A-Za-z0-9._-]+", "-", row["Model version"]).strip("-")
            for row in csv.DictReader(f)
            if row.get("Logs")
        }


def epoch_results(header: dict, summaries: list[dict], epoch: int) -> tuple[dict, list[dict], list[dict]]:
    selected = [json.loads(json.dumps(item)) for item in summaries if item.get("epoch", 1) == epoch]
    for item in selected:
        item["epoch"] = 1

    reductions = []
    for scorer in {name for item in selected for name in item.get("scores", {})}:
        samples = []
        for item in selected:
            score = item.get("scores", {}).get(scorer)
            if not score:
                continue
            raw = score.get("value")
            value = {"C": 1.0, "I": 0.0}.get(raw, raw if isinstance(raw, (int, float)) else None)
            if value is None:
                continue
            reduced = {"value": value, "sample_id": item["id"]}
            if score.get("answer") is not None:
                reduced["answer"] = score["answer"]
            samples.append(reduced)
        reductions.append({"scorer": scorer, "samples": samples})

    split_header = json.loads(json.dumps(header))
    split_header["eval"]["config"]["epochs"] = 1
    split_header["results"]["total_samples"] = len(selected)
    split_header["results"]["completed_samples"] = len(selected)
    by_scorer = {item["scorer"]: item["samples"] for item in reductions}
    for score in split_header["results"].get("scores", []):
        values = [item["value"] for item in by_scorer.get(score["scorer"], [])]
        if not values:
            continue
        accuracy = statistics.mean(values)
        stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        if "accuracy" in score.get("metrics", {}):
            score["metrics"]["accuracy"]["value"] = accuracy
        if "stderr" in score.get("metrics", {}):
            score["metrics"]["stderr"]["value"] = stderr
    return split_header, selected, reductions


def import_log(source: Path, destination: Path, benchmark: str) -> int:
    members = subprocess.run(
        ["bsdtar", "-tf", source], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in members):
        raise ValueError(f"unsafe archive path in {source}")

    epochs = sorted(
        {
            int(match.group(1))
            for name in members
            if (match := re.search(r"_epoch_(\d+)\.json$", name))
        }
    )
    if not epochs:
        raise ValueError(f"no epoch samples in {source}")

    with tempfile.TemporaryDirectory() as tmp:
        extracted = Path(tmp)
        subprocess.run(["bsdtar", "-xf", source, "-C", extracted], check=True)
        header = json.loads((extracted / "header.json").read_text())
        normalize_eval(header, benchmark)
        summaries = json.loads((extracted / "summaries.json").read_text())

        for index, epoch in enumerate(epochs):
            samples = sorted((extracted / "samples").glob(f"*_epoch_{epoch}.json"))
            output = destination / str(index) / source.name
            if output.exists():
                with zipfile.ZipFile(output) as archive:
                    if archive.testzip() is None:
                        continue

            split_header, split_summaries, reductions = epoch_results(header, summaries, epoch)
            if len(split_summaries) != len(samples):
                raise ValueError(f"summary/sample mismatch in {source} epoch {epoch}")

            output.parent.mkdir(parents=True, exist_ok=True)
            part = output.with_suffix(".eval.part")
            with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                archive.writestr("header.json", json.dumps(split_header).encode())
                archive.writestr("summaries.json", json.dumps(split_summaries).encode())
                archive.writestr("reductions.json", json.dumps(reductions).encode())
                for sample_path in samples:
                    sample = json.loads(sample_path.read_text())
                    sample["epoch"] = 1
                    name = re.sub(r"_epoch_\d+\.json$", "_epoch_1.json", sample_path.name)
                    archive.writestr(f"samples/{name}", json.dumps(sample).encode())
            with zipfile.ZipFile(part) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"invalid output archive {part}")
            part.replace(output)

    return len(epochs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("benchmark")
    parser.add_argument("--target", type=Path, default=Path("data"))
    args = parser.parse_args()

    names = model_names(args.metadata)
    sources = sorted(args.source.glob("*.eval"))
    missing = [path.stem for path in sources if path.stem not in names]
    if missing:
        raise ValueError(f"run IDs missing from metadata: {missing}")

    epoch_count = 0
    for number, source in enumerate(sources, 1):
        destination = args.target / args.benchmark / TASK_ARGS_HASH / names[source.stem]
        epoch_count += import_log(source, destination, args.benchmark)
        print(f"[{number}/{len(sources)}] {source.name} -> {destination}")

    print(f"Imported {len(sources)} logs as {epoch_count} single-epoch files")


if __name__ == "__main__":
    main()
