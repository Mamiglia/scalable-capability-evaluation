#!/usr/bin/env python3
"""Import Epoch AI SWE-bench Verified logs into data/swe_bench_mini, keeping only the mini items.

Source: external_data/epoch-swe-bench/logs (scripts/fetch_epoch_swebench.py). Only runs with
Epoch's bash agent are imported; the Claude Code / Codex scaffold runs are skipped. All bash
runs share one hash dir, keyed on the scaffold-defining args (agent, token_limit); the infra-only
skip_cache_warmup / cache_read_limit are left out of the key. Each log is one repeat (0) of the
model dir named by the Epoch filename (effort suffix kept). Epoch's dataset lacks 4 mini items,
so logs hold 46 samples. Sample payloads are copied byte for byte; the subset is stamped.

Dry run by default; --write to import. A second --write pass reports 0 imported.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import inspect_ai._util.zipfile  # noqa: F401  patches zstd (compress_type 93) into zipfile on py<3.14
from import_lyptus_evals import reductions, update_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit" / "logs"))
from eval_edits import stamp_edit  # noqa: E402

TAGS = {"epoch_ai", "imported"}
SCAFFOLD_ARGS = {"agent": "bash", "token_limit": 2000000}
ARGS_HASH = hashlib.sha256(json.dumps(SCAFFOLD_ARGS, sort_keys=True).encode()).hexdigest()[:8]


def mini_ids(data: Path) -> list[str]:
    native = next((data / "swe_bench_mini" / "477dc96c").glob("*/0/*.eval"))
    with zipfile.ZipFile(native) as archive:
        return json.loads(archive.read("header.json"))["eval"]["dataset"]["sample_ids"]


def valid_output(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            header = json.loads(archive.read("header.json"))
            return TAGS <= set(header["eval"].get("tags", [])) and archive.testzip() is None
    except (KeyError, OSError, zipfile.BadZipFile):
        return False


def import_log(source: Path, output: Path, mini: set[str]) -> dict:
    with zipfile.ZipFile(source) as archive:
        header = json.loads(archive.read("header.json"))
        summaries = [s for s in json.loads(archive.read("summaries.json")) if s["id"] in mini]
        ids = sorted(s["id"] for s in summaries)
        source_samples = len(header["eval"]["dataset"]["sample_ids"])

        dataset = header["eval"]["dataset"]
        dataset["samples"], dataset["sample_ids"] = len(ids), ids
        header["results"]["total_samples"] = header["results"]["completed_samples"] = len(ids)
        reduced = reductions(summaries)
        update_metrics(header, reduced)

        spec = SimpleNamespace(eval=SimpleNamespace(metadata=header["eval"].get("metadata"),
                                                    tags=sorted({*(header["eval"].get("tags") or []), *TAGS})))
        stamp_edit(spec, tool="scripts/judge/import_epoch_swebench.py",
                   reason="import: subset Epoch SWE-bench Verified run to the swe_bench_verified_mini items",
                   source=source.name, samples_changed=source_samples - len(ids),
                   before={"samples": source_samples}, after={"samples": len(ids)})
        header["eval"]["metadata"], header["eval"]["tags"] = spec.eval.metadata, spec.eval.tags
        header["tags"] = sorted({*(header.get("tags") or []), *spec.eval.tags})

        output.parent.mkdir(parents=True, exist_ok=True)
        part = output.with_suffix(".eval.part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as out:
            out.writestr("header.json", json.dumps(header).encode())
            out.writestr("summaries.json", json.dumps(summaries).encode())
            out.writestr("reductions.json", json.dumps(reduced).encode())
            for sample in summaries:
                name = f"samples/{sample['id']}_epoch_{sample['epoch']}.json"
                out.writestr(name, archive.read(name))
    if not valid_output(part):
        raise ValueError(f"invalid output archive {part}")
    part.replace(output)
    scored = [s for s in reduced[0]["samples"]] if reduced else []
    return {"samples": len(ids), "scored": len(scored),
            "accuracy": sum(s["value"] for s in scored) / len(ids) if ids else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--target", type=Path, default=Path("data"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    mini = set(mini_ids(args.target))
    planned, skipped = {}, []
    for source in sorted(args.source.glob("*.eval")):
        with zipfile.ZipFile(source) as archive:
            spec = json.loads(archive.read("header.json"))["eval"]
        if spec["metadata"].get("agent-type") != "bash":
            skipped.append(source.name)
            continue
        if {k: spec["task_args"].get(k) for k in SCAFFOLD_ARGS} != SCAFFOLD_ARGS:
            raise ValueError(f"unexpected scaffold args in {source}: {spec['task_args']}")
        model = source.name.split("__")[0].lower()
        if model in planned:
            raise ValueError(f"two bash runs for {model}: {planned[model].name}, {source.name}")
        planned[model] = source

    print(f"skipped (not bash agent): {skipped}")
    imported = 0
    for model, source in planned.items():
        output = args.target / "swe_bench_mini" / ARGS_HASH / model / "0" / f"{source.name.split('__')[1]}"
        if valid_output(output):
            continue
        if not args.write:
            print(f"would import {source.name} -> {output}")
            imported += 1
            continue
        print(f"{source.name} -> {output}: {import_log(source, output, mini)}", flush=True)
        imported += 1
    print(f"{'imported' if args.write else 'would import'} {imported} of {len(planned)} bash-agent logs")


if __name__ == "__main__":
    main()
