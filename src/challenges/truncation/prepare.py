"""Freeze matched truncated direct inputs for <domain>/<judge>/<semester>/<head|tail>-<max_length>; no API calls.

A condition reuses the full-transcript campaign's exact retained inputs, exclusions, references and
target truth, re-rendered with JudgeConfig.max_length (characters kept per transcript, applied to
calibration and target transcripts alike). Ratings are reused wherever the limit leaves the rendered
input unchanged; every truncated transcript is verified to be a prefix (head) or suffix (tail) of the full one.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import draccus
import pandas as pd

from capability.task import Task
import capability.task as cohort_module
from configs import DirectJudgeConfig
from configs.domain import DomainConfig
from evals.transcripts import read_transcripts
from judge.dataset import DIRECT_USER_PROMPT
from protocols.direct_judge import JudgeDirect
from protocols.direct_judge import judge as judge_module
from reference import tables

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/direct-judging-truncation"  # frozen inputs, logs, results
ARTIFACTS = REPO / "artifacts"  # visible campaign harnesses
SUFFIX = DIRECT_USER_PROMPT.split("{transcript}", 1)[1]  # rendered text after the transcript body
RATE_LIMITED = ("siliconflow", "novita")  # providers that need fewer connections and more retries


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


CAMPAIGNS = {
    # domain: (visible-campaign roots, in order; the first supplies configs, prompt and rating exports)
    "cyber": ["direct-judging-cyber"],
    # coding: the published refit = SWE/Terminal campaign (staged at e3d86809, filtered to the latest train
    # set) + AgentBench OS add-on (staged at c199498b), calibrated on the add-on's (latest) tables.
    "coding": ["direct-judging-coding-agentbench", "direct-judging-coding"],
}
SUBSET_DIR = {"direct-judging-cyber": "input-subsets", "direct-judging-coding": "input-subsets",
              "direct-judging-coding-agentbench": "hf-snapshot"}


def stage_source(domain_name):
    """One transcript root for the domain: the campaign's staged subset, or for coding a symlink tree
    joining the SWE/Terminal staging with the AgentBench snapshot (data/ holds LFS pointers only)."""
    sources = [ARTIFACTS / name for name in CAMPAIGNS[domain_name]]
    staged = {}
    for source in sources:
        for path in pd.read_csv(source / "semester-inputs.csv", usecols=["path"]).path.unique():
            staged.setdefault(path, source / SUBSET_DIR[source.name] / path)  # first campaign wins
    if len(sources) == 1:
        root = sources[0] / SUBSET_DIR[sources[0].name]
    else:
        root = ROOT / "staging" / domain_name
        for relative, target in staged.items():
            link = root / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                assert link.resolve() == target.resolve(), relative
            else:
                link.symlink_to(target.resolve())
    assert all((root / p).exists() for p in staged), "staged transcript missing"
    tables.DATA, tables.META = root, sources[0] / "meta"
    judge_module.DATA = tables.DATA

    def read_subset(path, require_correct=True, include_reasoning=True, items=None):
        if str(Path(path).relative_to(tables.DATA)) not in staged:
            return {}
        found = read_transcripts(path, require_correct, include_reasoning, items=None)
        return found if items is None else {key: value for key, value in found.items() if key in items}

    cohort_module.read_transcripts = read_subset
    return sources


def retained(sources, judge_name, semester, train):
    """Retained (task, item) -> render hash across the campaigns, restricted to the latest train items,
    plus the campaign job records (route, baseline log dirs) and manifest hashes."""
    expected, jobs, hashes = {}, [], {}
    for source in sources:
        manifest = json.loads((source / "run-manifest.json").read_text())
        job = next(job for job in manifest["judges"] if job["name"] == judge_name)
        assert semester in job["semesters"], (source.name, judge_name, semester)
        for t in job["tasks"]:
            if t["semester"] != semester:
                continue
            for s in t["samples"]:
                if (t["bench"], str(s["id"])) in train:
                    assert (t["name"], str(s["id"])) not in expected
                    expected[(t["name"], str(s["id"]))] = s["render_sha256"]
        jobs.append(job)
        hashes[source.name] = {"run_manifest_sha256": sha((source / "run-manifest.json").read_bytes()),
                               "hf_snapshot": manifest["hf_snapshot"], "prompt_sha256": manifest["source_hashes"]["prompt.txt"]}
    return expected, jobs, hashes


def prepare(domain_name, judge_name, semester, cut):
    mode, max_length = cut.split("-")
    max_length = int(max_length)
    sources = stage_source(domain_name)
    source = sources[0]
    config = DirectJudgeConfig.from_cli(source / "configs" / f"{judge_name}.yml")
    assert config.include_reasoning and config.max_length > max_length, (config.max_length, max_length)
    prompt = (source / "prompt.txt").read_text()
    domain = DomainConfig.from_cli(f"configs/domain/{domain_name}.yml")
    task = Task.from_semester(semester, domain)
    items = pd.read_csv(tables.META / domain_name / "items.csv", dtype={"item": str})
    train = set(map(tuple, items.loc[items[f"{semester}_train"], ["bench", "item"]].astype(str).to_numpy()))
    expected, jobs, source_hashes = retained(sources, judge_name, semester, train)
    assert all(h["prompt_sha256"] == sha(prompt) for h in source_hashes.values())
    full = JudgeDirect(config, domain)
    truncated = JudgeDirect(replace(config, max_length=max_length, truncate=mode), domain)
    originals = {(t.name, str(s.id)): s for t in full.tasks(task.references, task.test) for s in t.dataset}
    ratings_path = source / ("consolidated/results" if len(sources) > 1 else "results") / judge_name / semester / "ratings.csv"
    ratings = pd.read_csv(ratings_path, dtype={"item": str})
    assert not ratings.duplicated(["model", "bench", "item", "epoch"]).any()
    assert ratings.epoch.eq(1).all() and ratings.semester.eq(semester).all()
    rated = {(f"{r.bench}/{r.model}", r.item): r._asdict() for r in ratings.itertuples(index=False)}
    assert set(rated) <= set(expected), "rating export outside the retained set"
    tasks, reused, audit, observed = [], [], [], set()
    for t in truncated.tasks(task.references, task.test):
        assert t.metadata["include_reasoning"] is True and t.metadata["max_length"] == max_length and t.metadata["truncate"] == mode
        samples = []
        for sample in t.dataset:
            key = (t.name, str(sample.id))
            if key not in expected:
                continue  # preserve the full-transcript campaign's exclusions (context preflight)
            original = originals[key]
            assert sha(original.input) == expected[key], key
            assert not original.metadata["truncated"], key  # the baseline never cut a retained transcript
            assert sample.input.endswith(SUFFIX) and original.input.endswith(SUFFIX), key
            body, full_body = sample.input[:-len(SUFFIX)], original.input[:-len(SUFFIX)]
            kept, full_chars = sample.metadata["chars"], original.metadata["chars"]
            text, full_text = body[len(body) - kept:], full_body[len(full_body) - full_chars:]
            assert body[:len(body) - kept] == full_body[:len(full_body) - full_chars], key  # same question and options
            assert (full_text.startswith(text) if mode == "head" else full_text.endswith(text)), key  # deterministic cut
            cut = len(full_body) > len(body)
            assert sample.metadata["truncated"] is cut, key
            assert sample.metadata["chars"] == min(max_length, original.metadata["chars"]), key
            # A shorter input still satisfies the full campaign's context preflight.
            unchanged = sample.input == original.input
            assert unchanged is not cut
            reuse = unchanged and key in rated
            audit.append({"task": t.name, "item": str(sample.id), "full_sha256": expected[key],
                          "truncated_sha256": sha(sample.input), "truncated": cut, "reused": reuse,
                          "truncated_chars": sample.metadata["chars"], "full_chars": original.metadata["chars"]})
            if reuse:
                reused.append(rated[key])
            else:
                samples.append(sample.model_dump(mode="json"))
            observed.add(key)
        if samples:
            tasks.append({"name": t.name, "metadata": t.metadata, "samples": samples})
    assert observed == set(expected)
    assert len(reused) + sum(len(t["samples"]) for t in tasks) == len(expected)
    changed = sum(row["truncated"] for row in audit)
    summary = {"domain": domain_name, "judge": judge_name, "semester": semester, "truncate": mode, "max_length": max_length,
               "total_ratings": len(observed), "changed_inputs": changed, "reused_ratings": len(reused),
               "new_ratings": sum(len(t["samples"]) for t in tasks), "tasks": len(tasks),
               "references": len(task.references.models), "targets": len(task.truth),
               "truncated_reference_inputs": sum(r["truncated"] for r in audit if r["task"].split("/", 1)[1] in task.references.models.index),
               "truncated_target_inputs": sum(r["truncated"] for r in audit if r["task"].split("/", 1)[1] in task.truth.index)}
    if not changed:
        summary["status"] = "identical to full-transcript condition; no collection"
        print(json.dumps(summary), flush=True)
        return summary
    assert not config.judge.on_runpod, "truncation collection is API-hosted; no Runpod judge selected"
    output = ROOT / "inputs" / domain_name / judge_name / semester / f"{mode}-{max_length}"
    output.mkdir(parents=True, exist_ok=True)
    cfg = draccus.encode(truncated.config)
    assert cfg["max_length"] == max_length and cfg["truncate"] == mode
    cfg["system_prompt"] = "prompt.txt"
    cfg["log_dir"] = "logs"
    pinned = cfg["judge"]["model_args"]["provider"]["only"]
    limited = any(p.startswith(RATE_LIMITED) for p in pinned)
    cfg["_kwargs"].update(max_connections=8 if limited else 16, max_retries=10, timeout=600,
                          log_model_api=True, log_realtime=True, log_buffer=1, retry_cleanup=False)
    cfg["judge"]["model_args"]["provider"]["require_parameters"] = True
    cfg["judge"]["model_args"]["default_headers"] = {"X-OpenRouter-Metadata": "enabled"}
    payload = {"domain": domain_name, "semester": semester, "judge": judge_name, "truncate": mode, "max_length": max_length,
               "judge_config": jobs[0]["route"]["judge_config"], "model_config": jobs[0]["route"]["model_config"],
               "baseline_log_dirs": [job["log_dirs"][semester] for job in jobs],
               "config": cfg, "tasks": tasks, "reused_ratings": reused,
               "references": task.references.theta.to_dict(), "truth": task.truth.to_dict(),
               "source_campaigns": source_hashes, "source_ratings_sha256": sha(ratings_path.read_bytes()),
               "source_snapshot": source_hashes[source.name]["hf_snapshot"], "audit": audit}
    (output / "prompt.txt").write_text(prompt)
    (output / "job.json").write_text(json.dumps(payload, indent=2) + "\n")
    summary |= {"status": "prepared", "job_sha256": sha((output / "job.json").read_bytes()), "prompt_sha256": sha(prompt)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conditions", nargs="+", help="<domain>/<judge>/<semester>/<head|tail>-<max_length chars>")
    args = parser.parse_args()
    summaries = [prepare(*c.split("/")) for c in args.conditions]
    path = ROOT / "sweep-summary.json"
    previous = json.loads(path.read_text()) if path.exists() else []
    keys = {(s["domain"], s["judge"], s["semester"], s["truncate"], s["max_length"]) for s in summaries}
    kept = [s for s in previous if (s["domain"], s["judge"], s["semester"], s.get("truncate"), s["max_length"]) not in keys]
    path.write_text(json.dumps(kept + summaries, indent=2) + "\n")
