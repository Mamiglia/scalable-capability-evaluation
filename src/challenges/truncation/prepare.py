"""Freeze matched direct inputs for <domain>/<judge>/<semester>/<head|tail|ends>-<max_length> or
.../channels-<only|drop>-<channel> or .../thirds-<kept> (kept character thirds, on the 40 removal items per task); no API calls.

A condition reuses the full-transcript campaign's exact retained inputs, exclusions, references and
target truth, re-rendered with JudgeConfig.max_length (characters kept per transcript) or JudgeConfig.channels
(the transcript channels shown), applied to calibration and target transcripts alike. Ratings are reused
wherever the condition leaves the rendered input unchanged; every truncated transcript is verified to equal
the head, tail or ends cut of the full one. A channel condition that leaves a transcript under
MIN_RESPONSE_CHARS excludes that input, as pairwise judging skips such a pair.
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
from judge.dataset import DIRECT_USER_PROMPT, MIN_RESPONSE_CHARS, cut as cut_text
from protocols.direct_judge import JudgeDirect
from protocols.direct_judge import judge as judge_module
from reference import tables
from analysis.transcript_parts.paired_change import removal_items
from challenges.truncation.run import cut_fields, root

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
    "math": ["direct-judging-math"],
    "science": ["direct-judging-science"],
}
SUBSET_DIR = {"direct-judging-cyber": "input-subsets", "direct-judging-coding": "input-subsets",
              "direct-judging-coding-agentbench": "hf-snapshot", "direct-judging-math": "input-subsets",
              "direct-judging-science": "input-subsets"}


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
    fields = cut_fields(cut)
    channels = "channels" in fields
    thirds = fields.get("truncate", "").startswith("thirds-")
    sources = stage_source(domain_name)
    source = sources[0]
    config = DirectJudgeConfig.from_cli(source / "configs" / f"{judge_name}.yml")
    assert config.include_reasoning and not config.channels and config.max_length > fields.get("max_length", 0), (config.max_length, fields)
    max_length, mode = fields.get("max_length", config.max_length), fields.get("truncate", config.truncate)
    prompt = (source / "prompt.txt").read_text()
    domain = DomainConfig.from_cli(f"configs/domain/{domain_name}.yml")
    task = Task.from_semester(semester, domain)
    items = tables.load(domain)[1]
    train = set(map(tuple, items.loc[items[f"{semester}_train"], ["bench", "item"]].astype(str).to_numpy()))
    expected, jobs, source_hashes = retained(sources, judge_name, semester, train)
    assert all(h["prompt_sha256"] == sha(prompt) for h in source_hashes.values())
    full = JudgeDirect(config, domain)
    truncated = JudgeDirect(replace(config, **fields), domain)
    originals = {(t.name, str(s.id)): s for t in full.tasks(task.references, task.test) for s in t.dataset}
    ratings_path = source / ("consolidated/results" if len(sources) > 1 else "results") / judge_name / semester / "ratings.csv"
    ratings = pd.read_csv(ratings_path, dtype={"item": str})
    assert not ratings.duplicated(["model", "bench", "item", "epoch"]).any()
    assert ratings.epoch.eq(1).all() and ratings.semester.eq(semester).all()
    rated = {(f"{r.bench}/{r.model}", r.item): r._asdict() for r in ratings.itertuples(index=False)}
    assert set(rated) <= set(expected), "rating export outside the retained set"
    if thirds:  # thirds conditions judge only the removal analysis's 40 stratified train items per task
        keep = removal_items(task, 10, 0)  # 10 items per difficulty quartile, seed 0
        assert len(keep) == 40  # matched by item id alone, as paired_change.on_items does (math aime2025/aime2026 share ids)
        expected = {key: value for key, value in expected.items() if key[1] in keep}
        rated = {key: value for key, value in rated.items() if key[1] in keep}
    tasks, reused, audit, observed, excluded = [], [], [], set(), []
    for t in truncated.tasks(task.references, task.test):
        assert t.metadata["include_reasoning"] is True and t.metadata["max_length"] == max_length and t.metadata["truncate"] == mode
        assert all(t.metadata[k] == v for k, v in fields.items()), (t.name, fields)
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
            # A shorter input still satisfies the full campaign's context preflight.
            unchanged = sample.input == original.input
            if channels:
                assert not sample.metadata["truncated"] and kept <= full_chars, key  # a channel filter only removes text
                was_cut = False
                if sample.metadata["response_chars"] < MIN_RESPONSE_CHARS:
                    excluded.append({"task": t.name, "item": str(sample.id), "response_chars": sample.metadata["response_chars"]})
                    continue
            elif thirds:
                assert text == cut_text(full_text, max_length, mode) and not unchanged, key  # every transcript loses a third
                assert not sample.metadata["truncated"], key  # the flag records max_length cuts only
                was_cut = False
                if sample.metadata["response_chars"] < MIN_RESPONSE_CHARS:  # as pairwise thirds drop such pairs
                    excluded.append({"task": t.name, "item": str(sample.id), "response_chars": sample.metadata["response_chars"]})
                    continue
            else:
                assert text == cut_text(full_text, max_length, mode), key  # deterministic cut of the full transcript
                was_cut = len(full_text) > max_length
                assert sample.metadata["truncated"] is was_cut and (was_cut or text == full_text), key
                assert unchanged is not was_cut
            observed.add(key)
            reuse = unchanged and key in rated
            audit.append({"task": t.name, "item": str(sample.id), "full_sha256": expected[key],
                          "truncated_sha256": sha(sample.input), "truncated": was_cut, "changed": not unchanged, "reused": reuse,
                          "truncated_chars": sample.metadata["chars"], "full_chars": original.metadata["chars"]})
            if reuse:
                reused.append(rated[key])
            else:
                samples.append(sample.model_dump(mode="json"))
        if samples:
            tasks.append({"name": t.name, "metadata": t.metadata, "samples": samples})
    # A channel condition drops inputs it leaves (nearly) empty, and the renderer drops a transcript it leaves empty entirely.
    missing = set(expected) - observed - {(row["task"], row["item"]) for row in excluded}
    excluded += [{"task": name, "item": item, "response_chars": 0} for name, item in sorted(missing)] if channels else []
    assert channels or thirds or observed == set(expected)
    assert not thirds or observed | {(row["task"], row["item"]) for row in excluded} == set(expected)
    assert len(observed) + len(excluded) == len(expected)
    assert len(reused) + sum(len(t["samples"]) for t in tasks) == len(observed)
    changed = sum(row["changed"] for row in audit)
    summary = {"domain": domain_name, "judge": judge_name, "semester": semester, "condition": cut, "truncate": mode,
               "max_length": max_length, **({"channels": fields["channels"]} if channels else {}),
               "total_ratings": len(observed), "excluded_inputs": len(excluded), "changed_inputs": changed, "reused_ratings": len(reused),
               "new_ratings": sum(len(t["samples"]) for t in tasks), "tasks": len(tasks),
               "references": len(task.references.models), "targets": len(task.truth),
               "truncated_reference_inputs": sum(r["truncated"] for r in audit if r["task"].split("/", 1)[1] in task.references.models.index),
               "truncated_target_inputs": sum(r["truncated"] for r in audit if r["task"].split("/", 1)[1] in task.truth.index),
               "changed_reference_inputs": sum(r["changed"] for r in audit if r["task"].split("/", 1)[1] in task.references.models.index),
               "changed_target_inputs": sum(r["changed"] for r in audit if r["task"].split("/", 1)[1] in task.truth.index)}
    if not changed:
        summary["status"] = "identical to full-transcript condition; no collection"
        print(json.dumps(summary), flush=True)
        return summary
    assert not config.judge.on_runpod, "condition collection is API-hosted; no Runpod judge selected"
    output = root(f"{domain_name}/{judge_name}/{semester}/{cut}") / "inputs" / domain_name / judge_name / semester / cut
    output.mkdir(parents=True, exist_ok=True)
    cfg = draccus.encode(truncated.config)
    assert cfg["max_length"] == max_length and cfg["truncate"] == mode and all(cfg[k] == v for k, v in fields.items())
    cfg["system_prompt"] = "prompt.txt"
    cfg["log_dir"] = "logs/absolute"
    pinned = cfg["judge"]["model_args"]["provider"]["only"]
    limited = any(p.startswith(RATE_LIMITED) for p in pinned)
    cfg["_kwargs"].update(max_connections=8 if limited else 16, max_retries=10, timeout=600,
                          log_model_api=True, log_realtime=True, log_buffer=1, retry_cleanup=False)
    cfg["judge"]["model_args"]["provider"]["require_parameters"] = True
    cfg["judge"]["model_args"]["default_headers"] = {"X-OpenRouter-Metadata": "enabled"}
    payload = {"domain": domain_name, "semester": semester, "judge": judge_name, "condition": cut, "truncate": mode,
               "max_length": max_length, **({"channels": fields["channels"]} if channels else {}), "excluded": excluded,
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
    parser.add_argument("conditions", nargs="+", help="<domain>/<judge>/<semester>/<head|tail|ends>-<max_length chars>, .../channels-<only|drop>-<channel> or .../thirds-<kept>")
    args = parser.parse_args()
    key = lambda s: (s["domain"], s["judge"], s["semester"], s.get("condition") or f"{s.get('truncate')}-{s['max_length']}")
    for condition in args.conditions:  # merged into the root's sweep summary one by one, so a crash keeps earlier conditions
        summary = prepare(*condition.split("/"))
        path = root(condition) / "sweep-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(path.read_text()) if path.exists() else []
        path.write_text(json.dumps([s for s in previous if key(s) != key(summary)] + [summary], indent=2) + "\n")
