"""Build audited publication copies of a truncation or channel condition without changing source collection logs or calling models."""
import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from inspect_ai import score as rescore
from inspect_ai.log import read_eval_log, read_eval_log_sample, write_eval_log
from inspect_ai.model import ModelUsage

from configs import DirectJudgeConfig
from configs.domain import DomainConfig
from judge.scorer import direct_judge_scorer, parse_direct_judge_output
from protocols.direct_judge import JudgeDirect, from_logs
from protocols.direct_judge.scoring import calibrate, mean_scores
from capability.evaluate import score
from challenges.truncation.run import root, tag

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/audit/logs"))
from eval_edits import stamp_edit


CUT_WORDS = {"head": "first {n}", "tail": "last {n}", "ends": "first and last {n} ('...' between)"}
THIRD_NAMES = {"1": "first", "2": "middle", "3": "last"}
CHANNEL_NAMES = {"reasoning": "reasoning", "text": "assistant text", "calls": "tool calls", "outputs": "tool outputs",
                 "tools": "tool calls and tool outputs"}


def thirds_words(mode):
    """Plain description of a thirds mode: thirds-23 -> "without the first third", thirds-2 -> "only the middle third"."""
    kept = mode.removeprefix("thirds-")
    if len(kept) == 1:
        return f"only the {THIRD_NAMES[kept]} third"
    return f"without the {THIRD_NAMES[({'1', '2', '3'} - set(kept)).pop()]} third"


def channel_words(spec):
    """Plain description of a channel spec: only-calls -> "only the tool calls", drop-outputs -> "everything but the tool outputs"."""
    mode, _, channel = spec.partition("-")
    return ("only the " if mode == "only" else "everything but the ") + CHANNEL_NAMES[channel]


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def content(samples):
    return sha(json.dumps([{**sample.model_dump(mode="json", include={"id", "epoch", "input", "target", "messages", "output", "metadata"}),
                            "model_events": [e.model_dump(mode="json") for e in sample.events if e.event == "model"]}
                           for sample in samples], sort_keys=True))


def rebuild(log):
    log.eval.dataset.samples = len(log.samples)
    log.eval.dataset.sample_ids = [s.id for s in log.samples]
    before = content(log.samples)
    model = log.eval.model
    log = rescore(log, [direct_judge_scorer()], model="mockllm/model", action="overwrite", display="none")
    assert log.eval.model == model and content(log.samples) == before
    valid = sum(s.scores["direct_judge_scorer"].answer is not None for s in log.samples)
    log.results.total_samples = log.results.completed_samples = log.results.logged_samples = len(log.samples)
    summary = log.results.scores[0]
    summary.scored_samples, summary.unscored_samples = valid, len(log.samples) - valid
    summary.metrics["answer_rate"].value = valid / len(log.samples)
    usage = defaultdict(list)
    for sample in log.samples:
        for name, value in (sample.model_usage or {}).items():
            usage[name].append(value)
    log.stats.model_usage = {name: ModelUsage(**{field: (None if any(getattr(v, field) is None for v in values)
                                                              else sum(getattr(v, field) for v in values))
                                                   for field in ModelUsage.model_fields})
                            for name, values in usage.items()}
    return log


def prepare(condition):
    """`condition`: <domain>/<judge>/<semester>/<head|tail|ends>-<max_length> or .../channels-<spec>."""
    ROOT = root(condition)
    inputs = ROOT / "inputs" / condition
    audit = json.loads((ROOT / "audit" / f"{tag(condition)}.json").read_text())
    assert all(not audit[key] for key in ("blocking", "missing_new_keys", "score_repairs", "recollect", "review_flags")), "Audit unresolved"
    assert sha((inputs / "job.json").read_bytes()) == audit["job_sha256"], "Audited job changed"
    assert sha((inputs / "prompt.txt").read_bytes()) == audit["prompt_sha256"], "Audited prompt changed"
    payload = json.loads((inputs / "job.json").read_text())
    domain, judge, max_length, mode = payload["domain"], payload["judge"], payload["max_length"], payload["truncate"]
    channels = payload.get("channels")
    thirds = mode.startswith("thirds-")
    label = f"channels_{channels}" if channels else f"truncated_{mode}" if thirds else f"truncated_{mode}_{max_length}_chars"
    view = {"channels": channels} if channels else {}
    if thirds:
        shown = thirds_words(mode)
        judgment_input = (f"Each rendered transcript split into three equal character spans and shown {shown} (JudgeConfig.truncate={mode}, "
                          f"judge.dataset.cut; each run of dropped spans replaced by '\\n...\\n'); calibration and target transcripts alike; "
                          f"subject reasoning visible; only the 40 removal-analysis train items per task (10 per difficulty quartile, "
                          f"stratified_items seed 0); inputs left under 20 characters of model-written text excluded "
                          f"({len(payload['excluded'])}); retained set and exclusions otherwise identical to the full-transcript campaign.")
        unchanged_reason = "unused: every thirds input differs from the full transcript"
        edit_reason = f"Label the {mode} condition ({shown}) and recompute score summaries."
    elif channels:
        shown = channel_words(channels)
        judgment_input = (f"Each transcript rendered with {shown} (JudgeConfig.channels={channels}); inputs the filter leaves unchanged "
                          f"reuse their full-transcript ratings; inputs left under 20 characters of model-written text are excluded "
                          f"({len(payload['excluded'])}, listed in excluded_inputs); calibration and target transcripts alike; subject reasoning "
                          f"visible unless the filter removes it; retained set otherwise identical to the full-transcript campaign.")
        unchanged_reason = f"Every retained sample renders identically with {shown}; ratings reused from the full-transcript archive and relabelled."
        edit_reason = f"Label the {channels} channel condition, incorporate audited identical-input rating reuse, and recompute score summaries."
    else:
        CUT_WORDS = {k: v.format(n=max_length if k != "ends" else max_length // 2) for k, v in globals()["CUT_WORDS"].items()}
        judgment_input = f"Each transcript longer than {max_length} characters cut to its {CUT_WORDS[mode]} before rendering (JudgeConfig.max_length, truncate={mode}); shorter transcripts unchanged and their full-transcript ratings reused; calibration and target transcripts alike; subject reasoning visible; retained set and exclusions identical to the full-transcript campaign."
        unchanged_reason = f"Every retained sample is shorter than {max_length} characters; ratings reused from the full-transcript archive and relabelled."
        edit_reason = f"Label {mode} truncation at {max_length} characters, incorporate audited identical-input rating reuse, and recompute score summaries."
    route = {"judge_config": payload["judge_config"], "model_config": payload["model_config"]}
    failures = {(row["task"], row["item"]) for row in audit.get("declared_failures", [])}
    cfg = DirectJudgeConfig.from_cli(payload["config"])
    cfg.system_prompt, cfg.log_dir = str(inputs / "prompt.txt"), REPO / "logs" / "absolute"
    assert cfg.include_reasoning and cfg.max_length == max_length and cfg.truncate == mode and (cfg.channels or None) == channels
    protocol = JudgeDirect(cfg, DomainConfig.from_cli(REPO / f"configs/domain/{domain}.yml"))
    destination = protocol.log_dir(payload["semester"])
    stage = ROOT / "publication" / destination.relative_to(REPO / "logs")
    assert not stage.exists(), "Publication stage already exists; inspect rather than overwrite"
    stage.mkdir(parents=True)
    expected = defaultdict(set)
    for record in payload["audit"]:
        expected[record["task"]].add(record["item"])
    reuse = defaultdict(list)
    for row in audit["reused_provenance"]:
        source = Path(row["file"])
        assert sha(source.read_bytes()) == row["sha256"]
        sample = read_eval_log_sample(source, row["item"], 1, resolve_attachments=True)
        sample.metadata = dict(sample.metadata or {}) | {"reused_from_full_transcript": True,
            "reused_source_archive": source.name, "reused_source_sha256": row["sha256"]}
        reuse[row["task"]].append(sample)
    records, seen = [], set()
    for record in audit["files"]:
        source = Path(record["path"])
        assert sha(source.read_bytes()) == record["sha256"]
        log = read_eval_log(source, resolve_attachments=True)
        assert log.eval.task not in seen
        seen.add(log.eval.task)
        log.samples.extend(reuse.get(log.eval.task, []))
        log.samples.sort(key=lambda s: str(s.id))
        assert {str(s.id) for s in log.samples} == expected[log.eval.task]
        assert len(log.samples) == len(expected[log.eval.task])
        for sample in log.samples:
            answer, _ = parse_direct_judge_output(sample.output.completion)
            failed = (log.eval.task, str(sample.id)) in failures
            assert (answer is None) == failed
            assert sample.output.stop_reason != "max_tokens" or failed
        log.eval.metadata = dict(log.eval.metadata or {}) | {"condition_label": label,
            "include_reasoning": True, "max_length": max_length, "truncate": mode, **view, "condition_hash": destination.name,
            "reused_sample_count": len(reuse.get(log.eval.task, []))}
        log.eval.tags = sorted({*(log.eval.tags or []), label, "protocol:direct"})
        log = rebuild(log)
        stamp_edit(log, tool="prepare_publication.py", source=source.name,
            reason=edit_reason,
            source_sha256=record["sha256"], samples_changed=0,
            reused_samples_added=len(reuse.get(log.eval.task, [])), transcripts_modified=False)
        fingerprint = content(log.samples)
        target = stage / source.name
        write_eval_log(log, target)
        reread = read_eval_log(target, resolve_attachments=True)
        assert content(reread.samples) == fingerprint
        assert reread.eval.metadata["max_length"] == max_length
        assert reread.results.scores[0].scored_samples == sum(s.scores["direct_judge_scorer"].answer is not None for s in reread.samples)
        assert sha(source.read_bytes()) == record["sha256"]
        records.append({"file": target.name, "sha256": sha(target.read_bytes()),
                        "task": log.eval.task, "samples": len(log.samples),
                        "reused": len(reuse.get(log.eval.task, []))})
    for task_name in sorted(set(expected) - seen):
        # Every retained sample of this task is shorter than the limit, so the condition carries
        # the full-transcript archive's judgments, relabelled and stamped as reused.
        sources = [Path(row["file"]) for row in audit["reused_provenance"] if row["task"] == task_name]
        source = max(set(sources), key=sources.count)  # the main task log; targeted retry logs hold a few samples
        log = read_eval_log(source, resolve_attachments=True)
        assert log.eval.task == task_name
        log.samples = sorted(reuse[task_name], key=lambda s: str(s.id))
        assert {str(s.id) for s in log.samples} == expected[task_name]
        assert all(parse_direct_judge_output(s.output.completion)[0] is not None for s in log.samples)
        log.eval.metadata = dict(log.eval.metadata or {}) | {"condition_label": label,
            "include_reasoning": True, "max_length": max_length, "truncate": mode, **view, "condition_hash": destination.name,
            "reused_sample_count": len(log.samples), "reused_from_full_transcript_archive": source.name,
            "input_manifest_sha256": audit["job_sha256"]}
        log.eval.tags = sorted({*(log.eval.tags or []), label, "protocol:direct"})
        log = rebuild(log)
        stamp_edit(log, tool="prepare_publication.py", source=source.name,
            reason=unchanged_reason,
            source_sha256=sha(source.read_bytes()), samples_changed=0, reused_source_archives=sorted({p.name for p in sources}),
            reused_samples_added=len(log.samples), transcripts_modified=False)
        fingerprint = content(log.samples)
        target = stage / source.name
        write_eval_log(log, target)
        reread = read_eval_log(target, resolve_attachments=True)
        assert content(reread.samples) == fingerprint and reread.eval.metadata["max_length"] == max_length
        records.append({"file": target.name, "sha256": sha(target.read_bytes()), "task": task_name,
                        "samples": len(log.samples), "reused": len(log.samples), "fully_reused": True})
        seen.add(task_name)
    assert seen == set(expected)
    ratings = from_logs(stage)
    assert len(ratings) == sum(len(ids) for ids in expected.values()) - len(failures)
    assert not ratings.duplicated(["model", "bench", "item", "epoch"]).any()
    refs, truth = pd.Series(payload["references"]), pd.Series(payload["truth"])
    predictions = calibrate(ratings, refs).reindex(truth.index)
    assert channels or predictions.notna().all()  # a channel filter can leave a model no retained input
    ratings.to_csv(stage / "ratings.csv", index=False)
    pd.DataFrame({"semester": payload["semester"], "model": truth.index,
                  "theta_hat": predictions.to_numpy(), "theta": truth.to_numpy()}).to_csv(stage / "forecasts.csv", index=False)
    pd.DataFrame([score(predictions, truth)]).to_csv(stage / "metrics.csv", index=False)
    mean_scores(ratings).rename("mean_rating").to_frame().join(refs.rename("theta"), how="inner").to_csv(stage / "reference-scores.csv")
    (stage / "prompt.txt").write_text((inputs / "prompt.txt").read_text())
    archive_records = []
    consolidation_path = ROOT / f"consolidation-{tag(condition)}.json"
    if consolidation_path.exists():
        consolidation = json.loads(consolidation_path.read_text())
        archive_root = ROOT / "publication/legacy" / destination.relative_to(REPO / "logs")
        sources = [(Path(row["original"]), Path("superseded") / Path(row["original"]).name)
                   for row in consolidation["changed_logs"]]
        sources += [(path, Path("recollections") / path.relative_to(ROOT / "recollections" / condition))
                    for path in (ROOT / "recollections" / condition).rglob("*.eval")]
        for source, relative in sources:
            target = archive_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert sha(source.read_bytes()) == sha(target.read_bytes())
            archive_records.append({"source": str(source), "published_path": str(target.relative_to(ROOT / "publication")),
                                    "sha256": sha(target.read_bytes())})
    manifest = {"condition_label": label, "include_reasoning": True, "max_length": max_length, "truncate": mode, **view,
        "judge": judge, "judge_config": route["judge_config"], "model_config": route["model_config"],
        "effective_config": payload["config"], "domain": domain, "semester": payload["semester"],
        "source_snapshot": payload["source_snapshot"], "condition_hash": destination.name,
        "input_manifest_sha256": sha((inputs / "job.json").read_bytes()),
        "judgment_input": judgment_input,
        "truncated_inputs": sum(row["truncated"] for row in payload["audit"]),
        **({"changed_inputs": sum(row["changed"] for row in payload["audit"]), "excluded_inputs": payload["excluded"],
            "missing_forecasts": predictions.index[predictions.isna()].tolist()} if channels or thirds else {}),
        "calibration": "previous-semester references only", "ratings": len(ratings),
        "retained_judgments": sum(len(ids) for ids in expected.values()), "declared_failures": audit.get("declared_failures", []),
        "archived_attempts": archive_records,
        "reused_ratings": len(payload["reused_ratings"]), "files": records,
        "audit": {k: audit[k] for k in ("audited_at", "counts", "providers", "stop_reasons", "cache_provenance", "reused_provenance")}}
    (stage / "condition.json").write_text(json.dumps(manifest, indent=2) + "\n")
    heading = (f"# Direct judging: transcripts showing {shown}\n\n"
               f"{judge}, {domain}, {payload['semester']}; `channels: {channels}` (same filter on calibration and target transcripts). "
               f"{len(ratings)} audited ratings on the full-transcript campaign's retained set less {len(payload['excluded'])} inputs the filter leaves under 20 characters, "
               f"including {len(payload['reused_ratings'])} verified identical-input reuses of inputs the filter leaves unchanged.\n\n") if channels else (
               f"# Direct judging: transcripts {'showing ' if shown.startswith('only') else ''}{shown}\n\n"
               f"{judge}, {domain}, {payload['semester']}; `truncate: {mode}` (three equal character spans of the rendered transcript, same cut on calibration and target transcripts). "
               f"{len(ratings)} audited ratings on the full-transcript campaign's retained set restricted to the 40 removal-analysis train items per task.\n\n") if thirds else (
               f"# Direct judging: transcripts truncated to their {CUT_WORDS[mode]} characters\n\n"
               f"{judge}, {domain}, {payload['semester']}; `max_length: {max_length}`, `truncate: {mode}` (characters kept per transcript, same limit on calibration and target transcripts). "
               f"{len(ratings)} audited ratings on the full-transcript campaign's retained set, including {len(payload['reused_ratings'])} verified identical-input reuses of inputs shorter than the limit.\n\n")
    (stage / "README.md").write_text(heading +
        "The condition hash and log metadata separate it from the full-transcript run. "
        f"`condition.json` records the config, {'channel' if channels else 'thirds' if thirds else 'truncation'} label, source revisions, provider audit, and reuse provenance.\n\n"
        f"All inputs, final ratings, provider pins, cache origins, and sample identities were checked. {len(failures)} explicitly recorded unusable ratings remain missing after two same-config recollections; see `declared_failures` in `condition.json`. "
        "Publication copies carry edit stamps; original transcripts and API traces are preserved.\n")
    result = {"condition": condition, "domain": domain, "judge": judge, "stage": str(stage), "destination": str(destination),
              "logs": len(records), "ratings": len(ratings), "retained_judgments": sum(len(ids) for ids in expected.values()),
              "declared_failures": len(failures), "archived_files": len(archive_records), "condition_hash": destination.name}
    (ROOT / f"publication-{tag(condition)}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="<domain>/<judge>/<semester>/<head|tail|ends>-<max_length> or .../channels-<spec>")
    prepare(parser.parse_args().condition)
