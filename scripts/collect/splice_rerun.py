"""Merge re-run samples into a full rollout log.

The new log is the canonical one with every sample a re-run log redid replaced by the
re-run version (same id and epoch; a sample the canonical log lacks is appended), and the
header metrics recomputed from the samples.
A re-run sample that itself failed (errored, empty, cut at max_tokens) is not taken: the
original stays and the id is listed as still failing. Inputs are never modified.

Usage:
  uv run python scripts/collect/splice_rerun.py <canonical.eval> <rerun.eval>... --out <new.eval>
"""

import argparse
import math
import statistics
from pathlib import Path

from inspect_ai.log import (
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalSample,
    EvalSampleReductions,
    EvalSampleScore,
    EvalScore,
    read_eval_log,
    write_eval_log,
)

from judge.util import extract_score


def failure_kind(sample: EvalSample) -> str | None:
    """Why a rollout sample is unusable: errored, empty or truncated. None if it is fine."""
    if sample.error is not None:
        return "errored"
    output = sample.output
    if output is None or not output.choices:
        return "empty"
    if output.stop_reason == "max_tokens":
        return "truncated"
    # An agent turn that ends in a tool call has no text; that is not a failure.
    if not (output.completion or "").strip() and output.stop_reason != "tool_calls":
        return "empty"
    return None


def model_name(model: str) -> str:
    """'openai-api/runpod/Qwen/Qwen3-32B' and 'openrouter/qwen/qwen3-32b' are the same model."""
    return model.split("/")[-1].lower()


def splice(canonical: EvalLog, reruns: list[tuple[str, EvalLog]]) -> dict[tuple, str]:
    """Replace canonical samples with the re-run ones that did not fail, in place.

    Returns {(id, epoch): rerun log path} for the samples replaced. A later re-run of
    the same sample wins over an earlier one.
    """
    samples = list(canonical.samples or [])
    position = {(sample.id, sample.epoch): i for i, sample in enumerate(samples)}
    replaced = {}
    for path, rerun in reruns:
        # Same model may be served by another provider in the re-run (pod vs OpenRouter).
        if rerun.eval.task != canonical.eval.task or model_name(rerun.eval.model) != model_name(canonical.eval.model):
            raise ValueError(
                f"{path} is {rerun.eval.task} / {rerun.eval.model}, "
                f"the canonical log is {canonical.eval.task} / {canonical.eval.model}"
            )
        for sample in rerun.samples or []:
            key = (sample.id, sample.epoch)
            if failure_kind(sample):
                continue
            if key in position:
                samples[position[key]] = sample
            else:  # a sample the canonical run never wrote (dropped by a resume): appended
                position[key] = len(samples)
                samples.append(sample)
            replaced[key] = path
    canonical.samples = samples
    return replaced


def recompute_metric(name: str, values: list[float]) -> float:
    """Same formulas as inspect's accuracy/mean/std/stderr metrics."""
    n = len(values)
    if name in ("accuracy", "mean"):
        return sum(values) / n if n else 0.0
    if name == "std":
        return statistics.stdev(values) if n > 1 else 0.0
    if name == "stderr":
        return statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.0
    raise ValueError(f"cannot recompute metric {name!r}")


def metric_names(scorer_spec) -> list[str]:
    """Metric names a scorer registered, e.g. 'inspect_ai/mean' -> 'mean'."""
    if not isinstance(scorer_spec.metrics, list):
        raise ValueError(f"scorer {scorer_spec.name}: unsupported metrics spec {scorer_spec.metrics!r}")
    return [definition.name.rsplit("/", 1)[-1] for definition in scorer_spec.metrics]


def recompute_results(log: EvalLog) -> None:
    """Rebuild the header metrics and per-sample reductions from the samples.

    Scores are the 0/1 correctness judge.util.extract_score reads, which is what the
    metrics of every benchmark we collect are computed over.
    """
    assert log.eval.config.epochs in (None, 1), "only single-epoch logs are spliced"  # ponytail: no epoch reducer
    samples = log.samples or []
    if log.results is None:
        # A run that died before scoring has no results block; the scorer spec still says
        # which metrics it would have reported.
        log.results = EvalResults(scores=[
            EvalScore(name=spec.name, scorer=spec.name,
                      metrics={name: EvalMetric(name=name, value=0.0) for name in metric_names(spec)})
            for spec in log.eval.scorers or []
        ])
    log.reductions = []
    for score in log.results.scores:
        scored = [sample for sample in samples if sample.scores and score.name in sample.scores]
        values = [float(extract_score(sample)) for sample in scored]
        for metric in score.metrics.values():
            metric.value = recompute_metric(metric.name, values)
        if score.scored_samples is not None:
            score.scored_samples, score.unscored_samples = len(scored), len(samples) - len(scored)
        log.reductions.append(EvalSampleReductions(scorer=score.name, samples=[
            EvalSampleScore(**sample.scores[score.name].model_dump(), sample_id=sample.id) for sample in scored
        ]))
    log.results.total_samples = len(samples)
    log.results.completed_samples = sum(sample.error is None for sample in samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("canonical", type=Path, help="full log to fill")
    parser.add_argument("reruns", type=Path, nargs="+", help="partial logs from collect --sample-id")
    parser.add_argument("--out", type=Path, required=True, help="new log to write (must not exist)")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; not overwriting")

    log = read_eval_log(str(args.canonical))
    replaced = splice(log, [(str(path), read_eval_log(str(path))) for path in args.reruns])
    recompute_results(log)
    still_failing = {(s.id, s.epoch): kind for s in log.samples if (kind := failure_kind(s))}

    log.status, log.error = "success", None
    log.eval.config.sample_id = None
    log.eval.metadata = (log.eval.metadata or {}) | {
        "spliced_from": [{"id": id, "epoch": epoch, "rerun_log": path} for (id, epoch), path in replaced.items()],
        "still_failing": [{"id": id, "epoch": epoch, "kind": kind} for (id, epoch), kind in still_failing.items()],
    }
    write_eval_log(log, str(args.out))

    written = read_eval_log(str(args.out))
    assert {(s.id, s.epoch) for s in written.samples} == {(s.id, s.epoch) for s in log.samples}, "read-back mismatch"
    assert written.eval.config.sample_id is None and written.status == "success"

    metrics = {m.name: round(m.value, 4) for score in log.results.scores for m in score.metrics.values()}
    print(f"{args.out}: {len(log.samples)} samples, {len(replaced)} replaced, {metrics}")
    for (id, epoch), kind in still_failing.items():
        print(f"  STILL FAILING {id} (epoch {epoch}): {kind}")
    return 1 if still_failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
