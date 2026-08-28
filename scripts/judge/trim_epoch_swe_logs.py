#!/usr/bin/env python3
"""Cut Epoch SWE-bench Verified logs down to the swe-mini items and our message limit.

Epoch runs all 484 items with no turn cap; our swe-mini runs stop at message_limit=100.
Each Epoch log becomes one log holding the swe-mini items only, every sample cut to its
first --message-limit messages, Epoch's scores kept, the pool's task name and per-item
prompt (judging pairs logs of one task and samples of one input), written in the pool
layout <target>/<model>/0/<model>.eval.

Usage:
  uv run python scripts/judge/trim_epoch_swe_logs.py --config configs/epoch_logs_swe.yaml \\
      --items-from data/swe_bench_mini/477dc96c/claude-opus-4.7/0/*.eval
"""

import argparse
import glob
import statistics
from pathlib import Path

import yaml
from inspect_ai.log import EvalLog, read_eval_log, read_eval_log_samples, write_eval_log

from judge.util import extract_score


def trim_log(source: Path, pool_inputs: dict[str, str], message_limit: int, task: str) -> EvalLog:
    """pool_inputs: item id -> the prompt our pool logs carry as sample input; judging pairs
    samples with identical inputs, and Epoch's prompt wording differs from ours."""
    header = read_eval_log(str(source), header_only=True)
    samples = []
    for sample in read_eval_log_samples(str(source), all_samples_required=False, resolve_attachments=True):
        if sample.id not in pool_inputs:
            continue
        sample.input = pool_inputs[sample.id]
        sample.messages = sample.messages[:message_limit]
        sample.events, sample.attachments = [], {}
        samples.append(sample)
    samples.sort(key=lambda sample: str(sample.id))
    correct = [extract_score(sample) for sample in samples]

    spec = header.eval.model_copy(deep=True)
    spec.task = task   # judging pairs logs of the same task only
    spec.dataset.samples = len(samples)
    spec.dataset.sample_ids = [sample.id for sample in samples]
    spec.tags = sorted(set(spec.tags or []) | {"epoch_ai", "imported"})

    results = header.results.model_copy(deep=True)
    results.total_samples = results.completed_samples = len(samples)
    metrics = results.scores[0].metrics
    for name in ("accuracy", "mean"):           # the pool's swe logs report the score as "mean"
        if name in metrics:
            metrics[name].value = statistics.mean(correct)
    for name in ("stderr", "std"):
        if name in metrics:
            metrics[name].value = statistics.stdev(correct) / (len(correct) ** 0.5 if name == "stderr" else 1)

    return EvalLog(version=header.version, status=header.status, eval=spec, plan=header.plan,
                   results=results, stats=header.stats, samples=samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--items-from", required=True, help="pool eval log (glob) supplying the swe-mini items, task name and prompts")
    parser.add_argument("--target", type=Path, default=Path("data/swe_bench_mini/epoch-bash-agent"))
    parser.add_argument("--message-limit", type=int, default=100, help="same cap as our swe-mini runs")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    pool_path = glob.glob(args.items_from)[0]
    task = read_eval_log(pool_path, header_only=True).eval.task
    pool_inputs = {str(sample.id): sample.input
                   for sample in read_eval_log_samples(pool_path, all_samples_required=False)}

    for epoch_name, model in config["models"].items():
        source = Path(config["out_dir"]) / f"{epoch_name}.eval"
        output = args.target / model / "0" / f"{model}.eval"
        if not source.exists():
            print(f"missing {source}")
            continue
        log = trim_log(source, pool_inputs, args.message_limit, task)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(log, str(output))
        print(f"{model}: {len(log.samples)}/{len(pool_inputs)} items, "
              f"accuracy {log.results.scores[0].metrics['accuracy'].value:.2f} -> {output}")


if __name__ == "__main__":
    main()
