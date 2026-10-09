"""Per-run report for a collection sweep: accuracy, length, cost, and reasoning presence.

Complements scripts/audit/logs/validate_data.py, which checks structure (repeats, duplicates,
completeness). This one checks that a run says what it should:

    uv run python scripts/audit/logs/report_collection.py data/hle data/apex_2025 [--since '2026-08-19 22:20']

- accuracy: to compare against what the model is known to score. The model column is
  the data directory, so a rescored copy is listed separately from its original.
- out/sample, cost: what the run spent. Prices come from OpenRouter's live model list
  (cached in data/openrouter_endpoints_*.json is not used; the API is queried once).
- reasoning: share of samples with reasoning content. A model whose config asks for
  reasoning (`extra_body.reasoning.enabled`) and returns none has a broken pin, and its
  transcripts are useless for the reasoning condition.
"""

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples

from configs.paths import collection_logs, is_successful_eval


def prices() -> dict[str, tuple[float, float]]:
    """(prompt, completion) USD per token, by OpenRouter model id."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)["data"]
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in data}


def reasoning_share(path: Path, limit: int) -> float | None:
    """Share of the first `limit` samples whose assistant turns carry reasoning."""
    seen = with_reasoning = 0
    for sample in read_eval_log_samples(str(path), all_samples_required=False):
        if seen >= limit:
            break
        seen += 1
        for message in sample.messages:
            content = getattr(message, "content", None)
            if getattr(message, "role", None) != "assistant":
                continue
            if isinstance(content, list) and any(c.type == "reasoning" for c in content):
                with_reasoning += 1
                break
            if "<think>" in str(content):
                with_reasoning += 1
                break
    return with_reasoning / seen if seen else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--since", help="only logs modified after this (e.g. '2026-08-19 22:20')")
    parser.add_argument("--samples", type=int, default=25, help="samples read per log for the reasoning share")
    args = parser.parse_args()

    cutoff = 0.0
    if args.since:
        import datetime
        cutoff = datetime.datetime.fromisoformat(args.since).timestamp()

    price = prices()
    rows, totals = [], defaultdict(float)
    for root in args.roots:
        for path in collection_logs(root):
            if path.stat().st_mtime < cutoff or not is_successful_eval(path):
                continue
            log = read_eval_log(str(path), header_only=True)
            metrics = log.results.scores[0].metrics if log.results and log.results.scores else {}
            accuracy = next((m.value for k, m in metrics.items() if k in ("accuracy", "mean", "score")), None)
            model = log.eval.model
            usage = (log.stats.model_usage or {}).get(model)
            out_tokens = usage.output_tokens if usage else 0
            samples = log.results.completed_samples if log.results else 0
            cost = 0.0
            for used_model, u in (log.stats.model_usage or {}).items():
                p_in, p_out = price.get(used_model.split("/", 1)[-1], (0.0, 0.0))
                cost += u.input_tokens * p_in + u.output_tokens * p_out
            rows.append((
                path.parts[-5], path.parent.parent.name, path.parent.name, samples,
                accuracy, out_tokens / samples if samples else 0,
                reasoning_share(path, args.samples), cost,
            ))
            totals["cost"] += cost
            totals["out"] += out_tokens

    header = f"{'bench':16s}{'model':52s}{'rep':4s}{'n':>5s}{'acc':>7s}{'out/smp':>9s}{'reas':>6s}{'cost$':>8s}"
    print(header)
    print("-" * len(header))
    for bench, model, rep, n, acc, out, reas, cost in sorted(rows):
        acc_s = f"{acc:.3f}" if acc is not None else "-"
        reas_s = f"{reas:.0%}" if reas is not None else "-"
        print(f"{bench:16s}{model:52s}{rep:4s}{n:5d}{acc_s:>7s}{out:9.0f}{reas_s:>6s}{cost:8.2f}")
    print(f"\n{len(rows)} logs | {totals['out']/1e6:.1f}M output tokens | ${totals['cost']:.2f}")


if __name__ == "__main__":
    main()
