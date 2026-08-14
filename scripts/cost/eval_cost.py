"""Per-eval token/cost from collected .eval logs, as a multiplier vs GPQA.

Reads every .eval under a base dir, sums per-sample token usage, joins
model_pool.json prices, and prints mean tokens + $/item per (eval, model),
plus the ×GPQA multiplier. Length pilot only — noisy by design.

    uv run python scripts/cost/eval_cost.py --base data --baseline gpqa
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

# Explicit $/1M (in, out) for models the pool misses or names differently.
PRICE_OVERRIDES = {
    "claude-sonnet-4.5": (3.0, 15.0),
    "glm-4.6": (0.5, 2.0),
}


def load_prices(pool_path: Path) -> dict[str, tuple[float, float]]:
    """model shortname -> ($/1M input, $/1M output)."""
    pool = json.loads(pool_path.read_text())
    rows = pool if isinstance(pool, list) else pool.get("models", pool)
    out = {}
    for m in rows:
        name = (m.get("name") or m.get("id") or m.get("slug") or "").split("/")[-1]
        pin, pout = m.get("price_in"), m.get("price_out")
        if name and pin is not None and pout is not None:
            out[name] = (float(pin), float(pout))
    return out


def short_model(slug: str) -> str:
    return slug.removeprefix("openrouter/").split("/")[-1]


def sample_tokens(sample) -> dict[str, int]:
    """Sum ModelUsage across all models used in one sample."""
    tot = defaultdict(int)
    for usage in (sample.model_usage or {}).values():
        tot["input"] += usage.input_tokens or 0
        tot["output"] += usage.output_tokens or 0
        tot["reasoning"] += getattr(usage, "reasoning_tokens", 0) or 0
        tot["total"] += usage.total_tokens or 0
    return {k: tot[k] for k in ("input", "output", "reasoning", "total")}


def sample_correct(sample) -> int:
    """1 if any scorer marked the sample correct."""
    for sc in (sample.scores or {}).values():
        v = sc.value
        if v in ("C", 1, True):
            return 1
        if isinstance(v, float) and v >= 0.5:
            return 1
    return 0


def write_pivot_csv(path, rows, base_cost, models):
    """Matrix CSV: row1 = model names (each spanning 3 cols), row2 = metric
    labels, then one row per eval. Cell triplet = $/item, ×GPQA, n."""
    by_cell = {(r["eval"], r["model"]): r for r in rows}
    if not models:
        models = sorted({r["model"] for r in rows}, key=lambda m: base_cost.get(m) or 9e9)
    evals = sorted({r["eval"] for r in rows},
                   key=lambda e: max(r["cost"] or 0 for r in rows if r["eval"] == e))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        h1 = [""]
        for g in ["average"] + models:
            h1 += [g, "", ""]
        w.writerow(h1)
        w.writerow(["eval"] + ["$/item", "×GPQA", "n"] * (len(models) + 1))
        for e in evals:
            per_model, prices, mults, ns = [], [], [], []
            for m in models:
                r = by_cell.get((e, m))
                if not r:
                    per_model += ["", "", ""]
                    continue
                b = base_cost.get(m)
                if r["cost"] is None:
                    per_model += ["refused", "", r["n"]]
                    ns.append(r["n"])
                    continue
                mult = r["cost"] / b if b else None
                per_model += [f"{r['cost']:.4f}", f"{mult:.1f}" if mult is not None else "", r["n"]]
                prices.append(r["cost"])
                ns.append(r["n"])
                if mult is not None:
                    mults.append(mult)
            # first group = average across models (mean $/item, mean ×GPQA, total n)
            avg = [f"{sum(prices) / len(prices):.4f}" if prices else "",
                   f"{sum(mults) / len(mults):.1f}" if mults else "",
                   sum(ns) if ns else ""]
            w.writerow([e] + avg + per_model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", default=["data"], help="one or more log roots")
    ap.add_argument("--pool", type=Path, default=Path("model-sampler/model_pool.json"))
    ap.add_argument("--baseline", default="gpqa_diamond", help="eval name used as ×1.0 denominator")
    ap.add_argument("--newer-min", type=float, default=0,
                    help="only count logs whose file mtime is within the last N minutes (isolates a session)")
    ap.add_argument("--csv", type=Path, default=None, help="also write a pivoted matrix CSV here")
    ap.add_argument("--models", default=None, help="comma-separated model order/filter for the CSV columns")
    args = ap.parse_args()

    cutoff = time.time() - args.newer_min * 60 if args.newer_min else 0

    prices = load_prices(args.pool) | PRICE_OVERRIDES

    # group per (eval task, model): list of per-sample token dicts.
    # dedup by sample id so overlapping re-run logs don't double-count.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen_ids: set = set()
    seen = set()
    infos = [i for b in args.base for i in list_eval_logs(str(b))]
    for info in infos:
        if info.name in seen:
            continue
        seen.add(info.name)
        if cutoff:
            try:
                if os.path.getmtime(info.name.replace("file://", "")) < cutoff:
                    continue
            except OSError:
                pass
        log = read_eval_log(info)
        if not log.samples:
            continue
        task = log.eval.task.split("/")[-1]
        diff = (log.eval.task_args or {}).get("difficulty")
        if diff:
            task = f"{task}_{diff}"
        model = short_model(log.eval.model)
        for s in log.samples:
            tok = sample_tokens(s)
            if getattr(s, "error", None) or tok["total"] == 0:
                continue  # errored / no-usage sample (e.g. refusal, sandbox fail)
            sid = (task, model, str(s.id))
            if sid in seen_ids:
                continue  # same capsule/instance from an overlapping re-run log
            seen_ids.add(sid)
            groups[(task, model)].append({**tok, "correct": sample_correct(s)})

    # aggregate
    rows = []
    for (task, model), samples in groups.items():
        n = len(samples)
        mean = {k: sum(s[k] for s in samples) / n for k in ("input", "output", "reasoning", "total")}
        acc = sum(s["correct"] for s in samples) / n
        pin, pout = prices.get(model, (None, None))
        cost = None if pin is None else mean["input"] * pin / 1e6 + mean["output"] * pout / 1e6
        rows.append({"eval": task, "model": model, "n": n, "acc": acc, **mean, "cost": cost})

    # baseline cost per model for the multiplier
    base_cost = {r["model"]: r["cost"] for r in rows if r["eval"] == args.baseline}

    rows.sort(key=lambda r: (r["model"], r["cost"] or 0))
    hdr = f"{'eval':<16}{'model':<18}{'n':>3}{'acc':>6}{'in':>10}{'out':>9}{'reason':>8}{'$/item':>9}{'×gpqa':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        b = base_cost.get(r["model"])
        mult = f"{r['cost']/b:.1f}x" if (b and r["cost"]) else "-"
        cost = f"${r['cost']:.4f}" if r["cost"] is not None else "n/a"
        print(f"{r['eval']:<16}{r['model']:<18}{r['n']:>3}{r['acc']*100:>5.0f}%{r['input']:>10.0f}{r['output']:>9.0f}"
              f"{r['reasoning']:>8.0f}{cost:>9}{mult:>7}")
    total = sum(r["cost"] * r["n"] for r in rows if r["cost"] is not None)
    print("-" * len(hdr))
    print(f"TOTAL spend (priced rows, {sum(r['n'] for r in rows)} samples): ${total:.2f}")

    if args.csv:
        model_filter = args.models.split(",") if args.models else None
        if model_filter:
            rows = [r for r in rows if r["model"] in model_filter]
        write_pivot_csv(args.csv, rows, base_cost, model_filter)
        print(f"wrote pivot CSV -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
