"""Validate collected transcripts and dump example excerpts.

Default: per (eval, model), a validity summary — sample count, mean messages,
mean tool calls, tool-error rate, % scored, % correct. Flags anything that
looks empty / all-errored / unscored.

    uv run python scripts/cost/inspect_transcripts.py --base data data/gpqa --newer-min 700
    uv run python scripts/cost/inspect_transcripts.py --dump gdm_intercode_ctf glm-4.6
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log


def text_of(msg) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    parts = []
    for b in c or []:
        t = getattr(b, "text", None) or getattr(b, "reasoning", None)
        if t:
            parts.append(t)
    return " ".join(parts)


def sample_stats(sample) -> dict:
    msgs = sample.messages or []
    asst = [m for m in msgs if m.role == "assistant"]
    tool_msgs = [m for m in msgs if m.role == "tool"]
    tool_calls = sum(len(getattr(m, "tool_calls", None) or []) for m in asst)
    tool_errs = sum(1 for m in tool_msgs if getattr(m, "error", None))
    scored = bool(sample.scores)
    correct = False
    for sc in (sample.scores or {}).values():
        v = sc.value
        if v in ("C", 1, True) or (isinstance(v, float) and v >= 0.5):
            correct = True
    final = text_of(asst[-1]) if asst else ""
    return {
        "msgs": len(msgs),
        "asst": len(asst),
        "tool_calls": tool_calls,
        "tool_msgs": len(tool_msgs),
        "tool_errs": tool_errs,
        "scored": scored,
        "correct": correct,
        "final_len": len(final),
        "errored": bool(getattr(sample, "error", None)),
    }


def iter_samples(bases, newer_min):
    cutoff = time.time() - newer_min * 60 if newer_min else 0
    seen = set()
    for b in bases:
        for info in list_eval_logs(str(b)):
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
            model = log.eval.model.removeprefix("openrouter/").split("/")[-1]
            for s in log.samples:
                yield task, model, s


def dedup(bases, newer_min):
    """One stats dict per (task, model, sample id) — kill overlapping re-run
    logs. Prefer a scored, non-errored sample over an incomplete one."""
    best = {}
    for task, model, s in iter_samples(bases, newer_min):
        st = sample_stats(s)
        key = (task, model, str(s.id))
        rank = (st["scored"], not st["errored"], st["msgs"])
        if key not in best or rank > best[key][1]:
            best[key] = (st, rank)
    out = defaultdict(list)
    for (task, model, _), (st, _) in best.items():
        out[(task, model)].append(st)
    return out


def summary(bases, newer_min):
    groups = dedup(bases, newer_min)
    hdr = (f"{'eval':<24}{'model':<16}{'n':>3}{'msgs':>6}{'tool_calls':>11}"
           f"{'tool_err%':>10}{'scored%':>8}{'correct%':>9}{'flag':>8}")
    print(hdr)
    print("-" * len(hdr))
    for (task, model), ss in sorted(groups.items()):
        n = len(ss)
        mean = lambda k: sum(s[k] for s in ss) / n
        tc = sum(s["tool_calls"] for s in ss)
        te = sum(s["tool_errs"] for s in ss)
        err_rate = te / tc * 100 if tc else 0
        scored = sum(s["scored"] for s in ss) / n * 100
        correct = sum(s["correct"] for s in ss) / n * 100
        # valid = has an assistant turn, is scored, didn't error. (Agentic
        # samples legitimately end on a tool call with no final text, so we do
        # NOT require final_len > 0.)
        bad = [s for s in ss if s["msgs"] < 2 or s["asst"] == 0 or not s["scored"] or s["errored"]]
        flag = "OK" if not bad else f"{len(bad)}BAD"
        print(f"{task:<24}{model:<16}{n:>3}{mean('msgs'):>6.0f}{mean('tool_calls'):>11.1f}"
              f"{err_rate:>9.0f}%{scored:>7.0f}%{correct:>8.0f}%{flag:>8}")


def dump(bases, newer_min, task_want, model_want, max_chars):
    for task, model, s in iter_samples(bases, newer_min):
        if task != task_want or model != model_want:
            continue
        print(f"### {task} · {model} · sample {s.id} · scored={bool(s.scores)}\n")
        for m in s.messages or []:
            tcs = getattr(m, "tool_calls", None) or []
            body = text_of(m).strip().replace("\n", "\n    ")
            if len(body) > max_chars:
                body = body[:max_chars] + " …[truncated]"
            print(f"[{m.role}] {body}")
            for tc in tcs:
                args = getattr(tc, "arguments", {})
                print(f"    ->tool_call {getattr(tc,'function','?')}({args})"[:max_chars])
            print()
        return
    print(f"(no sample found for {task_want} / {model_want})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", default=["data"])
    ap.add_argument("--newer-min", type=float, default=0)
    ap.add_argument("--dump", nargs=2, metavar=("EVAL", "MODEL"), default=None)
    ap.add_argument("--max-chars", type=int, default=600)
    args = ap.parse_args()
    if args.dump:
        dump(args.base, args.newer_min, args.dump[0], args.dump[1], args.max_chars)
    else:
        summary(args.base, args.newer_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
