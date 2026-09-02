"""Word-count stats and sample transcripts for a condition's log dir.

Usage: word_stats.py <hash_dir> [--dump N out.md]
"""

import sys
from pathlib import Path
from statistics import median, quantiles

from inspect_ai.log import read_eval_log

BASELINE_MEDIANS = {
    "deepseek-v4-flash-0731": 67,
    "glm-4.6": 111,
    "gemini-3.1-flash-lite": 277,
    "qwen3-32b": 418,
    "ling-2.6-flash": 440,
    "qwen3-8b": 305,
    "qwen3-vl-30b-a3b-instruct": 258,
}


def main() -> None:
    base = Path(sys.argv[1])
    dump_n, dump_path = (int(sys.argv[3]), sys.argv[4]) if "--dump" in sys.argv else (0, None)
    dump = []
    print(f"{'model':28} {'n':>4} {'p25':>5} {'med':>5} {'p75':>5}   baseline_med")
    for d in sorted(base.iterdir()):
        evs = list(d.rglob("*.eval"))
        if not evs:
            continue
        try:
            log = read_eval_log(str(evs[0]))
        except Exception as e:
            print(f"{d.name:28} unreadable ({type(e).__name__})")
            continue
        samples = [s for s in (log.samples or []) if s.output and s.output.completion]
        wcs = [len(s.output.completion.split()) for s in samples]
        if len(wcs) < 4:
            print(f"{d.name:28} {len(wcs):4d} samples so far")
            continue
        q = quantiles(wcs, n=4)
        print(
            f"{d.name:28} {len(wcs):4d} {q[0]:5.0f} {median(wcs):5.0f} {q[2]:5.0f}"
            f"   {BASELINE_MEDIANS.get(d.name, '?')}"
        )
        for s in samples[:dump_n]:
            wc = len(s.output.completion.split())
            sc = s.scores["choice"].value if s.scores else "?"
            dump.append(f"\n## {d.name} — {wc} words, score {sc}\n\n{s.output.completion}\n")
    if dump_path:
        Path(dump_path).write_text("# words-200 sample transcripts (target 200)\n" + "\n".join(dump))
        print(f"\ndumped to {dump_path}")


if __name__ == "__main__":
    main()
