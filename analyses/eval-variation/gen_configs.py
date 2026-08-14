"""Emit judge configs for the eval-variation conditions.

Per condition, one yaml for the SWC local fleet and one for the API cheap
fleet (pins copied from the cheap-7 config so results are comparable).
The pooled config draws each item's transcripts from whichever prompt
condition (baseline / words200 / words50-100) gives the best length match.

Usage: gen_configs.py [--max-word-diff 50] [--out analyses/eval-variation/configs]
"""

import argparse
import glob
from pathlib import Path

import yaml

from configs.experiment import SYSTEM_PROMPT

# Stronger first. ECI from configs/comparisons/model_set.json
# (qwen3-32b missing there: 130.0, the export_viewer fallback value).
SUBJECTS_BY_ECI = [
    ("deepseek-v4-flash-0731", 150.22),
    ("gemini-3.1-flash-lite", 144.17),
    ("glm-4.6", 140.86),
    ("ling-2.6-flash", 131.80),
    ("qwen3-vl-30b-a3b-instruct", 131.27),
    ("qwen3-32b", 130.00),
    ("qwen3-8b", 127.90),
]

# condition -> dir holding <model>/0/*.eval
CONDITIONS = {
    "baseline": "data/gpqa/63793eb6",
    "words200": "data/eval-variation/gpqa-words200/38c54e19",
    "words50-100": "data/eval-variation/gpqa-words50-100/ab96a27b",
}

# Tone standardizers: pooled with nothing, judged separately. API fleet is the
# three near-free judges only (qwen3-8b comes from SWC; deepseek/qwen3-32b/glm
# tone blocks parked pending budget).
TONE_CONDITIONS = {
    "confident": "data/eval-variation/gpqa-confident/9414956a",
    "matteroffact": "data/eval-variation/gpqa-matteroffact/87a285f7",
}
# Post-hoc rewrites of baseline transcripts (~200 words); dirs appear after
# rewrite.py + convert_rewrites.py have run.
REWRITE_CONDITIONS = {
    "rewrite-self": "data/eval-variation/gpqa-rewrite-self",
    "rewrite-standard": "data/eval-variation/gpqa-rewrite-standard",
}
TONE_API_JUDGES = [
    "openrouter/google/gemini-3.1-flash-lite",
    "openrouter/inclusionai/ling-2.6-flash",
    "openrouter/qwen/qwen3-vl-30b-a3b-instruct",
    # second wave, user-authorized 2026-08-11 (resume skips the three above)
    "openrouter/deepseek/deepseek-v4-flash-0731",
    "openrouter/qwen/qwen3-32b",
    "openrouter/z-ai/glm-4.6",
]

# SWC vLLM fleet; run.py passes the endpoint via --model-base-url and the
# sbatch exports LOCAL_API_KEY=none. Slug tail must equal the job's vllm
# --served-model-name (one judge per job) so log dirs keep the judge identity.
SWC_JUDGES = [
    "openai-api/local/Qwen3-0.6B",
    "openai-api/local/Llama-3.2-3B-Instruct",
    "openai-api/local/Qwen3-8B",
]

# API fleet: slugs + pins lifted verbatim from the cheap-7 judging config.
# Priority order - budget cuts drop from the tail.
CHEAP7_CONFIG = "configs/judge_cheap7_gpqa.yaml"
API_JUDGE_PRIORITY = [
    "openrouter/deepseek/deepseek-v4-flash-0731",
    "openrouter/google/gemini-3.1-flash-lite",
    "openrouter/z-ai/glm-4.6",
    "openrouter/inclusionai/ling-2.6-flash",
    "openrouter/qwen/qwen3-32b",
    "openrouter/qwen/qwen3-8b",
    "openrouter/qwen/qwen3-vl-30b-a3b-instruct",
]


def resolve_log(cond_dir: str, model: str) -> str:
    # gemini's stock-scored log underreads accuracy ($C$ answers); judging
    # needs the rescored copy so both-correct keeps the right items.
    sub = f"{model}-rescored" if model == "gemini-3.1-flash-lite" else model
    hits = sorted(glob.glob(f"{cond_dir}/{sub}/0/*.eval"))
    assert hits, f"no eval log for {model} under {cond_dir} (rescore missing?)"
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-word-diff", type=int, default=50,
                    help="judge an item only if the two transcripts are within this many words")
    ap.add_argument("--out", type=Path, default=Path("analyses/eval-variation/configs"))
    args = ap.parse_args()

    names = [m for m, _ in SUBJECTS_BY_ECI]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    pins = yaml.safe_load(Path(CHEAP7_CONFIG).read_text())["judge_model_args"]
    api_pins = {j: pins[j] for j in API_JUDGE_PRIORITY}
    transcripts = {c: {m: resolve_log(d, m) for m in names} for c, d in CONDITIONS.items()}

    args.out.mkdir(parents=True, exist_ok=True)

    def emit(path: Path, cfg: dict, n_judges: int) -> None:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=120))
        print(f"{path}: {len(cfg['comparisons'])} comparisons ({n_judges} judges x {len(pairs)} pairs)")

    common = {"system_prompt": SYSTEM_PROMPT, "max_word_diff": args.max_word_diff,
              "include_reasoning": False}
    for cond in ("baseline", "words200", "words50-100"):
        emit(args.out / f"judge_{cond}.yaml", {
            "log_root": f"logs/eval-variation/{cond}", **common,
            "transcripts": transcripts[cond],
            "comparisons": [[j, a, b] for j in SWC_JUDGES for a, b in pairs],
        }, len(SWC_JUDGES))

    # API fleet: baseline is NOT re-judged (cheap-7 verdicts already exist;
    # analysis subsets them to the length-matched items).
    for cond in ("words200", "words50-100"):
        emit(args.out / f"judge_api_{cond}.yaml", {
            "log_root": f"logs/eval-variation/{cond}", **common,
            "transcripts": transcripts[cond],
            "judge_model_args": api_pins,
            "comparisons": [[j, a, b] for j in API_JUDGE_PRIORITY for a, b in pairs],
        }, len(API_JUDGE_PRIORITY))

    emit(args.out / "judge_api_lenmatch-pooled.yaml", {
        "log_root": "logs/eval-variation/lenmatch-pooled", **common,
        "pooled": True,
        "transcripts_by_cond": transcripts,
        "judge_model_args": api_pins,
        "comparisons": [[j, a, b] for j in API_JUDGE_PRIORITY for a, b in pairs],
    }, len(API_JUDGE_PRIORITY))

    # Post-hoc rewrite conditions (baseline transcripts rewritten to ~200
    # words). Judged by all six API judges incl glm (user-authorized);
    # qwen3-8b comes from SWC as everywhere else.
    for cond, cond_dir in REWRITE_CONDITIONS.items():
        if not Path(cond_dir).exists():
            continue
        rw_transcripts = {m: resolve_log(cond_dir, m) for m in names}
        emit(args.out / f"judge_{cond}.yaml", {
            "log_root": f"logs/eval-variation/{cond}", **common,
            "transcripts": rw_transcripts,
            "comparisons": [[j, a, b] for j in SWC_JUDGES for a, b in pairs],
        }, len(SWC_JUDGES))
        rw_judges = [j for j in API_JUDGE_PRIORITY if "qwen3-8b" not in j]
        emit(args.out / f"judge_api_{cond}.yaml", {
            "log_root": f"logs/eval-variation/{cond}", **common,
            "transcripts": rw_transcripts,
            "judge_model_args": {j: pins[j] for j in rw_judges},
            "comparisons": [[j, a, b] for j in rw_judges for a, b in pairs],
        }, len(rw_judges))

    for cond, cond_dir in TONE_CONDITIONS.items():
        tone_transcripts = {m: resolve_log(cond_dir, m) for m in names}
        emit(args.out / f"judge_{cond}.yaml", {
            "log_root": f"logs/eval-variation/{cond}", **common,
            "transcripts": tone_transcripts,
            "comparisons": [[j, a, b] for j in SWC_JUDGES for a, b in pairs],
        }, len(SWC_JUDGES))
        emit(args.out / f"judge_api_{cond}.yaml", {
            "log_root": f"logs/eval-variation/{cond}", **common,
            "transcripts": tone_transcripts,
            "judge_model_args": {j: pins[j] for j in TONE_API_JUDGES},
            "comparisons": [[j, a, b] for j in TONE_API_JUDGES for a, b in pairs],
        }, len(TONE_API_JUDGES))


if __name__ == "__main__":
    main()
