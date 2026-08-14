#!/usr/bin/env python3
"""Per-benchmark scatter: model release_date (x) vs eval accuracy (y).

One dot per model (mean accuracy across its successful eval logs), coloured by
the vendor inferred for that model. Dot size is intentionally kept fixed (the
param-size idea was dropped).

Metadata resolution order for each model dir shortname:
  1. configs/models/<org>/<name>.yml  -> vendor (org dir) + release_date
  2. name-parse of an explicit date in the dir name
  3. curated KNOWN_DATES map (real models + sequenced guesses for the mock ones)
  4. otherwise: vendor still shown (from name prefix) but point dropped from the
     date axis.

Run:  uv run python scripts/plot_benchmarks.py
"""
from __future__ import annotations
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from inspect_ai.log import read_eval_log

DATA = Path("data")
OUT = Path("analysis/benchmark_plots")

VARIANT_SUFFIX = re.compile(
    r"_(fp8|bf16|none|high|low|medium|minimal|max|_\d+k|xhigh|32k|16k|27k|59k|64k)(?=_|$)"
)
KNOWN_DATES: dict[str, str] = {
    # ---- Anthropic (real, dated by name for most; non-dated here) ----
    "claude-2.0": "2023-07-11",
    "claude-2.1": "2023-11-21",
    "claude-opus-4-6": "2026-02-01",
    "claude-opus-4-7": "2026-04-16",
    "claude-opus-4-8": "2026-06-01",
    "claude-opus-5": "2026-08-01",
    "claude-sonnet-4-6": "2026-03-01",
    "claude-sonnet-4-5": "2025-09-29",
    "claude-sonnet-5": "2026-09-01",
    "claude-fable-5": "2026-09-01",
    # ---- DeepSeek ----
    "deepseek-chat": "2024-05-01",
    "deepseek-llm-67b-chat": "2023-11-01",
    "deepseek-r1": "2025-01-20",
    "deepseek-r1-distill-llama-70b": "2025-01-20",
    "deepseek-r1-distill-qwen-14b": "2025-01-20",
    "deepseek-reasoner": "2025-09-01",
    "deepseek-v3": "2024-12-26",
    "deepseek-v4-pro": "2026-03-01",
    "deepseek-v4-flash-0731": "2026-04-24",
    # ---- Google / Gemini / Gemma ----
    "gemini-1.0-pro-001": "2023-12-13",
    "gemini-1.5-flash-001": "2024-05-01",
    "gemini-1.5-flash-002": "2024-09-24",
    "gemini-1.5-flash-8b-001": "2024-10-03",
    "gemini-1.5-pro-001": "2024-02-15",
    "gemini-1.5-pro-002": "2024-09-24",
    "gemini-2.0-flash-001": "2025-02-05",
    "gemini-2.0-flash-thinking-exp-01-21": "2024-12-19",
    "gemini-2.0-pro-exp-02-05": "2025-02-05",
    "gemini-2.5-flash": "2025-06-17",
    "gemini-2.5-flash-preview-04-17": "2025-04-17",
    "gemini-2.5-flash-preview-05-20": "2025-05-20",
    "gemini-2.5-pro": "2025-08-19",
    "gemini-2.5-pro-exp-03-25": "2025-03-25",
    "gemini-2.5-pro-preview-05-06": "2025-05-06",
    "gemini-2.5-pro-preview-06-05": "2025-06-05",
    "gemini-3-flash-preview": "2025-11-01",
    "gemini-3-pro-preview": "2025-11-01",
    "gemini-3.1-flash-lite": "2026-03-03",
    "gemini-3.1-pro-preview": "2026-04-01",
    "gemini-3.5-flash": "2026-07-01",
    "gemini-3.5-flash-lite": "2026-06-01",
    "gemini-3.6-flash": "2026-09-01",
    "gemma-2-9b-it": "2024-06-27",
    "gemma-2-27b-it": "2024-06-27",
    "gemma-3-12b-it": "2025-03-12",
    "gemma-3-27b-it": "2025-03-12",
    "gemma-7b": "2024-02-21",
    # ---- Meta / Llama ----
    "llama-7b": "2023-02-24",
    "llama-2-13b-chat-hf": "2023-07-18",
    "llama-2-70b-chat-hf": "2023-07-18",
    "llama-3.1-8b-instruct": "2024-07-23",
    "llama-3.1-70b-instruct": "2024-07-23",
    "llama-3.1-405b-instruct": "2024-07-23",
    "meta-llama-3-8b-instruct": "2024-04-18",
    "meta-llama-3-70b-instruct": "2024-04-18",
    "llama-3.2-90b-vision-instruct": "2024-09-25",
    "llama-3.3-70b-instruct": "2024-12-06",
    "llama-4-scout-17b-16e-instruct": "2025-04-05",
    "llama-4-maverick-17b-128e-instruct-fp8": "2025-04-05",
    "llama-3.1-tulu-3-70b-dpo": "2024-11-01",
    # ---- Mistral family ----
    "mistral-7b-instruct-v0.3": "2024-05-01",
    "mixtral-8x7b-instruct-v0.1": "2023-12-11",
    "open-mistral-7b": "2023-09-27",
    "open-mixtral-8x7b": "2023-12-11",
    "open-mixtral-8x22b": "2024-04-17",
    "mistral-large-2402": "2024-02-26",
    "mistral-large-2407": "2024-07-24",
    "mistral-large-2411": "2024-11-18",
    "mistral-medium-2505": "2025-05-01",
    "mistral-small-2501": "2025-01-01",
    "mistral-small-2503": "2025-03-01",
    "ministral-3b-2410": "2024-10-01",
    "ministral-8b-2410": "2024-10-01",
    "open-mistral-nemo-2407": "2024-07-18",
    "magistral-small-2506": "2025-06-01",
    "mistral-medium-3": "2026-03-01",
    # ---- Qwen ----
    "qwen-14b": "2023-10-30",
    "qwen1.5-32b-chat": "2024-02-01",
    "qwen1.5-72b-chat": "2024-02-01",
    "qwen2-72b-instruct": "2024-06-07",
    "qwen2.5-32b-instruct": "2024-09-01",
    "qwen2.5-72b-instruct": "2024-09-01",
    "qwen3-14b": "2025-04-29",
    "qwen3-32b": "2025-04-29",
    "qwen3-235b-a22b": "2025-07-01",
    "qwen3-235b-a22b-thinking-2507": "2025-07-01",
    "qwen3-vl-30b-a3b-instruct": "2025-10-04",
    "qwq-plus": "2025-04-01",
    "qwen3-max-2025-09-23": "2025-09-23",
    "qwen-max-2025-01-25": "2025-01-25",
    "qwen-plus-2025-01-25": "2025-01-25",
    "qwen-turbo-2024-11-01": "2024-11-01",
    "qwen3.5-4b": "2026-02-10",
    "qwen3.5-9b": "2026-02-10",
    "qwen3.5-flash": "2026-03-01",
    "qwen3.5-plus": "2026-02-01",
    "qwen3.5-122b-a10b": "2026-02-24",
    "qwen3.5-397b-a17b": "2026-02-01",
    "qwen3.6-27b": "2026-04-01",
    "qwen3.6-35b-a3b": "2026-05-01",
    "qwen3.6-flash": "2026-06-01",
    "qwen3.6-plus": "2026-06-01",
    "qwen3.6-max-preview": "2026-06-01",
    "qwen3.7-flash": "2026-08-01",
    "qwen3.7-plus": "2026-08-01",
    "qwen3.7-max": "2026-08-01",
    # ---- OpenAI / o-series / gpt-oss ----
    "gpt-3.5-turbo-0125": "2024-01-25",
    "gpt-3.5-turbo-1106": "2023-11-06",
    "gpt-4-0314": "2023-03-14",
    "gpt-4-0613": "2023-06-13",
    "gpt-4-1106-preview": "2023-11-06",
    "gpt-4-0125-preview": "2024-01-25",
    "gpt-4-turbo-2024-04-09": "2024-04-09",
    "gpt-4.1-2025-04-14": "2025-04-14",
    "gpt-4.1-mini-2025-04-14": "2025-04-14",
    "gpt-4.1-nano-2025-04-14": "2025-04-14",
    "gpt-4o-2024-05-13": "2024-05-13",
    "gpt-4o-2024-08-06": "2024-08-06",
    "gpt-4o-2024-11-20": "2024-11-20",
    "gpt-4o-mini-2024-07-18": "2024-07-18",
    "gpt-4.5-preview-2025-02-27": "2025-02-27",
    "gpt-5-2025-08-07": "2025-08-07",
    "gpt-5-mini-2025-08-07": "2025-08-07",
    "gpt-5-nano-2025-08-07": "2025-08-07",
    "o1-preview-2024-09-12": "2024-09-12",
    "o1-mini-2024-09-12": "2024-09-12",
    "o1-2024-12-17": "2024-12-17",
    "o3-mini-2025-01-31": "2025-01-31",
    "o3-2025-04-16": "2025-04-16",
    "o4-mini-2025-04-16": "2025-04-16",
    "o4-mini": "2025-04-16",
    "o3": "2025-04-16",
    "gpt-5.1-2025-11-13": "2025-11-13",
    "gpt-5.2-2025-12-11": "2025-12-11",
    "gpt-5.4-2026-03-05": "2026-03-05",
    "gpt-5.4-mini-2026-03-17": "2026-03-17",
    "gpt-5.4-nano-2026-03-17": "2026-03-17",
    "gpt-5.5": "2026-05-01",
    "gpt-5.5-pre-release": "2026-05-01",
    "gpt-5.5-pro-pre-release": "2026-05-01",
    "gpt-5.6-luna": "2026-08-01",
    "gpt-5.6-sol": "2026-08-01",
    "gpt-5.6-terra": "2026-08-01",
    "gpt-oss-20b": "2025-08-05",
    "gpt-oss-120b": "2025-08-05",
    "openai-gpt-oss-120b": "2025-08-05",
    # ---- xAI ----
    "grok-2-1212": "2024-08-13",
    "grok-3-beta": "2025-02-18",
    "grok-3-mini-beta": "2025-02-18",
    # ---- Moonshot / Kimi ----
    "kimi-k2-instruct": "2025-07-01",
    "kimi-k2-thinking": "2025-07-01",
    "kimi-k2-thinking-turbo": "2025-06-01",
    "kimi-k2.6": "2026-01-20",
    "kimi-k2.7-code": "2026-03-01",
    "kimi-k3": "2026-05-01",
    "fireworks-kimi-k2p5": "2024-12-01",
    # ---- Z.ai / GLM ----
    "glm-4.5": "2025-07-01",
    "glm-4.6": "2025-09-30",
    "glm-4.7": "2025-12-01",
    "glm-5": "2026-03-01",
    "glm-5.1": "2026-05-01",
    "glm-5.2": "2026-07-01",
    # ---- Others ----
    "dbrx-instruct": "2024-03-28",
    "eurus-2-7b-prime": "2024-10-01",
    "granite-4.0-h-micro": "2025-07-01",
    "granite-4.1-8b": "2026-02-01",
    "hermes-2-theta-llama-3-70b": "2024-08-01",
    "minimax-m3": "2025-05-01",
    "mpt-30b": "2023-05-01",
    "phi-3-medium-128k-instruct": "2024-05-01",
    "phi-4": "2024-12-12",
    "wizardlm-2-8x22b": "2024-04-01",
    "yi-34b-chat": "2023-11-01",
    "yi-1.5-34b-chat": "2024-07-01",
    "nvidia-nemotron-3-super-120b-a12b": "2025-02-01",
}

VENDOR_PREFIX: list[tuple[tuple[str, ...], str]] = [
    (("claude",), "Anthropic"),
    (("gpt-", "o1", "o3", "o4", "gpt-oss", "openai-"), "OpenAI"),
    (("gemini", "gemma"), "Google"),
    (("llama", "meta-"), "Meta"),
    (("qwen", "qwq"), "Qwen"),
    (("deepseek",), "DeepSeek"),
    (("glm",), "Z.ai"),
    (("kimi",), "Moonshot"),
    (("mistral", "mixtral", "ministral", "open-mistral", "magistral", "open-mixtral"), "Mistral"),
    (("grok",), "xAI"),
    (("granite",), "IBM"),
    (("phi",), "Microsoft"),
    (("mpt",), "MosaicML"),
    (("dbrx",), "Databricks"),
    (("eurus",), "Abacus"),
    (("nemotron",), "NVIDIA"),
    (("minimax",), "MiniMax"),
]


def load_config_map() -> dict[str, dict]:
    """shortname -> {'vendor': org, 'release': str|None} from configs/models."""
    out: dict[str, dict] = {}
    for f in glob.glob("configs/models/*/*.yml"):
        with open(f) as fh:
            d = yaml.safe_load(fh)
        if not d or "name" not in d:
            continue
        org = Path(f).parent.name
        sn = d["name"].split("/")[-1].lower()
        out[sn] = {"vendor": org, "release": d.get("release_date")}
        out[d["name"].lower()] = {"vendor": org, "release": d.get("release_date")}
    return out


def parse_name_date(name: str) -> str | None:
    """Extract an unambiguous date from a dir name."""
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", name)
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}"
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if m and int(m[2]) <= 12 and int(m[3]) <= 31:
        return f"{m[1]}-{m[2]}-{m[3]}"
    m = re.search(r"(20\d{2})-(\d{2})$", name)
    if m:
        return f"{m[1]}-{m[2]}-01"
    return None


def vendor_from_name(name: str) -> str:
    for prefix, v in VENDOR_PREFIX:
        if name.startswith(prefix):
            return v
    return "Unknown"


CANONICAL_VENDOR = {
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "ibm": "IBM",
    "inclusionai": "InclusionAI",
    "meta-llama": "Meta",
    "microsoft": "Microsoft",
    "mistralai": "Mistral",
    "moonshotai": "Moonshot",
    "mosaicml": "MosaicML",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "qwen": "Qwen",
    "zai": "Z.ai",
}


CONFIG_MAP = load_config_map()


def normalize_date(date: str | None) -> str | None:
    """Collapse a range like '2025-04-28/2025-04-29' to its start date."""
    if date:
        return date.split("/")[0].strip()
    return date


def resolve(model_dir: str):
    """Return (vendor, release_date_str) for a data model dir name."""
    base = model_dir
    while True:
        nxt = VARIANT_SUFFIX.sub("", base)
        if nxt == base:
            break
        base = nxt
    # config hit (try exact dir, then suffix-stripped base)
    for key in (model_dir, base, base.replace("-", "/")):
        info = CONFIG_MAP.get(key)
        if info:
            vendor = CANONICAL_VENDOR.get(info["vendor"], info["vendor"])
            return vendor, normalize_date(info.get("release"))
    vendor = vendor_from_name(base)
    date = parse_name_date(base)
    if not date:
        date = KNOWN_DATES.get(model_dir) or KNOWN_DATES.get(base)
    return vendor, date


def eval_accuracy(path: str) -> float | None:
    try:
        log = read_eval_log(path, header_only=True)
    except Exception:
        return None
    if log.status != "success" or not log.results or not log.results.scores:
        return None
    for score in log.results.scores:
        metrics = getattr(score, "metrics", None)
        if metrics and "accuracy" in metrics:
            val = metrics["accuracy"].value
            if isinstance(val, (int, float)):
                return float(val)
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # benchmark dirs at top level of data/
    benchmarks = sorted(
        p for p in DATA.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in {"legacy"}
    )
    for bench in benchmarks:
        evals = glob.glob(str(bench / "*" / "*" / "*" / "*.eval"))
        if not evals:
            continue
        # group accuracy by model dir
        per_model: dict[str, list[float]] = {}
        for path in evals:
            model_dir = Path(path).parts[-3]
            acc = eval_accuracy(path)
            if acc is not None:
                per_model.setdefault(model_dir, []).append(acc)
        if not per_model:
            print(f"[skip] {bench.name}: no successful evals")
            continue

        rows = []
        for model_dir, accs in per_model.items():
            vendor, date = resolve(model_dir)
            if not date:
                continue
            rows.append((model_dir, vendor, date, sum(accs) / len(accs)))
        if not rows:
            print(f"[skip] {bench.name}: no resolvable release dates ({len(per_model)} models)")
            continue

        rows.sort(key=lambda r: r[2])
        xs = mdates.datestr2num([r[2] for r in rows])
        ys = [r[3] for r in rows]
        vendors = sorted({r[1] for r in rows})
        palette = plt.cm.tab20([i / max(1, len(vendors) - 1) for i in range(len(vendors))])
        color_of = {v: palette[i] for i, v in enumerate(vendors)}

        fig, ax = plt.subplots(figsize=(11, 6))
        done: set[str] = set()
        for (model_dir, vendor, date, acc), x, y in zip(rows, xs, ys):
            if vendor not in done:
                lbl = vendor
                done.add(vendor)
            else:
                lbl = "_nolegend_"
            ax.scatter(x, y, s=70, color=color_of[vendor], edgecolor="k", linewidths=0.4, label=lbl)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Release date")
        ax.set_title(f"{bench.name}  (release_date vs accuracy, {len(rows)} models)")
        ax.legend(title="Vendor", ncol=2, fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = OUT / f"{bench.name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        nno = len(per_model) - len(rows)
        print(f"[ok] {bench.name}: {len(rows)} models plotted | {nno} skipped (no date) -> {out.name}")


if __name__ == "__main__":
    main()
