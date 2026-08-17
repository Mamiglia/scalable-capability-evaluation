#!/usr/bin/env python3
"""Pull reference benchmark scores for the judged models from Artificial Analysis.

Data (c) Artificial Analysis, https://artificialanalysis.ai/ — see their methodology
at https://artificialanalysis.ai/methodology. Needs ARTIFICIAL_ANALYSIS_API_KEY.

AA lists reasoning and non-reasoning entries as separate models; MODELS below picks
the variant matching how we collected the transcripts (reasoning on where the model
config enables it). Coverage is not uniform: scicode/gpqa/hle cover every model,
livecodebench and the math index only about half. Missing = null, never 0.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

API = "https://artificialanalysis.ai/api/v2/data/llms/models"

# collected model id -> Artificial Analysis model name
MODELS = {
    "Qwen/Qwen3-32B": "Qwen3 32B (Reasoning)",
    "Qwen/Qwen3.5-4B": "Qwen3.5 4B (Reasoning)",
    "anthropic/claude-opus-4.7": "Claude Opus 4.7 (Adaptive Reasoning, Max Effort)",
    "anthropic/claude-sonnet-4.5": "Claude 4.5 Sonnet (Reasoning)",
    "deepseek-ai/DeepSeek-R1-0528": "DeepSeek R1 0528 (May '25)",
    "deepseek-ai/DeepSeek-V3.2": "DeepSeek V3.2 (Reasoning)",
    "google/gemma-3-12b-it": "Gemma 3 12B Instruct",
    "google/gemma-4-E2B-it": "Gemma 4 E2B (Reasoning)",
    "ibm-granite/granite-4.1-8b": "Granite 4.1 8B",
    "inclusionai/ling-2.6-flash": "Ling 2.6 Flash",
    "mistralai/mistral-medium-3": "Mistral Medium 3",
    "moonshotai/kimi-k2.6": "Kimi K2.6",
    "moonshotai/kimi-k3": "Kimi K3 (max)",
    "openai/gpt-4.1": "GPT-4.1",
    "openai/gpt-5": "GPT-5 (high)",
    "openai/o3": "o3",
    "qwen/qwen3.5-9b": "Qwen3.5 9B (Reasoning)",
    "z-ai/glm-4.6": "GLM-4.6 (Reasoning)",
    "zai-org/GLM-5.2": "GLM-5.2 (max)",
}

METRICS = ["artificial_analysis_intelligence_index", "artificial_analysis_coding_index",
           "artificial_analysis_math_index", "scicode", "livecodebench", "terminalbench_hard",
           "terminalbench_v2_1", "tau2", "tau_banking", "gpqa", "aime_25", "math_500",
           "mmlu_pro", "hle", "ifbench", "lcr"]
SHORT = {"artificial_analysis_intelligence_index": "intelligence_index",
         "artificial_analysis_coding_index": "coding_index",
         "artificial_analysis_math_index": "math_index"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("configs/comparisons/aa_scores.csv"))
    args = parser.parse_args()
    key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
    if not key:
        print("set ARTIFICIAL_ANALYSIS_API_KEY (source .env)", file=sys.stderr)
        return 1
    response = requests.get(API, headers={"x-api-key": key}, timeout=60)
    response.raise_for_status()
    catalogue = {entry["name"]: entry for entry in response.json()["data"]}
    missing = sorted(set(MODELS.values()) - set(catalogue))
    if missing:
        raise SystemExit(f"AA no longer lists: {missing} — update MODELS")

    rows = [{"model": model, "aa_model": name,
             **{SHORT.get(m, m): catalogue[name]["evaluations"].get(m) for m in METRICS}}
            for model, name in MODELS.items()]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        handle.write(f"# Artificial Analysis (https://artificialanalysis.ai/), fetched {date.today()}\n")
        frame.to_csv(handle, index=False)
    counts = frame.drop(columns=["model", "aa_model"]).notna().sum().sort_values(ascending=False)
    print(f"wrote {args.out} ({len(frame)} models)")
    print("\n".join(f"  {metric:22s} {n}/{len(frame)} models" for metric, n in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
