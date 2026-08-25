"""Name SAE features and check the names' fidelity, following WIMHF (Movva et al. 2025).

Per feature: n_candidates descriptions, each written by the interpreter model from
top-activating transcripts (95th-100th percentile of positive activations) shown
against zero-activation transcripts; then, on scoring_examples random transcripts
where the feature is nonzero, the annotator model says whether the description is
present, and fidelity is the Pearson correlation between that answer and the signed
activation. The best candidate per feature is kept when p <= p_value_threshold
(0.05 / width, Bonferroni) and prevalence >= min_prevalence.
Paid calls via OpenRouter, pinned to the OpenAI endpoints: run with --dry-run first to see the volume.
"""
import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr

INTERPRET_PROMPT = """# Task Overview

You are a machine learning researcher who has trained a neural network on a dataset of answer transcripts written by AI models: each transcript is a model's answer to a question, or its tool-using trace for a task. All transcripts in the dataset are correct answers. You are trying to understand what text feature causes a specific neuron in the network to fire.

You are given transcripts where the neuron FIRES and transcripts where it does NOT fire.

Your goal is to identify the single feature that is present in the firing transcripts and absent from the non-firing ones, and to describe it as a concise natural-language concept.

# Instructions

First, consider what the firing transcripts have in common. Then check that the non-firing transcripts lack it.

## Example Features

- "works the problem with numbered steps"
- "does not state the final answer until the last line"
- "runs many shell commands before editing any file"

## Additional Rules

- The feature should be objective, focusing on concrete attributes of the text (structure, style, content, tone, length, formatting); "is helpful" or "is correct" are not concrete enough.
- Err on the side of being concise.
- The feature must not refer to "firing" or "non-firing" transcripts and must not use comparative words ("more", "longer"); it should be an attribute that can be present in an individual transcript.

# Examples

FIRING TRANSCRIPTS:
----------------
{firing}
----------------

NON-FIRING TRANSCRIPTS:
----------------
{non_firing}
----------------

# Output

Do not output anything besides the feature. Please suggest exactly one description, starting with "-" and surrounded by quotes. Your response is:
- \""""

ANNOTATE_PROMPT = """Given the TRANSCRIPT (an AI model's answer to a question, or its tool-using trace), decide whether the CONCEPT is clearly present in it.

- Only output "Yes" if the CONCEPT obviously and clearly applies to the transcript; otherwise output "No".
- Do not judge whether the answer is good; only whether the CONCEPT is present.

Output one of these exact strings: "Yes" or "No". Do not output anything else.

TRANSCRIPT:
{text}

CONCEPT: "{concept}"
Output:"""


def truncate_words(text: str, max_words: int) -> str:
    """Keep the head and tail of a long transcript, max_words words in total."""
    words = text.split()
    if len(words) <= max_words:
        return text
    half = max_words // 2
    return " ".join(words[:half]) + "\n[...middle of transcript truncated...]\n" + " ".join(words[-half:])


def interpretation_examples(activations: np.ndarray, rows: pd.DataFrame, *, n_firing: int,
                            n_zero: int, top_percentile: float, seed: int) -> dict[str, np.ndarray]:
    """Random n_firing rows from the top percentile of positive activations, plus n_zero zero rows."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(activations > 0)
    if not len(positive):
        return {"firing": np.array([], dtype=int), "zero": np.array([], dtype=int)}
    cutoff = np.percentile(activations[positive], top_percentile)
    top = positive[activations[positive] >= cutoff]
    zero = np.flatnonzero(activations == 0)
    return {"firing": rng.choice(top, size=min(n_firing, len(top)), replace=False),
            "zero": rng.choice(zero, size=min(n_zero, len(zero)), replace=False)}


def scoring_examples(activations: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Random rows with nonzero activation (signed), as in the paper's fidelity check."""
    nonzero = np.flatnonzero(activations != 0)
    return np.random.default_rng(seed).choice(nonzero, size=min(n, len(nonzero)), replace=False)


def parse_description(reply: str) -> str:
    reply = reply.strip()
    for prefix in ('- "', '-"', '"', "- "):
        if reply.startswith(prefix):
            reply = reply[len(prefix):]
    return reply.strip().strip('"').strip()


def parse_yes_no(reply: str) -> float | None:
    word = reply.strip().strip('"').strip(".").lower()
    return {"yes": 1.0, "no": 0.0}.get(word)


def fidelity(annotations: np.ndarray, activations: np.ndarray) -> tuple[float, float]:
    """Pearson correlation between yes/no annotation and signed activation, with its p-value."""
    valid = ~np.isnan(annotations)
    if valid.sum() < 3 or len(np.unique(annotations[valid])) < 2:
        return 0.0, 1.0
    result = pearsonr(activations[valid], annotations[valid])
    return float(result[0]), float(result[1])


class CachedAsker:
    """OpenRouter chat completions pinned to one endpoint, with a jsonl cache keyed by model + prompt.

    Every cached record keeps the provider and model that actually served the call.
    """

    def __init__(self, cache_path: Path, endpoints: dict[str, str], dry_run: bool = False):
        self.cache_path, self.endpoints, self.dry_run = cache_path, endpoints, dry_run
        self.lock = threading.Lock()
        self.cache = {}
        self.chars = 0
        self.calls = 0
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                record = json.loads(line)
                self.cache[record["key"]] = record["reply"]
        self.client = None

    def __call__(self, model: str, prompt: str, max_tokens: int) -> str:
        key = hashlib.sha256(f"{model}\0{prompt}".encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            self.chars += len(prompt)
            self.calls += 1
        if self.dry_run:
            return ""
        if self.client is None:
            import os
            import openai
            self.client = openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                                        api_key=os.environ["OPENROUTER_API_KEY"], max_retries=5)
        response = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            extra_body={"reasoning": {"effort": "low"},
                        "provider": {"only": [self.endpoints[model]], "allow_fallbacks": False,
                                     "require_parameters": True, "data_collection": "deny"}})
        reply = response.choices[0].message.content or ""
        served = {"provider": getattr(response, "provider", None) or (response.model_extra or {}).get("provider"),
                  "served_model": response.model, "id": response.id,
                  "usage": response.usage.model_dump() if response.usage else None}
        with self.lock:
            self.cache[key] = reply
            with self.cache_path.open("a") as handle:
                handle.write(json.dumps({"key": key, "model": model, "reply": reply} | served) + "\n")
        return reply


def describe_features(activations: np.ndarray, labels: list[str], rows: pd.DataFrame,
                      groups: dict[str, list[str]], settings: dict, ask, width: int,
                      *, workers: int = 8) -> pd.DataFrame:
    threshold = settings["p_value_threshold"] if settings.get("p_value_threshold") else 0.05 / width
    texts = [truncate_words(text, settings["max_words"]) for text in rows["transcript"]]
    candidate_tasks, scoring = [], {}
    for column, label in enumerate(labels):
        group = label.split(":")[0]
        mask = rows["benchmark"].isin(groups[group]).to_numpy()
        group_index = np.flatnonzero(mask)
        values = activations[mask, column]
        scoring[label] = {"group": group, "prevalence": float((values != 0).mean()),
                          "score_rows": group_index[scoring_examples(values, settings["scoring_examples"],
                                                                     settings["seed"] + column)]}
        for candidate in range(settings["n_candidates"]):
            chosen = interpretation_examples(
                values, rows.iloc[group_index], n_firing=settings["interpret_examples"],
                n_zero=settings["contrast_examples"], top_percentile=settings["top_percentile"],
                seed=settings["seed"] + candidate)
            if len(chosen["firing"]) < settings["interpret_examples"]:
                continue
            prompt = INTERPRET_PROMPT.format(
                firing="\n----------------\n".join(texts[i] for i in group_index[chosen["firing"]]),
                non_firing="\n----------------\n".join(texts[i] for i in group_index[chosen["zero"]]))
            candidate_tasks.append((label, candidate, prompt))

    def interpret(task):
        label, candidate, prompt = task
        return label, candidate, parse_description(ask(settings["interpreter_model"], prompt,
                                                       settings["interpreter_max_tokens"]))

    with ThreadPoolExecutor(workers) as pool:
        descriptions = list(pool.map(interpret, candidate_tasks))
    annotate_tasks = [(label, candidate, description, int(row))
                      for label, candidate, description in descriptions
                      for row in scoring[label]["score_rows"]]

    def annotate(task):
        label, candidate, description, row = task
        prompt = ANNOTATE_PROMPT.format(text=texts[row], concept=description)
        return label, candidate, row, parse_yes_no(ask(settings["annotator_model"], prompt,
                                                       settings["annotator_max_tokens"]))

    with ThreadPoolExecutor(workers) as pool:
        annotations = list(pool.map(annotate, annotate_tasks))
    answers = {}
    for label, candidate, row, answer in annotations:
        answers.setdefault((label, candidate), {})[row] = np.nan if answer is None else answer

    records = []
    for column, label in enumerate(labels):
        info = scoring[label]
        candidates = []
        for other_label, candidate, description in descriptions:
            if other_label != label:
                continue
            rows_scored = info["score_rows"]
            annotation = np.array([answers[(label, candidate)].get(int(row), np.nan) for row in rows_scored])
            correlation, p_value = fidelity(annotation, activations[rows_scored, column])
            candidates.append({"candidate": candidate, "description": description,
                               "correlation": correlation, "p_value": p_value,
                               "n_scored": int((~np.isnan(annotation)).sum())})
        record = {"feature": label, "group": info["group"], "prevalence": info["prevalence"],
                  "n_candidates": len(candidates), "candidates_json": json.dumps(candidates)}
        if not candidates:
            records.append(record | {"description": None, "correlation": None, "p_value": None,
                                     "kept": False, "reason": "too_few_firing_transcripts"})
            continue
        best = max(candidates, key=lambda item: item["correlation"])
        kept = (best["p_value"] <= threshold and best["correlation"] >= settings["min_correlation"]
                and info["prevalence"] >= settings["min_prevalence"])
        reason = None if kept else ("low_prevalence" if info["prevalence"] < settings["min_prevalence"]
                                    else "fidelity_not_significant")
        records.append(record | {"description": best["description"], "correlation": best["correlation"],
                                 "p_value": best["p_value"], "kept": bool(kept), "reason": reason})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True, help="for report_rows.parquet (transcript text)")
    parser.add_argument("--config", type=Path, default=Path("configs/workstream_b.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="count prompt characters and calls, call nothing")
    parser.add_argument("--max-features", type=int, help="describe only the first N features (smoke)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings, groups, width = config["descriptions"], config["sae"]["fit_groups"], config["sae"]["width"]
    rows = pd.read_parquet(args.dense_dir / "report_rows.parquet")
    if not pd.read_parquet(args.sae_dir / "rows.parquet")["transcript_key"].equals(rows["transcript_key"]):
        raise ValueError("report rows are not in SAE row order")
    activations = np.load(args.sae_dir / "activations.npy", allow_pickle=False)
    labels = [str(label) for label in np.load(args.sae_dir / "feature_labels.npy", allow_pickle=False)]
    if args.max_features:
        labels, activations = labels[:args.max_features], activations[:, :args.max_features]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ask = CachedAsker(args.output_dir / "llm_cache.jsonl", settings["openrouter_endpoints"], dry_run=args.dry_run)
    table = describe_features(activations, labels, rows, groups, settings, ask, width, workers=args.workers)
    if args.dry_run:
        print(f"features={len(labels)} calls={ask.calls} prompt_chars={ask.chars} (~{ask.chars // 4} tokens)")
        return
    table.to_csv(args.output_dir / "feature_descriptions.csv", index=False)
    print(f"features={len(table)} kept={int(table.kept.sum())} "
          f"median_correlation={table.correlation.median():.2f} calls={ask.calls}")


if __name__ == "__main__":
    main()
