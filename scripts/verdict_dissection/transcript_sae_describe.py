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
import re
import subprocess
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

ANNOTATE_PROMPT = """Below are numbered CONCEPTS and numbered TRANSCRIPTS (each an AI model's answer to a question, or its tool-using trace). For every concept, decide in which transcripts it is clearly present.

- Count a concept as present in a transcript only if it obviously and clearly applies to that transcript.
- Do not judge whether an answer is good; only whether each concept is present.

Output a JSON object mapping each concept number to the list of transcript numbers where that concept is present, for example {{"1": [2, 5], "2": []}}. Include every concept number, even with an empty list. Do not output anything else.

CONCEPTS:
{concepts}

{transcripts}

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
    """Random n_firing rows from the top percentile of positive activations (or simply the
    top n_firing rows when that bin is smaller), plus n_zero zero rows."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(activations > 0)
    if not len(positive):
        return {"firing": np.array([], dtype=int), "zero": np.array([], dtype=int)}
    cutoff = np.percentile(activations[positive], top_percentile)
    top = positive[activations[positive] >= cutoff]
    if len(top) < n_firing:
        top = positive[np.argsort(-activations[positive], kind="stable")[:n_firing]]
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


def parse_presence(reply: str, n_concepts: int, n_transcripts: int) -> np.ndarray | None:
    """concept x transcript 0/1 matrix from the batched annotation reply, None if unparseable."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    matrix = np.zeros((n_concepts, n_transcripts))
    for concept, numbers in parsed.items():
        if not str(concept).strip().isdigit() or not isinstance(numbers, list):
            return None
        for number in numbers:
            if isinstance(number, int) and 1 <= int(concept) <= n_concepts and 1 <= number <= n_transcripts:
                matrix[int(concept) - 1, number - 1] = 1.0
    return matrix


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

    def __init__(self, cache_path: Path, endpoints: dict[str, str], dry_run: bool = False,
                 offline: bool = False):
        self.cache_path, self.endpoints, self.dry_run, self.offline = cache_path, endpoints, dry_run, offline
        self.lock = threading.Lock()
        self.cache = {}
        self.chars = 0
        self.calls = 0
        self.pending = []
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                record = json.loads(line)
                self.cache[record["key"]] = record["reply"]
        self.client = None

    def __call__(self, model: str, prompt: str, max_tokens: int) -> str | None:
        key = hashlib.sha256(f"{model}\0{prompt}".encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            self.chars += len(prompt)
            self.calls += 1
            if self.offline:
                self.pending.append({"key": key, "model": model, "prompt": prompt})
                return None
        if self.dry_run:
            return ""
        if self.client is None:
            import os
            import openai
            self.client = openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                                        api_key=os.environ["OPENROUTER_API_KEY"], max_retries=5)
        response = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
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
                      *, workers: int = 8) -> pd.DataFrame | None:
    """Returns the feature table, or None while replies are still pending (offline mode)."""
    threshold = settings["p_value_threshold"] if settings.get("p_value_threshold") else 0.05 / width
    texts = [truncate_words(text, settings["max_words"]) for text in rows["transcript"]]
    batch = settings["annotation_batch_size"]
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
        reply = ask(settings["interpreter_model"], prompt, settings["interpreter_max_tokens"])
        return label, candidate, None if reply is None else parse_description(reply)

    with ThreadPoolExecutor(workers) as pool:
        descriptions = list(pool.map(interpret, candidate_tasks))
    if any(description is None for _, _, description in descriptions):
        return None
    candidates = {}
    for label, candidate, description in descriptions:
        candidates.setdefault(label, []).append((candidate, description))

    annotate_tasks = []
    for label, items in candidates.items():
        score_rows = scoring[label]["score_rows"]
        for start in range(0, len(score_rows), batch):
            annotate_tasks.append((label, score_rows[start:start + batch]))

    def annotate(task):
        label, batch_rows = task
        items = candidates[label]
        prompt = ANNOTATE_PROMPT.format(
            concepts="\n".join(f'{number}. "{description}"' for number, (_, description) in enumerate(items, 1)),
            transcripts="\n\n".join(f"TRANSCRIPT {number}:\n{texts[row]}" for number, row in enumerate(batch_rows, 1)))
        reply = ask(settings["annotator_model"], prompt, settings["annotator_max_tokens"])
        return label, batch_rows, None if reply is None else parse_presence(reply, len(items), len(batch_rows))

    with ThreadPoolExecutor(workers) as pool:
        annotations = list(pool.map(annotate, annotate_tasks))
    if getattr(ask, "pending", None):
        return None
    answers = {}      # (label, candidate) -> {row: 0/1 or nan}
    for label, batch_rows, matrix in annotations:
        for position, (candidate, _) in enumerate(candidates[label]):
            store = answers.setdefault((label, candidate), {})
            for number, row in enumerate(batch_rows):
                store[int(row)] = np.nan if matrix is None else matrix[position, number]

    records = []
    for column, label in enumerate(labels):
        info = scoring[label]
        scored = []
        for candidate, description in candidates.get(label, []):
            rows_scored = info["score_rows"]
            annotation = np.array([answers[(label, candidate)].get(int(row), np.nan) for row in rows_scored])
            correlation, p_value = fidelity(annotation, activations[rows_scored, column])
            scored.append({"candidate": candidate, "description": description,
                           "correlation": correlation, "p_value": p_value,
                           "n_scored": int((~np.isnan(annotation)).sum())})
        record = {"feature": label, "group": info["group"], "prevalence": info["prevalence"],
                  "n_candidates": len(scored), "candidates_json": json.dumps(scored)}
        if not scored:
            records.append(record | {"description": None, "correlation": None, "p_value": None,
                                     "kept": False, "reason": "too_few_firing_transcripts"})
            continue
        best = max(scored, key=lambda item: item["correlation"])
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
    parser.add_argument("--offline", action="store_true",
                        help="call nothing; write uncached prompts to pending.jsonl in the output dir, "
                             "to be answered elsewhere and appended to llm_cache.jsonl as {key, reply}")
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
    ask = CachedAsker(args.output_dir / "llm_cache.jsonl", settings["openrouter_endpoints"],
                      dry_run=args.dry_run, offline=args.offline)
    table = describe_features(activations, labels, rows, groups, settings, ask, width, workers=args.workers)
    if args.dry_run:
        print(f"features={len(labels)} calls={ask.calls} prompt_chars={ask.chars} (~{ask.chars // 4} tokens)")
        return
    if table is None:
        pending_path = args.output_dir / "pending.jsonl"
        pending_path.write_text("".join(json.dumps(item) + "\n" for item in ask.pending))
        by_model = {}
        for item in ask.pending:
            by_model[item["model"]] = by_model.get(item["model"], 0) + 1
        print(f"pending prompts written to {pending_path}: {by_model}")
        return
    table.to_csv(args.output_dir / "feature_descriptions.csv", index=False)
    print(f"features={len(table)} kept={int(table.kept.sum())} "
          f"median_correlation={table.correlation.median():.2f} calls={ask.calls}")


if __name__ == "__main__":
    main()
