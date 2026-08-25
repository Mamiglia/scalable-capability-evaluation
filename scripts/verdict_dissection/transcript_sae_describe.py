"""Name each SAE feature from its top-activating transcripts and check the name's fidelity.

For every feature: an LLM writes a one-sentence description from the
top-activating transcripts (plus non-activating contrast transcripts from the
same benchmark group); then, on held-out transcripts it has not seen, the LLM
predicts from the description alone whether each transcript activates the
feature. Fidelity = balanced accuracy of those predictions. Features whose
fidelity is below the configured floor are dropped from the description table.
Paid API calls: run with --dry-run first to see the token volume.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DESCRIBE_SYSTEM = (
    "You are labelling features of a sparse autoencoder trained on embeddings of "
    "model-written answer transcripts. You will see transcripts where a feature fires "
    "strongly and transcripts where it does not fire. Describe, in one sentence, what the "
    "firing transcripts share that the non-firing ones lack. Describe observable properties "
    "of the writing (structure, style, content, tone, length, formatting), not whether the "
    "answer is correct. Reply with the sentence only.")
FIDELITY_SYSTEM = (
    "You will see a one-sentence description of a property of answer transcripts, then "
    "numbered transcripts. For each transcript decide whether the description applies. "
    "Reply with JSON only: a list of the transcript numbers to which the description applies.")


def snippet(text: str, max_chars: int) -> str:
    """Head and tail of a long transcript; the whole text when it fits."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n[...]\n" + text[-half:]


def choose_examples(activations: np.ndarray, rows: pd.DataFrame, *, n_describe: int,
                    n_contrast: int, n_fidelity: int, seed: int) -> dict[str, np.ndarray]:
    """Item-deduplicated top-activating rows for description, disjoint held-out active and
    inactive rows for fidelity, all from the group's own rows (rows and activations aligned)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(-activations, kind="stable")
    active = [index for index in order if activations[index] > 0]
    seen, top = set(), []
    for index in active:
        item = rows.iloc[index].item_group_key
        if item not in seen:
            seen.add(item)
            top.append(index)
    inactive = np.flatnonzero(activations <= 0)
    describe = np.array(top[:n_describe], dtype=int)
    held_active = np.array(top[n_describe:n_describe + n_fidelity], dtype=int)
    inactive = rng.permutation(inactive)
    return {"describe": describe, "contrast": inactive[:n_contrast],
            "held_active": held_active,
            "held_inactive": inactive[n_contrast:n_contrast + len(held_active)]}


def describe_prompt(rows: pd.DataFrame, chosen: dict, max_chars: int) -> str:
    parts = ["Transcripts where the feature FIRES:"]
    for number, index in enumerate(chosen["describe"], 1):
        parts.append(f"\n=== Firing {number} ===\n{snippet(rows.iloc[index].transcript, max_chars)}")
    parts.append("\n\nTranscripts where the feature does NOT fire:")
    for number, index in enumerate(chosen["contrast"], 1):
        parts.append(f"\n=== Non-firing {number} ===\n{snippet(rows.iloc[index].transcript, max_chars)}")
    return "\n".join(parts)


def fidelity_prompt(description: str, rows: pd.DataFrame, indices: np.ndarray, max_chars: int) -> str:
    parts = [f"Description: {description}\n"]
    for number, index in enumerate(indices, 1):
        parts.append(f"\n=== Transcript {number} ===\n{snippet(rows.iloc[index].transcript, max_chars)}")
    return "\n".join(parts)


def parse_numbers(reply: str) -> set[int]:
    match = re.search(r"\[[\d,\s]*\]", reply)
    return set(json.loads(match.group(0))) if match else set()


def balanced_accuracy(predicted: set[int], truth: np.ndarray) -> float:
    """truth[i] is whether numbered transcript i+1 activates."""
    positives = {number for number, value in enumerate(truth, 1) if value}
    negatives = {number for number, value in enumerate(truth, 1) if not value}
    recall = len(predicted & positives) / len(positives)
    specificity = len(negatives - predicted) / len(negatives)
    return (recall + specificity) / 2


def make_ask(model: str):
    if model.startswith("claude-"):
        import anthropic
        client = anthropic.Anthropic()

        def ask(system, user):
            with client.messages.stream(model=model, max_tokens=4096, system=system,
                                        messages=[{"role": "user", "content": user}]) as stream:
                reply = stream.get_final_message()
            return "".join(block.text for block in reply.content if block.type == "text")
    else:
        import openai
        client = openai.OpenAI()

        def ask(system, user):
            reply = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": user}])
            return reply.choices[0].message.content
    return ask


def describe_features(activations: np.ndarray, labels: list[str], rows: pd.DataFrame,
                      groups: dict[str, list[str]], settings: dict, ask, log) -> pd.DataFrame:
    records = []
    for column, label in enumerate(labels):
        group = label.split(":")[0]
        mask = rows["benchmark"].isin(groups[group]).to_numpy()
        group_rows = rows[mask].reset_index(drop=True)
        chosen = choose_examples(
            activations[mask, column], group_rows, n_describe=settings["describe_examples"],
            n_contrast=settings["contrast_examples"], n_fidelity=settings["fidelity_examples"],
            seed=settings["seed"])
        record = {"feature": label, "group": group,
                  "active_rows": int((activations[mask, column] > 0).sum()),
                  "held_out_active": len(chosen["held_active"])}
        if len(chosen["describe"]) < settings["describe_examples"] or not len(chosen["held_active"]):
            records.append(record | {"description": None, "fidelity": None, "kept": False,
                                     "reason": "too_few_active_transcripts"})
            continue
        description = ask(DESCRIBE_SYSTEM, describe_prompt(group_rows, chosen, settings["example_chars"])).strip()
        held = np.concatenate([chosen["held_active"], chosen["held_inactive"]])
        truth = np.concatenate([np.ones(len(chosen["held_active"]), bool),
                                np.zeros(len(chosen["held_inactive"]), bool)])
        permutation = np.random.default_rng(settings["seed"] + column).permutation(len(held))
        held, truth = held[permutation], truth[permutation]
        reply = ask(FIDELITY_SYSTEM, fidelity_prompt(description, group_rows, held, settings["example_chars"]))
        fidelity = balanced_accuracy(parse_numbers(reply), truth)
        record |= {"description": description, "fidelity": fidelity,
                   "kept": fidelity >= settings["min_fidelity"], "reason": None}
        log({"feature": label, "description": description, "fidelity": fidelity,
             "fidelity_reply": reply, "held_out_keys": group_rows.iloc[held].transcript_key.tolist(),
             "truth": truth.tolist()})
        records.append(record)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True, help="for report_rows.parquet (transcript text)")
    parser.add_argument("--config", type=Path, default=Path("configs/workstream_b.yaml"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="count prompt characters, call nothing")
    parser.add_argument("--max-features", type=int, help="describe only the first N features (smoke)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings, groups = config["descriptions"], config["sae"]["fit_groups"]
    rows = pd.read_parquet(args.dense_dir / "report_rows.parquet")
    sae_rows = pd.read_parquet(args.sae_dir / "rows.parquet")
    if not sae_rows["transcript_key"].equals(rows["transcript_key"]):
        raise ValueError("report rows are not in SAE row order")
    activations = np.load(args.sae_dir / "activations.npy", allow_pickle=False)
    labels = [str(label) for label in np.load(args.sae_dir / "feature_labels.npy", allow_pickle=False)]
    if args.max_features:
        labels, activations = labels[:args.max_features], activations[:, :args.max_features]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chars = [0]
    if args.dry_run:
        def ask(system, user):
            chars[0] += len(system) + len(user)
            return "[]"
        log = lambda record: None
    else:
        ask = make_ask(args.model)
        trace = (args.output_dir / "description_trace.jsonl").open("a")
        log = lambda record: trace.write(json.dumps(record) + "\n")
    table = describe_features(activations, labels, rows, groups, settings, ask, log)
    if args.dry_run:
        print(f"features={len(labels)} prompt_chars={chars[0]} (~{chars[0] // 4} tokens)")
        return
    table.to_csv(args.output_dir / "feature_descriptions.csv", index=False)
    print(f"features={len(table)} kept={int(table.kept.sum())} "
          f"median_fidelity={table.fidelity.median():.2f}")


if __name__ == "__main__":
    main()
