"""Compare two independent fidelity annotators of the same feature descriptions.

Inputs: two description output dirs (transcript_sae_describe.py), each with
feature_descriptions.csv, llm_cache.jsonl and the pending.jsonl that listed that annotator's
prompts (kept as pending_<name>.jsonl). Prompts are identical across annotators except for the
model name in the cache key, so replies are aligned by prompt text.

Reports per feature cell: correlation and kept flag under each annotator, kept under both;
and pooled agreement of the presence labels (raw agreement and Cohen's kappa).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from transcript_sae_describe import parse_presence


def presence_matrix(prompt: str, reply: str) -> np.ndarray | None:
    n_concepts = len(re.findall(r"^\d+\. \"", prompt.split("CONCEPTS:\n")[1].split("\n\nTRANSCRIPT")[0], re.M))
    n_transcripts = len(re.findall(r"^TRANSCRIPT \d+:", prompt, re.M))
    return parse_presence(reply, n_concepts, n_transcripts)


def aligned_labels(a_pending: Path, a_cache: Path, b_pending: Path, b_cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Presence labels from both annotators for every prompt both answered, flattened."""
    replies = {}
    for name, cache in (("a", a_cache), ("b", b_cache)):
        replies[name] = {json.loads(line)["key"]: json.loads(line)["reply"] for line in cache.read_text().splitlines()}
    prompts = {}
    for name, pending in (("a", a_pending), ("b", b_pending)):
        for line in pending.read_text().splitlines():
            item = json.loads(line)
            prompts.setdefault(item["prompt"], {})[name] = item["key"]
    a_labels, b_labels = [], []
    for prompt, keys in prompts.items():
        if {"a", "b"} <= set(keys) and keys["a"] in replies["a"] and keys["b"] in replies["b"]:
            a, b = presence_matrix(prompt, replies["a"][keys["a"]]), presence_matrix(prompt, replies["b"][keys["b"]])
            if a is not None and b is not None:
                a_labels.append(a.ravel())
                b_labels.append(b.ravel())
    return np.concatenate(a_labels), np.concatenate(b_labels)


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    agree = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (agree - expected) / (1 - expected) if expected < 1 else float("nan")


def merge_tables(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str) -> pd.DataFrame:
    cols = ["feature", "group", "prevalence", "description", "correlation", "p_value", "kept"]
    out = a[cols].merge(b[["feature", "description", "correlation", "p_value", "kept"]], on="feature",
                        suffixes=(f"_{a_name}", f"_{b_name}"), validate="one_to_one")
    out["kept_both"] = out[f"kept_{a_name}"] & out[f"kept_{b_name}"]
    out["same_description"] = out[f"description_{a_name}"] == out[f"description_{b_name}"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="first annotator's output dir")
    parser.add_argument("--b", type=Path, required=True, help="second annotator's output dir")
    parser.add_argument("--a-name", default="codex")
    parser.add_argument("--b-name", default="haiku")
    parser.add_argument("--output", type=Path, required=True, help="merged table (csv)")
    args = parser.parse_args()
    table = merge_tables(pd.read_csv(args.a / "feature_descriptions.csv"), pd.read_csv(args.b / "feature_descriptions.csv"),
                         args.a_name, args.b_name)
    table.to_csv(args.output, index=False)
    a, b = aligned_labels(args.a / f"pending_{args.a_name}.jsonl", args.a / "llm_cache.jsonl",
                          args.b / f"pending_{args.b_name}.jsonl", args.b / "llm_cache.jsonl")
    print(f"cells={len(table)} kept_{args.a_name}={int(table[f'kept_{args.a_name}'].sum())} "
          f"kept_{args.b_name}={int(table[f'kept_{args.b_name}'].sum())} kept_both={int(table.kept_both.sum())} "
          f"same_best_description={int(table.same_description.sum())}")
    print(f"presence labels compared={len(a)} agreement={float((a == b).mean()):.3f} kappa={kappa(a, b):.3f} "
          f"positive_rate {args.a_name}={a.mean():.3f} {args.b_name}={b.mean():.3f}")
    for group, sub in table.groupby("group"):
        print(f"  {group:<14} kept_{args.a_name}={int(sub[f'kept_{args.a_name}'].sum()):2d} "
              f"kept_{args.b_name}={int(sub[f'kept_{args.b_name}'].sum()):2d} both={int(sub.kept_both.sum()):2d}")


if __name__ == "__main__":
    main()
