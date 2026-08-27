"""Exposition-style features: think-aloud, option-matching, conditional and declarative-label language.

    uv run python scripts/verdict_dissection/features_exposition.py

Counts per 100 words of the assistant prose (ANSWER line removed), from term lists read off the
words that separate preferred from rejected GPQA rollouts of the same model: think-aloud process
talk ("let's", "let me", "now we"), option matching ("the closest", "matches option", "≈"),
conditionals/alternatives ("if", "or", "however", "alternatively"), declarative labels ("Energy:",
bold labels, "Step N", "⇒"), and answer assertions ("the correct answer is"). Also the share of
lines that start with a label. Writes <features_root>/raw/exposition.parquet.
"""
import re
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
ANSWER_LINE = re.compile(r"^\s*\**ANSWER:.*$", re.M)
TERMS = {
    "think_aloud": r"\blet'?s\b|\blet me\b|\bnow (?:we|i|let)\b|\bwe (?:need|want|have) to\b|\bfirst,? (?:we|i|let)\b|\bnext,? (?:we|i|let)\b|\bwait\b|\bhmm\b|\bokay\b|\bso,? (?:we|i|let)\b",
    "option_matching": r"\bclosest\b|\bmatches? (?:option|choice|answer)\b|\bcorresponds? to (?:option|choice)\b|\bamong the (?:options|choices)\b|\bthe options\b|\bnone of the (?:options|choices)\b|≈|\\approx\b|\bapproximately\b|\broughly\b|\babout\b",
    "conditionals": r"\bif\b|\bunless\b|\bor\b|\beither\b|\bhowever\b|\bbut\b|\balternatively\b|\botherwise\b|\bwhereas\b",
    "answer_assertion": r"\bthe (?:correct|right|best) (?:answer|option|choice)\b|\banswer is\b|\bis correct\b|\bmust be\b",
    "consequence": r"\btherefore\b|\bthus\b|\bhence\b|⇒|\\Rightarrow\b|\bso the\b|\bwhich gives\b|\bgiving\b",
}
LABEL_LINE = re.compile(r"^\s*(?:[-*•]\s*)?(?:\*\*)?[A-Z][A-Za-z0-9 ()/'\-]{1,40}(?:\*\*)?:\s", re.M)
STEP_HEADER = re.compile(r"^\s*(?:#+\s*)?(?:\*\*)?step\s*\d+", re.I | re.M)
BOLD_LABEL = re.compile(r"\*\*[^*\n]{1,60}\*\*\s*:?")
PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in TERMS.items()}


def prose_of(parts_json):
    import json
    parts = json.loads(parts_json)
    return "\n".join(sub["content"] or "" for part in parts for sub in part["subparts"] if sub["kind"] == "content")


def exposition_of(parts_json):
    body = ANSWER_LINE.sub("", prose_of(parts_json))
    words = max(len(body.split()), 1)
    lines = [l for l in body.split("\n") if l.strip()]
    out = {f"{name}_per_100_words": 100 * len(p.findall(body)) / words for name, p in PATTERNS.items()}
    out |= {"label_line_share": sum(bool(LABEL_LINE.match(l)) for l in lines) / max(len(lines), 1),
            "step_headers": len(STEP_HEADER.findall(body)),
            "bold_labels_per_100_words": 100 * len(BOLD_LABEL.findall(body)) / words,
            "mean_line_words": words / max(len(lines), 1)}
    return out


def main():
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    features_root = Path(config["features_root"])
    rows = pd.read_parquet(features_root / "raw/surface-textual.parquet", columns=KEYS)
    store = pd.read_parquet(ROOT / config["freeze"] / "transcript_store.parquet", columns=KEYS + ["transcript_parts_json"])
    table = rows.merge(store, on=KEYS, how="left", validate="one_to_one")
    features = pd.DataFrame([exposition_of(p) for p in table.transcript_parts_json])
    out = pd.concat([table[KEYS], features], axis=1)
    out.to_parquet(features_root / "raw/exposition.parquet", index=False)
    print(out.drop(columns=KEYS).groupby(out.bench).mean().round(2).T.to_string())


if __name__ == "__main__":
    main()
