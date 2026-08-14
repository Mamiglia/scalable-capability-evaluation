"""The transferable taste pattern, topic-controlled.

The sentence-embedding clusters (cluster_features.py) are topic-confounded:
they mostly encode which GPQA problem a sentence came from. To isolate a
transferable within-transcript property, z-score each sentence's
length-adjusted taste occlusion delta (v_pref_gen_orth) within its own
transcript, pool, and read/validate the extremes.

Finding: the transferable taste signal is rhetorical framing, not math
content. Taste rewards the transcript's seams — a confident opening that
announces a methodical plan, and a clean boxed final answer — and is mildly
against the visible working (bare equations, arithmetic, option-elimination,
self-correction). "symbolic math -> taste" from the clusters was topic
confound; what survives topic control is packaging. R2 ~= 0.08 of
within-transcript taste variance.

The category rules below were INDUCED by reading the topic-controlled
extremes first, then tested — not pre-specified.

Usage: uv run python scripts/analyses/judge_interp/taste_pattern.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SENTS = ROOT / "interactive_viewer/data_ml/sentences.json"
AXIS = "v_pref_gen_orth_len"   # length-adjusted taste (preference beyond capability)
MIN_CHARS = 25                 # data_ml is unfiltered; drop bare fragments
MIN_SENTS = 4                  # transcripts need enough sentences to z-score


def category(s):
    """Induced from reading the topic-controlled extremes, then tested."""
    sl = s.strip()
    if "boxed" in s:
        return "boxed final answer"
    if re.search(r"(?i)\b(wait|re-evaluat|re-examine|simpler way|reconsider)\b", s):
        return "self-correction / rethink"
    if re.search(r"[A-D]\)\s*(Yes|No|Incorrect|Correct)", s) or \
       re.search(r"(?i)option\s*[A-D].*(incorrect|correct|->)", s):
        return "option-elimination check"
    symbolic = sum(c in "$\\={}^_" for c in sl) / max(len(sl), 1)
    if sl.startswith("$") or (symbolic > 0.18 and len(re.findall(r"[A-Za-z]{3,}", sl)) < 4):
        return "bare equation/computation"
    if sl.startswith("[Assistant]:") or \
       re.search(r"(?i)^(to (find|determine|solve|design|calculate)|let.?s (analy|break|solve|consider)"
                 r"|step-by-step|first,? we|we (will|need to|use|must))", sl) or \
       re.search(r"(?i)(analyze each|break down|step-by-step analysis)", sl):
        return "confident framing/opening"
    return "other"


def main():
    rows = [r for r in json.loads(SENTS.read_text()) if r["n_chars"] >= MIN_CHARS]
    by = defaultdict(list)
    for r in rows:
        by[r["key"]].append(r)

    z, cats = [], []
    for g in by.values():
        if len(g) < MIN_SENTS:
            continue
        v = np.array([x[AXIS] for x in g])
        zz = (v - v.mean()) / (v.std() + 1e-9)
        for x, zi in zip(g, zz):
            z.append(zi); cats.append(category(x["sentence"]))
    z = np.array(z)

    print(f"{len(z)} sentences, within-transcript z-taste ({AXIS})\n")
    means = {c: z[[i for i, cc in enumerate(cats) if cc == c]] for c in set(cats)}
    print(f"{'category':32} {'n':>4} {'mean z':>8}")
    for c, zi in sorted(means.items(), key=lambda kv: -kv[1].mean()):
        print(f"{c:32} {len(zi):>4} {zi.mean():>+8.2f}")

    pred = np.array([means[c].mean() for c in cats])
    r2 = 1 - ((z - pred) ** 2).sum() / ((z - z.mean()) ** 2).sum()
    print(f"\nwithin-transcript taste variance explained by these categories: R2 = {r2:.3f}")
    # sanity: framing+boxed positive, working negative
    assert means["confident framing/opening"].mean() > 0.3
    assert means["option-elimination check"].mean() < 0


if __name__ == "__main__":
    main()
