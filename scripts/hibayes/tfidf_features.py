"""Judge-side extraction of gpqa_tfa's 240 TF-IDF topic features.

The per-method decomposition needs judge coefficients for the TF-IDF component
features (nmf16/nmf32/nmf64 over word tf-idf, wlsa and clsa). The fitted
vectorizers and component models are gpqa_tfa's own, saved by topic_refit.py and
verified reproducible here by tfidf_parity.py.

Steps, mirroring prepare_gpqa exactly where the hand-built pipeline made a
choice:
  1. Walk the judge logs, apply the answer_masked view to each transcript, and
     project through the saved models. Deltas are major minus minor, keyed by
     the sample uuid.
  2. Inner-join on sample_uuid against gpqa_correctness.csv, so the row set is
     identical to the hand-built within-pair analysis.
  3. Scale each delta by its SD over contested rows without centring, and split
     into _within and _between parts by pair mean, as prepare_gpqa does.
  4. Write analysis/hibayes/features_tfidf/complete_cases.csv and one marginal
     config per component, ready for the features.sub sweep.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd

from gpqa_features import answer_masked
from prepare_gpqa import (LEFT_LABELS, judge_answer, log_identity, prompt_text,
                          transcript_text)

LOGS = pathlib.Path("/fast/jyeon/judge_logs/w1_gpqa")
HIB = pathlib.Path("/home/jyeon/LASR/small-judge/analysis/hibayes")
OUT = HIB / "features_tfidf"
BATCH = 4000

# Strict trailing newline, matching transcript_features.QUESTION_RE: without it
# a question containing its own "---" rule would be truncated silently.
QUESTION = re.compile(r"Input question:\s*\n---\n(.*?)\n---\n", re.DOTALL)


def collect_texts(with_question: bool = False):
    """One row per judged sample: uuid, [question,] major view, minor view."""
    rows = []
    seen = set()
    for path in sorted(LOGS.rglob("*.eval")):
        judge, pair, condition = log_identity(path, LOGS)
        with zipfile.ZipFile(path) as archive:
            if "header.json" not in archive.namelist():
                continue
            header = json.loads(archive.read("header.json"))
            if header.get("status") != "success":
                continue
            for name in archive.namelist():
                if not name.startswith("samples/") or not name.endswith(".json"):
                    continue
                sample = json.loads(archive.read(name))
                uuid = sample.get("uuid")
                if not uuid or uuid in seen:
                    continue
                seen.add(uuid)
                answer, _ = judge_answer(sample)
                if answer is None:
                    continue
                prompt = prompt_text(sample)
                left, right = transcript_text(prompt)
                target = sample.get("target")
                major_is_left = target in LEFT_LABELS
                major_text, minor_text = (left, right) if major_is_left else (right, left)
                row = (uuid,
                       answer_masked(major_text, prompt),
                       answer_masked(minor_text, prompt))
                if with_question:
                    q = QUESTION.search(prompt)
                    if q is None:
                        raise ValueError(f"no Input question block in {uuid}")
                    row = (uuid, q.group(1).strip(), row[1], row[2])
                rows.append(row)
    return rows


def project(texts, wv, cv, models):
    """240 component scores per text, batched."""
    out = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        Xw, Xc = wv.transform(chunk), cv.transform(chunk)
        cols = []
        for name in ("nmf16", "nmf32", "nmf64", "wlsa", "clsa"):
            X = Xc if name == "clsa" else Xw
            cols.append(models[name].transform(X).astype(np.float32))
        out.append(np.hstack(cols))
        print(f"  projected {min(i + BATCH, len(texts)):,}/{len(texts):,}", flush=True)
    return np.vstack(out)


def main():
    import joblib

    topic_models = pathlib.Path(
        "/fast/jyeon/gpqa_tfa/data_interim/topic_models/answer_masked"
    )
    wv = joblib.load(topic_models / "word_vectorizer.joblib")
    cv = joblib.load(topic_models / "char_vectorizer.joblib")
    models = {n: joblib.load(topic_models / f"{n}_model.joblib")
              for n in ("nmf16", "nmf32", "nmf64", "wlsa", "clsa")}
    names = ([f"nmf16_{i:03d}" for i in range(16)] + [f"nmf32_{i:03d}" for i in range(32)]
             + [f"nmf64_{i:03d}" for i in range(64)] + [f"wlsa_{i:03d}" for i in range(64)]
             + [f"clsa_{i:03d}" for i in range(64)])

    rows = collect_texts()
    print(f"judged samples with uuid and answer: {len(rows):,}")
    uuids = [r[0] for r in rows]
    # Dedup texts: reverse-order samples reuse the same transcripts.
    uniq = {}
    for _, a, b in rows:
        for t in (a, b):
            uniq.setdefault(t, len(uniq))
    print(f"unique view texts: {len(uniq):,}")
    order = sorted(uniq, key=uniq.get)
    scores = project(order, wv, cv, models)
    S = {t: scores[i] for t, i in uniq.items()}
    deltas = np.stack([S[a] - S[b] for _, a, b in rows])
    feats = pd.DataFrame(deltas, columns=[f"d_{n}" for n in names])
    feats.insert(0, "sample_uuid", uuids)

    keys = pd.read_csv(HIB / "gpqa_correctness.csv", low_memory=False,
                       usecols=["sample_uuid", "is_correct", "judge", "item",
                                "model_pair", "transcript_condition",
                                "major_is_right", "self_comparison"])
    merged = keys.merge(feats, on="sample_uuid", how="inner")
    print(f"rows after join with gpqa_correctness: {len(merged):,} "
          f"(csv has {len(keys):,})")
    assert len(merged) == len(keys), "judge rows missing from the extraction"

    # Scale and decompose exactly as prepare_gpqa: SD over contested rows,
    # no centring, within = scaled minus the pair mean.
    contested = merged["self_comparison"] == 0
    parts = {}
    dropped = []
    for col in [c for c in merged.columns if c.startswith("d_")]:
        scale = merged.loc[contested, col].std()
        if not np.isfinite(scale) or scale == 0:
            dropped.append(col)
            continue
        scaled = merged[col] / scale
        pair_mean = scaled.groupby(merged["model_pair"]).transform("mean")
        parts[f"{col}_between"] = pair_mean
        parts[f"{col}_within"] = scaled - pair_mean
    merged = pd.concat([merged, pd.DataFrame(parts, index=merged.index)], axis=1)
    print(f"constant features dropped: {len(dropped)}")

    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT / "complete_cases.csv", index=False)
    print(f"wrote {OUT / 'complete_cases.csv'}")

    template = (HIB / "features_within" /
                "marginal__formatting__d_n_bold.yaml").read_text()
    written = 0
    for n in names:
        col = f"d_{n}_within"
        if f"d_{n}" in dropped or col not in merged.columns:
            continue
        cfg = (template
               .replace("analysis/hibayes/features_within/complete_cases_factorised.csv",
                        "analysis/hibayes/features_tfidf/complete_cases.csv")
               .replace("d_n_bold_within", col))
        (OUT / f"marginal__tfidf__d_{n}.yaml").write_text(cfg)
        written += 1
    print(f"wrote {written} marginal configs to {OUT}")


if __name__ == "__main__":
    main()
