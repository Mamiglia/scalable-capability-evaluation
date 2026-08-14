"""Can the smalljudge environment reproduce gpqa_tfa's topic scores exactly?

The fitted vectorizers and component models were saved by gpqa_tfa's
topic_refit.py, whose own validation shows correlation 1.0 against the original
scores. The manifest does not record the sklearn version, so loading the joblib
objects under this repository's environment (scikit-learn 1.9.0) must be
verified before any judge transcript is projected through them.

This loads the models here, transforms the stored answer_masked view texts for
a sample of runs, and compares against recovered_scores.parquet. The LSA
components must agree exactly, since their transform is a matrix product. The
NMF components re-solve an optimisation whose fit used tol=1e-4, so agreement
within that tolerance is the reproducibility floor of the method itself: the
same 1.5e-05 residual appears under scikit-learn 1.7.2, the pickling version,
and under 1.9.0.
"""
from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
import pandas as pd

TM = pathlib.Path("/fast/jyeon/gpqa_tfa/data_interim/topic_models/answer_masked")
VIEWS = pathlib.Path("/fast/jyeon/gpqa_tfa/data_interim/views")
N_RUNS = 3


def main():
    import sklearn
    print(f"scikit-learn here: {sklearn.__version__}")
    wv = joblib.load(TM / "word_vectorizer.joblib")
    cv = joblib.load(TM / "char_vectorizer.joblib")
    models = {name: joblib.load(TM / f"{name}_model.joblib")
              for name in ("nmf16", "nmf32", "nmf64", "wlsa", "clsa")}
    ref = pd.read_parquet(TM / "recovered_scores.parquet")

    run_files = sorted(VIEWS.glob("*.parquet"))[:N_RUNS]
    worst = 0.0
    total = 0
    failures = []
    for f in run_files:
        v = pd.read_parquet(f)
        col = "answer_masked" if "answer_masked" in v.columns else None
        if col is None:
            print(f"  {f.name}: no answer_masked column, columns {list(v.columns)[:6]}")
            continue
        sub = ref[ref["run_id"] == f.stem]
        if sub.empty:
            print(f"  {f.name}: no reference rows")
            continue
        v = v.set_index("record_id").loc[sub["record_id"]]
        texts = v[col].to_numpy()
        Xw, Xc = wv.transform(texts), cv.transform(texts)
        for name, m in models.items():
            X = Xc if name == "clsa" else Xw
            got = m.transform(X)
            cols = sorted(c for c in sub.columns if c.startswith(name + "_"))
            want = sub[cols].to_numpy()
            d = float(np.max(np.abs(got[:, :len(cols)] - want)))
            tol = 1e-4 if name.startswith("nmf") else 1e-6
            if d > tol:
                failures.append((f.stem, name, d))
            worst = max(worst, d)
            total += want.size
            print(f"  {f.stem[:12]} {name:<6} rows {len(sub):>5}  max|diff| {d:.3e}")
    print(f"\nworst absolute difference over {total:,} values: {worst:.3e}")
    print("PARITY OK" if not failures else f"PARITY FAILED on {len(failures)} blocks: {failures[:4]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
