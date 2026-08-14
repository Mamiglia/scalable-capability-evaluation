#!/usr/bin/env python3
"""Compute the frozen GPQA transcript features on the text a judge saw.

gpqa_tfa measures which model-level transcript features track held-out GPQA accuracy.
This module lets the judge analysis measure which of the same features track the
judge's choice. Comparing the two coefficients is the point: a feature that moves
the judge but not accuracy is a judge bias.

The required feature code and capability results are frozen beside this module.
A fresh preparation or bias-map build therefore does not need the sibling
gpqa_tfa checkout. ``GPQA_TFA_DATA`` is needed only for the optional parity
check against upstream feature parquets.

Two things must match gpqa_tfa or the coefficients are not comparable:

  view         gpqa_tfa's primary results are computed on `answer_masked`, so the
               same view transform is applied here. The judge prompt carries the
               question, so the question-echo strip works as it does upstream.

  process gate process features are NaN, not 0, on answer-only and empty
               transcripts. Measured on the judged corpus this fires on 0.8% of
               rows and is flat across truncation levels, so truncation does not
               make it materially more common than upstream.

Self-check:
    transcript_features.py --self-check
compares this path against gpqa_tfa's frozen per-record parquets.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import gpqa_features

# Extracted from gpqa_tfa/evidence/primary_and_robustness.csv, whose SHA-256 was
# ca617a2aaf6d94a85ac9365b17f245048dd0af5cf48b9dd7eaad926bfe268118.
CAPABILITY_PATH = Path(__file__).with_name("capability_features.csv")
CAPABILITY_SHA256 = "fcc67b4df1bbdd75aa962cac656cd4fe88ab604c4ba07a1bb83c505c8d4485dc"

# The judge prompt puts the question in the first fenced block.
QUESTION_RE = re.compile(r"Input question:\s*\n---\n(.*?)\n---\n", re.DOTALL)
FEATURE_NAMES = tuple(sorted(
    set(gpqa_features.lexical_features("placeholder text here for the schema."))
    | set(gpqa_features.PROCESS_KEYS)
    | {"reasoning_to_response_ratio"}
))


def feature_names() -> tuple[str, ...]:
    """The 138 feature names, in a fixed order.

    Both upstream functions assign every key unconditionally, so any non-empty
    text yields the full schema; gpqa_tfa builds PROCESS_KEYS the same way.
    """
    return FEATURE_NAMES


# Frozen from gpqa_tfa/src/stats/primary_analysis.py at LASR commit
# 63605fd2199876cbdfb7cb80bdd2cd315a4d55d8. Its source SHA-256 was
# b17af7451d17a5cf95dd4c8eb918cb6db0713fc73886c8bea419d71a7222d112.
FAMILY_PATTERNS = [
    (r"^epca_", "embedding_pca"), (r"^sae_s\d+_", "sae_latent"),
    (r"^nmf\d+_", "nmf_topic"), (r"^wlsa_", "word_lsa"), (r"^clsa_", "char_lsa"),
    (r"^lex_", "reasoning_lexicon"), (r"^proc_", "process"), (r"^fw_", "function_word"),
    (r"^n_(headings|bullets|numbered|bold|italic|code|inline|math|tables|blockquote)"
     r"|_per_kword$|^frac_blank|^has_any_markdown", "formatting"),
    (r"(comma|period|colon|semicolon|qmark|exclam|lparen|dash|quote|ellipsis)_per_kword"
     r"|upper|allcaps|digit", "punctuation_casing"),
    (r"^(ttr|root_ttr|hapax|top_word|repetition|n_types)", "lexical_diversity"),
    (r"^(n_chars|n_words|n_sentences|n_paragraphs|n_lines|mean_|log_n_words)",
     "length_structure"),
    (r"answer_mask|option|reasoning_to_response", "answer_behaviour"),
]


def feature_family(name: str) -> str:
    for pattern, family in FAMILY_PATTERNS:
        if re.search(pattern, name):
            return family
    return "other"


def families() -> dict[str, list[str]]:
    """Feature names grouped by family, in the gpqa_tfa taxonomy."""
    grouped: dict[str, list[str]] = {}
    for name in feature_names():
        grouped.setdefault(feature_family(name), []).append(name)
    return grouped


def features(
    text: str,
    question: str = "",
    *,
    visibility: str | None = None,
) -> dict[str, float]:
    """The gpqa_tfa lexical + process features for one transcript as judged.

    `text` is the transcript exactly as it appeared in the judge prompt, so any
    truncation the judge task applied is already reflected in the counts.
    """
    view = gpqa_features.answer_masked(text or "", question or "")
    feats = gpqa_features.lexical_features(view)
    visibility = visibility or gpqa_features.classify_visibility(text or "")
    if visibility in ("answer_only", "empty"):
        feats.update(gpqa_features.PROCESS_NAN)
    else:
        feats.update(gpqa_features.process_features(view, feats["n_words"]))
    # Upstream computes this from the parsed record. The judge prompt shows one
    # concatenated transcript, so reasoning cannot be separated from response.
    feats["reasoning_to_response_ratio"] = np.nan
    return feats


def question_of(prompt: str) -> str:
    match = QUESTION_RE.search(prompt or "")
    return match.group(1) if match else ""


def deltas(major_text: str, minor_text: str, question: str = "") -> dict[str, float]:
    """major minus minor, for every feature.

    Signed against the major model, not against the left slot, so the value does
    not flip when the same pair is shown in the reversed order.
    """
    a, b = features(major_text, question), features(minor_text, question)
    return {f"d_{k}": a[k] - b[k] for k in feature_names()}


def capability_betas() -> pd.DataFrame:
    """Frozen feature-to-accuracy coefficients used by the bias comparison."""
    digest = hashlib.sha256(CAPABILITY_PATH.read_bytes()).hexdigest()
    if digest != CAPABILITY_SHA256:
        raise ValueError(
            f"{CAPABILITY_PATH} has SHA-256 {digest}, expected {CAPABILITY_SHA256}"
        )
    frame = pd.read_csv(CAPABILITY_PATH)
    if (len(frame) != 138 or set(frame["method"]) != {"hand-built"}
            or set(frame["feature"]) != set(FEATURE_NAMES)):
        raise ValueError(
            f"{CAPABILITY_PATH} must contain the 138 frozen hand-built features"
        )
    return frame.rename(columns={"beta": "beta_capability", "q_method": "q_capability"})[
        ["feature", "method", "beta_capability", "q_capability", "discovery",
         "beta_external", "p_external"]
    ]


EVIDENCE_PATH = Path(__file__).with_name("learned_capability_features.csv")
EVIDENCE_SHA256 = "ca617a2aaf6d94a85ac9365b17f245048dd0af5cf48b9dd7eaad926bfe268118"


def evidence_betas(method: str) -> pd.DataFrame:
    """Capability coefficients for one gpqa_tfa method (TF-IDF or SAE).

    Same columns as capability_betas(). The hand-built rows have a smaller local
    table. The other methods use the complete frozen table beside this module.
    """
    if method == "hand-built":
        return capability_betas()
    digest = hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest()
    if digest != EVIDENCE_SHA256:
        raise ValueError(
            f"{EVIDENCE_PATH} has SHA-256 {digest}, expected {EVIDENCE_SHA256}"
        )
    frame = pd.read_csv(EVIDENCE_PATH)
    frame = frame[frame["method"] == method]
    if frame.empty:
        raise ValueError(f"no rows with method {method!r} in {EVIDENCE_PATH}")
    return frame.rename(columns={"beta": "beta_capability", "q_method": "q_capability"})[
        ["feature", "method", "beta_capability", "q_capability", "discovery",
         "beta_external", "p_external"]
    ]


def _self_check(n_runs: int = 3, n_records: int = 400) -> int:
    """Reproduce gpqa_tfa's frozen features from its own view text.

    Two separate claims are checked, because they can fail independently:
      1. the local feature functions reproduce the frozen feature parquet
      2. the view transform reproduces the frozen `answer_masked` text
    """
    tfa_data = Path(os.environ.get("GPQA_TFA_DATA", "/fast/jyeon/gpqa_tfa"))
    views_dir = tfa_data / "data_interim/views"
    feats_dir = tfa_data / "data_interim/features/answer_masked"
    runs = sorted(p.stem for p in feats_dir.glob("*.parquet"))[:n_runs]
    if not runs:
        print(f"no frozen features under {feats_dir}")
        return 1

    worst_feat, worst_view, checked = 0.0, 0, 0
    for run in runs:
        view = pd.read_parquet(views_dir / f"{run}.parquet").head(n_records)
        frozen = pd.read_parquet(feats_dir / f"{run}.parquet").head(n_records)
        # The view transform, rebuilt from the raw response. The prompt lives in
        # parsed/, not views/, and without it the question-echo strip is skipped.
        parsed = pd.read_parquet(
            tfa_data / "data_interim/parsed" / f"{run}.parquet", columns=["record_id", "prompt"]
        ).set_index("record_id")["prompt"]
        prompts = [parsed.get(rid, "") for rid in view["record_id"]]
        rebuilt = [
            gpqa_features.answer_masked(r, prompt)
            for r, prompt in zip(view["raw_response"], prompts)
        ]
        worst_view += int((pd.Series(rebuilt) != view["answer_masked"].reset_index(drop=True)).sum())

        # Exercise this bridge, not only the upstream extractor. The reasoning
        # ratio has no equivalent because a judge sees one concatenated text.
        recomputed = pd.DataFrame([
            features(response, prompt, visibility=visibility)
            for response, prompt, visibility in zip(
                view["raw_response"], prompts, view["transcript_visibility"]
            )
        ])
        compared = [name for name in feature_names()
                    if name != "reasoning_to_response_ratio"]
        missing = (set(compared) - set(recomputed)) | (set(compared) - set(frozen))
        if missing:
            print(f"feature schema mismatch in {run}: {sorted(missing)[:8]}")
            return 1
        a = recomputed[compared].to_numpy(float)
        b = frozen[compared].to_numpy(float)
        if np.any(np.isnan(a) != np.isnan(b)):
            print(f"NaN-mask mismatch in {run}")
            return 1
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            worst_feat = max(worst_feat, float(np.abs(a[finite] - b[finite]).max()))
        checked += len(view)

    family_complete = all(feature_family(name) != "other" for name in feature_names())

    print(
        f"runs {len(runs)}  records {checked}  "
        f"comparable features {len(compared)} of {len(feature_names())}"
    )
    print(f"max |recomputed - frozen| over all features : {worst_feat:.3e}")
    print(f"view-transform mismatches                   : {worst_view} of {checked}")
    print(f"all features have a frozen family           : {family_complete}")
    for family, names in sorted(families().items(), key=lambda kv: -len(kv[1])):
        print(f"    {family:20s} {len(names):3d}")
    ok = worst_feat < 1e-6 and worst_view == 0 and family_complete
    print("FEATURE PARITY:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true",
                    help="verify parity against gpqa_tfa's frozen per-record features")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--records", type=int, default=400)
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check(args.runs, args.records)
    names = feature_names()
    print(f"{len(names)} features: {', '.join(names[:8])} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
