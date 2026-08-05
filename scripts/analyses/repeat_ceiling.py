"""How well could any model predict these verdicts?

Judges run at the provider's default temperature, so the same input can get
different answers across draws. A predictor cannot beat that: its floor is the
input's own entropy. This reads the repeat-judgement logs and reports where that
floor sits, on the same scale as score_prediction/metrics.csv (held-out NLL and
accuracy), so those numbers can be read as "close to the limit" or "far from it".

Two estimates bracket the truth:

  plug-in   scores each held-out draw against the distribution of ALL R draws,
            including itself, so it is optimistic - an upper bound.
  leave-one-out
            scores each draw against the other R-1, so its estimate of the
            input's rate is noisy, making it pessimistic - a lower bound.

Both converge as repeats grow; report the pair, not one number.

Usage:
  uv run python scripts/analyses/repeat_ceiling.py
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

from comparisons import JUDGE_ALIASES

LOG_ROOT = Path("logs/gpqa-pilot-rd2/repeat_judgements")
CONFIG = Path("configs/gpqa_repeats_rd2_260805.yaml")
OUT = Path("judge_results/gpqa-pilot-rd2/repeat_ceiling.csv")
MODELS_CSV = Path("judge_results/gpqa-pilot-rd2/models.csv")
SLOTS = ("1", "2")


def collect(log_root: Path, config: Path) -> pd.DataFrame:
    """One row per (judge, pair, item, slot): how many repeats picked slot 1."""
    by_path = {Path(p).resolve(): name
               for name, p in yaml.safe_load(config.read_text())["transcripts"].items()}
    counts: dict[tuple, list[str]] = defaultdict(list)
    n_logs = n_skipped = 0
    for f in sorted(log_root.rglob("*.eval")):
        log = read_eval_log(str(f))
        if log.status != "success":
            n_skipped += 1
            continue
        n_logs += 1
        meta = log.eval.metadata or {}
        raw = log.eval.model.split("/")[-1]
        judge = JUDGE_ALIASES.get(raw, raw)
        major = by_path.get(Path(meta["major_path"]).resolve())
        minor = by_path.get(Path(meta["minor_path"]).resolve())
        for s in log.samples or []:
            item, slot = str(s.id).rsplit("_", 1)
            score = next(iter((s.scores or {}).values()), None)
            picked = getattr(score, "answer", None)
            if picked in SLOTS:
                counts[(judge, major, minor, item, slot)].append(picked)
    rows = [dict(judge=j, major=m, minor=n, item=i, slot=s,
                 n_repeats=len(v), n_slot1=sum(x == "1" for x in v))
            for (j, m, n, i, s), v in counts.items()]
    print(f"{n_logs} logs read ({n_skipped} not success), {len(rows)} distinct inputs")
    return pd.DataFrame(rows)


def bracket(n_repeats: np.ndarray, k: np.ndarray, alpha: float) -> dict:
    """Plug-in (optimistic) and leave-one-out (pessimistic) NLL and accuracy.

    Each held-out draw is scored against a rate estimated from the repeats;
    plug-in includes the draw itself, leave-one-out excludes it. alpha smooths
    the rate so a lone minority draw does not give infinite NLL.
    """
    out = {}
    # Plug-in keeps the held-out draw, so its rate is never 0 and needs no
    # smoothing; smoothing it would put a floor under NLL and stop it bounding
    # a deterministic input at 0. Leave-one-out drops the draw, so a lone
    # minority draw would give an infinite NLL without smoothing.
    for name, drop_self, a in (("plugin", False, 0.0), ("loo", True, alpha)):
        nll, acc, w = 0.0, 0.0, 0
        for R, k1 in zip(n_repeats, k):
            for picked_one, m in ((True, k1), (False, R - k1)):
                if m == 0:
                    continue
                hits = k1 - (1 if (drop_self and picked_one) else 0)
                base = (R - 1) if drop_self else R
                p = (hits + a) / (base + 2 * a)                  # P(slot 1)
                p_true = p if picked_one else 1 - p
                nll += m * -np.log(p_true)
                acc += m * (1.0 if p_true > 0.5 else 0.5 if p_true == 0.5 else 0.0)
                w += m
        out[f"nll_{name}"] = nll / w
        out[f"acc_{name}"] = acc / w
    return out


def agreement(n_repeats: np.ndarray, k: np.ndarray) -> float:
    """P(two draws of the same input agree), unbiased; 1.0 = deterministic."""
    R, k = n_repeats.astype(float), k.astype(float)
    same = k * (k - 1) + (R - k) * (R - k - 1)
    return float(np.mean(same / (R * (R - 1))))


def summarise(df: pd.DataFrame, label: str, alpha: float, gap_max: float) -> dict:
    """Accuracy bracket over all this judge's inputs, and over the close pairs only.

    Close pairs are where the two models' benchmark accuracies differ by less than
    gap_max, so there is little capability difference for the judge to read.
    """
    def br(g):
        if not len(g):
            return None, None
        b = bracket(g.n_repeats.to_numpy(), g.n_slot1.to_numpy(), alpha)
        return b["acc_loo"], b["acc_plugin"]

    lo, hi = br(df)
    close = df[df.d_acc < gap_max]
    lo_c, hi_c = br(close)
    return {
        "model": label,
        "acc_ceiling_lo": lo,
        "acc_ceiling_hi": hi,
        "close_acc_ceiling_lo": lo_c,
        "close_acc_ceiling_hi": hi_c,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", type=Path, default=LOG_ROOT)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Laplace smoothing on the estimated rate")
    ap.add_argument("--gap-max", type=float, default=0.10,
                    help="close pairs = benchmark accuracies within this")
    ap.add_argument("--min-repeats", type=int, default=2,
                    help="inputs with fewer repeats cannot bracket anything")
    args = ap.parse_args()

    df = collect(args.log_root, args.config)
    if df.empty:
        print("no repeat logs yet")
        return 1
    df = df[df.n_repeats >= args.min_repeats]

    # capability gap of each pair, from the same accuracies the analyses use
    acc = pd.read_csv(MODELS_CSV).set_index("model")["accuracy"].to_dict()
    df["d_acc"] = (df.major.map(acc) - df.minor.map(acc)).abs()
    if df.d_acc.isna().any():
        missing = sorted(set(df[df.d_acc.isna()].major) | set(df[df.d_acc.isna()].minor))
        print(f"WARNING: no accuracy for {missing}; those inputs are excluded from close-pair columns")

    rows = [summarise(g, judge, args.alpha, args.gap_max)
            for judge, g in sorted(df.groupby("judge"))]
    rows.append(summarise(df, "all models pooled", args.alpha, args.gap_max))
    out = pd.DataFrame(rows).round(3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    inputs_csv = args.out.with_name(args.out.stem + "_inputs.csv")
    df.to_csv(inputs_csv, index=False)
    print(f"\nceilings -> {args.out}")
    print(f"per-input repeat counts -> {inputs_csv}\n")
    print(out.to_string(index=False))
    print(f"\nacc_ceiling_lo..hi = best accuracy any predictor could reach on these"
          f"\nverdicts; the true limit lies between the two."
          f"\nclose_* = same, over pairs whose benchmark accuracies differ by"
          f"\nless than {args.gap_max:.0%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
