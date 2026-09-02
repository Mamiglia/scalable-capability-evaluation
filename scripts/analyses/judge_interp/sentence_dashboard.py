"""Which sentences move each axis, straight from the occlusion data.

For every sentence in every mapped transcript, the contribution is the
drop in the axis score when that sentence is deleted. Two corrections
before ranking: deleting anything shifts the score, and longer
sentences shift it more, so contributions are residuals after
regressing on sentence length - what the sentence does beyond being
that many characters of text.

Ranks sentences per axis, and separately per author model, so the
"which judge-specific features" question can be read off directly:
a sentence high on v_pref_judge but not on v_cap is rewarded for
something other than the ability it signals.

Writes out_dir/sentence_dashboard.html and .csv.

Usage:
  uv run python scripts/analyses/judge_interp/sentence_dashboard.py \
      --maps <dir of token_map jsons> [--top 25]
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load(maps_dir: Path, elo: pd.DataFrame) -> pd.DataFrame:
    model_of = {(t, i): m for t, i, m in
                zip(elo["text_id"], elo["item"], elo["model"])}
    rows = []
    for path in sorted(maps_dir.glob("*.json")):
        d = json.loads(path.read_text())
        key = (d["text_id"], d["item"])
        for occ in d["occlusion"]:
            rec = dict(text_id=d["text_id"], item=d["item"],
                       model=model_of.get(key, "?"), elo=d.get("elo", 0.0),
                       sentence=occ["sentence"],
                       n_chars=occ.get("n_chars", len(occ["sentence"])))
            for axis, delta in occ["delta_last"].items():
                rec[axis] = -delta          # contribution: score with - without
            rows.append(rec)
    return pd.DataFrame(rows)


def residualise(df: pd.DataFrame, axis: str, within: bool = True) -> np.ndarray:
    """Contribution beyond length, and beyond the transcript it sits in.

    Transcripts differ enormously in how much any deletion moves them, so
    without centring within transcript the ranking just finds the most
    volatile transcripts twice - once at each end.
    """
    x, y = df["n_chars"].to_numpy(float), df[axis].to_numpy(float)
    if len(x) < 3:
        return y - y.mean()
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    if not within:
        return resid
    offset = pd.Series(resid).groupby(
        [df["text_id"].to_numpy(), df["item"].to_numpy()]).transform("mean")
    return resid - offset.to_numpy()


def table(df: pd.DataFrame, axis: str, top: int) -> str:
    hi = df.nlargest(top, f"{axis}_res")
    lo = df.nsmallest(top, f"{axis}_res")
    def rows(g, cls):
        out = []
        for r in g.itertuples():
            other = "v_cap" if axis == "v_pref_judge" else "v_pref_judge"
            out.append(
                f"<tr class='{cls}'><td class='num'>{getattr(r, axis + '_res'):+.2f}</td>"
                f"<td class='num dim'>{getattr(r, other + '_res'):+.2f}</td>"
                f"<td class='model'>{html.escape(str(r.model))}</td>"
                f"<td>{html.escape(r.sentence[:300])}</td></tr>")
        return "\n".join(out)
    other = "v_cap" if axis == "v_pref_judge" else "v_pref_judge"
    return f"""
<h2>{html.escape(axis)}</h2>
<table>
<thead><tr><th>{html.escape(axis)}</th><th>{html.escape(other)}</th>
<th>author model</th><th>sentence</th></tr></thead>
<tbody>
<tr class='sep'><td colspan='4'>raises the axis most</td></tr>
{rows(hi, 'hi')}
<tr class='sep'><td colspan='4'>lowers the axis most</td></tr>
{rows(lo, 'lo')}
</tbody></table>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-chars", type=int, default=25,
                    help="drop fragments shorter than this: list markers, not sentences")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]

    elo = pd.read_csv(out / "elo.csv")
    df = load(args.maps, elo)
    if df.empty:
        raise SystemExit(f"no token maps under {args.maps}")
    # The splitter turns "5. Because ..." into a bare "5." fragment; those are
    # list markers, not sentences, and they crowd the top of the ranking.
    n_before = len(df)
    df = df[df["n_chars"] >= args.min_chars].reset_index(drop=True)
    print(f"dropped {n_before - len(df)} fragments under {args.min_chars} chars")
    axes = [c for c in ("v_pref_judge", "v_cap", "approval") if c in df.columns]
    for axis in axes:
        # _res: also centred within transcript, for ranking sentences fairly.
        # _len: length-corrected only, since centring within transcript would
        # remove exactly the between-author variation the model table needs.
        df[f"{axis}_res"] = residualise(df, axis)
        df[f"{axis}_len"] = residualise(df, axis, within=False)
    print(f"{len(df)} sentences from {df['text_id'].nunique()} transcripts, "
          f"{df['model'].nunique()} author models")

    # per-model means: does a model's typical sentence sit high or low?
    per_model = df.groupby("model")[[f"{a}_len" for a in axes]].agg(["mean", "count"])
    print("\nper-model mean sentence contribution (residualised):")
    print(per_model.round(3).to_string())

    body = "\n".join(table(df, a, args.top) for a in axes)
    model_rows = "\n".join(
        f"<tr><td>{html.escape(str(m))}</td>"
        + "".join(f"<td class='num'>{per_model.loc[m, (f'{a}_len', 'mean')]:+.3f}</td>"
                  for a in axes)
        + f"<td class='num dim'>{int(per_model.loc[m, (f'{axes[0]}_len', 'count')])}</td></tr>"
        for m in per_model.index)
    page = f"""<!doctype html><meta charset='utf-8'>
<title>Sentences that move the judge's axes</title>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:2rem auto;max-width:1100px;color:#111}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
 th,td{{border-bottom:1px solid #e5e5e5;padding:.4rem .5rem;vertical-align:top;text-align:left}}
 th{{font-weight:600;border-bottom:2px solid #999}}
 .num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap;width:5rem}}
 .dim{{color:#888}} .model{{color:#555;white-space:nowrap;font-size:13px}}
 .sep td{{background:#f6f6f6;font-weight:600;border-bottom:1px solid #ccc}}
 .hi td{{background:#f7fbf7}} .lo td{{background:#fdf7f7}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:2rem}}
 p.note{{color:#555}}
</style>
<h1>Sentences that move the judge's axes</h1>
<p class='note'>{len(df)} sentences from {df['text_id'].nunique()} transcripts.
Contribution = drop in the axis score when the sentence is deleted, with the
effect of sentence length regressed out. The second column shows the same
sentence on the other axis: high on <code>v_pref_judge</code> but flat on
<code>v_cap</code> means the judge rewards it for something other than the
ability it signals.</p>
<h2>per author model</h2>
<table><thead><tr><th>model</th>{"".join(f"<th>{html.escape(a)}</th>" for a in axes)}<th>n</th></tr></thead>
<tbody>{model_rows}</tbody></table>
{body}"""
    (out / "sentence_dashboard.html").write_text(page)
    df.to_csv(out / "sentence_dashboard.csv", index=False)
    print(f"\n-> {out / 'sentence_dashboard.html'}")


if __name__ == "__main__":
    main()
