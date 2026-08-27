"""Readable contrast set: same two models, same item, same shown order, only the rollouts differ,
and the strong judges flip their verdict the same way between the two rollout pairs.

    uv run python scripts/verdict_dissection/within_model_contrasts.py [--per-bench 10] [--judges gemma-4-31b-it deepseek-v4-flash-0731]

Writes judge_results/verdict_dissection/within_model_contrasts/contrasts.parquet (every qualifying
group, both rollout pairs, picks of every judge) and contrasts.html (the shortest cases per benchmark:
question, the four transcripts in a 2x2 grid, each judge's pick and explanation per rollout pair).
"""
import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from judge.dataset import has_reasoning   # noqa: E402
FREEZE = ROOT / "judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66"
EXPLANATIONS = ROOT / "judge_results/verdict_dissection/judge_explanations.parquet"
OUT = ROOT / "judge_results/verdict_dissection/within_model_contrasts"
GROUP = ["bench", "item", "model_left", "model_right"]
VERDICT_KEY = ["bench", "judge", "judge_log", "judge_sample", "judge_epoch"]


def load_contrasts(agreeing_judges):
    pairs = pd.read_parquet(FREEZE / "pairs.parquet")
    explanations = pd.read_parquet(EXPLANATIONS, columns=VERDICT_KEY + ["explanation"])
    pairs = pairs.merge(explanations, on=VERDICT_KEY, how="left")
    transcripts = load_transcripts()
    answer_only = set(transcripts.index[~transcripts.transcript.map(has_reasoning)])
    pairs = pairs[~pairs.content_hash_l.isin(answer_only) & ~pairs.content_hash_r.isin(answer_only)]
    pairs = pairs[pairs.epoch_left == pairs.epoch_right]          # rollout pair k = (epoch k, epoch k)
    pairs["rollout"] = pairs.epoch_left
    # one row per judge x group with both rollout pairs judged
    counts = pairs.groupby(GROUP + ["judge"]).rollout.transform("nunique")
    pairs = pairs[counts == 2]
    wide = pairs.pivot_table(index=GROUP, columns=["judge", "rollout"], values="pick_left", aggfunc="first")
    # flip = pick on rollout 1 minus pick on rollout 0: +1 or -1 when the verdict changed
    flips = pd.DataFrame({judge: wide[(judge, 1)] - wide[(judge, 0)] for judge in wide.columns.levels[0]})
    keep = flips[agreeing_judges].notna().all(axis=1) & (flips[agreeing_judges] != 0).all(axis=1) \
        & (flips[agreeing_judges].nunique(axis=1) == 1)
    groups = flips[keep]
    rows = pairs.set_index(GROUP).loc[groups.index].reset_index()
    return rows, groups


def load_transcripts():
    store = pd.read_parquet(FREEZE / "transcript_store.parquet",
                            columns=["content_hash", "question", "transcript", "options_json"])
    return store.drop_duplicates("content_hash").set_index("content_hash")


def choose_cases(rows, transcripts, per_bench, seed=None):
    """Shortest cases per benchmark by total transcript characters, or a random draw with a seed."""
    length = transcripts.transcript.str.len()
    case_rows = rows.drop_duplicates(GROUP + ["rollout"])
    case_rows = case_rows.assign(chars=length.loc[case_rows.content_hash_l].values + length.loc[case_rows.content_hash_r].values)
    total = case_rows.groupby(GROUP).chars.sum().reset_index()
    if seed is not None:
        chosen = total.sample(frac=1, random_state=seed).groupby("bench").head(per_bench)
    else:
        chosen = total.sort_values("chars").groupby("bench").head(per_bench)
    return chosen.sort_values(["bench", "chars"])[GROUP].values.tolist()


def render_case(number, key, rows, transcripts, judges):
    bench, item, model_left, model_right = key
    case = rows[(rows.bench == bench) & (rows.item == item) & (rows.model_left == model_left) & (rows.model_right == model_right)]
    first = case.iloc[0]
    question = transcripts.loc[first.content_hash_l].question
    options = transcripts.loc[first.content_hash_l].options_json
    parts = [f'<section class="case" id="case-{number}"><h2>{number}. {html.escape(bench)} item {html.escape(str(item))}</h2>',
             f'<p class="models">left = <b>{html.escape(model_left)}</b> &nbsp; right = <b>{html.escape(model_right)}</b></p>',
             f'<details class="question"><summary>Question</summary><pre>{html.escape(question)}</pre>'
             + (f'<pre>{html.escape(json.dumps(json.loads(options), indent=1))}</pre>' if options else "") + '</details>']
    for rollout in (0, 1):
        draw = case[case.rollout == rollout]
        one = draw.iloc[0]
        picks = {row.judge: row.pick_left for row in draw.itertuples()}
        tally = " &nbsp; ".join(f'<span class="pick {"left" if picks[j] else "right"}">{html.escape(j)}: {"LEFT" if picks[j] else "RIGHT"}</span>'
                                for j in judges if j in picks)
        parts.append(f'<h3>Rollout pair {rollout}</h3><p class="tally">{tally}</p><div class="grid">')
        for side, content_hash in (("left", one.content_hash_l), ("right", one.content_hash_r)):
            won = sum(picks.get(j) == (side == "left") for j in judges if j in picks)
            parts.append(f'<div class="transcript {"won" if won > len(judges) / 2 else ""}"><div class="label">{side}</div>'
                         f'<pre>{html.escape(transcripts.loc[content_hash].transcript)}</pre></div>')
        parts.append('</div><details class="explanations"><summary>Judge explanations</summary>')
        for row in draw.sort_values("judge").itertuples():
            parts.append(f'<p><b>{html.escape(row.judge)}</b> → {"LEFT" if row.pick_left else "RIGHT"}: {html.escape(row.explanation or "")}</p>')
        parts.append('</details>')
    parts.append('</section>')
    return "\n".join(parts)


STYLE = """
body{font:14px/1.4 -apple-system,Helvetica,sans-serif;margin:0;padding:1em 2em;max-width:1600px}
nav a{margin-right:.6em;font-size:12px}
.case{border-top:2px solid #999;margin-top:2em;padding-top:1em}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1em}
.transcript{border:1px solid #ccc;padding:.5em;background:#fafafa}
.transcript.won{border:2px solid #2a7;background:#eefaf3}
.label{font-weight:bold;color:#666;margin-bottom:.3em}
pre{white-space:pre-wrap;word-wrap:break-word;font:12px/1.35 Menlo,monospace;margin:0;max-height:70vh;overflow:auto}
.pick.left{color:#25a}.pick.right{color:#a52}
.tally{font-size:13px}
details.explanations p{font-size:13px;border-left:3px solid #ddd;padding-left:.6em}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bench", type=int, default=10)
    parser.add_argument("--judges", nargs="+", default=["gemma-4-31b-it", "deepseek-v4-flash-0731"],
                        help="judges that must all flip the same way between the two rollout pairs")
    parser.add_argument("--benches", nargs="+", help="only these benchmarks (default: all)")
    parser.add_argument("--random", type=int, help="draw cases at random with this seed instead of the shortest")
    parser.add_argument("--output", default="contrasts.html")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, groups = load_contrasts(args.judges)
    if args.benches:
        rows = rows[rows.bench.isin(args.benches)]
        groups = groups[groups.index.get_level_values("bench").isin(args.benches)]
    rows.drop(columns=["judge_input", "judge_sample_metadata_json"]).to_parquet(OUT / "contrasts.parquet", index=False)
    print(f"{len(groups)} groups where {args.judges} flip the same way; {rows.judge.nunique()} judges")
    transcripts = load_transcripts()
    cases = choose_cases(rows, transcripts, args.per_bench, args.random)
    judges_shown = sorted(rows.judge.unique(), key=lambda j: (j not in args.judges, j))
    nav = " ".join(f'<a href="#case-{i + 1}">{i + 1}</a>' for i in range(len(cases)))
    body = "\n".join(render_case(i + 1, key, rows, transcripts, judges_shown) for i, key in enumerate(cases))
    intro = (f"<p>Same two models, same item, same shown order; only the rollouts differ. "
             f"{' and '.join(args.judges)} both flipped the same way between rollout pair 0 and 1 "
             f"({len(groups)} such groups; showing {args.per_bench} per benchmark, {'a random draw' if args.random is not None else 'the shortest'}). "
             f"Green box = the transcript most judges picked.</p>")
    (OUT / args.output).write_text(f"<!doctype html><meta charset=utf-8><title>Within-model contrasts</title>"
                                        f"<style>{STYLE}</style><h1>Within-model verdict flips</h1>{intro}<nav>{nav}</nav>{body}")
    print(f"wrote {OUT / args.output} with {len(cases)} cases")


if __name__ == "__main__":
    main()
