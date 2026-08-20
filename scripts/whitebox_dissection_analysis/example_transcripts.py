"""Coloured example prompts: each sentence tinted by its fitted per-token effect.

Blue = pushes the judge towards transcript 1, red = towards transcript 2,
intensity = per-token effect size within the prompt.
"""
import html
import pathlib

import numpy as np
import pandas as pd

S = pathlib.Path("/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/8727e515-2405-4685-a43d-ead3d07c1e1c/scratchpad")
OUT = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge_results/whitebox_dissection")

scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
pairs = pd.read_parquet(S / "pairs_ceph.parquet")
pairs = pairs[pairs.split == "train"]
meta = pairs.drop_duplicates(["pair_uid", "order"]).set_index(["pair_uid", "order"])

per = scores.groupby(["pair_uid", "order"]).agg(
    gap=("gap", "first"), r2=("heldout_r2", "first")).reset_index()
good = per[per.r2 > 0.25]
examples = pd.concat([
    good.nlargest(1, "gap"),                       # strong pick-1
    good.nsmallest(1, "gap"),                      # strong pick-2
    good.iloc[(good.gap.abs()).argsort()[:1]],     # near-tied
])

GROUP_NAMES = {"instr_format": "Instructions", "slot_labels": "Slot labels / options",
               "question": "Question", "transcript_1": "Transcript 1",
               "transcript_2": "Transcript 2", "answer_readout": "Answer cue"}

blocks = []
for _, ex in examples.iterrows():
    uid, order = ex.pair_uid, int(ex.order)
    units = scores[(scores.pair_uid == uid) & (scores.order == order)].sort_values("unit")
    row = meta.loc[(uid, order)]
    scale = units.beta_per_token.abs().quantile(0.95) or 1.0
    parts, last_group = [], None
    for _, u in units.iterrows():
        if u.group != last_group:
            parts.append(f"<h4>{GROUP_NAMES[u.group]}</h4>")
            last_group = u.group
        a = min(abs(u.beta_per_token) / scale, 1.0) * 0.75
        rgb = "42,120,214" if u.beta_per_token > 0 else "214,69,80"
        parts.append(
            f'<span title="per-token effect {u.beta_per_token:+.3f}" '
            f'style="background-color: rgba({rgb},{a:.2f})">{html.escape(str(u.text))}</span> ')
    pick = "1" if ex.gap > 0 else "2"
    caps = f"slot 1: {row.model_1} (capability {row.cap_1:+.2f}) · slot 2: {row.model_2} ({row.cap_2:+.2f})"
    truth = "1" if row.cap_1 > row.cap_2 else "2"
    verdictcls = "ok" if pick == truth else "bad"
    blocks.append(
        f"<section><div class='hdr'><b>{caps}</b><br>"
        f"verdict log-odds {ex.gap:+.2f} → judge picks transcript {pick} "
        f"<span class='{verdictcls}'>({'correct' if pick == truth else 'wrong'} — "
        f"more capable model is in slot {truth})</span> · "
        f"held-out R² {ex.r2:.2f}</div><div class='prompt'>{''.join(parts)}</div></section>")

page = f"""<meta charset="utf-8">
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 62rem; margin: 2rem auto; line-height: 1.55; }}
.prompt {{ font-size: 0.82rem; white-space: pre-wrap; }}
.hdr {{ margin: 2.2rem 0 0.6rem; padding: 0.5rem 0.7rem; background: #f2f2f2; border-radius: 6px; font-size: 0.85rem; }}
h4 {{ margin: 1.1rem 0 0.2rem; color: #666; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
.ok {{ color: #1baf7a; }} .bad {{ color: #d64550; }}
.legend span {{ padding: 0.1rem 0.45rem; border-radius: 3px; }}
</style>
<h2>Example prompts, sentence effects coloured</h2>
<p class="legend">Each sentence is tinted by its fitted per-token effect on the verdict:
<span style="background: rgba(42,120,214,0.55)">pushes towards transcript 1</span> ·
<span style="background: rgba(214,69,80,0.55)">pushes towards transcript 2</span>.
Intensity is scaled within each prompt (95th percentile). Hover a sentence for its value.
Effects are fitted from 512 random 8-token blankings per prompt (Llama-3.2-3B-Instruct judge).</p>
{''.join(blocks)}
"""
(OUT / "example_transcripts.html").write_text(page)
print("examples:", [(r.pair_uid[:18], int(r.order), round(r.gap, 2)) for _, r in examples.iterrows()])
