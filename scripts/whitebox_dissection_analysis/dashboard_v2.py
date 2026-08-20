"""Full transcript dashboard: every prompt, filterable index, click-to-load view.
Levels included per available score parquet (sentence now, word when computed)."""
import html
import json
import pathlib

import numpy as np
import pandas as pd

S = pathlib.Path("/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/8727e515-2405-4685-a43d-ead3d07c1e1c/scratchpad")
OUT = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge_results/whitebox_dissection")

LEVELS = {}
for level, fname in (("sentence", "llama_sentence_scores.parquet"),
                     ("word", "llama_word_scores.parquet")):
    p = S / fname
    if p.exists():
        LEVELS[level] = pd.read_parquet(p)

bg = pd.concat([pd.read_parquet(p) for p in sorted((S / "blankgroups_llama").rglob("part_*.parquet"))],
               ignore_index=True)
sp = pd.concat([pd.read_parquet(p) for p in sorted((S / "slotpref_full_llama").rglob("part_*.parquet"))])
sp["f_blank"] = sp["gap"] - sp["delta"]
blank = sp.set_index(["pair_uid", "order"])["f_blank"]
pairs = pd.read_parquet(S / "pairs_ceph.parquet")
meta = pairs[pairs.split == "train"].drop_duplicates(["pair_uid", "order"]).set_index(["pair_uid", "order"])

base = LEVELS["sentence"]
measured = bg.groupby("group").delta.mean()
n_prompts = base.groupby(["pair_uid", "order"]).ngroups
GROUPS = ["instr_format", "slot_labels", "question", "transcript_1", "transcript_2", "answer_readout"]

factors = {}
for level, sc in LEVELS.items():
    summed = sc.groupby("group").score.sum() / n_prompts
    factors[level] = (measured / summed).to_dict()

tlens = bg[bg.group.isin(["transcript_1", "transcript_2"])].pivot_table(
    index=["pair_uid", "order"], columns="group", values="n_tokens")

index_rows, data = [], {}
for (uid, order), g in base.groupby(["pair_uid", "order"]):
    row = meta.loc[(uid, order)]
    gap = float(g.gap.iloc[0])
    truth = 1 if row.cap_1 > row.cap_2 else 2
    pick = 1 if gap > 0 else 2
    key = f"{uid}|{order}"
    dlen = int(tlens.loc[(uid, order), "transcript_1"] - tlens.loc[(uid, order), "transcript_2"])
    index_rows.append({
        "k": key, "m1": row.model_1, "m2": row.model_2,
        "c1": round(float(row.cap_1), 2), "c2": round(float(row.cap_2), 2),
        "gap": round(gap, 2), "r2": round(float(g.heldout_r2.iloc[0]), 2),
        "dlen": dlen, "truth": truth, "pick": pick, "ok": int(pick == truth)})
    entry = {"f0": round(float(blank.get((uid, order), 0.012)), 3)}
    for level, sc in LEVELS.items():
        u = sc[(sc.pair_uid == uid) & (sc.order == order)].sort_values("unit")
        cal = (u.score * u.group.map(factors[level])).cumsum() + entry["f0"]
        xs = (u.n_tokens.cumsum() / u.n_tokens.sum()).round(4)
        scale = float(u.beta_per_token.abs().quantile(0.95)) or 1.0
        entry[level] = [[str(t), GROUPS.index(gr), round(float(b), 3),
                         float(x), round(float(c), 3)]
                        for t, gr, b, x, c in zip(u.text, u.group, u.beta_per_token, xs, cal)]
        entry[level + "_scale"] = round(scale, 4)
    data[key] = entry

level_opts = "".join(f"<label><input type='radio' name='lvl' value='{l}'"
                     f"{' checked' if l == 'sentence' else ''}> {l}</label> "
                     for l in LEVELS)
page = f"""<meta charset="utf-8">
<style>
body {{ font-family: -apple-system, sans-serif; margin: 1rem 2rem; line-height: 1.5; font-size: 0.85rem; }}
#idx {{ max-height: 17rem; overflow-y: auto; border: 1px solid #ddd; border-radius: 6px; margin: 0.6rem 0; }}
#idx table {{ border-collapse: collapse; width: 100%; font-size: 0.78rem; }}
#idx th, #idx td {{ padding: 0.15rem 0.5rem; text-align: left; white-space: nowrap; }}
#idx th {{ position: sticky; top: 0; background: #f2f2f2; cursor: pointer; }}
#idx tr:hover {{ background: #eef4fd; cursor: pointer; }}
#idx tr.sel {{ background: #dce9fb; }}
.row {{ display: flex; gap: 1.2rem; align-items: flex-start; }}
.prompt {{ flex: 1.35; font-size: 0.78rem; white-space: pre-wrap; max-height: 36rem; overflow-y: auto;
          border: 1px solid #e5e5e5; border-radius: 6px; padding: 0.7rem; }}
.plot {{ flex: 1; position: sticky; top: 1rem; }}
.plot svg {{ width: 100%; border: 1px solid #e5e5e5; border-radius: 6px; }}
.hdr {{ margin: 0.8rem 0 0.6rem; padding: 0.55rem 0.7rem; background: #f2f2f2; border-radius: 6px; }}
h4 {{ margin: 1rem 0 0.2rem; color: #666; font-size: 0.75rem; text-transform: uppercase; }}
.ok {{ color: #1baf7a; font-weight: 600; }} .bad {{ color: #d64550; font-weight: 600; }}
.sent:hover {{ outline: 1.5px solid #444; }}
.legend span, .ctl label {{ padding: 0.1rem 0.45rem; border-radius: 3px; }}
.ctl {{ margin: 0.4rem 0; }}
.cap {{ font-size: 0.72rem; color: #777; margin-top: 0.3rem; }}
</style>
<h2>Transcript dashboard — Llama judge, all {len(index_rows)} prompts</h2>
<p class="legend">Tint = fitted per-token effect:
<span style="background: rgba(42,120,214,0.55)">towards transcript 1</span> ·
<span style="background: rgba(214,69,80,0.55)">towards transcript 2</span>.
Click a row to load it. Hover text to place the marker on the curve.</p>
<div class="ctl">
filter: <select id="fok"><option value="">all</option><option value="1">judge correct</option>
<option value="0">judge wrong</option></select>
&nbsp; search models: <input id="fs" size="24">
&nbsp; highlight level: {level_opts}
&nbsp; <span id="count"></span></div>
<div id="idx"></div>
<div id="view"></div>
<script>
const IDX = {json.dumps(index_rows)};
const DATA = {json.dumps(data)};
const GROUPS = {json.dumps(["Instructions", "Slot labels / options", "Question",
                            "Transcript 1", "Transcript 2", "Answer cue"])};
let sortKey = 'gap', sortDir = 1, selected = null;
function fmt(x) {{ return (x > 0 ? '+' : '') + x; }}
function renderIndex() {{
  const ok = document.getElementById('fok').value;
  const q = document.getElementById('fs').value.toLowerCase();
  let rows = IDX.filter(r => (ok === '' || String(r.ok) === ok) &&
    (q === '' || (r.m1 + ' ' + r.m2).toLowerCase().includes(q)));
  rows.sort((a, b) => (a[sortKey] < b[sortKey] ? -1 : 1) * sortDir);
  document.getElementById('count').textContent = rows.length + ' shown';
  const cols = [['gap', 'verdict'], ['ok', 'correct?'], ['r2', 'R²'], ['dlen', 'Δlen'],
                ['m1', 'slot 1'], ['c1', 'cap 1'], ['m2', 'slot 2'], ['c2', 'cap 2']];
  let h = '<table><tr>' + cols.map(c =>
    `<th data-k="${{c[0]}}">${{c[1]}}</th>`).join('') + '</tr>';
  for (const r of rows.slice(0, 800)) {{
    h += `<tr data-k="${{r.k}}" class="${{r.k === selected ? 'sel' : ''}}">` +
      `<td>${{fmt(r.gap)}}</td><td>${{r.ok ? '✓' : '✗'}}</td><td>${{r.r2}}</td>` +
      `<td>${{fmt(r.dlen)}}</td><td>${{r.m1}}</td><td>${{r.c1}}</td>` +
      `<td>${{r.m2}}</td><td>${{r.c2}}</td></tr>`;
  }}
  document.getElementById('idx').innerHTML = h + '</table>';
  document.querySelectorAll('#idx th').forEach(th => th.onclick = () => {{
    const k = th.dataset.k;
    sortDir = (sortKey === k) ? -sortDir : -1; sortKey = k; renderIndex();
  }});
  document.querySelectorAll('#idx tr[data-k]').forEach(tr =>
    tr.onclick = () => {{ selected = tr.dataset.k; renderIndex(); renderView(); }});
}}
function level() {{ return document.querySelector('input[name=lvl]:checked').value; }}
function renderView() {{
  if (!selected) return;
  const r = IDX.find(x => x.k === selected), d = DATA[selected];
  const units = d[level()], scale = d[level() + '_scale'];
  let spans = '', lastG = -1;
  units.forEach((u, i) => {{
    if (u[1] !== lastG) {{ spans += `<h4>${{GROUPS[u[1]]}}</h4>`; lastG = u[1]; }}
    const a = Math.min(Math.abs(u[2]) / scale, 1) * 0.75;
    const rgb = u[2] > 0 ? '42,120,214' : '214,69,80';
    const esc = u[0].replace(/&/g, '&amp;').replace(/</g, '&lt;');
    spans += `<span class="sent" data-i="${{i}}" title="per-token ${{fmt(u[2])}}"` +
      ` style="background-color: rgba(${{rgb}},${{a.toFixed(2)}})">${{esc}}</span> `;
  }});
  const endY = units[units.length - 1][4];
  const hdr = `<b>Slot 1:</b> ${{r.m1}} (cap ${{fmt(r.c1)}}) &nbsp; <b>Slot 2:</b> ${{r.m2}} (${{fmt(r.c2)}})` +
    ` &nbsp; <b>truly stronger:</b> slot ${{r.truth}}<br><b>judge answered:</b> ${{r.pick}}` +
    ` (log-odds ${{fmt(r.gap)}}) — <span class="${{r.ok ? 'ok' : 'bad'}}">${{r.ok ? 'correct' : 'wrong'}}</span>` +
    ` · <b>additive account:</b> ${{endY > 0 ? 1 : 2}} (curve ends ${{fmt(endY)}}) · R² ${{r.r2}}`;
  document.getElementById('view').innerHTML =
    `<div class="hdr">${{hdr}}</div><div class="row"><div class="prompt">${{spans}}</div>` +
    `<div class="plot"><svg id="svg0" viewBox="0 0 340 300"></svg>` +
    `<div class="cap">calibrated running total; ● = actual verdict; hover text to place marker</div></div></div>`;
  drawCurve(units, r.gap, d.f0);
}}
function drawCurve(units, gap, f0) {{
  const svg = document.getElementById('svg0');
  const W = 340, H = 300, L = 42, R = 12, T = 14, B = 30;
  const ys = units.map(u => u[4]);
  const ymax = Math.max(2, ...ys.map(Math.abs), Math.abs(gap)) * 1.15;
  const sx = x => L + x * (W - L - R), sy = y => T + (1 - (y + ymax) / (2 * ymax)) * (H - T - B);
  let s = `<line x1="${{sx(0)}}" y1="${{sy(0)}}" x2="${{sx(1)}}" y2="${{sy(0)}}" stroke="#bbb" stroke-dasharray="4 3"/>`;
  for (const yv of [-Math.round(ymax / 2), 0, Math.round(ymax / 2)])
    s += `<text x="${{L - 6}}" y="${{sy(yv) + 3}}" font-size="9" fill="#888" text-anchor="end">${{yv}}</text>`;
  s += `<text x="${{(L + W - R) / 2}}" y="${{H - 6}}" font-size="9" fill="#888" text-anchor="middle">position in prompt</text>`;
  let path = `M ${{sx(0)}} ${{sy(f0)}}`;
  units.forEach(u => path += ` L ${{sx(u[3])}} ${{sy(u[4])}}`);
  s += `<path d="${{path}}" fill="none" stroke="#2a78d6" stroke-width="1.6"/>`;
  s += `<circle cx="${{sx(1)}}" cy="${{sy(gap)}}" r="4" fill="#111"/>`;
  s += `<text x="${{sx(1) - 8}}" y="${{sy(gap) + 3}}" font-size="9" fill="#111" text-anchor="end">judge: ${{fmt(gap)}}</text>`;
  s += `<g id="mk" visibility="hidden"><line id="mkl" y1="${{T}}" y2="${{H - B}}" stroke="#444"/>` +
       `<circle id="mkc" r="3.5" fill="#444"/></g>`;
  svg.innerHTML = s;
  document.querySelectorAll('.sent').forEach(el => {{
    el.onmouseenter = () => {{
      const u = units[+el.dataset.i], g = document.getElementById('mk');
      g.setAttribute('visibility', 'visible');
      const lx = sx(u[3]);
      const l = document.getElementById('mkl'); l.setAttribute('x1', lx); l.setAttribute('x2', lx);
      const c = document.getElementById('mkc'); c.setAttribute('cx', lx); c.setAttribute('cy', sy(u[4]));
    }};
    el.onmouseleave = () => document.getElementById('mk').setAttribute('visibility', 'hidden');
  }});
}}
document.getElementById('fok').onchange = renderIndex;
document.getElementById('fs').oninput = renderIndex;
document.querySelectorAll('input[name=lvl]').forEach(el => el.onchange = renderView);
renderIndex();
</script>
"""
(OUT / "llama3.2-3B_transcript_dashboard.html").write_text(page)
print("levels:", list(LEVELS), "| prompts:", len(index_rows),
      "| size MB:", round(len(page) / 1e6, 1))
