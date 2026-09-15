# judge-pilot interactive viewer

A static, self-contained dashboard for exploring the judge-pilot results: the
capability-gap curve (judge accuracy vs pair gap), with click-through to the
transcripts each judge saw, both counterbalanced orders, every judge's verdict
and explanation, and per-comparison truncation levels.

The View control switches between the **gap curve** and the **BT ranking**: the ranking
the judge's verdicts imply (Bradley-Terry ability from `ranking.csv`) plotted against the
ground-truth accuracy it is scored on, with the rank-correlation row from `scores.csv` and a
table showing where the judge's rank and the benchmark's disagree. It is offered only for
datasets exported from a log dir that has been through `uv run rank`.

The Dataset dropdown switches between the exported datasets (currently
`judge-pilot` and `gpqa-pilot-rd2`); each is a self-contained tree under
`data/{id}/`.

No build step, no server code, no CDN — just `index.html` + `app.js` +
`style.css` + a vendored `plotly.min.js` + the exported `data/`. Light theme only.

Each verdict row carries the judge's pick, confidence, explanation, and — collapsed — the
judge's own reasoning trace when the provider returned one. The modal header shows that
run's `judge.scorer` metrics (consistent accuracy, position bias, length correlation,
disagreement, no-answer rate); the same metrics pooled over every cross-model judgement in
the dataset sit in the table under the gap curve. `smoke.js` checks all of this against the
exported data — see the header of that file for how to run it.

## View locally

`fetch()` needs HTTP (not `file://`), so serve the folder:

```bash
python3 -m http.server 8765 --directory interactive_viewer
# open http://localhost:8765/
```

Over SSH, forward the port instead of exposing it — from your laptop:

```bash
ssh -N -L 8765:localhost:8765 tao@loki   # leave running, then open http://localhost:8765/
```

## Regenerate the data

The viewer reads `data/` produced from the judge/source `.eval` logs (which
stay local — they are not shipped). Which datasets get exported, and the labels
the dropdown shows, are listed in `configs/dashboard_datasets.yaml`. Re-run
after the logs or analysis change:

```bash
uv run python scripts/analyses/export_dashboard.py                  # every dataset
uv run python scripts/analyses/export_dashboard.py --only judge-pilot   # just one
```

### Domain-pipeline runs (`logs/rank/...`)

Runs from the new domain pipeline export straight from their judge logs — the pair metadata
carries the model shortnames and the ground-truth q_dst accuracies, and the stored prompt
carries the transcripts, so `data/` is never re-read and no `judge_results/` CSVs are needed:

```bash
uv run python scripts/analyses/export_rank_dashboard.py logs/rank/math/gemma-4-31b-it/*/
```

One dataset per condition dir, id `<domain>-q<q_src>-<judge>`, prepended to the menu.

The X-axis control offers every contiguous quartile subset of the domain (`Q1`…`Q4`,
`Q1–Q2`, `Q2–Q4`, …, `all`), computed from the old `data/meta/domains/<domain>/` artefacts
(`uv run domain`, removed 2026-09-15 with the quartile pipeline) — without them the axis falls
back to the `q_dst` accuracy recorded in the judge logs. It opens on `q_dst`, the subset the
pairs were oriented by; pick another and the hint line says so, since which model counts as
"major" does not change with the axis.

The choice drives **both** views: in the BT ranking it becomes the ground-truth axis, the
accuracy column, the accuracy rank (so the &Delta; column moves too), and the Kendall &tau;
in the caption, which is computed from the plotted points. That &tau; runs a few points off
the `scores.csv` row `uv run rank` wrote, because `ranking/scoring.py` means accuracy over
response rows while the pair orientation (`ExperimentConfig.accuracy`) averages repeats per
item first — models here have 2-5 repeats, so the two differ by up to 0.07 per model.

`include_reasoning` is part of the condition hash, so the CoT and no-CoT runs of a quartile
are separate datasets; the id carries `-cot` and the label says which.

Exported judge reasoning is what makes this big: ~4 kB per judgement on top of transcripts
that are themselves several times longer under CoT. The six MATH datasets come to ~350 MB.

`data/datasets.json` is the dropdown's menu (`[{id, label}]`, in config order).
Each `data/{id}/` holds `index.json` (plot points, metadata, fits, questions),
`transcripts/{model}.json` (deduped transcript text, `__ep1` for self-pairs),
`options/{model}.json` (that model's shuffled option order),
`answers/{model}.json` (the letter it answered, taken from the graded score —
not scraped from the transcript, whose format varies by model), and
`pairs/{major}__{minor}__{frac}.json` (every judge's both-order verdicts).

## Share it

The whole folder is ~100 MB (`data/` ~96 MB across both datasets), largest file
4.3 MB — within every host's limits.

**Drag-and-drop (fastest, no git).** Drop `interactive_viewer/` onto
<https://app.netlify.com/drop> (or Cloudflare Pages) for an instant public URL.

**GitHub Pages (versioned).**
1. `data/` is git-ignored by the repo's bare `data/` rule — un-ignore it first:
   change `data/` to `/data/` in `.gitignore`, or `git add -f interactive_viewer/data`.
2. Commit and push `interactive_viewer/` (including `data/`).
3. Settings → Pages → Deploy from a branch → your branch, folder `/ (root)`.
4. Served at `https://<user>.github.io/<repo>/interactive_viewer/`.
   Pages is free on public repos only; private repos need Pro/Team.

## Layout

```
interactive_viewer/
├── index.html      shell: controls, plot, transcript modal
├── app.js          plot, filters, legend-hover emphasis, modal, fetch/cache
├── style.css       house palette, light + dark
├── plotly.min.js   vendored (Plotly v2.35.2)
└── data/           exported datasets (regenerate with export_dashboard.py)
    ├── datasets.json   dropdown menu: [{id, label}]
    └── {id}/           one self-contained tree per dataset
```
