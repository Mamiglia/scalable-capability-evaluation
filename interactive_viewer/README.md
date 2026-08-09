# judge-pilot interactive viewer

A static, self-contained dashboard for exploring the judge-pilot results: the
capability-gap curve (judge accuracy vs pair gap), with click-through to the
transcripts each judge saw, both counterbalanced orders, every judge's verdict
and explanation, and per-comparison truncation levels.

The Dataset dropdown switches between the exported datasets (currently
`judge-pilot` and `gpqa-pilot-rd2`); each is a self-contained tree under
`data/{id}/`.

No build step, no server code, no CDN — just `index.html` + `app.js` +
`style.css` + a vendored `plotly.min.js` + the exported `data/`.

## View locally

`fetch()` needs HTTP (not `file://`), so serve the folder:

```bash
python3 -m http.server 8765 --directory interactive_viewer
# open http://localhost:8765/
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
