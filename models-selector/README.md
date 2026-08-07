# models-selector

Self-contained interactive dashboard: AI models by release date × Epoch Capability
Index (real + AA-estimated), coloured by developer, with a half-year × tier
selection grid and Save set. Everything (Plotly, data) is inlined into `index.html`.

## Open locally
Double-click `index.html`, or serve the folder:
`python3 -m http.server --directory models-selector 8770`

## Share (like small-judge-viewer)
Drag this folder onto https://app.netlify.com/drop (or Cloudflare Pages) for an
instant public URL. Or GitHub Pages: commit the folder, Settings → Pages →
your branch, folder `/models-selector`.

Regenerate `index.html` from the editable source
(`interactive_viewer/model_select.html` + `plotly.min.js`) when the data changes.
