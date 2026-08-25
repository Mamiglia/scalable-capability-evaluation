// Shared-SAE feature dashboard: data/index.json, data/feature_NN.json and, when traced, data/traces/feature_NN.json.
// Each example shows snippets around the text segments that move the feature most, shaded by their contribution.
const GROUP_LABELS = { gpqa: "GPQA", aime: "AIME", intercode: "InterCode", swe_bench_mini: "SWE-bench mini" };
const BUCKETS = [
  ["top", "Top"], ["50-90%", "50–90% quantile"], ["10-50%", "10–50% quantile"],
  ["0-10%", "0–10% quantile"], ["contrast", "Same question, zero activation"], ["negative", "Most negative"],
];
const SNIPPETS = 3;         // peaks shown per example
const state = { index: null, feature: 0, group: "gpqa", data: null, traces: null, blind: false };

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c !== "" && c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
};
const svgEl = (tag, attrs = {}) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
};
const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const exampleKey = (e) => [e.bench, e.item, e.model, e.epoch, e.log].join("|");

async function loadJson(path) {
  const r = await fetch(path);
  return r.ok ? r.json() : null;
}

// ---- sidebar -------------------------------------------------------------------------------
function renderSidebar() {
  $("#group-tabs").replaceChildren(...state.index.groups.map((g) =>
    el("button", { class: g === state.group ? "active" : "", onclick: () => { state.group = g; renderSidebar(); renderMain(); } }, GROUP_LABELS[g] || g)));
  $("#feature-list").replaceChildren(...state.index.features.map((f) => {
    const g = f.groups[state.group];
    const label = state.blind ? "" : (g.text || "(no description)");
    return el("li", { class: f.feature === state.feature ? "active" : "", onclick: () => selectFeature(f.feature) },
      el("span", { class: "num" }, f.feature),
      el("span", { class: "label" + (g.kept ? "" : " unkept"), title: label }, label),
      el("span", { class: "prev", title: "prevalence: share of this benchmark's transcripts where the feature is nonzero" }, `${(g.prevalence * 100).toFixed(0)}%`));
  }), el("li", { class: "explain" + (state.feature === "explain" ? " active" : ""), onclick: () => { state.feature = "explain"; renderSidebar(); renderExplanation(); } },
    el("span", { class: "num" }, "?"), el("span", { class: "label" }, "Explanation")));
}

const EXPLANATION = `
<h2>Method</h2>
<p>Transcripts are embedded with Qwen3-Embedding-4B (final layer, last token, transcript without the question). Each embedding is centred on the mean embedding of answers to the same question and renormalised. A BatchTopK sparse autoencoder (32 features, 4 active per transcript, signed activations, WIMHF recipe) is fitted on a training set balanced across benchmarks and applied to all 17,451 transcripts. An activation of zero means the feature is not among the transcript's 4 active features.</p>
<p>Names are assigned per benchmark. A language model is shown firing transcripts and zero-activation answers to the same questions and writes 5 candidate descriptions. A second model rates 300 transcripts for each candidate without seeing activations; the candidate with the highest Pearson correlation between its ratings and the activation is the name. The name is kept if prevalence ≥ 5%, at least 10 questions have a firing transcript, and r &gt; 0 with p ≤ 0.05/32. When the best candidate has a significant negative correlation it is kept as its negation ("does NOT: …"). Not kept means no verified name in that benchmark, not that the feature is absent.</p>
<p>Traces: the transcript is re-embedded at every prefix in one forward pass (one extra end token per prefix, masked to its prefix), which gives the feature value after each token. Green marks text where the value rises, red where it falls. The value at the last token equals the stored activation to within 0.03. Transcripts over 16k tokens are not traced.</p>
<h2>Panels</h2>
<p>Histogram of nonzero activations. Mean activation of each model against its accuracy on the benchmark, with Spearman ρ over models. Judged pairs in which the two transcripts differ on the feature: the fraction where the judge picked the transcript with the higher activation, per judge, and the fraction where that transcript is from the more accurate model (orange). 0.5 means no association; bars are Wilson 95% intervals.</p>
<h2>Examples</h2>
<p>Top 10 activations; samples from the 50–90%, 10–50% and 0–10% quantiles of positive activations; zero-activation answers to the same questions as the top examples; the 5 most negative; and for each model its highest, lowest and two zero-activation transcripts. Blind mode hides the names and shows 4 firing and 4 zero-activation transcripts in random order.</p>
<h2>Use in the GLM</h2>
<p>Each feature is one regressor, left minus right activation, in every benchmark. A slope is given its name only where the feature × benchmark cell is kept; slopes are pooled across benchmarks only where the four names describe the same behaviour.</p>`;

function renderExplanation() {
  $("#header").innerHTML = EXPLANATION;
  $("#summary").replaceChildren();
  $("#examples").replaceChildren();
  $("#main").scrollTop = 0;
}

async function selectFeature(i) {
  state.feature = i;
  const id = String(i).padStart(2, "0");
  [state.data, state.traces] = await Promise.all([loadJson(`data/feature_${id}.json`), loadJson(`data/traces/feature_${id}.json`)]);
  renderSidebar();
  renderMain();
  $("#main").scrollTop = 0;
}

// ---- header + density ----------------------------------------------------------------------
function renderHeader(group) {
  const d = group.description;
  const stats = el("div", { class: "stats" },
    `n ${group.n} · fires ${(group.prevalence * 100).toFixed(1)}%`,
    d ? ` · fidelity r ${fmt(d.correlation, 2)}, p ${d.p_value == null ? "–" : d.p_value.toExponential(0)}` : " · not named",
    d && d.reason ? ` · ${d.reason.replaceAll("_", " ")}` : "");
  if (state.blind) {
    $("#header").replaceChildren(el("h2", {}, `shared:${state.feature} @ ${GROUP_LABELS[state.group]}`),
      el("div", { class: "desc" }, "[name hidden]"), stats);
    return;
  }
  $("#header").replaceChildren(
    el("h2", {}, `shared:${state.feature} @ ${GROUP_LABELS[state.group]}`,
      d ? el("span", { class: "badge " + (d.kept ? "kept" : "unkept") }, d.kept ? (d.reason === "negated_candidate" ? "kept, negated" : "kept") : "not kept") : ""),
    el("div", { class: "desc" }, d && d.text ? d.text : "(no description)"),
    stats,
    d && d.candidates.length ? el("details", { class: "candidates" }, el("summary", {}, "candidates"),
      el("ol", {}, ...d.candidates.map((c) => el("li", {}, `${fmt(c.correlation, 3)} — ${c.description}`)))) : "",
    el("table", { class: "names" }, ...state.index.groups.map((g) => {
      const gd = state.data.groups[g].description;
      return el("tr", { class: g === state.group ? "current" : "" },
        el("td", { class: "g" }, GROUP_LABELS[g]),
        el("td", { class: "r" }, gd ? `${(state.data.groups[g].prevalence * 100).toFixed(0)}% · r ${fmt(gd.correlation, 2)}` : "–"),
        el("td", { class: gd && gd.kept ? "" : "unkept" }, gd && gd.text ? gd.text : "(not named)"));
    })));
}

// three summary panels: activation distribution, mean activation per model vs its accuracy, judges' pick rate by activation
function axisText(svg, x, y, text, anchor = "middle", cls = "") {
  const t = svgEl("text", { x, y, "text-anchor": anchor, class: cls }); t.textContent = text; svg.append(t); return t;
}

function panelDistribution(group) {
  const { counts, edges } = group.histogram;
  const W = 230, H = 140, pad = { l: 34, r: 18, t: 10, b: 30 };
  const nonzero = counts.map((c, i) => (edges[i] <= 0 && edges[i + 1] > 0 ? 0 : c));
  const zeroShare = 1 - group.prevalence, maxc = Math.max(...nonzero, 1);
  const x = (v) => pad.l + ((v - edges[0]) / (edges[edges.length - 1] - edges[0] || 1)) * (W - pad.l - pad.r);
  const svg = svgEl("svg", { width: W, height: H });
  svg.dataset.caption = `nonzero activations (${(zeroShare * 100).toFixed(0)}% are zero)`;
  nonzero.forEach((c, i) => {
    const mid = (edges[i] + edges[i + 1]) / 2, h = (c / maxc) * (H - pad.t - pad.b);
    svg.append(svgEl("rect", { x: x(edges[i]), width: Math.max(x(edges[i + 1]) - x(edges[i]) - 0.5, 0.5), y: H - pad.b - h, height: h, fill: mid > 0 ? "#2f7d32" : "#a33" }));
  });
  svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b, stroke: "#999" }));
  for (const v of [edges[0], 0, edges[edges.length - 1]]) axisText(svg, x(v), H - pad.b + 14, fmt(v, 2));
  axisText(svg, (pad.l + W - pad.r) / 2, H - 4, "activation", "middle", "axis");
  axisText(svg, pad.l - 4, H - pad.b, maxc, "end", "axis"); axisText(svg, pad.l - 4, pad.t + 8, "count", "end", "axis");
  return svg;
}

function spearman(xs, ys) {
  const rank = (a) => { const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]); const r = []; idx.forEach(([, i], k) => { r[i] = k; }); return r; };
  const rx = rank(xs), ry = rank(ys), n = xs.length;
  if (n < 3) return null;
  const mx = (n - 1) / 2, my = (n - 1) / 2;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { num += (rx[i] - mx) * (ry[i] - my); dx += (rx[i] - mx) ** 2; dy += (ry[i] - my) ** 2; }
  return num / Math.sqrt(dx * dy);
}

function panelByModel(group) {
  const rows = group.by_model.filter((m) => m.accuracy != null);
  const W = 270, H = 140, pad = { l: 40, r: 8, t: 10, b: 30 };
  const ys = rows.map((m) => m.mean_activation), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), span = hi - lo || 1;
  const x = (a) => pad.l + a * (W - pad.l - pad.r), y = (v) => H - pad.b - ((v - lo) / span) * (H - pad.t - pad.b);
  const svg = svgEl("svg", { width: W, height: H });
  const rho = spearman(rows.map((m) => m.accuracy), ys);
  svg.dataset.caption = `model accuracy vs mean activation, ρ = ${rho == null ? "–" : rho.toFixed(2)}`;
  svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: y(0), y2: y(0), stroke: "#bbb" }));
  svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b, stroke: "#999" }));
  rows.forEach((m) => {
    const c = svgEl("circle", { cx: x(m.accuracy), cy: y(m.mean_activation), r: 3.5, fill: "#1f4e9c", opacity: 0.8 });
    const title = svgEl("title"); title.textContent = `${m.model}: accuracy ${fmt(m.accuracy)}, mean activation ${fmt(m.mean_activation, 3)}, n=${m.n}`; c.append(title);
    svg.append(c);
  });
  for (const a of [0, 0.5, 1]) axisText(svg, x(a), H - pad.b + 14, a);
  axisText(svg, (pad.l + W - pad.r) / 2, H - 4, "model accuracy (within benchmark)", "middle", "axis");
  axisText(svg, pad.l - 4, y(hi) + 4, fmt(hi, 2), "end", "axis"); axisText(svg, pad.l - 4, y(lo) + 4, fmt(lo, 2), "end", "axis");
  return svg;
}

function panelAgreement(group) {
  // among judged pairs whose transcripts differ on the feature: share where the higher-activation side is picked / is the more accurate model
  const a = group.pair_agreement;
  if (!a) return svgEl("svg", { width: 10, height: 10 });
  const rows = [a.capability, ...a.judges].filter((r) => r.p != null);
  const W = 320, H = 12 + rows.length * 15 + 30, pad = { l: 125, r: 38, t: 10, b: 26 };
  const x = (v) => pad.l + ((v - 0.2) / 0.6) * (W - pad.l - pad.r);
  const svg = svgEl("svg", { width: W, height: H });
  svg.dataset.caption = "P(higher activation picked), pairs differing on the feature";
  for (const v of [0.3, 0.5, 0.7]) {
    svg.append(svgEl("line", { x1: x(v), x2: x(v), y1: pad.t, y2: H - pad.b, stroke: v === 0.5 ? "#999" : "#eee", "stroke-dasharray": v === 0.5 ? "3 3" : "" }));
    axisText(svg, x(v), H - pad.b + 12, v, "middle", "axis");
  }
  rows.forEach((r, i) => {
    const y = pad.t + 10 + i * 15, cap = r.who === "more accurate model";
    axisText(svg, pad.l - 6, y + 4, r.who, "end", cap ? "title" : "");
    if (r.lo != null) svg.append(svgEl("line", { x1: x(Math.max(0.2, r.lo)), x2: x(Math.min(0.8, r.hi)), y1: y, y2: y, stroke: cap ? "#c77d00" : "#1f4e9c", "stroke-width": 1.5 }));
    svg.append(svgEl("circle", { cx: x(Math.min(0.8, Math.max(0.2, r.p))), cy: y, r: 3.5, fill: cap ? "#c77d00" : "#1f4e9c" }));
    axisText(svg, W - pad.r + 4, y + 4, `n=${r.n}`, "start", "axis");
  });
  return svg;
}

function renderSummary(group) {
  if (!group.histogram) { $("#summary").replaceChildren(); return; }
  $("#summary").replaceChildren(...[panelDistribution(group), panelByModel(group), panelAgreement(group)].map((svg) =>
    el("figure", { class: "panel", style: `width:${Number(svg.getAttribute("width")) + 16}px` }, el("figcaption", {}, svg.dataset.caption || ""), svg)));
}

// ---- traces --------------------------------------------------------------------------------
// increments: how much each text segment moves the feature (prefix value after − before),
// smoothed over neighbouring segments so shading marks regions rather than single-token jitter
const SMOOTH = 3;
function increments(trace) {
  const raw = trace.values.map((v, i) => v - (i ? trace.values[i - 1] : 0));
  return raw.map((_, i) => {
    const from = Math.max(0, i - SMOOTH), to = Math.min(raw.length, i + SMOOTH + 1);
    let sum = 0; for (let j = from; j < to; j++) sum += raw[j];
    return sum / (to - from);
  });
}

function shade(v, scale) {
  const a = Math.min(Math.abs(v) / (scale || 1), 1) * 0.85;
  return v >= 0 ? `rgba(46,125,50,${a})` : `rgba(200,60,60,${a})`;
}

function segmentSpans(example, trace, from, to, inc, scale) {
  const out = [];
  for (let i = from; i < to; i++) {
    const [s, e] = trace.spans[i];
    out.push(el("span", { class: "seg", style: `background:${shade(inc[i], scale)}`, title: `Δ ${fmt(inc[i], 3)} · prefix value ${fmt(trace.values[i], 3)}` },
      example.transcript.slice(s, e)));
  }
  return out;
}

function traceChart(trace, inc, scale) {
  // full-width chart of the prefix value along the transcript; the area is shaded by the local contribution
  const W = 1000, H = 90, pad = { l: 36, r: 6, t: 8, b: 16 }, vals = trace.values, n = vals.length;
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = hi - lo || 1;
  const px = (i) => pad.l + (i / Math.max(n - 1, 1)) * (W - pad.l - pad.r), py = (v) => H - pad.b - ((v - lo) / span) * (H - pad.t - pad.b);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "trace-chart", preserveAspectRatio: "none" });
  vals.forEach((v, i) => {
    if (i === 0) return;
    svg.append(svgEl("rect", { x: px(i - 1), width: Math.max(px(i) - px(i - 1), 0.5), y: pad.t, height: H - pad.t - pad.b, fill: shade(inc[i], scale) }));
  });
  svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: py(0), y2: py(0), stroke: "#999" }));
  svg.append(svgEl("polyline", { points: vals.map((v, i) => `${px(i)},${py(v)}`).join(" "), fill: "none", stroke: "#1f4e9c", "stroke-width": 1.6, "vector-effect": "non-scaling-stroke" }));
  axisText(svg, pad.l - 4, py(hi) + 4, fmt(hi, 2), "end", "axis"); axisText(svg, pad.l - 4, py(lo) + 4, fmt(lo, 2), "end", "axis");
  axisText(svg, pad.l, H - 3, "start of transcript", "start", "axis"); axisText(svg, W - pad.r, H - 3, "end", "end", "axis");
  return svg;
}

const SNIPPET_CHARS = 450;  // context either side of a peak
function snippets(example, trace) {
  const inc = increments(trace), scale = Math.max(...inc.map(Math.abs), 1e-6), n = inc.length;
  const charsBefore = (i, chars) => { let j = i; while (j > 0 && trace.spans[i][1] - trace.spans[j][0] < chars) j--; return j; };
  const charsAfter = (i, chars) => { let j = i; while (j < n - 1 && trace.spans[j][1] - trace.spans[i][0] < chars) j++; return j + 1; };
  // peaks: where the prefix value changes most over the preceding window, non-overlapping
  const gain = trace.values.map((v, i) => Math.abs(v - trace.values[charsBefore(i, SNIPPET_CHARS)]));
  const order = gain.map((g, i) => [g, i]).sort((a, b) => b[0] - a[0]);
  const windows = [];
  for (const [, i] of order) {
    if (windows.length >= SNIPPETS) break;
    const f = charsBefore(i, SNIPPET_CHARS), t = charsAfter(i, SNIPPET_CHARS);
    if (windows.some(([wf, wt]) => f < wt && t > wf)) continue;
    windows.push([f, t]);
  }
  windows.sort((a, b) => a[0] - b[0]);
  const pre = el("div", { class: "text" });
  windows.forEach(([f, t], k) => {
    if (k === 0 && f > 0) pre.append(el("span", { class: "gap" }, "… "));
    pre.append(...segmentSpans(example, trace, f, t, inc, scale));
    pre.append(el("span", { class: "gap" }, t < inc.length ? " … " : ""));
  });
  const full = el("div", { class: "text full" }, ...segmentSpans(example, trace, 0, inc.length, inc, scale));
  full.hidden = true;
  const toggle = el("button", { class: "linkish", onclick: () => { full.hidden = !full.hidden; pre.hidden = !pre.hidden; toggle.textContent = full.hidden ? "full transcript" : "show snippets"; } }, "full transcript");
  return [el("div", { class: "trace-row" }, el("span", { class: "trace-legend" }, `value along the transcript; last token ${fmt(trace.final, 3)}, stored ${fmt(example.activation, 3)}`), toggle), traceChart(trace, inc, scale), pre, full];
}

function plainText(example) {
  const head = example.transcript.slice(0, 600), rest = example.transcript.length > 600;
  const pre = el("div", { class: "text" }, head, rest ? el("span", { class: "gap" }, " …") : "");
  if (!rest) return [pre];
  const full = el("div", { class: "text full" }, example.transcript); full.hidden = true;
  const toggle = el("button", { class: "linkish", onclick: () => { full.hidden = !full.hidden; pre.hidden = !pre.hidden; toggle.textContent = full.hidden ? "full transcript" : "show less"; } }, "full transcript");
  return [el("div", { class: "trace-row" }, el("span", { class: "trace-legend" }, "not traced"), toggle), pre, full];
}

// ---- example cards -------------------------------------------------------------------------
function card(example, opts = {}) {
  const cls = example.activation > 0 ? "pos" : example.activation < 0 ? "neg" : "zero";
  const meta = el("div", { class: "meta" },
    opts.hideActivation ? "" : el("span", { class: "act " + cls }, fmt(example.activation, 3)),
    opts.bucketTag ? el("span", { class: "tag" }, opts.bucketTag) : "",
    el("span", {}, el("b", {}, example.model)),
    el("span", {}, `acc ${fmt(example.accuracy)}`),
    el("span", {}, `picked ${fmt(example.pick_rate)} (n=${example.pick_n})`),
    el("span", { class: "dim" }, `${example.bench} · ${example.item} · ep${example.epoch} · ${example.split.replace("representation_", "")}`),
    el("details", { class: "question" }, el("summary", {}, "question"), el("div", { class: "text" }, example.question)),
    opts.guess ? el("label", { class: "guess" }, el("input", { type: "checkbox" }), " fires") : "");
  const trace = state.traces && state.traces[exampleKey(example)];
  const c = el("div", { class: "card" }, meta, ...(trace ? snippets(example, trace) : plainText(example)));
  if (opts.role) c.dataset.role = opts.role;
  return c;
}

// per model: its top, its most negative and a couple of zero-activation transcripts, in accuracy order
function renderByModel(group) {
  const rows = group.examples.filter((e) => e.bucket.startsWith("model "));
  if (!rows.length) return "";
  const order = group.by_model.map((m) => m.model);
  const models = [...new Set(rows.map((e) => e.model))].sort((a, b) => order.indexOf(b) - order.indexOf(a));
  const section = el("section", { class: "bucket" }, el("h3", {}, "By model"));
  for (const model of models) {
    const mine = rows.filter((e) => e.model === model);
    const info = group.by_model.find((m) => m.model === model) || {};
    const body = el("div", { class: "model-body" }); body.hidden = true;
    const head = el("div", { class: "model-head", onclick: () => { body.hidden = !body.hidden; if (!body.hidden && !body.childElementCount) body.append(...mine.map((e) => card(e, { bucketTag: e.bucket.replace("model ", "") }))); } },
      el("b", {}, model), el("span", { class: "dim" }, ` acc ${fmt(info.accuracy)} · mean activation ${fmt(info.mean_activation, 3)} · n=${info.n} · `),
      ...mine.map((e) => el("span", { class: "act " + (e.activation > 0 ? "pos" : e.activation < 0 ? "neg" : "zero") }, ` ${fmt(e.activation, 3)}`)));
    section.append(head, body);
  }
  return section;
}

function renderExamples(group) {
  const box = $("#examples");
  if (!state.blind) {
    box.replaceChildren(...BUCKETS.map(([bucket, title]) => {
      const rows = group.examples.filter((e) => e.bucket === bucket);
      return rows.length ? el("section", { class: "bucket" }, el("h3", {}, `${title} (${rows.length})`), ...rows.map((e) => card(e))) : "";
    }), renderByModel(group));
    return;
  }
  const firing = group.examples.filter((e) => e.bucket === "top").slice(0, 4), contrast = group.examples.filter((e) => e.bucket === "contrast").slice(0, 4);
  const pool = [...firing.map((e) => ({ e, role: "firing" })), ...contrast.map((e) => ({ e, role: "contrast" }))];
  for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  const cards = pool.map(({ e, role }) => card(e, { hideActivation: true, guess: true, role }));
  const reveal = el("button", { class: "reveal", onclick: () => {
    let right = 0;
    cards.forEach((c) => {
      const guessed = c.querySelector(".guess input").checked, fires = c.dataset.role === "firing";
      c.classList.add(fires ? "revealed-firing" : "revealed-contrast");
      c.querySelector(".guess").append(fires ? ": fires" : ": zero");
      if (guessed === fires) right++;
    });
    reveal.replaceWith(el("div", { class: "desc" }, `${right}/${cards.length} correct. Name: ${group.description && group.description.text ? group.description.text : "(none)"}`));
  } }, "reveal");
  box.replaceChildren(el("section", { class: "bucket" }, el("h3", {}, "4 firing, 4 zero, random order"), reveal, ...cards));
}

function renderMain() {
  if (state.feature === "explain") { renderExplanation(); return; }
  if (!state.data) return;
  const group = state.data.groups[state.group];
  renderHeader(group);
  renderSummary(group);
  renderExamples(group);
}

(async () => {
  state.index = await loadJson("data/index.json");
  $("#blind").addEventListener("change", (ev) => { state.blind = ev.target.checked; renderSidebar(); renderMain(); });
  await selectFeature(0);
})();
