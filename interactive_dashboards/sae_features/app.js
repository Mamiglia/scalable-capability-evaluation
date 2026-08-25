// Shared-SAE feature dashboard: data/index.json, data/feature_NN.json and, when traced, data/traces/feature_NN.json.
// Each example shows snippets around the text segments that move the feature most, shaded by their contribution.
const GROUP_LABELS = { gpqa: "GPQA", aime: "AIME", intercode: "InterCode", swe_bench_mini: "SWE-bench mini" };
const BUCKETS = [
  ["top", "Top activations"], ["50-90%", "Quantile 50–90% of positive activations"], ["10-50%", "Quantile 10–50%"],
  ["0-10%", "Quantile 0–10%"], ["contrast", "Same-question contrasts (activation 0)"], ["negative", "Most negative"],
];
const WINDOW = 15;          // segments either side of a peak in a snippet
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
  }));
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
    `${GROUP_LABELS[state.group]} · n = ${group.n} · prevalence ${(group.prevalence * 100).toFixed(1)}%`,
    d ? ` · fidelity r = ${fmt(d.correlation, 3)}, p = ${d.p_value == null ? "–" : d.p_value.toExponential(1)}` : " · not named (too few firing questions)",
    d && d.reason ? ` · ${d.reason.replaceAll("_", " ")}` : "",
    state.traces ? ` · ${Object.keys(state.traces).length} traced` : " · no traces yet");
  if (state.blind) {
    $("#header").replaceChildren(el("h2", {}, `shared:${state.feature} @ ${GROUP_LABELS[state.group]}`),
      el("div", { class: "desc" }, "[label hidden — read the examples, then reveal]"), stats);
    return;
  }
  $("#header").replaceChildren(
    el("h2", {}, `shared:${state.feature} @ ${GROUP_LABELS[state.group]}`,
      d ? el("span", { class: "badge " + (d.kept ? "kept" : "unkept") }, d.kept ? "kept" : "not kept") : ""),
    el("div", { class: "desc" }, d && d.text ? d.text : "(no description)"),
    stats,
    d && d.candidates.length ? el("details", { class: "candidates" }, el("summary", {}, `${d.candidates.length} candidate descriptions (fidelity r)`),
      el("ol", {}, ...d.candidates.map((c) => el("li", {}, `${fmt(c.correlation, 3)} — ${c.description}`)))) : "");
}

// three summary panels: activation distribution, mean activation per model vs its accuracy, judges' pick rate by activation
function axisText(svg, x, y, text, anchor = "middle", cls = "") {
  const t = svgEl("text", { x, y, "text-anchor": anchor, class: cls }); t.textContent = text; svg.append(t); return t;
}

function panelDistribution(group) {
  const { counts, edges } = group.histogram;
  const W = 300, H = 170, pad = { l: 34, r: 8, t: 22, b: 30 };
  const nonzero = counts.map((c, i) => (edges[i] <= 0 && edges[i + 1] > 0 ? 0 : c));
  const zeroShare = 1 - group.prevalence, maxc = Math.max(...nonzero, 1);
  const x = (v) => pad.l + ((v - edges[0]) / (edges[edges.length - 1] - edges[0] || 1)) * (W - pad.l - pad.r);
  const svg = svgEl("svg", { width: W, height: H, class: "panel" });
  axisText(svg, pad.l, 12, `activation, ${group.n} transcripts (${(zeroShare * 100).toFixed(0)}% zero, not drawn)`, "start", "title");
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
  const W = 360, H = 170, pad = { l: 40, r: 8, t: 22, b: 30 };
  const ys = rows.map((m) => m.mean_activation), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), span = hi - lo || 1;
  const x = (a) => pad.l + a * (W - pad.l - pad.r), y = (v) => H - pad.b - ((v - lo) / span) * (H - pad.t - pad.b);
  const svg = svgEl("svg", { width: W, height: H, class: "panel" });
  const rho = spearman(rows.map((m) => m.accuracy), ys);
  axisText(svg, pad.l, 12, `mean activation per model vs its accuracy · Spearman ρ = ${rho == null ? "–" : rho.toFixed(2)}`, "start", "title");
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

function panelPickRate(group) {
  const rows = group.pick_by_bin.filter((b) => b.mean_pick != null);
  const W = 340, H = 170, pad = { l: 40, r: 8, t: 22, b: 44 };
  const x = (i) => pad.l + (i + 0.5) * ((W - pad.l - pad.r) / rows.length), y = (v) => H - pad.b - v * (H - pad.t - pad.b);
  const svg = svgEl("svg", { width: W, height: H, class: "panel" });
  axisText(svg, pad.l, 12, "judges' mean pick rate by activation bin", "start", "title");
  svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: y(0.5), y2: y(0.5), stroke: "#bbb", "stroke-dasharray": "3 3" }));
  svg.append(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b, stroke: "#999" }));
  rows.forEach((b, i) => {
    const bw = (W - pad.l - pad.r) / rows.length * 0.7;
    svg.append(svgEl("rect", { x: x(i) - bw / 2, width: bw, y: y(b.mean_pick), height: y(0) - y(b.mean_pick), fill: b.bin === "negative" ? "#a33" : b.bin === "zero" ? "#999" : "#2f7d32" }));
    axisText(svg, x(i), H - pad.b + 12, b.bin.replace("positive ", "+"), "middle", "axis");
    axisText(svg, x(i), H - pad.b + 24, `n=${b.n}`, "middle", "axis");
  });
  for (const v of [0, 0.5, 1]) axisText(svg, pad.l - 4, y(v) + 4, v, "end", "axis");
  axisText(svg, (pad.l + W - pad.r) / 2, H - 4, "activation bin (positive bins = quintiles)", "middle", "axis");
  return svg;
}

function renderSummary(group) {
  if (!group.histogram) { $("#summary").replaceChildren(); return; }
  $("#summary").replaceChildren(panelDistribution(group), panelByModel(group), panelPickRate(group));
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

function snippets(example, trace) {
  const inc = increments(trace), scale = Math.max(...inc.map(Math.abs), 1e-6);
  // peaks: centres of the windows where the prefix value changes most over the window, non-overlapping
  const gain = trace.values.map((v, i) => Math.abs(v - trace.values[Math.max(0, i - WINDOW)]));
  const order = gain.map((g, i) => [g, i]).sort((a, b) => b[0] - a[0]);
  const windows = [];
  for (const [, i] of order) {
    if (windows.length >= SNIPPETS) break;
    if (windows.some(([f, t]) => i >= f - WINDOW && i < t + WINDOW)) continue;
    windows.push([Math.max(0, i - WINDOW), Math.min(inc.length, i + WINDOW + 1)]);
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
  const toggle = el("button", { class: "linkish", onclick: () => { full.hidden = !full.hidden; pre.hidden = !pre.hidden; toggle.textContent = full.hidden ? "show full transcript" : "show snippets"; } }, "show full transcript");
  return [el("div", { class: "trace-row" }, el("span", { class: "trace-legend" }, `feature value as the transcript is read left to right (final ${fmt(trace.final, 3)}, stored ${fmt(example.activation, 3)}); green/red = text that raises/lowers it`), toggle), traceChart(trace, inc, scale), pre, full];
}

function plainText(example) {
  const head = example.transcript.slice(0, 600), rest = example.transcript.length > 600;
  const pre = el("div", { class: "text" }, head, rest ? el("span", { class: "gap" }, " …") : "");
  if (!rest) return [pre];
  const full = el("div", { class: "text full" }, example.transcript); full.hidden = true;
  const toggle = el("button", { class: "linkish", onclick: () => { full.hidden = !full.hidden; pre.hidden = !pre.hidden; toggle.textContent = full.hidden ? "show full transcript" : "show less"; } }, "show full transcript");
  return [el("div", { class: "trace-row" }, el("span", { class: "trace-legend" }, "not traced yet"), toggle), pre, full];
}

// ---- example cards -------------------------------------------------------------------------
function card(example, opts = {}) {
  const cls = example.activation > 0 ? "pos" : example.activation < 0 ? "neg" : "zero";
  const meta = el("div", { class: "meta" },
    opts.hideActivation ? "" : el("span", { class: "act " + cls }, fmt(example.activation, 3)),
    opts.bucketTag ? el("span", { class: "tag" }, opts.bucketTag) : "",
    el("span", {}, el("b", {}, example.model)),
    el("span", {}, `model acc ${fmt(example.accuracy)}`),
    el("span", {}, `judges pick ${fmt(example.pick_rate)} (n=${example.pick_n})`),
    el("span", { class: "dim" }, `${example.bench} · ${example.item} · ep${example.epoch} · ${example.split.replace("representation_", "")}`),
    el("details", { class: "question" }, el("summary", {}, "question"), el("div", { class: "text" }, example.question)),
    opts.guess ? el("label", { class: "guess" }, el("input", { type: "checkbox" }), " my guess: fires") : "");
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
  const section = el("section", { class: "bucket" }, el("h3", {}, `By model (${models.length}; top, most negative, zero-activation) — click a model to expand`));
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
      c.querySelector(".guess").append(fires ? " → does fire" : " → does not fire (same-question contrast)");
      if (guessed === fires) right++;
    });
    reveal.replaceWith(el("div", { class: "desc" }, `${right}/${cards.length} correct. Label: ${group.description && group.description.text ? group.description.text : "(none)"}`));
  } }, "reveal");
  box.replaceChildren(el("section", { class: "bucket" }, el("h3", {}, "4 firing + 4 same-question contrasts, shuffled — tick the ones you think fire"), reveal, ...cards));
}

function renderMain() {
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
