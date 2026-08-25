// Shared-SAE feature dashboard: data/index.json, data/feature_NN.json and, when traced, data/traces/feature_NN.json.
// Each example shows snippets around the text segments that move the feature most, shaded by their contribution.
const GROUP_LABELS = { gpqa: "GPQA", aime: "AIME", intercode: "InterCode", swe_bench_mini: "SWE-bench mini" };
const BUCKETS = [
  ["top", "Top activations"], ["50-90%", "Quantile 50–90% of positive activations"], ["10-50%", "Quantile 10–50%"],
  ["0-10%", "Quantile 0–10%"], ["contrast", "Same-question contrasts (activation 0)"], ["negative", "Most negative"],
];
const WINDOW = 12;          // segments either side of a peak in a snippet
const SNIPPETS = 2;         // peaks shown per example
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
      el("span", { class: "prev" }, `${(g.prevalence * 100).toFixed(0)}%`));
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

function renderHistogram(group) {
  if (!group.histogram) { $("#histogram").replaceChildren(); return; }
  const { counts, edges } = group.histogram;
  const W = 520, H = 110, pad = 28, maxc = Math.max(...counts, 1);
  const x = (v) => pad + ((v - edges[0]) / (edges[edges.length - 1] - edges[0] || 1)) * (W - 2 * pad);
  const svg = svgEl("svg", { width: W, height: H });
  counts.forEach((c, i) => {
    const mid = (edges[i] + edges[i + 1]) / 2, h = Math.log1p(c) / Math.log1p(maxc) * (H - 2 * pad);
    svg.append(svgEl("rect", { x: x(edges[i]), width: Math.max(x(edges[i + 1]) - x(edges[i]) - 0.5, 0.5), y: H - pad - h, height: h,
      fill: mid > 0 ? "#2f7d32" : mid < 0 ? "#a33" : "#999" }));
  });
  for (const v of [edges[0], 0, edges[edges.length - 1]]) {
    const t = svgEl("text", { x: x(v), y: H - 8, "text-anchor": "middle" }); t.textContent = fmt(v, 2); svg.append(t);
  }
  const label = svgEl("text", { x: pad, y: 12 }); label.textContent = "activation density (log count; zero bin included)"; svg.append(label);
  $("#histogram").replaceChildren(svg);
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

function sparkline(trace) {
  const W = 220, H = 36, vals = trace.values, n = vals.length;
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = hi - lo || 1;
  const px = (i) => (i / Math.max(n - 1, 1)) * W, py = (v) => H - ((v - lo) / span) * (H - 4) - 2;
  const svg = svgEl("svg", { width: W, height: H, class: "spark" });
  svg.append(svgEl("line", { x1: 0, x2: W, y1: py(0), y2: py(0), stroke: "#bbb", "stroke-width": 1 }));
  svg.append(svgEl("polyline", { points: vals.map((v, i) => `${px(i)},${py(v)}`).join(" "), fill: "none", stroke: "#1f4e9c", "stroke-width": 1.2 }));
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
  return [el("div", { class: "trace-row" }, sparkline(trace), el("span", { class: "trace-legend" }, `prefix value along the transcript · final ${fmt(trace.final, 3)} (stored ${fmt(example.activation, 3)})`), toggle), pre, full];
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
    el("span", {}, el("b", {}, example.model)),
    el("span", {}, `model acc ${fmt(example.accuracy)}`),
    el("span", {}, `judges pick ${fmt(example.pick_rate)} (n=${example.pick_n})`),
    el("span", { class: "dim" }, `${example.bench} · ${example.item} · ep${example.epoch} · ${example.split.replace("representation_", "")}`),
    el("details", { class: "question" }, el("summary", {}, "question"), el("div", { class: "text" }, example.question)),
    opts.guess ? el("label", { class: "guess" }, el("input", { type: "checkbox" }), " fires?") : "");
  const trace = state.traces && state.traces[exampleKey(example)];
  const c = el("div", { class: "card" }, meta, ...(trace ? snippets(example, trace) : plainText(example)));
  if (opts.role) c.dataset.role = opts.role;
  return c;
}

function renderExamples(group) {
  const box = $("#examples");
  if (!state.blind) {
    box.replaceChildren(...BUCKETS.map(([bucket, title]) => {
      const rows = group.examples.filter((e) => e.bucket === bucket);
      return rows.length ? el("section", { class: "bucket" }, el("h3", {}, `${title} (${rows.length})`), ...rows.map((e) => card(e))) : "";
    }));
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
      c.querySelector(".guess").append(fires ? " → fires" : " → contrast");
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
  renderHistogram(group);
  renderExamples(group);
}

(async () => {
  state.index = await loadJson("data/index.json");
  $("#blind").addEventListener("change", (ev) => { state.blind = ev.target.checked; renderSidebar(); renderMain(); });
  await selectFeature(0);
})();
