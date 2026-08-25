// Shared-SAE feature viewer: reads data/index.json, data/feature_NN.json and optional data/traces/feature_NN.json.
const GROUP_LABELS = { gpqa: "GPQA", aime: "AIME", intercode: "InterCode", swe_bench_mini: "SWE-bench mini" };
const state = { index: null, feature: 0, group: "gpqa", data: null, traces: null, blind: false };

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
};
const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const exampleKey = (e) => [e.bench, e.item, e.model, e.epoch].join("|");

async function loadJson(path) {
  const r = await fetch(path);
  if (!r.ok) return null;
  return r.json();
}

function renderSidebar() {
  const tabs = $("#group-tabs");
  tabs.replaceChildren(...state.index.groups.map((g) =>
    el("button", { class: g === state.group ? "active" : "", onclick: () => { state.group = g; renderSidebar(); renderMain(); } }, GROUP_LABELS[g] || g)));
  const list = $("#feature-list");
  list.replaceChildren(...state.index.features.map((f) => {
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
}

function renderHeader(group) {
  const h = $("#header");
  const d = group.description;
  const stats = el("div", { class: "stats" },
    `${GROUP_LABELS[state.group]} · n = ${group.n} · prevalence ${(group.prevalence * 100).toFixed(1)}%`,
    d ? ` · fidelity r = ${fmt(d.correlation, 3)}, p = ${d.p_value == null ? "–" : d.p_value.toExponential(1)}` : " · not named (too few firing questions)",
    d && d.reason ? ` · ${d.reason.replaceAll("_", " ")}` : "");
  if (state.blind) {
    h.replaceChildren(el("h2", {}, `shared:${state.feature} @ ${GROUP_LABELS[state.group]}`), el("div", { class: "desc" }, "[label hidden — read the examples, then reveal]"), stats);
    return;
  }
  h.replaceChildren(
    el("h2", {}, `shared:${state.feature} @ ${GROUP_LABELS[state.group]}`,
      d ? el("span", { class: "badge " + (d.kept ? "kept" : "unkept") }, d.kept ? "kept" : "not kept") : ""),
    el("div", { class: "desc" }, d && d.text ? d.text : "(no description)"),
    stats,
    d && d.candidates.length ? el("details", { class: "candidates" }, el("summary", {}, `${d.candidates.length} candidate descriptions (fidelity r)`),
      el("ol", {}, ...d.candidates.map((c) => el("li", {}, `${fmt(c.correlation, 3)} — ${c.description}`)))) : "");
}

function renderHistogram(group) {
  const box = $("#histogram");
  if (!group.histogram) { box.replaceChildren(); return; }
  const { counts, edges } = group.histogram;
  const W = 520, H = 110, pad = 28, maxc = Math.max(...counts, 1);
  const x = (v) => pad + ((v - edges[0]) / (edges[edges.length - 1] - edges[0] || 1)) * (W - 2 * pad);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  counts.forEach((c, i) => {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    const mid = (edges[i] + edges[i + 1]) / 2;
    r.setAttribute("x", x(edges[i])); r.setAttribute("width", Math.max(x(edges[i + 1]) - x(edges[i]) - 0.5, 0.5));
    const h = Math.log1p(c) / Math.log1p(maxc) * (H - 2 * pad);
    r.setAttribute("y", H - pad - h); r.setAttribute("height", h);
    r.setAttribute("fill", mid > 0 ? "#2f7d32" : mid < 0 ? "#a33" : "#999");
    svg.append(r);
  });
  for (const v of [edges[0], 0, edges[edges.length - 1]]) {
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", x(v)); t.setAttribute("y", H - 8); t.setAttribute("text-anchor", "middle"); t.textContent = fmt(v, 2);
    svg.append(t);
  }
  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  label.setAttribute("x", pad); label.setAttribute("y", 12); label.textContent = "activation histogram (log count; zero bin included)";
  svg.append(label);
  box.replaceChildren(svg);
}

function traceColour(v, scale) {
  const a = Math.min(Math.abs(v) / (scale || 1), 1) * 0.75;
  return v >= 0 ? `rgba(46,125,50,${a})` : `rgba(170,51,51,${a})`;
}

function transcriptNode(example) {
  const trace = state.traces && state.traces[exampleKey(example)];
  const pre = el("pre", {});
  if (!trace) { pre.textContent = example.transcript; return [pre]; }
  const scale = Math.max(...trace.values.map(Math.abs), 1e-6);
  let cursor = 0;
  trace.offsets.forEach(([s, e], i) => {
    if (s > cursor) pre.append(example.transcript.slice(cursor, s));
    pre.append(el("span", { class: "t", style: `background:${traceColour(trace.values[i], scale)}`, title: fmt(trace.values[i], 3) }, example.transcript.slice(s, e)));
    cursor = e;
  });
  if (cursor < example.transcript.length) pre.append(example.transcript.slice(cursor));
  return [pre, el("div", { class: "trace-legend" }, `per-token pre-activation of the prefix embedding; final value ${fmt(trace.final, 3)} vs stored ${fmt(example.activation, 3)}`)];
}

function card(example, opts = {}) {
  const cls = example.activation > 0 ? "pos" : example.activation < 0 ? "neg" : "zero";
  const meta = el("div", { class: "meta" },
    opts.hideActivation ? "" : el("span", { class: "act " + cls }, fmt(example.activation, 3)),
    el("span", {}, el("b", {}, example.model)),
    el("span", {}, `acc ${fmt(example.accuracy)}`),
    el("span", {}, `judges pick ${fmt(example.pick_rate)} (n=${example.pick_n})`),
    el("span", {}, `${example.bench} · ${example.item} · ep${example.epoch}`),
    el("span", {}, example.split.replace("representation_", "")),
    opts.guess ? el("span", { class: "guess" }, el("label", {}, el("input", { type: "checkbox" }), " firing?")) : "");
  const body = el("div", { class: "body" },
    el("details", { class: "question" }, el("summary", {}, "question"), el("pre", {}, example.question)),
    ...transcriptNode(example));
  const c = el("div", { class: "card" }, meta, body);
  meta.addEventListener("click", (ev) => { if (ev.target.tagName !== "INPUT") c.classList.toggle("open"); });
  if (opts.role) c.dataset.role = opts.role;
  return c;
}

function renderExamples(group) {
  const box = $("#examples");
  box.className = state.blind ? "blind" : "";
  if (!state.blind) {
    box.replaceChildren(
      el("div", { class: "column" }, el("h3", {}, `top firing (${group.firing.length})`), ...group.firing.map((e) => card(e))),
      el("div", { class: "column" }, el("h3", {}, `same-question contrasts, activation 0 (${group.contrast.length})`), ...group.contrast.map((e) => card(e))),
      el("div", { class: "column" }, el("h3", {}, `most negative (${group.negative.length})`), ...group.negative.map((e) => card(e))));
    return;
  }
  const pool = [...group.firing.slice(0, 4).map((e) => ({ e, role: "firing" })), ...group.contrast.slice(0, 4).map((e) => ({ e, role: "contrast" }))];
  for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  const cards = pool.map(({ e, role }) => card(e, { hideActivation: true, guess: true, role }));
  const reveal = el("button", { class: "reveal", onclick: () => {
    let right = 0;
    cards.forEach((c) => {
      const guessed = c.querySelector("input").checked, firing = c.dataset.role === "firing";
      c.classList.add(firing ? "revealed-firing" : "revealed-contrast");
      c.querySelector(".guess").append(firing ? " → firing" : " → contrast");
      if (guessed === firing) right++;
    });
    reveal.replaceWith(el("div", { class: "desc" }, `${right}/${cards.length} correct. Label: ${group.description && group.description.text ? group.description.text : "(none)"}`));
  } }, "reveal");
  box.replaceChildren(el("div", { class: "column" }, el("h3", {}, "4 firing + 4 same-question contrasts, shuffled — tick the ones you think fire"), reveal, ...cards));
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
