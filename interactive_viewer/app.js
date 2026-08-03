/* Judge-pilot gap-curve viewer.
   Plot: judge accuracy vs capability gap, one point per run. Click -> the
   transcripts the judge saw, both counterbalanced orders, every judge's verdict.
   Vanilla JS + vendored Plotly. No build step, no CDN. */

"use strict";

/* --- ported truncation (must byte-match src/judge/dataset.py) ------------ */
const stripAnswer = t => { const i = t.lastIndexOf("ANSWER:");
  return i !== -1 ? t.slice(0, i).replace(/\s+$/, "") : t; };
const truncatePair = (a, b, f) => { a = stripAnswer(a); b = stripAnswer(b);
  const cut = Math.max(1, Math.floor(f * Math.min(a.length, b.length)));
  return [a.slice(0, cut), b.slice(0, cut)]; };
/* the model's own final GPQA answer letter (all judged items are both-correct) */
const answerLetter = t => { const i = t.lastIndexOf("ANSWER:");
  if (i === -1) return null;
  const m = /ANSWER:\s*\**\s*([A-D])/i.exec(t.slice(i, i + 48));
  return m ? m[1].toUpperCase() : null; };

/* --- house palette fallback (plots.py PALETTE), used only if colors{} miss  */
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948", "#8c6d31", "#0b7285"];

/* --- DOM refs ------------------------------------------------------------ */
const $ = id => document.getElementById(id);
const plotDiv = $("plot");
const emptyNote = $("empty-note");
const hintEl = $("ctl-hint");
const ctlTrunc = $("ctl-trunc"), ctlSubset = $("ctl-subset"), ctlArm = $("ctl-arm"),
      ctlColorby = $("ctl-colorby"), ctlCurves = $("ctl-curves");
const overlay = $("modal");

/* --- utils --------------------------------------------------------------- */
const num = v => (v === null || v === undefined || v === "") ? NaN : Number(v);
const fmt = (v, d) => { const n = num(v); return isNaN(n) ? "–" : n.toFixed(d === undefined ? 2 : d); };
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const naturalCompare = (a, b) =>
  String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });

function parseReleaseYear(rel) {
  if (!rel) return NaN;
  const p = String(rel).split("-").map(Number);
  const y = p[0], mo = p.length > 1 ? p[1] : 1, d = p.length > 2 ? p[2] : 1;
  if (isNaN(y)) return NaN;
  const doy = (Date.UTC(y, mo - 1, d) - Date.UTC(y, 0, 1)) / 86400000;
  return y + doy / 366;
}

/* --- global state -------------------------------------------------------- */
let INDEX = null;
let XMAX = 0.4;
let currentRunMap = [];   // parallel to plot traces: array-of-runs | null
let plotInited = false;

const pairCache = new Map();   // "maj__min__frac" -> Promise<pairData>
const transCache = new Map();  // stem -> Promise<{item:text}>

let MODAL = null;  // { run, swapped, item, filter, search, pair, tMajor, tMinor }

/* --- theme (Plotly needs explicit colours; read them from CSS) ----------- */
function themeColors() {
  const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return dark
    ? { surface: "#14140f", panel: "#1d1d17", ink: "#f2f0e9", ink2: "#dcdad0",
        muted: "#9c9a90", grid: "#33322f", border: "#35342d" }
    : { surface: "#ffffff", panel: "#f6f4ee", ink: "#0b0b0b", ink2: "#33322f",
        muted: "#4a4844", grid: "#d3d2c9", border: "#d9d7ce" };
}

/* --- metadata helpers ---------------------------------------------------- */
function modelMeta(m) {
  const x = (INDEX && INDEX.models && INDEX.models[m]) ? INDEX.models[m] : {};
  return { label: x.label || m, acc: x.accuracy, rel: x.release || "" };
}
const judgeLabel = j => modelMeta(j).label;
const relYM = rel => (rel ? String(rel).slice(0, 7) : "–");
const metaStr = m => "(" + fmt(modelMeta(m).acc) + ", " + relYM(modelMeta(m).rel) + ")";

/* --- in-modal comparison selectors (change judge / models, no plot trip) -- */
const uniq = arr => Array.from(new Set(arr));
const byAccDesc = arr => arr.slice().sort((a, b) => num(modelMeta(b).acc) - num(modelMeta(a).acc));
const majorsList = () => byAccDesc(uniq(INDEX.runs.map(r => r.major)));
const minorsFor = maj => byAccDesc(uniq(INDEX.runs.filter(r => r.major === maj).map(r => r.minor)));
const judgesFor = (maj, min) => (INDEX.judges || []).filter(j =>
  INDEX.runs.some(r => r.major === maj && r.minor === min && r.judge === j));
const fracsFor = (j, maj, min) => FRAC_ORDER.filter(f =>
  INDEX.runs.some(r => r.judge === j && r.major === maj && r.minor === min && r.frac === f));
const findRun = (j, maj, min, f) => INDEX.runs.find(r =>
  r.judge === j && r.major === maj && r.minor === min && r.frac === f);
/* snap a desired (major,minor,judge,frac) to the nearest run that exists */
function resolveRun(maj, min, j, f) {
  if (!minorsFor(maj).includes(min)) min = minorsFor(maj)[0];
  const js = judgesFor(maj, min); if (!js.includes(j)) j = js[0];
  const fs = fracsFor(j, maj, min); if (!fs.includes(f)) f = fs.includes("full") ? "full" : fs[0];
  return findRun(j, maj, min, f);
}
function fillSelect(sel, values, current, labelFn) {
  sel.innerHTML = "";
  values.forEach(v => {
    const o = document.createElement("option");
    o.value = v; o.textContent = labelFn(v);
    if (v === current) o.selected = true;
    sel.appendChild(o);
  });
}
function goTo(run) { if (run) openModal(run, true); }   // keep item across the switch
function pairsList() {   // every (major,minor) pair, gap-sorted
  const seen = new Map();
  INDEX.runs.forEach(r => { const k = r.major + "|" + r.minor; if (!seen.has(k)) seen.set(k, [r.major, r.minor]); });
  const gap = p => num(modelMeta(p[0]).acc) - num(modelMeta(p[1]).acc);
  return Array.from(seen.values()).sort((a, b) => gap(a) - gap(b));
}
function setupSelectors(run) {
  const maj = run.major, min = run.minor, j = run.judge, f = run.frac;
  // one dropdown over ALL pairs — pick any comparison without the judge constraining it
  const pairs = pairsList(), sp = $("sel-pair");
  sp.innerHTML = "";
  pairs.forEach((p, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = modelMeta(p[0]).label + " vs " + (p[0] === p[1] ? "itself" : modelMeta(p[1]).label);
    if (p[0] === maj && p[1] === min) o.selected = true;
    sp.appendChild(o);
  });
  sp.onchange = e => { const p = pairs[+e.target.value]; goTo(resolveRun(p[0], p[1], j, f)); };
  fillSelect($("sel-judge"), judgesFor(maj, min), j, judgeLabel);
  $("sel-judge").onchange = e => goTo(resolveRun(maj, min, e.target.value, f));
  $("meta-judge").textContent = metaStr(j);
  const fs = fracsFor(j, maj, min), tw = $("modal-trunc-wrap");
  if (fs.length <= 1) { tw.hidden = true; } else { tw.hidden = false; fillSelect($("modal-trunc"), fs, f, fracLabel); }
  $("modal-trunc").onchange = e => setModalFrac(e.target.value);
}
function judgeColor(j) {
  if (INDEX.colors && INDEX.colors[j]) return INDEX.colors[j];
  const i = INDEX.judges ? INDEX.judges.indexOf(j) : -1;
  return PALETTE[(i >= 0 ? i : 0) % PALETTE.length];
}
function judgeSortValue(j) {
  const m = modelMeta(j);
  return ctlColorby.value === "release" ? parseReleaseYear(m.rel) : num(m.acc);
}
function cmpJudge(a, b) {
  const va = judgeSortValue(a), vb = judgeSortValue(b);
  const na = isNaN(va), nb = isNaN(vb);
  if (na && nb) return naturalCompare(a, b);
  if (na) return 1;
  if (nb) return -1;
  if (va !== vb) return va - vb;
  return naturalCompare(a, b);
}

/* --- filtering ----------------------------------------------------------- */
function filteredRuns() {
  const frac = ctlTrunc.value, subset = ctlSubset.value, arm = ctlArm.value;
  return INDEX.runs.filter(r => {
    if (r.frac !== frac) return false;
    if (subset === "weak" && !r.weak_judge) return false;
    if (subset === "old" && !r.old_judge) return false;
    if (arm === "armA" && !r.arm_a) return false;
    return true;
  });
}

/* --- hover --------------------------------------------------------------- */
function hoverData(r) {
  const j = modelMeta(r.judge), M = modelMeta(r.major), N = modelMeta(r.minor);
  return [j.label, num(j.acc), j.rel, M.label, num(M.acc), M.rel,
          N.label, num(N.acc), N.rel, r.d_acc, r.score, r.score_lo, r.score_hi,
          r.n_items, r.slot_a];
}
const MAIN_HOVER =
  "<b>%{customdata[0]}</b> — judge · GPQA %{customdata[1]:.2f} · %{customdata[2]}<br>" +
  "<b>major:</b> %{customdata[3]} (GPQA %{customdata[4]:.2f} · %{customdata[5]})<br>" +
  "<b>minor:</b> %{customdata[6]} (GPQA %{customdata[7]:.2f} · %{customdata[8]})<br>" +
  "<b>gap:</b> %{customdata[9]:.3f} · <b>judge acc:</b> %{customdata[10]:.2f} " +
  "[%{customdata[11]:.2f}, %{customdata[12]:.2f}]<br>" +
  "<b>n:</b> %{customdata[13]} · <b>slot-A rate:</b> %{customdata[14]:.2f}<extra></extra>";
const SELF_HOVER =
  "<b>%{customdata[0]}</b> — judge · GPQA %{customdata[1]:.2f} · %{customdata[2]}<br>" +
  "<b>self-pair control:</b> %{customdata[3]} vs itself (should sit at chance)<br>" +
  "<b>judge acc:</b> %{customdata[10]:.2f} [%{customdata[11]:.2f}, %{customdata[12]:.2f}]<br>" +
  "<b>n:</b> %{customdata[13]} · <b>slot-A rate:</b> %{customdata[14]:.2f}<extra></extra>";

/* --- curve traces -------------------------------------------------------- */
function lineTrace(j, xs, ys, color) {
  return { type: "scatter", mode: "lines", legendgroup: j, showlegend: false,
           x: xs, y: ys, line: { color: color, width: 1.6 }, hoverinfo: "skip",
           xaxis: "x", yaxis: "y" };
}
function buildCurve(j, crossPts, color) {
  const mode = ctlCurves.value;
  if (mode === "off") return null;
  if (mode === "sigmoid") {
    if (ctlTrunc.value !== "full") return null;               // fits are full-run only
    const f = INDEX.fits && INDEX.fits[j];
    if (!f) return null;
    const xs = [], ys = [], N = 60;
    for (let i = 0; i <= N; i++) {
      const x = XMAX * i / N;
      xs.push(x); ys.push(1 / (1 + Math.exp(-(f.a + f.d * x))));
    }
    return lineTrace(j, xs, ys, color);
  }
  const pts = crossPts.slice().sort((a, b) => a.d_acc - b.d_acc);
  if (pts.length < 2) return null;
  return lineTrace(j, pts.map(r => r.d_acc), pts.map(r => r.score), color);
}

/* --- plot ---------------------------------------------------------------- */
function baseLayout(theme) {
  return {
    margin: { l: 20, r: 176, t: 12, b: 66 },
    paper_bgcolor: theme.surface, plot_bgcolor: theme.surface,
    font: { color: theme.ink2, size: 13 },
    hovermode: "closest",
    hoverlabel: { bgcolor: theme.panel, bordercolor: theme.border,
                  font: { color: theme.ink, size: 12 }, align: "left" },
    // Layout left→right: y-axis (ticks + label) · self strip · gap · main plot.
    // The ticked y-axis anchors to the self strip so it sits OUTSIDE (left of) it;
    // the main plot's y (y2) shares the scale and carries the gridlines.
    yaxis: { domain: [0, 1], range: [0, 1], anchor: "x2",
             title: { text: "judge accuracy" },
             showgrid: false, zeroline: false, linecolor: theme.muted,
             ticks: "outside", tickcolor: theme.muted },
    xaxis2: { domain: [0.12, 0.16], range: [-0.85, 0.85], anchor: "y",
              tickvals: [0], ticktext: ["self"], tickfont: { color: theme.muted },
              showgrid: false, zeroline: false, linecolor: theme.muted, ticks: "" },
    yaxis2: { domain: [0, 1], range: [0, 1], matches: "y", anchor: "x",
              showticklabels: false, gridcolor: theme.grid, showgrid: true, zeroline: false },
    xaxis: { domain: [0.24, 1], range: [0, XMAX], anchor: "y2",
             title: { text: "capability gap (GPQA accuracy difference)" },
             gridcolor: theme.grid, zeroline: false, linecolor: theme.muted,
             ticks: "outside", tickcolor: theme.muted },
    legend: { x: 1.02, y: 0.5, xanchor: "left", yanchor: "middle",
              title: { text: "judge — ordered by " +
                (ctlColorby.value === "release" ? "release" : "capability"),
                font: { color: theme.ink2, size: 12 } },
              font: { color: theme.ink2, size: 12 },
              groupclick: "togglegroup", bgcolor: "rgba(0,0,0,0)" },
    shapes: [
      { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: 0.5, y1: 0.5,
        line: { color: theme.muted, width: 1, dash: "dash" }, layer: "below" },
      { type: "line", xref: "paper", x0: 0.20, x1: 0.20, yref: "paper", y0: 0, y1: 1,
        line: { color: theme.border, width: 1 }, layer: "below" },
    ],
  };
}

function renderPlot() {
  if (!INDEX) return;
  const theme = themeColors();
  const runs = filteredRuns();

  let hint = runs.length + (runs.length === 1 ? " run shown" : " runs shown");
  if (ctlCurves.value === "sigmoid" && ctlTrunc.value !== "full")
    hint += " · sigmoid fits shown for Full only";
  hintEl.textContent = hint;

  if (!runs.length) {
    emptyNote.hidden = false;
    currentRunMap = [];
    Plotly.react(plotDiv, [], baseLayout(theme), { displayModeBar: false, responsive: true });
    if (!plotInited) { plotDiv.on("plotly_click", onPlotClick); plotInited = true; }
    return;
  }
  emptyNote.hidden = true;

  const self = runs.filter(r => r.is_self);
  const cross = runs.filter(r => !r.is_self);
  const present = Array.from(new Set(runs.map(r => r.judge)));
  const legendOrder = present.slice().sort(cmpJudge).reverse();  // strongest/newest on top

  // self-pair x positions: spread all self runs across the narrow strip, grouped by judge
  const selfByJudge = new Map();
  self.forEach(r => { if (!selfByJudge.has(r.judge)) selfByJudge.set(r.judge, []); selfByJudge.get(r.judge).push(r); });
  const flatSelf = [];
  legendOrder.forEach(j => (selfByJudge.get(j) || []).forEach(r => flatSelf.push(r)));
  const nSelf = flatSelf.length;
  const selfX = new Map();
  flatSelf.forEach((r, i) => selfX.set(r.run_id, nSelf > 1 ? (i - (nSelf - 1) / 2) / nSelf : 0));

  const traces = [], runMap = [];
  legendOrder.forEach(j => {
    const color = judgeColor(j), label = judgeLabel(j);
    const cj = cross.filter(r => r.judge === j);
    traces.push({
      type: "scatter", mode: "markers", name: label, legendgroup: j, showlegend: true,
      x: cj.map(r => r.d_acc), y: cj.map(r => r.score),
      customdata: cj.map(hoverData), hovertemplate: MAIN_HOVER,
      marker: { color: color, size: 9, symbol: "circle",
                line: { color: theme.surface, width: 0.8 } },
      xaxis: "x", yaxis: "y",
    });
    runMap.push(cj);

    const sj = selfByJudge.get(j) || [];
    if (sj.length) {
      traces.push({
        type: "scatter", mode: "markers", name: label, legendgroup: j, showlegend: false,
        x: sj.map(r => selfX.get(r.run_id)), y: sj.map(r => r.score),
        customdata: sj.map(hoverData), hovertemplate: SELF_HOVER,
        marker: { color: theme.surface, size: 9, symbol: "circle",
                  line: { color: color, width: 1.8 } },
        xaxis: "x2", yaxis: "y2",
      });
      runMap.push(sj);
    }

    const line = buildCurve(j, cj, color);
    if (line) { traces.push(line); runMap.push(null); }
  });

  currentRunMap = runMap;
  Plotly.react(plotDiv, traces, baseLayout(theme), { displayModeBar: false, responsive: true })
    .then(attachLegendHover);
  if (!plotInited) { plotDiv.on("plotly_click", onPlotClick); plotInited = true; }
}

function onPlotClick(ev) {
  if (!ev.points || !ev.points.length) return;
  const pt = ev.points[0];
  const list = currentRunMap[pt.curveNumber];
  if (!list) return;
  const run = list[pt.pointNumber];
  if (run) openModal(run);
}

/* --- legend hover: spotlight a judge, fade the rest (Epoch-style) --------- */
const DIM_OPACITY = 0.1;
function emphasizeJudge(j) {   // j === null clears
  const data = plotDiv.data || [];
  if (!data.length) return;
  Plotly.restyle(plotDiv,
    { opacity: data.map(t => (j === null || t.legendgroup === j) ? 1 : DIM_OPACITY) });
}
function attachLegendHover() {
  const label2judge = new Map();
  (INDEX.judges || []).forEach(j => label2judge.set(judgeLabel(j), j));
  plotDiv.querySelectorAll(".legend .traces").forEach(entry => {
    const t = entry.querySelector(".legendtext");
    const j = t ? label2judge.get(t.textContent) : null;
    if (!j) return;
    entry.style.cursor = "pointer";
    entry.onmouseenter = () => emphasizeJudge(j);
    entry.onmouseleave = () => emphasizeJudge(null);
  });
}

/* --- fetch + cache ------------------------------------------------------- */
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + " → HTTP " + res.status);
  return res.json();
}
function cachedJSON(cache, key, url) {
  if (!cache.has(key))
    cache.set(key, fetchJSON(url).catch(err => { cache.delete(key); throw err; }));
  return cache.get(key);
}
const getPair = (major, minor, frac) => {
  const key = major + "__" + minor + "__" + frac;
  return cachedJSON(pairCache, key, "data/pairs/" + key + ".json");
};
const getTranscript = stem => cachedJSON(transCache, stem, "data/transcripts/" + stem + ".json");

/* --- modal: open + header ----------------------------------------------- */
async function openModal(run, keepItem) {
  const prev = keepItem && MODAL ? MODAL : null;
  MODAL = { run: run, swapped: prev ? prev.swapped : false,
            item: prev ? prev.item : null, filter: prev ? prev.filter : "all",
            search: prev ? prev.search : "", pair: null, tMajor: null, tMinor: null };
  overlay.hidden = false;
  document.body.style.overflow = "hidden";
  $("detail").hidden = true;
  $("detail-loading").hidden = false;
  $("detail-loading").textContent = "Loading…";
  $("item-list").innerHTML = "";
  $("item-search").value = MODAL.search;
  buildHeader(run);

  try {
    const minorStem = run.is_self ? (run.minor + "__ep1") : run.minor;
    const results = await Promise.all([
      getPair(run.major, run.minor, run.frac),
      getTranscript(run.major),
      getTranscript(minorStem),
    ]);
    if (MODAL.run !== run) return;  // a later click superseded this one
    MODAL.pair = results[0];
    MODAL.tMajor = results[1];
    MODAL.tMinor = results[2];
    buildItemFilterBar();
    renderItemList();
  } catch (e) {
    $("detail-loading").textContent = "Could not load this run: " + e.message;
  }
}

function buildHeader(run) {
  setupSelectors(run);
  const M = modelMeta(run.major), N = modelMeta(run.minor);
  $("cmp-submeta").innerHTML = run.is_self
    ? esc(M.label) + " " + metaStr(run.major) + " vs itself (self-pair)"
    : esc(M.label) + " " + metaStr(run.major) + " vs " + esc(N.label) + " " + metaStr(run.minor) +
      " · <b>gap</b> " + fmt(run.d_acc);
  const parts = [
    "judge accuracy <b>" + fmt(run.score) + "</b> [" + fmt(run.score_lo) + ", " +
      fmt(run.score_hi) + "] · n=" + run.n_items,
    "disagreement <b>" + fmt(run.disagree_rate) + "</b>",
    "no-answer <b>" + fmt(run.noanswer_rate) + "</b>",
  ];
  $("modal-meta").innerHTML = parts.map(p => "<span>" + p + "</span>").join("");
}

/* --- in-modal truncation switch: same comparison, different cut level ----- */
const FRAC_ORDER = ["full", "90", "50", "10"];
const fracLabel = f => (f === "full" ? "Full" : f + "%");
async function setModalFrac(frac) {
  const base = MODAL.run;
  const nr = INDEX.runs.find(r => r.judge === base.judge && r.major === base.major &&
    r.minor === base.minor && r.frac === frac);
  if (!nr) return;
  MODAL.run = nr;               // frac-specific run drives header stats + truncation
  buildHeader(nr);
  $("detail").hidden = true;
  $("detail-loading").hidden = false;
  $("detail-loading").textContent = "Loading…";
  try {
    const pair = await getPair(nr.major, nr.minor, frac);
    if (MODAL.run !== nr) return;   // superseded by another switch
    MODAL.pair = pair;
    buildItemFilterBar();
    renderItemList();               // keeps the selected item if still present
  } catch (e) {
    $("detail-loading").textContent = "Could not load this level: " + e.message;
  }
}

/* --- verdict-order helpers ---------------------------------------------- */
function ordersOf(item, judge) {
  const cell = MODAL.pair[item];
  if (!cell || !cell[judge]) return null;
  return cell[judge].orders || null;
}
function orderBySlot(orders, slot) {
  if (!orders) return null;
  for (let i = 0; i < orders.length; i++) if (orders[i].slot_of_major === slot) return orders[i];
  return null;
}
const isNull = v => v === null || v === undefined;

/* J's per-item badge + filter category (J = the clicked judge) */
function itemBadge(item) {
  const run = MODAL.run, J = run.judge;
  const orders = ordersOf(item, J);
  const oa = orderBySlot(orders, "A"), ob = orderBySlot(orders, "B");

  if (run.is_self) {
    const pa = oa ? oa.pick : null, pb = ob ? ob.pick : null;
    let cat;
    if (isNull(pa) || isNull(pb)) cat = "noanswer";
    else if (pa === pb) cat = pa === "A" ? "slotA" : "slotB";
    else cat = "disagree";
    return { html: selfMark(pa) + selfMark(pb), cat: cat };
  }
  const sa = markState(oa), sb = markState(ob);
  let cat;
  if (sa === "null" || sb === "null") cat = "noanswer";
  else if (sa === "good" && sb === "good") cat = "both";
  else if (sa === "bad" && sb === "bad") cat = "wrong";
  else cat = "disagree";
  return { html: mark(sa) + mark(sb), cat: cat };
}
function markState(o) {
  if (!o || isNull(o.pick)) return "null";
  return o.picked_major === 1 ? "good" : "bad";
}
const mark = s => s === "good" ? '<span class="mk-good">✓</span>'
  : s === "bad" ? '<span class="mk-bad">✗</span>'
  : '<span class="mk-null">∅</span>';
const selfMark = p => isNull(p)
  ? '<span class="mk-null">∅</span>'
  : '<span class="mk-null" style="color:var(--ink2)">' + esc(p) + "</span>";

/* --- item filter bar + list --------------------------------------------- */
const CROSS_FILTERS = [["all", "all"], ["both", "both-correct"], ["wrong", "both-wrong"],
                       ["disagree", "disagreements"], ["noanswer", "no-answer"]];
const SELF_FILTERS = [["all", "all"], ["slotA", "slot-A picked"], ["slotB", "slot-B picked"],
                      ["disagree", "disagreements"]];

function buildItemFilterBar() {
  const defs = MODAL.run.is_self ? SELF_FILTERS : CROSS_FILTERS;
  const box = $("item-filter-btns");
  box.innerHTML = "";
  defs.forEach(d => {
    const b = document.createElement("button");
    b.textContent = d[1];
    b.dataset.filter = d[0];
    if (d[0] === MODAL.filter) b.className = "active";
    b.addEventListener("click", () => {
      MODAL.filter = d[0];
      Array.from(box.children).forEach(c => c.className = c.dataset.filter === d[0] ? "active" : "");
      renderItemList();
    });
    box.appendChild(b);
  });
  $("item-search").oninput = e => { MODAL.search = e.target.value.trim(); renderItemList(); };
}

function catMatches(cat) {
  if (MODAL.filter === "all") return true;
  return cat === MODAL.filter;
}

function renderItemList() {
  const list = $("item-list");
  list.innerHTML = "";
  const J = MODAL.run.judge;
  // anchor to the clicked judge: only items J actually judged (pair files hold
  // the union across judges, who each judged a slightly different subset)
  const items = Object.keys(MODAL.pair).filter(it => ordersOf(it, J)).sort(naturalCompare);
  const search = MODAL.search.toLowerCase();
  let visible = [];
  items.forEach(item => {
    const badge = itemBadge(item);
    if (!catMatches(badge.cat)) return;
    if (search && String(item).toLowerCase().indexOf(search) === -1) return;
    visible.push(item);
    const li = document.createElement("li");
    li.dataset.item = item;
    li.innerHTML = '<span class="item-badge">' + badge.html + "</span>" +
                   '<span class="item-id">' + esc(item) + "</span>";
    li.addEventListener("click", () => selectItem(item));
    list.appendChild(li);
  });

  if (!visible.length) {
    list.innerHTML = '<div class="no-items">No items match this filter.</div>';
    $("detail").hidden = true;
    $("detail-loading").hidden = false;
    $("detail-loading").textContent = "No items match this filter.";
    MODAL.item = null;
    return;
  }
  const keep = MODAL.item && visible.indexOf(MODAL.item) !== -1 ? MODAL.item : visible[0];
  selectItem(keep);
}

function selectItem(item) {
  MODAL.item = item;
  Array.from($("item-list").children).forEach(li =>
    li.classList && li.classList.toggle("active", li.dataset && li.dataset.item === item));
  renderDetail();
}

/* --- detail -------------------------------------------------------------- */
function sideLabel(model, epochTag) {
  const l = modelMeta(model).label;
  return MODAL.run.is_self ? (l + " " + epochTag) : l;
}

function renderDetail() {
  const run = MODAL.run, item = MODAL.item;
  if (!item) return;
  $("detail-loading").hidden = true;
  $("detail").hidden = false;

  const q = (INDEX.questions && INDEX.questions[item] !== undefined)
    ? INDEX.questions[item] : "(question text unavailable)";
  $("detail-question").textContent = q;

  const majorFull = (MODAL.tMajor && MODAL.tMajor[item]) || "";
  const minorFull = (MODAL.tMinor && MODAL.tMinor[item]) || "";
  let majT = majorFull, minT = minorFull;
  if (run.frac !== "full") {
    const cut = truncatePair(majorFull, minorFull, Number(run.frac) / 100);
    majT = cut[0]; minT = cut[1];
    $("trunc-note").hidden = false;
    $("trunc-note").textContent = "Transcripts truncated to " + run.frac +
      "% (answers removed, both sides cut to the same length — exactly as the judge saw them).";
  } else {
    $("trunc-note").hidden = true;
  }

  const swapped = MODAL.swapped;
  // arrangement: default major->A; swapped major->B. (self: epoch0 = major side, epoch1 = ep1 side)
  const labelMajor = sideLabel(run.major, "(epoch 0)");
  const labelMinor = sideLabel(run.minor, "(epoch 1)");
  const labelA = swapped ? labelMinor : labelMajor;
  const labelB = swapped ? labelMajor : labelMinor;
  const textA = swapped ? minT : majT;
  const textB = swapped ? majT : minT;

  const ansTag = a => a ? ' <span class="ans-tag">answered ' + a +
    ' <span class="mk-good">✓</span></span>' : "";
  $("colA-head").innerHTML = '<span class="slot">A</span> = ' + esc(labelA) + ansTag(answerLetter(textA));
  $("colB-head").innerHTML = '<span class="slot">B</span> = ' + esc(labelB) + ansTag(answerLetter(textB));
  $("colA-text").textContent = textA;
  $("colB-text").textContent = textB;
  $("colA-text").scrollTop = 0;
  $("colB-text").scrollTop = 0;

  const majSlot = swapped ? "B" : "A";
  renderVerdictList(item, majSlot, labelA, labelB);
}

function renderVerdictList(item, targetSlot, labelA, labelB) {
  const run = MODAL.run, J = run.judge;
  const container = $("verdict-list");
  container.innerHTML = "";
  const cell = MODAL.pair[item] || {};
  const order = (INDEX.judges || []).filter(j => cell[j]);
  // include any judges present but not in the fixed order, appended
  Object.keys(cell).forEach(j => { if (order.indexOf(j) === -1) order.push(j); });

  let starRow = null;
  order.forEach(j => {
    const meta = modelMeta(j);
    const o = orderBySlot(cell[j].orders, targetSlot);
    const row = document.createElement("div");
    row.className = "vrow" + (j === J ? " star" : "");

    // judge cell
    const jc = document.createElement("div");
    jc.className = "vjudge";
    jc.innerHTML = (j === J ? '<span class="jchip">★ your judge</span> ' : "") + esc(meta.label) +
      '<span class="sub">GPQA ' + fmt(meta.acc) + " · " + esc(meta.rel || "–") + "</span>";
    row.appendChild(jc);

    // pick + correctness cell
    const pc = document.createElement("div");
    pc.className = "vpick";
    if (!o || isNull(o.pick)) {
      pc.innerHTML = '<span class="mk-null">no answer ∅</span>';
    } else {
      const pickedLabel = o.pick === "A" ? labelA : o.pick === "B" ? labelB : "–";
      let corr = "";
      if (!run.is_self) corr = o.picked_major === 1
        ? ' <span class="mk-good">✓</span>'
        : o.picked_major === 0 ? ' <span class="mk-bad">✗</span>' : "";
      pc.innerHTML = 'picked <span class="letter">' + esc(o.pick) + "</span> = " +
        esc(pickedLabel) + corr;
    }
    row.appendChild(pc);

    // confidence cell
    const cc = document.createElement("div");
    cc.className = "vconf";
    cc.textContent = (o && !isNull(o.confidence)) ? ("conf " + o.confidence + "/5") : "";
    row.appendChild(cc);

    // explanation (spans full width)
    const ex = document.createElement("div");
    const text = o && o.explanation ? String(o.explanation) : "";
    if (text) { ex.className = "vexpl"; ex.textContent = text; }
    else { ex.className = "vexpl empty"; ex.textContent = "(no explanation recorded)"; }
    row.appendChild(ex);

    container.appendChild(row);
    if (j === J) starRow = row;
  });

  if (starRow) {
    const c = container.getBoundingClientRect(), s = starRow.getBoundingClientRect();
    container.scrollTop += (s.top - c.top) - (c.height / 2 - s.height / 2);
  }
}

/* --- modal chrome -------------------------------------------------------- */
function closeModal() {
  overlay.hidden = true;
  document.body.style.overflow = "";
  MODAL = null;
}
$("modal-close").addEventListener("click", closeModal);
$("swap-btn").addEventListener("click", () => {
  if (!MODAL || !MODAL.item) return;
  MODAL.swapped = !MODAL.swapped;
  renderDetail();
});
overlay.addEventListener("mousedown", e => { if (e.target === overlay) closeModal(); });
document.addEventListener("keydown", e => { if (e.key === "Escape" && !overlay.hidden) closeModal(); });

/* --- wiring + boot ------------------------------------------------------- */
[ctlTrunc, ctlSubset, ctlArm, ctlColorby, ctlCurves].forEach(c =>
  c.addEventListener("change", renderPlot));
if (window.matchMedia) {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const onTheme = () => renderPlot();
  if (mq.addEventListener) mq.addEventListener("change", onTheme);
  else if (mq.addListener) mq.addListener(onTheme);
}

async function boot() {
  try {
    INDEX = await fetchJSON("data/index.json");
  } catch (e) {
    hintEl.textContent = "";
    emptyNote.hidden = false;
    emptyNote.textContent = "Could not load data/index.json (" + e.message +
      "). Serve this directory over HTTP and rebuild the dataset.";
    return;
  }
  const xs = INDEX.runs.filter(r => !r.is_self).map(r => r.d_acc);
  XMAX = Math.max((xs.length ? Math.max.apply(null, xs) : 0.3) * 1.06, 0.05);
  renderPlot();
}
boot();
