/* gpqa-pilot-rd2 gap-curve viewer.
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

/* --- house palette fallback (plots.py PALETTE), used only if colors{} miss  */
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948", "#8c6d31", "#0b7285"];

/* --- DOM refs ------------------------------------------------------------ */
const $ = id => document.getElementById(id);
const plotDiv = $("plot");
const emptyNote = $("empty-note");
const hintEl = $("ctl-hint");
const ctlDataset = $("ctl-dataset");
const ctlView = $("ctl-view");
const rankWrap = $("rank-wrap"), rankPlot = $("rank-plot");
const ctlTrunc = $("ctl-trunc"), ctlSubset = $("ctl-subset"), ctlSource = $("ctl-source"),
      ctlColorby = $("ctl-colorby"), ctlCurves = $("ctl-curves"), ctlXmetric = $("ctl-xmetric");

/* x-axis gap under the selected metric; datasets without the fields (older
   bakes) fall back to their baked per-bench d_acc */
function gapOf(r) {
  const M = INDEX.models[r.major], N = INDEX.models[r.minor], k = ctlXmetric.value;
  const f = k === "eci" ? "eci" : "acc_" + k;
  if (M && N && M[f] != null && N[f] != null) return Math.abs(M[f] - N[f]);
  return r.d_acc;
}
/* the dataset may ship its own x-axis choices (INDEX.xmetrics); these are the
   fallback titles for the ones hardcoded in index.html */
const X_TITLES = { eci: "capability gap (ECI)", gpqa: "capability gap (GPQA accuracy difference)",
                   domain: "capability gap (ground-truth quartile accuracy difference)",
                   mbpp: "capability gap (MBPP accuracy difference)",
                   aime2025: "capability gap (AIME 2025 accuracy difference)",
                   aime2026: "capability gap (AIME 2026 accuracy difference)" };
const xmetric = key => ((INDEX && INDEX.xmetrics) || []).find(m => m.key === key);
const xTitle = () => {
  const m = xmetric(ctlXmetric.value);
  return (m && m.title) || X_TITLES[ctlXmetric.value] || X_TITLES.eci;
};
const XMETRIC_HTML = $("ctl-xmetric").innerHTML;   // the static fallback list
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
let DATASETS = [];        // [{id, label}] from data/datasets.json, menu order
let DATASET = null;       // the selected id; every data path hangs off it

/* Caches are per dataset: switching datasets clears them, since the same stem
   means a different file under a different dataset. */
let pairCache = new Map();   // "maj__min__frac" -> Promise<pairData>
let transCache = new Map();  // stem -> Promise<{item:text}>
let optsCache = new Map();   // stem -> Promise<{item:[choice]}>
let ansCache = new Map();    // stem -> Promise<{item:letter}>

let MODAL = null;  // { run, swapped, item, filter, search, pair, tMajor, tMinor }

/* --- theme (Plotly needs explicit colours; these mirror style.css) -------
   Light only, deliberately: the page no longer follows prefers-color-scheme. */
function themeColors() {
  return { surface: "#ffffff", panel: "#f6f4ee", ink: "#0b0b0b", ink2: "#33322f",
           muted: "#4a4844", grid: "#d3d2c9", border: "#d9d7ce" };
}

/* --- metadata helpers ---------------------------------------------------- */
function modelMeta(m) {
  const x = (INDEX && INDEX.models && INDEX.models[m]) ? INDEX.models[m] : {};
  return { label: x.label || m, acc: x.accuracy, rel: x.release || "" };
}
const judgeLabel = j => modelMeta(j).label;
/* Accuracy of one model under the selected x-metric — the ranking view plots and ranks by
   it, so switching the axis re-scores the ranking against a different slice of the domain. */
function accOf(m) {
  const x = (INDEX && INDEX.models && INDEX.models[m]) ? INDEX.models[m] : {};
  const v = x["acc_" + ctlXmetric.value];
  return v === null || v === undefined ? num(x.accuracy) : v;
}
const xmetricLabel = () => {
  const m = xmetric(ctlXmetric.value);
  return m ? m.label : (INDEX.metric_label || "accuracy");
};
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
  const frac = ctlTrunc.value, subset = ctlSubset.value, src = ctlSource.value;
  return INDEX.runs.filter(r => {
    if (r.frac !== frac) return false;
    if (subset === "weak" && !r.weak_judge) return false;
    if (subset === "old" && !r.old_judge) return false;
    if (src === "collected" && !r.arm_a) return false;
    if (src === "epoch" && !(INDEX.models[r.major].arm === "B" && INDEX.models[r.minor].arm === "B")) return false;
    return true;
  });
}

/* --- hover --------------------------------------------------------------- */
function hoverData(r) {
  const j = modelMeta(r.judge), M = modelMeta(r.major), N = modelMeta(r.minor);
  return [j.label, num(j.acc), j.rel, M.label, num(M.acc), M.rel,
          N.label, num(N.acc), N.rel, gapOf(r), r.score, r.score_lo, r.score_hi,
          r.n_items, r.slot_a];
}
const MAIN_HOVER =
  "<b>%{customdata[0]}</b> — judge · accuracy %{customdata[1]:.2f} · %{customdata[2]}<br>" +
  "<b>major:</b> %{customdata[3]} (accuracy %{customdata[4]:.2f} · %{customdata[5]})<br>" +
  "<b>minor:</b> %{customdata[6]} (accuracy %{customdata[7]:.2f} · %{customdata[8]})<br>" +
  "<b>gap:</b> %{customdata[9]:.3f} · <b>judge acc:</b> %{customdata[10]:.2f} " +
  "[%{customdata[11]:.2f}, %{customdata[12]:.2f}]<br>" +
  "<b>n:</b> %{customdata[13]} · <b>slot-1 rate:</b> %{customdata[14]:.2f}<extra></extra>";
const SELF_HOVER =
  "<b>%{customdata[0]}</b> — judge · accuracy %{customdata[1]:.2f} · %{customdata[2]}<br>" +
  "<b>self-pair control:</b> %{customdata[3]} vs itself (should sit at chance)<br>" +
  "<b>judge acc:</b> %{customdata[10]:.2f} [%{customdata[11]:.2f}, %{customdata[12]:.2f}]<br>" +
  "<b>n:</b> %{customdata[13]} · <b>slot-1 rate:</b> %{customdata[14]:.2f}<extra></extra>";

/* --- curve traces -------------------------------------------------------- */
function lineTrace(j, xs, ys, color) {
  return { type: "scatter", mode: "lines", legendgroup: j, showlegend: false,
           x: xs, y: ys, line: { color: color, width: 1.6 }, hoverinfo: "skip",
           xaxis: "x", yaxis: "y" };
}
/* the stored fits are per-judge sigmoids on the full, untruncated arm-A ladder;
   drawing them over a filtered subset misrepresents it, so only show them there */
function sigmoidOK() {
  return ctlTrunc.value === "full" && ctlSubset.value === "all" && ctlSource.value !== "epoch";
}
function buildCurve(j, crossPts, color) {
  const mode = ctlCurves.value;
  if (mode === "off") return null;
  if (mode === "sigmoid" && sigmoidOK()) {
    const f = INDEX.fits && INDEX.fits[j];
    if (!f) return null;
    const xs = [], ys = [], N = 60;
    for (let i = 0; i <= N; i++) {
      const x = XMAX * i / N;
      xs.push(x); ys.push(1 / (1 + Math.exp(-(f.a + f.d * x))));
    }
    return lineTrace(j, xs, ys, color);
  }
  // connect runs — also the honest fallback when a sigmoid fit can't represent the subset
  const pts = crossPts.slice().sort((a, b) => gapOf(a) - gapOf(b));
  if (pts.length < 2) return null;
  return lineTrace(j, pts.map(r => gapOf(r)), pts.map(r => r.score), color);
}

/* --- plot ---------------------------------------------------------------- */
/* paper x of the y-axis: the left edge of the self strip it anchors to */
const YAXIS_X = 0.12;

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
    xaxis2: { domain: [YAXIS_X, 0.16], range: [-0.85, 0.85], anchor: "y",
              tickvals: [0], ticktext: ["self"], tickfont: { color: theme.muted },
              showgrid: false, zeroline: false, linecolor: theme.muted, ticks: "" },
    yaxis2: { domain: [0, 1], range: [0, 1], matches: "y", anchor: "x",
              showticklabels: false, gridcolor: theme.grid, showgrid: true, zeroline: false },
    xaxis: { domain: [0.24, 1], range: [0, XMAX], anchor: "y2",
             title: { text: xTitle() },
             gridcolor: theme.grid, zeroline: false, linecolor: theme.muted,
             ticks: "outside", tickcolor: theme.muted },
    legend: { x: 1.02, y: 0.5, xanchor: "left", yanchor: "middle",
              title: { text: "judge — ordered by " +
                (ctlColorby.value === "release" ? "release" : "capability"),
                font: { color: theme.ink2, size: 12 } },
              font: { color: theme.ink2, size: 12 },
              groupclick: "togglegroup", bgcolor: "rgba(0,0,0,0)" },
    shapes: [
      // starts at the y-axis, not the paper edge, or it runs over the axis label
      { type: "line", xref: "paper", x0: YAXIS_X, x1: 1, yref: "y2", y0: 0.5, y1: 0.5,
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
  // which model of a pair is "major" was decided by one subset; say so when the x axis shows another
  if (INDEX.default_xmetric && ctlXmetric.value !== INDEX.default_xmetric) {
    const truth = xmetric(INDEX.default_xmetric), shown = xmetric(ctlXmetric.value);
    if (truth && shown)
      hint += " · pairs oriented by " + truth.label + " accuracy, x shows the " + shown.label + " gap";
  }
  if (ctlCurves.value === "sigmoid" && !sigmoidOK())
    hint += " · sigmoid fits apply to the full untruncated set — connected runs shown for this subset";
  hintEl.textContent = hint;

  if (!runs.length) {
    emptyNote.textContent = "No runs match the current filters.";
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
      x: cj.map(r => gapOf(r)), y: cj.map(r => r.score),
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

  // Plotly only instantiates x2/y2 when a trace uses them, and the ticked y-axis
  // anchors to x2 while x anchors to y2. Truncated runs have no self-pairs, so
  // without this placeholder those anchors dangle and nothing draws at all.
  if (!self.length) {
    traces.push({ type: "scatter", mode: "markers", x: [], y: [],
                  xaxis: "x2", yaxis: "y2", showlegend: false, hoverinfo: "skip" });
    runMap.push(null);
  }

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
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(url + " → HTTP " + res.status);
  return res.json();
}
function cachedJSON(cache, key, url) {
  if (!cache.has(key))
    cache.set(key, fetchJSON(url).catch(err => { cache.delete(key); throw err; }));
  return cache.get(key);
}
const dataURL = rest => "data/" + DATASET + "/" + rest;
const getPair = (major, minor, frac) => {
  const key = major + "__" + minor + "__" + frac;
  return cachedJSON(pairCache, key, dataURL("pairs/" + key + ".json"));
};
const getTranscript = stem => cachedJSON(transCache, stem, dataURL("transcripts/" + stem + ".json"));
/* Options are shuffled per model, so each side needs its own order for its
   answer letter to mean anything. Missing file (non-MCQ benchmark) -> {}. */
const getOptions = stem =>
  cachedJSON(optsCache, stem, dataURL("options/" + stem + ".json")).catch(() => ({}));
/* The letter each model answered, graded at export time. Not scraped from the
   transcript here: models write the answer line in whatever format they like,
   and truncated runs have it cut off. Missing file -> {}. */
const getAnswers = stem =>
  cachedJSON(ansCache, stem, dataURL("answers/" + stem + ".json")).catch(() => ({}));

/* --- modal: open + header ----------------------------------------------- */
async function openModal(run, keepItem) {
  const prev = keepItem && MODAL ? MODAL : null;
  MODAL = { run: run, swapped: prev ? prev.swapped : false,
            item: prev ? prev.item : null, filter: prev ? prev.filter : "all",
            search: prev ? prev.search : "", pair: null, tMajor: null, tMinor: null,
            oMajor: null, oMinor: null, aMajor: null, aMinor: null };
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
      getOptions(run.major),
      getOptions(minorStem),
      getAnswers(run.major),
      getAnswers(minorStem),
    ]);
    if (!MODAL || MODAL.run !== run) return;  // superseded by a later click, or closed
    MODAL.pair = results[0];
    MODAL.tMajor = results[1];
    MODAL.tMinor = results[2];
    MODAL.oMajor = results[3];
    MODAL.oMinor = results[4];
    MODAL.aMajor = results[5];
    MODAL.aMinor = results[6];
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
      " · <b>gap</b> " + fmt(gapOf(run));
  const m = run.metrics || {};
  const stat = (label, value, digits) =>
    label + " <b>" + (value === null || value === undefined ? "–" : fmt(value, digits)) + "</b>";
  const parts = [
    "judge accuracy <b>" + fmt(run.score) + "</b> [" + fmt(run.score_lo) + ", " +
      fmt(run.score_hi) + "] · n=" + run.n_items,
    stat("consistent", m.consistent_accuracy),
    stat("position bias", m.position_bias),
    stat("length corr", m.length_corr),
    "disagreement <b>" + fmt(run.disagree_rate) + "</b>",
    "no-answer <b>" + fmt(run.noanswer_rate) + "</b>",
  ];
  if (m.confidence !== null && m.confidence !== undefined) parts.push(stat("confidence", m.confidence, 1));
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
    if (!MODAL || MODAL.run !== nr) return;   // superseded by another switch, or closed
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
  const oa = orderBySlot(orders, "1"), ob = orderBySlot(orders, "2");

  if (run.is_self) {
    const pa = oa ? oa.pick : null, pb = ob ? ob.pick : null;
    let cat;
    if (isNull(pa) || isNull(pb)) cat = "noanswer";
    else if (pa === pb) cat = pa === "1" ? "slotA" : "slotB";
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
const SELF_FILTERS = [["all", "all"], ["slotA", "slot-1 picked"], ["slotB", "slot-2 picked"],
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

/* The options this side's model saw, in its own shuffled order, with the letter
   it answered marked — the same list the judge was shown above this transcript. */
function renderOptions(elId, choices, answered) {
  const el = $(elId);
  if (!el) return;
  if (!choices.length) { el.hidden = true; el.innerHTML = ""; return; }
  el.hidden = false;
  el.innerHTML = choices.map((c, i) => {
    const letter = String.fromCharCode(65 + i);
    const cls = letter === answered ? ' class="opt picked"' : ' class="opt"';
    return "<div" + cls + '><span class="opt-letter">' + letter + ")</span> " +
           esc(String(c).trim()) + "</div>";
  }).join("");
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

  const optsMaj = (MODAL.oMajor && MODAL.oMajor[item]) || [];
  const optsMin = (MODAL.oMinor && MODAL.oMinor[item]) || [];
  const optsA = swapped ? optsMin : optsMaj;
  const optsB = swapped ? optsMaj : optsMin;

  const ansMaj = (MODAL.aMajor && MODAL.aMajor[item]) || null;
  const ansMin = (MODAL.aMinor && MODAL.aMinor[item]) || null;
  const ansA = swapped ? ansMin : ansMaj;
  const ansB = swapped ? ansMaj : ansMin;

  const ansTag = a => a ? ' <span class="ans-tag">answered ' + a +
    ' <span class="mk-good">✓</span></span>' : "";
  $("colA-head").innerHTML = '<span class="slot">1</span> = ' + esc(labelA) + ansTag(ansA);
  $("colB-head").innerHTML = '<span class="slot">2</span> = ' + esc(labelB) + ansTag(ansB);
  renderOptions("colA-options", optsA, ansA);
  renderOptions("colB-options", optsB, ansB);
  $("colA-text").textContent = textA;
  $("colB-text").textContent = textB;
  $("colA-text").scrollTop = 0;
  $("colB-text").scrollTop = 0;

  const majSlot = swapped ? "2" : "1";
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
      '<span class="sub">acc ' + fmt(meta.acc) + " · " + esc(meta.rel || "–") + "</span>";
    row.appendChild(jc);

    // pick + correctness cell
    const pc = document.createElement("div");
    pc.className = "vpick";
    if (!o || isNull(o.pick)) {
      pc.innerHTML = '<span class="mk-null">no answer ∅</span>';
    } else {
      const pickedLabel = o.pick === "1" ? labelA : o.pick === "2" ? labelB : "–";
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

    // the judge's own CoT, when the provider returned one — long, so collapsed
    if (o && o.reasoning) {
      const det = document.createElement("details");
      det.className = "vreason";
      const sum = document.createElement("summary");
      sum.textContent = "judge reasoning — " + o.reasoning.length.toLocaleString() + " chars";
      const pre = document.createElement("pre");
      pre.textContent = o.reasoning;
      det.appendChild(sum);
      det.appendChild(pre);
      row.appendChild(det);
    }

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


/* --- judge metrics ------------------------------------------------------
   judge.scorer's metrics, pooled over every cross-model run in the dataset:
   they describe the judge, not the current filter selection. */
const METRIC_COLS = [
  ["accuracy", "accuracy", "share of judgements that named the more capable model; no verdict counts as wrong"],
  ["score", "score (answered)", "the same over answered judgements only — what the plot shows"],
  ["consistent_accuracy", "consistent", "right in BOTH orders; a pure position-reader scores 0 here"],
  ["answer_rate", "answer rate", "share of judgements with an ANSWER: line"],
  ["position_bias", "position bias", "-1 always picks Transcript 1, +1 always Transcript 2, 0 unbiased"],
  ["length_corr", "length corr", "phi between 'Transcript 1 is longer' and 'picked Transcript 1'"],
  ["disagree_rate", "disagreement", "share of items where the two orders contradict each other"],
  ["confidence", "confidence", "mean self-reported confidence, 1-5, blank when unreported"],
];

function metricCells(m) {
  return METRIC_COLS.map(([key]) =>
    '<td>' + (m && m[key] !== null && m[key] !== undefined ? fmt(m[key]) : "&ndash;") + "</td>").join("");
}

function renderJudgeMetrics() {
  const table = $("judge-metrics"), note = $("judge-metrics-note");
  const byJudge = INDEX.judge_metrics || {};
  const judges = Object.keys(byJudge).sort(cmpJudge);
  if (!judges.length) { table.innerHTML = ""; note.textContent = ""; return; }
  const head = ["judge", "runs", "n"].concat(METRIC_COLS.map(c => c[1]));
  const tips = ["", "cross-model pairs judged", "judgements pooled"].concat(METRIC_COLS.map(c => c[2]));
  table.innerHTML =
    "<thead><tr>" + head.map((h, i) =>
      '<th title="' + esc(tips[i]) + '">' + esc(h) + "</th>").join("") + "</tr></thead><tbody>" +
    judges.map(j => {
      const m = byJudge[j];
      return "<tr><td>" + esc(modelMeta(j).label) + "</td><td>" + (m.n_runs === undefined ? "&ndash;" : m.n_runs) +
        "</td><td>" + m.n + "</td>" + metricCells(m) + "</tr>";
    }).join("") + "</tbody>";
  const withSelf = judges.filter(j => byJudge[j].self_position_bias !== undefined);
  note.innerHTML = "Pooled over every cross-model judgement in this dataset (untruncated), " +
    "independent of the filters above." +
    (withSelf.length
      ? " Self-pair control — " + withSelf.map(j => esc(modelMeta(j).label) + " position bias <b>" +
          fmt(byJudge[j].self_position_bias) + "</b> (n=" + byJudge[j].self_n + ")").join(", ") + "."
      : "");
}

/* --- BT ranking view ------------------------------------------------------
   The ranking the judge's verdicts imply (Bradley-Terry ability, ranking.csv)
   against the ground truth it is scored on (q_dst accuracy). A model on the
   diagonal is placed correctly; vertical spread at one accuracy is where the
   judge disagrees with the benchmark. */
function rankLayout(theme) {
  return {
    margin: { l: 62, r: 20, t: 12, b: 56 },
    paper_bgcolor: theme.surface, plot_bgcolor: theme.surface,
    font: { color: theme.ink2, size: 13 },
    hovermode: "closest",
    hoverlabel: { bgcolor: theme.panel, bordercolor: theme.border,
                  font: { color: theme.ink, size: 12 }, align: "left" },
    showlegend: false,
    xaxis: { title: { text: xmetricLabel() + " accuracy (ground truth)" },
             gridcolor: theme.grid, zeroline: false, linecolor: theme.muted,
             ticks: "outside", tickcolor: theme.muted },
    yaxis: { title: { text: "Bradley–Terry ability (log-odds)" },
             gridcolor: theme.grid, zeroline: true, zerolinecolor: theme.grid,
             linecolor: theme.muted, ticks: "outside", tickcolor: theme.muted },
  };
}

/* rows joined to the model metadata, ordered by BT rank, with the rank the
   ground-truth accuracy would have given them */
function rankRows() {
  const rows = (INDEX.ranking || []).map(r => Object.assign({}, r, {
    label: modelMeta(r.model).label, acc: num(accOf(r.model)),
    win_rate: r.n_comparisons ? r.n_wins / r.n_comparisons : NaN,
  }));
  rows.slice().sort((a, b) => b.acc - a.acc).forEach((r, i) => { r.acc_rank = i + 1; });
  return rows.sort((a, b) => a.rank - b.rank);
}

function renderRanking() {
  const theme = themeColors();
  const rows = rankRows();
  const table = $("rank-table");
  if (!rows.length) {
    Plotly.react(rankPlot, [], rankLayout(theme), { displayModeBar: false, responsive: true });
    table.innerHTML = "";
    $("rank-caption").textContent =
      "No ranking.csv beside this dataset's logs — run `uv run rank <log dir> --domain <domain.yml>` and re-export.";
    return;
  }

  const misplaced = rows.filter(r => Math.abs(r.rank - r.acc_rank) >= 3);
  Plotly.react(rankPlot, [{
    type: "scatter", mode: "markers+text",
    x: rows.map(r => r.acc), y: rows.map(r => r.ability),
    text: rows.map(r => r.label), textposition: "middle right",
    textfont: { size: 9.5, color: theme.muted },
    marker: {
      size: rows.map(r => 6 + 10 * Math.sqrt(r.n_comparisons / Math.max(...rows.map(x => x.n_comparisons)))),
      color: PALETTE[0], opacity: 0.85, line: { color: theme.surface, width: 1 },
    },
    customdata: rows.map(r => [r.label, r.rank, r.acc_rank, r.n_comparisons, r.win_rate, r.n_ties]),
    hovertemplate: "<b>%{customdata[0]}</b><br>ability %{y:.2f} (BT rank %{customdata[1]})<br>" +
      "accuracy %{x:.3f} (rank %{customdata[2]})<br>" +
      "%{customdata[3]} comparisons · win rate %{customdata[4]:.2f} · %{customdata[5]} ties<extra></extra>",
  }], rankLayout(theme), { displayModeBar: false, responsive: true });

  const scores = INDEX.scores || [];
  const tau = (INDEX.taus || {})[ctlXmetric.value];
  $("rank-caption").innerHTML =
    "Kendall &tau; vs <b>" + esc(xmetricLabel()) + "</b> accuracy (the plotted axis): <b>" +
    fmt(tau, 3) + "</b> over " + rows.length + " models &middot; marker size = comparisons &middot; " +
    misplaced.length + " model(s) misplaced by 3+ ranks." +
    (scores.length
      ? "<br><code>scores.csv</code> (<code>uv run rank</code>): " +
        scores.map(s => "<b>" + esc(s.quartile) + "</b> " + fmt(s.correlation, 3)).join(" &middot; ") +
        " &mdash; a few points off the plotted &tau; because it means accuracy over response " +
        "rows, while the axis averages repeats per item first (the definition that oriented the pairs)."
      : "");

  const cols = ["BT rank", "model", "ability", xmetricLabel() + " acc", "acc rank",
                "&Delta;", "comparisons", "win rate", "ties"];
  table.innerHTML =
    "<thead><tr>" + cols.map(c => "<th>" + c + "</th>").join("") + "</tr></thead><tbody>" +
    rows.map(r => {
      const d = r.acc_rank - r.rank;
      const dCell = d === 0 ? '<span class="rank-thin">0</span>'
        : '<span class="' + (Math.abs(d) >= 3 ? "mk-bad" : "rank-thin") + '">' +
          (d > 0 ? "+" : "") + d + "</span>";
      return "<tr><td>" + r.rank + "</td><td>" + esc(r.label) + "</td><td>" + fmt(r.ability) +
        "</td><td>" + fmt(r.acc, 3) + "</td><td>" + r.acc_rank + "</td><td>" + dCell +
        "</td><td>" + r.n_comparisons + "</td><td>" + fmt(r.win_rate) +
        '</td><td class="rank-thin">' + r.n_ties + "</td></tr>";
    }).join("") + "</tbody>";
}

/* Ranking view hides the gap-curve controls: none of them apply to it. */
function applyView() {
  const ranking = ctlView.value === "ranking";
  rankWrap.hidden = !ranking;
  plotDiv.closest("main").hidden = ranking;
  document.querySelectorAll(".gap-only").forEach(el => { el.hidden = ranking; });
  hintEl.hidden = ranking;
  if (ranking) { renderRanking(); } else { renderPlot(); renderJudgeMetrics(); }
}

/* --- wiring + boot ------------------------------------------------------- */
[ctlTrunc, ctlSubset, ctlSource, ctlColorby, ctlCurves, ctlXmetric].forEach(c =>
  c.addEventListener("change", () => { rescaleX(); applyView(); }));
function loadFailed(what, e) {
  hintEl.textContent = "";
  emptyNote.hidden = false;
  emptyNote.textContent = "Could not load " + what + " (" + e.message +
    "). Serve this directory over HTTP and rebuild the dataset.";
}

/* Load the selected dataset's index and redraw. Caches are dropped first: they
   key on stems and pair keys, which mean different files in another dataset. */
async function loadDataset(id) {
  DATASET = id;
  pairCache = new Map(); transCache = new Map();
  optsCache = new Map(); ansCache = new Map();
  INDEX = null;
  try {
    INDEX = await fetchJSON(dataURL("index.json"));
  } catch (e) {
    currentRunMap = [];
    Plotly.react(plotDiv, [], baseLayout(themeColors()), { displayModeBar: false, responsive: true });
    loadFailed("data/" + id + "/index.json", e);
    return;
  }
  // A dataset that ships xmetrics replaces the static list (quartile subsets of a
  // domain, say); one that does not gets the list back as index.html wrote it.
  ctlXmetric.innerHTML = (INDEX.xmetrics && INDEX.xmetrics.length)
    ? INDEX.xmetrics.map(m => '<option value="' + esc(m.key) + '">' + esc(m.label) + "</option>").join("")
    : XMETRIC_HTML;
  const available = INDEX.available_xmetrics;
  Array.from(ctlXmetric.options).forEach(option => {
    const supported = !available || available.includes(option.value);
    option.disabled = !supported;
    option.hidden = !supported;
  });
  if (INDEX.default_xmetric) ctlXmetric.value = INDEX.default_xmetric;
  const rankable = (INDEX.ranking || []).length > 0;
  ctlView.options[1].disabled = !rankable;
  if (!rankable) ctlView.value = "gap";
  rescaleX();
  applyView();
}

function rescaleX() {
  const xs = INDEX && INDEX.runs ? INDEX.runs.filter(r => !r.is_self).map(r => gapOf(r)) : [];
  XMAX = Math.max((xs.length ? Math.max.apply(null, xs) : 0.3) * 1.06, 0.05);
}

async function boot() {
  try {
    DATASETS = await fetchJSON("data/datasets.json");
  } catch (e) {
    loadFailed("data/datasets.json", e);
    return;
  }
  if (!DATASETS.length) { loadFailed("data/datasets.json", new Error("no datasets")); return; }
  ctlDataset.innerHTML = DATASETS.map(d =>
    '<option value="' + esc(d.id) + '">' + esc(d.label) + "</option>").join("");
  ctlDataset.value = DATASETS[0].id;
  ctlView.addEventListener("change", applyView);
  ctlDataset.addEventListener("change", () => {
    closeModal();
    loadDataset(ctlDataset.value);
  });
  await loadDataset(DATASETS[0].id);
}
boot();
