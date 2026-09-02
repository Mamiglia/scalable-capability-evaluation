/* End-to-end smoke test: loads index.html + app.js in jsdom against the exported data,
   switches datasets / views / x-metrics, opens the modal and renders the verdicts.
   jsdom is not a dependency of this repo — install it wherever you like and point at it:

     npm --prefix /tmp/jd install jsdom@22        # jsdom 30 needs node >= 22
     node smoke.js . /tmp/jd

   Exits non-zero on the first broken expectation. Re-run after editing app.js or the
   exporter: it is the only executable check on the viewer's data contract. */
const fs = require("fs"), path = require("path");
const JSDOM_AT = process.argv[3];
const { JSDOM } = require(JSDOM_AT ? JSDOM_AT + "/node_modules/jsdom" : "jsdom");

const ROOT = process.argv[2] || ".";
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8")
  .replace('<script src="plotly.min.js"></script>', "");
const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, url: "http://x/" });
const { window } = dom;

const calls = { react: 0, layouts: [] };
window.Plotly = {
  react: (div, traces, layout) => { calls.react++; calls.layouts.push(layout); calls.traces = traces;
                                    div.data = traces; return Promise.resolve(div); },
  restyle: () => {}, };
window.HTMLElement.prototype.on = function () {};
window.fetch = async url => {
  const file = path.join(ROOT, url.replace(/^https?:\/\/[^/]+\//, "").split("?")[0]);
  if (!fs.existsSync(file)) return { ok: false, status: 404 };
  return { ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(file, "utf8")) };
};
// strict-mode eval keeps its own scope: hand the internals out explicitly
window.eval(fs.readFileSync(path.join(ROOT, "app.js"), "utf8") +
  "\n;window.__app = { get INDEX() { return INDEX; }, openModal };");

const $ = id => window.document.getElementById(id);
const fire = el => el.dispatchEvent(new window.Event("change"));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const check = (cond, msg) => { if (!cond) { console.error("FAIL:", msg); process.exitCode = 1; } };

(async () => {
  await sleep(300);
  const menu = JSON.parse(fs.readFileSync(path.join(ROOT, "data/datasets.json"), "utf8"));
  check($("ctl-dataset").options.length === menu.length, menu.length + " datasets in the menu");
  check(calls.react > 0, "gap plot drawn");

  // (4) x-axis options come from the dataset, and switching one redraws with a new title
  const xopts = Array.from($("ctl-xmetric").options).map(o => o.value);
  check(xopts.join(",") === "q1,q2,q3,q4,q1-2,q2-3,q3-4,q1-3,q2-4,all", "x-metric options: " + xopts);
  check($("ctl-xmetric").value === "q4", "defaults to the ground-truth quartile");
  const before = calls.layouts[calls.layouts.length - 1].xaxis.title.text;
  $("ctl-xmetric").value = "q1-3"; fire($("ctl-xmetric"));
  const after = calls.layouts[calls.layouts.length - 1].xaxis.title.text;
  check(/Q1–Q3/.test(after) && before !== after, "x title follows the metric: " + after);
  check(/oriented by Q4/.test($("ctl-hint").textContent), "hint warns about orientation");
  const xs = calls.traces.flatMap(t => t.x || []).filter(v => typeof v === "number");
  check(xs.length && xs.every(v => v >= 0 && v <= 1), "gaps in range on the new metric");
  $("ctl-xmetric").value = "q4"; fire($("ctl-xmetric"));

  // (3) pooled judge metrics table
  const mrows = $("judge-metrics").querySelectorAll("tbody tr");
  check(mrows.length === 1, "one judge row");
  const cells = Array.from(mrows[0].querySelectorAll("td")).map(td => td.textContent);
  check(cells.length === 11 && !cells.some(c => /undefined|NaN/.test(c)), "metric cells: " + cells.join("|"));
  check(/Pooled over every cross-model judgement/.test($("judge-metrics-note").textContent), "metrics note");

  // BT ranking view still works
  $("ctl-view").value = "ranking"; fire($("ctl-view"));
  const nRanked = window.__app.INDEX.ranking.length;
  check(!$("rank-wrap").hidden && nRanked > 0 &&
        $("rank-table").querySelectorAll("tbody tr").length === nRanked, "ranking view: " + nRanked + " models");
  check(Array.from(window.document.querySelectorAll(".gap-only")).every(el => el.hidden), "gap controls hidden");
  check(!$("ctl-xmetric").closest(".ctl").hidden, "x-metric control stays in the ranking view");

  // the x-metric drives the ranking view too: axis, accuracies, ranks and tau all move
  const rankRow = () => Array.from($("rank-table").querySelectorAll("tbody tr")[3].querySelectorAll("td"))
                             .map(td => td.textContent);
  const q4row = rankRow(), q4title = calls.layouts[calls.layouts.length - 1].xaxis.title.text;
  const q4caption = $("rank-caption").textContent;
  $("ctl-xmetric").value = "q1"; fire($("ctl-xmetric"));
  const q1row = rankRow(), q1title = calls.layouts[calls.layouts.length - 1].xaxis.title.text;
  check(/^Q1 accuracy/.test(q1title) && q1title !== q4title, "ranking x title follows the metric: " + q1title);
  check(q1row[3] !== q4row[3], "accuracy column re-read: " + q4row[3] + " -> " + q1row[3]);
  check($("rank-caption").textContent !== q4caption, "tau recomputed for the plotted axis");
  check(/Kendall τ vs Q1 accuracy/.test($("rank-caption").textContent),
        "caption names the axis: " + $("rank-caption").textContent.slice(0, 90));
  check(!/undefined|NaN/.test($("rank-table").innerHTML), "no holes in the retitled table");

  // The caption's tau is the one the exporter computed for that subset. It tracks
  // ranking.scoring's (scores.csv) without matching it: scores.csv means accuracy over
  // response rows, the axis averages repeats per item first.
  const scored = { q1: "Q1", q2: "Q2", q3: "Q3", q4: "Q4", all: "all" };
  for (const [key, quartile] of Object.entries(scored)) {
    $("ctl-xmetric").value = key; fire($("ctl-xmetric"));
    const shown = Number(/Kendall τ vs .*? accuracy \(the plotted axis\): (-?\d+\.\d+)/
                         .exec($("rank-caption").textContent)[1]);
    const baked = window.__app.INDEX.taus[key];
    check(Math.abs(shown - baked) < 0.0005, `tau on ${key}: caption ${shown} vs index.json ${baked}`);
    const scoresRow = window.__app.INDEX.scores.find(s => s.quartile === quartile).correlation;
    check(Math.abs(shown - scoresRow) < 0.1,
          `tau on ${key} is in the same country as scores.csv: ${shown} vs ${scoresRow.toFixed(3)}`);
  }
  $("ctl-xmetric").value = "q4"; fire($("ctl-xmetric"));
  $("ctl-view").value = "gap"; fire($("ctl-view"));
  check($("rank-wrap").hidden, "back to the gap view");

  // (2) modal: transcripts, verdicts, and the judge's reasoning expander
  const run = window.__app.INDEX.runs.find(r => r.n_items > 5);
  await window.__app.openModal(run);
  await sleep(600);
  check(!$("modal").hidden, "modal open");
  check(/position bias/.test($("modal-meta").textContent) && /length corr/.test($("modal-meta").textContent),
        "per-run metrics in the header: " + $("modal-meta").textContent.slice(0, 160));
  const items = $("item-list").querySelectorAll("li");
  check(items.length > 0, "item list populated");
  items[0].dispatchEvent(new window.Event("click"));
  await sleep(200);
  check($("colA-text").textContent.length > 50 && $("colB-text").textContent.length > 50, "both transcripts shown");
  const det = $("verdict-list").querySelectorAll("details.vreason");
  check(det.length > 0, "judge reasoning expander present");
  check(det.length > 0 && det[0].querySelector("pre").textContent.length > 200,
        "reasoning has content (" + (det[0] && det[0].querySelector("pre").textContent.length) + " chars)");
  check(det.length > 0 && /judge reasoning — [\d,]+ chars/.test(det[0].querySelector("summary").textContent),
        "expander summary: " + (det[0] && det[0].querySelector("summary").textContent));

  // (1) light only
  check(!/prefers-color-scheme/.test(fs.readFileSync(path.join(ROOT, "style.css"), "utf8")), "no dark CSS");
  check(calls.layouts.every(l => l.paper_bgcolor === "#ffffff"), "plots painted light");

  // dataset switch keeps working
  const other = menu.find(d => d.id !== $("ctl-dataset").value);
  $("ctl-dataset").value = other.id; fire($("ctl-dataset"));
  await sleep(400);
  check(window.__app.INDEX.source && $("ctl-dataset").value === other.id, "switched to " + other.id);
  check(/Q\d/.test(other.label), "labels name the condition: " + other.label);
  console.log(process.exitCode ? "SMOKE TEST FAILED" : "smoke test passed");
})();
