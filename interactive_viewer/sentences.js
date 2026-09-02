/* Sentence attribution viewer: taste-vs-capability scatter, lexical map,
   sortable table, and a transcript shaded by loading. Static: reads the
   exported json, no server code. */
(function () {
  "use strict";

  const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
  const state = { sentences: [], transcripts: null, meta: null,
                  sort: { key: "v_pref_judge_res", asc: false }, filtered: [] };

  const fmt = (v) => (v >= 0 ? "+" : "") + Number(v).toFixed(2);

  function filtered() {
    const q = document.getElementById("ctl-search").value.trim().toLowerCase();
    return q ? state.sentences.filter((s) => s.sentence.toLowerCase().includes(q))
             : state.sentences;
  }

  /* One trace per model, so plotly's own legend gives click-to-hide and
     double-click-to-isolate - the same handling as the transcript viewer.
     Seven hues cannot all clear the colour-blind separation gate on a
     scatter, so identity is carried by the legend and the hover text too,
     never by colour alone. */
  const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4",
                  "--series-5", "--series-6", "--series-7"];

  function drawScatter(rows) {
    const xk = document.getElementById("ctl-x").value;
    const yk = document.getElementById("ctl-y").value;
    const models = state.meta.models;
    const traces = models.map((m, i) => {
      const sub = rows.filter((r) => r.model === m);
      return {
        name: m,
        x: sub.map((r) => r[xk]), y: sub.map((r) => r[yk]),
        text: sub.map((r) => r.sentence.slice(0, 110)),
        customdata: sub.map((r) => state.index.get(r)),
        mode: "markers", type: "scattergl",
        marker: { size: 5, opacity: 0.7, color: css(SERIES[i % SERIES.length]),
                  line: { width: 0 } },
        hovertemplate: `<b>${m}</b><br>${xk.replace("_res", "")} %{x:+.2f}` +
                       `<br>${yk.replace("_res", "")} %{y:+.2f}<br>%{text}<extra></extra>`,
      };
    });
    const lay = layout(xk.replace("_res", ""), yk.replace("_res", ""), true);
    lay.showlegend = true;
    lay.legend = { orientation: "h", y: -0.18, font: { size: 11 } };
    lay.margin.b = 90;
    Plotly.react("plot-scatter", traces, lay, config());
  }

  function drawTsne(rows) {
    const key = document.getElementById("ctl-colour").value;
    state.tsneRows = rows;
    const vals = rows.map((r) => r[key]);
    const lim = Math.max(0.01, percentile(vals.map(Math.abs), 0.98));
    Plotly.react("plot-tsne", [{
      x: rows.map((r) => r.tx), y: rows.map((r) => r.ty),
      text: rows.map((r) => r.sentence.slice(0, 110)),
      customdata: rows.map((r) => state.index.get(r)),
      mode: "markers", type: "scattergl",
      marker: {
        size: 5, opacity: 0.75, color: vals, cmin: -lim, cmax: lim,
        colorscale: [[0, css("--div-neg-2")], [0.25, css("--div-neg-1")],
                     [0.5, css("--div-zero")], [0.75, css("--div-pos-1")],
                     [1, css("--div-pos-2")]],
        colorbar: { title: { text: key.replace("_res", ""), side: "right" },
                    thickness: 10, outlinewidth: 0,
                    tickfont: { color: css("--text-secondary") },
                    titlefont: { color: css("--text-secondary") } },
        line: { width: 0 },
      },
      hovertemplate: "%{text}<extra></extra>",
    }], layout("", "", false), config());
  }

  function layout(xtitle, ytitle, zeroline) {
    const ink = css("--text-secondary"), grid = css("--grid");
    return {
      margin: { l: 55, r: 10, t: 10, b: 45 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: ink, size: 12 },
      xaxis: { title: xtitle, gridcolor: grid, zeroline: zeroline,
               zerolinecolor: ink, zerolinewidth: 1, showticklabels: !!xtitle },
      yaxis: { title: ytitle, gridcolor: grid, zeroline: zeroline,
               zerolinecolor: ink, zerolinewidth: 1, showticklabels: !!ytitle },
      hoverlabel: { align: "left" }, showlegend: false, dragmode: "pan",
    };
  }
  const config = () => ({ displayModeBar: false, responsive: true, scrollZoom: true });

  function percentile(arr, p) {
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(p * s.length))] || 0;
  }

  function showDetail(row) {
    const el = document.getElementById("detail");
    el.hidden = false;
    el.innerHTML =
      `<div class="meta">v_pref_judge <strong>${fmt(row.v_pref_judge_res)}</strong> ·
       v_cap <strong>${fmt(row.v_cap_res)}</strong> ·
       ${escapeHtml(row.model)} · elo ${row.elo} · ${row.n_chars} chars ·
       <a href="#" data-key="${escapeHtml(row.key)}" class="to-transcript">open transcript</a></div>
       <div class="sent">${escapeHtml(row.sentence)}</div>`;
    el.querySelector(".to-transcript").addEventListener("click", (e) => {
      e.preventDefault();
      switchView("transcript");
      document.getElementById("ctl-transcript").value = row.key;
      drawTranscript();
    });
  }

  const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function drawTable(rows) {
    const { key, asc } = state.sort;
    const sorted = [...rows].sort((a, b) => {
      const va = a[key], vb = b[key];
      const c = typeof va === "string" ? va.localeCompare(vb) : va - vb;
      return asc ? c : -c;
    }).slice(0, 500);
    document.getElementById("table-body").innerHTML = sorted.map((r) =>
      `<tr><td class="num">${fmt(r.v_pref_judge_res)}</td>
       <td class="num">${fmt(r.v_cap_res)}</td>
       <td class="num">${r.n_chars}</td>
       <td>${escapeHtml(r.model)}</td>
       <td>${escapeHtml(r.sentence.slice(0, 400))}</td></tr>`).join("");
    document.getElementById("table-note").textContent =
      `showing ${sorted.length} of ${rows.length} sentences (top 500 by current sort)`;
    document.querySelectorAll("th.sortable").forEach((th) => {
      th.classList.toggle("sorted", th.dataset.sort === key);
      th.classList.toggle("asc", th.dataset.sort === key && asc);
    });
  }

  function shade(v, lim) {
    const t = Math.max(-1, Math.min(1, v / lim));
    const stops = t < 0
      ? [css("--div-zero"), css("--div-neg-1"), css("--div-neg-2")]
      : [css("--div-zero"), css("--div-pos-1"), css("--div-pos-2")];
    const a = Math.abs(t);
    return a < 0.5 ? mix(stops[0], stops[1], a / 0.5) : mix(stops[1], stops[2], (a - 0.5) / 0.5);
  }
  function mix(c1, c2, t) {
    const p = (c) => { const h = c.replace("#", "");
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
    const [a, b] = [p(c1), p(c2)];
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
  }

  function drawTranscript() {
    const key = document.getElementById("ctl-transcript").value;
    const axis = document.getElementById("ctl-taxis").value;
    const rows = state.transcripts.sentences.filter((s) => s.key === key)
      .sort((a, b) => a.pos - b.pos);
    const lim = Math.max(0.01, percentile(rows.map((r) => Math.abs(r[axis])), 0.95));
    const meta = state.transcripts.meta[key] || {};
    document.getElementById("transcript-body").innerHTML =
      `<p class="note">${escapeHtml(meta.model || "")} · elo ${meta.elo} ·
        ${rows.length} sentences · shaded by ${escapeHtml(axis.replace("_res", ""))}</p>` +
      rows.map((r) =>
        `<span class="s" style="background:${shade(r[axis], lim)}"
           title="${escapeHtml(axis.replace("_res", ""))} ${fmt(r[axis])}">${escapeHtml(r.sentence)}</span>`)
        .join(" ");
  }

  function switchView(name) {
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("is-active", t.dataset.view === name));
    document.querySelectorAll(".view").forEach((v) =>
      { v.hidden = v.id !== `view-${name}`; });
    if (name === "home") { redraw(); }
  }

  function redraw() {
    state.filtered = filtered();
    state.index = new Map(state.filtered.map((r, i) => [r, i]));
    document.getElementById("ctl-n").textContent =
      `${state.filtered.length} of ${state.sentences.length} sentences`;
    drawScatter(state.filtered);
    drawTsne(state.filtered);
    drawTable(state.filtered);
  }

  async function init() {
    const base = "data/sentences/";
    const [sentences, transcripts, meta] = await Promise.all(
      ["sentences.json", "transcripts.json", "meta.json"].map((f) =>
        fetch(base + f).then((r) => r.json())));
    state.sentences = sentences; state.transcripts = transcripts; state.meta = meta;

    // axis pickers get every direction the maps recorded
    ["ctl-x", "ctl-y", "ctl-colour", "ctl-taxis"].forEach((id) => {
      const sel = document.getElementById(id);
      meta.axes.forEach((a) => sel.add(new Option(a, `${a}_res`)));
    });
    document.getElementById("ctl-x").value = "v_cap_res";
    document.getElementById("ctl-y").value = "v_pref_judge_res";
    document.getElementById("ctl-colour").value = "v_pref_judge_res";
    document.getElementById("ctl-taxis").value = "v_pref_judge_res";
    const tsel = document.getElementById("ctl-transcript");
    Object.entries(transcripts.meta)
      .sort((a, b) => b[1].elo - a[1].elo)
      .forEach(([k, v]) => tsel.add(new Option(`${v.model} · elo ${v.elo} · ${k.slice(0, 8)}`, k)));

    document.querySelectorAll(".tab").forEach((t) =>
      t.addEventListener("click", () => switchView(t.dataset.view)));
    ["ctl-x", "ctl-y", "ctl-colour"].forEach((id) =>
      document.getElementById(id).addEventListener("change", redraw));
    document.getElementById("ctl-search").addEventListener("input", redraw);
    document.getElementById("ctl-transcript").addEventListener("change", drawTranscript);
    document.getElementById("ctl-taxis").addEventListener("change", drawTranscript);
    document.querySelectorAll("th.sortable").forEach((th) =>
      th.addEventListener("click", () => {
        const k = th.dataset.sort;
        state.sort = { key: k, asc: state.sort.key === k ? !state.sort.asc : false };
        drawTable(state.filtered);
      }));
    ["plot-scatter", "plot-tsne"].forEach((id) =>
      document.getElementById(id).on = null);

    redraw();
    drawTranscript();
    ["plot-scatter", "plot-tsne"].forEach((id) => {
      document.getElementById(id).on("plotly_click", (ev) => {
        showDetail(state.filtered[ev.points[0].customdata]);
      });
    });
  }

  init().catch((err) => {
    document.querySelector("main").innerHTML =
      `<p class="note">Could not load data: ${escapeHtml(err.message)}.
       Serve the folder over HTTP: <code>python3 -m http.server 8765 --directory interactive_viewer</code></p>`;
  });
})();
