/* Judge-preference attribution dashboard (ML per-layer directions).
   Static: reads ./data_ml/{sentences,transcripts,meta}.json, no server code.
   Occlusion sign: value > 0 => sentence drives the direction UP, < 0 => DOWN. */
(function () {
  "use strict";

  const css = (n) => getComputedStyle(document.body).getPropertyValue(n).trim();
  const fmt = (v) => (v >= 0 ? "+" : "") + Number(v).toFixed(3);
  const $ = (id) => document.getElementById(id);
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const TOP_N = 25;
  const state = { rows: [], tx: null, meta: null, dir: "v_pref_gen", adj: "len",
                  sort: { key: "raw", asc: false }, tableFilter: "",
                  tsne: [], tsneColor: "dir",
                  clusterCount: 0, clusterLabels: {}, pal: null };

  // per-sentence value on the selected direction + adjustment
  const val = (r) => r[`${state.dir}_${state.adj}`];
  const roleOf = (key) => (state.tx[key] || {}).model || (state.tx[key] || {}).role || "?";
  const eloOf = (key) => (state.tx[key] || {}).elo;
  const dirLabel = () => `${state.dir} (L${state.meta.layers[state.dir]})`;

  function percentile(arr, p) {
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(p * s.length))] || 0;
  }
  function mix(c1, c2, t) {
    const p = (c) => { const h = c.replace("#", "");
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
    const [a, b] = [p(c1), p(c2)];
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
  }
  function shade(v, lim) {                 // diverging: neg=blue lowers, pos=red raises
    const t = Math.max(-1, Math.min(1, v / (lim || 1)));
    const stops = t < 0
      ? [css("--div-zero"), css("--div-neg-1"), css("--div-neg-2")]
      : [css("--div-zero"), css("--div-pos-1"), css("--div-pos-2")];
    const a = Math.abs(t);
    return a < 0.5 ? mix(stops[0], stops[1], a / 0.5) : mix(stops[1], stops[2], (a - 0.5) / 0.5);
  }

  /* ---- Drivers: two ranked lists across all transcripts ---- */
  function drawDrivers() {
    const long = state.rows.filter((r) => r.n_chars >= state.meta.min_chars);
    const sorted = [...long].sort((a, b) => val(b) - val(a));
    const up = sorted.slice(0, TOP_N);
    const down = sorted.slice(-TOP_N).reverse();
    $("drv-dir-up").innerHTML = `<span class="dir-badge">${escapeHtml(dirLabel())}</span>`;
    $("drv-dir-down").innerHTML = `<span class="dir-badge">${escapeHtml(dirLabel())}</span>`;
    $("drivers-up").innerHTML = up.map((r) => driverRow(r, "up")).join("");
    $("drivers-down").innerHTML = down.map((r) => driverRow(r, "down")).join("");
    $("drivers-note").textContent =
      `Top ${TOP_N} of ${long.length} sentences (${state.rows.length - long.length}` +
      ` shorter than ${state.meta.min_chars} chars hidden) by ${state.adj} occlusion effect` +
      ` on ${dirLabel()}. Click a row to open its transcript.`;
    document.querySelectorAll(".driver-list li").forEach((li) =>
      li.addEventListener("click", () => openTranscript(li.dataset.key, +li.dataset.pos)));
  }
  function driverRow(r, cls) {
    return `<li data-key="${escapeHtml(r.key)}" data-pos="${r.pos}">
      <span class="val ${cls}">${fmt(val(r))}</span>
      <span class="sent">${escapeHtml(r.sentence.slice(0, 220))}</span>
      <span class="sub">${escapeHtml(roleOf(r.key))} · elo ${eloOf(r.key)} · ${r.n_chars} chars</span>
    </li>`;
  }

  /* ---- Transcript: sentences (or tokens) shaded by the selected direction ---- */
  function openTranscript(key, pos) {
    switchView("transcript");
    $("ctl-transcript").value = key;
    drawTranscript(pos);
  }
  function drawTranscript(highlightPos) {
    const key = $("ctl-transcript").value;
    const meta = state.tx[key] || {};
    const byTokens = $("ctl-tokens").checked;
    const header = `<p class="note">${escapeHtml(meta.role || "")} · item ${escapeHtml(String(meta.item))}` +
      ` · elo ${meta.elo} · ${meta.n_sentences} sentences · ` +
      (byTokens ? `tokens shaded by per-token projection z·v̂ on ${escapeHtml(dirLabel())}`
                : `sentences shaded by ${state.adj} occlusion effect on ${escapeHtml(dirLabel())}`) + `</p>`;
    if (byTokens) { $("transcript-body").innerHTML = header + tokenHtml(key); return; }
    const rows = state.rows.filter((r) => r.key === key).sort((a, b) => a.pos - b.pos);
    const lim = Math.max(1e-4, percentile(rows.map((r) => Math.abs(val(r))), 0.95));
    $("transcript-body").innerHTML = header + rows.map((r) =>
      `<span class="s${r.pos === highlightPos ? " here" : ""}" style="background:${shade(val(r), lim)}"
        title="${escapeHtml(dirLabel())} ${state.adj} ${fmt(val(r))} · ${r.n_chars} chars">${escapeHtml(r.sentence)}</span>`)
      .join(" ");
    if (highlightPos != null) document.querySelector(".s.here")?.scrollIntoView({ block: "center" });
  }
  function tokenHtml(key) {
    const t = state.tx[key] || {};
    const proj = (t.token_proj || {})[state.dir] || [];
    const toks = t.tokens || [];
    if (!proj.length) return `<p class="note">no per-token projections in this map.</p>`;
    const lim = Math.max(1e-4, percentile(proj.map(Math.abs), 0.98));
    return toks.map((tok, i) => {
      const clean = escapeHtml(tok.replace(/Ġ/g, " ").replace(/Ċ/g, "\n").replace(/▁/g, " "));
      return `<span class="tok" style="background:${shade(proj[i] || 0, lim)}"
        title="proj ${fmt(proj[i] || 0)}">${clean}</span>`;
    }).join("");
  }

  /* ---- Scatter: one point per transcript, x/y = last-token scores, colour = elo ---- */
  function drawScatter() {
    const xk = $("ctl-x").value, yk = $("ctl-y").value;
    const keys = Object.keys(state.tx);
    const elos = keys.map((k) => state.tx[k].elo);
    const lim = Math.max(0.01, percentile(elos.map(Math.abs), 0.98));
    Plotly.react("plot-scatter", [{
      x: keys.map((k) => state.tx[k].last[xk]), y: keys.map((k) => state.tx[k].last[yk]),
      customdata: keys,
      text: keys.map((k) => `${state.tx[k].model || state.tx[k].role} · elo ${state.tx[k].elo}`),
      mode: "markers", type: "scatter",
      marker: {
        size: 11, opacity: 0.9, color: elos, cmin: -lim, cmax: lim,
        colorscale: [[0, css("--div-neg-2")], [0.5, css("--div-zero")], [1, css("--div-pos-2")]],
        colorbar: { title: { text: "elo", side: "right" }, thickness: 10, outlinewidth: 0,
                    tickfont: { color: css("--text-secondary") } },
        line: { width: 0.5, color: css("--grid") },
      },
      hovertemplate: `%{text}<br>${xk} (L${state.meta.layers[xk]}) %{x:+.3f}` +
                     `<br>${yk} (L${state.meta.layers[yk]}) %{y:+.3f}<extra></extra>`,
    }], scatterLayout(`${xk} (L${state.meta.layers[xk]})`, `${yk} (L${state.meta.layers[yk]})`),
      { displayModeBar: false, responsive: true });
    const gd = $("plot-scatter");
    gd.removeAllListeners && gd.removeAllListeners("plotly_click");   // don't stack on redraw
    gd.on("plotly_click", (ev) => openTranscript(ev.points[0].customdata, null));
  }
  function scatterLayout(xt, yt) {
    const ink = css("--text-secondary"), grid = css("--grid");
    return {
      margin: { l: 60, r: 10, t: 10, b: 50 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: ink, size: 12 },
      xaxis: { title: xt, gridcolor: grid, zeroline: true, zerolinecolor: ink },
      yaxis: { title: yt, gridcolor: grid, zeroline: true, zerolinecolor: ink },
      hoverlabel: { align: "left" },
    };
  }

  /* ---- Table ---- */
  function sortKey(k) { return k === "raw" || k === "len" ? `${state.dir}_${k}` : k; }
  function cell(r, k) {
    if (k === "role") return roleOf(r.key);
    if (k === "elo") return eloOf(r.key);
    if (k === "n_chars" || k === "sentence") return r[k];
    return r[sortKey(k)];
  }
  function drawTable() {
    const { key, asc } = state.sort;
    const f = state.tableFilter.toLowerCase();
    const base = f ? state.rows.filter((r) => r.sentence.toLowerCase().includes(f)) : state.rows;
    const sorted = [...base].sort((a, b) => {
      const va = cell(a, key), vb = cell(b, key);
      const c = typeof va === "string" ? va.localeCompare(vb) : va - vb;
      return asc ? c : -c;
    }).slice(0, 500);
    $("th-raw").textContent = `${state.dir} raw`;
    $("th-len").textContent = `${state.dir} len-adj`;
    $("table-body").innerHTML = sorted.map((r) =>
      `<tr><td class="num">${fmt(r[`${state.dir}_raw`])}</td>
       <td class="num">${fmt(r[`${state.dir}_len`])}</td>
       <td class="num">${r.n_chars}</td>
       <td>${escapeHtml(roleOf(r.key))}</td>
       <td class="num">${eloOf(r.key)}</td>
       <td>${escapeHtml(r.sentence.slice(0, 400))}</td></tr>`).join("");
    $("table-note").innerHTML =
      (f ? `filtered to &ldquo;${escapeHtml(state.tableFilter)}&rdquo; · ` : "") +
      `showing ${sorted.length} of ${base.length} sentences (top 500 by current sort), direction ${escapeHtml(dirLabel())}` +
      (f ? ` · <a href="#" id="clear-filter">clear filter</a>` : "");
    if (f) $("clear-filter").addEventListener("click", (e) => {
      e.preventDefault(); state.tableFilter = ""; drawTable(); });
    document.querySelectorAll("th.sortable").forEach((th) => {
      th.classList.toggle("sorted", th.dataset.sort === key);
      th.classList.toggle("asc", th.dataset.sort === key && asc);
    });
  }

  /* ---- t-SNE: embedding map, coloured by direction delta or cluster ---- */
  function clusterPalette() {                 // golden-angle HSL, distinct per cluster
    if (state.pal) return state.pal;
    const pal = {};
    for (let i = 0; i < state.clusterCount; i++)
      pal[i] = `hsl(${(i * 137.508) % 360},62%,${i % 2 ? 58 : 46}%)`;
    return (state.pal = pal);
  }
  function tsneLayout() {
    const ink = css("--text-secondary");
    return {
      margin: { l: 8, r: 8, t: 8, b: 8 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: ink, size: 12 }, hoverlabel: { align: "left" },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false },
    };
  }
  function drawTsne() {
    const P = state.tsne;
    if (!P.length) { $("tsne-hint").textContent = "no t-SNE data — run the extras exporter."; return; }
    const byCluster = state.tsneColor === "cluster";
    let marker, custom, hover;
    if (byCluster) {
      const pal = clusterPalette();
      marker = { size: 7, opacity: 0.85, color: P.map((p) => pal[p.cluster_id]),
                 line: { width: 0.3, color: css("--grid") } };
      custom = P.map((p) => [p.sentence.slice(0, 140), p.cluster_label]);
      hover = "<b>%{customdata[1]}</b><br>%{customdata[0]}<extra></extra>";
      drawClusterLegend(pal);
      $("tsne-hint").textContent =
        `${P.length} sentences, coloured by k-means cluster (${state.clusterCount} clusters).` +
        ` Layout: item-centred embeddings (per-item mean removed, topic stripped);` +
        ` clusters were fit on raw embeddings, so they may scatter here.`;
    } else {
      const key = `${state.dir}_len`, vals = P.map((p) => p[key]);
      const lim = Math.max(1e-4, percentile(vals.map(Math.abs), 0.95));
      marker = { size: 7, opacity: 0.85, color: vals, cmin: -lim, cmax: lim,
        colorscale: [[0, css("--div-neg-2")], [0.5, css("--div-zero")], [1, css("--div-pos-2")]],
        colorbar: { title: { text: `${state.dir}<br>len-adj`, side: "right" }, thickness: 10,
                    outlinewidth: 0, tickfont: { color: css("--text-secondary") } },
        line: { width: 0.3, color: css("--grid") } };
      custom = P.map((p) => [p.sentence.slice(0, 140), fmt(p[key])]);
      hover = `${state.dir} len-adj %{customdata[1]}<br>%{customdata[0]}<extra></extra>`;
      $("tsne-legend").hidden = true;
      $("tsne-hint").textContent = `${P.length} sentences, coloured by ${dirLabel()} length-adjusted` +
        ` occlusion (blue lowers · red raises, symmetric about 0, robust 95th-pct clip).` +
        ` Layout: item-centred embeddings (per-item mean removed, topic stripped).`;
    }
    Plotly.react("plot-tsne", [{ x: P.map((p) => p.x), y: P.map((p) => p.y), customdata: custom,
      mode: "markers", type: "scattergl", marker, hovertemplate: hover }],
      tsneLayout(), { displayModeBar: false, responsive: true });
  }
  function drawClusterLegend(pal) {
    const el = $("tsne-legend"), labels = state.clusterLabels;
    el.innerHTML = Object.keys(labels).sort((a, b) => a - b).map((id) =>
      `<span class="cl-item"><span class="cl-dot" style="background:${pal[id]}"></span>` +
      `${escapeHtml(labels[id])}</span>`).join("");
    el.hidden = false;
  }

  /* ---- By transcript: each transcript's highest and lowest sentence ---- */
  function drawByItem() {
    const long = state.rows.filter((r) => r.n_chars >= state.meta.min_chars);
    const byKey = {};
    long.forEach((r) => (byKey[r.key] = byKey[r.key] || []).push(r));
    const keys = Object.keys(byKey).sort((a, b) => eloOf(b) - eloOf(a));
    $("byitem-body").innerHTML = keys.map((k) => {
      const rs = byKey[k].slice().sort((a, b) => val(b) - val(a));
      const hi = rs[0], lo = rs[rs.length - 1], t = state.tx[k] || {};
      return `<div class="bt-card" data-key="${escapeHtml(k)}">
        <div class="bt-head">${escapeHtml(roleOf(k))} &middot; elo ${eloOf(k)} &middot; item ${escapeHtml(String(t.item))}</div>
        <div class="bt-row"><span class="val up">${fmt(val(hi))}</span>
          <span class="sent" data-pos="${hi.pos}">${escapeHtml(hi.sentence.slice(0, 240))}</span></div>
        <div class="bt-row"><span class="val down">${fmt(val(lo))}</span>
          <span class="sent" data-pos="${lo.pos}">${escapeHtml(lo.sentence.slice(0, 240))}</span></div>
      </div>`;
    }).join("");
    $("byitem-note").textContent =
      `Each transcript's single highest (raises) and lowest (lowers) sentence on ${dirLabel()}` +
      ` (${state.adj}), within that transcript — topic-controlled. Sorted by elo. Click a sentence to open it.`;
    document.querySelectorAll("#view-byitem .bt-row .sent").forEach((el) =>
      el.addEventListener("click", () =>
        openTranscript(el.closest(".bt-card").dataset.key, +el.dataset.pos)));
  }

  /* ---- view plumbing ---- */
  function switchView(name) {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.view === name));
    document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== `view-${name}`; });
    if (name === "scatter") drawScatter();
    if (name === "tsne") drawTsne();
    if (name === "byitem") drawByItem();
  }
  function redraw() {
    $("dir-note").textContent =
      `${dirLabel()} — occlusion Δ = full-transcript last-token score minus score with the sentence removed.`;
    drawDrivers(); drawByItem(); drawTranscript(); drawTable();
    if (!$("view-scatter").hidden) drawScatter();
    if (!$("view-tsne").hidden) drawTsne();
  }

  async function init() {
    const base = "data_ml/";
    const get = (f) => fetch(base + f).then((r) => r.json());
    const opt = (f) => get(f).catch(() => null);   // extras may be absent pre-export
    const [rows, tx, meta, tsne] = await Promise.all(
      [get("sentences.json"), get("transcripts.json"), get("meta.json"),
       opt("tsne.json")]);
    state.rows = rows; state.tx = tx; state.meta = meta;
    state.tsne = tsne || [];
    state.clusterCount = state.tsne.reduce((m, p) => Math.max(m, p.cluster_id + 1), 0);
    state.tsne.forEach((p) => { state.clusterLabels[p.cluster_id] = p.cluster_label; });
    if (!meta.directions.includes(state.dir)) state.dir = meta.directions[0];

    $("ctl-dir").innerHTML = meta.directions.map((d) =>
      `<option value="${d}">${d} (L${meta.layers[d]})</option>`).join("");
    $("ctl-dir").value = state.dir;
    ["ctl-x", "ctl-y"].forEach((id) => { $(id).innerHTML = meta.directions.map((d) =>
      `<option value="${d}">${d} (L${meta.layers[d]})</option>`).join(""); });
    $("ctl-x").value = "v_cap"; $("ctl-y").value = state.dir;
    Object.entries(tx).sort((a, b) => b[1].elo - a[1].elo).forEach(([k, v]) =>
      $("ctl-transcript").add(new Option(`${v.model || v.role || "?"} · elo ${v.elo} · item ${v.item}`, k)));

    $("ctl-dir").addEventListener("change", (e) => { state.dir = e.target.value; redraw(); });
    $("ctl-adj").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      state.adj = b.dataset.adj;
      document.querySelectorAll("#ctl-adj button").forEach((x) => x.classList.toggle("active", x === b));
      redraw();
    });
    $("ctl-tsne-color").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      state.tsneColor = b.dataset.color;
      document.querySelectorAll("#ctl-tsne-color button").forEach((x) => x.classList.toggle("active", x === b));
      drawTsne();
    });
    document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
    $("ctl-transcript").addEventListener("change", () => drawTranscript());
    $("ctl-tokens").addEventListener("change", () => drawTranscript());
    ["ctl-x", "ctl-y"].forEach((id) => $(id).addEventListener("change", drawScatter));
    document.querySelectorAll("th.sortable").forEach((th) =>
      th.addEventListener("click", () => {
        const k = th.dataset.sort;
        state.sort = { key: k, asc: state.sort.key === k ? !state.sort.asc : false };
        drawTable();
      }));
    redraw();
  }

  init().catch((err) => {
    document.querySelector("main").innerHTML =
      `<p class="note">Could not load data: ${escapeHtml(err.message)}. Run the exporter, then serve the folder:
       <code>python3 -m http.server 8766 --directory interactive_viewer</code></p>`;
  });
})();
