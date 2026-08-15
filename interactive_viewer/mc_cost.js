"use strict";

const byId = (id) => document.getElementById(id);
const plotConfig = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"]};
const autoKeys = ["judge_pool", "anchor_pool", "comparator_pool", "J", "M", "N", "attempted_items", "presentations", "strength_filter"];
const identityKeys = ["judge", "m1", "m2", "judge_reasoning_class", "m1_reasoning_class", "m2_reasoning_class"];
const tokenMetrics = {
  candidate_input_tokens: "input", candidate_output_tokens: "output",
  candidate_ordinary_output_tokens: "ordinary", candidate_reasoning_tokens: "reasoning",
  judge_input_tokens: "input", judge_output_tokens: "output",
  judge_ordinary_output_tokens: "ordinary", judge_reasoning_tokens: "reasoning",
};
const state = {data: null};

function option(select, value, label = value) {
  const item = document.createElement("option");
  item.value = String(value);
  item.textContent = String(label);
  select.append(item);
}

function unique(rows, key) {
  return [...new Set(rows.map((row) => row[key]).filter((value) => value !== null && value !== undefined))]
    .sort((left, right) => String(left).localeCompare(String(right), undefined, {numeric: true}));
}

function number(value, digits = 2) {
  return value === null || value === undefined ? "—" : Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
}

function interval(row, currency = false) {
  if (!row) return "—";
  const format = currency
    ? (value) => value === null ? "—" : Number(value).toLocaleString(undefined, {style: "currency", currency: "USD", maximumFractionDigits: 4})
    : (value) => number(value);
  return [row.p5, row.p50, row.p95].map(format).join(" / ");
}

function plotTheme() {
  const styles = getComputedStyle(document.documentElement);
  return {
    paper_bgcolor: "transparent", plot_bgcolor: "transparent",
    font: {family: getComputedStyle(document.body).fontFamily, size: 12, color: styles.getPropertyValue("--ink").trim()},
    xaxis: {gridcolor: styles.getPropertyValue("--border").trim(), zerolinecolor: styles.getPropertyValue("--border").trim()},
    yaxis: {gridcolor: styles.getPropertyValue("--border").trim(), zerolinecolor: styles.getPropertyValue("--border").trim()},
    margin: {l: 52, r: 18, t: 16, b: 48}, showlegend: false, hovermode: "closest",
  };
}

function emptyPlot(id, message) {
  Plotly.react(id, [], {...plotTheme(), annotations: [{text: message, showarrow: false, font: {color: getComputedStyle(document.documentElement).getPropertyValue("--muted").trim()}}]}, plotConfig);
}

function bandPlot(id, rows, xKey, xTitle, yTitle) {
  const points = rows.filter((row) => row[xKey] !== null && row[xKey] !== undefined && row.p50 !== null)
    .sort((left, right) => Number(left[xKey]) - Number(right[xKey]));
  if (!points.length) return emptyPlot(id, "No exact exported rows match these controls.");
  const x = points.map((row) => row[xKey]);
  const theme = plotTheme();
  theme.xaxis.title = xTitle;
  theme.yaxis.title = yTitle;
  theme.yaxis.range = yTitle.includes("accuracy") ? [0, 1] : undefined;
  return Plotly.react(id, [
    {x, y: points.map((row) => row.p5), mode: "lines", line: {width: 0}, hoverinfo: "skip"},
    {x, y: points.map((row) => row.p95), mode: "lines", line: {width: 0}, fill: "tonexty", fillcolor: "rgba(37,111,191,.18)", hoverinfo: "skip"},
    {x, y: points.map((row) => row.p50), mode: "lines+markers", line: {color: "#256fbf", width: 2}, marker: {size: 7}, customdata: points.map((row) => [row.p5, row.p95]), hovertemplate: "%{x}<br>p50 %{y:.3f}<br>p5–p95 %{customdata[0]:.3f}–%{customdata[1]:.3f}<extra></extra>"},
  ], theme, plotConfig);
}

function automaticSelection() {
  return {
    judge_pool: byId("judge-pool").value,
    anchor_pool: byId("anchor-pool").value,
    comparator_pool: byId("comparator-pool").value,
    J: Number(byId("judge-count").value), M: Number(byId("anchor-count").value),
    N: Number(byId("comparator-count").value), attempted_items: Number(byId("attempted-items").value),
    presentations: Number(byId("presentations").value), strength_filter: byId("strength-filter").value === "true",
  };
}

function matchesAutomatic(row, selected, except = null) {
  return autoKeys.every((key) => key === except || row[key] === selected[key]);
}

function exactAutomaticRows() {
  const selected = automaticSelection();
  return state.data.automatic.filter((row) => matchesAutomatic(row, selected));
}

function catalogueModel(modelId) {
  return state.data.catalogue.find((row) => row.model_id === modelId);
}

function resolvedManualRow(row) {
  const identity = {};
  const missingClasses = [];
  for (const role of ["judge", "m1", "m2"]) {
    const model = row.querySelector(`[data-role="${role}"]`).value;
    const override = row.querySelector(`[data-class="${role}"]`).value;
    const reasoningClass = override || catalogueModel(model)?.reasoning_class;
    identity[role] = model;
    if (reasoningClass === "reasoning" || reasoningClass === "non_reasoning") {
      identity[`${role}_reasoning_class`] = reasoningClass;
    } else {
      missingClasses.push(role);
    }
  }
  return {identity: missingClasses.length ? null : identity, missingClasses};
}

function matchesIdentity(row, identity) {
  return identityKeys.every((key) => row[key] === identity[key]);
}

function selectedManualGroups() {
  return [...byId("manual-rows").rows].map((editorRow, index) => {
    const resolved = resolvedManualRow(editorRow);
    const models = Object.fromEntries(["judge", "m1", "m2"].map((role) => [role, editorRow.querySelector(`[data-role="${role}"]`).value]));
    return {
      index, ...resolved, models,
      rows: resolved.identity ? state.data.manual.filter((row) => matchesIdentity(row, resolved.identity)) : [],
    };
  });
}

function populateAutomaticControls() {
  const scenarios = state.data.automatic;
  const mappings = [
    ["judge-pool", "judge_pool"], ["anchor-pool", "anchor_pool"], ["comparator-pool", "comparator_pool"],
  ];
  for (const [id, key] of mappings) {
    const select = byId(id);
    select.replaceChildren();
    unique(scenarios, key).forEach((value) => option(select, value));
  }
  for (const [id, key] of [["judge-count", "J"], ["anchor-count", "M"], ["comparator-count", "N"], ["attempted-items", "attempted_items"], ["presentations", "presentations"]]) {
    const select = byId(id);
    select.replaceChildren();
    unique(scenarios, key).forEach((value) => option(select, value));
  }
  const strength = byId("strength-filter");
  strength.replaceChildren();
  const strengths = unique(scenarios, "strength_filter");
  strengths.forEach((value) => option(strength, value, value ? "Newer and stronger" : "Newer only"));
}

function classOptions(select, role, hasDefault, current = "") {
  select.replaceChildren();
  option(select, "", hasDefault ? "Catalogue default" : "Override required");
  option(select, "reasoning", "Reasoning override");
  option(select, "non_reasoning", "Non-reasoning override");
  select.value = current;
  select.setAttribute("aria-label", `${role.toUpperCase()} reasoning class`);
}

function addManualRow(initial = null) {
  const fragment = byId("manual-row-template").content.cloneNode(true);
  const row = fragment.querySelector("tr");
  for (const role of ["judge", "m1", "m2"]) {
    const select = row.querySelector(`[data-role="${role}"]`);
    select.setAttribute("aria-label", `${role.toUpperCase()} model`);
    state.data.catalogue.forEach((model) => option(select, model.model_id, model.display_name || model.model_id));
    if (initial?.[role]) select.value = initial[role];
    const classes = row.querySelector(`[data-class="${role}"]`);
    const defaultClass = catalogueModel(select.value)?.reasoning_class;
    const initialClass = initial?.[`${role}_reasoning_class`];
    classOptions(classes, role, Boolean(defaultClass), initialClass && initialClass !== defaultClass ? initialClass : "");
    select.addEventListener("change", () => {
      const current = classes.value;
      classOptions(classes, role, Boolean(catalogueModel(select.value)?.reasoning_class), current);
      render();
    });
    classes.addEventListener("change", render);
  }
  row.querySelector(".remove-row").addEventListener("click", () => { row.remove(); render(); });
  byId("manual-rows").append(row);
}

function metric(rows, name, predicate = () => true) {
  return rows.find((row) => row.metric === name && predicate(row));
}

function summaryItem(label, row, exact = false, currency = false) {
  const item = document.createElement("div");
  item.className = "summary-item";
  const name = document.createElement("span");
  name.className = "label";
  name.textContent = label;
  const value = document.createElement("span");
  value.className = "value";
  value.textContent = exact ? number(row?.value, 0) : row ? (currency ? interval(row, true).split(" / ")[1] : number(row.p50)) : "—";
  const detail = document.createElement("span");
  detail.className = "interval";
  detail.textContent = exact ? "exact pre-filter maximum" : row ? `p5–p95 ${currency ? interval(row, true) : `${number(row.p5)}–${number(row.p95)}`}` : "not exported";
  item.append(name, value, detail);
  return item;
}

function renderSummary(rows, manualGroups) {
  const summary = byId("run-summary");
  summary.replaceChildren();
  if (byId("mode").value === "manual") {
    summary.append(summaryItem("Manual rows", {value: manualGroups.length}, true));
    summary.append(summaryItem("Exact forecasts found", {value: manualGroups.filter((group) => group.rows.length).length}, true));
    summary.append(summaryItem("Rows needing an override or export", {value: manualGroups.filter((group) => !group.rows.length).length}, true));
    return;
  }
  summary.append(
    summaryItem("Usable items", metric(rows, "usable_items", (row) => row.comparison_id === null)),
    summaryItem("Judgment calls", metric(rows, "judgment_calls", (row) => row.comparison_id === null)),
    summaryItem("Maximum calls", metric(rows, "judgment_call_upper_bound"), true),
    summaryItem("Candidate calls", metric(rows, "candidate_calls")),
    summaryItem("Total cost", metric(rows, "cost", (row) => row.model === "__total__"), false, true),
  );
}

function renderBadges(autoRows, manualGroups) {
  const badges = byId("badges");
  badges.replaceChildren();
  const scenarioRows = byId("mode").value === "automatic"
    ? autoRows.slice(0, 1)
    : manualGroups.flatMap((group) => group.rows.slice(0, 1));
  const statuses = unique(scenarioRows, "scenario_prediction_status");
  for (const status of statuses.length ? statuses : ["predicted"]) {
    const badge = document.createElement("span");
    badge.className = `badge ${status === "observed" ? "good" : ""}`;
    badge.textContent = status === "observed" ? "Observed configuration" : "Posterior prediction";
    badges.append(badge);
  }
  if (scenarioRows.some((row) => row.scenario_extrapolation)) {
    const badge = document.createElement("span");
    badge.className = "badge warn";
    badge.textContent = "Extrapolation";
    badges.append(badge);
  }
}

function renderAccuracy(autoRows, manualGroups) {
  const rows = byId("mode").value === "automatic"
    ? autoRows.filter((row) => row.metric === "choice_accuracy" && row.comparison_id !== null)
    : manualGroups.flatMap((group) => group.rows.filter((row) => row.metric === "choice_accuracy" && row.comparison_id !== null));
  bandPlot("accuracy-plot", rows, "capability_gap", "M1 θ − M2 θ", "Higher-θ choice accuracy");
}

function renderScaling() {
  if (byId("mode").value === "manual") return emptyPlot("scaling-plot", "Scaling is exported for automatic designs only.");
  const selected = automaticSelection();
  const axis = byId("scaling-axis").value;
  const rows = state.data.scaling.filter((row) => row.metric === "choice_accuracy" && matchesAutomatic(row, selected, axis));
  const labels = {J: "Judges (J)", M: "Anchors (M)", N: "Comparators (N)", attempted_items: "Attempted items", presentations: "Presentations"};
  bandPlot("scaling-plot", rows, axis, labels[axis], "Higher-θ choice accuracy");
}

function joinedTokenRows(rows, scenario) {
  const joined = new Map();
  for (const row of rows) {
    if (!row.model || row.model === "__total__" || !row.role || row.role === "all") continue;
    const key = `${row.model}\u0000${row.reasoning_class}\u0000${row.role}`;
    if (!joined.has(key)) joined.set(key, {scenario, model: row.model, reasoning_class: row.reasoning_class, role: row.role});
    const target = joined.get(key);
    if (tokenMetrics[row.metric]) target[tokenMetrics[row.metric]] = row;
    if (row.metric === "cost") target.cost = row;
  }
  return [...joined.values()];
}

function renderTokens(autoRows, manualGroups) {
  const body = byId("token-table").tBodies[0];
  body.replaceChildren();
  const rows = byId("mode").value === "automatic"
    ? joinedTokenRows(autoRows, "Automatic")
    : manualGroups.flatMap((group) => joinedTokenRows(group.rows, `Row ${group.index + 1}`));
  if (!rows.length) {
    const cell = document.createElement("td");
    cell.colSpan = 9;
    cell.className = "empty-cell";
    cell.textContent = "No per-model token forecast is available for this exact scenario.";
    const row = document.createElement("tr"); row.append(cell); body.append(row); return;
  }
  for (const item of rows) {
    const row = document.createElement("tr");
    for (const value of [item.scenario, item.model, item.reasoning_class || "—", item.role, interval(item.input), interval(item.output), interval(item.ordinary), interval(item.reasoning), interval(item.cost, true)]) {
      const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    }
    body.append(row);
  }
}

function renderManualResults(groups) {
  const section = byId("manual-results");
  section.hidden = byId("mode").value !== "manual";
  const body = section.querySelector("tbody");
  body.replaceChildren();
  for (const group of groups) {
    const probability = metric(group.rows, "p_choose_m1", (row) => row.comparison_id !== null);
    const accuracy = metric(group.rows, "choice_accuracy", (row) => row.comparison_id !== null);
    const candidateCalls = metric(group.rows, "candidate_calls", (item) => item.comparison_id === null);
    const judgeCalls = metric(group.rows, "judgment_calls", (item) => item.comparison_id !== null);
    const candidateCost = metric(group.rows, "cost", (item) => item.model === "__total__" && item.role === "candidate");
    const judgeCost = metric(group.rows, "cost", (item) => item.model === "__total__" && item.role === "judge");
    const totalCost = metric(group.rows, "cost", (item) => item.model === "__total__" && item.role === "all");
    const row = document.createElement("tr");
    const values = [
      group.index + 1, `${group.models.judge}: ${group.models.m1} vs ${group.models.m2}`,
      interval(probability), interval(accuracy), interval(candidateCalls), interval(judgeCalls),
      interval(candidateCost, true), interval(judgeCost, true), interval(totalCost, true),
      group.missingClasses.length
        ? `Override required: ${group.missingClasses.join(", ")}`
        : group.rows.length ? (group.rows[0].scenario_prediction_status || "predicted") : "not exported",
    ];
    for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }
    body.append(row);
  }
}

function relevantRanking(autoRows, manualGroups) {
  if (byId("mode").value === "automatic") {
    const configurations = new Set(autoRows.map((row) => row.configuration));
    return state.data.ranking_recovery.filter((row) => configurations.has(row.configuration));
  }
  const identities = manualGroups.map((group) => group.identity).filter(Boolean);
  return state.data.ranking_recovery.filter((row) => identities.some((identity) => matchesIdentity(row, identity)));
}

function rankingScenarioKey(row) {
  const scenario = row.mode === "manual"
    ? identityKeys.map((key) => row[key])
    : [row.configuration, ...autoKeys.map((key) => row[key])];
  return JSON.stringify([...scenario, row.component_models]);
}

function renderRanking(rows) {
  const valid = rows.filter((row) => row.valid && row.metric === "kendall_tau");
  if (!valid.length) {
    emptyPlot("ranking-plot", "No available ranking recovery for this exact scenario.");
  } else {
    const grouped = new Map();
    for (const row of valid) {
      const key = rankingScenarioKey(row);
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(row);
    }
    const traces = [];
    for (const group of grouped.values()) {
      const row = group[0];
      const scenario = row.mode === "manual"
        ? `${row.judge}: ${row.m1} vs ${row.m2}`
        : `Grid ${String(row.configuration).slice(0, 8)}`;
      traces.push({
        x: [row.component_models, row.component_models, row.component_models],
        y: [row.p5, row.p50, row.p95], mode: "markers",
        marker: {size: [7, 11, 7], symbol: ["line-ew", "circle", "line-ew"]},
        name: `${scenario} · ${row.component_models}`,
        text: ["p5", "p50", "p95"],
        hovertemplate: "%{text} %{y:.3f}<extra>%{fullData.name}</extra>",
      });
    }
    const theme = plotTheme();
    theme.showlegend = true;
    theme.xaxis.title = "Stable connected-component membership";
    theme.yaxis.title = "Kendall τ";
    theme.yaxis.range = [-1, 1];
    Plotly.react("ranking-plot", traces, theme, plotConfig);
  }
  const list = byId("ranking-reasons");
  list.replaceChildren();
  const reasons = [...new Set(rows.flatMap((row) => row.partial_reasons || (row.reason ? [row.reason] : [])))].filter(Boolean).sort();
  reasons.forEach((reason) => { const item = document.createElement("li"); item.textContent = reason; list.append(item); });
}

function relevantWarnings(autoRows, manualGroups) {
  const configurations = new Set(autoRows.map((row) => row.configuration));
  return state.data.warnings.filter((warning) => {
    if (warning.code === "manual_bank_truncated") return true;
    if (byId("mode").value === "automatic") return !warning.configuration || configurations.has(warning.configuration) || matchesAutomatic(warning, automaticSelection());
    return !identityKeys.some((key) => key in warning) || manualGroups.some((group) => group.identity && matchesIdentity(warning, group.identity));
  });
}

function renderWarnings(warnings) {
  const list = byId("warning-list");
  list.replaceChildren();
  if (!warnings.length) {
    const item = document.createElement("li"); item.textContent = "No warnings for the current exact scenario."; list.append(item); return;
  }
  for (const warning of warnings) {
    const item = document.createElement("li");
    const code = document.createElement("code"); code.textContent = warning.code || "warning";
    const message = document.createElement("span"); message.textContent = warning.message || "See export metadata.";
    item.append(code, message); list.append(item);
  }
}

function renderManualGuidance(groups) {
  const guidance = byId("manual-guidance");
  const invalid = groups.filter((group) => group.missingClasses.length);
  const missing = groups.filter((group) => group.identity && !group.rows.length).map((group) => group.identity);
  guidance.hidden = !invalid.length && !missing.length;
  if (invalid.length) {
    guidance.textContent = `Choose a reasoning-class override for every model without a catalogue default (rows ${invalid.map((group) => group.index + 1).join(", ")}). No scenario identity is produced until those overrides are valid.`;
  } else if (missing.length) {
    guidance.textContent = `No exact exported prediction exists for ${missing.length === 1 ? "this configuration" : "these configurations"}. Add ${JSON.stringify(missing.length === 1 ? missing[0] : missing)} to dashboard.manual_triples, then rerun mc-cost export-dashboard --config configs/mc_cost/default.json.`;
  } else {
    guidance.textContent = "";
  }
}

function renderAutomaticGuidance(rows) {
  const guidance = byId("automatic-guidance");
  guidance.hidden = rows.length > 0;
  guidance.textContent = rows.length
    ? ""
    : "This exact grid or pool selection is not in the export. Add it to the dashboard configuration, then rerun mc-cost export-dashboard --config configs/mc_cost/default.json.";
}

function render() {
  if (!state.data) return;
  const manualMode = byId("mode").value === "manual";
  byId("automatic-controls").hidden = manualMode;
  byId("manual-controls").hidden = !manualMode;
  const autoRows = manualMode ? [] : exactAutomaticRows();
  const manualGroups = manualMode ? selectedManualGroups() : [];
  renderAutomaticGuidance(autoRows);
  renderManualGuidance(manualGroups);
  renderSummary(autoRows, manualGroups);
  renderBadges(autoRows, manualGroups);
  renderAccuracy(autoRows, manualGroups);
  renderScaling();
  renderTokens(autoRows, manualGroups);
  renderManualResults(manualGroups);
  renderRanking(relevantRanking(autoRows, manualGroups));
  renderWarnings(relevantWarnings(autoRows, manualGroups));
}

async function load() {
  try {
    const response = await fetch("data/mc_cost.json");
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    state.data = await response.json();
    populateAutomaticControls();
    const firstManual = state.data.manual[0];
    addManualRow(firstManual || null);
    byId("data-status").textContent = `Validated ${state.data.generated_at} · ${state.data.automatic.length.toLocaleString()} automatic rows · ${state.data.manual.length.toLocaleString()} manual rows`;
    render();
  } catch (error) {
    byId("data-status").textContent = `Dashboard data unavailable: ${error.message}`;
    byId("warning-list").replaceChildren(Object.assign(document.createElement("li"), {textContent: "Serve interactive_viewer over HTTP after running mc-cost export-dashboard."}));
    for (const plot of ["accuracy-plot", "scaling-plot", "ranking-plot"]) emptyPlot(plot, "Export data before viewing this plot.");
  }
}

for (const id of ["mode", "judge-pool", "anchor-pool", "comparator-pool", "judge-count", "anchor-count", "comparator-count", "attempted-items", "presentations", "strength-filter", "scaling-axis"]) {
  byId(id).addEventListener("change", render);
}
byId("add-manual-row").addEventListener("click", () => { addManualRow(); render(); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
load();
