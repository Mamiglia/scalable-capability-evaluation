(function (root) {
  "use strict";

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const Z90 = 1.6448536269514722;
  const number = (value, fallback = null) => value === "" || value == null ? fallback : Number(value);

  function csvRows(text) {
    const rows = [];
    let row = [], field = "", quoted = false;
    for (let i = 0; i <= text.length; i += 1) {
      const char = text[i] || "\n";
      if (char === '"' && quoted && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (char === '"') quoted = !quoted;
      else if (char === "," && !quoted) { row.push(field.trim()); field = ""; }
      else if ((char === "\n" || char === "\r") && !quoted) {
        if (char === "\r" && text[i + 1] === "\n") i += 1;
        row.push(field.trim()); field = "";
        if (row.some(Boolean)) rows.push(row);
        row = [];
      } else field += char;
    }
    if (quoted) throw new Error("CSV contains an unclosed quote");
    return rows;
  }

  function normalizeRow(row, index) {
    const result = {...row};
    for (const key of ["judge", "m1", "m2"]) {
      if (typeof result[key] !== "string" || !result[key].trim()) {
        throw new Error(`Row ${index + 1} requires ${key}`);
      }
      result[key] = result[key].trim();
    }
    result.items = number(result.items, 1);
    result.repeats = number(result.repeats ?? result.epochs ?? result.presentations, 1);
    delete result.epochs;
    delete result.presentations;
    if (!Number.isInteger(result.items) || result.items < 1 ||
        !Number.isInteger(result.repeats) || result.repeats < 1) {
      throw new Error(`Row ${index + 1} items and repeats must be positive integers`);
    }
    return result;
  }

  function parseComparisons(text, filename = "") {
    let rows;
    if (filename.toLowerCase().endsWith(".json") || text.trim().startsWith("[")) {
      rows = JSON.parse(text);
      if (!Array.isArray(rows)) throw new Error("JSON must be an array of comparison objects");
    } else {
      const parsed = csvRows(text);
      if (parsed.length < 2) throw new Error("CSV requires a header and at least one row");
      const headers = parsed[0];
      rows = parsed.slice(1).map((values) => Object.fromEntries(headers.map((key, i) => [key, values[i] || ""])));
    }
    return rows.map(normalizeRow);
  }

  function randomGenerator(seed) {
    let value = Number(seed) >>> 0;
    return () => {
      value += 0x6D2B79F5;
      let next = value;
      next = Math.imul(next ^ next >>> 15, next | 1);
      next ^= next + Math.imul(next ^ next >>> 7, next | 61);
      return ((next ^ next >>> 14) >>> 0) / 4294967296;
    };
  }

  function takeSample(values, count, random) {
    const copy = [...values];
    for (let i = 0; i < count; i += 1) {
      const selected = i + Math.floor(random() * (copy.length - i));
      [copy[i], copy[selected]] = [copy[selected], copy[i]];
    }
    return copy.slice(0, count);
  }

  function sampleDesign(catalogue, config) {
    const counts = [config.judges, config.anchors, config.comparators, config.items, config.repeats];
    if (!counts.every((value) => Number.isInteger(value) && value > 0)) {
      throw new Error("J, N, M, items, and repeats must be positive integers");
    }
    if (!["all", "oversight"].includes(config.condition)) throw new Error("condition must be all or oversight");
    const models = catalogue.filter((model) =>
      Number.isFinite(model.theta) && Number.isFinite(model.price_in) &&
      Number.isFinite(model.price_out) && Number.isFinite(releaseOrdinal(model.release)) &&
      ["reasoning", "non_reasoning"].includes(model.reasoning_class)
    );
    const comparators = (judge, anchor) => models.filter((model) =>
      model.model_id !== anchor.model_id && (config.condition === "all" ||
        releaseOrdinal(model.release) < releaseOrdinal(anchor.release))
    );
    const anchors = (judge) => models.filter((anchor) =>
      anchor.model_id !== judge.model_id &&
      (config.condition === "all" || (anchor.theta > judge.theta &&
        releaseOrdinal(anchor.release) > releaseOrdinal(judge.release))) &&
      comparators(judge, anchor).length >= config.comparators
    );
    const eligibleJudges = models.filter((judge) => anchors(judge).length >= config.anchors);
    if (eligibleJudges.length < config.judges) {
      throw new Error(`Only ${eligibleJudges.length} judges support J=${config.judges}, N=${config.anchors}, M=${config.comparators}`);
    }
    const random = randomGenerator(config.seed ?? 0);
    return takeSample(eligibleJudges, config.judges, random).flatMap((judge) =>
      takeSample(anchors(judge), config.anchors, random).flatMap((anchor) =>
        takeSample(comparators(judge, anchor), config.comparators, random).map((comparator) => ({
          judge: judge.model_id, m1: anchor.model_id, m2: comparator.model_id,
          items: config.items, repeats: config.repeats,
        }))
      )
    );
  }

  function modelMetadata(role, row, catalogue) {
    const model = catalogue.get(row[role]);
    const value = (name, fallback) => number(row[`${role}_${name}`], fallback);
    const metadata = {
      model_id: row[role],
      theta: value("theta", model?.theta),
      release: row[`${role}_release`] || model?.release,
      reasoning_class: row[`${role}_reasoning_class`] || model?.reasoning_class,
      price_in: value("price_in", model?.price_in),
      price_out: value("price_out", model?.price_out),
    };
    if (![metadata.theta, metadata.price_in, metadata.price_out].every(Number.isFinite) ||
        !["reasoning", "non_reasoning"].includes(metadata.reasoning_class)) {
      throw new Error(`${row[role]} is missing theta, reasoning_class, or input/output pricing`);
    }
    return metadata;
  }

  function lengthStats(model, observed, defaults) {
    return observed[model.model_id] || defaults[model.reasoning_class] || defaults._all;
  }

  function candidateAccuracy(model, data) {
    const stats = data.candidate_models[model.model_id];
    if (Number.isFinite(stats?.accuracy)) {
      const n = Math.max(1, number(stats.n, 1));
      const denominator = 1 + Z90 ** 2 / n;
      const center = (stats.accuracy + Z90 ** 2 / (2 * n)) / denominator;
      const half = Z90 * Math.sqrt(stats.accuracy * (1 - stats.accuracy) / n + Z90 ** 2 / (4 * n ** 2)) / denominator;
      return {mean: stats.accuracy, low: Math.max(0, center - half), high: Math.min(1, center + half)};
    }
    const x = [1, model.theta];
    const eta = data.correctness_fit.intercept + data.correctness_fit.theta * model.theta;
    const se = Math.sqrt(quadraticVariance(x, data.correctness_fit.covariance));
    return {mean: sigmoid(eta), low: sigmoid(eta - Z90 * se), high: sigmoid(eta + Z90 * se)};
  }

  function quadraticVariance(x, covariance) {
    if (!Array.isArray(covariance)) return 0;
    return Math.max(0, x.reduce((sum, left, i) => sum + x.reduce(
      (inner, right, j) => inner + left * right * Number(covariance[i]?.[j] || 0), 0,
    ), 0));
  }

  function releaseOrdinal(value) {
    if (Number.isFinite(value)) return value;
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : -Infinity;
  }

  function tokenCost(model, input, output) {
    return (input * model.price_in + output * model.price_out) / 1_000_000;
  }

  function tokenCostSd(model, inputSd, outputSd) {
    return Math.hypot(inputSd * model.price_in, outputSd * model.price_out) / 1_000_000;
  }

  const visible = (stats, suffix) => stats[`visible_${suffix}`] ?? stats[`output_${suffix}`];

  function estimateComparison(row, data, catalogue, sensitivity) {
    const judge = modelMetadata("judge", row, catalogue);
    const m1 = modelMetadata("m1", row, catalogue);
    const m2 = modelMetadata("m2", row, catalogue);
    const m1Length = lengthStats(m1, data.candidate_models, data.candidate_defaults);
    const m2Length = lengthStats(m2, data.candidate_models, data.candidate_defaults);
    const judgeLength = lengthStats(judge, data.judge_models, data.judge_defaults);
    const calls = row.items * row.repeats;
    const candidateCostPerCall = (
      tokenCost(m1, m1Length.input_mean, m1Length.output_mean) +
      tokenCost(m2, m2Length.input_mean, m2Length.output_mean)
    );
    const candidateCost = calls * candidateCostPerCall;
    const candidateCostHalf = Z90 * Math.sqrt(calls) * Math.hypot(
      tokenCostSd(m1, m1Length.input_sd, m1Length.output_sd),
      tokenCostSd(m2, m2Length.input_sd, m2Length.output_sd),
    );
    const candidateCostLow = Math.max(0, candidateCost - candidateCostHalf);
    const candidateCostHigh = candidateCost + candidateCostHalf;
    const judgeInput = judgeLength.input_overhead_mean + visible(m1Length, "mean") + visible(m2Length, "mean");
    const judgeCost = calls * tokenCost(judge, judgeInput, judgeLength.output_mean);
    const judgeCostHalf = Z90 * Math.sqrt(calls) * tokenCostSd(
      judge, Math.hypot(
        judgeLength.input_overhead_sd, visible(m1Length, "sd"), visible(m2Length, "sd"),
      ), judgeLength.output_sd,
    );
    const judgeCostLow = Math.max(0, judgeCost - judgeCostHalf);
    const judgeCostHigh = judgeCost + judgeCostHalf;
    const p1 = candidateAccuracy(m1, data), p2 = candidateAccuracy(m2, data);
    const gap = Math.abs(m1.theta - m2.theta);
    const oversight = (m1.theta + m2.theta) / 2 - judge.theta;
    const strongerSign = m1.theta >= m2.theta ? 1 : -1;
    const lengthDifference = strongerSign * (
      Math.log1p(4 * visible(m1Length, "mean")) - Math.log1p(4 * visible(m2Length, "mean"))
    );
    const fit = data.accuracy_glm;
    const x = [sensitivity * gap, sensitivity * gap * oversight, lengthDifference];
    const eta = fit.coefficients.capability_gap * x[0] +
      fit.coefficients.gap_x_oversight * x[1] + fit.coefficients.length_difference * x[2];
    const se = Math.sqrt(quadraticVariance(x, fit.covariance));
    const accuracy = sigmoid(eta);
    const accuracyHalf = Z90 * Math.sqrt(
      (accuracy * (1 - accuracy) * se) ** 2 + accuracy * (1 - accuracy) / calls,
    );
    const mutuallyCorrect = row.items * p1.mean * p2.mean;
    const mutualHalf = Z90 * Math.sqrt(row.items * p1.mean * p2.mean * (1 - p1.mean * p2.mean));
    return {
      ...row, capability_gap: gap, oversight,
      judging_accuracy: accuracy,
      judging_accuracy_low: Math.max(0, accuracy - accuracyHalf),
      judging_accuracy_high: Math.min(1, accuracy + accuracyHalf),
      mutually_correct: mutuallyCorrect,
      mutually_correct_low: Math.max(0, row.items * p1.low * p2.low - mutualHalf),
      mutually_correct_high: Math.min(row.items, row.items * p1.high * p2.high + mutualHalf),
      candidate_cost: candidateCost, candidate_cost_low: candidateCostLow, candidate_cost_high: candidateCostHigh,
      judge_cost: judgeCost, judge_cost_low: judgeCostLow, judge_cost_high: judgeCostHigh,
      total_cost: candidateCost + judgeCost,
      total_cost_low: Math.max(0, candidateCost + judgeCost - Math.hypot(candidateCostHalf, judgeCostHalf)),
      total_cost_high: candidateCost + judgeCost + Math.hypot(candidateCostHalf, judgeCostHalf),
      judge_input_tokens: judgeInput,
      m1_length: m1Length, m2_length: m2Length, judge_length: judgeLength,
      m1_accuracy: p1.mean, m1_accuracy_low: p1.low, m1_accuracy_high: p1.high,
      m2_accuracy: p2.mean, m2_accuracy_low: p2.low, m2_accuracy_high: p2.high,
    };
  }

  function gapCurve(data, catalogue, rows, sensitivity, condition) {
    if (!rows.length) throw new Error("The selected design has no comparisons");
    const design = rows.map((row) => {
      const judge = modelMetadata("judge", row, catalogue);
      const m1 = modelMetadata("m1", row, catalogue);
      const m2 = modelMetadata("m2", row, catalogue);
      return {oversight: (m1.theta + m2.theta) / 2 - judge.theta, calls: row.items * row.repeats};
    });
    const totalCalls = design.reduce((sum, row) => sum + row.calls, 0);
    const fit = data.accuracy_glm;
    const x = Array.from({length: 31}, (_, i) => i / 10);
    const values = x.map((gap) => {
      const predictions = design.map((row) => {
        const predictors = [sensitivity * gap, sensitivity * gap * row.oversight, 0];
        const eta = fit.coefficients.capability_gap * predictors[0] +
          fit.coefficients.gap_x_oversight * predictors[1];
        return {p: sigmoid(eta), predictors, calls: row.calls};
      });
      const mean = predictions.reduce((sum, row) => sum + row.calls * row.p, 0) / totalCalls;
      const gradient = [0, 1, 2].map((index) => predictions.reduce(
        (sum, row) => sum + row.calls * row.p * (1 - row.p) * row.predictors[index], 0,
      ) / totalCalls);
      const modelVariance = quadraticVariance(gradient, fit.covariance);
      const samplingVariance = predictions.reduce(
        (sum, row) => sum + row.calls * row.p * (1 - row.p), 0,
      ) / totalCalls ** 2;
      const half = Z90 * Math.sqrt(modelVariance + samplingVariance);
      return [mean, Math.max(0, mean - half), Math.min(1, mean + half)];
    });
    return {
      name: condition === "oversight" ? "Oversight" : "All", x,
      y: values.map((value) => value[0]), p5: values.map((value) => value[1]),
      p95: values.map((value) => value[2]),
    };
  }

  const api = {estimateComparison, gapCurve, parseComparisons, sampleDesign};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.McCostCalculator = api;
})(typeof window === "undefined" ? globalThis : window);
