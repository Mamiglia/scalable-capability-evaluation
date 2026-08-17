(function (root) {
  "use strict";

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const logit = (p) => Math.log(p / (1 - p));
  const Z95 = 1.959963984540054;
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

  function normalDraw(random) {
    return Math.sqrt(-2 * Math.log(Math.max(random(), Number.EPSILON))) *
      Math.cos(2 * Math.PI * random());
  }

  function takeSample(values, count, random) {
    const result = [];
    while (result.length < count && values.length) {
      const copy = [...values];
      for (let i = 0; i < copy.length; i += 1) {
        const selected = i + Math.floor(random() * (copy.length - i));
        [copy[i], copy[selected]] = [copy[selected], copy[i]];
      }
      result.push(...copy.slice(0, count - result.length));
    }
    return result;
  }

  function sampleDesign(catalogue, config) {
    const counts = [config.judges, config.anchors, config.comparators, config.items, config.repeats];
    if (!counts.every((value) => Number.isInteger(value) && value > 0)) {
      throw new Error("J, N, M, items, and repeats must be positive integers");
    }
    if (!["all", "oversight"].includes(config.condition)) throw new Error("condition must be all or oversight");
    const models = catalogue.filter((model) =>
      Number.isFinite(model.theta) && Number.isFinite(model.eci) && model.eci > 0 &&
      Number.isFinite(model.price_in) &&
      Number.isFinite(model.price_out) && Number.isFinite(releaseOrdinal(model.release)) &&
      ["reasoning", "non_reasoning"].includes(model.reasoning_class)
    );
    const comparators = (judge, anchor) => models.filter((model) =>
      model.model_id !== anchor.model_id && (config.condition === "all" ||
        releaseOrdinal(model.release) < releaseOrdinal(anchor.release))
    );
    const anchors = (judge) => models.filter((anchor) =>
      anchor.model_id !== judge.model_id &&
      (config.condition === "all" || (anchor.eci > judge.eci &&
        releaseOrdinal(anchor.release) > releaseOrdinal(judge.release))) &&
      comparators(judge, anchor).length > 0
    );
    const eligibleJudges = models.filter((judge) => anchors(judge).length > 0);
    if (!eligibleJudges.length) throw new Error("No catalogue model supports this condition");
    const random = randomGenerator(config.seed ?? 0);
    return takeSample(eligibleJudges, config.judges, random).flatMap((judge, judgeInstance) =>
      takeSample(anchors(judge), config.anchors, random).flatMap((anchor, anchorInstance) =>
        takeSample(comparators(judge, anchor), config.comparators, random).map((comparator, comparatorInstance) => ({
          judge: judge.model_id, m1: anchor.model_id, m2: comparator.model_id,
          judge_instance: judgeInstance, anchor_instance: anchorInstance,
          comparator_instance: comparatorInstance,
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
      eci: value("eci", model?.eci),
      release: row[`${role}_release`] || model?.release,
      reasoning_class: row[`${role}_reasoning_class`] || model?.reasoning_class,
      price_in: value("price_in", model?.price_in),
      price_out: value("price_out", model?.price_out),
    };
    if (![metadata.theta, metadata.eci, metadata.price_in, metadata.price_out].every(Number.isFinite) ||
        metadata.eci <= 0 ||
        !["reasoning", "non_reasoning"].includes(metadata.reasoning_class)) {
      throw new Error(`${row[role]} is missing theta, ECI, reasoning_class, or input/output pricing`);
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
      const denominator = 1 + Z95 ** 2 / n;
      const center = (stats.accuracy + Z95 ** 2 / (2 * n)) / denominator;
      const half = Z95 * Math.sqrt(stats.accuracy * (1 - stats.accuracy) / n + Z95 ** 2 / (4 * n ** 2)) / denominator;
      return {mean: stats.accuracy, low: Math.max(0, center - half), high: Math.min(1, center + half)};
    }
    const x = [1, model.theta];
    const eta = data.correctness_fit.intercept + data.correctness_fit.theta * model.theta;
    const se = Math.sqrt(quadraticVariance(x, data.correctness_fit.covariance));
    return {mean: sigmoid(eta), low: sigmoid(eta - Z95 * se), high: sigmoid(eta + Z95 * se)};
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

  function expectedOverlap(data, items) {
    const value = data.candidate_accuracy_default;
    const overlap = (accuracy) => Math.round(items * accuracy ** 2 * 1e12) / 1e12;
    return {mean: overlap(value.mean), low: overlap(value.low), high: overlap(value.high)};
  }

  function estimateComparison(row, data, catalogue, judgeAccuracy, correctFraction = 1) {
    if (!(judgeAccuracy >= 0.5 && judgeAccuracy < 1)) {
      throw new Error("judge accuracy must be between 50% and 100%");
    }
    if (!(correctFraction > 0 && correctFraction <= 1)) {
      throw new Error("mutually-correct samples must be between 0% and 100%");
    }
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
    const candidateCostHalf = Z95 * Math.sqrt(calls) * Math.hypot(
      tokenCostSd(m1, m1Length.input_sd, m1Length.output_sd),
      tokenCostSd(m2, m2Length.input_sd, m2Length.output_sd),
    );
    const candidateInputTokens = calls * (m1Length.input_mean + m2Length.input_mean);
    const candidateInputHalf = Z95 * Math.sqrt(calls) * Math.hypot(m1Length.input_sd, m2Length.input_sd);
    const candidateOutputTokens = calls * (m1Length.output_mean + m2Length.output_mean);
    const candidateOutputHalf = Z95 * Math.sqrt(calls) * Math.hypot(m1Length.output_sd, m2Length.output_sd);
    const candidateCostLow = Math.max(0, candidateCost - candidateCostHalf);
    const candidateCostHigh = candidateCost + candidateCostHalf;
    const judgeInput = judgeLength.input_overhead_mean + visible(m1Length, "mean") + visible(m2Length, "mean");
    const judgeCost = calls * tokenCost(judge, judgeInput, judgeLength.output_mean);
    const judgeCostHalf = Z95 * Math.sqrt(calls) * tokenCostSd(
      judge, Math.hypot(
        judgeLength.input_overhead_sd, visible(m1Length, "sd"), visible(m2Length, "sd"),
      ), judgeLength.output_sd,
    );
    const judgeInputTokens = calls * judgeInput;
    const judgeInputHalf = Z95 * Math.sqrt(calls) * Math.hypot(
      judgeLength.input_overhead_sd, visible(m1Length, "sd"), visible(m2Length, "sd"),
    );
    const judgeOutputTokens = calls * judgeLength.output_mean;
    const judgeOutputHalf = Z95 * Math.sqrt(calls) * judgeLength.output_sd;
    const judgeCostLow = Math.max(0, judgeCost - judgeCostHalf);
    const judgeCostHigh = judgeCost + judgeCostHalf;
    const p1 = candidateAccuracy(m1, data), p2 = candidateAccuracy(m2, data);
    const deltaEci = Math.abs(m1.eci - m2.eci) / Math.min(m1.eci, m2.eci);
    const strongerSign = m1.eci >= m2.eci ? 1 : -1;
    const lengthDifference = strongerSign * (
      Math.log1p(4 * visible(m1Length, "mean")) - Math.log1p(4 * visible(m2Length, "mean"))
    );
    const fit = data.accuracy_glm;
    const variance = fit.variance_components || {};
    const x = [0, lengthDifference];
    const eta = logit(judgeAccuracy) / 0.1 * deltaEci +
      fit.coefficients.length_difference * lengthDifference;
    const se = Math.sqrt(quadraticVariance(x, fit.covariance));
    const accuracy = sigmoid(eta);
    const usableItems = Math.max(Number.EPSILON, row.items * correctFraction);
    const usableCalls = usableItems * row.repeats;
    const logitVariance = (deltaEci * Number(variance.judge_slope_sd || 0)) ** 2 +
      2 * Number(variance.candidate_sd || 0) ** 2 +
      Number(variance.item_sd || 0) ** 2 / usableItems;
    const accuracyHalf = Z95 * Math.sqrt(
      (accuracy * (1 - accuracy)) ** 2 * (se ** 2 + logitVariance) +
      accuracy * (1 - accuracy) / usableCalls,
    );
    const mutuallyCorrect = row.items * correctFraction;
    const mutualHalf = Z95 * Math.sqrt(row.items * correctFraction * (1 - correctFraction));
    return {
      ...row, delta_eci: deltaEci,
      judging_accuracy: accuracy,
      judging_accuracy_low: Math.max(0, accuracy - accuracyHalf),
      judging_accuracy_high: Math.min(1, accuracy + accuracyHalf),
      mutually_correct: mutuallyCorrect,
      mutually_correct_low: Math.max(0, mutuallyCorrect - mutualHalf),
      mutually_correct_high: Math.min(row.items, mutuallyCorrect + mutualHalf),
      candidate_cost: candidateCost, candidate_cost_low: candidateCostLow, candidate_cost_high: candidateCostHigh,
      judge_cost: judgeCost, judge_cost_low: judgeCostLow, judge_cost_high: judgeCostHigh,
      total_cost: candidateCost + judgeCost,
      total_cost_low: Math.max(0, candidateCost + judgeCost - Math.hypot(candidateCostHalf, judgeCostHalf)),
      total_cost_high: candidateCost + judgeCost + Math.hypot(candidateCostHalf, judgeCostHalf),
      judge_input_tokens: judgeInput,
      candidate_input_tokens: candidateInputTokens,
      candidate_input_tokens_low: Math.max(0, candidateInputTokens - candidateInputHalf),
      candidate_input_tokens_high: candidateInputTokens + candidateInputHalf,
      candidate_output_tokens: candidateOutputTokens,
      candidate_output_tokens_low: Math.max(0, candidateOutputTokens - candidateOutputHalf),
      candidate_output_tokens_high: candidateOutputTokens + candidateOutputHalf,
      judge_input_tokens_total: judgeInputTokens,
      judge_input_tokens_total_low: Math.max(0, judgeInputTokens - judgeInputHalf),
      judge_input_tokens_total_high: judgeInputTokens + judgeInputHalf,
      judge_output_tokens: judgeOutputTokens,
      judge_output_tokens_low: Math.max(0, judgeOutputTokens - judgeOutputHalf),
      judge_output_tokens_high: judgeOutputTokens + judgeOutputHalf,
      m1_length: m1Length, m2_length: m2Length, judge_length: judgeLength,
      m1_accuracy: p1.mean, m1_accuracy_low: p1.low, m1_accuracy_high: p1.high,
      m2_accuracy: p2.mean, m2_accuracy_low: p2.low, m2_accuracy_high: p2.high,
    };
  }

  function summarizeResults(rows) {
    const valid = rows.filter((row) => !row.error);
    const interval = (name) => {
      const mean = valid.reduce((sum, row) => sum + row[name], 0);
      const lowHalf = Math.hypot(...valid.map((row) => row[name] - row[`${name}_low`]));
      const highHalf = Math.hypot(...valid.map((row) => row[`${name}_high`] - row[name]));
      return {mean, low: Math.max(0, mean - lowHalf), high: mean + highHalf};
    };
    return {
      pairs: valid.length,
      judgments: valid.reduce((sum, row) => sum + row.items * row.repeats, 0),
      usable_judgments: valid.reduce((sum, row) => sum + row.mutually_correct * row.repeats, 0),
      candidate_cost: interval("candidate_cost"), judge_cost: interval("judge_cost"),
      total_cost: interval("total_cost"),
      candidate_input_tokens: interval("candidate_input_tokens"),
      candidate_output_tokens: interval("candidate_output_tokens"),
      judge_input_tokens: interval("judge_input_tokens_total"),
      judge_output_tokens: interval("judge_output_tokens"),
    };
  }

  function gapCurve(data, catalogue, rows, judgeAccuracy, condition, correctFraction = 1) {
    if (!rows.length) throw new Error("The selected design has no comparisons");
    if (!(judgeAccuracy >= 0.5 && judgeAccuracy < 1)) {
      throw new Error("judge accuracy must be between 50% and 100%");
    }
    if (!(correctFraction > 0 && correctFraction <= 1)) {
      throw new Error("mutually-correct samples must be between 0% and 100%");
    }
    const fit = data.accuracy_glm;
    const variance = fit.variance_components || {};
    const x = Array.from({length: 31}, (_, i) => i / 100);
    const grouped = new Map();
    for (const row of rows) {
      const key = row.judge_instance ?? row.judge;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(row);
    }
    const random = randomGenerator(0xC0FFEE);
    const judgeGroups = [...grouped.values()];
    const rawSlopes = judgeGroups.map(() => normalDraw(random));
    const rawMean = rawSlopes.reduce((sum, value) => sum + value, 0) / rawSlopes.length;
    const rawSd = rawSlopes.length > 1 ? Math.sqrt(rawSlopes.reduce(
      (sum, value) => sum + (value - rawMean) ** 2, 0,
    ) / (rawSlopes.length - 1)) : 1;
    const judges = judgeGroups.map((judgeRows, index) => {
      const slope = (rawSlopes.length > 1 ? (rawSlopes[index] - rawMean) / rawSd : rawSlopes[index]) *
        Number(variance.judge_slope_sd || 0);
      const anchorCount = new Set(judgeRows.map((row) => row.anchor_instance ?? row.m1)).size;
      const pairCount = judgeRows.length;
      const usableItems = Math.max(Number.EPSILON, judgeRows.reduce(
        (sum, row) => sum + row.items * correctFraction, 0,
      ));
      const usableCalls = Math.max(Number.EPSILON, judgeRows.reduce(
        (sum, row) => sum + row.items * row.repeats * correctFraction, 0,
      ));
      const nestedLogitVariance = Number(variance.candidate_sd || 0) ** 2 *
        (1 / anchorCount + 1 / pairCount) + Number(variance.item_sd || 0) ** 2 / usableItems;
      const values = x.map((gap) => {
        const probability = sigmoid(logit(judgeAccuracy) / 0.1 * gap + slope * gap);
        const half = Z95 * Math.sqrt(
          (probability * (1 - probability)) ** 2 * nestedLogitVariance +
          probability * (1 - probability) / usableCalls,
        );
        return [probability, Math.max(0, probability - half), Math.min(1, probability + half)];
      });
      return {
        name: `Judge ${index + 1}: ${judgeRows[0].judge}`,
        y: values.map((value) => value[0]), p5: values.map((value) => value[1]),
        p95: values.map((value) => value[2]),
      };
    });
    return {
      name: condition === "oversight" ? "Oversight" : "All", x,
      y: x.map((gap) => sigmoid(logit(judgeAccuracy) / 0.1 * gap)), judges,
    };
  }

  const api = {estimateComparison, expectedOverlap, gapCurve, parseComparisons, sampleDesign, summarizeResults};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.McCostCalculator = api;
})(typeof window === "undefined" ? globalThis : window);
