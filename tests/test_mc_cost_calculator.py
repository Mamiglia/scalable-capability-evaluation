import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from mc_cost.calculator import build_calculator_payload


ROOT = Path(__file__).resolve().parents[1]


class _Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = {}
        self.order = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.elements[values["id"]] = (tag, values)
            self.order.append(values["id"])
        if tag == "script" and values.get("src"):
            self.scripts.append(values["src"])


def test_calculator_payload_uses_observed_gpqa_lengths_and_class_fallbacks():
    catalogue = pd.DataFrame([
        {"model_id": "a", "theta": 1.0, "eci": 120.0, "release": "2025-01-01", "reasoning_class": "reasoning", "price_in": 1.0, "price_out": 2.0},
        {"model_id": "b", "theta": 0.0, "eci": 100.0, "release": "2023-01-01", "reasoning_class": "reasoning", "price_in": 1.0, "price_out": 2.0},
        {"model_id": "c", "theta": -1.0, "eci": 90.0, "release": "2022-01-01", "reasoning_class": "non_reasoning", "price_in": 1.0, "price_out": 2.0},
        {"model_id": "d", "theta": 2.0, "eci": 140.0, "release": "2026-01-01", "reasoning_class": "reasoning", "price_in": 1.0, "price_out": 2.0},
        {"model_id": "j", "theta": 0.0, "eci": 95.0, "release": "2024-01-01", "reasoning_class": "reasoning", "price_in": 1.0, "price_out": 2.0},
    ])
    candidates = pd.DataFrame([
        {"model": "a", "input_tokens": 10, "output_tokens": 100, "response_chars": 40, "correct": 1, "reasoning_class": "reasoning"},
        {"model": "a", "input_tokens": 30, "output_tokens": 300, "response_chars": 80, "correct": 0, "reasoning_class": "reasoning"},
        {"model": "c", "input_tokens": 5, "output_tokens": 20, "response_chars": 20, "correct": 0, "reasoning_class": "non_reasoning"},
        {"model": "c", "input_tokens": 5, "output_tokens": 40, "response_chars": 40, "correct": 1, "reasoning_class": "non_reasoning"},
    ])
    judges = pd.DataFrame([
        {"model": "j", "input_tokens": 150, "output_tokens": 10, "transcript_chars": 400, "reasoning_class": "reasoning"},
        {"model": "j", "input_tokens": 250, "output_tokens": 30, "transcript_chars": 800, "reasoning_class": "reasoning"},
    ])
    judgments = pd.DataFrame([
        {"chosen_left": chosen, "left_theta": left, "right_theta": right,
         "judge_theta": judge, "left_chars": left_chars, "right_chars": right_chars}
        for chosen, left, right, judge, left_chars, right_chars in [
            (1, 1.0, 0.0, 0.0, 400, 200), (0, 1.0, 0.0, 0.0, 200, 400),
            (1, 2.0, 0.0, 0.0, 500, 200), (1, 0.0, 1.0, 0.0, 200, 400),
            (0, 0.0, 2.0, 0.0, 200, 500), (0, 0.0, 1.0, 1.0, 400, 200),
            (1, 1.0, 0.0, 1.0, 200, 400), (0, 0.0, 2.0, 1.0, 500, 200),
        ]
    ])
    judgments["left_release_day"] = [2100, 2100, 2200, 1900, 1900, 1900, 2100, 1900]
    judgments["right_release_day"] = [1900, 1900, 1900, 2100, 2200, 2100, 1900, 2200]
    judgments["judge_release_day"] = [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
    by_theta = {0.0: "b", 1.0: "a", 2.0: "d"}
    judgments["left_model"] = judgments["left_theta"].map(by_theta)
    judgments["right_model"] = judgments["right_theta"].map(by_theta)

    payload = build_calculator_payload(catalogue, candidates, judges, judgments)

    assert payload["candidate_models"]["a"] == {
        "accuracy": 0.5,
        "input_mean": 20.0,
        "input_sd": np.sqrt(200.0),
        "n": 2,
        "output_mean": 200.0,
        "output_sd": np.sqrt(20000.0),
        "source": "observed_gpqa",
        "visible_mean": 15.0,
        "visible_sd": np.sqrt(50.0),
    }
    assert payload["candidate_defaults"]["reasoning"]["output_mean"] == 200.0
    assert payload["judge_models"]["j"]["input_overhead_mean"] == 50.0
    assert payload["judge_models"]["j"]["output_mean"] == 20.0
    assert payload["judge_defaults"]["reasoning"]["source"] == "reasoning_class_fallback"
    assert set(payload["accuracy_glm"]["coefficients"]) == {"capability_gap", "length_difference"}
    assert all(np.isfinite(value) for value in payload["accuracy_glm"]["coefficients"].values())
    assert np.asarray(payload["accuracy_glm"]["covariance"]).shape == (2, 2)
    assert payload["catalogue"][0]["eci"] == 120.0
    assert payload["candidate_accuracy_default"]["low"] <= payload["candidate_accuracy_default"]["mean"]
    assert payload["candidate_accuracy_default"]["mean"] <= payload["candidate_accuracy_default"]["high"]
    json.dumps(payload, allow_nan=False)


def test_browser_calculator_parses_uploads_and_computes_costs_and_curves():
    script = ROOT / "interactive_viewer" / "mc_cost_calculator.js"
    node = r'''
const assert = require("node:assert/strict");
const calc = require(process.argv[1]);
const csv = calc.parseComparisons("judge,m1,m2,items,repeats\nj,a,b,10,2\n", "x.csv");
assert.deepEqual(csv, [{judge:"j",m1:"a",m2:"b",items:10,repeats:2}]);
assert.deepEqual(calc.parseComparisons('[{"judge":"j","m1":"a","m2":"b","items":10,"epochs":3}]', "x.json")[0].repeats, 3);
const catalogue = new Map([
  ["a", {model_id:"a",theta:1,eci:120,release:"2025-01-01",reasoning_class:"reasoning",price_in:1,price_out:2}],
  ["b", {model_id:"b",theta:0,eci:100,release:"2023-01-01",reasoning_class:"reasoning",price_in:1,price_out:2}],
  ["c", {model_id:"c",theta:.5,eci:110,release:"2024-06-01",reasoning_class:"reasoning",price_in:1,price_out:2}],
  ["d", {model_id:"d",theta:1.5,eci:140,release:"2026-01-01",reasoning_class:"reasoning",price_in:1,price_out:2}],
  ["e", {model_id:"e",theta:-1,eci:90,release:"2022-01-01",reasoning_class:"reasoning",price_in:1,price_out:2}],
  ["j", {model_id:"j",theta:-.5,eci:95,release:"2024-01-01",reasoning_class:"reasoning",price_in:1,price_out:2}],
  ["j2", {model_id:"j2",theta:.75,eci:105,release:"2024-01-01",reasoning_class:"reasoning",price_in:1,price_out:2}],
  ["bad", {model_id:"bad",theta:2,eci:null,release:"2026-02-01",reasoning_class:"reasoning",price_in:1,price_out:2}],
]);
const data = {
  candidate_models:{a:{input_mean:10,input_sd:1,output_mean:100,output_sd:10,visible_mean:50,visible_sd:5,accuracy:.8,source:"observed_gpqa"}},
  candidate_defaults:{reasoning:{input_mean:20,input_sd:2,output_mean:200,output_sd:20,visible_mean:100,visible_sd:10,source:"reasoning_class_fallback"}},
  judge_models:{j:{input_overhead_mean:50,input_overhead_sd:5,output_mean:20,output_sd:2,source:"observed_gpqa_judging"}},
  judge_defaults:{reasoning:{input_overhead_mean:60,input_overhead_sd:6,output_mean:30,output_sd:3,source:"reasoning_class_fallback"}},
  correctness_fit:{intercept:0,theta:0,covariance:[[.04,0],[0,.01]]},
  accuracy_glm:{coefficients:{capability_gap:1,length_difference:.2},covariance:[[.04,0],[0,.01]],judge_gap_sd:1.5,pair_residual_sd:.2},
  candidate_accuracy_default:{mean:.8,low:.6,high:.95},
};
assert.deepEqual(calc.expectedOverlap(data, 10), {mean:6.4,low:3.6,high:9.025});
const result = calc.estimateComparison(csv[0], data, catalogue, .6);
assert.equal(result.mutually_correct, 4);
assert.ok(result.mutually_correct_low < result.mutually_correct && result.mutually_correct < result.mutually_correct_high);
assert.ok(Math.abs(result.candidate_cost - .0126) < 1e-12);
assert.ok(Math.abs(result.judge_cost - .0048) < 1e-12);
assert.ok(result.candidate_cost_low < result.candidate_cost && result.candidate_cost < result.candidate_cost_high);
assert.ok(result.judge_cost_low < result.judge_cost && result.judge_cost < result.judge_cost_high);
assert.ok(result.judging_accuracy > .5);
assert.ok(result.judging_accuracy_low < result.judging_accuracy && result.judging_accuracy < result.judging_accuracy_high);
assert.equal(result.m1_length.source, "observed_gpqa");
assert.equal(result.m2_length.source, "reasoning_class_fallback");
assert.ok(Math.abs(result.delta_eci - .2) < 1e-12);

const sampled = calc.sampleDesign([...catalogue.values()], {condition:"oversight",judges:1,anchors:2,comparators:2,items:10,repeats:1,seed:3});
assert.equal(sampled.length, 4);
for (const row of sampled) {
  const judge = catalogue.get(row.judge), anchor = catalogue.get(row.m1), comparator = catalogue.get(row.m2);
  assert.ok(anchor.eci > judge.eci && Date.parse(anchor.release) > Date.parse(judge.release));
  assert.ok(Date.parse(comparator.release) < Date.parse(anchor.release));
  assert.notEqual(row.judge, "bad"); assert.notEqual(row.m1, "bad"); assert.notEqual(row.m2, "bad");
}
const curve = calc.gapCurve(data, catalogue, sampled, .6, "oversight");
assert.equal(curve.name, "Oversight");
assert.equal(curve.y[0], .5);
assert.equal(curve.x[10], .1);
assert.ok(Math.abs(curve.y[10] - .6) < 1e-12);
assert.ok(curve.p5[10] < curve.y[10] && curve.y[10] < curve.p95[10]);
assert.ok(curve.p95[10] - curve.p5[10] > .1);
const repeatedItems = sampled.map((row) => ({...row,items:100000,repeats:100}));
const repeatedItemsCurve = calc.gapCurve(data, catalogue, repeatedItems, .6, "oversight");
assert.ok(repeatedItemsCurve.p95[10] - repeatedItemsCurve.p5[10] < (curve.p95[10] - curve.p5[10]) / 100);
const repeatedEpochs = sampled.map((row) => ({...row,repeats:10}));
const repeatedEpochsCurve = calc.gapCurve(data, catalogue, repeatedEpochs, .6, "oversight");
assert.ok(repeatedEpochsCurve.p95[10] - repeatedEpochsCurve.p5[10] < curve.p95[10] - curve.p5[10]);
const repeated = sampled.flatMap((row, i) => [row, {...row, judge:`copy${i}`, _judge:catalogue.get(row.judge)}]);
const repeatedCatalogue = new Map(catalogue);
for (const [i, row] of sampled.entries()) repeatedCatalogue.set(`copy${i}`, catalogue.get(row.judge));
const tighter = calc.gapCurve(data, repeatedCatalogue, repeated, .6, "oversight");
assert.ok(tighter.p95[10] - tighter.p5[10] < curve.p95[10] - curve.p5[10]);
const future = calc.sampleDesign([...catalogue.values()], {condition:"oversight",judges:20,anchors:2,comparators:2,items:10,repeats:1,seed:3});
assert.equal(future.length, 80);
assert.equal(new Set(future.map((row) => row.judge_instance)).size, 20);
'''
    result = subprocess.run(
        ["node", "-e", node, str(script)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_exposes_the_upload_calculator_as_the_primary_workflow():
    parser = _Elements()
    parser.feed((ROOT / "interactive_viewer" / "mc_cost.html").read_text())

    assert parser.elements["comparison-upload"][0] == "input"
    assert parser.elements["comparison-upload"][1]["type"] == "file"
    assert parser.elements["calculator-condition"][0] == "select"
    for element in ("calculator-judges", "calculator-anchors", "calculator-comparators"):
        assert parser.elements[element][1]["type"] == "range"
        assert parser.elements[element][1]["value"] == "5"
    assert parser.elements["calculator-repeats"][1]["type"] == "range"
    assert parser.elements["judge-accuracy"][1]["type"] == "range"
    assert parser.elements["judge-accuracy-default"][0] == "span"
    assert parser.elements["calculator-items-estimate"][0] == "span"
    assert parser.elements["calculator-results"][0] == "table"
    assert parser.elements["gap-curves"][0] == "div"
    assert parser.order.index("gap-curves") < parser.order.index("calculator-results")
    assert parser.elements["calculator-comparison-details"][0] == "details"
    assert parser.elements["calculator-table-scroll"][0] == "div"
    assert "mc_cost_calculator.js" in parser.scripts
