import numpy as np
import pandas as pd
import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import (
    ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser,
    ContentReasoning, ContentText, ModelOutput,
)
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall

from capability.task import Cohort
from protocols.transcript_features import TranscriptRegression
from protocols.transcript_features.features import FEATURE_SETS, extract_features, lexical_features, transition_features
from protocols.transcript_features.protocol import ALPHAS
from protocols.transcript_features.reader import extract_log, feature_table, features_path, read_features


def cohort(directory, theta, texts, extra_samples=()):
    directory.mkdir(parents=True, exist_ok=True)
    items = pd.DataFrame({"bench": ["b", "b"], "item": ["0", "1"]})
    models = pd.DataFrame({"theta": theta, "path_b": None})
    for model, answers in texts.items():
        samples = [EvalSample(id=str(i), epoch=1, input="Question?", target="",
                             messages=[ChatMessageAssistant(content=answers[i])] if i < len(answers) else [],
                             scores={"correct": Score(value="C" if i < len(answers) else "I")})
                   for i in range(2)] + list(extra_samples)
        log = EvalLog(status="success", eval=EvalSpec(created="2026-09-14T12:00:00Z", task="b", model=model,
                      dataset=EvalDataset(name="b", samples=len(samples)), config=EvalConfig()), samples=samples)
        path = directory / f"{model}.eval"
        write_eval_log(log, path)
        extract_log(path).to_csv(features_path(path), index=False)  # as scripts/glm_features/extract.py does
        models.loc[model, "path_b"] = str(path)
    return Cohort(models, items, "semester")


def test_extraction_uses_assistant_text_and_excludes_tool_payloads():
    answer = "Oh wait.\n```python\nverify = 'perhaps'\n```\n`maybe`\n\\(wait\\)\nANSWER: wait"
    messages = [
        ChatMessageSystem(content="wait " * 100),
        ChatMessageUser(content="perhaps " * 100),
        ChatMessageAssistant(content=[
            ContentReasoning(reasoning="Perhaps I should verify."),
            ContentReasoning(reasoning="wait", summary="perhaps", redacted=True),
            ContentText(text=answer),
        ], tool_calls=[ToolCall(id=i, function="bash", arguments={"cmd": "maybe verify"}) for i in ("1", "2")]),
        ChatMessageTool(content="verify\n\n[Assistant]: maybe", tool_call_id="1"),
        ChatMessageUser(content="try again"),
    ]
    result = extract_features(messages)
    for feature in ("hedging", "personal_pronouns", "verification", "backtracking"):
        assert result[feature] == pytest.approx(1000 / 6)  # six prose words; "Oh wait" counts once
    assert result["log_chars"] == pytest.approx(np.log1p(24 + 1 + len(answer)))
    assert result["log_reasoning_chars"] == pytest.approx(np.log1p(24))
    assert len(result) == 13
    assert result["log_tool_calls"] == pytest.approx(np.log1p(2))  # repeated calls still count
    for text, feature, expected in [("I = 1", "personal_pronouns", 0), ("I\nthink", "hedging", 500),
                                     ("I don’t know", "uncertainty", 1000 / 3), ("waiting", "backtracking", 0)]:
        assert lexical_features(text)[feature] == expected
    hidden = extract_features(messages, include_reasoning=False)
    assert hidden["log_chars"] == pytest.approx(np.log1p(len(answer))) and hidden["visible_reasoning"] == 0
    assert hidden["hedging"] == 0 and hidden["backtracking"] == pytest.approx(1000 / 2)  # answer prose only
    assert lexical_features("costs $5 to $10 but $x$")["contrast"] == 1000 / 5  # dollar amounts are prose
    assert all(np.isnan(v) for v in extract_features([]).values())
    empty = extract_features([ChatMessageAssistant(content="")])
    assert empty["log_chars"] == empty["visible_reasoning"] == empty["hedging"] == empty["log_tool_calls"] == 0


def test_aggregation_preserves_item_rates_and_missing_observations(tmp_path):
    rates = transition_features(["tool_output", "tool_inspect", "tool_output", "tool_act", "tool_output",
                                 "feedback", "organization", "equation", "equation", "explanation"])
    assert len(rates) == 11  # six transition rates + four flags + action share
    assert rates["tool_action_share"] == .5
    assert transition_features(["tool_act"])["tool_action_share"] == 1.  # include the final call
    assert transition_features([])["tool_action_share"] == 0.
    assert rates["transition__tool_output__tool_inspect"] == rates["transition__tool_output__tool_act"] == 1 / 3
    assert rates["transition__equation__equation"] == rates["transition__equation__explanation"] == .5
    assert rates["transition__organization__equation"] == rates["source_observed__tool_output"] == 1.
    assert rates["source_observed__explanation"] == rates["transition__explanation__equation"] == 0.
    with pytest.raises(ValueError, match="unknown clause types"):
        transition_features(["invented_label"])
    extra = [EvalSample(id=item, epoch=epoch, input="Question?", target="",
                        messages=[ChatMessageAssistant(content="maybe " * 100)],
                        scores={"correct": Score(value=score)})
             for item, epoch, score in [("wrong", 1, "I"), ("wrong", 2, "C"),
                                        ("unselected", 1, "C"), ("later_only", 2, "C")]]
    extra.append(EvalSample(id="fallback", epoch=1, input="Question?", target="",
                           output=ModelOutput.from_content("a", "maybe"), scores={"correct": Score(value="C")}))
    observed = cohort(tmp_path, {"a": 0., "missing": np.nan}, {"a": ["maybe", "plain words here"]}, extra)
    observed.items = pd.DataFrame({"bench": "b", "item": ["0", "1", "wrong", "later_only", "not_collected"]})
    table = feature_table(observed)
    assert table.loc["a", "hedging"] == 500  # mean of per-item rates, not pooled word counts
    assert table.loc["a", "log_chars"] == pytest.approx((np.log(6) + np.log(17)) / 2)
    assert table.loc["missing"].isna().all()
    csv = features_path(observed.models.loc["a", "path_b"])
    stored = pd.read_csv(csv, dtype={"item": str})
    assert stored.reasoning.value_counts().to_dict() == {"visible": 7, "hidden": 7}  # every sample, both views
    assert stored.correct.sum() == 12 and set(stored.columns[:4]) == {"item", "epoch", "correct", "reasoning"}
    rows = read_features(csv, {"0", "wrong", "fallback", "later_only", "not_collected"})
    assert set(rows.index) == {"0", "fallback"}  # correct first attempts only; never a later correct epoch
    assert rows.loc["fallback", "hedging"] == 1000
    assert np.isnan(rows.loc["fallback", "log_tool_calls"])  # no history does not establish zero calls
    assert read_features(csv, {"0"}, reasoning="hidden").loc["0", "log_chars"] == rows.loc["0", "log_chars"]
    assert read_features(csv, {"not_collected"}).empty
    assert read_features(tmp_path / "nonexistent.csv", {"0"}).empty


def test_glm_variants_fit_references_only_and_reset_between_cohorts(tmp_path):
    train = cohort(tmp_path / "train", {"a": 0., "b": 1., "c": 2., "empty": 9., "unlabelled": np.inf},
                   {"a": [""], "b": ["a"], "c": ["aaa"], "unlabelled": ["aaaaaa"]})
    test = cohort(tmp_path / "test", {"t": np.nan, "empty": np.nan}, {"t": ["a" * 1023]})
    length, regex, tools, structure = [TranscriptRegression(variant=v) for v in ("length", "regex", "tools", "structure")]
    length_x, regex_x, tools_x = length.get_features(train), regex.get_features(train), tools.get_features(train)
    assert "hedging" not in length_x and "hedging" in regex_x and "log_tool_calls" not in regex_x
    pd.testing.assert_frame_equal(regex_x[length_x.columns], length_x)
    pd.testing.assert_frame_equal(tools_x[regex_x.columns], regex_x)
    assert tools_x.loc["a", "log_tool_calls"] == 0.
    with pytest.raises(NotImplementedError, match="RST"):
        structure.get_features(train)
    assert [len(FEATURE_SETS[v]) for v in ("length", "regex", "tools", "structure")] == [3, 12, 13, 24]
    assert TranscriptRegression(variant="regex", reasoning="hidden").name == "glm_regex_reasoning-hidden"
    length.fit(train)
    result = length.predict(test)
    empty = Cohort(train.models.iloc[:0], train.items, "empty")
    assert length.predict(empty).empty
    # theta is exactly linear in log_chars on the three references, so leave-one-out picks a light penalty.
    assert length.estimator[-1].alpha_ in ALPHAS
    assert result["t"] == pytest.approx(10, abs=0.5) and np.isnan(result["empty"])
    test.models.theta = -1000
    pd.testing.assert_series_equal(length.predict(test), result)
    length.fit(cohort(tmp_path / "next", {"a": 0.}, {"a": ["a"]}))
    assert length.predict(test).isna().all()
    length.fit(empty)
    assert length.predict(test).isna().all()


def test_rst_labeller_orders_clauses_and_protected_spans():
    """scripts/glm_features/rst_labels.py with a stub parser: one node per clause, protected spans in place."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rst_labels", "scripts/glm_features/rst_labels.py")
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)

    class Unit:
        def __init__(self, start, end, left=None, right=None, relation=None, nuclearity=None):
            self.start, self.end, self.left, self.right, self.relation, self.nuclearity = start, end, left, right, relation, nuclearity

    class StubParser:
        """Splits at sentence ends; joins every pair as elaboration with the first sentence as nucleus."""
        def __call__(self, text):
            cuts, leaves = [0] + [i + 1 for i, c in enumerate(text) if c == "."], []
            leaves = [Unit(a, b) for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]
            tree = leaves[0]
            for leaf in leaves[1:]:
                tree = Unit(tree.start, leaf.end, tree, leaf, "elaboration", "NS")
            return {"rst": [tree]}

    plain = "First clause. Second clause. Third clause."
    assert rl.prose_labels(StubParser(), plain) == ["elaboration"] * 3  # no protected span: every clause kept
    mixed = "Intro. $$x=1$$ Then more.\nANSWER: 3"
    assert rl.prose_labels(StubParser(), mixed) == ["elaboration", "equation", "elaboration", "answer"]
    labels, failed = rl.transcript_labels(StubParser(), [
        {"kind": "reasoning", "text": "Think. Again."}, {"kind": "text", "text": "Done."},
        {"kind": "tool_call", "text": '{"cmd": "cat a"}', "function": "bash"},
        {"kind": "tool_output", "text": "a"}, {"kind": "feedback", "text": "try again"}])
    assert failed == 0
    assert labels == [["elaboration", True], ["elaboration", True], ["root", False], ["tool_inspect", False],
                      ["tool_output", False], ["feedback", False]]
