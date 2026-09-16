"""Direct judging: correct first attempts, numeric ratings, and Inspect log compatibility."""

from types import SimpleNamespace
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ContentReasoning, ContentText, ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import Score, Target

from capability.task import Cohort
from configs.domain import DomainBench, DomainConfig
from configs import DirectJudgeConfig
from configs.judge import JudgeConfig
from configs.model import ModelConfig
from judge.dataset import build_direct_dataset
from judge.scorer import direct_judge_scorer
from judge.task import direct_judge_task
from protocols.direct_judge import JudgeDirect, from_logs
from protocols.direct_judge.scoring import calibrate, mean_scores


def test_direct_config_defaults_and_condition_identity(tmp_path):
    config = DirectJudgeConfig.from_cli({"name": "j", "judge": {"name": "j", "slug": "mockllm/model"}})
    assert config.system_prompt == "assets/system_prompt/direct.txt"
    assert config.log_dir == Path("logs")
    assert config.include_reasoning and not config.include_reverse
    pairwise = JudgeConfig(name="j", judge=config.judge, system_prompt=config.system_prompt,
                           include_reasoning=True, include_reverse=False)
    assert config.condition_hash != pairwise.condition_hash
    assert replace(config, include_reverse=True).condition_hash == config.condition_hash
    for change in ({"include_reasoning": False}, {"max_length": 100}, {"temperature": 0.5},
                   {"window": "keep-1/4"}, {"channels": "drop-text"}):
        assert replace(config, **change).condition_hash != config.condition_hash
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("First rating prompt.")
    changed = replace(config, system_prompt=str(prompt))
    previous = changed.condition_hash
    prompt.write_text("Revised rating prompt.")
    assert changed.condition_hash != previous


def source_log(tmp_path):
    path = tmp_path / "source.eval"
    samples = [EvalSample(
        id=item, epoch=epoch, input="Pick a colour.", target="A", choices=["blue", "red"],
        messages=[ChatMessageUser(content="Pick a colour."), ChatMessageAssistant(content=[
            ContentReasoning(reasoning="Private reasoning about the colours."), ContentText(text="ANSWER: A"),
        ])], scores={"correctness": Score(value="C" if correct else "I")},
    ) for item, epoch, correct in [("short", 1, True), ("wrong", 1, False),
                                   ("wrong", 2, True), ("unselected", 1, True)]]
    log = EvalLog(status="success", eval=EvalSpec(created="2026-09-14T12:00:00Z", task="b",
                  dataset=EvalDataset(name="b", samples=len(samples)), model="source-model", config=EvalConfig()),
                  samples=samples)
    write_eval_log(log, path)
    return path


def test_dataset_uses_correct_first_attempts_with_options_and_reasoning_control(tmp_path):
    path = source_log(tmp_path)
    dataset = build_direct_dataset(path, items={"short", "wrong"}, include_reasoning=False)
    assert [sample.id for sample in dataset] == ["short"]
    sample = dataset[0]
    assert sample.target == ""
    assert "Pick a colour." in sample.input and "A) blue\nB) red" in sample.input
    assert "ANSWER: A" in sample.input and "Private reasoning" not in sample.input
    assert "Transcript 2" not in sample.input and "source.eval" not in sample.input
    assert sample.metadata["response_chars"] == 9 and not sample.metadata["truncated"]

    shown = build_direct_dataset(path, items={"short"})[0]
    assert "Private reasoning" in shown.input
    shortened = build_direct_dataset(path, items={"short"}, max_len=5)[0]
    assert shortened.metadata["chars"] == 5 and shortened.metadata["truncated"]
    assert len(build_direct_dataset(path, items={"wrong"})) == 0
    assert len(build_direct_dataset(path, items=set())) == 0
    for length in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            build_direct_dataset(path, items={"short"}, max_len=length)


async def test_decimal_ratings_stay_unrounded_and_invalid_answers_stay_missing():
    scorer = direct_judge_scorer()
    for text, expected in [("ANSWER: 1", 1.), ("**ANSWER:** $7.5$", 7.5), ("ANSWER: 10", 10.),
                           ("ANSWER: 7. This model demonstrates strong capability.", 7.),
                           ("ANSWER: 7<|endoftext|>", 7.),
                           ("EXPLANATION: Strong reasoning. ANSWER: 8", 8.),
                           ("ANSWER: 2\nANSWER: 8.25", 8.25), ("The transcript says ANSWER: 2", None),
                           ("ANSWER: 7.5\nANSWER: 11", None), ("ANSWER: 7.5\nANSWER:", None),
                           ("ANSWER: 7/10", 7.), ("ANSWER: 7.5/10", 7.5), ("ANSWER: 7/11", None),
                           ("ANSWER: 0.9", None), ("ANSWER: 10.5", None),
                           ("ANSWER: nan", None), ("ANSWER: inf", None), ("", None)]:
        result = await scorer(SimpleNamespace(output=ModelOutput.from_content("judge", text)), Target(""))
        if expected is None:
            assert result.answer is None and np.isnan(result.value), text
        else:
            assert float(result.answer) == result.value == expected, text
    explained = await scorer(SimpleNamespace(output=ModelOutput.from_content(
        "judge", "EXPLANATION: Clear reasoning.\nANSWER: 7.5")), Target(""))
    assert explained.explanation == "Clear reasoning."
    scores = pd.DataFrame({"model": ["a", "a", "b", "t"], "item": ["0", "1", "0", "0"],
                           "bench": "b", "score": [1., 2., 3.5, 5.5]})
    assert mean_scores(scores)["b"] == 3.5
    assert calibrate(scores, pd.Series({"a": 0., "b": 2.}))["t"] == pytest.approx(4.)


def test_direct_protocol_shows_reasoning_when_reusing_pairwise_config(tmp_path):
    path = source_log(tmp_path)
    items = pd.DataFrame({"bench": ["b"], "item": ["short"]})
    train = Cohort(pd.DataFrame({"theta": [0.], "path_b": [str(path)]}, index=["r"]), items, "s")
    test = Cohort(pd.DataFrame({"theta": [np.nan], "path_b": [str(path)]}, index=["t"]), items, "s")
    domain = DomainConfig(name="d", benches=[DomainBench(bench="b")])
    domain.task = lambda bench: SimpleNamespace(name="b")
    config = JudgeConfig(name="j", judge=ModelConfig(name="j", slug="mockllm/model"),
                         system_prompt="assets/system_prompt/direct.txt", include_reasoning=False)
    judge = JudgeDirect(config, domain)
    tasks = judge.tasks(train, test)
    assert len(tasks) == 2
    assert all("Private reasoning" in task.dataset[0].input for task in tasks)
    assert all(task.metadata["include_reasoning"] for task in tasks)
    assert judge.config.include_reasoning and not config.include_reasoning
    assert judge.config.condition_hash != config.condition_hash

    hidden = JudgeDirect(config, domain, include_reasoning=False)
    hidden_tasks = hidden.tasks(train, test)
    assert len(hidden_tasks) == 2
    assert all("Private reasoning" not in task.dataset[0].input for task in hidden_tasks)
    assert all("ANSWER: A" in task.dataset[0].input for task in hidden_tasks)
    assert all(not task.metadata["include_reasoning"] for task in hidden_tasks)
    assert hidden.log_dir("s") != judge.log_dir("s")
    assert judge.log_dir("s").relative_to(config.log_dir).parts[0] == "direct"
    assert hidden.log_dir("s").relative_to(config.log_dir).parts[0] == "direct_reasoning-hidden"


def test_benchmark_prompt_overrides_and_log_identity_use_the_same_resolution(tmp_path):
    path = source_log(tmp_path)
    first, chosen = tmp_path / "first.txt", tmp_path / "chosen.txt"
    first.write_text("First benchmark prompt.")
    chosen.write_text("Chosen benchmark prompt.")
    domain = DomainConfig.from_cli({"name": "d", "benches": [
        {"bench": "b", "direct_system_prompt": str(first)},
        {"bench": "b_alias", "direct_system_prompt": str(chosen), "system_prompt": "pairwise.txt"},
    ]})
    domain.task = lambda bench: SimpleNamespace(name="b")
    items = pd.DataFrame({"bench": ["b"], "item": ["short"]})
    train = Cohort(pd.DataFrame({"theta": [0.], "path_b": [str(path)]}, index=["r"]), items, "s")
    test = Cohort(pd.DataFrame({"theta": [np.nan], "path_b": [str(path)]}, index=["t"]), items, "s")
    config = DirectJudgeConfig(name="j", judge=ModelConfig(name="j", slug="mockllm/model"), log_dir=tmp_path)
    judge = JudgeDirect(config, domain)
    tasks = judge.tasks(train, test)
    assert [task.name for task in tasks] == ["b/r", "b/t"]
    assert all(task.metadata["system_prompt"] == str(chosen) for task in tasks)
    before = judge.log_dir("s")
    first.write_text("An unused alias must not change the run.")
    domain.benches[-1].system_prompt = "another-pairwise.txt"
    assert judge.log_dir("s") == before
    chosen.write_text("Revised chosen prompt.")
    assert judge.log_dir("s") != before
    domain.benches[-1].direct_system_prompt = ""
    fallback = judge.log_dir("s")
    assert all(task.metadata["system_prompt"] == config.system_prompt for task in judge.tasks(train, test))
    chosen.write_text("An unused override must not change the run.")
    assert judge.log_dir("s") == fallback


def test_direct_task_roundtrips_decimal_and_missing_scores_through_inspect(tmp_path, monkeypatch):
    # Keep Inspect's trace/cache files inside the test directory.
    from inspect_ai._util import appdirs
    monkeypatch.setattr(appdirs, "user_data_path", lambda name: tmp_path / "inspect-data")
    monkeypatch.setattr(appdirs, "user_cache_path", lambda name: tmp_path / "inspect-cache")
    path = source_log(tmp_path)
    assert direct_judge_task(path, items={"wrong"}) is None
    task = direct_judge_task(path, items={"short", "unselected"}, task_name="b/m",
                             metadata={"bench": "b", "model": "m", "semester": "s"})
    assert task.metadata["protocol"] == "direct" and task.metadata["require_correct"]
    assert task.metadata["source_path"] == str(path)
    assert task.metadata["include_reasoning"] is True
    outputs = [ModelOutput.from_content("mockllm/model", text)
               for text in ("EXPLANATION: Clear reasoning.\nANSWER: 7.5", "No rating.")]
    for output in outputs:
        output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)  # no tokenizer download
    logs = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
                        log_dir=str(tmp_path / "judgments"), display="none", max_samples=1)
    assert logs[0].status == "success", logs[0].error
    table = from_logs(tmp_path / "judgments")
    assert table[["model", "item", "score"]].to_dict("records") == [
        {"model": "m", "item": "short", "score": 7.5}]
    assert mean_scores(table)["m"] == 7.5
    system = next(m.content for m in logs[0].samples[0].messages if m.role == "system")
    assert system == Path("assets/system_prompt/direct.txt").read_text()
    assert any("Private reasoning" in m.text for m in logs[0].samples[0].messages if m.role == "user")


def test_direct_judge_submits_only_unfinished_tasks(tmp_path, monkeypatch):
    from protocols.direct_judge import judge as judge_module

    config = DirectJudgeConfig(name="j", judge=ModelConfig(name="j", slug="mockllm/model"), log_dir=tmp_path)
    judge = JudgeDirect(config, DomainConfig(name="d", benches=[]))
    tasks = [SimpleNamespace(name="b/done"), SimpleNamespace(name="b/new")]
    monkeypatch.setattr(judge, "tasks", lambda train, test: tasks)
    log_dir = judge.log_dir("s")
    log_dir.mkdir(parents=True)
    log = EvalLog(status="success", eval=EvalSpec(created="2026-09-14T12:00:00Z", task="b/done",
                  dataset=EvalDataset(name="b", samples=1), model="mockllm/model", config=EvalConfig()))
    write_eval_log(log, log_dir / "done.eval")
    submitted = []
    monkeypatch.setattr(judge_module, "run_tasks",
                        lambda tasks, config, path: submitted.append([task.name for task in tasks]) or True)

    assert judge.judge(None, SimpleNamespace(semester="s")) == log_dir
    assert submitted == [["b/new"]]
    log.eval.task = "b/new"
    write_eval_log(log, log_dir / "new.eval")
    judge.judge(None, SimpleNamespace(semester="s"))
    assert submitted == [["b/new"]]


def test_evaluate_cli_loads_judge_yamls_and_writes_direct_forecasts(tmp_path, monkeypatch, capsys):
    from capability import evaluate
    from protocols.pairwise import JudgeBT
    from reference import tables

    source = source_log(tmp_path)
    domain_path, model_path, judge_path = (tmp_path / name for name in ("domain.yml", "model.yml", "judge.yml"))
    domain_path.write_text("name: d\nbenches: []\n")
    model_path.write_text("name: j\nslug: mockllm/model\n")
    judge_path.write_text(f"name: j\njudge: {model_path}\ninclude_reasoning: false\n")
    models = pd.DataFrame({"semester": ["previous", "previous", "s"],
                           "s_role": ["reference", "reference", "target"], "s_theta": [0., 2., 5.],
                           "path_b": str(source)}, index=["r1", "r2", "t"])
    items = pd.DataFrame({"bench": ["b"], "item": ["short"], "guess": [0.], "a": [1.], "b": [0.], "s_train": [True]})
    monkeypatch.setattr(tables, "load", lambda domain: (models, items))
    monkeypatch.setattr(tables, "load_semesters", lambda: {"previous": ["r1", "r2"], "s": ["t"]})
    monkeypatch.setattr(tables, "frozen_dir", lambda domain: tmp_path)

    def direct_scores(self, train, test):
        assert isinstance(self.config, DirectJudgeConfig) and self.config.include_reasoning
        assert self.config.system_prompt == "assets/system_prompt/direct.txt"
        return pd.DataFrame({"bench": "b", "item": "short", "semester": "s",
                             "model": ["r1", "r2", "t"], "score": [2., 4., 6.]})

    def pairwise_verdicts(self, train, test):
        assert not self.config.include_reasoning
        assert self.config.system_prompt == "assets/system_prompt/default.txt"
        return pd.DataFrame(columns=["bench", "item", "model_left", "model_right", "pick_left"])

    monkeypatch.setattr(JudgeDirect, "scores", direct_scores)
    monkeypatch.setattr(JudgeBT, "verdicts", pairwise_verdicts)
    monkeypatch.setattr("sys.argv", ["evaluate", str(domain_path), "--protocol", "chars",
                                    "--judge", str(judge_path), "--direct-judge", str(judge_path), "--semester", "s"])
    assert evaluate.main() == 0
    output = capsys.readouterr().out
    assert all(name in output for name in ("chars", "judge_j", "direct_judge_j"))
    prediction = pd.read_csv(tmp_path / "eval_direct_judge_j.csv")
    assert prediction.to_dict("records") == [{"semester": "s", "model": "t", "theta_hat": 4., "theta": 5.}]
