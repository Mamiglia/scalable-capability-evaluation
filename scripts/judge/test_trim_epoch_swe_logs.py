import sys
from datetime import datetime
from pathlib import Path

from inspect_ai.log import (EvalConfig, EvalDataset, EvalLog, EvalMetric, EvalResults, EvalSample,
                            EvalScore, EvalSpec, read_eval_log, write_eval_log)
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.scorer import Score

sys.path.insert(0, str(Path(__file__).parent))
from trim_epoch_swe_logs import trim_log  # noqa: E402


def fake_log(path: Path) -> None:
    def sample(item, correct, turns):
        messages = [ChatMessageSystem(content="sys"), ChatMessageUser(content="issue")]
        for turn in range(turns):
            messages += [ChatMessageAssistant(content=f"step {turn}"), ChatMessageTool(content="ok")]
        return EvalSample(id=item, epoch=1, input="issue", target="", messages=messages,
                          scores={"swe_bench_scorer": Score(value="C" if correct else "I")})

    samples = [sample("django__django-1", True, 5), sample("django__django-2", False, 1), sample("other", True, 1)]
    spec = EvalSpec(task="swe", dataset=EvalDataset(samples=3, sample_ids=[s.id for s in samples]),
                    model="epoch/x", config=EvalConfig(), created=datetime.now().isoformat(), run_id="r", task_id="t")
    results = EvalResults(total_samples=3, completed_samples=3, scores=[EvalScore(
        name="swe_bench_scorer", scorer="swe_bench_scorer",
        metrics={"accuracy": EvalMetric(name="accuracy", value=2 / 3)})])
    write_eval_log(EvalLog(status="success", eval=spec, results=results, samples=samples), str(path))


def test_trim_keeps_pool_items_and_cuts_messages(tmp_path):
    fake_log(tmp_path / "x.eval")
    log = trim_log(tmp_path / "x.eval", {"django__django-1", "django__django-2"}, message_limit=6)
    write_eval_log(log, str(tmp_path / "trimmed.eval"))
    back = read_eval_log(str(tmp_path / "trimmed.eval"))

    assert [s.id for s in back.samples] == ["django__django-1", "django__django-2"]
    assert len(back.samples[0].messages) == 6  # system + user + two assistant/tool rounds
    assert back.samples[0].scores["swe_bench_scorer"].value == "C"
    assert back.results.completed_samples == 2
    assert back.results.scores[0].metrics["accuracy"].value == 0.5
    assert "epoch_ai" in back.eval.tags
