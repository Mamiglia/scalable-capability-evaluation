# scripts/analyses/direct-judge/test_score_task.py
"""Checks for the scoring task: parsing, dataset shape, end-to-end with a mock judge."""
from pathlib import Path

from inspect_ai.log import read_eval_log

from score_task import parse_score_output

ROOT = Path(__file__).resolve().parents[3]


def test_parse_score_output():
    s, c, e = parse_score_output("EXPLANATION: Clean derivation.\nSCORE: 72\nCONFIDENCE: 4")
    assert (s, c, e) == (72, 4, "Clean derivation.")
    assert parse_score_output("**SCORE:** 88")[0] == 88
    assert parse_score_output("SCORE: 105")[0] is None   # out of range
    assert parse_score_output("no score here") == (None, None, None)
    assert parse_score_output("") == (None, None, None)


def _small_epoch_log():
    # any epoch log works; take the smallest gpqa one so the test stays fast
    logs = sorted((ROOT / "external_data" / "epoch-gpqa").glob("*.eval"),
                  key=lambda p: p.stat().st_size)
    return logs[0]


def test_build_score_dataset():
    from judge.util import extract_score
    from score_task import build_score_dataset

    log_path = _small_epoch_log()
    log = read_eval_log(str(log_path))
    ids = list(dict.fromkeys(s.id for s in log.samples if extract_score(s)))[:2]
    assert ids, f"no correct samples in {log_path.name}"

    ds = build_score_dataset(log_path, ids, max_len=2000, include_reasoning=False)
    assert [s.id for s in ds] == ids
    assert "Transcript:" in ds[0].input
    assert "Rate the capability" in ds[0].input
    assert "Transcript 2" not in ds[0].input  # single-transcript prompt, no pair leftovers


def test_score_task_end_to_end(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model, ModelOutput
    from judge.util import extract_score
    from score_task import score_task

    log_path = _small_epoch_log()
    log = read_eval_log(str(log_path))
    ids = list(dict.fromkeys(s.id for s in log.samples if extract_score(s)))[:2]

    reply = "EXPLANATION: Competent but routine.\nSCORE: 72\nCONFIDENCE: 3"
    judge = get_model("mockllm/model", custom_outputs=[
        ModelOutput.from_content("mockllm/model", reply) for _ in ids])
    result = inspect_eval(score_task(log_path, ids), model=judge,
                          log_dir=str(tmp_path), display="none")[0]
    assert result.status == "success"
    for s in result.samples:
        sc = list(s.scores.values())[0]
        assert sc.value == 72
        assert sc.metadata["confidence"] == 3


def test_induced_picks():
    from analyse import induced_picks

    scores = {("strong", "q1"): 80, ("weak", "q1"): 60,   # stronger wins
              ("strong", "q2"): 70, ("weak", "q2"): 70,   # tie
              ("strong", "q3"): 50, ("weak", "q3"): 65}   # stronger loses
    rows = induced_picks(scores, [["strong", "weak"]],
                         {"strong-vs-weak": ["q1", "q2", "q3", "q4"]})  # q4 unscored -> dropped
    assert [r["picked_major"] for r in rows] == [1.0, 0.5, 0.0]
    assert [r["tie"] for r in rows] == [False, True, False]
    assert len(rows) == 3
