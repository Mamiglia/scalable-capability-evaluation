from collect.runner import eval_runner


def _ok_worker():
    return 0


def _failing_worker():
    raise RuntimeError("boom")


def _nonzero_worker():
    return 1


def test_eval_runner_success():
    jobs = [(_ok_worker, (), "job_1"), (_ok_worker, (), "job_2")]
    assert eval_runner(jobs) == 0


def test_eval_runner_no_jobs():
    assert eval_runner([]) == 0


def test_eval_runner_reports_failures(capsys):
    jobs = [(_ok_worker, (), "job_1"), (_failing_worker, (), "job_2"), (_nonzero_worker, (), "job_3")]
    assert eval_runner(jobs) == 1
    out = capsys.readouterr().out
    assert "job_2" in out and "job_3" in out
    assert "boom" in out
