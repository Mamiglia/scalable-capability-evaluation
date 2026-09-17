"""JudgeBT: an LLM judge's verdicts on every (target, reference) pair of the task, judged on demand.

The run is one inspect eval_set per <domain, semester, judge, condition>, stored under
<config.log_dir>/<domain>/<semester>/<judge>/<condition hash>/ (logs/judge/ by default): one .eval per <bench, reference,
target>, on the train items both models got right on repeat 0, in both shown orders. A sample
is `<item>_<shown position of the reference>`; the scorer's answer "1"/"2" says which shown
transcript the judge picked. Tasks already finished in the dir are not re-run (eval_set resumes),
and a dir with every task finished is only read.

Targets are compared with the references, never with each other: the references are pinned
at theta* in the Bradley-Terry fit, so those are the comparisons that place a target. References
are also compared with each other under the existing collection scheme. These verdicts do
not move the pinned Bradley-Terry fit. Historical models are not added to either set of pairs.

Logs imported from older campaigns (scripts/judge/import_judge_logs.py) sit in the same dir with
an "(imported)" task-name suffix: they never count as the finished task, and where they overlap
the run's own verdicts on an (item, pair, shown order) the run's verdict wins.
"""

from pathlib import Path

import pandas as pd
from inspect_ai import Task
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

from capability.task import Cohort
from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from configs.paths import is_successful_eval
from evals.transcripts import read_header
from judge.run import run_tasks
from judge.task import judge_task
from protocols.pairwise.protocol import PairwiseBT
from reference.tables import DATA

VERDICT_COLUMNS = ["bench", "judge", "item", "model_left", "model_right", "pick_left", "semester", "imported"]
IMPORTED = " (imported)"  # task-name suffix of a log imported from an older campaign


def from_logs(log_dir: str | Path) -> pd.DataFrame:
    """Verdict table of one run dir: bench, judge, item, model_left, model_right, pick_left, semester."""
    rows = []
    for path in sorted(Path(log_dir).glob("*.eval")):
        log = read_eval_log(str(path), header_only=True)
        if log.status != "success":
            continue
        md = log.eval.metadata or {}
        major, minor = md.get("major") or md["name_major"], md.get("minor") or md["name_minor"]
        bench = md.get("bench") or md["benchmark_name"]
        for s in read_eval_log_sample_summaries(str(path)):
            # A sample the run tolerated an error on (fail_on_error) never reached the
            # scorer, so it carries no verdict.
            score = next(iter((s.scores or {}).values()), None)
            answer = score.answer if score else None
            if answer not in ("1", "2"):
                continue
            item, target = str(s.id).rsplit("_", 1)
            left, right = (major, minor) if target == "1" else (minor, major)
            rows.append(dict(bench=bench, judge=log.eval.model, item=item, model_left=left, model_right=right,
                             pick_left=float(answer == "1"), semester=md.get("semester") or "",
                             imported=log.eval.task.endswith(IMPORTED)))
    return pd.DataFrame(rows, columns=VERDICT_COLUMNS)


def stratified_items(items: pd.DataFrame, per_quartile: int, seed: int = 0) -> set[str]:
    """The task's train items, or a seeded subset of `per_quartile` items per global-b quartile."""
    if not per_quartile:
        return set(items.item)
    quartile = pd.qcut(items.b.rank(method="first"), 4, labels=False)
    return set(items.groupby(quartile, group_keys=False).apply(lambda g: g.sample(min(per_quartile, len(g)), random_state=seed)).item)


class JudgeBT(PairwiseBT):
    def __init__(self, config: JudgeConfig, domain: DomainConfig):
        self.config, self.domain = config, domain
        self.name = f"judge_{config.judge.shortname.lower()}"

    def log_dir(self, semester: str) -> Path:
        base = self.config.log_dir
        if self.config.condition_label:  # e.g. logs/judge_truncated-head-2000, logs/judge_window-keep-2of10
            base = base.with_name(f"{base.name}_{self.config.condition_label}")
        return base / self.domain.name / semester / self.config.judge.shortname.lower() / self.config.condition_hash

    def tasks(self, train: Cohort, test: Cohort, done: dict[str, set[str]] | None = None) -> list[Task]:
        """One judging task per <bench, reference, target> and per <bench, reference, reference> pair, on the train
        items both got right. Reference–reference verdicts do not calibrate pinned BT. `done` maps a task name to
        the sample ids ("<item>_<shown order>") that already have a verdict: those samples are not judged again."""
        tasks = []
        # Bench entries sharing a task name (hle.yml / hle_gold.yml: same items under two hashes)
        # are one bench: the frozen path_<bench> already points at whichever hash has the log.
        benches = {self.domain.task(bench).name: bench for bench in self.domain.benches}
        keep = stratified_items(train.items, self.config.train_per_quartile)
        references = list(train.models.index)
        pairs = [(r, t, test) for t in test.models.index for r in references]
        pairs += [(a, b, train) for i, a in enumerate(references) for b in references[i + 1:]]
        excluded = set(self.config.exclude_models) | set(self.domain.exclude_models)
        pairs = [p for p in pairs if not {p[0], p[1]} & excluded]
        orders = ("1", "2") if self.config.include_reverse else ("1",)
        for name, bench in benches.items():
            for major, minor, cohort in pairs:
                items = {item for b, item in train.transcripts[major] if b == name}
                items &= {item for b, item in cohort.transcripts[minor] if b == name} & keep
                judged = (done or {}).get(f"{name}/{major}|{minor}", set())
                items = {item for item in items if not all(f"{item}_{order}" in judged for order in orders)}
                if not items:
                    continue
                task = judge_task(
                    DATA / train.models.loc[major, f"path_{name}"], DATA / cohort.models.loc[minor, f"path_{name}"],
                    system_prompt=bench.system_prompt or self.config.system_prompt,
                    max_length=self.config.max_length,
                    include_reverse=self.config.include_reverse,
                    include_reasoning=self.config.include_reasoning,
                    truncate=self.config.truncate, window=self.config.window, channels=self.config.channels,
                    task_name=f"{name}/{major}|{minor}",
                    items=items,
                    # Data-dir model names, so from_logs reads logs alone.
                    metadata={"domain": self.domain.name, "bench": name, "major": major, "minor": minor,
                              "semester": test.semester},
                )
                if task is not None and judged & {f"{item}_{order}" for item in items for order in orders}:
                    # one shown order of an item already has a verdict: judge only the order without one
                    task.dataset = task.dataset.filter(lambda sample, judged=judged: sample.id not in judged)
                if task is not None:
                    tasks.append(task)
        return tasks

    def judged_samples(self, log_dir: Path) -> dict[str, set[str]]:
        """Task name -> the sample ids ("<item>_<shown order>") with a verdict in the run dir's successful logs. A
        sample the run tolerated an error on has no verdict, so it is left out and judged again."""
        judged: dict[str, set[str]] = {}
        for p in log_dir.glob("*.eval"):
            if is_successful_eval(p):  # imported names differ, so they never count
                judged.setdefault(read_header(p).eval.task, set()).update(
                    str(s.id) for s in read_eval_log_sample_summaries(str(p)) if not s.error)
        return judged

    def judge(self, train: Cohort, test: Cohort) -> Path:
        """The run dir for this task, judging whatever (pair, item, shown order) is not judged in it yet."""
        log_dir = self.log_dir(test.semester)
        # Resume at sample level: only the shown orders of a pair's items without a verdict in the dir are judged,
        # into a further log of the same task name (from_logs pools them). So a finished pair is never re-judged
        # after an eval_set kwarg tweak, a refrozen train set only adds its new items, and an order that errored
        # is judged again on its own.
        todo = self.tasks(train, test, self.judged_samples(log_dir))
        if todo:
            if not run_tasks(todo, self.config, log_dir):
                raise RuntimeError(f"judge run failed in {log_dir}")
        return log_dir

    def verdicts(self, train: Cohort, test: Cohort) -> pd.DataFrame:
        verdicts = from_logs(self.judge(train, test))
        # the run's own verdict wins over an imported one on the same (item, pair, shown order)
        verdicts = verdicts.sort_values("imported", kind="stable")
        return verdicts.drop_duplicates(["bench", "item", "model_left", "model_right"], keep="first")
