"""Any registered task restricted to a fixed list of sample ids.

`--sample_id` logs never count as collected data (`is_successful_eval`), and harbor's own
`dataset_task_names` cannot be combined with its package ref. Here the id list is a *task
argument*: it lands in the task_args hash, so the dir identifies the subset and the run stays a
full run (src/collect/tasks/mmlu_pro.py precedent). Ids, prompts, images and scorer are those of
the inner task, so rows pool with the full-bench hash when both are listed in a domain yml.
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from configs.paths import resolve_task_fn


@task
def subset(task: str = "", ids: list[str] = [], **task_args: Any) -> Task:
    """`task` (registry name or file.py@name) on the samples whose id is in `ids` (empty = all)."""
    if not task:  # configs.paths.resolve_task_fn instantiates with no args just to register the task
        return Task()
    inner = resolve_task_fn(task)(**task_args)
    if not ids:
        return inner
    wanted = set(ids)
    samples = [s for s in inner.dataset if str(s.id) in wanted]
    missing = wanted - {str(s.id) for s in samples}
    if missing:
        raise ValueError(f"ids not in {task}: {sorted(missing)}")
    inner.dataset = MemoryDataset(samples, name=inner.dataset.name, location=inner.dataset.location)
    # The @task wrapper retags the returned instance as "subset"; pin the inner name first so the
    # log's eval.task (and file name) stay the bench's, as judge/task.py pairs logs by it.
    inner._name = inner.name
    return inner
