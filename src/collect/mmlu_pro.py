"""MMLU-Pro restricted to a seeded random subsample of one or more subjects.

`inspect_evals/mmlu_pro` shuffles with no seed, so `--limit n` hands every model a
different n. Here the subsample is a *task argument*: it lands in the task_args hash, so
the dir identifies the subset and every model is scored on the same items — and the run
stays a full run, which `--sample_id` would not (`is_successful_eval` rejects those).
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from inspect_evals.mmlu_pro.mmlu_pro import mmlu_pro


@task
def mmlu_pro_subset(
    subjects: str | list[str] = [],
    n: int = 150,
    seed: int = 42,
    fewshot: int = 0,
) -> Task:
    """MMLU-Pro on `n` items drawn from `subjects` with `seed`.

    Args:
        subjects: MMLU-Pro categories to keep (e.g. ``["math"]``); empty = all.
        n: Subsample size. 0 or more than the subject has keeps everything.
        seed: Seed of the draw. Changing it changes the hash, hence the data dir.
        fewshot: Few-shot examples, passed through to the underlying task.
    """
    inner = mmlu_pro(subjects=subjects, fewshot=fewshot, shuffle=False)
    samples = list(inner.dataset)
    if 0 < n < len(samples):
        shuffled = MemoryDataset(samples)
        shuffled.shuffle(seed=seed)
        inner.dataset = MemoryDataset(
            list(shuffled)[:n], name=inner.dataset.name, location=inner.dataset.location
        )
    return inner
