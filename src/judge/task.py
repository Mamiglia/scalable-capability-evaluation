from pathlib import Path

from inspect_ai import Task
from inspect_ai.solver import generate, system_message

from configs.experiment import ExperimentConfig, successful_evals
from configs.logs import logger
from judge.dataset import MAX_LENGTH, build_dataset, read_header
from judge.scorer import judge_scorer

DEFAULT_SYSTEM_PROMPT = "assets/system_prompt/default.txt"
# Same prompt plus a note that both transcripts stop mid-reasoning on purpose.
TRUNCATED_SYSTEM_PROMPT = "assets/system_prompt/truncated.txt"


def judge_task(
    major_path: str | Path,
    minor_path: str | Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_length: int = MAX_LENGTH,
    require_both_correct: bool = True,
    include_reverse: bool = True,
    include_reasoning: bool = True,
    truncate_frac: float | None = None,
    experiment_name: str | None = None,
    task_name: str | None = None,
    items: set[str] | None = None,
    metadata: dict | None = None,
) -> Task | None:
    """Build the judging task that compares one major eval log against one minor log.

    Returns None when no question survives the filters (e.g. both models never agree on a
    correct answer), since inspect refuses an empty dataset."""
    major_path = Path(major_path)
    minor_path = Path(minor_path)

    log_major = read_header(major_path)
    log_minor = read_header(minor_path)

    if log_major.eval.task.rsplit("/", 1)[-1] != log_minor.eval.task.rsplit("/", 1)[-1]:
        raise ValueError(
            f"The two logs must be from the same task, got {log_major.eval.task!r} "
            f"({major_path}) and {log_minor.eval.task!r} ({minor_path})"
        )

    benchmark_name = log_major.eval.task
    name = task_name or f"{benchmark_name}={major_path.stem}|{minor_path.stem}"

    if truncate_frac is not None and system_prompt == DEFAULT_SYSTEM_PROMPT:
        # Truncated transcripts stop mid-reasoning, so the judge has to be told not to
        # read that as a failure.
        system_prompt = TRUNCATED_SYSTEM_PROMPT

    dataset = build_dataset(
        major_path=major_path,
        minor_path=minor_path,
        max_len=max_length,
        require_both_correct=require_both_correct,
        include_reverse=include_reverse,
        include_reasoning=include_reasoning,
        truncate_frac=truncate_frac,
        items=items,
    )
    if len(dataset) == 0:
        logger.warning("%s: no samples after filtering (%s vs %s), skipping", name, major_path, minor_path)
        return None

    return Task(
        name=name,
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=dataset,
        solver=[system_message(system_prompt), generate()],
        scorer=judge_scorer(),
        metadata={
            "major_path": str(major_path),
            "minor_path": str(minor_path),
            "system_prompt": system_prompt,
            "name_major": log_major.eval.model,
            "name_minor": log_minor.eval.model,
            "benchmark_name": benchmark_name,
            "require_both_correct": require_both_correct,
            "include_reverse": include_reverse,
            "include_reasoning": include_reasoning,
            "truncate_frac": truncate_frac,
            "max_length": max_length,
            "experiment_name": experiment_name,
            **(metadata or {}),
        },
    )


def build_tasks(experiment: ExperimentConfig) -> list[Task]:
    """One judging task per <bench of the domain, pair of models>, on the experiment's items."""
    exp = experiment
    items = exp.items()
    acc = exp.truth()
    # Which items each log got right is already in the domain's response table; asking the
    # logs again would mean streaming every one of them (minutes each for HLE).
    correct = exp.domain.load_responses().query("correct")
    tasks = []
    for bench in exp.domain.benches:
        task_cfg = exp.domain.task(bench)
        if task_cfg.name not in items:
            logger.warning("%s: no items to judge on %s", exp.name, task_cfg.name)
            continue
        logs = {m.shortname.lower(): successful_evals(task_cfg, m) for m in exp.models}
        solved = correct[correct.bench == task_cfg.name].groupby(["model", "repeat"]).item.agg(set)
        for major, minor in exp.ordered_pairs():
            names = major.shortname.lower(), minor.shortname.lower()
            if not all(logs[n] for n in names):
                logger.warning("%s: no logs on %s for %s, skipping pair", exp.name, task_cfg.name,
                               [n for n in names if not logs[n]])
                continue
            # Items both models solved in the (first) repeat being judged.
            both = items[task_cfg.name].intersection(
                *(solved.get((n, int(logs[n][0].parent.name)), set()) for n in names))
            if not both:
                logger.warning("%s: %s|%s share no correct item on %s, skipping", exp.name, *names, task_cfg.name)
                continue
            task = judge_task(
                logs[names[0]][0], logs[names[1]][0],
                system_prompt=bench.system_prompt or exp.system_prompt,
                max_length=exp.max_length,
                include_reverse=exp.include_reverse,
                include_reasoning=exp.include_reasoning,
                truncate_frac=exp.truncate_frac or None,
                experiment_name=exp.name,
                task_name=f"{task_cfg.name}/{names[0]}|{names[1]}",
                items=both,
                # Data-dir model names and the ground truth, so ranking.verdicts reads logs alone.
                metadata={"domain": exp.domain.name, "bench": task_cfg.name, "major": names[0],
                          "minor": names[1], "semester": exp.semester, "q_src": exp.q_src, "q_dst": exp.q_dst,
                          "acc_major": float(acc[names[0]]), "acc_minor": float(acc[names[1]])},
            )
            if task is not None:
                tasks.append(task)
    if not tasks:
        raise ValueError(f"{exp.name}: nothing to judge")
    return tasks
