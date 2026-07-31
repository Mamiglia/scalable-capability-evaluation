from pathlib import Path
import re
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message

from judge.dataset import build_dataset
from judge.scorer import judge_scorer
from registry.query import resolve_eval_path, resolve_eval_paths

SYSTEM_PROMPT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript A and Transcript B, both coming to the correct solution.
One of the two models is more capable than the other, and your task is to identify which is the smarter model (A or B). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response is better and why. 2-4 sentences.>
ANSWER: <A or B, indicating which response is better.>
CONFIDENCE: <An integer between 1 and 5 indicating your confidence in your answer.>
"""

def _reproduction_num(filename: str) -> int:
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0


def _resolve_transcript_spec(
    spec: int | str, available_paths: list[Path], model_name: str
) -> Path:
    filename = f"{spec}.eval" if isinstance(spec, int) else spec
    matches = [path for path in available_paths if path.name == filename]
    if len(matches) != 1:
        raise ValueError(
            f"Transcript '{filename}' for model '{model_name}' must match exactly once"
        )
    return matches[0]


def resolve_transcript_pairs(
    major_paths: list[Path],
    minor_paths: list[Path],
    mode: Literal["paired", "combinatorial", "custom"] = "paired",
    pairs: list | None = None,
    major_name: str = "major",
    minor_name: str = "minor",
) -> list[tuple[Path, Path]]:
    if mode == "paired":
        major_map = {p.stem: p for p in major_paths}
        minor_map = {p.stem: p for p in minor_paths}
        common_stems = sorted(
            set(major_map.keys()) & set(minor_map.keys()),
            key=lambda s: (_reproduction_num(s), s),
        )
        if not common_stems:
            raise ValueError("paired mode requires matching transcript filenames")
        return [(major_map[s], minor_map[s]) for s in common_stems]

    elif mode == "combinatorial":
        return [(m, n) for m in major_paths for n in minor_paths]

    elif mode == "custom":
        if not pairs:
            raise ValueError("pairs must be provided when mode is 'custom'")
        resolved_pairs = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Each pair in pairs must have exactly 2 elements, got {pair}")
            m_spec, n_spec = pair[0], pair[1]
            m_path = _resolve_transcript_spec(m_spec, major_paths, major_name)
            n_path = _resolve_transcript_spec(n_spec, minor_paths, minor_name)
            resolved_pairs.append((m_path, n_path))
        return resolved_pairs

    else:
        raise ValueError(
            f"Unknown comparison mode '{mode}'. Supported modes are: 'paired', 'combinatorial', 'custom'"
        )


@task
def judge_task(
    major: str = "",
    minor: str = "",
    benchmark_name: str = "",
    task_args: dict | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_length: int = 65536,
    require_both_correct: bool = True,
    include_reverse: bool = True,
    log_major_path: str | Path | None = None,
    log_minor_path: str | Path | None = None,
    experiment_name: str | None = None,
    task_name: str | None = None,
    dataset_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> Task:
    log_major_path = log_major_path or str(
        resolve_eval_path(
            major,
            benchmark_name,
            task_args,
            dataset_dir=Path(dataset_dir) if dataset_dir else None,
            registry_path=Path(registry_path) if registry_path else None,
        )
    )
    log_minor_path = log_minor_path or str(
        resolve_eval_path(
            minor,
            benchmark_name,
            task_args,
            dataset_dir=Path(dataset_dir) if dataset_dir else None,
            registry_path=Path(registry_path) if registry_path else None,
        )
    )
    name = task_name or f"{benchmark_name}/{major.split('/')[-1]}|{minor.split('/')[-1]}"

    return Task(
        name=name,
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=build_dataset(
            log_major_path=Path(log_major_path),
            log_minor_path=Path(log_minor_path),
            max_len=max_length,
            require_both_correct=require_both_correct,
            include_reverse=include_reverse,
        ),
        solver=[system_message(system_prompt), generate()],
        scorer=judge_scorer(),
        metadata={
            "log_major_path": str(log_major_path),
            "log_minor_path": str(log_minor_path),
            "system_prompt": system_prompt,
            "name_major": major,
            "name_minor": minor,
            "benchmark_name": benchmark_name,
            "task_args": task_args,
            "require_both_correct": require_both_correct,
            "include_reverse": include_reverse,
            "experiment_name": experiment_name,
        },
    )


def build_judge_tasks(
    major: str = "",
    minor: str = "",
    benchmark_name: str = "",
    task_args: dict | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_length: int = 65536,
    require_both_correct: bool = True,
    include_reverse: bool = True,
    experiment_name: str | None = None,
    dataset_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    mode: Literal["paired", "combinatorial", "custom"] = "paired",
    pairs: list | None = None,
) -> list[Task]:
    dataset_p = Path(dataset_dir) if dataset_dir else None
    registry_p = Path(registry_path) if registry_path else None

    major_paths = resolve_eval_paths(
        major, benchmark_name, task_args=task_args, dataset_dir=dataset_p, registry_path=registry_p
    )
    minor_paths = resolve_eval_paths(
        minor, benchmark_name, task_args=task_args, dataset_dir=dataset_p, registry_path=registry_p
    )

    matched_pairs = resolve_transcript_pairs(
        major_paths=major_paths,
        minor_paths=minor_paths,
        mode=mode,
        pairs=pairs,
        major_name=major,
        minor_name=minor,
    )

    tasks = []
    use_suffix = len(matched_pairs) > 1

    for p_major, p_minor in matched_pairs:
        if use_suffix:
            t_name = f"{benchmark_name}={major}:{p_major.stem}|{minor}:{p_minor.stem}"
        else:
            t_name = f"{benchmark_name}={major}|{minor}"

        t = judge_task(
            major=major,
            minor=minor,
            benchmark_name=benchmark_name,
            task_args=task_args,
            system_prompt=system_prompt,
            max_length=max_length,
            require_both_correct=require_both_correct,
            include_reverse=include_reverse,
            log_major_path=p_major,
            log_minor_path=p_minor,
            experiment_name=experiment_name,
            task_name=t_name,
            dataset_dir=dataset_dir,
            registry_path=registry_path,
        )
        tasks.append(t)

    return tasks
