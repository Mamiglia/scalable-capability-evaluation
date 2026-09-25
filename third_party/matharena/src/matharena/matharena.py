r"""Inspect tasks for MathArena's final-answer competitions.

MathArena: Evaluating LLMs on Uncontaminated Math Competitions
Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, Martin Vechev
https://arxiv.org/abs/2505.23281

Reproduces MathArena's own pipeline (https://github.com/eth-sri/matharena, MIT,
(c) 2024 SRI Lab, ETH Zurich — see LICENSE in this directory): the competition's
`instruction` prepended to the problem as a single user message, one generation
plus a single "last chance" reprompt when no parseable \boxed{} answer is found,
and purely symbolic grading via the vendored `matharena.parser`. No LLM grader.
`_PromptFields`, `last_chance_prompt` and the core of `extract_and_grade` are
ported verbatim from upstream (pure_model_solver.py, runner.py, grader.py).

One task is defined per final-answer competition config under `competitions/`,
named after its HF dataset:

    inspect eval matharena/aime_2025
    inspect eval matharena/apex_2025 -T last_chance=false

Proof, Lean, tool-using (euler) and multiple-choice image (kangaroo) competitions
are not registered: their grading needs MathArena's judge/tool stack.
"""

import logging
import re
import sys
from collections import UserDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message

from matharena.parser import (
    check_answers,
    extract_answer,
    extract_boxed_answer,
    parse_answer,
)

COMPETITIONS = Path(__file__).resolve().parent / "competitions"
EVAL_VERSION: str = yaml.safe_load((Path(__file__).parent / "eval.yaml").read_text())[
    "version"
]
logger = logging.getLogger(__name__)

# Models sometimes box astronomically large integers; grading str()s them.
sys.set_int_max_str_digits(0)

# Pinned HF dataset revisions (commit SHAs). Rolling datasets (arxivmath) keep
# growing upstream; pass `revision="main"` to a task to follow them.
HF_REVISIONS: dict[str, str] = {
    "MathArena/aime_2024_I": "ea5b061c3e8039dc9858defaafc407d04b995e9f",
    "MathArena/aime_2024_II": "29d5d31e9b46e215fc24d9b2a3047506823dd101",
    "MathArena/aime_2025": "c94da77eb22bbd6439e62a323bec18493a421302",
    "MathArena/aime_2026": "d2de22f3c656b4f56cf8981212186377d1e23bc3",
    "MathArena/apex_2025": "ac8a641db12cc87be39e61ea89f2e04c80c5f2e7",
    "MathArena/apex-shortlist": "f3efdf224ef665f129ddaae37699f6098c65781b",
    "MathArena/arxivmath": "f0a9cf83259855c31b37168eca646ec6f0e2598c",
    "MathArena/arxivmath-0426": "5e58127fba079b6db1c5aef0fee5dc61fc43da8d",
    "MathArena/arxivmath-1225": "90c60aa1eadbb72168640169d82154b079cdeaac",
    "MathArena/arxivmath-0226": "a038fd1b01405ea1102e165b6ef43693fd16ca6b",
    "MathArena/arxivmath-0126": "bac939ba603c93f905aa48072980ea681a2824cd",
    "MathArena/arxivmath-0326": "14ff32983ae4c72b75fb64eba750f3c27f46607a",
    "MathArena/arxivmath-0526": "69f701c7f266bfba235ef302ff9b19093f7455bb",
    "MathArena/brumo_2025": "6de9bc0987f5710c6376ee980e1483dbeb55cfda",
    "MathArena/cmimc_2025": "8f27517ce4ad1d05e023c0e7eca4344754ba2a04",
    "MathArena/hmmt_feb_2024": "b3cf7fb48814f76142d0847003a5aac0876f3bf4",
    "MathArena/hmmt_feb_2025": "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
    "MathArena/hmmt_feb_2026": "02fba4f74d8e68e73e66a02d540fd979c05c274c",
    "MathArena/hmmt_nov_2025": "118dbfb45c4c9467c672268ed55166642897aa46",
    "MathArena/smt_2025": "d8e99d83f1583842f46d4953cd960eea5d887e45",
}


def competition_task(
    competition: str,
    last_chance: bool = True,
    system_prompt: str | None = None,
    instruction: str | None = None,
    revision: str | None = None,
) -> Task:
    """Task for a competition named by its upstream config path, e.g. 'aime/aime_2025'.

    Args:
        competition: Upstream config path under `competitions/`.
        last_chance: Reprompt once when no answer parses (MathArena's default). Off
            grades the first response as-is.
        system_prompt: Optional system message. MathArena's pure-model runs send none,
            so setting this departs from their pipeline.
        instruction: Override the competition's instruction template (equivalent to
            MathArena's per-model `custom_instructions`). `{problem}` is appended
            if the template doesn't place it.
        revision: HF dataset revision. Defaults to the pinned commit in
            `HF_REVISIONS`; `"main"` follows upstream (relevant for rolling sets).
    """
    config = load_competition(competition)
    if not _supported(config):
        raise ValueError(
            f"{competition} needs MathArena's judge/tool stack; this bridge only "
            "reproduces the text-only final-answer pipeline"
        )
    dataset_path = config["dataset_path"]
    name = dataset_path.split("/")[-1].replace("-", "_")

    # Prompt assembly verbatim from matharena.runner.Runner.run.
    template = instruction if instruction is not None else config["instruction"]
    if "{problem}" not in template:
        template += "\n\n{problem}"

    def record_to_sample(record: dict[str, Any]) -> Sample:
        # Aggregate datasets (arxivmath) restart problem_idx per sub-competition
        # and carry a `competition` column; plain problem_idx elsewhere, matching
        # the problem ids in MathArena's published rollouts.
        sample_id = str(record["problem_idx"])
        if "competition" in record:
            sample_id = f"{record['competition']}/{sample_id}"
        return Sample(
            id=sample_id,
            input=template.format_map(_PromptFields({"problem": record["problem"]})),
            target=str(record["answer"]),
            metadata={"competition": competition, "problem": record["problem"]},
        )

    dataset = hf_dataset(
        dataset_path,
        split="train",
        revision=revision or HF_REVISIONS[dataset_path],
        sample_fields=record_to_sample,
    )
    # Warning, not an error: rolling datasets (e.g. arxivmath) outgrow their config.
    if "n_problems" in config and len(dataset) != config["n_problems"]:
        logger.warning(
            f"{dataset_path}: config says {config['n_problems']} problems, "
            f"dataset has {len(dataset)}"
        )

    solvers: list[Solver] = [
        matharena_solver(options=config.get("options"), last_chance=last_chance)
    ]
    if system_prompt is not None:
        solvers.insert(0, system_message(system_prompt))

    return Task(
        name=name,
        dataset=dataset,
        solver=solvers,
        scorer=matharena_scorer(config),
        metadata=_rolling_metadata(sample.id for sample in dataset),
        version=EVAL_VERSION,
    )


def _rolling_metadata(sample_ids: Iterable[Any]) -> dict[str, str] | None:
    """For rolling aggregates (arxivmath): the newest sub-competition, informatively.

    Aggregate sample ids are prefixed with the row's sub-competition config path
    (e.g. 'arxiv/february/3'); that config's `date` dates the rolling addition.
    Returns None for fixed competitions (plain ids).
    """
    subs = {str(i).rsplit("/", 1)[0] for i in sample_ids if "/" in str(i)}
    if not subs:
        return None
    dates: dict[str, str] = {}
    for sub in subs:
        try:
            date = load_competition(sub).get("date")
        except FileNotFoundError:
            date = None
        if isinstance(date, str):
            dates[sub] = date
    if not dates:
        return {"latest_addition": max(subs)}
    latest = max(dates, key=lambda sub: dates[sub])
    return {"latest_addition": latest, "latest_addition_date": dates[latest]}


class _PromptFields(UserDict[str, str]):
    """Verbatim from matharena.solvers.pure_model_solver (leaves stray {braces} intact)."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def last_chance_prompt(options: list[str] | None) -> str:
    """Verbatim from matharena.runner.Runner._build_last_chance_prompt."""
    prompt = """
        Your last message does not provide a final answer in a way that follows the formatting instructions.
        Please based on the conversation history, report the final answer again within \\boxed{}.
        If you did not find the answer, please use \\boxed{None}.
        Do not reason about the problem again or use tools, simply try to extract the final answer from the previous reasoning.
        Boxed answers in thinking/reasoning stages will be ignored; only the final response message is considered.
        """
    if options is not None:
        option_str = ", ".join([opt for opt in options])
        prompt += f"""
            Recall that this is a multiple choice problem, and the only valid answers to put within \\boxed{{}} are: {option_str}."""
    return "\n".join([line.strip() for line in prompt.split("\n")])


@solver
def matharena_solver(
    options: list[str] | None = None, last_chance: bool = True
) -> Solver:
    """Generate once; reprompt once (MathArena's "last chance") if no answer parses."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        # Reprompt condition verbatim from matharena.runner.Runner.run.
        answer, _ = extract_answer(state.output.completion, True)
        valid = answer is not None and (options is None or str(answer) in options)
        if last_chance and not valid:
            state.messages.append(ChatMessageUser(content=last_chance_prompt(options)))
            state = await generate(state)
        return state

    return solve


def extract_and_grade(
    last_message: str, gold_answer: str, competition_config: dict[str, Any]
) -> tuple[Any, bool, Any]:
    """matharena.grader.extract_and_grade, minus the lean/hash/review-flag branches.

    Returns (model_answer, is_correct, warning).
    """
    use_strict_parsing = competition_config.get("strict_parsing", False)
    use_exact_match = competition_config.get("exact_match_parsing", False)
    use_typed_delimiters = competition_config.get("typed_delimited_answers", True)

    gold_answer_is_list = "," in gold_answer

    if use_exact_match:
        # Upstream annotates extract_boxed_answer as returning str; it returns a tuple.
        boxed: Any = extract_boxed_answer(last_message, list_answer=gold_answer_is_list)
        model_answer, warning = boxed
        gold_is_int = re.fullmatch(r"\s*-?\d+\s*", str(gold_answer)) is not None

        def _normalize_exact_match(s: str) -> str:
            normalized = s.lower()
            if gold_is_int:
                normalized = re.sub(r"[a-z]", "", normalized)
                normalized = re.sub(r"[^0-9-]+", "", normalized)
            else:
                normalized = re.sub(r"[^a-z0-9]+", "", normalized)
            return normalized

        if model_answer is None:
            is_correct = False
        else:
            is_correct = _normalize_exact_match(
                str(model_answer)
            ) == _normalize_exact_match(str(gold_answer))
    else:
        model_answer, warning = extract_answer(
            last_message,
            strict_parsing=use_strict_parsing,
            parse=True,
            list_answer=gold_answer_is_list,
            typed_delimiters=use_typed_delimiters,
        )
        typed_gold_answer, _ = parse_answer(
            gold_answer,
            list_answer=gold_answer_is_list,
            typed_delimiters=use_typed_delimiters,
        )
        is_correct = bool(check_answers(model_answer, typed_gold_answer))  # type: ignore[no-untyped-call]

    return model_answer, is_correct, warning


@scorer(metrics=[accuracy(), stderr()])
def matharena_scorer(competition_config: dict[str, Any]) -> Scorer:
    """Symbolic grading of the last assistant message, as in MathArena."""

    async def score(state: TaskState, target: Target) -> Score:
        model_answer, is_correct, warning = extract_and_grade(
            state.output.completion, target.text, competition_config
        )
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=str(model_answer),
            metadata={"parse_warning": warning.name},
        )

    return score


def load_competition(competition: str) -> dict[str, Any]:
    """Config for a competition named by its upstream path, e.g. 'aime/aime_2025'."""
    config: dict[str, Any] = yaml.safe_load(
        (COMPETITIONS / f"{competition}.yaml").read_text()
    )
    return config


def _supported(config: dict[str, Any]) -> bool:
    return bool(
        config.get("final_answer", True)
        and not config.get("lean")
        and not config.get("tools")  # euler needs a code-execution sandbox
        and not config.get("options")  # kangaroo is multiple-choice over images
        and config.get("dataset_path")
    )


# One task per supported competition. Explicit defs (rather than a registration
# loop) because inspect discovers tasks by statically parsing @task decorators.
# tests/matharena/test_matharena.py fails if a config re-sync leaves these stale.


@task
def aime_2024_I(**kwargs: Any) -> Task:
    """MathArena aime/aime_2024_I (MathArena/aime_2024_I)."""
    return competition_task("aime/aime_2024_I", **kwargs)


@task
def aime_2024_II(**kwargs: Any) -> Task:
    """MathArena aime/aime_2024_II (MathArena/aime_2024_II)."""
    return competition_task("aime/aime_2024_II", **kwargs)


@task
def aime_2025(**kwargs: Any) -> Task:
    """MathArena aime/aime_2025 (MathArena/aime_2025)."""
    return competition_task("aime/aime_2025", **kwargs)


@task
def aime_2026(**kwargs: Any) -> Task:
    """MathArena aime/aime_2026 (MathArena/aime_2026)."""
    return competition_task("aime/aime_2026", **kwargs)


@task
def apex_2025(**kwargs: Any) -> Task:
    """MathArena apex/apex_2025 (MathArena/apex_2025)."""
    return competition_task("apex/apex_2025", **kwargs)


@task
def apex_shortlist(**kwargs: Any) -> Task:
    """MathArena apex/shortlist_2025 (MathArena/apex-shortlist)."""
    return competition_task("apex/shortlist_2025", **kwargs)


@task
def arxivmath(**kwargs: Any) -> Task:
    """MathArena arxiv/all (MathArena/arxivmath)."""
    return competition_task("arxiv/all", **kwargs)


@task
def arxivmath_0426(**kwargs: Any) -> Task:
    """MathArena arxiv/april (MathArena/arxivmath-0426)."""
    return competition_task("arxiv/april", **kwargs)


@task
def arxivmath_1225(**kwargs: Any) -> Task:
    """MathArena arxiv/december (MathArena/arxivmath-1225)."""
    return competition_task("arxiv/december", **kwargs)


@task
def arxivmath_0226(**kwargs: Any) -> Task:
    """MathArena arxiv/february (MathArena/arxivmath-0226)."""
    return competition_task("arxiv/february", **kwargs)


@task
def arxivmath_0126(**kwargs: Any) -> Task:
    """MathArena arxiv/january (MathArena/arxivmath-0126)."""
    return competition_task("arxiv/january", **kwargs)


@task
def arxivmath_0326(**kwargs: Any) -> Task:
    """MathArena arxiv/march (MathArena/arxivmath-0326)."""
    return competition_task("arxiv/march", **kwargs)


@task
def arxivmath_0526(**kwargs: Any) -> Task:
    """MathArena arxiv/may (MathArena/arxivmath-0526)."""
    return competition_task("arxiv/may", **kwargs)


@task
def brumo_2025(**kwargs: Any) -> Task:
    """MathArena brumo/brumo_2025 (MathArena/brumo_2025)."""
    return competition_task("brumo/brumo_2025", **kwargs)


@task
def cmimc_2025(**kwargs: Any) -> Task:
    """MathArena cmimc/cmimc_2025 (MathArena/cmimc_2025)."""
    return competition_task("cmimc/cmimc_2025", **kwargs)


@task
def hmmt_feb_2024(**kwargs: Any) -> Task:
    """MathArena hmmt/hmmt_feb_2024 (MathArena/hmmt_feb_2024)."""
    return competition_task("hmmt/hmmt_feb_2024", **kwargs)


@task
def hmmt_feb_2025(**kwargs: Any) -> Task:
    """MathArena hmmt/hmmt_feb_2025 (MathArena/hmmt_feb_2025)."""
    return competition_task("hmmt/hmmt_feb_2025", **kwargs)


@task
def hmmt_feb_2026(**kwargs: Any) -> Task:
    """MathArena hmmt/hmmt_feb_2026 (MathArena/hmmt_feb_2026)."""
    return competition_task("hmmt/hmmt_feb_2026", **kwargs)


@task
def hmmt_nov_2025(**kwargs: Any) -> Task:
    """MathArena hmmt/hmmt_nov_2025 (MathArena/hmmt_nov_2025)."""
    return competition_task("hmmt/hmmt_nov_2025", **kwargs)


@task
def smt_2025(**kwargs: Any) -> Task:
    """MathArena smt/smt_2025 (MathArena/smt_2025)."""
    return competition_task("smt/smt_2025", **kwargs)
