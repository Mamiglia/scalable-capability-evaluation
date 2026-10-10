"""Predicting further ahead: semester tasks whose references are older than the previous semester.

    uv run python -m challenges.horizon.run build            # freeze the new tasks (local, no cost)
    uv run python -m challenges.horizon.run judge [--limit N] [--task KEY ...]
    uv run python -m challenges.horizon.run transcript [--task KEY ...]   # Embedding + Deterministic features (free)

Science only. A task "<targets>_from_<references>" takes the
references' EASY/HARD split and train items exactly as the frozen task whose references they are
(same top-3, same seed), so only theta* of its references and targets is refitted, on those HARD
items. Tables and judge logs go under artifacts/prediction-horizon/, never data/meta or logs/pairwise,
so the main analysis never sees them. The judge is that frozen task's best judge on references
(paper: lowest leave-one-out error, linear variant, core condition), on 10 train items per quartile.
"""

import argparse
from pathlib import Path

import pandas as pd

from capability.task import TRAIN_COLUMNS, Cohort, Task
from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from configs.logs import logger
from protocols.pairwise.linear import JudgeLinear
from reference.freeze import local_fit, split_items, TOP_K
from reference.tables import load, load_responses, load_semesters

DOMAIN = Path("configs/domain/science.yml")
OUT = Path("artifacts/prediction-horizon")
META = OUT / "meta" / "science"
PER_QUARTILE = 10  # 40 of the frozen 100 train items
TRAIN_PER_QUARTILE = 25  # the freeze's own draw, which the new tasks must reproduce
# (reference semester, target semester, judge config, pair_max_length)
TASKS = [
    ("2025-I", "2026-I", "configs/judges/reasoning/minimax-m1.yml", 0),
    ("2024-II", "2026-I", "configs/judges/reasoning/qwen2.5-72b-instruct.yml", 50_000),
    ("2024-II", "2025-II", "configs/judges/reasoning/qwen2.5-72b-instruct.yml", 50_000),
    ("2025-I", "2024-II", "configs/judges/reasoning/minimax-m1.yml", 0),
    ("2025-I", "2024-I", "configs/judges/reasoning/minimax-m1.yml", 0),
    ("2024-I", "2025-I", "configs/judges/reasoning/gpt-4o-2024-05-13.yml", 0),
    ("2024-I", "2025-II", "configs/judges/reasoning/gpt-4o-2024-05-13.yml", 0),
    ("2024-I", "2026-I", "configs/judges/reasoning/gpt-4o-2024-05-13.yml", 0),
    ("2023", "2026-I", "configs/judges/reasoning/gpt-4-0613.yml", 15_000),  # 8k context
    ("2024-II", "2024-I", "configs/judges/reasoning/qwen2.5-72b-instruct.yml", 50_000),
    ("2024-II", "2023", "configs/judges/reasoning/qwen2.5-72b-instruct.yml", 50_000),
]


def key(references: str, targets: str) -> str:
    return f"{targets}_from_{references}"


def build() -> None:
    """Write META/{models,items}.csv: the frozen science tables plus one column set per new task. Tasks
    already in META are kept as they are (their judging used those theta*), so only new ones are fitted."""
    domain = DomainConfig.from_cli(DOMAIN)
    models, items = load(domain)
    if (META / "models.csv").exists():
        models = pd.read_csv(META / "models.csv", index_col="model")
        items = pd.read_csv(META / "items.csv", dtype={"item": str})
    responses = load_responses(domain)
    rosters = load_semesters()
    semesters = list(rosters)
    for references_semester, targets_semester, *_ in TASKS:
        name = key(references_semester, targets_semester)
        if f"{name}_role" in models:
            continue
        references, targets = rosters[references_semester], rosters[targets_semester]
        top = models.global_theta.loc[references].nlargest(TOP_K).index.tolist()
        split = split_items(responses, items, top, TRAIN_PER_QUARTILE, seed=0)
        # the frozen task with these references drew the same split: check, never re-draw
        frozen = semesters[semesters.index(references_semester) + 1]
        assert split.set.eq(items[f"{frozen}_set"].fillna("")).all(), f"{name}: EASY/HARD differs from {frozen}"
        assert split.train.eq(items[f"{frozen}_train"]).all(), f"{name}: train items differ from {frozen}"
        theta = local_fit(responses, items[split.set == "hard"], references + targets)
        role = pd.concat([pd.Series("reference", index=references), pd.Series("target", index=targets)])
        labels = role.rename("role").to_frame().join(theta)
        logger.info("%s: top %s, %d train, %d hard\n%s", name, top, int(split.train.sum()),
                    int((split.set == "hard").sum()), labels.round(2).to_string())
        items = items.join(split.add_prefix(f"{name}_"))
        models = models.join(labels.add_prefix(f"{name}_"))
    META.mkdir(parents=True, exist_ok=True)
    models.to_csv(META / "models.csv", float_format="%.4f")
    items.to_csv(META / "items.csv", index=False, float_format="%.4f")
    logger.info("wrote %s", META)


def task(name: str) -> Task:
    """A new task from META: references pinned at theta*, targets' theta* kept for the metric only."""
    models = pd.read_csv(META / "models.csv", index_col="model")
    items = pd.read_csv(META / "items.csv", dtype={"item": str})
    role = models[f"{name}_role"]
    frame = pd.DataFrame({"theta": models[f"{name}_theta"]}).join(models.filter(like="path_"))
    train = items[items[f"{name}_train"]][TRAIN_COLUMNS].reset_index(drop=True)
    references = Cohort(frame[role == "reference"], train, name)
    targets = frame[role == "target"]
    return Task(name, references, Cohort(frame.iloc[:0], train, name),
                Cohort(targets.assign(theta=float("nan")), train, name), targets.theta)


def judge(limit: int = 0, only: list[str] | None = None) -> None:
    """Judge every new task (resumable), then write its linear-variant forecasts."""
    domain = DomainConfig.from_cli(DOMAIN)
    for references_semester, targets_semester, config_path, pair_max_length in TASKS:
        name = key(references_semester, targets_semester)
        if only and name not in only:
            continue
        config = JudgeConfig.from_cli(config_path)
        config.log_dir = OUT / ("smoke" if limit else "logs") / "pairwise"
        config.train_per_quartile, config.pair_max_length, config.limit = PER_QUARTILE, pair_max_length, limit
        protocol = JudgeLinear(config, domain)
        logger.info("%s: %s, condition %s -> %s", name, config.judge.shortname, config.condition_hash,
                    protocol.log_dir(name))
        current = task(name)
        theta_hat = protocol(current).reindex(current.truth.index)
        if not limit:
            forecasts = pd.DataFrame({"task": name, "judge": config.judge.shortname, "model": current.truth.index,
                                      "theta_hat": theta_hat.to_numpy(float), "theta": current.truth.to_numpy()})
            forecasts.to_csv(OUT / f"forecasts_{name}.csv", index=False, float_format="%.4f")
            logger.info("%s: MSE %.3f\n%s", name, ((forecasts.theta_hat - forecasts.theta) ** 2).mean(),
                        forecasts.round(2).to_string())


def transcript(only: list[str] | None = None) -> None:
    """The protocols without a judge on every new task, as the paper runs them (all 100 train items, fit on the
    references): Deterministic features (glm_regex) and Embedding (full transcript). Writes
    OUT/forecasts_transcript_<task>.csv."""
    from protocols.embeddings import EmbeddingRegression
    from protocols import PROTOCOLS

    protocols = [("glm_regex", PROTOCOLS["glm_regex"]), ("embedding", EmbeddingRegression("full"))]
    for references_semester, targets_semester, *_ in TASKS:
        name = key(references_semester, targets_semester)
        if only and name not in only:
            continue
        current = task(name)
        frames = [pd.DataFrame({"task": name, "protocol": label, "model": current.truth.index,
                                "theta_hat": protocol(current).reindex(current.truth.index).to_numpy(float),
                                "theta": current.truth.to_numpy()}) for label, protocol in protocols]
        forecasts = pd.concat(frames, ignore_index=True)
        forecasts.to_csv(OUT / f"forecasts_transcript_{name}.csv", index=False, float_format="%.4f")
        mse = forecasts.assign(se=(forecasts.theta_hat - forecasts.theta) ** 2).groupby("protocol").se.mean()
        logger.info("%s: %s, missing %d", name, mse.round(2).to_dict(), int(forecasts.theta_hat.isna().sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["build", "judge", "transcript"])
    parser.add_argument("--limit", type=int, default=0, help="samples per judging task, into a smoke dir")
    parser.add_argument("--task", nargs="+", default=None, help="task keys, e.g. 2026-I_from_2025-I")
    args = parser.parse_args()
    if args.step == "build":
        build()
    elif args.step == "judge":
        judge(args.limit, args.task)
    else:
        transcript(args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
