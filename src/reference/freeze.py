"""Freeze a domain: the per-domain tables every benchmark-free protocol is evaluated on.

    uv run reference configs/domain/science.yml [--train 25] [--force]

Written once to data/meta/<domain>/ and committed there (a refit is `--force` + a new commit,
only once the domain's collection is complete). `build` runs the steps in order:

1. RESPONSES  every successful log of the domain -> one row per (bench, model, item, repeat)
              with `correct` and `guess` (evals.responses). No text.
2. GLOBAL IRT 2PL with guessing floor on every response -> items: a, b, guess; models: global_theta.
3. TASKS      one per semester of configs/semesters.yml from the second on (2023 is one):
              references = the previous semester's roster, targets = this one's. Per task:
   a. EASY / HARD: the TOP_K references by global theta define them on their repeat-0 answers —
      EASY = all correct, HARD = all answered but some wrong, "" = not all answered (a bench that
      generation never ran), out of the task. The train set is a fixed, seeded subset of EASY
      (--train N per quartile of global b; 0 = all EASY), item-level so identical for every model.
   b. LOCAL IRT: theta* of references + targets on the HARD items with a, b, guess fixed to the
      global fit. Older models are fitted separately on the same HARD set, preserving the original
      reference/target fit. Historical models without HARD responses are reported and left unlabelled.
      Every task's theta* is on the global scale; another task's labels are never substituted.
4. WRITE      models.csv, items.csv, responses.csv (the fit input), fit.yml (settings + provenance).

The frozen columns a consumer needs are listed in reference.tables, which reads them back.
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from configs.domain import DomainConfig
from configs.logs import logger
from evals.responses import eval_logs, repeat0_paths, responses as read_responses
from irt.data import encode
from irt.fitting import fit
from reference.tables import DATA, frozen_dir, load, load_semesters

TOP_K = 3  # reference models that define the EASY set


# ----------------------------------------------------------------------------- 2. global IRT

def posterior_summary(draws, name: str) -> pd.DataFrame:
    """Mean and sd of posterior draws (samples x n) as columns `name`, `name_sd`."""
    draws = np.asarray(draws)
    return pd.DataFrame({name: draws.mean(0), f"{name}_sd": draws.std(0)})


def global_fit(responses: pd.DataFrame, warmup=1000, samples=1000, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2PL with guessing floor on every response: all models, all items, all repeats.

    Returns
        models: indexed by model — global_theta, global_theta_sd
        items:  bench, item, guess, a, a_sd, b, b_sd
    """
    data = encode(responses)
    # The guessing floor is a property of the item (1/n_choices on multiple choice, else 0).
    # A few imported logs lack the `choices` field, so take the max over the item's responses.
    guess_per_item = responses.groupby(["bench", "item"]).guess.max()
    guess = np.array([guess_per_item[item] for item in data.items])

    posterior = fit(data, model="2pl", warmup=warmup, samples=samples, seed=seed, guess=guess)

    models = posterior_summary(posterior["theta"], "global_theta")
    models.index = pd.Index(data.models, name="model")
    items = pd.DataFrame(data.items, columns=["bench", "item"])
    items["guess"] = guess
    items[["a", "a_sd"]] = posterior_summary(posterior["a"], "a")
    items[["b", "b_sd"]] = posterior_summary(posterior["b"], "b")
    return models, items


# ----------------------------------------------------------------------------- 3. one semester task

def roster(semesters: dict[str, list[str]], semester: str, collected: set[str]) -> tuple[list[str], list[str]]:
    """(references, targets) of a semester task: the previous roster and this one, collected models only."""
    previous = list(semesters)[list(semesters).index(semester) - 1]
    references = [m for m in semesters[previous] if m in collected]
    targets = [m for m in semesters[semester] if m in collected]
    missing = sorted(set(semesters[previous] + semesters[semester]) - collected)
    if missing:
        logger.warning("%s: not collected: %s", semester, missing)
    if not references or not targets:  # a generation this domain was never run on (coding: pre-2025)
        logger.warning("%s: no task, roster not collected on this domain", semester)
        return [], []
    if len(references) < 2 or len(targets) < 2:
        raise ValueError(f"{semester}: {len(references)} references / {len(targets)} targets")
    return references, targets


def easy_hard(responses: pd.DataFrame, items: pd.DataFrame, top: list[str]) -> pd.Series:
    """Per item (aligned with `items`): "easy", "hard" or "", from the `top` models' repeat-0 answers.

    easy: every top model answered it correctly. hard: every top model answered, some wrongly.
    "":   some top model has no repeat-0 answer (its generation never ran that bench) -> out of the task.
    """
    top_repeat0 = responses[(responses.repeat == 0) & responses.model.isin(top)]
    per_item = top_repeat0.groupby(["bench", "item"]).agg(n_answered=("model", "nunique"), all_correct=("correct", "min"))
    answered_by_all = per_item[per_item.n_answered == len(top)]
    status = pd.Series(np.where(answered_by_all.all_correct, "easy", "hard"), index=answered_by_all.index)
    item_keys = pd.MultiIndex.from_frame(items[["bench", "item"]])
    return pd.Series(item_keys.map(status).fillna(""), index=items.index, name="set")


def train_subset(items: pd.DataFrame, easy: pd.Series, per_quartile: int, seed: int) -> pd.Series:
    """Per item (bool): the train set = `per_quartile` EASY items from each quartile of global b (0 = all EASY).

    Drawn once per task with a fixed seed, so every model is judged on the same items.
    """
    if not per_quartile:
        return easy.rename("train")
    candidates = items[easy]
    quartile = pd.qcut(candidates.b.rank(method="first"), 4, labels=False)
    sample = lambda group: group.sample(min(per_quartile, len(group)), random_state=seed)
    kept = candidates.groupby(quartile, group_keys=False).apply(sample, include_groups=False)
    return pd.Series(items.index.isin(kept.index), index=items.index, name="train")


def split_items(responses: pd.DataFrame, items: pd.DataFrame, top: list[str], train: int = 25,
                seed: int = 0) -> pd.DataFrame:
    """easy_hard + train_subset: one row per item, aligned with `items`: set, train."""
    status = easy_hard(responses, items, top)
    n_out = int((status == "").sum())
    if n_out:
        logger.info("  %d items lack a repeat-0 answer from a top model -> out of the task", n_out)
    return pd.DataFrame({"set": status, "train": train_subset(items, status == "easy", train, seed)})


def local_fit(responses: pd.DataFrame, hard: pd.DataFrame, models: list[str], warmup=1000,
              samples=1000, seed=0) -> pd.DataFrame:
    """theta* of `models` on the HARD items, with a, b, guess fixed to the global fit.

    `hard` is the HARD rows of the global item table (bench, item, a, b, guess). Only theta is
    sampled, so every semester's theta* lands on the global scale.

    Returns indexed by model — theta, theta_sd.
    """
    hard_responses = responses[responses.model.isin(models)].merge(hard[["bench", "item"]])
    unseen = set(models) - set(hard_responses.model)
    if unseen:
        raise ValueError(f"no HARD responses for {sorted(unseen)}: their theta would just be the prior")

    data = encode(hard_responses)
    params = hard.set_index(["bench", "item"]).loc[data.items]  # in the encoded item order
    posterior = fit(data, model="fixed_2pl", warmup=warmup, samples=samples, seed=seed,
                    a=params.a.to_numpy(), b=params.b.to_numpy(), guess=params.guess.to_numpy())
    theta = posterior_summary(posterior["theta"], "theta")
    theta.index = pd.Index(data.models, name="model")
    return theta


def semester_task(semester: str, references: list[str], targets: list[str], historical: list[str], responses: pd.DataFrame,
                  models: pd.DataFrame, items: pd.DataFrame, train: int, warmup: int, samples: int,
                  seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Steps 3a-3b for one semester.

    Returns
        item_columns:  <sem>_set, <sem>_train, aligned with `items`
        model_columns: <sem>_role, <sem>_theta, <sem>_theta_sd (references + targets + history)
        top:           the TOP_K references that defined EASY / HARD
    """
    top = models.global_theta.loc[references].nlargest(TOP_K).index.tolist()
    split = split_items(responses, items, top, train, seed)
    logger.info("%s: %d references, %d targets; top %s; items %s, %d train",
                semester, len(references), len(targets), top, split.set.value_counts().to_dict(), int(split.train.sum()))
    logger.info("%s: local 2PL fit (theta only) on the HARD items...", semester)
    hard = items[split.set == "hard"]
    theta = local_fit(responses, hard, references + targets, warmup, samples, seed)
    observed = set(responses.merge(hard[["bench", "item"]]).model)
    missing = sorted(set(historical) - observed)
    if missing:
        logger.warning("%s: historical models lack HARD responses; leaving theta* missing: %s", semester, missing)
    available = [m for m in historical if m in observed]
    if available:
        older_theta = local_fit(responses, hard, available, warmup, samples, seed)
        theta = pd.concat([theta, older_theta])

    role = pd.concat([pd.Series("reference", index=references), pd.Series("target", index=targets),
                      pd.Series("historical", index=historical)]).rename("role")
    labels = role.to_frame().join(theta)
    print(labels.round(2).to_string())
    return split.add_prefix(f"{semester}_"), labels.add_prefix(f"{semester}_"), top


# ----------------------------------------------------------------------------- build

def write(out: Path, models: pd.DataFrame, items: pd.DataFrame, responses: pd.DataFrame, settings: dict) -> None:
    """Step 4: the frozen tables."""
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / "models.csv", float_format="%.4f")
    items.to_csv(out / "items.csv", index=False, float_format="%.4f")
    fit_input = responses[["bench", "item", "model", "repeat", "correct"]]
    fit_input.sort_values(["bench", "item", "model", "repeat"]).to_csv(out / "responses.csv", index=False)
    (out / "fit.yml").write_text(yaml.safe_dump(settings, sort_keys=False))
    logger.info("wrote %s", out)


def changes(domain: DomainConfig) -> list[str]:
    """What differs between the collected logs / roster and the frozen tables. Cheap: reads no log contents."""
    models, _ = load(domain)
    frozen = pd.read_csv(frozen_dir(domain) / "responses.csv", dtype={"item": str})
    logs = eval_logs(domain)
    found = []
    now, then = set(logs.model), set(frozen.model)
    if new := sorted(now - then):
        found.append(f"models with logs that are not in the fit: {new}")
    if gone := sorted(then - now):
        found.append(f"models in the fit whose logs are gone: {gone}")
    repeats_now = logs.groupby(["bench", "model"]).repeat.nunique()
    repeats_then = frozen.groupby(["bench", "model"]).repeat.nunique().reindex(repeats_now.index).fillna(0)
    if more := repeats_now[repeats_now > repeats_then].index.tolist():
        found.append(f"new logs on (bench, model): {more}")
    paths = repeat0_paths(logs, DATA)
    for column in paths.columns:
        both = paths.index.intersection(models.index)
        if column in models and (moved := both[paths.loc[both, column].ne(models.loc[both, column]) & paths.loc[both, column].notna()].tolist()):
            found.append(f"repeat-0 log of {column.removeprefix('path_')} changed for: {moved}")
    semester_of = {m: s for s, roster in load_semesters().items() for m in roster}
    then_roster = models.semester.dropna()
    collected = now | then  # an uncollected roster model is not in the fit either way
    if changed := sorted(m for m in then_roster.index.union(semester_of) if m in collected and then_roster.get(m) != semester_of.get(m)):
        found.append(f"roster (configs/semesters.yml) changed for: {changed}")
    return found


def build(domain: DomainConfig, train=25, warmup=1000, samples=1000, seed=0, force=False) -> None:
    """Freeze a domain: steps 1-4 of the module docstring."""
    out = frozen_dir(domain)
    if (out / "models.csv").exists() and not force:
        found = changes(domain)
        for line in found:
            logger.warning("since the freeze: %s", line)
        if not found:
            logger.info("%s is frozen and nothing changed since: no refit needed", out)
            return
        raise FileExistsError(f"{out} is frozen and {len(found)} thing(s) changed since (above); refit with --force and commit")

    # 1. Every collected response of the domain, roster or not.
    responses = read_responses(domain)
    logger.info("%s: %d responses, %d models, %d items", domain.name, len(responses), responses.model.nunique(),
                len(responses[["bench", "item"]].drop_duplicates()))
    semesters = load_semesters()
    semester_of = {model: semester for semester, models in semesters.items() for model in models}

    # 2. Global fit; the models table also records each model's semester and repeat-0 logs.
    logger.info("global 2PL fit: NUTS, %d warmup + %d samples (minutes)...", warmup, samples)
    models, items = global_fit(responses, warmup, samples, seed)
    models.insert(0, "semester", models.index.map(semester_of))
    models = models.join(repeat0_paths(eval_logs(domain), DATA))

    # 3. One task per semester from the second on.
    top_per_semester = {}
    for index, semester in enumerate(list(semesters)[1:], start=1):
        references, targets = roster(semesters, semester, set(responses.model))
        if not references:
            continue
        historical = [m for past in list(semesters)[:index - 1] for m in semesters[past] if m in models.index]
        item_columns, model_columns, top = semester_task(
            semester, references, targets, historical, responses, models, items, train, warmup, samples, seed)
        items = items.join(item_columns)
        models = models.join(model_columns)
        top_per_semester[semester] = top

    # 4. Write.
    settings = dict(domain=domain.name, date=datetime.date.today().isoformat(), warmup=warmup, samples=samples,
                    seed=seed, train_per_quartile=train, top_k=TOP_K, top=top_per_semester)
    write(out, models, items, responses, settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--train", type=int, default=25, help="train items per difficulty quartile; 0 = all EASY")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite the frozen tables (then commit them)")
    args = parser.parse_args()
    try:
        build(DomainConfig.from_cli(args.domain), args.train, args.warmup, args.samples, args.seed, args.force)
    except FileExistsError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
