"""Semester tasks: the frozen per-domain tables every benchmark-free protocol is evaluated on.

    uv run reference configs/domain/science.yml [--train 25] [--force]

Per domain, all benches pooled, written once to data/meta/<domain>/ and committed there
(a refit is `--force` + a new commit, only once the domain's collection is complete):

1. GLOBAL IRT — 2PL with guessing floor on every model x item x repeat -> items.csv (a, b,
   guess), models.csv (global_theta).
2. SEMESTERS (configs/semesters.yml; 2023 is one) — each task from 2024-I on has 8 *reference*
   models (the previous semester) and 8 *target* models (this semester) a protocol must forecast.
3. EASY / HARD — EASY = items the TOP_K best reference models (by global theta) solve on repeat
   0; HARD = the rest; an item some top model has no repeat-0 response on (a bench that
   generation was never run on) is out of the task (set = ""). The train set is the EASY items,
   optionally a fixed random subset stratified by global-b quartile (--train N per quartile,
   0 = all) — item-level, so identical for every model. Protocols only see correct transcripts
   of the train items. -> items.csv <sem>_set, <sem>_train
4. LOCAL IRT — theta* of reference + target models on the HARD items with a, b, guess fixed to
   the global fit: "capability beyond what current benchmarks measure", and on the global scale
   for every semester. theta* is the target of every protocol. -> models.csv <sem>_role, <sem>_theta
5. responses.csv is the exact fit input; fit.yml the settings. score() compares a protocol's
   theta-hat with theta*: Pearson r, MSE, affine-fit MSE (MSE after the best shift+scale).

Tables used throughout:
    responses: one row per (bench, model, item, repeat) with `correct` (bool) and `guess`
               (the item's guessing floor). No text — transcripts are read from the logs later.
    items:     one row per (bench, item) with the global item parameters.
    models:    one row per model, indexed by model name.
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from configs.domain import DomainConfig
from domain.cli import eval_logs
from domain.cli import responses as read_responses
from irt.data import encode
from irt.fitting import fit

SEMESTERS = Path("configs/semesters.yml")
DATA = Path("data")
META = DATA / "meta"
TOP_K = 3  # reference models that define the EASY set


def load_semesters(path: Path = SEMESTERS) -> dict[str, list[str]]:
    """Load the semester models from configs/semesters.yml: {semester: [model, ...]}."""
    return {str(k): list(v) for k, v in yaml.safe_load(path.read_text()).items()}


def posterior_summary(draws, name: str) -> pd.DataFrame:
    """Mean and sd of posterior draws (samples x n) as columns `name`, `name_sd`."""
    draws = np.asarray(draws)
    return pd.DataFrame({name: draws.mean(0), f"{name}_sd": draws.std(0)})


def global_fit(responses: pd.DataFrame, warmup=1000, samples=1000, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the 2PL with guessing floor on every response: all models, all items, all repeats.

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


def split_items(responses: pd.DataFrame, items: pd.DataFrame, top: list[str], train: int = 25,
                seed: int = 0) -> pd.DataFrame:
    """Assign every item to the EASY set, the HARD set, or neither, for one semester task.

    EASY: every model in `top` (the best references) answered it correctly on repeat 0.
    HARD: every model in `top` answered it on repeat 0 and at least one got it wrong.
    "":   some model in `top` has no repeat-0 answer (its generation never ran that bench),
          so the item is out of this task.

    The train set is the EASY items, or a fixed random subset of them: `train` items from each
    quartile of global difficulty b within EASY (0 = keep all). It is drawn once per task, so
    every model is judged on the same items.

    Returns one row per item, aligned with `items`: set (easy / hard / ""), train (bool).
    """
    top_repeat0 = responses[(responses.repeat == 0) & responses.model.isin(top)]
    per_item = top_repeat0.groupby(["bench", "item"]).agg(
        n_answered=("model", "nunique"), all_correct=("correct", "min"))
    answered_by_all = per_item[per_item.n_answered == len(top)]
    status = pd.Series(np.where(answered_by_all.all_correct, "easy", "hard"), index=answered_by_all.index)

    item_keys = pd.MultiIndex.from_frame(items[["bench", "item"]])
    split = pd.DataFrame({"set": item_keys.map(status).fillna("")}, index=items.index)
    n_out = int((split.set == "").sum())
    if n_out:
        print(f"  {n_out} items lack a repeat-0 answer from a top model -> out of the task")

    split["train"] = split.set == "easy"
    if train:
        easy = items[split.train]
        quartile = pd.qcut(easy.b.rank(method="first"), 4, labels=False)
        sample = lambda group: group.sample(min(train, len(group)), random_state=seed)
        kept = easy.groupby(quartile, group_keys=False).apply(sample, include_groups=False)
        split["train"] = split.index.isin(kept.index)
    return split


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


def score(pred: pd.Series, truth: pd.Series) -> dict:
    """Compare theta-hat with theta* over the models both series index; NaN forecasts are dropped.

    r:          Pearson correlation — ignores shift and scale.
    mse:        raw squared error — the forecast has to be on the theta scale.
    affine_mse: squared error after the best linear map theta-hat -> theta* — ignores shift and
                scale but, unlike r, still penalises noise.
    """
    pred, truth = pred.align(truth, join="inner")
    known = pred.notna()
    pred, truth = pred[known], truth[known]
    if len(pred) < 3:
        return {"n": len(pred), "r": np.nan, "mse": np.nan, "affine_mse": np.nan}
    slope, intercept = np.polyfit(pred, truth, 1)
    return {
        "n": len(pred),
        "r": float(np.corrcoef(pred, truth)[0, 1]),
        "mse": float(((pred - truth) ** 2).mean()),
        "affine_mse": float(((intercept + slope * pred - truth) ** 2).mean()),
    }


def frozen_dir(domain: DomainConfig) -> Path:
    return META / domain.name


def load(domain: DomainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frozen tables of a domain: (models indexed by model, items)."""
    directory = frozen_dir(domain)
    if not (directory / "models.csv").exists():
        raise FileNotFoundError(f"{directory} not frozen — run `uv run reference {domain.source or domain.name}`")
    models = pd.read_csv(directory / "models.csv", index_col="model")
    items = pd.read_csv(directory / "items.csv", dtype={"item": str})
    return models, items


def log_paths(logs: pd.DataFrame) -> pd.DataFrame:
    """One column per bench, `path_<bench>`: each model's repeat-0 log, relative to data/.

    Protocols read transcripts from these logs. NaN where the model was never run on the bench.
    """
    repeat0 = logs[logs.repeat == 0].copy()
    duplicates = repeat0.groupby(["bench", "model"]).size()
    duplicates = duplicates[duplicates > 1]
    if len(duplicates):
        raise ValueError(f"several repeat-0 logs for {duplicates.index.tolist()}")
    repeat0["path"] = [str(Path(p).relative_to(DATA)) for p in repeat0.path]
    return repeat0.pivot(index="model", columns="bench", values="path").add_prefix("path_")


def build(domain: DomainConfig, train=25, warmup=1000, samples=1000, seed=0, force=False) -> None:
    """Fit everything and write the frozen tables of a domain (steps 1-5 of the module docstring)."""
    out = frozen_dir(domain)
    if (out / "models.csv").exists() and not force:
        raise FileExistsError(f"{out} is frozen; refit deliberately with --force and commit the result")

    responses = read_responses(domain)
    semesters = load_semesters()
    collected = set(responses.model)

    # 1. Global fit. Every model with logs on the domain takes part, roster or not.
    models, items = global_fit(responses, warmup, samples, seed)
    semester_of = {model: semester for semester, roster in semesters.items() for model in roster}
    models.insert(0, "semester", models.index.map(semester_of))
    models = models.join(log_paths(eval_logs(domain)))

    # 2. One task per semester: references = the previous semester's roster, targets = this one's.
    top_per_semester = {}
    for previous, semester in zip(semesters, list(semesters)[1:]):
        references = [m for m in semesters[previous] if m in collected]
        targets = [m for m in semesters[semester] if m in collected]
        missing = sorted(set(semesters[previous] + semesters[semester]) - collected)
        if missing:
            print(f"{semester}: not collected on {domain.name}: {missing}")
        if len(references) < 2 or len(targets) < 2:
            raise ValueError(f"{semester}: {len(references)} references / {len(targets)} targets on {domain.name}")

        # 3. EASY / HARD split, defined by the best references.
        top = models.global_theta.loc[references].nlargest(TOP_K).index.tolist()
        split = split_items(responses, items, top, train, seed)
        items[f"{semester}_set"] = split.set
        items[f"{semester}_train"] = split.train

        # 4. Local fit: theta* of references and targets on the HARD items.
        hard_items = items[split.set == "hard"]
        theta = local_fit(responses, hard_items, references + targets, warmup, samples, seed)
        role = pd.concat([pd.Series("reference", index=references), pd.Series("target", index=targets)])
        models[f"{semester}_role"] = role
        models = models.join(theta.add_prefix(f"{semester}_"))

        top_per_semester[semester] = top
        counts = split.set.value_counts().to_dict()
        print(f"{semester}: top {top}, {counts}, {int(split.train.sum())} train")
        print(theta.join(role.rename("role")).round(2).to_string())

    # 5. Write the frozen tables.
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / "models.csv", float_format="%.4f")
    items.to_csv(out / "items.csv", index=False, float_format="%.4f")
    fit_input = responses[["bench", "item", "model", "repeat", "correct"]]
    fit_input.sort_values(["bench", "item", "model", "repeat"]).to_csv(out / "responses.csv", index=False)
    settings = dict(domain=domain.name, date=datetime.date.today().isoformat(), warmup=warmup, samples=samples,
                    seed=seed, train_per_quartile=train, top_k=TOP_K, top=top_per_semester)
    (out / "fit.yml").write_text(yaml.safe_dump(settings, sort_keys=False))
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--train", type=int, default=25, help="train items per difficulty quartile; 0 = all EASY")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite the frozen tables (then commit them)")
    args = parser.parse_args()
    build(DomainConfig.from_cli(args.domain), args.train, args.warmup, args.samples, args.seed, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
