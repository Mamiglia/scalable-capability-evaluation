"""The frozen tables (data/meta/, written by `uv run reference`) and the roster.

Five long tables across every domain, one key each, plus the fit settings per domain. data/meta/full/ holds
everything (what the code reads); data/meta/ holds the paper's subset, derived by `write_paper`: the roster
tasks and PAPER's added-cohort targets, and only those models' rows of models.csv and samples.csv (the
global fit also used off-roster models, so its a, b are reproducible from full/ only).

    samples.csv     domain, bench, item, model, repeat, correct, log        one row per scored transcript;
                    log = its .eval relative to data/ (sample.id = item); blank if the log left the disk
    items.csv       domain, bench, item, a, a_sd, b, b_sd, c                global 2PL, one scale per domain
    models.csv      domain, model, semester, theta_global, theta_global_sd  semester = roster semester
                    (configs/semesters.yml), blank off-roster; theta_global blank for a cohort added after the fit
    tasks.csv       domain, semester, model, role, cohort, theta, theta_sd  role: reference / historical / target;
                    theta = this task's theta* on its HARD items (blank: no HARD responses);
                    cohort: roster, or e.g. frontier = extra targets added after the freeze
    task_items.csv  domain, semester, bench, item, split, train, quartile   split: easy / hard (items out of
                    the task are absent); quartile of b among the task's EASY items, blank on HARD
    <domain>/fit.yml  NUTS settings, seed, train_per_quartile, top-k references per task, cohort provenance

Read item ids as str. `load` gives the wide per-domain view most protocols read:

    models  index model; semester; global_theta(_sd); path_<bench> (repeat-0 log);
            per task <sem>_role, <sem>_theta, <sem>_theta_sd
    items   bench, item, guess, a, a_sd, b, b_sd; per task <sem>_set (easy / hard / NaN), <sem>_train
"""

import os
from pathlib import Path

import pandas as pd
import yaml

SEMESTERS = Path("configs/semesters.yml")
# Transcript logs; SMALL_JUDGE_DATA points at a checkout elsewhere (e.g. an external volume).
DATA = Path(os.environ.get("SMALL_JUDGE_DATA", "data"))
FEATURES = Path("logs/features")  # per-transcript feature CSVs mirroring data/ (generality-labs/protocol-outputs features/)
META = Path("data") / "meta"  # the frozen tables stay in the repo checkout
# Added-cohort targets the paper evaluates (iclr, closed-source targets on CODING); the rest stay in full/ only.
PAPER = {"frontier": ["claude-fable-5", "claude-opus-4.7", "claude-opus-4.8", "gemini-3.5-flash", "gpt-5", "gpt-5.6-sol"]}
KEYS = {"samples": ["domain", "bench", "item", "model", "repeat"], "items": ["domain", "bench", "item"],
        "models": ["domain", "model"], "tasks": ["domain", "semester", "model"],
        "task_items": ["domain", "semester", "bench", "item"]}
STR = {c: str for c in ["domain", "bench", "item", "model", "semester", "role", "cohort", "split", "log"]}
# A domain name that reads another domain's tables with one extra cohort as that cohort's tasks' targets.
COHORTS = {"coding_frontier": ("coding", "frontier")}


def _name(domain) -> str:
    return getattr(domain, "name", domain)


def frozen_dir(domain) -> Path:
    """data/meta/<domain>/: the fit settings (fit.yml) and figures."""
    return META / _name(domain)


def read(table: str, domain=None) -> pd.DataFrame:
    """One long table, optionally one domain's rows (by its own name, not resolved through COHORTS)."""
    frame = pd.read_csv(META / "full" / f"{table}.csv", dtype={c: t for c, t in STR.items()})
    return frame if domain is None else frame[frame.domain == _name(domain)].reset_index(drop=True)


def frozen(domain) -> bool:
    return (META / "full" / "models.csv").exists() and (read("models").domain == COHORTS.get(_name(domain), (_name(domain),))[0]).any()


def _csv(rows: pd.DataFrame, path: Path) -> None:
    if "quartile" in rows:  # an integer category, not a float (concat with blanks makes it float)
        rows = rows.assign(quartile=rows.quartile.astype("Int64"))
    rows.to_csv(path, index=False, float_format="%.4f")


def write(table: str, domain: str, rows: pd.DataFrame) -> None:
    """Replace one domain's rows of a long table (other domains untouched)."""
    rows = rows.assign(domain=domain)
    if rows.duplicated(KEYS[table]).any():
        raise ValueError(f"{table}: duplicate keys for {domain}")
    path = META / "full" / f"{table}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = pd.concat([read(table).query("domain != @domain"), rows])
    columns = KEYS[table] + [c for c in rows.columns if c not in KEYS[table]]
    _csv(rows[columns].sort_values(KEYS[table]), path)


def write_paper() -> None:
    """data/meta/*.csv: the paper's subset of data/meta/full/ (module docstring). Rerun after any `write`."""
    tasks = read("tasks")
    paper = [c == "roster" or m in PAPER.get(c, []) for m, c in zip(tasks.model, tasks.cohort)]
    tasks = tasks[paper]
    models = tasks[["domain", "model"]].drop_duplicates()
    for table in KEYS:
        rows = tasks if table == "tasks" else read(table)
        if table in ("models", "samples"):
            rows = rows.merge(models)
        _csv(rows.sort_values(KEYS[table]), META / f"{table}.csv")


def fit_settings(domain) -> dict:
    """<domain>/fit.yml of the domain whose tables `domain` reads."""
    base = COHORTS.get(_name(domain), (_name(domain),))[0]
    return yaml.safe_load((META / base / "fit.yml").read_text())


def _view(domain) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(models, tasks, samples) of the domain as `load` sees it, cohorts resolved."""
    name = _name(domain)
    base, cohort = COHORTS.get(name, (name, None))
    if not frozen(name):
        raise FileNotFoundError(f"{name} not frozen — run `uv run reference configs/domain/{base}.yml`")
    models, tasks, samples = read("models", base), read("tasks", base), read("samples", base)
    if cohort is None:  # the domain itself: its own fit, no added cohort
        tasks = tasks[tasks.cohort == "roster"]
        models = models[models.theta_global.notna()]
    else:  # the cohort replaces the roster's targets in the tasks it adds targets to
        added = tasks[tasks.cohort == cohort].semester.unique()
        tasks = tasks[tasks.cohort.isin(["roster", cohort])].copy()
        tasks.loc[tasks.semester.isin(added) & (tasks.cohort == "roster") & (tasks.role == "target"), "role"] = None
    return models, tasks, samples[samples.model.isin(models.model)]


def load(domain) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The wide frozen tables of a domain: (models indexed by model, items). See the module docstring."""
    models, tasks, samples = _view(domain)
    base = COHORTS.get(_name(domain), (_name(domain),))[0]
    wide = models.set_index("model")[["semester", "theta_global", "theta_global_sd"]].rename(
        columns={"theta_global": "global_theta", "theta_global_sd": "global_theta_sd"})
    paths = samples[samples.repeat == 0].dropna(subset=["log"]).drop_duplicates(["bench", "model"])
    items = read("items", base).drop(columns="domain").rename(columns={"c": "guess"})
    items = items[["bench", "item", "guess", "a", "a_sd", "b", "b_sd"]]
    pivot = paths.pivot(index="model", columns="bench", values="log").reindex(columns=sorted(items.bench.unique()))
    wide = wide.join(pivot.add_prefix("path_"))  # a column per bench, all NaN if nobody's log is on record
    task_items = read("task_items", base)
    for semester in sorted(tasks.semester.unique()):
        labels = tasks[tasks.semester == semester].set_index("model")[["role", "theta", "theta_sd"]]
        wide = wide.join(labels.add_prefix(f"{semester}_"))
        split = task_items[task_items.semester == semester][["bench", "item", "split", "train"]]
        items = items.merge(split.rename(columns={"split": f"{semester}_set", "train": f"{semester}_train"}), how="left")
        items[f"{semester}_train"] = items[f"{semester}_train"].eq(True)
    return wide.rename_axis("model"), items


def load_semesters(path: Path = SEMESTERS) -> dict[str, list[str]]:
    """configs/semesters.yml as {semester: [model shortname, ...]}, in file order."""
    return {str(k): list(v) for k, v in yaml.safe_load(path.read_text()).items()}


def load_responses(domain) -> pd.DataFrame:
    """Every scored response of the domain (the fit input, plus any added cohort's).

    Returns one row per (bench, item, model, repeat) with columns bench, item (str),
    model, repeat (int), correct (bool). Covers every item and repeat, EASY and HARD.
    """
    return _view(domain)[2][["bench", "item", "model", "repeat", "correct"]].reset_index(drop=True)


def to_long(models: pd.DataFrame, items: pd.DataFrame, responses: pd.DataFrame, logs: pd.DataFrame,
            cohort: str = "roster") -> dict[str, pd.DataFrame]:
    """One domain's wide (models, items, responses) as long rows (no `domain` column; `write` adds it).

    `logs` (bench, model, repeat, path relative to data/) gives the repeat > 0 logs; repeat 0 comes from
    the wide path_<bench> columns, the log each protocol reads.
    """
    semesters = [c.removesuffix("_role") for c in models.columns if c.endswith("_role")]
    tasks, task_items = [], []
    for semester in semesters:
        labels = models[models[f"{semester}_role"].notna() & models[f"{semester}_role"].ne("")]
        tasks.append(pd.DataFrame(dict(semester=semester, model=labels.index, role=labels[f"{semester}_role"].values,
                                       cohort=cohort, theta=labels[f"{semester}_theta"].values,
                                       theta_sd=labels[f"{semester}_theta_sd"].values)))
        split = items[f"{semester}_set"].fillna("")
        if items.loc[split != "", f"{semester}_train"].isna().any():
            raise ValueError(f"{semester}: train flag missing on a split item")
        kept = items[split != ""]
        easy = kept[split[split != ""] == "easy"]
        # as reference.freeze.train_subset draws the train set (which needs >= 4 EASY items)
        quartile = pd.qcut(easy.b.rank(method="first"), 4, labels=False) if len(easy) >= 4 else pd.Series(index=easy.index)
        task_items.append(pd.DataFrame(dict(semester=semester, bench=kept.bench, item=kept.item, split=split[split != ""],
                                            train=kept[f"{semester}_train"].astype(bool),
                                            quartile=quartile.reindex(kept.index).astype("Int64"))))
    repeat0 = models.filter(like="path_").stack().rename("log").reset_index()
    repeat0 = repeat0.assign(bench=repeat0.iloc[:, 1].str.removeprefix("path_"), repeat=0)[["bench", "model", "repeat", "log"]]
    later = logs[logs.repeat > 0].rename(columns={"path": "log"})[["bench", "model", "repeat", "log"]]
    samples = responses[["bench", "item", "model", "repeat", "correct"]].merge(pd.concat([repeat0, later]), how="left")
    return dict(
        samples=samples,
        items=items[["bench", "item", "a", "a_sd", "b", "b_sd", "guess"]].rename(columns={"guess": "c"}),
        models=pd.DataFrame(dict(model=models.index, semester=models.semester.values,
                                 theta_global=models.global_theta.values, theta_global_sd=models.global_theta_sd.values)),
        tasks=pd.concat(tasks),
        task_items=pd.concat(task_items),
    )
