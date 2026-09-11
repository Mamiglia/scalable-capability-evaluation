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
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from configs.domain import DomainConfig
from domain.cli import eval_logs, responses
from irt.data import encode
from irt.fitting import fit

SEMESTERS = Path("configs/semesters.yml")
DATA = Path("data")
META = DATA / "meta"
TOP_K = 3


def load_semesters(path: Path = SEMESTERS) -> dict[str, list[str]]:
    """Load the semester models from configs/semesters.yml: {semester: [model, ...]}."""
    return {str(k): list(v) for k, v in yaml.safe_load(path.read_text()).items()}


def global_fit(transcripts: pd.DataFrame, warmup=1000, samples=1000, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2PL + guessing over every response -> (models[global_theta, _sd] by model, items[bench, item, guess, a, b, ..])."""
    data = encode(transcripts)
    guess = transcripts.groupby(["bench", "item"]).guess.max()  # the floor is an item property; some imported logs lack `choices`
    guess = np.array([guess[i] for i in data.items])
    post = fit(data, model="2pl", warmup=warmup, samples=samples, seed=seed, guess=guess)
    theta, a, b = (np.asarray(post[k]) for k in ("theta", "a", "b"))
    models = pd.DataFrame(
        {"global_theta": theta.mean(0), "global_theta_sd": theta.std(0)}, 
        index=pd.Index(data.models, name="model")
    )
    items = pd.DataFrame(data.items, columns=["bench", "item"]).assign(
        guess=guess, a=a.mean(0), a_sd=a.std(0), b=b.mean(0), b_sd=b.std(0))
    return models, items


def split_items(transcripts: pd.DataFrame, items: pd.DataFrame, top: list[str], train: int = 25, seed: int = 0) -> pd.DataFrame:
    """`set` (easy / hard / "" = out of the task) and `train` per item. EASY = every model in `top` correct on repeat 0."""
    r0 = transcripts[(transcripts.repeat == 0) & transcripts.model.isin(top)]
    seen = r0.groupby(["bench", "item"]).agg(n=("model", "nunique"), ok=("correct", "min"))
    seen = seen[seen.n == len(top)]
    status = pd.Series(np.where(seen.ok, "easy", "hard"), index=seen.index)
    out = pd.DataFrame({"set": pd.MultiIndex.from_frame(items[["bench", "item"]]).map(status).fillna("")}, index=items.index)
    if (n_out := int((out.set == "").sum())):
        print(f"  {n_out} items without a repeat-0 response from every top model -> out of the task")
    out["train"] = out.set == "easy"
    if train:
        easy = items[out.train]
        quartile = pd.qcut(easy.b.rank(method="first"), 4, labels=False)
        keep = easy.groupby(quartile, group_keys=False).apply(
            lambda g: g.sample(min(train, len(g)), random_state=seed), include_groups=False)
        out["train"] = out.index.isin(keep.index)
    return out


def local_fit(transcripts: pd.DataFrame, hard: pd.DataFrame, models: list[str], warmup=1000, samples=1000, seed=0) -> pd.DataFrame:
    """theta of `models` on the `hard` items (bench, item, a, b, guess fixed to the global fit) -> [theta, theta_sd] by model."""
    r = transcripts[transcripts.model.isin(models)].merge(hard[["bench", "item"]])
    if missing := set(models) - set(r.model):
        raise ValueError(f"no HARD responses for {sorted(missing)}: theta would be the prior")
    data = encode(r)
    par = hard.set_index(["bench", "item"]).loc[data.items]
    post = fit(data, model="fixed_2pl", warmup=warmup, samples=samples, seed=seed,
               a=par.a.to_numpy(), b=par.b.to_numpy(), guess=par.guess.to_numpy())
    theta = np.asarray(post["theta"])
    return pd.DataFrame({"theta": theta.mean(0), "theta_sd": theta.std(0)}, index=pd.Index(data.models, name="model"))


def score(pred: pd.Series, truth: pd.Series) -> dict:
    """theta-hat vs theta* on the models both index (NaN forecasts dropped): Pearson r, MSE, affine-fit MSE."""
    p, t = pred.align(truth, join="inner")
    p, t = p[p.notna()], t[p.notna()]
    if len(p) < 3:
        return {"n": len(p), "r": np.nan, "mse": np.nan, "affine_mse": np.nan}
    beta, alpha = np.polyfit(p, t, 1)
    return {"n": len(p), "r": float(np.corrcoef(p, t)[0, 1]), "mse": float(((p - t) ** 2).mean()),
            "affine_mse": float(((alpha + beta * p - t) ** 2).mean())}


def frozen_dir(domain: DomainConfig) -> Path:
    return META / domain.name


def load(domain: DomainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frozen (models by model, items) tables of a domain."""
    d = frozen_dir(domain)
    if not (d / "models.csv").exists():
        raise FileNotFoundError(f"{d} not frozen — run `uv run reference {domain.source or domain.name}`")
    return pd.read_csv(d / "models.csv", index_col="model"), pd.read_csv(d / "items.csv", dtype={"item": str})


def log_paths(logs: pd.DataFrame) -> pd.DataFrame:
    """path_<bench> per model: the repeat-0 log, relative to data/ (protocols read transcripts from it)."""
    r0 = logs[logs.repeat == 0]
    if (dup := r0.groupby(["bench", "model"]).size()).gt(1).any():
        raise ValueError(f"several repeat-0 logs for {dup[dup > 1].index.tolist()}")
    return r0.assign(path=[str(Path(p).relative_to(DATA)) for p in r0.path]).pivot(
        index="model", columns="bench", values="path").add_prefix("path_")


def build(domain: DomainConfig, train=25, warmup=1000, samples=1000, seed=0, force=False) -> None:
    out = frozen_dir(domain)
    if (out / "models.csv").exists() and not force:
        raise FileExistsError(f"{out} is frozen; refit deliberately with --force and commit the result")
    transcripts = responses(domain)
    semesters = load_semesters()
    models, items = global_fit(transcripts, warmup, samples, seed)
    models.insert(0, "semester", models.index.map({m: s for s, ms in semesters.items() for m in ms}))
    models = models.join(log_paths(eval_logs(domain)))
    present = set(transcripts.model)
    tops = {}
    for prev, sem in zip(semesters, list(semesters)[1:]):
        ref = [m for m in semesters[prev] if m in present]
        tgt = [m for m in semesters[sem] if m in present]
        if missing := sorted(set(semesters[prev] + semesters[sem]) - present):
            print(f"{sem}: not collected on {domain.name}: {missing}")
        if len(ref) < 2 or len(tgt) < 2:
            raise ValueError(f"{sem}: {len(ref)} references / {len(tgt)} targets on {domain.name}")
        tops[sem] = top = models.global_theta.loc[ref].nlargest(TOP_K).index.tolist()
        split = split_items(transcripts, items, top, train, seed)
        local = local_fit(transcripts, items[split.set == "hard"], ref + tgt, warmup, samples, seed)
        items[f"{sem}_set"], items[f"{sem}_train"] = split.set, split.train
        models[f"{sem}_role"] = pd.Series({**dict.fromkeys(ref, "reference"), **dict.fromkeys(tgt, "target")})
        models = models.join(local.add_prefix(f"{sem}_"))
        print(f"{sem}: top {top}, {split.set.value_counts().to_dict()}, {int(split.train.sum())} train\n"
              + local.join(models[f"{sem}_role"]).round(2).to_string())
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / "models.csv", float_format="%.4f")
    items.to_csv(out / "items.csv", index=False, float_format="%.4f")
    transcripts[["bench", "item", "model", "repeat", "correct"]].sort_values(["bench", "item", "model", "repeat"]).to_csv(
        out / "responses.csv", index=False)
    (out / "fit.yml").write_text(yaml.safe_dump(dict(
        domain=domain.name, date=datetime.date.today().isoformat(), warmup=warmup, samples=samples, seed=seed,
        train_per_quartile=train, top_k=TOP_K, top=tops), sort_keys=False))
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--train", type=int, default=25, help="train items per difficulty quartile; 0 = all EASY")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite the frozen tables (then commit them)")
    a = parser.parse_args()
    build(DomainConfig.from_cli(a.domain), a.train, a.warmup, a.samples, a.seed, a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
