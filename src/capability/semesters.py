"""Semester tasks: the frozen per-domain tables every benchmark-free protocol is evaluated on.

    uv run reference configs/domain/science.yml [--train 25] [--force]

Per domain, all benches pooled, written once to data/meta/<domain>/ and committed there
(a refit is `--force` + a new commit, only once the domain's collection is complete):

1. GLOBAL IRT — 2PL with guessing floor on every model x item x repeat -> items.csv (a, b,
   guess), models.csv (global_theta).
2. SEMESTERS (configs/semesters.yml; 2023 is one) — each task from 2024-I on has 8 *reference*
   models (the previous semester) and 8 *target* models (this semester) a protocol must forecast.
3. EASY / HARD — EASY = items the TOP_K best reference models (by global theta) solve on repeat
   0; HARD = the rest (an item a top model has no repeat-0 response on is hard). The train set is
   the EASY items, optionally a fixed random subset stratified by global-b quartile (--train N per
   quartile, 0 = all) — item-level, so identical for every model. Protocols only see correct
   transcripts of the train items. -> items.csv <sem>_set, <sem>_train
4. LOCAL IRT — theta* of reference + target models on the HARD items with a, b, guess fixed to
   the global fit: "capability beyond what current benchmarks measure", and on the global scale
   for every semester. theta* is the target of every protocol. -> models.csv <sem>_role, <sem>_theta
5. responses.csv is the exact fit input; fit.yml the provenance. score() compares a protocol's
   theta-hat with theta*: Pearson r, MSE, affine-fit MSE (MSE after the best shift+scale).
"""

import argparse
import datetime
import subprocess
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
FILES = ("models.csv", "items.csv", "responses.csv", "fit.yml")


def load_semesters(path: Path = SEMESTERS) -> dict[str, list[str]]:
    return {str(k): list(v) for k, v in yaml.safe_load(path.read_text()).items()}


def _summary(draws, prefix: str) -> dict[str, np.ndarray]:
    d = np.asarray(draws)
    return {prefix: d.mean(0), f"{prefix}_sd": d.std(0),
            f"{prefix}_lo": np.percentile(d, 2.5, axis=0), f"{prefix}_hi": np.percentile(d, 97.5, axis=0)}


def _item_guess(resp: pd.DataFrame, items: list[tuple]) -> np.ndarray:
    guess = resp.groupby(["bench", "item"]).guess.max()  # the floor is an item property; some imported logs lack `choices`
    return np.array([guess[i] for i in items])


def global_fit(resp: pd.DataFrame, warmup=1000, samples=1000, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2PL + guessing over every response -> (models[model, global_theta..], items[bench, item, guess, a.., b..])."""
    data = encode(resp.rename(columns={"bench": "benchmark"}))
    guess = _item_guess(resp, data.items)
    post = fit(data, model="2pl", warmup=warmup, samples=samples, seed=seed, guess=guess)
    models = pd.DataFrame({"model": data.models, **_summary(post["theta"], "global_theta")})
    a, b = np.asarray(post["a"]), np.asarray(post["b"])
    items = pd.DataFrame(data.items, columns=["bench", "item"]).assign(
        guess=guess, a=a.mean(0), a_sd=a.std(0), b=b.mean(0), b_sd=b.std(0))
    return models, items


def split_items(resp: pd.DataFrame, items: pd.DataFrame, top: list[str], train: int = 25, seed: int = 0) -> pd.DataFrame:
    """items[bench, item] + `set` (easy/hard) + `train`. EASY = every model in `top` correct on repeat 0."""
    r0 = resp[(resp.repeat == 0) & resp.model.isin(top)]
    solved = r0.groupby(["bench", "item"]).agg(n=("model", "nunique"), ok=("correct", "min"))
    easy = solved[(solved.n == len(top)) & solved.ok.astype(bool)].index
    out = items[["bench", "item", "b"]].copy()
    out["set"] = np.where(pd.MultiIndex.from_frame(out[["bench", "item"]]).isin(easy), "easy", "hard")
    unseen = int((solved.n < len(top)).sum()) + int(len(out) - len(solved))
    if unseen:
        print(f"  {unseen} items lack a repeat-0 response from a top model -> hard")
    out["train"] = out.set == "easy"
    if train:
        e = out[out.train]
        q = pd.qcut(e.b.rank(method="first"), 4, labels=False)
        keep = e.groupby(q, group_keys=False).apply(
            lambda g: g.sample(min(train, len(g)), random_state=seed), include_groups=False)
        out["train"] = out.index.isin(keep.index)
    return out.drop(columns="b")


def local_fit(resp: pd.DataFrame, items: pd.DataFrame, split: pd.DataFrame, models: list[str],
              warmup=1000, samples=1000, seed=0) -> pd.DataFrame:
    """theta of `models` on the HARD items, item parameters fixed to the global fit -> [model, theta..]."""
    hard = split.loc[split.set == "hard", ["bench", "item"]]
    r = resp[resp.model.isin(models)].merge(hard)
    if missing := set(models) - set(r.model):
        raise ValueError(f"no HARD responses for {sorted(missing)}: theta would be the prior")
    data = encode(r.rename(columns={"bench": "benchmark"}))
    par = items.set_index(["bench", "item"]).loc[data.items]
    post = fit(data, model="fixed_2pl", warmup=warmup, samples=samples, seed=seed,
               a=par.a.to_numpy(), b=par.b.to_numpy(), guess=par.guess.to_numpy())
    return pd.DataFrame({"model": data.models, **_summary(post["theta"], "theta")})


def score(pred: pd.Series, truth: pd.Series) -> dict:
    """theta-hat vs theta* on the models both index: Pearson r, MSE, affine-fit MSE."""
    p, t = pred.align(truth, join="inner")
    beta, alpha = np.polyfit(p, t, 1)
    return {"n": len(p), "r": float(np.corrcoef(p, t)[0, 1]), "mse": float(((p - t) ** 2).mean()),
            "affine_mse": float(((alpha + beta * p - t) ** 2).mean())}


def frozen_dir(domain: DomainConfig) -> Path:
    return META / domain.name


def load(domain: DomainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frozen (models, items) tables of a domain."""
    d = frozen_dir(domain)
    if not (d / "models.csv").exists():
        raise FileNotFoundError(f"{d} not frozen — run `uv run reference {domain.source or domain.name}`")
    models = pd.read_csv(d / "models.csv")
    text = models.columns[models.dtypes == object]
    models[text] = models[text].fillna("")
    return models, pd.read_csv(d / "items.csv", dtype={"item": str})


def _log_paths(logs: pd.DataFrame) -> pd.DataFrame:
    """Per model: path_<bench> (the repeat-0 log, relative to data/) and repeats_<bench>."""
    r0 = logs[logs.repeat == 0]
    if (dup := r0.groupby(["bench", "model"]).size()).gt(1).any():
        raise ValueError(f"several repeat-0 logs for {dup[dup > 1].index.tolist()}")
    paths = r0.assign(path=r0.path.map(lambda p: str(Path(p).relative_to(DATA)))).pivot(
        index="model", columns="bench", values="path").add_prefix("path_")
    reps = logs.groupby(["model", "bench"]).repeat.nunique().unstack().add_prefix("repeats_")
    return paths.join(reps, how="outer").fillna({c: "" for c in paths.columns}).fillna(0).astype(
        {c: int for c in reps.columns}).reset_index()


def build(domain: DomainConfig, train=25, warmup=1000, samples=1000, seed=0, force=False) -> None:
    out = frozen_dir(domain)
    if (out / "models.csv").exists() and not force:
        raise FileExistsError(f"{out} is frozen; refit deliberately with --force and commit the result")
    resp = responses(domain)
    semesters = load_semesters()
    roster = {m: s for s, ms in semesters.items() for m in ms}
    models, items = global_fit(resp, warmup, samples, seed)
    models.insert(1, "semester", models.model.map(roster).fillna(""))
    models["n_responses"] = models.model.map(resp.groupby("model").size())
    models = models.merge(_log_paths(eval_logs(domain)), on="model", how="left")
    n_models = resp.groupby(["bench", "item"]).model.nunique()
    items["n_models"] = [n_models[(b, i)] for b, i in zip(items.bench, items.item)]
    theta = models.set_index("model").global_theta
    present = set(resp.model)
    prov = {}
    for prev, sem in zip(semesters, list(semesters)[1:]):
        ref = [m for m in semesters[prev] if m in present]
        tgt = [m for m in semesters[sem] if m in present]
        missing = sorted(set(semesters[prev] + semesters[sem]) - present)
        if missing:
            print(f"{sem}: not collected on {domain.name}: {missing}")
        if len(ref) < 2 or len(tgt) < 2:
            raise ValueError(f"{sem}: {len(ref)} references / {len(tgt)} targets on {domain.name}")
        top = theta.loc[ref].nlargest(TOP_K).index.tolist()
        split = split_items(resp, items, top, train, seed)
        local = local_fit(resp, items, split, ref + tgt, warmup, samples, seed).set_index("model")
        items[f"{sem}_set"], items[f"{sem}_train"] = split.set, split.train
        role = models.model.map({**{m: "reference" for m in ref}, **{m: "target" for m in tgt}}).fillna("")
        models[f"{sem}_role"] = role
        for c in local.columns:
            models[f"{sem}_{c}"] = models.model.map(local[c])
        prov[sem] = {"top": top, "missing": missing, "easy": int((split.set == "easy").sum()),
                     "hard": int((split.set == "hard").sum()), "train": int(split.train.sum())}
        print(f"{sem}: top {top}, {prov[sem]['easy']} easy / {prov[sem]['hard']} hard, {prov[sem]['train']} train\n"
              + local.assign(role=role.set_axis(models.model).loc[local.index]).round(2).to_string())
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / "models.csv", index=False, float_format="%.4f")
    items.to_csv(out / "items.csv", index=False, float_format="%.4f")
    resp[["bench", "item", "model", "repeat", "correct"]].sort_values(["bench", "item", "model", "repeat"]).to_csv(
        out / "responses.csv", index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    (out / "fit.yml").write_text(yaml.safe_dump({
        "domain": domain.name, "date": datetime.date.today().isoformat(), "commit": commit, "warmup": warmup,
        "samples": samples, "seed": seed, "train_per_quartile": train, "top_k": TOP_K,
        "n_models": len(models), "n_items": len(items), "n_responses": len(resp), "semesters": prov}, sort_keys=False))
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
