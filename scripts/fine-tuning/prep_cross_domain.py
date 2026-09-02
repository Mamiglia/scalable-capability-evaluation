"""Cross-domain judge-transfer pairs. Builds both-correct pairs over the models
present on ALL THREE Epoch domains (gpqa/aime/chess), labelled by either in-domain
accuracy-logit or a pooled ECI (general capability). Reuses prep_pairs.py helpers.

See docs/2026-08-09-cross-domain-judge-transfer-design.md / -plan.md.
Run from the worktree root: python scripts/analyses/judge_ft/prep_cross_domain.py ...
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# worktree src (judge.util / judge.dataset with strip_answer) + this dir (prep_pairs)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "src"))

from prep_pairs import (accuracy_logit, assign_splits, base_group,  # noqa: E402
                        build_pairs, has_reasoning, load_model_log,
                        subsample_stratified, token_filter)

import hashlib  # noqa: E402


def _header_accuracy(path):
    """(accuracy, total_samples) from the log header — no sample parsing.
    Same per-sample mean load_model_log recomputes; fine as a capability rank."""
    from inspect_ai.log import read_eval_log
    h = read_eval_log(path, header_only=True)
    if not (h.results and h.results.scores):
        return None
    n = h.results.total_samples
    for sc in h.results.scores:
        for name in ("accuracy", "mean"):
            if sc.metrics and name in sc.metrics:
                return float(sc.metrics[name].value), int(n)
    return None


def pooled_eci(logit_df: pd.DataFrame) -> pd.Series:
    """Mean of per-domain z-scored logits. A zero-variance domain contributes 0
    (not NaN), so pooling never blanks a model."""
    z = logit_df.copy()
    for c in z.columns:
        sd = z[c].std(ddof=0)
        z[c] = 0.0 if sd == 0 else (z[c] - z[c].mean()) / sd
    return z.mean(axis=1)


def _domain_logits(cfg, dom):
    """base_group -> (accuracy-logit, path) for one domain, from log headers.
    Keeps the first path per base_group (deterministic) so budget variants don't
    double-count. The path is kept so T2 loads transcripts for the shared set only."""
    prep = cfg["prep"]
    out = {}
    for p in sorted(glob.glob(cfg["domains"][dom])):
        g = base_group(Path(p).stem)
        if g in out:
            continue  # first path wins
        ha = _header_accuracy(p)
        if ha is None:
            continue
        acc, n = ha
        if not (prep["min_accuracy"] <= acc <= prep["max_accuracy"]):
            continue
        out[g] = (accuracy_logit(acc, n), p)
    return out


def build_model_pool(cfg) -> pd.DataFrame:
    """The set of models the experiment runs on: one row per base_group present on
    all three domains, with per-domain capability logits, pooled ECI, and a seeded
    held-out flag."""
    doms = list(cfg["domains"])
    per_domain = {d: _domain_logits(cfg, d) for d in doms}
    shared = sorted(set.intersection(*(set(r) for r in per_domain.values())))
    pool = pd.DataFrame(index=shared)
    pool.index.name = "base_group"
    for d in doms:
        pool[f"{d}_logit"] = [per_domain[d][g][0] for g in shared]
        pool[f"{d}_path"] = [per_domain[d][g][1] for g in shared]   # for T2 transcript load
    pool["eci"] = pooled_eci(pool[[f"{d}_logit" for d in doms]])
    rng = np.random.default_rng(cfg["prep"]["seed"])
    hold = set(rng.choice(shared, size=cfg["prep"]["n_holdout_models"], replace=False))
    pool["holdout"] = pool.index.isin(hold)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pool.sort_values("eci", ascending=False).to_csv(out / "model_pool.csv")
    return pool


def _item_held(item, frac) -> bool:
    """Deterministic per-item holdout, stable across cells and domains
    (item ids don't collide across the three benchmarks)."""
    h = int(hashlib.md5(str(item).encode()).hexdigest(), 16) % 1000
    return h < int(frac * 1000)


def load_domain_meta(cfg, dom, pool):
    """{(base_group, item): {text, options, question}} for one domain, over the
    pool models, first-correct transcripts that survive the reasoning filter.
    Also the item set. Transcript extraction is cached per model log."""
    prep = cfg["prep"]
    # per-domain cache: the same model has the same filename stem in every domain
    # dir, so a shared cache would cross-contaminate (gpqa transcripts served for aime/chess).
    cache = Path(cfg["out_dir"]) / "transcripts_cache" / dom
    meta, items = {}, set()
    for g in pool.index:
        f = load_model_log(pool.loc[g, f"{dom}_path"], cache, prep["max_len_chars"])
        for r in f.itertuples():
            if not has_reasoning(r.text, prep["min_reasoning_chars"]):
                continue
            item = str(r.item)   # chess ids are ints, gpqa/aime strs — normalise so sorted() works
            meta[(g, item)] = dict(text=r.text, options=r.options, question=r.question)
            items.add(item)
    return meta, items


def _tokenizer(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def build_domain_pairs(cfg, pool, dom, label_mode):
    """All both-correct pairs for one domain, split into train/eval_* by the
    shared item + model holdouts. Caps eval splits, token-filters, writes parquet.
    Skips if the parquet already exists (idempotent re-runs)."""
    prep = cfg["prep"]
    out = Path(cfg["out_dir"]) / f"pairs_{dom}_{label_mode}.parquet"
    if out.exists():
        return pd.read_parquet(out), 0
    groups = {g: g for g in pool.index}                       # each model is its own group
    caps = (pool["eci"] if label_mode == "eci" else pool[f"{dom}_logit"]).to_dict()
    min_gap = prep["min_cap_gap_eci"] if label_mode == "eci" else prep["min_cap_gap_logit"]

    meta, items = load_domain_meta(cfg, dom, pool)
    pairs = pd.DataFrame(build_pairs(meta, caps, groups, items, min_gap))
    pairs["domain"] = dom

    hold_models = set(pool.index[pool.holdout])
    hold_items = {i for i in pairs.item.unique() if _item_held(i, prep["item_holdout_frac"])}
    pairs["split"] = assign_splits(pairs, hold_models, hold_items).values
    pairs = pairs[pairs.split != "drop"].reset_index(drop=True)

    # cap each eval split (keep both orders of a pair together, stratify by model pair)
    keep = [pairs[pairs.split == "train"]]
    for sp in ("eval_mm", "eval_items", "eval_mixed"):
        g = pairs[pairs.split == sp]
        if not g.empty:
            keep.append(subsample_stratified(g, prep["max_eval_pairs_per_split"],
                                             by="model_1", seed=prep["seed"]))
    pairs = pd.concat(keep, ignore_index=True)

    tok = _tokenizer(prep["tokenizer"])
    pairs, dropped = token_filter(pairs, tok, prep["max_input_tokens"])
    pairs.to_parquet(out)
    return pairs, dropped


def assemble_cell(cfg, cell, label_mode, domain_pairs):
    """Training pairs for one cell = train splits of its domains, balanced across
    domains (round-robin) and capped. Writes train_<cell>_<label>.parquet."""
    prep = cfg["prep"]
    train_domains = cfg["cells"][cell]
    train = pd.concat([domain_pairs[(d, label_mode)].query("split == 'train'")
                       for d in train_domains], ignore_index=True)
    train = subsample_stratified(train, prep["max_train_pairs"], by="domain", seed=prep["seed"])
    out = Path(cfg["out_dir"]) / f"train_{cell}_{label_mode}.parquet"
    train.to_parquet(out)
    return train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_HERE / "cross_domain.yaml"))
    ap.add_argument("--pool-only", action="store_true")
    ap.add_argument("--build", action="store_true", help="build all domain pairs + cell trains")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    pool = build_model_pool(cfg)
    doms = list(cfg["domains"])
    print(f"model pool: {len(pool)} models on all of {doms}")
    print(f"ECI range: {pool.eci.min():.2f} .. {pool.eci.max():.2f}")
    held = pool.index[pool.holdout].tolist()
    print(f"held-out models (n={len(held)}): {held}")
    print(f"wrote {cfg['out_dir']}/model_pool.csv")
    if args.pool_only:
        return

    domain_pairs = {}
    for label in cfg["labels"]:
        for dom in doms:
            p, dropped = build_domain_pairs(cfg, pool, dom, label)
            domain_pairs[(dom, label)] = p
            bal = p[p.split.str.startswith("train")].label.mean()
            print(f"pairs {dom}/{label}: {len(p)} rows, splits="
                  f"{p.split.value_counts().to_dict()}, train label-mean={bal:.3f}, "
                  f"token-dropped={dropped}")
    # guard: domains must not share items (a transcript-cache collision would make
    # one domain's pairs carry another's items — this catches that class of bug)
    import itertools
    lab0 = cfg["labels"][0]
    item_sets = {d: set(domain_pairs[(d, lab0)].item) for d in doms}
    for a, b in itertools.combinations(doms, 2):
        ov = item_sets[a] & item_sets[b]
        assert not ov, f"item overlap {a}∩{b}: {len(ov)} items — cross-domain contamination"

    for label in cfg["labels"]:
        for cell in cfg["cells"]:
            t = assemble_cell(cfg, cell, label, domain_pairs)
            print(f"cell {cell}/{label}: {t.pair_uid.nunique()} pairs "
                  f"({t.domain.value_counts().to_dict()})")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
