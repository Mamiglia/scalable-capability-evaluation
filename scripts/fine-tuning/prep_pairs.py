"""Build counter-balanced both-correct GPQA pairs for judge fine-tuning.

Reads epoch-pool eval logs, caches per-model transcripts, emits
models_ft.csv, item_split.csv, items_ft.parquet, pairs.parquet and
pairs_selfpref.parquet under out_dir.
"""

import argparse
import glob
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import prompts
from judge.dataset import strip_answer

SUFFIX_RE = re.compile(r"(_\d+K|_high|_low|_medium|_minimal|_none|_xhigh)+$")


def base_group(model: str) -> str:
    """Strip trailing budget/effort suffixes so budget variants group together."""
    return SUFFIX_RE.sub("", model)


def has_reasoning(text: str, min_chars: int) -> bool:
    """False for answer-only transcripts: nothing left after the ANSWER line."""
    return len(strip_answer(text)) >= min_chars


def render_choices(choices) -> str:
    return "\n".join(f"{chr(ord('A') + i)}) {c.strip()}" for i, c in enumerate(choices))


def options_block(body: str, n: int) -> str:
    if not body:
        return ""
    return prompts.OPTIONS_BLOCK.format(n=n, options=body)


def make_pair_rows(item, s1_meta, s2_meta, question) -> list[dict]:
    """Two counter-balanced rows for one (item, model pair)."""
    m_lo, m_hi = sorted([s1_meta["model"], s2_meta["model"]])
    uid = f"{item}|{m_lo}|{m_hi}"
    rows = []
    for order, (a, b) in enumerate([(s1_meta, s2_meta), (s2_meta, s1_meta)]):
        rows.append(dict(
            pair_uid=uid, order=order, item=item,
            model_1=a["model"], model_2=b["model"],
            cap_1=a["cap"], cap_2=b["cap"],
            label=1 if a["cap"] > b["cap"] else 2,
            question=question,
            options_1=options_block(a["options"], 1),
            options_2=options_block(b["options"], 2),
            text_1=a["text"], text_2=b["text"],
        ))
    return rows


def check_counterbalance(df: pd.DataFrame) -> None:
    """Raise unless every pair_uid has two order-swapped rows and every
    split has exactly equal label counts."""
    for uid, g in df.groupby("pair_uid"):
        assert len(g) == 2, f"{uid}: {len(g)} rows"
        r1, r2 = g.iloc[0], g.iloc[1]
        assert r1.text_1 == r2.text_2 and r1.text_2 == r2.text_1, uid
        assert {r1.label, r2.label} == {1, 2}, uid
    for split, g in df.groupby("split"):
        n1, n2 = (g.label == 1).sum(), (g.label == 2).sum()
        assert n1 == n2, f"{split}: label counts {n1} vs {n2}"


def assign_splits(pairs_df, holdout_models, holdout_items) -> pd.Series:
    h1 = pairs_df.model_1.isin(holdout_models)
    h2 = pairs_df.model_2.isin(holdout_models)
    hi = pairs_df.item.isin(holdout_items)
    out = pd.Series("drop", index=pairs_df.index)
    out[hi & h1 & h2] = "eval_mm"
    out[hi & (h1 ^ h2)] = "eval_mixed"
    out[hi & ~h1 & ~h2] = "eval_items"
    out[~hi & ~h1 & ~h2] = "train"
    return out


def subsample_stratified(df, n_max, by, seed) -> pd.DataFrame:
    """Keep at most n_max unique pair_uids, round-robin across `by` strata.
    Both orders of a kept pair stay together."""
    uids = df.drop_duplicates("pair_uid")[["pair_uid", by]]
    if len(uids) <= n_max:
        return df
    rng = np.random.default_rng(seed)
    strata = {k: list(rng.permutation(g.pair_uid.values))
              for k, g in uids.groupby(by)}
    keep = []
    order = sorted(strata)
    while len(keep) < n_max and strata:
        for k in list(order):
            if k not in strata:
                continue
            keep.append(strata[k].pop())
            if not strata[k]:
                del strata[k]
            if len(keep) >= n_max:
                break
        order = [k for k in order if k in strata]
    return df[df.pair_uid.isin(set(keep))]


def question_text(sample) -> str:
    inp = sample.input
    if isinstance(inp, str):
        return inp
    for msg in inp:
        if getattr(msg, "role", None) == "user":
            return prompts_content(msg.content)
    return str(inp)


def prompts_content(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(getattr(b, "text", "") for b in content
                     if getattr(b, "type", None) == "text")


def load_model_log(path: str, cache_dir: Path, max_len_chars: int) -> pd.DataFrame:
    """Per-item first-correct transcript + model-level accuracy, cached."""
    from inspect_ai.log import read_eval_log
    from judge.util import extract_transcript, extract_score

    stem = Path(path).stem
    cache = cache_dir / f"{stem}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        if "text_alt" in df.columns:
            return df

    log = read_eval_log(path)
    n_total, n_correct_samples = 0, 0
    correct = {}  # item -> [(epoch, sample), ...]
    for s in log.samples:
        n_total += 1
        ok = extract_score(s)
        n_correct_samples += ok
        if ok:
            correct.setdefault(s.id, []).append((s.epoch, s))
    rows = []
    for item, hits in correct.items():
        hits.sort(key=lambda t: t[0])
        (epoch, s), rest = hits[0], hits[1:]
        rows.append(dict(
            model=stem, item=item, epoch=epoch,
            text=extract_transcript(s, include_reasoning=False)[:max_len_chars],
            # second correct epoch, for same-model control pairs
            text_alt=(extract_transcript(rest[0][1], include_reasoning=False)[:max_len_chars]
                      if rest else None),
            options_alt=(render_choices(rest[0][1].choices)
                         if rest and rest[0][1].choices else ""),
            options=render_choices(s.choices) if s.choices else "",
            question=question_text(s),
            target=str(s.target),
        ))
    df = pd.DataFrame(rows)
    df["n_samples"] = n_total
    df["accuracy"] = n_correct_samples / n_total if n_total else 0.0
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def accuracy_logit(p: float, n: int) -> float:
    p = min(max(p, 1 / n), 1 - 1 / n)
    return float(np.log(p / (1 - p)))


def token_filter(df, tok, max_input_tokens, batch=256):
    """Drop pairs whose rendered prompt exceeds max_input_tokens.
    Returns (kept_df, n_pairs_dropped)."""
    texts = []
    for r in df.itertuples():
        msgs = [{"role": "system", "content": prompts.SYSTEM_PROMPT_SHORT},
                {"role": "user", "content": prompts.user_text(
                    r.question, r.options_1, r.options_2, r.text_1, r.text_2)}]
        texts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False) + "ANSWER: ")
    lens = np.empty(len(texts), dtype=int)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=False)
        lens[i:i + len(enc["input_ids"])] = [len(x) for x in enc["input_ids"]]
    over = df[lens > max_input_tokens].pair_uid.unique()
    return df[~df.pair_uid.isin(set(over))], len(over)


def build_pairs(meta, caps, groups, items, min_cap_gap):
    """meta: {(model, item): row}; caps/groups: model -> value. Rows for every
    both-correct cross-group pair with enough capability gap."""
    models = sorted(caps)
    by_item = {}
    for (m, item) in meta:
        by_item.setdefault(item, []).append(m)
    all_rows = []
    for item in sorted(items):
        present = sorted(set(by_item.get(item, [])))
        for m1, m2 in itertools.combinations(present, 2):
            if groups[m1] == groups[m2]:
                continue
            if abs(caps[m1] - caps[m2]) < min_cap_gap:
                continue
            r1, r2 = meta[(m1, item)], meta[(m2, item)]
            a = dict(model=m1, cap=caps[m1], text=r1["text"], options=r1["options"])
            b = dict(model=m2, cap=caps[m2], text=r2["text"], options=r2["options"])
            all_rows.extend(make_pair_rows(item, a, b, question=r1["question"]))
    return all_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    prep = cfg["prep"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "transcripts_cache"

    # per-model transcripts, epoch pool
    paths = sorted(glob.glob(cfg["epoch_glob"]))
    frames = [load_model_log(p, cache_dir, prep["max_len_chars"]) for p in paths]
    print(f"loaded {len(frames)} epoch logs")

    stats = []
    for f in frames:
        if f.empty:
            continue
        m = f.model.iloc[0]
        stats.append(dict(model=m, base_group=base_group(m),
                          n_correct=len(f), accuracy=f.accuracy.iloc[0],
                          logit=accuracy_logit(f.accuracy.iloc[0], int(f.n_samples.iloc[0]))))
    models_df = pd.DataFrame(stats)
    models_df = models_df[(models_df.accuracy >= prep["min_accuracy"])
                          & (models_df.accuracy <= prep["max_accuracy"])].reset_index(drop=True)
    holdout_groups = set(prep["holdout_models"])
    models_df["holdout"] = models_df.base_group.isin(holdout_groups)
    models_df.sort_values("logit", ascending=False).to_csv(out_dir / "models_ft.csv", index=False)
    surviving = set(models_df.model)
    print(f"models: {len(stats)} read, {len(models_df)} surviving accuracy filter")

    # item split
    all_items = sorted({i for f in frames for i in f.item})
    rng = np.random.default_rng(prep["seed"])
    shuffled = list(rng.permutation(all_items))
    n_hold = int(round(prep["item_holdout_frac"] * len(all_items)))
    holdout_items = set(shuffled[:n_hold])
    pd.DataFrame(dict(item=all_items,
                      holdout=[i in holdout_items for i in all_items])
                 ).to_csv(out_dir / "item_split.csv", index=False)
    print(f"items: {len(all_items)} total, {len(holdout_items)} held out")

    # items_ft: question/options/target per item, from the alphabetically
    # first log that has the item (options and target from the same log stay
    # mutually consistent)
    items_ft = (pd.concat(frames)[["item", "question", "options", "target"]]
                .drop_duplicates("item", keep="first")
                .rename(columns=dict(options="options_rendered", target="target_letter")))
    items_ft.to_parquet(out_dir / "items_ft.parquet")

    # pair build
    min_reasoning = prep["min_reasoning_chars"]
    meta, n_degen = {}, 0
    for f in frames:
        for r in f.itertuples():
            if r.model not in surviving:
                continue
            if not has_reasoning(r.text, min_reasoning):
                n_degen += 1
                continue
            meta[(r.model, r.item)] = dict(text=r.text, options=r.options, question=r.question)
    print(f"transcripts dropped (answer-only <{min_reasoning} reasoning chars): {n_degen}")
    caps = dict(zip(models_df.model, models_df.logit))
    groups = dict(zip(models_df.model, models_df.base_group))
    rows = build_pairs(meta, caps, groups, all_items, prep["min_cap_gap"])
    pairs = pd.DataFrame(rows)
    pairs["mpair"] = ["|".join(u.split("|")[1:]) for u in pairs.pair_uid]
    print(f"candidate pairs: {pairs.pair_uid.nunique()}")

    holdout_model_names = set(models_df.model[models_df.holdout])
    pairs["split"] = assign_splits(pairs, holdout_model_names, holdout_items)
    pairs = pairs[pairs.split != "drop"]

    # budgets, then token filter on the survivors only (full pool is too big to tokenize)
    kept = []
    for split, budget in [("train", prep["max_train_pairs"]),
                          ("eval_mm", prep["max_eval_pairs_per_split"]),
                          ("eval_mixed", prep["max_eval_pairs_per_split"]),
                          ("eval_items", prep["max_eval_pairs_per_split"])]:
        sub = pairs[pairs.split == split]
        if len(sub):
            kept.append(subsample_stratified(sub, budget, "mpair", prep["seed"]))
    pairs = pd.concat(kept, ignore_index=True) if kept else pairs.iloc[:0]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(prep["tokenizer"])
    pairs, n_over = token_filter(pairs, tok, prep["max_input_tokens"])
    print(f"pairs dropped for length: {n_over}")

    # carve val out of train by item
    train_items = sorted(pairs.item[pairs.split == "train"].unique())
    n_val = int(round(cfg["train"]["val_frac"] * len(train_items)))
    val_items = set(rng.permutation(train_items)[:n_val])
    pairs.loc[(pairs.split == "train") & pairs.item.isin(val_items), "split"] = "val"

    check_counterbalance(pairs)
    pairs.drop(columns=["mpair"]).to_parquet(out_dir / "pairs.parquet")
    print(pairs.groupby("split").pair_uid.nunique().rename("pairs"))
    print("label balance:", pairs.groupby("split").label.mean().round(3).to_dict())

    # self-preference pairs from the pilot pool
    from inspect_ai.log import read_eval_log
    pilot_caps = pd.read_csv(cfg["pilot_models_csv"]).set_index("model").logit
    pilot_frames = []
    for p in sorted(glob.glob(cfg["pilot_glob"])):
        name = read_eval_log(p, header_only=True).eval.model.split("/")[-1]
        f = load_model_log(p, cache_dir, prep["max_len_chars"])
        if name not in pilot_caps.index:
            print(f"pilot model {name} not in {cfg['pilot_models_csv']}, skipped")
            continue
        f = f.assign(model=name)
        pilot_frames.append(f)
    pmeta = {(r.model, r.item): dict(text=r.text, options=r.options, question=r.question)
             for f in pilot_frames for r in f.itertuples()
             if has_reasoning(r.text, min_reasoning)}
    pmodels = sorted({m for m, _ in pmeta})
    assert "qwen3-32b" in pmodels, pmodels
    sp_rows = []
    for item in sorted(holdout_items):
        for other in pmodels:
            if other == "qwen3-32b":
                continue
            if ("qwen3-32b", item) not in pmeta or (other, item) not in pmeta:
                continue
            a = dict(model="qwen3-32b", cap=float(pilot_caps["qwen3-32b"]),
                     **{k: pmeta[("qwen3-32b", item)][k] for k in ("text", "options")})
            b = dict(model=other, cap=float(pilot_caps[other]),
                     **{k: pmeta[(other, item)][k] for k in ("text", "options")})
            sp_rows.extend(make_pair_rows(item, a, b,
                                          question=pmeta[("qwen3-32b", item)]["question"]))
    sp = pd.DataFrame(sp_rows)
    sp, sp_over = token_filter(sp, tok, prep["max_input_tokens"])
    sp["split"] = "selfpref"
    check_counterbalance(sp)
    sp.drop(columns=["split"]).to_parquet(out_dir / "pairs_selfpref.parquet")
    print(f"selfpref pairs: {sp.pair_uid.nunique()} ({sp_over} dropped for length)")

    # same-model control pairs: two correct epochs of one model, held-out
    # items; a judge without spurious signal sits at chance on these
    self_rows = []
    for f in frames:
        for r in f.itertuples():
            # text_alt is NaN (not None) after the parquet round trip
            if (r.model not in surviving or r.item not in holdout_items
                    or not isinstance(r.text_alt, str) or not r.text_alt
                    or not has_reasoning(r.text, min_reasoning)
                    or not has_reasoning(r.text_alt, min_reasoning)):
                continue
            a = dict(model=f"{r.model}#e1", cap=0.0, text=r.text, options=r.options)
            b = dict(model=f"{r.model}#e2", cap=0.0, text=r.text_alt, options=r.options_alt)
            rows2 = make_pair_rows(r.item, a, b, question=r.question)
            # equal caps make the label degenerate; re-point it at the
            # first-epoch side so counterbalance holds and P(picked==label)
            # reads P(pick epoch 1) = 0.5 under the null
            rows2[0]["label"], rows2[1]["label"] = 1, 2
            self_rows.extend(rows2)
    sf = pd.DataFrame(self_rows)
    sf["mpair"] = [u.split("|", 1)[1] for u in sf.pair_uid]
    sf = subsample_stratified(sf, prep["max_self_pairs"], "mpair", prep["seed"])
    sf, sf_over = token_filter(sf.drop(columns=["mpair"]), tok, prep["max_input_tokens"])
    sf["split"] = "self"
    check_counterbalance(sf)
    sf.drop(columns=["split"]).to_parquet(out_dir / "pairs_self.parquet")
    print(f"self pairs: {sf.pair_uid.nunique()} over {sf.model_1.str.split('#').str[0].nunique()} "
          f"models ({sf_over} dropped for length)")


if __name__ == "__main__":
    main()
