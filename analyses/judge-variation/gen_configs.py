"""Generate judging-variation configs: one yaml per bench x condition.

Usage: uv run python gen_configs.py --bench gpqa --seed 0
Benches: gpqa, otis-aime, chess (dir name external_data/epoch-<bench>).
Condition e also gets a _native variant: same pairs, sample_ids restricted to
items whose untruncated pair prompt fits a 32k-token judge.
"""
import argparse
import random
from pathlib import Path

import yaml

import prompts
from pool import load_members, sample_pairs, cot_ok, both_correct_ids

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
BINS = [(5, 15), (15, 25), (25, 40)]
PER_BIN = 8
PER_BIN_E = 8  # CoT-able pool is small (~10-13 members); take what the bins hold
ITEMS_PER_PAIR = 60            # per direction; reported, not silent
NATIVE_BUDGET_CHARS = 100_000  # ~25k tokens at 4 chars/tok, leaves headroom in 32k

CONDITIONS = {
    "a_baseline": dict(system_prompt=prompts.BASELINE, include_reasoning=False, max_length=65536),
    "b_ignore_superficial": dict(system_prompt=prompts.IGNORE_SUPERFICIAL, include_reasoning=False, max_length=65536),
    "c_ignore_superficial_confidence": dict(system_prompt=prompts.IGNORE_SUPERFICIAL_CONFIDENCE, include_reasoning=False, max_length=65536),
    "d_pick_weaker": dict(system_prompt=prompts.PICK_WEAKER, include_reasoning=False, max_length=65536, invert_at_analysis=True),
    "e_with_reasoning": dict(system_prompt=prompts.BASELINE, include_reasoning=True, max_length=250000),
}


def pair_fits_native(sample_a, sample_b):
    from judge.util import extract_transcript
    total = len(extract_transcript(sample_a, include_reasoning=True)) + \
            len(extract_transcript(sample_b, include_reasoning=True))
    return total <= NATIVE_BUDGET_CHARS


def emit_native_split(cfg, out_dir, bench):
    """e_native = same config, sample_ids restricted to items fitting 32k untruncated."""
    from inspect_ai.log import read_eval_log
    logs = {s: read_eval_log(str(ROOT / p)) for s, p in cfg["transcripts"].items()}
    by_id = {s: {x.id: x for x in log.samples} for s, log in logs.items()}
    native_ids = {}
    for a, b in cfg["comparisons"]:
        key = f"{a}-vs-{b}"
        fit = [i for i in cfg["sample_ids"][key]
               if pair_fits_native(by_id[a][i], by_id[b][i])]
        native_ids[key] = fit
        print(f"  native fit {key}: {len(fit)}/{len(cfg['sample_ids'][key])}")
    native = dict(cfg, log_root=cfg["log_root"] + "_native", sample_ids=native_ids)
    (out_dir / f"{bench}_e_with_reasoning_native.yaml").write_text(
        yaml.safe_dump(native, sort_keys=False, width=100000))


EXT_BINS = [(2, 5), (5, 15), (15, 25), (25, 40)]
EXT_PER_BIN = [6, 8, 8, 6]
# per bin for the typed e-extension: (stronger pool, weaker pool, take)
EXT_E_MIX = [("cot", "cot", 1), ("cot", "plain", 4), ("plain", "cot", 4), ("plain", "plain", 3)]


def extend(bench, seed):
    """Append new pairs to the existing yamls, matched across conditions.

    One untyped extension set (denser bins incl. near-ties) goes to a-d; a
    typed set stratified by member kind (cot x cot, cot x plain both ways,
    plain x plain) goes to e AND a-d, so e keeps its matched baseline and the
    mixed/without pairings exist under cot-shown. Existing pairs stay
    byte-identical, so finished pair dirs keep resuming.
    """
    members = load_members(ROOT / "external_data" / f"epoch-{bench}",
                           ROOT / "model-sampler" / "model_pool.json")
    pools = {"cot": [m for m in members if cot_ok(m["path"])],
             "plain": [m for m in members if not cot_ok(m["path"])], "any": members}
    rng = random.Random(seed)
    base_cfg = yaml.safe_load((HERE / "configs" / f"{bench}_a_baseline.yaml").read_text())
    taken = {tuple(p) for p in base_cfg["comparisons"]}

    def sample(pool_a, pool_b, lo, hi, take):
        eligible = [(a, b) for a in pools[pool_a] for b in pools[pool_b]
                    if lo <= a["eci"] - b["eci"] < hi
                    and (a["short"], b["short"]) not in taken]
        got = rng.sample(eligible, min(take, len(eligible)))
        taken.update((a["short"], b["short"]) for a, b in got)
        return got

    ext_main, ext_e = [], []
    for (lo, hi), take in zip(EXT_BINS, EXT_PER_BIN):
        ext_main += sample("any", "any", lo, hi, take)
        for pa, pb, k in EXT_E_MIX:
            ext_e += sample(pa, pb, lo, hi, k)

    # both-correct filter once per pair so every condition drops the same ones
    def with_ids(pairs):
        out = []
        for a, b in pairs:
            bc = both_correct_ids(a["path"], b["path"])
            if len(bc) >= 3:
                out.append((a, b, bc[:ITEMS_PER_PAIR]))
        return out

    ext_main, ext_e = with_ids(ext_main), with_ids(ext_e)
    print(f"{bench}: ext_main {len(ext_main)}, ext_e {len(ext_e)} "
          f"(cotxcot/mixed/plainxplain stratified)")

    for cond, spec in CONDITIONS.items():
        path = HERE / "configs" / f"{bench}_{cond}.yaml"
        cfg = yaml.safe_load(path.read_text())
        add = ext_e if spec["include_reasoning"] else ext_main + ext_e
        existing = {tuple(p) for p in cfg["comparisons"]}
        added = 0
        for a, b, ids in add:
            if (a["short"], b["short"]) in existing:
                continue
            cfg["comparisons"].append([a["short"], b["short"]])
            cfg["sample_ids"][f"{a['short']}-vs-{b['short']}"] = ids
            cfg["transcripts"][a["short"]] = str(a["path"].resolve().relative_to(ROOT.resolve()))
            cfg["transcripts"][b["short"]] = str(b["path"].resolve().relative_to(ROOT.resolve()))
            added += 1
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=100000))
        print(f"{bench}_{cond}: +{added} pairs -> {len(cfg['comparisons'])}")
        if cond == "e_with_reasoning" and added:
            emit_native_split(cfg, HERE / "configs", bench)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extend", action="store_true",
                    help="append denser-bin pairs to existing configs (seed should differ from the base run)")
    args = ap.parse_args()
    if args.extend:
        extend(args.bench, args.seed)
        return

    members = load_members(ROOT / "external_data" / f"epoch-{args.bench}",
                           ROOT / "model-sampler" / "model_pool.json")
    pairs_main = sample_pairs(members, BINS, PER_BIN, args.seed)
    # e needs both members' CoT visible, so its pairs come from the CoT-able
    # subset; a-d also judge these pairs, giving e its same-pair baseline.
    cot_members = [m for m in members if cot_ok(m["path"])]
    print(f"cot-available members: {len(cot_members)}")
    pairs_e = sample_pairs(cot_members, BINS, PER_BIN_E, args.seed)
    seen = {(a["short"], b["short"]) for a, b in pairs_main}
    pairs_all = pairs_main + [p for p in pairs_e
                              if (p[0]["short"], p[1]["short"]) not in seen]

    out_dir = HERE / "configs"
    out_dir.mkdir(exist_ok=True)
    for cond, spec in CONDITIONS.items():
        pairs = pairs_e if spec["include_reasoning"] else pairs_all
        rows, ids, transcripts = [], {}, {}
        for a, b in pairs:
            bc = both_correct_ids(a["path"], b["path"])
            if len(bc) < 3:
                print(f"  drop {a['short']}-vs-{b['short']}: only {len(bc)} both-correct")
                continue
            rows.append([a["short"], b["short"]])
            ids[f"{a['short']}-vs-{b['short']}"] = bc[:ITEMS_PER_PAIR]
            # repo-root-relative so configs work on the cluster too
            transcripts[a["short"]] = str(a["path"].resolve().relative_to(ROOT.resolve()))
            transcripts[b["short"]] = str(b["path"].resolve().relative_to(ROOT.resolve()))
        cfg = dict(
            log_root=f"logs/judging-variations/{args.bench}/{cond}",
            condition=cond, seed=args.seed, items_per_pair=ITEMS_PER_PAIR,
            transcripts=transcripts, comparisons=rows, sample_ids=ids, **spec,
        )
        (out_dir / f"{args.bench}_{cond}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, width=100000))
        print(f"{args.bench}_{cond}: {len(rows)} pairs")
        if cond == "e_with_reasoning" and rows:
            emit_native_split(cfg, out_dir, args.bench)


if __name__ == "__main__":
    main()
