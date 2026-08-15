"""Generate the fixed campaign subset and the 8 truncation-condition configs.

One subset of GPQA pairs and items, reused by every manipulation condition,
so all comparisons stay within-pair and within-item. Sampling is fixed-seed
and stratified by ECI gap (small/medium/large). Every chosen (pair, item) is
checked against the full-transcript baseline in judging_variations/verdicts.csv
before it is used - a pair with too little baseline coverage is skipped in
favour of the next candidate in its gap bin.

Usage:
  uv run python scripts/pairwise-newjudges/gen_manipulation_configs.py
"""
import random
import re
from pathlib import Path

import pandas as pd
import yaml

from configs.experiment import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[2]
GPQA_CONFIG = ROOT / "scripts/pairwise-newjudges/configs/gpqa.yaml"
ECI_CONFIG = ROOT / "scripts/analyses/judge-elo/configs/gpqa.yaml"
BASELINE_VERDICTS = ROOT / "judge_results/judging_variations/verdicts.csv"
OUT_DIR = ROOT / "scripts/pairwise-newjudges/configs/manipulations"

SEED = 0
N_PAIRS = 50
# Items drawn per pair. Bump and rerun to change the campaign size - nothing
# else in this file depends on the value.
ITEMS_PER_PAIR = 30
# How the 50 pairs split across the three ECI-gap bins (must sum to N_PAIRS).
GAP_BIN_TARGETS = {"small": 17, "medium": 17, "large": 16}
assert sum(GAP_BIN_TARGETS.values()) == N_PAIRS, "GAP_BIN_TARGETS must sum to N_PAIRS"
BASELINE_CONDITION = "a_baseline"
MAX_LENGTH = 65536
# Only used to check baseline coverage - never written into a config, so
# adding a fourth judge later is a CLI change to run.py, not a config edit.
COVERAGE_JUDGES = ["qwen3-8b", "qwen3-32b", "gpt-oss-120b"]
# Drop any pair naming an answer model matching this regex (case-insensitive)
# from the candidate pool before stratified sampling. None = no exclusion.
# Live campaign policy: exclude -flash/-low answer models (cheap/distilled
# variants), deepseek stays in - set here, not passed at the call site, so
# rerunning this file with no edits reproduces the live subset.
EXCLUDE_ANSWER_MODEL_PATTERN: str | None = r"-flash|-low"

# (condition name, truncate_keep, truncate_frac). The _100 conditions are the
# length-matching controls: frac=1.0 cuts only the longer transcript, down to
# the shorter one's length, from the front (first) or the back (end).
TRUNCATION_CONDITIONS = [
    ("trunc_first_10", "first", 0.10),
    ("trunc_first_50", "first", 0.50),
    ("trunc_first_90", "first", 0.90),
    ("trunc_first_100", "first", 1.00),
    ("trunc_end_10", "last", 0.10),
    ("trunc_end_50", "last", 0.50),
    ("trunc_end_90", "last", 0.90),
    ("trunc_end_100", "last", 1.00),
]


def pair_key(major: str, minor: str) -> str:
    return f"{major}-vs-{minor}"


def filtered_comparisons(comparisons: list, exclude_pattern: str | None) -> list:
    """Drop pairs naming an answer model matching exclude_pattern (case-insensitive)."""
    if not exclude_pattern:
        return comparisons
    rx = re.compile(exclude_pattern, re.IGNORECASE)
    return [p for p in comparisons if not (rx.search(p[0]) or rx.search(p[1]))]


def gap_bins(comparisons: list, eci: dict) -> tuple[dict, tuple[float, float]]:
    """Split the 160 pairs into three equal-sized bins by rank on |ECI gap|."""
    ranked = sorted(comparisons, key=lambda p: abs(eci[p[0]] - eci[p[1]]))
    n = len(ranked)
    a, b = n // 3, 2 * n // 3
    bins = {"small": ranked[:a], "medium": ranked[a:b], "large": ranked[b:]}
    gaps = [abs(eci[p[0]] - eci[p[1]]) for p in ranked]
    boundaries = (gaps[a - 1], gaps[b - 1])  # gap value at each bin edge
    return bins, boundaries


def covered_items(verdicts: pd.DataFrame, major: str, minor: str) -> list[str]:
    """Items with a successfully-parsed baseline row for every (judge, order) combination."""
    rows = verdicts[(verdicts.major == major) & (verdicts.minor == minor) & verdicts.parse_ok]
    needed = {(j, o) for j in COVERAGE_JUDGES for o in (1, 2)}
    covered = []
    for item_id, group in rows.groupby("item_id"):
        if needed <= set(zip(group.judge, group.order)):
            covered.append(item_id)
    return sorted(covered)


def select_pairs(bins: dict, verdicts: pd.DataFrame) -> tuple[dict, list[str]]:
    """Pick N_PAIRS pairs per GAP_BIN_TARGETS; skip pairs with too little coverage."""
    chosen = {}
    log = []
    for bin_name, target in GAP_BIN_TARGETS.items():
        pool = list(bins[bin_name])
        random.Random(f"{SEED}-{bin_name}").shuffle(pool)
        picked, cov = [], {}
        for major, minor in pool:
            if len(picked) == target:
                break
            items = covered_items(verdicts, major, minor)
            if len(items) >= ITEMS_PER_PAIR:
                picked.append((major, minor))
                cov[(major, minor)] = items
            else:
                log.append(f"  resample: {pair_key(major, minor)} ({bin_name}) "
                           f"only {len(items)}/{ITEMS_PER_PAIR} covered items - skipped")
        if len(picked) < target:
            raise RuntimeError(f"ran out of {bin_name}-gap pairs with full coverage "
                                f"({len(picked)}/{target})")
        chosen[bin_name] = [(major, minor, cov[(major, minor)]) for major, minor in picked]
    return chosen, log


def sample_pair_items(major: str, minor: str, covered: list[str]) -> list[str]:
    return sorted(random.Random(f"{SEED}-{major}-{minor}").sample(covered, ITEMS_PER_PAIR))


def build_subset(chosen: dict, transcripts: dict) -> tuple[list, dict, dict]:
    comparisons, sample_ids, used_transcripts = [], {}, {}
    for bin_name in ("small", "medium", "large"):
        for major, minor, covered in sorted(chosen[bin_name]):
            ids = sample_pair_items(major, minor, covered)
            comparisons.append([major, minor])
            sample_ids[pair_key(major, minor)] = ids
            used_transcripts[major] = transcripts[major]
            used_transcripts[minor] = transcripts[minor]
    return comparisons, sample_ids, used_transcripts


def write_configs(comparisons: list, sample_ids: dict, transcripts: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, keep, frac in TRUNCATION_CONDITIONS:
        cfg = {
            "log_root": f"logs/pairwise-newjudges/manipulations/{name}",
            "condition": name,
            "seed": SEED,
            "items_per_pair": ITEMS_PER_PAIR,
            "transcripts": transcripts,
            "comparisons": comparisons,
            "sample_ids": sample_ids,
            "truncate_fracs": [frac],
            "truncate_keep": keep,
            "include_reasoning": False,
            "max_length": MAX_LENGTH,
            "system_prompt": SYSTEM_PROMPT,
        }
        out_path = OUT_DIR / f"{name}.yaml"
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        print(f"wrote {out_path}")


def print_coverage_table(chosen: dict, verdicts: dict) -> None:
    print("\ncoverage table (pair, bin, gap, covered items used / needed)")
    eci = yaml.safe_load(ECI_CONFIG.read_text())["eci"]
    for bin_name in ("small", "medium", "large"):
        for major, minor, covered in sorted(chosen[bin_name]):
            gap = abs(eci[major] - eci[minor])
            needed = ITEMS_PER_PAIR * len(COVERAGE_JUDGES) * 2
            have = min(len(covered), ITEMS_PER_PAIR) * len(COVERAGE_JUDGES) * 2
            print(f"  {bin_name:6s} gap={gap:6.2f}  {pair_key(major, minor):55s} "
                  f"{have}/{needed} rows")


def main() -> None:
    gpqa_cfg = yaml.safe_load(GPQA_CONFIG.read_text())
    eci = yaml.safe_load(ECI_CONFIG.read_text())["eci"]
    verdicts = pd.read_csv(BASELINE_VERDICTS)
    baseline = verdicts[(verdicts.bench == "gpqa") & (verdicts.condition == BASELINE_CONDITION)]

    pool = filtered_comparisons(gpqa_cfg["comparisons"], EXCLUDE_ANSWER_MODEL_PATTERN)
    if EXCLUDE_ANSWER_MODEL_PATTERN:
        print(f"EXCLUDE_ANSWER_MODEL_PATTERN={EXCLUDE_ANSWER_MODEL_PATTERN!r}: "
              f"{len(gpqa_cfg['comparisons']) - len(pool)}/{len(gpqa_cfg['comparisons'])} "
              f"pairs dropped from the candidate pool before stratification\n")

    bins, boundaries = gap_bins(pool, eci)
    print(f"gap bins (tertile boundaries at {boundaries[0]:.2f} and {boundaries[1]:.2f}):")
    for name in ("small", "medium", "large"):
        print(f"  {name}: {len(bins[name])} pairs available, target {GAP_BIN_TARGETS[name]}")

    chosen, resample_log = select_pairs(bins, baseline)
    if resample_log:
        print("\nresampled pairs:")
        print("\n".join(resample_log))
    else:
        print("\nno resampling needed - every candidate had full baseline coverage")

    print_coverage_table(chosen, baseline)

    comparisons, sample_ids, transcripts = build_subset(chosen, gpqa_cfg["transcripts"])
    write_configs(comparisons, sample_ids, transcripts)

    n_pairs = len(comparisons)
    n_conditions = len(TRUNCATION_CONDITIONS)
    n_judges = len(COVERAGE_JUDGES)
    total_calls = n_pairs * ITEMS_PER_PAIR * 2 * n_judges * n_conditions
    print(f"\n{n_pairs} pairs x {ITEMS_PER_PAIR} items x 2 orders x {n_judges} judges "
          f"x {n_conditions} conditions = {total_calls} judge calls")


if __name__ == "__main__":
    main()
