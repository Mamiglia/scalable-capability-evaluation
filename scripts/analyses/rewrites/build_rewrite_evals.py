"""Convert paraphrase rewrites into per-model judge-readable .eval logs, plus the
matching manipulation config and pairs parquet.

Same conversion pattern as scripts/analyses/style_capability/answers_to_evals.py:
clone each covered item's original sample (question/choices/target unchanged),
swap in the rewrite as a mockllm completion scored by choice(), drop items the
model doesn't need. require_both_correct still works on the built store because
the rewrite keeps the original answer letter - checked at build time (see
rewrite_answer_letter): a rewrite whose final answer line is missing or
doesn't match the original's target is dropped, not silently kept, since the
rewrite QC (word-ratio floor) does not check this.

Parameterised by --rewriter, so a not-yet-landed set (e.g. gpt, currently only a
10-row pilot) is one command away and simply refuses to run until its jsonl has
every cell the subset needs - never silently built from a partial file.

Usage:
  uv run python scripts/analyses/rewrites/build_rewrite_evals.py --rewriter haiku
  uv run python scripts/analyses/rewrites/build_rewrite_evals.py --rewriter gpt
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import zipfile_zstd  # noqa: F401  epoch .eval zips use zstd; py<3.14 zipfile needs this shim
except ImportError:
    pass

import pandas as pd
import yaml
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from normalise_prefix import strip_role_prefix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/fine-tuning"))  # prep_pairs pair-construction helpers

from configs.experiment import SYSTEM_PROMPT  # noqa: E402
from judge.util import extract_score, first_correct_by_id  # noqa: E402

# any manipulations config names the same fixed 50-pair x 15-item subset
SUBSET_CONFIG = ROOT / "scripts/pairwise-newjudges/configs/manipulations/trunc_first_10.yaml"
EXCLUDED = ROOT / "judge_results/rewrites/excluded.json"
REWRITE_FILES = {
    "haiku": ROOT / "judge_results/rewrites/paraphrase_haiku.jsonl",
    "gpt": ROOT / "judge_results/rewrites/paraphrase_gpt.jsonl",
}
SOURCE_DIR = ROOT / "external_data/epoch-gpqa"
STORE_ROOT = ROOT / "data/rewrites"
MANIP_DIR = ROOT / "scripts/pairwise-newjudges/configs/manipulations"
PARQUET_DIR = ROOT / "judge_results/rewrites"
MODELS_CSV = ROOT / "judge_results/judge_ft/models_ft.csv"

# last literal "ANSWER: <letter>" anywhere in the text - same marker
# strip_answer()/choice()'s own template rely on; case-sensitive, matching
# every original transcript's actual convention (verified: 0 false positives
# across all 1227 originals, which use assorted trailing markdown/punctuation
# after the letter but always this literal marker before it).
ANSWER_RE = re.compile(r"ANSWER:\s*([A-Za-z])")


def subset_keys(cfg: dict) -> set[tuple[str, str]]:
    """(model, item_id) pairs used anywhere in the fixed campaign subset."""
    keys = set()
    for major, minor in cfg["comparisons"]:
        items = cfg["sample_ids"][f"{major}-vs-{minor}"]
        keys.update((major, i) for i in items)
        keys.update((minor, i) for i in items)
    return keys


def excluded_keys(path: Path) -> set[tuple[str, str]]:
    """Cells that failed the word-ratio QC floor - dropped from every condition."""
    cells = json.loads(path.read_text())["cells"]
    return {(c["model"], c["item_id"]) for c in cells}


def load_rewrites(path: Path) -> dict[tuple[str, str], str]:
    """(model, item_id) -> rewritten text, role prefix stripped so the store
    never double-prefixes a rewriter that echoed it (gpt did, haiku didn't)."""
    rows = (json.loads(l) for l in path.read_text().splitlines() if l)
    return {(r["model"], r["item_id"]): strip_role_prefix(r["rewritten"]) for r in rows}


def rewrite_answer_letter(text: str) -> str | None:
    """Last 'ANSWER: <letter>' in the rewrite, or None if it has no such
    marker at all (e.g. "Final Answer: B" - a different string, not this
    one). Deliberately tolerant of what follows the letter (markdown,
    trailing punctuation, code fences) since real transcripts use all of
    these; only the marker+letter itself is required."""
    matches = list(ANSWER_RE.finditer(text))
    return matches[-1].group(1).upper() if matches else None


def build_model_store(model: str, items: list[str], rewrites: dict,
                       out_dir: Path) -> tuple[list[str], list[dict]]:
    """Clone model's epoch-gpqa log: keep only `items` whose rewrite carries a
    well-formed final answer matching the original's target, swap in that
    text, drop the rest. Returns (built_items, bad) - bad lists every item
    dropped for a missing/mismatched answer, never silently kept.

    rewrite_answer_letter is a regex, not the judge pipeline's actual scorer,
    so it can itself be fooled (e.g. a rewrite reading "Final Answer: ANSWER:
    A" passes the letter check but trips choice()'s own parser into reading
    the literal word "ANSWER" as the answer, scoring "I"). So after building,
    the written log is read back and re-checked against extract_score - the
    same both-correct signal the rest of the pipeline trusts - and anything
    that still isn't "C" is dropped and the store rebuilt without it, rather
    than shipped on the strength of the regex alone."""
    log = read_eval_log(str(SOURCE_DIR / f"{model}.eval"))
    correct = first_correct_by_id(log)
    bad, keep = [], []
    for item in sorted(items):
        group = correct.get(item)
        if not group:
            raise ValueError(f"no correct epoch for {model}/{item} in {SOURCE_DIR}")
        s = group[0]
        got = rewrite_answer_letter(rewrites[(model, item)])
        if got != s.target.upper():
            bad.append(dict(model=model, item=item, target=s.target, got=got,
                             reason="answer letter missing or mismatched"))
            continue
        keep.append(item)

    while keep:
        samples, outputs = [], []
        for item in keep:
            s = correct[item][0]
            samples.append(Sample(id=item, input=s.input, choices=list(s.choices), target=s.target))
            outputs.append(ModelOutput.from_content("mockllm/model", rewrites[(model, item)]))
        task = Task(dataset=MemoryDataset(samples), solver=multiple_choice(),
                    scorer=choice(), name="gpqa_diamond")
        with tempfile.TemporaryDirectory() as tmp:
            inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
                         log_dir=tmp, display="none")
            [produced] = Path(tmp).glob("*.eval")
            written = read_eval_log(str(produced))
            failed = {s.id for s in written.samples if not extract_score(s)}
            if not failed:
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(produced), str(out_dir / f"{model}.eval"))
                break
        for item in failed:
            bad.append(dict(model=model, item=item, target=correct[item][0].target,
                             got=rewrite_answer_letter(rewrites[(model, item)]),
                             reason="letter matched but scored incorrect by the real scorer"))
        keep = [i for i in keep if i not in failed]
    return keep, bad


def filtered_sample_ids(cfg: dict, covered: set) -> dict[str, list[str]]:
    """Per-pair item lists trimmed to items both sides of the pair kept."""
    out = {}
    for major, minor in cfg["comparisons"]:
        key = f"{major}-vs-{minor}"
        out[key] = [i for i in cfg["sample_ids"][key]
                    if (major, i) in covered and (minor, i) in covered]
    return out


def write_config(cfg: dict, condition: str, models: list[str], sample_ids: dict) -> Path:
    out_cfg = {
        "log_root": f"logs/pairwise-newjudges/manipulations/{condition}",
        "condition": condition,
        "seed": cfg["seed"],
        "items_per_pair": cfg["items_per_pair"],
        "transcripts": {m: f"data/rewrites/{condition}/{m}.eval" for m in models},
        "comparisons": cfg["comparisons"],
        "sample_ids": sample_ids,
        "include_reasoning": False,
        "max_length": cfg["max_length"],
        "system_prompt": SYSTEM_PROMPT,
    }
    path = MANIP_DIR / f"{condition}.yaml"
    path.write_text(yaml.safe_dump(out_cfg, sort_keys=False, default_flow_style=False))
    return path


def write_pairs_parquet(cfg: dict, condition: str, models: list[str],
                         sample_ids: dict, store_dir: Path) -> tuple[Path, pd.DataFrame]:
    """Both-correct guard here is load_model_log's own extract_score filter
    (prep_pairs.py) - the same check src/judge/dataset.py's build_dataset
    applies via eligible(). That filter only drops what inspect's own
    scorer marked incorrect; it does not re-run rewrite_answer_letter, so a
    rewrite that fools the fuzzy choice() scorer (as build_model_store's
    check exists to catch before this point) would still pass it. Any
    (model, item) load_model_log drops is skipped here rather than raising
    a bare KeyError, and the count is reported - never a silent drop."""
    from prep_pairs import check_counterbalance, load_model_log, make_pair_rows

    caps = pd.read_csv(MODELS_CSV).set_index("model").logit
    cache_dir = STORE_ROOT / f"cache_{condition}"
    meta = {}
    for m in models:
        df = load_model_log(str(store_dir / f"{m}.eval"), cache_dir, cfg["max_length"])
        for r in df.itertuples():
            meta[(m, r.item)] = dict(text=r.text, options=r.options, question=r.question)

    rows, skipped = [], []
    for major, minor in cfg["comparisons"]:
        for item in sample_ids[f"{major}-vs-{minor}"]:
            if (major, item) not in meta or (minor, item) not in meta:
                skipped.append((major, minor, item))
                continue
            a = dict(model=major, cap=float(caps[major]), **meta[(major, item)])
            b = dict(model=minor, cap=float(caps[minor]), **meta[(minor, item)])
            rows.extend(make_pair_rows(item, a, b, question=meta[(major, item)]["question"]))
    if skipped:
        print(f"  {len(skipped)} pair-item(s) skipped from the parquet - not both-correct "
              f"in the built store (e.g. {skipped[:3]})")

    pairs = pd.DataFrame(rows)
    pairs["split"] = condition  # required by the reference schema; value unverified, no consumer yet
    check_counterbalance(pairs)
    out_path = PARQUET_DIR / f"pairs_gpqa_{condition}.parquet"
    pairs.to_parquet(out_path)
    return out_path, pairs


def _self_check() -> None:
    cfg = {"comparisons": [["a", "b"], ["b", "c"]],
           "sample_ids": {"a-vs-b": ["i1", "i2"], "b-vs-c": ["i2", "i3"]}}
    keys = subset_keys(cfg)
    assert keys == {("a", "i1"), ("a", "i2"), ("b", "i1"), ("b", "i2"),
                     ("b", "i3"), ("c", "i2"), ("c", "i3")}, keys

    covered = keys - {("b", "i2")}  # b/i2 excluded
    sids = filtered_sample_ids(cfg, covered)
    assert sids["a-vs-b"] == ["i1"], sids  # i2 dropped: b/i2 not covered
    assert sids["b-vs-c"] == ["i3"], sids  # i2 dropped on both pairs

    assert strip_role_prefix("[Assistant]: x") == "x"  # normalise_prefix reused, not duplicated

    assert rewrite_answer_letter("reasoning...\n\nANSWER: B") == "B"
    assert rewrite_answer_letter("reasoning...\n\nANSWER: C\n```") == "C"  # code-fenced, still fine
    assert rewrite_answer_letter("reasoning...\n\n**ANSWER: D**") == "D"  # bold, still fine
    assert rewrite_answer_letter("reasoning...\n\nFinal Answer: B") is None  # no literal "ANSWER:"
    assert rewrite_answer_letter("no answer line at all") is None

    print("self-check OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rewriter", choices=sorted(REWRITE_FILES))
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return 0
    if not args.rewriter:
        ap.error("--rewriter is required unless --self-check is given")
    rewriter = args.rewriter
    condition = f"paraphrase_{rewriter}"

    cfg = yaml.safe_load(SUBSET_CONFIG.read_text())
    assert cfg["include_reasoning"] is False, "subset must be final-text-only"

    subset = subset_keys(cfg)
    excluded = excluded_keys(EXCLUDED)
    covered = subset - excluded
    print(f"{len(subset)} subset cells - {len(excluded & subset)} excluded = {len(covered)} to build")

    rewrite_path = REWRITE_FILES[rewriter]
    if not rewrite_path.exists():
        raise SystemExit(f"{rewrite_path} not found - the {rewriter} rewrite set has not landed yet")
    rewrites = load_rewrites(rewrite_path)
    missing = covered - rewrites.keys()
    if missing:
        raise SystemExit(f"{len(missing)} covered cells missing from {rewrite_path.name} "
                          f"(e.g. {sorted(missing)[:3]}) - {rewriter} rewrite set is incomplete")

    by_model: dict[str, list[str]] = {}
    for model, item in covered:
        by_model.setdefault(model, []).append(item)

    out_dir = STORE_ROOT / condition
    n_built = 0
    all_bad = []
    for model, items in sorted(by_model.items()):
        built, bad = build_model_store(model, items, rewrites, out_dir)
        n_built += len(built)
        all_bad.extend(bad)
        print(f"  {model}: {len(built)} items -> {out_dir / f'{model}.eval'}")

    if all_bad:
        print(f"\n{len(all_bad)} cell(s) dropped: the rewrite's final answer didn't hold up "
              f"(the word-ratio QC does not check this):")
        for c in all_bad:
            print(f"  {c['model']}/{c['item']}: target={c['target']} got={c['got']} ({c['reason']})")
        covered -= {(c["model"], c["item"]) for c in all_bad}
    assert n_built == len(covered), f"built {n_built} sample-slots, expected {len(covered)}"

    models = sorted({m for m, _ in covered})
    sids = filtered_sample_ids(cfg, covered)
    cfg_path = write_config(cfg, condition, models, sids)
    print(f"wrote {cfg_path}")

    parquet_path, pairs = write_pairs_parquet(cfg, condition, models, sids, out_dir)
    print(f"wrote {parquet_path}: {pairs.pair_uid.nunique()} pairs, {len(pairs)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
