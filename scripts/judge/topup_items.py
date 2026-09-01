"""Items each pair still needs judged: both-correct in the final rollout logs, not yet judged.

Reads the pair configs, the final rollout log per model (a transcripts map, model config
key -> log path), and every finished judge log under the given roots (any layout; the log's
metadata says which rollouts it judged). Writes one run_judging.py config per judge with
`only_sample_ids:` for the order-specific samples missing for that judge, and prints the counts.

Usage:
  uv run python scripts/judge/topup_items.py --bench gpqa --epoch 0 \
      --transcripts configs/judge_more_pairs/transcripts/gpqa-epoch0.yaml \
      --pairs configs/judge_more_pairs/pairs_all.txt --judges configs/judge_more_pairs/judges.yaml \
      --judged /Volumes/T7/data/hf-judging/pilot/gpqa <pulled earth logs>/more-pairs/gpqa ... \
      --log-root logs/more-pairs-topup/gpqa/epoch-0 --out-dir configs/judge_more_pairs_topup/gpqa
"""

import argparse
import collections
import re
from pathlib import Path

import yaml
from inspect_ai.log import read_eval_log, read_eval_log_samples

from judge.dataset import build_dataset
from judge.util import extract_score

PAIRS_DIR = Path("configs/experiments/pairs")


def correct_items(log_path: Path) -> set[str]:
    return {str(s.id) for s in read_eval_log_samples(str(log_path), all_samples_required=False) if extract_score(s)}


def lineage(log_path: Path, strict: bool = False) -> set[str]:
    """Names of the rollout logs a verdict may come from for this final log.

    Strict: the log itself, the log it was built from, and every other final built from
    that same log (an earlier finalisation round; re-runs only replace samples that were
    never both-correct, so no verdict is invalidated). Otherwise also every other full log
    still in the epoch dir: some dirs hold two published collection runs and the verdict
    tables reconcile each judge log against the rollout it names. Verdicts on a rollout
    that is gone from the dataset never count."""
    meta = read_eval_log(str(log_path), header_only=True).eval.metadata or {}
    source = meta.get("finalised_from")
    names = {log_path.name, source} - {None}
    for sibling in log_path.parent.glob("*.eval"):
        if sibling.name.startswith("._") or sibling == log_path:
            continue
        try:
            header = read_eval_log(str(sibling), header_only=True)
        except Exception:
            continue
        if header.eval.config.sample_id or header.eval.config.limit:
            continue
        sibling_source = (header.eval.metadata or {}).get("finalised_from")
        if (source and sibling_source == source) or (not strict and header.status == "success"):
            names.add(sibling.name)
    return names


def canonical_prompt(prompt: str) -> str:
    """Ignore blank-line drift immediately inside the prompt's --- wrappers."""
    return re.sub(r"\n+(?=---\n)", "\n", re.sub(r"(?<=---)\n+", "\n", prompt))


def matching_sample_ids(samples, expected: dict[str, str]) -> set[str]:
    """Order-specific sample ids whose logged prompt exactly matches the current prompt."""
    return {
        sample_id
        for sample in samples
        if (current := expected.get(sample_id := str(sample.id))) is not None
        and canonical_prompt(current) == canonical_prompt(sample.input)
    }


def judged_items(roots: list[Path], bench: str, epoch: str, dir_of: dict[str, str],
                 accepted: dict[str, set[str]] | None = None,
                 expected: dict[str, dict[str, str]] | None = None) -> dict[tuple[str, str], set[str]]:
    """(judge, pair) -> items with a finished verdict, from the judge logs under roots.
    With `accepted` (model key -> rollout log names) a judge log counts only if both its
    rollouts are in their model's set."""
    name_of = {d: key for key, d in dir_of.items()}
    out = collections.defaultdict(set)
    for root in roots:
        for path in root.rglob("*.eval"):
            if path.name.startswith("._"):
                continue
            try:
                header = read_eval_log(str(path), header_only=True)
            except Exception:
                continue
            if header.status != "success":
                continue
            meta = header.eval.metadata or {}
            major, minor = Path(meta.get("major_path", "")), Path(meta.get("minor_path", ""))
            if len(major.parts) < 5 or major.parts[-5] != bench or major.parts[-2] != epoch:
                continue
            pair_models = (name_of.get(major.parts[-3].lower()), name_of.get(minor.parts[-3].lower()))
            if None in pair_models:
                continue
            pair = f"{pair_models[0]}-vs-{pair_models[1]}"
            if expected is None and accepted and (
                major.name not in accepted[pair_models[0]] or minor.name not in accepted[pair_models[1]]
            ):
                continue  # judged from another rollout of the model
            judge = header.eval.model.split("/")[-1]
            if expected is None:
                samples = read_eval_log_samples(str(path), all_samples_required=False)
                out[(judge, pair)].update(str(s.id) for s in samples)
            else:
                samples = read_eval_log(str(path)).samples or []
                out[(judge, pair)].update(
                    matching_sample_ids(samples, expected.get(pair, {}))
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True, help="benchmark dir name in the rollout paths, e.g. gpqa, aime2024")
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--transcripts", type=Path, required=True, help="yaml: model config key -> final rollout log")
    parser.add_argument("--pairs", type=Path, required=True, help="file of pair names")
    parser.add_argument("--judges", type=Path, required=True, help="judges template (judges.yaml)")
    parser.add_argument("--judged", type=Path, nargs="+", required=True, help="roots holding finished judge logs")
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strict-lineage", nargs="*", default=[],
                        help="models whose verdicts must come from the final log's own lineage (an older "
                             "rollout of theirs in the dir was superseded by decision, e.g. kimi-k3 on GPQA)")
    parser.add_argument("--content-match", action="store_true",
                        help="accept a verdict only when its judge prompt exactly matches the current transcripts")
    parser.add_argument("--phase1-only", type=int, metavar="N",
                        help="only the first N both-correct items in id order (the runner's phase-1 slice), "
                             "for benchmarks whose published judging is phase 1 only (InterCode, SWE)")
    args = parser.parse_args()

    transcripts = {k: Path(v) for k, v in yaml.safe_load(args.transcripts.read_text()).items()}
    dir_of = {k: p.parts[-3].lower() for k, p in transcripts.items()}
    pairs = args.pairs.read_text().split()
    template = yaml.safe_load(args.judges.read_text())
    judges = template["judges"]

    correct = {k: correct_items(p) for k, p in transcripts.items()}
    expected = None
    if args.content_match:
        expected = {}
        for pair in pairs:
            major, minor = pair.split("-vs-")
            if major in transcripts and minor in transcripts:
                dataset = build_dataset(transcripts[major], transcripts[minor], include_reasoning=False)
                expected[pair] = {str(s.id): s.input for s in dataset}
    judged = judged_items(
        args.judged,
        args.bench,
        args.epoch,
        dir_of,
        None if expected is not None else {
            k: lineage(p, strict=k in args.strict_lineage) for k, p in transcripts.items()
        },
        expected,
    )

    needed = collections.defaultdict(dict)  # judge -> pair -> order-specific sample ids
    for pair in pairs:
        major, minor = pair.split("-vs-")
        if major not in correct or minor not in correct:
            print(f"SKIP {pair}: no final log for both models")
            continue
        both = sorted(correct[major] & correct[minor])
        if args.phase1_only:
            both = both[:args.phase1_only]  # same order as the judge task's dataset
        wanted = {f"{item}_{order}" for item in both for order in (1, 2)}
        for judge in judges:
            short = judge.split("/")[-1]
            missing = sorted(wanted - judged.get((short, pair), set()))
            if missing:
                needed[judge][pair] = missing

    out_dir = args.out_dir / f"epoch-{args.epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for judge in judges:
        items = needed.get(judge, {})
        n = sum(len(v) for v in items.values())
        total += n
        print(f"{judge.split('/')[-1]:26s} pairs={len(items):3d} items={n:5d}")
        if not items:
            continue
        used = sorted({m for pair in items for m in pair.split("-vs-")})
        cfg = {
            "log_root": args.log_root,
            "transcripts": {m: str(transcripts[m]) for m in used},
            "comparisons": [[judge, *pair.split("-vs-")] for pair in items],
            "only_sample_ids": items,
        }
        pins = template.get("judge_model_args", {})
        if judge in pins:
            cfg["judge_model_args"] = {judge: pins[judge]}
        if judge in template.get("judge_generate_args", {}):
            cfg["generate_args"] = template["judge_generate_args"][judge]
        if judge in template.get("judge_max_connections", {}):
            cfg["max_connections"] = template["judge_max_connections"][judge]
        path = out_dir / f"{judge.split('/')[-1]}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=200))
    print(f"total order-specific samples to judge: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
