"""Import old judge logs (logs/{rank,pilot,pilot_reasoning,aime,apex}) into the JudgeBT layout.

    uv run python scripts/judge/import_judge_logs.py configs/domain/science.yml [--dry-run]

A log is imported when it is a successful, untruncated run on one of the domain's benches whose
two models are a (reference, target) pair of a frozen semester task. It is rewritten into
logs/judge/<domain>/<semester>/<judge>/<condition hash>/ with:
  - task name  bench/<reference>|<target> (imported)   -- the suffix keeps JudgeBT.judge from
    treating a partial old run as the finished task; its verdicts are pooled with the new run's
    and deduplicated per (item, pair, shown order), the new run winning.
  - metadata   major = reference, minor = target, semester, imported_from
  - samples    ids <item>_<position of the reference>; where the old major was the target the
    id suffix, target and score value flip so the convention holds.
The condition hash assumes the config-level default prompt (a bench's own prompt, e.g. mcq for
gpqa, is applied per bench by JudgeBT and is not part of the hash).
"""

import argparse
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.scorer import CORRECT, INCORRECT

from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from configs.model import ModelConfig
from reference import tables

AREAS = ["rank", "pilot", "pilot_reasoning", "aime", "apex"]
SUFFIX = " (imported)"


def semester_pairs(models: pd.DataFrame) -> dict[frozenset, tuple[str, str, str]]:
    """{frozenset(pair): (semester, reference, target)} over every task of the domain."""
    pairs = {}
    for semester in [c.removesuffix("_role") for c in models.columns if c.endswith("_role")]:
        role = models[f"{semester}_role"]
        for reference in role[role == "reference"].index:
            for target in role[role == "target"].index:
                pairs[frozenset((reference, target))] = (semester, reference, target)
    return pairs


def old_pair(md: dict) -> tuple[str, str] | None:
    """(major, minor) data-dir shortnames of an old log, from its metadata or its log paths."""
    if md.get("major") and md.get("minor"):
        return md["major"], md["minor"]
    paths = md.get("major_path", ""), md.get("minor_path", "")
    if all(str(p).startswith("data/") for p in paths):
        return tuple(Path(p).parts[3] for p in paths)  # data/<bench>/<hash>/<model>/...
    return None


def old_bench(md: dict) -> str | None:
    if md.get("bench"):
        return md["bench"]
    mp = str(md.get("major_path", ""))
    return Path(mp).parts[1] if mp.startswith("data/") else None


def flip(sample) -> None:
    """The old major was shown where the reference must be reported: swap the id suffix, target, score."""
    item, position = str(sample.id).rsplit("_", 1)
    new = {"1": "2", "2": "1"}[position]
    sample.id = f"{item}_{new}"
    sample.target = new
    for score in (sample.scores or {}).values():
        if score.answer in ("1", "2"):
            score.value = CORRECT if score.answer == new else INCORRECT


def import_log(path: Path, domain: DomainConfig, pairs: dict, benches: set[str], dry_run: bool) -> tuple[str, Path | None]:
    header = read_eval_log(str(path), header_only=True)
    md = header.eval.metadata or {}
    if header.status != "success":
        return "not successful", None
    if md.get("truncate_frac") is not None:
        return "truncated condition", None
    bench, pair = old_bench(md), old_pair(md)
    if bench not in benches or pair is None:
        return "not a bench of the domain", None
    if frozenset(pair) not in pairs:
        return "not a (reference, target) pair", None
    if md.get("max_length") is None or md.get("include_reverse") is None:
        return "condition not recorded", None
    semester, reference, target = pairs[frozenset(pair)]
    judge = ModelConfig(name=header.eval.model.split("/", 1)[-1], slug=header.eval.model)
    config = JudgeConfig(name="import", judge=judge, include_reasoning=bool(md.get("include_reasoning")),
                         include_reverse=bool(md["include_reverse"]), max_length=int(md["max_length"]))
    out_dir = config.log_dir / domain.name / semester / judge.shortname.lower() / config.condition_hash
    out = out_dir / path.name
    if out.exists():
        return "already imported", out
    if dry_run:
        return "would import", out

    log = read_eval_log(str(path))
    log.eval.task = f"{bench}/{reference}|{target}{SUFFIX}"
    log.eval.metadata = {**md, "domain": domain.name, "bench": bench, "major": reference, "minor": target,
                         "semester": semester, "imported_from": str(path)}
    if pair[0] != reference:  # the old major is the target: every sample's position labels refer to it
        for sample in log.samples or []:
            flip(sample)
        if log.eval.dataset and log.eval.dataset.sample_ids:
            log.eval.dataset.sample_ids = [s.id for s in log.samples or []]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, str(out))
    return "imported", out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    domain = DomainConfig.from_cli(args.domain)
    models, items = tables.load(domain)
    pairs, benches = semester_pairs(models), set(items.bench)

    outcomes = []
    for area in AREAS:
        for path in sorted((args.logs / area).rglob("*.eval")):
            outcome, out = import_log(path, domain, pairs, benches, args.dry_run)
            outcomes.append(dict(area=area, path=str(path), outcome=outcome, out=str(out) if out else ""))
    table = pd.DataFrame(outcomes)
    print(table.groupby(["area", "outcome"]).size().to_string())
    done = table[table.outcome.isin(["imported", "would import", "already imported"])]
    if len(done):
        print(done.assign(dest=done.out.map(lambda p: str(Path(p).parent))).groupby("dest").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
