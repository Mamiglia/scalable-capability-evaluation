"""Package judge logs into the flat layout of the LASR-G5/judging dataset.

One file per (judge, pair, epoch): <out>/<subset>/<bench>/<judge>/<pair>/
<stamp>_<major>|<minor>-<e>_<eval_id>.eval, subset 'aime' for AIME years, else 'pilot'.

The base log comes from the first run root that has a finished phase-1 log for the pair
(roots are given in priority order; phase-2 samples are appended), else from the published
copy for that epoch under --published. Samples from finished top-up logs
(<topup>/epoch-<e>/<judge>/<pair>/topup/) are appended; a re-judged sample under a
--rejudge root (rejudge_unanswered.py layout) that parsed replaces the unanswered original.
Every packaged file gets a fresh stamp so it sits next to the published copy, never over it.

Usage:
  uv run python scripts/judge/package_judging.py --bench gpqa --out <dir> \
      --roots <rejudge run root> <main root> <extra root> --published /Volumes/T7/data/hf-judging/pilot/gpqa \
      --topup <topup root> --rejudge <rejudge-empty root>...
"""

import argparse
import datetime as dt
from pathlib import Path

import yaml
from inspect_ai import score as inspect_score
from inspect_ai.log import EvalLog, EvalSample, read_eval_log, write_eval_log

from judge.dataset import build_dataset
from judge.scorer import judge_scorer
from topup_items import canonical_prompt


def success_paths(directory: Path) -> list[Path]:
    """Successful logs in chronological order."""
    finished = []
    for path in directory.glob("*.eval") if directory.is_dir() else []:
        if path.name.startswith("._"):
            continue
        try:
            header = read_eval_log(str(path), header_only=True)
        except Exception:
            continue
        if header.status == "success":
            finished.append((header.eval.created, path.name, path))
    return [path for _, _, path in sorted(finished)]


def latest_success_path(paths: list[Path]) -> Path | None:
    """Latest successful path by header creation time, then filename."""
    best, best_created = None, None
    for path in paths:
        try:
            header = read_eval_log(str(path), header_only=True)
        except Exception:
            continue
        created = (header.eval.created, path.name)
        if header.status == "success" and (best_created is None or created > best_created):
            best, best_created = path, created
    return best


def latest_success(directory: Path) -> Path | None:
    paths = success_paths(directory)
    return paths[-1] if paths else None


def published_epoch(path: Path) -> str:
    """'<stamp>_<major>|<minor>-<e>_<id>.eval' -> e."""
    epoch = path.stem.split("_", 1)[1].rsplit("_", 1)[0].rsplit("-", 1)[1]
    if epoch in {"0", "1"}:
        return epoch
    major = Path((read_eval_log(str(path), header_only=True).eval.metadata or {}).get("major_path", ""))
    return major.parts[-2] if len(major.parts) >= 2 else epoch


def published_pair(path: Path) -> str:
    pair = path.parts[-2]
    if "-vs-" in pair:
        return pair
    return read_eval_log(str(path), header_only=True).eval.metadata["experiment_name"]


def published_key(path: Path, canonical_judges: dict[str, str]) -> tuple[str, str, str]:
    judge = canonical_judges.get(path.parts[-3].casefold(), path.parts[-3])
    return judge, published_pair(path), published_epoch(path)


def canonical_key(
    key: tuple[str, str, str], canonical_judges: dict[str, str]
) -> tuple[str, str, str]:
    judge, pair, epoch = key
    return canonical_judges.get(judge.casefold(), judge), pair, epoch


def answered(sample: EvalSample) -> bool:
    return any(score.answer is not None for score in (sample.scores or {}).values())


def overlay_answered(samples: list[EvalSample], paths: list[Path]) -> list[EvalSample]:
    """Overlay every answered retry chronologically, only where the base was unanswered."""
    position = {(sample.id, sample.epoch): i for i, sample in enumerate(samples)}
    replaceable = {
        key for key, index in position.items() if not answered(samples[index])
    }
    for path in paths:
        for sample in read_eval_log(str(path)).samples or []:
            key = (sample.id, sample.epoch)
            if key in replaceable and answered(sample):
                samples[position[key]] = sample
    return samples


def rescore_packaged(log: EvalLog) -> EvalLog:
    """Recompute sample verdicts, metrics, and reductions after assembly."""
    return inspect_score(
        log,
        judge_scorer(),
        model="mockllm/model",
        action="overwrite",
        display="none",
        copy=False,
    )


def topup_logs(roots: list[Path]) -> dict[tuple[str, str, str], list[Path]]:
    out: dict[tuple[str, str, str], list[Path]] = {}
    for root in roots:
        for topup_dir in root.glob("epoch-*/*/*/topup"):
            key = (topup_dir.parts[-3], topup_dir.parts[-2], topup_dir.parts[-4].split("-")[-1])
            out.setdefault(key, []).extend(sorted(
                path for path in topup_dir.glob("*.eval")
                if not path.name.startswith("._")
                and read_eval_log(str(path), header_only=True).status == "success"
            ))
    return out


def selected_keys(keys, pairs: set[str] | None, epochs: set[str]) -> list[tuple[str, str, str]]:
    return sorted(key for key in keys if key[2] in epochs and (pairs is None or key[1] in pairs))


def current_samples(samples: list[EvalSample], expected: dict[str, str]) -> list[EvalSample]:
    kept = [sample for sample in samples if str(sample.id) in expected]
    assert {str(sample.id) for sample in kept} == set(expected), "current judge samples are missing"
    assert all(
        canonical_prompt(sample.input) == canonical_prompt(expected[str(sample.id)]) for sample in kept
    ), "packaged judge prompt does not match the current transcripts"
    return kept


def current_prompts(transcript_maps: list[Path], pairs: set[str], phase1_only: int | None):
    out: dict[tuple[str, str], dict[str, str]] = {}
    for epoch, path in enumerate(transcript_maps):
        transcripts = {key: Path(value) for key, value in yaml.safe_load(path.read_text()).items()}
        for pair in pairs:
            major, minor = pair.split("-vs-")
            if major not in transcripts or minor not in transcripts:
                continue
            samples = list(build_dataset(transcripts[major], transcripts[minor], include_reasoning=False))
            if phase1_only:
                samples = samples[:2 * phase1_only]
            out[(pair, str(epoch))] = {str(sample.id): sample.input for sample in samples}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="*", default=[], help="run log roots, highest priority first")
    parser.add_argument("--published", type=Path, help="published logs for this benchmark: <judge>/<pair>/<file>")
    parser.add_argument("--topup", type=Path, nargs="*", default=[], help="top-up run roots, oldest first")
    parser.add_argument("--rejudge", type=Path, nargs="*", default=[], help="rejudge_unanswered.py roots")
    parser.add_argument("--pairs-file", type=Path, help="keep only pairs listed in this whitespace-delimited file")
    parser.add_argument("--epochs", nargs="+", default=["0", "1"], help="judge epochs to package")
    parser.add_argument("--transcripts", type=Path, nargs=2, metavar=("EPOCH0", "EPOCH1"),
                        help="current transcript maps; prune and verify samples against them")
    parser.add_argument("--phase1-only", type=int,
                        help="with --transcripts, keep only the first N current both-correct items")
    parser.add_argument("--skip-judges", nargs="*", default=[], help="judge dirs to leave out")
    args = parser.parse_args()

    # (judge, pair, epoch) -> phase dirs of the winning run root
    runs: dict[tuple[str, str, str], Path] = {}
    for root in args.roots:
        for phase1 in root.glob("epoch-*/*/*/phase-1"):
            key = (phase1.parts[-3], phase1.parts[-2], phase1.parts[-4].split("-")[-1])
            if key not in runs and latest_success(phase1):
                runs[key] = phase1.parent
    topups = topup_logs(args.topup)
    canonical_judges = {judge.casefold(): judge for judge, _, _ in [*runs, *topups]}
    published_paths = []
    if args.published:
        published_paths = [
            path for path in args.published.glob("*/*/*.eval") if not path.name.startswith("._")
        ]
        # Match the live repository's existing directory case. Otherwise case variants in
        # run and top-up roots become duplicate keys on a case-insensitive filesystem.
        for path in published_paths:
            canonical_judges[path.parts[-3].casefold()] = path.parts[-3]
    runs = {canonical_key(key, canonical_judges): value for key, value in runs.items()}
    canonical_topups: dict[tuple[str, str, str], list[Path]] = {}
    for key, paths in topups.items():
        canonical_topups.setdefault(canonical_key(key, canonical_judges), []).extend(paths)
    topups = canonical_topups
    published_candidates: dict[tuple[str, str, str], list[Path]] = {}
    for path in published_paths:
        try:
            key = published_key(path, canonical_judges)
        except Exception:
            continue
        published_candidates.setdefault(key, []).append(path)
    published = {
        key: latest
        for key, paths in published_candidates.items()
        if (latest := latest_success_path(paths)) is not None
    }
    pairs = set(args.pairs_file.read_text().split()) if args.pairs_file else None
    keys = selected_keys(set(runs) | set(published) | {k for k, v in topups.items() if v}, pairs, set(args.epochs))
    if args.transcripts and pairs is None:
        parser.error("--transcripts requires --pairs-file")
    expected = current_prompts(args.transcripts, pairs, args.phase1_only) if args.transcripts else None

    subset = "aime" if args.bench.startswith("aime") else "pilot"
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S+00-00")
    counts = {"from runs": 0, "from published": 0, "topup samples": 0, "overlaid": 0}
    for judge, pair, epoch in keys:
        if judge in args.skip_judges:
            continue
        current = expected.get((pair, epoch), {}) if expected is not None else None
        if current == {}:
            continue  # no current both-correct items for this pair/epoch
        major, minor = pair.split("-vs-", 1)
        source = "runs" if (judge, pair, epoch) in runs else "published"
        if source == "runs":
            pair_dir = runs[(judge, pair, epoch)]
            log = read_eval_log(str(latest_success(pair_dir / "phase-1")))
            samples = list(log.samples or [])
            phase2 = latest_success(pair_dir / "phase-2")
            if phase2:
                samples += list(read_eval_log(str(phase2)).samples or [])
        elif (judge, pair, epoch) in published:
            log = read_eval_log(str(published[(judge, pair, epoch)]))
            samples = list(log.samples or [])
        else:  # a pair judged only in the top-up (never judged before by this judge)
            source = "topup only"
            log = read_eval_log(str(topups[(judge, pair, epoch)][0]))
            samples = []
        counts[f"from {source}"] = counts.get(f"from {source}", 0) + 1
        position = {(s.id, s.epoch): i for i, s in enumerate(samples)}

        for path in topups.get((judge, pair, epoch), []):
            for s in read_eval_log(str(path)).samples or []:
                key = (s.id, s.epoch)
                if key in position:
                    samples[position[key]] = s  # a later top-up of the same item wins
                else:
                    position[key] = len(samples)
                    samples.append(s)
                counts["topup samples"] += 1
        retry_paths = []
        for root in args.rejudge:
            for phase in ("phase-1", "phase-2", "topup"):
                retry_paths.extend(
                    success_paths(root / f"epoch-{epoch}" / judge / pair / phase)
                )
        retry_paths.sort(
            key=lambda path: (
                read_eval_log(str(path), header_only=True).eval.created,
                path.name,
            )
        )
        before = {(sample.id, sample.epoch): sample for sample in samples}
        samples = overlay_answered(samples, retry_paths)
        counts["overlaid"] += sum(
            before.get((sample.id, sample.epoch)) is not sample for sample in samples
        )

        if current is not None:
            try:
                samples = current_samples(samples, current)
            except AssertionError as error:
                raise AssertionError(f"{judge}/{pair}/epoch-{epoch}: {error}") from error

        ids = [(s.id, s.epoch) for s in samples]
        assert len(ids) == len(set(ids)), f"duplicate samples in {judge}/{pair}/{epoch}"
        log.samples = samples
        log.eval.config.sample_id = None
        log.status, log.error = "success", None
        log.eval.metadata = {**(log.eval.metadata or {}), "packaged_at": now.isoformat(), "packaged_from": source}
        log = rescore_packaged(log)
        out_dir = args.out / subset / args.bench / judge / pair
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{stamp}_{major}|{minor}-{epoch}_{log.eval.eval_id}.eval"
        if out.exists():
            raise RuntimeError(f"refusing to overwrite {out}")
        write_eval_log(log, str(out))
        assert len(read_eval_log(str(out)).samples) == len(samples), f"readback mismatch for {out}"
    print(f"{args.bench}: {len(keys)} (judge, pair, epoch) files; " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
