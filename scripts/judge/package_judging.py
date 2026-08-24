"""Package phased judge logs into the flat layout of the LASR-G5/judging dataset.

Source: <log_root>/epoch-<e>/<judge>/<major>-vs-<minor>/phase-{1,2}/*.eval (latest
successful log per phase). Output: <out>/pilot/<bench>/<judge>/<pair>/
<ts>_<major>|<minor>-<e>_<eval_id>.eval with the two phases' samples concatenated.

With <rejudge_root> (same layout, from rejudge_unanswered.py) a re-judged sample that
parsed replaces the original unanswered one.

Usage: uv run python scripts/judge/package_judging.py <log_root> <bench> <out_dir> [<rejudge_root>]
"""

import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log


def latest_success(phase_dir: Path) -> Path | None:
    best, best_created = None, None
    for path in phase_dir.glob("*.eval") if phase_dir.is_dir() else []:
        header = read_eval_log(str(path), header_only=True)
        if header.status == "success" and (best_created is None or header.eval.created > best_created):
            best, best_created = path, header.eval.created
    return best


def main() -> int:
    log_root, bench, out_root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    rejudge_root = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    packaged, n_samples, overlaid, skipped = 0, 0, 0, []
    for epoch_dir in sorted(log_root.glob("epoch-*")):
        epoch = epoch_dir.name.split("-")[-1]
        for judge_dir in sorted(p for p in epoch_dir.iterdir() if p.is_dir()):
            for pair_dir in sorted(p for p in judge_dir.iterdir() if p.is_dir() and "-vs-" in p.name):
                major, minor = pair_dir.name.split("-vs-", 1)
                phase1, phase2 = latest_success(pair_dir / "phase-1"), latest_success(pair_dir / "phase-2")
                if phase1 is None:
                    skipped.append(f"{judge_dir.name}/{pair_dir.name}/epoch-{epoch}: no finished phase-1 log")
                    continue
                log = read_eval_log(str(phase1))
                samples = list(log.samples or [])
                if phase2 is not None:  # only pairs with more than phase1_items items have one
                    samples += list(read_eval_log(str(phase2)).samples or [])
                ids = [(s.id, s.epoch) for s in samples]
                assert len(ids) == len(set(ids)), f"duplicate samples in {pair_dir}"
                if rejudge_root is not None:
                    redone = {}
                    for phase in ("phase-1", "phase-2"):
                        path = latest_success(rejudge_root / epoch_dir.name / judge_dir.name / pair_dir.name / phase)
                        for s in (read_eval_log(str(path)).samples or []) if path else []:
                            if any(sc.answer is not None for sc in (s.scores or {}).values()):
                                redone[(s.id, s.epoch)] = s
                    for i, s in enumerate(samples):
                        if (s.id, s.epoch) in redone and all(sc.answer is None for sc in (s.scores or {}).values()):
                            samples[i] = redone[(s.id, s.epoch)]
                            overlaid += 1
                log.samples = samples
                log.eval.config.sample_id = None
                if log.results is not None:
                    log.results.completed_samples = log.results.total_samples = len(samples)
                out_dir = out_root / "pilot" / bench / judge_dir.name / pair_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                stamp = log.eval.created.replace(":", "-").replace("+", "-")
                out = out_dir / f"{stamp}_{major}|{minor}-{epoch}_{log.eval.eval_id}.eval"
                write_eval_log(log, str(out))
                assert len(read_eval_log(str(out)).samples) == len(samples), f"readback mismatch for {out}"
                packaged += 1
                n_samples += len(samples)
    print(f"packaged {packaged} logs, {n_samples} samples, {overlaid} re-judged verdicts overlaid; {len(skipped)} skipped")
    for line in skipped:
        print("  SKIP", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
