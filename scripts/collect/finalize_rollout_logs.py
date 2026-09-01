"""Build the final rollout log for every model-epoch dir of a collection.

Final = original canonical log, with the re-run samples spliced in (partial logs written by
`collect --sample-id`, recognised by config.sample_id) and, for gpqa/aime, answers re-scored
under --malformed-answers accept. The result is written NEXT to the original with a fresh
creation stamp and name, so the verdict tables' canonical rule (later log with the same
sample set wins) picks it; the original is never touched. Partial logs move into a reruns/
subdir so they are not mistaken for canonical logs. Dirs that already hold a finalised log
are skipped.

Usage:
  uv run python scripts/collect/finalize_rollout_logs.py data/gpqa/b0a0d9c9 --bench gpqa
  uv run python scripts/collect/finalize_rollout_logs.py data/aime2024/44136fa3 --bench aime
  uv run python scripts/collect/finalize_rollout_logs.py data/intercode/463eb861 --bench intercode
"""

import argparse
import datetime as dt
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from inspect_ai.log import read_eval_log, read_eval_log_samples, write_eval_log

RESCORER = {"gpqa": "scripts/judge/rescore_mcq.py", "aime": "scripts/judge/rescore_aime.py"}


def log_kind(path: Path) -> str | None:
    """Classify a log as a targeted rerun, limited smoke run, final, or full run."""
    try:
        header = read_eval_log(str(path), header_only=True)
    except Exception:
        return None
    if header.eval.config.sample_id:
        return "rerun"
    if header.eval.config.limit:
        return "limited"
    if (header.eval.metadata or {}).get("finalised_from"):
        return "final"
    return "full"


def canonical_log(full_logs: list[Path]) -> Path:
    """The single-epoch full log with the most samples, then the latest created. A run stopped
    after its last sample keeps status 'started' with every sample present (and no results
    block), so status alone cannot decide; multi-epoch logs are earlier collections the
    judging never used."""
    headers = [(read_eval_log(str(p), header_only=True), p) for p in full_logs]
    headers = [(h, p) for h, p in headers if h.eval.config.epochs in (None, 1)] or headers
    done = [(h, p) for h, p in headers if h.status == "success"] or headers
    def size(h, p):
        if h.results and h.results.completed_samples:
            return h.results.completed_samples
        return sum(1 for _ in read_eval_log_samples(str(p), all_samples_required=False))
    sizes = {p: size(h, p) for h, p in done}
    biggest = max(sizes.values())
    return max((hp for hp in done if sizes[hp[1]] == biggest), key=lambda hp: (hp[0].eval.created, hp[1].name))[1]


def run(cmd: list[str], out: Path, allow_one: bool = False) -> str:
    """Run a step that must write `out`; optionally accept exit 1 from the splicer."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if (result.returncode != 0 and not (allow_one and result.returncode == 1)) or not out.exists():
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()


def pinned_logs(pin_files: list[Path]) -> dict[tuple[str, str], str]:
    """(model dir, epoch dir) -> log name the judges saw, from run_judging configs
    (their `transcripts:` block) or plain model -> log maps."""
    pins = {}
    for path in pin_files:
        loaded = yaml.safe_load(path.read_text())
        for log in (loaded.get("transcripts") or loaded).values():
            pins[(Path(log).parts[-3], Path(log).parts[-2])] = Path(log).name
    return pins


def finalise(epoch_dir: Path, bench: str, pinned: str | None = None) -> str:
    logs = [p for p in sorted(epoch_dir.glob("*.eval")) if not p.name.startswith("._")]
    kinds = {p: log_kind(p) for p in logs}
    full = [p for p, k in kinds.items() if k == "full"]
    partial = [p for p, k in kinds.items() if k == "rerun"]
    if any(k == "final" for k in kinds.values()):
        if not partial:
            return "already finalised"
        # a later re-run round: rebuild from the source with every partial, old and new; the
        # earlier final stays (judge logs name it) and the newer one is picked as canonical
        partial = sorted(p for p in (epoch_dir / "reruns").glob("*.eval") if not p.name.startswith("._")) + partial
    if not full:
        return "no full log"
    if not partial and bench not in RESCORER:
        return "nothing to do"  # no re-runs and no rescoring: the canonical log stands
    source = next((p for p in full if p.name == pinned), None) or canonical_log(full)
    notes = [f"from {'pinned' if source.name == pinned else 'unpinned'} {source.name[:19]}"]
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage.eval"
        if partial:
            notes.append(run(
                [sys.executable, "scripts/collect/splice_rerun.py", str(source),
                 *map(str, partial), "--out", str(stage)],
                stage,
                allow_one=True,
            ))
        else:
            shutil.copy2(source, stage)
        if bench in RESCORER:
            rescored = Path(tmp) / "rescored.eval"
            notes.append(run([sys.executable, RESCORER[bench], str(stage), "--out", str(rescored), "--malformed-answers", "accept"], rescored))
            stage = rescored
        log = read_eval_log(str(stage))
        now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        log.eval.created = now.isoformat()
        log.eval.metadata = {**(log.eval.metadata or {}), "finalised_from": source.name,
                             "rerun_logs": [p.name for p in partial], "finalised_at": now.isoformat()}
        stamp = now.strftime("%Y-%m-%dT%H-%M-%S+00-00")
        out = epoch_dir / f"{stamp}_{log.eval.task.split('/')[-1]}_{log.eval.eval_id}.eval"
        if out.exists():
            raise RuntimeError(f"refusing to overwrite {out}")
        write_eval_log(log, str(out))
        assert len(read_eval_log(str(out)).samples) == len(log.samples), f"readback mismatch for {out}"
    if partial:
        reruns = epoch_dir / "reruns"
        reruns.mkdir(exist_ok=True)
        for p in partial:
            if p.parent != reruns:
                shutil.move(str(p), reruns / p.name)
    return f"-> {out.name}: " + " | ".join(n for n in notes if n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collection", type=Path, help="data/<bench>/<collection> dir")
    parser.add_argument("--bench", required=True, choices=["gpqa", "aime", "intercode", "swe_bench_mini"])
    parser.add_argument("--only", nargs="*", help="model dir names to process (default: all)")
    parser.add_argument("--pins", type=Path, nargs="*", default=[],
                        help="run_judging configs (or model -> log yamls): the log a model was judged from is the source")
    parser.add_argument("--unpinned", nargs="*", default=[], help="model dirs whose pin is ignored (a cleaner re-collection replaces it)")
    args = parser.parse_args()
    pins = pinned_logs(args.pins)
    for model_dir in sorted(p for p in args.collection.iterdir() if p.is_dir() and not p.name.endswith("_none")):
        if args.only and model_dir.name not in args.only:
            continue
        for epoch_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()):
            pin = None if model_dir.name in args.unpinned else pins.get((model_dir.name, epoch_dir.name))
            print(f"{model_dir.name}/{epoch_dir.name}: {finalise(epoch_dir, args.bench, pin)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
