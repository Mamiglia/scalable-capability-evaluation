"""Run the judge comparisons listed in a config file.

The config (see gen_judge_config.py) has these sections:
  log_root:          where judge logs go
  transcripts:       model shortname -> eval log path
  comparisons:       list of [judge, major, minor] rows; the judge is an inspect
                     slug whose model YAML under configs/models supplies the
                     endpoint pin, generate args and runpod settings
Optional knobs:
  system_prompt:     prompt file shown to the judge (default assets/system_prompt/default.txt)
  phase1_items:      items judged in the early phase (default 20)
  epochs:            judge every sample this many times, to measure how much
                     verdicts vary for a fixed input (default 1)
  truncate_fracs:    list of fractions; each adds a length-control run with both
                     transcripts cut to that fraction of the shorter one
                     (logs under trunc-<pct>/)
  include_reasoning: judges see candidates' reasoning (protocol default: false)
  min_correct:       drop comparisons where either model has fewer than this many
                     correct samples (default 0: keep all)

Each comparison runs in two phases: first the first phase1_items items of
every comparison (so every analysis has low-n data early), then the rest.
Phases with a successful log are skipped, so re-running only does new work.

Usage:
  uv run python scripts/judge/run_judging.py --config configs/judge_cheap6_gpqa.yaml --dry-run
  uv run python scripts/judge/run_judging.py --config configs/judge_cheap6_gpqa.yaml
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import yaml
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log

from configs import ModelConfig
from judge.task import DEFAULT_SYSTEM_PROMPT, judge_task
from model import Model

DEFAULT_PHASE1_ITEMS = 20
MODEL_CONFIGS = Path("configs/models")


def configured_system_prompt(cfg: dict) -> str:
    return cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)


def judge_config(slug: str) -> ModelConfig:
    """The model YAML under configs/models whose inspect slug is ``slug``."""
    found = []
    for path in sorted(MODEL_CONFIGS.rglob("*.yml")):
        config = ModelConfig.from_cli(yaml.safe_load(path.read_text()))
        if config.slug == slug:
            found.append((path, config))
    if len(found) != 1:
        paths = [str(path) for path, _ in found] or "none"
        raise ValueError(f"judge {slug}: need exactly one model YAML in {MODEL_CONFIGS}, found {paths}")
    return found[0][1]


def has_success_log(directory: Path) -> bool:
    """Whether this phase already produced a finished log, so it can be skipped.

    Phase logs always carry a sample_id block, so configs.paths.is_successful_eval
    (which rejects partial runs) would call every phase pending; the log status alone
    decides here."""
    for path in directory.glob("*.eval"):
        try:
            if read_eval_log(str(path), header_only=True).status == "success":
                return True
        except Exception:
            continue
    return False


def correct_count(path: str | Path) -> int:
    """Number of correct samples in an eval log, from the header's accuracy."""
    results = read_eval_log(str(path), header_only=True).results
    accuracy = results.scores[0].metrics["accuracy"].value
    return round(accuracy * results.completed_samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", type=int, choices=(1, 2),
                        help="run only this phase; default runs both")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["transcripts"]
    log_root = Path(cfg["log_root"])
    phase1_items = cfg.get("phase1_items") or DEFAULT_PHASE1_ITEMS
    epochs = cfg.get("epochs") or 1
    fracs = cfg.get("truncate_fracs") or [None]
    system_prompt = configured_system_prompt(cfg)
    # Ablation switch: judges see candidates' reasoning. Protocol default stays False.
    include_reasoning = bool(cfg.get("include_reasoning", False))
    if "judge_model_args" in cfg:
        raise ValueError("judge_model_args is no longer read: endpoint pins live in the model YAMLs")
    # Models with fewer correct samples than this don't qualify for judging.
    min_correct = cfg.get("min_correct") or 0
    qualified = set(paths)
    if min_correct:
        counts = {model: correct_count(path) for model, path in paths.items()}
        for model in sorted(paths):
            if counts[model] < min_correct:
                qualified.discard(model)
                print(f"DROP {model}: {counts[model]} correct < min_correct={min_correct}")

    tasks = {}

    def get_task(judge_short, major, minor, frac):
        key = (judge_short, major, minor, frac)
        if key not in tasks:
            suffix = f"_t{int(frac * 100)}" if frac else ""
            tasks[key] = judge_task(
                major_path=Path(paths[major]),
                minor_path=Path(paths[minor]),
                system_prompt=system_prompt,
                include_reasoning=include_reasoning,
                truncate_frac=frac,
                task_name=f"{judge_short}_{major}-vs-{minor}{suffix}",
                experiment_name=args.config.stem,
            )
        return tasks[key]

    # (phase, judge, major, minor, frac, sample_ids, log_dir), phase 1 rows first.
    pending = []
    for phase in ((args.phase,) if args.phase else (1, 2)):
        for frac in fracs:
            root = log_root if frac is None else log_root / f"trunc-{int(frac * 100)}"
            for judge, major, minor in cfg["comparisons"]:
                if major not in qualified or minor not in qualified:
                    continue
                judge_short = judge.split("/")[-1]
                pair_dir = root / judge_short / f"{major}-vs-{minor}"
                if frac is None and has_success_log(pair_dir):
                    continue  # fully judged before phasing was introduced
                try:
                    task = get_task(judge_short, major, minor, frac)
                except ValueError as e:
                    print(f"SKIP {judge_short} {major}-vs-{minor}: {e}")
                    continue
                if task is None:
                    # judge_task returns None when no question survives filtering
                    # (e.g. no both-correct overlap, or a transcript log with no samples).
                    print(f"SKIP {judge_short} {major}-vs-{minor}: no samples after filtering")
                    continue
                ids = [s.id for s in task.dataset]
                split = 2 * phase1_items
                block = ids[:split] if phase == 1 else ids[split:]
                if not block:
                    continue
                phase_dir = pair_dir / f"phase-{phase}"
                if has_success_log(phase_dir):
                    continue
                pending.append((phase, judge, judge_short, major, minor, frac, block, phase_dir))

    n_total = 2 * len(fracs) * len(cfg["comparisons"])
    print(f"{len(pending)} phase-runs pending (of {n_total} possible)")
    if args.dry_run:
        for phase, _, judge_short, major, minor, frac, block, _ in pending:
            t = f" t={frac}" if frac else ""
            print(f"  phase{phase}{t} {judge_short:28s} {major}-vs-{minor} ({len(block)//2} items)")
        return 0

    failures = []
    models = {}
    with ExitStack() as stack:  # a judge's pod, if any, stays up from first use to the end
        for phase, judge, judge_short, major, minor, frac, block, phase_dir in pending:
            if judge not in models:
                models[judge] = stack.enter_context(Model(judge_config(judge)))
            success, _ = eval_set(
                tasks=[get_task(judge_short, major, minor, frac)],
                log_dir=str(phase_dir),
                log_dir_allow_dirty=True,  # failed attempts under an older pin leave logs behind
                sample_id=block,
                epochs=epochs,
                max_connections=40,
                **models[judge].eval_kwargs(),
            )
            if not success:
                failures.append(f"phase{phase} {judge_short} on {major}-vs-{minor}")
    if failures:
        print(f"{len(failures)} FAILED runs (rest of queue completed):")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
