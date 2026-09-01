"""Re-collect the individual samples whose reasoning the provider failed to return.

Only for models that emit raw chain-of-thought. A closed model returning a summary (gemini,
the Claude family, gpt-5) has no CoT to recover, so re-running it changes nothing -- those
belong in ``normalize_reasoning.py --summary-models`` instead.

A sample qualifies when the log shows plain reasoning on other samples but none on this one:
the model does think here, the response just came back without it. Each such sample is
re-collected on its own via inspect's ``sample_id`` filter, written to ``<repeat>/reruns/``
(the layout used for targeted reruns), and folded back in by ``splice_reruns.py``, which
records the edit inside the log.

    uv run python scripts/audit/logs/rerun_missing_reasoning.py                    # plan only
    uv run python scripts/audit/logs/rerun_missing_reasoning.py --write            # collect
    uv run python scripts/audit/logs/rerun_missing_reasoning.py --write -b gpqa hle
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log_samples

from configs.paths import collection_logs, is_successful_eval

# Benchmarks explicitly out of scope for this study.
SKIP_BENCH = {"mbpp", "core_bench_easy", "core_bench_medium", "core_bench_hard",
              # imported matharena data: no bench config here, so nothing to re-collect with
              "otis_mock_aime_2024_2025",
              # out of scope: needs swebench<5 and a container sandbox per sample
              "swe_bench_mini"}
# Model-name fragments whose providers return real CoT. A closed model's "reasoning" is a
# summary, so a missing one is not a dropped trace and re-running cannot recover it.
RAW_COT = ("deepseek", "qwen", "glm", "kimi", "nemotron", "gemma", "muse", "ling",
           "granite", "llama", "mistral", "gpt-oss", "qed")

BENCH_CONFIG = {"aime2024": "aime24", "aime2025": "aime25", "aime2026": "aime26",
                "swe_bench_mini": "swe-bench-mini", "intercode": "intercode",
                "gpqa": "gpqa", "hle": "hle", "apex_2025": "apex",
                "apex_shortlist": "apex_shortlist"}


def dropped_samples(path: Path, gold: set | None) -> list[str]:
    """Sample ids with no reasoning in a log that otherwise has raw CoT.

    ``gold`` restricts HLE to the HLE-Verified gold subset (the audited, unmodified items),
    which is what the analyses actually judge on -- re-collecting outside it buys nothing.
    """
    plain, missing = 0, []
    for sample in read_eval_log_samples(str(path), all_samples_required=False, resolve_attachments=True):
        if gold is not None and (sample.metadata or {}).get("uid") not in gold:
            continue
        blocks = [
            b for m in sample.messages if m.role == "assistant"
            for b in (m.content if isinstance(m.content, list) else [m.content])
            if getattr(b, "type", None) == "reasoning"
        ]
        if any(not getattr(b, "redacted", False) and (getattr(b, "reasoning", "") or "").strip()
               for b in blocks):
            plain += 1
        elif not blocks:
            missing.append(str(sample.id))
    return missing if plain and missing else []


def model_config(model_dir: str) -> Path | None:
    """The config whose shortname matches this data dir."""
    for candidate in Path("configs/models").rglob("*.yml"):
        text = candidate.read_text()
        for line in text.splitlines():
            if line.startswith("name:"):
                if line.split(":", 1)[1].strip().split("/")[-1].lower() == model_dir:
                    return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("-b", "--bench", nargs="*", help="limit to these benchmark dirs")
    parser.add_argument("--write", action="store_true", help="actually collect (costs money)")
    parser.add_argument("--plan", default="analysis/rerun_plan.json")
    parser.add_argument("--all-hle", action="store_true",
                        help="do not restrict HLE to the verified-gold subset")
    args = parser.parse_args()

    gold_uids = None
    if not args.all_hle:
        from inspect_evals.hle.hle import get_verified_gold_uids
        gold_uids = set(get_verified_gold_uids())
        print(f"HLE restricted to {len(gold_uids)} verified-gold items", file=sys.stderr)

    plan, skipped = [], []
    for path in (p for root in args.roots for p in collection_logs(root) if is_successful_eval(p)):
        bench, model = path.parts[1], path.parts[3]
        if bench in SKIP_BENCH or (args.bench and bench not in args.bench):
            continue
        if not any(k in model.lower() for k in RAW_COT):
            continue
        gold = gold_uids if bench == "hle" and not args.all_hle else None
        ids = dropped_samples(path, gold)
        if not ids:
            continue
        # Decide collectability now, not halfway through spending: an imported model has no
        # config here, so it cannot be re-collected at all.
        if not BENCH_CONFIG.get(bench) or not model_config(model):
            skipped.append((bench, model, len(ids)))
            continue
        plan.append({"bench": bench, "model": model, "repeat": path.parts[4],
                     "log": str(path), "ids": ids})

    total = sum(len(p["ids"]) for p in plan)
    by_model = defaultdict(int)
    for entry in plan:
        by_model[(entry["bench"], entry["model"])] += len(entry["ids"])
    print(f"{len(plan)} log(s), {total} sample(s) to re-collect")
    for (b, m), n in sorted(by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {b}/{m}")
    Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plan).write_text(json.dumps(plan, indent=1))
    if skipped:
        agg = defaultdict(int)
        for bench, model, n in skipped:
            agg[(bench, model)] += n
        print(f"\nnot collectable (no bench or model config here): "
              f"{sum(agg.values())} sample(s)", file=sys.stderr)
        for (b, m), n in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {b}/{m}", file=sys.stderr)
    print(f"\nplan written to {args.plan}", file=sys.stderr)
    if not args.write:
        print("planning only; pass --write to collect", file=sys.stderr)
        return 0

    staging = Path("analysis/rerun_staging")
    for entry in plan:
        bench_cfg = BENCH_CONFIG.get(entry["bench"])
        model_cfg = model_config(entry["model"])
        if not bench_cfg or not model_cfg:
            print(f"SKIP {entry['bench']}/{entry['model']}: "
                  f"{'no bench config' if not bench_cfg else 'no model config'}", file=sys.stderr)
            continue
        out = staging / entry["bench"] / entry["model"] / entry["repeat"]
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        command = ["uv", "run", "collect", "-b", f"configs/bench/{bench_cfg}.yml",
                   "-m", str(model_cfg), "-n", "1", "--jobs", "1", "--log_dir", str(out)]
        for sample_id in entry["ids"]:
            command += ["--sample_id", sample_id]
        print(f"\n=== {entry['bench']}/{entry['model']} r{entry['repeat']}: "
              f"{len(entry['ids'])} sample(s)", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"FAILED {entry['bench']}/{entry['model']} r{entry['repeat']}", file=sys.stderr)
            continue
        produced = sorted(out.rglob("*.eval"))
        if not produced:
            print(f"NO OUTPUT {entry['bench']}/{entry['model']} r{entry['repeat']}", file=sys.stderr)
            continue
        reruns = Path(entry["log"]).parent / "reruns"
        reruns.mkdir(exist_ok=True)
        for produced_log in produced:
            shutil.copy2(produced_log, reruns / produced_log.name)
        print(f"    -> {reruns} ({len(produced)} log(s))")

    print("\nnow fold them in:  uv run python scripts/audit/logs/splice_reruns.py --write",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
