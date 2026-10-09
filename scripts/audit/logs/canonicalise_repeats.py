"""Pick the canonical repeats per benchmark+model, and archive the rest.

A pair of repeats is only a control if both sides were produced the same way. This walks
every ``<bench>/<hash>/<model>/<repeat>/`` and classifies each log by what the judge would
actually be shown, then renumbers the survivors 0..n-1 and moves the rest out of the way:

  capped   -- samples cut at a max_tokens ceiling, so the chain of thought is missing its
              end (or all of it). Not recoverable by rescoring: archive to ``legacy/``.
  mixed    -- the log shows more than one reasoning shape (plain / redacted / none) across
              its samples, so the same model reads as thinking on some items and not others.
  none     -- uniformly no reasoning *while a sibling repeat of the same model+bench has it*.
              That is a dropped or differently-configured run, not the model's nature, so it
              moves to a ``<model>_none`` dir rather than being deleted.
  clean    -- one shape throughout, no cap. Kept.

Survivors are ordered by ``--prefer`` (default: logs already published to the dataset take
the lower repeat numbers, locally-collected ones fill in after), so a repeat number stays
pointing at the same run for anyone who already has the data.

    uv run python scripts/audit/logs/canonicalise_repeats.py                # dry run
    uv run python scripts/audit/logs/canonicalise_repeats.py --write
    uv run python scripts/audit/logs/canonicalise_repeats.py data/gpqa --write
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples

from configs.paths import collection_logs, is_successful_eval


def classify(path: Path) -> dict:
    """Reasoning shape, cap-truncation and config for one log."""
    path = Path(path)
    header = read_eval_log(str(path), header_only=True)
    generate = header.eval.model_generate_config
    config = generate.model_dump(exclude_none=True) if generate else {}
    shapes, truncated, n = defaultdict(int), 0, 0
    answers = []
    for sample in read_eval_log_samples(str(path), all_samples_required=False, resolve_attachments=True):
        n += 1
        # What the run actually generated, per sample. The bare completion will not do: on
        # aime it is often just "ANSWER: 204", which two genuine repeats of a strong model
        # produce identically. Full assistant text plus the output token count is the same
        # signal validate_data.py uses, and it only matches when the generation was replayed
        # (early collection's generate cache did that), not merely when the answers agree.
        text = [
            block if isinstance(block, str) else getattr(block, "text", None) or getattr(block, "reasoning", "")
            for message in sample.messages if message.role == "assistant"
            for block in (message.content if isinstance(message.content, list) else [message.content])
        ]
        usage = sample.output.usage
        answers.append((str(sample.id), "\n".join(text), usage.output_tokens if usage else None))
        blocks = [
            block
            for message in sample.messages if message.role == "assistant"
            for block in (message.content if isinstance(message.content, list) else [message.content])
            if getattr(block, "type", None) == "reasoning"
        ]
        plain = any(
            not getattr(b, "redacted", False) and (getattr(b, "reasoning", "") or "").strip()
            for b in blocks
        )
        shapes["plain" if plain else "redacted" if blocks else "none"] += 1
        try:
            truncated += str(sample.output.stop_reason) == "max_tokens"
        except Exception:  # a sample with no choices at all
            pass
    return {
        "path": str(path), "n": n, "shapes": dict(shapes), "truncated": truncated,
        "fingerprint": hashlib.sha256(repr(sorted(answers)).encode()).hexdigest()[:16],
        "edits": [e.get("tool") for e in (header.eval.metadata or {}).get("edits", [])],
        "max_tokens": config.get("max_tokens"), "extra_body": config.get("extra_body"),
        "created": header.eval.created[:19],
    }


def verdict(row: dict, siblings_have_reasoning: bool, cap_rate: float, mixed_rate: float) -> str:
    shapes = {k: v for k, v in row["shapes"].items() if v}
    # A long-CoT model running past the cap on a couple of hard items is ordinary; a ceiling
    # set too low for the model is not, and it truncates a large share of the run. Only the
    # latter is a broken run, so this is a rate, not a count.
    if row["n"] and row["truncated"] / row["n"] >= cap_rate:
        return "capped"
    # Only what the judge is *shown* can make a run inconsistent. `redacted` and `none` both
    # render as nothing (format_reasoning shows .reasoning only when the block is not
    # redacted), so a log that mixes those two is uniform to the judge and is not a defect.
    # The real split is plain vs not-plain.
    visible = shapes.get("plain", 0)
    hidden = shapes.get("redacted", 0) + shapes.get("none", 0)
    if visible and hidden:
        # Minority share, so a threshold of 0 keeps the strict reading (any mixing is a bad
        # run) while a small non-zero value tolerates a stray sample without discarding an
        # otherwise-uniform repeat -- and losing the model from the benchmark with it.
        minority = min(visible, hidden) / row["n"] if row["n"] else 0
        if minority > mixed_rate:
            return "mixed"
    if not visible and siblings_have_reasoning and not shapes.get("redacted"):
        return "none"
    return "clean"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--write", action="store_true", help="apply the moves (default: dry run)")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--cache", default="analysis/canonicalise_cache.json",
                        help="where to cache the per-log classification ('' to disable)")
    parser.add_argument("--cap-rate", type=float, default=0.2,
                        help="truncated share at which a run counts as capped (default 0.2)")
    parser.add_argument("--mixed-rate", type=float, default=0.0,
                        help="minority-shape share tolerated before a run counts as mixed "
                             "(default 0.0: any mixing archives the run)")
    parser.add_argument("--prefer", default="origin/main",
                        help="git ref whose logs take the lower repeat numbers ('' to order by date)")
    args = parser.parse_args()

    paths = [p for root in args.roots for p in collection_logs(root) if is_successful_eval(p)]
    # Classifying reads every sample of every log, which is minutes of work; the thresholds
    # are the part you actually want to try several values of, so the classification is
    # cached and only the verdicts are recomputed.
    cache = Path(args.cache) if args.cache else None
    rows = None
    if cache and cache.exists():
        cached = {r["path"]: r for r in json.loads(cache.read_text())}
        if set(cached) == {str(p) for p in paths}:
            rows = [cached[str(p)] for p in paths]
            print(f"{len(rows)} logs from {cache}", file=sys.stderr)
        else:
            print(f"{cache} is stale ({len(cached)} rows vs {len(paths)} logs); rescanning",
                  file=sys.stderr)
    if rows is None:
        print(f"{len(paths)} successful logs", file=sys.stderr)
        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(classify, paths))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows))

    published: set[str] = set()
    if args.prefer:
        listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", args.prefer],
                                 cwd="data", capture_output=True, text=True)
        published = {f"data/{line}" for line in listing.stdout.splitlines()}

    groups = defaultdict(list)
    for row in rows:
        parts = Path(row["path"]).parts
        groups[(parts[1], parts[2], parts[3])].append(row)

    # Every move is per *log*, not per repeat dir: a repeat dir can hold more than one
    # successful log (two collections resumed into the same dir), and those are distinct runs
    # that each need their own repeat number -- which is also what validate_data.py asks for.
    moves, kept, stats = [], 0, defaultdict(int)
    survivors = {}
    for (bench, hashdir, model), rows_in in sorted(groups.items()):
        has_reasoning = any(r["shapes"].get("plain") or r["shapes"].get("redacted") for r in rows_in)
        for row in rows_in:
            row["verdict"] = verdict(row, has_reasoning, args.cap_rate, args.mixed_rate)
        good = [r for r in rows_in if r["verdict"] == "clean"]
        # A repeat that reproduces another's answers exactly is not a second sample of the
        # model: keep the first and mark the rest, so the self-pair control stays meaningful.
        seen_fingerprints, unique = {}, []
        for row in sorted(good, key=lambda r: r["created"]):
            first = seen_fingerprints.setdefault(row["fingerprint"], row)
            if first is row:
                unique.append(row)
            else:
                row["verdict"] = "duplicate"
                row["duplicate_of"] = first["path"]
        good = unique
        base = Path("data") / bench / hashdir / model
        # Published logs take the low repeat numbers so anyone who already has the dataset
        # keeps their r0/r1; locally-collected ones fill in after. Within a group, a log keeps
        # the number it already has whenever that slot is free -- renumbering a repeat that is
        # already correct only churns the dataset and invalidates other people's references.
        ranked = sorted(good, key=lambda r: (r["path"] not in published, r["created"]))
        survivors[(bench, hashdir, model)] = ranked
        slots, deferred, taken = {}, [], set()
        for row in ranked:
            current = Path(row["path"]).parts[4]
            claim = int(current) if current.isdigit() else -1
            if 0 <= claim < len(ranked) and claim not in taken:
                slots[row["path"]], _ = claim, taken.add(claim)
            else:
                deferred.append(row)
        free = (n for n in range(len(ranked)) if n not in taken)
        for row in deferred:
            slots[row["path"]] = next(free)
        for row in ranked:
            stats["clean"] += 1
            kept += 1
            source, target = Path(row["path"]), slots[row["path"]]
            # a shared repeat dir still has to be split, even when the number is unchanged
            if source.parent != base / str(target) or len(list(source.parent.glob("*.eval"))) > 1:
                moves.append((row, base / str(target), f"r{source.parts[4]}->r{target}"))
        spare = defaultdict(int)
        for row in rows_in:
            if row["verdict"] == "clean":
                continue
            stats[row["verdict"]] += 1
            source = Path(row["path"])
            if row["verdict"] == "none":
                parent = Path("data") / bench / hashdir / f"{model}_none"
                existing = len([d for d in parent.iterdir() if d.name.isdigit()]) if parent.exists() else 0
                dest = parent / str(existing + spare[parent])
                spare[parent] += 1
            else:
                parent = Path("data") / "legacy" / bench
                dest = parent / f"{model}-r{source.parts[4]}-{row['verdict']}"
                while dest in {d for _, d, _ in moves} or dest.exists():
                    spare[parent] += 1
                    dest = parent / f"{model}-r{source.parts[4]}-{row['verdict']}-{spare[parent]}"
            moves.append((row, dest, row["verdict"]))

    for row, dest, why in moves:
        shapes = "+".join(f"{k}:{v}" for k, v in sorted(row["shapes"].items(), key=lambda kv: -kv[1]) if v)
        cap = f" cap={row['max_tokens']}" if row["max_tokens"] else ""
        trunc = f" truncated={row['truncated']}" if row["truncated"] else ""
        print(f"{why:28s} {row['path']}\n{'':28s}   n={row['n']} {shapes}{cap}{trunc}")

    # An audit fix (rescore, reasoning normalisation) lives inside the log it corrected. If a
    # stamped log is displaced by a freshly collected one, that provenance does not carry over
    # -- and the new log may still need the same fix. Name them so the fixers can be re-run.
    displaced = []
    for (bench, _, model), ranked in survivors.items():
        keeping = {r["path"] for r in ranked}
        gone = [r for r in groups[(bench, _, model)] if r["path"] not in keeping and r["edits"]]
        if gone and not any(r["edits"] for r in ranked):
            displaced += [(bench, model, r) for r in gone]
    if displaced:
        print(f"\n{len(displaced)} stamped log(s) displaced by unstamped ones -- re-run the fixer "
              f"over the kept log and let it stamp its own:", file=sys.stderr)
        for bench, model, row in displaced[:20]:
            print(f"    {bench}/{model}  lost {'+'.join(sorted(set(row['edits'])))}  {row['path']}",
                  file=sys.stderr)

    # Archiving is not free: a model left with fewer than two usable repeats drops out of the
    # self-pair control and, at zero, out of the study. Say which ones, loudly -- the fix is
    # re-collection, and that is a human decision, not this script's.
    short = sorted((len(v), k) for k, v in survivors.items() if len(v) < 2)
    if short:
        print(f"\n{len(short)} bench/model dirs left with <2 usable repeats -- re-collect or accept:",
              file=sys.stderr)
        for count, (bench, _, model) in short:
            print(f"    {count} repeat(s)  {bench}/{model}", file=sys.stderr)

    print(f"\n{kept} kept, {len(moves)} moves; " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())),
          file=sys.stderr)
    if not args.write:
        print("dry run; pass --write to apply", file=sys.stderr)
        return 0

    # Two phases via a staging dir: sources and destinations overlap (r3 -> r0 while r0 -> r1),
    # so every log is parked first and only then placed.
    stage = Path("data") / ".canonicalise-stage"
    stage.mkdir(parents=True, exist_ok=True)
    parked = []
    for index, (row, dest, _) in enumerate(moves):
        source = Path(row["path"])
        held = stage / str(index)
        held.mkdir()
        shutil.move(str(source), str(held / source.name))
        # sidecars describe the eval_set, not the log; carry them only when this log had the
        # repeat dir to itself, otherwise they would follow the wrong run.
        if not list(source.parent.glob("*.eval")):
            for extra in source.parent.iterdir():
                shutil.move(str(extra), str(held / extra.name))
        parked.append((held, dest, source.parent))
    for held, dest, _ in parked:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            for item in held.iterdir():
                shutil.move(str(item), str(dest / item.name))
            held.rmdir()
        else:
            shutil.move(str(held), str(dest))
    for _, _, old_parent in parked:  # drop repeat dirs nothing is left in
        if old_parent.exists() and not any(old_parent.iterdir()):
            old_parent.rmdir()
    stage.rmdir()
    print(f"applied {len(parked)} moves", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
