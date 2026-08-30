"""Write one resumable judging config for every frozen canonical-anchor pair."""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


def transcript_key(model: str, transcripts: dict) -> str:
    if model in transcripts:
        return model
    matches = [key for key in transcripts if key.replace("-", "") == model.replace("-", "")]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"no unique transcript key for {model}")


def write_configs(manifest: pd.DataFrame, base_configs: dict, output_root: Path,
                  *, extra_transcripts: dict | None = None) -> list[Path]:
    """Write one-pair configs with the item order frozen by ``manifest``."""
    extra_transcripts = extra_transcripts or {}
    base_configs = {
        (bench, judge): config | {
            "transcripts": config["transcripts"] | extra_transcripts.get(bench, {})
        }
        for (bench, judge), config in base_configs.items()
    }
    pairs, missing = [], Counter()
    for row in manifest.itertuples(index=False):
        transcripts = base_configs[(row.bench, row.judge)]["transcripts"]
        try:
            pairs.append(tuple(sorted((transcript_key(row.target, transcripts),
                                       transcript_key(row.opponent, transcripts)))))
        except ValueError:
            pairs.append(None)
            for model in (row.target, row.opponent):
                try:
                    transcript_key(model, transcripts)
                except ValueError:
                    missing[model] += 1
    if missing:
        detail = ", ".join(f"{model} ({count} rows)" for model, count in sorted(missing.items()))
        raise ValueError(f"unresolved manifest transcript sources: {detail}")

    manifest = manifest.assign(pair=pairs)
    desired = []
    for (bench, judge, pair), rows in manifest.groupby(["bench", "judge", "pair"], sort=True):
        base = base_configs[(bench, judge)]
        judges = {comparison[0] for comparison in base["comparisons"]}
        if len(judges) != 1:
            raise ValueError(f"{bench}/{judge}: expected one judge endpoint")
        items = rows["item"].tolist()
        if len(items) > 8 or len(items) != len(set(items)):
            raise ValueError(f"{bench}/{judge}/{pair}: expected one to eight unique items")
        path = output_root / bench / judge / f"{pair[0]}-vs-{pair[1]}.yaml"
        text = yaml.safe_dump({
            "log_root": f"logs/frontier-eci-canonical-anchors/{bench}",
            "system_prompt": base["system_prompt"],
            "include_reasoning": False,
            "only_items": items,
            "phase1_items": 8,
            "cache": False,
            "transcripts": base["transcripts"],
            "comparisons": [[judges.pop(), *pair]],
        }, sort_keys=False)
        desired.append((path, text))

    temporary = []
    try:
        for path, text in desired:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(text)
            temporary.append((path, temp))
        for path in output_root.rglob("*.yaml"):
            path.unlink()
        for path, temp in temporary:
            temp.replace(path)
    finally:
        for _, temp in temporary:
            temp.unlink(missing_ok=True)
    return [path for path, _ in desired]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    experiment = yaml.safe_load(args.config.read_text())
    manifest = pd.read_csv(Path(experiment["results"]) / "coverage" / "pair_items.csv")
    base_root = Path(experiment["base_experiment"]).parent / "runs"
    base_configs = {
        (path.parent.name, path.stem): yaml.safe_load(path.read_text())
        for path in base_root.glob("*/*.yaml")
    }
    outputs = write_configs(
        manifest, base_configs, args.config.parent / "runs",
        extra_transcripts=experiment.get("extra_transcripts"),
    )
    print(f"wrote {len(outputs)} one-pair configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
