"""Prepare matched CoT-on configs and audit the existing non-reasoning graph."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log_samples


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from judge.dataset import read_transcripts  # noqa: E402
from judge.util import extract_transcript  # noqa: E402
from scripts.frontier_eci.canonical_analysis import cap_pair_items  # noqa: E402
from scripts.frontier_eci.canonical_coverage import (  # noqa: E402
    _components, _connected, stable_items,
)
from scripts.frontier_eci.rank_regression import load_rank_analysis_inputs  # noqa: E402
from scripts.frontier_eci.write_topup_configs import (  # noqa: E402
    matched_pair_items, transcript_key, write_configs,
)


def components(block: pd.DataFrame) -> list[list[str]]:
    graph = _components(block)
    unseen, groups = set(graph), []
    while unseen:
        group = _connected(graph, min(unseen))
        groups.append(sorted(group))
        unseen -= group
    return sorted(groups, key=lambda group: (-len(group), group))


def select_pair_items(manifest: pd.DataFrame, limit: int) -> pd.DataFrame:
    keys = ["bench", "judge", "target", "opponent"]
    selected = [
        rows[rows.item.astype(str).isin(stable_items(rows.item.astype(str).tolist(), limit))]
        for _, rows in manifest.groupby(keys, sort=True)
    ]
    return (pd.concat(selected, ignore_index=True).sort_values(keys + ["item"])
            .reset_index(drop=True) if selected else manifest.iloc[0:0].copy())


def filter_available_sources(manifest: pd.DataFrame, base_configs: dict,
                             root: Path) -> pd.DataFrame:
    def available(row) -> bool:
        transcripts = base_configs[(row.bench, row.judge)]["transcripts"]
        return all((root / transcripts[transcript_key(model, transcripts)]).exists()
                   for model in (row.target, row.opponent))
    return manifest[manifest.apply(available, axis=1)].reset_index(drop=True)


def select_topup_items(left: set[str], right: set[str], q1: set[str], limit: int = 8) -> list[str]:
    return stable_items(sorted(left & right & q1), limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path,
                        default=ROOT / "logs/frontier-eci-canonical-anchors")
    parser.add_argument("--base-config-root", type=Path,
                        default=ROOT / "configs/frontier_eci/runs")
    parser.add_argument("--anchor-config", type=Path,
                        default=ROOT / "configs/frontier_eci_canonical_anchors/experiment.yaml")
    parser.add_argument("--config-root", type=Path,
                        default=ROOT / "configs/reasoning_ablation")
    parser.add_argument("--result-root", type=Path,
                        default=ROOT / "judge_results/frontier/reasoning_ablation")
    parser.add_argument("--judges", nargs="+", default=["gpt-5.2", "gemini-2.5-pro"])
    parser.add_argument("--reasoning-samples", type=int, default=50)
    parser.add_argument("--items-per-pair", type=int, default=4)
    args = parser.parse_args()

    base_configs = {
        (path.parent.name, path.stem): yaml.safe_load(path.read_text())
        for path in args.base_config_root.glob("*/*.yaml") if path.stem in args.judges
    }
    anchor = yaml.safe_load(args.anchor_config.read_text()) if args.anchor_config.exists() else {}
    extra = anchor.get("extra_transcripts", {})
    base_configs = {
        key: config | {"transcripts": config["transcripts"] | extra.get(key[0], {})}
        for key, config in base_configs.items()
    }
    _, _, protocol, _, _ = load_rank_analysis_inputs(args.anchor_config, args.logs_root)
    protocol = cap_pair_items(protocol[protocol.judge.isin(args.judges)], 8)
    manifest = matched_pair_items(protocol)
    available_manifest = filter_available_sources(manifest, base_configs, ROOT)
    selected_manifest = select_pair_items(available_manifest, args.items_per_pair)

    cot_root = args.config_root / "cot_on"
    cot_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cot_root / "pair_items.csv", index=False)
    selected_manifest.to_csv(cot_root / "selected_pair_items.csv", index=False)
    configs = write_configs(
        selected_manifest, base_configs, cot_root / "runs", include_reasoning=True,
        log_root_template="logs/reasoning_ablation/cot_on/{bench}",
    )

    desired_topups = [
        ("gpqa", "gpt-5.2", "gpt-4o-1120", "gpt-4.1"),
        ("gpqa", "gpt-5.2", "gpt-4.1", "mistral-medium-3"),
        ("swe_bench_mini", "gpt-5.2", "gpt-4.1", "mistral-medium-3"),
    ]
    transcript_items = {}
    for bench, judge, left, right in desired_topups:
        transcripts = base_configs[(bench, judge)]["transcripts"]
        for model in (left, right):
            path = transcripts[model]
            if path not in transcript_items:
                transcript_items[path] = set(
                    read_transcripts(ROOT / path, include_reasoning=False))
    topup_rows = []
    q1_items = {bench: set(rows.item.astype(str))
                for bench, rows in protocol.groupby("bench", sort=True)}
    for bench, judge, left, right in desired_topups:
        transcripts = base_configs[(bench, judge)]["transcripts"]
        items = select_topup_items(
            transcript_items[transcripts[left]], transcript_items[transcripts[right]],
            q1_items[bench])
        topup_rows.extend({
            "bench": bench, "judge": judge, "target": right,
            "opponent": left, "item": item,
        } for item in items)
    topup_manifest = pd.DataFrame(topup_rows)
    topup_root = args.config_root / "non_reasoning_topup"
    topup_root.mkdir(parents=True, exist_ok=True)
    topup_manifest.to_csv(topup_root / "pair_items.csv", index=False)
    topup_configs = write_configs(
        topup_manifest, base_configs, topup_root / "runs", include_reasoning=False,
        log_root_template="logs/reasoning_ablation/non_reasoning_topup/{bench}",
    )

    sources = sorted({
        (bench, model, path)
        for (bench, _), config in base_configs.items()
        for model, path in config["transcripts"].items()
    })
    reasoning_rows = []
    for bench, model, path in sources:
        scanned = reasoning = 0
        error = ""
        try:
            for sample in read_eval_log_samples(str(ROOT / path), all_samples_required=False):
                scanned += 1
                reasoning += extract_transcript(sample, True) != extract_transcript(sample, False)
                if scanned >= args.reasoning_samples:
                    break
        except Exception as exc:  # noqa: BLE001 - audit records missing/corrupt sources
            error = str(exc)
        reasoning_rows.append({
            "bench": bench, "model": model, "path": path, "samples_scanned": scanned,
            "samples_with_reasoning": int(reasoning), "error": error,
        })
    reasoning_table = pd.DataFrame(reasoning_rows)
    model_status = (reasoning_table.groupby("model", sort=True)
                    .agg(samples_scanned=("samples_scanned", "sum"),
                         samples_with_reasoning=("samples_with_reasoning", "sum"),
                         source_errors=("error", lambda values: " | ".join(v for v in values if v)))
                    .reset_index())
    model_status["is_reasoning"] = model_status.samples_with_reasoning.gt(0)
    non_reasoning = set(model_status.loc[
        model_status.samples_scanned.gt(0) & ~model_status.is_reasoning & model_status.source_errors.eq(""),
        "model"])
    non_reasoning_verdicts = protocol[
        protocol.model_left.isin(non_reasoning) & protocol.model_right.isin(non_reasoning)
    ].copy()
    coverage = []
    for bench in sorted(protocol.bench.unique()):
        for judge in args.judges:
            block = non_reasoning_verdicts[
                non_reasoning_verdicts.bench.eq(bench) & non_reasoning_verdicts.judge.eq(judge)]
            groups = components(block) if not block.empty else []
            coverage.append({
                "bench": bench, "judge": judge, "n_verdicts": len(block),
                "n_models": len(set(block.model_left) | set(block.model_right)),
                "n_pairs": len(block[["model_left", "model_right"]].apply(
                    lambda row: tuple(sorted(row.astype(str))), axis=1).drop_duplicates()),
                "n_components": len(groups),
                "largest_component_models": "|".join(groups[0]) if groups else "",
                "largest_component_size": len(groups[0]) if groups else 0,
            })

    preflight = args.result_root / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    reasoning_table.to_csv(preflight / "reasoning_by_log.csv", index=False)
    model_status.to_csv(preflight / "model_reasoning.csv", index=False)
    pd.DataFrame(coverage).to_csv(preflight / "non_reasoning_coverage.csv", index=False)
    non_reasoning_verdicts.to_parquet(preflight / "non_reasoning_verdicts.parquet", index=False)
    reserve_rows = []
    for path in [*configs, *topup_configs]:
        config = yaml.safe_load(path.read_text())
        _, target, opponent = config["comparisons"][0]
        reserve_rows.append({
            "config": path.relative_to(ROOT).as_posix(), "target": target,
            "opponent": opponent, "upper_cost": 0.5,
        })
    pd.DataFrame(reserve_rows).to_csv(preflight / "paid_reserves.csv", index=False)
    summary = {
        "judges": args.judges,
        "full_matched_pair_items": len(manifest),
        "available_matched_pair_items": len(available_manifest),
        "selected_cot_pair_items": len(selected_manifest),
        "items_per_pair": args.items_per_pair,
        "cot_on_configs": len(configs),
        "estimated_cot_on_calls": 2 * len(selected_manifest),
        "non_reasoning_topup_configs": len(topup_configs),
        "estimated_non_reasoning_topup_calls": 2 * len(topup_manifest),
        "models_scanned": len(model_status),
        "reasoning_models": int(model_status.is_reasoning.sum()),
        "non_reasoning_models": len(non_reasoning),
    }
    (preflight / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
