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
from judge.util import extract_transcript  # noqa: E402
from scripts.frontier_eci.canonical_analysis import cap_pair_items  # noqa: E402
from scripts.frontier_eci.canonical_coverage import _components, _connected  # noqa: E402
from scripts.frontier_eci.rank_regression import load_rank_analysis_inputs  # noqa: E402
from scripts.frontier_eci.write_topup_configs import matched_pair_items, write_configs  # noqa: E402


def components(block: pd.DataFrame) -> list[list[str]]:
    graph = _components(block)
    unseen, groups = set(graph), []
    while unseen:
        group = _connected(graph, min(unseen))
        groups.append(sorted(group))
        unseen -= group
    return sorted(groups, key=lambda group: (-len(group), group))


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

    cot_root = args.config_root / "cot_on"
    cot_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cot_root / "pair_items.csv", index=False)
    configs = write_configs(
        manifest, base_configs, cot_root / "runs", include_reasoning=True,
        log_root_template="logs/reasoning_ablation/cot_on/{bench}",
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
    summary = {
        "judges": args.judges,
        "matched_pair_items": len(manifest),
        "cot_on_configs": len(configs),
        "estimated_cot_on_calls": 2 * len(manifest),
        "models_scanned": len(model_status),
        "reasoning_models": int(model_status.is_reasoning.sum()),
        "non_reasoning_models": len(non_reasoning),
    }
    (preflight / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
