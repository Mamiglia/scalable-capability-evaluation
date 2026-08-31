"""Freeze canonical-anchor coverage and the direct-pair top-up manifest."""

import argparse
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log_samples

from judge.util import extract_score


ROOT = Path(__file__).resolve().parents[2]
PAIR_COLUMNS = ["bench", "judge", "target", "opponent", "item", "reason"]


def release_key(name: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", name.split("/")[-1].lower())


def release_date(releases: dict[str, str], model: str) -> pd.Timestamp:
    value = releases.get(model, releases.get(release_key(model)))
    return pd.Timestamp(value) if value is not None else pd.NaT


def prediction_cutoff(target: str, judge: str, releases: dict[str, str]) -> pd.Timestamp:
    target_date = release_date(releases, target)
    judge_date = release_date(releases, judge)
    return max(target_date, judge_date) if pd.notna(judge_date) else target_date


def eligible_models(models: list[str], target: str, judge: str,
                    releases: dict[str, str]) -> list[str]:
    cutoff = prediction_cutoff(target, judge, releases)
    return [model for model in models if pd.notna(release_date(releases, model))
            and release_date(releases, model) < cutoff]


def canonical_anchor_models(models: list[str], target: str, judge: str,
                            releases: dict[str, str]) -> list[str]:
    return [model for model in eligible_models(models, target, judge, releases) if model != target]


def stable_items(items: list[str], limit: int = 8) -> list[str]:
    return sorted({str(item) for item in items},
                  key=lambda item: hashlib.sha256(item.encode()).hexdigest())[:limit]


def choose_pairs(existing: pd.DataFrame, candidates: pd.DataFrame,
                 desired_direct_anchors: int) -> pd.DataFrame:
    """Choose missing direct anchors before requesting further items for known pairs."""
    known = existing.groupby("target").opponent.agg(set).to_dict() if not existing.empty else {}
    rows = []
    for target, options in candidates.groupby("target", sort=False):
        missing = options[~options.opponent.isin(known.get(target, set()))]
        count = len(known.get(target, set()))
        rows.append(missing.head(max(0, desired_direct_anchors - count)))
    return pd.concat(rows, ignore_index=True) if rows else candidates.iloc[0:0].copy()


def _components(edges: pd.DataFrame) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in edges[["model_left", "model_right"]].drop_duplicates().itertuples(index=False):
        graph[row.model_left].add(row.model_right)
        graph[row.model_right].add(row.model_left)
    return graph


def _connected(graph: dict[str, set[str]], target: str) -> set[str]:
    seen, queue = {target}, deque([target])
    while queue:
        model = queue.popleft()
        for neighbor in graph.get(model, set()) - seen:
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


def _names(value: list[str]) -> str:
    return json.dumps(sorted(value))


def _joint_correct_items(rollouts: pd.DataFrame, bench: str, target: str,
                         opponent: str) -> set[str]:
    rows = rollouts[rollouts.bench == bench]
    def correct_items(model: str) -> set[str]:
        return set(rows[rows.model == model].groupby("item").correct.all().loc[lambda x: x].index.astype(str))
    return correct_items(target) & correct_items(opponent)


def coverage_blocks(config: dict, rollouts: pd.DataFrame, verdicts: pd.DataFrame,
                    quartiles: pd.DataFrame) -> pd.DataFrame:
    releases = config["releases"]
    canonical = sorted(rollouts.loc[rollouts.is_canonical, "model"].dropna().unique())
    backfill = sorted(rollouts.loc[
        rollouts.get("is_historical_backfill", pd.Series(False, index=rollouts.index)),
        "model"].dropna().unique())
    targets = config["candidates"]
    judges = [judge.split("/")[-1] for group in config["judges"].values() for judge in group]
    q1 = quartiles[quartiles.item_quartile == 1][["bench", "item"]].copy()
    q1["item"] = q1.item.astype(str)
    verdicts = verdicts.copy()
    verdicts["item"] = verdicts.item.astype(str)
    verdicts = verdicts.merge(q1, on=["bench", "item"], how="inner")
    q1_rollouts = rollouts.copy()
    q1_rollouts["item"] = q1_rollouts.item.astype(str)
    q1_rollouts = q1_rollouts.merge(q1, on=["bench", "item"], how="inner")
    accuracy = q1_rollouts.groupby(
        ["bench", "model"]).correct.mean().to_dict()
    rows = []
    for bench in sorted(quartiles.bench.unique()):
        for judge in judges:
            bench_verdicts = verdicts[(verdicts.bench == bench) & (verdicts.judge == judge)]
            for target in targets:
                anchors = canonical_anchor_models(canonical, target, judge, releases)
                allowed = set(anchors) | {target}
                edges = bench_verdicts[bench_verdicts.model_left.isin(allowed)
                                      & bench_verdicts.model_right.isin(allowed)]
                graph = _components(edges)
                connected = sorted(set(anchors) & _connected(graph, target))
                direct = sorted(set(anchors) & set(
                    edges.loc[edges.model_left.eq(target), "model_right"].tolist()
                    + edges.loc[edges.model_right.eq(target), "model_left"].tolist()))
                outcomes = pd.concat([
                    edges[["model_left", "pick_left"]].rename(columns={"model_left": "model"}),
                    edges[["model_right", "pick_left"]].assign(pick_left=lambda d: 1 - d.pick_left)
                    .rename(columns={"model_right": "model"}),
                ])
                ability = outcomes.groupby("model").pick_left.mean().to_dict()
                existing = pd.DataFrame({"target": target, "opponent": direct})
                viable = [anchor for anchor in anchors
                          if _joint_correct_items(q1_rollouts, bench, target, anchor)]
                candidates = pd.DataFrame({"target": target,
                                           "opponent": viable,
                                           "hard_accuracy": [accuracy.get((bench, anchor), float("nan"))
                                                             for anchor in viable]})
                recommended = choose_pairs(existing, candidates, config["desired_direct_anchors"])
                recommendations = recommended.opponent.tolist()
                attainable = len(set(direct) | set(recommendations))
                historical_recommendations = []
                if attainable < config["minimum_anchors"]:
                    historical = canonical_anchor_models(backfill, target, judge, releases)
                    historical = [anchor for anchor in historical
                                  if _joint_correct_items(q1_rollouts, bench, target, anchor)]
                    already = sorted(set(direct) | set(recommendations))
                    historical_candidates = pd.DataFrame({
                        "target": target,
                        "opponent": historical,
                        "hard_accuracy": [accuracy.get((bench, anchor), float("nan"))
                                          for anchor in historical],
                    })
                    historical_recommendations = choose_pairs(
                        pd.DataFrame({"target": target, "opponent": already}),
                        historical_candidates,
                        config["minimum_anchors"],
                    ).opponent.tolist()
                    recommendations.extend(historical_recommendations)
                    attainable = len(set(direct) | set(recommendations))
                accuracies = [accuracy.get((bench, model)) for model in connected]
                accuracies = [value for value in accuracies if pd.notna(value)]
                abilities = [ability.get(model) for model in connected]
                abilities = [value for value in abilities if pd.notna(value)]
                exclusions = []
                if not anchors:
                    exclusions.append("no_eligible_canonical_anchors")
                if len(direct) < config["desired_direct_anchors"]:
                    exclusions.append("insufficient_direct_anchors")
                if attainable < config["minimum_anchors"]:
                    exclusions.append("needs_historical_backfill")
                rows.append({
                    "bench": bench, "judge": judge, "target": target,
                    "eligible_anchors": _names(anchors), "connected_anchors": _names(connected),
                    "direct_anchors": _names(direct), "recommended_anchors": _names(recommendations),
                    "historical_backfill_anchors": _names(historical_recommendations),
                    "eligible_anchor_count": len(anchors), "connected_anchor_count": len(connected),
                    "direct_anchor_count": len(direct), "attainable_direct_anchor_count": attainable,
                    "accuracy_proxy_min": min(accuracies) if accuracies else float("nan"),
                    "accuracy_proxy_max": max(accuracies) if accuracies else float("nan"),
                    "accuracy_proxy_range": max(accuracies) - min(accuracies) if accuracies else float("nan"),
                    "ability_proxy_min": min(abilities) if abilities else float("nan"),
                    "ability_proxy_max": max(abilities) if abilities else float("nan"),
                    "ability_proxy_range": max(abilities) - min(abilities) if abilities else float("nan"),
                    "status": "needs_historical_backfill" if attainable < config["minimum_anchors"]
                    else "identifiable",
                    "exclusion_reasons": _names(exclusions),
                })
    return pd.DataFrame(rows)


def pair_items(blocks: pd.DataFrame, rollouts: pd.DataFrame, verdicts: pd.DataFrame,
               limit: int) -> pd.DataFrame:
    rows = []
    for block in blocks.itertuples(index=False):
        anchors = json.loads(block.recommended_anchors)
        historical = set(json.loads(getattr(block, "historical_backfill_anchors", "[]")))
        bench = rollouts[(rollouts.bench == block.bench)
                         & (rollouts.item_quartile == 1)]
        for opponent in anchors:
            common = _joint_correct_items(bench, block.bench, block.target, opponent)
            ordered = sorted(common, key=lambda item: hashlib.sha256(
                f"{block.bench}|{block.target}|{opponent}|{item}".encode()).hexdigest())[:limit]
            rows.extend({"bench": block.bench, "judge": block.judge, "target": block.target,
                         "opponent": opponent, "item": item,
                         "reason": "historical_backfill" if opponent in historical
                         else "missing_direct_anchor"}
                        for item in ordered)
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    base = yaml.safe_load(_path(config["base_experiment"]).read_text())
    releases = dict(base["releases"])
    for model_config in (ROOT / "configs/models").rglob("*.yml"):
        spec = yaml.safe_load(model_config.read_text()) or {}
        if not spec.get("name") or not spec.get("release_date"):
            continue
        for name in (spec["name"].split("/")[-1], spec.get("slug", "").split("/")[-1]):
            if name:
                releases.setdefault(name, str(spec["release_date"])[:10])
                releases.setdefault(release_key(name), str(spec["release_date"])[:10])
    releases.update(config.get("releases", {}))
    base["releases"] = releases
    return {**base, **config, "releases": releases}


def historical_rollouts(sources: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows = []
    for bench, models in sources.items():
        for model, path in models.items():
            for sample in read_eval_log_samples(str(_path(path)), all_samples_required=False):
                rows.append({"bench": bench, "model": model, "item": str(sample.id),
                             "correct": extract_score(sample), "is_canonical": False,
                             "is_historical_backfill": True})
    return pd.DataFrame(rows, columns=[
        "bench", "model", "item", "correct", "is_canonical", "is_historical_backfill"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config)
    canonical = pd.read_parquet(_path(config["canonical_identity"]),
                                columns=["bench", "model", "item", "correct", "is_canonical"])
    frontier = pd.read_parquet(_path(config["frontier_rollouts"]))
    canonical = canonical[canonical.is_canonical]
    canonical["is_historical_backfill"] = False
    frontier = frontier.copy()
    frontier["is_canonical"] = False
    frontier["is_historical_backfill"] = False
    present = pd.MultiIndex.from_frame(canonical[["bench", "model"]]).unique()
    frontier_keys = pd.MultiIndex.from_frame(frontier[["bench", "model"]])
    historical = historical_rollouts(config.get("historical_backfill", {}))
    historical_keys = pd.MultiIndex.from_frame(historical[["bench", "model"]])
    rollouts = pd.concat([
        canonical,
        frontier[~frontier_keys.isin(present)],
        historical[~historical_keys.isin(present)],
    ], ignore_index=True)
    verdicts = pd.concat([pd.read_parquet(_path(path)) for path in config["verdicts"]], ignore_index=True)
    quartiles = pd.read_csv(_path(config["quartiles"]), dtype={"item": str})
    rollouts["item"] = rollouts.item.astype(str)
    rollouts = rollouts.merge(quartiles[["bench", "item", "item_quartile"]], on=["bench", "item"], how="inner")
    blocks = coverage_blocks(config, rollouts, verdicts, quartiles)
    items = pair_items(blocks, rollouts, verdicts, config["pair_item_cap"])
    out = _path(config["results"]) / "coverage"
    out.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(out / "blocks.csv", index=False)
    items.to_csv(out / "pair_items.csv", index=False)
    summary = {"identifiable_blocks": int((blocks.status == "identifiable").sum()),
               "top_up_blocks": int(blocks.recommended_anchors.map(json.loads).map(bool).sum()),
               "needs_historical_backfill_blocks": int((blocks.status == "needs_historical_backfill").sum()),
               "pair_items": len(items)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("identifiable={identifiable_blocks} top_up={top_up_blocks} backfill_required={needs_historical_backfill_blocks}".format(**summary))


if __name__ == "__main__":
    main()
