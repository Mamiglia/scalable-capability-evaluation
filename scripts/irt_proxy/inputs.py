"""Load and validate the response and pair-judgment populations.

Responses and judgments deliberately stay separate in this module. IRT can use
any evaluated response log, while Elo exists only for models connected by pair
judgments. Conflating those populations would inflate the downstream sample.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from irt.data import read_log

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ELO_DIR = ROOT / "scripts" / "analyses" / "judge-elo"
sys.path.insert(0, str(ELO_DIR))
from elo import components  # noqa: E402

DEEPSEEK = {
    "api-deepseek-v4-flash",
    "deepseek-v4-flash",
    "deepseek-v4-flash-0731",
}

# Downstream points represent ordinary model configurations. Faster/lighter,
# preview, and explicitly low/high-effort variants are different treatments,
# so they do not belong in the same model-ranking analysis.
NONSTANDARD_CANDIDATE_MARKERS = ("-low", "-high", "-preview", "-lite", "-flash")

# Epoch and the viewer occasionally spell the same endpoint differently. Keep
# this small and explicit: fuzzy name matching would make exclusions auditable
# only by accident.
MODEL_ALIASES = {
    "Llama-2-70b-chat-hf": "llama-2-70b",
    "Llama-3.1-8B-Instruct": "llama-3.1-8b",
    "Llama-3.1-70B-Instruct": "llama-3.1-70b",
    "Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "Meta-Llama-3-70B-Instruct": "llama-3-70b",
    "Mistral-7B-Instruct-v0.3": "mistral-7b-v0.3",
    "Mixtral-8x7B-Instruct-v0.1": "mixtral-8x7b",
    "Phi-3-medium-128k-instruct": "phi-3-medium",
    "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-0620",
    "claude-3-5-haiku-20241022": "claude-3-5-haiku",
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    "claude-3-haiku-20240307": "claude-3-haiku",
    "gemini-1.5-pro-002": "gemini-1.5-pro",
    "gemini-2.0-flash-001": "gemini-2.0-flash",
    "gemini-3.5-flash_high": "gemini-3.5-flash",
    "gemma-2-27b-it": "gemma-2-27b",
    "gemma-2-9b-it": "gemma-2-9b",
    "gemma-3-27b-it": "gemma-3-27b",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
    "gpt-4.5-preview-2025-02-27": "gpt-4.5-preview",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-5-2025-08-07_high": "gpt-5-high",
    "gpt-5-mini-2025-08-07_medium": "gpt-5-mini-medium",
    "gpt-5.4-mini-2026-03-17_high": "gpt-5.4-mini-high",
    "gpt-5.5_low": "gpt-5.5-low",
    "o3-mini-2025-01-31_medium": "o3-mini-medium",
    "qwen2.5-32b-instruct": "qwen2.5-32b",
    "qwen2.5-72b-instruct": "qwen-2.5-72b-instruct",
}

BENCHMARK_ALIASES = {
    "GPQA Diamond": "gpqa",
    "gpqa": "gpqa",
    "gpqa-diamond": "gpqa",
    "aime2025": "aime2025",
    "aime2026": "aime2026",
}


def canonical_model(name: str) -> str:
    """Return the viewer-facing name for a known model alias."""
    return MODEL_ALIASES.get(str(name), str(name))


def canonical_benchmark(name: str) -> str:
    """Normalize declared benchmark labels and reject unknown formats."""
    try:
        return BENCHMARK_ALIASES[str(name)]
    except KeyError as exc:
        # A silent default here could apply AIME's floor to a new benchmark.
        raise ValueError(f"unknown benchmark label: {name!r}") from exc


def is_deepseek(name: str) -> bool:
    """Recognize every known DeepSeek V4 Flash alias after canonicalization."""
    return canonical_model(name).lower() in DEEPSEEK


def is_standard_candidate(name: str) -> bool:
    """Keep only default/medium evaluated models; judge names are unrestricted."""
    model = canonical_model(name).lower().replace("_", "-")
    return not any(marker in model for marker in NONSTANDARD_CANDIDATE_MARKERS)


def load_config(path: Path) -> dict:
    """Read the human-editable run configuration."""
    return yaml.safe_load(path.read_text())


def select_local_eval_paths(
    roots: Mapping[str, Path],
) -> dict[tuple[str, str], Path]:
    """Choose exactly one canonical local log per benchmark/model.

    GPQA's ordinary Gemini log contains the correct transcripts but parser-
    broken labels. Its separately stored rescored log is therefore the scoring
    source. AIME did not have that parser problem and keeps its ordinary log.
    Selection happens before deduplication because these are distinct files.
    """
    selected = {}
    for raw_benchmark, root in roots.items():
        benchmark = canonical_benchmark(raw_benchmark)
        ordinary_directories = sorted(
            path
            for path in Path(root).iterdir()
            if path.is_dir() and not path.name.endswith("-rescored")
        )
        for directory in ordinary_directories:
            model = canonical_model(directory.name)
            scoring_directory = directory
            if benchmark == "gpqa" and model == "gemini-3.1-flash-lite":
                scoring_directory = directory.with_name(
                    "gemini-3.1-flash-lite-rescored",
                )
            matches = sorted((scoring_directory / "0").glob("*.eval"))
            if len(matches) != 1:
                raise ValueError(
                    f"expected one eval for {benchmark}/{model}, found {len(matches)}"
                )
            selected[(benchmark, model)] = matches[0]
    return selected


def read_named_evals(
    named_paths: Mapping[tuple[str, str], Path],
    *,
    source: str,
    source_priority: int,
    reader: Callable[[Path], Iterable[dict]] = read_log,
) -> pd.DataFrame:
    """Read evals using declared provenance instead of timestamp filenames."""
    rows = []
    for (benchmark, model), path in sorted(named_paths.items()):
        for response in reader(path):
            rows.append({
                **response,
                "benchmark": canonical_benchmark(benchmark),
                "model": canonical_model(model),
                "source": source,
                "source_priority": source_priority,
                "source_path": str(path),
            })
    return pd.DataFrame(rows)


def load_responses(root: Path, config: Mapping) -> pd.DataFrame:
    """Load broad Epoch GPQA responses plus nonduplicate local collections."""
    paths = config["paths"]
    epoch = pd.read_parquet(root / paths["epoch_responses"]).copy()
    epoch["benchmark"] = epoch["benchmark"].map(canonical_benchmark)
    epoch["model"] = epoch["model"].map(canonical_model)
    epoch["source"] = "epoch"
    epoch["source_priority"] = 0
    epoch["source_path"] = ""

    local_roots = {
        benchmark: root / path
        for benchmark, path in paths["local_responses"].items()
    }
    expansion_paths = {
        key: path
        for key, path in select_local_eval_paths(local_roots).items()
        if not is_deepseek(key[1])
    }

    pilot_config = yaml.safe_load((root / paths["pilot_responses"]).read_text())
    epoch_logs = (root / paths["epoch_logs"]).resolve()
    pilot_paths = {}
    for model, raw_path in pilot_config["transcripts"].items():
        path = (root / raw_path).resolve()
        # Some pilot YAML entries point directly into the imported Epoch pool.
        # Skipping those paths prevents one physical collection counting twice.
        if path.is_relative_to(epoch_logs) or is_deepseek(model):
            continue
        pilot_paths[("gpqa", canonical_model(model))] = path

    # If two manifests name the same physical log, the newer expansion
    # declaration wins. This is provenance precedence, not score cherry-picking.
    expansion_files = {path.resolve() for path in expansion_paths.values()}
    pilot_paths = {
        key: path
        for key, path in pilot_paths.items()
        if path.resolve() not in expansion_files
    }

    responses = pd.concat([
        epoch,
        read_named_evals(pilot_paths, source="pilot", source_priority=1),
        read_named_evals(expansion_paths, source="expansion", source_priority=2),
    ], ignore_index=True)
    responses = responses[~responses["model"].map(is_deepseek)].copy()

    clashes = responses.groupby(["benchmark", "item"])["question"].nunique()
    clashes = clashes[clashes > 1]
    if not clashes.empty:
        raise ValueError(f"clashing question text for items: {clashes.index.tolist()}")
    return responses


def load_votes(viewer_root: Path, datasets: Mapping[str, str]) -> pd.DataFrame:
    """Load order-level pair judgments under the asserted `full` condition.

    The glob itself excludes 10/50/90 files, then the parsed filename is checked
    again. The redundant assertion is intentional: condition contamination
    would change the estimand while still producing plausible-looking results.
    """
    rows = []
    for source, raw_benchmark in datasets.items():
        benchmark = canonical_benchmark(raw_benchmark)
        pair_directory = Path(viewer_root) / source / "pairs"
        for path in sorted(pair_directory.glob("*__*__full.json")):
            major, minor, condition = path.stem.split("__", 2)
            if condition != "full":
                raise ValueError(f"non-full judgment path loaded: {path}")
            major, minor = canonical_model(major), canonical_model(minor)
            if (
                major == minor
                or is_deepseek(major)
                or is_deepseek(minor)
                or not is_standard_candidate(major)
                or not is_standard_candidate(minor)
            ):
                continue

            payload = json.loads(path.read_text())
            for item_id, by_judge in payload.items():
                for raw_judge, verdict in by_judge.items():
                    judge = canonical_model(raw_judge)
                    seen_orders = set()
                    for order_result in verdict.get("orders", []):
                        if order_result is None:
                            continue
                        order = order_result.get("slot_of_major")
                        if order in seen_orders:
                            raise ValueError(f"duplicate presentation order in {path}")
                        seen_orders.add(order)
                        picked = order_result.get("picked_major")
                        parse_ok = (
                            not isinstance(picked, bool)
                            and isinstance(picked, (int, float))
                            and float(picked) in (0.0, 1.0)
                        )
                        rows.append({
                            "source": source,
                            "benchmark": benchmark,
                            "condition": condition,
                            "item_id": str(item_id),
                            "judge": judge,
                            "major": major,
                            "minor": minor,
                            "order": str(order),
                            "picked_major": float(picked) if parse_ok else np.nan,
                            "parse_ok": parse_ok,
                            "n_calls": 1,
                        })
    return pd.DataFrame(rows)


def load_metadata(
    viewer_root: Path,
    datasets: Mapping[str, str],
) -> pd.DataFrame:
    """Read evaluated-model labels, dates, and accuracies from viewer indexes."""
    rows = []
    for source, raw_benchmark in datasets.items():
        benchmark = canonical_benchmark(raw_benchmark)
        index = json.loads((Path(viewer_root) / source / "index.json").read_text())
        for raw_model, metadata in index.get("models", {}).items():
            model = canonical_model(raw_model)
            if is_deepseek(model):
                continue
            rows.append({
                "source": source,
                "benchmark": benchmark,
                "model": model,
                "label": metadata.get("label", model),
                "release": metadata.get("release") or None,
                "eci": metadata.get("eci"),
                "accuracy": metadata.get("accuracy"),
                **{
                    key: value
                    for key, value in metadata.items()
                    if key.startswith("acc_")
                },
            })
    return pd.DataFrame(rows)


def _graph_summary(votes: pd.DataFrame) -> tuple[set[str], list[set[str]]]:
    """Return graph membership without pretending components share an Elo scale."""
    valid = votes[votes["parse_ok"]] if "parse_ok" in votes else votes
    edges = sorted(set(zip(valid["major"], valid["minor"])))
    graph_models = {model for edge in edges for model in edge}
    graph_components = components(edges) if edges else []
    graph_components = sorted(
        graph_components,
        key=lambda component: (-len(component), tuple(sorted(component))),
    )
    return graph_models, graph_components


def underconnected_models(
    votes: pd.DataFrame,
    candidates: set[str],
    minimum: int = 4,
) -> dict[str, int]:
    """Return evaluated models with fewer than the required distinct opponents."""
    valid = votes[votes["parse_ok"]] if "parse_ok" in votes else votes
    opponents = {model: set() for model in candidates}
    for major, minor in valid[["major", "minor"]].drop_duplicates().itertuples(index=False):
        if major in opponents and minor in opponents:
            opponents[major].add(minor)
            opponents[minor].add(major)
    return {
        model: len(neighbours)
        for model, neighbours in sorted(opponents.items())
        if len(neighbours) < minimum
    }


def preflight_counts(
    responses: pd.DataFrame,
    votes: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Report response, evaluated, and judgment populations in distinct fields."""
    if not votes.empty and set(votes["condition"]) != {"full"}:
        raise ValueError("non-full judgments entered preflight")
    if "model" in responses and responses["model"].map(is_deepseek).any():
        raise ValueError("DeepSeek V4 Flash remains in responses")
    # DeepSeek may judge, but it must never become an evaluated response model
    # or a member of a fitted comparison pair.
    for column in ("major", "minor"):
        if column in votes and votes[column].map(is_deepseek).any():
            raise ValueError(f"DeepSeek V4 Flash remains in votes.{column}")
        if column in votes and not votes[column].map(is_standard_candidate).all():
            raise ValueError(f"nonstandard evaluated model remains in votes.{column}")

    rows = []
    benchmarks = sorted(set(responses["benchmark"]) | set(votes["benchmark"]))
    for benchmark in benchmarks:
        response_models = set(
            responses.loc[responses["benchmark"] == benchmark, "model"]
        )
        graph_models, graph_components = _graph_summary(
            votes[votes["benchmark"] == benchmark]
        )
        rows.append(_count_row(
            scope=benchmark,
            benchmark=benchmark,
            source="all",
            response_models=len(response_models),
            evaluated_models=np.nan,
            graph_models=graph_models,
            graph_components=graph_components,
        ))

    for source in sorted(votes["source"].unique()):
        source_votes = votes[votes["source"] == source]
        benchmark = source_votes["benchmark"].iloc[0]
        graph_models, graph_components = _graph_summary(source_votes)
        evaluated_models = np.nan
        if metadata is not None and not metadata.empty:
            evaluated_models = metadata.loc[
                metadata["source"] == source, "model"
            ].nunique()
        rows.append(_count_row(
            scope=f"source:{source}",
            benchmark=benchmark,
            source=source,
            response_models=np.nan,
            evaluated_models=evaluated_models,
            graph_models=graph_models,
            graph_components=graph_components,
        ))
    return pd.DataFrame(rows)


def _count_row(
    *,
    scope: str,
    benchmark: str,
    source: str,
    response_models,
    evaluated_models,
    graph_models: set[str],
    graph_components: list[set[str]],
) -> dict:
    """Format one population row while preserving every component explicitly."""
    return {
        "scope": scope,
        "benchmark": benchmark,
        "source": source,
        "n_response_models": response_models,
        "n_evaluated_models": evaluated_models,
        "n_graph_models": len(graph_models),
        "n_components": len(graph_components),
        "component_sizes": ",".join(str(len(c)) for c in graph_components),
        "component_models": ";".join(
            "|".join(sorted(component)) for component in graph_components
        ),
    }


def validate_inputs(
    responses: pd.DataFrame,
    votes: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    require_four_opponents: bool = True,
) -> pd.DataFrame:
    """Assert the manually verified repository-wide input invariants."""
    counts = preflight_counts(responses, votes, metadata)
    expected_evaluated = {
        "gpqa-pilot-rd2": 10,
        "data-expansion-gpqa": 44,
        "data-expansion-aime2025": 8,
        "data-expansion-aime2026": 8,
    }
    for source, wanted in expected_evaluated.items():
        row = counts.loc[counts["scope"] == f"source:{source}"].iloc[0]
        observed = int(row["n_evaluated_models"])
        if observed != wanted or int(row["n_graph_models"]) == 0:
            raise ValueError(
                f"unexpected populations for {source}: "
                f"evaluated={observed}, graph={int(row['n_graph_models'])}; "
                f"expected {wanted} evaluated and a nonempty graph"
            )

    gpqa = counts.loc[counts["scope"] == "gpqa"].iloc[0]
    if int(gpqa["n_components"]) != 1:
        raise ValueError(
            f"combined standard-effort GPQA graph is disconnected: "
            f"{gpqa['component_sizes']}"
        )

    gpqa_candidates = set(metadata.loc[
        metadata["benchmark"].eq("gpqa")
        & metadata["model"].map(is_standard_candidate),
        "model",
    ])
    underconnected = underconnected_models(
        votes[votes["benchmark"].eq("gpqa")],
        gpqa_candidates,
    )
    if underconnected and require_four_opponents:
        raise ValueError(
            "GPQA models need four distinct opponents: "
            f"{underconnected}"
        )

    # This simultaneously checks canonical naming, rescored-log selection, and
    # exclusion of the unrescored physical sibling.
    gemini = responses[
        (responses["benchmark"] == "gpqa")
        & (responses["model"] == "gemini-3.1-flash-lite")
        & (responses["source"] == "expansion")
    ]
    if len(gemini) != 792 or int(gemini["correct"].sum()) != 579:
        raise ValueError("GPQA Gemini did not use exactly one rescored 792-cell log")
    return counts
