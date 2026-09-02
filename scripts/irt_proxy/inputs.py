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
from inspect_ai.log import read_eval_log

from irt.data import read_log
from mc_cost.ranking import components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

HIBAYES_DIR = ROOT / "scripts" / "hibayes"
sys.path.insert(0, str(HIBAYES_DIR))
from style_features import feature_rates  # noqa: E402

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
    "glm4.6": "glm-4.6",
    "sonnet4.5": "claude-sonnet-4.5",
    "gpt5": "gpt-5",
    "opus4.7": "claude-opus-4.7",
    "gpt4.1": "gpt-4.1",
    "granite": "granite",
    "ling": "ling-2.6-flash",
}

BENCHMARK_ALIASES = {
    "GPQA Diamond": "gpqa",
    "gpqa": "gpqa",
    "gpqa-diamond": "gpqa",
    "aime2025": "aime2025",
    "aime2026": "aime2026",
}
VIEWER_SOURCE_ALIASES = {
    "hf-judging-2026-08-17-gpqa": "hf-judging-2026-08-17",
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


def canonical_viewer_source(name: str) -> str:
    """Map a viewer export directory to its canonical collection source."""
    return VIEWER_SOURCE_ALIASES.get(str(name), str(name))


def canonical_viewer_item(source: str, item: str) -> str:
    """Remove the HF export's response-epoch suffix from a GPQA item ID."""
    item = str(item)
    return item.split("__ep", 1)[0] if source == "hf-judging-2026-08-17" else item


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


def load_responses(
    root: Path,
    config: Mapping,
    *,
    all_candidates: bool = False,
) -> pd.DataFrame:
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
        if all_candidates or not is_deepseek(key[1])
    }

    pilot_config = yaml.safe_load((root / paths["pilot_responses"]).read_text())
    epoch_logs = (root / paths["epoch_logs"]).resolve()
    pilot_paths = {}
    for model, raw_path in pilot_config["transcripts"].items():
        path = (root / raw_path).resolve()
        # Some pilot YAML entries point directly into the imported Epoch pool.
        # Skipping those paths prevents one physical collection counting twice.
        if path.is_relative_to(epoch_logs) or (
            not all_candidates and is_deepseek(model)
        ):
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
    if not all_candidates:
        responses = responses[~responses["model"].map(is_deepseek)].copy()

    clashes = responses.groupby(["benchmark", "item"])["question"].nunique()
    clashes = clashes[clashes > 1]
    if not clashes.empty:
        raise ValueError(f"clashing question text for items: {clashes.index.tolist()}")
    return responses


def load_votes(
    viewer_root: Path,
    datasets: Mapping[str, str],
    *,
    all_candidates: bool = False,
) -> pd.DataFrame:
    """Load order-level pair judgments under the asserted `full` condition.

    The glob itself excludes 10/50/90 files, then the parsed filename is checked
    again. The redundant assertion is intentional: condition contamination
    would change the estimand while still producing plausible-looking results.
    """
    rows = []
    for viewer_dataset, raw_benchmark in datasets.items():
        source = canonical_viewer_source(viewer_dataset)
        benchmark = canonical_benchmark(raw_benchmark)
        pair_directory = Path(viewer_root) / viewer_dataset / "pairs"
        for path in sorted(pair_directory.glob("*__*__full.json")):
            major, minor, condition = path.stem.split("__", 2)
            if condition != "full":
                raise ValueError(f"non-full judgment path loaded: {path}")
            major, minor = canonical_model(major), canonical_model(minor)
            excluded = not all_candidates and (
                    major == minor
                    or
                    is_deepseek(major) or is_deepseek(minor)
                    or not is_standard_candidate(major)
                    or not is_standard_candidate(minor)
            )
            if excluded:
                continue

            payload = json.loads(path.read_text())
            for item_id, by_judge in payload.items():
                for raw_judge, verdict in by_judge.items():
                    judge = canonical_model(raw_judge)
                    seen_orders = set()
                    for position, order_result in enumerate(
                        verdict.get("orders") or [None], start=1,
                    ):
                        if order_result is None:
                            rows.append({
                                "source": source,
                                "viewer_dataset": viewer_dataset,
                                "artifact": f"{source}/pairs/{path.name}",
                                "round": f"{source}:{path.stem}",
                                "benchmark": benchmark,
                                "condition": condition,
                                "viewer_item_id": str(item_id),
                                "item_id": canonical_viewer_item(source, item_id),
                                "judge": judge,
                                "major": major,
                                "minor": minor,
                                "order": str(position),
                                "picked_major": np.nan,
                                "chosen_left": np.nan,
                                "parse_ok": False,
                                "exclusion_reason": "missing viewer verdict",
                                "n_calls": 1,
                            })
                            continue
                        order = order_result.get("slot_of_major")
                        if str(order) in {"1", "2"} and order in seen_orders:
                            raise ValueError(f"duplicate presentation order in {path}")
                        if str(order) in {"1", "2"}:
                            seen_orders.add(order)
                        picked = order_result.get("picked_major")
                        raw_pick = order_result.get("pick")
                        valid_order = str(order) in {"1", "2"}
                        if major == minor:
                            parse_ok = valid_order and str(raw_pick) in {"1", "2"}
                            chosen_left = float(str(raw_pick) == "1") if parse_ok else np.nan
                        else:
                            parse_ok = valid_order and (
                                not isinstance(picked, bool)
                                and isinstance(picked, (int, float))
                                and float(picked) in (0.0, 1.0)
                            )
                            chosen_left = (
                                float(picked) if str(order) == "1" else 1 - float(picked)
                            ) if parse_ok else np.nan
                        rows.append({
                            "source": source,
                            "viewer_dataset": viewer_dataset,
                            "artifact": f"{source}/pairs/{path.name}",
                            "round": f"{source}:{path.stem}",
                            "benchmark": benchmark,
                            "condition": condition,
                            "viewer_item_id": str(item_id),
                            "item_id": canonical_viewer_item(source, item_id),
                            "judge": judge,
                            "major": major,
                            "minor": minor,
                            "order": str(order),
                            "picked_major": (
                                float(picked) if parse_ok and major != minor else np.nan
                            ),
                            "chosen_left": chosen_left,
                            "parse_ok": parse_ok,
                            "exclusion_reason": (
                                "" if parse_ok else "invalid viewer verdict"
                            ),
                            "n_calls": 1,
                        })
    return pd.DataFrame(rows)


def _response_identity(path: str | Path) -> tuple[str, str]:
    """Return the benchmark/model encoded in a recorded response-log path."""
    parts = Path(path).parts
    try:
        index = parts.index("data")
        return canonical_benchmark(parts[index + 1]), canonical_model(parts[-3])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"cannot identify recorded response path: {path}") from exc


def _eval_index(root: Path, names: set[str] | None = None) -> dict[str, Path]:
    """Index raw evals by manifest filename and reject ambiguous archives."""
    indexed: dict[str, Path] = {}
    for path in sorted(Path(root).rglob("*.eval")):
        if names is not None and path.name not in names:
            continue
        if path.name in indexed:
            raise ValueError(f"duplicate raw eval basename: {path.name}")
        indexed[path.name] = path
    return indexed


def _hf_manifest_judge(path: Path, benchmark: str) -> str:
    prefix = path.parent.name.split(f"-{benchmark}-", 1)[0]
    return prefix.removeprefix("pilot-")


def load_hf_votes(
    manifest_root: Path,
    eval_root: Path,
    benchmark: str = "gpqa",
    reader: Callable[[Path], object] = read_eval_log,
    *,
    return_exclusions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load item-level HF judgments; manifest summaries never become votes."""
    manifests = [
        path for path in sorted(Path(manifest_root).rglob("logs.json"))
        if benchmark in path.parent.name.split("-")
    ]
    loaded, excluded = [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        judge = _hf_manifest_judge(path, benchmark)
        if judge.lower().startswith("phi"):
            excluded.extend({
                "source": "hf-judging-2026-08-17", "judge": judge,
                "artifact": f"hf-judging-2026-08-17/{Path(raw_name).name}",
                "reason": "excluded_phi_judge",
            } for raw_name, record in manifest.items()
            if isinstance(record, Mapping) and record.get("status") == "success")
            continue
        loaded.append((path, manifest))
    names = {
        Path(raw_name).name
        for _, manifest in loaded
        for raw_name, record in manifest.items()
        if isinstance(record, Mapping) and record.get("status") == "success"
    }
    evals = _eval_index(eval_root, names)
    rows = []
    seen_artifacts = set()
    for manifest_path, manifest in loaded:
        for raw_name, record in manifest.items():
            if not isinstance(record, Mapping) or record.get("status") != "success":
                continue
            try:
                path = evals[Path(raw_name).name]
            except KeyError as exc:
                raise ValueError(f"missing raw eval named by {manifest_path}: {raw_name}") from exc
            artifact = str(path.resolve())
            if artifact in seen_artifacts:
                continue
            seen_artifacts.add(artifact)
            log = reader(path)
            if getattr(log, "status", None) != "success":
                continue
            metadata = getattr(getattr(log, "eval", None), "metadata", None) or {}
            major_path, minor_path = metadata.get("major_path"), metadata.get("minor_path")
            if not major_path or not minor_path:
                raise ValueError(f"missing recorded response paths in {path}")
            benchmark, major = _response_identity(major_path)
            minor_benchmark, minor = _response_identity(minor_path)
            if benchmark != minor_benchmark:
                raise ValueError(f"mixed benchmarks in {path}")
            model_args = getattr(getattr(log, "eval", None), "model_args", None) or {}
            judge = getattr(getattr(log, "eval", None), "model", None) or manifest_path.parent.name
            for sample in getattr(log, "samples", None) or []:
                try:
                    item_id, major_slot = str(sample.id).rsplit("_", 1)
                except ValueError as exc:
                    raise ValueError(f"invalid HF sample id in {path}: {sample.id!r}") from exc
                if major_slot not in {"1", "2"}:
                    raise ValueError(f"invalid major slot in {path}: {major_slot!r}")
                score = next(iter((getattr(sample, "scores", None) or {}).values()), None)
                picked = getattr(score, "answer", None)
                major_left = major_slot == "1"
                rows.append({
                    "source": "hf-judging-2026-08-17", "benchmark": benchmark,
                    "artifact": f"hf-judging-2026-08-17/{path.name}",
                    "round": path.stem,
                    "condition": "natural", "judge": str(judge),
                    "judge_model_args": json.dumps(model_args, sort_keys=True, default=str),
                    "item_id": item_id, "order": int(major_slot),
                    "major_path": str(major_path), "minor_path": str(minor_path),
                    "left_model": major if major_left else minor,
                    "right_model": minor if major_left else major,
                    "chosen_left": float(picked == "1") if picked in {"1", "2"} else np.nan,
                })
    votes = pd.DataFrame(rows)
    exclusions = (
        pd.DataFrame(excluded)
        .groupby(["source", "judge", "artifact", "reason"], as_index=False, sort=True)
        .size().rename(columns={"size": "count"})
        if excluded else pd.DataFrame(columns=[
            "source", "judge", "artifact", "reason", "count",
        ])
    )
    return (votes, exclusions) if return_exclusions else votes


BIAS_FEATURES = (
    "log_chars", "personal_pronouns", "hedging", "backtracking",
    "uncertainty", "confidence", "questions",
)


def pair_feature_rows(votes: pd.DataFrame, lexicon: Mapping) -> pd.DataFrame:
    """Measure existing lexicon features directly on the judged transcripts."""
    def measured(text):
        return {"log_chars": np.log1p(len(text)), **feature_rates(text, lexicon)}

    left = pd.DataFrame([measured(text) for text in votes["left_text"]])
    right = pd.DataFrame([measured(text) for text in votes["right_text"]])
    result = votes.reset_index(drop=True).copy()
    for name in BIAS_FEATURES:
        result[f"delta_{name}"] = left[name] - right[name]
    return result


def read_response_tree(
    root: Path,
    benchmark: str,
    reader: Callable[[Path], object] = read_eval_log,
) -> pd.DataFrame:
    """Read exact transcripts from a downloaded ``model/<epoch>/*.eval`` tree."""
    from judge.util import extract_transcript

    rows = []
    for path in sorted(Path(root).glob("*/*/*.eval")):
        model, epoch = path.relative_to(root).parts[:2]
        log = reader(path)
        if getattr(log, "status", None) != "success":
            continue
        for sample in getattr(log, "samples", None) or []:
            score = next(iter((getattr(sample, "scores", None) or {}).values()), None)
            rows.append({
                "benchmark": canonical_benchmark(benchmark),
                "model": canonical_model(model), "item": str(sample.id),
                "epoch": int(epoch), "transcript": extract_transcript(sample, include_reasoning=False),
                "answer": getattr(score, "answer", None),
                "correct": getattr(score, "value", None) == "C",
                "target": getattr(sample, "target", None),
                "question": " ".join(str(getattr(sample, "input", "")).split())[:80],
                "source": "hf-restored", "source_priority": 3,
                "source_path": str(path),
            })
    return pd.DataFrame(rows)


def resolve_hf_responses(
    votes: pd.DataFrame,
    downloaded: pd.DataFrame,
    epoch: pd.DataFrame,
    recorded: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach recorded-path, downloaded, then canonical-Epoch transcripts."""
    rows, unresolved = [], []
    for vote in votes.to_dict("records"):
        row = dict(vote)
        major_left = int(row.get("order", 1)) == 1
        for side, path_column in (("left", "major_path" if major_left else "minor_path"),
                                  ("right", "minor_path" if major_left else "major_path")):
            model = row[f"{side}_model"]
            def select(frame):
                if frame.empty or not {"benchmark", "model", "item"}.issubset(frame):
                    return pd.DataFrame()
                return frame[
                    frame["benchmark"].eq(row["benchmark"])
                    & frame["model"].eq(model)
                    & frame["item"].astype(str).eq(str(row["item_id"]))
                ]

            exact = select(recorded) if recorded is not None and not recorded.empty else pd.DataFrame()
            if not exact.empty:
                exact = exact[exact.get("source_path", pd.Series("", index=exact.index)).eq(row[path_column])]
            matched = exact if len(exact) else select(downloaded)
            if matched.empty:
                matched = select(epoch)
            if len(matched) != 1:
                unresolved.append((row["benchmark"], model))
            else:
                row[f"{side}_text"] = matched.iloc[0]["transcript"]
                row[f"{side}_epoch"] = matched.iloc[0].get("epoch", 0)
        if "left_text" in row and "right_text" in row:
            rows.append(row)
    if unresolved:
        raise ValueError(f"unresolved HF responses: {sorted(set(unresolved))}")
    return pd.DataFrame(rows)


def _recorded_hf_responses(votes: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Read response logs explicitly named by HF judge metadata when present."""
    from judge.util import extract_transcript

    rows = []
    for path in sorted({str(path) for column in ("major_path", "minor_path") for path in votes[column]}):
        local = Path(path)
        if not local.is_absolute():
            local = Path(root) / local
        if not local.exists():
            continue
        benchmark, model = _response_identity(path)
        log = read_eval_log(local)
        if log.status != "success":
            continue
        for sample in log.samples or []:
            score = next(iter((getattr(sample, "scores", None) or {}).values()), None)
            rows.append({
                "benchmark": benchmark, "model": model, "item": str(sample.id),
                "epoch": getattr(sample, "epoch", 0),
                "transcript": extract_transcript(sample, include_reasoning=False),
                "answer": getattr(score, "answer", None),
                "correct": getattr(score, "value", None) == "C",
                "target": getattr(sample, "target", None),
                "question": " ".join(str(getattr(sample, "input", "")).split())[:80],
                "source": "hf-recorded", "source_priority": 4,
                "source_path": str(path),
            })
    return pd.DataFrame(rows)


def _epoch_transcript_rows(root: Path, config: Mapping) -> pd.DataFrame:
    """Read the canonical Epoch fallback transcripts for HF response resolution."""
    from judge.util import extract_transcript

    rows = []
    epoch_root = Path(root) / config["paths"]["epoch_logs"]
    for path in sorted(epoch_root.glob("*.eval")):
        log = read_eval_log(path)
        if getattr(log, "status", None) != "success":
            continue
        for sample in getattr(log, "samples", None) or []:
            rows.append({
                "benchmark": "gpqa", "model": canonical_model(path.stem),
                "item": str(sample.id), "epoch": getattr(sample, "epoch", 0),
                "transcript": extract_transcript(sample, include_reasoning=False),
                "source_path": str(path),
            })
    return pd.DataFrame(rows)


def _viewer_correction_votes(
    viewer_root: Path,
    datasets: Mapping[str, str],
    response_rows: pd.DataFrame,
    *,
    return_exclusions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Orient viewer major/minor rows into left/right exact transcript rows."""
    votes = load_votes(viewer_root, datasets, all_candidates=True)
    invalid = votes[~votes["parse_ok"]].copy()
    exclusions = (
        invalid.assign(reason=invalid["exclusion_reason"])
        .groupby(["source", "judge", "artifact", "reason"], as_index=False, sort=True)
        .size().rename(columns={"size": "count"})
    )
    votes = votes[votes["parse_ok"]].copy()
    rows, missing = [], []
    for (source, viewer_dataset), source_votes in votes.groupby(
        ["source", "viewer_dataset"], sort=True,
    ):
        texts = {
            canonical_model(path.stem.partition("__ep")[0])
            + (f"__ep{path.stem.partition('__ep')[2]}" if "__ep" in path.stem else ""):
            json.loads(path.read_text())
            for path in (
                Path(viewer_root) / viewer_dataset / "transcripts"
            ).glob("*.json")
        }
        for vote in source_votes.to_dict("records"):
            major_left = str(vote["order"]) == "1"
            major, minor = vote["major"], vote["minor"]
            item = vote.get("viewer_item_id", vote["item_id"])
            major_key = major
            minor_key = f"{minor}__ep1" if major == minor else minor
            try:
                major_text, minor_text = texts[major_key][item], texts[minor_key][item]
            except KeyError:
                missing.append({
                    "source": source, "judge": vote["judge"],
                    "artifact": vote["artifact"],
                    "reason": "missing_viewer_transcript",
                })
                continue
            def verified(model, text, epoch):
                matched = response_rows[
                    response_rows["benchmark"].eq(vote["benchmark"])
                    & response_rows["model"].eq(model)
                    & response_rows["item"].astype(str).eq(item)
                ]
                if "source" in matched:
                    matched = matched[matched["source"].eq(source)]
                if "epoch" in matched:
                    matched = matched[matched["epoch"].eq(epoch)]
                matched = matched[matched["transcript"].str.len().eq(len(text))]
                if matched.empty:
                    raise ValueError(f"viewer transcript length mismatch for {source}/{model}/{item}")
                matched = matched[matched["transcript"].eq(text)]
                if len(matched) != 1:
                    raise ValueError(f"viewer transcript mismatch for {source}/{model}/{item}")
                return matched.iloc[0]

            major_response = verified(major, major_text, 0)
            minor_response = verified(minor, minor_text, int(major == minor))
            rows.append({
                "source": source, "benchmark": vote["benchmark"], "condition": vote["condition"],
                "artifact": vote["artifact"], "round": vote["round"],
                "judge": vote["judge"], "item_id": vote["item_id"],
                "order": int(vote["order"]),
                "left_model": major if major_left else minor,
                "right_model": minor if major_left else major,
                "left_text": major_text if major_left else minor_text,
                "right_text": minor_text if major_left else major_text,
                "left_epoch": major_response["epoch"] if major_left else minor_response["epoch"],
                "right_epoch": minor_response["epoch"] if major_left else major_response["epoch"],
                "chosen_left": vote["chosen_left"],
            })
    if missing:
        missing_exclusions = (
            pd.DataFrame(missing)
            .groupby(["source", "judge", "artifact", "reason"], as_index=False, sort=True)
            .size().rename(columns={"size": "count"})
        )
        exclusions = pd.concat(
            [exclusions, missing_exclusions], ignore_index=True,
        ).sort_values(
            ["source", "judge", "artifact", "reason"], ignore_index=True,
        )
    result = pd.DataFrame(rows)
    return (result, exclusions) if return_exclusions else result


def _canonical_response_rows(root: Path, config: Mapping) -> pd.DataFrame:
    """Read each viewer source's authoritative transcript export."""
    viewer_root = Path(root) / config["paths"]["viewer_data"]
    rows = []
    for viewer_dataset, raw_benchmark in config["correction"]["viewer_datasets"].items():
        source = canonical_viewer_source(viewer_dataset)
        benchmark = canonical_benchmark(raw_benchmark)
        for path in sorted(
            (viewer_root / viewer_dataset / "transcripts").glob("*.json"),
        ):
            raw_model, separator, suffix = path.stem.partition("__ep")
            epoch = int(suffix) if separator else 0
            payload = json.loads(path.read_text())
            if not isinstance(payload, Mapping):
                raise ValueError(f"invalid viewer transcript artifact: {path}")
            for item, transcript in payload.items():
                if not isinstance(transcript, str):
                    raise ValueError(f"invalid viewer transcript in {path}: {item}")
                rows.append({
                    "source": source, "benchmark": benchmark,
                    "model": canonical_model(raw_model), "item": str(item),
                    "epoch": epoch, "transcript": transcript,
                    "source_path": f"{source}/transcripts/{path.name}",
                })
    result = pd.DataFrame(rows)
    identity = ["source", "benchmark", "model", "item", "epoch"]
    if not result.empty and result.duplicated(identity).any():
        raise ValueError("duplicate source-qualified viewer transcript")
    return result


def load_hf_benchmark_scores(root: Path, benchmark: str = "gpqa") -> pd.DataFrame:
    """Load canonical candidate cells from exported benchmark reductions."""
    rows = []
    for path in sorted(Path(root).glob("*/*/*/logs.json")):
        _, raw_model, raw_epoch, _ = path.relative_to(root).parts
        try:
            epoch = int(raw_epoch)
        except ValueError as exc:
            raise ValueError(f"invalid HF benchmark-score epoch: {path}") from exc
        manifest = json.loads(path.read_text())
        eligible = [
            (name, record) for name, record in manifest.items()
            if isinstance(record, Mapping)
            and record.get("status") == "success"
            and not record.get("invalidated", False)
        ]
        if not eligible:
            continue
        if len(eligible) != 1:
            raise ValueError(f"ambiguous HF benchmark-score entries: {path}")
        artifact, record = eligible[0]
        reductions = [
            reduction for reduction in record.get("reductions", [])
            if reduction.get("scorer") == "choice"
        ]
        if len(reductions) != 1:
            raise ValueError(f"expected one choice reduction: {path}")
        seen_items = set()
        for sample in reductions[0].get("samples", []):
            item = str(sample.get("sample_id", ""))
            value = sample.get("value")
            answer = sample.get("answer")
            transcript = sample.get("explanation")
            if not item or item in seen_items:
                raise ValueError(f"duplicate/missing HF benchmark-score item: {path}")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or float(value) not in (0.0, 1.0)
            ):
                raise ValueError(f"invalid HF benchmark correctness for {item}: {path}")
            if not isinstance(answer, str):
                raise ValueError(f"missing HF benchmark answer for {item}: {path}")
            if not isinstance(transcript, str):
                raise ValueError(f"missing HF benchmark transcript for {item}: {path}")
            seen_items.add(item)
            rows.append({
                "benchmark": canonical_benchmark(benchmark),
                "model": canonical_model(raw_model), "item": item,
                "epoch": epoch, "answer": answer.strip() or "<no_answer>",
                "transcript": transcript, "correct": bool(value),
                "target": None, "question": "",
                "source": "hf-benchmark-scores", "source_priority": 3,
                "source_path": f"{path}#{artifact}",
            })
    return pd.DataFrame(rows)


def canonical_scored_pool(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine exact GPQA scoring rows without collapsing distinct artifacts."""
    frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["benchmark"] = result["benchmark"].map(canonical_benchmark)
    result["model"] = result["model"].map(canonical_model)
    result = result[result["benchmark"].eq("gpqa")].copy()
    identity = [
        "source", "source_path", "benchmark", "model", "item", "epoch",
    ]
    duplicate = result.duplicated(identity, keep=False)
    if duplicate.any():
        conflicts = result.loc[duplicate].groupby(identity, dropna=False).nunique()
        if conflicts.gt(1).any(axis=None):
            raise ValueError("conflicting rows for one scored response artifact")
        result = result.drop_duplicates(identity)
    return result.reset_index(drop=True)


def load_correction_data(
    root: Path,
    config: Mapping,
    *,
    return_exclusions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the configured GPQA correction pool without changing legacy inputs."""
    paths = config["paths"]
    correction = config["correction"]
    viewer_root = Path(root) / paths["viewer_data"]
    viewer_responses = _canonical_response_rows(root, config)
    viewer_votes, viewer_exclusions = _viewer_correction_votes(
        viewer_root, correction["viewer_datasets"], viewer_responses,
        return_exclusions=True,
    )
    exclusions = [viewer_exclusions]
    scored_frames = [load_responses(root, config, all_candidates=True)]
    hf_mask = viewer_votes["source"].eq("hf-judging-2026-08-17")
    if hf_mask.any():
        score_root = correction.get("hf_benchmark_scores")
        if not score_root:
            raise ValueError("HF viewer requires correction.hf_benchmark_scores")
        hf_scores = load_hf_benchmark_scores(
            Path(root) / score_root, correction["benchmark"],
        )
        viewer_models = (
            set(viewer_votes.loc[hf_mask, "left_model"])
            | set(viewer_votes.loc[hf_mask, "right_model"])
        )
        scored_models = set(hf_scores["model"])
        hf_candidates = viewer_models & scored_models
        keep_hf = (
            viewer_votes["left_model"].isin(hf_candidates)
            & viewer_votes["right_model"].isin(hf_candidates)
        )
        unmatched = viewer_votes[hf_mask & ~keep_hf]
        if not unmatched.empty:
            exclusions.append(
                unmatched.assign(reason="missing_hf_benchmark_score")
                .groupby(["source", "judge", "artifact", "reason"], as_index=False, sort=True)
                .size().rename(columns={"size": "count"})
            )
        viewer_votes = viewer_votes[~hf_mask | keep_hf].copy()
        scored_frames.append(hf_scores[hf_scores["model"].isin(hf_candidates)])
    votes = [viewer_votes]
    combined = pd.concat(votes, ignore_index=True)
    lexicon = yaml.safe_load((Path(root) / correction["style_lexicon"]).read_text())
    combined = pair_feature_rows(combined, lexicon)
    responses = canonical_scored_pool(scored_frames)
    excluded = pd.concat(exclusions, ignore_index=True).sort_values(
        ["source", "judge", "artifact", "reason"], ignore_index=True,
    ) if exclusions else pd.DataFrame(columns=[
        "source", "judge", "artifact", "reason", "count",
    ])
    return (combined, responses, excluded) if return_exclusions else (combined, responses)


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
