"""Build the frozen, exact-input manifest for the 20260828 rejudge."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import urllib.request
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.solver import generate, system_message

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from judge.dataset import has_reasoning  # noqa: E402
from judge.scorer import judge_scorer  # noqa: E402
from configs import ModelConfig  # noqa: E402
from model import Model  # noqa: E402

SEED = 20260828
PER_CELL = 100
SKIP_JUDGES = {"ling-2.6-flash"}
SOURCE_KEYS = [
    "bench_source", "judge", "item", "model_left", "model_right",
    "epoch_left", "epoch_right", "log_left", "log_right", "judge_log",
    "judge_sample", "judge_epoch", "content_hash_l", "content_hash_r",
]
FREEZE = ROOT / "judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66"
OUT = ROOT / "judge_results/verdict_dissection/exact_rejudge_20260828"
JUDGING_ROOT = Path(os.environ.get("EXACT_REJUDGE_JUDGING_ROOT", "/Volumes/T7/data/hf-judging"))
HF_JUDGING_REPO = os.environ.get("EXACT_REJUDGE_HF_REPO", "LASR-G5/judging")
load_dotenv(ROOT / ".env")
MODEL_CONFIGS = {
    "gpt-oss-120b": ROOT / "configs/models/openai/gpt-oss-120b.yml",
    "deepseek-r1-0528": ROOT / "configs/models/deepseek/deepseek-r1-0528.yml",
    "o4-mini": ROOT / "configs/models/openai/o4-mini.yml",
    "gemma-4-31b-it": ROOT / "configs/models/google/gemma4-31b.yml",
    "deepseek-v4-flash-0731": ROOT / "configs/models/deepseek/deepseek-v4-flash-0731.yml",
    "Llama-3.1-8B-Instruct": ROOT / "configs/models/meta-llama/llama3.1-8b.yml",
}
OPENROUTER_ROUTES = {
    "gpt-oss-120b": ("openai/gpt-oss-120b", "deepinfra/bf16"),
    "deepseek-r1-0528": ("deepseek/deepseek-r1-0528", "siliconflow/fp8"),
    "o4-mini": ("openai/o4-mini", "openai"),
    "gemma-4-31b-it": ("google/gemma-4-31b-it", "novita/bf16"),
    "deepseek-v4-flash-0731": ("deepseek/deepseek-v4-flash-0731", "deepinfra/fp8"),
    "ling-2.6-flash": ("inclusionai/ling-2.6-flash", None),
}
OPENROUTER_HEADERS = {
    "X-OpenRouter-Cache": "false",
    "X-OpenRouter-Metadata": "enabled",
}
EXPECTED_CELLS = {
    (judge, bench)
    for judge, benches in {
        "gpt-oss-120b": ("aime", "gpqa", "intercode", "swe_bench_mini"),
        "deepseek-r1-0528": ("gpqa", "intercode", "swe_bench_mini"),
        "o4-mini": ("gpqa", "intercode", "swe_bench_mini"),
        "gemma-4-31b-it": ("aime", "gpqa", "intercode", "swe_bench_mini"),
        "deepseek-v4-flash-0731": ("aime", "gpqa", "intercode", "swe_bench_mini"),
        "Llama-3.1-8B-Instruct": ("gpqa", "intercode", "swe_bench_mini"),
    }.items()
    for bench in benches
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def seeded_rank(*parts: object, seed: int = SEED) -> str:
    return sha256_text("\0".join(map(str, (seed, *parts))))


def _bench(bench: str) -> str:
    return "aime" if str(bench).startswith("aime") else str(bench)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _source_draw_id(row: dict[str, object]) -> str:
    return sha256_text("\0".join(str(row[key]) for key in SOURCE_KEYS))


def _subset(bench_source: str) -> str:
    if str(bench_source).startswith("aime"):
        return f"aime/{bench_source}"
    return f"pilot/{bench_source}"


def _log_path(row: dict[str, object]) -> Path:
    relative = f'{_subset(str(row["bench_source"]))}/{row["judge_log"]}'
    path = JUDGING_ROOT / relative
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(HF_JUDGING_REPO, relative, repo_type="dataset",
                                    local_dir=JUDGING_ROOT))
    return path


def _as_jsonable(value: object) -> object:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else value


@lru_cache(maxsize=None)
def system_message_from_log(path, sample_id, epoch) -> dict[str, object]:
    """Read the one historical system message and provenance for a source sample."""
    path = str(path)
    for sample in read_eval_log(path).samples or []:
        if str(sample.id) != str(sample_id) or int(getattr(sample, "epoch", 0) or 0) != int(epoch):
            continue
        system = [message for message in (getattr(sample, "messages", None) or ())
                  if getattr(message, "role", None) == "system"]
        if len(system) != 1:
            raise ValueError(f"expected one system message in {path}:{sample_id}:{epoch}")
        output = getattr(sample, "output", None)
        return {
            "system_prompt": str(getattr(system[0], "content", "")),
            "usage": _as_jsonable(getattr(output, "usage", None)),
            "model_call": _as_jsonable(getattr(output, "model_call", None)),
        }
    raise ValueError(f"source sample not found: {path}:{sample_id}:{epoch}")


def _historical_prompt(row: dict[str, object]) -> dict[str, object]:
    path = _log_path(row)
    return system_message_from_log(str(path), str(row["judge_sample"]), int(row["judge_epoch"])) | {
        "system_source_log_path": str(path),
        "system_source_sample": str(row["judge_sample"]),
        "system_source_epoch": int(row["judge_epoch"]),
    }


def _reasoning_hashes(store: pd.DataFrame) -> set[str]:
    required = {"content_hash", "transcript"}
    if missing := required - set(store):
        raise ValueError(f"store missing columns: {sorted(missing)}")
    texts = store[["content_hash", "transcript"]].dropna().drop_duplicates("content_hash")
    return {str(row.content_hash) for row in texts.itertuples(index=False) if has_reasoning(str(row.transcript))}


def build_manifest(pairs, store, prompt_loader, seed: int = SEED, per_cell: int = PER_CELL) -> pd.DataFrame:
    """Select shared, answer-bearing historical inputs without reconstructing prompts."""
    required = (set(SOURCE_KEYS) - {"bench_source"}) | {"bench", "judge_config_hash", "judge_input", "pick_left"}
    if missing := required - set(pairs):
        raise ValueError(f"pairs missing columns: {sorted(missing)}")
    rows = pairs.copy()
    rows["bench_source"] = rows["bench"].astype(str)
    rows["bench"] = rows["bench_source"].map(_bench)
    eligible_hashes = _reasoning_hashes(store)
    rows = rows[(~rows.judge.isin(SKIP_JUDGES)) & rows.content_hash_l.astype(str).isin(eligible_hashes)
                & rows.content_hash_r.astype(str).isin(eligible_hashes)].copy()
    if rows.empty:
        raise ValueError("no answer-bearing rows after exclusions")
    rows["underlying_pair_hash"] = [
        sha256_text("\0".join(sorted((str(left), str(right)))))
        for left, right in zip(rows.content_hash_l, rows.content_hash_r)
    ]
    rows["input_hash"] = rows.judge_input.map(lambda value: sha256_text(str(value)))

    counts = rows.groupby(["judge", "bench", "judge_config_hash"], sort=False).size().rename("config_count")
    dominant = counts.reset_index()
    dominant["rank"] = [seeded_rank(*values, seed=seed) for values in
                        dominant[["judge", "bench", "judge_config_hash"]].itertuples(index=False, name=None)]
    dominant = (dominant
        .sort_values(["judge", "bench", "config_count", "rank"], ascending=[True, True, False, True])
        .drop_duplicates(["judge", "bench"]))
    total = counts.groupby(level=[0, 1]).sum().rename("config_total")
    dominant = dominant.merge(total, left_on=["judge", "bench"], right_index=True)
    rows = rows.merge(dominant[["judge", "bench", "judge_config_hash", "config_count", "config_total"]],
                      on=["judge", "bench", "judge_config_hash"], how="inner")
    rows["config_share"] = rows.config_count / rows.config_total
    rows["source_rank"] = [seeded_rank(*values, seed=seed) for values in
                           rows[SOURCE_KEYS].itertuples(index=False, name=None)]

    # One historical draw per judge/config/pair/input; all subsequent selection uses it.
    chosen = (rows.sort_values("source_rank")
              .drop_duplicates(["judge", "bench", "underlying_pair_hash", "input_hash"]))
    coverage = (chosen.groupby(["bench", "underlying_pair_hash", "input_hash"], sort=False).judge.nunique()
                .rename("coverage").reset_index())
    coverage["rank"] = [seeded_rank(*values, seed=seed) for values in
                        coverage[["bench", "underlying_pair_hash", "input_hash"]].itertuples(index=False, name=None)]
    shown = (coverage.sort_values(["bench", "underlying_pair_hash", "coverage", "rank"],
                                  ascending=[True, True, False, True])
             .drop_duplicates(["bench", "underlying_pair_hash"]))
    selected = chosen.merge(shown[["bench", "underlying_pair_hash", "input_hash"]],
                            on=["bench", "underlying_pair_hash", "input_hash"], how="inner")
    selected["pair_rank"] = [seeded_rank(*values, seed=seed) for values in
                             selected[["bench", "underlying_pair_hash"]].itertuples(index=False, name=None)]

    picked, cell_counts = [], Counter()
    for _, group in selected.sort_values(["pair_rank", "judge", "source_rank"]).groupby(
            ["bench", "underlying_pair_hash"], sort=False):
        for row in group.to_dict("records"):
            cell = (row["judge"], row["bench"])
            if cell_counts[cell] < per_cell:
                picked.append(row)
                cell_counts[cell] += 1
    manifest = pd.DataFrame(picked)
    if manifest.empty or not manifest.groupby(["judge", "bench"]).size().eq(per_cell).all():
        sizes = manifest.groupby(["judge", "bench"]).size().to_dict()
        expected = selected[["judge", "bench"]].drop_duplicates().to_dict("records")
        raise ValueError(f"insufficient eligible pairs for {expected}: selected {sizes}")

    output, prompt_cache = [], {}
    for row in manifest.to_dict("records"):
        prompt_key = (row["judge"], row["bench"], row["judge_config_hash"])
        if prompt_key not in prompt_cache:
            prompt_cache[prompt_key] = prompt_loader(row)
        info = prompt_cache[prompt_key]
        if isinstance(info, str):
            info = {"system_prompt": info}
        prompt = info.get("system_prompt")
        if not isinstance(prompt, str):
            raise ValueError(f"prompt loader returned no system prompt for {row['judge_sample']}")
        output.append({key: row[key] for key in SOURCE_KEYS} | {
            "bench": row["bench"], "judge_config_hash": row["judge_config_hash"],
            "dominant_config_hash": row["judge_config_hash"], "config_count": int(row["config_count"]),
            "config_share": float(row["config_share"]), "pick_left": int(row["pick_left"]),
            "target": "1" if int(row["pick_left"]) else "2", "underlying_pair_hash": row["underlying_pair_hash"],
            "source_draw_id": _source_draw_id(row), "judge_input": str(row["judge_input"]),
            "input_hash": row["input_hash"], "system_prompt": prompt,
            "system_prompt_hash": sha256_text(prompt), "seed": seed,
            "historical_usage_json": _json(info.get("usage")),
            "historical_model_call_json": _json(info.get("model_call")),
            "system_source_log_path": info.get("system_source_log_path"),
            "system_source_sample": info.get("system_source_sample"),
            "system_source_epoch": info.get("system_source_epoch"),
        })
    return pd.DataFrame(output).sort_values(["judge", "bench", "source_draw_id"]).reset_index(drop=True)


def validate_manifest(manifest, store, per_cell: int = PER_CELL) -> None:
    """Fail closed if a manifest ceases to be an exact, usable rejudge source."""
    required = set(SOURCE_KEYS) | {"bench", "judge_config_hash", "dominant_config_hash", "config_count",
               "config_share", "pick_left", "target", "underlying_pair_hash", "source_draw_id",
               "judge_input", "input_hash", "system_prompt", "system_prompt_hash"}
    if per_cell == PER_CELL:
        required |= {"system_source_log_path", "system_source_sample", "system_source_epoch",
                     "current_model", "current_config_json", "current_config_hash", "current_route"}
    if missing := required - set(manifest):
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if manifest.empty or not manifest.groupby(["judge", "bench"]).size().eq(per_cell).all():
        raise ValueError("manifest cells must each have exactly per_cell rows")
    if per_cell == PER_CELL and set(map(tuple, manifest[["judge", "bench"]].drop_duplicates().to_numpy())) != EXPECTED_CELLS:
        raise ValueError("manifest must contain the fixed 21 judge x benchmark cells")
    if manifest.judge.isin(SKIP_JUDGES).any():
        raise ValueError("manifest contains excluded judge")
    if not manifest.input_hash.eq(manifest.judge_input.map(sha256_text)).all():
        raise ValueError("judge_input hash mismatch")
    if not manifest.system_prompt_hash.eq(manifest.system_prompt.map(sha256_text)).all():
        raise ValueError("system prompt hash mismatch")
    if manifest.duplicated("source_draw_id").any():
        raise ValueError("duplicate source draw")
    if any(_source_draw_id(row) != row["source_draw_id"] for row in manifest.to_dict("records")):
        raise ValueError("source draw id mismatch")
    expected_pairs = [sha256_text("\0".join(sorted((str(left), str(right)))))
                      for left, right in zip(manifest.content_hash_l, manifest.content_hash_r)]
    if not manifest.underlying_pair_hash.eq(expected_pairs).all():
        raise ValueError("underlying pair hash mismatch")
    if manifest.duplicated(["judge", "bench", "underlying_pair_hash"]).any():
        raise ValueError("duplicate underlying pair in cell")
    if manifest.groupby(["bench", "underlying_pair_hash"]).input_hash.nunique().gt(1).any():
        raise ValueError("multiple shown orders for underlying pair")
    if not manifest.target.eq(manifest.pick_left.map(lambda value: "1" if int(value) else "2")).all():
        raise ValueError("target does not match historical verdict")
    if not manifest.pick_left.isin([0, 1]).all() or not manifest.seed.eq(SEED).all():
        raise ValueError("invalid historical verdict or sample seed")
    if not manifest.judge_config_hash.eq(manifest.dominant_config_hash).all():
        raise ValueError("non-dominant configuration in manifest")
    if per_cell == PER_CELL:
        for judge, group in manifest.groupby("judge"):
            expected = _current_config_record(judge)
            if any(not group[column].eq(value).all() for column, value in expected.items()):
                raise ValueError(f"current configuration snapshot mismatch for {judge}")
    if (manifest.config_count <= 0).any() or ~(manifest.config_share.between(0, 1)).all():
        raise ValueError("invalid dominant configuration accounting")
    reasoning = _reasoning_hashes(store)
    hashes = set(manifest.content_hash_l.astype(str)) | set(manifest.content_hash_r.astype(str))
    if not hashes <= reasoning:
        raise ValueError("manifest contains missing or answer-only transcript")
    source_columns = {"system_source_log_path", "system_source_sample", "system_source_epoch"}
    if source_columns <= set(manifest):
        sources = manifest.dropna(subset=list(source_columns)).drop_duplicates(list(source_columns))
        for row in sources.to_dict("records"):
            info = system_message_from_log(str(row["system_source_log_path"]),
                                           str(row["system_source_sample"]),
                                           int(row["system_source_epoch"]))
            if sha256_text(str(info["system_prompt"])) != row["system_prompt_hash"]:
                raise ValueError("unresolvable or mismatched log system message")


def hydrate_manifest(selection: pd.DataFrame, prompt_loader) -> pd.DataFrame:
    """Replace selection placeholders with exact system messages from representative Inspect logs."""
    frame = selection.copy()
    for _, group in frame.groupby(["judge", "bench", "judge_config_hash"], sort=True):
        index = group.index
        row = frame.loc[index[0]].to_dict()
        info = prompt_loader(row)
        prompt = info.get("system_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"no historical system message for {row['judge']} {row['bench']}")
        values = {
            "system_prompt": prompt, "system_prompt_hash": sha256_text(prompt),
            "historical_usage_json": _json(info.get("usage")),
            "historical_model_call_json": _json(info.get("model_call")),
            "system_source_log_path": info.get("system_source_log_path"),
            "system_source_sample": info.get("system_source_sample"),
            "system_source_epoch": info.get("system_source_epoch"),
        }
        for column, value in values.items():
            frame.loc[index, column] = value
    for judge, index in frame.groupby("judge").groups.items():
        for column, value in _current_config_record(judge).items():
            frame.loc[index, column] = value
    return frame.sort_values(["judge", "bench", "source_draw_id"]).reset_index(drop=True)


def write_manifest(manifest: pd.DataFrame, path: Path) -> None:
    """Atomically write once; a differing existing manifest is a fatal provenance error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as file:
        temporary = Path(file.name)
    try:
        manifest.to_parquet(temporary, index=False)
        if path.exists():
            if not pd.read_parquet(path).equals(pd.read_parquet(temporary)):
                raise FileExistsError(f"refusing to replace non-identical manifest: {path}")
            return
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_columns = list(dict.fromkeys((set(SOURCE_KEYS) - {"bench_source"}) |
                                      {"bench", "judge_config_hash", "judge_input", "pick_left"}))
    return (pd.read_parquet(FREEZE / "pairs.parquet", columns=pair_columns),
            pd.read_parquet(FREEZE / "transcript_store.parquet", columns=["content_hash", "transcript"]))


def _read_manifest(path: Path, judges: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, filters=[("judge", "in", judges)]) if judges else pd.read_parquet(path)


def _model_config(judge: str) -> ModelConfig:
    path = MODEL_CONFIGS[judge]
    return ModelConfig.from_cli(yaml.safe_load(path.read_text()))


def _current_config_record(judge: str) -> dict[str, str]:
    config = _model_config(judge)
    payload = {"model": config.slug, "model_args": config.model_args,
               "generate_args": config.generate_args, "runpod": config.runpod}
    return {"current_model": config.slug, "current_config_json": _json(payload),
            "current_config_hash": sha256_text(_json(payload)),
            "current_route": OPENROUTER_ROUTES.get(judge, (None, "runpod/a40"))[1]}


def fresh_eval_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Force a fresh stochastic draw while preserving every non-seed model option."""
    result = dict(kwargs)
    result.pop("seed", None)
    model_args = dict(result.get("model_args") or {})
    model_args.pop("seed", None)
    result["model_args"] = model_args
    if str(result.get("model", "")).startswith("openrouter/"):
        result["extra_headers"] = dict(result.get("extra_headers") or {}) | OPENROUTER_HEADERS
    return result


def task_from_manifest(rows: pd.DataFrame) -> Task:
    prompts = rows.system_prompt.unique()
    if len(prompts) != 1:
        raise ValueError("one Inspect task must have exactly one historical system message")
    samples = []
    metadata_columns = SOURCE_KEYS + [
        "bench", "judge_config_hash", "input_hash", "system_prompt_hash",
        "underlying_pair_hash", "seed",
    ]
    for row in rows.to_dict("records"):
        metadata = {key: _as_jsonable(row.get(key)) for key in metadata_columns}
        samples.append(Sample(id=row["source_draw_id"], input=row["judge_input"],
                              target=row["target"], metadata=metadata))
    judge, bench = rows.iloc[0][["judge", "bench"]]
    return Task(
        name=f"exact_rejudge_{judge}_{bench}_{rows.iloc[0].system_prompt_hash[:8]}",
        dataset=MemoryDataset(samples=samples),
        solver=[system_message(str(prompts[0])), generate()],
        scorer=judge_scorer(),
        metadata={"experiment": "exact_rejudge_20260828", "judge": judge, "bench": bench,
                  "system_prompt_hash": rows.iloc[0].system_prompt_hash},
    )


def _iter_logs(log_root: Path):
    for path in sorted(log_root.rglob("*.eval")) if log_root.exists() else ():
        try:
            yield path, read_eval_log(str(path))
        except Exception as error:
            raise ValueError(f"cannot read existing log {path}: {error}") from error


def terminal_sample_ids(log_root: Path) -> dict[str, str]:
    """IDs already attempted; no-answer, format failure, and uncertain paid errors are terminal."""
    terminal = {}
    for path, log in _iter_logs(log_root):
        for sample in log.samples or []:
            sample_id = str(sample.id)
            if sample_id in terminal:
                raise ValueError(f"duplicate submitted sample {sample_id}: {terminal[sample_id]} and {path}")
            terminal[sample_id] = str(path)
    return terminal


def _active_model(judge: str, llama_base_url: str | None):
    config = _model_config(judge)
    metadata = {}
    if judge == "Llama-3.1-8B-Instruct":
        if not llama_base_url:
            raise ValueError("EXACT_REJUDGE_LLAMA_BASE_URL is required; never provision an implicit fallback pod")
        model_name = config.runpod["model"]
        config = replace(
            config,
            slug=f"openai-api/runpod/{model_name}",
            model_args={"base_url": llama_base_url, "api_key": "inspectai", "stream": True},
            generate_args={"max_retries": 2},
            runpod={},
        )
        metadata = {"runpod_pod_id": os.environ.get("EXACT_REJUDGE_LLAMA_POD_ID"),
                    "runpod_gpu": "NVIDIA A40"}
    return Model(config), metadata


def run_manifest(manifest: pd.DataFrame, output_dir: Path, smoke: bool = False,
                 llama_base_url: str | None = None, judges: list[str] | None = None) -> None:
    log_root = output_dir / "logs"
    terminal = terminal_sample_ids(log_root)
    selected = manifest.copy()
    if judges:
        unknown = set(judges) - set(MODEL_CONFIGS)
        if unknown:
            raise ValueError(f"unknown judges: {sorted(unknown)}")
        selected = selected[selected.judge.isin(judges)].copy()
    if smoke:
        selected["smoke_rank"] = selected.source_draw_id.map(lambda value: seeded_rank("smoke", value))
        selected = (selected.sort_values("smoke_rank").groupby("judge", sort=False).head(5)
                    .drop(columns="smoke_rank"))
    selected = selected[~selected.source_draw_id.isin(terminal)].copy()
    print(f"{len(terminal)} terminal; {len(selected)} {'smoke' if smoke else 'full'} rows pending")
    if selected.empty:
        return
    stage = "smoke" if smoke else "full"
    for judge, judge_rows in selected.groupby("judge", sort=True):
        model, run_metadata = _active_model(judge, llama_base_url)
        with model:
            kwargs = fresh_eval_kwargs(model.eval_kwargs())
            if "seed" in kwargs or "seed" in (kwargs.get("model_args") or {}):
                raise ValueError(f"generation seed survived for {judge}")
            for (_, _, _), rows in judge_rows.groupby(
                    ["bench", "system_prompt_hash", "judge_config_hash"], sort=True):
                task = task_from_manifest(rows)
                success, _ = eval_set(
                    tasks=[task], log_dir=str(log_root / judge / stage),
                    log_dir_allow_dirty=True, epochs=1, retry_attempts=1,
                    retry_on_error=0, fail_on_error=False, score_on_error=True,
                    max_connections=min(10, len(rows)), log_model_api=True,
                    metadata=run_metadata or None, **kwargs,
                )
                print(f"{judge} {rows.iloc[0].bench}: {len(rows)} submitted; success={success}")


def _atomic_bytes(path: Path, content: bytes, replace_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace non-identical artifact: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)


def snapshot_endpoints(output_dir: Path) -> dict[str, dict]:
    snapshots = {}
    records = []
    for judge, (slug, _) in OPENROUTER_ROUTES.items():
        url = f"https://openrouter.ai/api/v1/models/{slug}/endpoints"
        path = output_dir / "endpoint_snapshots" / f"{slug.replace('/', '--')}.json"
        if path.exists():
            raw = path.read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "small-judge/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
            _atomic_bytes(path, raw)
        snapshots[judge] = json.loads(raw)
        records.append({"judge": judge, "slug": slug, "url": url,
                        "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": records}
    _atomic_bytes(output_dir / "endpoint_snapshots/manifest.json",
                  (json.dumps(manifest, indent=2) + "\n").encode(), replace_existing=True)
    return snapshots


def validate_routes(snapshots: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for judge, (_, expected) in OPENROUTER_ROUTES.items():
        endpoints = snapshots[judge].get("data", {}).get("endpoints", [])
        if expected is None:
            if endpoints:
                raise ValueError(f"{judge}: expected no OpenRouter endpoints")
            rows.append({"judge": judge, "endpoint": None, "provider": None,
                         "prompt_price": 0.0, "completion_price": 0.0})
            continue
        matches = [endpoint for endpoint in endpoints if endpoint.get("tag") == expected]
        if judge == "o4-mini":
            if len(endpoints) != 1 or str(endpoints[0].get("provider_name", "")).lower() != "openai":
                raise ValueError("o4-mini is no longer sole-served by OpenAI")
            matches = endpoints
        if len(matches) != 1 or matches[0].get("status") not in (None, 0):
            raise ValueError(f"{judge}: exact live endpoint unavailable: {expected}")
        endpoint = matches[0]
        if judge != "o4-mini":
            provider = (_model_config(judge).model_args or {}).get("provider", {})
            if provider.get("only") != [expected] or provider.get("allow_fallbacks") is not False:
                raise ValueError(f"{judge}: YAML is not exactly pinned to {expected}")
        pricing = endpoint.get("pricing") or {}
        rows.append({"judge": judge, "endpoint": expected,
                     "provider": endpoint.get("provider_name"),
                     "quantization": endpoint.get("quantization"),
                     "context_length": endpoint.get("context_length"),
                     "max_completion_tokens": endpoint.get("max_completion_tokens"),
                     "prompt_price": float(pricing.get("prompt") or 0),
                     "completion_price": float(pricing.get("completion") or 0)})
    return pd.DataFrame(rows)


def _usage(value: object) -> dict:
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def estimate_cost(manifest: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    prices = routes.set_index("judge").to_dict("index")
    rows = []
    for row in manifest.to_dict("records"):
        usage = _usage(row.get("historical_usage_json"))
        input_tokens = int(usage.get("input_tokens") or math.ceil(
            (len(row["system_prompt"]) + len(row["judge_input"])) / 3))
        output_tokens = int(usage.get("output_tokens") or 2048)
        price = prices.get(row["judge"], {})
        cost = input_tokens * price.get("prompt_price", 0) + output_tokens * price.get("completion_price", 0)
        rows.append({"judge": row["judge"], "bench": row["bench"],
                     "estimated_input_tokens": input_tokens,
                     "estimated_output_tokens": output_tokens, "estimated_cost_usd": cost})
    result = pd.DataFrame(rows).groupby(["judge", "bench"], as_index=False).sum()
    result["conservative_cost_usd"] = result.estimated_cost_usd * 1.25
    return result


def config_drift(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = manifest.drop_duplicates(["judge", "bench", "judge_config_hash"])
    for row in grouped.to_dict("records"):
        historical = read_eval_log(str(row["system_source_log_path"]), header_only=True).eval.model_dump(mode="json")
        current = _model_config(row["judge"])
        rows.append({
            "judge": row["judge"], "benchmark": row["bench"],
            "historical_config_hash": row["judge_config_hash"],
            "historical_config_count": row["config_count"],
            "historical_config_share": row["config_share"],
            "historical_model": historical.get("model"),
            "historical_model_args": _json(historical.get("model_args")),
            "historical_generate_config": _json(historical.get("model_generate_config")),
            "historical_system_prompt_hash": row["system_prompt_hash"],
            "current_model": current.slug,
            "current_model_args": _json(current.model_args),
            "current_generate_args": _json(current.generate_args),
        })
    return pd.DataFrame(rows)


def dry_run(manifest: pd.DataFrame, output_dir: Path) -> None:
    snapshots = snapshot_endpoints(output_dir)
    routes = validate_routes(snapshots)
    drift = config_drift(manifest)
    drift.to_csv(output_dir / "config_drift.csv", index=False)
    costs = estimate_cost(manifest, routes)
    terminal = terminal_sample_ids(output_dir / "logs")
    print(manifest.groupby(["judge", "bench"]).size().rename("calls").to_string())
    print("\nRoutes:\n" + routes.to_string(index=False))
    print("\nEstimated API cost (25% margin):\n" + costs.to_string(index=False))
    print(f"\nplanned={len(manifest)} terminal={len(terminal)} pending={len(manifest)-len(terminal)} ")
    print(f"conservative API total=${costs.conservative_cost_usd.sum():.4f} (RunPod billed separately)")


def _event_response(sample) -> dict:
    for event in reversed(getattr(sample, "events", None) or []):
        value = _as_jsonable(event)
        if not isinstance(value, dict):
            continue
        call = value.get("call") or {}
        response = call.get("response") if isinstance(call, dict) else None
        if isinstance(response, dict):
            return response
    return {}


def _result_rows(log_root: Path) -> dict[str, dict]:
    results = {}
    for path, log in _iter_logs(log_root):
        eval_metadata = log.eval.metadata or {}
        on_runpod = str(log.eval.model).startswith("openai-api/runpod/")
        for sample in log.samples or []:
            sample_id = str(sample.id)
            if sample_id in results:
                raise ValueError(f"duplicate output for {sample_id}")
            score = (sample.scores or {}).get("judge_scorer")
            if score is None and sample.scores:
                score = next(iter(sample.scores.values()))
            answer = getattr(score, "answer", None)
            metadata = getattr(score, "metadata", None) or {}
            response = _event_response(sample)
            events = [_as_jsonable(event) for event in (sample.events or [])]
            model_events = [value for value in events
                            if isinstance(value, dict) and value.get("event") == "model"]
            paid_generation_count = sum(
                bool((((value.get("call") or {}).get("response") or {}).get("id")))
                for value in model_events
            )
            router = response.get("openrouter_metadata") or {}
            available = ((router.get("endpoints") or {}).get("available") or []) if isinstance(router, dict) else []
            selected = next((endpoint for endpoint in available if endpoint.get("selected")), {})
            raw_usage = response.get("usage") or {}
            output_usage = _as_jsonable(getattr(getattr(sample, "output", None), "usage", None)) or {}
            usage = raw_usage | output_usage if isinstance(raw_usage, dict) and isinstance(output_usage, dict) else output_usage
            results[sample_id] = {
                "repeated_verdict": answer,
                "confidence": metadata.get("confidence"),
                "explanation": getattr(score, "explanation", None),
                "status": "error" if getattr(sample, "error", None) else "answered" if answer else "no_answer",
                "error": str(getattr(sample, "error", None) or "") or None,
                "provider": "RunPod" if on_runpod else selected.get("provider") or response.get("provider"),
                "endpoint": eval_metadata.get("runpod_pod_id") if on_runpod else selected.get("name") or selected.get("model"),
                "runpod_gpu": eval_metadata.get("runpod_gpu"),
                "provider_metadata_json": _json(router) if router else None,
                "generation_id": response.get("id"),
                "model_call_count": len(model_events),
                "paid_generation_count": paid_generation_count,
                "system_fingerprint": response.get("system_fingerprint"),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cached_input_tokens": usage.get("input_tokens_cache_read", 0),
                "cost_usd": raw_usage.get("cost") if isinstance(raw_usage, dict) else None,
                "log_path": str(path),
            }
    return results


def flatten(manifest: pd.DataFrame, log_root: Path) -> pd.DataFrame:
    results = _result_rows(log_root)
    rows = []
    for row in manifest.to_dict("records"):
        result = results.get(row["source_draw_id"], {"status": "missing", "repeated_verdict": None})
        repeated = result.get("repeated_verdict")
        rows.append(row | result | {
            "original_verdict": row["target"],
            "answered": repeated in ("1", "2"),
            "agreement": repeated == row["target"] if repeated in ("1", "2") else False,
            "configured_endpoint": OPENROUTER_ROUTES.get(row["judge"], (None, "runpod/a40"))[1],
        })
    return pd.DataFrame(rows)


def _wilson(successes: int, n: int) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.975)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - margin, centre + margin


def _bayes_upper(agreement: float) -> float:
    return 0.5 + math.sqrt((agreement - 0.5) / 2) if agreement >= 0.5 else math.nan


def summaries(rejudgments: pd.DataFrame, seed: int = SEED,
              bootstrap_replicates: int = 10_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = rejudgments.copy()
    frame["answered"] = frame.repeated_verdict.isin(["1", "2"])
    frame["agreement"] = frame.answered & frame.repeated_verdict.eq(frame.target)
    rows = []
    for (judge, bench), group in frame.groupby(["judge", "bench"], sort=True):
        answered = group[group.answered]
        agreement = float(group.agreement.mean())
        low, high = _wilson(int(group.agreement.sum()), len(group))
        rows.append({
            "judge": judge, "benchmark": bench, "n": len(group),
            "answer_rate": float(group.answered.mean()), "raw_agreement": agreement,
            "answered_only_agreement": float(answered.agreement.mean()) if len(answered) else math.nan,
            "wilson_95_low": low, "wilson_95_high": high,
            "transition_1_to_2": int(((answered.target == "1") & (answered.repeated_verdict == "2")).sum()),
            "transition_2_to_1": int(((answered.target == "2") & (answered.repeated_verdict == "1")).sum()),
            "bayes_accuracy_lower": agreement, "bayes_accuracy_upper": _bayes_upper(agreement),
        })
    summary = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    judge_rows = []
    for judge, group in frame.groupby("judge", sort=True):
        benches = [bench_group.agreement.to_numpy(dtype=float)
                   for _, bench_group in group.groupby("bench", sort=True)]
        observed = float(np.mean([values.mean() for values in benches]))
        draws = np.empty(bootstrap_replicates)
        for index in range(bootstrap_replicates):
            draws[index] = np.mean([rng.choice(values, len(values), replace=True).mean()
                                    for values in benches])
        judge_rows.append({
            "judge": judge, "benchmarks": len(benches), "n": len(group),
            "macro_agreement": observed,
            "bootstrap_95_low": float(np.quantile(draws, 0.025)),
            "bootstrap_95_high": float(np.quantile(draws, 0.975)),
            "bayes_accuracy_lower": observed, "bayes_accuracy_upper": _bayes_upper(observed),
        })
    return summary, pd.DataFrame(judge_rows)


def write_results(rejudgments: pd.DataFrame, output_dir: Path) -> None:
    if len(rejudgments) != PER_CELL * len(EXPECTED_CELLS) or rejudgments.source_draw_id.duplicated().any():
        raise ValueError("results do not match the 2,100-row manifest")
    if rejudgments.paid_generation_count.fillna(0).gt(1).any():
        raise ValueError("more than one paid generation for a manifest row")
    summary, by_judge = summaries(rejudgments)
    parquet = output_dir / "rejudgments.parquet"
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    rejudgments.to_parquet(temporary, index=False)
    temporary.replace(parquet)
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_judge.to_csv(output_dir / "summary_by_judge.csv", index=False)
    cost = (rejudgments.groupby(["judge", "provider"], dropna=False, as_index=False)
            .agg(n=("source_draw_id", "size"), input_tokens=("input_tokens", "sum"),
                 cached_input_tokens=("cached_input_tokens", "sum"),
                 output_tokens=("output_tokens", "sum"), cost_usd=("cost_usd", "sum"),
                 request_errors=("status", lambda values: int((values == "error").sum())),
                 no_answers=("status", lambda values: int((values == "no_answer").sum()))))
    cost["gpu_seconds"] = 0.0
    billing_path = output_dir / "runpod_billing.csv"
    runpod_cost = 0.0
    if billing_path.exists():
        billing = pd.read_csv(billing_path)
        runpod_cost = float(billing["amount"].sum())
        billing_rows = pd.DataFrame({
            "judge": billing["role"], "provider": "RunPod", "n": 0,
            "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
            "request_errors": 0, "no_answers": 0,
            "cost_usd": billing["amount"], "gpu_seconds": billing["time_billed_ms"] / 1000,
        })
        cost = pd.concat([cost, billing_rows], ignore_index=True)
    cost.to_csv(output_dir / "cost.csv", index=False)
    counts = rejudgments.status.value_counts()
    readme = f"""# Historical-to-current exact-input agreement

This experiment repeats each frozen shown transcript pair and order once. It is not a mathematical ceiling.

## Source and controls

- Freeze: `{FREEZE}`.
- `judge_input` is reused verbatim. The system message is verified from one authoritative HF Inspect log per dominant `judge_config_hash`, whose hash includes the prompt, then applied to every selected row with that configuration.
- The current `judge.dataset.has_reasoning` exclusion is applied before sampling; it removed 668 candidate pair rows in the six-judge scope.
- AIME 2024-26 is pooled; 100 distinct underlying pairs are selected per available judge x benchmark cell.
- Seed `{SEED}` is used only for sampling and bootstrap CIs. Model-generation seeds are removed.
- OpenRouter response replay caching is disabled with `X-OpenRouter-Cache: false`; router metadata is enabled.
- Pins: DeepInfra BF16 (gpt-oss), SiliconFlow FP8 (R1), sole OpenAI (o4-mini), Novita BF16 (Gemma), DeepInfra FP8 (V4), secure A40 (Llama). Fallbacks are disabled for multi-endpoint routes.
- Ling is excluded because its historical OpenRouter endpoint disappeared.

## Commands

```bash
python scripts/verdict_dissection/exact_rejudge.py select
python scripts/verdict_dissection/exact_rejudge.py hydrate
python scripts/verdict_dissection/exact_rejudge.py dry-run
python scripts/verdict_dissection/exact_rejudge.py run --smoke --judge JUDGE
python scripts/verdict_dissection/exact_rejudge.py run --judge JUDGE
python scripts/verdict_dissection/exact_rejudge.py flatten
```

The six judge-specific run commands were executed in parallel. Smoke rows are terminal full-sample rows and were not called again.

## Completion and failures

- Attempted: {len(rejudgments)} / {len(rejudgments)}.
- Answered: {int(rejudgments.answered.sum())}.
- No answer: {int(counts.get('no_answer', 0))}.
- Request errors: {int(counts.get('error', 0))}.
- Primary agreements: {int(rejudgments.agreement.sum())}.
- OpenRouter API cost: ${rejudgments.cost_usd.sum():.11f}.
- RunPod cost: ${runpod_cost:.11f}.
- Exact total paid cost: ${rejudgments.cost_usd.sum() + runpod_cost:.11f}.

DeepSeek R1's current `max_tokens=147456` leaves only 16,384 tokens for input on its 163,840-token SiliconFlow endpoint. This caused long SWE inputs to fail with HTTP 400; those rows were not retried and count as primary non-agreement.

## Provenance and caveats

Raw endpoint snapshots are in `endpoint_snapshots/`; all historical configuration counts are in `config_population.csv`; historical/current arguments are in `config_drift.csv`; per-response route metadata is in `rejudgments.parquet`.

Historical configurations differ from current routes for several cells, including provider, fallback, quantization, output cap, and vLLM backend. The estimate therefore includes temporal/provider drift as well as stochastic judge variation. Llama's streaming vLLM responses did not expose token counts; exact A40 billed time/cost is reported instead.
"""
    _atomic_bytes(output_dir / "README.md", readme.encode(), replace_existing=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "hydrate", "manifest", "validate", "dry-run", "run", "flatten"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--llama-base-url", default=os.environ.get("EXACT_REJUDGE_LLAMA_BASE_URL"))
    parser.add_argument("--judge", action="append", choices=sorted(MODEL_CONFIGS))
    args = parser.parse_args()
    path = args.output_dir / "sample_manifest.parquet"
    if args.command == "select":
        pairs, store = _frozen_inputs()
        selection = build_manifest(pairs, store, lambda row: {"system_prompt": "__PENDING__"})
        selection_path = args.output_dir / "sample_selection.parquet"
        write_manifest(selection, selection_path)
        sources = selection.drop_duplicates(["judge", "bench", "judge_config_hash"])[
            ["judge", "bench", "bench_source", "judge_config_hash", "judge_log", "judge_sample", "judge_epoch"]]
        sources.to_csv(args.output_dir / "prompt_sources.csv", index=False)
        print(f"{len(selection)} selected rows -> {selection_path}")
    elif args.command == "hydrate":
        _, store = _frozen_inputs()
        selection = pd.read_parquet(args.output_dir / "sample_selection.parquet")
        manifest = hydrate_manifest(selection, _historical_prompt)
        validate_manifest(manifest, store)
        write_manifest(manifest, path)
        print(f"{len(manifest)} hydrated rows -> {path}")
    elif args.command == "manifest":
        pairs, store = _frozen_inputs()
        manifest = build_manifest(pairs, store, _historical_prompt)
        validate_manifest(manifest, store)
        write_manifest(manifest, path)
        print(f"{len(manifest)} rows -> {path}")
    elif args.command == "validate":
        _, store = _frozen_inputs()
        validate_manifest(pd.read_parquet(path), store)
        print(f"valid: {path}")
    elif args.command == "dry-run":
        dry_run(pd.read_parquet(path), args.output_dir)
    elif args.command == "run":
        run_manifest(_read_manifest(path, args.judge), args.output_dir, args.smoke,
                     args.llama_base_url, args.judge)
    else:
        results = flatten(pd.read_parquet(path), args.output_dir / "logs")
        write_results(results, args.output_dir)
        print(f"{(results.status != 'missing').sum()} terminal rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
