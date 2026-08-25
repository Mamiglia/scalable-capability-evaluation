"""Minimal Top-K transcript SAE for frozen Workstream B embeddings."""

from __future__ import annotations

import argparse
import math
import json
import os
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch import nn

from transcript_embeddings import (atomic_artifact_directory, git_commit, json_ready,
                                   ordered_key_hash, sha256_file, verify_artifact_files)


def widths(n_train: int, input_dim: int) -> tuple[int, int, int] | None:
    limit = min(256, input_dim, n_train // 50)
    if limit < 1:
        return None
    wide = 1 << int(math.log2(limit))
    return None if wide // 4 < 16 else (wide // 4, wide // 2, wide)


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, width: int, k: int, mean: torch.Tensor):
        super().__init__()
        if not 0 < k <= width:
            raise ValueError("Top-K must be between one and dictionary width")
        if tuple(mean.shape) != (input_dim,):
            raise ValueError("training mean has the wrong dimension")
        self.k = k
        self.register_buffer("mean", mean.detach().float().clone())
        self.encoder = nn.Linear(input_dim, width)
        self.decoder = nn.Linear(width, input_dim, bias=False)
        self.normalise_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    @torch.no_grad()
    def normalise_decoder(self):
        self.decoder.weight.copy_(torch.nn.functional.normalize(self.decoder.weight, dim=0))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.encoder(inputs.float() - self.mean))
        top_values, indices = torch.topk(values, self.k, dim=1)
        return torch.zeros_like(values).scatter(1, indices, top_values)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.mean + self.decoder(code)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

    def reconstruct_with_residual(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction = self(inputs)
        return reconstruction, inputs - reconstruction


def _normalise_columns(dictionary: np.ndarray) -> np.ndarray:
    dictionary = np.asarray(dictionary, dtype=np.float64)
    norms = np.linalg.norm(dictionary, axis=0)
    if dictionary.ndim != 2 or np.any(norms == 0) or not np.isfinite(dictionary).all():
        raise ValueError("decoder columns must be finite and nonzero")
    return dictionary / norms


def decoder_matches(canonical: np.ndarray, alternatives: list[np.ndarray]) -> np.ndarray:
    canonical = _normalise_columns(canonical)
    matches = np.zeros((canonical.shape[1], len(alternatives)))
    for column, alternative in enumerate(alternatives):
        cosine = np.maximum(canonical.T @ _normalise_columns(alternative), 0.0)
        rows, cols = linear_sum_assignment(-cosine)
        matches[rows, column] = cosine[rows, cols]
    return matches


def retained_features(matches: np.ndarray, train_activations: np.ndarray,
                      *, min_cosine: float = 0.8, min_fits: int = 3,
                      min_support: int = 50) -> np.ndarray:
    matches, train_activations = np.asarray(matches), np.asarray(train_activations)
    if matches.shape[0] != train_activations.shape[1]:
        raise ValueError("match and activation feature counts differ")
    stable = (matches >= min_cosine).sum(axis=1) >= min_fits
    supported = (train_activations > 0).sum(axis=0) >= min_support
    return np.flatnonzero(stable & supported)


def bootstrap_train_indices(rows: pd.DataFrame, seed: int) -> np.ndarray:
    if not {"benchmark", "item_group_key"} <= set(rows):
        raise ValueError("bootstrap rows require benchmark and opaque item-group key")
    rng, sampled = np.random.default_rng(seed), []
    for _, group in rows.groupby("benchmark", sort=True):
        items = sorted(group["item_group_key"].astype(str).unique())
        for item in rng.choice(items, size=len(items), replace=True):
            sampled.extend(group.index[
                group["item_group_key"].astype(str).eq(item)].tolist())
    return np.asarray(sampled, dtype=np.int64)


@torch.inference_mode()
def reconstruct_retained(sae: TopKSAE, inputs: np.ndarray,
                         retained: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(sae.parameters()).device
    values = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    full_code = sae.encode(values)
    retained = np.asarray(retained, dtype=np.int64)
    code = full_code[:, retained]
    reconstruction = sae.mean + code @ sae.decoder.weight[:, retained].T
    return (code.cpu().numpy(), reconstruction.cpu().numpy(),
            (values - reconstruction).cpu().numpy())


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(len(left), np.nan)
    valid = denominator > 0
    result[valid] = (left[valid] * right[valid]).sum(axis=1) / denominator[valid]
    return result


def _median_cosine(left: np.ndarray, right: np.ndarray) -> float:
    values = _row_cosine(left, right)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def reconstruction_metrics(inputs: np.ndarray, reconstructions: np.ndarray,
                           mean: np.ndarray) -> dict[str, float]:
    inputs = np.asarray(inputs, dtype=np.float64)
    reconstructions = np.asarray(reconstructions, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if inputs.ndim != 2 or reconstructions.shape != inputs.shape:
        raise ValueError("inputs and reconstructions must have the same 2D shape")
    if mean.shape != (inputs.shape[1],):
        raise ValueError("training mean has the wrong dimension")
    centred, reconstructed_centred = inputs - mean, reconstructions - mean
    sst = np.square(centred).sum()
    if sst == 0:
        raise ValueError("validation inputs have zero variance around the training mean")
    return {
        "mse": float(np.square(inputs - reconstructions).mean()),
        "explained_variance": float(1 - np.square(inputs - reconstructions).sum() / sst),
        "median_centered_cosine": _median_cosine(centred, reconstructed_centred),
        "median_uncentered_cosine": _median_cosine(inputs, reconstructions),
        "median_mean_only_cosine": _median_cosine(inputs, np.broadcast_to(mean, inputs.shape)),
    }


def random_dictionary(input_dim: int, width: int, seed: int = 0) -> np.ndarray:
    decoder = np.random.default_rng(seed).standard_normal((input_dim, width))
    return _normalise_columns(decoder)


def fit_sae(train: np.ndarray, validation: np.ndarray, *, width: int, k: int,
            seed: int, batch_size: int = 512, max_epochs: int = 200,
            patience: int = 20, learning_rate: float = 1e-3,
            device: str | None = None,
            mean: np.ndarray | None = None) -> tuple[TopKSAE, dict]:
    train = np.asarray(train, dtype=np.float32)
    validation = np.asarray(validation, dtype=np.float32)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("train and validation must be 2D with the same input dimension")
    if not len(train) or not len(validation):
        raise ValueError("train and validation must be nonempty")

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.from_numpy(train)
    validation_tensor = torch.from_numpy(validation).to(device)
    frozen_mean = train_tensor.mean(0) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
    model = TopKSAE(train.shape[1], width, k, frozen_mean).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale = math.inf, -1, 0
    best_model = None
    best_optimiser = None

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_tensor), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = train_tensor[order[start:start + batch_size]].to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            optimiser.step()
            model.normalise_decoder()

        model.eval()
        with torch.inference_mode():
            validation_loss = torch.nn.functional.mse_loss(
                model(validation_tensor), validation_tensor).item()
        if validation_loss < best_loss - 1e-12:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_model = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            best_optimiser = deepcopy(optimiser.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_model)
    return model, {
        "best_epoch": best_epoch,
        "validation_mse": best_loss,
        "epochs_run": epoch + 1,
        "optimizer_state": best_optimiser,
    }


def _fit_summary(state: dict) -> dict:
    return {key: state[key] for key in ("best_epoch", "validation_mse", "epochs_run")}


def _random_reconstruction(inputs: np.ndarray, mean: np.ndarray,
                           decoder: np.ndarray, k: int) -> np.ndarray:
    scores = np.maximum((inputs - mean) @ decoder, 0)
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    code = np.zeros_like(scores)
    np.put_along_axis(code, indices, np.take_along_axis(scores, indices, axis=1), axis=1)
    return mean + code @ decoder.T


def _bootstrap_reconstruction_gate(inputs: np.ndarray, retained: np.ndarray,
                                   random: np.ndarray, mean: np.ndarray,
                                   rows: pd.DataFrame, replicates: int) -> dict:
    explained, cosine_differences = [], []
    rows = rows[["benchmark", "item_group_key"]].reset_index(drop=True)
    for replicate in range(replicates):
        indices = bootstrap_train_indices(rows, seed=replicate)
        retained_metrics = reconstruction_metrics(inputs[indices], retained[indices], mean)
        random_metrics = reconstruction_metrics(inputs[indices], random[indices], mean)
        explained.append(retained_metrics["explained_variance"])
        cosine_differences.append(
            retained_metrics["median_centered_cosine"]
            - random_metrics["median_centered_cosine"])
    return {
        "replicates": replicates,
        "explained_variance_interval": np.quantile(explained, [0.025, 0.975]).tolist(),
        "centered_cosine_difference_interval": np.quantile(
            cosine_differences, [0.025, 0.975]).tolist(),
    }


def run_sae_suite(inputs: np.ndarray, rows: pd.DataFrame, config: dict,
                  *, device: str | None = None
                  ) -> tuple[TopKSAE | None, dict, dict, dict | None]:
    """Fit narrow/primary/wide plus four noncanonical primary fits, development only."""
    inputs = np.asarray(inputs, dtype=np.float32)
    if len(inputs) != len(rows) or "split_role" not in rows:
        raise ValueError("embedding rows and split metadata do not align")
    if (rows["split_role"] == "confirmation").any():
        raise ValueError("confirmation rows cannot enter SAE fitting")
    train_mask = rows["split_role"].eq("representation_train").to_numpy()
    validation_mask = rows["split_role"].eq("representation_validation").to_numpy()
    train, validation = inputs[train_mask], inputs[validation_mask]
    realised = widths(len(train), inputs.shape[1])
    if realised is None:
        return None, {}, {"status": "null", "reason": "infeasible_width",
                          "n_train": len(train), "n_validation": len(validation)}, None
    narrow, primary, wide = realised
    frozen_mean = train.mean(axis=0)
    fit_kwargs = dict(
        k=config["top_k"], batch_size=config["batch_size"],
        max_epochs=config["max_epochs"], patience=config["patience"],
        learning_rate=config["learning_rate"], device=device,
        mean=frozen_mean,
    )

    if config.get("seeds", [0, 1, 2, 3, 4]) != [0, 1, 2, 3, 4]:
        raise ValueError("SAE seeds must be exactly [0, 1, 2, 3, 4]")
    canonical, canonical_state = fit_sae(
        train, validation, width=primary, seed=0, **fit_kwargs)
    reseed, reseed_state = fit_sae(
        train, validation, width=primary, seed=1, **fit_kwargs)
    alternatives, alternative_states = [reseed], [reseed_state]
    train_rows = rows.loc[
        train_mask, ["benchmark", "item_group_key"]].reset_index(drop=True)
    for seed in (2, 3, 4):
        sample = bootstrap_train_indices(train_rows, seed)
        model, state = fit_sae(
            train[sample], validation, width=primary, seed=seed, **fit_kwargs)
        alternatives.append(model)
        alternative_states.append(state)

    narrow_model, narrow_state = fit_sae(
        train, validation, width=narrow, seed=0, **fit_kwargs)
    wide_model, wide_state = fit_sae(
        train, validation, width=wide, seed=0, **fit_kwargs)
    canonical_decoder = canonical.decoder.weight.detach().cpu().numpy()
    alternative_decoders = [model.decoder.weight.detach().cpu().numpy()
                            for model in alternatives]
    matches = decoder_matches(canonical_decoder, alternative_decoders)
    full_code, full_reconstructions, _ = reconstruct_retained(
        canonical, inputs, np.arange(primary))
    train_code = full_code[train_mask]
    retained = retained_features(
        matches, train_code,
        min_cosine=config["min_decoder_cosine"],
        min_fits=config["min_noncanonical_matches"],
        min_support=config["min_training_support"],
    )
    activations, reconstructions, residuals = reconstruct_retained(canonical, inputs, retained)
    _, narrow_reconstructions, _ = reconstruct_retained(
        narrow_model, inputs, np.arange(narrow))
    _, wide_reconstructions, _ = reconstruct_retained(
        wide_model, inputs, np.arange(wide))

    mean = canonical.mean.detach().cpu().numpy()
    random_decoder = random_dictionary(inputs.shape[1], primary, seed=0)
    random_validation = _random_reconstruction(validation, mean, random_decoder, config["top_k"])
    retained_validation = reconstruction_metrics(
        validation, reconstructions[validation_mask], mean)
    random_metrics = reconstruction_metrics(validation, random_validation, mean)
    bootstrap = (_bootstrap_reconstruction_gate(
        validation, reconstructions[validation_mask], random_validation, mean,
        rows.loc[validation_mask], config["bootstrap_replicates"])
        if len(retained) else None)
    active_fraction = float((activations[validation_mask] > 0).any(axis=1).mean())
    if not len(retained):
        gate_status, gate_reason = "null", "zero_retained_features"
    else:
        passed = (
            bootstrap["explained_variance_interval"][0] > 0
            and retained_validation["explained_variance"]
            - random_metrics["explained_variance"] >= config["min_explained_variance_gain"]
            and retained_validation["median_centered_cosine"]
            - random_metrics["median_centered_cosine"] >= config["min_centered_cosine_gain"]
            and bootstrap["centered_cosine_difference_interval"][0] > 0
            and active_fraction >= config["min_validation_active_fraction"])
        gate_status, gate_reason = (("passed", None) if passed
                                    else ("null", "reconstruction_gate_failed"))
    report = {
        "status": gate_status,
        "reason": gate_reason,
        "n_train": len(train),
        "n_validation": len(validation),
        "widths": {"narrow": narrow, "primary": primary, "wide": wide},
        "training_transcripts_per_element": len(train) / primary,
        "retained_features": retained.tolist(),
        "retained_support": (train_code[:, retained] > 0).sum(0).tolist(),
        "full_validation": reconstruction_metrics(
            validation, full_reconstructions[validation_mask], mean),
        "retained_validation": retained_validation,
        "random_validation": random_metrics,
        "narrow_validation": reconstruction_metrics(
            validation, narrow_reconstructions[validation_mask], mean),
        "wide_validation": reconstruction_metrics(
            validation, wide_reconstructions[validation_mask], mean),
        "retained_validation_active_fraction": active_fraction,
        "bootstrap": bootstrap,
        "fits": {"canonical": _fit_summary(canonical_state),
                 "same_data_reseed": _fit_summary(reseed_state),
                 "bootstraps": [_fit_summary(state) for state in alternative_states[1:]],
                 "narrow": _fit_summary(narrow_state), "wide": _fit_summary(wide_state)},
    }
    arrays = {
        "activations": activations,
        "reconstructions": reconstructions,
        "residuals": residuals,
        "retained_feature_ids": retained,
        "match_cosines": matches,
        "canonical_decoder": canonical_decoder,
        "alternative_decoders": np.stack(alternative_decoders),
        "narrow_decoder": narrow_model.decoder.weight.detach().cpu().numpy(),
        "wide_decoder": wide_model.decoder.weight.detach().cpu().numpy(),
        "narrow_encoder_weight": narrow_model.encoder.weight.detach().cpu().numpy(),
        "narrow_encoder_bias": narrow_model.encoder.bias.detach().cpu().numpy(),
        "wide_encoder_weight": wide_model.encoder.weight.detach().cpu().numpy(),
        "wide_encoder_bias": wide_model.encoder.bias.detach().cpu().numpy(),
        "random_decoder": random_decoder,
    }
    return canonical, arrays, report, canonical_state


def write_sae_artifacts(output_dir: str | Path, sae: TopKSAE | None,
                        arrays: dict[str, np.ndarray], manifest: dict,
                        *, rows: pd.DataFrame | None = None,
    training_state: dict | None = None) -> dict:
    def build(stage: Path) -> dict:
        report_path = stage / "sae_report.json"
        report_path.write_text(json.dumps(
            json_ready(manifest.get("report", {})), sort_keys=True, allow_nan=False) + "\n")
        outputs = {report_path.name: sha256_file(report_path)}
        if sae is not None:
            weights = stage / "sae_weights.npz"
            np.savez(
                weights,
                mean=sae.mean.detach().cpu().numpy(),
                encoder_weight=sae.encoder.weight.detach().cpu().numpy(),
                encoder_bias=sae.encoder.bias.detach().cpu().numpy(),
                decoder_weight=sae.decoder.weight.detach().cpu().numpy(),
                k=np.array(sae.k),
            )
            outputs[weights.name] = sha256_file(weights)
        for name, values in arrays.items():
            path = stage / f"{name}.npy"
            values = np.asarray(values)
            np.save(path, values.astype(np.float32) if np.issubdtype(values.dtype, np.floating)
                    else values, allow_pickle=False)
            outputs[path.name] = sha256_file(path)
        if rows is not None and "activations" in arrays:
            row_index, feature = np.nonzero(arrays["activations"] > 0)
            feature_ids = np.asarray(arrays.get(
                "retained_feature_ids", np.arange(arrays["activations"].shape[1])))
            if len(feature_ids) != arrays["activations"].shape[1]:
                raise ValueError("retained feature IDs do not align with activations")
            sparse = pd.DataFrame({"row_index": row_index, "feature": feature_ids[feature],
                                   "value": arrays["activations"][row_index, feature]})
            sparse_path = stage / "activations_sparse.parquet"
            sparse.to_parquet(sparse_path, index=False)
            rows_path = stage / "rows.parquet"
            rows.to_parquet(rows_path, index=False)
            outputs[sparse_path.name] = sha256_file(sparse_path)
            outputs[rows_path.name] = sha256_file(rows_path)
        if training_state is not None:
            state_path = stage / "optimizer_state.pt"
            torch.save(training_state["optimizer_state"], state_path)
            outputs[state_path.name] = sha256_file(state_path)
        return dict(manifest) | {"outputs": outputs}

    return atomic_artifact_directory(output_dir, build)


def load_sae(path: str | Path) -> TopKSAE:
    weights = np.load(path, allow_pickle=False)
    mean = torch.from_numpy(weights["mean"])
    encoder_weight = torch.from_numpy(weights["encoder_weight"])
    sae = TopKSAE(len(mean), len(encoder_weight), int(weights["k"]), mean)
    with torch.no_grad():
        sae.encoder.weight.copy_(encoder_weight)
        sae.encoder.bias.copy_(torch.from_numpy(weights["encoder_bias"]))
        sae.decoder.weight.copy_(torch.from_numpy(weights["decoder_weight"]))
    return sae.eval()


def load_dense_artifacts(dense_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    dense_dir = Path(dense_dir)
    manifest = json.loads((dense_dir / "manifest.json").read_text())
    verify_artifact_files(dense_dir, manifest)
    required = {"rows.parquet", "transcript_only__final.npy"}
    if not required <= set(manifest["outputs"]):
        raise ValueError("dense manifest is missing SAE input artifacts")
    rows = pd.read_parquet(dense_dir / "rows.parquet")
    inputs = np.load(dense_dir / "transcript_only__final.npy", allow_pickle=False)
    if not {"transcript_key", "split_role"} <= set(rows):
        raise ValueError("dense rows are missing transcript_key or split_role")
    if {"model", "bench", "item", "epoch", "log", "sample_epoch", "content_hash"} & set(rows):
        raise ValueError("dense learner rows contain forbidden provenance columns")
    if rows["transcript_key"].duplicated().any():
        raise ValueError("duplicate dense transcript_key")
    if inputs.ndim != 2 or len(inputs) != len(rows) or not np.isfinite(inputs).all():
        raise ValueError("dense SAE inputs must be aligned, finite, and 2D")
    return rows, inputs, manifest


def load_development_inputs(dense_dir: str | Path, sae_dir: str | Path):
    dense_dir, sae_dir = Path(dense_dir), Path(sae_dir)
    dense_rows, _, dense_manifest = load_dense_artifacts(dense_dir)
    if "report_rows.parquet" not in dense_manifest["outputs"]:
        raise ValueError("dense artifact is missing the outcome-free report table")
    report_rows = pd.read_parquet(dense_dir / "report_rows.parquet")
    forbidden = {"model", "bench", "item", "epoch", "log", "sample_epoch",
                 "content_hash", "correct", "is_canonical", "judge", "verdict", "capability"}
    if forbidden & set(report_rows):
        raise ValueError("report table contains forbidden provenance or outcomes")
    dense_hash = ordered_key_hash(dense_rows["transcript_key"])
    report_hash = ordered_key_hash(report_rows["transcript_key"])
    if (dense_manifest.get("row_order_hash") != dense_hash
            or dense_manifest.get("report_row_order_hash") != report_hash
            or dense_hash != report_hash):
        raise ValueError("dense/report transcript-key row order does not match")

    sae_manifest = json.loads((sae_dir / "manifest.json").read_text())
    verify_artifact_files(sae_dir, sae_manifest)
    if sae_manifest.get("report", {}).get("status") != "passed":
        raise ValueError("development report requires a passed Qwen SAE")
    if sae_manifest.get("dense_manifest_sha256") != sha256_file(dense_dir / "manifest.json"):
        raise ValueError("Qwen SAE source dense-manifest hash does not match")
    sae_rows = pd.read_parquet(sae_dir / "rows.parquet")
    if ordered_key_hash(sae_rows["transcript_key"]) != dense_hash:
        raise ValueError("Qwen SAE transcript-key row order does not match dense rows")
    activations = np.load(sae_dir / "activations.npy", allow_pickle=False)
    feature_ids = np.load(sae_dir / "retained_feature_ids.npy", allow_pickle=False).astype(int)
    if len(activations) != len(report_rows) or activations.shape[1] != len(feature_ids):
        raise ValueError("Qwen SAE activations/features do not align with report rows")
    dense_views = {path.stem: np.load(path, allow_pickle=False)
                   for path in sorted(dense_dir.glob("*.npy"))}
    return report_rows, activations, dense_views, feature_ids, dense_manifest, sae_manifest


def dense_identity(manifest: dict) -> dict:
    fields = (
        "family", "model", "model_files", "config_sha256", "phase1_freeze_sha256",
        "embedding_code_sha256", "git_commit", "extraction", "runtime",
    )
    if missing := [field for field in fields if field not in manifest]:
        raise ValueError(f"dense manifest is missing identity fields: {missing}")
    return {field: manifest[field] for field in fields}


def _grouped_r2(features: np.ndarray, targets: np.ndarray, groups: pd.Series) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    unique = groups.nunique()
    if unique < 2 or not features.shape[1]:
        return math.nan
    predictions = np.zeros_like(targets, dtype=float)
    for train, test in GroupKFold(min(5, unique)).split(features, groups=groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features[train], targets[train])
        predictions[test] = model.predict(features[test])
    return float(r2_score(targets, predictions, multioutput="variance_weighted"))


def development_report(rows: pd.DataFrame, activations: np.ndarray,
                       dense_views: dict[str, np.ndarray], a1: pd.DataFrame,
                       a2_by_scorer: dict[str, pd.DataFrame],
                       *, bge_activations: np.ndarray | None = None,
                       feature_ids: np.ndarray | None = None,
                       bge_feature_ids: np.ndarray | None = None,
                       cross_family_mask: np.ndarray | None = None,
                       audit_map: pd.DataFrame | None = None) -> dict:
    """Outcome-blind atlas, shortcut audit, and optional cross-family matching."""
    forbidden = ("correct", "canonical", "capability", "verdict", "judge", "model", "pick_left")
    for frame in (a1, *a2_by_scorer.values()):
        for column in frame:
            if column != "a_judge_truncated" and any(
                    name in column.lower() for name in forbidden):
                raise ValueError(f"forbidden learner column: {column}")
    if "a_judge_truncated" not in a1:
        raise ValueError("A1 requires the allowed a_judge_truncated flag")
    required = {"transcript_key", "item_group_key", "benchmark", "transcript"}
    if not required <= set(rows) or len(rows) != len(activations):
        raise ValueError("development rows and activations do not align")
    if rows["transcript_key"].duplicated().any():
        raise ValueError("development transcript keys must be unique")
    if audit_map is None:
        raise ValueError("development report requires the post-freeze audit map")
    if set(a2_by_scorer) != {"llama", "qwen", "gemma"}:
        raise ValueError("development report requires all three frozen A2 scorers")
    if not any("missing" in column.lower() for column in a1.columns):
        raise ValueError("A1 requires explicit missingness indicators")
    expected_keys = set(rows["transcript_key"])
    for name, frame in (("a1", a1), *sorted(a2_by_scorer.items())):
        if ("transcript_key" not in frame or frame["transcript_key"].duplicated().any()
                or set(frame["transcript_key"]) != expected_keys):
            raise ValueError(f"required {name} inputs are unavailable or misaligned")
        if name != "a1" and ("coverage_complete" not in frame
                              or frame["coverage_complete"].isna().any()):
            raise ValueError(f"{name} requires an explicit coverage_complete indicator")
        numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
        values = [column for column in numeric
                  if "missing" not in column.lower() and column != "coverage_complete"]
        if name == "a1":
            indicators = [column for column in numeric if "missing" in column.lower()]
            unflagged = frame[values].isna().any(axis=1) & ~frame[indicators].astype(bool).any(axis=1)
            if unflagged.any():
                raise ValueError("A1 missing values require explicit missingness indicators")
        elif frame.loc[frame["coverage_complete"].eq(True), values].isna().any().any():
            raise ValueError(f"{name} complete-coverage rows contain missing scorer values")

    feature_ids = (np.arange(activations.shape[1]) if feature_ids is None
                   else np.asarray(feature_ids, dtype=int))
    if len(feature_ids) != activations.shape[1]:
        raise ValueError("canonical SAE feature IDs do not align with activations")
    atlas_rows = []
    for feature in range(activations.shape[1]):
        order = np.argsort(-activations[:, feature], kind="stable")
        unique, remainder, seen = [], [], set()
        for index in order:
            item = rows.iloc[index].item_group_key
            (unique if item not in seen else remainder).append(index)
            seen.add(item)
        for rank, index in enumerate((unique + remainder)[:20], start=1):
            atlas_rows.append({"feature": int(feature_ids[feature]), "rank": rank,
                               "transcript_key": rows.iloc[index].transcript_key,
                               "activation": float(activations[index, feature]),
                               "transcript": rows.iloc[index].transcript})
    atlas = pd.DataFrame(atlas_rows)

    def aligned(frame):
        return frame.set_index("transcript_key").loc[rows["transcript_key"]].reset_index()

    a1_aligned = aligned(a1)
    a1_columns = list(a1_aligned.select_dtypes(include=[np.number, "bool"]).columns)
    a1_complete = a1_aligned[a1_columns].notna().all(axis=1).to_numpy(copy=True)
    missing_columns = [column for column in a1_columns if "missing" in column.lower()]
    if missing_columns:
        a1_complete &= ~a1_aligned[missing_columns].astype(bool).any(axis=1).to_numpy()
    views = {"a1_complete": (a1_complete, a1_aligned[a1_columns].to_numpy(dtype=float))}
    all_complete, all_parts = a1_complete.copy(), [a1_aligned[a1_columns]]
    for scorer, frame in sorted(a2_by_scorer.items()):
        scorer_frame = aligned(frame)
        numeric = list(scorer_frame.select_dtypes(include=[np.number, "bool"]).columns)
        metrics = [column for column in numeric if column != "coverage_complete"]
        coverage = scorer_frame["coverage_complete"].eq(True).to_numpy()
        available = scorer_frame[metrics].notna().all(axis=1).to_numpy()
        parts = pd.concat([a1_aligned[a1_columns], scorer_frame[numeric]], axis=1)
        views[f"{scorer}_coverage_complete"] = (
            a1_complete & coverage & available, parts.to_numpy(dtype=float))
        views[f"{scorer}_coverage_truncated"] = (
            a1_complete & ~coverage & available, parts.to_numpy(dtype=float))
        all_complete &= coverage & available
        all_parts.append(scorer_frame[numeric])
    all_features = pd.concat(all_parts, axis=1).to_numpy(dtype=float)
    views["all_scorers_complete"] = (all_complete, all_features)
    r2 = {}
    for label, (mask, features) in views.items():
        masked = features[mask]
        r2[label] = {"n_rows": int(mask.sum()),
                     "sae": _grouped_r2(masked, np.asarray(activations)[mask],
                                        rows.loc[mask, "item_group_key"]),
                     "sae_by_feature": {
                         str(feature_ids[index]): _grouped_r2(
                             masked, np.asarray(activations)[mask, index],
                             rows.loc[mask, "item_group_key"])
                         for index in range(activations.shape[1])},
                     "dense": {name: _grouped_r2(masked, np.asarray(values)[mask],
                                                  rows.loc[mask, "item_group_key"])
                               for name, values in dense_views.items()}}

    correlations = {}
    b_measures = rows[["transcript_key", "chunk_count"]]
    def spearman(left, right):
        left, right = pd.Series(left).reset_index(drop=True), pd.Series(right).reset_index(drop=True)
        valid = left.notna() & right.notna()
        if valid.sum() < 2 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
            return math.nan
        return float(left[valid].corr(right[valid], method="spearman"))

    for scorer, frame in (("a1", a1), *sorted(a2_by_scorer.items()), ("b", b_measures)):
        aligned = rows[["transcript_key"]].merge(
            frame, on="transcript_key", how="left", validate="one_to_one")
        metrics = [column for column in aligned.select_dtypes(
            include=[np.number, "bool"]).columns if "missing" not in column.lower()]
        scorer_report = {}
        subsets = {"all": np.ones(len(aligned), dtype=bool)}
        if "coverage_complete" in aligned:
            subsets["coverage_complete"] = aligned["coverage_complete"].eq(True).to_numpy()
            subsets["coverage_truncated"] = aligned["coverage_complete"].eq(False).to_numpy()
        for label, mask in subsets.items():
            scorer_report[label] = {
                str(feature_ids[feature]): {metric: spearman(
                    activations[mask, feature], aligned.loc[mask, metric])
                    for metric in metrics}
                for feature in range(activations.shape[1])
            }
        correlations[scorer] = scorer_report

    tags = []
    for view, values in r2.items():
        for feature, score in values["sae_by_feature"].items():
            if np.isfinite(score) and score >= 0.5:
                scorer = next((name for name in ("llama", "qwen", "gemma")
                               if view.startswith(name)), "a1")
                tags.append({"feature": int(feature), "criterion": "oof_r2>=0.5",
                             "r2": float(score), "source_measure": "grouped_oof_r2",
                             "scorer": scorer, "coverage": view})
    for scorer, coverage_views in correlations.items():
        for coverage, features in coverage_views.items():
            for feature, measures in features.items():
                for measure, rho in measures.items():
                    if np.isfinite(rho) and abs(rho) >= 0.5:
                        tags.append({"feature": int(feature), "criterion": "abs_rho>=0.5",
                                     "rho": float(rho), "source_measure": measure,
                                     "scorer": scorer, "coverage": coverage})

    global_support = {str(feature_ids[index]): {
        "active_rows": int((activations[:, index] > 0).sum()),
        "total_rows": len(activations),
        "prevalence": float((activations[:, index] > 0).mean())}
        for index in range(activations.shape[1])}

    def grouped_support(groups):
        result = {}
        for name, indices in groups.items():
            result[str(name)] = {str(feature_ids[index]): {
                "active_rows": int((activations[list(indices), index] > 0).sum()),
                "total_rows": len(indices),
                "prevalence": float((activations[list(indices), index] > 0).mean())}
                for index in range(activations.shape[1])}
        return result

    task_family = rows["benchmark"].map({
        "gpqa": "GPQA", "aime2024": "AIME", "aime2025": "AIME",
        "aime2026": "AIME", "intercode": "InterCode", "swe_bench_mini": "SWE"})
    if task_family.isna().any():
        raise ValueError("unknown benchmark in task-family mapping")
    support = {"global": global_support,
               "benchmark": grouped_support(rows.groupby("benchmark", sort=True).groups),
               "task_family": grouped_support(task_family.groupby(task_family).groups)}
    audit = rows[["transcript_key"]].merge(
        audit_map, on="transcript_key", how="left", validate="one_to_one")
    if "model" not in audit or audit["model"].isna().any():
        raise ValueError("post-freeze audit map is missing component model identity")

    def provider(model):
        prefixes = (
            (("claude-",), "anthropic"), (("deepseek-",), "deepseek"),
            (("gemini-", "gemma-"), "google"), (("granite-",), "ibm"),
            (("ling-",), "inclusionai"), (("muse-",), "meta-models"),
            (("mistral-",), "mistralai"), (("kimi-",), "moonshotai"),
            (("nvidia-",), "nvidia"), (("gpt-", "o3"), "openai"),
            (("qwen",), "qwen"), (("glm-",), "z-ai"),
        )
        for candidates, name in prefixes:
            if model.startswith(candidates):
                return name
        raise ValueError(f"unknown frozen model provider: {model}")

    audit["provider"] = audit["model"].map(provider)
    provider_groups = audit.groupby("provider", sort=True).groups
    support["provider"] = grouped_support(provider_groups)
    prevalence = {
        "global": {feature: values["prevalence"]
                   for feature, values in support["global"].items()},
        **{stratum: {group: {feature: values["prevalence"]
                             for feature, values in features.items()}
                      for group, features in support[stratum].items()}
           for stratum in ("benchmark", "provider", "task_family")},
    }
    binary = (activations > 0).astype(np.int64)
    coactivation = {"feature_ids": feature_ids.tolist(),
                    "active_together": (binary.T @ binary).tolist()}

    cross_family = None
    if bge_activations is not None:
        bge = np.asarray(bge_activations)
        if len(bge) != len(activations):
            raise ValueError("Qwen and BGE activations do not align")
        bge_feature_ids = (np.arange(bge.shape[1]) if bge_feature_ids is None
                           else np.asarray(bge_feature_ids, dtype=int))
        def match(mask):
            if int(mask.sum()) < 2:
                return {"status": "not_exercised", "rows": int(mask.sum()),
                        "matches": [], "matched_fraction": None}
            left_values, right_values = activations[mask], bge[mask]
            rho = np.zeros((left_values.shape[1], right_values.shape[1]))
            overlap = np.zeros_like(rho)
            for left in range(rho.shape[0]):
                qwen_top = set(np.argsort(-left_values[:, left])[:20])
                for right in range(rho.shape[1]):
                    rho[left, right] = pd.Series(left_values[:, left]).corr(
                        pd.Series(right_values[:, right]), method="spearman")
                    overlap[left, right] = len(
                        qwen_top & set(np.argsort(-right_values[:, right])[:20]))
            row_ids, column_ids = linear_sum_assignment(-np.nan_to_num(rho, nan=-1.0))
            matches = [{"qwen_feature": int(feature_ids[left]),
                        "bge_feature": int(bge_feature_ids[right]),
                        "spearman_rho": float(rho[left, right]),
                        "top20_overlap": int(overlap[left, right])}
                       for left, right in zip(row_ids, column_ids, strict=True)]
            return {"status": "passed", "rows": int(mask.sum()), "matches": matches,
                    "matched_fraction": float(np.mean(
                        [item["spearman_rho"] >= 0.5 for item in matches]))}
        cross_family = match(np.ones(len(rows), dtype=bool))
        mask = (np.asarray(cross_family_mask, dtype=bool) if cross_family_mask is not None
                else np.ones(len(rows), dtype=bool))
        cross_family["single_chunk_both"] = match(mask)

        qwen_columns = {int(value): index for index, value in enumerate(feature_ids)}
        bge_columns = {int(value): index for index, value in enumerate(bge_feature_ids)}
        def matched_support(row_mask):
            result = []
            for item in cross_family["matches"]:
                qwen_active = activations[row_mask, qwen_columns[item["qwen_feature"]]] > 0
                bge_active = bge[row_mask, bge_columns[item["bge_feature"]]] > 0
                result.append({"qwen_feature": item["qwen_feature"],
                               "bge_feature": item["bge_feature"],
                               "rows": int(row_mask.sum()),
                               "qwen_active": int(qwen_active.sum()),
                               "bge_active": int(bge_active.sum()),
                               "coactive": int((qwen_active & bge_active).sum())})
            return result

        cross_family["matched_support"] = {
            "global": matched_support(np.ones(len(rows), dtype=bool)),
            "single_chunk_both": matched_support(mask),
            "benchmark": {str(name): matched_support(rows.index.isin(indices))
                          for name, indices in rows.groupby(
                              "benchmark", sort=True).groups.items()},
            "provider": {str(name): matched_support(audit.index.isin(indices))
                         for name, indices in provider_groups.items()},
        }
    return {"atlas": atlas, "support": support, "prevalence": prevalence,
            "coactivation": coactivation,
            "shortcut_audit": {"grouped_oof_r2": r2, "correlations": correlations,
                               "tags": tags},
            "cross_family": cross_family}


def write_development_report(output_dir: str | Path, report: dict,
                             source_hashes: dict[str, str]) -> dict:
    def build(stage: Path) -> dict:
        atlas_path = stage / "atlas.parquet"
        report["atlas"].to_parquet(atlas_path, index=False)
        audit_path = stage / "development_report.json"
        audit_path.write_text(json.dumps(json_ready({
            "support": report["support"], "prevalence": report["prevalence"],
            "coactivation": report["coactivation"],
            "shortcut_audit": report["shortcut_audit"],
            "cross_family": report["cross_family"],
        }), sort_keys=True, allow_nan=False) + "\n")
        return {"kind": "development_report", "status": "ready",
                "source_hashes": source_hashes,
                "outputs": {atlas_path.name: sha256_file(atlas_path),
                            audit_path.name: sha256_file(audit_path)}}

    return atomic_artifact_directory(output_dir, build)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/workstream_b.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply-from", type=Path,
                        help="Frozen development SAE directory; apply only, never fit")
    parser.add_argument("--report-from", type=Path,
                        help="Frozen development report required before SAE confirmation")
    parser.add_argument("--development-report", action="store_true")
    parser.add_argument("--sae-dir", type=Path)
    parser.add_argument("--a1", type=Path)
    parser.add_argument("--a2", action="append", default=[], metavar="SCORER=PARQUET")
    parser.add_argument("--bge-sae-dir", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("SAE fitting/application is Slurm-only")
    if not args.development_report and not torch.cuda.is_available():
        parser.error("SAE fitting/application requires a CUDA GPU")

    config = yaml.safe_load(args.config.read_text())
    rows, inputs, dense_manifest = load_dense_artifacts(args.dense_dir)
    if args.development_report:
        if args.sae_dir is None or args.a1 is None or not args.a2:
            parser.error("--development-report requires --sae-dir, --a1, and --a2")
        rows, activations, dense_views, feature_ids, dense_manifest, sae_manifest = (
            load_development_inputs(args.dense_dir, args.sae_dir))
        a2 = {}
        for value in args.a2:
            scorer, separator, path = value.partition("=")
            if not separator or not scorer:
                parser.error("--a2 values must be SCORER=PARQUET")
            a2[scorer] = pd.read_parquet(path)
        bge_activations = None
        bge_feature_ids = None
        cross_family_mask = None
        audit_map = pd.read_parquet(args.dense_dir / "audit_rows.parquet")
        source_hashes = {
            "qwen_dense": sha256_file(args.dense_dir / "manifest.json"),
            "qwen_sae": sha256_file(args.sae_dir / "manifest.json"),
            "a1": sha256_file(args.a1),
            **{f"a2_{name}": sha256_file(value.partition("=")[2])
               for name, value in zip(a2, args.a2, strict=True)},
        }
        if args.bge_sae_dir:
            bge_manifest = json.loads((args.bge_sae_dir / "manifest.json").read_text())
            verify_artifact_files(args.bge_sae_dir, bge_manifest)
            source_hashes["bge_sae"] = sha256_file(args.bge_sae_dir / "manifest.json")
            if bge_manifest.get("report", {}).get("status") == "passed":
                bge_rows = pd.read_parquet(args.bge_sae_dir / "rows.parquet")
                bge_values = np.load(args.bge_sae_dir / "activations.npy", allow_pickle=False)
                bge_feature_ids = np.load(
                    args.bge_sae_dir / "retained_feature_ids.npy", allow_pickle=False).astype(int)
                lookup = pd.Series(range(len(bge_rows)), index=bge_rows["transcript_key"])
                if not rows["transcript_key"].isin(lookup.index).all():
                    raise ValueError("Qwen and BGE development rows do not align")
                bge_activations = bge_values[
                    lookup.loc[rows["transcript_key"]].to_numpy()]
                bge_chunks = bge_rows.set_index("transcript_key").loc[
                    rows["transcript_key"], "chunk_count"].to_numpy()
                cross_family_mask = (rows["chunk_count"].to_numpy() == 1) & (bge_chunks == 1)
        report = development_report(rows, activations, dense_views,
                                    pd.read_parquet(args.a1), a2,
                                    bge_activations=bge_activations,
                                    feature_ids=feature_ids,
                                    bge_feature_ids=bge_feature_ids,
                                    cross_family_mask=cross_family_mask,
                                    audit_map=audit_map)
        write_development_report(args.output, report, source_hashes)
        return
    if args.apply_from:
        if set(rows["split_role"]) != {"confirmation"}:
            raise ValueError("frozen SAE application accepts confirmation rows only")
        development_manifest = json.loads((args.apply_from / "manifest.json").read_text())
        verify_artifact_files(args.apply_from, development_manifest)
        if development_manifest.get("report", {}).get("status") != "passed":
            raise ValueError("confirmation SAE application requires a frozen passed SAE")
        if args.report_from is None:
            raise ValueError("SAE confirmation requires the frozen development report")
        report_manifest = json.loads((args.report_from / "manifest.json").read_text())
        verify_artifact_files(args.report_from, report_manifest)
        source_sae_hash = sha256_file(args.apply_from / "manifest.json")
        source_key = f"{dense_manifest['family']}_sae"
        if (report_manifest.get("kind") != "development_report"
                or report_manifest.get("status") != "ready"
                or report_manifest.get("source_hashes", {}).get(source_key) != source_sae_hash):
            raise ValueError("SAE confirmation requires the matching frozen development report")
        if dense_manifest["family"] != development_manifest["dense_family"]:
            raise ValueError("confirmation dense family differs from the frozen SAE")
        if dense_identity(dense_manifest) != development_manifest["dense_identity"]:
            raise ValueError("confirmation dense identity differs from development extraction")
        sae = load_sae(args.apply_from / "sae_weights.npz")
        retained = np.load(args.apply_from / "retained_feature_ids.npy", allow_pickle=False).astype(int)
        activations, reconstructions, residuals = reconstruct_retained(sae, inputs, retained)
        arrays = {"activations": activations, "reconstructions": reconstructions,
                  "residuals": residuals, "retained_feature_ids": retained}
        report, state = {"status": "passed", "mode": "frozen_confirmation_application",
                         "source_sae_manifest_sha256": sha256_file(
                             args.apply_from / "manifest.json")}, None
    else:
        sae, arrays, report, state = run_sae_suite(inputs, rows, config["sae"])
    manifest = {
        "dense_manifest_sha256": sha256_file(args.dense_dir / "manifest.json"),
        "config_sha256": sha256_file(args.config),
        "sae_code_sha256": sha256_file(__file__),
        "git_commit": git_commit(__file__),
        "dense_family": dense_manifest["family"],
        "dense_identity": dense_identity(dense_manifest),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "report": report,
    }
    if args.apply_from:
        manifest["source_sae_manifest_sha256"] = sha256_file(
            args.apply_from / "manifest.json")
    write_sae_artifacts(args.output, sae, arrays, manifest,
                        rows=rows, training_state=state)


if __name__ == "__main__":
    main()
