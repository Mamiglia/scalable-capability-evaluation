"""Output-conditioned discovery on per-token residual recordings."""

import argparse
import dataclasses
import json
import math
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

import token_manifold_record


@dataclasses.dataclass
class MethodResult:
    method: str
    train_code: np.ndarray
    test_code: np.ndarray
    test_reconstruction: np.ndarray | None
    metadata: dict
    model: object | None = None


def prompt_weights(prompt_idx):
    """Give every prompt total weight one regardless of token count."""
    _, inverse, counts = np.unique(prompt_idx, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _relative_deciles(prompt_idx, group_id, position):
    decile = np.zeros(len(prompt_idx), dtype=np.int8)
    for prompt in np.unique(prompt_idx):
        for group in np.unique(group_id[prompt_idx == prompt]):
            idx = np.flatnonzero((prompt_idx == prompt) & (group_id == group))
            idx = idx[np.argsort(position[idx])]
            if len(idx) > 1:
                decile[idx] = np.rint(np.linspace(0, 9, len(idx))).astype(np.int8)
    return decile


def training_token_indices(prompt_idx, group_id, position, max_per_group=10):
    """Deterministic, bounded token sample for decoder fitting; prediction still uses all tokens."""
    selected = []
    for prompt in np.unique(prompt_idx):
        for group in np.unique(group_id[prompt_idx == prompt]):
            idx = np.flatnonzero((prompt_idx == prompt) & (group_id == group))
            idx = idx[np.argsort(position[idx])]
            take = np.rint(np.linspace(0, len(idx) - 1, min(max_per_group, len(idx)))).astype(int)
            selected.extend(idx[take].tolist())
    return np.asarray(selected, dtype=int)


def relative_token_indices(prompt_idx, position, n_per_prompt):
    """Select the same number of semantic-blind relative positions per prompt."""
    selected = []
    for prompt in np.unique(prompt_idx):
        idx = np.flatnonzero(prompt_idx == prompt)
        idx = idx[np.argsort(position[idx])]
        take = np.rint(np.linspace(0, len(idx) - 1,
                                  min(n_per_prompt, len(idx)))).astype(int)
        selected.extend(idx[take].tolist())
    return np.asarray(selected, dtype=int)


def code_deltas(code, prompt_idx, position):
    """Return adjacent code changes without crossing prompt boundaries."""
    deltas, rows = [], []
    for prompt in np.unique(prompt_idx):
        idx = np.flatnonzero(prompt_idx == prompt)
        idx = idx[np.argsort(position[idx])]
        if len(idx) > 1:
            deltas.append(np.diff(code[idx], axis=0))
            rows.extend(idx[1:].tolist())
    width = code.shape[1:]
    return (np.concatenate(deltas) if deltas else np.empty((0, *width), dtype=code.dtype),
            np.asarray(rows, dtype=int))


def _oof_ridge(X, gap, prompt_idx, item, weights, alphas, cv_folds, fit_idx):
    n_splits = min(cv_folds, len(np.unique(item)))
    prompt_rows = np.arange(len(gap))
    prompt_splits = list(GroupKFold(n_splits=n_splits).split(prompt_rows, gap, item))
    y = gap[prompt_idx]

    def predict(alpha, all_tokens):
        pred = np.full(len(y), np.nan, dtype=np.float64)
        for train_prompts, test_prompts in prompt_splits:
            train = fit_idx[np.isin(prompt_idx[fit_idx], train_prompts)]
            candidates = np.arange(len(y)) if all_tokens else fit_idx
            test = candidates[np.isin(prompt_idx[candidates], test_prompts)]
            mean = X[train].mean(axis=0)
            scale = X[train].std(axis=0)
            scale[scale < 1e-6] = 1
            model = Ridge(alpha=alpha, solver="lsqr", tol=1e-3).fit(
                (X[train] - mean) / scale, y[train], sample_weight=weights[train])
            coef = np.ravel(model.coef_) / scale
            intercept = float(model.intercept_) - mean @ coef
            pred[test] = X[test] @ coef + intercept
        return pred

    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        pred = predict(alpha, all_tokens=False)
        score = r2_score(y[fit_idx], pred[fit_idx], sample_weight=weights[fit_idx])
        if score > best_score:
            best_alpha, best_score = alpha, score
    return best_alpha, predict(best_alpha, all_tokens=True)


def ridge_timeline(data, alphas, cv_folds):
    """Decode final gap per layer with item-grouped OOF predictions at every token."""
    X = np.asarray(data["X"])
    prompt_idx = np.asarray(data["prompt_idx"])
    gap = np.asarray(data["gap"])
    item = np.asarray(data["item"])
    group_id = np.asarray(data["group_id"])
    position = np.asarray(data["position"])
    y = gap[prompt_idx]
    groups = item[prompt_idx]
    weights = prompt_weights(prompt_idx)
    decile = _relative_deciles(prompt_idx, group_id, position)
    fit_idx = training_token_indices(prompt_idx, group_id, position)
    oof_frames, summary_rows = [], []

    for layer in range(X.shape[1]):
        alpha, pred = _oof_ridge(X[:, layer], gap, prompt_idx, item, weights,
                                 alphas, cv_folds, fit_idx)
        frame = pd.DataFrame({"prompt_idx": prompt_idx, "item": groups, "position": position,
                              "group_id": group_id, "relative_decile": decile,
                              "layer": layer, "gap": y, "pred_gap": pred,
                              "weight": weights, "alpha": alpha})
        oof_frames.append(frame)
        for (structural_group, relative_decile), sub in frame.groupby(
                ["group_id", "relative_decile"], sort=True):
            summary_rows.append({"layer": layer, "structural_group": structural_group,
                                 "relative_decile": relative_decile, "alpha": alpha,
                                 "r2": r2_score(sub.gap, sub.pred_gap,
                                                sample_weight=sub.weight)})
    return pd.DataFrame(summary_rows), pd.concat(oof_frames, ignore_index=True)


def load_manifest(path, layer=None):
    """Verify and concatenate immutable activation shards."""
    path = pathlib.Path(path)
    manifest = json.loads(path.read_text())
    if layer is not None and layer not in manifest["layers"]:
        raise ValueError(f"layer {layer} not in manifest layers {manifest['layers']}")
    layer_idx = None if layer is None else manifest["layers"].index(layer)
    root = path.parent
    resid, prompt_idx, position, input_ids, offsets, group_id, prompts = [], [], [], [], [], [], []
    prompt_offset = 0
    for entry in manifest["shards"]:
        shard_path = root / entry["path"]
        if token_manifold_record.sha256(shard_path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {shard_path}")
        blob = torch.load(shard_path, map_location="cpu", weights_only=False)
        n_tokens = len(blob["prompt_idx"])
        if n_tokens != entry["n_tokens"] or len(blob["resid"]) != n_tokens:
            raise ValueError(f"token-count mismatch: {shard_path}")
        values = blob["resid"]
        if layer_idx is not None:
            values = values[:, layer_idx:layer_idx + 1]
        resid.append(values.float().numpy())
        prompt_idx.append(blob["prompt_idx"].numpy().astype(np.int64) + prompt_offset)
        position.append(blob["position"].numpy())
        input_ids.append(blob["input_ids"].numpy())
        offsets.append(blob["offsets"].numpy())
        group_id.append(blob["group_id"].numpy())
        prompts.extend(blob["prompts"])
        prompt_offset += len(blob["prompts"])
    if prompt_offset != manifest["n_prompts"] or sum(map(len, prompt_idx)) != manifest["n_tokens"]:
        raise ValueError("manifest totals disagree with shards")
    return {"X": np.concatenate(resid), "prompt_idx": np.concatenate(prompt_idx),
            "position": np.concatenate(position), "input_ids": np.concatenate(input_ids),
            "offsets": np.concatenate(offsets), "group_id": np.concatenate(group_id),
            "gap": np.asarray([prompt["gap"] for prompt in prompts], dtype=np.float64),
            "item": np.asarray([prompt["item"] for prompt in prompts]),
            "prompts": prompts, "manifest": manifest}


def fit_cebra(X_train, gap_train, X_test, output_dim, seed, iterations, device="cpu",
              conditional="time_delta", delta=None, num_hidden_units=32):
    """Fit output-gap-conditioned CEBRA and return train/test latent codes."""
    import cebra
    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs = {"model_architecture": "offset1-model", "conditional": conditional,
              "output_dimension": output_dim, "max_iterations": iterations,
              "batch_size": min(512, len(X_train)), "device": device,
              "num_hidden_units": num_hidden_units}
    if delta is not None:
        kwargs["delta"] = delta
    model = cebra.CEBRA(**kwargs)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    labels = np.asarray(gap_train, dtype=np.float32)
    if conditional in {"delta", "delta_normal"}:
        scale = labels.std()
        labels = (labels - labels.mean()) / (scale if scale >= 1e-6 else 1)
    model.fit(X_train, labels[:, None])
    return MethodResult("cebra", model.transform(X_train), model.transform(X_test), None,
                        {"output_dim": output_dim, "seed": seed}, model=model)


def code_gap_r2(result, gap_train, gap_test, alpha=1e-6):
    """Held-out gap decoding from fixed latent codes."""
    regressor = Ridge(alpha=alpha).fit(result.train_code, gap_train)
    return float(r2_score(gap_test, regressor.predict(result.test_code)))


def grouped_code_predictions(code_train, gap_train, item_train, code_test,
                             alphas=(0.01, 0.1, 1, 10, 100, 1000, 10_000,
                                     100_000, 1_000_000), cv_folds=4):
    """Tune a scaled ridge decoder inside item groups, then predict held-out codes."""
    splits = GroupKFold(n_splits=min(cv_folds, len(np.unique(item_train)))).split(
        code_train, gap_train, item_train)
    splits = list(splits)
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        pred = np.full(len(gap_train), np.nan)
        for train, valid in splits:
            mean = code_train[train].mean(axis=0)
            scale = code_train[train].std(axis=0)
            scale[scale < 1e-6] = 1
            model = Ridge(alpha=alpha, solver="lsqr").fit(
                (code_train[train] - mean) / scale, gap_train[train])
            pred[valid] = model.predict((code_train[valid] - mean) / scale)
        score = r2_score(gap_train, pred)
        if score > best_score:
            best_alpha, best_score = alpha, score
    mean = code_train.mean(axis=0)
    scale = code_train.std(axis=0)
    scale[scale < 1e-6] = 1
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(
        (code_train - mean) / scale, gap_train)
    return model.predict((code_test - mean) / scale), best_alpha


def grouped_knn_predictions(code_train, gap_train, item_train, code_test,
                            neighbors=(3, 5, 10, 20), cv_folds=4):
    """Tune a scaled KNN decoder inside item groups for curved latent geometry."""
    splits = list(GroupKFold(
        n_splits=min(cv_folds, len(np.unique(item_train)))).split(
            code_train, gap_train, item_train))
    best_neighbors, best_score = neighbors[0], -np.inf
    for n_neighbors in neighbors:
        if any(n_neighbors > len(train) for train, _ in splits):
            continue
        pred = np.full(len(gap_train), np.nan)
        for train, valid in splits:
            mean = code_train[train].mean(axis=0)
            scale = code_train[train].std(axis=0)
            scale[scale < 1e-6] = 1
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance").fit(
                (code_train[train] - mean) / scale, gap_train[train])
            pred[valid] = model.predict((code_train[valid] - mean) / scale)
        score = r2_score(gap_train, pred)
        if score > best_score:
            best_neighbors, best_score = n_neighbors, score
    mean = code_train.mean(axis=0)
    scale = code_train.std(axis=0)
    scale[scale < 1e-6] = 1
    model = KNeighborsRegressor(n_neighbors=best_neighbors, weights="distance").fit(
        (code_train - mean) / scale, gap_train)
    return model.predict((code_test - mean) / scale), best_neighbors


def fit_bsf(X_train, X_test, group_size, n_groups, l0, seed, epochs,
            variant="grassmannian", device="cpu", snr=0, batch_size=64):
    """Train a pinned Goodfire block-sparse featurizer."""
    import bsf
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = bsf.GrassmannianBSF if variant == "grassmannian" else bsf.VanillaBSF
    model = cls(d=X_train.shape[1], n_groups=n_groups, group_size=group_size, l0=l0)
    model = bsf.train(model, np.asarray(X_train, dtype=np.float32), epochs=epochs,
                      batch_size=min(batch_size, len(X_train)), snr=snr, device=device,
                      log_every=max(1, epochs))
    model.eval()
    with torch.no_grad():
        train_recon, train_code = model(torch.as_tensor(X_train, dtype=torch.float32, device=device))
        test_recon, test_code = model(torch.as_tensor(X_test, dtype=torch.float32, device=device))
    return MethodResult("bsf", train_code.flatten(1).cpu().numpy(),
                        test_code.flatten(1).cpu().numpy(), test_recon.cpu().numpy(),
                        {"seed": seed, "group_size": group_size, "n_groups": n_groups,
                         "l0": l0, "block_train_code": train_code.cpu().numpy()}, model=model)


def fit_sae(X_train, X_test, width, k, seed, epochs, device="cpu"):
    """Train the minimal BatchTopK SAE supplied by Goodfire's evaluation package."""
    from saes import BatchTopKSAE
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BatchTopKSAE(X_train.shape[1], width, k, device=device)
    X = torch.as_tensor(X_train, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(64):
            batch = X[idx].to(device)
            reconstruction, _ = model(batch)
            loss = torch.nn.functional.mse_loss(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.decoder.weight.data = torch.nn.functional.normalize(
                    model.decoder.weight.data, dim=0)
    model.eval()
    with torch.no_grad():
        train_recon, train_code = model(torch.as_tensor(X_train, dtype=torch.float32, device=device))
        test_recon, test_code = model(torch.as_tensor(X_test, dtype=torch.float32, device=device))
    return MethodResult("sae", train_code.cpu().numpy(), test_code.cpu().numpy(),
                        test_recon.cpu().numpy(), {"seed": seed, "width": width, "k": k},
                        model=model)


def fit_batchtopk(X_train, X_test, width, k, seed, steps, batch_size=256,
                  device="cpu"):
    """Train the pinned-reference BatchTopK objective on recorded activations."""
    from batch_topk import BatchTopKSAE
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BatchTopKSAE(X_train.shape[1], width, k).to(device)
    X = torch.as_tensor(X_train, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))
    generator = torch.Generator().manual_seed(seed)
    order, cursor = torch.randperm(len(X), generator=generator), 0
    for _ in range(steps):
        if cursor + min(batch_size, len(X)) > len(X):
            order, cursor = torch.randperm(len(X), generator=generator), 0
        idx = order[cursor:cursor + min(batch_size, len(X))]
        cursor += len(idx)
        loss, _ = model.training_loss(X[idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        model.project_decoder_grad_and_normalize()
        optimizer.step()
    batches = [X[start:start + batch_size].to(device)
               for start in range(0, len(X), batch_size)]
    model.calibrate_threshold(batches)

    def encode(values):
        tensor = torch.as_tensor(values, dtype=torch.float32)
        codes, reconstructions = [], []
        with torch.no_grad():
            for start in range(0, len(tensor), batch_size):
                batch = tensor[start:start + batch_size].to(device)
                code = model.encode(batch)
                codes.append(code.cpu())
                reconstructions.append(model.decode(code).cpu())
        return torch.cat(codes).numpy(), torch.cat(reconstructions).numpy()

    train_code, _ = encode(X_train)
    test_code, test_recon = encode(X_test)
    return MethodResult("batchtopk", train_code, test_code, test_recon,
                        {"seed": seed, "width": width, "k": k,
                         "threshold": float(model.threshold)}, model=model)


def ising_clusters(train_code, alpha, method, n_clusters=None):
    """Group SAE features from inverse-Ising conditional dependencies."""
    from unsupervised_clustering import compute_ising, cluster_leiden, cluster_spectral
    matrix, _, _ = compute_ising(np.asarray(train_code) > 0, alpha=alpha, n_jobs=1)
    if method == "spectral":
        clusters, _ = cluster_spectral(matrix, n_clusters=n_clusters)
    elif method == "leiden":
        clusters, _ = cluster_leiden(matrix)
    else:
        raise ValueError(f"unknown clustering method: {method}")
    labels = np.full(train_code.shape[1], -1, dtype=int)
    for label, members in clusters.items():
        labels[members] = label
    return labels


def method_configs(name, d_in, quick=False):
    """Bounded quick grid or the predeclared scientific dimensionality grid."""
    if quick:
        if name == "cebra":
            return [{"output_dim": 4, "iterations": 200}]
        if name == "bsf":
            return [{"group_size": 2, "n_groups": 8, "l0": 2, "epochs": 5}]
        if name == "sae":
            return [{"width": 16, "k": 2, "epochs": 5}]
    if name == "cebra":
        return [{"output_dim": dim, "iterations": 10_000} for dim in (1, 2, 4, 8, 16)]
    if name == "bsf":
        return [{"group_size": size, "n_groups": max(2, math.ceil(d_in / size)),
                 "l0": min(max(2, math.ceil(d_in / size)), max(1, math.ceil(32 / size))),
                 "epochs": 40}
                for size in (1, 2, 4, 8)]
    if name == "sae":
        return [{"width": d_in, "k": min(32, d_in), "epochs": 40}]
    raise ValueError(f"unknown method: {name}")


def fit_method(name, X_train, gap_train, X_test, seed, quick=False, config=None):
    """Small method dispatcher used by the bake-off."""
    config = method_configs(name, X_train.shape[1], quick)[0] if config is None else config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "raw":
        return MethodResult("raw", np.asarray(X_train), np.asarray(X_test),
                            np.asarray(X_test), {"seed": seed})
    if name == "pca":
        model = PCA(n_components=config["output_dim"], svd_solver="randomized",
                    random_state=seed).fit(X_train)
        return MethodResult("pca", model.transform(X_train), model.transform(X_test),
                            model.inverse_transform(model.transform(X_test)),
                            {"seed": seed, "output_dim": config["output_dim"]})
    if name == "random":
        rng = np.random.default_rng(seed)
        projection, _ = np.linalg.qr(rng.normal(size=(X_train.shape[1],
                                                       config["output_dim"])))
        train_code, test_code = X_train @ projection, X_test @ projection
        return MethodResult("random", train_code, test_code, test_code @ projection.T,
                            {"seed": seed, "output_dim": config["output_dim"]})
    if name == "cebra":
        assert gap_train is not None
        return fit_cebra(X_train, gap_train, X_test, config["output_dim"], seed,
                         config["iterations"], device="cuda_if_available",
                         conditional=config.get("conditional", "time_delta"),
                         delta=config.get("delta"),
                         num_hidden_units=config.get("num_hidden_units", 32))
    if name == "bsf":
        return fit_bsf(X_train, X_test, config["group_size"], config["n_groups"],
                       config["l0"], seed, config["epochs"],
                       variant=config.get("variant", "grassmannian"), device=device,
                       snr=config.get("snr", 0), batch_size=config.get("batch_size", 64))
    if name == "sae":
        return fit_sae(X_train, X_test, config["width"], config["k"], seed,
                       config["epochs"], device=device)
    if name == "batchtopk":
        return fit_batchtopk(X_train, X_test, config["width"], config["k"], seed,
                             config["steps"], config.get("batch_size", 256), device=device)
    raise ValueError(f"unknown method: {name}")


def _sample_prompt_groups(prompt_idx, group_id, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for prompt in np.unique(prompt_idx):
        for group in np.unique(group_id[prompt_idx == prompt]):
            candidates = np.flatnonzero((prompt_idx == prompt) & (group_id == group))
            selected.append(int(rng.choice(candidates)))
    return np.asarray(selected, dtype=int)


def trajectory_config(name, d_in, quick=False):
    """One prespecified calibration configuration per discovery family."""
    if name == "cebra":
        return {"output_dim": 3 if quick else 8,
                "iterations": 300 if quick else 10_000,
                "conditional": "delta_normal", "delta": 0.25,
                "num_hidden_units": 32 if quick else 128}
    if name == "bsf":
        n_groups = math.ceil(d_in / 3)
        return {"group_size": 3, "n_groups": n_groups, "l0": min(16, n_groups),
                "epochs": 10 if quick else 60, "variant": "grassmannian",
                "snr": 0.1, "batch_size": 512}
    if name == "sae":
        return {"width": d_in, "k": min(32, d_in),
                "epochs": 10 if quick else 60}
    if name == "batchtopk":
        return {"width": 2 * d_in, "k": min(32, 2 * d_in),
                "steps": 200 if quick else 1000, "batch_size": 256}
    if name == "raw":
        return {}
    if name in {"pca", "random"}:
        return {"output_dim": min(8, d_in)}
    raise ValueError(f"unknown method: {name}")


def _prompt_deciles(prompt_idx, position):
    decile = np.zeros(len(prompt_idx), dtype=np.int8)
    for prompt in np.unique(prompt_idx):
        idx = np.flatnonzero(prompt_idx == prompt)
        idx = idx[np.argsort(position[idx])]
        if len(idx) > 1:
            decile[idx] = np.rint(np.linspace(0, 9, len(idx))).astype(np.int8)
    return decile


def _normalise_trajectory(X_train, X_test, method):
    mean, scale = _trajectory_stats(X_train, method)
    return (X_train - mean) / scale, (X_test - mean) / scale


def _trajectory_stats(X_train, method):
    mean = X_train.mean(axis=0, keepdims=True)
    centered = X_train - mean
    if method == "bsf":
        scale = np.sqrt(np.square(centered).mean())
    else:
        scale = np.sqrt(np.square(centered).mean(axis=0, keepdims=True))
        scale[scale < 1e-6] = 1
    if np.ndim(scale) == 0 and scale < 1e-6:
        scale = 1.0
    return mean, scale


def run_trajectory(manifest, layers, method, out_dir, seeds, positions_per_prompt=64,
                   cv_folds=3, quick=False):
    """Fit one encoder per layer and score held-out state/change over token time."""
    out_dir = pathlib.Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    frames = []
    for layer in layers:
        data = load_manifest(manifest, layer=layer)
        selected = relative_token_indices(data["prompt_idx"], data["position"],
                                          positions_per_prompt)
        X = data["X"][selected, 0]
        prompt = data["prompt_idx"][selected]
        position = data["position"][selected]
        group_id = data["group_id"][selected]
        gap = data["gap"][prompt]
        item = data["item"][prompt]
        decile = _prompt_deciles(prompt, position)
        splits = list(GroupKFold(n_splits=min(cv_folds, len(np.unique(item)))).split(
            X, gap, item))
        config = trajectory_config(method, X.shape[1], quick)
        for seed in seeds:
            for fold, (train, test) in enumerate(splits):
                X_train, X_test = _normalise_trajectory(X[train], X[test], method)
                result = fit_method(method, X_train, gap[train], X_test, seed, quick, config)
                train_delta, train_rows = code_deltas(
                    result.train_code, prompt[train], position[train])
                test_delta, test_rows = code_deltas(
                    result.test_code, prompt[test], position[test])
                decoders = {"ridge": grouped_code_predictions}
                if method == "cebra":
                    decoders["knn"] = grouped_knn_predictions
                for decoder, decode in decoders.items():
                    pred, parameter = decode(
                        result.train_code, gap[train], item[train], result.test_code)
                    frames.append(pd.DataFrame({
                        "method": method, "decoder": decoder, "layer": layer,
                        "seed": seed, "fold": fold, "kind": "state",
                        "prompt_idx": prompt[test], "item": item[test],
                        "position": position[test], "structural_group": group_id[test],
                        "relative_decile": decile[test], "gap": gap[test],
                        "pred_gap": pred, "decoder_param": parameter}))
                    delta_pred, delta_parameter = decode(
                        train_delta, gap[train][train_rows], item[train][train_rows],
                        test_delta)
                    frames.append(pd.DataFrame({
                        "method": method, "decoder": decoder, "layer": layer,
                        "seed": seed, "fold": fold, "kind": "delta",
                        "prompt_idx": prompt[test][test_rows],
                        "item": item[test][test_rows],
                        "position": position[test][test_rows],
                        "structural_group": group_id[test][test_rows],
                        "relative_decile": decile[test][test_rows],
                        "gap": gap[test][test_rows], "pred_gap": delta_pred,
                        "decoder_param": delta_parameter}))
    oof = pd.concat(frames, ignore_index=True)
    rows = []
    keys = ["method", "decoder", "layer", "seed", "kind", "structural_group",
            "relative_decile"]
    for values, sub in oof.groupby(keys, sort=True):
        score = r2_score(sub.gap, sub.pred_gap, sample_weight=prompt_weights(sub.prompt_idx))
        rows.append(dict(zip(keys, values), r2=score, n_rows=len(sub),
                         decoder_param=float(np.median(sub.decoder_param))))
    table = pd.DataFrame(rows)
    _atomic_csv(table, out_dir / "trajectory.csv")
    _atomic_csv(oof, out_dir / "trajectory_oof.csv")
    return table


def run_trace(manifest, layer, method, out_dir, seed, positions_per_prompt=64,
              quick=False):
    """Fit a frozen layer model and export exact-token latent codes for localization."""
    out_dir = pathlib.Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    if method not in {"cebra", "bsf", "sae", "batchtopk"}:
        raise ValueError(f"trace requires a learned method, got {method}")
    data = load_manifest(manifest, layer=layer)
    selected = relative_token_indices(data["prompt_idx"], data["position"],
                                      positions_per_prompt)
    X = data["X"][:, 0]
    mean, scale = _trajectory_stats(X[selected], method)
    X_train = (X[selected] - mean) / scale
    X_all = (X - mean) / scale
    config = trajectory_config(method, X.shape[1], quick)
    prompt = data["prompt_idx"][selected]
    result = fit_method(method, X_train, data["gap"][prompt], X_all, seed, quick, config)
    payload = {
        "manifest": str(manifest), "method": method, "layer": layer, "seed": seed,
        "config": config, "prompt_idx": torch.as_tensor(data["prompt_idx"], dtype=torch.int32),
        "position": torch.as_tensor(data["position"], dtype=torch.int32),
        "input_ids": torch.as_tensor(data["input_ids"], dtype=torch.int32),
        "offsets": torch.as_tensor(data["offsets"], dtype=torch.int32),
        "group_id": torch.as_tensor(data["group_id"], dtype=torch.int8),
        "prompts": data["prompts"],
    }
    code = np.asarray(result.test_code)
    if method == "cebra":
        payload["code"] = torch.as_tensor(code, dtype=torch.float16)
    elif method == "sae":
        k = config["k"]
        indices = np.argpartition(code, -k, axis=1)[:, -k:]
        values = np.take_along_axis(code, indices, axis=1)
        payload["code_indices"] = torch.as_tensor(indices, dtype=torch.int32)
        payload["code_values"] = torch.as_tensor(values, dtype=torch.float16)
    elif method == "batchtopk":
        row, column = np.nonzero(code)
        counts = np.bincount(row, minlength=len(code))
        payload["code_row_ptr"] = torch.as_tensor(
            np.concatenate([[0], np.cumsum(counts)]), dtype=torch.int64)
        payload["code_indices"] = torch.as_tensor(column, dtype=torch.int32)
        payload["code_values"] = torch.as_tensor(code[row, column], dtype=torch.float16)
    else:
        blocks = code.reshape(len(code), config["n_groups"], config["group_size"])
        norms = np.square(blocks).sum(axis=2)
        indices = np.argpartition(norms, -config["l0"], axis=1)[:, -config["l0"]:]
        values = np.take_along_axis(blocks, indices[:, :, None], axis=1)
        payload["block_indices"] = torch.as_tensor(indices, dtype=torch.int32)
        payload["block_values"] = torch.as_tensor(values, dtype=torch.float16)
    out_dir.mkdir(parents=True)
    token_manifold_record.atomic_torch_save(payload, out_dir / "codes.pt")
    if method == "cebra":
        tmp = out_dir / "model.cebra.partial"
        result.model.save(str(tmp))
        os.replace(tmp, out_dir / "model.cebra")
        token_manifold_record.atomic_torch_save(
            {"mean": mean, "scale": scale, "config": config}, out_dir / "normalizer.pt")
    else:
        token_manifold_record.atomic_torch_save(
            {"state_dict": result.model.state_dict(), "mean": mean, "scale": scale,
             "config": config}, out_dir / "model.pt")
    return out_dir / "codes.pt"


def _reconstruction_r2(target, reconstruction):
    if reconstruction is None:
        return np.nan
    residual = np.square(target - reconstruction).sum()
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum()
    return float(1 - residual / max(total, 1e-12))


def run_bakeoff(manifest, layer, methods, out_dir, seeds, quick=False):
    """Run a bounded, equal-row method comparison on one recorded layer."""
    out_dir = pathlib.Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    data = load_manifest(manifest)
    layers = list(data["manifest"]["layers"])
    if layer not in layers:
        raise ValueError(f"layer {layer} not in manifest layers {layers}")
    layer_idx = layers.index(layer)
    selected = _sample_prompt_groups(data["prompt_idx"], data["group_id"], seeds[0])
    prompt = data["prompt_idx"][selected]
    X = data["X"][selected, layer_idx]
    y = data["gap"][prompt]
    groups = data["item"][prompt]
    train, test = next(GroupKFold(n_splits=2).split(X, y, groups))
    mean = X[train].mean(axis=0, keepdims=True)
    scale = np.sqrt(np.square(X[train] - mean).mean(axis=0, keepdims=True))
    scale[scale < 1e-6] = 1
    X_train, X_test = (X[train] - mean) / scale, (X[test] - mean) / scale
    rows, artifacts = [], []
    for method in methods:
        for config in method_configs(method, X.shape[1], quick):
            for seed in seeds:
                start = time.monotonic()
                result = fit_method(method, X_train, y[train], X_test, seed, quick, config)
                rows.append({"method": method, "config": json.dumps(config, sort_keys=True),
                             "seed": seed,
                             "test_gap_r2": code_gap_r2(result, y[train], y[test]),
                             "test_recon_r2": _reconstruction_r2(X_test, result.test_reconstruction),
                             "active_code_count": float((result.test_code != 0).sum(axis=1).mean()),
                             "runtime_s": time.monotonic() - start})
                artifacts.append({"method": method, "config": config, "seed": seed,
                                  "train_code": result.train_code, "test_code": result.test_code,
                                  "metadata": result.metadata})
    out_dir.mkdir(parents=True)
    token_manifold_record.atomic_torch_save(
        {"manifest": str(manifest), "layer": layer, "artifacts": artifacts,
         "train_rows": train, "test_rows": test, "mean": mean, "scale": scale},
        out_dir / "bakeoff.pt")
    _atomic_csv(pd.DataFrame(rows), out_dir / "bakeoff.csv")
    return pd.DataFrame(rows)


def _atomic_csv(df, path):
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".partial")
    if path.exists() or tmp.exists():
        raise FileExistsError(path if path.exists() else tmp)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    timeline = subparsers.add_parser("timeline")
    timeline.add_argument("--manifest", required=True)
    timeline.add_argument("--out", required=True)
    timeline.add_argument("--alphas", default="0.1,1,10,100")
    timeline.add_argument("--cv-folds", type=int, default=5)
    timeline.add_argument("--seed", type=int, default=0)
    bakeoff = subparsers.add_parser("bakeoff")
    bakeoff.add_argument("--manifest", required=True)
    bakeoff.add_argument("--layer", required=True, type=int)
    bakeoff.add_argument("--methods", default="cebra,bsf,sae")
    bakeoff.add_argument("--seeds", default="0,1,2,3,4")
    bakeoff.add_argument("--out", required=True)
    bakeoff.add_argument("--quick", action="store_true")
    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("--manifest", required=True)
    trajectory.add_argument("--layers", required=True)
    trajectory.add_argument("--method",
                            choices=("cebra", "bsf", "sae", "batchtopk", "raw",
                                     "pca", "random"),
                            required=True)
    trajectory.add_argument("--seeds", default="0,1,2")
    trajectory.add_argument("--positions-per-prompt", type=int, default=64)
    trajectory.add_argument("--cv-folds", type=int, default=3)
    trajectory.add_argument("--out", required=True)
    trajectory.add_argument("--quick", action="store_true")
    trace = subparsers.add_parser("trace")
    trace.add_argument("--manifest", required=True)
    trace.add_argument("--layer", required=True, type=int)
    trace.add_argument("--method", choices=("cebra", "bsf", "sae", "batchtopk"),
                       required=True)
    trace.add_argument("--seed", type=int, default=0)
    trace.add_argument("--positions-per-prompt", type=int, default=64)
    trace.add_argument("--out", required=True)
    trace.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.command == "timeline":
        data = load_manifest(args.manifest)
        alphas = tuple(float(value) for value in args.alphas.split(","))
        summary, oof = ridge_timeline(data, alphas, args.cv_folds)
        null_data = dict(data)
        null_data["gap"] = np.random.default_rng(args.seed).permutation(data["gap"])
        null, _ = ridge_timeline(null_data, alphas, args.cv_folds)
        keys = ["layer", "structural_group", "relative_decile"]
        summary = summary.merge(null[keys + ["r2"]].rename(columns={"r2": "null_r2"}),
                                on=keys, validate="one_to_one")
        out = pathlib.Path(args.out)
        _atomic_csv(summary, out)
        _atomic_csv(oof, out.with_name(out.stem + "_oof.csv"))
        print(f"token timeline -> {out}")
    elif args.command == "bakeoff":
        run_bakeoff(args.manifest, args.layer, tuple(args.methods.split(",")), args.out,
                    tuple(int(seed) for seed in args.seeds.split(",")), args.quick)
        print(f"token manifold bakeoff -> {args.out}")
    elif args.command == "trajectory":
        run_trajectory(args.manifest, tuple(int(layer) for layer in args.layers.split(",")),
                       args.method, args.out,
                       tuple(int(seed) for seed in args.seeds.split(",")),
                       args.positions_per_prompt, args.cv_folds, args.quick)
        print(f"token trajectory -> {args.out}")
    elif args.command == "trace":
        run_trace(args.manifest, args.layer, args.method, args.out, args.seed,
                  args.positions_per_prompt, args.quick)
        print(f"exact token trace -> {args.out}")


if __name__ == "__main__":
    main()
