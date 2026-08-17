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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import token_manifold_record


@dataclasses.dataclass
class MethodResult:
    method: str
    train_code: np.ndarray
    test_code: np.ndarray
    test_reconstruction: np.ndarray | None
    metadata: dict


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
            pred[test] = Ridge(alpha=alpha, solver="lsqr", tol=1e-3).fit(
                X[train], y[train], sample_weight=weights[train]).predict(X[test])
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


def load_manifest(path):
    """Verify and concatenate immutable activation shards."""
    path = pathlib.Path(path)
    manifest = json.loads(path.read_text())
    root = path.parent
    resid, prompt_idx, position, group_id, prompts = [], [], [], [], []
    prompt_offset = 0
    for entry in manifest["shards"]:
        shard_path = root / entry["path"]
        if token_manifold_record.sha256(shard_path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {shard_path}")
        blob = torch.load(shard_path, map_location="cpu", weights_only=False)
        n_tokens = len(blob["prompt_idx"])
        if n_tokens != entry["n_tokens"] or len(blob["resid"]) != n_tokens:
            raise ValueError(f"token-count mismatch: {shard_path}")
        resid.append(blob["resid"].float().numpy())
        prompt_idx.append(blob["prompt_idx"].numpy().astype(np.int64) + prompt_offset)
        position.append(blob["position"].numpy())
        group_id.append(blob["group_id"].numpy())
        prompts.extend(blob["prompts"])
        prompt_offset += len(blob["prompts"])
    if prompt_offset != manifest["n_prompts"] or sum(map(len, prompt_idx)) != manifest["n_tokens"]:
        raise ValueError("manifest totals disagree with shards")
    return {"X": np.concatenate(resid), "prompt_idx": np.concatenate(prompt_idx),
            "position": np.concatenate(position), "group_id": np.concatenate(group_id),
            "gap": np.asarray([prompt["gap"] for prompt in prompts], dtype=np.float64),
            "item": np.asarray([prompt["item"] for prompt in prompts]),
            "prompts": prompts, "manifest": manifest}


def fit_cebra(X_train, gap_train, X_test, output_dim, seed, iterations, device="cpu"):
    """Fit output-gap-conditioned CEBRA and return train/test latent codes."""
    import cebra
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = cebra.CEBRA(model_architecture="offset1-model", conditional="time_delta",
                        output_dimension=output_dim, max_iterations=iterations,
                        batch_size=min(512, len(X_train)), device=device)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    model.fit(X_train, np.asarray(gap_train, dtype=np.float32)[:, None])
    return MethodResult("cebra", model.transform(X_train), model.transform(X_test), None,
                        {"output_dim": output_dim, "seed": seed})


def code_gap_r2(result, gap_train, gap_test, alpha=1e-6):
    """Held-out gap decoding from fixed latent codes."""
    regressor = Ridge(alpha=alpha).fit(result.train_code, gap_train)
    return float(r2_score(gap_test, regressor.predict(result.test_code)))


def fit_bsf(X_train, X_test, group_size, n_groups, l0, seed, epochs,
            variant="grassmannian", device="cpu"):
    """Train a pinned Goodfire block-sparse featurizer."""
    import bsf
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = bsf.GrassmannianBSF if variant == "grassmannian" else bsf.VanillaBSF
    model = cls(d=X_train.shape[1], n_groups=n_groups, group_size=group_size, l0=l0)
    model = bsf.train(model, np.asarray(X_train, dtype=np.float32), epochs=epochs,
                      batch_size=min(64, len(X_train)), snr=0, device=device,
                      log_every=max(1, epochs))
    model.eval()
    with torch.no_grad():
        train_recon, train_code = model(torch.as_tensor(X_train, dtype=torch.float32, device=device))
        test_recon, test_code = model(torch.as_tensor(X_test, dtype=torch.float32, device=device))
    return MethodResult("bsf", train_code.flatten(1).cpu().numpy(),
                        test_code.flatten(1).cpu().numpy(), test_recon.cpu().numpy(),
                        {"seed": seed, "group_size": group_size, "n_groups": n_groups,
                         "l0": l0, "block_train_code": train_code.cpu().numpy()})


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
                        test_recon.cpu().numpy(), {"seed": seed, "width": width, "k": k})


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
    config = config or method_configs(name, X_train.shape[1], quick)[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cebra":
        assert gap_train is not None
        return fit_cebra(X_train, gap_train, X_test, config["output_dim"], seed,
                         config["iterations"], device="cuda_if_available")
    if name == "bsf":
        return fit_bsf(X_train, X_test, config["group_size"], config["n_groups"],
                       config["l0"], seed, config["epochs"], device=device)
    if name == "sae":
        return fit_sae(X_train, X_test, config["width"], config["k"], seed,
                       config["epochs"], device=device)
    raise ValueError(f"unknown method: {name}")


def _sample_prompt_groups(prompt_idx, group_id, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for prompt in np.unique(prompt_idx):
        for group in np.unique(group_id[prompt_idx == prompt]):
            candidates = np.flatnonzero((prompt_idx == prompt) & (group_id == group))
            selected.append(int(rng.choice(candidates)))
    return np.asarray(selected, dtype=int)


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


if __name__ == "__main__":
    main()
