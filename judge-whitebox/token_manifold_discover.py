"""Output-conditioned discovery on per-token residual recordings."""

import argparse
import dataclasses
import json
import os
import pathlib

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


def _oof_ridge(X, y, groups, weights, alphas, cv_folds):
    n_splits = min(cv_folds, len(np.unique(groups)))
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        pred = np.empty(len(y), dtype=np.float64)
        for train, test in splits:
            pred[test] = Ridge(alpha=alpha).fit(
                X[train], y[train], sample_weight=weights[train]).predict(X[test])
        score = r2_score(y, pred, sample_weight=weights)
        if score > best_score:
            best_alpha, best_score = alpha, score
    pred = np.empty(len(y), dtype=np.float64)
    for train, test in splits:
        pred[test] = Ridge(alpha=best_alpha).fit(
            X[train], y[train], sample_weight=weights[train]).predict(X[test])
    return best_alpha, pred


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
    oof_frames, summary_rows = [], []

    for layer in range(X.shape[1]):
        alpha, pred = _oof_ridge(X[:, layer], y, groups, weights, alphas, cv_folds)
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


if __name__ == "__main__":
    main()
