"""Per-benchmark sparse autoencoders on frozen transcript embeddings, WIMHF recipe.

One BatchTopK SAE (Movva et al., arXiv:2510.26202; code vendored in wimhf_sae.py)
per fit group, trained on representation_train rows with representation_validation
rows for early stopping, then applied to every row. All width features are kept
here with signed activations (sign chosen so the majority direction is positive); feature
retention happens later, from description
fidelity and prevalence (transcript_sae_describe.py).
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from transcript_embeddings import (atomic_artifact_directory, git_commit, json_ready,
                                   sha256_file, verify_artifact_files)
from wimhf_sae import SparseAutoencoder, load_model


def fit_sae(train: np.ndarray, validation: np.ndarray, config: dict) -> tuple[SparseAutoencoder, dict]:
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    sae = SparseAutoencoder(
        train.shape[1], config["width"], config["top_k"],
        dead_neuron_threshold_steps=config["dead_neuron_threshold_steps"],
        prefix_lengths=config["matryoshka_prefixes"])
    history = sae.fit(
        torch.tensor(train, dtype=torch.float), torch.tensor(validation, dtype=torch.float),
        batch_size=config["batch_size"], learning_rate=config["learning_rate"],
        n_epochs=config["max_epochs"], min_epochs=config["min_epochs"],
        patience=config["patience"], show_progress=False)
    return sae, history


@torch.no_grad()
def apply_sae(sae: SparseAutoencoder, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed thresholded activations and reconstructions for every row."""
    sae.eval()
    device = next(sae.parameters()).device
    reconstruction, info = sae(torch.tensor(inputs, dtype=torch.float, device=device))
    return info["activations"].cpu().numpy(), reconstruction.cpu().numpy()


def explained_variance(inputs: np.ndarray, reconstructions: np.ndarray) -> float:
    inputs, reconstructions = np.asarray(inputs, dtype=np.float64), np.asarray(reconstructions, dtype=np.float64)
    total = np.square(inputs - inputs.mean(axis=0)).sum()
    return float(1 - np.square(inputs - reconstructions).sum() / total) if total else float("nan")


def run_grouped_sae_suite(inputs: np.ndarray, rows: pd.DataFrame, config: dict
                          ) -> tuple[dict[str, SparseAutoencoder], dict, dict]:
    """One SAE per fit group (config['fit_groups']: group -> benchmarks).

    Activations are stacked column-wise across groups and are zero for rows
    outside a feature's group; feature_labels gives 'group:feature' per column.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    if len(inputs) != len(rows) or not {"split_role", "benchmark"} <= set(rows):
        raise ValueError("embedding rows and split metadata do not align")
    groups = config["fit_groups"]
    member = {bench: group for group, benches in groups.items() for bench in benches}
    if len(member) != sum(len(benches) for benches in groups.values()):
        raise ValueError("a benchmark appears in more than one fit group")
    if unknown := set(rows["benchmark"]) - set(member):
        raise ValueError(f"benchmarks without a fit group: {sorted(unknown)}")
    saes, reports, blocks = {}, {}, []
    reconstructions = np.zeros_like(inputs)
    for group, benches in groups.items():
        mask = rows["benchmark"].isin(benches).to_numpy()
        train_mask = mask & rows["split_role"].eq("representation_train").to_numpy()
        validation_mask = mask & rows["split_role"].eq("representation_validation").to_numpy()
        if not mask.any():
            continue
        report = {"n_rows": int(mask.sum()), "n_train": int(train_mask.sum()),
                  "n_validation": int(validation_mask.sum())}
        if train_mask.sum() < config["width"] or not validation_mask.any():
            reports[group] = report | {"status": "null", "reason": "too_few_rows"}
            continue
        sae, history = fit_sae(inputs[train_mask], inputs[validation_mask], config)
        activations, group_reconstructions = apply_sae(sae, inputs[mask])
        # canonical sign: each feature's majority direction is positive (regressors are sign-invariant)
        flipped = (activations < 0).sum(axis=0) > (activations > 0).sum(axis=0)
        activations[:, flipped] *= -1
        saes[group] = sae
        reconstructions[mask] = group_reconstructions
        train_activations = activations[train_mask[mask]]
        reports[group] = report | {
            "status": "fitted", "reason": None,
            "epochs_run": len(history["train_loss"]),
            "best_validation_loss": float(min(history["val_loss"])),
            "train_explained_variance": explained_variance(
                inputs[train_mask], group_reconstructions[train_mask[mask]]),
            "validation_explained_variance": explained_variance(
                inputs[validation_mask], group_reconstructions[validation_mask[mask]]),
            "prevalence": (activations != 0).mean(axis=0).tolist(),
            "train_prevalence": (train_activations != 0).mean(axis=0).tolist(),
            "sign_flipped": flipped.tolist(),
            "negative_share": [float((column[column != 0] < 0).mean()) if (column != 0).any() else None
                               for column in activations.T],
            "dead_features": int(((activations != 0).sum(axis=0) == 0).sum()),
        }
        blocks.append((group, mask, activations))
    columns = sum(values.shape[1] for _, _, values in blocks)
    stacked = np.zeros((len(inputs), columns), dtype=np.float32)
    labels, start = [], 0
    for group, mask, values in blocks:
        stacked[np.ix_(mask, np.arange(start, start + values.shape[1]))] = values
        labels += [f"{group}:{feature}" for feature in range(values.shape[1])]
        start += values.shape[1]
    arrays = {"activations": stacked, "reconstructions": reconstructions,
              "residuals": inputs - reconstructions,
              "feature_labels": np.array(labels, dtype=str)}
    return saes, arrays, reports


def write_sae_artifacts(output_dir: str | Path, saes: dict[str, SparseAutoencoder],
                        arrays: dict[str, np.ndarray], manifest: dict,
                        *, rows: pd.DataFrame | None = None) -> dict:
    def build(stage: Path) -> dict:
        report_path = stage / "sae_report.json"
        report_path.write_text(json.dumps(
            json_ready(manifest.get("report", {})), sort_keys=True, allow_nan=False) + "\n")
        outputs = {report_path.name: sha256_file(report_path)}
        for group, sae in saes.items():
            path = stage / f"sae__{group}.pt"
            sae.save(str(path))
            outputs[path.name] = sha256_file(path)
        for name, values in arrays.items():
            path = stage / f"{name}.npy"
            values = np.asarray(values)
            np.save(path, values.astype(np.float32) if np.issubdtype(values.dtype, np.floating)
                    else values, allow_pickle=False)
            outputs[path.name] = sha256_file(path)
        if rows is not None:
            rows_path = stage / "rows.parquet"
            rows.to_parquet(rows_path, index=False)
            outputs[rows_path.name] = sha256_file(rows_path)
        return dict(manifest) | {"outputs": outputs}

    return atomic_artifact_directory(output_dir, build)


def load_sae(path: str | Path) -> SparseAutoencoder:
    return load_model(str(path))


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


def dense_identity(manifest: dict) -> dict:
    fields = (
        "family", "model", "model_files", "config_sha256", "phase1_freeze_sha256",
        "embedding_code_sha256", "git_commit", "extraction", "runtime",
    )
    if missing := [field for field in fields if field not in manifest]:
        raise ValueError(f"dense manifest is missing identity fields: {missing}")
    return {field: manifest[field] for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/workstream_b.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    rows, inputs, dense_manifest = load_dense_artifacts(args.dense_dir)
    saes, arrays, group_reports = run_grouped_sae_suite(inputs, rows, config["sae"])
    fitted = sorted(group for group, report in group_reports.items() if report["status"] == "fitted")
    report = {"status": "fitted" if fitted else "null", "fitted_groups": fitted,
              "n_features": len(arrays["feature_labels"]), "groups": group_reports}
    manifest = {
        "dense_manifest_sha256": sha256_file(args.dense_dir / "manifest.json"),
        "config_sha256": sha256_file(args.config),
        "sae_code_sha256": sha256_file(__file__),
        "wimhf_sae_code_sha256": sha256_file(Path(__file__).with_name("wimhf_sae.py")),
        "git_commit": git_commit(__file__),
        "dense_family": dense_manifest["family"],
        "dense_identity": dense_identity(dense_manifest),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "torch": torch.__version__},
        "report": report,
    }
    write_sae_artifacts(args.output, saes, arrays, manifest, rows=rows)
    for group, item in group_reports.items():
        if item["status"] == "fitted":
            print(f"{group:15s} train={item['n_train']:5d} epochs={item['epochs_run']:3d} "
                  f"val EV={item['validation_explained_variance']:.3f} dead={item['dead_features']}")
        else:
            print(f"{group:15s} {item['status']} {item['reason']}")


if __name__ == "__main__":
    main()
