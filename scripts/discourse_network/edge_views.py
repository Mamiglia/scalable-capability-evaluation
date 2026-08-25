"""Pure support, Top-K, and exposure-adjusted helpers shared by edge/circuit views."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def topk_nonnegative(values: np.ndarray, k: int) -> list[tuple[int, float]]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or k < 1 or not np.isfinite(values).all():
        raise ValueError("Top-K input must be one finite vector and k must be positive")
    active = [(index, float(value)) for index, value in enumerate(values) if value > 0]
    return sorted(active, key=lambda pair: (-pair[1], pair[0]))[:k]


def retained_support(assignments: Iterable[int], minimum: int) -> dict[int, int]:
    if minimum < 1:
        raise ValueError("minimum support must be positive")
    return {int(key): count for key, count in Counter(assignments).items() if count >= minimum}


def support_floor(population: int, rate: float) -> int:
    if population < 0 or rate < 0:
        raise ValueError("support population and rate must be nonnegative")
    return max(50, math.ceil(rate * population))


def retained_across_splits(train: Iterable[int], validation: Iterable[int],
                           minimum_train: int, minimum_validation: int) -> dict[int, tuple[int, int]]:
    train_counts, validation_counts = Counter(train), Counter(validation)
    return {int(key): (count, validation_counts[key]) for key, count in train_counts.items()
            if count >= minimum_train and validation_counts[key] >= minimum_validation}


def aggregate_activations(rows: Iterable[dict[str, Any]], *,
                          inventory: Iterable[dict[str, Any]] | None = None,
                          feature_ids: Iterable[int] | None = None,
                          ) -> dict[int, dict[str, float]]:
    """Summarise sparse features with exposure-adjusted statistics, never sums/maxima."""
    rows = list(rows)
    if not rows and inventory is None:
        return {}
    inventory = list(inventory) if inventory is not None else None
    ancestry = ({str(row["unit_id"]): bool(row["in_final_ancestry"]) for row in inventory}
                if inventory is not None else {})
    if inventory is not None and len(ancestry) != len(inventory):
        raise ValueError("duplicate unit inventory key")
    values: dict[int, dict[str, float]] = defaultdict(dict)
    activation_keys = set()
    for row in rows:
        unit, feature = str(row["unit_id"]), int(row["feature_id"])
        key = (unit, row.get("sae_fit_id"), row.get("dictionary_id"), feature)
        if key in activation_keys:
            raise ValueError("duplicate activation key")
        activation_keys.add(key)
        active = float(row["activation"])
        if not np.isfinite(active) or active < 0:
            raise ValueError("SAE activations must be finite and nonnegative")
        if inventory is not None and unit not in ancestry:
            raise ValueError("activation unit is absent from the complete inventory")
        flag = ancestry.get(unit, bool(row.get("in_final_ancestry", False)))
        if unit in ancestry and ancestry[unit] != flag:
            raise ValueError("unit ancestry flag is inconsistent")
        ancestry[unit], values[feature][unit] = flag, active
    units = sorted(ancestry)
    feature_ids = set(values) if feature_ids is None else {int(value) for value in feature_ids}
    result = {}
    for feature in sorted(feature_ids):
        observed = values.get(feature, {})
        vector = np.array([observed.get(unit, 0.0) for unit in units])
        inside = vector[[ancestry[unit] for unit in units]]
        outside = vector[[not ancestry[unit] for unit in units]]
        result[feature] = {
            "mean": float(vector.mean()),
            "proportion_active": float(np.mean(vector > 0)),
            "median": float(np.quantile(vector, 0.5)),
            "q90": float(np.quantile(vector, 0.9)),
            "ancestry_contrast": float(inside.mean() - outside.mean())
            if len(inside) and len(outside) else 0.0,
        }
    return result


@dataclass(frozen=True)
class ScaledPCA:
    scaler: StandardScaler
    pca: PCA

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(np.asarray(values, dtype=float))).astype(np.float32)


def fit_scaled_pca(values: np.ndarray, components: int, *, seed: int = 0,
                   canonical: bool = False) -> ScaledPCA:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("PCA input must be a nonempty finite matrix")
    if components < 1 or values.shape[0] < components or values.shape[1] < components:
        label = "exactly 256" if canonical and components == 256 else f"exactly {components}"
        raise ValueError(f"PCA requires {label} components")
    scaler = StandardScaler().fit(values)
    return ScaledPCA(scaler, PCA(n_components=components, random_state=seed).fit(
        scaler.transform(values)))


@dataclass(frozen=True)
class SphericalClusters:
    estimator: KMeans

    def predict(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("spherical clustering requires nonzero rows")
        return self.estimator.predict(values / norms)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        values = values / np.linalg.norm(values, axis=1, keepdims=True)
        centers = self.estimator.cluster_centers_
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        logits = values @ centers.T
        logits -= logits.max(axis=1, keepdims=True)
        scores = np.exp(logits)
        return scores / scores.sum(axis=1, keepdims=True)


def fit_spherical_clusters(values: np.ndarray, clusters: int, *, seed: int = 0) -> SphericalClusters:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or clusters < 2 or len(values) < clusters or not np.isfinite(values).all():
        raise ValueError("invalid spherical-clustering population or cluster count")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("spherical clustering requires nonzero rows")
    return SphericalClusters(KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit(
        values / norms))


@dataclass(frozen=True)
class TopKSAE:
    decoder: np.ndarray
    top_k: int
    fit_id: str
    dictionary_id: str
    trainer: str = "torch_minibatch_topk"

    @classmethod
    def fit(cls, values: np.ndarray, *, width: int, top_k: int, seed: int = 0,
            iterations: int = 20, batch_size: int = 256, device: str = "cpu",
            dtype: str = "float32", learning_rate: float = 1e-3) -> "TopKSAE":
        values = np.asarray(values, dtype=float)
        if (values.ndim != 2 or not len(values) or width <= values.shape[1] or
                not 0 < top_k < width or iterations < 1 or not np.isfinite(values).all()):
            raise ValueError("Top-K SAE requires finite rows, an overcomplete width, and 0 < k < width")
        torch.manual_seed(seed)
        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(dtype)
        if torch_dtype is None or batch_size < 1:
            raise ValueError("Top-K SAE dtype/batch size is unsupported")
        target_device = torch.device(device)
        data = torch.as_tensor(values, dtype=torch_dtype, device=target_device)
        decoder = torch.nn.Parameter(torch.randn(width, values.shape[1], dtype=torch_dtype,
                                                 device=target_device))
        with torch.no_grad():
            decoder.copy_(torch.nn.functional.normalize(decoder.float(), dim=1).to(torch_dtype))
        optimizer = torch.optim.Adam([decoder], lr=learning_rate)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(iterations):
            for indices in torch.randperm(len(data), generator=generator).split(batch_size):
                batch = data[indices.to(target_device)]
                scores = torch.relu(batch @ decoder.T)
                top = torch.topk(scores, top_k, dim=1)
                activations = torch.zeros_like(scores).scatter(1, top.indices, top.values)
                loss = torch.mean((batch - activations @ decoder) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    decoder.copy_(torch.nn.functional.normalize(decoder.float(), dim=1).to(torch_dtype))
        decoder = decoder.detach().float().cpu().numpy()
        payload = decoder.astype(np.float32).tobytes()
        import hashlib
        dictionary_id = hashlib.sha256(payload).hexdigest()
        fit_id = hashlib.sha256(f"{dictionary_id}:{top_k}:{seed}".encode()).hexdigest()
        return cls(decoder.astype(np.float32), top_k, fit_id, dictionary_id)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.decoder.shape[1]:
            raise ValueError("SAE apply matrix has the wrong shape")
        return _topk_matrix(np.maximum(values @ self.decoder.T, 0.0), self.top_k).astype(np.float32)

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        return self.transform(values) @ self.decoder


def _topk_matrix(values: np.ndarray, k: int) -> np.ndarray:
    if k >= values.shape[1]:
        return values
    keep = np.argpartition(values, -k, axis=1)[:, -k:]
    result = np.zeros_like(values)
    rows = np.arange(len(values))[:, None]
    result[rows, keep] = values[rows, keep]
    return result


def hungarian_match(canonical: np.ndarray, refit: np.ndarray) -> dict[str, np.ndarray]:
    canonical, refit = np.asarray(canonical, dtype=float), np.asarray(refit, dtype=float)
    if canonical.shape != refit.shape or canonical.ndim != 2:
        raise ValueError("decoder dictionaries must have equal matrix shapes")
    canonical = canonical / np.linalg.norm(canonical, axis=1, keepdims=True)
    refit = refit / np.linalg.norm(refit, axis=1, keepdims=True)
    cosine = canonical @ refit.T
    left, right = linear_sum_assignment(-cosine)
    order = np.argsort(left)
    return {"canonical": left[order], "refit": right[order],
            "cosines": cosine[left[order], right[order]]}


def retrieval_precision_at_k(values: np.ndarray, labels: np.ndarray, k: int) -> float:
    values, labels = np.asarray(values, dtype=float), np.asarray(labels)
    if values.ndim != 2 or len(values) != len(labels) or not 0 < k < len(values):
        raise ValueError("retrieval inputs must align and 0 < k < population")
    normalized = values / np.linalg.norm(values, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    neighbours = np.argpartition(similarity, -k, axis=1)[:, -k:]
    return float(np.mean(labels[neighbours] == labels[:, None]))


def cluster_metrics(values: np.ndarray, assignments: np.ndarray,
                    refit_assignments: Iterable[np.ndarray] = ()) -> dict[str, float]:
    values, assignments = np.asarray(values, dtype=float), np.asarray(assignments)
    if values.ndim != 2 or len(values) != len(assignments) or len(set(assignments)) < 2:
        raise ValueError("cluster metrics require aligned rows and at least two clusters")
    stability = [adjusted_rand_score(assignments, np.asarray(refit))
                 for refit in refit_assignments]
    return {"silhouette": float(silhouette_score(values, assignments, metric="cosine")),
            "median_adjusted_rand": float(np.median(stability)) if stability else 1.0}


def sae_metrics(values: np.ndarray, sae: TopKSAE,
                matched_cosines: Iterable[np.ndarray] = ()) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    activations, reconstruction = sae.transform(values), sae.reconstruct(values)
    random = np.random.default_rng(0).normal(size=sae.decoder.shape)
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    random_activation = _topk_matrix(np.maximum(values @ random.T, 0.0), sae.top_k)
    random_mse = float(np.mean((values - random_activation @ random) ** 2))
    matched_cosines = [np.asarray(row, dtype=float) for row in matched_cosines]
    cosines = np.concatenate(matched_cosines) if matched_cosines else np.array([])
    active = (activations > 0).sum(axis=1)
    return {"reconstruction_mse": float(np.mean((values - reconstruction) ** 2)),
            "random_dictionary_mse": random_mse,
            "median_active": float(np.median(active)),
            "fraction_three_active": float(np.mean(active >= 3)),
            "median_matched_cosine": float(np.median(cosines)) if len(cosines) else 1.0}


def fit_sae_refits(values: np.ndarray, *, width: int, top_k: int,
                   groups: np.ndarray, seeds: Iterable[int] = (1, 2, 3, 4, 5), iterations: int = 20,
                   batch_size: int = 256, device: str = "cpu", dtype: str = "float32",
                   learning_rate: float = 1e-3
                   ) -> tuple[TopKSAE, list[TopKSAE], list[dict[str, np.ndarray]]]:
    """Canonical fit plus five seeded/bootstrap refits and signed-cosine matching."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if len(groups) != len(values):
        raise ValueError("SAE refits require one opaque item group per row")
    canonical = TopKSAE.fit(values, width=width, top_k=top_k, seed=0, iterations=iterations,
                            batch_size=batch_size, device=device, dtype=dtype,
                            learning_rate=learning_rate)
    refits, matches = [], []
    for seed in seeds:
        sample_indices = bootstrap_group_indices(groups, seed)
        sample = values[sample_indices]
        refit = TopKSAE.fit(sample, width=width, top_k=top_k, seed=seed, iterations=iterations,
                            batch_size=batch_size, device=device, dtype=dtype,
                            learning_rate=learning_rate)
        refits.append(refit)
        matches.append(hungarian_match(canonical.decoder, refit.decoder))
    return canonical, refits, matches


def fit_cluster_refits(values: np.ndarray, clusters: int,
                       groups: np.ndarray,
                       seeds: Iterable[int] = (1, 2, 3, 4, 5)) -> tuple[SphericalClusters, list[float]]:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if len(groups) != len(values):
        raise ValueError("cluster refits require one opaque item group per row")
    canonical = fit_spherical_clusters(values, clusters, seed=0)
    labels = canonical.predict(values)
    stability = []
    for seed in seeds:
        sample_indices = bootstrap_group_indices(groups, seed)
        refit = fit_spherical_clusters(values[sample_indices], clusters, seed=seed)
        stability.append(float(adjusted_rand_score(labels, refit.predict(values))))
    return canonical, stability


def bootstrap_group_indices(groups: np.ndarray, seed: int) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.array(sorted(set(groups.tolist())), dtype=object)
    if not len(unique):
        raise ValueError("group bootstrap needs opaque item groups")
    sampled = np.random.default_rng(seed).choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def resolve_view_mask(*, dense_pass: bool, cluster_pass: bool | None,
                      sae_pass: bool | None) -> tuple[dict[str, str], dict[str, bool]]:
    if not dense_pass:
        statuses = {"dense": "failed", "cluster": "null", "sae": "null"}
    else:
        statuses = {"dense": "pass", "cluster": "pass" if cluster_pass else "failed",
                    "sae": "pass" if sae_pass else "failed"}
    return statuses, {name: status == "pass" for name, status in statuses.items()}


def fit_independent_views(train: np.ndarray, validation: np.ndarray, groups: np.ndarray,
                          settings: dict[str, Any], gates: dict[str, Any], *, dense_pass: bool,
                          seed: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {"clusters": None, "sae": None, "errors": {}}
    if not dense_pass:
        result["statuses"], result["input_mask"] = resolve_view_mask(
            dense_pass=False, cluster_pass=None, sae_pass=None)
        return result
    cluster_pass = sae_pass = False
    try:
        choices = []
        for count in settings["cluster_counts"]:
            if count > len(train) or count > len(validation):
                continue
            candidate = fit_spherical_clusters(train, count, seed=seed)
            train_labels, validation_labels = candidate.predict(train), candidate.predict(validation)
            support = retained_across_splits(
                train_labels, validation_labels,
                support_floor(len(train), settings["cluster_support_rate"]),
                support_floor(len(validation), settings["cluster_support_rate"]))
            if len(support) == count and len(set(validation_labels)) >= 2:
                choices.append((cluster_metrics(validation, validation_labels)["silhouette"], -count))
        if not choices:
            raise ValueError("no cluster count passes validation support")
        count = -max(choices)[1]
        clusters, stability = fit_cluster_refits(
            train, count, groups=groups, seeds=range(1, settings["refits"] + 1))
        if np.median(stability) < gates["cluster_adjusted_rand"]:
            raise ValueError("cluster bootstrap stability gate failed")
        result.update(clusters=clusters, cluster_stability=stability)
        cluster_pass = True
    except (ValueError, RuntimeError) as error:
        result["errors"]["cluster"] = str(error)
    try:
        sae, refits, matches = fit_sae_refits(
            train, width=settings["sae_width"], top_k=settings["sae_top_k"], groups=groups,
            seeds=range(1, settings["refits"] + 1), iterations=settings["sae_epochs"],
            batch_size=settings["sae_batch_size"], device=settings["sae_device"],
            dtype=settings["sae_dtype"], learning_rate=settings["sae_learning_rate"])
        stable = {feature for feature in range(len(sae.decoder))
                  if sum(match["cosines"][feature] >= gates["matched_decoder_cosine"]
                         for match in matches) >= gates["matched_refits_required"]}
        train_active, validation_active = sae.transform(train), sae.transform(validation)
        supported = retained_across_splits(
            [feature for row in train_active for feature in np.flatnonzero(row)],
            [feature for row in validation_active for feature in np.flatnonzero(row)],
            support_floor(len(train), settings["sae_support_rate"]),
            support_floor(len(validation), settings["sae_support_rate"]))
        stable &= set(supported)
        metrics = sae_metrics(validation, sae, [match["cosines"] for match in matches])
        if (not stable or metrics["reconstruction_mse"] >= metrics["random_dictionary_mse"] or
                metrics["median_active"] < gates["edge_sae_median_active_features"] or
                metrics["fraction_three_active"] < gates["edge_sae_fraction_with_three_features"]):
            raise ValueError("SAE validation reconstruction/support/overlap gate failed")
        result.update(sae=sae, sae_refits=refits, matches=matches, stable_features=stable,
                      sae_validation_metrics=metrics)
        sae_pass = True
    except (ValueError, RuntimeError) as error:
        result["errors"]["sae"] = str(error)
    result["statuses"], result["input_mask"] = resolve_view_mask(
        dense_pass=True, cluster_pass=cluster_pass, sae_pass=sae_pass)
    return result


def aggregate_dense(vectors: np.ndarray, in_final_ancestry: Iterable[bool]) -> dict[str, np.ndarray]:
    vectors = np.asarray(vectors, dtype=float)
    flags = np.asarray(list(in_final_ancestry), dtype=bool)
    if vectors.ndim != 2 or not len(vectors) or len(vectors) != len(flags) or not np.isfinite(vectors).all():
        raise ValueError("dense rows and ancestry flags must be aligned, finite, and nonempty")
    inside, outside = vectors[flags], vectors[~flags]
    return {
        "mean": vectors.mean(axis=0), "median": np.quantile(vectors, 0.5, axis=0),
        "q10": np.quantile(vectors, 0.1, axis=0), "q90": np.quantile(vectors, 0.9, axis=0),
        "ancestry_contrast": inside.mean(axis=0) - outside.mean(axis=0)
        if len(inside) and len(outside) else np.zeros(vectors.shape[1]),
    }


def weighted_kind_summary(summaries: dict[str, np.ndarray],
                          true_proportions: dict[str, float]) -> np.ndarray:
    if set(summaries) != set(true_proportions) or not summaries:
        raise ValueError("each extractor-kind summary needs one true pre-cap proportion")
    weights = np.array([true_proportions[key] for key in sorted(summaries)], dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("extractor-kind proportions must be nonnegative and sum to one")
    matrix = np.stack([np.asarray(summaries[key], dtype=float) for key in sorted(summaries)])
    if not np.isfinite(matrix).all():
        raise ValueError("extractor-kind summaries must be finite")
    return np.average(matrix, axis=0, weights=weights)


def parallel_view_fields(dense_row: int | None, cluster_id: int | None,
                         sae_fit_id: str | None, dictionary_id: str | None,
                         cluster_probability: float | None = None) -> dict[str, Any]:
    """Keep dense, cluster, and SAE references as siblings; none selects another."""
    return {"dense_row": dense_row, "cluster_id": cluster_id,
            "cluster_probability": cluster_probability, "sae_fit_id": sae_fit_id,
            "dictionary_id": dictionary_id}
