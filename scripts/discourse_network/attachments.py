"""Sparse chronological attachment helpers; no outcome or identity inputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from .nodes import stable_id


IDENTIFIER = re.compile(r"\b(?:[A-Za-z_]\w*\d\w*|[A-Za-z_]\w*_[A-Za-z0-9_]*)\b|[=+*/<>-]+")
ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9_]+\b")
SYMBOL = re.compile(r"[=+*/<>-]+")
WORD = re.compile(r"\b[\w]+\b")
NODE_KINDS = ("question", "assistant_edu", "equation", "code", "tool_call", "tool_output",
              "user_feedback")


@dataclass(frozen=True)
class RelationGeometry:
    scaler: StandardScaler
    pca: PCA

    def transform(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        raw = _directed_pairs(source, target)
        return self.pca.transform(self.scaler.transform(raw)).astype(np.float32)


@dataclass(frozen=True)
class AttachmentHead:
    scaler: StandardScaler
    estimator: LogisticRegression
    C: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self.scaler.transform(values))

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        return self.estimator.decision_function(self.scaler.transform(values))


@dataclass(frozen=True)
class PlattScaler:
    estimator: LogisticRegression

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(np.asarray(logits).reshape(-1, 1))[:, 1]


def link_tool_edges(nodes: Sequence[dict[str, Any]], config_hash: str) -> list[dict[str, Any]]:
    calls, outputs = {}, []
    for node in nodes:
        if node["kind"] == "tool_call":
            call_id = node.get("call_id")
            if not call_id or call_id in calls:
                raise ValueError("tool calls require unique nonempty A call IDs")
            calls[call_id] = node
        elif node["kind"] == "tool_output":
            outputs.append(node)
    edges = []
    for output in outputs:
        call_id, target = output.get("call_id"), output["node_id"]
        call = calls.get(call_id)
        if call is None:
            raise ValueError(f"tool output has no earlier call with ID {call_id!r}")
        source = call["node_id"]
        if source >= target:
            raise ValueError("tool structural edges must be chronological")
        edges.append({
            "edge_id": stable_id(nodes[0]["transcript_key"], source, target, "structural", config_hash),
            "transcript_key": nodes[0]["transcript_key"], "source_node_id": source,
            "target_node_id": target, "edge_source": "structural", "structural_link_key": call_id,
            "accepted": True, "logit": None, "probability": None, "rank": None,
        })
    return sorted(edges, key=lambda row: (row["target_node_id"], row["source_node_id"]))


def _cosine(values: np.ndarray, target: int, candidates: list[int]) -> dict[int, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or target >= len(values) or not np.isfinite(values).all():
        raise ValueError("one finite embedding vector is required per node")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError("node embeddings must be nonzero")
    return {index: float(values[index] @ values[target] / (norms[index] * norms[target]))
            for index in candidates}


def _tokens(text: str, pattern: re.Pattern[str]) -> set[str]:
    return {token.casefold() for token in pattern.findall(text)}


def build_candidates(nodes: Sequence[dict[str, Any]], embeddings: np.ndarray,
                     target_node_id: int, *, cap: int = 32, preceding: int = 8,
                     semantic: int = 8, identifiers: int = 8,
                     structural: int = 4, reservoir_multiplier: int = 1) -> list[dict[str, Any]]:
    if not 1 <= cap or target_node_id <= 0 or reservoir_multiplier < 1:
        raise ValueError("candidate cap and target node ID must be positive")
    by_id = {node["node_id"]: node for node in nodes}
    if set(by_id) != set(range(len(nodes))) or target_node_id not in by_id:
        raise ValueError("nodes must have contiguous chronological IDs")
    if by_id[target_node_id].get("target_eligible") is not True:
        raise ValueError("attachment targets must be explicitly target_eligible")
    earlier = list(range(target_node_id))
    scores = _cosine(embeddings, target_node_id, earlier)
    reasons: dict[int, set[str]] = {}
    order: list[int] = []

    def add(indices: Iterable[int], reason: str) -> None:
        for index in indices:
            if index >= target_node_id:
                raise ValueError("candidate generation produced a nonchronological edge")
            if index not in reasons:
                reasons[index] = set()
                order.append(index)
            reasons[index].add(reason)

    primary_preceding = list(range(max(0, target_node_id - preceding), target_node_id))
    preceding_rank = primary_preceding + list(range(
        max(0, target_node_id - preceding * reservoir_multiplier),
        max(0, target_node_id - preceding)))
    semantic_rank = sorted(earlier, key=lambda index: (-scores[index], -index))
    target_text = by_id[target_node_id]["rendered_text"]
    target_identifiers = _tokens(target_text, IDENTIFIER)
    target_entities = _tokens(target_text, ENTITY)
    target_symbols = _tokens(target_text, SYMBOL)
    target_words = _tokens(target_text, WORD)
    audit = {}
    for index in earlier:
        text = by_id[index]["rendered_text"]
        identifiers_here = _tokens(text, IDENTIFIER)
        entities_here = _tokens(text, ENTITY)
        symbols_here = _tokens(text, SYMBOL)
        words_here = _tokens(text, WORD)
        union = target_words | words_here
        audit[index] = {
            "identifier_overlap": len(target_identifiers & identifiers_here),
            "entity_overlap": len(target_entities & entities_here),
            "symbol_overlap": len(target_symbols & symbols_here),
            "lexical_overlap": len(target_words & words_here) / len(union) if union else 0.0,
        }
    overlap = [(sum(audit[index][name] for name in
                    ("identifier_overlap", "entity_overlap", "symbol_overlap")), index)
               for index in earlier]
    ranked_overlap = [index for count, index in sorted(
        overlap, key=lambda pair: (-pair[0], -pair[1])) if count > 0]
    structural_ids = [index for index in reversed(earlier)
                      if by_id[index]["kind"] in {"tool_call", "tool_output", "user_feedback"}]
    ranked = ((preceding_rank, preceding, "preceding"),
              ([node["node_id"] for node in nodes[:target_node_id]
                if node["kind"] == "question"], 1, "question"),
              (semantic_rank, semantic, "semantic"),
              (ranked_overlap, identifiers, "identifier"),
              (structural_ids, structural, "structural_context"))
    for values, quota, reason in ranked:
        add(values[:quota], reason)
    for values, quota, reason in ranked:
        add(values[quota:quota * reservoir_multiplier], reason)
    for index in earlier:
        for field, reason in (("identifier_overlap", "shared_identifier"),
                              ("entity_overlap", "shared_entity"),
                              ("symbol_overlap", "shared_symbol")):
            if audit[index][field] and index in reasons:
                reasons[index].add(reason)

    selected = order[:cap]
    candidate_rank = {node_id: rank for rank, node_id in enumerate(selected, 1)}
    return [{
        "transcript_key": by_id[index]["transcript_key"], "source_node_id": index,
        "target_node_id": target_node_id, "node_distance": target_node_id - index,
        "adjacent": target_node_id - index == 1, "semantic_cosine": scores[index],
        **audit[index],
        "candidate_rank": candidate_rank[index],
        "reasons": sorted(reasons[index]),
    } for index in sorted(selected)]


def _directed_pairs(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or not source.shape[0]:
        raise ValueError("source and target must be nonempty matrices with equal shape")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("relation inputs must be finite")
    return np.concatenate((source, target, target - source, source * target), axis=1)


def fit_relation_geometry(source: np.ndarray, target: np.ndarray, *, n_components: int = 256,
                          seed: int = 0, pair_keys: Sequence[str] | None = None,
                          max_pairs: int = 1_000_000) -> RelationGeometry:
    """Fit only standardisation/PCA to node embeddings; no labels or metadata enter."""
    raw = _directed_pairs(source, target)
    if pair_keys is not None:
        if len(pair_keys) != len(raw):
            raise ValueError("pair_keys must align with relation pairs")
        order = sorted(range(len(raw)), key=lambda i: stable_id("pair", pair_keys[i]))[:max_pairs]
        raw = raw[order]
    elif len(raw) > max_pairs:
        raise ValueError("pair_keys are required for deterministic relation-pair sampling")
    scaler = StandardScaler().fit(raw)
    scaled = scaler.transform(raw)
    if n_components < 1 or scaled.shape[0] < n_components or scaled.shape[1] < n_components:
        raise ValueError(f"relation PCA requires exactly {n_components} components")
    return RelationGeometry(scaler, PCA(n_components=n_components, random_state=seed).fit(scaled))


def attachment_features(dense: np.ndarray, candidates: Sequence[dict[str, Any]],
                        nodes: Sequence[dict[str, Any]], *, mode: str = "full") -> np.ndarray:
    """Build the three frozen attachment controls without leaking candidate reasons."""
    dense = np.asarray(dense, dtype=float)
    if mode not in {"full", "dense", "no_text"} or dense.ndim != 2 or len(dense) != len(candidates):
        raise ValueError("invalid attachment feature mode or row alignment")
    if mode == "dense":
        return dense
    kind_index = {kind: number for number, kind in enumerate(NODE_KINDS)}
    structural = []
    controls = []
    for row in candidates:
        try:
            source_kind = kind_index[nodes[row["source_node_id"]]["kind"]]
            target_kind = kind_index[nodes[row["target_node_id"]]["kind"]]
        except (KeyError, IndexError) as exc:
            raise ValueError("candidate references an unknown node kind or ID") from exc
        kinds = np.zeros(2 * len(NODE_KINDS))
        kinds[source_kind], kinds[len(NODE_KINDS) + target_kind] = 1.0, 1.0
        structural.append(np.r_[kinds, math.log1p(row["node_distance"]), float(row["adjacent"])])
        controls.append([float(row.get("lexical_overlap", 0.0)),
                         float(row.get("symbol_overlap", 0.0)),
                         float(row.get("entity_overlap", 0.0))])
    structural = np.asarray(structural)
    return np.concatenate((dense, structural), axis=1) if mode == "full" else np.concatenate(
        (structural, np.asarray(controls)), axis=1)


def fit_attachment_head(values: np.ndarray, labels: np.ndarray, groups: np.ndarray, *,
                        cs: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
                        min_positive: int = 200) -> AttachmentHead:
    values, labels, groups = np.asarray(values, dtype=float), np.asarray(labels), np.asarray(groups)
    if values.ndim != 2 or len(values) != len(labels) or len(labels) != len(groups):
        raise ValueError("attachment values, labels, and item groups must align")
    if set(np.unique(labels)) != {0, 1} or len(np.unique(groups)) < 2:
        raise ValueError("attachment fitting needs both classes and at least two item groups")
    if int(labels.sum()) < min_positive:
        raise ValueError(f"attachment pilot is underpowered: need {min_positive} positive edges")
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    scored = []
    for c in cs:
        fold_scores = []
        for train, test in folds.split(values, labels, groups):
            if len(np.unique(labels[train])) < 2:
                continue
            scaler = StandardScaler().fit(values[train])
            model = LogisticRegression(C=float(c), solver="lbfgs", random_state=0)
            model.fit(scaler.transform(values[train]), labels[train])
            fold_scores.append(f1_score(labels[test], model.predict(scaler.transform(values[test])),
                                        zero_division=0))
        if fold_scores:
            scored.append((float(np.mean(fold_scores)), -float(c), float(c)))
    if not scored:
        raise ValueError("item-grouped folds never contained both training classes")
    chosen = max(scored)[2]
    scaler = StandardScaler().fit(values)
    estimator = LogisticRegression(C=chosen, solver="lbfgs", random_state=0)
    estimator.fit(scaler.transform(values), labels)
    return AttachmentHead(scaler, estimator, chosen)


def fit_platt_scaler(logits: np.ndarray, labels: np.ndarray) -> PlattScaler:
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels)
    if logits.ndim != 1 or len(logits) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Platt calibration needs aligned scalar logits with both classes")
    estimator = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
    estimator.fit(logits[:, None], labels)
    return PlattScaler(estimator)


def select_threshold(probabilities: np.ndarray, labels: np.ndarray, *,
                     precision_floor: float = 0.65, recall_floor: float = 0.50) -> float:
    probabilities, labels = np.asarray(probabilities, dtype=float), np.asarray(labels)
    if probabilities.ndim != 1 or len(probabilities) != len(labels) or not len(labels):
        raise ValueError("threshold inputs must be aligned nonempty vectors")
    choices = []
    for threshold in sorted(set(float(value) for value in probabilities)):
        predicted = probabilities >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0)
        if precision >= precision_floor and recall >= recall_floor:
            choices.append((f1, precision, threshold))
    if not choices:
        raise ValueError("no threshold meets the precision/recall floors")
    return max(choices)[2]


def select_parents(candidates: Sequence[dict[str, Any]], *, threshold: float,
                   max_parents: int = 3) -> list[dict[str, Any]]:
    if max_parents < 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid threshold or maximum-parent setting")
    accepted = [row for row in candidates if float(row["probability"]) >= threshold]
    ranked = sorted(accepted, key=lambda row: (-float(row["probability"]), row["source_node_id"]))[:max_parents]
    return [dict(row) | {"rank": rank, "accepted": True, "threshold": threshold,
                         "maximum_parents": max_parents}
            for rank, row in enumerate(ranked, 1)]


def full_earlier_baselines(nodes: Sequence[dict[str, Any]], target_node_id: int) -> list[dict[str, Any]]:
    """Score every earlier node; baseline coverage is never constrained by neural candidates."""
    if not 0 < target_node_id < len(nodes):
        raise ValueError("baseline target must have earlier nodes")
    texts = [str(node["rendered_text"]) for node in nodes[:target_node_id + 1]]
    matrix = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b").fit_transform(texts)
    tfidf = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    target = texts[-1]
    target_symbols, target_entities = _tokens(target, SYMBOL), _tokens(target, ENTITY)
    target_identifiers = _tokens(target, IDENTIFIER)
    return [{
        "source_node_id": source, "target_node_id": target_node_id,
        "recency": 1.0 / (target_node_id - source), "tfidf": float(tfidf[source]),
        "shared_symbols": len(target_symbols & _tokens(texts[source], SYMBOL)),
        "shared_entities": len(target_entities & _tokens(texts[source], ENTITY)),
        "shared_identifiers": len(target_identifiers & _tokens(texts[source], IDENTIFIER)),
    } for source in range(target_node_id)]


def candidate_recall(gold_parents: Iterable[int], candidate_parents: Iterable[int]) -> float:
    gold, candidates = set(gold_parents), set(candidate_parents)
    if not gold:
        raise ValueError("candidate recall requires at least one gold parent")
    return len(gold & candidates) / len(gold)


def parent_cap_audit(gold_parent_sets: Iterable[Iterable[int]], cap: int = 3) -> float:
    sets = [set(parents) for parents in gold_parent_sets]
    if not sets or cap < 1:
        raise ValueError("parent-cap audit needs targets and a positive cap")
    return sum(len(parents) > cap for parents in sets) / len(sets)


def transfer_metrics(labels: Sequence[int], predictions: Sequence[int],
                     subsets: Sequence[str]) -> dict[str, dict[str, float]]:
    labels, predictions, subsets = np.asarray(labels), np.asarray(predictions), np.asarray(subsets)
    if not len(labels) or len(labels) != len(predictions) or len(labels) != len(subsets):
        raise ValueError("transfer metrics require aligned nonempty inputs")

    def score(mask):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[mask], predictions[mask], average="binary", zero_division=0)
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
                "n": int(mask.sum())}

    result = {"overall": score(np.ones(len(labels), dtype=bool))}
    result.update({str(subset): score(subsets == subset) for subset in sorted(set(subsets))})
    return result


def topology_sensitivity(primary: dict[str, Sequence[float]],
                         alternatives: Sequence[dict[str, Sequence[float]]]) -> dict[str, float]:
    """Minimum transcript-level Spearman rho for each nonconstant topology statistic."""
    result = {}
    for name, reference in primary.items():
        reference = np.asarray(reference, dtype=float)
        if len(reference) < 2 or np.all(reference == reference[0]):
            continue
        correlations = []
        for alternative in alternatives:
            values = np.asarray(alternative[name], dtype=float)
            if values.shape != reference.shape or np.all(values == values[0]):
                correlations.append(0.0)
            else:
                statistic = float(spearmanr(reference, values).statistic)
                correlations.append(statistic if np.isfinite(statistic) else 0.0)
        result[name] = min(correlations) if correlations else 1.0
    return result


def enforce_attachment_gates(*, gold_parent_sets: Sequence[Iterable[int]],
                             candidate_parent_sets: Sequence[Iterable[int]],
                             labels: Sequence[int], predictions: Sequence[int],
                             subsets: Sequence[str], primary_topology: dict[str, Sequence[float]],
                             alternative_topologies: Sequence[dict[str, Sequence[float]]],
                             candidate_floor: float = 0.95, precision_floor: float = 0.65,
                             recall_floor: float = 0.50, subset_precision: float = 0.55,
                             subset_recall: float = 0.40, topology_floor: float = 0.80,
                             parent_cap: int = 3, over_cap_ceiling: float = 0.05) -> dict[str, Any]:
    if len(gold_parent_sets) != len(candidate_parent_sets) or not gold_parent_sets:
        raise ValueError("candidate-gate parent sets must align")
    recalls = [candidate_recall(gold, candidates)
               for gold, candidates in zip(gold_parent_sets, candidate_parent_sets)]
    transfer = transfer_metrics(labels, predictions, subsets)
    sensitivity = topology_sensitivity(primary_topology, alternative_topologies)
    over_cap = parent_cap_audit(gold_parent_sets, parent_cap)
    failures = []
    if np.mean(recalls) < candidate_floor:
        failures.append("candidate recall")
    if (transfer["overall"]["precision"] < precision_floor or
            transfer["overall"]["recall"] < recall_floor):
        failures.append("pooled attachment transfer")
    if any(metrics["precision"] < subset_precision or metrics["recall"] < subset_recall
           for name, metrics in transfer.items() if name != "overall"):
        failures.append("subset attachment transfer")
    if any(value < topology_floor for value in sensitivity.values()):
        failures.append("topology sensitivity")
    if over_cap > over_cap_ceiling:
        failures.append("gold parent-cap audit")
    if failures:
        raise ValueError(f"attachment gates failed: {', '.join(failures)}")
    return {"candidate_recall": float(np.mean(recalls)), "transfer": transfer,
            "topology_spearman": sensitivity, "over_parent_cap": over_cap}
