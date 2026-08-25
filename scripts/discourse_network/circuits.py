"""Deterministic sparse-DAG topology and bounded circuit extraction."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Iterable, Sequence

from .edge_views import parallel_view_fields
from .nodes import stable_id


def marker_input_mask(nodes: Sequence[dict[str, Any]]) -> dict[str, bool]:
    return {"correction": any(bool(node.get("backtracking_marker_available")) for node in nodes),
            "verification": any(bool(node.get("verification_marker_available")) for node in nodes)}


def _graph(nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]):
    ids = [node["node_id"] for node in nodes]
    if ids != list(range(len(nodes))):
        raise ValueError("nodes must have chronological contiguous IDs")
    outgoing, incoming, edge_by_pair = defaultdict(list), defaultdict(list), {}
    for edge in edges:
        source, target = int(edge["source_node_id"]), int(edge["target_node_id"])
        if not 0 <= source < target < len(nodes) or (source, target) in edge_by_pair:
            raise ValueError("edges must be unique and chronological")
        outgoing[source].append(target)
        incoming[target].append(source)
        edge_by_pair[source, target] = edge
    for mapping in (outgoing, incoming):
        for values in mapping.values():
            values.sort()
    return outgoing, incoming, edge_by_pair


def _ancestors(sink: int, incoming: dict[int, list[int]]) -> set[int]:
    seen, stack = {sink}, [sink]
    while stack:
        for source in incoming.get(stack.pop(), ()):
            if source not in seen:
                seen.add(source)
                stack.append(source)
    return seen


def _reverse_distances(sink: int, incoming: dict[int, list[int]]) -> dict[int, int]:
    distance, queue = {sink: 0}, deque([sink])
    while queue:
        node = queue.popleft()
        for source in incoming.get(node, ()):
            if source not in distance:
                distance[source] = distance[node] + 1
                queue.append(source)
    return distance


def topology_features(nodes: Sequence[dict[str, Any]],
                      edges: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    outgoing, incoming, _ = _graph(nodes, edges)
    depth = [0] * len(nodes)
    for node_id in range(len(nodes)):
        if incoming.get(node_id):
            depth[node_id] = 1 + max(depth[source] for source in incoming[node_id])
    unseen, components = set(range(len(nodes))), 0
    while unseen:
        components += 1
        queue = [unseen.pop()]
        while queue:
            node = queue.pop()
            for neighbour in (*outgoing.get(node, ()), *incoming.get(node, ())):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    sinks = [node["node_id"] for node in nodes if node.get("is_final_sink")]
    final = _ancestors(sinks[0], incoming) if len(sinks) == 1 else set()
    edge_count = len(edges)
    path_counts, previous = {}, {node: 1 for node in range(len(nodes))}
    for length in range(1, 5):
        current = {node: sum(previous[target] for target in outgoing.get(node, ()))
                   for node in range(len(nodes))}
        if length >= 2:
            path_counts[f"path_{length}_count"] = sum(current.values())
        previous = current
    calls = {node.get("call_id"): node["node_id"] for node in nodes if node["kind"] == "tool_call"}
    outputs = {node.get("call_id"): node["node_id"] for node in nodes if node["kind"] == "tool_output"}
    tool_motifs = 0
    for call_id in set(calls) & set(outputs):
        call, output = calls[call_id], outputs[call_id]
        if output not in outgoing.get(call, ()):
            continue
        before = sum(nodes[source]["kind"] in {"assistant_edu", "equation", "code"}
                     for source in incoming.get(call, ()))
        after = sum(nodes[target]["kind"] in {"assistant_edu", "equation", "code"}
                    for target in outgoing.get(output, ()))
        tool_motifs += before * after
    return {
        "node_count": len(nodes), "edge_count": edge_count,
        "longest_path_edges": max(depth, default=0),
        "branch_nodes": sum(len(outgoing[node]) > 1 for node in range(len(nodes))),
        "convergence_nodes": sum(len(incoming[node]) > 1 for node in range(len(nodes))),
        "weak_components": components,
        "disconnected_nodes": sum(not outgoing[node] and not incoming[node] for node in range(len(nodes))),
        "long_range_edge_proportion": (sum(target - source > 8 for source, targets in outgoing.items()
                                           for target in targets) / edge_count if edge_count else 0.0),
        "final_ancestor_fraction": len(final) / len(nodes) if nodes else 0.0,
        "abandoned_node_proportion": (len(nodes) - len(final)) / len(nodes) if nodes else 0.0,
        "question_integration": sum(source == 0 for source, targets in outgoing.items() for _ in targets),
        "tool_feedback_integration": sum(nodes[source]["kind"] in {"tool_call", "tool_output", "user_feedback"}
                                         for source, targets in outgoing.items() for _ in targets),
        "tool_chain_motifs": tool_motifs,
        "correction_motifs": sum(bool(node.get("backtracking_marker")) for node in nodes),
        "verification_motifs": sum(bool(node.get("verification_marker")) for node in nodes),
    } | path_counts


def _induced_edges(node_ids: Iterable[int], edge_by_pair: dict[tuple[int, int], dict[str, Any]]):
    kept = set(node_ids)
    return [edge for (source, target), edge in sorted(edge_by_pair.items())
            if source in kept and target in kept]


def _bounded_neighbourhood(anchor: int, outgoing: dict[int, list[int]],
                           incoming: dict[int, list[int]], cap: int, radius: int = 1) -> list[int]:
    distance, queue = {anchor: 0}, deque([anchor])
    while queue:
        node = queue.popleft()
        if distance[node] == radius:
            continue
        for neighbour in (*incoming.get(node, ()), *outgoing.get(node, ())):
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return sorted(distance, key=lambda node: (distance[node], abs(node - anchor), node))[:cap]


def _shortest_kind_path(start: int, adjacency: dict[int, list[int]],
                        nodes: Sequence[dict[str, Any]], kinds: set[str], *,
                        reverse: bool = False) -> list[int]:
    queue, seen = deque([[start]]), {start}
    while queue:
        path = queue.popleft()
        for neighbour in adjacency.get(path[-1], ()):
            if neighbour in seen:
                continue
            candidate = path + [neighbour]
            if nodes[neighbour]["kind"] in kinds:
                return list(reversed(candidate)) if reverse else candidate
            seen.add(neighbour)
            queue.append(candidate)
    return [start]


def _bounded_tool_path(pre: list[int], post: list[int], call: int, output: int,
                       cap: int) -> list[int]:
    chosen, before, after = {call, output}, list(reversed(pre[:-1])), post[1:]
    while len(chosen) < cap and (before or after):
        if before and len(chosen) < cap:
            chosen.add(before.pop(0))
        if after and len(chosen) < cap:
            chosen.add(after.pop(0))
    return sorted(chosen)


def _record(kind: str, nodes: Sequence[dict[str, Any]], node_ids: Iterable[int],
            edges: Sequence[dict[str, Any]], config_hash: str, *, anchor_ids: Iterable[int] = (),
            truncated: bool = False, marker_gate: str | None = None) -> dict[str, Any]:
    node_ids = sorted(set(node_ids))
    edge_ids = [edge["edge_id"] for edge in edges]
    transcript_key = nodes[0]["transcript_key"]
    circuit_id = stable_id(transcript_key, kind, config_hash, node_ids, edge_ids)
    incoming, outgoing, depth = Counter(), Counter(), {node: 0 for node in node_ids}
    for edge in edges:
        source, target = edge["source_node_id"], edge["target_node_id"]
        outgoing[source] += 1
        incoming[target] += 1
        depth[target] = max(depth[target], depth[source] + 1)
    return {
        "circuit_id": circuit_id, "transcript_key": transcript_key, "extractor_kind": kind,
        "anchor_ids": list(anchor_ids), "node_ids": node_ids, "edge_ids": edge_ids,
        "node_count": len(node_ids), "edge_count": len(edge_ids), "truncated": truncated,
        "final_ancestry": any(nodes[node].get("is_final_sink", False) for node in node_ids),
        "abandoned_sibling": kind == "abandoned_branch",
        "marker_gate": marker_gate,
        "path_length": len(edges) if kind == "path" else None,
        "depth": max(depth.values(), default=0),
        "branch_count": sum(value > 1 for value in outgoing.values()),
        "convergence_count": sum(value > 1 for value in incoming.values()),
        "tool_motif": kind == "tool", "correction_motif": kind == "correction",
        "verification_motif": kind == "verification",
        "start_node_id": min(node_ids), "end_node_id": max(node_ids),
    } | parallel_view_fields(None, None, None, None)


def extract_circuits(nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]], *,
                     config_hash: str, max_local_nodes: int = 12, max_final_nodes: int = 64,
                     max_learned: int = 32) -> list[dict[str, Any]]:
    if max_local_nodes < 3 or max_final_nodes < 1 or max_learned < 1:
        raise ValueError("circuit bounds must be positive and local circuits need three nodes")
    outgoing, incoming, edge_by_pair = _graph(nodes, edges)
    circuits = []

    def add(kind: str, node_ids: Iterable[int], anchors: Iterable[int] = (), truncated: bool = False,
            marker_gate: str | None = None, circuit_edges: Sequence[dict[str, Any]] | None = None):
        node_ids = sorted(set(node_ids))
        if len(node_ids) < 2:
            return
        circuits.append(_record(kind, nodes, node_ids,
                                _induced_edges(node_ids, edge_by_pair) if circuit_edges is None else circuit_edges,
                                config_hash, anchor_ids=anchors, truncated=truncated,
                                marker_gate=marker_gate))

    def paths(path: list[int]):
        edge_length = len(path) - 1
        if 2 <= edge_length <= 4 and len(path) <= max_local_nodes:
            path_edges = [edge_by_pair[path[i], path[i + 1]] for i in range(len(path) - 1)]
            add("path", path, (path[0], path[-1]), circuit_edges=path_edges)
        if edge_length == 4:
            return
        for target in outgoing.get(path[-1], ()):
            paths(path + [target])

    for start in range(len(nodes)):
        paths([start])
    for anchor in range(len(nodes)):
        if len(outgoing.get(anchor, ())) > 1:
            add("branch", _bounded_neighbourhood(anchor, outgoing, incoming, max_local_nodes), (anchor,))
        if len(incoming.get(anchor, ())) > 1:
            add("convergence", _bounded_neighbourhood(anchor, outgoing, incoming, max_local_nodes), (anchor,))
        for marker, kind in (("backtracking_marker", "correction"),
                             ("verification_marker", "verification")):
            if nodes[anchor].get(marker):
                add(kind, _bounded_neighbourhood(anchor, outgoing, incoming, max_local_nodes, radius=2),
                    (anchor,), marker_gate=marker)

    calls = {node.get("call_id"): node["node_id"] for node in nodes if node["kind"] == "tool_call"}
    outputs = {node.get("call_id"): node["node_id"] for node in nodes if node["kind"] == "tool_output"}
    for call_id in sorted(set(calls) & set(outputs)):
        call, output = calls[call_id], outputs[call_id]
        assistant_kinds = {"assistant_edu", "equation", "code"}
        pre = _shortest_kind_path(call, incoming, nodes, assistant_kinds, reverse=True)
        post = _shortest_kind_path(output, outgoing, nodes, assistant_kinds)
        if len(pre) > 1 and len(post) > 1:
            add("tool", _bounded_tool_path(pre, post, call, output, max_local_nodes), (call, output))

    sinks = [node["node_id"] for node in nodes if node.get("is_final_sink")]
    if len(sinks) == 1:
        sink, ancestors = sinks[0], _ancestors(sinks[0], incoming)
        distance = _reverse_distances(sink, incoming)
        radius_two = sorted((node for node, steps in distance.items() if steps <= 2),
                            key=lambda node: (distance[node], node))[:max_local_nodes]
        add("conclusion_radius_two", radius_two, (sink,))
        mandatory = {node["node_id"] for node in nodes
                     if node["kind"] in {"question", "tool_call", "tool_output", "user_feedback"}} & ancestors
        if len(mandatory | {sink}) > max_final_nodes:
            raise ValueError("final-ancestry bound cannot retain required structural nodes")
        nearest = sorted(ancestors - mandatory - {sink}, key=lambda node: (distance[node], node))
        kept = mandatory | {sink} | set(nearest[:max_final_nodes - len(mandatory) - 1])
        add("final_ancestry", kept, (sink,), truncated=len(kept) < len(ancestors))
        for anchor, targets in outgoing.items():
            if anchor in ancestors:
                abandoned = [target for target in targets if target not in ancestors]
                for target in abandoned:
                    members = _bounded_neighbourhood(target, outgoing, incoming, max_local_nodes)
                    add("abandoned_branch", members, (anchor, target))

    unique = {row["circuit_id"]: row for row in circuits}
    precap_counts = Counter(row["extractor_kind"] for row in unique.values())
    precap_total = sum(precap_counts.values())
    grouped = defaultdict(list)
    for row in unique.values():
        grouped[row["extractor_kind"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["circuit_id"])
    selected, kinds = [], sorted(grouped)
    while len(selected) < max_learned and any(grouped.values()):
        for kind in kinds:
            if grouped[kind] and len(selected) < max_learned:
                selected.append(grouped[kind].pop(0))
    for row in selected:
        row["precap_count"] = precap_counts[row["extractor_kind"]]
        row["true_kind_proportion"] = precap_counts[row["extractor_kind"]] / precap_total
    return sorted(selected, key=lambda row: (row["extractor_kind"], row["circuit_id"]))
