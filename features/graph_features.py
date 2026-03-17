from __future__ import annotations

from collections import deque
from math import comb
from typing import Any

import pandas as pd

from schema.event import Event


GRAPH_FEATURE_COLUMNS = [
    "graph_node_count",
    "graph_edge_count",
    "graph_density",
    "graph_avg_out_degree",
    "graph_max_out_degree",
    "graph_root_count",
    "graph_component_count",
    "graph_max_depth",
    "graph_cross_community_ratio",
    "graph_structural_virality",
]

_METADATA_FEATURE_KEYS = {
    "graph_node_count": ["node_count", "num_nodes", "n_nodes", "cascade_nodes"],
    "graph_edge_count": ["edge_count", "num_edges", "n_edges", "cascade_edges"],
    "graph_density": ["density", "graph_density", "cascade_density"],
    "graph_avg_out_degree": ["avg_out_degree", "average_out_degree"],
    "graph_max_out_degree": ["max_out_degree"],
    "graph_root_count": ["root_count", "num_roots"],
    "graph_component_count": ["component_count", "num_components"],
    "graph_max_depth": ["max_depth", "cascade_depth", "structural_depth"],
    "graph_cross_community_ratio": ["cross_community_ratio", "community_crossing_ratio"],
    "graph_structural_virality": ["structural_virality", "wiener_index_normalized"],
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metadata_features(event: Event) -> dict[str, float]:
    features: dict[str, float] = {}
    for feature_name, keys in _METADATA_FEATURE_KEYS.items():
        for key in keys:
            value = _to_float(event.metadata.get(key))
            if value is not None:
                features[feature_name] = value
                break
    return features


def _parse_edges(cascade_graph: dict[str, Any]) -> list[tuple[str, str]]:
    if "edge_index" in cascade_graph:
        sources, targets = cascade_graph["edge_index"]
        return [(str(source), str(target)) for source, target in zip(sources, targets)]

    for key in ("edges", "links"):
        if key in cascade_graph:
            edges = []
            for item in cascade_graph[key]:
                if isinstance(item, dict):
                    source = item.get("source")
                    target = item.get("target")
                else:
                    source, target = item
                if source is not None and target is not None:
                    edges.append((str(source), str(target)))
            return edges

    for key in ("adjacency", "successors"):
        if key in cascade_graph and isinstance(cascade_graph[key], dict):
            edges = []
            for source, targets in cascade_graph[key].items():
                for target in targets:
                    edges.append((str(source), str(target)))
            return edges
    return []


def _parse_nodes_and_communities(cascade_graph: dict[str, Any], edges: list[tuple[str, str]]) -> tuple[set[str], dict[str, str]]:
    nodes = {node for edge in edges for node in edge}
    communities: dict[str, str] = {}

    for item in cascade_graph.get("nodes", []):
        if isinstance(item, dict):
            node_id = item.get("id") or item.get("node_id") or item.get("name")
            if node_id is not None:
                node = str(node_id)
                nodes.add(node)
                community = item.get("community") or item.get("group") or item.get("cluster") or item.get("community_id")
                if community is not None:
                    communities[node] = str(community)
        else:
            nodes.add(str(item))

    node_attributes = cascade_graph.get("node_attributes", {})
    if isinstance(node_attributes, dict):
        for node_id, attrs in node_attributes.items():
            node = str(node_id)
            nodes.add(node)
            if isinstance(attrs, dict):
                community = attrs.get("community") or attrs.get("group") or attrs.get("cluster") or attrs.get("community_id")
                if community is not None:
                    communities[node] = str(community)

    return nodes, communities


def _compute_component_count(nodes: set[str], adjacency: dict[str, set[str]], reverse_adjacency: dict[str, set[str]]) -> int:
    seen: set[str] = set()
    components = 0
    for node in nodes:
        if node in seen:
            continue
        components += 1
        queue = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            neighbours = adjacency.get(current, set()) | reverse_adjacency.get(current, set())
            for neighbour in neighbours:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
    return components


def _compute_max_depth(nodes: set[str], adjacency: dict[str, set[str]], indegree: dict[str, int]) -> int:
    roots = [node for node in nodes if indegree.get(node, 0) == 0]
    if not roots:
        roots = list(nodes)
    max_depth = 0
    for root in roots:
        queue = deque([(root, 0)])
        seen = {root}
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            for neighbour in adjacency.get(node, set()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append((neighbour, depth + 1))
    return max_depth


def _structural_virality(nodes: set[str], adjacency: dict[str, set[str]], reverse_adjacency: dict[str, set[str]]) -> float:
    if len(nodes) < 2:
        return 0.0
    total_distance = 0.0
    pair_count = 0
    undirected = {node: adjacency.get(node, set()) | reverse_adjacency.get(node, set()) for node in nodes}
    for source in nodes:
        queue = deque([(source, 0)])
        seen = {source}
        while queue:
            node, distance = queue.popleft()
            if node != source:
                total_distance += distance
                pair_count += 1
            for neighbour in undirected.get(node, set()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append((neighbour, distance + 1))
    return total_distance / pair_count if pair_count else 0.0


def extract_graph_features(event: Event) -> dict[str, float]:
    features = _extract_metadata_features(event)
    cascade_graph = event.cascade_graph
    if isinstance(cascade_graph, dict):
        edges = _parse_edges(cascade_graph)
        nodes, communities = _parse_nodes_and_communities(cascade_graph, edges)
        if nodes or edges:
            adjacency = {node: set() for node in nodes}
            reverse_adjacency = {node: set() for node in nodes}
            indegree = {node: 0 for node in nodes}
            outdegree = {node: 0 for node in nodes}
            cross_community = 0
            for source, target in edges:
                adjacency.setdefault(source, set()).add(target)
                reverse_adjacency.setdefault(target, set()).add(source)
                indegree[target] = indegree.get(target, 0) + 1
                outdegree[source] = outdegree.get(source, 0) + 1
                if communities.get(source) and communities.get(target) and communities[source] != communities[target]:
                    cross_community += 1

            node_count = len(nodes)
            edge_count = len(edges)
            possible_edges = node_count * (node_count - 1)
            features.update(
                {
                    "graph_node_count": float(node_count),
                    "graph_edge_count": float(edge_count),
                    "graph_density": float(edge_count / possible_edges) if possible_edges else 0.0,
                    "graph_avg_out_degree": float(sum(outdegree.values()) / node_count) if node_count else 0.0,
                    "graph_max_out_degree": float(max(outdegree.values(), default=0)),
                    "graph_root_count": float(sum(1 for node in nodes if indegree.get(node, 0) == 0)),
                    "graph_component_count": float(_compute_component_count(nodes, adjacency, reverse_adjacency)),
                    "graph_max_depth": float(_compute_max_depth(nodes, adjacency, indegree)),
                    "graph_cross_community_ratio": float(cross_community / edge_count) if edge_count else 0.0,
                    "graph_structural_virality": float(_structural_virality(nodes, adjacency, reverse_adjacency)),
                }
            )

    features["graph_has_signal"] = 1.0 if any(name in features for name in GRAPH_FEATURE_COLUMNS) else 0.0
    for name in GRAPH_FEATURE_COLUMNS:
        features.setdefault(name, 0.0)
    return features


def build_graph_feature_frame(events: list[Event]) -> pd.DataFrame:
    rows = []
    for event in events:
        row = extract_graph_features(event)
        row.update(
            {
                "event_id": event.event_id,
                "dataset": event.dataset,
                "platform": event.platform,
                "topic": event.topic,
                "split": event.split,
                "is_sse": int(event.is_sse),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
