from __future__ import annotations

from datetime import datetime, timezone

from schema.event import Event
from features.graph_features import extract_graph_features


def test_extract_graph_features_from_edges_and_communities() -> None:
    event = Event(
        event_id="toy::graph1",
        dataset="toy",
        platform="web",
        topic=None,
        start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        engagement_series=[1, 2],
        final_cascade_size=2,
        split="train",
        cascade_graph={
            "nodes": [
                {"id": "A", "community": "c1"},
                {"id": "B", "community": "c1"},
                {"id": "C", "community": "c2"},
            ],
            "edges": [
                {"source": "A", "target": "B"},
                {"source": "A", "target": "C"},
            ],
        },
    )

    features = extract_graph_features(event)
    assert features["graph_has_signal"] == 1.0
    assert features["graph_node_count"] == 3.0
    assert features["graph_edge_count"] == 2.0
    assert features["graph_max_out_degree"] == 2.0
    assert features["graph_root_count"] == 1.0
    # density = 2 / (3*2)
    assert abs(features["graph_density"] - (2.0 / 6.0)) < 1e-9
    # one cross-community edge (A->C) out of two
    assert abs(features["graph_cross_community_ratio"] - 0.5) < 1e-9


def test_extract_graph_features_from_metadata_fallback() -> None:
    event = Event(
        event_id="toy::graph2",
        dataset="toy",
        platform="web",
        topic=None,
        start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        engagement_series=[1],
        final_cascade_size=1,
        split="train",
        cascade_graph=None,
        metadata={"node_count": 10, "edge_count": 9, "max_depth": 3},
    )

    features = extract_graph_features(event)
    assert features["graph_has_signal"] == 1.0
    assert features["graph_node_count"] == 10.0
    assert features["graph_edge_count"] == 9.0
    assert features["graph_max_depth"] == 3.0

