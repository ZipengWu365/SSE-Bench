from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datasets.adapters.synthetic_sse import SyntheticSseConfig, prepare_dataset
from schema.event import load_events_jsonl


def _count_jsonl_lines(path: Path) -> int:
    # Use schema loader which understands gzip and will validate.
    return len(load_events_jsonl(path))


def test_synthetic_prepare_dataset_writes_expected_artifacts(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    config = SyntheticSseConfig(
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        seed=123,
        events=200,
        sse_probability=0.2,
        communities=3,
        nodes_per_community=80,
        max_graph_nodes=32,
        max_graph_edges=128,
    )

    summary = prepare_dataset(config)

    events_path = Path(summary["processed_events_path"])
    index_path = Path(summary["processed_index_path"])
    assert events_path.exists()
    assert index_path.exists()

    # Summary is also written to a tracked artifact path within artifacts_dir.
    summary_path = artifacts_dir / "synthetic_sse_summary.json"
    assert summary_path.exists()
    disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert disk_summary["dataset"] == summary["dataset"]

    assert summary["total_events"] == config.events
    assert summary["split_counts"]["train"] + summary["split_counts"]["val"] + summary["split_counts"]["test"] == config.events

    sse_events = int(summary["sse_events"])
    assert sum(summary["sse_by_split"].values()) == sse_events
    assert abs(float(summary["sse_rate"]) - (sse_events / config.events)) < 1e-9
    assert int(summary["slice_minutes"]) == config.slice_minutes

    # Index table has the expected number of rows and window feature columns.
    frame = pd.read_parquet(index_path)
    assert len(frame) == config.events
    assert "w60_observed_size" in frame.columns
    assert "w360_growth_rate" in frame.columns
    assert "w1440_burstiness" in frame.columns

    assert _count_jsonl_lines(events_path) == config.events


def test_synthetic_events_have_community_graph_and_self_consistent_labels(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    config = SyntheticSseConfig(
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        seed=7,
        events=120,
        sse_probability=0.3,
        communities=4,
        nodes_per_community=60,
        max_graph_nodes=24,
        max_graph_edges=80,
    )
    summary = prepare_dataset(config)
    events = load_events_jsonl(Path(summary["processed_events_path"]))
    assert len(events) == config.events

    sampled = events[:25]
    for event in sampled:
        assert isinstance(event.cascade_graph, dict)
        nodes = event.cascade_graph.get("nodes", [])
        assert nodes, "synthetic control track should include graph nodes"
        assert "community" in nodes[0]
        assert "time_minutes" in nodes[0]
        if event.is_sse:
            assert event.time_to_sse_minutes is not None
        else:
            assert event.time_to_sse_minutes is None

