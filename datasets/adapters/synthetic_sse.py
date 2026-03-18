from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.early_observations import OBSERVATION_WINDOWS_MINUTES, extract_series_window_features
from schema.event import Event


DATASET_NAME = "synthetic_sse_graph"


@dataclass(slots=True)
class SyntheticSseConfig:
    processed_dir: Path = Path("data/processed/synthetic_sse")
    artifacts_dir: Path = Path("artifacts")
    seed: int = 42
    events: int = 20000
    sse_probability: float = 0.05
    slice_minutes: int = 20
    max_observation_bins: int = 72  # 24h at 20-min bins
    communities: int = 4
    nodes_per_community: int = 250
    base_reproduction: float = 0.8
    sse_reproduction: float = 1.6
    base_cross_community_prob: float = 0.02
    sse_cross_community_prob: float = 0.15
    onset_min_minutes: int = 40
    onset_max_minutes: int = 360
    max_graph_nodes: int = 64
    max_graph_edges: int = 256
    train_fraction: float = 0.70
    val_fraction: float = 0.15


def _split_by_time(frame: pd.DataFrame, train_fraction: float, val_fraction: float) -> tuple[dict[str, str], dict[str, str]]:
    ordered = frame.sort_values("start_time").reset_index(drop=True)
    total = len(ordered)
    train_index = max(int(total * train_fraction) - 1, 0)
    val_index = max(int(total * (train_fraction + val_fraction)) - 1, 0)
    train_cutoff = ordered.iloc[min(train_index, total - 1)]["start_time"]
    val_cutoff = ordered.iloc[min(val_index, total - 1)]["start_time"]
    split = np.where(
        ordered["start_time"] <= train_cutoff,
        "train",
        np.where(ordered["start_time"] <= val_cutoff, "val", "test"),
    )
    lookup = dict(zip(ordered["event_id"], split))
    boundaries = {"train_cutoff": pd.Timestamp(train_cutoff).isoformat(), "val_cutoff": pd.Timestamp(val_cutoff).isoformat()}
    return lookup, boundaries


def _choose_target_community(
    rng: np.random.Generator,
    source_community: int,
    communities: int,
    cross_prob: float,
) -> int:
    if communities <= 1:
        return 0
    if rng.random() < cross_prob:
        candidates = [c for c in range(communities) if c != source_community]
        return int(rng.choice(candidates))
    return int(source_community)


def _simulate_event(
    rng: np.random.Generator,
    event_id: str,
    start_time: datetime,
    is_sse: bool,
    config: SyntheticSseConfig,
) -> tuple[Event, dict[str, Any]]:
    total_nodes = config.communities * config.nodes_per_community
    available_nodes = np.arange(total_nodes, dtype=int)
    rng.shuffle(available_nodes)

    onset_bins_min = max(1, int(config.onset_min_minutes // config.slice_minutes))
    onset_bins_max = max(onset_bins_min, int(config.onset_max_minutes // config.slice_minutes))
    onset_bin = int(rng.integers(onset_bins_min, onset_bins_max + 1))
    onset_minutes = onset_bin * config.slice_minutes if is_sse else None

    base_r = float(config.base_reproduction)
    sse_r = float(config.sse_reproduction)
    base_cross = float(config.base_cross_community_prob)
    sse_cross = float(config.sse_cross_community_prob)

    root = int(available_nodes[0])
    infected_set = {root}
    infection_time_bins = {root: 0}
    parent = {root: None}

    frontier = [root]
    cursor = 1
    while frontier:
        source = frontier.pop(0)
        source_bin = int(infection_time_bins[source])
        # Apply a regime shift only for SSE events.
        reproduction = sse_r if is_sse and source_bin >= onset_bin else base_r
        cross_prob = sse_cross if is_sse and source_bin >= onset_bin else base_cross

        offspring = int(rng.poisson(max(reproduction, 0.0)))
        for _ in range(offspring):
            if cursor >= len(available_nodes):
                break
            target = int(available_nodes[cursor])
            cursor += 1
            if target in infected_set:
                continue

            source_comm = int(source // config.nodes_per_community)
            target_comm = _choose_target_community(rng, source_comm, config.communities, cross_prob)

            # Ensure target node belongs to the chosen community by remapping within that community block.
            # This keeps communities well-defined without expensive sampling.
            local_index = int(target % config.nodes_per_community)
            target = int(target_comm * config.nodes_per_community + local_index)
            if target in infected_set:
                continue

            target_bin = min(source_bin + int(rng.integers(1, 4)), config.max_observation_bins - 1)
            infected_set.add(target)
            infection_time_bins[target] = target_bin
            parent[target] = source
            frontier.append(target)

        if len(infected_set) >= config.nodes_per_community * config.communities:
            break
        if len(infected_set) >= (config.max_graph_nodes * 5):
            # Avoid pathological growth while keeping variance between regimes.
            break

    # Build cumulative engagement series with a fixed length so window features are meaningful.
    counts_by_bin = np.zeros(config.max_observation_bins, dtype=int)
    for bin_index in infection_time_bins.values():
        counts_by_bin[int(bin_index)] += 1
    engagement = np.cumsum(counts_by_bin).astype(int).tolist()

    # Determine whether we actually crossed onset (for SSE) to avoid nonsense labels in edge cases.
    if is_sse and onset_minutes is not None and engagement[onset_bin] <= 1:
        # If the event did not really grow by onset, demote to non-SSE for this run.
        is_sse = False
        onset_minutes = None

    # Cascade graph (community-aware) is stored as nodes + directed edges (true parent links).
    nodes_sorted = sorted(infection_time_bins.items(), key=lambda item: (item[1], item[0]))
    limited_nodes = nodes_sorted[: config.max_graph_nodes]
    kept = {node_id for node_id, _ in limited_nodes}

    node_records = []
    for node_id, bin_index in limited_nodes:
        node_records.append(
            {
                "id": str(node_id),
                "community": int(node_id // config.nodes_per_community),
                "time_bin": int(bin_index),
                "time_minutes": int(bin_index * config.slice_minutes),
            }
        )

    edge_records = []
    cross_edges = 0
    for node_id, _ in limited_nodes:
        source = parent.get(node_id)
        if source is None or source not in kept:
            continue
        edge_records.append({"source": str(source), "target": str(node_id)})
        if (source // config.nodes_per_community) != (node_id // config.nodes_per_community):
            cross_edges += 1
        if len(edge_records) >= config.max_graph_edges:
            break

    cascade_graph: dict[str, Any] = {
        "nodes": node_records,
        "edges": edge_records,
        "inference": "synthetic_true_transmission",
    }

    final_size = int(engagement[-1])
    event = Event(
        event_id=event_id,
        dataset=DATASET_NAME,
        platform="synthetic",
        topic="sse" if is_sse else "base",
        start_time=start_time,
        engagement_series=engagement,
        sentiment_series=[0.0],
        cascade_graph=cascade_graph,
        is_sse=bool(is_sse),
        time_to_sse_minutes=int(onset_minutes) if onset_minutes is not None else None,
        final_cascade_size=final_size,
        split=None,
        metadata={
            "slice_minutes": config.slice_minutes,
            "true_onset_minutes": onset_minutes,
            "true_reproduction_base": base_r,
            "true_reproduction_sse": sse_r if is_sse else None,
            "true_cross_community_prob_base": base_cross,
            "true_cross_community_prob_sse": sse_cross if is_sse else None,
            "graph_node_count": float(len(node_records)),
            "graph_edge_count": float(len(edge_records)),
            "graph_cross_edges": float(cross_edges),
        },
    )

    diagnostics = {
        "event_id": event_id,
        "is_sse": int(event.is_sse),
        "final_cascade_size": final_size,
        "time_to_sse_minutes": event.time_to_sse_minutes,
        "graph_nodes": len(node_records),
        "graph_edges": len(edge_records),
        "graph_cross_edge_ratio": float(cross_edges / max(len(edge_records), 1)),
    }
    return event, diagnostics


def prepare_dataset(config: SyntheticSseConfig | None = None) -> dict[str, object]:
    config = config or SyntheticSseConfig()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(config.seed))
    base_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    # Ensure deterministic chronological ordering by construction.
    start_times = [base_time + timedelta(minutes=5 * idx) for idx in range(int(config.events))]
    sse_flags = rng.random(int(config.events)) < float(config.sse_probability)

    events: list[Event] = []
    diag_rows: list[dict[str, Any]] = []
    for idx, (start_time, is_sse) in enumerate(zip(start_times, sse_flags)):
        event_id = f"{DATASET_NAME}::{idx:08d}"
        event, diag = _simulate_event(rng, event_id=event_id, start_time=start_time, is_sse=bool(is_sse), config=config)
        events.append(event)
        diag_rows.append(diag)

    split_lookup, split_boundaries = _split_by_time(
        pd.DataFrame({"event_id": [e.event_id for e in events], "start_time": [e.start_time for e in events]}),
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )
    for event in events:
        event.split = split_lookup[event.event_id]

    events_path = config.processed_dir / "events.jsonl.gz"
    index_path = config.processed_dir / "event_index.parquet"
    summary_path = config.artifacts_dir / "synthetic_sse_summary.json"
    quality_path = config.artifacts_dir / "synthetic_sse_data_quality.csv"

    # Write events.
    with gzip.open(events_path, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(event.to_json())
            handle.write("\n")

    # Write index.
    index_rows: list[dict[str, object]] = []
    for event in events:
        row: dict[str, object] = {
            "event_id": event.event_id,
            "dataset": event.dataset,
            "platform": event.platform,
            "topic": event.topic,
            "split": event.split,
            "start_time": event.start_time.isoformat(),
            "is_sse": int(event.is_sse),
            "time_to_sse_minutes": event.time_to_sse_minutes,
            "final_cascade_size": event.final_cascade_size,
            "slice_minutes": config.slice_minutes,
        }
        for window_minutes in OBSERVATION_WINDOWS_MINUTES:
            window_features = extract_series_window_features(
                engagement_series=event.engagement_series,
                sentiment_series=event.sentiment_series,
                window_minutes=window_minutes,
                slice_minutes=config.slice_minutes,
            )
            for name, value in window_features.items():
                row[f"w{window_minutes}_{name}"] = value
        index_rows.append(row)
    pd.DataFrame(index_rows).to_parquet(index_path, index=False)

    # Quality diagnostics (tracked artifact).
    quality = pd.DataFrame(diag_rows)
    split_map = {event.event_id: event.split for event in events}
    quality["split"] = quality["event_id"].map(split_map)
    quality.to_csv(quality_path, index=False, quoting=csv.QUOTE_MINIMAL)

    split_counts = {split: int(count) for split, count in pd.Series([e.split for e in events]).value_counts().sort_index().items()}
    sse_by_split = {split: int(count) for split, count in pd.Series([e.split for e in events if e.is_sse]).value_counts().sort_index().items()}
    summary: dict[str, object] = {
        "dataset": DATASET_NAME,
        "processed_events_path": str(events_path),
        "processed_index_path": str(index_path),
        "split_boundaries": split_boundaries,
        "total_events": int(len(events)),
        "sse_events": int(sum(1 for e in events if e.is_sse)),
        "sse_rate": float(np.mean([1.0 if e.is_sse else 0.0 for e in events])),
        "slice_minutes": int(config.slice_minutes),
        "max_observation_bins": int(config.max_observation_bins),
        "split_counts": split_counts,
        "sse_by_split": sse_by_split,
        "artifacts": {"data_quality": str(quality_path)},
        "generator_config": {
            "seed": int(config.seed),
            "events": int(config.events),
            "sse_probability": float(config.sse_probability),
            "communities": int(config.communities),
            "nodes_per_community": int(config.nodes_per_community),
            "base_reproduction": float(config.base_reproduction),
            "sse_reproduction": float(config.sse_reproduction),
            "base_cross_community_prob": float(config.base_cross_community_prob),
            "sse_cross_community_prob": float(config.sse_cross_community_prob),
            "onset_min_minutes": int(config.onset_min_minutes),
            "onset_max_minutes": int(config.onset_max_minutes),
            "max_graph_nodes": int(config.max_graph_nodes),
            "max_graph_edges": int(config.max_graph_edges),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

