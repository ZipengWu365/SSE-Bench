from __future__ import annotations

import csv
import gzip
import json
import math
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import pandas as pd

from features.early_observations import OBSERVATION_WINDOWS_MINUTES, extract_series_window_features
from schema.event import Event


DOWNLOAD_URL = "https://snap.stanford.edu/infopath/memes-w5-all-2011-03-2012-02-n5000-call-nc10-cl2-cascades-keywords.tgz"
DATASET_NAME = "snap_infopath_keywords"
EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(slots=True)
class InfoPathConfig:
    raw_dir: Path = Path("data/raw/infopath")
    processed_dir: Path = Path("data/processed/infopath_sse")
    artifacts_dir: Path = Path("artifacts")
    download_url: str = DOWNLOAD_URL
    force_download: bool = False
    stream_remote: bool = False
    extract_selected_files: bool = True
    max_keyword_files: int | None = 3
    max_events: int | None = 15000
    size_quantile: float = 0.99
    growth_sigma: float = 3.0
    min_topic_events: int = 50
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    slice_minutes: int = 60
    max_observation_bins: int = 336
    max_graph_nodes: int = 64
    max_graph_edges: int = 256

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / "infopath_cascades_keywords.tgz"

    @property
    def extract_dir(self) -> Path:
        return self.raw_dir / "extracted"


def _download_archive(config: InfoPathConfig) -> Path:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    if config.archive_path.exists() and not config.force_download:
        return config.archive_path
    with urllib.request.urlopen(config.download_url) as response, open(config.archive_path, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    return config.archive_path


def _open_tar(config: InfoPathConfig):
    if config.stream_remote:
        response = urllib.request.urlopen(config.download_url)
        return tarfile.open(fileobj=response, mode="r|gz")
    archive_path = _download_archive(config)
    return tarfile.open(archive_path, mode="r:gz")


def _parse_keyword(member_name: str) -> str:
    stem = Path(member_name).name
    if "-keywords-in-body-" in stem:
        return stem.split("-keywords-in-body-", 1)[1].removesuffix("-cascades.txt")
    return stem.removesuffix(".txt")


def _parse_cascade_line(line: str):
    if not line or ";" not in line:
        return None
    cascade_id, payload = line.split(";", 1)
    parts = [part.strip() for part in payload.split(",") if part.strip()]
    if not cascade_id or len(parts) < 2 or len(parts) % 2 != 0:
        return None
    infections = []
    for site_id, timestamp in zip(parts[0::2], parts[1::2]):
        try:
            infections.append((int(site_id), float(timestamp)))
        except ValueError:
            continue
    return (cascade_id, infections) if infections else None


def _build_engagement_series(
    infections: list[tuple[int, float]],
    slice_minutes: int,
    max_observation_bins: int,
) -> tuple[list[int], int]:
    first_ts = infections[0][1]
    counts_by_bin: dict[int, int] = {}
    overflow_count = 0
    max_seen_bin = 0
    for _, timestamp in infections:
        bin_index = max(0, int(((timestamp - first_ts) * 60.0) // slice_minutes))
        if bin_index >= max_observation_bins:
            overflow_count += 1
            continue
        counts_by_bin[bin_index] = counts_by_bin.get(bin_index, 0) + 1
        max_seen_bin = max(max_seen_bin, bin_index)
    series = []
    running_total = 0
    for bin_index in range(max_seen_bin + 1 or 1):
        running_total += counts_by_bin.get(bin_index, 0)
        series.append(running_total)
    return (series or [1]), overflow_count


def _build_cascade_graph(
    infections: list[tuple[int, float]],
    site_lookup: dict[int, str],
    max_nodes: int,
    max_edges: int,
) -> dict[str, object]:
    limited = infections[:max_nodes]
    nodes = [
        {
            "id": str(site_id),
            "label": site_lookup.get(site_id, str(site_id)),
            "time_hours": timestamp,
        }
        for site_id, timestamp in limited
    ]
    edges = []
    for index in range(1, len(limited)):
        source_id, source_ts = limited[index - 1]
        target_id, target_ts = limited[index]
        if source_id == target_id:
            continue
        edges.append(
            {
                "source": str(source_id),
                "target": str(target_id),
                "delta_hours": float(target_ts - source_ts),
            }
        )
        if len(edges) >= max_edges:
            break
    return {
        "nodes": nodes,
        "edges": edges,
        "inference": "temporal_predecessor_proxy",
    }


def _sse_thresholds(events_frame: pd.DataFrame, quantile: float, sigma: float, min_topic_events: int) -> pd.DataFrame:
    positive_growth = events_frame.loc[events_frame["growth_max"] > 0, "growth_max"]
    global_growth_threshold = max(1.0, float(positive_growth.mean()) + sigma * float(positive_growth.std())) if not positive_growth.empty else 1.0
    global_size_threshold = float(events_frame["final_cascade_size"].quantile(quantile))

    rows = []
    for topic, topic_frame in events_frame.groupby("topic", sort=True):
        use_global = len(topic_frame) < min_topic_events
        topic_positive = topic_frame.loc[topic_frame["growth_max"] > 0, "growth_max"]
        growth_threshold = (
            global_growth_threshold
            if use_global or topic_positive.empty
            else max(1.0, float(topic_positive.mean()) + sigma * float(topic_positive.std()))
        )
        size_threshold = global_size_threshold if use_global else float(topic_frame["final_cascade_size"].quantile(quantile))
        rows.append(
            {
                "topic": topic,
                "events": int(len(topic_frame)),
                "size_threshold": float(size_threshold),
                "growth_threshold": float(growth_threshold),
                "use_global_thresholds": bool(use_global),
            }
        )
    return pd.DataFrame(rows)


def _split_by_time(events_frame: pd.DataFrame, train_fraction: float, val_fraction: float) -> tuple[dict[str, str], dict[str, str]]:
    ordered = events_frame[["event_id", "start_time"]].sort_values("start_time").reset_index(drop=True)
    total = len(ordered)
    train_index = max(int(total * train_fraction) - 1, 0)
    val_index = max(int(total * (train_fraction + val_fraction)) - 1, 0)
    train_cutoff = ordered.iloc[min(train_index, total - 1)]["start_time"]
    val_cutoff = ordered.iloc[min(val_index, total - 1)]["start_time"]
    ordered["split"] = np.where(
        ordered["start_time"] <= train_cutoff,
        "train",
        np.where(ordered["start_time"] <= val_cutoff, "val", "test"),
    )
    lookup = dict(zip(ordered["event_id"], ordered["split"]))
    return lookup, {"train_cutoff": train_cutoff.isoformat(), "val_cutoff": val_cutoff.isoformat()}


def _iter_tar_members(config: InfoPathConfig) -> Iterator[tuple[str, BinaryIO]]:
    keyword_count = 0
    with _open_tar(config) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("-cascades.txt"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            yield member.name, handle
            keyword_count += 1
            if config.max_keyword_files is not None and keyword_count >= config.max_keyword_files:
                break


def prepare_dataset(config: InfoPathConfig | None = None) -> dict[str, object]:
    config = config or InfoPathConfig()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.extract_dir.mkdir(parents=True, exist_ok=True)

    event_records: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    total_events = 0
    topic_event_limit = None
    if config.max_events is not None and config.max_keyword_files is not None and config.max_keyword_files > 0:
        topic_event_limit = max(1, math.ceil(config.max_events / config.max_keyword_files))

    for member_name, handle in _iter_tar_members(config):
        topic = _parse_keyword(member_name)
        site_lookup: dict[int, str] = {}
        parsed_events = 0
        skipped_lines = 0
        truncated_events = 0
        overflow_infections_total = 0
        durations = []
        final_sizes = []

        while True:
            raw_line = handle.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                break
            try:
                site_id, site_name = line.split(",", 1)
                site_lookup[int(site_id)] = site_name
            except ValueError:
                skipped_lines += 1

        extracted_path = None
        extracted_handle = None
        if config.extract_selected_files:
            extracted_path = config.extract_dir / Path(member_name).name
            extracted_path.parent.mkdir(parents=True, exist_ok=True)
            extracted_handle = extracted_path.open("wt", encoding="utf-8")

        try:
            for raw_line in handle:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if extracted_handle is not None:
                    extracted_handle.write(line)
                    extracted_handle.write("\n")
                if not line:
                    continue
                parsed = _parse_cascade_line(line)
                if parsed is None:
                    skipped_lines += 1
                    continue
                cascade_id, infections = parsed
                dedup = {}
                for site_id, timestamp in infections:
                    if site_id not in dedup or timestamp < dedup[site_id]:
                        dedup[site_id] = timestamp
                ordered = sorted(dedup.items(), key=lambda item: item[1])
                if not ordered:
                    skipped_lines += 1
                    continue

                engagement_series, overflow_count = _build_engagement_series(
                    ordered,
                    slice_minutes=config.slice_minutes,
                    max_observation_bins=config.max_observation_bins,
                )
                if overflow_count:
                    truncated_events += 1
                overflow_infections_total += overflow_count
                graph = _build_cascade_graph(
                    ordered,
                    site_lookup=site_lookup,
                    max_nodes=config.max_graph_nodes,
                    max_edges=config.max_graph_edges,
                )
                start_time = EPOCH_UTC + timedelta(hours=ordered[0][1])
                duration_hours = float(ordered[-1][1] - ordered[0][1]) if len(ordered) > 1 else 0.0
                growth_series = np.diff([0, *engagement_series])
                growth_max = int(growth_series.max()) if len(growth_series) else int(engagement_series[-1])

                event_records.append(
                    {
                        "event_id": f"{DATASET_NAME}::{topic}::{cascade_id}",
                        "dataset": DATASET_NAME,
                        "platform": "web",
                        "topic": topic,
                        "start_time": start_time,
                        "engagement_series": engagement_series,
                        "final_cascade_size": int(len(ordered)),
                        "growth_max": int(growth_max),
                        "cascade_graph": graph,
                        "metadata": {
                            "source_member": member_name,
                            "cascade_id": cascade_id,
                            "slice_minutes": config.slice_minutes,
                            "duration_hours": duration_hours,
                            "overflow_infections": int(overflow_count),
                            "node_count": float(len(graph["nodes"])),
                            "edge_count": float(len(graph["edges"])),
                            "max_depth": float(max(0, len(graph["nodes"]) - 1)),
                        },
                    }
                )
                parsed_events += 1
                total_events += 1
                durations.append(duration_hours)
                final_sizes.append(len(ordered))
                if topic_event_limit is not None and parsed_events >= topic_event_limit:
                    break
                if config.max_events is not None and total_events >= config.max_events:
                    break
        finally:
            if extracted_handle is not None:
                extracted_handle.close()

        quality_rows.append(
            {
                "topic": topic,
                "source_member": member_name,
                "site_dictionary_size": int(len(site_lookup)),
                "events_parsed": int(parsed_events),
                "skipped_lines": int(skipped_lines),
                "truncated_events": int(truncated_events),
                "overflow_infections_total": int(overflow_infections_total),
                "mean_final_cascade_size": float(np.mean(final_sizes)) if final_sizes else 0.0,
                "mean_duration_hours": float(np.mean(durations)) if durations else 0.0,
            }
        )

        if config.max_events is not None and total_events >= config.max_events:
            break

    if not event_records:
        raise ValueError("No InfoPath events were parsed. Check download settings or file limits.")

    events_frame = pd.DataFrame(event_records)
    split_lookup, split_boundaries = _split_by_time(events_frame, config.train_fraction, config.val_fraction)
    thresholds = _sse_thresholds(
        events_frame,
        quantile=config.size_quantile,
        sigma=config.growth_sigma,
        min_topic_events=config.min_topic_events,
    )
    threshold_lookup = thresholds.set_index("topic").to_dict(orient="index")

    events_path = config.processed_dir / "events.jsonl.gz"
    index_path = config.processed_dir / "event_index.parquet"
    thresholds_path = config.artifacts_dir / "infopath_topic_thresholds.csv"
    quality_path = config.artifacts_dir / "infopath_data_quality.csv"
    summary_path = config.artifacts_dir / "infopath_summary.json"
    if events_path.exists():
        events_path.unlink()

    index_rows: list[dict[str, object]] = []
    positive_onsets = []
    with gzip.open(events_path, "wt", encoding="utf-8") as handle:
        for row in events_frame.itertuples(index=False):
            topic_thresholds = threshold_lookup[row.topic]
            size_threshold = float(topic_thresholds["size_threshold"])
            growth_threshold = float(topic_thresholds["growth_threshold"])
            growth_series = np.diff([0, *row.engagement_series])
            is_sse = bool(row.final_cascade_size >= size_threshold and growth_series.max(initial=0) >= growth_threshold)
            crossing = np.where(growth_series >= growth_threshold)[0]
            time_to_sse_minutes = int(crossing[0] * config.slice_minutes) if is_sse and len(crossing) else None
            if time_to_sse_minutes is not None:
                positive_onsets.append(time_to_sse_minutes)

            event = Event(
                event_id=row.event_id,
                dataset=row.dataset,
                platform=row.platform,
                topic=row.topic,
                start_time=row.start_time,
                engagement_series=list(row.engagement_series),
                sentiment_series=None,
                cascade_graph=row.cascade_graph,
                is_sse=is_sse,
                time_to_sse_minutes=time_to_sse_minutes,
                final_cascade_size=int(row.final_cascade_size),
                split=split_lookup[row.event_id],
                metadata={
                    **row.metadata,
                    "size_threshold": size_threshold,
                    "growth_threshold": growth_threshold,
                },
            )
            handle.write(event.to_json())
            handle.write("\n")

            feature_row = {
                "event_id": event.event_id,
                "dataset": event.dataset,
                "platform": event.platform,
                "topic": event.topic,
                "split": event.split,
                "start_time": event.start_time.isoformat(),
                "is_sse": int(event.is_sse),
                "time_to_sse_minutes": event.time_to_sse_minutes,
                "final_cascade_size": event.final_cascade_size,
                "size_threshold": size_threshold,
                "growth_threshold": growth_threshold,
                "graph_node_count": float(len(event.cascade_graph.get("nodes", []))) if event.cascade_graph else 0.0,
                "graph_edge_count": float(len(event.cascade_graph.get("edges", []))) if event.cascade_graph else 0.0,
            }
            for window_minutes in OBSERVATION_WINDOWS_MINUTES:
                window_features = extract_series_window_features(
                    engagement_series=event.engagement_series,
                    sentiment_series=event.sentiment_series,
                    window_minutes=window_minutes,
                    slice_minutes=config.slice_minutes,
                )
                for name, value in window_features.items():
                    feature_row[f"w{window_minutes}_{name}"] = value
            index_rows.append(feature_row)

    index_frame = pd.DataFrame(index_rows)
    index_frame.to_parquet(index_path, index=False)
    thresholds.to_csv(thresholds_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False, quoting=csv.QUOTE_MINIMAL)

    split_counts = {split: int(count) for split, count in index_frame["split"].value_counts().sort_index().items()}
    topic_counts = {topic: int(count) for topic, count in index_frame["topic"].value_counts().sort_index().items()}
    summary = {
        "dataset": DATASET_NAME,
        "download_url": config.download_url,
        "stream_remote": config.stream_remote,
        "max_keyword_files": config.max_keyword_files,
        "max_events": config.max_events,
        "slice_minutes": config.slice_minutes,
        "max_observation_bins": config.max_observation_bins,
        "raw_archive_path": str(config.archive_path),
        "extract_dir": str(config.extract_dir),
        "processed_events_path": str(events_path),
        "processed_index_path": str(index_path),
        "split_boundaries": split_boundaries,
        "total_events": int(len(index_frame)),
        "sse_events": int(index_frame["is_sse"].sum()),
        "sse_rate": float(index_frame["is_sse"].mean()),
        "positive_time_to_sse_labels": int(len(positive_onsets)),
        "split_counts": split_counts,
        "topic_counts": topic_counts,
        "malformed_lines_total": int(sum(row["skipped_lines"] for row in quality_rows)),
        "timeseries_overflow_total": int(sum(row["overflow_infections_total"] for row in quality_rows)),
        "median_time_to_sse_minutes": float(np.median(positive_onsets)) if positive_onsets else None,
        "artifacts": {
            "topic_thresholds": str(thresholds_path),
            "data_quality": str(quality_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
