from __future__ import annotations

import csv
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features.early_observations import OBSERVATION_WINDOWS_MINUTES, extract_series_window_features
from schema.event import Event


DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/432/"
    "news%2Bpopularity%2Bin%2Bmultiple%2Bsocial%2Bmedia%2Bplatforms.zip"
)
DATASET_NAME = "uci_news_social_feedback"
SLICE_MINUTES = 20
TIME_SERIES_COLUMNS = [f"TS{index}" for index in range(1, 145)]
PLATFORM_COLUMNS = {
    "facebook": "Facebook",
    "googleplus": "GooglePlus",
    "linkedin": "LinkedIn",
}


@dataclass(slots=True)
class UciNewsConfig:
    raw_dir: Path = Path("data/raw/uci_news")
    processed_dir: Path = Path("data/processed/uci_news_sse")
    artifacts_dir: Path = Path("artifacts")
    download_url: str = DOWNLOAD_URL
    size_quantile: float = 0.99
    growth_sigma: float = 3.0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    force_download: bool = False

    @property
    def zip_path(self) -> Path:
        return self.raw_dir / "uci_news.zip"

    @property
    def extract_dir(self) -> Path:
        return self.raw_dir / "extracted"


def _feedback_files(extract_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in (extract_dir / "Data").glob("*.csv")
        if path.name != "News_Final.csv"
    )


def _parse_cohort(path: Path) -> tuple[str, str]:
    platform, topic = path.stem.split("_", 1)
    return platform.lower(), topic.lower()


def _download_zip(config: UciNewsConfig) -> Path:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    if config.zip_path.exists() and not config.force_download:
        return config.zip_path
    with urllib.request.urlopen(config.download_url) as response, open(config.zip_path, "wb") as handle:
        shutil.copyfileobj(response, handle)
    return config.zip_path


def _extract_zip(config: UciNewsConfig) -> Path:
    news_final = config.extract_dir / "Data" / "News_Final.csv"
    if news_final.exists() and not config.force_download:
        return config.extract_dir
    if config.extract_dir.exists():
        shutil.rmtree(config.extract_dir)
    config.extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(config.zip_path) as archive:
        archive.extractall(config.extract_dir)
    return config.extract_dir


def _load_news_table(extract_dir: Path) -> pd.DataFrame:
    news = pd.read_csv(extract_dir / "Data" / "News_Final.csv")
    news["IDLink"] = news["IDLink"].astype(int)
    news["Topic"] = news["Topic"].astype(str).str.lower()
    news["PublishDate"] = pd.to_datetime(news["PublishDate"], errors="coerce", utc=True)
    fallback_time = pd.Timestamp("1970-01-01T00:00:00Z")
    news["PublishDate"] = news["PublishDate"].fillna(fallback_time)
    news = news.drop_duplicates(subset=["IDLink", "Topic"], keep="first").reset_index(drop=True)
    return news


def _build_split_lookup(
    extract_dir: Path,
    news: pd.DataFrame,
    config: UciNewsConfig,
) -> tuple[dict[str, str], dict[str, str]]:
    split_frames = []
    news_index = news[["IDLink", "Topic", "PublishDate"]].copy()
    fallback_time = pd.Timestamp("1970-01-01T00:00:00Z")
    for feedback_path in _feedback_files(extract_dir):
        platform, topic = _parse_cohort(feedback_path)
        subset = pd.read_csv(feedback_path, usecols=["IDLink"])
        subset["IDLink"] = subset["IDLink"].astype(int)
        subset["Topic"] = topic
        subset = subset.merge(news_index, on=["IDLink", "Topic"], how="left")
        subset["PublishDate"] = subset["PublishDate"].fillna(fallback_time)
        subset["event_id"] = (
            DATASET_NAME
            + "::"
            + platform
            + "::"
            + topic
            + "::"
            + subset["IDLink"].astype(str)
        )
        split_frames.append(subset[["event_id", "PublishDate"]])

    split_table = pd.concat(split_frames, ignore_index=True).sort_values("PublishDate").reset_index(drop=True)
    total = len(split_table)
    train_index = max(int(total * config.train_fraction) - 1, 0)
    val_index = max(int(total * (config.train_fraction + config.val_fraction)) - 1, 0)
    train_cutoff = split_table.iloc[min(train_index, total - 1)]["PublishDate"]
    val_cutoff = split_table.iloc[min(val_index, total - 1)]["PublishDate"]

    split_table["split"] = np.where(
        split_table["PublishDate"] <= train_cutoff,
        "train",
        np.where(split_table["PublishDate"] <= val_cutoff, "val", "test"),
    )
    lookup = dict(zip(split_table["event_id"], split_table["split"]))
    boundaries = {
        "train_cutoff": train_cutoff.isoformat(),
        "val_cutoff": val_cutoff.isoformat(),
    }
    return lookup, boundaries


def _clean_engagement_matrix(values: np.ndarray) -> np.ndarray:
    matrix = values.astype(float, copy=True)
    matrix[matrix < 0] = np.nan
    mask = np.isnan(matrix)
    indices = np.where(~mask, np.arange(matrix.shape[1]), 0)
    np.maximum.accumulate(indices, axis=1, out=indices)
    filled = matrix[np.arange(matrix.shape[0])[:, None], indices]
    filled[np.isnan(filled)] = 0
    return np.maximum.accumulate(filled, axis=1).astype(int)


def prepare_dataset(config: UciNewsConfig | None = None) -> dict[str, object]:
    config = config or UciNewsConfig()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    _download_zip(config)
    extract_dir = _extract_zip(config)
    news = _load_news_table(extract_dir)
    split_lookup, split_boundaries = _build_split_lookup(extract_dir, news, config)
    news = news.set_index("IDLink", drop=False)

    events_path = config.processed_dir / "events.jsonl.gz"
    index_path = config.processed_dir / "event_index.parquet"
    summary_path = config.artifacts_dir / "uci_news_summary.json"
    thresholds_path = config.artifacts_dir / "uci_news_cohort_thresholds.csv"

    event_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []

    if events_path.exists():
        events_path.unlink()

    import gzip

    with gzip.open(events_path, "wt", encoding="utf-8") as handle:
        for feedback_path in _feedback_files(extract_dir):
            platform, topic = _parse_cohort(feedback_path)
            feedback = pd.read_csv(feedback_path)
            feedback["IDLink"] = feedback["IDLink"].astype(int)
            feedback["Topic"] = topic
            merged = feedback.merge(news.reset_index(drop=True), on=["IDLink", "Topic"], how="left", validate="1:1")
            engagement = _clean_engagement_matrix(merged[TIME_SERIES_COLUMNS].to_numpy(dtype=float))

            declared_final = merged[PLATFORM_COLUMNS[platform]].to_numpy(dtype=float)
            declared_final = np.where(declared_final >= 0, declared_final, 0)
            final_size = np.maximum(engagement[:, -1], declared_final.astype(int))
            engagement[:, -1] = final_size

            growth = np.diff(engagement, prepend=0, axis=1)
            positive_growth = growth[growth > 0]
            growth_threshold = float(
                max(
                    1.0,
                    positive_growth.mean() + config.growth_sigma * positive_growth.std(),
                )
            ) if positive_growth.size else 1.0
            size_threshold = float(np.quantile(final_size, config.size_quantile))

            crossing_mask = growth >= growth_threshold
            first_cross = np.where(crossing_mask.any(axis=1), crossing_mask.argmax(axis=1) + 1, -1)
            is_sse = (final_size >= size_threshold) & (growth.max(axis=1) >= growth_threshold)
            sentiment_mean = (
                merged[["SentimentTitle", "SentimentHeadline"]]
                .fillna(0.0)
                .mean(axis=1)
                .to_numpy(dtype=float)
            )

            cohort_rows.append(
                {
                    "platform": platform,
                    "topic": topic,
                    "events": int(len(merged)),
                    "size_threshold": size_threshold,
                    "growth_threshold": growth_threshold,
                    "sse_events": int(is_sse.sum()),
                    "sse_rate": float(is_sse.mean()),
                }
            )

            for row_index, row in enumerate(merged.itertuples(index=False)):
                event_id = f"{DATASET_NAME}::{platform}::{topic}::{int(row.IDLink)}"
                split = split_lookup[event_id]
                engagement_series = engagement[row_index].astype(int).tolist()
                sentiment_series = [round(float(sentiment_mean[row_index]), 6)]
                time_to_sse_minutes = (
                    int(first_cross[row_index] * SLICE_MINUTES)
                    if bool(is_sse[row_index]) and first_cross[row_index] > 0
                    else None
                )
                event = Event(
                    event_id=event_id,
                    dataset=DATASET_NAME,
                    platform=platform,
                    topic=topic,
                    start_time=pd.Timestamp(row.PublishDate).to_pydatetime(),
                    engagement_series=engagement_series,
                    sentiment_series=sentiment_series,
                    cascade_graph=None,
                    is_sse=bool(is_sse[row_index]),
                    time_to_sse_minutes=time_to_sse_minutes,
                    final_cascade_size=int(final_size[row_index]),
                    split=split,
                    metadata={
                        "source": row.Source,
                        "title": row.Title,
                        "headline": row.Headline,
                        "topic": topic,
                        "slice_minutes": SLICE_MINUTES,
                        "sentiment_title": float(row.SentimentTitle),
                        "sentiment_headline": float(row.SentimentHeadline),
                        "declared_final_size": int(declared_final[row_index]),
                        "size_threshold": size_threshold,
                        "growth_threshold": growth_threshold,
                    },
                )
                handle.write(event.to_json())
                handle.write("\n")

                feature_row: dict[str, object] = {
                    "event_id": event.event_id,
                    "dataset": event.dataset,
                    "platform": event.platform,
                    "topic": event.topic,
                    "split": event.split,
                    "start_time": event.start_time.isoformat(),
                    "source": row.Source,
                    "is_sse": int(event.is_sse),
                    "time_to_sse_minutes": event.time_to_sse_minutes,
                    "final_cascade_size": event.final_cascade_size,
                    "size_threshold": size_threshold,
                    "growth_threshold": growth_threshold,
                    "sentiment_title": float(row.SentimentTitle),
                    "sentiment_headline": float(row.SentimentHeadline),
                }
                for window_minutes in OBSERVATION_WINDOWS_MINUTES:
                    window_features = extract_series_window_features(
                        engagement_series=engagement_series,
                        sentiment_series=sentiment_series,
                        window_minutes=window_minutes,
                        slice_minutes=SLICE_MINUTES,
                    )
                    for name, value in window_features.items():
                        feature_row[f"w{window_minutes}_{name}"] = value
                event_rows.append(feature_row)

    index_frame = pd.DataFrame(event_rows)
    index_frame.to_parquet(index_path, index=False)

    cohort_frame = pd.DataFrame(cohort_rows)
    cohort_frame.to_csv(thresholds_path, index=False, quoting=csv.QUOTE_MINIMAL)

    split_counts = {
        split: int(count)
        for split, count in index_frame["split"].value_counts().sort_index().items()
    }
    platform_counts = {
        platform: int(count)
        for platform, count in index_frame["platform"].value_counts().sort_index().items()
    }
    summary: dict[str, object] = {
        "dataset": DATASET_NAME,
        "download_url": config.download_url,
        "raw_zip_path": str(config.zip_path),
        "processed_events_path": str(events_path),
        "processed_index_path": str(index_path),
        "split_boundaries": split_boundaries,
        "total_events": int(len(index_frame)),
        "sse_events": int(index_frame["is_sse"].sum()),
        "sse_rate": float(index_frame["is_sse"].mean()),
        "split_counts": split_counts,
        "platform_counts": platform_counts,
        "observation_windows_minutes": list(OBSERVATION_WINDOWS_MINUTES),
        "slice_minutes": SLICE_MINUTES,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
