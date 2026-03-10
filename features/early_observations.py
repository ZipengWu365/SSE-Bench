from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from schema.event import Event


OBSERVATION_WINDOWS_MINUTES = (20, 60, 360, 1440)


def window_to_steps(window_minutes: int, slice_minutes: int = 20) -> int:
    return max(1, math.ceil(window_minutes / slice_minutes))


def _extract_from_arrays(
    engagement_series: list[int],
    sentiment_series: list[float] | None,
    window_minutes: int,
    slice_minutes: int = 20,
) -> dict[str, float]:
    steps = window_to_steps(window_minutes, slice_minutes=slice_minutes)
    engagement = np.asarray(engagement_series[:steps], dtype=float)
    growth = np.diff(np.concatenate(([0.0], engagement)))
    sentiment = np.asarray(sentiment_series or [0.0], dtype=float)
    mean_growth = float(growth.mean()) if growth.size else 0.0
    burstiness = float(growth.std() / mean_growth) if mean_growth > 0 else 0.0

    return {
        "observed_size": float(engagement[-1]),
        "growth_rate": float(engagement[-1] / max(window_minutes, 1)),
        "max_increment": float(growth.max()) if growth.size else 0.0,
        "mean_increment": mean_growth,
        "burstiness": burstiness,
        "acceleration": float(growth[-1] - growth[0]) if growth.size > 1 else 0.0,
        "sentiment_mean": float(sentiment.mean()) if sentiment.size else 0.0,
        "sentiment_abs": float(np.abs(sentiment).mean()) if sentiment.size else 0.0,
    }


def extract_window_features(
    event: Event,
    window_minutes: int,
    slice_minutes: int = 20,
) -> dict[str, float | int | str | None]:
    features = _extract_from_arrays(
        engagement_series=event.engagement_series,
        sentiment_series=event.sentiment_series,
        window_minutes=window_minutes,
        slice_minutes=slice_minutes,
    )
    features.update(
        {
            "event_id": event.event_id,
            "dataset": event.dataset,
            "platform": event.platform,
            "topic": event.topic,
            "split": event.split,
            "window_minutes": window_minutes,
            "slice_minutes": slice_minutes,
            "is_sse": int(event.is_sse),
            "time_to_sse_minutes": event.time_to_sse_minutes,
            "final_cascade_size": event.final_cascade_size,
            "start_time": event.start_time.isoformat(),
        }
    )
    return features


def extract_series_window_features(
    engagement_series: list[int],
    sentiment_series: list[float] | None,
    window_minutes: int,
    slice_minutes: int = 20,
) -> dict[str, float]:
    return _extract_from_arrays(
        engagement_series=engagement_series,
        sentiment_series=sentiment_series,
        window_minutes=window_minutes,
        slice_minutes=slice_minutes,
    )


def build_feature_frame(
    events: Iterable[Event],
    window_minutes: int,
    slice_minutes: int = 20,
) -> pd.DataFrame:
    rows = [
        extract_window_features(event, window_minutes=window_minutes, slice_minutes=slice_minutes)
        for event in events
    ]
    return pd.DataFrame(rows)
