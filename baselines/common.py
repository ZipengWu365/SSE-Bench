from __future__ import annotations

from pathlib import Path

import pandas as pd

from features import window_feature_columns


DEFAULT_INDEX_PATH = Path("data/processed/uci_news_sse/event_index.parquet")
DEFAULT_EVENTS_PATH = Path("data/processed/uci_news_sse/events.jsonl.gz")


def load_index(path: str | Path = DEFAULT_INDEX_PATH) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def feature_frame_for_window(frame: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    columns = window_feature_columns(window_minutes)
    return frame[columns].astype(float)
