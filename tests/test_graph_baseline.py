from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from schema.event import Event, save_events_jsonl


def _write_events(tmp_path: Path, events: list[Event]) -> Path:
    path = tmp_path / "events.jsonl.gz"
    save_events_jsonl(events, path)
    return path


def test_graph_baseline_errors_when_no_graph_signal(tmp_path: Path) -> None:
    from baselines import graph_cascade_classifier as mod

    events = [
        Event(
            event_id="toy::no_graph_train",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[1, 2],
            final_cascade_size=2,
            split="train",
            is_sse=False,
        ),
        Event(
            event_id="toy::no_graph_val",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 2, tzinfo=timezone.utc),
            engagement_series=[1, 1],
            final_cascade_size=1,
            split="val",
            is_sse=False,
        ),
        Event(
            event_id="toy::no_graph_test",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 3, tzinfo=timezone.utc),
            engagement_series=[1],
            final_cascade_size=1,
            split="test",
            is_sse=True,
        ),
    ]
    events_path = _write_events(tmp_path, events)
    with pytest.raises(ValueError, match="No graph features found"):
        mod.run_baseline(events_path=events_path)


def test_graph_baseline_errors_when_missing_split_coverage(tmp_path: Path) -> None:
    from baselines import graph_cascade_classifier as mod

    events = [
        Event(
            event_id="toy::graph_train",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[1, 2],
            final_cascade_size=2,
            split="train",
            is_sse=False,
            cascade_graph={"edges": [("A", "B")]},
        ),
        Event(
            event_id="toy::graph_test",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 2, tzinfo=timezone.utc),
            engagement_series=[1, 3],
            final_cascade_size=3,
            split="test",
            is_sse=True,
            cascade_graph={"edges": [("A", "C")]},
        ),
    ]
    events_path = _write_events(tmp_path, events)
    with pytest.raises(ValueError, match="Insufficient graph-aware split coverage"):
        mod.run_baseline(events_path=events_path)


def test_graph_baseline_smoke_with_dummy_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import graph_cascade_classifier as mod

    class DummyXGBClassifier:
        def __init__(self, *args, **kwargs) -> None:
            self.p = 0.5

        def fit(self, X, y, eval_set=None, verbose: bool = False):
            self.p = float(np.asarray(y, dtype=float).mean()) if len(y) else 0.5
            return self

        def predict_proba(self, X):
            n = int(np.asarray(X).shape[0])
            p1 = np.full(n, self.p, dtype=float)
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])

    monkeypatch.setattr(mod, "XGBClassifier", DummyXGBClassifier)

    events = [
        Event(
            event_id="toy::graph_train_0",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[1, 2],
            final_cascade_size=2,
            split="train",
            is_sse=False,
            cascade_graph={"nodes": ["A", "B"], "edges": [("A", "B")]},
        ),
        Event(
            event_id="toy::graph_train_1",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[1, 4],
            final_cascade_size=4,
            split="train",
            is_sse=True,
            cascade_graph={"nodes": ["A", "C"], "edges": [("A", "C")]},
        ),
        Event(
            event_id="toy::graph_val",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 2, tzinfo=timezone.utc),
            engagement_series=[1, 1, 2],
            final_cascade_size=2,
            split="val",
            is_sse=False,
            cascade_graph={"edges": [("X", "Y")]},
        ),
        Event(
            event_id="toy::graph_test",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 3, tzinfo=timezone.utc),
            engagement_series=[1, 5],
            final_cascade_size=5,
            split="test",
            is_sse=True,
            cascade_graph={"edges": [("A", "D")]},
        ),
    ]

    events_path = _write_events(tmp_path, events)
    report = mod.run_baseline(events_path=events_path)
    assert report["graph_signal_events"] == len(events)
    assert report["train_events"] == 2
    assert report["val_events"] == 1
    assert report["test_events"] == 1
    assert "classification" in report
    assert "calibration" in report

