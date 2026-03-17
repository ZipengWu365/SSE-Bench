from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from schema.event import Event, load_events_jsonl, save_events_jsonl


def test_event_roundtrip_dict_and_json() -> None:
    event = Event(
        event_id="toy::1",
        dataset="toy",
        platform="web",
        topic="demo",
        start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        engagement_series=[0, 1, 2, 5],
        sentiment_series=[0.25],
        cascade_graph=None,
        is_sse=True,
        time_to_sse_minutes=40,
        final_cascade_size=5,
        split="train",
        metadata={"slice_minutes": 20, "note": "ok"},
    )

    payload = event.to_dict()
    rebuilt = Event.from_dict(payload)
    assert rebuilt.event_id == event.event_id
    assert rebuilt.engagement_series == event.engagement_series
    assert rebuilt.final_cascade_size == event.final_cascade_size
    assert rebuilt.start_time == event.start_time
    assert rebuilt.metadata["slice_minutes"] == 20

    parsed = json.loads(event.to_json())
    assert parsed["event_id"] == "toy::1"


def test_event_validation_rejects_negative_engagement() -> None:
    with pytest.raises(ValueError, match="engagement_series must be non-negative"):
        Event(
            event_id="toy::bad",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[0, -1, 2],
            final_cascade_size=2,
        )


def test_event_validation_requires_final_at_least_last_observed() -> None:
    with pytest.raises(ValueError, match="final_cascade_size must be at least the last observed size"):
        Event(
            event_id="toy::bad2",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[1, 2, 3],
            final_cascade_size=2,
        )


def test_save_and_load_events_jsonl_gz(tmp_path: Path) -> None:
    events = [
        Event(
            event_id="toy::a",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[1, 2],
            final_cascade_size=2,
            split="train",
        ),
        Event(
            event_id="toy::b",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 2, tzinfo=timezone.utc),
            engagement_series=[1, 1, 1],
            final_cascade_size=1,
            split="test",
        ),
    ]
    path = tmp_path / "events.jsonl.gz"
    save_events_jsonl(events, path)
    loaded = load_events_jsonl(path)
    assert [event.event_id for event in loaded] == ["toy::a", "toy::b"]
    assert loaded[0].engagement_series == [1, 2]

