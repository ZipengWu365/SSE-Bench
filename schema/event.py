from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class Event:
    event_id: str
    dataset: str
    platform: str
    topic: str | None
    start_time: datetime
    engagement_series: list[int]
    sentiment_series: list[float] | None = None
    cascade_graph: dict[str, Any] | None = None
    is_sse: bool = False
    time_to_sse_minutes: int | None = None
    final_cascade_size: int = 0
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not self.engagement_series:
            raise ValueError("engagement_series must be non-empty")
        if any(value < 0 for value in self.engagement_series):
            raise ValueError("engagement_series must be non-negative")
        if self.final_cascade_size < self.engagement_series[-1]:
            raise ValueError("final_cascade_size must be at least the last observed size")
        if self.sentiment_series and len(self.sentiment_series) not in {1, len(self.engagement_series)}:
            raise ValueError("sentiment_series must have length 1 or match engagement_series")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "dataset": self.dataset,
            "platform": self.platform,
            "topic": self.topic,
            "start_time": self.start_time.isoformat(),
            "engagement_series": self.engagement_series,
            "sentiment_series": self.sentiment_series,
            "cascade_graph": self.cascade_graph,
            "is_sse": self.is_sse,
            "time_to_sse_minutes": self.time_to_sse_minutes,
            "final_cascade_size": self.final_cascade_size,
            "split": self.split,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Event":
        return cls(
            event_id=payload["event_id"],
            dataset=payload["dataset"],
            platform=payload["platform"],
            topic=payload.get("topic"),
            start_time=datetime.fromisoformat(payload["start_time"]),
            engagement_series=[int(value) for value in payload["engagement_series"]],
            sentiment_series=(
                None
                if payload.get("sentiment_series") is None
                else [float(value) for value in payload["sentiment_series"]]
            ),
            cascade_graph=payload.get("cascade_graph"),
            is_sse=bool(payload.get("is_sse", False)),
            time_to_sse_minutes=payload.get("time_to_sse_minutes"),
            final_cascade_size=int(payload.get("final_cascade_size", 0)),
            split=payload.get("split"),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def save_events_jsonl(events: Iterable[Event], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    with opener(destination, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(event.to_json())
            handle.write("\n")


def load_events_jsonl(path: str | Path) -> list[Event]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        return [Event.from_dict(json.loads(line)) for line in handle if line.strip()]
