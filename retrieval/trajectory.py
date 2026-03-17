from __future__ import annotations

import numpy as np

from schema.event import Event


def trajectory_vector(
    series: list[int],
    steps: int | None = None,
    log_scale: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    length = steps or len(series)
    values = np.asarray(series[:length], dtype=float)
    if values.size < length:
        values = np.pad(values, (0, length - values.size), constant_values=values[-1] if values.size else 0.0)
    if log_scale:
        values = np.log1p(values)
    if not normalize:
        return values
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else values


def trajectory_matrix(
    events: list[Event],
    steps: int | None = None,
    log_scale: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    if not events:
        return np.empty((0, 0), dtype=float)
    target_steps = steps or max(len(event.engagement_series) for event in events)
    return np.vstack(
        [
            trajectory_vector(
                event.engagement_series,
                steps=target_steps,
                log_scale=log_scale,
                normalize=normalize,
            )
            for event in events
        ]
    )


def top_k_similar(query: Event, candidates: list[Event], k: int = 5, steps: int = 18) -> list[tuple[str, float]]:
    query_vector = trajectory_vector(query.engagement_series, steps=steps)
    scored = []
    for candidate in candidates:
        candidate_vector = trajectory_vector(candidate.engagement_series, steps=steps)
        score = float(np.dot(query_vector, candidate_vector))
        scored.append((candidate.event_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
