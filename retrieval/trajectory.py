from __future__ import annotations

import numpy as np

from schema.event import Event


def _normalize(series: list[int], length: int) -> np.ndarray:
    values = np.asarray(series[:length], dtype=float)
    if values.size < length:
        values = np.pad(values, (0, length - values.size), constant_values=values[-1] if values.size else 0.0)
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else values


def top_k_similar(query: Event, candidates: list[Event], k: int = 5, steps: int = 18) -> list[tuple[str, float]]:
    query_vector = _normalize(query.engagement_series, steps)
    scored = []
    for candidate in candidates:
        candidate_vector = _normalize(candidate.engagement_series, steps)
        score = float(np.dot(query_vector, candidate_vector))
        scored.append((candidate.event_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
