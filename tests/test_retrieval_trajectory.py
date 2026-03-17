from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from retrieval.trajectory import trajectory_matrix
from schema.event import Event


def test_trajectory_matrix_pads_to_max_length_when_steps_none() -> None:
    events = [
        Event(
            event_id="toy::short",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[1, 2],
            final_cascade_size=2,
        ),
        Event(
            event_id="toy::long",
            dataset="toy",
            platform="web",
            topic=None,
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            engagement_series=[1, 2, 3, 4],
            final_cascade_size=4,
        ),
    ]

    matrix = trajectory_matrix(events, steps=None, log_scale=False, normalize=False)
    assert matrix.shape == (2, 4)
    assert np.allclose(matrix[0], np.array([1.0, 2.0, 2.0, 2.0]))
    assert np.allclose(matrix[1], np.array([1.0, 2.0, 3.0, 4.0]))

