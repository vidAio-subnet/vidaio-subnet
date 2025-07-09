from unittest.mock import patch

import pytest

from services.video_scheduler.video_utils import (
    generate_chunk_timestamps, generate_youtube_chunk_timestamps
)


@pytest.mark.parametrize("random_start", [0, 0.125, 0.5, 0.75])
@pytest.mark.parametrize(
    "total_duration,clip_duration,expected_n_chunks",
    [
        (10.95, 5.0, 6),
        (5.1, 5.0, 1),
        (5.9, 5.0, 1),
        (3.5, 5.0, 1),
    ],
)
def test_generate_chunk_timestamps(
    random_start: float,
    total_duration: float,
    clip_duration: float,
    expected_n_chunks: int,
):
    with patch("random.uniform", return_value=random_start):
        chunks = generate_chunk_timestamps(total_duration, clip_duration)
    assert len(chunks) == expected_n_chunks
    for chunk_no, chunk_start, chunk_end in chunks:
        assert chunk_end - chunk_start == min(total_duration, clip_duration)
        assert chunk_start >= 0
        assert chunk_end <= total_duration
        assert chunk_start - int(chunk_start) in [0, random_start]


@pytest.mark.parametrize("random_start", [0, 0.125, 0.5, 0.75])
@pytest.mark.parametrize(
    "total_duration,clip_duration,expected_n_chunks",
    [
        (10.99, 5.0, 3),
        (5.1, 5.0, 1),
        (5.9, 5.0, 1),
    ],
)
def test_generate_youtube_chunk_timestamps(
    random_start: float,
    total_duration: float,
    clip_duration: float,
    expected_n_chunks: int,
):
    with patch("random.uniform", return_value=random_start):
        chunks = generate_youtube_chunk_timestamps(total_duration, clip_duration)
    assert len(chunks) == expected_n_chunks
    for chunk_no, chunk_start, chunk_end in chunks:
        assert chunk_end - chunk_start == min(total_duration, clip_duration)
        assert chunk_start >= 0
        assert chunk_end <= total_duration
        assert chunk_start - int(chunk_start) in [0, random_start]
