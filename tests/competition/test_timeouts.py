from __future__ import annotations

import unittest
from types import SimpleNamespace

from vidaio_subnet_core.competition.timeouts import (
    competition_execution_lease_seconds,
    competition_execution_timeout_seconds,
    competition_scoring_timeout_seconds,
    estimated_encoding_runtime_seconds,
    estimated_vmaf_runtime_seconds,
)


def video(
    *,
    duration_seconds: float = 600,
    width: int = 3840,
    height: int = 2160,
    frame_count: int = 18_000,
):
    return SimpleNamespace(
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        frame_count=frame_count,
    )


class CompetitionTimeoutTests(unittest.TestCase):
    def test_execution_schedules_4k_chunks_across_four_parallel_lanes(self) -> None:
        four_chunks = tuple(video() for _ in range(4))
        five_chunks = tuple(video() for _ in range(5))

        self.assertEqual(estimated_encoding_runtime_seconds(four_chunks), 120)
        self.assertEqual(competition_execution_timeout_seconds(four_chunks), 240)
        self.assertEqual(competition_execution_lease_seconds(four_chunks), 360)

        self.assertEqual(estimated_encoding_runtime_seconds(five_chunks), 240)
        self.assertEqual(competition_execution_timeout_seconds(five_chunks), 360)
        self.assertEqual(competition_execution_lease_seconds(five_chunks), 480)

    def test_execution_scales_work_by_resolution_and_splits_long_inputs(self) -> None:
        full_hd = video(width=1920, height=1080)
        one_hour_full_hd = video(
            duration_seconds=3600,
            width=1920,
            height=1080,
            frame_count=108_000,
        )
        one_hour_4k = video(duration_seconds=3600, frame_count=108_000)

        self.assertEqual(estimated_encoding_runtime_seconds((full_hd,)), 120)
        self.assertEqual(
            estimated_encoding_runtime_seconds((one_hour_full_hd,)),
            180,
        )
        self.assertEqual(
            estimated_encoding_runtime_seconds((one_hour_4k,)),
            720,
        )

    def test_execution_assigns_a_full_chunk_to_sub_ten_minute_videos(self) -> None:
        short_4k = video(duration_seconds=5, frame_count=150)
        short_full_hd = video(
            duration_seconds=5,
            width=1920,
            height=1080,
            frame_count=150,
        )
        just_over_ten_minutes = video(
            duration_seconds=601,
            frame_count=18_030,
        )

        self.assertEqual(estimated_encoding_runtime_seconds((short_4k,)), 120)
        self.assertEqual(
            estimated_encoding_runtime_seconds((short_full_hd,)),
            120,
        )
        self.assertEqual(
            estimated_encoding_runtime_seconds((just_over_ten_minutes,)),
            240,
        )

    def test_scoring_sums_resolution_adjusted_frame_work_at_200_fps(self) -> None:
        four_k = video(frame_count=2_000)
        full_hd = video(width=1920, height=1080, frame_count=8_000)

        self.assertEqual(estimated_vmaf_runtime_seconds((four_k, full_hd)), 20)
        self.assertEqual(
            competition_scoring_timeout_seconds((four_k, full_hd)),
            140,
        )

    def test_empty_batches_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one video"):
            competition_execution_timeout_seconds(())
        with self.assertRaisesRegex(ValueError, "at least one video"):
            competition_scoring_timeout_seconds(())

    def test_manifest_minimums_are_not_prorated_for_partial_batches(self) -> None:
        one_short_video = (video(duration_seconds=5, frame_count=150),)

        self.assertEqual(
            competition_execution_timeout_seconds(
                one_short_video,
                minimum_timeout_seconds=600,
            ),
            600,
        )
        self.assertEqual(
            competition_execution_lease_seconds(
                one_short_video,
                minimum_timeout_seconds=600,
            ),
            720,
        )
        self.assertEqual(
            competition_scoring_timeout_seconds(
                one_short_video,
                minimum_timeout_seconds=300,
            ),
            300,
        )

    def test_workload_estimates_can_exceed_manifest_minimums(self) -> None:
        five_one_hour_4k_videos = tuple(
            video(duration_seconds=3600, frame_count=108_000)
            for _ in range(5)
        )

        self.assertGreater(
            competition_execution_timeout_seconds(
                five_one_hour_4k_videos,
                minimum_timeout_seconds=600,
            ),
            600,
        )
        self.assertGreater(
            competition_scoring_timeout_seconds(
                five_one_hour_4k_videos,
                minimum_timeout_seconds=300,
            ),
            300,
        )


if __name__ == "__main__":
    unittest.main()
