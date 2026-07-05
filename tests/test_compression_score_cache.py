import unittest

from services.scoring.compression_score_cache import (
    find_duplicate_compression_scores,
)


class CompressionScoreCacheTests(unittest.TestCase):
    def find_duplicates(
        self,
        cache,
        uid,
        input_id="video-1",
        vmaf_score=85.89,
        base_vmaf_score=88.22,
        vmaf_threshold=85.0,
        compression_rate=0.0907,
        final_score=0.5399,
    ):
        return find_duplicate_compression_scores(
            [uid],
            [input_id],
            [vmaf_score],
            [base_vmaf_score],
            [vmaf_threshold],
            [compression_rate],
            [final_score],
            cache,
        )

    def test_subsequent_uid_is_duplicate_across_scorer_batches(self):
        cache = {}

        self.assertEqual(self.find_duplicates(cache, uid=83), {})
        self.assertEqual(self.find_duplicates(cache, uid=41), {0: 83})

    def test_hidden_float_differences_and_derived_score_drift_still_match(self):
        cache = {}

        self.assertEqual(
            self.find_duplicates(
                cache,
                uid=41,
                vmaf_score=89.110001,
                base_vmaf_score=91.110001,
                vmaf_threshold=89.0,
                compression_rate=0.08240001,
                final_score=0.5714,
            ),
            {},
        )
        self.assertEqual(
            self.find_duplicates(
                cache,
                uid=83,
                vmaf_score=89.109999,
                base_vmaf_score=91.109999,
                vmaf_threshold=89.0,
                compression_rate=0.08239999,
                final_score=0.5715,
            ),
            {0: 41},
        )

    def test_subsequent_uid_loses_each_matching_synthetic_input_slot(self):
        cache = {}
        input_ids = [f"video-{idx}" for idx in range(5)]
        vmaf_scores = [85.89, 85.16, 85.88, 85.13, 85.04]
        base_vmaf_scores = [88.22, 87.50, 88.20, 87.46, 87.45]
        compression_rates = [0.0907, 0.0708, 0.0905, 0.0713, 0.0787]
        final_scores = [0.5399, 0.6341, 0.5409, 0.6307, 0.5887]

        first_duplicates = find_duplicate_compression_scores(
            [83] * 5,
            input_ids,
            vmaf_scores,
            base_vmaf_scores,
            [85.0] * 5,
            compression_rates,
            final_scores,
            cache,
        )
        subsequent_duplicates = find_duplicate_compression_scores(
            [41] * 5,
            input_ids,
            vmaf_scores,
            base_vmaf_scores,
            [85.0] * 5,
            compression_rates,
            final_scores,
            cache,
        )

        self.assertEqual(first_duplicates, {})
        self.assertEqual(subsequent_duplicates, {idx: 83 for idx in range(5)})

    def test_same_metrics_for_different_inputs_get_separate_slots(self):
        cache = {}

        self.assertEqual(self.find_duplicates(cache, uid=83, input_id="video-1"), {})
        self.assertEqual(self.find_duplicates(cache, uid=41, input_id="video-2"), {})

    def test_compression_rate_outside_tolerance_gets_a_separate_slot(self):
        cache = {}

        self.assertEqual(self.find_duplicates(cache, uid=83), {})
        self.assertEqual(
            self.find_duplicates(cache, uid=41, compression_rate=0.1010),
            {},
        )

    def test_small_compression_rate_drift_across_batches_is_duplicate(self):
        cache = {}
        input_ids = [f"video-{idx}" for idx in range(5)]
        vmaf_scores = [85.27, 85.40, 85.48, 85.22, 85.06]
        base_vmaf_scores = [87.49, 87.39, 87.61, 87.32, 86.92]

        first_duplicates = find_duplicate_compression_scores(
            [23] * 5,
            input_ids,
            vmaf_scores,
            base_vmaf_scores,
            [85.0] * 5,
            [0.0909, 0.1403, 0.1170, 0.1331, 0.1229],
            [0.5359, 0.4097, 0.4574, 0.4217, 0.4415],
            cache,
        )
        subsequent_duplicates = find_duplicate_compression_scores(
            [164] * 5,
            input_ids,
            vmaf_scores,
            base_vmaf_scores,
            [85.0] * 5,
            [0.0912, 0.1406, 0.1173, 0.1334, 0.1232],
            [0.5347, 0.4092, 0.4566, 0.4211, 0.4409],
            cache,
        )

        self.assertEqual(first_duplicates, {})
        self.assertEqual(subsequent_duplicates, {idx: 23 for idx in range(5)})

    def test_non_positive_result_does_not_claim_a_slot(self):
        cache = {}

        self.assertEqual(self.find_duplicates(cache, uid=83, final_score=0.0), {})
        self.assertEqual(self.find_duplicates(cache, uid=41), {})
        self.assertEqual(cache["video-1"].popitem()[1], [(0.0907, 41)])

    def test_same_uid_can_reuse_its_own_signature(self):
        cache = {}

        self.assertEqual(self.find_duplicates(cache, uid=83), {})
        self.assertEqual(self.find_duplicates(cache, uid=83), {})


if __name__ == "__main__":
    unittest.main()
