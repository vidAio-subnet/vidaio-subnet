from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vidaio_subnet_core" / "competition" / "phase0.py"
SPEC = importlib.util.spec_from_file_location("vidaio_competition_phase0_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Phase 0 module from {MODULE_PATH}")
phase0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phase0
SPEC.loader.exec_module(phase0)


class ImageSizePolicyTests(unittest.TestCase):
    def test_exact_25_gb_is_accepted_and_one_byte_over_is_rejected(self) -> None:
        self.assertEqual(
            phase0.enforce_image_size(phase0.IMAGE_SIZE_LIMIT_BYTES),
            25_000_000_000,
        )
        with self.assertRaises(phase0.ImageSizeLimitExceeded):
            phase0.enforce_image_size(phase0.IMAGE_SIZE_LIMIT_BYTES + 1)

    def test_invalid_sizes_are_rejected(self) -> None:
        for value in (True, 1.5, "100"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                phase0.enforce_image_size(value)
        with self.assertRaises(ValueError):
            phase0.enforce_image_size(-1)

    def test_host_file_quota_primitive_stops_writer(self) -> None:
        result = phase0._file_quota_probe(128 * 1024)
        self.assertEqual(result.status, phase0.GateStatus.PASSED, result.as_dict())
        self.assertLessEqual(result.details["written_bytes"], 128 * 1024)


class RawPatBoundaryTests(unittest.TestCase):
    def test_wire_has_pat_but_persistent_record_does_not(self) -> None:
        pat = "github_pat_" + "abc123XYZ" * 5
        submission = phase0.RawPatSubmission(
            competition_id="competition-1",
            hotkey="hotkey-1",
            repository_url="https://github.com/example/private.git",
            github_pat=pat,
        )
        self.assertIn(pat, json.dumps(submission.to_wire()))
        self.assertNotIn(pat, json.dumps(submission.to_persistent_record()))

    def test_redactor_covers_nested_values_keys_patterns_and_exceptions(self) -> None:
        pat = "github_pat_" + "abc123XYZ" * 5
        redactor = phase0.SecretRedactor([pat])
        value = {
            "message": f"clone failed: {pat}",
            "token": "arbitrary-value",
            "nested": [RuntimeError(f"failed with {pat}")],
            "pattern_only": "ghp_" + "A" * 30,
        }
        sanitized = redactor.redact(value)
        self.assertFalse(redactor.contains_secret(sanitized))
        self.assertEqual(sanitized["token"], phase0.REDACTED)
        self.assertNotIn(pat, json.dumps(sanitized))

    def test_local_pat_gate_is_partial_until_actual_stack_is_exercised(self) -> None:
        result = phase0.pat_redaction_gate()
        self.assertEqual(result.status, phase0.GateStatus.PARTIAL, result.as_dict())
        self.assertFalse(result.details["actual_protocol_integration_proven"])


class CompetitionScoringTests(unittest.TestCase):
    def test_configurable_exponent_gives_thirty_minutes_ten_times_the_weight(
        self,
    ) -> None:
        short = phase0._duration_weight(10, 3600, 2.02)
        long = phase0._duration_weight(1800, 3600, 2.02)
        self.assertAlmostEqual(long / short, 10.0, places=2)

    def test_approved_60_25_15_components(self) -> None:
        rows = [
            phase0.CompetitionItemResult(
                "best", "short", 10, 1.0, Decimal("0.01"), True
            ),
            phase0.CompetitionItemResult(
                "best", "long", 100, 1.0, Decimal("0.01"), True
            ),
            phase0.CompetitionItemResult(
                "other", "short", 10, 0.5, Decimal("0.02"), True
            ),
            phase0.CompetitionItemResult(
                "other", "long", 100, 0.5, Decimal("0.02"), True
            ),
        ]
        scores = phase0.calculate_competition_scores(
            rows, max_video_length_seconds=100
        )
        self.assertAlmostEqual(scores["best"].final_score, 1.0)
        self.assertAlmostEqual(scores["other"].quality_aggregate, 0.5)
        self.assertAlmostEqual(scores["other"].cost_aggregate, 0.5)
        self.assertAlmostEqual(scores["other"].length_coverage, 1.0)
        self.assertAlmostEqual(scores["other"].final_score, 0.575)

    def test_cost_score_is_relative_to_cheapest_valid_contender(self) -> None:
        target = phase0.CompetitionItemResult(
            "target", "input", 10, 0.6, Decimal("0.02"), True
        )
        alone = phase0.calculate_competition_scores(
            [target],
            max_video_length_seconds=10,
        )["target"]
        with_competitors = phase0.calculate_competition_scores(
            [
                target,
                phase0.CompetitionItemResult(
                    "better", "input", 10, 1.0, Decimal("0.001"), True
                ),
                phase0.CompetitionItemResult(
                    "worse", "input", 10, 0.1, Decimal("1.00"), True
                ),
                phase0.CompetitionItemResult(
                    "failed-cheap", "input", 10, 1.0, Decimal("0.0001"), False
                ),
            ],
            max_video_length_seconds=10,
        )["target"]
        self.assertEqual(alone.quality_aggregate, with_competitors.quality_aggregate)
        self.assertEqual(alone.length_coverage, with_competitors.length_coverage)
        self.assertEqual(alone.cost_aggregate, 1.0)
        self.assertAlmostEqual(with_competitors.cost_aggregate, 0.05)
        self.assertGreater(alone.final_score, with_competitors.final_score)

    def test_longer_video_has_more_effect_and_failed_item_scores_zero(self) -> None:
        gate = phase0.score_formula_gate()
        self.assertEqual(gate.status, phase0.GateStatus.PASSED, gate.as_dict())
        self.assertTrue(gate.details["long_dominates"])
        self.assertTrue(gate.details["long_failure_penalized"])

    def test_every_contender_must_have_a_terminal_row_for_every_input(self) -> None:
        rows = [
            phase0.CompetitionItemResult(
                "complete", "a", 10, 1.0, Decimal("1"), True
            ),
            phase0.CompetitionItemResult(
                "complete", "b", 20, 1.0, Decimal("1"), True
            ),
            phase0.CompetitionItemResult(
                "incomplete", "a", 10, 1.0, Decimal("1"), True
            ),
        ]
        with self.assertRaisesRegex(ValueError, "complete evaluation index"):
            phase0.calculate_competition_scores(
                rows, max_video_length_seconds=20
            )

    def test_invalid_numeric_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            phase0.CompetitionItemResult("a", "b", 1, 1.0, 1.0, True)
        with self.assertRaises(ValueError):
            phase0.CompetitionItemResult(
                "a", "b", 1, 1.0, Decimal("NaN"), True
            )


class BillingAllocationTests(unittest.TestCase):
    def test_runtime_share_preserves_exact_total(self) -> None:
        result = phase0.allocate_batch_cost(
            Decimal("1.00"), {"short": 1.0, "long": 3.0}
        )
        self.assertEqual(result["short"], Decimal("0.2500"))
        self.assertEqual(result["long"], Decimal("0.7500"))
        self.assertEqual(sum(result.values()), Decimal("1.00"))

    def test_zero_runtime_uses_equal_share_and_preserves_remainder(self) -> None:
        result = phase0.allocate_batch_cost(
            Decimal("1.00"), {"b": 0.0, "a": 0.0, "c": 0.0}
        )
        self.assertEqual(sum(result.values()), Decimal("1.00"))
        self.assertEqual(result["a"], result["b"])
        self.assertGreater(result["c"], Decimal("0"))


if __name__ == "__main__":
    unittest.main()
