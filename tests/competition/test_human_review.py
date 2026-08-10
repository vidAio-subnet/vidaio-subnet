from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select, text


ROOT = Path(__file__).resolve().parents[2]
for package_name, package_path in (
    ("vidaio_subnet_core", ROOT / "vidaio_subnet_core"),
    ("vidaio_subnet_core.competition", ROOT / "vidaio_subnet_core" / "competition"),
):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

from vidaio_subnet_core.competition.config import load_manifest  # noqa: E402
from vidaio_subnet_core.competition.execution import (  # noqa: E402
    CompetitionExecutionCoordinator,
)
from vidaio_subnet_core.competition.models import (  # noqa: E402
    ContenderPerformanceHistory,
)
from vidaio_subnet_core.competition.repository import (  # noqa: E402
    CompetitionRepository,
)
from vidaio_subnet_core.competition.state import CompetitionState  # noqa: E402
from vidaio_subnet_core.competition.validation import (  # noqa: E402
    ValidationFinding,
    ValidationReason,
    ValidationReport,
    ValidationStatus,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


class HumanReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database_url = f"sqlite:///{Path(self.temp.name) / 'competition.db'}"
        self.repository = CompetitionRepository(self.database_url)
        self.manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        self.competition_id = self.manifest.competition_id
        self.repository.insert_manifest(self.manifest, now=NOW, actor="test")

    def tearDown(self) -> None:
        self.repository.engine.dispose()
        self.temp.cleanup()

    def advance_to(self, target: CompetitionState) -> None:
        sequence = (
            CompetitionState.ENROLLING,
            CompetitionState.FINALIZING_SUBMISSIONS,
            CompetitionState.VALIDATING,
            CompetitionState.BUILDING,
            CompetitionState.EVALUATING,
            CompetitionState.SCORING,
            CompetitionState.AWAITING_END_TIME,
            CompetitionState.COMPLETED,
        )
        current = CompetitionState(self.repository.get(self.competition_id).status)
        for next_state in sequence:
            if current == target:
                return
            self.repository.transition(
                self.competition_id,
                next_state,
                expected=current,
                now=NOW,
                actor="test",
            )
            current = next_state
        self.assertEqual(current, target)

    def record_contender(
        self,
        hotkey: str,
        *,
        validation_status: ValidationStatus = ValidationStatus.ACCEPTED,
        commit_time: datetime = NOW,
    ) -> None:
        if validation_status == ValidationStatus.REVIEW_REQUIRED:
            reason = ValidationReason.REMOTE_DOWNLOAD_REVIEW
            findings = (
                ValidationFinding(
                    reason,
                    "miner/compression/app.py",
                    "remote download behavior requires review",
                    False,
                ),
            )
        else:
            reason = ValidationReason.ACCEPTED
            findings = ()
        self.repository.record_pinned_contender(
            competition_id=self.competition_id,
            hotkey=hotkey,
            repository_url_hash=(hotkey[0] if hotkey else "a") * 64,
            repository_display=f"github.com/test/{hotkey}",
            pinned_commit_sha=(hotkey[0] if hotkey else "b") * 40,
            pinned_tree_sha="f" * 40,
            latest_commit_time=commit_time.isoformat(),
            validation=ValidationReport(
                validation_status,
                reason,
                "c" * 64,
                1,
                100,
                findings,
            ),
            now=NOW,
            actor="test",
        )

    def add_scored_history(self, hotkey: str, cost: str) -> None:
        with self.repository._sessions.begin() as session:
            session.add(
                ContenderPerformanceHistory(
                    competition_id=self.competition_id,
                    hotkey=hotkey,
                    evaluation_id="same-video",
                    batch_id=f"batch-{hotkey}",
                    canonical_batch_index=0,
                    attempt=1,
                    idempotency_key=f"{self.competition_id}:{hotkey}:same-video:1",
                    status="SCORED",
                    task_type="COMPRESSION",
                    scale_factor=1,
                    duration_seconds=60,
                    length_weight=1.0,
                    vmaf_threshold=89,
                    vmaf_score=95,
                    compression_rate=2.0,
                    quality_component=0,
                    cost_efficiency_component=0,
                    completion_value=0,
                    estimated_cost_usd=Decimal(cost),
                    currency="USD",
                    scoring_version=self.manifest.scoring_version,
                    created_at=NOW.isoformat(),
                    updated_at=NOW.isoformat(),
                )
            )

    def test_readability_review_releases_validation_hold(self) -> None:
        self.record_contender(
            "review-me", validation_status=ValidationStatus.REVIEW_REQUIRED
        )
        self.advance_to(CompetitionState.VALIDATING)

        review = self.repository.resolve_readability_review(
            self.competition_id,
            "review-me",
            accepted=True,
            operator_identity="reviewer@example.com",
            reason="The URL is a pinned package source used only at image build time.",
            now=NOW,
        )

        contender = self.repository.get_contender(self.competition_id, "review-me")
        self.assertEqual(contender.validation_status, "ACCEPTED")
        self.assertEqual(contender.status, "ACCEPTED")
        self.assertEqual(review.review_type, "READABILITY_ELIGIBILITY")
        self.assertEqual(json.loads(review.decision_json)["decision"], "ACCEPTED")

    def test_disqualification_recalculates_relative_cost_and_ranking(self) -> None:
        for hotkey, cost in (("cheap", "0.01"), ("middle", "0.02"), ("costly", "0.04")):
            self.record_contender(hotkey)
            self.add_scored_history(hotkey, cost)
        self.advance_to(CompetitionState.SCORING)
        self.repository.score_competition(
            self.competition_id,
            self.manifest,
            now=NOW,
            actor="test",
        )
        before = self.repository.get_contender(self.competition_id, "middle")
        self.assertAlmostEqual(before.cost_efficiency_aggregate, 0.5)
        before_score = before.final_score
        self.repository.transition(
            self.competition_id,
            CompetitionState.AWAITING_END_TIME,
            expected=CompetitionState.SCORING,
            now=NOW,
            actor="test",
        )

        review = self.repository.disqualify_contender(
            self.competition_id,
            "cheap",
            operator_identity="reviewer@example.com",
            reason="Submission violated the published provenance rule.",
            now=NOW,
        )

        middle = self.repository.get_contender(self.competition_id, "middle")
        cheap = self.repository.get_contender(self.competition_id, "cheap")
        self.assertAlmostEqual(middle.cost_efficiency_aggregate, 1.0)
        self.assertGreater(middle.final_score, before_score)
        self.assertTrue(cheap.manual_disqualified)
        self.assertFalse(cheap.eligible)
        self.assertIsNone(cheap.final_score)
        self.assertEqual(
            cheap.manual_disqualification_review_id,
            review.id,
        )
        with self.repository._sessions() as session:
            raw_vmaf = session.scalar(
                select(ContenderPerformanceHistory.vmaf_score).where(
                    ContenderPerformanceHistory.hotkey == "cheap"
                )
            )
        self.assertEqual(raw_vmaf, 95)

        self.repository.transition(
            self.competition_id,
            CompetitionState.COMPLETED,
            expected=CompetitionState.AWAITING_END_TIME,
            now=NOW + timedelta(minutes=1),
            actor="test",
        )
        self.assertEqual(
            self.repository.get_contender(self.competition_id, "middle").final_rank,
            1,
        )
        self.assertIsNone(
            self.repository.get_contender(self.competition_id, "cheap").final_rank
        )

    def test_exact_tie_review_overrides_only_deterministic_tie_fallback(self) -> None:
        self.record_contender("early", commit_time=NOW - timedelta(days=1))
        self.record_contender("late", commit_time=NOW)
        with self.repository.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE contender_metadata SET status='SCORED', final_score=0.75 "
                    "WHERE competition_id=:competition_id"
                ),
                {"competition_id": self.competition_id},
            )
        self.advance_to(CompetitionState.AWAITING_END_TIME)

        self.repository.order_exact_tie(
            self.competition_id,
            ["late", "early"],
            operator_identity="reviewer@example.com",
            reason="Reviewed exact tie order.",
            now=NOW,
        )
        self.repository.transition(
            self.competition_id,
            CompetitionState.COMPLETED,
            expected=CompetitionState.AWAITING_END_TIME,
            now=NOW + timedelta(minutes=1),
            actor="test",
        )

        self.assertEqual(
            self.repository.get_contender(self.competition_id, "late").final_rank,
            1,
        )
        self.assertEqual(
            self.repository.get_contender(self.competition_id, "early").final_rank,
            2,
        )

    def test_review_packet_previews_ranking_without_persisting_final_rank(
        self,
    ) -> None:
        self.record_contender("early", commit_time=NOW - timedelta(days=1))
        self.record_contender("late", commit_time=NOW)
        self.add_scored_history("early", "0.01")
        self.add_scored_history("late", "0.02")
        with self.repository.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE contender_metadata SET status='SCORED', "
                    "final_score=0.75, media_score_aggregate=0.8, "
                    "cost_efficiency_aggregate=0.6, length_coverage=1.0, "
                    "average_vmaf=95.0, average_compression_rate=2.0, "
                    "successful_items=1, failed_items=0, "
                    "estimated_cost_usd=CASE hotkey "
                    "WHEN 'early' THEN 0.01 ELSE 0.02 END "
                    "WHERE competition_id=:competition_id"
                ),
                {"competition_id": self.competition_id},
            )
            connection.execute(
                text(
                    "UPDATE contender_performance_history SET status='FAILED', "
                    "reason_code='OUTPUT_INVALID' "
                    "WHERE competition_id=:competition_id AND hotkey='late'"
                ),
                {"competition_id": self.competition_id},
            )
            connection.execute(
                text(
                    "UPDATE contender_metadata SET successful_items=0, "
                    "failed_items=1 WHERE competition_id=:competition_id "
                    "AND hotkey='late'"
                ),
                {"competition_id": self.competition_id},
            )
        self.advance_to(CompetitionState.AWAITING_END_TIME)

        packet = self.repository.competition_review_packet(self.competition_id)

        self.assertTrue(packet["provisional"])
        self.assertEqual(
            [row["contender_id"] for row in packet["ranking"]],
            ["early", "late"],
        )
        self.assertEqual([row["rank"] for row in packet["ranking"]], [1, 2])
        self.assertEqual(
            packet["exact_tie_groups"],
            [
                {
                    "final_score": 0.75,
                    "contender_ids": ["early", "late"],
                    "effective_order": ["early", "late"],
                    "decision_source": "COMMIT_METADATA",
                    "review_id": None,
                }
            ],
        )
        self.assertEqual(
            packet["review_queue"]["unreviewed_exact_tie_scores"],
            [0.75],
        )
        late = packet["ranking"][1]
        self.assertEqual(late["score_components"]["media_score"], 0.8)
        self.assertEqual(late["cost"]["source"], "ESTIMATED")
        self.assertEqual(late["cost"]["estimated_cost_usd"], "0.0200000000")
        self.assertEqual(
            late["failures"],
            [
                {
                    "evaluation_id": "same-video",
                    "attempt": 1,
                    "status": "FAILED",
                    "reason_code": "OUTPUT_INVALID",
                }
            ],
        )
        self.assertEqual(late["pinned_commit"]["committer_time"], NOW.isoformat())
        self.assertEqual(late["static_validation"]["status"], "ACCEPTED")
        self.assertIsNone(
            self.repository.get_contender(self.competition_id, "early").final_rank
        )
        self.assertIsNone(
            self.repository.get_contender(self.competition_id, "late").final_rank
        )

        cli = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/competition_review.py"),
                "--database-url",
                self.database_url,
                "list",
                "--competition-id",
                self.competition_id,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(cli.returncode, 0, cli.stderr)
        cli_packet = json.loads(cli.stdout)
        self.assertEqual(cli_packet["ranking"], packet["ranking"])
        self.assertEqual(cli_packet["exact_tie_groups"], packet["exact_tie_groups"])

        review = self.repository.order_exact_tie(
            self.competition_id,
            ["late", "early"],
            operator_identity="reviewer@example.com",
            reason="Reviewed exact tie order.",
            now=NOW,
        )
        reviewed_packet = self.repository.competition_review_packet(
            self.competition_id
        )

        self.assertEqual(
            [row["contender_id"] for row in reviewed_packet["ranking"]],
            ["late", "early"],
        )
        self.assertEqual(
            reviewed_packet["exact_tie_groups"][0]["decision_source"],
            "HUMAN_REVIEW",
        )
        self.assertEqual(
            reviewed_packet["exact_tie_groups"][0]["review_id"],
            review.id,
        )
        self.assertEqual(reviewed_packet["reviews"][0]["id"], review.id)

    def test_non_tied_contenders_cannot_be_manually_reordered(self) -> None:
        self.record_contender("one")
        self.record_contender("two")
        with self.repository.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE contender_metadata SET status='SCORED', "
                    "final_score=CASE hotkey WHEN 'one' THEN 0.8 ELSE 0.7 END "
                    "WHERE competition_id=:competition_id"
                ),
                {"competition_id": self.competition_id},
            )
        self.advance_to(CompetitionState.AWAITING_END_TIME)

        with self.assertRaisesRegex(ValueError, "non-tied"):
            self.repository.order_exact_tie(
                self.competition_id,
                ["two", "one"],
                operator_identity="reviewer@example.com",
                reason="Not allowed.",
                now=NOW,
            )

    def test_restart_recalculation_is_database_only(self) -> None:
        class RecoveryRepository:
            calls = []

            @staticmethod
            def get(_competition_id):
                return SimpleNamespace(scores_need_recalculation=True)

            def recalculate_competition_scores(
                self, competition_id, *, now, actor
            ):
                self.calls.append((competition_id, now, actor))

        class ModalMustNotRun:
            def __getattr__(self, name):
                raise AssertionError(f"Modal operation unexpectedly requested: {name}")

        class RecoveryCoordinator(CompetitionExecutionCoordinator):
            cleanup_called = False

            async def _terminate_contender_sandboxes(self, _manifest):
                self.cleanup_called = True

        repository = RecoveryRepository()
        coordinator = RecoveryCoordinator(
            manager=None,
            repository=repository,
            build_service=ModalMustNotRun(),
            sandbox_runner=ModalMustNotRun(),
            artifact_root=ROOT,
            actor="validator:test",
            accepted_build_statuses=frozenset({"MODAL_ACCEPTED"}),
            clock=lambda: NOW,
        )

        asyncio.run(
            coordinator._advance(
                self.manifest,
                CompetitionState.AWAITING_END_TIME,
            )
        )

        self.assertEqual(
            repository.calls,
            [(self.competition_id, NOW, "validator:test")],
        )
        self.assertTrue(coordinator.cleanup_called)


if __name__ == "__main__":
    unittest.main()
