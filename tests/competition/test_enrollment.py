from __future__ import annotations

import asyncio
import importlib
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[2]
START = datetime(2026, 7, 16, 0, 0, tzinfo=timezone.utc)


def install_package_stub(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


if "bittensor" not in sys.modules:
    bittensor = types.ModuleType("bittensor")

    class Synapse(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        dendrite: object | None = None
        axon: object | None = None

    bittensor.Synapse = Synapse
    sys.modules["bittensor"] = bittensor

install_package_stub("vidaio_subnet_core", ROOT / "vidaio_subnet_core")
install_package_stub(
    "vidaio_subnet_core.competition", ROOT / "vidaio_subnet_core" / "competition"
)

protocol = importlib.import_module("vidaio_subnet_core.protocol")
from vidaio_subnet_core.competition.config import (  # noqa: E402
    CompetitionConfig,
    CompetitionManifest,
)
from vidaio_subnet_core.competition.enrollment import (  # noqa: E402
    CompetitionEnrollmentDispatcher,
    CompetitionMinerEndpoint,
)
from vidaio_subnet_core.competition.manager import CompetitionManager  # noqa: E402
from vidaio_subnet_core.competition.repository import (  # noqa: E402
    CompetitionRepository,
)
from vidaio_subnet_core.competition.state import ContenderState  # noqa: E402
from vidaio_subnet_core.competition.validation import (  # noqa: E402
    ValidationFinding,
    ValidationReason,
    ValidationReport,
    ValidationStatus,
)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FakeIntake:
    def __init__(self) -> None:
        self.calls = []
        self.credentials = []

    def finalize(self, submission, **kwargs):
        self.calls.append((submission, kwargs))
        self.credentials.append(submission.github_pat)
        submission.github_pat = ""
        return SimpleNamespace(repository_display="github.com/example/private")


class FakeForwarder:
    def __init__(self, *, wrong_hotkey: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.responses = []
        self.wrong_hotkey = wrong_hotkey

    async def __call__(self, endpoint, synapse, _timeout_seconds):
        if isinstance(synapse, protocol.CompetitionInvitationProtocol):
            self.calls.append(("invitation", endpoint.hotkey))
            participating = endpoint.hotkey == "participating-hotkey"
            synapse.invitation_response = protocol.CompetitionInvitationResponse(
                competition_id=synapse.competition_id,
                echo_nonce=synapse.invitation_nonce,
                participating=participating,
                supported_competition_type=(
                    protocol.CompetitionType.COMPRESSION if participating else None
                ),
                refusal_reason=None if participating else "not configured",
            )
        else:
            self.calls.append(("submission", endpoint.hotkey))
            synapse.submission_response = protocol.CompetitionSubmissionResponse(
                competition_id=synapse.competition_id,
                echo_nonce=synapse.request_nonce,
                status=protocol.CompetitionSubmissionStatus.READY,
                repository_url="https://github.com/example/private.git",
                github_pat="github_pat_" + "Z" * 40,
            )
        response_hotkey = "different-hotkey" if self.wrong_hotkey else endpoint.hotkey
        synapse.axon = SimpleNamespace(hotkey=response_hotkey)
        self.responses.append(synapse)
        return synapse


class EnrollmentDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.clock = FakeClock(START)
        self.repository = CompetitionRepository(f"sqlite:///{root / 'competition.db'}")
        self.config = CompetitionConfig(
            mode_enabled=True,
            database_url=f"sqlite:///{root / 'competition.db'}",
            artifact_root=root / "artifacts",
            artifact_backup_bucket="private-test-bucket",
            owner_id="test-validator",
        )
        manifest_payload = json.loads(
            (
                ROOT
                / "competitions/manifests/examples/compression-competition.json"
            ).read_text(encoding="utf-8")
        )
        self.manifest = CompetitionManifest.model_validate(manifest_payload)
        self.manager = CompetitionManager(
            self.config, self.repository, clock=self.clock
        )
        self.manager.register_manifest(self.manifest)
        self.manager.tick()
        self.competition = self.repository.get(self.manifest.competition_id)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def endpoints() -> list[CompetitionMinerEndpoint]:
        return [
            CompetitionMinerEndpoint(1, "participating-hotkey", "coldkey-1", object()),
            CompetitionMinerEndpoint(2, "declining-hotkey", "coldkey-2", object()),
        ]

    def test_invites_then_polls_participants_at_manifest_interval(self) -> None:
        forwarder = FakeForwarder()
        intake = FakeIntake()
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            intake,
            forwarder,
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, self.endpoints()))

        participating = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        declining = self.repository.get_contender(
            self.manifest.competition_id, "declining-hotkey"
        )
        self.assertEqual(participating.status, ContenderState.PARTICIPATING.value)
        self.assertEqual(participating.invitation_attempts, 1)
        self.assertEqual(participating.submission_poll_attempts, 1)
        self.assertEqual(declining.status, ContenderState.REJECTED.value)
        self.assertEqual(
            forwarder.calls,
            [
                ("invitation", "participating-hotkey"),
                ("invitation", "declining-hotkey"),
                ("submission", "participating-hotkey"),
            ],
        )
        self.assertEqual(len(intake.calls), 1)
        self.assertTrue(intake.credentials[0].startswith("github_pat_"))
        self.assertEqual(
            forwarder.responses[-1].submission_response.github_pat, ""
        )

        restarted_dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            intake,
            forwarder,
            owner_id="test-validator-after-restart",
            clock=self.clock,
        )
        asyncio.run(
            restarted_dispatcher.run_once(self.competition, self.endpoints())
        )
        self.assertEqual(len(forwarder.calls), 3)

        self.clock.now += timedelta(minutes=31)
        asyncio.run(
            restarted_dispatcher.run_once(self.competition, self.endpoints())
        )
        self.assertEqual(
            forwarder.calls[-1], ("submission", "participating-hotkey")
        )
        self.assertEqual(
            len([call for call in forwarder.calls if call[0] == "invitation"]), 2
        )

    def test_rejects_response_from_a_different_hotkey(self) -> None:
        forwarder = FakeForwarder(wrong_hotkey=True)
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            forwarder,
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, self.endpoints()[:1]))

        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.INVITED.value)
        self.assertEqual(contender.reason_code, "HOTKEY_MISMATCH")
        self.assertEqual(
            [call for call in forwarder.calls if call[0] == "submission"], []
        )

    def test_next_poll_returns_rejection_feedback_and_accepts_resubmission(self) -> None:
        forwarder = FakeForwarder()
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            forwarder,
            owner_id="test-validator",
            clock=self.clock,
        )
        endpoints = self.endpoints()[:1]
        asyncio.run(dispatcher.run_once(self.competition, endpoints))
        rejected_report = ValidationReport(
            status=ValidationStatus.REJECTED,
            reason_code=ValidationReason.OBFUSCATION_REVIEW,
            repository_tree_sha256="c" * 64,
            file_count=1,
            total_bytes=100,
            findings=(
                ValidationFinding(
                    ValidationReason.OBFUSCATION_REVIEW,
                    "solution/codec.pyc",
                    "opaque executable extension '.pyc' is not reviewable",
                    True,
                ),
            ),
        )
        self.repository.record_pinned_contender(
            competition_id=self.manifest.competition_id,
            hotkey="participating-hotkey",
            repository_url_hash="d" * 64,
            repository_display="github.com/example/private",
            pinned_commit_sha="a" * 40,
            pinned_tree_sha="b" * 40,
            latest_commit_time=self.clock.now.isoformat(),
            validation=rejected_report,
            now=self.clock.now,
            actor="test-validator",
        )

        self.clock.now += timedelta(minutes=31)
        asyncio.run(dispatcher.run_once(self.competition, endpoints))

        rejected_feedback = forwarder.responses[-1]
        self.assertEqual(rejected_feedback.last_submission_status.value, "REJECTED")
        self.assertEqual(
            rejected_feedback.last_submission_reason_code, "OBFUSCATION_REVIEW"
        )
        self.assertIn("solution/codec.pyc", rejected_feedback.last_submission_reason_detail)
        self.assertEqual(rejected_feedback.last_pinned_commit_sha, "a" * 40)
        self.assertEqual(rejected_feedback.submission_revision, 1)

        accepted_report = ValidationReport(
            status=ValidationStatus.ACCEPTED,
            reason_code=ValidationReason.ACCEPTED,
            repository_tree_sha256="e" * 64,
            file_count=1,
            total_bytes=100,
            findings=(),
        )
        self.repository.record_pinned_contender(
            competition_id=self.manifest.competition_id,
            hotkey="participating-hotkey",
            repository_url_hash="f" * 64,
            repository_display="github.com/example/corrected",
            pinned_commit_sha="b" * 40,
            pinned_tree_sha="c" * 40,
            latest_commit_time=self.clock.now.isoformat(),
            validation=accepted_report,
            now=self.clock.now,
            actor="test-validator",
        )
        self.clock.now += timedelta(minutes=31)
        asyncio.run(dispatcher.run_once(self.competition, endpoints))

        accepted_feedback = forwarder.responses[-1]
        self.assertEqual(accepted_feedback.last_submission_status.value, "ACCEPTED")
        self.assertIsNone(accepted_feedback.last_submission_reason_detail)
        self.assertEqual(accepted_feedback.last_pinned_commit_sha, "b" * 40)
        self.assertEqual(accepted_feedback.submission_revision, 2)


if __name__ == "__main__":
    unittest.main()
