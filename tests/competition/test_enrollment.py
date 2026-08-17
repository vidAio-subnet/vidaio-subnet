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
from vidaio_subnet_core.competition.state import (  # noqa: E402
    CompetitionState,
    ContenderState,
)
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
    def __init__(
        self,
        *,
        wrong_hotkey: bool = False,
        participating_hotkeys: set[str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.responses = []
        self.wrong_hotkey = wrong_hotkey
        self.participating_hotkeys = (
            {"participating-hotkey"}
            if participating_hotkeys is None
            else participating_hotkeys
        )

    async def __call__(self, endpoint, synapse, _timeout_seconds):
        if isinstance(synapse, protocol.CompetitionInvitationProtocol):
            self.calls.append(("invitation", endpoint.hotkey))
            participating = endpoint.hotkey in self.participating_hotkeys
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


class MissingCompetitionEndpointForwarder:
    def __init__(self, endpoint_name: str, *, status_code: int = 404) -> None:
        self.endpoint_name = endpoint_name
        self.status_code = status_code

    async def __call__(self, endpoint, synapse, _timeout_seconds):
        is_invitation = isinstance(
            synapse, protocol.CompetitionInvitationProtocol
        )
        current_endpoint = "invitation" if is_invitation else "submission"
        missing = current_endpoint == self.endpoint_name
        if is_invitation and not missing:
            synapse.invitation_response = protocol.CompetitionInvitationResponse(
                competition_id=synapse.competition_id,
                echo_nonce=synapse.invitation_nonce,
                participating=True,
                supported_competition_type=protocol.CompetitionType.COMPRESSION,
            )
        status_code = self.status_code if missing else 200
        status_message = "Not Found" if missing else "Success"
        synapse.dendrite = SimpleNamespace(
            status_code=status_code,
            status_message=status_message,
        )
        synapse.axon = SimpleNamespace(
            hotkey=endpoint.hotkey,
            status_code=status_code,
            status_message=status_message,
        )
        return synapse


class WrongCompetitionForwarder:
    async def __call__(self, endpoint, synapse, _timeout_seconds):
        synapse.invitation_response = protocol.CompetitionInvitationResponse(
            competition_id="different-competition",
            echo_nonce=synapse.invitation_nonce,
            participating=False,
        )
        synapse.dendrite = SimpleNamespace(
            status_code=200,
            status_message="Success",
        )
        synapse.axon = SimpleNamespace(
            hotkey=endpoint.hotkey,
            status_code=200,
            status_message="Success",
        )
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
        self.assertCountEqual(
            forwarder.calls[:2],
            [
                ("invitation", "participating-hotkey"),
                ("invitation", "declining-hotkey"),
            ],
        )
        self.assertEqual(
            forwarder.calls[2], ("submission", "participating-hotkey")
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
            len([call for call in forwarder.calls if call[0] == "invitation"]), 3
        )

    def test_declined_miner_is_reinvited_after_enabling_competition(self) -> None:
        endpoint = self.endpoints()[1]
        forwarder = FakeForwarder(participating_hotkeys=set())
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            forwarder,
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, [endpoint]))

        contender = self.repository.get_contender(
            self.manifest.competition_id, endpoint.hotkey
        )
        self.assertEqual(contender.status, ContenderState.REJECTED.value)
        self.assertEqual(contender.reason_code, "INVITATION_DECLINED")

        forwarder.participating_hotkeys.add(endpoint.hotkey)
        self.clock.now += self.manifest.contender_ping_interval
        asyncio.run(dispatcher.run_once(self.competition, [endpoint]))

        contender = self.repository.get_contender(
            self.manifest.competition_id, endpoint.hotkey
        )
        self.assertEqual(contender.status, ContenderState.PARTICIPATING.value)
        self.assertIsNone(contender.reason_code)
        self.assertEqual(contender.invitation_attempts, 2)
        self.assertIn(("submission", endpoint.hotkey), forwarder.calls)

    def test_below_minimum_alpha_stake_is_retried_until_miner_qualifies(self) -> None:
        self.competition.manifest_json = self.manifest.model_copy(
            update={"minimum_alpha_stake": 10}
        ).normalized_json()
        endpoints = [
            CompetitionMinerEndpoint(
                1, "participating-hotkey", "coldkey-1", object(), alpha_stake=9
            )
        ]
        forwarder = FakeForwarder()
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            forwarder,
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, endpoints))

        self.assertEqual(forwarder.calls, [("invitation", "participating-hotkey")])
        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.INVITED.value)
        self.assertEqual(contender.reason_code, "ALPHA_STAKE_BELOW_MINIMUM")
        self.assertIn("Observed alpha stake 9", contender.reason_detail)
        self.assertEqual(contender.invitation_attempts, 1)
        first_invited_at = contender.last_invited_at
        rejection = next(
            response
            for response in forwarder.responses
            if isinstance(response, protocol.CompetitionInvitationProtocol)
            and response.axon.hotkey == "participating-hotkey"
        )
        self.assertEqual(
            rejection.eligibility_reason_code, "ALPHA_STAKE_BELOW_MINIMUM"
        )
        self.assertEqual(rejection.observed_alpha_stake, 9)
        self.assertEqual(rejection.minimum_alpha_stake, 10)

        # Rows written by older validators were terminally rejected. They are
        # requeued on the next metagraph snapshot during enrollment.
        with self.repository.engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE contender_metadata SET status='REJECTED' "
                "WHERE competition_id=? AND hotkey=?",
                (self.manifest.competition_id, "participating-hotkey"),
            )

        self.clock.now += self.manifest.contender_ping_interval
        asyncio.run(dispatcher.run_once(self.competition, endpoints))

        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.INVITED.value)
        self.assertEqual(contender.reason_code, "ALPHA_STAKE_BELOW_MINIMUM")
        self.assertEqual(contender.invitation_attempts, 2)
        self.assertNotEqual(contender.last_invited_at, first_invited_at)

        self.clock.now += self.manifest.contender_ping_interval
        qualified_endpoints = [
            CompetitionMinerEndpoint(
                1, "participating-hotkey", "coldkey-1", object(), alpha_stake=10
            )
        ]
        asyncio.run(dispatcher.run_once(self.competition, qualified_endpoints))

        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.PARTICIPATING.value)
        self.assertIsNone(contender.reason_code)
        self.assertIsNone(contender.reason_detail)
        self.assertEqual(contender.invitation_attempts, 3)
        self.assertIn(("submission", "participating-hotkey"), forwarder.calls)

    def test_unavailable_alpha_stake_is_reported_and_persisted(self) -> None:
        self.competition.manifest_json = self.manifest.model_copy(
            update={"minimum_alpha_stake": 10}
        ).normalized_json()
        endpoint = CompetitionMinerEndpoint(
            1, "participating-hotkey", "coldkey-1", object(), alpha_stake=None
        )
        forwarder = FakeForwarder()
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            forwarder,
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, [endpoint]))

        rejected = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(rejected.status, ContenderState.REJECTED.value)
        self.assertEqual(rejected.reason_code, "ALPHA_STAKE_UNAVAILABLE")
        invitation = forwarder.responses[0]
        self.assertEqual(
            invitation.eligibility_reason_code, "ALPHA_STAKE_UNAVAILABLE"
        )
        self.assertIsNone(invitation.observed_alpha_stake)
        self.assertEqual(invitation.minimum_alpha_stake, 10)
        self.assertNotIn(("submission", "participating-hotkey"), forwarder.calls)

    def test_final_alpha_stake_uses_pre_transition_snapshot(self) -> None:
        self.competition.manifest_json = self.manifest.model_copy(
            update={"minimum_alpha_stake": 10}
        ).normalized_json()
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            FakeForwarder(),
            owner_id="test-validator",
            clock=self.clock,
        )
        asyncio.run(
            dispatcher.run_once(
                self.competition,
                [
                    CompetitionMinerEndpoint(
                        1,
                        "participating-hotkey",
                        "coldkey-1",
                        object(),
                        alpha_stake=10,
                    )
                ],
            )
        )
        accepted_report = ValidationReport(
            status=ValidationStatus.ACCEPTED,
            reason_code=ValidationReason.ACCEPTED,
            repository_tree_sha256="c" * 64,
            file_count=1,
            total_bytes=100,
            findings=(),
        )
        self.repository.record_pinned_contender(
            competition_id=self.manifest.competition_id,
            hotkey="participating-hotkey",
            repository_url_hash="d" * 64,
            repository_display="github.com/example/private",
            pinned_commit_sha="a" * 40,
            pinned_tree_sha="b" * 40,
            latest_commit_time=self.clock.now.isoformat(),
            validation=accepted_report,
            now=self.clock.now,
            actor="test-validator",
        )

        self.clock.now = self.manifest.contender_finalisation_time

        def finalize_alpha_stake(competition, now) -> None:
            self.assertEqual(competition.status, CompetitionState.ENROLLING.value)
            rejected = self.repository.reject_ineligible_contenders(
                competition.competition_id,
                {"participating-hotkey": (1, 9)},
                10,
                now=now,
            )
            self.assertEqual(rejected, 1)

        self.manager.tick(before_finalization=finalize_alpha_stake)
        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.REJECTED.value)
        self.assertEqual(contender.validation_status, ValidationStatus.REJECTED.value)
        self.assertEqual(contender.reason_code, "ALPHA_STAKE_BELOW_MINIMUM")

        # Once the scheduler transitions, later stake observations cannot
        # rewrite the finalized contender set.
        second_result = self.repository.reject_ineligible_contenders(
            self.manifest.competition_id,
            {"participating-hotkey": (1, 11)},
            10,
            now=self.clock.now,
        )
        self.assertEqual(second_result, 0)
        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.REJECTED.value)

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

    def test_missing_invitation_endpoint_with_404_is_contender_not_found(self) -> None:
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            MissingCompetitionEndpointForwarder("invitation"),
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, self.endpoints()[:1]))

        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.INVITED.value)
        self.assertEqual(contender.reason_code, "CONTENDER_NOT_FOUND")
        self.assertEqual(
            contender.reason_detail,
            "miner does not expose the competition invitation endpoint",
        )

    def test_untouched_invitation_response_is_contender_not_found(self) -> None:
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            MissingCompetitionEndpointForwarder("invitation", status_code=200),
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, self.endpoints()[:1]))

        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.INVITED.value)
        self.assertEqual(contender.reason_code, "CONTENDER_NOT_FOUND")
        self.assertEqual(
            contender.reason_detail,
            "miner does not expose the competition invitation endpoint",
        )

    def test_nonempty_wrong_competition_remains_id_mismatch(self) -> None:
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            WrongCompetitionForwarder(),
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, self.endpoints()[:1]))

        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.INVITED.value)
        self.assertEqual(contender.reason_code, "COMPETITION_ID_MISMATCH")
        self.assertEqual(
            contender.reason_detail,
            "invitation response belongs to another competition",
        )

    def test_missing_submission_endpoint_is_contender_not_found(self) -> None:
        dispatcher = CompetitionEnrollmentDispatcher(
            self.repository,
            FakeIntake(),
            MissingCompetitionEndpointForwarder("submission", status_code=200),
            owner_id="test-validator",
            clock=self.clock,
        )

        asyncio.run(dispatcher.run_once(self.competition, self.endpoints()[:1]))

        contender = self.repository.get_contender(
            self.manifest.competition_id, "participating-hotkey"
        )
        self.assertEqual(contender.status, ContenderState.PARTICIPATING.value)
        self.assertEqual(contender.reason_code, "CONTENDER_NOT_FOUND")
        self.assertEqual(
            contender.reason_detail,
            "miner does not expose the competition submission endpoint",
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
