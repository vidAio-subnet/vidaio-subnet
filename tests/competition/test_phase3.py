from __future__ import annotations

# ruff: noqa: E402 -- package stubs keep focused tests independent of Bittensor.

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def install_package_stub(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


install_package_stub("vidaio_subnet_core", ROOT / "vidaio_subnet_core")
install_package_stub(
    "vidaio_subnet_core.competition", ROOT / "vidaio_subnet_core" / "competition"
)

from vidaio_subnet_core.competition.build import (
    BuildEvidence,
    BuildReason,
    BuildRequest,
    CompetitionBuildService,
    ModalImageBuildBackend,
    ModalImageBuilder,
    TrustedBuildError,
    TrustedImageBuilder,
)
from vidaio_subnet_core.competition.config import CompetitionConfig, load_manifest
from vidaio_subnet_core.competition.contracts import (
    CompetitionCompressionItem,
    CompetitionCompressionRequest,
)
from vidaio_subnet_core.competition.modal_runner import (
    MAX_SANDBOX_LIFETIME,
    CompetitionModalRunner,
    ModalSandboxBackend,
    SandboxHandle,
    SandboxResourceAllocation,
    SandboxRunnerError,
    _output_volume_name,
)
from vidaio_subnet_core.competition.execution import CompetitionExecutionCoordinator
from vidaio_subnet_core.competition.intake import pinned_repository_source
from vidaio_subnet_core.competition.manager import CompetitionManager
from vidaio_subnet_core.competition.phase0 import IMAGE_SIZE_LIMIT_BYTES
from vidaio_subnet_core.competition.repository import CompetitionRepository
from vidaio_subnet_core.competition.state import CompetitionState
from vidaio_subnet_core.competition.validation import (
    ValidationReason,
    ValidationReport,
    ValidationStatus,
)


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
IMAGE_ID = "im-immutable-phase3"
IMAGE_DIGEST = "sha256:" + "d" * 64
TREE_SHA = "b" * 40


class FakeBuilder:
    builder_id = "trusted-quota-builder"

    def __init__(self, evidence: BuildEvidence):
        self.evidence = evidence

    def build(self, request: BuildRequest) -> BuildEvidence:
        return self.evidence


def build_evidence(**overrides) -> BuildEvidence:
    values = {
        "builder_id": "trusted-quota-builder",
        "source_tree_sha": TREE_SHA,
        "image_id": IMAGE_ID,
        "image_digest": IMAGE_DIGEST,
        "image_size_bytes": IMAGE_SIZE_LIMIT_BYTES,
        "quota_limit_bytes": IMAGE_SIZE_LIMIT_BYTES,
        "quota_enforced_during_build": True,
        "built_at": NOW,
    }
    values.update(overrides)
    return BuildEvidence(**values)


class TrustedBuildTests(unittest.TestCase):
    def test_exact_25gb_is_accepted_and_bound_to_pinned_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = BuildRequest("competition", "hotkey", Path(temp), TREE_SHA)
            evidence = TrustedImageBuilder(
                FakeBuilder(build_evidence()),
                trusted_builder_ids=frozenset({"trusted-quota-builder"}),
            ).build(request)
        self.assertEqual(evidence.image_size_bytes, IMAGE_SIZE_LIMIT_BYTES)

    def test_oversize_unenforced_and_source_mismatch_fail_closed(self) -> None:
        cases = (
            (
                {"image_size_bytes": IMAGE_SIZE_LIMIT_BYTES + 1},
                BuildReason.IMAGE_SIZE_LIMIT_EXCEEDED,
            ),
            (
                {"quota_enforced_during_build": False},
                BuildReason.BUILD_QUOTA_NOT_ENFORCED,
            ),
            (
                {"source_tree_sha": "c" * 40},
                BuildReason.SOURCE_EVIDENCE_MISMATCH,
            ),
        )
        for overrides, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temp:
                builder = TrustedImageBuilder(
                    FakeBuilder(build_evidence(**overrides)),
                    trusted_builder_ids=frozenset({"trusted-quota-builder"}),
                )
                with self.assertRaises(TrustedBuildError) as captured:
                    builder.build(
                        BuildRequest("competition", "hotkey", Path(temp), TREE_SHA)
                    )
                self.assertEqual(captured.exception.reason_code, reason)

    def test_build_service_persists_rejection_reason(self) -> None:
        class FakeRepository:
            rejection = None

            def get_contender(self, competition_id, hotkey):
                return types.SimpleNamespace(pinned_tree_sha=TREE_SHA)

            def record_build_rejection(self, **kwargs):
                self.rejection = kwargs

        repository = FakeRepository()
        builder = TrustedImageBuilder(
            FakeBuilder(build_evidence(quota_enforced_during_build=False)),
            trusted_builder_ids=frozenset({"trusted-quota-builder"}),
        )
        service = CompetitionBuildService(
            repository, builder, actor="validator:test", clock=lambda: NOW
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaises(TrustedBuildError),
        ):
            service.build_contender("competition", "hotkey", Path(temp))
        self.assertEqual(
            repository.rejection["reason_code"],
            BuildReason.BUILD_QUOTA_NOT_ENFORCED.value,
        )

    def test_build_service_keeps_infrastructure_failures_retryable(self) -> None:
        class BrokenBuilder:
            def build(self, request):
                raise RuntimeError("temporary Modal outage")

        class FakeRepository:
            retry = None

            def get_contender(self, competition_id, hotkey):
                return types.SimpleNamespace(pinned_tree_sha=TREE_SHA)

            def record_build_retryable_failure(self, **kwargs):
                self.retry = kwargs

        repository = FakeRepository()
        service = CompetitionBuildService(
            repository, BrokenBuilder(), actor="validator:test", clock=lambda: NOW
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaises(TrustedBuildError) as captured,
        ):
            service.build_contender("competition", "hotkey", Path(temp))
        self.assertEqual(
            captured.exception.reason_code, BuildReason.BUILD_INFRASTRUCTURE_ERROR
        )
        self.assertEqual(repository.retry["reason_code"], "BUILD_INFRASTRUCTURE_ERROR")

    def test_modal_builder_records_unattested_image_binding(self) -> None:
        class FakeImage:
            object_id = "im-development"

            def build(self, app):
                self.app = app
                return self

        class FakeModal:
            class App:
                @staticmethod
                def lookup(name, **kwargs):
                    return types.SimpleNamespace(app_id="ap-build")

            class Image:
                @staticmethod
                def from_dockerfile(path, **kwargs):
                    return FakeImage()

        request = BuildRequest("competition", "hotkey", ROOT, TREE_SHA)
        evidence = ModalImageBuilder(ModalImageBuildBackend(modal_api=FakeModal)).build(
            request
        )
        self.assertEqual(evidence.image_id, "im-development")
        self.assertFalse(evidence.quota_enforced_during_build)
        self.assertEqual(evidence.builder_id, "modal-direct-unattested")
        self.assertEqual(evidence.security_level, "MODAL_UNATTESTED")
        self.assertEqual(evidence.image_size_measurement, "UNAVAILABLE")

    def test_modal_builder_streams_output_only_in_dev_mode(self) -> None:
        class OutputContext:
            def __enter__(self):
                FakeModal.output_context_entries += 1

            def __exit__(self, exc_type, exc, traceback):
                return False

        class FakeImage:
            object_id = "im-development"

            def build(self, app):
                return self

        class FakeModal:
            enable_output_calls = 0
            output_context_entries = 0

            class App:
                @staticmethod
                def lookup(name, **kwargs):
                    return types.SimpleNamespace(app_id="ap-build")

            class Image:
                @staticmethod
                def from_dockerfile(path, **kwargs):
                    return FakeImage()

            @classmethod
            def enable_output(cls):
                cls.enable_output_calls += 1
                return OutputContext()

        request = BuildRequest("competition", "hotkey", ROOT, TREE_SHA)
        for value, expected_calls in (("False", 0), ("true", 1)):
            with self.subTest(DEV_MODE=value):
                FakeModal.enable_output_calls = 0
                FakeModal.output_context_entries = 0
                with patch.dict(os.environ, {"DEV_MODE": value}):
                    ModalImageBuilder(
                        ModalImageBuildBackend(modal_api=FakeModal)
                    ).build(request)
                self.assertEqual(FakeModal.enable_output_calls, expected_calls)
                self.assertEqual(FakeModal.output_context_entries, expected_calls)

    def test_modal_build_timeout_stops_isolated_app_and_is_terminal(self) -> None:
        class FakeModal:
            pass

        backend = ModalImageBuildBackend(modal_api=FakeModal)
        backend._subprocess_builds = True
        request = BuildRequest(
            "compression-2026-w30",
            "miner-hotkey",
            ROOT,
            TREE_SHA,
            modal_build_timeout_seconds=420,
        )

        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch(
                "vidaio_subnet_core.competition.build.subprocess.run",
                side_effect=run,
            ),
            self.assertRaises(TrustedBuildError) as captured,
        ):
            backend.build(request)

        self.assertEqual(captured.exception.reason_code, BuildReason.BUILD_TIMEOUT)
        self.assertEqual(calls[0][1]["timeout"], 420)
        self.assertEqual(calls[1][0][1:4], ["-m", "modal", "app"])
        self.assertIn("stop", calls[1][0])
        self.assertIn("compression-2026-w30", calls[1][0][-1])

    def test_build_service_persists_timeout_as_rejection(self) -> None:
        class TimeoutBuilder:
            def build(self, request):
                raise TrustedBuildError(
                    BuildReason.BUILD_TIMEOUT,
                    "Modal image build exceeded 600 seconds",
                )

        class FakeRepository:
            rejection = None

            def get_contender(self, competition_id, hotkey):
                return types.SimpleNamespace(pinned_tree_sha=TREE_SHA)

            def record_build_rejection(self, **kwargs):
                self.rejection = kwargs

        repository = FakeRepository()
        service = CompetitionBuildService(
            repository,
            TimeoutBuilder(),
            actor="validator:test",
            clock=lambda: NOW,
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaises(TrustedBuildError),
        ):
            service.build_contender("competition", "hotkey", Path(temp))

        self.assertEqual(repository.rejection["reason_code"], "BUILD_TIMEOUT")


class FakeSandboxBackend:
    def __init__(self) -> None:
        self.created = []
        self.handles: dict[str, SandboxHandle] = {}
        self.handles_by_name: dict[tuple[str, str], SandboxHandle] = {}
        self.terminated: list[str] = []
        self.isolation = {
            "passed": True,
            "network_blocked": True,
            "direct_ip_blocked": True,
            "dns_lookup_blocked": True,
            "https_blocked": True,
            "input_read_only": True,
            "output_writable": True,
            "reference_mount_absent": True,
            "credentials_absent": True,
        }
        self.invoke_error: Exception | None = None
        self.output_commits = 0
        self.prepared_outputs: list[tuple[str, ...]] = []
        self.invoke_timeouts: list[int] = []
        self.stale_named_handle: SandboxHandle | None = None

    def create(self, spec):
        if self.stale_named_handle is not None:
            raise RuntimeError("Sandbox with this name already exists")
        self.created.append(spec)
        sandbox_id = f"sb-{len(self.created)}"
        handle = SandboxHandle(f"ap-{len(self.created)}", sandbox_id, object())
        self.handles[sandbox_id] = handle
        self.handles_by_name[(spec.app_name, spec.sandbox_name)] = handle
        return handle

    def attach(self, app_id, sandbox_id):
        if sandbox_id in self.terminated:
            raise RuntimeError("terminated")
        handle = self.handles[sandbox_id]
        return SandboxHandle(app_id, sandbox_id, handle.raw)

    def attach_by_name(self, app_name, sandbox_name):
        if self.stale_named_handle is not None:
            return self.stale_named_handle
        handle = self.handles_by_name[(app_name, sandbox_name)]
        return SandboxHandle(handle.app_id, handle.sandbox_id, handle.raw)

    def verify_isolation(self, handle):
        return dict(self.isolation)

    def inspect_resources(self, handle):
        del handle
        return SandboxResourceAllocation(
            gpu_type="H100",
            gpu_count=1,
            cpu_cores=12.0,
            report={
                "gpu_names": ["NVIDIA H100 80GB HBM3"],
                "allocated_cpu_cores": 12.0,
            },
        )

    def health(self, handle, timeout_seconds):
        return {"status": "ok"}

    def prepare_outputs(self, handle, output_paths):
        del handle
        self.prepared_outputs.append(tuple(output_paths))

    def invoke(self, handle, payload_json, timeout_seconds):
        if not self.prepared_outputs:
            raise RuntimeError("output directories were not prepared")
        self.invoke_timeouts.append(timeout_seconds)
        if self.invoke_error:
            raise self.invoke_error
        request = json.loads(payload_json)
        return json.dumps(
            {
                "results": [
                    {"output_path": item["output_path"]} for item in request["items"]
                ],
            }
        )

    def commit_outputs(self, handle):
        del handle
        self.output_commits += 1

    def terminate(self, handle):
        self.terminated.append(handle.sandbox_id)
        if handle is self.stale_named_handle:
            self.stale_named_handle = None


class SandboxLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.database_url = f"sqlite:///{root / 'competition.db'}"
        self.repository = CompetitionRepository(self.database_url)
        self.manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        self.hotkey = "5validator-controlled-hotkey"
        self.repository.insert_manifest(self.manifest, now=NOW, actor="test")
        validation = ValidationReport(
            ValidationStatus.ACCEPTED,
            ValidationReason.ACCEPTED,
            "a" * 64,
            4,
            100,
            (),
        )
        self.repository.record_pinned_contender(
            competition_id=self.manifest.competition_id,
            hotkey=self.hotkey,
            repository_url_hash="c" * 64,
            repository_display="github.com/acme/solution",
            pinned_commit_sha="a" * 40,
            pinned_tree_sha=TREE_SHA,
            latest_commit_time=NOW.isoformat(),
            validation=validation,
            now=NOW,
            actor="test",
        )
        self.repository.record_build_evidence(
            competition_id=self.manifest.competition_id,
            hotkey=self.hotkey,
            image_id=IMAGE_ID,
            image_digest=IMAGE_DIGEST,
            image_size_bytes=IMAGE_SIZE_LIMIT_BYTES,
            evidence=build_evidence().as_dict(),
            now=NOW,
            actor="test",
        )
        self.backend = FakeSandboxBackend()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runner(self, repository=None):
        return CompetitionModalRunner(
            repository or self.repository,
            self.backend,
            actor="validator:test",
            clock=lambda: NOW,
        )

    def test_output_volume_name_identifies_competition_and_miner(self) -> None:
        name = _output_volume_name(
            "vidaio-competition-compression-2026-w29-output",
            "compression-2026-w30",
            "5FRUjvY1AxpK6YxYxTLHjHcEwwYfb4Ah29LFQ7vGbLqFmJLo",
        )
        self.assertEqual(
            name,
            "vidaio-compression-2026-w30-5FRUjvY1AxpK-bLqFmJLo-bad385",
        )
        self.assertLessEqual(len(name), 63)

    def test_validator_controls_resources_mounts_and_warm_reuse(self) -> None:
        runner = self.runner()
        first = runner.ensure_warm(self.manifest, self.hotkey, now=NOW)
        second = runner.ensure_warm(
            self.manifest, self.hotkey, now=NOW + timedelta(minutes=10)
        )
        self.assertEqual(first.handle.sandbox_id, second.handle.sandbox_id)
        self.assertEqual(len(self.backend.created), 1)
        spec = self.backend.created[0]
        self.assertEqual(spec.image_id, IMAGE_ID)
        self.assertEqual(spec.gpu_type, self.manifest.allowed_gpus[0])
        self.assertEqual(spec.cpu_request, 16)
        self.assertEqual(spec.cpu_limit, 16)
        self.assertEqual(first.record.allocated_gpu_type, "H100")
        self.assertEqual(first.record.allocated_gpu_count, 1)
        self.assertEqual(first.record.allocated_cpu_cores, 12.0)
        self.assertEqual(
            json.loads(first.record.isolation_report_json)["resource_allocation"][
                "gpu_names"
            ],
            ["NVIDIA H100 80GB HBM3"],
        )
        self.assertEqual(
            spec.input_volume_name, self.manifest.evaluation_input_volume_name
        )
        self.assertIsNone(self.manifest.evaluation_reference_volume_name)
        self.assertNotEqual(
            spec.output_volume_name, self.manifest.evaluation_input_volume_name
        )
        self.assertIn(self.manifest.competition_id, spec.output_volume_name)
        self.assertIn(self.hotkey, spec.output_volume_name)
        self.assertLessEqual(len(spec.output_volume_name), 63)
        self.assertEqual(
            spec.lifetime_seconds, int(MAX_SANDBOX_LIFETIME.total_seconds())
        )

    def test_process_recreation_reattaches_without_rebuild_or_new_sandbox(self) -> None:
        first = self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)
        recreated_repository = CompetitionRepository(self.database_url)
        second = self.runner(recreated_repository).ensure_warm(
            self.manifest, self.hotkey, now=NOW + timedelta(hours=1)
        )
        self.assertEqual(first.handle.sandbox_id, second.handle.sandbox_id)
        self.assertEqual(len(self.backend.created), 1)

    def test_stale_named_sandbox_is_terminated_and_recreated(self) -> None:
        self.backend.stale_named_handle = SandboxHandle(
            "ap-stale", "sb-stale", object()
        )

        session = self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)

        self.assertIn("sb-stale", self.backend.terminated)
        self.assertEqual(len(self.backend.created), 1)
        self.assertNotEqual(session.handle.sandbox_id, "sb-stale")

    def test_competitions_with_same_name_prefix_get_distinct_sandbox_names(self) -> None:
        second_manifest = self.manifest.model_copy(
            update={"competition_id": f"{self.manifest.competition_id}-rerun"}
        )
        self.repository.insert_manifest(second_manifest, now=NOW, actor="test")
        validation = ValidationReport(
            ValidationStatus.ACCEPTED,
            ValidationReason.ACCEPTED,
            "a" * 64,
            4,
            100,
            (),
        )
        self.repository.record_pinned_contender(
            competition_id=second_manifest.competition_id,
            hotkey=self.hotkey,
            repository_url_hash="c" * 64,
            repository_display="github.com/acme/solution",
            pinned_commit_sha="a" * 40,
            pinned_tree_sha=TREE_SHA,
            latest_commit_time=NOW.isoformat(),
            validation=validation,
            now=NOW,
            actor="test",
        )
        self.repository.record_build_evidence(
            competition_id=second_manifest.competition_id,
            hotkey=self.hotkey,
            image_id=IMAGE_ID,
            image_digest=IMAGE_DIGEST,
            image_size_bytes=IMAGE_SIZE_LIMIT_BYTES,
            evidence=build_evidence().as_dict(),
            now=NOW,
            actor="test",
        )

        self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)
        self.runner().ensure_warm(second_manifest, self.hotkey, now=NOW)

        first, second = self.backend.created
        self.assertNotEqual(first.app_name, second.app_name)
        self.assertNotEqual(first.sandbox_name, second.sandbox_name)

    def test_restart_can_find_sandbox_by_name_if_crash_precedes_id_commit(self) -> None:
        with patch.object(
            self.repository, "bind_sandbox_identity", side_effect=SystemExit("crash")
        ):
            with self.assertRaises(SystemExit):
                self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)
        stranded = self.repository.latest_sandbox(
            self.manifest.competition_id, self.hotkey
        )
        self.assertEqual(stranded.status, "STARTING")
        self.assertIsNone(stranded.modal_sandbox_id)
        self.assertTrue(stranded.modal_sandbox_name)

        recreated = CompetitionRepository(self.database_url)
        recovered = self.runner(recreated).ensure_warm(
            self.manifest, self.hotkey, now=NOW + timedelta(minutes=1)
        )
        self.assertEqual(recovered.handle.sandbox_id, "sb-1")
        self.assertEqual(len(self.backend.created), 1)

    def test_rollover_terminates_old_sandbox_and_reuses_image_and_output(self) -> None:
        first = self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)
        second = self.runner().ensure_warm(
            self.manifest,
            self.hotkey,
            now=NOW + MAX_SANDBOX_LIFETIME - timedelta(minutes=4),
        )
        self.assertNotEqual(first.handle.sandbox_id, second.handle.sandbox_id)
        self.assertIn(first.handle.sandbox_id, self.backend.terminated)
        self.assertEqual(
            self.backend.created[0].image_id, self.backend.created[1].image_id
        )
        self.assertEqual(
            self.backend.created[0].output_volume_name,
            self.backend.created[1].output_volume_name,
        )
        self.assertEqual(second.record.generation, 2)

    def test_termination_preserves_contender_output_volume_mapping(self) -> None:
        session = self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)
        output_volume = session.record.output_volume_name

        self.runner().terminate(
            self.manifest, self.hotkey, now=NOW + timedelta(minutes=1)
        )

        self.assertIn(session.handle.sandbox_id, self.backend.terminated)
        self.assertEqual(
            self.repository.latest_sandbox(
                self.manifest.competition_id, self.hotkey
            ).status,
            "TERMINATED",
        )
        self.assertEqual(
            self.repository.get_contender(
                self.manifest.competition_id, self.hotkey
            ).output_volume_name,
            output_volume,
        )

    def test_failed_active_isolation_probe_terminates_and_never_activates(self) -> None:
        self.backend.isolation["passed"] = False
        self.backend.isolation["https_blocked"] = False
        with self.assertRaises(SandboxRunnerError) as captured:
            self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)
        self.assertEqual(captured.exception.reason_code, "ISOLATION_PROBE_FAILED")
        self.assertEqual(self.backend.terminated, ["sb-1"])
        self.assertEqual(
            self.repository.latest_sandbox(
                self.manifest.competition_id, self.hotkey
            ).status,
            "FAILED",
        )

    def test_raw_startup_failure_persists_redacted_diagnostic_detail(self) -> None:
        with patch.object(
            self.backend, "create", side_effect=RuntimeError("GPU capacity unavailable")
        ):
            with self.assertRaisesRegex(
                SandboxRunnerError, "SANDBOX_START_FAILED.*GPU capacity unavailable"
            ):
                self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)
        closed = [
            event
            for event in self.repository.list_events(self.manifest.competition_id)
            if event.event_type == "SANDBOX_CLOSED"
        ][-1]
        payload = json.loads(closed.payload_json)
        self.assertEqual(payload["reason_code"], "SANDBOX_START_FAILED")
        self.assertEqual(payload["detail"], "GPU capacity unavailable")

    def test_batch_failure_forces_termination(self) -> None:
        request = CompetitionCompressionRequest(
            competition_id=self.manifest.competition_id,
            hotkey=self.hotkey,
            batch_id="batch-1",
            items=(
                CompetitionCompressionItem(
                    evaluation_id="input-1",
                    input_path="/evaluation-inputs/input-1.mp4",
                    output_path="/output/input-1.mp4",
                    codec="AV1",
                    vmaf_threshold=90.0,
                ),
            ),
        )
        response = self.runner().invoke_batch(
            self.manifest,
            request,
            timeout_seconds=321,
            now=NOW,
        )
        self.assertEqual(response.results[0].output_path, "/output/input-1.mp4")
        self.assertEqual(self.backend.prepared_outputs, [("/output/input-1.mp4",)])
        self.assertEqual(self.backend.invoke_timeouts, [321])
        self.assertEqual(self.backend.output_commits, 1)
        self.assertEqual(
            self.repository.latest_sandbox(
                self.manifest.competition_id, self.hotkey
            ).batch_timeout_seconds,
            321,
        )

        self.backend.invoke_error = TimeoutError("forced timeout")
        with self.assertRaisesRegex(SandboxRunnerError, "BATCH_EXECUTION_FAILED"):
            self.runner().invoke_batch(
                self.manifest,
                request.model_copy(update={"batch_id": "batch-2"}),
                now=NOW,
            )
        self.assertIn("sb-1", self.backend.terminated)

    def test_contenders_get_distinct_output_volumes(self) -> None:
        second_hotkey = "different-hotkey"
        validation = ValidationReport(
            ValidationStatus.ACCEPTED,
            ValidationReason.ACCEPTED,
            "e" * 64,
            4,
            100,
            (),
        )
        self.repository.record_pinned_contender(
            competition_id=self.manifest.competition_id,
            hotkey=second_hotkey,
            repository_url_hash="f" * 64,
            repository_display="github.com/acme/other",
            pinned_commit_sha="1" * 40,
            pinned_tree_sha="2" * 40,
            latest_commit_time=NOW.isoformat(),
            validation=validation,
            now=NOW,
            actor="test",
        )
        self.repository.record_build_evidence(
            competition_id=self.manifest.competition_id,
            hotkey=second_hotkey,
            image_id="im-other",
            image_digest="sha256:" + "3" * 64,
            image_size_bytes=1,
            evidence={"builder_id": "trusted-quota-builder"},
            now=NOW,
            actor="test",
        )
        self.runner().ensure_warm(self.manifest, self.hotkey, now=NOW)
        self.runner().ensure_warm(self.manifest, second_hotkey, now=NOW)
        self.assertNotEqual(
            self.backend.created[0].output_volume_name,
            self.backend.created[1].output_volume_name,
        )


class LiveExecutionCoordinatorTests(unittest.TestCase):
    def test_accepted_submission_builds_and_starts_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            database_url = f"sqlite:///{root / 'competition.db'}"
            artifact_root = root / "artifacts"
            repository = CompetitionRepository(database_url)
            manifest = load_manifest(
                ROOT / "competitions/manifests/examples/compression-competition.json"
            )
            repository.insert_manifest(manifest, now=NOW, actor="test")
            repository.transition(
                manifest.competition_id,
                CompetitionState.ENROLLING,
                expected=CompetitionState.SCHEDULED,
                now=NOW,
                actor="test",
            )
            repository.transition(
                manifest.competition_id,
                CompetitionState.FINALIZING_SUBMISSIONS,
                expected=CompetitionState.ENROLLING,
                now=NOW,
                actor="test",
            )
            validation = ValidationReport(
                ValidationStatus.ACCEPTED,
                ValidationReason.ACCEPTED,
                "a" * 64,
                4,
                100,
                (),
            )
            hotkey = "accepted-contender"
            repository.record_pinned_contender(
                competition_id=manifest.competition_id,
                hotkey=hotkey,
                repository_url_hash="c" * 64,
                repository_display="github.com/acme/solution",
                pinned_commit_sha="a" * 40,
                pinned_tree_sha=TREE_SHA,
                latest_commit_time=NOW.isoformat(),
                validation=validation,
                now=NOW,
                actor="test",
            )
            source = pinned_repository_source(
                artifact_root, manifest.competition_id, hotkey
            )
            source.mkdir(parents=True)

            manager = CompetitionManager(
                CompetitionConfig(
                    mode_enabled=True,
                    database_url=database_url,
                    artifact_root=artifact_root,
                    artifact_backup_bucket="private-test-bucket",
                    owner_id="validator:test",
                ),
                repository,
                clock=lambda: NOW,
            )
            build_service = CompetitionBuildService(
                repository,
                TrustedImageBuilder(
                    FakeBuilder(build_evidence()),
                    trusted_builder_ids=frozenset({"trusted-quota-builder"}),
                ),
                actor="validator:test",
                clock=lambda: NOW,
            )
            sandbox_backend = FakeSandboxBackend()
            runner = CompetitionModalRunner(
                repository,
                sandbox_backend,
                actor="validator:test",
                clock=lambda: NOW,
            )
            coordinator = CompetitionExecutionCoordinator(
                manager,
                repository,
                build_service,
                runner,
                artifact_root=artifact_root,
                actor="validator:test",
                accepted_build_statuses=frozenset({"ACCEPTED"}),
                clock=lambda: NOW,
            )

            asyncio.run(coordinator.run_once())
            self.assertEqual(
                repository.get(manifest.competition_id).status,
                CompetitionState.FINALIZING_SUBMISSIONS.value,
            )
            self.assertEqual(len(sandbox_backend.created), 0)

            repository.record_submission_backup_success(
                manifest.competition_id,
                bucket="private-test-bucket",
                prefix=(
                    f"competition_artifacts/{manifest.competition_id}/"
                    "submission-snapshots/test"
                ),
                archive_key=f"competition_artifacts-{manifest.competition_id}.tar.gz",
                inventory_key="inventory.json",
                checksum="f" * 64,
                size_bytes=100,
                contender_count=1,
                now=NOW,
                actor="validator:test",
            )
            asyncio.run(coordinator.run_once())

            competition = repository.get(manifest.competition_id)
            contender = repository.get_contender(manifest.competition_id, hotkey)
            sandbox = repository.latest_sandbox(manifest.competition_id, hotkey)
            self.assertEqual(competition.status, CompetitionState.EVALUATING.value)
            self.assertEqual(contender.build_status, "ACCEPTED")
            self.assertEqual(contender.image_id, IMAGE_ID)
            self.assertEqual(sandbox.status, "RUNNING")
            self.assertEqual(len(sandbox_backend.created), 1)

    def test_builds_and_sandbox_startup_share_the_manifest_parallel_limit(
        self,
    ) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        ).model_copy(
            update={
                "max_parallel_contenders": 2,
                "modal_build_timeout": timedelta(minutes=7),
            }
        )
        hotkeys = [f"contender-{index}" for index in range(4)]

        class ParallelRepository:
            sandbox_mode = False

            def list_contenders(self, competition_id):
                del competition_id
                return [
                    types.SimpleNamespace(
                        hotkey=hotkey,
                        validation_status="ACCEPTED",
                        build_status="ACCEPTED" if self.sandbox_mode else None,
                    )
                    for hotkey in hotkeys
                ]

            def mark_build_started(self, **kwargs):
                del kwargs

            @staticmethod
            def contender_evaluation_is_complete(_competition_id, _hotkey):
                return False

        class ParallelProbe:
            def __init__(self):
                self.active = 0
                self.peak = 0
                self.lock = threading.Lock()

            def pause_while_counted(self):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.05)
                with self.lock:
                    self.active -= 1

        build_probe = ParallelProbe()
        sandbox_probe = ParallelProbe()

        class ParallelBuildService:
            timeouts = []

            def build_contender(
                self, competition_id, hotkey, source, modal_build_timeout_seconds
            ):
                del competition_id, source
                self.timeouts.append(modal_build_timeout_seconds)
                build_probe.pause_while_counted()
                return types.SimpleNamespace(
                    image_id=f"im-{hotkey}", builder_id="modal-direct-unattested"
                )

        class ParallelSandboxRunner:
            def ensure_warm(self, competition_manifest, hotkey):
                del competition_manifest
                sandbox_probe.pause_while_counted()
                return types.SimpleNamespace(
                    handle=types.SimpleNamespace(sandbox_id=f"sb-{hotkey}"),
                    record=types.SimpleNamespace(generation=1),
                )

        repository = ParallelRepository()
        coordinator = CompetitionExecutionCoordinator(
            manager=None,
            repository=repository,
            build_service=ParallelBuildService(),
            sandbox_runner=ParallelSandboxRunner(),
            artifact_root=ROOT,
            actor="validator:test",
            accepted_build_statuses=frozenset({"ACCEPTED"}),
            clock=lambda: NOW,
        )

        asyncio.run(coordinator._build_accepted_contenders(manifest))
        repository.sandbox_mode = True
        self.assertTrue(asyncio.run(coordinator._ensure_sandboxes(manifest)))

        self.assertEqual(build_probe.peak, 2)
        self.assertEqual(ParallelBuildService.timeouts, [420.0] * len(hotkeys))
        self.assertEqual(sandbox_probe.peak, 2)

    def test_evaluation_does_not_claim_work_when_sandbox_startup_fails(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )

        class Repository:
            @staticmethod
            def list_contenders(_competition_id):
                return [
                    types.SimpleNamespace(hotkey="contender-a", build_status="ACCEPTED")
                ]

            @staticmethod
            def contender_evaluation_is_complete(_competition_id, _hotkey):
                return False

        class FailingRunner:
            @staticmethod
            def ensure_warm(_manifest, _hotkey):
                raise SandboxRunnerError(
                    "SANDBOX_START_FAILED", "GPU capacity unavailable"
                )

        class Coordinator(CompetitionExecutionCoordinator):
            evaluation_called = False

            async def _evaluate_contenders(self, _manifest):
                self.evaluation_called = True

        coordinator = Coordinator(
            manager=None,
            repository=Repository(),
            build_service=None,
            sandbox_runner=FailingRunner(),
            artifact_root=ROOT,
            actor="validator:test",
            accepted_build_statuses=frozenset({"ACCEPTED"}),
            clock=lambda: NOW,
        )
        asyncio.run(coordinator._advance(manifest, CompetitionState.EVALUATING))
        self.assertFalse(coordinator.evaluation_called)


class FakeMount:
    def __init__(self, name):
        self.name = name
        self.read_only = None

    def with_mount_options(self, *, read_only):
        self.read_only = read_only
        return self


class FakeModalApi:
    class App:
        @staticmethod
        def lookup(name, **kwargs):
            return types.SimpleNamespace(app_id="ap-1")

    class Image:
        @staticmethod
        def from_id(image_id):
            return ("image", image_id)

    class Volume:
        mounts = {}

        @classmethod
        def from_name(cls, name, **kwargs):
            mount = FakeMount(name)
            cls.mounts[name] = (mount, kwargs)
            return mount

    class Sandbox:
        kwargs = None

        @classmethod
        def create(cls, *args, **kwargs):
            cls.kwargs = kwargs
            return types.SimpleNamespace(object_id="sb-1")


class ModalAdapterPolicyTests(unittest.TestCase):
    def test_actual_modal_adapter_prices_observed_upgraded_gpu_and_cpu(self) -> None:
        class Stream:
            @staticmethod
            def read():
                return json.dumps(
                    {
                        "gpu_names": ["NVIDIA H200"],
                        "allocated_cpu_cores": 12.0,
                        "cpu_quota_cores": 12.0,
                        "affinity_logical_cpu_count": 24,
                        "affinity_physical_cpu_cores": 12.0,
                    }
                )

        class Process:
            returncode = 0
            stdout = Stream()
            stderr = Stream()

            @staticmethod
            def wait():
                return None

        class RawSandbox:
            @staticmethod
            def exec(*_args, **_kwargs):
                return Process()

        backend = ModalSandboxBackend(environment_name="dev", modal_api=FakeModalApi)
        allocation = backend.inspect_resources(
            SandboxHandle("ap-1", "sb-1", RawSandbox())
        )
        self.assertEqual(allocation.gpu_type, "H200")
        self.assertEqual(allocation.gpu_count, 1)
        self.assertEqual(allocation.cpu_cores, 12.0)

    def test_actual_modal_adapter_passes_no_credentials_ports_or_reference_mount(
        self,
    ) -> None:
        backend = ModalSandboxBackend(environment_name="dev", modal_api=FakeModalApi)
        spec = types.SimpleNamespace(
            competition_id="competition",
            hotkey="hotkey",
            generation=1,
            app_name="app",
            sandbox_name="sandbox",
            image_id=IMAGE_ID,
            input_volume_name="inputs",
            output_volume_name="outputs-hotkey",
            gpu_type="L4",
            cpu_request=16.0,
            cpu_limit=16.0,
            lifetime_seconds=3600,
        )
        backend.create(spec)
        kwargs = FakeModalApi.Sandbox.kwargs
        self.assertTrue(kwargs["block_network"])
        self.assertEqual(kwargs["secrets"], [])
        self.assertFalse(kwargs["include_oidc_identity_token"])
        self.assertEqual(kwargs["encrypted_ports"], [])
        self.assertEqual(kwargs["h2_ports"], [])
        self.assertEqual(kwargs["unencrypted_ports"], [])
        self.assertEqual(set(kwargs["volumes"]), {"/evaluation-inputs", "/output"})
        self.assertTrue(kwargs["volumes"]["/evaluation-inputs"].read_only)
        self.assertFalse(kwargs["volumes"]["/output"].read_only)
        self.assertEqual(kwargs["cpu"], (16.0, 16.0))
        self.assertEqual(kwargs["gpu"], "L4")

    def test_v1_output_volume_fails_with_actionable_reason(self) -> None:
        class V1OutputModalApi(FakeModalApi):
            class Volume:
                @staticmethod
                def from_name(name, **kwargs):
                    if kwargs.get("version") == 2:
                        raise RuntimeError(
                            f"Volume '{name}' exists but has version v1, not v2 as "
                            "requested."
                        )
                    return FakeMount(name)

        backend = ModalSandboxBackend(
            environment_name="main", modal_api=V1OutputModalApi
        )
        spec = types.SimpleNamespace(
            competition_id="competition",
            hotkey="hotkey",
            generation=1,
            app_name="app",
            sandbox_name="sandbox",
            image_id=IMAGE_ID,
            input_volume_name="inputs",
            output_volume_name="outputs-hotkey",
            gpu_type="L4",
            cpu_request=16.0,
            cpu_limit=16.0,
            lifetime_seconds=3600,
        )
        with self.assertRaisesRegex(
            SandboxRunnerError,
            "OUTPUT_VOLUME_VERSION_MISMATCH.*modal volume create --version=2",
        ) as captured:
            backend.create(spec)
        self.assertEqual(
            captured.exception.reason_code, "OUTPUT_VOLUME_VERSION_MISMATCH"
        )

    def test_actual_modal_adapter_prepares_only_output_volume_parents(self) -> None:
        class Stream:
            @staticmethod
            def read():
                return ""

        class Process:
            returncode = 0
            stdout = Stream()
            stderr = Stream()

            @staticmethod
            def wait():
                return None

        class RawSandbox:
            calls = []

            @classmethod
            def exec(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return Process()

        backend = ModalSandboxBackend(environment_name="dev", modal_api=FakeModalApi)
        handle = SandboxHandle("ap-1", "sb-1", RawSandbox())
        backend.prepare_outputs(
            handle,
            (
                "/output/evaluations/batch-1/input-1.mp4",
                "/output/evaluations/batch-1/input-2.mp4",
            ),
        )
        self.assertEqual(
            RawSandbox.calls,
            [
                (
                    (
                        "mkdir",
                        "-p",
                        "--",
                        "/output/evaluations/batch-1",
                    ),
                    {"timeout": 30},
                )
            ],
        )
        with self.assertRaisesRegex(SandboxRunnerError, "OUTPUT_PATH_INVALID"):
            backend.prepare_outputs(handle, ("/output/../etc/escape.mp4",))
        self.assertEqual(len(RawSandbox.calls), 1)

    def test_actual_modal_adapter_flushes_payload_and_eof_before_waiting(self) -> None:
        payload = '{"competition_id":"competition"}'

        class Writer:
            def __init__(self) -> None:
                self.calls = []
                self.eof_queued = False

            def write(self, data: bytes) -> None:
                self.calls.append(("write", data))

            def write_eof(self) -> None:
                self.eof_queued = True
                self.calls.append(("write_eof", None))

            def drain(self) -> None:
                if not self.eof_queued:
                    raise AssertionError("Modal stdin EOF must be queued before drain")
                self.calls.append(("drain", None))

        class Stream:
            @staticmethod
            def read():
                return '{"status":"ok"}'

        class Process:
            returncode = 0
            stdout = Stream()
            stderr = Stream()

            def __init__(self) -> None:
                self.stdin = Writer()

            @staticmethod
            def wait():
                return None

        process = Process()

        class RawSandbox:
            @staticmethod
            def exec(*args, **kwargs):
                return process

        backend = ModalSandboxBackend(environment_name="dev", modal_api=FakeModalApi)
        handle = SandboxHandle("ap-1", "sb-1", RawSandbox())
        rendered = backend.invoke(handle, payload, timeout_seconds=60)

        self.assertEqual(rendered, '{"status":"ok"}')
        self.assertEqual(
            process.stdin.calls,
            [
                ("write", payload.encode("utf-8")),
                ("write_eof", None),
                ("drain", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
