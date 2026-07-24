from __future__ import annotations

# ruff: noqa: E402 -- package stubs keep focused tests independent of Bittensor.

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError


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

from vidaio_subnet_core.competition.config import BossConfig, load_manifest
from vidaio_subnet_core.competition.contracts import (
    CompetitionCompressionItem,
    CompetitionCompressionRequest,
    CompetitionCompressionResponse,
    CompetitionCompressionResult,
)
from vidaio_subnet_core.competition.intake import (
    CompetitionSubmissionIntakeService,
    IntakeError,
    RepositoryIntake,
    RepositorySubmission,
    SecureGitTransport,
    pinned_repository_source,
    validate_polled_submission,
)
from vidaio_subnet_core.competition.qualification import MediaInfo, WarmupQualifier
from vidaio_subnet_core.competition.repository import CompetitionRepository
from vidaio_subnet_core.competition.state import CompetitionState
from vidaio_subnet_core.competition.validation import (
    RepositoryStaticValidator,
    ValidationReason,
    ValidationStatus,
    _TRUSTED_SDK_DIGESTS,
)

PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "competition_preflight_script", ROOT / "scripts" / "competition_preflight.py"
)
competition_preflight = importlib.util.module_from_spec(PREFLIGHT_SPEC)
assert PREFLIGHT_SPEC.loader is not None
PREFLIGHT_SPEC.loader.exec_module(competition_preflight)


def write_template(root: Path) -> None:
    files = {
        "miner/modal_workers.py": "VALUE = 1\n",
        "miner/compression/app.py": "def compress():\n    return 'ok'\n",
        "miner/common_preflight.py": "def main():\n    return 0\n",
        "requirements.txt": "fastapi==0.110.3\n",
        "competition_solution.json": json.dumps(
            {
                "schema_version": 2,
                "competition_type": "COMPRESSION",
                "routes": ["/health", "/compress"],
                "local_path_io": True,
                "preflight": "miner/common_preflight.py",
                "sdk": "miner/competition_sdk.py",
                "entrypoint": "miner/modal_workers.py",
            }
        ),
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for name in ("common_preflight.py", "competition_sdk.py"):
        destination = root / "miner" / name
        destination.write_text(
            (ROOT / "miner" / name).read_text(encoding="utf-8"), encoding="utf-8"
        )


def write_boss_export(repository_root: Path, relative_path: Path) -> Path:
    boss = repository_root / relative_path
    boss.mkdir(parents=True)
    write_template(boss)
    extra_files = {
        ".vidaio-sdk-export": "validated export\n",
        "scripts/competition_modal_build.py": "def main():\n    return 0\n",
    }
    for name, content in extra_files.items():
        path = boss / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    fixture_relative = Path("competitions/fixtures/compression_warmup_input.mp4")
    (boss / fixture_relative).parent.mkdir(parents=True, exist_ok=True)
    (boss / fixture_relative).write_bytes(b"boss warmup fixture")
    (repository_root / fixture_relative).parent.mkdir(parents=True, exist_ok=True)
    (repository_root / fixture_relative).write_bytes(b"validator warmup fixture")
    return boss


def request_item(**overrides) -> CompetitionCompressionItem:
    values = {
        "evaluation_id": "item-1",
        "input_path": "/evaluation-inputs/item-1.mp4",
        "output_path": "/output/hotkey/item-1.mp4",
        "codec": "AV1",
        "vmaf_threshold": 90.0,
    }
    values.update(overrides)
    return CompetitionCompressionItem(**values)


class CompetitionPreflightTests(unittest.TestCase):
    def manifest_with_boss(self, repository_path: Path):
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        return manifest.model_copy(
            update={
                "boss": BossConfig(
                    repository_path=repository_path,
                    boss_hotkey="boss-hotkey",
                )
            }
        )

    def test_validate_boss_flag_is_parsed(self) -> None:
        args = competition_preflight.parse_args(
            ["competition.json", "--validate-boss"]
        )
        self.assertTrue(args.validate_boss)

    def test_validate_boss_requires_manifest_configuration(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(
            ValueError,
            "requires boss.repository_path and boss.boss_hotkey",
        ):
            competition_preflight.validate_manifest_boss(manifest, Path(temp))

    def test_validate_boss_accepts_complete_static_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository_root = Path(temp)
            relative_path = Path("competition_boss/current")
            boss = write_boss_export(repository_root, relative_path)
            result = competition_preflight.validate_manifest_boss(
                self.manifest_with_boss(relative_path),
                repository_root,
            )

        self.assertEqual(result["status"], ValidationStatus.ACCEPTED.value)
        self.assertEqual(result["boss_hotkey"], "boss-hotkey")
        self.assertEqual(result["repository_path"], relative_path.as_posix())
        self.assertEqual(result["resolved_repository_path"], str(boss.resolve()))
        self.assertEqual(
            result["repository"]["reason_code"],
            ValidationReason.ACCEPTED.value,
        )

    def test_validate_boss_rejects_objectively_ineligible_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository_root = Path(temp)
            relative_path = Path("competition_boss/current")
            boss = write_boss_export(repository_root, relative_path)
            unsafe = boss / "miner/compression/unsafe.py"
            unsafe.write_text("eval('unsafe')\n", encoding="utf-8")

            result = competition_preflight.validate_manifest_boss(
                self.manifest_with_boss(relative_path),
                repository_root,
            )

        self.assertEqual(result["status"], ValidationStatus.REJECTED.value)
        self.assertEqual(
            result["repository"]["reason_code"],
            ValidationReason.DYNAMIC_EXECUTION.value,
        )


class ContractTests(unittest.TestCase):
    def test_result_contains_only_the_output_path(self) -> None:
        result = CompetitionCompressionResult(output_path="/output/hotkey/item-1.mp4")
        self.assertEqual(result.output_path, "/output/hotkey/item-1.mp4")
        self.assertEqual(result.model_dump(), {"output_path": result.output_path})
        with self.assertRaises(ValidationError):
            CompetitionCompressionResult(output_path=None, success=False)

        response = CompetitionCompressionResponse(results=(result,))
        self.assertEqual(
            response.model_dump(), {"results": ({"output_path": result.output_path},)}
        )
        with self.assertRaises(ValidationError):
            CompetitionCompressionResponse(results=(result,), hotkey="another-miner")

    def test_local_contract_accepts_batches_and_rejects_unsafe_paths(self) -> None:
        request = CompetitionCompressionRequest(
            competition_id="compression-1",
            hotkey="hotkey",
            batch_id="batch-1",
            items=(request_item(),),
        )
        self.assertEqual(request.items[0].vmaf_threshold, 90.0)
        for field, value in (
            ("input_path", "https://example.com/input.mp4"),
            ("input_path", "/evaluation-inputs/../secret.mp4"),
            ("output_path", "/tmp/output.mp4"),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                request_item(**{field: value})

    def test_duplicate_ids_and_outputs_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CompetitionCompressionRequest(
                competition_id="compression-1",
                hotkey="hotkey",
                batch_id="batch-1",
                items=(request_item(), request_item()),
            )

    def test_vbr_items_require_an_approved_target_bitrate(self) -> None:
        item = request_item(codec_mode="VBR", target_bitrate=8_000_000)
        self.assertEqual(item.target_bitrate, 8_000_000)
        for overrides in (
            {"codec_mode": "VBR"},
            {"codec_mode": "VBR", "target_bitrate": 7_000_000},
            {"codec_mode": "CRF", "target_bitrate": 5_000_000},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                request_item(**overrides)


class StaticValidationTests(unittest.TestCase):
    def test_pre_pricing_sdk_release_remains_trusted(self) -> None:
        trusted = _TRUSTED_SDK_DIGESTS["miner/competition_sdk.py"]
        self.assertIn(
            "249c0579a661f4f701271976f4fd04d9f3c85e88580831ff5ff1afc76b84aafa",
            trusted,
        )
        self.assertIn(
            "be79bb2880a213cb2044d207f9912e12ea6dc8781a66229bd693ad107f804c5e",
            trusted,
        )

    def test_readable_template_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_template(root)
            report = RepositoryStaticValidator().validate(root)
            self.assertEqual(report.status, ValidationStatus.ACCEPTED)
            self.assertEqual(report.reason_code, ValidationReason.ACCEPTED)

    def test_local_sdk_exports_are_excluded_from_repository_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_template(root)
            generated = root / "miner/.competition-sdk/generated/miner"
            generated.mkdir(parents=True)
            (generated / "competition_sdk.py").write_text(
                "exec('generated local export')\n", encoding="utf-8"
            )
            report = RepositoryStaticValidator().validate(root)
            self.assertEqual(report.status, ValidationStatus.ACCEPTED)
            self.assertFalse(
                any(".competition-sdk" in finding.path for finding in report.findings)
            )

    def test_rejections_have_stable_reason_codes(self) -> None:
        cases = {
            ValidationReason.DYNAMIC_EXECUTION: (
                "miner/compression/unsafe.py",
                "eval('1')\n",
            ),
            ValidationReason.GIT_LFS_POINTER: (
                "weights.bin",
                "version https://git-lfs.github.com/spec/v1\n",
            ),
            ValidationReason.COMMITTED_SECRET: ("secret.txt", "github_pat_" + "A" * 40),
        }
        for expected, (name, content) in cases.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_template(root)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                report = RepositoryStaticValidator().validate(root)
                self.assertEqual(report.status, ValidationStatus.REJECTED)
                self.assertIn(
                    expected, {finding.reason_code for finding in report.findings}
                )

    def test_oversize_obfuscation_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_template(root)
            (root / "miner" / "large.bin").write_bytes(b"x" * 33)
            report = RepositoryStaticValidator(max_file_bytes=32).validate(root)
            self.assertEqual(report.reason_code, ValidationReason.FILE_TOO_LARGE)

        with (
            tempfile.TemporaryDirectory() as temp,
            tempfile.TemporaryDirectory() as outside,
        ):
            root = Path(temp)
            write_template(root)
            try:
                os.symlink(Path(outside), root / "miner" / "escape")
            except OSError:
                self.skipTest("symlinks unavailable")
            report = RepositoryStaticValidator().validate(root)
            self.assertIn(
                ValidationReason.PATH_ESCAPE,
                {finding.reason_code for finding in report.findings},
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_template(root)
            (root / "miner" / "compression" / "app.py").write_text(
                "x='" + "a" * 1100 + "'\n", encoding="utf-8"
            )
            report = RepositoryStaticValidator().validate(root)
            self.assertEqual(report.status, ValidationStatus.REJECTED)
            self.assertEqual(report.reason_code, ValidationReason.OBFUSCATION_REVIEW)

    def test_encoded_loader_and_minified_statements_are_rejected(self) -> None:
        cases = (
            "import base64\nvalue = base64.b64decode('" + "A" * 128 + "')\n",
            "import marshal\nvalue = marshal.loads(b'" + "A" * 128 + "')\n",
            ";".join(f"value_{index} = {index}" for index in range(13)) + "\n",
        )
        for source in cases:
            with self.subTest(source=source[:32]), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_template(root)
                (root / "miner" / "compression" / "app.py").write_text(
                    source, encoding="utf-8"
                )
                report = RepositoryStaticValidator().validate(root)
                self.assertEqual(report.status, ValidationStatus.REJECTED)
                self.assertEqual(
                    report.reason_code, ValidationReason.OBFUSCATION_REVIEW
                )

    def test_known_obfuscator_runtimes_and_opaque_extensions_are_rejected(self) -> None:
        cases = (
            ("solution/codec.py", "from pyarmor_runtime_000000 import __pyarmor__\n"),
            ("solution/codec.pyc", "compiled-bytecode"),
            ("solution/native.pyd", "native-extension"),
            ("solution/bundle.pyz", "packed-python"),
            ("solution/runtime.wasm", "opaque-runtime"),
        )
        for name, content in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_template(root)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

                report = RepositoryStaticValidator().validate(root)

                self.assertEqual(report.status, ValidationStatus.REJECTED)
                self.assertIn(
                    ValidationReason.OBFUSCATION_REVIEW,
                    {finding.reason_code for finding in report.findings},
                )

    def test_modified_canonical_sdk_tool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_template(root)
            with (root / "miner/common_preflight.py").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("\n# modified\n")
            report = RepositoryStaticValidator().validate(root)
            self.assertEqual(report.status, ValidationStatus.REJECTED)
            self.assertEqual(report.reason_code, ValidationReason.SDK_TOOL_MODIFIED)


class FakeGitTransport:
    def clone_and_pin(self, repository_url: str, github_pat: str, destination: Path):
        assert github_pat.startswith("github_")
        destination.mkdir(parents=True)
        write_template(destination)
        (destination / ".git").mkdir()
        (destination / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        return "a" * 40, "b" * 40, "2026-07-14T10:00:00+00:00"


class ObfuscatedFakeGitTransport(FakeGitTransport):
    def clone_and_pin(self, repository_url: str, github_pat: str, destination: Path):
        pinned = super().clone_and_pin(
            repository_url, github_pat, destination
        )
        (destination / "miner" / "compression" / "app.py").write_text(
            "payload='" + "A" * 1100 + "'\n", encoding="utf-8"
        )
        return pinned


class RevisionGitTransport:
    def __init__(self, revisions: list[tuple[str, str]]) -> None:
        self.revisions = revisions
        self.calls = 0

    def clone_and_pin(self, repository_url: str, github_pat: str, destination: Path):
        assert github_pat.startswith("github_")
        commit, source = self.revisions[self.calls]
        self.calls += 1
        destination.mkdir(parents=True)
        write_template(destination)
        (destination / "miner" / "compression" / "app.py").write_text(
            source, encoding="utf-8"
        )
        (destination / ".git").mkdir()
        return commit, commit, "2026-07-14T10:00:00+00:00"


class IntakeTests(unittest.TestCase):
    def test_secure_git_transport_captures_sha_and_removes_origin(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(
                ["git", "-C", str(source), "config", "user.name", "Phase Two"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "config",
                    "user.email",
                    "phase2@example.invalid",
                ],
                check=True,
            )
            write_template(source)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(source), "commit", "-q", "-m", "template"],
                check=True,
            )
            destination = root / "clone"
            commit, tree, committed_at = SecureGitTransport().clone_and_pin(
                str(source), "not-used-for-local-clone", destination
            )
            self.assertRegex(commit, r"^[0-9a-f]{40}$")
            self.assertRegex(tree, r"^[0-9a-f]{40}$")
            self.assertIn("T", committed_at)
            remotes = subprocess.run(
                ["git", "-C", str(destination), "remote"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(remotes.stdout.strip(), "")

    def test_clone_is_pinned_credential_free_and_immutable(self) -> None:
        token = "github_" + "pat-secret"
        with tempfile.TemporaryDirectory() as temp:
            submission = RepositorySubmission(
                "competition-1",
                "hot/key",
                "https://github.com/acme/private.git",
                token,
                "nonce-1",
            )
            pinned = RepositoryIntake(Path(temp), transport=FakeGitTransport()).accept(
                submission
            )
            self.assertEqual(pinned.commit_sha, "a" * 40)
            self.assertEqual(pinned.validation.status, ValidationStatus.ACCEPTED)
            self.assertEqual(submission.github_pat, "")
            artifacts = "".join(
                path.read_text(errors="ignore")
                for path in pinned.source_path.parent.rglob("*")
                if path.is_file()
            )
            self.assertNotIn(token, artifacts)
            self.assertNotIn(
                "repository_url",
                json.loads((pinned.source_path.parent / "submission.json").read_text()),
            )
            self.assertFalse(
                os.access(
                    pinned.source_path / "miner" / "compression" / "app.py", os.W_OK
                )
            )
            with self.assertRaisesRegex(IntakeError, "SUBMISSION_ALREADY_PINNED"):
                RepositoryIntake(Path(temp), transport=FakeGitTransport()).accept(
                    RepositorySubmission(
                        "competition-1",
                        "hot/key",
                        "https://github.com/acme/private.git",
                        token,
                        "nonce-1",
                    )
                )

    def test_invalid_submission_also_discards_pat(self) -> None:
        submission = RepositorySubmission(
            "competition-1", "hotkey", "file:///tmp/repository", "secret", "nonce"
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(IntakeError, "REPOSITORY_URL_INVALID"),
        ):
            RepositoryIntake(Path(temp), transport=FakeGitTransport()).accept(
                submission
            )
        self.assertEqual(submission.github_pat, "")

    def test_finalized_poll_is_persisted_without_credentials(self) -> None:
        class FakeRepository:
            recorded = None

            def record_pinned_contender(self, **values):
                self.recorded = values

        repository = FakeRepository()
        with tempfile.TemporaryDirectory() as temp:
            service = CompetitionSubmissionIntakeService(
                RepositoryIntake(Path(temp), transport=FakeGitTransport()),
                repository,
                clock=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
            submission = RepositorySubmission(
                "competition-1",
                "hotkey",
                "https://github.com/acme/private.git",
                "github_pat-secret",
                "nonce",
            )
            service.finalize(
                submission,
                expected_competition_id="competition-1",
                expected_hotkey="hotkey",
                expected_nonce="nonce",
                actor="validator:test",
                uid_snapshot=7,
            )
        self.assertEqual(repository.recorded["pinned_commit_sha"], "a" * 40)
        self.assertEqual(repository.recorded["uid_snapshot"], 7)
        self.assertNotIn("github_pat", repository.recorded)

    def test_pinned_metadata_and_event_persist_in_competition_database(self) -> None:
        now = datetime(2026, 7, 14, tzinfo=timezone.utc)
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_template(source)
            validation = RepositoryStaticValidator().validate(source)
            repository = CompetitionRepository(f"sqlite:///{root / 'competition.db'}")
            repository.insert_manifest(manifest, now=now, actor="validator:test")
            row = repository.record_pinned_contender(
                competition_id=manifest.competition_id,
                hotkey="hotkey",
                repository_url_hash="c" * 64,
                repository_display="github.com/acme/private",
                pinned_commit_sha="a" * 40,
                pinned_tree_sha="b" * 40,
                latest_commit_time=now.isoformat(),
                validation=validation,
                now=now,
                actor="validator:test",
                uid_snapshot=7,
            )
            events = repository.list_events(manifest.competition_id)
        self.assertEqual(row.pinned_tree_sha, "b" * 40)
        self.assertEqual(row.validation_status, "ACCEPTED")
        self.assertEqual(row.status, "ACCEPTED")
        self.assertEqual(events[-1].event_type, "CONTENDER_REPOSITORY_PINNED")
        self.assertNotIn("github_pat", events[-1].payload_json)

    def test_obfuscated_submission_is_collected_as_rejected(self) -> None:
        now = datetime(2026, 7, 14, tzinfo=timezone.utc)
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = CompetitionRepository(f"sqlite:///{root / 'competition.db'}")
            repository.insert_manifest(manifest, now=now, actor="validator:test")
            service = CompetitionSubmissionIntakeService(
                RepositoryIntake(
                    root / "artifacts", transport=ObfuscatedFakeGitTransport()
                ),
                repository,
                clock=lambda: now,
            )
            service.finalize(
                RepositorySubmission(
                    manifest.competition_id,
                    "obfuscated-hotkey",
                    "https://github.com/acme/private.git",
                    "github_pat-secret",
                    "nonce",
                ),
                expected_competition_id=manifest.competition_id,
                expected_hotkey="obfuscated-hotkey",
                expected_nonce="nonce",
                actor="validator:test",
            )
            row = repository.get_contender(
                manifest.competition_id, "obfuscated-hotkey"
            )

        self.assertIsNotNone(row)
        self.assertEqual(row.validation_status, "REJECTED")
        self.assertEqual(row.status, "REJECTED")
        self.assertEqual(row.reason_code, "OBFUSCATION_REVIEW")
        self.assertIn("miner/compression/app.py", row.reason_detail)

    def test_submission_revisions_replace_artifacts_until_finalisation(self) -> None:
        now = datetime(2026, 7, 16, tzinfo=timezone.utc)
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        revisions = RevisionGitTransport(
            [
                ("a" * 40, "VERSION = 'accepted-v1'\n"),
                ("b" * 40, "payload='" + "A" * 1100 + "'\n"),
                ("c" * 40, "VERSION = 'accepted-v3'\n"),
                ("d" * 40, "VERSION = 'too-late-v4'\n"),
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = CompetitionRepository(f"sqlite:///{root / 'competition.db'}")
            repository.insert_manifest(manifest, now=now, actor="validator:test")
            repository.transition(
                manifest.competition_id,
                CompetitionState.ENROLLING,
                expected=CompetitionState.SCHEDULED,
                now=now,
                actor="validator:test",
            )
            service = CompetitionSubmissionIntakeService(
                RepositoryIntake(root / "artifacts", transport=revisions),
                repository,
                clock=lambda: now,
            )

            def submit(repository_name: str) -> None:
                service.finalize(
                    RepositorySubmission(
                        manifest.competition_id,
                        "revision-hotkey",
                        f"https://github.com/acme/{repository_name}.git",
                        "github_pat-secret",
                        "nonce",
                    ),
                    expected_competition_id=manifest.competition_id,
                    expected_hotkey="revision-hotkey",
                    expected_nonce="nonce",
                    actor="validator:test",
                )

            submit("first")
            first = repository.get_contender(
                manifest.competition_id, "revision-hotkey"
            )
            self.assertEqual(first.status, "ACCEPTED")
            self.assertEqual(first.submission_revision, 1)

            submit("second")
            rejected = repository.get_contender(
                manifest.competition_id, "revision-hotkey"
            )
            self.assertEqual(rejected.status, "REJECTED")
            self.assertEqual(rejected.submission_revision, 2)
            self.assertEqual(rejected.pinned_commit_sha, "b" * 40)

            submit("corrected")
            accepted = repository.get_contender(
                manifest.competition_id, "revision-hotkey"
            )
            self.assertEqual(accepted.status, "ACCEPTED")
            self.assertEqual(accepted.submission_revision, 3)
            self.assertEqual(accepted.pinned_commit_sha, "c" * 40)
            source = pinned_repository_source(
                root / "artifacts", manifest.competition_id, "revision-hotkey"
            )
            self.assertIn(
                "accepted-v3",
                (source / "miner" / "compression" / "app.py").read_text(),
            )
            contender_parent = source.parent.parent
            self.assertFalse(
                any("previous" in path.name for path in contender_parent.iterdir())
            )

            repository.transition(
                manifest.competition_id,
                CompetitionState.FINALIZING_SUBMISSIONS,
                expected=CompetitionState.ENROLLING,
                now=now,
                actor="validator:test",
            )
            with self.assertRaisesRegex(ValueError, "after finalisation"):
                submit("too-late")
            unchanged = repository.get_contender(
                manifest.competition_id, "revision-hotkey"
            )
            self.assertEqual(unchanged.submission_revision, 3)
            self.assertEqual(unchanged.pinned_commit_sha, "c" * 40)
            self.assertIn(
                "accepted-v3",
                (source / "miner" / "compression" / "app.py").read_text(),
            )

            replacement_events = [
                event
                for event in repository.list_events(manifest.competition_id)
                if event.event_type == "CONTENDER_REPOSITORY_REPLACED"
            ]
            self.assertEqual(len(replacement_events), 2)

    def test_poll_binding_rejects_replay_and_hotkey_mismatch(self) -> None:
        base = RepositorySubmission(
            "competition-1",
            "hotkey",
            "https://github.com/acme/private.git",
            "secret",
            "nonce",
        )
        validate_polled_submission(
            base,
            expected_competition_id="competition-1",
            expected_hotkey="hotkey",
            expected_nonce="nonce",
        )
        with self.assertRaisesRegex(IntakeError, "SUBMISSION_REPLAY"):
            validate_polled_submission(
                base,
                expected_competition_id="competition-1",
                expected_hotkey="hotkey",
                expected_nonce="other",
            )
        with self.assertRaisesRegex(IntakeError, "HOTKEY_MISMATCH"):
            validate_polled_submission(
                base,
                expected_competition_id="competition-1",
                expected_hotkey="other",
                expected_nonce="nonce",
            )

    def test_intake_service_discards_pat_when_poll_binding_fails(self) -> None:
        class UnusedRepository:
            pass

        submission = RepositorySubmission(
            "competition-1",
            "hotkey",
            "https://github.com/acme/private.git",
            "secret",
            "stale",
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(IntakeError, "SUBMISSION_REPLAY"),
        ):
            CompetitionSubmissionIntakeService(
                RepositoryIntake(Path(temp), transport=FakeGitTransport()),
                UnusedRepository(),
            ).finalize(
                submission,
                expected_competition_id="competition-1",
                expected_hotkey="hotkey",
                expected_nonce="fresh",
                actor="validator:test",
            )
        self.assertEqual(submission.github_pat, "")


class FakeInspector:
    def inspect(self, path: Path) -> MediaInfo:
        if str(path).startswith("/output"):
            return MediaInfo(320, 180, 5.0, "av1", "mov,mp4", "yuv420p", "1:1", 50)
        return MediaInfo(320, 180, 5.0, "h264", "mov,mp4", "yuv420p", "1:1", 100)


class FakeQualificationClient:
    def health(self):
        return {"status": "ok", "competition_local_io": {"remote_io_disabled": True}}

    def compress(self, request):
        item = request.items[0]
        return CompetitionCompressionResponse(
            results=(CompetitionCompressionResult(output_path=item.output_path),),
        )


class QualificationTests(unittest.TestCase):
    def test_health_and_local_compression_qualification(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        result = WarmupQualifier(FakeQualificationClient(), FakeInspector()).qualify(
            manifest, repository_root=ROOT, contender_hotkey="hotkey"
        )
        self.assertTrue(result.health_ok)
        self.assertEqual(result.output_media.codec, "av1")


if __name__ == "__main__":
    unittest.main()
