from __future__ import annotations

import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path


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

from vidaio_subnet_core.competition.artifact_backup import (  # noqa: E402
    CompetitionArtifactBackupError,
    CompetitionArtifactBackupService,
    CompetitionDatabaseBackupError,
)
from vidaio_subnet_core.competition.config import (  # noqa: E402
    BossConfig,
    CompetitionConfig,
    load_manifest,
)
from vidaio_subnet_core.competition.intake import (  # noqa: E402
    boss_contender_id,
    pinned_repository_source,
)
from vidaio_subnet_core.competition.manager import CompetitionManager  # noqa: E402
from vidaio_subnet_core.competition.repository import (  # noqa: E402
    CompetitionRepository,
)
from vidaio_subnet_core.competition.state import CompetitionState  # noqa: E402
from vidaio_subnet_core.competition.validation import (  # noqa: E402
    ValidationReason,
    ValidationReport,
    ValidationStatus,
)


NOW = datetime(2026, 7, 17, 0, 0, tzinfo=timezone.utc)
PRIVATE_BUCKET = "validator-private-artifacts"


class _AcceptingBossValidator:
    def validate(self, root: Path) -> ValidationReport:
        files = [path for path in root.rglob("*") if path.is_file()]
        return ValidationReport(
            ValidationStatus.ACCEPTED,
            ValidationReason.ACCEPTED,
            "e" * 64,
            len(files),
            sum(path.stat().st_size for path in files),
            (),
        )


class FakeS3:
    def __init__(
        self, *, public_acl: bool = False, public_bucket_acl: bool = False
    ) -> None:
        self.public_acl = public_acl
        self.public_bucket_acl = public_bucket_acl
        self.objects: dict[tuple[str, str], dict[str, object]] = {}
        self.upload_count = 0

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.upload_count += 1
        self.objects[(bucket, key)] = {
            "body": Path(filename).read_bytes(),
            "content_type": ExtraArgs["ContentType"],
            "metadata": ExtraArgs["Metadata"],
            "extra_args": dict(ExtraArgs),
        }

    def put_object(self, *, Bucket, Key, Body, ContentType, Metadata):
        self.upload_count += 1
        self.objects[(Bucket, Key)] = {
            "body": bytes(Body),
            "content_type": ContentType,
            "metadata": Metadata,
        }

    def head_object(self, *, Bucket, Key):
        uploaded = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(uploaded["body"]),
            "Metadata": uploaded["metadata"],
        }

    def get_object_acl(self, *, Bucket, Key):
        del Bucket, Key
        if not self.public_acl:
            return {"Grants": []}
        return {
            "Grants": [
                {
                    "Grantee": {
                        "URI": "http://acs.amazonaws.com/groups/global/AllUsers"
                    },
                    "Permission": "READ",
                }
            ]
        }

    def get_bucket_acl(self, *, Bucket):
        del Bucket
        if not self.public_bucket_acl:
            return {"Grants": []}
        return {
            "Grants": [
                {
                    "Grantee": {
                        "URI": "http://acs.amazonaws.com/groups/global/AllUsers"
                    },
                    "Permission": "READ",
                }
            ]
        }


class CompetitionArtifactBackupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.repository_root = root
        self.artifact_root = root / "artifacts"
        self.database_url = f"sqlite:///{root / 'competition.db'}"
        self.repository = CompetitionRepository(self.database_url)
        self.hotkey = "miner-hotkey"
        self.boss_source = root / "competition_boss/current"
        self.boss_source.mkdir(parents=True)
        (self.boss_source / "main.py").write_text(
            "print('boss snapshot')\n", encoding="utf-8"
        )
        self.manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        ).model_copy(
            update={
                "boss": BossConfig(
                    repository_path=Path("competition_boss/current"),
                    boss_hotkey=self.hotkey,
                )
            }
        )
        manager = CompetitionManager(
            CompetitionConfig(
                mode_enabled=True,
                database_url=self.database_url,
                artifact_root=self.artifact_root,
                artifact_backup_bucket=PRIVATE_BUCKET,
                owner_id="validator:test",
            ),
            self.repository,
            clock=lambda: NOW,
        )
        manager.register_manifest(self.manifest)
        self.repository.transition(
            self.manifest.competition_id,
            CompetitionState.ENROLLING,
            expected=CompetitionState.SCHEDULED,
            now=NOW,
            actor="test",
        )
        self.repository.transition(
            self.manifest.competition_id,
            CompetitionState.FINALIZING_SUBMISSIONS,
            expected=CompetitionState.ENROLLING,
            now=NOW,
            actor="test",
        )
        source = pinned_repository_source(
            self.artifact_root, self.manifest.competition_id, self.hotkey
        )
        source.mkdir(parents=True)
        (source / "main.py").write_text("print('snapshot')\n", encoding="utf-8")
        (source.parent / "submission.json").write_text(
            json.dumps({"pinned_commit_sha": "a" * 40}) + "\n",
            encoding="utf-8",
        )
        self.repository.record_pinned_contender(
            competition_id=self.manifest.competition_id,
            hotkey=self.hotkey,
            repository_url_hash="b" * 64,
            repository_display="github.com/example/private-repository",
            pinned_commit_sha="a" * 40,
            pinned_tree_sha="c" * 40,
            latest_commit_time=NOW.isoformat(),
            validation=ValidationReport(
                ValidationStatus.ACCEPTED,
                ValidationReason.ACCEPTED,
                "d" * 64,
                1,
                18,
                (),
            ),
            now=NOW,
            actor="test",
        )

    def tearDown(self) -> None:
        self.repository.engine.dispose()
        self.temp.cleanup()

    def service(self, s3: FakeS3) -> CompetitionArtifactBackupService:
        service = CompetitionArtifactBackupService(
            self.repository,
            artifact_root=self.artifact_root,
            bucket=PRIVATE_BUCKET,
            s3_client=s3,
            database_url=self.database_url,
            repository_root=self.repository_root,
            clock=lambda: NOW,
        )
        service.boss_validator = _AcceptingBossValidator()
        return service

    def complete_competition(self) -> None:
        current = CompetitionState.FINALIZING_SUBMISSIONS
        for target in (
            CompetitionState.VALIDATING,
            CompetitionState.BUILDING,
            CompetitionState.EVALUATING,
            CompetitionState.SCORING,
            CompetitionState.AWAITING_END_TIME,
            CompetitionState.COMPLETED,
        ):
            self.repository.transition(
                self.manifest.competition_id,
                target,
                expected=current,
                now=NOW,
                actor="test",
            )
            current = target

    def test_uploads_private_archive_and_inventory_and_returns_bucket_path(self):
        s3 = FakeS3()
        service = self.service(s3)

        result = service.backup(self.manifest.competition_id)

        self.assertEqual(
            result.s3_uri,
            f"s3://{PRIVATE_BUCKET}/{result.prefix}/",
        )
        self.assertIn(self.manifest.competition_id, result.prefix)
        self.assertIn(self.manifest.competition_id, result.archive_key)
        self.assertNotIn("?", result.s3_uri)
        self.assertNotIn("http", result.s3_uri)
        self.assertEqual(s3.upload_count, 2)
        for uploaded in s3.objects.values():
            self.assertRegex(uploaded["metadata"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn(
            "ACL", s3.objects[(PRIVATE_BUCKET, result.archive_key)]["extra_args"]
        )

        inventory = json.loads(
            s3.objects[(PRIVATE_BUCKET, result.inventory_key)]["body"]
        )
        self.assertEqual(inventory["competition_id"], self.manifest.competition_id)
        self.assertEqual(inventory["visibility"], "PRIVATE")
        self.assertEqual(inventory["archive"]["key"], result.archive_key)
        inventory_by_id = {
            contender["hotkey"]: contender for contender in inventory["contenders"]
        }
        boss_id = boss_contender_id(self.hotkey)
        self.assertEqual(set(inventory_by_id), {self.hotkey, boss_id})
        self.assertFalse(inventory_by_id[self.hotkey]["is_boss"])
        self.assertTrue(inventory_by_id[boss_id]["is_boss"])
        self.assertEqual(inventory_by_id[boss_id]["payout_hotkey"], self.hotkey)

        archive_bytes = s3.objects[(PRIVATE_BUCKET, result.archive_key)]["body"]
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            names = set(archive.getnames())
        archive_root = f"competition_artifacts/{self.manifest.competition_id}"
        self.assertIn(f"{archive_root}/manifest.normalized.json", names)
        self.assertIn(
            f"{archive_root}/contenders/{self.hotkey}/source/main.py",
            names,
        )
        self.assertIn(
            f"{archive_root}/contenders/{boss_id}/source/main.py",
            names,
        )

        boss = self.repository.get_contender(self.manifest.competition_id, boss_id)
        self.assertTrue(boss.is_boss)
        self.assertEqual(boss.repository_display, "manifest:competition_boss/current")

        competition = self.repository.get(self.manifest.competition_id)
        self.assertEqual(competition.submission_backup_status, "COMPLETED")
        self.assertEqual(competition.submission_backup_prefix, result.prefix)
        self.assertEqual(competition.submission_backup_archive_key, result.archive_key)
        self.assertEqual(
            competition.submission_backup_inventory_key, result.inventory_key
        )
        self.assertEqual(competition.submission_backup_checksum, result.archive_sha256)
        self.assertEqual(competition.submission_backup_contender_count, 2)
        self.assertEqual(
            self.repository.list_events(self.manifest.competition_id)[-1].event_type,
            "CONTENDER_SUBMISSION_SNAPSHOTS_BACKED_UP",
        )

        repeated = service.backup(self.manifest.competition_id)
        self.assertEqual(repeated, result)
        self.assertEqual(s3.upload_count, 2)

    def test_public_acl_fails_closed_and_private_retry_can_complete(self):
        with self.assertRaisesRegex(CompetitionArtifactBackupError, "public ACL grant"):
            self.service(FakeS3(public_acl=True)).backup(self.manifest.competition_id)

        failed = self.repository.get(self.manifest.competition_id)
        self.assertEqual(failed.status, CompetitionState.FINALIZING_SUBMISSIONS.value)
        self.assertEqual(failed.submission_backup_status, "FAILED")
        self.assertIn("public ACL grant", failed.submission_backup_error)

        result = self.service(FakeS3()).backup(self.manifest.competition_id)
        completed = self.repository.get(self.manifest.competition_id)
        self.assertEqual(completed.submission_backup_status, "COMPLETED")
        self.assertEqual(completed.submission_backup_prefix, result.prefix)

    def test_public_bucket_acl_fails_before_any_upload(self):
        s3 = FakeS3(public_bucket_acl=True)
        with self.assertRaisesRegex(
            CompetitionArtifactBackupError, "bucket.*public ACL grant"
        ):
            self.service(s3).backup(self.manifest.competition_id)

        self.assertEqual(s3.upload_count, 0)
        failed = self.repository.get(self.manifest.competition_id)
        self.assertEqual(failed.submission_backup_status, "FAILED")
        self.assertIn("bucket", failed.submission_backup_error)

    def test_completed_competition_uploads_consistent_sqlite_snapshot(self):
        self.complete_competition()
        s3 = FakeS3()
        service = self.service(s3)

        result = service.backup_database(self.manifest.competition_id)

        self.assertEqual(s3.upload_count, 1)
        self.assertEqual(result.bucket, PRIVATE_BUCKET)
        self.assertTrue(result.key.endswith("/competition.db"))
        uploaded = s3.objects[(PRIVATE_BUCKET, result.key)]
        self.assertEqual(uploaded["content_type"], "application/vnd.sqlite3")
        self.assertEqual(uploaded["metadata"]["sha256"], result.sha256)
        self.assertTrue(uploaded["body"].startswith(b"SQLite format 3\x00"))

        with tempfile.TemporaryDirectory() as temp:
            restored = Path(temp) / "restored.sqlite3"
            restored.write_bytes(uploaded["body"])
            with sqlite3.connect(restored) as connection:
                integrity = connection.execute("PRAGMA integrity_check").fetchone()
                status = connection.execute(
                    "SELECT status FROM competitions WHERE competition_id=?",
                    (self.manifest.competition_id,),
                ).fetchone()
        self.assertEqual(integrity, ("ok",))
        self.assertEqual(status, (CompetitionState.COMPLETED.value,))

        record = self.repository.database_backup_record(
            self.manifest.competition_id
        )
        self.assertEqual(record["key"], result.key)
        self.assertEqual(record["sha256"], result.sha256)
        self.assertEqual(record["size_bytes"], result.size_bytes)
        self.assertEqual(
            self.repository.list_events(self.manifest.competition_id)[-1].event_type,
            "COMPETITION_DATABASE_BACKED_UP",
        )
        self.assertEqual(
            self.repository.list_completed_pending_database_backup(), []
        )

        repeated = service.backup_database(self.manifest.competition_id)
        self.assertEqual(repeated, result)
        self.assertEqual(s3.upload_count, 1)

    def test_database_backup_failure_is_retryable(self):
        self.complete_competition()
        s3 = FakeS3(public_bucket_acl=True)

        with self.assertRaisesRegex(CompetitionDatabaseBackupError, "public ACL"):
            self.service(s3).backup_database(self.manifest.competition_id)

        self.assertEqual(s3.upload_count, 0)
        self.assertIsNone(
            self.repository.database_backup_record(self.manifest.competition_id)
        )
        self.assertEqual(
            self.repository.list_completed_pending_database_backup()[0].competition_id,
            self.manifest.competition_id,
        )

        private_s3 = FakeS3()
        result = self.service(private_s3).backup_database(
            self.manifest.competition_id
        )
        self.assertEqual(private_s3.upload_count, 1)
        self.assertEqual(
            self.repository.database_backup_record(self.manifest.competition_id)[
                "key"
            ],
            result.key,
        )


if __name__ == "__main__":
    unittest.main()
