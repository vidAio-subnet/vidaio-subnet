from __future__ import annotations

import importlib
import hashlib
import json
import sys
import tempfile
import types
import unittest
import ast
import os
from io import StringIO
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError
from loguru import logger
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError


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

from vidaio_subnet_core.competition.config import (  # noqa: E402
    CompetitionConfig,
    CompetitionManifest,
    load_manifest,
)
from vidaio_subnet_core.competition.manager import CompetitionManager  # noqa: E402
from vidaio_subnet_core.competition.migrations import (  # noqa: E402
    apply_competition_migrations,
)
from vidaio_subnet_core.competition.repository import (  # noqa: E402
    CompetitionRepository,
    utc_iso,
)
from vidaio_subnet_core.competition.state import CompetitionState  # noqa: E402


START = datetime(2026, 7, 16, 0, 0, tzinfo=timezone.utc)  # Thursday


def manifest_data(competition_id: str = "compression-2026-w29") -> dict:
    return {
        "schema_version": 2,
        "scoring_version": "3",
        "competition_id": competition_id,
        "competition_type": "COMPRESSION",
        "competition_start_time": START.isoformat(),
        "contender_ping_interval": "30m",
        "contender_finalisation_time": (START + timedelta(days=1)).isoformat(),
        "human_review_deadline": (START + timedelta(days=6, hours=23)).isoformat(),
        "competition_end_time": (START + timedelta(days=7)).isoformat(),
        "required_routes": ["/compress"],
        "allowed_gpus": ["L4", "L40S"],
        "max_cpu_cores": 32,
        "requested_cpu_cores": 16,
        "container_size_limit_gb": 25,
        "evaluation_batch_size": 5,
        "evaluation_batched_run_timeout": "10m",
        "scoring_batched_run_timeout": "5m",
        "min_video_length": "5s",
        "max_video_length": "1h",
        "length_weight_exponent": 2.02,
        "required_output_codec": "AV1",
        "vmaf_threshold": 90.0,
        "vmaf_sample_count": 10,
        "minimum_compression_ratio": 1.25,
        "scoring_seed": 20260716,
        "warmup_input_path": "competitions/fixtures/compression_warmup_input.mp4",
        "boss": {
            "repository_path": None,
            "boss_hotkey": None,
        },
        "evaluation_input_volume_name": f"{competition_id}-inputs",
        "evaluation_index_path": "/validator-evaluation/index.json",
        "output_volume_prefix": f"{competition_id}-output",
        "max_parallel_contenders": 4,
        "max_attempts_per_item": 2,
        "scoring_factors": {
            "quality": "0.6",
            "cost_efficiency": "0.25",
            "length_coverage": "0.15",
            "runtime": "0",
        },
        "cost_floor_usd": "0.000001",
        "score_precision": 8,
    }


def make_manifest(competition_id: str = "compression-2026-w29") -> CompetitionManifest:
    return CompetitionManifest.model_validate(manifest_data(competition_id))


class FakeClock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class ManifestTests(unittest.TestCase):
    def test_modal_build_timeout_defaults_to_ten_minutes_and_is_configurable(
        self,
    ) -> None:
        self.assertEqual(make_manifest().modal_build_timeout, timedelta(minutes=10))
        configured = manifest_data()
        configured["modal_build_timeout"] = "7m"
        self.assertEqual(
            CompetitionManifest.model_validate(configured).modal_build_timeout,
            timedelta(minutes=7),
        )

    def test_scoring_timeout_is_independent_of_inference_timeout(self) -> None:
        configured = manifest_data()
        configured["evaluation_batched_run_timeout"] = "20m"
        configured["scoring_batched_run_timeout"] = "7m"

        manifest = CompetitionManifest.model_validate(configured)

        self.assertEqual(
            manifest.evaluation_batched_run_timeout, timedelta(minutes=20)
        )
        self.assertEqual(
            manifest.scoring_batched_run_timeout, timedelta(minutes=7)
        )

    def test_length_weight_exponent_is_configurable_and_backward_compatible(
        self,
    ) -> None:
        configured = manifest_data()
        self.assertEqual(
            CompetitionManifest.model_validate(configured).length_weight_exponent,
            2.02,
        )
        configured.pop("length_weight_exponent")
        self.assertEqual(
            CompetitionManifest.model_validate(configured).length_weight_exponent,
            1.0,
        )

    def test_normalization_and_digest_are_deterministic(self) -> None:
        first = make_manifest()
        second = CompetitionManifest.model_validate(
            json.loads(json.dumps(manifest_data(), sort_keys=True))
        )
        self.assertEqual(first.normalized_json(), second.normalized_json())
        self.assertEqual(first.digest(), second.digest())
        self.assertEqual(len(first.digest()), 64)
        restored = CompetitionManifest.model_validate_json(first.normalized_json())
        self.assertEqual(restored, first)

    def test_repository_example_manifest_loads(self) -> None:
        manifest = load_manifest(
            ROOT
            / "competitions"
            / "manifests"
            / "examples"
            / "compression-competition.json"
        )
        self.assertEqual(manifest.competition_type, "COMPRESSION")
        self.assertEqual(manifest.container_size_limit_gb, 25)

    def test_accepts_manifest_selected_scoring_version_and_weights(self) -> None:
        payload = manifest_data()
        payload["scoring_version"] = "1"
        payload["scoring_factors"] = {
            "quality": "0.333333333333333",
            "cost_efficiency": "0.333333333333333",
            "length_coverage": "0.333333333333334",
            "runtime": "0",
        }

        manifest = CompetitionManifest.model_validate(payload)

        self.assertEqual(manifest.scoring_version, "1")
        self.assertEqual(
            manifest.scoring_factors.quality,
            Decimal("0.333333333333333"),
        )
        self.assertEqual(
            manifest.scoring_factors.cost_efficiency,
            Decimal("0.333333333333333"),
        )
        self.assertEqual(
            manifest.scoring_factors.length_coverage,
            Decimal("0.333333333333334"),
        )

    def test_rejects_non_thursday_wrong_routes_and_invalid_weights(self) -> None:
        cases = []
        wrong_day = manifest_data()
        wrong_day["competition_start_time"] = (START + timedelta(days=1)).isoformat()
        wrong_day["contender_finalisation_time"] = (
            START + timedelta(days=2)
        ).isoformat()
        cases.append(wrong_day)
        wrong_route = manifest_data()
        wrong_route["required_routes"] = ["/upscale"]
        cases.append(wrong_route)
        weights_do_not_sum = manifest_data()
        weights_do_not_sum["scoring_factors"]["quality"] = "0.5"
        cases.append(weights_do_not_sum)
        negative_weight = manifest_data()
        negative_weight["scoring_factors"] = {
            "quality": "-0.1",
            "cost_efficiency": "0.5",
            "length_coverage": "0.6",
            "runtime": "0",
        }
        cases.append(negative_weight)
        runtime_weight = manifest_data()
        runtime_weight["scoring_factors"] = {
            "quality": "0.5",
            "cost_efficiency": "0.25",
            "length_coverage": "0.15",
            "runtime": "0.1",
        }
        cases.append(runtime_weight)
        invalid_scoring_version = manifest_data()
        invalid_scoring_version["scoring_version"] = "../1"
        cases.append(invalid_scoring_version)
        removed_cost_reference = manifest_data()
        removed_cost_reference["cost_efficiency_reference_usd"] = "0.02"
        cases.append(removed_cost_reference)
        for invalid_exponent in (0, -1, 10.01, float("inf"), float("nan")):
            invalid_length_weight = manifest_data()
            invalid_length_weight["length_weight_exponent"] = invalid_exponent
            cases.append(invalid_length_weight)
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                CompetitionManifest.model_validate(payload)

    def test_runtime_warmup_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "warmup fixture is missing"):
                make_manifest().validate_runtime_paths(Path(temp_dir))

    def test_boss_repository_path_is_relative_and_sdk_export_shaped(self) -> None:
        for invalid_path in (".", "../boss", "/tmp/boss"):
            payload = manifest_data()
            payload["boss"] = {
                "repository_path": invalid_path,
                "boss_hotkey": "boss-hotkey",
            }
            with self.subTest(path=invalid_path), self.assertRaises(ValidationError):
                CompetitionManifest.model_validate(payload)

        invalid_pairs = (
            {"repository_path": "competition_boss/current", "boss_hotkey": None},
            {"repository_path": None, "boss_hotkey": "boss-hotkey"},
            {"repository_path": "competition_boss/current", "boss_hotkey": " "},
        )
        for boss in invalid_pairs:
            payload = manifest_data()
            payload["boss"] = boss
            with self.subTest(boss=boss), self.assertRaises(ValidationError):
                CompetitionManifest.model_validate(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            warmup = root / "competitions/fixtures/compression_warmup_input.mp4"
            warmup.parent.mkdir(parents=True)
            warmup.write_bytes(b"warmup")
            boss = root / "competition_boss/current"
            for relative in (
                ".vidaio-sdk-export",
                "competition_solution.json",
                "requirements.txt",
                "miner/modal_workers.py",
                "scripts/competition_modal_build.py",
                "competitions/fixtures/compression_warmup_input.mp4",
            ):
                path = boss / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")

            payload = manifest_data()
            payload["boss"] = {
                "repository_path": "competition_boss/current",
                "boss_hotkey": "boss-hotkey",
            }
            manifest = CompetitionManifest.model_validate(payload)
            manifest.validate_runtime_paths(root)

            (boss / ".vidaio-sdk-export").unlink()
            with self.assertRaisesRegex(
                ValueError, "complete competition_sdk.py export"
            ):
                manifest.validate_runtime_paths(root)

    def test_runtime_config_uses_default_off_feature_flag(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(CompetitionConfig(_env_file=None).mode_enabled)
        enabled_settings = {
            "COMPETITION_MODE_ENABLED": "true",
            "COMPETITION_ARTIFACT_BACKUP_BUCKET": "private-test-bucket",
        }
        with patch.dict(os.environ, enabled_settings, clear=True):
            self.assertTrue(CompetitionConfig(_env_file=None).mode_enabled)

    def test_competition_mode_requires_a_private_artifact_backup_bucket(self) -> None:
        with patch.dict(
            os.environ, {"COMPETITION_MODE_ENABLED": "true"}, clear=True
        ), self.assertRaisesRegex(ValidationError, "ARTIFACT_BACKUP_BUCKET"):
            CompetitionConfig(_env_file=None)

    def test_modal_execution_requires_explicit_size_attestation_acknowledgement(
        self,
    ) -> None:
        settings = {
            "COMPETITION_MODE_ENABLED": "true",
            "COMPETITION_EXECUTION_ENABLED": "true",
            "COMPETITION_BUILD_BACKEND": "modal",
            "COMPETITION_ARTIFACT_BACKUP_BUCKET": "private-test-bucket",
        }
        with patch.dict(os.environ, settings, clear=True), self.assertRaisesRegex(
            ValidationError, "ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION"
        ):
            CompetitionConfig(_env_file=None)
        settings[
            "COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION"
        ] = "true"
        with patch.dict(os.environ, settings, clear=True):
            config = CompetitionConfig(_env_file=None)
        self.assertTrue(config.execution_enabled)
        self.assertEqual(config.build_backend, "modal")


class PersistenceAndLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.database_url = f"sqlite:///{root / 'competition.db'}"
        self.artifact_root = root / "artifacts"
        self.clock = FakeClock(START - timedelta(hours=1))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(
        self, owner: str = "test-owner", enabled: bool = True
    ) -> CompetitionConfig:
        return CompetitionConfig(
            mode_enabled=enabled,
            database_url=self.database_url,
            artifact_root=self.artifact_root,
            artifact_backup_bucket="private-test-bucket",
            owner_id=owner,
            lease_ttl_seconds=30,
        )

    def manager(self, owner: str = "test-owner") -> CompetitionManager:
        repository = CompetitionRepository(self.database_url)
        return CompetitionManager(self.config(owner), repository, clock=self.clock)

    def test_default_off_does_not_create_or_touch_database(self) -> None:
        database_path = Path(self.temp.name) / "disabled.db"
        manager = CompetitionManager(
            CompetitionConfig(
                mode_enabled=False,
                database_url=f"sqlite:///{database_path}",
                artifact_root=self.artifact_root,
            ),
            clock=self.clock,
        )
        self.assertEqual(manager.tick(), [])
        self.assertFalse(database_path.exists())

    def test_scheduler_logs_idle_bootstrap_and_state_transition(self) -> None:
        manager = CompetitionManager(
            CompetitionConfig(
                mode_enabled=True,
                database_url=self.database_url,
                artifact_root=self.artifact_root,
                artifact_backup_bucket="private-test-bucket",
                manifest_glob=str(Path(self.temp.name) / "missing" / "*.json"),
                owner_id="log-test-owner",
            ),
            CompetitionRepository(self.database_url),
            clock=self.clock,
        )
        output = StringIO()
        sink_id = logger.add(output, format="{message}", level="DEBUG")
        try:
            self.assertEqual(manager.bootstrap_manifests(), [])
            manager.register_manifest(make_manifest())
            self.clock.now = START
            self.assertEqual(
                manager.tick(),
                [("compression-2026-w29", CompetitionState.ENROLLING)],
            )
            self.clock.now = START + timedelta(hours=6)
            self.assertEqual(manager.tick(), [])
        finally:
            logger.remove(sink_id)

        messages = output.getvalue()
        self.assertIn("no manifests matched", messages)
        self.assertIn("compression-2026-w29=SCHEDULED", messages)
        self.assertIn(
            "SCHEDULED -> ENROLLING (trigger=competition_start_time)", messages
        )
        self.assertIn(
            "Competition phase countdown: id=compression-2026-w29 "
            "current=ENROLLING next=FINALIZING_SUBMISSIONS",
            messages,
        )
        self.assertIn(
            "time_left=18h source=manifest.contender_finalisation_time",
            messages,
        )

    def test_migration_is_explicit_idempotent_and_separate_from_inference(self) -> None:
        first = CompetitionRepository(self.database_url)
        second = CompetitionRepository(self.database_url)
        self.assertEqual(first.schema_version, 1)
        self.assertEqual(second.schema_version, 1)
        with first.engine.connect() as connection:
            migration_rows = connection.execute(
                text(
                    "SELECT version, name FROM competition_schema_migrations "
                    "ORDER BY version"
                )
            ).all()
        self.assertEqual(
            [tuple(row) for row in migration_rows],
            [(1, "initial_competition_schema")],
        )
        tables = set(inspect(first.engine).get_table_names())
        self.assertTrue(
            {
                "competitions",
                "contender_metadata",
                "contender_performance_history",
                "competition_evaluation_items",
                "competition_batches",
                "competition_human_reviews",
                "competition_events",
                "competition_sandboxes",
                "competition_schema_migrations",
            }
            <= tables
        )
        self.assertNotIn("miner_metadata", tables)
        self.assertNotIn("miner_performance_history", tables)
        for table_name in tables:
            if not table_name.startswith(("competition", "contender")):
                continue
            column_names = {
                column["name"].lower()
                for column in inspect(first.engine).get_columns(table_name)
            }
            self.assertFalse(
                column_names & {"pat", "github_pat", "token", "secret", "password"},
                table_name,
            )
        contender_columns = {
            column["name"]
            for column in inspect(first.engine).get_columns("contender_metadata")
        }
        self.assertTrue(
            {
                "last_invited_at",
                "last_submission_poll_at",
                "invitation_attempts",
                "submission_poll_attempts",
                "submission_revision",
                "reason_detail",
                "manual_disqualified",
                "manual_disqualification_review_id",
                "media_score_aggregate",
            }
            <= contender_columns
        )
        evaluation_columns = {
            column["name"]
            for column in inspect(first.engine).get_columns(
                "competition_evaluation_items"
            )
        }
        self.assertTrue(
            {"canonical_batch_index", "canonical_batch_position"}
            <= evaluation_columns
        )
        history_columns = {
            column["name"]
            for column in inspect(first.engine).get_columns(
                "contender_performance_history"
            )
        }
        self.assertTrue(
            {
                "canonical_batch_index",
                "media_score",
                "media_compression_component",
                "media_vmaf_component",
                "media_score_reason",
            }
            <= history_columns
        )
        batch_columns = {
            column["name"]
            for column in inspect(first.engine).get_columns("competition_batches")
        }
        self.assertIn("canonical_batch_index", batch_columns)
        competition_columns = {
            column["name"]
            for column in inspect(first.engine).get_columns("competitions")
        }
        self.assertTrue(
            {
                "scores_need_recalculation",
                "submission_backup_status",
                "submission_backup_bucket",
                "submission_backup_prefix",
                "submission_backup_archive_key",
                "submission_backup_inventory_key",
                "submission_backup_checksum",
                "submission_backup_size_bytes",
                "submission_backup_contender_count",
                "submission_backup_completed_at",
                "submission_backup_error",
            }
            <= competition_columns
        )
        expected_indexes = {
            "contender_metadata": {"ix_contender_review_eligibility"},
            "competition_batches": {
                "ix_competition_batches_lease",
                "ix_competition_batches_scoring_deadline",
                "ix_competition_batch_canonical",
            },
            "contender_performance_history": {
                "ix_history_dispatch_state",
                "ix_history_competition_contender_canonical_batch",
            },
            "competition_human_reviews": {"ix_competition_reviews_active"},
        }
        for table_name, required_indexes in expected_indexes.items():
            actual_indexes = {
                index["name"]
                for index in inspect(first.engine).get_indexes(table_name)
            }
            self.assertTrue(required_indexes <= actual_indexes, table_name)
        with first.engine.connect() as connection:
            active_guard = connection.execute(
                text(
                    "SELECT sql FROM sqlite_master WHERE type='index' "
                    "AND name='uq_competitions_single_running'"
                )
            ).scalar_one()
        self.assertIn("CREATE UNIQUE INDEX", active_guard)
        self.assertIn("WHERE status IN", active_guard)

    def test_squashed_baseline_rejects_retired_migration_history(self) -> None:
        engine = create_engine(
            f"sqlite:///{Path(self.temp.name) / 'legacy-migrations.db'}",
            future=True,
        )
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE competition_schema_migrations ("
                    "version INTEGER PRIMARY KEY, name VARCHAR(128) NOT NULL, "
                    "applied_at VARCHAR(40) NOT NULL)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO competition_schema_migrations "
                    "(version, name, applied_at) VALUES "
                    "(1, 'initial_competition_control_plane', :applied_at)"
                ),
                {"applied_at": START.isoformat()},
            )
        with self.assertRaisesRegex(
            RuntimeError, "retired pre-production migration history"
        ):
            apply_competition_migrations(engine)
        engine.dispose()

    def test_fake_clock_advances_and_resumes_every_state(self) -> None:
        manager = self.manager("owner-register")
        manager.register_manifest(make_manifest())
        competition_id = "compression-2026-w29"
        self.assertEqual(
            manager.repository.get(competition_id).status,
            CompetitionState.SCHEDULED.value,
        )

        self.clock.now = START
        manager = self.manager("owner-enroll")
        self.assertEqual(manager.tick(), [(competition_id, CompetitionState.ENROLLING)])

        self.clock.now = START + timedelta(days=1)
        manager = self.manager("owner-finalize-submissions")
        self.assertEqual(
            manager.tick(),
            [(competition_id, CompetitionState.FINALIZING_SUBMISSIONS)],
        )

        expected_pipeline = [
            CompetitionState.VALIDATING,
            CompetitionState.BUILDING,
            CompetitionState.EVALUATING,
            CompetitionState.SCORING,
            CompetitionState.AWAITING_END_TIME,
        ]
        for index, expected in enumerate(expected_pipeline):
            manager = self.manager(f"owner-stage-{index}")
            self.assertEqual(manager.complete_current_stage(competition_id), expected)
            self.assertEqual(
                manager.repository.get(competition_id).status, expected.value
            )

        manager = self.manager("owner-await")
        self.assertEqual(manager.tick(), [])
        self.clock.now = START + timedelta(days=7)
        manager = self.manager("owner-complete")
        self.assertEqual(manager.tick(), [(competition_id, CompetitionState.COMPLETED)])
        row = manager.repository.get(competition_id)
        self.assertEqual(row.status, CompetitionState.COMPLETED.value)
        self.assertEqual(row.state_version, 8)

        events = manager.repository.list_events(competition_id)
        self.assertEqual(events[0].event_type, "MANIFEST_REGISTERED")
        self.assertEqual(
            len([event for event in events if event.event_type == "STATE_TRANSITION"]),
            8,
        )

    def test_scheduler_lease_excludes_other_owner_until_expiry(self) -> None:
        repository = CompetitionRepository(self.database_url)
        repository.insert_manifest(
            make_manifest(), now=self.clock.now, actor="register"
        )
        self.assertTrue(
            repository.acquire_scheduler_lease(
                "compression-2026-w29",
                owner="a",
                now=self.clock.now,
                ttl_seconds=30,
            )
        )
        self.assertFalse(
            repository.acquire_scheduler_lease(
                "compression-2026-w29",
                owner="b",
                now=self.clock.now,
                ttl_seconds=30,
            )
        )
        self.assertTrue(
            repository.acquire_scheduler_lease(
                "compression-2026-w29",
                owner="b",
                now=self.clock.now + timedelta(seconds=31),
                ttl_seconds=30,
            )
        )

    def test_only_one_competition_can_be_active(self) -> None:
        manager = self.manager()
        manager.register_manifest(make_manifest("a-competition"))
        manager.register_manifest(make_manifest("b-competition"))
        self.clock.now = START
        transitions = manager.tick()
        self.assertEqual(transitions, [("a-competition", CompetitionState.ENROLLING)])
        self.assertEqual(
            manager.repository.get("b-competition").status,
            CompetitionState.SCHEDULED.value,
        )
        with self.assertRaises(IntegrityError):
            manager.repository.transition(
                "b-competition",
                CompetitionState.ENROLLING,
                expected=CompetitionState.SCHEDULED,
                now=self.clock.now,
                actor="racing-scheduler",
            )

    def test_manifest_configured_boss_repository_path_is_persisted(self) -> None:
        payload = manifest_data()
        payload["boss"] = {
            "repository_path": "competition_boss/current",
            "boss_hotkey": "boss-hotkey",
        }
        manifest = CompetitionManifest.model_validate(payload)

        row = self.manager().register_manifest(manifest)

        self.assertEqual(row.boss_repository_path, "competition_boss/current")
        self.assertEqual(row.boss_hotkey, "boss-hotkey")
        self.assertIsNone(row.source_competition_id)

    def test_nonterminal_manifest_update_is_applied_audited_and_fast_forwarded(
        self,
    ) -> None:
        manager = self.manager()
        original = make_manifest()
        manager.register_manifest(original)
        changed = manifest_data()
        changed["competition_start_time"] = (START - timedelta(days=7)).isoformat()
        changed["contender_finalisation_time"] = (
            START - timedelta(days=6)
        ).isoformat()
        changed["scoring_seed"] = 2
        updated = CompetitionManifest.model_validate(changed)

        output = StringIO()
        sink_id = logger.add(output, format="{message}", level="WARNING")
        try:
            row = manager.register_manifest(updated)
        finally:
            logger.remove(sink_id)

        self.assertEqual(row.manifest_digest, updated.digest())
        self.assertEqual(row.manifest_json, updated.normalized_json())
        self.assertEqual(
            row.start_time, utc_iso(updated.competition_start_time)
        )
        self.assertEqual(
            row.contender_finalisation_time,
            utc_iso(updated.contender_finalisation_time),
        )
        self.assertIn("Competition manifest updated", output.getvalue())
        self.assertIn("competition_start_time", output.getvalue())

        artifact_dir = self.artifact_root / updated.competition_id
        self.assertEqual(
            (artifact_dir / "manifest.normalized.json").read_text(encoding="utf-8"),
            updated.normalized_json() + "\n",
        )
        self.assertEqual(
            (
                artifact_dir
                / "manifest.revisions"
                / f"{original.digest()}.normalized.json"
            ).read_text(encoding="utf-8"),
            original.normalized_json() + "\n",
        )

        events = manager.repository.list_events(updated.competition_id)
        update_event = events[-1]
        self.assertEqual(update_event.event_type, "MANIFEST_UPDATED")
        payload = json.loads(update_event.payload_json)
        self.assertEqual(payload["old_manifest_digest"], original.digest())
        self.assertEqual(payload["new_manifest_digest"], updated.digest())
        self.assertEqual(
            payload["changed_fields"]["scoring_seed"], {"old": 20260716, "new": 2}
        )

        self.assertEqual(
            manager.tick(),
            [
                (updated.competition_id, CompetitionState.ENROLLING),
                (
                    updated.competition_id,
                    CompetitionState.FINALIZING_SUBMISSIONS,
                ),
            ],
        )

    def test_terminal_manifest_update_is_rejected_and_events_are_modifiable(
        self,
    ) -> None:
        manager = self.manager()
        manager.register_manifest(make_manifest())
        manager.cancel("compression-2026-w29", "test complete")
        changed = manifest_data()
        changed["scoring_seed"] = 2
        with self.assertRaisesRegex(ValueError, "terminal competition"):
            manager.register_manifest(CompetitionManifest.model_validate(changed))

        repository = manager.repository
        token = "github_pat_" + "A" * 40
        repository.append_event(
            "compression-2026-w29",
            "CANARY",
            actor="test",
            payload={"github_pat": token, "message": token},
            now=self.clock.now,
        )
        event = repository.list_events("compression-2026-w29")[-1]
        self.assertNotIn(token, event.payload_json)
        self.assertIn("[REDACTED]", event.payload_json)
        with repository.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE competition_events SET event_type='ALTERED' WHERE id=:id"
                ),
                {"id": event.id},
            )
        self.assertEqual(
            repository.list_events("compression-2026-w29")[-1].event_type,
            "ALTERED",
        )
        with repository.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM competition_events WHERE id=:id"),
                {"id": event.id},
            )
        self.assertNotIn(
            event.id,
            {
                value.id
                for value in repository.list_events("compression-2026-w29")
            },
        )

    def test_database_only_manifest_update_reconciles_stale_artifact(self) -> None:
        manager = self.manager()
        original = make_manifest()
        manager.register_manifest(original)
        original_artifact = original.normalized_json() + "\n"

        changed = manifest_data()
        changed["scoring_seed"] = 2
        updated = CompetitionManifest.model_validate(changed)
        manager.repository.insert_manifest(
            updated,
            now=self.clock.now,
            actor="interrupted-update",
        )

        output = StringIO()
        sink_id = logger.add(output, format="{message}", level="WARNING")
        try:
            row = manager.register_manifest(updated)
        finally:
            logger.remove(sink_id)

        self.assertEqual(row.manifest_digest, updated.digest())
        artifact_dir = self.artifact_root / updated.competition_id
        self.assertEqual(
            (artifact_dir / "manifest.normalized.json").read_text(encoding="utf-8"),
            updated.normalized_json() + "\n",
        )
        observed_sha256 = hashlib.sha256(original_artifact.encode("utf-8")).hexdigest()
        observed_path = (
            artifact_dir
            / "manifest.revisions"
            / f"observed-{observed_sha256}.json"
        )
        self.assertEqual(
            observed_path.read_text(encoding="utf-8"), original_artifact
        )
        divergence = manager.repository.list_events(updated.competition_id)[-1]
        self.assertEqual(divergence.event_type, "MANIFEST_ARTIFACT_DIVERGENCE")
        payload = json.loads(divergence.payload_json)
        self.assertEqual(payload["artifact_manifest_digest"], original.digest())
        self.assertEqual(payload["database_manifest_digest"], updated.digest())
        self.assertIn("artifact divergence preserved and reconciled", output.getvalue())


class ProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if "bittensor" not in sys.modules:
            bittensor = types.ModuleType("bittensor")

            class Synapse(BaseModel):
                model_config = ConfigDict(arbitrary_types_allowed=True)
                dendrite: object | None = None

            bittensor.Synapse = Synapse
            sys.modules["bittensor"] = bittensor
        cls.protocol = importlib.import_module("vidaio_subnet_core.protocol")

    def test_raw_pat_round_trip_is_wire_visible_but_repr_hidden(self) -> None:
        p = self.protocol
        token = "github_pat_" + "B" * 40
        synapse = p.CompetitionSubmissionProtocol(
            competition_id="compression-2026-w29",
            manifest_digest="a" * 64,
            request_nonce="request-nonce-0001",
            requested_at=START,
            submission_response=p.CompetitionSubmissionResponse(
                competition_id="compression-2026-w29",
                echo_nonce="request-nonce-0001",
                status=p.CompetitionSubmissionStatus.READY,
                repository_url="https://github.com/example/private.git",
                github_pat=token,
            ),
        )
        wire = synapse.model_dump_json()
        self.assertIn(token, wire)
        self.assertNotIn(token, repr(synapse))
        restored = p.CompetitionSubmissionProtocol.model_validate_json(wire)
        self.assertEqual(restored.submission_response.github_pat, token)

    def test_submission_feedback_round_trips_to_the_miner(self) -> None:
        p = self.protocol
        synapse = p.CompetitionSubmissionProtocol(
            competition_id="compression-2026-w29",
            manifest_digest="a" * 64,
            request_nonce="request-nonce-0001",
            requested_at=START,
            last_submission_status=p.CompetitionSubmissionReviewStatus.REJECTED,
            last_submission_reason_code="OBFUSCATION_REVIEW",
            last_submission_reason_detail="solution/codec.py: PyArmor runtime detected",
            last_pinned_commit_sha="b" * 40,
            submission_revision=2,
        )

        restored = p.CompetitionSubmissionProtocol.model_validate_json(
            synapse.model_dump_json()
        )

        self.assertEqual(
            restored.last_submission_status,
            p.CompetitionSubmissionReviewStatus.REJECTED,
        )
        self.assertEqual(restored.last_submission_reason_code, "OBFUSCATION_REVIEW")
        self.assertIn("PyArmor", restored.last_submission_reason_detail)
        self.assertEqual(restored.last_pinned_commit_sha, "b" * 40)
        self.assertEqual(restored.submission_revision, 2)

    def test_protocol_timestamps_dump_as_bittensor_safe_iso_strings(self) -> None:
        p = self.protocol
        invitation = p.CompetitionInvitationProtocol(
            competition_id="compression-2026-w29",
            manifest_digest="a" * 64,
            registration_deadline=START,
            invitation_nonce="invitation-nonce-001",
        )
        submission = p.CompetitionSubmissionProtocol(
            competition_id="compression-2026-w29",
            manifest_digest="a" * 64,
            request_nonce="submission-nonce-001",
            requested_at=START,
        )

        self.assertEqual(
            invitation.model_dump()["registration_deadline"], START.isoformat()
        )
        self.assertEqual(submission.model_dump()["requested_at"], START.isoformat())
        self.assertEqual(
            p.CompetitionInvitationProtocol.model_validate(
                invitation.model_dump()
            ).registration_deadline,
            START,
        )

    def test_transport_defaults_are_header_safe_and_fail_closed(self) -> None:
        p = self.protocol
        invitation = p.CompetitionInvitationProtocol()
        submission = p.CompetitionSubmissionProtocol()
        invitation_required = set(invitation.model_json_schema().get("required", []))
        submission_required = set(submission.model_json_schema().get("required", []))

        self.assertFalse(
            invitation_required
            & {
                "competition_id",
                "manifest_digest",
                "registration_deadline",
                "invitation_nonce",
            }
        )
        self.assertFalse(
            submission_required
            & {
                "competition_id",
                "manifest_digest",
                "request_nonce",
                "requested_at",
            }
        )
        self.assertFalse(invitation.is_open_invitation(START))
        self.assertFalse(submission.is_fresh_request(START))

        valid_invitation = p.CompetitionInvitationProtocol(
            competition_id="compression-2026-w29",
            manifest_digest="a" * 64,
            registration_deadline=START + timedelta(minutes=5),
            invitation_nonce="invitation-nonce-001",
        )
        valid_submission = p.CompetitionSubmissionProtocol(
            competition_id="compression-2026-w29",
            manifest_digest="a" * 64,
            request_nonce="submission-nonce-001",
            requested_at=START - timedelta(minutes=1),
        )
        self.assertTrue(valid_invitation.is_open_invitation(START))
        self.assertTrue(valid_submission.is_fresh_request(START))

    def test_cross_competition_nonce_and_invalid_repository_are_rejected(self) -> None:
        p = self.protocol
        with self.assertRaises(ValidationError):
            p.CompetitionSubmissionResponse(
                status=p.CompetitionSubmissionStatus.READY,
                repository_url="https://evil.example/repository.git",
                github_pat="secret",
            )
        with self.assertRaises(ValidationError):
            p.CompetitionInvitationProtocol(
                competition_id="compression-2026-w29",
                manifest_digest="a" * 64,
                registration_deadline=START,
                invitation_nonce="invitation-nonce-001",
                invitation_response=p.CompetitionInvitationResponse(
                    competition_id="different-competition",
                    echo_nonce="invitation-nonce-001",
                ),
            )


class MinerWiringTests(unittest.TestCase):
    def test_concrete_miner_implements_every_base_abstract_handler(self) -> None:
        base_tree = ast.parse(
            (ROOT / "vidaio_subnet_core" / "base" / "miner.py").read_text(
                encoding="utf-8"
            )
        )
        miner_tree = ast.parse(
            (ROOT / "neurons" / "miner.py").read_text(encoding="utf-8")
        )

        base_class = next(
            node
            for node in base_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BaseMiner"
        )
        miner_class = next(
            node
            for node in miner_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Miner"
        )
        abstract_names = {
            node.name
            for node in base_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
                for decorator in node.decorator_list
            )
        }
        concrete_names = {
            node.name
            for node in miner_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual(abstract_names - concrete_names, set())
        self.assertTrue(
            {
                "forward_competition_invitation",
                "forward_competition_submission",
                "blacklist_competition_invitation",
                "blacklist_competition_submission",
            }
            <= concrete_names
        )


if __name__ == "__main__":
    unittest.main()
