from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import text

from scripts.competition_dataset import (
    DatasetPipelineReceipt,
    _load_receipt,
    _prompt_yes_no,
    _require_receipt,
    _require_unchanged_sources,
    _source_snapshots,
    _write_receipt,
    ensure_manifest_registered,
    main as dataset_main,
)
from services.scoring.scoring_function import calculate_compression_score
from vidaio_subnet_core.competition.config import (
    CompetitionConfig,
    ScoringFactors,
    load_manifest,
)
from vidaio_subnet_core.competition.contracts import (
    CompetitionCompressionResponse,
    CompetitionCompressionResult,
)
from vidaio_subnet_core.competition.dataset import (
    COMPRESSION_VBR_BITRATES_BPS,
    COMPRESSION_VMAF_THRESHOLDS,
    CompressionEvaluationIndexItem,
    DatasetError,
    EvaluationIndex,
    EvaluationIndexItem,
    ModalVolumeStore,
    exclude_invalid_sources,
    local_index_validation_issues,
    prepare_index,
    prepare_index_candidates,
)
from vidaio_subnet_core.competition.execution import CompetitionExecutionCoordinator
from vidaio_subnet_core.competition.intake import boss_contender_id
from vidaio_subnet_core.competition.manager import CompetitionManager
from vidaio_subnet_core.competition.media_contracts import (
    CompetitionScoringMedia,
    CompetitionTaskAdapter,
    IndexedMediaAsset,
    UpscalingCompetitionAdapterStub,
    UpscalingEvaluationIndexItemStub,
)
from vidaio_subnet_core.competition.modal_runner import SandboxRunnerError
from vidaio_subnet_core.competition.repository import (
    AttemptOutcome,
    CompetitionRepository,
    DeferredBatchResult,
)
from vidaio_subnet_core.competition.qualification import MediaInfo, QualificationError
from vidaio_subnet_core.competition.pricing import (
    GPU_PRICE_PER_SECOND_USD,
    SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD,
    canonical_modal_gpu_type,
    estimate_sandbox_cost,
)
from vidaio_subnet_core.competition.scoring import (
    CompetitionItemScorer,
    ItemScoringError,
    ScoredItem,
    compute_aggregates,
    length_weight,
)
from vidaio_subnet_core.competition.state import CompetitionState
from vidaio_subnet_core.competition.timeouts import (
    competition_execution_lease_seconds,
)
from vidaio_subnet_core.competition.validation import (
    ValidationReason,
    ValidationReport,
    ValidationStatus,
)


ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 7, 16, 20, 0, tzinfo=timezone.utc)
TREE_SHA = "a" * 40
ALLOCATED_RESOURCES = {
    "allocated_gpu_type": "H200",
    "allocated_gpu_count": 1,
    "allocated_cpu_cores": 8.0,
}


def media_metadata(**updates):
    return {
        "duration_seconds": 10.0,
        "width": 1280,
        "height": 720,
        "frame_count": 300,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "sample_aspect_ratio": "1:1",
        **updates,
    }


def index_item(evaluation_id: str, duration: float) -> EvaluationIndexItem:
    return EvaluationIndexItem(
        evaluation_id=evaluation_id,
        source_path=f"inputs/{evaluation_id}.mp4",
        size_bytes=1000,
        sha256="b" * 64,
        duration_seconds=duration,
        width=1280,
        height=720,
        frame_count=max(1, int(duration * 30)),
        codec="h264",
        pixel_format="yuv420p",
        sample_aspect_ratio="1:1",
    )


class Phase4DatasetTests(unittest.TestCase):
    def test_prepare_assigns_one_seeded_random_query_to_each_video(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        metadata = media_metadata()
        with tempfile.TemporaryDirectory() as temp:
            for index in range(12):
                (Path(temp) / f"video-{index:02d}.mp4").write_bytes(
                    f"video-fixture-{index}".encode()
                )
            with patch(
                "vidaio_subnet_core.competition.dataset._probe_video",
                return_value=metadata,
            ):
                first = prepare_index(manifest, Path(temp))
                second = prepare_index(manifest, Path(temp))

        self.assertEqual(first, second)
        self.assertEqual(len(first.items), 12)
        self.assertEqual(len({item.source_path for item in first.items}), 12)
        assignments = first.canonical_batch_assignments(manifest)
        for item in first.items:
            batch_index, _position = assignments[item.evaluation_id]
            self.assertEqual(
                item.source_path,
                f"batches/{batch_index:06d}/{item.logical_source_path}",
            )
            self.assertEqual(
                item.sandbox_path,
                f"/evaluation-inputs/{item.logical_source_path}",
            )
        self.assertTrue(
            all(
                isinstance(item, CompressionEvaluationIndexItem) for item in first.items
            )
        )
        self.assertTrue(
            {item.vmaf_threshold for item in first.items}
            <= set(COMPRESSION_VMAF_THRESHOLDS)
        )
        self.assertEqual({item.codec_mode for item in first.items}, {"CRF", "VBR"})
        self.assertTrue(
            {item.target_bitrate for item in first.items if item.codec_mode == "VBR"}
            <= set(COMPRESSION_VBR_BITRATES_BPS)
        )
        reloaded = EvaluationIndex.model_validate_json(first.normalized_json())
        self.assertEqual(reloaded, first)
        self.assertEqual(first.schema_version, 2)
        with self.assertRaisesRegex(ValueError, "schema_version 2"):
            EvaluationIndex.model_validate(
                {**first.model_dump(mode="json"), "schema_version": 1}
            )

    def test_prepare_collects_unknown_media_and_keeps_valid_sources(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        valid_metadata = media_metadata()
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp)
            (source_root / "bad.mp4").write_bytes(b"bad")
            (source_root / "good.mp4").write_bytes(b"good")

            def probe(path, _executable):
                if Path(path).name == "bad.mp4":
                    return {
                        **valid_metadata,
                        "sample_aspect_ratio": "unknown",
                    }
                return valid_metadata

            with patch(
                "vidaio_subnet_core.competition.dataset._probe_video",
                side_effect=probe,
            ):
                candidate, issues = prepare_index_candidates(manifest, source_root)
                with self.assertRaisesRegex(
                    DatasetError, "sample_aspect_ratio must be a positive rational"
                ):
                    prepare_index(manifest, source_root)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(len(candidate.items), 1)
        self.assertEqual(
            candidate.items[0].source_path,
            "batches/000000/inputs/good.mp4",
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].source_path, "inputs/bad.mp4")

    def test_manifest_validation_rejects_unscoped_source_path(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        evaluation_index = EvaluationIndex(
            competition_id=manifest.competition_id,
            items=(index_item("unscoped", 10),),
        )

        with self.assertRaisesRegex(DatasetError, "canonical batch"):
            evaluation_index.validate_for_manifest(manifest)

    def test_local_validation_reprobes_and_excludes_all_source_variants(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp)
            bad_payload = b"bad-video"
            good_payload = b"good-video"
            (source_root / "bad.mp4").write_bytes(bad_payload)
            (source_root / "good.mp4").write_bytes(good_payload)
            common_bad = {
                "source_path": "inputs/bad.mp4",
                "size_bytes": len(bad_payload),
                "sha256": sha256(bad_payload).hexdigest(),
                **media_metadata(sample_aspect_ratio="unknown"),
            }
            bad_items = (
                CompressionEvaluationIndexItem(
                    evaluation_id="bad-crf",
                    codec_mode="CRF",
                    vmaf_threshold=85,
                    **common_bad,
                ),
                CompressionEvaluationIndexItem(
                    evaluation_id="bad-vbr",
                    codec_mode="VBR",
                    vmaf_threshold=89,
                    target_bitrate=5_000_000,
                    **common_bad,
                ),
            )
            good_item = EvaluationIndexItem(
                evaluation_id="good",
                source_path="inputs/good.mp4",
                size_bytes=len(good_payload),
                sha256=sha256(good_payload).hexdigest(),
                **media_metadata(),
            )
            evaluation_index = EvaluationIndex(
                competition_id=manifest.competition_id,
                items=(*bad_items, good_item),
            ).with_batch_scoped_source_paths(manifest)

            def probe(path, _executable):
                return media_metadata(
                    sample_aspect_ratio=(
                        "unknown" if Path(path).name == "bad.mp4" else "1:1"
                    )
                )

            with patch(
                "vidaio_subnet_core.competition.dataset._probe_video",
                side_effect=probe,
            ):
                issues = local_index_validation_issues(
                    evaluation_index, manifest, source_root
                )

        self.assertEqual(len(issues), 1)
        self.assertEqual(set(issues[0].evaluation_ids), {"bad-crf", "bad-vbr"})
        filtered = exclude_invalid_sources(evaluation_index, issues, manifest)
        self.assertEqual([item.evaluation_id for item in filtered.items], ["good"])

    def test_disqualification_prompt_requires_yes_or_no(self) -> None:
        with patch("builtins.input", side_effect=("maybe", "yes")):
            self.assertTrue(
                _prompt_yes_no("Continue? ", assume_yes=False)
            )
        with patch("builtins.input", return_value="no"):
            self.assertFalse(
                _prompt_yes_no("Continue? ", assume_yes=False)
            )

    def test_duration_error_lists_every_source_path_and_measured_value(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        minimum = manifest.min_video_length.total_seconds()
        maximum = manifest.max_video_length.total_seconds()
        evaluation_index = EvaluationIndex(
            competition_id=manifest.competition_id,
            items=(
                index_item("too-short", max(0.001, minimum / 2)).model_copy(
                    update={"source_path": "inputs/nested/short-clip.mp4"}
                ),
                index_item("too-long", maximum + 1).model_copy(
                    update={"source_path": "inputs/long-clip.mp4"}
                ),
            ),
        ).with_batch_scoped_source_paths(manifest)
        with self.assertRaises(DatasetError) as captured:
            evaluation_index.validate_for_manifest(manifest)
        message = str(captured.exception)
        self.assertIn("2 video(s)", message)
        self.assertIn("inputs/nested/short-clip.mp4", message)
        self.assertIn("inputs/long-clip.mp4", message)
        self.assertIn(f"duration={maximum + 1:.3f}s", message)

    def test_dataset_cli_executes_directly_outside_repository_cwd(self) -> None:
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/competition_dataset.py"),
                    "--help",
                ],
                cwd=temp,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Prepare, validate, upload", result.stdout)

    def test_pipeline_receipt_enforces_order_and_source_identity(self) -> None:
        manifest = SimpleNamespace(
            competition_id="pipeline-receipt-test",
            evaluation_batch_size=5,
            digest=lambda: "a" * 64,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "sources"
            source_root.mkdir()
            payload = b"video-source"
            (source_root / "one.mp4").write_bytes(payload)
            item = EvaluationIndexItem(
                evaluation_id="one",
                source_path="inputs/one.mp4",
                size_bytes=len(payload),
                sha256=sha256(payload).hexdigest(),
                **media_metadata(),
            )
            evaluation_index = EvaluationIndex(
                competition_id=manifest.competition_id,
                items=(item,),
            ).with_batch_scoped_source_paths(manifest)
            index_path = root / "index.json"
            index_path.write_text(
                evaluation_index.normalized_json(),
                encoding="utf-8",
            )
            receipt = DatasetPipelineReceipt(
                stage="prepared",
                manifest_digest=manifest.digest(),
                index_digest=evaluation_index.digest(),
                source_root=str(source_root.resolve()),
                sources=_source_snapshots(evaluation_index, source_root),
                updated_at=NOW,
            )
            _write_receipt(str(index_path), receipt)

            loaded = _require_receipt(
                str(index_path),
                manifest=manifest,
                index=evaluation_index,
                allowed_stages=("prepared",),
                required_stage="prepared",
            )
            _require_unchanged_sources(loaded, evaluation_index, source_root)
            with self.assertRaisesRegex(DatasetError, "required prior stage"):
                _require_receipt(
                    str(index_path),
                    manifest=manifest,
                    index=evaluation_index,
                    allowed_stages=("validated",),
                    required_stage="validated",
                )

            (source_root / "one.mp4").write_bytes(b"changed-source")
            with self.assertRaisesRegex(DatasetError, "changed after preparation"):
                _require_unchanged_sources(loaded, evaluation_index, source_root)

    def test_dataset_cli_enforces_pipeline_without_repeating_media_probe(self) -> None:
        manifest = SimpleNamespace(
            competition_id="pipeline-cli-test",
            scoring_seed=123,
            min_video_length=timedelta(seconds=1),
            max_video_length=timedelta(minutes=1),
            evaluation_batch_size=5,
            evaluation_input_volume_name="pipeline-cli-inputs",
            evaluation_index_path="/validator-evaluation/index.json",
            digest=lambda: "c" * 64,
        )
        state = {"uploaded": False, "sealed": False}

        class Store:
            def __init__(self, *, environment_name):
                self.environment_name = environment_name

            def upload(
                self,
                _manifest,
                _index,
                _source_root,
                *,
                validate_local,
            ):
                assert validate_local is False
                state["uploaded"] = True

            def load_index(self, _manifest, *, validate_for_manifest):
                assert validate_for_manifest is False
                return EvaluationIndex.model_validate_json(
                    Path(state["index_path"]).read_text(encoding="utf-8")
                )

            def validation_issues(self, *_args, **_kwargs):
                raise AssertionError("seal repeated remote media validation")

        class Repository:
            def __init__(self, _database_url):
                pass

            def get(self, _competition_id):
                return SimpleNamespace(manifest_digest=manifest.digest())

            def seal_evaluation_dataset(
                self,
                _competition_id,
                evaluation_index,
                *,
                now,
                actor,
            ):
                assert now is not None
                assert actor == "competition-dataset-cli"
                state["sealed"] = True
                return evaluation_index.digest()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "sources"
            source_root.mkdir()
            (source_root / "one.mp4").write_bytes(b"source-video")
            index_path = root / "index.json"
            state["index_path"] = str(index_path)
            common = [
                "--manifest",
                str(root / "manifest.json"),
                "--source-dir",
                str(source_root),
                "--index",
                str(index_path),
            ]

            with (
                patch("scripts.competition_dataset.load_manifest", return_value=manifest),
                patch(
                    "vidaio_subnet_core.competition.dataset._probe_video",
                    return_value=media_metadata(),
                ) as probe,
                patch("scripts.competition_dataset.ModalVolumeStore", Store),
                patch("scripts.competition_dataset.CompetitionRepository", Repository),
            ):
                with patch.object(
                    sys,
                    "argv",
                    ["competition_dataset.py", "prepare", *common],
                ):
                    self.assertEqual(dataset_main(), 0)
                self.assertEqual(probe.call_count, 1)
                self.assertEqual(_load_receipt(str(index_path)).stage, "prepared")

                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "competition_dataset.py",
                            "upload",
                            *common,
                            "--environment",
                            "main",
                        ],
                    ),
                    self.assertRaisesRegex(DatasetError, "required prior stage"),
                ):
                    dataset_main()

                with patch.object(
                    sys,
                    "argv",
                    ["competition_dataset.py", "validate", *common],
                ):
                    self.assertEqual(dataset_main(), 0)
                self.assertEqual(probe.call_count, 1)
                self.assertEqual(_load_receipt(str(index_path)).stage, "validated")

                with patch.object(
                    sys,
                    "argv",
                    [
                        "competition_dataset.py",
                        "upload",
                        *common,
                        "--environment",
                        "main",
                    ],
                ):
                    self.assertEqual(dataset_main(), 0)
                self.assertTrue(state["uploaded"])
                self.assertEqual(_load_receipt(str(index_path)).stage, "uploaded")

                with patch.object(
                    sys,
                    "argv",
                    [
                        "competition_dataset.py",
                        "seal",
                        "--manifest",
                        str(root / "manifest.json"),
                        "--index",
                        str(index_path),
                        "--environment",
                        "main",
                        "--database-url",
                        f"sqlite:///{root / 'competition.db'}",
                    ],
                ):
                    self.assertEqual(dataset_main(), 0)
                self.assertTrue(state["sealed"])
                self.assertEqual(_load_receipt(str(index_path)).stage, "sealed")
                self.assertEqual(probe.call_count, 1)

    def test_seal_preflight_can_register_manifest_before_validator_boot(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        with tempfile.TemporaryDirectory() as temp:
            repository = CompetitionRepository(f"sqlite:///{Path(temp) / 'preboot.db'}")
            self.assertTrue(
                ensure_manifest_registered(
                    repository,
                    manifest,
                    now=NOW,
                    actor="dataset-cli",
                )
            )
            registered = repository.get(manifest.competition_id)
            self.assertEqual(registered.status, CompetitionState.SCHEDULED.value)
            self.assertEqual(registered.manifest_digest, manifest.digest())
            self.assertFalse(
                ensure_manifest_registered(
                    repository,
                    manifest,
                    now=NOW,
                    actor="dataset-cli",
                )
            )
            with self.assertRaisesRegex(RuntimeError, "different manifest revision"):
                ensure_manifest_registered(
                    repository,
                    manifest.model_copy(
                        update={"scoring_seed": manifest.scoring_seed + 1}
                    ),
                    now=NOW,
                    actor="dataset-cli",
                )
            repository.engine.dispose()

    def test_upload_reads_back_every_source_and_refuses_index_replacement(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )

        class Batch:
            def __init__(self, volume):
                self.volume = volume

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def put_file(self, local_path, remote_path):
                self.volume.files[str(remote_path).lstrip("/")] = Path(
                    local_path
                ).read_bytes()

        class Volume:
            files = {}

            @classmethod
            def from_name(cls, *_args, **_kwargs):
                return cls

            @classmethod
            def read_file(cls, path):
                if path not in cls.files:
                    raise FileNotFoundError(path)
                return iter((cls.files[path],))

            @classmethod
            def batch_upload(cls, *, force):
                assert force
                return Batch(cls)

        class Environment:
            @classmethod
            def from_name(cls, _name, *, create_if_missing=False):
                assert create_if_missing
                return cls

            @classmethod
            def hydrate(cls):
                return cls

        modal_api = SimpleNamespace(Environment=Environment, Volume=Volume)
        store = ModalVolumeStore(environment_name="main", modal_api=modal_api)
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp)
            values = {"a.mp4": b"a" * 11, "b.mp4": b"b" * 17}
            items = []
            for name, payload in values.items():
                (source_root / name).write_bytes(payload)
                items.append(
                    EvaluationIndexItem(
                        evaluation_id=name.removesuffix(".mp4"),
                        source_path=f"inputs/{name}",
                        size_bytes=len(payload),
                        sha256=sha256(payload).hexdigest(),
                        duration_seconds=10,
                        width=1280,
                        height=720,
                        frame_count=300,
                        codec="h264",
                        pixel_format="yuv420p",
                        sample_aspect_ratio="1:1",
                    )
                )
            evaluation_index = EvaluationIndex(
                competition_id=manifest.competition_id, items=tuple(items)
            ).with_batch_scoped_source_paths(manifest)
            metadata = media_metadata()
            with patch(
                "vidaio_subnet_core.competition.dataset._probe_video",
                return_value=metadata,
            ):
                store.upload(manifest, evaluation_index, source_root)
            with patch(
                "vidaio_subnet_core.competition.dataset.validate_local_index",
                side_effect=AssertionError("local validation was repeated"),
            ):
                store.upload(
                    manifest,
                    evaluation_index,
                    source_root,
                    validate_local=False,
                )
            self.assertEqual(
                Volume.files["batches/000000/inputs/a.mp4"], values["a.mp4"]
            )
            self.assertEqual(
                Volume.files["batches/000000/inputs/b.mp4"], values["b.mp4"]
            )
            self.assertFalse(
                any(path.startswith("inputs/") for path in Volume.files)
            )

            narrowed = EvaluationIndex(
                competition_id=manifest.competition_id,
                items=(items[0],),
            ).with_batch_scoped_source_paths(manifest)
            with (
                patch(
                    "vidaio_subnet_core.competition.dataset._probe_video",
                    return_value=metadata,
                ),
                self.assertRaisesRegex(DatasetError, "dataset is immutable"),
            ):
                store.upload(manifest, narrowed, source_root)
            self.assertEqual(store.load_index(manifest), evaluation_index)

            replaced = EvaluationIndex(
                competition_id=manifest.competition_id,
                items=tuple(reversed(items)),
            ).with_batch_scoped_source_paths(manifest)
            with (
                patch(
                    "vidaio_subnet_core.competition.dataset._probe_video",
                    return_value=metadata,
                ),
                self.assertRaisesRegex(DatasetError, "different index"),
            ):
                store.upload(manifest, replaced, source_root)

    def test_upload_creates_missing_modal_environment_and_volume(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        state = {
            "environment_exists": False,
            "environment_lookups": [],
            "volume_exists": False,
            "files": {},
            "volume_lookups": [],
        }

        class NotFoundError(Exception):
            pass

        class EnvironmentReference:
            def __init__(self, create_if_missing):
                self.create_if_missing = create_if_missing

            def hydrate(self):
                if not state["environment_exists"]:
                    if not self.create_if_missing:
                        raise NotFoundError("environment not found")
                    state["environment_exists"] = True
                return self

        class Environment:
            @classmethod
            def from_name(cls, name, *, create_if_missing=False):
                state["environment_lookups"].append((name, create_if_missing))
                return EnvironmentReference(create_if_missing)

        class Batch:
            def __init__(self, volume):
                self.volume = volume

            def __enter__(self):
                if not state["environment_exists"]:
                    raise NotFoundError("environment not found")
                if not state["volume_exists"]:
                    if not self.volume.create_if_missing:
                        raise NotFoundError("volume not found")
                    state["volume_exists"] = True
                return self

            def __exit__(self, *_args):
                return False

            def put_file(self, local_path, remote_path):
                state["files"][str(remote_path).lstrip("/")] = Path(
                    local_path
                ).read_bytes()

        class VolumeReference:
            def __init__(self, create_if_missing):
                self.create_if_missing = create_if_missing

            def read_file(self, path):
                if not state["environment_exists"]:
                    raise NotFoundError("environment not found")
                if not state["volume_exists"]:
                    raise NotFoundError("volume not found")
                if path not in state["files"]:
                    raise FileNotFoundError(path)
                return iter((state["files"][path],))

            def batch_upload(self, *, force):
                assert force
                return Batch(self)

        class Volume:
            @classmethod
            def from_name(cls, name, *, environment_name, create_if_missing=False):
                state["volume_lookups"].append(
                    (name, environment_name, create_if_missing)
                )
                return VolumeReference(create_if_missing)

        modal_api = SimpleNamespace(
            Environment=Environment,
            Volume=Volume,
            exception=SimpleNamespace(NotFoundError=NotFoundError),
        )
        store = ModalVolumeStore(environment_name="main", modal_api=modal_api)
        payload = b"video"
        item = EvaluationIndexItem(
            evaluation_id="one",
            source_path="inputs/one.mp4",
            size_bytes=len(payload),
            sha256=sha256(payload).hexdigest(),
            duration_seconds=10,
            width=1280,
            height=720,
            frame_count=300,
            codec="h264",
            pixel_format="yuv420p",
            sample_aspect_ratio="1:1",
        )
        evaluation_index = EvaluationIndex(
            competition_id=manifest.competition_id,
            items=(item,),
        ).with_batch_scoped_source_paths(manifest)

        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp)
            (source_root / "one.mp4").write_bytes(payload)
            metadata = media_metadata()
            with (
                patch("vidaio_subnet_core.competition.dataset.logger") as log,
                patch(
                    "vidaio_subnet_core.competition.dataset._probe_video",
                    return_value=metadata,
                ),
            ):
                store.upload(manifest, evaluation_index, source_root)

        self.assertTrue(state["environment_exists"])
        self.assertIn(("main", True), state["environment_lookups"])
        self.assertTrue(state["volume_exists"])
        self.assertIn(
            (manifest.evaluation_input_volume_name, "main", True),
            state["volume_lookups"],
        )
        log.info.assert_any_call(
            "Modal Volume '{}' was not found in environment '{}'; "
            "creating it before dataset upload",
            manifest.evaluation_input_volume_name,
            "main",
        )
        log.info.assert_any_call(
            "Uploaded evaluation dataset to Modal Volume '{}' in environment "
            "'{}' (created automatically if it was missing)",
            manifest.evaluation_input_volume_name,
            "main",
        )
        self.assertEqual(state["files"]["batches/000000/inputs/one.mp4"], payload)
        self.assertNotIn(
            "batches/000000/validator-evaluation/index.json", state["files"]
        )


class Phase4FutureUpscalingContractTests(unittest.TestCase):
    @staticmethod
    def asset(path: str, width: int, height: int) -> IndexedMediaAsset:
        return IndexedMediaAsset(
            relative_path=path,
            size_bytes=100,
            sha256=sha256(path.encode()).hexdigest(),
            duration_seconds=10,
            width=width,
            height=height,
            frame_count=300,
            codec="h264",
            pixel_format="yuv420p",
            sample_aspect_ratio="1:1",
        )

    def test_upscaling_stub_keeps_ground_truth_reference_and_output_distinct(
        self,
    ) -> None:
        item = UpscalingEvaluationIndexItemStub(
            evaluation_id="upscale-one",
            ground_truth=self.asset("ground-truth/one.mp4", 1920, 1080),
            reference=self.asset("references/one.mp4", 960, 540),
        )
        self.assertEqual(
            item.contender_input_path,
            "/evaluation-inputs/references/one.mp4",
        )
        self.assertEqual(
            item.trusted_quality_reference_path,
            "/evaluation-ground-truth/ground-truth/one.mp4",
        )

        media = CompetitionScoringMedia.upscaling(
            ground_truth_video=b"ground-truth",
            downsampled_reference_video=b"downsampled-reference",
            miner_processed_video=b"miner-output",
        )
        self.assertEqual(media.ground_truth_video, b"ground-truth")
        self.assertEqual(media.reference_video, b"downsampled-reference")
        self.assertEqual(media.miner_processed_video, b"miner-output")
        with self.assertRaisesRegex(NotImplementedError, "reserved"):
            UpscalingCompetitionAdapterStub().score_media(
                None,
                item,
                media,
                runtime_seconds=1,
                **ALLOCATED_RESOURCES,
            )


class Phase4RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database_url = f"sqlite:///{Path(self.temp.name) / 'competition.db'}"
        self.repository = CompetitionRepository(self.database_url)
        self.manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        ).model_copy(update={"max_attempts_per_item": 2})
        self.competition_id = self.manifest.competition_id
        self.hotkey = "contender-a"
        self.repository.insert_manifest(self.manifest, now=NOW, actor="test")
        self.add_contender(self.hotkey)

    def add_contender(self, hotkey: str) -> None:
        validation = ValidationReport(
            ValidationStatus.ACCEPTED,
            ValidationReason.ACCEPTED,
            "c" * 64,
            1,
            100,
            (),
        )
        self.repository.record_pinned_contender(
            competition_id=self.competition_id,
            hotkey=hotkey,
            repository_url_hash="d" * 64,
            repository_display="github.com/test/contender",
            pinned_commit_sha="e" * 40,
            pinned_tree_sha=TREE_SHA,
            latest_commit_time=NOW.isoformat(),
            validation=validation,
            now=NOW,
            actor="test",
        )
        self.repository.record_build_evidence(
            competition_id=self.competition_id,
            hotkey=hotkey,
            image_id="im-test",
            image_digest="sha256:" + "f" * 64,
            image_size_bytes=0,
            evidence={"builder_id": "test"},
            build_status="MODAL_ACCEPTED",
            now=NOW,
            actor="test",
        )

    def tearDown(self) -> None:
        self.repository.engine.dispose()
        self.temp.cleanup()

    def seal(self, *items: EvaluationIndexItem) -> EvaluationIndex:
        evaluation_index = EvaluationIndex(
            competition_id=self.competition_id, items=items
        ).with_batch_scoped_source_paths(self.manifest)
        self.repository.seal_evaluation_dataset(
            self.competition_id,
            evaluation_index,
            now=NOW,
            actor="test",
        )
        return evaluation_index

    def claim(
        self,
        now: datetime,
        max_items: int = 5,
        lease_seconds: int | None = 60,
        minimum_execution_timeout_seconds: float = 0,
        hotkey: str | None = None,
    ):
        return self.repository.claim_evaluation_batch(
            self.competition_id,
            hotkey or self.hotkey,
            owner="worker-a",
            max_items=max_items,
            max_attempts=self.manifest.max_attempts_per_item,
            lease_seconds=lease_seconds,
            minimum_execution_timeout_seconds=minimum_execution_timeout_seconds,
            scoring_version=self.manifest.scoring_version,
            vmaf_threshold=self.manifest.vmaf_threshold,
            max_video_length_seconds=self.manifest.max_video_length.total_seconds(),
            length_weight_exponent=self.manifest.length_weight_exponent,
            now=now,
        )

    def test_completion_persists_deterministic_podium_and_reward_ladder(self):
        validation = ValidationReport(
            ValidationStatus.ACCEPTED,
            ValidationReason.ACCEPTED,
            "c" * 64,
            1,
            100,
            (),
        )
        winner_hotkey = "winner"
        boss_id = boss_contender_id(winner_hotkey)
        contenders = (
            (winner_hotkey, 0.94, NOW, "1" * 40, 11, False),
            (boss_id, 0.95, NOW - timedelta(days=3), "5" * 40, 11, True),
            ("runner", 0.90, NOW - timedelta(days=2), "3" * 40, 12, False),
            ("third", 0.90, NOW - timedelta(days=1), "2" * 40, 13, False),
            ("fourth", 0.80, NOW, "4" * 40, 14, False),
        )
        for hotkey, _score, commit_time, commit_sha, uid, is_boss in contenders:
            self.repository.record_pinned_contender(
                competition_id=self.competition_id,
                hotkey=hotkey,
                repository_url_hash=hotkey[0] * 64,
                repository_display=f"github.com/test/{hotkey}",
                pinned_commit_sha=commit_sha,
                pinned_tree_sha=TREE_SHA,
                latest_commit_time=commit_time.isoformat(),
                validation=validation,
                now=NOW,
                actor="test",
                uid_snapshot=uid,
                is_boss=is_boss,
            )
        with self.repository.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE competitions SET boss_hotkey=:boss_hotkey "
                    "WHERE competition_id=:competition_id"
                ),
                {
                    "boss_hotkey": winner_hotkey,
                    "competition_id": self.competition_id,
                },
            )
            for hotkey, score, *_rest in contenders:
                connection.execute(
                    text(
                        "UPDATE contender_metadata SET status='SCORED', "
                        "final_score=:score WHERE competition_id=:competition_id "
                        "AND hotkey=:hotkey"
                    ),
                    {
                        "competition_id": self.competition_id,
                        "hotkey": hotkey,
                        "score": score,
                    },
                )

        current = CompetitionState.SCHEDULED
        for target in (
            CompetitionState.ENROLLING,
            CompetitionState.FINALIZING_SUBMISSIONS,
            CompetitionState.VALIDATING,
            CompetitionState.BUILDING,
            CompetitionState.EVALUATING,
            CompetitionState.SCORING,
            CompetitionState.AWAITING_END_TIME,
            CompetitionState.COMPLETED,
        ):
            self.repository.transition(
                self.competition_id,
                target,
                expected=current,
                now=NOW,
                actor="test",
            )
            current = target

        ranked = {
            row.hotkey: row.final_rank
            for row in self.repository.list_contenders(self.competition_id)
            if row.final_rank is not None
        }
        self.assertEqual(
            ranked,
            {boss_id: 1, "runner": 2, "third": 3, "fourth": 4},
        )
        submitted_winner = self.repository.get_contender(
            self.competition_id, winner_hotkey
        )
        self.assertIsNone(submitted_winner.final_rank)
        self.assertFalse(submitted_winner.eligible)
        self.assertEqual(
            submitted_winner.reason_code,
            "LOWER_SCORING_SOLUTION_FOR_HOTKEY",
        )
        competition = self.repository.get(self.competition_id)
        self.assertEqual(competition.winner_hotkey, winner_hotkey)
        self.assertEqual(competition.winner_uid_at_finalisation, 11)
        self.assertEqual(
            self.repository.latest_competition_reward_recipients(),
            (
                (winner_hotkey, 1, 0.70),
                ("runner", 2, 0.20),
                ("third", 3, 0.10),
            ),
        )
        completion_payload = json.loads(
            self.repository.list_events(self.competition_id)[-1].payload_json
        )
        self.assertEqual(completion_payload["podium"][0]["contender_id"], boss_id)
        self.assertTrue(completion_payload["podium"][0]["is_boss"])

    def test_submitted_solution_beats_boss_solution_for_same_hotkey(self):
        validation = ValidationReport(
            ValidationStatus.ACCEPTED,
            ValidationReason.ACCEPTED,
            "c" * 64,
            1,
            100,
            (),
        )
        payout_hotkey = "boss-owner"
        boss_id = boss_contender_id(payout_hotkey)
        for contender_id, score, is_boss in (
            (payout_hotkey, 0.96, False),
            (boss_id, 0.95, True),
        ):
            self.repository.record_pinned_contender(
                competition_id=self.competition_id,
                hotkey=contender_id,
                repository_url_hash=("a" if is_boss else "b") * 64,
                repository_display=f"test:{contender_id}",
                pinned_commit_sha=("6" if is_boss else "7") * 40,
                pinned_tree_sha=TREE_SHA,
                latest_commit_time=NOW.isoformat(),
                validation=validation,
                now=NOW,
                actor="test",
                uid_snapshot=21,
                is_boss=is_boss,
            )
            with self.repository.engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE contender_metadata SET status='SCORED', "
                        "final_score=:score WHERE competition_id=:competition_id "
                        "AND hotkey=:hotkey"
                    ),
                    {
                        "competition_id": self.competition_id,
                        "hotkey": contender_id,
                        "score": score,
                    },
                )
        with self.repository.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE competitions SET boss_hotkey=:boss_hotkey "
                    "WHERE competition_id=:competition_id"
                ),
                {
                    "boss_hotkey": payout_hotkey,
                    "competition_id": self.competition_id,
                },
            )

        current = CompetitionState.SCHEDULED
        for target in (
            CompetitionState.ENROLLING,
            CompetitionState.FINALIZING_SUBMISSIONS,
            CompetitionState.VALIDATING,
            CompetitionState.BUILDING,
            CompetitionState.EVALUATING,
            CompetitionState.SCORING,
            CompetitionState.AWAITING_END_TIME,
            CompetitionState.COMPLETED,
        ):
            self.repository.transition(
                self.competition_id,
                target,
                expected=current,
                now=NOW,
                actor="test",
            )
            current = target

        submitted = self.repository.get_contender(self.competition_id, payout_hotkey)
        boss = self.repository.get_contender(self.competition_id, boss_id)
        self.assertEqual(submitted.final_rank, 1)
        self.assertTrue(submitted.eligible)
        self.assertIsNone(boss.final_rank)
        self.assertFalse(boss.eligible)
        self.assertEqual(
            self.repository.latest_competition_reward_recipients(),
            ((payout_hotkey, 1, 0.70),),
        )

    def test_latest_completed_podium_atomically_replaces_incumbent(self):
        def complete(competition_id: str, hotkey: str, when: datetime) -> None:
            with self.repository.engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE contender_metadata SET status='SCORED', "
                        "final_score=0.9 WHERE competition_id=:competition_id "
                        "AND hotkey=:hotkey"
                    ),
                    {
                        "competition_id": competition_id,
                        "hotkey": hotkey,
                    },
                )
            current = CompetitionState.SCHEDULED
            for target in (
                CompetitionState.ENROLLING,
                CompetitionState.FINALIZING_SUBMISSIONS,
                CompetitionState.VALIDATING,
                CompetitionState.BUILDING,
                CompetitionState.EVALUATING,
                CompetitionState.SCORING,
                CompetitionState.AWAITING_END_TIME,
                CompetitionState.COMPLETED,
            ):
                self.repository.transition(
                    competition_id,
                    target,
                    expected=current,
                    now=when,
                    actor="test",
                )
                current = target

        complete(self.competition_id, self.hotkey, NOW)
        self.assertEqual(
            self.repository.latest_competition_reward_context(),
            (self.competition_id, ((self.hotkey, 1, 0.70),)),
        )

        second_manifest = self.manifest.model_copy(
            update={
                "competition_id": "compression-2026-w30",
                "competition_start_time": (
                    self.manifest.competition_start_time + timedelta(days=7)
                ),
                "contender_finalisation_time": (
                    self.manifest.contender_finalisation_time + timedelta(days=7)
                ),
                "human_review_deadline": (
                    self.manifest.human_review_deadline + timedelta(days=7)
                ),
                "competition_end_time": (
                    self.manifest.competition_end_time + timedelta(days=7)
                ),
            }
        )
        second_hotkey = "new-winner"
        second_when = NOW + timedelta(days=7)
        self.repository.insert_manifest(second_manifest, now=second_when, actor="test")
        validation = ValidationReport(
            ValidationStatus.ACCEPTED,
            ValidationReason.ACCEPTED,
            "c" * 64,
            1,
            100,
            (),
        )
        self.repository.record_pinned_contender(
            competition_id=second_manifest.competition_id,
            hotkey=second_hotkey,
            repository_url_hash="9" * 64,
            repository_display="github.com/test/new-winner",
            pinned_commit_sha="8" * 40,
            pinned_tree_sha="7" * 40,
            latest_commit_time=second_when.isoformat(),
            validation=validation,
            now=second_when,
            actor="test",
        )
        complete(second_manifest.competition_id, second_hotkey, second_when)

        self.assertEqual(
            self.repository.latest_competition_reward_context(),
            (
                second_manifest.competition_id,
                ((second_hotkey, 1, 0.70),),
            ),
        )
        self.assertEqual(
            self.repository.get(self.competition_id).winner_hotkey,
            self.hotkey,
        )

    def test_seal_is_immutable_and_successful_items_are_exactly_once(self) -> None:
        evaluation_index = self.seal(index_item("one", 10), index_item("two", 20))
        self.assertEqual(
            self.repository.get(self.competition_id).dataset_index_checksum,
            evaluation_index.digest(),
        )
        self.assertFalse(
            self.repository.contender_evaluation_is_complete(
                self.competition_id, self.hotkey
            )
        )
        with self.assertRaises(ValueError):
            self.repository.seal_evaluation_dataset(
                self.competition_id,
                EvaluationIndex(
                    competition_id=self.competition_id,
                    items=(index_item("different", 10),),
                ),
                now=NOW,
                actor="test",
            )

        batch = self.claim(NOW)
        self.assertEqual(len(batch.evaluations), 2)
        outcomes = tuple(
            AttemptOutcome(
                evaluation.history_id,
                "SCORED",
                metrics=ScoredItem(
                    input_checksum=evaluation.item.sha256,
                    output_checksum="1" * 64,
                    input_size_bytes=1000,
                    output_size_bytes=400,
                    vmaf_score=95,
                    compression_ratio=2.5,
                    runtime_seconds=2,
                    estimated_cost_usd=Decimal("0.01"),
                ),
                processing_started_at=NOW,
                processing_finished_at=NOW + timedelta(seconds=2),
            )
            for evaluation in batch.evaluations
        )
        self.repository.record_batch_outcomes(
            self.competition_id,
            self.hotkey,
            batch.batch_id,
            outcomes,
            max_attempts=2,
            modal_sandbox_id="sb-test",
            wall_runtime_seconds=2.2,
            now=NOW + timedelta(seconds=3),
            actor="test",
        )
        self.assertIsNone(self.claim(NOW + timedelta(seconds=4)))
        self.assertTrue(
            self.repository.evaluation_is_complete(
                self.competition_id, frozenset({"MODAL_ACCEPTED"})
            )
        )
        self.assertTrue(
            self.repository.contender_evaluation_is_complete(
                self.competition_id, self.hotkey
            )
        )

        restarted = CompetitionRepository(self.database_url)
        self.assertIsNone(
            restarted.claim_evaluation_batch(
                self.competition_id,
                self.hotkey,
                owner="worker-after-restart",
                max_items=5,
                max_attempts=2,
                lease_seconds=60,
                scoring_version=self.manifest.scoring_version,
                vmaf_threshold=self.manifest.vmaf_threshold,
                max_video_length_seconds=(
                    self.manifest.max_video_length.total_seconds()
                ),
                length_weight_exponent=self.manifest.length_weight_exponent,
                now=NOW + timedelta(minutes=5),
            )
        )
        restarted.score_competition(
            self.competition_id,
            self.manifest,
            now=NOW + timedelta(minutes=5),
            actor="test",
        )
        contender = restarted.get_contender(self.competition_id, self.hotkey)
        self.assertEqual(contender.successful_items, 2)
        self.assertEqual(contender.failed_items, 0)
        media_score = calculate_compression_score(
            vmaf_score=95,
            compression_rate=0.4,
            vmaf_threshold=self.manifest.vmaf_threshold,
        )[0]
        self.assertAlmostEqual(
            contender.final_score,
            round(
                0.6 * media_score + 0.25 * 1 + 0.15 * 1,
                self.manifest.score_precision,
            ),
        )
        restarted.engine.dispose()

    def test_claim_batch_uses_distinct_source_videos(self) -> None:
        common = {
            "source_path": "inputs/shared.mp4",
            "size_bytes": 1000,
            "sha256": "b" * 64,
            "duration_seconds": 10,
            "width": 1280,
            "height": 720,
            "frame_count": 300,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "sample_aspect_ratio": "1:1",
        }
        shared_queries = tuple(
            CompressionEvaluationIndexItem(
                evaluation_id=f"shared-crf-vmaf{int(threshold)}",
                codec_mode="CRF",
                vmaf_threshold=threshold,
                **common,
            )
            for threshold in COMPRESSION_VMAF_THRESHOLDS
        )
        distinct_items = tuple(
            index_item(f"distinct-{index}", 10) for index in range(4)
        )
        self.seal(*shared_queries, *distinct_items)

        batch = self.claim(NOW)

        self.assertEqual(len(batch.evaluations), 5)
        source_paths = [
            evaluation.item.source_path for evaluation in batch.evaluations
        ]
        self.assertEqual(len(source_paths), len(set(source_paths)))

    def test_sealed_canonical_batches_are_identical_for_every_contender(self) -> None:
        items = tuple(
            index_item(f"item-{index:02d}", 5 + index)
            for index in range(12)
        )
        self.seal(*items)
        self.add_contender("contender-b")

        persisted = [
            (
                row.evaluation_id,
                row.canonical_batch_index,
                row.canonical_batch_position,
            )
            for row in self.repository.evaluation_items(self.competition_id)
        ]
        self.assertEqual(
            persisted,
            [
                (f"item-{index:02d}", index // 5, index % 5)
                for index in range(12)
            ],
        )

        first = self.claim(NOW)
        second = self.repository.claim_evaluation_batch(
            self.competition_id,
            "contender-b",
            owner="worker-b",
            max_items=self.manifest.evaluation_batch_size,
            max_attempts=self.manifest.max_attempts_per_item,
            lease_seconds=60,
            scoring_version=self.manifest.scoring_version,
            vmaf_threshold=self.manifest.vmaf_threshold,
            max_video_length_seconds=self.manifest.max_video_length.total_seconds(),
            length_weight_exponent=self.manifest.length_weight_exponent,
            now=NOW + timedelta(seconds=1),
        )

        self.assertNotEqual(first.batch_id, second.batch_id)
        self.assertEqual(
            [value.item.evaluation_id for value in first.evaluations],
            [value.item.evaluation_id for value in second.evaluations],
        )
        self.assertEqual(len(first.evaluations), 5)
        with self.repository.engine.connect() as connection:
            dispatches = connection.execute(
                text(
                    "SELECT hotkey, batch_id, canonical_batch_index "
                    "FROM competition_batches ORDER BY hotkey"
                )
            ).all()
            histories = connection.execute(
                text(
                    "SELECT hotkey, COUNT(*), "
                    "COUNT(DISTINCT canonical_batch_index), "
                    "MIN(canonical_batch_index), MAX(canonical_batch_index) "
                    "FROM contender_performance_history GROUP BY hotkey "
                    "ORDER BY hotkey"
                )
            ).all()
        self.assertEqual(
            dispatches,
            [
                ("contender-a", first.batch_id, 0),
                ("contender-b", second.batch_id, 0),
            ],
        )
        self.assertEqual(
            histories,
            [
                ("contender-a", 5, 1, 0, 0),
                ("contender-b", 5, 1, 0, 0),
            ],
        )

    def test_outcomes_reject_a_mismatched_canonical_batch_index(self) -> None:
        self.seal(index_item("item-00", 10))
        claimed = self.claim(NOW)
        with self.repository.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE contender_performance_history "
                    "SET canonical_batch_index=1 WHERE id=:history_id"
                ),
                {"history_id": claimed.evaluations[0].history_id},
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "performance history canonical batch does not match",
        ):
            self.repository.record_batch_outcomes(
                self.competition_id,
                self.hotkey,
                claimed.batch_id,
                (
                    AttemptOutcome(
                        claimed.evaluations[0].history_id,
                        "FAILED",
                        reason_code="OUTPUT_MISSING",
                    ),
                ),
                max_attempts=2,
                modal_sandbox_id="sb-test",
                wall_runtime_seconds=1,
                now=NOW + timedelta(seconds=1),
                actor="test",
            )

    def test_sealed_canonical_batch_size_is_immutable(self) -> None:
        self.seal(index_item("item-00", 10))

        with self.assertRaisesRegex(
            ValueError, "cannot change evaluation_batch_size"
        ):
            self.repository.insert_manifest(
                self.manifest.model_copy(update={"evaluation_batch_size": 4}),
                now=NOW + timedelta(seconds=1),
                actor="test",
            )

    def test_failed_item_is_not_retried_before_the_next_canonical_batch(self) -> None:
        self.seal(
            *(index_item(f"item-{index:02d}", 10 + index) for index in range(7))
        )
        first = self.claim(NOW)
        self.assertEqual(
            [value.item.evaluation_id for value in first.evaluations],
            [f"item-{index:02d}" for index in range(5)],
        )

        def scored(evaluation) -> AttemptOutcome:
            return AttemptOutcome(
                evaluation.history_id,
                "SCORED",
                metrics=ScoredItem(
                    input_checksum=evaluation.item.sha256,
                    output_checksum="1" * 64,
                    input_size_bytes=1000,
                    output_size_bytes=400,
                    vmaf_score=95,
                    compression_ratio=2.5,
                    runtime_seconds=1,
                    estimated_cost_usd=Decimal("0.001"),
                ),
            )

        self.repository.record_batch_outcomes(
            self.competition_id,
            self.hotkey,
            first.batch_id,
            (
                AttemptOutcome(
                    first.evaluations[0].history_id,
                    "FAILED",
                    reason_code="BATCH_EXECUTION_FAILED",
                ),
                *(scored(evaluation) for evaluation in first.evaluations[1:]),
            ),
            max_attempts=2,
            modal_sandbox_id="sb-test",
            wall_runtime_seconds=5,
            now=NOW + timedelta(seconds=5),
            actor="test",
        )

        second_batch = self.claim(NOW + timedelta(seconds=6))
        self.assertEqual(
            [value.item.evaluation_id for value in second_batch.evaluations],
            ["item-05", "item-06"],
        )
        with self.repository.engine.connect() as connection:
            failed = connection.execute(
                text(
                    "SELECT attempt, status, reason_code "
                    "FROM contender_performance_history "
                    "WHERE evaluation_id='item-00'"
                )
            ).one()
        self.assertEqual(
            tuple(failed),
            (1, "FAILED", "BATCH_EXECUTION_FAILED"),
        )

    def test_claim_uses_a_workload_derived_execution_lease(self) -> None:
        items = tuple(
            index_item(f"four-k-{index}", 600).model_copy(
                update={
                    "width": 3840,
                    "height": 2160,
                    "frame_count": 18_000,
                }
            )
            for index in range(5)
        )
        self.seal(*items)

        batch = self.claim(NOW, lease_seconds=None)

        expected = competition_execution_lease_seconds(
            evaluation.item for evaluation in batch.evaluations
        )
        with self.repository.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT timeout_seconds, lease_expires_at "
                    "FROM competition_batches WHERE batch_id=:batch_id"
                ),
                {"batch_id": batch.batch_id},
            ).one()
        self.assertEqual(row.timeout_seconds, expected)
        self.assertEqual(
            row.lease_expires_at,
            (NOW + timedelta(seconds=expected)).isoformat(),
        )

    def test_claim_applies_full_manifest_minimum_to_a_partial_batch(self) -> None:
        self.seal(index_item("one-short-video", 5))

        batch = self.claim(
            NOW,
            lease_seconds=None,
            minimum_execution_timeout_seconds=600,
        )

        with self.repository.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT timeout_seconds, lease_expires_at "
                    "FROM competition_batches WHERE batch_id=:batch_id"
                ),
                {"batch_id": batch.batch_id},
            ).one()
        self.assertEqual(row.timeout_seconds, 720)
        self.assertEqual(
            row.lease_expires_at,
            (NOW + timedelta(seconds=720)).isoformat(),
        )

    def test_expired_claim_is_terminal_and_cannot_be_replayed(self) -> None:
        self.seal(index_item("one", 10))
        first = self.claim(NOW)
        self.assertEqual(first.evaluations[0].attempt, 1)

        restarted = CompetitionRepository(self.database_url)
        replay = restarted.claim_evaluation_batch(
            self.competition_id,
            self.hotkey,
            owner="worker-after-crash",
            max_items=5,
            max_attempts=2,
            lease_seconds=60,
            scoring_version=self.manifest.scoring_version,
            vmaf_threshold=self.manifest.vmaf_threshold,
            max_video_length_seconds=self.manifest.max_video_length.total_seconds(),
            length_weight_exponent=self.manifest.length_weight_exponent,
            now=NOW + timedelta(seconds=61),
        )
        self.assertIsNone(replay)
        restarted.record_batch_outcomes(
            self.competition_id,
            self.hotkey,
            first.batch_id,
            (
                AttemptOutcome(
                    first.evaluations[0].history_id,
                    "SCORED",
                    metrics=ScoredItem(
                        input_checksum=first.evaluations[0].item.sha256,
                        output_checksum="f" * 64,
                        input_size_bytes=1000,
                        output_size_bytes=400,
                        vmaf_score=99,
                        compression_ratio=2.5,
                        runtime_seconds=1,
                        estimated_cost_usd=Decimal("0.001"),
                    ),
                ),
            ),
            max_attempts=2,
            modal_sandbox_id="sb-stale",
            wall_runtime_seconds=1,
            now=NOW + timedelta(seconds=61),
            actor="stale-worker",
        )
        with restarted.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT attempt, status, reason_code "
                    "FROM contender_performance_history "
                    "ORDER BY attempt"
                )
            ).all()
            batch_status = connection.execute(
                text(
                    "SELECT status FROM competition_batches "
                    "WHERE batch_id=:batch_id"
                ),
                {"batch_id": first.batch_id},
            ).scalar_one()
        self.assertEqual(
            rows,
            [(1, "FAILED", "BATCH_LEASE_EXPIRED")],
        )
        self.assertEqual(batch_status, "FAILED")
        self.assertTrue(
            restarted.evaluation_is_complete(
                self.competition_id, frozenset({"MODAL_ACCEPTED"})
            )
        )
        restarted.engine.dispose()

    def test_scoring_uses_a_fresh_deadline_after_sandbox_execution(self) -> None:
        self.seal(index_item("one", 10))
        first = self.claim(NOW)

        transitioned = self.repository.begin_batch_scoring(
            self.competition_id,
            self.hotkey,
            first.batch_id,
            owner="worker-a",
            scoring_timeout_seconds=300,
            now=NOW + timedelta(seconds=50),
        )

        self.assertTrue(transitioned)
        with self.repository.engine.connect() as connection:
            batch_row = connection.execute(
                text(
                    "SELECT status, timeout_seconds, scoring_timeout_seconds, "
                    "lease_owner, lease_expires_at, scoring_expires_at "
                    "FROM competition_batches WHERE batch_id=:batch_id"
                ),
                {"batch_id": first.batch_id},
            ).one()
        self.assertEqual(batch_row.status, "SCORING")
        self.assertEqual(batch_row.timeout_seconds, 60)
        self.assertEqual(batch_row.scoring_timeout_seconds, 300)
        self.assertIsNone(batch_row.lease_owner)
        self.assertIsNone(batch_row.lease_expires_at)
        self.assertEqual(
            batch_row.scoring_expires_at,
            (NOW + timedelta(seconds=350)).isoformat(),
        )
        self.assertIsNone(self.claim(NOW + timedelta(seconds=100)))

        replay = self.claim(NOW + timedelta(seconds=351))
        self.assertIsNone(replay)
        with self.repository.engine.connect() as connection:
            first_attempt = connection.execute(
                text(
                    "SELECT status, reason_code FROM contender_performance_history "
                    "WHERE batch_id=:batch_id"
                ),
                {"batch_id": first.batch_id},
            ).one()
        self.assertEqual(tuple(first_attempt), ("FAILED", "SCORING_LEASE_EXPIRED"))

    def test_batch_cost_and_runtime_include_failed_scoring_items(self) -> None:
        self.seal(index_item("one", 10), index_item("two", 20))
        batch = self.claim(NOW)
        scored_media = calculate_compression_score(
            vmaf_score=96,
            compression_rate=0.4,
            vmaf_threshold=self.manifest.vmaf_threshold,
        )
        outcomes = (
            AttemptOutcome(
                batch.evaluations[0].history_id,
                "SCORED",
                metrics=ScoredItem(
                    input_checksum=batch.evaluations[0].item.sha256,
                    output_checksum="a" * 64,
                    input_size_bytes=1000,
                    output_size_bytes=400,
                    vmaf_score=96,
                    compression_ratio=2.5,
                    media_score=scored_media[0],
                    media_compression_component=scored_media[1],
                    media_vmaf_component=scored_media[2],
                    media_score_reason=scored_media[3],
                    runtime_seconds=5,
                    estimated_cost_usd=Decimal("0.01"),
                ),
            ),
            AttemptOutcome(
                batch.evaluations[1].history_id,
                "FAILED",
                reason_code="MEDIA_SCORE_ZERO",
                metrics=ScoredItem(
                    input_checksum=batch.evaluations[1].item.sha256,
                    output_checksum="b" * 64,
                    input_size_bytes=1000,
                    output_size_bytes=500,
                    vmaf_score=80,
                    compression_ratio=2,
                    media_score=0,
                    media_compression_component=0,
                    media_vmaf_component=0,
                    media_score_reason="below hard cutoff",
                    runtime_seconds=5,
                    estimated_cost_usd=Decimal("0.01"),
                ),
            ),
        )
        self.repository.record_batch_outcomes(
            self.competition_id,
            self.hotkey,
            batch.batch_id,
            outcomes,
            max_attempts=2,
            modal_sandbox_id="sb-test",
            wall_runtime_seconds=10,
            batch_estimated_cost_usd=Decimal("0.02"),
            cost_attribution_method="BATCH_WALL_EQUAL_SHARE",
            now=NOW + timedelta(seconds=10),
            actor="test",
        )

        with self.repository.engine.connect() as connection:
            persisted_batch = connection.execute(
                text(
                    "SELECT wall_runtime_seconds, active_runtime_seconds, "
                    "estimated_cost_usd FROM competition_batches "
                    "WHERE batch_id=:batch_id"
                ),
                {"batch_id": batch.batch_id},
            ).one()
            histories = connection.execute(
                text(
                    "SELECT status, vmaf_score, compression_rate, "
                    "media_score, media_compression_component, "
                    "media_vmaf_component, handler_runtime_seconds, "
                    "estimated_cost_usd "
                    "FROM contender_performance_history ORDER BY evaluation_id"
                )
            ).all()
        self.assertEqual(tuple(persisted_batch), (10.0, 10.0, 0.02))
        self.assertEqual(
            histories[0],
            (
                "SCORED",
                96.0,
                2.5,
                scored_media[0],
                scored_media[1],
                scored_media[2],
                5.0,
                0.01,
            ),
        )
        self.assertEqual(
            histories[1],
            ("FAILED", 80.0, 2.0, 0.0, 0.0, 0.0, 5.0, 0.01),
        )

        self.repository.score_competition(
            self.competition_id,
            self.manifest,
            now=NOW + timedelta(seconds=11),
            actor="test",
        )
        contender = self.repository.get_contender(self.competition_id, self.hotkey)
        self.assertEqual(Decimal(str(contender.estimated_cost_usd)), Decimal("0.02"))
        self.assertEqual(contender.active_runtime_seconds, 10.0)

    def test_repository_rejects_automatic_retry_outcomes(self) -> None:
        self.seal(index_item("one", 10))
        first = self.claim(NOW)
        with self.assertRaisesRegex(ValueError, "invalid evaluation outcome 'RETRY'"):
            self.repository.record_batch_outcomes(
                self.competition_id,
                self.hotkey,
                first.batch_id,
                (
                    AttemptOutcome(
                        first.evaluations[0].history_id,
                        "RETRY",
                        reason_code="BATCH_EXECUTION_FAILED",
                        retryable=True,
                    ),
                ),
                max_attempts=2,
                modal_sandbox_id="sb-test",
                wall_runtime_seconds=4,
                batch_estimated_cost_usd=Decimal("0.01"),
                now=NOW + timedelta(seconds=4),
                actor="test",
            )

    def test_infrastructure_failure_blocks_scoring_and_can_be_auditably_requeued(
        self,
    ) -> None:
        self.seal(index_item("one", 10))
        first = self.claim(NOW)
        self.repository.record_batch_outcomes(
            self.competition_id,
            self.hotkey,
            first.batch_id,
            (
                AttemptOutcome(
                    first.evaluations[0].history_id,
                    "FAILED",
                    reason_code="SANDBOX_START_FAILED",
                ),
            ),
            max_attempts=2,
            modal_sandbox_id="sb-broken-1",
            wall_runtime_seconds=1820,
            batch_estimated_cost_usd=Decimal("1.82"),
            now=NOW + timedelta(seconds=1),
            actor="test",
        )
        self.assertIsNone(self.claim(NOW + timedelta(seconds=2)))
        self.assertEqual(
            self.repository.evaluation_infrastructure_blocker(
                self.competition_id, frozenset({"MODAL_ACCEPTED"})
            ),
            {"failed_items": 1, "reasons": {"SANDBOX_START_FAILED": 1}},
        )

        with self.repository.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE competitions SET status='AWAITING_END_TIME' "
                    "WHERE competition_id=:competition_id"
                ),
                {"competition_id": self.competition_id},
            )
            connection.execute(
                text(
                    "UPDATE contender_metadata SET status='SCORED', final_score=0, "
                    "failed_items=1, pending_items=0 "
                    "WHERE competition_id=:competition_id"
                ),
                {"competition_id": self.competition_id},
            )
            connection.execute(
                text(
                    "UPDATE competition_evaluation_items SET status='SCORED', "
                    "dispatch_status='COMPLETED', score_status='SCORED' "
                    "WHERE competition_id=:competition_id"
                ),
                {"competition_id": self.competition_id},
            )

        result = self.repository.requeue_infrastructure_failures(
            self.competition_id,
            reason_codes=frozenset({"SANDBOX_START_FAILED"}),
            now=NOW + timedelta(seconds=4),
            actor="operator:test",
        )
        self.assertEqual(result["requeued_evaluations"], 1)
        self.assertEqual(result["requeued_attempts"], 1)
        self.assertEqual(
            self.repository.get(self.competition_id).status,
            CompetitionState.EVALUATING.value,
        )
        contender = self.repository.get_contender(self.competition_id, self.hotkey)
        self.assertIsNone(contender.final_score)
        self.assertEqual(contender.pending_items, 1)
        second = self.claim(NOW + timedelta(seconds=5))
        self.assertEqual(second.evaluations[0].attempt, 2)
        with self.repository.engine.connect() as connection:
            statuses = connection.execute(
                text(
                    "SELECT attempt, status FROM contender_performance_history "
                    "ORDER BY attempt"
                )
            ).all()
        self.assertEqual(statuses, [(1, "REQUEUED"), (2, "RUNNING")])
        self.assertEqual(
            self.repository.list_events(self.competition_id)[-1].event_type,
            "EVALUATION_INFRASTRUCTURE_REQUEUED",
        )

        self.repository.record_batch_outcomes(
            self.competition_id,
            self.hotkey,
            second.batch_id,
            (
                AttemptOutcome(
                    second.evaluations[0].history_id,
                    "SCORED",
                    metrics=ScoredItem(
                        input_checksum=second.evaluations[0].item.sha256,
                        output_checksum="a" * 64,
                        input_size_bytes=1000,
                        output_size_bytes=400,
                        vmaf_score=96,
                        compression_ratio=2.5,
                        runtime_seconds=1,
                        estimated_cost_usd=Decimal("0.01"),
                    ),
                ),
            ),
            max_attempts=2,
            modal_sandbox_id="sb-replacement",
            wall_runtime_seconds=1,
            batch_estimated_cost_usd=Decimal("0.01"),
            now=NOW + timedelta(seconds=6),
            actor="test",
        )
        self.repository.score_competition(
            self.competition_id,
            self.manifest,
            now=NOW + timedelta(seconds=7),
            actor="test",
        )
        contender = self.repository.get_contender(self.competition_id, self.hotkey)
        self.assertEqual(
            Decimal(str(contender.estimated_cost_usd)), Decimal("0.01")
        )
        self.assertEqual(contender.active_runtime_seconds, 1.0)

    def test_one_infrastructure_failure_blocks_scoring_among_scored_items(
        self,
    ) -> None:
        second_hotkey = "contender-b"
        self.add_contender(second_hotkey)
        self.seal(index_item("one", 10))
        scored_batch = self.claim(NOW)
        self.repository.record_batch_outcomes(
            self.competition_id,
            self.hotkey,
            scored_batch.batch_id,
            (
                AttemptOutcome(
                    scored_batch.evaluations[0].history_id,
                    "SCORED",
                    metrics=ScoredItem(
                        input_checksum=scored_batch.evaluations[0].item.sha256,
                        output_checksum="a" * 64,
                        input_size_bytes=1000,
                        output_size_bytes=400,
                        vmaf_score=96,
                        compression_ratio=2.5,
                        runtime_seconds=1,
                        estimated_cost_usd=Decimal("0.01"),
                    ),
                ),
            ),
            max_attempts=2,
            modal_sandbox_id="sb-scored",
            wall_runtime_seconds=1,
            batch_estimated_cost_usd=Decimal("0.01"),
            now=NOW + timedelta(seconds=1),
            actor="test",
        )
        failed_batch = self.claim(
            NOW + timedelta(seconds=1), hotkey=second_hotkey
        )
        self.repository.record_batch_outcomes(
            self.competition_id,
            second_hotkey,
            failed_batch.batch_id,
            (
                AttemptOutcome(
                    failed_batch.evaluations[0].history_id,
                    "FAILED",
                    reason_code="SANDBOX_DISAPPEARED",
                ),
            ),
            max_attempts=2,
            modal_sandbox_id="sb-disappeared",
            wall_runtime_seconds=2,
            batch_estimated_cost_usd=Decimal("0.01"),
            now=NOW + timedelta(seconds=2),
            actor="test",
        )

        self.assertEqual(
            self.repository.evaluation_infrastructure_blocker(
                self.competition_id, frozenset({"MODAL_ACCEPTED"})
            ),
            {"failed_items": 1, "reasons": {"SANDBOX_DISAPPEARED": 1}},
        )

    def test_deferred_round_persistence_rolls_back_batch_and_scores_together(
        self,
    ) -> None:
        item = index_item("deferred-one", 10)
        self.seal(item)
        claimed = self.claim(NOW)
        metrics = ScoredItem(
            input_checksum=item.sha256,
            output_checksum="f" * 64,
            input_size_bytes=item.size_bytes,
            output_size_bytes=400,
            vmaf_score=96,
            compression_ratio=2.5,
            runtime_seconds=1,
            estimated_cost_usd=Decimal("0.001"),
        )
        result = DeferredBatchResult(
            claimed=claimed,
            outcomes=(
                AttemptOutcome(
                    claimed.evaluations[0].history_id,
                    "SCORED",
                    metrics=metrics,
                ),
            ),
            modal_sandbox_id="sb-deferred",
            wall_runtime_seconds=1,
            batch_estimated_cost_usd=Decimal("0.001"),
            cost_attribution_method="test",
            finished_at=NOW,
        )
        components = {
            claimed.evaluations[0].history_id: (1.0, 1.0, 1.0, 1.0, 1.0)
        }

        with self.assertRaisesRegex(
            ValueError,
            "aggregates do not match persisted contenders",
        ):
            self.repository.persist_deferred_competition_scores(
                self.competition_id,
                self.manifest,
                batch_results=(result,),
                aggregates=(SimpleNamespace(hotkey="wrong-contender"),),
                components=components,
                now=NOW,
                actor="test",
            )

        with self.repository.engine.connect() as connection:
            history_status = connection.execute(
                text(
                    "SELECT status FROM contender_performance_history "
                    "WHERE id=:history_id"
                ),
                {"history_id": claimed.evaluations[0].history_id},
            ).scalar_one()
            batch_status = connection.execute(
                text(
                    "SELECT status FROM competition_batches "
                    "WHERE competition_id=:competition_id AND batch_id=:batch_id"
                ),
                {
                    "competition_id": self.competition_id,
                    "batch_id": claimed.batch_id,
                },
            ).scalar_one()
        self.assertEqual(history_status, "RUNNING")
        self.assertEqual(batch_status, "RUNNING")

    def test_deferred_claims_advance_without_committing_prior_outcomes(self) -> None:
        items = tuple(index_item(f"deferred-{index}", 10) for index in range(6))
        self.seal(*items)
        claim_kwargs = {
            "owner": "worker-a",
            "max_items": self.manifest.evaluation_batch_size,
            "max_attempts": self.manifest.max_attempts_per_item,
            "minimum_execution_timeout_seconds": (
                self.manifest.evaluation_batched_run_timeout.total_seconds()
            ),
            "scoring_version": self.manifest.scoring_version,
            "vmaf_threshold": self.manifest.vmaf_threshold,
            "max_video_length_seconds": (
                self.manifest.max_video_length.total_seconds()
            ),
            "length_weight_exponent": self.manifest.length_weight_exponent,
            "now": NOW,
            "defer_outcomes": True,
        }
        first = self.repository.claim_evaluation_batch(
            self.competition_id,
            self.hotkey,
            **claim_kwargs,
        )
        completed_ids = frozenset(
            evaluation.item.evaluation_id for evaluation in first.evaluations
        )
        second = self.repository.claim_evaluation_batch(
            self.competition_id,
            self.hotkey,
            deferred_evaluation_ids=completed_ids,
            **claim_kwargs,
        )

        self.assertIsNotNone(second)
        self.assertNotEqual(first.batch_id, second.batch_id)
        self.assertEqual(len(first.evaluations), 5)
        self.assertEqual(len(second.evaluations), 1)
        with self.repository.engine.connect() as connection:
            statuses = connection.execute(
                text(
                    "SELECT DISTINCT status FROM contender_performance_history "
                    "WHERE competition_id=:competition_id"
                ),
                {"competition_id": self.competition_id},
            ).scalars().all()
        self.assertEqual(statuses, ["RUNNING"])


class FakeInspector:
    def __init__(self, *, output_frames: int = 300, output_sar: str = "1:1"):
        self.output_frames = output_frames
        self.output_sar = output_sar

    def inspect(self, path: Path) -> MediaInfo:
        output = path.name == "output.mp4"
        return MediaInfo(
            width=1280,
            height=720,
            duration_seconds=10,
            codec="av1" if output else "h264",
            container="mov,mp4,m4a,3gp,3g2,mj2",
            pixel_format="yuv420p",
            sample_aspect_ratio=self.output_sar if output else "1:1",
            size_bytes=path.stat().st_size,
            frame_count=self.output_frames if output else 300,
        )


class Phase4ScoringTests(unittest.TestCase):
    def test_current_modal_sandbox_rates_and_gpu_name_mapping(self) -> None:
        self.assertEqual(
            set(GPU_PRICE_PER_SECOND_USD),
            {
                "B300",
                "B200",
                "H200",
                "H100",
                "RTX-PRO-6000",
                "A100-80GB",
                "A100-40GB",
                "L40S",
                "A10",
                "L4",
                "T4",
            },
        )
        self.assertEqual(SANDBOX_CPU_PRICE_PER_CORE_SECOND_USD, Decimal("0.00003942"))
        self.assertEqual(canonical_modal_gpu_type("NVIDIA H200"), "H200")
        self.assertEqual(canonical_modal_gpu_type("NVIDIA A100-SXM4-80GB"), "A100-80GB")
        self.assertEqual(
            canonical_modal_gpu_type("NVIDIA RTX PRO 6000 Blackwell Server Edition"),
            "RTX-PRO-6000",
        )
        self.assertEqual(
            estimate_sandbox_cost(
                allocated_gpu_type="H200",
                allocated_gpu_count=1,
                allocated_cpu_cores=8,
                runtime_seconds=10,
            ),
            Decimal("0.01576360"),
        )

    def test_aggregate_uses_manifest_selected_scoring_weights(self) -> None:
        row = SimpleNamespace(
            id=1,
            hotkey="custom-weights",
            evaluation_id="same-input",
            status="SCORED",
            length_weight=1.0,
            vmaf_threshold=89,
            vmaf_score=95,
            compression_rate=2.0,
            reconciled_cost_usd=None,
            estimated_cost_usd=Decimal("0.01"),
        )
        factors = ScoringFactors(
            quality=Decimal("0.2"),
            cost_efficiency=Decimal("0.3"),
            length_coverage=Decimal("0.5"),
            runtime=Decimal("0"),
        )
        manifest = self.manifest.model_copy(
            update={
                "scoring_version": "custom-weights",
                "scoring_factors": factors,
            }
        )

        aggregates, _components = compute_aggregates(manifest, (row,))

        media_score = calculate_compression_score(
            vmaf_score=95,
            compression_rate=0.5,
            vmaf_threshold=89,
        )[0]
        self.assertAlmostEqual(
            aggregates[0].final_score,
            round(
                0.2 * media_score + 0.3 * 1 + 0.5 * 1,
                manifest.score_precision,
            ),
        )

    def setUp(self) -> None:
        self.manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        self.source = b"s" * 1000
        self.output = b"o" * 400
        self.item = EvaluationIndexItem(
            evaluation_id="score-me",
            source_path="inputs/score-me.mp4",
            size_bytes=len(self.source),
            sha256=sha256(self.source).hexdigest(),
            duration_seconds=10,
            width=1280,
            height=720,
            frame_count=300,
            codec="h264",
            pixel_format="yuv420p",
            sample_aspect_ratio="1:1",
        )

    def test_valid_output_records_deterministic_metrics_and_cost(self) -> None:
        scorer = CompetitionItemScorer(
            inspector=FakeInspector(), vmaf=lambda *_args: 96.5
        )
        self.assertIsInstance(scorer, CompetitionTaskAdapter)
        result = scorer.score(
            self.manifest,
            self.item,
            self.source,
            self.output,
            runtime_seconds=2.5,
            **ALLOCATED_RESOURCES,
        )
        self.assertEqual(result.compression_ratio, 2.5)
        self.assertEqual(result.vmaf_score, 96.5)
        expected_media = calculate_compression_score(
            vmaf_score=96.5,
            compression_rate=0.4,
            vmaf_threshold=self.manifest.vmaf_threshold,
        )
        self.assertEqual(result.media_score, expected_media[0])
        self.assertEqual(result.media_compression_component, expected_media[1])
        self.assertEqual(result.media_vmaf_component, expected_media[2])
        self.assertEqual(result.estimated_cost_usd, Decimal("0.003940900"))

    def test_absolute_media_curve_handles_vmaf_100_target(self) -> None:
        media_score, compression, quality, reason = calculate_compression_score(
            vmaf_score=100,
            compression_rate=0.4,
            vmaf_threshold=100,
        )
        self.assertGreater(media_score, 0)
        self.assertGreater(compression, 0)
        self.assertEqual(quality, 1)
        self.assertEqual(reason, "success")

    def test_unprobeable_output_uses_validator_owned_media_reason(self) -> None:
        class OutputProbeFailureInspector(FakeInspector):
            def inspect(self, path: Path) -> MediaInfo:
                if path.name == "output.mp4":
                    raise QualificationError(
                        "MEDIA_PROBE_INVALID", "ffprobe returned incomplete metadata"
                    )
                return super().inspect(path)

        with self.assertRaises(ItemScoringError) as captured:
            CompetitionItemScorer(
                inspector=OutputProbeFailureInspector(),
                vmaf=lambda *_args: 100,
            ).score(
                self.manifest,
                self.item,
                self.source,
                self.output,
                runtime_seconds=2.5,
                **ALLOCATED_RESOURCES,
            )

        self.assertEqual(captured.exception.reason_code, "OUTPUT_MEDIA_INVALID")
        self.assertEqual(captured.exception.metrics.output_size_bytes, len(self.output))
        self.assertEqual(captured.exception.metrics.compression_ratio, 2.5)

    def test_compression_media_adapter_rejects_three_file_contract(self) -> None:
        scorer = CompetitionItemScorer(
            inspector=FakeInspector(), vmaf=lambda *_args: 96.5
        )
        with self.assertRaises(ItemScoringError) as captured:
            scorer.score_media(
                self.manifest,
                self.item,
                CompetitionScoringMedia.upscaling(
                    ground_truth_video=b"ground-truth",
                    downsampled_reference_video=self.source,
                    miner_processed_video=self.output,
                ),
                runtime_seconds=1,
                **ALLOCATED_RESOURCES,
            )
        self.assertEqual(
            captured.exception.reason_code, "COMPETITION_MEDIA_CONTRACT_MISMATCH"
        )

    def test_changed_frame_count_or_aspect_ratio_is_rejected(self) -> None:
        for inspector, reason in (
            (FakeInspector(output_frames=280), "OUTPUT_FRAME_COUNT_CHANGED"),
            (FakeInspector(output_sar="4:3"), "OUTPUT_ASPECT_RATIO_CHANGED"),
        ):
            with (
                self.subTest(reason=reason),
                self.assertRaises(ItemScoringError) as captured,
            ):
                CompetitionItemScorer(
                    inspector=inspector, vmaf=lambda *_args: 96.5
                ).score(
                    self.manifest,
                    self.item,
                    self.source,
                    self.output,
                    runtime_seconds=2.5,
                    **ALLOCATED_RESOURCES,
                )
            self.assertEqual(captured.exception.reason_code, reason)
            self.assertEqual(captured.exception.metrics.compression_ratio, 2.5)
            self.assertEqual(captured.exception.metrics.vmaf_score, 96.5)

    def test_vmaf_failure_retains_measured_metrics_and_cost(self) -> None:
        scorer = CompetitionItemScorer(
            inspector=FakeInspector(), vmaf=lambda *_args: 80.0
        )
        with self.assertRaises(ItemScoringError) as captured:
            scorer.score(
                self.manifest,
                self.item,
                self.source,
                self.output,
                runtime_seconds=2.5,
                **ALLOCATED_RESOURCES,
            )
        self.assertEqual(captured.exception.reason_code, "MEDIA_SCORE_ZERO")
        self.assertEqual(captured.exception.metrics.media_score, 0)
        self.assertEqual(captured.exception.metrics.compression_ratio, 2.5)
        self.assertEqual(captured.exception.metrics.vmaf_score, 80.0)
        self.assertEqual(
            captured.exception.metrics.estimated_cost_usd,
            Decimal("0.003940900"),
        )

    def test_query_variant_vmaf_threshold_overrides_manifest_default(self) -> None:
        variant = CompressionEvaluationIndexItem(
            **self.item.model_dump(),
            codec_mode="CRF",
            vmaf_threshold=85,
        )
        result = CompetitionItemScorer(
            inspector=FakeInspector(), vmaf=lambda *_args: 86.0
        ).score(
            self.manifest,
            variant,
            self.source,
            self.output,
            runtime_seconds=1,
            **ALLOCATED_RESOURCES,
        )
        self.assertEqual(result.vmaf_score, 86.0)

        stricter = variant.model_copy(update={"vmaf_threshold": 93})
        soft_result = CompetitionItemScorer(
            inspector=FakeInspector(), vmaf=lambda *_args: 92.0
        ).score(
            self.manifest,
            stricter,
            self.source,
            self.output,
            runtime_seconds=1,
            **ALLOCATED_RESOURCES,
        )
        self.assertGreater(soft_result.media_score, 0)
        self.assertLess(soft_result.media_score, result.media_score)

    def test_manifest_exponent_gives_thirty_minutes_ten_times_the_weight(
        self,
    ) -> None:
        short_weight = length_weight(10, self.manifest)
        long_weight = length_weight(1800, self.manifest)

        self.assertAlmostEqual(long_weight / short_weight, 10.0, places=2)

    def test_aggregate_gives_failed_long_item_zero_in_all_components(self) -> None:
        short_weight = length_weight(10, self.manifest)
        long_weight = length_weight(100, self.manifest)

        def row(identifier, hotkey, evaluation_id, status, weight, ratio, cost):
            return SimpleNamespace(
                id=identifier,
                hotkey=hotkey,
                evaluation_id=evaluation_id,
                status=status,
                length_weight=weight,
                vmaf_threshold=90,
                compression_rate=ratio,
                reconciled_cost_usd=None,
                estimated_cost_usd=Decimal(cost),
                vmaf_score=95 if status == "SCORED" else None,
            )

        histories = (
            row(1, "complete", "short", "SCORED", short_weight, 2, "0.02"),
            row(2, "complete", "long", "SCORED", long_weight, 4, "0.04"),
            row(3, "partial", "short", "SCORED", short_weight, 4, "0.01"),
            row(4, "partial", "long", "FAILED", long_weight, None, "0.01"),
        )
        aggregates, components = compute_aggregates(self.manifest, histories)
        by_hotkey = {value.hotkey: value for value in aggregates}
        self.assertEqual(components[4], (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(by_hotkey["complete"].length_coverage, 1.0)
        self.assertLess(by_hotkey["partial"].length_coverage, 0.5)
        self.assertGreater(
            by_hotkey["complete"].final_score,
            by_hotkey["partial"].final_score,
        )

    def test_cost_score_is_relative_to_cheapest_valid_contender(self) -> None:
        def row(identifier, hotkey, ratio, cost):
            return SimpleNamespace(
                id=identifier,
                hotkey=hotkey,
                evaluation_id="same-input",
                status="SCORED",
                length_weight=1.0,
                vmaf_threshold=89,
                vmaf_score=95,
                compression_rate=ratio,
                reconciled_cost_usd=None,
                estimated_cost_usd=Decimal(cost),
            )

        target = row(1, "target", 5.0, "0.03")
        target_alone, components_alone = compute_aggregates(
            self.manifest,
            (target,),
        )
        with_competitors, components_with_competitors = compute_aggregates(
            self.manifest,
            (
                target,
                row(2, "better-ratio", 20.0, "0.005"),
                row(3, "worse-ratio", 1.5, "0.20"),
                row(4, "invalid-cheap", 1.0, "0.0001"),
            ),
        )
        target_with_competitors = {
            value.hotkey: value for value in with_competitors
        }["target"]
        self.assertEqual(
            target_alone[0].media_score_aggregate,
            target_with_competitors.media_score_aggregate,
        )
        self.assertEqual(target_alone[0].cost_efficiency_aggregate, 1.0)
        self.assertAlmostEqual(
            target_with_competitors.cost_efficiency_aggregate,
            float(Decimal("0.005") / Decimal("0.03")),
        )
        self.assertGreater(
            target_alone[0].final_score,
            target_with_competitors.final_score,
        )
        self.assertEqual(
            components_alone[target.id][:3],
            components_with_competitors[target.id][:3],
        )
        self.assertEqual(components_alone[target.id][3], 1.0)
        self.assertAlmostEqual(
            components_with_competitors[target.id][3],
            float(Decimal("0.005") / Decimal("0.03")),
        )
        self.assertEqual(
            components_with_competitors[4],
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )


class Phase4CoordinatorTests(unittest.TestCase):
    def test_sandbox_forward_failure_is_persisted_as_terminal(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        claimed = SimpleNamespace(
            batch_id="batch-forward-failed",
            canonical_batch_index=0,
            hotkey="contender",
            evaluations=(
                SimpleNamespace(
                    history_id=7,
                    attempt=1,
                    item=index_item("long-video", 3600),
                ),
            ),
        )

        class Repository:
            outcomes = None

            @staticmethod
            def latest_sandbox(_competition_id, _hotkey):
                return SimpleNamespace(
                    modal_sandbox_id="sb-forward-failed",
                    **ALLOCATED_RESOURCES,
                )

            def record_batch_outcomes(self, *args, **_kwargs):
                self.outcomes = args[3]

        class Runner:
            @staticmethod
            def invoke_batch(*_args, **_kwargs):
                raise SandboxRunnerError(
                    "SANDBOX_EXEC_FAILED",
                    "contender forward failed",
                )

        repository = Repository()
        coordinator = CompetitionExecutionCoordinator(
            manager=None,
            repository=repository,
            build_service=None,
            sandbox_runner=Runner(),
            artifact_root=ROOT,
            actor="validator:test",
            accepted_build_statuses=frozenset({"MODAL_ACCEPTED"}),
            dataset_store=None,
            item_scorer=None,
            clock=lambda: NOW,
        )

        asyncio.run(coordinator._execute_batch(manifest, claimed))

        outcome = repository.outcomes[0]
        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.reason_code, "SANDBOX_EXEC_FAILED")
        self.assertFalse(outcome.retryable)

    def test_validator_scores_returned_output_and_owns_failure_reason(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        source = b"s" * 1000
        output = b"o" * 1200
        item = CompressionEvaluationIndexItem(
            evaluation_id="vbr-output",
            source_path="inputs/source.mp4",
            size_bytes=len(source),
            sha256=sha256(source).hexdigest(),
            duration_seconds=10,
            width=1280,
            height=720,
            frame_count=300,
            codec="h264",
            pixel_format="yuv420p",
            sample_aspect_ratio="1:1",
            codec_mode="VBR",
            vmaf_threshold=89,
            target_bitrate=8_000_000,
        )
        claimed = SimpleNamespace(
            batch_id="batch-failed-output",
            canonical_batch_index=0,
            hotkey="contender",
            evaluations=(SimpleNamespace(history_id=7, attempt=1, item=item),),
        )

        class Repository:
            outcomes = None

            @staticmethod
            def get_contender(_competition_id, _hotkey):
                return SimpleNamespace(output_volume_name="contender-output")

            @staticmethod
            def latest_sandbox(_competition_id, _hotkey):
                return SimpleNamespace(
                    modal_sandbox_id="sb-failed-output", **ALLOCATED_RESOURCES
                )

            def record_batch_outcomes(self, *args, **_kwargs):
                self.outcomes = args[3]

        class Store:
            output_read_path = None

            def read_bytes(self, volume_name, path):
                if volume_name == manifest.evaluation_input_volume_name:
                    return source
                self.output_read_path = path
                return output

        class Runner:
            @staticmethod
            def invoke_batch(
                _manifest, request, *, input_sub_path, timeout_seconds, now
            ):
                del input_sub_path
                del timeout_seconds
                requested = request.items[0]
                return CompetitionCompressionResponse(
                    results=(
                        CompetitionCompressionResult(output_path=requested.output_path),
                    ),
                )

        metrics = ScoredItem(
            input_checksum=item.sha256,
            output_checksum=sha256(output).hexdigest(),
            input_size_bytes=len(source),
            output_size_bytes=len(output),
            vmaf_score=95,
            compression_ratio=len(source) / len(output),
            runtime_seconds=1,
            estimated_cost_usd=Decimal("0.001"),
        )

        class Scorer:
            @staticmethod
            def score_media(*_args, **_kwargs):
                raise ItemScoringError(
                    "OUTPUT_NOT_SMALLER",
                    "output is larger than input",
                    metrics=metrics,
                )

        repository = Repository()
        store = Store()
        coordinator = CompetitionExecutionCoordinator(
            manager=None,
            repository=repository,
            build_service=None,
            sandbox_runner=Runner(),
            artifact_root=ROOT,
            actor="validator:test",
            accepted_build_statuses=frozenset({"MODAL_ACCEPTED"}),
            dataset_store=store,
            item_scorer=Scorer(),
            clock=lambda: NOW,
        )

        asyncio.run(coordinator._execute_batch(manifest, claimed))

        self.assertTrue(store.output_read_path.endswith("vbr-output.mp4"))
        outcome = repository.outcomes[0]
        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.reason_code, "OUTPUT_NOT_SMALLER")
        self.assertEqual(outcome.metrics.output_size_bytes, len(output))

        class PassingScorer:
            @staticmethod
            def score_media(*_args, **_kwargs):
                return metrics

        repository.outcomes = None
        coordinator.item_scorer = PassingScorer()
        asyncio.run(coordinator._execute_batch(manifest, claimed))
        self.assertEqual(repository.outcomes[0].status, "SCORED")
        self.assertIsNone(repository.outcomes[0].reason_code)

    def test_fast_contender_terminates_before_slow_contender_finishes(self) -> None:
        manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        fast_terminated = threading.Event()

        class Repository:
            contenders = (
                SimpleNamespace(
                    hotkey="fast",
                    build_status="MODAL_ACCEPTED",
                    output_volume_name="fast-output",
                ),
                SimpleNamespace(
                    hotkey="slow",
                    build_status="MODAL_ACCEPTED",
                    output_volume_name="slow-output",
                ),
            )
            claimed = set()
            complete = set()

            def list_contenders(self, _competition_id):
                return self.contenders

            def contender_evaluation_is_complete(self, _competition_id, hotkey):
                return hotkey in self.complete

            def claim_evaluation_batch(self, _competition_id, hotkey, **_kwargs):
                if hotkey in self.claimed:
                    return None
                self.claimed.add(hotkey)
                return SimpleNamespace(hotkey=hotkey)

            def get_contender(self, _competition_id, hotkey):
                return next(row for row in self.contenders if row.hotkey == hotkey)

            @staticmethod
            def latest_sandbox(_competition_id, hotkey):
                return SimpleNamespace(
                    status=(
                        "TERMINATED"
                        if hotkey == "fast" and fast_terminated.is_set()
                        else "RUNNING"
                    ),
                    modal_sandbox_id=f"sb-{hotkey}",
                    **ALLOCATED_RESOURCES,
                )

        class Runner:
            terminated = []

            def terminate(self, _manifest, hotkey, *, now):
                self.terminated.append((hotkey, now))
                if hotkey == "fast":
                    fast_terminated.set()

        repository = Repository()
        runner = Runner()

        class Coordinator(CompetitionExecutionCoordinator):
            async def _execute_batch(self, _manifest, claimed):
                if claimed.hotkey == "slow":
                    observed = await asyncio.to_thread(fast_terminated.wait, 1)
                    self.assert_fast_was_terminated(observed)
                repository.complete.add(claimed.hotkey)

            @staticmethod
            def assert_fast_was_terminated(observed):
                if not observed:
                    raise AssertionError(
                        "fast contender stayed warm while slow contender was running"
                    )

        coordinator = Coordinator(
            manager=None,
            repository=repository,
            build_service=None,
            sandbox_runner=runner,
            artifact_root=ROOT,
            actor="validator:test",
            accepted_build_statuses=frozenset({"MODAL_ACCEPTED"}),
            clock=lambda: NOW,
        )
        asyncio.run(coordinator._evaluate_contenders(manifest))

        self.assertEqual(
            runner.terminated,
            [("fast", NOW), ("slow", NOW)],
        )

    def test_dispatch_scores_and_advances_to_awaiting_end_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            database_url = f"sqlite:///{root / 'competition.db'}"
            repository = CompetitionRepository(database_url)
            manifest = load_manifest(
                ROOT / "competitions/manifests/examples/compression-competition.json"
            )
            competition_id = manifest.competition_id
            hotkey = "coordinator-contender"
            repository.insert_manifest(manifest, now=NOW, actor="test")
            for target in (
                CompetitionState.ENROLLING,
                CompetitionState.FINALIZING_SUBMISSIONS,
                CompetitionState.VALIDATING,
                CompetitionState.BUILDING,
                CompetitionState.EVALUATING,
            ):
                repository.transition(
                    competition_id,
                    target,
                    now=NOW,
                    actor="test",
                )
            repository.record_pinned_contender(
                competition_id=competition_id,
                hotkey=hotkey,
                repository_url_hash="a" * 64,
                repository_display="github.com/test/coordinator",
                pinned_commit_sha="b" * 40,
                pinned_tree_sha="c" * 40,
                latest_commit_time=NOW.isoformat(),
                validation=ValidationReport(
                    ValidationStatus.ACCEPTED,
                    ValidationReason.ACCEPTED,
                    "d" * 64,
                    1,
                    100,
                    (),
                ),
                now=NOW,
                actor="test",
            )
            repository.record_build_evidence(
                competition_id=competition_id,
                hotkey=hotkey,
                image_id="im-phase4",
                image_digest="sha256:" + "e" * 64,
                image_size_bytes=1,
                evidence={"builder_id": "test"},
                build_status="MODAL_ACCEPTED",
                now=NOW,
                actor="test",
            )
            with repository.engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE contender_metadata SET output_volume_name="
                        "'phase4-output' WHERE competition_id=:competition_id"
                    ),
                    {"competition_id": competition_id},
                )

            source = b"s" * 1000
            output = b"o" * 400
            item = CompressionEvaluationIndexItem(
                evaluation_id="one-vbr-vmaf89-8mbps",
                source_path="inputs/one.mp4",
                size_bytes=len(source),
                sha256=sha256(source).hexdigest(),
                duration_seconds=10,
                width=1280,
                height=720,
                frame_count=300,
                codec="h264",
                pixel_format="yuv420p",
                sample_aspect_ratio="1:1",
                codec_mode="VBR",
                vmaf_threshold=89,
                target_bitrate=8_000_000,
            )
            evaluation_index = EvaluationIndex(
                competition_id=competition_id, items=(item,)
            ).with_batch_scoped_source_paths(manifest)
            repository.seal_evaluation_dataset(
                competition_id, evaluation_index, now=NOW, actor="test"
            )

            class Store:
                output_path = None

                def load_index(self, _manifest):
                    return evaluation_index

                def read_bytes(self, volume_name, path):
                    if volume_name == manifest.evaluation_input_volume_name:
                        return source
                    self.output_path = path
                    return output

            class Runner:
                def __init__(self):
                    self.terminated = []
                    self.requests = []

                @staticmethod
                def probe_random_input_subpath(
                    _manifest, _hotkey, _candidate_input_sub_paths
                ):
                    return {"global_index_absent": True}

                def invoke_batch(
                    self,
                    _manifest,
                    request,
                    *,
                    input_sub_path,
                    timeout_seconds,
                    now,
                ):
                    del input_sub_path
                    del timeout_seconds
                    self.requests.append(request)
                    return CompetitionCompressionResponse(
                        results=tuple(
                            CompetitionCompressionResult(output_path=value.output_path)
                            for value in request.items
                        ),
                    )

                def terminate(self, terminated_manifest, terminated_hotkey, *, now):
                    self.terminated.append(
                        (terminated_manifest.competition_id, terminated_hotkey, now)
                    )

            class Scorer:
                def score_media(
                    self,
                    _manifest,
                    scored_item,
                    media,
                    *,
                    runtime_seconds,
                    **_resources,
                ):
                    assert media.ground_truth_video is None
                    return ScoredItem(
                        input_checksum=scored_item.sha256,
                        output_checksum=sha256(output).hexdigest(),
                        input_size_bytes=len(source),
                        output_size_bytes=len(output),
                        vmaf_score=96,
                        compression_ratio=2.5,
                        runtime_seconds=runtime_seconds,
                        estimated_cost_usd=Decimal("0.001"),
                    )

            store = Store()
            manager = CompetitionManager(
                CompetitionConfig(
                    mode_enabled=True,
                    database_url=database_url,
                    artifact_root=root / "artifacts",
                    artifact_backup_bucket="private-test-bucket",
                    owner_id="validator:test",
                ),
                repository,
                clock=lambda: NOW,
            )
            runner = Runner()
            coordinator = CompetitionExecutionCoordinator(
                manager,
                repository,
                build_service=None,
                sandbox_runner=runner,
                artifact_root=root / "artifacts",
                actor="validator:test",
                accepted_build_statuses=frozenset({"MODAL_ACCEPTED"}),
                dataset_store=store,
                item_scorer=Scorer(),
                defer_round_commit=True,
                clock=lambda: NOW,
            )

            def latest_sandbox(*_args, **_kwargs):
                return SimpleNamespace(
                    status="TERMINATED" if runner.terminated else "RUNNING",
                    modal_sandbox_id="sb-phase4",
                    **ALLOCATED_RESOURCES,
                )

            with patch.object(
                repository,
                "latest_sandbox",
                side_effect=latest_sandbox,
            ), patch.object(
                repository,
                "record_batch_outcomes",
                side_effect=AssertionError(
                    "deferred execution must not persist individual batches"
                ),
            ), patch.object(
                repository,
                "persist_deferred_competition_scores",
                wraps=repository.persist_deferred_competition_scores,
            ) as deferred_commit:
                asyncio.run(coordinator.run_once())
                # The terminal phase remains idempotent on subsequent ticks.
                asyncio.run(coordinator.run_once())

            deferred_commit.assert_called_once()
            self.assertEqual(
                repository.get(competition_id).status,
                CompetitionState.AWAITING_END_TIME.value,
            )
            self.assertTrue(store.output_path.startswith("batches/batch-"))
            self.assertIn("/evaluations/one-vbr-vmaf89-8mbps.mp4", store.output_path)
            self.assertFalse(store.output_path.startswith("/output/"))
            media_score = calculate_compression_score(
                vmaf_score=96,
                compression_rate=0.4,
                vmaf_threshold=item.vmaf_threshold,
            )[0]
            self.assertAlmostEqual(
                repository.get_contender(competition_id, hotkey).final_score,
                round(
                    0.6 * media_score + 0.25 * 1 + 0.15 * 1,
                    manifest.score_precision,
                ),
            )
            self.assertEqual(
                runner.terminated,
                [(competition_id, hotkey, NOW)],
            )
            dispatched_item = runner.requests[0].items[0]
            self.assertEqual(dispatched_item.codec_mode, "VBR")
            self.assertEqual(dispatched_item.target_bitrate, 8_000_000)
            self.assertEqual(dispatched_item.vmaf_threshold, 89)
            self.assertEqual(
                repository.get_contender(competition_id, hotkey).output_volume_name,
                "phase4-output",
            )
            repository.engine.dispose()


if __name__ == "__main__":
    unittest.main()
