from __future__ import annotations

import asyncio
import json
import unittest
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import httpx

from vidaio_subnet_core.competition.config import load_manifest
from vidaio_subnet_core.competition.dataset import EvaluationIndexItem
from vidaio_subnet_core.competition.contracts import (
    CompetitionCompressionResponse,
    CompetitionCompressionResult,
)
from vidaio_subnet_core.competition.execution import CompetitionExecutionCoordinator
from vidaio_subnet_core.competition.scoring_api import (
    CompetitionAggregateHistory,
    CompetitionAggregatePayload,
    CompetitionAggregateResponse,
    CompetitionHistoryComponent,
    CompetitionScoredItemPayload,
    CompetitionScoringBatchItem,
    CompetitionScoringBatchResult,
    CompetitionScoringClient,
)
from vidaio_subnet_core.competition.timeouts import (
    competition_execution_timeout_seconds,
    competition_scoring_timeout_seconds,
)


ROOT = Path(__file__).resolve().parents[2]


class CompetitionScoringClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest(
            ROOT / "competitions/manifests/examples/compression-competition.json"
        )
        self.item = EvaluationIndexItem(
            evaluation_id="evaluation-1",
            source_path="inputs/evaluation-1.mp4",
            size_bytes=1000,
            sha256="b" * 64,
            duration_seconds=10,
            width=1280,
            height=720,
            frame_count=300,
            codec="h264",
            pixel_format="yuv420p",
            sample_aspect_ratio="1:1",
        )

    def test_posts_batched_item_scoring_request(self) -> None:
        async def run():
            async def handler(request: httpx.Request) -> httpx.Response:
                self.assertEqual(request.url.path, "/score_compression_competition")
                payload = json.loads(request.content)
                self.assertEqual(
                    payload["items"][0]["item"]["evaluation_id"],
                    "evaluation-1",
                )
                self.assertEqual(payload["output_volume_name"], "contender-output")
                self.assertEqual(payload["items"][0]["allocated_gpu_type"], "H200")
                self.assertEqual(payload["items"][0]["allocated_cpu_cores"], 8.0)
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "evaluation_id": "evaluation-1",
                                "status": "SCORED",
                                "metrics": {
                                    "input_checksum": "b" * 64,
                                    "output_checksum": "c" * 64,
                                    "input_size_bytes": 1000,
                                    "output_size_bytes": 400,
                                    "vmaf_score": 95.5,
                                    "compression_ratio": 2.5,
                                    "media_score": 0.75,
                                    "media_compression_component": 0.25,
                                    "media_vmaf_component": 0.85,
                                    "media_score_reason": "success",
                                    "runtime_seconds": 1.25,
                                    "estimated_cost_usd": "0.001",
                                    "cost_attribution_method": "test",
                                },
                            }
                        ]
                    },
                )

            async with httpx.AsyncClient(
                base_url="http://scoring.test",
                transport=httpx.MockTransport(handler),
            ) as http_client:
                client = CompetitionScoringClient(
                    "http://unused.test",
                    modal_environment="dev",
                    client=http_client,
                )
                results = await client.score_batch(
                    self.manifest,
                    output_volume_name="contender-output",
                    items=(
                        CompetitionScoringBatchItem(
                            item=self.item,
                            output_path="evaluations/batch-1/evaluation-1.mp4",
                            runtime_seconds=1.25,
                            allocated_gpu_type="H200",
                            allocated_gpu_count=1,
                            allocated_cpu_cores=8.0,
                        ),
                    ),
                )
                self.assertEqual(results[0].status, "SCORED")
                self.assertEqual(
                    results[0].metrics.to_scored_item().estimated_cost_usd,
                    Decimal("0.001"),
                )
                self.assertEqual(results[0].metrics.media_score, 0.75)

        asyncio.run(run())

    def test_coordinator_delegates_batch_without_reading_media_locally(self) -> None:
        output = b"compressed"
        claimed = SimpleNamespace(
            hotkey="hotkey-1",
            batch_id="batch-1",
            evaluations=(
                SimpleNamespace(history_id=7, attempt=1, item=self.item),
            ),
        )

        class Repository:
            outcomes = None

            @staticmethod
            def get_contender(_competition_id, _hotkey):
                return SimpleNamespace(output_volume_name="contender-output")

            @staticmethod
            def latest_sandbox(_competition_id, _hotkey):
                return SimpleNamespace(
                    modal_sandbox_id="sandbox-1",
                    allocated_gpu_type="H200",
                    allocated_gpu_count=1,
                    allocated_cpu_cores=8.0,
                )

            def record_batch_outcomes(self, *args, **_kwargs):
                self.outcomes = args[3]

            def begin_batch_scoring(self, *args, **kwargs):
                self.scoring_phase = (args, kwargs)
                return True

        runner_state = {}

        class Runner:
            @staticmethod
            def invoke_batch(_manifest, request, *, timeout_seconds, now):
                del now
                runner_state["timeout_seconds"] = timeout_seconds
                return CompetitionCompressionResponse(
                    results=(
                        CompetitionCompressionResult(
                            output_path=request.items[0].output_path
                        ),
                    )
                )

        class Store:
            @staticmethod
            def read_bytes(*_args, **_kwargs):
                raise AssertionError("remote scoring must own competition media reads")

        class RemoteScorer:
            async def score_batch(self, manifest, *, output_volume_name, items):
                self.request = (manifest, output_volume_name, items)
                return (
                    CompetitionScoringBatchResult(
                        evaluation_id="evaluation-1",
                        status="SCORED",
                        metrics=CompetitionScoredItemPayload(
                            input_checksum="b" * 64,
                            output_checksum=sha256(output).hexdigest(),
                            input_size_bytes=1000,
                            output_size_bytes=len(output),
                            vmaf_score=95,
                            compression_ratio=2.5,
                            runtime_seconds=1,
                            estimated_cost_usd=Decimal("0.001"),
                            cost_attribution_method="test",
                        ),
                    ),
                )

        repository = Repository()
        scorer = RemoteScorer()
        coordinator = CompetitionExecutionCoordinator(
            manager=None,
            repository=repository,
            build_service=None,
            sandbox_runner=Runner(),
            artifact_root=ROOT,
            actor="validator:test",
            accepted_build_statuses=frozenset({"MODAL_ACCEPTED"}),
            dataset_store=Store(),
            item_scorer=scorer,
        )
        asyncio.run(coordinator._execute_batch(self.manifest, claimed))

        self.assertEqual(repository.outcomes[0].status, "SCORED")
        self.assertEqual(repository.scoring_phase[0][2], "batch-1")
        self.assertEqual(
            repository.scoring_phase[1]["scoring_timeout_seconds"],
            competition_scoring_timeout_seconds(
                (self.item,),
                minimum_timeout_seconds=(
                    self.manifest.scoring_batched_run_timeout.total_seconds()
                ),
            ),
        )
        self.assertEqual(
            runner_state["timeout_seconds"],
            competition_execution_timeout_seconds(
                (self.item,),
                minimum_timeout_seconds=(
                    self.manifest.evaluation_batched_run_timeout.total_seconds()
                ),
            ),
        )
        self.assertEqual(scorer.request[1], "contender-output")
        self.assertEqual(
            scorer.request[2][0].output_path,
            "evaluations/batch-1/evaluation-1.mp4",
        )

    def test_posts_aggregate_scoring_request(self) -> None:
        async def run():
            async def handler(request: httpx.Request) -> httpx.Response:
                self.assertEqual(
                    request.url.path,
                    "/score_compression_competition_aggregates",
                )
                return httpx.Response(
                    200,
                    json={
                        "aggregates": [
                            {
                                "hotkey": "hotkey-1",
                                "media_score_aggregate": 1,
                                "quality_aggregate": 1,
                                "cost_efficiency_aggregate": 1,
                                "length_coverage": 1,
                                "final_score": 1,
                                "average_vmaf": 95,
                                "average_compression_ratio": 2.5,
                                "estimated_cost_usd": "0.001",
                                "successful_items": 1,
                                "failed_items": 0,
                            }
                        ],
                        "components": [
                            {
                                "history_id": 7,
                                "media_score": 1,
                                "compression": 1,
                                "vmaf_quality": 1,
                                "cost_efficiency": 1,
                                "completion": 1,
                            }
                        ],
                    },
                )

            async with httpx.AsyncClient(
                base_url="http://scoring.test",
                transport=httpx.MockTransport(handler),
            ) as http_client:
                client = CompetitionScoringClient(
                    "http://unused.test",
                    modal_environment="dev",
                    client=http_client,
                )
                response = await client.score_aggregates(
                    self.manifest,
                    (
                        CompetitionAggregateHistory(
                            id=7,
                            hotkey="hotkey-1",
                            evaluation_id="evaluation-1",
                            status="SCORED",
                            length_weight=1,
                            vmaf_score=95,
                            compression_rate=2.5,
                            estimated_cost_usd=Decimal("0.001"),
                        ),
                    ),
                )
                self.assertEqual(response.aggregates[0].final_score, 1)
                self.assertEqual(response.components[0].history_id, 7)

        asyncio.run(run())

    def test_coordinator_delegates_final_aggregation_before_persistence(self) -> None:
        class Repository:
            persisted = None

            @staticmethod
            def list_contenders(_competition_id):
                return ()

            @staticmethod
            def competition_scoring_rows(_competition_id):
                return (
                    {
                        "id": 7,
                        "hotkey": "hotkey-1",
                        "evaluation_id": "evaluation-1",
                        "status": "SCORED",
                        "length_weight": 1,
                        "vmaf_score": 95,
                        "compression_rate": 2.5,
                        "estimated_cost_usd": Decimal("0.001"),
                        "reconciled_cost_usd": None,
                    },
                )

            def persist_competition_scores(self, *args, **kwargs):
                self.persisted = (args, kwargs)

            @staticmethod
            def score_competition(*_args, **_kwargs):
                raise AssertionError("final scores must be computed remotely")

        class RemoteScorer:
            async def score_aggregates(self, manifest, histories):
                self.request = (manifest, histories)
                return CompetitionAggregateResponse(
                    aggregates=(
                        CompetitionAggregatePayload(
                            hotkey="hotkey-1",
                            media_score_aggregate=1,
                            quality_aggregate=1,
                            cost_efficiency_aggregate=1,
                            length_coverage=1,
                            final_score=1,
                            average_vmaf=95,
                            average_compression_ratio=2.5,
                            estimated_cost_usd=Decimal("0.001"),
                            successful_items=1,
                            failed_items=0,
                        ),
                    ),
                    components=(
                        CompetitionHistoryComponent(
                            history_id=7,
                            media_score=1,
                            compression=1,
                            vmaf_quality=1,
                            cost_efficiency=1,
                            completion=1,
                        ),
                    ),
                )

        class Manager:
            completed = False

            def complete_current_stage(self, _competition_id):
                self.completed = True

        repository = Repository()
        scorer = RemoteScorer()
        manager = Manager()
        coordinator = CompetitionExecutionCoordinator(
            manager=manager,
            repository=repository,
            build_service=None,
            sandbox_runner=None,
            artifact_root=ROOT,
            actor="validator:test",
            accepted_build_statuses=frozenset(),
            dataset_store=None,
            item_scorer=scorer,
        )
        asyncio.run(coordinator._score_and_await_end_time(self.manifest))

        self.assertTrue(manager.completed)
        self.assertEqual(scorer.request[1][0].id, 7)
        self.assertEqual(repository.persisted[1]["aggregates"][0].final_score, 1)
        self.assertEqual(
            repository.persisted[1]["components"][7],
            (1, 1, 1, 1, 1),
        )


if __name__ == "__main__":
    unittest.main()
