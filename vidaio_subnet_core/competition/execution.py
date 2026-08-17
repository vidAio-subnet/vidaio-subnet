"""Live contender build and sandbox orchestration for competition validators."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

from loguru import logger

from .build import CompetitionBuildService, TrustedBuildError
from .config import CompetitionManifest, calculate_length_weight
from .contracts import CompetitionCompressionItem, CompetitionCompressionRequest
from .dataset import DatasetError
from .intake import pinned_repository_source
from .manager import CompetitionManager
from .media_contracts import CompetitionScoringMedia
from .modal_runner import CompetitionModalRunner, SandboxRunnerError
from .phase0 import SecretRedactor
from .repository import (
    AttemptOutcome,
    CompetitionRepository,
    DeferredBatchResult,
    INFRASTRUCTURE_FAILURE_REASON_CODES,
)
from .scoring import (
    COST_ATTRIBUTION_METHOD,
    ItemScoringError,
    estimate_item_cost,
)
from .scoring_api import (
    CompetitionAggregateHistory,
    CompetitionScoringBatchItem,
)
from .state import CompetitionState
from .timeouts import (
    competition_execution_timeout_seconds,
    competition_scoring_timeout_seconds,
)


LOG_REDACTOR = SecretRedactor()


def _persisted_outcome_summary(outcomes) -> dict:
    """Project the statuses record_batch_outcomes will persist for this claim."""

    scored = sum(outcome.status == "SCORED" for outcome in outcomes)
    return {
        "scored": scored,
        "retry": 0,
        "failed": len(outcomes) - scored,
        "reasons": dict(
            sorted(
                Counter(
                    outcome.reason_code
                    for outcome in outcomes
                    if outcome.reason_code is not None
                ).items()
            )
        ),
    }


class CompetitionExecutionCoordinator:
    """Advance accepted submissions through image build and sandbox startup."""

    def __init__(
        self,
        manager: CompetitionManager,
        repository: CompetitionRepository,
        build_service: CompetitionBuildService,
        sandbox_runner: CompetitionModalRunner,
        *,
        artifact_root: Path,
        actor: str,
        accepted_build_statuses: frozenset[str],
        dataset_store=None,
        item_scorer=None,
        defer_round_commit: bool = False,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.manager = manager
        self.repository = repository
        self.build_service = build_service
        self.sandbox_runner = sandbox_runner
        self.artifact_root = artifact_root
        self.actor = actor
        self.accepted_build_statuses = accepted_build_statuses
        self.dataset_store = dataset_store
        self.item_scorer = item_scorer
        self.defer_round_commit = defer_round_commit
        self.clock = clock

    async def run_once(self) -> None:
        competitions = await asyncio.to_thread(self.repository.list_nonterminal)
        for competition in competitions:
            state = CompetitionState(competition.status)
            if state not in {
                CompetitionState.FINALIZING_SUBMISSIONS,
                CompetitionState.VALIDATING,
                CompetitionState.BUILDING,
                CompetitionState.EVALUATING,
                CompetitionState.SCORING,
                CompetitionState.AWAITING_END_TIME,
            }:
                continue
            manifest = CompetitionManifest.model_validate_json(
                competition.manifest_json
            )
            await self._advance(manifest, state)

    async def _advance(
        self, manifest: CompetitionManifest, state: CompetitionState
    ) -> None:
        competition_id = manifest.competition_id
        if state == CompetitionState.AWAITING_END_TIME:
            competition = await asyncio.to_thread(
                self.repository.get, competition_id
            )
            if competition is not None and getattr(
                competition, "scores_need_recalculation", False
            ):
                # Database-only recovery: this never builds, starts, or invokes
                # a contender sandbox and does not rerun media/VMAF scoring.
                await asyncio.to_thread(
                    self.repository.recalculate_competition_scores,
                    competition_id,
                    now=self.clock(),
                    actor=self.actor,
                )
            await self._terminate_contender_sandboxes(manifest)
            return

        if state == CompetitionState.SCORING:
            await self._score_and_await_end_time(manifest)
            return

        if state == CompetitionState.FINALIZING_SUBMISSIONS:
            competition = await asyncio.to_thread(
                self.repository.get, competition_id
            )
            if (
                competition is None
                or competition.submission_backup_status != "COMPLETED"
            ):
                logger.warning(
                    "Competition validation is waiting for the private submission "
                    "snapshot backup: id={} backup_status={}",
                    competition_id,
                    getattr(competition, "submission_backup_status", None),
                )
                return
            state = self.manager.complete_current_stage(competition_id)

        if state == CompetitionState.VALIDATING:
            contenders = await asyncio.to_thread(
                self.repository.list_contenders, competition_id
            )
            submitted = [row for row in contenders if row.pinned_commit_sha]
            awaiting_review = [
                row
                for row in submitted
                if row.validation_status == "REVIEW_REQUIRED"
                and not getattr(row, "manual_disqualified", False)
            ]
            if awaiting_review:
                logger.warning(
                    "Competition validation is waiting for human review: id={} "
                    "hotkeys={}",
                    competition_id,
                    [row.hotkey for row in awaiting_review],
                )
                return
            accepted = [
                row
                for row in submitted
                if row.validation_status == "ACCEPTED"
                and not getattr(row, "manual_disqualified", False)
            ]
            if not accepted:
                self.manager.fail(
                    competition_id,
                    "no statically accepted contender repositories were submitted",
                )
                return
            state = self.manager.complete_current_stage(competition_id)

        if state == CompetitionState.BUILDING:
            await self._build_accepted_contenders(manifest)
            contenders = await asyncio.to_thread(
                self.repository.list_contenders, competition_id
            )
            eligible = [
                row
                for row in contenders
                if row.validation_status == "ACCEPTED"
                and not getattr(row, "manual_disqualified", False)
            ]
            if any(row.build_status in {None, "BUILDING", "RETRY"} for row in eligible):
                return
            built = [
                row
                for row in eligible
                if row.build_status in self.accepted_build_statuses and row.image_id
            ]
            if not built:
                self.manager.fail(competition_id, "all contender image builds failed")
                return
            state = self.manager.complete_current_stage(competition_id)

        if state == CompetitionState.EVALUATING:
            contenders = await asyncio.to_thread(
                self.repository.list_contenders, competition_id
            )
            disqualified = [
                contender
                for contender in contenders
                if getattr(contender, "manual_disqualified", False)
                and contender.build_status in self.accepted_build_statuses
            ]
            await asyncio.gather(
                *(
                    self._terminate_contender_sandbox(
                        manifest,
                        contender,
                        trigger="contender_manually_disqualified",
                    )
                    for contender in disqualified
                )
            )
            active_built = [
                contender
                for contender in contenders
                if not getattr(contender, "manual_disqualified", False)
                and contender.build_status in self.accepted_build_statuses
            ]
            if not active_built:
                self.manager.fail(
                    competition_id,
                    "all built contenders were manually disqualified",
                )
                return
            if self.dataset_store is None or self.item_scorer is None:
                logger.warning(
                    "Competition evaluation dispatcher is unavailable: id={} "
                    "dataset_store={} item_scorer={}",
                    competition_id,
                    self.dataset_store is not None,
                    self.item_scorer is not None,
                )
                return
            competition = await asyncio.to_thread(self.repository.get, competition_id)
            if competition is None or not competition.dataset_index_checksum:
                logger.warning(
                    "Competition evaluation is waiting for a sealed dataset: id={} "
                    "volume={} index_path={}",
                    competition_id,
                    manifest.evaluation_input_volume_name,
                    manifest.evaluation_index_path,
                )
                return
            try:
                evaluation_index = await asyncio.to_thread(
                    self.dataset_store.load_index, manifest
                )
            except DatasetError as exc:
                logger.error(
                    "Competition evaluation index unavailable: id={} reason={}",
                    competition_id,
                    LOG_REDACTOR.redact_text(str(exc)),
                )
                return
            if evaluation_index.digest() != competition.dataset_index_checksum:
                logger.error(
                    "Competition evaluation index checksum mismatch: id={} "
                    "expected={} observed={}",
                    competition_id,
                    competition.dataset_index_checksum,
                    evaluation_index.digest(),
                )
                return
            assignments = evaluation_index.canonical_batch_assignments(manifest)
            probe_input_sub_paths = tuple(
                f"/batches/{batch_index:06d}"
                for batch_index in sorted(
                    {batch_index for batch_index, _position in assignments.values()}
                )
            )
            sandboxes_ready = await self._ensure_sandboxes(
                manifest,
                probe_input_sub_paths=probe_input_sub_paths,
            )
            if not sandboxes_ready:
                logger.warning(
                    "Competition random batch-subpath isolation probe failed: id={}",
                    competition_id,
                )
                return
            deferred_results = await self._evaluate_contenders(manifest)
            if self.defer_round_commit:
                expected = {
                    (contender.hotkey, item.evaluation_id)
                    for contender in active_built
                    for item in evaluation_index.items
                }
                observed = {
                    (result.claimed.hotkey, evaluation.item.evaluation_id)
                    for result in deferred_results
                    for evaluation in result.claimed.evaluations
                }
                if observed != expected:
                    missing = sorted(expected - observed)
                    unexpected = sorted(observed - expected)
                    logger.error(
                        "Deferred competition evaluation is incomplete: id={} "
                        "missing={} unexpected={}",
                        competition_id,
                        missing,
                        unexpected,
                    )
                    return
                blocker = self._deferred_infrastructure_blocker(deferred_results)
                if blocker is not None:
                    logger.error(
                        "Competition evaluation halted after systemic infrastructure "
                        "failure: id={} failed_items={} reasons={}",
                        competition_id,
                        blocker["failed_items"],
                        blocker["reasons"],
                    )
                    return
                state = self.manager.complete_current_stage(competition_id)
                if state == CompetitionState.SCORING:
                    await self._score_and_await_end_time(
                        manifest,
                        evaluation_items=len(evaluation_index.items),
                        deferred_results=deferred_results,
                    )
                return
            complete = await asyncio.to_thread(
                self.repository.evaluation_is_complete,
                competition_id,
                self.accepted_build_statuses,
            )
            if complete:
                blocker = await asyncio.to_thread(
                    self.repository.evaluation_infrastructure_blocker,
                    competition_id,
                    self.accepted_build_statuses,
                )
                if blocker is not None:
                    logger.error(
                        "Competition evaluation halted after systemic infrastructure "
                        "failure: id={} failed_items={} reasons={}; fix the validator "
                        "infrastructure and requeue these attempts",
                        competition_id,
                        blocker["failed_items"],
                        blocker["reasons"],
                    )
                    return
                state = self.manager.complete_current_stage(competition_id)
                if state == CompetitionState.SCORING:
                    await self._score_and_await_end_time(
                        manifest, evaluation_items=len(evaluation_index.items)
                    )

    async def _score_and_await_end_time(
        self,
        manifest: CompetitionManifest,
        *,
        evaluation_items: int | None = None,
        deferred_results: tuple[DeferredBatchResult, ...] = (),
    ) -> None:
        """Finish scoring and release compute while preserving every Volume."""

        await self._terminate_contender_sandboxes(manifest)
        score_kwargs = {}
        if hasattr(self.item_scorer, "score_aggregates"):
            scoring_rows = (
                self._deferred_scoring_rows(manifest, deferred_results)
                if deferred_results
                else await asyncio.to_thread(
                    self.repository.competition_scoring_rows,
                    manifest.competition_id,
                )
            )
            remote_scores = await self.item_scorer.score_aggregates(
                manifest,
                tuple(
                    CompetitionAggregateHistory.model_validate(value)
                    for value in scoring_rows
                ),
            )
            score_kwargs = {
                "aggregates": remote_scores.aggregates,
                "components": {
                    value.history_id: (
                        value.media_score,
                        value.compression,
                        value.vmaf_quality,
                        value.cost_efficiency,
                        value.completion,
                    )
                    for value in remote_scores.components
                },
            }
        if deferred_results:
            if not score_kwargs:
                from .scoring import compute_aggregates

                scoring_rows = self._deferred_scoring_rows(
                    manifest, deferred_results
                )
                aggregates, components = compute_aggregates(manifest, scoring_rows)
                score_kwargs = {
                    "aggregates": aggregates,
                    "components": components,
                }
            persist_scores = self.repository.persist_deferred_competition_scores
            score_kwargs["batch_results"] = deferred_results
        else:
            persist_scores = (
                self.repository.persist_competition_scores
                if score_kwargs
                else self.repository.score_competition
            )
        await asyncio.to_thread(
            persist_scores,
            manifest.competition_id,
            manifest,
            now=self.clock(),
            actor=self.actor,
            **score_kwargs,
        )
        self.manager.complete_current_stage(manifest.competition_id)
        logger.info(
            "Competition Phase 4 complete: id={} evaluation_items={}",
            manifest.competition_id,
            evaluation_items if evaluation_items is not None else "persisted",
        )

    @staticmethod
    def _deferred_infrastructure_blocker(
        results: tuple[DeferredBatchResult, ...],
    ) -> dict | None:
        outcomes = [
            outcome for result in results for outcome in result.outcomes
        ]
        if not outcomes:
            return None
        reasons: dict[str, int] = {}
        for outcome in outcomes:
            reason = outcome.reason_code or "UNKNOWN"
            if (
                outcome.status == "FAILED"
                and reason in INFRASTRUCTURE_FAILURE_REASON_CODES
            ):
                reasons[reason] = reasons.get(reason, 0) + 1
        if not reasons:
            return None
        return {
            "failed_items": sum(reasons.values()),
            "reasons": dict(sorted(reasons.items())),
        }

    @staticmethod
    def _deferred_scoring_rows(
        manifest: CompetitionManifest,
        results: tuple[DeferredBatchResult, ...],
    ) -> tuple[CompetitionAggregateHistory, ...]:
        rows = []
        seen_history_ids: set[int] = set()
        for result in results:
            evaluations = {
                evaluation.history_id: evaluation
                for evaluation in result.claimed.evaluations
            }
            outcomes = {
                outcome.history_id: outcome for outcome in result.outcomes
            }
            if set(evaluations) != set(outcomes) or len(outcomes) != len(
                result.outcomes
            ):
                raise ValueError(
                    "deferred outcomes do not exactly match their claimed batch"
                )
            item_cost = result.batch_estimated_cost_usd / len(result.outcomes)
            for history_id, evaluation in evaluations.items():
                if history_id in seen_history_ids:
                    raise ValueError(
                        f"duplicate deferred history ID {history_id}"
                    )
                seen_history_ids.add(history_id)
                outcome = outcomes[history_id]
                metrics = outcome.metrics
                rows.append(
                    CompetitionAggregateHistory(
                        id=history_id,
                        hotkey=result.claimed.hotkey,
                        evaluation_id=evaluation.item.evaluation_id,
                        status=outcome.status,
                        length_weight=calculate_length_weight(
                            evaluation.item.duration_seconds,
                            manifest.max_video_length.total_seconds(),
                            manifest.length_weight_exponent,
                        ),
                        vmaf_threshold=float(
                            getattr(
                                evaluation.item,
                                "vmaf_threshold",
                                manifest.vmaf_threshold,
                            )
                        ),
                        vmaf_score=(
                            metrics.vmaf_score if metrics is not None else None
                        ),
                        compression_rate=(
                            metrics.compression_ratio
                            if metrics is not None
                            else None
                        ),
                        media_score=(
                            metrics.media_score if metrics is not None else None
                        ),
                        media_compression_component=(
                            metrics.media_compression_component
                            if metrics is not None
                            else None
                        ),
                        media_vmaf_component=(
                            metrics.media_vmaf_component
                            if metrics is not None
                            else None
                        ),
                        estimated_cost_usd=item_cost,
                        reconciled_cost_usd=None,
                    )
                )
        return tuple(rows)

    async def _terminate_contender_sandboxes(
        self, manifest: CompetitionManifest
    ) -> None:
        """Terminate active compute without deleting persistent input/output data."""

        contenders = await asyncio.to_thread(
            self.repository.list_contenders, manifest.competition_id
        )
        await asyncio.gather(
            *(
                self._terminate_contender_sandbox(
                    manifest,
                    contender,
                    trigger="competition_evaluation_complete",
                )
                for contender in contenders
                if contender.build_status in self.accepted_build_statuses
            ),
        )

    async def _terminate_contender_sandbox(
        self,
        manifest: CompetitionManifest,
        contender,
        *,
        trigger: str,
    ) -> None:
        latest = await asyncio.to_thread(
            self.repository.latest_sandbox,
            manifest.competition_id,
            contender.hotkey,
        )
        if latest is None or latest.status not in {"STARTING", "RUNNING"}:
            return
        try:
            await asyncio.to_thread(
                self.sandbox_runner.terminate,
                manifest,
                contender.hotkey,
                now=self.clock(),
            )
        except Exception as exc:
            logger.error(
                "Contender Sandbox termination failed: competition_id={} "
                "hotkey={} sandbox_id={} trigger={} reason={}",
                manifest.competition_id,
                contender.hotkey,
                latest.modal_sandbox_id,
                trigger,
                LOG_REDACTOR.redact_text(str(exc)),
            )
        else:
            logger.info(
                "Contender Sandbox terminated after evaluation: competition_id={} "
                "hotkey={} sandbox_id={} trigger={} output_volume={} "
                "volume_retained=true",
                manifest.competition_id,
                contender.hotkey,
                latest.modal_sandbox_id,
                trigger,
                contender.output_volume_name,
            )

    async def _contenders_requiring_evaluation(self, manifest, contenders):
        completion = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self.repository.contender_evaluation_is_complete,
                    manifest.competition_id,
                    contender.hotkey,
                )
                for contender in contenders
            )
        )
        completed = [
            contender
            for contender, is_complete in zip(contenders, completion, strict=True)
            if is_complete
        ]
        await asyncio.gather(
            *(
                self._terminate_contender_sandbox(
                    manifest,
                    contender,
                    trigger="contender_dataset_complete",
                )
                for contender in completed
            )
        )
        return [
            contender
            for contender, is_complete in zip(contenders, completion, strict=True)
            if not is_complete
        ]

    async def _build_accepted_contenders(self, manifest: CompetitionManifest) -> None:
        contenders = await asyncio.to_thread(
            self.repository.list_contenders, manifest.competition_id
        )
        pending = [
            contender
            for contender in contenders
            if contender.validation_status == "ACCEPTED"
            and not getattr(contender, "manual_disqualified", False)
            and contender.build_status
            not in self.accepted_build_statuses | {"REJECTED"}
        ]
        semaphore = asyncio.Semaphore(manifest.max_parallel_contenders)
        await asyncio.gather(
            *(
                self._build_contender(manifest, contender.hotkey, semaphore)
                for contender in pending
            )
        )

    async def _build_contender(
        self,
        manifest: CompetitionManifest,
        hotkey: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            source = pinned_repository_source(
                self.artifact_root, manifest.competition_id, hotkey
            )
            await asyncio.to_thread(
                self.repository.mark_build_started,
                competition_id=manifest.competition_id,
                hotkey=hotkey,
                now=self.clock(),
                actor=self.actor,
            )
            logger.info(
                "Building contender image: competition_id={} hotkey={} source={}",
                manifest.competition_id,
                hotkey,
                source,
            )
            try:
                evidence = await asyncio.to_thread(
                    self.build_service.build_contender,
                    manifest.competition_id,
                    hotkey,
                    source,
                    manifest.modal_build_timeout.total_seconds(),
                )
                logger.info(
                    "Contender image ready: competition_id={} hotkey={} "
                    "image_id={} builder={}",
                    manifest.competition_id,
                    hotkey,
                    evidence.image_id,
                    evidence.builder_id,
                )
            except TrustedBuildError as exc:
                logger.error(
                    "Contender image build rejected: competition_id={} hotkey={} "
                    "reason={}",
                    manifest.competition_id,
                    hotkey,
                    LOG_REDACTOR.redact_text(str(exc)),
                )

    async def _ensure_sandboxes(
        self,
        manifest: CompetitionManifest,
        *,
        probe_input_sub_paths: tuple[str, ...],
    ) -> bool:
        contenders = await asyncio.to_thread(
            self.repository.list_contenders, manifest.competition_id
        )
        built = [
            contender
            for contender in contenders
            if contender.build_status in self.accepted_build_statuses
            and not getattr(contender, "manual_disqualified", False)
        ]
        pending = await self._contenders_requiring_evaluation(manifest, built)
        if not pending:
            return bool(built)
        # Volume sub_path isolation is a competition-wide platform property;
        # one disposable Sandbox mounts a randomly selected existing batch to
        # prove it before dispatch. Every real batch still runs the same active
        # isolation probe after mounting its own path.
        probe_contender = pending[0]
        semaphore = asyncio.Semaphore(manifest.max_parallel_contenders)
        result = await self._ensure_sandbox(
            manifest,
            probe_contender.hotkey,
            semaphore,
            probe_input_sub_paths=probe_input_sub_paths,
        )
        return result

    async def _ensure_sandbox(
        self,
        manifest: CompetitionManifest,
        hotkey: str,
        semaphore: asyncio.Semaphore,
        *,
        probe_input_sub_paths: tuple[str, ...],
    ) -> bool:
        async with semaphore:
            try:
                report = await asyncio.to_thread(
                    self.sandbox_runner.probe_random_input_subpath,
                    manifest,
                    hotkey,
                    probe_input_sub_paths,
                )
                logger.info(
                    "Random batch-subpath isolation probe passed: competition_id={} "
                    "hotkey={} global_index_absent={}",
                    manifest.competition_id,
                    hotkey,
                    report.get("global_index_absent"),
                )
                return True
            except SandboxRunnerError as exc:
                logger.error(
                    "Contender sandbox startup failed and will be retried: "
                    "competition_id={} hotkey={} reason={}",
                    manifest.competition_id,
                    hotkey,
                    LOG_REDACTOR.redact_text(str(exc)),
                )
                return False

    async def _evaluate_contenders(
        self, manifest: CompetitionManifest
    ) -> tuple[DeferredBatchResult, ...]:
        contenders = await asyncio.to_thread(
            self.repository.list_contenders, manifest.competition_id
        )
        built = [
            contender
            for contender in contenders
            if contender.build_status in self.accepted_build_statuses
            and not getattr(contender, "manual_disqualified", False)
        ]
        pending = await self._contenders_requiring_evaluation(manifest, built)
        semaphore = asyncio.Semaphore(manifest.max_parallel_contenders)
        contender_results = await asyncio.gather(
            *(
                self._evaluate_contender(manifest, contender.hotkey, semaphore)
                for contender in pending
            )
        )
        return tuple(
            result
            for results in contender_results
            for result in results
        )

    async def _evaluate_contender(
        self,
        manifest: CompetitionManifest,
        hotkey: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple[DeferredBatchResult, ...]:
        async with semaphore:
            deferred_results: list[DeferredBatchResult] = []
            deferred_evaluation_ids: set[str] = set()
            while True:
                claimed = await asyncio.to_thread(
                    self.repository.claim_evaluation_batch,
                    manifest.competition_id,
                    hotkey,
                    owner=self.actor,
                    max_items=manifest.evaluation_batch_size,
                    max_attempts=manifest.max_attempts_per_item,
                    minimum_execution_timeout_seconds=(
                        manifest.evaluation_batched_run_timeout.total_seconds()
                    ),
                    scoring_version=manifest.scoring_version,
                    vmaf_threshold=manifest.vmaf_threshold,
                    max_video_length_seconds=(
                        manifest.max_video_length.total_seconds()
                    ),
                    length_weight_exponent=manifest.length_weight_exponent,
                    now=self.clock(),
                    deferred_evaluation_ids=frozenset(
                        deferred_evaluation_ids
                    ),
                    defer_outcomes=self.defer_round_commit,
                )
                if claimed is None:
                    complete = self.defer_round_commit and bool(
                        deferred_evaluation_ids
                    )
                    if not self.defer_round_commit:
                        complete = await asyncio.to_thread(
                            self.repository.contender_evaluation_is_complete,
                            manifest.competition_id,
                            hotkey,
                        )
                    if complete:
                        contender = await asyncio.to_thread(
                            self.repository.get_contender,
                            manifest.competition_id,
                            hotkey,
                        )
                        if contender is not None:
                            await self._terminate_contender_sandbox(
                                manifest,
                                contender,
                                trigger="contender_dataset_complete",
                            )
                    return tuple(deferred_results)
                result = await self._execute_batch(manifest, claimed)
                if self.defer_round_commit:
                    if not isinstance(result, DeferredBatchResult):
                        raise RuntimeError(
                            "deferred competition batch returned no result"
                        )
                    deferred_results.append(result)
                    deferred_evaluation_ids.update(
                        evaluation.item.evaluation_id
                        for evaluation in claimed.evaluations
                    )

    async def _execute_batch(
        self, manifest: CompetitionManifest, claimed
    ) -> DeferredBatchResult:
        output_paths = {
            evaluation.item.evaluation_id: (
                f"/output/evaluations/{evaluation.item.evaluation_id}.mp4"
            )
            for evaluation in claimed.evaluations
        }
        request = CompetitionCompressionRequest(
            competition_id=manifest.competition_id,
            hotkey=claimed.hotkey,
            batch_id=claimed.batch_id,
            items=tuple(
                CompetitionCompressionItem(
                    evaluation_id=evaluation.item.evaluation_id,
                    input_path=evaluation.item.sandbox_path,
                    output_path=output_paths[evaluation.item.evaluation_id],
                    codec=manifest.required_output_codec,
                    codec_mode=getattr(evaluation.item, "codec_mode", "CRF"),
                    target_bitrate=getattr(evaluation.item, "target_bitrate", None),
                    vmaf_threshold=float(
                        getattr(
                            evaluation.item,
                            "vmaf_threshold",
                            manifest.vmaf_threshold,
                        )
                    ),
                )
                for evaluation in claimed.evaluations
            ),
        )
        execution_timeout = competition_execution_timeout_seconds(
            (evaluation.item for evaluation in claimed.evaluations),
            minimum_timeout_seconds=(
                manifest.evaluation_batched_run_timeout.total_seconds()
            ),
        )
        started = time.monotonic()
        batch_wall_runtime: float | None = None
        outcomes: list[AttemptOutcome] = []
        modal_sandbox_id = None
        recorded_scoring_timeout: float | None = None
        try:
            logger.info(
                "Dispatching competition batch: competition_id={} hotkey={} "
                "batch_id={} timeout_seconds={} items={}",
                manifest.competition_id,
                claimed.hotkey,
                claimed.batch_id,
                execution_timeout,
                [value.item.evaluation_id for value in claimed.evaluations],
            )
            dispatch_started = time.monotonic()
            response = await asyncio.to_thread(
                self.sandbox_runner.invoke_batch,
                manifest,
                request,
                input_sub_path=f"/batches/{claimed.canonical_batch_index:06d}",
                timeout_seconds=execution_timeout,
                now=self.clock(),
            )
            batch_wall_runtime = time.monotonic() - dispatch_started
            item_runtime = batch_wall_runtime / len(claimed.evaluations)
            latest = await asyncio.to_thread(
                self.repository.latest_sandbox,
                manifest.competition_id,
                claimed.hotkey,
            )
            modal_sandbox_id = latest.modal_sandbox_id if latest is not None else None
            if latest is None or not (
                latest.allocated_gpu_type
                and latest.allocated_gpu_count
                and latest.allocated_cpu_cores
            ):
                raise SandboxRunnerError(
                    "SANDBOX_ALLOCATION_MISSING",
                    "running Sandbox has no verified allocated GPU/CPU record",
                )
            allocated_gpu_type = latest.allocated_gpu_type
            allocated_gpu_count = latest.allocated_gpu_count
            allocated_cpu_cores = latest.allocated_cpu_cores
            if len(response.results) != len(claimed.evaluations):
                raise SandboxRunnerError(
                    "COMPRESSION_RESPONSE_ITEMS_MISMATCH",
                    "sandbox response count does not exactly match the request",
                )
            contender = await asyncio.to_thread(
                self.repository.get_contender,
                manifest.competition_id,
                claimed.hotkey,
            )
            if contender is None or not contender.output_volume_name:
                raise DatasetError("contender output Volume is unavailable")
            scoring_items = []
            legacy_scoring_items = []
            for evaluation, result in zip(
                claimed.evaluations, response.results, strict=True
            ):
                if not result.output_path:
                    outcomes.append(
                        AttemptOutcome(
                            evaluation.history_id,
                            "FAILED",
                            "CONTENDER_OUTPUT_MISSING",
                        )
                    )
                    continue
                if result.output_path != output_paths[evaluation.item.evaluation_id]:
                    outcomes.append(
                        AttemptOutcome(
                            evaluation.history_id,
                            "FAILED",
                            "OUTPUT_PATH_MISMATCH",
                        )
                    )
                    continue
                relative_output_path = (
                    f"batches/{claimed.batch_id}/"
                    f"{result.output_path.removeprefix('/output/')}"
                )
                scoring_items.append(
                    CompetitionScoringBatchItem(
                        item=evaluation.item,
                        output_path=relative_output_path,
                        # Attribute validator-measured batch wall time equally
                        # because the route may process items concurrently.
                        runtime_seconds=item_runtime,
                        allocated_gpu_type=allocated_gpu_type,
                        allocated_gpu_count=allocated_gpu_count,
                        allocated_cpu_cores=allocated_cpu_cores,
                    )
                )
                legacy_scoring_items.append((evaluation, relative_output_path))

            if scoring_items and hasattr(self.item_scorer, "score_batch"):
                recorded_scoring_timeout = competition_scoring_timeout_seconds(
                    (value.item for value in scoring_items),
                    minimum_timeout_seconds=(
                        manifest.scoring_batched_run_timeout.total_seconds()
                    ),
                )
                if not self.defer_round_commit:
                    scoring_started = self.clock()
                    scoring_claimed = await asyncio.to_thread(
                        self.repository.begin_batch_scoring,
                        manifest.competition_id,
                        claimed.hotkey,
                        claimed.batch_id,
                        owner=self.actor,
                        scoring_timeout_seconds=recorded_scoring_timeout,
                        now=scoring_started,
                    )
                    if not scoring_claimed:
                        raise RuntimeError(
                            "batch execution lease was lost before scoring started"
                        )
                remote_results = await self.item_scorer.score_batch(
                    manifest,
                    output_volume_name=contender.output_volume_name,
                    items=tuple(scoring_items),
                )
                evaluation_by_id = {
                    evaluation.item.evaluation_id: evaluation
                    for evaluation, _ in legacy_scoring_items
                }
                for remote_result in remote_results:
                    evaluation = evaluation_by_id[remote_result.evaluation_id]
                    outcomes.append(
                        AttemptOutcome(
                            evaluation.history_id,
                            remote_result.status,
                            remote_result.reason_code,
                            metrics=(
                                remote_result.metrics.to_scored_item()
                                if remote_result.metrics is not None
                                else None
                            ),
                        )
                    )
            else:
                # Retained as a test seam for task adapters. Production wires a
                # CompetitionScoringClient and never scores media in-process.
                for evaluation, relative_output_path in legacy_scoring_items:
                    try:
                        source_bytes, output_bytes = await asyncio.gather(
                            asyncio.to_thread(
                                self.dataset_store.read_bytes,
                                manifest.evaluation_input_volume_name,
                                evaluation.item.source_path,
                            ),
                            asyncio.to_thread(
                                self.dataset_store.read_bytes,
                                contender.output_volume_name,
                                relative_output_path,
                            ),
                        )
                        metrics = await asyncio.to_thread(
                            self.item_scorer.score_media,
                            manifest,
                            evaluation.item,
                            CompetitionScoringMedia.compression(
                                source_bytes, output_bytes
                            ),
                            runtime_seconds=item_runtime,
                            allocated_gpu_type=allocated_gpu_type,
                            allocated_gpu_count=allocated_gpu_count,
                            allocated_cpu_cores=allocated_cpu_cores,
                        )
                        outcomes.append(
                            AttemptOutcome(
                                evaluation.history_id,
                                "SCORED",
                                metrics=metrics,
                            )
                        )
                    except DatasetError as exc:
                        logger.error(
                            "Competition output Volume read failed; failing item: "
                            "competition_id={} hotkey={} batch_id={} "
                            "evaluation_id={} reason={}",
                            manifest.competition_id,
                            claimed.hotkey,
                            claimed.batch_id,
                            evaluation.item.evaluation_id,
                            LOG_REDACTOR.redact_text(str(exc)),
                        )
                        outcomes.append(
                            AttemptOutcome(
                                evaluation.history_id,
                                "FAILED",
                                "VOLUME_READ_FAILED",
                            )
                        )
                    except ItemScoringError as exc:
                        outcomes.append(
                            AttemptOutcome(
                                evaluation.history_id,
                                "FAILED",
                                exc.reason_code,
                                metrics=exc.metrics,
                            )
                        )
        except SandboxRunnerError as exc:
            outcome_reason = (
                "CONTENDER_OUTPUT_MISSING"
                if exc.reason_code
                in {"SANDBOX_EXEC_FAILED", "SANDBOX_EXEC_TIMEOUT"}
                else exc.reason_code
            )
            logger.error(
                "Competition contender batch request failed terminally: "
                "competition_id={} "
                "hotkey={} batch_id={} runner_reason_code={} "
                "outcome_reason_code={} detail={}",
                manifest.competition_id,
                claimed.hotkey,
                claimed.batch_id,
                exc.reason_code,
                outcome_reason,
                LOG_REDACTOR.redact_text(str(exc)),
            )
            outcomes = [
                AttemptOutcome(
                    evaluation.history_id,
                    "FAILED",
                    outcome_reason,
                )
                for evaluation in claimed.evaluations
            ]
        except DatasetError as exc:
            logger.error(
                "Competition dataset access failed terminally: competition_id={} "
                "hotkey={} batch_id={} reason={}",
                manifest.competition_id,
                claimed.hotkey,
                claimed.batch_id,
                LOG_REDACTOR.redact_text(str(exc)),
            )
            outcomes = [
                AttemptOutcome(
                    evaluation.history_id,
                    "FAILED",
                    "VOLUME_READ_FAILED",
                )
                for evaluation in claimed.evaluations
            ]
        except Exception as exc:
            logger.error(
                "Competition batch infrastructure failure is terminal: "
                "competition_id={} hotkey={} batch_id={} reason={}",
                manifest.competition_id,
                claimed.hotkey,
                claimed.batch_id,
                LOG_REDACTOR.redact_text(str(exc)),
            )
            outcomes = [
                AttemptOutcome(
                    evaluation.history_id,
                    "FAILED",
                    "EVALUATION_INFRASTRUCTURE_ERROR",
                )
                for evaluation in claimed.evaluations
            ]
        finished = self.clock()
        recorded_wall_runtime = (
            batch_wall_runtime
            if batch_wall_runtime is not None
            else time.monotonic() - started
        )
        latest = await asyncio.to_thread(
            self.repository.latest_sandbox,
            manifest.competition_id,
            claimed.hotkey,
        )
        has_allocation = latest is not None and all(
            (
                getattr(latest, "allocated_gpu_type", None),
                getattr(latest, "allocated_gpu_count", None),
                getattr(latest, "allocated_cpu_cores", None),
            )
        )
        batch_cost = (
            estimate_item_cost(
                recorded_wall_runtime,
                allocated_gpu_type=latest.allocated_gpu_type,
                allocated_gpu_count=latest.allocated_gpu_count,
                allocated_cpu_cores=latest.allocated_cpu_cores,
            )
            if has_allocation
            else Decimal("0")
        )
        attribution_method = (
            COST_ATTRIBUTION_METHOD
            if has_allocation
            else f"{COST_ATTRIBUTION_METHOD}_NO_RUNNING_SANDBOX_ALLOCATION"
        )
        deferred_result = DeferredBatchResult(
            claimed=claimed,
            outcomes=tuple(outcomes),
            modal_sandbox_id=modal_sandbox_id,
            wall_runtime_seconds=recorded_wall_runtime,
            batch_estimated_cost_usd=batch_cost,
            cost_attribution_method=attribution_method,
            finished_at=finished,
            scoring_timeout_seconds=recorded_scoring_timeout,
        )
        if not self.defer_round_commit:
            await asyncio.to_thread(
                self.repository.record_batch_outcomes,
                manifest.competition_id,
                claimed.hotkey,
                claimed.batch_id,
                deferred_result.outcomes,
                max_attempts=manifest.max_attempts_per_item,
                modal_sandbox_id=modal_sandbox_id,
                wall_runtime_seconds=recorded_wall_runtime,
                batch_estimated_cost_usd=batch_cost,
                cost_attribution_method=attribution_method,
                now=finished,
                actor=self.actor,
            )
            summary = _persisted_outcome_summary(outcomes)
            logger.info(
                "Competition batch persisted: competition_id={} hotkey={} "
                "batch_id={} scored={} failed={} retry={} reasons={}",
                manifest.competition_id,
                claimed.hotkey,
                claimed.batch_id,
                summary["scored"],
                summary["failed"],
                summary["retry"],
                summary["reasons"],
            )
        else:
            logger.info(
                "Competition batch retained for deferred round commit: "
                "competition_id={} hotkey={} batch_id={} items={}",
                manifest.competition_id,
                claimed.hotkey,
                claimed.batch_id,
                len(outcomes),
            )
        return deferred_result


__all__ = ["CompetitionExecutionCoordinator"]
