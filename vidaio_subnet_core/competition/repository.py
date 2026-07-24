"""Transactional SQLite repository for the competition control plane."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Engine,
    and_,
    create_engine,
    event,
    exists,
    func,
    or_,
    select,
    update,
)
from sqlalchemy.orm import Session, sessionmaker

from .batching import canonical_batch_assignments
from .config import CompetitionManifest, calculate_length_weight
from .migrations import SCHEMA_VERSION, apply_competition_migrations
from .models import (
    Competition,
    CompetitionBatch,
    CompetitionEvent,
    CompetitionEvaluationItem,
    CompetitionHumanReview,
    CompetitionSandbox,
    ContenderMetadata,
    ContenderPerformanceHistory,
)
from .phase0 import SecretRedactor
from .rewards import competition_reward_share
from .state import CompetitionState, ContenderState, assert_transition_allowed
from .timeouts import competition_execution_lease_seconds
from .validation import ValidationReport, ValidationStatus


NONTERMINAL_STATES = tuple(
    state.value for state in CompetitionState if not state.terminal
)
RUNNING_STATES = tuple(
    state.value
    for state in CompetitionState
    if state
    not in {
        CompetitionState.SCHEDULED,
        CompetitionState.COMPLETED,
        CompetitionState.FAILED,
        CompetitionState.CANCELLED,
    }
)
RESUBMITTABLE_PINNED_STATES = frozenset(
    {
        ContenderState.ACCEPTED.value,
        ContenderState.REJECTED.value,
        ContenderState.REVIEW_REQUIRED.value,
    }
)

# Failures owned by validator infrastructure rather than contender scoring. A
# competition must not be finalized when every evaluation ended for only these
# reasons.
INFRASTRUCTURE_FAILURE_REASON_CODES = frozenset(
    {
        "BATCH_EXECUTION_FAILED",
        "BATCH_LEASE_EXPIRED",
        "EVALUATION_INFRASTRUCTURE_ERROR",
        "OUTPUT_VOLUME_VERSION_MISMATCH",
        "SANDBOX_EXEC_FAILED",
        "SANDBOX_RECOVERY_FAILED",
        "SANDBOX_START_FAILED",
        "SCORING_LEASE_EXPIRED",
        "VOLUME_READ_FAILED",
    }
)

READABILITY_REVIEW = "READABILITY_ELIGIBILITY"
DISQUALIFICATION_REVIEW = "DISQUALIFICATION"
EXACT_TIE_REVIEW = "EXACT_TIE_ORDER"


@dataclass(frozen=True)
class ClaimedEvaluation:
    history_id: int
    attempt: int
    item: Any


@dataclass(frozen=True)
class ClaimedBatch:
    batch_id: str
    hotkey: str
    evaluations: tuple[ClaimedEvaluation, ...]


@dataclass(frozen=True)
class AttemptOutcome:
    history_id: int
    status: str
    reason_code: str | None = None
    retryable: bool = False
    metrics: Any = None
    processing_started_at: datetime | None = None
    processing_finished_at: datetime | None = None


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _can_poll_submission(row: ContenderMetadata) -> bool:
    if row.status == ContenderState.PARTICIPATING.value:
        return True
    return bool(
        row.pinned_commit_sha
        and row.status in RESUBMITTABLE_PINNED_STATES
        and row.reason_code != "SUBMISSION_WITHDRAWN"
    )


def _safe_reason_detail(value: str | None) -> str | None:
    if not value:
        return None
    sanitized = "".join(
        character if character.isprintable() else " " for character in value
    )
    return sanitized[:500]


class CompetitionRepository:
    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        if engine is None:
            if not database_url:
                raise ValueError("database_url or engine is required")
            engine = create_engine(database_url)
        self.engine = engine

        @event.listens_for(self.engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        apply_competition_migrations(self.engine)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._redactor = SecretRedactor()

    @property
    def schema_version(self) -> int:
        with self.engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT MAX(version) FROM competition_schema_migrations"
            ).scalar_one()
        return int(row or 0)

    def insert_manifest(
        self,
        manifest: CompetitionManifest,
        *,
        now: datetime,
        actor: str,
    ) -> Competition:
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            existing = session.get(Competition, manifest.competition_id)
            if existing is not None:
                new_digest = manifest.digest()
                if existing.manifest_digest == new_digest:
                    return existing

                current_state = CompetitionState(existing.status)
                if current_state.terminal:
                    raise ValueError(
                        "cannot update the manifest for terminal competition "
                        f"{manifest.competition_id!r} in state {current_state.value}; "
                        "use a new competition_id"
                    )

                normalized_json = manifest.normalized_json()
                old_manifest = json.loads(existing.manifest_json)
                new_manifest = json.loads(normalized_json)
                if (
                    existing.dataset_index_checksum
                    and old_manifest.get("evaluation_batch_size")
                    != new_manifest.get("evaluation_batch_size")
                ):
                    raise ValueError(
                        "cannot change evaluation_batch_size after the evaluation "
                        "dataset has been sealed"
                    )
                changed_fields = {
                    field: {
                        "old": old_manifest.get(field),
                        "new": new_manifest.get(field),
                    }
                    for field in sorted(set(old_manifest) | set(new_manifest))
                    if old_manifest.get(field) != new_manifest.get(field)
                }
                old_digest = existing.manifest_digest
                existing.competition_type = manifest.competition_type
                existing.schema_version = manifest.schema_version
                existing.scoring_version = manifest.scoring_version
                existing.manifest_json = normalized_json
                existing.manifest_digest = new_digest
                existing.start_time = utc_iso(manifest.competition_start_time)
                existing.contender_finalisation_time = utc_iso(
                    manifest.contender_finalisation_time
                )
                existing.human_review_deadline = utc_iso(manifest.human_review_deadline)
                existing.end_time = utc_iso(manifest.competition_end_time)
                existing.input_volume_name = manifest.evaluation_input_volume_name
                existing.reference_volume_name = (
                    manifest.evaluation_reference_volume_name
                    or manifest.evaluation_input_volume_name
                )
                existing.evaluation_index_path = manifest.evaluation_index_path
                existing.source_competition_id = None
                existing.boss_repository_path = (
                    str(manifest.boss.repository_path)
                    if manifest.boss.repository_path is not None
                    else None
                )
                existing.boss_hotkey = manifest.boss.boss_hotkey
                existing.updated_at = now_text
                self._append_event(
                    session,
                    competition_id=manifest.competition_id,
                    event_type="MANIFEST_UPDATED",
                    from_state=None,
                    to_state=None,
                    actor=actor,
                    payload={
                        "competition_state": current_state.value,
                        "old_manifest_digest": old_digest,
                        "new_manifest_digest": new_digest,
                        "changed_fields": changed_fields,
                    },
                    now=now,
                )
                session.flush()
                return existing

            row = Competition(
                competition_id=manifest.competition_id,
                competition_type=manifest.competition_type,
                schema_version=manifest.schema_version,
                scoring_version=manifest.scoring_version,
                manifest_json=manifest.normalized_json(),
                manifest_digest=manifest.digest(),
                status=CompetitionState.SCHEDULED.value,
                start_time=utc_iso(manifest.competition_start_time),
                contender_finalisation_time=utc_iso(
                    manifest.contender_finalisation_time
                ),
                human_review_deadline=utc_iso(manifest.human_review_deadline),
                end_time=utc_iso(manifest.competition_end_time),
                input_volume_name=manifest.evaluation_input_volume_name,
                # The compression source is also the scorer reference. Keep the
                # legacy non-null column pointed at the same immutable Volume.
                reference_volume_name=(
                    manifest.evaluation_reference_volume_name
                    or manifest.evaluation_input_volume_name
                ),
                evaluation_index_path=manifest.evaluation_index_path,
                source_competition_id=None,
                boss_repository_path=(
                    str(manifest.boss.repository_path)
                    if manifest.boss.repository_path is not None
                    else None
                ),
                boss_hotkey=manifest.boss.boss_hotkey,
                created_at=now_text,
                updated_at=now_text,
                state_version=0,
            )
            session.add(row)
            self._append_event(
                session,
                competition_id=manifest.competition_id,
                event_type="MANIFEST_REGISTERED",
                from_state=None,
                to_state=CompetitionState.SCHEDULED,
                actor=actor,
                payload={
                    "manifest_digest": manifest.digest(),
                    "schema_version": manifest.schema_version,
                },
                now=now,
            )
            return row

    def get(self, competition_id: str) -> Competition | None:
        with self._sessions() as session:
            return session.get(Competition, competition_id)

    def list_nonterminal(self) -> list[Competition]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(Competition)
                    .where(Competition.status.in_(NONTERMINAL_STATES))
                    .order_by(Competition.start_time, Competition.competition_id)
                )
            )

    def list_completed_pending_database_backup(self) -> list[Competition]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(Competition)
                    .where(
                        Competition.status == CompetitionState.COMPLETED.value,
                        ~exists(
                            select(1).where(
                                CompetitionEvent.competition_id
                                == Competition.competition_id,
                                CompetitionEvent.event_type
                                == "COMPETITION_DATABASE_BACKED_UP",
                            )
                        ),
                    )
                    .order_by(Competition.end_time, Competition.competition_id)
                )
            )

    def latest_competition_reward_recipients(
        self,
    ) -> tuple[tuple[str, int, float], ...]:
        """Return the latest completed competition's ranked podium."""

        return self.latest_competition_reward_context()[1]

    def latest_competition_reward_context(
        self,
    ) -> tuple[str | None, tuple[tuple[str, int, float], ...]]:
        """Return the selected competition ID and its ranked podium atomically."""

        with self._sessions() as session:
            competition = session.scalar(
                select(Competition)
                .where(
                    Competition.status == CompetitionState.COMPLETED.value,
                    Competition.winner_hotkey.is_not(None),
                )
                .order_by(
                    Competition.finalized_at.desc(),
                    Competition.competition_id.desc(),
                )
                .limit(1)
            )
            if competition is None:
                return None, ()
            rows = session.scalars(
                select(ContenderMetadata)
                .where(
                    ContenderMetadata.competition_id == competition.competition_id,
                    ContenderMetadata.eligible.is_(True),
                    ContenderMetadata.final_rank.between(1, 3),
                )
                .order_by(ContenderMetadata.final_rank)
            )
            return (
                competition.competition_id,
                tuple(
                    (
                        (competition.boss_hotkey if row.is_boss else row.hotkey),
                        int(row.final_rank),
                        competition_reward_share(int(row.final_rank)),
                    )
                    for row in rows
                ),
            )

    def seal_evaluation_dataset(
        self,
        competition_id: str,
        evaluation_index: Any,
        *,
        now: datetime,
        actor: str,
    ) -> str:
        """Persist an immutable normalized index before any evaluation dispatch."""

        digest = evaluation_index.digest()
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            if evaluation_index.competition_id != competition_id:
                raise ValueError("evaluation index competition mismatch")
            if competition.dataset_index_checksum:
                if competition.dataset_index_checksum != digest:
                    raise ValueError("sealed evaluation index is immutable")
                return digest
            existing_count = session.scalar(
                select(func.count(CompetitionEvaluationItem.id)).where(
                    CompetitionEvaluationItem.competition_id == competition_id
                )
            )
            if existing_count:
                raise ValueError("evaluation items exist without a sealed index")
            manifest = CompetitionManifest.model_validate_json(
                competition.manifest_json
            )
            assignments = canonical_batch_assignments(
                (
                    (item.evaluation_id, item.source_path)
                    for item in evaluation_index.items
                ),
                manifest.evaluation_batch_size,
            )
            canonical_batch_count = (
                max(
                    batch_index
                    for batch_index, _position in assignments.values()
                )
                + 1
            )
            for item in evaluation_index.items:
                batch_index, batch_position = assignments[item.evaluation_id]
                session.add(
                    CompetitionEvaluationItem(
                        competition_id=competition_id,
                        evaluation_id=item.evaluation_id,
                        normalized_json=item.model_dump_json(),
                        canonical_batch_index=batch_index,
                        canonical_batch_position=batch_position,
                        status="SEALED",
                        dispatch_status="PENDING",
                        score_status="PENDING",
                        current_attempt=0,
                        checksum=item.sha256,
                        duration_seconds=item.duration_seconds,
                        created_at=now_text,
                        updated_at=now_text,
                    )
                )
            competition.dataset_index_checksum = digest
            competition.input_volume_checksum = hashlib.sha256(
                "".join(item.sha256 for item in evaluation_index.items).encode("ascii")
            ).hexdigest()
            competition.updated_at = now_text
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="EVALUATION_DATASET_SEALED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "index_checksum": digest,
                    "item_count": len(evaluation_index.items),
                    "canonical_batch_count": canonical_batch_count,
                    "canonical_batch_size": manifest.evaluation_batch_size,
                },
                now=now,
            )
        return digest

    def evaluation_items(self, competition_id: str) -> list[CompetitionEvaluationItem]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(CompetitionEvaluationItem)
                    .where(CompetitionEvaluationItem.competition_id == competition_id)
                    .order_by(
                        CompetitionEvaluationItem.canonical_batch_index,
                        CompetitionEvaluationItem.canonical_batch_position,
                    )
                )
            )

    def claim_evaluation_batch(
        self,
        competition_id: str,
        hotkey: str,
        *,
        owner: str,
        max_items: int,
        max_attempts: int,
        lease_seconds: int | None = None,
        minimum_execution_timeout_seconds: float = 0,
        scoring_version: str,
        vmaf_threshold: float,
        max_video_length_seconds: float,
        length_weight_exponent: float,
        now: datetime,
    ) -> ClaimedBatch | None:
        from .dataset import parse_evaluation_index_item_json

        # Retained in the repository API for manifest compatibility. Evaluation
        # failures are terminal; only the audited repair workflow can requeue.
        del max_attempts
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            contender = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if contender is None or contender.manual_disqualified:
                return None
            expired_batches = list(
                session.scalars(
                    select(CompetitionBatch).where(
                        CompetitionBatch.competition_id == competition_id,
                        CompetitionBatch.hotkey == hotkey,
                        or_(
                            and_(
                                CompetitionBatch.status == "RUNNING",
                                CompetitionBatch.lease_expires_at <= now_text,
                            ),
                            and_(
                                CompetitionBatch.status == "SCORING",
                                CompetitionBatch.scoring_expires_at <= now_text,
                            ),
                        ),
                    )
                )
            )
            for batch in expired_batches:
                expired_status = batch.status
                batch.status = "FAILED"
                batch.updated_at = now_text
                for history in session.scalars(
                    select(ContenderPerformanceHistory).where(
                        ContenderPerformanceHistory.competition_id == competition_id,
                        ContenderPerformanceHistory.hotkey == hotkey,
                        ContenderPerformanceHistory.batch_id == batch.batch_id,
                        ContenderPerformanceHistory.status == "RUNNING",
                    )
                ):
                    history.status = "FAILED"
                    history.reason_code = (
                        "SCORING_LEASE_EXPIRED"
                        if expired_status == "SCORING"
                        else "BATCH_LEASE_EXPIRED"
                    )
                    history.updated_at = now_text

            items = list(
                session.scalars(
                    select(CompetitionEvaluationItem)
                    .where(CompetitionEvaluationItem.competition_id == competition_id)
                    .order_by(
                        CompetitionEvaluationItem.canonical_batch_index,
                        CompetitionEvaluationItem.canonical_batch_position,
                    )
                )
            )
            selected: list[tuple[CompetitionEvaluationItem, Any, int]] = []
            selected_batch_index: int | None = None
            canonical_batches: dict[int, list[CompetitionEvaluationItem]] = {}
            for item_row in items:
                canonical_batches.setdefault(
                    item_row.canonical_batch_index, []
                ).append(item_row)

            for canonical_batch_index, batch_items in canonical_batches.items():
                if len(batch_items) > max_items:
                    raise ValueError(
                        f"canonical batch {canonical_batch_index} contains "
                        f"{len(batch_items)} items but max_items is {max_items}"
                    )
                batch_has_running_attempt = False
                batch_selection: list[
                    tuple[CompetitionEvaluationItem, Any, int]
                ] = []
                for item_row in batch_items:
                    attempts = list(
                        session.scalars(
                            select(ContenderPerformanceHistory)
                            .where(
                                ContenderPerformanceHistory.competition_id
                                == competition_id,
                                ContenderPerformanceHistory.hotkey == hotkey,
                                ContenderPerformanceHistory.evaluation_id
                                == item_row.evaluation_id,
                            )
                            .order_by(ContenderPerformanceHistory.attempt)
                        )
                    )
                    if any(row.status in {"SCORED", "FAILED"} for row in attempts):
                        continue
                    if any(row.status == "RUNNING" for row in attempts):
                        batch_has_running_attempt = True
                        continue
                    legacy_retries = [
                        row for row in attempts if row.status == "RETRY"
                    ]
                    if legacy_retries:
                        for row in legacy_retries:
                            row.status = "FAILED"
                            row.reason_code = (
                                row.reason_code or "AUTOMATIC_RETRY_DISABLED"
                            )
                            row.updated_at = now_text
                        continue
                    item = parse_evaluation_index_item_json(item_row.normalized_json)
                    batch_selection.append(
                        (
                            item_row,
                            item,
                            max((row.attempt for row in attempts), default=0) + 1,
                        )
                    )
                if batch_has_running_attempt:
                    return None
                if batch_selection:
                    selected = batch_selection
                    selected_batch_index = canonical_batch_index
                    break
            if not selected:
                return None
            if selected_batch_index is None:
                raise RuntimeError("selected canonical batch has no index")

            if lease_seconds is None:
                lease_seconds = competition_execution_lease_seconds(
                    (item for _, item, _ in selected),
                    minimum_timeout_seconds=minimum_execution_timeout_seconds,
                )
            lease_expiry = utc_iso(now + timedelta(seconds=lease_seconds))
            seed = "|".join(
                [
                    competition_id,
                    hotkey,
                    now_text,
                    f"canonical-batch:{selected_batch_index}",
                ]
                + [f"{row.evaluation_id}:{attempt}" for row, _, attempt in selected]
            )
            batch_id = "batch-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
            batch = CompetitionBatch(
                competition_id=competition_id,
                hotkey=hotkey,
                batch_id=batch_id,
                canonical_batch_index=selected_batch_index,
                status="RUNNING",
                lease_owner=owner,
                lease_expires_at=lease_expiry,
                timeout_seconds=float(lease_seconds),
                reconciliation_status="PENDING",
                created_at=now_text,
                updated_at=now_text,
            )
            session.add(batch)
            session.flush()
            claimed = []
            for item_row, item, attempt in selected:
                weight = calculate_length_weight(
                    item.duration_seconds,
                    max_video_length_seconds,
                    length_weight_exponent,
                )
                history = ContenderPerformanceHistory(
                    competition_id=competition_id,
                    hotkey=hotkey,
                    evaluation_id=item.evaluation_id,
                    batch_id=batch_id,
                    canonical_batch_index=selected_batch_index,
                    attempt=attempt,
                    idempotency_key=(
                        f"{competition_id}:{hotkey}:{item.evaluation_id}:{attempt}"
                    ),
                    input_checksum=item.sha256,
                    input_size_bytes=item.size_bytes,
                    status="RUNNING",
                    task_type="COMPRESSION",
                    scale_factor=1,
                    duration_seconds=item.duration_seconds,
                    length_weight=weight,
                    vmaf_threshold=float(
                        getattr(item, "vmaf_threshold", vmaf_threshold)
                    ),
                    quality_component=0,
                    cost_efficiency_component=0,
                    completion_value=0,
                    currency="USD",
                    scoring_version=scoring_version,
                    created_at=now_text,
                    updated_at=now_text,
                )
                session.add(history)
                session.flush()
                claimed.append(ClaimedEvaluation(history.id, attempt, item))
                item_row.current_attempt = max(item_row.current_attempt, attempt)
                item_row.dispatch_status = "RUNNING"
                item_row.updated_at = now_text
            return ClaimedBatch(batch_id, hotkey, tuple(claimed))

    def begin_batch_scoring(
        self,
        competition_id: str,
        hotkey: str,
        batch_id: str,
        *,
        owner: str,
        scoring_timeout_seconds: float,
        now: datetime,
    ) -> bool:
        """End the Sandbox lease and begin a separately timed scoring phase."""

        if scoring_timeout_seconds <= 0:
            raise ValueError("scoring_timeout_seconds must be positive")
        now_text = utc_iso(now)
        scoring_expiry = utc_iso(
            now + timedelta(seconds=float(scoring_timeout_seconds))
        )
        with self._sessions.begin() as session:
            batch = session.scalar(
                select(CompetitionBatch).where(
                    CompetitionBatch.competition_id == competition_id,
                    CompetitionBatch.hotkey == hotkey,
                    CompetitionBatch.batch_id == batch_id,
                )
            )
            if batch is None or batch.status != "RUNNING" or batch.lease_owner != owner:
                return False
            batch.status = "SCORING"
            batch.scoring_timeout_seconds = float(scoring_timeout_seconds)
            batch.scoring_expires_at = scoring_expiry
            batch.lease_owner = None
            batch.lease_expires_at = None
            batch.updated_at = now_text
            return True

    def record_batch_outcomes(
        self,
        competition_id: str,
        hotkey: str,
        batch_id: str,
        outcomes: tuple[AttemptOutcome, ...],
        *,
        max_attempts: int,
        modal_sandbox_id: str | None,
        wall_runtime_seconds: float,
        batch_estimated_cost_usd: Decimal | None = None,
        cost_attribution_method: str | None = None,
        now: datetime,
        actor: str,
    ) -> None:
        # Retained in the repository API for manifest compatibility. Automatic
        # evaluation retry scheduling is deliberately disabled.
        del max_attempts
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            batch = session.scalar(
                select(CompetitionBatch).where(
                    CompetitionBatch.competition_id == competition_id,
                    CompetitionBatch.hotkey == hotkey,
                    CompetitionBatch.batch_id == batch_id,
                )
            )
            if batch is None:
                raise KeyError(batch_id)
            # Late or duplicate deliveries must never revive an expired batch
            # or overwrite a result already accepted for a newer attempt.
            if batch.status not in {"RUNNING", "SCORING"}:
                return
            expected_ids = set(
                session.scalars(
                    select(ContenderPerformanceHistory.id).where(
                        ContenderPerformanceHistory.competition_id == competition_id,
                        ContenderPerformanceHistory.hotkey == hotkey,
                        ContenderPerformanceHistory.batch_id == batch_id,
                        ContenderPerformanceHistory.status == "RUNNING",
                    )
                )
            )
            outcome_ids = {outcome.history_id for outcome in outcomes}
            if outcome_ids != expected_ids or len(outcome_ids) != len(outcomes):
                raise ValueError("batch outcomes do not exactly match the active claim")
            if batch_estimated_cost_usd is None:
                batch_estimated_cost_usd = sum(
                    (
                        Decimal(str(outcome.metrics.estimated_cost_usd))
                        for outcome in outcomes
                        if outcome.metrics is not None
                    ),
                    Decimal(0),
                )
            item_runtime = wall_runtime_seconds / len(outcomes)
            item_cost = batch_estimated_cost_usd / len(outcomes)
            for outcome in outcomes:
                if outcome.status not in {"SCORED", "FAILED"}:
                    raise ValueError(f"invalid evaluation outcome {outcome.status!r}")
                if outcome.retryable:
                    raise ValueError("automatic evaluation retries are disabled")
                history = session.get(ContenderPerformanceHistory, outcome.history_id)
                if history is None or history.batch_id != batch_id:
                    raise KeyError(outcome.history_id)
                if history.canonical_batch_index != batch.canonical_batch_index:
                    raise RuntimeError(
                        "performance history canonical batch does not match "
                        f"dispatch batch {batch_id}"
                    )
                if history.status in {"SCORED", "FAILED"}:
                    continue
                metrics = outcome.metrics
                if metrics is not None:
                    history.input_checksum = metrics.input_checksum
                    history.output_checksum = metrics.output_checksum
                    history.input_size_bytes = metrics.input_size_bytes
                    history.output_size_bytes = metrics.output_size_bytes
                    history.vmaf_score = metrics.vmaf_score
                    history.compression_rate = metrics.compression_ratio
                    history.media_score = metrics.media_score
                    history.compression_score = metrics.media_compression_component
                    history.media_compression_component = (
                        metrics.media_compression_component
                    )
                    history.media_vmaf_component = metrics.media_vmaf_component
                    history.media_score_reason = metrics.media_score_reason
                history.handler_runtime_seconds = item_runtime
                history.estimated_cost_usd = item_cost
                history.raw_cost_usd = item_cost
                history.cost_attribution_method = cost_attribution_method or (
                    metrics.cost_attribution_method if metrics is not None else None
                )
                if outcome.status == "SCORED" and metrics is not None:
                    history.status = "SCORED"
                    history.completion_value = 1
                elif outcome.status == "SCORED":
                    raise ValueError("scored evaluation outcome requires metrics")
                else:
                    history.status = "FAILED"
                history.reason_code = outcome.reason_code
                history.processing_started_at = (
                    utc_iso(outcome.processing_started_at)
                    if outcome.processing_started_at
                    else None
                )
                history.processing_finished_at = (
                    utc_iso(outcome.processing_finished_at)
                    if outcome.processing_finished_at
                    else None
                )
                history.modal_sandbox_id = modal_sandbox_id
                history.updated_at = now_text
            batch.status = "COMPLETED"
            batch.lease_owner = None
            batch.lease_expires_at = None
            batch.scoring_expires_at = None
            batch.modal_sandbox_id = modal_sandbox_id
            batch.wall_runtime_seconds = wall_runtime_seconds
            batch.active_runtime_seconds = wall_runtime_seconds
            batch.estimated_cost_usd = batch_estimated_cost_usd
            batch.updated_at = now_text
            self._refresh_contender_counts(session, competition_id, hotkey, now_text)
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="EVALUATION_BATCH_RECORDED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": hotkey,
                    "batch_id": batch_id,
                    "status": batch.status,
                    "item_count": len(outcomes),
                },
                now=now,
            )

    def _refresh_contender_counts(
        self, session: Session, competition_id: str, hotkey: str, now_text: str
    ) -> None:
        contender = session.scalar(
            select(ContenderMetadata).where(
                ContenderMetadata.competition_id == competition_id,
                ContenderMetadata.hotkey == hotkey,
            )
        )
        if contender is None:
            return
        rows = list(
            session.scalars(
                select(ContenderPerformanceHistory).where(
                    ContenderPerformanceHistory.competition_id == competition_id,
                    ContenderPerformanceHistory.hotkey == hotkey,
                )
            )
        )
        terminal = {
            row.evaluation_id: row
            for row in sorted(rows, key=lambda value: value.attempt)
            if row.status in {"SCORED", "FAILED"}
        }
        contender.successful_items = sum(
            row.status == "SCORED" for row in terminal.values()
        )
        contender.failed_items = sum(
            row.status == "FAILED" for row in terminal.values()
        )
        item_count = (
            session.scalar(
                select(func.count(CompetitionEvaluationItem.id)).where(
                    CompetitionEvaluationItem.competition_id == competition_id
                )
            )
            or 0
        )
        contender.pending_items = max(
            0, int(item_count) - contender.successful_items - contender.failed_items
        )
        contender.updated_at = now_text

    def evaluation_is_complete(
        self, competition_id: str, accepted_build_statuses: frozenset[str]
    ) -> bool:
        with self._sessions() as session:
            item_count = int(
                session.scalar(
                    select(func.count(CompetitionEvaluationItem.id)).where(
                        CompetitionEvaluationItem.competition_id == competition_id
                    )
                )
                or 0
            )
            if item_count == 0:
                return False
            contenders = list(
                session.scalars(
                    select(ContenderMetadata).where(
                        ContenderMetadata.competition_id == competition_id,
                        ContenderMetadata.build_status.in_(accepted_build_statuses),
                        ContenderMetadata.manual_disqualified.is_(False),
                    )
                )
            )
            if not contenders:
                return False
            for contender in contenders:
                terminal_ids = set(
                    session.scalars(
                        select(ContenderPerformanceHistory.evaluation_id).where(
                            ContenderPerformanceHistory.competition_id
                            == competition_id,
                            ContenderPerformanceHistory.hotkey == contender.hotkey,
                            ContenderPerformanceHistory.status.in_(
                                ("SCORED", "FAILED")
                            ),
                        )
                    )
                )
                if len(terminal_ids) != item_count:
                    return False
            return True

    def contender_evaluation_is_complete(
        self, competition_id: str, hotkey: str
    ) -> bool:
        """Return whether one contender has a terminal result for every item."""

        with self._sessions() as session:
            item_count = int(
                session.scalar(
                    select(func.count(CompetitionEvaluationItem.id)).where(
                        CompetitionEvaluationItem.competition_id == competition_id
                    )
                )
                or 0
            )
            if item_count == 0:
                return False
            terminal_ids = set(
                session.scalars(
                    select(ContenderPerformanceHistory.evaluation_id).where(
                        ContenderPerformanceHistory.competition_id == competition_id,
                        ContenderPerformanceHistory.hotkey == hotkey,
                        ContenderPerformanceHistory.status.in_(("SCORED", "FAILED")),
                    )
                )
            )
            return len(terminal_ids) == item_count

    def evaluation_infrastructure_blocker(
        self, competition_id: str, accepted_build_statuses: frozenset[str]
    ) -> dict[str, Any] | None:
        """Describe an all-infrastructure terminal result, if one exists."""

        with self._sessions() as session:
            item_count = int(
                session.scalar(
                    select(func.count(CompetitionEvaluationItem.id)).where(
                        CompetitionEvaluationItem.competition_id == competition_id
                    )
                )
                or 0
            )
            hotkeys = list(
                session.scalars(
                    select(ContenderMetadata.hotkey).where(
                        ContenderMetadata.competition_id == competition_id,
                        ContenderMetadata.build_status.in_(accepted_build_statuses),
                        ContenderMetadata.manual_disqualified.is_(False),
                    )
                )
            )
            if item_count == 0 or not hotkeys:
                return None
            rows = list(
                session.scalars(
                    select(ContenderPerformanceHistory)
                    .where(
                        ContenderPerformanceHistory.competition_id == competition_id,
                        ContenderPerformanceHistory.hotkey.in_(hotkeys),
                        ContenderPerformanceHistory.status.in_(("SCORED", "FAILED")),
                    )
                    .order_by(ContenderPerformanceHistory.attempt)
                )
            )
            latest = {(row.hotkey, row.evaluation_id): row for row in rows}
            if len(latest) != item_count * len(hotkeys):
                return None
            terminal = list(latest.values())
            if any(row.status == "SCORED" for row in terminal):
                return None
            reasons: dict[str, int] = {}
            for row in terminal:
                reason = row.reason_code or "UNKNOWN"
                if reason not in INFRASTRUCTURE_FAILURE_REASON_CODES:
                    return None
                reasons[reason] = reasons.get(reason, 0) + 1
            return {
                "failed_items": len(terminal),
                "reasons": dict(sorted(reasons.items())),
            }

    def requeue_infrastructure_failures(
        self,
        competition_id: str,
        *,
        reason_codes: frozenset[str] = INFRASTRUCTURE_FAILURE_REASON_CODES,
        now: datetime,
        actor: str,
    ) -> dict[str, Any]:
        """Auditably reset terminal validator-infrastructure attempts.

        Attempt rows are retained as REQUEUED and future attempts continue with
        monotonically increasing attempt numbers, preserving the full history.
        """

        if not reason_codes:
            raise ValueError("at least one infrastructure reason code is required")
        unsupported = reason_codes - INFRASTRUCTURE_FAILURE_REASON_CODES
        if unsupported:
            raise ValueError(
                "refusing to requeue non-infrastructure reason code(s): "
                + ", ".join(sorted(unsupported))
            )
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            current = CompetitionState(competition.status)
            if current not in {
                CompetitionState.EVALUATING,
                CompetitionState.AWAITING_END_TIME,
            }:
                raise ValueError(
                    "infrastructure attempts can only be requeued while EVALUATING "
                    f"or AWAITING_END_TIME, not {current.value}"
                )
            terminal_failures = list(
                session.scalars(
                    select(ContenderPerformanceHistory).where(
                        ContenderPerformanceHistory.competition_id == competition_id,
                        ContenderPerformanceHistory.status == "FAILED",
                        ContenderPerformanceHistory.reason_code.in_(reason_codes),
                    )
                )
            )
            targets = sorted(
                {(row.hotkey, row.evaluation_id) for row in terminal_failures}
            )
            if not targets:
                return {
                    "requeued_evaluations": 0,
                    "requeued_attempts": 0,
                    "reason_codes": sorted(reason_codes),
                    "state": current.value,
                }

            requeued_attempts = 0
            batch_ids: set[str] = set()
            hotkeys: set[str] = set()
            for hotkey, evaluation_id in targets:
                histories = list(
                    session.scalars(
                        select(ContenderPerformanceHistory).where(
                            ContenderPerformanceHistory.competition_id
                            == competition_id,
                            ContenderPerformanceHistory.hotkey == hotkey,
                            ContenderPerformanceHistory.evaluation_id == evaluation_id,
                        )
                    )
                )
                if any(row.status == "SCORED" for row in histories):
                    raise RuntimeError(
                        "refusing to requeue an evaluation that already has a score: "
                        f"{hotkey}/{evaluation_id}"
                    )
                for row in histories:
                    if row.status in {"FAILED", "RETRY"}:
                        row.status = "REQUEUED"
                        row.updated_at = now_text
                        batch_ids.add(row.batch_id)
                        requeued_attempts += 1
                hotkeys.add(hotkey)

            for batch in session.scalars(
                select(CompetitionBatch).where(
                    CompetitionBatch.competition_id == competition_id,
                    CompetitionBatch.batch_id.in_(batch_ids),
                )
            ):
                batch.status = "REQUEUED"
                batch.updated_at = now_text

            for item in session.scalars(
                select(CompetitionEvaluationItem).where(
                    CompetitionEvaluationItem.competition_id == competition_id
                )
            ):
                item.status = "SEALED"
                item.dispatch_status = "PENDING"
                item.score_status = "PENDING"
                item.updated_at = now_text

            contenders_to_reset = list(
                session.scalars(
                    select(ContenderMetadata).where(
                        ContenderMetadata.competition_id == competition_id,
                        or_(
                            ContenderMetadata.hotkey.in_(hotkeys),
                            ContenderMetadata.status == ContenderState.SCORED.value,
                        ),
                    )
                )
            )
            for contender in contenders_to_reset:
                contender.status = ContenderState.BUILT.value
                contender.final_score = None
                contender.average_vmaf = None
                contender.average_compression_rate = None
                contender.media_score_aggregate = None
                contender.quality_aggregate = None
                contender.cost_efficiency_aggregate = None
                contender.length_coverage = None
                contender.estimated_cost_usd = None
                contender.reconciled_cost_usd = None
                contender.active_runtime_seconds = None
                contender.cold_start_runtime_seconds = None
                contender.final_rank = None
                contender.eligible = None
                contender.updated_at = now_text
                self._refresh_contender_counts(
                    session, competition_id, contender.hotkey, now_text
                )

            if current == CompetitionState.AWAITING_END_TIME:
                previous_version = competition.state_version
                result = session.execute(
                    update(Competition)
                    .where(
                        Competition.competition_id == competition_id,
                        Competition.status == current.value,
                        Competition.state_version == previous_version,
                    )
                    .values(
                        status=CompetitionState.EVALUATING.value,
                        status_reason="validator infrastructure attempts requeued",
                        state_version=previous_version + 1,
                        updated_at=now_text,
                    )
                )
                if result.rowcount != 1:
                    raise RuntimeError("competition state changed concurrently")

            payload = {
                "requeued_evaluations": len(targets),
                "requeued_attempts": requeued_attempts,
                "reason_codes": sorted(reason_codes),
            }
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="EVALUATION_INFRASTRUCTURE_REQUEUED",
                from_state=current,
                to_state=CompetitionState.EVALUATING,
                actor=actor,
                payload=payload,
                now=now,
            )
            return {**payload, "state": CompetitionState.EVALUATING.value}

    def competition_scoring_rows(self, competition_id: str) -> tuple[dict, ...]:
        """Return terminal attempts for the current eligible normalization set."""

        with self._sessions() as session:
            rows = list(
                session.scalars(
                    select(ContenderPerformanceHistory)
                    .join(
                        ContenderMetadata,
                        and_(
                            ContenderMetadata.competition_id
                            == ContenderPerformanceHistory.competition_id,
                            ContenderMetadata.hotkey
                            == ContenderPerformanceHistory.hotkey,
                        ),
                    )
                    .where(
                        ContenderPerformanceHistory.competition_id == competition_id,
                        ContenderPerformanceHistory.status.in_(("SCORED", "FAILED")),
                        ContenderMetadata.manual_disqualified.is_(False),
                    )
                    .order_by(
                        ContenderPerformanceHistory.hotkey,
                        ContenderPerformanceHistory.evaluation_id,
                        ContenderPerformanceHistory.attempt,
                    )
                )
            )
            latest = {}
            for row in rows:
                latest[(row.hotkey, row.evaluation_id)] = row
            return tuple(
                {
                    "id": row.id,
                    "hotkey": row.hotkey,
                    "evaluation_id": row.evaluation_id,
                    "status": row.status,
                    "length_weight": row.length_weight,
                    "vmaf_threshold": row.vmaf_threshold,
                    "vmaf_score": row.vmaf_score,
                    "compression_rate": row.compression_rate,
                    "media_score": row.media_score,
                    "media_compression_component": (
                        row.media_compression_component
                    ),
                    "media_vmaf_component": row.media_vmaf_component,
                    "estimated_cost_usd": row.estimated_cost_usd,
                    "reconciled_cost_usd": row.reconciled_cost_usd,
                }
                for row in latest.values()
            )

    def persist_competition_scores(
        self,
        competition_id: str,
        manifest: CompetitionManifest,
        *,
        now: datetime,
        actor: str,
        aggregates,
        components: dict[int, tuple[float, float, float, float, float]],
    ) -> None:
        """Persist already-computed results from the remote scoring worker."""

        self.score_competition(
            competition_id,
            manifest,
            now=now,
            actor=actor,
            aggregates=aggregates,
            components=components,
        )

    def score_competition(
        self,
        competition_id: str,
        manifest: CompetitionManifest,
        *,
        now: datetime,
        actor: str,
        aggregates=None,
        components: dict[int, tuple[float, float, float, float, float]] | None = None,
        event_type: str = "COMPETITION_SCORES_COMPUTED",
    ) -> None:
        if (aggregates is None) != (components is None):
            raise ValueError("aggregates and components must be supplied together")

        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            rows = list(
                session.scalars(
                    select(ContenderPerformanceHistory)
                    .join(
                        ContenderMetadata,
                        and_(
                            ContenderMetadata.competition_id
                            == ContenderPerformanceHistory.competition_id,
                            ContenderMetadata.hotkey
                            == ContenderPerformanceHistory.hotkey,
                        ),
                    )
                    .where(
                        ContenderPerformanceHistory.competition_id == competition_id,
                        ContenderPerformanceHistory.status.in_(("SCORED", "FAILED")),
                        ContenderMetadata.manual_disqualified.is_(False),
                    )
                    .order_by(
                        ContenderPerformanceHistory.hotkey,
                        ContenderPerformanceHistory.evaluation_id,
                        ContenderPerformanceHistory.attempt,
                    )
                )
            )
            latest = {}
            for row in rows:
                latest[(row.hotkey, row.evaluation_id)] = row
            if aggregates is None:
                # Backward-compatible test/admin seam. Production competition
                # execution supplies results from the remote scoring worker.
                from .scoring import compute_aggregates

                aggregates, components = compute_aggregates(manifest, latest.values())
            else:
                expected_history_ids = {row.id for row in latest.values()}
                if set(components) != expected_history_ids:
                    raise ValueError(
                        "remote scoring components do not match persisted histories"
                    )
                expected_hotkeys = {row.hotkey for row in latest.values()}
                if {value.hotkey for value in aggregates} != expected_hotkeys:
                    raise ValueError(
                        "remote scoring aggregates do not match persisted contenders"
                    )
            usage_by_hotkey: dict[str, tuple[Decimal, float]] = {}
            for batch in session.scalars(
                select(CompetitionBatch).where(
                    CompetitionBatch.competition_id == competition_id,
                    CompetitionBatch.estimated_cost_usd.is_not(None),
                )
            ):
                cost, runtime = usage_by_hotkey.get(batch.hotkey, (Decimal(0), 0.0))
                usage_by_hotkey[batch.hotkey] = (
                    cost + Decimal(str(batch.estimated_cost_usd or 0)),
                    runtime + float(batch.active_runtime_seconds or 0),
                )
            disqualified = list(
                session.scalars(
                    select(ContenderMetadata).where(
                        ContenderMetadata.competition_id == competition_id,
                        ContenderMetadata.manual_disqualified.is_(True),
                    )
                )
            )
            for contender in disqualified:
                contender.final_score = None
                contender.media_score_aggregate = None
                contender.quality_aggregate = None
                contender.cost_efficiency_aggregate = None
                contender.length_coverage = None
                contender.final_rank = None
                contender.eligible = False
                contender.updated_at = now_text
            for row in latest.values():
                (
                    media_score,
                    compression_component,
                    vmaf_component,
                    cost,
                    completion,
                ) = components[row.id]
                row.media_score = media_score
                row.compression_score = compression_component
                row.media_compression_component = compression_component
                row.media_vmaf_component = vmaf_component
                row.quality_component = media_score
                row.cost_efficiency_component = cost
                row.completion_value = completion
                row.updated_at = now_text
            for aggregate in aggregates:
                contender = session.scalar(
                    select(ContenderMetadata).where(
                        ContenderMetadata.competition_id == competition_id,
                        ContenderMetadata.hotkey == aggregate.hotkey,
                    )
                )
                if contender is None:
                    raise KeyError((competition_id, aggregate.hotkey))
                contender.media_score_aggregate = aggregate.media_score_aggregate
                contender.quality_aggregate = aggregate.media_score_aggregate
                contender.cost_efficiency_aggregate = (
                    aggregate.cost_efficiency_aggregate
                )
                contender.length_coverage = aggregate.length_coverage
                contender.final_score = aggregate.final_score
                contender.average_vmaf = aggregate.average_vmaf
                contender.average_compression_rate = aggregate.average_compression_ratio
                usage_cost, usage_runtime = usage_by_hotkey.get(
                    aggregate.hotkey, (aggregate.estimated_cost_usd, 0.0)
                )
                contender.estimated_cost_usd = usage_cost
                contender.active_runtime_seconds = usage_runtime
                contender.successful_items = aggregate.successful_items
                contender.failed_items = aggregate.failed_items
                contender.pending_items = 0
                contender.status = ContenderState.SCORED.value
                contender.updated_at = now_text
            for item in session.scalars(
                select(CompetitionEvaluationItem).where(
                    CompetitionEvaluationItem.competition_id == competition_id
                )
            ):
                item.status = "SCORED"
                item.dispatch_status = "COMPLETED"
                item.score_status = "SCORED"
                item.updated_at = now_text
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            competition.scores_need_recalculation = False
            competition.updated_at = now_text
            self._append_event(
                session,
                competition_id=competition_id,
                event_type=event_type,
                from_state=CompetitionState(competition.status),
                to_state=None,
                actor=actor,
                payload={
                    "contender_count": len(aggregates),
                    "scoring_version": manifest.scoring_version,
                },
                now=now,
            )

    def recalculate_competition_scores(
        self,
        competition_id: str,
        *,
        now: datetime,
        actor: str,
    ) -> None:
        """Rebuild derived scores using persisted SQL history only.

        This path never accesses media, invokes a scoring service, or dispatches
        contender/Modal work. Terminal history already contains every raw input
        needed by ``compute_aggregates``.
        """

        competition = self.get(competition_id)
        if competition is None:
            raise KeyError(competition_id)
        if CompetitionState(competition.status) != CompetitionState.AWAITING_END_TIME:
            raise ValueError(
                "score recalculation is only allowed while awaiting the end time"
            )
        manifest = CompetitionManifest.model_validate_json(competition.manifest_json)
        self.score_competition(
            competition_id,
            manifest,
            now=now,
            actor=actor,
            event_type="COMPETITION_SCORES_RECALCULATED",
        )

    def another_competition_is_running(self, competition_id: str) -> bool:
        with self._sessions() as session:
            return (
                session.scalar(
                    select(Competition.competition_id)
                    .where(
                        Competition.competition_id != competition_id,
                        Competition.status.in_(RUNNING_STATES),
                    )
                    .limit(1)
                )
                is not None
            )

    def acquire_scheduler_lease(
        self,
        competition_id: str,
        *,
        owner: str,
        now: datetime,
        ttl_seconds: int,
    ) -> bool:
        now_text = utc_iso(now)
        expiry = utc_iso(now + timedelta(seconds=ttl_seconds))
        with self.engine.begin() as connection:
            result = connection.execute(
                update(Competition)
                .where(
                    Competition.competition_id == competition_id,
                    or_(
                        Competition.scheduler_lease_owner.is_(None),
                        Competition.scheduler_lease_owner == owner,
                        Competition.scheduler_lease_expires_at.is_(None),
                        Competition.scheduler_lease_expires_at <= now_text,
                    ),
                )
                .values(
                    scheduler_lease_owner=owner,
                    scheduler_lease_expires_at=expiry,
                    updated_at=now_text,
                )
            )
        return result.rowcount == 1

    def release_scheduler_lease(self, competition_id: str, *, owner: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(Competition)
                .where(
                    Competition.competition_id == competition_id,
                    Competition.scheduler_lease_owner == owner,
                )
                .values(scheduler_lease_owner=None, scheduler_lease_expires_at=None)
            )

    def transition(
        self,
        competition_id: str,
        target: CompetitionState,
        *,
        now: datetime,
        actor: str,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
        expected: CompetitionState | None = None,
    ) -> Competition:
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            row = session.get(Competition, competition_id)
            if row is None:
                raise KeyError(competition_id)
            current = CompetitionState(row.status)
            if expected is not None and current != expected:
                raise RuntimeError(
                    f"stale competition state: expected {expected.value}, found {current.value}"
                )
            assert_transition_allowed(current, target)
            if target == CompetitionState.COMPLETED and row.scores_need_recalculation:
                raise RuntimeError(
                    "competition scores require recalculation before completion"
                )
            podium = (
                self._finalize_competition_ranking(session, competition_id)
                if target == CompetitionState.COMPLETED
                else ()
            )
            previous_version = row.state_version
            winner = podium[0] if podium else None
            result = session.execute(
                update(Competition)
                .where(
                    Competition.competition_id == competition_id,
                    Competition.status == current.value,
                    Competition.state_version == previous_version,
                )
                .values(
                    status=target.value,
                    status_reason=reason,
                    state_version=previous_version + 1,
                    updated_at=now_text,
                    finalized_at=now_text
                    if target == CompetitionState.COMPLETED
                    else row.finalized_at,
                    winner_hotkey=winner["hotkey"] if winner else row.winner_hotkey,
                    winner_uid_at_finalisation=(
                        winner["uid_snapshot"]
                        if winner
                        else row.winner_uid_at_finalisation
                    ),
                )
            )
            if result.rowcount != 1:
                raise RuntimeError("competition state changed concurrently")
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="STATE_TRANSITION",
                from_state=current,
                to_state=target,
                actor=actor,
                payload={
                    "reason": reason,
                    **(payload or {}),
                    **({"podium": podium} if podium else {}),
                },
                now=now,
            )
            session.flush()
            return session.get(Competition, competition_id)

    def _finalize_competition_ranking(
        self,
        session: Session,
        competition_id: str,
    ) -> tuple[dict[str, Any], ...]:
        (
            competition,
            _ordered,
            ranked,
            duplicate_payout_contenders,
            _tie_reviews,
        ) = self._competition_ranking_view(session, competition_id)

        for contender in duplicate_payout_contenders:
            payout_identity = self._payout_hotkey(competition, contender)
            contender.final_rank = None
            contender.eligible = False
            contender.reason_code = "LOWER_SCORING_SOLUTION_FOR_HOTKEY"
            contender.reason_detail = (
                "A higher-scoring solution was selected for payout hotkey "
                f"{payout_identity}"
            )

        for rank, contender in enumerate(ranked, start=1):
            contender.final_rank = rank
            contender.eligible = True

        return tuple(
            {
                "rank": contender.final_rank,
                "hotkey": self._payout_hotkey(competition, contender),
                "contender_id": contender.hotkey,
                "is_boss": bool(contender.is_boss),
                "solution_type": "boss" if contender.is_boss else "submission",
                "uid_snapshot": contender.uid_snapshot,
                "final_score": contender.final_score,
                "reward_share": competition_reward_share(contender.final_rank),
            }
            for contender in ranked[:3]
        )

    @staticmethod
    def _payout_hotkey(
        competition: Competition,
        contender: ContenderMetadata,
    ) -> str:
        if contender.is_boss:
            if not competition.boss_hotkey:
                raise ValueError("boss contender is missing its payout hotkey")
            return competition.boss_hotkey
        return contender.hotkey

    def _competition_ranking_view(
        self,
        session: Session,
        competition_id: str,
    ) -> tuple[
        Competition,
        list[ContenderMetadata],
        list[ContenderMetadata],
        list[ContenderMetadata],
        list[CompetitionHumanReview],
    ]:
        """Return the deterministic ranking view without mutating persisted ranks."""

        competition = session.get(Competition, competition_id)
        if competition is None:
            raise KeyError(competition_id)
        contenders = list(
            session.scalars(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.status == ContenderState.SCORED.value,
                    ContenderMetadata.final_score.is_not(None),
                    ContenderMetadata.eligible.is_not(False),
                    ContenderMetadata.manual_disqualified.is_(False),
                )
            )
        )

        tie_positions: dict[tuple[float, str], int] = {}
        tie_reviews = list(
            session.scalars(
                select(CompetitionHumanReview)
                .where(
                    CompetitionHumanReview.competition_id == competition_id,
                    CompetitionHumanReview.review_type == EXACT_TIE_REVIEW,
                    CompetitionHumanReview.superseded.is_(False),
                )
                .order_by(CompetitionHumanReview.id)
            )
        )
        for review in tie_reviews:
            decision = json.loads(review.decision_json)
            for position, hotkey in enumerate(decision["ordered_hotkeys"]):
                tie_positions[(float(decision["final_score"]), hotkey)] = position

        def ranking_key(contender: ContenderMetadata) -> tuple[Any, ...]:
            try:
                commit_time = parse_utc(contender.latest_commit_time)
            except (TypeError, ValueError):
                commit_time = datetime.max.replace(tzinfo=timezone.utc)
            score = float(contender.final_score)
            reviewed_position = tie_positions.get((score, contender.hotkey))
            return (
                -score,
                reviewed_position is None,
                reviewed_position if reviewed_position is not None else 0,
                commit_time,
                contender.pinned_commit_sha or "\uffff",
                contender.hotkey,
            )

        contenders.sort(key=ranking_key)
        ranked: list[ContenderMetadata] = []
        duplicate_payout_contenders: list[ContenderMetadata] = []
        seen_payout_hotkeys: set[str] = set()
        for contender in contenders:
            payout_identity = self._payout_hotkey(competition, contender)
            if payout_identity in seen_payout_hotkeys:
                duplicate_payout_contenders.append(contender)
                continue
            seen_payout_hotkeys.add(payout_identity)
            ranked.append(contender)

        return (
            competition,
            contenders,
            ranked,
            duplicate_payout_contenders,
            tie_reviews,
        )

    def append_event(
        self,
        competition_id: str,
        event_type: str,
        *,
        actor: str,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        with self._sessions.begin() as session:
            if session.get(Competition, competition_id) is None:
                raise KeyError(competition_id)
            self._append_event(
                session,
                competition_id=competition_id,
                event_type=event_type,
                from_state=None,
                to_state=None,
                actor=actor,
                payload=payload,
                now=now,
            )

    def list_events(self, competition_id: str) -> list[CompetitionEvent]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(CompetitionEvent)
                    .where(CompetitionEvent.competition_id == competition_id)
                    .order_by(CompetitionEvent.id)
                )
            )

    def list_human_reviews(
        self, competition_id: str
    ) -> list[CompetitionHumanReview]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(CompetitionHumanReview)
                    .where(CompetitionHumanReview.competition_id == competition_id)
                    .order_by(CompetitionHumanReview.id)
                )
            )

    def competition_review_packet(self, competition_id: str) -> dict[str, Any]:
        """Build a read-only provisional/final ranking packet from SQLite."""

        with self._sessions() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            state = CompetitionState(competition.status)
            if competition.scores_need_recalculation:
                raise ValueError(
                    "ranking review packet requires current recalculated scores"
                )

            (
                competition,
                ordered,
                ranked,
                duplicate_payout_contenders,
                active_tie_reviews,
            ) = self._competition_ranking_view(session, competition_id)
            all_contenders = list(
                session.scalars(
                    select(ContenderMetadata)
                    .where(ContenderMetadata.competition_id == competition_id)
                    .order_by(ContenderMetadata.hotkey)
                )
            )
            histories = list(
                session.scalars(
                    select(ContenderPerformanceHistory)
                    .where(
                        ContenderPerformanceHistory.competition_id
                        == competition_id
                    )
                    .order_by(
                        ContenderPerformanceHistory.hotkey,
                        ContenderPerformanceHistory.evaluation_id,
                        ContenderPerformanceHistory.attempt,
                    )
                )
            )
            latest_histories: dict[tuple[str, str], ContenderPerformanceHistory] = {}
            for history in histories:
                latest_histories[(history.hotkey, history.evaluation_id)] = history
            histories_by_hotkey: dict[str, list[ContenderPerformanceHistory]] = {}
            for history in latest_histories.values():
                histories_by_hotkey.setdefault(history.hotkey, []).append(history)

            reviews = list(
                session.scalars(
                    select(CompetitionHumanReview)
                    .where(
                        CompetitionHumanReview.competition_id == competition_id
                    )
                    .order_by(CompetitionHumanReview.id)
                )
            )
            review_payloads = [
                {
                    "id": review.id,
                    "operator_identity": review.operator_identity,
                    "review_type": review.review_type,
                    "contenders": json.loads(review.contenders_json),
                    "decision": json.loads(review.decision_json),
                    "reason": review.reason,
                    "supersedes_review_id": review.supersedes_review_id,
                    "superseded": bool(review.superseded),
                    "integrity_hash": review.integrity_hash,
                    "created_at": review.created_at,
                }
                for review in reviews
            ]

            active_tie_review_by_score = {
                float(json.loads(review.decision_json)["final_score"]): review
                for review in active_tie_reviews
            }
            ties_by_score: dict[float, list[ContenderMetadata]] = {}
            for contender in ordered:
                ties_by_score.setdefault(float(contender.final_score), []).append(
                    contender
                )
            exact_tie_groups = []
            tied_scores: set[float] = set()
            for final_score, tied in sorted(
                ties_by_score.items(), key=lambda item: item[0], reverse=True
            ):
                if len(tied) < 2:
                    continue
                tied_scores.add(final_score)
                review = active_tie_review_by_score.get(final_score)
                exact_tie_groups.append(
                    {
                        "final_score": final_score,
                        "contender_ids": sorted(row.hotkey for row in tied),
                        "effective_order": [row.hotkey for row in tied],
                        "decision_source": (
                            "HUMAN_REVIEW" if review is not None else "COMMIT_METADATA"
                        ),
                        "review_id": review.id if review is not None else None,
                    }
                )

            provisional_rank = {
                contender.hotkey: rank
                for rank, contender in enumerate(ranked, start=1)
            }
            duplicate_ids = {row.hotkey for row in duplicate_payout_contenders}

            def decimal_text(value: Any) -> str | None:
                return None if value is None else str(Decimal(str(value)))

            def contender_payload(contender: ContenderMetadata) -> dict[str, Any]:
                contender_histories = sorted(
                    histories_by_hotkey.get(contender.hotkey, ()),
                    key=lambda row: (row.evaluation_id, row.attempt),
                )
                reconciled_count = sum(
                    row.reconciled_cost_usd is not None
                    for row in contender_histories
                )
                if not contender_histories:
                    cost_source = "NONE"
                elif reconciled_count == len(contender_histories):
                    cost_source = "RECONCILED"
                elif reconciled_count:
                    cost_source = "MIXED"
                else:
                    cost_source = "ESTIMATED"
                reconciled_total = sum(
                    (
                        Decimal(str(row.reconciled_cost_usd))
                        for row in contender_histories
                        if row.reconciled_cost_usd is not None
                    ),
                    Decimal(0),
                )
                failures = [
                    {
                        "evaluation_id": row.evaluation_id,
                        "attempt": row.attempt,
                        "status": row.status,
                        "reason_code": row.reason_code,
                    }
                    for row in contender_histories
                    if row.status != "SCORED"
                ]
                score = (
                    float(contender.final_score)
                    if contender.final_score is not None
                    else None
                )
                exclusion_reason = None
                if contender.hotkey in duplicate_ids:
                    exclusion_reason = "LOWER_SCORING_SOLUTION_FOR_HOTKEY"
                elif contender.manual_disqualified:
                    exclusion_reason = "MANUALLY_DISQUALIFIED"
                elif contender.eligible is False:
                    exclusion_reason = contender.reason_code or "INELIGIBLE"
                elif contender.status != ContenderState.SCORED.value:
                    exclusion_reason = contender.reason_code or contender.status
                rank = provisional_rank.get(contender.hotkey)
                return {
                    "rank": rank,
                    "reward_share": (
                        competition_reward_share(rank) if rank is not None else 0.0
                    ),
                    "payout_hotkey": (
                        self._payout_hotkey(competition, contender)
                        if contender.pinned_commit_sha
                        else contender.hotkey
                    ),
                    "contender_id": contender.hotkey,
                    "solution_type": (
                        "boss" if contender.is_boss else "submission"
                    ),
                    "uid_snapshot": contender.uid_snapshot,
                    "eligible": exclusion_reason is None,
                    "exclusion_reason": exclusion_reason,
                    "score_components": {
                        "media_score": contender.media_score_aggregate,
                        "cost_efficiency": contender.cost_efficiency_aggregate,
                        "length_coverage": contender.length_coverage,
                        "final_score": contender.final_score,
                        "average_vmaf": contender.average_vmaf,
                        "average_compression_ratio": (
                            contender.average_compression_rate
                        ),
                    },
                    "item_totals": {
                        "pending": contender.pending_items,
                        "successful": contender.successful_items,
                        "failed": contender.failed_items,
                    },
                    "cost": {
                        "source": cost_source,
                        "estimated_cost_usd": decimal_text(
                            contender.estimated_cost_usd
                        ),
                        "reconciled_cost_usd": (
                            str(reconciled_total) if reconciled_count else None
                        ),
                    },
                    "failures": failures,
                    "static_validation": {
                        "status": contender.validation_status,
                        "reason_code": contender.reason_code,
                        "reason_detail": contender.reason_detail,
                    },
                    "pinned_commit": {
                        "sha": contender.pinned_commit_sha,
                        "tree_sha": contender.pinned_tree_sha,
                        "committer_time": contender.latest_commit_time,
                        "repository": contender.repository_display,
                    },
                    "tie_break_source": (
                        (
                            "HUMAN_REVIEW"
                            if score in active_tie_review_by_score
                            else "COMMIT_METADATA"
                        )
                        if score in tied_scores
                        else "SCORE"
                    ),
                }

            contender_payloads = [contender_payload(row) for row in all_contenders]
            contender_payloads.sort(
                key=lambda row: (
                    row["rank"] is None,
                    row["rank"] if row["rank"] is not None else 0,
                    row["contender_id"],
                )
            )
            return {
                "schema_version": 1,
                "competition_id": competition_id,
                "competition_status": state.value,
                "provisional": state != CompetitionState.COMPLETED,
                "manifest_digest": competition.manifest_digest,
                "scoring_version": competition.scoring_version,
                "score_precision": CompetitionManifest.model_validate_json(
                    competition.manifest_json
                ).score_precision,
                "human_review_deadline": competition.human_review_deadline,
                "competition_end_time": competition.end_time,
                "scores_need_recalculation": bool(
                    competition.scores_need_recalculation
                ),
                "ranking": [
                    row for row in contender_payloads if row["rank"] is not None
                ],
                "excluded_contenders": [
                    row for row in contender_payloads if row["rank"] is None
                ],
                "exact_tie_groups": exact_tie_groups,
                "review_queue": {
                    "readability_contender_ids": sorted(
                        row.hotkey
                        for row in all_contenders
                        if row.validation_status
                        == ValidationStatus.REVIEW_REQUIRED.value
                        and not row.manual_disqualified
                    ),
                    "unreviewed_exact_tie_scores": [
                        group["final_score"]
                        for group in exact_tie_groups
                        if group["review_id"] is None
                    ],
                },
                "reviews": review_payloads,
            }

    @staticmethod
    def _assert_review_identity(operator_identity: str, reason: str) -> None:
        if not operator_identity.strip():
            raise ValueError("operator_identity is required")
        if not reason.strip():
            raise ValueError("a non-empty review reason is required")

    @staticmethod
    def _assert_review_deadline(competition: Competition, now: datetime) -> None:
        if now > parse_utc(competition.human_review_deadline):
            raise ValueError("the competition human_review_deadline has passed")
        if CompetitionState(competition.status).terminal:
            raise ValueError("human review cannot modify a terminal competition")

    def _add_human_review(
        self,
        session: Session,
        *,
        competition_id: str,
        operator_identity: str,
        review_type: str,
        contenders: list[str],
        decision: dict[str, Any],
        reason: str,
        now: datetime,
        supersedes_review_id: int | None = None,
    ) -> CompetitionHumanReview:
        created_at = utc_iso(now)
        safe_reason = self._redactor.redact_text(reason.strip())
        integrity_payload = {
            "competition_id": competition_id,
            "operator_identity": operator_identity.strip(),
            "review_type": review_type,
            "contenders": contenders,
            "decision": decision,
            "reason": safe_reason,
            "supersedes_review_id": supersedes_review_id,
            "created_at": created_at,
        }
        integrity_hash = hashlib.sha256(
            json.dumps(
                integrity_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        review = CompetitionHumanReview(
            competition_id=competition_id,
            operator_identity=operator_identity.strip(),
            review_type=review_type,
            contenders_json=json.dumps(
                contenders, sort_keys=True, separators=(",", ":")
            ),
            decision_json=json.dumps(
                decision, sort_keys=True, separators=(",", ":")
            ),
            reason=safe_reason,
            supersedes_review_id=supersedes_review_id,
            superseded=False,
            integrity_hash=integrity_hash,
            created_at=created_at,
        )
        session.add(review)
        session.flush()
        return review

    def resolve_readability_review(
        self,
        competition_id: str,
        hotkey: str,
        *,
        accepted: bool,
        operator_identity: str,
        reason: str,
        now: datetime,
    ) -> CompetitionHumanReview:
        """Resolve the sole ambiguous static-validation class."""

        self._assert_review_identity(operator_identity, reason)
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            self._assert_review_deadline(competition, now)
            if CompetitionState(competition.status) != CompetitionState.VALIDATING:
                raise ValueError(
                    "readability reviews are only accepted during VALIDATING"
                )
            contender = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if contender is None:
                raise KeyError((competition_id, hotkey))
            if contender.validation_status != ValidationStatus.REVIEW_REQUIRED.value:
                raise ValueError("contender is not awaiting a readability review")
            prior_reason = contender.reason_code
            decision = "ACCEPTED" if accepted else "REJECTED"
            review = self._add_human_review(
                session,
                competition_id=competition_id,
                operator_identity=operator_identity,
                review_type=READABILITY_REVIEW,
                contenders=[hotkey],
                decision={
                    "decision": decision,
                    "prior_validation_reason": prior_reason,
                    "submission_revision": contender.submission_revision,
                },
                reason=reason,
                now=now,
            )
            contender.validation_status = decision
            contender.status = (
                ContenderState.ACCEPTED.value
                if accepted
                else ContenderState.REJECTED.value
            )
            contender.reason_code = (
                "HUMAN_READABILITY_ACCEPTED"
                if accepted
                else "HUMAN_READABILITY_REJECTED"
            )
            contender.reason_detail = _safe_reason_detail(review.reason)
            contender.updated_at = now_text
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="HUMAN_READABILITY_REVIEW_RECORDED",
                from_state=CompetitionState.VALIDATING,
                to_state=None,
                actor=operator_identity,
                payload={
                    "review_id": review.id,
                    "hotkey": hotkey,
                    "decision": decision,
                    "prior_validation_reason": prior_reason,
                },
                now=now,
            )
            return review

    def disqualify_contender(
        self,
        competition_id: str,
        hotkey: str,
        *,
        operator_identity: str,
        reason: str,
        now: datetime,
    ) -> CompetitionHumanReview:
        """Manually exclude a contender without altering its raw measurements."""

        self._assert_review_identity(operator_identity, reason)
        now_text = utc_iso(now)
        should_recalculate = False
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            self._assert_review_deadline(competition, now)
            state = CompetitionState(competition.status)
            if state not in {
                CompetitionState.VALIDATING,
                CompetitionState.BUILDING,
                CompetitionState.EVALUATING,
                CompetitionState.AWAITING_END_TIME,
            }:
                raise ValueError(
                    "manual disqualification is only allowed during VALIDATING, "
                    "BUILDING, EVALUATING, or AWAITING_END_TIME"
                )
            contender = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if contender is None or contender.pinned_commit_sha is None:
                raise KeyError((competition_id, hotkey))
            if contender.manual_disqualified:
                raise ValueError("contender is already manually disqualified")
            review = self._add_human_review(
                session,
                competition_id=competition_id,
                operator_identity=operator_identity,
                review_type=DISQUALIFICATION_REVIEW,
                contenders=[hotkey],
                decision={
                    "decision": "DISQUALIFIED",
                    "submission_revision": contender.submission_revision,
                    "raw_metrics_preserved": True,
                },
                reason=reason,
                now=now,
            )
            contender.manual_disqualified = True
            contender.manual_disqualification_review_id = review.id
            contender.eligible = False
            contender.final_rank = None
            contender.reason_code = "MANUALLY_DISQUALIFIED"
            contender.reason_detail = _safe_reason_detail(review.reason)
            contender.updated_at = now_text
            should_recalculate = state == CompetitionState.AWAITING_END_TIME
            if should_recalculate:
                competition.scores_need_recalculation = True
                competition.updated_at = now_text
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_MANUALLY_DISQUALIFIED",
                from_state=state,
                to_state=None,
                actor=operator_identity,
                payload={
                    "review_id": review.id,
                    "hotkey": hotkey,
                    "score_recalculation_required": should_recalculate,
                },
                now=now,
            )

        if should_recalculate:
            self.recalculate_competition_scores(
                competition_id, now=now, actor=operator_identity
            )
        return review

    def order_exact_tie(
        self,
        competition_id: str,
        ordered_hotkeys: list[str],
        *,
        operator_identity: str,
        reason: str,
        now: datetime,
    ) -> CompetitionHumanReview:
        """Record an order for one complete exact-score tie group."""

        self._assert_review_identity(operator_identity, reason)
        if len(ordered_hotkeys) < 2 or len(set(ordered_hotkeys)) != len(
            ordered_hotkeys
        ):
            raise ValueError("an exact tie order needs at least two unique hotkeys")
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            self._assert_review_deadline(competition, now)
            if CompetitionState(competition.status) != CompetitionState.AWAITING_END_TIME:
                raise ValueError(
                    "exact tie reviews are only allowed during AWAITING_END_TIME"
                )
            if competition.scores_need_recalculation:
                raise ValueError(
                    "exact tie review requires current recalculated scores"
                )
            contenders = list(
                session.scalars(
                    select(ContenderMetadata).where(
                        ContenderMetadata.competition_id == competition_id,
                        ContenderMetadata.hotkey.in_(ordered_hotkeys),
                        ContenderMetadata.status == ContenderState.SCORED.value,
                        ContenderMetadata.final_score.is_not(None),
                        ContenderMetadata.manual_disqualified.is_(False),
                        ContenderMetadata.eligible.is_not(False),
                    )
                )
            )
            if {row.hotkey for row in contenders} != set(ordered_hotkeys):
                raise ValueError("every ordered hotkey must be scored and eligible")
            scores = {float(row.final_score) for row in contenders}
            if len(scores) != 1:
                raise ValueError("human review cannot reorder non-tied contenders")
            final_score = scores.pop()
            full_tie = set(
                session.scalars(
                    select(ContenderMetadata.hotkey).where(
                        ContenderMetadata.competition_id == competition_id,
                        ContenderMetadata.status == ContenderState.SCORED.value,
                        ContenderMetadata.final_score == final_score,
                        ContenderMetadata.manual_disqualified.is_(False),
                        ContenderMetadata.eligible.is_not(False),
                    )
                )
            )
            if set(ordered_hotkeys) != full_tie:
                raise ValueError(
                    "ordered_hotkeys must contain the complete exact-score tie group"
                )
            previous = session.scalar(
                select(CompetitionHumanReview)
                .where(
                    CompetitionHumanReview.competition_id == competition_id,
                    CompetitionHumanReview.review_type == EXACT_TIE_REVIEW,
                    CompetitionHumanReview.superseded.is_(False),
                )
                .order_by(CompetitionHumanReview.id.desc())
            )
            supersedes_id = None
            if previous is not None:
                previous_decision = json.loads(previous.decision_json)
                if float(previous_decision["final_score"]) == final_score:
                    previous.superseded = True
                    supersedes_id = previous.id
            review = self._add_human_review(
                session,
                competition_id=competition_id,
                operator_identity=operator_identity,
                review_type=EXACT_TIE_REVIEW,
                contenders=ordered_hotkeys,
                decision={
                    "decision": "ORDERED",
                    "final_score": final_score,
                    "ordered_hotkeys": ordered_hotkeys,
                },
                reason=reason,
                now=now,
                supersedes_review_id=supersedes_id,
            )
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="EXACT_TIE_ORDER_RECORDED",
                from_state=CompetitionState.AWAITING_END_TIME,
                to_state=None,
                actor=operator_identity,
                payload={
                    "review_id": review.id,
                    "final_score": final_score,
                    "ordered_hotkeys": ordered_hotkeys,
                    "supersedes_review_id": supersedes_id,
                },
                now=now,
            )
            return review

    def record_submission_backup_success(
        self,
        competition_id: str,
        *,
        bucket: str,
        prefix: str,
        archive_key: str,
        inventory_key: str,
        checksum: str,
        size_bytes: int,
        contender_count: int,
        now: datetime,
        actor: str,
    ) -> Competition:
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            if competition.status != CompetitionState.FINALIZING_SUBMISSIONS.value:
                raise ValueError(
                    "submission backup can only complete during finalisation"
                )
            if competition.submission_backup_status == "COMPLETED":
                if (
                    competition.submission_backup_bucket != bucket
                    or competition.submission_backup_prefix != prefix
                    or competition.submission_backup_checksum != checksum
                ):
                    raise ValueError("completed submission backup is immutable")
                return competition
            competition.submission_backup_status = "COMPLETED"
            competition.submission_backup_bucket = bucket
            competition.submission_backup_prefix = prefix
            competition.submission_backup_archive_key = archive_key
            competition.submission_backup_inventory_key = inventory_key
            competition.submission_backup_checksum = checksum
            competition.submission_backup_size_bytes = size_bytes
            competition.submission_backup_contender_count = contender_count
            competition.submission_backup_completed_at = now_text
            competition.submission_backup_error = None
            competition.updated_at = now_text
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_SUBMISSION_SNAPSHOTS_BACKED_UP",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "bucket": bucket,
                    "prefix": prefix,
                    "archive_key": archive_key,
                    "inventory_key": inventory_key,
                    "archive_sha256": checksum,
                    "archive_size_bytes": size_bytes,
                    "contender_count": contender_count,
                    "visibility": "PRIVATE",
                },
                now=now,
            )
            session.flush()
            return competition

    def record_submission_backup_failure(
        self,
        competition_id: str,
        *,
        bucket: str,
        prefix: str,
        detail: str,
        now: datetime,
        actor: str,
    ) -> Competition:
        sanitized_detail = (
            _safe_reason_detail(self._redactor.redact_text(detail))
            or "submission backup failed"
        )
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            if competition.status != CompetitionState.FINALIZING_SUBMISSIONS.value:
                return competition
            if competition.submission_backup_status == "COMPLETED":
                return competition
            competition.submission_backup_status = "FAILED"
            competition.submission_backup_bucket = bucket
            competition.submission_backup_prefix = prefix
            competition.submission_backup_error = sanitized_detail
            competition.updated_at = utc_iso(now)
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_SUBMISSION_SNAPSHOT_BACKUP_FAILED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "bucket": bucket,
                    "prefix": prefix,
                    "detail": sanitized_detail,
                },
                now=now,
            )
            session.flush()
            return competition

    def database_backup_record(self, competition_id: str) -> dict[str, Any] | None:
        with self._sessions() as session:
            event_row = session.scalar(
                select(CompetitionEvent)
                .where(
                    CompetitionEvent.competition_id == competition_id,
                    CompetitionEvent.event_type == "COMPETITION_DATABASE_BACKED_UP",
                )
                .order_by(CompetitionEvent.id.desc())
                .limit(1)
            )
            return json.loads(event_row.payload_json) if event_row is not None else None

    def record_database_backup_success(
        self,
        competition_id: str,
        *,
        bucket: str,
        key: str,
        checksum: str,
        size_bytes: int,
        now: datetime,
        actor: str,
    ) -> dict[str, Any]:
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            if competition.status != CompetitionState.COMPLETED.value:
                raise ValueError(
                    "database backup can only complete for a completed competition"
                )
            existing = session.scalar(
                select(CompetitionEvent)
                .where(
                    CompetitionEvent.competition_id == competition_id,
                    CompetitionEvent.event_type == "COMPETITION_DATABASE_BACKED_UP",
                )
                .order_by(CompetitionEvent.id.desc())
                .limit(1)
            )
            if existing is not None:
                return json.loads(existing.payload_json)
            payload = {
                "bucket": bucket,
                "key": key,
                "sha256": checksum,
                "size_bytes": size_bytes,
                "visibility": "PRIVATE",
            }
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="COMPETITION_DATABASE_BACKED_UP",
                from_state=None,
                to_state=None,
                actor=actor,
                payload=payload,
                now=now,
            )
            return payload

    def ensure_invitation_candidates(
        self,
        competition_id: str,
        candidates: list[tuple[str, int | None, str | None]],
        *,
        now: datetime,
        actor: str,
    ) -> int:
        """Persist the current metagraph snapshot before sending invitations."""

        now_text = utc_iso(now)
        created = 0
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            if competition.status != CompetitionState.ENROLLING.value:
                return 0
            for hotkey, uid_snapshot, coldkey_snapshot in candidates:
                row = session.scalar(
                    select(ContenderMetadata).where(
                        ContenderMetadata.competition_id == competition_id,
                        ContenderMetadata.hotkey == hotkey,
                    )
                )
                if row is None:
                    row = ContenderMetadata(
                        competition_id=competition_id,
                        hotkey=hotkey,
                        uid_snapshot=uid_snapshot,
                        coldkey_snapshot=coldkey_snapshot,
                        status=ContenderState.INVITED.value,
                        is_boss=False,
                        invitation_attempts=0,
                        submission_poll_attempts=0,
                        pending_items=0,
                        successful_items=0,
                        failed_items=0,
                        created_at=now_text,
                        updated_at=now_text,
                    )
                    session.add(row)
                    self._append_event(
                        session,
                        competition_id=competition_id,
                        event_type="CONTENDER_INVITATION_QUEUED",
                        from_state=None,
                        to_state=None,
                        actor=actor,
                        payload={
                            "hotkey": hotkey,
                            "uid_snapshot": uid_snapshot,
                        },
                        now=now,
                    )
                    created += 1
                elif row.status in {
                    ContenderState.INVITED.value,
                    ContenderState.PARTICIPATING.value,
                }:
                    row.uid_snapshot = uid_snapshot
                    row.coldkey_snapshot = coldkey_snapshot
                    row.updated_at = now_text
        return created

    def list_due_invitation_candidates(
        self,
        competition_id: str,
        *,
        retry_before: datetime,
    ) -> list[ContenderMetadata]:
        retry_before_text = utc_iso(retry_before)
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(ContenderMetadata)
                    .where(
                        ContenderMetadata.competition_id == competition_id,
                        ContenderMetadata.status == ContenderState.INVITED.value,
                        or_(
                            ContenderMetadata.last_invited_at.is_(None),
                            ContenderMetadata.last_invited_at <= retry_before_text,
                        ),
                    )
                    .order_by(ContenderMetadata.uid_snapshot, ContenderMetadata.hotkey)
                )
            )

    def claim_invitation_attempt(
        self,
        competition_id: str,
        hotkey: str,
        *,
        retry_before: datetime,
        now: datetime,
    ) -> bool:
        retry_before_text = utc_iso(retry_before)
        with self.engine.begin() as connection:
            result = connection.execute(
                update(ContenderMetadata)
                .where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                    ContenderMetadata.status == ContenderState.INVITED.value,
                    or_(
                        ContenderMetadata.last_invited_at.is_(None),
                        ContenderMetadata.last_invited_at <= retry_before_text,
                    ),
                    select(Competition.competition_id)
                    .where(
                        Competition.competition_id == competition_id,
                        Competition.status == CompetitionState.ENROLLING.value,
                    )
                    .exists(),
                )
                .values(
                    last_invited_at=utc_iso(now),
                    invitation_attempts=ContenderMetadata.invitation_attempts + 1,
                    updated_at=utc_iso(now),
                )
            )
        return result.rowcount == 1

    def record_invitation_response(
        self,
        competition_id: str,
        hotkey: str,
        *,
        participating: bool,
        refusal_reason: str | None,
        now: datetime,
        actor: str,
    ) -> ContenderMetadata:
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if row is None:
                raise KeyError((competition_id, hotkey))
            competition = session.get(Competition, competition_id)
            if (
                competition is None
                or competition.status != CompetitionState.ENROLLING.value
            ):
                return row
            if row.status != ContenderState.INVITED.value:
                return row
            row.status = (
                ContenderState.PARTICIPATING.value
                if participating
                else ContenderState.REJECTED.value
            )
            row.reason_code = None if participating else "INVITATION_DECLINED"
            row.updated_at = now_text
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_INVITATION_RESPONSE",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": hotkey,
                    "participating": participating,
                    "refusal_reason": refusal_reason,
                },
                now=now,
            )
            session.flush()
            return row

    def list_due_submission_candidates(
        self,
        competition_id: str,
        *,
        retry_before: datetime,
    ) -> list[ContenderMetadata]:
        retry_before_text = utc_iso(retry_before)
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(ContenderMetadata)
                    .where(
                        ContenderMetadata.competition_id == competition_id,
                        or_(
                            ContenderMetadata.status
                            == ContenderState.PARTICIPATING.value,
                            and_(
                                ContenderMetadata.pinned_commit_sha.is_not(None),
                                ContenderMetadata.status.in_(
                                    RESUBMITTABLE_PINNED_STATES
                                ),
                                ContenderMetadata.reason_code != "SUBMISSION_WITHDRAWN",
                            ),
                        ),
                        or_(
                            ContenderMetadata.last_submission_poll_at.is_(None),
                            ContenderMetadata.last_submission_poll_at
                            <= retry_before_text,
                        ),
                    )
                    .order_by(ContenderMetadata.uid_snapshot, ContenderMetadata.hotkey)
                )
            )

    def claim_submission_poll(
        self,
        competition_id: str,
        hotkey: str,
        *,
        retry_before: datetime,
        now: datetime,
    ) -> bool:
        retry_before_text = utc_iso(retry_before)
        with self.engine.begin() as connection:
            result = connection.execute(
                update(ContenderMetadata)
                .where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                    or_(
                        ContenderMetadata.status == ContenderState.PARTICIPATING.value,
                        and_(
                            ContenderMetadata.pinned_commit_sha.is_not(None),
                            ContenderMetadata.status.in_(RESUBMITTABLE_PINNED_STATES),
                            ContenderMetadata.reason_code != "SUBMISSION_WITHDRAWN",
                        ),
                    ),
                    or_(
                        ContenderMetadata.last_submission_poll_at.is_(None),
                        ContenderMetadata.last_submission_poll_at <= retry_before_text,
                    ),
                    select(Competition.competition_id)
                    .where(
                        Competition.competition_id == competition_id,
                        Competition.status == CompetitionState.ENROLLING.value,
                    )
                    .exists(),
                )
                .values(
                    last_submission_poll_at=utc_iso(now),
                    submission_poll_attempts=(
                        ContenderMetadata.submission_poll_attempts + 1
                    ),
                    updated_at=utc_iso(now),
                )
            )
        return result.rowcount == 1

    def record_submission_poll_result(
        self,
        competition_id: str,
        hotkey: str,
        *,
        status: str,
        reason: str | None,
        now: datetime,
        actor: str,
    ) -> ContenderMetadata:
        if status not in {"NOT_READY", "WITHDRAWN"}:
            raise ValueError(f"unsupported submission poll status: {status}")
        with self._sessions.begin() as session:
            row = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if row is None:
                raise KeyError((competition_id, hotkey))
            if not _can_poll_submission(row):
                return row
            if status == "WITHDRAWN":
                row.status = ContenderState.REJECTED.value
                row.validation_status = ValidationStatus.REJECTED.value
                row.reason_code = "SUBMISSION_WITHDRAWN"
                row.reason_detail = _safe_reason_detail(
                    self._redactor.redact_text(reason or "")
                )
            elif row.pinned_commit_sha is None:
                row.reason_code = "SUBMISSION_NOT_READY"
                row.reason_detail = _safe_reason_detail(
                    self._redactor.redact_text(reason or "")
                )
            row.updated_at = utc_iso(now)
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_SUBMISSION_POLL",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={"hotkey": hotkey, "status": status, "reason": reason},
                now=now,
            )
            session.flush()
            return row

    def record_enrollment_error(
        self,
        competition_id: str,
        hotkey: str,
        *,
        stage: str,
        reason_code: str,
        detail: str,
        now: datetime,
        actor: str,
    ) -> None:
        sanitized_detail = _safe_reason_detail(self._redactor.redact_text(detail)) or ""
        with self._sessions.begin() as session:
            row = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if row is None:
                raise KeyError((competition_id, hotkey))
            if stage == "INVITATION" and row.status != ContenderState.INVITED.value:
                return
            if stage == "SUBMISSION_POLL" and not _can_poll_submission(row):
                return
            if row.pinned_commit_sha is None:
                row.reason_code = reason_code[:128]
                row.reason_detail = sanitized_detail
            row.updated_at = utc_iso(now)
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_ENROLLMENT_ERROR",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": hotkey,
                    "stage": stage,
                    "reason_code": reason_code,
                    "detail": sanitized_detail,
                },
                now=now,
            )

    def record_pinned_contender(
        self,
        *,
        competition_id: str,
        hotkey: str,
        repository_url_hash: str,
        repository_display: str,
        pinned_commit_sha: str,
        pinned_tree_sha: str,
        latest_commit_time: str,
        validation: ValidationReport,
        now: datetime,
        actor: str,
        uid_snapshot: int | None = None,
        coldkey_snapshot: str | None = None,
        is_boss: bool = False,
    ) -> ContenderMetadata:
        """Persist only the credential-free result of a finalized clone."""

        now_text = utc_iso(now)
        status_by_validation = {
            ValidationStatus.ACCEPTED: ContenderState.ACCEPTED,
            ValidationStatus.REVIEW_REQUIRED: ContenderState.REVIEW_REQUIRED,
            ValidationStatus.REJECTED: ContenderState.REJECTED,
        }
        with self._sessions.begin() as session:
            competition = session.get(Competition, competition_id)
            if competition is None:
                raise KeyError(competition_id)
            row = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            replacing = row is not None and row.pinned_commit_sha is not None
            if replacing:
                if (
                    row.pinned_commit_sha == pinned_commit_sha
                    and row.pinned_tree_sha == pinned_tree_sha
                    and row.repository_url_hash == repository_url_hash
                    and bool(row.is_boss) == is_boss
                ):
                    return row
                if competition.status != CompetitionState.ENROLLING.value:
                    raise ValueError(
                        "contender submissions cannot be replaced after finalisation"
                    )
            previous_submission = (
                {
                    "repository_url_hash": row.repository_url_hash,
                    "repository_display": row.repository_display,
                    "pinned_commit_sha": row.pinned_commit_sha,
                    "pinned_tree_sha": row.pinned_tree_sha,
                    "submission_revision": row.submission_revision or 0,
                    "validation_status": row.validation_status,
                }
                if replacing
                else None
            )
            if row is None:
                row = ContenderMetadata(
                    competition_id=competition_id,
                    hotkey=hotkey,
                    created_at=now_text,
                    pending_items=0,
                    successful_items=0,
                    failed_items=0,
                    is_boss=is_boss,
                )
                session.add(row)
            row.uid_snapshot = uid_snapshot
            row.coldkey_snapshot = coldkey_snapshot
            row.is_boss = is_boss
            row.repository_url_hash = repository_url_hash
            row.repository_display = repository_display
            row.pinned_commit_sha = pinned_commit_sha
            row.pinned_tree_sha = pinned_tree_sha
            row.latest_commit_time = latest_commit_time
            row.submission_revision = int(row.submission_revision or 0) + 1
            row.validation_status = validation.status.value
            row.reason_code = validation.reason_code.value
            primary_finding = next(
                (
                    finding
                    for finding in validation.findings
                    if finding.reason_code == validation.reason_code
                ),
                None,
            )
            row.reason_detail = _safe_reason_detail(
                self._redactor.redact_text(
                    f"{primary_finding.path}: {primary_finding.detail}"
                )
                if primary_finding is not None
                else None
            )
            row.status = status_by_validation[validation.status].value
            row.build_status = None
            row.image_id = None
            row.image_digest = None
            row.image_size_bytes = None
            row.build_evidence_json = None
            row.modal_app_id = None
            row.modal_sandbox_id = None
            row.output_volume_name = None
            row.pending_items = 0
            row.successful_items = 0
            row.failed_items = 0
            row.updated_at = now_text
            self._append_event(
                session,
                competition_id=competition_id,
                event_type=(
                    "CONTENDER_REPOSITORY_REPLACED"
                    if replacing
                    else "CONTENDER_REPOSITORY_PINNED"
                ),
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": hotkey,
                    "is_boss": is_boss,
                    "repository_url_hash": repository_url_hash,
                    "repository_display": repository_display,
                    "pinned_commit_sha": pinned_commit_sha,
                    "pinned_tree_sha": pinned_tree_sha,
                    "latest_commit_time": latest_commit_time,
                    "submission_revision": row.submission_revision,
                    "validation_status": validation.status.value,
                    "reason_code": validation.reason_code.value,
                    "reason_detail": row.reason_detail,
                    "previous_submission": previous_submission,
                },
                now=now,
            )
            session.flush()
            return row

    def get_contender(
        self, competition_id: str, hotkey: str
    ) -> ContenderMetadata | None:
        with self._sessions() as session:
            return session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )

    def list_contenders(self, competition_id: str) -> list[ContenderMetadata]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(ContenderMetadata)
                    .where(ContenderMetadata.competition_id == competition_id)
                    .order_by(ContenderMetadata.hotkey)
                )
            )

    def mark_build_started(
        self,
        *,
        competition_id: str,
        hotkey: str,
        now: datetime,
        actor: str,
    ) -> None:
        now_text = utc_iso(now)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if row is None:
                raise KeyError((competition_id, hotkey))
            if row.build_status not in {None, "BUILDING", "RETRY"}:
                return
            starting_attempt = row.build_status != "BUILDING"
            row.build_status = "BUILDING"
            row.reason_code = "BUILD_IN_PROGRESS"
            row.updated_at = now_text
            if starting_attempt:
                self._append_event(
                    session,
                    competition_id=competition_id,
                    event_type="CONTENDER_BUILD_STARTED",
                    from_state=None,
                    to_state=None,
                    actor=actor,
                    payload={"hotkey": hotkey},
                    now=now,
                )

    def record_build_evidence(
        self,
        *,
        competition_id: str,
        hotkey: str,
        image_id: str,
        image_digest: str,
        image_size_bytes: int,
        evidence: dict[str, Any],
        now: datetime,
        actor: str,
        build_status: str = "ACCEPTED",
    ) -> ContenderMetadata:
        """Persist accepted, credential-free image build evidence."""

        now_text = utc_iso(now)
        sanitized = self._redactor.redact(evidence)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if row is None:
                raise KeyError((competition_id, hotkey))
            if row.image_id is not None and (
                row.image_id != image_id or row.image_digest != image_digest
            ):
                raise ValueError("accepted contender build is immutable")
            row.image_id = image_id
            row.image_digest = image_digest
            row.image_size_bytes = image_size_bytes
            row.build_evidence_json = json.dumps(
                sanitized, sort_keys=True, separators=(",", ":"), default=str
            )
            if build_status not in {
                "ACCEPTED",
                "MODAL_ACCEPTED",
                "DEVELOPMENT_ACCEPTED",
            }:
                raise ValueError("invalid accepted build status")
            row.build_status = build_status
            row.reason_code = build_status
            row.status = ContenderState.BUILT.value
            row.updated_at = now_text
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_IMAGE_PINNED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": hotkey,
                    "image_id": image_id,
                    "image_digest": image_digest,
                    "image_size_bytes": image_size_bytes,
                    "build_status": build_status,
                    "build_evidence": sanitized,
                },
                now=now,
            )
            session.flush()
            return row

    def record_build_rejection(
        self,
        *,
        competition_id: str,
        hotkey: str,
        reason_code: str,
        detail: str,
        now: datetime,
        actor: str,
    ) -> None:
        with self._sessions.begin() as session:
            row = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if row is None:
                raise KeyError((competition_id, hotkey))
            row.build_status = "REJECTED"
            row.reason_code = reason_code
            row.status = ContenderState.REJECTED.value
            row.updated_at = utc_iso(now)
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_BUILD_REJECTED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": hotkey,
                    "reason_code": reason_code,
                    "detail": detail,
                },
                now=now,
            )

    def record_build_retryable_failure(
        self,
        *,
        competition_id: str,
        hotkey: str,
        reason_code: str,
        detail: str,
        now: datetime,
        actor: str,
    ) -> None:
        with self._sessions.begin() as session:
            row = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if row is None:
                raise KeyError((competition_id, hotkey))
            row.build_status = "RETRY"
            row.reason_code = reason_code
            row.updated_at = utc_iso(now)
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="CONTENDER_BUILD_RETRY_SCHEDULED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": hotkey,
                    "reason_code": reason_code,
                    "detail": detail,
                },
                now=now,
            )

    def reserve_sandbox_generation(
        self,
        *,
        competition_id: str,
        hotkey: str,
        image_id: str,
        image_digest: str,
        output_volume_name: str,
        gpu_type: str,
        requested_cpu_cores: int,
        max_cpu_cores: int,
        batch_timeout_seconds: float,
        created_at: datetime,
        expires_at: datetime,
        actor: str,
    ) -> CompetitionSandbox:
        with self._sessions.begin() as session:
            contender = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == competition_id,
                    ContenderMetadata.hotkey == hotkey,
                )
            )
            if contender is None:
                raise KeyError((competition_id, hotkey))
            if contender.image_id != image_id or contender.image_digest != image_digest:
                raise ValueError("sandbox image must match accepted build evidence")
            generation = (
                int(
                    session.scalar(
                        select(func.max(CompetitionSandbox.generation)).where(
                            CompetitionSandbox.competition_id == competition_id,
                            CompetitionSandbox.hotkey == hotkey,
                        )
                    )
                    or 0
                )
                + 1
            )
            now_text = utc_iso(created_at)
            row = CompetitionSandbox(
                competition_id=competition_id,
                hotkey=hotkey,
                generation=generation,
                status="STARTING",
                image_id=image_id,
                image_digest=image_digest,
                output_volume_name=output_volume_name,
                gpu_type=gpu_type,
                requested_cpu_cores=requested_cpu_cores,
                max_cpu_cores=max_cpu_cores,
                batch_timeout_seconds=batch_timeout_seconds,
                network_blocked=True,
                input_read_only=True,
                secrets_attached=False,
                created_at=now_text,
                expires_at=utc_iso(expires_at),
                updated_at=now_text,
            )
            session.add(row)
            session.flush()
            self._append_event(
                session,
                competition_id=competition_id,
                event_type="SANDBOX_GENERATION_RESERVED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": hotkey,
                    "generation": generation,
                    "image_id": image_id,
                    "output_volume_name": output_volume_name,
                    "gpu_type": gpu_type,
                    "cpu": [requested_cpu_cores, max_cpu_cores],
                    "expires_at": utc_iso(expires_at),
                },
                now=created_at,
            )
            return row

    def activate_sandbox(
        self,
        sandbox_row_id: int,
        *,
        modal_app_id: str,
        modal_sandbox_id: str,
        isolation_report: dict[str, Any],
        allocated_gpu_type: str,
        allocated_gpu_count: int,
        allocated_cpu_cores: float,
        now: datetime,
        actor: str,
    ) -> CompetitionSandbox:
        with self._sessions.begin() as session:
            row = session.get(CompetitionSandbox, sandbox_row_id)
            if row is None:
                raise KeyError(sandbox_row_id)
            if row.status not in {"STARTING", "RUNNING"}:
                raise ValueError("only a starting sandbox can become active")
            if row.modal_app_id and row.modal_app_id != modal_app_id:
                raise ValueError("bound Modal app identity is immutable")
            if row.modal_sandbox_id and row.modal_sandbox_id != modal_sandbox_id:
                raise ValueError("bound Modal sandbox identity is immutable")
            sanitized = self._redactor.redact(isolation_report)
            now_text = utc_iso(now)
            row.status = "RUNNING"
            row.modal_app_id = modal_app_id
            row.modal_sandbox_id = modal_sandbox_id
            if allocated_gpu_count <= 0 or allocated_cpu_cores <= 0:
                raise ValueError("allocated Sandbox resources must be positive")
            row.allocated_gpu_type = allocated_gpu_type
            row.allocated_gpu_count = allocated_gpu_count
            row.allocated_cpu_cores = allocated_cpu_cores
            row.isolation_report_json = json.dumps(
                sanitized, sort_keys=True, separators=(",", ":"), default=str
            )
            row.last_health_at = now_text
            row.updated_at = now_text
            contender = session.scalar(
                select(ContenderMetadata).where(
                    ContenderMetadata.competition_id == row.competition_id,
                    ContenderMetadata.hotkey == row.hotkey,
                )
            )
            if contender is None:
                raise KeyError((row.competition_id, row.hotkey))
            contender.modal_app_id = modal_app_id
            contender.modal_sandbox_id = modal_sandbox_id
            contender.output_volume_name = row.output_volume_name
            contender.status = ContenderState.RUNNING.value
            contender.updated_at = now_text
            self._append_event(
                session,
                competition_id=row.competition_id,
                event_type="SANDBOX_ISOLATION_VERIFIED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": row.hotkey,
                    "generation": row.generation,
                    "modal_app_id": modal_app_id,
                    "modal_sandbox_id": modal_sandbox_id,
                    "allocated_gpu_type": allocated_gpu_type,
                    "allocated_gpu_count": allocated_gpu_count,
                    "allocated_cpu_cores": allocated_cpu_cores,
                    "isolation_report": sanitized,
                },
                now=now,
            )
            session.flush()
            return row

    def update_sandbox_batch_timeout(
        self,
        sandbox_row_id: int,
        *,
        batch_timeout_seconds: float,
        now: datetime,
    ) -> CompetitionSandbox:
        """Record the dynamic deadline most recently applied to this generation."""

        if batch_timeout_seconds <= 0:
            raise ValueError("batch timeout must be positive")
        with self._sessions.begin() as session:
            row = session.get(CompetitionSandbox, sandbox_row_id)
            if row is None:
                raise KeyError(sandbox_row_id)
            if row.status not in {"STARTING", "RUNNING"}:
                raise ValueError("only an active sandbox can receive a batch timeout")
            row.batch_timeout_seconds = float(batch_timeout_seconds)
            row.updated_at = utc_iso(now)
            session.flush()
            return row

    def bind_sandbox_identity(
        self,
        sandbox_row_id: int,
        *,
        modal_app_id: str,
        modal_sandbox_id: str,
        now: datetime,
    ) -> CompetitionSandbox:
        """Persist external IDs before probes so a restart can reattach."""

        with self._sessions.begin() as session:
            row = session.get(CompetitionSandbox, sandbox_row_id)
            if row is None:
                raise KeyError(sandbox_row_id)
            if row.status != "STARTING":
                raise ValueError("sandbox identity can only bind while STARTING")
            if row.modal_app_id and row.modal_app_id != modal_app_id:
                raise ValueError("Modal app identity is already bound")
            if row.modal_sandbox_id and row.modal_sandbox_id != modal_sandbox_id:
                raise ValueError("Modal sandbox identity is already bound")
            row.modal_app_id = modal_app_id
            row.modal_sandbox_id = modal_sandbox_id
            row.updated_at = utc_iso(now)
            session.flush()
            return row

    def bind_sandbox_names(
        self,
        sandbox_row_id: int,
        *,
        modal_app_name: str,
        modal_sandbox_name: str,
        now: datetime,
    ) -> CompetitionSandbox:
        """Persist deterministic lookup names before the external create call."""

        with self._sessions.begin() as session:
            row = session.get(CompetitionSandbox, sandbox_row_id)
            if row is None:
                raise KeyError(sandbox_row_id)
            if row.status != "STARTING":
                raise ValueError("sandbox names can only bind while STARTING")
            if row.modal_app_name and row.modal_app_name != modal_app_name:
                raise ValueError("Modal app name is already bound")
            if row.modal_sandbox_name and row.modal_sandbox_name != modal_sandbox_name:
                raise ValueError("Modal sandbox name is already bound")
            row.modal_app_name = modal_app_name
            row.modal_sandbox_name = modal_sandbox_name
            row.updated_at = utc_iso(now)
            session.flush()
            return row

    def latest_sandbox(
        self, competition_id: str, hotkey: str
    ) -> CompetitionSandbox | None:
        with self._sessions() as session:
            return session.scalar(
                select(CompetitionSandbox)
                .where(
                    CompetitionSandbox.competition_id == competition_id,
                    CompetitionSandbox.hotkey == hotkey,
                )
                .order_by(CompetitionSandbox.generation.desc())
                .limit(1)
            )

    def list_recoverable_sandboxes(self) -> list[CompetitionSandbox]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(CompetitionSandbox)
                    .where(CompetitionSandbox.status.in_(("STARTING", "RUNNING")))
                    .order_by(
                        CompetitionSandbox.competition_id,
                        CompetitionSandbox.hotkey,
                        CompetitionSandbox.generation,
                    )
                )
            )

    def touch_sandbox(self, sandbox_row_id: int, *, now: datetime) -> None:
        now_text = utc_iso(now)
        with self.engine.begin() as connection:
            connection.execute(
                update(CompetitionSandbox)
                .where(
                    CompetitionSandbox.id == sandbox_row_id,
                    CompetitionSandbox.status == "RUNNING",
                )
                .values(last_health_at=now_text, updated_at=now_text)
            )

    def close_sandbox(
        self,
        sandbox_row_id: int,
        *,
        status: str,
        reason_code: str,
        detail: str | None = None,
        now: datetime,
        actor: str,
    ) -> None:
        if status not in {"TERMINATED", "EXPIRED", "FAILED", "ORPHANED"}:
            raise ValueError("invalid terminal sandbox status")
        with self._sessions.begin() as session:
            row = session.get(CompetitionSandbox, sandbox_row_id)
            if row is None:
                raise KeyError(sandbox_row_id)
            if row.status in {"TERMINATED", "EXPIRED", "FAILED", "ORPHANED"}:
                return
            now_text = utc_iso(now)
            row.status = status
            row.reason_code = reason_code
            row.terminated_at = now_text
            row.updated_at = now_text
            self._append_event(
                session,
                competition_id=row.competition_id,
                event_type="SANDBOX_CLOSED",
                from_state=None,
                to_state=None,
                actor=actor,
                payload={
                    "hotkey": row.hotkey,
                    "generation": row.generation,
                    "modal_sandbox_id": row.modal_sandbox_id,
                    "status": status,
                    "reason_code": reason_code,
                    "detail": detail,
                },
                now=now,
            )

    def _append_event(
        self,
        session: Session,
        *,
        competition_id: str,
        event_type: str,
        from_state: CompetitionState | None,
        to_state: CompetitionState | None,
        actor: str,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        sanitized = self._redactor.redact(payload)
        payload_json = json.dumps(
            sanitized, sort_keys=True, separators=(",", ":"), default=str
        )
        session.add(
            CompetitionEvent(
                competition_id=competition_id,
                event_type=event_type,
                from_state=from_state.value if from_state else None,
                to_state=to_state.value if to_state else None,
                actor=actor,
                payload_json=payload_json,
                payload_hash=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                occurred_at=utc_iso(now),
            )
        )


__all__ = [
    "CompetitionRepository",
    "SCHEMA_VERSION",
    "parse_utc",
    "utc_iso",
]
