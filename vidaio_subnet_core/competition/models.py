"""Competition-only SQLAlchemy models.

These tables intentionally use their own metadata so inference persistence cannot
accidentally query or mutate competition history.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import declarative_base


CompetitionBase = declarative_base()


class CompetitionSchemaMigration(CompetitionBase):
    __tablename__ = "competition_schema_migrations"

    version = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)
    applied_at = Column(String(40), nullable=False)


class Competition(CompetitionBase):
    __tablename__ = "competitions"

    competition_id = Column(String(64), primary_key=True)
    competition_type = Column(String(32), nullable=False)
    schema_version = Column(Integer, nullable=False)
    scoring_version = Column(String(32), nullable=False)
    manifest_json = Column(Text, nullable=False)
    manifest_digest = Column(String(64), nullable=False, unique=True)
    status = Column(String(32), nullable=False, index=True)
    status_reason = Column(String(512))
    start_time = Column(String(40), nullable=False, index=True)
    contender_finalisation_time = Column(String(40), nullable=False)
    human_review_deadline = Column(String(40), nullable=False)
    end_time = Column(String(40), nullable=False)
    input_volume_name = Column(String(64), nullable=False)
    # Legacy non-null column: compression rows point it at input_volume_name.
    reference_volume_name = Column(String(64), nullable=False)
    evaluation_index_path = Column(String(512), nullable=False)
    dataset_index_checksum = Column(String(128))
    input_volume_checksum = Column(String(128))
    # Retained for migration compatibility; compression has no second reference.
    reference_volume_checksum = Column(String(128))
    # Retained for migration compatibility with the previous winner-import design.
    source_competition_id = Column(String(64))
    boss_repository_path = Column(String(512))
    boss_hotkey = Column(String(128))
    submission_backup_status = Column(
        String(32), nullable=False, default="PENDING", server_default="PENDING"
    )
    submission_backup_bucket = Column(String(256))
    submission_backup_prefix = Column(String(512))
    submission_backup_archive_key = Column(String(768))
    submission_backup_inventory_key = Column(String(768))
    submission_backup_checksum = Column(String(64))
    submission_backup_size_bytes = Column(Integer)
    submission_backup_contender_count = Column(Integer)
    submission_backup_completed_at = Column(String(40))
    submission_backup_error = Column(String(500))
    winner_hotkey = Column(String(128))
    winner_uid_at_finalisation = Column(Integer)
    finalized_at = Column(String(40))
    scores_need_recalculation = Column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    created_at = Column(String(40), nullable=False)
    updated_at = Column(String(40), nullable=False)
    scheduler_lease_owner = Column(String(256))
    scheduler_lease_expires_at = Column(String(40), index=True)
    state_version = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_competitions_status_start", "status", "start_time"),
        Index(
            "uq_competitions_single_running",
            text("1"),
            unique=True,
            sqlite_where=text(
                "status IN ("
                "'ENROLLING','FINALIZING_SUBMISSIONS','VALIDATING','BUILDING',"
                "'EVALUATING','SCORING','AWAITING_END_TIME')"
            ),
        ),
    )


class ContenderMetadata(CompetitionBase):
    __tablename__ = "contender_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(
        String(64),
        ForeignKey("competitions.competition_id", ondelete="CASCADE"),
        nullable=False,
    )
    hotkey = Column(String(128), nullable=False)
    uid_snapshot = Column(Integer)
    coldkey_snapshot = Column(String(128))
    status = Column(String(32), nullable=False, index=True)
    is_boss = Column(Boolean, nullable=False, default=False)
    source_competition_id = Column(String(64))
    repository_url_hash = Column(String(64))
    repository_display = Column(String(256))
    pinned_commit_sha = Column(String(64))
    pinned_tree_sha = Column(String(64))
    latest_commit_time = Column(String(40))
    submission_revision = Column(
        Integer, nullable=False, default=0, server_default="0"
    )
    validation_status = Column(String(32))
    build_status = Column(String(32))
    reason_code = Column(String(128))
    reason_detail = Column(String(500))
    last_invited_at = Column(String(40))
    last_submission_poll_at = Column(String(40))
    invitation_attempts = Column(Integer, nullable=False, default=0, server_default="0")
    submission_poll_attempts = Column(
        Integer, nullable=False, default=0, server_default="0"
    )
    image_id = Column(String(256))
    image_digest = Column(String(256))
    image_size_bytes = Column(Integer)
    build_evidence_json = Column(Text)
    modal_app_id = Column(String(128))
    modal_sandbox_id = Column(String(128))
    output_volume_name = Column(String(128))
    pending_items = Column(Integer, nullable=False, default=0)
    successful_items = Column(Integer, nullable=False, default=0)
    failed_items = Column(Integer, nullable=False, default=0)
    final_score = Column(Float)
    average_vmaf = Column(Float)
    average_compression_rate = Column(Float)
    media_score_aggregate = Column(Float)
    quality_aggregate = Column(Float)
    cost_efficiency_aggregate = Column(Float)
    length_coverage = Column(Float)
    estimated_cost_usd = Column(Numeric(20, 10))
    reconciled_cost_usd = Column(Numeric(20, 10))
    active_runtime_seconds = Column(Float)
    cold_start_runtime_seconds = Column(Float)
    final_rank = Column(Integer)
    eligible = Column(Boolean)
    # Human-review eligibility is independent from static/objective validation.
    # Raw histories and measured metrics remain intact when this is set.
    manual_disqualified = Column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    manual_disqualification_review_id = Column(
        Integer, ForeignKey("competition_human_reviews.id")
    )
    created_at = Column(String(40), nullable=False)
    updated_at = Column(String(40), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "competition_id", "hotkey", name="uq_contender_competition_hotkey"
        ),
        Index("ix_contender_competition_status", "competition_id", "status"),
        Index(
            "ix_contender_review_eligibility",
            "competition_id",
            "manual_disqualified",
            "status",
        ),
    )


class CompetitionSandbox(CompetitionBase):
    """One immutable lifecycle generation of a contender's warm sandbox."""

    __tablename__ = "competition_sandboxes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(
        String(64),
        ForeignKey("competitions.competition_id", ondelete="CASCADE"),
        nullable=False,
    )
    hotkey = Column(String(128), nullable=False)
    generation = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, index=True)
    image_id = Column(String(256), nullable=False)
    image_digest = Column(String(256), nullable=False)
    modal_app_name = Column(String(128))
    modal_sandbox_name = Column(String(128))
    modal_app_id = Column(String(128))
    modal_sandbox_id = Column(String(128))
    output_volume_name = Column(String(128), nullable=False)
    gpu_type = Column(String(64), nullable=False)
    requested_cpu_cores = Column(Integer, nullable=False)
    max_cpu_cores = Column(Integer, nullable=False)
    allocated_gpu_type = Column(String(64))
    allocated_gpu_count = Column(Integer)
    allocated_cpu_cores = Column(Float)
    # Dynamic miner invocation deadline most recently applied to this generation.
    batch_timeout_seconds = Column(Float, nullable=False)
    network_blocked = Column(Boolean, nullable=False)
    input_read_only = Column(Boolean, nullable=False)
    secrets_attached = Column(Boolean, nullable=False)
    isolation_report_json = Column(Text)
    reason_code = Column(String(128))
    created_at = Column(String(40), nullable=False)
    expires_at = Column(String(40), nullable=False, index=True)
    last_health_at = Column(String(40))
    terminated_at = Column(String(40))
    updated_at = Column(String(40), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "competition_id",
            "hotkey",
            "generation",
            name="uq_competition_sandbox_generation",
        ),
        Index(
            "ix_competition_sandbox_owner_status",
            "competition_id",
            "hotkey",
            "status",
        ),
    )


class CompetitionEvaluationItem(CompetitionBase):
    __tablename__ = "competition_evaluation_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(
        String(64),
        ForeignKey("competitions.competition_id", ondelete="CASCADE"),
        nullable=False,
    )
    evaluation_id = Column(String(128), nullable=False)
    normalized_json = Column(Text, nullable=False)
    canonical_batch_index = Column(Integer, nullable=False)
    canonical_batch_position = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, index=True)
    dispatch_status = Column(String(32), nullable=False, default="PENDING")
    score_status = Column(String(32), nullable=False, default="PENDING")
    current_attempt = Column(Integer, nullable=False, default=0)
    lease_owner = Column(String(256))
    lease_expires_at = Column(String(40), index=True)
    checksum = Column(String(128), nullable=False)
    duration_seconds = Column(Float, nullable=False)
    created_at = Column(String(40), nullable=False)
    updated_at = Column(String(40), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "competition_id", "evaluation_id", name="uq_competition_evaluation"
        ),
        UniqueConstraint(
            "competition_id",
            "canonical_batch_index",
            "canonical_batch_position",
            name="uq_competition_canonical_batch_position",
        ),
        Index("ix_evaluation_competition_status", "competition_id", "status"),
    )


class CompetitionBatch(CompetitionBase):
    __tablename__ = "competition_batches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(String(64), nullable=False)
    hotkey = Column(String(128), nullable=False)
    batch_id = Column(String(128), nullable=False)
    canonical_batch_index = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, index=True)
    lease_owner = Column(String(256))
    lease_expires_at = Column(String(40), index=True)
    modal_sandbox_id = Column(String(128))
    # Dynamic lease duration for this Sandbox execution attempt. It includes a
    # recovery grace beyond the miner invocation deadline. Scoring has a
    # separate, independently timed deadline recorded below.
    timeout_seconds = Column(Float)
    scoring_timeout_seconds = Column(Float)
    scoring_expires_at = Column(String(40), index=True)
    wall_runtime_seconds = Column(Float)
    active_runtime_seconds = Column(Float)
    cold_start_seconds = Column(Float)
    resource_usage_json = Column(Text)
    reconciliation_status = Column(String(32), nullable=False, default="PENDING")
    estimated_cost_usd = Column(Numeric(20, 10))
    reconciled_cost_usd = Column(Numeric(20, 10))
    created_at = Column(String(40), nullable=False)
    updated_at = Column(String(40), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "competition_id", "hotkey", "batch_id", name="uq_competition_batch"
        ),
        Index(
            "ix_competition_batch_canonical",
            "competition_id",
            "hotkey",
            "canonical_batch_index",
        ),
        Index(
            "ix_competition_batches_scoring_deadline",
            "competition_id",
            "hotkey",
            "status",
            "scoring_expires_at",
        ),
        Index(
            "ix_competition_batches_lease",
            "competition_id",
            "hotkey",
            "status",
            "lease_expires_at",
        ),
    )


class ContenderPerformanceHistory(CompetitionBase):
    __tablename__ = "contender_performance_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(String(64), nullable=False)
    hotkey = Column(String(128), nullable=False)
    evaluation_id = Column(String(128), nullable=False)
    batch_id = Column(String(128), nullable=False)
    canonical_batch_index = Column(Integer, nullable=False)
    attempt = Column(Integer, nullable=False)
    idempotency_key = Column(String(256), nullable=False, unique=True)
    input_checksum = Column(String(128))
    output_checksum = Column(String(128))
    input_size_bytes = Column(Integer)
    output_size_bytes = Column(Integer)
    status = Column(String(32), nullable=False, index=True)
    # Legacy schema fields retained until Phase 4 rebuilds the history table.
    task_type = Column(String(32), nullable=False)
    scale_factor = Column(Integer, nullable=False, default=1)
    duration_seconds = Column(Float, nullable=False)
    length_weight = Column(Float, nullable=False)
    vmaf_threshold = Column(Float)
    vmaf_score = Column(Float)
    compression_rate = Column(Float)
    compression_score = Column(Float)
    media_score = Column(Float)
    media_compression_component = Column(Float)
    media_vmaf_component = Column(Float)
    media_score_reason = Column(String(500))
    quality_component = Column(Float, nullable=False, default=0)
    cost_efficiency_component = Column(Float, nullable=False, default=0)
    completion_value = Column(Float, nullable=False, default=0)
    handler_runtime_seconds = Column(Float)
    queue_time_seconds = Column(Float)
    cold_start_seconds = Column(Float)
    raw_cost_usd = Column(Numeric(20, 10))
    estimated_cost_usd = Column(Numeric(20, 10))
    reconciled_cost_usd = Column(Numeric(20, 10))
    currency = Column(String(8), nullable=False, default="USD")
    cost_attribution_method = Column(String(64))
    modal_app_id = Column(String(128))
    modal_sandbox_id = Column(String(128))
    modal_input_id = Column(String(128))
    reason_code = Column(String(128))
    scoring_version = Column(String(32), nullable=False)
    processing_started_at = Column(String(40))
    processing_finished_at = Column(String(40))
    created_at = Column(String(40), nullable=False)
    updated_at = Column(String(40), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "competition_id",
            "hotkey",
            "evaluation_id",
            "attempt",
            name="uq_contender_evaluation_attempt",
        ),
        Index(
            "ix_history_competition_contender_evaluation",
            "competition_id",
            "hotkey",
            "evaluation_id",
        ),
        Index(
            "ix_history_competition_contender_canonical_batch",
            "competition_id",
            "hotkey",
            "canonical_batch_index",
        ),
        Index(
            "ix_history_dispatch_state",
            "competition_id",
            "hotkey",
            "evaluation_id",
            "status",
            "attempt",
        ),
    )


class CompetitionHumanReview(CompetitionBase):
    __tablename__ = "competition_human_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(
        String(64),
        ForeignKey("competitions.competition_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    operator_identity = Column(String(256), nullable=False)
    review_type = Column(String(32), nullable=False)
    contenders_json = Column(Text, nullable=False)
    decision_json = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    supersedes_review_id = Column(
        Integer, ForeignKey("competition_human_reviews.id")
    )
    superseded = Column(Boolean, nullable=False, default=False)
    integrity_hash = Column(String(64), nullable=False, unique=True)
    created_at = Column(String(40), nullable=False)

    __table_args__ = (
        Index(
            "ix_competition_reviews_active",
            "competition_id",
            "review_type",
            "superseded",
        ),
    )


class CompetitionEvent(CompetitionBase):
    __tablename__ = "competition_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(String(64), nullable=False)
    event_type = Column(String(64), nullable=False)
    from_state = Column(String(32))
    to_state = Column(String(32))
    actor = Column(String(256), nullable=False)
    payload_json = Column(Text, nullable=False)
    payload_hash = Column(String(64), nullable=False)
    occurred_at = Column(String(40), nullable=False, index=True)

    __table_args__ = (
        Index("ix_competition_events_competition_id", "competition_id", "id"),
    )
