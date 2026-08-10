"""Restart-safe competition scheduler and state manager."""

from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Awaitable, Callable

from loguru import logger
from sqlalchemy.exc import IntegrityError

from .config import CompetitionConfig, CompetitionManifest, load_manifest
from .models import Competition
from .phase0 import SecretRedactor
from .repository import CompetitionRepository, parse_utc
from .state import CompetitionState, PIPELINE_SUCCESSOR


Clock = Callable[[], datetime]
TickHook = Callable[[], Awaitable[None]]
LOG_REDACTOR = SecretRedactor()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_time_left(deadline: datetime, now: datetime) -> str:
    remaining_seconds = max(0, ceil((deadline - now).total_seconds()))
    days, remaining_seconds = divmod(remaining_seconds, 24 * 60 * 60)
    hours, remaining_seconds = divmod(remaining_seconds, 60 * 60)
    minutes, seconds = divmod(remaining_seconds, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


def _scheduled_phase_countdown(
    competition: Competition, now: datetime
) -> tuple[CompetitionState, str, datetime, str] | None:
    state = CompetitionState(competition.status)
    scheduled_transitions = {
        CompetitionState.SCHEDULED: (
            CompetitionState.ENROLLING,
            "competition_start_time",
            competition.start_time,
        ),
        CompetitionState.ENROLLING: (
            CompetitionState.FINALIZING_SUBMISSIONS,
            "contender_finalisation_time",
            competition.contender_finalisation_time,
        ),
        CompetitionState.AWAITING_END_TIME: (
            CompetitionState.COMPLETED,
            "competition_end_time",
            competition.end_time,
        ),
    }
    scheduled = scheduled_transitions.get(state)
    if scheduled is None:
        return None
    next_state, manifest_field, deadline_text = scheduled
    deadline = parse_utc(deadline_text)
    return next_state, manifest_field, deadline, _format_time_left(deadline, now)


def _manifest_changed_fields(
    previous_normalized_json: str, manifest: CompetitionManifest
) -> dict[str, dict[str, object]]:
    previous = json.loads(previous_normalized_json)
    current = json.loads(manifest.normalized_json())
    return {
        field: {"old": previous.get(field), "new": current.get(field)}
        for field in sorted(set(previous) | set(current))
        if previous.get(field) != current.get(field)
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class CompetitionManager:
    def __init__(
        self,
        config: CompetitionConfig,
        repository: CompetitionRepository | None = None,
        *,
        clock: Clock = utc_now,
    ) -> None:
        self.config = config
        self._clock = clock
        self.repository = repository
        if self.config.mode_enabled and self.repository is None:
            self.repository = CompetitionRepository(self.config.database_url)

    @property
    def enabled(self) -> bool:
        return self.config.mode_enabled

    def _require_repository(self) -> CompetitionRepository:
        if not self.enabled or self.repository is None:
            raise RuntimeError("competition mode is disabled")
        return self.repository

    def register_manifest(
        self,
        manifest: CompetitionManifest,
        *,
        validate_runtime_paths: bool = False,
        repository_root: Path | None = None,
    ) -> Competition:
        repository = self._require_repository()
        if validate_runtime_paths:
            manifest.validate_runtime_paths(repository_root or Path.cwd())
        previous = repository.get(manifest.competition_id)
        new_digest = manifest.digest()
        changed_fields = (
            _manifest_changed_fields(previous.manifest_json, manifest)
            if previous is not None and previous.manifest_digest != new_digest
            else {}
        )
        artifact_dir = self.config.artifact_root / manifest.competition_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / "manifest.normalized.json"
        normalized = manifest.normalized_json() + "\n"
        divergent_artifact: dict[str, str | None] | None = None
        if destination.exists():
            existing_artifact = destination.read_text(encoding="utf-8")
            permitted_artifacts = {normalized}
            if previous is not None:
                permitted_artifacts.add(previous.manifest_json + "\n")
            if existing_artifact not in permitted_artifacts:
                artifact_sha256 = hashlib.sha256(
                    existing_artifact.encode("utf-8")
                ).hexdigest()
                artifact_manifest_digest: str | None = None
                try:
                    artifact_manifest_digest = CompetitionManifest.model_validate_json(
                        existing_artifact
                    ).digest()
                except Exception:
                    pass
                revisions = artifact_dir / "manifest.revisions"
                revisions.mkdir(parents=True, exist_ok=True)
                observed_artifact = revisions / f"observed-{artifact_sha256}.json"
                if (
                    observed_artifact.exists()
                    and observed_artifact.read_text(encoding="utf-8")
                    != existing_artifact
                ):
                    raise RuntimeError(
                        "digest-addressed observed manifest artifact has conflicting "
                        "content"
                    )
                if not observed_artifact.exists():
                    _atomic_write(observed_artifact, existing_artifact)
                divergent_artifact = {
                    "artifact_sha256": artifact_sha256,
                    "artifact_manifest_digest": artifact_manifest_digest,
                    "database_manifest_digest": (
                        previous.manifest_digest if previous is not None else None
                    ),
                    "requested_manifest_digest": new_digest,
                    "archived_path": str(observed_artifact),
                }

        row = repository.insert_manifest(
            manifest,
            now=self._clock(),
            actor=self.config.owner_id,
        )
        if divergent_artifact is not None:
            repository.append_event(
                manifest.competition_id,
                "MANIFEST_ARTIFACT_DIVERGENCE",
                actor=self.config.owner_id,
                payload=divergent_artifact,
                now=self._clock(),
            )
        if changed_fields and previous is not None:
            revisions = artifact_dir / "manifest.revisions"
            revisions.mkdir(parents=True, exist_ok=True)
            previous_artifact = revisions / (
                f"{previous.manifest_digest}.normalized.json"
            )
            previous_normalized = previous.manifest_json + "\n"
            if (
                previous_artifact.exists()
                and previous_artifact.read_text(encoding="utf-8")
                != previous_normalized
            ):
                raise RuntimeError(
                    "persisted manifest revision differs from the database record"
                )
            if not previous_artifact.exists():
                _atomic_write(previous_artifact, previous_normalized)
            logger.warning(
                "Competition manifest updated: id={} state={} old_digest={} "
                "new_digest={} changed_fields={}",
                manifest.competition_id,
                previous.status,
                previous.manifest_digest,
                new_digest,
                json.dumps(changed_fields, sort_keys=True, separators=(",", ":")),
            )
        if changed_fields or divergent_artifact is not None or not destination.exists():
            _atomic_write(destination, normalized)
        if divergent_artifact is not None:
            logger.warning(
                "Competition manifest artifact divergence preserved and reconciled: "
                "id={} details={}",
                manifest.competition_id,
                json.dumps(
                    divergent_artifact, sort_keys=True, separators=(",", ":")
                ),
            )
        return row

    def bootstrap_manifests(self) -> list[str]:
        if not self.enabled:
            return []
        manifest_paths = sorted(glob.glob(self.config.manifest_glob))
        if not manifest_paths:
            logger.warning(
                "Competition mode is enabled, but no manifests matched {!r}; "
                "only competitions already registered in the database can be processed",
                self.config.manifest_glob,
            )
            return []

        logger.info(
            "Competition manifest bootstrap found {} manifest(s) matching {!r}",
            len(manifest_paths),
            self.config.manifest_glob,
        )
        registered: list[str] = []
        for path_text in manifest_paths:
            manifest = load_manifest(path_text)
            row = self.register_manifest(
                manifest,
                validate_runtime_paths=True,
                repository_root=Path.cwd(),
            )
            registered.append(manifest.competition_id)
            logger.info(
                "Competition manifest ready: id={} type={} status={} start={} end={} source={}",
                manifest.competition_id,
                manifest.competition_type,
                row.status,
                row.start_time,
                row.end_time,
                path_text,
            )
        return registered

    def tick(self) -> list[tuple[str, CompetitionState]]:
        if not self.enabled:
            return []
        repository = self._require_repository()
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("competition clock must return a timezone-aware datetime")
        now = now.astimezone(timezone.utc)
        transitions: list[tuple[str, CompetitionState]] = []
        snapshots = repository.list_nonterminal()
        if snapshots:
            logger.debug(
                "Competition scheduler tick at {}: {} non-terminal competition(s): {}",
                now.isoformat(),
                len(snapshots),
                ", ".join(
                    f"{snapshot.competition_id}={snapshot.status}"
                    for snapshot in snapshots
                ),
            )
            for snapshot in snapshots:
                countdown = _scheduled_phase_countdown(snapshot, now)
                if countdown is None:
                    continue
                next_state, manifest_field, deadline, time_left = countdown
                logger.debug(
                    "Competition phase countdown: id={} current={} next={} "
                    "deadline={} time_left={} source=manifest.{}",
                    snapshot.competition_id,
                    snapshot.status,
                    next_state.value,
                    deadline.isoformat(),
                    time_left,
                    manifest_field,
                )
        else:
            logger.debug(
                "Competition scheduler tick at {}: no non-terminal competitions",
                now.isoformat(),
            )

        for snapshot in snapshots:
            competition_id = snapshot.competition_id
            if not repository.acquire_scheduler_lease(
                competition_id,
                owner=self.config.owner_id,
                now=now,
                ttl_seconds=self.config.lease_ttl_seconds,
            ):
                logger.debug(
                    "Competition scheduler skipped id={}: lease is held by another owner",
                    competition_id,
                )
                continue
            try:
                current = repository.get(competition_id)
                if current is None:
                    logger.warning(
                        "Competition scheduler could not reload id={} after acquiring its lease",
                        competition_id,
                    )
                    continue
                state = CompetitionState(current.status)
                if state == CompetitionState.SCHEDULED and now >= parse_utc(
                    current.start_time
                ):
                    if repository.another_competition_is_running(competition_id):
                        logger.debug(
                            "Competition id={} is due to start but another competition is active; "
                            "leaving it SCHEDULED",
                            competition_id,
                        )
                        continue
                    try:
                        previous_state = state
                        current = repository.transition(
                            competition_id,
                            CompetitionState.ENROLLING,
                            expected=CompetitionState.SCHEDULED,
                            now=now,
                            actor=self.config.owner_id,
                            payload={"trigger": "competition_start_time"},
                        )
                    except IntegrityError:
                        # A different scheduler may have activated another ID
                        # after our read. The partial unique index is authoritative.
                        logger.info(
                            "Competition id={} lost the activation race to another active "
                            "competition; leaving it SCHEDULED",
                            competition_id,
                        )
                        continue
                    state = CompetitionState(current.status)
                    transitions.append((competition_id, state))
                    logger.info(
                        "Competition state transition: id={} {} -> {} "
                        "(trigger=competition_start_time)",
                        competition_id,
                        previous_state.value,
                        state.value,
                    )
                if state == CompetitionState.ENROLLING and now >= parse_utc(
                    current.contender_finalisation_time
                ):
                    previous_state = state
                    current = repository.transition(
                        competition_id,
                        CompetitionState.FINALIZING_SUBMISSIONS,
                        expected=CompetitionState.ENROLLING,
                        now=now,
                        actor=self.config.owner_id,
                        payload={"trigger": "contender_finalisation_time"},
                    )
                    state = CompetitionState(current.status)
                    transitions.append((competition_id, state))
                    logger.info(
                        "Competition state transition: id={} {} -> {} "
                        "(trigger=contender_finalisation_time)",
                        competition_id,
                        previous_state.value,
                        state.value,
                    )
                if state == CompetitionState.AWAITING_END_TIME and now >= parse_utc(
                    current.end_time
                ):
                    previous_state = state
                    current = repository.transition(
                        competition_id,
                        CompetitionState.COMPLETED,
                        expected=CompetitionState.AWAITING_END_TIME,
                        now=now,
                        actor=self.config.owner_id,
                        payload={"trigger": "competition_end_time"},
                    )
                    transitions.append(
                        (competition_id, CompetitionState(current.status))
                    )
                    logger.info(
                        "Competition state transition: id={} {} -> {} "
                        "(trigger=competition_end_time)",
                        competition_id,
                        previous_state.value,
                        current.status,
                    )
            finally:
                repository.release_scheduler_lease(
                    competition_id, owner=self.config.owner_id
                )
        return transitions

    def complete_current_stage(self, competition_id: str) -> CompetitionState:
        repository = self._require_repository()
        row = repository.get(competition_id)
        if row is None:
            raise KeyError(competition_id)
        current = CompetitionState(row.status)
        target = PIPELINE_SUCCESSOR.get(current)
        if target is None:
            raise ValueError(
                f"state {current.value} is not a completable pipeline stage"
            )
        repository.transition(
            competition_id,
            target,
            expected=current,
            now=self._clock(),
            actor=self.config.owner_id,
            payload={"trigger": "stage_complete"},
        )
        logger.info(
            "Competition state transition: id={} {} -> {} (trigger=stage_complete)",
            competition_id,
            current.value,
            target.value,
        )
        return target

    def fail(self, competition_id: str, reason: str) -> None:
        repository = self._require_repository()
        row = repository.get(competition_id)
        if row is None:
            raise KeyError(competition_id)
        repository.transition(
            competition_id,
            CompetitionState.FAILED,
            expected=CompetitionState(row.status),
            now=self._clock(),
            actor=self.config.owner_id,
            reason=reason,
        )
        logger.error(
            "Competition failed: id={} previous_state={} reason={}",
            competition_id,
            row.status,
            LOG_REDACTOR.redact_text(reason),
        )

    def cancel(self, competition_id: str, reason: str) -> None:
        repository = self._require_repository()
        row = repository.get(competition_id)
        if row is None:
            raise KeyError(competition_id)
        repository.transition(
            competition_id,
            CompetitionState.CANCELLED,
            expected=CompetitionState(row.status),
            now=self._clock(),
            actor=self.config.owner_id,
            reason=reason,
        )
        logger.warning(
            "Competition cancelled: id={} previous_state={} reason={}",
            competition_id,
            row.status,
            LOG_REDACTOR.redact_text(reason),
        )

    async def run(self, *, after_tick: TickHook | None = None) -> None:
        if not self.enabled:
            return
        logger.info(
            "Competition scheduler starting: owner={} interval={}s lease_ttl={}s "
            "manifest_glob={!r} artifact_root={}",
            self.config.owner_id,
            self.config.scheduler_interval_seconds,
            self.config.lease_ttl_seconds,
            self.config.manifest_glob,
            self.config.artifact_root,
        )
        try:
            registered = self.bootstrap_manifests()
            logger.info(
                "Competition scheduler bootstrap complete: {} manifest(s) ready",
                len(registered),
            )
            while True:
                try:
                    await asyncio.to_thread(self.tick)
                    if after_tick is not None:
                        await after_tick()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Competition scheduler cycle failed; retrying after {}s",
                        self.config.scheduler_interval_seconds,
                    )
                await asyncio.sleep(self.config.scheduler_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Competition scheduler stopped")
            raise
        except Exception:
            logger.exception("Competition scheduler stopped after an unexpected error")
            raise
