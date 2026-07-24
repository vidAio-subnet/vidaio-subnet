"""Persisted competition lifecycle states and transition policy."""

from __future__ import annotations

from enum import Enum


class CompetitionState(str, Enum):
    SCHEDULED = "SCHEDULED"
    ENROLLING = "ENROLLING"
    FINALIZING_SUBMISSIONS = "FINALIZING_SUBMISSIONS"
    VALIDATING = "VALIDATING"
    BUILDING = "BUILDING"
    EVALUATING = "EVALUATING"
    SCORING = "SCORING"
    AWAITING_END_TIME = "AWAITING_END_TIME"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in {
            CompetitionState.COMPLETED,
            CompetitionState.FAILED,
            CompetitionState.CANCELLED,
        }


class ContenderState(str, Enum):
    INVITED = "INVITED"
    PARTICIPATING = "PARTICIPATING"
    RECEIVED = "RECEIVED"
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"
    BUILT = "BUILT"
    RUNNING = "RUNNING"
    SCORED = "SCORED"
    FAILED = "FAILED"


PIPELINE_SUCCESSOR: dict[CompetitionState, CompetitionState] = {
    CompetitionState.FINALIZING_SUBMISSIONS: CompetitionState.VALIDATING,
    CompetitionState.VALIDATING: CompetitionState.BUILDING,
    CompetitionState.BUILDING: CompetitionState.EVALUATING,
    CompetitionState.EVALUATING: CompetitionState.SCORING,
    CompetitionState.SCORING: CompetitionState.AWAITING_END_TIME,
}

ALLOWED_TRANSITIONS: dict[CompetitionState, frozenset[CompetitionState]] = {
    CompetitionState.SCHEDULED: frozenset(
        {
            CompetitionState.ENROLLING,
            CompetitionState.FAILED,
            CompetitionState.CANCELLED,
        }
    ),
    CompetitionState.ENROLLING: frozenset(
        {
            CompetitionState.FINALIZING_SUBMISSIONS,
            CompetitionState.FAILED,
            CompetitionState.CANCELLED,
        }
    ),
    CompetitionState.FINALIZING_SUBMISSIONS: frozenset(
        {
            CompetitionState.VALIDATING,
            CompetitionState.FAILED,
            CompetitionState.CANCELLED,
        }
    ),
    CompetitionState.VALIDATING: frozenset(
        {CompetitionState.BUILDING, CompetitionState.FAILED, CompetitionState.CANCELLED}
    ),
    CompetitionState.BUILDING: frozenset(
        {
            CompetitionState.EVALUATING,
            CompetitionState.FAILED,
            CompetitionState.CANCELLED,
        }
    ),
    CompetitionState.EVALUATING: frozenset(
        {CompetitionState.SCORING, CompetitionState.FAILED, CompetitionState.CANCELLED}
    ),
    CompetitionState.SCORING: frozenset(
        {
            CompetitionState.AWAITING_END_TIME,
            CompetitionState.FAILED,
            CompetitionState.CANCELLED,
        }
    ),
    CompetitionState.AWAITING_END_TIME: frozenset(
        {
            CompetitionState.COMPLETED,
            CompetitionState.FAILED,
            CompetitionState.CANCELLED,
        }
    ),
    CompetitionState.COMPLETED: frozenset(),
    CompetitionState.FAILED: frozenset(),
    CompetitionState.CANCELLED: frozenset(),
}


def assert_transition_allowed(
    current: CompetitionState, target: CompetitionState
) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(
            f"invalid competition transition: {current.value} -> {target.value}"
        )
