"""Deterministic competition-wide evaluation batch assignments."""

from __future__ import annotations

from collections.abc import Iterable


def canonical_batch_assignments(
    items: Iterable[tuple[str, str]],
    batch_size: int,
) -> dict[str, tuple[int, int]]:
    """Assign ``evaluation_id`` values to stable zero-based batches.

    Items are ordered by evaluation ID. Query variants that share a source path
    are placed in different batches so a contender never processes the same
    source video twice in one request.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pending = sorted(items)
    evaluation_ids = [evaluation_id for evaluation_id, _source_path in pending]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("evaluation IDs must be unique")

    assignments: dict[str, tuple[int, int]] = {}
    batch_index = 0
    while pending:
        selected: list[tuple[str, str]] = []
        deferred: list[tuple[str, str]] = []
        selected_sources: set[str] = set()
        for item in pending:
            evaluation_id, source_path = item
            if len(selected) < batch_size and source_path not in selected_sources:
                selected.append(item)
                selected_sources.add(source_path)
            else:
                deferred.append(item)
        for position, (evaluation_id, _source_path) in enumerate(selected):
            assignments[evaluation_id] = (batch_index, position)
        pending = deferred
        batch_index += 1
    return assignments


__all__ = ["canonical_batch_assignments"]
