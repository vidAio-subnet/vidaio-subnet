from collections.abc import MutableMapping, Sequence
from typing import TypeAlias


CompressionScoreSignature: TypeAlias = tuple[
    float,
    float | None,
    float,
    float,
    float,
]
CompressionScoreCache: TypeAlias = MutableMapping[
    str,
    MutableMapping[CompressionScoreSignature, int],
]


def find_duplicate_compression_scores(
    uids: Sequence[int],
    synthetic_input_ids: Sequence[str],
    vmaf_scores: Sequence[float],
    base_vmaf_scores: Sequence[float | None],
    vmaf_thresholds: Sequence[float],
    compression_rates: Sequence[float],
    final_scores: Sequence[float],
    cache: CompressionScoreCache,
) -> dict[int, int]:
    """Return result indices whose exact score signature was claimed by an earlier UID.

    The cache is partitioned by synthetic input so two different challenges can
    legitimately produce the same metrics. Non-positive results do not represent
    a successful solution and therefore neither claim nor consume a slot.
    """
    lengths = {
        len(uids),
        len(synthetic_input_ids),
        len(vmaf_scores),
        len(base_vmaf_scores),
        len(vmaf_thresholds),
        len(compression_rates),
        len(final_scores),
    }
    if len(lengths) != 1:
        raise ValueError("Compression score fields must have equal lengths")

    duplicate_owners: dict[int, int] = {}
    for idx, (
        uid,
        synthetic_input_id,
        vmaf_score,
        base_vmaf_score,
        vmaf_threshold,
        compression_rate,
        final_score,
    ) in enumerate(
        zip(
            uids,
            synthetic_input_ids,
            vmaf_scores,
            base_vmaf_scores,
            vmaf_thresholds,
            compression_rates,
            final_scores,
        )
    ):
        if not synthetic_input_id or final_score <= 0:
            continue

        signature: CompressionScoreSignature = (
            vmaf_score,
            base_vmaf_score,
            vmaf_threshold,
            compression_rate,
            final_score,
        )
        input_cache = cache.setdefault(synthetic_input_id, {})
        first_uid = input_cache.setdefault(signature, uid)
        if first_uid != uid:
            duplicate_owners[idx] = first_uid

    return duplicate_owners
