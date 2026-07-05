import math
from collections.abc import MutableMapping, Sequence
from typing import TypeAlias


CompressionScoreClaim: TypeAlias = tuple[
    float,
    float | None,
    float,
    int,
]
CompressionScoreClaim: TypeAlias = tuple[float, int]
CompressionScoreCache: TypeAlias = MutableMapping[
    str,
    MutableMapping[str, list[CompressionScoreClaim]],
]


VMAF_DUPLICATE_TOLERANCE = 0.2
COMPRESSION_RATE_DUPLICATE_TOLERANCE = 0.01


def _vmaf_matches(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=VMAF_DUPLICATE_TOLERANCE,
    )


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
    """Return result indices whose normalized signature belongs to an earlier UID.

    The cache is partitioned by synthetic input so two different challenges can
    legitimately produce the same metrics. Non-positive results do not represent
    a successful solution and therefore neither claim nor consume a slot. Primary
    VMAF and compression-rate metrics use explicit tolerances so minor encoder or
    measurement drift does not let the same solution claim multiple slots. The
    VMAF threshold must still match. Final score is derived from those metrics and
    is only used here to identify successful rows.
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

        threshold_signature = f"{vmaf_threshold:.2f}"
        input_cache = cache.setdefault(synthetic_input_id, {})
        claims = input_cache.setdefault(threshold_signature, [])
        first_uid = next(
            (
                owner_uid
                for (
                    claimed_vmaf_score,
                    claimed_base_vmaf_score,
                    claimed_rate,
                    owner_uid,
                ) in claims
                if (
                    _vmaf_matches(vmaf_score, claimed_vmaf_score)
                    and _vmaf_matches(base_vmaf_score, claimed_base_vmaf_score)
                    and math.isclose(
                        compression_rate,
                        claimed_rate,
                        rel_tol=0.0,
                        abs_tol=COMPRESSION_RATE_DUPLICATE_TOLERANCE,
                    )
                )
            ),
            None,
        )
        if first_uid is None:
            claims.append((vmaf_score, base_vmaf_score, compression_rate, uid))
        elif first_uid != uid:
            duplicate_owners[idx] = first_uid

    return duplicate_owners
