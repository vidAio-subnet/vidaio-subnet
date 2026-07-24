"""Competition podium reward policy."""

from __future__ import annotations


COMPETITION_REWARD_SHARES = (0.70, 0.20, 0.10)


def competition_reward_share(rank: int) -> float:
    if 1 <= rank <= len(COMPETITION_REWARD_SHARES):
        return COMPETITION_REWARD_SHARES[rank - 1]
    return 0.0


__all__ = [
    "COMPETITION_REWARD_SHARES",
    "competition_reward_share",
]
