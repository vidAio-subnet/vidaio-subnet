import pandas as pd
from ..global_config import CONFIG


def build_rate_limit(metagraph) -> dict[int, int]:
    stake_array = metagraph.S
    w_uids = [
        i
        for i in range(len(stake_array))
        if stake_array[i] > CONFIG.bandwidth.min_stake
    ]
    w_stakes = [stake_array[i] for i in w_uids]
    total_stake = sum(w_stakes)
    normalized_stakes = [stake / total_stake for stake in w_stakes]
    rate_limits = {
        uid: max(10, int(CONFIG.bandwidth.total_requests * normalized_stake))
        for uid, normalized_stake in zip(w_uids, normalized_stakes)
    }
    return rate_limits
