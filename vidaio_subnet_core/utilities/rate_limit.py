# import pandas as pd
# from ..global_config import CONFIG


# def build_rate_limit(metagraph, uid: int) -> int:
#     stake_array = metagraph.S
#     w_uids = [
#         i
#         for i in range(len(stake_array))
#         if stake_array[i] > CONFIG.bandwidth.min_stake
#     ]
#     if uid not in w_uids:
#         return 0
#     w_stakes = [stake_array[i] for i in w_uids]
#     total_stake = sum(w_stakes)
#     normalized_stakes = [stake / total_stake for stake in w_stakes]
#     rate_limit = CONFIG.bandwidth.total_requests * normalized_stakes[w_uids.index(uid)]
#     return rate_limit



import pandas as pd
from ..global_config import CONFIG

def build_rate_limit(metagraph, uid: int) -> int:
    """
    Compute the rate limit for a given UID based on stake weight.

    Args:
        metagraph: The metagraph object containing stake data.
        uid (int): The user ID whose rate limit needs to be computed.

    Returns:
        int: The computed rate limit for the given UID.
    """
    stake_array = metagraph.S
    min_stake = CONFIG.bandwidth.min_stake

    # Filter nodes with valid stakes
    valid_uids_stakes = [(i, stake) for i, stake in enumerate(stake_array) if stake > min_stake]

    if not valid_uids_stakes or uid not in [uid for uid, _ in valid_uids_stakes]:
        return 0  # UID is not eligible for rate limiting

    # Separate UID list and stakes
    uids, stakes = zip(*valid_uids_stakes)
    total_stake = sum(stakes)

    if total_stake == 0:
        return 0  # Avoid division by zero

    # Compute normalized stake
    normalized_stake = stakes[uids.index(uid)] / total_stake
    return int(CONFIG.bandwidth.total_requests * normalized_stake)
