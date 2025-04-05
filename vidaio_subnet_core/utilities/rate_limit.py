import pandas as pd
from ..global_config import CONFIG
from loguru import logger

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

    logger.info(f"Total stake: {total_stake}")
    logger.info(f"Validator currentl stake: {stakes[uids.index(uid)]}")

    if total_stake == 0:
        return 0  # Avoid division by zero

    # Compute normalized stake
    normalized_stake = stakes[uids.index(uid)] / total_stake
    vali_rate_limit = int(CONFIG.bandwidth.total_requests * normalized_stake)
    logger.info(f"Validator rate limit: {vali_rate_limit}")

    return vali_rate_limit
