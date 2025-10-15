import random
import bittensor as bt
import numpy as np
from typing import List
import os
import shutil
from loguru import logger

def check_uid_availability(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    """
    Check if a UID is available. The UID should be available if it is serving and has less than the specified stake limit.
    
    Args:
        metagraph (bt.metagraph.Metagraph): Metagraph object
        uid (int): UID to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    
    Returns:
        bool: True if UID is available, False otherwise
    """
    axon = metagraph.axons[uid]
    if not axon.is_serving:
        return False
    if metagraph.validator_permit[uid] and metagraph.S[uid] >= vpermit_tao_limit:
        return False
    return True

def get_organic_forward_uids(self, count: int = None, task_type : str = None, vpermit_tao_limit: int = 100000000, exclude: List[int] = None) -> np.ndarray:
    """
    Get a list of UIDs that are available for forwarding, randomly sampled from
    incentive-ranked groups of 20 (top 20, next 20, etc.) until count is filled.

    Args:
        count (int): Number of UIDs to return
        task_type (str): Task type to filter miners by
        vpermit_tao_limit (int): Validator permit tao limit
        exclude (List[int]): List of UIDs to exclude

    Returns:
        np.ndarray: Array of selected UIDs
    """
    exclude = exclude or []
    incentives = self.metagraph.I

    miner_uids, task_types, content_lengths = self.miner_manager.get_miner_task_info()

    # Filter by task type and content length
    filtered_uids = [
        uid for i, uid in enumerate(miner_uids)
        if task_types[i] == task_type and content_lengths[i] > 7.5
    ]

    # Sort miners by incentive (descending)
    sorted_uids = sorted(
        [{"uid": uid, "incentive": incentives[uid]} for uid in filtered_uids],
        key=lambda x: x["incentive"],
        reverse=True
    )

    chosen_uids = []
    batch_size = 20
    start_idx = 0

    while len(chosen_uids) < count and start_idx < len(sorted_uids):
        end_idx = start_idx + batch_size
        batch = sorted_uids[start_idx:end_idx]
        batch_uids = [
            m["uid"] for m in batch
            if check_uid_availability(self.metagraph, m["uid"], vpermit_tao_limit)
            and m["uid"] not in exclude
        ]

        # Number of UIDs still needed
        remaining = count - len(chosen_uids)
        if batch_uids:
            selected = np.random.choice(
                batch_uids,
                size=min(remaining, len(batch_uids)),
                replace=False
            )
            chosen_uids.extend(selected)

        start_idx += batch_size  # move to next 20

    return np.array(chosen_uids, dtype=int)