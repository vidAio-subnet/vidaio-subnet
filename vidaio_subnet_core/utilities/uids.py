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
    Get a list of UIDs that are available for forwarding, selected from
    the top 10 miners by accumulate_score for the given task_type. 
    The top x miners chosen here is consistent with `TOP_N` in the miner manager.

    Args:
        count (int): Number of UIDs to return
        task_type (str): Task type to filter miners by ('upscaling' or 'compression')
        vpermit_tao_limit (int): Validator permit tao limit
        exclude (List[int]): List of UIDs to exclude

    Returns:
        np.ndarray: Array of selected UIDs
    """
    exclude = exclude or []

    # Get top 20 hotkeys for this task type from MinerMetadata, ordered by accumulate_score desc
    top_hotkeys = self.miner_manager.get_top_hotkeys_by_task(task_type, limit=10)

    if not top_hotkeys:
        logger.warning(f"No hotkeys found for task_type={task_type}")
        return np.array([], dtype=int)

    # Build a hotkey -> uid mapping from the metagraph
    hotkey_to_uid = {}
    for uid, axon in enumerate(self.metagraph.axons):
        hotkey_to_uid[axon.hotkey] = uid

    # Map hotkeys to UIDs, preserving the accumulate_score ordering
    candidate_uids = []
    for hotkey in top_hotkeys:
        uid = hotkey_to_uid.get(hotkey)
        if uid is not None \
                and uid not in exclude \
                and check_uid_availability(self.metagraph, uid, vpermit_tao_limit):
            candidate_uids.append(uid)

    if not candidate_uids:
        logger.warning(f"No available UIDs found for task_type={task_type}")
        return np.array([], dtype=int)

    # Sample up to `count` UIDs from the candidates
    selected = np.random.choice(
        candidate_uids,
        size=min(count, len(candidate_uids)),
        replace=False
    )

    return np.array(selected, dtype=int)