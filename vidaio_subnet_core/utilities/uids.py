import random
import bittensor as bt
import numpy as np
from typing import List
import os
import shutil

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

def get_organic_forward_uids(self, count: int = None, vpermit_tao_limit: int = 20000, exclude: List[int] = None) -> np.ndarray:
    """
    Get a list of UIDs that are available for forwarding, sorted by incentive.
    
    Args:
        count (int): Number of UIDs to return
        vpermit_tao_limit (int): Validator permit tao limit
        exclude (List[int]): List of UIDs to exclude
    
    Returns:
        np.ndarray: Array of available UIDs
    """
    exclude = exclude or []
    incentives = self.metagraph.I
    miner_info = [{"uid": uid, "incentive": incentives[uid]} for uid in range(self.metagraph.n.item())]

    sorted_uids = sorted(miner_info, key=lambda x: x["incentive"], reverse=True)
    
    candidate_uids = [
        uid for uid in (miner["uid"] for miner in sorted_uids)
        if check_uid_availability(self.metagraph, uid, vpermit_tao_limit) and uid not in exclude
    ]

    avail_uids = [
        uid for uid in (miner["uid"] for miner in sorted_uids)
        if check_uid_availability(self.metagraph, uid, vpermit_tao_limit)
    ]

    count = min(count, len(avail_uids))
    
    if len(candidate_uids) < count:
        candidate_uids.extend(avail_uids[:count - len(candidate_uids)])
        
    return np.array(candidate_uids[:count])
