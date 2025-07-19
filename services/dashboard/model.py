from dataclasses import dataclass
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class MinerData:
    validator_uid: int
    validator_hotkey: str
    request_type: str
    miner_uids: List[int]
    miner_hotkeys: List[str]
    vmaf_scores: List[float]
    pieapp_scores: List[float]
    final_scores: List[float]
    accumulate_scores: List[float]
    task_urls: List[str]
    processed_urls: List[str]
    status: List[str]
    timestamp: str
    p_time: List[float]
    
    def __post_init__(self):
        """Validate that list fields have consistent lengths"""
        list_fields = [
            self.miner_uids, 
            self.miner_hotkeys,
            self.vmaf_scores,
            self.pieapp_scores,
            self.final_scores,
            self.accumulate_scores,
            self.status,
            self.task_urls,
            self.processed_urls,
            self.p_time
        ]
        
        # Check if all lists have the same length
        lengths = [len(field) for field in list_fields]
        if len(set(lengths)) > 1:
            raise ValueError("All list fields must have the same length")
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create a MinerData instance from a dictionary"""
        return cls(**data)
    
    def to_dict(self):
        """Convert the MinerData instance to a dictionary"""
        return {
            "validator_uid": self.validator_uid,
            "validator_hotkey": self.validator_hotkey,
            "miner_uids": self.miner_uids,
            "miner_hotkeys": self.miner_hotkeys,
            "timestamp": self.timestamp,
            "request_type": self.request_type,
            "vmaf_scores": self.vmaf_scores,
            "pieapp_scores": self.pieapp_scores,
            "final_scores": self.final_scores,
            "accumulate_scores": self.accumulate_scores,
            "status": self.status,
            "task_url": self.task_urls,
            "processed_url": self.processed_urls,
            "p_time": self.p_time
        }

@dataclass
class MinerInfo:
    """
    Class representing information about a miner.
    """
    miner_uid: int
    miner_hotkey: str
    trust: float
    incentive: float
    emission: float
    daily_reward: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the MinerInfo instance to a dictionary"""
        return {
            "miner_uid": self.miner_uid,
            "miner_hotkey": self.miner_hotkey,
            "trust": self.trust,
            "incentive": self.incentive,
            "emission": self.emission,
            "daily_reward": self.daily_reward
        }