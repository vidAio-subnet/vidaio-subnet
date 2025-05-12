from typing import List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RequestData:
    validator_uid: int
    validator_hotkey: str
    request_type: str
    task_url: str
    miner_uids: List[int]
    miner_hotkeys: List[str]
    vmaf_scores: List[float]
    pieapp_scores: List[float]
    final_scores: List[float]
    accumulate_scores: List[float]
    processed_url: List[str]
    status: List[str]
    timestamp: str
    
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
            self.processed_url
        ]
        
        # Check if all lists have the same length
        lengths = [len(field) for field in list_fields]
        if len(set(lengths)) > 1:
            raise ValueError("All list fields must have the same length")
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create a RequestData instance from a dictionary"""
        return cls(**data)
    
    def to_dict(self):
        """Convert the RequestData instance to a dictionary"""
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
            "task_url": self.task_url,
            "processed_url": self.processed_url
        }
