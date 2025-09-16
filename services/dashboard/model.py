from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import field

@dataclass
class BaseMinerData(ABC):
    validator_uid: int
    validator_hotkey: str
    request_type: str
    miner_uids: List[int]
    miner_hotkeys: List[str]
    vmaf_scores: List[float]
    final_scores: List[float]
    accumulate_scores: List[float]
    applied_multipliers: List[float]
    status: List[str]
    timestamp: str
    
    def __post_init__(self):
        """Validate that list fields have consistent lengths"""
        list_fields = [
            self.miner_uids, 
            self.miner_hotkeys,
            self.vmaf_scores,
            self.final_scores,
            self.accumulate_scores,
            self.status,
            self.applied_multipliers
        ]
        
        lengths = [len(field) for field in list_fields]
        if len(set(lengths)) > 1:
            raise ValueError("All list fields must have the same length")
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

@dataclass
class UpscalingMinerData(BaseMinerData):
    processing_task_type: str = "upscaling"
    pieapp_scores: List[float] = None
    
    def to_dict(self):
        return {
            "validator_uid": self.validator_uid,
            "validator_hotkey": self.validator_hotkey,
            "processing_task_type": self.processing_task_type,
            "request_type": self.request_type,
            "miner_uids": self.miner_uids,
            "miner_hotkeys": self.miner_hotkeys,
            "timestamp": self.timestamp,
            "vmaf_scores": self.vmaf_scores,
            "final_scores": self.final_scores,
            "accumulate_scores": self.accumulate_scores,
            "applied_multipliers": self.applied_multipliers,
            "status": self.status,
        }

@dataclass
class CompressionMinerData(BaseMinerData):
    processing_task_type: str = "compression"
    compression_rates: List[float] = None
    vmaf_thresholds: List[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.compression_rates is not None:
            if len(self.compression_rates) != len(self.miner_uids):
                raise ValueError("compression_rates must have the same length as other list fields")
        if self.vmaf_thresholds is not None:
            if len(self.vmaf_thresholds) != len(self.miner_uids):
                raise ValueError("vmaf_thresholds must have the same length as other list fields")
    def to_dict(self):
        result = {
            "validator_uid": self.validator_uid,
            "validator_hotkey": self.validator_hotkey,
            "processing_task_type": self.processing_task_type,
            "request_type": self.request_type,
            "miner_uids": self.miner_uids,
            "miner_hotkeys": self.miner_hotkeys,
            "timestamp": self.timestamp,
            "vmaf_scores": self.vmaf_scores,
            "vmaf_thresholds": self.vmaf_thresholds,
            "final_scores": self.final_scores,
            "accumulate_scores": self.accumulate_scores,
            "applied_multipliers": self.applied_multipliers,
            "status": self.status,
        }
        
        if self.compression_rates is not None:
            result["compression_rates"] = self.compression_rates
            
        return result

@dataclass
class MinerInfo:
    miners: List[Dict[str, Any]] = field(default_factory=list)
    
    def append(self, miner_uid, miner_hotkey, trust, incentive, emission, daily_reward, processing_task_type):
        self.miners.append({
            "miner_uid": int(miner_uid),
            "miner_hotkey": miner_hotkey,
            "trust": float(trust),
            "incentive": float(incentive),
            "emission": float(emission),
            "daily_reward": float(daily_reward),
            "processing_task_type": processing_task_type
        })
    
    def to_dict(self):
        return {
            "miners": self.miners
        }