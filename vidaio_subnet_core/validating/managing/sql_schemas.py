# from sqlalchemy import create_engine, Column, Integer, Float, String
# from sqlalchemy.ext.declarative import declarative_base

# Base = declarative_base()


# class MinerMetadata(Base):
#     __tablename__ = "miner_metadata"

#     uid = Column(Integer, primary_key=True)
#     accumulate_score = Column(Float, default=0.0)
#     hotkey = Column(String, nullable=True)

#     def __init__(self, uid, accumulate_score=0.0, hotkey = None):
#         self.uid = uid
#         self.accumulate_score = accumulate_score
#         self.hotkey = hotkey

#     def to_dict(self):
#         """Convert metadata to dictionary format."""
#         return {"uid": self.uid, "accumulate_score": self.accumulate_score, "hotkey": self.hotkey}


from sqlalchemy import create_engine, Column, Integer, Float, String, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()


class MinerMetadata(Base):
    __tablename__ = "miner_metadata"

    uid = Column(Integer, primary_key=True)
    hotkey = Column(String, nullable=True, index=True)
    accumulate_score = Column(Float, default=0.0)
    
    # Performance metrics
    current_s_f = Column(Float, default=0.0)
    current_s_q = Column(Float, default=0.0)
    current_s_l = Column(Float, default=0.0)
    current_s_pre = Column(Float, default=0.0)
    
    # Historical performance tracking
    performance_multiplier = Column(Float, default=1.0)
    bonus_multiplier = Column(Float, default=1.0)
    penalty_f_multiplier = Column(Float, default=1.0)
    penalty_q_multiplier = Column(Float, default=1.0)
    
    # Counters for performance windows
    bonus_count = Column(Integer, default=0)  # S_F > 0.77 count
    penalty_f_count = Column(Integer, default=0)  # S_F < 0.45 count
    penalty_q_count = Column(Integer, default=0)  # S_Q < 0.5 count
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    performance_history = relationship("MinerPerformanceHistory", back_populates="miner", 
                                       cascade="all, delete-orphan", lazy="dynamic")

    def __init__(self, uid, hotkey=None, accumulate_score=0.0):
        self.uid = uid
        self.hotkey = hotkey
        self.accumulate_score = accumulate_score

    def to_dict(self):
        """Convert metadata to dictionary format."""
        return {
            "uid": self.uid,
            "hotkey": self.hotkey,
            "accumulate_score": self.accumulate_score,
            "current_metrics": {
                "s_f": self.current_s_f,
                "s_q": self.current_s_q,
                "s_l": self.current_s_l,
                "s_pre": self.current_s_pre,
            },
            "performance_multipliers": {
                "total": self.performance_multiplier,
                "bonus": self.bonus_multiplier,
                "penalty_f": self.penalty_f_multiplier,
                "penalty_q": self.penalty_q_multiplier,
            },
            "counters": {
                "bonus_count": self.bonus_count,
                "penalty_f_count": self.penalty_f_count,
                "penalty_q_count": self.penalty_q_count,
            },
            "timestamps": {
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            }
        }
    
    def update_performance_metrics(self, s_f, s_q, s_l, s_pre, content_length):
        """Update current performance metrics and create history entry"""
        # Update current metrics
        self.current_s_f = s_f
        self.current_s_q = s_q
        self.current_s_l = s_l
        self.current_s_pre = s_pre
        
        # Create new history entry
        history_entry = MinerPerformanceHistory(
            miner_uid=self.uid,
            s_f=s_f,
            s_q=s_q,
            s_l=s_l,
            s_pre=s_pre,
            content_length=content_length
        )
        
        # Update multiplier counters based on thresholds
        if s_f > 0.77:
            self.bonus_count += 1
        if s_f < 0.45:
            self.penalty_f_count += 1
        if s_q < 0.5:
            self.penalty_q_count += 1
            
        # Return the history entry to be added to the session
        return history_entry
    
    def recalculate_multipliers(self, history_window=10):
        """Recalculate all performance multipliers based on historical data"""
        # Get the most recent entries up to history_window
        recent_entries = self.performance_history.order_by(
            MinerPerformanceHistory.created_at.desc()
        ).limit(history_window).all()
        
        if len(recent_entries) < history_window:
            # Not enough history yet, use default multipliers
            self.performance_multiplier = 1.0
            self.bonus_multiplier = 1.0
            self.penalty_f_multiplier = 1.0
            self.penalty_q_multiplier = 1.0
            return
        
        # Count occurrences in the window
        bonus_count = sum(1 for entry in recent_entries if entry.s_f > 0.77)
        penalty_f_count = sum(1 for entry in recent_entries if entry.s_f < 0.45)
        penalty_q_count = sum(1 for entry in recent_entries if entry.s_q < 0.5)
        
        # Update counter fields
        self.bonus_count = bonus_count
        self.penalty_f_count = penalty_f_count
        self.penalty_q_count = penalty_q_count
        
        # Calculate individual multiplier components
        self.bonus_multiplier = 1.0 + (bonus_count / history_window) * 0.15
        self.penalty_f_multiplier = 1.0 - (penalty_f_count / history_window) * 0.20
        self.penalty_q_multiplier = 1.0 - (penalty_q_count / history_window) * 0.25
        
        # Calculate combined multiplier
        self.performance_multiplier = self.bonus_multiplier * self.penalty_f_multiplier * self.penalty_q_multiplier


class MinerPerformanceHistory(Base):
    __tablename__ = "miner_performance_history"
    
    id = Column(Integer, primary_key=True)
    miner_uid = Column(Integer, ForeignKey("miner_metadata.uid"), index=True)
    
    # Performance metrics
    s_f = Column(Float, default=0.0)
    s_q = Column(Float, default=0.0)
    s_l = Column(Float, default=0.0)
    s_pre = Column(Float, default=0.0)
    
    # Content details
    content_length = Column(Integer, default=5)  # in seconds
    
    # Additional details that could be useful
    vmaf_score = Column(Float, nullable=True)
    pie_app_score = Column(Float, nullable=True)
    
    # Optional: Additional processing metadata as JSON
    processing_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to parent
    miner = relationship("MinerMetadata", back_populates="performance_history")
    
    def __init__(self, miner_uid, s_f, s_q, s_l, s_pre, content_length, 
                 vmaf_score=None, pie_app_score=None, processing_metadata=None):
        self.miner_uid = miner_uid
        self.s_f = s_f
        self.s_q = s_q
        self.s_l = s_l
        self.s_pre = s_pre
        self.content_length = content_length
        self.vmaf_score = vmaf_score
        self.pie_app_score = pie_app_score
        self.processing_metadata = processing_metadata
    
    def to_dict(self):
        """Convert history entry to dictionary format."""
        return {
            "id": self.id,
            "miner_uid": self.miner_uid,
            "metrics": {
                "s_f": self.s_f,
                "s_q": self.s_q,
                "s_l": self.s_l,
                "s_pre": self.s_pre,
            },
            "content_length": self.content_length,
            "vmaf_score": self.vmaf_score,
            "pie_app_score": self.pie_app_score,
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
