from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class MinerMetadata(Base):
    """
    Stores aggregate miner statistics and current state information
    """
    __tablename__ = "miner_metadata"
    
    processing_task_type = Column(String(32), nullable=True)

    uid = Column(Integer, primary_key=True)
    hotkey = Column(String(64), nullable=False)
    accumulate_score = Column(Float, default=0.0)
    
    bonus_multiplier = Column(Float, default=1.0)
    penalty_f_multiplier = Column(Float, default=1.0)
    penalty_q_multiplier = Column(Float, default=1.0)
    
    total_multiplier = Column(Float, default=1.0)
    
    avg_s_q = Column(Float, default=0.0)
    avg_s_l = Column(Float, default=0.0)
    avg_s_f = Column(Float, default=0.0)
    
    bonus_count = Column(Integer, default=0)
    penalty_q_count = Column(Integer, default=0)
    penalty_f_count = Column(Integer, default=0)

    avg_content_length = Column(Float, default=0.0)
    avg_compression_rate = Column(Float, default=0.0)

    last_update_timestamp = Column(DateTime, default=datetime.now)
    total_rounds_completed = Column(Integer, default=0)
    performance_tier = Column(String(32), default='New Miner')
    
    success_rate = Column(Float, default=0.0)
    longest_content_processed = Column(Float, default=0.0)
    
    # Compression-specific fields
    
    performance_history = relationship("MinerPerformanceHistory", 
                                      back_populates="miner",
                                      order_by="desc(MinerPerformanceHistory.timestamp)",
                                      lazy="dynamic")
    
    def __repr__(self):
        return f"<Miner(uid='{self.uid}', tier='{self.performance_tier}')>"


class MinerPerformanceHistory(Base):
    """
    Stores detailed performance history for each mining round
    """
    __tablename__ = "miner_performance_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(Integer, ForeignKey('miner_metadata.uid'), nullable=False)
    round_id = Column(String(64), nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    
    processed_task_type = Column(String(32), nullable=True)
    
    s_q = Column(Float, nullable=False)
    s_l = Column(Float, nullable=False)
    s_f = Column(Float, nullable=False)
    
    content_length = Column(Float, nullable=False)
    content_type = Column(String(32))
    
    vmaf_score = Column(Float)
    pie_app_score = Column(Float)
    processing_time = Column(Float)
    
    # Compression-specific fields
    compression_rate = Column(Float, default=0.0)
    vmaf_threshold = Column(Float, default=0.0)
    
    applied_multiplier = Column(Float, default=1.0)
    
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    miner = relationship("MinerMetadata", back_populates="performance_history")
    
    def __repr__(self):
        return f"<Performance(uid='{self.uid}', round='{self.round_id}', s_f={self.s_f:.2f})>"
