from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class MinerMetadata(Base):
    __tablename__ = "miner_metadata"

    uid = Column(Integer, primary_key=True)
    accumulate_score = Column(Float, default=0.0)
    hotkey = Column(String, nullable=True)

    def __init__(self, uid, accumulate_score=0.0, hotkey = None):
        self.uid = uid
        self.accumulate_score = accumulate_score
        self.hotkey = hotkey

    def to_dict(self):
        """Convert metadata to dictionary format."""
        return {"uid": self.uid, "accumulate_score": self.accumulate_score, "hotkey": self.hotkey}
