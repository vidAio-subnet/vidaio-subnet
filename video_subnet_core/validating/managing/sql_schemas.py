from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class MinerMetadata(Base):
    __tablename__ = "miner_metadata"

    uid = Column(Integer, primary_key=True)
    accumulate_score = Column(Float, default=0.0)

    def __init__(self, uid, accumulate_score=0.0):
        self.uid = uid
        self.accumulate_score = accumulate_score

    def to_dict(self):
        """Convert metadata to dictionary format."""
        return {"uid": self.uid, "accumulate_score": self.accumulate_score}
