import bittensor as bt
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger
from sqlalchemy.ext.declarative import declarative_base
from .sql_schemas import MinerMetadata
from .serving_counter import ServingCounter
from ...global_config import CONFIG


class MinerManager:
    def __init__(self, uid, wallet, metagraph):
        self.uid = uid
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = metagraph
        self.redis_client = redis.Redis(
            host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db
        )
        self.engine = create_engine(CONFIG.sql.url)
        declarative_base().metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.initialize_serving_counter(self.metagraph.uids)

    def initialize_serving_counter(self, uids: list[int]):
        self.serving_counters = {
            uid: ServingCounter(
                rate_limit=CONFIG.rate_limit,
                uid=uid,
                redis_client=self.redis_client,
            )
            for uid in uids
        }

    def query(self, uids: list[int] = []) -> dict[int, MinerMetadata]:
        query = self.session.query(MinerMetadata)
        if uids:
            query = query.filter(MinerMetadata.uid.in_(uids))
        return {miner.uid: miner for miner in query.all()}

    def step(self, scores: list[float], total_uids: list[int]):
        for uid, score in zip(total_uids, scores):
            miner = self.query([uid])
            if miner is None:
                miner = MinerMetadata(uid=uid)
                self.session.add(miner)
            # EMA with 0.9 decay factor
            miner.accumulate_score = (
                miner.accumulate_score * CONFIG.score.decay_factor
                + score * (1 - CONFIG.score.decay_factor)
            )
            miner.accumulate_score = max(0, miner.accumulate_score)
        self.session.commit()
        logger.success(f"Updated metadata for {len(total_uids)} uids")
