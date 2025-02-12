import bittensor as bt
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
from .sql_schemas import MinerMetadata
from .serving_counter import ServingCounter
from ...global_config import CONFIG
from ...utilities.rate_limit import build_rate_limit


class MinerManager:
    def __init__(self, uid, wallet, metagraph):
        logger.info(f"Initializing MinerManager with uid: {uid}")
        self.uid = uid
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = metagraph
        logger.info(f"Connecting to Redis at {CONFIG.redis.host}:{CONFIG.redis.port}")
        self.redis_client = redis.Redis(
            host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db
        )
        logger.info(f"Creating SQL engine with URL: {CONFIG.sql.url}")
        self.engine = create_engine(CONFIG.sql.url)
        declarative_base().metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        logger.info("Initializing serving counters")
        self.initialize_serving_counter(self.metagraph.uids)
        logger.success("MinerManager initialization complete")
        
    def initialize_serving_counter(self, uids: list[int]):
        rate_limit = build_rate_limit(self.metagraph, self.uid)
        logger.info(f"Creating serving counters for {len(uids)} UIDs with rate_limit {rate_limit}")
        self.serving_counters = {
            uid: ServingCounter(
                rate_limit=build_rate_limit(self.metagraph, uid),  # Use uid here
                uid=uid,
                redis_client=self.redis_client,
            )
            for uid in uids
        }
        logger.debug("Serving counters initialized with rate limits")  # Fixed closing brace
        for uid in uids:
            print(f"{uid}: {self.serving_counters[uid].rate_limit}")    
    

    def query(self, uids: list[int] = []) -> dict[int, MinerMetadata]:
        logger.debug(f"Querying metadata for UIDs: {uids if uids else 'all'}")
        query = self.session.query(MinerMetadata)
        if uids:
            query = query.filter(MinerMetadata.uid.in_(uids))
        result = {miner.uid: miner for miner in query.all()}
        logger.debug(f"Found {len(result)} miner metadata records")
        return result

    def step(self, scores: list[float], total_uids: list[int]):
        logger.info(f"Updating scores for {len(total_uids)} miners")
        for uid, score in zip(total_uids, scores):
            logger.debug(f"Processing UID {uid} with score {score}")
            miner = self.query([uid])
            if miner is None:
                logger.info(f"Creating new metadata record for UID {uid}")
                miner = MinerMetadata(uid=uid)
                self.session.add(miner)
            # EMA with decay factor
            miner.accumulate_score = (
                miner.accumulate_score * CONFIG.score.decay_factor
                + score * (1 - CONFIG.score.decay_factor)
            )
            miner.accumulate_score = max(0, miner.accumulate_score)
            logger.debug(
                f"Updated accumulate_score for UID {uid}: {miner.accumulate_score}"
            )
        self.session.commit()
        logger.success(f"Updated metadata for {len(total_uids)} uids")

    def consume(self, uids: list[int]) -> list[int]:
        logger.info(f"Consuming {len(uids)} UIDs")
        filtered_uids = [uid for uid in uids if self.serving_counters[uid].increment()]
        logger.info(f"Filtered to {len(filtered_uids)} UIDs after rate limiting")
        return filtered_uids

    @property
    def weights(self):
        uids = []
        scores = []
        for uid, miner in self.query().items():
            uids.append(uid)
            scores.append(miner.accumulate_score)

        scores = np.array(scores)
        scores = scores / scores.sum()
        return uids, scores
