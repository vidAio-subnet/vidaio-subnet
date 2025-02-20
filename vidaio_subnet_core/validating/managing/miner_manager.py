# import bittensor as bt
# import redis
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from loguru import logger
# # from sqlalchemy.ext.declarative import declarative_base
# import numpy as np
# from .sql_schemas import MinerMetadata, Base
# from .serving_counter import ServingCounter
# from ...global_config import CONFIG
# from ...utilities.rate_limit import build_rate_limit


# class MinerManager:
#     def __init__(self, uid, wallet, metagraph):
#         logger.info(f"Initializing MinerManager with uid: {uid}")
#         self.uid = uid
#         self.wallet = wallet
#         self.dendrite = bt.dendrite(wallet=self.wallet)
#         self.metagraph = metagraph
#         logger.info(f"Connecting to Redis at {CONFIG.redis.host}:{CONFIG.redis.port}")
#         self.redis_client = redis.Redis(
#             host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db
#         )
#         logger.info(f"Creating SQL engine with URL: {CONFIG.sql.url}")
#         self.engine = create_engine(CONFIG.sql.url)
#         Base.metadata.create_all(self.engine)
#         Session = sessionmaker(bind=self.engine)
#         self.session = Session()
#         logger.info("Initializing serving counters")
#         self.initialize_serving_counter(self.metagraph.uids)
#         logger.success("MinerManager initialization complete")
        
#     def initialize_serving_counter(self, uids: list[int]):
#         rate_limit = build_rate_limit(self.metagraph, self.uid)
#         logger.info(f"Creating serving counters for {len(uids)} UIDs with rate_limit {rate_limit}")
#         self.serving_counters = {
#             uid: ServingCounter(
#                 rate_limit=build_rate_limit(self.metagraph, uid),  # Use uid here
#                 uid=uid,
#                 redis_client=self.redis_client,
#             )
#             for uid in uids
#         }
#         logger.debug("Serving counters initialized with rate limits")  # Fixed closing brace
#         for uid in uids:
#             print(f"{uid}: {self.serving_counters[uid].rate_limit}")    
    

#     def query(self, uids: list[int] = []) -> dict[int, MinerMetadata]:
#         logger.debug(f"Querying metadata for UIDs: {uids if uids else 'all'}")
#         query = self.session.query(MinerMetadata)
#         if uids:
#             query = query.filter(MinerMetadata.uid.in_(uids))
#         result = {miner.uid: miner for miner in query.all()}
#         logger.debug(f"Found {len(result)} miner metadata records")
#         return result

#     def step(self, scores: list[float], total_uids: list[int]):
#         logger.info(f"Updating scores for {len(total_uids)} miners")
#         for uid, score in zip(total_uids, scores):
#             logger.debug(f"Processing UID {uid} with score {score}")
#             miner = self.query([uid]).get(uid, None)
#             logger.info(f"Miner: {miner}")
#             if miner is None:
#                 logger.info(f"Creating new metadata record for UID {uid}")
#                 miner = MinerMetadata(uid=uid)
#                 self.session.add(miner)
#             # EMA with decay factor
#             miner.accumulate_score = (
#                 miner.accumulate_score * CONFIG.score.decay_factor
#                 + score * (1 - CONFIG.score.decay_factor)
#             )
#             miner.accumulate_score = max(0, miner.accumulate_score)
#             logger.debug(
#                 f"Updated accumulate_score for UID {uid}: {miner.accumulate_score}"
#             )
#         self.session.commit()
#         logger.success(f"Updated metadata for {len(total_uids)} uids")

#     def consume(self, uids: list[int]) -> list[int]:
#         logger.info(f"Consuming {len(uids)} UIDs")
#         filtered_uids = [uid for uid in uids if self.serving_counters[uid].increment()]
#         logger.info(f"Filtered to {len(filtered_uids)} UIDs after rate limiting")
#         return filtered_uids

#     @property
#     def weights(self):
#         uids = []
#         scores = []
        
#         # Collect uids and scores
#         for uid, miner in self.query().items():
#             uids.append(uid)
#             scores.append(miner.accumulate_score)

#         # Convert to NumPy array
#         scores = np.array(scores)

#         # Normalize scores
#         scores = scores / scores.sum()

#         # Sort uids and rearrange scores accordingly
#         sorted_indices = np.argsort(uids)  # Get sorting indices for uids
#         uids = np.array(uids)[sorted_indices]  # Apply sorting to uids
#         scores = scores[sorted_indices]  # Reorder scores to match sorted uids

#         return uids, scores




import bittensor as bt
import redis
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

from .sql_schemas import MinerMetadata, Base
from .serving_counter import ServingCounter
from ...global_config import CONFIG
from ...utilities.rate_limit import build_rate_limit


class MinerManager:
    def __init__(self, uid, wallet, metagraph):
        """
        Initializes the MinerManager, handling Redis, SQL, and serving counters.

        Args:
            uid (int): Unique identifier of the miner.
            wallet (bt.wallet): Wallet instance for transactions.
            metagraph: Metagraph containing stake data.
        """
        logger.info(f"Initializing MinerManager for UID: {uid}")

        self.uid = uid
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = metagraph

        # Connect to Redis
        self._initialize_redis()

        # Connect to SQL
        self._initialize_sql()

        # Initialize serving counters
        self.initialize_serving_counter(metagraph.uids)

        logger.success("MinerManager initialization complete")

    def _initialize_redis(self):
        """Connect to Redis and handle errors gracefully."""
        try:
            logger.info(f"Connecting to Redis at {CONFIG.redis.host}:{CONFIG.redis.port}")
            self.redis_client = redis.Redis(
                host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _initialize_sql(self):
        """Set up the SQL database connection and session."""
        try:
            logger.info(f"Creating SQL engine with URL: {CONFIG.sql.url}")
            self.engine = create_engine(CONFIG.sql.url)
            Base.metadata.create_all(self.engine)
            self.session = sessionmaker(bind=self.engine)()
        except Exception as e:
            logger.error(f"Failed to connect to the database: {e}")
            raise

    def initialize_serving_counter(self, uids: list[int]):
        """
        Initializes the serving counter for each UID based on stake-weighted rate limits.

        Args:
            uids (list[int]): List of UIDs for which serving counters should be created.
        """
        logger.info(f"Initializing serving counters for {len(uids)} UIDs")

        self.serving_counters = {}
        for uid in uids:
            rate_limit = build_rate_limit(self.metagraph, uid)
            self.serving_counters[uid] = ServingCounter(
                rate_limit=rate_limit, uid=uid, redis_client=self.redis_client
            )
        
        logger.debug("Serving counters initialized successfully")

    def query(self, uids: list[int] = None) -> dict[int, MinerMetadata]:
        """
        Fetch miner metadata from the database.

        Args:
            uids (list[int], optional): List of UIDs to query. If None, queries all.

        Returns:
            dict[int, MinerMetadata]: A dictionary of miner metadata objects keyed by UID.
        """
        try:
            logger.debug(f"Querying miner metadata for: {uids if uids else 'all UIDs'}")
            query = self.session.query(MinerMetadata)
            if uids:
                query = query.filter(MinerMetadata.uid.in_(uids))
            result = {miner.uid: miner for miner in query.all()}
            logger.debug(f"Retrieved {len(result)} miner records")
            return result
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return {}

    def step(self, scores: list[float], total_uids: list[int]):
        """
        Updates miner scores using Exponential Moving Average (EMA).

        Args:
            scores (list[float]): List of updated scores.
            total_uids (list[int]): Corresponding UIDs.
        """
        logger.info(f"Updating scores for {len(total_uids)} miners")

        try:
            miners = self.query(total_uids)

            for uid, score in zip(total_uids, scores):
                miner = miners.get(uid)
                if miner is None:
                    logger.info(f"Creating new metadata record for UID {uid}")
                    miner = MinerMetadata(uid=uid)
                    self.session.add(miner)

                # Apply EMA update
                decay_factor = CONFIG.score.decay_factor
                miner.accumulate_score = (
                    miner.accumulate_score * decay_factor + score * (1 - decay_factor)
                )
                miner.accumulate_score = max(0, miner.accumulate_score)

                logger.debug(f"Updated UID {uid} score to {miner.accumulate_score}")

            self.session.commit()
            logger.success(f"Scores updated for {len(total_uids)} UIDs")
        except Exception as e:
            logger.error(f"Error updating miner scores: {e}")
            self.session.rollback()

    def consume(self, uids: list[int]) -> list[int]:
        """
        Filters UIDs based on their rate limit consumption.

        Args:
            uids (list[int]): List of UIDs requesting consumption.

        Returns:
            list[int]: List of UIDs that passed the rate limit check.
        """
        logger.info(f"Consuming {len(uids)} UIDs")
        filtered_uids = [uid for uid in uids if self.serving_counters[uid].increment()]
        logger.info(f"{len(filtered_uids)} UIDs allowed after rate limiting")
        return filtered_uids

    @property
    def weights(self):
        """
        Computes and normalizes miner scores, returning sorted UIDs and corresponding weights.

        Returns:
            tuple: (uids, scores) where both are sorted numpy arrays.
        """
        try:
            miners = self.query()

            if not miners:
                logger.warning("No miner data available for weight calculation")
                return np.array([]), np.array([])

            uids, scores = zip(*[(uid, miner.accumulate_score) for uid, miner in miners.items()])

            # Convert scores to NumPy array and normalize
            scores = np.array(scores, dtype=np.float64)
            total_score = scores.sum()

            if total_score == 0:
                logger.warning("Total score is zero, returning uniform distribution")
                return np.array(uids), np.ones_like(scores) / len(scores)

            scores /= total_score  # Normalize scores

            # Sort UIDs and apply sorting to scores
            sorted_indices = np.argsort(uids)
            return np.array(uids)[sorted_indices], scores[sorted_indices]

        except Exception as e:
            logger.error(f"Error computing weights: {e}")
            return np.array([]), np.array([])
