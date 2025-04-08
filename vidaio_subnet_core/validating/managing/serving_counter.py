from ...global_config import CONFIG
import redis
from loguru import logger


class ServingCounter:

    def __init__(
        self,
        rate_limit: int,
        uid: int,
        redis_client: redis.Redis,
    ):
        self.rate_limit = rate_limit
        self.redis_client = redis_client
        self.key = f"{CONFIG.redis.miner_manager_key}:{uid}"

    def increment(self) -> bool:
        """
        Increment request counter and check rate limit.

        Uses atomic Redis INCR operation and sets expiry on first increment.

        Reset the counter after EPOCH_LENGTH seconds.

        Returns:
            bool: True if under rate limit, False if exceeded
        """
        count = self.redis_client.incr(self.key)

        if count == 1:
            self.redis_client.expire(self.key, CONFIG.bandwidth.request_interval)

        if count <= self.rate_limit + 1:
            return True

        logger.info(f"Rate limit exceeded for {self.key}")
        return False
