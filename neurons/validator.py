from video_subnet_core import validating, ultilites, CONFIG, base, protocol
import bittensor as bt
import random
import aiohttp
import asyncio
from loguru import logger

class Validator(base.BaseValidator):
    def __init__(self, config: bt.Config):
        super().__init__(config)
        self.config = config
        logger.info(f"Initializing validator with config: {config}")
        self.miner_manager = validating.managing.MinerManager(
            uid=config.uid, wallet=self.wallet, metagraph=self.metagraph
        )
        logger.info("Initialized miner manager")
        self.challenge_synthesizer = validating.synthesizing.Synthesizer()
        logger.info("Initialized challenge synthesizer")
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info("Initialized dendrite")
        self.score_client = aiohttp.ClientSession(
            base_url=f"{CONFIG.score.host}:{CONFIG.score.port}"
        )
        logger.info(f"Initialized score client with base URL: {CONFIG.score.host}:{CONFIG.score.port}")

    async def forward(self):
        logger.info("Starting forward pass")
        uids = list(range(self.metagraph.hotkeys))
        logger.debug(f"Initial UIDs: {uids}")
        uids = self.miner_manager.consume(uids)
        logger.info(f"Filtered UIDs after consumption: {uids}")
        axons = [self.metagraph.axons[uid] for uid in uids]
        miners = zip(axons, uids)
        batch_size = 4
        miner_batches = [
            miners[i : i + batch_size] for i in range(0, len(miners), batch_size)
        ]
        logger.info(f"Created {len(miner_batches)} batches of size {batch_size}")
        
        for batch_idx, batch in enumerate(miner_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(miner_batches)}")
            synapse = self.challenge_synthesizer.build_protocol()
            logger.debug("Built challenge protocol")
            uids, axons = zip(*batch)
            logger.debug(f"Processing UIDs in batch: {uids}")
            responses = await self.dendrite.forward(
                axons=axons, synapse=synapse, timeout=12
            )
            logger.info(f"Received {len(responses)} responses from miners")
            await self.score(uids, responses)
            logger.debug("Waiting 4 seconds before next batch")
            await asyncio.sleep(4)

    async def score(self, uids: list[int], responses: list[protocol.Synapse]):
        logger.info(f"Starting scoring for {len(uids)} miners")
        scores = []
        for uid, response in zip(uids, responses):
            logger.debug(f"Scoring miner {uid}")
            async with self.score_client.post(
                "/score", json={"uid": uid, "response": response}
            ) as resp:
                score: float = await resp.json()
                logger.debug(f"Miner {uid} received score: {score}")
                scores.append(score)

        logger.info(f"Updating miner manager with {len(scores)} scores")
        self.miner_manager.step(scores, uids)

